"""服务端域适应模块：CORAL + MMD + 域对抗训练

将原 server.py 中 server_representation_learning() 的核心逻辑
封装为独立模块，与 Flower GapsStrategy 解耦。

数据依赖：
  - val_loader: 源域验证集 (如传感器 1-2 数据)
  - calib_loader: 目标域校准集 (如传感器 3-5 数据的 10%)
  - semantic_protos: 来自 EMA 更新的全局语义原型

数学原理:
  CORAL: 最小化源域与目标域特征协方差矩阵的 Frobenius 范数差
  MMD: 用高斯核在 RKHS 中最小化两域分布的最大均值差异
  对抗: 训练域判别器，通过 GRL 使特征提取器产生域不变特征
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DomainDiscriminator, GradientReversalLayer
from utils import compute_mmd, compute_mmd2, deep_coral_loss, deep_coral_loss_class_conditional

logger = logging.getLogger('gaps.server')


def cross_domain_same_class_phase_mmd2(
    feat_s: torch.Tensor,
    y_s: torch.Tensor,
    phase_s: torch.Tensor,
    feat_t: torch.Tensor,
    y_t: torch.Tensor,
    phase_t: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    """Average MMD-squared across matched source/target class-phase cells."""
    y_s_ids = y_s.view(-1).long()
    y_t_ids = y_t.view(-1).long()
    phase_s_ids = phase_s.view(-1).long()
    phase_t_ids = phase_t.view(-1).long()
    terms = []
    for class_id in range(int(num_classes)):
        source_phases = torch.unique(phase_s_ids[y_s_ids == class_id])
        target_phases = torch.unique(phase_t_ids[y_t_ids == class_id])
        if source_phases.numel() == 0 or target_phases.numel() == 0:
            continue
        phases = sorted(
            set(int(value.item()) for value in source_phases)
            & set(int(value.item()) for value in target_phases)
        )
        for phase_id in phases:
            source_mask = (y_s_ids == class_id) & (phase_s_ids == phase_id)
            target_mask = (y_t_ids == class_id) & (phase_t_ids == phase_id)
            if int(source_mask.sum()) < 2 or int(target_mask.sum()) < 2:
                continue
            terms.append(compute_mmd2(feat_s[source_mask], feat_t[target_mask]))
    if terms:
        return torch.stack(terms).mean()
    return torch.zeros((), device=feat_s.device, dtype=feat_s.dtype)


def wasserstein_feature_objective(
    discriminator: nn.Module,
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
) -> torch.Tensor:
    """Return the critic gap for encoder minimization with a frozen critic."""
    parameters = list(discriminator.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        return discriminator(feat_s).mean() - discriminator(feat_t).mean()
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


class ServerDomainAdaptation:
    """服务端域适应训练器

    每轮联邦聚合后，使用全局模型在源域/目标域数据上执行若干步优化，
    对齐两域特征分布。修改全局模型权重的副本，产出 domain-adapted checkpoint。

    Attributes:
        model: 全局模型副本
        val_loader: 源域验证数据加载器
        calib_loader: 目标域校准数据加载器
        semantic_protos: EMA 全局语义原型 (nn.ParameterDict)
        device: 计算设备
        optimizer: 服务端优化器 (Adam)
        domain_discriminator: Wasserstein 域判别器 (可选)
        disc_optimizer: 域判别器优化器 (可选)
        grl: 梯度反转层 (可选)
        hyperparams: 超参名字典
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        calib_loader: Optional[DataLoader],
        semantic_protos: Dict[str, torch.Tensor],
        device: torch.device,
        hyperparams: Optional[dict] = None,
    ):
        self.model = copy.deepcopy(model).to(device)
        self.val_loader = val_loader
        self.calib_loader = calib_loader
        self.device = device
        self.hp = hyperparams or {}

        # 语义原型: str key → Parameter.  Unlike the earlier Flower DA
        # smoke version, prototypes can now participate in server-side
        # adaptation, matching the single-machine server optimization more
        # closely.
        self.semantic_protos: nn.ParameterDict = nn.ParameterDict()
        self._set_semantic_protos(semantic_protos)

        # ── 域判别器 (WGAN-GP + GRL) ──
        self.domain_discriminator: Optional[DomainDiscriminator] = None
        self.disc_optimizer: Optional[torch.optim.Optimizer] = None
        self.grl: Optional[GradientReversalLayer] = None
        if self.hp.get('USE_ADVERSARIAL_DOMAIN', False):
            feat_dim = self.hp.get('HIDDEN_DIM2', 64)
            self.domain_discriminator = DomainDiscriminator(
                feat_dim=feat_dim, hidden_dim=feat_dim // 2
            ).to(device)
            self.disc_optimizer = torch.optim.Adam(
                self.domain_discriminator.parameters(),
                lr=self.hp.get('ADV_DOMAIN_LR', 0.001),
            )
            self.grl = GradientReversalLayer(lambda_grl=1.0)

        # Per-round client statistics used by single-machine-style server losses.
        self.client_mus: List[Dict] = []
        self.client_counts: List[Dict] = []
        self.client_ids: List[int] = []
        self.client_weights: Optional[torch.Tensor] = None
        self.client_residual_targets: List[Optional[torch.Tensor]] = []
        self.device_residuals: nn.ParameterDict = nn.ParameterDict()

        # ── 优化器: 模型参数 + 可学习语义原型 + 可选设备残差 ──
        self.optimizer = self._build_optimizer()
        self.target_ce_class_weights = self._build_target_ce_class_weights()

        self.logger = logger

    def _method_definition(self) -> dict:
        return {
            "mmd_objective": str(
                self.hp.get('MMD_OBJECTIVE', 'legacy_quartic')
            ),
            "stage_alignment": str(
                self.hp.get('STAGE_ALIGNMENT', 'legacy_intra_domain')
            ),
            "adv_feature_objective": str(
                self.hp.get('ADV_FEATURE_OBJECTIVE', 'legacy_grl_plus')
            ),
            "proto_pair_l2_enabled": bool(
                self.hp.get('USE_PROTO_MMD', False)
                and float(self.hp.get('LAMBDA_PROTO_MMD', 0.0)) > 0.0
            ),
            "proto_pair_l2_trainable": False,
        }

    def _set_semantic_protos(self, semantic_protos: Dict[str, torch.Tensor]) -> None:
        self.semantic_protos = nn.ParameterDict()
        learn_protos = bool(self.hp.get('DA_LEARN_SEMANTIC_PROTOS', True))
        for key, tensor in semantic_protos.items():
            param = nn.Parameter(
                tensor.clone().detach().float().to(self.device),
                requires_grad=learn_protos,
            )
            self.semantic_protos[str(key)] = param

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_params = list(self.model.parameters())
        opt_params.extend(
            p for p in self.semantic_protos.parameters() if p.requires_grad
        )
        opt_params.extend(
            p for p in self.device_residuals.parameters() if p.requires_grad
        )
        return torch.optim.Adam(
            opt_params,
            lr=self.hp.get('SERVER_OPT_LR', 1e-4),
        )

    def get_semantic_protos(self) -> Dict[str, torch.Tensor]:
        """Return detached semantic prototypes after DA for GapsStrategy."""
        return {
            key: param.detach().cpu().clone()
            for key, param in self.semantic_protos.items()
        }

    def reset_round_state(
        self,
        model: nn.Module,
        semantic_protos: Dict[str, torch.Tensor],
    ) -> None:
        """Attach a new round model and rebuild optimizers bound to its params.

        Flower aggregation creates a fresh model for each server round. If we
        replace ``self.model`` without recreating the optimizer, Adam keeps
        references to the previous round's parameters and the adapted checkpoint
        remains identical to the plain checkpoint.
        """
        self.model = model.to(self.device)
        self._set_semantic_protos(semantic_protos)
        self.optimizer = self._build_optimizer()
        self.target_ce_class_weights = self._build_target_ce_class_weights()

    @staticmethod
    def _parse_proto_key(key) -> Optional[Tuple[int, int]]:
        """Parse tuple/string prototype keys into (class_id, phase_id)."""
        if isinstance(key, tuple) and len(key) >= 2:
            return int(key[0]), int(key[1])
        text = str(key).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        text = text.replace("_", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _proto_key_str(key) -> Optional[str]:
        parsed = ServerDomainAdaptation._parse_proto_key(key)
        if parsed is None:
            return None
        return f"({parsed[0]},{parsed[1]})"

    def _set_round_client_statistics(
        self,
        client_mus: Optional[List[Dict]],
        client_counts: Optional[List[Dict]],
        client_weights: Optional[torch.Tensor],
        client_ids: Optional[List[int]],
        client_residuals: Optional[List[Optional[torch.Tensor]]],
    ) -> None:
        """Cache client statistics uploaded through Flower metrics.

        These statistics are used to reproduce the main server-side losses from
        the single-machine Server.server_representation_learning path without
        giving the server access to raw client training data.
        """
        self.client_mus = client_mus or []
        self.client_counts = client_counts or [{} for _ in self.client_mus]
        self.client_ids = client_ids or list(range(len(self.client_mus)))
        if client_weights is None and self.client_mus:
            self.client_weights = torch.ones(len(self.client_mus), device=self.device) / max(1, len(self.client_mus))
        elif client_weights is not None:
            self.client_weights = client_weights.detach().float().to(self.device)
        else:
            self.client_weights = None
        self.client_residual_targets = client_residuals or [None for _ in self.client_ids]

        # Ensure one trainable residual vector per client that uploaded a target
        # residual.  This mirrors the single-machine device_residuals parameters.
        learn_residuals = bool(self.hp.get('USE_PROTO_DECOUPLING', True))
        for cid, residual in zip(self.client_ids, self.client_residual_targets):
            if residual is None:
                continue
            key = str(int(cid))
            vec = residual.detach().float().to(self.device).view(-1)
            if key not in self.device_residuals:
                self.device_residuals[key] = nn.Parameter(vec.clone(), requires_grad=learn_residuals)
            else:
                # Keep the parameter value from previous rounds, but if the
                # shape changes, safely reinitialize from the current upload.
                if self.device_residuals[key].numel() != vec.numel():
                    self.device_residuals[key] = nn.Parameter(vec.clone(), requires_grad=learn_residuals)

    def _semantic_proto(self, key) -> Optional[torch.Tensor]:
        str_key = self._proto_key_str(key)
        parsed = self._parse_proto_key(key)
        if str_key is None or parsed is None:
            return None
        if str_key in self.semantic_protos:
            return self.semantic_protos[str_key]
        compact_key = f"{parsed[0]},{parsed[1]}"
        if compact_key in self.semantic_protos:
            return self.semantic_protos[compact_key]
        underscore_key = f"{parsed[0]}_{parsed[1]}"
        if underscore_key in self.semantic_protos:
            return self.semantic_protos[underscore_key]
        return None

    def _full_proto_for_client(self, key, client_id: int) -> Optional[torch.Tensor]:
        """Return semantic prototype plus optional trainable device residual."""
        sem = self._semantic_proto(key)
        if sem is None:
            return None
        res = self.device_residuals.get(str(int(client_id)))
        return sem + res if res is not None else sem

    def _count_for_key(self, count_dict: Dict, key) -> float:
        if key in count_dict:
            return float(count_dict[key])
        str_key = self._proto_key_str(key)
        if str_key in count_dict:
            return float(count_dict[str_key])
        parsed = self._parse_proto_key(key)
        if parsed is not None:
            compact = f"{parsed[0]},{parsed[1]}"
            if compact in count_dict:
                return float(count_dict[compact])
        return 1.0

    def _compute_server_proto_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-machine-style prototype/residual losses from client uploads.

        Notes:
            We intentionally do not use a direct local-prototype-to-semantic
            alignment loss here. Uploaded local prototypes are fixed client
            statistics and may include device-specific offsets; directly fitting
            semantic prototypes to them would conflict with the semantic +
            device-residual decomposition. The prototype fitting term below uses
            semantic_proto + device_residual instead.
        """
        proto_loss_terms = []
        mmd_proto_loss = torch.tensor(0.0, device=self.device)
        residual_loss = torch.tensor(0.0, device=self.device)

        if not self.client_mus:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero

        weights = self.client_weights
        if weights is None or len(weights) != len(self.client_mus):
            weights = torch.ones(len(self.client_mus), device=self.device) / max(1, len(self.client_mus))

        # 1) proto_loss: client prototype should match semantic+device residual.
        for i, mu_dict in enumerate(self.client_mus):
            cnt_dict = self.client_counts[i] if i < len(self.client_counts) else {}
            cid = self.client_ids[i] if i < len(self.client_ids) else i
            w = weights[i] if i < len(weights) else torch.tensor(1.0 / max(1, len(self.client_mus)), device=self.device)
            for key, mu_i in mu_dict.items():
                full_proto = self._full_proto_for_client(key, cid)
                if full_proto is None:
                    continue
                n = torch.tensor(self._count_for_key(cnt_dict, key), device=self.device, dtype=torch.float32)
                local_mu = mu_i.detach().float().to(self.device)
                proto_loss_terms.append(w * n * torch.norm(full_proto - local_mu, p=2).pow(2))

        proto_loss = torch.stack(proto_loss_terms).mean() if proto_loss_terms else torch.tensor(0.0, device=self.device)

        # 2) residual_loss: trainable per-client residual follows uploaded estimate.
        res_terms = []
        if bool(self.hp.get('USE_PROTO_DECOUPLING', True)):
            for cid, residual in zip(self.client_ids, self.client_residual_targets):
                if residual is None:
                    continue
                key = str(int(cid))
                if key in self.device_residuals:
                    target = residual.detach().float().to(self.device).view(-1)
                    res_terms.append(torch.norm(self.device_residuals[key] - target, p=2).pow(2))
        if res_terms:
            residual_loss = torch.stack(res_terms).mean()

        # 3) mmd_proto_loss: consistency of different clients' local prototypes.
        if bool(self.hp.get('USE_PROTO_MMD', False)) and len(self.client_mus) >= 2:
            by_key = defaultdict(list)
            for mu_dict in self.client_mus:
                for key, mu in mu_dict.items():
                    str_key = self._proto_key_str(key)
                    if str_key is not None:
                        by_key[str_key].append(mu.detach().float().to(self.device))
            terms = []
            for _key, vectors in by_key.items():
                if len(vectors) < 2:
                    continue
                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        # A one-sample MMD is numerically weak; use L2 as a
                        # stable surrogate for prototype-level distribution gap.
                        terms.append(torch.norm(vectors[i] - vectors[j], p=2).pow(2))
            if terms:
                mmd_proto_loss = torch.stack(terms).mean()

        return proto_loss, mmd_proto_loss, residual_loss

    def _compute_align_reg_legacy_loss(self) -> torch.Tensor:
        """Legacy direct local-prototype-to-semantic alignment loss.

        This reproduces the older single-machine ``align_reg_loss`` term:
        uploaded local prototypes are treated as fixed targets and semantic
        prototypes are directly pulled toward them.  It is disabled by default
        because it may mix device-specific offsets into semantic prototypes; use
        it only for strong-baseline reproduction/diagnosis.
        """
        if not bool(self.hp.get('USE_ALIGN_REG_LEGACY', False)):
            return torch.tensor(0.0, device=self.device)
        if not self.client_mus or not self.semantic_protos:
            return torch.tensor(0.0, device=self.device)
        terms = []
        for mu_dict in self.client_mus:
            for key, local_mu in mu_dict.items():
                sem = self._semantic_proto(key)
                if sem is None:
                    continue
                terms.append(torch.norm(sem - local_mu.detach().float().to(self.device), p=2).pow(2))
        return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=self.device)

    def _compute_consistency_loss(
        self,
        feats: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        phases: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Feature-to-semantic-prototype consistency on source batches."""
        if feats is None or labels is None or phases is None or not self.semantic_protos:
            return torch.tensor(0.0, device=self.device)
        labels = self._as_class_ids(labels)
        phases = self._as_class_ids(phases)
        if bool(self.hp.get('USE_CONTRASTIVE_CONSISTENCY', True)):
            try:
                from utils import contrastive_loss_with_protos
                return contrastive_loss_with_protos(
                    feats,
                    labels,
                    phases,
                    self.semantic_protos,
                    temperature=float(self.hp.get('CONTRAST_TEMPERATURE', 0.1)),
                )
            except Exception:
                # Fall back to L2 if the utility expects a slightly different
                # prototype container in a deployment environment.
                pass
        terms = []
        for i in range(feats.size(0)):
            proto = self._semantic_proto((int(labels[i].item()), int(phases[i].item())))
            if proto is not None:
                terms.append(torch.norm(feats[i] - proto, p=2).pow(2))
        return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=self.device)

    def _compute_stage_mmd_loss(
        self,
        feat_s: Optional[torch.Tensor],
        y_s: Optional[torch.Tensor],
        phase_s: Optional[torch.Tensor],
        feat_t: Optional[torch.Tensor],
        y_t: Optional[torch.Tensor],
        phase_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stage-wise MMD inside each class, using source and target if available."""
        if not bool(self.hp.get('USE_MMD_ALIGNMENT', True)):
            return torch.tensor(0.0, device=self.device)
        mode = str(self.hp.get('STAGE_ALIGNMENT', 'legacy_intra_domain'))
        if mode == 'cross_domain_same_class_phase':
            if any(
                value is None
                for value in (feat_s, y_s, phase_s, feat_t, y_t, phase_t)
            ):
                return torch.tensor(0.0, device=self.device)
            return cross_domain_same_class_phase_mmd2(
                feat_s,
                self._as_class_ids(y_s),
                self._as_class_ids(phase_s),
                feat_t,
                self._as_class_ids(y_t),
                self._as_class_ids(phase_t),
                num_classes=int(self.hp.get('NUM_CLASSES', 4)),
            )
        if mode != 'legacy_intra_domain':
            raise ValueError(f"unsupported STAGE_ALIGNMENT: {mode}")
        terms = []
        num_classes = int(self.hp.get('NUM_CLASSES', 4))
        for feats, labels, phases in ((feat_s, y_s, phase_s), (feat_t, y_t, phase_t)):
            if feats is None or labels is None or phases is None:
                continue
            labels = self._as_class_ids(labels)
            phases = self._as_class_ids(phases)
            for c in range(num_classes):
                class_mask = labels == c
                if class_mask.sum() < 2:
                    continue
                class_feats = feats[class_mask]
                class_phases = phases[class_mask]
                unique_phases = torch.unique(class_phases)
                if unique_phases.numel() < 2:
                    continue
                phase_values = [int(v.item()) for v in unique_phases]
                for i in range(len(phase_values)):
                    for j in range(i + 1, len(phase_values)):
                        m_i = class_phases == phase_values[i]
                        m_j = class_phases == phase_values[j]
                        if m_i.sum() > 1 and m_j.sum() > 1:
                            terms.append(compute_mmd(class_feats[m_i], class_feats[m_j]) ** 2)
        return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=self.device)

    def run_adaptation(
        self,
        num_steps: int = 30,
        client_mus: Optional[List[Dict]] = None,
        client_counts: Optional[List[Dict]] = None,
        client_weights: Optional[torch.Tensor] = None,
        client_ids: Optional[List[int]] = None,
        client_residuals: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[nn.Module, dict]:
        """执行服务端域适应优化

        Args:
            num_steps: 优化步数 K
            client_mus: 客户端本地原型 (可选，用于原型对齐损失)
            client_counts: 客户端原型计数
            client_weights: 客户端聚合权重

        Returns:
            (adapted_model, diagnostics_dict)
        """
        self._set_round_client_statistics(
            client_mus=client_mus,
            client_counts=client_counts,
            client_weights=client_weights,
            client_ids=client_ids,
            client_residuals=client_residuals,
        )
        # Device residual parameters may be created from this round's uploads, so
        # rebuild the optimizer before the adaptation steps.
        self.optimizer = self._build_optimizer()

        self.model.train()
        if self.domain_discriminator is not None:
            self.domain_discriminator.train()

        source_iter = iter(self.val_loader)
        source_align_iter = iter(self.val_loader)
        target_iter = iter(self.calib_loader) if self.calib_loader is not None else None

        diagnostics: Dict[str, List[float]] = {
            "val_loss": [],
            "coral_loss": [],
            "mmd_global": [],
            "mmd_class": [],
            "adv_loss": [],
            "proto_anchor": [],
            "proto_loss": [],
            "consist_loss": [],
            "residual_loss": [],
            "mmd_proto_loss": [],
            "stage_mmd_loss": [],
            "align_reg_legacy_loss": [],
            "target_ce_loss": [],
            "target_ce_acc": [],
            "weighted_coral_loss": [],
            "weighted_mmd_global": [],
            "weighted_mmd_class": [],
            "weighted_adv_loss": [],
            "weighted_proto_anchor": [],
            "weighted_proto_loss": [],
            "weighted_consist_loss": [],
            "weighted_residual_loss": [],
            "weighted_mmd_proto_loss": [],
            "weighted_stage_mmd_loss": [],
            "weighted_align_reg_legacy_loss": [],
            "weighted_target_ce_loss": [],
            "total_loss": [],
        }

        for step in range(num_steps):
            self.optimizer.zero_grad()

            # ── 1. 源域分类验证损失 ──
            val_loss = torch.tensor(0.0, device=self.device)
            val_batches = 0
            max_val_batches = self.hp.get('MAX_VAL_BATCHES', 10)
            for _ in range(max_val_batches):
                try:
                    batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.val_loader)
                    batch = next(source_iter)
                x = batch[0].to(self.device)
                y_cls = batch[1].to(self.device).long()
                logits, _, _ = self.model(x)
                val_loss += F.cross_entropy(logits, y_cls)
                val_batches += 1
            if val_batches > 0:
                val_loss = val_loss / val_batches

            # ── 2. 获取源域/目标域特征 ──
            _logits_s, cls_feat_s, y_s, phase_s, source_align_iter = self._sample_model_outputs(
                source_align_iter, self.val_loader, reset_on_exhaust=True
            )
            logits_t, cls_feat_t, y_t, phase_t = None, None, None, None
            if target_iter is not None:
                logits_t, cls_feat_t, y_t, phase_t, target_iter = self._sample_model_outputs(
                    target_iter, self.calib_loader, reset_on_exhaust=True
                )

            # ── 3. CORAL 协方差对齐损失 ──
            coral_loss = torch.tensor(0.0, device=self.device)
            if (self.hp.get('USE_DEEP_CORAL', False) and
                    cls_feat_s is not None and cls_feat_t is not None):
                if (self.hp.get('CORAL_CLASS_CONDITIONAL', False) and
                        y_s is not None and y_t is not None):
                    coral_loss = deep_coral_loss_class_conditional(
                        cls_feat_s, y_s, cls_feat_t, y_t,
                        num_classes=self.hp.get('NUM_CLASSES', 4),
                    )
                else:
                    coral_loss = deep_coral_loss(cls_feat_s, cls_feat_t)

            # ── 4. MMD 对齐损失 ──
            mmd_global, mmd_class, proto_anchor = (
                self._compute_mmd_losses(cls_feat_s, y_s, cls_feat_t, y_t)
            )

            # ── 5. 域对抗训练 ──
            adv_loss = torch.tensor(0.0, device=self.device)
            if (self.domain_discriminator is not None and
                    cls_feat_s is not None and cls_feat_t is not None):
                adv_loss = self._compute_adversarial_loss(cls_feat_s, y_s, cls_feat_t, y_t)

            target_ce_loss = torch.tensor(0.0, device=self.device)
            target_ce_acc = torch.tensor(0.0, device=self.device)
            if (self.hp.get('LAMBDA_TARGET_CE', 0.0) > 0.0 and
                    logits_t is not None and y_t is not None):
                y_t_ids = self._as_class_ids(y_t)
                target_ce_loss = F.cross_entropy(
                    logits_t,
                    y_t_ids,
                    weight=self.target_ce_class_weights,
                    label_smoothing=self.hp.get('TARGET_CE_LABEL_SMOOTHING', 0.0),
                )
                target_ce_acc = (logits_t.argmax(dim=1) == y_t_ids).float().mean()

            # ── 6. 单机 server_representation_learning 对齐项 ──
            proto_loss, mmd_proto_loss, residual_loss = self._compute_server_proto_losses()
            consist_loss = self._compute_consistency_loss(cls_feat_s, y_s, phase_s)
            stage_mmd_loss = self._compute_stage_mmd_loss(
                cls_feat_s, y_s, phase_s, cls_feat_t, y_t, phase_t
            )
            align_reg_legacy_loss = self._compute_align_reg_legacy_loss()

            # ── 7. 损失组合 ──
            lambda_coral = self.hp.get('LAMBDA_DEEP_CORAL', 0.1)
            lambda_global_mmd = self.hp.get('LAMBDA_GLOBAL_MMD', 0.5)
            lambda_class_mmd = self.hp.get('LAMBDA_CLASS_MMD', 0.5)
            lambda_proto = self.hp.get('LAMBDA_PROTO_ANCHOR', 0.3)
            lambda_adv = self.hp.get('LAMBDA_ADV_DOMAIN', 0.1)
            lambda_target_ce = self.hp.get('LAMBDA_TARGET_CE', 0.0)
            lambda_proto_loss = self.hp.get('LAMBDA_PROTO', 0.05)
            lambda_consistency = self.hp.get('LAMBDA_CONSISTENCY', 2.0)
            lambda_residual = self.hp.get('LAMBDA_RES', 0.1)
            lambda_proto_mmd = self.hp.get('LAMBDA_PROTO_MMD', 0.2)
            lambda_stage_mmd = self.hp.get('LAMBDA_STAGE_MMD', 0.2)
            lambda_align_reg_legacy = self.hp.get('LAMBDA_ALIGN_REG_LEGACY', 0.05)
            weighted_coral = lambda_coral * coral_loss
            weighted_global_mmd = lambda_global_mmd * mmd_global
            weighted_class_mmd = lambda_class_mmd * mmd_class
            weighted_proto = lambda_proto * proto_anchor
            weighted_adv = lambda_adv * adv_loss
            weighted_target_ce = lambda_target_ce * target_ce_loss
            weighted_proto_loss = lambda_proto_loss * proto_loss
            weighted_consist = lambda_consistency * consist_loss
            weighted_residual = lambda_residual * residual_loss
            weighted_mmd_proto = lambda_proto_mmd * mmd_proto_loss
            weighted_stage_mmd = lambda_stage_mmd * stage_mmd_loss
            weighted_align_reg_legacy = lambda_align_reg_legacy * align_reg_legacy_loss
            total_loss = (
                val_loss
                + weighted_coral
                + weighted_global_mmd
                + weighted_class_mmd
                + weighted_proto
                + weighted_adv
                + weighted_target_ce
                + weighted_proto_loss
                + weighted_consist
                + weighted_residual
                + weighted_mmd_proto
                + weighted_stage_mmd
                + weighted_align_reg_legacy
            )

            total_loss.backward()
            trainable_params = []
            for group in self.optimizer.param_groups:
                trainable_params.extend(group.get("params", []))
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            self.optimizer.step()

            # 记录诊断
            diagnostics["val_loss"].append(float(val_loss.item()))
            diagnostics["coral_loss"].append(float(coral_loss.item()))
            diagnostics["mmd_global"].append(float(mmd_global.item()))
            diagnostics["mmd_class"].append(float(mmd_class.item()))
            diagnostics["adv_loss"].append(float(adv_loss.item()))
            diagnostics["proto_anchor"].append(float(proto_anchor.item()))
            diagnostics["proto_loss"].append(float(proto_loss.item()))
            diagnostics["consist_loss"].append(float(consist_loss.item()))
            diagnostics["residual_loss"].append(float(residual_loss.item()))
            diagnostics["mmd_proto_loss"].append(float(mmd_proto_loss.item()))
            diagnostics["stage_mmd_loss"].append(float(stage_mmd_loss.item()))
            diagnostics["align_reg_legacy_loss"].append(float(align_reg_legacy_loss.item()))
            diagnostics["target_ce_loss"].append(float(target_ce_loss.item()))
            diagnostics["target_ce_acc"].append(float(target_ce_acc.item()))
            diagnostics["weighted_coral_loss"].append(float(weighted_coral.item()))
            diagnostics["weighted_mmd_global"].append(float(weighted_global_mmd.item()))
            diagnostics["weighted_mmd_class"].append(float(weighted_class_mmd.item()))
            diagnostics["weighted_adv_loss"].append(float(weighted_adv.item()))
            diagnostics["weighted_proto_anchor"].append(float(weighted_proto.item()))
            diagnostics["weighted_proto_loss"].append(float(weighted_proto_loss.item()))
            diagnostics["weighted_consist_loss"].append(float(weighted_consist.item()))
            diagnostics["weighted_residual_loss"].append(float(weighted_residual.item()))
            diagnostics["weighted_mmd_proto_loss"].append(float(weighted_mmd_proto.item()))
            diagnostics["weighted_stage_mmd_loss"].append(float(weighted_stage_mmd.item()))
            diagnostics["weighted_align_reg_legacy_loss"].append(float(weighted_align_reg_legacy.item()))
            diagnostics["weighted_target_ce_loss"].append(float(weighted_target_ce.item()))
            diagnostics["total_loss"].append(float(total_loss.item()))

        # 计算摘要
        summary = {
            key: float(torch.tensor(values).mean().item()) if values else 0.0
            for key, values in diagnostics.items()
        }
        summary.update(self._method_definition())
        summary["num_steps"] = num_steps
        summary["lambda_coral"] = float(self.hp.get('LAMBDA_DEEP_CORAL', 0.1))
        summary["lambda_global_mmd"] = float(self.hp.get('LAMBDA_GLOBAL_MMD', 0.5))
        summary["lambda_class_mmd"] = float(self.hp.get('LAMBDA_CLASS_MMD', 0.5))
        summary["lambda_proto_anchor"] = float(self.hp.get('LAMBDA_PROTO_ANCHOR', 0.3))
        summary["lambda_adv"] = float(self.hp.get('LAMBDA_ADV_DOMAIN', 0.1))
        summary["lambda_target_ce"] = float(self.hp.get('LAMBDA_TARGET_CE', 0.0))
        summary["lambda_proto"] = float(self.hp.get('LAMBDA_PROTO', 0.05))
        summary["lambda_consistency"] = float(self.hp.get('LAMBDA_CONSISTENCY', 2.0))
        summary["lambda_residual"] = float(self.hp.get('LAMBDA_RES', 0.1))
        summary["lambda_proto_mmd"] = float(self.hp.get('LAMBDA_PROTO_MMD', 0.2))
        summary["lambda_stage_mmd"] = float(self.hp.get('LAMBDA_STAGE_MMD', 0.2))
        summary["target_ce_label_smoothing"] = float(
            self.hp.get('TARGET_CE_LABEL_SMOOTHING', 0.0)
        )
        summary["target_ce_class_balanced"] = bool(
            self.hp.get('TARGET_CE_CLASS_BALANCED', False)
        )
        summary["target_ce_class_weights"] = (
            [float(v) for v in self.target_ce_class_weights.detach().cpu().tolist()]
            if self.target_ce_class_weights is not None
            else None
        )
        summary["semantic_proto_count"] = int(len(self.semantic_protos))
        summary["semantic_proto_trainable"] = int(
            sum(1 for p in self.semantic_protos.parameters() if p.requires_grad)
        )
        summary["device_residual_count"] = int(len(self.device_residuals))
        summary["client_proto_groups"] = int(sum(1 for m in self.client_mus if m))

        self.logger.info(
            f"[DA] Adaptation done: val_loss={summary['val_loss']:.4f}, "
            f"coral={summary['coral_loss']:.4f}, "
            f"mmd_g={summary['mmd_global']:.4f}, "
            f"mmd_c={summary['mmd_class']:.4f}, "
            f"adv={summary['adv_loss']:.4f}"
        )
        return self.model, summary

    def _sample_model_outputs(
        self,
        data_iter,
        data_loader: DataLoader,
        reset_on_exhaust: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        object,
    ]:
        """从数据迭代器采样一个批次，提取分类特征

        当迭代器耗尽时，若 reset_on_exhaust=True 则从 data_loader 重建迭代器继续采样。

        Args:
            data_iter: 当前数据迭代器
            data_loader: 原始 DataLoader，用于在耗尽时重建迭代器
            reset_on_exhaust: 耗尽时是否重新创建迭代器

        Returns:
            (特征张量 (B,D), 标签张量 (B,), 新的迭代器)
            若无法获取数据则返回 (None, None, data_iter)
        """
        try:
            batch = next(data_iter)
        except StopIteration:
            if not reset_on_exhaust:
                return None, None, None, None, data_iter
            data_iter = iter(data_loader)
            try:
                batch = next(data_iter)
            except StopIteration:
                return None, None, None, None, data_iter

        x = batch[0].to(self.device)
        # GasSensorWindowDataset returns (x, y_cls, y_reg, phase).
        y_cls = batch[1].to(self.device).long() if len(batch) >= 2 else None
        y_phase = batch[3].to(self.device).long() if len(batch) >= 4 else None
        logits, feat, _ = self.model(x)
        return logits, feat, y_cls, y_phase, data_iter

    def _build_target_ce_class_weights(self) -> Optional[torch.Tensor]:
        """Estimate inverse-frequency class weights from target calibration data."""
        if not self.hp.get('TARGET_CE_CLASS_BALANCED', False):
            return None
        if self.calib_loader is None:
            return None

        num_classes = int(self.hp.get('NUM_CLASSES', 4))
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for batch in self.calib_loader:
            if len(batch) < 2:
                continue
            labels = self._as_class_ids(batch[1])
            labels = labels.detach().cpu()
            counts += torch.bincount(labels, minlength=num_classes).float()

        present = counts > 0
        if int(present.sum().item()) == 0:
            return None

        weights = torch.zeros_like(counts)
        weights[present] = (
            counts[present].sum()
            / (float(present.sum().item()) * counts[present])
        )
        return weights.to(self.device)

    def _sample_features(
        self,
        data_iter,
        data_loader: DataLoader,
        reset_on_exhaust: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], object]:
        """Backward-compatible wrapper returning only features and labels."""
        _logits, feat, y_cls, _phase, data_iter = self._sample_model_outputs(
            data_iter,
            data_loader,
            reset_on_exhaust=reset_on_exhaust,
        )
        return feat, y_cls, data_iter

    @staticmethod
    def _as_class_ids(labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Convert labels to integer class ids with shape (B,)."""
        if labels is None:
            return None
        if labels.ndim > 1:
            return torch.argmax(labels, dim=1).long()
        return labels.long()

    def _semantic_class_proto(self, class_id: int) -> Optional[torch.Tensor]:
        """Average semantic prototypes over phases for one class."""
        protos = []
        for key, tensor in self.semantic_protos.items():
            try:
                text = str(key).strip()
                if text.startswith("(") and text.endswith(")"):
                    text = text[1:-1]
                text = text.replace("_", ",")
                cls_text, _phase_text = text.split(",", maxsplit=1)
                cls_id = int(cls_text.strip())
            except ValueError:
                continue
            if cls_id == int(class_id):
                protos.append(tensor)
        if not protos:
            return None
        return torch.stack(protos, dim=0).mean(dim=0)

    def _distribution_mmd(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(self.hp.get('MMD_OBJECTIVE', 'legacy_quartic'))
        if mode == 'legacy_quartic':
            return compute_mmd(features1, features2) ** 2
        if mode == 'mmd2':
            return compute_mmd2(features1, features2)
        raise ValueError(f"unsupported MMD_OBJECTIVE: {mode}")

    def _compute_mmd_losses(
        self,
        feat_s: Optional[torch.Tensor],
        y_s: Optional[torch.Tensor],
        feat_t: Optional[torch.Tensor],
        y_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 MMD + 原型锚定损失

        1. 全局 MMD: MMD²(feat_s, feat_t)
        2. 类条件 MMD: 每类内部 MMD²(feat_s[c], feat_t[c])
        3. 原型锚定: 源域/目标域类中心与语义原型的 L2 距离

        Returns:
            (global_mmd, class_mmd, proto_anchor)
        """
        g_mmd = torch.tensor(0.0, device=self.device)
        c_mmd = torch.tensor(0.0, device=self.device)
        anchor = torch.tensor(0.0, device=self.device)

        if feat_s is None or feat_t is None:
            return g_mmd, c_mmd, anchor

        use_mmd = bool(self.hp.get('USE_MMD_ALIGNMENT', True))
        use_proto_anchor = (
            float(self.hp.get('LAMBDA_PROTO_ANCHOR', 0.0)) > 0.0
            and len(self.semantic_protos) > 0
        )

        # 全局 MMD must obey --da-use-mmd / USE_MMD_ALIGNMENT.
        if use_mmd:
            g_mmd = self._distribution_mmd(feat_s, feat_t)

        num_classes = self.hp.get('NUM_CLASSES', 4)
        use_class_labels = y_s is not None and y_t is not None

        if use_class_labels and (use_mmd or use_proto_anchor):
            y_s_ids = self._as_class_ids(y_s)
            y_t_ids = self._as_class_ids(y_t)
            class_count = 0
            anchor_count = 0
            for c in range(num_classes):
                src_mask = (y_s_ids == c)
                tgt_mask = (y_t_ids == c)
                if src_mask.sum() > 1 and tgt_mask.sum() > 1:
                    if use_mmd:
                        c_mmd += self._distribution_mmd(
                            feat_s[src_mask], feat_t[tgt_mask]
                        )
                    class_count += 1
                    if use_proto_anchor:
                        mu_sem = self._semantic_class_proto(c)
                        if mu_sem is not None:
                            mu_src = feat_s[src_mask].mean(dim=0)
                            mu_tgt = feat_t[tgt_mask].mean(dim=0)
                            anchor += torch.norm(mu_src - mu_sem, p=2).pow(2)
                            anchor += torch.norm(mu_tgt - mu_sem, p=2).pow(2)
                            anchor_count += 1
            if class_count > 0 and use_mmd:
                c_mmd = c_mmd / class_count
            if anchor_count > 0:
                anchor = anchor / anchor_count

        return g_mmd, c_mmd, anchor

    def _compute_adversarial_loss(
        self,
        feat_s: torch.Tensor,
        y_s: Optional[torch.Tensor],
        feat_t: torch.Tensor,
        y_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算域对抗损失 (WGAN-GP + GRL)

        n_critic 步判别器更新 → 1 步 GRL 特征更新

        Args:
            feat_s: 源域特征 (B_s, D)
            y_s: 源域标签
            feat_t: 目标域特征 (B_t, D)
            y_t: 目标域标签

        Returns:
            对抗损失 (用于特征提取器优化)
        """
        n_critic = self.hp.get('ADV_CRITIC_ITERS', 3)
        gp_lambda = self.hp.get('ADV_GRADIENT_PENALTY', 10.0)
        class_cond = self.hp.get('ADV_CLASS_CONDITIONAL', True)
        y_s_ids = self._as_class_ids(y_s)
        y_t_ids = self._as_class_ids(y_t)

        # n_critic 步判别器更新
        for _ in range(n_critic):
            self.disc_optimizer.zero_grad()

            if class_cond and y_s_ids is not None and y_t_ids is not None:
                disc_loss = torch.tensor(0.0, device=self.device)
                gp_loss = torch.tensor(0.0, device=self.device)
                valid = 0
                num_cls = self.hp.get('NUM_CLASSES', 4)
                for c in range(num_cls):
                    src_m = (y_s_ids == c)
                    tgt_m = (y_t_ids == c)
                    if src_m.sum() > 0 and tgt_m.sum() > 0:
                        d_s = self.domain_discriminator(feat_s[src_m].detach())
                        d_t = self.domain_discriminator(feat_t[tgt_m].detach())
                        disc_loss -= (d_s.mean() - d_t.mean())
                        valid += 1
                        if gp_lambda > 0:
                            gp_loss += self._gradient_penalty(
                                feat_s[src_m].detach(), feat_t[tgt_m].detach()
                            )
                if valid > 0:
                    disc_loss = disc_loss / valid
                    if gp_lambda > 0:
                        disc_loss += gp_lambda * gp_loss / valid
            else:
                d_s = self.domain_discriminator(feat_s.detach())
                d_t = self.domain_discriminator(feat_t.detach())
                disc_loss = -(d_s.mean() - d_t.mean())
                if gp_lambda > 0:
                    gp = self._gradient_penalty(feat_s.detach(), feat_t.detach())
                    disc_loss += gp_lambda * gp

            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.domain_discriminator.parameters(), max_norm=1.0
            )
            self.disc_optimizer.step()

        feature_objective = str(
            self.hp.get('ADV_FEATURE_OBJECTIVE', 'legacy_grl_plus')
        )
        if feature_objective == 'wasserstein_min':
            if class_cond and y_s_ids is not None and y_t_ids is not None:
                terms = []
                for c in range(int(self.hp.get('NUM_CLASSES', 4))):
                    src_m = y_s_ids == c
                    tgt_m = y_t_ids == c
                    if src_m.sum() > 0 and tgt_m.sum() > 0:
                        terms.append(
                            wasserstein_feature_objective(
                                self.domain_discriminator,
                                feat_s[src_m],
                                feat_t[tgt_m],
                            )
                        )
                if terms:
                    return torch.stack(terms).mean()
                return torch.tensor(0.0, device=self.device)
            return wasserstein_feature_objective(
                self.domain_discriminator, feat_s, feat_t
            )
        if feature_objective != 'legacy_grl_plus':
            raise ValueError(
                f"unsupported ADV_FEATURE_OBJECTIVE: {feature_objective}"
            )

        # Legacy v2 path: GRL updates the feature extractor.
        if class_cond and y_s_ids is not None and y_t_ids is not None:
            adv = torch.tensor(0.0, device=self.device)
            valid = 0
            num_cls = self.hp.get('NUM_CLASSES', 4)
            for c in range(num_cls):
                src_m = (y_s_ids == c)
                tgt_m = (y_t_ids == c)
                if src_m.sum() > 0 and tgt_m.sum() > 0:
                    f_s_grl = self.grl(feat_s[src_m])
                    f_t_grl = self.grl(feat_t[tgt_m])
                    w_dist = (self.domain_discriminator(f_s_grl).mean() -
                              self.domain_discriminator(f_t_grl).mean())
                    adv += w_dist
                    valid += 1
            if valid > 0:
                adv = adv / valid
        else:
            f_s_grl = self.grl(feat_s)
            f_t_grl = self.grl(feat_t)
            w_dist = (self.domain_discriminator(f_s_grl).mean() -
                      self.domain_discriminator(f_t_grl).mean())
            adv = w_dist

        return adv

    def _gradient_penalty(
        self, feat_src: torch.Tensor, feat_tgt: torch.Tensor
    ) -> torch.Tensor:
        """WGAN-GP 梯度惩罚

        强制判别器在 interpolates 点上的梯度范数接近 1，
        保证 1-Lipschitz 连续性。

        Args:
            feat_src: 源域特征 (N_s, D)
            feat_tgt: 目标域特征 (N_t, D)

        Returns:
            梯度惩罚标量
        """
        min_sz = min(feat_src.size(0), feat_tgt.size(0))
        if min_sz < 2:
            return torch.tensor(0.0, device=self.device)

        f_s = feat_src[:min_sz]
        f_t = feat_tgt[:min_sz]
        alpha = torch.rand(min_sz, 1, device=self.device)
        interpolates = (alpha * f_s + (1 - alpha) * f_t).detach().requires_grad_(True)

        disc_out = self.domain_discriminator(interpolates)
        grads = torch.autograd.grad(
            outputs=disc_out,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = grads.view(min_sz, -1).norm(2, dim=1)
        return ((grad_norm - 1.0) ** 2).mean()

    def get_model_state_dict(self) -> dict:
        """获取域适应后模型的 state_dict"""
        return self.model.state_dict()

    def save_diagnostics(self, path: Path, round_idx: int) -> None:
        """保存域适应诊断到 JSON 文件 (由外部调用)"""
        pass
