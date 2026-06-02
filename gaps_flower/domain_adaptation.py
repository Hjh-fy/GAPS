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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DomainDiscriminator, GradientReversalLayer
from utils import compute_mmd, deep_coral_loss, deep_coral_loss_class_conditional

logger = logging.getLogger('gaps.server')


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

        # 语义原型: str key → Tensor
        self.semantic_protos: Dict[str, torch.Tensor] = {}
        for key, tensor in semantic_protos.items():
            self.semantic_protos[key] = tensor.clone().detach().to(device)

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

        # ── 优化器 (模型参数) ──
        opt_params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(
            opt_params,
            lr=self.hp.get('SERVER_OPT_LR', 1e-4),
        )

        self.logger = logger

    def run_adaptation(
        self,
        num_steps: int = 30,
        client_mus: Optional[List[Dict]] = None,
        client_counts: Optional[List[Dict]] = None,
        client_weights: Optional[torch.Tensor] = None,
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
        self.model.train()
        if self.domain_discriminator is not None:
            self.domain_discriminator.train()

        source_iter = iter(self.val_loader)
        target_iter = iter(self.calib_loader) if self.calib_loader is not None else None

        diagnostics: Dict[str, List[float]] = {
            "val_loss": [],
            "coral_loss": [],
            "mmd_global": [],
            "mmd_class": [],
            "adv_loss": [],
            "proto_anchor": [],
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
                x, y_cls = batch[0].to(self.device), batch[1].to(self.device)
                logits, _, _ = self.model(x)
                val_loss += F.cross_entropy(logits, y_cls)
                val_batches += 1
            if val_batches > 0:
                val_loss = val_loss / val_batches

            # ── 2. 获取源域/目标域特征 ──
            cls_feat_s, y_s = self._sample_features(source_iter)
            cls_feat_t, y_t = None, None
            if target_iter is not None:
                cls_feat_t, y_t = self._sample_features(target_iter, reset_on_exhaust=True)

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

            # ── 6. 损失组合 ──
            total_loss = val_loss
            total_loss += self.hp.get('LAMBDA_DEEP_CORAL', 0.1) * coral_loss
            total_loss += self.hp.get('LAMBDA_GLOBAL_MMD', 0.5) * mmd_global
            total_loss += self.hp.get('LAMBDA_CLASS_MMD', 0.5) * mmd_class
            total_loss += self.hp.get('LAMBDA_PROTO_ANCHOR', 0.3) * proto_anchor
            total_loss += self.hp.get('LAMBDA_ADV_DOMAIN', 0.1) * adv_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # 记录诊断
            diagnostics["val_loss"].append(float(val_loss.item()))
            diagnostics["coral_loss"].append(float(coral_loss.item()))
            diagnostics["mmd_global"].append(float(mmd_global.item()))
            diagnostics["mmd_class"].append(float(mmd_class.item()))
            diagnostics["adv_loss"].append(float(adv_loss.item()))
            diagnostics["proto_anchor"].append(float(proto_anchor.item()))
            diagnostics["total_loss"].append(float(total_loss.item()))

        # 计算摘要
        summary = {
            key: float(torch.tensor(values).mean().item()) if values else 0.0
            for key, values in diagnostics.items()
        }
        summary["num_steps"] = num_steps

        self.logger.info(
            f"[DA] Adaptation done: val_loss={summary['val_loss']:.4f}, "
            f"coral={summary['coral_loss']:.4f}, "
            f"mmd_g={summary['mmd_global']:.4f}, "
            f"mmd_c={summary['mmd_class']:.4f}, "
            f"adv={summary['adv_loss']:.4f}"
        )
        return self.model, summary

    def _sample_features(
        self,
        data_iter,
        reset_on_exhaust: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从数据迭代器采样一个批次，提取分类特征

        Args:
            data_iter: 数据迭代器
            reset_on_exhaust: 耗尽时是否重新创建迭代器

        Returns:
            (特征张量 (B,D), 标签张量 (B,)) 或 (None, None)
        """
        try:
            batch = next(data_iter)
        except StopIteration:
            if not reset_on_exhaust:
                return None, None
            # 无法自动重置外部迭代器，返回 None
            return None, None

        x = batch[0].to(self.device)
        y_cls = batch[1].to(self.device) if len(batch) >= 2 else None
        _, feat, _ = self.model(x)
        return feat, y_cls

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

        # 全局 MMD
        g_mmd = compute_mmd(feat_s, feat_t) ** 2

        num_classes = self.hp.get('NUM_CLASSES', 4)
        use_class_cond = (self.hp.get('USE_MMD_ALIGNMENT', True) and
                           y_s is not None and y_t is not None)

        if use_class_cond:
            class_count = 0
            for c in range(num_classes):
                src_mask = (y_s == c)
                tgt_mask = (y_t == c)
                if src_mask.sum() > 1 and tgt_mask.sum() > 1:
                    c_mmd += compute_mmd(feat_s[src_mask], feat_t[tgt_mask]) ** 2
                    class_count += 1
                    # 原型锚定
                    proto_key = f"({c},0)"
                    if proto_key in self.semantic_protos:
                        mu_sem = self.semantic_protos[proto_key]
                        mu_src = feat_s[src_mask].mean(dim=0)
                        mu_tgt = feat_t[tgt_mask].mean(dim=0)
                        anchor += torch.norm(mu_src - mu_sem, p=2).pow(2)
                        anchor += torch.norm(mu_tgt - mu_sem, p=2).pow(2)
            if class_count > 0:
                c_mmd = c_mmd / num_classes
                anchor = anchor / class_count

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

        # n_critic 步判别器更新
        for _ in range(n_critic):
            self.disc_optimizer.zero_grad()

            if class_cond and y_s is not None and y_t is not None:
                disc_loss = torch.tensor(0.0, device=self.device)
                gp_loss = torch.tensor(0.0, device=self.device)
                valid = 0
                num_cls = self.hp.get('NUM_CLASSES', 4)
                for c in range(num_cls):
                    src_m = (y_s == c)
                    tgt_m = (y_t == c)
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

        # GRL: 特征提取器最小化域距离
        lambda_adv = self.hp.get('LAMBDA_ADV_DOMAIN', 0.1)
        if class_cond and y_s is not None and y_t is not None:
            adv = torch.tensor(0.0, device=self.device)
            valid = 0
            num_cls = self.hp.get('NUM_CLASSES', 4)
            for c in range(num_cls):
                src_m = (y_s == c)
                tgt_m = (y_t == c)
                if src_m.sum() > 0 and tgt_m.sum() > 0:
                    f_s_grl = self.grl(feat_s[src_m])
                    f_t_grl = self.grl(feat_t[tgt_m])
                    w_dist = (self.domain_discriminator(f_s_grl).mean() -
                              self.domain_discriminator(f_t_grl).mean())
                    adv += w_dist
                    valid += 1
            if valid > 0:
                adv = lambda_adv * adv / valid
        else:
            f_s_grl = self.grl(feat_s)
            f_t_grl = self.grl(feat_t)
            w_dist = (self.domain_discriminator(f_s_grl).mean() -
                      self.domain_discriminator(f_t_grl).mean())
            adv = lambda_adv * w_dist

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
