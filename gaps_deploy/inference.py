"""
统一推理入口模块

提供部署推理的统一接口 DeployPredictor，将分类 → 回归 → 校准 → QC
整个链路封装为单个 predict() 调用。

推理流程:
    1. 特征标准化 (可选)
    2. 分类模型前向: logits, cls_feat, reg_feat
    3. 路由选择: hard/soft/response_anchored
    4. 回归模型前向: pred_ppm (原始)
    5. 校准映射: apply affine/bias/phase_affine
    6. 响应锚定修正: (可选)
    7. 风险分数计算
    8. QC 双阈值决策: accept/review/reject

输出格式 (DeployResult):
    - pred_gas: 预测气体名称
    - pred_class: 预测类别 ID
    - pred_ppm: 模型原始预测浓度
    - calibrated_ppm: 校准后浓度
    - qc_status: accept / review / reject
    - risk_score: 风险分数
    - risk_reasons: 风险原因列表
    - model_version: 模型版本号
    - client_id: 客户端 ID
    - confidence: 分类置信度
    - route_info: 路由信息

使用示例:
    >>> from gaps_deploy import DeployPredictor
    >>> predictor = DeployPredictor.from_package("deployment_package")
    >>> results = predictor.predict_batch(features, client_id="C3")
    >>> for r in results:
    ...     print(f"{r.pred_gas}: {r.calibrated_ppm:.1f} ppm [{r.qc_status}]")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .calibration import RegressionCalibrator
from .deploy_config import DeployConfig
from .qc_policy import (
    QCDecision,
    QCPolicy,
    RiskScoreComputer,
    TwoThresholdDecider,
)
from .r4a_residual import R4AArtifactSet

logger = logging.getLogger(__name__)

GAS_NAMES = ["Ethanol", "CO", "Ethylene", "Methane"]

# 各类别浓度统计 (min, max)，用于 [0,1] 归一化浓度还原到 ppm 空间
# 与 utils.py 中 CONC_STATS 保持一致
CONC_STATS: Dict[int, Dict[str, float]] = {
    0: {"min": 12.5, "max": 125.0},   # 乙醇
    1: {"min": 25.0, "max": 250.0},   # CO
    2: {"min": 12.5, "max": 125.0},   # 乙烯
    3: {"min": 25.0, "max": 250.0},   # 甲烷
}


@dataclass
class DeployResult:
    """单窗口部署推理结果

    属性:
        pred_gas: 预测气体名称 ("Ethanol", "CO", "Ethylene", "Methane")
        pred_class: 预测类别 ID (0-3)
        pred_ppm: 模型原始预测浓度 (ppm)
        calibrated_ppm: 校准后浓度 (ppm)，仅 accept 时有效
        qc_status: QC 决策 ("accept" | "review" | "reject")
        risk_score: 综合风险分数 (risk_ratio)
        risk_reasons: 触发风险的风险分数名称列表
        model_version: 模型版本号
        client_id: 客户端标识
        confidence: 分类置信度 (max softmax)
        top1_confidence: top1 置信度
        top2_confidence: top2 置信度
        confidence_margin: top1 - top2 边缘
        phase: 阶段标签 (0=早期, 1=中期, 2=晚期)
        route_info: 路由信息字典
        risk_scores: 各风险分数详细值
    """
    pred_gas: str = ""
    pred_class: int = -1
    pred_ppm: float = 0.0
    calibrated_ppm: float = 0.0
    base_r3ak16_raw_ppm: float = 0.0
    routed_pred_ppm: float = 0.0
    final_ppm: float = 0.0
    qc_status: str = "accept"
    risk_score: float = 0.0
    risk_reasons: List[str] = field(default_factory=list)
    model_version: str = ""
    client_id: str = ""
    confidence: float = 0.0
    top1_confidence: float = 0.0
    top2_confidence: float = 0.0
    confidence_margin: float = 0.0
    phase: int = -1
    r4a_applied: int = 0
    r4a_delta: float = 0.0
    r4a_raw_delta: float = 0.0
    r4a_target_class: int = -1
    route_info: Dict[str, Any] = field(default_factory=dict)
    risk_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于 CSV/JSON 输出)"""
        return {
            "pred_gas": self.pred_gas,
            "pred_class": self.pred_class,
            "pred_ppm": self.pred_ppm,
            "calibrated_ppm": self.calibrated_ppm,
            "base_r3ak16_raw_ppm": self.base_r3ak16_raw_ppm,
            "routed_pred_ppm": self.routed_pred_ppm,
            "final_ppm": self.final_ppm,
            "qc_status": self.qc_status,
            "risk_score": self.risk_score,
            "risk_reasons": "|".join(self.risk_reasons),
            "model_version": self.model_version,
            "client_id": self.client_id,
            "confidence": self.confidence,
            "top1_confidence": self.top1_confidence,
            "top2_confidence": self.top2_confidence,
            "confidence_margin": self.confidence_margin,
            "phase": self.phase,
            "r4a_applied": self.r4a_applied,
            "r4a_delta": self.r4a_delta,
            "r4a_raw_delta": self.r4a_raw_delta,
            "r4a_target_class": self.r4a_target_class,
            **{f"risk_{k}": v for k, v in self.risk_scores.items()},
        }

    @classmethod
    def fieldnames(cls) -> List[str]:
        """返回 CSV 列名"""
        return [
            "client_id", "pred_gas", "pred_class", "pred_ppm", "calibrated_ppm",
            "base_r3ak16_raw_ppm", "routed_pred_ppm", "final_ppm",
            "qc_status", "risk_score", "risk_reasons", "model_version",
            "confidence", "top1_confidence", "top2_confidence", "confidence_margin",
            "phase", "r4a_applied", "r4a_delta", "r4a_raw_delta", "r4a_target_class",
        ]


class DeployPredictor:
    """部署推理总控器

    加载模型、校准参数和 QC 策略，提供统一的 predict() 接口。

    初始化方式:
        1. from_package(deploy_dir): 从部署包目录加载
        2. from_config(config): 从 DeployConfig 对象加载
        3. 直接构造: DeployPredictor(model_A, model_B, calibrator, decider)

    使用示例:
        >>> predictor = DeployPredictor.from_package("deployment_package")
        >>> # 单窗口推理
        >>> result = predictor.predict_single(features, client_id="C3")
        >>> # 批量推理
        >>> results = predictor.predict_batch(features_batch, client_id="C3")
    """

    def __init__(
        self,
        model_A: nn.Module,          # 分类模型
        model_B: nn.Module,          # 回归模型
        calibrator: RegressionCalibrator,
        risk_computer: RiskScoreComputer,
        qc_decider: TwoThresholdDecider,
        config: Optional[DeployConfig] = None,
        device: str = "cpu",
        specialist_models: Optional[Dict[int, nn.Module]] = None,
        full_model: Optional[nn.Module] = None,
        r4a_artifacts: Optional[R4AArtifactSet] = None,
    ):
        """
        参数:
            model_A: 分类模型 (输出 logits, cls_feat, reg_feat)
            model_B: 回归模型 (输出 pred_ppm)
            calibrator: 回归校准器
            risk_computer: 风险分数计算器
            qc_decider: QC 双阈值决策器
            config: 部署配置
            device: 推理设备
        """
        self.model_A = model_A
        self.model_B = model_B
        self.calibrator = calibrator
        self.risk_computer = risk_computer
        self.qc_decider = qc_decider
        self.config = config or DeployConfig()
        self.device = torch.device(device)
        self.model_version = "unknown"
        self.specialist_models = specialist_models or {}
        self.full_model = full_model
        self.r4a_artifacts = r4a_artifacts or R4AArtifactSet()
        self.last_timing_ms: Dict[str, float] = {}

        # 移动到设备并设为 eval 模式
        self.model_A.to(self.device).eval()
        self.model_B.to(self.device).eval()
        if self.full_model is not None:
            self.full_model.to(self.device).eval()
        for model in self.specialist_models.values():
            model.to(self.device).eval()

    @classmethod
    def from_package(cls, deploy_dir: str, device: str = "cpu") -> "DeployPredictor":
        """从部署包目录加载

        部署包结构:
            deploy_dir/
            ├── models/
            │   ├── classification_model.pth
            │   ├── regression_model.pth
            │   └── model_config.json
            ├── calibration/
            │   ├── routing_config.json
            │   └── calibration_stats.json
            ├── qc/
            │   ├── selected_policy.json
            │   └── qc_config.json
            └── config/
                └── deploy_config.json

        参数:
            deploy_dir: 部署包根目录
            device: 推理设备

        返回:
            DeployPredictor 实例
        """
        deploy_dir = Path(deploy_dir)

        # 加载配置
        config_path = deploy_dir / "config" / "deploy_config.json"
        if config_path.exists():
            config = DeployConfig.from_json(str(config_path))
        else:
            config = DeployConfig()

        # 加载模型配置
        model_config_path = deploy_dir / "models" / "model_config.json"
        if model_config_path.exists():
            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
        else:
            model_config = {"num_classes": 4, "feat_dim": 64, "encoder_type": "tcn", "transformer_d_model": 48}

        num_classes = int(model_config.get("num_classes", config.num_classes))
        model_A = cls._create_classifier_model(model_config, config)
        model_B = cls._create_regression_model(model_config, config)

        # 加载 checkpoint
        cls_ckpt = deploy_dir / "models" / "classification_model.pth"
        reg_ckpt = deploy_dir / "models" / "regression_model.pth"
        model_version: Union[str, int] = "unknown"

        if cls_ckpt.exists():
            state = torch.load(cls_ckpt, map_location="cpu", weights_only=False)
            model_A.load_state_dict(cls._extract_model_state(state), strict=False)
            model_version = state.get("round", state.get("model_version", "unknown"))
            logger.info(f"加载分类模型: {cls_ckpt}, version={model_version}")

        if reg_ckpt.exists():
            state = torch.load(reg_ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = model_B.load_state_dict(
                cls._extract_model_state(state),
                strict=False,
            )
            if missing:
                logger.info(f"回归模型加载: {len(missing)} 个缺失键")
            if unexpected:
                logger.warning(f"回归模型加载: {len(unexpected)} 个多余键")
            logger.info(f"加载回归模型: {reg_ckpt}")

        full_model = None
        if config.full_model_checkpoint:
            full_ckpt = deploy_dir / config.full_model_checkpoint
        else:
            full_ckpt = deploy_dir / "models" / "full_model.pth"
        if full_ckpt.exists():
            full_model = cls._create_regression_model(model_config, config)
            state = torch.load(full_ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = full_model.load_state_dict(
                cls._extract_model_state(state),
                strict=False,
            )
            if missing:
                logger.info(f"full 校准模型加载: {len(missing)} 个缺失键")
            if unexpected:
                logger.warning(f"full 校准模型加载: {len(unexpected)} 个多余键")
            logger.info(f"加载 full 校准模型: {full_ckpt}")

        # 加载校准器
        calibrator = RegressionCalibrator(
            num_classes=num_classes,
            num_phases=config.num_phases,
            conc_ranges=config.class_concentration_ranges,
        )
        routing_path = deploy_dir / "calibration" / "routing_config.json"
        if routing_path.exists():
            calibrator.load_routing_config_json(str(routing_path))
            logger.info(f"加载校准路由配置: {routing_path}")

        # 加载校准参考数据 (用于响应签名比较)
        calib_refs = {}
        calib_stats_path = deploy_dir / "calibration" / "calibration_stats.json"
        if calib_stats_path.exists():
            calib_refs = cls._load_calibration_refs(calib_stats_path)
            if calib_refs:
                logger.info(f"加载校准参考数据: {calib_stats_path}")
            else:
                logger.info(f"校准统计文件不含 response refs, QC 响应风险跳过: {calib_stats_path}")

        risk_computer = RiskScoreComputer(calib_refs=calib_refs)

        # 加载 QC 策略
        qc_decider = TwoThresholdDecider()
        qc_policy_path = deploy_dir / "qc" / "selected_policy.json"
        if qc_policy_path.exists():
            qc_decider.load_policies_json(str(qc_policy_path))
            logger.info(f"加载 QC 策略: {qc_policy_path}")

        specialist_models = cls._load_specialist_models(
            deploy_dir / "models" / "specialists",
            model_config,
            config,
        )
        r4a_artifacts = R4AArtifactSet.from_dir(deploy_dir / "r4a")

        predictor = cls(
            model_A=model_A,
            model_B=model_B,
            calibrator=calibrator,
            risk_computer=risk_computer,
            qc_decider=qc_decider,
            config=config,
            device=device,
            specialist_models=specialist_models,
            full_model=full_model,
            r4a_artifacts=r4a_artifacts,
        )
        predictor.model_version = str(model_version)
        return predictor

    @classmethod
    def from_config(cls, config: DeployConfig) -> "DeployPredictor":
        """从 DeployConfig 对象加载 (用于自定义路径)"""
        num_classes = int(config.model_config.get("num_classes", config.num_classes))
        model_A = cls._create_classifier_model(config.model_config, config)
        model_B = cls._create_regression_model(config.model_config, config)

        if config.classifier_checkpoint:
            state = torch.load(config.classifier_checkpoint, map_location="cpu", weights_only=False)
            model_A.load_state_dict(cls._extract_model_state(state), strict=False)

        if config.regression_checkpoint:
            state = torch.load(config.regression_checkpoint, map_location="cpu", weights_only=False)
            model_B.load_state_dict(cls._extract_model_state(state), strict=False)

        full_model = None
        if config.full_model_checkpoint:
            full_model = cls._create_regression_model(config.model_config, config)
            state = torch.load(config.full_model_checkpoint, map_location="cpu", weights_only=False)
            full_model.load_state_dict(cls._extract_model_state(state), strict=False)

        calibrator = RegressionCalibrator(
            num_classes=num_classes,
            num_phases=config.num_phases,
            conc_ranges=config.class_concentration_ranges,
        )
        if config.routing_config_path:
            calibrator.load_routing_config_json(config.routing_config_path)

        risk_computer = RiskScoreComputer()

        qc_decider = TwoThresholdDecider()
        if config.qc_policy_path:
            qc_decider.load_policies_json(config.qc_policy_path)

        return cls(
            model_A=model_A,
            model_B=model_B,
            calibrator=calibrator,
            risk_computer=risk_computer,
            qc_decider=qc_decider,
            config=config,
            device=config.device,
            full_model=full_model,
        )

    @staticmethod
    def _extract_model_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
        """Return the model state_dict from common checkpoint layouts."""
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                return checkpoint["model_state"]
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
        return checkpoint

    @staticmethod
    def _load_calibration_refs(path: Path) -> Dict[int, Dict[str, Any]]:
        """Load optional response-reference stats, ignoring plain calibration metrics."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        refs: Dict[int, Dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return refs

        ref_root = raw.get("response_refs", raw)
        if not isinstance(ref_root, dict):
            return refs

        for cls_key, ref in ref_root.items():
            try:
                cls_id = int(cls_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(ref, dict):
                continue
            if not any(arr_key in ref for arr_key in ("center", "scale", "z_sigs")):
                continue
            parsed = dict(ref)
            for arr_key in ("center", "scale", "z_sigs"):
                if arr_key in parsed:
                    parsed[arr_key] = np.asarray(parsed[arr_key])
            refs[cls_id] = parsed
        return refs

    @staticmethod
    def _apply_model_config(base_config: Any, model_config: Dict[str, Any]) -> Any:
        """Apply deploy JSON model fields to an FLConfig-like object."""
        mapping = {
            "num_classes": "NUM_CLASSES",
            "num_sensors": "INPUT_DIM",
            "input_dim": "INPUT_DIM",
            "feat_dim": "HIDDEN_DIM2",
            "encoder_type": "USE_TRANSFORMER_ENCODER",
            "transformer_d_model": "TRANSFORMER_D_MODEL",
            "transformer_nhead": "TRANSFORMER_NHEAD",
            "transformer_num_layers": "TRANSFORMER_NUM_LAYERS",
            "transformer_ff_dim": "TRANSFORMER_FF_DIM",
            "reg_head_depth": "REG_HEAD_DEPTH",
            "reg_output_mode": "REG_OUTPUT_MODE",
            "reg_window_stats": "REG_WINDOW_STATS",
            "reg_window_stats_mode": "REG_WINDOW_STATS_MODE",
            "reg_window_stats_dim": "REG_WINDOW_STATS_DIM",
            "reg_response_branch": "REG_RESPONSE_BRANCH",
            "reg_dct_k": "REG_DCT_K",
            "reg_dct_gamma_init": "REG_DCT_GAMMA_INIT",
            "reg_dct_dropout": "REG_DCT_DROPOUT",
            "reg_msconv_channels": "REG_MSCONV_CHANNELS",
            "reg_msconv_kernels": "REG_MSCONV_KERNELS",
            "reg_msconv_gamma_init": "REG_MSCONV_GAMMA_INIT",
            "reg_msconv_dropout": "REG_MSCONV_DROPOUT",
            "reg_tcn_adapter": "REG_TCN_ADAPTER",
            "reg_tcn_adapter_kernel": "REG_TCN_ADAPTER_KERNEL",
            "reg_tcn_adapter_gamma_init": "REG_TCN_ADAPTER_GAMMA_INIT",
            "reg_tcn_adapter_dropout": "REG_TCN_ADAPTER_DROPOUT",
            "reg_use_shared_trunk": "REG_USE_SHARED_TRUNK",
            "reg_shared_trunk_dim": "REG_SHARED_TRUNK_DIM",
            "reg_gas_emb_dim": "REG_GAS_EMB_DIM",
            "reg_residual_head_depth": "REG_RESIDUAL_HEAD_DEPTH",
            "use_reg_ratio_branch": "USE_REG_RATIO_BRANCH",
            "reg_ratio_gamma_init": "REG_RATIO_GAMMA_INIT",
            "reg_ratio_dropout": "REG_RATIO_DROPOUT",
        }
        for key, attr in mapping.items():
            if key not in model_config:
                continue
            value = model_config[key]
            if key == "encoder_type":
                value = str(value).lower() == "transformer"
            setattr(base_config, attr, value)
        return base_config

    @classmethod
    def _create_classifier_model(
        cls,
        model_config: Dict[str, Any],
        deploy_config: DeployConfig,
    ) -> nn.Module:
        """Create the same classification model family used by Flower."""
        from config import FLConfig
        from utils import create_model_by_config

        cfg = cls._apply_model_config(FLConfig(), model_config)
        cfg.DEVICE = deploy_config.device
        cfg.USE_REG_LOSS = False
        return create_model_by_config(cfg, with_reg_head=False)

    @classmethod
    def _create_regression_model(
        cls,
        model_config: Dict[str, Any],
        deploy_config: DeployConfig,
    ) -> nn.Module:
        """Create regression model B using the regression-task config path."""
        from gaps_flower.regression_task import create_regression_model, make_regression_config

        cfg = make_regression_config(device=deploy_config.device, batch_size=32)
        cfg = cls._apply_model_config(cfg, model_config)
        return create_regression_model(cfg)

    @classmethod
    def _load_specialist_models(
        cls,
        specialist_dir: Path,
        model_config: Dict[str, Any],
        deploy_config: DeployConfig,
    ) -> Dict[int, nn.Module]:
        """Load optional per-class specialist regression models."""
        models: Dict[int, nn.Module] = {}
        if not specialist_dir.exists():
            return models
        for ckpt_path in sorted(specialist_dir.glob("*.pth")):
            stem = ckpt_path.stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            if not digits:
                logger.warning(f"跳过无法解析 class id 的 specialist checkpoint: {ckpt_path}")
                continue
            class_id = int(digits)
            model = cls._create_regression_model(model_config, deploy_config)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(
                cls._extract_model_state(state),
                strict=False,
            )
            if missing:
                logger.info(f"specialist class {class_id}: {len(missing)} 个缺失键")
            if unexpected:
                logger.warning(f"specialist class {class_id}: {len(unexpected)} 个多余键")
            models[class_id] = model
            logger.info(f"加载 specialist class {class_id}: {ckpt_path}")
        return models

    def predict_single(
        self,
        features: np.ndarray,
        client_id: str = "ALL",
        phase: int = -1,
    ) -> DeployResult:
        """单窗口推理

        参数:
            features: 输入特征 (T, C) 形状，T=时间步长, C=通道数
            client_id: 客户端标识
            phase: 阶段标签 (0=早期, 1=中期, 2=晚期, -1=未知)

        返回:
            DeployResult 推理结果
        """
        results = self.predict_batch(
            features[np.newaxis, ...], client_id=client_id, phase=phase
        )
        return results[0]

    @staticmethod
    def _denormalize_by_class(pred_norm: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """将归一化浓度 [0, 1] 按类别反归一化到 ppm 空间

        反归一化公式:
            ppm = pred_norm * (max_c - min_c) + min_c

        其中 min_c, max_c 来自 CONC_STATS 各类别浓度范围。

        参数:
            pred_norm: 归一化浓度 (N,) 形状，值域 [0, 1]
            class_ids: 类别 ID (N,) 形状

        返回:
            (N,) ppm 浓度
        """
        pred_norm = np.asarray(pred_norm, dtype=np.float64).ravel()
        class_ids = np.asarray(class_ids, dtype=int).ravel()
        ppm = np.zeros_like(pred_norm)
        for cls_id in sorted(CONC_STATS.keys()):
            mask = class_ids == cls_id
            if mask.any():
                min_c = CONC_STATS[cls_id]["min"]
                max_c = CONC_STATS[cls_id]["max"]
                ppm[mask] = pred_norm[mask] * (max_c - min_c) + min_c
        return ppm

    def predict_batch(
        self,
        features: np.ndarray,
        client_id: str = "ALL",
        phase: Union[int, Sequence[int], np.ndarray] = -1,
    ) -> List[DeployResult]:
        """批量窗口推理

        参数:
            features: 输入特征 (N, T, C) 形状
            client_id: 客户端标识
            phase: 阶段标签 (批量统一)

        返回:
            DeployResult 列表
        """
        total_started = time.perf_counter()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # (T, C) → (1, T, C)

        phase_raw = self._normalize_phase_input(phase, len(features))
        phase_for_model = np.where(phase_raw < 0, 0, phase_raw).astype(np.int64)
        x = torch.from_numpy(features).to(self.device)

        with torch.no_grad():
            # 1. 分类模型前向
            stage_started = time.perf_counter()
            logits, _cls_feat, _cls_reg_feat = self.model_A(x)

            # 2. 分类后处理
            probs = torch.softmax(logits, dim=1)
            top1_vals, top1_ids = probs.max(dim=1)
            top2_vals = torch.zeros_like(top1_vals)
            if probs.size(1) > 1:
                sorted_probs, _ = probs.sort(dim=1, descending=True)
                top2_vals = sorted_probs[:, 1]
            classification_ms = (time.perf_counter() - stage_started) * 1000.0

            # 3. 回归模型前向
            stage_started = time.perf_counter()
            pred_cls = top1_ids  # 使用分类结果作为路由
            phase_tensor = torch.from_numpy(phase_for_model).long().to(self.device)
            _, _, reg_feat = self.model_B(x)
            base_pred_norm = self.model_B.forward_reg(
                reg_feat,
                y_cls=pred_cls,
                y_phase=phase_tensor,
            )
            pred_ppm = base_pred_norm.clone()
            if self.full_model is not None:
                for class_id, mode in self.calibrator.selected_modes.items():
                    if mode != "full":
                        continue
                    mask = pred_cls == int(class_id)
                    if not mask.any():
                        continue
                    _, _, full_feat = self.full_model(x[mask])
                    pred_ppm[mask] = self.full_model.forward_reg(
                        full_feat,
                        y_cls=pred_cls[mask],
                        y_phase=phase_tensor[mask],
                    )
            for class_id, specialist_model in self.specialist_models.items():
                mode = self.calibrator.selected_modes.get(int(class_id), "none")
                if mode not in {"specialist", "specialist_full"}:
                    continue
                mask = pred_cls == int(class_id)
                if not mask.any():
                    continue
                _, _, specialist_feat = specialist_model(x[mask])
                pred_ppm[mask] = specialist_model.forward_reg(
                    specialist_feat,
                    y_cls=pred_cls[mask],
                    y_phase=phase_tensor[mask],
                )
            regression_ms = (time.perf_counter() - stage_started) * 1000.0

        # 转为 numpy
        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        base_pred_norm_np = base_pred_norm.detach().cpu().numpy().ravel()
        pred_ppm_norm = pred_ppm.detach().cpu().numpy().ravel()  # [0, 1] 归一化值
        pred_cls_np = pred_cls.detach().cpu().numpy().ravel()
        top1_np = top1_vals.detach().cpu().numpy().ravel()
        top2_np = top2_vals.detach().cpu().numpy().ravel()
        features_np = features

        # 4. 反归一化: [0, 1] → ppm 空间
        #    回归模型 B 输出的是归一化浓度, 需要按类别反归一化到 ppm
        base_pred_ppm_np = self._denormalize_by_class(base_pred_norm_np, pred_cls_np)
        pred_ppm_np = self._denormalize_by_class(pred_ppm_norm, pred_cls_np)

        # 5. 校准 (在 ppm 空间进行)
        stage_started = time.perf_counter()
        calibrated_ppm = self.calibrator.calibrate(pred_ppm_np, pred_cls_np, phase_for_model)
        calibration_ms = (time.perf_counter() - stage_started) * 1000.0

        # 6. 风险分数计算
        stage_started = time.perf_counter()
        risk_scores_list = self.risk_computer.compute_batch(
            logits_batch=logits_np,
            pred_ppm_batch=calibrated_ppm,
            class_ids=pred_cls_np,
            features_batch=features_np,
        )
        risk_ms = (time.perf_counter() - stage_started) * 1000.0

        # 7. QC 双阈值决策
        stage_started = time.perf_counter()
        decisions = self.qc_decider.decide_batch(
            risk_scores_list,
            client_ids=[client_id] * len(pred_ppm_np),
        )
        qc_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        calibrated_ppm, r4a_meta = self._apply_r4a_batch(
            calibrated_ppm=calibrated_ppm,
            pred_cls_np=pred_cls_np,
            decisions=decisions,
            features_np=features_np,
            client_id=client_id,
            phase_raw=phase_raw,
            top1_np=top1_np,
            top2_np=top2_np,
            risk_scores_list=risk_scores_list,
        )
        r4a_ms = (time.perf_counter() - stage_started) * 1000.0

        # 8. 组装结果
        stage_started = time.perf_counter()
        results = []
        for i in range(len(pred_ppm_np)):
            cls_id = int(pred_cls_np[i])
            margin = float(top1_np[i] - top2_np[i])

            result = DeployResult(
                pred_gas=GAS_NAMES[cls_id] if 0 <= cls_id < len(GAS_NAMES) else f"Class{cls_id}",
                pred_class=cls_id,
                pred_ppm=float(pred_ppm_np[i]),
                calibrated_ppm=float(calibrated_ppm[i]) if decisions[i].decision == "accept" else float(calibrated_ppm[i]),
                base_r3ak16_raw_ppm=float(base_pred_ppm_np[i]),
                routed_pred_ppm=float(pred_ppm_np[i]),
                final_ppm=float(calibrated_ppm[i]),
                qc_status=decisions[i].decision,
                risk_score=decisions[i].risk_ratio,
                risk_reasons=decisions[i].risk_reasons,
                model_version=self.model_version,
                client_id=client_id,
                confidence=float(top1_np[i]),
                top1_confidence=float(top1_np[i]),
                top2_confidence=float(top2_np[i]),
                confidence_margin=margin,
                phase=int(phase_raw[i]),
                r4a_applied=int(r4a_meta[i]["applied"]),
                r4a_delta=float(r4a_meta[i]["delta"]),
                r4a_raw_delta=float(r4a_meta[i]["raw_delta"]),
                r4a_target_class=int(r4a_meta[i]["target_class"]),
                route_info={
                    "route_mode": "hard",
                    "softmax": probs_np[i].tolist(),
                },
                risk_scores=risk_scores_list[i],
            )
            results.append(result)

        assemble_ms = (time.perf_counter() - stage_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        self.last_timing_ms = {
            "batch_size": int(len(features)),
            "classification_ms": float(classification_ms),
            "regression_ms": float(regression_ms),
            "auto_v2_calibration_ms": float(calibration_ms),
            "risk_score_ms": float(risk_ms),
            "qc_decision_ms": float(qc_ms),
            "r4a_ms": float(r4a_ms),
            "result_assembly_ms": float(assemble_ms),
            "total_inference_ms": float(total_ms),
        }

        return results

    def _apply_r4a_batch(
        self,
        calibrated_ppm: np.ndarray,
        pred_cls_np: np.ndarray,
        decisions: Sequence[QCDecision],
        features_np: np.ndarray,
        client_id: str,
        phase_raw: np.ndarray,
        top1_np: np.ndarray,
        top2_np: np.ndarray,
        risk_scores_list: Sequence[Dict[str, float]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply optional R4a artifacts after QC decisions."""
        corrected = np.asarray(calibrated_ppm, dtype=np.float64).copy()
        meta: List[Dict[str, Any]] = []
        for i in range(len(corrected)):
            margin = float(top1_np[i] - top2_np[i])
            new_ppm, delta, raw_delta, applied, target_class = self.r4a_artifacts.apply(
                calibrated_ppm=float(corrected[i]),
                pred_class=int(pred_cls_np[i]),
                qc_status=decisions[i].decision,
                feature_window=features_np[i],
                client_id=client_id,
                phase=int(phase_raw[i]),
                confidence=float(top1_np[i]),
                top1_confidence=float(top1_np[i]),
                top2_confidence=float(top2_np[i]),
                confidence_margin=margin,
                risk_score=float(decisions[i].risk_ratio),
                risk_scores=risk_scores_list[i],
            )
            corrected[i] = new_ppm
            meta.append({
                "applied": int(applied),
                "delta": float(delta),
                "raw_delta": float(raw_delta),
                "target_class": int(target_class),
            })
        return corrected, meta

    @staticmethod
    def _normalize_phase_input(
        phase: Union[int, Sequence[int], np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        """Return per-sample phase ids, preserving -1 for unknown phases."""
        if np.isscalar(phase):
            return np.full(n_samples, int(phase), dtype=np.int64)
        phase_arr = np.asarray(phase, dtype=np.int64).reshape(-1)
        if len(phase_arr) != n_samples:
            raise ValueError(
                f"phase length {len(phase_arr)} does not match batch size {n_samples}"
            )
        return phase_arr

    def predict_generator(
        self,
        data_loader: torch.utils.data.DataLoader,
        client_id: str = "ALL",
    ) -> "Generator[List[DeployResult], None, None]":
        """从 DataLoader 批量推理 (生成器，适合大数据集)

        参数:
            data_loader: PyTorch DataLoader
            client_id: 客户端标识

        Yields:
            DeployResult 列表 (每批)
        """
        self.model_A.eval()
        self.model_B.eval()
        if self.full_model is not None:
            self.full_model.eval()
        for model in self.specialist_models.values():
            model.eval()

        with torch.no_grad():
            for x, y_cls, y_reg_full, y_phase in data_loader:
                x = x.to(self.device)

                logits, cls_feat, reg_feat = self.model_A(x)
                probs = torch.softmax(logits, dim=1)
                top1_vals, top1_ids = probs.max(dim=1)
                top2_vals = torch.zeros_like(top1_vals)
                if probs.size(1) > 1:
                    sorted_probs, _ = probs.sort(dim=1, descending=True)
                    top2_vals = sorted_probs[:, 1]
                _, _, reg_feat_b = self.model_B(x)
                base_pred_norm = self.model_B.forward_reg(
                    reg_feat_b,
                    y_cls=top1_ids,
                    y_phase=y_phase.to(self.device),
                )
                pred_norm = base_pred_norm.clone()
                y_phase_device = y_phase.to(self.device)
                if self.full_model is not None:
                    for class_id, mode in self.calibrator.selected_modes.items():
                        if mode != "full":
                            continue
                        mask = top1_ids == int(class_id)
                        if not mask.any():
                            continue
                        _, _, full_feat = self.full_model(x[mask])
                        pred_norm[mask] = self.full_model.forward_reg(
                            full_feat,
                            y_cls=top1_ids[mask],
                            y_phase=y_phase_device[mask],
                        )
                for class_id, specialist_model in self.specialist_models.items():
                    mode = self.calibrator.selected_modes.get(int(class_id), "none")
                    if mode not in {"specialist", "specialist_full"}:
                        continue
                    mask = top1_ids == int(class_id)
                    if not mask.any():
                        continue
                    _, _, specialist_feat = specialist_model(x[mask])
                    pred_norm[mask] = specialist_model.forward_reg(
                        specialist_feat,
                        y_cls=top1_ids[mask],
                        y_phase=y_phase_device[mask],
                    )

                logits_np = logits.cpu().numpy()
                base_pred_norm_np = base_pred_norm.cpu().numpy().ravel()
                pred_norm_np = pred_norm.cpu().numpy().ravel()
                pred_cls_np = top1_ids.cpu().numpy().ravel()
                top1_np = top1_vals.cpu().numpy().ravel()
                top2_np = top2_vals.cpu().numpy().ravel()
                features_np = x.cpu().numpy()
                y_phase_np = y_phase.cpu().numpy().ravel()
                base_pred_ppm_np = self._denormalize_by_class(base_pred_norm_np, pred_cls_np)
                pred_ppm_np = self._denormalize_by_class(pred_norm_np, pred_cls_np)

                calibrated_ppm = self.calibrator.calibrate(pred_ppm_np, pred_cls_np, y_phase_np)

                risk_scores_list = self.risk_computer.compute_batch(
                    logits_batch=logits_np,
                    pred_ppm_batch=calibrated_ppm,
                    class_ids=pred_cls_np,
                    features_batch=features_np,
                )
                decisions = self.qc_decider.decide_batch(
                    risk_scores_list,
                    client_ids=[client_id] * len(pred_ppm_np),
                )

                calibrated_ppm, r4a_meta = self._apply_r4a_batch(
                    calibrated_ppm=calibrated_ppm,
                    pred_cls_np=pred_cls_np,
                    decisions=decisions,
                    features_np=features_np,
                    client_id=client_id,
                    phase_raw=y_phase_np,
                    top1_np=top1_np,
                    top2_np=top2_np,
                    risk_scores_list=risk_scores_list,
                )

                batch_results = []
                for i in range(len(pred_ppm_np)):
                    margin = float(top1_np[i] - top2_np[i])
                    batch_results.append(DeployResult(
                        pred_gas=GAS_NAMES[pred_cls_np[i]] if 0 <= pred_cls_np[i] < len(GAS_NAMES) else f"Class{pred_cls_np[i]}",
                        pred_class=int(pred_cls_np[i]),
                        pred_ppm=float(pred_ppm_np[i]),
                        calibrated_ppm=float(calibrated_ppm[i]),
                        base_r3ak16_raw_ppm=float(base_pred_ppm_np[i]),
                        routed_pred_ppm=float(pred_ppm_np[i]),
                        final_ppm=float(calibrated_ppm[i]),
                        qc_status=decisions[i].decision,
                        risk_score=decisions[i].risk_ratio,
                        risk_reasons=decisions[i].risk_reasons,
                        model_version=self.model_version,
                        client_id=client_id,
                        confidence=float(top1_np[i]),
                        top1_confidence=float(top1_np[i]),
                        top2_confidence=float(top2_np[i]),
                        confidence_margin=margin,
                        phase=int(y_phase_np[i]),
                        r4a_applied=int(r4a_meta[i]["applied"]),
                        r4a_delta=float(r4a_meta[i]["delta"]),
                        r4a_raw_delta=float(r4a_meta[i]["raw_delta"]),
                        r4a_target_class=int(r4a_meta[i]["target_class"]),
                        risk_scores=risk_scores_list[i],
                    ))

                yield batch_results

    def to(self, device: str) -> "DeployPredictor":
        """移动到指定设备"""
        self.device = torch.device(device)
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        if self.full_model is not None:
            self.full_model.to(self.device)
        for model in self.specialist_models.values():
            model.to(self.device)
        return self
