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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    route_info: Dict[str, Any] = field(default_factory=dict)
    risk_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于 CSV/JSON 输出)"""
        return {
            "pred_gas": self.pred_gas,
            "pred_class": self.pred_class,
            "pred_ppm": self.pred_ppm,
            "calibrated_ppm": self.calibrated_ppm,
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
            **{f"risk_{k}": v for k, v in self.risk_scores.items()},
        }

    @classmethod
    def fieldnames(cls) -> List[str]:
        """返回 CSV 列名"""
        return [
            "client_id", "pred_gas", "pred_class", "pred_ppm", "calibrated_ppm",
            "qc_status", "risk_score", "risk_reasons", "model_version",
            "confidence", "top1_confidence", "top2_confidence", "confidence_margin",
            "phase",
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

        # 移动到设备并设为 eval 模式
        self.model_A.to(self.device).eval()
        self.model_B.to(self.device).eval()

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

        # 创建模型 (延迟导入以避免循环依赖)
        # FedGasModel = FedGasBaseModel (仅分类), FedGasMultiTaskModel (分类+回归)
        from model import FedGasModel, FedGasMultiTaskModel

        num_classes = model_config.get("num_classes", 4)
        feat_dim = model_config.get("feat_dim", 64)
        encoder_type = model_config.get("encoder_type", "tcn")
        transformer_d_model = model_config.get("transformer_d_model", 48)

        model_A = FedGasModel(
            num_classes=num_classes,
            feat_dim=feat_dim,
            encoder_type=encoder_type,
            transformer_d_model=transformer_d_model,
        )
        # 模型 B 必须使用 FedGasMultiTaskModel 以支持回归头
        model_B = FedGasMultiTaskModel(
            num_classes=num_classes,
            feat_dim=feat_dim,
            encoder_type=encoder_type,
            transformer_d_model=transformer_d_model,
            use_dual_proj=True,
            reg_grad_detach=True,
        )

        # 加载 checkpoint
        cls_ckpt = deploy_dir / "models" / "classification_model.pth"
        reg_ckpt = deploy_dir / "models" / "regression_model.pth"

        if cls_ckpt.exists():
            state = torch.load(cls_ckpt, map_location="cpu", weights_only=False)
            model_A.load_state_dict(state.get("model_state", state), strict=False)
            model_version = state.get("round", state.get("model_version", "unknown"))
            logger.info(f"加载分类模型: {cls_ckpt}, version={model_version}")

        if reg_ckpt.exists():
            state = torch.load(reg_ckpt, map_location="cpu", weights_only=False)
            model_B.load_state_dict(state.get("model_state", state), strict=False)
            logger.info(f"加载回归模型: {reg_ckpt}")

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
            with open(calib_stats_path, "r", encoding="utf-8") as f:
                calib_refs = json.load(f)
            # 转换 numpy 数组
            for cls_key, ref in calib_refs.items():
                for arr_key in ["center", "scale", "z_sigs"]:
                    if arr_key in ref:
                        ref[arr_key] = np.array(ref[arr_key])
            logger.info(f"加载校准参考数据: {calib_stats_path}")

        risk_computer = RiskScoreComputer(calib_refs=calib_refs)

        # 加载 QC 策略
        qc_decider = TwoThresholdDecider()
        qc_policy_path = deploy_dir / "qc" / "selected_policy.json"
        if qc_policy_path.exists():
            qc_decider.load_policies_json(str(qc_policy_path))
            logger.info(f"加载 QC 策略: {qc_policy_path}")

        predictor = cls(
            model_A=model_A,
            model_B=model_B,
            calibrator=calibrator,
            risk_computer=risk_computer,
            qc_decider=qc_decider,
            config=config,
            device=device,
        )
        predictor.model_version = str(model_version)
        return predictor

    @classmethod
    def from_config(cls, config: DeployConfig) -> "DeployPredictor":
        """从 DeployConfig 对象加载 (用于自定义路径)"""
        # 创建模型 (与 from_package 相同逻辑)
        from model import FedGasModel, FedGasMultiTaskModel

        num_classes = config.model_config.get("num_classes", config.num_classes)
        feat_dim = config.model_config.get("feat_dim", 64)
        encoder_type = config.model_config.get("encoder_type", "tcn")
        transformer_d_model = config.model_config.get("transformer_d_model", 48)

        model_A = FedGasModel(
            num_classes=num_classes,
            feat_dim=feat_dim,
            encoder_type=encoder_type,
            transformer_d_model=transformer_d_model,
        )
        # 模型 B 必须使用 FedGasMultiTaskModel 以支持回归头
        model_B = FedGasMultiTaskModel(
            num_classes=num_classes,
            feat_dim=feat_dim,
            encoder_type=encoder_type,
            transformer_d_model=transformer_d_model,
            use_dual_proj=True,
            reg_grad_detach=True,
        )

        if config.classifier_checkpoint:
            state = torch.load(config.classifier_checkpoint, map_location="cpu", weights_only=False)
            model_A.load_state_dict(state.get("model_state", state), strict=False)

        if config.regression_checkpoint:
            state = torch.load(config.regression_checkpoint, map_location="cpu", weights_only=False)
            model_B.load_state_dict(state.get("model_state", state), strict=False)

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
        )

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
        phase: int = -1,
    ) -> List[DeployResult]:
        """批量窗口推理

        参数:
            features: 输入特征 (N, T, C) 形状
            client_id: 客户端标识
            phase: 阶段标签 (批量统一)

        返回:
            DeployResult 列表
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # (T, C) → (1, T, C)

        x = torch.from_numpy(features).to(self.device)

        with torch.no_grad():
            # 1. 分类模型前向
            logits, cls_feat, reg_feat = self.model_A(x)

            # 2. 分类后处理
            probs = torch.softmax(logits, dim=1)
            top1_vals, top1_ids = probs.max(dim=1)
            top2_vals = torch.zeros_like(top1_vals)
            if probs.size(1) > 1:
                sorted_probs, _ = probs.sort(dim=1, descending=True)
                top2_vals = sorted_probs[:, 1]

            # 3. 回归模型前向
            pred_cls = top1_ids  # 使用分类结果作为路由
            phase_tensor = torch.full_like(pred_cls, max(phase, 0))
            pred_ppm = self.model_B.forward_reg(reg_feat, y_cls=pred_cls, y_phase=phase_tensor)

        # 转为 numpy
        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        pred_ppm_norm = pred_ppm.detach().cpu().numpy().ravel()  # [0, 1] 归一化值
        pred_cls_np = pred_cls.detach().cpu().numpy().ravel()
        top1_np = top1_vals.detach().cpu().numpy().ravel()
        top2_np = top2_vals.detach().cpu().numpy().ravel()
        features_np = features

        # 4. 反归一化: [0, 1] → ppm 空间
        #    回归模型 B 输出的是归一化浓度, 需要按类别反归一化到 ppm
        pred_ppm_np = self._denormalize_by_class(pred_ppm_norm, pred_cls_np)

        # 5. 校准 (在 ppm 空间进行)
        phase_ids = np.full_like(pred_cls_np, max(phase, 0))
        calibrated_ppm = self.calibrator.calibrate(pred_ppm_np, pred_cls_np, phase_ids)

        # 6. 风险分数计算
        risk_scores_list = self.risk_computer.compute_batch(
            logits_batch=logits_np,
            pred_ppm_batch=calibrated_ppm,
            class_ids=pred_cls_np,
            features_batch=features_np,
        )

        # 7. QC 双阈值决策
        decisions = self.qc_decider.decide_batch(
            risk_scores_list,
            client_ids=[client_id] * len(pred_ppm_np),
        )

        # 8. 组装结果
        results = []
        for i in range(len(pred_ppm_np)):
            cls_id = int(pred_cls_np[i])
            margin = float(top1_np[i] - top2_np[i])

            result = DeployResult(
                pred_gas=GAS_NAMES[cls_id] if 0 <= cls_id < len(GAS_NAMES) else f"Class{cls_id}",
                pred_class=cls_id,
                pred_ppm=float(pred_ppm_np[i]),
                calibrated_ppm=float(calibrated_ppm[i]) if decisions[i].decision == "accept" else float(calibrated_ppm[i]),
                qc_status=decisions[i].decision,
                risk_score=decisions[i].risk_ratio,
                risk_reasons=decisions[i].risk_reasons,
                model_version=self.model_version,
                client_id=client_id,
                confidence=float(top1_np[i]),
                top1_confidence=float(top1_np[i]),
                top2_confidence=float(top2_np[i]),
                confidence_margin=margin,
                phase=phase,
                route_info={
                    "route_mode": "hard",
                    "softmax": probs_np[i].tolist(),
                },
                risk_scores=risk_scores_list[i],
            )
            results.append(result)

        return results

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

        with torch.no_grad():
            for x, y_cls, y_reg_full, y_phase in data_loader:
                x = x.to(self.device)

                logits, cls_feat, reg_feat = self.model_A(x)
                probs = torch.softmax(logits, dim=1)
                top1_vals, top1_ids = probs.max(dim=1)
                pred_ppm = self.model_B.forward_reg(reg_feat, y_cls=top1_ids, y_phase=y_phase.to(self.device))

                logits_np = logits.cpu().numpy()
                pred_ppm_np = pred_ppm.cpu().numpy().ravel()
                pred_cls_np = top1_ids.cpu().numpy().ravel()
                features_np = x.cpu().numpy()
                y_phase_np = y_phase.cpu().numpy().ravel()

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

                batch_results = []
                for i in range(len(pred_ppm_np)):
                    batch_results.append(DeployResult(
                        pred_gas=GAS_NAMES[pred_cls_np[i]] if 0 <= pred_cls_np[i] < len(GAS_NAMES) else f"Class{pred_cls_np[i]}",
                        pred_class=int(pred_cls_np[i]),
                        pred_ppm=float(pred_ppm_np[i]),
                        calibrated_ppm=float(calibrated_ppm[i]),
                        qc_status=decisions[i].decision,
                        risk_score=decisions[i].risk_ratio,
                        risk_reasons=decisions[i].risk_reasons,
                        model_version=self.model_version,
                        client_id=client_id,
                        phase=int(y_phase_np[i]),
                    ))

                yield batch_results

    def to(self, device: str) -> "DeployPredictor":
        """移动到指定设备"""
        self.device = torch.device(device)
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        return self