"""
部署配置管理模块

管理部署推理所需的所有配置参数，包括模型路径、校准参数路径、
QC 策略路径、风险阈值等。支持从 JSON 文件加载和命令行参数覆盖。

配置加载优先级: JSON 文件 < 环境变量 < 命令行参数
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DeployConfig:
    """部署推理配置

    属性:
        classifier_checkpoint: 分类模型A checkpoint 路径
        regression_checkpoint: 回归模型B checkpoint 路径
        full_model_checkpoint: 目标端 full 校准模型 checkpoint 路径，可选
        calibration_params_path: 校准参数 JSON 路径 (affine/bias/phase_affine)
        routing_config_path: 路由配置 JSON 路径 (per-class 模式选择)
        qc_policy_path: QC 策略 JSON 路径 (风险分数列表 + 阈值)
        qc_thresholds_path: 双阈值 JSON 路径 (low_ratio, high_ratio)
        calibration_ref_dir: 校准集参考目录 (用于响应签名比较)
        device: 推理设备 (cpu/cuda)
        model_config: 模型结构配置 (num_classes, d_model, encoder_type 等)
        gas_names: 气体名称列表
        num_classes: 分类类别数
        num_phases: 阶段数 (早期/中期/晚期)
        class_concentration_ranges: 各类别浓度范围 (用于 clamp)
        risk_score_names: 使用的风险分数列表
        low_ratio: 双阈值下界 (≤ low_ratio → accept)
        high_ratio: 双阈值上界 (> high_ratio → reject)
        high_error_ppm: 高误差阈值 (ppm)
    """

    # 模型路径
    classifier_checkpoint: str = ""
    regression_checkpoint: str = ""
    full_model_checkpoint: str = ""

    # 校准参数路径
    calibration_params_path: str = ""
    routing_config_path: str = ""

    # QC 路径
    qc_policy_path: str = ""
    qc_thresholds_path: str = ""

    # 校准参考目录 (用于响应签名比较)
    calibration_ref_dir: str = ""

    # 推理设备
    device: str = "cpu"

    # 模型配置
    model_config: Dict[str, Any] = field(default_factory=dict)

    # 气体与类别
    gas_names: List[str] = field(default_factory=lambda: ["Ethanol", "CO", "Ethylene", "Methane"])
    num_classes: int = 4
    num_phases: int = 3

    # 浓度范围 (用于 clamp 校准输出)
    class_concentration_ranges: Dict[int, tuple] = field(default_factory=lambda: {
        0: (12.5, 125.0),
        1: (25.0, 250.0),
        2: (12.5, 125.0),
        3: (25.0, 250.0),
    })

    # QC 参数
    risk_score_names: List[str] = field(default_factory=lambda: [
        "composite_response_risk",
        "route_response_risk",
        "class_response_margin_risk",
        "class_response_rank_risk",
        "response_signature_norm",
        "response_conc_gap_norm",
        "response_mean_conc_gap_norm",
        "classifier_uncertainty",
        "margin_risk",
    ])
    low_ratio: float = 0.90
    high_ratio: float = 1.10
    high_error_ppm: float = 40.0

    def to_dict(self) -> Dict[str, Any]:
        """将配置序列化为字典"""
        return {
            "classifier_checkpoint": self.classifier_checkpoint,
            "regression_checkpoint": self.regression_checkpoint,
            "full_model_checkpoint": self.full_model_checkpoint,
            "calibration_params_path": self.calibration_params_path,
            "routing_config_path": self.routing_config_path,
            "qc_policy_path": self.qc_policy_path,
            "qc_thresholds_path": self.qc_thresholds_path,
            "calibration_ref_dir": self.calibration_ref_dir,
            "device": self.device,
            "model_config": self.model_config,
            "gas_names": self.gas_names,
            "num_classes": self.num_classes,
            "num_phases": self.num_phases,
            "class_concentration_ranges": {
                str(k): list(v) for k, v in self.class_concentration_ranges.items()
            },
            "risk_score_names": self.risk_score_names,
            "low_ratio": self.low_ratio,
            "high_ratio": self.high_ratio,
            "high_error_ppm": self.high_error_ppm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployConfig":
        """从字典加载配置"""
        config = cls()
        for key, value in data.items():
            if key == "class_concentration_ranges":
                config.class_concentration_ranges = {
                    int(k): tuple(v) for k, v in value.items()
                }
            elif hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_json(cls, path: str) -> "DeployConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        """保存配置到 JSON 文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def validate(self) -> List[str]:
        """验证配置完整性，返回缺失项列表"""
        errors: List[str] = []
        if not self.classifier_checkpoint or not Path(self.classifier_checkpoint).exists():
            errors.append(f"分类模型 checkpoint 不存在: {self.classifier_checkpoint}")
        if not self.regression_checkpoint or not Path(self.regression_checkpoint).exists():
            errors.append(f"回归模型 checkpoint 不存在: {self.regression_checkpoint}")
        if self.full_model_checkpoint and not Path(self.full_model_checkpoint).exists():
            errors.append(f"full 校准模型 checkpoint 不存在: {self.full_model_checkpoint}")
        if self.calibration_params_path and not Path(self.calibration_params_path).exists():
            errors.append(f"校准参数文件不存在: {self.calibration_params_path}")
        if self.qc_policy_path and not Path(self.qc_policy_path).exists():
            errors.append(f"QC 策略文件不存在: {self.qc_policy_path}")
        return errors
