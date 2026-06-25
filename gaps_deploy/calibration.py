"""
回归校准层模块

实现目标域回归模型的校准逻辑，支持以下校准模式:
    - none: 不校准，直接使用源域回归模型输出
    - bias_only: 仅校准偏置项 (y = y_pred + b)
    - affine_only: 仿射校准 (y = a * y_pred + b)
    - phase_affine_only: 按阶段 (早期/中期/晚期) 分别校准
    - full: 完整微调模型 (需要加载 full_model checkpoint)
    - specialist: 单类专家模型 (需要加载 specialist checkpoint)

校准参数通过最小二乘法在 ppm 原始浓度空间拟合，拟合在 oracle 模式
(使用真实类别选择回归头) 下进行，确保校准不引入路由误差。

核心类:
    - RegressionCalibrator: 回归校准器，管理所有校准模式
    - AffineCalibrator: 仿射/偏置校准拟合与预测
    - PhaseAffineCalibrator: 按阶段仿射校准
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 气体名称映射
GAS_NAMES = ["Ethanol", "CO", "Ethylene", "Methane"]

# 各类别默认浓度范围 (ppm)，用于 clamp 校准输出防止异常值
DEFAULT_CONC_RANGES: Dict[int, Tuple[float, float]] = {
    0: (12.5, 125.0),
    1: (25.0, 250.0),
    2: (12.5, 125.0),
    3: (25.0, 250.0),
}


class AffineCalibrator:
    """单类仿射/偏置校准器

    在 ppm 原始浓度空间拟合:
        bias_only:  y_true = y_pred + b
        affine_only: y_true = a * y_pred + b

    拟合使用最小二乘法 (np.linalg.lstsq)，确保数值稳定。
    当预测方差接近零时，affine 自动退化为 bias_only。

    属性:
        a: 缩放系数 (affine) 或 1.0 (bias)
        b: 偏置项
        n_samples: 拟合样本数
        calib_r2: 校准拟合 R²
        calib_mae: 校准拟合 MAE
        mode: 校准模式 ("bias_only" | "affine_only")
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, mode: str = "affine_only"):
        self.a = a
        self.b = b
        self.n_samples: int = 0
        self.calib_r2: float = 0.0
        self.calib_mae: float = 0.0
        self.mode = mode

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "AffineCalibrator":
        """拟合校准参数

        参数:
            y_true: 真实浓度 (N,) 形状
            y_pred: 模型预测浓度 (N,) 形状

        返回:
            self，支持链式调用
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        self.n_samples = len(y_true)

        if self.n_samples < 2:
            self.a, self.b = 1.0, 0.0
            return self

        residual = y_true - y_pred

        if self.mode == "bias_only":
            # 只拟合偏置: y_true = y_pred + b
            self.a = 1.0
            self.b = float(np.mean(residual))
        else:
            # affine: y_true = a * y_pred + b
            y_pred_var = np.var(y_pred)
            if y_pred_var < 1e-12:
                # 预测方差接近零，退化到 bias_only
                self.a = 1.0
                self.b = float(np.mean(residual))
            else:
                A = np.column_stack([y_pred, np.ones_like(y_pred)])
                coeffs, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
                self.a, self.b = float(coeffs[0]), float(coeffs[1])

        # 计算校准后 R² 和 MAE
        y_adj = self.a * y_pred + self.b
        ss_res = np.sum((y_true - y_adj) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        self.calib_r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
        self.calib_mae = float(np.mean(np.abs(y_true - y_adj)))

        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """对预测值应用校准

        参数:
            y_pred: 模型预测浓度 (N,) 形状

        返回:
            (N,) 校准后浓度
        """
        return self.a * np.asarray(y_pred) + self.b

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "a": self.a,
            "b": self.b,
            "n_samples": self.n_samples,
            "calib_r2": self.calib_r2,
            "calib_mae": self.calib_mae,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffineCalibrator":
        """从字典加载"""
        calib = cls(a=data.get("a", 1.0), b=data.get("b", 0.0), mode=data.get("mode", "affine_only"))
        calib.n_samples = data.get("n_samples", 0)
        calib.calib_r2 = data.get("calib_r2", 0.0)
        calib.calib_mae = data.get("calib_mae", 0.0)
        return calib


class PhaseAffineCalibrator:
    """按阶段 (早期/中期/晚期) 仿射校准器

    对每个阶段 (phase) 独立拟合 affine 参数，适用于不同阶段
    传感器响应特性差异较大的场景。

    属性:
        phase_calibrators: {phase_id: AffineCalibrator} 映射
        num_phases: 阶段总数
    """

    def __init__(self, num_phases: int = 3):
        self.phase_calibrators: Dict[int, AffineCalibrator] = {}
        self.num_phases = num_phases

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_phase: np.ndarray,
        mode: str = "affine_only",
    ) -> "PhaseAffineCalibrator":
        """按阶段拟合校准参数

        参数:
            y_true: 真实浓度 (N,)
            y_pred: 模型预测浓度 (N,)
            y_phase: 阶段标签 (N,)，0=早期, 1=中期, 2=晚期
            mode: 校准模式 ("bias_only" | "affine_only")

        返回:
            self
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_phase = np.asarray(y_phase, dtype=int)

        for phase in range(self.num_phases):
            mask = y_phase == phase
            if mask.sum() >= 2:
                calib = AffineCalibrator(mode=mode)
                calib.fit(y_true[mask], y_pred[mask])
                self.phase_calibrators[phase] = calib
            else:
                # 样本不足，使用 identity
                self.phase_calibrators[phase] = AffineCalibrator(a=1.0, b=0.0, mode=mode)

        return self

    def predict(self, y_pred: np.ndarray, y_phase: np.ndarray) -> np.ndarray:
        """按阶段应用校准

        参数:
            y_pred: 模型预测浓度 (N,)
            y_phase: 阶段标签 (N,)

        返回:
            (N,) 校准后浓度
        """
        y_pred = np.asarray(y_pred)
        y_phase = np.asarray(y_phase, dtype=int)
        result = np.zeros_like(y_pred)
        for phase in range(self.num_phases):
            mask = y_phase == phase
            if mask.any() and phase in self.phase_calibrators:
                result[mask] = self.phase_calibrators[phase].predict(y_pred[mask])
            elif mask.any():
                result[mask] = y_pred[mask]
        return result

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "num_phases": self.num_phases,
            "phase_calibrators": {
                str(phase): calib.to_dict()
                for phase, calib in self.phase_calibrators.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseAffineCalibrator":
        """从字典加载"""
        calib = cls(num_phases=data.get("num_phases", 3))
        for phase_str, calib_data in data.get("phase_calibrators", {}).items():
            calib.phase_calibrators[int(phase_str)] = AffineCalibrator.from_dict(calib_data)
        return calib


class RegressionCalibrator:
    """回归校准总控器

    管理 per-class + per-mode 的校准参数，支持:
        - none: 不校准
        - bias_only: 偏置校准
        - affine_only: 仿射校准
        - phase_affine_only: 按阶段仿射校准
        - full: 使用完整微调模型 (通过 routing_config 指定)
        - specialist: 使用单类专家模型 (通过 routing_config 指定)

    路由配置 (routing_config) 格式:
        {
            "selected_modes": {0: "affine_only", 1: "full", 2: "none", 3: "bias_only"},
            "affine_params": {0: {"a": 1.05, "b": -2.3}, ...},
            "phase_affine_params": {0: {"num_phases": 3, "phase_calibrators": {...}}, ...},
        }

    使用示例:
        >>> calib = RegressionCalibrator(num_classes=4)
        >>> calib.fit_affine(train_loader, model, classifier_model, device)
        >>> calib.load_routing_config("routing_config.json")
        >>> calibrated = calib.calibrate(pred_ppm, class_ids, phase_ids)
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_phases: int = 3,
        conc_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        """
        参数:
            num_classes: 分类类别数
            num_phases: 阶段数
            conc_ranges: 各类别浓度范围 (用于 clamp)，默认使用 DEFAULT_CONC_RANGES
        """
        self.num_classes = num_classes
        self.num_phases = num_phases
        self.conc_ranges = conc_ranges or DEFAULT_CONC_RANGES

        # per-class 校准参数
        self.affine_params: Dict[int, AffineCalibrator] = {}
        self.bias_params: Dict[int, AffineCalibrator] = {}
        self.phase_affine_params: Dict[int, PhaseAffineCalibrator] = {}

        # 路由配置: {class_id: mode}
        self.selected_modes: Dict[int, str] = {
            c: "none" for c in range(num_classes)
        }

    def fit_affine(
        self,
        class_id: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mode: str = "affine_only",
    ) -> AffineCalibrator:
        """为指定类别拟合仿射/偏置校准参数

        参数:
            class_id: 类别 ID
            y_true: 真实浓度 (N,)
            y_pred: 模型预测浓度 (N,)
            mode: "affine_only" | "bias_only"

        返回:
            拟合后的 AffineCalibrator
        """
        calib = AffineCalibrator(mode=mode)
        calib.fit(y_true, y_pred)
        if mode == "bias_only":
            self.bias_params[class_id] = calib
        else:
            self.affine_params[class_id] = calib
        return calib

    def fit_phase_affine(
        self,
        class_id: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_phase: np.ndarray,
        mode: str = "affine_only",
    ) -> PhaseAffineCalibrator:
        """为指定类别拟合按阶段校准参数

        参数:
            class_id: 类别 ID
            y_true: 真实浓度 (N,)
            y_pred: 模型预测浓度 (N,)
            y_phase: 阶段标签 (N,)
            mode: "affine_only" | "bias_only"

        返回:
            拟合后的 PhaseAffineCalibrator
        """
        calib = PhaseAffineCalibrator(num_phases=self.num_phases)
        calib.fit(y_true, y_pred, y_phase, mode=mode)
        self.phase_affine_params[class_id] = calib
        return calib

    def calibrate(
        self,
        pred_ppm: np.ndarray,
        class_ids: np.ndarray,
        phase_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """根据路由配置对预测浓度应用校准

        参数:
            pred_ppm: 模型预测浓度 (N,)
            class_ids: 预测类别 ID (N,)
            phase_ids: 阶段 ID (N,)，phase_affine 模式需要

        返回:
            (N,) 校准后浓度，自动 clamp 到合理范围
        """
        pred_ppm = np.asarray(pred_ppm, dtype=np.float64)
        class_ids = np.asarray(class_ids, dtype=int)
        if phase_ids is not None:
            phase_ids = np.asarray(phase_ids, dtype=int)
        else:
            phase_ids = np.zeros_like(class_ids)

        calibrated = pred_ppm.copy()

        for c in range(self.num_classes):
            mask = class_ids == c
            if not mask.any():
                continue

            mode = self.selected_modes.get(c, "none")

            if mode == "bias_only" and c in self.bias_params:
                calibrated[mask] = self.bias_params[c].predict(pred_ppm[mask])
            elif mode == "affine_only" and c in self.affine_params:
                calibrated[mask] = self.affine_params[c].predict(pred_ppm[mask])
            elif mode == "phase_affine_only" and c in self.phase_affine_params:
                calibrated[mask] = self.phase_affine_params[c].predict(
                    pred_ppm[mask], phase_ids[mask]
                )

        # 按类别 clamp 到合理浓度范围，防止校准后异常值
        for c in range(self.num_classes):
            mask = class_ids == c
            if mask.any() and c in self.conc_ranges:
                lo, hi = self.conc_ranges[c]
                calibrated[mask] = np.clip(calibrated[mask], lo, hi)

        return calibrated

    def load_routing_config(self, routing_config: Dict[str, Any]) -> None:
        """从字典加载路由配置

        参数:
            routing_config: 路由配置字典，
                {"selected_modes": {0: "affine_only", ...}, "affine_params": {...}, ...}
        """
        # 加载 per-class 模式选择
        for c_str, mode in routing_config.get("selected_modes", {}).items():
            c = int(c_str)
            self.selected_modes[c] = mode

        # 加载 affine 参数
        for c_str, params in routing_config.get("affine_params", {}).items():
            c = int(c_str)
            if isinstance(params, dict):
                calib = AffineCalibrator.from_dict(params)
                if calib.mode == "bias_only" or self.selected_modes.get(c) == "bias_only":
                    self.bias_params[c] = calib
                else:
                    self.affine_params[c] = calib

        # 加载 phase_affine 参数
        for c_str, params in routing_config.get("phase_affine_params", {}).items():
            c = int(c_str)
            if isinstance(params, dict):
                self.phase_affine_params[c] = PhaseAffineCalibrator.from_dict(params)

    def load_routing_config_json(self, path: str) -> None:
        """从 JSON 文件加载路由配置"""
        with open(path, "r", encoding="utf-8") as f:
            self.load_routing_config(json.load(f))

    def save_routing_config_json(self, path: str) -> None:
        """保存路由配置到 JSON 文件"""
        data = {
            "selected_modes": {str(k): v for k, v in self.selected_modes.items()},
            "affine_params": {
                str(k): v.to_dict() for k, v in self.affine_params.items()
            },
            "phase_affine_params": {
                str(k): v.to_dict() for k, v in self.phase_affine_params.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def fit_all_from_loaders(
        self,
        data_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        classifier_model: nn.Module,
        device: torch.device,
    ) -> None:
        """从 DataLoader 采集 oracle 预测，一次性拟合所有类别的校准参数

        对每个类别分别拟合 bias_only 和 affine_only 两种模式，
        后续通过 validation 选择最优模式。

        参数:
            data_loader: 校准数据加载器
            model: 回归模型 (需有 forward_reg 方法)
            classifier_model: 分类模型 (确认标签)
            device: 推理设备
        """
        model.eval()
        classifier_model.eval()

        # 按类别收集预测值
        stores: Dict[int, Dict[str, List[float]]] = {
            c: {"true": [], "pred": [], "phase": []}
            for c in range(self.num_classes)
        }

        with torch.no_grad():
            for x, y_cls, y_reg_full, y_phase in data_loader:
                x = x.to(device)
                y_cls = y_cls.to(device)
                y_reg_full = y_reg_full.to(device)
                y_phase = y_phase.to(device)

                _, _, reg_feat = model(x)
                # oracle 模式：使用真实类别选择回归头
                pred_norm = model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
                y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]

                # 反归一化到 ppm 空间 (简化版，实际需要 _denormalize_norm_by_class)
                pred_ppm = pred_norm.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()
                y_cls_np = y_cls.detach().cpu().numpy()
                y_phase_np = y_phase.detach().cpu().numpy()

                for c in range(self.num_classes):
                    mask = y_cls_np == c
                    if mask.any():
                        stores[c]["true"].extend(y_true_np[mask].tolist())
                        stores[c]["pred"].extend(pred_ppm[mask].tolist())
                        stores[c]["phase"].extend(y_phase_np[mask].tolist())

        for c in range(self.num_classes):
            if len(stores[c]["true"]) >= 2:
                y_true = np.array(stores[c]["true"])
                y_pred = np.array(stores[c]["pred"])
                y_phase = np.array(stores[c]["phase"], dtype=int)

                self.fit_affine(c, y_true, y_pred, mode="bias_only")
                self.fit_affine(c, y_true, y_pred, mode="affine_only")
                self.fit_phase_affine(c, y_true, y_pred, y_phase, mode="affine_only")

                logger.info(
                    f"  Class {c}: bias R²={self.bias_params[c].calib_r2:.4f}, "
                    f"affine R²={self.affine_params[c].calib_r2:.4f}"
                )
