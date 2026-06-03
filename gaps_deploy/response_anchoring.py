"""
路由感知响应锚定模块

实现回归推理中的路由选择逻辑，根据分类特征和回归特征
选择最优的回归头 (head) 进行浓度预测。

路由模式:
    1. hard_route (硬路由):
       - 根据分类 logits 最大值选择唯一回归头
       - 简单高效，适合分类精度高的场景

    2. soft_route (软路由):
       - 根据 softmax 概率对各回归头输出加权平均
       - 当分类置信度低时，融合多个回归头的预测
       - 温度参数 τ 控制 softmax 的平滑程度

    3. response_anchored_route (响应锚定路由):
       - 结合分类置信度和响应签名相似度进行路由
       - 当预测响应与校准集响应差异大时，降低该类别权重

核心类:
    - RoutePredictor: 路由预测器
    - ResponseAnchoring: 响应锚定修正

使用示例:
    >>> router = RoutePredictor(route_mode="hard")
    >>> head_id = router.route(logits, cls_feat, reg_feat)
    >>> pred_ppm = model.reg_heads[head_id](reg_feat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 各类别默认浓度范围 (ppm)
DEFAULT_CONC_RANGES: Dict[int, Tuple[float, float]] = {
    0: (12.5, 125.0),
    1: (25.0, 250.0),
    2: (12.5, 125.0),
    3: (25.0, 250.0),
}


@dataclass
class RouteResult:
    """路由结果

    属性:
        head_id: 选择的回归头 ID (hard route)
        weights: 各回归头的权重 (soft route)
        confidence: 路由置信度 (max softmax)
        is_soft: 是否使用软路由
        clamped: 是否对输出进行了 clamp
    """
    head_id: int = 0
    weights: Optional[np.ndarray] = None
    confidence: float = 1.0
    is_soft: bool = False
    clamped: bool = False


class RoutePredictor:
    """路由预测器

    根据分类模型输出选择最优回归头。

    路由策略:
        - hard: 选择 logits 最大值对应的回归头
        - soft_topk: 对 top-k 个类别按 softmax 权重加权
        - response_anchored: 结合响应签名相似度调整权重

    属性:
        route_mode: 路由模式 ("hard" | "soft_topk" | "response_anchored")
        soft_topk: 软路由 top-k 值
        soft_temperature: 软路由温度参数 τ
        min_confidence: 触发软路由的最小置信度阈值
        max_margin: 触发软路由的最大边缘阈值
        num_classes: 类别数
    """

    def __init__(
        self,
        route_mode: str = "hard",
        soft_topk: int = 2,
        soft_temperature: float = 1.0,
        min_confidence: float = 0.95,
        max_margin: float = 0.10,
        num_classes: int = 4,
    ):
        """
        参数:
            route_mode: 路由模式
            soft_topk: 软路由时考虑的 top-k 类别
            soft_temperature: softmax 温度参数 (τ > 1 平滑, τ < 1 锐化)
            min_confidence: 低于此置信度触发软路由
            max_margin: 当 top1-top2 边缘小于此值时触发软路由
            num_classes: 类别数
        """
        self.route_mode = route_mode
        self.soft_topk = soft_topk
        self.soft_temperature = soft_temperature
        self.min_confidence = min_confidence
        self.max_margin = max_margin
        self.num_classes = num_classes

    def route(
        self,
        logits: np.ndarray,
        cls_feat: Optional[np.ndarray] = None,
        reg_feat: Optional[np.ndarray] = None,
    ) -> RouteResult:
        """选择最优回归头

        参数:
            logits: 分类 logits (num_classes,)
            cls_feat: 分类特征向量 (可选)
            reg_feat: 回归特征向量 (可选)

        返回:
            RouteResult 路由结果
        """
        logits = np.asarray(logits, dtype=np.float64)
        probs = self._softmax(logits)

        top1 = float(np.max(probs))
        top2 = float(np.sort(probs)[-2]) if len(probs) > 1 else 0.0
        margin = top1 - top2
        head_id = int(np.argmax(probs))

        # 判断是否需要软路由
        use_soft = (
            self.route_mode in ("soft_topk", "response_anchored")
            and (top1 < self.min_confidence or margin < self.max_margin)
        )

        if not use_soft:
            return RouteResult(
                head_id=head_id,
                confidence=top1,
                is_soft=False,
            )

        # 软路由: 对 top-k 类别按 softmax 权重加权
        if self.soft_temperature > 0 and abs(self.soft_temperature - 1.0) > 1e-8:
            # 温度缩放
            scaled_logits = logits / self.soft_temperature
            soft_probs = self._softmax(scaled_logits)
        else:
            soft_probs = probs

        # 只保留 top-k
        topk_indices = np.argsort(soft_probs)[-self.soft_topk:]
        weights = np.zeros(self.num_classes)
        for idx in topk_indices:
            weights[idx] = soft_probs[idx]
        weights = weights / (weights.sum() + 1e-10)

        return RouteResult(
            head_id=head_id,
            weights=weights,
            confidence=top1,
            is_soft=True,
        )

    def route_batch(
        self,
        logits_batch: np.ndarray,
        cls_feat_batch: Optional[np.ndarray] = None,
        reg_feat_batch: Optional[np.ndarray] = None,
    ) -> List[RouteResult]:
        """批量路由

        参数:
            logits_batch: (N, num_classes) 分类 logits

        返回:
            RouteResult 列表
        """
        results = []
        for i in range(len(logits_batch)):
            cls_feat = cls_feat_batch[i] if cls_feat_batch is not None else None
            reg_feat = reg_feat_batch[i] if reg_feat_batch is not None else None
            results.append(self.route(logits_batch[i], cls_feat, reg_feat))
        return results

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """稳定版 softmax"""
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / (exp.sum() + 1e-10)


class ResponseAnchoring:
    """响应锚定修正

    当预测响应的签名与校准集签名差异较大时，降低该类别路由权重，
    并将预测浓度向校准集参考浓度锚定。

    锚定公式:
        corrected = (1 - α) * pred_ppm + α * ref_ppm
        其中 α = min(1, response_norm / anchor_scale)

    这确保了在传感器漂移严重时，预测不会偏离校准集太远。

    使用示例:
        >>> anchoring = ResponseAnchoring()
        >>> corrected = anchoring.anchor(
        ...     pred_ppm=185.0,
        ...     class_id=1,
        ...     response_norm=2.5,
        ...     calib_refs=calib_refs,
        ... )
    """

    def __init__(
        self,
        anchor_scale: float = 3.0,
        anchor_alpha: float = 0.3,
        enable_anchoring: bool = True,
    ):
        """
        参数:
            anchor_scale: 响应标准化距离的缩放因子
            anchor_alpha: 基础锚定强度
            enable_anchoring: 是否启用锚定
        """
        self.anchor_scale = anchor_scale
        self.anchor_alpha = anchor_alpha
        self.enable_anchoring = enable_anchoring

    def anchor(
        self,
        pred_ppm: float,
        class_id: int,
        response_norm: float,
        calib_ref_conc: Optional[float] = None,
        conc_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> float:
        """对预测浓度进行响应锚定修正

        参数:
            pred_ppm: 模型预测浓度 (ppm)
            class_id: 预测类别
            response_norm: 响应标准化距离 (response_signature_norm)
            calib_ref_conc: 校准集参考浓度 (最近邻校准样本的浓度)
            conc_ranges: 各类别浓度范围

        返回:
            锚定后浓度 (ppm)
        """
        if not self.enable_anchoring or calib_ref_conc is None:
            return pred_ppm

        # α = min(1, response_norm / anchor_scale)
        alpha = min(1.0, abs(response_norm) / max(self.anchor_scale, 1e-8))
        alpha = alpha * self.anchor_alpha

        corrected = (1.0 - alpha) * pred_ppm + alpha * calib_ref_conc

        # clamp 到合理范围
        if conc_ranges and class_id in conc_ranges:
            lo, hi = conc_ranges[class_id]
            corrected = min(max(corrected, lo), hi)

        return float(corrected)

    def anchor_batch(
        self,
        pred_ppm_batch: np.ndarray,
        class_ids: np.ndarray,
        response_norms: np.ndarray,
        calib_ref_concs: Optional[np.ndarray] = None,
        conc_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """批量响应锚定

        参数:
            pred_ppm_batch: (N,) 预测浓度
            class_ids: (N,) 预测类别
            response_norms: (N,) 响应标准化距离
            calib_ref_concs: (N,) 校准集参考浓度

        返回:
            (N,) 锚定后浓度
        """
        result = np.array(pred_ppm_batch, dtype=np.float64)
        if calib_ref_concs is None:
            return result

        for i in range(len(result)):
            result[i] = self.anchor(
                result[i],
                int(class_ids[i]),
                float(response_norms[i]),
                float(calib_ref_concs[i]) if calib_ref_concs is not None else None,
                conc_ranges,
            )
        return result