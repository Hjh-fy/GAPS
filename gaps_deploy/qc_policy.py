"""
QC 质量控制层模块

实现部署推理中的质量控制和风险筛选功能，将窗口分为:
    - accept (自动接收): 低风险，直接输出校准后浓度
    - review (复核): 中等风险，建议人工复核
    - reject (拒绝): 高风险，自动拒绝输出

核心流程:
    1. 计算风险分数 (RiskScoreComputer)
    2. 双阈值决策 (TwoThresholdDecider)
    3. 策略选择与 guardrails 验证 (PolicySelector)

风险分数类型:
    - classifier_uncertainty: 1 - max_softmax，分类器不确定性
    - margin_risk: 1 - confidence_margin，置信度边缘风险
    - response_signature_norm: 响应向量与校准签名的标准化距离
    - response_conc_gap_norm: 浓度预测与校准响应的标准化差距
    - class_response_rank_risk: 类别响应排序风险
    - class_response_margin_risk: 类别响应边缘风险
    - route_response_risk: 路由响应综合风险
    - composite_response_risk: 综合风险 (max of all above)

数学原理:
    风险比 (risk_ratio) = max(score_i / threshold_i)
    当 risk_ratio ≤ low_ratio → accept
    当 risk_ratio > high_ratio → reject
    否则 → review

    low_ratio 和 high_ratio 从部署验证集校准得到，典型值:
        low_ratio ∈ [0.80, 0.95]
        high_ratio ∈ [1.00, 1.20]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QCPolicy:
    """单条 QC 策略

    属性:
        policy_name: 策略名称 (如 "multirisk_v3", "default_t8")
        scores: 使用的风险分数名称列表
        thresholds: {score_name: threshold_value} 阈值映射
        low_ratio: 双阈值下界 (≤ low_ratio → accept)
        high_ratio: 双阈值上界 (> high_ratio → reject)
        group: 策略适用的客户端组 ("ALL" 或 "C1", "C3" 等)
    """
    policy_name: str = "default"
    scores: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    low_ratio: float = 0.90
    high_ratio: float = 1.10
    group: str = "ALL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "scores": self.scores,
            "thresholds": self.thresholds,
            "low_ratio": self.low_ratio,
            "high_ratio": self.high_ratio,
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QCPolicy":
        return cls(
            policy_name=data.get("policy_name", "default"),
            scores=data.get("scores", []),
            thresholds=data.get("thresholds", {}),
            low_ratio=data.get("low_ratio", 0.90),
            high_ratio=data.get("high_ratio", 1.10),
            group=data.get("group", "ALL"),
        )


@dataclass
class QCDecision:
    """单窗口 QC 决策结果

    属性:
        decision: "accept" | "review" | "reject"
        risk_ratio: 最大风险比 (max(score_i / threshold_i))
        risk_scores: {score_name: value} 各风险分数值
        risk_reasons: 触发拒绝的风险分数名称列表
        policy_name: 使用的策略名称
    """
    decision: str = "accept"
    risk_ratio: float = 0.0
    risk_scores: Dict[str, float] = field(default_factory=dict)
    risk_reasons: List[str] = field(default_factory=list)
    policy_name: str = ""


class RiskScoreComputer:
    """风险分数计算器

    从部署推理结果中计算各类风险分数，用于 QC 决策。

    核心风险分数计算逻辑 (从 evaluate_single_window_reliability.py 移植):
        1. classifier_uncertainty = 1 - max_softmax
        2. margin_risk = 1 - (top1_conf - top2_conf)
        3. response_signature_norm = sig_dist / loocv_p90
        4. response_conc_gap_norm = |pred_ppm - nearest_calib_conc| / 25.0
        5. class_response_rank_risk = rank - 1
        6. class_response_margin_risk = max(0, pred_norm - best_norm)
        7. route_response_risk = max(rank_risk, margin_risk, 10 * uncertainty)
        8. composite_response_risk = max of all above

    使用示例:
        >>> computer = RiskScoreComputer()
        >>> scores = computer.compute(
        ...     logits=logits_tensor,
        ...     pred_ppm=pred_ppm,
        ...     class_id=class_id,
        ...     calib_refs=calib_refs,
        ...     features=features,
        ... )
    """

    # 默认风险分数列表
    DEFAULT_SCORES = [
        "classifier_uncertainty",
        "margin_risk",
        "response_signature_norm",
        "response_conc_gap_norm",
        "response_mean_conc_gap_norm",
        "class_response_rank_risk",
        "class_response_margin_risk",
        "route_response_risk",
        "composite_response_risk",
    ]

    def __init__(self, calib_refs: Optional[Dict[int, Any]] = None):
        """
        参数:
            calib_refs: 校准集参考数据，格式:
                {class_id: {
                    "center": ndarray,      # 响应签名中心
                    "scale": ndarray,        # 响应签名 scale
                    "z_sigs": ndarray,       # 标准化签名
                    "loocv_p90": float,      # LOOCV P90 距离
                    "rows": [...]           # 校准集样本
                }}
        """
        self.calib_refs = calib_refs or {}

    def compute(
        self,
        logits: np.ndarray,
        pred_ppm: float,
        class_id: int,
        features: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """计算单窗口的全部风险分数

        参数:
            logits: 分类 logits (num_classes,) 或 softmax 概率
            pred_ppm: 模型预测浓度 (ppm)
            class_id: 预测类别 ID
            features: 原始特征 (T, C)，用于响应签名比较
            extra_info: 额外信息 (nearest_calib_conc, confidence_margin 等)

        返回:
            {score_name: value} 风险分数字典
        """
        scores: Dict[str, float] = {}

        # 1. 分类器不确定性: 1 - max(softmax)
        if logits is not None:
            probs = self._softmax(logits)
            top1 = float(np.max(probs))
            top2 = float(np.sort(probs)[-2]) if len(probs) > 1 else 0.0
            scores["classifier_uncertainty"] = float(1.0 - top1)
            scores["margin_risk"] = float(1.0 - (top1 - top2))
        else:
            scores["classifier_uncertainty"] = 0.0
            scores["margin_risk"] = 0.0

        # 2. 响应签名风险 (需要 features 和 calib_refs)
        if features is not None and self.calib_refs:
            self._compute_response_scores(scores, features, pred_ppm, class_id)
        else:
            for key in [
                "response_signature_norm",
                "response_conc_gap_norm",
                "response_mean_conc_gap_norm",
                "class_response_rank_risk",
                "class_response_margin_risk",
                "class_response_rank",
                "class_response_margin",
                "best_response_class",
                "best_response_norm",
                "pred_response_norm",
            ]:
                if key not in scores:
                    scores[key] = 0.0

        # 3. 从 extra_info 补充
        if extra_info:
            for key in ["response_signature_norm", "response_conc_gap_norm",
                        "response_mean_conc_gap_norm", "class_response_rank_risk",
                        "class_response_margin_risk"]:
                if key in extra_info and key not in scores:
                    scores[key] = float(extra_info.get(key, 0.0))

        # 4. 路由响应风险: max(rank_risk, margin_risk, 10 * uncertainty)
        scores["route_response_risk"] = float(np.nanmax([
            scores.get("class_response_rank_risk", 0.0),
            scores.get("class_response_margin_risk", 0.0),
            scores.get("classifier_uncertainty", 0.0) * 10.0,
        ]))

        # 5. 综合风险: max of all
        scores["composite_response_risk"] = float(np.nanmax([
            scores.get("response_signature_norm", 0.0),
            scores.get("response_conc_gap_norm", 0.0),
            scores.get("response_mean_conc_gap_norm", 0.0),
            scores.get("class_response_rank_risk", 0.0),
            scores.get("class_response_margin_risk", 0.0),
            scores.get("classifier_uncertainty", 0.0) * 10.0,
        ]))

        return scores

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """稳定版 softmax"""
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / (exp.sum() + 1e-10)

    def _compute_response_scores(
        self,
        scores: Dict[str, float],
        features: np.ndarray,
        pred_ppm: float,
        class_id: int,
    ) -> None:
        """计算响应签名相关风险分数"""
        pred_ref = self.calib_refs.get(class_id)
        pred_item = None
        ranked_items: List[Dict[str, float]] = []
        ranking_enabled = self._response_ranking_enabled()

        for ref_cls, ref in self.calib_refs.items():
            try:
                ref_cls_int = int(ref_cls)
            except (TypeError, ValueError):
                continue
            nearest = self._nearest_response(features, ref)
            if nearest is None:
                continue
            item = {
                "class": float(ref_cls_int),
                "dist": nearest["dist"],
                "norm": nearest["norm"],
                "nearest_idx": float(nearest["nearest_idx"]),
                "nearest_conc": nearest["nearest_conc"],
            }
            ranked_items.append(item)
            if ref_cls_int == int(class_id):
                pred_item = item

        ranked_items.sort(key=lambda item: item["norm"])

        if pred_item is not None:
            scores["response_signature_norm"] = float(pred_item["norm"])
            nearest_conc = pred_item["nearest_conc"]
            if np.isfinite(nearest_conc):
                scores["response_conc_gap_norm"] = float(abs(pred_ppm - nearest_conc) / 25.0)
                scores["nearest_calib_conc"] = float(nearest_conc)
            else:
                scores["response_conc_gap_norm"] = 0.0
                scores["nearest_calib_conc"] = float("nan")
            scores["nearest_calib_idx"] = float(pred_item["nearest_idx"])
        elif pred_ref is not None:
            scores["response_signature_norm"] = 0.0
            scores["response_conc_gap_norm"] = 0.0
        else:
            scores["response_signature_norm"] = 0.0
            scores["response_conc_gap_norm"] = 0.0

        if ranking_enabled and ranked_items and pred_item is not None:
            classes = [int(item["class"]) for item in ranked_items]
            rank = 1 + classes.index(int(class_id))
            best = ranked_items[0]
            pred_norm = float(pred_item["norm"])
            best_norm = float(best["norm"])
            margin = pred_norm - best_norm
            scores["best_response_class"] = float(best["class"])
            scores["best_response_norm"] = best_norm
            scores["pred_response_norm"] = pred_norm
            scores["class_response_rank"] = float(rank)
            scores["class_response_rank_risk"] = float(rank - 1)
            scores["class_response_margin"] = float(margin)
            scores["class_response_margin_risk"] = float(max(0.0, margin))
            scores["best_response_nearest_calib_conc"] = float(best["nearest_conc"])
        else:
            scores["best_response_class"] = -1.0
            scores["best_response_norm"] = 0.0
            scores["pred_response_norm"] = 0.0
            scores["class_response_rank"] = 0.0
            scores["class_response_rank_risk"] = 0.0
            scores["class_response_margin"] = 0.0
            scores["class_response_margin_risk"] = 0.0

    def _response_ranking_enabled(self) -> bool:
        """Return whether class-wise response ranking should affect QC.

        Old v1 packages used 8-D time-mean response references and were tuned
        with rank/margin risks effectively disabled. Keep those packages stable
        unless the calibration artifact explicitly opts in, or unless it uses
        the new 40-D response descriptor generated for QC v2.
        """
        for ref in self.calib_refs.values():
            if not isinstance(ref, dict):
                continue
            if bool(ref.get("response_ranking_enabled", False)):
                return True
            signature = str(ref.get("signature", "")).lower()
            signature_dim = int(ref.get("signature_dim", self._ref_dim(ref)) or 0)
            if signature == "mean_std_amp_slope_noise" or signature_dim >= 40:
                return True
        return False

    @staticmethod
    def _response_signature(features: np.ndarray, target_dim: int) -> Optional[np.ndarray]:
        """Return a response signature compatible with the reference dimension.

        Existing deployment packages store an 8-D time-mean signature. Newer
        packages store a 40-D descriptor: mean, std, amplitude, slope, and
        short-term noise for each sensor channel.
        """
        feat = np.asarray(features, dtype=np.float64)
        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat[0]
        if feat.ndim != 2:
            flat = feat.ravel()
            return flat if flat.size == target_dim else None

        if target_dim == feat.shape[1]:
            return feat.mean(axis=0)

        mean_ch = feat.mean(axis=0)
        std_ch = feat.std(axis=0)
        amp_ch = feat.max(axis=0) - feat.min(axis=0)
        edge = max(1, int(round(feat.shape[0] * 0.10)))
        slope_ch = feat[-edge:, :].mean(axis=0) - feat[:edge, :].mean(axis=0)
        diff = np.diff(feat, axis=0)
        noise_ch = diff.std(axis=0) if diff.size else np.zeros_like(mean_ch)
        descriptor = np.concatenate([mean_ch, std_ch, amp_ch, slope_ch, noise_ch], axis=0)
        if descriptor.size == target_dim:
            return descriptor
        if mean_ch.size == target_dim:
            return mean_ch
        return None

    @staticmethod
    def _ref_dim(ref: Dict[str, Any]) -> int:
        center = np.asarray(ref.get("center", []), dtype=np.float64)
        if center.ndim == 1 and center.size > 0:
            return int(center.size)
        z_sigs = np.asarray(ref.get("z_sigs", []), dtype=np.float64)
        if z_sigs.ndim == 2 and z_sigs.shape[1] > 0:
            return int(z_sigs.shape[1])
        return 0

    def _nearest_response(self, features: np.ndarray, ref: Dict[str, Any]) -> Optional[Dict[str, float]]:
        target_dim = self._ref_dim(ref)
        if target_dim <= 0:
            return None
        sig = self._response_signature(features, target_dim)
        if sig is None:
            return None

        center = np.asarray(ref.get("center", np.zeros(target_dim)), dtype=np.float64).reshape(-1)
        scale = np.asarray(ref.get("scale", np.ones(target_dim)), dtype=np.float64).reshape(-1)
        if center.size != target_dim or scale.size != target_dim or sig.size != target_dim:
            return None
        scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
        z_sig = (sig - center) / scale

        z_sigs = np.asarray(ref.get("z_sigs", []), dtype=np.float64)
        if z_sigs.ndim != 2 or z_sigs.shape[1] != target_dim or z_sigs.shape[0] == 0:
            return None

        dists = np.linalg.norm(z_sigs - z_sig.reshape(1, -1), axis=1)
        if dists.size > 1:
            zero_mask = dists <= 1e-12
            if np.any(zero_mask) and np.any(~zero_mask):
                dists = np.where(zero_mask, np.inf, dists)
        nearest_idx = int(np.argmin(dists))
        sig_dist = float(dists[nearest_idx])
        loocv_p90 = float(ref.get("loocv_p90", 1.0))
        if not np.isfinite(loocv_p90) or loocv_p90 < 1e-8:
            loocv_p90 = 1.0

        rows = ref.get("rows", [])
        nearest_conc = float("nan")
        if rows and nearest_idx < len(rows):
            try:
                nearest_conc = float(rows[nearest_idx].get("concentration", np.nan))
            except (TypeError, ValueError):
                nearest_conc = float("nan")

        return {
            "dist": sig_dist,
            "norm": float(sig_dist / loocv_p90),
            "nearest_idx": float(nearest_idx),
            "nearest_conc": nearest_conc,
        }

    def compute_batch(
        self,
        logits_batch: np.ndarray,
        pred_ppm_batch: np.ndarray,
        class_ids: np.ndarray,
        features_batch: Optional[np.ndarray] = None,
        extra_info_batch: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, float]]:
        """批量计算风险分数

        参数:
            logits_batch: (N, num_classes) 分类 logits
            pred_ppm_batch: (N,) 预测浓度
            class_ids: (N,) 预测类别
            features_batch: (N, T, C) 原始特征
            extra_info_batch: 额外信息列表

        返回:
            [{score: value}, ...] 每个窗口的风险分数
        """
        results = []
        for i in range(len(pred_ppm_batch)):
            features = features_batch[i] if features_batch is not None else None
            extra = extra_info_batch[i] if extra_info_batch else None
            results.append(
                self.compute(
                    logits=logits_batch[i],
                    pred_ppm=float(pred_ppm_batch[i]),
                    class_id=int(class_ids[i]),
                    features=features,
                    extra_info=extra,
                )
            )
        return results


class TwoThresholdDecider:
    """双阈值决策器

    根据风险分数和阈值，将窗口分为 accept / review / reject。

    决策逻辑:
        risk_ratio = max(score_i / threshold_i for i in scores)
        if risk_ratio ≤ low_ratio:
            → accept (自动输出浓度)
        elif risk_ratio > high_ratio:
            → reject (拒绝，不输出浓度)
        else:
            → review (建议人工复核)

    使用示例:
        >>> decider = TwoThresholdDecider()
        >>> decider.load_policy(policy)
        >>> decision = decider.decide(risk_scores, client_id="C3")
        >>> print(decision.decision, decision.risk_ratio)
    """

    def __init__(self):
        # 策略映射: {group: QCPolicy}
        self.policies: Dict[str, QCPolicy] = {}
        self.default_policy: Optional[QCPolicy] = None

    def load_policy(self, policy: QCPolicy) -> None:
        """加载一条 QC 策略

        参数:
            policy: QCPolicy 对象
        """
        self.policies[policy.group] = policy
        if policy.group == "ALL":
            self.default_policy = policy

    def load_policies_json(self, path: str) -> None:
        """从 JSON 文件加载多条策略"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        policies_data = data if isinstance(data, list) else data.get("policies", [data])
        for policy_data in policies_data:
            self.load_policy(QCPolicy.from_dict(policy_data))

    def get_policy(self, client_id: str = "ALL") -> Optional[QCPolicy]:
        """获取指定客户端对应的策略"""
        return self.policies.get(client_id) or self.default_policy

    def decide(
        self,
        risk_scores: Dict[str, float],
        client_id: str = "ALL",
    ) -> QCDecision:
        """对单窗口进行双阈值决策

        参数:
            risk_scores: {score_name: value} 风险分数
            client_id: 客户端标识 ("C1", "C3" 等)

        返回:
            QCDecision 决策结果
        """
        policy = self.get_policy(client_id)
        if policy is None:
            return QCDecision(
                decision="accept",
                risk_ratio=0.0,
                risk_scores=risk_scores,
                risk_reasons=[],
                policy_name="no_policy",
            )

        # 计算风险比: max(score_i / threshold_i)
        max_ratio = 0.0
        risk_reasons: List[str] = []
        for score_name in policy.scores:
            value = risk_scores.get(score_name, 0.0)
            threshold = policy.thresholds.get(score_name, np.inf)
            if np.isfinite(value) and np.isfinite(threshold) and abs(threshold) > 1e-12:
                ratio = value / threshold
                if ratio > max_ratio:
                    max_ratio = ratio
                if ratio > policy.high_ratio:
                    risk_reasons.append(score_name)

        # 双阈值决策
        if max_ratio <= policy.low_ratio:
            decision = "accept"
        elif max_ratio > policy.high_ratio:
            decision = "reject"
        else:
            decision = "review"

        return QCDecision(
            decision=decision,
            risk_ratio=float(max_ratio),
            risk_scores=risk_scores,
            risk_reasons=risk_reasons,
            policy_name=policy.policy_name,
        )

    def decide_batch(
        self,
        risk_scores_batch: List[Dict[str, float]],
        client_ids: Optional[List[str]] = None,
    ) -> List[QCDecision]:
        """批量双阈值决策

        参数:
            risk_scores_batch: 风险分数列表
            client_ids: 客户端 ID 列表 (与 risk_scores_batch 一一对应)

        返回:
            QCDecision 列表
        """
        if client_ids is None:
            client_ids = ["ALL"] * len(risk_scores_batch)
        return [
            self.decide(scores, cid)
            for scores, cid in zip(risk_scores_batch, client_ids)
        ]


class PolicySelector:
    """QC 策略选择器

    从候选策略中选择满足 guardrails 约束的最优策略。
    对比 baseline vs candidate，确保:
        - coverage (接受率) ≥ baseline_coverage - δ_coverage
        - accepted_P90AE ≤ baseline_P90AE + δ_P90
        - high_error_recall ≥ baseline_recall + δ_recall
        - accepted_high_error_rate ≤ baseline_high_error_rate + δ_high_rate

    使用示例:
        >>> selector = PolicySelector()
        >>> best = selector.select(candidates, baseline, constraints)
        >>> print(best.policy_name)
    """

    @dataclass
    class GuardrailConstraints:
        """Guardrail 约束条件"""
        min_coverage_delta: float = -0.05  # 允许覆盖率下降最多 5%
        max_p90_delta: float = 5.0         # 允许 P90AE 增加最多 5 ppm
        min_recall_delta: float = 0.0      # 要求高误差召回率至少保持
        max_high_rate_delta: float = 0.02  # 允许接受窗口高误差率增加最多 2%

    @dataclass
    class PolicyMetrics:
        """策略评估指标"""
        policy_name: str = ""
        coverage: float = 0.0
        accepted_P90AE: float = 0.0
        high_error_recall: float = 0.0
        accepted_high_error_rate: float = 0.0
        accepted_MAE: float = 0.0
        accepted_R2: float = 0.0

    def select(
        self,
        candidates: List[PolicyMetrics],
        baseline: PolicyMetrics,
        constraints: Optional[GuardrailConstraints] = None,
    ) -> Optional[PolicyMetrics]:
        """从候选策略中选择满足约束的最优策略

        参数:
            candidates: 候选策略列表
            baseline: 基线策略
            constraints: 约束条件

        返回:
            最优策略，若无满足约束的策略则返回 None
        """
        if constraints is None:
            constraints = self.GuardrailConstraints()

        valid = []
        for cand in candidates:
            if self._pass_guardrail(cand, baseline, constraints):
                valid.append(cand)

        if not valid:
            logger.warning("策略选择: 无候选策略满足 guardrail 约束")
            return None

        # 选择 P90AE 最低的策略
        valid.sort(key=lambda c: (c.accepted_P90AE, -c.coverage))
        best = valid[0]
        logger.info(
            f"策略选择: 最优策略={best.policy_name}, "
            f"coverage={best.coverage:.3f}, P90AE={best.accepted_P90AE:.1f}"
        )
        return best

    def _pass_guardrail(
        self,
        cand: PolicyMetrics,
        base: PolicyMetrics,
        c: GuardrailConstraints,
    ) -> bool:
        """检查候选策略是否满足 guardrail 约束"""
        return (
            cand.coverage >= base.coverage + c.min_coverage_delta
            and cand.accepted_P90AE <= base.accepted_P90AE + c.max_p90_delta
            and cand.high_error_recall >= base.high_error_recall + c.min_recall_delta
            and cand.accepted_high_error_rate <= base.accepted_high_error_rate + c.max_high_rate_delta
        )
