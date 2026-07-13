"""R4a residual-calibration artifact loading and inference-time application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

CONC_RANGES = {
    0: (12.5, 125.0),
    1: (25.0, 250.0),
    2: (12.5, 125.0),
    3: (25.0, 250.0),
}


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _client_matches(artifact_client: str, runtime_client: str) -> bool:
    expected = str(artifact_client or "").strip().upper()
    actual = str(runtime_client or "").strip().upper()
    if not expected:
        return True
    if expected == actual:
        return True
    expected_num = expected[1:] if expected.startswith("C") else expected
    actual_num = actual[1:] if actual.startswith("C") else actual
    return expected_num == actual_num


def _ppm_norm(ppm: float, cls_id: int) -> float:
    lo, hi = CONC_RANGES.get(int(cls_id), (0.0, 1.0))
    return float((ppm - lo) / max(hi - lo, 1e-12))


def _clamp_ppm(ppm: float, cls_id: int) -> float:
    lo, hi = CONC_RANGES.get(int(cls_id), (ppm, ppm))
    return float(np.clip(ppm, lo, hi))


def _dct_low_k(window: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(window, dtype=np.float64)
    n = arr.shape[0]
    basis = np.cos(np.pi / n * (np.arange(n, dtype=np.float64) + 0.5)[:, None] * np.arange(k, dtype=np.float64)[None, :])
    coeff = basis.T @ arr / max(n, 1)
    return coeff.T.reshape(-1)


def _window_stats(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float64)
    t = np.linspace(-0.5, 0.5, arr.shape[0], dtype=np.float64)
    denom = float(np.sum(t**2))
    slope = (t[:, None] * arr).sum(axis=0) / max(denom, 1e-12)
    return np.concatenate([
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
        arr[-1] - arr[0],
        slope,
    ])


def _predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return design @ coef


def _risk_lookup(row: Dict[str, Any], key: str) -> float:
    value = _to_float(row.get(key))
    if np.isfinite(value):
        return value
    if key.startswith("risk_"):
        return _to_float(row.get(key[len("risk_"):]), 0.0)
    return 0.0


def _row_features(row: Dict[str, Any], window: np.ndarray, target_class: int, dct_k: int) -> np.ndarray:
    pred_ppm = _to_float(row.get("calibrated_ppm"))
    pred_cls = _to_int(row.get("pred_class"), target_class)
    phase = _to_int(row.get("phase"), -1)
    phase_onehot = np.zeros(3, dtype=np.float64)
    if 0 <= phase < 3:
        phase_onehot[phase] = 1.0

    scalar_keys = [
        "confidence",
        "top1_confidence",
        "top2_confidence",
        "confidence_margin",
        "risk_score",
        "risk_classifier_uncertainty",
        "risk_margin_risk",
        "risk_response_signature_norm",
        "risk_response_conc_gap_norm",
        "risk_class_response_rank_risk",
        "risk_class_response_margin_risk",
        "risk_route_response_risk",
        "risk_composite_response_risk",
    ]
    scalars = np.asarray([_risk_lookup(row, key) for key in scalar_keys], dtype=np.float64)
    base = np.asarray([
        pred_ppm,
        _ppm_norm(pred_ppm, pred_cls),
        _ppm_norm(pred_ppm, target_class),
    ], dtype=np.float64)
    return np.concatenate([base, phase_onehot, scalars, _window_stats(window), _dct_low_k(window, dct_k)])


class R4AResidualArtifact:
    """One target-class residual calibrator."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.enabled = bool(data.get("enabled", False))
        self.client_id = str(data.get("client_id", "") or "")
        self.target_class = int(data.get("target_class", -1))
        self.dct_k = int(data.get("dct_k", 0))
        self.shrink_alpha = float(data.get("shrink_alpha", 0.0))
        self.mean = np.asarray(data.get("mean", []), dtype=np.float64)
        self.std = np.asarray(data.get("std", []), dtype=np.float64)
        self.coef = np.asarray(data.get("coef", []), dtype=np.float64)
        self.gate = data.get("application_gate", {}) or {}

    @classmethod
    def from_json(cls, path: Path) -> "R4AResidualArtifact":
        with path.open("r", encoding="utf-8-sig") as f:
            return cls(json.load(f))

    def can_apply(self, row: Dict[str, Any], client_id: str) -> bool:
        if not self.enabled:
            return False
        if not _client_matches(self.client_id, client_id):
            return False
        if _to_int(row.get("pred_class")) != self.target_class:
            return False
        statuses = {
            str(item).strip().lower()
            for item in self.gate.get("apply_statuses", [])
            if str(item).strip()
        }
        if statuses and str(row.get("qc_status", "")).strip().lower() not in statuses:
            return False
        max_risk = self.gate.get("max_apply_risk")
        if max_risk is not None:
            risk = _to_float(row.get("risk_composite_response_risk"), _to_float(row.get("risk_score")))
            if not np.isfinite(risk) or risk > float(max_risk):
                return False
        return True

    def apply(self, row: Dict[str, Any], window: np.ndarray, client_id: str) -> Tuple[float, float, bool]:
        original = _to_float(row.get("calibrated_ppm"))
        if not self.can_apply(row, client_id):
            return original, 0.0, False
        feat = _row_features(row, window, self.target_class, self.dct_k)
        delta = float(_predict_ridge(((feat - self.mean) / self.std).reshape(1, -1), self.coef)[0])
        correction = self.shrink_alpha * delta
        max_abs = self.gate.get("max_abs_correction")
        if max_abs is not None:
            correction = float(np.clip(correction, -float(max_abs), float(max_abs)))
        corrected = _clamp_ppm(original + correction, self.target_class)
        return corrected, delta, True


class R4AArtifactSet:
    """Collection of optional R4a artifacts loaded from a package."""

    def __init__(self, artifacts: Sequence[R4AResidualArtifact] | None = None):
        self.artifacts = list(artifacts or [])

    @classmethod
    def from_dir(cls, path: Path) -> "R4AArtifactSet":
        if not path.exists() or not path.is_dir():
            return cls()
        artifacts: List[R4AResidualArtifact] = []
        for item in sorted(path.glob("*.json")):
            if item.name.lower().endswith("manifest.json") or item.name.lower() == "r4a_manifest.json":
                continue
            artifact = R4AResidualArtifact.from_json(item)
            if artifact.enabled:
                artifacts.append(artifact)
        return cls(artifacts)

    def apply(
        self,
        calibrated_ppm: float,
        pred_class: int,
        qc_status: str,
        feature_window: np.ndarray,
        client_id: str,
        phase: int = -1,
        confidence: float = 0.0,
        top1_confidence: float = 0.0,
        top2_confidence: float = 0.0,
        confidence_margin: float = 0.0,
        risk_score: float = 0.0,
        risk_scores: Dict[str, float] | None = None,
    ) -> Tuple[float, float, float, bool, int]:
        row: Dict[str, Any] = {
            "calibrated_ppm": calibrated_ppm,
            "pred_class": pred_class,
            "qc_status": qc_status,
            "phase": phase,
            "confidence": confidence,
            "top1_confidence": top1_confidence,
            "top2_confidence": top2_confidence,
            "confidence_margin": confidence_margin,
            "risk_score": risk_score,
        }
        for key, value in (risk_scores or {}).items():
            row[key] = value
            row[f"risk_{key}"] = value

        for artifact in self.artifacts:
            corrected, raw_delta, applied = artifact.apply(row, feature_window, client_id)
            if applied:
                return corrected, corrected - calibrated_ppm, raw_delta, True, artifact.target_class
        return calibrated_ppm, 0.0, 0.0, False, -1

    def __len__(self) -> int:
        return len(self.artifacts)
