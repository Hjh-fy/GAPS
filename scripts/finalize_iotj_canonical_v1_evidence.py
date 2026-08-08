"""Close canonical-v1 submission evidence without changing trained models.

The module consumes immutable canonical-v1 prediction artifacts.  All policy
constants below are fixed before the post-run summaries are generated.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from run_regression_head_ablation import CLASS_RANGES


QUALITY_STRATA_POLICY = {
    "Q0": "observed=1, empty=0, max_missing_run=0, interpolated=0",
    "Q1": "observed>=0.98, empty<=0.02, max_missing_run<=1, interpolated<=0.02",
    "Q2": "observed>=0.90, empty<=0.10, max_missing_run<=3, interpolated<=0.10",
    "Q3": "otherwise",
}


def assign_quality_stratum(
    *,
    observed_ratio: float,
    empty_bin_ratio: float,
    max_missing_run: int,
    short_gap_interpolated_ratio: float,
) -> str:
    """Apply the predeclared acquisition-only Q0-Q3 thresholds."""
    observed = float(observed_ratio)
    empty = float(empty_bin_ratio)
    missing = int(max_missing_run)
    interpolated = float(short_gap_interpolated_ratio)
    values = np.asarray([observed, empty, interpolated], dtype=np.float64)
    if not np.isfinite(values).all() or missing < 0:
        raise ValueError("invalid acquisition quality metadata")
    if observed == 1.0 and empty == 0.0 and missing == 0 and interpolated == 0.0:
        return "Q0"
    if observed >= 0.98 and empty <= 0.02 and missing <= 1 and interpolated <= 0.02:
        return "Q1"
    if observed >= 0.90 and empty <= 0.10 and missing <= 3 and interpolated <= 0.10:
        return "Q2"
    return "Q3"


def classify_qc_decision(
    risk: float, *, accept_threshold: float, review_threshold: float
) -> str:
    """Preserve the frozen accept set and split its complement into review/reject."""
    accept = float(accept_threshold)
    review = float(review_threshold)
    if review < accept:
        raise ValueError("review threshold must be >= accept threshold")
    value = float(risk)
    if value <= accept:
        return "accepted"
    if value <= review:
        return "review"
    return "reject"


def summarize_error_scope(
    rows: Sequence[Mapping[str, Any]], prediction_key: str = "pred_84d_h1_ppm"
) -> dict[str, Any]:
    """Return the submission metrics for a non-empty regression population."""
    if not rows:
        raise RuntimeError("FAIL_CLOSED empty regression scope")
    truth = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    classes = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise RuntimeError("FAIL_CLOSED non-finite regression scope")
    error = prediction - truth
    ranges = np.asarray([float(CLASS_RANGES[int(value)]) for value in classes])
    centered = truth - float(np.mean(truth))
    total = float(np.sum(centered**2))
    return {
        "N": int(len(rows)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "NRMSE_range": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "P90AE": float(np.quantile(np.abs(error), 0.90, method="higher")),
        "Bias": float(np.mean(error)),
        "R2": float(1.0 - np.sum(error**2) / total) if total > 0 else float("nan"),
    }


def interval_overlap_seconds(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> float:
    """Return intersection length of two half-open physical-time intervals."""
    if float(end_a) < float(start_a) or float(end_b) < float(start_b):
        raise ValueError("interval end precedes start")
    return float(max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b))))


def engineering_claims(
    *, points_old: int, points_new: int, channels: int, parameter_bytes: int
) -> dict[str, Any]:
    """Separate temporal input reduction from parameter communication payload."""
    if min(points_old, points_new, channels, parameter_bytes) <= 0:
        raise ValueError("engineering dimensions/bytes must be positive")
    old_bytes = int(points_old) * int(channels) * 4
    new_bytes = int(points_new) * int(channels) * 4
    return {
        "legacy_points_per_window": int(points_old),
        "canonical_points_per_window": int(points_new),
        "legacy_input_tensor_bytes_fp32": old_bytes,
        "canonical_input_tensor_bytes_fp32": new_bytes,
        "temporal_input_reduction": 1.0 - float(points_new) / float(points_old),
        "parameter_bytes": int(parameter_bytes),
        "parameter_communication_reduction": 0.0,
    }

