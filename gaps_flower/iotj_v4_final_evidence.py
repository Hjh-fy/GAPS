"""Pure, fail-closed evidence contracts for the IoT-J v4 analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import numpy as np

from gaps_flower.canonical_r1_v1 import (
    bootstrap_paired_group_deltas,
    summarize_bootstrap,
)


BUDGET_COUNTS = {20: 320, 15: 240, 10: 160, 5: 80}
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 5000

_METHODS = frozenset({"R83_TARGET_ONLY", "R84_CONCAT"})


def pooled_classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute one classification metric after merging all target predictions."""
    if not rows:
        raise ValueError("classification rows must be non-empty")
    required_targets = {"C3", "C4", "C5"}
    targets = {row.get("target") for row in rows}
    if targets != required_targets:
        raise ValueError(
            "target population must contain exactly non-empty C3, C4, and C5 rows"
        )
    try:
        truth = np.asarray([row["true_class"] for row in rows])
        prediction = np.asarray([row["predicted_class"] for row in rows])
    except (KeyError, TypeError) as exc:
        raise ValueError("classification rows require true_class and predicted_class") from exc
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("classification labels must be aligned one-dimensional values")
    try:
        finite = np.isfinite(truth.astype(float)).all() and np.isfinite(prediction.astype(float)).all()
    except (TypeError, ValueError) as exc:
        raise ValueError("classification labels must be finite numeric values") from exc
    if not finite:
        raise ValueError("classification labels must be finite numeric values")

    labels = np.unique(np.concatenate((truth, prediction)))
    f1_values = []
    for label in labels:
        true_positive = int(np.sum((truth == label) & (prediction == label)))
        false_positive = int(np.sum((truth != label) & (prediction == label)))
        false_negative = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append((2.0 * true_positive / denominator) if denominator else 0.0)
    return {
        "aggregation": "merged_predictions_global_metric",
        "N": int(len(truth)),
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(np.mean(f1_values)),
    }


def _identity(row: Mapping[str, Any], *, source: str) -> str:
    value = str(row.get("physical_identity", "")).strip()
    if not value:
        raise ValueError(f"{source} row has missing physical_identity")
    return value


def _reject_nonfinite(row: Mapping[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, Number) and not isinstance(value, (bool, np.bool_)):
            if not np.isfinite(value):
                raise ValueError(f"non-finite value for {key}")
        elif isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if not np.isfinite(parsed):
                raise ValueError(f"non-finite value for {key}")


def _semantic_label(row: Mapping[str, Any], *, source: str) -> float:
    aliases = [key for key in ("class_id", "true_class", "label") if key in row]
    if not aliases:
        raise ValueError(f"{source} row has no label alias")
    normalized: list[float] = []
    for key in aliases:
        value = row[key]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{source} row has invalid label")
        try:
            label = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} row has invalid label") from exc
        if not np.isfinite(label):
            raise ValueError(f"{source} row has non-finite label")
        normalized.append(label)
    if any(label != normalized[0] for label in normalized[1:]):
        raise ValueError(f"{source} row has conflicting label aliases")
    return normalized[0]


def match_budget_identities(
    budget_rows: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate budget evidence and return rows in stable canonical identity order."""
    canonical: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(canonical_rows):
        identity = _identity(row, source="canonical")
        if identity in canonical:
            raise ValueError(f"duplicate canonical physical_identity: {identity}")
        _reject_nonfinite(row)
        canonical[identity] = (index, row)
        _semantic_label(row, source="canonical")
    if not canonical:
        raise ValueError("canonical rows must be non-empty")

    seen: set[tuple[int, str, str]] = set()
    validated: list[tuple[int, int, str, dict[str, Any]]] = []
    for row in budget_rows:
        identity = _identity(row, source="budget")
        if identity not in canonical:
            raise ValueError(f"missing canonical identity: {identity}")
        try:
            budget = int(row["budget_pct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("budget row has invalid budget") from exc
        if budget not in BUDGET_COUNTS or row.get("budget_pct") != budget:
            raise ValueError(f"unexpected budget: {row.get('budget_pct')!r}")
        method = str(row.get("method", ""))
        if method not in _METHODS:
            raise ValueError(f"unexpected method: {method!r}")
        key = (budget, method, identity)
        if key in seen:
            raise ValueError(f"duplicate budget physical_identity: {identity}")
        seen.add(key)
        _reject_nonfinite(row)
        canonical_row = canonical[identity][1]
        if _semantic_label(row, source="budget") != _semantic_label(
            canonical_row, source="canonical"
        ):
            raise ValueError(f"reordered label for physical_identity: {identity}")
        validated.append((canonical[identity][0], budget, method, dict(row)))
    if not validated:
        raise ValueError("budget rows must be non-empty")
    present_identities = {item[3]["physical_identity"] for item in validated}
    missing_identities = sorted(set(canonical) - present_identities)
    if missing_identities:
        raise ValueError(f"missing budget identities: {missing_identities}")
    by_budget: dict[int, dict[str, set[str]]] = {}
    for _, budget, method, row in validated:
        by_budget.setdefault(budget, {}).setdefault(method, set()).add(
            row["physical_identity"]
        )
    for budget, method_rows in by_budget.items():
        if set(method_rows) != _METHODS:
            raise ValueError(f"budget {budget} must contain exactly both methods")
        if method_rows["R83_TARGET_ONLY"] != method_rows["R84_CONCAT"]:
            raise ValueError(f"budget {budget} method identity sets differ")
    validated.sort(key=lambda item: (item[1], item[2], item[0]))
    return [row for _, _, _, row in validated]


def _paired_points(truth, pred_r83, pred_r84, gas_range) -> dict[str, float]:
    y, first, second, ranges = map(
        lambda value: np.asarray(value, dtype=np.float64),
        (truth, pred_r83, pred_r84, gas_range),
    )
    error_first = first - y
    error_second = second - y
    return {
        "RMSE": float(np.sqrt(np.mean(error_second**2)) - np.sqrt(np.mean(error_first**2))),
        "MAE": float(np.mean(np.abs(error_second)) - np.mean(np.abs(error_first))),
        "NRMSE_range": float(
            np.sqrt(np.mean((error_second / ranges) ** 2))
            - np.sqrt(np.mean((error_first / ranges) ** 2))
        ),
    }


def paired_grouped_interval(truth, pred_r83, pred_r84, gas_range, groups):
    raw_groups = np.asarray(groups)
    numeric = [
        np.asarray(value, dtype=np.float64)
        for value in (truth, pred_r83, pred_r84, gas_range)
    ]
    arrays = [*numeric, raw_groups]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("paired bootstrap inputs must be one-dimensional")
    if not arrays[0].size or any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("paired bootstrap inputs must have identical non-empty shape")
    for value in raw_groups.tolist():
        if value is None or (
            isinstance(value, Number)
            and not isinstance(value, (bool, np.bool_))
            and not np.isfinite(value)
        ):
            raise RuntimeError("FAIL_CLOSED bootstrap grouping key is missing")
    groups_array = raw_groups.astype(str)
    serialized_nulls = {"none", "null", "nan", "nat"}
    for value in groups_array:
        if not value.startswith("C5|") or value.count("|") != 1:
            raise RuntimeError("FAIL_CLOSED bootstrap grouping key must be C5|<filename>")
        suffix = value.removeprefix("C5|")
        if (
            not suffix
            or suffix != suffix.strip()
            or suffix.casefold() in serialized_nulls
        ):
            raise RuntimeError("FAIL_CLOSED bootstrap grouping key has invalid suffix")
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("paired bootstrap inputs contain non-finite values")
    if np.any(numeric[3] <= 0):
        raise ValueError("gas_range must be positive")
    output = bootstrap_paired_group_deltas(
        numeric[0],
        numeric[1],
        numeric[2],
        groups_array,
        numeric[3],
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    return summarize_bootstrap(
        "C5",
        output,
        point=_paired_points(*numeric),
        n_groups=len(np.unique(groups_array)),
        n_replicates=BOOTSTRAP_REPLICATES,
    )


def classify_provenance(checks: Mapping[str, bool]) -> dict[str, Any]:
    """Classify named provenance checks without truthy-value coercion."""
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("checks must be a non-empty mapping")
    if any(not isinstance(name, str) or not name.strip() for name in checks):
        raise ValueError("checks require non-empty string names")
    if any(type(value) is not bool for value in checks.values()):
        raise ValueError("provenance checks must contain boolean values")
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "classification": "FAIL_CLOSED" if failed else "VERIFIED",
        "passed": not failed,
        "failed_checks": failed,
    }
