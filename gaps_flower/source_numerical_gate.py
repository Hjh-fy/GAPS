"""Fixed source-only numerical validity gate for canonical SCAFFOLD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SourceGateVerdict:
    passed: bool
    checks: dict[str, bool]
    diagnostics: dict[str, float | int | list[float]]
    action: str
    lr_search_performed: bool = False
    target_information_accessed: bool = False


def _finite(values: Sequence[float]) -> bool:
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def evaluate_source_gate(
    client_diagnostics: Sequence[Mapping[str, Sequence[float]]],
    *,
    source_accuracy: float,
    source_class_counts: Mapping[int, int],
) -> SourceGateVerdict:
    """Evaluate preregistered C1/C2-only optimization checks without tuning."""
    trajectories = [list(row.get("ce_trajectory", ())) for row in client_diagnostics]
    grad_norms = [
        float(value)
        for row in client_diagnostics
        for value in row.get("grad_norms", ())
    ]
    parameter_norms = [
        float(value)
        for row in client_diagnostics
        for value in row.get("parameter_norms", ())
    ]
    ce_values = [float(value) for row in trajectories for value in row]
    counts = [int(value) for value in source_class_counts.values()]
    total = sum(counts)
    majority_prior = max(counts) / total if total > 0 and counts else 1.0

    all_finite = (
        _finite(ce_values)
        and _finite(grad_norms)
        and _finite(parameter_norms)
        and math.isfinite(float(source_accuracy))
    )
    per_client_decrease: list[bool] = []
    first_means: list[float] = []
    final_means: list[float] = []
    for trajectory in trajectories:
        if not trajectory:
            per_client_decrease.append(False)
            continue
        quarter = max(1, len(trajectory) // 4)
        first = sum(trajectory[:quarter]) / quarter
        final = sum(trajectory[-quarter:]) / quarter
        first_means.append(float(first))
        final_means.append(float(final))
        per_client_decrease.append(
            math.isfinite(first) and math.isfinite(final) and final < first
        )
    checks = {
        "all_finite": all_finite,
        "ce_decreased": bool(per_client_decrease) and all(per_client_decrease),
        "gradient_norm_valid": _finite(grad_norms)
        and min(grad_norms) > 0.0
        and max(grad_norms) < 1e4,
        "parameter_norm_valid": _finite(parameter_norms)
        and min(parameter_norms) > 0.0
        and max(parameter_norms) < 1e4,
        "source_discrimination": math.isfinite(float(source_accuracy))
        and float(source_accuracy) > majority_prior,
        "source_only": True,
        "no_lr_search": True,
    }
    passed = all(checks.values())
    return SourceGateVerdict(
        passed=passed,
        checks=checks,
        diagnostics={
            "source_accuracy": float(source_accuracy),
            "majority_class_prior": float(majority_prior),
            "client_first_quarter_ce": first_means,
            "client_final_quarter_ce": final_means,
            "max_gradient_norm": max(grad_norms) if grad_norms else float("nan"),
            "max_parameter_norm": (
                max(parameter_norms) if parameter_norms else float("nan")
            ),
            "client_count": len(client_diagnostics),
        },
        action="PROCEED_FIXED_CONFIGURATION" if passed else "FAIL_CLOSED_NO_LR_SEARCH",
    )
