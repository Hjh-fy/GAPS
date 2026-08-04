"""Build a deployment QC policy from prediction CSV risk columns.

This script converts an offline risk sweep decision into the JSON policy format
loaded by :class:`gaps_deploy.qc_policy.TwoThresholdDecider`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _policy_score_name(score_column: str) -> str:
    if score_column.startswith("risk_") and score_column != "risk_score":
        return score_column[len("risk_"):]
    if score_column == "risk_score":
        raise ValueError(
            "`risk_score` is the post-policy ratio written to CSV, not a raw "
            "runtime risk score. Use a raw column such as "
            "`risk_composite_response_risk`."
        )
    return score_column


def _parse_client_rates(raw: str) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for item in _split_csv(raw):
        if ":" not in item:
            raise ValueError(f"Invalid client rate item: {item!r}; expected C5:0.15")
        client_id, value = item.split(":", 1)
        rate = float(value)
        if not 0.0 < rate < 1.0:
            raise ValueError(f"Review rate must be in (0, 1), got {item!r}")
        rates[client_id.strip()] = rate
    return rates


def _parse_client_floats(raw: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for item in _split_csv(raw):
        if ":" not in item:
            raise ValueError(f"Invalid client value item: {item!r}; expected C5:2.0")
        client_id, value = item.split(":", 1)
        parsed = float(value)
        if parsed <= 0.0:
            raise ValueError(f"Client value must be positive, got {item!r}")
        values[client_id.strip()] = parsed
    return values


def _assign_groups(paths: Sequence[Path], groups_raw: str) -> Dict[Path, str]:
    groups = _split_csv(groups_raw)
    if groups and len(groups) != len(paths):
        raise ValueError(
            f"--groups length ({len(groups)}) must match --inputs length ({len(paths)})"
        )
    return {path: groups[i] if groups else "" for i, path in enumerate(paths)}


def _load_grouped_scores(
    paths: Sequence[Path],
    path_groups: Dict[Path, str],
    score_column: str,
) -> Tuple[Dict[str, List[float]], List[float]]:
    by_group: Dict[str, List[float]] = {}
    all_values: List[float] = []
    for path in paths:
        rows = _read_rows(path)
        if rows and score_column not in rows[0]:
            raise KeyError(f"{path} does not contain score column {score_column!r}")
        fallback_group = path_groups.get(path, "")
        for row in rows:
            value = _to_float(row.get(score_column))
            if not np.isfinite(value):
                continue
            group = fallback_group or row.get("client_id", "ALL") or "ALL"
            by_group.setdefault(group, []).append(float(value))
            all_values.append(float(value))
    if not all_values:
        raise ValueError(f"No finite values found for {score_column!r}")
    return by_group, all_values


def _threshold_for_review_rate(values: Iterable[float], review_rate: float) -> Tuple[float, int, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot build a threshold from zero finite scores")
    if not 0.0 < review_rate < 1.0:
        raise ValueError(f"review_rate must be in (0, 1), got {review_rate}")

    sorted_desc = np.sort(arr)[::-1]
    n_flag = int(np.ceil(arr.size * review_rate))
    n_flag = min(max(n_flag, 1), int(arr.size))
    kth = float(sorted_desc[n_flag - 1])
    if abs(kth) <= 1e-12:
        threshold = 1e-12
    else:
        threshold = float(np.nextafter(kth, -np.inf))
    actual_rate = float(np.mean(arr > threshold))
    return threshold, n_flag, actual_rate


def _make_policy(
    group: str,
    score_name: str,
    threshold: float,
    review_rate: float,
    low_ratio: float,
    high_ratio: float,
) -> Dict[str, Any]:
    pct = int(round(review_rate * 100))
    return {
        "policy_name": f"quantile_{score_name}_top{pct}_{group}",
        "group": group,
        "scores": [score_name],
        "thresholds": {score_name: float(threshold)},
        "low_ratio": float(low_ratio),
        "high_ratio": float(high_ratio),
    }


def build_policy(
    inputs: Sequence[str],
    groups: str,
    score_column: str,
    global_review_rate: float,
    client_review_rates: str,
    low_ratio: float,
    high_ratio: float,
    threshold_scale: float = 1.0,
    client_threshold_scales: str = "",
) -> Dict[str, Any]:
    if threshold_scale <= 0.0:
        raise ValueError(f"threshold_scale must be positive, got {threshold_scale}")
    paths = [Path(path) for path in inputs]
    score_name = _policy_score_name(score_column)
    by_group, all_values = _load_grouped_scores(
        paths,
        _assign_groups(paths, groups),
        score_column,
    )

    policies: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []

    threshold, n_flag, actual_rate = _threshold_for_review_rate(
        all_values,
        global_review_rate,
    )
    raw_threshold = threshold
    threshold = threshold * float(threshold_scale)
    policies.append(
        _make_policy(
            group="ALL",
            score_name=score_name,
            threshold=threshold,
            review_rate=global_review_rate,
            low_ratio=low_ratio,
            high_ratio=high_ratio,
        )
    )
    reports.append({
        "group": "ALL",
        "n": len(all_values),
        "target_review_rate": global_review_rate,
        "target_review_count": n_flag,
        "raw_threshold": raw_threshold,
        "threshold_scale": float(threshold_scale),
        "threshold": threshold,
        "estimated_review_rate": float(np.mean(np.asarray(all_values) > threshold)),
        "raw_estimated_review_rate": actual_rate,
    })

    client_scales = _parse_client_floats(client_threshold_scales)
    for group, review_rate in _parse_client_rates(client_review_rates).items():
        if group not in by_group:
            raise KeyError(f"No rows found for client/group {group!r}")
        threshold, n_flag, actual_rate = _threshold_for_review_rate(
            by_group[group],
            review_rate,
        )
        raw_threshold = threshold
        scale = float(client_scales.get(group, threshold_scale))
        threshold = threshold * scale
        policies.append(
            _make_policy(
                group=group,
                score_name=score_name,
                threshold=threshold,
                review_rate=review_rate,
                low_ratio=low_ratio,
                high_ratio=high_ratio,
            )
        )
        reports.append({
            "group": group,
            "n": len(by_group[group]),
            "target_review_rate": review_rate,
            "target_review_count": n_flag,
            "raw_threshold": raw_threshold,
            "threshold_scale": scale,
            "threshold": threshold,
            "estimated_review_rate": float(np.mean(np.asarray(by_group[group]) > threshold)),
            "raw_estimated_review_rate": actual_rate,
        })

    return {
        "metadata": {
            "score_column": score_column,
            "runtime_score": score_name,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
            "threshold_scale": threshold_scale,
            "client_threshold_scales": client_scales,
            "inputs": [str(path) for path in paths],
            "reports": reports,
        },
        "policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build selected_policy.json from deployment prediction CSVs."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument("--score-column", default="risk_composite_response_risk")
    parser.add_argument("--global-review-rate", type=float, default=0.10)
    parser.add_argument("--client-review-rates", default="")
    parser.add_argument("--low-ratio", type=float, default=1.0)
    parser.add_argument("--high-ratio", type=float, default=999.0)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--client-threshold-scales", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy = build_policy(
        inputs=args.inputs,
        groups=args.groups,
        score_column=args.score_column,
        global_review_rate=args.global_review_rate,
        client_review_rates=args.client_review_rates,
        low_ratio=args.low_ratio,
        high_ratio=args.high_ratio,
        threshold_scale=args.threshold_scale,
        client_threshold_scales=args.client_threshold_scales,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved QC policy: {output}")
    for report in policy["metadata"]["reports"]:
        print(
            f"{report['group']}: n={report['n']}, "
            f"target={report['target_review_rate']:.2%}, "
            f"raw_threshold={report['raw_threshold']:.6g}, "
            f"scale={report['threshold_scale']:.3g}, "
            f"threshold={report['threshold']:.6g}, "
            f"estimated={report['estimated_review_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
