"""Assemble the formal C5 R0-R7 regression ladder from scored expert streams."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_regression_head_ablation import fnum, inum, metrics


CO_CLASS = 1
PREDICTION_KEYS = {
    "R0": "baseline_final_ppm",
    "R1": "target_ridge_rich_only_ppm",
    "R2": "h23_anchor_ppm",
    "R3": "h23_plus_ppm",
    "R4": "target_ridge_plus_source_preds_ppm",
    "R5": "R5_ppm",
    "R6": "R6_ppm",
    "R7": "R7_ppm",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in payload for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def _selector_candidates(rows: Sequence[dict[str, Any]], score_key: str) -> list[float]:
    values = sorted(
        {
            fnum(row.get(score_key))
            for row in rows
            if inum(row.get("route_class", row.get("pred_class"))) == CO_CLASS
            and math.isfinite(fnum(row.get(score_key)))
        }
    )
    if not values:
        return [1.0]
    candidates = [values[0] - 1e-12]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-12)
    return candidates


def _selector_prediction(row: dict[str, Any], score_key: str, threshold: float) -> tuple[float, bool]:
    route_class = inum(row.get("route_class", row.get("pred_class")))
    use_h8 = route_class == CO_CLASS and fnum(row.get(score_key)) >= threshold
    key = "target_ridge_plus_source_preds_ppm" if use_h8 else "h23_plus_ppm"
    return fnum(row.get(key)), use_h8


def select_r6_policy(
    validation_rows: Sequence[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    if not validation_rows or any(str(row.get("split")) != "calibration" for row in validation_rows):
        raise ValueError("R6 selector accepts calibration-validation rows only")
    best: tuple[float, float, float] | None = None
    best_threshold = 1.0
    audits: list[dict[str, Any]] = []
    for threshold in _selector_candidates(validation_rows, score_key):
        selected: list[dict[str, Any]] = []
        h8_count = 0
        for row in validation_rows:
            item = dict(row)
            prediction, use_h8 = _selector_prediction(item, score_key, threshold)
            item["R6_ppm"] = prediction
            selected.append(item)
            h8_count += int(use_h8)
        rmse = metrics(selected, "R6_ppm")["RMSE"]
        if rmse is None:
            continue
        usage = h8_count / len(selected)
        audits.append({"threshold": threshold, "RMSE": rmse, "h8_usage_rate": usage})
        score = (float(rmse), usage, -threshold)
        if best is None or score < best:
            best = score
            best_threshold = threshold
    if best is None:
        raise ValueError("R6 selector could not evaluate any threshold")
    return {
        "schema_version": 1,
        "score_key": score_key,
        "threshold": best_threshold,
        "rule": "predicted_class=CO and deployment risk >= threshold -> R4 H8 else R3 H2.3+",
        "selection_split": "calibration_validation",
        "selection_uses_test_labels": False,
        "selected_validation_RMSE": best[0],
        "selected_h8_usage_rate": best[1],
        "candidate_audit": audits,
    }


def apply_ladder(
    rows: Sequence[dict[str, Any]],
    r6_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    score_key = str(r6_policy["score_key"])
    threshold = float(r6_policy["threshold"])
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_values: dict[str, float] = {}
        for mode, key in tuple(PREDICTION_KEYS.items())[:5]:
            value = fnum(item.get(key))
            if not math.isfinite(value):
                raise ValueError(f"missing or non-finite {mode} prediction {key}")
            base_values[mode] = value
        route_class = inum(item.get("route_class", item.get("pred_class")))
        h23 = base_values["R3"]
        h8 = base_values["R4"]
        for mode in ("R0", "R1", "R2", "R3", "R4"):
            item[f"{mode}_ppm"] = base_values[mode]
        r5_uses_h8 = route_class == CO_CLASS
        item["R5_uses_h8"] = int(r5_uses_h8)
        item["R5_ppm"] = h8 if r5_uses_h8 else h23
        r6_prediction, r6_uses_h8 = _selector_prediction(item, score_key, threshold)
        item["R6_score_key"] = score_key
        item["R6_threshold"] = threshold
        item["R6_uses_h8"] = int(r6_uses_h8)
        item["R6_ppm"] = r6_prediction
        true_ppm = fnum(item.get("true_ppm"))
        r7_uses_h8 = abs(h8 - true_ppm) < abs(h23 - true_ppm)
        item["R7_uses_h8"] = int(r7_uses_h8)
        item["R7_uses_test_truth"] = 1
        item["R7_ppm"] = h8 if r7_uses_h8 else h23
        output.append(item)
    return output


def summarize_ladder(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if str(row.get("split")) == "test"]
    if not test_rows:
        raise ValueError("R0-R7 summary requires test rows")
    total = len(test_rows)
    scopes: list[tuple[str, list[dict[str, Any]]]] = [
        ("S_ALL", test_rows),
        (
            "S_CC",
            [row for row in test_rows if inum(row.get("pred_class")) == inum(row.get("true_class"))],
        ),
        (
            "S_CW",
            [row for row in test_rows if inum(row.get("pred_class")) != inum(row.get("true_class"))],
        ),
    ]
    for class_id in range(4):
        scopes.append(
            (f"gas_{class_id}", [row for row in test_rows if inum(row.get("true_class")) == class_id])
        )
    output: list[dict[str, Any]] = []
    for mode, source_key in PREDICTION_KEYS.items():
        pred_key = f"{mode}_ppm"
        for scope, selected in scopes:
            if not selected:
                continue
            result = metrics(selected, pred_key)
            output.append(
                {
                    "mode": mode,
                    "source_prediction_key": source_key,
                    "scope": scope,
                    "N": len(selected),
                    "coverage": len(selected) / total,
                    **{key: value for key, value in result.items() if key != "N"},
                    "uses_test_truth_at_runtime": int(mode == "R7"),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-scored", required=True)
    parser.add_argument("--test-scored", required=True)
    parser.add_argument("--risk-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _read_csv(args.validation_scored)
    test = _read_csv(args.test_scored)
    selection = json.loads(Path(args.risk_selection).read_text(encoding="utf-8"))
    score_key = str(selection["selected_score"])
    policy = select_r6_policy(validation, score_key)
    validation_ladder = apply_ladder(validation, policy)
    test_ladder = apply_ladder(test, policy)
    summary = summarize_ladder(test_ladder)
    _write_csv(output_dir / "calibration_validation_r0_r7.csv", validation_ladder)
    _write_csv(output_dir / "test_r0_r7.csv", test_ladder)
    _write_csv(output_dir / "r0_r7_summary.csv", summary)
    (output_dir / "r6_policy.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "protocol": {"source_clients": [1, 2], "target_clients": [5]},
        "row_counts": {"calibration_validation": len(validation), "test": len(test)},
        "selection_uses_test_labels": False,
        "r7_oracle_uses_test_truth": True,
        "outputs": {
            "summary": str(output_dir / "r0_r7_summary.csv"),
            "test_predictions": str(output_dir / "test_r0_r7.csv"),
            "r6_policy": str(output_dir / "r6_policy.json"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
