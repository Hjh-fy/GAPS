"""Select a C5-only P4 risk gate over H2.3+ and no-rescue H8 streams."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_regression_head_ablation import fnum, inum, metrics


H23_KEY = "h23_plus_ppm"
H8_KEY = "target_ridge_plus_source_preds_ppm"
RISK_KEY = "risk_score"
CO_CLASS = 1


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row.get("client")), str(row.get("split")), inum(row.get("sample_index"))


def merge_streams(
    h23_rows: Sequence[dict[str, Any]],
    h8_rows: Sequence[dict[str, Any]],
    *,
    expected_split: str,
) -> list[dict[str, Any]]:
    h8_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in h8_rows:
        key = row_key(row)
        if key in h8_by_key:
            raise ValueError(f"duplicate H8 row: {key}")
        h8_by_key[key] = row
    output: list[dict[str, Any]] = []
    for row in h23_rows:
        key = row_key(row)
        if key[0] != "C5" or key[1] != expected_split:
            raise ValueError(f"unexpected H2.3 row role: {key}")
        h8 = h8_by_key.pop(key, None)
        if h8 is None:
            raise ValueError(f"missing H8 row: {key}")
        if inum(row.get("true_class")) != inum(h8.get("true_class")):
            raise ValueError(f"true-class mismatch: {key}")
        item = dict(row)
        item["h23_ppm"] = fnum(row.get(H23_KEY))
        item["h8_ppm"] = fnum(h8.get(H8_KEY))
        item["qc_risk_value"] = fnum(row.get(RISK_KEY))
        output.append(item)
    if h8_by_key:
        raise ValueError(f"unmatched H8 rows: {len(h8_by_key)}")
    return output


def threshold_candidates(rows: Sequence[dict[str, Any]]) -> list[float]:
    risks = sorted(
        {
            fnum(row.get("qc_risk_value"))
            for row in rows
            if inum(row.get("route_class", row.get("pred_class"))) == CO_CLASS
        }
    )
    if not risks:
        return [math.inf]
    candidates = [math.inf, risks[0] - 1e-12]
    candidates.extend((left + right) / 2.0 for left, right in zip(risks, risks[1:]))
    candidates.append(risks[-1] + 1e-12)
    return sorted(set(candidates))


def apply_threshold(
    rows: Sequence[dict[str, Any]],
    threshold: float,
    *,
    output_key: str = "p4_ppm",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        route_class = inum(row.get("route_class", row.get("pred_class")))
        use_h8 = route_class == CO_CLASS and fnum(row.get("qc_risk_value")) >= threshold
        item["p4_threshold"] = threshold
        item["p4_uses_h8"] = int(use_h8)
        item["selected_profile"] = "H8_C5_no_rescue" if use_h8 else "H2.3+_C5"
        item[output_key] = fnum(row.get("h8_ppm" if use_h8 else "h23_ppm"))
        output.append(item)
    return output


def _rmse(rows: Sequence[dict[str, Any]], pred_key: str) -> float:
    value = metrics(rows, pred_key).get("RMSE")
    return float(value) if value is not None else math.inf


def select_threshold(
    validation_rows: Sequence[dict[str, Any]],
    *,
    max_nonco_delta: float = 1.0,
) -> tuple[float, list[dict[str, Any]]]:
    if any(str(row.get("split")) != "calibration" for row in validation_rows):
        raise ValueError("P4 threshold selection accepts calibration-validation rows only")
    nonco = [row for row in validation_rows if inum(row.get("true_class")) != CO_CLASS]
    h23_nonco_rmse = _rmse(nonco, "h23_ppm") if nonco else None
    audits: list[dict[str, Any]] = []
    best: tuple[float, float, float] | None = None
    best_threshold = math.inf
    for threshold in threshold_candidates(validation_rows):
        selected = apply_threshold(validation_rows, threshold)
        overall_rmse = _rmse(selected, "p4_ppm")
        selected_nonco = [row for row in selected if inum(row.get("true_class")) != CO_CLASS]
        nonco_rmse = _rmse(selected_nonco, "p4_ppm") if selected_nonco else None
        nonco_delta = (
            nonco_rmse - h23_nonco_rmse
            if nonco_rmse is not None and h23_nonco_rmse is not None
            else None
        )
        usage = sum(inum(row.get("p4_uses_h8")) for row in selected) / max(len(selected), 1)
        feasible = nonco_delta is None or nonco_delta <= max_nonco_delta + 1e-12
        audits.append(
            {
                "threshold": threshold,
                "N": len(selected),
                "RMSE": overall_rmse,
                "nonCO_RMSE": nonco_rmse,
                "nonCO_delta_vs_H23": nonco_delta,
                "h8_usage_rate": usage,
                "feasible": int(feasible),
            }
        )
        score = (overall_rmse, usage, -threshold)
        if feasible and (best is None or score < best):
            best = score
            best_threshold = threshold
    return best_threshold, audits


def oracle_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        true_ppm = fnum(row.get("true_ppm"))
        h23_error = abs(fnum(row.get("h23_ppm")) - true_ppm)
        h8_error = abs(fnum(row.get("h8_ppm")) - true_ppm)
        use_h8 = h8_error < h23_error
        item["oracle_uses_h8"] = int(use_h8)
        item["oracle_ppm"] = fnum(row.get("h8_ppm" if use_h8 else "h23_ppm"))
        output.append(item)
    return output


def summarize_methods(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    simple_gate = apply_threshold(rows, -math.inf, output_key="simple_co_gate_ppm")
    p4_rows = list(rows)
    oracle = oracle_rows(rows)
    output: list[dict[str, Any]] = []
    for method, method_rows, pred_key in (
        ("H2.3+_C5", rows, "h23_ppm"),
        ("H8_C5_no_rescue", rows, "h8_ppm"),
        ("simple_predicted_CO_gate", simple_gate, "simple_co_gate_ppm"),
        ("P4_C5_risk_gate", p4_rows, "p4_ppm"),
        ("oracle_expert", oracle, "oracle_ppm"),
    ):
        result = metrics(method_rows, pred_key)
        output.append(
            {
                "split": split,
                "method": method,
                **result,
                "h8_usage_rate": (
                    sum(inum(row.get("p4_uses_h8")) for row in method_rows) / max(len(method_rows), 1)
                    if method == "P4_C5_risk_gate"
                    else ""
                ),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    h23_validation = _read_csv(args.h23_validation)
    h8_validation = _read_csv(args.h8_validation)
    h23_test = _read_csv(args.h23_test)
    h8_test = _read_csv(args.h8_test)
    validation = merge_streams(h23_validation, h8_validation, expected_split="calibration")
    test = merge_streams(h23_test, h8_test, expected_split="test")
    threshold, candidates = select_threshold(
        validation, max_nonco_delta=args.max_nonco_delta
    )
    validation_p4 = apply_threshold(validation, threshold)
    test_p4 = apply_threshold(test, threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        *summarize_methods(validation_p4, "calibration_validation"),
        *summarize_methods(test_p4, "test"),
    ]
    _write_csv(output_dir / "p4_threshold_candidates.csv", candidates)
    _write_csv(output_dir / "p4_validation_predictions.csv", validation_p4)
    _write_csv(output_dir / "p4_test_predictions.csv", test_p4)
    _write_csv(output_dir / "p4_method_summary.csv", summary)
    policy = {
        "schema_version": 1,
        "protocol": {"source_clients": [1, 2], "target_clients": [5]},
        "client": "C5",
        "selector_rule": "predicted_class=CO and risk_score>=threshold -> H8_C5_no_rescue else H2.3+_C5",
        "risk_key": RISK_KEY,
        "threshold": threshold,
        "max_nonco_delta": args.max_nonco_delta,
        "selection_split": "calibration_validation",
        "selection_uses_test_labels": False,
        "c4_rescue_enabled": False,
    }
    (output_dir / "p4_policy.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "policy": policy,
        "inputs": {
            "h23_validation": args.h23_validation,
            "h8_validation": args.h8_validation,
            "h23_test": args.h23_test,
            "h8_test": args.h8_test,
        },
        "row_counts": {"validation": len(validation), "test": len(test)},
        "outputs": {
            "policy": str(output_dir / "p4_policy.json"),
            "test_predictions": str(output_dir / "p4_test_predictions.csv"),
            "summary": str(output_dir / "p4_method_summary.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h23-validation", required=True)
    parser.add_argument("--h8-validation", required=True)
    parser.add_argument("--h23-test", required=True)
    parser.add_argument("--h8-test", required=True)
    parser.add_argument("--max-nonco-delta", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
