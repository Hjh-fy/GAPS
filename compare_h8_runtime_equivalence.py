from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["client"]), str(row.get("split", "test")), int(float(row["sample_index"])))


def fnum(value: Any) -> float:
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare H8 analysis predictions with runtime predictions.")
    parser.add_argument("--analysis", default="results/co_only_source_aug_hybrid_stratcalval_20260625/co_only_source_aug_hybrid_predictions.csv")
    parser.add_argument("--analysis-column", default="h8_pred_co_source_aug_else_h23_ppm")
    parser.add_argument("--runtime", default="results/runtime_validation_h8_source_aug_candidate_20260625/runtime_predictions.csv")
    parser.add_argument("--runtime-column", default="co_corrected_ppm")
    parser.add_argument("--output-dir", default="results/equivalence_h8_source_aug_candidate_20260625")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    analysis_rows = read_csv(Path(args.analysis))
    runtime_rows = read_csv(Path(args.runtime))
    analysis_by_key = {key(row): row for row in analysis_rows}
    runtime_by_key = {key(row): row for row in runtime_rows}
    missing_in_analysis = sorted(set(runtime_by_key) - set(analysis_by_key))
    missing_in_runtime = sorted(set(analysis_by_key) - set(runtime_by_key))

    compared: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    diffs: list[float] = []
    for item_key in sorted(set(analysis_by_key) & set(runtime_by_key)):
        a = analysis_by_key[item_key]
        r = runtime_by_key[item_key]
        av = fnum(a[args.analysis_column])
        rv = fnum(r[args.runtime_column])
        diff = abs(av - rv)
        diffs.append(diff)
        row = {
            "client": item_key[0],
            "split": item_key[1],
            "sample_index": item_key[2],
            "analysis": av,
            "runtime": rv,
            "abs_diff": diff,
            "pred_class_analysis": a.get("pred_class"),
            "gas_class_runtime": r.get("gas_class"),
            "true_class": a.get("true_class"),
            "true_ppm": a.get("true_ppm"),
        }
        compared.append(row)
        if diff > args.tolerance:
            mismatches.append(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "equivalence_rows.csv", compared)
    write_csv(out / "mismatch_rows.csv", mismatches)
    write_csv(out / "missing_rows.csv", [{"missing_in": "analysis", "key": item} for item in missing_in_analysis] + [{"missing_in": "runtime", "key": item} for item in missing_in_runtime])
    diff_arr = np.asarray(diffs, dtype=np.float64)
    summary = {
        "analysis_predictions": args.analysis,
        "analysis_column": args.analysis_column,
        "runtime_predictions": args.runtime,
        "runtime_column": args.runtime_column,
        "tolerance": args.tolerance,
        "analysis_rows": len(analysis_rows),
        "runtime_rows": len(runtime_rows),
        "rows_compared": len(compared),
        "missing_in_analysis": len(missing_in_analysis),
        "missing_in_runtime": len(missing_in_runtime),
        "num_mismatch": len(mismatches),
        "max_abs_diff": float(diff_arr.max()) if diff_arr.size else None,
        "mean_abs_diff": float(diff_arr.mean()) if diff_arr.size else None,
        "p99_abs_diff": float(np.percentile(diff_arr, 99)) if diff_arr.size else None,
    }
    (out / "equivalence_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
