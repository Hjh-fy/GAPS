"""Check runtime equivalence for an exported target Ridge profile artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gaps_deploy.inference import DeployResult, GAS_NAMES
from gaps_deploy.rich_residual import RichResidualPolicy
from run_regression_head_ablation import client_name, fnum, inum, load_split, read_csv


DEFAULT_DATA_ROOT = Path("dataset/client_data_c45src_c123tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_ARTIFACT = Path(
    "results/deployment_target_ridge_c45_c123_candidate_20260626/rich_residual_candidate.json"
)
DEFAULT_PREDICTIONS = Path(
    "results/formal_target_ridge_auto_v2_c45_c123_20260625/formal_target_ridge_predictions.csv"
)
DEFAULT_OUT_DIR = Path("results/equivalence_target_ridge_c45_c123_candidate_20260626")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def result_from_row(row: dict[str, str]) -> DeployResult:
    pred_class = inum(row.get("pred_class"))
    return DeployResult(
        pred_gas=GAS_NAMES[pred_class] if 0 <= pred_class < len(GAS_NAMES) else "",
        pred_class=pred_class,
        pred_ppm=fnum(row.get("pred_ppm")),
        calibrated_ppm=fnum(row.get("calibrated_ppm")),
        base_r3ak16_raw_ppm=fnum(row.get("base_r3ak16_raw_ppm", row.get("raw_ppm"))),
        routed_pred_ppm=fnum(row.get("routed_pred_ppm", row.get("pred_ppm"))),
        final_ppm=fnum(row.get("final_ppm", row.get("baseline_final_ppm"))),
        qc_status=str(row.get("qc_status", "accept")),
        risk_score=fnum(row.get("risk_score")),
        client_id=client_name(row.get("client") or row.get("client_id")),
        confidence=fnum(row.get("confidence")),
        top1_confidence=fnum(row.get("top1_confidence")),
        top2_confidence=fnum(row.get("top2_confidence")),
        confidence_margin=fnum(row.get("confidence_margin")),
        phase=inum(row.get("phase")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--expected-column", default="ridge_direct_ppm")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-clients", default="")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    policy = RichResidualPolicy.from_json(args.artifact)
    rows = [row for row in read_csv(args.predictions) if str(row.get("split")) == "test"]
    if args.target_clients:
        target_clients = {client_name(item) for item in args.target_clients.split(",") if item.strip()}
        rows = [row for row in rows if client_name(row.get("client") or row.get("client_id")) in target_clients]
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    diffs: list[float] = []
    for row in rows:
        client = client_name(row.get("client") or row.get("client_id"))
        split = str(row.get("split"))
        key = (client, split)
        if key not in cache:
            cache[key] = load_split(args.data_root, client, split)
        features, _cls, _reg, phase, meta_rows = cache[key]
        idx = inum(row.get("sample_index"))
        meta = dict(meta_rows[idx]) if idx < len(meta_rows) else {}
        for name in [
            "response_phase",
            "phase_label",
            "window_start_s",
            "window_end_s",
            "window_center_s",
            "t_onset",
            "t_min",
            "interpolated_ratio",
            "max_gap_inside_window",
        ]:
            if row.get(name) not in (None, ""):
                meta.setdefault(name, row.get(name))
        result = result_from_row(row)
        if result.phase < 0 and idx < len(phase):
            result.phase = int(phase[idx])
        runtime_ppm = policy.apply(features[idx], result, client, meta=meta)
        expected = fnum(row.get(args.expected_column))
        diff = abs(runtime_ppm - expected)
        diffs.append(diff)
        item = {
            "client": client,
            "split": split,
            "sample_index": idx,
            "pred_class": result.pred_class,
            "expected_ppm": expected,
            "runtime_ppm": runtime_ppm,
            "abs_diff": diff,
        }
        comparisons.append(item)
        if diff > args.tolerance:
            mismatches.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target_ridge_runtime_comparisons.csv", comparisons)
    write_csv(args.output_dir / "target_ridge_runtime_mismatches.csv", mismatches)
    summary = {
        "artifact": str(args.artifact),
        "predictions": str(args.predictions),
        "expected_column": args.expected_column,
        "num_rows": len(comparisons),
        "num_mismatch": len(mismatches),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "tolerance": args.tolerance,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "equivalence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
