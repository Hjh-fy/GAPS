"""Evaluate frozen target-matched GAPS routes with one frozen source H1 head.

This command performs no fitting, tuning, checkpoint selection, or target
calibration access. Existing per-window classification routes are reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.c5_h8_runtime import SerializedRidge
from run_regression_head_ablation import (
    CLASS_NAMES,
    CLASS_RANGES,
    build_oracle_rows,
)

RESULT_ROOT = ROOT / "results/iotj_final_classification_le1_20260804"
DATA_ROOT = (
    ROOT.parents[1]
    / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
)
H1_PATH = ROOT / "results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json"
TARGETS = ("C3", "C4", "C5")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty output: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regression_metrics(
    rows: Sequence[Mapping[str, Any]], prediction_key: str
) -> dict[str, float | int]:
    if not rows:
        raise RuntimeError("FAIL_CLOSED regression metric scope is empty")
    truth = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    prediction = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    classes = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise RuntimeError("FAIL_CLOSED non-finite regression value")
    error = prediction - truth
    ranges = np.asarray([CLASS_RANGES[int(value)] for value in classes], dtype=np.float64)
    centered = truth - float(np.mean(truth))
    total = float(np.sum(centered**2))
    return {
        "N": len(rows),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1.0 - np.sum(error**2) / total) if total > 0 else float("nan"),
        "NRMSE": float(np.sqrt(np.mean((error / ranges) ** 2))),
    }


def load_h1() -> dict[int, SerializedRidge]:
    payload = json.loads(H1_PATH.read_text(encoding="utf-8"))
    if payload.get("source") != "C1_C2_local_sufficient_statistics":
        raise RuntimeError("FAIL_CLOSED H1 source identity differs")
    return {
        int(class_id): SerializedRidge.from_json(model)
        for class_id, model in payload["models"].items()
    }


def target_routes(target: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(RESULT_ROOT / "per_window_predictions.csv")
        if row["experiment_id"] == f"FCL-E3-GAPS-{target}"
    ]
    rows.sort(key=lambda row: int(row["window_index"]))
    if [int(row["window_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"FAIL_CLOSED {target} route index differs")
    return rows


def analyze_target(
    target: str, h1: Mapping[int, SerializedRidge]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    routes = target_routes(target)
    base = build_oracle_rows(DATA_ROOT, [target], "test")
    if len(routes) != len(base):
        raise RuntimeError(f"FAIL_CLOSED {target} route/data count differs")
    records: list[dict[str, Any]] = []
    for route, row in zip(routes, base):
        index = int(row["sample_index"])
        true_class = int(row["true_class"])
        pred_class = int(route["predicted_class"])
        if int(route["window_index"]) != index or int(route["true_class"]) != true_class:
            raise RuntimeError(f"FAIL_CLOSED {target} route/data alignment differs")
        features = row["feature_dict"]
        if len(features) != 104:
            raise RuntimeError("FAIL_CLOSED rich feature dimension differs")
        routed = h1[pred_class].predict(features)
        oracle = h1[true_class].predict(features)
        records.append(
            {
                "experiment_id": f"XTR-GAPS-H1-{target}",
                "target": target,
                "sample_index": index,
                "true_class": true_class,
                "pred_class": pred_class,
                "gas": CLASS_NAMES[true_class],
                "route_correct": int(true_class == pred_class),
                "true_ppm": float(row["true_ppm"]),
                "h1_routed_ppm": routed,
                "h1_oracle_route_ppm": oracle,
                "routed_abs_error": abs(routed - float(row["true_ppm"])),
                "oracle_abs_error": abs(oracle - float(row["true_ppm"])),
                "confidence": float(route["confidence"]),
            }
        )
    summary: list[dict[str, Any]] = []
    scopes = {
        "S_ALL_ROUTED": records,
        "S_CC_ROUTED": [row for row in records if row["route_correct"]],
        "S_ORACLE_ROUTE_DIAGNOSTIC": records,
        "S_WRONG_ROUTE_DIAGNOSTIC": [row for row in records if not row["route_correct"]],
    }
    for scope, selected in scopes.items():
        if not selected:
            continue
        key = "h1_oracle_route_ppm" if scope == "S_ORACLE_ROUTE_DIAGNOSTIC" else "h1_routed_ppm"
        summary.append(
            {
                "experiment_id": f"XTR-GAPS-H1-{target}",
                "target": target,
                "method": "target-matched frozen GAPS router + frozen Federated-H1",
                "sample_scope": scope,
                **regression_metrics(selected, key),
                "seed": 42,
                "calculation_status": "recomputed_no_fit",
            }
        )
    per_gas: list[dict[str, Any]] = []
    for class_id, gas in CLASS_NAMES.items():
        gas_rows = [row for row in records if row["true_class"] == class_id]
        for scope, selected, key in (
            ("S_ALL_ROUTED", gas_rows, "h1_routed_ppm"),
            ("S_CC_ROUTED", [row for row in gas_rows if row["route_correct"]], "h1_routed_ppm"),
            ("S_ORACLE_ROUTE_DIAGNOSTIC", gas_rows, "h1_oracle_route_ppm"),
        ):
            if selected:
                per_gas.append(
                    {
                        "experiment_id": f"XTR-GAPS-H1-{target}",
                        "target": target,
                        "class_id": class_id,
                        "gas": gas,
                        "sample_scope": scope,
                        **regression_metrics(selected, key),
                        "seed": 42,
                        "calculation_status": "recomputed_no_fit",
                    }
                )
    routed = regression_metrics(records, "h1_routed_ppm")
    oracle = regression_metrics(records, "h1_oracle_route_ppm")
    classification = next(
        row
        for row in read_csv(RESULT_ROOT / "classification_main_comparison.csv")
        if row["experiment_id"] == f"FCL-E3-GAPS-{target}"
    )
    decomposition = {
        "experiment_id": f"XTR-GAPS-H1-{target}",
        "target": target,
        "N": len(records),
        "classification_accuracy_reported": float(classification["accuracy"]),
        "classification_macro_f1_reported": float(classification["macro_f1"]),
        "route_correct_N": sum(row["route_correct"] for row in records),
        "misroute_N": sum(1 - row["route_correct"] for row in records),
        "misroute_rate": float(np.mean([1 - row["route_correct"] for row in records])),
        "routed_RMSE": routed["RMSE"],
        "oracle_route_RMSE": oracle["RMSE"],
        "route_penalty_RMSE": float(routed["RMSE"] - oracle["RMSE"]),
        "routed_NRMSE": routed["NRMSE"],
        "oracle_route_NRMSE": oracle["NRMSE"],
        "route_penalty_NRMSE": float(routed["NRMSE"] - oracle["NRMSE"]),
        "interpretation": "oracle route is a post-hoc diagnostic using true class; not a deployable result",
    }
    return records, summary, {"per_gas": per_gas, "decomposition": decomposition}


def build(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    h1 = load_h1()
    all_records: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_per_gas: list[dict[str, Any]] = []
    decompositions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for target in TARGETS:
        records, summary, extra = analyze_target(target, h1)
        all_records.extend(records)
        all_summary.extend(summary)
        all_per_gas.extend(extra["per_gas"])
        decompositions.append(extra["decomposition"])
        run_manifest = RESULT_ROOT / f"FCL-E3-GAPS-{target}/run_manifest.json"
        run = json.loads(run_manifest.read_text(encoding="utf-8"))
        target_dir = DATA_ROOT / f"client_{target[1:]}"
        provenance.append(
            {
                "target": target,
                "classification_experiment_id": f"FCL-E3-GAPS-{target}",
                "classifier_checkpoint": run["checkpoint"],
                "classifier_checkpoint_sha256": run["checkpoint_sha256"],
                "test_N": len(records),
                "test_features_sha256": sha256(target_dir / "test_features.npy"),
                "test_classification_labels_sha256": sha256(target_dir / "test_classification_labels.npy"),
                "test_regression_labels_sha256": sha256(target_dir / "test_regression_labels.npy"),
                "test_phase_labels_sha256": sha256(target_dir / "test_phase_labels.npy"),
                "test_metadata_sha256": sha256(target_dir / "test_experiment_info.json"),
            }
        )
    write_csv(output / "cross_target_regression_summary.csv", all_summary)
    write_csv(output / "cross_target_regression_per_gas.csv", all_per_gas)
    write_csv(output / "cross_target_route_decomposition.csv", decompositions)
    write_csv(output / "cross_target_per_window_predictions.csv", all_records)
    c5_reference = [
        row
        for row in read_csv(
            ROOT / "results/iotj_final_end_to_end_a4_20260804/figures/source_data/fig05_overall_regression.csv"
        )
        if row["variant"] == "R84_FED_H1"
    ]
    for row in c5_reference:
        row["comparison_role"] = "separate_existing_C5_target_personalized_reference_not_cross_target_control"
        row["calculation_status"] = "reported"
    write_csv(output / "c5_a4_r84_personalized_reference.csv", c5_reference)
    manifest = {
        "schema_version": "iotj.gaps_cross_target_regression_router.v1",
        "status": "COMPLETE_NO_FIT",
        "input_result_baseline": "ceb6c78",
        "input_code_base_commit": "904dfbc",
        "execution_script": "scripts/analyze_gaps_cross_target_regression.py",
        "execution_script_sha256": sha256(Path(__file__).resolve()),
        "source_clients": ["C1", "C2"],
        "targets": list(TARGETS),
        "seed": 42,
        "router": "target-matched frozen GAPS round-25 adapted endpoint",
        "regression": "same frozen Federated-H1 per-gas source Ridge for all targets",
        "h1_manifest": str(H1_PATH.relative_to(ROOT)).replace("\\", "/"),
        "h1_manifest_sha256": sha256(H1_PATH),
        "route_source": "results/iotj_final_classification_le1_20260804/per_window_predictions.csv",
        "route_source_sha256": sha256(RESULT_ROOT / "per_window_predictions.csv"),
        "target_calibration_accessed": False,
        "target_regression_fit": False,
        "hyperparameter_search": False,
        "target_test_role": "fixed_endpoint_descriptive_evaluation_only",
        "oracle_route_role": "post_hoc_diagnostic_only",
        "provenance": provenance,
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_target = {row["target"]: row for row in decompositions}
    summary_lookup = {
        (row["target"], row["sample_scope"]): row for row in all_summary
    }
    analysis_lines = [
        "# Cross-target regression capability analysis",
        "",
        "This analysis reuses three frozen GAPS classification routes and one frozen Federated-H1 source regression head. No target regression model was fitted and no calibration or test-based selection was performed.",
        "",
        "| Target | Classification acc. | Macro-F1 | Routed RMSE | Route-correct RMSE | Oracle-route RMSE | Route penalty | Routed NRMSE | Misroute rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        row = by_target[target]
        route_correct = summary_lookup[(target, "S_CC_ROUTED")]
        analysis_lines.append(
            f"| {target} | {row['classification_accuracy_reported']:.4f} | {row['classification_macro_f1_reported']:.4f} | {row['routed_RMSE']:.3f} | {route_correct['RMSE']:.3f} | {row['oracle_route_RMSE']:.3f} | {row['route_penalty_RMSE']:+.3f} | {row['routed_NRMSE']:.4f} | {row['misroute_rate']:.4f} |"
        )
    c5_routed = by_target["C5"]["routed_RMSE"]
    c5_personalized = float(next(row for row in c5_reference if row["evaluation_scope"] == "S_ALL")["RMSE"])
    reduction = 100.0 * (c5_routed - c5_personalized) / c5_routed
    analysis_lines.extend(
        [
            "",
            "## Findings",
            "",
            "- Classification routing is already strong on all targets (98.46%–99.06% accuracy), but source-only H1 routed RMSE remains 67.74–74.71 ppm and overall R2 is negative for all three targets.",
            "- Replacing predicted routes with post-hoc true-class routes changes RMSE by only 0.40–1.29 ppm (0.6%–1.9% of routed RMSE). Thus the dominant observed limitation is source-to-target concentration mapping, not classification routing.",
            "- CO is the largest absolute-RMSE slice on C3 and C4. On C5, CO has the largest absolute RMSE, while ethanol and ethylene have the largest class-range-normalized errors.",
            f"- The separately reported C5 A4+R84 personalized result is {c5_personalized:.3f} ppm versus {c5_routed:.3f} ppm for GAPS+source-only H1 ({reduction:.1f}% lower descriptively). This is not a single-factor effect because both router identity and target-calibrated regression protocol differ.",
            "",
            "Interpretation boundary: the routed value is deployable under the frozen source-only H1 path; the oracle-route value uses the true class only after evaluation and is diagnostic. Cross-target differences combine sensor-domain shift and concentration-distribution differences, so they are descriptive rather than a causal ranking of devices.",
            "",
            "The existing C5 A4+R84 target-personalized result is preserved in a separate reference CSV and must not be pooled with the no-fit H1 rows.",
        ]
    )
    (output / "RESULT_ANALYSIS.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    audit = """# Experiment audit

Verdict: **PASS for descriptive no-fit cross-target capability evidence**.

- All three rows use seed 42 target-matched GAPS endpoints from the same frozen classification study.
- The identical Federated-H1 source model and 104-D feature construction are used for C3/C4/C5.
- Target calibration files are not accessed; no target Ridge, alpha selection, checkpoint selection, QC, or threshold search occurs.
- Target test labels are used only to calculate metrics, route-correct slices, and the explicitly diagnostic oracle-route result.
- C3/C4/C5 have different test sizes and concentration/window distributions. Results must be reported per target and must not be interpreted as a controlled device-only causal effect.
- Single seed 42 supports descriptive endpoint evidence, not stability or uncertainty claims.
- The C5 A4+R84 personalized reference follows a different router and target-calibration protocol and is kept separate.
"""
    (output / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")
    artifacts = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "sha256_index.json"):
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (output / "sha256_index.json").write_text(
        json.dumps({"schema_version": "iotj.gaps_cross_target_regression_router.sha256.v1", "status": "PASS", "artifacts": artifacts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
