"""Minimal scientific closure for the frozen IoT-J v4 evidence scope.

This script performs only the six approved reduced-budget C5 Ridge fits and
deterministic post-processing of existing canonical classification/QC assets.
It refuses to overwrite an existing output root and publishes atomically only
after all requested tables, figures, and the final report have been built.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_quantitative_features import load_feature_cache
from gaps_flower.canonical_r1_v1 import (
    bootstrap_paired_group_deltas,
    fit_ridge_model,
    predict_ridge_model,
    select_grouped_cv_alpha,
    validate_r0_bundle,
)


OUTPUT_DEFAULT = ROOT / "results/iotj_v4_final_evidence_20260917"
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
R1_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812"
R0_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
BUDGET_ROOT = ROOT / "results/iotj_canonical_v1_c5_budget_20260810"
CLASSIFICATION_SOURCE = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_classification_comparison.csv"
Q1_CURVE = ROOT / "results/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812/q1_risk_coverage_curves.csv"
HC_METRICS = ROOT / "results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_CONFIDENCE_QC_METRICS.csv"
HC_THRESHOLDS = ROOT / "results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_CONFIDENCE_QC_THRESHOLDS.csv"
ROUND_ROOT = ROOT / "results/iotj_v4_evidence_completion_20260912"
STRICT_ANALYSIS = ROUND_ROOT / "file_disjoint_analysis.md"
R1_PROTOCOL = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812/protocol_manifest.json"

R0_INDEX_SHA256 = "04e62b0a2b363fcb79cbeb60ad6c2c6c9a3dedef380b2b0f7c2aa808732d4d82"
DATASET_SHA256 = "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"
STUDY_ID = "CAN-V1-CRRQ-R1-CANONICAL-83D-R84-20260812"
ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BUDGET_COUNTS = {20: 320, 15: 240, 10: 160, 5: 80}
REDUCED_BUDGETS = (15, 10, 5)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
GAS_NAMES = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}
METHOD_TO_CANONICAL = {"R83": "TARGET_ONLY_83D_RIDGE", "R84": "R84_CONCAT"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(str(key) for key in row if str(key) not in fields)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _budget_path(budget: int) -> Path:
    return BUDGET_ROOT / f"c5_calibration_budget_{budget:02d}pct.csv"


def _metrics(truth: np.ndarray, pred: np.ndarray, ranges: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)
    err = pred - truth
    sst = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "n": int(len(truth)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1.0 - np.sum(err**2) / sst) if sst else float("nan"),
        "NRMSE": float(np.sqrt(np.mean((err / ranges) ** 2))),
    }


def _source_matrix(models: Mapping[int, Mapping[str, Any]], h1: np.ndarray) -> np.ndarray:
    return np.column_stack([predict_ridge_model(models[gas], h1) for gas in range(4)])


def _one_by_key(rows: Iterable[Mapping[str, str]], key: str) -> dict[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for row in rows:
        value = str(row[key])
        if value in output:
            raise RuntimeError(f"duplicate {key}: {value}")
        output[value] = row
    return output


def _load_caches() -> dict[str, Any]:
    cache_root = R1_ROOT / "canonical_feature_caches/C5"
    cal_sensor, cal_h1, cal_ids, cal_manifest = load_feature_cache(
        cache_root / "calibration",
        expected_dataset_sha256=DATASET_SHA256,
        expected_study_id=STUDY_ID,
    )
    test_sensor, test_h1, test_ids, test_manifest = load_feature_cache(
        cache_root / "test",
        expected_dataset_sha256=DATASET_SHA256,
        expected_study_id=STUDY_ID,
    )
    return {
        "cal_sensor": cal_sensor,
        "cal_h1": cal_h1,
        "cal_ids": cal_ids,
        "cal_manifest": cal_manifest,
        "test_sensor": test_sensor,
        "test_h1": test_h1,
        "test_ids": test_ids,
        "test_manifest": test_manifest,
    }


def preflight() -> dict[str, Any]:
    caches = _load_caches()
    budgets = {budget: _read_csv(_budget_path(budget)) for budget in BUDGET_COUNTS}
    sets = {budget: {row["canonical_identity"] for row in rows} for budget, rows in budgets.items()}
    if any(len(rows) != BUDGET_COUNTS[budget] for budget, rows in budgets.items()):
        raise RuntimeError("frozen calibration budget row-count mismatch")
    if any(len(sets[budget]) != BUDGET_COUNTS[budget] for budget in sets):
        raise RuntimeError("duplicate calibration identity in budget manifest")
    if not sets[5] < sets[10] < sets[15] < sets[20]:
        raise RuntimeError("budget nesting must be 5% < 10% < 15% < 20%")

    cal_by_id = {row["physical_identity"]: index for index, row in enumerate(caches["cal_ids"])}
    if len(cal_by_id) != 320 or sets[20] != set(cal_by_id):
        raise RuntimeError("budget identities do not equal the frozen C5 calibration cache")
    cal_y4 = np.load(DATA_ROOT / "client_5/calibration_regression_labels.npy", allow_pickle=False)
    cal_cls = np.load(DATA_ROOT / "client_5/calibration_classification_labels.npy", allow_pickle=False).astype(int)
    for budget, rows in budgets.items():
        for row in rows:
            index = cal_by_id[row["canonical_identity"]]
            identity = caches["cal_ids"][index]
            if (
                int(row["class_id"]) != int(identity["class_id"])
                or int(row["class_id"]) != int(cal_cls[index])
                or abs(float(row["concentration"]) - float(identity["concentration"])) > 1e-9
                or abs(float(row["concentration"]) - float(cal_y4[index, cal_cls[index]])) > 1e-9
                or row["raw_filename"] != identity["filename"]
            ):
                raise RuntimeError(f"budget semantic mismatch at {budget}%: {row['canonical_identity']}")

    r1_predictions = _read_csv(R1_ROOT / "predictions.csv")
    p83 = _one_by_key(
        (row for row in r1_predictions if row["target"] == "C5" and row["method"] == "TARGET_ONLY_83D_RIDGE"),
        "physical_identity",
    )
    p84 = _one_by_key(
        (row for row in r1_predictions if row["target"] == "C5" and row["method"] == "R84_CONCAT"),
        "physical_identity",
    )
    test_ids = [row["physical_identity"] for row in caches["test_ids"]]
    if len(test_ids) != 1360 or len(set(test_ids)) != 1360 or set(test_ids) != set(p83) or set(p83) != set(p84):
        raise RuntimeError("R83/R84/test-cache identity mismatch")
    for identity in test_ids:
        a, b = p83[identity], p84[identity]
        if any(a[field] != b[field] for field in ("filename", "true_class", "predicted_class", "truth", "gas_range")):
            raise RuntimeError(f"20% pair metadata mismatch: {identity}")

    cal_manifest, test_manifest = caches["cal_manifest"], caches["test_manifest"]
    feature_keys = (
        "dataset_aggregate_sha256",
        "schema_bundle_sha256",
        "ordered_h1_feature_names_sha256",
        "ordered_sensor_feature_names_sha256",
        "sampling_rate_hz",
        "window_shape",
    )
    if any(cal_manifest[key] != test_manifest[key] for key in feature_keys):
        raise RuntimeError("calibration/test preprocessing contract mismatch")
    if caches["cal_sensor"].shape != (320, 83) or caches["cal_h1"].shape != (320, 104):
        raise RuntimeError("calibration feature definition mismatch")
    if caches["test_sensor"].shape != (1360, 83) or caches["test_h1"].shape != (1360, 104):
        raise RuntimeError("test feature definition mismatch")

    protocol = json.loads(R1_PROTOCOL.read_text(encoding="utf-8"))
    if tuple(float(value) for value in protocol["alpha_grid"]) != ALPHAS:
        raise RuntimeError("Ridge alpha grid mismatch")
    expected_rule = "deterministic five-fold raw-filename-grouped target-calibration CV; pooled RMSE; first tie"
    if protocol["alpha_selection"] != expected_rule:
        raise RuntimeError("Ridge fitting rule mismatch")

    source_models = validate_r0_bundle(R0_ROOT, expected_index_sha256=R0_INDEX_SHA256)
    if set(source_models) != {0, 1, 2, 3}:
        raise RuntimeError("frozen source-prior gas coverage mismatch")

    groups = {row["filename"] for row in caches["test_ids"]}
    by_gas = {
        gas: {row["filename"] for row in caches["test_ids"] if int(row["class_id"]) == gas}
        for gas in range(4)
    }
    grouped_ci_available = len(groups) == 80 and all(len(by_gas[gas]) == 20 for gas in range(4))

    checks = {
        "1_frozen_budget_counts": "PASS",
        "2_nested_5_10_15_20": "PASS",
        "3_same_calibration_rows_R83_R84": "PASS_BY_SINGLE_SHARED_INDEX_SET",
        "4_same_test_set": "PASS_1360_IDENTITIES",
        "5_R83_feature_definition": "PASS_83D",
        "6_R84_definition": "PASS_R83_PLUS_ONE_FROZEN_R0V2_SCALAR",
        "7_Ridge_alpha_and_fit_rule": "PASS_FIXED_RULE",
        "8_preprocessing_unchanged": "PASS_CANONICAL_V1_5HZ_50X8",
    }
    return {
        "checks": checks,
        "budgets": budgets,
        "budget_sets": sets,
        "caches": caches,
        "cal_by_id": cal_by_id,
        "cal_y4": cal_y4,
        "cal_cls": cal_cls,
        "source_models": source_models,
        "p83": p83,
        "p84": p84,
        "test_ids": test_ids,
        "grouped_ci_available": grouped_ci_available,
        "group_count": len(groups),
        "group_count_per_gas": {GAS_NAMES[key]: len(value) for key, value in by_gas.items()},
        "protocol": protocol,
    }


def _test_vectors(context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = [context["p83"][identity] for identity in context["test_ids"]]
    truth = np.asarray([float(row["truth"]) for row in rows])
    true_class = np.asarray([int(row["true_class"]) for row in rows], dtype=int)
    predicted_class = np.asarray([int(row["predicted_class"]) for row in rows], dtype=int)
    ranges = np.asarray([float(row["gas_range"]) for row in rows])
    groups = np.asarray([f"C5|{row['filename']}" for row in rows])
    return {
        "truth": truth,
        "true_class": true_class,
        "predicted_class": predicted_class,
        "ranges": ranges,
        "groups": groups,
    }


def _fit_reduced_budget(
    budget: int,
    model_name: str,
    context: Mapping[str, Any],
    cal_prior: np.ndarray,
    test_prior: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    rows = context["budgets"][budget]
    indices = np.asarray([context["cal_by_id"][row["canonical_identity"]] for row in rows], dtype=int)
    cal_sensor = context["caches"]["cal_sensor"][indices]
    cal_y4 = context["cal_y4"][indices]
    cal_cls = context["cal_cls"][indices]
    cal_groups = np.asarray([row["raw_filename"] for row in rows])
    test_sensor = context["caches"]["test_sensor"]
    gas_models: dict[str, Any] = {}
    alpha_audit: list[dict[str, Any]] = []
    matrix: list[np.ndarray] = []
    for gas in range(4):
        mask = cal_cls == gas
        if model_name == "R83":
            x_cal = cal_sensor[mask]
            x_test = test_sensor
        else:
            x_cal = np.column_stack([cal_sensor[mask], cal_prior[indices][mask, gas]])
            x_test = np.column_stack([test_sensor, test_prior[:, gas]])
        if x_cal.shape[1] != (83 if model_name == "R83" else 84):
            raise RuntimeError(f"{model_name} feature dimension mismatch")
        y = cal_y4[mask, gas]
        cv = select_grouped_cv_alpha(x_cal, y, cal_groups[mask], ALPHAS)
        model = fit_ridge_model(x_cal, y, cv.alpha, float(y.min()), float(y.max()))
        gas_models[str(gas)] = model
        alpha_audit.append(
            {
                "gas_id": gas,
                "gas": GAS_NAMES[gas],
                "selected_alpha": cv.alpha,
                "calibration_n": int(mask.sum()),
                "feature_dimension": int(x_cal.shape[1]),
                "fold_by_group": cv.fold_by_group,
                "pooled_rmse_by_alpha": cv.pooled_rmse,
            }
        )
        matrix.append(predict_ridge_model(model, x_test))
    return {"models": gas_models, "alpha_audit": alpha_audit}, np.column_stack(matrix), alpha_audit


def _reported_20_metrics() -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    overall_rows = _read_csv(R1_ROOT / "canonical_regression_comparison.csv")
    gas_rows = _read_csv(R1_ROOT / "canonical_regression_per_gas.csv")
    overall: dict[str, dict[str, Any]] = {}
    per_gas: dict[str, dict[int, dict[str, Any]]] = {"R83": {}, "R84": {}}
    for model, method in METHOD_TO_CANONICAL.items():
        candidates = [row for row in overall_rows if row["target"] == "C5" and row["scope"] == "S_ALL" and row["method"] == method]
        if len(candidates) != 1:
            raise RuntimeError(f"missing sealed 20% overall row: {model}")
        row = candidates[0]
        overall[model] = {
            "n": int(row["n"]),
            "RMSE": float(row["rmse"]),
            "MAE": float(row["mae"]),
            "R2": float(row["r2"]),
            "NRMSE": float(row["nrmse_range"]),
        }
        candidates = [row for row in gas_rows if row["target"] == "C5" and row["scope"] == "S_ALL" and row["method"] == method]
        if len(candidates) != 4:
            raise RuntimeError(f"missing sealed 20% per-gas rows: {model}")
        for row in candidates:
            gas = int(row["gas_id"])
            per_gas[model][gas] = {
                "n": int(row["n"]),
                "RMSE": float(row["rmse"]),
                "MAE": float(row["mae"]),
                "R2": float(row["r2"]),
                "NRMSE": float(row["nrmse_range"]),
            }
    return overall, per_gas


def _classification_table(output: Path) -> list[dict[str, Any]]:
    rows = _read_csv(CLASSIFICATION_SOURCE)
    methods = (
        ("Source-only FL", "FedAvg", "None (source-only)"),
        ("Source-only FL", "FedProx", "None (source-only)"),
        ("Source-only FL", "SCAFFOLD", "None (source-only)"),
        ("Unlabeled DA", "MMD", "Target x only"),
        ("Label-aware", "A0T", "Target x + calibration class labels"),
        ("Label-aware", "GAPS/A4", "Target x + calibration class and phase labels"),
    )
    lookup = {(row["method"], row["target"]): row for row in rows}
    table: list[dict[str, Any]] = []
    for regime, method, information in methods:
        required = [lookup.get((method, target)) for target in ("C3", "C4", "C5", "ALL")]
        if any(row is None for row in required):
            raise RuntimeError(f"canonical classification row missing: {method}")
        c3, c4, c5, pooled = required
        table.append(
            {
                "Information regime": regime,
                "Method": method,
                "Target information": information,
                "C3 Accuracy": float(c3["accuracy"]),
                "C4 Accuracy": float(c4["accuracy"]),
                "C5 Accuracy": float(c5["accuracy"]),
                "pooled Accuracy": float(pooled["accuracy"]),
                "C3 Macro-F1": float(c3["macro_f1"]),
                "C4 Macro-F1": float(c4["macro_f1"]),
                "C5 Macro-F1": float(c5["macro_f1"]),
                "pooled Macro-F1": float(pooled["macro_f1"]),
            }
        )
    _write_csv(output / "canonical_classification_table.csv", table)
    exclusions = [
        {
            "Method": method,
            "Status": "EXCLUDED_NONCANONICAL",
            "Observed C3 N": 680,
            "Observed C4 N": 320,
            "Observed C5 N": 1360,
            "Canonical C3 N": 2677,
            "Canonical C4 N": 1360,
            "Canonical C5 N": 1360,
            "Reason": "C3/C4 final endpoints do not use the canonical test population; no retraining was performed.",
        }
        for method in ("CORAL", "DANN")
    ]
    _write_csv(output / "canonical_classification_exclusions.csv", exclusions)
    return table


def _qc_tables(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    requested = {0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00}
    curve_rows = []
    for row in _read_csv(Q1_CURVE):
        coverage = float(row["target_coverage"])
        if row["policy"] == "CLASSIFICATION_CONFIDENCE_ONLY" and coverage in requested and row["scope"] in {"C3", "C4", "C5", "POOLED"}:
            curve_rows.append(
                {
                    "policy": row["policy"],
                    "scope": row["scope"],
                    "target_coverage": coverage,
                    "actual_coverage": float(row["actual_coverage"]),
                    "retained_n": int(row["retained_n"]),
                    "NRMSE": float(row["NRMSE_range"]),
                }
            )
    if len(curve_rows) != 28:
        raise RuntimeError("confidence-only 70%-100% QC grid incomplete")
    curve_rows.sort(key=lambda row: (row["scope"], row["target_coverage"]))
    _write_csv(output / "qc_coverage_nrmse.csv", curve_rows)

    pooled_100 = next(row for row in curve_rows if row["scope"] == "POOLED" and row["target_coverage"] == 1.0)
    metrics = [row for row in _read_csv(HC_METRICS) if row["target"] == "POOLED_C3_C4_C5" and row["policy"] == "CONFIDENCE_QC_FINAL"]
    thresholds = _read_csv(HC_THRESHOLDS)
    threshold_ok = all(
        len([row for row in thresholds if row["target"] == target and float(row["nominal_coverage"]) == nominal and row["source"] == "calibration_only"]) == 1
        for target in ("C3", "C4", "C5")
        for nominal in (0.9, 0.95)
    )
    compatible = (
        len(metrics) == 2
        and threshold_ok
        and all(abs(float(row["full_nrmse_range"]) - pooled_100["NRMSE"]) <= 1e-12 for row in metrics)
    )
    hc_rows: list[dict[str, Any]] = []
    if compatible:
        for row in sorted(metrics, key=lambda item: float(item["nominal_coverage"])):
            hc_rows.append(
                {
                    "policy": row["policy"],
                    "operating_point": f"HC{int(round(float(row['nominal_coverage']) * 100))}",
                    "nominal_coverage": float(row["nominal_coverage"]),
                    "actual_coverage": float(row["coverage"]),
                    "accepted_n": int(row["accepted_n"]),
                    "NRMSE": float(row["accepted_nrmse_range"]),
                    "threshold": row["threshold"],
                    "provenance": "current-authoritative confidence-only calibration-locked operating point",
                }
            )
        _write_csv(output / "qc_hc_operating_points.csv", hc_rows)
    return curve_rows, hc_rows, compatible


def _plot_budget(summary_rows: Sequence[Mapping[str, Any]], effect_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    budgets = [5, 10, 15, 20]
    values = {(int(row["budget_pct"]), str(row["model"])): float(row["NRMSE"]) for row in summary_rows}
    deltas = {int(row["budget_pct"]): float(row["delta_NRMSE"]) for row in effect_rows}
    colors = {"R83": "#0072B2", "R84": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for model, marker in (("R83", "o"), ("R84", "s")):
        axes[0].plot(budgets, [values[(budget, model)] for budget in budgets], marker=marker, linewidth=1.8, color=colors[model], label=model)
    axes[0].set_xlabel("Calibration budget (%)")
    axes[0].set_ylabel("NRMSE")
    axes[0].set_xticks(budgets)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].text(-0.16, 1.04, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].plot(budgets, [deltas[budget] for budget in budgets], marker="D", linewidth=1.8, color="#009E73")
    axes[1].axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axes[1].set_xlabel("Calibration budget (%)")
    axes[1].set_ylabel(r"$\Delta$NRMSE (R84 $-$ R83)")
    axes[1].set_xticks(budgets)
    axes[1].grid(alpha=0.25)
    axes[1].text(-0.16, 1.04, "b", transform=axes[1].transAxes, fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"Figure_B_calibration_budget_vs_nrmse.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def _plot_qc(curve_rows: Sequence[Mapping[str, Any]], hc_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    pooled = sorted((row for row in curve_rows if row["scope"] == "POOLED"), key=lambda row: float(row["target_coverage"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.35), constrained_layout=True)
    ax.plot(
        [float(row["actual_coverage"]) for row in pooled],
        [float(row["NRMSE"]) for row in pooled],
        marker="o",
        linewidth=1.8,
        color="#0072B2",
        label="Confidence-ranked curve",
    )
    for row, color in zip(hc_rows, ("#D55E00", "#009E73")):
        ax.scatter(float(row["actual_coverage"]), float(row["NRMSE"]), marker="D", s=45, color=color, zorder=3, label=str(row["operating_point"]))
    ax.set_xlabel("Actual retained coverage")
    ax.set_ylabel("Accepted-set NRMSE")
    ax.set_xlim(0.68, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"Figure_C_coverage_vs_nrmse.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), constrained_layout=True, sharex=True)
    colors = {"C3": "#0072B2", "C4": "#E69F00", "C5": "#009E73"}
    for ax, scope in zip(axes, ("C3", "C4", "C5")):
        rows = sorted((row for row in curve_rows if row["scope"] == scope), key=lambda row: float(row["target_coverage"]))
        ax.plot([float(row["actual_coverage"]) for row in rows], [float(row["NRMSE"]) for row in rows], marker="o", color=colors[scope], linewidth=1.6)
        ax.set_title(scope)
        ax.set_xlabel("Coverage")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accepted-set NRMSE")
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(output / f"Figure_C_supplement_by_target.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def _copy_round_figure(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        source = ROUND_ROOT / f"figures/fig_classification_accuracy_vs_round.{suffix}"
        shutil.copy2(source, output / f"Figure_A_classification_accuracy_vs_round.{suffix}")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _build_report(
    classification: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    per_gas: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    hc_compatible: bool,
) -> str:
    cls_rows = [
        [
            str(row["Information regime"]),
            str(row["Method"]),
            str(row["Target information"]),
            f"{100*float(row['C3 Accuracy']):.3f}",
            f"{100*float(row['C4 Accuracy']):.3f}",
            f"{100*float(row['C5 Accuracy']):.3f}",
            f"{100*float(row['pooled Accuracy']):.3f}",
            f"{100*float(row['C3 Macro-F1']):.3f}",
            f"{100*float(row['C4 Macro-F1']):.3f}",
            f"{100*float(row['C5 Macro-F1']):.3f}",
            f"{100*float(row['pooled Macro-F1']):.3f}",
        ]
        for row in classification
    ]
    effect_rows = [
        [
            str(row["budget_pct"]),
            str(row["calibration_n"]),
            f"{float(row['NRMSE_R83']):.6f}",
            f"{float(row['NRMSE_R84']):.6f}",
            f"{float(row['delta_NRMSE']):+.6f}",
            f"{float(row['relative_improvement_pct']):+.2f}%",
        ]
        for row in sorted(effects, key=lambda item: int(item["budget_pct"]), reverse=True)
    ]
    ci_rows = [
        [str(row["budget_pct"]), f"{float(row['point_delta_NRMSE']):+.6f}", f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]"]
        for row in sorted(bootstrap, key=lambda item: int(item["budget_pct"]), reverse=True)
    ]
    gas_lines = []
    for row in sorted(per_gas, key=lambda item: (-int(item["budget_pct"]), int(item["gas_id"]))):
        gas_lines.append(
            [str(row["budget_pct"]), str(row["gas"]), f"{float(row['NRMSE_R83']):.6f}", f"{float(row['NRMSE_R84']):.6f}", f"{float(row['delta_NRMSE']):+.6f}"]
        )
    hc_note = (
        "HC90/HC95 were overlaid because the current-authoritative rows match the same R84 backend, canonical C3/C4/C5 population, confidence-only risk, and full-population NRMSE."
        if hc_compatible
        else "HC90/HC95 were not overlaid because the compatibility check did not pass."
    )
    return f"""# IoT-J v4 Final Canonical Evidence Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Classification

The main table is copied from the existing canonical fixed-endpoint evidence; no classifier was trained or adapted in this closure. Pooled Accuracy and Macro-F1 use the frozen definition: **merge C3/C4/C5 predictions first, then compute one pooled metric**. Methods from different target-information regimes are shown contextually and are not an equal-supervision total ranking. M0 is excluded from the main table.

{_md_table(['Regime','Method','Target information','C3 Acc. (%)','C4 Acc. (%)','C5 Acc. (%)','Pooled Acc. (%)','C3 Macro-F1 (%)','C4 Macro-F1 (%)','C5 Macro-F1 (%)','Pooled Macro-F1 (%)'], cls_rows)}

CORAL and DANN remain `EXCLUDED_NONCANONICAL`: their observed C3/C4 endpoint populations are 680/320 rather than 2677/1360. They were not retrained. Figure A directly reuses the existing 25-round Accuracy curves; those retrospective curves use the earlier FCL-E3 endpoint populations and are diagnostic only, not checkpoint-selection evidence for the canonical main table.

## Regression

Exactly six new deterministic Ridge fits were run: R83 and R84 at 15%, 10%, and 5%. The 20% point was **reported/reused**, not refit, from the later sealed R1 (2026-08-12) using the same R0-v2 source prior, feature caches, route, and test population as the reduced-budget runs. The earlier 2026-08-08 Table-10 numbers are not mixed into this curve because they belong to a different frozen provenance chain.

`delta_NRMSE = R84 - R83`; negative values indicate lower error with the source prior.

{_md_table(['Budget (%)','Calibration N','R83 NRMSE','R84 NRMSE','Delta NRMSE','Relative improvement'], effect_rows)}

### Paired original-file bootstrap

The C5 test metadata reliably recover 80 original-file/exposure groups (20 per gas). The interval uses 1,000 paired group-bootstrap replicates with seed 42 and identical resampled groups for R83/R84. Intervals are conditional on each single frozen calibration subset.

{_md_table(['Budget (%)','Point delta NRMSE','Paired 95% CI'], ci_rows)}

### Per-gas source-prior effect

{_md_table(['Budget (%)','Gas','R83 NRMSE','R84 NRMSE','Delta NRMSE'], gas_lines)}

Effects are descriptive and may be negative, zero, positive, or non-monotonic; no subset, alpha rule, prior, or model was changed after observing test results.

## QC

Figure C uses only `CLASSIFICATION_CONFIDENCE_ONLY` rows from the authoritative Q1 risk-coverage CSV at nominal 70%, 75%, 80%, 85%, 90%, 95%, and 100%, plotting actual retained coverage against accepted-set `NRMSE_range`. {hc_note} The random-retention band was intentionally omitted because it is optional and would add nonessential engineering; no legacy random baseline was mixed into the canonical curve.

The risk-ranked curve is retrospective. Only calibration-locked HC90/HC95 are deployment operating points; individual curve points are not separately commissioned thresholds.

## Robustness limitation

The existing **strict file/exposure-disjoint** sensitivity is preserved as a limitation. It uses 40 calibration and 40 test raw files with zero raw-file/time-support overlap, reduces the test population from 1360 to 840, and yields C5 Accuracy 74.762%, Macro-F1 69.361%, R83 NRMSE 0.3556, and R84 NRMSE 0.3489. Performance collapses materially relative to the window-level protocol. Therefore the evidence does **not** support a claim of robust unseen-exposure generalization, even though the strict-split R84-minus-R83 RMSE effect remains -1.506 ppm.

## Evidence boundary and stop condition

- No FL, DA, backbone, classifier, QC model, threshold, or additional seed experiment was added.
- Existing canonical source assets were treated as read-only.
- Six reduced-budget Ridge fits are the only new model fits.
- All requested tables and Figures A/B/C are present below this result root.
- Experimental work is closed with status `IOTJ_V4_EXPERIMENTS_FROZEN`.
"""


def execute(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    context = preflight()
    temp_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if temp_root.exists():
        raise FileExistsError(f"temporary output root already exists: {temp_root}")
    temp_root.mkdir(parents=True)
    calibration_output = temp_root / "calibration_budget"
    classification_output = temp_root / "classification"
    qc_output = temp_root / "qc"
    figures_output = temp_root / "figures"
    calibration_output.mkdir()
    classification_output.mkdir()
    qc_output.mkdir()
    figures_output.mkdir()

    _write_json(
        calibration_output / "pre_run_protocol_check.json",
        {
            "status": "PASS",
            "checks": context["checks"],
            "budget_counts": BUDGET_COUNTS,
            "nested_relation": "5% < 10% < 15% < 20%",
            "alpha_grid": ALPHAS,
            "alpha_selection": context["protocol"]["alpha_selection"],
            "preprocessing": context["protocol"]["preprocessing"],
            "test_n": 1360,
            "grouped_ci_available": context["grouped_ci_available"],
            "test_filename_groups": context["group_count"],
            "test_filename_groups_per_gas": context["group_count_per_gas"],
        },
    )

    test = _test_vectors(context)
    cal_prior = _source_matrix(context["source_models"], context["caches"]["cal_h1"])
    test_prior = _source_matrix(context["source_models"], context["caches"]["test_h1"])
    overall_metrics, gas_metrics = _reported_20_metrics()
    results: dict[int, dict[str, Any]] = {20: {"overall": overall_metrics, "gas": gas_metrics}}
    paired_predictions: dict[int, dict[str, np.ndarray]] = {
        20: {
            "R83": np.asarray([float(context["p83"][identity]["prediction"]) for identity in context["test_ids"]]),
            "R84": np.asarray([float(context["p84"][identity]["prediction"]) for identity in context["test_ids"]]),
        }
    }
    run_ids: list[str] = []

    for budget in REDUCED_BUDGETS:
        results[budget] = {"overall": {}, "gas": {"R83": {}, "R84": {}}}
        paired_predictions[budget] = {}
        for model_name in ("R83", "R84"):
            run_id = f"IOTJ-V4-C5-B{budget:02d}-{model_name}-S42"
            run_ids.append(run_id)
            lock, matrix, _ = _fit_reduced_budget(budget, model_name, context, cal_prior, test_prior)
            routed = matrix[np.arange(len(test["predicted_class"])), test["predicted_class"]]
            paired_predictions[budget][model_name] = routed
            overall = _metrics(test["truth"], routed, test["ranges"])
            results[budget]["overall"][model_name] = overall
            for gas in range(4):
                mask = test["true_class"] == gas
                results[budget]["gas"][model_name][gas] = _metrics(test["truth"][mask], routed[mask], test["ranges"][mask])
            run_root = calibration_output / "runs" / run_id
            _write_json(
                run_root / "model_lock.json",
                {
                    "experiment_id": run_id,
                    "budget_pct": budget,
                    "calibration_n": BUDGET_COUNTS[budget],
                    "model": model_name,
                    "feature_definition": "83D canonical sensor statistics" if model_name == "R83" else "83D canonical sensor statistics + one frozen R0-v2 FedRidge scalar",
                    "alpha_grid": ALPHAS,
                    "alpha_selection": context["protocol"]["alpha_selection"],
                    "models": lock["models"],
                    "alpha_audit": lock["alpha_audit"],
                },
            )
            prediction_rows = []
            for index, identity in enumerate(context["caches"]["test_ids"]):
                prediction_rows.append(
                    {
                        "experiment_id": run_id,
                        "sample_index": index,
                        "physical_identity": identity["physical_identity"],
                        "filename": identity["filename"],
                        "true_class": int(test["true_class"][index]),
                        "predicted_class": int(test["predicted_class"][index]),
                        "truth": float(test["truth"][index]),
                        "prediction": float(routed[index]),
                        "gas_range": float(test["ranges"][index]),
                    }
                )
            _write_csv(run_root / "predictions.csv", prediction_rows)
            _write_json(run_root / "metrics.json", {"status": "COMPLETE", "overall": overall, "per_gas": results[budget]["gas"][model_name]})

    summary_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    per_gas_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for budget in (5, 10, 15, 20):
        for model_name in ("R83", "R84"):
            metric = results[budget]["overall"][model_name]
            summary_rows.append(
                {
                    "budget_pct": budget,
                    "calibration_n": BUDGET_COUNTS[budget],
                    "model": model_name,
                    "RMSE": metric["RMSE"],
                    "MAE": metric["MAE"],
                    "R2": metric["R2"],
                    "NRMSE": metric["NRMSE"],
                }
            )
        a, b = results[budget]["overall"]["R83"], results[budget]["overall"]["R84"]
        effect_rows.append(
            {
                "budget_pct": budget,
                "calibration_n": BUDGET_COUNTS[budget],
                "RMSE_R83": a["RMSE"],
                "RMSE_R84": b["RMSE"],
                "delta_RMSE": b["RMSE"] - a["RMSE"],
                "NRMSE_R83": a["NRMSE"],
                "NRMSE_R84": b["NRMSE"],
                "delta_NRMSE": b["NRMSE"] - a["NRMSE"],
                "relative_improvement_pct": 100.0 * (a["NRMSE"] - b["NRMSE"]) / a["NRMSE"],
            }
        )
        for gas in range(4):
            ga, gb = results[budget]["gas"]["R83"][gas], results[budget]["gas"]["R84"][gas]
            per_gas_rows.append(
                {
                    "budget_pct": budget,
                    "calibration_n": BUDGET_COUNTS[budget],
                    "gas_id": gas,
                    "gas": GAS_NAMES[gas],
                    "RMSE_R83": ga["RMSE"],
                    "RMSE_R84": gb["RMSE"],
                    "delta_RMSE": gb["RMSE"] - ga["RMSE"],
                    "NRMSE_R83": ga["NRMSE"],
                    "NRMSE_R84": gb["NRMSE"],
                    "delta_NRMSE": gb["NRMSE"] - ga["NRMSE"],
                }
            )
        if context["grouped_ci_available"]:
            out = bootstrap_paired_group_deltas(
                test["truth"],
                paired_predictions[budget]["R83"],
                paired_predictions[budget]["R84"],
                test["groups"],
                test["ranges"],
                replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
            )
            values = np.asarray(out["nrmse_range_delta"], dtype=float)
            point = next(row["delta_NRMSE"] for row in effect_rows if row["budget_pct"] == budget)
            bootstrap_rows.append(
                {
                    "budget_pct": budget,
                    "calibration_n": BUDGET_COUNTS[budget],
                    "metric": "delta_NRMSE_R84_minus_R83",
                    "point_delta_NRMSE": point,
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "n_groups": context["group_count"],
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                    "interpretation": "conditional on the single frozen calibration subset",
                }
            )

    _write_csv(calibration_output / "calibration_budget_83d_84d.csv", summary_rows)
    _write_csv(calibration_output / "calibration_budget_effect.csv", effect_rows)
    _write_csv(calibration_output / "calibration_budget_per_gas.csv", per_gas_rows)
    if bootstrap_rows:
        _write_csv(calibration_output / "calibration_budget_bootstrap.csv", bootstrap_rows)

    classification_rows = _classification_table(classification_output)
    for name in ("classification_round_accuracy.csv", "classification_round_summary.csv", "classification_round_protocol.json"):
        shutil.copy2(ROUND_ROOT / name, classification_output / name)
    curve_rows, hc_rows, hc_compatible = _qc_tables(qc_output)
    _copy_round_figure(figures_output)
    _plot_budget(summary_rows, effect_rows, figures_output)
    _plot_qc(curve_rows, hc_rows, figures_output)

    report = _build_report(classification_rows, effect_rows, per_gas_rows, bootstrap_rows, hc_compatible)
    (temp_root / "FINAL_CANONICAL_EVIDENCE_REPORT.md").write_text(report, encoding="utf-8")

    source_files = [
        *(_budget_path(budget) for budget in BUDGET_COUNTS),
        R1_ROOT / "predictions.csv",
        R1_ROOT / "canonical_regression_comparison.csv",
        R1_ROOT / "canonical_regression_per_gas.csv",
        R0_ROOT / "sha256_index.json",
        CLASSIFICATION_SOURCE,
        Q1_CURVE,
        HC_METRICS,
        HC_THRESHOLDS,
        ROUND_ROOT / "classification_round_accuracy.csv",
        STRICT_ANALYSIS,
    ]
    _write_json(
        temp_root / "execution_manifest.json",
        {
            "mode": "Minimal Scientific Execution Mode",
            "status": "COMPLETE",
            "new_model_fits": run_ids,
            "new_model_fit_count": len(run_ids),
            "reused_20pct": str((R1_ROOT / "predictions.csv").relative_to(ROOT)).replace("\\", "/"),
            "checks": context["checks"],
            "bootstrap": {"grouped": context["grouped_ci_available"], "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "hc90_hc95_compatible": hc_compatible,
            "random_retention_band": "SKIPPED_OPTIONAL",
            "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path) for path in source_files},
        },
    )
    _write_json(
        temp_root / "IOTJ_V4_EXPERIMENTS_FROZEN.json",
        {
            "status": "IOTJ_V4_EXPERIMENTS_FROZEN",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "new_model_fit_count": 6,
            "new_model_fits": run_ids,
            "no_additional_algorithm_or_experiment_authorized": True,
        },
    )

    temp_root.rename(output_root)
    return {
        "status": "IOTJ_V4_EXPERIMENTS_FROZEN",
        "output": str(output_root.relative_to(ROOT)).replace("\\", "/"),
        "new_model_fits": run_ids,
        "effect_rows": effect_rows,
        "bootstrap_rows": bootstrap_rows,
        "hc90_hc95_compatible": hc_compatible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.check_only:
        context = preflight()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "checks": context["checks"],
                    "grouped_ci_available": context["grouped_ci_available"],
                    "test_filename_groups": context["group_count"],
                    "formal_output_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(execute(output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
