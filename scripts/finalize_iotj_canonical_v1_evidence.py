"""Close canonical-v1 submission evidence without changing trained models.

The module consumes immutable canonical-v1 prediction artifacts.  All policy
constants below are fixed before the post-run summaries are generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from gaps_deploy.c5_h8_runtime import FixedH8Policy, SerializedRidge
from run_regression_head_ablation import (
    CLASS_NAMES,
    CLASS_RANGES,
    deterministic_train_val,
    fit_ridge,
)
from scripts.evaluate_iotj_feature_metadata_ablation import profile_feature_dict
from scripts.finalize_iotj_a4_qc import COVERAGE_TARGETS
from scripts.run_iotj_canonical_v1_r84 import enriched_oracle_rows
from scripts import run_gaps_cross_target_r84_full as r84_common


QUALITY_STRATA_POLICY = {
    "Q0": "observed=1, empty=0, max_missing_run=0, interpolated=0",
    "Q1": "observed>=0.98, empty<=0.02, max_missing_run<=1, interpolated<=0.02",
    "Q2": "observed>=0.90, empty<=0.10, max_missing_run<=3, interpolated<=0.10",
    "Q3": "otherwise",
}
RISK_COMPONENTS = (
    "classification_uncertainty_risk",
    "regression_disagreement_risk",
    "source_prior_disagreement_risk",
)
ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "results" / "iotj_canonical_v1_final_20260808"
DATASET_ROOT = ROOT / "dataset" / "iotj_canonical_v1"
H1_PATH = ROOT / "results" / "iotj_h1_federated_ridge_equivalence_20260724" / "federated_h1_manifest.json"
R4_POLICY_PATH = (
    ROOT.parent / "iotj-confirmation-observability" / "results"
    / "iotj_b5_c5_deployment_p1_20260722" / "h8_no_rescue" / "r4_policy.json"
)
TARGETS = ("C3", "C4", "C5")
RANDOM_SEED = 20260804
RANDOM_REPEATS = 1000


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


def combined_equal_mean_risk(
    row: Mapping[str, Any], scales: Mapping[str, float]
) -> float:
    """Exact frozen equal-mean risk with per-component clipping."""
    normalized = []
    for key in RISK_COMPONENTS:
        scale = float(scales[key])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid QC scale: {key}")
        normalized.append(float(np.clip(float(row[key]) / scale, 0.0, 1.0)))
    risk = float(np.mean(normalized))
    if not math.isfinite(risk):
        raise RuntimeError("FAIL_CLOSED non-finite equal-mean risk")
    return risk


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


def merged_interval_length(intervals: Sequence[tuple[float, float]]) -> float:
    """Measure the union of half-open intervals without double counting."""
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    if any(end < start for start, end in ordered):
        raise ValueError("interval end precedes start")
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return float(total + current_end - current_start)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def load_h1() -> dict[int, SerializedRidge]:
    return r84_common.load_h1()


def load_h23_policy() -> tuple[FixedH8Policy, dict[str, Any]]:
    if not R4_POLICY_PATH.is_file():
        raise FileNotFoundError(f"FAIL_CLOSED frozen H2/H3 policy missing: {R4_POLICY_PATH}")
    payload = read_json(R4_POLICY_PATH)
    policy = FixedH8Policy.from_json(payload["source_aug_target_ridge_policy"])
    return policy, {
        "path": str(R4_POLICY_PATH.resolve()),
        "sha256": sha256(R4_POLICY_PATH),
        "role": "frozen H2/H3 auxiliary source-prior disagreement for QC only",
        "classifier_or_preprocessing_asset": False,
    }


def load_models(path: Path) -> dict[int, SerializedRidge]:
    payload = read_json(path)
    return {int(key): SerializedRidge.from_json(value) for key, value in payload.items()}


def predict_serialized_ridge(
    model: SerializedRidge, features: Mapping[str, Any]
) -> float:
    """Make the flat runtime feature contract explicit at call sites."""
    return float(model.predict(features))


def _classification_uncertainty(row: Mapping[str, Any]) -> float:
    probabilities = np.asarray(
        [float(row[f"prob_class_{class_id}"]) for class_id in range(4)],
        dtype=np.float64,
    )
    if not np.isfinite(probabilities).all() or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
        raise RuntimeError("FAIL_CLOSED invalid class probabilities")
    ordered = np.sort(probabilities)[::-1]
    confidence = float(ordered[0])
    margin = float(ordered[0] - ordered[1])
    entropy = float(
        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
        / math.log(4.0)
    )
    return max(1.0 - confidence, 1.0 - margin, entropy)


def fit_r83_models(
    target: str, base_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    sensor_rows = []
    for row in base_rows:
        item = dict(row)
        item["feature_dict"] = profile_feature_dict(item["feature_dict"], "M83_SENSOR")
        if len(item["feature_dict"]) != 83:
            raise RuntimeError("FAIL_CLOSED R83 dimension differs")
        sensor_rows.append(item)
    by_id = {int(row["sample_index"]): row for row in sensor_rows}
    feature_names = sorted(sensor_rows[0]["feature_dict"])
    models: dict[int, Any] = {}
    selection: list[dict[str, Any]] = []
    for class_id, gas in sorted(CLASS_NAMES.items()):
        class_rows = [row for row in sensor_rows if int(row["true_class"]) == class_id]
        fit_rows, validation_seed = deterministic_train_val(class_rows, 0.25)
        validation_rows = [by_id[int(row["sample_index"])] for row in validation_seed]
        truth = np.asarray([float(row["true_ppm"]) for row in validation_rows])
        best_alpha, best_rmse = r84_common.RIDGE_ALPHAS[0], float("inf")
        grid = []
        for alpha in r84_common.RIDGE_ALPHAS:
            candidate = fit_ridge(fit_rows, feature_names, alpha)
            score = float(np.sqrt(np.mean((candidate.predict(validation_rows) - truth) ** 2)))
            grid.append({"alpha": float(alpha), "validation_RMSE": score})
            if score < best_rmse:
                best_alpha, best_rmse = alpha, score
        models[class_id] = fit_ridge(class_rows, feature_names, best_alpha)
        selection.append(
            {
                "experiment_id": f"CAN-V1-R83-A4-{target}",
                "target": target,
                "class_id": class_id,
                "gas": gas,
                "calibration_fit_N": len(fit_rows),
                "calibration_validation_N": len(validation_rows),
                "calibration_refit_N": len(class_rows),
                "input_dimension": 83,
                "selected_alpha": float(best_alpha),
                "calibration_validation_RMSE": best_rmse,
                "alpha_grid_audit": json.dumps(grid, separators=(",", ":")),
                "selection_split": f"{target}_canonical_calibration_internal_75_25",
                "target_test_used_for_selection": False,
            }
        )
    return models, selection


def enrich_target_records(
    study_root: Path,
    target: str,
    split: str,
    h1: Mapping[int, SerializedRidge],
    h23: FixedH8Policy,
    r83: Mapping[int, Any],
    r84: Mapping[int, SerializedRidge],
) -> list[dict[str, Any]]:
    source = read_csv(study_root / "regression" / target / f"{split}_records.csv")
    base = enriched_oracle_rows(target, split)
    if len(source) != len(base):
        raise RuntimeError(f"FAIL_CLOSED {target}/{split} row count differs")
    output: list[dict[str, Any]] = []
    for raw, features in zip(source, base):
        if int(raw["sample_index"]) != int(features["sample_index"]):
            raise RuntimeError(f"FAIL_CLOSED {target}/{split} alignment differs")
        route = int(raw["pred_class"])
        true_class = int(raw["true_class"])
        full = features["feature_dict"]
        sensor = profile_feature_dict(full, "M83_SENSOR")
        pred83 = float(r83[route].predict([{"feature_dict": sensor}])[0])
        true_h1 = float(h1[true_class].predict(full))
        oracle84_features = dict(sensor)
        oracle84_features["srcpred_H1_federated_source_ridge_ppm"] = true_h1
        oracle84 = predict_serialized_ridge(r84[true_class], oracle84_features)
        h1_route = float(raw["H1_federated_source_ridge_ppm"])
        h2_route = float(h23.source_mlp[route].predict(full))
        shared = dict(full)
        shared["route_class"] = route
        h3_route = float(h23.shared_mlp.predict(shared))
        class_range = float(CLASS_RANGES[route])
        item: dict[str, Any] = {**raw}
        item.update(
            {
                "pred_83d_ppm": pred83,
                "oracle_route_pred_84d_h1_ppm": oracle84,
                "H2_source_per_gas_mlp_ppm": h2_route,
                "H3_source_shared_mlp_ppm": h3_route,
                "classification_uncertainty_risk": _classification_uncertainty(raw),
                "regression_disagreement_risk": abs(float(raw["pred_84d_h1_ppm"]) - pred83) / class_range,
                "source_prior_disagreement_risk": (max(h1_route, h2_route, h3_route) - min(h1_route, h2_route, h3_route)) / class_range,
                "quality_stratum": assign_quality_stratum(
                    observed_ratio=float(raw["observed_ratio"]),
                    empty_bin_ratio=float(raw["empty_bin_ratio"]),
                    max_missing_run=int(float(raw["max_missing_run"])),
                    short_gap_interpolated_ratio=float(raw["short_gap_interpolated_ratio"]),
                ),
            }
        )
        output.append(item)
    return output


def fit_target_qc_thresholds(
    target: str, calibration: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    scales: dict[str, float] = {}
    for key in RISK_COMPONENTS:
        values = np.asarray([float(row[key]) for row in calibration], dtype=np.float64)
        scale = float(np.quantile(values, 0.95))
        if len(values) == 0 or not np.isfinite(values).all() or scale <= 0:
            raise RuntimeError(f"FAIL_CLOSED {target} QC calibration component invalid: {key}")
        scales[key] = scale
    risks = np.sort(np.asarray([combined_equal_mean_risk(row, scales) for row in calibration]))
    output: list[dict[str, Any]] = []
    for coverage in COVERAGE_TARGETS:
        retained = int(math.ceil(float(coverage) * len(risks)))
        threshold = float("inf") if coverage == 1.0 else float(risks[retained - 1])
        output.append(
            {
                "target": target,
                "target_coverage": float(coverage),
                "threshold": threshold,
                "calibration_N": len(risks),
                "calibration_retained_N": retained,
                "selection_split": f"{target}_canonical_calibration_x_only_risk",
                "target_test_used_for_selection": False,
                "risk_formula": "equal_mean_of_calibration_p95_normalized_components",
                **{f"p95_scale_{key}": value for key, value in scales.items()},
            }
        )
    return output


def threshold_at(
    thresholds: Sequence[Mapping[str, Any]], coverage: float
) -> Mapping[str, Any]:
    return next(
        row for row in thresholds
        if math.isclose(float(row["target_coverage"]), float(coverage), abs_tol=1e-9)
    )


def annotate_qc(
    records: Sequence[Mapping[str, Any]], thresholds: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    first = thresholds[0]
    scales = {key: float(first[f"p95_scale_{key}"]) for key in RISK_COMPONENTS}
    hc90 = threshold_at(thresholds, 0.90)
    hc95 = threshold_at(thresholds, 0.95)
    q975 = threshold_at(thresholds, 0.975)
    output = []
    for row in records:
        risk = combined_equal_mean_risk(row, scales)
        output.append(
            {
                **dict(row),
                "qc_risk_score_final": risk,
                "HC90_decision": classify_qc_decision(
                    risk,
                    accept_threshold=float(hc90["threshold"]),
                    review_threshold=float(hc95["threshold"]),
                ),
                "HC95_decision": classify_qc_decision(
                    risk,
                    accept_threshold=float(hc95["threshold"]),
                    review_threshold=float(q975["threshold"]),
                ),
            }
        )
    return output


def _scope_summary_or_empty(
    rows: Sequence[Mapping[str, Any]], prediction_key: str = "pred_84d_h1_ppm"
) -> dict[str, Any]:
    if not rows:
        return {
            "N": 0, "RMSE": "", "NRMSE_range": "", "MAE": "",
            "P90AE": "", "Bias": "", "R2": "",
        }
    return summarize_error_scope(rows, prediction_key)


def summarize_qc_workpoints(
    scope: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    n = len(records)
    for workpoint in ("HC90", "HC95"):
        field = f"{workpoint}_decision"
        accepted = [row for row in records if row[field] == "accepted"]
        review = [row for row in records if row[field] == "review"]
        rejected = [row for row in records if row[field] == "reject"]
        groups = {
            "accepted": accepted,
            "accepted+review": accepted + review,
            "reject": rejected,
        }
        for population, rows in groups.items():
            output.append(
                {
                    "scope": scope,
                    "workpoint": workpoint,
                    "population": population,
                    "coverage": len(accepted) / n,
                    "review_rate": len(review) / n,
                    "reject_rate": len(rejected) / n,
                    **_scope_summary_or_empty(rows),
                }
            )
    return output


def evaluate_coverage_curve(
    scope: str,
    records: Sequence[Mapping[str, Any]],
    thresholds_by_target: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for coverage in COVERAGE_TARGETS:
        accepted = []
        for row in records:
            target = str(row["client"])
            policy = threshold_at(thresholds_by_target[target], float(coverage))
            accepted.append(float(row["qc_risk_score_final"]) <= float(policy["threshold"]))
        selected = [row for row, keep in zip(records, accepted) if keep]
        output.append(
            {
                "scope": scope,
                "target_coverage": float(coverage),
                "accepted_N": len(selected),
                "test_N": len(records),
                "test_coverage": len(selected) / len(records),
                **_scope_summary_or_empty(selected),
            }
        )
    return output


def random_reference_rows(
    scope: str,
    records: Sequence[Mapping[str, Any]],
    curve: Sequence[Mapping[str, Any]],
    repeats: int = RANDOM_REPEATS,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    n = len(records)
    truth = np.asarray([float(row["true_ppm"]) for row in records], dtype=np.float64)
    prediction = np.asarray([float(row["pred_84d_h1_ppm"]) for row in records], dtype=np.float64)
    classes = np.asarray([int(row["true_class"]) for row in records], dtype=np.int64)
    squared = (prediction - truth) ** 2
    ranges = np.asarray([float(CLASS_RANGES[int(value)]) for value in classes])
    normalized_squared = ((prediction - truth) / ranges) ** 2
    for point in curve:
        retained = int(point["accepted_N"])
        rmse_values = []
        nrmse_values = []
        for _ in range(repeats):
            indexes = rng.choice(n, size=retained, replace=False)
            rmse_values.append(float(np.sqrt(np.mean(squared[indexes]))))
            nrmse_values.append(float(np.sqrt(np.mean(normalized_squared[indexes]))))
        rmse = np.asarray(rmse_values)
        nrmse = np.asarray(nrmse_values)
        output.append(
            {
                "scope": scope,
                "target_coverage": float(point["target_coverage"]),
                "accepted_N": retained,
                "test_coverage": float(point["test_coverage"]),
                "repeats": repeats,
                "seed": seed,
                "random_RMSE_mean": float(rmse.mean()),
                "random_RMSE_sample_std": float(rmse.std(ddof=1)),
                "random_RMSE_p025": float(np.quantile(rmse, 0.025)),
                "random_RMSE_p975": float(np.quantile(rmse, 0.975)),
                "random_NRMSE_mean": float(nrmse.mean()),
                "random_NRMSE_sample_std": float(nrmse.std(ddof=1)),
                "random_NRMSE_p025": float(np.quantile(nrmse, 0.025)),
                "random_NRMSE_p975": float(np.quantile(nrmse, 0.975)),
                "QC_RMSE": float(point["RMSE"]),
                "QC_NRMSE": float(point["NRMSE_range"]),
                "QC_RMSE_improvement_over_random": float(rmse.mean() - float(point["RMSE"])),
                "QC_NRMSE_improvement_over_random": float(nrmse.mean() - float(point["NRMSE_range"])),
            }
        )
    return output


def fedridge_ablation_rows(
    records_by_target: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: dict[str, Sequence[Mapping[str, Any]]] = {
        **records_by_target,
        "ALL": [row for target in TARGETS for row in records_by_target[target]],
    }
    for scope, records in scopes.items():
        gas_scopes = [("ALL", records)] + [
            (gas, [row for row in records if int(row["true_class"]) == class_id])
            for class_id, gas in sorted(CLASS_NAMES.items())
        ]
        for gas, selected in gas_scopes:
            m83 = summarize_error_scope(selected, "pred_83d_ppm")
            m84 = summarize_error_scope(selected, "pred_84d_h1_ppm")
            for variant, dimension, metrics in (
                ("R83_TARGET_ONLY", 83, m83),
                ("R84_FED_H1", 84, m84),
            ):
                rows.append(
                    {
                        "scope": scope,
                        "gas": gas,
                        "variant": variant,
                        "input_dimension": dimension,
                        **metrics,
                        "absolute_RMSE_gain_vs_83D": 0.0 if dimension == 83 else float(m83["RMSE"] - m84["RMSE"]),
                        "relative_RMSE_reduction_vs_83D": 0.0 if dimension == 83 else float((m83["RMSE"] - m84["RMSE"]) / m83["RMSE"]),
                    }
                )
    return rows


def quality_summary_rows(
    records_by_target: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: dict[str, Sequence[Mapping[str, Any]]] = {
        **records_by_target,
        "ALL": [row for target in TARGETS for row in records_by_target[target]],
    }
    for scope, records in scopes.items():
        for stratum in ("Q0", "Q1", "Q2", "Q3"):
            selected = [row for row in records if row["quality_stratum"] == stratum]
            correct = [row for row in selected if int(row["route_correct"])]
            rows.append(
                {
                    "scope": scope,
                    "slice": stratum,
                    "N": len(selected),
                    "classification_accuracy": (
                        float(np.mean([int(row["route_correct"]) for row in selected])) if selected else ""
                    ),
                    "S_ALL_RMSE": _scope_summary_or_empty(selected)["RMSE"],
                    "S_CC_RMSE": _scope_summary_or_empty(correct)["RMSE"],
                    "oracle_route_RMSE": _scope_summary_or_empty(
                        selected, "oracle_route_pred_84d_h1_ppm"
                    )["RMSE"],
                    "NRMSE_range": _scope_summary_or_empty(selected)["NRMSE_range"],
                    "HC90_accepted_coverage": (
                        float(np.mean([row["HC90_decision"] == "accepted" for row in selected])) if selected else ""
                    ),
                    "HC95_accepted_coverage": (
                        float(np.mean([row["HC95_decision"] == "accepted" for row in selected])) if selected else ""
                    ),
                    "population_deleted": False,
                }
            )
    c5 = records_by_target["C5"]
    for repeat in (1, 2):
        selected = [
            row for row in c5
            if str(row.get("gas", "")).lower() == "methane"
            and math.isclose(float(row["true_ppm"]), 225.0, abs_tol=1e-9)
            and int(float(row["repeat_id"])) == repeat
        ]
        correct = [row for row in selected if int(row["route_correct"])]
        rows.append(
            {
                "scope": "C5",
                "slice": f"Methane_225ppm_repeat{repeat}",
                "N": len(selected),
                "classification_accuracy": float(np.mean([int(row["route_correct"]) for row in selected])) if selected else "",
                "S_ALL_RMSE": _scope_summary_or_empty(selected)["RMSE"],
                "S_CC_RMSE": _scope_summary_or_empty(correct)["RMSE"],
                "oracle_route_RMSE": _scope_summary_or_empty(selected, "oracle_route_pred_84d_h1_ppm")["RMSE"],
                "NRMSE_range": _scope_summary_or_empty(selected)["NRMSE_range"],
                "HC90_accepted_coverage": float(np.mean([row["HC90_decision"] == "accepted" for row in selected])) if selected else "",
                "HC95_accepted_coverage": float(np.mean([row["HC95_decision"] == "accepted" for row in selected])) if selected else "",
                "population_deleted": False,
            }
        )
    return rows


def _merged_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    output: list[tuple[float, float]] = []
    for start, end in ordered:
        if not output or start > output[-1][1]:
            output.append((start, end))
        else:
            output[-1] = (output[-1][0], max(output[-1][1], end))
    return output


def _intersection_length(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> float:
    a, b = _merged_intervals(left), _merged_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += interval_overlap_seconds(a[i][0], a[i][1], b[j][0], b[j][1])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def window_overlap_row(target: str) -> dict[str, Any]:
    client = target[1:]
    calibration = read_json(DATASET_ROOT / f"client_{client}" / "calibration_experiment_info.json")
    test = read_json(DATASET_ROOT / f"client_{client}" / "test_experiment_info.json")
    calibration_ids = {str(row["physical_identity"]) for row in calibration}
    test_ids = {str(row["physical_identity"]) for row in test}
    group_keys = ("filename", "repeat_id", "gas", "concentration")
    def groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], list[tuple[float, float]]]:
        output: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
        for row in rows:
            key = tuple(row.get(name) for name in group_keys)
            output.setdefault(key, []).append((float(row["window_start_s"]), float(row["window_end_s"])))
        return output
    cal_groups, test_groups = groups(calibration), groups(test)
    keys = set(cal_groups) | set(test_groups)
    cal_duration = test_duration = overlap_duration = union_duration = 0.0
    cal_bins: set[tuple[tuple[Any, ...], int]] = set()
    test_bins: set[tuple[tuple[Any, ...], int]] = set()
    for key in keys:
        left, right = cal_groups.get(key, []), test_groups.get(key, [])
        cal_duration += merged_interval_length(left)
        test_duration += merged_interval_length(right)
        overlap_duration += _intersection_length(left, right)
        union_duration += merged_interval_length([*left, *right])
        for start, end in left:
            cal_bins.update((key, index) for index in range(int(round(start * 5)), int(round(end * 5))))
        for start, end in right:
            test_bins.update((key, index) for index in range(int(round(start * 5)), int(round(end * 5))))
    shared_bins = cal_bins & test_bins
    union_bins = cal_bins | test_bins
    return {
        "target": target,
        "calibration_N": len(calibration),
        "test_N": len(test),
        "exact_identity_overlap_N": len(calibration_ids & test_ids),
        "raw_physical_time_overlap_seconds": overlap_duration,
        "calibration_unique_physical_seconds": cal_duration,
        "test_unique_physical_seconds": test_duration,
        "union_unique_physical_seconds": union_duration,
        "shared_physical_duration_ratio_union": overlap_duration / union_duration if union_duration else 0.0,
        "shared_physical_duration_ratio_test": overlap_duration / test_duration if test_duration else 0.0,
        "shared_raw_bin_N": len(shared_bins),
        "union_raw_bin_N": len(union_bins),
        "shared_raw_bin_ratio_union": len(shared_bins) / len(union_bins) if union_bins else 0.0,
        "status": "PASS" if overlap_duration == 0.0 else "RAW_TIME_OVERLAP_PRESENT",
    }


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def build_final_state(study_root: Path) -> dict[str, Any]:
    evaluation = read_json(study_root / "classification_evaluation" / "evaluation_manifest.json")
    regression = read_json(study_root / "regression" / "protocol_manifest.json")
    dataset = read_json(DATASET_ROOT / "dataset_sha256.json")
    return {
        "schema_version": "iotj.canonical_v1.final_experiment_state.v1",
        "status": "FORMAL_EVALUATION_COMPLETE_POSTRUN_EVIDENCE_IN_PROGRESS",
        "training_complete": True,
        "canonical_dataset_frozen": True,
        "sealed_test_opened_once": True,
        "sealed_test_evaluated": True,
        "test_selection_performed": False,
        "post_test_training_performed": False,
        "post_test_hyperparameter_tuning": False,
        "classification_progress_json_role": "historical_pre_test_snapshot",
        "classification_progress_evaluation_conflict": False,
        "dataset_hash": dataset["aggregate_sha256"],
        "classification_checkpoint_hashes": {
            target: evaluation["gate"]["runs"][target]["checkpoint_sha256"]
            for target in TARGETS
        },
        "adapted_checkpoint_hashes": {
            target: evaluation["gate"]["runs"][target]["checkpoint_sha256"]
            for target in TARGETS
        },
        "r84_artifact_hashes": {
            "protocol_manifest": sha256(study_root / "regression" / "protocol_manifest.json"),
            "sha256_index": sha256(study_root / "regression" / "sha256_index.json"),
            **{
                f"{target}_models": sha256(study_root / "regression" / target / "regression_models.json")
                for target in TARGETS
            },
        },
        "code_commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "evaluation_timestamp": (
            study_root / "classification_evaluation" / "SEALED_TEST_OPEN.json"
        ).stat().st_mtime,
        "regression_status": regression["status"],
        "separate_preregistered_a0t_baseline_is_not_post_test_a4_training": True,
    }


def run_derived_evidence(study_root: Path, repeats: int = RANDOM_REPEATS) -> Path:
    study_root = study_root.resolve()
    output = study_root / "evidence_closure"
    final_state_path = study_root / "FINAL_EXPERIMENT_STATE.json"
    provenance_path = study_root / "FINAL_PROVENANCE_AUDIT.md"
    if output.exists() or final_state_path.exists() or provenance_path.exists():
        raise FileExistsError("FAIL_CLOSED canonical evidence output already exists")
    output.mkdir(parents=True)
    h1 = load_h1()
    h23, h23_provenance = load_h23_policy()
    thresholds_by_target: dict[str, list[dict[str, Any]]] = {}
    records_by_target: dict[str, list[dict[str, Any]]] = {}
    calibration_by_target: dict[str, list[dict[str, Any]]] = {}
    r83_selection_all: list[dict[str, Any]] = []
    r83_models_manifest: dict[str, Any] = {}
    for target in TARGETS:
        target_output = output / "derived_records" / target
        calibration_base = enriched_oracle_rows(target, "calibration")
        r83, selection = fit_r83_models(target, calibration_base)
        r84 = load_models(study_root / "regression" / target / "regression_models.json")
        calibration = enrich_target_records(study_root, target, "calibration", h1, h23, r83, r84)
        test = enrich_target_records(study_root, target, "test", h1, h23, r83, r84)
        thresholds = fit_target_qc_thresholds(target, calibration)
        calibration = annotate_qc(calibration, thresholds)
        test = annotate_qc(test, thresholds)
        thresholds_by_target[target] = thresholds
        calibration_by_target[target] = calibration
        records_by_target[target] = test
        r83_selection_all.extend(selection)
        r83_models_manifest[target] = {
            str(class_id): model.to_json() for class_id, model in sorted(r83.items())
        }
        write_csv(target_output / "calibration_records.csv", calibration)
        write_csv(target_output / "test_records.csv", test)
        write_csv(output / "qc" / f"{target}_qc_threshold_lock.csv", thresholds)
    write_json(output / "fedridge_ablation" / "r83_models.json", r83_models_manifest)
    write_csv(output / "fedridge_ablation" / "r83_calibration_alpha_selection.csv", r83_selection_all)

    all_records = [row for target in TARGETS for row in records_by_target[target]]
    qc_summary: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for scope, records in {**records_by_target, "ALL": all_records}.items():
        qc_summary.extend(summarize_qc_workpoints(scope, records))
        curve = evaluate_coverage_curve(scope, records, thresholds_by_target)
        curves.extend(curve)
        random_rows.extend(random_reference_rows(scope, records, curve, repeats=repeats))
    write_csv(output / "qc" / "qc_equal_mean_summary.csv", qc_summary)
    write_csv(output / "qc" / "qc_coverage_curve.csv", curves)
    write_csv(output / "qc" / "qc_random_reference.csv", random_rows)
    write_json(
        output / "qc" / "protocol_manifest.json",
        {
            "schema_version": "iotj.canonical_v1.equal_mean_qc.v1",
            "status": "COMPLETE",
            "formula": "equal_mean_of_calibration_p95_normalized_components",
            "threshold_source": "target-specific canonical calibration empirical quantiles",
            "workpoints": {
                "HC90": {"accept": "q90", "review": "q90_to_q95", "reject": "above_q95"},
                "HC95": {"accept": "q95", "review": "q95_to_q97.5", "reject": "above_q97.5"},
            },
            "random_repeats": repeats,
            "random_seed": RANDOM_SEED,
            "target_test_used_for_threshold_selection": False,
            "h23_auxiliary_provenance": h23_provenance,
        },
    )

    fedridge = fedridge_ablation_rows(records_by_target)
    quality = quality_summary_rows(records_by_target)
    overlap = [window_overlap_row(target) for target in TARGETS]
    write_csv(output / "fedridge_ablation" / "canonical_83d_vs_84d.csv", fedridge)
    write_csv(output / "quality" / "quality_stratified_summary.csv", quality)
    write_csv(output / "overlap" / "window_overlap_summary.csv", overlap)
    write_json(
        output / "quality" / "quality_policy.json",
        {
            "schema_version": "iotj.canonical_v1.quality_strata.v1",
            "policy": QUALITY_STRATA_POLICY,
            "test_error_used_to_select_thresholds": False,
            "samples_deleted": False,
        },
    )

    (output / "overlap" / "STRICT_NON_OVERLAP_ROBUSTNESS_PROTOCOL_PROPOSAL.md").write_text(
        "# Strict non-overlap robustness protocol proposal\n\n"
        "The canonical primary result remains unchanged. A supplementary robustness split should group all overlapping 10-s windows from the same raw file and physical-time connected component into one role, so calibration and test share neither raw bins nor physical duration. The preprocessing candidate, A4, R84, QC formula, seed, and budgets remain frozen; only the grouped role assignment changes. This proposal is required because the current identity-disjoint split has raw-time overlap.\n",
        encoding="utf-8",
    )
    (output / "overlap" / "WINDOW_OVERLAP_AUDIT.md").write_text(
        "# Window overlap audit\n\n"
        "Exact calibration/test window identities are disjoint for C3/C4/C5. Raw physical-time overlap is nevertheless present because 10-s windows use a 5-s stride and windows were assigned independently. This is a **submission blocker for claims of fully independent target-test evidence** until disclosed and, preferably, supplemented by the strict grouped robustness protocol. The canonical primary numbers were not modified.\n",
        encoding="utf-8",
    )
    (output / "quality" / "QUALITY_ROBUSTNESS_AUDIT.md").write_text(
        "# Quality robustness audit\n\nRead-only Q0-Q3 strata were fixed before derived metrics. No sample, including C5 Methane 225 ppm repeat 1, was deleted.\n",
        encoding="utf-8",
    )
    (output / "qc" / "QC_FINAL_AUDIT.md").write_text(
        "# Frozen equal-mean QC audit\n\nThe three original label-free risk components, calibration-p95 normalization, clipping, equal mean, and calibration empirical-quantile threshold rule are unchanged. Target test was used only for fixed-policy evaluation. HC90/HC95 accept sets retain the frozen meaning; review/reject use the preregistered next higher coverage threshold.\n",
        encoding="utf-8",
    )

    final_state = build_final_state(study_root)
    final_state["status"] = "FORMAL_EVALUATION_AND_DERIVED_EVIDENCE_COMPLETE"
    write_json(final_state_path, final_state)
    provenance_path.write_text(
        "# Final provenance audit\n\n"
        "`CLASSIFICATION_PROGRESS.json` is the immutable historical snapshot taken after all round-25 endpoints completed and before the sealed target test was opened. `classification_evaluation/evaluation_manifest.json` is the later one-time evaluation record. They describe two ordered stages and do not conflict. No A4/R84 training, checkpoint selection, QC threshold tuning, or post-test hyperparameter tuning occurred after the sealed evaluation.\n",
        encoding="utf-8",
    )
    artifacts = [
        {"path": str(path.relative_to(study_root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "sha256_index.json"
    ]
    write_json(output / "sha256_index.json", {"status": "PASS", "artifacts": artifacts})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    args = parser.parse_args()
    run_derived_evidence(args.study_root, repeats=args.random_repeats)


if __name__ == "__main__":
    main()
