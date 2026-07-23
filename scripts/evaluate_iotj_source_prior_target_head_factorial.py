"""Run the formal C5 source-prior x target-head factorial ablation.

The experiment reuses frozen runtime-v4 B5 and pooled H1/H2/H3 assets.  It
does not modify runtime/QC and deliberately freezes the calibration decision
gate before loading C5 test inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_deploy.c5_h8_runtime import C5H8Runtime
from run_formal_target_mlp_auto_v2_eval import MLPHead, fit_mlp
from run_regression_head_ablation import (
    CLASS_NAMES,
    CLASS_RANGES,
    build_oracle_rows,
    deterministic_train_val,
    fit_ridge,
)


SCHEMA_VERSION = "iotj.source_prior_target_head_factorial.v1"
MODEL_SELECTION_SPLIT = "calibration_validation"
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MLP_ALPHAS = (0.001, 0.01, 0.1, 1.0)
MLP_HIDDEN_GRID = ((16,), (32,), (64,), (32, 16))
SOURCE_KEYS = (
    "H1_source_ridge_ppm",
    "H2_source_per_gas_mlp_ppm",
    "H3_source_shared_mlp_ppm",
)
VARIANTS: dict[str, dict[str, Any]] = {
    "E1_RIDGE_RICH": {"head": "ridge", "source_keys": ()},
    "E2_RIDGE_H1": {"head": "ridge", "source_keys": (SOURCE_KEYS[0],)},
    "E2_RIDGE_H2": {"head": "ridge", "source_keys": (SOURCE_KEYS[1],)},
    "E2_RIDGE_H3": {"head": "ridge", "source_keys": (SOURCE_KEYS[2],)},
    "E1_RIDGE_PRIOR": {"head": "ridge", "source_keys": SOURCE_KEYS},
    "E1_MLP_RICH": {"head": "mlp", "source_keys": ()},
    "E1_MLP_PRIOR": {"head": "mlp", "source_keys": SOURCE_KEYS},
}
FROZEN_BASELINE_VARIANTS = {
    "E1_RIDGE_RICH",
    "E1_RIDGE_PRIOR",
    "E1_MLP_RICH",
}
NEW_MODEL_VARIANTS = {
    "E2_RIDGE_H1",
    "E2_RIDGE_H2",
    "E2_RIDGE_H3",
    "E1_MLP_PRIOR",
}
FORMAL_OUTPUT_FILES = (
    "protocol_manifest.json",
    "asset_audit.json",
    "calibration_selection.csv",
    "calibration_validation_predictions.csv",
    "calibration_validation_summary.csv",
    "decision_gate.json",
    "target_head_manifest.json",
    "test_predictions.csv",
    "test_summary.csv",
    "per_gas_summary.csv",
    "factorial_effects.csv",
    "README.md",
)
FROZEN_ASSETS = (
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/row_map_1360.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/parity_report.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/runtime_rows.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/parity_report.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/runtime_rows.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/r4_policy.json",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/manifest.json",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/target_validation_rich_only.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/target_validation_plus_source_preds.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/validation_fit_audit.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/target_predictions_rich_only.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/target_predictions_plus_source_preds.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h23_plus/manifest.json",
    "results/iotj_b5_c5_deployment_p1_20260722/h23_plus/h23_reference.json",
    "results/iotj_b5_c5_deployment_p1_20260722/h23_plus/c5_h23_plus_validation_predictions.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h23_plus/c5_h23_plus_test_predictions.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/h23_plus/c5_h23_plus_fit_audit.csv",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def origin_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "origin/codex/iotj-confirmation-observability"],
        text=True,
    ).strip()


def frozen_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in FROZEN_ASSETS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen asset: {relative}")
        result[relative] = sha256_file(path)
    return result


def require_new_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def feature_names_for(
    rows: Sequence[Mapping[str, Any]], source_keys: Sequence[str]
) -> list[str]:
    names = sorted(rows[0]["feature_dict"])
    return [
        *names,
        *(f"srcpred_{key}" for key in source_keys),
    ]


def add_source_features(
    rows: Sequence[Mapping[str, Any]], source_keys: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        features = dict(item["feature_dict"])
        for key in source_keys:
            value = float(item[key])
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite frozen source prior: {key}")
            features[f"srcpred_{key}"] = value
        item["feature_dict"] = features
        output.append(item)
    expected = feature_names_for(rows, source_keys)
    if any(sorted(row["feature_dict"]) != sorted(expected) for row in output):
        raise RuntimeError("Target-head feature schema drift")
    return output


def classify_windows(
    runtime: C5H8Runtime,
    windows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        _logits, predicted = runtime.classify(
            np.asarray(windows[start : start + batch_size], dtype=np.float32)
        )
        predictions.append(np.asarray(predicted, dtype=np.int64))
    result = np.concatenate(predictions)
    if result.shape != (len(windows),) or not np.isin(result, (0, 1, 2, 3)).all():
        raise RuntimeError("Canonical B5 emitted invalid predicted classes")
    return result


def prepare_split_rows(
    data_root: Path,
    split: str,
    runtime: C5H8Runtime,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = build_oracle_rows(data_root, ["C5"], split)
    windows = np.load(
        data_root / "client_5" / f"{split}_features.npy",
        allow_pickle=True,
    )
    if len(rows) != len(windows):
        raise RuntimeError(f"C5 {split} row/window count mismatch")
    predicted = classify_windows(runtime, windows, batch_size)
    by_index = {int(row["sample_index"]): row for row in rows}
    if set(by_index) != set(range(len(windows))):
        raise RuntimeError(f"C5 {split} sample indexes are not canonical")
    oracle_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    for index in range(len(windows)):
        base = dict(by_index[index])
        if len(base["feature_dict"]) != 104:
            raise RuntimeError(
                f"Expected 104D rich features, got {len(base['feature_dict'])}"
            )
        route = int(predicted[index])
        true_class = int(base["true_class"])
        oracle = dict(base)
        oracle["pred_class"] = route
        oracle["route_class"] = true_class
        oracle.update(
            runtime.h8_policy.predict_components(
                oracle["feature_dict"], true_class
            )
        )
        deployment = dict(base)
        deployment["pred_class"] = route
        deployment["route_class"] = route
        deployment.update(
            runtime.h8_policy.predict_components(
                deployment["feature_dict"], route
            )
        )
        oracle_rows.append(oracle)
        deployment_rows.append(deployment)
    return oracle_rows, deployment_rows


def rmse(rows: Sequence[Mapping[str, Any]], pred_key: str) -> float:
    true = np.asarray([float(row["true_ppm"]) for row in rows])
    pred = np.asarray([float(row[pred_key]) for row in rows])
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mlp_parameter_count(model: MLPHead) -> int:
    mlp = model.model.steps[-1][1]
    return int(
        sum(array.size for array in mlp.coefs_)
        + sum(array.size for array in mlp.intercepts_)
    )


def serialized_mlp_parameter_count(model: Any) -> int:
    return int(
        sum(array.size for array in model.coefs)
        + sum(array.size for array in model.intercepts)
    )


def fit_new_variant_calibration(
    oracle_rows: Sequence[Mapping[str, Any]],
    deployment_rows: Sequence[Mapping[str, Any]],
    variant: str,
    seed: int,
) -> tuple[
    dict[int, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    spec = VARIANTS[variant]
    train_features = add_source_features(oracle_rows, spec["source_keys"])
    deployment_features = add_source_features(
        deployment_rows, spec["source_keys"]
    )
    names = sorted(train_features[0]["feature_dict"])
    expected_dimension = 104 + len(spec["source_keys"])
    if len(names) != expected_dimension:
        raise RuntimeError(
            f"{variant} dimension mismatch: {len(names)} != {expected_dimension}"
        )
    deployment_by_id = {
        int(row["sample_index"]): row for row in deployment_features
    }
    final_models: dict[int, Any] = {}
    validation_rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for class_id in sorted(CLASS_NAMES):
        class_rows = [
            row
            for row in train_features
            if int(row["true_class"]) == class_id
        ]
        fit_rows, val_seed_rows = deterministic_train_val(class_rows, 0.25)
        val_rows = [
            dict(deployment_by_id[int(row["sample_index"])])
            for row in val_seed_rows
        ]
        y_val = np.asarray(
            [float(row["true_ppm"]) for row in val_rows], dtype=np.float64
        )
        if spec["head"] == "ridge":
            best_score = float("inf")
            best_alpha = RIDGE_ALPHAS[0]
            grid: list[dict[str, Any]] = []
            for alpha in RIDGE_ALPHAS:
                candidate = fit_ridge(fit_rows, names, alpha)
                score = float(
                    np.sqrt(
                        np.mean((candidate.predict(val_rows) - y_val) ** 2)
                    )
                )
                grid.append({"alpha": alpha, "validation_RMSE": score})
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
            selection_model = fit_ridge(fit_rows, names, best_alpha)
            final_model = fit_ridge(class_rows, names, best_alpha)
            hyper = {"best_alpha": best_alpha, "best_hidden": ""}
            parameter_count = len(final_model.coef)
        else:
            best_score = float("inf")
            best_alpha = MLP_ALPHAS[0]
            best_hidden = MLP_HIDDEN_GRID[0]
            grid = []
            model_seed = int(seed) + class_id + 500
            for hidden in MLP_HIDDEN_GRID:
                for alpha in MLP_ALPHAS:
                    candidate = fit_mlp(
                        fit_rows, names, hidden, alpha, model_seed
                    )
                    score = float(
                        np.sqrt(
                            np.mean((candidate.predict(val_rows) - y_val) ** 2)
                        )
                    )
                    grid.append(
                        {
                            "hidden": list(hidden),
                            "alpha": alpha,
                            "validation_RMSE": score,
                        }
                    )
                    if score < best_score:
                        best_score = score
                        best_hidden = hidden
                        best_alpha = alpha
            selection_model = fit_mlp(
                fit_rows, names, best_hidden, best_alpha, model_seed
            )
            final_model = fit_mlp(
                class_rows, names, best_hidden, best_alpha, model_seed
            )
            hyper = {
                "best_alpha": best_alpha,
                "best_hidden": list(best_hidden),
            }
            parameter_count = mlp_parameter_count(final_model)
        predictions = selection_model.predict(val_rows)
        for row, value in zip(val_rows, predictions):
            item = dict(row)
            item[f"{variant}_ppm"] = float(value)
            item["selection_split"] = MODEL_SELECTION_SPLIT
            validation_rows.append(item)
        final_models[class_id] = final_model
        audit.append(
            {
                "variant": variant,
                "head": spec["head"],
                "class_id": class_id,
                "gas": CLASS_NAMES[class_id],
                "fit_n": len(fit_rows),
                "validation_n": len(val_rows),
                "input_dimension": len(names),
                "best_alpha": hyper["best_alpha"],
                "best_hidden": json.dumps(hyper["best_hidden"]),
                "validation_RMSE": best_score,
                "grid_audit": json.dumps(grid, ensure_ascii=False),
                "selection_source": MODEL_SELECTION_SPLIT,
                "model_parameter_count": parameter_count,
            }
        )
    validation_rows.sort(key=lambda row: int(row["sample_index"]))
    return (
        final_models,
        validation_rows,
        audit,
        {
            "variant": variant,
            "head": spec["head"],
            "input_dimension": len(names),
            "trainable_parameter_count": int(
                sum(
                    len(model.coef)
                    if spec["head"] == "ridge"
                    else mlp_parameter_count(model)
                    for model in final_models.values()
                )
            ),
            "feature_names": names,
            "selected_hyperparameters": {
                str(row["class_id"]): {
                    "best_alpha": row["best_alpha"],
                    "best_hidden": json.loads(row["best_hidden"]),
                }
                for row in audit
            },
        },
    )


def apply_new_models(
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    models: Mapping[int, Any],
) -> list[dict[str, Any]]:
    spec = VARIANTS[variant]
    features = add_source_features(rows, spec["source_keys"])
    output = [dict(row) for row in features]
    for class_id, model in models.items():
        indexes = [
            index
            for index, row in enumerate(output)
            if int(row["route_class"]) == class_id
        ]
        values = model.predict([output[index] for index in indexes])
        for index, value in zip(indexes, values):
            output[index][f"{variant}_ppm"] = float(value)
    if any(f"{variant}_ppm" not in row for row in output):
        raise RuntimeError(f"{variant} did not produce every routed prediction")
    values = np.asarray([row[f"{variant}_ppm"] for row in output])
    if not np.isfinite(values).all():
        raise RuntimeError(f"{variant} emitted NaN/Inf")
    return output


def normalize_frozen_rows(
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    prediction_field: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["sample_index"] = int(item["sample_index"])
        item["true_class"] = int(item["true_class"])
        item["true_ppm"] = float(item["true_ppm"])
        item["pred_class"] = int(item["pred_class"])
        item["route_class"] = int(item.get("route_class", item["pred_class"]))
        item[f"{variant}_ppm"] = float(item[prediction_field])
        output.append(item)
    output.sort(key=lambda row: int(row["sample_index"]))
    return output


def merge_prediction_sets(
    prediction_sets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    indexes = {
        variant: {int(row["sample_index"]): row for row in rows}
        for variant, rows in prediction_sets.items()
    }
    expected = set(next(iter(indexes.values())))
    if any(set(rows) != expected for rows in indexes.values()):
        raise RuntimeError("Variant row identity mismatch")
    output: list[dict[str, Any]] = []
    for sample_index in sorted(expected):
        first = next(iter(indexes.values()))[sample_index]
        item = {
            "client": "C5",
            "sample_index": sample_index,
            "true_class": int(first["true_class"]),
            "true_ppm": float(first["true_ppm"]),
            "pred_class": int(first["pred_class"]),
            "route_class": int(first["route_class"]),
        }
        for variant, rows in indexes.items():
            row = rows[sample_index]
            identity = (
                int(row["true_class"]),
                float(row["true_ppm"]),
                int(row["pred_class"]),
            )
            expected_identity = (
                item["true_class"],
                item["true_ppm"],
                item["pred_class"],
            )
            if (
                identity[0] != expected_identity[0]
                or abs(identity[1] - expected_identity[1]) > 1e-9
                or identity[2] != expected_identity[2]
            ):
                raise RuntimeError(
                    f"Variant row content mismatch at {sample_index}"
                )
            item[f"{variant}_ppm"] = float(row[f"{variant}_ppm"])
        output.append(item)
    return output


def overall_metrics(
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    model_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    key = f"{variant}_ppm"
    true = np.asarray([float(row["true_ppm"]) for row in rows])
    pred = np.asarray([float(row[key]) for row in rows])
    classes = np.asarray([int(row["true_class"]) for row in rows])
    route = np.asarray([int(row["pred_class"]) for row in rows])
    error = pred - true
    correct = route == classes
    co_high = (classes == 1) & (true >= 200.0) & (true <= 250.0)
    ranges = np.asarray([CLASS_RANGES[int(value)] for value in classes])
    return {
        "variant": variant,
        "N": len(rows),
        "S_ALL_RMSE": float(np.sqrt(np.mean(error ** 2))),
        "S_ALL_MAE": float(np.mean(np.abs(error))),
        "S_ALL_NRMSE": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "S_CC_N": int(correct.sum()),
        "S_CC_RMSE": float(np.sqrt(np.mean(error[correct] ** 2))),
        "CO_RMSE": float(np.sqrt(np.mean(error[classes == 1] ** 2))),
        "CO_high_200_250_N": int(co_high.sum()),
        "CO_high_200_250_RMSE": float(
            np.sqrt(np.mean(error[co_high] ** 2))
        ),
        "model_parameter_count": int(
            model_manifest["trainable_parameter_count"]
        ),
        "target_input_dimension": int(model_manifest["input_dimension"]),
    }


def per_gas_metrics(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> list[dict[str, Any]]:
    key = f"{variant}_ppm"
    output: list[dict[str, Any]] = []
    for class_id, gas in sorted(CLASS_NAMES.items()):
        selected = [
            row for row in rows if int(row["true_class"]) == class_id
        ]
        true = np.asarray([float(row["true_ppm"]) for row in selected])
        pred = np.asarray([float(row[key]) for row in selected])
        error = pred - true
        output.append(
            {
                "variant": variant,
                "class_id": class_id,
                "gas": gas,
                "N": len(selected),
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "MAE": float(np.mean(np.abs(error))),
            }
        )
    return output


def validation_summary(
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    model_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "N": len(rows),
        "calibration_validation_RMSE": rmse(
            rows, f"{variant}_ppm"
        ),
        "model_parameter_count": int(
            model_manifest["trainable_parameter_count"]
        ),
        "target_input_dimension": int(model_manifest["input_dimension"]),
    }


def freeze_decision_gate(
    summaries: Mapping[str, Mapping[str, Any]],
    formal_commit: str,
) -> dict[str, Any]:
    ridge = float(
        summaries["E1_RIDGE_PRIOR"]["calibration_validation_RMSE"]
    )
    mlp = float(
        summaries["E1_MLP_PRIOR"]["calibration_validation_RMSE"]
    )
    improvement = 100.0 * (ridge - mlp) / ridge
    promoted = improvement > 5.0
    return {
        "schema_version": "iotj.source_prior_target_head_gate.v1",
        "formal_run_commit": formal_commit,
        "selection_frozen_at": utc_now(),
        "selection_scope": "C5_calibration_validation_only",
        "calibration_validation_rmse": {
            variant: float(row["calibration_validation_RMSE"])
            for variant, row in summaries.items()
        },
        "ridge_prior_calibration_validation_RMSE": ridge,
        "mlp_prior_calibration_validation_RMSE": mlp,
        "mlp_prior_improvement_vs_ridge_prior_percent": improvement,
        "threshold_percent_strictly_greater_than": 5.0,
        "selected_candidate": (
            "E1_MLP_PRIOR" if promoted else "E1_RIDGE_PRIOR"
        ),
        "selection_status": (
            "NEW_CANDIDATE_PENDING_CONFIRMATION"
            if promoted
            else "KEEP_RUNTIME_V4"
        ),
        "runtime_action": "none",
        "test_opened_after_selection": False,
        "test_metrics_used_for_selection": False,
        "final_test_evaluation_timestamp": None,
    }


def effect_row(
    scope: str,
    effect: str,
    baseline: str,
    intervention: str,
    metrics_by_variant: Mapping[str, float],
) -> dict[str, Any]:
    base = float(metrics_by_variant[baseline])
    value = float(metrics_by_variant[intervention])
    return {
        "scope": scope,
        "effect": effect,
        "baseline": baseline,
        "intervention": intervention,
        "baseline_RMSE": base,
        "intervention_RMSE": value,
        "absolute_RMSE_reduction": base - value,
        "relative_improvement_percent": 100.0 * (base - value) / base,
        "positive_means_intervention_better": True,
        "selection_use": scope == MODEL_SELECTION_SPLIT,
    }


def factorial_effects(
    scope: str, metrics_by_variant: Mapping[str, float]
) -> list[dict[str, Any]]:
    return [
        effect_row(
            scope,
            "source_prior_gain_for_Ridge",
            "E1_RIDGE_RICH",
            "E1_RIDGE_PRIOR",
            metrics_by_variant,
        ),
        effect_row(
            scope,
            "source_prior_gain_for_MLP",
            "E1_MLP_RICH",
            "E1_MLP_PRIOR",
            metrics_by_variant,
        ),
        effect_row(
            scope,
            "MLP_vs_Ridge_without_prior",
            "E1_RIDGE_RICH",
            "E1_MLP_RICH",
            metrics_by_variant,
        ),
        effect_row(
            scope,
            "MLP_vs_Ridge_with_prior",
            "E1_RIDGE_PRIOR",
            "E1_MLP_PRIOR",
            metrics_by_variant,
        ),
    ]


def validate_reference_parity(
    deployment_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    require_all_rows: bool,
) -> dict[str, Any]:
    observed = {
        int(row["sample_index"]): row for row in deployment_rows
    }
    maximum = {key: 0.0 for key in SOURCE_KEYS}
    route_mismatch = 0
    for reference in reference_rows:
        index = int(reference["sample_index"])
        row = observed.get(index)
        if row is None:
            raise RuntimeError(f"Frozen reference row missing in replay: {index}")
        route_mismatch += int(
            int(reference["pred_class"]) != int(row["pred_class"])
        )
        for key in SOURCE_KEYS:
            maximum[key] = max(
                maximum[key],
                abs(float(reference[key]) - float(row[key])),
            )
    if require_all_rows and len(reference_rows) != len(deployment_rows):
        raise RuntimeError("Frozen reference does not cover full split")
    if route_mismatch or any(value > 2e-9 for value in maximum.values()):
        raise RuntimeError(
            "Canonical B5/R4 source-prior replay differs from frozen reference"
        )
    return {
        "N": len(reference_rows),
        "route_mismatch_count": route_mismatch,
        "max_abs_source_prediction_difference": maximum,
    }


def baseline_model_manifests(runtime: C5H8Runtime) -> dict[str, Any]:
    mlp_parameters = sum(
        serialized_mlp_parameter_count(model)
        for model in runtime.h23_policy.mlp.values()
    )
    return {
        "E1_RIDGE_RICH": {
            "variant": "E1_RIDGE_RICH",
            "head": "ridge",
            "input_dimension": 104,
            "trainable_parameter_count": 4 * 105,
            "source": "frozen_R4_rich_only_reference",
        },
        "E1_RIDGE_PRIOR": {
            "variant": "E1_RIDGE_PRIOR",
            "head": "ridge",
            "input_dimension": 107,
            "trainable_parameter_count": sum(
                len(model.coef) for model in runtime.h8_policy.target_ridge.values()
            ),
            "source": "frozen_runtime_v4_R4_reference",
        },
        "E1_MLP_RICH": {
            "variant": "E1_MLP_RICH",
            "head": "mlp",
            "input_dimension": 104,
            "trainable_parameter_count": mlp_parameters,
            "source": "frozen_H2.3_MLP_anchor_reference",
        },
    }


def baseline_selection_rows(
    h8_validation_audit: Path, h23_fit_audit: Path
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in read_csv(h8_validation_audit):
        family = row["family"]
        if family == "target_ridge_rich_only":
            variant = "E1_RIDGE_RICH"
        elif family == "target_ridge_plus_source_preds":
            variant = "E1_RIDGE_PRIOR"
        else:
            continue
        output.append(
            {
                "variant": variant,
                "head": "ridge",
                "class_id": int(row["class_id"]),
                "gas": row["gas"],
                "fit_n": int(row["train_N"]),
                "validation_n": int(row["val_N"]),
                "input_dimension": 104 if variant.endswith("RICH") else 107,
                "best_alpha": float(row["best_alpha"]),
                "best_hidden": "[]",
                "validation_RMSE": float(row["best_val_RMSE"]),
                "grid_audit": "frozen_reference",
                "selection_source": MODEL_SELECTION_SPLIT,
                "model_parameter_count": 105 if variant.endswith("RICH") else 108,
            }
        )
    for row in read_csv(h23_fit_audit):
        if row["family"] != "h2_c5_grid_mlp":
            continue
        hidden = [
            int(value)
            for value in row["best_hidden"].strip("()").split(",")
            if value.strip()
        ]
        input_dimension = 104
        widths = [input_dimension, *hidden, 1]
        parameters = sum(
            widths[index] * widths[index + 1] + widths[index + 1]
            for index in range(len(widths) - 1)
        )
        output.append(
            {
                "variant": "E1_MLP_RICH",
                "head": "mlp",
                "class_id": int(row["class_id"]),
                "gas": row["gas"],
                "fit_n": int(row["train_N"]),
                "validation_n": int(row["val_N"]),
                "input_dimension": input_dimension,
                "best_alpha": float(row["best_alpha"]),
                "best_hidden": json.dumps(hidden),
                "validation_RMSE": float(row["best_val_RMSE"]),
                "grid_audit": row["grid_audit"],
                "selection_source": MODEL_SELECTION_SPLIT,
                "model_parameter_count": parameters,
            }
        )
    return output


def validate_h23_protocol(path: str | Path, seed: int) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "hidden_grid": [list(value) for value in MLP_HIDDEN_GRID],
        "mlp_alphas": list(MLP_ALPHAS),
        "val_ratio": 0.25,
        "seed": seed,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Frozen H2.3 protocol differs for {key}: "
                f"{manifest.get(key)} != {value}"
            )
    return expected


def run_contract_check(args: argparse.Namespace) -> None:
    root = Path.cwd()
    frozen = frozen_hashes(root)
    runtime = C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    h23_protocol = validate_h23_protocol(args.h23_manifest, args.seed)
    _oracle, deployment = prepare_split_rows(
        Path(args.data_root), "calibration", runtime, args.batch_size
    )
    replay = validate_reference_parity(
        deployment,
        read_csv(args.h8_validation_prior),
        require_all_rows=False,
    )
    print(
        json.dumps(
            {
                "status": "contract_verified",
                "git_commit": git_commit(),
                "origin_commit": origin_commit(),
                "local_origin_equal": git_commit() == origin_commit(),
                "runtime_model": type(runtime.model).__name__,
                "rich_feature_dimension": 104,
                "prior_feature_dimension": 107,
                "h23_protocol": h23_protocol,
                "calibration_validation_replay": replay,
                "frozen_assets_sha256": frozen,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    output = Path(args.output_dir)
    require_new_empty_output(output)
    if args.formal_run and git_commit() != origin_commit():
        raise RuntimeError("Local HEAD and origin HEAD differ")
    frozen_before = frozen_hashes(root)
    runtime = C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    if runtime.h8_policy is None or runtime.h23_policy is None:
        raise RuntimeError("Runtime contract lacks frozen R4/H2.3 policies")
    validate_h23_protocol(args.h23_manifest, args.seed)

    # Phase 1: calibration only.
    calibration_oracle, calibration_deployment = prepare_split_rows(
        Path(args.data_root), "calibration", runtime, args.batch_size
    )
    if len(calibration_oracle) != 320:
        raise RuntimeError("C5 calibration must contain exactly 320 rows")
    frozen_h8_validation = read_csv(args.h8_validation_prior)
    calibration_replay = validate_reference_parity(
        calibration_deployment,
        frozen_h8_validation,
        require_all_rows=False,
    )

    models: dict[str, dict[int, Any]] = {}
    selection_rows = baseline_selection_rows(
        Path(args.h8_validation_fit_audit),
        Path(args.h23_fit_audit),
    )
    model_manifests = baseline_model_manifests(runtime)
    validation_sets: dict[str, list[dict[str, Any]]] = {
        "E1_RIDGE_RICH": normalize_frozen_rows(
            read_csv(args.h8_validation_rich),
            "E1_RIDGE_RICH",
            "target_ridge_rich_only_ppm",
        ),
        "E1_RIDGE_PRIOR": normalize_frozen_rows(
            frozen_h8_validation,
            "E1_RIDGE_PRIOR",
            "target_ridge_plus_source_preds_ppm",
        ),
        "E1_MLP_RICH": normalize_frozen_rows(
            read_csv(args.h23_validation),
            "E1_MLP_RICH",
            "h23_anchor_ppm",
        ),
    }
    for variant in sorted(NEW_MODEL_VARIANTS):
        (
            variant_models,
            validation_rows,
            audit,
            model_manifest,
        ) = fit_new_variant_calibration(
            calibration_oracle,
            calibration_deployment,
            variant,
            args.seed,
        )
        models[variant] = variant_models
        validation_sets[variant] = validation_rows
        selection_rows.extend(audit)
        model_manifests[variant] = model_manifest
    validation_merged = merge_prediction_sets(validation_sets)
    if len(validation_merged) != 80:
        raise RuntimeError("Calibration-validation must contain exactly 80 rows")
    validation_summaries = {
        variant: validation_summary(
            validation_sets[variant], variant, model_manifests[variant]
        )
        for variant in VARIANTS
    }
    decision_gate = freeze_decision_gate(
        validation_summaries, git_commit()
    )
    decision_gate["canonical_calibration_validation_replay"] = (
        calibration_replay
    )
    decision_gate_path = output / "decision_gate.json"
    write_json(decision_gate_path, decision_gate)
    persisted = json.loads(decision_gate_path.read_text(encoding="utf-8"))
    if persisted["test_opened_after_selection"] is not False:
        raise RuntimeError("Decision gate was not frozen before test opening")

    # Phase 2: one-time test evaluation after selection freeze.
    _test_oracle, test_deployment = prepare_split_rows(
        Path(args.data_root), "test", runtime, args.batch_size
    )
    if len(test_deployment) != 1360:
        raise RuntimeError("C5 test must contain exactly 1360 rows")
    frozen_h8_test = read_csv(args.h8_test_prior)
    test_replay = validate_reference_parity(
        test_deployment, frozen_h8_test, require_all_rows=True
    )
    test_sets: dict[str, list[dict[str, Any]]] = {
        "E1_RIDGE_RICH": normalize_frozen_rows(
            read_csv(args.h8_test_rich),
            "E1_RIDGE_RICH",
            "target_ridge_rich_only_ppm",
        ),
        "E1_RIDGE_PRIOR": normalize_frozen_rows(
            frozen_h8_test,
            "E1_RIDGE_PRIOR",
            "target_ridge_plus_source_preds_ppm",
        ),
        "E1_MLP_RICH": normalize_frozen_rows(
            read_csv(args.h23_test),
            "E1_MLP_RICH",
            "h23_anchor_ppm",
        ),
    }
    for variant in sorted(NEW_MODEL_VARIANTS):
        test_sets[variant] = apply_new_models(
            test_deployment, variant, models[variant]
        )
    test_merged = merge_prediction_sets(test_sets)
    summaries = [
        overall_metrics(test_sets[variant], variant, model_manifests[variant])
        for variant in VARIANTS
    ]
    per_gas = [
        row
        for variant in VARIANTS
        for row in per_gas_metrics(test_sets[variant], variant)
    ]
    test_rmse = {
        row["variant"]: float(row["S_ALL_RMSE"]) for row in summaries
    }
    validation_rmse = {
        variant: float(
            validation_summaries[variant][
                "calibration_validation_RMSE"
            ]
        )
        for variant in VARIANTS
    }
    effects = [
        *factorial_effects(MODEL_SELECTION_SPLIT, validation_rmse),
        *factorial_effects("test_generalization_only", test_rmse),
    ]
    decision_gate["test_opened_after_selection"] = True
    decision_gate["final_test_evaluation_timestamp"] = utc_now()
    decision_gate["test_metrics_used_for_selection"] = False
    decision_gate["generalization_test_S_ALL_RMSE"] = test_rmse
    decision_gate["canonical_test_replay"] = test_replay
    write_json(decision_gate_path, decision_gate)

    frozen_after = frozen_hashes(root)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen runtime/source-head assets changed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "formal" if args.formal_run else "smoke_only",
        "advancement_eligible": bool(args.formal_run),
        "formal_run_commit": git_commit(),
        "origin_commit_at_run": origin_commit(),
        "device": args.device,
        "seed": args.seed,
        "data_root": str(Path(args.data_root).resolve()),
        "classifier_route": {
            "loader": "C5H8Runtime.from_runtime_contract",
            "runtime_contract": str(Path(args.runtime_contract).resolve()),
            "model_class": type(runtime.model).__name__,
        },
        "split": {
            "calibration": 320,
            "calibration_fit": 240,
            "calibration_validation": 80,
            "test": 1360,
        },
        "source_prior": {
            "keys": list(SOURCE_KEYS),
            "loader": "frozen runtime-v4 FixedH8Policy",
            "source_heads_retrained": False,
            "calibration_fit_route": "true_class_formal_R4_semantics",
            "validation_test_route": "canonical_B5_predicted_class",
        },
        "ridge_alphas": list(RIDGE_ALPHAS),
        "mlp_protocol": {
            "hidden_grid": [list(value) for value in MLP_HIDDEN_GRID],
            "alphas": list(MLP_ALPHAS),
            "activation": "relu",
            "solver": "lbfgs",
            "max_iter": 800,
            "early_stopping": False,
            "seed": args.seed,
            "model_seed_rule": "seed + class_id + 500",
        },
        "variants": VARIANTS,
        "test_used_for_fit_select_or_refit": False,
        "runtime_v4_modified": False,
        "qc_modified": False,
        "frozen_assets_sha256_before": frozen_before,
        "frozen_assets_sha256_after": frozen_after,
        "formal_output_files": list(FORMAL_OUTPUT_FILES),
    }
    write_json(output / "protocol_manifest.json", manifest)
    write_json(
        output / "asset_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "calibration_validation_replay": calibration_replay,
            "test_replay": test_replay,
            "frozen_assets_unchanged": True,
            "frozen_assets_sha256": frozen_after,
        },
    )
    write_json(
        output / "target_head_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "models": model_manifests,
        },
    )
    selection_fields = (
        "variant",
        "head",
        "class_id",
        "gas",
        "fit_n",
        "validation_n",
        "input_dimension",
        "best_alpha",
        "best_hidden",
        "validation_RMSE",
        "grid_audit",
        "selection_source",
        "model_parameter_count",
    )
    write_csv(
        output / "calibration_selection.csv",
        sorted(
            selection_rows,
            key=lambda row: (row["variant"], int(row["class_id"])),
        ),
        selection_fields,
    )
    prediction_fields = (
        "client",
        "sample_index",
        "true_class",
        "true_ppm",
        "pred_class",
        "route_class",
        *(f"{variant}_ppm" for variant in VARIANTS),
    )
    write_csv(
        output / "calibration_validation_predictions.csv",
        validation_merged,
        prediction_fields,
    )
    write_csv(
        output / "test_predictions.csv", test_merged, prediction_fields
    )
    validation_fields = (
        "variant",
        "N",
        "calibration_validation_RMSE",
        "model_parameter_count",
        "target_input_dimension",
    )
    write_csv(
        output / "calibration_validation_summary.csv",
        [validation_summaries[variant] for variant in VARIANTS],
        validation_fields,
    )
    summary_fields = (
        "variant",
        "N",
        "S_ALL_RMSE",
        "S_ALL_MAE",
        "S_ALL_NRMSE",
        "S_CC_N",
        "S_CC_RMSE",
        "CO_RMSE",
        "CO_high_200_250_N",
        "CO_high_200_250_RMSE",
        "model_parameter_count",
        "target_input_dimension",
    )
    write_csv(output / "test_summary.csv", summaries, summary_fields)
    write_csv(
        output / "per_gas_summary.csv",
        per_gas,
        ("variant", "class_id", "gas", "N", "RMSE", "MAE"),
    )
    write_csv(
        output / "factorial_effects.csv",
        effects,
        (
            "scope",
            "effect",
            "baseline",
            "intervention",
            "baseline_RMSE",
            "intervention_RMSE",
            "absolute_RMSE_reduction",
            "relative_improvement_percent",
            "positive_means_intervention_better",
            "selection_use",
        ),
    )
    (output / "README.md").write_text(
        "# IoT-J Source-Prior × Target-Head Factorial\n\n"
        "This directory contains the formal E1/E2 single-seed structural "
        "ablation. Runtime v4, QC, B5, and pooled H1/H2/H3 are unchanged. "
        "Candidate selection uses C5 calibration-validation only; test is "
        "generalization-only and cannot change the decision gate.\n",
        encoding="utf-8",
    )
    print(f"Formal E1/E2 output written to {output}")


def parse_args() -> argparse.Namespace:
    root = "results/iotj_b5_c5_deployment_p1_20260722"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
    )
    parser.add_argument(
        "--runtime-contract",
        default=f"{root}/c5_h8_runtime_contract_b5_v4/runtime_contract.json",
    )
    parser.add_argument(
        "--h23-manifest", default=f"{root}/h23_plus/manifest.json"
    )
    parser.add_argument(
        "--h23-validation",
        default=f"{root}/h23_plus/c5_h23_plus_validation_predictions.csv",
    )
    parser.add_argument(
        "--h23-test",
        default=f"{root}/h23_plus/c5_h23_plus_test_predictions.csv",
    )
    parser.add_argument(
        "--h23-fit-audit",
        default=f"{root}/h23_plus/c5_h23_plus_fit_audit.csv",
    )
    parser.add_argument(
        "--h8-validation-rich",
        default=f"{root}/h8_no_rescue/target_validation_rich_only.csv",
    )
    parser.add_argument(
        "--h8-validation-prior",
        default=f"{root}/h8_no_rescue/target_validation_plus_source_preds.csv",
    )
    parser.add_argument(
        "--h8-validation-fit-audit",
        default=f"{root}/h8_no_rescue/validation_fit_audit.csv",
    )
    parser.add_argument(
        "--h8-test-rich",
        default=f"{root}/h8_no_rescue/target_predictions_rich_only.csv",
    )
    parser.add_argument(
        "--h8-test-prior",
        default=f"{root}/h8_no_rescue/target_predictions_plus_source_preds.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/iotj_source_prior_target_head_factorial_20260723",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--formal-run", action="store_true")
    parser.add_argument("--contract-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.formal_run and parsed.contract_check:
        raise ValueError("--formal-run and --contract-check are mutually exclusive")
    if parsed.contract_check:
        run_contract_check(parsed)
    else:
        run(parsed)
