"""Audit pooled H1 against a sufficient-statistics federated reconstruction.

The server-facing functions in this module accept only typed aggregate records.
Raw source rows, feature matrices, and labels are consumed only by client-side
helpers.  This is an algorithm-equivalence audit, not a secure-aggregation or
privacy mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_deploy.c5_h8_runtime import C5H8Runtime
from run_regression_head_ablation import (
    CLASS_NAMES,
    RidgeHead,
    build_oracle_rows,
    deterministic_train_val,
    fit_ridge,
    fit_select_refit,
    matrix_from_rows,
)
from scripts.evaluate_iotj_source_prior_target_head_factorial import (
    CLASS_RANGES,
    FROZEN_ASSETS,
    RIDGE_ALPHAS,
    baseline_model_manifests,
    normalize_frozen_rows,
    overall_metrics,
    per_gas_metrics,
    prepare_split_rows,
    read_csv,
    validate_reference_parity,
)


SCHEMA_VERSION = "iotj.h1_federated_ridge_equivalence.v1"
EXPERIMENT_ID = "IOTJ-H1-FEDERATED-RIDGE-EQUIVALENCE-S42-V1"
TARGET_VARIANTS = (
    "RIDGE_RICH_ONLY",
    "RIDGE_H1_POOLED",
    "RIDGE_H1_FEDERATED_STATS",
)
EXPECTED_FROZEN_HASHES = {
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json": "54a42bb9f622c441a889a36fb1e585cb437e04c11128eb0578cfef6fd7711c3c",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/row_map_1360.json": "7c37cc00d7fdb47e53130d5eeadea913ae96b88aeb8bfe3c6d081d9683a5fd35",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/parity_report.json": "f0610d39b44643c2eb77889e7cf747fb1ae79a2fd89535bd5123e8302c12dde3",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/runtime_rows.csv": "511529671daafd541c1b880fc673f55d0604bd0a9952af16fd2b15a325841f84",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/parity_report.json": "bda8e790eff6d51c59e37fca5f00e5bd5ca1048df7421d86390e675d2ffa06c3",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/runtime_rows.csv": "42c8090228b8d3e454ba7b35854f1260d69527b043734ef2e82bb12fd04ce941",
}
TOLERANCES = {
    "scaler_max_abs_difference": 1e-10,
    "coefficient_max_abs_difference": 1e-8,
    "prediction_max_abs_difference_ppm": 1e-6,
    "practical_prediction_max_abs_difference_ppm": 1e-3,
    "practical_metric_max_abs_difference_ppm": 1e-2,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def origin_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "origin/codex/iotj-confirmation-observability"],
        text=True,
    ).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def require_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def frozen_hashes(root: Path) -> dict[str, str]:
    observed = {
        relative: sha256_file(root / relative) for relative in EXPECTED_FROZEN_HASHES
    }
    if observed != EXPECTED_FROZEN_HASHES:
        raise RuntimeError("Frozen runtime-v4/HC95/HC90 assets differ from audit hashes")
    return observed


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _row_provenance(
    client_id: str, gas_id: int, phase: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{client_id}|{gas_id}|{phase}".encode())
    for row in rows:
        digest.update(
            (
                f"{row['client']}|{row['split']}|{row['sample_index']}|"
                f"{row['true_class']}|{float(row['true_ppm']):.17g}\n"
            ).encode()
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalFeatureMoments:
    client_id: str
    gas_id: int
    phase: str
    n: int
    sum_x: np.ndarray
    sum_x2: np.ndarray
    provenance_sha256: str


@dataclass(frozen=True)
class GlobalScaler:
    gas_id: int
    phase: str
    n: int
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class LocalNormalEquations:
    client_id: str
    gas_id: int
    phase: str
    n: int
    a: np.ndarray
    b: np.ndarray
    y_min: float
    y_max: float
    provenance_sha256: str


@dataclass(frozen=True)
class LocalValidationScore:
    client_id: str
    gas_id: int
    alpha: float
    n: int
    sse: float
    provenance_sha256: str


def _finite_matrix(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    x = matrix_from_rows(rows, feature_names)
    if not np.isfinite(x).all():
        raise ValueError("H1 source features must be finite for exact stats audit")
    return x


# Client-side: raw rows are intentionally accepted here and nowhere server-side.
def client_feature_moments(
    client_id: str,
    gas_id: int,
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> LocalFeatureMoments:
    selected = [
        row
        for row in rows
        if str(row["client"]) == client_id and int(row["true_class"]) == gas_id
    ]
    if not selected:
        raise ValueError(f"No rows for {client_id}/gas={gas_id}/{phase}")
    x = _finite_matrix(selected, feature_names)
    return LocalFeatureMoments(
        client_id=client_id,
        gas_id=gas_id,
        phase=phase,
        n=len(selected),
        sum_x=np.sum(x, axis=0, dtype=np.float64),
        sum_x2=np.sum(x * x, axis=0, dtype=np.float64),
        provenance_sha256=_row_provenance(client_id, gas_id, phase, selected),
    )


def _require_stat_types(values: Sequence[Any], expected: type, label: str) -> None:
    if not values or any(type(value) is not expected for value in values):
        raise TypeError(f"{label} accepts only {expected.__name__} records")


# Server-side: this API cannot accept raw rows, X, or y.
def server_aggregate_scaler(
    records: Sequence[LocalFeatureMoments],
) -> GlobalScaler:
    _require_stat_types(records, LocalFeatureMoments, "server_aggregate_scaler")
    gas_ids = {record.gas_id for record in records}
    phases = {record.phase for record in records}
    clients = {record.client_id for record in records}
    if len(gas_ids) != 1 or len(phases) != 1 or len(clients) != len(records):
        raise ValueError("Scaler statistics require one gas/phase and unique clients")
    n = sum(record.n for record in records)
    sum_x = np.sum(np.stack([record.sum_x for record in records]), axis=0)
    sum_x2 = np.sum(np.stack([record.sum_x2 for record in records]), axis=0)
    mean = sum_x / n
    variance = np.maximum(sum_x2 / n - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    return GlobalScaler(
        gas_id=next(iter(gas_ids)),
        phase=next(iter(phases)),
        n=n,
        mean=mean,
        scale=scale,
    )


# Client-side: compute normal equations after receiving the global scaler.
def client_normal_equations(
    client_id: str,
    gas_id: int,
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    scaler: GlobalScaler,
) -> LocalNormalEquations:
    if scaler.gas_id != gas_id or scaler.phase != phase:
        raise ValueError("Scaler gas/phase mismatch")
    selected = [
        row
        for row in rows
        if str(row["client"]) == client_id and int(row["true_class"]) == gas_id
    ]
    x = _finite_matrix(selected, feature_names)
    y = np.asarray([float(row["true_ppm"]) for row in selected], dtype=np.float64)
    z = (x - scaler.mean) / scaler.scale
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    return LocalNormalEquations(
        client_id=client_id,
        gas_id=gas_id,
        phase=phase,
        n=len(selected),
        a=design.T @ design,
        b=design.T @ y,
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        provenance_sha256=_row_provenance(client_id, gas_id, phase, selected),
    )


# Server-side: reconstruct only from typed normal-equation statistics.
def server_reconstruct_ridge(
    records: Sequence[LocalNormalEquations],
    scaler: GlobalScaler,
    feature_names: Sequence[str],
    alpha: float,
) -> RidgeHead:
    _require_stat_types(records, LocalNormalEquations, "server_reconstruct_ridge")
    if any(
        record.gas_id != scaler.gas_id or record.phase != scaler.phase
        for record in records
    ):
        raise ValueError("Normal-equation gas/phase mismatch")
    if len({record.client_id for record in records}) != len(records):
        raise ValueError("Normal equations require independent client provenance")
    a = np.sum(np.stack([record.a for record in records]), axis=0)
    b = np.sum(np.stack([record.b for record in records]), axis=0)
    regularizer = np.eye(a.shape[0], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    coef = np.linalg.pinv(a + regularizer) @ b
    return RidgeHead(
        alpha=float(alpha),
        feature_names=list(feature_names),
        mean=scaler.mean.copy(),
        scale=scaler.scale.copy(),
        coef=coef,
        clip_min=min(record.y_min for record in records),
        clip_max=max(record.y_max for record in records),
    )


# Client-side: validation labels remain local; only SSE and count leave the client.
def client_validation_score(
    client_id: str,
    gas_id: int,
    rows: Sequence[Mapping[str, Any]],
    model: RidgeHead,
) -> LocalValidationScore:
    selected = [
        row
        for row in rows
        if str(row["client"]) == client_id and int(row["true_class"]) == gas_id
    ]
    y = np.asarray([float(row["true_ppm"]) for row in selected], dtype=np.float64)
    error = model.predict(selected, clip=True) - y
    return LocalValidationScore(
        client_id=client_id,
        gas_id=gas_id,
        alpha=model.alpha,
        n=len(selected),
        sse=float(error @ error),
        provenance_sha256=_row_provenance(
            client_id, gas_id, "source_calibration_validation", selected
        ),
    )


def server_validation_rmse(records: Sequence[LocalValidationScore]) -> float:
    _require_stat_types(records, LocalValidationScore, "server_validation_rmse")
    if len({record.client_id for record in records}) != len(records):
        raise ValueError("Validation scores require independent clients")
    if len({(record.gas_id, record.alpha) for record in records}) != 1:
        raise ValueError("Validation scores require one gas/alpha")
    return float(np.sqrt(sum(record.sse for record in records) / sum(r.n for r in records)))


def _federated_fit_phase(
    rows: Sequence[Mapping[str, Any]],
    clients: Sequence[str],
    gas_id: int,
    phase: str,
    feature_names: Sequence[str],
    alpha: float,
) -> tuple[RidgeHead, GlobalScaler, list[LocalFeatureMoments], list[LocalNormalEquations]]:
    moments = [
        client_feature_moments(client, gas_id, phase, rows, feature_names)
        for client in clients
    ]
    scaler = server_aggregate_scaler(moments)
    equations = [
        client_normal_equations(
            client, gas_id, phase, rows, feature_names, scaler
        )
        for client in clients
    ]
    model = server_reconstruct_ridge(equations, scaler, feature_names, alpha)
    return model, scaler, moments, equations


def fit_federated_h1(
    source_train: Sequence[Mapping[str, Any]],
    source_calibration: Sequence[Mapping[str, Any]],
    clients: Sequence[str],
    feature_names: Sequence[str],
    alphas: Sequence[float],
) -> tuple[
    dict[int, RidgeHead],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, GlobalScaler],
]:
    models: dict[int, RidgeHead] = {}
    alpha_audit: list[dict[str, Any]] = []
    stats_manifest: list[dict[str, Any]] = []
    final_scalers: dict[int, GlobalScaler] = {}
    for gas_id in sorted(CLASS_NAMES):
        candidates: list[tuple[float, float, RidgeHead]] = []
        for alpha in alphas:
            model, scaler, moments, equations = _federated_fit_phase(
                source_train,
                clients,
                gas_id,
                "source_train_selection",
                feature_names,
                float(alpha),
            )
            scores = [
                client_validation_score(
                    client, gas_id, source_calibration, model
                )
                for client in clients
            ]
            score = server_validation_rmse(scores)
            candidates.append((score, float(alpha), model))
            alpha_audit.append(
                {
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "alpha": float(alpha),
                    "federated_validation_RMSE": score,
                    "validation_N": sum(item.n for item in scores),
                }
            )
        _, best_alpha, _ = min(candidates, key=lambda item: (item[0], alphas.index(item[1])))
        combined = [*source_train, *source_calibration]
        final, scaler, moments, equations = _federated_fit_phase(
            combined,
            clients,
            gas_id,
            "source_train_plus_calibration_refit",
            feature_names,
            best_alpha,
        )
        models[gas_id] = final
        final_scalers[gas_id] = scaler
        for moment, equation in zip(moments, equations):
            stats_manifest.append(
                {
                    "client_id": moment.client_id,
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "phase": moment.phase,
                    "n": moment.n,
                    "feature_moments_sha256": _array_hash(
                        moment.sum_x, moment.sum_x2
                    ),
                    "normal_equations_sha256": _array_hash(
                        equation.a, equation.b
                    ),
                    "row_provenance_sha256": moment.provenance_sha256,
                    "server_received": [
                        "n",
                        "sum_x",
                        "sum_x2",
                        "A",
                        "b",
                        "y_min",
                        "y_max",
                    ],
                }
            )
    return models, alpha_audit, stats_manifest, final_scalers


def fit_pooled_h1(
    source_train: Sequence[Mapping[str, Any]],
    source_calibration: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    alphas: Sequence[float],
) -> tuple[dict[int, RidgeHead], list[dict[str, Any]]]:
    models: dict[int, RidgeHead] = {}
    audit: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        train = [row for row in source_train if int(row["true_class"]) == gas_id]
        validation = [
            row for row in source_calibration if int(row["true_class"]) == gas_id
        ]
        model, details = fit_select_refit(train, validation, feature_names, alphas)
        models[gas_id] = model
        for row in details["alpha_audit"]:
            audit.append(
                {
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "alpha": float(row["alpha"]),
                    "pooled_validation_RMSE": float(row["val_RMSE"]),
                    "validation_N": len(validation),
                }
            )
    return models, audit


def apply_h1(
    rows: Sequence[Mapping[str, Any]],
    models: Mapping[int, RidgeHead],
    output_key: str,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for gas_id, model in models.items():
        indexes = [
            index
            for index, row in enumerate(output)
            if int(row["route_class"]) == gas_id
        ]
        predictions = model.predict([output[index] for index in indexes], clip=True)
        for index, prediction in zip(indexes, predictions):
            output[index][output_key] = float(prediction)
    if any(output_key not in row for row in output):
        raise RuntimeError(f"Missing routed H1 prediction: {output_key}")
    return output


def _attach_h1_feature(
    rows: Sequence[Mapping[str, Any]], prediction_key: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        features = dict(row["feature_dict"])
        features["srcpred_H1_source_ridge_ppm"] = float(row[prediction_key])
        item["feature_dict"] = features
        output.append(item)
    return output


def fit_target_ridge_h1(
    calibration_oracle: Sequence[Mapping[str, Any]],
    calibration_deployment: Sequence[Mapping[str, Any]],
    source_models: Mapping[int, RidgeHead],
    variant: str,
) -> tuple[
    dict[int, RidgeHead],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    key = f"{variant}_source_h1_ppm"
    oracle = _attach_h1_feature(apply_h1(calibration_oracle, source_models, key), key)
    deployment = _attach_h1_feature(
        apply_h1(calibration_deployment, source_models, key), key
    )
    by_index = {int(row["sample_index"]): row for row in deployment}
    feature_names = sorted(oracle[0]["feature_dict"])
    if len(feature_names) != 105:
        raise RuntimeError("Ridge+H1 target feature dimension must equal 105")
    models: dict[int, RidgeHead] = {}
    validation_output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        class_rows = [row for row in oracle if int(row["true_class"]) == gas_id]
        fit_rows, validation_seeds = deterministic_train_val(class_rows, 0.25)
        validation_rows = [
            dict(by_index[int(row["sample_index"])]) for row in validation_seeds
        ]
        best_alpha = float(RIDGE_ALPHAS[0])
        best_rmse = float("inf")
        grid: list[dict[str, float]] = []
        y = np.asarray([float(row["true_ppm"]) for row in validation_rows])
        for alpha in RIDGE_ALPHAS:
            candidate = fit_ridge(fit_rows, feature_names, alpha)
            prediction = candidate.predict(validation_rows, clip=True)
            score = float(np.sqrt(np.mean((prediction - y) ** 2)))
            grid.append({"alpha": float(alpha), "validation_RMSE": score})
            if score < best_rmse:
                best_rmse = score
                best_alpha = float(alpha)
        selection_model = fit_ridge(fit_rows, feature_names, best_alpha)
        final_model = fit_ridge(class_rows, feature_names, best_alpha)
        models[gas_id] = final_model
        predictions = selection_model.predict(validation_rows, clip=True)
        for row, prediction in zip(validation_rows, predictions):
            item = dict(row)
            item[f"{variant}_ppm"] = float(prediction)
            validation_output.append(item)
        audit.append(
            {
                "variant": variant,
                "gas_id": gas_id,
                "gas": CLASS_NAMES[gas_id],
                "fit_N": len(fit_rows),
                "validation_N": len(validation_rows),
                "best_alpha": best_alpha,
                "best_validation_RMSE": best_rmse,
                "grid_audit": json.dumps(grid),
            }
        )
    validation_output.sort(key=lambda row: int(row["sample_index"]))
    return models, validation_output, audit


def apply_target_ridge_h1(
    test_deployment: Sequence[Mapping[str, Any]],
    source_models: Mapping[int, RidgeHead],
    target_models: Mapping[int, RidgeHead],
    variant: str,
) -> list[dict[str, Any]]:
    key = f"{variant}_source_h1_ppm"
    rows = _attach_h1_feature(apply_h1(test_deployment, source_models, key), key)
    output = [dict(row) for row in rows]
    for gas_id, model in target_models.items():
        indexes = [
            index
            for index, row in enumerate(output)
            if int(row["route_class"]) == gas_id
        ]
        prediction = model.predict([output[index] for index in indexes], clip=True)
        for index, value in zip(indexes, prediction):
            output[index][f"{variant}_ppm"] = float(value)
    return output


def _source_equivalence(
    pooled: Mapping[int, RidgeHead],
    federated: Mapping[int, RidgeHead],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scaler_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        left, right = pooled[gas_id], federated[gas_id]
        scaler_rows.append(
            {
                "gas_id": gas_id,
                "gas": CLASS_NAMES[gas_id],
                "mean_max_abs_difference": float(
                    np.max(np.abs(left.mean - right.mean))
                ),
                "std_max_abs_difference": float(
                    np.max(np.abs(left.scale - right.scale))
                ),
            }
        )
        coefficient_rows.append(
            {
                "gas_id": gas_id,
                "gas": CLASS_NAMES[gas_id],
                "pooled_alpha": left.alpha,
                "federated_alpha": right.alpha,
                "alpha_equal": left.alpha == right.alpha,
                "intercept_abs_difference": float(abs(left.coef[0] - right.coef[0])),
                "coef_max_abs_difference": float(
                    np.max(np.abs(left.coef[1:] - right.coef[1:]))
                ),
                "clip_min_abs_difference": abs(left.clip_min - right.clip_min),
                "clip_max_abs_difference": abs(left.clip_max - right.clip_max),
            }
        )
    return scaler_rows, coefficient_rows


def _prediction_equivalence(
    split: str,
    rows: Sequence[Mapping[str, Any]],
    pooled: Mapping[int, RidgeHead],
    federated: Mapping[int, RidgeHead],
) -> list[dict[str, Any]]:
    pooled_rows = apply_h1(rows, pooled, "pooled")
    federated_rows = apply_h1(rows, federated, "federated")
    output: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        indexes = [
            index
            for index, row in enumerate(rows)
            if int(row["route_class"]) == gas_id
        ]
        left = np.asarray([pooled_rows[index]["pooled"] for index in indexes])
        right = np.asarray([federated_rows[index]["federated"] for index in indexes])
        delta = right - left
        output.append(
            {
                "split": split,
                "gas_id": gas_id,
                "gas": CLASS_NAMES[gas_id],
                "N": len(indexes),
                "max_abs_difference_ppm": float(np.max(np.abs(delta))),
                "prediction_difference_RMSE_ppm": float(
                    np.sqrt(np.mean(delta * delta))
                ),
            }
        )
    return output


def _validation_summary(
    variant: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    error = np.asarray(
        [float(row[f"{variant}_ppm"]) - float(row["true_ppm"]) for row in rows]
    )
    return {
        "variant": variant,
        "N": len(rows),
        "calibration_validation_RMSE": float(np.sqrt(np.mean(error * error))),
        "model_parameter_count": 4 * 106,
        "target_input_dimension": 105,
    }


def _model_manifest(
    models: Mapping[int, RidgeHead], source: str, protocol: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "protocol": protocol,
        "per_gas_independent": True,
        "feature_dimension": 104,
        "alpha_grid": list(RIDGE_ALPHAS),
        "models": {
            str(gas_id): {
                "gas": CLASS_NAMES[gas_id],
                **model.to_json(),
                "model_numeric_sha256": _array_hash(
                    model.mean, model.scale, model.coef
                ),
            }
            for gas_id, model in sorted(models.items())
        },
    }


def _server_api_audit() -> dict[str, Any]:
    functions = (
        server_aggregate_scaler,
        server_reconstruct_ridge,
        server_validation_rmse,
    )
    signatures = {function.__name__: str(inspect.signature(function)) for function in functions}
    forbidden = {"rows", "x", "y", "labels", "samples"}
    for function in functions:
        if forbidden.intersection(inspect.signature(function).parameters):
            raise RuntimeError(f"Server API accepts forbidden raw input: {function.__name__}")
    return {
        "server_api_signatures": signatures,
        "raw_rows_or_X_y_parameters_present": False,
        "claim_boundary": (
            "source raw samples remain local and only aggregated sufficient "
            "statistics are used to reconstruct the global Ridge solution."
        ),
    }


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    output = Path(args.output_dir)
    require_new_output(output)
    if args.formal_run and git_commit() != origin_commit():
        raise RuntimeError("Formal run requires local HEAD == origin HEAD")
    frozen_before = frozen_hashes(root)
    server_api = _server_api_audit()

    data_root = Path(args.data_root)
    clients = ("C1", "C2")
    source_train = build_oracle_rows(data_root, clients, "train")
    source_calibration = build_oracle_rows(data_root, clients, "calibration")
    if len(source_train) != 4720 or len(source_calibration) != 640:
        raise RuntimeError("Unexpected C1/C2 source train/calibration cardinality")
    feature_names = sorted(source_train[0]["feature_dict"])
    if len(feature_names) != 104:
        raise RuntimeError("H1 source feature dimension must equal 104")

    pooled, pooled_alpha_audit = fit_pooled_h1(
        source_train, source_calibration, feature_names, RIDGE_ALPHAS
    )
    federated, federated_alpha_audit, local_stats, _ = fit_federated_h1(
        source_train,
        source_calibration,
        clients,
        feature_names,
        RIDGE_ALPHAS,
    )
    scaler_rows, coefficient_rows = _source_equivalence(pooled, federated)
    alpha_by_key = {
        (row["gas_id"], row["alpha"]): row for row in pooled_alpha_audit
    }
    alpha_rows = []
    for row in federated_alpha_audit:
        pooled_row = alpha_by_key[(row["gas_id"], row["alpha"])]
        alpha_rows.append(
            {
                **row,
                "pooled_validation_RMSE": pooled_row["pooled_validation_RMSE"],
                "validation_RMSE_abs_difference": abs(
                    row["federated_validation_RMSE"]
                    - pooled_row["pooled_validation_RMSE"]
                ),
            }
        )

    runtime = C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    calibration_oracle, calibration_deployment = prepare_split_rows(
        data_root, "calibration", runtime, args.batch_size
    )
    calibration_parity = validate_reference_parity(
        calibration_deployment,
        read_csv(args.h8_validation_prior),
        require_all_rows=False,
    )
    if calibration_parity["N"] != 80 or calibration_parity["route_mismatch_count"]:
        raise RuntimeError("B5 calibration-validation route parity failed")

    pooled_target, pooled_validation, pooled_target_audit = fit_target_ridge_h1(
        calibration_oracle,
        calibration_deployment,
        pooled,
        "RIDGE_H1_POOLED",
    )
    fed_target, fed_validation, fed_target_audit = fit_target_ridge_h1(
        calibration_oracle,
        calibration_deployment,
        federated,
        "RIDGE_H1_FEDERATED_STATS",
    )
    validation_summaries = [
        _validation_summary("RIDGE_H1_POOLED", pooled_validation),
        _validation_summary("RIDGE_H1_FEDERATED_STATS", fed_validation),
    ]

    # Freeze equivalence inputs and target selection before test is opened.
    preliminary_gate = {
        "schema_version": SCHEMA_VERSION,
        "selection_scope": "equivalence_only",
        "test_opened_after_selection": False,
        "test_used_for_scaler_alpha_fit_or_refit": False,
        "runtime_action": "none",
        "tolerances": TOLERANCES,
    }
    write_json(output / "equivalence_decision.json", preliminary_gate)

    _test_oracle, test_deployment = prepare_split_rows(
        data_root, "test", runtime, args.batch_size
    )
    test_parity = validate_reference_parity(
        test_deployment, read_csv(args.h8_test_prior), require_all_rows=True
    )
    if len(test_deployment) != 1360 or test_parity["route_mismatch_count"]:
        raise RuntimeError("B5 1360-row route parity failed")

    prediction_rows = [
        *_prediction_equivalence(
            "C5_calibration_all_320", calibration_deployment, pooled, federated
        ),
        *_prediction_equivalence("C5_test_1360", test_deployment, pooled, federated),
    ]
    pooled_test = apply_target_ridge_h1(
        test_deployment, pooled, pooled_target, "RIDGE_H1_POOLED"
    )
    fed_test = apply_target_ridge_h1(
        test_deployment,
        federated,
        fed_target,
        "RIDGE_H1_FEDERATED_STATS",
    )
    rich_test = normalize_frozen_rows(
        read_csv(args.h8_test_rich),
        "RIDGE_RICH_ONLY",
        "target_ridge_rich_only_ppm",
    )
    rich_validation = normalize_frozen_rows(
        read_csv(args.h8_validation_rich),
        "RIDGE_RICH_ONLY",
        "target_ridge_rich_only_ppm",
    )
    rich_manifest = {
        "trainable_parameter_count": 4 * 105,
        "input_dimension": 104,
    }
    h1_manifest = {
        "trainable_parameter_count": 4 * 106,
        "input_dimension": 105,
    }
    regression_rows = [
        {
            "variant": "RIDGE_RICH_ONLY",
            "calibration_validation_RMSE": float(
                np.sqrt(
                    np.mean(
                        np.asarray(
                            [
                                float(row["RIDGE_RICH_ONLY_ppm"])
                                - float(row["true_ppm"])
                                for row in rich_validation
                            ]
                        )
                        ** 2
                    )
                )
            ),
            **{
                key: value
                for key, value in overall_metrics(
                    rich_test, "RIDGE_RICH_ONLY", rich_manifest
                ).items()
                if key != "variant"
            },
        }
    ]
    for variant, validation, test in (
        ("RIDGE_H1_POOLED", pooled_validation, pooled_test),
        ("RIDGE_H1_FEDERATED_STATS", fed_validation, fed_test),
    ):
        regression_rows.append(
            {
                "variant": variant,
                "calibration_validation_RMSE": _validation_summary(
                    variant, validation
                )["calibration_validation_RMSE"],
                **{
                    key: value
                    for key, value in overall_metrics(test, variant, h1_manifest).items()
                    if key != "variant"
                },
            }
        )
    gas_rows = [
        *per_gas_metrics(rich_test, "RIDGE_RICH_ONLY"),
        *per_gas_metrics(pooled_test, "RIDGE_H1_POOLED"),
        *per_gas_metrics(fed_test, "RIDGE_H1_FEDERATED_STATS"),
    ]

    max_scaler = max(
        max(row["mean_max_abs_difference"], row["std_max_abs_difference"])
        for row in scaler_rows
    )
    max_coef = max(
        max(row["intercept_abs_difference"], row["coef_max_abs_difference"])
        for row in coefficient_rows
    )
    max_prediction = max(
        row["max_abs_difference_ppm"] for row in prediction_rows
    )
    alpha_equal = all(row["alpha_equal"] for row in coefficient_rows)
    pooled_metrics = next(
        row for row in regression_rows if row["variant"] == "RIDGE_H1_POOLED"
    )
    fed_metrics = next(
        row
        for row in regression_rows
        if row["variant"] == "RIDGE_H1_FEDERATED_STATS"
    )
    s_all_difference = abs(
        pooled_metrics["S_ALL_RMSE"] - fed_metrics["S_ALL_RMSE"]
    )
    s_cc_difference = abs(
        pooled_metrics["S_CC_RMSE"] - fed_metrics["S_CC_RMSE"]
    )
    if (
        alpha_equal
        and max_scaler <= TOLERANCES["scaler_max_abs_difference"]
        and max_coef <= TOLERANCES["coefficient_max_abs_difference"]
        and max_prediction <= TOLERANCES["prediction_max_abs_difference_ppm"]
    ):
        decision = "EXACT_EQUIVALENCE"
    elif (
        alpha_equal
        and max_prediction
        <= TOLERANCES["practical_prediction_max_abs_difference_ppm"]
        and s_all_difference
        <= TOLERANCES["practical_metric_max_abs_difference_ppm"]
        and s_cc_difference
        <= TOLERANCES["practical_metric_max_abs_difference_ppm"]
    ):
        decision = "PRACTICAL_EQUIVALENCE"
    else:
        decision = "NOT_EQUIVALENT"

    gate = {
        **preliminary_gate,
        "decision": decision,
        "alpha_equal_all_gases": alpha_equal,
        "max_scaler_abs_difference": max_scaler,
        "max_coefficient_or_intercept_abs_difference": max_coef,
        "max_C5_H1_prediction_abs_difference_ppm": max_prediction,
        "Ridge_H1_S_ALL_RMSE_abs_difference": s_all_difference,
        "Ridge_H1_S_CC_RMSE_abs_difference": s_cc_difference,
        "test_opened_after_selection": True,
        "test_used_for_scaler_alpha_fit_or_refit": False,
        "test_evaluation_timestamp": utc_now(),
        "recommendation": (
            "paper_preferred_simplified_source_reference_pending_multi_seed"
            if decision == "EXACT_EQUIVALENCE"
            else "report_numerical_error_runtime_unchanged"
            if decision == "PRACTICAL_EQUIVALENCE"
            else "stop_not_equivalent"
        ),
        "runtime_action": "none",
        "b5_route_parity": {
            "calibration_validation": calibration_parity,
            "test": test_parity,
        },
        "evidence_boundary": server_api["claim_boundary"],
    }
    write_json(output / "equivalence_decision.json", gate)

    write_json(
        output / "pooled_h1_manifest.json",
        _model_manifest(
            pooled,
            "centralized_C1_C2_raw_rows",
            "existing_formal_H1_custom_closed_form_Ridge",
        ),
    )
    write_json(
        output / "federated_h1_manifest.json",
        {
            **_model_manifest(
                federated,
                "C1_C2_local_sufficient_statistics",
                "two_phase_global_scaler_then_normal_equations",
            ),
            **server_api,
        },
    )
    write_json(
        output / "local_statistics_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "records": local_stats,
            "raw_rows_transmitted_to_server": False,
            "raw_X_y_transmitted_to_server": False,
            "independent_client_provenance": True,
        },
    )
    write_csv(
        output / "scaler_equivalence.csv",
        scaler_rows,
        (
            "gas_id",
            "gas",
            "mean_max_abs_difference",
            "std_max_abs_difference",
        ),
    )
    write_csv(
        output / "alpha_selection_audit.csv",
        alpha_rows,
        (
            "gas_id",
            "gas",
            "alpha",
            "pooled_validation_RMSE",
            "federated_validation_RMSE",
            "validation_RMSE_abs_difference",
            "validation_N",
        ),
    )
    write_csv(
        output / "coefficient_equivalence.csv",
        coefficient_rows,
        (
            "gas_id",
            "gas",
            "pooled_alpha",
            "federated_alpha",
            "alpha_equal",
            "intercept_abs_difference",
            "coef_max_abs_difference",
            "clip_min_abs_difference",
            "clip_max_abs_difference",
        ),
    )
    write_csv(
        output / "h1_prediction_equivalence.csv",
        prediction_rows,
        (
            "split",
            "gas_id",
            "gas",
            "N",
            "max_abs_difference_ppm",
            "prediction_difference_RMSE_ppm",
        ),
    )
    write_csv(
        output / "regression_comparison.csv",
        regression_rows,
        tuple(regression_rows[0]),
    )
    write_csv(
        output / "per_gas_summary.csv",
        gas_rows,
        tuple(gas_rows[0]),
    )
    write_json(
        output / "protocol_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "formal" if args.formal_run else "smoke_only",
            "formal_run_commit": git_commit(),
            "origin_commit_at_run": origin_commit(),
            "local_origin_equal": git_commit() == origin_commit(),
            "created_at": utc_now(),
            "source_clients": list(clients),
            "target_client": "C5",
            "seed": args.seed,
            "data_root": str(data_root.resolve()),
            "split": {
                "C1_C2_source_train": len(source_train),
                "C1_C2_source_calibration": len(source_calibration),
                "C5_calibration": 320,
                "C5_calibration_fit": 240,
                "C5_calibration_validation": 80,
                "C5_test": 1360,
            },
            "h1": {
                "per_gas_independent": True,
                "feature_dimension": 104,
                "scaler": "mean and population std (ddof=0); scale<1e-9 -> 1",
                "intercept": "explicit constant column; unregularized",
                "objective": "pinv(D.T@D + diag(0,alpha,...)) @ D.T@y",
                "alpha_grid": list(RIDGE_ALPHAS),
                "selection": "pooled C1+C2 calibration RMSE with clipped predictions",
                "refit": "C1+C2 train+calibration",
                "clipping": "per-gas refit-label min/max",
            },
            "federated_exchange": {
                "phase_1": ["n_i", "sum_x_i", "sum_x2_i"],
                "phase_2": ["A_i", "b_i", "y_min_i", "y_max_i"],
                "alpha_selection": ["calibration_SSE_i", "calibration_count_i"],
                "raw_rows_transmitted": False,
                "raw_X_y_transmitted": False,
            },
            "target_variants": list(TARGET_VARIANTS),
            "test_used_for_fit_select_or_refit": False,
            "C5_calibration_used_for_source_H1_training": False,
            "C1_C2_source_test_used_for_H1_train_or_select": False,
            "runtime_v4_modified": False,
            "QC_modified": False,
            "frozen_assets_sha256_before": frozen_before,
            "frozen_assets_sha256_after": frozen_hashes(root),
        },
    )
    readme = f"""# H1 pooled-to-federated sufficient-statistics equivalence

- Decision: `{decision}`
- Formal commit: `{git_commit()}`
- Source: C1/C2; target: C5; B5 seed 42; C5 calibration/test: 320/1360.
- H1 is a 104D per-gas custom closed-form Ridge. The federated path exchanges
  feature moments, normal equations, and calibration SSE/count only.
- Maximum scaler difference: `{max_scaler:.17g}`
- Maximum coefficient/intercept difference: `{max_coef:.17g}`
- Maximum C5 H1 prediction difference: `{max_prediction:.17g}` ppm
- Ridge+H1 pooled S_ALL/S_CC RMSE: `{pooled_metrics['S_ALL_RMSE']:.12f}` /
  `{pooled_metrics['S_CC_RMSE']:.12f}` ppm
- Ridge+H1 federated S_ALL/S_CC RMSE: `{fed_metrics['S_ALL_RMSE']:.12f}` /
  `{fed_metrics['S_CC_RMSE']:.12f}` ppm

Evidence boundary: source raw samples remain local and only aggregated
sufficient statistics are used to reconstruct the global Ridge solution.
This audit does not claim secure aggregation, differential privacy,
cryptographic privacy, or that sufficient statistics are non-leaking.
Runtime v4 and QC remain unchanged.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    frozen_after = frozen_hashes(root)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen assets changed during equivalence audit")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


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
        "--h8-validation-rich",
        default=f"{root}/h8_no_rescue/target_validation_rich_only.csv",
    )
    parser.add_argument(
        "--h8-validation-prior",
        default=f"{root}/h8_no_rescue/target_validation_plus_source_preds.csv",
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
        default="results/iotj_h1_federated_ridge_equivalence_20260724",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--formal-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
