"""Canonical source-only sufficient-statistics Ridge reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


SCALE_FLOOR = 1e-9
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _finite_matrix(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a non-empty finite 2D matrix")
    return array


def _finite_vector(values: np.ndarray, n: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) != n or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector aligned to X")
    return array


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalFeatureMoments:
    client_id: str
    gas_id: int
    role: str
    n: int
    sum_x: np.ndarray
    sum_x2: np.ndarray
    provenance_sha256: str


@dataclass(frozen=True)
class GlobalScaler:
    gas_id: int
    role: str
    n: int
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class LocalNormalEquations:
    client_id: str
    gas_id: int
    role: str
    n: int
    x_t_x: np.ndarray
    x_t_y: np.ndarray
    y_y: float
    y_min: float
    y_max: float
    provenance_sha256: str


@dataclass(frozen=True)
class LocalValidationScore:
    client_id: str
    gas_id: int
    alpha: float
    role: str
    n: int
    sse: float
    provenance_sha256: str


@dataclass(frozen=True)
class CanonicalRidgeModel:
    gas_id: int
    role: str
    alpha: float
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    clip_min: float
    clip_max: float

    def predict_matrix(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        x = _finite_matrix(values, "prediction X")
        if x.shape[1] != len(self.feature_names):
            raise ValueError("prediction feature dimension mismatch")
        design = np.concatenate(
            [np.ones((len(x), 1), dtype=np.float64), (x - self.mean) / self.scale],
            axis=1,
        )
        prediction = design @ self.coef
        if clip:
            prediction = np.clip(prediction, self.clip_min, self.clip_max)
        return prediction

    def to_json(self) -> dict[str, Any]:
        return {
            "gas_id": self.gas_id,
            "role": self.role,
            "alpha": self.alpha,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "intercept_regularized": False,
            "solver": "numpy.linalg.pinv",
        }


def client_feature_moments(
    client_id: str, gas_id: int, role: str, values: np.ndarray
) -> LocalFeatureMoments:
    x = _finite_matrix(values, "client feature X")
    return LocalFeatureMoments(
        client_id=str(client_id),
        gas_id=int(gas_id),
        role=str(role),
        n=len(x),
        sum_x=np.sum(x, axis=0, dtype=np.float64),
        sum_x2=np.sum(x * x, axis=0, dtype=np.float64),
        provenance_sha256=_array_sha256(x),
    )


def _require_exact_records(records: Sequence[Any], expected: type, label: str) -> None:
    if not records or any(type(record) is not expected for record in records):
        raise TypeError(f"{label} accepts only {expected.__name__} records")


def server_aggregate_scaler(records: Sequence[LocalFeatureMoments]) -> GlobalScaler:
    _require_exact_records(records, LocalFeatureMoments, "server_aggregate_scaler")
    if len({record.client_id for record in records}) != len(records):
        raise ValueError("feature moments require unique clients")
    if len({(record.gas_id, record.role) for record in records}) != 1:
        raise ValueError("feature moments require one gas/role")
    dimensions = {record.sum_x.shape for record in records}
    if len(dimensions) != 1:
        raise ValueError("feature moment dimensions differ")
    n = sum(record.n for record in records)
    sum_x = np.sum(np.stack([record.sum_x for record in records]), axis=0)
    sum_x2 = np.sum(np.stack([record.sum_x2 for record in records]), axis=0)
    mean = sum_x / n
    variance = np.maximum(sum_x2 / n - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(np.abs(scale) < SCALE_FLOOR, 1.0, scale)
    first = records[0]
    return GlobalScaler(first.gas_id, first.role, n, mean, scale)


def client_normal_equations(
    client_id: str,
    gas_id: int,
    role: str,
    values: np.ndarray,
    targets: np.ndarray,
    scaler: GlobalScaler,
) -> LocalNormalEquations:
    x = _finite_matrix(values, "client normal-equation X")
    y = _finite_vector(targets, len(x), "client normal-equation y")
    if scaler.gas_id != int(gas_id) or scaler.role != str(role):
        raise ValueError("normal-equation scaler gas/role mismatch")
    if x.shape[1] != len(scaler.mean):
        raise ValueError("normal-equation feature dimension mismatch")
    z = (x - scaler.mean) / scaler.scale
    design = np.concatenate([np.ones((len(z), 1), dtype=np.float64), z], axis=1)
    return LocalNormalEquations(
        client_id=str(client_id),
        gas_id=int(gas_id),
        role=str(role),
        n=len(x),
        x_t_x=design.T @ design,
        x_t_y=design.T @ y,
        y_y=float(y @ y),
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        provenance_sha256=_array_sha256(x, y),
    )


def server_reconstruct_ridge(
    records: Sequence[LocalNormalEquations],
    scaler: GlobalScaler,
    feature_names: Sequence[str],
    alpha: float,
) -> CanonicalRidgeModel:
    _require_exact_records(records, LocalNormalEquations, "server_reconstruct_ridge")
    if len({record.client_id for record in records}) != len(records):
        raise ValueError("normal equations require unique clients")
    if any(record.gas_id != scaler.gas_id or record.role != scaler.role for record in records):
        raise ValueError("normal-equation gas/role mismatch")
    if len(feature_names) != len(scaler.mean):
        raise ValueError("model feature-name dimension mismatch")
    x_t_x = np.sum(np.stack([record.x_t_x for record in records]), axis=0)
    x_t_y = np.sum(np.stack([record.x_t_y for record in records]), axis=0)
    regularizer = np.eye(x_t_x.shape[0], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    coef = np.linalg.pinv(x_t_x + regularizer) @ x_t_y
    return CanonicalRidgeModel(
        gas_id=scaler.gas_id,
        role=scaler.role,
        alpha=float(alpha),
        feature_names=tuple(feature_names),
        mean=scaler.mean.copy(),
        scale=scaler.scale.copy(),
        coef=coef,
        clip_min=min(record.y_min for record in records),
        clip_max=max(record.y_max for record in records),
    )


def federated_fit(
    client_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    role: str,
    feature_names: Sequence[str],
    alpha: float,
) -> tuple[CanonicalRidgeModel, dict[str, Sequence[Any]]]:
    if not client_data:
        raise ValueError("federated fit requires clients")
    moments = [
        client_feature_moments(client, gas_id, role, values)
        for client, (values, _targets) in sorted(client_data.items())
    ]
    scaler = server_aggregate_scaler(moments)
    equations = [
        client_normal_equations(client, gas_id, role, values, targets, scaler)
        for client, (values, targets) in sorted(client_data.items())
    ]
    model = server_reconstruct_ridge(equations, scaler, feature_names, alpha)
    return model, {"moments": moments, "normal_equations": equations}


def pooled_fit(
    values: np.ndarray,
    targets: np.ndarray,
    *,
    gas_id: int,
    role: str,
    feature_names: Sequence[str],
    alpha: float,
) -> CanonicalRidgeModel:
    x = _finite_matrix(values, "pooled X")
    y = _finite_vector(targets, len(x), "pooled y")
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(np.abs(scale) < SCALE_FLOOR, 1.0, scale)
    z = (x - mean) / scale
    design = np.concatenate([np.ones((len(z), 1), dtype=np.float64), z], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + regularizer) @ (design.T @ y)
    return CanonicalRidgeModel(
        gas_id=int(gas_id),
        role=str(role),
        alpha=float(alpha),
        feature_names=tuple(feature_names),
        mean=mean,
        scale=scale,
        coef=coef,
        clip_min=float(np.min(y)),
        clip_max=float(np.max(y)),
    )


def client_validation_score(
    client_id: str,
    gas_id: int,
    role: str,
    values: np.ndarray,
    targets: np.ndarray,
    model: CanonicalRidgeModel,
) -> LocalValidationScore:
    x = _finite_matrix(values, "validation X")
    y = _finite_vector(targets, len(x), "validation y")
    error = model.predict_matrix(x, clip=True) - y
    return LocalValidationScore(
        client_id=str(client_id),
        gas_id=int(gas_id),
        alpha=model.alpha,
        role=str(role),
        n=len(y),
        sse=float(error @ error),
        provenance_sha256=_array_sha256(x, y),
    )


def server_validation_rmse(records: Sequence[LocalValidationScore]) -> float:
    _require_exact_records(records, LocalValidationScore, "server_validation_rmse")
    if len({record.client_id for record in records}) != len(records):
        raise ValueError("validation scores require unique clients")
    if len({(record.gas_id, record.alpha, record.role) for record in records}) != 1:
        raise ValueError("validation scores require one gas/alpha/role")
    return float(np.sqrt(sum(record.sse for record in records) / sum(record.n for record in records)))


def select_source_alpha(
    source_train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_calibration: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    feature_names: Sequence[str],
    alphas: Sequence[float] = RIDGE_ALPHAS,
    train_role: str = "source_train",
    validation_role: str = "source_calibration",
) -> tuple[float, list[dict[str, Any]]]:
    if train_role != "source_train" or validation_role != "source_calibration":
        raise RuntimeError("canonical alpha selection is source-only train/calibration")
    if tuple(float(value) for value in alphas) != tuple(RIDGE_ALPHAS):
        raise RuntimeError("canonical alpha grid differs from the registered source-only grid")
    audit: list[dict[str, Any]] = []
    best_alpha = float(alphas[0])
    best_rmse = float("inf")
    for alpha in alphas:
        model, _stats = federated_fit(
            source_train,
            gas_id=gas_id,
            role=train_role,
            feature_names=feature_names,
            alpha=float(alpha),
        )
        scores = [
            client_validation_score(
                client,
                gas_id,
                validation_role,
                values,
                targets,
                model,
            )
            for client, (values, targets) in sorted(source_calibration.items())
        ]
        score = server_validation_rmse(scores)
        audit.append(
            {
                "gas_id": int(gas_id),
                "alpha": float(alpha),
                "source_calibration_RMSE": score,
                "source_calibration_N": int(sum(item.n for item in scores)),
                "target_input_accessed": False,
                "source_test_accessed": False,
            }
        )
        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)
    return best_alpha, audit
