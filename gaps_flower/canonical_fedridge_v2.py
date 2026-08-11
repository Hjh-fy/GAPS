"""Stable, mergeable feature moments for canonical FedRidge R0-v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


FLOAT64_EPS = np.finfo(np.float64).eps
SCALE_FLOOR = 1e-9
CLIENT_ORDER = ("C1", "C2")
CANONICAL_FEATURE_DIMENSIONS = 104
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class LocalCentralMomentsV2:
    client_id: str
    gas_id: int
    role: str
    n: int
    mean: np.ndarray
    m2: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    provenance_sha256: str

    def __post_init__(self) -> None:
        for field in ("mean", "m2", "minimum", "maximum"):
            object.__setattr__(self, field, _readonly_copy(getattr(self, field)))


@dataclass(frozen=True)
class StableGlobalScalerV2:
    gas_id: int
    role: str
    n: int
    mean: np.ndarray
    variance: np.ndarray
    raw_scale: np.ndarray
    scale: np.ndarray
    safe_scale_mask: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    aggregation_order: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "mean",
            "variance",
            "raw_scale",
            "scale",
            "minimum",
            "maximum",
        ):
            object.__setattr__(self, field, _readonly_copy(getattr(self, field)))
        object.__setattr__(
            self,
            "safe_scale_mask",
            _readonly_copy(self.safe_scale_mask, dtype=np.bool_),
        )


@dataclass(frozen=True)
class LocalNormalEquationsV2:
    client_id: str
    gas_id: int
    role: str
    n: int
    a: np.ndarray
    b: np.ndarray
    y_y: float
    y_min: float
    y_max: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _readonly_copy(self.a))
        object.__setattr__(self, "b", _readonly_copy(self.b))


@dataclass(frozen=True)
class AggregatedNormalEquationsV2:
    gas_id: int
    role: str
    n: int
    a: np.ndarray
    b: np.ndarray
    y_y: float
    y_min: float
    y_max: float
    aggregation_order: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _readonly_copy(self.a))
        object.__setattr__(self, "b", _readonly_copy(self.b))
        object.__setattr__(self, "aggregation_order", tuple(self.aggregation_order))


@dataclass(frozen=True)
class CanonicalRidgeModelV2:
    gas_id: int
    role: str
    alpha: float
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    clip_min: float
    clip_max: float
    intercept_regularized: bool = False

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("model feature names must be non-empty strings")
        if self.intercept_regularized is not False:
            raise ValueError("canonical Ridge intercept must be unregularized")
        for field in ("mean", "scale", "coef"):
            object.__setattr__(self, field, _readonly_copy(getattr(self, field)))
        object.__setattr__(self, "feature_names", names)
        _validate_model_state(self)

    def predict_matrix(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        x = _finite_matrix(values)
        if x.shape[1] != len(self.feature_names):
            raise ValueError("prediction feature dimension mismatch")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (x - self.mean) / self.scale
            design = np.column_stack(
                [np.ones(len(x), dtype=np.float64), standardized]
            )
            prediction = design @ self.coef
        if not np.isfinite(design).all() or not np.isfinite(prediction).all():
            raise ValueError("Ridge prediction overflowed float64")
        if clip:
            prediction = np.clip(prediction, self.clip_min, self.clip_max)
        return np.asarray(prediction, dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return {
            "gas_id": int(self.gas_id),
            "role": self.role,
            "alpha": float(self.alpha),
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
            "intercept_regularized": False,
            "solver": "numpy.linalg.pinv",
        }


def _readonly_copy(values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=np.dtype(dtype)).reshape(array.shape)


def _finite_matrix(values: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be a non-empty finite 2D matrix") from error
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] == 0
        or not np.isfinite(array).all()
    ):
        raise ValueError("values must be a non-empty finite 2D matrix")
    return array


def _finite_vector(values: np.ndarray, n: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite 1D vector aligned to X") from error
    if array.ndim != 1 or len(array) != n or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 1D vector aligned to X")
    return array


def _validate_metadata(client_id: str, gas_id: int, role: str) -> None:
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("client_id must be a non-empty string")
    if isinstance(gas_id, (bool, np.bool_)) or not isinstance(
        gas_id, (int, np.integer)
    ):
        raise ValueError("gas_id must be an integer")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def local_central_moments(
    client_id: str, gas_id: int, role: str, values: np.ndarray
) -> LocalCentralMomentsV2:
    """Summarize one client's finite feature matrix with two-pass moments."""
    _validate_metadata(client_id, gas_id, role)
    x = _finite_matrix(values)
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary_mean = np.mean(x, axis=0, dtype=np.float64)
        fallback_scale = np.max(np.abs(x), axis=0)
        normalized_mean = np.mean(
            x / fallback_scale, axis=0, dtype=np.float64
        )
        fallback_mean = np.clip(normalized_mean, -1.0, 1.0) * fallback_scale
        mean = np.where(np.isfinite(ordinary_mean), ordinary_mean, fallback_mean)
        m2 = np.sum((x - mean) ** 2, axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(m2).all():
        raise ValueError("local central moments overflowed float64")
    return LocalCentralMomentsV2(
        client_id=client_id,
        gas_id=int(gas_id),
        role=role,
        n=int(x.shape[0]),
        mean=_readonly_copy(mean),
        m2=_readonly_copy(m2),
        minimum=_readonly_copy(np.min(x, axis=0)),
        maximum=_readonly_copy(np.max(x, axis=0)),
        provenance_sha256=_array_sha256(x),
    )


def _validated_records(
    records: Sequence[LocalCentralMomentsV2],
    expected_client_order: Sequence[str],
) -> tuple[list[LocalCentralMomentsV2], tuple[str, ...]]:
    items = list(records)
    order = tuple(expected_client_order)
    if not order or any(not isinstance(client_id, str) or not client_id for client_id in order):
        raise ValueError("expected client order must contain non-empty client IDs")
    if len(set(order)) != len(order):
        raise ValueError("expected client order contains duplicate clients")
    if not items or any(type(record) is not LocalCentralMomentsV2 for record in items):
        raise TypeError("merge_central_moments accepts LocalCentralMomentsV2 records")

    counts = Counter(record.client_id for record in items)
    duplicates = sorted(client_id for client_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate clients: {duplicates}")
    missing = [client_id for client_id in order if client_id not in counts]
    if missing:
        raise ValueError(f"missing clients: {missing}")
    extras = sorted(client_id for client_id in counts if client_id not in set(order))
    if extras:
        raise ValueError(f"extra clients: {extras}")

    rank = {client_id: index for index, client_id in enumerate(order)}
    items.sort(key=lambda record: rank[record.client_id])
    first = items[0]
    if any(record.gas_id != first.gas_id for record in items[1:]):
        raise ValueError("gas mismatch between central-moment records")
    if any(record.role != first.role for record in items[1:]):
        raise ValueError("role mismatch between central-moment records")

    dimension = first.mean.shape
    fields = ("mean", "m2", "minimum", "maximum")
    for record in items:
        if (
            isinstance(record.n, (bool, np.bool_))
            or not isinstance(record.n, (int, np.integer))
            or record.n <= 0
        ):
            raise ValueError("central-moment record n must be positive")
        for field in fields:
            values = getattr(record, field)
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.float64
                or values.ndim != 1
                or values.shape != dimension
                or values.size == 0
                or not np.isfinite(values).all()
            ):
                raise ValueError("dimension or dtype mismatch in central-moment records")
        if np.any(record.m2 < 0.0):
            raise ValueError("central-moment M2 must be nonnegative")
        if np.any(record.minimum > record.maximum):
            raise ValueError("central-moment minimum exceeds maximum")
    return items, order


def merge_central_moments(
    records: Sequence[LocalCentralMomentsV2],
    *,
    expected_client_order: Sequence[str] = CLIENT_ORDER,
) -> StableGlobalScalerV2:
    """Merge local moments deterministically with the Chan equations."""
    items, order = _validated_records(records, expected_client_order)
    first = items[0]
    n = int(first.n)
    mean = np.array(first.mean, dtype=np.float64, copy=True)
    m2 = np.array(first.m2, dtype=np.float64, copy=True)
    minimum = np.array(first.minimum, dtype=np.float64, copy=True)
    maximum = np.array(first.maximum, dtype=np.float64, copy=True)

    with np.errstate(over="ignore", invalid="ignore"):
        for record in items[1:]:
            total_n = n + int(record.n)
            delta = record.mean - mean
            m2 = m2 + record.m2 + delta * delta * (n * record.n / total_n)
            mean = mean + delta * (record.n / total_n)
            minimum = np.minimum(minimum, record.minimum)
            maximum = np.maximum(maximum, record.maximum)
            n = total_n

        variance = np.maximum(m2 / np.float64(n), np.float64(0.0))
        raw_scale = np.sqrt(variance)
    if not all(
        np.isfinite(values).all()
        for values in (mean, m2, variance, raw_scale, minimum, maximum)
    ):
        raise ValueError("merged central moments overflowed float64")
    safe_scale_mask = raw_scale < SCALE_FLOOR
    scale = np.where(safe_scale_mask, np.float64(1.0), raw_scale)
    return StableGlobalScalerV2(
        gas_id=first.gas_id,
        role=first.role,
        n=n,
        mean=_readonly_copy(mean),
        variance=_readonly_copy(variance),
        raw_scale=_readonly_copy(raw_scale),
        scale=_readonly_copy(scale),
        safe_scale_mask=_readonly_copy(safe_scale_mask, dtype=np.bool_),
        minimum=_readonly_copy(minimum),
        maximum=_readonly_copy(maximum),
        aggregation_order=order,
    )


def _scaler_matches(left: StableGlobalScalerV2, right: StableGlobalScalerV2) -> bool:
    scalar_fields = ("gas_id", "role", "n", "aggregation_order")
    array_fields = (
        "mean",
        "variance",
        "raw_scale",
        "scale",
        "safe_scale_mask",
        "minimum",
        "maximum",
    )
    return all(getattr(left, field) == getattr(right, field) for field in scalar_fields) and all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in array_fields
    )


def feature_numerical_audit_rows(
    records: Sequence[LocalCentralMomentsV2],
    scaler: StableGlobalScalerV2,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Build one numerical-audit row per ordered canonical feature."""
    if type(scaler) is not StableGlobalScalerV2:
        raise TypeError("feature audit requires a StableGlobalScalerV2")
    names = tuple(feature_names)
    if len(scaler.mean) != CANONICAL_FEATURE_DIMENSIONS:
        raise ValueError("feature audit requires exactly 104 dimensions")
    if len(names) != len(scaler.mean):
        raise ValueError("feature-name dimension mismatch")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature names must be non-empty strings")
    reconstructed = merge_central_moments(
        records, expected_client_order=scaler.aggregation_order
    )
    if not _scaler_matches(reconstructed, scaler):
        raise ValueError("feature-audit records do not match scaler")

    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(names):
        rows.append(
            {
                "gas_id": int(scaler.gas_id),
                "role": scaler.role,
                "feature_index": index,
                "feature_name": feature_name,
                "n": int(scaler.n),
                "minimum": float(scaler.minimum[index]),
                "maximum": float(scaler.maximum[index]),
                "mean": float(scaler.mean[index]),
                "population_variance": float(scaler.variance[index]),
                "raw_scale": float(scaler.raw_scale[index]),
                "dynamic_range": float(
                    scaler.maximum[index] - scaler.minimum[index]
                ),
                "safe_scale_floor": float(SCALE_FLOOR),
                "safe_scale_applied": bool(scaler.safe_scale_mask[index]),
                "canonical_scale": float(scaler.scale[index]),
                "aggregation_order": scaler.aggregation_order,
                "dtype": "float64",
            }
        )
    return rows


def _validate_scaler(scaler: StableGlobalScalerV2) -> None:
    if type(scaler) is not StableGlobalScalerV2:
        raise TypeError("normal equations require a StableGlobalScalerV2")
    _validate_metadata("scaler", scaler.gas_id, scaler.role)
    if (
        isinstance(scaler.n, (bool, np.bool_))
        or not isinstance(scaler.n, (int, np.integer))
        or scaler.n <= 0
    ):
        raise ValueError("scaler n must be positive")
    dimension = scaler.mean.shape
    for field in (
        "mean",
        "variance",
        "raw_scale",
        "scale",
        "minimum",
        "maximum",
    ):
        values = getattr(scaler, field)
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.float64
            or values.ndim != 1
            or values.shape != dimension
            or values.size == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid scaler array state")
    if (
        not isinstance(scaler.safe_scale_mask, np.ndarray)
        or scaler.safe_scale_mask.dtype != np.bool_
        or scaler.safe_scale_mask.shape != dimension
    ):
        raise ValueError("invalid scaler safe-scale mask")
    if (
        np.any(scaler.variance < 0.0)
        or np.any(scaler.raw_scale < 0.0)
        or np.any(scaler.scale <= 0.0)
        or np.any(scaler.minimum > scaler.maximum)
    ):
        raise ValueError("invalid scaler numerical state")
    expected_scale = np.where(scaler.safe_scale_mask, 1.0, scaler.raw_scale)
    if not np.array_equal(scaler.scale, expected_scale):
        raise ValueError("invalid scaler canonical scale")
    order = scaler.aggregation_order
    if (
        not isinstance(order, tuple)
        or not order
        or len(set(order)) != len(order)
        or any(not isinstance(client, str) or not client for client in order)
    ):
        raise ValueError("invalid scaler aggregation order")


def local_normal_equations_v2(
    client_id: str,
    gas_id: int,
    role: str,
    values: np.ndarray,
    targets: np.ndarray,
    scaler: StableGlobalScalerV2,
) -> LocalNormalEquationsV2:
    """Build one client's float64 normal equations on the global scaling."""
    _validate_metadata(client_id, gas_id, role)
    _validate_scaler(scaler)
    x = _finite_matrix(values)
    y = _finite_vector(targets, len(x), "normal-equation targets")
    if scaler.gas_id != int(gas_id) or scaler.role != role:
        raise ValueError("normal-equation scaler gas/role mismatch")
    if x.shape[1] != len(scaler.mean):
        raise ValueError("normal-equation feature dimension mismatch")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = (x - scaler.mean) / scaler.scale
        design = np.column_stack(
            [np.ones(len(x), dtype=np.float64), standardized]
        )
        a = design.T @ design
        b = design.T @ y
        y_y = float(y @ y)
    if (
        not np.isfinite(design).all()
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(y_y)
    ):
        raise ValueError("local normal equations overflowed float64")
    return LocalNormalEquationsV2(
        client_id=client_id,
        gas_id=int(gas_id),
        role=role,
        n=int(len(x)),
        a=a,
        b=b,
        y_y=y_y,
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        provenance_sha256=_array_sha256(x, y),
    )


def _validated_equation_records(
    records: Sequence[LocalNormalEquationsV2],
    expected_client_order: Sequence[str],
) -> tuple[list[LocalNormalEquationsV2], tuple[str, ...]]:
    items = list(records)
    order = tuple(expected_client_order)
    if not order or any(not isinstance(client, str) or not client for client in order):
        raise ValueError("expected client order must contain non-empty client IDs")
    if len(set(order)) != len(order):
        raise ValueError("expected client order contains duplicate clients")
    if not items or any(type(record) is not LocalNormalEquationsV2 for record in items):
        raise TypeError(
            "aggregate_normal_equations_v2 accepts LocalNormalEquationsV2 records"
        )
    counts = Counter(record.client_id for record in items)
    duplicates = sorted(client for client, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate clients: {duplicates}")
    missing = [client for client in order if client not in counts]
    if missing:
        raise ValueError(f"missing clients: {missing}")
    extras = sorted(client for client in counts if client not in set(order))
    if extras:
        raise ValueError(f"extra clients: {extras}")
    rank = {client: index for index, client in enumerate(order)}
    items.sort(key=lambda record: rank[record.client_id])
    first = items[0]
    if any(record.gas_id != first.gas_id for record in items[1:]):
        raise ValueError("gas mismatch between normal-equation records")
    if any(record.role != first.role for record in items[1:]):
        raise ValueError("role mismatch between normal-equation records")
    design_dimension = first.b.shape
    for record in items:
        _validate_metadata(record.client_id, record.gas_id, record.role)
        if (
            isinstance(record.n, (bool, np.bool_))
            or not isinstance(record.n, (int, np.integer))
            or record.n <= 0
        ):
            raise ValueError("normal-equation record n must be positive")
        if (
            not isinstance(record.a, np.ndarray)
            or record.a.dtype != np.float64
            or record.a.ndim != 2
            or record.a.shape != (design_dimension[0], design_dimension[0])
            or design_dimension[0] < 2
            or not np.isfinite(record.a).all()
        ):
            raise ValueError("dimension or dtype mismatch in normal-equation records")
        if (
            not isinstance(record.b, np.ndarray)
            or record.b.dtype != np.float64
            or record.b.ndim != 1
            or record.b.shape != design_dimension
            or not np.isfinite(record.b).all()
        ):
            raise ValueError("dimension or dtype mismatch in normal-equation records")
        if not np.array_equal(record.a, record.a.T):
            raise ValueError("normal-equation A must be symmetric")
        scalars = (record.y_y, record.y_min, record.y_max)
        if (
            any(isinstance(value, (bool, np.bool_)) for value in scalars)
            or not np.isfinite(np.asarray(scalars, dtype=np.float64)).all()
            or record.y_y < 0.0
            or record.y_min > record.y_max
        ):
            raise ValueError("invalid normal-equation target summary")
        if not isinstance(record.provenance_sha256, str) or not record.provenance_sha256:
            raise ValueError("normal-equation provenance must be non-empty")
    return items, order


def aggregate_normal_equations_v2(
    records: Sequence[LocalNormalEquationsV2],
    *,
    expected_client_order: Sequence[str] = CLIENT_ORDER,
) -> AggregatedNormalEquationsV2:
    """Aggregate local A/b records sequentially in the registered order."""
    items, order = _validated_equation_records(records, expected_client_order)
    first = items[0]
    a = np.zeros_like(first.a, dtype=np.float64)
    b = np.zeros_like(first.b, dtype=np.float64)
    y_y = np.float64(0.0)
    n = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for record in items:
            a = a + record.a
            b = b + record.b
            y_y = y_y + np.float64(record.y_y)
            n += int(record.n)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(y_y):
        raise ValueError("aggregated normal equations overflowed float64")
    return AggregatedNormalEquationsV2(
        gas_id=first.gas_id,
        role=first.role,
        n=n,
        a=a,
        b=b,
        y_y=float(y_y),
        y_min=min(record.y_min for record in items),
        y_max=max(record.y_max for record in items),
        aggregation_order=order,
    )


def _validate_aggregated_equations(
    equations: AggregatedNormalEquationsV2,
) -> None:
    if type(equations) is not AggregatedNormalEquationsV2:
        raise TypeError("Ridge reconstruction requires AggregatedNormalEquationsV2")
    _validate_metadata("aggregated", equations.gas_id, equations.role)
    if (
        isinstance(equations.n, (bool, np.bool_))
        or not isinstance(equations.n, (int, np.integer))
        or equations.n <= 0
    ):
        raise ValueError("aggregated normal-equation n must be positive")
    if (
        not isinstance(equations.a, np.ndarray)
        or equations.a.dtype != np.float64
        or equations.a.ndim != 2
        or equations.a.shape[0] < 2
        or equations.a.shape[0] != equations.a.shape[1]
        or not np.isfinite(equations.a).all()
        or not np.array_equal(equations.a, equations.a.T)
    ):
        raise ValueError("invalid aggregated normal-equation A")
    if (
        not isinstance(equations.b, np.ndarray)
        or equations.b.dtype != np.float64
        or equations.b.ndim != 1
        or equations.b.shape != (equations.a.shape[0],)
        or not np.isfinite(equations.b).all()
    ):
        raise ValueError("invalid aggregated normal-equation b")
    scalars = (equations.y_y, equations.y_min, equations.y_max)
    if (
        any(isinstance(value, (bool, np.bool_)) for value in scalars)
        or not np.isfinite(np.asarray(scalars, dtype=np.float64)).all()
        or equations.y_y < 0.0
        or equations.y_min > equations.y_max
    ):
        raise ValueError("invalid aggregated target summary")
    order = equations.aggregation_order
    if (
        not isinstance(order, tuple)
        or not order
        or len(set(order)) != len(order)
        or any(not isinstance(client, str) or not client for client in order)
    ):
        raise ValueError("invalid normal-equation aggregation order")


def _validate_model_state(model: CanonicalRidgeModelV2) -> None:
    _validate_metadata("model", model.gas_id, model.role)
    if (
        isinstance(model.alpha, (bool, np.bool_))
        or not isinstance(model.alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(float(model.alpha))
        or float(model.alpha) < 0.0
    ):
        raise ValueError("Ridge alpha must be finite and nonnegative")
    dimension = len(model.feature_names)
    if (
        model.mean.dtype != np.float64
        or model.mean.ndim != 1
        or model.mean.shape != (dimension,)
        or not np.isfinite(model.mean).all()
        or model.scale.dtype != np.float64
        or model.scale.ndim != 1
        or model.scale.shape != (dimension,)
        or not np.isfinite(model.scale).all()
        or np.any(model.scale <= 0.0)
        or model.coef.dtype != np.float64
        or model.coef.ndim != 1
        or model.coef.shape != (dimension + 1,)
        or not np.isfinite(model.coef).all()
    ):
        raise ValueError("invalid Ridge model array state")
    bounds = (model.clip_min, model.clip_max)
    if (
        any(isinstance(value, (bool, np.bool_)) for value in bounds)
        or not np.isfinite(np.asarray(bounds, dtype=np.float64)).all()
        or model.clip_min > model.clip_max
    ):
        raise ValueError("invalid Ridge clipping bounds")


def reconstruct_ridge_v2(
    equations: AggregatedNormalEquationsV2,
    scaler: StableGlobalScalerV2,
    feature_names: Sequence[str],
    alpha: float,
) -> CanonicalRidgeModelV2:
    """Reconstruct Ridge with an explicitly unregularized intercept."""
    _validate_aggregated_equations(equations)
    _validate_scaler(scaler)
    names = tuple(feature_names)
    if len(names) != len(scaler.mean) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("model feature-name dimension mismatch")
    if (
        equations.gas_id != scaler.gas_id
        or equations.role != scaler.role
        or equations.n != scaler.n
        or equations.aggregation_order != scaler.aggregation_order
    ):
        raise ValueError("normal-equation/scaler state mismatch")
    if equations.a.shape != (len(names) + 1, len(names) + 1):
        raise ValueError("normal-equation feature dimension mismatch")
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(float(alpha))
        or float(alpha) < 0.0
    ):
        raise ValueError("Ridge alpha must be finite and nonnegative")
    regularizer = np.eye(equations.a.shape[0], dtype=np.float64)
    regularizer[0, 0] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        system = equations.a + np.float64(alpha) * regularizer
    if not np.isfinite(system).all():
        raise ValueError("regularized normal equations overflowed float64")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coef = np.linalg.pinv(system) @ equations.b
    except np.linalg.LinAlgError as error:
        raise ValueError("Ridge pseudoinverse failed") from error
    if not np.isfinite(coef).all():
        raise ValueError("Ridge reconstruction produced non-finite coefficients")
    return CanonicalRidgeModelV2(
        gas_id=equations.gas_id,
        role=equations.role,
        alpha=float(alpha),
        feature_names=names,
        mean=scaler.mean,
        scale=scaler.scale,
        coef=coef,
        clip_min=float(equations.y_min),
        clip_max=float(equations.y_max),
        intercept_regularized=False,
    )


def pooled_reference_fit_v2(
    values: np.ndarray,
    targets: np.ndarray,
    *,
    gas_id: int,
    role: str,
    feature_names: Sequence[str],
    alpha: float,
) -> tuple[CanonicalRidgeModelV2, AggregatedNormalEquationsV2]:
    """Fit the independent pooled reference over one exact concatenated row set."""
    _validate_metadata("POOLED", gas_id, role)
    x = _finite_matrix(values)
    y = _finite_vector(targets, len(x), "pooled targets")
    moment = local_central_moments("POOLED", gas_id, role, x)
    scaler = merge_central_moments(
        [moment], expected_client_order=("POOLED",)
    )
    local = local_normal_equations_v2(
        "POOLED", gas_id, role, x, y, scaler
    )
    equations = aggregate_normal_equations_v2(
        [local], expected_client_order=("POOLED",)
    )
    return (
        reconstruct_ridge_v2(equations, scaler, feature_names, alpha),
        equations,
    )


def _validate_alpha_selection_contract(
    source_train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_calibration: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    train_role: str,
    validation_role: str,
    alphas: Sequence[float],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    tuple[float, ...],
]:
    if train_role != "source_train" or validation_role != "source_calibration":
        raise RuntimeError(
            "canonical alpha selection is source-only train/calibration"
        )
    try:
        raw_alphas = tuple(alphas)
        if any(isinstance(value, (bool, np.bool_)) for value in raw_alphas):
            raise ValueError("boolean alpha")
        alpha_grid = tuple(float(value) for value in raw_alphas)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "canonical alpha grid differs from the registered source-only grid"
        ) from error
    if alpha_grid != RIDGE_ALPHAS:
        raise RuntimeError(
            "canonical alpha grid differs from the registered source-only grid"
        )

    normalized: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for label, mapping in (
        ("source_train", source_train),
        ("source_calibration", source_calibration),
    ):
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{label} must be a client mapping")
        keys = tuple(mapping.keys())
        if any(
            isinstance(key, str)
            and any(token in key.casefold() for token in ("target", "test"))
            for key in keys
        ):
            raise RuntimeError(
                "canonical alpha selection is source-only; target/test mapping key rejected"
            )
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{label} client keys must be non-empty strings")
        if len(keys) != len(set(keys)) or set(keys) != set(CLIENT_ORDER):
            raise ValueError(f"{label} requires exactly C1 and C2")
        ordered: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for client in CLIENT_ORDER:
            pair = mapping[client]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"{label}[{client}] must be an (X, y) pair")
            x = _finite_matrix(pair[0])
            y = _finite_vector(pair[1], len(x), f"{label}[{client}] targets")
            ordered[client] = (x, y)
        normalized.append(ordered)
    return normalized[0], normalized[1], alpha_grid


def _federated_training_components_v2(
    source_train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    role: str,
) -> tuple[StableGlobalScalerV2, AggregatedNormalEquationsV2]:
    moments = [
        local_central_moments(client, gas_id, role, source_train[client][0])
        for client in CLIENT_ORDER
    ]
    scaler = merge_central_moments(moments)
    local = [
        local_normal_equations_v2(
            client,
            gas_id,
            role,
            source_train[client][0],
            source_train[client][1],
            scaler,
        )
        for client in CLIENT_ORDER
    ]
    return scaler, aggregate_normal_equations_v2(local)


def _pooled_training_components_v2(
    source_train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    role: str,
) -> tuple[StableGlobalScalerV2, AggregatedNormalEquationsV2]:
    x = np.vstack([source_train[client][0] for client in CLIENT_ORDER])
    y = np.concatenate([source_train[client][1] for client in CLIENT_ORDER])
    moment = local_central_moments("POOLED", gas_id, role, x)
    scaler = merge_central_moments(
        [moment], expected_client_order=("POOLED",)
    )
    local = local_normal_equations_v2(
        "POOLED", gas_id, role, x, y, scaler
    )
    equations = aggregate_normal_equations_v2(
        [local], expected_client_order=("POOLED",)
    )
    return scaler, equations


def _clipped_sse(model: CanonicalRidgeModelV2, x: np.ndarray, y: np.ndarray) -> float:
    prediction = model.predict_matrix(x, clip=True)
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction - y
        sse = float(error @ error)
    if not np.isfinite(error).all() or not np.isfinite(sse) or sse < 0.0:
        raise ValueError("source-calibration SSE overflowed float64")
    return sse


def _audit_alpha(
    gas_id: int, alpha: float, sse: float, n: int
) -> dict[str, Any]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rmse = float(np.sqrt(np.float64(sse) / np.float64(n)))
    if n <= 0 or not np.isfinite(rmse):
        raise ValueError("source-calibration RMSE is non-finite")
    return {
        "gas_id": int(gas_id),
        "alpha": float(alpha),
        "source_calibration_RMSE": rmse,
        "source_calibration_N": int(n),
        "target_input_accessed": False,
        "source_test_accessed": False,
    }


def select_source_alpha_v2(
    source_train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_calibration: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    feature_names: Sequence[str],
    alphas: Sequence[float] = RIDGE_ALPHAS,
    train_role: str = "source_train",
    validation_role: str = "source_calibration",
) -> tuple[float, list[dict[str, Any]]]:
    """Select alpha from distributed source calibration SSE/count only."""
    train, calibration, alpha_grid = _validate_alpha_selection_contract(
        source_train,
        source_calibration,
        train_role=train_role,
        validation_role=validation_role,
        alphas=alphas,
    )
    _validate_metadata("selection", gas_id, train_role)
    scaler, equations = _federated_training_components_v2(
        train, gas_id=gas_id, role=train_role
    )
    audit: list[dict[str, Any]] = []
    best_alpha = alpha_grid[0]
    best_rmse = float("inf")
    for alpha in alpha_grid:
        model = reconstruct_ridge_v2(
            equations, scaler, feature_names, alpha
        )
        total_sse = np.float64(0.0)
        total_n = 0
        with np.errstate(over="ignore", invalid="ignore"):
            for client in CLIENT_ORDER:
                x, y = calibration[client]
                total_sse = total_sse + np.float64(_clipped_sse(model, x, y))
                total_n += len(y)
        if not np.isfinite(total_sse):
            raise ValueError("distributed source-calibration SSE overflowed float64")
        row = _audit_alpha(gas_id, alpha, float(total_sse), total_n)
        audit.append(row)
        if row["source_calibration_RMSE"] < best_rmse:
            best_rmse = row["source_calibration_RMSE"]
            best_alpha = alpha
    return float(best_alpha), audit


def select_pooled_alpha_v2(
    source_train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_calibration: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    gas_id: int,
    feature_names: Sequence[str],
    alphas: Sequence[float] = RIDGE_ALPHAS,
    train_role: str = "source_train",
    validation_role: str = "source_calibration",
) -> tuple[float, list[dict[str, Any]]]:
    """Select pooled alpha on the same canonical source rows as federation."""
    train, calibration, alpha_grid = _validate_alpha_selection_contract(
        source_train,
        source_calibration,
        train_role=train_role,
        validation_role=validation_role,
        alphas=alphas,
    )
    _validate_metadata("selection", gas_id, train_role)
    scaler, equations = _pooled_training_components_v2(
        train, gas_id=gas_id, role=train_role
    )
    calibration_x = np.vstack(
        [calibration[client][0] for client in CLIENT_ORDER]
    )
    calibration_y = np.concatenate(
        [calibration[client][1] for client in CLIENT_ORDER]
    )
    audit: list[dict[str, Any]] = []
    best_alpha = alpha_grid[0]
    best_rmse = float("inf")
    for alpha in alpha_grid:
        model = reconstruct_ridge_v2(
            equations, scaler, feature_names, alpha
        )
        row = _audit_alpha(
            gas_id,
            alpha,
            _clipped_sse(model, calibration_x, calibration_y),
            len(calibration_y),
        )
        audit.append(row)
        if row["source_calibration_RMSE"] < best_rmse:
            best_rmse = row["source_calibration_RMSE"]
            best_alpha = alpha
    return float(best_alpha), audit
