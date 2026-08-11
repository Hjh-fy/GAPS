"""Stable, mergeable feature moments for canonical FedRidge R0-v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np


FLOAT64_EPS = np.finfo(np.float64).eps
SCALE_FLOOR = 1e-9
CLIENT_ORDER = ("C1", "C2")
CANONICAL_FEATURE_DIMENSIONS = 104


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


def _validate_metadata(client_id: str, gas_id: int, role: str) -> None:
    if not isinstance(client_id, str) or not client_id:
        raise ValueError("client_id must be a non-empty string")
    if isinstance(gas_id, (bool, np.bool_)) or not isinstance(
        gas_id, (int, np.integer)
    ):
        raise ValueError("gas_id must be an integer")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
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
