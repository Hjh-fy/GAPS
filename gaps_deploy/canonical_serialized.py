"""Minimal serialized estimators required by the canonical-v1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


class CanonicalRuntimeError(ValueError):
    pass


def _finite_vector(value: object, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise CanonicalRuntimeError(f"{label} must be a non-empty finite vector")
    return vector


def _feature_vector(features: Mapping[str, object], names: tuple[str, ...]) -> np.ndarray:
    try:
        values = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise CanonicalRuntimeError("required runtime feature is missing or non-numeric") from error
    if not np.all(np.isfinite(values)):
        raise CanonicalRuntimeError("runtime feature must be finite")
    return values


@dataclass(frozen=True)
class SerializedRidge:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    clip_min: float
    clip_max: float

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SerializedRidge":
        names = tuple(str(name) for name in payload.get("feature_names", ()))
        mean, scale, coef = (
            _finite_vector(payload.get(key), key) for key in ("mean", "scale", "coef")
        )
        if not names or len(names) != len(mean) or len(coef) != len(mean) + 1:
            raise CanonicalRuntimeError("serialized Ridge dimensions are inconsistent")
        clip_min, clip_max = float(payload["clip_min"]), float(payload["clip_max"])
        if not math.isfinite(clip_min) or not math.isfinite(clip_max) or clip_min > clip_max:
            raise CanonicalRuntimeError("serialized Ridge clip bounds are invalid")
        return cls(
            names, mean, np.where(np.abs(scale) < 1e-9, 1.0, scale), coef,
            clip_min, clip_max,
        )

    def predict(self, features: Mapping[str, object]) -> float:
        values = _feature_vector(features, self.feature_names)
        result = float(np.concatenate(([1.0], (values - self.mean) / self.scale)) @ self.coef)
        if not math.isfinite(result):
            raise CanonicalRuntimeError("serialized Ridge produced a non-finite prediction")
        return float(np.clip(result, self.clip_min, self.clip_max))


@dataclass(frozen=True)
class SerializedMLP:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefs: tuple[np.ndarray, ...]
    intercepts: tuple[np.ndarray, ...]
    activation: str
    clip_min: float
    clip_max: float
    append_route_onehot: bool = False

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SerializedMLP":
        names = tuple(str(name) for name in payload.get("feature_names", ()))
        mean, scale = (_finite_vector(payload.get(key), key) for key in ("mean", "scale"))
        coefs = tuple(np.asarray(item, dtype=np.float64) for item in payload.get("coefs", ()))
        intercepts = tuple(
            np.asarray(item, dtype=np.float64).reshape(-1)
            for item in payload.get("intercepts", ())
        )
        append_route_onehot = len(mean) == len(names) + 4
        if not names or len(mean) not in {len(names), len(names) + 4} or len(coefs) != len(intercepts) or not coefs:
            raise CanonicalRuntimeError("serialized MLP dimensions are inconsistent")
        width = len(mean)
        for weights, bias in zip(coefs, intercepts):
            if weights.ndim != 2 or weights.shape != (width, len(bias)):
                raise CanonicalRuntimeError("serialized MLP layer dimensions are inconsistent")
            width = len(bias)
        if width != 1:
            raise CanonicalRuntimeError("serialized MLP output must be scalar")
        return cls(
            names, mean, np.where(np.abs(scale) < 1e-9, 1.0, scale), coefs,
            intercepts, str(payload.get("activation")), float(payload["clip_min"]),
            float(payload["clip_max"]), append_route_onehot,
        )

    def predict(self, features: Mapping[str, object]) -> float:
        values = _feature_vector(features, self.feature_names)
        if self.append_route_onehot:
            route = int(features["route_class"])
            if route not in (0, 1, 2, 3):
                raise CanonicalRuntimeError("shared MLP route_class is outside 0..3")
            onehot = np.zeros(4, dtype=np.float64)
            onehot[route] = 1.0
            values = np.concatenate((values, onehot))
        values = (values - self.mean) / self.scale
        for index, (weights, bias) in enumerate(zip(self.coefs, self.intercepts)):
            values = values @ weights + bias
            if index < len(self.coefs) - 1:
                if self.activation != "relu":
                    raise CanonicalRuntimeError(
                        f"unsupported serialized MLP activation: {self.activation}"
                    )
                values = np.maximum(values, 0.0)
        return float(np.clip(float(values[0]), self.clip_min, self.clip_max))


@dataclass(frozen=True)
class FixedH8Policy:
    source_ridge: Mapping[int, SerializedRidge]
    source_mlp: Mapping[int, SerializedMLP]
    shared_mlp: SerializedMLP
    target_ridge: Mapping[int, SerializedRidge]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FixedH8Policy":
        source, models = payload.get("source_heads"), payload.get("models")
        if not isinstance(source, Mapping) or not isinstance(models, list):
            raise CanonicalRuntimeError("fixed H8 policy is malformed")

        def indexed(items: object, factory: Any) -> dict[int, Any]:
            if not isinstance(items, list):
                raise CanonicalRuntimeError("fixed H8 head list is malformed")
            output = {
                int(item["class_id"]): factory.from_json(item)
                for item in items if isinstance(item, Mapping)
            }
            if set(output) != {0, 1, 2, 3}:
                raise CanonicalRuntimeError("fixed H8 policy requires four class heads")
            return output

        shared = source.get("shared_mlp")
        if not isinstance(shared, Mapping):
            raise CanonicalRuntimeError("fixed H8 policy has no shared MLP")
        return cls(
            indexed(source.get("ridge_per_gas"), SerializedRidge),
            indexed(source.get("mlp_per_gas"), SerializedMLP),
            SerializedMLP.from_json(shared),
            indexed(models, SerializedRidge),
        )
