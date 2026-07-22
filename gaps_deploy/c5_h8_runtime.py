"""Numerical primitives for the formal C1/C2-to-C5 fixed-H8 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


class C5H8RuntimeError(ValueError):
    """Raised when a frozen C5/H8 runtime component cannot be evaluated safely."""


def _finite_vector(value: object, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise C5H8RuntimeError(f"{label} must be a non-empty finite vector")
    return vector


def _feature_vector(features: Mapping[str, object], names: tuple[str, ...], mean: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for index, name in enumerate(names):
        try:
            value = float(features.get(name, mean[index]))
        except (TypeError, ValueError):
            value = float(mean[index])
        values.append(value if math.isfinite(value) else float(mean[index]))
    return np.asarray(values, dtype=np.float64)


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
        mean, scale, coef = (_finite_vector(payload.get(key), key) for key in ("mean", "scale", "coef"))
        if not names or len(names) != len(mean) or len(coef) != len(mean) + 1:
            raise C5H8RuntimeError("serialized Ridge dimensions are inconsistent")
        try:
            clip_min, clip_max = float(payload["clip_min"]), float(payload["clip_max"])
        except (KeyError, TypeError, ValueError) as error:
            raise C5H8RuntimeError("serialized Ridge clip bounds are invalid") from error
        if not math.isfinite(clip_min) or not math.isfinite(clip_max) or clip_min > clip_max:
            raise C5H8RuntimeError("serialized Ridge clip bounds are invalid")
        return cls(names, mean, np.where(np.abs(scale) < 1e-9, 1.0, scale), coef, clip_min, clip_max)

    def predict(self, features: Mapping[str, object]) -> float:
        values = _feature_vector(features, self.feature_names, self.mean)
        result = float(np.concatenate(([1.0], (values - self.mean) / self.scale) ) @ self.coef)
        if not math.isfinite(result):
            raise C5H8RuntimeError("serialized Ridge produced a non-finite prediction")
        return float(np.clip(result, self.clip_min, self.clip_max))


@dataclass(frozen=True)
class SerializedMLP:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefs: tuple[np.ndarray, ...]
    intercepts: tuple[np.ndarray, ...]
    activation: str
    out_activation: str
    clip_min: float
    clip_max: float
    append_route_onehot: bool = False

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SerializedMLP":
        names = tuple(str(name) for name in payload.get("feature_names", ()))
        mean, scale = (_finite_vector(payload.get(key), key) for key in ("mean", "scale"))
        coefs = tuple(np.asarray(item, dtype=np.float64) for item in payload.get("coefs", ()))
        intercepts = tuple(np.asarray(item, dtype=np.float64).reshape(-1) for item in payload.get("intercepts", ()))
        append_route_onehot = len(mean) == len(names) + 4
        if not names or len(mean) not in {len(names), len(names) + 4} or len(coefs) != len(intercepts) or not coefs:
            raise C5H8RuntimeError("serialized MLP dimensions are inconsistent")
        width = len(mean)
        for weights, bias in zip(coefs, intercepts):
            if weights.ndim != 2 or weights.shape != (width, len(bias)) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
                raise C5H8RuntimeError("serialized MLP layer dimensions are inconsistent")
            width = len(bias)
        if width != 1:
            raise C5H8RuntimeError("serialized MLP output must be scalar")
        clip_min, clip_max = float(payload["clip_min"]), float(payload["clip_max"])
        return cls(names, mean, np.where(np.abs(scale) < 1e-9, 1.0, scale), coefs, intercepts, str(payload.get("activation")), str(payload.get("out_activation")), clip_min, clip_max, append_route_onehot)

    def predict(self, features: Mapping[str, object]) -> float:
        base_mean = self.mean[: len(self.feature_names)]
        values = _feature_vector(features, self.feature_names, base_mean)
        if self.append_route_onehot:
            try:
                route_class = int(features["route_class"])
            except (KeyError, TypeError, ValueError) as error:
                raise C5H8RuntimeError("shared MLP requires a valid route_class") from error
            if route_class not in (0, 1, 2, 3):
                raise C5H8RuntimeError("shared MLP route_class is outside 0..3")
            onehot = np.zeros(4, dtype=np.float64)
            onehot[route_class] = 1.0
            values = np.concatenate((values, onehot))
        values = (values - self.mean) / self.scale
        for index, (weights, bias) in enumerate(zip(self.coefs, self.intercepts)):
            values = values @ weights + bias
            if index < len(self.coefs) - 1:
                if self.activation != "relu":
                    raise C5H8RuntimeError(f"unsupported serialized MLP activation: {self.activation}")
                values = np.maximum(values, 0.0)
        result = float(values[0])
        if not math.isfinite(result):
            raise C5H8RuntimeError("serialized MLP produced a non-finite prediction")
        return float(np.clip(result, self.clip_min, self.clip_max))
