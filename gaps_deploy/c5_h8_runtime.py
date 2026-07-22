"""Numerical primitives for the formal C1/C2-to-C5 fixed-H8 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from model import FedGasBaseModel

from .c5_h8_bundle import C5H8Bundle, load_c5_h8_bundle
from .package_contract import load_checkpoint_state, load_state_dict_strict
from .rich_residual import target_ridge_features


class C5H8RuntimeError(ValueError):
    """Raised when a frozen C5/H8 runtime component cannot be evaluated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: object, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise C5H8RuntimeError(f"{label} must be a non-empty finite vector")
    return vector


def _feature_vector(features: Mapping[str, object], names: tuple[str, ...]) -> np.ndarray:
    values: list[float] = []
    for name in names:
        if name not in features:
            raise C5H8RuntimeError(f"required runtime feature is missing: {name}")
        try:
            value = float(features[name])
        except (TypeError, ValueError) as error:
            raise C5H8RuntimeError(f"runtime feature is not numeric: {name}") from error
        if not math.isfinite(value):
            raise C5H8RuntimeError(f"runtime feature is not finite: {name}")
        values.append(value)
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
        values = _feature_vector(features, self.feature_names)
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
        values = _feature_vector(features, self.feature_names)
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


@dataclass(frozen=True)
class FixedH8Policy:
    """Frozen R4/H8 source-head augmentation and C5 predicted-class Ridge route."""

    source_ridge: Mapping[int, SerializedRidge]
    source_mlp: Mapping[int, SerializedMLP]
    shared_mlp: SerializedMLP
    target_ridge: Mapping[int, SerializedRidge]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FixedH8Policy":
        source = payload.get("source_heads")
        models = payload.get("models")
        if not isinstance(source, Mapping) or not isinstance(models, list):
            raise C5H8RuntimeError("fixed H8 policy is malformed")
        def indexed(items: object, factory: Any) -> dict[int, Any]:
            if not isinstance(items, list): raise C5H8RuntimeError("fixed H8 head list is malformed")
            output = {int(item["class_id"]): factory.from_json(item) for item in items if isinstance(item, Mapping)}
            if set(output) != {0, 1, 2, 3}: raise C5H8RuntimeError("fixed H8 policy requires four class heads")
            return output
        shared = source.get("shared_mlp")
        if not isinstance(shared, Mapping): raise C5H8RuntimeError("fixed H8 policy has no shared MLP")
        return cls(indexed(source.get("ridge_per_gas"), SerializedRidge), indexed(source.get("mlp_per_gas"), SerializedMLP), SerializedMLP.from_json(shared), indexed(models, SerializedRidge))

    def predict_components(self, features: Mapping[str, object], predicted_class: int) -> dict[str, float]:
        if predicted_class not in self.target_ridge:
            raise C5H8RuntimeError("fixed H8 predicted class is outside 0..3")
        values = dict(features); values["route_class"] = predicted_class
        h1 = self.source_ridge[predicted_class].predict(values)
        h2 = self.source_mlp[predicted_class].predict(values)
        h3 = self.shared_mlp.predict(values)
        values["srcpred_H1_source_ridge_ppm"] = h1
        values["srcpred_H2_source_per_gas_mlp_ppm"] = h2
        values["srcpred_H3_source_shared_mlp_ppm"] = h3
        return {
            "H1_source_ridge_ppm": h1,
            "H2_source_per_gas_mlp_ppm": h2,
            "H3_source_shared_mlp_ppm": h3,
            "target_ridge_plus_source_preds_ppm": self.target_ridge[predicted_class].predict(values),
        }

    def predict(self, features: Mapping[str, object], predicted_class: int) -> float:
        return self.predict_components(features, predicted_class)["target_ridge_plus_source_preds_ppm"]


@dataclass(frozen=True)
class H23Policy:
    """Frozen C5 H2.3 MLP anchor plus weak-Ridge convex blend."""

    mlp: Mapping[int, SerializedMLP]
    ridge: Mapping[int, SerializedRidge]
    blend_weight: float

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "H23Policy":
        def indexed(items: object, factory: Any) -> dict[int, Any]:
            if not isinstance(items, list):
                raise C5H8RuntimeError("H2.3 head list is malformed")
            output = {int(item["class_id"]): factory.from_json(item) for item in items if isinstance(item, Mapping)}
            if set(output) != {0, 1, 2, 3}:
                raise C5H8RuntimeError("H2.3 policy requires four class heads")
            return output

        try:
            weight = float(payload["blend_weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise C5H8RuntimeError("H2.3 blend weight is invalid") from error
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise C5H8RuntimeError("H2.3 blend weight is invalid")
        if (
            payload.get("anchor") != "per_gas_mlp"
            or payload.get("secondary") != "regfeat_ridge"
            or payload.get("target_client") != "C5"
        ):
            raise C5H8RuntimeError("H2.3 policy semantics differ")
        return cls(indexed(payload.get("mlp_models"), SerializedMLP), indexed(payload.get("ridge_models"), SerializedRidge), weight)

    def predict_components(self, features: Mapping[str, object], predicted_class: int) -> dict[str, float]:
        if predicted_class not in self.mlp:
            raise C5H8RuntimeError("H2.3 predicted class is outside 0..3")
        anchor = self.mlp[predicted_class].predict(features)
        weak_ridge = self.ridge[predicted_class].predict(features)
        blended = anchor + self.blend_weight * (weak_ridge - anchor)
        if not math.isfinite(blended):
            raise C5H8RuntimeError("H2.3 blend produced a non-finite prediction")
        return {"h23_anchor_ppm": anchor, "h23_weak_ridge_ppm": weak_ridge, "h23_plus_ppm": blended}


class C5H8Runtime:
    """Strict B5 classifier loader for the versioned C5/H8 runtime contract."""

    def __init__(self, model: FedGasBaseModel, device: str = "cpu", *, bundle: C5H8Bundle | None = None, h8_policy: FixedH8Policy | None = None, h23_policy: H23Policy | None = None) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.bundle = bundle
        self.h8_policy = h8_policy
        self.h23_policy = h23_policy

    @classmethod
    def from_runtime_contract(cls, contract_path: Path, device: str = "cpu") -> "C5H8Runtime":
        path = Path(contract_path)
        try:
            contract = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise C5H8RuntimeError(f"invalid runtime contract: {path}") from error
        if contract.get("schema_version") != "iotj.c5_h8_runtime_contract.v1" or contract.get("status") != "ready":
            raise C5H8RuntimeError("runtime contract is not ready")
        manifest = contract.get("bundle_manifest", {})
        if not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str):
            raise C5H8RuntimeError("runtime contract has no bundle manifest")
        manifest_path = Path(manifest["path"])
        if _sha256(manifest_path) != manifest.get("sha256"):
            raise C5H8RuntimeError("bundle manifest hash differs from runtime contract")
        bundle = load_c5_h8_bundle(manifest_path.parent)
        classifier = bundle.manifest.get("assets", {}).get("classifier", {})
        checkpoint = bundle.asset_paths["classifier"]
        config = dict(contract.get("classifier_model", {}))
        if config.pop("architecture", None) != "FedGasBaseModel":
            raise C5H8RuntimeError("runtime contract classifier architecture differs")
        try:
            model = FedGasBaseModel(**config)
        except TypeError as error:
            raise C5H8RuntimeError("runtime contract classifier configuration is invalid") from error
        _, state = load_checkpoint_state(checkpoint)
        load_state_dict_strict(model, state, checkpoint)
        r4_payload = json.loads(bundle.asset_paths["r4_policy"].read_text(encoding="utf-8"))
        h23_payload = json.loads(bundle.asset_paths["h23_reference"].read_text(encoding="utf-8"))
        classifier_hash = classifier.get("sha256")
        if r4_payload.get("classifier_sha256") != classifier_hash or h23_payload.get("classifier_sha256") != classifier_hash:
            raise C5H8RuntimeError("expert policy classifier hash differs")
        return cls(
            model,
            device=device,
            bundle=bundle,
            h8_policy=FixedH8Policy.from_json(r4_payload["source_aug_target_ridge_policy"]),
            h23_policy=H23Policy.from_json(h23_payload["h23_reference_policy"]),
        )

    def extract_backbone(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        if values.ndim != 3 or values.shape[1:] != (100, 8) or not np.all(np.isfinite(values)):
            raise C5H8RuntimeError("classifier windows must be finite (N,100,8) float32")
        with torch.no_grad():
            logits_tensor, cls_tensor, reg_tensor = self.model(torch.from_numpy(values).to(self.device))
            probabilities_tensor = torch.softmax(logits_tensor, dim=1)
        arrays = tuple(item.detach().cpu().numpy() for item in (logits_tensor, probabilities_tensor, cls_tensor, reg_tensor))
        logits, probabilities, cls_features, reg_features = arrays
        if logits.shape != (len(values), 4) or probabilities.shape != logits.shape or cls_features.shape != (len(values), 64) or reg_features.shape != (len(values), 64):
            raise C5H8RuntimeError("classifier produced invalid backbone shapes")
        if not all(np.isfinite(item).all() for item in arrays):
            raise C5H8RuntimeError("classifier produced non-finite backbone outputs")
        predicted = np.argmax(logits, axis=1).astype(np.int64)
        return logits, probabilities, predicted, cls_features, reg_features

    def classify(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, _probabilities, predicted, _cls_features, _reg_features = self.extract_backbone(windows)
        return logits, predicted

    def infer_experts(self, windows: np.ndarray, metadata: list[Mapping[str, Any]], phases: np.ndarray) -> list[dict[str, Any]]:
        if self.h8_policy is None or self.h23_policy is None:
            raise C5H8RuntimeError("runtime expert policies are not loaded")
        values = np.asarray(windows, dtype=np.float32)
        phase_values = np.asarray(phases)
        if values.ndim != 3 or len(metadata) != len(values) or phase_values.shape != (len(values),):
            raise C5H8RuntimeError("runtime windows, metadata, and phases are not aligned")
        if not np.issubdtype(phase_values.dtype, np.integer) or not np.isin(phase_values, (0, 1, 2)).all():
            raise C5H8RuntimeError("runtime phases must be integer values within 0..2")
        logits, probabilities, predicted, cls_features, reg_features = self.extract_backbone(values)
        rows: list[dict[str, Any]] = []
        for index in range(len(values)):
            if not isinstance(metadata[index], Mapping):
                raise C5H8RuntimeError(f"runtime metadata row is invalid: {index}")
            meta = dict(metadata[index]); meta["phase"] = int(phase_values[index])
            features = target_ridge_features(values[index], meta)
            features.update({f"reg_feat_{j:03d}": float(value) for j, value in enumerate(reg_features[index])})
            route = int(predicted[index])
            h8 = self.h8_policy.predict_components(features, route)
            h23 = self.h23_policy.predict_components(features, route)
            ordered = np.sort(probabilities[index])
            entropy = float(-(probabilities[index] * np.log(np.maximum(probabilities[index], 1e-12))).sum())
            row: dict[str, Any] = {
                "sample_index": index,
                "filename": str(meta.get("filename", "")),
                "repeat_id": meta.get("repeat_id", ""),
                "phase": int(phase_values[index]),
                "pred_class": route,
                "route_class": route,
                "confidence": float(ordered[-1]),
                "confidence_margin": float(ordered[-1] - ordered[-2]),
                "deployment_risk_classifier_entropy": entropy / math.log(4.0),
                "deployment_risk_margin": max(0.0, 1.0 - float(ordered[-1] - ordered[-2])),
                **h8,
                **h23,
            }
            row.update({f"prob_{j}": float(probabilities[index, j]) for j in range(4)})
            row.update({f"cls_feat_{j:03d}": float(value) for j, value in enumerate(cls_features[index])})
            rows.append(row)
        return rows
