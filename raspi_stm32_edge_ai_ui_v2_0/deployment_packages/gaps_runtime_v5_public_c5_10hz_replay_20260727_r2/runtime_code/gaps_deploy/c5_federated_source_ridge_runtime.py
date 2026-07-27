"""Independent B5 -> Federated H1 -> C5 Ridge runtime candidate.

This module intentionally has no H2/H3, legacy rescue, risk, or QC dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from model import FedGasBaseModel

from .c5_federated_source_ridge_bundle import (
    FederatedSourceRidgeBundle,
    FederatedSourceRidgeBundleError,
    load_federated_source_ridge_bundle,
    sha256_file,
)
from .package_contract import load_checkpoint_state, load_state_dict_strict
from .rich_residual import target_ridge_features


class C5FederatedSourceRidgeRuntimeError(RuntimeError):
    pass


CONTRACT_KEYS = {"schema_version", "status", "bundle_manifest", "classifier_model", "inputs", "outputs", "qc_status", "offline_reference"}
DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
OUTPUT_FIELDS = ["sample_index", "pred_class", "source_h1_ppm", "prediction_ppm", "max_probability", "qc_status", "auto_output_ppm"]
RUNTIME_ASSET_KEYS = {"classifier", "federated_h1", "target_ridge"}
CLASSIFIER_MODEL_KEYS = {
    "architecture",
    "num_sensors",
    "num_classes",
    "feat_dim",
    "encoder_type",
    "tcn_norm",
    "use_cls_proj",
}


@dataclass(frozen=True)
class SerializedRidgeV5:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    clip_min: float
    clip_max: float
    alpha: float

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], dimension: int) -> "SerializedRidgeV5":
        names = list(payload.get("feature_names", []))
        mean = np.asarray(payload.get("mean"), dtype=np.float64)
        scale = np.asarray(payload.get("scale"), dtype=np.float64)
        coef = np.asarray(payload.get("coef"), dtype=np.float64)
        if len(names) != dimension or len(set(names)) != dimension or mean.shape != (dimension,) or scale.shape != (dimension,) or coef.shape != (dimension + 1,):
            raise C5FederatedSourceRidgeRuntimeError("Ridge schema dimension differs")
        scalars = [payload.get("alpha"), payload.get("clip_min"), payload.get("clip_max")]
        if not np.isfinite(np.concatenate((mean, scale, coef))).all() or not all(math.isfinite(float(value)) for value in scalars):
            raise C5FederatedSourceRidgeRuntimeError("Ridge contains NaN/Inf")
        return cls(names, mean, scale, coef, float(scalars[1]), float(scalars[2]), float(scalars[0]))

    def predict(self, features: Mapping[str, Any]) -> float:
        if set(features) != set(self.feature_names):
            raise C5FederatedSourceRidgeRuntimeError("Ridge input feature schema differs")
        values = np.asarray([float(features[name]) for name in self.feature_names], dtype=np.float64)
        if not np.isfinite(values).all():
            raise C5FederatedSourceRidgeRuntimeError("Ridge input contains NaN/Inf")
        scale = np.where(np.abs(self.scale) < 1e-9, 1.0, self.scale)
        prediction = float(self.coef[0] + ((values - self.mean) / scale) @ self.coef[1:])
        if not math.isfinite(prediction):
            raise C5FederatedSourceRidgeRuntimeError("Ridge output contains NaN/Inf")
        return float(np.clip(prediction, self.clip_min, self.clip_max))


def _load_heads(path: Path, dimension: int) -> dict[int, SerializedRidgeV5]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise C5FederatedSourceRidgeRuntimeError("Ridge asset is invalid") from error
    records = payload.get("models")
    if not isinstance(records, Mapping) or set(records) != {"0", "1", "2", "3"}:
        raise C5FederatedSourceRidgeRuntimeError("Ridge asset requires exactly four gas heads")
    return {gas: SerializedRidgeV5.from_json(records[str(gas)], dimension) for gas in range(4)}


class C5FederatedSourceRidgeRuntime:
    def __init__(
        self,
        model: torch.nn.Module,
        source_h1: Mapping[int, SerializedRidgeV5],
        target_ridge: Mapping[int, SerializedRidgeV5],
        *,
        device: str = "cpu",
        feature_extractor: Callable[[np.ndarray, Mapping[str, Any]], dict[str, float]] = target_ridge_features,
        bundle: FederatedSourceRidgeBundle | None = None,
        contract: Mapping[str, Any] | None = None,
    ) -> None:
        if set(source_h1) != {0, 1, 2, 3} or set(target_ridge) != {0, 1, 2, 3}:
            raise C5FederatedSourceRidgeRuntimeError("runtime requires four source and target heads")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.source_h1 = dict(source_h1)
        self.target_ridge = dict(target_ridge)
        self.feature_extractor = feature_extractor
        self.bundle = bundle
        self.contract = contract

    @classmethod
    def _from_verified_assets(
        cls,
        asset_paths: Mapping[str, Path],
        classifier_model: Mapping[str, Any],
        *,
        device: str = "cpu",
        bundle: FederatedSourceRidgeBundle | None = None,
        contract: Mapping[str, Any] | None = None,
    ) -> "C5FederatedSourceRidgeRuntime":
        if set(asset_paths) != RUNTIME_ASSET_KEYS:
            raise C5FederatedSourceRidgeRuntimeError(
                "verified runtime assets must be exactly classifier, federated_h1, target_ridge"
            )
        paths = {name: Path(path) for name, path in asset_paths.items()}
        if any(not path.is_file() for path in paths.values()):
            raise C5FederatedSourceRidgeRuntimeError("verified runtime asset is missing")
        config = dict(classifier_model)
        if set(config) != CLASSIFIER_MODEL_KEYS:
            raise C5FederatedSourceRidgeRuntimeError(
                "classifier configuration schema differs"
            )
        if config.pop("architecture", None) != "FedGasBaseModel":
            raise C5FederatedSourceRidgeRuntimeError(
                "classifier architecture differs"
            )
        try:
            model = FedGasBaseModel(**config)
        except TypeError as error:
            raise C5FederatedSourceRidgeRuntimeError(
                "classifier configuration is invalid"
            ) from error
        checkpoint = paths["classifier"]
        _, state = load_checkpoint_state(checkpoint)
        load_state_dict_strict(model, state, checkpoint)
        return cls(
            model,
            _load_heads(paths["federated_h1"], 104),
            _load_heads(paths["target_ridge"], 105),
            device=device,
            bundle=bundle,
            contract=contract,
        )

    @classmethod
    def from_runtime_contract(cls, path: Path, device: str = "cpu") -> "C5FederatedSourceRidgeRuntime":
        contract_path = Path(path)
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise C5FederatedSourceRidgeRuntimeError("runtime contract is invalid") from error
        if set(contract) != CONTRACT_KEYS:
            raise C5FederatedSourceRidgeRuntimeError("runtime contract top-level schema differs")
        if contract.get("schema_version") != "iotj.c5_federated_source_ridge_runtime_contract.v1" or contract.get("status") != "ready":
            raise C5FederatedSourceRidgeRuntimeError("runtime contract is not ready")
        descriptor = contract.get("bundle_manifest")
        if not isinstance(descriptor, Mapping) or set(descriptor) != DESCRIPTOR_KEYS or not isinstance(descriptor.get("path"), str):
            raise C5FederatedSourceRidgeRuntimeError("runtime contract bundle descriptor is invalid")
        manifest_path = Path(descriptor["path"])
        if not manifest_path.is_file() or manifest_path.stat().st_size != descriptor.get("bytes") or sha256_file(manifest_path) != descriptor.get("sha256"):
            raise C5FederatedSourceRidgeRuntimeError("runtime contract bundle identity differs")
        try:
            bundle = load_federated_source_ridge_bundle(manifest_path)
        except FederatedSourceRidgeBundleError as error:
            raise C5FederatedSourceRidgeRuntimeError(str(error)) from error
        inputs = contract.get("inputs")
        if not isinstance(inputs, Mapping) or set(inputs) != {"features", "metadata", "phase_labels", "row_count", "window_shape", "source_dtype", "runtime_dtype"}:
            raise C5FederatedSourceRidgeRuntimeError("runtime input contract schema differs")
        if inputs.get("row_count") != 1360 or inputs.get("window_shape") != [100, 8] or inputs.get("source_dtype") != "float64" or inputs.get("runtime_dtype") != "float32":
            raise C5FederatedSourceRidgeRuntimeError("runtime input contract differs")
        for name in ("features", "metadata", "phase_labels"):
            if not isinstance(inputs[name], Mapping) or set(inputs[name]) != DESCRIPTOR_KEYS:
                raise C5FederatedSourceRidgeRuntimeError(f"runtime input descriptor differs: {name}")
        offline = contract.get("offline_reference")
        if not isinstance(offline, Mapping) or set(offline) != DESCRIPTOR_KEYS:
            raise C5FederatedSourceRidgeRuntimeError("runtime offline reference descriptor differs")
        if contract.get("outputs") != OUTPUT_FIELDS or contract.get("qc_status") != "disabled_pending_dependency_audit":
            raise C5FederatedSourceRidgeRuntimeError("runtime output or QC contract differs")
        return cls._from_verified_assets(
            bundle.asset_paths,
            contract.get("classifier_model", {}),
            device=device,
            bundle=bundle,
            contract=contract,
        )

    def infer(self, windows: np.ndarray, metadata: Sequence[Mapping[str, Any]], phases: np.ndarray) -> list[dict[str, Any]]:
        values = np.asarray(windows, dtype=np.float32)
        phase_values = np.asarray(phases)
        if values.ndim != 3 or values.shape[1:] != (100, 8) or not np.isfinite(values).all():
            raise C5FederatedSourceRidgeRuntimeError("runtime windows must be finite (N,100,8)")
        if len(metadata) != len(values) or phase_values.shape != (len(values),):
            raise C5FederatedSourceRidgeRuntimeError("runtime inputs are not aligned")
        if not np.issubdtype(phase_values.dtype, np.integer) or not np.isin(phase_values, (0, 1, 2)).all():
            raise C5FederatedSourceRidgeRuntimeError("runtime phases must be integer values within 0..2")
        if any(not isinstance(meta, Mapping) for meta in metadata):
            raise C5FederatedSourceRidgeRuntimeError("runtime metadata row is invalid")
        with torch.no_grad():
            logits_tensor, _cls, _reg = self.model(torch.from_numpy(values).to(self.device))
            probabilities_tensor = torch.softmax(logits_tensor, dim=1)
        logits = logits_tensor.detach().cpu().numpy()
        probabilities = probabilities_tensor.detach().cpu().numpy()
        if logits.shape != (len(values), 4) or not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise C5FederatedSourceRidgeRuntimeError("classifier output schema differs")
        predicted = np.argmax(logits, axis=1).astype(np.int64)
        rows: list[dict[str, Any]] = []
        for index, route in enumerate(predicted.tolist()):
            meta = dict(metadata[index])
            meta["phase"] = int(phase_values[index])
            rich = self.feature_extractor(values[index], meta)
            if len(rich) != 104 or not all(math.isfinite(float(value)) for value in rich.values()):
                raise C5FederatedSourceRidgeRuntimeError("rich feature extractor did not emit finite 104D")
            h1 = self.source_h1[route].predict(rich)
            target = dict(rich)
            target["srcpred_H1_source_ridge_ppm"] = h1
            prediction = self.target_ridge[route].predict(target)
            rows.append({
                "sample_index": index,
                "pred_class": route,
                "prediction_ppm": prediction,
                "source_h1_ppm": h1,
                "max_probability": float(np.max(probabilities[index])),
                "qc_status": "disabled_pending_dependency_audit",
                "auto_output_ppm": None,
            })
        return rows
