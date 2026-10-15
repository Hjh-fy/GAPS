"""QC-enabled extension of the B5 -> Federated H1 -> C5 Ridge Runtime v5."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .c5_federated_source_ridge_runtime import (
    C5FederatedSourceRidgeRuntime,
    C5FederatedSourceRidgeRuntimeError,
)
from .runtime_v5_qc import RuntimeV5QCPolicy, descriptor
from .runtime_v5_qc_bundle import RuntimeV5QCBundleError, load_runtime_v5_qc_bundle


class C5FederatedSourceRidgeQCRuntimeError(RuntimeError):
    pass


class C5FederatedSourceRidgeQCRuntime:
    def __init__(self, base: C5FederatedSourceRidgeRuntime, policy: RuntimeV5QCPolicy, workpoint: str) -> None:
        if workpoint not in ("HC95", "HC90"):
            raise C5FederatedSourceRidgeQCRuntimeError("Runtime v5 QC workpoint is invalid")
        self.base = base
        self.policy = policy
        self.workpoint = workpoint

    @classmethod
    def from_runtime_contract(cls, path: Path, *, device: str = "cpu") -> "C5FederatedSourceRidgeQCRuntime":
        try:
            contract = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise C5FederatedSourceRidgeQCRuntimeError("Runtime v5 QC contract is invalid") from error
        required = {"schema_version", "status", "bundle_manifest", "workpoint", "outputs"}
        if set(contract) != required or contract.get("schema_version") != "iotj.c5_federated_source_ridge_qc_runtime_contract.v1" or contract.get("status") != "locked":
            raise C5FederatedSourceRidgeQCRuntimeError("Runtime v5 QC contract schema/status differs")
        bundle_record = contract.get("bundle_manifest")
        if not isinstance(bundle_record, Mapping) or set(bundle_record) != {"path", "bytes", "sha256"}:
            raise C5FederatedSourceRidgeQCRuntimeError("Runtime v5 QC bundle descriptor differs")
        bundle_path = Path(str(bundle_record["path"]))
        if descriptor(bundle_path) != bundle_record:
            raise C5FederatedSourceRidgeQCRuntimeError("Runtime v5 QC bundle identity differs")
        try:
            bundle = load_runtime_v5_qc_bundle(bundle_path)
            base = C5FederatedSourceRidgeRuntime.from_runtime_contract(bundle.base_runtime_contract, device=device)
            policy = RuntimeV5QCPolicy.from_path(bundle.qc_policy)
        except (RuntimeV5QCBundleError, C5FederatedSourceRidgeRuntimeError, ValueError) as error:
            raise C5FederatedSourceRidgeQCRuntimeError(str(error)) from error
        expected_outputs = [
            "sample_index", "pred_class", "source_h1_ppm", "prediction_ppm",
            "raw_risk_components", "normalized_risk_components", "deployment_risk",
            "qc_workpoint", "qc_decision", "auto_output_ppm",
        ]
        if contract.get("outputs") != expected_outputs:
            raise C5FederatedSourceRidgeQCRuntimeError("Runtime v5 QC output schema differs")
        return cls(base, policy, str(contract["workpoint"]))

    def infer(self, windows: np.ndarray, metadata: Sequence[Mapping[str, Any]], phases: np.ndarray) -> list[dict[str, Any]]:
        values = np.asarray(windows, dtype=np.float32)
        phase_values = np.asarray(phases)
        if values.ndim != 3 or values.shape[1:] != (100, 8) or not np.isfinite(values).all():
            raise C5FederatedSourceRidgeQCRuntimeError("runtime windows must be finite (N,100,8)")
        if len(metadata) != len(values) or phase_values.shape != (len(values),):
            raise C5FederatedSourceRidgeQCRuntimeError("runtime inputs are not aligned")
        if not np.issubdtype(phase_values.dtype, np.integer) or not np.isin(phase_values, (0, 1, 2)).all():
            raise C5FederatedSourceRidgeQCRuntimeError("runtime phases are invalid")
        with torch.no_grad():
            logits_tensor, representation_tensor, _regression = self.base.model(torch.from_numpy(values).to(self.base.device))
            probabilities_tensor = torch.softmax(logits_tensor, dim=1)
        probabilities = probabilities_tensor.detach().cpu().numpy().astype(np.float64)
        representations = representation_tensor.detach().cpu().numpy().astype(np.float64)
        if probabilities.shape != (len(values), 4) or representations.shape != (len(values), 64) or not np.isfinite(probabilities).all() or not np.isfinite(representations).all():
            raise C5FederatedSourceRidgeQCRuntimeError("classifier QC outputs differ or contain NaN/Inf")
        routes = np.argmax(probabilities, axis=1).astype(np.int64)
        rows: list[dict[str, Any]] = []
        for index, route in enumerate(routes.tolist()):
            meta = dict(metadata[index])
            meta["phase"] = int(phase_values[index])
            rich = self.base.feature_extractor(values[index], meta)
            if len(rich) != 104 or not all(math.isfinite(float(value)) for value in rich.values()):
                raise C5FederatedSourceRidgeQCRuntimeError("rich feature extractor output differs")
            h1 = self.base.source_h1[route].predict(rich)
            target_features = dict(rich)
            target_features["srcpred_H1_source_ridge_ppm"] = h1
            prediction = self.base.target_ridge[route].predict(target_features)
            score = self.policy.score(
                probabilities=probabilities[index], representation=representations[index],
                pred_class=route, source_h1_ppm=h1, prediction_ppm=prediction,
            )
            decision, auto = self.policy.decision(score["deployment_risk"], prediction, self.policy.payload["workpoints"][self.workpoint])
            rows.append({
                "sample_index": index,
                "pred_class": route,
                "source_h1_ppm": h1,
                "prediction_ppm": prediction,
                "raw_risk_components": {key.removeprefix("raw_"): score[key] for key in score if key.startswith("raw_")},
                "normalized_risk_components": {key.removeprefix("percentile_"): score[key] for key in score if key.startswith("percentile_")},
                "deployment_risk": score["deployment_risk"],
                "qc_workpoint": self.workpoint,
                "qc_decision": decision,
                "auto_output_ppm": auto,
            })
        return rows
