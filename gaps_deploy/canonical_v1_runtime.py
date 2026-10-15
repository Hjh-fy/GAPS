"""Exact canonical-v1 A4 + R84_FED_H1 + frozen equal-mean QC runtime."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from model import FedGasBaseModel
from run_regression_head_ablation import CLASS_RANGES, rich_feature_dict
from .canonical_serialized import FixedH8Policy, SerializedRidge


RISK_COMPONENTS = (
    "classification_uncertainty_risk",
    "regression_disagreement_risk",
    "source_prior_disagreement_risk",
)


def preprocess_canonical_window(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window)
    if values.shape != (50, 8):
        raise ValueError(f"canonical runtime requires a 50x8 window, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("canonical runtime window must be finite")
    return np.ascontiguousarray(values, dtype=np.float32)


def load_serialized_models_payload(payload: Mapping[str, Any]) -> dict[int, SerializedRidge]:
    if "models" in payload:
        payload = payload["models"]
    return {int(key): SerializedRidge.from_json(value) for key, value in payload.items()}


def _serialized_models(path: Path) -> dict[int, SerializedRidge]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_serialized_models_payload(payload)


class CanonicalV1Runtime:
    status = "FINAL_DEPLOYED_RUNTIME"

    def __init__(self, package_root: str | Path, target: str, device: str = "cpu") -> None:
        self.root = Path(package_root)
        self.target = str(target).upper()
        manifest = json.loads((self.root / "package_manifest.json").read_text(encoding="utf-8"))
        if self.target not in manifest["targets"]:
            raise ValueError(f"target is not packaged: {self.target}")
        assets = manifest["assets"]
        target_assets = manifest["targets"][self.target]
        self.device = torch.device(device)
        self.model = FedGasBaseModel(
            num_classes=4,
            num_sensors=8,
            feat_dim=64,
            encoder_type="tcn",
            use_cls_proj=True,
            tcn_norm="instance",
        ).to(self.device)
        checkpoint = torch.load(
            self.root / target_assets["classifier"]["path"],
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.h1 = _serialized_models(self.root / assets["federated_h1"]["path"])
        policy = json.loads((self.root / assets["h23_policy"]["path"]).read_text(encoding="utf-8"))
        self.h23 = FixedH8Policy.from_json(policy["source_aug_target_ridge_policy"])
        self.r83 = _serialized_models(self.root / target_assets["r83_models"]["path"])
        self.r84 = _serialized_models(self.root / target_assets["r84_models"]["path"])
        with (self.root / target_assets["qc_thresholds"]["path"]).open(
            encoding="utf-8", newline=""
        ) as handle:
            threshold_rows = list(csv.DictReader(handle))
        self.thresholds = {
            float(row["target_coverage"]): float(row["threshold"])
            for row in threshold_rows
        }
        first = threshold_rows[0]
        self.scales = {
            key: float(first[f"p95_scale_{key}"]) for key in RISK_COMPONENTS
        }

    def _classifier(self, values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            logits, _features, _regression = self.model(
                torch.from_numpy(values.reshape(1, 50, 8)).to(self.device)
            )
            return torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float64)

    def _regression(
        self,
        values: np.ndarray,
        probabilities: np.ndarray,
        metadata: Mapping[str, Any],
        phase: int,
    ) -> dict[str, Any]:
        route = int(np.argmax(probabilities))
        full = rich_feature_dict(values, int(phase), dict(metadata))
        h1 = self.h1[route].predict(full)
        h2 = self.h23.source_mlp[route].predict(full)
        shared = dict(full)
        shared["route_class"] = route
        h3 = self.h23.shared_mlp.predict(shared)
        sensor = {name: full[name] for name in self.r83[route].feature_names}
        if len(sensor) != 83:
            raise RuntimeError(
                f"canonical R83 contract requires 83 features, got {len(sensor)}"
            )
        pred83 = self.r83[route].predict(sensor)
        augmented = dict(sensor)
        augmented["srcpred_H1_federated_source_ridge_ppm"] = h1
        pred84 = self.r84[route].predict(augmented)
        return {
            "route": route,
            "pred_83d_ppm": float(pred83),
            "pred_84d_h1_ppm": float(pred84),
            "h1": float(h1),
            "h2": float(h2),
            "h3": float(h3),
        }

    def _qc(self, probabilities: np.ndarray, regression: Mapping[str, Any]) -> dict[str, Any]:
        route = int(regression["route"])
        ordered = np.sort(probabilities)[::-1]
        confidence = float(ordered[0])
        margin = float(ordered[0] - ordered[1])
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
            / math.log(4.0)
        )
        components = {
            "classification_uncertainty_risk": max(1.0 - confidence, 1.0 - margin, entropy),
            "regression_disagreement_risk": abs(
                float(regression["pred_84d_h1_ppm"]) - float(regression["pred_83d_ppm"])
            ) / float(CLASS_RANGES[route]),
            "source_prior_disagreement_risk": (
                max(float(regression["h1"]), float(regression["h2"]), float(regression["h3"]))
                - min(float(regression["h1"]), float(regression["h2"]), float(regression["h3"]))
            ) / float(CLASS_RANGES[route]),
        }
        risk = float(np.mean([
            np.clip(components[key] / self.scales[key], 0.0, 1.0)
            for key in RISK_COMPONENTS
        ]))
        def decision(accept: float, review: float) -> str:
            if risk <= self.thresholds[accept]:
                return "accepted"
            if risk <= self.thresholds[review]:
                return "review"
            return "reject"
        return {
            **components,
            "qc_risk_score_final": risk,
            "HC90_decision": decision(0.90, 0.95),
            "HC95_decision": decision(0.95, 0.975),
        }

    def infer_one_timed(
        self, window: np.ndarray, metadata: Mapping[str, Any], phase: int
    ) -> tuple[dict[str, Any], dict[str, float]]:
        total_start = time.perf_counter_ns()
        start = total_start
        values = preprocess_canonical_window(window)
        after_preprocess = time.perf_counter_ns()
        probabilities = self._classifier(values)
        after_classifier = time.perf_counter_ns()
        regression = self._regression(values, probabilities, metadata, phase)
        after_regression = time.perf_counter_ns()
        qc = self._qc(probabilities, regression)
        after_qc = time.perf_counter_ns()
        result = {
            "runtime_status": self.status,
            "target_profile": self.target,
            "pred_class": int(regression["route"]),
            **{f"prob_class_{index}": float(value) for index, value in enumerate(probabilities)},
            "pred_83d_ppm": regression["pred_83d_ppm"],
            "pred_84d_h1_ppm": regression["pred_84d_h1_ppm"],
            **qc,
        }
        timings = {
            "preprocessing_ms": (after_preprocess - start) / 1e6,
            "classifier_ms": (after_classifier - after_preprocess) / 1e6,
            "r84_ms": (after_regression - after_classifier) / 1e6,
            "qc_ms": (after_qc - after_regression) / 1e6,
            "total_pipeline_ms": (after_qc - total_start) / 1e6,
        }
        return result, timings

    def infer_one(
        self, window: np.ndarray, metadata: Mapping[str, Any], phase: int
    ) -> dict[str, Any]:
        return self.infer_one_timed(window, metadata, phase)[0]
