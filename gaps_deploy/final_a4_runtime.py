"""Exact frozen A4 + R84_FED_H1 + equal-mean QC deployment runtime."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from model import FedGasBaseModel
from run_regression_head_ablation import CLASS_RANGES, rich_feature_dict
from .c5_h8_runtime import FixedH8Policy, SerializedRidge


class FinalA4Runtime:
    status = "FINAL_DEPLOYED_RUNTIME"

    def __init__(self, package_root: str | Path, device: str = "cpu") -> None:
        root = Path(package_root)
        manifest = json.loads((root / "FINAL_DEPLOYMENT_MANIFEST.json").read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.model = FedGasBaseModel(
            num_classes=4, num_sensors=8, feat_dim=64, encoder_type="tcn",
            use_cls_proj=True, tcn_norm="instance",
        ).to(self.device)
        checkpoint = torch.load(root / manifest["assets"]["classifier"]["path"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()

        h1 = json.loads((root / manifest["assets"]["federated_h1"]["path"]).read_text(encoding="utf-8"))
        self.h1 = {int(key): SerializedRidge.from_json(value) for key, value in h1["models"].items()}
        regression = json.loads((root / manifest["assets"]["regression_models"]["path"]).read_text(encoding="utf-8"))
        self.r83 = {int(key): SerializedRidge.from_json(value) for key, value in regression["R83_TARGET_ONLY"].items()}
        self.r84 = {int(key): SerializedRidge.from_json(value) for key, value in regression["R84_FED_H1"].items()}
        old_policy = json.loads((root / manifest["assets"]["r4_policy"]["path"]).read_text(encoding="utf-8"))
        self.source_policy = FixedH8Policy.from_json(old_policy["source_aug_target_ridge_policy"])

        with (root / manifest["assets"]["qc_threshold_lock"]["path"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.thresholds = {f"HC{int(round(float(row['target_coverage']) * 100))}": float(row["threshold"]) for row in rows}
        first = rows[0]
        self.scales = np.asarray([
            float(first["p95_scale_classification_uncertainty_risk"]),
            float(first["p95_scale_regression_disagreement_risk"]),
            float(first["p95_scale_source_prior_disagreement_risk"]),
        ], dtype=np.float64)

    def infer_one(self, window: np.ndarray, metadata: Mapping[str, Any], phase: int) -> dict[str, Any]:
        values = np.asarray(window, dtype=np.float32).reshape(1, 100, 8)
        with torch.inference_mode():
            logits, _features, _regression = self.model(torch.from_numpy(values).to(self.device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float64)
        route = int(np.argmax(probabilities))
        full = rich_feature_dict(values[0], int(phase), dict(metadata))
        h1 = self.h1[route].predict(full)
        h2 = self.source_policy.source_mlp[route].predict(full)
        source_values = dict(full); source_values["route_class"] = route
        h3 = self.source_policy.shared_mlp.predict(source_values)
        sensor = {name: full[name] for name in self.r83[route].feature_names}
        pred83 = self.r83[route].predict(sensor)
        augmented = dict(sensor); augmented["srcpred_H1_federated_source_ridge_ppm"] = h1
        pred84 = self.r84[route].predict(augmented)
        ordered = np.sort(probabilities)
        confidence = float(ordered[-1]); margin = float(ordered[-1] - ordered[-2])
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum() / math.log(4.0))
        classification_risk = max(1.0 - confidence, 1.0 - margin, entropy)
        regression_risk = abs(pred84 - pred83) / CLASS_RANGES[route]
        source_risk = (max(h1, h2, h3) - min(h1, h2, h3)) / CLASS_RANGES[route]
        final_risk = float(np.mean(np.clip(np.asarray([classification_risk, regression_risk, source_risk]) / self.scales, 0.0, 1.0)))
        return {
            "runtime_status": self.status,
            "pred_class": route,
            **{f"prob_class_{index}": float(value) for index, value in enumerate(probabilities)},
            "pred_83d_ppm": pred83,
            "pred_84d_h1_ppm": pred84,
            "classification_uncertainty_risk": classification_risk,
            "regression_disagreement_risk": regression_risk,
            "source_prior_disagreement_risk": source_risk,
            "qc_risk_score_final": final_risk,
            "accepted_hc90": int(final_risk <= self.thresholds["HC90"]),
            "accepted_hc95": int(final_risk <= self.thresholds["HC95"]),
        }
