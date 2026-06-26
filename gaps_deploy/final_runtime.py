"""Standalone-facing runtime for the frozen C12->C345 deployment bundle.

The base predictor remains responsible for classification, R3aK16 routing,
auto_v2 calibration, risk scoring, and QC. This wrapper fixes the public output
schema and applies the optional guarded CO residual layer without overwriting
``final_ppm``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .inference import DeployPredictor, DeployResult
from .rich_residual import RichResidualPolicy


OUTPUT_FIELDS = [
    "gas_class",
    "gas_name",
    "class_prob",
    "base_r3ak16_raw_ppm",
    "routed_pred_ppm",
    "final_ppm",
    "co_corrected_ppm",
    "auto_output_ppm",
    "qc_decision",
    "risk_score",
]

# The correction gate must never grow beyond deployment-visible fields.
CO_GATE_FIELDS = {
    "client_id",
    "pred_class",
    "routed_pred_ppm",
    "qc_decision",
}


def normalize_client_id(client_id: str | int) -> str:
    text = str(client_id).strip().upper()
    if text.startswith("CLIENT_"):
        text = text.split("_", 1)[1]
    if text.startswith("C"):
        return f"C{int(text[1:])}"
    return f"C{int(text)}"


class FinalDeployRuntime:
    """Load one client package and expose the frozen nine-field output."""

    def __init__(
        self,
        bundle_dir: str | Path,
        client_id: str | int,
        device: str = "cpu",
    ) -> None:
        load_started = time.perf_counter()
        self.bundle_dir = Path(bundle_dir).resolve()
        self.client_id = normalize_client_id(client_id)
        config_path = self.bundle_dir / "runtime_config.json"
        self.runtime_config = json.loads(config_path.read_text(encoding="utf-8"))

        package_rel = self.runtime_config["client_packages"].get(self.client_id)
        if not package_rel:
            raise ValueError(f"No package configured for {self.client_id}")
        self.predictor = DeployPredictor.from_package(
            str(self.bundle_dir / package_rel), device=device
        )

        norm_path = self.bundle_dir / self.runtime_config["norm_stats"]
        norm = np.load(norm_path)
        self.norm_mean = np.asarray(norm["mean"], dtype=np.float32)
        self.norm_std = np.asarray(norm["std"], dtype=np.float32)
        if not np.all(np.isfinite(self.norm_mean)) or not np.all(np.isfinite(self.norm_std)):
            raise ValueError("norm_stats contains non-finite values")

        params_path = self.bundle_dir / self.runtime_config["co_correction_params"]
        self.co_params = json.loads(params_path.read_text(encoding="utf-8"))
        artifact_rel = self.runtime_config.get("rich_residual_artifact", "")
        self.rich_residual = RichResidualPolicy.from_json(self.bundle_dir / artifact_rel) if artifact_rel else RichResidualPolicy()
        self.last_timing_ms: Dict[str, float] = {}
        self.model_load_ms = (time.perf_counter() - load_started) * 1000.0

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        if array.ndim != 3 or array.shape[1:] != (100, 8):
            raise ValueError(f"Expected (N,100,8) or (100,8), got {array.shape}")
        normalization = self.runtime_config.get("normalization", {})
        if bool(normalization.get("enabled", False)):
            array = (array - self.norm_mean) / np.maximum(self.norm_std, 1e-8)
        return array

    @staticmethod
    def _ridge_feature_values(result: DeployResult) -> Dict[str, float]:
        routed = float(result.routed_pred_ppm)
        risks = result.risk_scores or {}
        return {
            "routed_pred_ppm": routed,
            "base_r3ak16_raw_ppm": float(result.base_r3ak16_raw_ppm),
            "risk_response_signature_norm": float(risks.get("response_signature_norm", 0.0)),
            "risk_response_conc_gap_norm": float(risks.get("response_conc_gap_norm", 0.0)),
            "risk_composite_response_risk": float(risks.get("composite_response_risk", 0.0)),
            "confidence_margin": float(result.confidence_margin),
            "phase": float(result.phase),
            "routed_hinge_125": max(0.0, routed - 125.0),
            "routed_hinge_175": max(0.0, routed - 175.0),
        }

    def _co_gate(self, result: DeployResult) -> bool:
        scope = self.co_params.get("scope", {})
        return bool(
            self.co_params.get("enabled", False)
            and self.client_id in set(scope.get("clients", []))
            and int(result.pred_class) == int(scope.get("class_id", 1))
            and float(result.routed_pred_ppm) >= float(scope.get("routed_pred_ppm_min", 175.0))
            and str(result.qc_status) == "review"
        )

    def _co_corrected_ppm(self, result: DeployResult) -> float:
        if not self._co_gate(result):
            return float(result.final_ppm)
        model = self.co_params.get("models", {}).get(self.client_id)
        if not model:
            return float(result.final_ppm)
        values = self._ridge_feature_values(result)
        names = list(model["feature_names"])
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coef = np.asarray(model["coef"], dtype=np.float64)
        x = np.asarray([values[name] for name in names], dtype=np.float64)
        x = np.where(np.isfinite(x), x, mean)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        design = np.concatenate([[1.0], (x - mean) / scale])
        delta = float(design @ coef)
        return float(result.routed_pred_ppm + delta)

    def _artifact_corrected_ppm(
        self,
        window: np.ndarray,
        result: DeployResult,
        meta: Dict[str, Any] | None = None,
    ) -> float:
        if self.rich_residual.enabled:
            return self.rich_residual.apply(window, result, self.client_id, meta=meta)
        return self._co_corrected_ppm(result)

    @staticmethod
    def _public_row(result: DeployResult, co_corrected_ppm: float) -> Dict[str, Any]:
        qc_decision = str(result.qc_status)
        return {
            "gas_class": int(result.pred_class),
            "gas_name": str(result.pred_gas),
            "class_prob": float(result.confidence),
            "base_r3ak16_raw_ppm": float(result.base_r3ak16_raw_ppm),
            "routed_pred_ppm": float(result.routed_pred_ppm),
            "final_ppm": float(result.final_ppm),
            "co_corrected_ppm": float(co_corrected_ppm),
            "auto_output_ppm": float(co_corrected_ppm) if qc_decision == "accept" else "",
            "qc_decision": qc_decision,
            "risk_score": float(result.risk_score),
        }

    def predict_batch(
        self,
        features: np.ndarray,
        phase: int | Sequence[int] | np.ndarray = -1,
        metadata: Sequence[Dict[str, Any]] | None = None,
    ) -> list[Dict[str, Any]]:
        total_started = time.perf_counter()
        raw = np.asarray(features, dtype=np.float32)
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]
        prepared = self._prepare_features(features)
        base_results = self.predictor.predict_batch(
            prepared, client_id=self.client_id, phase=phase
        )
        correction_started = time.perf_counter()
        metadata = metadata or []
        rows = [
            self._public_row(
                result,
                self._artifact_corrected_ppm(
                    raw[idx],
                    result,
                    metadata[idx] if idx < len(metadata) else None,
                ),
            )
            for idx, result in enumerate(base_results)
        ]
        correction_ms = (time.perf_counter() - correction_started) * 1000.0
        self.last_timing_ms = dict(self.predictor.last_timing_ms)
        self.last_timing_ms["co_correction_ms"] = float(correction_ms)
        self.last_timing_ms["runtime_total_ms"] = float(
            (time.perf_counter() - total_started) * 1000.0
        )
        return rows

    def predict_single(
        self,
        features: np.ndarray,
        phase: int = -1,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.predict_batch(features, phase=phase, metadata=[metadata] if metadata else None)[0]


def _phase_values(path: str, count: int, default_phase: int) -> np.ndarray | int:
    if not path:
        return int(default_phase)
    values = np.load(path, allow_pickle=True).astype(np.int64).reshape(-1)
    if len(values) < count:
        raise ValueError(f"phase file has {len(values)} rows, expected at least {count}")
    return values[:count]


def _metadata_values(path: str, count: int) -> list[Dict[str, Any]]:
    if not path:
        return []
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("metadata file must contain a JSON list")
    if len(values) < count:
        raise ValueError(f"metadata file has {len(values)} rows, expected at least {count}")
    return [dict(item) if isinstance(item, dict) else {} for item in values[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final fixed-DA deployment bundle")
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--input", required=True, help=".npy file with (100,8) or (N,100,8)")
    parser.add_argument("--phase-file", default="")
    parser.add_argument("--metadata-file", default="", help="Optional experiment_info JSON list for response_phase-aware policies")
    parser.add_argument("--phase", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    features = np.load(args.input, allow_pickle=True).astype(np.float32)
    if args.limit > 0 and features.ndim == 3:
        features = features[: args.limit]
    count = 1 if features.ndim == 2 else len(features)
    phases = _phase_values(args.phase_file, count, args.phase)
    metadata = _metadata_values(args.metadata_file, count)
    runtime = FinalDeployRuntime(args.bundle, args.client_id, device=args.device)
    rows = runtime.predict_batch(features, phase=phases, metadata=metadata)
    payload: Any = rows[0] if features.ndim == 2 else rows
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
