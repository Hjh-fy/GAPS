"""Optional edge-AI runtime for the STM32 gas-sensor upper-computer.

The UI stays usable without PyTorch or an AI package.  When a deployment package
is loaded, this module performs the deployment-only steps:

1. collect selected sensor channels from real STM32 frames;
2. build a baseline and transform samples with the package-defined feature mode;
3. create a fixed-length sliding window;
4. apply training-set normalization;
5. run either a TorchScript package or the frozen GAPS Runtime-v5 core;
6. expose classification, concentration and QC decisions to the UI.

Training, Flower aggregation, server-side domain adaptation and package export
remain in the main GAPS repository.  This module deliberately does not import
GAPS training code, which keeps Raspberry Pi deployment small and reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_GAS_NAMES = ["Ethanol", "CO", "Ethylene", "Methane"]
SUPPORTED_MODEL_BACKENDS = {"torchscript", "gaps_runtime_v5"}
RUNTIME_V5_METADATA_FIELDS = {
    "window_start_s",
    "window_end_s",
    "window_center_s",
    "t_onset",
    "t_min",
    "interpolated_ratio",
    "max_gap_inside_window",
    "response_phase",
    "phase_label",
}


class EdgeAIPackageError(RuntimeError):
    """Raised when a deployment package is incomplete or inconsistent."""


@dataclass
class EdgeAIResult:
    timestamp_iso: str
    stream_frame_index: int
    predicted_class: int
    predicted_gas: str
    confidence: float
    class_probabilities: List[float]
    consensus_predicted_class: int
    consensus_predicted_gas: str
    consensus_confidence: float
    consensus_probabilities: List[float]
    consensus_window_count: int
    task_type: str
    has_concentration: bool
    ppm_base_prediction: Optional[float]
    ppm_full_prediction: Optional[float]
    ppm_auto_output: Optional[float]
    decision: str
    selected_calibration: str
    selected_policy: str
    risk_score: Optional[float]
    risk_score_name: str
    inference_latency_ms: float
    observed_hz: float
    window_size: int
    package_name: str
    package_fingerprint: str
    dataset_profile: str
    device_profile: str
    model_backend: str
    normalization_applied: bool
    experiment_phase: str
    inference_id: int
    window_start_timestamp_iso: str
    window_end_timestamp_iso: str
    window_start_stream_frame_index: int
    window_end_stream_frame_index: int
    window_connection_id: int
    window_recording_complete: bool
    recording_session_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeAIPackage:
    """Read and validate a portable deployment package."""

    def __init__(self, package_dir: Path) -> None:
        self.package_dir = Path(package_dir).expanduser().resolve()
        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            raise EdgeAIPackageError(f"Missing manifest.json in {self.package_dir}")
        try:
            manifest_bytes = manifest_path.read_bytes()
            self.manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise EdgeAIPackageError(f"Invalid manifest.json: {exc}") from exc

        self.schema_version = int(self.manifest.get("schema_version", 1))
        if self.schema_version not in {1, 2, 3, 4}:
            raise EdgeAIPackageError(f"Unsupported package schema_version={self.schema_version}")

        self.package_name = str(self.manifest.get("package_name") or self.package_dir.name)
        self.dataset_profile = str(self.manifest.get("dataset_profile") or "unspecified")
        self.device_profile = str(self.manifest.get("device_profile") or "unspecified")
        self.model_backend = str(self.manifest.get("model_backend") or "torchscript").lower()
        if self.model_backend not in SUPPORTED_MODEL_BACKENDS:
            raise EdgeAIPackageError(
                f"Unsupported model_backend={self.model_backend}; "
                f"supported backends are {sorted(SUPPORTED_MODEL_BACKENDS)}"
            )
        if self.model_backend == "gaps_runtime_v5" and self.schema_version < 3:
            raise EdgeAIPackageError("gaps_runtime_v5 requires package schema_version=3")

        input_cfg = dict(self.manifest.get("input") or {})
        self.sensor_fields = list(input_cfg.get("sensor_fields") or [])
        if not self.sensor_fields:
            raise EdgeAIPackageError("manifest.input.sensor_fields must contain model input field names")
        if len(set(str(x) for x in self.sensor_fields)) != len(self.sensor_fields):
            raise EdgeAIPackageError("manifest.input.sensor_fields must not contain duplicates")
        self.sensor_fields = [str(x) for x in self.sensor_fields]

        self.expected_hz = float(
            input_cfg.get("target_sample_hz", input_cfg.get("expected_hz", 10.0))
        )
        if not math.isfinite(self.expected_hz) or self.expected_hz <= 0:
            raise EdgeAIPackageError("input.target_sample_hz/expected_hz must be positive")
        self.raw_sample_hz = float(input_cfg.get("raw_sample_hz", self.expected_hz))
        if not math.isfinite(self.raw_sample_hz) or self.raw_sample_hz <= 0:
            raise EdgeAIPackageError("input.raw_sample_hz must be positive")

        self.window_duration_s = self._optional_positive_float(
            input_cfg.get("window_duration_s"), "input.window_duration_s"
        )
        self.stride_duration_s = self._optional_positive_float(
            input_cfg.get("stride_duration_s"), "input.stride_duration_s"
        )
        self.baseline_duration_s = self._optional_nonnegative_float(
            input_cfg.get("baseline_duration_s"), "input.baseline_duration_s"
        )
        self.unstable_duration_s = self._optional_nonnegative_float(
            input_cfg.get("unstable_duration_s", 0.0), "input.unstable_duration_s"
        ) or 0.0

        self.window_size = (
            self._seconds_to_samples(self.window_duration_s, "window_duration_s")
            if self.window_duration_s is not None
            else int(input_cfg.get("window_size", 100))
        )
        self.stride = (
            self._seconds_to_samples(self.stride_duration_s, "stride_duration_s")
            if self.stride_duration_s is not None
            else int(input_cfg.get("stride", 50))
        )
        self.feature_mode = str(input_cfg.get("feature_mode", "relative_adc")).lower()
        if self.feature_mode == "precomputed":
            self.baseline_samples = 0
            self.baseline_duration_s = 0.0
        elif self.baseline_duration_s is not None:
            self.baseline_samples = max(
                1, self._seconds_to_samples(self.baseline_duration_s, "baseline_duration_s")
            )
        else:
            self.baseline_samples = int(input_cfg.get("baseline_samples", 300))
            self.baseline_duration_s = (
                float(self.baseline_samples) / self.expected_hz
                if self.baseline_samples > 0
                else 0.0
            )
        self.unstable_samples = int(round(self.unstable_duration_s * self.expected_hz))
        self.min_rate_ratio = float(input_cfg.get("min_rate_ratio", 0.70))
        self.max_rate_ratio = float(input_cfg.get("max_rate_ratio", 1.30))
        self.allow_rate_mismatch = bool(input_cfg.get("allow_rate_mismatch", False))
        self.max_gap_s = float(
            input_cfg.get("max_gap_s", max(3.0 / self.expected_hz, 1.0))
        )
        self.reject_implausible_frames = bool(input_cfg.get("reject_implausible_frames", True))
        self.adc_vref = float(input_cfg.get("adc_vref", 3.3))
        self.adc_max = float(input_cfg.get("adc_max", 4095.0))
        self.rload_ohm = [float(x) for x in input_cfg.get("rload_ohm", [])]

        if self.window_size <= 1 or self.stride <= 0:
            raise EdgeAIPackageError("window_size must be >1 and stride must be >0")
        if not 0 < self.min_rate_ratio <= self.max_rate_ratio:
            raise EdgeAIPackageError("input rate ratios must satisfy 0 < min_rate_ratio <= max_rate_ratio")
        if not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise EdgeAIPackageError("input.max_gap_s must be positive")
        if self.baseline_samples < 1 and self.feature_mode not in {"raw_adc", "precomputed"}:
            raise EdgeAIPackageError("baseline_samples must be positive for baseline-relative features")
        if self.feature_mode == "relative_conductance" and len(self.rload_ohm) != len(self.sensor_fields):
            raise EdgeAIPackageError(
                "relative_conductance requires one input.rload_ohm value per sensor field"
            )
        if self.feature_mode == "relative_conductance" and any(
            not math.isfinite(value) or value <= 0 for value in self.rload_ohm
        ):
            raise EdgeAIPackageError(
                "relative_conductance input.rload_ohm values must all be positive and finite"
            )
        if self.feature_mode not in {
            "raw_adc",
            "relative_adc",
            "relative_conductance",
            "precomputed",
        }:
            raise EdgeAIPackageError(f"Unsupported feature_mode={self.feature_mode}")
        if self.model_backend == "gaps_runtime_v5":
            if len(self.sensor_fields) != 8:
                raise EdgeAIPackageError("gaps_runtime_v5 requires exactly 8 sensor fields")
            if self.window_size != 100:
                raise EdgeAIPackageError("gaps_runtime_v5 requires a 100-sample window")
            if self.feature_mode != "precomputed":
                raise EdgeAIPackageError(
                    "the current gaps_runtime_v5 UI adapter accepts only frozen "
                    "precomputed relative-conductance inputs"
                )

        phase_cfg = dict(self.manifest.get("phase_control") or {})
        self.phase_mode = str(
            phase_cfg.get("mode", "automatic" if self.schema_version == 1 else "event_driven")
        ).lower()
        if self.phase_mode not in {"automatic", "event_driven"}:
            raise EdgeAIPackageError("phase_control.mode must be automatic or event_driven")
        self.inference_phases = [
            str(x).lower()
            for x in phase_cfg.get("inference_phases", ["exposure", "recovery"])
        ]
        if not self.inference_phases:
            raise EdgeAIPackageError("phase_control.inference_phases must not be empty")

        norm_cfg = dict(self.manifest.get("normalization") or {})
        self.normalization_enabled = bool(
            norm_cfg.get("enabled", True if self.schema_version == 1 else False)
        )
        self.norm_path: Optional[Path] = None
        if self.normalization_enabled:
            norm_file = str(norm_cfg.get("file", "norm_stats.npz"))
            self.norm_path = self._resolve_member(norm_file, "normalization file")
            if not self.norm_path.exists():
                raise EdgeAIPackageError(f"Missing normalization file: {self.norm_path}")
            try:
                norm = np.load(self.norm_path)
                self.mean = np.asarray(
                    norm[str(norm_cfg.get("mean_key", "mean"))], dtype=np.float32
                )
                self.std = np.asarray(
                    norm[str(norm_cfg.get("std_key", "std"))], dtype=np.float32
                )
            except Exception as exc:
                raise EdgeAIPackageError(f"Failed to load normalization statistics: {exc}") from exc
            if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
                raise EdgeAIPackageError("Normalization mean/std must contain only finite values")
            while self.mean.ndim > 2 and self.mean.shape[0] == 1:
                self.mean = np.squeeze(self.mean, axis=0)
            while self.std.ndim > 2 and self.std.shape[0] == 1:
                self.std = np.squeeze(self.std, axis=0)
            self.std = np.where(np.abs(self.std) > 1e-8, self.std, 1.0).astype(np.float32)
        else:
            self.mean = np.zeros((1, len(self.sensor_fields)), dtype=np.float32)
            self.std = np.ones((1, len(self.sensor_fields)), dtype=np.float32)
        if self.model_backend == "gaps_runtime_v5" and self.normalization_enabled:
            raise EdgeAIPackageError(
                "gaps_runtime_v5 forbids additional UI-side Z-score normalization"
            )

        expected_channels = len(self.sensor_fields)
        try:
            np.broadcast_to(self.mean, (self.window_size, expected_channels))
            np.broadcast_to(self.std, (self.window_size, expected_channels))
        except ValueError as exc:
            raise EdgeAIPackageError(
                f"mean/std cannot broadcast to ({self.window_size}, {expected_channels}); "
                f"got mean={self.mean.shape}, std={self.std.shape}"
            ) from exc

        output_cfg = dict(self.manifest.get("output") or {})
        self.gas_names = list(output_cfg.get("gas_names") or DEFAULT_GAS_NAMES)
        if not self.gas_names or any(not str(name).strip() for name in self.gas_names):
            raise EdgeAIPackageError("output.gas_names must contain non-empty names")
        self.gas_names = [str(name) for name in self.gas_names]
        self.task_type = str(
            output_cfg.get("task_type", "classification_regression")
        ).strip().lower()
        if self.task_type not in {"classification", "classification_regression"}:
            raise EdgeAIPackageError(
                "output.task_type must be classification or classification_regression"
            )
        self.has_concentration = bool(
            output_cfg.get(
                "has_concentration",
                self.task_type == "classification_regression",
            )
        )
        if self.task_type == "classification" and self.has_concentration:
            raise EdgeAIPackageError(
                "classification-only packages must set output.has_concentration=false"
            )
        self.qc_cfg = dict(self.manifest.get("qc") or {})
        min_confidence = float(self.qc_cfg.get("min_confidence", 0.0))
        accept_max_risk = float(
            self.qc_cfg.get(
                "accept_max_risk", self.qc_cfg.get("max_risk_score", float("inf"))
            )
        )
        reject_min_risk = float(self.qc_cfg.get("reject_min_risk", float("inf")))
        low_confidence_decision = str(
            self.qc_cfg.get("low_confidence_decision", "review")
        )
        if not 0.0 <= min_confidence <= 1.0:
            raise EdgeAIPackageError("qc.min_confidence must be between 0 and 1")
        if accept_max_risk > reject_min_risk:
            raise EdgeAIPackageError(
                "QC thresholds must satisfy accept_max_risk <= reject_min_risk"
            )
        if low_confidence_decision not in {"review", "reject"}:
            raise EdgeAIPackageError(
                "qc.low_confidence_decision must be review or reject"
            )
        self.calibration_cfg = dict(self.manifest.get("calibration") or {})
        self.calibration_path: Optional[Path] = None
        calibration_file = self.manifest.get("calibration_file")
        if calibration_file:
            self.calibration_path = self._resolve_member(
                str(calibration_file), "calibration file"
            )
            if not self.calibration_path.exists():
                raise EdgeAIPackageError(f"Missing calibration file: {self.calibration_path}")
            try:
                self.calibration_cfg = json.loads(
                    self.calibration_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise EdgeAIPackageError(f"Invalid calibration file: {exc}") from exc

        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.model_path: Optional[Path] = None
        self.model_sha256 = ""
        self.runtime_v5_binding_path: Optional[Path] = None
        self.runtime_v5_binding_sha256 = ""
        self.runtime_v5_release_id = ""
        self.runtime_v5_phase_id = -1
        self.runtime_v5_metadata: Dict[str, Any] = {}
        self.runtime_v5_code_root: Optional[Path] = None
        self.runtime_v5_code_manifest_path: Optional[Path] = None
        self.runtime_v5_code_manifest_sha256 = ""
        self.runtime_v5_code_source_commit = ""
        fingerprint_parts = [self.manifest_sha256]
        integrity_cfg = dict(self.manifest.get("integrity") or {})
        if self.model_backend == "torchscript":
            model_file = str(self.manifest.get("model_file", "model.ts"))
            self.model_path = self._resolve_member(model_file, "TorchScript model")
            if not self.model_path.exists():
                raise EdgeAIPackageError(f"Missing TorchScript model: {self.model_path}")
            self.model_sha256 = self._sha256(self.model_path)
            fingerprint_parts.append(self.model_sha256)
            expected_sha = str(integrity_cfg.get("model_sha256") or "").lower()
            if expected_sha and expected_sha != self.model_sha256:
                raise EdgeAIPackageError(
                    f"TorchScript SHA-256 mismatch: expected {expected_sha}, got {self.model_sha256}"
                )
        else:
            runtime_cfg = dict(self.manifest.get("runtime_v5") or {})
            binding_file = str(runtime_cfg.get("binding_file") or "")
            if not binding_file:
                raise EdgeAIPackageError("runtime_v5.binding_file is required")
            self.runtime_v5_binding_path = self._resolve_member(
                binding_file, "Runtime-v5 portable binding"
            )
            if not self.runtime_v5_binding_path.is_file():
                raise EdgeAIPackageError(
                    f"Missing Runtime-v5 portable binding: {self.runtime_v5_binding_path}"
                )
            self.runtime_v5_binding_sha256 = self._sha256(
                self.runtime_v5_binding_path
            )
            expected_binding_sha = str(
                integrity_cfg.get("runtime_v5_binding_sha256") or ""
            ).lower()
            if (
                expected_binding_sha
                and expected_binding_sha != self.runtime_v5_binding_sha256
            ):
                raise EdgeAIPackageError(
                    "Runtime-v5 binding SHA-256 mismatch: "
                    f"expected {expected_binding_sha}, "
                    f"got {self.runtime_v5_binding_sha256}"
                )
            try:
                binding_payload = json.loads(
                    self.runtime_v5_binding_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise EdgeAIPackageError(
                    f"Invalid Runtime-v5 portable binding: {exc}"
                ) from exc
            self.runtime_v5_release_id = str(
                runtime_cfg.get("release_id") or binding_payload.get("release_id") or ""
            )
            if not self.runtime_v5_release_id:
                raise EdgeAIPackageError("runtime_v5.release_id is required")
            if binding_payload.get("release_id") != self.runtime_v5_release_id:
                raise EdgeAIPackageError(
                    "Runtime-v5 release identity differs between manifest and binding"
                )
            try:
                self.runtime_v5_phase_id = int(runtime_cfg["fixed_phase_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EdgeAIPackageError(
                    "runtime_v5.fixed_phase_id must be an integer within 0..2"
                ) from exc
            if self.runtime_v5_phase_id not in (0, 1, 2):
                raise EdgeAIPackageError(
                    "runtime_v5.fixed_phase_id must be an integer within 0..2"
                )
            metadata = runtime_cfg.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise EdgeAIPackageError("runtime_v5.metadata must be an object")
            unexpected = set(metadata) - RUNTIME_V5_METADATA_FIELDS
            if unexpected:
                raise EdgeAIPackageError(
                    "runtime_v5.metadata contains unsupported fields: "
                    + ", ".join(sorted(unexpected))
                )
            self.runtime_v5_metadata = dict(metadata)
            expected_phase_label = {
                0: "early",
                1: "middle",
                2: "late",
            }[self.runtime_v5_phase_id]
            if (
                str(self.runtime_v5_metadata.get("phase_label") or "").lower()
                != expected_phase_label
            ):
                raise EdgeAIPackageError(
                    "runtime_v5.metadata.phase_label must match fixed_phase_id "
                    f"({self.runtime_v5_phase_id} -> {expected_phase_label})"
                )
            fingerprint_parts.append(self.runtime_v5_binding_sha256)
            code_root_value = str(runtime_cfg.get("code_root") or "")
            code_manifest_value = str(runtime_cfg.get("code_manifest_file") or "")
            if bool(code_root_value) != bool(code_manifest_value):
                raise EdgeAIPackageError(
                    "runtime_v5.code_root and code_manifest_file must be provided together"
                )
            if code_root_value:
                self.runtime_v5_code_root = self._resolve_member(
                    code_root_value, "Runtime-v5 code root"
                )
                self.runtime_v5_code_manifest_path = self._resolve_member(
                    code_manifest_value, "Runtime-v5 code manifest"
                )
                if not self.runtime_v5_code_root.is_dir():
                    raise EdgeAIPackageError(
                        f"Missing Runtime-v5 code root: {self.runtime_v5_code_root}"
                    )
                try:
                    self.runtime_v5_code_manifest_path.relative_to(
                        self.runtime_v5_code_root
                    )
                except ValueError as exc:
                    raise EdgeAIPackageError(
                        "Runtime-v5 code manifest must be inside code_root"
                    ) from exc
                (
                    self.runtime_v5_code_manifest_sha256,
                    self.runtime_v5_code_source_commit,
                ) = self._verify_runtime_v5_code_bundle(
                    self.runtime_v5_code_root,
                    self.runtime_v5_code_manifest_path,
                )
                expected_code_manifest_sha = str(
                    integrity_cfg.get("runtime_v5_code_manifest_sha256") or ""
                ).lower()
                if (
                    expected_code_manifest_sha
                    and expected_code_manifest_sha
                    != self.runtime_v5_code_manifest_sha256
                ):
                    raise EdgeAIPackageError(
                        "Runtime-v5 code manifest SHA-256 mismatch: "
                        f"expected {expected_code_manifest_sha}, "
                        f"got {self.runtime_v5_code_manifest_sha256}"
                    )
                declared_commit = str(
                    runtime_cfg.get("code_source_commit") or ""
                ).lower()
                if declared_commit != self.runtime_v5_code_source_commit:
                    raise EdgeAIPackageError(
                        "Runtime-v5 code source commit differs"
                    )
                fingerprint_parts.append(self.runtime_v5_code_manifest_sha256)

        for optional_path in (self.norm_path, self.calibration_path):
            if optional_path is not None:
                fingerprint_parts.append(self._sha256(optional_path))
        self.package_fingerprint = hashlib.sha256(
            "|".join(fingerprint_parts).encode("ascii")
        ).hexdigest()
        for label, path, key in (
            ("normalization", self.norm_path, "normalization_sha256"),
            ("calibration", self.calibration_path, "calibration_sha256"),
        ):
            expected = str(integrity_cfg.get(key) or "").lower()
            if expected and path is not None:
                actual = self._sha256(path)
                if expected != actual:
                    raise EdgeAIPackageError(
                        f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
                    )

    def _resolve_member(self, relative: str, label: str) -> Path:
        path = (self.package_dir / str(relative)).resolve()
        if path != self.package_dir and self.package_dir not in path.parents:
            raise EdgeAIPackageError(f"{label} escapes the package directory: {relative}")
        return path

    def _seconds_to_samples(self, seconds: float, label: str) -> int:
        samples = int(round(float(seconds) * self.expected_hz))
        if samples < 1:
            raise EdgeAIPackageError(
                f"{label}={seconds} s produces fewer than one sample at {self.expected_hz} Hz"
            )
        return samples

    @staticmethod
    def _optional_positive_float(value: Any, label: str) -> Optional[float]:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out) or out <= 0:
            raise EdgeAIPackageError(f"{label} must be positive")
        return out

    @staticmethod
    def _optional_nonnegative_float(value: Any, label: str) -> Optional[float]:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out) or out < 0:
            raise EdgeAIPackageError(f"{label} must be non-negative")
        return out

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_runtime_v5_code_bundle(
        cls, code_root: Path, manifest_path: Path
    ) -> tuple[str, str]:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EdgeAIPackageError(
                f"Invalid Runtime-v5 code manifest: {exc}"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "source_commit", "files"}
            or payload.get("schema_version") != "gaps.runtime_v5.code_bundle.v1"
        ):
            raise EdgeAIPackageError("Runtime-v5 code manifest schema differs")
        source_commit = str(payload.get("source_commit") or "").lower()
        if (
            len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise EdgeAIPackageError(
                "Runtime-v5 code manifest source commit is invalid"
            )
        files = payload.get("files")
        if not isinstance(files, Mapping) or not files:
            raise EdgeAIPackageError("Runtime-v5 code manifest contains no files")
        actual_python_files = {
            path.relative_to(code_root).as_posix()
            for path in code_root.rglob("*.py")
            if "__pycache__" not in path.parts
        }
        if set(files) != actual_python_files:
            raise EdgeAIPackageError(
                "Runtime-v5 code bundle file set differs from its manifest"
            )
        for relative, descriptor in files.items():
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or Path(relative).is_absolute()
                or any(part in {"", ".", ".."} for part in relative.split("/"))
            ):
                raise EdgeAIPackageError(
                    f"Runtime-v5 code path is not portable: {relative}"
                )
            if (
                not isinstance(descriptor, Mapping)
                or set(descriptor) != {"bytes", "sha256"}
                or not isinstance(descriptor.get("bytes"), int)
                or isinstance(descriptor.get("bytes"), bool)
            ):
                raise EdgeAIPackageError(
                    f"Runtime-v5 code descriptor differs: {relative}"
                )
            expected_sha = str(descriptor.get("sha256") or "").lower()
            if (
                len(expected_sha) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha
                )
            ):
                raise EdgeAIPackageError(
                    f"Runtime-v5 code SHA-256 is invalid: {relative}"
                )
            path = (code_root / relative).resolve()
            try:
                path.relative_to(code_root)
            except ValueError as exc:
                raise EdgeAIPackageError(
                    f"Runtime-v5 code file escapes code_root: {relative}"
                ) from exc
            if (
                not path.is_file()
                or path.stat().st_size != descriptor["bytes"]
                or cls._sha256(path) != expected_sha
            ):
                raise EdgeAIPackageError(
                    f"Runtime-v5 code file identity differs: {relative}"
                )
        return cls._sha256(manifest_path), source_commit


def prewarm_torchscript_package(package_dir: Path) -> Dict[str, Any]:
    """Initialize Torch and the exact TorchScript graph on the UI main thread.

    PyTorch CPU kernels can crash intermittently when their first execution
    occurs inside a Qt QThread on Raspberry Pi.  A single verified zero-window
    inference initializes those kernels before the package is handed to the
    worker.  Runtime-v5 packages are deliberately left unchanged.
    """
    package = EdgeAIPackage(Path(package_dir))
    if package.model_backend != "torchscript":
        return {
            "prewarmed": False,
            "model_backend": package.model_backend,
            "package_fingerprint": package.package_fingerprint,
        }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - deployment environment
        raise EdgeAIPackageError(
            "PyTorch is required to prewarm a TorchScript package"
        ) from exc
    assert package.model_path is not None
    try:
        model = torch.jit.load(str(package.model_path), map_location="cpu")
        model.eval()
        example = torch.zeros(
            (1, package.window_size, len(package.sensor_fields)),
            dtype=torch.float32,
        )
        with torch.no_grad():
            model(example)
    except Exception as exc:
        raise EdgeAIPackageError(
            f"TorchScript main-thread prewarm failed: {exc}"
        ) from exc
    return {
        "prewarmed": True,
        "model_backend": package.model_backend,
        "package_fingerprint": package.package_fingerprint,
        "input_shape": [1, package.window_size, len(package.sensor_fields)],
    }


class EdgeAIRuntime:
    """Stateful single-stream deployment runtime."""

    def __init__(self, package_dir: Path) -> None:
        self.package = EdgeAIPackage(package_dir)
        self.torch: Any = None
        self.model: Any = None
        self.runtime_v5: Any = None
        if self.package.model_backend == "torchscript":
            try:
                import torch
            except Exception as exc:  # pragma: no cover - deployment environment
                raise EdgeAIPackageError(
                    "PyTorch is required only for AI inference. "
                    "Install a Raspberry-Pi-compatible torch build."
                ) from exc
            self.torch = torch
            try:
                assert self.package.model_path is not None
                self.model = torch.jit.load(
                    str(self.package.model_path), map_location="cpu"
                )
                self.model.eval()
            except Exception as exc:
                raise EdgeAIPackageError(
                    f"Failed to load TorchScript model: {exc}"
                ) from exc
        else:
            self._load_runtime_v5()

        self._baseline_rows: List[np.ndarray] = []
        self._baseline: Optional[np.ndarray] = None
        self._window: Deque[np.ndarray] = deque(maxlen=self.package.window_size)
        self._window_meta: Deque[Dict[str, Any]] = deque(maxlen=self.package.window_size)
        self._timestamps: Deque[float] = deque(maxlen=max(20, self.package.window_size))
        self._samples_since_inference = 0
        self._warmup_collected = 0
        self._experiment_phase = (
            "automatic" if self.package.phase_mode == "automatic" else "unmarked"
        )
        self._last_input_ts: Optional[float] = None
        self._gap_reset_count = 0
        self._invalid_frame_count = 0
        self._inference_counter = 0
        self._consensus_probability_sum = np.zeros(
            len(self.package.gas_names), dtype=np.float64
        )
        self._consensus_window_count = 0
        if self.package.phase_mode != "automatic":
            self._last_status = "waiting_for_baseline_phase"
        elif self.package.unstable_samples > 0:
            self._last_status = "collecting_warmup"
        elif self.baseline_ready:
            self._last_status = "collecting_window"
        else:
            self._last_status = "collecting_baseline"

    def _load_runtime_v5(self) -> None:
        configured = str(os.environ.get("GAPS_RUNTIME_V5_PYTHONPATH", "")).strip()
        for value in configured.split(os.pathsep):
            if not value:
                continue
            root = Path(value).expanduser().resolve()
            if not root.is_dir():
                raise EdgeAIPackageError(
                    f"GAPS_RUNTIME_V5_PYTHONPATH entry is not a directory: {root}"
                )
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        if self.package.runtime_v5_code_root is not None:
            code_root_text = str(self.package.runtime_v5_code_root)
            if code_root_text in sys.path:
                sys.path.remove(code_root_text)
            sys.path.insert(0, code_root_text)
        try:
            module = importlib.import_module("gaps_deploy.runtime_v5_portable")
            load_runtime_v5_from_portable_binding = getattr(
                module, "load_runtime_v5_from_portable_binding"
            )
        except Exception as exc:
            raise EdgeAIPackageError(
                "Failed to import the formal GAPS Runtime-v5 implementation. "
                "Use a package with a verified runtime_v5 code bundle, or set "
                "GAPS_RUNTIME_V5_PYTHONPATH to a compatible repository/project "
                "root that contains gaps_deploy."
            ) from exc
        if self.package.runtime_v5_code_root is not None:
            try:
                module_path = Path(str(module.__file__)).resolve()
                module_path.relative_to(self.package.runtime_v5_code_root)
            except (TypeError, ValueError) as exc:
                raise EdgeAIPackageError(
                    "Imported Runtime-v5 implementation is outside the verified "
                    "package code root"
                ) from exc
        try:
            assert self.package.runtime_v5_binding_path is not None
            self.runtime_v5 = load_runtime_v5_from_portable_binding(
                self.package.runtime_v5_binding_path, device="cpu"
            )
        except Exception as exc:
            raise EdgeAIPackageError(
                f"Failed to load verified GAPS Runtime-v5 package: {exc}"
            ) from exc

    @property
    def baseline_ready(self) -> bool:
        return (
            self.package.feature_mode in {"raw_adc", "precomputed"}
            or self._baseline is not None
        )

    def reset_stream(self, keep_baseline: bool = False) -> None:
        self._window.clear()
        self._window_meta.clear()
        self._timestamps.clear()
        self._samples_since_inference = 0
        self._last_input_ts = None
        self._reset_consensus()
        if not keep_baseline:
            self._baseline_rows.clear()
            self._baseline = None
            self._warmup_collected = 0
            if self.package.phase_mode == "event_driven":
                self._experiment_phase = "unmarked"
        if self.package.phase_mode == "event_driven" and self._experiment_phase == "unmarked":
            self._last_status = "waiting_for_baseline_phase"
        elif not self.baseline_ready:
            self._last_status = (
                "collecting_warmup"
                if self._warmup_collected < self.package.unstable_samples
                else "collecting_baseline"
            )
        elif self._phase_allows_inference():
            self._last_status = "collecting_window"
        else:
            self._last_status = "baseline_ready"

    def set_experiment_phase(self, phase: str) -> None:
        """Apply a real experiment boundary to the streaming AI state."""
        phase = str(phase or "").strip().lower()
        if phase not in {"unmarked", "baseline", "exposure", "recovery"}:
            raise EdgeAIPackageError(f"Unsupported experiment phase: {phase}")
        previous = self._experiment_phase
        if phase == "baseline":
            self.reset_stream(keep_baseline=False)
            self._experiment_phase = "baseline"
            self._last_status = (
                "collecting_warmup"
                if self.package.unstable_samples > 0
                else "collecting_baseline"
            )
            return
        self._experiment_phase = phase
        if phase != previous:
            self._clear_window()
            self._reset_consensus()
        if not self.baseline_ready:
            self._last_status = "baseline_required"
        elif self._phase_allows_inference():
            self._last_status = "collecting_window"
        else:
            self._last_status = "baseline_ready"

    def status(self) -> Dict[str, Any]:
        baseline_n = len(self._baseline_rows) if self._baseline is None else self.package.baseline_samples
        return {
            "state": self._last_status,
            "package_name": self.package.package_name,
            "baseline_ready": self.baseline_ready,
            "baseline_collected": baseline_n,
            "baseline_required": (
                0
                if self.package.feature_mode in {"raw_adc", "precomputed"}
                else self.package.baseline_samples
            ),
            "warmup_collected": self._warmup_collected,
            "warmup_required": self.package.unstable_samples,
            "window_collected": len(self._window),
            "window_required": self.package.window_size,
            "observed_hz": self._observed_hz(),
            "expected_hz": self.package.expected_hz,
            "feature_mode": self.package.feature_mode,
            "schema_version": self.package.schema_version,
            "dataset_profile": self.package.dataset_profile,
            "device_profile": self.package.device_profile,
            "model_backend": self.package.model_backend,
            "task_type": self.package.task_type,
            "has_concentration": self.package.has_concentration,
            "runtime_v5_release_id": self.package.runtime_v5_release_id,
            "runtime_v5_qc_status": (
                "disabled_pending_dependency_audit"
                if self.package.model_backend == "gaps_runtime_v5"
                else ""
            ),
            "normalization_enabled": self.package.normalization_enabled,
            "package_fingerprint": self.package.package_fingerprint,
            "experiment_phase": self._experiment_phase,
            "phase_mode": self.package.phase_mode,
            "gap_reset_count": self._gap_reset_count,
            "invalid_frame_count": self._invalid_frame_count,
            "max_gap_s": self.package.max_gap_s,
            "consensus_window_count": self._consensus_window_count,
        }

    def append_row(self, row: Dict[str, Any]) -> Optional[EdgeAIResult]:
        ts = self._float(row.get("timestamp_unix"), default=time.time())
        if not math.isfinite(ts):
            raise EdgeAIPackageError("Incoming frame timestamp is not finite")
        if (
            self.package.feature_mode == "precomputed"
            and row.get("_model_input_precomputed") is not True
        ):
            raise EdgeAIPackageError(
                "This replay-only package requires explicitly marked "
                "precomputed model inputs and rejects raw STM32 serial frames"
            )

        if self.package.reject_implausible_frames and not self._row_is_plausible(row):
            self._invalid_frame_count += 1
            self._clear_window()
            self._last_input_ts = ts
            self._last_status = "invalid_frame_reset"
            return None

        if self._last_input_ts is not None:
            gap_s = ts - self._last_input_ts
            if gap_s <= 0 or gap_s > self.package.max_gap_s:
                self._gap_reset_count += 1
                self._clear_window()
                self._last_status = "gap_reset"
        self._last_input_ts = ts

        raw = self._extract_raw(row)
        self._timestamps.append(ts)

        if self.package.phase_mode == "event_driven" and self._experiment_phase == "unmarked":
            self._last_status = "waiting_for_baseline_phase"
            return None

        if self._warmup_collected < self.package.unstable_samples:
            self._warmup_collected += 1
            self._last_status = "collecting_warmup"
            return None

        if (
            self.package.feature_mode not in {"raw_adc", "precomputed"}
            and self._baseline is None
        ):
            if self.package.phase_mode == "event_driven" and self._experiment_phase != "baseline":
                self._last_status = "baseline_required"
                return None
            self._baseline_rows.append(raw)
            if len(self._baseline_rows) >= self.package.baseline_samples:
                self._baseline = np.mean(np.stack(self._baseline_rows, axis=0), axis=0).astype(np.float32)
                self._baseline_rows.clear()
                self._last_status = (
                    "collecting_window"
                    if self._phase_allows_inference()
                    else "baseline_ready"
                )
            else:
                self._last_status = "collecting_baseline"
            return None

        if not self._phase_allows_inference():
            self._last_status = "baseline_ready"
            return None

        feature = self._transform(raw)
        self._window.append(feature)
        self._window_meta.append(self._row_meta(row, ts))
        self._samples_since_inference += 1
        if len(self._window) < self.package.window_size:
            self._last_status = "collecting_window"
            return None
        if self._samples_since_inference < self.package.stride:
            self._last_status = "window_ready"
            return None

        observed_hz = self._observed_hz()
        rate_ok = self._rate_ok(observed_hz)
        if not rate_ok and not self.package.allow_rate_mismatch:
            self._last_status = "rate_mismatch"
            return None

        self._samples_since_inference = 0
        self._last_status = "inferencing"
        result = self._infer(row, observed_hz)
        self._last_status = "ready"
        return result

    def _clear_window(self) -> None:
        self._window.clear()
        self._window_meta.clear()
        self._timestamps.clear()
        self._samples_since_inference = 0

    def _reset_consensus(self) -> None:
        self._consensus_probability_sum.fill(0.0)
        self._consensus_window_count = 0

    def _phase_allows_inference(self) -> bool:
        if self.package.phase_mode == "automatic":
            return True
        return self._experiment_phase in set(self.package.inference_phases)

    @staticmethod
    def _row_is_plausible(row: Dict[str, Any]) -> bool:
        value = row.get("frame_plausible", 1)
        try:
            return bool(int(float(value)))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _row_meta(row: Dict[str, Any], ts: float) -> Dict[str, Any]:
        return {
            "timestamp_unix": float(ts),
            "timestamp_iso": str(row.get("timestamp_iso", "")),
            "stream_frame_index": int(
                EdgeAIRuntime._float(
                    row.get("stream_frame_index", row.get("frame_index", -1)), -1
                )
            ),
            "connection_id": int(EdgeAIRuntime._float(row.get("connection_id", -1), -1)),
            "recording_active": bool(row.get("_recording_active", False)),
            "recording_session_id": str(row.get("_recording_session_id", "")),
        }

    def _extract_raw(self, row: Dict[str, Any]) -> np.ndarray:
        values: List[float] = []
        for field in self.package.sensor_fields:
            value = self._float(row.get(field), default=math.nan)
            if not math.isfinite(value):
                raise EdgeAIPackageError(f"Incoming frame is missing numeric field: {field}")
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _transform(self, raw: np.ndarray) -> np.ndarray:
        mode = self.package.feature_mode
        if mode in {"raw_adc", "precomputed"}:
            return raw.astype(np.float32)
        assert self._baseline is not None
        eps = 1e-6
        if mode == "relative_adc":
            return ((raw - self._baseline) / (np.abs(self._baseline) + eps)).astype(np.float32)
        if mode == "relative_conductance":
            current_g = self._adc_to_conductance(raw)
            base_g = self._adc_to_conductance(self._baseline)
            return ((current_g - base_g) / (np.abs(base_g) + eps)).astype(np.float32)
        raise EdgeAIPackageError(f"Unsupported feature mode: {mode}")

    def _adc_to_conductance(self, raw: np.ndarray) -> np.ndarray:
        voltage = np.clip(raw, 0.0, self.package.adc_max) * self.package.adc_vref / self.package.adc_max
        voltage = np.clip(voltage, 1e-6, self.package.adc_vref - 1e-6)
        rload = np.asarray(self.package.rload_ohm, dtype=np.float32)
        resistance = (self.package.adc_vref - voltage) * rload / voltage
        return 1.0 / np.clip(resistance, 1e-6, None)

    def _infer(self, row: Dict[str, Any], observed_hz: float) -> EdgeAIResult:
        window = np.stack(list(self._window), axis=0).astype(np.float32)
        if self.package.normalization_enabled:
            window = (window - self.package.mean) / self.package.std
        if not np.all(np.isfinite(window)):
            raise EdgeAIPackageError("Prepared AI window contains non-finite values")
        if self.package.model_backend == "gaps_runtime_v5":
            return self._infer_runtime_v5(window, row, observed_hz)
        x = self.torch.from_numpy(window).unsqueeze(0)

        start = time.perf_counter()
        with self.torch.no_grad():
            output = self.model(x)
        latency_ms = (time.perf_counter() - start) * 1000.0
        parsed = self._parse_model_output(output)

        logits = parsed["logits"]
        if not hasattr(logits, "dim") or logits.dim() < 1:
            raise EdgeAIPackageError("TorchScript logits output is not a tensor with a class dimension")
        if logits.shape[-1] != len(self.package.gas_names):
            raise EdgeAIPackageError(
                f"TorchScript logits has {logits.shape[-1]} classes, "
                f"manifest declares {len(self.package.gas_names)}"
            )
        probs = self.torch.softmax(logits, dim=-1)
        confidence_tensor, class_tensor = probs.max(dim=-1)
        pred_class = int(class_tensor.reshape(-1)[0].item())
        confidence = float(confidence_tensor.reshape(-1)[0].item())
        if not math.isfinite(confidence):
            raise EdgeAIPackageError("TorchScript confidence is not finite")
        gas = self.package.gas_names[pred_class] if 0 <= pred_class < len(self.package.gas_names) else f"Class{pred_class}"

        probability_values = (
            probs.reshape(-1, len(self.package.gas_names))[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if not np.all(np.isfinite(probability_values)):
            raise EdgeAIPackageError("TorchScript class probabilities are not finite")
        self._consensus_probability_sum += probability_values
        self._consensus_window_count += 1
        consensus_values = (
            self._consensus_probability_sum / self._consensus_window_count
        )
        consensus_class = int(np.argmax(consensus_values))
        consensus_confidence = float(consensus_values[consensus_class])
        consensus_gas = self.package.gas_names[consensus_class]

        if self.package.has_concentration:
            ppm_tensor = parsed["ppm"]
            if not hasattr(ppm_tensor, "reshape"):
                raise EdgeAIPackageError("TorchScript ppm output is not tensor-like")
            flat_ppm = ppm_tensor.reshape(-1)
            if flat_ppm.numel() < 1:
                raise EdgeAIPackageError("TorchScript ppm output is empty")
            if flat_ppm.numel() == len(self.package.gas_names):
                base_ppm = float(flat_ppm[pred_class].item())
            else:
                base_ppm = float(flat_ppm[0].item())
            if not math.isfinite(base_ppm):
                raise EdgeAIPackageError("TorchScript ppm output is not finite")
            ppm, calibration_name = self._apply_calibration(base_ppm, pred_class)
        else:
            base_ppm = None
            ppm = None
            calibration_name = "not_applicable"

        risk_score = parsed.get("risk_score")
        risk_value: Optional[float]
        if not self.package.has_concentration:
            risk_value = None
            risk_name = "not_validated_for_lab_classification"
        elif risk_score is None:
            risk_value = 1.0 - confidence
            risk_name = "classifier_uncertainty"
        else:
            risk_value = float(risk_score.reshape(-1)[0].item())
            risk_name = str(self.package.qc_cfg.get("risk_score_name", "model_risk_score"))
        if risk_value is not None and not math.isfinite(risk_value):
            raise EdgeAIPackageError("TorchScript/QC risk score is not finite")

        if not self.package.has_concentration:
            decision = "unavailable_qc_not_validated"
            accepted = False
            selected_policy = "classification_qc_not_validated"
        else:
            min_conf = float(self.package.qc_cfg.get("min_confidence", 0.0))
            accept_max = float(self.package.qc_cfg.get("accept_max_risk", self.package.qc_cfg.get("max_risk_score", float("inf"))))
            reject_min = float(self.package.qc_cfg.get("reject_min_risk", float("inf")))
            if confidence < min_conf:
                decision = str(self.package.qc_cfg.get("low_confidence_decision", "review"))
            elif risk_value is not None and risk_value >= reject_min:
                decision = "reject"
            elif risk_value is None or risk_value <= accept_max:
                decision = "accept"
            else:
                decision = "review"
            accepted = decision == "accept"
            selected_policy = str(self.package.qc_cfg.get("policy_name", "package_qc"))
        meta = list(self._window_meta)
        if len(meta) != self.package.window_size:
            raise EdgeAIPackageError("AI window metadata is incomplete")
        first_meta = meta[0]
        last_meta = meta[-1]
        connection_ids = {int(item["connection_id"]) for item in meta}
        recording_ids = {
            str(item["recording_session_id"])
            for item in meta
            if bool(item["recording_active"])
        }
        recording_complete = bool(
            meta
            and all(bool(item["recording_active"]) for item in meta)
            and len(recording_ids) == 1
            and "" not in recording_ids
        )
        recording_session_id = next(iter(recording_ids)) if recording_complete else ""
        self._inference_counter += 1

        return EdgeAIResult(
            timestamp_iso=str(row.get("timestamp_iso", "")),
            stream_frame_index=int(self._float(row.get("stream_frame_index", row.get("frame_index", -1)), -1)),
            predicted_class=pred_class,
            predicted_gas=gas,
            confidence=confidence,
            class_probabilities=probability_values.astype(float).tolist(),
            consensus_predicted_class=consensus_class,
            consensus_predicted_gas=consensus_gas,
            consensus_confidence=consensus_confidence,
            consensus_probabilities=consensus_values.astype(float).tolist(),
            consensus_window_count=self._consensus_window_count,
            task_type=self.package.task_type,
            has_concentration=self.package.has_concentration,
            ppm_base_prediction=base_ppm,
            ppm_full_prediction=ppm,
            ppm_auto_output=ppm if accepted and ppm is not None else None,
            decision=decision,
            selected_calibration=calibration_name,
            selected_policy=selected_policy,
            risk_score=risk_value,
            risk_score_name=risk_name,
            inference_latency_ms=latency_ms,
            observed_hz=observed_hz,
            window_size=self.package.window_size,
            package_name=self.package.package_name,
            package_fingerprint=self.package.package_fingerprint,
            dataset_profile=self.package.dataset_profile,
            device_profile=self.package.device_profile,
            model_backend=self.package.model_backend,
            normalization_applied=self.package.normalization_enabled,
            experiment_phase=self._experiment_phase,
            inference_id=self._inference_counter,
            window_start_timestamp_iso=str(first_meta["timestamp_iso"]),
            window_end_timestamp_iso=str(last_meta["timestamp_iso"]),
            window_start_stream_frame_index=int(first_meta["stream_frame_index"]),
            window_end_stream_frame_index=int(last_meta["stream_frame_index"]),
            window_connection_id=(
                next(iter(connection_ids)) if len(connection_ids) == 1 else -1
            ),
            window_recording_complete=recording_complete,
            recording_session_id=recording_session_id,
        )

    def _infer_runtime_v5(
        self,
        window: np.ndarray,
        row: Dict[str, Any],
        observed_hz: float,
    ) -> EdgeAIResult:
        if self.runtime_v5 is None:
            raise EdgeAIPackageError("GAPS Runtime-v5 is not loaded")
        metadata = dict(self.package.runtime_v5_metadata)
        row_metadata = row.get("_gaps_runtime_metadata")
        if row_metadata is not None:
            if not isinstance(row_metadata, Mapping):
                raise EdgeAIPackageError(
                    "_gaps_runtime_metadata must be a mapping when provided"
                )
            unexpected = set(row_metadata) - RUNTIME_V5_METADATA_FIELDS
            if unexpected:
                raise EdgeAIPackageError(
                    "_gaps_runtime_metadata contains unsupported fields: "
                    + ", ".join(sorted(unexpected))
                )
            metadata.update(row_metadata)

        start = time.perf_counter()
        try:
            rows = self.runtime_v5.infer(
                window[np.newaxis, :, :],
                [metadata],
                np.asarray([self.package.runtime_v5_phase_id], dtype=np.int64),
            )
        except Exception as exc:
            raise EdgeAIPackageError(f"GAPS Runtime-v5 inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise EdgeAIPackageError("GAPS Runtime-v5 output row count differs")
        output = rows[0]
        required = {
            "sample_index",
            "pred_class",
            "prediction_ppm",
            "source_h1_ppm",
            "max_probability",
            "qc_status",
            "auto_output_ppm",
        }
        if set(output) != required or output.get("sample_index") != 0:
            raise EdgeAIPackageError("GAPS Runtime-v5 output schema differs")
        pred_class = output.get("pred_class")
        if (
            not isinstance(pred_class, int)
            or isinstance(pred_class, bool)
            or pred_class not in (0, 1, 2, 3)
        ):
            raise EdgeAIPackageError("GAPS Runtime-v5 predicted class is invalid")
        prediction_ppm = self._finite_runtime_value(output, "prediction_ppm")
        source_h1_ppm = self._finite_runtime_value(output, "source_h1_ppm")
        confidence = self._finite_runtime_value(output, "max_probability")
        if not 0.0 <= confidence <= 1.0:
            raise EdgeAIPackageError(
                "GAPS Runtime-v5 max_probability is outside 0..1"
            )
        if output.get("qc_status") != "disabled_pending_dependency_audit":
            raise EdgeAIPackageError("GAPS Runtime-v5 QC status differs")
        if output.get("auto_output_ppm") is not None:
            raise EdgeAIPackageError(
                "GAPS Runtime-v5 core must not emit an automatic ppm output"
            )

        meta = list(self._window_meta)
        if len(meta) != self.package.window_size:
            raise EdgeAIPackageError("AI window metadata is incomplete")
        first_meta = meta[0]
        last_meta = meta[-1]
        connection_ids = {int(item["connection_id"]) for item in meta}
        recording_ids = {
            str(item["recording_session_id"])
            for item in meta
            if bool(item["recording_active"])
        }
        recording_complete = bool(
            meta
            and all(bool(item["recording_active"]) for item in meta)
            and len(recording_ids) == 1
            and "" not in recording_ids
        )
        recording_session_id = (
            next(iter(recording_ids)) if recording_complete else ""
        )
        self._inference_counter += 1
        gas = self.package.gas_names[pred_class]
        return EdgeAIResult(
            timestamp_iso=str(row.get("timestamp_iso", "")),
            stream_frame_index=int(
                self._float(
                    row.get("stream_frame_index", row.get("frame_index", -1)), -1
                )
            ),
            predicted_class=pred_class,
            predicted_gas=gas,
            confidence=confidence,
            class_probabilities=[],
            consensus_predicted_class=pred_class,
            consensus_predicted_gas=gas,
            consensus_confidence=confidence,
            consensus_probabilities=[],
            consensus_window_count=1,
            task_type=self.package.task_type,
            has_concentration=self.package.has_concentration,
            ppm_base_prediction=source_h1_ppm,
            ppm_full_prediction=prediction_ppm,
            ppm_auto_output=None,
            decision="disabled_pending_dependency_audit",
            selected_calibration="c5_105d_target_ridge",
            selected_policy="runtime_v5_core_qc_disabled",
            risk_score=None,
            risk_score_name="not_available_qc_disabled",
            inference_latency_ms=latency_ms,
            observed_hz=observed_hz,
            window_size=self.package.window_size,
            package_name=self.package.package_name,
            package_fingerprint=self.package.package_fingerprint,
            dataset_profile=self.package.dataset_profile,
            device_profile=self.package.device_profile,
            model_backend=self.package.model_backend,
            normalization_applied=False,
            experiment_phase=self._experiment_phase,
            inference_id=self._inference_counter,
            window_start_timestamp_iso=str(first_meta["timestamp_iso"]),
            window_end_timestamp_iso=str(last_meta["timestamp_iso"]),
            window_start_stream_frame_index=int(first_meta["stream_frame_index"]),
            window_end_stream_frame_index=int(last_meta["stream_frame_index"]),
            window_connection_id=(
                next(iter(connection_ids)) if len(connection_ids) == 1 else -1
            ),
            window_recording_complete=recording_complete,
            recording_session_id=recording_session_id,
        )

    @staticmethod
    def _finite_runtime_value(output: Mapping[str, Any], field: str) -> float:
        value = output.get(field)
        if isinstance(value, bool):
            raise EdgeAIPackageError(
                f"GAPS Runtime-v5 output field is invalid: {field}"
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise EdgeAIPackageError(
                f"GAPS Runtime-v5 output field is invalid: {field}"
            ) from exc
        if not math.isfinite(result):
            raise EdgeAIPackageError(
                f"GAPS Runtime-v5 output field is non-finite: {field}"
            )
        return result

    def _apply_calibration(self, value: float, pred_class: int) -> tuple[float, str]:
        per_class = dict(self.package.calibration_cfg.get("per_class") or {})
        cfg = dict(per_class.get(str(pred_class)) or per_class.get(pred_class) or {})
        mode = str(cfg.get("mode", "none")).lower()
        out = float(value)
        if mode == "bias":
            out = out + float(cfg.get("bias", 0.0))
        elif mode == "affine":
            out = float(cfg.get("scale", 1.0)) * out + float(cfg.get("bias", 0.0))
        elif mode == "piecewise_affine":
            threshold = float(cfg.get("threshold", 0.0))
            prefix = "low" if out <= threshold else "high"
            out = float(cfg.get(f"{prefix}_scale", 1.0)) * out + float(cfg.get(f"{prefix}_bias", 0.0))
        elif mode not in {"none", ""}:
            raise EdgeAIPackageError(f"Unsupported calibration mode={mode} for class {pred_class}")
        if "min" in cfg:
            out = max(out, float(cfg["min"]))
        if "max" in cfg:
            out = min(out, float(cfg["max"]))
        return float(out), mode or "none"

    def _parse_model_output(self, output: Any) -> Dict[str, Any]:
        if isinstance(output, dict):
            logits = output.get("logits")
            ppm = output.get("ppm")
            if ppm is None:
                ppm = output.get("pred_ppm")
            risk = output.get("risk_score")
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            logits, ppm = output[0], output[1]
            risk = output[2] if len(output) >= 3 else None
        elif self.package.task_type == "classification" and hasattr(output, "dim"):
            logits = output
            ppm = None
            risk = None
        else:
            raise EdgeAIPackageError(
                "TorchScript output must be logits for classification, or a dict/tuple "
                "containing logits and ppm for classification_regression"
            )
        if logits is None:
            raise EdgeAIPackageError("TorchScript output is missing logits")
        if self.package.has_concentration and ppm is None:
            raise EdgeAIPackageError("TorchScript output is missing ppm")
        return {"logits": logits, "ppm": ppm, "risk_score": risk}

    def _observed_hz(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        duration = float(self._timestamps[-1] - self._timestamps[0])
        if duration <= 1e-9:
            return 0.0
        return float((len(self._timestamps) - 1) / duration)

    def _rate_ok(self, observed_hz: float) -> bool:
        if self.package.expected_hz <= 0 or observed_hz <= 0:
            return False
        ratio = observed_hz / self.package.expected_hz
        return self.package.min_rate_ratio <= ratio <= self.package.max_rate_ratio

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
