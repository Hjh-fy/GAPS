#!/usr/bin/env python3
"""End-to-end self-test for the schema-v2 streaming edge-AI runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from edge_ai_runtime import EdgeAIPackage, EdgeAIPackageError, EdgeAIRuntime

try:
    import torch
except ImportError:  # Acquisition-only installations intentionally omit PyTorch.
    torch = None


if torch is not None:
    class DummyEdgeModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            score = x.mean(dim=(1, 2))
            logits = torch.stack(
                [score + 4.0, score, score - 1.0, score - 2.0], dim=1
            )
            ppm = torch.stack(
                [score + 50.0, score + 100.0, score + 75.0, score + 125.0],
                dim=1,
            )
            risk = torch.full_like(score, 0.1)
            return logits, ppm, risk


def row(index: int, *, plausible: bool = True, session: str = "session-a") -> dict:
    return {
        "timestamp_iso": f"2026-01-01T00:00:{index:02d}",
        "timestamp_unix": float(index),
        "stream_frame_index": index,
        "connection_id": 1,
        "frame_plausible": int(plausible),
        "adc_ch0_pa0": 100.0 + index,
        "adc_ch1_pa1": 200.0 + index,
        "_recording_active": True,
        "_recording_session_id": session,
    }


def main() -> int:
    if torch is None:
        print("Schema-v2 runtime self-test skipped: PyTorch is not installed.")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = Path(tmp)
        model_path = package_dir / "model.ts"
        example = torch.zeros((1, 4, 2), dtype=torch.float32)
        traced = torch.jit.trace(DummyEdgeModel().eval(), example)
        traced.save(str(model_path))
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

        manifest = {
            "schema_version": 2,
            "package_name": "schema_v2_runtime_self_test",
            "dataset_profile": "lab_1hz_test",
            "device_profile": "stm32_test",
            "model_backend": "torchscript",
            "model_file": "model.ts",
            "input": {
                "sensor_fields": ["adc_ch0_pa0", "adc_ch1_pa1"],
                "feature_mode": "relative_adc",
                "raw_sample_hz": 1.0,
                "target_sample_hz": 1.0,
                "unstable_duration_s": 2.0,
                "baseline_duration_s": 3.0,
                "window_duration_s": 4.0,
                "stride_duration_s": 2.0,
                "max_gap_s": 2.0,
                "min_rate_ratio": 0.8,
                "max_rate_ratio": 1.2,
                "reject_implausible_frames": True,
            },
            "normalization": {"enabled": False},
            "phase_control": {
                "mode": "event_driven",
                "inference_phases": ["exposure", "recovery"],
            },
            "output": {
                "gas_names": ["Ethanol", "CO", "Ethylene", "Methane"]
            },
            "qc": {
                "policy_name": "self_test_qc",
                "min_confidence": 0.0,
                "accept_max_risk": 0.3,
                "reject_min_risk": 0.6,
            },
            "integrity": {"model_sha256": model_sha256},
        }
        (package_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        runtime = EdgeAIRuntime(package_dir)
        assert runtime.status()["state"] == "waiting_for_baseline_phase"
        assert runtime.package.normalization_enabled is False
        assert runtime.package.window_size == 4
        assert runtime.package.baseline_samples == 3

        runtime.set_experiment_phase("baseline")
        for index in range(5):
            assert runtime.append_row(row(index)) is None
        assert runtime.baseline_ready
        assert runtime.status()["state"] == "baseline_ready"

        runtime.set_experiment_phase("exposure")
        result = None
        for index in range(5, 9):
            result = runtime.append_row(row(index))
        assert result is not None
        payload = result.to_dict()
        assert payload["decision"] == "accept"
        assert payload["normalization_applied"] is False
        assert payload["experiment_phase"] == "exposure"
        assert payload["window_start_stream_frame_index"] == 5
        assert payload["window_end_stream_frame_index"] == 8
        assert payload["window_connection_id"] == 1
        assert payload["window_recording_complete"] is True
        assert payload["recording_session_id"] == "session-a"
        assert payload["package_fingerprint"]

        assert runtime.append_row(row(9, plausible=False)) is None
        assert runtime.status()["invalid_frame_count"] == 1
        assert runtime.status()["window_collected"] == 0

        assert runtime.append_row(row(10)) is None
        assert runtime.append_row(row(14)) is None
        assert runtime.status()["gap_reset_count"] >= 1
        assert runtime.status()["window_collected"] == 1

        built_dir = package_dir / "built_package"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("build_edge_ai_package.py")),
                "--model-ts",
                str(model_path),
                "--output-dir",
                str(built_dir),
                "--package-name",
                "builder_v2_test",
                "--dataset-profile",
                "lab_1hz_test",
                "--device-profile",
                "stm32_test",
                "--sensor-fields",
                "adc_ch0_pa0,adc_ch1_pa1",
                "--raw-sample-hz",
                "1",
                "--target-sample-hz",
                "1",
                "--unstable-duration-s",
                "2",
                "--baseline-duration-s",
                "3",
                "--window-duration-s",
                "4",
                "--stride-duration-s",
                "2",
                "--feature-mode",
                "relative_adc",
                "--normalization",
                "disabled",
                "--phase-mode",
                "event_driven",
            ],
            check=True,
        )
        built = EdgeAIPackage(built_dir)
        assert built.schema_version == 2
        assert built.normalization_enabled is False
        assert built.expected_hz == 1.0
        assert built.window_size == 4
        assert built.package_fingerprint
        with (built_dir / "model.ts").open("ab") as handle:
            handle.write(b"tamper")
        try:
            EdgeAIPackage(built_dir)
        except EdgeAIPackageError as exc:
            assert "SHA-256 mismatch" in str(exc)
        else:
            raise AssertionError("tampered model package was not rejected")

    print("Schema-v2 edge AI runtime self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
