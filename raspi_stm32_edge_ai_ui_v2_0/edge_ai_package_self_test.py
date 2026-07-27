#!/usr/bin/env python3
"""Headless schema/preprocessing package validation without loading PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from edge_ai_runtime import EdgeAIPackage


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "model.ts").write_bytes(b"placeholder")
        np.savez(root / "norm_stats.npz", mean=np.zeros((1, 8), np.float32), std=np.ones((1, 8), np.float32))
        manifest = {
            "schema_version": 1,
            "package_name": "self_test",
            "model_file": "model.ts",
            "input": {
                "sensor_fields": [f"adc_ch{i}" for i in range(8)],
                "window_size": 100,
                "stride": 50,
                "feature_mode": "relative_adc",
                "baseline_samples": 10,
                "expected_hz": 10.0
            },
            "normalization": {"file": "norm_stats.npz"},
            "output": {"gas_names": ["Ethanol", "CO", "Ethylene", "Methane"]}
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        package = EdgeAIPackage(root)
        assert package.window_size == 100
        assert len(package.sensor_fields) == 8
        assert package.mean.shape == (1, 8)
    print("Edge AI package schema self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
