#!/usr/bin/env python3
"""Build a portable GAPS edge-AI package from already-exported artifacts.

This helper intentionally does not convert a training checkpoint.  Export the
final classification/regression/calibration graph to TorchScript inside the
GAPS repository first, then use this script to assemble and validate the files
that the Raspberry Pi UI consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import List

import numpy as np

from edge_ai_runtime import EdgeAIPackage


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a GAPS Raspberry Pi deployment package.")
    parser.add_argument("--model-ts", required=True)
    parser.add_argument(
        "--norm-stats",
        default="",
        help="Required only when --normalization enabled.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-name", default="gaps_edge_package")
    parser.add_argument("--dataset-profile", default="unspecified")
    parser.add_argument("--device-profile", default="unspecified")
    parser.add_argument(
        "--sensor-fields",
        default=",".join([f"adc_ch{i}_{'pa'+str(i) if i < 8 else 'pb'+str(i-8)}" for i in range(8)]),
        help="Comma-separated fields from raw.csv. Prefer passing this explicitly.",
    )
    parser.add_argument("--raw-sample-hz", type=float, default=10.0)
    parser.add_argument("--target-sample-hz", type=float, default=10.0)
    parser.add_argument("--unstable-duration-s", type=float, default=0.0)
    parser.add_argument("--baseline-duration-s", type=float, default=30.0)
    parser.add_argument("--window-duration-s", type=float, default=10.0)
    parser.add_argument("--stride-duration-s", type=float, default=5.0)
    parser.add_argument("--feature-mode", choices=["raw_adc", "relative_adc", "relative_conductance"], default="relative_adc")
    parser.add_argument("--rload-ohm", default="", help="Required for relative_conductance; comma-separated.")
    parser.add_argument(
        "--normalization",
        choices=["enabled", "disabled"],
        default="disabled",
        help="Must match the frozen training/runtime contract.",
    )
    parser.add_argument(
        "--phase-mode",
        choices=["automatic", "event_driven"],
        default="event_driven",
    )
    parser.add_argument("--inference-phases", default="exposure,recovery")
    parser.add_argument("--max-gap-s", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--accept-max-risk", type=float, default=0.30)
    parser.add_argument("--reject-min-risk", type=float, default=0.60)
    parser.add_argument("--calibration-json", default="")
    args = parser.parse_args()

    model_src = Path(args.model_ts).expanduser().resolve()
    if not model_src.exists():
        raise FileNotFoundError(model_src)
    normalization_enabled = args.normalization == "enabled"
    norm_src = Path(args.norm_stats).expanduser().resolve() if args.norm_stats else None
    if normalization_enabled and (norm_src is None or not norm_src.exists()):
        raise FileNotFoundError("--norm-stats is required when normalization is enabled")

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, out / "model.ts")
    if normalization_enabled and norm_src is not None:
        shutil.copy2(norm_src, out / "norm_stats.npz")
        # Fail early if the expected normalization arrays are absent.
        norm = np.load(out / "norm_stats.npz")
        if "mean" not in norm or "std" not in norm:
            raise ValueError("norm_stats.npz must contain arrays named mean and std")

    sensor_fields = parse_csv(args.sensor_fields)
    rload_ohm = [float(x) for x in parse_csv(args.rload_ohm)]
    if args.feature_mode == "relative_conductance" and len(rload_ohm) != len(sensor_fields):
        raise ValueError(
            "--rload-ohm must provide one value per --sensor-fields entry "
            "for relative_conductance"
        )
    model_sha256 = hashlib.sha256((out / "model.ts").read_bytes()).hexdigest()
    max_gap_s = (
        float(args.max_gap_s)
        if args.max_gap_s > 0
        else max(3.0 / float(args.target_sample_hz), 1.0)
    )
    integrity = {"model_sha256": model_sha256}
    if normalization_enabled:
        integrity["normalization_sha256"] = hashlib.sha256(
            (out / "norm_stats.npz").read_bytes()
        ).hexdigest()
    manifest = {
        "schema_version": 2,
        "package_name": args.package_name,
        "dataset_profile": args.dataset_profile,
        "device_profile": args.device_profile,
        "model_backend": "torchscript",
        "model_file": "model.ts",
        "input": {
            "sensor_fields": sensor_fields,
            "feature_mode": args.feature_mode,
            "raw_sample_hz": args.raw_sample_hz,
            "target_sample_hz": args.target_sample_hz,
            "unstable_duration_s": args.unstable_duration_s,
            "baseline_duration_s": args.baseline_duration_s,
            "window_duration_s": args.window_duration_s,
            "stride_duration_s": args.stride_duration_s,
            "min_rate_ratio": 0.70,
            "max_rate_ratio": 1.30,
            "allow_rate_mismatch": False,
            "max_gap_s": max_gap_s,
            "reject_implausible_frames": True,
        },
        "normalization": {
            "enabled": normalization_enabled,
            **(
                {"file": "norm_stats.npz", "mean_key": "mean", "std_key": "std"}
                if normalization_enabled
                else {}
            ),
        },
        "phase_control": {
            "mode": args.phase_mode,
            "inference_phases": parse_csv(args.inference_phases),
        },
        "output": {"gas_names": ["Ethanol", "CO", "Ethylene", "Methane"]},
        "qc": {
            "policy_name": "gaps_package_qc",
            "min_confidence": args.min_confidence,
            "accept_max_risk": args.accept_max_risk,
            "reject_min_risk": args.reject_min_risk,
                "risk_score_name": "model_or_classifier_risk",
        },
        "integrity": integrity,
    }
    if args.feature_mode == "relative_conductance":
        manifest["input"]["rload_ohm"] = rload_ohm
    if args.calibration_json:
        cal_src = Path(args.calibration_json).expanduser().resolve()
        shutil.copy2(cal_src, out / "calibration.json")
        manifest["calibration_file"] = "calibration.json"
        manifest["integrity"]["calibration_sha256"] = hashlib.sha256(
            (out / "calibration.json").read_bytes()
        ).hexdigest()

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    package = EdgeAIPackage(out)
    print(f"Package validated: {package.package_name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
