#!/usr/bin/env python3
"""Replay frozen public calibration windows through the full UI runtime path.

This diagnostic deliberately loads no class or concentration labels. The
frozen calibration parity rows are used only as a deployment-output oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np

from edge_ai_runtime import EdgeAIPackageError, EdgeAIRuntime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay N×100×8 public calibration windows through EdgeAIRuntime "
            "and compare with frozen Runtime-v5 parity rows."
        )
    )
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 replays all rows; a positive value replays the first N rows.",
    )
    return parser.parse_args()


def stream_one_window(
    runtime: EdgeAIRuntime, window: np.ndarray, window_index: int
) -> dict[str, Any]:
    runtime.reset_stream(keep_baseline=False)
    result = None
    base_timestamp = 1_700_000_000.0 + window_index * 20.0
    session_id = f"public-calibration-window-{window_index:04d}"
    for sample_index, values in enumerate(window):
        row: dict[str, Any] = {
            field: float(values[channel])
            for channel, field in enumerate(runtime.package.sensor_fields)
        }
        row.update(
            {
                "timestamp_unix": base_timestamp + sample_index / 10.0,
                "timestamp_iso": (
                    f"diagnostic-window-{window_index:04d}-"
                    f"sample-{sample_index:03d}"
                ),
                "stream_frame_index": window_index * 100 + sample_index,
                "connection_id": 1,
                "frame_plausible": 1,
                "_recording_active": True,
                "_recording_session_id": session_id,
                "_model_input_precomputed": True,
            }
        )
        candidate = runtime.append_row(row)
        if candidate is not None:
            if result is not None:
                raise RuntimeError(
                    f"window {window_index} emitted more than one inference"
                )
            result = candidate
    if result is None:
        raise RuntimeError(
            f"window {window_index} emitted no inference; state={runtime.status()}"
        )
    payload = result.to_dict()
    if not payload["window_recording_complete"]:
        raise RuntimeError(f"window {window_index} lost recording provenance")
    if payload["recording_session_id"] != session_id:
        raise RuntimeError(f"window {window_index} recording identity differs")
    return payload


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {output}")
    features_path = Path(args.features).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    windows = np.load(features_path, allow_pickle=False)
    if (
        windows.ndim != 3
        or windows.shape[1:] != (100, 8)
        or not np.issubdtype(windows.dtype, np.number)
        or not np.isfinite(windows).all()
    ):
        raise ValueError("features must be a finite numeric N×100×8 array")
    with reference_path.open(newline="", encoding="utf-8") as handle:
        references = list(csv.DictReader(handle))
    if len(references) != len(windows):
        raise ValueError("reference row count differs from feature row count")
    row_count = len(windows) if args.limit <= 0 else min(args.limit, len(windows))
    if row_count < 1:
        raise ValueError("replay row count must be positive")

    runtime = EdgeAIRuntime(Path(args.package_dir))
    if runtime.package.model_backend != "gaps_runtime_v5":
        raise ValueError("replay package is not a gaps_runtime_v5 package")
    if runtime.package.feature_mode != "precomputed":
        raise ValueError("replay package does not declare precomputed inputs")
    raw_probe = {
        field: 0.0 for field in runtime.package.sensor_fields
    }
    raw_probe.update(
        {
            "timestamp_unix": 1_700_000_000.0,
            "frame_plausible": 1,
        }
    )
    try:
        runtime.append_row(raw_probe)
    except EdgeAIPackageError as exc:
        if "rejects raw STM32 serial frames" not in str(exc):
            raise
    else:
        raise RuntimeError("replay-only package accepted an unmarked raw frame")

    route_mismatch = 0
    qc_mismatch = 0
    auto_output_violation = 0
    max_differences = {
        "prediction_ppm": 0.0,
        "source_h1_ppm": 0.0,
        "max_probability": 0.0,
    }
    started = time.perf_counter()
    for index in range(row_count):
        result = stream_one_window(runtime, windows[index], index)
        reference = references[index]
        route_mismatch += int(
            int(result["predicted_class"]) != int(reference["pred_class"])
        )
        qc_mismatch += int(
            str(result["decision"]) != str(reference["qc_status"])
        )
        auto_output_violation += int(result["ppm_auto_output"] is not None)
        comparisons = {
            "prediction_ppm": (
                float(result["ppm_full_prediction"]),
                float(reference["prediction_ppm"]),
            ),
            "source_h1_ppm": (
                float(result["ppm_base_prediction"]),
                float(reference["source_h1_ppm"]),
            ),
            "max_probability": (
                float(result["confidence"]),
                float(reference["max_probability"]),
            ),
        }
        for field, (actual, expected) in comparisons.items():
            max_differences[field] = max(
                max_differences[field], abs(actual - expected)
            )
    elapsed = time.perf_counter() - started
    passed = (
        route_mismatch == 0
        and qc_mismatch == 0
        and auto_output_violation == 0
        and all(value <= 1e-6 for value in max_differences.values())
    )
    payload = {
        "schema_version": "gaps.edge_ui.runtime_v5_stream_replay.v1",
        "status": "PASS" if passed else "FAIL",
        "scope": (
            "diagnostic_public_calibration_stream_replay_not_formal_evidence_"
            "or_benchmark"
        ),
        "package_name": runtime.package.package_name,
        "package_fingerprint": runtime.package.package_fingerprint,
        "runtime_v5_release_id": runtime.package.runtime_v5_release_id,
        "row_count": row_count,
        "input_shape": list(windows[:row_count].shape),
        "route_mismatch_count": route_mismatch,
        "qc_status_mismatch_count": qc_mismatch,
        "auto_output_violation_count": auto_output_violation,
        "raw_serial_guard_verified": True,
        "max_abs_differences": max_differences,
        "elapsed_seconds_end_to_end_diagnostic": elapsed,
        "formal_test_opened": False,
        "classification_or_concentration_labels_opened": False,
        "training_or_fitting_performed": False,
        "formal_metrics_computed": False,
        "formal_benchmark_performed": False,
        "inputs": {
            "calibration_features": {
                "path": str(features_path),
                "bytes": features_path.stat().st_size,
                "sha256": sha256(features_path),
            },
            "frozen_calibration_parity_rows": {
                "path": str(reference_path),
                "bytes": reference_path.stat().st_size,
                "sha256": sha256(reference_path),
            },
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
