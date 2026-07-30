"""Validate the A4 classification package through the UI runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = REPO_ROOT / "raspi_stm32_edge_ai_ui_v2_0"
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from edge_ai_runtime import EdgeAIRuntime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_row(
    values: np.ndarray,
    fields: list[str],
    timestamp: float,
    frame_index: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp_unix": float(timestamp),
        "timestamp_iso": f"replay-{timestamp:.3f}",
        "stream_frame_index": int(frame_index),
        "connection_id": 1,
        "frame_plausible": 1,
        "_model_input_precomputed": True,
        "_recording_active": False,
        "_recording_session_id": "",
    }
    row.update({field: float(value) for field, value in zip(fields, values)})
    return row


def replay_one_window(
    runtime: EdgeAIRuntime,
    unnormalized_window: np.ndarray,
    base_timestamp: float,
    frame_offset: int,
) -> dict[str, Any]:
    runtime.reset_stream(keep_baseline=True)
    runtime.set_experiment_phase("exposure")
    result = None
    for index, values in enumerate(unnormalized_window):
        result = runtime.append_row(
            make_row(
                values,
                runtime.package.sensor_fields,
                base_timestamp + index,
                frame_offset + index,
            )
        )
    if result is None:
        raise RuntimeError("runtime produced no result for a complete 100-sample window")
    return result.to_dict()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")

    features = np.load(features_path).astype(np.float32, copy=False)
    labels = np.load(labels_path).astype(np.int64, copy=False).reshape(-1)
    if features.shape[1:] != (100, 6) or labels.shape != (len(features),):
        raise ValueError(f"unexpected validation shapes: {features.shape}/{labels.shape}")

    runtime = EdgeAIRuntime(package_dir)
    if runtime.package.task_type != "classification":
        raise ValueError("package is not classification-only")
    if runtime.package.has_concentration:
        raise ValueError("classification-only package unexpectedly has concentration output")
    mean = np.broadcast_to(runtime.package.mean, (100, 6)).astype(np.float32)
    std = np.broadcast_to(runtime.package.std, (100, 6)).astype(np.float32)
    unnormalized = (features * std + mean).astype(np.float32)
    runtime_normalized = ((unnormalized - mean) / std).astype(np.float32)
    normalization_roundtrip_max_abs_diff = float(
        np.max(
            np.abs(
                runtime_normalized.astype(np.float64)
                - features.astype(np.float64)
            )
        )
    )

    direct_model = torch.jit.load(str(runtime.package.model_path), map_location="cpu")
    direct_model.eval()
    direct_probability_rows: list[np.ndarray] = []
    with torch.no_grad():
        for window in runtime_normalized:
            direct_logits = direct_model(torch.from_numpy(window[None, :, :]))
            direct_probability_rows.append(
                torch.softmax(direct_logits, dim=1).cpu().numpy()[0]
            )
    direct_probs = np.stack(direct_probability_rows, axis=0)
    direct_pred = np.argmax(direct_probs, axis=1)

    runtime_pred: list[int] = []
    runtime_probs: list[list[float]] = []
    latency_ms: list[float] = []
    for index, window in enumerate(unnormalized):
        result = replay_one_window(
            runtime,
            window,
            base_timestamp=float(index * 200),
            frame_offset=index * 100,
        )
        if result["ppm_base_prediction"] is not None:
            raise RuntimeError("classification replay emitted ppm_base_prediction")
        if result["ppm_full_prediction"] is not None:
            raise RuntimeError("classification replay emitted ppm_full_prediction")
        if result["ppm_auto_output"] is not None:
            raise RuntimeError("classification replay emitted ppm_auto_output")
        if result["decision"] != "unavailable_qc_not_validated":
            raise RuntimeError(f"unexpected classification QC decision: {result['decision']}")
        runtime_pred.append(int(result["predicted_class"]))
        runtime_probs.append([float(x) for x in result["class_probabilities"]])
        latency_ms.append(float(result["inference_latency_ms"]))

    runtime_pred_array = np.asarray(runtime_pred, dtype=np.int64)
    runtime_prob_array = np.asarray(runtime_probs, dtype=np.float64)
    max_probability_abs_diff = float(
        np.max(np.abs(runtime_prob_array - direct_probs.astype(np.float64)))
    )

    # Independent two-window consensus audit.  The second 100-sample runtime
    # window contains the last 50 samples of window 0 followed by its first 50.
    runtime.reset_stream(keep_baseline=True)
    runtime.set_experiment_phase("exposure")
    first_result = None
    for index, values in enumerate(unnormalized[0]):
        first_result = runtime.append_row(
            make_row(values, runtime.package.sensor_fields, float(index), index)
        )
    if first_result is None:
        raise RuntimeError("consensus audit first window missing")
    second_result = None
    for offset, values in enumerate(unnormalized[0, :50], start=100):
        second_result = runtime.append_row(
            make_row(values, runtime.package.sensor_fields, float(offset), offset)
        )
    if second_result is None:
        raise RuntimeError("consensus audit second window missing")
    second_normalized = np.concatenate(
        [runtime_normalized[0, 50:], runtime_normalized[0, :50]], axis=0
    )[None, :, :]
    with torch.no_grad():
        second_probs = torch.softmax(
            direct_model(torch.from_numpy(second_normalized)), dim=1
        ).cpu().numpy()[0]
    expected_consensus = (
        direct_probs[0].astype(np.float64) + second_probs.astype(np.float64)
    ) / 2.0
    actual_consensus = np.asarray(
        second_result.consensus_probabilities, dtype=np.float64
    )
    consensus_max_abs_diff = float(
        np.max(np.abs(actual_consensus - expected_consensus))
    )

    summary = {
        "experiment_id": "A4-EDGE-P2A",
        "package_dir": str(package_dir),
        "package_fingerprint": runtime.package.package_fingerprint,
        "model_sha256": sha256(runtime.package.model_path),
        "norm_stats_sha256": sha256(runtime.package.norm_path),
        "features_sha256": sha256(features_path),
        "labels_sha256": sha256(labels_path),
        "n_windows": int(len(features)),
        "direct_correct": int(np.sum(direct_pred == labels)),
        "runtime_correct": int(np.sum(runtime_pred_array == labels)),
        "runtime_accuracy": float(np.mean(runtime_pred_array == labels)),
        "direct_runtime_class_match_count": int(
            np.sum(direct_pred == runtime_pred_array)
        ),
        "direct_runtime_class_match_rate": float(
            np.mean(direct_pred == runtime_pred_array)
        ),
        "max_probability_abs_diff": max_probability_abs_diff,
        "normalization_roundtrip_max_abs_diff": (
            normalization_roundtrip_max_abs_diff
        ),
        "consensus_window_count": int(second_result.consensus_window_count),
        "consensus_max_abs_diff": consensus_max_abs_diff,
        "latency_ms_median": float(np.median(latency_ms)),
        "latency_ms_p95": float(np.percentile(latency_ms, 95)),
        "classification_only_ppm_null": True,
        "classification_qc_unavailable": True,
    }
    if summary["direct_runtime_class_match_rate"] != 1.0:
        raise RuntimeError(f"runtime class parity failed: {summary}")
    if max_probability_abs_diff > 1e-5:
        raise RuntimeError(f"runtime probability parity failed: {summary}")
    if consensus_max_abs_diff > 1e-6:
        raise RuntimeError(f"consensus parity failed: {summary}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
