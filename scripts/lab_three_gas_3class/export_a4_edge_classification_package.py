"""Export the frozen REC-A4 round-25 classifier as a replay-safe UI package.

The package accepts precomputed six-channel relative-resistance features.  It
intentionally rejects raw STM32 frames until the physical ADC-to-resistance
mapping is frozen and independently verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.evaluate_checkpoint import load_checkpoint_model


GAS_NAMES = ["乙醛", "甲烷", "乙酸"]
SENSOR_FIELDS = [
    "lab_ch1_relative_resistance_smoothed",
    "lab_ch2_relative_resistance_smoothed",
    "lab_ch4_relative_resistance_smoothed",
    "lab_ch6_relative_resistance_smoothed",
    "lab_ch8_relative_resistance_smoothed",
    "lab_ch9_relative_resistance_smoothed",
]


class ClassificationOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(x)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package-name",
        default="lab_3gas_a4_stable_p2_to_p3_round25_replay_candidate",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-commit", default="")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def validate_inputs(
    checkpoint: Path,
    norm_stats: Path,
    features_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    for path in (checkpoint, norm_stats, features_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    features = np.load(features_path).astype(np.float32, copy=False)
    labels = np.load(labels_path).astype(np.int64, copy=False).reshape(-1)
    if features.ndim != 3 or features.shape[1:] != (100, 6):
        raise ValueError(
            f"validation features must have shape [N,100,6], got {features.shape}"
        )
    if labels.shape != (features.shape[0],):
        raise ValueError(
            f"validation labels must have shape ({features.shape[0]},), got {labels.shape}"
        )
    norm = np.load(norm_stats)
    if set(norm.files) != {"mean", "std", "selected_channels"}:
        raise ValueError(f"unexpected norm_stats keys: {norm.files}")
    if norm["mean"].shape != (1, 1, 6) or norm["std"].shape != (1, 1, 6):
        raise ValueError(
            f"norm mean/std must be (1,1,6), got {norm['mean'].shape}/{norm['std'].shape}"
        )
    if norm["selected_channels"].tolist() != [1, 2, 4, 6, 8, 9]:
        raise ValueError(
            f"unexpected selected channels: {norm['selected_channels'].tolist()}"
        )
    return features, labels


def compare_models(
    scripted: torch.jit.ScriptModule,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    reference_logits: np.ndarray,
) -> dict[str, Any]:
    script_logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size])
            script_logits.append(scripted(x).cpu().numpy())
    logits_a = np.asarray(reference_logits, dtype=np.float64)
    logits_b = np.concatenate(script_logits, axis=0).astype(np.float64)
    probs_a = torch.softmax(torch.from_numpy(logits_a), dim=1).numpy()
    probs_b = torch.softmax(torch.from_numpy(logits_b), dim=1).numpy()
    pred_a = np.argmax(logits_a, axis=1)
    pred_b = np.argmax(logits_b, axis=1)
    return {
        "n_samples": int(len(features)),
        "eager_correct": int(np.sum(pred_a == labels)),
        "torchscript_correct": int(np.sum(pred_b == labels)),
        "eager_accuracy": float(np.mean(pred_a == labels)),
        "torchscript_accuracy": float(np.mean(pred_b == labels)),
        "class_match_count": int(np.sum(pred_a == pred_b)),
        "class_match_rate": float(np.mean(pred_a == pred_b)),
        "max_logit_abs_diff": float(np.max(np.abs(logits_a - logits_b))),
        "max_probability_abs_diff": float(np.max(np.abs(probs_a - probs_b))),
    }


def collect_logits(
    model: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size])
            rows.append(model(x).cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    features_path = args.validation_features.expanduser().resolve()
    labels_path = args.validation_labels.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing deployment package: {output_dir}"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    features, labels = validate_inputs(
        checkpoint, norm_stats, features_path, labels_path
    )
    model, _, checkpoint_payload = load_checkpoint_model(
        str(checkpoint), torch.device("cpu"), args.batch_size
    )
    model_config = checkpoint_payload.get("model_config") or {}
    expected_config = {
        "num_classes": 3,
        "input_dim": 6,
        "seq_len": 100,
        "profile": "proto_replay",
    }
    for key, expected in expected_config.items():
        if model_config.get(key) != expected:
            raise ValueError(
                f"checkpoint model_config.{key}={model_config.get(key)!r}, "
                f"expected {expected!r}"
            )
    if int(checkpoint_payload.get("round", -1)) != 25:
        raise ValueError("REC-A4 deployment package requires round 25")

    wrapper = ClassificationOnlyWrapper(model).eval()
    reference_logits = collect_logits(wrapper, features, args.batch_size)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    example = torch.from_numpy(features[:1])
    # Disable the native MHA fastpath before tracing.  Its fused operator is
    # numerically valid in synchronous replay but crashes when invoked from the
    # Qt worker thread on both Windows and Raspberry Pi.  The standard graph is
    # checked against the original fastpath reference on every frozen window.
    scripted = torch.jit.trace(wrapper, example, check_trace=False)
    parity = compare_models(
        scripted,
        features,
        labels,
        args.batch_size,
        reference_logits,
    )
    if parity["class_match_rate"] != 1.0:
        raise RuntimeError(f"class parity failed: {parity}")
    if parity["max_logit_abs_diff"] > 1e-5:
        raise RuntimeError(f"logit parity failed: {parity}")
    if parity["max_probability_abs_diff"] > 1e-6:
        raise RuntimeError(f"probability parity failed: {parity}")

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.ts"
    norm_path = output_dir / "norm_stats.npz"
    scripted.save(str(model_path))
    shutil.copy2(norm_stats, norm_path)

    source_commit = str(args.source_commit or current_commit())
    parity.update(
        {
            "checkpoint_sha256": sha256(checkpoint),
            "norm_stats_sha256": sha256(norm_path),
            "model_sha256": sha256(model_path),
            "validation_features_sha256": sha256(features_path),
            "validation_labels_sha256": sha256(labels_path),
            "source_commit": source_commit,
        }
    )
    (output_dir / "parity_summary.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "protocol": "REC-A4-STABLE150",
        "target": "P3",
        "round": 25,
        "stable_test_correct": 359,
        "stable_test_total": 360,
        "stable_test_accuracy": 359 / 360,
        "full_scope_denominator": 420,
        "coverage": 360 / 420,
        "claim_boundary": (
            "99.72% applies only to the 360-window stable test scope; "
            "it is not a full-time accuracy claim."
        ),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 4,
        "package_name": str(args.package_name),
        "dataset_profile": "lab_3gas_P2src_P3tgt_A4_stable_1hz",
        "device_profile": "replay_only_pending_stm32_resistance_contract",
        "model_backend": "torchscript",
        "model_file": "model.ts",
        "input": {
            "sensor_fields": SENSOR_FIELDS,
            "target_sample_hz": 1.0,
            "raw_sample_hz": 1.0,
            "window_size": 100,
            "stride": 50,
            "feature_mode": "precomputed",
            "max_gap_s": 1.5,
            "min_rate_ratio": 0.95,
            "max_rate_ratio": 1.05,
            "reject_implausible_frames": True,
        },
        "phase_control": {
            "mode": "event_driven",
            "inference_phases": ["exposure"],
            "stable_window_start_offset_s": 250.0,
            "first_result_nominal_offset_s": 350.0,
        },
        "normalization": {
            "enabled": True,
            "file": "norm_stats.npz",
            "mean_key": "mean",
            "std_key": "std",
            "scope": "P2_train_only",
        },
        "output": {
            "task_type": "classification",
            "has_concentration": False,
            "gas_names": GAS_NAMES,
            "probability_aggregation": "exposure_running_arithmetic_mean",
        },
        "qc": {
            "enabled": False,
            "policy_name": "classification_qc_not_validated",
        },
        "preprocessing_contract": {
            "physical_input": "resistance_ohm",
            "selected_channels": [1, 2, 4, 6, 8, 9],
            "resample": "strict_1hz_linear_interpolation",
            "baseline": "per_channel_median_pre_exposure_300s",
            "transform": "(R-R0)/abs(R0)",
            "smoothing": "centered_reflect_moving_average_5s",
            "window": "100s",
            "stride": "50s",
            "online_centered_filter_delay_s": 2.0,
            "adapter_state": "not_included_in_replay_candidate",
        },
        "deployment_scope": {
            "replay_only": True,
            "raw_stm32_frames_allowed": False,
            "live_blocker": (
                "CH1/2/4/6/8/9 frame mapping and six-channel "
                "ADC-to-resistance electrical contract are unverified"
            ),
        },
        "provenance": {
            "experiment_id": "A4-EDGE-P1A",
            "training_run": checkpoint_payload.get("run_name"),
            "checkpoint_round": int(checkpoint_payload.get("round")),
            "checkpoint_sha256": parity["checkpoint_sha256"],
            "source_commit": source_commit,
            "torchscript_mha_fastpath": "disabled_for_qt_worker_portability",
        },
        "integrity": {
            "model_sha256": parity["model_sha256"],
            "normalization_sha256": parity["norm_stats_sha256"],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = """# REC-A4 classification-only replay candidate

This package validates the laboratory three-gas A4 model inside the UI v2.2
runtime.  It accepts only explicitly marked, precomputed relative-resistance
features and intentionally rejects raw STM32 serial frames.

There is no concentration output.  `ppm_*` values remain null and QC remains
Unavailable because no target-validated classification QC policy exists.

The reported 359/360 (99.72%) accuracy covers 360/420 (85.71%) stable windows.
Do not present it as full-time accuracy.  Live serial inference remains blocked
until the six-channel ADC-to-resistance and channel-mapping contract is frozen.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **parity}, ensure_ascii=False))


if __name__ == "__main__":
    main()
