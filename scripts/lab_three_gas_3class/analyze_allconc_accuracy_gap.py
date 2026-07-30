"""Diagnose the laboratory all-concentration P2-to-P3 accuracy gap.

This is a read-only diagnostic over frozen datasets and reported result JSON.
It does not load GAPS checkpoints or replace the formal evaluation.  Simple
linear classifiers are used only as separability probes and are labeled as
recomputed diagnostics in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab-root",
        type=Path,
        default=root
        / "dataset"
        / "client_data_lab_3gas_allconc_timepurged_p2src_v1"
        / "fold_1",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=root
        / "dataset"
        / "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
    )
    parser.add_argument(
        "--lab-summary",
        type=Path,
        default=root
        / "results"
        / "lab_3gas_allconc_accuracy_gap_20260730"
        / "input"
        / "formal_evaluation_summary.json",
    )
    parser.add_argument(
        "--lab-run-config",
        type=Path,
        default=root
        / "results"
        / "lab_3gas_allconc_accuracy_gap_20260730"
        / "input"
        / "lab_run_config.json",
    )
    parser.add_argument(
        "--public-run-config",
        type=Path,
        default=root
        / "results"
        / "iotj_b2_b5_cross_direction_20260713"
        / "B2_proto_replay_corrected_server_da_f1_c1_to_c5_s42_r25"
        / "run_config.json",
    )
    parser.add_argument(
        "--public-metrics",
        type=Path,
        default=root
        / "results"
        / "iotj_b2_b5_cross_direction_20260713_f1_summary"
        / "B2_proto_replay_corrected_server_da_f1_c1_to_c5_s42_r25"
        / "classification_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "results"
        / "lab_3gas_allconc_accuracy_gap_20260730"
        / "diagnostics_v2.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(client_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(client_dir / f"{split}_features.npy")
    y = np.load(client_dir / f"{split}_classification_labels.npy").astype(int)
    return x.astype(np.float64), y


def response_descriptors(x: np.ndarray) -> np.ndarray:
    """Return mean/std/amplitude/slope for each channel."""

    channel_mean = x.mean(axis=1)
    channel_std = x.std(axis=1)
    channel_amplitude = x.max(axis=1) - x.min(axis=1)
    edge = max(1, min(10, x.shape[1] // 4))
    channel_slope = x[:, -edge:, :].mean(axis=1) - x[:, :edge, :].mean(axis=1)
    return np.concatenate(
        [channel_mean, channel_std, channel_amplitude, channel_slope],
        axis=1,
    )


def relative_resistance_to_conductance(
    x_z: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Approximate relative conductance from saved relative-resistance windows.

    If r=(R-R0)/R0, then (G-G0)/G0=R0/R-1=-r/(1+r).  The saved
    pipeline smoothed r before this conversion, so this is an analysis-only
    approximation rather than a replacement dataset build.
    """

    relative_resistance = x_z * std + mean
    denominator = np.maximum(1.0 + relative_resistance, 1e-4)
    return -relative_resistance / denominator


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=5000,
            random_state=42,
        ),
    )
    model.fit(response_descriptors(x_train), y_train)
    pred = model.predict(response_descriptors(x_test))
    return (
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "n_train_windows": int(len(y_train)),
            "n_test_windows": int(len(y_test)),
            "feature_contract": "per-channel mean/std/amplitude/edge-slope",
            "status": "recomputed_separability_probe_not_formal_gaps_metric",
        },
        pred,
    )


def accuracy_by_manifest_field(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    field: str,
) -> dict[str, dict[str, Any]]:
    if len(manifest) != len(y_true):
        raise ValueError(f"Manifest length mismatch for {field}")
    result: dict[str, dict[str, Any]] = {}
    values = manifest[field].astype(str).to_numpy()
    for value in sorted(set(values)):
        mask = values == value
        result[value] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
        }
    return result


def standardized_nearest_neighbor_distance(
    calibration: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    calibration_flat = calibration.reshape(len(calibration), -1)
    test_flat = test.reshape(len(test), -1)
    combined = np.concatenate([calibration_flat, test_flat], axis=0)
    scale = combined.std(axis=0)
    scale[scale < 1e-8] = 1.0
    center = combined.mean(axis=0)
    calibration_scaled = (calibration_flat - center) / scale
    test_scaled = (test_flat - center) / scale
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nearest.fit(calibration_scaled)
    distance = nearest.kneighbors(test_scaled, return_distance=True)[0][:, 0]
    normalized = distance / np.sqrt(calibration_scaled.shape[1])
    return {
        "mean_per_dimension": float(normalized.mean()),
        "median_per_dimension": float(np.median(normalized)),
        "p90_per_dimension": float(np.quantile(normalized, 0.9)),
    }


def filename_overlap(client_dir: Path) -> dict[str, Any]:
    calibration = json.loads(
        (client_dir / "calibration_experiment_info.json").read_text(
            encoding="utf-8"
        )
    )
    test = json.loads(
        (client_dir / "test_experiment_info.json").read_text(encoding="utf-8")
    )
    calibration_files = {str(item["filename"]) for item in calibration}
    test_files = {str(item["filename"]) for item in test}
    overlap = calibration_files & test_files
    return {
        "calibration_unique_files": len(calibration_files),
        "test_unique_files": len(test_files),
        "shared_unique_files": len(overlap),
        "calibration_file_coverage": len(overlap) / len(calibration_files),
        "test_file_coverage": len(overlap) / len(test_files),
    }


def class_domain_geometry(
    source: np.ndarray,
    source_y: np.ndarray,
    target: np.ndarray,
    target_y: np.ndarray,
) -> dict[str, Any]:
    source_desc = response_descriptors(source)
    target_desc = response_descriptors(target)
    scale = source_desc.std(axis=0)
    scale[scale < 1e-8] = 1.0
    center = source_desc.mean(axis=0)
    source_scaled = (source_desc - center) / scale
    target_scaled = (target_desc - center) / scale
    classes = sorted(set(source_y.tolist()) & set(target_y.tolist()))
    source_centroids = {
        cls: source_scaled[source_y == cls].mean(axis=0) for cls in classes
    }
    target_centroids = {
        cls: target_scaled[target_y == cls].mean(axis=0) for cls in classes
    }
    shifts = {
        str(cls): float(
            np.linalg.norm(target_centroids[cls] - source_centroids[cls])
            / np.sqrt(source_scaled.shape[1])
        )
        for cls in classes
    }
    target_pairwise = {}
    for index, left in enumerate(classes):
        for right in classes[index + 1 :]:
            target_pairwise[f"{left}-{right}"] = float(
                np.linalg.norm(target_centroids[left] - target_centroids[right])
                / np.sqrt(source_scaled.shape[1])
            )
    return {
        "source_to_target_class_centroid_shift_per_dimension": shifts,
        "target_pairwise_class_centroid_distance_per_dimension": target_pairwise,
        "descriptor": "per-channel mean/std/amplitude/edge-slope; scaled by P2 train",
    }


def channel_diagnostics(
    source: np.ndarray,
    source_y: np.ndarray,
    target: np.ndarray,
    target_y: np.ndarray,
    channel_ids: list[int],
) -> dict[str, Any]:
    source_means = source.mean(axis=1)
    target_means = target.mean(axis=1)
    result: dict[str, Any] = {}
    for index, channel_id in enumerate(channel_ids):
        class_source_means = {
            str(cls): float(source_means[source_y == cls, index].mean())
            for cls in sorted(np.unique(source_y).tolist())
        }
        class_target_means = {
            str(cls): float(target_means[target_y == cls, index].mean())
            for cls in sorted(np.unique(target_y).tolist())
        }
        target_class_centers = np.asarray(list(class_target_means.values()))
        between = float(np.var(target_class_centers))
        within_parts = [
            target_means[target_y == cls, index]
            for cls in sorted(np.unique(target_y).tolist())
        ]
        within = float(np.mean([np.var(values) for values in within_parts]))
        result[f"CH{channel_id}"] = {
            "p3_mean_in_p2_z_space": float(target_means[:, index].mean()),
            "p3_std_in_p2_z_space": float(target_means[:, index].std()),
            "p2_class_means": class_source_means,
            "p3_class_means": class_target_means,
            "class_specific_domain_shifts": {
                cls: class_target_means[cls] - class_source_means[cls]
                for cls in class_source_means
            },
            "p3_between_to_within_class_variance_ratio": (
                between / within if within > 0 else None
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    lab_root = args.lab_root.resolve()
    public_root = args.public_root.resolve()

    p2_train, p2_train_y = load_split(lab_root / "client_2", "train")
    p2_calibration, p2_calibration_y = load_split(
        lab_root / "client_2", "calibration"
    )
    p3_calibration, p3_calibration_y = load_split(
        lab_root / "client_3", "calibration"
    )
    p3_test, p3_test_y = load_split(lab_root / "client_3", "test")
    p3_manifest = pd.read_csv(
        lab_root / "client_3" / "test_window_manifest.csv",
        encoding="utf-8-sig",
    )
    norm = np.load(lab_root / "norm_stats.npz")
    norm_mean = norm["mean"].astype(np.float64)
    norm_std = norm["std"].astype(np.float64)

    probes: dict[str, Any] = {}
    probes["p2_train_to_p3_test_relative_resistance"], pred_source = fit_probe(
        p2_train,
        p2_train_y,
        p3_test,
        p3_test_y,
    )
    probes["p3_calibration_to_p3_test_relative_resistance"], pred_target = (
        fit_probe(
            p3_calibration,
            p3_calibration_y,
            p3_test,
            p3_test_y,
        )
    )
    combined_x = np.concatenate([p2_train, p3_calibration], axis=0)
    combined_y = np.concatenate([p2_train_y, p3_calibration_y], axis=0)
    probes[
        "p2_train_plus_p3_calibration_to_p3_test_relative_resistance"
    ], pred_combined = fit_probe(combined_x, combined_y, p3_test, p3_test_y)

    p2_train_g = relative_resistance_to_conductance(
        p2_train, norm_mean, norm_std
    )
    p3_calibration_g = relative_resistance_to_conductance(
        p3_calibration, norm_mean, norm_std
    )
    p3_test_g = relative_resistance_to_conductance(
        p3_test, norm_mean, norm_std
    )
    probes["p2_train_to_p3_test_relative_conductance_approx"], pred_source_g = (
        fit_probe(p2_train_g, p2_train_y, p3_test_g, p3_test_y)
    )
    probes[
        "p3_calibration_to_p3_test_relative_conductance_approx"
    ], pred_target_g = fit_probe(
        p3_calibration_g,
        p3_calibration_y,
        p3_test_g,
        p3_test_y,
    )
    probes[
        "p2_train_plus_p3_calibration_to_p3_test_relative_conductance_approx"
    ], pred_combined_g = fit_probe(
        np.concatenate([p2_train_g, p3_calibration_g], axis=0),
        combined_y,
        p3_test_g,
        p3_test_y,
    )

    channel_ids = [1, 2, 4, 6, 8, 9]
    probes["target_relative_conductance_leave_one_channel_out"] = {}
    probes["target_relative_conductance_single_channel"] = {}
    for channel_index, channel_id in enumerate(channel_ids):
        keep = [
            index
            for index in range(len(channel_ids))
            if index != channel_index
        ]
        leave_one_out, _ = fit_probe(
            p3_calibration_g[:, :, keep],
            p3_calibration_y,
            p3_test_g[:, :, keep],
            p3_test_y,
        )
        single_channel, _ = fit_probe(
            p3_calibration_g[:, :, [channel_index]],
            p3_calibration_y,
            p3_test_g[:, :, [channel_index]],
            p3_test_y,
        )
        probes["target_relative_conductance_leave_one_channel_out"][
            f"without_CH{channel_id}"
        ] = leave_one_out
        probes["target_relative_conductance_single_channel"][
            f"CH{channel_id}"
        ] = single_channel

    for key, pred in (
        ("source_relative_resistance", pred_source),
        ("target_relative_resistance", pred_target),
        ("combined_relative_resistance", pred_combined),
        ("source_relative_conductance_approx", pred_source_g),
        ("target_relative_conductance_approx", pred_target_g),
        ("combined_relative_conductance_approx", pred_combined_g),
    ):
        probes[key + "_by_base_window_index"] = accuracy_by_manifest_field(
            p3_manifest,
            p3_test_y,
            pred,
            "base_window_index",
        )
        probes[key + "_by_gas"] = accuracy_by_manifest_field(
            p3_manifest,
            p3_test_y,
            pred,
            "gas_en",
        )
        probes[key + "_by_version"] = accuracy_by_manifest_field(
            p3_manifest,
            p3_test_y,
            pred,
            "version",
        )

    p2_relative = p2_train * norm_std + norm_mean
    p3_relative = p3_test * norm_std + norm_mean

    public_c1_train, _ = load_split(public_root / "client_1", "train")
    public_c5_calibration, _ = load_split(
        public_root / "client_5", "calibration"
    )
    public_c5_test, _ = load_split(public_root / "client_5", "test")

    lab_summary = json.loads(args.lab_summary.read_text(encoding="utf-8"))
    lab_config = json.loads(args.lab_run_config.read_text(encoding="utf-8"))
    public_config = json.loads(
        args.public_run_config.read_text(encoding="utf-8")
    )
    public_metrics = json.loads(args.public_metrics.read_text(encoding="utf-8"))

    lab_args = lab_config["args"]
    public_args = public_config["args"]
    da_fields = [
        "profile",
        "domain_adapt_steps",
        "da_use_coral",
        "da_use_mmd",
        "da_use_adversarial",
        "da_mmd_objective",
        "da_stage_alignment",
        "da_adv_feature_objective",
        "da_lambda_coral",
        "da_lambda_global_mmd",
        "da_lambda_class_mmd",
        "da_lambda_adv",
        "da_lambda_proto_mmd",
        "da_lambda_stage_mmd",
        "da_server_opt_lr",
    ]

    payload = {
        "analysis_id": "LAB3GAS-ALLCONC-P2P3-ACCURACY-GAP-20260730",
        "status": "diagnostic_not_formal_experiment",
        "provenance": {
            "lab_summary": {
                "path": str(args.lab_summary.resolve()),
                "sha256": sha256(args.lab_summary),
            },
            "lab_run_config": {
                "path": str(args.lab_run_config.resolve()),
                "sha256": sha256(args.lab_run_config),
            },
            "public_run_config": {
                "path": str(args.public_run_config.resolve()),
                "sha256": sha256(args.public_run_config),
            },
            "public_metrics": {
                "path": str(args.public_metrics.resolve()),
                "sha256": sha256(args.public_metrics),
            },
        },
        "reported_metrics": {
            "lab": {
                "selected_round": int(lab_summary["selected_round"]),
                "adapted_test_window_accuracy": float(
                    lab_summary["final"]["adapted"]["target_test"]["global"][
                        "window"
                    ]["accuracy"]
                ),
                "adapted_test_window_macro_f1": float(
                    lab_summary["final"]["adapted"]["target_test"]["global"][
                        "window"
                    ]["macro_f1"]
                ),
                "adapted_test_exposure_accuracy": float(
                    lab_summary["final"]["adapted"]["target_test"]["global"][
                        "exposure"
                    ]["accuracy"]
                ),
                "test_windows": int(len(p3_test_y)),
                "test_exposures": int(p3_manifest["exposure_id"].nunique()),
            },
            "public_b2": {
                "selected_checkpoint_policy": "final_round_25",
                "round": int(public_metrics["round"]),
                "adapted_test_window_accuracy": float(
                    public_metrics["metrics"]["test"]["accuracy"]
                ),
                "adapted_test_window_macro_f1": float(
                    public_metrics["metrics"]["test"]["macro_f1"]
                ),
                "test_windows": int(public_metrics["metrics"]["test"]["N"]),
            },
        },
        "protocol_differences": {
            "lab": {
                "classes": 3,
                "channels": 6,
                "source_train_windows": int(len(p2_train)),
                "source_exposures": 30,
                "target_calibration_windows": int(len(p3_calibration)),
                "target_test_windows": int(len(p3_test)),
                "target_exposures": int(p3_manifest["exposure_id"].nunique()),
                "gas_interval_seconds_used": [0, 1200],
                "phase_labels": 1,
                "split_level": "raw-time-purged windows within each exposure",
                "calibration_test_raw_time_overlap": 0,
                "checkpoint_policy": "source calibration then earliest tie",
            },
            "public_b2": {
                "classes": 4,
                "channels": 8,
                "source_train_windows": int(len(public_c1_train)),
                "source_unique_files": 160,
                "target_calibration_windows": int(len(public_c5_calibration)),
                "target_test_windows": int(len(public_c5_test)),
                "response_window_protocol_seconds": [60, 170],
                "phase_labels": 3,
                "split_level": "window-level class-concentration stratified",
                "target_filename_overlap": filename_overlap(
                    public_root / "client_5"
                ),
                "checkpoint_policy": "final adapted round 25",
            },
        },
        "da_configuration_differences": {
            field: {
                "lab": lab_args.get(field, "unknown"),
                "public_b2": public_args.get(field, "unknown"),
                "same": lab_args.get(field) == public_args.get(field),
            }
            for field in da_fields
        },
        "lab_distribution": {
            "p3_test_channel_mean_in_p2_z_space": p3_test.mean(
                axis=(0, 1)
            ).tolist(),
            "p3_test_channel_std_in_p2_z_space": p3_test.std(
                axis=(0, 1)
            ).tolist(),
            "p2_relative_resistance_fraction_below_minus_0_8": float(
                np.mean(p2_relative < -0.8)
            ),
            "p2_relative_resistance_fraction_below_minus_0_9": float(
                np.mean(p2_relative < -0.9)
            ),
            "p3_relative_resistance_fraction_below_minus_0_8": float(
                np.mean(p3_relative < -0.8)
            ),
            "p3_relative_resistance_fraction_below_minus_0_9": float(
                np.mean(p3_relative < -0.9)
            ),
            "class_domain_geometry": class_domain_geometry(
                p2_train,
                p2_train_y,
                np.concatenate([p3_calibration, p3_test], axis=0),
                np.concatenate([p3_calibration_y, p3_test_y], axis=0),
            ),
            "channel_diagnostics": channel_diagnostics(
                p2_train,
                p2_train_y,
                np.concatenate([p3_calibration, p3_test], axis=0),
                np.concatenate([p3_calibration_y, p3_test_y], axis=0),
                channel_ids,
            ),
        },
        "calibration_test_similarity": {
            "lab_time_purged": standardized_nearest_neighbor_distance(
                p3_calibration, p3_test
            ),
            "public_window_random": standardized_nearest_neighbor_distance(
                public_c5_calibration, public_c5_test
            ),
            "interpretation": (
                "Lower distance means test windows are closer to a calibration "
                "window in standardized flattened feature space."
            ),
        },
        "separability_probes": probes,
        "limitations": [
            "One seed and one source-target direction.",
            "Repeated windows are not independent experimental units.",
            "Linear probe results are diagnostic and are not GAPS results.",
            (
                "Relative-conductance probes convert already smoothed relative "
                "resistance; a formal comparison must rebuild from raw R before "
                "smoothing."
            ),
            (
                "The formal GAPS summary contains aggregate confusion matrices "
                "but no per-window prediction stream, so GAPS error localization "
                "by time/concentration is currently unavailable."
            ),
            "Nominal gas boundaries have not been replaced by detected onsets.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
