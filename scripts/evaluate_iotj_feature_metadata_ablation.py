"""Five-seed target-feature metadata ablation for the IoT-J GAPS study.

The evaluator reuses the canonical B5 routes and frozen federated H1 source
head.  All target-Ridge alpha choices are persisted from C5 calibration before
the C5 test split is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_iotj_b5_regression_multiseed as base


SCHEMA_VERSION = "iotj.feature_metadata_ablation.v1"

GLOBAL_SENSOR_KEYS = {
    "global_mean",
    "global_std",
    "global_min",
    "global_max",
    "global_amp",
    "global_absdiff_mean",
    "global_absdiff_max",
    "slope_mean",
    "slope_std",
    "amp_mean",
    "amp_std",
    "amp_top1",
    "amp_top2",
    "amp_top3",
    "amp_top4",
    "amp_top1_top2_ratio",
    "amp_top1_top3_ratio",
    "amp_top1_top4_ratio",
    "slope_top1_top2_ratio",
}

ONLINE_SAFE_KEYS = {
    "window_start_s",
    "window_end_s",
    "window_center_s",
    "window_len_s",
    "t_onset",
    "center_minus_onset",
    "interpolated_ratio",
    "max_gap_inside_window",
}

EXPECTED_EXCLUDED_FROM_SAFE = {
    "t_min",
    "center_minus_t_min",
    "response_phase_main_response",
    "response_phase_recovery",
    "response_phase_unknown",
    "phase_label_early",
    "phase_label_middle",
    "phase_label_late",
    "phase_label_unknown",
    "phase_id_0",
    "phase_id_1",
    "phase_id_2",
    "phase_id_unknown",
}

PROFILE_DIMENSIONS = {
    "M83_SENSOR": 83,
    "M91_ONLINE_SAFE": 91,
    "M104_FULL": 104,
}

VARIANTS = {
    "M83_SENSOR__NO_H1": ("M83_SENSOR", False),
    "M83_SENSOR__H1": ("M83_SENSOR", True),
    "M91_ONLINE_SAFE__NO_H1": ("M91_ONLINE_SAFE", False),
    "M91_ONLINE_SAFE__H1": ("M91_ONLINE_SAFE", True),
    "M104_FULL__NO_H1": ("M104_FULL", False),
    "M104_FULL__H1": ("M104_FULL", True),
}


def sensor_keys(feature_dict: Mapping[str, Any]) -> set[str]:
    return {
        key
        for key in feature_dict
        if key.startswith("ch") or key in GLOBAL_SENSOR_KEYS
    }


def validate_feature_schema(feature_dict: Mapping[str, Any]) -> dict[str, list[str]]:
    full = set(feature_dict)
    sensor = sensor_keys(feature_dict)
    metadata = full - sensor
    if len(full) != 104 or len(sensor) != 83 or len(metadata) != 21:
        raise RuntimeError(
            f"feature schema differs: full={len(full)} sensor={len(sensor)} "
            f"metadata={len(metadata)}"
        )
    if not ONLINE_SAFE_KEYS <= metadata:
        raise RuntimeError(
            f"online-safe fields missing: {sorted(ONLINE_SAFE_KEYS - metadata)}"
        )
    if metadata - ONLINE_SAFE_KEYS != EXPECTED_EXCLUDED_FROM_SAFE:
        raise RuntimeError(
            "metadata partition differs: "
            f"observed_excluded={sorted(metadata - ONLINE_SAFE_KEYS)}"
        )
    return {
        "sensor_keys": sorted(sensor),
        "metadata_keys": sorted(metadata),
        "online_safe_keys": sorted(ONLINE_SAFE_KEYS),
        "excluded_from_safe": sorted(EXPECTED_EXCLUDED_FROM_SAFE),
    }


def profile_feature_dict(
    feature_dict: Mapping[str, Any], profile: str
) -> dict[str, float]:
    schema = validate_feature_schema(feature_dict)
    if profile == "M83_SENSOR":
        keys = schema["sensor_keys"]
    elif profile == "M91_ONLINE_SAFE":
        keys = sorted(set(schema["sensor_keys"]) | ONLINE_SAFE_KEYS)
    elif profile == "M104_FULL":
        keys = sorted(feature_dict)
    else:
        raise ValueError(f"unknown feature profile: {profile}")
    output = {key: float(feature_dict[key]) for key in keys}
    if len(output) != PROFILE_DIMENSIONS[profile]:
        raise RuntimeError(f"{profile} dimension differs: {len(output)}")
    return output


def add_variant_features(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> list[dict[str, Any]]:
    profile, use_h1 = VARIANTS[variant]
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        features = profile_feature_dict(item["feature_dict"], profile)
        if use_h1:
            features["srcpred_H1_federated_source_ridge_ppm"] = float(
                item[base.PRIOR_KEYS[0]]
            )
        item["feature_dict"] = features
        output.append(item)
    expected = PROFILE_DIMENSIONS[profile] + int(use_h1)
    if any(len(row["feature_dict"]) != expected for row in output):
        raise RuntimeError(f"{variant} target input dimension differs")
    return output


def fit_seed_calibration(
    seed: int,
    oracle: Sequence[Mapping[str, Any]],
    deployment: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[int, Any]], list[dict[str, Any]]]:
    models: dict[str, dict[int, Any]] = {}
    selection: list[dict[str, Any]] = []
    for variant, (profile, use_h1) in VARIANTS.items():
        fit_features = add_variant_features(oracle, variant)
        deploy_features = add_variant_features(deployment, variant)
        deploy_by_id = {
            int(row["sample_index"]): row for row in deploy_features
        }
        names = sorted(fit_features[0]["feature_dict"])
        models[variant] = {}
        for class_id, gas in sorted(base.CLASS_NAMES.items()):
            class_rows = [
                row
                for row in fit_features
                if int(row["true_class"]) == class_id
            ]
            fit_rows, validation_seed_rows = base.deterministic_train_val(
                class_rows, 0.25
            )
            validation_rows = [
                deploy_by_id[int(row["sample_index"])]
                for row in validation_seed_rows
            ]
            if (len(class_rows), len(fit_rows), len(validation_rows)) != (80, 60, 20):
                raise RuntimeError(f"seed{seed} {variant} {gas} split differs")
            true = np.asarray([float(row["true_ppm"]) for row in validation_rows])
            best_alpha = base.RIDGE_ALPHAS[0]
            best_rmse = float("inf")
            grid: list[dict[str, float]] = []
            for alpha in base.RIDGE_ALPHAS:
                candidate = base.fit_ridge(fit_rows, names, alpha)
                pred = candidate.predict(validation_rows)
                score = float(np.sqrt(np.mean((pred - true) ** 2)))
                grid.append({"alpha": alpha, "validation_RMSE": score})
                if score < best_rmse:
                    best_alpha, best_rmse = alpha, score
            models[variant][class_id] = base.fit_ridge(
                class_rows, names, best_alpha
            )
            selection.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "profile": profile,
                    "uses_H1": int(use_h1),
                    "class_id": class_id,
                    "gas": gas,
                    "calibration_fit_N": len(fit_rows),
                    "calibration_validation_N": len(validation_rows),
                    "target_input_dimension": len(names),
                    "selected_alpha": best_alpha,
                    "calibration_validation_RMSE": best_rmse,
                    "alpha_grid_audit": json.dumps(grid, separators=(",", ":")),
                    "selection_split": "C5_calibration_internal_validation",
                }
            )
    return models, selection


def apply_models(
    deployment: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[int, Any]],
) -> list[dict[str, Any]]:
    output = [
        {
            "sample_index": int(row["sample_index"]),
            "true_class": int(row["true_class"]),
            "true_ppm": float(row["true_ppm"]),
            "pred_class": int(row["pred_class"]),
        }
        for row in deployment
    ]
    for variant in VARIANTS:
        features = add_variant_features(deployment, variant)
        for class_id in base.CLASS_NAMES:
            indexes = [
                idx
                for idx, row in enumerate(features)
                if int(row["route_class"]) == class_id
            ]
            pred = models[variant][class_id].predict(
                [features[idx] for idx in indexes]
            )
            for idx, value in zip(indexes, pred):
                output[idx][f"{variant}_ppm"] = float(value)
    values = np.asarray(
        [[row[f"{variant}_ppm"] for variant in VARIANTS] for row in output]
    )
    if values.shape != (len(output), len(VARIANTS)) or not np.isfinite(values).all():
        raise RuntimeError("metadata-ablation predictions are incomplete/non-finite")
    return output


def metric_block(
    rows: Sequence[Mapping[str, Any]], variant: str, mask: np.ndarray
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    true = np.asarray([float(row["true_ppm"]) for row in rows])
    pred = np.asarray([float(row[f"{variant}_ppm"]) for row in rows])
    classes = np.asarray([int(row["true_class"]) for row in rows])
    ranges = np.asarray([base.CLASS_RANGES[int(value)] for value in classes])
    error = pred - true
    if not selected.any():
        raise RuntimeError("metric mask is empty")
    return {
        "N": int(selected.sum()),
        "RMSE": float(np.sqrt(np.mean(error[selected] ** 2))),
        "MAE": float(np.mean(np.abs(error[selected]))),
        "NRMSE": float(np.sqrt(np.mean((error[selected] / ranges[selected]) ** 2))),
    }


def per_seed_metrics(
    seed: int, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classes = np.asarray([int(row["true_class"]) for row in rows])
    route = np.asarray([int(row["pred_class"]) for row in rows])
    masks = {
        "S_ALL": np.ones(len(rows), dtype=bool),
        "S_CC": classes == route,
    }
    overall: list[dict[str, Any]] = []
    gases: list[dict[str, Any]] = []
    for variant, (profile, use_h1) in VARIANTS.items():
        blocks = {name: metric_block(rows, variant, mask) for name, mask in masks.items()}
        overall.append(
            {
                "seed": seed,
                "variant": variant,
                "profile": profile,
                "uses_H1": int(use_h1),
                "input_dimension": PROFILE_DIMENSIONS[profile] + int(use_h1),
                "S_ALL_N": blocks["S_ALL"]["N"],
                "S_ALL_RMSE": blocks["S_ALL"]["RMSE"],
                "S_ALL_MAE": blocks["S_ALL"]["MAE"],
                "S_ALL_NRMSE": blocks["S_ALL"]["NRMSE"],
                "S_CC_N": blocks["S_CC"]["N"],
                "S_CC_RMSE": blocks["S_CC"]["RMSE"],
                "S_CC_MAE": blocks["S_CC"]["MAE"],
                "S_CC_NRMSE": blocks["S_CC"]["NRMSE"],
                "route_error_count": int((classes != route).sum()),
            }
        )
        for class_id, gas in sorted(base.CLASS_NAMES.items()):
            block = metric_block(rows, variant, classes == class_id)
            gases.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "profile": profile,
                    "uses_H1": int(use_h1),
                    "class_id": class_id,
                    "gas": gas,
                    **block,
                }
            )
    return overall, gases


def paired_comparisons(
    per_seed: Sequence[Mapping[str, Any]],
    per_gas: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {(int(row["seed"]), str(row["variant"])): row for row in per_seed}
    comparisons = {
        "M83_H1_minus_M104_H1": ("M83_SENSOR__H1", "M104_FULL__H1"),
        "M91_H1_minus_M104_H1": ("M91_ONLINE_SAFE__H1", "M104_FULL__H1"),
        "H1_effect_M83": ("M83_SENSOR__H1", "M83_SENSOR__NO_H1"),
        "H1_effect_M91": ("M91_ONLINE_SAFE__H1", "M91_ONLINE_SAFE__NO_H1"),
        "H1_effect_M104": ("M104_FULL__H1", "M104_FULL__NO_H1"),
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, (left, right) in comparisons.items():
        deltas = []
        relative = []
        for seed in base.SEEDS:
            lrow, rrow = lookup[(seed, left)], lookup[(seed, right)]
            delta = float(lrow["S_CC_NRMSE"]) - float(rrow["S_CC_NRMSE"])
            rel = delta / float(rrow["S_CC_NRMSE"])
            rows.append(
                {
                    "comparison": name,
                    "seed": seed,
                    "left_variant": left,
                    "right_variant": right,
                    "delta_S_CC_NRMSE": delta,
                    "relative_delta_S_CC_NRMSE": rel,
                }
            )
            deltas.append(delta)
            relative.append(rel)
        values = np.asarray(deltas)
        rel_values = np.asarray(relative)
        std = float(values.std(ddof=1))
        half = 2.7764451051977987 * std / math.sqrt(len(values))
        summary[name] = {
            "mean_delta_S_CC_NRMSE": float(values.mean()),
            "mean_relative_delta_S_CC_NRMSE": float(rel_values.mean()),
            "sample_std_delta": std,
            "descriptive_t95_CI": [float(values.mean() - half), float(values.mean() + half)],
            "left_wins": int((values < 0).sum()),
            "right_wins": int((values > 0).sum()),
        }

    gas_lookup = {
        (int(row["seed"]), str(row["variant"]), str(row["gas"])): float(row["NRMSE"])
        for row in per_gas
    }
    persistent: dict[str, int] = {}
    for gas in base.CLASS_NAMES.values():
        count = sum(
            (
                gas_lookup[(seed, "M91_ONLINE_SAFE__H1", gas)]
                - gas_lookup[(seed, "M104_FULL__H1", gas)]
            )
            / gas_lookup[(seed, "M104_FULL__H1", gas)]
            > 0.10
            for seed in base.SEEDS
        )
        persistent[gas] = int(count)
    safe = summary["M91_H1_minus_M104_H1"]
    summary["H-FMETA-02_gate"] = {
        "mean_relative_degradation_le_5pct": (
            float(safe["mean_relative_delta_S_CC_NRMSE"]) <= 0.05
        ),
        "no_gas_gt_10pct_worse_in_at_least_3_seeds": max(persistent.values()) < 3,
        "gas_gt_10pct_worse_seed_counts": persistent,
    }
    summary["H-FMETA-02_gate"]["all_pass"] = all(
        value
        for key, value in summary["H-FMETA-02_gate"].items()
        if key != "gas_gt_10pct_worse_seed_counts"
    )
    return rows, summary


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    output = Path(args.output_dir)
    base.require_empty_output(output)
    frozen_before = base.frozen_hashes(root)
    inventory = base.checkpoint_inventory(Path(args.multiseed_root))
    runtime = base.C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    h1, source_manifest = base.load_source_heads(Path(args.h1_manifest), runtime)
    device = torch.device(args.device)

    # Validate the frozen feature partition before any calibration model is fit.
    schema_row = base.build_oracle_rows(Path(args.data_root), ["C5"], "calibration")[0]
    feature_schema = validate_feature_schema(schema_row["feature_dict"])
    base.write_json(output / "feature_schema_lock.json", {
        "schema_version": SCHEMA_VERSION,
        "profile_dimensions": PROFILE_DIMENSIONS,
        "variant_dimensions": {
            key: PROFILE_DIMENSIONS[profile] + int(use_h1)
            for key, (profile, use_h1) in VARIANTS.items()
        },
        **feature_schema,
    })

    models_by_seed: dict[int, dict[str, dict[int, Any]]] = {}
    selection_all: list[dict[str, Any]] = []
    route_manifest: dict[str, Any] = {}
    for seed in base.SEEDS:
        routes = base.classifier_routes(
            inventory[seed]["checkpoint"], Path(args.data_root), "calibration",
            device, args.batch_size,
        )
        oracle, deployment = base.prepare_rows(
            Path(args.data_root), "calibration", routes, h1, runtime
        )
        models, selection = fit_seed_calibration(seed, oracle, deployment)
        models_by_seed[seed] = models
        selection_all.extend(selection)
        base.write_csv(
            output / f"calibration_selection_seed{seed}.csv",
            selection,
            tuple(selection[0]),
        )
        route_manifest[str(seed)] = {
            "checkpoint": str(inventory[seed]["checkpoint"]),
            "checkpoint_sha256": inventory[seed]["checkpoint_sha256"],
            "calibration_rows": len(routes),
        }

    lock_path = output / "calibration_selection_lock.json"
    lock = {
        "schema_version": SCHEMA_VERSION,
        "locked_at": base.utc_now(),
        "code_commit": base.git_commit(),
        "script_sha256": base.sha256_file(Path(__file__)),
        "selection_split": "C5_calibration_240_fit_80_validation",
        "selected_alphas": {
            str(seed): {
                variant: {
                    str(row["class_id"]): row["selected_alpha"]
                    for row in selection_all
                    if int(row["seed"]) == seed and row["variant"] == variant
                }
                for variant in VARIANTS
            }
            for seed in base.SEEDS
        },
        "test_used_for_fit_select_or_refit": False,
        "test_opened_after_lock": False,
        "test_opened_at": None,
    }
    base.write_json(lock_path, lock)
    if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise RuntimeError("calibration selection lock did not persist")

    lock["test_opened_after_lock"] = True
    lock["test_opened_at"] = base.utc_now()
    per_seed: list[dict[str, Any]] = []
    per_gas: list[dict[str, Any]] = []
    for seed in base.SEEDS:
        routes = base.classifier_routes(
            inventory[seed]["checkpoint"], Path(args.data_root), "test",
            device, args.batch_size,
        )
        _oracle, deployment = base.prepare_rows(
            Path(args.data_root), "test", routes, h1, runtime
        )
        predictions = apply_models(deployment, models_by_seed[seed])
        overall, gases = per_seed_metrics(seed, predictions)
        per_seed.extend(overall)
        per_gas.extend(gases)
        route_manifest[str(seed)]["test_rows"] = len(routes)
        route_manifest[str(seed)]["frozen_test_route"] = str(
            inventory[seed]["frozen_test_route"]
        )
    lock["test_evaluation_completed_at"] = base.utc_now()
    base.write_json(lock_path, lock)

    summary = base.summarize(
        per_seed,
        ("variant", "profile", "uses_H1", "input_dimension"),
        (
            "S_ALL_RMSE", "S_ALL_MAE", "S_ALL_NRMSE",
            "S_CC_RMSE", "S_CC_MAE", "S_CC_NRMSE",
        ),
    )
    gas_summary = base.summarize(
        per_gas,
        ("variant", "profile", "uses_H1", "class_id", "gas"),
        ("RMSE", "MAE", "NRMSE"),
    )
    paired_rows, decision = paired_comparisons(per_seed, per_gas)
    frozen_after = base.frozen_hashes(root)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen runtime/QC assets changed during ablation")

    base.write_csv(output / "per_seed_metrics.csv", per_seed, tuple(per_seed[0]))
    base.write_csv(output / "multiseed_summary.csv", summary, tuple(summary[0]))
    base.write_csv(output / "per_seed_per_gas_metrics.csv", per_gas, tuple(per_gas[0]))
    base.write_csv(output / "per_gas_multiseed_summary.csv", gas_summary, tuple(gas_summary[0]))
    base.write_csv(output / "paired_profile_differences.csv", paired_rows, tuple(paired_rows[0]))
    base.write_json(output / "feature_metadata_ablation_decision.json", decision)
    base.write_json(output / "protocol_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "registered_confirmatory",
        "code_commit": base.git_commit(),
        "script_sha256": base.sha256_file(Path(__file__)),
        "seed_set": list(base.SEEDS),
        "data_root": str(Path(args.data_root).resolve()),
        "split": {"C5_calibration": 320, "fit": 240, "validation": 80, "C5_test": 1360},
        "profiles": PROFILE_DIMENSIONS,
        "variants": VARIANTS,
        "ridge_alpha_grid": list(base.RIDGE_ALPHAS),
        "test_used_for_fit_select_or_refit": False,
        "source_heads_retrained": False,
        "QC": "none",
        "known_split_limitation": "window-level split; allows_file_overlap=true",
        "frozen_hashes_before": frozen_before,
        "frozen_hashes_after": frozen_after,
    })
    base.write_json(output / "classifier_route_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "routes": route_manifest,
        "same_route_and_mask_within_seed": True,
    })
    base.write_json(output / "source_head_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        **source_manifest,
        "source_heads_retrained": False,
    })
    (output / "README.md").write_text(
        "# IoT-J feature metadata ablation\n\n"
        "Registered five-seed 3x2 comparison of 83-D sensor-only, 91-D "
        "online-safe, and 104-D full target features, each without/with the "
        "frozen federated H1 prior. Test was opened only after all calibration "
        "alpha decisions were persisted. Existing runtime and QC assets were "
        "not modified.\n",
        encoding="utf-8",
    )
    base.write_json(output / "sha256_index.json", {
        path.name: {"sha256": base.sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "sha256_index.json"
    })
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--data-root",
        default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
    )
    parser.add_argument("--multiseed-root", default="results/iotj_b5_multiseed_20260724")
    parser.add_argument(
        "--runtime-contract",
        default=(
            "results/iotj_b5_c5_deployment_p1_20260722/"
            "c5_h8_runtime_contract_b5_v4/runtime_contract.json"
        ),
    )
    parser.add_argument(
        "--h1-manifest",
        default=(
            "results/iotj_h1_federated_ridge_equivalence_20260724/"
            "federated_h1_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir", default="results/iotj_feature_metadata_ablation_20260803"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
