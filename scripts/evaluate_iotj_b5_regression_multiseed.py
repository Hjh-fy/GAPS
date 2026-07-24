"""Formal five-seed B5 routed regression comparison (RG0/RG1/RG2).

The program is intentionally two-phase inside one process.  It evaluates only
C5 calibration until every per-seed alpha decision has been persisted in
``calibration_selection_lock.json``.  C5 test is loaded only after that file is
read back and validated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_deploy.c5_h8_runtime import C5H8Runtime, SerializedRidge
from run_regression_head_ablation import (
    CLASS_NAMES,
    CLASS_RANGES,
    build_oracle_rows,
    deterministic_train_val,
    fit_ridge,
)
from scripts.summarize_iotj_classification_ablation import (
    evaluate_checkpoint_stream,
)


SCHEMA_VERSION = "iotj.b5_regression_multiseed.v1"
SEEDS = (42, 43, 44, 45, 46)
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
PRIOR_KEYS = (
    "H1_federated_source_ridge_ppm",
    "H2_source_per_gas_mlp_ppm",
    "H3_source_shared_mlp_ppm",
)
VARIANTS: dict[str, tuple[str, ...]] = {
    "RG0_RICH_ONLY": (),
    "RG1_FEDERATED_H1": (PRIOR_KEYS[0],),
    "RG2_ALL_PRIOR": PRIOR_KEYS,
}
FROZEN_RUNTIME_ASSETS = {
    "bundle_manifest": (
        "results/iotj_b5_c5_deployment_p1_20260722/"
        "bundle_candidate/manifest.json"
    ),
    "c5_test_features": (
        "dataset/client_data_c1234src_c5tgt_2080_"
        "timeaware_60_170_window_fullgrid/client_5/test_features.npy"
    ),
    "c5_test_metadata": (
        "dataset/client_data_c1234src_c5tgt_2080_"
        "timeaware_60_170_window_fullgrid/client_5/test_experiment_info.json"
    ),
    "c5_test_phase_labels": (
        "dataset/client_data_c1234src_c5tgt_2080_"
        "timeaware_60_170_window_fullgrid/client_5/test_phase_labels.npy"
    ),
    "hc95_reference": (
        "results/iotj_b5_c5_deployment_p1_20260722/"
        "high_coverage_qc/test_hc95_records.csv"
    ),
    "hc90_reference": (
        "results/iotj_b5_c5_deployment_p1_20260722/"
        "high_coverage_qc/test_hc90_records.csv"
    ),
}
EXPECTED_FROZEN_HASHES = {
    "bundle_manifest": "a2514bd74ba0a98334d146af218922ee84884a53b93b0d4c44414723abee73b5",
    "c5_test_features": "7955cb70b24fa86ce109a52ca3b2231ad543b8ba8be0276781ffa03384143a82",
    "c5_test_metadata": "9b48459f52698b11fad66c0a2c63c9ede22292555e4bcaa71125e1f7e90097bf",
    "c5_test_phase_labels": "a69f333c8418fa3bf94c599a2d684cd122b4a46df2ff405bced227b68fcdb8b5",
    "hc95_reference": "33d04439376852bb976d9a4ed5f09235107b296c5f839c75ed667fdecc598860",
    "hc90_reference": "6051e7787915e0163ffd815dc089626e751906474c858072c5c0520c615dccb3",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def origin_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "origin/codex/iotj-confirmation-observability"],
        text=True,
    ).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def frozen_hashes(root: Path) -> dict[str, str]:
    observed = {
        key: sha256_file(root / relative)
        for key, relative in FROZEN_RUNTIME_ASSETS.items()
    }
    if observed != EXPECTED_FROZEN_HASHES:
        raise RuntimeError(
            f"runtime v4/HC frozen hashes differ: {observed}"
        )
    return observed


def checkpoint_inventory(multiseed_root: Path) -> dict[int, dict[str, Any]]:
    expected_hashes = {
        42: "9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c",
        43: "4a2174e85f069fa04a02bbbf8e0467dc42f8d25d015b58eadc57fe4a98784ab6",
        44: "cc1b07da93a45165f3acb47b86ed98caf263f7f0c8625ceb1ecca751067553f3",
        45: "b7c5d398beb5c8734fe2650b04d479a46f4e17bc0ac7c86eab266e2686581e8f",
        46: "26bcc33066a10268ce21ac7011ba636982c9073a645658e96c5d454b69608913",
    }
    metrics_paths = {
        seed: (
            multiseed_root
            / ("seed42_reference" if seed == 42 else f"seed{seed}")
            / "classification_evaluation"
            / f"seed{seed}_classification_metrics.json"
        )
        for seed in SEEDS
    }
    inventory: dict[int, dict[str, Any]] = {}
    for seed, metrics_path in metrics_paths.items():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(payload["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        observed = sha256_file(checkpoint)
        if int(payload["seed"]) != seed or observed != expected_hashes[seed]:
            raise RuntimeError(f"seed{seed} checkpoint identity differs")
        route_path = metrics_path.parent / f"seed{seed}_test_predictions.csv"
        inventory[seed] = {
            "checkpoint": checkpoint.resolve(),
            "checkpoint_sha256": observed,
            "classification_metrics": metrics_path.resolve(),
            "frozen_test_route": route_path.resolve(),
        }
    return inventory


def load_source_heads(
    h1_manifest_path: Path, runtime: C5H8Runtime
) -> tuple[dict[int, SerializedRidge], dict[str, Any]]:
    h1_hash = sha256_file(h1_manifest_path)
    if h1_hash != "d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc":
        raise RuntimeError("federated H1 manifest hash differs")
    payload = json.loads(h1_manifest_path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, Mapping) or set(models) != {"0", "1", "2", "3"}:
        raise RuntimeError("federated H1 requires four fixed gas heads")
    h1 = {
        int(class_id): SerializedRidge.from_json(model)
        for class_id, model in models.items()
    }
    if runtime.h8_policy is None:
        raise RuntimeError("frozen R4 policy is unavailable")
    return h1, {
        "federated_h1_manifest": str(h1_manifest_path.resolve()),
        "federated_h1_manifest_sha256": h1_hash,
        "federated_h1_status": "PRACTICAL_EQUIVALENCE",
        "h1_training_depends_on_classifier_seed": False,
        "h2_h3_training_depends_on_classifier_seed": False,
        "source_heads_retrained": False,
        "H1": "sufficient-statistics federated per-gas Ridge",
        "H2": "frozen pooled-source per-gas MLP from R4",
        "H3": "frozen pooled-source shared MLP from R4",
    }


def classifier_routes(
    checkpoint: Path,
    data_root: Path,
    split: str,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows, _metrics = evaluate_checkpoint_stream(
        checkpoint,
        data_root=data_root,
        target_client=5,
        split=split,
        device=device,
        batch_size=batch_size,
    )
    expected = 320 if split == "calibration" else 1360
    if len(rows) != expected:
        raise RuntimeError(f"C5 {split} route N differs: {len(rows)}")
    indexes = [int(row["sample_index"]) for row in rows]
    if indexes != list(range(expected)):
        raise RuntimeError(f"C5 {split} route keys are not canonical/unique")
    if any(int(row["pred_class"]) not in CLASS_NAMES for row in rows):
        raise RuntimeError(f"C5 {split} route contains an invalid class")
    return rows


def source_components(
    feature_dict: Mapping[str, Any],
    route_class: int,
    h1: Mapping[int, SerializedRidge],
    runtime: C5H8Runtime,
) -> dict[str, float]:
    if route_class not in CLASS_NAMES or runtime.h8_policy is None:
        raise RuntimeError("source-prior route is invalid")
    features = dict(feature_dict)
    features["route_class"] = route_class
    values = {
        PRIOR_KEYS[0]: h1[route_class].predict(features),
        PRIOR_KEYS[1]: runtime.h8_policy.source_mlp[route_class].predict(features),
        PRIOR_KEYS[2]: runtime.h8_policy.shared_mlp.predict(features),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("source head emitted NaN/Inf")
    return values


def prepare_rows(
    data_root: Path,
    split: str,
    routes: Sequence[Mapping[str, Any]],
    h1: Mapping[int, SerializedRidge],
    runtime: C5H8Runtime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = build_oracle_rows(data_root, ["C5"], split)
    if len(base) != len(routes):
        raise RuntimeError(f"C5 {split} base/route count differs")
    oracle: list[dict[str, Any]] = []
    deployment: list[dict[str, Any]] = []
    for index, (row, route) in enumerate(zip(base, routes)):
        if int(row["sample_index"]) != index or int(route["sample_index"]) != index:
            raise RuntimeError(f"C5 {split} row alignment differs")
        if int(row["true_class"]) != int(route["true_class"]):
            raise RuntimeError(f"C5 {split} label alignment differs")
        if len(row["feature_dict"]) != 104:
            raise RuntimeError("rich feature schema is not 104D")
        true_class = int(row["true_class"])
        pred_class = int(route["pred_class"])
        fit_row = dict(row)
        fit_row["pred_class"] = pred_class
        fit_row["route_class"] = true_class
        fit_row.update(
            source_components(
                fit_row["feature_dict"], true_class, h1, runtime
            )
        )
        deploy_row = dict(row)
        deploy_row["pred_class"] = pred_class
        deploy_row["route_class"] = pred_class
        deploy_row.update(
            source_components(
                deploy_row["feature_dict"], pred_class, h1, runtime
            )
        )
        oracle.append(fit_row)
        deployment.append(deploy_row)
    return oracle, deployment


def add_variant_features(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> list[dict[str, Any]]:
    keys = VARIANTS[variant]
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        features = dict(item["feature_dict"])
        for key in keys:
            features[f"srcpred_{key}"] = float(item[key])
        item["feature_dict"] = features
        output.append(item)
    expected = 104 + len(keys)
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
    deployment_by_id = {
        int(row["sample_index"]): row for row in deployment
    }
    for variant in VARIANTS:
        fit_features = add_variant_features(oracle, variant)
        deploy_features = add_variant_features(deployment, variant)
        deploy_by_id = {
            int(row["sample_index"]): row for row in deploy_features
        }
        names = sorted(fit_features[0]["feature_dict"])
        models[variant] = {}
        for class_id, gas in sorted(CLASS_NAMES.items()):
            class_rows = [
                row
                for row in fit_features
                if int(row["true_class"]) == class_id
            ]
            fit_rows, validation_seed_rows = deterministic_train_val(
                class_rows, 0.25
            )
            validation_rows = [
                deploy_by_id[int(row["sample_index"])]
                for row in validation_seed_rows
            ]
            if not (
                len(class_rows) == 80
                and len(fit_rows) == 60
                and len(validation_rows) == 20
            ):
                raise RuntimeError(
                    f"seed{seed} {gas} split is not 60/20"
                )
            true = np.asarray(
                [float(row["true_ppm"]) for row in validation_rows]
            )
            grid: list[dict[str, float]] = []
            best_alpha = RIDGE_ALPHAS[0]
            best_rmse = float("inf")
            for alpha in RIDGE_ALPHAS:
                candidate = fit_ridge(fit_rows, names, alpha)
                pred = candidate.predict(validation_rows)
                score = float(np.sqrt(np.mean((pred - true) ** 2)))
                grid.append({"alpha": alpha, "validation_RMSE": score})
                if score < best_rmse:
                    best_alpha, best_rmse = alpha, score
            models[variant][class_id] = fit_ridge(
                class_rows, names, best_alpha
            )
            selection.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "class_id": class_id,
                    "gas": gas,
                    "calibration_fit_N": len(fit_rows),
                    "calibration_validation_N": len(validation_rows),
                    "target_input_dimension": len(names),
                    "selected_alpha": best_alpha,
                    "calibration_validation_RMSE": best_rmse,
                    "alpha_grid_audit": json.dumps(grid),
                    "selection_split": "C5_calibration_internal_validation",
                }
            )
    if set(deployment_by_id) != set(range(320)):
        raise RuntimeError("calibration deployment row universe differs")
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
        for class_id in CLASS_NAMES:
            indexes = [
                index
                for index, row in enumerate(features)
                if int(row["route_class"]) == class_id
            ]
            pred = models[variant][class_id].predict(
                [features[index] for index in indexes]
            )
            for index, value in zip(indexes, pred):
                output[index][f"{variant}_ppm"] = float(value)
    values = np.asarray(
        [
            [row[f"{variant}_ppm"] for variant in VARIANTS]
            for row in output
        ],
        dtype=np.float64,
    )
    if values.shape != (len(output), 3) or not np.isfinite(values).all():
        raise RuntimeError("target Ridge output is missing or non-finite")
    return output


def metric_block(
    rows: Sequence[Mapping[str, Any]], variant: str, mask: np.ndarray
) -> dict[str, Any]:
    true = np.asarray([float(row["true_ppm"]) for row in rows])
    pred = np.asarray([float(row[f"{variant}_ppm"]) for row in rows])
    classes = np.asarray([int(row["true_class"]) for row in rows])
    error = pred - true
    selected = np.asarray(mask, dtype=bool)
    ranges = np.asarray([CLASS_RANGES[int(value)] for value in classes])
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
    true = np.asarray([float(row["true_ppm"]) for row in rows])
    all_mask = np.ones(len(rows), dtype=bool)
    correct = classes == route
    co = classes == 1
    co_high = co & (true >= 200.0) & (true <= 250.0)
    overall: list[dict[str, Any]] = []
    gases: list[dict[str, Any]] = []
    for variant in VARIANTS:
        s_all = metric_block(rows, variant, all_mask)
        s_cc = metric_block(rows, variant, correct)
        co_metrics = metric_block(rows, variant, co)
        co_high_metrics = metric_block(rows, variant, co_high)
        overall.append(
            {
                "seed": seed,
                "variant": variant,
                "S_ALL_N": s_all["N"],
                "S_ALL_RMSE": s_all["RMSE"],
                "S_ALL_MAE": s_all["MAE"],
                "S_ALL_NRMSE": s_all["NRMSE"],
                "S_CC_N": s_cc["N"],
                "S_CC_RMSE": s_cc["RMSE"],
                "S_CC_MAE": s_cc["MAE"],
                "S_CC_NRMSE": s_cc["NRMSE"],
                "route_error_count": int((~correct).sum()),
                "CO_RMSE": co_metrics["RMSE"],
                "CO_high_N": co_high_metrics["N"],
                "CO_high_RMSE": co_high_metrics["RMSE"],
            }
        )
        for class_id, gas in sorted(CLASS_NAMES.items()):
            block = metric_block(rows, variant, classes == class_id)
            gases.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "class_id": class_id,
                    "gas": gas,
                    "N": block["N"],
                    "RMSE": block["RMSE"],
                    "MAE": block["MAE"],
                }
            )
    return overall, gases


def summarize(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, selected in sorted(groups.items()):
        item = dict(zip(group_keys, group))
        item["seed_count"] = len(selected)
        for key in value_keys:
            values = np.asarray([float(row[key]) for row in selected])
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_sample_std"] = float(values.std(ddof=1))
            item[f"{key}_median"] = float(np.median(values))
            item[f"{key}_min"] = float(values.min())
            item[f"{key}_max"] = float(values.max())
        output.append(item)
    return output


def paired_rg1_rg2(
    per_seed: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {
        (int(row["seed"]), str(row["variant"])): row for row in per_seed
    }
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        h1 = lookup[(seed, "RG1_FEDERATED_H1")]
        all_prior = lookup[(seed, "RG2_ALL_PRIOR")]
        delta = float(h1["S_CC_RMSE"]) - float(all_prior["S_CC_RMSE"])
        paired.append(
            {
                "seed": seed,
                "RG1_S_CC_RMSE": h1["S_CC_RMSE"],
                "RG2_S_CC_RMSE": all_prior["S_CC_RMSE"],
                "delta_RG1_minus_RG2": delta,
                "relative_delta_percent": 100.0
                * delta
                / float(all_prior["S_CC_RMSE"]),
                "winner": (
                    "RG1_FEDERATED_H1" if delta < 0 else "RG2_ALL_PRIOR"
                ),
            }
        )
    values = np.asarray([row["delta_RG1_minus_RG2"] for row in paired])
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    # t(0.975, 4); descriptive interval, not an inferential selection rule.
    half_width = 2.7764451051977987 * std / math.sqrt(len(values))
    summary = {
        "N": len(values),
        "mean_delta_RG1_minus_RG2": mean,
        "sample_std": std,
        "median": float(median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "RG1_wins": int((values < 0).sum()),
        "RG2_wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "descriptive_t95_CI_low": mean - half_width,
        "descriptive_t95_CI_high": mean + half_width,
    }
    return paired, summary


def final_gate(
    per_seed: Sequence[Mapping[str, Any]],
    common_summary: Sequence[Mapping[str, Any]],
    per_gas_summary: Sequence[Mapping[str, Any]],
    paired_summary: Mapping[str, Any],
) -> dict[str, Any]:
    overall = {
        str(row["variant"]): row
        for row in summarize(
            per_seed,
            ("variant",),
            ("S_CC_RMSE", "S_ALL_RMSE", "CO_RMSE", "CO_high_RMSE"),
        )
    }
    rg1, rg2 = overall["RG1_FEDERATED_H1"], overall["RG2_ALL_PRIOR"]
    primary = (
        float(rg1["S_CC_RMSE_mean"]) - float(rg2["S_CC_RMSE_mean"])
    ) / float(rg2["S_CC_RMSE_mean"])
    s_all_delta = float(rg1["S_ALL_RMSE_mean"]) - float(rg2["S_ALL_RMSE_mean"])
    co_relative = (
        float(rg1["CO_RMSE_mean"]) - float(rg2["CO_RMSE_mean"])
    ) / float(rg2["CO_RMSE_mean"])
    co_high_relative = (
        float(rg1["CO_high_RMSE_mean"]) - float(rg2["CO_high_RMSE_mean"])
    ) / float(rg2["CO_high_RMSE_mean"])
    common = {str(row["variant"]): row for row in common_summary}
    common_relative = (
        float(common["RG1_FEDERATED_H1"]["COMMON_CORRECT_RMSE"])
        - float(common["RG2_ALL_PRIOR"]["COMMON_CORRECT_RMSE"])
    ) / float(common["RG2_ALL_PRIOR"]["COMMON_CORRECT_RMSE"])
    gas_seed = {
        (int(row["seed"]), str(row["variant"]), str(row["gas"])): float(row["RMSE"])
        for row in per_gas_summary
    }
    persistent_bad: dict[str, int] = {}
    for gas in CLASS_NAMES.values():
        count = sum(
            (
                gas_seed[(seed, "RG1_FEDERATED_H1", gas)]
                - gas_seed[(seed, "RG2_ALL_PRIOR", gas)]
            )
            / gas_seed[(seed, "RG2_ALL_PRIOR", gas)]
            > 0.10
            for seed in SEEDS
        )
        persistent_bad[gas] = int(count)
    checks = {
        "primary_S_CC_relative_le_1pct": primary <= 0.01,
        "S_ALL_absolute_delta_le_0_5ppm": s_all_delta <= 0.5,
        "CO_relative_worsening_le_5pct": co_relative <= 0.05,
        "CO_high_relative_worsening_le_5pct": co_high_relative <= 0.05,
        "no_gas_gt_10pct_worse_in_at_least_3_seeds": max(persistent_bad.values()) < 3,
        "COMMON_CORRECT_relative_worsening_le_1pct": common_relative <= 0.01,
    }
    all_pass = all(checks.values())
    ci_low = float(paired_summary["descriptive_t95_CI_low"])
    ci_high = float(paired_summary["descriptive_t95_CI_high"])
    inconsistent = (
        int(paired_summary["RG1_wins"]) > 0
        and int(paired_summary["RG2_wins"]) > 0
    )
    broad_crossing = ci_low < 0 < ci_high
    if all_pass and (inconsistent or broad_crossing):
        selection = "DESCRIPTIVE_NONINFERIOR_BUT_UNCERTAIN"
        action = "KEEP_RUNTIME_V4_PENDING_PAPER_DECISION"
    elif all_pass:
        selection = "SELECT_B5_FEDERATED_H1"
        action = "BUILD_RUNTIME_V5_CANDIDATE"
    else:
        selection = "SELECT_B5_ALL_PRIOR"
        action = "KEEP_RUNTIME_V4"
    return {
        "schema_version": "iotj.b5_regression_final_gate.v1",
        "decision": selection,
        "runtime_recommendation": action,
        "checks": checks,
        "observed": {
            "primary_S_CC_relative_delta": primary,
            "S_ALL_absolute_delta_ppm": s_all_delta,
            "CO_relative_delta": co_relative,
            "CO_high_relative_delta": co_high_relative,
            "COMMON_CORRECT_relative_delta": common_relative,
            "gas_gt_10pct_worse_seed_counts": persistent_bad,
            "paired_descriptive_t95_CI": [ci_low, ci_high],
            "paired_directions_inconsistent": inconsistent,
        },
        "test_used_for_selection": False,
        "runtime_modified": False,
        "qc_modified": False,
    }


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    output = Path(args.output_dir)
    require_empty_output(output)
    if args.formal_run and git_commit() != origin_commit():
        raise RuntimeError("formal run requires local HEAD == origin HEAD")
    frozen_before = frozen_hashes(root)
    inventory = checkpoint_inventory(Path(args.multiseed_root))
    runtime = C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    h1, source_manifest = load_source_heads(
        Path(args.h1_manifest), runtime
    )
    device = torch.device(args.device)

    models_by_seed: dict[int, dict[str, dict[int, Any]]] = {}
    selection_all: list[dict[str, Any]] = []
    route_manifest: dict[str, Any] = {}
    # Calibration-only phase.  No test route, label, feature or metric is read.
    for seed in SEEDS:
        routes = classifier_routes(
            inventory[seed]["checkpoint"],
            Path(args.data_root),
            "calibration",
            device,
            args.batch_size,
        )
        oracle, deployment = prepare_rows(
            Path(args.data_root), "calibration", routes, h1, runtime
        )
        models, selection = fit_seed_calibration(
            seed, oracle, deployment
        )
        models_by_seed[seed] = models
        selection_all.extend(selection)
        selection_path = output / f"calibration_selection_seed{seed}.csv"
        write_csv(
            selection_path,
            selection,
            (
                "seed",
                "variant",
                "class_id",
                "gas",
                "calibration_fit_N",
                "calibration_validation_N",
                "target_input_dimension",
                "selected_alpha",
                "calibration_validation_RMSE",
                "alpha_grid_audit",
                "selection_split",
            ),
        )
        route_manifest[str(seed)] = {
            "checkpoint": str(inventory[seed]["checkpoint"]),
            "checkpoint_sha256": inventory[seed]["checkpoint_sha256"],
            "calibration_rows": len(routes),
            "calibration_route_sha256": hashlib.sha256(
                json.dumps(
                    [
                        [row["sample_index"], row["true_class"], row["pred_class"]]
                        for row in routes
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }

    lock_path = output / "calibration_selection_lock.json"
    lock = {
        "schema_version": "iotj.b5_regression_calibration_lock.v1",
        "locked_at": utc_now(),
        "formal_run_commit": git_commit(),
        "selection_split": "C5_calibration_240_fit_80_validation",
        "seed_routes": route_manifest,
        "selected_alphas": {
            str(seed): {
                variant: {
                    str(row["class_id"]): row["selected_alpha"]
                    for row in selection_all
                    if int(row["seed"]) == seed and row["variant"] == variant
                }
                for variant in VARIANTS
            }
            for seed in SEEDS
        },
        "test_used_for_fit_select_or_refit": False,
        "test_opened_after_lock": False,
        "test_opened_at": None,
    }
    write_json(lock_path, lock)
    persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    if persisted != lock or persisted["test_opened_after_lock"]:
        raise RuntimeError("calibration selection lock did not persist")

    # Test-generalization phase starts only after the lock above is durable.
    test_opened_at = utc_now()
    per_seed: list[dict[str, Any]] = []
    per_gas: list[dict[str, Any]] = []
    test_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    correct_sets: list[set[int]] = []
    for seed in SEEDS:
        routes = classifier_routes(
            inventory[seed]["checkpoint"],
            Path(args.data_root),
            "test",
            device,
            args.batch_size,
        )
        # Exact equality to the already frozen classification route.
        with Path(inventory[seed]["frozen_test_route"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            frozen_route = list(csv.DictReader(handle))
        if len(frozen_route) != 1360 or any(
            int(a["sample_index"]) != int(b["sample_index"])
            or int(a["pred_class"]) != int(b["pred_class"])
            for a, b in zip(routes, frozen_route)
        ):
            raise RuntimeError(f"seed{seed} replay differs from frozen route")
        _oracle, deployment = prepare_rows(
            Path(args.data_root), "test", routes, h1, runtime
        )
        predictions = apply_models(deployment, models_by_seed[seed])
        test_rows_by_seed[seed] = predictions
        correct_sets.append(
            {
                int(row["sample_index"])
                for row in predictions
                if int(row["true_class"]) == int(row["pred_class"])
            }
        )
        overall, gases = per_seed_metrics(seed, predictions)
        per_seed.extend(overall)
        per_gas.extend(gases)
        route_manifest[str(seed)].update(
            {
                "test_rows": len(routes),
                "frozen_test_route": str(
                    inventory[seed]["frozen_test_route"]
                ),
                "frozen_test_route_sha256": sha256_file(
                    inventory[seed]["frozen_test_route"]
                ),
                "test_replay_equal": True,
            }
        )

    common = set.intersection(*correct_sets)
    if not common:
        raise RuntimeError("COMMON_CORRECT intersection is empty")
    common_rows = [{"sample_index": index} for index in sorted(common)]
    common_summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        values: list[float] = []
        true: list[float] = []
        for seed in SEEDS:
            for row in test_rows_by_seed[seed]:
                if int(row["sample_index"]) in common:
                    values.append(float(row[f"{variant}_ppm"]))
                    true.append(float(row["true_ppm"]))
        error = np.asarray(values) - np.asarray(true)
        common_summary.append(
            {
                "variant": variant,
                "COMMON_CORRECT_unique_rows": len(common),
                "seed_row_evaluations": len(values),
                "COMMON_CORRECT_RMSE": float(np.sqrt(np.mean(error ** 2))),
                "COMMON_CORRECT_MAE": float(np.mean(np.abs(error))),
            }
        )

    summary = summarize(
        per_seed,
        ("variant",),
        (
            "S_ALL_RMSE",
            "S_ALL_MAE",
            "S_ALL_NRMSE",
            "S_CC_RMSE",
            "S_CC_MAE",
            "S_CC_NRMSE",
            "CO_RMSE",
            "CO_high_RMSE",
        ),
    )
    per_gas_multi = summarize(
        per_gas, ("variant", "class_id", "gas"), ("RMSE", "MAE")
    )
    paired, paired_summary = paired_rg1_rg2(per_seed)
    decision = final_gate(
        per_seed, common_summary, per_gas, paired_summary
    )

    lock.update(
        {
            "test_opened_after_lock": True,
            "test_opened_at": test_opened_at,
            "test_evaluation_completed_at": utc_now(),
        }
    )
    write_json(lock_path, lock)
    frozen_after = frozen_hashes(root)
    if frozen_before != frozen_after:
        raise RuntimeError("runtime v4/HC assets changed during evaluation")

    write_json(
        output / "protocol_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "formal" if args.formal_run else "smoke_only",
            "formal_run_commit": git_commit(),
            "origin_commit_at_run": origin_commit(),
            "seed_set": list(SEEDS),
            "split": {
                "C5_calibration": 320,
                "calibration_fit": 240,
                "calibration_validation": 80,
                "C5_test": 1360,
            },
            "variants": {
                key: {
                    "target_head": "per-gas Ridge",
                    "input_dimension": 104 + len(value),
                    "source_prior_keys": list(value),
                }
                for key, value in VARIANTS.items()
            },
            "ridge_alpha_grid": list(RIDGE_ALPHAS),
            "selection": "calibration-validation only, per seed/gas/variant",
            "refit": "full 320-row calibration after alpha lock",
            "test_used_for_fit_select_or_refit": False,
            "common_correct_definition": (
                "intersection of classification-correct sample_index sets "
                "across seeds 42,43,44,45,46"
            ),
            "runtime_v4_modified": False,
            "qc_modified": False,
            "source_heads_retrained": False,
            "frozen_hashes_before": frozen_before,
            "frozen_hashes_after": frozen_after,
        },
    )
    write_json(
        output / "classifier_route_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "routes": route_manifest,
            "same_route_and_mask_used_across_variants_within_seed": True,
        },
    )
    write_json(
        output / "source_head_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            **source_manifest,
            "r4_policy": str(Path(args.r4_policy).resolve()),
            "r4_policy_sha256": sha256_file(args.r4_policy),
        },
    )
    write_csv(
        output / "per_seed_regression_metrics.csv",
        per_seed,
        tuple(per_seed[0]),
    )
    write_csv(
        output / "regression_multiseed_summary.csv",
        summary,
        tuple(summary[0]),
    )
    write_csv(
        output / "paired_rg1_vs_rg2.csv",
        [*paired, {"seed": "SUMMARY", **paired_summary}],
        tuple({key for row in [*paired, paired_summary] for key in row}),
    )
    write_csv(
        output / "common_correct_rows.csv",
        common_rows,
        ("sample_index",),
    )
    write_csv(
        output / "common_correct_summary.csv",
        common_summary,
        tuple(common_summary[0]),
    )
    write_csv(
        output / "per_gas_multiseed_summary.csv",
        per_gas_multi,
        tuple(per_gas_multi[0]),
    )
    # Retain per-seed/per-gas rows needed to audit the majority-seed guard.
    write_csv(
        output / "per_seed_per_gas_metrics.csv",
        per_gas,
        tuple(per_gas[0]),
    )
    write_json(output / "final_regression_decision.json", decision)

    sha_paths = [
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "sha256_index.json"
    ]
    write_json(
        output / "sha256_index.json",
        {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(sha_paths)
        },
    )
    (output / "README.md").write_text(
        "# B5 regression five-seed confirmation\n\n"
        "Formal RG0/RG1/RG2 comparison for seeds 42–46. Alpha selection "
        "used only C5 calibration fit/validation. Test was opened only "
        "after `calibration_selection_lock.json` was persisted. Runtime v4, "
        "QC, classifiers, and source heads were not modified or retrained.\n",
        encoding="utf-8",
    )
    # Refresh the index so README is covered as well.
    write_json(
        output / "sha256_index.json",
        {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "sha256_index.json"
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--data-root",
        default=(
            "dataset/client_data_c1234src_c5tgt_2080_"
            "timeaware_60_170_window_fullgrid"
        ),
    )
    parser.add_argument(
        "--multiseed-root",
        default="results/iotj_b5_multiseed_20260724",
    )
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
        "--r4-policy",
        default=(
            "results/iotj_b5_c5_deployment_p1_20260722/"
            "h8_no_rescue/r4_policy.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/iotj_b5_regression_multiseed_20260724",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
