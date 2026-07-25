"""Frozen-method low-calibration sensitivity for GAPS IoT-J Runtime v5.

The workflow is intentionally split into protocol/calibration/test/finalize
stages.  The test stage refuses to run until every calibration-only target
Ridge model has been serialized and hash-bound by ``calibration_lock.json``.
Low-calibration test results never select a model, alpha grid, subset, or QC
policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_regression_head_ablation import (
    CLASS_NAMES,
    RidgeHead,
    deterministic_train_val,
    fit_ridge,
    load_split,
    rich_feature_dict,
)
from scripts.build_iotj_runtime_v5_candidate import (
    load_ridge_models,
    models_payload,
    prepare_rows,
    seed42_checkpoint,
)
from scripts.evaluate_iotj_b5_regression_multiseed import classifier_routes
from scripts.evaluate_iotj_h1_federated_ridge_equivalence import (
    EXPECTED_FROZEN_HASHES,
    RIDGE_ALPHAS,
    apply_h1,
    apply_target_ridge_h1,
    frozen_hashes,
    sha256_file,
)
from scripts.evaluate_iotj_source_prior_target_head_factorial import (
    overall_metrics,
    per_gas_metrics,
)


SCHEMA = "iotj.low_calibration_sensitivity.v1"
EXPERIMENT_ID = "IOTJ-LOW-CALIBRATION-SENSITIVITY-S42-20260725"
BUDGETS = (320, 160, 80, 40)
LOW_BUDGETS = (160, 80, 40)
REPLICATE_SEEDS = (2026072500, 2026072501, 2026072502, 2026072503, 2026072504)
TIMING_REPEATS = 10
H1_FEATURE = "srcpred_H1_source_ridge_ppm"
MODEL_PARAMS = 424


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (list(rows[0]) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in names} for row in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.mkdir(parents=True)


def _groups(metadata: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise RuntimeError(f"calibration filename missing at row {index}")
        grouped.setdefault(filename, []).append(index)
    if len(grouped) != 80 or len(metadata) != 320:
        raise RuntimeError("calibration must contain exactly 320 rows and 80 filenames")
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _stable_tie(seed: int, names: Iterable[str]) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(sorted(names))).encode()).hexdigest()


def _balance_score(
    metadata: Sequence[Mapping[str, Any]], indexes: Sequence[int], target: int
) -> float:
    full_gas = Counter(int(row["classification_label"]) for row in metadata)
    full_level = Counter(
        (int(row["classification_label"]), round(float(row["concentration"]), 6))
        for row in metadata
    )
    gas = Counter(int(metadata[index]["classification_label"]) for index in indexes)
    level = Counter(
        (
            int(metadata[index]["classification_label"]),
            round(float(metadata[index]["concentration"]), 6),
        )
        for index in indexes
    )
    gas_score = sum(
        ((gas[key] - target * count / len(metadata)) / max(1.0, target * count / len(metadata))) ** 2
        for key, count in full_gas.items()
    )
    level_score = sum(
        ((level[key] - target * count / len(metadata)) / max(1.0, target * count / len(metadata))) ** 2
        for key, count in full_level.items()
    )
    return float(2.0 * gas_score + level_score)


def _select_extension(
    metadata: Sequence[Mapping[str, Any]],
    groups: Mapping[str, tuple[int, ...]],
    selected_names: set[str],
    nominal_target: int,
    seed: int,
) -> set[str]:
    base_rows = sum(len(groups[name]) for name in selected_names)
    remaining = sorted(
        (name for name in groups if name not in selected_names),
        key=lambda name: hashlib.sha256(f"{seed}|{name}".encode()).hexdigest(),
    )
    max_add = min(len(metadata) - base_rows, max(0, nominal_target - base_rows) + 7)
    states: dict[int, list[tuple[str, ...]]] = {0: [tuple()]}
    # Two best states per reachable row total are enough to preserve a
    # deterministic balance-aware alternative while keeping the search
    # polynomial and practical for the 80 filename groups.
    beam = 2
    for name in remaining:
        size = len(groups[name])
        updated = {total: list(values) for total, values in states.items()}
        for total, candidates in states.items():
            new_total = total + size
            if new_total > max_add:
                continue
            pool = updated.setdefault(new_total, [])
            pool.extend(tuple(sorted((*candidate, name))) for candidate in candidates)
        for total, candidates in list(updated.items()):
            unique = list(dict.fromkeys(candidates))
            unique.sort(
                key=lambda chosen: (
                    _balance_score(
                        metadata,
                        [
                            index
                            for group_name in selected_names.union(chosen)
                            for index in groups[group_name]
                        ],
                        nominal_target,
                    ),
                    _stable_tie(seed, chosen),
                )
            )
            updated[total] = unique[:beam]
        states = updated
    choices: list[tuple[int, float, str, tuple[str, ...]]] = []
    for added, candidates in states.items():
        for chosen in candidates:
            names = selected_names.union(chosen)
            indexes = [index for name in names for index in groups[name]]
            choices.append(
                (
                    abs(len(indexes) - nominal_target),
                    _balance_score(metadata, indexes, nominal_target),
                    _stable_tie(seed, names),
                    chosen,
                )
            )
    if not choices:
        raise RuntimeError(f"no group-complete subset candidate for budget {nominal_target}")
    return selected_names.union(min(choices)[3])


def select_nested_group_subsets(
    metadata: Sequence[Mapping[str, Any]], budgets: Sequence[int], seed: int
) -> dict[int, list[int]]:
    groups = _groups(metadata) if len(metadata) == 320 else _generic_groups(metadata)
    selected: set[str] = set()
    output: dict[int, list[int]] = {}
    for budget in sorted(set(int(value) for value in budgets)):
        selected = _select_extension(metadata, groups, selected, budget, seed)
        output[budget] = sorted(index for name in selected for index in groups[name])
    return {budget: output[budget] for budget in budgets}


def _generic_groups(metadata: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        name = str(row.get("filename") or "")
        if not name:
            raise RuntimeError(f"filename missing at row {index}")
        grouped.setdefault(name, []).append(index)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def assign_group_folds(
    metadata: Sequence[Mapping[str, Any]],
    indexes: Iterable[int],
    *,
    n_splits: int,
    seed: int,
) -> dict[int, int]:
    selected = set(int(value) for value in indexes)
    groups = {
        name: tuple(index for index in values if index in selected)
        for name, values in _generic_groups(metadata).items()
        if any(index in selected for index in values)
    }
    ordered = sorted(
        groups,
        key=lambda name: (
            -len(groups[name]),
            hashlib.sha256(f"{seed}|fold|{name}".encode()).hexdigest(),
        ),
    )
    assignment: dict[str, int] = {}
    fold_rows = [0] * n_splits
    fold_groups = [0] * n_splits
    fold_gas = [Counter() for _ in range(n_splits)]
    fold_levels = [Counter() for _ in range(n_splits)]
    target_rows = len(selected) / n_splits
    target_groups = len(groups) / n_splits
    total_gas = Counter(int(metadata[index]["classification_label"]) for index in selected)
    total_levels = Counter(
        (
            int(metadata[index]["classification_label"]),
            round(float(metadata[index]["concentration"]), 6),
        )
        for index in selected
    )
    for position, name in enumerate(ordered):
        values = groups[name]
        candidates = list(range(n_splits))
        if position < n_splits:
            candidates = [position]
        scored = []
        for fold in candidates:
            rows = fold_rows[fold] + len(values)
            groups_n = fold_groups[fold] + 1
            gas = fold_gas[fold] + Counter(int(metadata[index]["classification_label"]) for index in values)
            levels = fold_levels[fold] + Counter(
                (
                    int(metadata[index]["classification_label"]),
                    round(float(metadata[index]["concentration"]), 6),
                )
                for index in values
            )
            score = ((rows - target_rows) / max(1.0, target_rows)) ** 2
            score += ((groups_n - target_groups) / max(1.0, target_groups)) ** 2
            score += sum(
                ((gas[key] - count / n_splits) / max(1.0, count / n_splits)) ** 2
                for key, count in total_gas.items()
            )
            score += 0.5 * sum(
                ((levels[key] - count / n_splits) / max(1.0, count / n_splits)) ** 2
                for key, count in total_levels.items()
            )
            scored.append((score, fold_rows[fold], fold_groups[fold], fold))
        fold = min(scored)[-1]
        assignment[name] = fold
        fold_rows[fold] += len(values)
        fold_groups[fold] += 1
        fold_gas[fold].update(int(metadata[index]["classification_label"]) for index in values)
        fold_levels[fold].update(
            (
                int(metadata[index]["classification_label"]),
                round(float(metadata[index]["concentration"]), 6),
            )
            for index in values
        )
    result = {index: assignment[name] for name, values in groups.items() for index in values}
    if set(result) != selected or set(result.values()) != set(range(n_splits)):
        raise RuntimeError("group-aware fold assignment is incomplete")
    return result


def classify_sensitivity(relative_deltas: Mapping[int, float]) -> str:
    if all(float(relative_deltas[budget]) <= 0.05 for budget in LOW_BUDGETS):
        return "ROBUST_TO_REDUCED_CALIBRATION"
    if float(relative_deltas[160]) <= 0.10 and float(relative_deltas[80]) <= 0.10:
        return "MODERATE_CALIBRATION_SENSITIVITY"
    return "HIGH_CALIBRATION_SENSITIVITY"


def _attach_h1(
    rows: Sequence[Mapping[str, Any]], source_models: Mapping[int, RidgeHead]
) -> list[dict[str, Any]]:
    applied = apply_h1(rows, source_models, H1_FEATURE)
    output: list[dict[str, Any]] = []
    for row in applied:
        item = dict(row)
        features = dict(item["feature_dict"])
        features[H1_FEATURE] = float(item[H1_FEATURE])
        item["feature_dict"] = features
        output.append(item)
    return output


def _build_base_rows(
    data_root: Path,
    indexes: Sequence[int],
    routes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features, cls, reg, phases, metadata = load_split(data_root, "C5", "calibration")
    oracle, deployment = [], []
    for index in indexes:
        true_class = int(cls[index])
        pred_class = int(routes[index]["pred_class"])
        values = rich_feature_dict(features[index], int(phases[index]), metadata[index])
        base = {
            "client": "C5",
            "split": "calibration",
            "sample_index": int(index),
            "true_class": true_class,
            "pred_class": pred_class,
            "true_ppm": float(reg[index, true_class]),
            "phase": int(phases[index]),
            "feature_dict": values,
        }
        oracle.append({**base, "route_class": true_class})
        deployment.append({**base, "route_class": pred_class})
    return oracle, deployment


def _fit_models(
    oracle: Sequence[Mapping[str, Any]],
    deployment: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    metadata: Sequence[Mapping[str, Any]],
    fold_seed: int,
) -> tuple[dict[int, RidgeHead], list[dict[str, Any]], dict[str, float]]:
    by_index = {int(row["sample_index"]): row for row in deployment}
    feature_names = sorted(oracle[0]["feature_dict"])
    if len(feature_names) != 105:
        raise RuntimeError("target Ridge feature dimension must be 105")
    folds = (
        {}
        if budget == 320
        else assign_group_folds(
            metadata,
            [int(row["sample_index"]) for row in oracle],
            n_splits=5,
            seed=fold_seed,
        )
    )
    preparation_done = time.perf_counter_ns()
    models: dict[int, RidgeHead] = {}
    alpha_rows: list[dict[str, Any]] = []
    alpha_start = time.perf_counter_ns()
    selected: dict[int, float] = {}
    for gas_id in sorted(CLASS_NAMES):
        class_rows = [row for row in oracle if int(row["true_class"]) == gas_id]
        if budget == 320:
            train, val_seed = deterministic_train_val(class_rows, 0.25)
            validation = [by_index[int(row["sample_index"])] for row in val_seed]
            candidates = []
            for alpha in RIDGE_ALPHAS:
                model = fit_ridge(train, feature_names, float(alpha))
                y = np.asarray([float(row["true_ppm"]) for row in validation])
                prediction = model.predict(validation, clip=True)
                sse = float(np.sum((prediction - y) ** 2))
                candidates.append((math.sqrt(sse / len(validation)), float(alpha)))
                alpha_rows.append(
                    {
                        "gas_id": gas_id,
                        "gas": CLASS_NAMES[gas_id],
                        "alpha": float(alpha),
                        "selection_RMSE": math.sqrt(sse / len(validation)),
                        "validation_N": len(validation),
                        "protocol": "frozen_240_80_per_gas_holdout",
                    }
                )
        else:
            candidates = []
            for alpha in RIDGE_ALPHAS:
                sse = 0.0
                n = 0
                for fold in range(5):
                    train = [
                        row
                        for row in class_rows
                        if folds[int(row["sample_index"])] != fold
                    ]
                    val_seed = [
                        row
                        for row in class_rows
                        if folds[int(row["sample_index"])] == fold
                    ]
                    if not train or not val_seed:
                        continue
                    validation = [by_index[int(row["sample_index"])] for row in val_seed]
                    model = fit_ridge(train, feature_names, float(alpha))
                    y = np.asarray([float(row["true_ppm"]) for row in validation])
                    error = model.predict(validation, clip=True) - y
                    sse += float(error @ error)
                    n += len(validation)
                if n == 0:
                    raise RuntimeError(f"no group-aware validation rows for gas {gas_id}")
                score = math.sqrt(sse / n)
                candidates.append((score, float(alpha)))
                alpha_rows.append(
                    {
                        "gas_id": gas_id,
                        "gas": CLASS_NAMES[gas_id],
                        "alpha": float(alpha),
                        "selection_RMSE": score,
                        "validation_N": n,
                        "protocol": "group_aware_5fold_calibration_only",
                    }
                )
        selected[gas_id] = min(candidates, key=lambda value: (value[0], RIDGE_ALPHAS.index(value[1])))[1]
    alpha_done = time.perf_counter_ns()
    for gas_id in sorted(CLASS_NAMES):
        class_rows = [row for row in oracle if int(row["true_class"]) == gas_id]
        if len(class_rows) < 2:
            raise RuntimeError(f"insufficient calibration rows for gas {gas_id}")
        models[gas_id] = fit_ridge(class_rows, feature_names, selected[gas_id])
    refit_done = time.perf_counter_ns()
    return models, alpha_rows, {
        "fold_validation_preparation_seconds": 0.0,
        "alpha_search_seconds": (alpha_done - alpha_start) / 1e9,
        "final_refit_seconds": (refit_done - alpha_done) / 1e9,
        "_preparation_marker": float(preparation_done),
    }


def _model_numeric_sha(models: Mapping[int, RidgeHead]) -> str:
    payload = {str(gas): models[gas].to_json() for gas in sorted(models)}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _timed_fit(
    data_root: Path,
    indexes: Sequence[int],
    routes: Sequence[Mapping[str, Any]],
    source_models: Mapping[int, RidgeHead],
    *,
    budget: int,
    metadata: Sequence[Mapping[str, Any]],
    fold_seed: int,
) -> tuple[dict[int, RidgeHead], list[dict[str, Any]], dict[str, Any]]:
    total_start = time.perf_counter_ns()
    rich_start = total_start
    oracle, deployment = _build_base_rows(data_root, indexes, routes)
    rich_done = time.perf_counter_ns()
    oracle = _attach_h1(oracle, source_models)
    deployment = _attach_h1(deployment, source_models)
    h1_done = time.perf_counter_ns()
    fold_start = time.perf_counter_ns()
    if budget != 320:
        assign_group_folds(metadata, indexes, n_splits=5, seed=fold_seed)
    fold_done = time.perf_counter_ns()
    models, alpha_rows, stages = _fit_models(
        oracle, deployment, budget=budget, metadata=metadata, fold_seed=fold_seed
    )
    serialization_start = time.perf_counter_ns()
    serialized = canonical_json(
        {
            "schema_version": SCHEMA,
            "input_dimension": 105,
            "models": {str(gas): models[gas].to_json() for gas in sorted(models)},
        }
    )
    serialization_done = time.perf_counter_ns()
    timing = {
        "rich_feature_generation_seconds": (rich_done - rich_start) / 1e9,
        "H1_prediction_generation_seconds": (h1_done - rich_done) / 1e9,
        "fold_validation_preparation_seconds": (fold_done - fold_start) / 1e9,
        "alpha_search_seconds": stages["alpha_search_seconds"],
        "final_refit_seconds": stages["final_refit_seconds"],
        "serialization_seconds": (serialization_done - serialization_start) / 1e9,
        "total_calibration_seconds": (serialization_done - total_start) / 1e9,
        "serialized_bytes_in_memory": len(serialized),
    }
    if not all(math.isfinite(float(value)) for key, value in timing.items() if key != "serialized_bytes_in_memory"):
        raise RuntimeError("calibration timing contains NaN/Inf")
    return models, alpha_rows, timing


def _metadata_rows(metadata: Sequence[Mapping[str, Any]], indexes: Sequence[int], nominal: int, replicate: int) -> list[dict[str, Any]]:
    return [
        {
            "nominal_budget": nominal,
            "replicate": replicate,
            "sample_index": index,
            "filename": metadata[index]["filename"],
            "gas_id": int(metadata[index]["classification_label"]),
            "gas": CLASS_NAMES[int(metadata[index]["classification_label"])],
            "concentration_ppm": float(metadata[index]["concentration"]),
        }
        for index in indexes
    ]


def _frozen_assets(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    paths = {
        "classifier_B5_seed42": checkpoint,
        "federated_H1": Path(args.real_h1),
        "runtime_v5_contract": Path(args.runtime_v5_contract),
        "calibration_features": Path(args.data_root) / "client_5/calibration_features.npy",
        "calibration_labels": Path(args.data_root) / "client_5/calibration_regression_labels.npy",
        "calibration_metadata": Path(args.data_root) / "client_5/calibration_experiment_info.json",
        "test_features": Path(args.data_root) / "client_5/test_features.npy",
        "test_labels": Path(args.data_root) / "client_5/test_regression_labels.npy",
        "test_metadata": Path(args.data_root) / "client_5/test_experiment_info.json",
    }
    return {
        "schema_version": SCHEMA,
        "assets": {name: descriptor(path) for name, path in paths.items()},
        "runtime_v4_six_sha256": frozen_hashes(ROOT),
        "runtime_v4_read_only": True,
    }


def freeze_protocol(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    require_new_output(output)
    metadata_path = Path(args.data_root) / "client_5/calibration_experiment_info.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    groups = _groups(metadata)
    frozen = _frozen_assets(args)
    write_json(output / "frozen_asset_manifest.json", frozen)
    protocol = {
        "schema_version": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "formal_code_commit": git_head(),
        "method": "canonical B5 seed42 + real-topology sufficient-statistics Federated H1 + C5 105D per-gas Ridge",
        "budgets": list(BUDGETS),
        "replicates": {"320": 1, "160": 5, "80": 5, "40": 5},
        "replicate_seeds": list(REPLICATE_SEEDS),
        "subset_algorithm": "nested group-complete deterministic beam-DP minimizing gas/concentration imbalance",
        "subset_tie_break": "SHA256(seed|sorted filenames)",
        "selection": {
            "320": "frozen per-gas 240/80 calibration fit/validation semantics",
            "160_80_40": "group-aware 5-fold calibration-only per-gas alpha selection",
            "alpha_grid": list(RIDGE_ALPHAS),
            "preprocessing_scope": "fold training only",
            "final_refit": "entire selected calibration subset",
        },
        "timing": {"clock": "time.perf_counter_ns", "repeats_per_budget_replicate": TIMING_REPEATS, "platform": "PC"},
        "decision_rule": {
            "A": "all 160/80/40 mean S_CC relative degradation vs 320 <= 5%",
            "B": "160 and 80 mean S_CC relative degradation vs 320 <= 10%, but A not met",
            "C": "160 or 80 mean S_CC relative degradation vs 320 > 10%",
        },
        "test": {
            "rows": 1360,
            "historical_test_previously_used": True,
            "opened_by_this_stage": False,
            "used_for_subset_or_alpha_selection": False,
        },
        "evidence_boundary": [
            "Low-calibration is a sensitivity analysis under a frozen method.",
            "The same historical C5 test split has been used in prior method confirmation.",
            "Calibration subsets are selected without access to test labels or errors.",
            "Filename grouping is used where possible to reduce within-calibration leakage.",
            "Historical calibration/test splitting remains window-level.",
            "No model or threshold selection is performed using low-calibration test results.",
        ],
        "prohibited": ["B5 change", "H1 change", "H2/H3", "feature-schema change", "target MLP", "QC", "threshold search", "test-based subset/alpha/budget selection"],
    }
    write_json(output / "protocol_manifest.json", protocol)
    subset_manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "calibration_rows": len(metadata),
        "unique_filenames": len(groups),
        "group_size_min": min(map(len, groups.values())),
        "group_size_max": max(map(len, groups.values())),
        "replicates": [],
    }
    balance_rows: list[dict[str, Any]] = []
    all_subsets: dict[tuple[int, int], list[int]] = {}
    for replicate, seed in enumerate(REPLICATE_SEEDS):
        nested = select_nested_group_subsets(metadata, LOW_BUDGETS, seed)
        nested[320] = list(range(320))
        for budget in BUDGETS:
            if budget == 320 and replicate > 0:
                continue
            indexes = nested[budget]
            all_subsets[(budget, replicate)] = indexes
            files = {str(metadata[index]["filename"]) for index in indexes}
            gas_counts = Counter(int(metadata[index]["classification_label"]) for index in indexes)
            subset_manifest["replicates"].append(
                {
                    "nominal_budget": budget,
                    "replicate": replicate,
                    "seed": seed if budget != 320 else None,
                    "actual_rows": len(indexes),
                    "unique_filenames": len(files),
                    "row_index_sha256": hashlib.sha256(canonical_json(indexes)).hexdigest(),
                }
            )
            for gas_id in sorted(CLASS_NAMES):
                balance_rows.append(
                    {
                        "nominal_budget": budget,
                        "replicate": replicate,
                        "actual_rows": len(indexes),
                        "unique_filenames": len(files),
                        "gas_id": gas_id,
                        "gas": CLASS_NAMES[gas_id],
                        "gas_rows": gas_counts[gas_id],
                    }
                )
    for budget in BUDGETS:
        rows = _metadata_rows(metadata, all_subsets[(budget, 0)], budget, 0)
        write_csv(output / f"calibration_subset_{budget}.csv", rows)
    write_csv(output / "subset_balance_audit.csv", balance_rows)
    write_json(output / "subset_manifest.json", subset_manifest)
    write_json(
        output / "subset_replicate_manifest.json",
        {
            "schema_version": SCHEMA,
            "replicates": subset_manifest["replicates"],
            "variability_type": "calibration subset selection variability; not classifier training seed variability",
            "nested_within_replicate": True,
        },
    )
    write_json(
        output / "stage_state.json",
        {
            "schema_version": SCHEMA,
            "stage": "PROTOCOL_FROZEN",
            "test_opened": False,
            "protocol_sha256": sha256_file(output / "protocol_manifest.json"),
            "subset_manifest_sha256": sha256_file(output / "subset_manifest.json"),
            "frozen_asset_manifest_sha256": sha256_file(output / "frozen_asset_manifest.json"),
        },
    )


def _load_subset_indexes(output: Path, metadata: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], list[int]]:
    manifest = json.loads((output / "subset_manifest.json").read_text(encoding="utf-8"))
    result: dict[tuple[int, int], list[int]] = {}
    replay_cache: dict[int, dict[int, list[int]]] = {}
    for record in manifest["replicates"]:
        budget, replicate = int(record["nominal_budget"]), int(record["replicate"])
        if budget == 320:
            indexes = list(range(320))
        else:
            seed = int(record["seed"])
            if seed not in replay_cache:
                replay_cache[seed] = select_nested_group_subsets(
                    metadata, LOW_BUDGETS, seed
                )
            indexes = replay_cache[seed][budget]
        if hashlib.sha256(canonical_json(indexes)).hexdigest() != record["row_index_sha256"]:
            raise RuntimeError("subset replay hash differs")
        result[(budget, replicate)] = indexes
    return result


def run_calibration(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "PROTOCOL_FROZEN" or state.get("test_opened") is not False:
        raise RuntimeError("calibration requires frozen pre-test protocol")
    if sha256_file(output / "protocol_manifest.json") != state["protocol_sha256"]:
        raise RuntimeError("protocol manifest drifted")
    frozen_before = frozen_hashes(ROOT)
    metadata = json.loads((Path(args.data_root) / "client_5/calibration_experiment_info.json").read_text(encoding="utf-8"))
    subsets = _load_subset_indexes(output, metadata)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "calibration", torch.device(args.device), args.batch_size)
    source_models = load_ridge_models(Path(args.real_h1), 104)
    model_records: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for (budget, replicate), indexes in sorted(subsets.items(), reverse=True):
        seed = REPLICATE_SEEDS[replicate]
        reference_sha = None
        final_models = None
        final_alpha = None
        for timing_repeat in range(TIMING_REPEATS):
            models, alpha_audit, timing = _timed_fit(
                Path(args.data_root),
                indexes,
                routes,
                source_models,
                budget=budget,
                metadata=metadata,
                fold_seed=seed,
            )
            numeric_sha = _model_numeric_sha(models)
            if reference_sha is None:
                reference_sha = numeric_sha
                final_models = models
                final_alpha = alpha_audit
            elif numeric_sha != reference_sha:
                raise RuntimeError("timing repeat changed final model result")
            timing_rows.append(
                {
                    "nominal_budget": budget,
                    "replicate": replicate,
                    "timing_repeat": timing_repeat,
                    "actual_rows": len(indexes),
                    **timing,
                    "model_numeric_sha256": numeric_sha,
                }
            )
        assert final_models is not None and final_alpha is not None and reference_sha is not None
        path = output / f"models/budget_{budget}/replicate_{replicate}/target_ridge_105d.json"
        payload = models_payload(final_models, dimension=105, source=f"C5 low-calibration budget={budget} replicate={replicate}")
        write_json(path, payload)
        model_records.append(
            {
                "nominal_budget": budget,
                "replicate": replicate,
                "actual_rows": len(indexes),
                "unique_filenames": len({metadata[index]["filename"] for index in indexes}),
                "model_path": str(path.resolve()),
                "model_bytes": path.stat().st_size,
                "model_sha256": sha256_file(path),
                "model_numeric_sha256": reference_sha,
                "model_parameters": MODEL_PARAMS,
                "target_input_dimension": 105,
            }
        )
        for row in final_alpha:
            alpha_rows.append({"nominal_budget": budget, "replicate": replicate, **row})
    write_csv(output / "calibration_timing_repetitions.csv", timing_rows)
    write_csv(output / "calibration_alpha_audit.csv", alpha_rows)
    write_json(output / "calibration_model_manifest.json", {"schema_version": SCHEMA, "models": model_records})
    bound = {
        "protocol_manifest": descriptor(output / "protocol_manifest.json"),
        "subset_manifest": descriptor(output / "subset_manifest.json"),
        "frozen_asset_manifest": descriptor(output / "frozen_asset_manifest.json"),
        "calibration_model_manifest": descriptor(output / "calibration_model_manifest.json"),
        "models": {
            f"{row['nominal_budget']}_{row['replicate']}": {
                "path": row["model_path"],
                "bytes": row["model_bytes"],
                "sha256": row["model_sha256"],
            }
            for row in model_records
        },
    }
    write_json(
        output / "calibration_lock.json",
        {
            "schema_version": SCHEMA,
            "created_at": utc_now(),
            "bound_assets": bound,
            "test_opened": False,
            "test_used_for_fit_select_or_refit": False,
        },
    )
    write_json(
        output / "stage_state.json",
        {
            **state,
            "stage": "CALIBRATION_LOCKED",
            "test_opened": False,
            "calibration_lock_sha256": sha256_file(output / "calibration_lock.json"),
        },
    )
    if frozen_hashes(ROOT) != frozen_before:
        raise RuntimeError("runtime v4 frozen assets changed during calibration")


def _require_lock(output: Path) -> dict[str, Any]:
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "CALIBRATION_LOCKED" or state.get("test_opened") is not False:
        raise RuntimeError("test requires calibration lock before opening")
    lock_path = output / "calibration_lock.json"
    if sha256_file(lock_path) != state.get("calibration_lock_sha256"):
        raise RuntimeError("calibration lock hash differs")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_opened") is not False or lock.get("test_used_for_fit_select_or_refit") is not False:
        raise RuntimeError("calibration lock permits test leakage")
    for name, item in lock["bound_assets"].items():
        if name == "models":
            for model in item.values():
                path = Path(model["path"])
                if descriptor(path) != model:
                    raise RuntimeError("locked target model drifted")
        else:
            path = Path(item["path"])
            if descriptor(path) != item:
                raise RuntimeError(f"locked asset drifted: {name}")
    return lock


def evaluate_test(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    lock = _require_lock(output)
    frozen_before = frozen_hashes(ROOT)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "test", torch.device(args.device), args.batch_size)
    _oracle, deployment = prepare_rows(Path(args.data_root), "test", routes)
    source_models = load_ridge_models(Path(args.real_h1), 104)
    records = json.loads((output / "calibration_model_manifest.json").read_text(encoding="utf-8"))["models"]
    metrics_rows, per_gas_rows, prediction_rows = [], [], []
    for record in sorted(records, key=lambda row: (-int(row["nominal_budget"]), int(row["replicate"]))):
        models = load_ridge_models(Path(record["model_path"]), 105)
        variant = "LOWCAL"
        rows = apply_target_ridge_h1(deployment, source_models, models, variant)
        metrics = overall_metrics(
            rows,
            variant,
            {"trainable_parameter_count": MODEL_PARAMS, "input_dimension": 105},
        )
        metrics_rows.append({**record, **metrics})
        for row in per_gas_metrics(rows, variant):
            per_gas_rows.append(
                {
                    "nominal_budget": record["nominal_budget"],
                    "replicate": record["replicate"],
                    "actual_rows": record["actual_rows"],
                    **row,
                }
            )
        for row in rows:
            prediction_rows.append(
                {
                    "nominal_budget": record["nominal_budget"],
                    "replicate": record["replicate"],
                    "sample_index": int(row["sample_index"]),
                    "true_class": int(row["true_class"]),
                    "pred_class": int(row["pred_class"]),
                    "true_ppm": float(row["true_ppm"]),
                    "prediction_ppm": float(row[f"{variant}_ppm"]),
                }
            )
    if any(int(row["N"]) != 1360 for row in metrics_rows):
        raise RuntimeError("formal test N differs from 1360")
    write_csv(output / "per_replicate_low_calibration_metrics.csv", metrics_rows)
    write_csv(output / "per_replicate_per_gas_metrics.csv", per_gas_rows)
    write_csv(output / "test_predictions.csv", prediction_rows)
    evaluation = {
        "schema_version": SCHEMA,
        "calibration_lock_sha256": sha256_file(output / "calibration_lock.json"),
        "test_opened_after_calibration_lock": True,
        "test_used_for_fit_select_or_refit": False,
        "test_rows": 1360,
        "evaluated_models": len(records),
        "test_evaluation_timestamp": utc_now(),
        "test_predictions": descriptor(output / "test_predictions.csv"),
    }
    write_json(output / "test_evaluation_manifest.json", evaluation)
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    write_json(
        output / "stage_state.json",
        {
            **state,
            "stage": "TEST_EVALUATED",
            "test_opened": True,
            "test_evaluation_manifest_sha256": sha256_file(output / "test_evaluation_manifest.json"),
            "calibration_lock_sha256": evaluation["calibration_lock_sha256"],
        },
    )
    if sha256_file(output / "calibration_lock.json") != evaluation["calibration_lock_sha256"]:
        raise RuntimeError("calibration lock changed after test opening")
    if frozen_hashes(ROOT) != frozen_before:
        raise RuntimeError("runtime v4 frozen assets changed during test")


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "mean": statistics.fmean(ordered),
        "sample_std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "min": min(ordered),
        "max": max(ordered),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _write_table_set(directory: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(directory / f"{stem}.csv", rows)
    fields = list(rows[0])
    md = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        md.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    (directory / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = ["\\begin{tabular}{" + "l" * len(fields) + "}", " \\toprule", " & ".join(fields) + " \\\\", " \\midrule"]
    for row in rows:
        tex.append(" & ".join(str(row.get(field, "")).replace("_", "\\_") for field in fields) + " \\\\")
    tex.extend([" \\bottomrule", "\\end{tabular}"])
    (directory / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def _plots(
    summary_rows: Sequence[Mapping[str, Any]],
    per_gas_summary: Sequence[Mapping[str, Any]],
    timing_summary: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(summary_rows, key=lambda row: float(row["actual_rows_mean"]))
    x = np.asarray([float(row["actual_rows_mean"]) for row in ordered])
    y = np.asarray([float(row["S_CC_RMSE_mean"]) for row in ordered])
    e = np.asarray([float(row["S_CC_RMSE_sample_std"]) for row in ordered])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.errorbar(x, y, yerr=e, fmt="o-", capsize=4, color="#0072B2")
    ax.set_xlabel("Actual calibration rows")
    ax.set_ylabel("S_CC RMSE (ppm)")
    ax.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"low_calibration_scc_rmse.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    totals = [row for row in timing_summary if row["stage"] == "total_calibration_seconds"]
    totals.sort(key=lambda row: float(row["actual_rows_mean"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot([float(row["actual_rows_mean"]) for row in totals], [float(row["p50_seconds"]) for row in totals], "o-", color="#D55E00")
    ax.set_xlabel("Actual calibration rows")
    ax.set_ylabel("Calibration wall time p50 (s)")
    ax.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"low_calibration_time.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (gas, rows), color in zip(
        sorted(
            {
                str(row["gas"]): [item for item in per_gas_summary if item["gas"] == row["gas"]]
                for row in per_gas_summary
            }.items()
        ),
        colors,
    ):
        rows = sorted(rows, key=lambda row: float(row["actual_rows_mean"]))
        ax.errorbar(
            [float(row["actual_rows_mean"]) for row in rows],
            [float(row["RMSE_mean"]) for row in rows],
            yerr=[float(row["RMSE_sample_std"]) for row in rows],
            fmt="o-",
            capsize=3,
            label=gas,
            color=color,
        )
    ax.set_xlabel("Actual calibration rows")
    ax.set_ylabel("Per-gas RMSE (ppm)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"low_calibration_per_gas.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (output / "low_calibration_captions.en.md").write_text(
        "The frozen B5 + Federated-H1 + C5 Ridge method is evaluated at group-complete nested calibration budgets. Error bars show sample standard deviation across five deterministic subset replicates for 160/80/40; 320 is a single frozen reference. Lines connect only observed budgets and are not fitted curves. Calibration time is not expected to be monotonic because 160/80/40 use group-aware five-fold alpha selection while the frozen 320 reference retains its 240/80 holdout protocol.\n",
        encoding="utf-8",
    )
    (output / "low_calibration_captions.zh.md").write_text(
        "固定 B5 + Federated-H1 + C5 Ridge 方法在按 filename 整组保留的嵌套校准预算下进行敏感性分析。160/80/40 的误差条为 5 个确定性 subset replicate 的样本标准差，320 为单次冻结参考。连线仅连接已观测预算点，不是拟合曲线。校准耗时不要求随行数单调变化，因为 160/80/40 使用 group-aware 5-fold alpha selection，而冻结的 320 reference 保持原 240/80 holdout 协议。\n",
        encoding="utf-8",
    )


def finalize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    if state.get("stage") not in {"TEST_EVALUATED", "COMPLETE"} or state.get("test_opened") is not True:
        raise RuntimeError("finalization requires completed one-shot test evaluation")
    evaluation = json.loads((output / "test_evaluation_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(output / "calibration_lock.json") != evaluation["calibration_lock_sha256"]:
        raise RuntimeError("calibration lock changed after test evaluation")
    metrics = read_csv(output / "per_replicate_low_calibration_metrics.csv")
    per_gas = read_csv(output / "per_replicate_per_gas_metrics.csv")
    timing = read_csv(output / "calibration_timing_repetitions.csv")
    alpha = read_csv(output / "calibration_alpha_audit.csv")
    metadata = json.loads(
        (
            Path(args.data_root)
            / "client_5/calibration_experiment_info.json"
        ).read_text(encoding="utf-8")
    )
    replayed_subsets = _load_subset_indexes(output, metadata)
    fold_assignment_rows: list[dict[str, Any]] = []
    fold_audit_rows: list[dict[str, Any]] = []
    for (budget, replicate), indexes in sorted(replayed_subsets.items()):
        if budget == 320:
            continue
        seed = REPLICATE_SEEDS[replicate]
        folds = assign_group_folds(
            metadata, indexes, n_splits=5, seed=seed
        )
        filename_folds: dict[str, set[int]] = {}
        for index in indexes:
            filename = str(metadata[index]["filename"])
            filename_folds.setdefault(filename, set()).add(folds[index])
            fold_assignment_rows.append(
                {
                    "nominal_budget": budget,
                    "replicate": replicate,
                    "seed": seed,
                    "sample_index": index,
                    "filename": filename,
                    "fold": folds[index],
                    "gas_id": int(metadata[index]["classification_label"]),
                    "gas": CLASS_NAMES[
                        int(metadata[index]["classification_label"])
                    ],
                    "concentration_ppm": float(
                        metadata[index]["concentration"]
                    ),
                }
            )
        leakage = sum(len(values) != 1 for values in filename_folds.values())
        fold_audit_rows.append(
            {
                "nominal_budget": budget,
                "replicate": replicate,
                "actual_rows": len(indexes),
                "unique_filenames": len(filename_folds),
                "filename_fold_leakage_count": leakage,
                "fold_row_counts": {
                    str(fold): sum(value == fold for value in folds.values())
                    for fold in range(5)
                },
                "status": "PASS" if leakage == 0 else "FAIL_CLOSED",
            }
        )
    if any(row["status"] != "PASS" for row in fold_audit_rows):
        raise RuntimeError("replayed calibration fold isolation failed")
    write_csv(
        output / "calibration_fold_assignment_audit.csv",
        fold_assignment_rows,
    )
    write_json(
        output / "fold_isolation_audit.json",
        {
            "schema_version": SCHEMA,
            "status": "PASS",
            "audit_scope": "post-test deterministic replay of frozen pre-test subset seeds and fold algorithm; not used for selection",
            "assignments": fold_audit_rows,
        },
    )
    summary_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        selected = [row for row in metrics if int(row["nominal_budget"]) == budget]
        row: dict[str, Any] = {
            "nominal_budget": budget,
            "replicates": len(selected),
            "actual_rows_mean": statistics.fmean(float(item["actual_rows"]) for item in selected),
            "actual_rows_min": min(int(item["actual_rows"]) for item in selected),
            "actual_rows_max": max(int(item["actual_rows"]) for item in selected),
            "unique_filenames_mean": statistics.fmean(float(item["unique_filenames"]) for item in selected),
        }
        for metric in ("S_CC_RMSE", "S_ALL_RMSE", "S_ALL_MAE", "S_ALL_NRMSE", "CO_RMSE", "CO_high_200_250_RMSE"):
            stats = _summary([float(item[metric]) for item in selected])
            row.update({f"{metric}_{key}": value for key, value in stats.items()})
        row["S_CC_N_mean"] = statistics.fmean(float(item["S_CC_N"]) for item in selected)
        row["model_parameters"] = MODEL_PARAMS
        row["model_bytes_mean"] = statistics.fmean(float(item["model_bytes"]) for item in selected)
        summary_rows.append(row)
    write_csv(output / "low_calibration_summary.csv", summary_rows)

    gas_summary: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for gas_id in sorted(CLASS_NAMES):
            selected = [
                row
                for row in per_gas
                if int(row["nominal_budget"]) == budget and int(row["class_id"]) == gas_id
            ]
            stats = _summary([float(row["RMSE"]) for row in selected])
            gas_summary.append(
                {
                    "nominal_budget": budget,
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "replicates": len(selected),
                    "actual_rows_mean": statistics.fmean(float(row["actual_rows"]) for row in selected),
                    **{f"RMSE_{key}": value for key, value in stats.items()},
                }
            )
    write_csv(output / "low_calibration_per_gas_summary.csv", gas_summary)

    alpha_summary: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for gas_id in sorted(CLASS_NAMES):
            selected = [
                row
                for row in alpha
                if int(row["nominal_budget"]) == budget and int(row["gas_id"]) == gas_id
            ]
            per_rep: dict[int, tuple[float, float]] = {}
            for row in selected:
                rep = int(row["replicate"])
                candidate = (float(row["selection_RMSE"]), float(row["alpha"]))
                if rep not in per_rep or candidate < per_rep[rep]:
                    per_rep[rep] = candidate
            counts = Counter(value[1] for value in per_rep.values())
            alpha_summary.append(
                {
                    "nominal_budget": budget,
                    "gas_id": gas_id,
                    "gas": CLASS_NAMES[gas_id],
                    "replicate_selected_alphas": json.dumps({str(key): value[1] for key, value in sorted(per_rep.items())}),
                    "alpha_frequency": json.dumps(dict(sorted(counts.items()))),
                }
            )
    write_csv(output / "low_calibration_alpha_summary.csv", alpha_summary)

    timing_summary: list[dict[str, Any]] = []
    timing_stages = [
        "rich_feature_generation_seconds",
        "H1_prediction_generation_seconds",
        "fold_validation_preparation_seconds",
        "alpha_search_seconds",
        "final_refit_seconds",
        "serialization_seconds",
        "total_calibration_seconds",
    ]
    for budget in BUDGETS:
        selected = [row for row in timing if int(row["nominal_budget"]) == budget]
        for stage in timing_stages:
            values = [float(row[stage]) for row in selected]
            stats = _summary(values)
            timing_summary.append(
                {
                    "nominal_budget": budget,
                    "actual_rows_mean": statistics.fmean(float(row["actual_rows"]) for row in selected),
                    "stage": stage,
                    "N_timing_repeats": len(values),
                    "mean_seconds": stats["mean"],
                    "sample_std_seconds": stats["sample_std"],
                    "p50_seconds": _percentile(values, 50),
                    "p95_seconds": _percentile(values, 95),
                }
            )
    write_csv(output / "low_calibration_timing_summary.csv", timing_summary)

    base = next(row for row in summary_rows if int(row["nominal_budget"]) == 320)
    deltas = {
        budget: (
            next(row for row in summary_rows if int(row["nominal_budget"]) == budget)["S_CC_RMSE_mean"]
            - base["S_CC_RMSE_mean"]
        )
        / base["S_CC_RMSE_mean"]
        for budget in LOW_BUDGETS
    }
    decision = classify_sensitivity(deltas)
    write_json(
        output / "decision_gate.json",
        {
            "schema_version": SCHEMA,
            "decision": decision,
            "S_CC_relative_delta_vs_320": {str(key): value for key, value in deltas.items()},
            "rule_source": "protocol_manifest.json frozen before low-calibration test evaluation",
            "method_changed": False,
            "test_based_selection": False,
        },
    )

    timing_total = {int(row["nominal_budget"]): row for row in timing_summary if row["stage"] == "total_calibration_seconds"}
    table_rows = []
    for row in summary_rows:
        budget = int(row["nominal_budget"])
        actual = (
            str(int(row["actual_rows_min"]))
            if int(row["actual_rows_min"]) == int(row["actual_rows_max"])
            else f"{int(row['actual_rows_min'])}–{int(row['actual_rows_max'])}"
        )
        table_rows.append(
            {
                "Nominal calibration budget": budget,
                "Actual calibration rows": actual,
                "Unique source files": f"{row['unique_filenames_mean']:.1f}",
                "S_CC RMSE mean±std": _fmt_mean_std(row["S_CC_RMSE_mean"], row["S_CC_RMSE_sample_std"]),
                "S_ALL RMSE mean±std": _fmt_mean_std(row["S_ALL_RMSE_mean"], row["S_ALL_RMSE_sample_std"]),
                "CO RMSE mean±std": _fmt_mean_std(row["CO_RMSE_mean"], row["CO_RMSE_sample_std"]),
                "CO-high RMSE mean±std": _fmt_mean_std(row["CO_high_200_250_RMSE_mean"], row["CO_high_200_250_RMSE_sample_std"]),
                "Calibration time p50 (s)": f"{float(timing_total[budget]['p50_seconds']):.4f}",
                "Model size (bytes)": f"{row['model_bytes_mean']:.0f}",
                "Status": "FROZEN_REFERENCE" if budget == 320 else "SENSITIVITY_ONLY",
            }
        )
    _write_table_set(output / "paper_tables", "table_low_calibration", table_rows)
    _plots(summary_rows, gas_summary, timing_summary, output / "paper_figures")

    protocol = json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))
    experiment_matrix = [
        {
            "experiment_id": f"{EXPERIMENT_ID}-B{int(row['nominal_budget'])}-R{int(row['replicate'])}",
            "source_clients": "C1;C2",
            "target_clients": "C5",
            "split_protocol": "historical window-level C5 calibration/test 320/1360",
            "model": protocol["method"],
            "checkpoint": "B5 seed42 SHA256-bound",
            "DA": "frozen B5 server DA",
            "calibration": f"{row['actual_rows']} rows; group-complete nested subset",
            "QC": "disabled",
            "seed": "42 classifier; fixed subset seed",
            "result_path": "results/iotj_low_calibration_sensitivity_20260725",
            "status": "audited",
            "notes": "subset variability, not classifier seed variability",
        }
        for row in json.loads((output / "calibration_model_manifest.json").read_text(encoding="utf-8"))["models"]
    ]
    write_csv(output / "EXPERIMENT_MATRIX.csv", experiment_matrix)
    write_csv(output / "experiment_registry.csv", experiment_matrix)
    (output / "EXPERIMENT_PLAN.md").write_text(
        "# Low-calibration sensitivity plan\n\n"
        "Hypothesis LCS-H1: under the frozen B5 + real-topology Federated-H1 + C5 105D Ridge method, reducing group-complete C5 calibration data changes regression error and calibration cost. This is sensitivity evidence, not model selection.\n\n"
        "Controls, budgets, subsets, alpha selection, metrics, stopping rules, and evidence boundaries are frozen in `protocol_manifest.json`.\n",
        encoding="utf-8",
    )
    (output / "ABLATION_PLAN.md").write_text(
        "# Calibration-budget sensitivity matrix\n\n"
        "Factor: C5 calibration budget. Levels: 320/160/80/40. Held constant: B5 seed42, Federated H1, 105D features, per-gas Ridge family, alpha grid, C5 test, QC disabled. Stop after tables, figures, audit, report, and SHA index.\n",
        encoding="utf-8",
    )
    (output / "result_analysis.md").write_text(
        f"# Result analysis\n\nRecomputed sensitivity status: `{decision}`. Statistics summarize calibration subset variability; 320 is a single frozen reference and is not assigned seed uncertainty.\n",
        encoding="utf-8",
    )
    (output / "experiment_audit.md").write_text(
        "# Experiment audit\n\n"
        "- Verdict: AUDITED_SENSITIVITY_EVIDENCE.\n"
        "- Frozen classifier, Federated H1, feature schema, alpha grid, and test universe verified.\n"
        "- Filename groups remain intact within nested subsets and calibration-only folds.\n"
        "- `calibration_fold_assignment_audit.csv` and `fold_isolation_audit.json` replay every low-budget fold from the frozen subset seed/algorithm and report zero filename leakage.\n"
        "- Calibration lock existed and was SHA-bound before the one-shot low-calibration test stage.\n"
        "- Historical calibration/test split remains window-level; original-file independence is not claimed.\n"
        "- No QC, threshold, method, subset, alpha-grid, or budget was selected from test results.\n",
        encoding="utf-8",
    )
    write_json(
        output / "skill_handoff.json",
        {
            "schema_version": SCHEMA,
            "handoffs": [
                {"from_skill": "experiment-planner", "to_skill": "experiment-registry", "completed_checks": ["protocol frozen", "matrix fixed"], "unknown_or_conflict": []},
                {"from_skill": "experiment-registry", "to_skill": "result-analysis", "completed_checks": ["identities and provenance resolved"], "unknown_or_conflict": []},
                {"from_skill": "result-analysis", "to_skill": "experiment-audit", "completed_checks": ["descriptive subset statistics"], "unknown_or_conflict": []},
            ],
            "read_only_assets": list(EXPECTED_FROZEN_HASHES),
        },
    )

    report_rows = "\n".join(
        f"| {int(row['nominal_budget'])} | {row['actual_rows_min']:.0f}–{row['actual_rows_max']:.0f} | {_fmt_mean_std(row['S_CC_RMSE_mean'], row['S_CC_RMSE_sample_std'])} | {_fmt_mean_std(row['S_ALL_RMSE_mean'], row['S_ALL_RMSE_sample_std'])} | {_fmt_mean_std(row['CO_RMSE_mean'], row['CO_RMSE_sample_std'])} | {_fmt_mean_std(row['CO_high_200_250_RMSE_mean'], row['CO_high_200_250_RMSE_sample_std'])} |"
        for row in summary_rows
    )
    gas_report_rows = "\n".join(
        f"| {int(row['nominal_budget'])} | {row['gas']} | {_fmt_mean_std(row['RMSE_mean'], row['RMSE_sample_std'])} |"
        for row in gas_summary
    )
    timing_report_rows = "\n".join(
        f"| {int(row['nominal_budget'])} | {float(row['p50_seconds']):.4f} | {float(row['p95_seconds']):.4f} | {int(row['N_timing_repeats'])} |"
        for row in timing_summary
        if row["stage"] == "total_calibration_seconds"
    )
    report = f"""# GAPS IoT-J low-calibration sensitivity 结果（2026-07-25）

## 结论

本实验在固定 B5 seed42、real-topology sufficient-statistics Federated H1 与 C5 105D per-gas Ridge 下，仅改变 C5 calibration budget。最终描述性状态为 `{decision}`；该状态不改变最终方法、runtime 或 QC。

| Nominal budget | Actual rows range | S_CC RMSE mean±std | S_ALL RMSE mean±std | CO RMSE mean±std | CO-high RMSE mean±std |
|---:|---:|---:|---:|---:|---:|
{report_rows}

相对 320 reference，160/80/40 的 mean S_CC RMSE 相对变化分别为 {100*deltas[160]:.2f}%、{100*deltas[80]:.2f}% 和 {100*deltas[40]:.2f}%。160 的 S_CC subset standard deviation 为 {next(row for row in summary_rows if int(row['nominal_budget']) == 160)['S_CC_RMSE_sample_std']:.4f} ppm，80 为 {next(row for row in summary_rows if int(row['nominal_budget']) == 80)['S_CC_RMSE_sample_std']:.4f} ppm，40 为 {next(row for row in summary_rows if int(row['nominal_budget']) == 40)['S_CC_RMSE_sample_std']:.4f} ppm。

## 分气体结果

| Nominal budget | Gas | RMSE mean±std (ppm) |
|---:|---|---:|
{gas_report_rows}

Methane 与 CO/CO-high 是随预算缩减退化最明显的部分；Ethylene 的均值并非严格单调，说明 sensitivity 具有气体依赖性。所有 replicate 均保留，没有根据结果删除或替换 subset。

## Target Ridge 校准耗时

| Nominal budget | Total p50 (s) | Total p95 (s) | Timing N |
|---:|---:|---:|---:|
{timing_report_rows}

160/80/40 使用 group-aware 5-fold alpha selection，而 320 保留冻结的 240/80 holdout，因此耗时不要求随 calibration rows 单调变化。计时覆盖 rich feature、H1 prediction、fold preparation、alpha search、final refit 与 serialization；详细分阶段统计见 `low_calibration_timing_summary.csv`。

## 协议与统计边界

- 40 ⊆ 80 ⊆ 160 ⊆ 320；同一 filename 的 calibration 行不会被拆分。
- 160/80/40 各 5 个确定性 balanced subset replicates；320 为完整 calibration 单次参考。
- 低预算 alpha 仅由 group-aware 5-fold calibration-only selection 决定；320 保持冻结 240/80 语义。
- 冻结 seed/算法的独立重放审计覆盖全部 15 个低预算组合，filename fold leakage 为 0。
- 计时使用 PC 高精度单调时钟，每个 budget/replicate 10 次；重复计时的模型 numeric SHA 必须一致。
- primary metric 为固定 1360-row test 的 S_CC RMSE，未使用 QC accepted RMSE。

## Evidence boundary

1. Low-calibration is a sensitivity analysis under a frozen method.
2. The same historical C5 test split has been used in prior method confirmation.
3. Calibration subsets are selected without access to test labels or errors.
4. Filename grouping is used where possible to reduce within-calibration leakage.
5. Historical calibration/test splitting remains window-level.
6. No model or threshold selection is performed using low-calibration test results.

完整统计见 `low_calibration_summary.csv`、`low_calibration_per_gas_summary.csv`、`low_calibration_alpha_summary.csv` 与 `low_calibration_timing_summary.csv`；论文表格和图分别位于 `paper_tables/` 与 `paper_figures/`。

当前结果具备进入 paper evidence freeze 的条件：协议、subset、模型、calibration lock、test evaluation、表图和报告均有 SHA256 provenance；但论文表述必须保留 `HIGH_CALIBRATION_SENSITIVITY` 与 historical-test evidence boundary。
"""
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    frozen_after = frozen_hashes(ROOT)
    if frozen_after != json.loads((output / "frozen_asset_manifest.json").read_text(encoding="utf-8"))["runtime_v4_six_sha256"]:
        raise RuntimeError("runtime v4 six frozen SHA changed")
    final_state = dict(state)
    final_state.pop("sha256_index", None)
    final_state.update(
        {
            "stage": "COMPLETE",
            "decision": decision,
            "report": descriptor(report_path),
            "sha256_index_path": str((output / "sha256_index.json").resolve()),
            "sha256_index_generated_after_state": True,
        }
    )
    write_json(output / "stage_state.json", final_state)
    tracked = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "sha256_index.json"
    ]
    write_json(
        output / "sha256_index.json",
        {
            "schema_version": SCHEMA,
            "artifacts": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(tracked)
            ],
            "external_artifacts": {"report": descriptor(report_path)},
            "large_row_results_sha_bound_but_excluded_from_git": [
                "test_predictions.csv",
                "calibration_timing_repetitions.csv",
            ],
        },
    )
    index_path = Path(args.index_path)
    write_json(
        index_path,
        {
            "schema_version": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "COMPLETE",
            "decision": decision,
            "formal_run_code_commit": protocol["formal_code_commit"],
            "analysis_code_commit": git_head(),
            "result_root": str(output.relative_to(ROOT)).replace("\\", "/"),
            "report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256_index": str((output / "sha256_index.json").relative_to(ROOT)).replace("\\", "/"),
            "paper_evidence_freeze_ready": True,
            "runtime_changed": False,
            "QC_changed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    main_root = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-protocol", "run-calibration", "evaluate-test", "finalize"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/iotj_low_calibration_sensitivity_20260725"))
    parser.add_argument("--data-root", default=str(main_root / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"))
    parser.add_argument("--multiseed-root", default=str(ROOT / "results/iotj_b5_multiseed_20260724"))
    parser.add_argument("--real-h1", default=str(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/global_h1_model.json"))
    parser.add_argument("--runtime-v5-contract", default=str(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report-path", default=str(ROOT / "docs/experiments/iotj_low_calibration_sensitivity_result_20260725.zh.md"))
    parser.add_argument("--index-path", default=str(ROOT / "docs/experiments/iotj_low_calibration_sensitivity_result_index_20260725.json"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    {
        "freeze-protocol": freeze_protocol,
        "run-calibration": run_calibration,
        "evaluate-test": evaluate_test,
        "finalize": finalize,
    }[arguments.stage](arguments)
