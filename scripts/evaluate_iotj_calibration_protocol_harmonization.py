"""Post-freeze calibration-protocol harmonization audit for GAPS IoT-J.

This workflow compares two calibration-budget tracks without changing the
frozen B5 + Federated-H1 + 105D per-gas Ridge method:

* G: filename-group-aware five-fold alpha selection at every budget.
* H: the historical window-level 240/80 holdout, downsized inside its frozen
  fit and validation pools.

The workflow is staged so that every calibration model and reused result is
hash-bound before the already-used C5 test is evaluated descriptively.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_regression_head_ablation import CLASS_NAMES, RidgeHead, fit_ridge, load_split
from scripts.build_iotj_runtime_v5_candidate import (
    load_ridge_models,
    models_payload,
    prepare_rows,
    seed42_checkpoint,
)
from scripts.evaluate_iotj_b5_regression_multiseed import classifier_routes
from scripts.evaluate_iotj_h1_federated_ridge_equivalence import (
    RIDGE_ALPHAS,
    apply_target_ridge_h1,
    frozen_hashes,
    sha256_file,
)
from scripts.evaluate_iotj_low_calibration_sensitivity import (
    H1_FEATURE,
    MODEL_PARAMS,
    REPLICATE_SEEDS,
    _attach_h1,
    _build_base_rows,
    _model_numeric_sha,
    assign_group_folds,
    descriptor,
    read_csv,
    write_csv,
    write_json,
)
from scripts.evaluate_iotj_source_prior_target_head_factorial import (
    overall_metrics,
    per_gas_metrics,
)


SCHEMA = "iotj.calibration_protocol_harmonization.v1"
EXPERIMENT_ID = "IOTJ-CALIBRATION-PROTOCOL-HARMONIZATION-20260726"
BUDGETS = (320, 160, 80, 40)
LOW_BUDGETS = (160, 80, 40)
FIT_TARGETS = {160: 120, 80: 60, 40: 30}
VAL_TARGETS = {160: 40, 80: 20, 40: 10}
DEGRADATION_THRESHOLD = 0.10
PROTOCOL_DIFFERENCE_THRESHOLD = 0.20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.mkdir(parents=True)


def _metadata(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.data_root) / "client_5/calibration_experiment_info.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 320:
        raise RuntimeError("calibration universe must contain 320 rows")
    return rows


def _historical_pools(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    rows = read_csv(Path(args.historical_validation))
    validation = sorted(int(row["sample_index"]) for row in rows)
    if len(validation) != 80 or len(set(validation)) != 80:
        raise RuntimeError("historical validation membership must contain 80 unique rows")
    fit = sorted(set(range(320)) - set(validation))
    if len(fit) != 240:
        raise RuntimeError("historical fit membership must contain 240 rows")
    return fit, validation


def _distribution(metadata: Sequence[Mapping[str, Any]], indexes: Sequence[int]) -> dict[str, Any]:
    filenames = {str(metadata[index]["filename"]) for index in indexes}
    gas = Counter(str(CLASS_NAMES[int(metadata[index]["classification_label"])]) for index in indexes)
    levels = Counter(
        f"{CLASS_NAMES[int(metadata[index]['classification_label'])]}:{float(metadata[index]['concentration']):g}"
        for index in indexes
    )
    return {
        "rows": len(indexes),
        "unique_filenames": len(filenames),
        "gas_counts": dict(sorted(gas.items())),
        "gas_concentration_counts": dict(sorted(levels.items())),
        "row_index_sha256": sha_payload(list(indexes)),
    }


def _balanced_order(
    metadata: Sequence[Mapping[str, Any]], indexes: Sequence[int], seed: int, pool: str
) -> list[int]:
    """Return one deterministic, balance-aware ordering; every prefix is nested."""
    remaining = set(int(index) for index in indexes)
    selected: list[int] = []
    gas_counts: Counter[int] = Counter()
    level_counts: Counter[tuple[int, float]] = Counter()
    full_gas = Counter(int(metadata[i]["classification_label"]) for i in indexes)
    full_level = Counter(
        (int(metadata[i]["classification_label"]), round(float(metadata[i]["concentration"]), 6))
        for i in indexes
    )
    total = len(indexes)
    while remaining:
        target = len(selected) + 1
        candidates = []
        for index in remaining:
            gas = int(metadata[index]["classification_label"])
            level = (gas, round(float(metadata[index]["concentration"]), 6))
            expected_gas = target * full_gas[gas] / total
            expected_level = target * full_level[level] / total
            balance = (
                abs((gas_counts[gas] + 1) - expected_gas) / max(1.0, expected_gas)
                + abs((level_counts[level] + 1) - expected_level) / max(1.0, expected_level)
            )
            tie = hashlib.sha256(f"{seed}|{pool}|{index}".encode()).hexdigest()
            candidates.append((balance, gas_counts[gas], level_counts[level], tie, index))
        index = min(candidates)[-1]
        selected.append(index)
        remaining.remove(index)
        gas = int(metadata[index]["classification_label"])
        level = (gas, round(float(metadata[index]["concentration"]), 6))
        gas_counts[gas] += 1
        level_counts[level] += 1
    return selected


def historical_nested_subsets(
    metadata: Sequence[Mapping[str, Any]],
    fit_pool: Sequence[int],
    validation_pool: Sequence[int],
    seed: int,
) -> dict[int, dict[str, list[int]]]:
    fit_order = _balanced_order(metadata, fit_pool, seed, "fit")
    val_order = _balanced_order(metadata, validation_pool, seed, "validation")
    result: dict[int, dict[str, list[int]]] = {}
    for budget in LOW_BUDGETS:
        result[budget] = {
            "fit": sorted(fit_order[: FIT_TARGETS[budget]]),
            "validation": sorted(val_order[: VAL_TARGETS[budget]]),
        }
    for larger, smaller in ((160, 80), (80, 40)):
        if not set(result[smaller]["fit"]) <= set(result[larger]["fit"]):
            raise RuntimeError("historical fit subsets are not nested")
        if not set(result[smaller]["validation"]) <= set(result[larger]["validation"]):
            raise RuntimeError("historical validation subsets are not nested")
    return result


def classify_protocol_sensitivity(
    group_relative: Mapping[int, float], historical_relative: Mapping[int, float]
) -> tuple[str, dict[str, Any]]:
    g_degraded = all(float(group_relative[b]) > DEGRADATION_THRESHOLD for b in (80, 40))
    h_degraded = all(float(historical_relative[b]) > DEGRADATION_THRESHOLD for b in (80, 40))
    max_gap = max(abs(float(group_relative[b]) - float(historical_relative[b])) for b in LOW_BUDGETS)
    if g_degraded and h_degraded:
        decision = (
            "BUDGET_SENSITIVITY_ROBUST_ACROSS_PROTOCOLS"
            if max_gap <= PROTOCOL_DIFFERENCE_THRESHOLD
            else "SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT"
        )
    else:
        decision = "SENSITIVITY_STRONGLY_PROTOCOL_DEPENDENT"
    return decision, {
        "groupaware_degraded_at_80_and_40": g_degraded,
        "historical_degraded_at_80_and_40": h_degraded,
        "max_absolute_relative_degradation_gap": max_gap,
    }


def _validate_prior(args: argparse.Namespace) -> dict[str, Any]:
    prior = Path(args.prior_lowcal)
    required = [
        "frozen_asset_manifest.json",
        "protocol_manifest.json",
        "subset_manifest.json",
        "fold_isolation_audit.json",
        "calibration_model_manifest.json",
        "per_replicate_low_calibration_metrics.csv",
        "per_replicate_per_gas_metrics.csv",
        "calibration_alpha_audit.csv",
        "sha256_index.json",
    ]
    for name in required:
        if not (prior / name).is_file():
            raise RuntimeError(f"missing prior official low-calibration asset: {name}")
    frozen = json.loads((prior / "frozen_asset_manifest.json").read_text(encoding="utf-8"))
    current = _frozen_assets(args)
    for name in ("classifier_B5_seed42", "federated_H1", "calibration_features", "calibration_labels",
                 "calibration_metadata", "test_features", "test_labels", "test_metadata"):
        if frozen["assets"][name]["sha256"] != current["assets"][name]["sha256"]:
            raise RuntimeError(f"prior low-calibration frozen asset differs: {name}")
    protocol = json.loads((prior / "protocol_manifest.json").read_text(encoding="utf-8"))
    if protocol["replicate_seeds"] != list(REPLICATE_SEEDS) or protocol["selection"]["alpha_grid"] != list(RIDGE_ALPHAS):
        raise RuntimeError("prior low-calibration seeds or alpha grid differ")
    fold = json.loads((prior / "fold_isolation_audit.json").read_text(encoding="utf-8"))
    if fold.get("status") != "PASS":
        raise RuntimeError("prior group-aware fold audit is not PASS")
    metrics = [row for row in read_csv(prior / "per_replicate_low_calibration_metrics.csv")
               if int(row["nominal_budget"]) in LOW_BUDGETS]
    if len(metrics) != 15:
        raise RuntimeError("prior group-aware lower-budget metrics must contain 15 rows")
    return {"status": "PASS", "files": {name: descriptor(prior / name) for name in required}}


def _frozen_assets(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    paths = {
        "classifier_B5_seed42": checkpoint,
        "federated_H1": Path(args.real_h1),
        "feature_schema_105d_contract": Path(args.runtime_v5_contract),
        "calibration_features": Path(args.data_root) / "client_5/calibration_features.npy",
        "calibration_labels": Path(args.data_root) / "client_5/calibration_regression_labels.npy",
        "calibration_metadata": Path(args.data_root) / "client_5/calibration_experiment_info.json",
        "test_features": Path(args.data_root) / "client_5/test_features.npy",
        "test_labels": Path(args.data_root) / "client_5/test_regression_labels.npy",
        "test_metadata": Path(args.data_root) / "client_5/test_experiment_info.json",
        "historical_validation_membership": Path(args.historical_validation),
        "historical_frozen_target_ridge": Path(args.historical_model),
    }
    return {
        "schema_version": SCHEMA,
        "assets": {key: descriptor(path) for key, path in paths.items()},
        "runtime_v4_six_sha256": frozen_hashes(ROOT),
        "runtime_v4_read_only": True,
    }


def freeze_protocol(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    require_new_output(output)
    metadata = _metadata(args)
    fit, validation = _historical_pools(args)
    fit_names = {str(metadata[i]["filename"]) for i in fit}
    val_names = {str(metadata[i]["filename"]) for i in validation}
    overlap = sorted(fit_names & val_names)
    historical = {
        "schema_version": SCHEMA,
        "protocol": "historical_window_level_240_fit_80_validation",
        "fit": _distribution(metadata, fit),
        "validation": _distribution(metadata, validation),
        "filename_overlap_count": len(overlap),
        "validation_filenames_overlapping_fit": len(overlap),
        "filename_overlap": overlap,
        "is_group_aware": False,
        "fit_row_indexes": fit,
        "validation_row_indexes": validation,
    }
    write_json(output / "historical_holdout_audit.json", historical)
    frozen = _frozen_assets(args)
    write_json(output / "frozen_asset_manifest.json", frozen)
    prior = _validate_prior(args)
    write_json(output / "prior_groupaware_reuse_audit.json", prior)
    protocol = {
        "schema_version": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "formal_code_commit": git_head(),
        "method": "frozen B5 seed42 + Federated H1 + 105D C5 per-gas Ridge",
        "budgets": list(BUDGETS),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "track_G": {
            "name": "harmonized filename-group-aware 5-fold",
            "selection": "per-gas alpha by calibration-only OOF RMSE",
            "320": "full 320 rows; five fold-assignment replicates",
            "160_80_40": "read-only reuse of SHA-verified official low-calibration results",
            "final_refit": "entire current calibration subset",
        },
        "track_H": {
            "name": "historical window-level holdout downsizing",
            "budgets": {"320": "240+80", "160": "120+40", "80": "60+20", "40": "30+10"},
            "membership": "fit subset only from frozen 240 fit pool; validation subset only from frozen 80 validation pool",
            "nested": "40 subset_of 80 subset_of 160 separately inside fit and validation pools",
            "selection": "per-gas alpha by corresponding historical fit/validation split",
            "final_refit": "all fit+validation rows at current budget",
        },
        "alpha_grid": list(RIDGE_ALPHAS),
        "decision_rule": {
            "degradation_metric": "relative mean S_CC RMSE increase versus the same track's 320 baseline",
            "clearly_degraded": "both budget 80 and 40 exceed 10%",
            "obvious_protocol_magnitude_difference": "maximum absolute lower-budget curve gap exceeds 20 percentage points",
            "A": "both tracks clearly degraded and maximum curve gap <=20 percentage points",
            "B": "both tracks clearly degraded and maximum curve gap >20 percentage points",
            "C": "mainly group-aware track clearly degraded; operational fallback when both-track criterion is not met",
        },
        "test": {
            "rows": 1360,
            "previously_used": True,
            "opened_by_this_workflow": False,
            "descriptive_only": True,
            "used_for_selection": False,
        },
        "evidence_boundary": [
            "post-freeze calibration-protocol harmonization audit",
            "historical 320 retains frozen 240/80 window-level semantics",
            "group-aware track uses the same five-fold selection protocol at all budgets",
            "historical downsizing preserves original fit/validation pool membership",
            "same previously used C5 test is evaluated descriptively",
            "no protocol, model, subset, fold, alpha, or method is selected from test results",
        ],
    }
    write_json(output / "protocol_manifest.json", protocol)
    write_json(output / "stage_state.json", {
        "schema_version": SCHEMA,
        "stage": "PROTOCOL_FROZEN",
        "test_opened": False,
        "protocol_sha256": sha256_file(output / "protocol_manifest.json"),
        "frozen_asset_sha256": sha256_file(output / "frozen_asset_manifest.json"),
        "historical_holdout_audit_sha256": sha256_file(output / "historical_holdout_audit.json"),
        "prior_groupaware_reuse_audit_sha256": sha256_file(output / "prior_groupaware_reuse_audit.json"),
    })


def _fit_groupaware(
    oracle: Sequence[Mapping[str, Any]], deployment: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]], indexes: Sequence[int], seed: int,
) -> tuple[dict[int, RidgeHead], list[dict[str, Any]]]:
    folds = assign_group_folds(metadata, indexes, n_splits=5, seed=seed)
    by_index = {int(row["sample_index"]): row for row in deployment}
    feature_names = sorted(oracle[0]["feature_dict"])
    models: dict[int, RidgeHead] = {}
    audit: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        rows = [row for row in oracle if int(row["true_class"]) == gas_id]
        scores = []
        for alpha in RIDGE_ALPHAS:
            sse, n = 0.0, 0
            for fold in range(5):
                train = [row for row in rows if folds[int(row["sample_index"])] != fold]
                val_seed = [row for row in rows if folds[int(row["sample_index"])] == fold]
                validation = [by_index[int(row["sample_index"])] for row in val_seed]
                if not train or not validation:
                    raise RuntimeError(f"empty group-aware fold for gas {gas_id}")
                model = fit_ridge(train, feature_names, float(alpha))
                y = np.asarray([float(row["true_ppm"]) for row in validation])
                error = model.predict(validation, clip=True) - y
                sse += float(error @ error)
                n += len(validation)
            rmse = math.sqrt(sse / n)
            scores.append((rmse, float(alpha)))
            audit.append({"track": "G", "nominal_budget": 320, "seed": seed,
                          "gas_id": gas_id, "gas": CLASS_NAMES[gas_id], "alpha": float(alpha),
                          "selection_RMSE": rmse, "validation_N": n})
        chosen = min(scores, key=lambda item: (item[0], RIDGE_ALPHAS.index(item[1])))[1]
        models[gas_id] = fit_ridge(rows, feature_names, chosen)
    return models, audit


def _fit_historical(
    oracle: Sequence[Mapping[str, Any]], deployment: Sequence[Mapping[str, Any]],
    fit_indexes: Sequence[int], val_indexes: Sequence[int], budget: int, seed: int,
) -> tuple[dict[int, RidgeHead], list[dict[str, Any]]]:
    oracle_by_index = {int(row["sample_index"]): row for row in oracle}
    deployment_by_index = {int(row["sample_index"]): row for row in deployment}
    feature_names = sorted(oracle[0]["feature_dict"])
    models: dict[int, RidgeHead] = {}
    audit: list[dict[str, Any]] = []
    for gas_id in sorted(CLASS_NAMES):
        fit_rows = [oracle_by_index[i] for i in fit_indexes if int(oracle_by_index[i]["true_class"]) == gas_id]
        val_rows = [deployment_by_index[i] for i in val_indexes if int(oracle_by_index[i]["true_class"]) == gas_id]
        if not fit_rows or not val_rows:
            raise RuntimeError(f"empty historical fit/validation gas stratum {gas_id}")
        scores = []
        for alpha in RIDGE_ALPHAS:
            model = fit_ridge(fit_rows, feature_names, float(alpha))
            y = np.asarray([float(row["true_ppm"]) for row in val_rows])
            error = model.predict(val_rows, clip=True) - y
            rmse = math.sqrt(float(error @ error) / len(val_rows))
            scores.append((rmse, float(alpha)))
            audit.append({"track": "H", "nominal_budget": budget, "seed": seed,
                          "gas_id": gas_id, "gas": CLASS_NAMES[gas_id], "alpha": float(alpha),
                          "selection_RMSE": rmse, "fit_N": len(fit_rows), "validation_N": len(val_rows)})
        chosen = min(scores, key=lambda item: (item[0], RIDGE_ALPHAS.index(item[1])))[1]
        refit = [oracle_by_index[i] for i in sorted((*fit_indexes, *val_indexes))
                 if int(oracle_by_index[i]["true_class"]) == gas_id]
        models[gas_id] = fit_ridge(refit, feature_names, chosen)
    return models, audit


def run_calibration(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    if state["stage"] != "PROTOCOL_FROZEN" or state["test_opened"] is not False:
        raise RuntimeError("calibration requires frozen pre-test protocol")
    if sha256_file(output / "protocol_manifest.json") != state["protocol_sha256"]:
        raise RuntimeError("protocol drifted")
    frozen_before = frozen_hashes(ROOT)
    metadata = _metadata(args)
    fit_pool, val_pool = _historical_pools(args)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "calibration", torch.device(args.device), args.batch_size)
    source_models = load_ridge_models(Path(args.real_h1), 104)
    oracle, deployment = _build_base_rows(Path(args.data_root), list(range(320)), routes)
    oracle, deployment = _attach_h1(oracle, source_models), _attach_h1(deployment, source_models)
    records, alpha_rows, subset_records, balance_rows = [], [], [], []

    for replicate, seed in enumerate(REPLICATE_SEEDS):
        models, audit = _fit_groupaware(oracle, deployment, metadata, list(range(320)), seed)
        path = output / f"track_groupaware/models/budget_320/replicate_{replicate}/target_ridge_105d.json"
        write_json(path, models_payload(models, dimension=105, source=f"harmonized group-aware 320 seed={seed}"))
        records.append({"track": "G", "nominal_budget": 320, "replicate": replicate, "seed": seed,
                        "actual_rows": 320, "unique_filenames": 80, "model_path": str(path.resolve()),
                        "model_sha256": sha256_file(path), "model_numeric_sha256": _model_numeric_sha(models),
                        "model_bytes": path.stat().st_size})
        alpha_rows.extend({"replicate": replicate, **row} for row in audit)

    for replicate, seed in enumerate(REPLICATE_SEEDS):
        nested = historical_nested_subsets(metadata, fit_pool, val_pool, seed)
        for budget in LOW_BUDGETS:
            fit_indexes, val_indexes = nested[budget]["fit"], nested[budget]["validation"]
            if not set(fit_indexes) <= set(fit_pool) or not set(val_indexes) <= set(val_pool):
                raise RuntimeError("historical subset crossed frozen pools")
            models, audit = _fit_historical(oracle, deployment, fit_indexes, val_indexes, budget, seed)
            path = output / f"track_historical_holdout/models/budget_{budget}/replicate_{replicate}/target_ridge_105d.json"
            write_json(path, models_payload(models, dimension=105, source=f"historical holdout budget={budget} seed={seed}"))
            records.append({"track": "H", "nominal_budget": budget, "replicate": replicate, "seed": seed,
                            "actual_rows": budget,
                            "unique_filenames": len({str(metadata[i]["filename"]) for i in (*fit_indexes, *val_indexes)}),
                            "fit_rows": len(fit_indexes), "validation_rows": len(val_indexes),
                            "model_path": str(path.resolve()), "model_sha256": sha256_file(path),
                            "model_numeric_sha256": _model_numeric_sha(models), "model_bytes": path.stat().st_size})
            alpha_rows.extend({"replicate": replicate, **row} for row in audit)
            subset_records.append({"nominal_budget": budget, "replicate": replicate, "seed": seed,
                                   "fit_indexes": fit_indexes, "validation_indexes": val_indexes,
                                   "fit_sha256": sha_payload(fit_indexes), "validation_sha256": sha_payload(val_indexes)})
            for pool_name, indexes in (("fit", fit_indexes), ("validation", val_indexes)):
                dist = _distribution(metadata, indexes)
                balance_rows.append({"nominal_budget": budget, "replicate": replicate, "seed": seed,
                                     "pool": pool_name, "rows": len(indexes),
                                     "unique_filenames": dist["unique_filenames"],
                                     "gas_counts": json.dumps(dist["gas_counts"], sort_keys=True),
                                     "gas_concentration_counts": json.dumps(dist["gas_concentration_counts"], sort_keys=True)})

    prior_audit = _validate_prior(args)
    prior_manifest = json.loads((Path(args.prior_lowcal) / "calibration_model_manifest.json").read_text(encoding="utf-8"))
    reused = []
    for record in prior_manifest["models"]:
        if int(record["nominal_budget"]) not in LOW_BUDGETS:
            continue
        path = Path(record["model_path"])
        if sha256_file(path) != record["model_sha256"]:
            raise RuntimeError("prior reused model hash drifted")
        reused.append({"track": "G", **record, "reuse_source": str(Path(args.prior_lowcal).resolve())})
    write_json(output / "track_groupaware/groupaware_320_fold_manifest.json", {
        "schema_version": SCHEMA, "rows": 320, "filenames": 80,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "models": [row for row in records if row["track"] == "G"],
    })
    write_json(output / "track_groupaware/reused_lower_budget_manifest.json", {
        "schema_version": SCHEMA, "audit": prior_audit, "models": reused,
    })
    write_json(output / "track_historical_holdout/historical_subset_manifest.json", {
        "schema_version": SCHEMA, "fit_pool_sha256": sha_payload(fit_pool),
        "validation_pool_sha256": sha_payload(val_pool), "replicates": subset_records,
    })
    write_csv(output / "track_historical_holdout/historical_subset_balance_audit.csv", balance_rows)
    write_csv(output / "calibration_alpha_audit.csv", alpha_rows)
    write_json(output / "calibration_model_manifest.json", {
        "schema_version": SCHEMA, "new_models": records, "reused_groupaware_models": reused,
        "historical_320_model": descriptor(Path(args.historical_model)),
    })
    lock_assets = [
        output / "protocol_manifest.json", output / "frozen_asset_manifest.json",
        output / "historical_holdout_audit.json", output / "prior_groupaware_reuse_audit.json",
        output / "calibration_model_manifest.json",
        output / "track_groupaware/groupaware_320_fold_manifest.json",
        output / "track_groupaware/reused_lower_budget_manifest.json",
        output / "track_historical_holdout/historical_subset_manifest.json",
        output / "calibration_alpha_audit.csv",
    ] + [Path(row["model_path"]) for row in records]
    write_json(output / "calibration_lock.json", {
        "schema_version": SCHEMA, "created_at": utc_now(), "test_opened": False,
        "test_used_for_selection": False,
        "bound_assets": [descriptor(path) for path in lock_assets],
    })
    write_json(output / "stage_state.json", {**state, "stage": "CALIBRATION_LOCKED",
        "test_opened": False, "calibration_lock_sha256": sha256_file(output / "calibration_lock.json")})
    if frozen_hashes(ROOT) != frozen_before:
        raise RuntimeError("runtime v4 frozen assets changed")


def _require_lock(output: Path) -> dict[str, Any]:
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    if state["stage"] != "CALIBRATION_LOCKED" or state["test_opened"] is not False:
        raise RuntimeError("test requires calibration lock")
    if sha256_file(output / "calibration_lock.json") != state["calibration_lock_sha256"]:
        raise RuntimeError("calibration lock drifted")
    lock = json.loads((output / "calibration_lock.json").read_text(encoding="utf-8"))
    for item in lock["bound_assets"]:
        if descriptor(Path(item["path"])) != item:
            raise RuntimeError(f"locked asset drifted: {item['path']}")
    return lock


def _evaluate_model(
    deployment: Sequence[Mapping[str, Any]], source_models: Mapping[int, RidgeHead],
    model_path: Path, record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models = load_ridge_models(model_path, 105)
    rows = apply_target_ridge_h1(deployment, source_models, models, "HARM")
    metrics = overall_metrics(rows, "HARM", {"trainable_parameter_count": MODEL_PARAMS, "input_dimension": 105})
    gas = per_gas_metrics(rows, "HARM")
    return {**record, **metrics}, [{**record, **row} for row in gas]


def evaluate_test(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    lock = _require_lock(output)
    frozen_before = frozen_hashes(ROOT)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "test", torch.device(args.device), args.batch_size)
    _oracle, deployment = prepare_rows(Path(args.data_root), "test", routes)
    source_models = load_ridge_models(Path(args.real_h1), 104)
    manifest = json.loads((output / "calibration_model_manifest.json").read_text(encoding="utf-8"))
    group_metrics = [row for row in read_csv(Path(args.prior_lowcal) / "per_replicate_low_calibration_metrics.csv")
                     if int(row["nominal_budget"]) in LOW_BUDGETS]
    group_gas = [row for row in read_csv(Path(args.prior_lowcal) / "per_replicate_per_gas_metrics.csv")
                 if int(row["nominal_budget"]) in LOW_BUDGETS]
    for row in group_metrics:
        row.update({"track": "G", "reuse": "official_lowcal_20260725"})
    for row in group_gas:
        row.update({"track": "G", "reuse": "official_lowcal_20260725"})
    historical_metrics, historical_gas = [], []
    for record in manifest["new_models"]:
        metric, gas = _evaluate_model(deployment, source_models, Path(record["model_path"]), record)
        if int(metric["N"]) != 1360:
            raise RuntimeError("test N differs from 1360")
        (group_metrics if record["track"] == "G" else historical_metrics).append(metric)
        (group_gas if record["track"] == "G" else historical_gas).extend(gas)

    prior_metrics = read_csv(Path(args.prior_lowcal) / "per_replicate_low_calibration_metrics.csv")
    prior_gas = read_csv(Path(args.prior_lowcal) / "per_replicate_per_gas_metrics.csv")
    h320 = dict(next(row for row in prior_metrics if int(row["nominal_budget"]) == 320))
    h320.update({"track": "H", "replicate": 0, "reuse": "frozen_historical_320"})
    historical_metrics.append(h320)
    for row in prior_gas:
        if int(row["nominal_budget"]) == 320:
            item = dict(row)
            item.update({"track": "H", "replicate": 0, "reuse": "frozen_historical_320"})
            historical_gas.append(item)
    write_csv(output / "track_groupaware/groupaware_320_per_replicate.csv",
              [row for row in group_metrics if int(row["nominal_budget"]) == 320])
    write_csv(output / "track_groupaware/groupaware_per_replicate.csv", group_metrics)
    write_csv(output / "track_groupaware/groupaware_per_gas.csv", group_gas)
    write_csv(output / "track_historical_holdout/historical_per_replicate.csv", historical_metrics)
    write_csv(output / "track_historical_holdout/historical_per_gas.csv", historical_gas)
    write_json(output / "test_evaluation_manifest.json", {
        "schema_version": SCHEMA, "calibration_lock_sha256": sha256_file(output / "calibration_lock.json"),
        "test_opened_after_calibration_lock": True, "test_previously_used": True,
        "test_used_for_selection": False, "test_rows": 1360,
        "timestamp": utc_now(), "new_models_evaluated": len(manifest["new_models"]),
    })
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    write_json(output / "stage_state.json", {**state, "stage": "TEST_EVALUATED", "test_opened": True})
    if frozen_hashes(ROOT) != frozen_before or frozen_hashes(ROOT) != json.loads(
        (output / "frozen_asset_manifest.json").read_text(encoding="utf-8"))["runtime_v4_six_sha256"]:
        raise RuntimeError("runtime v4 frozen assets changed")


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def _summaries(rows: Sequence[Mapping[str, Any]], gas_rows: Sequence[Mapping[str, Any]], track: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, gas_summary = [], []
    metrics = ("S_CC_RMSE", "S_ALL_RMSE", "S_ALL_MAE", "S_ALL_NRMSE", "CO_RMSE", "CO_high_200_250_RMSE")
    for budget in BUDGETS:
        selected = [row for row in rows if int(row["nominal_budget"]) == budget]
        item: dict[str, Any] = {"track": track, "nominal_budget": budget, "replicates": len(selected),
                                "S_CC_N": statistics.fmean(float(row["S_CC_N"]) for row in selected)}
        for metric in metrics:
            for key, value in _stats([float(row[metric]) for row in selected]).items():
                item[f"{metric}_{key}"] = value
        summary.append(item)
        for gas_id in sorted(CLASS_NAMES):
            selected_gas = [row for row in gas_rows if int(row["nominal_budget"]) == budget and int(row["class_id"]) == gas_id]
            stat = _stats([float(row["RMSE"]) for row in selected_gas])
            gas_summary.append({"track": track, "nominal_budget": budget, "gas_id": gas_id,
                                "gas": CLASS_NAMES[gas_id], "replicates": len(selected_gas),
                                **{f"RMSE_{key}": value for key, value in stat.items()}})
    return summary, gas_summary


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _write_tables(output: Path, rows: list[dict[str, Any]]) -> None:
    directory = output / "paper_tables"
    write_csv(directory / "table_calibration_protocol_comparison.csv", rows)
    headers = list(rows[0])
    md = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    md += "\n".join("| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "table_calibration_protocol_comparison.md").write_text(md, encoding="utf-8")
    tex = "\\begin{tabular}{" + "l" * len(headers) + "}\n" + " & ".join(headers) + " \\\\\n\\hline\n"
    tex += "\n".join(" & ".join(str(row[h]) for h in headers) + " \\\\" for row in rows)
    tex += "\n\\end{tabular}\n"
    (directory / "table_calibration_protocol_comparison.tex").write_text(tex, encoding="utf-8")


def _plots(output: Path, group: Sequence[Mapping[str, Any]], historical: Sequence[Mapping[str, Any]]) -> None:
    directory = output / "paper_figures"
    directory.mkdir(parents=True, exist_ok=True)
    maps = ({int(row["nominal_budget"]): row for row in group}, {int(row["nominal_budget"]): row for row in historical})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, data, title in zip(axes, maps, ("A  Group-aware 5-fold", "B  Historical holdout")):
        x = list(BUDGETS[::-1])
        y = [float(data[b]["S_CC_RMSE_mean"]) for b in x]
        e = [float(data[b]["S_CC_RMSE_sample_std"]) for b in x]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
        ax.set_title(title); ax.set_xlabel("Actual calibration rows"); ax.grid(alpha=.25)
    axes[0].set_ylabel("S_CC RMSE (ppm)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"calibration_protocol_sensitivity_comparison.{suffix}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    x = list(BUDGETS[::-1])
    delta = [float(maps[0][b]["S_CC_RMSE_mean"]) - float(maps[1][b]["S_CC_RMSE_mean"]) for b in x]
    ax.axhline(0, color="black", lw=.8); ax.plot(x, delta, marker="o")
    ax.set(xlabel="Calibration budget", ylabel="Group-aware − historical S_CC RMSE (ppm)")
    ax.grid(alpha=.25); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"calibration_protocol_delta.{suffix}", dpi=300)
    plt.close(fig)
    (directory / "captions.zh.md").write_text(
        "双轨校准预算敏感性。误差棒为样本标准差；historical 320 为固定参考，标准差记为 0。折线仅连接观测点，未拟合平滑曲线。\n",
        encoding="utf-8")
    (directory / "captions.en.md").write_text(
        "Two-track calibration-budget sensitivity. Error bars are sample standard deviations; historical 320 is a fixed reference (SD=0). Lines only connect observed points.\n",
        encoding="utf-8")


def _sha_index(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "sha256_index.json"):
        records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path), "git_policy": "SHA_ONLY" if "models" in path.parts else "COMMIT"})
    return {"schema_version": SCHEMA, "created_at": utc_now(), "files": records}


def finalize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    state = json.loads((output / "stage_state.json").read_text(encoding="utf-8"))
    if state["stage"] != "TEST_EVALUATED" or state["test_opened"] is not True:
        raise RuntimeError("finalize requires completed descriptive test evaluation")
    g_rows = read_csv(output / "track_groupaware/groupaware_per_replicate.csv")
    g_gas = read_csv(output / "track_groupaware/groupaware_per_gas.csv")
    h_rows = read_csv(output / "track_historical_holdout/historical_per_replicate.csv")
    h_gas = read_csv(output / "track_historical_holdout/historical_per_gas.csv")
    gs, gg = _summaries(g_rows, g_gas, "G")
    hs, hg = _summaries(h_rows, h_gas, "H")
    write_csv(output / "track_groupaware/groupaware_budget_summary.csv", gs)
    write_csv(output / "track_groupaware/groupaware_per_gas_summary.csv", gg)
    write_csv(output / "track_historical_holdout/historical_budget_summary.csv", hs)
    write_csv(output / "track_historical_holdout/historical_per_gas_summary.csv", hg)
    gm, hm = ({int(row["nominal_budget"]): row for row in gs},
              {int(row["nominal_budget"]): row for row in hs})
    ggmap = {(int(row["nominal_budget"]), row["gas"]): row for row in gg}
    hgmap = {(int(row["nominal_budget"]), row["gas"]): row for row in hg}
    table = []
    for budget in BUDGETS:
        table.append({
            "calibration_budget": budget,
            "historical_S_CC_mean_std": _fmt(hm[budget]["S_CC_RMSE_mean"], hm[budget]["S_CC_RMSE_sample_std"]),
            "groupaware_S_CC_mean_std": _fmt(gm[budget]["S_CC_RMSE_mean"], gm[budget]["S_CC_RMSE_sample_std"]),
            "protocol_delta_ppm": float(gm[budget]["S_CC_RMSE_mean"]) - float(hm[budget]["S_CC_RMSE_mean"]),
            "historical_S_ALL": _fmt(hm[budget]["S_ALL_RMSE_mean"], hm[budget]["S_ALL_RMSE_sample_std"]),
            "groupaware_S_ALL": _fmt(gm[budget]["S_ALL_RMSE_mean"], gm[budget]["S_ALL_RMSE_sample_std"]),
            "historical_CO_RMSE": hgmap[(budget, "CO")]["RMSE_mean"],
            "groupaware_CO_RMSE": ggmap[(budget, "CO")]["RMSE_mean"],
            "historical_CO_high_RMSE": hm[budget]["CO_high_200_250_RMSE_mean"],
            "groupaware_CO_high_RMSE": gm[budget]["CO_high_200_250_RMSE_mean"],
            "historical_Methane_RMSE": hgmap[(budget, "Methane")]["RMSE_mean"],
            "groupaware_Methane_RMSE": ggmap[(budget, "Methane")]["RMSE_mean"],
        })
    _write_tables(output, table)
    _plots(output, gs, hs)
    g_rel = {b: (float(gm[b]["S_CC_RMSE_mean"]) - float(gm[320]["S_CC_RMSE_mean"])) / float(gm[320]["S_CC_RMSE_mean"]) for b in LOW_BUDGETS}
    h_rel = {b: (float(hm[b]["S_CC_RMSE_mean"]) - float(hm[320]["S_CC_RMSE_mean"])) / float(hm[320]["S_CC_RMSE_mean"]) for b in LOW_BUDGETS}
    decision, audit = classify_protocol_sensitivity(g_rel, h_rel)
    write_json(output / "decision_gate.json", {
        "schema_version": SCHEMA, "decision": decision,
        "groupaware_relative_S_CC_delta": {str(k): v for k, v in g_rel.items()},
        "historical_relative_S_CC_delta": {str(k): v for k, v in h_rel.items()},
        "operational_rule_audit": audit, "method_changed": False,
        "existing_HIGH_CALIBRATION_SENSITIVITY_overwritten": False,
        "test_based_selection": False,
    })
    history = json.loads((output / "historical_holdout_audit.json").read_text(encoding="utf-8"))
    lines = "\n".join(
        f"| {b} | {_fmt(hm[b]['S_CC_RMSE_mean'], hm[b]['S_CC_RMSE_sample_std'])} | "
        f"{_fmt(gm[b]['S_CC_RMSE_mean'], gm[b]['S_CC_RMSE_sample_std'])} | "
        f"{float(gm[b]['S_CC_RMSE_mean'])-float(hm[b]['S_CC_RMSE_mean']):.4f} |"
        for b in BUDGETS)
    report = f"""# GAPS IoT-J calibration-protocol harmonization audit（2026-07-26）

## 结论

描述性判定为 `{decision}`。该判定不修改既有 `HIGH_CALIBRATION_SENSITIVITY` 记录、最终方法、runtime 或 QC。

| Calibration rows | Historical S_CC RMSE | Group-aware S_CC RMSE | G−H (ppm) |
|---:|---:|---:|---:|
{lines}

历史 240/80 审计：fit 240 行、validation 80 行；fit filename={history['fit']['unique_filenames']}，
validation filename={history['validation']['unique_filenames']}，跨池 filename overlap={history['filename_overlap_count']}。
因此历史轨迹是 window-level holdout，不能声称 original-file independent 或 group-aware。

Group-aware 320 的标准差来自 fold/alpha-selection variability；group-aware 160/80/40 来自
subset + fold variability；historical 160/80/40 来自 holdout subset variability；historical
320 是固定单次 reference。

## Evidence boundary

本工作是 post-freeze calibration-protocol harmonization audit。两条轨迹使用同一冻结 B5、
Federated H1、105D schema、Ridge family 与 alpha grid。相同且此前已经使用过的 C5 1360 行
test 仅作描述性评估；没有根据 test 选择 protocol、subset、fold、alpha 或模型。
"""
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_json(output / "experiment_audit.json", {
        "schema_version": SCHEMA, "verdict": "AUDITED_POST_FREEZE_SENSITIVITY_EVIDENCE",
        "historical_filename_overlap": history["filename_overlap_count"],
        "calibration_lock_before_test": True, "test_selection": False,
        "runtime_v4_six_sha_unchanged": frozen_hashes(ROOT) == json.loads(
            (output / "frozen_asset_manifest.json").read_text(encoding="utf-8"))["runtime_v4_six_sha256"],
    })
    write_json(output / "stage_state.json", {**state, "stage": "COMPLETE", "decision": decision})
    write_json(output / "sha256_index.json", _sha_index(output))
    index = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT_ID, "status": "COMPLETE",
        "decision": decision, "result_root": str(output.resolve()),
        "report": descriptor(report_path), "result_sha256_index": descriptor(output / "sha256_index.json"),
        "formal_code_commit": json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))["formal_code_commit"],
        "evidence_boundary": "post-freeze descriptive harmonization audit; not a new confirmatory test",
    }
    write_json(Path(args.index_path), index)


def parse_args() -> argparse.Namespace:
    main_root = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-protocol", "run-calibration", "evaluate-test", "finalize"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/iotj_calibration_protocol_harmonization_20260726"))
    parser.add_argument("--data-root", default=str(main_root / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"))
    parser.add_argument("--multiseed-root", default=str(ROOT / "results/iotj_b5_multiseed_20260724"))
    parser.add_argument("--real-h1", default=str(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/global_h1_model.json"))
    parser.add_argument("--runtime-v5-contract", default=str(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json"))
    parser.add_argument("--historical-validation", default=str(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/calibration_validation_predictions.csv"))
    parser.add_argument("--historical-model", default=str(ROOT / "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/target_ridge_105d_manifest.json"))
    parser.add_argument("--prior-lowcal", default=str(ROOT / "results/iotj_low_calibration_sensitivity_20260725"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report-path", default=str(ROOT / "docs/experiments/iotj_calibration_protocol_harmonization_result_20260726.zh.md"))
    parser.add_argument("--index-path", default=str(ROOT / "docs/experiments/iotj_calibration_protocol_harmonization_result_index_20260726.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"freeze-protocol": freeze_protocol, "run-calibration": run_calibration,
     "evaluate-test": evaluate_test, "finalize": finalize}[args.stage](args)
