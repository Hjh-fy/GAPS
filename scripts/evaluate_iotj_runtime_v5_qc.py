"""Two-phase calibration-only QC closure for the frozen Runtime v5 candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_deploy.c5_federated_source_ridge_qc_runtime import C5FederatedSourceRidgeQCRuntime
from gaps_deploy.c5_federated_source_ridge_runtime import C5FederatedSourceRidgeRuntime, SerializedRidgeV5
from gaps_deploy.runtime_v5_qc import (
    COMPONENTS,
    QCCandidate,
    RuntimeV5QCPolicy,
    assign_group_folds,
    descriptor,
    fit_feature_reference,
    fit_regression_consistency_scales,
    make_selection_lock,
    require_selection_lock,
    sha256_file,
)
from gaps_deploy.runtime_v5_qc_bundle import bundle_asset_record
from scripts.evaluate_iotj_b5_regression_multiseed import frozen_hashes


SCHEMA = "iotj.runtime_v5_qc_closure.v1"
OUTPUT_DEFAULT = "results/iotj_b5_c5_runtime_v5_qc_20260725"
BASE_DEFAULT = "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json"
DATA_DEFAULT = "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
EPSILON = 1e-8
FOLD_SEED = 20260725
GAS_NAMES = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}
CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in records)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_finite_rows(rows: Sequence[Mapping[str, Any]], numeric: Sequence[str]) -> None:
    for index, row in enumerate(rows):
        for key in numeric:
            value = float(row.get(key, np.nan))
            if not math.isfinite(value):
                raise RuntimeError(f"row {index} field {key} contains NaN/Inf")


def load_split_inputs(data_root: Path, split: str) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    client = data_root / "client_5"
    windows = np.load(client / f"{split}_features.npy")
    metadata = json.loads((client / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
    phases = np.load(client / f"{split}_phase_labels.npy").reshape(-1)
    classes = np.load(client / f"{split}_classification_labels.npy").reshape(-1)
    regression = np.load(client / f"{split}_regression_labels.npy")
    expected = 320 if split == "calibration" else 1360
    if windows.shape != (expected, 100, 8) or len(metadata) != expected or phases.shape != (expected,) or classes.shape != (expected,) or regression.shape != (expected, 4):
        raise RuntimeError(f"C5 {split} input schema/count differs")
    if not np.isfinite(windows).all() or not np.isfinite(regression).all() or not np.isin(classes, (0, 1, 2, 3)).all():
        raise RuntimeError(f"C5 {split} contains NaN/Inf or invalid labels")
    return windows, metadata, phases, classes.astype(np.int64), regression.astype(np.float64)


def extract_rows(base: C5FederatedSourceRidgeRuntime, data_root: Path, split: str, batch_size: int) -> list[dict[str, Any]]:
    windows, metadata, phases, classes, regression = load_split_inputs(data_root, split)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(windows), batch_size):
        values = np.asarray(windows[start : start + batch_size], dtype=np.float32)
        with torch.no_grad():
            logits, representation, _reg = base.model(torch.from_numpy(values).to(base.device))
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        representation_values = representation.detach().cpu().numpy().astype(np.float64)
        if representation_values.shape != (len(values), 64) or probabilities.shape != (len(values), 4):
            raise RuntimeError("B5 QC representation/probability schema differs")
        for offset in range(len(values)):
            index = start + offset
            route = int(np.argmax(probabilities[offset]))
            meta = dict(metadata[index])
            filename = meta.get("filename")
            if not isinstance(filename, str) or not filename:
                raise RuntimeError(f"C5 {split} filename is missing: {index}")
            feature_meta = dict(meta)
            feature_meta["phase"] = int(phases[index])
            rich = base.feature_extractor(values[offset], feature_meta)
            if len(rich) != 104 or not all(math.isfinite(float(value)) for value in rich.values()):
                raise RuntimeError("rich feature schema differs or contains NaN/Inf")
            h1 = base.source_h1[route].predict(rich)
            target_features = dict(rich)
            target_features["srcpred_H1_source_ridge_ppm"] = h1
            prediction = base.target_ridge[route].predict(target_features)
            true_class = int(classes[index])
            h1_true = base.source_h1[true_class].predict(rich)
            rows.append({
                "sample_index": index,
                "row_key": f"C5:{split}:{index}",
                "split": split,
                "filename": filename,
                "gas_code": str(meta.get("gas_code", "")),
                "concentration_code": str(meta.get("concentration_code", "")),
                "repeat_id": int(meta.get("repeat_id", -1)),
                "phase": int(phases[index]),
                "true_class": true_class,
                "pred_class": route,
                "true_ppm": float(regression[index, true_class]),
                "probabilities": probabilities[offset].tolist(),
                "representation": representation_values[offset].tolist(),
                "feature_dict": rich,
                "source_h1_ppm": h1,
                "source_h1_true_route_ppm": h1_true,
                "prediction_ppm": prediction,
            })
    keys = [row["row_key"] for row in rows]
    if len(keys) != len(set(keys)) or [row["sample_index"] for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"C5 {split} row keys are duplicate/noncanonical")
    return rows


def fit_target_heads(train_rows: Sequence[Mapping[str, Any]], base: C5FederatedSourceRidgeRuntime) -> dict[int, SerializedRidgeV5]:
    heads: dict[int, SerializedRidgeV5] = {}
    for gas in range(4):
        selected = [row for row in train_rows if int(row["true_class"]) == gas]
        names = list(base.target_ridge[gas].feature_names)
        if not selected or names.count("srcpred_H1_source_ridge_ppm") != 1 or len(names) != 105:
            raise RuntimeError(f"fold target Ridge gas {gas} training schema differs")
        matrix_rows = []
        for row in selected:
            features = dict(row["feature_dict"])
            features["srcpred_H1_source_ridge_ppm"] = float(row["source_h1_true_route_ppm"])
            matrix_rows.append([float(features[name]) for name in names])
        x = np.asarray(matrix_rows, dtype=np.float64)
        y = np.asarray([float(row["true_ppm"]) for row in selected], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError("fold target Ridge contains NaN/Inf")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
        design = np.concatenate([np.ones((len(x), 1)), (x - mean) / scale], axis=1)
        alpha = float(base.target_ridge[gas].alpha)
        regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
        regularizer[0, 0] = 0.0
        coef = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ y
        heads[gas] = SerializedRidgeV5(names, mean, scale, coef, float(y.min()), float(y.max()), alpha)
    return heads


def apply_heads(rows: Sequence[Mapping[str, Any]], heads: Mapping[int, SerializedRidgeV5]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        route = int(row["pred_class"])
        features = dict(row["feature_dict"])
        features["srcpred_H1_source_ridge_ppm"] = float(row["source_h1_ppm"])
        item = dict(row)
        item["prediction_ppm"] = heads[route].predict(features)
        output.append(item)
    return output


def policy_payload(
    candidate: str,
    reference: Mapping[str, Any],
    scales: Mapping[str, Any],
    distributions: Mapping[str, Sequence[float]],
    workpoints: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "iotj.runtime_v5_qc_policy.v1",
        "status": "locked",
        "selected_candidate": candidate,
        "epsilon": EPSILON,
        "feature_reference": reference,
        "regression_consistency_scale": scales,
        "component_distributions": {key: list(distributions[key]) for key in COMPONENTS},
        "workpoints": dict(workpoints or {"HC95": {"accept_threshold": 0.0, "reject_threshold": 1.0}, "HC90": {"accept_threshold": 0.0, "reject_threshold": 1.0}}),
        "decision_semantics": {"auto_output_only_for_accept": True},
    }


def raw_and_score_rows(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    scales: Mapping[str, Any],
    distributions: Mapping[str, Sequence[float]] | None,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    bootstrap = {key: [0.0] for key in COMPONENTS}
    raw_policy = RuntimeV5QCPolicy.from_payload(policy_payload("QC3", reference, scales, distributions or bootstrap))
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        raw = raw_policy.raw_components(
            probabilities=row["probabilities"], representation=row["representation"],
            pred_class=int(row["pred_class"]), source_h1_ppm=float(row["source_h1_ppm"]),
            prediction_ppm=float(row["prediction_ppm"]),
        )
        raw_rows.append({**dict(row), **{f"raw_{key}": value for key, value in raw.items()}})
    fitted = {key: sorted(float(row[f"raw_{key}"]) for row in raw_rows) for key in COMPONENTS} if distributions is None else {key: list(distributions[key]) for key in COMPONENTS}
    return raw_rows, fitted


def add_candidate_scores(rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Any], scales: Mapping[str, Any], distributions: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    policies = {
        candidate.value: RuntimeV5QCPolicy.from_payload(policy_payload(candidate.value, reference, scales, distributions))
        for candidate in QCCandidate
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw = {key: float(row[f"raw_{key}"]) for key in COMPONENTS}
        percentiles = {
            key: float(np.searchsorted(np.asarray(distributions[key]), raw[key], side="right") / len(distributions[key]))
            for key in COMPONENTS
        }
        for key, value in percentiles.items():
            item[f"percentile_{key}"] = value
        for name, policy in policies.items():
            aggregate = policy.aggregate_percentiles(percentiles)
            item[f"{name}_risk"] = aggregate["deployment_risk"]
        output.append(item)
    return output


def rank_average(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: Sequence[float], errors: Sequence[float]) -> float:
    x, y = rank_average(values), rank_average(errors)
    if np.std(x) <= 0 or np.std(y) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def subset_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"N": 0, "RMSE": None, "MAE": None, "NRMSE": None}
    errors = np.asarray([float(row["prediction_ppm"]) - float(row["true_ppm"]) for row in rows], dtype=np.float64)
    normalized = np.asarray([errors[index] / CLASS_RANGES[int(row["true_class"])] for index, row in enumerate(rows)], dtype=np.float64)
    return {
        "N": len(rows),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(np.abs(errors))),
        "NRMSE": float(np.sqrt(np.mean(normalized**2))),
    }


def candidate_metrics(rows: Sequence[Mapping[str, Any]], candidate: str) -> dict[str, Any]:
    risk_key = f"{candidate}_risk"
    risks = np.asarray([float(row[risk_key]) for row in rows], dtype=np.float64)
    errors = np.asarray([abs(float(row["prediction_ppm"]) - float(row["true_ppm"])) for row in rows], dtype=np.float64)
    if len(rows) != 320 or not np.isfinite(risks).all() or not np.isfinite(errors).all():
        raise RuntimeError(f"{candidate} OOF rows are incomplete/non-finite")
    order = np.argsort(risks, kind="mergesort")
    deciles = np.array_split(order, 10)
    decile_rmse = [float(np.sqrt(np.mean(errors[indexes] ** 2))) for indexes in deciles]
    adjacent = [decile_rmse[index + 1] >= decile_rmse[index] for index in range(9)]
    result: dict[str, Any] = {
        "candidate": candidate,
        "N": len(rows),
        "spearman_risk_vs_abs_error": spearman(risks, errors),
        "decile_monotonicity_fraction": float(np.mean(adjacent)),
        "lowest_risk_decile_RMSE": decile_rmse[0],
        "highest_risk_decile_RMSE": decile_rmse[-1],
        "tail_enrichment_ratio": float(decile_rmse[-1] / max(float(np.sqrt(np.mean(errors**2))), 1e-12)),
    }
    gas_collapse = False
    for workpoint, accept_q, reject_q in (("HC95", 0.95, 0.9875), ("HC90", 0.90, 0.975)):
        accept_threshold = float(np.quantile(risks, accept_q, method="linear"))
        reject_threshold = float(np.quantile(risks, reject_q, method="linear"))
        accepted = [row for row in rows if float(row[risk_key]) <= accept_threshold]
        rejected = [row for row in rows if float(row[risk_key]) > reject_threshold]
        metrics = subset_metrics(accepted)
        result.update({
            f"{workpoint}_accept_threshold": accept_threshold,
            f"{workpoint}_reject_threshold": reject_threshold,
            f"{workpoint}_accepted_N": metrics["N"],
            f"{workpoint}_accepted_yield": metrics["N"] / len(rows),
            f"{workpoint}_accepted_RMSE": metrics["RMSE"],
            f"{workpoint}_reject_RMSE": subset_metrics(rejected)["RMSE"],
        })
        for gas in range(4):
            gas_rows = [row for row in rows if int(row["true_class"]) == gas]
            gas_accepted = [row for row in gas_rows if float(row[risk_key]) <= accept_threshold]
            gas_yield = len(gas_accepted) / len(gas_rows)
            result[f"{workpoint}_{GAS_NAMES[gas]}_yield"] = gas_yield
            result[f"{workpoint}_{GAS_NAMES[gas]}_RMSE"] = subset_metrics(gas_accepted)["RMSE"]
            gas_collapse = gas_collapse or gas_yield < 0.50
    result["risk_direction_pass"] = bool(result["spearman_risk_vs_abs_error"] > 0 and result["highest_risk_decile_RMSE"] > result["lowest_risk_decile_RMSE"] and result["tail_enrichment_ratio"] > 1.0)
    result["gas_collapse"] = gas_collapse
    result["candidate_pass"] = bool(result["risk_direction_pass"] and not gas_collapse)
    return result


def material_improvement(complex_row: Mapping[str, Any], simple_row: Mapping[str, Any]) -> bool:
    improvements = [
        float(simple_row[f"{workpoint}_accepted_RMSE"]) - float(complex_row[f"{workpoint}_accepted_RMSE"])
        for workpoint in ("HC95", "HC90")
    ]
    yield_losses = [
        float(simple_row[f"{workpoint}_accepted_yield"]) - float(complex_row[f"{workpoint}_accepted_yield"])
        for workpoint in ("HC95", "HC90")
    ]
    return min(improvements) >= 0.0 and max(improvements) >= 0.25 and max(yield_losses) <= 0.01


def select_candidate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["candidate"]): row for row in summaries}
    passing = [name for name in ("QC1", "QC2", "QC3") if bool(by_name[name]["candidate_pass"])]
    if not passing:
        return {"status": "INCONCLUSIVE", "selected_candidate": None, "selection_reason": "no candidate passed risk direction/tail enrichment and gas-collapse gates", "test_may_open": False}
    selected = passing[0]
    reason = f"{selected} is the simplest passing candidate"
    for complex_name in ("QC2", "QC3"):
        if complex_name not in passing:
            continue
        predecessor = "QC1" if complex_name == "QC2" else "QC2"
        if selected == predecessor and material_improvement(by_name[complex_name], by_name[predecessor]):
            selected = complex_name
            reason = f"{complex_name} passed and materially/stably improved {predecessor}"
    if selected == "QC3" and not material_improvement(by_name["QC3"], by_name["QC2"]):
        selected = "QC2"
        reason = "QC3 did not materially improve QC2; select QC2 by preregistered simplicity rule"
    return {"status": "SELECTED_AND_LOCKABLE", "selected_candidate": selected, "selection_reason": reason, "test_may_open": True}


def public_oof_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "sample_index", "row_key", "fold", "filename", "gas_code", "concentration_code",
        "repeat_id", "true_class", "pred_class", "true_ppm", "source_h1_ppm", "prediction_ppm",
    ]
    output = {key: row[key] for key in fields}
    output["abs_error_ppm"] = abs(float(row["prediction_ppm"]) - float(row["true_ppm"]))
    output.update({f"raw_{key}": row[f"raw_{key}"] for key in COMPONENTS})
    output.update({f"percentile_{key}": row[f"percentile_{key}"] for key in COMPONENTS})
    output.update({f"{candidate.value}_risk": row[f"{candidate.value}_risk"] for candidate in QCCandidate})
    return output


def freeze_protocol(args: argparse.Namespace, output: Path, base_contract: Path, frozen: Mapping[str, str]) -> None:
    data_root = Path(args.data_root)
    base_payload = json.loads(base_contract.read_text(encoding="utf-8"))
    bundle_manifest = Path(base_payload["bundle_manifest"]["path"])
    bundle_payload = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    bundle_root = bundle_manifest.parent
    runtime_assets = {
        name: descriptor(bundle_root / record["bundle_path"])
        for name, record in bundle_payload["assets"].items()
    }
    assets = {
        "base_runtime_v5_contract": descriptor(base_contract),
        "base_runtime_v5_bundle_manifest": descriptor(bundle_manifest),
        **runtime_assets,
        "calibration_features": descriptor(data_root / "client_5/calibration_features.npy"),
        "calibration_metadata": descriptor(data_root / "client_5/calibration_experiment_info.json"),
        "calibration_phase_labels": descriptor(data_root / "client_5/calibration_phase_labels.npy"),
        "test_features": descriptor(data_root / "client_5/test_features.npy"),
        "test_metadata": descriptor(data_root / "client_5/test_experiment_info.json"),
        "test_phase_labels": descriptor(data_root / "client_5/test_phase_labels.npy"),
    }
    write_json(output / "frozen_runtime_v5_assets.json", {"schema_version": SCHEMA, "build_commit": git_commit(), "assets": assets, "runtime_v4_hc_frozen_sha256": dict(frozen)})
    write_json(output / "candidate_risk_definitions.json", {
        "schema_version": SCHEMA,
        "candidates": {
            "QC1": "mean(ECDF(entropy), ECDF(inverse_margin))",
            "QC2": "mean(confidence_group, mean(ECDF(prototype_distance), ECDF(support_distance)))",
            "QC3": "mean(confidence_group, distance_group, ECDF(abs(target_Ridge-H1)/per_predicted_gas_MAD_scale))",
        },
        "component_internal_mean_then_group_mean": True,
        "ecdf_side": "right",
        "epsilon": EPSILON,
        "forbidden": ["H2", "H3", "H2.3", "all_prior", "v4 risk values", "test labels or residuals"],
    })
    write_json(output / "protocol_manifest.json", {
        "schema_version": "iotj.runtime_v5_qc_protocol.v2",
        "experiment_id": "IOTJ-B5-C5-RUNTIME-V5-QC-20260725",
        "status": "protocol_frozen_before_test",
        "build_commit": git_commit(),
        "source_clients": ["C1", "C2"], "target_client": "C5", "classifier": "B5_seed42_frozen",
        "regression": "real_topology_Federated_H1_plus_C5_105D_per_gas_Ridge",
        "calibration_rows": 320, "test_rows": 1360,
        "protocol_amendment": {
            "version": 2,
            "reason": "data audit disproved the fixed-four-calibration-row assumption before test opening",
            "previous_assumption": "exactly 4 calibration rows per filename",
            "replacement": "all 1..7 calibration rows sharing a filename remain in one fold",
            "previous_test_opened": False,
            "qc_selection_lock_existed_before_amendment": False,
            "runtime_v4_hc_six_sha_unchanged": True,
        },
        "folds": {
            "count": 5, "seed": FOLD_SEED, "algorithm": "deterministic largest-group-first greedy lexicographic balance",
            "balance_order": ["total rows", "gas rows", "gas-concentration rows", "group count", "fold id"],
            "group_key": "filename", "group_count": 80, "group_size_min": 1, "group_size_max": 7,
            "all_calibration_rows_for_filename_stay_together": True, "same_group_cross_fold_forbidden": True,
        },
        "fold_local_assets": ["target Ridge", "B5 prototype/support", "component ECDF", "regression-consistency MAD scale"],
        "selection": {"test_used": False, "material_RMSE_ppm": 0.25, "max_yield_loss": 0.01, "gas_collapse_yield_floor": 0.50, "simplest_passing": True},
        "workpoints": {"HC95": {"accept_quantile": 0.95, "reject_quantile": 0.9875}, "HC90": {"accept_quantile": 0.90, "reject_quantile": 0.975}},
        "auto_output_only_for_accept": True,
        "evidence_boundary": {
            "supported": "calibration-derived confidence, representation-distance, and source-to-target regression-consistency signals for selective output",
            "filename_grouping_scope": "calibration OOF folds only",
            "historical_calibration_test_split": "window-level",
            "original_file_level_calibration_test_independence_claim_allowed": False,
        },
    })


def calibrate_and_lock(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty QC result root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen_before = frozen_hashes(REPO_ROOT)
    base_contract = Path(args.base_runtime_contract).resolve()
    base = C5FederatedSourceRidgeRuntime.from_runtime_contract(base_contract, device=args.device)
    freeze_protocol(args, output, base_contract, frozen_before)
    rows = extract_rows(base, Path(args.data_root), "calibration", args.batch_size)
    metadata = [{"filename": row["filename"], "classification_label": row["true_class"], "gas_code": row["gas_code"], "concentration_code": row["concentration_code"], "repeat_id": row["repeat_id"]} for row in rows]
    assignments, fold_audit = assign_group_folds(metadata, n_splits=5, seed=FOLD_SEED)
    fold_manifest_rows = [{"sample_index": index, "row_key": rows[index]["row_key"], "filename": rows[index]["filename"], "true_class": rows[index]["true_class"], "gas": GAS_NAMES[int(rows[index]["true_class"])], "concentration_code": rows[index]["concentration_code"], "fold": assignments[index]} for index in range(320)]
    write_json(output / "qc_calibration_fold_manifest.json", {"schema_version": SCHEMA, "audit": fold_audit, "rows": fold_manifest_rows})

    oof_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for fold in range(5):
        train = [dict(row) for index, row in enumerate(rows) if assignments[index] != fold]
        validation = [dict(row) for index, row in enumerate(rows) if assignments[index] == fold]
        if len(train) + len(validation) != 320 or not train or not validation or {row["filename"] for row in train} & {row["filename"] for row in validation}:
            raise RuntimeError(f"fold {fold} group isolation/count failed")
        heads = fit_target_heads(train, base)
        train_pred = apply_heads(train, heads)
        validation_pred = apply_heads(validation, heads)
        reference = fit_feature_reference(train_pred, epsilon=EPSILON)
        scales = fit_regression_consistency_scales(train_pred, epsilon=EPSILON)
        raw_train, distributions = raw_and_score_rows(train_pred, reference, scales, None)
        raw_validation, _ = raw_and_score_rows(validation_pred, reference, scales, distributions)
        scored = add_candidate_scores(raw_validation, reference, scales, distributions)
        for row in scored:
            row["fold"] = fold
            oof_rows.append(row)
        fold_records.append({
            "fold": fold, "train_N": len(train), "validation_N": len(validation),
            "train_group_count": len({row["filename"] for row in train}),
            "validation_group_count": len({row["filename"] for row in validation}),
            "group_overlap_count": 0,
            "target_alpha": {str(gas): heads[gas].alpha for gas in range(4)},
            "feature_reference_train_only": True, "ecdf_train_only": True, "mad_scale_train_only": True,
            "raw_train_N": len(raw_train),
        })
    oof_rows.sort(key=lambda row: int(row["sample_index"]))
    if len(oof_rows) != 320 or [int(row["sample_index"]) for row in oof_rows] != list(range(320)):
        raise RuntimeError("OOF row coverage/key differs")
    write_csv(output / "qc_calibration_oof_rows.csv", [public_oof_row(row) for row in oof_rows])
    write_json(output / "qc_calibration_oof_audit.json", {"schema_version": SCHEMA, "status": "PASS", "folds": fold_records, "row_count": 320, "row_key_unique": True, "group_cross_fold_count": 0, "test_opened": False, "test_used_for_any_calibration_or_selection": False})

    summaries = [candidate_metrics(oof_rows, candidate.value) for candidate in QCCandidate]
    decision = select_candidate(summaries)
    write_csv(output / "qc_candidate_calibration_summary.csv", summaries)
    write_json(output / "qc_candidate_selection.json", {"schema_version": SCHEMA, **decision, "selection_rule": {"simplest_passing": True, "material_improvement_ppm": 0.25, "max_yield_decline": 0.01, "QC3_without_material_QC2_gain_selects_QC2": True}, "test_opened": False})
    if not decision["test_may_open"]:
        write_json(output / "decision_gate.json", {"schema_version": SCHEMA, "decision": "RUNTIME_V5_QC_INCONCLUSIVE", "reason": decision["selection_reason"], "test_opened_after_selection": False})
        raise RuntimeError("OOF selection is inconclusive; test remains closed")

    selected = str(decision["selected_candidate"])
    full_reference = fit_feature_reference(rows, epsilon=EPSILON)
    full_scales = fit_regression_consistency_scales(rows, epsilon=EPSILON)
    full_raw, full_distributions = raw_and_score_rows(rows, full_reference, full_scales, None)
    full_scored = add_candidate_scores(full_raw, full_reference, full_scales, full_distributions)
    selected_risks = np.asarray([float(row[f"{selected}_risk"]) for row in full_scored])
    workpoints = {
        "HC95": {"accept_threshold": float(np.quantile(selected_risks, 0.95, method="linear")), "reject_threshold": float(np.quantile(selected_risks, 0.9875, method="linear"))},
        "HC90": {"accept_threshold": float(np.quantile(selected_risks, 0.90, method="linear")), "reject_threshold": float(np.quantile(selected_risks, 0.975, method="linear"))},
    }
    policy = policy_payload(selected, full_reference, full_scales, full_distributions, workpoints)
    policy_path = output / "runtime_v5_qc_policy.json"
    write_json(policy_path, policy)
    write_json(output / "regression_consistency_scale.json", full_scales)
    lock_assets = {
        "qc_policy": policy_path,
        "base_runtime_contract": base_contract,
        "fold_manifest": output / "qc_calibration_fold_manifest.json",
        "candidate_selection": output / "qc_candidate_selection.json",
    }
    lock = make_selection_lock(selected_candidate=selected, selection_reason=str(decision["selection_reason"]), policy_path=policy_path, bound_assets=lock_assets, build_commit=git_commit())
    lock_path = output / "qc_selection_lock.json"
    write_json(lock_path, lock)
    require_selection_lock(lock_path, lock_assets)
    write_json(output / "calibration_lock_receipt.json", {"schema_version": SCHEMA, "qc_selection_lock": descriptor(lock_path), "persisted_before_any_test_open": True, "test_opened_after_lock": False, "created_at": utc_now()})
    if frozen_hashes(REPO_ROOT) != frozen_before:
        raise RuntimeError("runtime v4/HC frozen assets changed during calibration")
    print(json.dumps({"status": "LOCKED_BEFORE_TEST", "selected_candidate": selected, "qc_selection_lock": descriptor(lock_path)}, indent=2))


def decision_rows(rows: Sequence[Mapping[str, Any]], policy: RuntimeV5QCPolicy, workpoint: str) -> list[dict[str, Any]]:
    risk_key = f"{policy.payload['selected_candidate']}_risk"
    output: list[dict[str, Any]] = []
    for row in rows:
        risk = float(row[risk_key])
        decision, auto = policy.decision(risk, float(row["prediction_ppm"]), policy.payload["workpoints"][workpoint])
        item = {
            "sample_index": int(row["sample_index"]), "row_key": row["row_key"], "split": row["split"],
            "filename": row["filename"], "true_class": int(row["true_class"]), "pred_class": int(row["pred_class"]),
            "true_ppm": float(row["true_ppm"]), "source_h1_ppm": float(row["source_h1_ppm"]),
            "prediction_ppm": float(row["prediction_ppm"]), "abs_error_ppm": abs(float(row["prediction_ppm"]) - float(row["true_ppm"])),
            "deployment_risk": risk, "qc_workpoint": workpoint, "qc_decision": decision, "auto_output_ppm": auto,
        }
        item.update({f"raw_{key}": float(row[f"raw_{key}"]) for key in COMPONENTS})
        item.update({f"percentile_{key}": float(row[f"percentile_{key}"]) for key in COMPONENTS})
        output.append(item)
    return output


def workpoint_summary(rows: Sequence[Mapping[str, Any]], workpoint: str) -> dict[str, Any]:
    accepted = [row for row in rows if row["qc_decision"] == "accept"]
    review = [row for row in rows if row["qc_decision"] == "review"]
    rejected = [row for row in rows if row["qc_decision"] == "reject"]
    summary = {
        "workpoint": workpoint,
        "N": len(rows), "accept_N": len(accepted), "review_N": len(review), "reject_N": len(rejected),
        "accepted_yield": len(accepted) / len(rows),
        "full_RMSE": subset_metrics(rows)["RMSE"],
        "accepted_RMSE": subset_metrics(accepted)["RMSE"],
        "accepted_MAE": subset_metrics(accepted)["MAE"],
        "accepted_NRMSE": subset_metrics(accepted)["NRMSE"],
        "review_RMSE": subset_metrics(review)["RMSE"],
        "reject_RMSE": subset_metrics(rejected)["RMSE"],
        "misclassified_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in rows),
        "misclassified_accept_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in accepted),
        "misclassified_review_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in review),
        "misclassified_reject_N": sum(int(row["pred_class"]) != int(row["true_class"]) for row in rejected),
    }
    co_high = [row for row in rows if int(row["true_class"]) == 1 and 200 <= float(row["true_ppm"]) <= 250]
    co_high_accepted = [row for row in co_high if row["qc_decision"] == "accept"]
    summary["CO_high_N"] = len(co_high)
    summary["CO_high_accepted_yield"] = len(co_high_accepted) / len(co_high)
    summary["CO_high_accepted_RMSE"] = subset_metrics(co_high_accepted)["RMSE"]
    return summary


def per_gas_summaries(rows: Sequence[Mapping[str, Any]], workpoint: str) -> list[dict[str, Any]]:
    output = []
    for gas in range(4):
        selected = [row for row in rows if int(row["true_class"]) == gas]
        accepted = [row for row in selected if row["qc_decision"] == "accept"]
        output.append({
            "workpoint": workpoint, "gas_id": gas, "gas": GAS_NAMES[gas], "N": len(selected),
            "accept_N": len(accepted), "accepted_yield": len(accepted) / len(selected),
            "full_RMSE": subset_metrics(selected)["RMSE"], "accepted_RMSE": subset_metrics(accepted)["RMSE"],
        })
    return output


def risk_deciles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["deployment_risk"]), int(row["sample_index"])))
    output = []
    for decile, indexes in enumerate(np.array_split(np.arange(len(ordered)), 10), start=1):
        selected = [ordered[int(index)] for index in indexes]
        output.append({
            "decile": decile, "N": len(selected),
            "risk_min": min(float(row["deployment_risk"]) for row in selected),
            "risk_max": max(float(row["deployment_risk"]) for row in selected),
            "RMSE": subset_metrics(selected)["RMSE"], "MAE": subset_metrics(selected)["MAE"],
            "misclassification_rate": np.mean([int(row["pred_class"]) != int(row["true_class"]) for row in selected]),
        })
    return output


def v4_baseline(workpoint: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = REPO_ROOT / f"results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_{workpoint.lower()}_records.csv"
    source = read_csv(path)
    rows = [{
        "sample_index": int(float(row["sample_index"])), "true_class": int(float(row["true_class"])),
        "true_ppm": float(row["true_ppm"]), "prediction_ppm": float(row["target_ridge_plus_source_preds_ppm"]),
        "qc_decision": row["qc_decision"],
    } for row in source]
    if len(rows) != 1360 or [row["sample_index"] for row in rows] != list(range(1360)):
        raise RuntimeError(f"v4 {workpoint} baseline row universe differs")
    return workpoint_summary(rows, workpoint), per_gas_summaries(rows, workpoint)


def compare_v4_guards(v5_summary: Mapping[str, Any], v5_gas: Sequence[Mapping[str, Any]], v4_summary: Mapping[str, Any], v4_gas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall_rmse = float(v5_summary["accepted_RMSE"]) <= float(v4_summary["accepted_RMSE"]) + 0.5
    overall_yield = float(v5_summary["accepted_yield"]) >= float(v4_summary["accepted_yield"]) - 0.02
    per_gas = []
    for v5, v4 in zip(v5_gas, v4_gas):
        yield_drop = float(v4["accepted_yield"]) - float(v5["accepted_yield"])
        rmse_benefit = float(v5["accepted_RMSE"]) < float(v4["accepted_RMSE"])
        passed = yield_drop <= 0.10 or rmse_benefit
        per_gas.append({"gas": v5["gas"], "yield_drop_vs_v4": yield_drop, "v5_has_RMSE_benefit": rmse_benefit, "pass": passed})
    return {"accepted_RMSE_guard_pass": overall_rmse, "accepted_yield_guard_pass": overall_yield, "per_gas_guards": per_gas, "all_pass": bool(overall_rmse and overall_yield and all(row["pass"] for row in per_gas))}


def build_qc_bundle(output: Path, base_contract: Path, policy_path: Path, build_commit: str) -> tuple[Path, dict[str, Path]]:
    bundle = output / "runtime_v5_qc_bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    base_copy = bundle / "base_runtime_contract.json"
    policy_copy = bundle / "qc_policy.json"
    shutil.copyfile(base_contract, base_copy)
    shutil.copyfile(policy_path, policy_copy)
    manifest_path = bundle / "manifest.json"
    write_json(manifest_path, {
        "schema_version": "iotj.runtime_v5_qc_bundle.v1", "status": "locked", "build_commit": build_commit,
        "assets": {"base_runtime_contract": bundle_asset_record(bundle, base_copy), "qc_policy": bundle_asset_record(bundle, policy_copy)},
        "dependency_contract": {"allowed": ["base_runtime_contract", "qc_policy"], "forbidden": ["H2", "H3", "H2.3", "all_prior", "legacy_rescue", "runtime_v4_risk"], "legacy_fallback": False},
    })
    contracts: dict[str, Path] = {}
    for workpoint in ("HC95", "HC90"):
        contract_path = output / f"runtime_v5_qc_{workpoint.lower()}_contract.json"
        write_json(contract_path, {
            "schema_version": "iotj.c5_federated_source_ridge_qc_runtime_contract.v1", "status": "locked",
            "bundle_manifest": descriptor(manifest_path), "workpoint": workpoint,
            "outputs": ["sample_index", "pred_class", "source_h1_ppm", "prediction_ppm", "raw_risk_components", "normalized_risk_components", "deployment_risk", "qc_workpoint", "qc_decision", "auto_output_ppm"],
        })
        contracts[workpoint] = contract_path
    return manifest_path, contracts


def runtime_parity(
    args: argparse.Namespace,
    output: Path,
    contracts: Mapping[str, Path],
    offline_by_split: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    combined_test: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for split in ("calibration", "test"):
        windows, metadata, phases, _classes, _regression = load_split_inputs(Path(args.data_root), split)
        expected = 320 if split == "calibration" else 1360
        aggregate = {"class_mismatch": 0, "prediction_max": 0.0, "raw_max": 0.0, "normalized_max": 0.0, "risk_max": 0.0, "decision_mismatch": 0, "auto_mismatch": 0, "row_key_mismatch": 0, "missing_nonfinite": 0}
        rows_by_workpoint: dict[str, list[dict[str, Any]]] = {}
        for workpoint in ("HC95", "HC90"):
            runtime = C5FederatedSourceRidgeQCRuntime.from_runtime_contract(contracts[workpoint], device=args.device)
            runtime_rows: list[dict[str, Any]] = []
            for start in range(0, expected, args.batch_size):
                batch = runtime.infer(windows[start : start + args.batch_size], metadata[start : start + args.batch_size], phases[start : start + args.batch_size])
                for offset, row in enumerate(batch):
                    item = dict(row); item["sample_index"] = start + offset; runtime_rows.append(item)
            reference = list(offline_by_split[split][workpoint])
            if len(runtime_rows) != expected or len(reference) != expected:
                raise RuntimeError(f"runtime QC {split}/{workpoint} row count differs")
            rows_by_workpoint[workpoint] = runtime_rows
            for index, (runtime_row, offline) in enumerate(zip(runtime_rows, reference)):
                aggregate["row_key_mismatch"] += int(int(runtime_row["sample_index"]) != index or int(offline["sample_index"]) != index)
                aggregate["class_mismatch"] += int(int(runtime_row["pred_class"]) != int(offline["pred_class"]))
                aggregate["prediction_max"] = max(aggregate["prediction_max"], abs(float(runtime_row["prediction_ppm"]) - float(offline["prediction_ppm"])))
                for key in COMPONENTS:
                    aggregate["raw_max"] = max(aggregate["raw_max"], abs(float(runtime_row["raw_risk_components"][key]) - float(offline[f"raw_{key}"])))
                    aggregate["normalized_max"] = max(aggregate["normalized_max"], abs(float(runtime_row["normalized_risk_components"][key]) - float(offline[f"percentile_{key}"])))
                aggregate["risk_max"] = max(aggregate["risk_max"], abs(float(runtime_row["deployment_risk"]) - float(offline["deployment_risk"])))
                aggregate["decision_mismatch"] += int(runtime_row["qc_decision"] != offline["qc_decision"])
                left, right = runtime_row["auto_output_ppm"], offline["auto_output_ppm"]
                aggregate["auto_mismatch"] += int((left is None) != (right is None) or (left is not None and abs(float(left) - float(right)) > 1e-6))
        passed = aggregate["class_mismatch"] == aggregate["decision_mismatch"] == aggregate["auto_mismatch"] == aggregate["row_key_mismatch"] == aggregate["missing_nonfinite"] == 0 and aggregate["prediction_max"] <= 1e-6 and aggregate["raw_max"] <= 1e-10 and aggregate["normalized_max"] <= 1e-10 and aggregate["risk_max"] <= 1e-10
        report = {"schema_version": SCHEMA, "status": "PASS" if passed else "FAIL_CLOSED", "N": expected, "workpoints": ["HC95", "HC90"], **aggregate, "tolerances": {"prediction_ppm": 1e-6, "raw_risk": 1e-10, "normalized_risk": 1e-10, "deployment_risk": 1e-10}}
        write_json(output / f"runtime_qc_{split}_parity_report.json", report)
        reports[split] = report
        if not passed:
            raise RuntimeError(f"runtime QC {split} parity failed: {report}")
        if split == "test":
            for index in range(expected):
                combined_test.append({
                    "sample_index": index, "row_key": f"C5:test:{index}",
                    "pred_class": rows_by_workpoint["HC95"][index]["pred_class"],
                    "prediction_ppm": rows_by_workpoint["HC95"][index]["prediction_ppm"],
                    "deployment_risk": rows_by_workpoint["HC95"][index]["deployment_risk"],
                    "HC95_decision": rows_by_workpoint["HC95"][index]["qc_decision"], "HC95_auto_output_ppm": rows_by_workpoint["HC95"][index]["auto_output_ppm"],
                    "HC90_decision": rows_by_workpoint["HC90"][index]["qc_decision"], "HC90_auto_output_ppm": rows_by_workpoint["HC90"][index]["auto_output_ppm"],
                })
    write_csv(output / "runtime_qc_test_rows.csv", combined_test)


def evaluate_test(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    frozen_before = frozen_hashes(REPO_ROOT)
    lock_path = output / "qc_selection_lock.json"
    policy_path = output / "runtime_v5_qc_policy.json"
    base_contract = Path(args.base_runtime_contract).resolve()
    lock_assets = {"qc_policy": policy_path, "base_runtime_contract": base_contract, "fold_manifest": output / "qc_calibration_fold_manifest.json", "candidate_selection": output / "qc_candidate_selection.json"}
    lock = require_selection_lock(lock_path, lock_assets)
    receipt = json.loads((output / "calibration_lock_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("persisted_before_any_test_open") is not True or receipt.get("qc_selection_lock", {}).get("sha256") != sha256_file(lock_path):
        raise RuntimeError("qc_selection_lock was not persisted before test opening")
    policy = RuntimeV5QCPolicy.from_path(policy_path)
    if policy.payload["selected_candidate"] != lock["selected_candidate"]:
        raise RuntimeError("test-stage candidate differs from selection lock")
    base = C5FederatedSourceRidgeRuntime.from_runtime_contract(base_contract, device=args.device)

    calibration_rows = extract_rows(base, Path(args.data_root), "calibration", args.batch_size)
    calibration_raw, _ = raw_and_score_rows(calibration_rows, policy.payload["feature_reference"], policy.payload["regression_consistency_scale"], policy.payload["component_distributions"])
    calibration_scored = add_candidate_scores(calibration_raw, policy.payload["feature_reference"], policy.payload["regression_consistency_scale"], policy.payload["component_distributions"])
    calibration_decisions = {workpoint: decision_rows(calibration_scored, policy, workpoint) for workpoint in ("HC95", "HC90")}

    test_rows = extract_rows(base, Path(args.data_root), "test", args.batch_size)
    test_raw, _ = raw_and_score_rows(test_rows, policy.payload["feature_reference"], policy.payload["regression_consistency_scale"], policy.payload["component_distributions"])
    test_scored = add_candidate_scores(test_raw, policy.payload["feature_reference"], policy.payload["regression_consistency_scale"], policy.payload["component_distributions"])
    test_decisions = {workpoint: decision_rows(test_scored, policy, workpoint) for workpoint in ("HC95", "HC90")}
    write_csv(output / "hc95_test_rows.csv", test_decisions["HC95"])
    write_csv(output / "hc90_test_rows.csv", test_decisions["HC90"])

    summaries = [workpoint_summary(test_decisions[workpoint], workpoint) for workpoint in ("HC95", "HC90")]
    gas_rows = [row for workpoint in ("HC95", "HC90") for row in per_gas_summaries(test_decisions[workpoint], workpoint)]
    write_csv(output / "qc_test_summary.csv", summaries)
    write_csv(output / "qc_per_gas_summary.csv", gas_rows)
    deciles = risk_deciles(test_decisions["HC95"])
    write_csv(output / "qc_risk_decile_summary.csv", deciles)

    guard_records = []
    all_v4_guards = True
    for workpoint in ("HC95", "HC90"):
        v5_summary = next(row for row in summaries if row["workpoint"] == workpoint)
        v5_gas = [row for row in gas_rows if row["workpoint"] == workpoint]
        v4_summary, v4_gas = v4_baseline(workpoint)
        guard = compare_v4_guards(v5_summary, v5_gas, v4_summary, v4_gas)
        all_v4_guards = all_v4_guards and guard["all_pass"]
        guard_records.append({"workpoint": workpoint, "v5": v5_summary, "v4": v4_summary, "guard": guard})
    risk_direction_test = bool(deciles[-1]["RMSE"] > deciles[0]["RMSE"] and all(summary["reject_RMSE"] is not None and summary["accepted_RMSE"] is not None and float(summary["reject_RMSE"]) > float(summary["accepted_RMSE"]) for summary in summaries))
    write_json(output / "comparison_vs_runtime_v4.json", {"schema_version": SCHEMA, "automatic_v4_load": True, "records": guard_records, "all_v4_promotion_guards_pass": all_v4_guards, "highest_decile_RMSE_gt_lowest": deciles[-1]["RMSE"] > deciles[0]["RMSE"], "reject_RMSE_gt_accepted_both": risk_direction_test})

    manifest_path, contracts = build_qc_bundle(output, base_contract, policy_path, git_commit())
    runtime_parity(args, output, contracts, {"calibration": calibration_decisions, "test": test_decisions})
    parity_cal = json.loads((output / "runtime_qc_calibration_parity_report.json").read_text(encoding="utf-8"))
    parity_test = json.loads((output / "runtime_qc_test_parity_report.json").read_text(encoding="utf-8"))
    if not risk_direction_test:
        decision = "KEEP_RUNTIME_V4"
        reason = "test risk direction or reject-vs-accepted ordering failed"
    elif all_v4_guards:
        decision = "PROMOTE_RUNTIME_V5_WITH_QC"
        reason = "OOF selection, test risk direction, v4 guards, frozen assets, and runtime parity all passed"
    else:
        decision = "RUNTIME_V5_QC_VALID_BUT_NOT_SUPERIOR"
        reason = "risk and runtime parity are valid, but one or more preregistered v4 promotion guards failed"
    gate = {
        "schema_version": SCHEMA, "decision": decision, "reason": reason,
        "selected_candidate": policy.payload["selected_candidate"], "qc_selection_lock_sha256": sha256_file(lock_path),
        "test_opened_after_selection": True, "test_opened_after_lock": True,
        "test_used_for_fit_select_refit_normalization_scale_threshold_or_rule": False,
        "final_test_evaluation_timestamp": utc_now(), "risk_direction_test_pass": risk_direction_test,
        "all_v4_promotion_guards_pass": all_v4_guards,
        "calibration_runtime_parity_pass": parity_cal["status"] == "PASS", "test_runtime_parity_pass": parity_test["status"] == "PASS",
        "runtime_v5_qc_bundle": descriptor(manifest_path), "runtime_v4_hc_frozen_sha256": dict(frozen_before),
        "next_stage": "Pi/PC benchmark" if decision == "PROMOTE_RUNTIME_V5_WITH_QC" else "retain runtime v4 as formal baseline; do not tune v5 QC on test",
    }
    if frozen_hashes(REPO_ROOT) != frozen_before:
        raise RuntimeError("runtime v4/HC frozen assets changed during test evaluation")
    write_json(output / "decision_gate.json", gate)
    write_json(output / "test_open_receipt.json", {"schema_version": SCHEMA, "selection_lock": descriptor(lock_path), "test_opened_after_lock": True, "candidate_or_policy_changed_after_lock": False, "timestamp": utc_now()})
    print(json.dumps(gate, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["calibrate-lock", "evaluate-test"])
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    parser.add_argument("--base-runtime-contract", default=BASE_DEFAULT)
    parser.add_argument("--data-root", default=DATA_DEFAULT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "calibrate-lock":
        calibrate_and_lock(args)
    else:
        evaluate_test(args)


if __name__ == "__main__":
    main()
