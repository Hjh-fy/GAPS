"""Run frozen R84_FED_H1 on the three canonical-v1 A4 endpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.state_fingerprint import checkpoint_provenance
from run_regression_head_ablation import CLASS_NAMES, build_oracle_rows, deterministic_train_val, fit_ridge
from scripts import run_gaps_cross_target_r84_full as common
from scripts.summarize_iotj_classification_ablation import evaluate_checkpoint_stream
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


STUDY_ID = "iotj_canonical_v1_final_20260808"
SCHEMA_VERSION = "iotj.canonical_v1.r84_fed_h1.v1"
TARGETS = ("C3", "C4", "C5")
DATA_ROOT = ROOT / "dataset" / "iotj_canonical_v1"
DEFAULT_STUDY_ROOT = ROOT / "results" / STUDY_ID
RIDGE_ALPHAS = common.RIDGE_ALPHAS
CLASSIFICATION_EXPERIMENT_PREFIX = "CANONICAL-V1-A4"
REGRESSION_EXPERIMENT_PREFIX = "CAN-V1-R84-A4"
SPLIT_PROTOCOL = "canonical_v1_target_20_80"


def expected_counts(target: str) -> dict[str, Any]:
    target = target.upper()
    client = int(target[1:])
    directory = DATA_ROOT / f"client_{client}"
    calibration = np.load(directory / "calibration_classification_labels.npy", allow_pickle=False)
    test = np.load(directory / "test_classification_labels.npy", allow_pickle=False)
    return {
        "calibration": int(len(calibration)),
        "test": int(len(test)),
        "per_class": {class_id: int(np.sum(calibration == class_id)) for class_id in range(4)},
    }


def checkpoint_for(classification_root: Path, target: str) -> tuple[Path, dict[str, Any]]:
    run_dir = classification_root / f"{CLASSIFICATION_EXPERIMENT_PREFIX}-{target}"
    marker = json.loads((run_dir / "fixed_endpoint_complete.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if marker.get("experiment_id") != f"{CLASSIFICATION_EXPERIMENT_PREFIX}-{target}":
        raise RuntimeError(f"FAIL_CLOSED {target} completion identity differs")
    protocol = manifest.get("protocol", {})
    if protocol.get("classifier_router") != "A4" or protocol.get("local_epochs") != 1:
        raise RuntimeError(f"FAIL_CLOSED {target} is not final A4 LE1")
    if marker.get("target_test_opened") is not False or manifest.get("target_test_opened") is not False:
        raise RuntimeError(f"FAIL_CLOSED {target} test opened before R84 lock")
    provenance = checkpoint_provenance(checkpoint)
    if provenance["formal_round"] != 25:
        raise RuntimeError(f"FAIL_CLOSED {target} checkpoint is not round25")
    if provenance["whole_file_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"FAIL_CLOSED {target} checkpoint hash differs")
    return checkpoint, provenance


def route_rows(
    checkpoint: Path,
    target: str,
    split: str,
    device: torch.device,
    batch_size: int,
    *,
    expected_endpoint: tuple[str, int] = ("round", 25),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, metrics = evaluate_checkpoint_stream(
        checkpoint,
        data_root=DATA_ROOT,
        target_client=int(target[1:]),
        split=split,
        device=device,
        batch_size=batch_size,
        expected_endpoint=expected_endpoint,
    )
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"FAIL_CLOSED {target}/{split} route order differs")
    return rows, metrics


def enriched_oracle_rows(target: str, split: str) -> list[dict[str, Any]]:
    rows = build_oracle_rows(DATA_ROOT, [target], split)
    metadata = json.loads(
        (DATA_ROOT / f"client_{target[1:]}" / f"{split}_experiment_info.json").read_text(encoding="utf-8")
    )
    if len(rows) != len(metadata):
        raise RuntimeError(f"FAIL_CLOSED {target}/{split} metadata rows differ")
    keep = (
        "physical_identity", "filename", "repeat_id", "gas", "gas_code",
        "concentration", "window_start_s", "window_end_s", "quality_flag",
        "samples_per_bin_mean", "empty_bin_ratio", "observed_ratio",
        "short_gap_interpolated_ratio", "max_missing_run", "baseline_n_raw_samples",
        "duplicate_timestamps", "sampling_completeness",
    )
    for row, meta in zip(rows, metadata):
        row.update({key: meta.get(key) for key in keep})
    return rows


def prepare_rows(
    target: str,
    split: str,
    routes: Sequence[Mapping[str, Any]],
    h1: Mapping[int, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = enriched_oracle_rows(target, split)
    if len(base) != len(routes):
        raise RuntimeError(f"FAIL_CLOSED {target}/{split} route/data count differs")
    oracle: list[dict[str, Any]] = []
    deployment: list[dict[str, Any]] = []
    for row, route in zip(base, routes):
        if int(row["sample_index"]) != int(route["sample_index"]) or int(row["true_class"]) != int(route["true_class"]):
            raise RuntimeError(f"FAIL_CLOSED {target}/{split} route alignment differs")
        true_class = int(row["true_class"])
        pred_class = int(route["pred_class"])
        oracle_item = {**row, "pred_class": pred_class}
        oracle_item["H1_federated_source_ridge_ppm"] = h1[true_class].predict(row["feature_dict"])
        deployment_item = {**row, "pred_class": pred_class}
        deployment_item["H1_federated_source_ridge_ppm"] = h1[pred_class].predict(row["feature_dict"])
        for class_id in range(4):
            deployment_item[f"prob_class_{class_id}"] = float(route[f"prob_{class_id}"])
        deployment_item["confidence"] = float(route["confidence"])
        oracle.append(oracle_item)
        deployment.append(deployment_item)
    return oracle, deployment


def fit_models(
    target: str,
    oracle: Sequence[Mapping[str, Any]],
    deployment: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    oracle_r84 = [common.r84_row(row) for row in oracle]
    deployment_by_id = {int(row["sample_index"]): common.r84_row(row) for row in deployment}
    feature_names = sorted(oracle_r84[0]["feature_dict"])
    counts = expected_counts(target)
    models: dict[int, Any] = {}
    selection: list[dict[str, Any]] = []
    for class_id, gas in sorted(CLASS_NAMES.items()):
        class_rows = [row for row in oracle_r84 if int(row["true_class"]) == class_id]
        fit_rows, validation_seed = deterministic_train_val(class_rows, 0.25)
        validation_rows = [deployment_by_id[int(row["sample_index"])] for row in validation_seed]
        if len(class_rows) != counts["per_class"][class_id] or len(fit_rows) + len(validation_rows) != len(class_rows):
            raise RuntimeError(f"FAIL_CLOSED {target}/{gas} calibration count differs")
        truth = np.asarray([float(row["true_ppm"]) for row in validation_rows])
        best_alpha, best_rmse = RIDGE_ALPHAS[0], float("inf")
        grid: list[dict[str, float]] = []
        for alpha in RIDGE_ALPHAS:
            candidate = fit_ridge(fit_rows, feature_names, alpha)
            score = float(np.sqrt(np.mean((candidate.predict(validation_rows) - truth) ** 2)))
            grid.append({"alpha": float(alpha), "validation_RMSE": score})
            if score < best_rmse:
                best_alpha, best_rmse = alpha, score
        models[class_id] = fit_ridge(class_rows, feature_names, best_alpha)
        selection.append(
            {
                "experiment_id": f"{REGRESSION_EXPERIMENT_PREFIX}-{target}",
                "target": target,
                "class_id": class_id,
                "gas": gas,
                "calibration_fit_N": len(fit_rows),
                "calibration_validation_N": len(validation_rows),
                "calibration_refit_N": len(class_rows),
                "input_dimension": len(feature_names),
                "selected_alpha": float(best_alpha),
                "calibration_validation_RMSE": best_rmse,
                "alpha_grid_audit": json.dumps(grid, separators=(",", ":")),
                "selection_split": f"{target}_canonical_calibration_internal_75_25",
                "target_test_used_for_selection": False,
            }
        )
    return models, selection


def summarize(target: str, records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    main, per_gas, route, per_concentration = common.summarize(target, records)
    for rows in (main, per_gas, route, per_concentration):
        for row in rows:
            row["experiment_id"] = f"{REGRESSION_EXPERIMENT_PREFIX}-{target}"
            row["classifier_router"] = "A4"
            row["split_protocol"] = SPLIT_PROTOCOL
    return {"main": main, "per_gas": per_gas, "route": route, "per_concentration": per_concentration}


def special_slices(target: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    if target == "C5":
        selected = [
            row for row in records
            if str(row.get("gas")) == "methane" and math.isclose(float(row["true_ppm"]), 225.0)
            and int(row.get("repeat_id") or -1) == 1
        ]
        if not selected:
            raise RuntimeError("FAIL_CLOSED C5 methane225 repeat1 slice missing")
        slices.append({"target": target, "slice": "methane_225ppm_repeat1", **common.metrics(selected)})
    return slices


def run_target(
    classification_root: Path,
    output: Path,
    target: str,
    h1: Mapping[int, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    target_dir = output / target
    target_dir.mkdir(parents=True)
    checkpoint, provenance = checkpoint_for(classification_root, target)
    counts = expected_counts(target)
    calibration_routes, calibration_cls = route_rows(checkpoint, target, "calibration", device, batch_size)
    if len(calibration_routes) != counts["calibration"]:
        raise RuntimeError(f"FAIL_CLOSED {target} calibration N differs")
    oracle, deployment = prepare_rows(target, "calibration", calibration_routes, h1)
    models, selection = fit_models(target, oracle, deployment)
    common.write_csv(target_dir / "calibration_alpha_selection.csv", selection)
    common.write_json(target_dir / "regression_models.json", {str(key): model.to_json() for key, model in sorted(models.items())})
    lock = target_dir / "calibration_selection_lock.json"
    common.write_json(
        lock,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "SEALED_BEFORE_TARGET_TEST",
            "target": target,
            "target_test_opened": False,
            "fixed_alpha_grid": list(RIDGE_ALPHAS),
            "selection": selection,
            "classifier_checkpoint_provenance": provenance,
        },
    )
    common.validate_lock(lock)
    calibration_records = common.apply_r84_models(deployment, models)
    common.write_csv(target_dir / "calibration_records.csv", calibration_records)

    test_routes, test_cls = route_rows(checkpoint, target, "test", device, batch_size)
    if len(test_routes) != counts["test"]:
        raise RuntimeError(f"FAIL_CLOSED {target} test N differs")
    _oracle_test, deployment_test = prepare_rows(target, "test", test_routes, h1)
    records = common.apply_r84_models(deployment_test, models)
    summaries = summarize(target, records)
    common.write_csv(target_dir / "test_records.csv", records)
    common.write_csv(target_dir / "regression_summary.csv", summaries["main"])
    common.write_csv(target_dir / "regression_per_gas.csv", summaries["per_gas"])
    common.write_csv(target_dir / "regression_route_decomposition.csv", summaries["route"])
    common.write_csv(target_dir / "regression_per_concentration.csv", summaries["per_concentration"])
    slices = special_slices(target, records)
    if slices:
        common.write_csv(target_dir / "special_slices.csv", slices)
    common.write_json(
        target_dir / "target_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "target": target,
            "seed": 42,
            "calibration_N": len(calibration_records),
            "test_N": len(records),
            "classifier_checkpoint_provenance": provenance,
            "calibration_lock_sha256": common.sha256(lock),
            "calibration_classification": calibration_cls,
            "test_classification": test_cls,
            "target_test_used_for_selection": False,
        },
    )
    return {**summaries, "special_slices": slices, "provenance": provenance, "test_classification": test_cls}


def build(study_root: Path, device_text: str, batch_size: int) -> None:
    study_root = study_root.resolve()
    classification_root = study_root / "classification"
    output = study_root / "regression"
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED regression output exists: {output}")
    dataset_before = verify_dataset(DATA_ROOT)
    if dataset_before["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED canonical dataset hash gate failed")
    output.mkdir(parents=True)
    h1 = common.load_h1()
    results = {
        target: run_target(classification_root, output, target, h1, torch.device(device_text), batch_size)
        for target in TARGETS
    }
    for filename, key in (
        ("cross_target_r84_summary.csv", "main"),
        ("cross_target_r84_per_gas.csv", "per_gas"),
        ("cross_target_r84_route_decomposition.csv", "route"),
        ("cross_target_r84_per_concentration.csv", "per_concentration"),
    ):
        common.write_csv(output / filename, [row for target in TARGETS for row in results[target][key]])
    special = [row for target in TARGETS for row in results[target]["special_slices"]]
    if special:
        common.write_csv(output / "cross_target_r84_special_slices.csv", special)
    dataset_after = verify_dataset(DATA_ROOT)
    if dataset_after != dataset_before:
        raise RuntimeError("FAIL_CLOSED canonical dataset changed during R84")
    common.write_json(
        output / "protocol_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "study_id": STUDY_ID,
            "seed": 42,
            "dataset": dataset_after,
            "target_counts": {target: expected_counts(target) for target in TARGETS},
            "ridge_alpha_grid": list(RIDGE_ALPHAS),
            "h1_manifest_sha256": common.sha256(common.H1_PATH),
            "classifier_router": "A4",
            "target_test_used_for_selection": False,
            "hyperparameter_search_beyond_frozen_alpha_grid": False,
            "classifier_checkpoint_provenance": {target: results[target]["provenance"] for target in TARGETS},
        },
    )
    lookup = {(row["target"], row["evaluation_scope"]): row for target in TARGETS for row in results[target]["main"]}
    lines = [
        "# Canonical v1 A4 + R84_FED_H1 results", "",
        "| Target | Calibration/Test | Router Acc. | S_ALL RMSE | S_CC RMSE | S_ALL R2 | S_ALL NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        counts = expected_counts(target)
        all_row, cc_row = lookup[(target, "S_ALL")], lookup[(target, "S_CC")]
        lines.append(
            f"| {target} | {counts['calibration']}/{counts['test']} | {100*results[target]['test_classification']['accuracy']:.3f}% | "
            f"{all_row['RMSE']:.3f} | {cc_row['RMSE']:.3f} | {all_row['R2']:.4f} | {all_row['NRMSE']:.4f} |"
        )
    lines.extend(["", "Seed42 fixed-endpoint descriptive evidence; target test was opened only after each calibration lock was persisted."])
    (output / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "EXPERIMENT_AUDIT.md").write_text(
        "# Experiment audit\n\nVerdict: **PASS pending generated hash index.**\n\n"
        "- Final A4 LE1 target-specific checkpoints only.\n"
        "- Frozen 84-D feature definition, Federated-H1 asset, alpha grid, calibration-only selection, and full-calibration refit.\n"
        "- No target-test checkpoint, alpha, threshold, or model selection.\n",
        encoding="utf-8",
    )
    artifacts = [
        {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": common.sha256(path)}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "sha256_index.json"
    ]
    common.write_json(output / "sha256_index.json", {"schema_version": f"{SCHEMA_VERSION}.sha256", "status": "PASS", "artifacts": artifacts})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build(args.study_root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
