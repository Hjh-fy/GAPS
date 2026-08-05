"""Evaluate corrected role-aware target-specific GAPS routers with full R84."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gaps_flower.state_fingerprint import checkpoint_provenance
from run_regression_head_ablation import CLASS_NAMES, build_oracle_rows, fit_ridge
from scripts import run_gaps_cross_target_r84_full as common


ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "iotj_gaps_roleaware_r84_full_20260805"
DATA_ROOT = (
    ROOT.parents[1]
    / "dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"
)
DEFAULT_STUDY_ROOT = ROOT / "results" / STUDY_ID
TARGETS = ("C3", "C4", "C5")
SCHEMA_VERSION = "iotj.gaps_roleaware_r84_full.v1"


def expected_calibration_counts(target: str) -> dict[str, int]:
    values = {
        "C3": {
            "calibration": 680,
            "test": 2680,
            "per_gas": 170,
            "fit_per_gas": 130,
            "validation_per_gas": 40,
        },
        "C4": {
            "calibration": 320,
            "test": 1360,
            "per_gas": 80,
            "fit_per_gas": 60,
            "validation_per_gas": 20,
        },
        "C5": {
            "calibration": 320,
            "test": 1360,
            "per_gas": 80,
            "fit_per_gas": 60,
            "validation_per_gas": 20,
        },
    }
    try:
        return dict(values[target.upper()])
    except KeyError as exc:
        raise ValueError(f"unknown target: {target}") from exc


def checkpoint_for(classification_root: Path, target: str) -> tuple[Path, dict[str, Any]]:
    run_dir = classification_root / f"FCL-RW-GAPS-{target}"
    marker = json.loads((run_dir / "fixed_endpoint_complete.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError(f"FAIL_CLOSED missing {target} corrected checkpoint")
    provenance = checkpoint_provenance(checkpoint)
    if provenance["formal_round"] != 25:
        raise RuntimeError(f"FAIL_CLOSED {target} endpoint is not round 25")
    if provenance["whole_file_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"FAIL_CLOSED {target} checkpoint provenance differs")
    if marker.get("experiment_id") != f"FCL-RW-GAPS-{target}":
        raise RuntimeError(f"FAIL_CLOSED {target} completion marker differs")
    return checkpoint, provenance


def route_rows(
    checkpoint: Path,
    target: str,
    split: str,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, metrics = common.evaluate_checkpoint_stream(
        checkpoint,
        data_root=DATA_ROOT,
        target_client=int(target[1:]),
        split=split,
        device=device,
        batch_size=batch_size,
    )
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"FAIL_CLOSED {target}/{split} sample order differs")
    return rows, metrics


def prepare_rows(
    target: str,
    split: str,
    routes: Sequence[Mapping[str, Any]],
    h1: Mapping[int, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = build_oracle_rows(DATA_ROOT, [target], split)
    if len(base) != len(routes):
        raise RuntimeError(f"FAIL_CLOSED {target}/{split} route/data count differs")
    oracle: list[dict[str, Any]] = []
    deployment: list[dict[str, Any]] = []
    for row, route in zip(base, routes):
        if (
            int(row["sample_index"]) != int(route["sample_index"])
            or int(row["true_class"]) != int(route["true_class"])
        ):
            raise RuntimeError(f"FAIL_CLOSED {target}/{split} alignment differs")
        true_class = int(row["true_class"])
        pred_class = int(route["pred_class"])
        item = {**row, "pred_class": pred_class}
        oracle_item = dict(item)
        oracle_item["H1_federated_source_ridge_ppm"] = h1[true_class].predict(
            row["feature_dict"]
        )
        deploy_item = dict(item)
        deploy_item["H1_federated_source_ridge_ppm"] = h1[pred_class].predict(
            row["feature_dict"]
        )
        for class_id in range(4):
            deploy_item[f"prob_class_{class_id}"] = float(route[f"prob_{class_id}"])
        deploy_item["confidence"] = float(route["confidence"])
        oracle.append(oracle_item)
        deployment.append(deploy_item)
    return oracle, deployment


def fit_models(
    target: str,
    oracle: Sequence[Mapping[str, Any]],
    deployment: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    counts = expected_calibration_counts(target)
    oracle_r84 = [common.r84_row(row) for row in oracle]
    deployment_by_id = {
        int(row["sample_index"]): common.r84_row(row) for row in deployment
    }
    feature_names = sorted(oracle_r84[0]["feature_dict"])
    models: dict[int, Any] = {}
    selection: list[dict[str, Any]] = []
    for class_id, gas in sorted(CLASS_NAMES.items()):
        class_rows = [row for row in oracle_r84 if int(row["true_class"]) == class_id]
        fit_rows, validation_seed = common.deterministic_train_val(class_rows, 0.25)
        validation_rows = [
            deployment_by_id[int(row["sample_index"])] for row in validation_seed
        ]
        observed = (len(class_rows), len(fit_rows), len(validation_rows))
        expected = (
            counts["per_gas"],
            counts["fit_per_gas"],
            counts["validation_per_gas"],
        )
        if observed != expected:
            raise RuntimeError(
                f"FAIL_CLOSED {target}/{gas} calibration split {observed} != {expected}"
            )
        truth = np.asarray([float(row["true_ppm"]) for row in validation_rows])
        best_alpha = common.RIDGE_ALPHAS[0]
        best_rmse = float("inf")
        grid: list[dict[str, float]] = []
        for alpha in common.RIDGE_ALPHAS:
            candidate = fit_ridge(fit_rows, feature_names, alpha)
            prediction = candidate.predict(validation_rows)
            score = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            grid.append({"alpha": float(alpha), "validation_RMSE": score})
            if score < best_rmse:
                best_alpha, best_rmse = alpha, score
        models[class_id] = fit_ridge(class_rows, feature_names, best_alpha)
        selection.append(
            {
                "experiment_id": f"XTR-RW-R84-GAPS-{target}",
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
                "selection_split": f"{target}_roleaware_calibration_internal_75_25",
                "target_test_used_for_selection": False,
            }
        )
    return models, selection


def summarize(
    target: str, records: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_main, base_gas, base_route, base_concentration = common.summarize(target, records)
    experiment_id = f"XTR-RW-R84-GAPS-{target}"
    for rows in (base_main, base_gas, base_route, base_concentration):
        for row in rows:
            row["experiment_id"] = experiment_id
            row["split_protocol"] = "role_aware_target_20_80"
    return base_main, base_gas, base_route, base_concentration


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
    checkpoint, checkpoint_identity = checkpoint_for(classification_root, target)
    counts = expected_calibration_counts(target)

    calibration_routes, calibration_cls = route_rows(
        checkpoint, target, "calibration", device, batch_size
    )
    if len(calibration_routes) != counts["calibration"]:
        raise RuntimeError(f"FAIL_CLOSED {target} calibration count differs")
    oracle, deployment = prepare_rows(target, "calibration", calibration_routes, h1)
    models, selection = fit_models(target, oracle, deployment)
    common.write_csv(target_dir / "calibration_alpha_selection.csv", selection)
    common.write_json(
        target_dir / "regression_models.json",
        {str(key): value.to_json() for key, value in sorted(models.items())},
    )
    lock = target_dir / "calibration_selection_lock.json"
    common.write_json(
        lock,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "SEALED_BEFORE_TARGET_TEST",
            "target": target,
            "target_test_opened": False,
            "split_protocol": "role_aware_target_20_80",
            "fixed_alpha_grid": list(common.RIDGE_ALPHAS),
            "selection": selection,
            "classifier_checkpoint_provenance": checkpoint_identity,
        },
    )
    common.validate_lock(lock)
    calibration_records = common.apply_r84_models(deployment, models)
    common.write_csv(target_dir / "calibration_records.csv", calibration_records)

    test_routes, test_cls = route_rows(checkpoint, target, "test", device, batch_size)
    if len(test_routes) != counts["test"]:
        raise RuntimeError(f"FAIL_CLOSED {target} test count differs")
    _test_oracle, test_deployment = prepare_rows(target, "test", test_routes, h1)
    records = common.apply_r84_models(test_deployment, models)
    main, per_gas, route, per_concentration = summarize(target, records)
    common.write_csv(target_dir / "test_records.csv", records)
    common.write_csv(target_dir / "regression_summary.csv", main)
    common.write_csv(target_dir / "regression_per_gas.csv", per_gas)
    common.write_csv(target_dir / "regression_route_decomposition.csv", route)
    common.write_csv(target_dir / "regression_per_concentration.csv", per_concentration)
    common.write_json(
        target_dir / "target_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "experiment_id": f"XTR-RW-R84-GAPS-{target}",
            "target": target,
            "seed": 42,
            "split_protocol": "role_aware_target_20_80",
            "calibration_N": len(calibration_records),
            "test_N": len(records),
            "classifier_checkpoint_provenance": checkpoint_identity,
            "calibration_lock_sha256": common.sha256(lock),
            "calibration_classification": calibration_cls,
            "test_classification": test_cls,
            "target_test_used_for_selection": False,
        },
    )
    return {
        "main": main,
        "per_gas": per_gas,
        "route": route,
        "per_concentration": per_concentration,
        "test_classification": test_cls,
        "checkpoint_provenance": checkpoint_identity,
    }


def build(study_root: Path, device_text: str, batch_size: int) -> None:
    study_root = study_root.resolve()
    classification_root = study_root / "classification"
    output = study_root / "regression"
    if output.exists():
        raise FileExistsError(f"Refusing existing regression output: {output}")
    output.mkdir(parents=True)
    h1 = common.load_h1()
    device = torch.device(device_text)
    results: dict[str, Any] = {}
    for target in TARGETS:
        results[target] = run_target(
            classification_root, output, target, h1, device, batch_size
        )

    for name, key in (
        ("cross_target_r84_summary.csv", "main"),
        ("cross_target_r84_per_gas.csv", "per_gas"),
        ("cross_target_r84_route_decomposition.csv", "route"),
        ("cross_target_r84_per_concentration.csv", "per_concentration"),
    ):
        rows = [row for target in TARGETS for row in results[target][key]]
        common.write_csv(output / name, rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "study_id": STUDY_ID,
        "seed": 42,
        "split_protocol": "role_aware_target_20_80",
        "data_root": str(DATA_ROOT),
        "split_manifest_sha256": common.sha256(DATA_ROOT / "split_protocol_manifest.json"),
        "target_counts": {target: expected_calibration_counts(target) for target in TARGETS},
        "ridge_alpha_grid": list(common.RIDGE_ALPHAS),
        "classifier_retrained_for_corrected_split": True,
        "source_target_roles": {"source": ["C1", "C2"], "target": list(TARGETS)},
        "target_test_used_for_selection": False,
        "hyperparameter_search": False,
        "classifier_checkpoint_provenance": {
            target: results[target]["checkpoint_provenance"] for target in TARGETS
        },
    }
    common.write_json(output / "protocol_manifest.json", manifest)

    lookup = {
        (row["target"], row["evaluation_scope"]): row
        for target in TARGETS for row in results[target]["main"]
    }
    lines = [
        "# Corrected role-aware GAPS + R84 results",
        "",
        "All targets use the registered role-aware 20/80 split and a matching target-specific GAPS checkpoint trained without opening target test.",
        "",
        "| Target | Calibration/Test | Router Acc. | S_ALL RMSE | S_CC RMSE | S_ALL R2 | S_ALL NRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        counts = expected_calibration_counts(target)
        all_row = lookup[(target, "S_ALL")]
        cc_row = lookup[(target, "S_CC")]
        accuracy = results[target]["test_classification"]["accuracy"]
        lines.append(
            f"| {target} | {counts['calibration']}/{counts['test']} | {100*accuracy:.2f}% | "
            f"{all_row['RMSE']:.3f} | {cc_row['RMSE']:.3f} | {all_row['R2']:.4f} | {all_row['NRMSE']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Seed 42 is a fixed-endpoint descriptive result. Window-level observations are not independent client replicates, so no uncertainty or device-level significance claim is made.",
        ]
    )
    (output / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "EXPERIMENT_AUDIT.md").write_text(
        "# Experiment audit\n\nVerdict: **PASS only if all generated SHA and lock checks pass.**\n\n"
        "- Correct source/target roles and role-aware 20/80 counts are fixed.\n"
        "- Each classifier uses only its target calibration during interleaved GAPS adaptation.\n"
        "- Each regression lock is persisted before its target test is opened.\n"
        "- No target test checkpoint/alpha selection or hyperparameter search is permitted.\n"
        "- Seed 42 supports descriptive fixed-endpoint claims only.\n",
        encoding="utf-8",
    )
    artifacts = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "sha256_index.json"):
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": common.sha256(path),
            }
        )
    common.write_json(
        output / "sha256_index.json",
        {"schema_version": f"{SCHEMA_VERSION}.sha256", "status": "PASS", "artifacts": artifacts},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build(args.study_root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
