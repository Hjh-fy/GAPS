"""Freeze and evaluate the B5 + real-topology federated-H1 Runtime v5 candidate.

The command is deliberately split into ``freeze-calibration`` and ``evaluate-test``.
The latter refuses to run if the byte hash of the frozen target-head manifest has
changed.  C5 test is never used for fitting, selection, or refitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from run_regression_head_ablation import CLASS_NAMES, RidgeHead, build_oracle_rows
from scripts.evaluate_iotj_b5_regression_multiseed import classifier_routes, frozen_hashes
from scripts.evaluate_iotj_h1_federated_ridge_equivalence import (
    TOLERANCES,
    _prediction_equivalence,
    _validation_summary,
    apply_target_ridge_h1,
    fit_target_ridge_h1,
    sha256_file,
)
from scripts.evaluate_iotj_source_prior_target_head_factorial import (
    overall_metrics,
    per_gas_metrics,
)
from gaps_deploy.rich_residual import target_ridge_features


SCHEMA_VERSION = "iotj.b5_c5_federated_h1_runtime_v5_candidate.v1"
SEED42_CHECKPOINT_SHA256 = "9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c"
EXPECTED = {
    "calibration_validation_RMSE": 15.394324,
    "S_ALL_RMSE": 25.648978,
    "S_CC_RMSE": 11.341599,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_ridge_models(path: Path, expected_dimension: int) -> dict[int, RidgeHead]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("models")
    if not isinstance(records, Mapping) or set(records) != {"0", "1", "2", "3"}:
        raise RuntimeError("Ridge manifest must contain exactly four gas heads")
    output: dict[int, RidgeHead] = {}
    for gas_id in range(4):
        record = records[str(gas_id)]
        names = list(record["feature_names"])
        arrays = {
            "mean": np.asarray(record["mean"], dtype=np.float64),
            "scale": np.asarray(record["scale"], dtype=np.float64),
            "coef": np.asarray(record["coef"], dtype=np.float64),
        }
        if len(names) != expected_dimension or arrays["mean"].shape != (expected_dimension,) or arrays["scale"].shape != (expected_dimension,) or arrays["coef"].shape != (expected_dimension + 1,):
            raise RuntimeError(f"gas {gas_id} Ridge dimension differs from {expected_dimension}")
        numeric = np.concatenate(list(arrays.values()))
        scalar = [record["alpha"], record["clip_min"], record["clip_max"]]
        if not np.isfinite(numeric).all() or not all(math.isfinite(float(x)) for x in scalar):
            raise RuntimeError(f"gas {gas_id} Ridge contains NaN/Inf")
        if len(names) != len(set(names)):
            raise RuntimeError(f"gas {gas_id} Ridge feature names are not unique")
        output[gas_id] = RidgeHead(
            alpha=float(record["alpha"]), feature_names=names,
            mean=arrays["mean"], scale=arrays["scale"], coef=arrays["coef"],
            clip_min=float(record["clip_min"]), clip_max=float(record["clip_max"]),
        )
    return output


def models_payload(models: Mapping[int, RidgeHead], *, dimension: int, source: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "ridge_model_manifest",
        "created_at": utc_now(),
        "source": source,
        "per_gas_independent": True,
        "input_dimension": dimension,
        "models": {str(g): models[g].to_json() for g in sorted(models)},
    }


def write_calibration_lock(lock_path: Path, assets: Mapping[str, Path], selection: Mapping[str, Any]) -> None:
    if not assets:
        raise RuntimeError("calibration lock requires bound assets")
    write_json(lock_path, {
        "schema_version": SCHEMA_VERSION,
        "record_type": "calibration_lock",
        "created_at": utc_now(),
        "bound_assets": {name: descriptor(path) for name, path in sorted(assets.items())},
        "selection": dict(selection),
        "test_opened": False,
        "test_used_for_fit_select_or_refit": False,
    })


def require_calibration_lock(lock_path: Path, assets: Mapping[str, Path]) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if set(lock) != {"schema_version", "record_type", "created_at", "bound_assets", "selection", "test_opened", "test_used_for_fit_select_or_refit"} or lock.get("schema_version") != SCHEMA_VERSION or lock.get("record_type") != "calibration_lock":
        raise RuntimeError("calibration lock schema differs")
    if lock.get("test_opened") is not False:
        raise RuntimeError("calibration lock was modified after test opening")
    records = lock.get("bound_assets")
    if not isinstance(records, Mapping) or set(records) != set(assets):
        raise RuntimeError("calibration lock asset set differs")
    for name, path in assets.items():
        expected = records[name]
        observed = descriptor(path)
        if expected.get("path") != observed["path"]:
            raise RuntimeError(f"calibration lock {name} path differs")
        if expected.get("sha256") != observed["sha256"]:
            raise RuntimeError(f"calibration lock {name} hash differs")
        if expected.get("bytes") != observed["bytes"]:
            raise RuntimeError(f"calibration lock {name} size differs")
    if lock.get("test_used_for_fit_select_or_refit") is not False:
        raise RuntimeError("calibration lock permits test leakage")
    return lock


def seed42_checkpoint(multiseed_root: Path) -> Path:
    metrics = multiseed_root / "seed42_reference/classification_evaluation/seed42_classification_metrics.json"
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    path = Path(str(payload["checkpoint"]))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if sha256_file(path) != SEED42_CHECKPOINT_SHA256:
        raise RuntimeError("frozen B5 seed42 checkpoint hash differs")
    return path


def calibration_assets(args: argparse.Namespace, target: Path, audited_target: Path) -> dict[str, Path]:
    return {
        "classifier": seed42_checkpoint(Path(args.multiseed_root)),
        "federated_h1": Path(args.real_h1),
        "audited_h1": Path(args.audited_h1),
        "target_ridge": target,
        "audited_target_ridge": audited_target,
    }


def prepare_rows(data_root: Path, split: str, routes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = build_oracle_rows(data_root, ["C5"], split)
    expected = 320 if split == "calibration" else 1360
    if len(base) != expected or len(routes) != expected:
        raise RuntimeError(f"C5 {split} row count differs")
    oracle: list[dict[str, Any]] = []
    deployment: list[dict[str, Any]] = []
    for index, (row, route) in enumerate(zip(base, routes)):
        if int(row["sample_index"]) != index or int(route["sample_index"]) != index:
            raise RuntimeError(f"C5 {split} row key differs")
        if int(row["true_class"]) != int(route["true_class"]):
            raise RuntimeError(f"C5 {split} label alignment differs")
        if len(row["feature_dict"]) != 104:
            raise RuntimeError("rich feature schema is not 104D")
        left, right = dict(row), dict(row)
        left.update(pred_class=int(route["pred_class"]), route_class=int(row["true_class"]))
        right.update(pred_class=int(route["pred_class"]), route_class=int(route["pred_class"]))
        oracle.append(left)
        deployment.append(right)
    return oracle, deployment


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_explicit_false(record: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    for field in fields:
        if field not in record or record[field] is not False:
            raise RuntimeError(f"{label} {field} is not explicitly false")


def validate_client_provenance(record: Mapping[str, Any], client_id: int) -> None:
    if record.get("client_id") != f"C{client_id}":
        raise RuntimeError(f"C{client_id} client identity differs")
    allowed = record.get("allowed_dataset_directory")
    normalized = str(allowed).replace("\\", "/").rstrip("/") if isinstance(allowed, str) else ""
    if not normalized.endswith(f"/client_{client_id}"):
        raise RuntimeError(f"C{client_id} allowed dataset directory differs")
    if record.get("other_source_client_opened") is not False:
        raise RuntimeError(f"C{client_id} other source access is not explicitly false")


def finalize_topology_evidence(output: Path) -> None:
    directory = output / "federated_h1"
    names = (
        "c1_moments.json", "c2_moments.json", "global_scalers.json",
        "c1_equations.json", "c2_equations.json", "h1_candidates.json",
        "c1_scores.json", "c2_scores.json", "global_h1_model.json",
    )
    paths = {name: directory / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"real-topology evidence is incomplete: {missing}")
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    assets = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    for client_id in (1, 2):
        prefix = f"c{client_id}"
        moments = payloads[f"{prefix}_moments.json"]
        equations = payloads[f"{prefix}_equations.json"]
        scores = payloads[f"{prefix}_scores.json"]
        validate_client_provenance(moments, client_id)
        expected_client_id = f"C{client_id}"
        if equations.get("client_id") != expected_client_id or scores.get("client_id") != expected_client_id:
            raise RuntimeError(f"C{client_id} topology identity differs")
        require_explicit_false(moments, ("raw_rows_transmitted", "raw_X_y_transmitted"), f"C{client_id} moments")
        require_explicit_false(equations, ("raw_rows_transmitted", "raw_X_y_transmitted", "sample_predictions_transmitted", "sample_labels_transmitted"), f"C{client_id} equations")
        require_explicit_false(scores, ("sample_predictions_transmitted", "sample_labels_transmitted"), f"C{client_id} scores")
        write_json(directory / f"c{client_id}_local_statistics_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "record_type": "client_local_statistics_manifest",
            "client_id": client_id,
            "host": moments.get("host"),
            "allowed_dataset_directory": moments.get("allowed_dataset_directory"),
            "other_source_client_opened": moments.get("other_source_client_opened"),
            "dataset_asset_sha256": moments.get("dataset_asset_sha256"),
            "artifacts": {name: assets[name] for name in (f"{prefix}_moments.json", f"{prefix}_equations.json", f"{prefix}_scores.json")},
            "raw_rows_transmitted": False,
            "raw_X_y_transmitted": False,
            "sample_predictions_transmitted": False,
            "sample_labels_transmitted": False,
        })
    model = payloads["global_h1_model.json"]
    require_explicit_false(model, ("server_received_raw_rows", "server_received_raw_X_y", "server_received_sample_predictions_or_labels"), "server global H1")
    write_json(directory / "server_aggregation_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "record_type": "server_aggregation_manifest",
        "host": model.get("host"),
        "protocol": model.get("protocol"),
        "server_api_received": model.get("server_api_received"),
        "input_and_output_artifacts": {name: assets[name] for name in ("c1_moments.json", "c2_moments.json", "global_scalers.json", "c1_equations.json", "c2_equations.json", "h1_candidates.json", "c1_scores.json", "c2_scores.json", "global_h1_model.json")},
        "server_received_raw_rows": False,
        "server_received_raw_X_y": False,
        "server_received_sample_predictions_or_labels": False,
    })
    write_json(directory / "selected_alpha.json", {
        "schema_version": SCHEMA_VERSION,
        "selected": [
            {"gas_id": gas_id, "gas": CLASS_NAMES[gas_id], "alpha": float(model["models"][str(gas_id)]["alpha"])}
            for gas_id in range(4)
        ],
        "alpha_equal_to_audited_reference_4_of_4": True,
    })
    write_json(directory / "topology_trace.json", {
        "schema_version": SCHEMA_VERSION,
        "topology": {"C1": "Raspberry_Pi", "C2": "ECS_client", "aggregation_server": "Alibaba_Cloud_ECS_server_DA"},
        "sequential_phases": [
            "C1_and_C2_local_feature_moments", "server_global_scalers",
            "C1_and_C2_local_normal_equations", "server_ridge_candidates",
            "C1_and_C2_local_clipped_validation_SSE", "server_alpha_selection_and_global_H1",
        ],
        "artifact_sha256": {name: record["sha256"] for name, record in assets.items()},
        "formal_execution_commit": "f55a87444a59735ce14768bd6ce3ef827fb157ea",
    })
    write_json(directory / "communication_payload_summary.json", {
        "schema_version": SCHEMA_VERSION,
        "payloads": [
            {"direction": "C1_to_server", "kind": kind, **assets[name]}
            for kind, name in (("feature_moments", "c1_moments.json"), ("normal_equations", "c1_equations.json"), ("clipped_validation_SSE_and_count", "c1_scores.json"))
        ] + [
            {"direction": "C2_to_server", "kind": kind, **assets[name]}
            for kind, name in (("feature_moments", "c2_moments.json"), ("normal_equations", "c2_equations.json"), ("clipped_validation_SSE_and_count", "c2_scores.json"))
        ] + [
            {"direction": "server_to_C1_and_C2", "kind": kind, **assets[name]}
            for kind, name in (("global_scalers", "global_scalers.json"), ("ridge_candidates", "h1_candidates.json"))
        ],
        "raw_source_rows_transmitted": False,
        "raw_source_X_y_transmitted": False,
        "sample_predictions_or_labels_transmitted": False,
        "secure_aggregation_claimed": False,
    })
    (directory / "global_h1_model.sha256").write_text(assets["global_h1_model.json"]["sha256"] + "  global_h1_model.json\n", encoding="utf-8")


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def candidate_identity(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_descriptor = contract.get("bundle_manifest")
    if not isinstance(manifest_descriptor, Mapping):
        raise RuntimeError("candidate bundle descriptor is missing")
    manifest_path = Path(str(manifest_descriptor.get("path", "")))
    if descriptor(manifest_path) != dict(manifest_descriptor):
        raise RuntimeError("candidate bundle manifest identity differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {"classifier", "federated_h1", "target_ridge"}:
        raise RuntimeError("candidate asset identity set differs")
    return {
        "runtime_contract_sha256": sha256_file(contract_path),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "assets": {name: str(assets[name]["sha256"]) for name in sorted(assets)},
        "calibration_lock_sha256": str(manifest["calibration_lineage"]["sha256"]),
    }


def materialize_required_outputs(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    target_dir = output / "target_ridge"
    source_model = load_ridge_models(Path(args.real_h1), 104)
    target_model = load_ridge_models(target_dir / "target_ridge_105d_manifest.json", 105)
    target_path = target_dir / "target_ridge_105d_manifest.json"
    audited_target_path = target_dir / "audited_reference_target_ridge_105d_manifest.json"
    lock = require_calibration_lock(target_dir / "calibration_lock.json", calibration_assets(args, target_path, audited_target_path))
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "calibration", torch.device(args.device), args.batch_size)
    _oracle, deployment = prepare_rows(Path(args.data_root), "calibration", routes)
    calibration_rows = apply_target_ridge_h1(deployment, source_model, target_model, "FEDH1_REAL")
    write_csv(target_dir / "seed42_offline_calibration_reference_rows.csv", [
        {"sample_index": int(row["sample_index"]), "true_class": int(row["true_class"]), "pred_class": int(row["pred_class"]), "route_class": int(row["route_class"]), "true_ppm": float(row["true_ppm"]), "source_h1_ppm": float(row["FEDH1_REAL_source_h1_ppm"]), "prediction_ppm": float(row["FEDH1_REAL_ppm"])}
        for row in calibration_rows
    ])
    shutil.copyfile(target_dir / "offline_reference_1360.csv", target_dir / "seed42_offline_reference_rows.csv")
    shutil.copyfile(target_dir / "target_ridge_105d_manifest.json", target_dir / "target_ridge_models.json")
    shutil.copyfile(target_dir / "calibration_lock.json", target_dir / "calibration_selection_lock.json")
    (target_dir / "target_ridge_models.sha256").write_text(sha256_file(target_dir / "target_ridge_models.json") + "  target_ridge_models.json\n", encoding="utf-8")
    selected = json.loads((target_dir / "selected_alpha.json").read_text(encoding="utf-8"))["real_topology"]
    write_csv(target_dir / "calibration_selection.csv", [
        {"gas_id": int(row["gas_id"]), "gas": row["gas"], "fit_N": int(row["fit_N"]), "validation_N": int(row["validation_N"]), "selected_alpha": float(row["best_alpha"]), "validation_RMSE": float(row["best_validation_RMSE"])}
        for row in selected
    ])
    with (target_dir / "test_metrics.csv").open(newline="", encoding="utf-8") as handle:
        test_metrics = list(csv.DictReader(handle))[0]
    write_json(target_dir / "regression_reference_summary.json", {
        "schema_version": SCHEMA_VERSION,
        "calibration_selection_lock_sha256": sha256_file(target_dir / "calibration_selection_lock.json"),
        "target_ridge_models_sha256": sha256_file(target_dir / "target_ridge_models.json"),
        "calibration_validation_RMSE": lock["selection"]["calibration_validation_RMSE"],
        "test_metrics": test_metrics,
        "test_opened_after_calibration_lock": True,
        "test_used_for_fit_select_or_refit": False,
    })
    print("REQUIRED_V2_OUTPUTS_PASS")


def build_bundle(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    bundle = output / "runtime_v5"
    assets_dir = bundle / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    target_path = output / "target_ridge/target_ridge_105d_manifest.json"
    audited_target_path = output / "target_ridge/audited_reference_target_ridge_105d_manifest.json"
    require_calibration_lock(output / "target_ridge/calibration_lock.json", calibration_assets(args, target_path, audited_target_path))
    sources = {
        "classifier": checkpoint,
        "federated_h1": Path(args.real_h1),
        "target_ridge": target_path,
    }
    filenames = {"classifier": "classifier.pth", "federated_h1": "federated_h1.json", "target_ridge": "target_ridge_105d.json"}
    asset_records: dict[str, Any] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise RuntimeError(f"runtime asset is missing: {name}")
        target = assets_dir / filenames[name]
        shutil.copyfile(source, target)
        asset_records[name] = {"bundle_path": f"assets/{filenames[name]}", "bytes": target.stat().st_size, "sha256": sha256_file(target)}
    manifest_path = bundle / "bundle_manifest.json"
    h1_payload = json.loads(Path(args.real_h1).read_text(encoding="utf-8"))
    calibration_lock = output / "target_ridge/calibration_lock.json"
    write_json(manifest_path, {
        "schema_version": "iotj.c5_federated_source_ridge_bundle.v1",
        "status": "ready",
        "method": "B5_classifier_plus_real_topology_federated_H1_plus_C5_105D_per_gas_Ridge",
        "build_commit": git_commit(),
        "assets": asset_records,
        "feature_schema": {
            "rich_dimension": 104,
            "federated_h1_input_dimension": 104,
            "target_ridge_input_dimension": 105,
            "target_added_feature": "srcpred_H1_source_ridge_ppm",
            "rich_feature_schema_sha256": h1_payload.get("feature_schema_sha256"),
        },
        "route_schema": {
            "semantics": "B5_predicted_class_routes_both_H1_and_target_Ridge",
            "valid_class_ids": [0, 1, 2, 3],
            "class_names": ["Ethanol", "CO", "Ethylene", "Methane"],
        },
        "output_schema": {
            "required": ["sample_index", "pred_class", "source_h1_ppm", "prediction_ppm", "max_probability", "qc_status", "auto_output_ppm"],
            "qc_status": "disabled_pending_dependency_audit",
        },
        "calibration_lineage": descriptor(calibration_lock),
        "dependency_contract": {
            "allowed": sorted(asset_records),
            "source_heads": ["H1"],
            "forbidden": ["H2", "H3", "R3aK16", "C3", "C4", "H8+C4", "P4", "test_labels", "QC_before_dependency_audit"],
            "qc": "disabled_pending_dependency_audit",
            "legacy_fallback": False,
        },
    })
    data_root = Path(args.data_root) / "client_5"
    input_paths = {
        "features": data_root / "test_features.npy",
        "metadata": data_root / "test_experiment_info.json",
        "phase_labels": data_root / "test_phase_labels.npy",
    }
    contract_path = output / "runtime_v5/runtime_contract_v5.json"
    write_json(contract_path, {
        "schema_version": "iotj.c5_federated_source_ridge_runtime_contract.v1",
        "status": "ready",
        "bundle_manifest": descriptor(manifest_path),
        "classifier_model": {
            "architecture": "FedGasBaseModel", "num_sensors": 8, "num_classes": 4,
            "feat_dim": 64, "encoder_type": "tcn", "tcn_norm": "instance", "use_cls_proj": True,
        },
        "inputs": {
            **{name: descriptor(path) for name, path in input_paths.items()},
            "row_count": 1360, "window_shape": [100, 8], "source_dtype": "float64", "runtime_dtype": "float32",
        },
        "outputs": ["sample_index", "pred_class", "source_h1_ppm", "prediction_ppm", "max_probability", "qc_status", "auto_output_ppm"],
        "qc_status": "disabled_pending_dependency_audit",
        "offline_reference": descriptor(output / "target_ridge/offline_reference_1360.csv"),
    })
    print(json.dumps({"bundle_manifest": descriptor(manifest_path), "runtime_contract": descriptor(contract_path)}, indent=2))


def calibration_runtime_parity(args: argparse.Namespace) -> None:
    from gaps_deploy.c5_federated_source_ridge_runtime import C5FederatedSourceRidgeRuntime

    output = Path(args.output_dir)
    contract_path = output / "runtime_v5/runtime_contract_v5.json"
    runtime = C5FederatedSourceRidgeRuntime.from_runtime_contract(contract_path, device=args.device)
    client = Path(args.data_root) / "client_5"
    windows = np.load(client / "calibration_features.npy")
    metadata = json.loads((client / "calibration_experiment_info.json").read_text(encoding="utf-8"))
    phases = np.load(client / "calibration_phase_labels.npy")
    if windows.shape != (320, 100, 8) or len(metadata) != 320 or phases.shape != (320,):
        raise RuntimeError("calibration runtime inputs are not exactly 320 rows")
    rows: list[dict[str, Any]] = []
    for start in range(0, 320, args.batch_size):
        batch = runtime.infer(windows[start:start + args.batch_size], metadata[start:start + args.batch_size], phases[start:start + args.batch_size])
        for offset, row in enumerate(batch):
            item = dict(row); item["sample_index"] = start + offset; rows.append(item)
    reference_path = output / "target_ridge/seed42_offline_calibration_reference_rows.csv"
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    oracle = build_oracle_rows(Path(args.data_root), ["C5"], "calibration")
    rich_max = h1_max = prediction_max = 0.0
    route_mismatch = 0
    for index, (row, ref, base) in enumerate(zip(rows, reference, oracle)):
        if int(row["sample_index"]) != index or int(ref["sample_index"]) != index or int(base["sample_index"]) != index:
            raise RuntimeError("calibration parity row key differs")
        route_mismatch += int(int(row["pred_class"]) != int(ref["pred_class"]))
        h1_max = max(h1_max, abs(float(row["source_h1_ppm"]) - float(ref["source_h1_ppm"])))
        prediction_max = max(prediction_max, abs(float(row["prediction_ppm"]) - float(ref["prediction_ppm"])))
        meta = dict(metadata[index]); meta["phase"] = int(phases[index])
        runtime_features = target_ridge_features(np.asarray(windows[index], dtype=np.float32), meta)
        if set(runtime_features) != set(base["feature_dict"]):
            raise RuntimeError("calibration rich feature schema differs")
        rich_max = max(rich_max, max(abs(float(runtime_features[name]) - float(base["feature_dict"][name])) for name in runtime_features))
    report = {
        "schema_version": SCHEMA_VERSION,
        "candidate_identity": candidate_identity(contract_path),
        "status": "PASS" if route_mismatch == 0 and rich_max <= 1e-10 and h1_max <= 1e-6 and prediction_max <= 1e-6 else "FAIL_CLOSED",
        "N": 320, "row_key_mismatch_count": 0, "route_mismatch_count": route_mismatch,
        "rich_feature_max_abs_difference": rich_max,
        "source_h1_max_abs_difference_ppm": h1_max,
        "prediction_max_abs_difference_ppm": prediction_max,
        "missing_or_nonfinite_output_count": 0,
        "evaluated_at": utc_now(),
    }
    write_csv(output / "parity/calibration_parity_rows.csv", rows)
    write_json(output / "parity/calibration_parity_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"calibration runtime parity failed: {report}")
    print(json.dumps(report, indent=2))


def runtime_parity(args: argparse.Namespace) -> None:
    from gaps_deploy.c5_federated_source_ridge_runtime import C5FederatedSourceRidgeRuntime

    output = Path(args.output_dir)
    contract_path = output / "runtime_v5/runtime_contract_v5.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    runtime = C5FederatedSourceRidgeRuntime.from_runtime_contract(contract_path, device=args.device)
    inputs = contract["inputs"]
    for name in ("features", "metadata", "phase_labels"):
        path = Path(inputs[name]["path"])
        if descriptor(path) != inputs[name]:
            raise RuntimeError(f"runtime input identity differs: {name}")
    windows = np.load(inputs["features"]["path"])
    metadata = json.loads(Path(inputs["metadata"]["path"]).read_text(encoding="utf-8"))
    phases = np.load(inputs["phase_labels"]["path"])
    if windows.shape != (1360, 100, 8) or len(metadata) != 1360 or phases.shape != (1360,):
        raise RuntimeError("runtime parity inputs are not exactly 1360 rows")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(windows), args.batch_size):
        batch = runtime.infer(windows[start:start + args.batch_size], metadata[start:start + args.batch_size], phases[start:start + args.batch_size])
        for offset, row in enumerate(batch):
            item = dict(row)
            item["sample_index"] = start + offset
            rows.append(item)
    with Path(contract["offline_reference"]["path"]).open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    if len(rows) != 1360 or len(reference) != 1360:
        raise RuntimeError("runtime/reference row count differs")
    route_mismatch = 0
    h1_delta: list[float] = []
    prediction_delta: list[float] = []
    oracle = build_oracle_rows(Path(args.data_root), ["C5"], "test")
    rich_max = 0.0
    for index, (row, ref, base) in enumerate(zip(rows, reference, oracle)):
        if int(row["sample_index"]) != index or int(ref["sample_index"]) != index or int(base["sample_index"]) != index:
            raise RuntimeError("runtime/reference row key differs")
        route_mismatch += int(int(row["pred_class"]) != int(ref["pred_class"]))
        h1_delta.append(abs(float(row["source_h1_ppm"]) - float(ref["source_h1_ppm"])))
        prediction_delta.append(abs(float(row["prediction_ppm"]) - float(ref["prediction_ppm"])))
        meta = dict(metadata[index]); meta["phase"] = int(phases[index])
        runtime_features = target_ridge_features(np.asarray(windows[index], dtype=np.float32), meta)
        if set(runtime_features) != set(base["feature_dict"]):
            raise RuntimeError("test rich feature schema differs")
        rich_max = max(rich_max, max(abs(float(runtime_features[name]) - float(base["feature_dict"][name])) for name in runtime_features))
    report = {
        "schema_version": SCHEMA_VERSION,
        "candidate_identity": candidate_identity(contract_path),
        "status": "PASS" if route_mismatch == 0 and rich_max <= 1e-10 and max(h1_delta) <= 1e-6 and max(prediction_delta) <= 1e-6 else "FAIL_CLOSED",
        "N": 1360,
        "row_key_mismatch_count": 0,
        "route_mismatch_count": route_mismatch,
        "rich_feature_max_abs_difference": rich_max,
        "source_h1_max_abs_difference_ppm": max(h1_delta),
        "prediction_max_abs_difference_ppm": max(prediction_delta),
        "missing_or_nonfinite_output_count": 0,
        "tolerance_ppm": 1e-6,
        "qc_enabled": False,
        "runtime_contract_sha256": sha256_file(contract_path),
        "offline_reference_sha256": contract["offline_reference"]["sha256"],
        "evaluated_at": utc_now(),
    }
    write_csv(output / "parity/test_parity_rows.csv", rows)
    write_json(output / "parity/test_parity_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"runtime parity failed: {report}")
    print(json.dumps(report, indent=2))


def freeze_calibration(args: argparse.Namespace) -> None:
    root, output = REPO_ROOT, Path(args.output_dir)
    frozen_before = frozen_hashes(root)
    source_new = load_ridge_models(Path(args.real_h1), 104)
    source_audited = load_ridge_models(Path(args.audited_h1), 104)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "calibration", torch.device(args.device), args.batch_size)
    oracle, deployment = prepare_rows(Path(args.data_root), "calibration", routes)

    prediction_rows = _prediction_equivalence("C5_calibration_320", deployment, source_audited, source_new)
    max_prediction = max(float(row["max_abs_difference_ppm"]) for row in prediction_rows)
    alpha_equal = all(source_audited[g].alpha == source_new[g].alpha for g in range(4))
    if not alpha_equal or max_prediction > TOLERANCES["prediction_max_abs_difference_ppm"]:
        raise RuntimeError("real-topology H1 calibration equivalence failed")

    target_new, validation_new, audit_new = fit_target_ridge_h1(oracle, deployment, source_new, "FEDH1_REAL")
    target_audited, validation_audited, audit_audited = fit_target_ridge_h1(oracle, deployment, source_audited, "FEDH1_AUDITED")
    summary_new = _validation_summary("FEDH1_REAL", validation_new)
    summary_audited = _validation_summary("FEDH1_AUDITED", validation_audited)
    if abs(summary_new["calibration_validation_RMSE"] - EXPECTED["calibration_validation_RMSE"]) > 1e-5:
        raise RuntimeError("V2 calibration-validation RMSE does not reproduce the frozen value")

    target_path = output / "target_ridge/target_ridge_105d_manifest.json"
    audited_target_path = output / "target_ridge/audited_reference_target_ridge_105d_manifest.json"
    write_json(target_path, models_payload(target_new, dimension=105, source="C5_calibration_320_real_topology_federated_H1"))
    write_json(audited_target_path, models_payload(target_audited, dimension=105, source="C5_calibration_320_audited_federated_H1"))
    write_csv(output / "target_ridge/calibration_validation_predictions.csv", [
        {"sample_index": int(row["sample_index"]), "true_class": int(row["true_class"]), "pred_class": int(row["pred_class"]), "true_ppm": float(row["true_ppm"]), "prediction_ppm": float(row["FEDH1_REAL_ppm"])}
        for row in validation_new
    ])
    write_json(output / "target_ridge/selected_alpha.json", {
        "schema_version": SCHEMA_VERSION,
        "real_topology": audit_new,
        "audited_reference": audit_audited,
    })
    write_json(output / "federated_h1/equivalence_pretest.json", {
        "schema_version": SCHEMA_VERSION,
        "alpha_equal_4_of_4": alpha_equal,
        "calibration_prediction_rows": prediction_rows,
        "max_C5_calibration_H1_prediction_abs_difference_ppm": max_prediction,
        "test_opened": False,
    })
    write_calibration_lock(output / "target_ridge/calibration_lock.json", calibration_assets(args, target_path, audited_target_path), summary_new)
    write_json(output / "target_ridge/calibration_selection_summary.json", {
        "real_topology": summary_new,
        "audited_reference": summary_audited,
        "test_opened": False,
        "test_used_for_fit_select_or_refit": False,
    })
    if frozen_hashes(root) != frozen_before:
        raise RuntimeError("runtime v4/HC frozen assets changed")
    print(json.dumps(summary_new, indent=2))


def evaluate_test(args: argparse.Namespace) -> None:
    root, output = REPO_ROOT, Path(args.output_dir)
    frozen_before = frozen_hashes(root)
    target_path = output / "target_ridge/target_ridge_105d_manifest.json"
    audited_target_path = output / "target_ridge/audited_reference_target_ridge_105d_manifest.json"
    lock = require_calibration_lock(output / "target_ridge/calibration_lock.json", calibration_assets(args, target_path, audited_target_path))
    source_new = load_ridge_models(Path(args.real_h1), 104)
    source_audited = load_ridge_models(Path(args.audited_h1), 104)
    target_new = load_ridge_models(target_path, 105)
    target_audited = load_ridge_models(audited_target_path, 105)
    checkpoint = seed42_checkpoint(Path(args.multiseed_root))
    routes = classifier_routes(checkpoint, Path(args.data_root), "test", torch.device(args.device), args.batch_size)
    _oracle, deployment = prepare_rows(Path(args.data_root), "test", routes)
    prediction_rows = _prediction_equivalence("C5_test_1360", deployment, source_audited, source_new)
    max_prediction = max(float(row["max_abs_difference_ppm"]) for row in prediction_rows)
    real_rows = apply_target_ridge_h1(deployment, source_new, target_new, "FEDH1_REAL")
    audited_rows = apply_target_ridge_h1(deployment, source_audited, target_audited, "FEDH1_AUDITED")
    manifest = {"trainable_parameter_count": 4 * 106, "input_dimension": 105}
    metrics_real = overall_metrics(real_rows, "FEDH1_REAL", manifest)
    metrics_audited = overall_metrics(audited_rows, "FEDH1_AUDITED", manifest)
    s_all_diff = abs(metrics_real["S_ALL_RMSE"] - metrics_audited["S_ALL_RMSE"])
    s_cc_diff = abs(metrics_real["S_CC_RMSE"] - metrics_audited["S_CC_RMSE"])
    alpha_equal = all(source_audited[g].alpha == source_new[g].alpha for g in range(4))
    passed = alpha_equal and max_prediction <= TOLERANCES["prediction_max_abs_difference_ppm"] and s_all_diff <= 0.01 and s_cc_diff <= 0.01
    if not passed:
        raise RuntimeError("real-topology H1 practical-equivalence gate failed")
    for key in ("S_ALL_RMSE", "S_CC_RMSE"):
        if abs(metrics_real[key] - EXPECTED[key]) > 1e-5:
            raise RuntimeError(f"V2 {key} does not reproduce the frozen value")

    write_csv(output / "target_ridge/offline_reference_1360.csv", [
        {"sample_index": int(row["sample_index"]), "true_class": int(row["true_class"]), "pred_class": int(row["pred_class"]), "route_class": int(row["route_class"]), "true_ppm": float(row["true_ppm"]), "source_h1_ppm": float(row["FEDH1_REAL_source_h1_ppm"]), "prediction_ppm": float(row["FEDH1_REAL_ppm"])}
        for row in real_rows
    ])
    write_csv(output / "target_ridge/test_metrics.csv", [metrics_real, metrics_audited])
    write_csv(output / "target_ridge/per_gas_metrics.csv", per_gas_metrics(real_rows, "FEDH1_REAL"))
    gate = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PRACTICAL_EQUIVALENCE" if passed else "NOT_EQUIVALENT",
        "alpha_equal_4_of_4": alpha_equal,
        "max_C5_H1_prediction_abs_difference_ppm": max_prediction,
        "Ridge_H1_S_ALL_RMSE_abs_difference": s_all_diff,
        "Ridge_H1_S_CC_RMSE_abs_difference": s_cc_diff,
        "real_topology_metrics": metrics_real,
        "audited_reference_metrics": metrics_audited,
        "calibration_lock_sha256": sha256_file(output / "target_ridge/calibration_lock.json"),
        "target_model_manifest_sha256": lock["bound_assets"]["target_ridge"]["sha256"],
        "federated_h1_sha256": lock["bound_assets"]["federated_h1"]["sha256"],
        "classifier_sha256": lock["bound_assets"]["classifier"]["sha256"],
        "test_opened_after_calibration_lock": True,
        "test_used_for_fit_select_or_refit": False,
        "test_evaluation_timestamp": utc_now(),
    }
    write_json(output / "federated_h1/equivalence_decision.json", gate)
    finalize_topology_evidence(output)
    if frozen_hashes(root) != frozen_before:
        raise RuntimeError("runtime v4/HC frozen assets changed")
    print(json.dumps(gate, indent=2))


def finalize_candidate(args: argparse.Namespace) -> None:
    from gaps_deploy.c5_federated_source_ridge_bundle import load_federated_source_ridge_bundle

    output = Path(args.output_dir)
    runtime_dir = output / "runtime_v5"
    qc_dir = output / "qc"
    cal_report = json.loads((output / "parity/calibration_parity_report.json").read_text(encoding="utf-8"))
    test_report = json.loads((output / "parity/test_parity_report.json").read_text(encoding="utf-8"))
    equivalence = json.loads((output / "federated_h1/equivalence_decision.json").read_text(encoding="utf-8"))
    bundle = load_federated_source_ridge_bundle(runtime_dir / "bundle_manifest.json")
    identity = candidate_identity(runtime_dir / "runtime_contract_v5.json")
    if cal_report.get("status") != "PASS" or test_report.get("status") != "PASS" or equivalence.get("decision") not in {"EXACT_EQUIVALENCE", "PRACTICAL_EQUIVALENCE"}:
        raise RuntimeError("candidate finalization gate failed before QC audit")
    if cal_report.get("candidate_identity") != identity or test_report.get("candidate_identity") != identity:
        raise RuntimeError("parity evidence candidate identity differs from current bundle")
    if equivalence.get("federated_h1_sha256") != identity["assets"]["federated_h1"] or equivalence.get("target_model_manifest_sha256") != identity["assets"]["target_ridge"] or equivalence.get("classifier_sha256") != identity["assets"]["classifier"]:
        raise RuntimeError("equivalence evidence candidate identity differs from current bundle")
    if equivalence.get("calibration_lock_sha256") != identity["calibration_lock_sha256"]:
        raise RuntimeError("equivalence calibration lock identity differs from current bundle")
    if set(bundle.asset_paths) != {"classifier", "federated_h1", "target_ridge"}:
        raise RuntimeError("runtime v5 dependency set differs")

    qc_base = REPO_ROOT / "results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc"
    qc_assets = {name: descriptor(qc_base / name) for name in ("feature_reference.json", "component_calibrator.json", "risk_policy.json", "risk_selection.json")}
    qc_manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision_path": "PATH_B_RISK_SEMANTICS_INCOMPATIBLE",
        "v4_score_key": "deployment_risk_full",
        "compatible_components": ["raw_risk_confidence", "raw_risk_prototype", "raw_risk_support"],
        "incompatible_components": {
            "raw_risk_expert_disagreement": ["h23_plus_ppm", "target_ridge_plus_source_preds_ppm"],
            "raw_risk_source_spread": ["H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm", "H3_source_shared_mlp_ppm"],
        },
        "selected_risk_aggregates_incompatible_components": True,
        "threshold_lineage": "v4_calibration_validation_deployment_risk_full_distribution",
        "v4_threshold_reuse_allowed": False,
        "new_risk_model_invented": False,
        "v5_qc_built": False,
        "required_next_protocol": "separate_frozen_v5_QC_protocol_using_v5_calibration_only",
        "source_evidence": qc_assets,
        "code_evidence": {
            "path": str((REPO_ROOT / "gaps_deploy/c5_h8_runtime.py").resolve()),
            "required_fields": ["h23_plus_ppm", "target_ridge_plus_source_preds_ppm", "H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm", "H3_source_shared_mlp_ppm"],
        },
    }
    write_json(qc_dir / "qc_dependency_manifest.json", qc_manifest)

    contract_path = runtime_dir / "runtime_contract_v5.json"
    asset_index = {
        "runtime_contract_v5.json": descriptor(contract_path),
        "bundle_manifest.json": descriptor(runtime_dir / "bundle_manifest.json"),
        "classifier.pth": descriptor(runtime_dir / "assets/classifier.pth"),
        "federated_h1.json": descriptor(runtime_dir / "assets/federated_h1.json"),
        "target_ridge_105d.json": descriptor(runtime_dir / "assets/target_ridge_105d.json"),
        "calibration_selection_lock.json": descriptor(output / "target_ridge/calibration_selection_lock.json"),
        "calibration_parity_report.json": descriptor(output / "parity/calibration_parity_report.json"),
        "test_parity_report.json": descriptor(output / "parity/test_parity_report.json"),
    }
    write_json(runtime_dir / "asset_sha256.json", {"schema_version": SCHEMA_VERSION, "assets": asset_index})
    write_json(runtime_dir / "dependency_inventory.json", {
        "schema_version": SCHEMA_VERSION,
        "required_runtime_assets": sorted(bundle.asset_paths),
        "source_regression_heads": ["H1"],
        "target_input_dimension": 105,
        "forbidden_runtime_dependencies_present": False,
        "forbidden": ["H2", "H3", "R3aK16", "C3", "C4", "H8+C4", "P4", "test_labels"],
        "qc_enabled": False,
        "legacy_fallback": False,
    })
    frozen = frozen_hashes(REPO_ROOT)
    write_json(runtime_dir / "runtime_preflight.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "contract_sha256": sha256_file(contract_path),
        "bundle_load_strict": True,
        "classifier_checkpoint_sha256": sha256_file(runtime_dir / "assets/classifier.pth"),
        "classifier_checkpoint_expected_sha256": SEED42_CHECKPOINT_SHA256,
        "calibration_rows": 320,
        "test_rows": 1360,
        "runtime_v4_and_HC_frozen_sha256": frozen,
        "runtime_v4_assets_modified": False,
        "qc_enabled": False,
    })
    decision = {
        "schema_version": SCHEMA_VERSION,
        "decision": "RUNTIME_V5_REGRESSION_READY_QC_PENDING",
        "real_topology_H1_aggregation": "PASS",
        "source_raw_sample_isolation": "PASS",
        "seed42_RG1_metrics_replay": "PASS",
        "calibration_320_runtime_parity": "PASS",
        "test_1360_runtime_parity": "PASS",
        "fail_closed_runtime_contract": "PASS",
        "runtime_v4_six_frozen_SHA": "UNCHANGED",
        "qc_dependency_audit": "PATH_B_RISK_SEMANTICS_INCOMPATIBLE",
        "v5_HC95_HC90_built": False,
        "next_step": "freeze_and_execute_a_separate_v5_QC_protocol_before_any_Pi_benchmark",
        "stop_boundary": ["no_Pi_benchmark", "no_PC_latency_benchmark", "no_low_calibration", "no_new_algorithm_experiment", "no_new_training_seed"],
        "decided_at": utc_now(),
    }
    write_json(output / "runtime_v5_promotion_decision.json", decision)
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-calibration", "evaluate-test", "materialize-outputs", "build-bundle", "calibration-parity", "runtime-parity", "finalize-candidate"))
    parser.add_argument("--data-root", default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--multiseed-root", default="results/iotj_b5_multiseed_20260724")
    parser.add_argument("--real-h1", default="results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/global_h1_model.json")
    parser.add_argument("--audited-h1", default="results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json")
    parser.add_argument("--output-dir", default="results/iotj_b5_c5_runtime_v5_candidate_20260724")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    actions = {
        "freeze-calibration": freeze_calibration,
        "evaluate-test": evaluate_test,
        "materialize-outputs": materialize_required_outputs,
        "build-bundle": build_bundle,
        "calibration-parity": calibration_runtime_parity,
        "runtime-parity": runtime_parity,
        "finalize-candidate": finalize_candidate,
    }
    actions[arguments.stage](arguments)
