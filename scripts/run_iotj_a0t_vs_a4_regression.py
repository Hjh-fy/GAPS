"""Run the frozen canonical-v1 A0T versus GAPS/A4 R84 comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.state_fingerprint import checkpoint_provenance
import numpy as np
import torch
from run_regression_head_ablation import CLASS_NAMES, CLASS_RANGES, fit_ridge
from scripts import run_gaps_cross_target_r84_full as r84_common
from scripts.run_iotj_canonical_v1_r84 import expected_counts, prepare_rows, route_rows
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset

DATA_ROOT = ROOT / "dataset" / "iotj_canonical_v1"
H1_MANIFEST = (
    ROOT
    / "results"
    / "iotj_h1_federated_ridge_equivalence_20260724"
    / "federated_h1_manifest.json"
)
EXPECTED_DATASET_SHA256 = "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"
EXPECTED_H1_SHA256 = "d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc"
SPLIT_PROTOCOL = "canonical_v1_target_20_80"
REGRESSION_PROFILE = "R84_FED_H1_fixed_alpha"
SEED = 42
DEFAULT_OUTPUT = ROOT / "results" / "iotj_canonical_v1_final" / "a0t_vs_a4_regression"
CANONICAL_A4_REGRESSION = ROOT / "results" / "iotj_canonical_v1_final_20260808" / "regression"
CANONICAL_QC_ROOT = (
    ROOT
    / "results"
    / "iotj_canonical_v1_final_20260808"
    / "evidence_closure"
    / "qc"
)

FROZEN_ALPHAS = {
    "C3": {0: 100.0, 1: 0.0, 2: 0.1, 3: 0.1},
    "C4": {0: 1.0, 1: 10.0, 2: 0.1, 3: 10.0},
    "C5": {0: 1.0, 1: 0.01, 2: 10.0, 3: 0.1},
}

CHECKPOINT_SHA256 = {
    ("A0T", "C3"): "4894be9a943876dc46e219ffcb68d1d7ce0fdb3981ae9255b0aba2ce4e6b5728",
    ("A0T", "C4"): "eee28075336170682abc4fb7e17fd01f481776ea06d175c2cf0decada85ec609",
    ("A0T", "C5"): "b46d1f5fe9df53b425d207df965af2656ca4290e1fe0cb6f723cdd8f0e007fa5",
    ("A4", "C3"): "e2364290ffc7fd9748fe86edb3745dca0eac692165f6c8aba1825728ddcd4414",
    ("A4", "C4"): "422a49f28331e5486d215a8d34bc9a972dc8fc1992f8b5bf27428329143599c3",
    ("A4", "C5"): "3965ec8618a2d496804bbc141f49e00b451fce05e9edbefde721f0dd4f912b93",
}


@dataclass(frozen=True)
class EndpointSpec:
    experiment_id: str
    method: str
    target: str
    checkpoint: Path
    checkpoint_sha256: str
    classification_manifest: Path
    completion_marker: Path
    dataset_root: Path = DATA_ROOT
    split_protocol: str = SPLIT_PROTOCOL
    calibration: str = "canonical_target_calibration_20pct"
    regression_profile: str = REGRESSION_PROFILE
    h1_manifest: Path = H1_MANIFEST
    h1_sha256: str = EXPECTED_H1_SHA256
    seed: int = SEED

    @property
    def held_constants(self) -> tuple[Any, ...]:
        return (
            self.target,
            self.dataset_root,
            self.split_protocol,
            self.calibration,
            self.regression_profile,
            self.h1_manifest,
            self.h1_sha256,
            self.seed,
        )


def _run_root(method: str, target: str) -> Path:
    base = ROOT / "results" / "iotj_canonical_v1_final_20260808"
    if method == "A0T":
        return base / "a0t_equal_label" / "classification" / f"CANONICAL-V1-A0T-{target}"
    return base / "classification" / f"CANONICAL-V1-A4-{target}"


def endpoint_specs() -> tuple[EndpointSpec, ...]:
    specs: list[EndpointSpec] = []
    for method in ("A0T", "A4"):
        for target in ("C3", "C4", "C5"):
            run = _run_root(method, target)
            classifier_id = f"CANONICAL-V1-{method}-{target}"
            specs.append(
                EndpointSpec(
                    experiment_id=f"CAN-V1-REG-{method}-{target}-S42",
                    method=method,
                    target=target,
                    checkpoint=run / "remote_server" / "server_latest_adapted.pth",
                    checkpoint_sha256=CHECKPOINT_SHA256[(method, target)],
                    classification_manifest=run / "run_manifest.json",
                    completion_marker=run / "fixed_endpoint_complete.json",
                )
            )
    return tuple(specs)


def frozen_alphas() -> dict[str, dict[int, float]]:
    return {target: dict(values) for target, values in FROZEN_ALPHAS.items()}


def audit_endpoint_pair(a0t: EndpointSpec, a4: EndpointSpec) -> dict[str, Any]:
    if a0t.target != a4.target or a0t.method != "A0T" or a4.method != "A4":
        raise RuntimeError("held-constant drift: endpoint pair identity differs")
    allowed = {
        "checkpoint",
        "checkpoint_sha256",
        "classification_manifest",
        "completion_marker",
        "experiment_id",
        "method",
    }
    left = asdict(a0t)
    right = asdict(a4)
    drift = sorted(key for key in left if left[key] != right[key])
    unexpected = sorted(set(drift) - allowed)
    if unexpected:
        raise RuntimeError(f"held-constant drift: {unexpected}")
    return {"status": "PASS", "target": a0t.target, "varying_fields": drift}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(spec: EndpointSpec) -> dict[str, Any]:
    if not spec.checkpoint.is_file():
        raise RuntimeError(f"checkpoint missing: {spec.experiment_id}")
    if sha256(spec.checkpoint) != spec.checkpoint_sha256:
        raise RuntimeError(f"checkpoint SHA256 differs: {spec.experiment_id}")
    manifest = json.loads(spec.classification_manifest.read_text(encoding="utf-8"))
    marker = json.loads(spec.completion_marker.read_text(encoding="utf-8"))
    classifier_id = f"CANONICAL-V1-{spec.method}-{spec.target}"
    if manifest.get("experiment_id") != classifier_id or marker.get("experiment_id") != classifier_id:
        raise RuntimeError(f"classification identity differs: {spec.experiment_id}")
    if manifest.get("target_test_opened") is not False or marker.get("target_test_opened") is not False:
        raise RuntimeError(f"target test was opened before endpoint lock: {spec.experiment_id}")
    if manifest.get("checkpoint_sha256") != spec.checkpoint_sha256:
        raise RuntimeError(f"manifest checkpoint SHA256 differs: {spec.experiment_id}")
    provenance = checkpoint_provenance(spec.checkpoint)
    if int(provenance.get("formal_round", -1)) != 25:
        raise RuntimeError(f"checkpoint is not formal round25: {spec.experiment_id}")
    if provenance.get("whole_file_sha256") != spec.checkpoint_sha256:
        raise RuntimeError(f"checkpoint provenance SHA256 differs: {spec.experiment_id}")
    return {
        "experiment_id": spec.experiment_id,
        "classification_experiment_id": classifier_id,
        "method": spec.method,
        "target": spec.target,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_sha256": spec.checkpoint_sha256,
        "ordered_state_content_fingerprint": provenance["ordered_state_content_fingerprint"],
        "formal_round": 25,
        "classification_manifest_sha256": sha256(spec.classification_manifest),
        "completion_marker_sha256": sha256(spec.completion_marker),
    }


def _audit_frozen_alphas() -> dict[str, dict[int, float]]:
    observed: dict[str, dict[int, float]] = {}
    for target in ("C3", "C4", "C5"):
        path = CANONICAL_A4_REGRESSION / target / "calibration_alpha_selection.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = {int(row["class_id"]): float(row["selected_alpha"]) for row in rows}
        if values != FROZEN_ALPHAS[target]:
            raise RuntimeError(f"frozen alpha provenance differs: {target}")
        observed[target] = values
    return observed


def _write_registry(path: Path, checkpoints: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id", "source_clients", "target_clients", "split_protocol",
        "model", "checkpoint", "DA", "calibration", "QC", "seed",
        "result_path", "metrics", "status", "notes", "code_commit",
        "config_path", "dataset_path", "created_at", "evidence_status", "provenance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in checkpoints:
            writer.writerow({
                "experiment_id": item["experiment_id"],
                "source_clients": "C1;C2",
                "target_clients": item["target"],
                "split_protocol": SPLIT_PROTOCOL,
                "model": REGRESSION_PROFILE,
                "checkpoint": item["checkpoint"],
                "DA": item["method"],
                "calibration": "canonical_target_calibration_20pct",
                "QC": "frozen_equal_mean_HC90_HC95",
                "seed": SEED,
                "result_path": f"endpoints/{item['experiment_id']}",
                "metrics": "pending_fixed_endpoint_evaluation",
                "status": "registered",
                "notes": "classifier checkpoint is the only upstream method factor",
                "code_commit": "pre_run_freeze",
                "config_path": "PRE_RUN_FREEZE.json",
                "dataset_path": str(DATA_ROOT),
                "created_at": "pre_run",
                "evidence_status": "draft",
                "provenance": item["checkpoint_sha256"],
            })


def audit_inputs(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED output already exists: {output}")
    dataset = verify_dataset(DATA_ROOT)
    if dataset.get("status") != "PASS" or dataset.get("aggregate_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("canonical dataset hash differs")
    if sha256(H1_MANIFEST) != EXPECTED_H1_SHA256:
        raise RuntimeError("Federated-H1 SHA256 differs")
    specs = endpoint_specs()
    for target in ("C3", "C4", "C5"):
        pair = [spec for spec in specs if spec.target == target]
        audit_endpoint_pair(pair[0], pair[1])
    checkpoints = [audit_checkpoint(spec) for spec in specs]
    alphas = _audit_frozen_alphas()
    qc = {}
    for target in ("C3", "C4", "C5"):
        lock = CANONICAL_QC_ROOT / f"{target}_qc_threshold_lock.csv"
        if not lock.is_file():
            raise RuntimeError(f"frozen QC lock missing: {target}")
        qc[target] = {"path": str(lock), "sha256": sha256(lock)}
    output.mkdir(parents=True)
    result = {
        "schema_version": "iotj.canonical_v1.a0t_vs_a4_regression.freeze.v1",
        "status": "PASS",
        "endpoint_count": 6,
        "target_test_state": "SEALED",
        "alpha_selection_performed": False,
        "classifier_training_performed": False,
        "dataset": dataset,
        "h1": {"path": str(H1_MANIFEST), "sha256": EXPECTED_H1_SHA256},
        "frozen_alphas": alphas,
        "qc_locks": qc,
        "checkpoints": checkpoints,
    }
    (output / "PRE_RUN_FREEZE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_registry(output / "experiment_registry.csv", checkpoints)
    return result


def fit_fixed_alpha_models(
    target: str, oracle_rows: Sequence[Mapping[str, Any]]
) -> dict[int, Any]:
    if target not in FROZEN_ALPHAS:
        raise ValueError(f"unsupported target: {target}")
    if not oracle_rows:
        raise RuntimeError("FAIL_CLOSED empty calibration rows")
    models: dict[int, Any] = {}
    for class_id in range(4):
        rows = [dict(row) for row in oracle_rows if int(row["true_class"]) == class_id]
        if not rows:
            raise RuntimeError(f"FAIL_CLOSED empty calibration class: {target}/{class_id}")
        feature_names = sorted(str(name) for name in rows[0]["feature_dict"])
        if any(sorted(str(name) for name in row["feature_dict"]) != feature_names for row in rows):
            raise RuntimeError(f"FAIL_CLOSED feature ordering differs: {target}/{class_id}")
        models[class_id] = fit_ridge(rows, feature_names, FROZEN_ALPHAS[target][class_id])
    return models


def apply_scope_models(
    rows: Sequence[Mapping[str, Any]], models: Mapping[int, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        route = int(item["pred_class"])
        prediction = float(models[route].predict([item])[0])
        truth = float(item["true_ppm"])
        item.update(
            {
                "route_correct": int(route == int(item["true_class"])),
                "pred_84d_h1_ppm": prediction,
                "pred_ppm": prediction,
                "abs_error": abs(prediction - truth),
                "squared_error": (prediction - truth) ** 2,
            }
        )
        output.append(item)
    return output


def build_four_scopes(
    deployment_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    models: Mapping[int, Any],
) -> dict[str, list[dict[str, Any]]]:
    if len(deployment_rows) != len(oracle_rows):
        raise RuntimeError("FAIL_CLOSED deployment/oracle row count differs")
    deployment_ids = [int(row["sample_index"]) for row in deployment_rows]
    oracle_ids = [int(row["sample_index"]) for row in oracle_rows]
    if deployment_ids != oracle_ids:
        raise RuntimeError("FAIL_CLOSED deployment/oracle row order differs")
    s_all = apply_scope_models(deployment_rows, models)
    s_cc = [row for row in s_all if int(row["route_correct"]) == 1]
    forced_oracle = [
        {**dict(row), "pred_class": int(row["true_class"])} for row in oracle_rows
    ]
    oracle_all = apply_scope_models(forced_oracle, models)
    correct_ids = {int(row["sample_index"]) for row in s_cc}
    oracle_cc = [row for row in oracle_all if int(row["sample_index"]) in correct_ids]
    if [int(row["sample_index"]) for row in oracle_cc] != [int(row["sample_index"]) for row in s_cc]:
        raise RuntimeError("FAIL_CLOSED Oracle_CC/S_CC indices differ")
    return {"S_ALL": s_all, "S_CC": s_cc, "Oracle_ALL": oracle_all, "Oracle_CC": oracle_cc}


def summarize_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("FAIL_CLOSED empty regression scope")
    truth = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row["pred_ppm"]) for row in rows], dtype=np.float64)
    classes = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise RuntimeError("FAIL_CLOSED non-finite regression scope")
    error = pred - truth
    ranges = np.asarray([float(CLASS_RANGES[int(value)]) for value in classes])
    centered = truth - float(np.mean(truth))
    total = float(np.sum(centered**2))
    return {
        "N": int(len(rows)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "NRMSE_range": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "R2": float(1.0 - np.sum(error**2) / total) if total > 0 else float("nan"),
        "Bias": float(np.mean(error)),
    }


def orchestrate_sealed_run(
    specs: Sequence[EndpointSpec],
    output: Path,
    fit_stage: Callable[[EndpointSpec, Path], Path],
    test_stage: Callable[[EndpointSpec, Path], Any],
) -> None:
    endpoint_root = output / "endpoints"
    locks: list[Path] = []
    for spec in specs:
        locks.append(fit_stage(spec, endpoint_root / spec.experiment_id))
    if len(locks) != 6 or any(not lock.is_file() for lock in locks):
        raise RuntimeError("FAIL_CLOSED all six calibration locks must exist before target test")
    for lock in locks:
        payload = json.loads(lock.read_text(encoding="utf-8"))
        if payload.get("status") != "SEALED_BEFORE_TARGET_TEST":
            raise RuntimeError(f"FAIL_CLOSED invalid calibration lock: {lock}")
    for spec in specs:
        test_stage(spec, endpoint_root / spec.experiment_id)


def special_slice_rows(
    target: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if target != "C5":
        return []
    selected = [
        dict(row)
        for row in rows
        if str(row.get("gas", "")).lower() == "methane"
        and abs(float(row["true_ppm"]) - 225.0) <= 1e-9
        and int(row.get("repeat_id", -1)) == 1
    ]
    if not selected:
        raise RuntimeError("FAIL_CLOSED C5 Methane 225 ppm repeat1 slice missing")
    return selected


def _without_features(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "feature_dict"} for row in rows]


def _model_manifest(models: Mapping[int, Any]) -> dict[str, Any]:
    return {str(class_id): model.to_json() for class_id, model in sorted(models.items())}


def _scope_summary_rows(spec: EndpointSpec, scopes: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "method": spec.method,
            "target": spec.target,
            "scope": scope,
            "seed": SEED,
            **summarize_scope(rows),
        }
        for scope, rows in scopes.items()
    ]


def _per_gas_rows(spec: EndpointSpec, scopes: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, rows in scopes.items():
        for class_id, gas in CLASS_NAMES.items():
            selected = [row for row in rows if int(row["true_class"]) == class_id]
            output.append(
                {
                    "experiment_id": spec.experiment_id,
                    "method": spec.method,
                    "target": spec.target,
                    "scope": scope,
                    "class_id": class_id,
                    "gas": gas,
                    "seed": SEED,
                    **summarize_scope(selected),
                }
            )
    return output


def _per_concentration_rows(
    spec: EndpointSpec, scopes: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, rows in scopes.items():
        groups = sorted({(int(row["true_class"]), float(row["true_ppm"])) for row in rows})
        for class_id, concentration in groups:
            selected = [
                row
                for row in rows
                if int(row["true_class"]) == class_id
                and abs(float(row["true_ppm"]) - concentration) <= 1e-9
            ]
            output.append(
                {
                    "experiment_id": spec.experiment_id,
                    "method": spec.method,
                    "target": spec.target,
                    "scope": scope,
                    "class_id": class_id,
                    "gas": CLASS_NAMES[class_id],
                    "true_ppm": concentration,
                    "seed": SEED,
                    **summarize_scope(selected),
                }
            )
    return output


def _test_manifest_hashes(target: str) -> dict[str, str]:
    directory = DATA_ROOT / f"client_{target[1:]}"
    names = (
        "test_features.npy",
        "test_classification_labels.npy",
        "test_regression_labels.npy",
        "test_phase_labels.npy",
        "test_experiment_info.json",
    )
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED target test manifest inputs missing: {target}/{missing}")
    return {name: sha256(directory / name) for name in names}


def _fit_endpoint_calibration(
    spec: EndpointSpec,
    endpoint_dir: Path,
    h1: Mapping[int, Any],
    device: torch.device,
    batch_size: int,
    model_cache: dict[str, dict[int, Any]],
) -> Path:
    endpoint_dir.mkdir(parents=True)
    provenance = audit_checkpoint(spec)
    routes, classification = route_rows(
        spec.checkpoint, spec.target, "calibration", device, batch_size
    )
    counts = expected_counts(spec.target)
    if len(routes) != int(counts["calibration"]):
        raise RuntimeError(f"FAIL_CLOSED calibration count differs: {spec.experiment_id}")
    oracle, deployment = prepare_rows(spec.target, "calibration", routes, h1)
    oracle_r84 = [r84_common.r84_row(row) for row in oracle]
    deployment_r84 = [r84_common.r84_row(row) for row in deployment]
    models = fit_fixed_alpha_models(spec.target, oracle_r84)
    model_cache[spec.experiment_id] = models
    model_path = endpoint_dir / "r84_models.json"
    r84_common.write_json(model_path, _model_manifest(models))
    calibration_scopes = build_four_scopes(deployment_r84, oracle_r84, models)
    r84_common.write_csv(
        endpoint_dir / "calibration_s_all.csv", _without_features(calibration_scopes["S_ALL"])
    )
    lock_path = endpoint_dir / "calibration_lock.json"
    r84_common.write_json(
        lock_path,
        {
            "schema_version": "iotj.canonical_v1.a0t_vs_a4_regression.lock.v1",
            "status": "SEALED_BEFORE_TARGET_TEST",
            "experiment_id": spec.experiment_id,
            "method": spec.method,
            "target": spec.target,
            "target_test_opened": False,
            "alpha_selection_performed": False,
            "fixed_alphas": FROZEN_ALPHAS[spec.target],
            "calibration_N": len(routes),
            "classification_metrics": classification,
            "checkpoint": provenance,
            "r84_models_sha256": sha256(model_path),
        },
    )
    return lock_path


def _evaluate_endpoint_test(
    spec: EndpointSpec,
    endpoint_dir: Path,
    h1: Mapping[int, Any],
    device: torch.device,
    batch_size: int,
    model_cache: Mapping[str, Mapping[int, Any]],
    expected_classifier_endpoint: tuple[str, int] = ("round", 25),
) -> dict[str, Any]:
    lock_path = endpoint_dir / "calibration_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "SEALED_BEFORE_TARGET_TEST" or lock.get("target_test_opened") is not False:
        raise RuntimeError(f"FAIL_CLOSED invalid calibration lock: {spec.experiment_id}")
    routes, classification = route_rows(
        spec.checkpoint,
        spec.target,
        "test",
        device,
        batch_size,
        expected_endpoint=expected_classifier_endpoint,
    )
    counts = expected_counts(spec.target)
    if len(routes) != int(counts["test"]):
        raise RuntimeError(f"FAIL_CLOSED test count differs: {spec.experiment_id}")
    oracle, deployment = prepare_rows(spec.target, "test", routes, h1)
    oracle_r84 = [r84_common.r84_row(row) for row in oracle]
    deployment_r84 = [r84_common.r84_row(row) for row in deployment]
    scopes = build_four_scopes(deployment_r84, oracle_r84, model_cache[spec.experiment_id])
    prediction_hashes: dict[str, str] = {}
    for scope, rows in scopes.items():
        path = endpoint_dir / f"test_{scope.lower()}.csv"
        r84_common.write_csv(path, _without_features(rows))
        prediction_hashes[scope] = sha256(path)
    summary = _scope_summary_rows(spec, scopes)
    per_gas = _per_gas_rows(spec, scopes)
    per_concentration = _per_concentration_rows(spec, scopes)
    r84_common.write_csv(endpoint_dir / "scope_summary.csv", summary)
    r84_common.write_csv(endpoint_dir / "per_gas.csv", per_gas)
    r84_common.write_csv(endpoint_dir / "per_concentration.csv", per_concentration)
    special_rows = special_slice_rows(spec.target, scopes["S_ALL"])
    if special_rows:
        special = {
            "experiment_id": spec.experiment_id,
            "method": spec.method,
            "target": spec.target,
            "slice": "methane_225ppm_repeat1",
            **summarize_scope(special_rows),
        }
        r84_common.write_csv(endpoint_dir / "special_slices.csv", [special])
    manifest = {
        "schema_version": "iotj.canonical_v1.a0t_vs_a4_regression.endpoint.v1",
        "status": "COMPLETE",
        "experiment_id": spec.experiment_id,
        "method": spec.method,
        "target": spec.target,
        "seed": SEED,
        "calibration_lock_sha256": sha256(lock_path),
        "checkpoint_sha256": spec.checkpoint_sha256,
        "prediction_sha256": prediction_hashes,
        "test_manifest_sha256": _test_manifest_hashes(spec.target),
        "test_classification": classification,
        "target_test_used_for_selection": False,
        "alpha_selection_performed": False,
    }
    r84_common.write_json(endpoint_dir / "endpoint_manifest.json", manifest)
    return {"summary": summary, "per_gas": per_gas, "per_concentration": per_concentration, "manifest": manifest}


def execute_study(output: Path, device_text: str, batch_size: int) -> dict[str, Any]:
    output = output.resolve()
    freeze_path = output / "PRE_RUN_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("FAIL_CLOSED PRE_RUN_FREEZE.json is required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS" or freeze.get("target_test_state") != "SEALED":
        raise RuntimeError("FAIL_CLOSED PRE_RUN_FREEZE is invalid")
    if (output / "endpoints").exists():
        raise FileExistsError("FAIL_CLOSED regression endpoints already exist")
    dataset_before = verify_dataset(DATA_ROOT)
    if dataset_before.get("aggregate_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("FAIL_CLOSED canonical dataset changed after freeze")
    h1 = r84_common.load_h1()
    specs = endpoint_specs()
    model_cache: dict[str, dict[int, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    device = torch.device(device_text)

    def fit_stage(spec: EndpointSpec, endpoint_dir: Path) -> Path:
        return _fit_endpoint_calibration(spec, endpoint_dir, h1, device, batch_size, model_cache)

    def test_stage(spec: EndpointSpec, endpoint_dir: Path) -> None:
        results[spec.experiment_id] = _evaluate_endpoint_test(
            spec, endpoint_dir, h1, device, batch_size, model_cache
        )

    orchestrate_sealed_run(specs, output, fit_stage, test_stage)
    dataset_after = verify_dataset(DATA_ROOT)
    if dataset_after != dataset_before:
        raise RuntimeError("FAIL_CLOSED canonical dataset changed during evaluation")
    all_summary = [row for spec in specs for row in results[spec.experiment_id]["summary"]]
    all_per_gas = [row for spec in specs for row in results[spec.experiment_id]["per_gas"]]
    all_per_concentration = [
        row for spec in specs for row in results[spec.experiment_id]["per_concentration"]
    ]
    r84_common.write_csv(output / "scope_metrics_raw.csv", all_summary)
    r84_common.write_csv(output / "per_gas_metrics_raw.csv", all_per_gas)
    r84_common.write_csv(output / "per_concentration_metrics_raw.csv", all_per_concentration)
    manifest = {
        "schema_version": "iotj.canonical_v1.a0t_vs_a4_regression.protocol.v1",
        "status": "FIXED_ENDPOINTS_COMPLETE",
        "endpoint_count": 6,
        "seed": SEED,
        "dataset": dataset_after,
        "target_test_opened_after_all_calibration_locks": True,
        "target_test_used_for_selection": False,
        "classifier_training_performed": False,
        "alpha_selection_performed": False,
        "frozen_alphas": FROZEN_ALPHAS,
        "endpoint_manifests": {
            spec.experiment_id: results[spec.experiment_id]["manifest"] for spec in specs
        },
    }
    r84_common.write_json(output / "protocol_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = (
        audit_inputs(args.output)
        if args.audit_only
        else execute_study(args.output, args.device, args.batch_size)
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
