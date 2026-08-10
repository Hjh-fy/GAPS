"""Run the frozen canonical-v1 A0T versus GAPS/A4 R84 comparison."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from gaps_flower.state_fingerprint import checkpoint_provenance
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


ROOT = Path(__file__).resolve().parents[1]
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
