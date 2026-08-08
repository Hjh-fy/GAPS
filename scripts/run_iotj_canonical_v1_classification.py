"""Run the frozen GAPS classifier from scratch on canonical-v1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_iotj_final_classification_le1 as frozen


STUDY_ID = "iotj_canonical_v1_final_20260808"
TARGETS = ("C3", "C4", "C5")
DATASET_NAME = "iotj_canonical_v1"
LOCAL_DATA_ROOT = ROOT / "dataset" / DATASET_NAME
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"
DEFAULT_OUTPUT = ROOT / "results" / STUDY_ID / "classification"
DOC_ROOT = ROOT / "docs" / "experiments" / "iotj_canonical_v1"
_FROZEN_BUILDER = frozen.build_flower_commands


def canonical_classification_config() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "dataset": DATASET_NAME,
        "rounds": 25,
        "local_epochs": 5,
        "batch_size": 32,
        "seed": 42,
        "optimizer": "Adam",
        "optimizer_lr": 5e-4,
        "checkpoint_reuse": False,
        "checkpoint_selection": "fixed_round_25",
        "hyperparameter_search": False,
        "adaptation_target_split": "calibration",
        "da_window_shape": [50, 8],
        "target_test_selection": False,
    }


def _replace(values: list[str], old: str, new: str) -> list[str]:
    return [value.replace(old, new) for value in values]


def _set_option(values: list[str], option: str, value: str) -> None:
    index = values.index(option)
    values[index + 1] = value


def build_canonical_commands(target: str) -> dict[str, Any]:
    """Clone only the frozen algorithm command and change run/data protocol."""
    target = target.upper()
    if target not in TARGETS:
        raise ValueError(f"unknown canonical target: {target}")
    legacy_id = f"FCL-E3-GAPS-{target}"
    experiment_id = f"CANONICAL-V1-GAPS-{target}"
    commands = _FROZEN_BUILDER(legacy_id)
    replacements = (
        (frozen.REMOTE_DATA_ROOT, REMOTE_DATA_ROOT),
        (frozen.PI_DATA_ROOT, PI_DATA_ROOT),
        (frozen.C2_DATA_ROOT, C2_DATA_ROOT),
        (legacy_id, experiment_id),
    )
    for role in ("server", "client_c1", "client_c2"):
        values = list(commands[role])
        for old, new in replacements:
            values = _replace(values, old, new)
        commands[role] = values
    _set_option(commands["client_c1"], "--local-epochs", "5")
    _set_option(commands["client_c2"], "--local-epochs", "5")
    commands["server"].extend(["--da-window-length", "50"])
    commands["protocol"].update(canonical_classification_config())
    commands["protocol"].update(
        {
            "experiment_id": experiment_id,
            "target": target,
            "initialization": "fresh_seed42_random_initialization",
            "source_clients": ["C1", "C2"],
            "adaptation_target_fields": ["x", "class", "phase"],
        }
    )
    return commands


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(DOC_ROOT.glob("*")):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    digest.update(json.dumps(canonical_classification_config(), sort_keys=True).encode())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def validate_preflight() -> dict[str, Any]:
    preflight_path = ROOT / "results" / "iotj_canonical_v1" / "preflight.json"
    dataset_hash_path = LOCAL_DATA_ROOT / "dataset_sha256.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    dataset_hash = json.loads(dataset_hash_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("FAIL_CLOSED canonical dataset preflight is not PASS")
    if preflight.get("aggregate_sha256") != dataset_hash.get("aggregate_sha256"):
        raise RuntimeError("FAIL_CLOSED canonical dataset hash differs from preflight")
    expected = {
        "C1": {"train": 2360, "calibration": 320, "test": 680},
        "C2": {"train": 2360, "calibration": 320, "test": 680},
        "C3": {"calibration": 678, "test": 2677},
        "C4": {"calibration": 320, "test": 1360},
        "C5": {"calibration": 320, "test": 1360},
    }
    if preflight.get("counts") != expected:
        raise RuntimeError("FAIL_CLOSED canonical split counts differ")
    return {
        "preflight_path": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "dataset_aggregate_sha256": dataset_hash["aggregate_sha256"],
        "counts": expected,
    }


def audit_protocol() -> dict[str, Any]:
    """Audit the complete command surface before any remote process starts."""
    findings: list[dict[str, Any]] = []
    checkpoint_tokens = False
    test_references = False
    for target in TARGETS:
        commands = build_canonical_commands(target)
        flat = [
            value
            for role in ("server", "client_c1", "client_c2")
            for value in commands[role]
        ]
        joined = " ".join(flat).lower()
        checkpoint_tokens = checkpoint_tokens or any(
            token in flat for token in ("--checkpoint", "--resume", "--resume-from")
        )
        test_references = test_references or any(
            marker in joined for marker in ("test_features", "test_labels", "test_classification")
        )
        server = commands["server"]
        server_calib = server[server.index("--server-calib-data") + 1]
        server_val = server[server.index("--server-val-data") + 1]
        target_ok = server_calib == f"{REMOTE_DATA_ROOT}/client_{target[1:]}"
        source_ok = server_val == f"{REMOTE_DATA_ROOT}/client_1,{REMOTE_DATA_ROOT}/client_2"
        warmup_ok = server[server.index("--selective-warmup") + 1] == "5"
        strict_cal_ok = server[server.index("--strict-calibration-split") + 1] == "true"
        target_ce_zero = server[server.index("--da-lambda-target-ce") + 1] == "0.0"
        da_window_shape_ok = server[server.index("--da-window-length") + 1] == "50"
        findings.append(
            {
                "target": target,
                "canonical_target_calibration_root": target_ok,
                "canonical_source_validation_roots": source_ok,
                "selective_rounds_1_to_5_warmup": warmup_ok,
                "strict_calibration_split": strict_cal_ok,
                "target_ce_weight_zero": target_ce_zero,
                "da_window_shape_50x8": da_window_shape_ok,
                "local_epochs": commands["protocol"]["local_epochs"],
                "checkpoint_reuse": commands["protocol"]["checkpoint_reuse"],
            }
        )
    all_targets_fixed = all(
        row["canonical_target_calibration_root"]
        and row["canonical_source_validation_roots"]
        and row["selective_rounds_1_to_5_warmup"]
        and row["strict_calibration_split"]
        and row["target_ce_weight_zero"]
        and row["da_window_shape_50x8"]
        and row["local_epochs"] == 5
        and row["checkpoint_reuse"] is False
        for row in findings
    )
    status = "PASS" if all_targets_fixed and not checkpoint_tokens and not test_references else "FAIL"
    return {
        "schema_version": "iotj.canonical_v1.classification.protocol_audit.v1",
        "status": status,
        "all_targets_fixed": all_targets_fixed,
        "checkpoint_reuse_tokens_present": checkpoint_tokens,
        "test_arrays_referenced_by_training_commands": test_references,
        "findings": findings,
    }


def execute_target(
    target: str,
    output: Path,
    freeze_commit: str,
    digest: str,
    args: argparse.Namespace,
) -> None:
    experiment_id = f"CANONICAL-V1-GAPS-{target}"
    run_dir = output / experiment_id
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial canonical run exists: {run_dir}")
    original_root = frozen.RESULT_ROOT
    original_builder = frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = output
        frozen.build_flower_commands = lambda _experiment_id: build_canonical_commands(target)
        frozen.execute_full_fl(
            experiment_id,
            protocol_hash=digest,
            lock_payload={"freeze_commit": freeze_commit},
            ecs_host=args.ecs_host,
            pi_host=args.pi_host,
            c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT = original_root
        frozen.build_flower_commands = original_builder


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = validate_preflight()
    protocol_audit = audit_protocol()
    if protocol_audit["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED canonical classification protocol audit failed")
    digest = protocol_hash()
    freeze_commit = git_head()
    lock = {
        "schema_version": "iotj.canonical_v1.classification.pre_run.v1",
        "status": "FROZEN",
        "freeze_commit": freeze_commit,
        "protocol_hash": digest,
        "classification": canonical_classification_config(),
        "data": preflight,
        "protocol_audit": protocol_audit,
        "targets": list(TARGETS),
        "test_open_policy": "only after all fixed round25 endpoints complete",
    }
    lock_path = output.parent / "CLASSIFICATION_PRE_RUN_FREEZE.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise RuntimeError("FAIL_CLOSED canonical classification freeze differs")
    else:
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for index, target in enumerate(TARGETS):
        progress = {
            "status": "RUNNING",
            "current_target": target,
            "completed_targets": list(TARGETS[:index]),
            "target_test_opened": False,
        }
        (output.parent / "CLASSIFICATION_PROGRESS.json").write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8"
        )
        execute_target(target, output, freeze_commit, digest, args)
    (output.parent / "CLASSIFICATION_PROGRESS.json").write_text(
        json.dumps(
            {
                "status": "FIXED_ENDPOINTS_COMPLETE_TEST_STILL_SEALED",
                "current_target": None,
                "completed_targets": list(TARGETS),
                "target_test_opened": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
