"""Run frozen A4 + R84 on the strict raw-file-disjoint robustness split."""
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

from scripts import run_iotj_canonical_v1_classification as canonical
from scripts import run_iotj_final_classification_le1 as frozen
from tools.preflight_iotj_canonical_v1_strict_nonoverlap import run_preflight


TARGETS = ("C3", "C4", "C5")
DATASET_NAME = "iotj_canonical_v1_strict_nonoverlap"
DATA_ROOT = ROOT / "dataset" / DATASET_NAME
PARENT_DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/strict_nonoverlap/run"


def strict_run_config() -> dict[str, Any]:
    return {
        "study": "STRICT_GROUPED_NON_OVERLAP",
        "dataset": DATASET_NAME,
        "dataset_aggregate_sha256": "881de29938460ad1a7564aca1f01a2b3f41cdc4820284397a05a0b3b218816c4",
        "parent_dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
        "classifier": "A4",
        "regression": "R84_FED_H1",
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_lr": 5e-4,
        "seed": 42,
        "target_counts": {"C3": {"calibration": 678, "test": 2515}, "C4": {"calibration": 320, "test": 840}, "C5": {"calibration": 320, "test": 840}},
        "target_test_selection": False,
        "hyperparameter_search": False,
    }


def build_strict_commands(target: str) -> dict[str, Any]:
    target = target.upper()
    if target not in TARGETS:
        raise ValueError(target)
    commands = canonical.build_canonical_commands(target)
    old_id = f"CANONICAL-V1-A4-{target}"
    new_id = f"CAN-V1-STRICT-A4-{target}"
    replacements = (
        (canonical.REMOTE_DATA_ROOT, REMOTE_DATA_ROOT),
        (canonical.PI_DATA_ROOT, PI_DATA_ROOT),
        (canonical.C2_DATA_ROOT, C2_DATA_ROOT),
        (old_id, new_id),
    )
    for role in ("server", "client_c1", "client_c2"):
        values = list(commands[role])
        for old, new in replacements:
            values = [value.replace(old, new) for value in values]
        commands[role] = values
    commands["protocol"].update({
        **strict_run_config(),
        "experiment_id": new_id,
        "target": target,
        "classifier_router": "A4",
        "strict_exact_window_overlap_count": 0,
        "strict_raw_file_overlap_count": 0,
        "strict_raw_time_overlap_seconds": 0.0,
        "checkpoint_reuse": False,
        "checkpoint_selection": "fixed_round_25",
    })
    return commands


def protocol_hash() -> str:
    digest = hashlib.sha256(json.dumps(strict_run_config(), sort_keys=True).encode())
    digest.update((DATA_ROOT / "strict_non_overlap_protocol.json").read_bytes())
    for target in TARGETS:
        digest.update(json.dumps(build_strict_commands(target), sort_keys=True).encode())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def write_or_validate_freeze(path: Path) -> dict[str, Any]:
    preflight = run_preflight(DATA_ROOT, PARENT_DATA_ROOT)
    payload = {
        "schema_version": "iotj.canonical_v1.strict_nonoverlap.run_freeze.v1",
        "status": "FROZEN",
        "freeze_commit": git_head(),
        "protocol_hash": protocol_hash(),
        "config": strict_run_config(),
        "preflight": preflight,
        "commands": {target: build_strict_commands(target) for target in TARGETS},
        "test_open_policy": "after all three fixed round25 A4 endpoints; R84 calibration lock before each target test",
    }
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError("FAIL_CLOSED strict run freeze differs")
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def execute_target(target: str, output: Path, lock: dict[str, Any], args: argparse.Namespace) -> None:
    experiment_id = f"CAN-V1-STRICT-A4-{target}"
    run_dir = output / "classification" / experiment_id
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial strict run exists: {run_dir}")
    original_root, original_builder = frozen.RESULT_ROOT, frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = output / "classification"
        frozen.build_flower_commands = lambda _experiment_id: build_strict_commands(target)
        frozen.execute_full_fl(
            experiment_id,
            protocol_hash=lock["protocol_hash"],
            lock_payload={"freeze_commit": lock["freeze_commit"], "study": "STRICT_GROUPED_NON_OVERLAP"},
            ecs_host=args.ecs_host, pi_host=args.pi_host, c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT, frozen.build_flower_commands = original_root, original_builder


def execute_r84(output: Path, device: str) -> None:
    from scripts import run_iotj_canonical_v1_r84 as r84

    if (output / "regression/protocol_manifest.json").is_file():
        return
    original = (
        r84.DATA_ROOT, r84.STUDY_ID, r84.SCHEMA_VERSION,
        r84.CLASSIFICATION_EXPERIMENT_PREFIX, r84.REGRESSION_EXPERIMENT_PREFIX,
        r84.SPLIT_PROTOCOL,
    )
    try:
        r84.DATA_ROOT = DATA_ROOT
        r84.STUDY_ID = "iotj_canonical_v1_strict_nonoverlap_20260809"
        r84.SCHEMA_VERSION = "iotj.canonical_v1.strict_nonoverlap.r84.v1"
        r84.CLASSIFICATION_EXPERIMENT_PREFIX = "CAN-V1-STRICT-A4"
        r84.REGRESSION_EXPERIMENT_PREFIX = "CAN-V1-STRICT-R84-A4"
        r84.SPLIT_PROTOCOL = "strict_grouped_raw_file_nonoverlap"
        r84.build(output, device, 32)
    finally:
        (
            r84.DATA_ROOT, r84.STUDY_ID, r84.SCHEMA_VERSION,
            r84.CLASSIFICATION_EXPERIMENT_PREFIX, r84.REGRESSION_EXPERIMENT_PREFIX,
            r84.SPLIT_PROTOCOL,
        ) = original


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = write_or_validate_freeze(output / "STRICT_RUN_PRE_RUN_FREEZE.json")
    for target in TARGETS:
        execute_target(target, output, lock, args)
    execute_r84(output, args.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
