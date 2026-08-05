"""Run the three corrected target-specific GAPS classifiers on role-aware splits."""

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


STUDY_ID = "iotj_gaps_roleaware_r84_full_20260805"
DOC_ROOT = ROOT / "docs/experiments" / STUDY_ID
DEFAULT_OUTPUT = ROOT / "results" / STUDY_ID / "classification"
DATASET_NAME = "client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"
LOCAL_DATA_ROOT = ROOT.parents[1] / "dataset" / DATASET_NAME
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"
TARGETS = ("C3", "C4", "C5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def protocol_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        DOC_ROOT / "EXPERIMENT_PLAN.md",
        DOC_ROOT / "EXPERIMENT_MATRIX.csv",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _replace(values: list[str], old: str, new: str) -> list[str]:
    return [value.replace(old, new) for value in values]


def build_roleaware_commands(target: str) -> dict[str, Any]:
    """Clone the frozen FCL-E3 command and change only split/run identity."""
    target = target.upper()
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    legacy_id = f"FCL-E3-GAPS-{target}"
    formal_id = f"FCL-RW-GAPS-{target}"
    commands = frozen.build_flower_commands(legacy_id)
    replacements = (
        (frozen.REMOTE_DATA_ROOT, REMOTE_DATA_ROOT),
        (frozen.PI_DATA_ROOT, PI_DATA_ROOT),
        (frozen.C2_DATA_ROOT, C2_DATA_ROOT),
        (legacy_id, formal_id),
    )
    for role in ("server", "client_c1", "client_c2"):
        values = list(commands[role])
        for old, new in replacements:
            values = _replace(values, old, new)
        commands[role] = values
    commands["protocol"].update(
        {
            "study_id": STUDY_ID,
            "split_protocol": "role_aware_target_20_80",
            "dataset": DATASET_NAME,
            "target": target,
            "target_test_selection": False,
        }
    )
    return commands


def validate_local_split() -> dict[str, Any]:
    expected = {
        "C3": (3360, 680, 2680),
        "C4": (1680, 320, 1360),
        "C5": (1680, 320, 1360),
    }
    observed: dict[str, Any] = {}
    for target, counts in expected.items():
        path = LOCAL_DATA_ROOT / f"client_{target[1:]}" / "stats.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = (
            int(payload["n_total"]),
            int(payload["n_calibration"]),
            int(payload["n_test"]),
        )
        if actual != counts or payload.get("role") != "target":
            raise RuntimeError(f"FAIL_CLOSED {target} role-aware split differs: {actual}")
        observed[target] = {
            "stats_path": str(path),
            "stats_sha256": sha256(path),
            "total": actual[0],
            "calibration": actual[1],
            "test": actual[2],
        }
    return observed


def execute_target(
    target: str,
    output: Path,
    freeze_commit: str,
    digest: str,
    ecs_host: str,
    pi_host: str,
    c2_host: str,
    timeout_hours: float,
) -> None:
    experiment_id = f"FCL-RW-GAPS-{target}"
    run_dir = output / experiment_id
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial run exists: {run_dir}")

    original_root = frozen.RESULT_ROOT
    original_builder = frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = output
        frozen.build_flower_commands = lambda _experiment_id: build_roleaware_commands(target)
        frozen.execute_full_fl(
            experiment_id,
            protocol_hash=digest,
            lock_payload={"freeze_commit": freeze_commit},
            ecs_host=ecs_host,
            pi_host=pi_host,
            c2_host=c2_host,
            timeout_hours=timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT = original_root
        frozen.build_flower_commands = original_builder


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split = validate_local_split()
    digest = protocol_hash()
    freeze_commit = git_head()
    lock_path = output.parent / "PRE_RUN_FREEZE.json"
    lock = {
        "schema_version": "iotj.gaps_roleaware_r84.pre_run.v1",
        "status": "FROZEN",
        "study_id": STUDY_ID,
        "freeze_commit": freeze_commit,
        "protocol_hash": digest,
        "seed": 42,
        "classifier_protocol": {
            "rounds": 25,
            "local_epochs": 1,
            "batch_size": 32,
            "optimizer": "Adam",
            "optimizer_lr": 5e-4,
            "selective_warmup_rounds": [1, 2, 3, 4, 5],
            "fixed_endpoint": 25,
        },
        "roleaware_split": split,
        "target_test_opened": False,
        "hyperparameter_search": False,
    }
    if lock_path.exists():
        observed = json.loads(lock_path.read_text(encoding="utf-8"))
        if observed != lock:
            raise RuntimeError("FAIL_CLOSED pre-run freeze differs")
    else:
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
        execute_target(
            target, output, freeze_commit, digest,
            args.ecs_host, args.pi_host, args.c2_host, args.timeout_hours,
        )
    (output.parent / "CLASSIFICATION_PROGRESS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
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
    parser.add_argument("--timeout-hours", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
