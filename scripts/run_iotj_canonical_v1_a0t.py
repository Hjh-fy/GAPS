"""Run the preregistered canonical equal-label target-CE-only A0T baseline."""

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


TARGETS = ("C3", "C4", "C5")
DEFAULT_OUTPUT = (
    ROOT / "results" / "iotj_canonical_v1_final_20260808"
    / "a0t_equal_label" / "classification"
)
FREEZE_PATH = (
    ROOT / "docs" / "experiments" / "iotj_canonical_v1"
    / "EVIDENCE_CLOSURE_PROTOCOL_FREEZE.json"
)


def canonical_a0t_config() -> dict[str, Any]:
    return {
        "method": "A0T_equal_label_target_CE_only",
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "optimizer": "Adam",
        "optimizer_lr": 5e-4,
        "target_ce_steps_per_round": 100,
        "target_ce_weight": 1.0,
        "target_ce_label_smoothing": 0.0,
        "target_ce_class_balanced": False,
        "target_label_budget": "same_canonical_calibration_as_A4",
        "checkpoint_reuse": False,
        "checkpoint_selection": "fixed_round_25",
        "target_test_selection": False,
        "hyperparameter_search": False,
        "window_shape": [50, 8],
    }


def _set_option(values: list[str], option: str, value: str) -> None:
    values[values.index(option) + 1] = value


def build_a0t_commands(target: str) -> dict[str, Any]:
    target = target.upper()
    if target not in TARGETS:
        raise ValueError(f"unknown canonical target: {target}")
    commands = canonical.build_canonical_commands(target)
    experiment_id = f"CANONICAL-V1-A0T-{target}"
    old_id = f"CANONICAL-V1-A4-{target}"
    for role in ("server", "client_c1", "client_c2"):
        commands[role] = [value.replace(old_id, experiment_id) for value in commands[role]]
    server = commands["server"]
    for option, value in (
        ("--profile", "ce_only"),
        ("--ablation-variant", "A0T"),
        ("--target-information-method", "gaps"),
        ("--use-selective-agg", "false"),
        ("--require-selective-after-warmup", "false"),
        ("--use-proto-mmd", "false"),
        ("--use-domain-adapt", "true"),
        ("--da-use-coral", "false"),
        ("--da-use-mmd", "false"),
        ("--da-use-adversarial", "false"),
        ("--da-lambda-coral", "0.0"),
        ("--da-lambda-global-mmd", "0.0"),
        ("--da-lambda-class-mmd", "0.0"),
        ("--da-lambda-proto-anchor", "0.0"),
        ("--da-lambda-adv", "0.0"),
        ("--da-lambda-target-ce", "1.0"),
        ("--da-lambda-proto", "0.0"),
        ("--da-lambda-consistency", "0.0"),
        ("--da-lambda-residual", "0.0"),
        ("--da-lambda-proto-mmd", "0.0"),
        ("--da-lambda-stage-mmd", "0.0"),
    ):
        _set_option(server, option, value)
    _set_option(commands["client_c1"], "--profile", "ce_only")
    _set_option(commands["client_c2"], "--profile", "ce_only")
    commands["protocol"].update(canonical_a0t_config())
    commands["protocol"].update(
        {
            "experiment_id": experiment_id,
            "target": target,
            "classifier_router": "A0T",
            "adaptation_target_fields": ["x", "class"],
            "all_non_ce_target_adaptation_losses": 0.0,
            "comparison_to": f"CANONICAL-V1-A4-{target}",
        }
    )
    return commands


def protocol_hash() -> str:
    digest = hashlib.sha256()
    digest.update(FREEZE_PATH.read_bytes())
    digest.update(json.dumps(canonical_a0t_config(), sort_keys=True).encode("utf-8"))
    for target in TARGETS:
        digest.update(json.dumps(build_a0t_commands(target), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_or_create_freeze(lock_path: Path, current_head: str, digest: str) -> dict[str, Any]:
    invariant = {
        "schema_version": "iotj.canonical_v1.a0t.pre_run.v1",
        "status": "FROZEN",
        "protocol_hash": digest,
        "config": canonical_a0t_config(),
        "targets": list(TARGETS),
        "canonical_match_audit": "no fully matched canonical-v1 A0T artifact existed",
        "test_open_policy": "only after all three fixed round25 endpoints complete",
    }
    if lock_path.exists():
        observed = json.loads(lock_path.read_text(encoding="utf-8"))
        for key, value in invariant.items():
            if observed.get(key) != value:
                raise RuntimeError(f"FAIL_CLOSED A0T pre-run freeze differs: {key}")
        if not observed.get("freeze_commit"):
            raise RuntimeError("FAIL_CLOSED A0T pre-run freeze has no commit")
        return observed
    payload = {**invariant, "freeze_commit": current_head}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def execute_target(
    target: str,
    output: Path,
    freeze_commit: str,
    digest: str,
    args: argparse.Namespace,
) -> None:
    experiment_id = f"CANONICAL-V1-A0T-{target}"
    run_dir = output / experiment_id
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial A0T run exists: {run_dir}")
    original_root = frozen.RESULT_ROOT
    original_builder = frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = output
        frozen.build_flower_commands = lambda _experiment_id: build_a0t_commands(target)
        frozen.execute_full_fl(
            experiment_id,
            protocol_hash=digest,
            lock_payload={"freeze_commit": freeze_commit, "baseline": "canonical_A0T"},
            ecs_host=args.ecs_host,
            pi_host=args.pi_host,
            c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT = original_root
        frozen.build_flower_commands = original_builder


def run(args: argparse.Namespace) -> None:
    canonical.validate_preflight()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = protocol_hash()
    lock_path = output.parent / "A0T_PRE_RUN_FREEZE.json"
    lock = load_or_create_freeze(lock_path, git_head(), digest)
    freeze_commit = str(lock["freeze_commit"])
    for target in TARGETS:
        execute_target(target, output, freeze_commit, digest, args)


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
