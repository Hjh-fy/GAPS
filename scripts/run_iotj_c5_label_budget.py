"""Run the frozen six-endpoint canonical-v1 C5 low-label Flower study."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "results/iotj_canonical_v1_c5_budget_20260810"
RUN_ROOT = STUDY_ROOT / "classification"
BUDGET_DATA = STUDY_ROOT / "budget_data"
REMOTE_BUDGET_ROOT = "/root/GAPS/dataset/iotj_canonical_v1_c5_budget_20260810"
DESIGN = ROOT / "docs/superpowers/specs/2026-08-10-c5-low-label-commissioning-design.md"
METHODS = ("A0T", "A4")
BUDGETS = (15, 10, 5)


def _frozen_modules():
    from scripts import run_iotj_canonical_v1_a0t as a0t
    from scripts import run_iotj_canonical_v1_classification as a4
    from scripts import run_iotj_final_classification_le1 as frozen
    return a0t, a4, frozen


def experiment_id(method: str, budget: int) -> str:
    method = method.upper()
    if method not in METHODS or int(budget) not in BUDGETS:
        raise ValueError(f"unsupported C5 budget configuration: {method}/{budget}")
    return f"CAN-V1-C5-LB-{method}-B{int(budget):02d}-S42"


def _replace(values: list[str], old: str, new: str) -> list[str]:
    return [value.replace(old, new) for value in values]


def _set_option(values: list[str], option: str, value: str) -> None:
    values[values.index(option) + 1] = value


def build_budget_commands(method: str, budget: int) -> dict[str, Any]:
    method = method.upper()
    budget = int(budget)
    run_id = experiment_id(method, budget)
    a0t, a4, _frozen = _frozen_modules()
    if method == "A0T":
        commands = copy.deepcopy(a0t.build_a0t_commands("C5"))
        old_id = "CANONICAL-V1-A0T-C5"
    else:
        commands = copy.deepcopy(a4.build_canonical_commands("C5"))
        old_id = "CANONICAL-V1-A4-C5"
    for role in ("server", "client_c1", "client_c2"):
        commands[role] = _replace(commands[role], old_id, run_id)
    _set_option(
        commands["server"],
        "--server-calib-data",
        f"{REMOTE_BUDGET_ROOT}/client_5_budget_{budget:02d}",
    )
    commands["protocol"].update({
        "experiment_id": run_id,
        "study_id": "iotj_canonical_v1_c5_budget_20260810",
        "method": method,
        "target": "C5",
        "budget_pct": budget,
        "calibration_n": {15: 240, 10: 160, 5: 80}[budget],
        "calibration_strata": 40,
        "calibration_per_stratum": {15: 6, 10: 4, 5: 2}[budget],
        "nested_family": "05_subset_10_subset_15_subset_existing_20",
        "initialization": "fresh_seed42_random_initialization",
        "checkpoint_reuse": False,
        "checkpoint_selection": "fixed_round_25",
        "target_test_selection": False,
        "hyperparameter_search": False,
    })
    return commands


def audit_commands() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    target_test_referenced = False
    checkpoint_reuse_referenced = False
    for method in METHODS:
        for budget in BUDGETS:
            commands = build_budget_commands(method, budget)
            server = commands["server"]
            flat = [
                token
                for role in ("server", "client_c1", "client_c2")
                for token in commands[role]
            ]
            joined = " ".join(flat).lower()
            target_test_referenced |= any(
                marker in joined
                for marker in ("test_features", "test_classification", "test_labels")
            )
            checkpoint_reuse_referenced |= any(
                token in flat for token in ("--checkpoint", "--resume", "--resume-from")
            )
            row = {
                "experiment_id": experiment_id(method, budget),
                "method": method,
                "budget_pct": budget,
                "rounds_25": server[server.index("--rounds") + 1] == "25",
                "steps_100": server[server.index("--domain-adapt-steps") + 1] == "100",
                "server_lr_5e_4": server[server.index("--da-server-opt-lr") + 1] == "0.0005",
                "local_epochs_1": all(
                    commands[role][commands[role].index("--local-epochs") + 1] == "1"
                    for role in ("client_c1", "client_c2")
                ),
                "seed_42": all(
                    commands[role][commands[role].index("--seed") + 1] == "42"
                    for role in ("server", "client_c1", "client_c2")
                ),
                "budget_path": server[server.index("--server-calib-data") + 1]
                == f"{REMOTE_BUDGET_ROOT}/client_5_budget_{budget:02d}",
                "fixed_endpoint": commands["protocol"]["checkpoint_selection"]
                == "fixed_round_25",
                "fresh_initialization": commands["protocol"]["checkpoint_reuse"] is False,
            }
            if method == "A0T":
                row["method_surface"] = (
                    server[server.index("--profile") + 1] == "ce_only"
                    and server[server.index("--ablation-variant") + 1] == "A0T"
                    and server[server.index("--da-lambda-target-ce") + 1] == "1.0"
                    and server[server.index("--use-proto-mmd") + 1] == "false"
                )
            else:
                row["method_surface"] = (
                    server[server.index("--profile") + 1] == "ce_stats"
                    and server[server.index("--ablation-variant") + 1] == "A4"
                    and server[server.index("--da-lambda-target-ce") + 1] == "0.0"
                    and server[server.index("--use-proto-mmd") + 1] == "true"
                )
            rows.append(row)
    status = (
        "PASS"
        if len(rows) == 6
        and all(all(value is True for key, value in row.items() if key not in {"experiment_id", "method", "budget_pct"}) for row in rows)
        and not target_test_referenced
        and not checkpoint_reuse_referenced
        else "FAIL"
    )
    return {
        "schema_version": "gaps.iotj.c5_label_budget.command_audit.v1",
        "status": status,
        "run_count": len(rows),
        "target_test_referenced": target_test_referenced,
        "checkpoint_reuse_referenced": checkpoint_reuse_referenced,
        "runs": rows,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        DESIGN,
        STUDY_ROOT / "EXPERIMENT_PLAN.md",
        STUDY_ROOT / "EXPERIMENT_MATRIX.csv",
        STUDY_ROOT / "ABLATION_PLAN.md",
        STUDY_ROOT / "experiment_registry.csv",
        STUDY_ROOT / "c5_calibration_budget_manifest_sha256.json",
        STUDY_ROOT / "calibration_budget_audit.json",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps(
        [build_budget_commands(method, budget) for method in METHODS for budget in BUDGETS],
        sort_keys=True,
    ).encode("utf-8"))
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_or_create_freeze(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError("FAIL_CLOSED C5 label-budget pre-run freeze differs")
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def preflight() -> dict[str, Any]:
    audit_path = STUDY_ROOT / "calibration_budget_audit.json"
    index_path = STUDY_ROOT / "c5_calibration_budget_manifest_sha256.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("FAIL_CLOSED calibration budget audit is not PASS")
    if audit.get("counts") != {"20": 320, "15": 240, "10": 160, "5": 80}:
        raise RuntimeError("FAIL_CLOSED calibration budget counts differ")
    if audit.get("strata") != 40 or not audit.get("nested"):
        raise RuntimeError("FAIL_CLOSED nested strata audit differs")
    if audit.get("calibration_test_exact_identity_overlap") != 0:
        raise RuntimeError("FAIL_CLOSED calibration/test identity overlap")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for relative, expected in index["files"].items():
        path = STUDY_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"FAIL_CLOSED budget evidence SHA differs: {relative}")
    command_audit = audit_commands()
    if command_audit["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED command audit failed")
    return {
        "status": "PASS",
        "budget_manifest_sha256": sha256(index_path),
        "calibration_audit_sha256": sha256(audit_path),
        "command_audit": command_audit,
    }


def _budget_archive(manifest_sha256: str) -> Path:
    archive = STUDY_ROOT / "inputs" / f"c5_budget_data_{manifest_sha256[:12]}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with tarfile.open(archive, "w") as handle:
            for directory in sorted(BUDGET_DATA.iterdir()):
                if directory.name in {"client_5_budget_15", "client_5_budget_10", "client_5_budget_05"}:
                    handle.add(directory, arcname=directory.name)
    return archive


def deploy_budget_data(ecs_host: str, manifest_sha256: str) -> dict[str, str]:
    _a0t, _a4, frozen = _frozen_modules()
    archive = _budget_archive(manifest_sha256)
    archive_sha = sha256(archive)
    marker = f"{REMOTE_BUDGET_ROOT}/.manifest_sha256"
    observed = frozen._ssh(
        ecs_host,
        f"if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi",
    ).strip()
    if observed:
        if observed != manifest_sha256:
            raise RuntimeError("FAIL_CLOSED remote budget manifest hash differs")
        return {"remote_root": REMOTE_BUDGET_ROOT, "archive_sha256": archive_sha}
    partial = frozen._ssh(
        ecs_host,
        f"if test -e {shlex.quote(REMOTE_BUDGET_ROOT)}; then echo PARTIAL; fi",
    ).strip()
    if partial:
        raise RuntimeError("FAIL_CLOSED partial unhashed remote budget data exists")
    remote_archive = f"/tmp/c5_budget_{archive_sha}.tar"
    frozen._run(["scp", "-p", str(archive), f"{ecs_host}:{remote_archive}"], timeout=600)
    frozen._ssh(
        ecs_host,
        " && ".join([
            f"mkdir -p {shlex.quote(REMOTE_BUDGET_ROOT)}",
            f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(REMOTE_BUDGET_ROOT)}",
            f"printf '%s\\n' {shlex.quote(manifest_sha256)} > {shlex.quote(marker)}",
        ]),
        timeout=600,
    )
    return {"remote_root": REMOTE_BUDGET_ROOT, "archive_sha256": archive_sha}


def execute(args: argparse.Namespace) -> None:
    check = preflight()
    digest = _protocol_hash()
    freeze_commit = _git_head()
    freeze_payload = {
        "schema_version": "gaps.iotj.c5_label_budget.pre_run.v1",
        "status": "FROZEN",
        "freeze_commit": freeze_commit,
        "protocol_hash": digest,
        "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
        "preflight": check,
        "run_order": [experiment_id(method, budget) for method in METHODS for budget in BUDGETS],
        "test_open_policy": "only after all six fixed round25 endpoints complete",
        "stop_rule": "stop after unified C5 evaluation; no C3/C4, multi-seed, lower budget, R84, or QC",
    }
    load_or_create_freeze(STUDY_ROOT / "PRE_RUN_FREEZE.json", freeze_payload)
    deploy_budget_data(args.ecs_host, check["budget_manifest_sha256"])
    _a0t, _a4, frozen = _frozen_modules()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for method in METHODS:
        for budget in BUDGETS:
            run_id = experiment_id(method, budget)
            run_dir = RUN_ROOT / run_id
            if (run_dir / "fixed_endpoint_complete.json").is_file():
                frozen.experiment_resume_status(run_dir, expected_protocol_hash=digest)
                completed.append(run_id)
                continue
            if run_dir.exists():
                raise FileExistsError(f"FAIL_CLOSED partial budget run exists: {run_dir}")
            (STUDY_ROOT / "RUN_PROGRESS.json").write_text(
                json.dumps({
                    "status": "RUNNING",
                    "current_run": run_id,
                    "completed_runs": completed,
                    "target_test_opened": False,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            original_root = frozen.RESULT_ROOT
            original_builder = frozen.build_flower_commands
            try:
                frozen.RESULT_ROOT = RUN_ROOT
                frozen.build_flower_commands = lambda _experiment_id, m=method, b=budget: build_budget_commands(m, b)
                frozen.execute_full_fl(
                    run_id,
                    protocol_hash=digest,
                    lock_payload={"freeze_commit": freeze_commit, "study": "C5 low-label commissioning"},
                    ecs_host=args.ecs_host,
                    pi_host=args.pi_host,
                    c2_host=args.c2_host,
                    timeout_hours=args.timeout_hours,
                )
            finally:
                frozen.RESULT_ROOT = original_root
                frozen.build_flower_commands = original_builder
            completed.append(run_id)
    (STUDY_ROOT / "RUN_PROGRESS.json").write_text(
        json.dumps({
            "status": "FIXED_ENDPOINTS_COMPLETE_TEST_STILL_SEALED",
            "current_run": None,
            "completed_runs": completed,
            "target_test_opened": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
