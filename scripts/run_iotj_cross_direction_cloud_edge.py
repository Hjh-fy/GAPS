"""Run frozen cross-direction B2/B5 manifests on ECS, Pi, and PC."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__:
    from scripts.generate_iotj_cross_direction_commands import (
        APPROVED_DIRECTION_IDS,
        APPROVED_GROUPS,
        validate_manifest,
    )
    from scripts.run_iotj_classification_cloud_edge import (
        ECS_SYNC_FLOWER_FILES,
        ECS_SYNC_ROOT_FILES,
        PI_SYNC_FLOWER_FILES,
        PI_SYNC_ROOT_FILES,
        _copy_run_back,
        _log,
        _popen_hidden,
        _remote_dir_has_files,
        _remote_launch_script,
        _remote_python,
        _remote_run_state,
        _run,
        _scp_to_remote,
        _ssh,
        _start_tunnels,
        _terminate_processes,
        _wait_for_pi,
    )
else:
    from generate_iotj_cross_direction_commands import (
        APPROVED_DIRECTION_IDS,
        APPROVED_GROUPS,
        validate_manifest,
    )
    from run_iotj_classification_cloud_edge import (
        ECS_SYNC_FLOWER_FILES,
        ECS_SYNC_ROOT_FILES,
        PI_SYNC_FLOWER_FILES,
        PI_SYNC_ROOT_FILES,
        _copy_run_back,
        _log,
        _popen_hidden,
        _remote_dir_has_files,
        _remote_launch_script,
        _remote_python,
        _remote_run_state,
        _run,
        _scp_to_remote,
        _ssh,
        _start_tunnels,
        _terminate_processes,
        _wait_for_pi,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "IOTJ-B2-B5-CROSS-DIRECTION-20260713"
DEFAULT_COMMAND_ROOT = REPO_ROOT / "results" / "iotj_b2_b5_cross_direction_20260713_commands"
DEFAULT_LOCAL_RESULTS_ROOT = REPO_ROOT / "results" / "iotj_b2_b5_cross_direction_20260713"
DEFAULT_LOCAL_LOG_ROOT = REPO_ROOT / "results" / "iotj_b2_b5_cross_direction_20260713_local_logs"
DATA_SUFFIXES = (
    "features",
    "classification_labels",
    "phase_labels",
    "regression_labels",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_ordered_manifests(
    command_root: Path, seed: int
) -> list[tuple[Path, dict[str, Any]]]:
    index_path = command_root / "command_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("study_id") != STUDY_ID:
        raise ValueError(f"unexpected command-index study: {index_path}")
    selected: list[tuple[Path, dict[str, Any]]] = []
    for run_name in index.get("training_runs", ()):
        path = command_root / str(run_name) / "command_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"unexpected manifest study: {path}")
        if manifest.get("run_name") != run_name:
            raise ValueError(f"run-name mismatch: {path}")
        if int(manifest["protocol"]["training_seed"]) != seed:
            continue
        if not manifest.get("scheduled_for_training"):
            raise ValueError(f"manifest is not scheduled for training: {path}")
        validate_manifest(manifest)
        selected.append((path, manifest))
    expected = [
        (direction_id, group_id)
        for direction_id in APPROVED_DIRECTION_IDS
        for group_id in APPROVED_GROUPS
    ]
    found = [
        (manifest["direction_id"], manifest["group_id"])
        for _path, manifest in selected
    ]
    if found != expected:
        raise ValueError(f"seed {seed} queue mismatch: expected {expected}, got {found}")
    return selected


def active_executors(manifest: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    rows = manifest["commands"]["clients"]
    return {
        executor: tuple(
            int(row["client_id"])
            for row in rows
            if row["executor"] == executor
        )
        for executor in ("pi", "pc")
    }


def assert_remote_run_is_fresh(
    *, running: bool, rounds: int, has_files: bool, remote_run_dir: str
) -> None:
    if running or rounds or has_files:
        raise RuntimeError(f"refusing to overwrite partial remote run: {remote_run_dir}")


def _split_files(client_id: int, split: str) -> set[str]:
    return {
        f"client_{client_id}/{split}_{suffix}.npy" for suffix in DATA_SUFFIXES
    }


def required_data_files(
    manifests: Sequence[tuple[Path, dict[str, Any]]], location: str
) -> dict[str, tuple[str, ...]]:
    if location not in {"pi", "pc", "ecs"}:
        raise ValueError(f"unknown data location: {location}")
    required: dict[str, set[str]] = {}
    for _path, manifest in manifests:
        protocol = manifest["protocol"]
        data_root = str(protocol["data_root"])
        if location == "ecs":
            files = required.setdefault(
                data_root, {"split_info.json", "norm_stats.npz"}
            )
            for client_id in protocol["source_clients"]:
                files.update(_split_files(int(client_id), "calibration"))
            files.update(
                _split_files(int(protocol["target_clients"][0]), "calibration")
            )
        else:
            clients = [
                int(row["client_id"])
                for row in manifest["commands"]["clients"]
                if row["executor"] == location
            ]
            if not clients:
                continue
            files = required.setdefault(
                data_root, {"split_info.json", "norm_stats.npz"}
            )
            for client_id in clients:
                files.update(_split_files(client_id, "train"))
    return {key: tuple(sorted(value)) for key, value in sorted(required.items())}


def _provenance_hashes(
    manifests: Sequence[tuple[Path, dict[str, Any]]]
) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for _path, manifest in manifests:
        root = str(manifest["protocol"]["data_root"])
        for relative, digest in manifest["provenance"]["active_file_sha256"].items():
            key = (root, str(relative))
            if key in hashes and hashes[key] != digest:
                raise ValueError(f"conflicting data hash for {root}/{relative}")
            hashes[key] = str(digest)
    return hashes


def verify_local_data(
    manifests: Sequence[tuple[Path, dict[str, Any]]], location: str
) -> None:
    hashes = _provenance_hashes(manifests)
    for root, relative_paths in required_data_files(manifests, location).items():
        for relative in relative_paths:
            path = REPO_ROOT / "dataset" / root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = hashes[(root, relative)]
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(f"local data hash mismatch: {path}")


def _remote_hashes(
    host: str,
    python_bin: str,
    project: str,
    requirements: dict[str, tuple[str, ...]],
) -> dict[str, str | None]:
    relative_paths = [
        f"dataset/{root}/{relative}"
        for root, paths in requirements.items()
        for relative in paths
    ]
    source = f"""
import hashlib
import json
from pathlib import Path
project = Path({project!r})
result = {{}}
for relative in {relative_paths!r}:
    path = project / relative
    if not path.is_file():
        result[relative] = None
        continue
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    result[relative] = digest.hexdigest()
print(json.dumps(result, sort_keys=True))
"""
    output = _remote_python(host, python_bin, source, timeout=300)
    return json.loads(output.splitlines()[-1])


def sync_missing_remote_data(
    host: str,
    python_bin: str,
    project: str,
    manifests: Sequence[tuple[Path, dict[str, Any]]],
    location: str,
) -> int:
    requirements = required_data_files(manifests, location)
    hashes = _provenance_hashes(manifests)
    current = _remote_hashes(host, python_bin, project, requirements)
    copied = 0
    for root, relative_paths in requirements.items():
        for relative in relative_paths:
            remote_relative = f"dataset/{root}/{relative}"
            expected = hashes[(root, relative)]
            if current.get(remote_relative) == expected:
                continue
            local_path = REPO_ROOT / "dataset" / root / relative
            if not local_path.is_file() or _sha256(local_path) != expected:
                raise RuntimeError(f"local source data failed provenance: {local_path}")
            remote_path = f"{project}/{remote_relative}"
            parent = remote_path.rsplit("/", 1)[0]
            _ssh(host, f"mkdir -p '{parent}'")
            _run(["scp", "-p", str(local_path), f"{host}:{remote_path}"], timeout=600)
            copied += 1
    verified = _remote_hashes(host, python_bin, project, requirements)
    for root, relative_paths in requirements.items():
        for relative in relative_paths:
            remote_relative = f"dataset/{root}/{relative}"
            if verified.get(remote_relative) != hashes[(root, relative)]:
                raise RuntimeError(f"remote data hash mismatch: {host}:{remote_relative}")
    return copied


def _sync_runtime(
    host: str,
    project: str,
    command_root: Path,
    *,
    root_files: Iterable[str],
    flower_files: Iterable[str],
) -> None:
    _ssh(host, f"mkdir -p '{project}/gaps_flower' '{project}/results'")
    _scp_to_remote(
        [REPO_ROOT / name for name in root_files],
        f"{host}:{project}/",
    )
    _scp_to_remote(
        [REPO_ROOT / "gaps_flower" / name for name in flower_files],
        f"{host}:{project}/gaps_flower/",
    )
    _run(
        ["scp", "-pr", str(command_root), f"{host}:{project}/results/"],
        timeout=300,
    )


def _assert_ecs_idle(ecs_host: str) -> None:
    busy = _ssh(
        ecs_host,
        "ps -eo pid,args | grep '[g]aps_flower.server_app' || true",
    ).stdout.strip()
    if busy:
        raise RuntimeError(f"an ECS Flower server is already running:\n{busy}")


def _preflight_code(ecs_host: str, ecs_project: str, pi_host: str, pi_project: str) -> None:
    ecs_source = (
        f"import os\nos.chdir({ecs_project!r})\n"
        "from gaps_flower.task import make_config\n"
        "from utils import compute_mmd2\n"
        "c=make_config(profile='proto_replay',seed=42)\n"
        "assert c.SEED==42 and c.USE_REPLAY_DISTILL and c.USE_PROTO_DECOUPLING\n"
        "print('ECS_CROSS_DIRECTION_OK')\n"
    )
    ecs_output = _remote_python(
        ecs_host, "/root/gaps_env/bin/python", ecs_source, timeout=60
    )
    if "ECS_CROSS_DIRECTION_OK" not in ecs_output:
        raise RuntimeError(f"unexpected ECS preflight output: {ecs_output}")
    pi_source = (
        f"import os\nos.chdir({pi_project!r})\n"
        "from gaps_flower.task import make_config\n"
        "c=make_config(profile='proto_replay',seed=42)\n"
        "assert c.SEED==42 and c.USE_REPLAY_DISTILL and c.USE_PROTO_DECOUPLING\n"
        "print('PI_CROSS_DIRECTION_OK')\n"
    )
    pi_output = _remote_python(
        pi_host,
        "/home/gaps/GAPS/gaps_rpi_env/bin/python",
        pi_source,
        timeout=60,
    )
    if "PI_CROSS_DIRECTION_OK" not in pi_output:
        raise RuntimeError(f"unexpected Pi preflight output: {pi_output}")
    local = _run(
        [
            sys.executable,
            "-c",
            (
                "from gaps_flower.task import make_config; "
                "c=make_config(profile='proto_replay',seed=42); "
                "assert c.USE_REPLAY_DISTILL and c.USE_PROTO_DECOUPLING; "
                "print('PC_CROSS_DIRECTION_OK')"
            ),
        ],
        timeout=60,
    )
    if "PC_CROSS_DIRECTION_OK" not in local.stdout:
        raise RuntimeError(f"unexpected PC preflight output: {local.stdout}")


def audit_recovered_run(run_dir: Path, *, expected_rounds: int) -> dict[str, Any]:
    required = (
        "history.json",
        "run_config.json",
        "server_latest.pth",
        "server_latest_adapted.pth",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"recovered run is missing files: {','.join(missing)}")
    client_stats = sorted(run_dir.glob("client_stats_round_*.json"))
    if len(client_stats) != expected_rounds:
        raise RuntimeError(
            f"expected {expected_rounds} client-stat files, found {len(client_stats)}"
        )
    domain_files = sorted(run_dir.glob("domain_adapt_round_*.json"))
    if len(domain_files) != expected_rounds:
        raise RuntimeError(
            f"expected {expected_rounds} domain adaptation files, found {len(domain_files)}"
        )
    checkpoint = run_dir / "server_latest_adapted.pth"
    audit = {
        "run_name": run_dir.name,
        "expected_rounds": expected_rounds,
        "client_stat_files": len(client_stats),
        "domain_adapt_files": len(domain_files),
        "checkpoint_sha256": _sha256(checkpoint),
        "artifact_files": sum(1 for path in run_dir.rglob("*") if path.is_file()),
    }
    (run_dir / "recovery_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _client_row(manifest: dict[str, Any], executor: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in manifest["commands"]["clients"]
        if row["executor"] == executor
    ]
    if len(rows) > 1:
        raise ValueError(f"manifest assigns multiple {executor} clients")
    return rows[0] if rows else None


def _run_one(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    ecs_host: str,
    pi_host: str,
    ecs_project: str,
    pi_project: str,
    commands_root_name: str,
    results_root: str,
    local_results_root: Path,
    local_log_root: Path,
    poll_seconds: int,
    timeout_minutes: int,
) -> None:
    run_name = str(manifest["run_name"])
    expected_rounds = int(manifest["training"]["rounds"])
    remote_run_dir = f"{ecs_project}/{results_root}/{run_name}"
    complete, running, rounds = _remote_run_state(
        ecs_host, ecs_project, results_root, run_name, requires_adapted=True
    )
    if complete:
        _log(f"Skipping complete run {run_name}; recovering existing artifacts")
        _copy_run_back(ecs_host, remote_run_dir, local_results_root)
        audit_recovered_run(local_results_root / run_name, expected_rounds=expected_rounds)
        return
    assert_remote_run_is_fresh(
        running=running,
        rounds=rounds,
        has_files=_remote_dir_has_files(ecs_host, remote_run_dir),
        remote_run_dir=remote_run_dir,
    )

    manifest_dir = manifest_path.parent.name
    server_script = f"{ecs_project}/results/{commands_root_name}/{manifest_dir}/server_command.sh"
    server_log = f"{remote_run_dir}/server_launch.log"
    local_run_logs = local_log_root / run_name
    local_run_logs.mkdir(parents=True, exist_ok=True)
    pi_row = _client_row(manifest, "pi")
    pc_row = _client_row(manifest, "pc")

    _log(f"Starting {run_name}: ECS server")
    server_pid = _remote_launch_script(
        ecs_host, "/root/gaps_env/bin/python", ecs_project, server_script, server_log
    )
    _log(f"ECS server pid={server_pid}; waiting 15 seconds")
    time.sleep(15)
    complete, running, _ = _remote_run_state(
        ecs_host, ecs_project, results_root, run_name, requires_adapted=True
    )
    if not running and not complete:
        raise RuntimeError(f"ECS server exited during startup: {run_name}")

    if pi_row is not None:
        pi_script = (
            f"{pi_project}/results/{commands_root_name}/{manifest_dir}/"
            f"{pi_row['script_name']}"
        )
        pi_log = (
            f"{pi_project}/results/iotj_b2_b5_cross_direction_20260713_pi_logs/"
            f"{run_name}/client_{pi_row['client_id']}.log"
        )
        pi_pid = _remote_launch_script(
            pi_host,
            "/home/gaps/GAPS/gaps_rpi_env/bin/python",
            pi_project,
            pi_script,
            pi_log,
        )
        _log(f"Raspberry Pi C{pi_row['client_id']} pid={pi_pid}")

    pc_process: subprocess.Popen[Any] | None = None
    stdout_handle = None
    stderr_handle = None
    if pc_row is not None:
        command = [str(item) for item in pc_row["command"]]
        stdout_handle = (
            local_run_logs / f"client_{pc_row['client_id']}_pc.stdout.log"
        ).open("ab", buffering=0)
        stderr_handle = (
            local_run_logs / f"client_{pc_row['client_id']}_pc.stderr.log"
        ).open("ab", buffering=0)
        pc_process = _popen_hidden(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        _log(f"PC C{pc_row['client_id']} pid={pc_process.pid}")

    deadline = time.monotonic() + timeout_minutes * 60
    try:
        while True:
            time.sleep(poll_seconds)
            complete, running, rounds = _remote_run_state(
                ecs_host, ecs_project, results_root, run_name, requires_adapted=True
            )
            pi_health = _ssh(
                pi_host,
                "vcgencmd measure_temp 2>/dev/null; vcgencmd get_throttled 2>/dev/null",
                check=False,
            ).stdout.replace("\n", " ").strip()
            pc_exit = pc_process.poll() if pc_process is not None else None
            _log(
                f"progress run={run_name} rounds={rounds}/{expected_rounds} "
                f"complete={complete} server_running={running} pc_exit={pc_exit} "
                f"pi={pi_health}"
            )
            if complete and not running:
                break
            if not running and not complete:
                raise RuntimeError(
                    f"server exited before completion: {run_name}, rounds={rounds}"
                )
            if pc_process is not None and pc_exit is not None and not complete:
                raise RuntimeError(
                    f"PC C{pc_row['client_id']} exited early with code {pc_exit}: {run_name}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"run exceeded {timeout_minutes} minutes: {run_name}"
                )
    finally:
        if pc_process is not None:
            if pc_process.poll() is None:
                pc_process.terminate()
            try:
                pc_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pc_process.kill()
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()

    _copy_run_back(ecs_host, remote_run_dir, local_results_root)
    audit = audit_recovered_run(
        local_results_root / run_name, expected_rounds=expected_rounds
    )
    _log(
        f"Completed {run_name}; checkpoint={audit['checkpoint_sha256'][:12]} "
        f"files={audit['artifact_files']}"
    )


def _filter_queue(
    manifests: Sequence[tuple[Path, dict[str, Any]]],
    directions: set[str],
    groups: set[str],
) -> list[tuple[Path, dict[str, Any]]]:
    selected = [
        row
        for row in manifests
        if row[1]["direction_id"] in directions and row[1]["group_id"] in groups
    ]
    if not selected:
        raise ValueError("the requested direction/group filter selected no runs")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-hosts", default="gaps@192.168.31.184")
    parser.add_argument("--ecs-project", default="/root/GAPS")
    parser.add_argument("--pi-project", default="/home/gaps/GAPS/flower_runtime")
    parser.add_argument("--command-root", type=Path, default=DEFAULT_COMMAND_ROOT)
    parser.add_argument(
        "--results-root", default="results/iotj_b2_b5_cross_direction_20260713"
    )
    parser.add_argument(
        "--local-results-root", type=Path, default=DEFAULT_LOCAL_RESULTS_ROOT
    )
    parser.add_argument("--local-log-root", type=Path, default=DEFAULT_LOCAL_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--directions", default=",".join(APPROVED_DIRECTION_IDS))
    parser.add_argument("--groups", default=",".join(APPROVED_GROUPS))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--run-timeout-minutes", type=int, default=300)
    parser.add_argument("--wait-for-pi-minutes", type=int, default=360)
    parser.add_argument("--pi-retry-seconds", type=int, default=60)
    parser.add_argument("--skip-code-sync", action="store_true")
    parser.add_argument("--skip-data-sync", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    all_manifests = load_ordered_manifests(args.command_root, args.seed)
    directions = {item.strip() for item in args.directions.split(",") if item.strip()}
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    unknown_directions = directions - set(APPROVED_DIRECTION_IDS)
    unknown_groups = groups - set(APPROVED_GROUPS)
    if unknown_directions or unknown_groups:
        raise ValueError(
            f"unapproved filters: directions={sorted(unknown_directions)}, "
            f"groups={sorted(unknown_groups)}"
        )
    manifests = _filter_queue(all_manifests, directions, groups)
    _log(
        "Loaded queue: "
        + ", ".join(
            f"{manifest['direction_id']}/{manifest['group_id']}"
            for _path, manifest in manifests
        )
    )
    if args.dry_run:
        for path, manifest in manifests:
            topology = active_executors(manifest)
            _log(
                f"dry-run {manifest['run_name']} pi={topology['pi']} "
                f"pc={topology['pc']} <- {path}"
            )
        return 0

    verify_local_data(all_manifests, "pc")
    _assert_ecs_idle(args.ecs_host)
    pi_hosts = tuple(item.strip() for item in args.pi_hosts.split(",") if item.strip())
    pi_host = _wait_for_pi(
        pi_hosts, args.wait_for_pi_minutes, args.pi_retry_seconds
    )
    if not args.skip_code_sync:
        _sync_runtime(
            args.ecs_host,
            args.ecs_project,
            args.command_root,
            root_files=ECS_SYNC_ROOT_FILES,
            flower_files=ECS_SYNC_FLOWER_FILES,
        )
        _sync_runtime(
            pi_host,
            args.pi_project,
            args.command_root,
            root_files=PI_SYNC_ROOT_FILES,
            flower_files=PI_SYNC_FLOWER_FILES,
        )
    if not args.skip_data_sync:
        ecs_copied = sync_missing_remote_data(
            args.ecs_host,
            "/root/gaps_env/bin/python",
            args.ecs_project,
            all_manifests,
            "ecs",
        )
        pi_copied = sync_missing_remote_data(
            pi_host,
            "/home/gaps/GAPS/gaps_rpi_env/bin/python",
            args.pi_project,
            all_manifests,
            "pi",
        )
        _log(f"Data sync copied ECS={ecs_copied}, Pi={pi_copied} files")
    _preflight_code(args.ecs_host, args.ecs_project, pi_host, args.pi_project)
    _assert_ecs_idle(args.ecs_host)
    pi_health = _ssh(
        pi_host,
        "df -h .; vcgencmd measure_temp 2>/dev/null; vcgencmd get_throttled 2>/dev/null",
        check=False,
    ).stdout.strip()
    _log(f"Preflight complete; Pi health:\n{pi_health}")
    if args.preflight_only:
        return 0

    tunnels = _start_tunnels(args.ecs_host, pi_host)
    try:
        for manifest_path, manifest in manifests:
            _run_one(
                manifest_path,
                manifest,
                ecs_host=args.ecs_host,
                pi_host=pi_host,
                ecs_project=args.ecs_project,
                pi_project=args.pi_project,
                commands_root_name=args.command_root.name,
                results_root=args.results_root,
                local_results_root=args.local_results_root,
                local_log_root=args.local_log_root,
                poll_seconds=args.poll_seconds,
                timeout_minutes=args.run_timeout_minutes,
            )
    finally:
        _terminate_processes(tunnels)
    _log("All requested cross-direction classification runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
