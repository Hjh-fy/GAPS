"""Run frozen IoT-J classification ablations on ECS + Raspberry Pi + PC."""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_NAME = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
DEFAULT_COMMAND_ROOT = REPO_ROOT / "results" / "iotj_classification_ablation_20260711_v2_commands"
DEFAULT_LOCAL_RESULTS_ROOT = REPO_ROOT / "results" / "iotj_classification_ablation_20260711_v2"
DEFAULT_LOCAL_LOG_ROOT = REPO_ROOT / "results" / "iotj_classification_ablation_20260711_v2_local_client_logs"
DEFAULT_GROUPS = ("A0", "A0T", "A2", "A3", "A4", "A4S", "A5", "A6", "A7")
PI_SYNC_ROOT_FILES = ("client.py", "config.py", "federated_dataset.py", "model.py", "utils.py")
PI_SYNC_FLOWER_FILES = ("task.py", "client_app.py")


def _run(
    command: Sequence[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def _ssh(host: str, command: str, *, timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-n",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=2",
            host,
            command,
        ],
        timeout=timeout,
        check=check,
    )


def _remote_python(host: str, python_bin: str, source: str, *, timeout: int = 30) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    command = f"{python_bin} -c \"import base64;exec(base64.b64decode('{encoded}').decode('utf-8'))\""
    return _ssh(host, command, timeout=timeout).stdout.strip()


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def _load_manifest(command_root: Path, group_id: str, seed: int) -> tuple[Path, dict[str, Any]]:
    matches = sorted(command_root.glob(f"{group_id}_*_s{seed}_r25/command_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest for {group_id}/seed={seed}; found {len(matches)}")
    path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    if protocol.get("source_clients") != [1, 2] or protocol.get("target_clients") != [5]:
        raise RuntimeError(f"refusing non-C12-to-C5 manifest: {path}")
    if not payload.get("scheduled_for_training"):
        raise RuntimeError(f"manifest is not scheduled for training: {path}")
    return path, payload


def _wait_for_pi(hosts: Sequence[str], wait_minutes: int, retry_seconds: int) -> str:
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        for host in hosts:
            result = _ssh(host, "echo PI_READY", timeout=15, check=False)
            if result.returncode == 0 and "PI_READY" in result.stdout:
                _log(f"Raspberry Pi reachable at {host}")
                return host
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Raspberry Pi remained unreachable for {wait_minutes} minutes")
        _log(f"Raspberry Pi unreachable at {', '.join(hosts)}; retrying in {retry_seconds}s")
        time.sleep(retry_seconds)


def _scp_to_remote(paths: Sequence[Path], destination: str, timeout: int = 120) -> None:
    command = ["scp", "-p", *[str(path) for path in paths], destination]
    _run(command, timeout=timeout)


def _sync_pi(pi_host: str, command_root: Path, pi_project: str) -> None:
    _log("Synchronizing exact client runtime and command manifests to Raspberry Pi")
    _ssh(
        pi_host,
        f"mkdir -p '{pi_project}/gaps_flower' '{pi_project}/results'",
    )
    _scp_to_remote(
        [REPO_ROOT / name for name in PI_SYNC_ROOT_FILES],
        f"{pi_host}:{pi_project}/",
    )
    _scp_to_remote(
        [REPO_ROOT / "gaps_flower" / name for name in PI_SYNC_FLOWER_FILES],
        f"{pi_host}:{pi_project}/gaps_flower/",
    )
    _run(
        ["scp", "-pr", str(command_root), f"{pi_host}:{pi_project}/results/"],
        timeout=180,
    )


def _preflight_ecs(ecs_host: str, ecs_project: str) -> None:
    data_path = f"{ecs_project}/dataset/{DATA_ROOT_NAME}/client_5/calibration_features.npy"
    _ssh(ecs_host, f"test -f '{data_path}'")
    source = (
        f"import os\nos.chdir({ecs_project!r})\n"
        "from gaps_flower.task import make_config\n"
        "c=make_config(profile='proto_only',seed=43)\n"
        "assert c.SEED==43 and c.USE_ALIGN and not c.USE_REPLAY_DISTILL and not c.USE_REG_LOSS\n"
        "print('ECS_CODE_OK')\n"
    )
    output = _remote_python(ecs_host, "/root/gaps_env/bin/python", source)
    if "ECS_CODE_OK" not in output:
        raise RuntimeError(f"unexpected ECS code preflight output: {output}")
    busy = _ssh(
        ecs_host,
        "ps -eo pid,args | grep '[g]aps_flower.server_app' || true",
        check=True,
    ).stdout.strip()
    if busy:
        raise RuntimeError(f"an ECS Flower server is already running:\n{busy}")


def _preflight_pi(pi_host: str, pi_project: str) -> None:
    data_path = f"{pi_project}/dataset/{DATA_ROOT_NAME}/client_1/train_features.npy"
    _ssh(pi_host, f"test -f '{data_path}'")
    source = (
        "from gaps_flower.task import make_config\n"
        "c=make_config(profile='replay_only',seed=44)\n"
        "assert c.SEED==44 and not c.USE_ALIGN and c.USE_REPLAY_DISTILL and not c.USE_REG_LOSS\n"
        "print('PI_CODE_OK')\n"
    )
    output = _remote_python(
        pi_host,
        "/home/gaps/GAPS/gaps_rpi_env/bin/python",
        f"import os\nos.chdir({pi_project!r})\n" + source,
    )
    if "PI_CODE_OK" not in output:
        raise RuntimeError(f"unexpected Pi code preflight output: {output}")


def _preflight_pc() -> None:
    data_path = REPO_ROOT / "dataset" / DATA_ROOT_NAME / "client_2" / "train_features.npy"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    source = (
        "from gaps_flower.task import make_config; "
        "c=make_config(profile='ce_only',seed=45); "
        "assert c.SEED==45 and not c.USE_ALIGN and not c.USE_REPLAY_DISTILL; "
        "print('PC_CODE_OK')"
    )
    result = _run([sys.executable, "-c", source], timeout=30)
    if "PC_CODE_OK" not in result.stdout:
        raise RuntimeError(f"unexpected PC code preflight output: {result.stdout}")


def _popen_hidden(command: Sequence[str], **kwargs: Any) -> subprocess.Popen[Any]:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(list(command), creationflags=creationflags, **kwargs)


def _start_tunnels(ecs_host: str, pi_host: str) -> list[subprocess.Popen[Any]]:
    common = [
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-N",
    ]
    local = _popen_hidden(
        ["ssh", *common, "-L", "127.0.0.1:18080:127.0.0.1:8080", ecs_host],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    if local.poll() is not None:
        raise RuntimeError("PC-to-ECS SSH tunnel failed to start")
    reverse = _popen_hidden(
        ["ssh", *common, "-R", "127.0.0.1:18080:127.0.0.1:18080", pi_host],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    if reverse.poll() is not None:
        local.terminate()
        raise RuntimeError("PC-to-Pi reverse SSH tunnel failed to start")
    return [local, reverse]


def _terminate_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def _remote_launch_script(
    host: str,
    python_bin: str,
    project: str,
    script_path: str,
    log_path: str,
) -> int:
    source = f"""
import os
import subprocess
from pathlib import Path
project = Path({project!r})
script = Path({script_path!r})
log_path = Path({log_path!r})
log_path.parent.mkdir(parents=True, exist_ok=True)
log = log_path.open('ab', buffering=0)
process = subprocess.Popen(
    ['bash', str(script)],
    cwd=project,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=log,
    start_new_session=True,
    close_fds=True,
)
print(process.pid)
"""
    output = _remote_python(host, python_bin, source)
    return int(output.splitlines()[-1])


def _remote_run_state(
    ecs_host: str,
    ecs_project: str,
    results_root: str,
    run_name: str,
    requires_adapted: bool,
) -> tuple[bool, bool, int]:
    run_dir = f"{ecs_project}/{results_root}/{run_name}"
    required = ["history.json", "client_stats_round_025.json", "server_latest.pth"]
    if requires_adapted:
        required.append("server_latest_adapted.pth")
    tests = " && ".join(f"test -f '{run_dir}/{name}'" for name in required)
    complete = _ssh(ecs_host, f"{tests} && echo COMPLETE || true").stdout.strip() == "COMPLETE"
    running = bool(
        _ssh(
            ecs_host,
            f"ps -eo pid,args | grep '[g]aps_flower.server_app' | grep '{run_name}' || true",
        ).stdout.strip()
    )
    rounds_text = _ssh(
        ecs_host,
        f"find '{run_dir}' -maxdepth 1 -name 'client_stats_round_*.json' 2>/dev/null | wc -l",
    ).stdout.strip()
    return complete, running, int(rounds_text or 0)


def _remote_dir_has_files(host: str, path: str) -> bool:
    output = _ssh(
        host,
        f"test -d '{path}' && find '{path}' -mindepth 1 -maxdepth 1 -print -quit || true",
    ).stdout.strip()
    return bool(output)


def _copy_run_back(ecs_host: str, remote_run_dir: str, local_results_root: Path) -> None:
    local_results_root.mkdir(parents=True, exist_ok=True)
    _run(
        ["scp", "-pr", f"{ecs_host}:{remote_run_dir}", str(local_results_root)],
        timeout=300,
    )


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
    remote_run_dir = f"{ecs_project}/{results_root}/{run_name}"
    requires_adapted = bool(manifest["server_adaptation"]["enabled"])
    complete, running, rounds = _remote_run_state(
        ecs_host, ecs_project, results_root, run_name, requires_adapted
    )
    if complete:
        _log(f"Skipping complete run {run_name}")
        _copy_run_back(ecs_host, remote_run_dir, local_results_root)
        return
    if running or rounds or _remote_dir_has_files(ecs_host, remote_run_dir):
        raise RuntimeError(f"refusing to overwrite partial remote run: {remote_run_dir}")

    manifest_dir = manifest_path.parent.name
    server_script = f"{ecs_project}/results/{commands_root_name}/{manifest_dir}/server_command.sh"
    pi_script = f"{pi_project}/results/{commands_root_name}/{manifest_dir}/client_c1_pi_command.sh"
    server_log = f"{remote_run_dir}/server_launch.log"
    pi_log = f"{pi_project}/results/iotj_classification_ablation_20260711_pi_logs/{run_name}/client_1.log"
    local_run_logs = local_log_root / run_name
    local_run_logs.mkdir(parents=True, exist_ok=True)

    _log(f"Starting {run_name}: ECS server")
    server_pid = _remote_launch_script(
        ecs_host, "/root/gaps_env/bin/python", ecs_project, server_script, server_log
    )
    _log(f"ECS server pid={server_pid}; waiting 15 seconds")
    time.sleep(15)

    complete, running, _ = _remote_run_state(
        ecs_host, ecs_project, results_root, run_name, requires_adapted
    )
    if not running and not complete:
        raise RuntimeError(f"ECS server exited during startup: {run_name}")

    _log(f"Starting {run_name}: Raspberry Pi C1")
    pi_pid = _remote_launch_script(
        pi_host,
        "/home/gaps/GAPS/gaps_rpi_env/bin/python",
        pi_project,
        pi_script,
        pi_log,
    )
    _log(f"Raspberry Pi client pid={pi_pid}")

    pc_command = [str(item) for item in manifest["commands"]["client_c2_pc"]]
    stdout_handle = (local_run_logs / "client_2_pc.stdout.log").open("ab", buffering=0)
    stderr_handle = (local_run_logs / "client_2_pc.stderr.log").open("ab", buffering=0)
    pc_process = _popen_hidden(
        pc_command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    _log(f"PC C2 client pid={pc_process.pid}")

    deadline = time.monotonic() + timeout_minutes * 60
    try:
        while True:
            time.sleep(poll_seconds)
            complete, running, rounds = _remote_run_state(
                ecs_host, ecs_project, results_root, run_name, requires_adapted
            )
            pi_health = _ssh(
                pi_host,
                "vcgencmd measure_temp 2>/dev/null; vcgencmd get_throttled 2>/dev/null",
                check=False,
            ).stdout.replace("\n", " ").strip()
            _log(
                f"progress run={run_name} rounds={rounds}/25 complete={complete} "
                f"server_running={running} pc_exit={pc_process.poll()} pi={pi_health}"
            )
            if complete and not running:
                break
            if not running and not complete:
                raise RuntimeError(f"server exited before completion: {run_name}, rounds={rounds}")
            if pc_process.poll() is not None and not complete:
                raise RuntimeError(f"PC C2 exited early with code {pc_process.returncode}: {run_name}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run exceeded {timeout_minutes} minutes: {run_name}")
    finally:
        if pc_process.poll() is None:
            pc_process.terminate()
        try:
            pc_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pc_process.kill()
        stdout_handle.close()
        stderr_handle.close()

    _copy_run_back(ecs_host, remote_run_dir, local_results_root)
    _log(f"Completed and recovered {run_name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument(
        "--pi-hosts",
        default="gaps@192.168.31.184",
        help="Comma-separated Raspberry Pi SSH endpoints tried in order",
    )
    parser.add_argument("--ecs-project", default="/root/GAPS")
    parser.add_argument("--pi-project", default="/home/gaps/GAPS/flower_runtime")
    parser.add_argument("--command-root", type=Path, default=DEFAULT_COMMAND_ROOT)
    parser.add_argument("--results-root", default="results/iotj_classification_ablation_20260711_v2")
    parser.add_argument("--local-results-root", type=Path, default=DEFAULT_LOCAL_RESULTS_ROOT)
    parser.add_argument("--local-log-root", type=Path, default=DEFAULT_LOCAL_LOG_ROOT)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--run-timeout-minutes", type=int, default=300)
    parser.add_argument("--wait-for-pi-minutes", type=int, default=360)
    parser.add_argument("--pi-retry-seconds", type=int, default=60)
    parser.add_argument("--skip-pi-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    manifests = [_load_manifest(args.command_root, group, args.seed) for group in groups]
    _log(f"Loaded {len(manifests)} frozen manifests: {', '.join(groups)}")
    if args.dry_run:
        for path, manifest in manifests:
            _log(f"dry-run {manifest['run_name']} <- {path}")
        return 0

    _preflight_pc()
    _preflight_ecs(args.ecs_host, args.ecs_project)
    pi_hosts = tuple(item.strip() for item in args.pi_hosts.split(",") if item.strip())
    pi_host = _wait_for_pi(pi_hosts, args.wait_for_pi_minutes, args.pi_retry_seconds)
    if not args.skip_pi_sync:
        _sync_pi(pi_host, args.command_root, args.pi_project)
    _preflight_pi(pi_host, args.pi_project)
    tunnels = _start_tunnels(args.ecs_host, pi_host)
    try:
        for path, manifest in manifests:
            _run_one(
                path,
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
    _log("All requested classification runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
