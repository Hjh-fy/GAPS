"""Run P0-A pure CE-only FedAvg on the frozen three-host topology."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_iotj_r1_m2_distributed_baselines import (  # noqa: E402
    deploy_archive,
    ensure_idle,
    process_command,
    run,
    sha256_file,
    ssh,
)
from scripts.run_iotj_confirmation_observability import (  # noqa: E402
    _start_ecs_c2_tunnels,
    _terminate_processes,
)

EXPERIMENT_ID = "P0A-PURE-FEDAVG-LE1-S42"
ROUNDS = 25
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
SEED = 42
CLIENT_LR = 5e-4


def server_argv(remote_output: str) -> list[str]:
    return [
        "/root/gaps_env/bin/python", "-m", "gaps_flower.server_app",
        "--server-address", "0.0.0.0:8080", "--rounds", str(ROUNDS),
        "--min-clients", "2", "--seed", str(SEED), "--run-name", EXPERIMENT_ID,
        "--output-dir", remote_output, "--save-history", "true",
        "--strategy", "fedavg", "--profile", "ce_only",
    ]


def client_argv(python: str, client_id: int, data_root: str) -> list[str]:
    return [
        python, "-m", "gaps_flower.client_app", "--server-address", "127.0.0.1:18080",
        "--client-id", str(client_id), "--data-root", data_root, "--device", "cpu",
        "--local-epochs", str(LOCAL_EPOCHS), "--batch-size", str(BATCH_SIZE),
        "--profile", "ce_only", "--seed", str(SEED), "--proximal-mu", "0",
    ]


def _checkpoint_index(remote_copy: Path) -> list[dict]:
    rows = []
    for round_id in range(1, ROUNDS + 1):
        path = remote_copy / f"server_round_{round_id:03d}.pth"
        if not path.is_file():
            raise RuntimeError(f"FAIL_CLOSED missing checkpoint: {path}")
        rows.append({
            "round": round_id, "filename": path.name,
            "absolute_local_path": str(path.resolve()), "role": "source_fedavg",
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        })
    latest = remote_copy / "server_latest.pth"
    if not latest.is_file() or sha256_file(latest) != rows[-1]["sha256"]:
        raise RuntimeError("FAIL_CLOSED server_latest.pth is not round 25")
    return rows


def _client_curve(remote_copy: Path) -> list[dict]:
    rows = []
    for round_id in range(1, ROUNDS + 1):
        payload = json.loads(
            (remote_copy / f"client_stats_round_{round_id:03d}.json").read_text(encoding="utf-8")
        )
        for client in payload["clients"]:
            rows.append({
                "round": round_id, "client_id": f"C{int(client['client_id'])}",
                "train_ce_mean": float(client["train_ce_mean"]),
                "train_accuracy": float(client["train_accuracy"]),
                "num_examples": int(client["num_examples"]),
                "local_epochs": int(client["local_epochs"]),
                "fit_seconds": float(client["fit_seconds"]),
                "train_ce_averaging": client["train_ce_averaging"],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=3.0)
    parser.add_argument("--output-root", default="results/iotj_p0_routing_simplification_20260803")
    args = parser.parse_args()

    archive = Path(args.source_archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = sha256_file(archive)
    short_hash = archive_hash[:12]
    output_dir = (REPO_ROOT / args.output_root / "P0A_PURE_FEDAVG_LE1_S42").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    runtimes = {
        "ecs": f"/root/GAPS/p0_runtime/{short_hash}",
        "pi": f"/home/gaps/GAPS/p0_runtime/{short_hash}",
        "c2": f"/root/GAPS/p0_runtime/{short_hash}",
    }
    remote_output = f"/root/GAPS/p0_runs/{EXPERIMENT_ID}_{short_hash}"
    hosts = [args.ecs_host, args.pi_host, args.c2_host]
    ensure_idle(hosts)
    if ssh(args.ecs_host, f"if test -e {shlex.quote(remote_output)}; then echo EXISTS; fi").strip():
        raise FileExistsError(f"remote formal output exists: {remote_output}")
    for host, runtime in ((args.ecs_host, runtimes["ecs"]), (args.pi_host, runtimes["pi"]), (args.c2_host, runtimes["c2"])):
        deploy_archive(host, archive, runtime)

    server_data = "/root/GAPS/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    pi_data = "/home/gaps/GAPS/flower_runtime/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    c2_data = "/root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    for host, path, clients in ((args.ecs_host, server_data, "1 2 5"), (args.pi_host, pi_data, "1"), (args.c2_host, c2_data, "2")):
        ssh(host, f"for c in {clients}; do test -d {shlex.quote(path)}/client_$c || exit 17; done")

    commands = {
        "server": server_argv(remote_output),
        "client_c1": client_argv("/home/gaps/GAPS/gaps_rpi_env/bin/python", 1, pi_data),
        "client_c2": client_argv("/root/gaps_c2_cpu_env/bin/python", 2, c2_data),
    }
    (output_dir / "locked_commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    preflight = {
        "status": "PASS", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "source_archive": str(archive), "source_archive_sha256": archive_hash,
        "hosts": {"server": args.ecs_host, "C1": args.pi_host, "C2": args.c2_host},
        "dataset_checked": True, "residual_flower_processes": False,
    }
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    processes: list[subprocess.Popen] = []
    tunnels: list[subprocess.Popen] = []
    handles = []
    started = time.perf_counter()
    try:
        for label in ("server", "client_c1", "client_c2"):
            handles += [(output_dir / f"{label}.stdout.log").open("w", encoding="utf-8"), (output_dir / f"{label}.stderr.log").open("w", encoding="utf-8")]
        processes.append(subprocess.Popen(process_command(args.ecs_host, runtimes["ecs"], commands["server"]), stdout=handles[0], stderr=handles[1], text=True))
        time.sleep(5)
        if processes[0].poll() is not None:
            raise RuntimeError("server exited before clients")
        tunnels = list(_start_ecs_c2_tunnels(args.ecs_host, args.pi_host, args.c2_host))
        processes.append(subprocess.Popen(process_command(args.pi_host, runtimes["pi"], commands["client_c1"]), stdout=handles[2], stderr=handles[3], text=True))
        processes.append(subprocess.Popen(process_command(args.c2_host, runtimes["c2"], commands["client_c2"]), stdout=handles[4], stderr=handles[5], text=True))
        deadline = time.monotonic() + args.timeout_hours * 3600
        while any(p.poll() is None for p in processes):
            if time.monotonic() > deadline:
                raise TimeoutError("P0-A exceeded timeout")
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"process failure: {[p.poll() for p in processes]}")
            time.sleep(10)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f"non-zero process codes: {[p.returncode for p in processes]}")
    finally:
        _terminate_processes(processes); _terminate_processes(tunnels)
        for handle in handles: handle.close()

    remote_copy = output_dir / "remote_server"
    run(["scp", "-r", f"{args.ecs_host}:{remote_output}", str(remote_copy)], timeout=600)
    checkpoint_index = _checkpoint_index(remote_copy)
    client_curve = _client_curve(remote_copy)
    (output_dir / "checkpoint_index.json").write_text(json.dumps(checkpoint_index, indent=2) + "\n", encoding="utf-8")
    (output_dir / "client_training_curve.json").write_text(json.dumps(client_curve, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "iotj.p0.source_fedavg.v1", "experiment_id": EXPERIMENT_ID,
        "status": "completed", "seed": SEED, "dataset": "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
        "source_clients": ["C1", "C2"], "target_client": "C5", "target_access": "none",
        "training": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE, "client_lr": CLIENT_LR, "optimizer": "Adam", "profile": "ce_only", "aggregation": "sample_weighted_FedAvg", "fedprox_mu": 0.0},
        "instrumentation": {"train_ce_averaging": "sample_weighted_over_local_minibatches", "extra_forward_pass": False},
        "target_test_used_for_selection": False, "formal_checkpoint_round": 25,
        "source_archive_sha256": archive_hash, "wall_seconds": time.perf_counter() - started,
        "checkpoint_index": str(output_dir / "checkpoint_index.json"),
    }
    (output_dir / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
