"""Run one registered R1-M2 federated baseline on the frozen three-host topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_iotj_confirmation_observability import (  # noqa: E402
    _start_ecs_c2_tunnels,
    _terminate_processes,
)


EXPERIMENT_IDS = {
    "fedprox-source": "R1M2-FEDPROX-SOURCE-S42",
    "fedavg-same-adapter": "R1M2-FEDAVG-SAME-ADAPTER-S42",
}
SEED = 42
ROUNDS = 25
LOCAL_EPOCHS = 5
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def ssh(host: str, command: str, *, timeout: float = 120.0) -> str:
    return run(
        ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        timeout=timeout,
    ).stdout


def deploy_archive(host: str, archive: Path, runtime: str) -> None:
    digest = sha256_file(archive)
    remote_archive = f"/tmp/r1m2_source_{digest}.tar"
    check = ssh(
        host,
        f"if test -f {shlex.quote(runtime + '/.source_archive_sha256')}; then cat {shlex.quote(runtime + '/.source_archive_sha256')}; fi",
    ).strip()
    if check:
        if check != digest:
            raise RuntimeError(f"runtime hash mismatch on {host}: {check} != {digest}")
        return
    run(["scp", "-p", str(archive), f"{host}:{remote_archive}"], timeout=300)
    command = (
        f"test ! -e {shlex.quote(runtime)} && "
        f"mkdir -p {shlex.quote(runtime)} && "
        f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(runtime)} && "
        f"printf '%s\\n' {shlex.quote(digest)} > {shlex.quote(runtime + '/.source_archive_sha256')}"
    )
    ssh(host, command, timeout=300)


def process_command(host: str, runtime: str, argv: list[str]) -> list[str]:
    remote = f"cd {shlex.quote(runtime)} && exec {shlex.join(argv)}"
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, remote
    ]


def ensure_idle(hosts: list[str]) -> None:
    for host in hosts:
        output = ssh(host, "pgrep -af 'gaps_flower.(server_app|client_app)' || true")
        if output.strip():
            raise RuntimeError(f"residual Flower process on {host}:\n{output}")


def server_argv(experiment: str, remote_output: str, server_data_root: str) -> list[str]:
    common = [
        "/root/gaps_env/bin/python", "-m", "gaps_flower.server_app",
        "--server-address", "0.0.0.0:8080",
        "--rounds", str(ROUNDS),
        "--min-clients", "2",
        "--seed", str(SEED),
        "--run-name", EXPERIMENT_IDS[experiment],
        "--output-dir", remote_output,
        "--save-history", "true",
    ]
    if experiment == "fedprox-source":
        return [*common, "--strategy", "fedavg", "--profile", "ce_only"]
    return [
        *common,
        "--strategy", "gaps",
        "--profile", "ce_stats",
        "--use-selective-agg", "false",
        "--use-proto-mmd", "false",
        "--da-preset", "none",
        "--use-domain-adapt", "true",
        "--server-val-data", f"{server_data_root}/client_1,{server_data_root}/client_2",
        "--server-calib-data", f"{server_data_root}/client_5",
        "--domain-adapt-steps", "100",
        "--domain-adapt-warmup", "0",
        "--da-use-coral", "true",
        "--da-use-mmd", "true",
        "--da-use-adversarial", "true",
        "--da-mmd-objective", "mmd2",
        "--da-stage-alignment", "cross_domain_same_class_phase",
        "--da-adv-feature-objective", "wasserstein_min",
        "--da-coral-class-conditional", "true",
        "--strict-calibration-split", "true",
        "--da-device", "cpu",
        "--use-adapted-as-global", "true",
        "--da-lambda-coral", "0.5",
        "--da-lambda-global-mmd", "0.5",
        "--da-lambda-class-mmd", "0.5",
        "--da-lambda-proto-anchor", "0.3",
        "--da-lambda-adv", "0.5",
        "--da-lambda-target-ce", "0.0",
        "--da-lambda-proto", "0.05",
        "--da-lambda-consistency", "2.0",
        "--da-lambda-residual", "0.1",
        "--da-lambda-proto-mmd", "0.0",
        "--da-lambda-stage-mmd", "0.2",
        "--da-target-ce-label-smoothing", "0.0",
        "--da-target-ce-class-balanced", "false",
        "--da-server-opt-lr", "0.0005",
    ]


def client_argv(
    experiment: str,
    *,
    python: str,
    client_id: int,
    data_root: str,
) -> list[str]:
    argv = [
        python, "-m", "gaps_flower.client_app",
        "--server-address", "127.0.0.1:18080",
        "--client-id", str(client_id),
        "--data-root", data_root,
        "--device", "cpu",
        "--local-epochs", str(LOCAL_EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--profile", "ce_only" if experiment == "fedprox-source" else "ce_stats",
        "--seed", str(SEED),
    ]
    if experiment == "fedprox-source":
        argv.extend(["--proximal-mu", "0.01"])
    return argv


def parameter_bytes(checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(
        sum(tensor.numel() * tensor.element_size() for tensor in payload["model_state"].values())
    )


def add_macro_f1(summary: dict) -> None:
    for row in summary.get("clients", []):
        confusion = row.get("confusion_matrix", [])
        f1_values = []
        recalls = {}
        for class_id, class_row in enumerate(confusion):
            tp = int(class_row[class_id])
            fn = int(sum(class_row) - tp)
            fp = int(sum(other[class_id] for other in confusion) - tp)
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            recalls[str(class_id)] = float(recall)
            f1_values.append(float(f1))
        row["macro_f1"] = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
        row["per_class_recall"] = recalls
    total = sum(int(row.get("num_examples", 0)) for row in summary.get("clients", []))
    summary["weighted_macro_f1"] = (
        sum(float(row["macro_f1"]) * int(row.get("num_examples", 0)) for row in summary.get("clients", [])) / total
        if total
        else 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_IDS), required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=5.0)
    parser.add_argument(
        "--output-root",
        default="results/iotj_r1_m2_baseline_fairness_seed42_20260803",
    )
    args = parser.parse_args()

    archive = Path(args.source_archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = sha256_file(archive)
    short_hash = archive_hash[:12]
    experiment_id = EXPERIMENT_IDS[args.experiment]
    output_dir = (REPO_ROOT / args.output_root / experiment_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    runtimes = {
        "ecs": f"/root/GAPS/r1m2_runtime/{short_hash}",
        "pi": f"/home/gaps/GAPS/r1m2_runtime/{short_hash}",
        "c2": f"/root/GAPS/r1m2_runtime/{short_hash}",
    }
    remote_output = f"/root/GAPS/r1m2_runs/{experiment_id}"
    hosts = [args.ecs_host, args.pi_host, args.c2_host]
    ensure_idle(hosts)
    if ssh(args.ecs_host, f"if test -e {shlex.quote(remote_output)}; then echo EXISTS; fi").strip():
        raise FileExistsError(f"remote formal output already exists: {remote_output}")

    deploy_archive(args.ecs_host, archive, runtimes["ecs"])
    deploy_archive(args.pi_host, archive, runtimes["pi"])
    deploy_archive(args.c2_host, archive, runtimes["c2"])

    server_data = "/root/GAPS/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    pi_data = "/home/gaps/GAPS/flower_runtime/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    c2_data = "/root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    commands = {
        "server": server_argv(args.experiment, remote_output, server_data),
        "client_c1": client_argv(
            args.experiment,
            python="/home/gaps/GAPS/gaps_rpi_env/bin/python",
            client_id=1,
            data_root=pi_data,
        ),
        "client_c2": client_argv(
            args.experiment,
            python="/root/gaps_c2_cpu_env/bin/python",
            client_id=2,
            data_root=c2_data,
        ),
    }
    (output_dir / "locked_commands.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    processes: list[subprocess.Popen] = []
    tunnels: list[subprocess.Popen] = []
    handles = []
    started = time.perf_counter()
    try:
        for label in ("server", "client_c1", "client_c2"):
            handles.extend(
                [
                    (output_dir / f"{label}.stdout.log").open("w", encoding="utf-8"),
                    (output_dir / f"{label}.stderr.log").open("w", encoding="utf-8"),
                ]
            )
        server = subprocess.Popen(
            process_command(args.ecs_host, runtimes["ecs"], commands["server"]),
            stdout=handles[0], stderr=handles[1], text=True,
        )
        processes.append(server)
        time.sleep(5)
        if server.poll() is not None:
            raise RuntimeError(f"server exited before clients: rc={server.returncode}")
        tunnels = list(_start_ecs_c2_tunnels(args.ecs_host, args.pi_host, args.c2_host))
        c1 = subprocess.Popen(
            process_command(args.pi_host, runtimes["pi"], commands["client_c1"]),
            stdout=handles[2], stderr=handles[3], text=True,
        )
        c2 = subprocess.Popen(
            process_command(args.c2_host, runtimes["c2"], commands["client_c2"]),
            stdout=handles[4], stderr=handles[5], text=True,
        )
        processes.extend([c1, c2])
        deadline = time.monotonic() + args.timeout_hours * 3600
        while any(process.poll() is None for process in processes):
            if time.monotonic() > deadline:
                raise TimeoutError(f"distributed run exceeded {args.timeout_hours} hours")
            for label, process in zip(("server", "client_c1", "client_c2"), processes):
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{label} failed with rc={process.returncode}")
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(f"non-zero process return codes: {[p.returncode for p in processes]}")
    finally:
        _terminate_processes(processes)
        _terminate_processes(tunnels)
        for handle in handles:
            handle.close()

    wall_seconds = time.perf_counter() - started
    remote_copy = output_dir / "remote_server"
    run(["scp", "-r", f"{args.ecs_host}:{remote_output}", str(remote_copy)], timeout=600)
    checkpoint_name = (
        "server_latest.pth"
        if args.experiment == "fedprox-source"
        else "server_latest_adapted.pth"
    )
    checkpoint = remote_copy / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metrics_path = output_dir / "metrics.json"
    run(
        [
            sys.executable, "-m", "gaps_flower.evaluate_checkpoint",
            "--checkpoint", str(checkpoint),
            "--data-root", str(REPO_ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"),
            "--client-ids", "5", "--split", "test", "--device", "cpu",
            "--batch-size", str(BATCH_SIZE), "--output", str(metrics_path),
        ],
        timeout=600,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    add_macro_f1(metrics)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_bytes = parameter_bytes(checkpoint)
    manifest = {
        "schema_version": "iotj.r1_m2.seed42.distributed.v1",
        "experiment_id": experiment_id,
        "status": "completed",
        "seed": SEED,
        "source_archive": str(archive),
        "source_archive_sha256": archive_hash,
        "topology": {
            "server": args.ecs_host,
            "C1": args.pi_host,
            "C2": args.c2_host,
            "transport": "Flower over three loopback-only SSH tunnels",
        },
        "training": {
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "client_lr": 0.0005,
            "fedprox_mu": 0.01 if args.experiment == "fedprox-source" else 0.0,
            "profile": "ce_only" if args.experiment == "fedprox-source" else "ce_stats",
            "target_test_used_for_selection": False,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": metrics,
        "cost": {
            "wall_seconds": float(wall_seconds),
            "communication_rounds": ROUNDS,
            "model_parameter_bytes_per_message": model_bytes,
            "model_payload_bytes_total": int(4 * ROUNDS * model_bytes),
            "fedprox_additional_transmitted_state_bytes": 0,
            "statistics_payload_note": "ce_stats prototype/statistic JSON is additional to the model-payload total and retained in round artifacts",
        },
        "commands_path": str(output_dir / "locked_commands.json"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
