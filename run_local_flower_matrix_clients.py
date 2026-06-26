"""Launch local Flower clients for one matrix run.

Use this after starting the corresponding Alibaba Cloud/ECS server command generated
by ``generate_flower_matrix_commands.py``.

Example:

    python run_local_flower_matrix_clients.py \
      --matrix-config configs/source_target_matrix_20260627.json \
      --run-id F6_C12_to_C345 \
      --server-address YOUR_ECS_IP:8080 \
      --local-project-dir "D:/A Python learning/Federated Learning/TRAE SOLO" \
      --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" \
      --local-python python \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_matrix(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = dict(payload.get("defaults", {}))
    runs = []
    for run in payload.get("runs", []):
        merged = dict(defaults)
        merged.update(run)
        runs.append(merged)
    return defaults, runs


def get_run(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    for run in runs:
        if str(run.get("run_id")) == run_id:
            return run
    raise SystemExit(f"Unknown run_id: {run_id}")


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", buffering=1)


def build_client_cmd(run: dict[str, Any], args: argparse.Namespace, client_id: int) -> list[str]:
    return [
        str(args.local_python),
        "-m",
        "gaps_flower.client_app",
        "--server-address",
        str(args.server_address),
        "--client-id",
        str(int(client_id)),
        "--data-root",
        str(args.local_data_root),
        "--device",
        str(args.device),
        "--local-epochs",
        str(int(run.get("local_epochs", 5))),
        "--batch-size",
        str(int(run.get("batch_size", 32))),
        "--profile",
        str(run.get("profile", "strong_cls")),
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch local source clients for one Flower matrix run.")
    parser.add_argument("--matrix-config", default="configs/source_target_matrix_20260627.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--local-project-dir", default=".")
    parser.add_argument("--local-data-root", required=True)
    parser.add_argument("--local-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-root", default="results/source_target_classification_matrix_20260627_local_client_logs")
    parser.add_argument("--start-interval-s", type=float, default=2.0)
    parser.add_argument("--wait", action="store_true", help="Wait until all local client processes exit.")
    args = parser.parse_args()

    _defaults, runs = load_matrix(args.matrix_config)
    run = get_run(runs, args.run_id)
    log_dir = Path(args.log_root) / str(args.run_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    manifest = {
        "run_id": args.run_id,
        "source_clients": [int(x) for x in run["source_clients"]],
        "target_clients": [int(x) for x in run["target_clients"]],
        "server_address": args.server_address,
        "client_logs": {},
    }

    for client_id in run["source_clients"]:
        cmd = build_client_cmd(run, args, int(client_id))
        log_path = log_dir / f"client_{int(client_id)}.log"
        log_f = open_log(log_path)
        log_f.write("# COMMAND\n" + " ".join(cmd) + "\n\n")
        print(f"Starting C{client_id}: {' '.join(cmd)}")
        popen = subprocess.Popen(
            cmd,
            cwd=args.local_project_dir,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        processes.append((int(client_id), popen, log_path))
        manifest["client_logs"][f"C{int(client_id)}"] = str(log_path)
        time.sleep(float(args.start_interval_s))

    write_json(log_dir / "local_client_manifest.json", manifest)

    if args.wait:
        returncodes = {}
        for client_id, popen, _log_path in processes:
            returncodes[f"C{client_id}"] = popen.wait()
        manifest["returncodes"] = returncodes
        write_json(log_dir / "local_client_manifest.json", manifest)
        failed = {k: v for k, v in returncodes.items() if v != 0}
        if failed:
            raise SystemExit(f"Some clients failed: {failed}")
    else:
        print(f"Started {len(processes)} client process(es). Logs: {log_dir}")


if __name__ == "__main__":
    main()
