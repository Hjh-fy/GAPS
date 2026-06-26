"""Generate commands for Stage-A Flower source-target classification matrix.

This script does not execute remote commands. It writes reproducible command files
for each source-target run:

- server command: run on Alibaba Cloud/ECS
- local client commands: run on the local PC, one process per source client

Use this first with the 10-run matrix config, inspect commands, then run the
server command and launch local clients with ``run_local_flower_matrix_clients.py``.

Important: the dataset root must match each run's source/target roles. If a path
contains a tag such as ``c12src_c345tgt``, the generator will warn when the run
uses a client outside that encoded source/target role.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable


BOOL_TRUE = {"true", "1", "yes", "y", "on"}
ROLE_TAG_RE = re.compile(r"c([0-9]+)src_c([0-9]+)tgt", re.IGNORECASE)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in BOOL_TRUE


def remote_join(*parts: str) -> str:
    """Join Linux/POSIX remote paths even when this script runs on Windows."""
    cleaned = [str(part).replace("\\", "/").rstrip("/") for part in parts if str(part)]
    if not cleaned:
        return ""
    first, *rest = cleaned
    return posixpath.join(first, *rest)


def remote_client_dir(data_root: str, client_id: int) -> str:
    return remote_join(data_root, f"client_{int(client_id)}")


def comma_remote_client_dirs(data_root: str, clients: Iterable[int]) -> str:
    return ",".join(remote_client_dir(data_root, int(client)) for client in clients)


def q_posix(value: str) -> str:
    return shlex.quote(str(value))


def q_windows_cmd(value: str) -> str:
    value = str(value)
    if not value:
        return '""'
    # cmd.exe uses double quotes; escape embedded double quotes conservatively.
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"' if any(ch.isspace() for ch in escaped) else escaped


def merge_run(defaults: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(run)
    return merged


def load_matrix(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = dict(payload.get("defaults", {}))
    runs = [merge_run(defaults, run) for run in payload.get("runs", [])]
    return defaults, runs


def select_runs(runs: list[dict[str, Any]], names: str | None) -> list[dict[str, Any]]:
    if not names or names.lower() == "all":
        return runs
    wanted = {item.strip() for item in names.split(",") if item.strip()}
    selected = [run for run in runs if str(run.get("run_id")) in wanted]
    missing = wanted - {str(run.get("run_id")) for run in selected}
    if missing:
        raise SystemExit(f"Unknown run_id(s): {sorted(missing)}")
    return selected


def parse_role_tag(path_text: str) -> tuple[set[int], set[int]] | None:
    """Parse role hints like ``c12src_c345tgt`` from a dataset path."""
    match = ROLE_TAG_RE.search(str(path_text).replace("\\", "/"))
    if not match:
        return None
    src = {int(ch) for ch in match.group(1)}
    tgt = {int(ch) for ch in match.group(2)}
    return src, tgt


def role_split_warnings(run: dict[str, Any], *, local_data_root: str, remote_data_root: str) -> list[str]:
    """Return warnings when a run conflicts with encoded role-aware split names.

    Example: if data_root contains ``c12src_c345tgt`` then C1/C2 should be used
    as source clients and C3/C4/C5 should be used as target clients. Runs such as
    C345 -> C1 need their own role-aware split root and should not reuse this one.
    """
    warnings: list[str] = []
    run_src = {int(x) for x in run["source_clients"]}
    run_tgt = {int(x) for x in run["target_clients"]}
    for label, root in [("local_data_root", local_data_root), ("remote_data_root", remote_data_root)]:
        parsed = parse_role_tag(root)
        if not parsed:
            continue
        encoded_src, encoded_tgt = parsed
        bad_src = sorted(run_src - encoded_src)
        bad_tgt = sorted(run_tgt - encoded_tgt)
        if bad_src or bad_tgt:
            warnings.append(
                f"{label} appears role-aware with source={sorted(encoded_src)} target={sorted(encoded_tgt)}, "
                f"but run uses source={sorted(run_src)} target={sorted(run_tgt)}; "
                f"source_mismatch={bad_src}, target_mismatch={bad_tgt}. "
                "Generate/use a matching role-aware split root before running this command."
            )
    return warnings


def build_server_command(run: dict[str, Any], args: argparse.Namespace) -> str:
    source_clients = [int(item) for item in run["source_clients"]]
    target_clients = [int(item) for item in run["target_clients"]]
    output_dir = remote_join(args.remote_output_root, str(run["run_id"]))
    server_val_data = comma_remote_client_dirs(args.remote_data_root, source_clients)
    server_calib_data = comma_remote_client_dirs(args.remote_data_root, target_clients)

    parts = [
        "cd", q_posix(args.remote_project_dir), "&&",
        args.remote_python,
        "-m", "gaps_flower.server_app",
        "--server-address", q_posix(args.server_bind_address),
        "--rounds", str(int(run.get("rounds", 25))),
        "--min-clients", str(len(source_clients)),
        "--strategy", q_posix(str(run.get("strategy", "gaps"))),
        "--profile", q_posix(str(run.get("profile", "strong_cls"))),
        "--run-name", q_posix(str(run["run_id"])),
        "--output-dir", q_posix(output_dir),
        "--save-history", "true",
        "--use-domain-adapt", str(parse_bool(run.get("use_domain_adapt", True))).lower(),
        "--server-val-data", q_posix(server_val_data),
        "--server-calib-data", q_posix(server_calib_data),
        "--domain-adapt-steps", str(int(run.get("domain_adapt_steps", 30))),
        "--domain-adapt-warmup", str(int(run.get("domain_adapt_warmup", 3))),
        "--da-use-coral", str(parse_bool(run.get("da_use_coral", True))).lower(),
        "--da-use-mmd", str(parse_bool(run.get("da_use_mmd", True))).lower(),
        "--da-use-adversarial", str(parse_bool(run.get("da_use_adversarial", False))).lower(),
        "--da-coral-class-conditional", str(parse_bool(run.get("da_coral_class_conditional", True))).lower(),
        "--strict-calibration-split", str(parse_bool(run.get("strict_calibration_split", True))).lower(),
        "--da-device", q_posix(str(run.get("da_device", "cpu"))),
        "--use-adapted-as-global", str(parse_bool(run.get("use_adapted_as_global", True))).lower(),
        "--da-lambda-coral", str(float(run.get("da_lambda_coral", 0.1))),
        "--da-lambda-global-mmd", str(float(run.get("da_lambda_global_mmd", 0.5))),
        "--da-lambda-class-mmd", str(float(run.get("da_lambda_class_mmd", 0.5))),
        "--da-lambda-proto-anchor", str(float(run.get("da_lambda_proto_anchor", 0.3))),
        "--da-lambda-proto", str(float(run.get("da_lambda_proto", 0.05))),
        "--da-lambda-consistency", str(float(run.get("da_lambda_consistency", 2.0))),
        "--da-lambda-residual", str(float(run.get("da_lambda_residual", 0.1))),
        "--da-lambda-proto-mmd", str(float(run.get("da_lambda_proto_mmd", 0.2))),
        "--da-lambda-stage-mmd", str(float(run.get("da_lambda_stage_mmd", 0.2))),
        "--da-server-opt-lr", str(float(run.get("da_server_opt_lr", 1e-4))),
    ]
    return " ".join(parts)


def build_client_command(run: dict[str, Any], args: argparse.Namespace, client_id: int) -> list[str]:
    return [
        str(args.local_python),
        "-m", "gaps_flower.client_app",
        "--server-address", str(args.server_public_address),
        "--client-id", str(int(client_id)),
        "--data-root", str(args.local_data_root),
        "--device", str(args.local_device),
        "--local-epochs", str(int(run.get("local_epochs", 5))),
        "--batch-size", str(int(run.get("batch_size", 32))),
        "--profile", str(run.get("profile", "strong_cls")),
    ]


def client_command_text(cmd: list[str], *, shell: str) -> str:
    if shell == "windows_cmd":
        return " ".join(q_windows_cmd(part) for part in cmd)
    if shell == "posix":
        return " ".join(q_posix(part) for part in cmd)
    return subprocess.list2cmdline(cmd)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def emit_run(run: dict[str, Any], args: argparse.Namespace) -> None:
    run_id = str(run["run_id"])
    out = Path(args.output_dir) / run_id
    server_cmd = build_server_command(run, args)
    client_cmds = {
        f"C{int(client)}": build_client_command(run, args, int(client))
        for client in run["source_clients"]
    }
    warnings = role_split_warnings(run, local_data_root=args.local_data_root, remote_data_root=args.remote_data_root)
    manifest = {
        "run_id": run_id,
        "source_clients": [int(item) for item in run["source_clients"]],
        "target_clients": [int(item) for item in run["target_clients"]],
        "purpose": run.get("purpose", ""),
        "server_command": server_cmd,
        "client_commands": {key: value for key, value in client_cmds.items()},
        "remote_output_dir": remote_join(args.remote_output_root, run_id),
        "warnings": warnings,
        "expected_outputs": [
            "history.json",
            "server_latest.pth",
            "server_latest_adapted.pth when DA is enabled and warmup has passed",
            "client_stats_round_*.json",
            "prototype_stats_round_*.json"
        ]
    }
    write_json(out / "command_manifest.json", manifest)
    write_text(out / "server_command.sh", "#!/usr/bin/env bash\nset -e\n" + server_cmd + "\n")

    client_lines = ["@echo off", "REM Start each client in a separate terminal if desired."]
    for warning in warnings:
        client_lines.append(f"REM WARNING: {warning}")
    for name, cmd in client_cmds.items():
        client_lines.append(f"start \"{run_id}_{name}\" cmd /k {client_command_text(cmd, shell='windows_cmd')}")
    write_text(out / "local_clients_windows.bat", "\n".join(client_lines) + "\n")
    write_text(out / "local_clients_commands.txt", "\n".join(client_command_text(cmd, shell="posix") for cmd in client_cmds.values()) + "\n")
    if warnings:
        print(f"[WARN] {run_id}:")
        for warning in warnings:
            print(f"  - {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Flower matrix server/client commands.")
    parser.add_argument("--matrix-config", default="configs/source_target_matrix_20260627.json")
    parser.add_argument("--runs", default="all", help="Comma-separated run IDs or all")
    parser.add_argument("--output-dir", default="results/source_target_classification_matrix_20260627_commands")
    parser.add_argument("--remote-project-dir", required=True)
    parser.add_argument("--remote-python", default="source ~/gaps_env/bin/activate && python")
    parser.add_argument("--remote-data-root", required=True)
    parser.add_argument("--remote-output-root", default="results/source_target_classification_matrix_20260627")
    parser.add_argument("--server-bind-address", default="0.0.0.0:8080")
    parser.add_argument("--server-public-address", required=True)
    parser.add_argument("--local-python", default="python")
    parser.add_argument("--local-data-root", required=True)
    parser.add_argument("--local-device", default="cuda")
    args = parser.parse_args()

    _defaults, runs = load_matrix(args.matrix_config)
    selected = select_runs(runs, args.runs)
    for run in selected:
        emit_run(run, args)
    write_json(Path(args.output_dir) / "matrix_command_index.json", {"runs": [str(run["run_id"]) for run in selected]})
    print(f"Generated commands for {len(selected)} run(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
