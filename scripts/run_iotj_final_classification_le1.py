"""Frozen, resumable orchestration for the IoT-J final classification suite."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / "docs/experiments/iotj_final_classification_le1_20260804"
RESULT_ROOT = REPO_ROOT / "results/iotj_final_classification_le1_20260804"
MATRIX_PATH = DOC_ROOT / "EXPERIMENT_MATRIX.csv"
PROTOCOL_PATH = DOC_ROOT / "PROTOCOL.md"
TARGET_POLICY_PATH = DOC_ROOT / "TARGET_INFORMATION_POLICY.md"
FORMAL_LOCK = RESULT_ROOT / "formal_training_started.lock"
P0A_SOURCE_CHECKPOINT = (
    REPO_ROOT.parent
    / "iotj-confirmation-observability/results/iotj_p0_routing_simplification_20260803"
    / "P0A_PURE_FEDAVG_LE1_S42/remote_server/server_round_025.pth"
)
IMPORTED_CHECKPOINT = RESULT_ROOT / "inputs/P0A_round25_source.pth"
INPUT_MANIFEST = RESULT_ROOT / "inputs/P0A_import_manifest.json"
LOCAL_DATA_ROOT = (
    REPO_ROOT.parents[1]
    / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
)
SEED = 42
ROUNDS = 25
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 5e-4
REMOTE_DATA_ROOT = "/root/GAPS/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
PI_DATA_ROOT = "/home/gaps/GAPS/flower_runtime/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
C2_DATA_ROOT = "/root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"

NEW_FULL_FL_IDS = frozenset(
    {
        "FCL-E1-FEDPROX",
        "FCL-E1-SCAFFOLD",
        "FCL-E3-GAPS-C3",
        "FCL-E3-GAPS-C4",
        "FCL-E3-GAPS-C5",
        "FCL-E4-A1",
        "FCL-E4-A2",
        "FCL-E4-A3",
        "FCL-E4-A4",
        "FCL-E4-A5",
    }
)


def load_registered_matrix(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("experiment_id") for row in rows):
        raise RuntimeError("FAIL_CLOSED invalid or empty experiment matrix")
    if len({row["experiment_id"] for row in rows}) != len(rows):
        raise RuntimeError("FAIL_CLOSED duplicate experiment_id")
    return rows


def execution_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    values = list(rows)
    ids = {row["experiment_id"] for row in values}
    return {
        "registered_configs": len(values),
        "new_full_fl_runs": len(ids & NEW_FULL_FL_IDS),
        "e2_adaptation_branches": sum(item.startswith("FCL-E2-") for item in ids),
    }


def _target_from_id(experiment_id: str) -> str:
    suffix = str(experiment_id).rsplit("-", 1)[-1].upper()
    return suffix if suffix in {"C3", "C4", "C5"} else "C5"


def _gaps_server_args(experiment_id: str, target: str) -> tuple[list[str], str]:
    profile = {
        "FCL-E4-A1": "proto_only",
        "FCL-E4-A2": "proto_replay",
        "FCL-E4-A3": "proto_replay",
        "FCL-E4-A4": "ce_stats",
        "FCL-E4-A5": "proto_replay",
    }.get(experiment_id, "proto_replay")
    selective = experiment_id.startswith("FCL-E3-") or experiment_id == "FCL-E4-A3"
    domain_adapt = experiment_id.startswith("FCL-E3-") or experiment_id in {
        "FCL-E4-A4",
        "FCL-E4-A5",
    }
    variant = "A6" if experiment_id == "FCL-E3-GAPS-C5" else experiment_id.rsplit("-", 1)[-1]
    target_information_method = (
        experiment_id.rsplit("-", 1)[-1].lower()
        if experiment_id in {"FCL-E4-A4", "FCL-E4-A5"}
        else "gaps"
    )
    args = [
        "--strategy", "gaps",
        "--profile", profile,
        "--ablation-variant", variant,
        "--target-information-method", target_information_method,
        "--use-selective-agg", str(selective).lower(),
        "--selective-warmup", "5",
        "--require-selective-after-warmup", str(selective).lower(),
        "--selective-min-scale", "0.3",
        "--use-proto-mmd", "true",
        "--use-domain-adapt", str(domain_adapt).lower(),
    ]
    if domain_adapt:
        args.extend(
            [
                "--server-val-data",
                f"{REMOTE_DATA_ROOT}/client_1,{REMOTE_DATA_ROOT}/client_2",
                "--server-calib-data",
                f"{REMOTE_DATA_ROOT}/client_{target[1:]}",
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
                "--da-lambda-proto-mmd", "0.2",
                "--da-lambda-stage-mmd", "0.2",
                "--da-server-opt-lr", "0.0005",
            ]
        )
    return args, profile


def build_flower_commands(experiment_id: str) -> dict:
    experiment_id = str(experiment_id)
    if experiment_id not in NEW_FULL_FL_IDS:
        raise ValueError(f"not a registered new full FL run: {experiment_id}")
    target = _target_from_id(experiment_id)
    remote_output = f"/root/GAPS/iotj_final_classification_le1_20260804/{experiment_id}"
    server = [
        "/root/gaps_env/bin/python", "-m", "gaps_flower.server_app",
        "--server-address", "0.0.0.0:8080",
        "--rounds", str(ROUNDS),
        "--min-clients", "2",
        "--seed", str(SEED),
        "--run-name", experiment_id,
        "--output-dir", remote_output,
        "--save-history", "true",
    ]
    profile = "ce_only"
    optimizer = "Adam"
    optimizer_note = "frozen GAPS experimental protocol"
    client_extra: list[str] = []
    if experiment_id == "FCL-E1-FEDPROX":
        server.extend(["--strategy", "fedavg", "--profile", profile])
        client_extra.extend(["--proximal-mu", "0.01"])
    elif experiment_id == "FCL-E1-SCAFFOLD":
        server.extend(
            ["--strategy", "scaffold", "--profile", profile, "--scaffold-lr", "0.0005"]
        )
        client_extra.extend(["--optimizer", "scaffold_sgd"])
        optimizer = "SGD"
        optimizer_note = "canonical SCAFFOLD implementation"
    else:
        gaps_args, profile = _gaps_server_args(experiment_id, target)
        server.extend(gaps_args)
        optimizer_note = "proposed method protocol"

    def client(python: str, client_id: int, data_root: str) -> list[str]:
        return [
            python, "-m", "gaps_flower.client_app",
            "--server-address", "127.0.0.1:18080",
            "--client-id", str(client_id),
            "--data-root", data_root,
            "--device", "cpu",
            "--local-epochs", str(LOCAL_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--profile", profile,
            "--seed", str(SEED),
            *client_extra,
        ]

    return {
        "server": server,
        "client_c1": client(
            "/home/gaps/GAPS/gaps_rpi_env/bin/python", 1, PI_DATA_ROOT
        ),
        "client_c2": client(
            "/root/gaps_c2_cpu_env/bin/python", 2, C2_DATA_ROOT
        ),
        "protocol": {
            "optimizer": optimizer,
            "optimizer_lr": LR,
            "optimizer_note": optimizer_note,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "target": target,
            "target_test_selection": False,
        },
    }


def build_e2_spec(experiment_id: str) -> dict:
    parts = str(experiment_id).upper().split("-")
    if len(parts) != 4 or parts[:2] != ["FCL", "E2"]:
        raise ValueError(f"invalid E2 id: {experiment_id}")
    method, target = parts[2], parts[3]
    if method not in {"CORAL", "MMD", "DANN"} or target not in {"C3", "C4", "C5"}:
        raise ValueError(f"invalid E2 id: {experiment_id}")
    return {
        "experiment_id": str(experiment_id).upper(),
        "method": method.lower(),
        "target": target,
        "source_checkpoint_role": "P0A_round25",
        "target_fields": ["x"],
        "steps": 100,
        "optimizer": "Adam",
        "optimizer_lr": LR,
        "coefficient": 0.5,
        "source_batch_convention": "combined_registered_C1_C2_calibration",
        "target_ce": False,
        "conditional": False,
        "pseudo_labels": False,
        "checkpoint_selection": "fixed_step_100",
        "hyperparameter_search": False,
        "seed": SEED,
    }


def protocol_freeze_hash(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda p: str(p))
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def start_formal_lock(
    path: str | Path, *, digest: str, freeze_commit: str
) -> Path:
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        raise FileExistsError(f"formal lock already exists: {lock}")
    payload = {
        "schema_version": "iotj.final_classification.formal_lock.v1",
        "protocol_hash": str(digest),
        "freeze_commit": str(freeze_commit),
        "seed": SEED,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mutation_policy": "registered matrix and protocol are immutable",
    }
    lock.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


def assert_formal_lock_matches(path: str | Path, *, digest: str) -> dict:
    lock = Path(path)
    if not lock.is_file():
        raise RuntimeError("FAIL_CLOSED formal training lock missing")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != str(digest):
        raise RuntimeError("FAIL_CLOSED protocol hash mismatch after formal start")
    return payload


def write_completion_marker(
    run_dir: str | Path,
    *,
    experiment_id: str,
    protocol_hash: str,
    endpoint: dict,
) -> Path:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "fixed_endpoint_complete.json"
    if marker.exists():
        raise FileExistsError(f"immutable completion marker already exists: {marker}")
    payload = {
        "schema_version": "iotj.final_classification.completion.v1",
        "experiment_id": str(experiment_id),
        "protocol_hash": str(protocol_hash),
        "fixed_endpoint": dict(endpoint),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_test_opened": False,
    }
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return marker


def experiment_resume_status(
    run_dir: str | Path, *, expected_protocol_hash: str
) -> str:
    marker = Path(run_dir) / "fixed_endpoint_complete.json"
    if not marker.exists():
        return "pending"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != str(expected_protocol_hash):
        raise RuntimeError("FAIL_CLOSED completed run protocol hash mismatch")
    return "complete"


def current_protocol_hash() -> str:
    return protocol_freeze_hash([MATRIX_PATH, PROTOCOL_PATH, TARGET_POLICY_PATH])


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _ssh(host: str, command: str, *, timeout: float = 120.0) -> str:
    return _run(
        ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        timeout=timeout,
    ).stdout


def _source_archive(freeze_commit: str) -> tuple[Path, str]:
    archive = RESULT_ROOT / "inputs" / f"source_{freeze_commit[:12]}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        _run(
            ["git", "archive", "--format=tar", "-o", str(archive), freeze_commit],
            timeout=300,
        )
    return archive, _sha256_file(archive)


def _deploy_archive(host: str, archive: Path, runtime: str) -> None:
    digest = _sha256_file(archive)
    marker = f"{runtime}/.source_archive_sha256"
    observed = _ssh(
        host,
        f"if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi",
    ).strip()
    if observed:
        if observed != digest:
            raise RuntimeError(f"FAIL_CLOSED runtime hash mismatch on {host}")
        return
    partial = _ssh(
        host, f"if test -e {shlex.quote(runtime)}; then echo PARTIAL; fi"
    ).strip()
    if partial:
        raise RuntimeError(f"FAIL_CLOSED partial unhashed runtime exists on {host}: {runtime}")
    remote_archive = f"/tmp/iotj_final_{digest}.tar"
    _run(["scp", "-p", str(archive), f"{host}:{remote_archive}"], timeout=600)
    _ssh(
        host,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(runtime)}",
                f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(runtime)}",
                f"printf '%s\\n' {shlex.quote(digest)} > {shlex.quote(marker)}",
            ]
        ),
        timeout=600,
    )


def _process_command(host: str, runtime: str, argv: list[str]) -> list[str]:
    remote = f"cd {shlex.quote(runtime)} && exec {shlex.join(argv)}"
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, remote]


def _ensure_idle(hosts: list[str]) -> None:
    for host in hosts:
        output = _ssh(host, "pgrep -af 'gaps_flower.(server_app|client_app)' || true")
        if output.strip():
            raise RuntimeError(f"FAIL_CLOSED residual Flower process on {host}:\n{output}")


def execute_full_fl(
    experiment_id: str,
    *,
    protocol_hash: str,
    lock_payload: dict,
    ecs_host: str,
    pi_host: str,
    c2_host: str,
    timeout_hours: float,
) -> Path:
    from scripts.run_iotj_confirmation_observability import (
        _start_ecs_c2_tunnels,
        _terminate_processes,
    )

    commands = build_flower_commands(experiment_id)
    run_dir = RESULT_ROOT / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED incomplete run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "locked_run_spec.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive, archive_hash = _source_archive(str(lock_payload["freeze_commit"]))
    short_hash = archive_hash[:12]
    runtimes = {
        "ecs": f"/root/GAPS/iotj_final_runtime/{short_hash}",
        "pi": f"/home/gaps/GAPS/iotj_final_runtime/{short_hash}",
        "c2": f"/root/GAPS/iotj_final_runtime/{short_hash}",
    }
    hosts = [ecs_host, pi_host, c2_host]
    _ensure_idle(hosts)
    remote_output = f"/root/GAPS/iotj_final_classification_le1_20260804/{experiment_id}"
    if _ssh(
        ecs_host,
        f"if test -e {shlex.quote(remote_output)}; then echo EXISTS; fi",
    ).strip():
        raise FileExistsError(f"FAIL_CLOSED remote formal output exists: {remote_output}")
    for host, key in zip(hosts, ("ecs", "pi", "c2")):
        _deploy_archive(host, archive, runtimes[key])

    processes: list[subprocess.Popen] = []
    tunnels: list[subprocess.Popen] = []
    handles = []
    started = time.perf_counter()
    try:
        for role in ("server", "client_c1", "client_c2"):
            handles.extend(
                [
                    (run_dir / f"{role}.stdout.log").open("w", encoding="utf-8"),
                    (run_dir / f"{role}.stderr.log").open("w", encoding="utf-8"),
                ]
            )
        processes.append(
            subprocess.Popen(
                _process_command(ecs_host, runtimes["ecs"], commands["server"]),
                stdout=handles[0], stderr=handles[1], text=True,
            )
        )
        time.sleep(5)
        if processes[0].poll() is not None:
            raise RuntimeError(f"server exited before clients: rc={processes[0].returncode}")
        tunnels = list(_start_ecs_c2_tunnels(ecs_host, pi_host, c2_host))
        processes.extend(
            [
                subprocess.Popen(
                    _process_command(pi_host, runtimes["pi"], commands["client_c1"]),
                    stdout=handles[2], stderr=handles[3], text=True,
                ),
                subprocess.Popen(
                    _process_command(c2_host, runtimes["c2"], commands["client_c2"]),
                    stdout=handles[4], stderr=handles[5], text=True,
                ),
            ]
        )
        deadline = time.monotonic() + float(timeout_hours) * 3600.0
        while any(process.poll() is None for process in processes):
            if time.monotonic() > deadline:
                raise TimeoutError(f"distributed run exceeded {timeout_hours} hours")
            for role, process in zip(("server", "client_c1", "client_c2"), processes):
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{role} failed with rc={process.returncode}")
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(
                f"non-zero process return codes: {[process.returncode for process in processes]}"
            )
    finally:
        _terminate_processes(processes)
        _terminate_processes(tunnels)
        for handle in handles:
            handle.close()
    wall_seconds = time.perf_counter() - started
    remote_copy = run_dir / "remote_server"
    _run(["scp", "-r", f"{ecs_host}:{remote_output}", str(remote_copy)], timeout=1200)
    checkpoint_name = (
        "server_latest_adapted.pth"
        if commands["protocol"]["optimizer_note"] == "proposed method protocol"
        and "--use-domain-adapt" in commands["server"]
        and commands["server"][commands["server"].index("--use-domain-adapt") + 1] == "true"
        else "server_latest.pth"
    )
    checkpoint = remote_copy / checkpoint_name
    if not checkpoint.is_file():
        raise RuntimeError(f"FAIL_CLOSED round25 checkpoint missing: {checkpoint}")
    manifest = {
        "schema_version": "iotj.final_classification.full_fl.v1",
        "experiment_id": experiment_id,
        "protocol_hash": protocol_hash,
        "source_archive_sha256": archive_hash,
        "wall_seconds": wall_seconds,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "target_test_opened": False,
        "protocol": commands["protocol"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return write_completion_marker(
        run_dir,
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        endpoint={"round": 25, "checkpoint": checkpoint_name},
    )


def execute_e2(experiment_id: str, *, protocol_hash: str) -> Path:
    import csv as csv_module
    import torch

    from federated_dataset import create_merged_calibration_loader
    from gaps_flower.canonical_uda import run_canonical_uda
    from gaps_flower.evaluate_checkpoint import load_checkpoint_model
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
    from gaps_flower.target_information import TargetAccessLedger, load_target_calibration_x

    spec = build_e2_spec(experiment_id)
    run_dir = RESULT_ROOT / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED incomplete run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "locked_run_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _checkpoint = load_checkpoint_model(
        str(IMPORTED_CHECKPOINT), device, BATCH_SIZE
    )
    expected = ordered_state_content_fingerprint(model.state_dict())
    source_loader = create_merged_calibration_loader(
        [LOCAL_DATA_ROOT / "client_1", LOCAL_DATA_ROOT / "client_2"],
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    ledger = TargetAccessLedger(run_dir / "target_access_ledger.jsonl")
    target_client = int(spec["target"][1:])
    target_loader = load_target_calibration_x(
        LOCAL_DATA_ROOT / f"client_{target_client}",
        method=spec["method"],
        ledger=ledger,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    adapted, diagnostics, seconds = run_canonical_uda(
        spec["method"],
        model,
        source_loader,
        target_loader,
        device,
        num_steps=100,
        model_lr=LR,
        alignment_weight=0.5,
        expected_source_fingerprint=expected,
        seed=SEED,
        formal=True,
    )
    checkpoint_path = run_dir / "adapted_step_100.pth"
    torch.save(
        {
            "model_state": adapted.state_dict(),
            "source_checkpoint_ordered_fingerprint": expected,
            "experiment_id": experiment_id,
            "step": 100,
        },
        checkpoint_path,
    )
    with (run_dir / "adaptation_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "iotj.final_classification.e2.v1",
                "experiment_id": experiment_id,
                "protocol_hash": protocol_hash,
                "spec": spec,
                "adaptation_seconds": seconds,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "target_test_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return write_completion_marker(
        run_dir,
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        endpoint={"steps": 100, "checkpoint": checkpoint_path.name},
    )


def execute_diagnostic_or_reuse(experiment_id: str, *, protocol_hash: str) -> Path:
    import numpy as np

    from scripts.evaluate_iotj_final_classification_le1 import (
        _write_csv,
        sensor_channel_shift_rows,
        sensor_covariance_diagnostics,
    )

    run_dir = RESULT_ROOT / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED incomplete run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    if experiment_id == "FCL-E0-SHIFT":
        source = np.concatenate(
            [
                np.load(LOCAL_DATA_ROOT / f"client_{client}/calibration_features.npy", allow_pickle=False)
                for client in (1, 2)
            ],
            axis=0,
        )
        channel_rows = []
        covariance_rows = []
        for target in (3, 4, 5):
            values = np.load(
                LOCAL_DATA_ROOT / f"client_{target}/calibration_features.npy",
                allow_pickle=False,
            )
            channel_rows.extend(
                sensor_channel_shift_rows(source, values, target_id=f"C{target}")
            )
            covariance_rows.append(
                sensor_covariance_diagnostics(source, values, target_id=f"C{target}")
            )
        _write_csv(run_dir / "sensor_channel_shift.csv", channel_rows)
        _write_csv(run_dir / "sensor_covariance_shift.csv", covariance_rows)
        endpoint = {"targets": ["C3", "C4", "C5"], "split": "calibration_x_only"}
    elif experiment_id == "FCL-E1-FEDAVG":
        source_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
        (run_dir / "reuse_manifest.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        endpoint = {
            "round": 25,
            "checkpoint": str(IMPORTED_CHECKPOINT),
            "reuse": "P0A_ordered_content_verified",
        }
    else:
        raise ValueError(f"unsupported diagnostic/reuse id: {experiment_id}")
    return write_completion_marker(
        run_dir,
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        endpoint=endpoint,
    )


def _write_registered_specs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_registered_matrix(MATRIX_PATH)
    payload = {
        "protocol_hash": current_protocol_hash(),
        "counts": execution_counts(rows),
        "e2": [
            build_e2_spec(row["experiment_id"])
            for row in rows
            if row["experiment_id"].startswith("FCL-E2-")
        ],
        "full_fl_commands": {
            experiment_id: build_flower_commands(experiment_id)
            for experiment_id in sorted(NEW_FULL_FL_IDS)
        },
    }
    (RESULT_ROOT / "locked_execution_specs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_inputs() -> dict:
    from scripts.prepare_iotj_final_classification_inputs import import_checkpoint

    _write_registered_specs()
    manifest = import_checkpoint(P0A_SOURCE_CHECKPOINT, IMPORTED_CHECKPOINT)
    INPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    INPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_scaffold_source_gate() -> dict:
    """Run the one-shot preregistered C1/C2 numerical gate; never tune."""
    import numpy as np
    import torch

    from federated_dataset import create_train_loader
    from gaps_flower.scaffold import ScaffoldClientControlState, _zero_control
    from gaps_flower.source_numerical_gate import evaluate_source_gate
    from gaps_flower.task import create_model, make_config

    if not LOCAL_DATA_ROOT.is_dir():
        raise RuntimeError(f"FAIL_CLOSED source dataset missing: {LOCAL_DATA_ROOT}")
    config = make_config(
        device="cpu",
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        profile="ce_only",
        seed=SEED,
    )
    base_model = create_model(config)
    base_state = copy.deepcopy(base_model.state_dict())
    diagnostics = []
    total_correct = 0.0
    total_examples = 0
    class_counts: dict[int, int] = {}
    client_rows = []
    for client_id in (1, 2):
        torch.manual_seed(SEED)
        loader = create_train_loader(
            LOCAL_DATA_ROOT / f"client_{client_id}",
            batch_size=BATCH_SIZE,
            shuffle=True,
            normalize=False,
            num_workers=0,
        )
        model = create_model(config)
        model.load_state_dict(base_state, strict=True)
        state = ScaffoldClientControlState.from_model(model)
        result = state.train(
            model,
            loader,
            server_control=_zero_control(model),
            lr=LR,
            local_epochs=LOCAL_EPOCHS,
            device=torch.device("cpu"),
        )
        diagnostics.append(
            {
                "ce_trajectory": result.ce_trajectory,
                "grad_norms": result.grad_norms,
                "parameter_norms": result.parameter_norms,
            }
        )
        examples = len(loader.dataset)
        total_correct += result.train_accuracy * examples
        total_examples += examples
        labels = np.load(
            LOCAL_DATA_ROOT / f"client_{client_id}/train_classification_labels.npy",
            allow_pickle=False,
        ).reshape(-1)
        for class_id, count in zip(*np.unique(labels.astype(int), return_counts=True)):
            class_counts[int(class_id)] = class_counts.get(int(class_id), 0) + int(count)
        client_rows.append(
            {
                "client_id": client_id,
                "examples": examples,
                "steps": result.steps,
                "train_accuracy": result.train_accuracy,
                "optimizer": result.optimizer_name,
                "optimizer_lr": result.optimizer_lr,
                "adam_state_present": result.adam_state_present,
            }
        )
    verdict = evaluate_source_gate(
        diagnostics,
        source_accuracy=total_correct / max(total_examples, 1),
        source_class_counts=class_counts,
    )
    payload = {
        "schema_version": "iotj.final_classification.scaffold_source_gate.v1",
        "passed": verdict.passed,
        "checks": verdict.checks,
        "diagnostics": verdict.diagnostics,
        "action": verdict.action,
        "lr_search_performed": verdict.lr_search_performed,
        "target_information_accessed": verdict.target_information_accessed,
        "protocol": {
            "clients": ["C1", "C2"],
            "optimizer": "SGD",
            "optimizer_lr": LR,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "clients": client_rows,
    }
    output = RESULT_ROOT / "preflight/scaffold_source_numerical_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verdict.passed:
        raise RuntimeError("FAIL_CLOSED SCAFFOLD source-only numerical gate failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "source-gate", "preflight", "start-formal", "evaluate", "analyze", "audit"):
        subparsers.add_parser(command)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment-id", required=True)
    run_parser.add_argument("--ecs-host", default="root@121.40.139.213")
    run_parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    run_parser.add_argument("--c2-host", default="root@114.55.171.63")
    run_parser.add_argument("--timeout-hours", type=float, default=10.0)
    args = parser.parse_args()

    if args.command == "prepare":
        manifest = prepare_inputs()
        print(json.dumps({
            "status": "PASS",
            "protocol_hash": current_protocol_hash(),
            "ordered_checkpoint_fingerprint": manifest["copy"]["ordered_state_content_fingerprint"],
        }))
        return
    if args.command == "source-gate":
        payload = run_scaffold_source_gate()
        print(json.dumps({"status": "PASS", "checks": payload["checks"]}))
        return
    if args.command == "start-formal":
        _write_registered_specs()
        lock = start_formal_lock(
            FORMAL_LOCK, digest=current_protocol_hash(), freeze_commit=_git_head()
        )
        print(json.dumps({"status": "STARTED", "lock": str(lock)}))
        return
    if args.command == "run":
        digest = current_protocol_hash()
        assert_formal_lock_matches(FORMAL_LOCK, digest=digest)
        experiment_id = str(args.experiment_id).upper()
        run_dir = RESULT_ROOT / experiment_id
        status = experiment_resume_status(
            run_dir, expected_protocol_hash=digest
        )
        if status == "complete":
            print(json.dumps({"status": "SKIP_COMPLETE", "experiment_id": experiment_id}))
            return
        if experiment_id in NEW_FULL_FL_IDS:
            marker = execute_full_fl(
                experiment_id,
                protocol_hash=digest,
                lock_payload=assert_formal_lock_matches(FORMAL_LOCK, digest=digest),
                ecs_host=args.ecs_host,
                pi_host=args.pi_host,
                c2_host=args.c2_host,
                timeout_hours=args.timeout_hours,
            )
        elif experiment_id.startswith("FCL-E2-"):
            marker = execute_e2(experiment_id, protocol_hash=digest)
        elif experiment_id in {"FCL-E0-SHIFT", "FCL-E1-FEDAVG"}:
            marker = execute_diagnostic_or_reuse(
                experiment_id, protocol_hash=digest
            )
        else:
            raise ValueError(
                f"registered row is reuse-only through another endpoint: {experiment_id}"
            )
        print(json.dumps({"status": "COMPLETE", "experiment_id": experiment_id, "marker": str(marker)}))
        return
    module = "scripts.audit_iotj_final_classification_le1"
    if args.command in {"preflight", "audit"}:
        stage = "pre-run" if args.command == "preflight" else "post-run"
        subprocess.run(
            [sys.executable, "-m", module, "--stage", stage, "--strict"],
            cwd=REPO_ROOT,
            check=True,
        )
        return
    raise RuntimeError(f"{args.command} backend is not yet available")


if __name__ == "__main__":
    main()
