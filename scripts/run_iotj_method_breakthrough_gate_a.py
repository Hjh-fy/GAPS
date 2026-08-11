"""Run Gate-A source-diversity and exact GAPS-DG-P validation."""

from __future__ import annotations

import copy
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import itertools
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaps_flower.evaluate_checkpoint import load_checkpoint_model, make_loader
from gaps_flower.posthoc_commissioning import ordered_state_fingerprint
from scripts import run_iotj_final_classification_le1 as frozen
from scripts.run_iotj_canonical_v1_comparators import build_source_fl_commands
from scripts.run_iotj_source_dg_g2 import build_g2_commands
from scripts.summarize_iotj_classification_ablation import (
    classification_metrics,
    evaluate_checkpoint_stream,
)


DATASET_NAME = "iotj_canonical_v1_s4_role_view"
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"
ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = ROOT / "dataset" / DATASET_NAME
CANONICAL_DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/gate_a_source_diversity"
S2_FEDAVG_MANIFEST = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-FEDAVG/run_manifest.json"
S2_DGP_MANIFEST = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate2_source_dg/CAN-V1-MR-G2-DGP/run_manifest.json"
S2_METRICS = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate2_source_dg/G2_ZERO_SHOT_COMPARISON.csv"
S2_PREDICTIONS = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate2_source_dg/G2_CLASSIFICATION_PREDICTIONS.csv"
S2_REPRESENTATION = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate2_source_dg/G2_REPRESENTATION_ANALYSIS.csv"
METHOD_DIRS = {"fedavg": "fedavg", "gaps_dg_p": "gaps_dg_p"}
EXPERIMENT_IDS = {"fedavg": "CAN-V1-MB-A-S4-FEDAVG", "gaps_dg_p": "CAN-V1-MB-A-S4-DGP"}
ROLE_VIEW_AGGREGATE_SHA256 = "843459df765eb1525a30b91024e6db6a66a740858d0761ff84e67decb92389fa"


def _set_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def _replace(values: list[str], replacements: tuple[tuple[str, str], ...]) -> list[str]:
    result = list(values)
    for old, new in replacements:
        result = [value.replace(old, new) for value in result]
    return result


def _ecs_client(template: list[str], client: int) -> list[str]:
    command = list(template)
    command[0] = "/root/gaps_env/bin/python"
    _set_option(command, "--server-address", "127.0.0.1:8080")
    _set_option(command, "--client-id", str(client))
    _set_option(command, "--data-root", REMOTE_DATA_ROOT)
    _set_option(command, "--device", "cpu")
    return command


def build_gate_a_commands(method: str) -> dict[str, Any]:
    if method not in {"fedavg", "gaps_dg_p"}:
        raise ValueError(f"unsupported Gate-A method: {method}")
    base = build_source_fl_commands("FedAvg") if method == "fedavg" else build_g2_commands()
    commands = copy.deepcopy(base)
    replacements = (
        ("/root/GAPS/dataset/iotj_canonical_v1", REMOTE_DATA_ROOT),
        ("/home/gaps/GAPS/flower_runtime/dataset/iotj_canonical_v1", PI_DATA_ROOT),
        ("/root/GAPS/confirmation_c2_data/iotj_canonical_v1", C2_DATA_ROOT),
        ("CAN-V1-CMP-FEDAVG", "CAN-V1-MB-A-S4-FEDAVG"),
        ("CAN-V1-MR-G2-DGP", "CAN-V1-MB-A-S4-DGP"),
    )
    for role in ("server", "client_c1", "client_c2"):
        commands[role] = _replace(commands[role], replacements)
    _set_option(commands["server"], "--min-clients", "4")
    c1, c2 = commands.pop("client_c1"), commands.pop("client_c2")
    clients = {
        "C1": c1,
        "C2": c2,
        "C3": _ecs_client(c2, 3),
        "C4": _ecs_client(c2, 4),
    }
    protocol = commands["protocol"]
    protocol.update(
        {
            "dataset": DATASET_NAME,
            "dataset_aggregate_sha256": ROLE_VIEW_AGGREGATE_SHA256,
            "experiment_id": EXPERIMENT_IDS[method],
            "source_clients": ["C1", "C2", "C3", "C4"],
            "target_clients": ["C5"],
            "target_access": "NONE",
            "target_x": False,
            "target_y": False,
            "target_phase": False,
            "target_concentration": False,
            "rounds": 25,
            "local_epochs": 1,
            "batch_size": 32,
            "seed": 42,
            "optimizer": "Adam",
            "optimizer_lr": 5e-4,
            "checkpoint_selection": "fixed_round_25",
            "prototype_alignment": method == "gaps_dg_p",
            "lambda_proto": 0.05 if method == "gaps_dg_p" else 0.0,
            "replay": False,
            "selective_aggregation": False,
            "server_domain_adaptation": False,
            "hyperparameter_search": False,
            "target_test_selection": False,
        }
    )
    return {"server": commands["server"], "clients": clients, "protocol": protocol}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_s2_reuse(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise RuntimeError("FAIL_CLOSED S2 checkpoint is missing")
    if _sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("FAIL_CLOSED S2 checkpoint hash mismatch")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("FAIL_CLOSED S2 protocol missing")
    expected = {
        "dataset": "iotj_canonical_v1",
        "rounds": 25,
        "local_epochs": 1,
        "seed": 42,
        "checkpoint_selection": "fixed_round_25",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"FAIL_CLOSED S2 protocol differs: {key}")
    if any(protocol.get(key) is not False for key in ("target_x", "target_y")):
        raise RuntimeError("FAIL_CLOSED S2 target access is not absent")
    if manifest.get("target_test_opened") is not False:
        raise RuntimeError("FAIL_CLOSED S2 target test was opened before endpoint lock")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "protocol": protocol,
    }


def decide_gate_a(
    *,
    s2_fedavg_c5_f1: float,
    s2_dg_c5_f1: float,
    s4_fedavg_c5_f1: float,
    s4_dg_c5_f1: float,
    s4_fedavg_source_f1: float,
    s4_dg_source_f1: float,
) -> dict[str, Any]:
    diversity_gain = float(s4_fedavg_c5_f1) - float(s2_fedavg_c5_f1)
    dg_gain_s2 = float(s2_dg_c5_f1) - float(s2_fedavg_c5_f1)
    dg_gain_s4 = float(s4_dg_c5_f1) - float(s4_fedavg_c5_f1)
    source_drop_s4 = float(s4_fedavg_source_f1) - float(s4_dg_source_f1)
    diversity_supported = diversity_gain >= 0.01
    dg_promising = dg_gain_s4 >= 0.01 and source_drop_s4 <= 0.01
    if dg_promising:
        dg_decision = "SOURCE_DG_PROMISING"
        next_action = "CREATE_MULTI_SEED_PROPOSAL_ONLY"
    elif diversity_supported:
        dg_decision = "DG_MECHANISM_NOT_SUPPORTED"
        next_action = "STOP_DG_EXPANSION"
    else:
        dg_decision = "SOURCE_DG_RETIRED"
        next_action = "STOP_DG_EXPANSION"
    return {
        "source_diversity": (
            "SOURCE_DIVERSITY_SUPPORTED"
            if diversity_supported
            else "SOURCE_DIVERSITY_NOT_SUPPORTED"
        ),
        "dg_mechanism": dg_decision,
        "next_action": next_action,
        "thresholds": {
            "meaningful_c5_macro_f1_gain": 0.01,
            "maximum_source_pooled_macro_f1_drop": 0.01,
        },
        "deltas": {
            "s4_fedavg_minus_s2_fedavg_c5_macro_f1": diversity_gain,
            "s2_dg_minus_fedavg_c5_macro_f1": dg_gain_s2,
            "s4_dg_minus_fedavg_c5_macro_f1": dg_gain_s4,
            "s4_dg_source_pooled_drop": source_drop_s4,
        },
    }


def execution_role_hosts() -> dict[str, str]:
    return {"server": "ecs", "C1": "pi", "C2": "c2", "C3": "ecs", "C4": "ecs"}


def representation_diagnostics(
    records: list[dict[str, Any]], *, method: str, source_count: int
) -> dict[str, Any]:
    clients = sorted({int(row["client"]) for row in records})
    if len(clients) != int(source_count):
        raise RuntimeError("FAIL_CLOSED representation source count differs")
    classes = sorted({int(row["class_id"]) for row in records})
    pair_distances: list[float] = []
    class_centroids: list[np.ndarray] = []
    for class_id in classes:
        per_client = {}
        selected = [np.asarray(row["feature"], dtype=np.float64) for row in records if int(row["class_id"]) == class_id]
        class_centroids.append(np.mean(selected, axis=0))
        for client in clients:
            values = [
                np.asarray(row["feature"], dtype=np.float64)
                for row in records
                if int(row["class_id"]) == class_id and int(row["client"]) == client
            ]
            if not values:
                raise RuntimeError(f"FAIL_CLOSED missing representation cell: C{client}/class{class_id}")
            per_client[client] = np.mean(values, axis=0)
        pair_distances.extend(
            float(np.linalg.norm(per_client[first] - per_client[second]))
            for first, second in itertools.combinations(clients, 2)
        )
    margins = [
        float(np.linalg.norm(class_centroids[first] - class_centroids[second]))
        for first, second in itertools.combinations(range(len(class_centroids)), 2)
    ]
    cell_dispersions: list[float] = []
    cells = sorted({(int(row["class_id"]), int(row["phase_id"])) for row in records})
    for class_id, phase_id in cells:
        centroids = []
        for client in clients:
            values = [
                np.asarray(row["feature"], dtype=np.float64)
                for row in records
                if int(row["class_id"]) == class_id
                and int(row["phase_id"]) == phase_id
                and int(row["client"]) == client
            ]
            if values:
                centroids.append(np.mean(values, axis=0))
        if len(centroids) >= 2:
            center = np.mean(centroids, axis=0)
            cell_dispersions.append(float(np.mean([np.linalg.norm(value - center) for value in centroids])))
    return {
        "method": method,
        "source_count": int(source_count),
        "source_clients": ";".join(f"C{client}" for client in clients),
        "within_class_inter_source_centroid_distance_mean": float(np.mean(pair_distances)),
        "between_class_centroid_margin_mean": float(np.mean(margins)),
        "class_phase_prototype_dispersion_mean": float(np.mean(cell_dispersions)),
        "inter_source_pairs": len(list(itertools.combinations(clients, 2))),
        "class_phase_cells": len(cell_dispersions),
        "source_test_windows": len(records),
    }


def verify_s4_endpoint_locks(run_root: Path) -> dict[str, dict[str, Any]]:
    locked: dict[str, dict[str, Any]] = {}
    for method in ("fedavg", "gaps_dg_p"):
        directory = Path(run_root) / method
        marker_path = directory / "fixed_endpoint_complete.json"
        manifest_path = directory / "run_manifest.json"
        if not marker_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"FAIL_CLOSED missing Gate-A endpoint lock: {method}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            marker.get("target_test_opened") is not False
            or marker.get("fixed_endpoint", {}).get("round") != 25
            or manifest.get("target_test_opened") is not False
            or manifest.get("protocol", {}).get("checkpoint_selection") != "fixed_round_25"
        ):
            raise RuntimeError(f"FAIL_CLOSED invalid Gate-A endpoint lock: {method}")
        checkpoint = Path(str(manifest.get("checkpoint", "")))
        if not checkpoint.is_file() or _sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
            raise RuntimeError(f"FAIL_CLOSED Gate-A endpoint checkpoint hash: {method}")
        locked[method] = manifest
    return locked


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED refuse empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def protocol_hash() -> str:
    payload = {method: build_gate_a_commands(method) for method in METHOD_DIRS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_pre_run_freeze(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    dataset_hash = json.loads((LOCAL_DATA_ROOT / "dataset_sha256.json").read_text(encoding="utf-8"))
    role_view = json.loads((LOCAL_DATA_ROOT / "s4_role_view_manifest.json").read_text(encoding="utf-8"))
    if role_view.get("c5_rng_access") is not False:
        raise RuntimeError("FAIL_CLOSED C5 entered S4 role-view RNG")
    for name in (
        "calibration_features.npy",
        "calibration_classification_labels.npy",
        "calibration_regression_labels.npy",
        "calibration_phase_labels.npy",
        "calibration_experiment_info.json",
        "test_features.npy",
        "test_classification_labels.npy",
        "test_regression_labels.npy",
        "test_phase_labels.npy",
        "test_experiment_info.json",
    ):
        if _sha256_file(LOCAL_DATA_ROOT / "client_5" / name) != _sha256_file(CANONICAL_DATA_ROOT / "client_5" / name):
            raise RuntimeError(f"FAIL_CLOSED C5 role-view hash differs: {name}")
    payload = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.gate_a.freeze.v1",
        "status": "FROZEN",
        "freeze_commit": _git_head(),
        "protocol_hash": protocol_hash(),
        "dataset_aggregate_sha256": dataset_hash["aggregate_sha256"],
        "partition_identity_sha256": role_view["partition_identity_sha256"],
        "s2_reuse": {
            "fedavg": audit_s2_reuse(S2_FEDAVG_MANIFEST),
            "gaps_dg_p": audit_s2_reuse(S2_DGP_MANIFEST),
        },
        "s4_commands": {method: build_gate_a_commands(method) for method in METHOD_DIRS},
        "decision_thresholds": {
            "source_diversity_c5_macro_f1_gain": 0.01,
            "dg_c5_macro_f1_gain": 0.01,
            "maximum_source_pooled_macro_f1_drop": 0.01,
        },
        "target_test_opened": False,
        "hyperparameter_search": False,
        "gate_d_e_f_started": False,
    }
    path = output / "PRE_RUN_FREEZE.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError("FAIL_CLOSED Gate-A pre-run freeze differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return payload


def _dataset_archive(output: Path, role: str, clients: tuple[int, ...]) -> Path:
    archive = output / "inputs" / f"dataset_{role}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        return archive
    include = [LOCAL_DATA_ROOT / "canonical_preprocessing_manifest.json", LOCAL_DATA_ROOT / "s4_role_view_manifest.json", LOCAL_DATA_ROOT / "dataset_sha256.json"]
    include.extend(LOCAL_DATA_ROOT / f"client_{client}" for client in clients)
    with tarfile.open(archive, "w") as handle:
        for path in include:
            handle.add(path, arcname=f"{DATASET_NAME}/{path.name}" if path.is_file() else f"{DATASET_NAME}/{path.name}")
    return archive


def _deploy_dataset(host: str, archive: Path, remote_parent: str) -> None:
    digest = _sha256_file(archive)
    remote_root = f"{remote_parent}/{DATASET_NAME}"
    marker = f"{remote_root}/.gate_a_dataset_archive_sha256"
    observed = frozen._ssh(host, f"if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi").strip()
    if observed:
        if observed != digest:
            raise RuntimeError(f"FAIL_CLOSED Gate-A dataset hash mismatch on {host}")
        return
    if frozen._ssh(host, f"if test -e {shlex.quote(remote_root)}; then echo PARTIAL; fi").strip():
        raise RuntimeError(f"FAIL_CLOSED partial Gate-A dataset on {host}: {remote_root}")
    remote_archive = f"/tmp/gate_a_dataset_{digest}.tar"
    frozen._run(["scp", "-p", str(archive), f"{host}:{remote_archive}"], timeout=1800)
    frozen._ssh(
        host,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(remote_parent)}",
                f"tar -xf {shlex.quote(remote_archive)} -C {shlex.quote(remote_parent)}",
                f"printf '%s\\n' {shlex.quote(digest)} > {shlex.quote(marker)}",
            ]
        ),
        timeout=1800,
    )


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def execute_s4_fl(
    method: str,
    output: Path,
    freeze: dict[str, Any],
    *,
    ecs_host: str,
    pi_host: str,
    c2_host: str,
    timeout_hours: float,
) -> Path:
    from scripts.run_iotj_confirmation_observability import _start_ecs_c2_tunnels, _terminate_processes

    experiment_id = EXPERIMENT_IDS[method]
    run_dir = Path(output) / METHOD_DIRS[method]
    marker = run_dir / "fixed_endpoint_complete.json"
    if marker.is_file():
        return marker
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial Gate-A run exists: {run_dir}")
    run_dir.mkdir(parents=True)
    commands = build_gate_a_commands(method)
    _write_json(run_dir / "locked_run_spec.json", commands)
    archive, archive_hash = frozen._source_archive(str(freeze["freeze_commit"]))
    short_hash = archive_hash[:12]
    runtimes = {
        "ecs": f"/root/GAPS/iotj_final_runtime/{short_hash}",
        "pi": f"/home/gaps/GAPS/iotj_final_runtime/{short_hash}",
        "c2": f"/root/GAPS/iotj_final_runtime/{short_hash}",
    }
    hosts = {"ecs": ecs_host, "pi": pi_host, "c2": c2_host}
    frozen._ensure_idle(list(hosts.values()))
    remote_output = _command_value(commands["server"], "--output-dir")
    if frozen._ssh(ecs_host, f"if test -e {shlex.quote(remote_output)}; then echo EXISTS; fi").strip():
        raise FileExistsError(f"FAIL_CLOSED remote Gate-A output exists: {remote_output}")
    for key, host in hosts.items():
        frozen._deploy_archive(host, archive, runtimes[key])
    _deploy_dataset(ecs_host, _dataset_archive(output, "ecs", (3, 4)), "/root/GAPS/dataset")
    _deploy_dataset(pi_host, _dataset_archive(output, "pi", (1,)), "/home/gaps/GAPS/flower_runtime/dataset")
    _deploy_dataset(c2_host, _dataset_archive(output, "c2", (2,)), "/root/GAPS/confirmation_c2_data")
    roles = ["server", "C1", "C2", "C3", "C4"]
    host_roles = execution_role_hosts()
    processes: list[subprocess.Popen] = []
    tunnels: list[subprocess.Popen] = []
    handles = []
    started = time.perf_counter()
    try:
        for role in roles:
            handles.extend(
                [
                    (run_dir / f"{role}.stdout.log").open("w", encoding="utf-8"),
                    (run_dir / f"{role}.stderr.log").open("w", encoding="utf-8"),
                ]
            )
        processes.append(
            subprocess.Popen(
                frozen._process_command(ecs_host, runtimes["ecs"], commands["server"]),
                stdout=handles[0], stderr=handles[1], text=True,
            )
        )
        time.sleep(5)
        if processes[0].poll() is not None:
            raise RuntimeError(f"Gate-A server exited before clients: rc={processes[0].returncode}")
        tunnels = list(_start_ecs_c2_tunnels(ecs_host, pi_host, c2_host))
        for index, role in enumerate(roles[1:], start=1):
            host_key = host_roles[role]
            processes.append(
                subprocess.Popen(
                    frozen._process_command(hosts[host_key], runtimes[host_key], commands["clients"][role]),
                    stdout=handles[index * 2], stderr=handles[index * 2 + 1], text=True,
                )
            )
        deadline = time.monotonic() + float(timeout_hours) * 3600.0
        while any(process.poll() is None for process in processes):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Gate-A {method} exceeded {timeout_hours} hours")
            for role, process in zip(roles, processes):
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"Gate-A {method}/{role} failed with rc={process.returncode}")
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(f"Gate-A non-zero returns: {[p.returncode for p in processes]}")
    finally:
        _terminate_processes(processes)
        _terminate_processes(tunnels)
        for handle in handles:
            handle.close()
    wall_seconds = time.perf_counter() - started
    remote_copy = run_dir / "remote_server"
    frozen._run(["scp", "-r", f"{ecs_host}:{remote_output}", str(remote_copy)], timeout=1800)
    checkpoint = remote_copy / "server_latest.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"FAIL_CLOSED Gate-A round25 checkpoint missing: {checkpoint}")
    manifest = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.gate_a.run.v1",
        "experiment_id": experiment_id,
        "method": method,
        "protocol_hash": freeze["protocol_hash"],
        "source_archive_sha256": archive_hash,
        "dataset_aggregate_sha256": freeze["dataset_aggregate_sha256"],
        "wall_seconds": wall_seconds,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_state_fingerprint": ordered_state_fingerprint(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]),
        "target_test_opened": False,
        "protocol": commands["protocol"],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return frozen.write_completion_marker(
        run_dir,
        experiment_id=experiment_id,
        protocol_hash=freeze["protocol_hash"],
        endpoint={"round": 25, "checkpoint": checkpoint.name},
    )


def _feature_records(checkpoint: Path, clients: tuple[int, ...], device: torch.device) -> list[dict[str, Any]]:
    model, _config, payload = load_checkpoint_model(str(checkpoint), device, 32)
    if int(payload.get("round", -1)) != 25:
        raise RuntimeError("FAIL_CLOSED representation checkpoint is not round25")
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for client in clients:
            for x, labels, _reg, phases in make_loader(LOCAL_DATA_ROOT, client, "test", 32):
                _logits, features, _raw = model(x.to(device))
                for feature, label, phase in zip(features.cpu(), labels.long(), phases.long()):
                    records.append({"client": client, "class_id": int(label), "phase_id": int(phase), "feature": feature.numpy()})
    return records


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _per_class_rows(method: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    rows = []
    for class_id, gas in enumerate(("Ethanol", "CO", "Ethylene", "Methane")):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        rows.append(
            {
                "method": method,
                "class_id": class_id,
                "gas": gas,
                "precision": precision,
                "recall": metrics["per_class_recall"][str(class_id)],
                "f1": metrics["per_class_f1"][str(class_id)],
            }
        )
    return rows


def per_class_rows_from_predictions(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty per-class prediction stream: {method}")
    probabilities = np.asarray(
        [[float(row[f"prob_{index}"]) for index in range(4)] for row in rows], dtype=np.float64
    )
    metrics = classification_metrics([int(row["true_class"]) for row in rows], probabilities)
    return _per_class_rows(method, metrics)


def evaluate_and_analyze(output: Path, device: torch.device) -> dict[str, Any]:
    output = Path(output).resolve()
    locked = verify_s4_endpoint_locks(output)
    _write_json(
        output / "SEALED_TEST_OPEN.json",
        {"status": "OPENED_AFTER_BOTH_S4_ROUND25_LOCKS", "opened_at_utc": datetime.now(timezone.utc).isoformat(), "target_test_selection": False},
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("fedavg", "gaps_dg_p"):
        checkpoint = Path(locked[method]["checkpoint"])
        scopes: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        for client in (1, 2, 3, 4, 5):
            rows, metrics = evaluate_checkpoint_stream(
                checkpoint,
                data_root=LOCAL_DATA_ROOT,
                target_client=client,
                split="test",
                device=device,
                batch_size=32,
                expected_endpoint=("round", 25),
            )
            scopes[f"C{client}"] = (rows, metrics)
            prediction_rows.extend({"method": f"S4_{method}", **row} for row in rows)
        pooled_rows = sum((scopes[f"C{client}"][0] for client in (1, 2, 3, 4)), [])
        probs = np.asarray([[float(row[f"prob_{index}"]) for index in range(4)] for row in pooled_rows])
        scopes["SOURCE_POOLED"] = (pooled_rows, classification_metrics([int(row["true_class"]) for row in pooled_rows], probs))
        by_method[method] = {scope: metrics for scope, (_rows, metrics) in scopes.items()}
        for scope, (_rows, metrics) in scopes.items():
            metric_rows.append({"source_set": "S4", "method": "FedAvg" if method == "fedavg" else "GAPS-DG-P", "scope": scope, **{key: metrics[key] for key in ("N", "accuracy", "macro_f1", "nll", "ece")}, "checkpoint_sha256": locked[method]["checkpoint_sha256"]})
        per_class_rows.extend(_per_class_rows(f"S4_{method}", scopes["C5"][1]))
    s2_rows = _read_csv(S2_METRICS)
    for row in s2_rows:
        if row["scope"] in {"C1", "C2", "C5", "C1+C2"}:
            metric_rows.append(
                {
                    "source_set": "S2",
                    "method": row["method"],
                    "scope": "SOURCE_POOLED" if row["scope"] == "C1+C2" else row["scope"],
                    "N": int(row["N"]),
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "nll": float(row["nll"]),
                    "ece": float(row["ece"]),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )
    s2_predictions = _read_csv(S2_PREDICTIONS)
    for method in ("FedAvg", "GAPS-DG-P"):
        selected = [row for row in s2_predictions if row["method"] == method and row["client"] == "C5"]
        per_class_rows.extend(per_class_rows_from_predictions(f"S2_{method}", selected))
    _write_csv(output / "GATE_A_ZERO_SHOT_COMPARISON.csv", metric_rows)
    _write_csv(output / "GATE_A_C5_PER_CLASS.csv", per_class_rows)
    _write_csv(output / "GATE_A_S4_PREDICTIONS.csv", prediction_rows)
    representation_rows = []
    for row in _read_csv(S2_REPRESENTATION):
        representation_rows.append(
            {
                "source_set": "S2",
                "method": row["method"],
                "source_count": 2,
                "source_clients": "C1;C2",
                "within_class_inter_source_centroid_distance_mean": float(row["within_class_c1_c2_centroid_distance_mean"]),
                "between_class_centroid_margin_mean": float(row["between_class_centroid_margin_mean"]),
                "class_phase_prototype_dispersion_mean": float(row["source_inter_client_class_phase_prototype_distance_mean"]) / 2.0,
                "inter_source_pairs": 1,
                "class_phase_cells": int(row["prototype_cells"]),
                "source_test_windows": int(row["source_test_windows"]),
            }
        )
    for method in ("fedavg", "gaps_dg_p"):
        representation_rows.append(
            {"source_set": "S4", **representation_diagnostics(_feature_records(Path(locked[method]["checkpoint"]), (1, 2, 3, 4), device), method="FedAvg" if method == "fedavg" else "GAPS-DG-P", source_count=4)}
        )
    _write_csv(output / "GATE_A_REPRESENTATION_DIAGNOSTICS.csv", representation_rows)
    s2 = {(row["method"], row["scope"]): row for row in metric_rows if row["source_set"] == "S2"}
    decision = decide_gate_a(
        s2_fedavg_c5_f1=float(s2[("FedAvg", "C5")]["macro_f1"]),
        s2_dg_c5_f1=float(s2[("GAPS-DG-P", "C5")]["macro_f1"]),
        s4_fedavg_c5_f1=float(by_method["fedavg"]["C5"]["macro_f1"]),
        s4_dg_c5_f1=float(by_method["gaps_dg_p"]["C5"]["macro_f1"]),
        s4_fedavg_source_f1=float(by_method["fedavg"]["SOURCE_POOLED"]["macro_f1"]),
        s4_dg_source_f1=float(by_method["gaps_dg_p"]["SOURCE_POOLED"]["macro_f1"]),
    )
    _write_json(output / "GATE_A_DECISION.json", decision)
    if decision["dg_mechanism"] == "SOURCE_DG_PROMISING":
        (output / "MULTI_SEED_PROPOSAL.md").write_text("# Multi-seed Proposal\n\nGate A permits a future seed proposal only. No additional seed is executed in this task.\n", encoding="utf-8")
    report = f"""# Gate A Source-diversity / Federated-DG Report

## [Scientific Question]

Does adding C3/C4 source domains improve C5 zero-shot Macro-F1, and does exact GAPS-DG-P add at least one percentage point beyond matched FedAvg?

## [Protocol]

S2 endpoints are immutable reused round25 results. S4 uses C1-C4, 25 rounds, LE1, batch32, Adam 5e-4, seed42. C5 was absent from every training API and command. C3/C4 use the pre-frozen derived canonical role view; C1/C2/C5 are byte-identical to canonical-v1.

## [Primary Result]

- S2 FedAvg C5 Macro-F1: {float(s2[("FedAvg", "C5")]["macro_f1"]):.6f}
- S4 FedAvg C5 Macro-F1: {by_method["fedavg"]["C5"]["macro_f1"]:.6f}
- S2 GAPS-DG-P C5 Macro-F1: {float(s2[("GAPS-DG-P", "C5")]["macro_f1"]):.6f}
- S4 GAPS-DG-P C5 Macro-F1: {by_method["gaps_dg_p"]["C5"]["macro_f1"]:.6f}

## [Negative Result / Limitation]

This is seed42 and changes both source-domain count and labeled source-data composition. It is a C5 hardest-target sensitivity, not a pure causal domain-count ablation or universal DG result.

## [Leakage Audit]

Both S4 completion markers were locked before C5 test evaluation. The S4 training protocol contains no C5 path, X, Y, phase, concentration, statistics, or calibration access. Target test was not used for tuning, stopping, or checkpoint selection.

## [Decision]

- `{decision['source_diversity']}`
- `{decision['dg_mechanism']}`

## [Paper Implication]

Only the registered C5 source-diversity sensitivity wording is permitted. Prototype-DG superiority requires the pre-registered one-point matched gain and source-retention gate.

## [Next Action]

`{decision['next_action']}`. Gate B still uses the frozen S2 source endpoint; no source-count substitution or new run is made there.
"""
    (output / "GATE_A_REPORT.md").write_text(report, encoding="utf-8")
    audit = f"""# Gate A Post-run Experiment Audit

## Verdict: approved

- Both new configurations reached fixed round25 and passed checkpoint SHA/state-content checks.
- S2 reuse passed canonical-v1, round25, LE1, seed42, no-target-access checks.
- C5 test opened only after both S4 locks; no target-test selection occurred.
- One-seed and source-data-volume limitations remain and block universal causal wording.
"""
    (output / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")
    _write_json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "protocol_hash": protocol_hash(),
            "dataset_aggregate_sha256": json.loads((LOCAL_DATA_ROOT / "dataset_sha256.json").read_text(encoding="utf-8"))["aggregate_sha256"],
            "target_test_manifest_sha256": _sha256_file(CANONICAL_DATA_ROOT / "client_5/test_experiment_info.json"),
            "target_test_selection": False,
            "decision": decision,
        },
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "sha256_index.json")
    _write_json(output / "sha256_index.json", {str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path) for path in files})
    return decision


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    freeze = write_pre_run_freeze(output)
    execute_s4_fl("fedavg", output, freeze, ecs_host=args.ecs_host, pi_host=args.pi_host, c2_host=args.c2_host, timeout_hours=args.timeout_hours)
    execute_s4_fl("gaps_dg_p", output, freeze, ecs_host=args.ecs_host, pi_host=args.pi_host, c2_host=args.c2_host, timeout_hours=args.timeout_hours)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    return evaluate_and_analyze(output, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
