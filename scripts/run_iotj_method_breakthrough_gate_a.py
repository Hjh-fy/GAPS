"""Gate-A source-diversity and exact GAPS-DG-P protocol utilities."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_iotj_canonical_v1_comparators import build_source_fl_commands
from scripts.run_iotj_source_dg_g2 import build_g2_commands


DATASET_NAME = "iotj_canonical_v1_s4_role_view"
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"


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
