"""Prepare immutable B5 seed42 server-DA budget ablation inputs.

The completed LE1 + DA100 run is the reference.  Client training, model,
dataset, topology, seed, rounds, and every DA setting except the optimizer
step count remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.freeze_iotj_confirmation_protocol import (
    ALGORITHM_CONFIG_FIELDS,
    canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET_RUN_ID = "c12_to_c5__b5__s42"
HASH_FIELD = "protocol_manifest_sha256"
TOPOLOGY_HASH_FIELD = "execution_topology_manifest_sha256"
ALLOWED_STEPS = frozenset({30, 50, 80})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _replace_option(command: Sequence[str], option: str, value: str) -> list[str]:
    result = list(command)
    positions = [index for index, item in enumerate(result) if item == option]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one {option} in command")
    position = positions[0]
    if position + 1 >= len(result):
        raise ValueError(f"{option} has no value")
    result[position + 1] = value
    return result


def _algorithm_hash(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {field: manifest[field] for field in ALGORITHM_CONFIG_FIELDS}
    )


def _validate_self_hash(
    payload: Mapping[str, Any], hash_field: str, label: str
) -> None:
    claimed = payload.get(hash_field)
    body = {key: value for key, value in payload.items() if key != hash_field}
    if claimed != canonical_sha256(body):
        raise ValueError(f"{label} self SHA256 mismatch")


def build_level(
    *,
    da_steps: int,
    base_protocol: Path,
    base_command_root: Path,
    base_topology: Path,
    source_manifest: Path,
    dataset_manifest: Path,
    source_archive: Path,
    output_root: Path,
) -> dict[str, Any]:
    if da_steps not in ALLOWED_STEPS:
        raise ValueError(f"da_steps must be one of {sorted(ALLOWED_STEPS)}")
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {output_root}")

    protocol = _read_json(base_protocol)
    topology = _read_json(base_topology)
    _validate_self_hash(protocol, HASH_FIELD, "base protocol")
    _validate_self_hash(topology, TOPOLOGY_HASH_FIELD, "base topology")
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list):
        raise ValueError("base protocol schedule is missing")

    command_manifests: dict[str, dict[str, Any]] = {}
    for row in schedule:
        if not isinstance(row, Mapping) or not isinstance(row.get("run_id"), str):
            raise ValueError("invalid base protocol schedule row")
        run_id = str(row["run_id"])
        path = Path(base_command_root) / run_id / "command_manifest.json"
        manifest = _read_json(path)
        if manifest.get("run_id") != run_id:
            raise ValueError(f"base command identity mismatch: {path}")
        if manifest.get("algorithm_config_sha256") != _algorithm_hash(manifest):
            raise ValueError(f"base command algorithm SHA mismatch: {path}")
        command_manifests[run_id] = copy.deepcopy(manifest)

    target = command_manifests[TARGET_RUN_ID]
    if target.get("group_id") != "B5":
        raise ValueError("target command is not B5")
    if target.get("protocol", {}).get("training_seed") != 42:
        raise ValueError("target command is not seed42")
    if target.get("training", {}).get("local_epochs") != 1:
        raise ValueError("base local_epochs must be 1")
    if target.get("server_adaptation", {}).get("steps") != 100:
        raise ValueError("base server DA steps must be 100")

    commands = target.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("target commands are missing")
    commands["server_ecs"] = _replace_option(
        commands["server_ecs"], "--domain-adapt-steps", str(da_steps)
    )
    run_name = (
        "B5_proto_replay_corrected_full_da_c12_to_c5_"
        f"s42_r25_le1_da{da_steps}"
    )
    commands["server_ecs"] = _replace_option(
        commands["server_ecs"], "--run-name", run_name
    )
    target["run_name"] = run_name
    target["server_adaptation"]["steps"] = da_steps
    target["execution_stage"] = "post_freeze_server_da_budget_ablation"
    target["server_da_budget_ablation"] = {
        "baseline_steps_per_round": 100,
        "variant_steps_per_round": da_steps,
        "only_algorithm_factor_changed": "server_adaptation.steps",
        "evidence_status": "POST_FREEZE_SINGLE_SEED_SERVER_DA_BUDGET_SENSITIVITY",
    }
    target["algorithm_config_sha256"] = _algorithm_hash(target)

    target_rows = [
        row
        for row in schedule
        if isinstance(row, dict) and row.get("run_id") == TARGET_RUN_ID
    ]
    if len(target_rows) != 1:
        raise ValueError("target schedule row is not unique")
    target_rows[0]["algorithm_config_sha256"] = target["algorithm_config_sha256"]
    protocol["server_da_budget_ablation"] = {
        "experiment_id": f"IOTJ-B5-LE1-DA{da_steps}-S42-20260731",
        "baseline_steps_per_round": 100,
        "variant_steps_per_round": da_steps,
        "total_steps": 25 * da_steps,
        "single_seed": 42,
        "post_freeze": True,
        "test_not_used_for_training_selection_or_stopping": True,
    }
    protocol.pop(HASH_FIELD, None)
    protocol[HASH_FIELD] = canonical_sha256(protocol)
    for manifest in command_manifests.values():
        manifest[HASH_FIELD] = protocol[HASH_FIELD]

    topology["algorithm_config_sha256_by_run"][TARGET_RUN_ID] = target[
        "algorithm_config_sha256"
    ]
    topology["server_da_budget_ablation"] = {
        "run_id": TARGET_RUN_ID,
        "variant_steps_per_round": da_steps,
        "placement_unchanged": True,
    }
    topology.pop(TOPOLOGY_HASH_FIELD, None)
    topology[TOPOLOGY_HASH_FIELD] = canonical_sha256(topology)

    protocol_path = output_root / "confirmation_protocol_manifest.json"
    topology_path = output_root / "execution_topology_manifest.json"
    command_root = output_root / "commands"
    _write_json(protocol_path, protocol)
    _write_json(topology_path, topology)
    for run_id, manifest in command_manifests.items():
        _write_json(command_root / run_id / "command_manifest.json", manifest)

    frozen_inputs = {
        "schema_version": "iotj.b5_server_da_budget_inputs.v1",
        "experiment_id": f"IOTJ-B5-LE1-DA{da_steps}-S42-20260731",
        "target_run_id": TARGET_RUN_ID,
        "seed": 42,
        "local_epochs": 1,
        "server_da_steps_per_round": da_steps,
        "server_da_total_steps": 25 * da_steps,
        "protocol_manifest": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "self_sha256": protocol[HASH_FIELD],
        },
        "execution_topology_manifest": {
            "path": str(topology_path),
            "sha256": _sha256(topology_path),
            "self_sha256": topology[TOPOLOGY_HASH_FIELD],
        },
        "command_root": str(command_root),
        "target_algorithm_config_sha256": target["algorithm_config_sha256"],
        "unchanged_inputs": {
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": _sha256(source_manifest),
            },
            "dataset_manifest": {
                "path": str(dataset_manifest),
                "sha256": _sha256(dataset_manifest),
            },
            "source_archive": {
                "path": str(source_archive),
                "sha256": _sha256(source_archive),
            },
            "base_protocol": {
                "path": str(base_protocol),
                "sha256": _sha256(base_protocol),
            },
            "base_topology": {
                "path": str(base_topology),
                "sha256": _sha256(base_topology),
            },
        },
        "held_constant": {
            "classifier": "B5",
            "seed": 42,
            "rounds": 25,
            "local_epochs": 1,
            "batch_size": 32,
            "client_optimizer": "Adam",
            "client_lr": 0.0005,
            "source_clients": ["C1", "C2"],
            "target_client": "C5",
            "topology_id": topology["topology_id"],
        },
        "only_intended_variable": "server_adaptation.steps",
        "evidence_boundary": (
            "Post-freeze single-seed server-DA compute-budget sensitivity; "
            "no frozen B5, runtime, regression, QC, or paper evidence reselection."
        ),
    }
    manifest_path = output_root / "derived_input_manifest.json"
    _write_json(manifest_path, frozen_inputs)
    frozen_inputs["derived_input_manifest_sha256"] = _sha256(manifest_path)
    return frozen_inputs


def _parser() -> argparse.ArgumentParser:
    base = (
        ROOT
        / "results/iotj_b5_local_epoch_ablation_20260729"
        / "le1/protocol_inputs"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--da-steps", type=int, choices=sorted(ALLOWED_STEPS), required=True
    )
    parser.add_argument(
        "--base-protocol",
        type=Path,
        default=base / "confirmation_protocol_manifest.json",
    )
    parser.add_argument(
        "--base-command-root", type=Path, default=base / "commands"
    )
    parser.add_argument(
        "--base-topology",
        type=Path,
        default=base / "execution_topology_manifest.json",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/c2e_summary/source_archive_manifest.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=ROOT / "results/c2e_summary/dataset_manifest.json",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT / "results/c2e/source/confirmation_source.tar",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_level(
        da_steps=args.da_steps,
        base_protocol=args.base_protocol,
        base_command_root=args.base_command_root,
        base_topology=args.base_topology,
        source_manifest=args.source_manifest,
        dataset_manifest=args.dataset_manifest,
        source_archive=args.source_archive,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
