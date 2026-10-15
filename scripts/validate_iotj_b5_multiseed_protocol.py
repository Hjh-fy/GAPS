"""Fail-closed validation for the frozen B5 five-seed M0 protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SEEDS = (42, 43, 44, 45, 46)
TRAIN_SEEDS = (43, 44, 45, 46)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_seed_text(value: str, seed: int) -> str:
    return (
        value.replace(f"s{seed}", "s<SEED>")
        .replace(f"_seed{seed}", "_seed<SEED>")
    )


def canonical_algorithm_payload(payload: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Remove only seed and seed-derived identity/provenance fields."""
    value = copy.deepcopy(dict(payload))
    value["algorithm_config_sha256"] = "<DERIVED_HASH>"
    value["execution_stage"] = "<NON_ALGORITHM_STAGE_LABEL>"
    value["run_id"] = "<SEED_DERIVED_RUN_ID>"
    value["run_name"] = "<SEED_DERIVED_RUN_NAME>"
    value["protocol"]["training_seed"] = "<SEED>"
    for command in value["commands"].values():
        for index, token in enumerate(command):
            if index and command[index - 1] == "--seed":
                command[index] = "<SEED>"
            elif isinstance(token, str):
                command[index] = _replace_seed_text(token, seed)
    return value


def validate_protocol(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    checkpoint = root / manifest["seed42_reference"]["checkpoint_path"]
    if not checkpoint.is_file():
        errors.append("seed42 checkpoint missing")
    elif sha256_file(checkpoint) != manifest["seed42_reference"]["checkpoint_sha256"]:
        errors.append("seed42 checkpoint SHA256 mismatch")

    for relative, expected_hash in manifest["frozen_input_files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"frozen input missing: {relative}")
        elif sha256_file(path) != expected_hash:
            errors.append(f"frozen input SHA256 mismatch: {relative}")

    command_payloads: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        record = manifest["command_manifests"][str(seed)]
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"seed{seed} command manifest missing")
            continue
        if sha256_file(path) != record["file_sha256"]:
            errors.append(f"seed{seed} command manifest file SHA256 mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("algorithm_config_sha256") != record["algorithm_config_sha256"]:
            errors.append(f"seed{seed} algorithm config SHA256 mismatch")
        if payload.get("protocol", {}).get("training_seed") != seed:
            errors.append(f"seed{seed} training seed mismatch")
        if payload.get("protocol", {}).get("split_seed") != 42:
            errors.append(f"seed{seed} split seed drift")
        command_payloads[seed] = payload

    if len(command_payloads) == len(SEEDS):
        reference = canonical_algorithm_payload(command_payloads[42], 42)
        for seed in TRAIN_SEEDS:
            observed = canonical_algorithm_payload(command_payloads[seed], seed)
            if observed != reference:
                errors.append(
                    f"seed{seed} has non-seed algorithm differences versus seed42"
                )

    expected_topology = {
        "--ecs-host": "root@121.40.139.213",
        "--pi-hosts": "gaps@192.168.137.172",
        "--c2-host": "root@114.55.171.63",
        "--c2-python": "/root/gaps_c2_cpu_env/bin/python",
    }
    launch_root = manifest_path.parent / "commands"
    for seed in TRAIN_SEEDS:
        launch = launch_root / f"launch_seed{seed}.cmd"
        if not launch.is_file():
            errors.append(f"seed{seed} launch command missing")
            continue
        text = launch.read_text(encoding="utf-8")
        if f'--runs "B5:{seed}"' not in text:
            errors.append(f"seed{seed} launch run selector mismatch")
        for flag, expected in expected_topology.items():
            if f'{flag} "{expected}"' not in text:
                errors.append(f"seed{seed} launch topology mismatch for {flag}")
        if f"seed{seed}/raw" not in text:
            errors.append(f"seed{seed} raw output is not isolated")

    source_assets = (
        (
            "federated H1",
            manifest["component_seed_coupling"]["federated_H1_source_model"],
        ),
        (
            "pooled H1/H2/H3",
            manifest["component_seed_coupling"]["pooled_H1_H2_H3_source_models"],
        ),
    )
    for label, record in source_assets:
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"{label} source asset missing")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"{label} source asset SHA256 mismatch")

    return {
        "schema_version": "iotj.b5_multiseed_m0_validation.v1",
        "status": "ready_for_preflight" if not errors else "blocked",
        "errors": errors,
        "seed42_retrained": False,
        "new_training_seeds": list(TRAIN_SEEDS),
        "normalized_algorithm_only_seed_varies": not any(
            "non-seed algorithm differences" in error for error in errors
        ),
        "three_host_preflight_status": "not_run",
        "formal_training_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/iotj_b5_multiseed_20260724/protocol_manifest.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate_protocol(Path.cwd(), args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ready_for_preflight":
        raise SystemExit(1)
