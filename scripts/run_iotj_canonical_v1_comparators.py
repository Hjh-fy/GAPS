"""Run the frozen minimal canonical-v1 comparator matrix without test access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_iotj_final_classification_le1 as frozen


_FROZEN_BUILDER = frozen.build_flower_commands
METHODS = ("FedAvg", "FedProx", "SCAFFOLD")
TARGETS = ("C3", "C4", "C5")
DATASET_NAME = "iotj_canonical_v1"
LOCAL_DATA_ROOT = ROOT / "dataset" / DATASET_NAME
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators"


def canonical_comparator_config() -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
        "source_fl_methods": list(METHODS),
        "posthoc_da_methods": ["MMD"],
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "hyperparameter_search": False,
        "target_test_selection": False,
    }


def _set_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def _remove_option(command: list[str], option: str) -> None:
    if option in command:
        index = command.index(option)
        del command[index:index + 2]


def build_source_fl_commands(method: str) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown canonical comparator: {method}")
    historical_id = "FCL-E1-SCAFFOLD" if method == "SCAFFOLD" else "FCL-E1-FEDPROX"
    experiment_id = f"CAN-V1-CMP-{method.upper()}"
    commands = _FROZEN_BUILDER(historical_id)
    replacements = (
        (frozen.REMOTE_DATA_ROOT, REMOTE_DATA_ROOT),
        (frozen.PI_DATA_ROOT, PI_DATA_ROOT),
        (frozen.C2_DATA_ROOT, C2_DATA_ROOT),
        (historical_id, experiment_id),
    )
    for role in ("server", "client_c1", "client_c2"):
        values = list(commands[role])
        for old, new in replacements:
            values = [value.replace(old, new) for value in values]
        commands[role] = values
    for role in ("client_c1", "client_c2"):
        _set_option(commands[role], "--local-epochs", "1")
    if method == "FedAvg":
        for role in ("client_c1", "client_c2"):
            _remove_option(commands[role], "--proximal-mu")
    optimizer = "SGD" if method == "SCAFFOLD" else "Adam"
    optimizer_note = "canonical SCAFFOLD implementation" if method == "SCAFFOLD" else "frozen GAPS experimental protocol"
    commands["protocol"].update({
        **canonical_comparator_config(),
        "experiment_id": experiment_id,
        "method": method,
        "optimizer": optimizer,
        "optimizer_lr": 5e-4,
        "optimizer_note": optimizer_note,
        "target_information_regime": "source_only",
        "target_x": False,
        "target_y": False,
        "checkpoint_reuse": False,
        "checkpoint_selection": "fixed_round_25",
    })
    return commands


def canonical_mmd_spec(target: str) -> dict[str, Any]:
    target = target.upper()
    if target not in TARGETS:
        raise ValueError(target)
    return {
        "experiment_id": f"CAN-V1-CMP-MMD-{target}",
        "method": "mmd",
        "target": target,
        "source_checkpoint_role": "canonical_FedAvg_round25",
        "source_checkpoint_reuse": "same checkpoint for C3/C4/C5",
        "target_fields": ["x"],
        "steps": 100,
        "optimizer": "Adam",
        "optimizer_lr": 5e-4,
        "alignment_weight": 0.5,
        "source_batch_convention": "combined_canonical_C1_C2_calibration",
        "target_ce": False,
        "conditional": False,
        "pseudo_labels": False,
        "checkpoint_selection": "fixed_step_100",
        "target_test_selection": False,
        "hyperparameter_search": False,
        "seed": 42,
    }


def protocol_hash() -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(canonical_comparator_config(), sort_keys=True).encode())
    for method in METHODS:
        digest.update(json.dumps(build_source_fl_commands(method), sort_keys=True).encode())
    for target in TARGETS:
        digest.update(json.dumps(canonical_mmd_spec(target), sort_keys=True).encode())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def write_or_validate_freeze(path: Path) -> dict[str, Any]:
    invariant = {
        "schema_version": "iotj.canonical_v1.comparator.pre_run.v1",
        "status": "FROZEN",
        "protocol_hash": protocol_hash(),
        "config": canonical_comparator_config(),
        "source_fl": {method: build_source_fl_commands(method)["protocol"] for method in METHODS},
        "mmd": [canonical_mmd_spec(target) for target in TARGETS],
        "test_open_policy": "after every source-FL and MMD fixed endpoint completes",
    }
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        for key, value in invariant.items():
            if observed.get(key) != value:
                raise RuntimeError(f"FAIL_CLOSED canonical comparator freeze differs: {key}")
        if not observed.get("freeze_commit"):
            raise RuntimeError("FAIL_CLOSED canonical comparator freeze has no commit")
        return observed
    payload = {**invariant, "freeze_commit": git_head()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_scaffold_source_gate(output: Path) -> dict[str, Any]:
    original_root, original_data = frozen.RESULT_ROOT, frozen.LOCAL_DATA_ROOT
    try:
        frozen.RESULT_ROOT = output
        frozen.LOCAL_DATA_ROOT = LOCAL_DATA_ROOT
        return frozen.run_scaffold_source_gate()
    finally:
        frozen.RESULT_ROOT, frozen.LOCAL_DATA_ROOT = original_root, original_data


def execute_source_fl(method: str, output: Path, lock: dict[str, Any], args: argparse.Namespace) -> None:
    experiment_id = f"CAN-V1-CMP-{method.upper()}"
    run_dir = output / "source_fl" / experiment_id
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial comparator run exists: {run_dir}")
    original_root, original_builder = frozen.RESULT_ROOT, frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = output / "source_fl"
        frozen.build_flower_commands = lambda _experiment_id: build_source_fl_commands(method)
        frozen.execute_full_fl(
            experiment_id,
            protocol_hash=lock["protocol_hash"],
            lock_payload={"freeze_commit": lock["freeze_commit"], "baseline": method},
            ecs_host=args.ecs_host,
            pi_host=args.pi_host,
            c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT, frozen.build_flower_commands = original_root, original_builder


def _fedavg_checkpoint(output: Path) -> Path:
    manifest_path = output / "source_fl/CAN-V1-CMP-FEDAVG/run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("FAIL_CLOSED canonical FedAvg completion missing before MMD")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError("FAIL_CLOSED canonical FedAvg checkpoint missing")
    return checkpoint


def execute_mmd(target: str, output: Path, lock: dict[str, Any]) -> None:
    import torch

    from federated_dataset import create_merged_calibration_loader
    from gaps_flower.canonical_uda import run_canonical_uda
    from gaps_flower.evaluate_checkpoint import load_checkpoint_model
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
    from gaps_flower.target_information import TargetAccessLedger, load_target_calibration_x

    spec = canonical_mmd_spec(target)
    run_dir = output / "mmd" / spec["experiment_id"]
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial MMD run exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "locked_run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = _fedavg_checkpoint(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _payload = load_checkpoint_model(str(checkpoint), device, 32)
    source_fingerprint = ordered_state_content_fingerprint(model.state_dict())
    source_loader = create_merged_calibration_loader(
        [LOCAL_DATA_ROOT / "client_1", LOCAL_DATA_ROOT / "client_2"], batch_size=32, num_workers=0
    )
    ledger = TargetAccessLedger(run_dir / "target_access_ledger.jsonl")
    target_loader = load_target_calibration_x(
        LOCAL_DATA_ROOT / f"client_{target[1:]}", method="mmd", ledger=ledger,
        batch_size=32, shuffle=True, seed=42,
    )
    adapted, diagnostics, seconds = run_canonical_uda(
        "mmd", model, source_loader, target_loader, device,
        num_steps=100, model_lr=5e-4, alignment_weight=0.5,
        expected_source_fingerprint=source_fingerprint, seed=42, formal=True,
    )
    checkpoint_path = run_dir / "adapted_step_100.pth"
    torch.save({
        "model_state": adapted.state_dict(),
        "source_checkpoint_ordered_fingerprint": source_fingerprint,
        "experiment_id": spec["experiment_id"], "step": 100,
    }, checkpoint_path)
    with (run_dir / "adaptation_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader(); writer.writerows(diagnostics)
    manifest = {
        "schema_version": "iotj.canonical_v1.comparator.mmd.v1",
        "experiment_id": spec["experiment_id"], "protocol_hash": lock["protocol_hash"],
        "spec": spec, "adaptation_seconds": seconds, "device": str(device),
        "source_checkpoint": str(checkpoint), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": frozen._sha256_file(checkpoint_path), "target_test_opened": False,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frozen.write_completion_marker(
        run_dir, experiment_id=spec["experiment_id"], protocol_hash=lock["protocol_hash"],
        endpoint={"steps": 100, "checkpoint": checkpoint_path.name},
    )


def run(args: argparse.Namespace) -> None:
    from scripts.run_iotj_canonical_v1_classification import validate_preflight

    validate_preflight()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = write_or_validate_freeze(output / "COMPARATOR_PRE_RUN_FREEZE.json")
    gate_path = output / "preflight/scaffold_source_numerical_gate.json"
    if not gate_path.is_file():
        run_scaffold_source_gate(output)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("FAIL_CLOSED SCAFFOLD source gate failed")
    for method in METHODS:
        execute_source_fl(method, output, lock, args)
    for target in TARGETS:
        execute_mmd(target, output, lock)
    (output / "FIXED_ENDPOINTS_COMPLETE_TEST_STILL_SEALED.json").write_text(
        json.dumps({"status": "PASS", "source_fl": list(METHODS), "mmd_targets": list(TARGETS), "target_test_opened": False}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
