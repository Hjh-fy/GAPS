"""Run the single frozen GAPS-DG-P source-only Gate-2 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.evaluate_checkpoint import load_checkpoint_model, make_loader  # noqa: E402
from gaps_flower.posthoc_commissioning import sha256_file  # noqa: E402
from scripts import run_iotj_final_classification_le1 as frozen  # noqa: E402
from scripts.summarize_iotj_classification_ablation import (  # noqa: E402
    classification_metrics,
    evaluate_checkpoint_stream,
)


EXPERIMENT_ID = "CAN-V1-MR-G2-DGP"
DATASET_NAME = "iotj_canonical_v1"
LOCAL_DATA_ROOT = ROOT / "dataset" / DATASET_NAME
REMOTE_DATA_ROOT = f"/root/GAPS/dataset/{DATASET_NAME}"
PI_DATA_ROOT = f"/home/gaps/GAPS/flower_runtime/dataset/{DATASET_NAME}"
C2_DATA_ROOT = f"/root/GAPS/confirmation_c2_data/{DATASET_NAME}"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate2_source_dg"
FEDAVG_CHECKPOINT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-FEDAVG/remote_server/server_latest.pth"
_FROZEN_BUILDER = frozen.build_flower_commands


def _set_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def build_g2_commands() -> dict[str, Any]:
    commands = _FROZEN_BUILDER("FCL-E4-A1")
    replacements = (
        (frozen.REMOTE_DATA_ROOT, REMOTE_DATA_ROOT),
        (frozen.PI_DATA_ROOT, PI_DATA_ROOT),
        (frozen.C2_DATA_ROOT, C2_DATA_ROOT),
        ("FCL-E4-A1", EXPERIMENT_ID),
    )
    for role in ("server", "client_c1", "client_c2"):
        values = list(commands[role])
        for old, new in replacements:
            values = [value.replace(old, new) for value in values]
        commands[role] = values
    for role in ("server", "client_c1", "client_c2"):
        _set_option(commands[role], "--profile", "dg_proto")
    for role in ("client_c1", "client_c2"):
        _set_option(commands[role], "--local-epochs", "1")
    _set_option(commands["server"], "--ablation-variant", "G2-DG-P")
    _set_option(commands["server"], "--use-selective-agg", "false")
    _set_option(commands["server"], "--require-selective-after-warmup", "false")
    _set_option(commands["server"], "--use-proto-mmd", "false")
    _set_option(commands["server"], "--use-domain-adapt", "false")
    commands["protocol"].update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset": DATASET_NAME,
            "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            "method": "GAPS-DG-P",
            "source_clients": ["C1", "C2"],
            "target": "C5",
            "target_x": False,
            "target_y": False,
            "target_phase": False,
            "source_phase": True,
            "source_phase_observability": "acquisition_window_time_metadata",
            "lambda_proto": 0.05,
            "prototype_key": "class_x_response_phase",
            "prototype_aggregation": "sample_count_weighted_mean_then_EMA_0.8",
            "round_1": "CE_ONLY_UPLOAD_PROTOTYPES",
            "round_2_to_25": "CE_PLUS_GLOBAL_PROTOTYPE_ALIGNMENT",
            "selective_aggregation": False,
            "replay": False,
            "server_domain_adaptation": False,
            "hyperparameter_search": False,
            "checkpoint_selection": "fixed_round_25",
            "target_test_selection": False,
        }
    )
    return commands


def g2_round_contract(round_id: int) -> str:
    if not 1 <= int(round_id) <= 25:
        raise ValueError("G2 round must be in [1, 25]")
    return "CE_ONLY_UPLOAD_PROTOTYPES" if int(round_id) == 1 else "CE_PLUS_GLOBAL_PROTOTYPE_ALIGNMENT"


def protocol_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_g2_commands(), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_freeze(output: Path) -> dict[str, Any]:
    path = output / "G2_PRE_RUN_FREEZE.json"
    payload = {
        "schema_version": "iotj.canonical_v1.g2.pre_run.v1",
        "status": "FROZEN",
        "freeze_commit": _git_head(),
        "protocol_hash": protocol_hash(),
        "commands": build_g2_commands(),
        "target_test_opened": False,
        "decision_threshold": {
            "c5_macro_f1_min_gain": 0.01,
            "merged_source_macro_f1_max_drop": 0.01,
        },
    }
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError("FAIL_CLOSED G2 pre-run freeze differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return payload


def execute_fl(output: Path, freeze: dict[str, Any], args: argparse.Namespace) -> None:
    run_dir = output / EXPERIMENT_ID
    if (run_dir / "fixed_endpoint_complete.json").is_file():
        return
    if run_dir.exists():
        raise FileExistsError(f"FAIL_CLOSED partial G2 run exists: {run_dir}")
    original_root, original_builder = frozen.RESULT_ROOT, frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = output
        frozen.build_flower_commands = lambda _experiment_id: build_g2_commands()
        frozen.execute_full_fl(
            EXPERIMENT_ID,
            protocol_hash=freeze["protocol_hash"],
            lock_payload={"freeze_commit": freeze["freeze_commit"], "gate": "G2"},
            ecs_host=args.ecs_host,
            pi_host=args.pi_host,
            c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT = original_root
        frozen.build_flower_commands = original_builder


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[f"prob_{i}"]) for i in range(4)] for row in rows])


def _evaluate(checkpoint: Path, device: torch.device):
    scopes: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for client in (1, 2, 5):
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
    merged = scopes["C1"][0] + scopes["C2"][0]
    scopes["C1+C2"] = (
        merged,
        classification_metrics(
            [int(row["true_class"]) for row in merged], _probabilities(merged)
        ),
    )
    return scopes


def _feature_records(checkpoint: Path, device: torch.device) -> list[dict[str, Any]]:
    model, _config, payload = load_checkpoint_model(str(checkpoint), device, 32)
    if int(payload.get("round", -1)) != 25:
        raise RuntimeError("representation checkpoint is not round25")
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for client in (1, 2):
            loader = make_loader(LOCAL_DATA_ROOT, client, "test", 32)
            for batch in loader:
                x, labels, _reg, phases = batch
                _logits, features, _raw = model(x.to(device))
                for feature, label, phase in zip(
                    features.cpu(), labels.long(), phases.long()
                ):
                    records.append(
                        {
                            "client": client,
                            "class_id": int(label),
                            "phase_id": int(phase),
                            "feature": feature.numpy(),
                        }
                    )
    return records


def _representation_metrics(checkpoint: Path, method: str, device: torch.device) -> dict[str, Any]:
    records = _feature_records(checkpoint, device)
    proto_distances: list[float] = []
    within_class: list[float] = []
    class_centroids: list[np.ndarray] = []
    for class_id in range(4):
        class_features = np.stack([row["feature"] for row in records if row["class_id"] == class_id])
        class_centroids.append(class_features.mean(axis=0))
        c1 = np.stack([row["feature"] for row in records if row["class_id"] == class_id and row["client"] == 1])
        c2 = np.stack([row["feature"] for row in records if row["class_id"] == class_id and row["client"] == 2])
        within_class.append(float(np.linalg.norm(c1.mean(axis=0) - c2.mean(axis=0))))
        phases = sorted(set(row["phase_id"] for row in records if row["class_id"] == class_id))
        for phase in phases:
            p1 = [row["feature"] for row in records if row["class_id"] == class_id and row["phase_id"] == phase and row["client"] == 1]
            p2 = [row["feature"] for row in records if row["class_id"] == class_id and row["phase_id"] == phase and row["client"] == 2]
            if p1 and p2:
                proto_distances.append(float(np.linalg.norm(np.mean(p1, axis=0) - np.mean(p2, axis=0))))
    margins = [
        float(np.linalg.norm(class_centroids[i] - class_centroids[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    return {
        "method": method,
        "source_inter_client_class_phase_prototype_distance_mean": float(np.mean(proto_distances)),
        "within_class_c1_c2_centroid_distance_mean": float(np.mean(within_class)),
        "between_class_centroid_margin_mean": float(np.mean(margins)),
        "prototype_cells": len(proto_distances),
        "source_test_windows": len(records),
    }


def evaluate_and_analyze(output: Path, device: torch.device) -> dict[str, Any]:
    run_dir = output / EXPERIMENT_ID
    marker = json.loads((run_dir / "fixed_endpoint_complete.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if marker.get("target_test_opened") is not False or int(marker["fixed_endpoint"]["round"]) != 25:
        raise RuntimeError("FAIL_CLOSED invalid G2 endpoint lock")
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("FAIL_CLOSED G2 checkpoint hash mismatch")
    _write_json(
        output / "SEALED_TEST_OPEN.json",
        {"status": "OPENED_AFTER_G2_ROUND25_LOCK", "target_test_selection": False},
    )
    endpoints = {"FedAvg": FEDAVG_CHECKPOINT, "GAPS-DG-P": checkpoint}
    all_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method, endpoint in endpoints.items():
        scopes = _evaluate(endpoint, device)
        all_metrics[method] = {scope: metrics for scope, (_rows, metrics) in scopes.items()}
        for scope, (rows, metrics) in scopes.items():
            metric_rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "N": metrics["N"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "nll": metrics["nll"],
                    "ece": metrics["ece"],
                    "checkpoint_sha256": sha256_file(endpoint),
                }
            )
            if scope != "C1+C2":
                prediction_rows.extend({"method": method, **row} for row in rows)
    _write_csv(output / "G2_ZERO_SHOT_COMPARISON.csv", metric_rows)
    _write_csv(output / "G2_CLASSIFICATION_PREDICTIONS.csv", prediction_rows)
    representation = [
        _representation_metrics(endpoint, method, device)
        for method, endpoint in endpoints.items()
    ]
    _write_csv(output / "G2_REPRESENTATION_ANALYSIS.csv", representation)
    c5_gain = all_metrics["GAPS-DG-P"]["C5"]["macro_f1"] - all_metrics["FedAvg"]["C5"]["macro_f1"]
    source_drop = all_metrics["FedAvg"]["C1+C2"]["macro_f1"] - all_metrics["GAPS-DG-P"]["C1+C2"]["macro_f1"]
    decision = "SOURCE_DG_SUPPORTED" if c5_gain >= 0.01 and source_drop <= 0.01 else "SOURCE_DG_NOT_SUPPORTED"
    _write_json(
        output / "G2_DECISION.json",
        {"decision": decision, "c5_macro_f1_gain": c5_gain, "merged_source_macro_f1_drop": source_drop},
    )
    if decision == "SOURCE_DG_SUPPORTED":
        (output / "FEDERATED_DG_BASELINE_PROPOSAL.md").write_text(
            "# Federated DG Baseline Proposal\n\nFuture formal comparison may include FedAvg, the existing FedProx/SCAFFOLD endpoints, one classic FDG comparator, and GAPS-DG-P. This Gate did not implement or run additional FDG methods.\n",
            encoding="utf-8",
        )
    report = [
        "# Gate 2 Source-only DG Analysis",
        "",
        f"Decision: `{decision}`.",
        f"C5 Macro-F1 gain: {c5_gain:+.9f}.",
        f"Merged C1+C2 Macro-F1 drop: {source_drop:+.9f}.",
        "",
        "GAPS-DG-P training used C1/C2 X, class, and acquisition-time phase only. No C5 X/Y/phase path was present in any training command.",
    ]
    (output / "G2_RESULT_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "checkpoint_sha256": sha256_file(checkpoint),
            "fedavg_checkpoint_sha256": sha256_file(FEDAVG_CHECKPOINT),
            "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            "target_test_manifest_sha256": sha256_file(LOCAL_DATA_ROOT / "client_5/test_experiment_info.json"),
            "prediction_sha256": sha256_file(output / "G2_CLASSIFICATION_PREDICTIONS.csv"),
            "target_test_selection": False,
            "decision": decision,
        },
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "sha256_index.json")
    _write_json(output / "sha256_index.json", {str(path.relative_to(output)): sha256_file(path) for path in files})
    return {"decision": decision, "c5_macro_f1_gain": c5_gain, "source_drop": source_drop}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if (output / "SEALED_TEST_OPEN.json").exists():
        raise FileExistsError("FAIL_CLOSED G2 evaluation already exists")
    freeze = write_freeze(output)
    execute_fl(output, freeze, args)
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
