"""Run the frozen Phase-1 S4 FedAvg versus GAPS-DG-P multi-seed study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iotj_method_breakthrough_gate_a import (
    LOCAL_DATA_ROOT,
    ROLE_VIEW_AGGREGATE_SHA256,
    _git_head,
    _per_class_rows,
    _sha256_file,
    _write_csv,
    _write_json,
    build_gate_a_commands,
    evidence_files_for_hash_index,
    execute_s4_fl,
    verify_s4_endpoint_locks,
)
from scripts.summarize_iotj_classification_ablation import (
    classification_metrics,
    evaluate_checkpoint_stream,
)


DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase1_s4_dg_multiseed"
GATE_A_ROOT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/gate_a_source_diversity"
PLAN_ROOT = ROOT / "docs/experiments/iotj_canonical_v1_final/method_breakthrough"
FREEZE_COMMIT = "0a0720e"
METHOD_LABELS = {"fedavg": "FedAvg", "gaps_dg_p": "GAPS-DG-P"}
NEW_SEEDS = (41, 43)
ALL_SEEDS = (41, 42, 43)


def phase1_run_specs() -> list[dict[str, Any]]:
    rows = []
    for seed in ALL_SEEDS:
        for method in ("fedavg", "gaps_dg_p"):
            method_id = "FEDAVG" if method == "fedavg" else "DGP"
            rows.append(
                {
                    "phase": "P1",
                    "experiment_id": f"CAN-V1-MB-P1-S4-{method_id}-S{seed}",
                    "method": method,
                    "seed": seed,
                    "execution": "reuse" if seed == 42 else "train",
                }
            )
    return rows


def build_multiseed_commands(method: str, seed: int) -> dict[str, Any]:
    if int(seed) not in ALL_SEEDS:
        raise ValueError(f"unregistered Phase-1 seed: {seed}")
    method_id = "FEDAVG" if method == "fedavg" else "DGP"
    experiment_id = f"CAN-V1-MB-P1-S4-{method_id}-S{int(seed)}"
    return build_gate_a_commands(method, seed=int(seed), experiment_id=experiment_id)


def decide_s4_dg_multiseed(paired_gains: dict[int, float]) -> dict[str, Any]:
    if sorted(int(seed) for seed in paired_gains) != list(ALL_SEEDS):
        raise RuntimeError("FAIL_CLOSED Phase-1 decision requires exactly seeds 41/42/43")
    gains = [float(paired_gains[seed]) for seed in ALL_SEEDS]
    mean_gain = float(statistics.mean(gains))
    sample_sd = float(statistics.stdev(gains))
    all_positive = all(value > 0.0 for value in gains)
    if all_positive and mean_gain >= 0.03 and sample_sd <= 0.05:
        decision = "SOURCE_DG_SUPPORTED"
    elif mean_gain > 0.0 and (not all_positive or sample_sd > 0.05):
        decision = "SOURCE_DG_UNSTABLE"
    else:
        decision = "SOURCE_DG_NOT_CONFIRMED"
    return {
        "decision": decision,
        "evaluated_seeds": list(ALL_SEEDS),
        "paired_macro_f1_gains": {str(seed): float(paired_gains[seed]) for seed in ALL_SEEDS},
        "all_seeds_positive": all_positive,
        "mean_gain": mean_gain,
        "sample_sd": sample_sd,
        "thresholds": {"minimum_mean_gain": 0.03, "maximum_sample_sd": 0.05},
        "next_action": "STOP_SEED_EXPANSION_AND_ENTER_PHASE2",
    }


def _checkpoint_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> Path:
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = (manifest_path.parent / checkpoint).resolve()
    return checkpoint


def verify_new_endpoint_locks(output: Path) -> dict[tuple[int, str], dict[str, Any]]:
    locked: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in NEW_SEEDS:
        for method in ("fedavg", "gaps_dg_p"):
            directory = Path(output) / f"seed_{seed}" / method
            marker_path = directory / "fixed_endpoint_complete.json"
            manifest_path = directory / "run_manifest.json"
            if not marker_path.is_file() or not manifest_path.is_file():
                raise RuntimeError(f"FAIL_CLOSED missing Phase-1 endpoint lock: seed{seed}/{method}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            protocol = manifest.get("protocol", {})
            if marker.get("fixed_endpoint", {}).get("round") != 25:
                raise RuntimeError(f"FAIL_CLOSED endpoint is not round25: seed{seed}/{method}")
            if int(protocol.get("seed", -1)) != seed:
                raise RuntimeError(f"FAIL_CLOSED endpoint seed mismatch: seed{seed}/{method}")
            if protocol.get("target_access") != "NONE" or manifest.get("target_test_opened") is not False:
                raise RuntimeError(f"FAIL_CLOSED target access before lock: seed{seed}/{method}")
            if protocol.get("checkpoint_selection") != "fixed_round_25":
                raise RuntimeError(f"FAIL_CLOSED checkpoint selection mismatch: seed{seed}/{method}")
            checkpoint = _checkpoint_from_manifest(manifest, manifest_path)
            if not checkpoint.is_file() or _sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
                raise RuntimeError(f"FAIL_CLOSED checkpoint hash mismatch: seed{seed}/{method}")
            locked[(seed, method)] = {**manifest, "checkpoint": str(checkpoint)}
    if len(locked) != 4:
        raise RuntimeError("FAIL_CLOSED Phase-1 requires exactly four new endpoint locks")
    return locked


def audit_seed42_reuse() -> dict[tuple[int, str], dict[str, Any]]:
    gate_a = verify_s4_endpoint_locks(GATE_A_ROOT)
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for method, manifest in gate_a.items():
        run_manifest_path = GATE_A_ROOT / method / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        protocol = run_manifest.get("protocol", {})
        if protocol.get("seed") != 42 or protocol.get("target_access") != "NONE":
            raise RuntimeError(f"FAIL_CLOSED seed42 reuse protocol mismatch: {method}")
        checkpoint = Path(manifest["checkpoint"])
        if _sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"FAIL_CLOSED seed42 reuse checkpoint mismatch: {method}")
        result[(42, method)] = {**run_manifest, "checkpoint": str(checkpoint)}
    return result


def phase1_protocol_hash() -> str:
    payload = {
        "runs": phase1_run_specs(),
        "new_commands": {
            f"seed_{seed}/{method}": build_multiseed_commands(method, seed)
            for seed in NEW_SEEDS
            for method in ("fedavg", "gaps_dg_p")
        },
        "decision_thresholds": {"mean_gain": 0.03, "sample_sd": 0.05},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_pre_run_freeze(output: Path) -> dict[str, Any]:
    for required in ("EXPERIMENT_PLAN.md", "EXPERIMENT_MATRIX.csv", "EXPERIMENT_REGISTRY.csv", "PRE_EXECUTION_AUDIT.md"):
        if not (PLAN_ROOT / required).is_file():
            raise RuntimeError(f"FAIL_CLOSED missing frozen plan artifact: {required}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"], cwd=ROOT
    )
    if ancestor.returncode != 0:
        raise RuntimeError("FAIL_CLOSED pre-run freeze commit is not in HEAD ancestry")
    seed42 = audit_seed42_reuse()
    payload = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.phase1.freeze.v1",
        "status": "FROZEN",
        "freeze_commit": _git_head(),
        "planning_freeze_commit": FREEZE_COMMIT,
        "protocol_hash": phase1_protocol_hash(),
        "dataset": "iotj_canonical_v1_s4_role_view",
        "dataset_aggregate_sha256": ROLE_VIEW_AGGREGATE_SHA256,
        "run_specs": phase1_run_specs(),
        "new_commands": {
            f"seed_{seed}/{method}": build_multiseed_commands(method, seed)
            for seed in NEW_SEEDS
            for method in ("fedavg", "gaps_dg_p")
        },
        "seed42_reuse": {
            method: {
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            }
            for (_seed, method), manifest in seed42.items()
        },
        "target_access": "NONE_DURING_SOURCE_TRAINING",
        "target_test_opened": False,
        "hyperparameter_search": False,
        "additional_seed_expansion": False,
    }
    path = Path(output) / "PRE_RUN_FREEZE.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("FAIL_CLOSED Phase-1 pre-run freeze differs")
    else:
        _write_json(path, payload)
    return payload


def _write_progress(output: Path, *, status: str, completed: list[str], active: str | None) -> None:
    _write_json(
        Path(output) / "RUN_PROGRESS.json",
        {
            "status": status,
            "completed_new_endpoints": completed,
            "completed_count": len(completed),
            "required_new_endpoint_count": 4,
            "active_endpoint": active,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def train_new_endpoints(output: Path, freeze: dict[str, Any], args: argparse.Namespace) -> None:
    completed = []
    for spec in phase1_run_specs():
        if spec["execution"] != "train":
            continue
        endpoint = f"seed_{spec['seed']}/{spec['method']}"
        marker = Path(output) / endpoint / "fixed_endpoint_complete.json"
        if marker.is_file():
            completed.append(endpoint)
            continue
        _write_progress(output, status="RUNNING", completed=completed, active=endpoint)
        execute_s4_fl(
            spec["method"],
            output,
            freeze,
            ecs_host=args.ecs_host,
            pi_host=args.pi_host,
            c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
            seed=spec["seed"],
            experiment_id=spec["experiment_id"],
            run_subdir=endpoint,
        )
        completed.append(endpoint)
    verify_new_endpoint_locks(output)
    _write_progress(output, status="ALL_ENDPOINTS_LOCKED", completed=completed, active=None)


def _metric_row(seed: int, method: str, scope: str, metrics: dict[str, Any], checkpoint_sha: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "method": METHOD_LABELS[method],
        "scope": scope,
        "N": metrics["N"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "nll": metrics["nll"],
        "ece": metrics["ece"],
        "checkpoint_sha256": checkpoint_sha,
    }


def evaluate_and_analyze(output: Path, device: torch.device) -> dict[str, Any]:
    output = Path(output).resolve()
    locked = {**verify_new_endpoint_locks(output), **audit_seed42_reuse()}
    if set(locked) != {(seed, method) for seed in ALL_SEEDS for method in ("fedavg", "gaps_dg_p")}:
        raise RuntimeError("FAIL_CLOSED Phase-1 endpoint set differs before sealed test")
    _write_json(
        output / "SEALED_TEST_OPEN.json",
        {
            "status": "OPENED_AFTER_ALL_SIX_ENDPOINTS_AUDITED",
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_test_selection": False,
        },
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    c5_f1: dict[tuple[int, str], float] = {}
    for seed in ALL_SEEDS:
        for method in ("fedavg", "gaps_dg_p"):
            manifest = locked[(seed, method)]
            checkpoint = Path(manifest["checkpoint"])
            source_rows: list[dict[str, Any]] = []
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
                metric_rows.append(_metric_row(seed, method, f"C{client}", metrics, manifest["checkpoint_sha256"]))
                if client < 5:
                    source_rows.extend(rows)
                else:
                    prediction_rows.extend({"seed": seed, "method": METHOD_LABELS[method], **row} for row in rows)
                    c5_f1[(seed, method)] = float(metrics["macro_f1"])
                    per_class_rows.extend(
                        {"seed": seed, **row}
                        for row in _per_class_rows(METHOD_LABELS[method], metrics)
                    )
            probabilities = np.asarray(
                [[float(row[f"prob_{index}"]) for index in range(4)] for row in source_rows], dtype=np.float64
            )
            source_metrics = classification_metrics([int(row["true_class"]) for row in source_rows], probabilities)
            metric_rows.append(_metric_row(seed, method, "SOURCE_POOLED", source_metrics, manifest["checkpoint_sha256"]))
    _write_csv(output / "S4_DG_MULTI_SEED.csv", metric_rows)
    _write_csv(output / "S4_DG_MULTI_SEED_C5_PER_CLASS.csv", per_class_rows)
    _write_csv(output / "S4_DG_MULTI_SEED_PREDICTIONS.csv", prediction_rows)
    gains = {seed: c5_f1[(seed, "gaps_dg_p")] - c5_f1[(seed, "fedavg")] for seed in ALL_SEEDS}
    decision = decide_s4_dg_multiseed(gains)
    _write_json(output / "PHASE1_DECISION.json", decision)
    comparison_lines = "\n".join(
        f"| {seed} | {c5_f1[(seed, 'fedavg')]:.6f} | {c5_f1[(seed, 'gaps_dg_p')]:.6f} | {gains[seed]:+.6f} |"
        for seed in ALL_SEEDS
    )
    report = f"""# S4 DG Multi-seed Confirmation Report

## Protocol

FedAvg and the exact frozen DG-P mechanism were compared under S4 C1-C4 source-only training for seeds 41/42/43. Seed42 is immutable Gate-A reuse; seeds41/43 are new fixed-round25 runs. C5 was unavailable to all training APIs and was evaluated only after all endpoints were locked.

## C5 result

| Seed | FedAvg Macro-F1 | GAPS-DG-P Macro-F1 | Paired gain |
|---:|---:|---:|---:|
{comparison_lines}

- Mean paired gain: {decision['mean_gain']:+.6f}
- Paired-gain sample SD: {decision['sample_sd']:.6f}
- Decision: `{decision['decision']}`

## Scope and limitation

This is a registered three-seed C5 hardest-target source-DG confirmation. It does not establish universal cross-target superiority, and no additional seeds are authorized after observing the result.

## Next action

`{decision['next_action']}` under the already frozen Phase-2 matrix.
"""
    (output / "S4_DG_MULTI_SEED_REPORT.md").write_text(report, encoding="utf-8")
    (output / "EXPERIMENT_AUDIT.md").write_text(
        "# Phase-1 post-run audit\n\n"
        "- Exactly four new endpoints and two immutable seed42 reuse endpoints were evaluated.\n"
        "- Every endpoint is fixed round25 and checkpoint-hash verified.\n"
        "- C5 was absent from source training and opened only after endpoint lock.\n"
        "- No target-test checkpoint selection, hyperparameter search, or seed expansion occurred.\n",
        encoding="utf-8",
    )
    _write_json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "protocol_hash": phase1_protocol_hash(),
            "dataset_aggregate_sha256": ROLE_VIEW_AGGREGATE_SHA256,
            "target_test_selection": False,
            "evaluated_seeds": list(ALL_SEEDS),
            "decision": decision,
        },
    )
    _write_json(
        output / "sha256_index.json",
        {
            str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
            for path in evidence_files_for_hash_index(output)
        },
    )
    _write_progress(output, status="COMPLETE", completed=[f"seed_{seed}/{method}" for seed in NEW_SEEDS for method in ("fedavg", "gaps_dg_p")], active=None)
    return decision


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    freeze = write_pre_run_freeze(output)
    train_new_endpoints(output, freeze, args)
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
