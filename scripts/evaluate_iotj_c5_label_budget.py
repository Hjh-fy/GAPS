"""Open the frozen C5 test once after all six low-label endpoints complete."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_iotj_c5_label_budget import BUDGETS, METHODS, experiment_id
from scripts.summarize_iotj_classification_ablation import (
    classification_metrics,
    evaluate_checkpoint_stream,
)
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


STUDY_ROOT = ROOT / "results/iotj_canonical_v1_c5_budget_20260810"
RUN_ROOT = STUDY_ROOT / "classification"
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
DEFAULT_OUTPUT = STUDY_ROOT / "evaluation"
FEDAVG_MANIFEST = (
    ROOT / "results/iotj_canonical_v1_scientific_validation_20260809"
    / "comparators/source_fl/CAN-V1-CMP-FEDAVG/run_manifest.json"
)
EXISTING_20_MANIFESTS = {
    "A0T": ROOT / "results/iotj_canonical_v1_final_20260808/a0t_equal_label/classification/CANONICAL-V1-A0T-C5/run_manifest.json",
    "A4": ROOT / "results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C5/run_manifest.json",
}
CLASS_NAMES = ("ethanol", "carbon_monoxide", "ethylene", "methane")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_gate(run_root: Path, expected_protocol_hash: str) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for budget in BUDGETS:
            run_id = experiment_id(method, budget)
            run = run_root / run_id
            marker_path = run / "fixed_endpoint_complete.json"
            manifest_path = run / "run_manifest.json"
            if not marker_path.is_file() or not manifest_path.is_file():
                raise RuntimeError(f"FAIL_CLOSED completion evidence missing: {run_id}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            protocol = manifest.get("protocol", {})
            endpoint = marker.get("fixed_endpoint", {})
            if marker.get("experiment_id") != run_id or manifest.get("experiment_id") != run_id:
                raise RuntimeError(f"FAIL_CLOSED experiment identity differs: {run_id}")
            if marker.get("protocol_hash") != expected_protocol_hash or manifest.get("protocol_hash") != expected_protocol_hash:
                raise RuntimeError(f"FAIL_CLOSED protocol hash differs: {run_id}")
            if int(endpoint.get("round", -1)) != 25:
                raise RuntimeError(f"FAIL_CLOSED endpoint is not round25: {run_id}")
            if marker.get("target_test_opened") is not False or manifest.get("target_test_opened") is not False:
                raise RuntimeError(f"FAIL_CLOSED target test opened early: {run_id}")
            required = {
                "method": method,
                "budget_pct": budget,
                "target": "C5",
                "rounds": 25,
                "local_epochs": 1,
                "seed": 42,
                "checkpoint_reuse": False,
                "checkpoint_selection": "fixed_round_25",
                "target_test_selection": False,
            }
            if any(protocol.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"FAIL_CLOSED protocol differs: {run_id}")
            checkpoint = Path(manifest["checkpoint"])
            if not checkpoint.is_file() or sha256(checkpoint) != manifest.get("checkpoint_sha256"):
                raise RuntimeError(f"FAIL_CLOSED checkpoint SHA differs: {run_id}")
            runs[run_id] = {
                "method": method,
                "budget_pct": budget,
                "checkpoint": checkpoint,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "marker_sha256": sha256(marker_path),
                "manifest_sha256": sha256(manifest_path),
            }
    return {"status": "PASS", "runs": runs, "protocol_hash": expected_protocol_hash}


def per_class_rows(
    confusion_matrix: Sequence[Sequence[int]], *, class_names: Sequence[str] = CLASS_NAMES
) -> list[dict[str, Any]]:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion matrix/class name dimensions differ")
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        tp = int(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class_id": class_id,
            "class_name": str(class_name),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return rows


def probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([
        [float(row[f"prob_{class_id}"]) for class_id in range(4)]
        for row in rows
    ])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_retention_row(
    *,
    method: str,
    budget: int,
    source_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "budget_pct": budget,
        "source_N": int(source_metrics["N"]),
        "source_accuracy": float(source_metrics["accuracy"]),
        "source_macro_f1": float(source_metrics["macro_f1"]),
        "fedavg_source_accuracy": float(baseline_metrics["accuracy"]),
        "fedavg_source_macro_f1": float(baseline_metrics["macro_f1"]),
        "source_accuracy_retention_delta": float(source_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "source_macro_f1_retention_delta": float(source_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _merged_source(checkpoint: Path, device: torch.device, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for client in (1, 2):
        client_rows, _ = evaluate_checkpoint_stream(
            checkpoint,
            data_root=DATA_ROOT,
            target_client=client,
            split="test",
            device=device,
            batch_size=batch_size,
        )
        rows.extend(client_rows)
    if len(rows) != 1360:
        raise RuntimeError(f"FAIL_CLOSED merged source test N differs: {len(rows)}")
    return rows, classification_metrics(
        [int(row["true_class"]) for row in rows], probabilities(rows)
    )


def _fedavg_source(device: torch.device, batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(FEDAVG_MANIFEST.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("FAIL_CLOSED frozen FedAvg source checkpoint SHA differs")
    _rows, metrics = _merged_source(checkpoint, device, batch_size)
    return metrics, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "manifest_sha256": sha256(FEDAVG_MANIFEST),
    }


def run(output: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED C5 budget evaluation already exists: {output}")
    freeze = json.loads((STUDY_ROOT / "PRE_RUN_FREEZE.json").read_text(encoding="utf-8"))
    gate = completion_gate(RUN_ROOT, str(freeze["protocol_hash"]))
    dataset_before = verify_dataset(DATA_ROOT)
    if dataset_before["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED canonical dataset hash failed before evaluation")
    output.mkdir(parents=True)
    (output / "SEALED_TEST_OPEN.json").write_text(
        json.dumps({
            "schema_version": "gaps.iotj.c5_label_budget.sealed_test.v1",
            "status": "OPENED_AFTER_ALL_SIX_FIXED_ENDPOINTS",
            "selection_performed": False,
            "hyperparameter_search": False,
            "gate": {
                "status": gate["status"],
                "protocol_hash": gate["protocol_hash"],
                "run_ids": list(gate["runs"]),
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_source, baseline_provenance = _fedavg_source(device, batch_size)
    target_rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    source_retention: list[dict[str, Any]] = []
    for method, manifest_path in EXISTING_20_MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"FAIL_CLOSED existing 20% {method} checkpoint SHA differs")
        _source_rows, source_metrics = _merged_source(checkpoint, device, batch_size)
        source_retention.append(source_retention_row(
            method=method,
            budget=20,
            source_metrics=source_metrics,
            baseline_metrics=baseline_source,
            checkpoint_sha256=manifest["checkpoint_sha256"],
        ))
    per_class: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confusion_dir = output / "c5_budget_confusion_matrices"
    confusion_dir.mkdir()
    for run_id, item in gate["runs"].items():
        method = str(item["method"])
        budget = int(item["budget_pct"])
        checkpoint = Path(item["checkpoint"])
        target_rows, target_metrics = evaluate_checkpoint_stream(
            checkpoint,
            data_root=DATA_ROOT,
            target_client=5,
            split="test",
            device=device,
            batch_size=batch_size,
        )
        if len(target_rows) != 1360:
            raise RuntimeError(f"FAIL_CLOSED C5 sealed test N differs: {run_id}")
        _source_rows, source_metrics = _merged_source(checkpoint, device, batch_size)
        metric_row = {
            "method": method,
            "budget_pct": budget,
            "calibration_n": {15: 240, 10: 160, 5: 80}[budget],
            "N": int(target_metrics["N"]),
            "accuracy": float(target_metrics["accuracy"]),
            "macro_f1": float(target_metrics["macro_f1"]),
            "nll": float(target_metrics["nll"]),
            "ece": float(target_metrics["ece"]),
            "mean_confidence": float(target_metrics["mean_confidence"]),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "seed": 42,
            "selection_role": "fixed_round25_one_time_sealed_test",
        }
        target_rows_by_method[method].append(metric_row)
        source_retention.append(source_retention_row(
            method=method,
            budget=budget,
            source_metrics=source_metrics,
            baseline_metrics=baseline_source,
            checkpoint_sha256=str(item["checkpoint_sha256"]),
        ))
        class_rows = per_class_rows(target_metrics["confusion_matrix"])
        per_class.extend([
            {"method": method, "budget_pct": budget, **row}
            for row in class_rows
        ])
        (confusion_dir / f"{run_id}.json").write_text(
            json.dumps({
                "method": method,
                "budget_pct": budget,
                "class_names": list(CLASS_NAMES),
                "confusion_matrix": target_metrics["confusion_matrix"],
                "per_class": class_rows,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        for row in target_rows:
            prediction_rows.append({
                **row,
                "method": method,
                "budget_pct": budget,
                "checkpoint_sha256": item["checkpoint_sha256"],
            })

    write_csv(output / "c5_budget_a0t_metrics.csv", sorted(target_rows_by_method["A0T"], key=lambda row: -int(row["budget_pct"])))
    write_csv(output / "c5_budget_a4_metrics.csv", sorted(target_rows_by_method["A4"], key=lambda row: -int(row["budget_pct"])))
    write_csv(output / "c5_budget_source_retention.csv", sorted(source_retention, key=lambda row: (str(row["method"]), -int(row["budget_pct"]))))
    write_csv(output / "c5_budget_per_class_metrics.csv", sorted(per_class, key=lambda row: (str(row["method"]), -int(row["budget_pct"]), int(row["class_id"]))))
    write_csv(output / "c5_budget_predictions.csv", prediction_rows)
    dataset_after = verify_dataset(DATA_ROOT)
    if dataset_after != dataset_before:
        raise RuntimeError("FAIL_CLOSED canonical dataset changed during sealed evaluation")
    manifest = {
        "schema_version": "gaps.iotj.c5_label_budget.evaluation.v1",
        "status": "PASS",
        "target_test_opened_after_all_six_endpoints": True,
        "selection_performed": False,
        "dataset": dataset_after,
        "gate": {
            "protocol_hash": gate["protocol_hash"],
            "runs": {
                run_id: {key: str(value) for key, value in item.items()}
                for run_id, item in gate["runs"].items()
            },
        },
        "fedavg_source_reference": baseline_provenance,
        "outputs": {
            "a0t": "c5_budget_a0t_metrics.csv",
            "a4": "c5_budget_a4_metrics.csv",
            "source_retention": "c5_budget_source_retention.csv",
            "per_class": "c5_budget_per_class_metrics.csv",
            "confusion_matrices": "c5_budget_confusion_matrices/",
            "predictions": "c5_budget_predictions.csv",
        },
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"status": "PASS", "runs": 6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output, torch.device(args.device), args.batch_size), indent=2))


if __name__ == "__main__":
    main()
