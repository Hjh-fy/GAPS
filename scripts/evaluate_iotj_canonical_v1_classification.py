"""Open canonical-v1 sealed tests once after all A4 endpoints complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_iotj_classification_ablation import (
    classification_metrics,
    evaluate_checkpoint_stream,
)
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


TARGETS = ("C3", "C4", "C5")
DATA_ROOT = ROOT / "dataset" / "iotj_canonical_v1"
STUDY_ROOT = ROOT / "results" / "iotj_canonical_v1_final_20260808"
CLASSIFICATION_ROOT = STUDY_ROOT / "classification"
DEFAULT_OUTPUT = STUDY_ROOT / "classification_evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completion_gate(classification_root: Path) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    protocol_hashes: set[str] = set()
    for target in TARGETS:
        run = classification_root / f"CANONICAL-V1-A4-{target}"
        marker_path = run / "fixed_endpoint_complete.json"
        manifest_path = run / "run_manifest.json"
        if not marker_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"FAIL_CLOSED {target} completion evidence missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_id = f"CANONICAL-V1-A4-{target}"
        protocol = manifest.get("protocol", {})
        endpoint = marker.get("fixed_endpoint", {})
        if marker.get("experiment_id") != expected_id or manifest.get("experiment_id") != expected_id:
            raise RuntimeError(f"FAIL_CLOSED {target} experiment identity differs")
        if int(endpoint.get("round", -1)) != 25:
            raise RuntimeError(f"FAIL_CLOSED {target} fixed endpoint is not round25")
        if marker.get("target_test_opened") is not False or manifest.get("target_test_opened") is not False:
            raise RuntimeError(f"FAIL_CLOSED {target} test was opened before final evaluation")
        required = {
            "classifier_router": "A4",
            "local_epochs": 1,
            "rounds": 25,
            "checkpoint_reuse": False,
            "target_test_selection": False,
        }
        if any(protocol.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"FAIL_CLOSED {target} A4 protocol differs")
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError(f"FAIL_CLOSED {target} checkpoint missing")
        if manifest.get("checkpoint_sha256") and sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"FAIL_CLOSED {target} checkpoint SHA256 differs")
        protocol_hashes.add(str(manifest.get("protocol_hash")))
        runs[target] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(checkpoint),
            "marker_sha256": sha256(marker_path),
            "manifest_sha256": sha256(manifest_path),
        }
    if len(protocol_hashes) != 1:
        raise RuntimeError("FAIL_CLOSED target protocol hashes differ")
    return {
        "status": "PASS",
        "targets": list(TARGETS),
        "protocol_hash": next(iter(protocol_hashes)),
        "runs": runs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[f"prob_{class_id}"]) for class_id in range(4)] for row in rows])


def metric_row(scope: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "scope": scope,
        "N": int(metrics["N"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "nll": float(metrics["nll"]),
        "ece": float(metrics["ece"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
        "per_class_recall": json.dumps(metrics["per_class_recall"], separators=(",", ":")),
        "seed": 42,
        "selection_role": "fixed_endpoint_one_time_sealed_test",
        **extra,
    }


def run(output: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED classification evaluation already exists: {output}")
    gate = completion_gate(CLASSIFICATION_ROOT)
    dataset_before = verify_dataset(DATA_ROOT)
    if dataset_before["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED canonical dataset changed before sealed evaluation")
    output.mkdir(parents=True)
    seal = {
        "schema_version": "iotj.canonical_v1.sealed_test_open.v1",
        "status": "OPENED_AFTER_ALL_FIXED_ENDPOINTS",
        "gate": {
            "status": gate["status"],
            "targets": gate["targets"],
            "protocol_hash": gate["protocol_hash"],
        },
        "selection_performed": False,
        "hyperparameter_search": False,
    }
    (output / "SEALED_TEST_OPEN.json").write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")

    target_streams: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        checkpoint = gate["runs"][target]["checkpoint"]
        target_rows, target_metrics = evaluate_checkpoint_stream(
            checkpoint,
            data_root=DATA_ROOT,
            target_client=int(target[1:]),
            split="test",
            device=device,
            batch_size=batch_size,
        )
        expected_n = {"C3": 2677, "C4": 1360, "C5": 1360}[target]
        if len(target_rows) != expected_n:
            raise RuntimeError(f"FAIL_CLOSED {target} sealed-test N differs")
        for row in target_rows:
            row.update({"router_target": target, "checkpoint_sha256": gate["runs"][target]["checkpoint_sha256"]})
        target_streams.extend(target_rows)

        source_rows: list[dict[str, Any]] = []
        for source_client in (1, 2):
            rows, _ = evaluate_checkpoint_stream(
                checkpoint,
                data_root=DATA_ROOT,
                target_client=source_client,
                split="test",
                device=device,
                batch_size=batch_size,
            )
            source_rows.extend(rows)
        source_metrics = classification_metrics(
            [int(row["true_class"]) for row in source_rows], probabilities(source_rows)
        )
        metrics_rows.append(
            metric_row(
                target,
                target_metrics,
                router_target=target,
                checkpoint_sha256=gate["runs"][target]["checkpoint_sha256"],
                source_macro_f1=float(source_metrics["macro_f1"]),
                source_target_f1_gap=float(source_metrics["macro_f1"] - target_metrics["macro_f1"]),
            )
        )

    all_metrics = classification_metrics(
        [int(row["true_class"]) for row in target_streams], probabilities(target_streams)
    )
    metrics_rows.append(
        metric_row(
            "ALL",
            all_metrics,
            router_target="target_specific_A4",
            checkpoint_sha256="C3_C4_C5_target_specific",
            source_macro_f1="",
            source_target_f1_gap="",
        )
    )
    write_csv(output / "classification_predictions.csv", target_streams)
    write_csv(output / "classification_metrics.csv", metrics_rows)
    dataset_after = verify_dataset(DATA_ROOT)
    if dataset_after != dataset_before:
        raise RuntimeError("FAIL_CLOSED canonical dataset changed during sealed evaluation")
    manifest = {
        "schema_version": "iotj.canonical_v1.classification_evaluation.v1",
        "status": "PASS",
        "target_test_opened_after_all_training": True,
        "selection_performed": False,
        "dataset": dataset_after,
        "gate": {
            "protocol_hash": gate["protocol_hash"],
            "runs": {target: {key: str(value) for key, value in item.items()} for target, item in gate["runs"].items()},
        },
        "outputs": {
            "metrics": "classification_metrics.csv",
            "predictions": "classification_predictions.csv",
        },
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "metrics": metrics_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output, torch.device(args.device), args.batch_size), indent=2))


if __name__ == "__main__":
    main()
