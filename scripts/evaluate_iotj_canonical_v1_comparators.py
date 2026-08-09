"""One-time sealed-test evaluation for the minimal canonical comparator matrix."""
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

from scripts.summarize_iotj_classification_ablation import classification_metrics, evaluate_checkpoint_stream
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


TARGETS = ("C3", "C4", "C5")
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
VALIDATION_ROOT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809"
COMPARATOR_ROOT = VALIDATION_ROOT / "comparators"
A0T_ROOT = ROOT / "results/iotj_canonical_v1_final_20260808/a0t_equal_label/classification"
GAPS_ROOT = ROOT / "results/iotj_canonical_v1_final_20260808/classification"
DEFAULT_OUTPUT = VALIDATION_ROOT / "classification_comparison"


def method_contracts() -> dict[str, dict[str, Any]]:
    return {
        "FedAvg": {"optimizer": "Adam", "optimizer_lr": 5e-4, "optimizer_note": "frozen GAPS experimental protocol", "target_x": False, "target_y": False, "target_phase": False, "target_concentration": False, "information_regime": "source-only"},
        "FedProx": {"optimizer": "Adam", "optimizer_lr": 5e-4, "optimizer_note": "frozen GAPS experimental protocol", "target_x": False, "target_y": False, "target_phase": False, "target_concentration": False, "information_regime": "source-only"},
        "SCAFFOLD": {"optimizer": "SGD", "optimizer_lr": 5e-4, "optimizer_note": "canonical SCAFFOLD implementation", "target_x": False, "target_y": False, "target_phase": False, "target_concentration": False, "information_regime": "source-only"},
        "MMD": {"optimizer": "Adam", "optimizer_lr": 5e-4, "optimizer_note": "canonical post-hoc global MMD2", "target_x": True, "target_y": False, "target_phase": False, "target_concentration": False, "information_regime": "unlabeled x-only"},
        "A0T": {"optimizer": "Adam", "optimizer_lr": 5e-4, "optimizer_note": "equal-label target-CE-only commissioning", "target_x": True, "target_y": True, "target_phase": False, "target_concentration": False, "information_regime": "equal-label supervised commissioning"},
        "GAPS/A4": {"optimizer": "Adam", "optimizer_lr": 5e-4, "optimizer_note": "proposed method protocol", "target_x": True, "target_y": True, "target_phase": True, "target_concentration": False, "information_regime": "labeled structured commissioning; target CE weight zero"},
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_complete(run_dir: Path, expected_id: str, endpoint_kind: str) -> tuple[Path, dict[str, Any]]:
    marker_path, manifest_path = run_dir / "fixed_endpoint_complete.json", run_dir / "run_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"FAIL_CLOSED incomplete endpoint: {expected_id}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if marker.get("experiment_id") != expected_id or manifest.get("experiment_id") != expected_id:
        raise RuntimeError(f"FAIL_CLOSED endpoint identity differs: {expected_id}")
    endpoint = marker.get("fixed_endpoint", {})
    if endpoint_kind == "round25" and int(endpoint.get("round", -1)) != 25:
        raise RuntimeError(f"FAIL_CLOSED endpoint is not round25: {expected_id}")
    if endpoint_kind == "step100" and int(endpoint.get("steps", -1)) != 100:
        raise RuntimeError(f"FAIL_CLOSED endpoint is not step100: {expected_id}")
    if marker.get("target_test_opened") is not False or manifest.get("target_test_opened") is not False:
        raise RuntimeError(f"FAIL_CLOSED test opened before matrix gate: {expected_id}")
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError(f"FAIL_CLOSED checkpoint provenance differs: {expected_id}")
    return checkpoint, {"checkpoint_sha256": manifest["checkpoint_sha256"], "marker_sha256": sha256(marker_path), "manifest_sha256": sha256(manifest_path)}


def completion_gate() -> dict[str, dict[str, dict[str, Any]]]:
    gated: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in method_contracts()}
    for method in ("FedAvg", "FedProx", "SCAFFOLD"):
        experiment_id = f"CAN-V1-CMP-{method.upper()}"
        checkpoint, provenance = _load_complete(COMPARATOR_ROOT / "source_fl" / experiment_id, experiment_id, "round25")
        for target in TARGETS:
            gated[method][target] = {"checkpoint": checkpoint, **provenance}
    for target in TARGETS:
        experiment_id = f"CAN-V1-CMP-MMD-{target}"
        checkpoint, provenance = _load_complete(COMPARATOR_ROOT / "mmd" / experiment_id, experiment_id, "step100")
        gated["MMD"][target] = {"checkpoint": checkpoint, **provenance}
        experiment_id = f"CANONICAL-V1-A0T-{target}"
        checkpoint, provenance = _load_complete(A0T_ROOT / experiment_id, experiment_id, "round25")
        gated["A0T"][target] = {"checkpoint": checkpoint, **provenance}
        experiment_id = f"CANONICAL-V1-A4-{target}"
        checkpoint, provenance = _load_complete(GAPS_ROOT / experiment_id, experiment_id, "round25")
        gated["GAPS/A4"][target] = {"checkpoint": checkpoint, **provenance}
    return gated


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[f"prob_{class_id}"]) for class_id in range(4)] for row in rows])


def _metric_row(method: str, target: str, rows: list[dict[str, Any]], source_macro_f1: float | str, checkpoint_sha256: str) -> dict[str, Any]:
    metrics = classification_metrics([int(row["true_class"]) for row in rows], _probabilities(rows))
    contract = method_contracts()[method]
    return {
        "method": method, "target": target, "N": int(metrics["N"]),
        "accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"]),
        "nll": float(metrics["nll"]), "ece": float(metrics["ece"]),
        "source_macro_f1": source_macro_f1,
        "source_target_f1_gap": float(source_macro_f1) - float(metrics["macro_f1"]) if source_macro_f1 != "" else "",
        "checkpoint_sha256": checkpoint_sha256, "seed": 42,
        "rounds": 25, "local_epochs": 1, "batch_size": 32,
        **contract,
        "selection_role": "fixed_endpoint_one_time_sealed_test",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output: Path, device: torch.device) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED comparator evaluation output exists: {output}")
    gate = completion_gate()
    before = verify_dataset(DATA_ROOT)
    if before["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED canonical dataset verification failed")
    output.mkdir(parents=True)
    (output / "SEALED_TEST_OPEN.json").write_text(json.dumps({
        "status": "OPENED_AFTER_ALL_FIXED_ENDPOINTS", "methods": list(method_contracts()),
        "targets": list(TARGETS), "selection_performed": False, "hyperparameter_search": False,
    }, indent=2) + "\n", encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method in method_contracts():
        all_rows: list[dict[str, Any]] = []
        for target in TARGETS:
            item = gate[method][target]
            target_rows, _ = evaluate_checkpoint_stream(item["checkpoint"], data_root=DATA_ROOT, target_client=int(target[1:]), split="test", device=device, batch_size=32)
            source_rows: list[dict[str, Any]] = []
            for client in (1, 2):
                rows, _ = evaluate_checkpoint_stream(item["checkpoint"], data_root=DATA_ROOT, target_client=client, split="test", device=device, batch_size=32)
                source_rows.extend(rows)
            source_metrics = classification_metrics([int(row["true_class"]) for row in source_rows], _probabilities(source_rows))
            metric_rows.append(_metric_row(method, target, target_rows, float(source_metrics["macro_f1"]), item["checkpoint_sha256"]))
            for row in target_rows:
                prediction_rows.append({"method": method, "target": target, "checkpoint_sha256": item["checkpoint_sha256"], **row})
            all_rows.extend(target_rows)
        metric_rows.append(_metric_row(method, "ALL", all_rows, "", "target_specific_or_shared_as_manifested"))
    write_csv(output / "canonical_classification_comparison.csv", metric_rows)
    write_csv(output / "canonical_classification_predictions.csv", prediction_rows)
    after = verify_dataset(DATA_ROOT)
    if after != before:
        raise RuntimeError("FAIL_CLOSED canonical dataset changed during evaluation")
    manifest = {"schema_version": "iotj.canonical_v1.comparator_evaluation.v1", "status": "PASS", "dataset": after, "selection_performed": False, "gate": {method: {target: {key: str(value) for key, value in item.items() if key != "checkpoint"} for target, item in targets.items()} for method, targets in gate.items()}}
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "rows": len(metric_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.output, torch.device(args.device)), indent=2))


if __name__ == "__main__":
    main()
