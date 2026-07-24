from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from scripts.summarize_iotj_classification_ablation import (
    evaluate_checkpoint_stream,
)


EXPECTED_ROWS = 1360
EXPECTED_CLASSES = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen B5 multi-seed checkpoint on C5 test."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--row-map", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    run_id = f"c12_to_c5__b5__s{args.seed}"
    attempt_id = f"{run_id}__a001"
    rows, metrics = evaluate_checkpoint_stream(
        args.checkpoint,
        data_root=args.data_root,
        target_client=5,
        split="test",
        device=torch.device(args.device),
        batch_size=args.batch_size,
    )
    metadata = json.loads(
        (args.data_root / "client_5" / "test_experiment_info.json").read_text(
            encoding="utf-8"
        )
    )
    row_map_payload = json.loads(args.row_map.read_text(encoding="utf-8"))
    row_map = {
        int(item["runtime_index"]): item for item in row_map_payload["rows"]
    }
    if not (
        len(rows)
        == len(metadata)
        == len(row_map)
        == EXPECTED_ROWS
    ):
        raise ValueError("C5 test, metadata, or row-map row count mismatch")
    for runtime_index, row in enumerate(rows):
        mapping = row_map[runtime_index]
        info = metadata[runtime_index]
        if (
            str(info["filename"]) != str(mapping["filename"])
            or int(info["repeat_id"]) != int(mapping["repeat_id"])
        ):
            raise ValueError(f"row-map metadata mismatch at {runtime_index}")
        reference_index = int(mapping["reference_index"])
        row.update(
            {
                "runtime_index": runtime_index,
                "reference_index": reference_index,
                "filename": str(info["filename"]),
                "repeat_id": int(info["repeat_id"]),
                "row_key": (
                    f"{info['filename']}:{int(info['repeat_id'])}:"
                    f"{reference_index}"
                ),
                "predicted_route": int(row["pred_class"]),
            }
        )
    row_keys = {str(row["row_key"]) for row in rows}
    if len(row_keys) != EXPECTED_ROWS:
        raise ValueError("predicted route row keys are not unique")
    if any(int(row["predicted_route"]) not in range(EXPECTED_CLASSES) for row in rows):
        raise ValueError("predicted route is outside the frozen class set")

    confusion = metrics["confusion_matrix"]
    metrics["per_class_precision"] = {
        str(class_id): (
            float(confusion[class_id][class_id])
            / sum(float(confusion[row][class_id]) for row in range(EXPECTED_CLASSES))
            if sum(
                float(confusion[row][class_id])
                for row in range(EXPECTED_CLASSES)
            )
            else 0.0
        )
        for class_id in range(EXPECTED_CLASSES)
    }

    args.output_dir.mkdir(parents=True)
    prediction_path = (
        args.output_dir / f"seed{args.seed}_test_predictions.csv"
    )
    with prediction_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "iotj.b5_seed_classification_evaluation.v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "split": "C5_test",
        "ece_bins": 15,
        "metrics": metrics,
        "predicted_route_rows": len(rows),
        "unique_row_keys": len(row_keys),
        "row_map_contract_sha256": _sha256(args.runtime_contract),
        "test_used_for_training_selection_or_stopping": False,
    }
    metrics_path = (
        args.output_dir / f"seed{args.seed}_classification_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
