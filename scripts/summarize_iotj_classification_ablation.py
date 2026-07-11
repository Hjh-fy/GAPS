"""Evaluate and summarize frozen C12-to-C5 classification checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from gaps_flower.evaluate_checkpoint import (
    expected_calibration_error,
    load_checkpoint_model,
    make_loader,
    resolve_device,
)


DATA_ROOT = Path("dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid")
DEFAULT_RUN_ROOT = Path("results/iotj_classification_ablation_20260711")
DEFAULT_OUTPUT_ROOT = Path("results/iotj_classification_ablation_20260711_summary")
FINAL_ROUND = 25
NUM_CLASSES = 4
CONFIRMATION_GROUPS = ("A0", "A4", "A5", "A7")
CONFIRMATION_SEEDS = (42, 43, 44, 45, 46)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_identity(run_dir: Path) -> tuple[str, int]:
    match = re.match(r"(A[0-7])_.*_s(\d+)_r25$", run_dir.name)
    if not match:
        raise ValueError(f"cannot parse ablation identity from {run_dir.name}")
    return match.group(1), int(match.group(2))


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"run_config args must be an object: {path}")
    return args


def resolve_final_checkpoint(run_dir: Path, run_config: dict[str, Any]) -> tuple[Path, str]:
    use_da = bool(run_config.get("use_domain_adapt"))
    if use_da:
        path = run_dir / "server_latest_adapted.pth"
        label = "final_adapted"
    else:
        path = run_dir / "server_latest.pth"
        label = "final_aggregated"
    if not path.is_file():
        raise FileNotFoundError(path)
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    return path, label


def classification_metrics(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
    *,
    ece_bins: int = 15,
) -> dict[str, Any]:
    true = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (len(true), NUM_CLASSES):
        raise ValueError(f"probability shape mismatch: {probs.shape} vs {(len(true), NUM_CLASSES)}")
    pred = probs.argmax(axis=1)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_id, pred_id in zip(true, pred):
        confusion[int(true_id), int(pred_id)] += 1
    per_class_recall: dict[str, float | None] = {}
    per_class_f1: dict[str, float | None] = {}
    valid_f1: list[float] = []
    for class_id in range(NUM_CLASSES):
        tp = float(confusion[class_id, class_id])
        fn = float(confusion[class_id].sum() - tp)
        fp = float(confusion[:, class_id].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        per_class_recall[str(class_id)] = recall
        per_class_f1[str(class_id)] = f1
        if f1 is not None:
            valid_f1.append(f1)
    indices = np.arange(len(true))
    true_probs = np.clip(probs[indices, true], 1e-12, 1.0)
    confidence = probs.max(axis=1)
    correct = pred == true
    ece = expected_calibration_error(
        torch.from_numpy(confidence), torch.from_numpy(correct), ece_bins
    )
    return {
        "N": int(len(true)),
        "accuracy": float(correct.mean()) if len(true) else 0.0,
        "macro_f1": float(np.mean(valid_f1)) if valid_f1 else 0.0,
        "nll": float(-np.log(true_probs).mean()) if len(true) else 0.0,
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()) if len(true) else 0.0,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_checkpoint_stream(
    checkpoint_path: Path,
    *,
    data_root: Path,
    split: str,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, config, checkpoint = load_checkpoint_model(
        str(checkpoint_path), device, batch_size
    )
    if int(checkpoint.get("round", -1)) != FINAL_ROUND:
        raise ValueError(f"expected round {FINAL_ROUND}: {checkpoint_path}")
    loader = make_loader(data_root, 5, split, config.BATCH_SIZE)
    rows: list[dict[str, Any]] = []
    all_true: list[int] = []
    all_probs: list[np.ndarray] = []
    sample_index = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            true = batch[1].long().to(device)
            logits, _, _ = model(x)
            probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, k=2, dim=1).values
            confidence, pred = probs.max(dim=1)
            margin = top2[:, 0] - top2[:, 1]
            logits_np = logits.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            true_np = true.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            confidence_np = confidence.detach().cpu().numpy()
            margin_np = margin.detach().cpu().numpy()
            for idx in range(len(true_np)):
                row = {
                    "client": "C5",
                    "split": split,
                    "sample_index": sample_index,
                    "true_class": int(true_np[idx]),
                    "pred_class": int(pred_np[idx]),
                    "route_correct": int(true_np[idx] == pred_np[idx]),
                    "confidence": float(confidence_np[idx]),
                    "margin": float(margin_np[idx]),
                }
                for class_id in range(NUM_CLASSES):
                    row[f"logit_{class_id}"] = float(logits_np[idx, class_id])
                    row[f"prob_{class_id}"] = float(probs_np[idx, class_id])
                rows.append(row)
                sample_index += 1
            all_true.extend(true_np.tolist())
            all_probs.append(probs_np)
    probabilities = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, NUM_CLASSES))
    return rows, classification_metrics(all_true, probabilities)


def evaluate_run(
    run_dir: Path,
    *,
    data_root: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    group_id, seed = _run_identity(run_dir)
    run_config = _load_run_config(run_dir)
    if int(run_config.get("rounds", -1)) != FINAL_ROUND:
        raise ValueError(f"run_config is not round 25: {run_dir}")
    if int(run_config.get("seed", -1)) != seed:
        raise ValueError(f"run_config seed mismatch: {run_dir}")
    checkpoint, checkpoint_label = resolve_final_checkpoint(run_dir, run_config)
    run_output = output_root / run_dir.name
    split_metrics: dict[str, Any] = {}
    for split in ("calibration", "test"):
        rows, metrics = evaluate_checkpoint_stream(
            checkpoint,
            data_root=data_root,
            split=split,
            device=device,
            batch_size=batch_size,
        )
        _write_csv(run_output / f"classification_{split}_stream.csv", rows)
        split_metrics[split] = metrics
    payload = {
        "run_name": run_dir.name,
        "group_id": group_id,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_label": checkpoint_label,
        "round": FINAL_ROUND,
        "target_clients": [5],
        "metrics": split_metrics,
    }
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / "classification_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def flatten_test_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]["test"]
    row = {
        "run_name": payload["run_name"],
        "group_id": payload["group_id"],
        "seed": payload["seed"],
        "checkpoint_label": payload["checkpoint_label"],
        "round": payload["round"],
        "N": metrics["N"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "nll": metrics["nll"],
        "ece": metrics["ece"],
        "mean_confidence": metrics["mean_confidence"],
    }
    for class_id, value in metrics["per_class_recall"].items():
        row[f"recall_{class_id}"] = value
    row["confusion_matrix_json"] = json.dumps(metrics["confusion_matrix"])
    return row


def aggregate_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = ("accuracy", "macro_f1", "nll", "ece", "mean_confidence")
    for group_id in sorted({str(row["group_id"]) for row in rows}):
        selected = [row for row in rows if row["group_id"] == group_id]
        summary: dict[str, Any] = {
            "group_id": group_id,
            "num_seeds": len(selected),
            "seeds": ",".join(str(row["seed"]) for row in sorted(selected, key=lambda item: int(item["seed"]))),
        }
        for key in metrics:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def validate_confirmation_seeds(rows: Sequence[dict[str, Any]]) -> None:
    for group_id in CONFIRMATION_GROUPS:
        found = sorted(int(row["seed"]) for row in rows if row["group_id"] == group_id)
        if found != list(CONFIRMATION_SEEDS):
            raise ValueError(
                f"{group_id} confirmation seeds must equal {list(CONFIRMATION_SEEDS)}; got {found}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--require-confirmation-seeds", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    run_dirs = sorted(
        path
        for path in args.run_root.iterdir()
        if path.is_dir() and re.match(r"A[0-7]_.*_s\d+_r25$", path.name)
    ) if args.run_root.is_dir() else []
    if not run_dirs:
        raise FileNotFoundError(f"no completed ablation run directories under {args.run_root}")
    payloads = [
        evaluate_run(
            run_dir,
            data_root=args.data_root,
            output_root=args.output_root,
            device=device,
            batch_size=args.batch_size,
        )
        for run_dir in run_dirs
    ]
    rows = [flatten_test_metrics(payload) for payload in payloads]
    if args.require_confirmation_seeds:
        validate_confirmation_seeds(rows)
    summaries = aggregate_groups(rows)
    _write_csv(args.output_root / "classification_per_run.csv", rows)
    _write_csv(args.output_root / "classification_group_summary.csv", summaries)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "groups": len(summaries),
                "device": str(device),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
