"""Evaluate and compare paired B2/B5 cross-direction checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __package__:
    from scripts.run_iotj_cross_direction_cloud_edge import load_ordered_manifests
    from scripts.summarize_iotj_classification_ablation import (
        _write_csv,
        classification_metrics,
        evaluate_run,
        flatten_test_metrics,
    )
else:
    from run_iotj_cross_direction_cloud_edge import load_ordered_manifests
    from summarize_iotj_classification_ablation import (
        _write_csv,
        classification_metrics,
        evaluate_run,
        flatten_test_metrics,
    )

from gaps_flower.evaluate_checkpoint import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND_ROOT = REPO_ROOT / "results" / "iotj_b2_b5_cross_direction_20260713_commands"
DEFAULT_RUN_ROOT = REPO_ROOT / "results" / "iotj_b2_b5_cross_direction_20260713"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "iotj_b2_b5_cross_direction_20260713_summary"
ROW_KEY = ("client", "split", "sample_index", "true_class")
NUM_CLASSES = 4


def _typed_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["client"]),
        str(row["split"]),
        int(row["sample_index"]),
        int(row["true_class"]),
    )


def _probabilities(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"prob_{class_id}"]) for class_id in range(NUM_CLASSES)]
            for row in rows
        ],
        dtype=np.float64,
    )


def _mcnemar_exact_p(b2_only_correct: int, b5_only_correct: int) -> float:
    discordant = b2_only_correct + b5_only_correct
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) * (0.5 ** discordant)
        for k in range(min(b2_only_correct, b5_only_correct) + 1)
    )
    return float(min(1.0, 2.0 * tail))


def _worst_recall(metrics: Mapping[str, Any]) -> float:
    values = [
        float(value)
        for value in metrics["per_class_recall"].values()
        if value is not None
    ]
    return min(values) if values else 0.0


def _stratified_bootstrap_indices(
    true_labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    sampled = [
        rng.choice(indices, size=len(indices), replace=True)
        for class_id in range(NUM_CLASSES)
        if len(indices := np.flatnonzero(true_labels == class_id))
    ]
    return np.concatenate(sampled) if sampled else np.empty(0, dtype=np.int64)


def compare_streams(
    b2_rows: Sequence[Mapping[str, Any]],
    b5_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260713,
    bootstrap_reps: int = 2000,
) -> dict[str, Any]:
    b2_keys = [_typed_key(row) for row in b2_rows]
    b5_keys = [_typed_key(row) for row in b5_rows]
    if b2_keys != b5_keys:
        raise ValueError("B2/B5 row keys are not identically aligned")
    if not b2_rows:
        raise ValueError("cannot compare empty B2/B5 streams")
    true = np.asarray([key[3] for key in b2_keys], dtype=np.int64)
    b2_probs = _probabilities(b2_rows)
    b5_probs = _probabilities(b5_rows)
    b2_metrics = classification_metrics(true, b2_probs)
    b5_metrics = classification_metrics(true, b5_probs)
    b2_pred = b2_probs.argmax(axis=1)
    b5_pred = b5_probs.argmax(axis=1)
    b2_correct = b2_pred == true
    b5_correct = b5_pred == true
    b2_only = int(np.sum(b2_correct & ~b5_correct))
    b5_only = int(np.sum(~b2_correct & b5_correct))

    accuracy_delta = 100.0 * (
        float(b2_metrics["accuracy"]) - float(b5_metrics["accuracy"])
    )
    macro_delta = 100.0 * (
        float(b2_metrics["macro_f1"]) - float(b5_metrics["macro_f1"])
    )
    worst_delta = 100.0 * (
        _worst_recall(b2_metrics) - _worst_recall(b5_metrics)
    )
    rng = np.random.default_rng(bootstrap_seed)
    accuracy_bootstrap = np.empty(bootstrap_reps, dtype=np.float64)
    macro_bootstrap = np.empty(bootstrap_reps, dtype=np.float64)
    for index in range(bootstrap_reps):
        sampled = _stratified_bootstrap_indices(true, rng)
        sampled_true = true[sampled]
        b2_sample = classification_metrics(sampled_true, b2_probs[sampled])
        b5_sample = classification_metrics(sampled_true, b5_probs[sampled])
        accuracy_bootstrap[index] = 100.0 * (
            float(b2_sample["accuracy"]) - float(b5_sample["accuracy"])
        )
        macro_bootstrap[index] = 100.0 * (
            float(b2_sample["macro_f1"]) - float(b5_sample["macro_f1"])
        )
    accuracy_ci = np.percentile(accuracy_bootstrap, [2.5, 97.5])
    macro_ci = np.percentile(macro_bootstrap, [2.5, 97.5])
    return {
        "N": len(true),
        "b2_accuracy": float(b2_metrics["accuracy"]),
        "b5_accuracy": float(b5_metrics["accuracy"]),
        "accuracy_delta_pp": accuracy_delta,
        "accuracy_delta_pp_ci_low": min(float(accuracy_ci[0]), accuracy_delta),
        "accuracy_delta_pp_ci_high": max(float(accuracy_ci[1]), accuracy_delta),
        "b2_macro_f1": float(b2_metrics["macro_f1"]),
        "b5_macro_f1": float(b5_metrics["macro_f1"]),
        "macro_f1_delta_pp": macro_delta,
        "macro_f1_delta_pp_ci_low": min(float(macro_ci[0]), macro_delta),
        "macro_f1_delta_pp_ci_high": max(float(macro_ci[1]), macro_delta),
        "b2_worst_recall": _worst_recall(b2_metrics),
        "b5_worst_recall": _worst_recall(b5_metrics),
        "worst_recall_delta_pp": worst_delta,
        "b2_nll": float(b2_metrics["nll"]),
        "b5_nll": float(b5_metrics["nll"]),
        "nll_delta": float(b2_metrics["nll"]) - float(b5_metrics["nll"]),
        "b2_ece": float(b2_metrics["ece"]),
        "b5_ece": float(b5_metrics["ece"]),
        "ece_delta": float(b2_metrics["ece"]) - float(b5_metrics["ece"]),
        "b2_only_correct": b2_only,
        "b5_only_correct": b5_only,
        "both_correct": int(np.sum(b2_correct & b5_correct)),
        "both_wrong": int(np.sum(~b2_correct & ~b5_correct)),
        "mcnemar_exact_p": _mcnemar_exact_p(b2_only, b5_only),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_reps": bootstrap_reps,
        "b2_confusion_matrix": b2_metrics["confusion_matrix"],
        "b5_confusion_matrix": b5_metrics["confusion_matrix"],
    }


def classify_direction(
    *,
    accuracy_delta_pp: float,
    macro_f1_delta_pp: float,
    worst_recall_delta_pp: float,
    margin_pp: float = 0.5,
) -> str:
    deltas = (accuracy_delta_pp, macro_f1_delta_pp, worst_recall_delta_pp)
    return "B2_noninferior" if all(value >= -margin_pp for value in deltas) else "B5_favored"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-root", type=Path, default=DEFAULT_COMMAND_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--margin-pp", type=float, default=0.5)
    args = parser.parse_args(argv)

    manifests = load_ordered_manifests(args.command_root, args.seed)
    device = resolve_device(args.device)
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    per_run_rows: list[dict[str, Any]] = []
    for _manifest_path, manifest in manifests:
        run_dir = args.run_root / manifest["run_name"]
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        target_client = int(manifest["protocol"]["target_clients"][0])
        data_root = REPO_ROOT / "dataset" / manifest["protocol"]["data_root"]
        payload = evaluate_run(
            run_dir,
            data_root=data_root,
            target_client=target_client,
            output_root=args.output_root,
            device=device,
            batch_size=args.batch_size,
        )
        identity = (str(manifest["direction_id"]), str(manifest["group_id"]))
        payloads[identity] = payload
        row = flatten_test_metrics(payload)
        row.update(
            {
                "direction_id": manifest["direction_id"],
                "target_client": target_client,
                "data_root": manifest["protocol"]["data_root"],
            }
        )
        per_run_rows.append(row)

    comparisons: list[dict[str, Any]] = []
    for direction_id in dict.fromkeys(
        manifest["direction_id"] for _path, manifest in manifests
    ):
        b2 = payloads[(str(direction_id), "B2")]
        b5 = payloads[(str(direction_id), "B5")]
        b2_rows = _read_csv(
            args.output_root / b2["run_name"] / "classification_test_stream.csv"
        )
        b5_rows = _read_csv(
            args.output_root / b5["run_name"] / "classification_test_stream.csv"
        )
        comparison = compare_streams(
            b2_rows,
            b5_rows,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_reps=args.bootstrap_reps,
        )
        comparison.update(
            {
                "direction_id": direction_id,
                "seed": args.seed,
                "target_client": b2["target_clients"][0],
                "decision_margin_pp": args.margin_pp,
                "decision": classify_direction(
                    accuracy_delta_pp=float(comparison["accuracy_delta_pp"]),
                    macro_f1_delta_pp=float(comparison["macro_f1_delta_pp"]),
                    worst_recall_delta_pp=float(
                        comparison["worst_recall_delta_pp"]
                    ),
                    margin_pp=args.margin_pp,
                ),
            }
        )
        comparisons.append(comparison)

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "classification_per_run.csv", per_run_rows)
    csv_comparisons = []
    for row in comparisons:
        flat = dict(row)
        flat["b2_confusion_matrix"] = json.dumps(flat["b2_confusion_matrix"])
        flat["b5_confusion_matrix"] = json.dumps(flat["b5_confusion_matrix"])
        csv_comparisons.append(flat)
    _write_csv(args.output_root / "paired_direction_comparison.csv", csv_comparisons)
    (args.output_root / "paired_direction_comparison.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "runs": len(per_run_rows),
                "comparisons": len(comparisons),
                "seed": args.seed,
                "device": str(device),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
