"""Evaluate frozen per-round B5 checkpoints on the historical C5 test split.

This is a retrospective diagnostic only.  The resulting curves must not be
used for early stopping, checkpoint selection, or DA-budget selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from gaps_flower.evaluate_checkpoint import load_checkpoint_model, make_loader
from scripts.summarize_iotj_classification_ablation import (
    NUM_CLASSES,
    classification_metrics,
)


EXPECTED_ROUNDS = 25
EXPECTED_ROWS = 1360
CHECKPOINT_PATTERN = re.compile(r"server_round_(\d{3})_adapted\.pth")
DATA_ROOT = Path(
    "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
)
VARIANTS = {
    "DA100": {
        "steps": 100,
        "root": Path(
            "results/iotj_b5_local_epoch_ablation_20260729/le1/raw/"
            "c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training"
        ),
        "evidence_status": "existing_canonical_reference",
    },
    "DA80": {
        "steps": 80,
        "root": Path(
            "results/iotj_b5_server_da_budget_ablation_20260731/da80/raw/"
            "c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training"
        ),
        "evidence_status": "canonical_postflight_pass",
    },
    "DA50": {
        "steps": 50,
        "root": Path(
            "results/iotj_b5_server_da_budget_ablation_20260731/da50/raw/"
            "c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training"
        ),
        "evidence_status": "canonical_postflight_pass",
    },
    "DA30": {
        "steps": 30,
        "root": Path(
            "results/iotj_b5_server_da_budget_ablation_20260731/da30/raw/"
            "c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training"
        ),
        "evidence_status": (
            "blocked_observability_contract_technical_result_only"
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _checkpoints(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in root.glob("server_round_*_adapted.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            result[int(match.group(1))] = path
    if set(result) != set(range(1, EXPECTED_ROUNDS + 1)):
        raise RuntimeError(
            f"FAIL_CLOSED checkpoint rounds mismatch under {root}: "
            f"{sorted(result)}"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _evaluate_model(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    all_true: list[int] = []
    all_probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_cpu, true_cpu in batches:
            logits, _, _ = model(x_cpu.to(device))
            probabilities = F.softmax(logits, dim=1)
            all_true.extend(true_cpu.numpy().tolist())
            all_probabilities.append(probabilities.cpu().numpy())
    probabilities = np.concatenate(all_probabilities, axis=0)
    metrics = classification_metrics(all_true, probabilities)
    if int(metrics["N"]) != EXPECTED_ROWS or not _finite(metrics):
        raise RuntimeError("FAIL_CLOSED invalid C5 per-round metrics")
    return metrics


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: int(row["round"]),
        )
        accuracy = [float(row["accuracy"]) for row in selected]
        final_accuracy = accuracy[-1]
        lower = final_accuracy - 0.005
        sustained_round = None
        for index in range(len(accuracy)):
            if all(value >= lower for value in accuracy[index:]):
                sustained_round = index + 1
                break
        consecutive_delta = [
            accuracy[index] - accuracy[index - 1]
            for index in range(1, len(accuracy))
        ]
        best_index = int(np.argmax(accuracy))
        result[variant] = {
            "server_da_steps_per_round": VARIANTS[variant]["steps"],
            "evidence_status": VARIANTS[variant]["evidence_status"],
            "round_1_accuracy": accuracy[0],
            "round_5_accuracy": accuracy[4],
            "round_10_accuracy": accuracy[9],
            "round_15_accuracy": accuracy[14],
            "round_20_accuracy": accuracy[19],
            "round_25_accuracy": final_accuracy,
            "best_accuracy": accuracy[best_index],
            "best_accuracy_round_descriptive_only": best_index + 1,
            "last_5_accuracy_mean": statistics.mean(accuracy[-5:]),
            "last_5_accuracy_sample_std": statistics.stdev(accuracy[-5:]),
            "accuracy_range": max(accuracy) - min(accuracy),
            "round_to_round_decrease_count": sum(
                delta < 0 for delta in consecutive_delta
            ),
            "largest_round_to_round_drop": min(consecutive_delta),
            "earliest_round_sustaining_final_minus_0_5pp": sustained_round,
        }
    return result


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    colors = {
        "DA100": "#0072B2",
        "DA80": "#009E73",
        "DA50": "#E69F00",
        "DA30": "#D55E00",
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    for variant in VARIANTS:
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: int(row["round"]),
        )
        rounds = [int(row["round"]) for row in selected]
        linestyle = "--" if variant == "DA30" else "-"
        markerfacecolor = "white" if variant == "DA30" else colors[variant]
        axes[0].plot(
            rounds,
            [100.0 * float(row["accuracy"]) for row in selected],
            label=variant,
            color=colors[variant],
            linestyle=linestyle,
            linewidth=1.7,
            marker="o",
            markersize=3.5,
            markerfacecolor=markerfacecolor,
            markevery=2,
        )
        axes[1].plot(
            rounds,
            [float(row["nll"]) for row in selected],
            label=variant,
            color=colors[variant],
            linestyle=linestyle,
            linewidth=1.7,
        )
    axes[0].set_ylabel("C5 test Accuracy (%)")
    axes[0].set_title(
        "Retrospective per-round target-domain diagnostic (seed 42)"
    )
    axes[0].legend(ncol=4, frameon=False, loc="lower right")
    axes[1].set_ylabel("C5 test NLL")
    axes[1].set_xlabel("Federated round")
    axes[1].set_xticks([1, 5, 10, 15, 20, 25])
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].text(
        0.01,
        0.03,
        "DA30 dashed/open: non-canonical observability status",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    fig.savefig(
        output_dir / "b5_server_da_round_curve.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_dir / "b5_server_da_round_curve.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {args.output_dir}")
    checkpoint_sets = {
        variant: _checkpoints(item["root"])
        for variant, item in VARIANTS.items()
    }
    args.output_dir.mkdir(parents=True)
    protocol = {
        "schema_version": "iotj.b5_server_da_round_curve_protocol.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "classifier": "B5",
            "seed": 42,
            "source_clients": ["C1", "C2"],
            "target_client": "C5",
            "rounds": EXPECTED_ROUNDS,
            "local_epochs_per_round": 1,
            "split": "historical_C5_test_1360",
        },
        "variants": {
            variant: {
                "server_da_steps_per_round": item["steps"],
                "checkpoint_root": str(item["root"]),
                "checkpoint_count": len(checkpoint_sets[variant]),
                "evidence_status": item["evidence_status"],
            }
            for variant, item in VARIANTS.items()
        },
        "test_access_boundary": (
            "Retrospective diagnostic after the historical test was opened. "
            "Metrics are prohibited from early stopping, checkpoint selection, "
            "DA-budget selection, or frozen-method reselection."
        ),
    }
    (args.output_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    device = torch.device(args.device)
    first_path = checkpoint_sets["DA100"][1]
    first_model, first_config, first_checkpoint = load_checkpoint_model(
        str(first_path), device, args.batch_size
    )
    if int(first_checkpoint.get("round", -1)) != 1:
        raise RuntimeError("FAIL_CLOSED first checkpoint round mismatch")
    loader = make_loader(DATA_ROOT, 5, "test", first_config.BATCH_SIZE)
    batches = [
        (batch[0].cpu(), batch[1].long().cpu())
        for batch in loader
    ]
    if sum(len(true) for _, true in batches) != EXPECTED_ROWS:
        raise RuntimeError("FAIL_CLOSED C5 test row count mismatch")
    del first_model

    rows: list[dict[str, Any]] = []
    for variant, item in VARIANTS.items():
        for round_id, checkpoint_path in checkpoint_sets[variant].items():
            started = time.perf_counter()
            model, config, checkpoint = load_checkpoint_model(
                str(checkpoint_path), device, args.batch_size
            )
            if (
                int(checkpoint.get("round", -1)) != round_id
                or int(config.NUM_CLASSES) != NUM_CLASSES
            ):
                raise RuntimeError(
                    f"FAIL_CLOSED checkpoint identity mismatch: {checkpoint_path}"
                )
            metrics = _evaluate_model(model, batches, device)
            confusion = metrics["confusion_matrix"]
            row = {
                "variant": variant,
                "seed": 42,
                "round": round_id,
                "server_da_steps_per_round": item["steps"],
                "server_da_total_steps_so_far": round_id * item["steps"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "ece": metrics["ece"],
                "mean_confidence": metrics["mean_confidence"],
                "error_count": EXPECTED_ROWS
                - sum(int(confusion[index][index]) for index in range(4)),
                "recall_class_0": metrics["per_class_recall"]["0"],
                "recall_class_1": metrics["per_class_recall"]["1"],
                "recall_class_2": metrics["per_class_recall"]["2"],
                "recall_class_3": metrics["per_class_recall"]["3"],
                "checkpoint_sha256": _sha256(checkpoint_path),
                "evaluation_seconds": time.perf_counter() - started,
                "evidence_status": item["evidence_status"],
                "test_used_for_selection": False,
            }
            rows.append(row)
            print(
                f"{variant} round={round_id:02d} "
                f"accuracy={row['accuracy']:.6f} "
                f"macro_f1={row['macro_f1']:.6f}",
                flush=True,
            )
            del model

    csv_path = args.output_dir / "per_round_c5_metrics.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "iotj.b5_server_da_round_curve_summary.v1",
        "rows": len(rows),
        "metrics_source": str(csv_path),
        "variants": _summary(rows),
        "evidence_boundary": protocol["test_access_boundary"],
        "formal_checkpoint_or_budget_selection": False,
    }
    (args.output_dir / "round_curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(rows, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
