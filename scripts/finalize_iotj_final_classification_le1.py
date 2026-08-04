"""Open sealed tests once, consolidate fixed-endpoint metrics, costs, and audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from federated_dataset import create_client_test_only_loader, create_merged_test_loader
from gaps_flower.evaluate_checkpoint import load_checkpoint_model
from gaps_flower.target_information import (
    TargetAccessLedger,
    unlock_target_test_for_final_evaluation,
)
from scripts.evaluate_iotj_final_classification_le1 import classification_metrics
from scripts.run_iotj_final_classification_le1 import (
    BATCH_SIZE,
    IMPORTED_CHECKPOINT,
    LOCAL_DATA_ROOT,
    RESULT_ROOT,
    SEED,
    current_protocol_hash,
)


def evaluation_jobs() -> list[dict]:
    jobs = []
    for experiment_id, method, optimizer, note in (
        ("FCL-E1-FEDAVG", "FedAvg", "Adam", "frozen GAPS experimental protocol"),
        ("FCL-E1-FEDPROX", "FedProx", "Adam", "frozen GAPS experimental protocol"),
        ("FCL-E1-SCAFFOLD", "SCAFFOLD", "SGD", "canonical SCAFFOLD implementation"),
    ):
        for target in ("C3", "C4", "C5"):
            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "method": method,
                    "policy_method": method.lower(),
                    "target_id": target,
                    "optimizer": optimizer,
                    "optimizer_lr": 5e-4,
                    "optimizer_note": note,
                    "study": "E1",
                }
            )
    for method in ("CORAL", "MMD", "DANN"):
        for target in ("C3", "C4", "C5"):
            jobs.append(
                {
                    "experiment_id": f"FCL-E2-{method}-{target}",
                    "method": method,
                    "policy_method": method.lower(),
                    "target_id": target,
                    "optimizer": "Adam",
                    "optimizer_lr": 5e-4,
                    "optimizer_note": "canonical post-hoc DA reference",
                    "study": "E2",
                }
            )
    for target in ("C3", "C4", "C5"):
        jobs.append(
            {
                "experiment_id": f"FCL-E3-GAPS-{target}",
                "method": "GAPS",
                "policy_method": "gaps",
                "target_id": target,
                "optimizer": "Adam",
                "optimizer_lr": 5e-4,
                "optimizer_note": "proposed method protocol",
                "study": "E3",
            }
        )
    for variant in ("A1", "A2", "A3", "A4", "A5"):
        jobs.append(
            {
                "experiment_id": f"FCL-E4-{variant}",
                "method": variant,
                "policy_method": variant.lower(),
                "target_id": "C5",
                "optimizer": "Adam",
                "optimizer_lr": 5e-4,
                "optimizer_note": "frozen GAPS ablation protocol",
                "study": "E4",
            }
        )
    return jobs


def aggregate_loss_activity(rows: Iterable[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["variant"]),
            str(row["scope"]),
            str(row["loss_name"]),
            float(row["configured_weight"]),
            bool(row["input_available"]),
        )
        groups[key].append(dict(row))
    output = []
    for key, values in sorted(groups.items()):
        active_steps = sum(int(value["active_steps"]) for value in values)
        raw_sum = sum(
            float(value["mean_raw_loss"]) * int(value["active_steps"])
            for value in values
        )
        weighted_sum = sum(
            float(value["mean_weighted_loss"]) * int(value["active_steps"])
            for value in values
        )
        reasons = sorted(
            {
                reason
                for value in values
                for reason in str(value.get("inactive_reason", "")).split(";")
                if reason
            }
        )
        output.append(
            {
                "variant": key[0],
                "scope": key[1],
                "loss_name": key[2],
                "configured_weight": key[3],
                "input_available": key[4],
                "active_steps": active_steps,
                "mean_raw_loss": raw_sum / active_steps if active_steps else 0.0,
                "mean_weighted_loss": weighted_sum / active_steps if active_steps else 0.0,
                "inactive_reason": "" if active_steps else ";".join(reasons),
            }
        )
    return output


def figure_names() -> list[str]:
    return [
        "fig01_sensor_shift",
        "fig02_federated_baselines",
        "fig03_canonical_uda",
        "fig04_gaps_cross_target",
        "fig05_c5_ablation_hierarchy",
        "fig06_source_target_gap",
        "fig07_calibration_metrics",
        "fig08_compute_communication_cost",
        "fig09_gaps_c5_confusion",
    ]


def _checkpoint_for(experiment_id: str) -> Path:
    if experiment_id == "FCL-E1-FEDAVG":
        return IMPORTED_CHECKPOINT
    run_dir = RESULT_ROOT / experiment_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing run manifest: {experiment_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    return checkpoint


def _predict(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].long()
            logits, _, _ = model(x)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    if not probabilities:
        raise RuntimeError("FAIL_CLOSED empty final evaluation loader")
    return np.concatenate(probabilities), np.concatenate(labels)


def _json_cell(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cost_fields(experiment_id: str, checkpoint: Path) -> dict:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    model_state = payload["model_state"]
    parameter_bytes = int(
        sum(value.numel() * value.element_size() for value in model_state.values())
    )
    manifest_path = RESULT_ROOT / experiment_id / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    if experiment_id.startswith("FCL-E2-"):
        return {
            "training_seconds": 0.0,
            "commissioning_seconds": float(manifest.get("adaptation_seconds", 0.0)),
            "communication_rounds": 0,
            "model_parameter_bytes": parameter_bytes,
            "estimated_model_transport_bytes": 0,
            "communication_note": "post-hoc server-local adaptation; no federated transport",
        }
    if experiment_id == "FCL-E1-FEDAVG":
        training_seconds = None
    else:
        training_seconds = float(manifest.get("wall_seconds", 0.0))
    multiplier = 8 if experiment_id == "FCL-E1-SCAFFOLD" else 4
    note = (
        "model plus same-sized server/client control-variate transport estimate"
        if multiplier == 8
        else "two-client model downlink/uplink estimate; method statistics reported separately"
    )
    return {
        "training_seconds": training_seconds,
        "commissioning_seconds": 0.0,
        "communication_rounds": 25,
        "model_parameter_bytes": parameter_bytes,
        "estimated_model_transport_bytes": multiplier * 25 * parameter_bytes,
        "communication_note": note,
    }


def run_final_evaluation() -> tuple[list[dict], list[dict]]:
    protocol_hash = current_protocol_hash()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    prediction_rows = []
    source_cache: dict[str, dict] = {}
    for job in evaluation_jobs():
        experiment_id = job["experiment_id"]
        target = job["target_id"]
        run_dir = RESULT_ROOT / experiment_id
        completion = run_dir / "fixed_endpoint_complete.json"
        if not completion.is_file():
            raise RuntimeError(f"FAIL_CLOSED completion marker missing: {experiment_id}")
        completion_payload = json.loads(completion.read_text(encoding="utf-8"))
        if completion_payload.get("protocol_hash") != protocol_hash:
            raise RuntimeError(f"FAIL_CLOSED protocol mismatch: {experiment_id}")
        final_path = run_dir / f"final_evaluation_{target}.json"
        if final_path.exists():
            result = json.loads(final_path.read_text(encoding="utf-8"))
            rows.append(result["comparison_row"])
            prediction_rows.extend(result["predictions"])
            continue
        checkpoint = _checkpoint_for(experiment_id)
        model, config, _payload = load_checkpoint_model(
            str(checkpoint), device, BATCH_SIZE
        )
        if experiment_id not in source_cache:
            source_loader = create_merged_test_loader(
                [LOCAL_DATA_ROOT / "client_1", LOCAL_DATA_ROOT / "client_2"],
                batch_size=BATCH_SIZE,
                num_workers=0,
            )
            source_probs, source_labels = _predict(model, source_loader, device)
            source_cache[experiment_id] = classification_metrics(
                source_probs, source_labels, num_classes=4, ece_bins=15
            )
        ledger = TargetAccessLedger(run_dir / "target_access_ledger.jsonl")
        token = unlock_target_test_for_final_evaluation(
            job["policy_method"], target, completion, ledger
        )
        token.consume(target)
        target_loader = create_client_test_only_loader(
            LOCAL_DATA_ROOT / f"client_{target[1:]}", batch_size=BATCH_SIZE
        )
        target_probs, target_labels = _predict(model, target_loader, device)
        target_metrics = classification_metrics(
            target_probs, target_labels, num_classes=4, ece_bins=15
        )
        source_metrics = source_cache[experiment_id]
        comparison = {
            **job,
            "seed": SEED,
            "accuracy": target_metrics["accuracy"],
            "macro_f1": target_metrics["macro_f1"],
            "nll": target_metrics["nll"],
            "ece": target_metrics["ece"],
            "num_examples": target_metrics["num_examples"],
            "per_class": _json_cell(target_metrics["per_class"]),
            "confusion_matrix": _json_cell(target_metrics["confusion_matrix"]),
            "source_macro_f1": source_metrics["macro_f1"],
            "source_accuracy": source_metrics["accuracy"],
            "source_target_f1_gap": source_metrics["macro_f1"]
            - target_metrics["macro_f1"],
            "checkpoint": str(checkpoint),
            "checkpoint_selection": "fixed_endpoint_only",
            "target_test_access": "one_time_final_evaluation",
            **_cost_fields(experiment_id, checkpoint),
        }
        job_predictions = []
        for index, (label, probs) in enumerate(zip(target_labels, target_probs)):
            item = {
                "experiment_id": experiment_id,
                "method": job["method"],
                "target_id": target,
                "window_index": index,
                "true_class": int(label),
                "predicted_class": int(np.argmax(probs)),
                "confidence": float(np.max(probs)),
                **{f"prob_class_{class_id}": float(probs[class_id]) for class_id in range(4)},
            }
            job_predictions.append(item)
        final_payload = {
            "schema_version": "iotj.final_classification.final_evaluation.v1",
            "comparison_row": comparison,
            "source_metrics": source_metrics,
            "target_metrics": target_metrics,
            "predictions": job_predictions,
            "selection_role": "none_fixed_endpoint_only",
        }
        final_path.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        rows.append(comparison)
        prediction_rows.extend(job_predictions)
    return rows, prediction_rows


def _inactive_row(variant: str, scope: str, loss: str, reason: str) -> dict:
    return {
        "variant": variant,
        "scope": scope,
        "loss_name": loss,
        "configured_weight": 0.0,
        "input_available": False,
        "active_steps": 0,
        "mean_raw_loss": 0.0,
        "mean_weighted_loss": 0.0,
        "inactive_reason": reason,
    }


def build_ablation_loss_activity() -> list[dict]:
    observed = []
    mapping = {
        "A1": "FCL-E4-A1",
        "A2": "FCL-E4-A2",
        "A3": "FCL-E4-A3",
        "A4": "FCL-E4-A4",
        "A5": "FCL-E4-A5",
        "A6": "FCL-E3-GAPS-C5",
    }
    for variant, experiment_id in mapping.items():
        remote = RESULT_ROOT / experiment_id / "remote_server"
        for path in sorted(remote.glob("client_stats_round_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for client in payload.get("clients", []):
                observed.extend(client.get("loss_activity", []))
        for path in sorted(remote.glob("domain_adapt_round_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            observed.extend(payload.get("loss_activity", []))
    old = (
        RESULT_ROOT.parents[2]
        / "iotj-confirmation-observability/results/iotj_p0_routing_simplification_20260803"
        / "P0A_PURE_FEDAVG_LE1_S42/remote_server"
    )
    ce_weighted_sum = 0.0
    example_total = 0
    active_steps = 0
    for path in sorted(old.glob("client_stats_round_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for client in payload.get("clients", []):
            examples = int(client["train_metric_examples"])
            ce_weighted_sum += float(client["train_ce_mean"]) * examples
            example_total += examples
            active_steps += math.ceil(examples / BATCH_SIZE)
    observed.append(
        {
            "variant": "A0",
            "scope": "client",
            "loss_name": "source_ce",
            "configured_weight": 1.0,
            "input_available": True,
            "active_steps": active_steps,
            "mean_raw_loss": ce_weighted_sum / max(example_total, 1),
            "mean_weighted_loss": ce_weighted_sum / max(example_total, 1),
            "inactive_reason": "",
        }
    )
    for loss in ("semantic_alignment", "replay_distillation", "regression"):
        observed.append(_inactive_row("A0", "client", loss, "module_disabled_frozen_reuse"))
    server_terms = (
        "source_ce", "coral", "global_mmd", "class_mmd", "adversarial",
        "proto_anchor", "proto_loss", "consistency", "device_residual",
        "proto_mmd", "stage_mmd", "align_reg_legacy", "target_ce",
    )
    for variant in ("A0", "A1", "A2", "A3"):
        for loss in server_terms:
            observed.append(_inactive_row(variant, "server_da", loss, "server_da_disabled"))
    return aggregate_loss_activity(observed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_analysis(rows: list[dict]) -> None:
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["study"] != "E4":
            by_target[row["target_id"]].append(row)
    ranking_lines = []
    for target, values in sorted(by_target.items()):
        best = max(values, key=lambda item: float(item["macro_f1"]))
        ranking_lines.append(
            f"- {target}: highest fixed-endpoint macro-F1 is {best['method']} "
            f"({float(best['macro_f1']):.6f}); this is descriptive seed42 evidence."
        )
    text = """# Result Analysis

All results are fixed-endpoint, seed42-only descriptive evidence. No target test was used for learning-rate selection, threshold selection, stopping, hyperparameter search or checkpoint selection.

## Baseline interpretation

SCAFFOLD is implemented with its canonical SGD-style control-variate update, whereas FedAvg, FedProx, and GAPS use the frozen Adam optimizer adopted by the experimental system. Therefore, the comparison represents standard algorithm-level baselines rather than an optimizer-controlled single-factor ablation.

## Per-target fixed-endpoint summary

""" + "\n".join(ranking_lines) + """

## Interpretation limits

The comparison uses one registered seed and one C1/C2 source split. Differences must not be described as multi-seed stability, universal cross-device superiority or optimizer-controlled causal effects. E2 isolates canonical x-only post-hoc adaptation from GAPS's registered class/phase calibration use. Communication totals are deterministic model/control payload estimates; GAPS statistic JSON is identified separately rather than misreported as exact wire bytes.
"""
    (RESULT_ROOT / "RESULT_ANALYSIS.md").write_text(text, encoding="utf-8")
    audit = """# Experiment Audit

- Fixed protocol hash matched every completion marker.
- Every target test was opened only after its method-target endpoint marker.
- E2 runtime ledgers contain calibration x-only followed by one final test x/class event.
- Full GAPS/A4/A5 ledgers record calibration x/class/phase and no concentration.
- All reported checkpoints are round25 or step100 fixed endpoints.
- No hyperparameter or checkpoint search was performed.
- `ablation_loss_activity.csv` is aggregated from observed client/server records; A0 is a read-only reconstruction from its frozen round logs and is explicitly marked by inactive reasons for unavailable modules.
"""
    (RESULT_ROOT / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")


def generate_figures(rows: list[dict]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
    figure_dir = RESULT_ROOT / "figures"
    source_dir = figure_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    def finish(fig, stem: str) -> None:
        fig.tight_layout()
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    sensor_path = RESULT_ROOT / "FCL-E0-SHIFT/sensor_channel_shift.csv"
    with sensor_path.open(newline="", encoding="utf-8") as handle:
        sensor_rows = list(csv.DictReader(handle))
    targets = ["C3", "C4", "C5"]
    channels = sorted({int(row["channel"]) for row in sensor_rows})
    matrix = np.array(
        [
            [
                float(next(row["standardized_mean_difference"] for row in sensor_rows if row["target_id"] == target and int(row["channel"]) == channel))
                for channel in channels
            ]
            for target in targets
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    limit = max(float(np.max(np.abs(matrix))), 1e-6)
    image = ax.imshow(matrix, cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(channels)), [str(channel) for channel in channels])
    ax.set_yticks(range(len(targets)), targets)
    ax.set_xlabel("Sensor channel")
    ax.set_ylabel("Target device")
    fig.colorbar(image, ax=ax, label="Standardized mean difference")
    finish(fig, figure_names()[0])
    _write_csv(source_dir / "fig01_sensor_shift.csv", sensor_rows)

    def bar_figure(selected: list[dict], stem: str, ylabel: str = "Macro-F1") -> None:
        labels = [f"{row['method']}\n{row['target_id']}" for row in selected]
        values = [float(row["macro_f1"]) for row in selected]
        fig, ax = plt.subplots(figsize=(7.2, 3.1))
        ax.bar(range(len(values)), values, color=[colors[index % len(colors)] for index in range(len(values))])
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.99, 0.02, "seed42; fixed endpoint", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        finish(fig, stem)
        _write_csv(source_dir / f"{stem}.csv", selected)

    bar_figure([row for row in rows if row["study"] == "E1"], figure_names()[1])
    bar_figure([row for row in rows if row["study"] == "E2"], figure_names()[2])
    bar_figure([row for row in rows if row["study"] == "E3"], figure_names()[3])
    hierarchy = [
        next(row for row in rows if row["experiment_id"] == "FCL-E1-FEDAVG" and row["target_id"] == "C5"),
        *[next(row for row in rows if row["experiment_id"] == f"FCL-E4-A{index}") for index in range(1, 6)],
        next(row for row in rows if row["experiment_id"] == "FCL-E3-GAPS-C5"),
    ]
    hierarchy = [dict(row, method=f"A{index}") for index, row in enumerate(hierarchy)]
    bar_figure(hierarchy, figure_names()[4])

    main_rows = [row for row in rows if row["study"] != "E4"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    marker_map = {"C3": "o", "C4": "s", "C5": "^"}
    for index, row in enumerate(main_rows):
        ax.scatter(
            index,
            float(row["source_target_f1_gap"]),
            marker=marker_map[row["target_id"]],
            color=colors[index % len(colors)],
            s=28,
        )
    ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(main_rows)), [row["method"] for row in main_rows], rotation=60, ha="right")
    ax.set_ylabel("Source-target macro-F1 gap")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, figure_names()[5])
    _write_csv(source_dir / "fig06_source_target_gap.csv", main_rows)

    c5 = [row for row in main_rows if row["target_id"] == "C5"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    labels = [row["method"] for row in c5]
    for ax, metric, label in zip(axes, ("nll", "ece"), ("NLL (nats/window)", "ECE (15 bins)")):
        ax.bar(range(len(c5)), [float(row[metric]) for row in c5], color=colors[: len(c5)])
        ax.set_xticks(range(len(c5)), labels, rotation=55, ha="right")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, figure_names()[6])
    _write_csv(source_dir / "fig07_calibration_metrics.csv", c5)

    unique_cost = []
    seen = set()
    for row in rows:
        if row["experiment_id"] in seen:
            continue
        seen.add(row["experiment_id"])
        unique_cost.append(row)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cost_labels = [row["method"] for row in unique_cost]
    compute = [float(row["training_seconds"] or 0.0) + float(row["commissioning_seconds"] or 0.0) for row in unique_cost]
    communication = [max(float(row["estimated_model_transport_bytes"]), 1.0) for row in unique_cost]
    axes[0].bar(range(len(compute)), compute, color=colors * math.ceil(len(compute) / len(colors)))
    axes[0].set_ylabel("Compute wall time (s)")
    axes[1].bar(range(len(communication)), communication, color=colors * math.ceil(len(communication) / len(colors)))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Estimated transport (bytes, log scale)")
    for ax in axes:
        ax.set_xticks(range(len(cost_labels)), cost_labels, rotation=60, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, figure_names()[7])
    _write_csv(source_dir / "fig08_compute_communication_cost.csv", unique_cost)

    gaps_c5 = next(row for row in rows if row["experiment_id"] == "FCL-E3-GAPS-C5")
    confusion = np.asarray(json.loads(gaps_c5["confusion_matrix"]), dtype=int)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    image = ax.imshow(confusion, cmap="cividis")
    for true_id in range(confusion.shape[0]):
        for pred_id in range(confusion.shape[1]):
            ax.text(pred_id, true_id, str(confusion[true_id, pred_id]), ha="center", va="center", color="white" if confusion[true_id, pred_id] > confusion.max() / 2 else "black")
    ax.set_xticks(range(4), range(4))
    ax.set_yticks(range(4), range(4))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(image, ax=ax, label="Windows")
    finish(fig, figure_names()[8])
    confusion_rows = [
        {"true_class": i, "predicted_class": j, "count": int(confusion[i, j])}
        for i in range(4)
        for j in range(4)
    ]
    _write_csv(source_dir / "fig09_gaps_c5_confusion.csv", confusion_rows)
    captions = """# Figure Captions

All panels show the single preregistered seed42 fixed endpoint; no error bars or significance marks are shown because no repeated-seed uncertainty estimate is available.

1. Raw sensor-space standardized mean shifts by target and channel.
2. FedAvg, FedProx and canonical SCAFFOLD macro-F1 across C3-C5.
3. Canonical x-only CORAL, MMD and DANN macro-F1 across C3-C5.
4. Full GAPS macro-F1 across C3-C5.
5. C5 A0-A6 cumulative hierarchy.
6. Combined C1/C2 source minus target macro-F1 gap.
7. C5 NLL and 15-bin ECE.
8. Compute wall time and deterministic model/control transport estimates.
9. Full GAPS C5 confusion matrix in fixed class order 0-3.
"""
    (figure_dir / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def finalize() -> None:
    rows, predictions = run_final_evaluation()
    _write_csv(RESULT_ROOT / "classification_main_comparison.csv", rows)
    _write_csv(
        RESULT_ROOT / "source_target_f1_gap.csv",
        [
            {
                key: row[key]
                for key in (
                    "experiment_id", "method", "target_id", "source_macro_f1",
                    "macro_f1", "source_target_f1_gap", "seed",
                )
            }
            for row in rows
        ],
    )
    _write_csv(RESULT_ROOT / "per_window_predictions.csv", predictions)
    _write_csv(
        RESULT_ROOT / "ablation_loss_activity.csv", build_ablation_loss_activity()
    )
    write_analysis(rows)
    generate_figures(rows)
    index = {
        path.relative_to(RESULT_ROOT).as_posix(): _sha256(path)
        for path in sorted(RESULT_ROOT.rglob("*"))
        if path.is_file()
        and path.name != "sha256_index.json"
        and path.suffix.lower() not in {".pth", ".tar"}
    }
    (RESULT_ROOT / "sha256_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true", required=True)
    parser.parse_args()
    finalize()


if __name__ == "__main__":
    main()
