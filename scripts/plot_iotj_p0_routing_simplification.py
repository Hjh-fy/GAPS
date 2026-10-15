"""Generate P0 figures only from audited machine-readable CSV/JSON inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def save(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args(); root = args.result_root.resolve(); source = root / "P0A_PURE_FEDAVG_LE1_S42"; p0b = root / "P0B_ROUNDWISE_COMMISSIONING_S42"
    metrics = read_csv(p0b / "roundwise_routing_metrics.csv"); colors = {"source_only": "#0072B2", "simple_target_ce": "#009E73", "full_target_adapter": "#D55E00"}
    for metric, ylabel, stem in (("macro_f1", "C5 Macro-F1", "target_macro_f1_vs_round"), ("accuracy", "C5 Accuracy", "target_accuracy_vs_round")):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method in colors:
            selected = sorted((row for row in metrics if row["method"] == method), key=lambda row: int(row["source_round"]))
            ax.plot([int(row["source_round"]) for row in selected], [float(row[metric]) for row in selected], label=method, color=colors[method], linewidth=1.7)
        ax.set(xlabel="Frozen source FedAvg round", ylabel=ylabel); ax.set_xticks([1, 5, 10, 15, 20, 25]); ax.grid(alpha=.25); ax.legend(frameon=False); save(fig, p0b, stem)
    curve = json.loads((source / "client_training_curve.json").read_text(encoding="utf-8")); fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for client, color in (("C1", "#0072B2"), ("C2", "#E69F00")):
        selected = sorted((row for row in curve if row["client_id"] == client), key=lambda row: row["round"]); ax.plot([row["round"] for row in selected], [row["train_ce_mean"] for row in selected], label=client, color=color)
    ax.set(xlabel="Federated round", ylabel="Sample-weighted local CE"); ax.grid(alpha=.25); ax.legend(frameon=False); save(fig, p0b, "client_ce_vs_round")
    simple = [row for row in read_csv(p0b / "simple_ce_commissioning_diagnostics.csv") if int(row["source_round"]) == 25]; fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.plot([int(row["step"]) for row in simple], [float(row["target_calibration_ce"]) for row in simple], color="#009E73"); ax.set(xlabel="Commissioning step", ylabel="C5 calibration CE"); ax.grid(alpha=.25); save(fig, p0b, "simple_ce_commissioning_curve_round25")
    full = [row for row in read_csv(p0b / "full_da_commissioning_diagnostics.csv") if int(row["source_round"]) == 25]; activity = read_csv(p0b / "server_loss_activity_summary.csv"); active = [row["loss_name"] for row in activity if row["activity_status"] == "ACTIVE"]
    weighted = {"coral_loss": "weighted_coral_loss", "mmd_global": "weighted_mmd_global", "mmd_class": "weighted_mmd_class", "stage_mmd_loss": "weighted_stage_mmd_loss", "adv_loss": "weighted_adv_loss", "proto_anchor": "weighted_proto_anchor", "proto_loss": "weighted_proto_loss", "consist_loss": "weighted_consist_loss", "residual_loss": "weighted_residual_loss"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name in active:
        if name in weighted: ax.plot([int(row["step"]) for row in full], [float(row[weighted[name]]) for row in full], label=name, linewidth=1.3)
    ax.set(xlabel="DA step", ylabel="Weighted active loss"); ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); save(fig, p0b, "full_da_loss_components_round25")


if __name__ == "__main__": main()
