"""Render the final IEEE IoT-J end-to-end A4 evidence figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VARIANT_ORDER = ["R83_TARGET_ONLY", "R84_FED_H1", "R86_ALL_PRIORS"]
VARIANT_LABELS = {
    "R83_TARGET_ONLY": "83D sensor",
    "R84_FED_H1": "84D + H1",
    "R86_ALL_PRIORS": "86D + H1/H2/H3",
}
COLORS = ["#0072B2", "#E69F00", "#009E73"]
HATCHES = ["///", "", "xx"]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_fig5(main_csv: Path, per_gas_csv: Path, output: Path) -> tuple[Path, Path]:
    """Plot overall routed concentration estimation and per-gas RMSE."""
    main = _read_csv(main_csv)
    per_gas = _read_csv(per_gas_csv)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))

    scopes = ["S_ALL", "S_CC"]
    x = np.arange(len(scopes), dtype=float)
    width = 0.24
    for index, variant in enumerate(VARIANT_ORDER):
        values = [
            float(
                next(
                    row["RMSE"]
                    for row in main
                    if row["variant"] == variant and row["evaluation_scope"] == scope
                )
            )
            for scope in scopes
        ]
        axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            label=VARIANT_LABELS[variant],
            color=COLORS[index],
            hatch=HATCHES[index],
            edgecolor="black",
            linewidth=0.45,
        )
    axes[0].set_xticks(x, ["End-to-end\n($S_{ALL}$)", "Correct route\n($S_{CC}$)"])
    axes[0].set_ylabel("RMSE (ppm)")
    axes[0].set_title("(a) Overall performance", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[0].set_axisbelow(True)

    gas_order = ["Ethanol", "CO", "Ethylene", "Methane"]
    gx = np.arange(len(gas_order), dtype=float)
    for index, variant in enumerate(VARIANT_ORDER):
        values = [
            float(
                next(
                    row["RMSE"]
                    for row in per_gas
                    if row["variant"] == variant
                    and row["evaluation_scope"] == "S_ALL"
                    and row["gas"] == gas
                )
            )
            for gas in gas_order
        ]
        axes[1].bar(
            gx + (index - 1) * width,
            values,
            width,
            color=COLORS[index],
            hatch=HATCHES[index],
            edgecolor="black",
            linewidth=0.45,
        )
    axes[1].set_xticks(gx, gas_order, rotation=18, ha="right")
    axes[1].set_ylabel("End-to-end RMSE (ppm)")
    axes[1].set_title("(b) Per-gas performance", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[1].set_axisbelow(True)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

    png = output / "Fig5_concentration_estimation_per_gas.png"
    pdf = output / "Fig5_concentration_estimation_per_gas.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_fig6(main_csv: Path, budget_csv: Path, output: Path) -> tuple[Path, Path]:
    """Plot source-prior ablation and the separately sourced budget study."""
    main = _read_csv(main_csv)
    budget = sorted(_read_csv(budget_csv), key=lambda row: float(row["nominal_budget"]))
    output.mkdir(parents=True, exist_ok=True)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))

    x = np.arange(len(VARIANT_ORDER), dtype=float)
    width = 0.32
    for scope_index, (scope, label, color) in enumerate(
        [("S_ALL", "$S_{ALL}$", "#0072B2"), ("S_CC", "$S_{CC}$", "#D55E00")]
    ):
        values = [
            float(
                next(
                    row["NRMSE"]
                    for row in main
                    if row["variant"] == variant and row["evaluation_scope"] == scope
                )
            )
            for variant in VARIANT_ORDER
        ]
        axes[0].bar(
            x + (scope_index - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
            hatch="///" if scope_index == 0 else "",
            edgecolor="black",
            linewidth=0.45,
        )
    axes[0].set_xticks(x, ["83D\nsensor", "84D\n+ H1", "86D\n+ all priors"])
    axes[0].set_ylabel("NRMSE")
    axes[0].set_title("(a) Source-prior ablation", loc="left", fontweight="bold")
    axes[0].text(
        0.0,
        1.01,
        "Fixed A4 router, seed 42",
        transform=axes[0].transAxes,
        fontsize=6.5,
        color="#555555",
    )
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[0].set_axisbelow(True)

    budgets = np.asarray([float(row["nominal_budget"]) for row in budget])
    means = np.asarray([float(row["S_ALL_RMSE_mean"]) for row in budget])
    stds = np.asarray([float(row["S_ALL_RMSE_sample_std"]) for row in budget])
    axes[1].errorbar(
        budgets,
        means,
        yerr=stds,
        color="#009E73",
        marker="o",
        markersize=4,
        capsize=2.5,
        linewidth=1.4,
        label="Mean $\\pm$ sample SD",
    )
    axes[1].set_xticks(budgets.astype(int))
    axes[1].set_xlabel("Calibration windows")
    axes[1].set_ylabel("End-to-end RMSE (ppm)")
    axes[1].set_title("(b) Calibration budget", loc="left", fontweight="bold")
    axes[1].text(
        0.0,
        1.01,
        "Group-aware track G, 5 replicates",
        transform=axes[1].transAxes,
        fontsize=6.5,
        color="#555555",
    )
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[1].set_axisbelow(True)
    fig.tight_layout()

    png = output / "Fig6_source_prior_ablation_calibration_budget.png"
    pdf = output / "Fig6_source_prior_ablation_calibration_budget.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_fig7(curve_csv: Path, random_csv: Path, output: Path) -> tuple[Path, Path]:
    """Plot QC coverage/error trade-off against a matched random reference."""
    curve = sorted(_read_csv(curve_csv), key=lambda row: float(row["test_coverage"]))
    random_by_target = {
        float(row["target_coverage"]): row for row in _read_csv(random_csv)
    }
    output.mkdir(parents=True, exist_ok=True)
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))

    coverage = np.asarray([float(row["test_coverage"]) for row in curve])
    proposed = np.asarray([float(row["NRMSE"]) for row in curve])
    random_mean = np.asarray(
        [float(random_by_target[float(row["target_coverage"])]["random_NRMSE_mean"]) for row in curve]
    )
    random_std = np.asarray(
        [float(random_by_target[float(row["target_coverage"])]["random_NRMSE_sample_std"]) for row in curve]
    )
    axes[0].fill_between(
        coverage,
        random_mean - random_std,
        random_mean + random_std,
        color="#999999",
        alpha=0.25,
        label="Random mean $\\pm$ SD",
    )
    axes[0].plot(coverage, random_mean, "--", color="#777777", linewidth=1.1)
    axes[0].plot(
        coverage,
        proposed,
        color="#0072B2",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="Label-free QC",
    )
    for target, label, offset in [(0.90, "HC90", (4, -13)), (0.95, "HC95", (4, 6))]:
        row = next(
            item
            for item in curve
            if np.isclose(float(item["target_coverage"]), target)
        )
        axes[0].annotate(
            f"{label}\n({100*float(row['test_coverage']):.1f}% actual)",
            (float(row["test_coverage"]), float(row["NRMSE"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.5},
        )
    axes[0].set_xlabel("Retained test coverage")
    axes[0].set_ylabel("NRMSE")
    axes[0].set_title("(a) Coverage–error trade-off", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[0].set_axisbelow(True)

    capture_specs = [
        ("misroute_capture_rate", "Misroutes", "#D55E00", "o"),
        ("error_ge_40ppm_capture_rate", "Error $\\geq$40 ppm", "#009E73", "s"),
        ("top10pct_error_capture_rate", "Top 10% error", "#CC79A7", "^"),
    ]
    for key, label, color, marker in capture_specs:
        axes[1].plot(
            coverage,
            [float(row[key]) for row in curve],
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.3,
            label=label,
        )
    axes[1].set_xlabel("Retained test coverage")
    axes[1].set_ylabel("Rejected-event capture rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("(b) Risk-event capture", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[1].set_axisbelow(True)
    fig.tight_layout()

    png = output / "Fig7_qc_coverage_nrmse_random_hc.png"
    pdf = output / "Fig7_qc_coverage_nrmse_random_hc.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_fig8(system_csv: Path, physical_csv: Path, output: Path) -> tuple[Path, Path]:
    """Plot communication, Pi 5 deployment, and physical run validation."""
    system = _read_csv(system_csv)
    physical = _read_csv(physical_csv)[0]
    communication = [row for row in system if row["record_type"] == "communication"]
    runtimes = [row for row in system if row["record_type"] == "pi5_runtime"]
    output.mkdir(parents=True, exist_ok=True)
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.15))

    comm_values = [float(row["bytes"]) / 1e6 for row in communication]
    comm_labels = ["25-round\nFlower", "One-shot\nH1"]
    axes[0, 0].bar(
        np.arange(len(communication)),
        comm_values,
        color=["#0072B2", "#E69F00"],
        hatch=["///", ""],
        edgecolor="black",
        linewidth=0.5,
        width=0.58,
    )
    axes[0, 0].set_xticks(np.arange(len(communication)), comm_labels)
    axes[0, 0].set_ylabel("Serialized exchange (MB)")
    axes[0, 0].set_title("(a) Communication", loc="left", fontweight="bold")
    axes[0, 0].text(
        0.02,
        0.96,
        "Measured application payload\nvs. theoretical H1 payload",
        transform=axes[0, 0].transAxes,
        va="top",
        fontsize=6.3,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    axes[0, 0].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[0, 0].set_axisbelow(True)

    runtime_labels = ["V4 full", "V5 core", "V5 QC2"][: len(runtimes)]
    rx = np.arange(len(runtimes), dtype=float)
    width = 0.34
    axes[0, 1].bar(
        rx - width / 2,
        [float(row["pi_p50_ms"]) for row in runtimes],
        width,
        label="p50",
        color="#009E73",
        edgecolor="black",
        linewidth=0.45,
    )
    axes[0, 1].bar(
        rx + width / 2,
        [float(row["pi_p95_ms"]) for row in runtimes],
        width,
        label="p95",
        color="#CC79A7",
        hatch="xx",
        edgecolor="black",
        linewidth=0.45,
    )
    axes[0, 1].set_xticks(rx, runtime_labels)
    axes[0, 1].set_ylabel("Pi 5 latency (ms/window)")
    axes[0, 1].set_title("(b) Raspberry Pi 5 latency", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, ncol=2, loc="upper center")
    axes[0, 1].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[0, 1].set_axisbelow(True)

    bars = axes[1, 0].bar(
        rx,
        [float(row["pi_throughput_windows_per_s"]) for row in runtimes],
        color="#56B4E9",
        edgecolor="black",
        linewidth=0.45,
        width=0.55,
        label="Throughput",
    )
    axes[1, 0].set_xticks(rx, runtime_labels)
    axes[1, 0].set_ylabel("Throughput (windows/s)", color="#0072B2")
    axes[1, 0].tick_params(axis="y", labelcolor="#0072B2")
    rss_axis = axes[1, 0].twinx()
    rss_axis.plot(
        rx,
        [float(row["pi_peak_rss_mib"]) for row in runtimes],
        color="#D55E00",
        marker="o",
        linewidth=1.3,
        label="Peak RSS",
    )
    rss_axis.set_ylabel("Peak RSS (MiB)", color="#D55E00")
    rss_axis.tick_params(axis="y", labelcolor="#D55E00")
    axes[1, 0].set_title("(c) Pi 5 throughput and memory", loc="left", fontweight="bold")
    axes[1, 0].legend([bars, rss_axis.lines[0]], ["Throughput", "Peak RSS"], frameon=False, loc="upper left")
    axes[1, 0].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axes[1, 0].set_axisbelow(True)

    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("(d) Three-machine Flower validation", loc="left", fontweight="bold")
    box = {"boxstyle": "round,pad=0.35", "edgecolor": "#333333", "linewidth": 0.8}
    ax.text(0.12, 0.62, "Client C1\nCPU", ha="center", va="center", bbox={**box, "facecolor": "#D9EAF7"})
    ax.text(0.50, 0.62, "Cloud server\nA4 aggregation", ha="center", va="center", bbox={**box, "facecolor": "#FCE5CD"})
    ax.text(0.88, 0.62, "Client C2\nCPU", ha="center", va="center", bbox={**box, "facecolor": "#D9EAF7"})
    ax.annotate("", xy=(0.39, 0.62), xytext=(0.23, 0.62), arrowprops={"arrowstyle": "<->", "lw": 1.0})
    ax.annotate("", xy=(0.77, 0.62), xytext=(0.61, 0.62), arrowprops={"arrowstyle": "<->", "lw": 1.0})
    status_color = "#009E73" if physical["status"] == "PASS" else "#D55E00"
    ax.text(
        0.50,
        0.25,
        f"{physical['status']}  |  {physical['completed_rounds']}/{physical['expected_rounds']} rounds\n"
        f"seed {physical['seed']}  |  target {physical['target']}  |  {float(physical['wall_seconds'])/60:.1f} min",
        ha="center",
        va="center",
        color=status_color,
        fontweight="bold",
        fontsize=7.2,
    )
    ax.text(
        0.50,
        0.08,
        "Fixed endpoint; target test closed during training",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#555555",
    )
    fig.tight_layout()

    png = output / "Fig8_communication_pi5_physical_validation.png"
    pdf = output / "Fig8_communication_pi5_physical_validation.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regression-root",
        default="results/iotj_final_end_to_end_a4_20260804/regression",
    )
    parser.add_argument(
        "--output",
        default="results/iotj_final_end_to_end_a4_20260804/figures",
    )
    parser.add_argument(
        "--budget-csv",
        default=(
            "results/iotj_calibration_protocol_harmonization_20260726/"
            "track_groupaware/groupaware_budget_summary.csv"
        ),
    )
    parser.add_argument(
        "--qc-root", default="results/iotj_final_end_to_end_a4_20260804/qc"
    )
    parser.add_argument(
        "--system-root", default="results/iotj_final_end_to_end_a4_20260804/system"
    )
    parser.add_argument("--figure", choices=["5", "6", "7", "8", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.regression_root)
    if args.figure in {"5", "all"}:
        plot_fig5(
            root / "regression_main_summary.csv",
            root / "regression_per_gas.csv",
            Path(args.output),
        )
    if args.figure in {"6", "all"}:
        plot_fig6(
            root / "regression_main_summary.csv",
            Path(args.budget_csv),
            Path(args.output),
        )
    if args.figure in {"7", "all"}:
        qc = Path(args.qc_root)
        plot_fig7(
            qc / "qc_coverage_curve.csv",
            qc / "qc_random_reference.csv",
            Path(args.output),
        )
    if args.figure in {"8", "all"}:
        system = Path(args.system_root)
        plot_fig8(
            system / "system_deployment_summary.csv",
            system / "physical_validation_audit.csv",
            Path(args.output),
        )


if __name__ == "__main__":
    main()
