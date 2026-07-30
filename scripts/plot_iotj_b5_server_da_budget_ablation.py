"""Plot the B5 server-DA compute-budget sensitivity summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    png_path = args.output_prefix.with_suffix(".png")
    svg_path = args.output_prefix.with_suffix(".svg")
    if png_path.exists() or svg_path.exists():
        raise FileExistsError("REFUSE_TO_OVERWRITE existing plot")

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = sorted(
        payload["rows"],
        key=lambda row: int(row["server_da_steps_per_round"]),
    )
    steps = [int(row["server_da_steps_per_round"]) for row in rows]
    canonical = [bool(row["canonical_validator_accepted"]) for row in rows]
    colors = ["#0072B2" if accepted else "#D55E00" for accepted in canonical]
    faces = [color if accepted else "white" for color, accepted in zip(colors, canonical)]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.6),
        constrained_layout=True,
    )

    accuracy = [100.0 * float(row["accuracy"]) for row in rows]
    macro_f1 = [100.0 * float(row["macro_f1"]) for row in rows]
    baseline_accuracy = next(
        value for value, step in zip(accuracy, steps) if step == 100
    )
    baseline_macro_f1 = next(
        value for value, step in zip(macro_f1, steps) if step == 100
    )
    ax = axes[0]
    ax.plot(steps, accuracy, color="#0072B2", linewidth=1.5, label="Accuracy")
    ax.plot(steps, macro_f1, color="#009E73", linewidth=1.5, label="Macro-F1")
    for index, step in enumerate(steps):
        ax.scatter(
            step,
            accuracy[index],
            s=48,
            marker="o",
            facecolor=faces[index],
            edgecolor=colors[index],
            linewidth=1.5,
            zorder=3,
        )
        ax.scatter(
            step,
            macro_f1[index],
            s=48,
            marker="s",
            facecolor=faces[index] if canonical[index] else "white",
            edgecolor="#009E73" if canonical[index] else "#D55E00",
            linewidth=1.5,
            zorder=3,
        )
    ax.axhline(
        baseline_accuracy - 0.5,
        color="#0072B2",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )
    ax.axhline(
        baseline_macro_f1 - 0.5,
        color="#009E73",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_ylabel("C5 test score (%)")
    ax.set_title("(a) Classification")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    nll = [float(row["nll"]) for row in rows]
    ece = [float(row["ece"]) for row in rows]
    ax.plot(steps, nll, color="#CC79A7", marker="o", label="NLL")
    ax.plot(steps, ece, color="#E69F00", marker="s", label="ECE")
    for index, accepted in enumerate(canonical):
        if not accepted:
            ax.scatter(
                steps[index],
                nll[index],
                s=60,
                facecolor="white",
                edgecolor="#D55E00",
                linewidth=1.5,
                zorder=4,
            )
            ax.scatter(
                steps[index],
                ece[index],
                s=60,
                marker="s",
                facecolor="white",
                edgecolor="#D55E00",
                linewidth=1.5,
                zorder=4,
            )
    ax.set_ylabel("Loss / calibration error")
    ax.set_title("(b) Calibration")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    hours = [float(row["training_wall_hours"]) for row in rows]
    ax.bar(
        steps,
        hours,
        width=10,
        color=faces,
        edgecolor=colors,
        linewidth=1.5,
    )
    for step, value in zip(steps, hours):
        ax.text(step, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Attempt wall time (h)")
    ax.set_title("(c) Training time")
    ax.set_ylim(0, max(hours) * 1.18)

    for ax in axes:
        ax.set_xlabel("Server DA steps per round")
        ax.set_xticks(steps)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        -0.04,
        "Open orange marker/bar: DA30 technical result; observability validator not accepted.",
        ha="center",
        fontsize=8,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(png_path), "svg": str(svg_path)}, indent=2))


if __name__ == "__main__":
    main()
