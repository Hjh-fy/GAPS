"""Summarize five completed centralized three-gas baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(results_root: Path, seed: int) -> dict:
    rows = []
    exposure_confusions = []
    for fold in range(1, 6):
        metrics_path = results_root / f"fold_{fold}_seed_{seed}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing fold result: {metrics_path}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        window = payload["test"]["window"]
        exposure = payload["test"]["exposure"]
        rows.append(
            {
                "fold": fold,
                "window_accuracy": float(window["accuracy"]),
                "window_macro_f1": float(window["macro_f1"]),
                "exposure_accuracy": float(exposure["accuracy"]),
                "exposure_macro_f1": float(exposure["macro_f1"]),
                "exposure_confusion_matrix": exposure["confusion_matrix"],
                "metrics_path": str(metrics_path),
            }
        )
        exposure_confusions.append(
            np.asarray(exposure["confusion_matrix"], dtype=np.int64)
        )

    metric_names = (
        "window_accuracy",
        "window_macro_f1",
        "exposure_accuracy",
        "exposure_macro_f1",
    )
    aggregate = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "status": "smoke_baseline_only",
        "seed": seed,
        "n_folds": 5,
        "folds": rows,
        "aggregate": aggregate,
        "summed_exposure_confusion_matrix": np.sum(
            exposure_confusions,
            axis=0,
        ).tolist(),
        "limitations": [
            "Nominal rather than manually verified gas boundaries were used.",
            "Only one seed and five training epochs were run.",
            "This is centralized CE training, not a federated comparison.",
            "Fold groups also correspond to concentration/order ranks.",
        ],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "results" / "lab_three_gas_centralized_smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize(args.results_root.resolve(), args.seed)
    output = args.output or args.results_root / f"fivefold_seed_{args.seed}_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    exposure = summary["aggregate"]["exposure_macro_f1"]
    print(
        f"Exposure Macro-F1: {exposure['mean']:.4f} "
        f"+/- {exposure['std_population']:.4f}; output={output}"
    )


if __name__ == "__main__":
    main()
