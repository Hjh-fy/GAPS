"""Diagnostic upper-bound sweep for C4 high-CO route-rescue gates.

This script intentionally evaluates gates on target test to estimate potential
headroom. It is not a formal calibration-selected rule.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_regression_head_ablation import summarize, write_csv


CO_CLASS = 1
BASE_KEY = "h8_pred_co_source_aug_else_h23_ppm"


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metrics(rows: pd.DataFrame, pred_key: str) -> dict[str, float]:
    if rows.empty:
        return {"N": 0, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "P90AE": np.nan}
    err = rows[pred_key].astype(float) - rows["true_ppm"].astype(float)
    return {
        "N": float(len(rows)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
        "P90AE": float(np.percentile(np.abs(err), 90)),
    }


def gate_mask(df: pd.DataFrame, pred_classes: set[int], phase: str, max_final: float, min_risk: float, max_conf_margin: float) -> pd.Series:
    mask = (
        (df["client"].astype(str) == "C4")
        & (df["pred_class"].astype(int).isin(pred_classes))
        & (df["final_ppm"].astype(float) <= max_final)
        & (df["risk_score"].astype(float) >= min_risk)
    )
    if phase != "any":
        mask &= df["response_phase"].astype(str) == phase
    if max_conf_margin < 1.0:
        mask &= df["confidence_margin"].astype(float) <= max_conf_margin
    return mask


def apply_gate(df: pd.DataFrame, mask: pd.Series, rescue_ppm: float) -> pd.DataFrame:
    out = df.copy()
    out["c4_route_rescue_upper_hit"] = mask.astype(int)
    out["candidate_ppm"] = out[BASE_KEY].astype(float)
    out.loc[mask, "candidate_ppm"] = float(rescue_ppm)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep C4 route-rescue upper-bound gates.")
    parser.add_argument("--predictions", default="results/co_only_source_aug_hybrid_stratcalval_20260625/co_only_source_aug_hybrid_predictions.csv")
    parser.add_argument("--output-dir", default="results/c4_route_rescue_upper_bound_20260625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    df = df[df["split"].astype(str) == "test"].copy()

    pred_class_sets = {
        "ethanol": {0},
        "ethylene": {2},
        "ethanol_ethylene": {0, 2},
        "all_nonco": {0, 2, 3},
    }
    phases = ["any", "main_response", "recovery"]
    max_finals = [20.0, 30.0, 50.0]
    min_risks = [2.0, 4.0, 6.0]
    max_conf_margins = [1.0, 0.9, 0.7, 0.2]
    rescue_ppms = [200.0, 225.0, 250.0]

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for class_label, phase, max_final, min_risk, max_margin, rescue_ppm in product(
        pred_class_sets, phases, max_finals, min_risks, max_conf_margins, rescue_ppms
    ):
        mask = gate_mask(df, pred_class_sets[class_label], phase, max_final, min_risk, max_margin)
        cand = apply_gate(df, mask, rescue_ppm)
        c4_high = cand[
            (cand["client"].astype(str) == "C4")
            & (cand["true_class"].astype(int) == CO_CLASS)
            & (cand["true_ppm"].astype(float) >= 200.0)
        ]
        c4_nonco = cand[
            (cand["client"].astype(str) == "C4")
            & (cand["true_class"].astype(int) != CO_CLASS)
        ]
        all_nonco = cand[cand["true_class"].astype(int) != CO_CLASS]
        true_high_hits = cand[mask & (cand["true_class"].astype(int) == CO_CLASS) & (cand["true_ppm"].astype(float) >= 200.0)]
        false_hits = cand[mask & ~((cand["true_class"].astype(int) == CO_CLASS) & (cand["true_ppm"].astype(float) >= 200.0))]
        item: dict[str, Any] = {
            "class_label": class_label,
            "pred_classes": ",".join(str(v) for v in sorted(pred_class_sets[class_label])),
            "phase": phase,
            "max_final": max_final,
            "min_risk": min_risk,
            "max_conf_margin": max_margin,
            "rescue_ppm": rescue_ppm,
            "hit_N": int(mask.sum()),
            "true_c4_high_hits": int(len(true_high_hits)),
            "false_hits": int(len(false_hits)),
        }
        for prefix, subset in [
            ("ALL", cand),
            ("C4_CO_high", c4_high),
            ("C4_nonCO", c4_nonco),
            ("nonCO_ALL", all_nonco),
        ]:
            m = metrics(subset, "candidate_ppm")
            for key, value in m.items():
                item[f"{prefix}_{key}"] = value
        rows.append(item)
        if len(true_high_hits) and len(false_hits) == 0:
            tmp = cand.loc[mask].copy()
            tmp["gate_id"] = len(rows) - 1
            prediction_rows.extend(tmp.to_dict("records"))

    result = pd.DataFrame(rows)
    result = result.sort_values(["C4_CO_high_RMSE", "ALL_RMSE", "false_hits"])
    result.to_csv(out / "c4_route_rescue_upper_bound_sweep.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "c4_route_rescue_zero_false_hit_examples.csv", index=False)

    best = result.head(20)
    lines = [
        "# C4 Route-Rescue Upper-Bound Sweep",
        "",
        "Diagnostic only: gates are evaluated on target test to estimate possible headroom. Do not treat this as a formal selected rule.",
        "",
        "| rank | classes | phase | max_final | min_risk | max_margin | rescue | hits | true high | false | ALL | C4 high | C4 nonCO | nonCO |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(best.iterrows(), start=1):
        lines.append(
            "| {rank} | {classes} | {phase} | {max_final:.0f} | {min_risk:.1f} | {max_margin:.1f} | {rescue:.0f} | {hits:.0f} | {true_hits:.0f} | {false_hits:.0f} | {all_rmse:.2f} | {c4_high:.2f} | {c4_nonco:.2f} | {nonco:.2f} |".format(
                rank=rank,
                classes=row["class_label"],
                phase=row["phase"],
                max_final=fnum(row["max_final"]),
                min_risk=fnum(row["min_risk"]),
                max_margin=fnum(row["max_conf_margin"]),
                rescue=fnum(row["rescue_ppm"]),
                hits=fnum(row["hit_N"]),
                true_hits=fnum(row["true_c4_high_hits"]),
                false_hits=fnum(row["false_hits"]),
                all_rmse=fnum(row["ALL_RMSE"]),
                c4_high=fnum(row["C4_CO_high_RMSE"]),
                c4_nonco=fnum(row["C4_nonCO_RMSE"]),
                nonco=fnum(row["nonCO_ALL_RMSE"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The best rows estimate how much C4 high-CO could improve if a route-rescue gate were available.",
            "- Rows with nonzero false hits are risky because they would overwrite non-CO or lower-concentration windows.",
            "- A formal rule must be selected on calibration-validation and then evaluated on test.",
        ]
    )
    (out / "c4_route_rescue_upper_bound_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote C4 route-rescue upper-bound sweep to {out}")


if __name__ == "__main__":
    main()
