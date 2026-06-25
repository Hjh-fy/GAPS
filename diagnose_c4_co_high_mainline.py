"""Diagnose persistent C4 high-CO regression errors across mainline candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CO_CLASS = 1
PRED_KEYS = {
    "baseline": "baseline_final_ppm",
    "h2_3": "h2_3_ppm",
    "source_aug": "source_aug_target_ridge_ppm",
    "h8": "h8_pred_co_source_aug_else_h23_ppm",
}


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metrics(df: pd.DataFrame, pred_key: str) -> dict[str, Any]:
    if df.empty:
        return {"N": 0, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "P90AE": np.nan, "Under50Rate": np.nan}
    err = df[pred_key].astype(float) - df["true_ppm"].astype(float)
    ae = err.abs()
    return {
        "N": int(len(df)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(ae)),
        "Bias": float(np.mean(err)),
        "P90AE": float(np.percentile(ae, 90)),
        "Under50Rate": float(np.mean(df[pred_key].astype(float) < 50.0)),
    }


def summarize_by(df: pd.DataFrame, group_cols: list[str], pred_keys: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        prefix = {col: value for col, value in zip(group_cols, group)}
        for label, pred_key in pred_keys.items():
            item = dict(prefix)
            item["candidate"] = label
            item.update(metrics(sub, pred_key))
            rows.append(item)
    return pd.DataFrame(rows)


def add_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_class_name"] = out["pred_class"].map({0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}).fillna(out["pred_class"].astype(str))
    out["route_group"] = np.where(out["pred_class"].astype(int) == CO_CLASS, "pred_CO", "pred_nonCO")
    out["co_bin"] = pd.cut(
        out["true_ppm"].astype(float),
        bins=[-np.inf, 100.0, 175.0, np.inf],
        labels=["low_25_100", "mid_125_175", "high_200_250"],
    ).astype(str)
    for label, pred_key in PRED_KEYS.items():
        out[f"{label}_error"] = out[pred_key].astype(float) - out["true_ppm"].astype(float)
        out[f"{label}_abs_error"] = out[f"{label}_error"].abs()
    return out


def selected_table(df: pd.DataFrame, group_col: str, candidates: list[str]) -> str:
    if df.empty:
        return "(empty)"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row.get(group_col, "")),
                str(row.get("candidate", "")),
                f"{fnum(row.get('N')):.0f}",
                f"{fnum(row.get('RMSE')):.2f}",
                f"{fnum(row.get('Bias')):.2f}",
                f"{fnum(row.get('P90AE')):.2f}",
                f"{fnum(row.get('Under50Rate')):.2f}",
            ]
        )
    lines = [
        f"| {group_col} | candidate | N | RMSE | Bias | P90AE | Under50Rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(out: Path, overall: pd.DataFrame, by_phase: pd.DataFrame, by_route: pd.DataFrame, top_errors: pd.DataFrame) -> None:
    overall_lines = [
        "| candidate | N | RMSE | MAE | Bias | P90AE | Under50Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        overall_lines.append(
            "| {candidate} | {N:.0f} | {RMSE:.2f} | {MAE:.2f} | {Bias:.2f} | {P90AE:.2f} | {Under50Rate:.2f} |".format(
                candidate=row["candidate"],
                N=fnum(row["N"]),
                RMSE=fnum(row["RMSE"]),
                MAE=fnum(row["MAE"]),
                Bias=fnum(row["Bias"]),
                P90AE=fnum(row["P90AE"]),
                Under50Rate=fnum(row["Under50Rate"]),
            )
        )

    top_lines = [
        "| sample | phase | pred | true | h2.3 | h8 | h8 abs err | filename | repeat |",
        "|---:|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in top_errors.head(12).iterrows():
        top_lines.append(
            "| {sample} | {phase} | {pred} | {true:.1f} | {h23:.1f} | {h8:.1f} | {err:.1f} | {file} | {repeat} |".format(
                sample=int(row["sample_index"]),
                phase=str(row.get("response_phase", "")),
                pred=str(row.get("pred_class_name", "")),
                true=fnum(row.get("true_ppm")),
                h23=fnum(row.get("h2_3_ppm")),
                h8=fnum(row.get("h8_pred_co_source_aug_else_h23_ppm")),
                err=fnum(row.get("h8_abs_error")),
                file=str(row.get("filename", "")),
                repeat=fnum(row.get("repeat_id"), 0.0),
            )
        )

    text = f"""# C4 High-CO Mainline Diagnosis

Scope: C4, true CO, true ppm >= 200, target test, no QC filtering.

## Overall

{chr(10).join(overall_lines)}

## By Response Phase

{selected_table(by_phase, "response_phase", list(PRED_KEYS))}

## By Predicted Route

{selected_table(by_route, "route_group", list(PRED_KEYS))}

## Top H8 Absolute Errors

{chr(10).join(top_lines)}

## Reading

- If `route_group=pred_nonCO` has high Under50Rate, the failure is route-driven and residual correction cannot fully fix it.
- If `route_group=pred_CO` still has high RMSE/Bias, the concentration mapping itself remains weak for C4 high CO.
- Phase-specific concentration bias points to recovery/main-response mismatch and should guide the next C4-specific candidate.
"""
    (out / "c4_co_high_mainline_diagnosis_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose C4 high-CO mainline errors.")
    parser.add_argument("--predictions", default="results/co_only_source_aug_hybrid_stratcalval_20260625/co_only_source_aug_hybrid_predictions.csv")
    parser.add_argument("--output-dir", default="results/c4_co_high_mainline_diagnosis_20260625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions)
    df = add_common_fields(df)
    c4_high = df[
        (df["split"].astype(str) == "test")
        & (df["client"].astype(str) == "C4")
        & (df["true_class"].astype(int) == CO_CLASS)
        & (df["true_ppm"].astype(float) >= 200.0)
    ].copy()

    overall = pd.DataFrame([{"candidate": label, **metrics(c4_high, pred_key)} for label, pred_key in PRED_KEYS.items()])
    by_phase = summarize_by(c4_high, ["response_phase"], PRED_KEYS)
    by_route = summarize_by(c4_high, ["route_group"], PRED_KEYS)
    by_pred_class = summarize_by(c4_high, ["pred_class_name"], PRED_KEYS)
    by_file = summarize_by(c4_high, ["filename"], PRED_KEYS)
    top_errors = c4_high.sort_values("h8_abs_error", ascending=False)

    overall.to_csv(out / "c4_high_overall.csv", index=False)
    by_phase.to_csv(out / "c4_high_by_phase.csv", index=False)
    by_route.to_csv(out / "c4_high_by_route.csv", index=False)
    by_pred_class.to_csv(out / "c4_high_by_pred_class.csv", index=False)
    by_file.to_csv(out / "c4_high_by_file.csv", index=False)
    top_errors.drop(columns=[], errors="ignore").head(50).to_csv(out / "c4_high_top_h8_errors.csv", index=False)
    write_report(out, overall, by_phase, by_route, top_errors)
    print(f"Wrote C4 high-CO diagnosis to {out}")


if __name__ == "__main__":
    main()
