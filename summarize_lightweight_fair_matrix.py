"""Summarize the lightweight fair auto_v2 matrix.

This report combines:

- original R3aK16 baseline;
- H2.3 / H8 current mainline references;
- L1 source-lightweight + full residual auto_v2 results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SCOPES = [
    "ALL",
    "C3-CO",
    "C4-CO",
    "C5-CO",
    "C3-CO_high_200_250",
    "C4-CO_high_200_250",
    "C5-CO_high_200_250",
    "nonCO_ALL",
]


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def from_l1_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    label_map = {
        "baseline_final_ppm": ("B0 R3aK16 + original auto_v2", "baseline"),
        "source_ridge_forced_identity": ("L0 source Ridge direct", "direct-transfer"),
        "source_ridge_forced_affine": ("L0 source Ridge + target affine", "affine"),
        "source_ridge_val_selected": ("L1 source Ridge + full residual auto_v2", "L1 fair"),
        "source_per_gas_mlp_forced_identity": ("L0 source per-gas MLP direct", "direct-transfer"),
        "source_per_gas_mlp_forced_affine": ("L0 source per-gas MLP + target affine", "affine"),
        "source_per_gas_mlp_val_selected": ("L1 source per-gas MLP + full residual auto_v2", "L1 fair"),
        "source_shared_mlp_forced_identity": ("L0 source shared MLP direct", "direct-transfer"),
        "source_shared_mlp_forced_affine": ("L0 source shared MLP + target affine", "affine"),
        "source_shared_mlp_val_selected": ("L1 source shared MLP + full residual auto_v2", "L1 fair"),
        "source_ridge_forced_ridge_phase": ("Diagnostic source Ridge + forced ridge_phase", "diagnostic"),
        "source_ridge_forced_piecewise_ridge": ("Diagnostic source Ridge + forced piecewise", "diagnostic"),
        "source_per_gas_mlp_forced_ridge_phase": ("Diagnostic source per-gas MLP + forced ridge_phase", "diagnostic"),
        "source_per_gas_mlp_forced_piecewise_ridge": ("Diagnostic source per-gas MLP + forced piecewise", "diagnostic"),
        "source_shared_mlp_forced_ridge_phase": ("Diagnostic source shared MLP + forced ridge_phase", "diagnostic"),
        "source_shared_mlp_forced_piecewise_ridge": ("Diagnostic source shared MLP + forced piecewise", "diagnostic"),
    }
    for mode, (label, status) in label_map.items():
        item: dict[str, Any] = {"label": label, "mode": mode, "status": status}
        sub = df[(df["split"] == "test") & (df["mode"] == mode)]
        for scope in SCOPES:
            row = sub[sub["scope"] == scope]
            item[scope] = fnum(row["RMSE"].iloc[0]) if not row.empty else float("nan")
        rows.append(item)
    return pd.DataFrame(rows)


def from_mainline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    wanted = {
        "H2_3_mlp_c3_ridge_c4_c5grid": "B1 H2.3 target direct-head mainline",
        "H8_pred_co_source_aug_else_h23": "B2 H8 CO-specialist candidate",
    }
    col_map = {
        "ALL": "all_rmse",
        "C3-CO": "c3_co_rmse",
        "C4-CO": "c4_co_rmse",
        "C5-CO": "c5_co_rmse",
        "C3-CO_high_200_250": "c3_co_high_rmse",
        "C4-CO_high_200_250": "c4_co_high_rmse",
        "C5-CO_high_200_250": "c5_co_high_rmse",
        "nonCO_ALL": "nonco_all_rmse",
    }
    for mode, label in wanted.items():
        src = df[df["mode"] == mode]
        if src.empty:
            continue
        src = src.iloc[0]
        item: dict[str, Any] = {"label": label, "mode": mode, "status": str(src.get("status", "reference"))}
        for scope, col in col_map.items():
            item[scope] = fnum(src.get(col))
        rows.append(item)
    return pd.DataFrame(rows)


def write_report(out: Path, summary: pd.DataFrame) -> None:
    columns = ["label", "status", *SCOPES]
    lines = ["| candidate | status | " + " | ".join(SCOPES) + " |", "|---|---|" + "|".join(["---:"] * len(SCOPES)) + "|"]
    for _, row in summary[columns].iterrows():
        values = [str(row["label"]), str(row["status"])]
        values.extend("" if pd.isna(row[scope]) else f"{float(row[scope]):.2f}" for scope in SCOPES)
        lines.append("| " + " | ".join(values) + " |")

    best_l1 = summary[summary["status"] == "L1 fair"].sort_values("ALL").head(1)
    best_diag = summary[summary["status"] == "diagnostic"].sort_values("ALL").head(1)
    best_l1_text = "n/a"
    if not best_l1.empty:
        row = best_l1.iloc[0]
        best_l1_text = f"{row['label']} with ALL RMSE {float(row['ALL']):.2f}"
    best_diag_text = "n/a"
    if not best_diag.empty:
        row = best_diag.iloc[0]
        best_diag_text = f"{row['label']} with ALL RMSE {float(row['ALL']):.2f}"

    text = f"""# Lightweight Fair Matrix Summary

Date: 2026-06-25

Scope: C12 -> C345 target test, no-QC full-set.

## Summary

{chr(10).join(lines)}

## Reading

- Direct source-lightweight transfer still collapses, confirming that source fit quality alone is not enough.
- Target affine calibration rescues some error, but remains worse than B0.
- Full residual auto_v2 changes the conclusion materially: L1 source-lightweight heads improve from about 36-71 ALL RMSE down to about 22-23 ALL RMSE.
- Best formal L1 result: {best_l1_text}.
- Best diagnostic forced result: {best_diag_text}.
- L1 now beats the original B0 baseline on ALL RMSE, C3 CO, C5 CO, and nonCO is roughly comparable.
- L1 still does not reach H2.3/H8 performance, especially on C4 CO and C4 high-CO.

## Decision

Lightweight source heads should not replace the H2.3/H8 performance mainline yet.

They are now credible deployment-lite candidates, because full target residual auto_v2 makes them much stronger than the earlier affine-only result. The next useful checks are parameter count, artifact size, and runtime latency. A unified L2 selector is worth keeping as a follow-up, but current evidence says it should treat lightweight heads as optional candidates, not the default route.
"""
    (out / "lightweight_fair_matrix_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize lightweight fair matrix.")
    parser.add_argument("--l1-summary", default="results/source_lightweight_full_auto_v2_20260625_fair/source_lightweight_full_auto_v2_summary.csv")
    parser.add_argument("--mainline-summary", default="results/target_direct_head_mainline_20260625/target_direct_head_mainline_summary.csv")
    parser.add_argument("--output-dir", default="results/lightweight_fair_matrix_20260625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(
        [
            from_l1_summary(Path(args.l1_summary)),
            from_mainline(Path(args.mainline_summary)),
        ],
        ignore_index=True,
    )
    summary = summary.sort_values(["ALL", "label"], na_position="last")
    summary.to_csv(out / "lightweight_fair_matrix_summary.csv", index=False)
    write_report(out, summary)
    print(f"Wrote lightweight fair matrix summary to {out}")


if __name__ == "__main__":
    main()
