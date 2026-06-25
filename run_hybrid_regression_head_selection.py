"""Evaluate simple hybrid selection between target Ridge and target MLP heads.

This is an analysis-only experiment. It reuses existing formal H1/H2 prediction
CSVs and tests whether a per-client head choice can keep H2's overall gains while
recovering H1's C5 high-CO strength.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_regression_head_ablation import read_csv, summarize, write_csv


RIDGE_PATH = Path("results/formal_target_ridge_auto_v2_20260624/formal_target_ridge_predictions.csv")
MLP_PATH = Path("results/formal_target_mlp_auto_v2_20260624/formal_target_mlp_predictions.csv")
C5_GRID_MLP_PATH = Path("results/formal_target_mlp_c5_grid_20260624/formal_target_mlp_predictions.csv")
OUT_DIR = Path("results/hybrid_regression_head_selection_20260624")

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


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["client"], row["split"], row["sample_index"])


def fnum(value: Any) -> float:
    return float(value)


def metric_value(rows: list[dict[str, Any]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in rows:
        if row.get("mode") == mode and row.get("scope") == scope:
            return fnum(row.get(metric))
    return None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ridge_rows = read_csv(RIDGE_PATH)
    mlp_rows = read_csv(MLP_PATH)
    c5_grid_rows = read_csv(C5_GRID_MLP_PATH) if C5_GRID_MLP_PATH.exists() else []
    ridge_by_key = {row_key(row): row for row in ridge_rows}
    mlp_by_key = {row_key(row): row for row in mlp_rows}
    c5_grid_by_key = {row_key(row): row for row in c5_grid_rows}

    missing_in_ridge = sorted(set(mlp_by_key) - set(ridge_by_key))
    missing_in_mlp = sorted(set(ridge_by_key) - set(mlp_by_key))
    if missing_in_ridge or missing_in_mlp:
        raise RuntimeError(
            f"Prediction row mismatch: missing_in_ridge={len(missing_in_ridge)}, "
            f"missing_in_mlp={len(missing_in_mlp)}"
        )

    merged: list[dict[str, Any]] = []
    for key in sorted(mlp_by_key):
        mlp = dict(mlp_by_key[key])
        ridge = ridge_by_key[key]
        client = mlp["client"]

        mlp_ppm = fnum(mlp["hybrid_select_mlp_plus_c4_rescue_ppm"])
        ridge_ppm = fnum(ridge["hybrid_select_ridge_plus_c4_rescue_ppm"])
        c5_grid_ppm = (
            fnum(c5_grid_by_key[key]["hybrid_select_mlp_plus_c4_rescue_ppm"])
            if key in c5_grid_by_key
            else None
        )

        # H2.1: MLP for C3/C4, Ridge for C5.
        mlp["client_hybrid_mlp_c3c4_ridge_c5_ppm"] = ridge_ppm if client == "C5" else mlp_ppm
        mlp["client_hybrid_mlp_c3c4_ridge_c5_head"] = "ridge" if client == "C5" else "mlp"

        # H2.2: MLP only for C3, Ridge for C4/C5. This checks whether the
        # MLP win mostly comes from C3 while preserving H1 on C4/C5.
        mlp["client_hybrid_mlp_c3_ridge_c4c5_ppm"] = mlp_ppm if client == "C3" else ridge_ppm
        mlp["client_hybrid_mlp_c3_ridge_c4c5_head"] = "mlp" if client == "C3" else "ridge"

        # H2.3: MLP for C3, Ridge for C4, expanded-grid C5 MLP for C5.
        if client == "C3":
            mlp["client_hybrid_mlp_c3_ridge_c4_c5grid_ppm"] = mlp_ppm
            mlp["client_hybrid_mlp_c3_ridge_c4_c5grid_head"] = "mlp"
        elif client == "C5" and c5_grid_ppm is not None:
            mlp["client_hybrid_mlp_c3_ridge_c4_c5grid_ppm"] = c5_grid_ppm
            mlp["client_hybrid_mlp_c3_ridge_c4_c5grid_head"] = "c5_grid_mlp"
        else:
            mlp["client_hybrid_mlp_c3_ridge_c4_c5grid_ppm"] = ridge_ppm
            mlp["client_hybrid_mlp_c3_ridge_c4_c5grid_head"] = "ridge"

        # H2.4: MLP for C3/C4, expanded-grid C5 MLP for C5.
        if client == "C5" and c5_grid_ppm is not None:
            mlp["client_hybrid_mlp_c3c4_c5grid_ppm"] = c5_grid_ppm
            mlp["client_hybrid_mlp_c3c4_c5grid_head"] = "c5_grid_mlp"
        else:
            mlp["client_hybrid_mlp_c3c4_c5grid_ppm"] = mlp_ppm
            mlp["client_hybrid_mlp_c3c4_c5grid_head"] = "mlp"

        # Direct results copied for side-by-side output.
        mlp["h2_mlp_plus_c4_rescue_ppm"] = mlp_ppm
        mlp["h1_ridge_plus_c4_rescue_ppm"] = ridge_ppm
        merged.append(mlp)

    write_csv(OUT_DIR / "hybrid_head_predictions.csv", merged)

    summary: list[dict[str, Any]] = []
    modes = [
        ("A0_baseline_final", "baseline_final_ppm"),
        ("H1_hybrid_ridge_plus_c4_rescue", "h1_ridge_plus_c4_rescue_ppm"),
        ("H2_hybrid_mlp_plus_c4_rescue", "h2_mlp_plus_c4_rescue_ppm"),
        ("H2_1_mlp_c3c4_ridge_c5", "client_hybrid_mlp_c3c4_ridge_c5_ppm"),
        ("H2_2_mlp_c3_ridge_c4c5", "client_hybrid_mlp_c3_ridge_c4c5_ppm"),
        ("H2_3_mlp_c3_ridge_c4_c5grid", "client_hybrid_mlp_c3_ridge_c4_c5grid_ppm"),
        ("H2_4_mlp_c3c4_c5grid", "client_hybrid_mlp_c3c4_c5grid_ppm"),
    ]
    for mode, pred_key in modes:
        summary.extend(summarize(merged, pred_key, mode, "test"))
    write_csv(OUT_DIR / "hybrid_head_summary.csv", summary)

    md = [
        "# Hybrid Regression Head Selection",
        "",
        "This analysis combines already-fit H1 Ridge and H2 shallow MLP heads.",
        "No new fitting is performed here.",
        "",
        "## Candidates",
        "",
        "- H2.1: MLP for C3/C4, Ridge for C5.",
        "- H2.2: MLP for C3, Ridge for C4/C5.",
        "- H2.3: MLP for C3, Ridge for C4, expanded-grid MLP for C5.",
        "- H2.4: MLP for C3/C4, expanded-grid MLP for C5.",
        "",
        "## RMSE Table",
        "",
        "| mode | ALL | C3 CO | C4 CO | C5 CO | C3 CO high | C4 CO high | C5 CO high | nonCO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, _ in modes:
        md.append(
            "| {mode} | {all} | {c3} | {c4} | {c5} | {c3h} | {c4h} | {c5h} | {nonco} |".format(
                mode=mode,
                all=fmt(metric_value(summary, mode, "ALL")),
                c3=fmt(metric_value(summary, mode, "C3-CO")),
                c4=fmt(metric_value(summary, mode, "C4-CO")),
                c5=fmt(metric_value(summary, mode, "C5-CO")),
                c3h=fmt(metric_value(summary, mode, "C3-CO_high_200_250")),
                c4h=fmt(metric_value(summary, mode, "C4-CO_high_200_250")),
                c5h=fmt(metric_value(summary, mode, "C5-CO_high_200_250")),
                nonco=fmt(metric_value(summary, mode, "nonCO_ALL")),
            )
        )
    md.extend(
        [
            "",
            "## Notes",
            "",
            "- H2.1 tests the obvious repair for H2's weaker C5 high-CO result.",
            "- H2.2 tests whether C3 alone accounts for most of the MLP benefit.",
        ]
    )
    (OUT_DIR / "hybrid_head_selection_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "ridge_predictions": str(RIDGE_PATH),
        "mlp_predictions": str(MLP_PATH),
        "c5_grid_mlp_predictions": str(C5_GRID_MLP_PATH) if C5_GRID_MLP_PATH.exists() else None,
        "output_dir": str(OUT_DIR),
        "row_count": len(merged),
        "modes": [mode for mode, _ in modes],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "rows": len(merged)}, indent=2))


if __name__ == "__main__":
    main()
