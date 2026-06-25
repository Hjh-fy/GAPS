"""C45 -> C123 target-direct optimal-config analysis.

This script summarizes the reverse direction with the same no-QC full-set
priority used for C12 -> C345:

- baseline R3aK16 + original auto_v2 final ppm;
- formal target Ridge direct head;
- formal target MLP direct head;
- calibration-guided Ridge/MLP client hybrids;
- test-only hybrid oracle for diagnostic headroom only.

The script does not retrain models. It combines the existing formal Ridge/MLP
prediction artifacts for source C4/C5 -> target C1/C2/C3.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_regression_head_ablation import summarize, write_csv


TARGET_CLIENTS = ["C1", "C2", "C3"]
SCOPES = [
    "ALL",
    "C1-CO",
    "C2-CO",
    "C3-CO",
    "C1-CO_high_200_250",
    "C2-CO_high_200_250",
    "C3-CO_high_200_250",
    "nonCO_ALL",
]


def key_cols(df: pd.DataFrame) -> pd.Series:
    return df["client"].astype(str) + "|" + df["split"].astype(str) + "|" + df["sample_index"].astype(int).astype(str)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metric_value(summary: list[dict[str, Any]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope:
            value = row.get(metric)
            return None if value in (None, "") else float(value)
    return None


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def load_predictions(ridge_path: Path, mlp_path: Path, source_aug_path: Path | None = None) -> pd.DataFrame:
    ridge = pd.read_csv(ridge_path)
    mlp = pd.read_csv(mlp_path)
    ridge = ridge[ridge["split"].astype(str) == "test"].copy()
    mlp = mlp[mlp["split"].astype(str) == "test"].copy()
    ridge["key"] = key_cols(ridge)
    mlp["key"] = key_cols(mlp)
    keep = [
        "key",
        "baseline_final_ppm",
        "ridge_direct_ppm",
        "val_select_ridge_ppm",
        "hybrid_select_ridge_ppm",
        "ridge_direct_plus_c4_rescue_ppm",
    ]
    merged = ridge.merge(
        mlp[
            [
                "key",
                "mlp_direct_ppm",
                "val_select_mlp_ppm",
                "hybrid_select_mlp_ppm",
                "mlp_direct_plus_c4_rescue_ppm",
            ]
        ],
        on="key",
        how="inner",
    )
    if len(merged) != len(ridge) or len(merged) != len(mlp):
        raise RuntimeError(f"Prediction row mismatch: ridge={len(ridge)} mlp={len(mlp)} merged={len(merged)}")
    base_cols = [
        "client",
        "split",
        "sample_index",
        "true_class",
        "true_gas",
        "true_ppm",
        "pred_class",
        "final_ppm",
        "risk_score",
        "response_phase",
        "phase_label",
    ]
    out = merged[
        [
            "key",
            *base_cols,
            *[c for c in keep if c != "key"],
            "mlp_direct_ppm",
            "val_select_mlp_ppm",
            "hybrid_select_mlp_ppm",
            "mlp_direct_plus_c4_rescue_ppm",
        ]
    ].copy()
    if source_aug_path and source_aug_path.exists():
        source_aug = pd.read_csv(source_aug_path)
        source_aug = source_aug[source_aug["split"].astype(str) == "test"].copy()
        source_aug["key"] = key_cols(source_aug)
        out = out.merge(
            source_aug[["key", "target_ridge_plus_source_preds_ppm"]].rename(
                columns={"target_ridge_plus_source_preds_ppm": "source_aug_target_ridge_ppm"}
            ),
            on="key",
            how="left",
        )
    return out


def load_selection(ridge_selection: Path, mlp_selection: Path) -> dict[str, str]:
    ridge = pd.read_csv(ridge_selection)
    mlp = pd.read_csv(mlp_selection)
    out: dict[str, str] = {}
    for client in TARGET_CLIENTS:
        r = ridge[ridge["client"].astype(str) == client].iloc[0]
        m = mlp[mlp["client"].astype(str) == client].iloc[0]
        ridge_all = fnum(r["ridge_ALL_RMSE"])
        mlp_all = fnum(m["mlp_ALL_RMSE"])
        out[client] = "mlp" if mlp_all < ridge_all else "ridge"
    return out


def add_hybrids(df: pd.DataFrame, calibration_profile: dict[str, str]) -> list[tuple[str, str, str]]:
    modes = [
        ("A0_baseline_final", "baseline_final_ppm", "reference"),
        ("H1_target_Ridge_direct", "ridge_direct_ppm", "formal direct head"),
        ("H2_target_MLP_direct", "mlp_direct_ppm", "formal direct head"),
    ]
    if "source_aug_target_ridge_ppm" in df.columns:
        modes.append(("H2b_source_aug_target_Ridge", "source_aug_target_ridge_ppm", "source-augmented target Ridge"))
        df["H8_style_source_aug_CO_else_Ridge_ppm"] = df["ridge_direct_ppm"].astype(float)
        mask = df["pred_class"].astype(int) == 1
        df.loc[mask, "H8_style_source_aug_CO_else_Ridge_ppm"] = df.loc[mask, "source_aug_target_ridge_ppm"].astype(float)
        modes.append(
            (
                "H8_style_source_aug_CO_else_Ridge",
                "H8_style_source_aug_CO_else_Ridge_ppm",
                "CO specialist switch + Ridge fallback",
            )
        )
    df["calibration_client_hybrid_ppm"] = df["ridge_direct_ppm"].astype(float)
    for client, head in calibration_profile.items():
        mask = df["client"].astype(str) == client
        if head == "mlp":
            df.loc[mask, "calibration_client_hybrid_ppm"] = df.loc[mask, "mlp_direct_ppm"].astype(float)
    modes.append(("H3_calibration_client_hybrid", "calibration_client_hybrid_ppm", "calibration-selected Ridge/MLP"))

    # Diagnostic only: choose the best client-level Ridge/MLP assignment by test ALL RMSE.
    best_combo: tuple[str, ...] | None = None
    best_rmse = float("inf")
    tmp_rows: list[dict[str, Any]] = []
    for combo in itertools.product(["ridge", "mlp"], repeat=len(TARGET_CLIENTS)):
        col = "tmp_combo_ppm"
        df[col] = df["ridge_direct_ppm"].astype(float)
        for client, head in zip(TARGET_CLIENTS, combo):
            if head == "mlp":
                df.loc[df["client"].astype(str) == client, col] = df.loc[
                    df["client"].astype(str) == client, "mlp_direct_ppm"
                ].astype(float)
        rows = summarize(df.to_dict("records"), col, "tmp", "test")
        all_rmse = metric_value(rows, "tmp", "ALL")
        tmp_rows.append({"combo": ",".join(f"{c}:{h}" for c, h in zip(TARGET_CLIENTS, combo)), "ALL_RMSE": all_rmse})
        if all_rmse is not None and all_rmse < best_rmse:
            best_rmse = all_rmse
            best_combo = combo
    if best_combo is None:
        return modes
    df["test_oracle_client_hybrid_ppm"] = df["ridge_direct_ppm"].astype(float)
    for client, head in zip(TARGET_CLIENTS, best_combo):
        if head == "mlp":
            df.loc[df["client"].astype(str) == client, "test_oracle_client_hybrid_ppm"] = df.loc[
                df["client"].astype(str) == client, "mlp_direct_ppm"
            ].astype(float)
    modes.append(("H4_test_oracle_client_hybrid", "test_oracle_client_hybrid_ppm", "test-only diagnostic"))
    df.attrs["test_oracle_combos"] = tmp_rows
    df.attrs["best_test_oracle_combo"] = ",".join(f"{c}:{h}" for c, h in zip(TARGET_CLIENTS, best_combo))
    return modes


def write_report(
    out: Path,
    summary: list[dict[str, Any]],
    modes: list[tuple[str, str, str]],
    calibration_profile: dict[str, str],
    oracle_rows: list[dict[str, Any]],
    best_oracle: str,
) -> None:
    headers = ["mode", "family", "ALL", "NRMSE", "C1 CO", "C2 CO", "C3 CO", "C1 high", "C2 high", "C3 high", "nonCO"]
    lines = [
        "# C45 -> C123 Optimal-Config Target Analysis",
        "",
        "Scope: source clients C4/C5, target clients C1/C2/C3, target test, no-QC full-set.",
        "",
        f"- Calibration-selected client profile: `{json.dumps(calibration_profile, ensure_ascii=False)}`",
        f"- Best test-only diagnostic profile: `{best_oracle}`",
        "",
        "## Test Metrics",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for mode, _pred_key, family in modes:
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    family,
                    fmt(metric_value(summary, mode, "ALL")),
                    fmt(metric_value(summary, mode, "ALL", "NRMSE"), 4),
                    fmt(metric_value(summary, mode, "C1-CO")),
                    fmt(metric_value(summary, mode, "C2-CO")),
                    fmt(metric_value(summary, mode, "C3-CO")),
                    fmt(metric_value(summary, mode, "C1-CO_high_200_250")),
                    fmt(metric_value(summary, mode, "C2-CO_high_200_250")),
                    fmt(metric_value(summary, mode, "C3-CO_high_200_250")),
                    fmt(metric_value(summary, mode, "nonCO_ALL")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Test-Only Hybrid Grid",
            "",
            "| combo | ALL RMSE |",
            "| --- | ---: |",
        ]
    )
    for row in sorted(oracle_rows, key=lambda r: fnum(r["ALL_RMSE"])):
        lines.append(f"| {row['combo']} | {fmt(fnum(row['ALL_RMSE']))} |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Reverse direction C45 -> C123 benefits strongly from target direct-head calibration.",
            "- Formal target Ridge is the current best clean reverse-direction candidate among Ridge/MLP direct heads.",
            "- Source-augmented target Ridge and H8-style CO switching improve C3 CO/high-CO, but they worsen ALL RMSE and nonCO, so they are diagnostic CO-specialist variants rather than the reverse mainline.",
            "- Calibration selection prefers MLP on C1/C2/C3, but this overfits calibration badly; test ALL is worse than all-Ridge.",
            "- The test-only oracle is diagnostic only. If it does not beat Ridge materially, there is little reason to build a more complex H2.3-style profile for this direction.",
            "- C4 route-rescue is not relevant in this direction because C4 is a source client, not a target client.",
            "",
        ]
    )
    (out / "c45_c123_optimal_config_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze C45 -> C123 optimal target-direct configs.")
    parser.add_argument("--ridge-predictions", default="results/formal_target_ridge_auto_v2_c45_c123_20260625/formal_target_ridge_predictions.csv")
    parser.add_argument("--mlp-predictions", default="results/formal_target_mlp_auto_v2_c45_c123_20260625/formal_target_mlp_predictions.csv")
    parser.add_argument("--ridge-selection", default="results/formal_target_ridge_auto_v2_c45_c123_20260625/formal_target_ridge_selection_table.csv")
    parser.add_argument("--mlp-selection", default="results/formal_target_mlp_auto_v2_c45_c123_20260625/formal_target_mlp_selection_table.csv")
    parser.add_argument(
        "--source-aug-predictions",
        default="results/source_augmented_target_ridge_c45_c123_20260626/target_predictions_plus_source_preds_plus_c4_rescue.csv",
    )
    parser.add_argument("--output-dir", default="results/c45_c123_optimal_config_analysis_20260626")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_predictions(Path(args.ridge_predictions), Path(args.mlp_predictions), Path(args.source_aug_predictions))
    profile = load_selection(Path(args.ridge_selection), Path(args.mlp_selection))
    modes = add_hybrids(df, profile)

    summary: list[dict[str, Any]] = []
    rows = df.to_dict("records")
    for mode, pred_key, _family in modes:
        summary.extend(summarize(rows, pred_key, mode, "test"))

    write_csv(out / "c45_c123_optimal_config_summary.csv", summary)
    df.to_csv(out / "c45_c123_optimal_config_predictions.csv", index=False)
    oracle_rows = list(df.attrs.get("test_oracle_combos", []))
    best_oracle = str(df.attrs.get("best_test_oracle_combo", ""))
    write_csv(out / "c45_c123_test_oracle_hybrid_grid.csv", oracle_rows)
    write_report(out, summary, modes, profile, oracle_rows, best_oracle)
    manifest = {
        "ridge_predictions": args.ridge_predictions,
        "mlp_predictions": args.mlp_predictions,
        "ridge_selection": args.ridge_selection,
        "mlp_selection": args.mlp_selection,
        "source_aug_predictions": args.source_aug_predictions,
        "output_dir": args.output_dir,
        "rows": int(len(df)),
        "calibration_profile": profile,
        "best_test_oracle_combo": best_oracle,
        "modes": [{"mode": mode, "pred_key": pred_key, "family": family} for mode, pred_key, family in modes],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "rows": len(df), "modes": len(modes)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
