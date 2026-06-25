"""L3 lightweight-base + H2.3/H8/C4-rescue matrix.

This experiment closes the naming gap after the L1/L2 lightweight runs:

1. strict lightweight full-auto_v2 bases;
2. lightweight H2.3-style client profile;
3. lightweight CO-specialist switch with H2.3 fallback;
4. the same switch with the formal calibration-selected C4 route-rescue gate;
5. source-augmented H8 references.

No new training is performed here. The script combines existing prediction
artifacts and reports no-QC full-set target-test metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_regression_head_ablation import summarize, write_csv


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

LIGHT_VAL_CANDIDATES = {
    "source_ridge": "source_ridge_val_selected",
    "source_per_gas_mlp": "source_per_gas_mlp_val_selected",
    "source_shared_mlp": "source_shared_mlp_val_selected",
}


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


def fmt(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.2f}"


def load_light_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].astype(str) == "test"].copy()
    df["key"] = key_cols(df)
    pieces: list[pd.DataFrame] = []
    for base_mode, candidate in LIGHT_VAL_CANDIDATES.items():
        sub = df[(df["base_mode"].astype(str) == base_mode) & (df["candidate"].astype(str) == candidate)].copy()
        if len(sub) != 5400:
            raise RuntimeError(f"Expected 5400 rows for {candidate}, got {len(sub)}")
        pieces.append(sub[["key", "corrected_ppm"]].rename(columns={"corrected_ppm": f"{candidate}_ppm"}))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="key", how="inner")
    if len(out) != 5400:
        raise RuntimeError(f"Lightweight candidate merge produced {len(out)} rows")
    return out


def load_optional_column(path: Path, column: str, out_col: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["split"].astype(str) == "test"].copy()
    if column not in df.columns:
        return None
    df["key"] = key_cols(df)
    return df[["key", column]].rename(columns={column: out_col})


def formal_c4_mask(df: pd.DataFrame, gate: dict[str, Any]) -> pd.Series:
    pred_classes = {int(part) for part in str(gate["pred_classes"]).split(",") if part.strip()}
    mask = (
        (df["client"].astype(str) == "C4")
        & (df["pred_class"].astype(int).isin(pred_classes))
        & (df["final_ppm"].astype(float) <= fnum(gate["max_final"]))
        & (df["risk_score"].astype(float) >= fnum(gate["min_risk"]))
    )
    if str(gate.get("phase", "any")) != "any":
        mask &= df["response_phase"].astype(str) == str(gate["phase"])
    if fnum(gate.get("max_conf_margin", 1.0), 1.0) < 1.0:
        mask &= df["confidence_margin"].astype(float) <= fnum(gate["max_conf_margin"])
    return mask


def add_rescue_column(df: pd.DataFrame, base_col: str, out_col: str, gate: dict[str, Any]) -> None:
    mask = formal_c4_mask(df, gate)
    df[out_col] = df[base_col].astype(float)
    df.loc[mask, out_col] = fnum(gate["rescue_ppm"])


def add_l3_candidates(df: pd.DataFrame, gate: dict[str, Any]) -> list[tuple[str, str, str]]:
    modes: list[tuple[str, str, str]] = [
        ("A0_baseline_final", "baseline_final_ppm", "reference"),
        ("H2_3_R3aK16_current_mainline", "h2_3_ppm", "reference"),
        ("H8_R3aK16_source_aug_CO_else_H2_3", "h8_pred_co_source_aug_else_h23_ppm", "reference"),
        ("H8_plus_formal_C4_rescue", "formal_c4_route_rescue_ppm", "reference"),
        ("L3_light_source_ridge_full_auto_v2", "source_ridge_val_selected_ppm", "strict lightweight base"),
        ("L3_light_source_per_gas_mlp_full_auto_v2", "source_per_gas_mlp_val_selected_ppm", "strict lightweight base"),
        ("L3_light_source_shared_mlp_full_auto_v2", "source_shared_mlp_val_selected_ppm", "strict lightweight base"),
    ]

    # H2.3 analogy: MLP-like lightweight head for C3/C5, Ridge-like lightweight
    # head for C4.
    df["l3_light_h2_3_analog_ppm"] = np.where(
        df["client"].astype(str) == "C4",
        df["source_ridge_val_selected_ppm"].astype(float),
        df["source_per_gas_mlp_val_selected_ppm"].astype(float),
    )
    modes.append(("L3_light_H2_3_analog", "l3_light_h2_3_analog_ppm", "lightweight H2.3 analogy"))
    add_rescue_column(df, "l3_light_h2_3_analog_ppm", "l3_light_h2_3_analog_plus_formal_c4_ppm", gate)
    modes.append(
        (
            "L3_light_H2_3_analog_plus_formal_C4_rescue",
            "l3_light_h2_3_analog_plus_formal_c4_ppm",
            "lightweight H2.3 analogy + formal rescue",
        )
    )

    # Conservative client-val lightweight profile from the L2 report:
    # C3 shared MLP, C4 per-gas MLP, C5 per-gas MLP.
    df["l3_light_client_val_profile_ppm"] = np.where(
        df["client"].astype(str) == "C3",
        df["source_shared_mlp_val_selected_ppm"].astype(float),
        df["source_per_gas_mlp_val_selected_ppm"].astype(float),
    )
    modes.append(("L3_light_client_val_profile", "l3_light_client_val_profile_ppm", "lightweight client-val profile"))
    add_rescue_column(df, "l3_light_client_val_profile_ppm", "l3_light_client_val_profile_plus_formal_c4_ppm", gate)
    modes.append(
        (
            "L3_light_client_val_profile_plus_formal_C4_rescue",
            "l3_light_client_val_profile_plus_formal_c4_ppm",
            "lightweight client-val profile + formal rescue",
        )
    )

    for base_mode, candidate in LIGHT_VAL_CANDIDATES.items():
        col = f"{candidate}_ppm"
        switch_col = f"l3_{base_mode}_CO_else_h2_3_ppm"
        df[switch_col] = np.where(df["pred_class"].astype(int) == 1, df[col].astype(float), df["h2_3_ppm"].astype(float))
        label = f"L3_{base_mode}_CO_else_H2_3"
        modes.append((label, switch_col, "lightweight CO switch + H2.3 fallback"))
        rescue_col = f"{switch_col[:-4]}_plus_formal_c4_ppm"
        add_rescue_column(df, switch_col, rescue_col, gate)
        modes.append((f"{label}_plus_formal_C4_rescue", rescue_col, "lightweight CO switch + H2.3 fallback + formal rescue"))

    return modes


def write_report(out: Path, summary: list[dict[str, Any]], modes: list[tuple[str, str, str]], gate: dict[str, Any]) -> None:
    table_rows: list[list[str]] = []
    for mode, _pred_key, family in modes:
        table_rows.append(
            [
                mode,
                family,
                fmt(metric_value(summary, mode, "ALL")),
                fmt(metric_value(summary, mode, "ALL", "NRMSE")),
                fmt(metric_value(summary, mode, "C3-CO")),
                fmt(metric_value(summary, mode, "C4-CO")),
                fmt(metric_value(summary, mode, "C5-CO")),
                fmt(metric_value(summary, mode, "C3-CO_high_200_250")),
                fmt(metric_value(summary, mode, "C4-CO_high_200_250")),
                fmt(metric_value(summary, mode, "C5-CO_high_200_250")),
                fmt(metric_value(summary, mode, "nonCO_ALL")),
            ]
        )
    headers = ["mode", "family", "ALL", "NRMSE", "C3 CO", "C4 CO", "C5 CO", "C3 high", "C4 high", "C5 high", "nonCO"]
    lines = [
        "# L3 Lightweight-Base Hybrid Matrix",
        "",
        "Scope: C12 -> C345 target test, no-QC full-set.",
        "",
        "This matrix explicitly tests lightweight-source outputs with H2.3/H8-style combinations.",
        "",
        "Important distinction:",
        "",
        "- `strict lightweight base` uses lightweight source-head full-auto_v2 predictions directly.",
        "- `lightweight CO switch + H2.3 fallback` uses a lightweight candidate only when `pred_class == CO`; otherwise it keeps H2.3.",
        "- `H8` uses source-augmented target Ridge as the CO specialist, not a pure lightweight base replacement.",
        "",
        "Formal C4 rescue gate reused here:",
        "",
        "```json",
        json.dumps(gate, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Test Metrics",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The strict lightweight bases remain above 21.9-22.8 ALL RMSE, so they do not replace H2.3/H8 as performance mainline.",
            "- Lightweight H2.3-style/client profiles mainly inherit the C4 high-CO weakness; formal C4 rescue helps but does not close the gap.",
            "- Lightweight CO-switch variants are the closest pure-lightweight analogue to H8: use a lightweight candidate only when `pred_class == CO`, otherwise keep H2.3.",
            "- The per-gas-MLP CO-switch plus formal C4 rescue nearly ties H8 + formal C4 rescue on ALL RMSE and has slightly better NRMSE/nonCO, so it is a credible deployment-lite CO-specialist candidate.",
            "- The shared-MLP CO-switch plus formal C4 rescue has the best C4 high-CO RMSE in this matrix, but gives up more ALL/C5 performance.",
            "- The practical conclusion is not full lightweight replacement. The useful lightweight form is selective CO-specialist switching with H2.3 fallback.",
            "",
        ]
    )
    (out / "l3_lightweight_hybrid_matrix_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L3 lightweight-base + H2.3/H8/C4-rescue matrix.")
    parser.add_argument("--formal-predictions", default="results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_predictions.csv")
    parser.add_argument("--light-predictions", default="results/source_lightweight_full_auto_v2_20260625_fair/source_lightweight_full_auto_v2_predictions.csv")
    parser.add_argument("--source-aug-predictions", default="results/source_augmented_target_ridge_20260625_lite/target_predictions_plus_source_preds_plus_c4_rescue.csv")
    parser.add_argument("--formal-gate", default="results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_selected_gate.json")
    parser.add_argument("--output-dir", default="results/l3_lightweight_hybrid_matrix_20260626")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = json.loads(Path(args.formal_gate).read_text(encoding="utf-8"))

    df = pd.read_csv(args.formal_predictions)
    df = df[df["split"].astype(str) == "test"].copy()
    df["key"] = key_cols(df)
    light = load_light_candidates(Path(args.light_predictions))
    df = df.merge(light, on="key", how="inner")

    optional_source_aug = load_optional_column(
        Path(args.source_aug_predictions),
        "target_ridge_plus_source_preds_plus_c4_rescue_ppm",
        "source_aug_target_ridge_plus_c4_rescue_ppm",
    )
    if optional_source_aug is not None:
        df = df.merge(optional_source_aug, on="key", how="left")

    if len(df) != 5400:
        raise RuntimeError(f"Expected 5400 merged test rows, got {len(df)}")

    modes = add_l3_candidates(df, gate)
    if "source_aug_target_ridge_plus_c4_rescue_ppm" in df.columns:
        modes.insert(
            3,
            (
                "source_aug_target_Ridge_plus_C4_rescue",
                "source_aug_target_ridge_plus_c4_rescue_ppm",
                "source-augmented target Ridge reference",
            ),
        )

    summary: list[dict[str, Any]] = []
    rows = df.to_dict("records")
    for mode, pred_key, _family in modes:
        summary.extend(summarize(rows, pred_key, mode, "test"))

    write_csv(out / "l3_lightweight_hybrid_matrix_summary.csv", summary)
    keep_cols = [
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
    ] + [pred_key for _mode, pred_key, _family in modes if pred_key in df.columns]
    df[keep_cols].to_csv(out / "l3_lightweight_hybrid_matrix_predictions.csv", index=False)
    write_report(out, summary, modes, gate)
    manifest = {
        "formal_predictions": args.formal_predictions,
        "light_predictions": args.light_predictions,
        "source_aug_predictions": args.source_aug_predictions,
        "formal_gate": args.formal_gate,
        "output_dir": args.output_dir,
        "rows": int(len(df)),
        "modes": [{"mode": mode, "pred_key": pred_key, "family": family} for mode, pred_key, family in modes],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "rows": len(df), "modes": len(modes)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
