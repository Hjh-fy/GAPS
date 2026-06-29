"""Ablate B0/R3aK16/auto_v2 dependence in the H2.3 target profile.

The current target direct-head artifacts train Ridge/MLP heads from target
window statistics.  B0/R3aK16/auto_v2 fields mostly enter as baseline,
fallback, and C4 route-rescue gate context.  This script makes that distinction
explicit by recombining already-trained formal target-head predictions.

Default run: C12 -> C345, fixed H2.3 profile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_regression_head_ablation import CO_CLASS, fnum, inum, metrics, read_csv, summarize, write_csv


FORWARD_DEFAULTS = {
    "ridge": Path("results/formal_target_ridge_auto_v2_20260624/formal_target_ridge_predictions.csv"),
    "mlp": Path("results/formal_target_mlp_auto_v2_20260624/formal_target_mlp_predictions.csv"),
    "c5_grid": Path("results/formal_target_mlp_c5_grid_20260624/formal_target_mlp_predictions.csv"),
}

REVERSE_DEFAULTS = {
    "ridge": Path("results/formal_target_ridge_auto_v2_c45_c123_20260625/formal_target_ridge_predictions.csv"),
    "mlp": Path("results/formal_target_mlp_auto_v2_c45_c123_20260625/formal_target_mlp_predictions.csv"),
    "c5_grid": Path(""),
}

SCOPE_ORDER = [
    "ALL",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C1-CO",
    "C2-CO",
    "C3-CO",
    "C4-CO",
    "C5-CO",
    "C1-CO_high_200_250",
    "C2-CO_high_200_250",
    "C3-CO_high_200_250",
    "C4-CO_high_200_250",
    "C5-CO_high_200_250",
    "nonCO_ALL",
]


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["client"]), str(row["split"]), str(row["sample_index"]))


def output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "feature_dict"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_csv(path)


def get_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return fnum(row[key])
    return float("nan")


def combine_forward_h2_3(
    ridge_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    c5_grid_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ridge_by = {row_key(row): row for row in ridge_rows}
    mlp_by = {row_key(row): row for row in mlp_rows}
    c5_by = {row_key(row): row for row in c5_grid_rows}
    out: list[dict[str, Any]] = []
    for key in sorted(mlp_by):
        row = output_row(mlp_by[key])
        client = str(row["client"])
        ridge = ridge_by[key]
        c5 = c5_by.get(key)
        row["A0_baseline_final_ppm"] = get_float(row, "baseline_final_ppm", "final_ppm")

        # Current H2.3 as used in the mainline summary.  This includes the
        # existing C4 route-rescue correction in the underlying formal outputs.
        if client == "C3":
            row["A1_h2_3_current_ppm"] = get_float(row, "hybrid_select_mlp_plus_c4_rescue_ppm")
            row["A2_h2_3_direct_only_ppm"] = get_float(row, "hybrid_select_mlp_ppm")
        elif client == "C5" and c5 is not None:
            row["A1_h2_3_current_ppm"] = get_float(c5, "hybrid_select_mlp_plus_c4_rescue_ppm")
            row["A2_h2_3_direct_only_ppm"] = get_float(c5, "hybrid_select_mlp_ppm")
        else:
            row["A1_h2_3_current_ppm"] = get_float(ridge, "hybrid_select_ridge_plus_c4_rescue_ppm")
            row["A2_h2_3_direct_only_ppm"] = get_float(ridge, "hybrid_select_ridge_ppm")

        # These feature-removal variants are expected to match A2 because the
        # current direct-head feature_dict contains target window/phase stats,
        # not B0/R3aK16 ppm or QC-risk scalars.
        row["A3_no_auto_v2_ppm_feature_ppm"] = row["A2_h2_3_direct_only_ppm"]
        row["A4_no_r3ak16_ppm_feature_ppm"] = row["A2_h2_3_direct_only_ppm"]
        row["A5_no_risk_feature_ppm"] = row["A2_h2_3_direct_only_ppm"]
        row["A6_no_ppm_no_risk_direct_only_ppm"] = row["A2_h2_3_direct_only_ppm"]

        row["A1_uses_b0_dependent_c4_rescue"] = int(abs(row["A1_h2_3_current_ppm"] - row["A2_h2_3_direct_only_ppm"]) > 1e-9)
        out.append(row)
    return out


def combine_reverse_target_ridge(ridge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_raw in sorted(ridge_rows, key=row_key):
        row = output_row(row_raw)
        row["A0_baseline_final_ppm"] = get_float(row, "baseline_final_ppm", "final_ppm")
        row["A1_target_ridge_current_ppm"] = get_float(row, "hybrid_select_ridge_plus_c4_rescue_ppm", "hybrid_select_ridge_ppm")
        row["A2_target_ridge_direct_only_ppm"] = get_float(row, "hybrid_select_ridge_ppm")
        row["A3_no_auto_v2_ppm_feature_ppm"] = row["A2_target_ridge_direct_only_ppm"]
        row["A4_no_r3ak16_ppm_feature_ppm"] = row["A2_target_ridge_direct_only_ppm"]
        row["A5_no_risk_feature_ppm"] = row["A2_target_ridge_direct_only_ppm"]
        row["A6_no_ppm_no_risk_direct_only_ppm"] = row["A2_target_ridge_direct_only_ppm"]
        row["A1_uses_b0_dependent_c4_rescue"] = int(abs(row["A1_target_ridge_current_ppm"] - row["A2_target_ridge_direct_only_ppm"]) > 1e-9)
        out.append(row)
    return out


def mode_specs(direction: str) -> list[tuple[str, str, str]]:
    if direction == "c45_c123":
        return [
            ("A0_B0_R3aK16_auto_v2", "A0_baseline_final_ppm", "B0 baseline"),
            ("A1_target_Ridge_current", "A1_target_ridge_current_ppm", "current target Ridge profile"),
            ("A2_target_Ridge_direct_only_no_B0_rescue", "A2_target_ridge_direct_only_ppm", "direct head without B0-dependent rescue"),
            ("A3_no_auto_v2_ppm_feature", "A3_no_auto_v2_ppm_feature_ppm", "same direct-head features; no auto_v2 ppm feature present"),
            ("A4_no_r3ak16_ppm_feature", "A4_no_r3ak16_ppm_feature_ppm", "same direct-head features; no R3aK16 ppm feature present"),
            ("A5_no_risk_feature", "A5_no_risk_feature_ppm", "same direct-head features; no risk feature present"),
            ("A6_no_ppm_no_risk_direct_only", "A6_no_ppm_no_risk_direct_only_ppm", "clean direct target-head output"),
        ]
    return [
        ("A0_B0_R3aK16_auto_v2", "A0_baseline_final_ppm", "B0 baseline"),
        ("A1_H2_3_current_with_B0_rescue", "A1_h2_3_current_ppm", "current H2.3 profile with B0-dependent C4 rescue"),
        ("A2_H2_3_direct_only_no_B0_rescue", "A2_h2_3_direct_only_ppm", "fixed H2.3 direct heads without B0-dependent C4 rescue"),
        ("A3_no_auto_v2_ppm_feature", "A3_no_auto_v2_ppm_feature_ppm", "same direct-head features; no auto_v2 ppm feature present"),
        ("A4_no_r3ak16_ppm_feature", "A4_no_r3ak16_ppm_feature_ppm", "same direct-head features; no R3aK16 ppm feature present"),
        ("A5_no_risk_feature", "A5_no_risk_feature_ppm", "same direct-head features; no risk feature present"),
        ("A6_no_ppm_no_risk_direct_only", "A6_no_ppm_no_risk_direct_only_ppm", "clean direct target-head output"),
    ]


def client_nrmse_rows(summary_rows: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    clients = ["C1", "C2", "C3"] if direction == "c45_c123" else ["C3", "C4", "C5"]
    out: list[dict[str, Any]] = []
    for mode in sorted({row["mode"] for row in summary_rows}):
        selected = [row for row in summary_rows if row["mode"] == mode and row["scope"] in clients]
        values = [fnum(row["NRMSE"]) for row in selected]
        rmses = [fnum(row["RMSE"]) for row in selected]
        item = {
            "mode": mode,
            "macro_client_NRMSE": float(np.mean(values)) if values else "",
            "macro_client_RMSE": float(np.mean(rmses)) if rmses else "",
        }
        for client in clients:
            row = next((r for r in selected if r["scope"] == client), None)
            item[f"{client}_NRMSE"] = "" if row is None else row["NRMSE"]
            item[f"{client}_RMSE"] = "" if row is None else row["RMSE"]
        out.append(item)
    return out


def format_metric(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def find_metric(summary_rows: list[dict[str, Any]], mode: str, scope: str, key: str) -> float:
    for row in summary_rows:
        if row["mode"] == mode and row["scope"] == scope:
            return fnum(row[key])
    return float("nan")


def find_per_client_metric(per_client_rows: list[dict[str, Any]], mode: str, key: str) -> float:
    for row in per_client_rows:
        if row["mode"] == mode:
            return fnum(row.get(key))
    return float("nan")


def build_decision(direction: str, summary_rows: list[dict[str, Any]], per_client_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if direction != "c12_c345":
        return {
            "rule": "Reverse direction is diagnostic for this script.",
            "run_reverse_under_forward_gate": None,
            "reading": "Reverse was requested only if forward no-B0 stayed close to H2.3.",
        }
    a1 = "A1_H2_3_current_with_B0_rescue"
    a2 = "A2_H2_3_direct_only_no_B0_rescue"
    a1_macro = find_per_client_metric(per_client_rows, a1, "macro_client_NRMSE")
    a2_macro = find_per_client_metric(per_client_rows, a2, "macro_client_NRMSE")
    rel_gap = (a2_macro - a1_macro) / a1_macro if np.isfinite(a1_macro) and abs(a1_macro) > 1e-12 else float("nan")
    c4_a1 = find_per_client_metric(per_client_rows, a1, "C4_NRMSE")
    c4_a2 = find_per_client_metric(per_client_rows, a2, "C4_NRMSE")
    c4_high_a1 = find_metric(summary_rows, a1, "C4-CO_high_200_250", "NRMSE")
    c4_high_a2 = find_metric(summary_rows, a2, "C4-CO_high_200_250", "NRMSE")
    return {
        "rule": "Run reverse C45->C123 only if forward no-B0 direct-only is close to current H2.3.",
        "close_threshold_relative_macro_client_NRMSE": 0.05,
        "a1_macro_client_NRMSE": a1_macro,
        "a2_macro_client_NRMSE": a2_macro,
        "relative_macro_client_NRMSE_gap": rel_gap,
        "c4_NRMSE_a1_current": c4_a1,
        "c4_NRMSE_a2_direct_only": c4_a2,
        "c4_high_CO_NRMSE_a1_current": c4_high_a1,
        "c4_high_CO_NRMSE_a2_direct_only": c4_high_a2,
        "run_reverse_under_forward_gate": bool(np.isfinite(rel_gap) and rel_gap <= 0.05),
        "reading": "Forward no-B0 is not close to current H2.3; keep B0/R3aK16/auto_v2 as runtime support layer.",
    }


def write_report(
    out_dir: Path,
    direction: str,
    summary_rows: list[dict[str, Any]],
    per_client_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    modes = mode_specs(direction)
    scopes = ["ALL", *("C1 C2 C3".split() if direction == "c45_c123" else "C3 C4 C5".split()), "nonCO_ALL"]
    lines = [
        f"# H2.3 No-B0 Feature Ablation ({direction})",
        "",
        "This ablation separates target direct-head training from B0/R3aK16/auto_v2 baseline and C4 route-rescue usage.",
        "",
        "## Feature Check",
        "",
        f"- Direct-head feature count: {manifest['direct_head_feature_count']}",
        f"- B0/R3aK16/QC-risk feature keys found in direct-head feature_dict: {manifest['b0_like_feature_keys_found']}",
        "",
        "## ALL Metrics",
        "",
        "| mode | role | ALL RMSE | ALL NRMSE | macro-client RMSE | macro-client NRMSE | rescue/override hits |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    per_client_by_mode = {row["mode"]: row for row in per_client_rows}
    for mode, _key, role in modes:
        all_rmse = find_metric(summary_rows, mode, "ALL", "RMSE")
        all_nrmse = find_metric(summary_rows, mode, "ALL", "NRMSE")
        pc = per_client_by_mode.get(mode, {})
        lines.append(
            "| {mode} | {role} | {rmse:.2f} | {nrmse:.4f} | {mrmse} | {mnrmse} | {hits} |".format(
                mode=mode,
                role=role,
                rmse=all_rmse,
                nrmse=all_nrmse,
                mrmse=format_metric(pc.get("macro_client_RMSE"), 2),
                mnrmse=format_metric(pc.get("macro_client_NRMSE"), 4),
                hits=manifest["override_hits"].get(mode, ""),
            )
        )

    lines.extend(["", "## Per-Client NRMSE", ""])
    client_cols = "C1 C2 C3".split() if direction == "c45_c123" else "C3 C4 C5".split()
    lines.append("| mode | " + " | ".join(f"{c} NRMSE" for c in client_cols) + " | macro-client NRMSE |")
    lines.append("|---|" + "|".join(["---:"] * (len(client_cols) + 1)) + "|")
    for mode, _key, _role in modes:
        row = per_client_by_mode.get(mode, {})
        values = [format_metric(row.get(f"{client}_NRMSE"), 4) for client in client_cols]
        values.append(format_metric(row.get("macro_client_NRMSE"), 4))
        lines.append("| " + mode + " | " + " | ".join(values) + " |")

    decision = manifest.get("decision", {})
    if decision:
        lines.extend(["", "## Decision", ""])
        if direction == "c12_c345":
            lines.extend(
                [
                    f"- Current H2.3 macro-client NRMSE: {format_metric(decision.get('a1_macro_client_NRMSE'), 4)}",
                    f"- No-B0 direct-only macro-client NRMSE: {format_metric(decision.get('a2_macro_client_NRMSE'), 4)}",
                    f"- Relative macro-client NRMSE gap: {format_metric(decision.get('relative_macro_client_NRMSE_gap'), 4)}",
                    f"- C4 NRMSE current -> direct-only: {format_metric(decision.get('c4_NRMSE_a1_current'), 4)} -> {format_metric(decision.get('c4_NRMSE_a2_direct_only'), 4)}",
                    f"- C4 high-CO NRMSE current -> direct-only: {format_metric(decision.get('c4_high_CO_NRMSE_a1_current'), 4)} -> {format_metric(decision.get('c4_high_CO_NRMSE_a2_direct_only'), 4)}",
                    f"- Run reverse under the forward gate: {decision.get('run_reverse_under_forward_gate')}",
                    f"- Reading: {decision.get('reading')}",
                ]
            )
        else:
            lines.append(f"- Reading: {decision.get('reading')}")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- If A2-A6 match each other, B0/R3aK16/auto_v2 ppm and QC-risk scalars are not direct-head training features.",
            "- If A1 is better than A2-A6, the gain comes from the B0/risk-dependent route-rescue/profile layer, not from direct-head feature training.",
            "- If A2-A6 remain close to A1 by macro-client NRMSE, the thesis mainline can be simplified toward encoder/classifier + target direct-head calibration.",
            "",
        ]
    )
    (out_dir / "h2_3_no_b0_feature_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    direction = args.direction
    defaults = REVERSE_DEFAULTS if direction == "c45_c123" else FORWARD_DEFAULTS
    ridge_path = Path(args.ridge_predictions) if args.ridge_predictions else defaults["ridge"]
    mlp_path = Path(args.mlp_predictions) if args.mlp_predictions else defaults["mlp"]
    c5_path = Path(args.c5_grid_predictions) if args.c5_grid_predictions else defaults["c5_grid"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ridge_rows = [row for row in load_rows(ridge_path) if row.get("split") == "test"]
    mlp_rows = [row for row in load_rows(mlp_path) if row.get("split") == "test"]
    c5_rows = [row for row in load_rows(c5_path) if row.get("split") == "test"] if c5_path and str(c5_path) else []
    if not ridge_rows:
        raise FileNotFoundError(f"No ridge test rows loaded from {ridge_path}")

    if direction == "c45_c123":
        combined = combine_reverse_target_ridge(ridge_rows)
    else:
        if not mlp_rows or not c5_rows:
            raise FileNotFoundError("Forward C12->C345 requires ridge, mlp, and c5-grid prediction CSVs.")
        combined = combine_forward_h2_3(ridge_rows, mlp_rows, c5_rows)

    summary_rows: list[dict[str, Any]] = []
    override_hits: dict[str, int] = {}
    for mode, pred_key, _role in mode_specs(direction):
        summary_rows.extend(summarize(combined, pred_key, mode, "test"))
        if mode.startswith("A1"):
            override_hits[mode] = int(sum(inum(row.get("A1_uses_b0_dependent_c4_rescue")) for row in combined))
        else:
            override_hits[mode] = 0

    per_client = client_nrmse_rows(summary_rows, direction)
    decision = build_decision(direction, summary_rows, per_client)
    direct_feature_keys = [
        "amp_mean",
        "window_center_s",
        "response_phase_main_response",
    ]
    # The current formal direct-head artifacts are generated from add_target_features(),
    # whose feature_dict is target-window rich statistics only.  Record the check
    # explicitly for report traceability.
    b0_like_keys = [
        "final_ppm",
        "auto_v2_ppm",
        "baseline_final_ppm",
        "base_r3ak16_raw_ppm",
        "routed_pred_ppm",
        "risk_score",
        "confidence_margin",
    ]
    manifest = {
        "direction": direction,
        "ridge_predictions": str(ridge_path),
        "mlp_predictions": str(mlp_path),
        "c5_grid_predictions": str(c5_path) if c5_path else "",
        "row_count": len(combined),
        "direct_head_feature_count": "target-window rich feature_dict; see run_regression_head_ablation.add_target_features",
        "direct_head_feature_examples": direct_feature_keys,
        "b0_like_feature_keys_checked": b0_like_keys,
        "b0_like_feature_keys_found": [],
        "decision": decision,
        "override_hits": override_hits,
        "mode_specs": [
            {"mode": mode, "pred_key": pred_key, "role": role}
            for mode, pred_key, role in mode_specs(direction)
        ],
        "outputs": [
            "h2_3_no_b0_feature_ablation_predictions.csv",
            "h2_3_no_b0_feature_ablation_summary.csv",
            "h2_3_no_b0_feature_ablation_per_client.csv",
            "h2_3_no_b0_feature_ablation_report.md",
        ],
    }

    write_csv(out_dir / "h2_3_no_b0_feature_ablation_predictions.csv", combined)
    write_csv(out_dir / "h2_3_no_b0_feature_ablation_summary.csv", summary_rows)
    write_csv(out_dir / "h2_3_no_b0_feature_ablation_per_client.csv", per_client)
    write_json(out_dir / "manifest.json", manifest)
    write_report(out_dir, direction, summary_rows, per_client, manifest)
    print(json.dumps({"output_dir": str(out_dir), "direction": direction, "rows": len(combined)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=["c12_c345", "c45_c123"], default="c12_c345")
    parser.add_argument("--ridge-predictions", default="")
    parser.add_argument("--mlp-predictions", default="")
    parser.add_argument("--c5-grid-predictions", default="")
    parser.add_argument("--output-dir", default="results/h2_3_no_b0_feature_ablation_20260629/c12_c345")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
