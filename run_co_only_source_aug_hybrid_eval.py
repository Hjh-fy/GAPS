"""CO-only hybrid between H2.3 and source-augmented target Ridge.

This experiment keeps the current H2.3 prediction for non-CO routes and uses
the source-prediction-augmented target Ridge only when a deployment-visible
CO-side condition is met.

The goal is to test whether the source-augmented target Ridge is best treated
as a CO specialist instead of a full replacement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from run_regression_head_ablation import fnum, inum, read_csv, summarize, write_csv


DEFAULT_H23_KEY = "client_hybrid_mlp_c3_ridge_c4_c5grid_ppm"
DEFAULT_SOURCE_AUG_KEY = "target_ridge_plus_source_preds_plus_c4_rescue_ppm"


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("client")), str(row.get("split")), inum(row.get("sample_index")))


def metric_value(summary: list[dict[str, Any]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope:
            value = row.get(metric)
            return None if value in (None, "") else float(value)
    return None


def make_hybrid_rows(
    h23_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    h23_key: str,
    source_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_key = {row_key(row): row for row in source_rows}
    if len(source_by_key) != len(source_rows):
        raise ValueError("Source-augmented prediction rows do not have unique (client, split, sample_index) keys.")

    out: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    mismatch_count = 0
    for row in h23_rows:
        key = row_key(row)
        src = source_by_key.get(key)
        if src is None:
            raise KeyError(f"Missing source-augmented row for key={key}")

        for field in ["true_class", "true_ppm", "pred_class"]:
            if str(row.get(field)) != str(src.get(field)):
                mismatch_count += 1
                break

        item = dict(row)
        h23_ppm = fnum(row.get(h23_key))
        source_ppm = fnum(src.get(source_key))
        pred_class = inum(row.get("pred_class"))
        c4_gate = inum(src.get("c4_rescue_applied"), 0) == 1

        item["h2_3_ppm"] = h23_ppm
        item["source_aug_target_ridge_ppm"] = source_ppm
        item["source_aug_c4_rescue_applied"] = int(c4_gate)
        item["h8_pred_co_source_aug_else_h23_ppm"] = source_ppm if pred_class == 1 else h23_ppm
        item["h8_pred_co_or_c4_gate_source_aug_else_h23_ppm"] = source_ppm if pred_class == 1 or c4_gate else h23_ppm
        item["h8_pred_co_switch"] = int(pred_class == 1)
        item["h8_pred_co_or_c4_gate_switch"] = int(pred_class == 1 or c4_gate)
        out.append(item)

    for switch_key in ["h8_pred_co_switch", "h8_pred_co_or_c4_gate_switch"]:
        by_client = Counter()
        by_pred_gas = Counter()
        true_co = 0
        nonco = 0
        total = 0
        for row in out:
            if inum(row.get(switch_key)) != 1:
                continue
            total += 1
            by_client[str(row.get("client"))] += 1
            by_pred_gas[str(row.get("pred_gas"))] += 1
            if inum(row.get("true_class")) == 1:
                true_co += 1
            else:
                nonco += 1
        audits.append(
            {
                "switch_rule": switch_key,
                "switched_N": total,
                "true_CO_N": true_co,
                "nonCO_N": nonco,
                "by_client": json.dumps(dict(sorted(by_client.items())), ensure_ascii=False),
                "by_pred_gas": json.dumps(dict(sorted(by_pred_gas.items())), ensure_ascii=False),
                "key_field_mismatches": mismatch_count,
            }
        )
    return out, audits


def write_report(out_dir: Path, summary: list[dict[str, Any]], audits: list[dict[str, Any]]) -> None:
    scopes = [
        "ALL",
        "C3-CO",
        "C4-CO",
        "C5-CO",
        "C3-CO_high_200_250",
        "C4-CO_high_200_250",
        "C5-CO_high_200_250",
        "nonCO_ALL",
    ]
    modes = [
        "A0_baseline_final",
        "H2_3_current",
        "source_aug_target_ridge_plus_c4_rescue",
        "H8_pred_co_source_aug_else_h23",
        "H8_pred_co_or_c4_gate_source_aug_else_h23",
    ]
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        values = []
        for scope in scopes:
            value = metric_value(summary, mode, scope)
            values.append("" if value is None else f"{value:.2f}")
        lines.append("| " + mode + " | " + " | ".join(values) + " |")

    nrmse_lines = ["| mode | ALL NRMSE | nonCO NRMSE |", "|---|---:|---:|"]
    for mode in modes:
        all_nrmse = metric_value(summary, mode, "ALL", "NRMSE")
        nonco_nrmse = metric_value(summary, mode, "nonCO_ALL", "NRMSE")
        nrmse_lines.append(
            f"| {mode} | {'' if all_nrmse is None else f'{all_nrmse:.4f}'} | "
            f"{'' if nonco_nrmse is None else f'{nonco_nrmse:.4f}'} |"
        )

    audit_lines = ["| switch rule | switched N | true CO N | nonCO N | by client | by pred gas | mismatches |", "|---|---:|---:|---:|---|---|---:|"]
    for row in audits:
        audit_lines.append(
            f"| {row['switch_rule']} | {row['switched_N']} | {row['true_CO_N']} | {row['nonCO_N']} | "
            f"`{row['by_client']}` | `{row['by_pred_gas']}` | {row['key_field_mismatches']} |"
        )

    text = f"""# CO-Only Source-Augmented Hybrid Evaluation

Question:

- Can source-augmented target Ridge be used as a CO specialist while H2.3 keeps the non-CO behavior?

Protocol:

- Base prediction: current H2.3 `client_hybrid_mlp_c3_ridge_c4_c5grid_ppm`.
- CO specialist: `target_ridge_plus_source_preds_plus_c4_rescue_ppm`.
- Target test only, no QC.
- Switching rules use deployment-visible fields only:
  - `H8_pred_co_source_aug_else_h23`: switch when `pred_class == CO`.
  - `H8_pred_co_or_c4_gate_source_aug_else_h23`: switch when `pred_class == CO` or the existing C4 rescue gate fires.

## Target Test RMSE

{chr(10).join(lines)}

## Target Test NRMSE

{chr(10).join(nrmse_lines)}

## Switch Audit

{chr(10).join(audit_lines)}

## Reading

- This is not a source-only transfer test.
- It tests whether source-lightweight predictions are useful after target calibration, but only where the CO-specific signal helps.
- If this beats H2.3 on ALL RMSE/NRMSE without hurting nonCO, it should enter the formal target direct-head selector as a candidate.
"""
    (out_dir / "co_only_source_aug_hybrid_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CO-only source-augmented hybrid on top of H2.3.")
    parser.add_argument("--h23-predictions", default="results/hybrid_regression_head_selection_20260624/hybrid_head_predictions.csv")
    parser.add_argument("--source-aug-predictions", default="results/source_augmented_target_ridge_20260625_lite/target_predictions_plus_source_preds_plus_c4_rescue.csv")
    parser.add_argument("--h23-key", default=DEFAULT_H23_KEY)
    parser.add_argument("--source-key", default=DEFAULT_SOURCE_AUG_KEY)
    parser.add_argument("--output-dir", default="results/co_only_source_aug_hybrid_20260625")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h23_rows = read_csv(args.h23_predictions)
    source_rows = read_csv(args.source_aug_predictions)
    rows, audits = make_hybrid_rows(h23_rows, source_rows, args.h23_key, args.source_key)

    summary: list[dict[str, Any]] = []
    summary.extend(summarize(rows, "baseline_final_ppm", "A0_baseline_final", "test"))
    summary.extend(summarize(rows, "h2_3_ppm", "H2_3_current", "test"))
    summary.extend(summarize(rows, "source_aug_target_ridge_ppm", "source_aug_target_ridge_plus_c4_rescue", "test"))
    summary.extend(summarize(rows, "h8_pred_co_source_aug_else_h23_ppm", "H8_pred_co_source_aug_else_h23", "test"))
    summary.extend(
        summarize(
            rows,
            "h8_pred_co_or_c4_gate_source_aug_else_h23_ppm",
            "H8_pred_co_or_c4_gate_source_aug_else_h23",
            "test",
        )
    )

    write_csv(out_dir / "co_only_source_aug_hybrid_predictions.csv", rows)
    write_csv(out_dir / "co_only_source_aug_hybrid_summary.csv", summary)
    write_csv(out_dir / "switch_audit.csv", audits)
    write_report(out_dir, summary, audits)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "h23_predictions": args.h23_predictions,
                "source_aug_predictions": args.source_aug_predictions,
                "h23_key": args.h23_key,
                "source_key": args.source_key,
                "output_files": [
                    "co_only_source_aug_hybrid_predictions.csv",
                    "co_only_source_aug_hybrid_summary.csv",
                    "switch_audit.csv",
                    "co_only_source_aug_hybrid_report.md",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote CO-only source-augmented hybrid evaluation to {out_dir}")


if __name__ == "__main__":
    main()
