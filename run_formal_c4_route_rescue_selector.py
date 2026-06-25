"""Formal calibration-selected C4 route-rescue extension for H8.

The previous C4 route-rescue sweep was a test-only upper bound. This script
selects the gate on the target calibration split only, then evaluates the fixed
gate on target test H8 predictions.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from run_c4_route_rescue_upper_bound import BASE_KEY, apply_gate, gate_mask, metrics
from run_formal_target_ridge_auto_v2_eval import attach_response_phase
from run_regression_head_ablation import client_name, fnum, inum, read_csv, summarize, write_csv


CO_CLASS = 1


def candidate_grid() -> list[dict[str, Any]]:
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
    for class_label, phase, max_final, min_risk, max_margin, rescue_ppm in product(
        pred_class_sets, phases, max_finals, min_risks, max_conf_margins, rescue_ppms
    ):
        rows.append(
            {
                "class_label": class_label,
                "pred_classes": sorted(pred_class_sets[class_label]),
                "phase": phase,
                "max_final": max_final,
                "min_risk": min_risk,
                "max_conf_margin": max_margin,
                "rescue_ppm": rescue_ppm,
            }
        )
    return rows


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["client"] = client_name(item.get("client") or item.get("client_id"))
        item["true_class"] = inum(item.get("true_class"))
        item["pred_class"] = inum(item.get("pred_class"))
        item["sample_index"] = inum(item.get("sample_index"))
        item["true_ppm"] = fnum(item.get("true_ppm"))
        item["final_ppm"] = fnum(item.get("final_ppm"))
        item["risk_score"] = fnum(item.get("risk_score"), 0.0)
        item["confidence_margin"] = fnum(item.get("confidence_margin"), 1.0)
        out.append(item)
    return out


def score_gate(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    mask = gate_mask(
        df,
        pred_classes=set(spec["pred_classes"]),
        phase=str(spec["phase"]),
        max_final=fnum(spec["max_final"]),
        min_risk=fnum(spec["min_risk"]),
        max_conf_margin=fnum(spec["max_conf_margin"]),
    )
    true_high = (
        (df["client"].astype(str) == "C4")
        & (df["true_class"].astype(int) == CO_CLASS)
        & (df["true_ppm"].astype(float) >= 200.0)
    )
    false = mask & ~true_high
    item = dict(spec)
    item["pred_classes"] = ",".join(str(v) for v in spec["pred_classes"])
    item["hit_N"] = int(mask.sum())
    item["true_c4_high_hits"] = int((mask & true_high).sum())
    item["false_hits"] = int(false.sum())
    item["calib_c4_high_N"] = int(true_high.sum())
    item["calib_c4_high_recall"] = (
        float(item["true_c4_high_hits"] / item["calib_c4_high_N"])
        if item["calib_c4_high_N"]
        else 0.0
    )
    return item


def select_gate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in scores if int(row["true_c4_high_hits"]) > 0]
    if not candidates:
        candidates = scores
    zero_false = [row for row in candidates if int(row["false_hits"]) == 0]
    if zero_false:
        candidates = zero_false
    return sorted(
        candidates,
        key=lambda row: (
            int(row["false_hits"]),
            -int(row["true_c4_high_hits"]),
            -fnum(row["rescue_ppm"]),
            fnum(row["max_final"]),
            -fnum(row["min_risk"]),
            len(str(row["pred_classes"]).split(",")),
        ),
    )[0]


def parse_pred_classes(text: Any) -> set[int]:
    return {int(part) for part in str(text).split(",") if str(part).strip()}


def apply_selected_gate(df: pd.DataFrame, gate: dict[str, Any]) -> pd.DataFrame:
    mask = gate_mask(
        df,
        pred_classes=parse_pred_classes(gate["pred_classes"]),
        phase=str(gate["phase"]),
        max_final=fnum(gate["max_final"]),
        min_risk=fnum(gate["min_risk"]),
        max_conf_margin=fnum(gate["max_conf_margin"]),
    )
    return apply_gate(df, mask, fnum(gate["rescue_ppm"]))


def audit_gate(df: pd.DataFrame, pred_key: str, label: str) -> dict[str, Any]:
    c4_high = df[
        (df["client"].astype(str) == "C4")
        & (df["true_class"].astype(int) == CO_CLASS)
        & (df["true_ppm"].astype(float) >= 200.0)
    ]
    c4_nonco = df[
        (df["client"].astype(str) == "C4")
        & (df["true_class"].astype(int) != CO_CLASS)
    ]
    all_nonco = df[df["true_class"].astype(int) != CO_CLASS]
    hit = df[df.get("c4_route_rescue_upper_hit", 0).astype(int) == 1]
    return {
        "label": label,
        "hit_N": int(len(hit)),
        "hit_true_c4_high_N": int(
            (
                (hit["client"].astype(str) == "C4")
                & (hit["true_class"].astype(int) == CO_CLASS)
                & (hit["true_ppm"].astype(float) >= 200.0)
            ).sum()
        ) if not hit.empty else 0,
        "hit_false_N": int(
            (~(
                (hit["client"].astype(str) == "C4")
                & (hit["true_class"].astype(int) == CO_CLASS)
                & (hit["true_ppm"].astype(float) >= 200.0)
            )).sum()
        ) if not hit.empty else 0,
        **{f"C4_high_{k}": v for k, v in metrics(c4_high, pred_key).items()},
        **{f"C4_nonCO_{k}": v for k, v in metrics(c4_nonco, pred_key).items()},
        **{f"nonCO_{k}": v for k, v in metrics(all_nonco, pred_key).items()},
        **{f"ALL_{k}": v for k, v in metrics(df, pred_key).items()},
    }


def write_report(out: Path, selected_gate: dict[str, Any], top_scores: list[dict[str, Any]], summary: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
    gate_json = json.dumps(selected_gate, ensure_ascii=False)
    score_lines = [
        "| rank | classes | phase | max_final | min_risk | max_margin | rescue | hits | true high | false | recall |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top_scores[:15], start=1):
        score_lines.append(
            f"| {idx} | {row['pred_classes']} | {row['phase']} | {fnum(row['max_final']):.0f} | "
            f"{fnum(row['min_risk']):.1f} | {fnum(row['max_conf_margin']):.1f} | {fnum(row['rescue_ppm']):.0f} | "
            f"{row['hit_N']} | {row['true_c4_high_hits']} | {row['false_hits']} | {fnum(row['calib_c4_high_recall']):.3f} |"
        )

    scope_rows = [
        row for row in summary
        if row.get("scope") in {"ALL", "C4-CO_high_200_250", "C4-nonCO", "nonCO_ALL"}
    ]
    summary_lines = ["| mode | scope | RMSE | Bias | P90AE | N |", "|---|---|---:|---:|---:|---:|"]
    for row in scope_rows:
        summary_lines.append(
            f"| {row['mode']} | {row['scope']} | {fnum(row.get('RMSE')):.2f} | {fnum(row.get('Bias')):.2f} | {fnum(row.get('P90AE')):.2f} | {fnum(row.get('N')):.0f} |"
        )

    audit_lines = ["| label | hit | true high | false | ALL | C4 high | C4 nonCO | nonCO |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in audit:
        audit_lines.append(
            f"| {row['label']} | {row['hit_N']} | {row['hit_true_c4_high_N']} | {row['hit_false_N']} | "
            f"{fnum(row['ALL_RMSE']):.2f} | {fnum(row['C4_high_RMSE']):.2f} | {fnum(row['C4_nonCO_RMSE']):.2f} | {fnum(row['nonCO_RMSE']):.2f} |"
        )

    text = f"""# Formal C4 Route-Rescue Selector

Selection uses target calibration only. Test is used only after selecting the fixed gate.

## Selected Gate

```json
{gate_json}
```

## Calibration Candidate Ranking

{chr(10).join(score_lines)}

## Test Summary

{chr(10).join(summary_lines)}

## Test Gate Audit

{chr(10).join(audit_lines)}

## Reading

- This is stricter than the previous upper-bound sweep because the gate is selected on calibration.
- A useful gate should reduce C4 high-CO without adding false hits on C4/nonCO or nonCO overall.
- If the selected calibration gate does not reproduce the test upper bound, the C4 rescue pattern is likely file/repeat-specific and needs stronger validation.
"""
    (out / "formal_c4_route_rescue_selector_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal calibration-selected C4 route-rescue selector.")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--h8-test-predictions", default="results/co_only_source_aug_hybrid_stratcalval_20260625/co_only_source_aug_hybrid_predictions.csv")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--output-dir", default="results/formal_c4_route_rescue_selector_20260625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_cal = [
        row for row in read_csv(Path(args.target_predictions))
        if str(row.get("split")) == "calibration"
    ]
    cal_rows = normalize_rows(attach_response_phase(raw_cal, Path(args.data_root)))
    cal_df = pd.DataFrame(cal_rows)
    scores = [score_gate(cal_df, spec) for spec in candidate_grid()]
    scores = sorted(scores, key=lambda row: (int(row["false_hits"]), -int(row["true_c4_high_hits"]), -fnum(row["rescue_ppm"])))
    selected = select_gate(scores)

    test_df = pd.read_csv(args.h8_test_predictions)
    test_df = test_df[test_df["split"].astype(str) == "test"].copy()
    test_df = apply_selected_gate(test_df, selected)
    test_df["formal_c4_route_rescue_ppm"] = test_df["candidate_ppm"].astype(float)

    summary: list[dict[str, Any]] = []
    base_rows = test_df.to_dict("records")
    summary.extend(summarize(base_rows, BASE_KEY, "H8_pred_CO_source_aug", "test"))
    summary.extend(summarize(base_rows, "formal_c4_route_rescue_ppm", "H8_plus_formal_c4_route_rescue", "test"))
    audit = [
        audit_gate(test_df.assign(c4_route_rescue_upper_hit=0), BASE_KEY, "H8_pred_CO_source_aug"),
        audit_gate(test_df, "formal_c4_route_rescue_ppm", "H8_plus_formal_c4_route_rescue"),
    ]

    write_csv(out / "formal_c4_route_rescue_calibration_scores.csv", scores)
    write_csv(out / "formal_c4_route_rescue_summary.csv", summary)
    write_csv(out / "formal_c4_route_rescue_audit.csv", audit)
    test_df.to_csv(out / "formal_c4_route_rescue_predictions.csv", index=False)
    (out / "formal_c4_route_rescue_selected_gate.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out, selected, scores, summary, audit)
    print(f"Wrote formal C4 route-rescue selector to {out}")


if __name__ == "__main__":
    main()
