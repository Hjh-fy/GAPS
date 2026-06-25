"""Formal calibration-only selection for the C4 high-CO rescue candidate.

This upgrades the earlier C4 rescue diagnostic into an auto_v2-style candidate:

1. Generate deployable rescue gates.
2. Select a gate using the target calibration split only.
3. Evaluate the selected gate on the target test split.
4. Combine the selected gate with A1/A2 rich residual candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_c4_high_co_rescue_gate_eval import (
    apply_candidate,
    candidate_grid,
    gate_audit,
    normalize_rows,
)
from run_combined_rich_residual_c4_rescue_eval import add_c4_metadata
from run_target_co_rich_residual_calibration import fnum, metrics, read_csv, summarize, write_csv


def split_nonco_summary(rows: list[dict[str, Any]], candidate: str, split: str) -> list[dict[str, Any]]:
    split_rows = [row for row in rows if row["split"] == split]
    c4_nonco = [row for row in split_rows if row["client"] == "C4" and row["true_class"] != 1]
    all_nonco = [row for row in split_rows if row["true_class"] != 1]
    return [
        {"mode": candidate, "split": split, "scope": "nonCO_ALL", **metrics(all_nonco, "corrected_ppm")},
        {"mode": candidate, "split": split, "scope": "C4-nonCO", **metrics(c4_nonco, "corrected_ppm")},
    ]


def build_all_summaries(rows: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["calibration", "test"]:
        out.extend(summarize(rows, "corrected_ppm", candidate, split))
        out.extend(split_nonco_summary(rows, candidate, split))
    return out


def metric(summary_rows: list[dict[str, Any]], candidate: str, split: str, scope: str, field: str = "RMSE") -> float:
    for row in summary_rows:
        if row.get("mode") == candidate and row.get("split") == split and row.get("scope") == scope:
            return fnum(row.get(field), float("inf"))
    return float("inf")


def selection_table(
    rows: list[dict[str, Any]],
    min_calib_gated: int,
    min_precision: float,
    max_nonco_rmse_delta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_rows = apply_candidate(rows, candidate_grid()[0])
    baseline_summary = build_all_summaries(baseline_rows, "A0_baseline")
    base_all = metric(baseline_summary, "A0_baseline", "calibration", "ALL")
    base_high = metric(baseline_summary, "A0_baseline", "calibration", "C4-CO_high_200_250")
    base_nonco = metric(baseline_summary, "A0_baseline", "calibration", "C4-nonCO")

    table: list[dict[str, Any]] = []
    best_spec: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    for spec in candidate_grid()[1:]:
        candidate_rows = apply_candidate(rows, spec)
        summary = build_all_summaries(candidate_rows, spec["candidate"])
        audit_rows = gate_audit(candidate_rows, spec)
        calib_audit = next(row for row in audit_rows if row["split"] == "calibration")
        all_rmse = metric(summary, spec["candidate"], "calibration", "ALL")
        high_rmse = metric(summary, spec["candidate"], "calibration", "C4-CO_high_200_250")
        nonco_rmse = metric(summary, spec["candidate"], "calibration", "C4-nonCO")
        gated_n = int(calib_audit["gated_N"])
        high_precision = fnum(calib_audit.get("high_precision"), 0.0)
        nonco_n = int(calib_audit["gated_nonCO_N"])
        passes = (
            gated_n >= min_calib_gated
            and high_precision >= min_precision
            and nonco_n == 0
            and all_rmse < base_all
            and high_rmse < base_high
            and nonco_rmse <= base_nonco + max_nonco_rmse_delta
        )
        item = {
            "candidate": spec["candidate"],
            "phase": spec["phase"],
            "risk_threshold": spec["risk_threshold"],
            "max_ppm": spec["max_ppm"],
            "pred_classes": spec["pred_classes"],
            "rescue_ppm": spec["rescue_ppm"],
            "calib_ALL_RMSE": all_rmse,
            "calib_C4_CO_high_RMSE": high_rmse,
            "calib_C4_nonCO_RMSE": nonco_rmse,
            "delta_calib_ALL_RMSE": all_rmse - base_all,
            "delta_calib_C4_CO_high_RMSE": high_rmse - base_high,
            "delta_calib_C4_nonCO_RMSE": nonco_rmse - base_nonco,
            "calib_gated_N": gated_n,
            "calib_gated_true_CO_high_N": calib_audit["gated_true_CO_high_N"],
            "calib_gated_nonCO_N": nonco_n,
            "calib_high_precision": high_precision,
            "passes_selection_constraints": int(passes),
        }
        table.append(item)
        if passes:
            score = (all_rmse, high_rmse, -gated_n)
            if best_score is None or score < best_score:
                best_score = score
                best_spec = spec
    if best_spec is None:
        # Fall back to the best C4-high candidate, but keep this explicit.
        best_row = min(table, key=lambda row: (row["calib_C4_CO_high_RMSE"], row["calib_ALL_RMSE"]))
        best_name = best_row["candidate"]
        best_spec = next(spec for spec in candidate_grid()[1:] if spec["candidate"] == best_name)
    table.sort(key=lambda row: (1 - int(row["passes_selection_constraints"]), row["calib_ALL_RMSE"], row["calib_C4_CO_high_RMSE"]))
    return table, best_spec


def apply_gate_preserving_residual(
    rows: list[dict[str, Any]],
    source_candidate: str,
    output_candidate: str,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    source_rows = [dict(row) for row in rows if str(row.get("candidate")) == source_candidate]
    gate_rows = apply_candidate(source_rows, spec)
    out: list[dict[str, Any]] = []
    for original, gated in zip(source_rows, gate_rows):
        item = dict(original)
        item["candidate"] = output_candidate
        item["selected_c4_rescue_gate"] = spec["candidate"]
        item["c4_rescue_applied"] = int(gated.get("rescue_applied", 0))
        if item["c4_rescue_applied"]:
            item["corrected_ppm"] = fnum(gated.get("corrected_ppm"))
        else:
            item["corrected_ppm"] = fnum(original.get("corrected_ppm", original.get("final_ppm")))
        out.append(item)
    return out


def output_audit(rows: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["test"]:
        split_rows = [row for row in rows if row["split"] == split]
        gated = [
            row for row in split_rows
            if int(row.get("c4_rescue_applied", row.get("rescue_applied", 0))) == 1
        ]
        true_high = [row for row in gated if row["true_class"] == 1 and fnum(row["true_ppm"]) >= 200.0]
        nonco = [row for row in gated if row["true_class"] != 1]
        out.append(
            {
                "candidate": candidate,
                "split": split,
                "gated_N": len(gated),
                "gated_true_CO_high_N": len(true_high),
                "gated_nonCO_N": len(nonco),
                "high_precision": len(true_high) / len(gated) if gated else None,
            }
        )
    return out


def report_table(summary: list[dict[str, Any]], candidates: list[str], scopes: list[str]) -> str:
    lines = ["| candidate | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for candidate in candidates:
        vals = []
        for scope in scopes:
            vals.append(f"{metric(summary, candidate, 'test', scope):.2f}")
        lines.append("| " + candidate + " | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    out: Path,
    selected_spec: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> None:
    candidates = [
        "A0_baseline_auto_v2_final",
        "A1_auto_v2_rich_residual_val_select",
        "A2_forced_ridge_phase",
        "C4_formal_rescue_only",
        "C1_A1_plus_formal_c4_rescue",
        "C2_A2_plus_formal_c4_rescue",
    ]
    scopes = ["ALL", "C3-CO", "C4-CO", "C5-CO", "C4-CO_high_200_250", "C5-CO_high_200_250"]
    top = selection_rows[:10]
    select_lines = [
        "| candidate | pass | calib ALL | calib C4 high | calib C4 nonCO | gated | high precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        select_lines.append(
            "| {candidate} | {passes} | {all:.2f} | {high:.2f} | {nonco:.2f} | {gated} | {prec:.2f} |".format(
                candidate=row["candidate"],
                passes=row["passes_selection_constraints"],
                all=row["calib_ALL_RMSE"],
                high=row["calib_C4_CO_high_RMSE"],
                nonco=row["calib_C4_nonCO_RMSE"],
                gated=row["calib_gated_N"],
                prec=row["calib_high_precision"],
            )
        )
    audit_lines = ["| candidate | gated N | true high N | nonCO N | high precision |", "|---|---:|---:|---:|---:|"]
    for row in audits:
        if row.get("split") == "test":
            audit_lines.append(
                "| {candidate} | {gated_N} | {gated_true_CO_high_N} | {gated_nonCO_N} | {prec} |".format(
                    candidate=row["candidate"],
                    gated_N=row["gated_N"],
                    gated_true_CO_high_N=row["gated_true_CO_high_N"],
                    gated_nonCO_N=row["gated_nonCO_N"],
                    prec="" if row.get("high_precision") in (None, "") else f"{fnum(row.get('high_precision')):.2f}",
                )
            )
    text = f"""# Formal C4 Rescue auto_v2 Evaluation

Selection uses calibration split only. Test split is used only for final reporting.

Selected gate:

- `{selected_spec['candidate']}`

## Calibration Selection Top Candidates

{chr(10).join(select_lines)}

## Test RMSE

{report_table(summary, candidates, scopes)}

## Test Gate Audit

{chr(10).join(audit_lines)}

## Reading

- This is the formal version of the C4 route-rescue diagnostic.
- If the selected gate keeps non-CO hits low on test while improving C4 high CO, it can be promoted into the next auto_v2 candidate flow.
- The combined candidates still need reverse-direction replication before being treated as final.
"""
    (out / "formal_c4_rescue_auto_v2_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal calibration-only C4 rescue selection and combination.")
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-calib-gated", type=int, default=1)
    parser.add_argument("--min-precision", type=float, default=1.0)
    parser.add_argument("--max-nonco-rmse-delta", type=float, default=0.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline_rows = normalize_rows(read_csv(args.baseline_predictions), Path(args.data_root))
    selection_rows, selected_spec = selection_table(
        baseline_rows,
        min_calib_gated=args.min_calib_gated,
        min_precision=args.min_precision,
        max_nonco_rmse_delta=args.max_nonco_rmse_delta,
    )

    candidate_rows = add_c4_metadata(read_csv(args.candidate_predictions), Path(args.data_root))
    output_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    base_candidates = [
        "A0_baseline_auto_v2_final",
        "A1_auto_v2_rich_residual_val_select",
        "A2_forced_ridge_phase",
    ]
    for candidate in base_candidates:
        rows = [dict(row) for row in candidate_rows if str(row.get("candidate")) == candidate]
        for row in rows:
            row["c4_rescue_applied"] = 0
            row["selected_c4_rescue_gate"] = ""
        output_rows.extend(rows)
        summary.extend(summarize(rows, "corrected_ppm", candidate, "test"))
        audits.extend(output_audit(rows, candidate))

    rescue_only = apply_gate_preserving_residual(
        candidate_rows,
        "A0_baseline_auto_v2_final",
        "C4_formal_rescue_only",
        selected_spec,
    )
    output_rows.extend(rescue_only)
    summary.extend(summarize(rescue_only, "corrected_ppm", "C4_formal_rescue_only", "test"))
    audits.extend(output_audit(rescue_only, "C4_formal_rescue_only"))

    for source, output_name in [
        ("A1_auto_v2_rich_residual_val_select", "C1_A1_plus_formal_c4_rescue"),
        ("A2_forced_ridge_phase", "C2_A2_plus_formal_c4_rescue"),
    ]:
        rows = apply_gate_preserving_residual(candidate_rows, source, output_name, selected_spec)
        output_rows.extend(rows)
        summary.extend(summarize(rows, "corrected_ppm", output_name, "test"))
        audits.extend(output_audit(rows, output_name))

    write_csv(out / "formal_c4_rescue_selection_table.csv", selection_rows)
    write_csv(out / "formal_c4_rescue_auto_v2_predictions.csv", output_rows)
    write_csv(out / "formal_c4_rescue_auto_v2_summary.csv", summary)
    write_csv(out / "formal_c4_rescue_auto_v2_audit.csv", audits)
    write_report(out, selected_spec, selection_rows, summary, audits)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "baseline_predictions": args.baseline_predictions,
                "candidate_predictions": args.candidate_predictions,
                "data_root": args.data_root,
                "selected_gate": {k: v for k, v in selected_spec.items() if k != "pred_class_set"},
                "selection_constraints": {
                    "min_calib_gated": args.min_calib_gated,
                    "min_precision": args.min_precision,
                    "max_nonco_rmse_delta": args.max_nonco_rmse_delta,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote formal C4 rescue auto_v2 evaluation to {out}")


if __name__ == "__main__":
    main()
