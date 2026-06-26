"""Audit H8 + formal C4 route-rescue guardrails.

The audit is intentionally based on saved analysis/runtime outputs.  It checks
whether the formal C4 rescue only hits the intended high-CO failure pattern and
whether non-CO metrics remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ANALYSIS = Path("results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_predictions.csv")
DEFAULT_SELECTED_GATE = Path("results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_selected_gate.json")
DEFAULT_EQUIV = Path("results/equivalence_h8_formal_c4_rescue_candidate_20260626/equivalence_summary.json")
DEFAULT_OUT_DIR = Path("results/h8_c4_guardrail_audit_20260626")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def inum(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rmse(rows: list[dict[str, str]], pred_col: str) -> float | None:
    errors: list[float] = []
    for row in rows:
        pred = fnum(row.get(pred_col))
        true = fnum(row.get("true_ppm"))
        if math.isfinite(pred) and math.isfinite(true):
            errors.append((pred - true) ** 2)
    if not errors:
        return None
    return math.sqrt(sum(errors) / len(errors))


def row_scope(row: dict[str, str]) -> dict[str, bool]:
    client = str(row.get("client") or row.get("client_id"))
    true_class = inum(row.get("true_class"))
    true_ppm = fnum(row.get("true_ppm"))
    return {
        "c4": client == "C4",
        "co": true_class == 1,
        "nonco": true_class != 1,
        "c4_high_co": client == "C4" and true_class == 1 and true_ppm >= 200.0,
        "c4_nonco": client == "C4" and true_class != 1,
        "nonco_all": true_class != 1,
    }


def metric_block(rows: list[dict[str, str]], before_col: str, after_col: str) -> list[dict[str, Any]]:
    scopes = {
        "C4_high_CO": lambda r: row_scope(r)["c4_high_co"],
        "C4_nonCO": lambda r: row_scope(r)["c4_nonco"],
        "nonCO_ALL": lambda r: row_scope(r)["nonco_all"],
        "ALL": lambda r: True,
    }
    out: list[dict[str, Any]] = []
    for name, predicate in scopes.items():
        selected = [row for row in rows if predicate(row)]
        before = rmse(selected, before_col)
        after = rmse(selected, after_col)
        out.append(
            {
                "scope": name,
                "N": len(selected),
                "before_col": before_col,
                "after_col": after_col,
                "RMSE_before": before,
                "RMSE_after": after,
                "delta_RMSE": None if before is None or after is None else after - before,
            }
        )
    return out


def summarize_hits(hits: list[dict[str, str]], all_rows: list[dict[str, str]]) -> dict[str, Any]:
    c4_high_total = sum(row_scope(row)["c4_high_co"] for row in all_rows)
    hit_true_c4_high = sum(row_scope(row)["c4_high_co"] for row in hits)
    hit_nonco = sum(row_scope(row)["nonco"] for row in hits)
    hit_false = len(hits) - hit_true_c4_high
    by_phase = Counter(str(row.get("response_phase", "")) or "unknown" for row in hits)
    by_pred_class = Counter(str(row.get("pred_class", "")) for row in hits)
    by_file_repeat_phase = Counter(
        (
            str(row.get("filename", "")),
            str(row.get("repeat_id", "")),
            str(row.get("response_phase", "")),
        )
        for row in hits
    )
    return {
        "hit_N": len(hits),
        "hit_true_C4_high_CO_N": hit_true_c4_high,
        "hit_false_N": hit_false,
        "hit_nonCO_N": hit_nonco,
        "C4_high_CO_total_N": c4_high_total,
        "C4_high_CO_recall": hit_true_c4_high / c4_high_total if c4_high_total else None,
        "hit_by_response_phase": dict(sorted(by_phase.items())),
        "hit_by_pred_class": dict(sorted(by_pred_class.items())),
        "hit_by_file_repeat_phase": [
            {
                "filename": key[0],
                "repeat_id": key[1],
                "response_phase": key[2],
                "N": count,
            }
            for key, count in by_file_repeat_phase.most_common()
        ],
    }


def write_report(path: Path, summary: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# H8 + Formal C4 Guardrail Audit",
        "",
        f"- analysis_predictions: `{summary['analysis_predictions']}`",
        f"- selected_gate: `{summary['selected_gate_path']}`",
        f"- equivalence_summary: `{summary.get('equivalence_summary_path', '')}`",
        "",
        "## Gate Hits",
        "",
        f"- hit_N: {summary['hit_N']}",
        f"- hit_true_C4_high_CO_N: {summary['hit_true_C4_high_CO_N']}",
        f"- hit_false_N: {summary['hit_false_N']}",
        f"- hit_nonCO_N: {summary['hit_nonCO_N']}",
        f"- C4_high_CO_recall: {summary['C4_high_CO_recall']}",
        f"- guardrail_status: {summary['guardrail_status']}",
        "",
        "## RMSE Before/After",
        "",
        "| scope | N | RMSE before | RMSE after | delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['scope']} | {row['N']} | {row['RMSE_before']:.6g} | {row['RMSE_after']:.6g} | {row['delta_RMSE']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pass` requires zero false hits, zero nonCO hits, and runtime equivalence with zero mismatches when an equivalence summary is provided.",
            "- H2.3 should remain the balanced default; this audit only supports H8+C4 as a CO/high-CO specialist candidate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--selected-gate", type=Path, default=DEFAULT_SELECTED_GATE)
    parser.add_argument("--equivalence-summary", type=Path, default=DEFAULT_EQUIV)
    parser.add_argument("--before-column", default="h8_pred_co_source_aug_else_h23_ppm")
    parser.add_argument("--after-column", default="formal_c4_route_rescue_ppm")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    rows = [row for row in read_csv(args.analysis) if str(row.get("split", "test")) == "test"]
    hits = [
        row
        for row in rows
        if abs(fnum(row.get(args.after_column)) - fnum(row.get(args.before_column))) > args.tolerance
    ]
    selected_gate = json.loads(args.selected_gate.read_text(encoding="utf-8")) if args.selected_gate.exists() else {}
    metrics = metric_block(rows, args.before_column, args.after_column)
    summary = summarize_hits(hits, rows)
    summary.update(
        {
            "analysis_predictions": str(args.analysis),
            "selected_gate_path": str(args.selected_gate),
            "selected_gate": selected_gate,
            "before_column": args.before_column,
            "after_column": args.after_column,
            "rows": len(rows),
        }
    )
    equivalence = None
    if args.equivalence_summary.exists():
        equivalence = json.loads(args.equivalence_summary.read_text(encoding="utf-8"))
        summary["equivalence_summary_path"] = str(args.equivalence_summary)
        summary["equivalence"] = equivalence

    equivalence_ok = equivalence is None or int(equivalence.get("num_mismatch", -1)) == 0
    false_ok = int(summary["hit_false_N"]) == 0
    nonco_ok = int(summary["hit_nonCO_N"]) == 0
    summary["guardrail_status"] = "pass" if false_ok and nonco_ok and equivalence_ok else "fail"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "h8_c4_route_rescue_hit_rows.csv", hits)
    write_csv(args.output_dir / "h8_c4_guardrail_metrics.csv", metrics)
    (args.output_dir / "h8_c4_guardrail_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "h8_c4_guardrail_audit.md", summary, metrics)
    print(json.dumps({"output_dir": str(args.output_dir), "guardrail_status": summary["guardrail_status"]}, indent=2))


if __name__ == "__main__":
    main()
