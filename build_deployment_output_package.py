"""Build deployment-facing prediction outputs from final QC records.

The package turns experiment CSVs into a system-style output table with clear
decisions: accept automatic ppm, or mark the window for warning/manual review.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def inum(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def client_label(row: Dict) -> str:
    if row.get("client"):
        return str(row["client"])
    return f"C{inum(row.get('client_id'))}"


def risk_for_row(row: Dict, accepted_key: str) -> Dict[str, float | str]:
    # Prefer the selected policy ratios when they exist. Fall back to visible risk scores.
    ratio_keys = [
        "post_veto_max_ratio",
        "multi_max_ratio",
        "default_t8_max_ratio",
    ]
    for key in ratio_keys:
        value = fnum(row.get(key))
        if np.isfinite(value):
            return {"risk_score": value, "risk_score_name": key}
    candidates = [
        "response_mean_conc_gap_norm",
        "composite_response_risk",
        "class_response_margin_risk",
        "route_response_risk",
    ]
    values = [(key, fnum(row.get(key))) for key in candidates]
    finite = [(key, value) for key, value in values if np.isfinite(value)]
    if not finite:
        return {"risk_score": np.nan, "risk_score_name": ""}
    key, value = max(finite, key=lambda item: item[1])
    return {"risk_score": value, "risk_score_name": key}


def decision_for(row: Dict, accepted_key: str) -> str:
    return "accept" if inum(row.get(accepted_key), 1) == 1 else "manual_review"


def build_rows(records: Sequence[Dict], pred_key: str, accepted_key: str, workpoint: str) -> List[Dict]:
    out = []
    for row in records:
        risk = risk_for_row(row, accepted_key)
        pred = fnum(row.get(pred_key, row.get("pred_ppm")))
        true = fnum(row.get("true_ppm"))
        decision = decision_for(row, accepted_key)
        out.append({
            "client": client_label(row),
            "sample_index": inum(row.get("sample_index")),
            "filename": row.get("filename", ""),
            "phase": inum(row.get("phase"), -1),
            "predicted_gas": row.get("pred_gas", ""),
            "predicted_class": inum(row.get("pred_class"), -1),
            "ppm_full_prediction": pred,
            "ppm_auto_output": pred if decision == "accept" else "",
            "decision": decision,
            "workpoint": workpoint,
            "risk_score": risk["risk_score"],
            "risk_score_name": risk["risk_score_name"],
            "selected_policy": row.get("post_veto_policy") or row.get("multi_policy") or row.get("default_t8_policy", ""),
            "selected_source_model": row.get("source_route_selected_model", row.get("selected_source_model", "")),
            "true_gas_for_evaluation": row.get("true_gas", ""),
            "true_ppm_for_evaluation": true,
            "abs_error_for_evaluation": abs(pred - true) if np.isfinite(pred) and np.isfinite(true) else "",
        })
    return out


def summarize(rows: Sequence[Dict]) -> List[Dict]:
    groups = {"ALL": list(rows)}
    for client in sorted({r["client"] for r in rows}):
        groups[client] = [r for r in rows if r["client"] == client]
    out = []
    for group, sub in groups.items():
        accepted = [r for r in sub if r["decision"] == "accept"]
        reviewed = [r for r in sub if r["decision"] != "accept"]
        errors = np.asarray([fnum(r.get("abs_error_for_evaluation")) for r in accepted], dtype=np.float64)
        errors = errors[np.isfinite(errors)]
        out.append({
            "group": group,
            "total_n": len(sub),
            "accepted_n": len(accepted),
            "manual_review_n": len(reviewed),
            "accepted_coverage": len(accepted) / max(1, len(sub)),
            "accepted_MAE_eval": float(np.mean(errors)) if errors.size else "",
            "accepted_P90AE_eval": float(np.percentile(errors, 90)) if errors.size else "",
        })
    return out


def write_readme(path: Path, output_csv: Path, summary_csv: Path, workpoint: str) -> None:
    lines = [
        "# Deployment Output Package",
        "",
        f"- workpoint: `{workpoint}`",
        f"- output table: `{output_csv}`",
        f"- summary table: `{summary_csv}`",
        "",
        "## Columns",
        "- `ppm_full_prediction`: model prediction for every window.",
        "- `ppm_auto_output`: filled only when `decision=accept`; blank for manual-review windows.",
        "- `decision`: `accept` or `manual_review`.",
        "- `risk_score` / `risk_score_name`: deployment-visible risk evidence used for QC/debugging.",
        "- `true_*_for_evaluation`: retained for offline evaluation; not required in deployment.",
        "",
        "## Thesis Wording",
        "Rejected windows are not failed predictions; they are windows where the system refuses silent automatic concentration output and requests warning/retest/manual review.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deployment-facing output CSV.")
    parser.add_argument("--records_csv", default="results/T8_post_veto_qc/src45_tgt123_source_route_guard5_tail_with_disagreement/post_veto_test_records.csv")
    parser.add_argument("--output_dir", default="results/deployment_output_package/source_route_post_veto")
    parser.add_argument("--pred_key", default="route_pred_ppm")
    parser.add_argument("--accepted_key", default="post_veto_accepted")
    parser.add_argument("--workpoint", default="source_route_post_veto_cov0.848")
    args = parser.parse_args()

    records = read_csv(Path(args.records_csv))
    rows = build_rows(records, args.pred_key, args.accepted_key, args.workpoint)
    summary_rows = summarize(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "deployment_window_outputs.csv"
    summary_csv = out_dir / "deployment_output_summary.csv"
    write_csv(output_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_readme(out_dir / "README.md", output_csv, summary_csv, args.workpoint)
    print((out_dir / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
