"""Summarize QC as a post-hoc deployment reliability layer.

The report compares already-selected model profiles without using QC metrics for
model selection.  It reads runtime prediction CSVs and reports accept/review/
reject coverage, subset errors, and high-error interception.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_PROFILES = [
    {
        "profile": "H2.3",
        "role": "balanced",
        "predictions": "results/runtime_validation_h2_3_mlp_ridge_candidate_20260624/runtime_predictions.csv",
        "pred_key": "co_corrected_ppm",
    },
    {
        "profile": "H8+C4",
        "role": "co_priority",
        "predictions": "results/runtime_validation_h8_formal_c4_rescue_candidate_20260626/runtime_predictions.csv",
        "pred_key": "co_corrected_ppm",
    },
]

CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def metrics(rows: list[dict[str, str]], pred_key: str) -> dict[str, Any]:
    if not rows:
        return {"N": 0, "RMSE": "", "MAE": "", "NRMSE": "", "P90AE": "", "Bias": ""}
    pred = np.asarray([fnum(row[pred_key]) for row in rows], dtype=np.float64)
    true = np.asarray([fnum(row["true_ppm"]) for row in rows], dtype=np.float64)
    true_class = np.asarray([inum(row["true_class"]) for row in rows], dtype=np.int64)
    err = pred - true
    ranges = np.asarray([CLASS_RANGES[int(cls)] for cls in true_class], dtype=np.float64)
    return {
        "N": int(len(rows)),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAE": float(np.mean(np.abs(err))),
        "NRMSE": float(np.sqrt(np.mean((err / ranges) ** 2))),
        "P90AE": float(np.percentile(np.abs(err), 90)),
        "Bias": float(np.mean(err)),
    }


def is_high_error(row: dict[str, str], pred_key: str, abs_ppm: float, rel_ratio: float) -> bool:
    err = abs(fnum(row[pred_key]) - fnum(row["true_ppm"]))
    class_range = CLASS_RANGES.get(inum(row["true_class"]), 1.0)
    return bool(err >= abs_ppm or err / max(class_range, 1e-9) >= rel_ratio)


def scope_rows(rows: list[dict[str, str]], scope: str) -> list[dict[str, str]]:
    if scope == "ALL":
        return rows
    if scope == "CO":
        return [row for row in rows if inum(row["true_class"]) == 1]
    if scope == "nonCO":
        return [row for row in rows if inum(row["true_class"]) != 1]
    return [row for row in rows if row.get("client") == scope]


def summarize_profile(
    profile: dict[str, Any],
    rows: list[dict[str, str]],
    scopes: Iterable[str],
    abs_ppm: float,
    rel_ratio: float,
) -> list[dict[str, Any]]:
    pred_key = str(profile["pred_key"])
    out: list[dict[str, Any]] = []
    for scope in scopes:
        selected = scope_rows(rows, scope)
        total_n = len(selected)
        accept = [row for row in selected if str(row.get("qc_decision")) == "accept"]
        review = [row for row in selected if str(row.get("qc_decision")) == "review"]
        reject = [row for row in selected if str(row.get("qc_decision")) == "reject"]
        nonreject = accept + review
        high = [row for row in selected if is_high_error(row, pred_key, abs_ppm, rel_ratio)]
        high_intercepted = [row for row in high if str(row.get("qc_decision")) != "accept"]
        high_accepted = [row for row in high if str(row.get("qc_decision")) == "accept"]
        full_m = metrics(selected, pred_key)
        acc_m = metrics(accept, pred_key)
        nonrej_m = metrics(nonreject, pred_key)
        rej_m = metrics(reject, pred_key)
        out.append(
            {
                "profile": profile["profile"],
                "role": profile["role"],
                "scope": scope,
                "N": total_n,
                "full_RMSE": full_m["RMSE"],
                "full_NRMSE": full_m["NRMSE"],
                "accept_N": len(accept),
                "accept_coverage": len(accept) / total_n if total_n else "",
                "accept_RMSE": acc_m["RMSE"],
                "accept_NRMSE": acc_m["NRMSE"],
                "review_N": len(review),
                "review_coverage": len(review) / total_n if total_n else "",
                "reject_N": len(reject),
                "reject_coverage": len(reject) / total_n if total_n else "",
                "accepted_review_N": len(nonreject),
                "accepted_review_coverage": len(nonreject) / total_n if total_n else "",
                "accepted_review_RMSE": nonrej_m["RMSE"],
                "accepted_review_NRMSE": nonrej_m["NRMSE"],
                "reject_RMSE": rej_m["RMSE"],
                "reject_NRMSE": rej_m["NRMSE"],
                "high_error_N": len(high),
                "high_error_accept_N": len(high_accepted),
                "high_error_intercepted_N": len(high_intercepted),
                "high_error_recall": len(high_intercepted) / len(high) if high else "",
                "high_error_definition": f"abs_error>={abs_ppm:g}ppm OR normalized_error>={rel_ratio:g}",
            }
        )
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    all_rows = [row for row in rows if row["scope"] == "ALL"]
    columns = [
        "profile",
        "role",
        "full_RMSE",
        "accept_coverage",
        "accept_RMSE",
        "review_coverage",
        "reject_coverage",
        "accepted_review_RMSE",
        "reject_RMSE",
        "high_error_recall",
    ]
    lines = [
        "# QC Post-Hoc Reliability Report",
        "",
        "QC is evaluated here as a deployment reliability layer after model-profile selection. These metrics must not be used to choose H2.3 versus H8+C4.",
        "",
        "## ALL Scope",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in all_rows:
        lines.append("| " + " | ".join(fmt(row.get(col), 4) if col != "profile" and col != "role" else str(row.get(col, "")) for col in columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `full_RMSE` is the model-capability metric already used by the mainline selector.",
            "- Accepted/review/reject subsets describe how QC routes outputs for deployment.",
            "- `auto_output_ppm` should be interpreted as the automatic output only for accepted rows; review/reject predictions remain audit values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-json", default="", help="Optional JSON list overriding default profile prediction CSVs.")
    parser.add_argument("--output-dir", default="results/qc_posthoc_reliability_20260626")
    parser.add_argument("--scopes", default="ALL,C3,C4,C5,CO,nonCO")
    parser.add_argument("--high-error-abs-ppm", type=float, default=20.0)
    parser.add_argument("--high-error-rel-ratio", type=float, default=0.10)
    args = parser.parse_args()

    profiles = DEFAULT_PROFILES
    if args.profiles_json:
        profiles = json.loads(Path(args.profiles_json).read_text(encoding="utf-8"))
    scopes = [item.strip() for item in args.scopes.split(",") if item.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        pred_rows = read_csv(Path(profile["predictions"]))
        rows.extend(
            summarize_profile(
                profile,
                pred_rows,
                scopes=scopes,
                abs_ppm=args.high_error_abs_ppm,
                rel_ratio=args.high_error_rel_ratio,
            )
        )
    write_csv(out_dir / "qc_posthoc_summary.csv", rows)
    write_report(out_dir / "qc_posthoc_report.md", rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "profiles": profiles,
                "scopes": scopes,
                "high_error_abs_ppm": args.high_error_abs_ppm,
                "high_error_rel_ratio": args.high_error_rel_ratio,
                "outputs": {
                    "summary": str(out_dir / "qc_posthoc_summary.csv"),
                    "report": str(out_dir / "qc_posthoc_report.md"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "profiles": [p["profile"] for p in profiles]}, indent=2))


if __name__ == "__main__":
    main()
