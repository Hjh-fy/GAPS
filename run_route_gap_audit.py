"""Compare real-route and oracle-route profile metrics.

Positive gap means the real-route error is higher than the oracle-route error,
so the difference is attributable to classification/route noise under the same
profile family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import fnum, inum, read_csv, write_csv


PROFILE_ORDER = {
    "H2.3": 0,
    "H2.3+": 1,
    "H8+C4": 2,
    "client_selector": 3,
}


def canonical_profile(profile: str) -> str:
    text = profile.lower().replace(" ", "").replace("_", "")
    if "selector" in text:
        return "client_selector"
    if "h2.3+" in text or "h23+" in text:
        return "H2.3+"
    if "h8" in text or "formalc4" in text:
        return "H8+C4"
    if "h2.3" in text or "h23" in text:
        return "H2.3"
    return profile


def gap_pct(gap: float, real_value: float) -> float | str:
    if real_value == 0:
        return ""
    return gap / real_value


def build_gap_rows(
    real_rows: Sequence[dict[str, Any]],
    oracle_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    real_index: dict[tuple[str, str], dict[str, Any]] = {}
    oracle_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in real_rows:
        real_index[(canonical_profile(str(row.get("profile", ""))), str(row.get("scope", "")))] = row
    for row in oracle_rows:
        oracle_index[(canonical_profile(str(row.get("profile", ""))), str(row.get("scope", "")))] = row

    out: list[dict[str, Any]] = []
    for family, scope in sorted(
        set(real_index) & set(oracle_index),
        key=lambda key: (PROFILE_ORDER.get(key[0], 99), ["ALL", "C3", "C4", "C5"].index(key[1]) if key[1] in {"ALL", "C3", "C4", "C5"} else key[1]),
    ):
        real = real_index[(family, scope)]
        oracle = oracle_index[(family, scope)]
        real_full_rmse = fnum(real.get("full_RMSE"))
        oracle_full_rmse = fnum(oracle.get("full_RMSE"))
        gap_full_rmse = real_full_rmse - oracle_full_rmse
        real_full_nrmse = fnum(real.get("full_NRMSE"))
        oracle_full_nrmse = fnum(oracle.get("full_NRMSE"))
        real_cr_rmse = fnum(real.get("coverage_review_RMSE"))
        oracle_cr_rmse = fnum(oracle.get("coverage_review_RMSE"))
        real_cr_nrmse = fnum(real.get("coverage_review_NRMSE"))
        oracle_cr_nrmse = fnum(oracle.get("coverage_review_NRMSE"))
        out.append(
            {
                "profile_family": family,
                "scope": scope,
                "real_profile": real.get("profile", ""),
                "oracle_profile": oracle.get("profile", ""),
                "N": inum(real.get("N")),
                "real_full_RMSE": real_full_rmse,
                "oracle_full_RMSE": oracle_full_rmse,
                "gap_full_RMSE": gap_full_rmse,
                "gap_full_RMSE_pct_of_real": gap_pct(gap_full_rmse, real_full_rmse),
                "real_full_NRMSE": real_full_nrmse,
                "oracle_full_NRMSE": oracle_full_nrmse,
                "gap_full_NRMSE": real_full_nrmse - oracle_full_nrmse,
                "real_coverage_review_RMSE": real_cr_rmse,
                "oracle_coverage_review_RMSE": oracle_cr_rmse,
                "gap_coverage_review_RMSE": real_cr_rmse - oracle_cr_rmse,
                "real_coverage_review_NRMSE": real_cr_nrmse,
                "oracle_coverage_review_NRMSE": oracle_cr_nrmse,
                "gap_coverage_review_NRMSE": real_cr_nrmse - oracle_cr_nrmse,
            }
        )
    return out


def write_report(out_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Real-Route vs Oracle-Route Gap Audit",
        "",
        "Positive gap means real-route error is higher than oracle-route error.",
        "",
        "## Full Metric Gap",
        "",
        "| profile | scope | real RMSE/NRMSE | oracle RMSE/NRMSE | gap RMSE/NRMSE | gap RMSE % of real |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {scope} | {real_rmse} / {real_nrmse} | {oracle_rmse} / {oracle_nrmse} | {gap_rmse} / {gap_nrmse} | {pct}% |".format(
                profile=row["profile_family"],
                scope=row["scope"],
                real_rmse=format_float(row["real_full_RMSE"], 2),
                real_nrmse=format_float(row["real_full_NRMSE"], 4),
                oracle_rmse=format_float(row["oracle_full_RMSE"], 2),
                oracle_nrmse=format_float(row["oracle_full_NRMSE"], 4),
                gap_rmse=format_float(row["gap_full_RMSE"], 2),
                gap_nrmse=format_float(row["gap_full_NRMSE"], 4),
                pct=format_float(100 * fnum(row["gap_full_RMSE_pct_of_real"]), 1),
            )
        )

    lines.extend(
        [
            "",
            "## Accepted+Review Gap",
            "",
            "| profile | scope | real RMSE/NRMSE | oracle RMSE/NRMSE | gap RMSE/NRMSE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {profile} | {scope} | {real_rmse} / {real_nrmse} | {oracle_rmse} / {oracle_nrmse} | {gap_rmse} / {gap_nrmse} |".format(
                profile=row["profile_family"],
                scope=row["scope"],
                real_rmse=format_float(row["real_coverage_review_RMSE"], 2),
                real_nrmse=format_float(row["real_coverage_review_NRMSE"], 4),
                oracle_rmse=format_float(row["oracle_coverage_review_RMSE"], 2),
                oracle_nrmse=format_float(row["oracle_coverage_review_NRMSE"], 4),
                gap_rmse=format_float(row["gap_coverage_review_RMSE"], 2),
                gap_nrmse=format_float(row["gap_coverage_review_NRMSE"], 4),
            )
        )

    (out_dir / "real_vs_oracle_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_gap_rows(read_csv(args.real_metrics), read_csv(args.oracle_metrics))
    write_csv(out_dir / "real_vs_oracle_gap_metrics.csv", rows)
    write_report(out_dir, rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "real_metrics": args.real_metrics,
                "oracle_metrics": args.oracle_metrics,
                "outputs": ["real_vs_oracle_gap_metrics.csv", "real_vs_oracle_gap_report.md"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-metrics",
        default="results/f6_fixed_da_strong_r25_profile_replay_20260630/profile_qc_coverage_audit/profile_post_qc_metrics.csv",
    )
    parser.add_argument(
        "--oracle-metrics",
        default="results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/profile_qc_coverage_audit/profile_post_qc_metrics.csv",
    )
    parser.add_argument("--output-dir", default="results/real_vs_oracle_gap_audit_20260630")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
