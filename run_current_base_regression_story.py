"""Summarize the frozen current-base regression mainline story.

This script reads existing experiment CSVs only. It does not train or replay
models. Outputs are compact tables and a Chinese story report for the current
F6/H2.3+/H8 guarded mainline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import fnum, inum, read_csv, write_csv


MAINLINE_PROFILE_SPECS = [
    ("H2.3 oracle-route", ["H2_3_oracle_route", "H2_3_current_r25_replay"], "h23"),
    ("H2.3+ oracle-route weak-blend", ["oracle-route_H2.3+"], "guarded"),
    ("H8+C4 oracle-route", ["oracle-route_H8+C4"], "guarded"),
    ("Guarded practical oracle-route", ["oracle-route_guarded_profile"], "guarded"),
]

POST_QC_PROFILES = [
    "H2.3+ oracle-route weak-blend",
    "H8+C4 oracle-route",
    "Guarded practical oracle-route",
    "Client prior C34 H2.3+ / C5 H8+C4 oracle-route",
]

DEFAULT_SCOPES = ["ALL", "C3", "C4", "C5"]
DEFAULT_GAP_FAMILIES = ["H2.3", "H2.3+", "H8+C4", "client_selector"]


def require_row(
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
    values: Sequence[str],
    scope: str,
    context: str,
) -> dict[str, Any]:
    allowed = {str(value) for value in values}
    for row in rows:
        if str(row.get(key)) in allowed and str(row.get("scope")) == scope:
            return row
    raise ValueError(f"Missing metric row for {context}: {key} in {sorted(allowed)}, scope={scope}")


def numeric_metric_row(profile: str, scope: str, row: dict[str, Any], *, rmse_key: str, nrmse_key: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "scope": scope,
        "N": int(inum(row.get("N"))),
        "RMSE": fnum(row.get(rmse_key)),
        "NRMSE": fnum(row.get(nrmse_key)),
    }


def build_mainline_summary(
    guarded_metric_rows: Sequence[dict[str, Any]],
    h23_metric_rows: Sequence[dict[str, Any]],
    *,
    scopes: Sequence[str] = DEFAULT_SCOPES,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile, modes, source in MAINLINE_PROFILE_SPECS:
        source_rows = h23_metric_rows if source == "h23" else guarded_metric_rows
        for scope in scopes:
            row = require_row(source_rows, key="mode", values=modes, scope=scope, context="oracle full mainline")
            out.append(numeric_metric_row(profile, scope, row, rmse_key="RMSE", nrmse_key="NRMSE"))
    return out


def build_post_qc_summary(
    post_qc_rows: Sequence[dict[str, Any]],
    *,
    scopes: Sequence[str] = DEFAULT_SCOPES,
    profiles: Sequence[str] = POST_QC_PROFILES,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile in profiles:
        for scope in scopes:
            row = require_row(post_qc_rows, key="profile", values=[profile], scope=scope, context="post QC")
            out.append(
                {
                    "profile": profile,
                    "scope": scope,
                    "N": int(inum(row.get("N"))),
                    "coverage_review": fnum(row.get("coverage_review")),
                    "nonreject_N": int(inum(row.get("nonreject_N"))),
                    "coverage_review_RMSE": fnum(row.get("coverage_review_RMSE")),
                    "coverage_review_NRMSE": fnum(row.get("coverage_review_NRMSE")),
                }
            )
    return out


def build_route_gap_summary(
    gap_rows: Sequence[dict[str, Any]],
    *,
    scopes: Sequence[str] = DEFAULT_SCOPES,
    profile_families: Sequence[str] = DEFAULT_GAP_FAMILIES,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family in profile_families:
        for scope in scopes:
            row = require_row(gap_rows, key="profile_family", values=[family], scope=scope, context="route gap")
            out.append(
                {
                    "profile_family": family,
                    "scope": scope,
                    "N": int(inum(row.get("N"))),
                    "gap_full_RMSE": fnum(row.get("gap_full_RMSE")),
                    "gap_full_NRMSE": fnum(row.get("gap_full_NRMSE")),
                    "gap_full_RMSE_pct_of_real": fnum(row.get("gap_full_RMSE_pct_of_real")),
                }
            )
    return out


def build_low_cal_summary(
    profile_choice_rows: Sequence[dict[str, Any]],
    blend_rows: Sequence[dict[str, Any]],
    *,
    budget: int = 96,
) -> list[dict[str, Any]]:
    blend_by_key = {
        (str(row.get("route")), str(row.get("client")), int(inum(row.get("budget_per_client")))): row
        for row in blend_rows
    }
    out: list[dict[str, Any]] = []
    for row in profile_choice_rows:
        if int(inum(row.get("budget_per_client"))) != int(budget):
            continue
        key = (str(row.get("route")), str(row.get("client")), int(budget))
        blend = blend_by_key.get(key, {})
        out.append(
            {
                "route": key[0],
                "client": key[1],
                "budget_per_client": int(budget),
                "profile_mode": str(row.get("profile_mode")),
                "profile_mode_rate": fnum(row.get("profile_mode_rate")),
                "H8_C4_rate": fnum(row.get("H8_C4_rate")),
                "blend_weight_mode": fnum(blend.get("weight_mode")),
                "blend_weight_mode_rate": fnum(blend.get("weight_mode_rate")),
            }
        )
    return sorted(out, key=lambda item: (item["route"], item["client"]))


def table_metric(rows: Sequence[dict[str, Any]], profile: str, scope: str, key: str) -> str:
    for row in rows:
        if row.get("profile") == profile and row.get("scope") == scope:
            digits = 4 if key.endswith("NRMSE") else 3
            return format_float(row.get(key), digits)
    return ""


def metric_cell(rows: Sequence[dict[str, Any]], profile: str, scope: str, *, rmse_key: str, nrmse_key: str) -> str:
    rmse = table_metric(rows, profile, scope, rmse_key)
    nrmse = table_metric(rows, profile, scope, nrmse_key)
    return f"{rmse} / {nrmse}" if rmse or nrmse else ""


def write_story_report(
    out_dir: Path,
    *,
    mainline_rows: Sequence[dict[str, Any]],
    post_qc_rows: Sequence[dict[str, Any]],
    route_gap_rows: Sequence[dict[str, Any]],
    low_cal_rows: Sequence[dict[str, Any]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 当前基座回归主线故事",
        "",
        "## Frozen Base And R3aK16",
        "",
        "当前阶段冻结 F6 分类基座、backbone features、H2.3+/H8 profile predictions 和 QC records。R3aK16/auto_v2 保留为 baseline、fallback 和 gate context，不再作为每轮回归优化都要重训的主线回归头。",
        "",
        "## Oracle-route Full",
        "",
        "| profile | ALL RMSE/NRMSE | C3 | C4 | C5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in [
        "H2.3 oracle-route",
        "H2.3+ oracle-route weak-blend",
        "H8+C4 oracle-route",
        "Guarded practical oracle-route",
    ]:
        cells = [metric_cell(mainline_rows, profile, scope, rmse_key="RMSE", nrmse_key="NRMSE") for scope in DEFAULT_SCOPES]
        lines.append(f"| {profile} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Accepted+Review",
            "",
            "| profile | ALL RMSE/NRMSE | C3 | C4 | C5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for profile in POST_QC_PROFILES:
        cells = [
            metric_cell(
                post_qc_rows,
                profile,
                scope,
                rmse_key="coverage_review_RMSE",
                nrmse_key="coverage_review_NRMSE",
            )
            for scope in DEFAULT_SCOPES
        ]
        lines.append(f"| {profile} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Route Gap",
            "",
            "| family | scope | N | gap RMSE | gap NRMSE | gap RMSE / real |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in route_gap_rows:
        if row["scope"] not in {"ALL", "C5"}:
            continue
        lines.append(
            "| {family} | {scope} | {n} | {rmse} | {nrmse} | {pct}% |".format(
                family=row["profile_family"],
                scope=row["scope"],
                n=row["N"],
                rmse=format_float(row["gap_full_RMSE"], 3),
                nrmse=format_float(row["gap_full_NRMSE"], 4),
                pct=format_float(100 * fnum(row["gap_full_RMSE_pct_of_real"]), 1),
            )
        )

    lines.extend(
        [
            "",
            "## Low Calibration Stability",
            "",
            "| route | client | budget | profile mode | mode rate | H8+C4 rate | blend weight mode | weight mode rate |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in low_cal_rows:
        lines.append(
            "| {route} | {client} | {budget} | {profile_mode} | {mode_rate}% | {h8_rate}% | {weight} | {weight_rate}% |".format(
                route=row["route"],
                client=row["client"],
                budget=row["budget_per_client"],
                profile_mode=row["profile_mode"],
                mode_rate=format_float(100 * fnum(row["profile_mode_rate"]), 1),
                h8_rate=format_float(100 * fnum(row["H8_C4_rate"]), 1),
                weight=format_float(row["blend_weight_mode"], 2),
                weight_rate=format_float(100 * fnum(row["blend_weight_mode_rate"]), 1),
            )
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- 主报告使用 oracle-route full-set 回答分类正确下的回归能力。",
            "- Accepted+Review 是部署补充，不替代 oracle-route 主指标。",
            "- real-route full-set 的大 gap 说明主要污染来自 classification/route error，尤其是 C5。",
            "- 当前基座内的后续优化应集中到 C5 CO-priority calibration/rescue。",
        ]
    )
    report_path = out_dir / "current_base_regression_story.zh.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mainline_rows = build_mainline_summary(
        read_csv(args.guarded_metrics),
        read_csv(args.h23_oracle_summary),
    )
    post_qc_rows = build_post_qc_summary(read_csv(args.post_qc_metrics))
    route_gap_rows = build_route_gap_summary(read_csv(args.route_gap_metrics))
    low_cal_rows = build_low_cal_summary(
        read_csv(args.profile_choice_summary),
        read_csv(args.blend_summary),
        budget=args.stability_budget,
    )

    write_csv(out_dir / "current_base_regression_mainline_summary.csv", mainline_rows)
    write_csv(out_dir / "current_base_regression_post_qc_summary.csv", post_qc_rows)
    write_csv(out_dir / "current_base_regression_route_gap_summary.csv", route_gap_rows)
    write_csv(out_dir / "current_base_regression_low_cal_summary.csv", low_cal_rows)
    report_path = write_story_report(
        out_dir,
        mainline_rows=mainline_rows,
        post_qc_rows=post_qc_rows,
        route_gap_rows=route_gap_rows,
        low_cal_rows=low_cal_rows,
    )
    manifest = {
        "guarded_metrics": args.guarded_metrics,
        "h23_oracle_summary": args.h23_oracle_summary,
        "post_qc_metrics": args.post_qc_metrics,
        "route_gap_metrics": args.route_gap_metrics,
        "profile_choice_summary": args.profile_choice_summary,
        "blend_summary": args.blend_summary,
        "stability_budget": args.stability_budget,
        "outputs": [
            "current_base_regression_mainline_summary.csv",
            "current_base_regression_post_qc_summary.csv",
            "current_base_regression_route_gap_summary.csv",
            "current_base_regression_low_cal_summary.csv",
            "current_base_regression_story.zh.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "report": str(report_path)}, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded-metrics", default="results/guarded_profile_selector_nonco05_20260630/guarded_profile_metrics.csv")
    parser.add_argument("--h23-oracle-summary", default="results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/h2_3_profile_replay/h2_3_profile_summary.csv")
    parser.add_argument("--post-qc-metrics", default="results/guarded_profile_selector_20260630/profile_qc_oracle/profile_post_qc_metrics.csv")
    parser.add_argument("--route-gap-metrics", default="results/real_vs_oracle_gap_audit_20260630/real_vs_oracle_gap_metrics.csv")
    parser.add_argument("--profile-choice-summary", default="results/low_calibration_profile_choice_stress_20260630/low_calibration_profile_choice_selection_summary.csv")
    parser.add_argument("--blend-summary", default="results/low_calibration_blend_stress_20260630/low_calibration_selection_summary.csv")
    parser.add_argument("--stability-budget", type=int, default=96)
    parser.add_argument("--output-dir", default="results/current_base_regression_mainline_20260701")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
