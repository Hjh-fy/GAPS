# Current Base Regression Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen-current-base regression story summarizer that reads existing CSV outputs, writes compact summary tables, and generates a Chinese report for the current F6/H2.3+/H8 guarded mainline.

**Architecture:** Add one focused summary script that only reads existing artifacts and never retrains models. Tests cover row extraction, missing-input failures, report section coverage, and output manifests. The generated results live under `results/current_base_regression_mainline_20260701/`; committed source changes are limited to script, tests, and final docs.

**Tech Stack:** Python standard library, existing CSV helpers from `run_regression_head_ablation.py`, existing formatting helper from `run_profile_qc_coverage_audit.py`, pytest.

---

## File Structure

- Create `run_current_base_regression_story.py`: CLI summary script. It reads frozen profile metrics, post-QC metrics, route-gap metrics, low-cal stress summaries, writes normalized CSV summaries, writes `current_base_regression_story.zh.md`, and writes `manifest.json`.
- Create `tests/test_current_base_regression_story.py`: unit tests for the summary helpers and report writer using small in-memory rows.
- Create `docs/superpowers/reports/2026-07-01-current-base-regression-mainline.zh.md`: committed narrative copy of the generated story report after running the script.
- Results directory, not committed: `results/current_base_regression_mainline_20260701/`.

---

### Task 1: Summary Helper Tests

**Files:**
- Create: `tests/test_current_base_regression_story.py`
- Later create: `run_current_base_regression_story.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_current_base_regression_story.py` with:

```python
from pathlib import Path

import pytest

from run_current_base_regression_story import (
    build_low_cal_summary,
    build_mainline_summary,
    build_post_qc_summary,
    build_route_gap_summary,
    write_story_report,
)


def test_build_mainline_summary_extracts_oracle_profiles():
    rows = [
        {"mode": "oracle-route_H2.3+", "scope": "ALL", "N": "10", "RMSE": "9.8", "NRMSE": "0.05"},
        {"mode": "oracle-route_H2.3+", "scope": "C3", "N": "4", "RMSE": "9.1", "NRMSE": "0.04"},
        {"mode": "oracle-route_H8+C4", "scope": "ALL", "N": "10", "RMSE": "9.0", "NRMSE": "0.051"},
        {"mode": "oracle-route_guarded_profile", "scope": "ALL", "N": "10", "RMSE": "9.1", "NRMSE": "0.049"},
    ]
    h23_rows = [
        {"mode": "H2_3_oracle_route", "scope": "ALL", "N": "10", "RMSE": "10.5", "NRMSE": "0.056"},
    ]

    out = build_mainline_summary(rows, h23_rows, scopes=["ALL", "C3"])

    assert out[0] == {
        "profile": "H2.3 oracle-route",
        "scope": "ALL",
        "N": 10,
        "RMSE": 10.5,
        "NRMSE": 0.056,
    }
    assert {"profile": "Guarded practical oracle-route", "scope": "ALL", "N": 10, "RMSE": 9.1, "NRMSE": 0.049} in out


def test_build_mainline_summary_requires_profile_scope_pair():
    rows = [{"mode": "oracle-route_H2.3+", "scope": "ALL", "N": "10", "RMSE": "9.8", "NRMSE": "0.05"}]

    with pytest.raises(ValueError, match="Missing metric row"):
        build_mainline_summary(rows, [], scopes=["ALL"])


def test_build_post_qc_summary_extracts_accepted_review_metrics():
    rows = [
        {
            "profile": "Guarded practical oracle-route",
            "scope": "ALL",
            "N": "10",
            "coverage_review": "0.75",
            "nonreject_N": "8",
            "coverage_review_RMSE": "6.3",
            "coverage_review_NRMSE": "0.035",
        }
    ]

    out = build_post_qc_summary(rows, scopes=["ALL"])

    assert out == [
        {
            "profile": "Guarded practical oracle-route",
            "scope": "ALL",
            "N": 10,
            "coverage_review": 0.75,
            "nonreject_N": 8,
            "coverage_review_RMSE": 6.3,
            "coverage_review_NRMSE": 0.035,
        }
    ]


def test_build_route_gap_summary_keeps_core_gap_fields():
    rows = [
        {
            "profile_family": "H2.3+",
            "scope": "C5",
            "N": "1360",
            "gap_full_RMSE": "27.09",
            "gap_full_NRMSE": "0.259",
            "gap_full_RMSE_pct_of_real": "0.69",
        }
    ]

    assert build_route_gap_summary(rows, scopes=["C5"], profile_families=["H2.3+"]) == [
        {
            "profile_family": "H2.3+",
            "scope": "C5",
            "N": 1360,
            "gap_full_RMSE": 27.09,
            "gap_full_NRMSE": 0.259,
            "gap_full_RMSE_pct_of_real": 0.69,
        }
    ]


def test_build_low_cal_summary_keeps_budget_96_modes():
    profile_choice_rows = [
        {
            "route": "oracle-route",
            "budget_per_client": "96",
            "client": "C5",
            "H2_3_plus_rate": "0.0",
            "H8_C4_rate": "1.0",
            "profile_mode": "H8+C4",
            "profile_mode_rate": "1.0",
        }
    ]
    blend_rows = [
        {
            "route": "oracle-route",
            "budget_per_client": "96",
            "client": "C5",
            "weight_mode": "0.25",
            "weight_mode_rate": "1.0",
        }
    ]

    out = build_low_cal_summary(profile_choice_rows, blend_rows, budget=96)

    assert out == [
        {
            "route": "oracle-route",
            "client": "C5",
            "budget_per_client": 96,
            "profile_mode": "H8+C4",
            "profile_mode_rate": 1.0,
            "H8_C4_rate": 1.0,
            "blend_weight_mode": 0.25,
            "blend_weight_mode_rate": 1.0,
        }
    ]


def test_write_story_report_contains_required_sections(tmp_path: Path):
    report = write_story_report(
        tmp_path,
        mainline_rows=[{"profile": "Guarded practical oracle-route", "scope": "ALL", "N": 10, "RMSE": 9.1, "NRMSE": 0.049}],
        post_qc_rows=[{"profile": "Guarded practical oracle-route", "scope": "ALL", "N": 10, "coverage_review": 0.75, "nonreject_N": 8, "coverage_review_RMSE": 6.3, "coverage_review_NRMSE": 0.035}],
        route_gap_rows=[{"profile_family": "H2.3+", "scope": "C5", "N": 1360, "gap_full_RMSE": 27.09, "gap_full_NRMSE": 0.259, "gap_full_RMSE_pct_of_real": 0.69}],
        low_cal_rows=[{"route": "oracle-route", "client": "C5", "budget_per_client": 96, "profile_mode": "H8+C4", "profile_mode_rate": 1.0, "H8_C4_rate": 1.0, "blend_weight_mode": 0.25, "blend_weight_mode_rate": 1.0}],
    )

    text = Path(report).read_text(encoding="utf-8")
    assert "Oracle-route Full" in text
    assert "Accepted+Review" in text
    assert "Route Gap" in text
    assert "R3aK16" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_current_base_regression_story.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'run_current_base_regression_story'`.

- [ ] **Step 3: Keep red tests uncommitted**

Run:

```powershell
git status --short -- tests/test_current_base_regression_story.py
```

Expected: `?? tests/test_current_base_regression_story.py`. Leave the red test uncommitted; Task 2 commits the test and implementation together after green verification.

---

### Task 2: Summary Script Implementation

**Files:**
- Create: `run_current_base_regression_story.py`
- Modify: `tests/test_current_base_regression_story.py` only if assertions need field-name alignment.

- [ ] **Step 1: Implement the script skeleton and helpers**

Create `run_current_base_regression_story.py` with these public functions and CLI defaults:

```python
"""Summarize the frozen current-base regression mainline story.

This script reads existing experiment CSVs only. It does not train or replay
models. Outputs are compact tables and a Chinese story report for the current
F6/H2.3+/H8 guarded mainline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import fnum, inum, read_csv, write_csv


MAINLINE_PROFILE_SPECS = [
    ("H2.3 oracle-route", "H2_3_oracle_route", "h23"),
    ("H2.3+ oracle-route weak-blend", "oracle-route_H2.3+", "guarded"),
    ("H8+C4 oracle-route", "oracle-route_H8+C4", "guarded"),
    ("Guarded practical oracle-route", "oracle-route_guarded_profile", "guarded"),
]

POST_QC_PROFILES = [
    "H2.3+ oracle-route weak-blend",
    "H8+C4 oracle-route",
    "Guarded practical oracle-route",
    "Client prior C34 H2.3+ / C5 H8+C4 oracle-route",
]

DEFAULT_SCOPES = ["ALL", "C3", "C4", "C5"]
DEFAULT_GAP_FAMILIES = ["H2.3", "H2.3+", "H8+C4", "client_selector"]


def require_row(rows: Sequence[dict[str, Any]], *, key: str, value: str, scope: str, context: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get(key)) == value and str(row.get("scope")) == scope:
            return row
    raise ValueError(f"Missing metric row for {context}: {key}={value}, scope={scope}")


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
    for profile, mode, source in MAINLINE_PROFILE_SPECS:
        source_rows = h23_metric_rows if source == "h23" else guarded_metric_rows
        for scope in scopes:
            row = require_row(source_rows, key="mode", value=mode, scope=scope, context="oracle full mainline")
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
            row = require_row(post_qc_rows, key="profile", value=profile, scope=scope, context="post QC")
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
            row = require_row(gap_rows, key="profile_family", value=family, scope=scope, context="route gap")
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
```

- [ ] **Step 2: Implement report writer**

Append report writer helpers:

```python
def table_metric(rows: Sequence[dict[str, Any]], profile: str, scope: str, key: str) -> str:
    for row in rows:
        if row.get("profile") == profile and row.get("scope") == scope:
            return format_float(row.get(key), 3 if key.endswith("RMSE") else 4)
    return ""


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
    for profile in ["H2.3 oracle-route", "H2.3+ oracle-route weak-blend", "H8+C4 oracle-route", "Guarded practical oracle-route"]:
        cells = []
        for scope in ["ALL", "C3", "C4", "C5"]:
            cells.append(f"{table_metric(mainline_rows, profile, scope, 'RMSE')} / {table_metric(mainline_rows, profile, scope, 'NRMSE')}")
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
    for profile in ["H2.3+ oracle-route weak-blend", "H8+C4 oracle-route", "Guarded practical oracle-route", "Client prior C34 H2.3+ / C5 H8+C4 oracle-route"]:
        cells = []
        for scope in ["ALL", "C3", "C4", "C5"]:
            cells.append(
                f"{table_metric(post_qc_rows, profile, scope, 'coverage_review_RMSE')} / "
                f"{table_metric(post_qc_rows, profile, scope, 'coverage_review_NRMSE')}"
            )
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
```

- [ ] **Step 3: Implement CLI runner**

Append CLI code:

```python
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
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/test_current_base_regression_story.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit script and tests**

Run:

```powershell
git add -- run_current_base_regression_story.py tests/test_current_base_regression_story.py
git commit -m "feat: add current base regression story summarizer"
```

Expected: commit succeeds with only the new script and test.

---

### Task 3: Generate Current-Base Story Outputs

**Files:**
- Generate untracked ignored outputs under `results/current_base_regression_mainline_20260701/`
- Create committed report: `docs/superpowers/reports/2026-07-01-current-base-regression-mainline.zh.md`

- [ ] **Step 1: Run the summarizer**

Run:

```powershell
python run_current_base_regression_story.py --output-dir results/current_base_regression_mainline_20260701
```

Expected: JSON output includes:

```json
{
  "output_dir": "results\\current_base_regression_mainline_20260701",
  "report": "results\\current_base_regression_mainline_20260701\\current_base_regression_story.zh.md"
}
```

- [ ] **Step 2: Inspect summary outputs**

Run:

```powershell
Import-Csv results/current_base_regression_mainline_20260701/current_base_regression_mainline_summary.csv | Where-Object { $_.scope -eq 'ALL' } | Format-Table -AutoSize
Import-Csv results/current_base_regression_mainline_20260701/current_base_regression_post_qc_summary.csv | Where-Object { $_.scope -eq 'ALL' } | Format-Table -AutoSize
Import-Csv results/current_base_regression_mainline_20260701/current_base_regression_route_gap_summary.csv | Where-Object { $_.scope -in @('ALL','C5') } | Format-Table -AutoSize
```

Expected key values:

- Guarded practical oracle-route ALL full RMSE/NRMSE near `9.109 / 0.0489`.
- Guarded practical oracle-route ALL Accepted+Review RMSE/NRMSE near `6.375 / 0.0356`.
- H2.3+ C5 route gap RMSE near `27.09`.

- [ ] **Step 3: Copy generated report into committed docs**

Use `apply_patch` to create `docs/superpowers/reports/2026-07-01-current-base-regression-mainline.zh.md` from the generated report content, with one additional top note:

```markdown
> Generated from `run_current_base_regression_story.py` on 2026-07-01 using frozen current-base CSV artifacts.
```

- [ ] **Step 4: Commit the report**

Run:

```powershell
git add -- docs/superpowers/reports/2026-07-01-current-base-regression-mainline.zh.md
git commit -m "docs: add current base regression mainline story"
```

Expected: commit succeeds with only the story report.

---

### Task 4: Verification And Final Handoff

**Files:**
- No new source files unless verification exposes a defect.

- [ ] **Step 1: Run focused verification**

Run:

```powershell
python -m pytest tests/test_current_base_regression_story.py tests/test_guarded_profile_selector.py tests/test_profile_qc_coverage_audit.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run supporting regression tests**

Run:

```powershell
python -m pytest tests/test_low_calibration_profile_choice_stress.py tests/test_low_calibration_blend_stress.py tests/test_route_gap_audit.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short --branch
```

Expected:

- Current branch remains `codex/regression-aware-fusion`.
- Only unrelated pre-existing untracked files remain.
- No staged files remain.

- [ ] **Step 4: Final response**

Report:

- Script path: `run_current_base_regression_story.py`.
- Output directory: `results/current_base_regression_mainline_20260701/`.
- Report path: `docs/superpowers/reports/2026-07-01-current-base-regression-mainline.zh.md`.
- Key metrics: oracle full guarded practical `9.109 / 0.0489`, Accepted+Review `6.375 / 0.0356`, C5 H2.3+ route gap around `27.09`.
- Tests run and pass.

---

## Self-Review

- Spec coverage: The plan implements frozen-base reading, R3aK16 freeze explanation, oracle-route full summary, Accepted+Review summary, route-gap summary, low-cal stability summary, and C5-focused next-step story.
- Scope check: The plan stays inside current C12 source to C345 target current-base artifacts and does not add cross-domain validation.
- Type consistency: Helper names used by tests match the implementation names in Task 2.
- Completeness scan: The plan contains concrete file paths, functions, commands, expected outputs, and commit messages.
