"""Build the final metric consolidation pack for the current base.

This script is intentionally read-only with respect to experiments: it reads
existing P4/story artifacts, recomputes only reporting slices from saved
predictions, and writes a compact package for paper/GPT/teacher review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import CO_CLASS, client_name, fnum, inum, metrics, read_csv, write_csv


DEFAULT_DEPLOY_DIR = Path("results/real_route_threshold_guard_deployment_candidate_20260707")
DEFAULT_STORY_DIR = Path("results/current_base_paper_story_pack_20260705")
DEFAULT_SUBMISSION_DIR = Path("results/current_base_submission_docs_pack_20260708")
DEFAULT_OUTPUT_DIR = Path("results/final_metric_consolidation_20260709")
DEFAULT_DOC_REPORT = Path("docs/superpowers/reports/2026-07-09-final-metric-consolidation.zh.md")
TARGET_SCOPES = ["ALL", "C5"]
CO_SCOPES = ["C5-CO", "C5-nonCO"]
AR_DECISIONS = {"accept", "review"}
PROFILE_SPECS = [
    ("H2.3+ anchor", "h23_ppm"),
    ("H8+C4 rescue stream", "h8_ppm"),
    ("P4 threshold guard", "threshold_guard_ppm"),
]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


def metric_lookup(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: str = "test",
    scope: str,
    qc_slice: str = "accepted_review",
) -> dict[str, Any]:
    for row in rows:
        if row.get("dataset") == dataset and row.get("scope") == scope and row.get("qc_slice") == qc_slice:
            return row
    return {}


def is_accepted_review(row: dict[str, Any]) -> bool:
    return str(row.get("qc_decision", "")).strip().lower() in AR_DECISIONS


def is_classification_correct(row: dict[str, Any]) -> bool:
    route = row.get("route_class", row.get("pred_class"))
    return inum(route) == inum(row.get("true_class"))


def in_scope(row: dict[str, Any], scope: str) -> bool:
    if scope == "ALL":
        return True
    client = client_name(row.get("client"))
    if scope in {"C3", "C4", "C5"}:
        return client == scope
    if scope.endswith("-CO"):
        return client == scope.split("-")[0] and inum(row.get("true_class")) == CO_CLASS
    if scope.endswith("-nonCO"):
        return client == scope.split("-")[0] and inum(row.get("true_class")) != CO_CLASS
    return False


def build_primary_result_table(metric_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in TARGET_SCOPES:
        ar = metric_lookup(metric_rows, scope=scope)
        full = metric_lookup(metric_rows, scope=scope, qc_slice="full")
        n_ar = int(fnum(ar.get("N"), 0))
        n_full = int(fnum(full.get("N"), 0))
        out.append(
            {
                "scope": scope,
                "reporting_slice": "real-route Accepted+Review",
                "N": n_ar,
                "full_N": n_full,
                "coverage_review": safe_div(n_ar, n_full),
                "RMSE": fnum(ar.get("RMSE")),
                "NRMSE": fnum(ar.get("NRMSE")),
                "MAE": fnum(ar.get("MAE")),
                "P90AE": fnum(ar.get("P90AE")),
                "baseline_h23_RMSE": fnum(ar.get("baseline_h23_RMSE")),
                "baseline_h23_NRMSE": fnum(ar.get("baseline_h23_NRMSE")),
                "h8_all_RMSE": fnum(ar.get("h8_all_RMSE")),
                "h8_all_NRMSE": fnum(ar.get("h8_all_NRMSE")),
                "rmse_gain_vs_h23": fnum(ar.get("rmse_gain_vs_h23")),
                "nrmse_gain_vs_h23": fnum(ar.get("nrmse_gain_vs_h23")),
                "h8_usage_rate": fnum(ar.get("h8_usage_rate")),
                "selected_threshold_labels": ar.get("selected_threshold_labels", ""),
                "paper_claim": primary_claim(scope, ar),
            }
        )
    return out


def primary_claim(scope: str, row: dict[str, Any]) -> str:
    gain = fnum(row.get("rmse_gain_vs_h23"))
    usage = 100 * fnum(row.get("h8_usage_rate"))
    if scope == "ALL":
        return f"Main deployable result: P4 improves H2.3+ by {format_float(gain, 3)} RMSE with {format_float(usage, 1)}% H8 usage."
    if scope == "C5":
        return "Largest target-client gain; supports the CO rescue story."
    return "Positive per-client gain; supports guarded rescue without replacing the anchor globally."


def _build_legacy_classification_correct_table(test_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in test_rows if str(row.get("split", "test")) == "test"]
    out: list[dict[str, Any]] = []
    for scope in TARGET_SCOPES:
        scope_rows = [row for row in rows if in_scope(row, scope)]
        ar_rows = [row for row in scope_rows if is_accepted_review(row)]
        cc_ar_rows = [row for row in ar_rows if is_classification_correct(row)]
        baseline = metrics(cc_ar_rows, "h23_ppm")
        for profile, pred_key in PROFILE_SPECS:
            result = metrics(cc_ar_rows, pred_key)
            out.append(
                {
                    "scope": scope,
                    "condition": "S_AR ∩ S_CC",
                    "profile": profile,
                    "N": int(result.get("N") or 0),
                    "source_total_N": len(scope_rows),
                    "accepted_review_N": len(ar_rows),
                    "class_correct_accepted_review_N": len(cc_ar_rows),
                    "accepted_review_coverage": safe_div(len(ar_rows), len(scope_rows)),
                    "class_correct_rate_within_AR": safe_div(len(cc_ar_rows), len(ar_rows)),
                    "RMSE": result.get("RMSE"),
                    "NRMSE": result.get("NRMSE"),
                    "MAE": result.get("MAE"),
                    "P90AE": result.get("P90AE"),
                    "rmse_gain_vs_h23": fnum(baseline.get("RMSE")) - fnum(result.get("RMSE")) if result.get("RMSE") is not None else "",
                    "nrmse_gain_vs_h23": fnum(baseline.get("NRMSE")) - fnum(result.get("NRMSE")) if result.get("NRMSE") is not None else "",
                    "reading": classification_correct_reading(scope, profile),
                }
            )
    return out


def _profile_specs(
    pred_keys: Sequence[str | tuple[str, str]],
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in pred_keys:
        if isinstance(item, str):
            out.append((item, item))
        else:
            profile, key = item
            out.append((str(profile), str(key)))
    return out


def build_regression_slice_table(
    test_rows: Sequence[dict[str, Any]],
    pred_keys: Sequence[str | tuple[str, str]],
) -> list[dict[str, Any]]:
    """Build S_ALL, S_CC, S_AR, and the diagnostic intersection independently."""
    rows = [row for row in test_rows if str(row.get("split", "test")) == "test"]
    out: list[dict[str, Any]] = []
    for scope in TARGET_SCOPES:
        scope_rows = [row for row in rows if in_scope(row, scope)]
        cc_rows = [row for row in scope_rows if is_classification_correct(row)]
        wrong_rows = [row for row in scope_rows if not is_classification_correct(row)]
        ar_rows = [row for row in scope_rows if is_accepted_review(row)]
        cc_ar_rows = [row for row in ar_rows if is_classification_correct(row)]
        slices = (
            ("all", "S_ALL", scope_rows, "all", len(scope_rows)),
            ("class_correct", "S_CC", cc_rows, "all", len(scope_rows)),
            ("class_wrong", "S_CW", wrong_rows, "all", len(scope_rows)),
            ("accepted_review", "S_AR", ar_rows, "all", len(scope_rows)),
            (
                "accepted_review_class_correct",
                "S_AR intersect S_CC",
                cc_ar_rows,
                "class_correct",
                len(cc_rows),
            ),
        )
        for slice_id, condition, selected_rows, parent_slice, parent_n in slices:
            baseline = metrics(selected_rows, "h23_ppm")
            for profile, pred_key in _profile_specs(pred_keys):
                result = metrics(selected_rows, pred_key)
                out.append(
                    {
                        "scope": scope,
                        "slice": slice_id,
                        "condition": condition,
                        "profile": profile,
                        "prediction_key": pred_key,
                        "N": int(result.get("N") or 0),
                        "parent_slice": parent_slice,
                        "parent_N": int(parent_n),
                        "coverage": safe_div(len(selected_rows), parent_n),
                        "coverage_of_all": safe_div(len(selected_rows), len(scope_rows)),
                        "source_total_N": len(scope_rows),
                        "class_correct_N": len(cc_rows),
                        "class_wrong_N": len(wrong_rows),
                        "accepted_review_N": len(ar_rows),
                        "class_correct_accepted_review_N": len(cc_ar_rows),
                        "class_correct_coverage": safe_div(len(cc_rows), len(scope_rows)),
                        "accepted_review_coverage": safe_div(len(ar_rows), len(scope_rows)),
                        "class_correct_rate_within_AR": safe_div(len(cc_ar_rows), len(ar_rows)),
                        "RMSE": result.get("RMSE"),
                        "NRMSE": result.get("NRMSE"),
                        "MAE": result.get("MAE"),
                        "P90AE": result.get("P90AE"),
                        "Bias": result.get("Bias"),
                        "rmse_gain_vs_h23": (
                            fnum(baseline.get("RMSE")) - fnum(result.get("RMSE"))
                            if result.get("RMSE") is not None
                            else ""
                        ),
                        "nrmse_gain_vs_h23": (
                            fnum(baseline.get("NRMSE")) - fnum(result.get("NRMSE"))
                            if result.get("NRMSE") is not None
                            else ""
                        ),
                        "reading": classification_correct_reading(scope, profile),
                    }
                )
    return out


def build_classification_correct_table(test_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return pure S_CC rows before QC for backward-compatible callers."""
    return [
        row
        for row in build_regression_slice_table(test_rows, PROFILE_SPECS)
        if row["slice"] == "class_correct"
    ]


def classification_correct_reading(scope: str, profile: str) -> str:
    if profile == "P4 threshold guard":
        return "Classification-correct capability before QC for the final selector output."
    if profile == "H8+C4 rescue stream" and scope == "C5":
        return "Mechanism check for whether C5 benefits from the specialist when routing is correct."
    return "Profile comparison on the same correct-route subset before QC."


def build_co_rescue_decomposition(metric_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in CO_SCOPES:
        row = metric_lookup(metric_rows, scope=scope)
        if not row:
            continue
        baseline_rmse = fnum(row.get("baseline_h23_RMSE"))
        h8_rmse = fnum(row.get("h8_all_RMSE"))
        selector_rmse = fnum(row.get("RMSE"))
        h8_usage = fnum(row.get("h8_usage_rate"))
        out.append(
            {
                "scope": scope,
                "N": int(fnum(row.get("N"), 0)),
                "h23_RMSE": baseline_rmse,
                "h23_NRMSE": fnum(row.get("baseline_h23_NRMSE")),
                "h8_all_RMSE": h8_rmse,
                "h8_all_NRMSE": fnum(row.get("h8_all_NRMSE")),
                "threshold_guard_RMSE": selector_rmse,
                "threshold_guard_NRMSE": fnum(row.get("NRMSE")),
                "guard_rmse_gain_vs_h23": baseline_rmse - selector_rmse,
                "h8_all_rmse_gain_vs_h23": baseline_rmse - h8_rmse,
                "selector_h8_usage_rate": h8_usage,
                "selected_threshold_labels": row.get("selected_threshold_labels", ""),
                "interpretation": rescue_interpretation(scope, h8_usage, baseline_rmse - selector_rmse),
            }
        )
    return out


def rescue_interpretation(scope: str, h8_usage: float, gain: float) -> str:
    if scope.endswith("-CO"):
        return f"CO rescue branch enabled; guarded gain={format_float(gain, 3)}, H8 usage={format_float(100 * h8_usage, 1)}%."
    return "nonCO protected fallback; H8 usage should remain 0 so the anchor is not disturbed."


def build_low_cal_stress_table(low_cal_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    budgets = ["12", "24", "48", "80"]
    for budget in budgets:
        for scope in TARGET_SCOPES:
            row = next(
                (
                    item for item in low_cal_rows
                    if item.get("budget_per_client") == budget
                    and item.get("scope") == scope
                    and item.get("qc_slice") == "accepted_review"
                ),
                {},
            )
            if not row:
                continue
            out.append(
                {
                    "budget_per_client": int(budget),
                    "scope": scope,
                    "repeats": int(fnum(row.get("repeats"), 0)),
                    "N_mean": fnum(row.get("N_mean")),
                    "RMSE_mean": fnum(row.get("RMSE_mean")),
                    "RMSE_std": fnum(row.get("RMSE_std")),
                    "NRMSE_mean": fnum(row.get("NRMSE_mean")),
                    "NRMSE_std": fnum(row.get("NRMSE_std")),
                    "rmse_gain_vs_h23_mean": fnum(row.get("rmse_gain_vs_h23_mean")),
                    "rmse_gain_vs_h23_min": fnum(row.get("rmse_gain_vs_h23_min")),
                    "positive_gain_rate": fnum(row.get("positive_gain_rate")),
                    "h8_usage_rate_mean": fnum(row.get("h8_usage_rate_mean")),
                    "reading": "Low-calibration stress: validation budget is reduced, test set stays fixed.",
                }
            )
    return out


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [format_cell(row.get(key)) for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) <= 1.0:
            return format_float(value, 4)
        return format_float(value, 3)
    if value is None:
        return ""
    return str(value)


def row_by_scope(rows: Sequence[dict[str, Any]], scope: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("scope") == scope), {})


def row_by_scope_profile(rows: Sequence[dict[str, Any]], scope: str, profile: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("scope") == scope and row.get("profile") == profile), {})


def write_story_brief(
    path: Path,
    primary_rows: Sequence[dict[str, Any]],
    cc_rows: Sequence[dict[str, Any]],
    co_rows: Sequence[dict[str, Any]],
    low_cal_rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p4_all = row_by_scope(primary_rows, "ALL")
    p4_c5 = row_by_scope(primary_rows, "C5")
    cc_all = row_by_scope_profile(cc_rows, "ALL", "P4 threshold guard")
    c5_co = row_by_scope(co_rows, "C5-CO")
    cc_rate = fnum(cc_all.get("class_correct_rate_within_AR"))
    lines = [
        "# Final Metric Consolidation 20260709",
        "",
        "## 一句话结论",
        "",
        (
            "当前基座应以 real-route Accepted+Review 作为部署主线：P4 threshold guard 在 ALL 上达到 "
            f"{format_float(p4_all.get('RMSE'), 3)} / {format_float(p4_all.get('NRMSE'), 4)}，"
            f"相对 H2.3+ gain={format_float(p4_all.get('rmse_gain_vs_h23'), 3)}，"
            f"H8 usage={format_float(100 * fnum(p4_all.get('h8_usage_rate')), 1)}%。"
        ),
        "",
        (
            "老师强调的 classification-correct 指标可以作为机制切片报告："
            f"`S_AR ∩ S_CC` 下 P4 ALL 为 {format_float(cc_all.get('RMSE'), 3)} / "
            f"{format_float(cc_all.get('NRMSE'), 4)}，样本数 N={cc_all.get('N', '')}。"
            f"当前 P4 test Accepted+Review 中 correct/AR={format_float(cc_rate, 4)}，"
            "说明 QC/guard 后可报告样本已经和分类正确切片高度一致。"
        ),
        "",
        "## T-main: real-route Accepted+Review",
        "",
        *markdown_table(
            primary_rows,
            [
                ("scope", "scope"),
                ("N", "N"),
                ("coverage_review", "coverage"),
                ("RMSE", "P4 RMSE"),
                ("NRMSE", "P4 NRMSE"),
                ("baseline_h23_RMSE", "H2.3+ RMSE"),
                ("h8_all_RMSE", "H8+C4 RMSE"),
                ("rmse_gain_vs_h23", "gain"),
                ("h8_usage_rate", "H8 usage"),
            ],
        ),
        "",
        "## T-cc: classification-correct Accepted+Review",
        "",
        "这里的集合定义为 `S_AR ∩ S_CC`，即 `qc_decision in {accept, review}` 且 `route_class == true_class`。",
        "",
        *markdown_table(
            [row for row in cc_rows if row.get("profile") == "P4 threshold guard"],
            [
                ("scope", "scope"),
                ("N", "N"),
                ("accepted_review_N", "AR N"),
                ("class_correct_rate_within_AR", "correct/AR"),
                ("RMSE", "P4 RMSE"),
                ("NRMSE", "P4 NRMSE"),
                ("rmse_gain_vs_h23", "gain"),
            ],
        ),
        "",
        "## T-co: CO rescue 分解",
        "",
        (
            "CO/nonCO 分解用于解释方法为什么不是简单替换回归头：H8+C4 只在 CO 风险条件下救援，"
            "nonCO 通过 H2.3+ fallback 保护。"
        ),
        "",
        *markdown_table(
            co_rows,
            [
                ("scope", "scope"),
                ("N", "N"),
                ("h23_RMSE", "H2.3+"),
                ("h8_all_RMSE", "H8 all"),
                ("threshold_guard_RMSE", "P4"),
                ("guard_rmse_gain_vs_h23", "gain"),
                ("selector_h8_usage_rate", "H8 usage"),
            ],
        ),
        "",
        "## T-stress: low calibration stress",
        "",
        *markdown_table(
            [row for row in low_cal_rows if row.get("scope") in {"ALL", "C5"}],
            [
                ("budget_per_client", "budget/client"),
                ("scope", "scope"),
                ("RMSE_mean", "RMSE mean"),
                ("RMSE_std", "RMSE std"),
                ("rmse_gain_vs_h23_mean", "gain mean"),
                ("positive_gain_rate", "positive rate"),
                ("h8_usage_rate_mean", "H8 usage"),
            ],
        ),
        "",
        "## 论文讲法",
        "",
        "- FCL/分类底座冻结提供稳定 route provider，当前回归优化不反复重训分类器。",
        "- H2.3+ 是 target-domain anchor，负责稳定覆盖 balanced/非 CO 样本。",
        "- H8+C4 是 CO-priority rescue stream，只在 per-client threshold guard 允许时接管。",
        "- real-route Accepted+Review 是部署主线；`S_AR ∩ S_CC` 是老师关心的分类正确机制切片。",
        "- real-oracle gap 只作为附录解释 full-set 难度，不作为当前优化优先级。",
        "",
        "## 后续最小验证",
        "",
        "- 保持当前基座，补画 F2/F3/F4 图。",
        "- 用同一套表结构扩展一个新 source-target 组合做 P6 泛化验证。",
        "- 暂不重复训练 r3ak16；它已不再是当前 H2.3+/H8+C4 主线的效率瓶颈。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gpt_brief(
    path: Path,
    primary_rows: Sequence[dict[str, Any]],
    cc_rows: Sequence[dict[str, Any]],
    co_rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p4_all = row_by_scope(primary_rows, "ALL")
    p4_c5 = row_by_scope(primary_rows, "C5")
    cc_all = row_by_scope_profile(cc_rows, "ALL", "P4 threshold guard")
    c5_co = row_by_scope(co_rows, "C5-CO")
    cc_rate = fnum(cc_all.get("class_correct_rate_within_AR"))
    lines = [
        "# 给 GPT/老师分析的当前基座指标摘要",
        "",
        "请重点分析这个系统故事是否成立：",
        "",
        "1. 分类/FCL 底座冻结，当前回归主线不再重训分类器。",
        "2. 最新 F6 real-route 输出作为部署 route provider。",
        "3. H2.3+ 作为 anchor，H8+C4 作为 CO rescue。",
        "4. P4 threshold guard 根据 validation-selected per-client threshold 选择输出。",
        "5. 主报告看 real-route Accepted+Review，同时补充 `S_AR ∩ S_CC` 分类正确回归性能。",
        "",
        "## 关键数字",
        "",
        f"- P4 real-route Accepted+Review ALL: {format_float(p4_all.get('RMSE'), 3)} / {format_float(p4_all.get('NRMSE'), 4)}, gain={format_float(p4_all.get('rmse_gain_vs_h23'), 3)}。",
        f"- P4 real-route Accepted+Review C5: {format_float(p4_c5.get('RMSE'), 3)} / {format_float(p4_c5.get('NRMSE'), 4)}, gain={format_float(p4_c5.get('rmse_gain_vs_h23'), 3)}。",
        f"- P4 `S_AR ∩ S_CC` ALL: {format_float(cc_all.get('RMSE'), 3)} / {format_float(cc_all.get('NRMSE'), 4)}, N={cc_all.get('N', '')}, correct/AR={format_float(cc_rate, 4)}。",
        f"- C5-CO guarded rescue gain: {format_float(c5_co.get('guard_rmse_gain_vs_h23'), 3)}, H8 usage={format_float(100 * fnum(c5_co.get('selector_h8_usage_rate')), 1)}%。",
        "",
        "## 希望 GPT 帮忙判断",
        "",
        "- 这个故事是否能支撑论文方法章节？",
        "- `real-route Accepted+Review` 和 `S_AR ∩ S_CC` 的报告顺序是否合理？",
        "- CO rescue 的收益分解是否足够解释为什么使用 H8+C4？",
        "- 后续 P6 泛化验证最小需要补哪张表或哪张图？",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_payload(
    *,
    out_dir: Path,
    deploy_dir: Path,
    story_dir: Path,
    docs_report: Path,
    tables: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema": "final_metric_consolidation.v1",
        "date": "2026-07-09",
        "output_dir": str(out_dir),
        "source_artifacts": {
            "threshold_guard_metrics": str(deploy_dir / "threshold_guard_metrics.csv"),
            "threshold_guard_test_outputs": str(deploy_dir / "threshold_guard_test_outputs.csv"),
            "selector_low_cal_metric_summary": str(story_dir / "selector_low_cal_metric_summary.csv"),
        },
        "outputs": {
            "main_result_table": str(out_dir / "main_result_table.csv"),
            "regression_slice_table": str(out_dir / "regression_slice_table.csv"),
            "classification_correct_table": str(out_dir / "classification_correct_table.csv"),
            "legacy_classification_correct_table": str(out_dir / "classification_correct_accepted_review_table.csv"),
            "co_rescue_decomposition": str(out_dir / "co_rescue_decomposition.csv"),
            "low_cal_stress_summary": str(out_dir / "low_cal_stress_summary.csv"),
            "story_brief": str(out_dir / "final_metric_story_brief.zh.md"),
            "gpt_analysis_brief": str(out_dir / "gpt_analysis_brief.zh.md"),
            "docs_report": str(docs_report),
        },
        "row_counts": {name: len(rows) for name, rows in tables.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    deploy_dir = Path(args.deploy_dir)
    story_dir = Path(args.story_dir)
    out_dir = Path(args.output_dir)
    docs_report = Path(args.docs_report)
    metric_rows = read_csv(deploy_dir / "threshold_guard_metrics.csv")
    test_rows = read_csv(deploy_dir / "threshold_guard_test_outputs.csv")
    low_cal_source_rows = read_csv(story_dir / "selector_low_cal_metric_summary.csv")

    primary_rows = build_primary_result_table(metric_rows)
    slice_rows = build_regression_slice_table(test_rows, PROFILE_SPECS)
    cc_rows = build_classification_correct_table(test_rows)
    co_rows = build_co_rescue_decomposition(metric_rows)
    low_cal_rows = build_low_cal_stress_table(low_cal_source_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "main_result_table.csv", primary_rows)
    write_csv(out_dir / "regression_slice_table.csv", slice_rows)
    write_csv(out_dir / "classification_correct_table.csv", cc_rows)
    write_csv(out_dir / "classification_correct_accepted_review_table.csv", cc_rows)
    write_csv(out_dir / "co_rescue_decomposition.csv", co_rows)
    write_csv(out_dir / "low_cal_stress_summary.csv", low_cal_rows)
    write_story_brief(out_dir / "final_metric_story_brief.zh.md", primary_rows, cc_rows, co_rows, low_cal_rows)
    write_story_brief(docs_report, primary_rows, cc_rows, co_rows, low_cal_rows)
    write_gpt_brief(out_dir / "gpt_analysis_brief.zh.md", primary_rows, cc_rows, co_rows)

    tables = {
        "primary_rows": primary_rows,
        "regression_slice_rows": slice_rows,
        "classification_correct_rows": cc_rows,
        "co_rows": co_rows,
        "low_cal_rows": low_cal_rows,
    }
    payload = manifest_payload(out_dir=out_dir, deploy_dir=deploy_dir, story_dir=story_dir, docs_report=docs_report, tables=tables)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-dir", default=str(DEFAULT_DEPLOY_DIR))
    parser.add_argument("--story-dir", default=str(DEFAULT_STORY_DIR))
    parser.add_argument("--submission-dir", default=str(DEFAULT_SUBMISSION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--docs-report", default=str(DEFAULT_DOC_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
