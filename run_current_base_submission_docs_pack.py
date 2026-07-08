"""Generate paper-method, system-index, and archive-plan docs.

This script is a documentation pack builder. It reads frozen current-base
evidence and teacher-briefing artifacts, then writes reproducible Chinese
drafts for paper writing and workspace organization. It does not train models,
recompute predictions, delete files, or move files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import fnum, read_csv, write_csv


DEFAULT_FREEZE_DIR = Path("results/current_base_evidence_freeze_20260708")
DEFAULT_METHOD_DIR = Path("results/current_base_method_story_20260708")
DEFAULT_TEACHER_DIR = Path("results/current_base_teacher_briefing_pack_20260708")
DEFAULT_OUTPUT_DIR = Path("results/current_base_submission_docs_pack_20260708")
DEFAULT_METHOD_DOC = Path("docs/superpowers/reports/2026-07-08-current-base-paper-method-chapter-draft.zh.md")
DEFAULT_SYSTEM_INDEX_DOC = Path("docs/superpowers/reports/2026-07-08-current-base-system-doc-index.zh.md")
DEFAULT_ARCHIVE_PLAN_DOC = Path("docs/superpowers/reports/2026-07-08-current-base-intermediate-archive-plan.zh.md")


CORE_KEEP_PATHS = {
    "export_real_route_threshold_guard_deployment_candidate.py",
    "run_current_base_evidence_freeze.py",
    "run_current_base_method_story.py",
    "run_current_base_paper_story_pack.py",
    "run_current_base_teacher_briefing_pack.py",
    "run_current_base_submission_docs_pack.py",
    "run_light_route_gap_appendix.py",
    "run_real_route_c5_rescue_audit.py",
    "run_real_route_c5_selector_validation.py",
    "run_real_route_selector_low_cal_stress.py",
    "run_route_gap_audit.py",
    "tests/test_current_base_evidence_freeze.py",
    "tests/test_current_base_method_story.py",
    "tests/test_current_base_paper_story_pack.py",
    "tests/test_current_base_teacher_briefing_pack.py",
    "tests/test_current_base_submission_docs_pack.py",
    "tests/test_light_route_gap_appendix.py",
    "tests/test_real_route_c5_rescue_audit.py",
    "tests/test_real_route_c5_selector_validation.py",
    "tests/test_real_route_selector_low_cal_stress.py",
    "tests/test_real_route_threshold_guard_deployment_candidate.py",
    "tests/test_route_gap_audit.py",
    "results/current_base_evidence_freeze_20260708",
    "results/current_base_method_story_20260708",
    "results/current_base_paper_story_pack_20260705",
    "results/current_base_teacher_briefing_pack_20260708",
    "results/current_base_submission_docs_pack_20260708",
    "results/light_route_gap_appendix_20260708",
    "results/real_route_c5_rescue_audit_20260705",
    "results/real_route_c5_selector_validation_20260705",
    "results/real_route_selector_low_cal_stress_20260706",
    "results/real_route_threshold_guard_deployment_candidate_20260707",
    "docs/superpowers/reports/2026-07-05-current-base-paper-story-pack.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-evidence-freeze.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-method-story.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-system-docs.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-teacher-briefing.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-paper-method-chapter-draft.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-system-doc-index.zh.md",
    "docs/superpowers/reports/2026-07-08-current-base-intermediate-archive-plan.zh.md",
}


def norm_path(path: Path) -> str:
    return path.as_posix()


def metric_by_id(rows: Sequence[dict[str, Any]], metric_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("metric_id") == metric_id:
            return row
    return {}


def markdown_table(rows: Sequence[dict[str, Any]], headers: Sequence[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _key in headers) + " |",
        "| " + " | ".join("---" for _title, _key in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _title, key in headers) + " |")
    return lines


def threshold_text(params: Sequence[dict[str, str]]) -> str:
    by_name = {row["name"]: row["value"] for row in params}
    return " / ".join(f"{name}={by_name.get(name, '')}" for name in ["tau_C3", "tau_C4", "tau_C5"])


def build_method_chapter_sections(
    headline_rows: Sequence[dict[str, Any]],
    modules: Sequence[dict[str, str]],
    params: Sequence[dict[str, str]],
    sequence_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    p4_all = metric_by_id(headline_rows, "P4_ALL")
    p4_c5 = metric_by_id(headline_rows, "P4_C5")
    module_names = "、".join(row["module"] for row in modules[:6])
    figure_table_ids = "、".join(row["item_id"] for row in sequence_rows)
    taus = threshold_text(params)
    return [
        {
            "section_id": "S1",
            "title": "问题定义：真实 route 下的目标域回归",
            "role": "problem framing",
            "core_text": "目标域 C3/C4/C5 的气体浓度回归不是单纯换一个回归头的问题；真实部署时分类 route 可能含噪，CO 与 nonCO 的误差结构也不同，因此需要把分类上下文、目标域校准和回归输出选择放在同一个系统里报告。",
            "evidence_refs": "T1, T2",
            "write_note": "开头不要写成模型堆叠竞赛，而要写成部署条件下的 profile calibration 问题。",
        },
        {
            "section_id": "S2",
            "title": "系统总览：F6 real-route 到双回归流",
            "role": "system overview",
            "core_text": f"系统由 {module_names} 组成。F6 分类基座先给出 real-route/risk context，H2.3+ 作为稳健目标域 anchor，H8+C4 作为 CO-priority rescue stream，最后由 per-client threshold guard 做输出选择。",
            "evidence_refs": "F1",
            "write_note": "这里放 F1，并明确 runtime 不能使用 true label、test label 或 oracle route。",
        },
        {
            "section_id": "S3",
            "title": "F6 分类基座：可部署 route context",
            "role": "classification base",
            "core_text": "F6 的作用不是直接改善回归数值，而是在部署时提供 route_class 与风险上下文，决定后续是否允许进入 CO rescue。论文中应强调 real-route 是主线，oracle-route/classification-correct 只用于机制解释或附录对照。",
            "evidence_refs": "F1, T6, F5",
            "write_note": "老师已强调重点报告分类正确下的回归性能，但主线仍要守住 real-route，可把 classification-correct 作为机制补充。",
        },
        {
            "section_id": "S4",
            "title": "H2.3+ 与 H8+C4：anchor-rescue 双流设计",
            "role": "regression streams",
            "core_text": "H2.3+ 是默认输出流，承担 balanced target calibration 与 nonCO safety；H8+C4 不是全量替代，而是针对 CO 高残差区域的 formal rescue stream。这个设计解释了为什么 r3ak16 一类额外回归头当前不应再重复训练：它没有进入 P4 可部署主线，也没有改变 anchor-rescue 的证据结构。",
            "evidence_refs": "T3, F3",
            "write_note": "可以把 r3ak16 放进负结果/效率决策说明，不作为后续主线重复训练对象。",
        },
        {
            "section_id": "S5",
            "title": "Per-client threshold guard：验证集选择的安全门控",
            "role": "selector formula",
            "core_text": f"对客户端 c 的样本 i，记 F6 给出的 route 为 r_i，风险分数为 q_i，验证集选择阈值为 tau_c，则 ŷ_i = H8+C4_i if r_i=CO and q_i>=tau_c; otherwise ŷ_i = H2.3+_i。当前阈值为 {taus}。这个公式把 H8+C4 限制在 CO rescue 条件内，避免 nonCO 被高风险回归头误伤。",
            "evidence_refs": "T4",
            "write_note": "必须写清 tau_c 来自 calibration/validation，不来自 test tuning。",
        },
        {
            "section_id": "S6",
            "title": "QC Accepted+Review：论文主报告切片",
            "role": "reporting protocol",
            "core_text": "Accepted+Review 是当前论文主报告切片，因为它对应系统认为可报告或需人工复核但仍可纳入分析的样本；Reject 保留为风险暴露和部署边界，不作为主性能口径。",
            "evidence_refs": "T2, T3",
            "write_note": "这一段回应老师要求：重点报告分类正确/可报告路径下的回归指标，同时保留 full-set 作为端到端难度背景。",
        },
        {
            "section_id": "S7",
            "title": "低 calibration stress：阈值选择是否稳定",
            "role": "validation stress",
            "core_text": "低 calibration stress test 通过 12/24/48/80 的 per-client validation budget 重采样，检查 threshold selector 在小校准集下是否还能保持正收益。当前 budget=80 时 C3/C4/C5 positive gain rate 达到 100%，可作为稳定性证据。",
            "evidence_refs": "T5, F4",
            "write_note": "12/24/48 写趋势，不要夸大为所有低预算完全稳定。",
        },
        {
            "section_id": "S8",
            "title": "主结果陈述：P4 deployment candidate",
            "role": "main result",
            "core_text": f"P4 threshold guard 在 Accepted+Review ALL 上达到 {format_float(p4_all.get('RMSE'), 3)} / {format_float(p4_all.get('NRMSE'), 4)}，相对 H2.3+ gain={format_float(p4_all.get('rmse_gain_vs_h23'), 3)}，H8 usage={format_float(100 * fnum(p4_all.get('h8_usage_rate')), 1)}%。C5 达到 {format_float(p4_c5.get('RMSE'), 3)} / {format_float(p4_c5.get('NRMSE'), 4)}，gain={format_float(p4_c5.get('rmse_gain_vs_h23'), 3)}，说明 guard 的主要收益来自高难度 C5 CO rescue。",
            "evidence_refs": "T3, F2, F3",
            "write_note": "这段可以作为方法后接实验主结果的桥段。",
        },
        {
            "section_id": "S9",
            "title": "写作边界与后续验证",
            "role": "writing boundary",
            "core_text": f"当前证据边界是 current-base C12->C345，图表顺序为 {figure_table_ids}。P5 route-gap 是附录解释，不替代 real-route 主线。下一步 P6 应扩展不同 source-target 组合，用同一套 T/F 结构复现结论。",
            "evidence_refs": "T6, T7, F5",
            "write_note": "结尾要主动讲边界，避免老师追问时显得是在回避泛化问题。",
        },
    ]


def build_document_index(
    *,
    freeze_dir: Path,
    method_dir: Path,
    teacher_dir: Path,
    output_dir: Path,
) -> list[dict[str, str]]:
    rows = [
        {
            "artifact_id": "FZD",
            "category": "evidence freeze",
            "path": norm_path(freeze_dir / "current_base_evidence_freeze.zh.md"),
            "purpose": "冻结 P1-P5 current-base 证据与检查结果",
            "stage": "Freeze",
            "status": "source",
        },
        {
            "artifact_id": "MET",
            "category": "metrics",
            "path": norm_path(freeze_dir / "frozen_headline_metrics.csv"),
            "purpose": "P4/P3/P5 主指标来源",
            "stage": "Freeze",
            "status": "source",
        },
        {
            "artifact_id": "TCK",
            "category": "paper tables",
            "path": norm_path(freeze_dir / "paper_table_checklist.csv"),
            "purpose": "T1-T7 表格清单",
            "stage": "Freeze",
            "status": "source",
        },
        {
            "artifact_id": "FCK",
            "category": "paper figures",
            "path": norm_path(freeze_dir / "paper_figure_checklist.csv"),
            "purpose": "F1-F5 图清单",
            "stage": "Freeze",
            "status": "source",
        },
        {
            "artifact_id": "MTH",
            "category": "method story",
            "path": norm_path(method_dir / "method_story_blueprint.zh.md"),
            "purpose": "模块-方法-证据叙事蓝图",
            "stage": "Story",
            "status": "source",
        },
        {
            "artifact_id": "SYS",
            "category": "system docs",
            "path": norm_path(method_dir / "system_documentation.zh.md"),
            "purpose": "实验命令、参数、清理建议初版",
            "stage": "Story",
            "status": "source",
        },
        {
            "artifact_id": "TBR",
            "category": "teacher briefing",
            "path": norm_path(teacher_dir / "teacher_briefing.zh.md"),
            "purpose": "给老师看的汇报版",
            "stage": "Briefing",
            "status": "source",
        },
        {
            "artifact_id": "TFS",
            "category": "teacher briefing",
            "path": norm_path(teacher_dir / "table_figure_sequence.csv"),
            "purpose": "T1-T7 / F1-F5 汇报顺序",
            "stage": "Briefing",
            "status": "source",
        },
        {
            "artifact_id": "SLD",
            "category": "teacher briefing",
            "path": norm_path(teacher_dir / "teacher_slide_outline.csv"),
            "purpose": "8 页汇报提纲",
            "stage": "Briefing",
            "status": "source",
        },
    ]
    for figure_id, filename in [
        ("F1", "F1_system_pipeline.svg"),
        ("F2", "F2_threshold_guard_gains.svg"),
        ("F3", "F3_co_nonco_safety.svg"),
        ("F4", "F4_low_cal_stability.svg"),
        ("F5", "F5_route_gap_appendix.svg"),
    ]:
        rows.append(
            {
                "artifact_id": figure_id,
                "category": "paper figures",
                "path": norm_path(teacher_dir / "figures" / filename),
                "purpose": f"{figure_id} 论文/汇报图",
                "stage": "Briefing",
                "status": "source",
            }
        )
    rows.extend(
        [
            {
                "artifact_id": "PMD",
                "category": "paper draft",
                "path": norm_path(output_dir / "paper_method_chapter_draft.zh.md"),
                "purpose": "论文方法章节中文草稿",
                "stage": "SubmissionDocs",
                "status": "generated",
            },
            {
                "artifact_id": "SDX",
                "category": "system index",
                "path": norm_path(output_dir / "system_documentation_index.zh.md"),
                "purpose": "系统文档目录化索引",
                "stage": "SubmissionDocs",
                "status": "generated",
            },
            {
                "artifact_id": "ARC",
                "category": "archive plan",
                "path": norm_path(output_dir / "intermediate_archive_plan.zh.md"),
                "purpose": "中间文件归档建议",
                "stage": "SubmissionDocs",
                "status": "generated",
            },
        ]
    )
    return rows


def classify_path(rel: str, path: Path) -> dict[str, str]:
    if rel in CORE_KEEP_PATHS:
        return {
            "status": "keep_current_core",
            "archive_bucket": "",
            "reason": "当前论文/汇报/复现主线需要保留",
            "recommended_action": "keep",
        }
    if rel.startswith("results/"):
        return {
            "status": "archive_candidate_unreviewed",
            "archive_bucket": "_local_archive_20260708/results_exploratory",
            "reason": "非冻结 current-base 结果目录，可能是探索阶段中间产物",
            "recommended_action": "review_then_move",
        }
    if rel.startswith("docs/superpowers/reports/"):
        return {
            "status": "manual_review_required",
            "archive_bucket": "_local_archive_20260708/docs_superseded",
            "reason": "旧报告可能仍含可追溯讨论，人工确认后再归档",
            "recommended_action": "review_before_move",
        }
    name = path.name
    if name.startswith(("audit_", "diagnose_", "compare_", "summarize_", "plot_", "sweep_")):
        return {
            "status": "archive_candidate_unreviewed",
            "archive_bucket": "_local_archive_20260708/scripts_exploratory",
            "reason": "探索阶段分析/诊断脚本，当前冻结主线未直接依赖",
            "recommended_action": "review_then_move",
        }
    if name.startswith(("run_c5_", "run_co_", "run_time_aware_", "run_target_", "run_weighted_", "run_auto_", "run_guarded_", "run_soft_")):
        return {
            "status": "archive_candidate_unreviewed",
            "archive_bucket": "_local_archive_20260708/scripts_experiment_runners",
            "reason": "候选实验 runner，未进入当前 P1-P5 冻结主线",
            "recommended_action": "review_then_move",
        }
    if rel.startswith("gaps_deploy/"):
        return {
            "status": "manual_review_required",
            "archive_bucket": "_local_archive_20260708/gaps_deploy_review",
            "reason": "部署/迁移相关工具可能仍有复用价值，需人工确认",
            "recommended_action": "review_before_move",
        }
    if rel.startswith("scripts/"):
        return {
            "status": "manual_review_required",
            "archive_bucket": "_local_archive_20260708/scripts_review",
            "reason": "scripts 子目录可能包含数据 split、远程运行或云端实验入口",
            "recommended_action": "review_before_move",
        }
    if rel.startswith("tests/"):
        return {
            "status": "manual_review_required",
            "archive_bucket": "_local_archive_20260708/tests_review",
            "reason": "测试文件不应自动归档，需确认覆盖对象是否废弃",
            "recommended_action": "review_before_move",
        }
    return {
        "status": "manual_review_required",
        "archive_bucket": "_local_archive_20260708/misc_review",
        "reason": "不属于当前冻结主线，但用途不明确",
        "recommended_action": "review_before_move",
    }


def build_archive_inventory(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    candidates: list[Path] = []
    candidates.extend(sorted(root.glob("*.py")))
    candidates.extend(sorted(root.glob("*.ps1")))
    candidates.extend(sorted(root.glob("*.cmd")))
    candidates.extend(sorted((root / "scripts").glob("*")) if (root / "scripts").exists() else [])
    candidates.extend(sorted((root / "gaps_deploy").glob("*.py")) if (root / "gaps_deploy").exists() else [])
    candidates.extend(sorted((root / "tests").glob("test_*.py")) if (root / "tests").exists() else [])
    candidates.extend(sorted((root / "results").iterdir()) if (root / "results").exists() else [])
    reports_dir = root / "docs" / "superpowers" / "reports"
    candidates.extend(sorted(reports_dir.glob("*.md")) if reports_dir.exists() else [])

    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        rel = norm_path(path.relative_to(root))
        if rel in seen:
            continue
        seen.add(rel)
        kind = "result_dir" if path.is_dir() and rel.startswith("results/") else "dir" if path.is_dir() else path.suffix.lstrip(".") or "file"
        classification = classify_path(rel, path)
        rows.append({"path": rel, "kind": kind, **classification})
    return sorted(rows, key=lambda row: (row["status"], row["path"]))


def summarize_archive_inventory(rows: Sequence[dict[str, str]]) -> list[dict[str, int]]:
    counts = Counter(row["status"] for row in rows)
    return [{"status": status, "count": counts[status]} for status in sorted(counts)]


def write_method_chapter(
    path: Path,
    *,
    sections: Sequence[dict[str, str]],
    modules: Sequence[dict[str, str]],
    params: Sequence[dict[str, str]],
    sequence_rows: Sequence[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 论文方法章节草稿",
        "",
        "> 面向当前基座 C12->C345 的论文方法章节初稿。可作为第 3/4 章的中文底稿，后续再翻译/压缩成正式论文语气。",
        "",
        "## 章节主线",
        "",
        "本文方法部分建议围绕一个核心句展开：在真实部署 route 下，目标域气体回归需要以 F6 分类基座提供的 route/risk context 为入口，以 H2.3+ 作为稳健 target calibration anchor，以 H8+C4 作为 CO-priority rescue stream，再通过 validation-selected per-client threshold guard 选择最终输出。",
        "",
        "## 方法公式",
        "",
        "对客户端 `c` 的样本 `i`，记 F6 输出的 route 为 `r_i`，风险分数为 `q_i`，客户端阈值为 `tau_c`：",
        "",
        "```text",
        "ŷ_i = H8+C4_i,  if r_i = CO and q_i >= tau_c",
        "ŷ_i = H2.3+_i, otherwise",
        "```",
        "",
        f"当前阈值：{threshold_text(params)}。阈值只来自 calibration/validation，test 仅用于最终 audit。",
        "",
        "## 可直接改写的章节段落",
        "",
    ]
    for row in sections:
        lines.extend(
            [
                f"### {row['section_id']} {row['title']}",
                "",
                row["core_text"],
                "",
                f"证据位置：{row['evidence_refs']}。",
                "",
                f"写作提醒：{row['write_note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 模块-方法对应表",
            "",
            *markdown_table(
                modules,
                [
                    ("id", "module_id"),
                    ("module", "module"),
                    ("principle", "method_principle"),
                    ("evidence", "evidence"),
                    ("paper section", "paper_section"),
                ],
            ),
            "",
            "## T/F 使用位置",
            "",
            *markdown_table(
                sequence_rows,
                [
                    ("item", "item_id"),
                    ("kind", "kind"),
                    ("title", "title"),
                    ("source", "source"),
                    ("placement", "placement"),
                ],
            ),
            "",
            "## 方法章节边界",
            "",
            "- real-route 是主线，oracle-route/classification-correct 只作为机制解释或附录。",
            "- H8+C4 是受控 CO rescue，不是全量替换 H2.3+。",
            "- r3ak16 当前不进入主线复训，除非后续 P6 发现新的 source-target 组合需要额外候选。",
            "- P5 route-gap 用来解释 full-set 难度，不替代 Accepted+Review 主结果。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_system_index(
    path: Path,
    *,
    document_rows: Sequence[dict[str, str]],
    commands: Sequence[dict[str, str]],
    params: Sequence[dict[str, str]],
    sequence_rows: Sequence[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 系统文档目录化索引",
        "",
        "## 读文档顺序",
        "",
        "建议按 `Evidence Freeze -> Method Story -> Teacher Briefing -> Submission Docs` 的顺序阅读。这样先看到冻结指标，再看到方法叙事，最后进入论文草稿和归档清单。",
        "",
        "## T1-T7 / F1-F5 资产顺序",
        "",
        *markdown_table(
            sequence_rows,
            [
                ("item", "item_id"),
                ("kind", "kind"),
                ("title", "title"),
                ("source", "source"),
                ("placement", "placement"),
            ],
        ),
        "",
        "## 文档与资产索引",
        "",
        *markdown_table(
            document_rows,
            [
                ("id", "artifact_id"),
                ("category", "category"),
                ("path", "path"),
                ("purpose", "purpose"),
                ("stage", "stage"),
                ("status", "status"),
            ],
        ),
        "",
        "## 实验命令索引",
        "",
        *markdown_table(
            commands,
            [
                ("stage", "stage"),
                ("purpose", "purpose"),
                ("command", "command"),
                ("outputs", "key_outputs"),
                ("notes", "notes"),
            ],
        ),
        "",
        "## 参数索引",
        "",
        *markdown_table(params, [("name", "name"), ("value", "value"), ("used by", "used_by"), ("meaning", "meaning")]),
        "",
        "## 后续维护规则",
        "",
        "- 新实验先写入独立 `results/<name>_<date>/`，再进入 freeze/story/briefing。",
        "- 能进入论文主线的结果必须有 manifest、CSV 指标、报告 Markdown 和测试覆盖。",
        "- 文档中的主指标以 `frozen_headline_metrics.csv` 为准，不直接从零散日志手动摘数。",
        "- 清理中间文件时只移动、不删除，并保留归档 README。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_archive_plan(
    path: Path,
    *,
    archive_rows: Sequence[dict[str, str]],
    summary_rows: Sequence[dict[str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 中间文件归档清单",
        "",
        "> 这是归档建议，不是自动清理脚本。原则是只移动、不删除；每个 `archive_candidate_unreviewed` 都需要人工确认后再移动。",
        "",
        "## 状态汇总",
        "",
        *markdown_table(summary_rows, [("status", "status"), ("count", "count")]),
        "",
        "## 推荐处理动作",
        "",
        "| status | meaning | action |",
        "| --- | --- | --- |",
        "| keep_current_core | 当前论文、汇报、复现主线依赖 | 保留在原位 |",
        "| archive_candidate_unreviewed | 大概率是探索阶段中间产物 | 人工确认后移动到建议 bucket |",
        "| manual_review_required | 可能还有工具或历史价值 | 先看依赖和最近用途，再决定是否归档 |",
        "",
        "## 建议归档步骤",
        "",
        "```powershell",
        "New-Item -ItemType Directory -Force _local_archive_20260708",
        "# 示例：确认某个文件不再使用后，再执行移动",
        "# Move-Item -LiteralPath 'diagnose_old_probe.py' -Destination '_local_archive_20260708/scripts_exploratory/'",
        "```",
        "",
        "执行移动前建议重新跑：",
        "",
        "```powershell",
        "python -m pytest tests/test_current_base_submission_docs_pack.py tests/test_current_base_teacher_briefing_pack.py tests/test_current_base_method_story.py tests/test_current_base_evidence_freeze.py -q",
        "```",
        "",
        "## 逐项清单",
        "",
        *markdown_table(
            archive_rows,
            [
                ("path", "path"),
                ("kind", "kind"),
                ("status", "status"),
                ("bucket", "archive_bucket"),
                ("reason", "reason"),
                ("action", "recommended_action"),
            ],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extend_commands(commands: list[dict[str, str]]) -> list[dict[str, str]]:
    existing = {row["stage"] for row in commands}
    out = list(commands)
    if "Briefing" not in existing:
        out.append(
            {
                "stage": "Briefing",
                "purpose": "Build F1-F5 and teacher-facing briefing package",
                "command": "python run_current_base_teacher_briefing_pack.py",
                "key_outputs": "results/current_base_teacher_briefing_pack_20260708/teacher_briefing.zh.md",
                "notes": "Consumes frozen evidence; no training.",
            }
        )
    if "SubmissionDocs" not in existing:
        out.append(
            {
                "stage": "SubmissionDocs",
                "purpose": "Build paper method draft, system index, and archive plan",
                "command": "python run_current_base_submission_docs_pack.py",
                "key_outputs": "results/current_base_submission_docs_pack_20260708/",
                "notes": "Documentation pack only; no file moves.",
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    freeze_dir = Path(args.freeze_dir)
    method_dir = Path(args.method_dir)
    teacher_dir = Path(args.teacher_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headline_rows = read_csv(freeze_dir / "frozen_headline_metrics.csv")
    modules = read_csv(method_dir / "module_method_mapping.csv")
    commands = extend_commands(read_csv(method_dir / "experiment_command_registry.csv"))
    params = read_csv(method_dir / "parameter_reference.csv")
    sequence_rows = read_csv(teacher_dir / "table_figure_sequence.csv")

    sections = build_method_chapter_sections(headline_rows, modules, params, sequence_rows)
    document_rows = build_document_index(freeze_dir=freeze_dir, method_dir=method_dir, teacher_dir=teacher_dir, output_dir=out_dir)
    archive_rows = build_archive_inventory(Path(args.workspace_root))
    summary_rows = summarize_archive_inventory(archive_rows)

    write_csv(out_dir / "paper_method_chapter_sections.csv", sections)
    write_csv(out_dir / "system_document_index.csv", document_rows)
    write_csv(out_dir / "intermediate_archive_inventory.csv", archive_rows)
    write_csv(out_dir / "archive_plan_summary.csv", summary_rows)

    method_path = out_dir / "paper_method_chapter_draft.zh.md"
    system_path = out_dir / "system_documentation_index.zh.md"
    archive_path = out_dir / "intermediate_archive_plan.zh.md"
    write_method_chapter(method_path, sections=sections, modules=modules, params=params, sequence_rows=sequence_rows)
    write_system_index(system_path, document_rows=document_rows, commands=commands, params=params, sequence_rows=sequence_rows)
    write_archive_plan(archive_path, archive_rows=archive_rows, summary_rows=summary_rows)

    if args.method_doc:
        write_method_chapter(Path(args.method_doc), sections=sections, modules=modules, params=params, sequence_rows=sequence_rows)
    if args.system_index_doc:
        write_system_index(Path(args.system_index_doc), document_rows=document_rows, commands=commands, params=params, sequence_rows=sequence_rows)
    if args.archive_plan_doc:
        write_archive_plan(Path(args.archive_plan_doc), archive_rows=archive_rows, summary_rows=summary_rows)

    manifest = {
        "freeze_dir": str(freeze_dir),
        "method_dir": str(method_dir),
        "teacher_dir": str(teacher_dir),
        "outputs": [
            "paper_method_chapter_sections.csv",
            "system_document_index.csv",
            "intermediate_archive_inventory.csv",
            "archive_plan_summary.csv",
            "paper_method_chapter_draft.zh.md",
            "system_documentation_index.zh.md",
            "intermediate_archive_plan.zh.md",
        ],
        "docs_reports": {
            "method_doc": args.method_doc,
            "system_index_doc": args.system_index_doc,
            "archive_plan_doc": args.archive_plan_doc,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "method_chapter": str(method_path),
                "system_index": str(system_path),
                "archive_plan": str(archive_path),
                "archive_summary": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--method-dir", default=str(DEFAULT_METHOD_DIR))
    parser.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--method-doc", default=str(DEFAULT_METHOD_DOC))
    parser.add_argument("--system-index-doc", default=str(DEFAULT_SYSTEM_INDEX_DOC))
    parser.add_argument("--archive-plan-doc", default=str(DEFAULT_ARCHIVE_PLAN_DOC))
    parser.add_argument("--workspace-root", default=".")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
