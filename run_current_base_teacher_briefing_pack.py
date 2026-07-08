"""Build teacher-facing briefing materials for the current-base story.

Outputs:
- F1-F5 SVG figures.
- A teacher-facing Chinese briefing markdown.
- Table/figure sequence and slide outline CSVs.

The script only reads frozen evidence and method-story artifacts. It does not
train models or recompute predictions.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import fnum, read_csv, write_csv


DEFAULT_FREEZE_DIR = Path("results/current_base_evidence_freeze_20260708")
DEFAULT_METHOD_DIR = Path("results/current_base_method_story_20260708")
DEFAULT_OUTPUT_DIR = Path("results/current_base_teacher_briefing_pack_20260708")
DEFAULT_DOCS_REPORT = Path("docs/superpowers/reports/2026-07-08-current-base-teacher-briefing.zh.md")


def metric_by_id(rows: Sequence[dict[str, Any]], metric_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("metric_id") == metric_id:
            return row
    return {}


def svg_text(x: float, y: float, text: str, *, size: int = 15, weight: str = "400", fill: str = "#1f2937", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}" font-family="Arial, Microsoft YaHei, sans-serif">{html.escape(text)}</text>'


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#334155", radius: int = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'


def svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#475569") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#arrow)"/>'


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#475569"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_svg(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_f1_system_pipeline(path: Path, policy: dict[str, Any]) -> Path:
    thresholds = policy.get("per_client_thresholds", {})
    threshold_text = " / ".join(
        f"{client}: {thresholds.get(client, {}).get('threshold_label', '')}"
        for client in ["C3", "C4", "C5"]
    )
    lines = svg_header(1320, 760)
    lines.extend(
        [
            svg_text(660, 46, "F1 System Pipeline: FCL Backbone Freeze -> Guarded CO Rescue", size=24, weight="700"),
            svg_text(660, 78, "主线：长期部署/FCL 底座冻结分类 route provider；当前 F6 route 接 H2.3+ anchor + H8+C4 rescue + threshold guard", size=15, fill="#475569"),
            svg_rect(70, 160, 180, 92, "#e0f2fe", "#0284c7"),
            svg_text(160, 190, "Target samples", size=16, weight="700"),
            svg_text(160, 216, "C3 / C4 / C5", size=14),
            svg_text(160, 238, "sensor responses", size=13, fill="#475569"),
            svg_arrow(250, 206, 335, 206),
            svg_rect(335, 150, 210, 112, "#eef2ff", "#4f46e5"),
            svg_text(440, 182, "F6 real-route", size=16, weight="700"),
            svg_text(440, 208, "classification base", size=16, weight="700"),
            svg_text(440, 234, "route_class / risk", size=13, fill="#475569"),
            svg_arrow(545, 206, 635, 128),
            svg_arrow(545, 206, 635, 306),
            svg_rect(635, 88, 255, 110, "#dcfce7", "#16a34a"),
            svg_text(762, 122, "H2.3+ target profile", size=16, weight="700"),
            svg_text(762, 148, "default anchor", size=14),
            svg_text(762, 174, "balanced calibration / nonCO safety", size=13, fill="#475569"),
            svg_rect(635, 266, 255, 110, "#fee2e2", "#dc2626"),
            svg_text(762, 300, "H8+C4 formal rescue", size=16, weight="700"),
            svg_text(762, 326, "CO-priority candidate", size=14),
            svg_text(762, 352, "rescues C5 CO high residuals", size=13, fill="#475569"),
            svg_arrow(890, 143, 980, 245),
            svg_arrow(890, 321, 980, 268),
            svg_rect(980, 196, 260, 142, "#fef3c7", "#d97706"),
            svg_text(1110, 228, "Per-client threshold guard", size=16, weight="700"),
            svg_text(1110, 256, "if route=CO and risk >= tau_c", size=14),
            svg_text(1110, 282, "select H8+C4; else H2.3+", size=14),
            svg_text(1110, 312, threshold_text, size=12, fill="#475569"),
            svg_arrow(1110, 338, 1110, 430),
            svg_rect(945, 430, 330, 116, "#f8fafc", "#64748b"),
            svg_text(1110, 462, "QC Accepted+Review reporting", size=16, weight="700"),
            svg_text(1110, 490, "primary paper metric", size=14),
            svg_text(1110, 518, "reject rows remain risk exposure", size=13, fill="#475569"),
            svg_rect(560, 505, 350, 120, "#f5f3ff", "#7c3aed"),
            svg_text(735, 538, "Validation-only threshold selection", size=16, weight="700"),
            svg_text(735, 566, "calibration predictions choose tau_c", size=14),
            svg_text(735, 594, "test labels only for final audit", size=13, fill="#475569"),
            svg_arrow(735, 505, 1010, 338, "#7c3aed"),
            svg_text(660, 706, "Runtime excludes true labels, test labels, oracle route, and oracle-best selector.", size=14, weight="700", fill="#334155"),
            svg_text(660, 732, "Backbone freeze principle: do not retrain classifier for local regression rescue; update target regression profiles and guards instead.", size=13, fill="#475569"),
            "</svg>",
        ]
    )
    return write_svg(path, lines)


def write_bar_chart(path: Path, title: str, rows: Sequence[tuple[str, float, str]], *, y_label: str, fill: str = "#2563eb") -> Path:
    width, height = 980, 520
    left, top, plot_w, plot_h = 90, 95, 820, 330
    max_val = max([value for _label, value, _note in rows] + [1.0])
    lines = svg_header(width, height)
    lines.append(svg_text(width / 2, 45, title, size=22, weight="700"))
    lines.append(svg_text(30, top + plot_h / 2, y_label, size=13, fill="#475569", anchor="start"))
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#64748b" stroke-width="1.5"/>')
    bar_w = plot_w / (len(rows) * 1.55)
    gap = bar_w * 0.55
    for idx, (label, value, note) in enumerate(rows):
        x = left + idx * (bar_w + gap) + gap
        h = plot_h * value / max_val
        y = top + plot_h - h
        lines.append(svg_rect(x, y, bar_w, h, fill, "#1e40af", radius=4))
        lines.append(svg_text(x + bar_w / 2, y - 10, format_float(value, 3), size=13, weight="700"))
        lines.append(svg_text(x + bar_w / 2, top + plot_h + 28, label, size=14, weight="700"))
        if note:
            lines.append(svg_text(x + bar_w / 2, top + plot_h + 50, note, size=12, fill="#475569"))
    lines.append("</svg>")
    return write_svg(path, lines)


def write_f2_gain_chart(path: Path, headline_rows: Sequence[dict[str, Any]]) -> Path:
    rows = []
    for scope in ["ALL", "C3", "C4", "C5"]:
        item = metric_by_id(headline_rows, f"P4_{scope}")
        rows.append((scope, fnum(item.get("rmse_gain_vs_h23")), f"RMSE {format_float(item.get('RMSE'), 2)}"))
    return write_bar_chart(path, "F2 P4 Threshold Guard Gains vs H2.3+ (Accepted+Review)", rows, y_label="RMSE gain", fill="#2563eb")


def write_f3_safety_chart(path: Path, headline_rows: Sequence[dict[str, Any]]) -> Path:
    width, height = 1060, 560
    lines = svg_header(width, height)
    lines.append(svg_text(width / 2, 44, "F3 CO Rescue and nonCO Safety", size=22, weight="700"))
    lines.append(svg_text(width / 2, 72, "CO gains are positive; nonCO H8 usage remains 0%", size=14, fill="#475569"))
    rows = []
    for client in ["C3", "C4", "C5"]:
        co = metric_by_id(headline_rows, f"P4_{client}-CO")
        nonco = metric_by_id(headline_rows, f"P4_{client}-nonCO")
        rows.append((client, fnum(co.get("rmse_gain_vs_h23")), fnum(nonco.get("h8_usage_rate"))))
    left, top, plot_w, plot_h = 110, 120, 820, 310
    max_gain = max(row[1] for row in rows)
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#64748b"/>')
    group_w = plot_w / len(rows)
    for idx, (client, gain, usage) in enumerate(rows):
        x0 = left + idx * group_w + 40
        gain_h = plot_h * gain / max_gain
        usage_h = plot_h * usage
        lines.append(svg_rect(x0, top + plot_h - gain_h, 80, gain_h, "#dc2626", "#991b1b", radius=4))
        lines.append(svg_rect(x0 + 105, top + plot_h - usage_h, 80, usage_h, "#0f766e", "#115e59", radius=4))
        lines.append(svg_text(x0 + 40, top + plot_h - gain_h - 10, format_float(gain, 2), size=13, weight="700"))
        lines.append(svg_text(x0 + 145, top + plot_h - usage_h - 10, f"{format_float(100 * usage, 1)}%", size=13, weight="700"))
        lines.append(svg_text(x0 + 92, top + plot_h + 30, client, size=15, weight="700"))
    lines.append(svg_rect(740, 92, 18, 18, "#dc2626", "#991b1b", radius=2))
    lines.append(svg_text(765, 106, "CO RMSE gain", size=13, anchor="start"))
    lines.append(svg_rect(740, 120, 18, 18, "#0f766e", "#115e59", radius=2))
    lines.append(svg_text(765, 134, "nonCO H8 usage", size=13, anchor="start"))
    lines.append("</svg>")
    return write_svg(path, lines)


def write_f4_low_cal_chart(path: Path, low_cal_rows: Sequence[dict[str, Any]]) -> Path:
    width, height = 1060, 560
    left, top, plot_w, plot_h = 95, 105, 850, 330
    budgets = [12, 24, 48, 80]
    scopes = ["ALL", "C3", "C4", "C5"]
    colors = {"ALL": "#111827", "C3": "#2563eb", "C4": "#16a34a", "C5": "#dc2626"}
    data: dict[str, list[float]] = {scope: [] for scope in scopes}
    for scope in scopes:
        for budget in budgets:
            row = next(
                (
                    item for item in low_cal_rows
                    if str(item.get("budget_per_client")) == str(budget)
                    and item.get("scope") == scope
                    and item.get("qc_slice") == "accepted_review"
                ),
                {},
            )
            data[scope].append(fnum(row.get("rmse_gain_vs_h23_mean")))
    max_val = max(max(values) for values in data.values())
    lines = svg_header(width, height)
    lines.append(svg_text(width / 2, 44, "F4 Low-calibration Stress: Gain vs Budget", size=22, weight="700"))
    lines.append(svg_text(width / 2, 72, "Repeated validation resampling, fixed real-route test evaluation", size=14, fill="#475569"))
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#64748b"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#64748b"/>')
    x_positions = [left + idx * plot_w / (len(budgets) - 1) for idx in range(len(budgets))]
    for x, budget in zip(x_positions, budgets):
        lines.append(svg_text(x, top + plot_h + 30, str(budget), size=13, weight="700"))
    for scope in scopes:
        points = []
        for idx, gain in enumerate(data[scope]):
            x = x_positions[idx]
            y = top + plot_h - plot_h * gain / max_val
            points.append((x, y))
        path_d = " ".join(("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}" for idx, (x, y) in enumerate(points))
        lines.append(f'<path d="{path_d}" fill="none" stroke="{colors[scope]}" stroke-width="3"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[scope]}"/>')
        lx, ly = points[-1]
        lines.append(svg_text(lx + 16, ly + 4, scope, size=13, weight="700", fill=colors[scope], anchor="start"))
    lines.append(svg_text(width / 2, 505, "budget per client", size=14, fill="#475569"))
    lines.append("</svg>")
    return write_svg(path, lines)


def write_f5_gap_chart(path: Path, gap_rows: Sequence[dict[str, Any]]) -> Path:
    rows = []
    for row in gap_rows:
        rows.append((f"{row.get('profile_family')} {row.get('scope')}", fnum(row.get("gap_full_RMSE")), f"{format_float(100 * fnum(row.get('gap_full_RMSE_pct_of_real')), 1)}%"))
    return write_bar_chart(path, "F5 Light Route-gap Appendix: Full-set Gap", rows, y_label="Gap RMSE", fill="#7c3aed")


def build_table_figure_sequence(table_rows: Sequence[dict[str, Any]], figure_rows: Sequence[dict[str, Any]], figure_paths: dict[str, str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    sequence = [
        ("F1", "opening_system_story", "先讲系统为什么这样分层：real-route -> 双回归流 -> threshold guard -> QC report。"),
        ("T3", "main_result", "先给 P4 主结果，说明这是当前最可提交结果。"),
        ("F2", "main_result_visual", "用 per-client gain 图让老师快速看到 ALL/C3/C4/C5 都有收益。"),
        ("F3", "safety", "说明 CO rescue 与 nonCO safety 同时成立。"),
        ("T4", "threshold_provenance", "强调 tau 来自 validation/calibration，不是 test tuning。"),
        ("F4", "stability", "解释低 calibration 下 selector 选择过程的稳定性。"),
        ("T1", "full_context", "full-set 作为实际端到端难度背景。"),
        ("T2", "qc_context", "Accepted+Review 是论文主报告切片。"),
        ("T5", "stress_table", "低 calibration stress 的表格证据。"),
        ("T6", "route_gap_appendix", "route-gap 只作 appendix，不抢主线。"),
        ("F5", "route_gap_visual", "说明 C5 full-set gap 的分类/route-noise 背景。"),
        ("T7", "claim_evidence", "最后用 claim-evidence matrix 收束贡献和边界。"),
    ]
    table_by_id = {row["table_id"]: row for row in table_rows}
    figure_by_id = {row["figure_id"]: row for row in figure_rows}
    for order, (item_id, placement, talk_track) in enumerate(sequence, start=1):
        if item_id.startswith("T"):
            src = table_by_id.get(item_id, {})
            out.append(
                {
                    "order": str(order),
                    "item_id": item_id,
                    "kind": "table",
                    "title": str(src.get("title", "")),
                    "source": str(src.get("source", "")),
                    "placement": placement,
                    "talk_track": talk_track,
                }
            )
        else:
            src = figure_by_id.get(item_id, {})
            out.append(
                {
                    "order": str(order),
                    "item_id": item_id,
                    "kind": "figure",
                    "title": str(src.get("title", "")),
                    "source": figure_paths.get(item_id, str(src.get("source", ""))),
                    "placement": placement,
                    "talk_track": talk_track,
                }
            )
    return out


def build_slide_outline() -> list[dict[str, str]]:
    return [
        {"slide": "1", "title": "问题与一句话主线", "content": "真实部署 route 下，目标域回归需要 profile calibration + guarded rescue。", "assets": "K1/K2 claims"},
        {"slide": "2", "title": "系统主线图", "content": "FCL/backbone freeze -> F6 real-route -> H2.3+ anchor / H8+C4 rescue -> threshold guard -> QC Accepted+Review。", "assets": "F1"},
        {"slide": "3", "title": "主结果", "content": "P4 Accepted+Review ALL/C3/C4/C5 全部优于 H2.3+。", "assets": "T3 + F2"},
        {"slide": "4", "title": "为什么 guard 安全", "content": "CO 上允许 rescue，nonCO H8 usage=0。", "assets": "F3"},
        {"slide": "5", "title": "阈值来源与无泄漏", "content": "tau_C3/C4/C5 由 validation/calibration 选择，test 只做最终审计。", "assets": "T4"},
        {"slide": "6", "title": "低 calibration 稳定性", "content": "预算 12/24/48/80 重采样；budget=80 时 C3/C4/C5 positive gain rate=100%。", "assets": "T5 + F4"},
        {"slide": "7", "title": "route-gap 作为附录解释", "content": "C5 full-set gap 最大，解释 route-noise 背景，不替代 real-route 主线。", "assets": "T6 + F5"},
        {"slide": "8", "title": "贡献、边界与下一步", "content": "贡献 K1-K5；边界是 current-base C12->C345；下一步 P6 跨 source-target 验证。", "assets": "T7"},
    ]


def p4_metric_table(headline_rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for scope in ["ALL", "C3", "C4", "C5"]:
        item = metric_by_id(headline_rows, f"P4_{scope}")
        rows.append(
            {
                "scope": scope,
                "RMSE/NRMSE": f"{format_float(item.get('RMSE'), 3)} / {format_float(item.get('NRMSE'), 4)}",
                "H2.3+ RMSE": format_float(item.get("baseline_h23_RMSE"), 3),
                "H8-all RMSE": format_float(item.get("h8_all_RMSE"), 3),
                "gain": format_float(item.get("rmse_gain_vs_h23"), 3),
                "H8 usage": f"{format_float(100 * fnum(item.get('h8_usage_rate')), 1)}%",
            }
        )
    return rows


def markdown_table(rows: Sequence[dict[str, Any]], headers: Sequence[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _key in headers) + " |",
        "| " + " | ".join("---" for _title, _key in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _title, key in headers) + " |")
    return lines


def write_teacher_briefing(
    path: Path,
    *,
    headline_rows: Sequence[dict[str, Any]],
    sequence_rows: Sequence[dict[str, str]],
    slide_rows: Sequence[dict[str, str]],
    figure_paths: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p4_all = metric_by_id(headline_rows, "P4_ALL")
    lines = [
        "# 当前基座给老师汇报版",
        "",
        "## 汇报主线",
        "",
        "这次汇报不按“尝试过哪些模型”展开，而按系统逻辑展开：长期部署/FCL 底座先给出分类基座冻结原则；当前主线继承这个原则，用最新 F6 分类基座提供 route 与 risk context；H2.3+ 作为稳健目标域 calibration anchor；H8+C4 作为 CO-priority rescue stream；最后用 validation-selected per-client threshold guard 安全选择输出。",
        "",
        "## 先讲的主结果",
        "",
        f"P4 threshold guard 在 Accepted+Review ALL 上达到 {format_float(p4_all.get('RMSE'), 3)} / {format_float(p4_all.get('NRMSE'), 4)}，相对 H2.3+ gain={format_float(p4_all.get('rmse_gain_vs_h23'), 3)}，H8 usage={format_float(100 * fnum(p4_all.get('h8_usage_rate')), 1)}%。",
        "",
        *markdown_table(p4_metric_table(headline_rows), [("scope", "scope"), ("RMSE/NRMSE", "RMSE/NRMSE"), ("H2.3+", "H2.3+ RMSE"), ("H8-all", "H8-all RMSE"), ("gain", "gain"), ("H8 usage", "H8 usage")]),
        "",
        "## F1-F5 图表资产",
        "",
        *markdown_table([{"figure": key, "path": value} for key, value in figure_paths.items()], [("figure", "figure"), ("path", "path")]),
        "",
        "## T1-T7 / F1-F5 汇报顺序",
        "",
        *markdown_table(sequence_rows, [("order", "order"), ("id", "item_id"), ("kind", "kind"), ("title", "title"), ("source", "source"), ("talk track", "talk_track")]),
        "",
        "## 8 页汇报提纲",
        "",
        *markdown_table(slide_rows, [("slide", "slide"), ("title", "title"), ("content", "content"), ("assets", "assets")]),
        "",
        "## 老师可能追问的点",
        "",
        "- 为什么不用 oracle-route 当主结果：因为主线是实际部署 real-route；oracle/classification-correct 只作机制解释。",
        "- 为什么 H8+C4 不全量替换：因为 H8 是 CO rescue stream，nonCO 由 H2.3+ 保护，P4 nonCO H8 usage=0。",
        "- threshold 有没有 test 泄漏：没有，tau_C3/C4/C5 来自 validation/calibration；test 只做最终审计。",
        "- 低 calibration 是否稳定：P3 budget=80 达到 C3/C4/C5 positive gain rate=100%，12/24/48 作为趋势证据。",
        "- 下一步是什么：P6 跨 source-target 验证，用同一套 T/F 结构复现。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    freeze_dir = Path(args.freeze_dir)
    method_dir = Path(args.method_dir)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    headline_rows = read_csv(freeze_dir / "frozen_headline_metrics.csv")
    table_rows = read_csv(freeze_dir / "paper_table_checklist.csv")
    figure_rows = read_csv(freeze_dir / "paper_figure_checklist.csv")
    low_cal_rows = read_csv("results/current_base_paper_story_pack_20260705/selector_low_cal_metric_summary.csv")
    gap_rows = read_csv("results/current_base_paper_story_pack_20260705/light_route_gap_appendix_table.csv")
    policy = json.loads(Path("results/real_route_threshold_guard_deployment_candidate_20260707/threshold_guard_policy.json").read_text(encoding="utf-8"))
    figure_paths = {
        "F1": str(write_f1_system_pipeline(fig_dir / "F1_system_pipeline.svg", policy)),
        "F2": str(write_f2_gain_chart(fig_dir / "F2_threshold_guard_gains.svg", headline_rows)),
        "F3": str(write_f3_safety_chart(fig_dir / "F3_co_nonco_safety.svg", headline_rows)),
        "F4": str(write_f4_low_cal_chart(fig_dir / "F4_low_cal_stability.svg", low_cal_rows)),
        "F5": str(write_f5_gap_chart(fig_dir / "F5_route_gap_appendix.svg", gap_rows)),
    }
    sequence_rows = build_table_figure_sequence(table_rows, figure_rows, figure_paths)
    slide_rows = build_slide_outline()
    write_csv(out_dir / "table_figure_sequence.csv", sequence_rows)
    write_csv(out_dir / "teacher_slide_outline.csv", slide_rows)
    report = out_dir / "teacher_briefing.zh.md"
    write_teacher_briefing(report, headline_rows=headline_rows, sequence_rows=sequence_rows, slide_rows=slide_rows, figure_paths=figure_paths)
    if args.docs_report:
        write_teacher_briefing(Path(args.docs_report), headline_rows=headline_rows, sequence_rows=sequence_rows, slide_rows=slide_rows, figure_paths=figure_paths)
    manifest = {
        "freeze_dir": str(freeze_dir),
        "method_dir": str(method_dir),
        "figures": figure_paths,
        "outputs": [
            "figures/F1_system_pipeline.svg",
            "figures/F2_threshold_guard_gains.svg",
            "figures/F3_co_nonco_safety.svg",
            "figures/F4_low_cal_stability.svg",
            "figures/F5_route_gap_appendix.svg",
            "table_figure_sequence.csv",
            "teacher_slide_outline.csv",
            "teacher_briefing.zh.md",
        ],
        "docs_report": args.docs_report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "report": str(report), "figures": figure_paths}, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--method-dir", default=str(DEFAULT_METHOD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--docs-report", default=str(DEFAULT_DOCS_REPORT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
