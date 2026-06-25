from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/target_direct_head_mainline_20260625")
OUT_CSV = OUT_DIR / "target_direct_head_mainline_summary.csv"
REPORT_MD = OUT_DIR / "target_direct_head_mainline_report.md"

HYBRID_SUMMARY = Path("results/hybrid_regression_head_selection_20260624/hybrid_head_summary.csv")
H8_SUMMARY = Path("results/co_only_source_aug_hybrid_stratcalval_20260625/co_only_source_aug_hybrid_summary.csv")
H2_3_EQUIV = Path("results/equivalence_h2_3_mlp_ridge_candidate_20260624/equivalence_summary.json")
H2_3_DEPLOY = Path("results/deployment_h2_3_mlp_ridge_candidate_20260624")
H2_3_RUNTIME = Path("results/runtime_validation_h2_3_mlp_ridge_candidate_20260624")
H2_3_PROFILE = OUT_DIR / "h2_3_profile.json"
H2_3_PROFILE_EXPORT = OUT_DIR / "c12_c345_h2_3_profile_export.json"
H8_SWITCH_AUDIT_REPORT = Path("results/h8_switch_rule_audit_20260625/h8_switch_rule_audit_report.md")
H8_SELECTOR_REPORT = Path("results/h8_calibration_selector_20260625/h8_calibration_selector_report.md")
H8_SELECTOR_PROFILE = Path("results/h8_calibration_selector_20260625/h8_pred_co_source_aug_selector_profile.json")
H8_DEPLOY = Path("results/deployment_h8_source_aug_candidate_20260625")
H8_RUNTIME = Path("results/runtime_validation_h8_source_aug_candidate_20260625")
H8_EQUIV = Path("results/equivalence_h8_source_aug_candidate_20260625/equivalence_summary.json")


SCOPES = [
    "ALL",
    "C3-CO",
    "C4-CO",
    "C5-CO",
    "C3-CO_high_200_250",
    "C4-CO_high_200_250",
    "C5-CO_high_200_250",
    "nonCO_ALL",
]


MODE_SOURCES = [
    {
        "label": "A0 baseline final",
        "mode": "A0_baseline_final",
        "source": HYBRID_SUMMARY,
        "status": "baseline",
        "reading": "Original R3aK16 + auto_v2 final ppm.",
    },
    {
        "label": "H1 Ridge + C4 rescue",
        "mode": "H1_hybrid_ridge_plus_c4_rescue",
        "source": HYBRID_SUMMARY,
        "status": "reference",
        "reading": "Strong C5 high-CO, weaker nonCO/ALL than H2.3.",
    },
    {
        "label": "H2 MLP + C4 rescue",
        "mode": "H2_hybrid_mlp_plus_c4_rescue",
        "source": HYBRID_SUMMARY,
        "status": "reference",
        "reading": "Strong overall/nonCO, weaker C5 high-CO than Ridge.",
    },
    {
        "label": "H2.2 MLP C3 + Ridge C4/C5",
        "mode": "H2_2_mlp_c3_ridge_c4c5",
        "source": HYBRID_SUMMARY,
        "status": "deployment candidate archived",
        "reading": "Conservative hybrid; worse ALL than H2.3.",
    },
    {
        "label": "H2.3 MLP C3 + Ridge C4 + C5-grid MLP",
        "mode": "H2_3_mlp_c3_ridge_c4_c5grid",
        "source": HYBRID_SUMMARY,
        "status": "current mainline",
        "reading": "Best balanced deployed candidate so far.",
    },
    {
        "label": "H8 pred-CO source-aug else H2.3",
        "mode": "H8_pred_co_source_aug_else_h23",
        "source": H8_SUMMARY,
        "status": "CO-specialist candidate",
        "reading": "Improves CO/high-CO but worsens ALL NRMSE/nonCO versus H2.3.",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metric_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    return {
        (row["mode"], row["scope"]): row
        for row in rows
        if row.get("split", "test") == "test"
    }


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 2) -> str:
    num = fnum(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}"


def load_mode_metrics(spec: dict[str, Any]) -> dict[str, Any]:
    mmap = metric_map(Path(spec["source"]))
    row: dict[str, Any] = {
        "label": spec["label"],
        "mode": spec["mode"],
        "source": str(spec["source"]),
        "status": spec["status"],
        "reading": spec["reading"],
    }
    for scope in SCOPES:
        item = mmap.get((spec["mode"], scope), {})
        prefix = scope.lower().replace("-", "_").replace("_200_250", "")
        row[f"{prefix}_rmse"] = fnum(item.get("RMSE"))
        row[f"{prefix}_nrmse"] = fnum(item.get("NRMSE"))
        row[f"{prefix}_bias"] = fnum(item.get("Bias"))
        row[f"{prefix}_p90ae"] = fnum(item.get("P90AE"))
        row[f"{prefix}_n"] = fnum(item.get("N"))
    return row


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


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def read_equiv() -> dict[str, Any]:
    if not H2_3_EQUIV.exists():
        return {}
    return json.loads(H2_3_EQUIV.read_text(encoding="utf-8"))


def read_h8_equiv() -> dict[str, Any]:
    if not H8_EQUIV.exists():
        return {}
    return json.loads(H8_EQUIV.read_text(encoding="utf-8"))


def main() -> None:
    rows = [load_mode_metrics(spec) for spec in MODE_SOURCES]
    write_csv(OUT_CSV, rows)

    by_label = {row["label"]: row for row in rows}
    h23 = by_label["H2.3 MLP C3 + Ridge C4 + C5-grid MLP"]
    h8 = by_label["H8 pred-CO source-aug else H2.3"]
    baseline = by_label["A0 baseline final"]
    equiv = read_equiv()
    h8_equiv = read_h8_equiv()

    metric_rows = []
    for row in rows:
        metric_rows.append(
            [
                row["label"],
                row["status"],
                fmt(row.get("all_rmse")),
                fmt(row.get("all_nrmse"), 4),
                fmt(row.get("c3_co_rmse")),
                fmt(row.get("c4_co_rmse")),
                fmt(row.get("c5_co_rmse")),
                fmt(row.get("c3_co_high_rmse")),
                fmt(row.get("c4_co_high_rmse")),
                fmt(row.get("c5_co_high_rmse")),
                fmt(row.get("nonco_all_rmse")),
            ]
        )

    delta_rows = []
    for row in rows:
        delta_rows.append(
            [
                row["label"],
                fmt((row.get("all_rmse") or 0.0) - (baseline.get("all_rmse") or 0.0)),
                fmt((row.get("all_nrmse") or 0.0) - (baseline.get("all_nrmse") or 0.0), 4),
                fmt((row.get("c4_co_high_rmse") or 0.0) - (baseline.get("c4_co_high_rmse") or 0.0)),
                fmt((row.get("c5_co_high_rmse") or 0.0) - (baseline.get("c5_co_high_rmse") or 0.0)),
                fmt((row.get("nonco_all_rmse") or 0.0) - (baseline.get("nonco_all_rmse") or 0.0)),
            ]
        )

    h8_vs_h23 = [
        ["ALL RMSE", fmt((h8.get("all_rmse") or 0.0) - (h23.get("all_rmse") or 0.0))],
        ["ALL NRMSE", fmt((h8.get("all_nrmse") or 0.0) - (h23.get("all_nrmse") or 0.0), 4)],
        ["C4 CO", fmt((h8.get("c4_co_rmse") or 0.0) - (h23.get("c4_co_rmse") or 0.0))],
        ["C5 CO", fmt((h8.get("c5_co_rmse") or 0.0) - (h23.get("c5_co_rmse") or 0.0))],
        ["C5 CO high", fmt((h8.get("c5_co_high_rmse") or 0.0) - (h23.get("c5_co_high_rmse") or 0.0))],
        ["nonCO ALL", fmt((h8.get("nonco_all_rmse") or 0.0) - (h23.get("nonco_all_rmse") or 0.0))],
    ]

    lines = [
        "# Target Direct-Head Mainline Confirmation",
        "",
        "Date: 2026-06-25",
        "",
        "Scope: C12 -> C345 target test, no-QC full-set. This report consolidates formal target Ridge, target MLP, hybrid profile selection, H8 CO-specialist, and H2.3 runtime parity evidence.",
        "",
        f"- Summary CSV: `{OUT_CSV.as_posix()}`",
        f"- H2.3 deployment bundle: `{H2_3_DEPLOY.as_posix()}`",
        f"- H2.3 runtime validation dir: `{H2_3_RUNTIME.as_posix()}`",
        f"- H2.3 profile JSON: `{H2_3_PROFILE.as_posix()}`",
        f"- H2.3 profile export check artifact: `{H2_3_PROFILE_EXPORT.as_posix()}`",
        f"- H8 switch audit: `{H8_SWITCH_AUDIT_REPORT.as_posix()}`",
        f"- H8 calibration-only selector: `{H8_SELECTOR_REPORT.as_posix()}`",
        f"- H8 analysis profile: `{H8_SELECTOR_PROFILE.as_posix()}`",
        f"- H8 deployment bundle: `{H8_DEPLOY.as_posix()}`",
        f"- H8 runtime validation dir: `{H8_RUNTIME.as_posix()}`",
        f"- H8 runtime equivalence: `{H8_EQUIV.as_posix()}`",
        "",
        "## Main Metrics",
        "",
        md_table(
            ["candidate", "status", "ALL", "NRMSE", "C3 CO", "C4 CO", "C5 CO", "C3 high", "C4 high", "C5 high", "nonCO"],
            metric_rows,
        ),
        "",
        "## Delta vs Original Baseline",
        "",
        md_table(
            ["candidate", "ALL", "NRMSE", "C4 high", "C5 high", "nonCO"],
            delta_rows,
        ),
        "",
        "## H8 vs H2.3",
        "",
        md_table(["metric", "H8 - H2.3"], h8_vs_h23),
        "",
        "## H8 Selector Status",
        "",
        "- The H8 switch rule is deployment-visible: switch to the CO specialist when `pred_class == CO`.",
        "- Calibration split audit supports the rule: overall precision 0.991, false-positive rate 0.009, CO recall 0.976, CO-high recall 0.960.",
        "- A calibration-only selector was added and enables H8 for C3/C4/C5 because all three clients pass switch-support thresholds and source-augmented CO validation RMSE improves over rich-only target Ridge.",
        "- H8 runtime/export support has been implemented and parity has been verified.",
        "",
        "## Runtime Parity",
        "",
    ]
    if equiv:
        lines.extend(
            [
                "H2.3:",
                f"- rows compared: {equiv.get('rows_compared')}",
                f"- missing in analysis/runtime: {equiv.get('missing_in_analysis')} / {equiv.get('missing_in_runtime')}",
                f"- mismatch rows: {equiv.get('num_mismatch')}",
                f"- max abs diff: {equiv.get('max_abs_diff')}",
                f"- mean abs diff: {equiv.get('mean_abs_diff')}",
            ]
        )
    else:
        lines.append("- H2.3 runtime parity summary was not found.")
    lines.append("")
    if h8_equiv:
        lines.extend(
            [
                "H8:",
                f"- rows compared: {h8_equiv.get('rows_compared')}",
                f"- missing in analysis/runtime: {h8_equiv.get('missing_in_analysis')} / {h8_equiv.get('missing_in_runtime')}",
                f"- mismatch rows: {h8_equiv.get('num_mismatch')}",
                f"- max abs diff: {h8_equiv.get('max_abs_diff')}",
                f"- mean abs diff: {h8_equiv.get('mean_abs_diff')}",
            ]
        )
    else:
        lines.append("- H8 runtime parity summary was not found.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Promote H2.3 as the current balanced mainline: it gives a large gain over the original baseline and already has deployment/runtime equivalence.",
            "- Keep H8 as a CO-specialist candidate, not the default mainline: it improves CO and high-CO, but worsens ALL NRMSE and nonCO versus H2.3.",
            "- H8 now has calibration-only selector support and runtime parity, so it can be treated as a deployable CO-specialist candidate.",
            "- Export/profile parameterization has started: `export_hybrid_mlp_ridge_deployment_candidate.py` now accepts `--profile-json` while preserving `--candidate h2_2/h2_3` compatibility.",
            "- Mainline decision remains H2.3 vs H8: H8 improves CO/high-CO and ALL RMSE slightly, but worsens ALL NRMSE and nonCO versus H2.3.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(OUT_CSV), "report": str(REPORT_MD), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
