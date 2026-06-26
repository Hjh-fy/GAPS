"""Summarize bidirectional target-side profile selection.

This script owns direction-level regression/profile reporting.  The formal
deployment selector in ``select_target_profile.py`` owns runtime mode selection
for the current deployable C12->C345 bundle.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/bidirectional_profile_selection_20260626")
C12_SUMMARY = Path("results/target_direct_head_mainline_20260625/target_direct_head_mainline_summary.csv")
C45_SUMMARY = Path("results/c45_c123_optimal_config_analysis_20260626/c45_c123_optimal_config_summary.csv")
FORMAL_SELECTOR = Path("results/target_profile_selector_20260626/selected_profiles.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def metric_long(rows: list[dict[str, str]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in rows:
        if row.get("mode") == mode and row.get("scope") == scope:
            return fnum(row.get(metric))
    return None


def metric_wide(rows: list[dict[str, str]], mode: str, field: str) -> float | None:
    for row in rows:
        if row.get("mode") == mode:
            return fnum(row.get(field))
    return None


def fmt(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def c12_row(rows: list[dict[str, str]], mode: str, label: str, role: str) -> dict[str, Any]:
    return {
        "direction": "C12_to_C345",
        "source_clients": "C1,C2",
        "target_clients": "C3,C4,C5",
        "mode": mode,
        "label": label,
        "role": role,
        "ALL_RMSE": metric_wide(rows, mode, "all_rmse"),
        "ALL_NRMSE": metric_wide(rows, mode, "all_nrmse"),
        "CO_RMSE_by_target": "; ".join(
            [
                f"C3={fmt(metric_wide(rows, mode, 'c3_co_rmse'))}",
                f"C4={fmt(metric_wide(rows, mode, 'c4_co_rmse'))}",
                f"C5={fmt(metric_wide(rows, mode, 'c5_co_rmse'))}",
            ]
        ),
        "CO_high_RMSE_by_target": "; ".join(
            [
                f"C3={fmt(metric_wide(rows, mode, 'c3_co_high_rmse'))}",
                f"C4={fmt(metric_wide(rows, mode, 'c4_co_high_rmse'))}",
                f"C5={fmt(metric_wide(rows, mode, 'c5_co_high_rmse'))}",
            ]
        ),
        "nonCO_ALL_RMSE": metric_wide(rows, mode, "nonco_all_rmse"),
    }


def c45_row(rows: list[dict[str, str]], mode: str, label: str, role: str) -> dict[str, Any]:
    return {
        "direction": "C45_to_C123",
        "source_clients": "C4,C5",
        "target_clients": "C1,C2,C3",
        "mode": mode,
        "label": label,
        "role": role,
        "ALL_RMSE": metric_long(rows, mode, "ALL"),
        "ALL_NRMSE": metric_long(rows, mode, "ALL", "NRMSE"),
        "CO_RMSE_by_target": "; ".join(
            [
                f"C1={fmt(metric_long(rows, mode, 'C1-CO'))}",
                f"C2={fmt(metric_long(rows, mode, 'C2-CO'))}",
                f"C3={fmt(metric_long(rows, mode, 'C3-CO'))}",
            ]
        ),
        "CO_high_RMSE_by_target": "; ".join(
            [
                f"C1={fmt(metric_long(rows, mode, 'C1-CO_high_200_250'))}",
                f"C2={fmt(metric_long(rows, mode, 'C2-CO_high_200_250'))}",
                f"C3={fmt(metric_long(rows, mode, 'C3-CO_high_200_250'))}",
            ]
        ),
        "nonCO_ALL_RMSE": metric_long(rows, mode, "nonCO_ALL"),
    }


def profile_c12() -> dict[str, Any]:
    formal = read_json(FORMAL_SELECTOR)
    deployment_lite = formal.get("profiles", {}).get("deployment_lite", {})
    return {
        "schema": "gaps_target_profile_selection.v1",
        "direction": "C12_to_C345",
        "source_clients": ["C1", "C2"],
        "target_clients": ["C3", "C4", "C5"],
        "selection_priority": ["no_qc_full_set_ALL_RMSE", "ALL_NRMSE", "CO_high_RMSE", "nonCO_RMSE"],
        "balanced_mainline": {
            "profile": "H2.3",
            "description": "MLP for C3, Ridge for C4, C5-grid MLP for C5.",
            "artifact": "results/deployment_h2_3_mlp_ridge_candidate_20260624",
        },
        "co_specialist_candidate": {
            "profile": "H8_plus_formal_C4_route_rescue",
            "description": "Source-aug CO specialist when pred_class==CO, H2.3 fallback, plus calibration-selected C4 route rescue.",
            "artifact": "results/deployment_h8_formal_c4_rescue_candidate_20260625",
            "status": "runtime_ready_specialist",
            "runtime_parity": "results/equivalence_h8_formal_c4_rescue_candidate_20260626/equivalence_summary.json",
            "guardrail_audit": "results/h8_c4_guardrail_audit_20260626/h8_c4_guardrail_summary.json",
            "feature_schema": "results/feature_schema_validation_h8_formal_c4_rescue_20260626/feature_schema_validation.json",
        },
        "deployment_lite_candidate": {
            "profile": deployment_lite.get("selected_profile", "H2.3"),
            "description": "L1 remains analysis-only until an exported runtime bundle proves a size or latency advantage.",
            "status": "fallback",
            "reason": deployment_lite.get("reason", "L1 runtime bundle is missing."),
        },
        "route_rescue": {
            "enabled": True,
            "client": "C4",
            "gate": "client=C4, pred_class=Ethanol, final_ppm<=20, risk_score>=6, rescue_ppm=250",
        },
    }


def profile_c45() -> dict[str, Any]:
    return {
        "schema": "gaps_target_profile_selection.v1",
        "direction": "C45_to_C123",
        "source_clients": ["C4", "C5"],
        "target_clients": ["C1", "C2", "C3"],
        "selection_priority": ["no_qc_full_set_ALL_RMSE", "ALL_NRMSE", "CO_high_RMSE", "nonCO_RMSE"],
        "balanced_mainline": {
            "profile": "target_Ridge_direct_all_clients",
            "description": "Target Ridge direct heads for C1/C2/C3.",
            "status": "analysis_candidate",
        },
        "co_specialist_candidate": {
            "profile": "H8_style_source_aug_CO_else_Ridge",
            "description": "Improves C3 CO/high-CO but worsens ALL and nonCO; keep diagnostic only.",
            "status": "diagnostic_only",
        },
        "deployment_lite_candidate": {
            "profile": "target_Ridge_direct_all_clients",
            "description": "No reverse-direction lite runtime bundle is available; keep the balanced target Ridge profile.",
            "status": "fallback",
        },
        "route_rescue": {
            "enabled": False,
            "reason": "C4 is a source client in this direction; no target C4 route-rescue is applicable.",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c12 = read_csv(C12_SUMMARY)
    c45 = read_csv(C45_SUMMARY)
    rows = [
        c12_row(c12, "A0_baseline_final", "baseline final", "baseline"),
        c12_row(c12, "H2_3_mlp_c3_ridge_c4_c5grid", "H2.3 balanced mainline", "balanced_mainline"),
        c12_row(c12, "H8_plus_formal_c4_route_rescue", "H8 + formal C4 rescue", "runtime_ready_co_specialist"),
        c45_row(c45, "A0_baseline_final", "baseline final", "baseline"),
        c45_row(c45, "H1_target_Ridge_direct", "target Ridge direct", "balanced_mainline"),
        c45_row(c45, "H8_style_source_aug_CO_else_Ridge", "H8-style source-aug CO else Ridge", "diagnostic_co_specialist"),
    ]
    write_csv(OUT_DIR / "bidirectional_profile_selection_summary.csv", rows)
    (OUT_DIR / "selected_profile_c12_to_c345.json").write_text(
        json.dumps(profile_c12(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "selected_profile_c45_to_c123.json").write_text(
        json.dumps(profile_c45(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    headers = [
        "direction",
        "label",
        "role",
        "ALL_RMSE",
        "ALL_NRMSE",
        "CO_RMSE_by_target",
        "CO_high_RMSE_by_target",
        "nonCO_ALL_RMSE",
    ]
    lines = [
        "# Bidirectional Target Profile Selection",
        "",
        "This report makes the current selection policy explicit: the framework is shared, but the chosen profile is direction-specific.",
        "",
        "## Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["direction"]),
                    str(row["label"]),
                    str(row["role"]),
                    fmt(row["ALL_RMSE"]),
                    fmt(row["ALL_NRMSE"], 4),
                    str(row["CO_RMSE_by_target"]),
                    str(row["CO_high_RMSE_by_target"]),
                    fmt(row["nonCO_ALL_RMSE"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- C12 -> C345: H2.3 is the balanced mainline; H8 + formal C4 rescue is the guarded CO-priority runtime-ready specialist.",
            "- C45 -> C123: use target Ridge direct as the clean balanced mainline; H8-style source-aug switching is diagnostic because it improves C3 CO/high-CO but worsens ALL/nonCO.",
            "- Deployment-lite is not established in either direction yet. L1 remains analysis-only until an exported runtime bundle proves a size or latency advantage.",
            "- Therefore the final system should expose a direction-specific profile selector, not a single hard-coded regression head.",
            "",
        ]
    )
    (OUT_DIR / "bidirectional_profile_selection_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
