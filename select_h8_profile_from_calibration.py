from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SWITCH_AUDIT = Path("results/h8_switch_rule_audit_20260625/h8_pred_co_switch_split_audit.csv")
SOURCE_AUG_FIT_AUDIT = Path("results/source_augmented_target_ridge_stratcalval_20260625/fit_audit.csv")
OUT_DIR = Path("results/h8_calibration_selector_20260625")
OUT_CSV = OUT_DIR / "h8_calibration_selector_clients.csv"
OUT_PROFILE = OUT_DIR / "h8_pred_co_source_aug_selector_profile.json"
OUT_REPORT = OUT_DIR / "h8_calibration_selector_report.md"

CLIENTS = ["C3", "C4", "C5"]
CO_CLASS = "1"


THRESHOLDS = {
    "min_switch_precision": 0.95,
    "max_switch_false_positive_rate": 0.05,
    "min_co_recall": 0.90,
    "min_co_high_recall": 0.80,
    "max_co_val_rmse_delta": 0.0,
}


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


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def switch_by_client() -> dict[str, dict[str, str]]:
    rows = read_csv(SWITCH_AUDIT)
    return {
        row["client"]: row
        for row in rows
        if row.get("split") == "calibration" and row.get("client") in CLIENTS
    }


def co_fit_by_client() -> dict[str, dict[str, dict[str, str]]]:
    rows = read_csv(SOURCE_AUG_FIT_AUDIT)
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("class_id") != CO_CLASS:
            continue
        out.setdefault(row["client"], {})[row["feature_set"]] = row
    return out


def select_clients() -> list[dict[str, Any]]:
    switch = switch_by_client()
    fit = co_fit_by_client()
    rows: list[dict[str, Any]] = []
    for client in CLIENTS:
        srow = switch.get(client, {})
        frow = fit.get(client, {})
        rich = frow.get("rich_only", {})
        aug = frow.get("rich_plus_source_preds", {})
        rich_val = fnum(rich.get("best_val_RMSE"), float("inf"))
        aug_val = fnum(aug.get("best_val_RMSE"), float("inf"))
        delta = aug_val - rich_val
        precision = fnum(srow.get("switch_precision_CO"))
        fp_rate = fnum(srow.get("switch_false_positive_rate"))
        co_recall = fnum(srow.get("CO_recall_by_switch"))
        high_recall = fnum(srow.get("CO_high_recall_by_switch"))
        checks = {
            "precision_pass": precision >= THRESHOLDS["min_switch_precision"],
            "fp_pass": fp_rate <= THRESHOLDS["max_switch_false_positive_rate"],
            "co_recall_pass": co_recall >= THRESHOLDS["min_co_recall"],
            "high_recall_pass": high_recall >= THRESHOLDS["min_co_high_recall"],
            "co_val_gain_pass": delta < THRESHOLDS["max_co_val_rmse_delta"],
        }
        rows.append(
            {
                "client": client,
                "enable_h8_pred_co_switch": int(all(checks.values())),
                "switch_precision_CO": precision,
                "switch_false_positive_rate": fp_rate,
                "CO_recall_by_switch": co_recall,
                "CO_high_recall_by_switch": high_recall,
                "rich_only_CO_val_RMSE": rich_val,
                "source_aug_CO_val_RMSE": aug_val,
                "source_aug_minus_rich_CO_val_RMSE": delta,
                **{key: int(value) for key, value in checks.items()},
            }
        )
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "client",
        "enable",
        "precision",
        "FP",
        "CO recall",
        "high recall",
        "rich CO val",
        "src-aug CO val",
        "delta",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["client"]),
                    str(row["enable_h8_pred_co_switch"]),
                    fmt(row["switch_precision_CO"]),
                    fmt(row["switch_false_positive_rate"]),
                    fmt(row["CO_recall_by_switch"]),
                    fmt(row["CO_high_recall_by_switch"]),
                    fmt(row["rich_only_CO_val_RMSE"], 2),
                    fmt(row["source_aug_CO_val_RMSE"], 2),
                    fmt(row["source_aug_minus_rich_CO_val_RMSE"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    rows = select_clients()
    write_csv(OUT_CSV, rows)
    enabled = [row["client"] for row in rows if int(row["enable_h8_pred_co_switch"]) == 1]
    profile = {
        "schema": "h8_co_specialist_selector.v1",
        "base_profile": "h2_3",
        "candidate_name": "h8_pred_co_source_aug_else_h23_calibration_selected",
        "switch_rule": {
            "type": "pred_class_equals",
            "class_id": 1,
            "enabled_clients": enabled,
            "specialist": "source_aug_target_ridge_plus_c4_rescue",
            "fallback": "h2_3",
        },
        "thresholds": THRESHOLDS,
        "selector_inputs": {
            "switch_audit": str(SWITCH_AUDIT),
            "source_aug_fit_audit": str(SOURCE_AUG_FIT_AUDIT),
        },
        "client_decisions": rows,
        "runtime_status": "analysis_profile_only; source-aug runtime artifact not yet implemented",
    }
    OUT_PROFILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# H8 Calibration-Only Selector",
        "",
        "Goal: decide whether the H8 CO-specialist switch should enter the next formal/runtime stage without using target test metrics.",
        "",
        f"- Output CSV: `{OUT_CSV.as_posix()}`",
        f"- Output profile: `{OUT_PROFILE.as_posix()}`",
        "",
        "## Selection Criteria",
        "",
        "- `pred_class == CO` switch precision on calibration must be at least 0.95.",
        "- switch false-positive rate must be at most 0.05.",
        "- CO recall must be at least 0.90.",
        "- CO-high recall must be at least 0.80.",
        "- source-augmented target Ridge must improve CO calibration-val RMSE vs rich-only target Ridge.",
        "",
        "## Client Decisions",
        "",
        table(rows),
        "",
        "## Decision",
        "",
        f"- Enabled clients: {', '.join(enabled) if enabled else 'none'}.",
        "- This is not a deployment artifact yet. It is a calibration-only analysis profile.",
        "- Next required step: implement/export source-aug target Ridge runtime support, then run parity against the H8 analysis CSV.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "profile": str(OUT_PROFILE), "enabled": enabled}, indent=2))


if __name__ == "__main__":
    main()
