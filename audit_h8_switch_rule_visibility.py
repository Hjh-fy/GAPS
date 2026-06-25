from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_PREDICTIONS = Path(
    "results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/"
    "ppm_layer_co_audit/target_layer_predictions.csv"
)
H8_SWITCH_AUDIT = Path("results/co_only_source_aug_hybrid_stratcalval_20260625/switch_audit.csv")
OUT_DIR = Path("results/h8_switch_rule_audit_20260625")
OUT_CSV = OUT_DIR / "h8_pred_co_switch_split_audit.csv"
REPORT_MD = OUT_DIR / "h8_switch_rule_audit_report.md"

CO_CLASS = 1
CO_HIGH_MIN = 200.0


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


def inum(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def client_name(row: dict[str, Any]) -> str:
    raw = row.get("client") or row.get("client_id")
    text = str(raw).upper()
    if text.startswith("C"):
        return f"C{int(text[1:])}"
    return f"C{int(float(text))}"


def split_audit(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["calibration", "test"]:
        split_rows = [row for row in rows if str(row.get("split")) == split]
        for client in ["ALL", "C3", "C4", "C5"]:
            crows = split_rows if client == "ALL" else [row for row in split_rows if client_name(row) == client]
            switched = [row for row in crows if inum(row.get("pred_class")) == CO_CLASS]
            true_co = [row for row in switched if inum(row.get("true_class")) == CO_CLASS]
            nonco = [row for row in switched if inum(row.get("true_class")) != CO_CLASS]
            true_co_high = [
                row
                for row in switched
                if inum(row.get("true_class")) == CO_CLASS and fnum(row.get("true_ppm")) >= CO_HIGH_MIN
            ]
            all_true_co = [row for row in crows if inum(row.get("true_class")) == CO_CLASS]
            all_true_co_high = [
                row
                for row in crows
                if inum(row.get("true_class")) == CO_CLASS and fnum(row.get("true_ppm")) >= CO_HIGH_MIN
            ]
            pred_gas = Counter(str(row.get("pred_gas")) for row in switched)
            out.append(
                {
                    "split": split,
                    "client": client,
                    "total_N": len(crows),
                    "switch_N": len(switched),
                    "true_CO_in_switch_N": len(true_co),
                    "nonCO_in_switch_N": len(nonco),
                    "true_CO_high_in_switch_N": len(true_co_high),
                    "all_true_CO_N": len(all_true_co),
                    "all_true_CO_high_N": len(all_true_co_high),
                    "switch_rate": len(switched) / len(crows) if crows else 0.0,
                    "switch_precision_CO": len(true_co) / len(switched) if switched else 0.0,
                    "switch_false_positive_rate": len(nonco) / len(switched) if switched else 0.0,
                    "CO_recall_by_switch": len(true_co) / len(all_true_co) if all_true_co else 0.0,
                    "CO_high_recall_by_switch": len(true_co_high) / len(all_true_co_high) if all_true_co_high else 0.0,
                    "pred_gas_counts": json.dumps(dict(sorted(pred_gas.items())), ensure_ascii=False),
                }
            )
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "split",
        "client",
        "N",
        "switch",
        "true CO",
        "nonCO",
        "CO high",
        "precision",
        "FP rate",
        "CO recall",
        "high recall",
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
                    str(row["split"]),
                    str(row["client"]),
                    str(row["total_N"]),
                    str(row["switch_N"]),
                    str(row["true_CO_in_switch_N"]),
                    str(row["nonCO_in_switch_N"]),
                    str(row["true_CO_high_in_switch_N"]),
                    fmt(row["switch_precision_CO"]),
                    fmt(row["switch_false_positive_rate"]),
                    fmt(row["CO_recall_by_switch"]),
                    fmt(row["CO_high_recall_by_switch"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    rows = read_csv(TARGET_PREDICTIONS)
    audit = split_audit(rows)
    write_csv(OUT_CSV, audit)

    h8_audit_text = ""
    if H8_SWITCH_AUDIT.exists():
        h8_audit_text = H8_SWITCH_AUDIT.read_text(encoding="utf-8-sig")

    report = [
        "# H8 Switch Rule Visibility Audit",
        "",
        "Question: is the H8 `pred_class == CO` switch a deployment-visible rule with reasonable calibration/test support?",
        "",
        f"- Input target predictions: `{TARGET_PREDICTIONS.as_posix()}`",
        f"- Output CSV: `{OUT_CSV.as_posix()}`",
        "- Rule audited here: switch to CO specialist when the deployed classifier predicts CO.",
        "- This audit does not evaluate ppm improvement; it checks support, false positives, and leakage risk.",
        "",
        "## Split Audit",
        "",
        table(audit),
        "",
        "## Existing H8 Test Switch Audit",
        "",
        "```csv",
        h8_audit_text.strip(),
        "```",
        "",
        "## Reading",
        "",
        "- The rule uses only deployment-visible `pred_class`, so it is eligible for runtime use.",
        "- Calibration support and false-positive rate should be compared with test before promoting H8.",
        "- If calibration and test switch behavior are aligned, the next step is a formal selector that chooses H2.3 vs H8 using calibration-validation only.",
        "- H8 still needs a richer runtime artifact because its CO specialist depends on source-head predictions, not just the existing target Ridge policy.",
        "",
    ]
    REPORT_MD.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "rows": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
