"""Analyze the frozen canonical-v1 C5 low-label commissioning study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "results/iotj_canonical_v1_c5_budget_20260810"
EVALUATION = STUDY_ROOT / "evaluation"
DOCS = ROOT / "docs/experiments/iotj_canonical_v1_final"
CANONICAL_COMPARISON = DOCS / "canonical_classification_comparison.csv"
BUDGET_ORDER = (20, 15, 10, 5)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def build_comparison(
    existing_20pct: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = {
        (str(row["method"]), int(row["budget_pct"])): row
        for row in [*existing_20pct, *new_rows]
    }
    required = {(method, budget) for method in ("A0T", "A4") for budget in BUDGET_ORDER}
    if set(records) != required:
        raise RuntimeError(f"comparison matrix differs: missing={sorted(required - set(records))}")
    baseline = {
        method: _number(records[(method, 20)], "macro_f1")
        for method in ("A0T", "A4")
    }
    output: list[dict[str, Any]] = []
    for budget in BUDGET_ORDER:
        a0t = records[("A0T", budget)]
        a4 = records[("A4", budget)]
        a0t_f1 = _number(a0t, "macro_f1")
        a4_f1 = _number(a4, "macro_f1")
        output.append({
            "budget_pct": budget,
            "calibration_n": {20: 320, 15: 240, 10: 160, 5: 80}[budget],
            "covered_strata": 40,
            "a0t_accuracy": _number(a0t, "accuracy"),
            "a0t_macro_f1": a0t_f1,
            "a0t_nll": _number(a0t, "nll"),
            "a0t_ece": _number(a0t, "ece"),
            "a4_accuracy": _number(a4, "accuracy"),
            "a4_macro_f1": a4_f1,
            "a4_nll": _number(a4, "nll"),
            "a4_ece": _number(a4, "ece"),
            "a4_minus_a0t_macro_f1": round(a4_f1 - a0t_f1, 12),
            "a0t_delta_vs_20pct": round(a0t_f1 - baseline["A0T"], 12),
            "a4_delta_vs_20pct": round(a4_f1 - baseline["A4"], 12),
            "a0t_source_macro_f1": _number(a0t, "source_macro_f1"),
            "a4_source_macro_f1": _number(a4, "source_macro_f1"),
            "seed": 42,
        })
    return output


def scientific_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    practical = any(
        int(row["budget_pct"]) in {10, 5}
        and float(row["a4_minus_a0t_macro_f1"]) >= 0.01
        for row in rows
    )
    return {
        "conclusion": "LABEL_EFFICIENCY_SUPPORTED" if practical else "LABEL_EFFICIENCY_NOT_SUPPORTED",
        "multi_seed_recommendation": "YES_PROPOSAL_ONLY" if practical else "NO",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(study: Path, docs: Path) -> dict[str, Any]:
    analysis_path = study / "C5_LABEL_BUDGET_FINAL_ANALYSIS.md"
    if analysis_path.exists():
        raise FileExistsError(f"analysis already exists: {analysis_path}")
    canonical = read_csv(CANONICAL_COMPARISON)
    existing: list[dict[str, Any]] = []
    for method, canonical_method in (("A0T", "A0T"), ("A4", "GAPS/A4")):
        matches = [
            row for row in canonical
            if row["method"] == canonical_method and row["target"] == "C5"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"canonical 20% row differs: {canonical_method}")
        row = matches[0]
        existing.append({
            "method": method,
            "budget_pct": 20,
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "nll": row["nll"],
            "ece": row["ece"],
            "source_macro_f1": row["source_macro_f1"],
        })
    target_rows = [
        *read_csv(study / "evaluation/c5_budget_a0t_metrics.csv"),
        *read_csv(study / "evaluation/c5_budget_a4_metrics.csv"),
    ]
    source_rows = {
        (row["method"], int(row["budget_pct"])): row
        for row in read_csv(study / "evaluation/c5_budget_source_retention.csv")
    }
    combined: list[dict[str, Any]] = []
    for row in target_rows:
        key = (row["method"], int(row["budget_pct"]))
        combined.append({
            **row,
            "source_macro_f1": source_rows[key]["source_macro_f1"],
        })
    comparison = build_comparison(existing, combined)
    decision = scientific_decision(comparison)
    write_csv(study / "c5_budget_comparison.csv", comparison)
    for name in (
        "c5_budget_a0t_metrics.csv",
        "c5_budget_a4_metrics.csv",
        "c5_budget_source_retention.csv",
    ):
        shutil.copy2(study / "evaluation" / name, study / name)

    by_budget = {int(row["budget_pct"]): row for row in comparison}
    gaps = "/".join(f"{100 * float(by_budget[budget]['a4_minus_a0t_macro_f1']):+.4f} pp" for budget in BUDGET_ORDER)
    a0t_curve = "/".join(f"{100 * float(by_budget[budget]['a0t_macro_f1']):.4f}%" for budget in BUDGET_ORDER)
    a4_curve = "/".join(f"{100 * float(by_budget[budget]['a4_macro_f1']):.4f}%" for budget in BUDGET_ORDER)
    report = f"""# C5 low-label commissioning analysis

Status: **{decision['conclusion']}**.

## Frozen scope

This is the seed42 canonical-v1 C5 window-level commissioning label-budget sensitivity. It reuses the formal 20% target metrics and adds exactly six fresh 25-round endpoints. It does not establish few-shot unseen-experiment, strict cross-experiment, or deployment-independent generalization and does not weaken the existing strict C5 collapse finding.

## Primary result

- Budgets 20/15/10/5% use 320/240/160/80 calibration windows.
- All budgets cover 40/40 class × concentration strata, so performance changes are attributable primarily to quantity reduction rather than concentration-support loss.
- A0T Macro-F1, 20/15/10/5%: {a0t_curve}.
- GAPS/A4 Macro-F1, 20/15/10/5%: {a4_curve}.
- GAPS/A4 minus A0T, 20/15/10/5%: {gaps}.

## Required answers

1. The complete A0T curve is reported above.
2. The complete GAPS/A4 curve is reported above.
3. Gap expansion is assessed directly in c5_budget_comparison.csv.
4. The first practically meaningful difference is the first 10% or 5% row with a prespecified one-percentage-point gap; absent such a row, no label-efficiency advantage is supported.
5. Practical significance uses the preregistered one-percentage-point gate; seed42 alone is not stability evidence.
6. The 5% subset covers all 40 strata.
7. Stratum coverage is therefore 40/40 (100%).
8. Any performance change is more consistent with label-quantity reduction than stratum-support loss under this nested family.
9. Source forgetting is reported relative to canonical FedAvg in c5_budget_source_retention.csv.
10. GAPS/A4 may be called more label-efficient only when status is LABEL_EFFICIENCY_SUPPORTED.
11. Main-text use requires claim wording consistent with the status and the strict-boundary limitation above.
12. Multi-seed recommendation: **{decision['multi_seed_recommendation']}**. No additional run was launched automatically.

## Stop rule

The six-run C5 classification study is complete. No C3/C4, lower-budget, multi-seed, R84, QC, method, preprocessing, or hyperparameter extension is authorized by this result.
"""
    analysis_path.write_text(report, encoding="utf-8")
    evaluation_manifest = json.loads((study / "evaluation/evaluation_manifest.json").read_text(encoding="utf-8"))
    checkpoint_index = {
        run_id: item["checkpoint_sha256"]
        for run_id, item in evaluation_manifest["gate"]["runs"].items()
    }
    (study / "checkpoint_sha256.json").write_text(
        json.dumps(checkpoint_index, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "gaps.iotj.c5_label_budget.summary.v1",
        "status": decision["conclusion"],
        "multi_seed_recommendation": decision["multi_seed_recommendation"],
        "counts": {"20": 320, "15": 240, "10": 160, "5": 80},
        "strata_coverage": {"20": "40/40", "15": "40/40", "10": "40/40", "5": "40/40"},
        "seed": 42,
        "strict_boundary": "canonical_window_level_only; existing strict C5 collapse retained",
        "comparison_sha256": sha256(study / "c5_budget_comparison.csv"),
    }
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(analysis_path, docs / "C5_LABEL_BUDGET_ANALYSIS.md")
    shutil.copy2(study / "c5_budget_comparison.csv", docs / "c5_label_budget_comparison.csv")
    shutil.copy2(study / "c5_budget_strata_coverage.csv", docs / "c5_label_budget_strata_coverage.csv")
    (docs / "c5_label_budget_manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    evidence_paths = [
        analysis_path,
        study / "c5_budget_comparison.csv",
        study / "c5_budget_source_retention.csv",
        study / "c5_budget_strata_coverage.csv",
        study / "checkpoint_sha256.json",
        study / "evaluation/evaluation_manifest.json",
        study / "PRE_RUN_FREEZE.json",
    ]
    evidence = {
        path.relative_to(study).as_posix(): sha256(path)
        for path in evidence_paths
    }
    (study / "evidence_sha256.json").write_text(
        json.dumps({"schema_version": "gaps.iotj.c5_label_budget.evidence.v1", "files": evidence}, indent=2) + "\n",
        encoding="utf-8",
    )
    if decision["multi_seed_recommendation"] == "YES_PROPOSAL_ONLY":
        (study / "C5_LOW_BUDGET_MULTI_SEED_PROPOSAL.md").write_text(
            "# C5 low-budget multi-seed proposal\n\n"
            "A practical seed42 gap met the preregistered proposal gate. A future, separately authorized study may use three independent nested families. This file does not authorize execution.\n",
            encoding="utf-8",
        )
    return {**decision, "rows": len(comparison)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY_ROOT)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args()
    print(json.dumps(run(args.study.resolve(), args.docs.resolve()), indent=2))


if __name__ == "__main__":
    main()
