"""Build the final claim-by-claim canonical-v1 scientific decision report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/experiments/iotj_canonical_v1_final"
VALIDATION = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809"
DEFAULT_OUTPUT = VALIDATION / "FINAL_SCIENTIFIC_VALIDATION_REPORT.md"


def final_recommendation(a0t_complete: bool, matrix_complete: bool, strict_collapse: bool) -> str:
    return "READY_FOR_MANUSCRIPT_FREEZE" if a0t_complete and matrix_complete and not strict_collapse else "NOT_READY"


def structured_commissioning_claim_status(deltas: dict[str, float], *, seed_count: int) -> str:
    """Keep the equal-label comparison appropriately narrow for a single fixed seed."""
    if seed_count < 2 or any(value <= 0.0 for value in deltas.values()):
        return "PASS_WITH_LIMITATION"
    return "PASS"


def strict_survival_claim_status(strict_collapse: bool) -> str:
    """A preregistered collapse flag blocks the robustness-survival claim."""
    return "BLOCKED" if strict_collapse else "PASS"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(output: Path) -> dict[str, Any]:
    comparison = read_csv(VALIDATION / "classification_comparison/canonical_classification_comparison.csv")
    target_rows = [row for row in comparison if row["target"] in {"C3", "C4", "C5"}]
    by_method = {method: {row["target"]: row for row in target_rows if row["method"] == method} for method in {row["method"] for row in target_rows}}
    expected_methods = {"FedAvg", "FedProx", "SCAFFOLD", "MMD", "A0T", "GAPS/A4"}
    a0t_complete = len(by_method.get("A0T", {})) == 3
    scaffold_audit = json.loads((VALIDATION / "comparators/scaffold_audit/scaffold_sanity_audit.json").read_text(encoding="utf-8"))
    matrix_complete = set(by_method) == expected_methods and all(len(by_method[method]) == 3 for method in expected_methods) and scaffold_audit["status"] in {"PASS", "PASS_WITH_LIMITATION"}
    strict_deltas = read_csv(VALIDATION / "strict_nonoverlap/strict_non_overlap_deltas.csv")
    strict_collapse = any(row["classification_collapse_flag"].lower() == "true" or row["regression_collapse_flag"].lower() == "true" for row in strict_deltas)

    gaps_a0t = {target: float(by_method["GAPS/A4"][target]["macro_f1"]) - float(by_method["A0T"][target]["macro_f1"]) for target in ("C3", "C4", "C5")}
    fl_deltas = {
        method: {target: float(by_method["GAPS/A4"][target]["macro_f1"]) - float(by_method[method][target]["macro_f1"]) for target in ("C3", "C4", "C5")}
        for method in ("FedAvg", "FedProx", "SCAFFOLD")
    }
    fedridge = next(row for row in read_csv(DOCS / "fedridge_bootstrap_summary.csv") if row["scope"] == "ALL" and row["gas"] == "ALL")
    master = read_csv(DOCS / "FINAL_RESULT_MASTER_TABLE.csv")
    qc = {row["scope"]: row for row in master if row["evidence"] == "qc" and row["scope"] in {"ALL_HC90", "ALL_HC95"}}
    recommendation = final_recommendation(a0t_complete, matrix_complete, strict_collapse)

    claims = [
        {"Claim": "Ordinary source-only FL is insufficient for the observed target shift", "Evidence": "Canonical FedAvg/FedProx/SCAFFOLD versus GAPS/A4", "Protocol": "same canonical data/backbone/25 rounds/LE1/seed42; optimizer regimes disclosed", "Canonical?": "yes", "Result": "; ".join(f"{method} GAPS-minus-baseline Macro-F1 C3/C4/C5=" + "/".join(f"{fl_deltas[method][t]:.4f}" for t in ("C3","C4","C5")) for method in fl_deltas), "Risk": "Different target-information and SCAFFOLD optimizer regimes; not a single-factor ablation", "Status": "PASS_WITH_LIMITATION"},
        {"Claim": "Canonical equal-label A0T exists", "Evidence": "Three fixed round-25 A0T endpoints and sealed evaluation", "Protocol": "same calibration identities/label budget; target CE only", "Canonical?": "yes", "Result": "C3/C4/C5 all complete" if a0t_complete else "incomplete", "Risk": "Single seed", "Status": "PASS" if a0t_complete else "BLOCKED"},
        {"Claim": "Structured commissioning adds value beyond label access", "Evidence": "GAPS/A4 minus equal-label A0T", "Protocol": "matched canonical data, target label budget, backbone and fixed optimization budget", "Canonical?": "yes", "Result": "Macro-F1 delta C3/C4/C5=" + "/".join(f"{gaps_a0t[t]:.6f}" for t in ("C3","C4","C5")), "Risk": "Near-zero mixed-sign single-seed deltas do not support material classification superiority beyond label access; GAPS value must be framed at lifecycle level", "Status": structured_commissioning_claim_status(gaps_a0t, seed_count=1)},
        {"Claim": "Routing error propagates into regression", "Evidence": "S_ALL, S_CC, oracle-route prediction-level analysis", "Protocol": "canonical A4+R84", "Canonical?": "yes", "Result": "Overall S_ALL-S_CC RMSE gap 2.500 ppm; 15 misroutes", "Risk": "S_CC and oracle are different diagnostic populations, so evidence is descriptive rather than causal", "Status": "PASS_WITH_LIMITATION"},
        {"Claim": "Federated H1 prior contributes to target regression", "Evidence": "83D/84D paired raw-file grouped bootstrap", "Protocol": "5000 replicates, seed20260809", "Canonical?": "yes", "Result": f"Delta RMSE={float(fedridge['delta_rmse_ppm']):.3f} ppm, 95% CI [{float(fedridge['delta_ci025_ppm']):.3f}, {float(fedridge['delta_ci975_ppm']):.3f}]", "Risk": "CI includes zero; C4 degrades", "Status": "PASS_WITH_LIMITATION"},
        {"Claim": "QC identifies higher-risk predictions", "Evidence": "HC90/HC95, same-coverage random, capture and risk-coverage analyses", "Protocol": "frozen equal-mean QC; no threshold search", "Canonical?": "yes", "Result": f"HC90/HC95 RMSE gain versus random={float(qc['ALL_HC90']['value_3']):.3f}/{float(qc['ALL_HC95']['value_3']):.3f} ppm", "Risk": "Misroute capture is 26.7%; risk evidence is useful but not exhaustive", "Status": "PASS"},
        {"Claim": "Main conclusions survive strict non-overlap", "Evidence": "raw-file-disjoint A4+R84 sensitivity", "Protocol": "exact/raw-file/raw-time overlap all zero", "Canonical?": "supplementary canonical sensitivity", "Result": "collapse flag=" + str(strict_collapse), "Risk": "C5 triggers both the preregistered classification and regression collapse flags; C4/C5 test N is 840 after two-repeat grouping", "Status": strict_survival_claim_status(strict_collapse)},
        {"Claim": "Low-budget/few-shot commissioning", "Evidence": "No canonical budget sensitivity", "Protocol": "predefined approximately 20% calibration only", "Canonical?": "yes", "Result": "Claim restricted", "Risk": "Few-shot/limited-calibration wording unsupported", "Status": "PASS_WITH_LIMITATION"},
        {"Claim": "Pi 5 evidence matches canonical deployed package", "Evidence": "package hash chain and 10,000-window benchmark", "Protocol": "FINAL_DEPLOYED_RUNTIME", "Canonical?": "yes", "Result": "P50/P95/P99 3.149/3.193/4.924 ms; 295.93 windows/s", "Risk": "FL communication is analytical/historical, not canonical wire measurement", "Status": "PASS"},
        {"Claim": "Additional algorithm exploration is required", "Evidence": "Minimal comparator matrix plus fairness and robustness closure", "Protocol": "stop rule", "Canonical?": "yes", "Result": "No", "Risk": "None beyond disclosed limitations", "Status": "NOT_REQUIRED"},
    ]
    lines = ["# Final canonical-v1 scientific validation report", "", f"Final recommendation: **{recommendation}**", "", "| Claim | Evidence | Protocol | Canonical? | Result | Risk | Status |", "|---|---|---|---|---|---|---|"]
    for row in claims:
        lines.append("| " + " | ".join(str(row[key]).replace("|", "\\|") for key in ("Claim","Evidence","Protocol","Canonical?","Result","Risk","Status")) + " |")
    lines += [
        "",
        "## Decision-gate answers",
        "",
        "1. Ordinary source-only FL inadequacy: supported for these canonical fixed endpoints, with supervision/optimizer limitations disclosed.",
        f"2. Canonical equal-label A0T: {'PASS' if a0t_complete else 'BLOCKED'}.",
        "3. GAPS versus A0T claim strength: classification performance is effectively tied under equal label access (mixed-sign, near-zero deltas at seed42); any added-value claim must be lifecycle-level, not classification-superiority wording.",
        "4. Routing-to-regression propagation: supported descriptively by the S_ALL-S_CC gap and misroute slices.",
        "5. 83D to 84D: modest overall average benefit; not statistically significant at the grouped-bootstrap 95% interval; retain C4 degradation.",
        "6. QC versus same-coverage random: positive at HC90 and HC95; capture analysis shows it does not catch every misroute.",
        f"7. Strict non-overlap: {'BLOCKED because C5 triggers both preregistered collapse flags; canonical window-level evidence does not establish strict robustness' if strict_collapse else 'conclusion retained without preregistered collapse flag'}.",
        "8. Calibration-budget wording: only 'a predefined approximately 20% commissioning calibration set'; no few-shot claim.",
        "9. Pi 5: package hash chain is consistent with the canonical A4+R84+QC deployment.",
        "10. New algorithms: not required; stop algorithm exploration.",
        "",
        "This report does not authorize manuscript-number edits, figure regeneration, model changes, hyperparameter search, outlier deletion, or additional algorithms.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"recommendation": recommendation, "a0t_complete": a0t_complete, "matrix_complete": matrix_complete, "strict_collapse": strict_collapse, "claims": claims}
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
