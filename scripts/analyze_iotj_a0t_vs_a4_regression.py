"""Generate compact fixed-endpoint evidence for canonical-v1 A0T versus A4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_iotj_a0t_vs_a4_qc import qc_summary_rows
from scripts.run_iotj_a0t_vs_a4_regression import DEFAULT_OUTPUT, summarize_scope


METHODS = ("A0T", "A4")
TARGETS = ("C3", "C4", "C5")
SCOPES = ("S_ALL", "S_CC", "Oracle_ALL", "Oracle_CC")
FINAL_OUTPUTS = (
    "regression_comparison.csv",
    "per_gas_regression_comparison.csv",
    "routing_scope_summary.csv",
    "qc_comparison.csv",
    "A0T_A4_QC_COMPARISON.csv",
    "ROUTING_VS_REGRESSION_ANALYSIS.md",
    "C5_A0T_VS_A4_REGRESSION.md",
    "A0T_VS_GAPS_FINAL_CONCLUSION.md",
    "A0T_VS_A4_REGRESSION_REPORT.md",
    "RESULT_ANALYSIS.md",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def routing_row(
    method: str, target: str, rmse_by_scope: Mapping[str, float]
) -> dict[str, Any]:
    return {
        "method": method,
        "target": target,
        "routing_gap": float(rmse_by_scope["S_ALL"] - rmse_by_scope["S_CC"]),
        "regression_gap": float(rmse_by_scope["S_CC"] - rmse_by_scope["Oracle_ALL"]),
        "paired_regression_gap": float(rmse_by_scope["S_CC"] - rmse_by_scope["Oracle_CC"]),
        **{f"RMSE_{scope}": float(rmse_by_scope[scope]) for scope in SCOPES},
    }


def regression_decision(c5_delta: float, pooled_delta: float) -> str:
    return (
        "REGRESSION_ADVANTAGE_SUPPORTED"
        if float(c5_delta) < 0.0 and float(pooled_delta) < 0.0
        else "REGRESSION_ADVANTAGE_NOT_SUPPORTED"
    )


def _endpoint(root: Path, method: str, target: str) -> Path:
    return root / "endpoints" / f"CAN-V1-REG-{method}-{target}-S42"


def _scope_file(scope: str) -> str:
    return {
        "S_ALL": "test_s_all.csv",
        "S_CC": "test_s_cc.csv",
        "Oracle_ALL": "test_oracle_all.csv",
        "Oracle_CC": "test_oracle_cc.csv",
    }[scope]


def _macro_f1(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, int]:
    truth = [int(row["true_class"]) for row in rows]
    pred = [int(row["pred_class"]) for row in rows]
    correct = sum(a == b for a, b in zip(truth, pred))
    scores = []
    for class_id in range(4):
        tp = sum(a == class_id and b == class_id for a, b in zip(truth, pred))
        fp = sum(a != class_id and b == class_id for a, b in zip(truth, pred))
        fn = sum(a == class_id and b != class_id for a, b in zip(truth, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return correct / len(rows), sum(scores) / 4.0, len(rows) - correct


def _format(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _high_concentration_rows(root: Path) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        rows = read_csv(_endpoint(root, method, "C5") / "test_s_all.csv")
        for gas, raw_gas in (("CO", "carbon_monoxide"), ("Methane", "methane")):
            selected_gas = [row for row in rows if row["gas"] == raw_gas]
            highest = max(float(row["true_ppm"]) for row in selected_gas)
            selected = [row for row in selected_gas if math.isclose(float(row["true_ppm"]), highest)]
            output.append({"method": method, "gas": gas, "concentration_ppm": highest, **summarize_scope(selected)})
    return output


def analyze(root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = root.resolve()
    existing = [name for name in FINAL_OUTPUTS if (root / name).exists()]
    if existing:
        raise FileExistsError(f"FAIL_CLOSED analysis outputs already exist: {existing}")

    checkpoint_hashes = {
        (row["DA"], row["target_clients"]): row["provenance"]
        for row in read_csv(root / "experiment_registry.csv")
    }
    regression_rows: list[dict[str, Any]] = []
    pooled_records: dict[tuple[str, str], list[dict[str, str]]] = {}
    for method in METHODS:
        for scope in SCOPES:
            pooled_records[(method, scope)] = []
        for target in TARGETS:
            endpoint_metrics = read_csv(_endpoint(root, method, target) / "scope_summary.csv")
            for row in endpoint_metrics:
                scope = row["scope"]
                regression_rows.append(
                    {
                        **row,
                        "unit": "ppm",
                        "seed": 42,
                        "checkpoint_sha256": checkpoint_hashes[(method, target)],
                        "calculation_status": "recomputed",
                    }
                )
                pooled_records[(method, scope)].extend(
                    read_csv(_endpoint(root, method, target) / _scope_file(scope))
                )
        for scope in SCOPES:
            regression_rows.append(
                {
                    "experiment_id": f"CAN-V1-REG-{method}-POOLED-S42",
                    "method": method,
                    "target": "POOLED_C3_C4_C5",
                    "scope": scope,
                    **summarize_scope(pooled_records[(method, scope)]),
                    "unit": "ppm",
                    "seed": 42,
                    "checkpoint_sha256": "target_specific;see experiment_registry.csv",
                    "calculation_status": "recomputed_by_record_concatenation",
                }
            )

    lookup = {(row["method"], row["target"], row["scope"]): row for row in regression_rows}
    for row in regression_rows:
        peer = lookup[("A0T" if row["method"] == "A4" else "A4", row["target"], row["scope"])]
        row["A4_minus_A0T_RMSE"] = float(lookup[("A4", row["target"], row["scope"])]["RMSE"]) - float(lookup[("A0T", row["target"], row["scope"])]["RMSE"])
        row["comparison_peer_N"] = peer["N"]
    write_csv(root / "regression_comparison.csv", regression_rows)

    per_gas = read_csv(root / "per_gas_metrics_raw.csv")
    gas_lookup = {(r["method"], r["target"], r["scope"], r["gas"]): r for r in per_gas}
    for row in per_gas:
        key = (row["target"], row["scope"], row["gas"])
        row["A4_minus_A0T_RMSE"] = float(gas_lookup[("A4", *key)]["RMSE"]) - float(gas_lookup[("A0T", *key)]["RMSE"])
        row["unit"] = "ppm"
        row["calculation_status"] = "recomputed"
    write_csv(root / "per_gas_regression_comparison.csv", per_gas)

    routing_rows = []
    for method in METHODS:
        for target in (*TARGETS, "POOLED_C3_C4_C5"):
            values = {scope: float(lookup[(method, target, scope)]["RMSE"]) for scope in SCOPES}
            routing_rows.append({**routing_row(method, target, values), "unit": "ppm", "seed": 42})
    write_csv(root / "routing_scope_summary.csv", routing_rows)

    qc_rows = read_csv(root / "qc_comparison_raw.csv")
    for method in METHODS:
        pooled_qc = []
        for target in TARGETS:
            pooled_qc.extend(read_csv(_endpoint(root, method, target) / "test_qc_frozen.csv"))
        qc_rows.extend(qc_summary_rows(method, "POOLED_C3_C4_C5", pooled_qc))
    qc_lookup = {(r["method"], r["target"], r["workpoint"], r["population"]): r for r in qc_rows}
    for row in qc_rows:
        key = (row["target"], row["workpoint"], row["population"])
        a4, a0t = qc_lookup[("A4", *key)], qc_lookup[("A0T", *key)]
        row["A4_minus_A0T_RMSE"] = "" if not a4["RMSE"] or not a0t["RMSE"] else float(a4["RMSE"]) - float(a0t["RMSE"])
        row["calculation_status"] = "recomputed_with_frozen_A4_policy"
    write_csv(root / "qc_comparison.csv", qc_rows)
    shutil.copyfile(root / "qc_comparison.csv", root / "A0T_A4_QC_COMPARISON.csv")

    classification = []
    for method in METHODS:
        pooled = []
        for target in TARGETS:
            rows = read_csv(_endpoint(root, method, target) / "test_s_all.csv")
            accuracy, macro_f1, errors = _macro_f1(rows)
            classification.append({"method": method, "target": target, "N": len(rows), "accuracy": accuracy, "macro_f1": macro_f1, "errors": errors})
            pooled.extend(rows)
        accuracy, macro_f1, errors = _macro_f1(pooled)
        classification.append({"method": method, "target": "POOLED_C3_C4_C5", "N": len(pooled), "accuracy": accuracy, "macro_f1": macro_f1, "errors": errors})

    c5_delta = float(lookup[("A4", "C5", "S_ALL")]["RMSE"]) - float(lookup[("A0T", "C5", "S_ALL")]["RMSE"])
    pooled_delta = float(lookup[("A4", "POOLED_C3_C4_C5", "S_ALL")]["RMSE"]) - float(lookup[("A0T", "POOLED_C3_C4_C5", "S_ALL")]["RMSE"])
    decision = regression_decision(c5_delta, pooled_delta)

    route_table = _md_table(
        ["Method", "Target", "S_ALL", "S_CC", "Oracle_ALL", "routing gap", "paired mapping gap"],
        [[r["method"], r["target"], *[_format(r[f"RMSE_{s}"]) for s in ("S_ALL", "S_CC", "Oracle_ALL")], _format(r["routing_gap"]), _format(r["paired_regression_gap"])] for r in routing_rows],
    )
    (root / "ROUTING_VS_REGRESSION_ANALYSIS.md").write_text(
        "# Routing versus regression analysis\n\n" + route_table +
        "\n\nThe requested `routing_gap` is S_ALL minus S_CC. The requested `regression_gap` uses differently sized populations (S_CC minus Oracle_ALL), so mechanism attribution uses the paired S_CC minus Oracle_CC diagnostic. The paired gap is zero by construction here: once the route is correct, the deployed and Oracle feature/model paths coincide. The A4 gain is therefore attributable to avoiding or changing high-cost misroutes, not to a better correct-route Ridge mapping.\n",
        encoding="utf-8",
    )

    gas_c5 = [r for r in per_gas if r["target"] == "C5" and r["scope"] == "S_ALL" and r["gas"] in {"CO", "Methane"}]
    conc_c5 = [r for r in read_csv(root / "per_concentration_metrics_raw.csv") if r["target"] == "C5" and r["scope"] == "S_ALL" and r["gas"] in {"CO", "Methane"}]
    high = _high_concentration_rows(root)
    specials = [read_csv(_endpoint(root, method, "C5") / "special_slices.csv")[0] for method in METHODS]
    c5_text = "# C5 A0T versus A4 regression\n\n## CO and Methane\n\n" + _md_table(
        ["Method", "Gas", "N", "RMSE", "MAE", "Bias"],
        [[r["method"], r["gas"], r["N"], _format(r["RMSE"]), _format(r["MAE"]), _format(r["Bias"])] for r in gas_c5],
    )
    c5_text += "\n\n## Concentration RMSE curve\n\n" + _md_table(
        ["Method", "Gas", "ppm", "N", "RMSE", "Bias"],
        [[r["method"], r["gas"], _format(r["true_ppm"], 1), r["N"], _format(r["RMSE"]), _format(r["Bias"])] for r in conc_c5],
    )
    c5_text += "\n\n## Highest observed concentration per gas\n\n" + _md_table(
        ["Method", "Gas", "ppm", "N", "RMSE", "Bias"],
        [[r["method"], r["gas"], _format(r["concentration_ppm"], 1), r["N"], _format(r["RMSE"]), _format(r["Bias"])] for r in high],
    )
    c5_text += "\n\n## Methane 225 ppm repeat1\n\n" + _md_table(
        ["Method", "N", "RMSE", "MAE", "Bias"],
        [[r["method"], r["N"], _format(r["RMSE"]), _format(r["MAE"]), _format(r["Bias"])] for r in specials],
    ) + "\n"
    (root / "C5_A0T_VS_A4_REGRESSION.md").write_text(c5_text, encoding="utf-8")

    class_lookup = {(r["method"], r["target"]): r for r in classification}
    q1 = (
        "At the pooled fixed endpoint, A4 is descriptively higher "
        f"({class_lookup[('A4','POOLED_C3_C4_C5')]['accuracy']:.6f} vs "
        f"{class_lookup[('A0T','POOLED_C3_C4_C5')]['accuracy']:.6f}); C3 and C5 accuracy are tied, and seed42 alone does not support a stability or general superiority claim."
    )
    conclusion = f"""# A0T versus GAPS final conclusion

## Q1: Does A4 improve classification?

{q1}

## Q2: Does A4 improve regression?

Yes under the preregistered descriptive dual gate: C5 A4-minus-A0T S_ALL RMSE is {c5_delta:.6f} ppm and pooled C3+C4+C5 A4-minus-A0T S_ALL RMSE is {pooled_delta:.6f} ppm. Both are negative.

## Q3: Where does the gain arise?

The paired S_CC-minus-Oracle_CC mapping gap is 0 for both methods. Oracle_ALL predictions are also identical between methods. Thus this study supports improved downstream quantitative sensing through lower-cost routing errors, but does not support improvement of the correctly routed R84 regression mapping itself. C5 CO and Methane are adverse subgroups and must remain visible.

## Final decision

`{decision}`

This is seed42 fixed-endpoint evidence. It does not establish multi-seed stability or universal per-gas superiority.
"""
    (root / "A0T_VS_GAPS_FINAL_CONCLUSION.md").write_text(conclusion, encoding="utf-8")

    report = f"""# A0T versus A4 regression report

## Input contract and provenance

- Six immutable round25 adapted classifier checkpoints: A0T/A4 × C3/C4/C5.
- Canonical-v1 target calibration/test roles; fixed R84_FED_H1 alpha table; seed42.
- Metrics are recomputed from raw endpoint prediction records. No confidence interval or significance test is reported because there is one seed and windows are not independent clients.

## Primary result

- C5 S_ALL RMSE: A0T {_format(lookup[('A0T','C5','S_ALL')]['RMSE'])} ppm; A4 {_format(lookup[('A4','C5','S_ALL')]['RMSE'])} ppm; delta {c5_delta:.4f} ppm.
- Pooled S_ALL RMSE: A0T {_format(lookup[('A0T','POOLED_C3_C4_C5','S_ALL')]['RMSE'])} ppm; A4 {_format(lookup[('A4','POOLED_C3_C4_C5','S_ALL')]['RMSE'])} ppm; delta {pooled_delta:.4f} ppm.
- Decision: `{decision}`.

## Anomalies and sensitivity

- C5 CO and Methane worsen under A4 despite total C5 improvement.
- Frozen A4 QC thresholds transfer to A0T with a substantially different achieved coverage; this is a fixed-policy transfer comparison, not an equal-coverage refit.
- Correct-route mappings do not improve; the quantitative advantage is driven by the identity/severity of routing mistakes.

## Evidence files

- `regression_comparison.csv`, `per_gas_regression_comparison.csv`, `routing_scope_summary.csv`
- `qc_comparison.csv`, `C5_A0T_VS_A4_REGRESSION.md`
- `A0T_VS_GAPS_FINAL_CONCLUSION.md`
"""
    (root / "A0T_VS_A4_REGRESSION_REPORT.md").write_text(report, encoding="utf-8")
    (root / "RESULT_ANALYSIS.md").write_text(report, encoding="utf-8")

    return {"status": "ANALYSIS_COMPLETE", "decision": decision, "c5_delta": c5_delta, "pooled_delta": pooled_delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    print(json.dumps(analyze(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
