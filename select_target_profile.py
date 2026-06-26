"""Create a formal target-profile selection record.

The selector separates deployment modes from test-set metric reporting:

* ``balanced`` defaults to H2.3 as the stable no-QC full-set mainline.
* ``co_priority`` may select H8+C4 only when guardrail, feature-schema, and
  runtime-parity checks pass.
* ``deployment_lite`` may select L1 only after an exported bundle proves a clear
  size or latency advantage; otherwise it falls back safely.

The output is a record for reporting and deployment bookkeeping. It does not use
target test metrics for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def row_by_profile(rows: list[dict[str, str]], profile: str) -> dict[str, str]:
    for row in rows:
        if row.get("profile") == profile:
            return row
    return {}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def lite_decision(benchmark_rows: list[dict[str, str]]) -> dict[str, Any]:
    h23 = row_by_profile(benchmark_rows, "H2.3")
    l1 = row_by_profile(benchmark_rows, "L1")
    if not l1 or l1.get("status") != "ok":
        return {
            "selected_profile": "H2.3",
            "fallback_profile": "H2.3",
            "reason": "L1 has no exported runtime bundle or benchmark pass yet.",
            "lite_status": l1.get("status", "missing") if l1 else "missing",
            "size_improvement": None,
            "latency_improvement": None,
        }
    h23_size = fnum(h23.get("artifact_size_mb"))
    l1_size = fnum(l1.get("artifact_size_mb"))
    h23_latency = fnum(h23.get("mean_latency_ms_per_window"))
    l1_latency = fnum(l1.get("mean_latency_ms_per_window"))
    size_improvement = (h23_size - l1_size) / h23_size if h23_size > 0 else 0.0
    latency_improvement = (h23_latency - l1_latency) / h23_latency if h23_latency > 0 else 0.0
    if max(size_improvement, latency_improvement) >= 0.30:
        selected = "L1"
        reason = "L1 benchmark shows >=30% size or latency advantage."
    else:
        selected = "H2.3"
        reason = "L1 benchmark does not show a meaningful deployment-lite advantage."
    return {
        "selected_profile": selected,
        "fallback_profile": "H2.3",
        "reason": reason,
        "lite_status": l1.get("status"),
        "size_improvement": size_improvement,
        "latency_improvement": latency_improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guardrail-summary", default="results/h8_c4_guardrail_audit_20260626/h8_c4_guardrail_summary.json")
    parser.add_argument("--feature-schema", default="results/feature_schema_validation_h8_formal_c4_rescue_20260626/feature_schema_validation.json")
    parser.add_argument("--equivalence-summary", default="results/equivalence_h8_formal_c4_rescue_candidate_20260626/equivalence_summary.json")
    parser.add_argument("--benchmark-summary", default="results/runtime_profile_benchmark_20260626/profile_summary.csv")
    parser.add_argument("--output-dir", default="results/target_profile_selector_20260626")
    args = parser.parse_args()

    guardrail = read_json(Path(args.guardrail_summary))
    schema = read_json(Path(args.feature_schema))
    equivalence = read_json(Path(args.equivalence_summary))
    benchmark_rows = read_csv_rows(Path(args.benchmark_summary))

    guardrail_pass = guardrail.get("guardrail_status") == "pass"
    schema_pass = schema.get("status") == "pass"
    parity_pass = int(equivalence.get("num_mismatch", -1)) == 0 and fnum(equivalence.get("max_abs_diff"), 999.0) < 1e-6
    h8c4_valid = guardrail_pass and schema_pass and parity_pass

    lite = lite_decision(benchmark_rows)
    h8c4_benchmark = row_by_profile(benchmark_rows, "H8+C4")
    h23_benchmark = row_by_profile(benchmark_rows, "H2.3")
    selected = {
        "selector_version": "target_profile_selector.v1",
        "generated_from": {
            "guardrail_summary": args.guardrail_summary,
            "feature_schema": args.feature_schema,
            "equivalence_summary": args.equivalence_summary,
            "benchmark_summary": args.benchmark_summary,
        },
        "test_used_for_selection": False,
        "selection_evidence": {
            "guardrail_status": guardrail.get("guardrail_status", "missing"),
            "guardrail_hit_N": guardrail.get("hit_N"),
            "guardrail_false_hit_N": guardrail.get("hit_false_N"),
            "guardrail_nonCO_hit_N": guardrail.get("hit_nonCO_N"),
            "feature_schema_status": schema.get("status", "missing"),
            "runtime_parity_num_mismatch": equivalence.get("num_mismatch"),
            "runtime_parity_max_abs_diff": equivalence.get("max_abs_diff"),
            "benchmark_profiles": {
                row.get("profile", ""): {
                    "status": row.get("status", ""),
                    "artifact_size_mb": row.get("artifact_size_mb", ""),
                    "mean_latency_ms_per_window": row.get("mean_latency_ms_per_window", ""),
                    "auto_output_field_present": row.get("auto_output_field_present", ""),
                }
                for row in benchmark_rows
            },
        },
        "profiles": {
            "balanced": {
                "selected_profile": "H2.3",
                "fallback_profile": "R3aK16/B0",
                "reason": "H2.3 remains the balanced no-QC full-set mainline.",
                "runtime_benchmark_status": h23_benchmark.get("status", "missing"),
            },
            "co_priority": {
                "selected_profile": "H8_plus_formal_C4_route_rescue" if h8c4_valid else "H2.3",
                "fallback_profile": "H2.3",
                "reason": (
                    "H8+C4 selected because guardrail, feature schema, and runtime parity all pass."
                    if h8c4_valid
                    else "H8+C4 not selected because one or more guardrail/schema/parity checks failed."
                ),
                "guardrail_status": guardrail.get("guardrail_status", "missing"),
                "feature_schema_status": schema.get("status", "missing"),
                "runtime_parity_status": "pass" if parity_pass else "fail",
                "runtime_benchmark_status": h8c4_benchmark.get("status", "missing"),
            },
            "deployment_lite": {
                **lite,
                "runtime_benchmark_required": True,
            },
        },
        "limitations": [
            "H8+C4 route rescue is a high-precision, low-recall C4-specific gate selected from calibration evidence.",
            "Deployment-lite remains pending until L1 has an exported runtime bundle and a clear size or latency advantage.",
            "QC remains a post-hoc reliability layer, not the model capability selector.",
        ],
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected_profiles.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Target Profile Selector",
        "",
        f"- test_used_for_selection: `{selected['test_used_for_selection']}`",
        f"- guardrail_status: `{selected['selection_evidence']['guardrail_status']}`",
        f"- feature_schema_status: `{selected['selection_evidence']['feature_schema_status']}`",
        f"- runtime_parity_num_mismatch: `{selected['selection_evidence']['runtime_parity_num_mismatch']}`",
        "",
        "| mode | selected_profile | fallback | reason |",
        "| --- | --- | --- | --- |",
    ]
    for mode, item in selected["profiles"].items():
        lines.append(
            f"| {mode} | {item['selected_profile']} | {item.get('fallback_profile', '')} | {item.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "Limitations:",
            "- H8+C4 is a guarded CO-priority specialist, not the balanced default.",
            "- L1 is pending until a real deployment bundle proves a size/latency advantage.",
            "",
        ]
    )
    (out / "target_profile_selector_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "selected": selected["profiles"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
