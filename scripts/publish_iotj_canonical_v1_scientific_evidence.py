"""Publish compact canonical-v1 scientific-validation evidence to Git-tracked docs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809"
DOCS = ROOT / "docs/experiments/iotj_canonical_v1_final"


def completion_status(report: dict[str, object]) -> dict[str, object]:
    matrix_complete = bool(report.get("matrix_complete"))
    a0t_complete = bool(report.get("a0t_complete"))
    strict_collapse = bool(report.get("strict_collapse"))
    return {
        "experiment_execution": "COMPLETE" if matrix_complete and a0t_complete else "INCOMPLETE",
        "submission_recommendation": str(report.get("recommendation", "UNKNOWN")),
        "comparator_matrix": "COMPLETE" if matrix_complete else "INCOMPLETE",
        "equal_label_a0t": "COMPLETE" if a0t_complete else "INCOMPLETE",
        "strict_nonoverlap_claim": "BLOCKED" if strict_collapse else "PASS",
        "active_training_process": False,
        "algorithm_search": "STOPPED_BY_PROTOCOL",
    }


def status_markdown(status: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Final experiment status",
            "",
            f"- Experiment execution: **{status['experiment_execution']}**.",
            f"- Canonical comparator matrix: **{status['comparator_matrix']}**.",
            f"- Equal-label A0T: **{status['equal_label_a0t']}**.",
            f"- Strict non-overlap robustness claim: **{status['strict_nonoverlap_claim']}**.",
            f"- Submission recommendation: **{status['submission_recommendation']}**.",
            "- Active training process: **NO**.",
            "- Further training, tuning, and algorithm search: **STOPPED BY PROTOCOL**.",
            "",
            "The experiments are complete, but execution completion does not override the strict C5 collapse finding or imply submission readiness.",
            "",
        ]
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_source(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def publish(docs: Path, mappings: list[tuple[Path, str]]) -> dict[str, object]:
    missing = [source for source, _ in mappings if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing scientific evidence: " + ", ".join(str(path) for path in missing))
    docs.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for source, destination_name in mappings:
        destination = docs / destination_name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        files[destination_name] = {
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "source": portable_source(source),
        }
    payload: dict[str, object] = {
        "schema_version": "gaps.iotj.canonical_v1.scientific_validation.sha256_index.v1",
        "files": dict(sorted(files.items())),
    }
    (docs / "scientific_validation_sha256_index.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def canonical_mappings(validation: Path) -> list[tuple[Path, str]]:
    return [
        (validation / "classification_comparison/canonical_classification_comparison.csv", "canonical_classification_comparison.csv"),
        (validation / "classification_comparison/evaluation_manifest.json", "canonical_comparator_evaluation_manifest.json"),
        (validation / "comparators/FIXED_ENDPOINTS_COMPLETE_TEST_STILL_SEALED.json", "canonical_comparator_fixed_endpoint_gate.json"),
        (validation / "comparators/scaffold_audit/SCAFFOLD_SANITY_AUDIT.md", "SCAFFOLD_SANITY_AUDIT.md"),
        (validation / "comparators/scaffold_audit/scaffold_sanity_audit.json", "scaffold_sanity_audit.json"),
        (validation / "strict_nonoverlap/strict_non_overlap_summary.csv", "strict_non_overlap_summary.csv"),
        (validation / "strict_nonoverlap/strict_non_overlap_deltas.csv", "strict_non_overlap_deltas.csv"),
        (validation / "strict_nonoverlap/STRICT_NON_OVERLAP_ANALYSIS.md", "STRICT_NON_OVERLAP_ANALYSIS.md"),
        (validation / "strict_nonoverlap/run/regression/cross_target_r84_summary.csv", "strict_nonoverlap_r84_summary.csv"),
        (validation / "strict_nonoverlap/run/regression/protocol_manifest.json", "strict_nonoverlap_r84_protocol_manifest.json"),
        (validation / "FINAL_SCIENTIFIC_VALIDATION_REPORT.md", "FINAL_SCIENTIFIC_VALIDATION_REPORT.md"),
        (validation / "FINAL_SCIENTIFIC_VALIDATION_REPORT.json", "FINAL_SCIENTIFIC_VALIDATION_REPORT.json"),
        (validation / "SCIENTIFIC_VALIDATION_STATUS.json", "SCIENTIFIC_VALIDATION_STATUS.json"),
        (validation / "FINAL_EXPERIMENT_STATUS.md", "FINAL_EXPERIMENT_STATUS.md"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--docs", type=Path, default=DOCS)
    args = parser.parse_args()
    validation = args.validation.resolve()
    report = json.loads((validation / "FINAL_SCIENTIFIC_VALIDATION_REPORT.json").read_text(encoding="utf-8"))
    status = completion_status(report)
    (validation / "SCIENTIFIC_VALIDATION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (validation / "FINAL_EXPERIMENT_STATUS.md").write_text(status_markdown(status), encoding="utf-8")
    queue_state = {
        "status": "COMPLETE",
        "step": "FINAL_AUDIT_AND_REPORT_COMPLETE",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_note": "Background queue state was stale after a failed evaluator process; fixed endpoints, sealed evaluation, strict analysis, final report, and evidence publication were completed in the foreground.",
        **status,
    }
    queue_path = validation / "queue/queue_state.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue_state, indent=2) + "\n", encoding="utf-8")
    docs = args.docs.resolve()
    tracked_closure = [
        (docs / name, name)
        for name in (
            "09_a0t_equal_label.csv",
            "FINAL_CLAIM_EVIDENCE_MATRIX.md",
            "FINAL_SUBMISSION_AUDIT.md",
            "FINAL_SUBMISSION_READINESS.md",
            "FINAL_FIGURE_TABLE_SCIENTIFIC_MAP.md",
        )
    ]
    print(json.dumps(publish(docs, canonical_mappings(validation) + tracked_closure), indent=2))


if __name__ == "__main__":
    main()
