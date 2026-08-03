"""Strict audit gates for the frozen IoT-J final classification suite."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from gaps_flower.strategy import selective_aggregation_phase
from scripts.run_iotj_final_classification_le1 import (
    DOC_ROOT,
    IMPORTED_CHECKPOINT,
    INPUT_MANIFEST,
    LOCAL_DATA_ROOT,
    MATRIX_PATH,
    NEW_FULL_FL_IDS,
    PROTOCOL_PATH,
    RESULT_ROOT,
    TARGET_POLICY_PATH,
    build_e2_spec,
    build_flower_commands,
    current_protocol_hash,
    execution_counts,
    load_registered_matrix,
)


def audit_training_commands(commands: dict[str, list[str]]) -> dict:
    offenders = []
    for role, argv in commands.items():
        if not isinstance(argv, list):
            continue
        for token in argv:
            lowered = str(token).replace("\\", "/").lower()
            if "target-test" in lowered or "test_features" in lowered or "/test_" in lowered:
                offenders.append({"role": role, "token": str(token)})
    return {
        "check_id": "TARGET-TEST-COMMAND-SEAL",
        "passed": not offenders,
        "severity": "blocking",
        "details": {"offenders": offenders},
    }


def summarize_findings(findings: Iterable[dict]) -> dict:
    rows = list(findings)
    blocking = [
        str(row["check_id"])
        for row in rows
        if row.get("severity") == "blocking" and not row.get("passed", False)
    ]
    return {
        "status": "PASS" if not blocking else "FAIL",
        "blocking_failures": blocking,
        "checks_total": len(rows),
        "checks_passed": sum(bool(row.get("passed")) for row in rows),
    }


def _finding(check_id: str, passed: bool, details, *, severity: str = "blocking") -> dict:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "severity": severity,
        "details": details,
    }


def pre_run_findings() -> list[dict]:
    findings: list[dict] = []
    rows = load_registered_matrix(MATRIX_PATH)
    counts = execution_counts(rows)
    findings.append(
        _finding(
            "MATRIX-BUDGET",
            counts
            == {
                "registered_configs": 21,
                "new_full_fl_runs": 10,
                "e2_adaptation_branches": 9,
            },
            counts,
        )
    )
    protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8") if PROTOCOL_PATH.is_file() else ""
    required_protocol_terms = (
        "optimizer_lr",
        "canonical SCAFFOLD implementation",
        "rounds 1 through 5",
        "round 6 onward",
        "ordered state-content fingerprint",
        "source_target_f1_gap",
    )
    findings.append(
        _finding(
            "PROTOCOL-DISCLOSURE",
            all(term in protocol_text for term in required_protocol_terms),
            {"required_terms": list(required_protocol_terms)},
        )
    )
    policy_text = (
        TARGET_POLICY_PATH.read_text(encoding="utf-8")
        if TARGET_POLICY_PATH.is_file()
        else ""
    )
    findings.append(
        _finding(
            "TARGET-POLICY-METHOD-SPECIFIC",
            all(
                term in policy_text
                for term in (
                    "E2 CORAL",
                    "E2 MMD",
                    "E2 DANN",
                    "E3 Full GAPS / E4 A4-A6",
                    "Any train/adapt/stop/select stage | target test",
                )
            ),
            {"policy": str(TARGET_POLICY_PATH)},
        )
    )
    import_payload = {}
    if INPUT_MANIFEST.is_file():
        import_payload = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    findings.append(
        _finding(
            "P0A-ORDERED-CONTENT-EQUALITY",
            bool(
                IMPORTED_CHECKPOINT.is_file()
                and import_payload.get("equality_verified") is True
                and import_payload.get("equality_basis")
                == "ordered_state_content_fingerprint"
                and import_payload.get("formal_round") == 25
            ),
            import_payload or {"manifest": "missing"},
        )
    )
    source_gate_path = RESULT_ROOT / "preflight/scaffold_source_numerical_gate.json"
    source_gate = (
        json.loads(source_gate_path.read_text(encoding="utf-8"))
        if source_gate_path.is_file()
        else {}
    )
    findings.append(
        _finding(
            "SCAFFOLD-SOURCE-ONLY-NUMERICAL-GATE",
            bool(
                source_gate.get("passed") is True
                and source_gate.get("lr_search_performed") is False
                and source_gate.get("target_information_accessed") is False
            ),
            source_gate or {"report": "missing"},
        )
    )
    shared_expected_files = (
        "calibration_features.npy",
        "test_features.npy",
        "test_classification_labels.npy",
    )
    dataset_missing = [
        str(LOCAL_DATA_ROOT / f"client_{client_id}" / name)
        for client_id in range(1, 6)
        for name in shared_expected_files
        if not (LOCAL_DATA_ROOT / f"client_{client_id}" / name).is_file()
    ]
    dataset_missing.extend(
        str(LOCAL_DATA_ROOT / f"client_{client_id}" / name)
        for client_id in (1, 2)
        for name in ("train_features.npy", "train_classification_labels.npy")
        if not (LOCAL_DATA_ROOT / f"client_{client_id}" / name).is_file()
    )
    findings.append(
        _finding(
            "DATASET-C1-C5-COMPLETE",
            not dataset_missing,
            {"data_root": str(LOCAL_DATA_ROOT), "missing": dataset_missing},
        )
    )
    phase_map = {
        round_idx: selective_aggregation_phase(round_idx, warmup=5)
        for round_idx in range(1, 7)
    }
    findings.append(
        _finding(
            "SELECTIVE-WARMUP-BOUNDARY",
            all(phase_map[index] == "fedavg_warmup" for index in range(1, 6))
            and phase_map[6] == "selective",
            phase_map,
        )
    )
    expected_activity_columns = {
        "loss_name",
        "configured_weight",
        "input_available",
        "active_steps",
        "mean_raw_loss",
        "mean_weighted_loss",
        "inactive_reason",
    }
    from gaps_flower.loss_activity import LossActivityAccumulator

    accumulator = LossActivityAccumulator(scope="audit", variant="A0")
    accumulator.record(
        loss_name="source_ce",
        configured_weight=1.0,
        input_available=True,
        raw_loss=1.0,
        active=True,
        inactive_reason="",
    )
    actual_columns = set(accumulator.rows()[0])
    findings.append(
        _finding(
            "ABLATION-LOSS-ACTIVITY-SCHEMA",
            expected_activity_columns <= actual_columns,
            {"columns": sorted(actual_columns)},
        )
    )
    command_findings = []
    for experiment_id in sorted(NEW_FULL_FL_IDS):
        payload = build_flower_commands(experiment_id)
        command_findings.append(
            audit_training_commands(
                {key: payload[key] for key in ("server", "client_c1", "client_c2")}
            )
        )
    offenders = [row for row in command_findings if not row["passed"]]
    findings.append(
        _finding(
            "ALL-FL-COMMANDS-TARGET-TEST-SEALED",
            not offenders,
            {"offenders": offenders},
        )
    )
    e2_specs = [
        build_e2_spec(row["experiment_id"])
        for row in rows
        if row["experiment_id"].startswith("FCL-E2-")
    ]
    findings.append(
        _finding(
            "E2-CANONICAL-X-ONLY",
            len(e2_specs) == 9
            and all(
                spec["target_fields"] == ["x"]
                and spec["steps"] == 100
                and spec["target_ce"] is False
                and spec["conditional"] is False
                and spec["hyperparameter_search"] is False
                for spec in e2_specs
            ),
            e2_specs,
        )
    )
    locked_specs = RESULT_ROOT / "locked_execution_specs.json"
    locked_payload = (
        json.loads(locked_specs.read_text(encoding="utf-8"))
        if locked_specs.is_file()
        else {}
    )
    current_hash = current_protocol_hash()
    findings.append(
        _finding(
            "LOCKED-SPECS-MATCH-PROTOCOL",
            locked_payload.get("protocol_hash") == current_hash,
            {
                "expected": current_hash,
                "observed": locked_payload.get("protocol_hash"),
            },
        )
    )
    return findings


def post_run_findings() -> list[dict]:
    findings = pre_run_findings()
    rows = load_registered_matrix(MATRIX_PATH)
    missing_markers = [
        row["experiment_id"]
        for row in rows
        if not (RESULT_ROOT / row["experiment_id"] / "fixed_endpoint_complete.json").is_file()
    ]
    findings.append(
        _finding(
            "ALL-FIXED-ENDPOINTS-COMPLETE",
            not missing_markers,
            {"missing": missing_markers},
        )
    )
    required_outputs = (
        "classification_main_comparison.csv",
        "ablation_loss_activity.csv",
        "source_target_f1_gap.csv",
        "RESULT_ANALYSIS.md",
        "EXPERIMENT_AUDIT.md",
        "sha256_index.json",
    )
    missing_outputs = [name for name in required_outputs if not (RESULT_ROOT / name).is_file()]
    findings.append(
        _finding(
            "FINAL-EVIDENCE-PACKAGE",
            not missing_outputs,
            {"missing": missing_outputs},
        )
    )
    return findings


def write_report(stage: str, findings: list[dict]) -> dict:
    summary = summarize_findings(findings)
    payload = {
        "schema_version": "iotj.final_classification.audit.v1",
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "findings": findings,
    }
    output = RESULT_ROOT / "preflight" if stage == "pre-run" else RESULT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{stage.replace('-', '_')}_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre-run", "post-run"), required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    findings = pre_run_findings() if args.stage == "pre-run" else post_run_findings()
    report = write_report(args.stage, findings)
    print(json.dumps(report["summary"], sort_keys=True))
    if args.strict and report["summary"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
