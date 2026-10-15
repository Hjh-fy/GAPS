import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_iotj_v4_final_pre_run import (
    GROUPING_KEY,
    INFORMATION_NOTE,
    POOLED_RULE,
    audit_alpha_rule,
    audit_budget_manifest,
    audit_comparator_population,
    audit_exact_hash,
    audit_qc_provenance,
    audit_schema_hashes,
    build_candidate_registry,
    final_verdict,
    write_outputs,
)


EXPECTED_INFORMATION_NOTE = (
    "Cross-regime numbers are reported for contextual comparison and must not "
    "be interpreted as an equal-supervision ranking."
)


def _budget_rows():
    rows = {}
    for pct, count in ((20, 320), (15, 240), (10, 160), (5, 80)):
        rows[pct] = [
            {
                "canonical_identity": f"C5|file-{index // 4:03d}|{index}",
                "raw_filename": f"file-{index // 4:03d}",
                "gas": ("ethanol", "carbon_monoxide", "ethylene", "methane")[
                    (index // 80) % 4
                ],
            }
            for index in range(count)
        ]
    return rows


def _test_rows():
    gas_codes = ("Ea", "CO", "Ey", "Me")
    return [
        {
            "physical_identity": f"C5|test-{gas}-{file_index:02d}|{window}",
            "filename": f"test-{gas}-{file_index:02d}.txt",
            "gas_code": gas,
        }
        for gas in gas_codes
        for file_index in range(20)
        for window in range(17)
    ]


def test_protocol_strings_are_literal_scope_contract():
    assert POOLED_RULE == (
        "concatenate C3/C4/C5 predictions, then compute one Accuracy and one global Macro-F1"
    )
    assert INFORMATION_NOTE == EXPECTED_INFORMATION_NOTE
    assert GROUPING_KEY == "C5|filename"


def test_unknown_critical_field_blocks_only_affected_component():
    verdict = final_verdict(
        {"budget_grouping_key": "unknown", "classification": "PASS"}
    )
    assert verdict["regression_budget"] == "BLOCKED"
    assert verdict["classification"] == "PASS"


def test_unknown_qc_does_not_block_other_components():
    verdict = final_verdict(
        {
            "classification": "PASS",
            "regression_budget": "PASS",
            "qc_authority": "unknown",
        }
    )
    assert verdict == {
        "classification": "PASS",
        "regression_budget": "PASS",
        "qc": "BLOCKED",
    }


@pytest.mark.parametrize("name", ["CORAL", "DANN"])
def test_mismatched_test_population_excludes_comparator(name):
    item = audit_comparator_population(
        name,
        {"C3": 680, "C4": 320, "C5": 1360},
        {"C3": 2677, "C4": 1360, "C5": 1360},
    )
    assert item["status"] == "EXCLUDED_NONCANONICAL"
    assert item["retrain"] is False
    assert item["replacement"] is False


def test_comparator_population_requires_exact_target_set():
    item = audit_comparator_population(
        "CORAL", {"C3": 2677, "C4": 1360}, {"C3": 2677, "C4": 1360, "C5": 1360}
    )
    assert item["status"] == "EXCLUDED_NONCANONICAL"


def test_budget_manifest_passes_counts_nesting_overlap_and_groups():
    result = audit_budget_manifest(_budget_rows(), _test_rows())
    assert result["status"] == "PASS"
    assert result["counts"] == {20: 320, 15: 240, 10: 160, 5: 80}
    assert result["nested"] is True
    assert result["calibration_test_exact_identity_overlap"] == 0
    assert result["test_filename_groups"] == 80
    assert result["test_filename_groups_per_gas"] == {
        "CO": 20,
        "Ea": 20,
        "Ey": 20,
        "Me": 20,
    }


@pytest.mark.parametrize("failure", ["count", "nesting", "overlap", "blank_group", "gas_groups"])
def test_budget_manifest_fail_closes_each_identity_gate(failure):
    budgets = _budget_rows()
    tests = _test_rows()
    if failure == "count":
        budgets[5] = budgets[5][:-1]
    elif failure == "nesting":
        budgets[5][0] = {**budgets[5][0], "canonical_identity": "C5|not-in-10"}
    elif failure == "overlap":
        tests[0] = {**tests[0], "physical_identity": budgets[5][0]["canonical_identity"]}
    elif failure == "blank_group":
        tests[0] = {**tests[0], "filename": ""}
    else:
        tests[0] = {**tests[0], "gas_code": "CO"}
    result = audit_budget_manifest(budgets, tests)
    assert result["status"] == "BLOCKED"
    assert result["failed_checks"]


def test_schema_hashes_require_exact_83d_and_104d_contract():
    expected = {"83D": "a" * 64, "104D": "b" * 64}
    assert audit_schema_hashes(expected, expected)["status"] == "PASS"
    result = audit_schema_hashes({**expected, "104D": "c" * 64}, expected)
    assert result["status"] == "BLOCKED"
    assert result["failed_checks"] == ["104D"]


def test_immutable_hash_gate_rejects_mismatch_and_malformed_digest():
    assert audit_exact_hash("a" * 64, "a" * 64, "R0-v2 model lock")["status"] == "PASS"
    mismatch = audit_exact_hash("a" * 64, "b" * 64, "R0-v2 model lock")
    assert mismatch["status"] == "BLOCKED"
    assert mismatch["failed_checks"] == ["R0-v2 model lock"]
    assert audit_exact_hash("not-a-hash", "a" * 64, "R1 test identity")["status"] == "BLOCKED"


def test_alpha_rule_is_exact_and_mismatch_blocks():
    exact = {
        "grid": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "folds": 5,
        "grouping": "raw_filename",
        "objective": "pooled RMSE",
        "tie_break": "first-in-grid",
        "selection_data": "calibration-only",
    }
    assert audit_alpha_rule(exact)["status"] == "PASS"
    assert audit_alpha_rule({**exact, "selection_data": "test"})["status"] == "BLOCKED"


def test_qc_is_confidence_only_and_authority_compatible():
    result = audit_qc_provenance(
        q1={
            "backend": "R84_CONCAT",
            "population": "C3+C4+C5 canonical test",
            "risk_definition": "classification confidence",
            "policy": "CONFIDENCE_QC_FINAL",
            "authority": "AUTHORITATIVE",
        },
        metrics={
            "backend": "R84_CONCAT",
            "population": "C3+C4+C5 canonical test",
            "risk_definition": "classification confidence",
            "policy": "CONFIDENCE_QC_FINAL",
            "authority": "AUTHORITATIVE",
        },
        thresholds={
            "backend": "R84_CONCAT",
            "population": "C3+C4+C5 canonical test",
            "risk_definition": "classification confidence",
            "policy": "CONFIDENCE_QC_FINAL",
            "authority": "AUTHORITATIVE",
        },
    )
    assert result["status"] == "PASS"
    assert result["authorized_overlays"] == [
        "FINAL_CONFIDENCE_QC_METRICS.csv",
        "FINAL_CONFIDENCE_QC_THRESHOLDS.csv",
    ]
    assert result["legacy"]["05_qc_final.csv"] == "HISTORICAL_SUPERSEDED"
    assert result["legacy"]["legacy_random_retention"] == "HISTORICAL_SUPERSEDED"


@pytest.mark.parametrize("field", ["backend", "population", "risk_definition", "authority"])
def test_qc_compatibility_fail_closes_on_each_required_field(field):
    base = {
        "backend": "R84_CONCAT",
        "population": "C3+C4+C5 canonical test",
        "risk_definition": "classification confidence",
        "policy": "CONFIDENCE_QC_FINAL",
        "authority": "AUTHORITATIVE",
    }
    result = audit_qc_provenance(
        q1=base,
        metrics={**base, field: "mismatch"},
        thresholds=base,
    )
    assert result["status"] == "BLOCKED"
    assert result["authorized_overlays"] == []
    assert field in result["failed_checks"]


def test_candidate_registry_is_exact_six_reduced_budget_runs():
    rows = build_candidate_registry()
    assert len(rows) == 6
    assert {(row["budget_pct"], row["method"]) for row in rows} == {
        (15, "R83_TARGET_ONLY"),
        (15, "R84_CONCAT"),
        (10, "R83_TARGET_ONLY"),
        (10, "R84_CONCAT"),
        (5, "R83_TARGET_ONLY"),
        (5, "R84_CONCAT"),
    }
    assert all(row["status"] == "REGISTERED_NOT_RUN" for row in rows)
    assert not any(row["budget_pct"] == 20 for row in rows)


def test_write_outputs_is_exclusive_and_contains_five_contract_files(tmp_path):
    root = tmp_path / "formal"
    registry = build_candidate_registry()
    audit = {"status": "PASS", "components": {"classification": "PASS"}}
    created = write_outputs(root, audit, registry, {"asset": "d" * 64})
    assert {path.name for path in created} == {
        "FINAL_PRE_RUN_AUDIT.md",
        "final_pre_run_audit.json",
        "qc_provenance_audit.json",
        "experiment_registry.csv",
        "input_sha256.json",
    }
    with pytest.raises(FileExistsError, match="exists|overwrite|exclusive"):
        write_outputs(root, audit, registry, {"asset": "d" * 64})


def test_invalid_registry_is_rejected_before_output_root_creation(tmp_path):
    root = tmp_path / "formal"
    with pytest.raises(RuntimeError, match="empty experiment registry"):
        write_outputs(root, {"status": "BLOCKED"}, [], {})
    assert not root.exists()


def test_cli_dry_run_is_machine_readable_and_zero_write(tmp_path):
    root = tmp_path / "must-not-exist"
    script = Path(__file__).parents[1] / "scripts" / "audit_iotj_v4_final_pre_run.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--scope-commit",
            "5f554cdfa05af272a4d496c7063f317174bc0d81",
            "--output-root",
            str(root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "dry-run"
    assert payload["status"] in {"PASS", "BLOCKED"}
    assert set(payload["components"]) == {"classification", "regression_budget", "qc"}
    assert payload["writes_performed"] is False
    assert payload["test_release_performed"] is False
    assert payload["training_performed"] is False
    assert not root.exists()
