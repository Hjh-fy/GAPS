from __future__ import annotations

from pathlib import Path

from scripts.audit_iotj_canonical_v1_scientific_validation import (
    CANONICAL_DATASET_HASH,
    audit_existing_evidence,
    validate_missing_run_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase0_marks_legacy_comparators_noncanonical() -> None:
    audit = audit_existing_evidence(ROOT)
    assert audit["FedAvg"]["status"] == "CANONICAL_COMPARATOR_MISSING"
    assert audit["FedProx"]["status"] == "CANONICAL_COMPARATOR_MISSING"
    assert audit["SCAFFOLD"]["status"] == "CANONICAL_COMPARATOR_MISSING"
    assert audit["MMD"]["status"] == "CANONICAL_COMPARATOR_MISSING"
    assert audit["A0T"]["status"] == "BLOCKED_NOT_RUN"
    assert audit["GAPS/A4"]["status"] == "CANONICAL_COMPLETE"


def test_phase0_accepts_only_matching_canonical_hash() -> None:
    audit = audit_existing_evidence(ROOT)
    assert audit["GAPS/A4"]["dataset_hash"] == CANONICAL_DATASET_HASH
    assert audit["FedAvg"]["observed_dataset"] != "iotj_canonical_v1"


def test_missing_run_matrix_is_minimal_and_unique() -> None:
    rows = validate_missing_run_matrix(ROOT)
    ids = [row["experiment_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert {row["method"] for row in rows} == {
        "FedAvg", "FedProx", "SCAFFOLD", "MMD", "A0T", "STRICT_A4_R84"
    }
    assert all(row["seed"] == 42 for row in rows)
    assert all(row["target_test_selection"] is False for row in rows)
