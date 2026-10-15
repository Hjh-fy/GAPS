from scripts.build_iotj_final_scientific_validation_report import (
    final_recommendation,
    strict_survival_claim_status,
    structured_commissioning_claim_status,
)


def test_final_recommendation_requires_complete_matrix_and_no_strict_collapse():
    assert final_recommendation(True, True, False) == "READY_FOR_MANUSCRIPT_FREEZE"
    assert final_recommendation(False, True, False) == "NOT_READY"
    assert final_recommendation(True, False, False) == "NOT_READY"
    assert final_recommendation(True, True, True) == "NOT_READY"


def test_single_seed_equal_label_comparison_stays_limited():
    assert structured_commissioning_claim_status(
        {"C3": 0.000001, "C4": 0.0029, "C5": -0.00001}, seed_count=1
    ) == "PASS_WITH_LIMITATION"


def test_strict_collapse_blocks_survival_claim():
    assert strict_survival_claim_status(True) == "BLOCKED"
    assert strict_survival_claim_status(False) == "PASS"
