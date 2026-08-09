from scripts.build_iotj_final_scientific_validation_report import final_recommendation


def test_final_recommendation_requires_complete_matrix_and_no_strict_collapse():
    assert final_recommendation(True, True, False) == "READY_FOR_MANUSCRIPT_FREEZE"
    assert final_recommendation(False, True, False) == "NOT_READY"
    assert final_recommendation(True, False, False) == "NOT_READY"
    assert final_recommendation(True, True, True) == "NOT_READY"
