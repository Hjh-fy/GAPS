import pytest

from scripts.build_iotj_c5_regression_inputs import convert_pipeline_record


def base_record(client: str = "C5") -> dict[str, object]:
    return {
        "client": client,
        "client_id": 5,
        "row_id": 7,
        "true_class": 1,
        "pred_cls": 1,
        "route_cls": 1,
        "true_ppm": 100.0,
        "base_raw_ppm": 90.0,
        "final_calibrated_ppm": 95.0,
        "class_confidence": 0.9,
        "class_margin": 0.7,
        "composite_response_risk": 0.2,
        "classifier_entropy_risk": 0.1,
        "route_response_risk": 0.05,
        "phase": 2,
    }


def test_convert_pipeline_record_builds_c5_target_head_contract() -> None:
    row = convert_pipeline_record(base_record(), "calibration")

    assert row["client"] == "C5"
    assert row["split"] == "calibration"
    assert row["sample_index"] == 7
    assert row["pred_class"] == 1
    assert row["route_correct"] == 1
    assert row["final_ppm"] == 95.0
    assert row["confidence"] == 0.9
    assert row["confidence_margin"] == 0.7


def test_convert_pipeline_record_rejects_non_c5_target() -> None:
    with pytest.raises(ValueError, match="only C5"):
        convert_pipeline_record(base_record("C4"), "test")
