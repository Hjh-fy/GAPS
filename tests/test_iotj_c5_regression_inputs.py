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
        "deployment_risk_classifier_entropy": 0.1,
        "deployment_risk_margin": 0.3,
        "deployment_risk_route_response": 0.05,
        "deployment_risk_composite": 0.3,
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


def test_deployment_risk_schema_is_truth_invariant_and_not_legacy_aliased() -> None:
    first = base_record()
    second = dict(first)
    second["true_class"] = 3
    second["true_ppm"] = 250.0

    first_out = convert_pipeline_record(first, "test")
    second_out = convert_pipeline_record(second, "test")
    risk_keys = sorted(key for key in first_out if key.startswith("deployment_risk_"))

    assert risk_keys
    assert {key: first_out[key] for key in risk_keys} == {
        key: second_out[key] for key in risk_keys
    }
    assert "risk_score" not in first_out
    assert first_out["legacy_true_range_composite_risk"] == 0.2
