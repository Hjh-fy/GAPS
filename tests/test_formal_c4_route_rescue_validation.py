from run_formal_c4_route_rescue_selector import apply_gate_to_predictions


def test_apply_gate_to_predictions_writes_formal_ppm_for_validation_rows():
    rows = [
        {
            "client": "C4",
            "split": "calibration",
            "pred_class": 0,
            "final_ppm": 10.0,
            "risk_score": 7.0,
            "confidence_margin": 0.5,
            "response_phase": "recovery",
            "h8_pred_co_source_aug_else_h23_ppm": 40.0,
        },
        {
            "client": "C3",
            "split": "calibration",
            "pred_class": 0,
            "final_ppm": 10.0,
            "risk_score": 7.0,
            "confidence_margin": 0.5,
            "response_phase": "recovery",
            "h8_pred_co_source_aug_else_h23_ppm": 41.0,
        },
    ]
    gate = {
        "pred_classes": "0",
        "phase": "any",
        "max_final": 20.0,
        "min_risk": 6.0,
        "max_conf_margin": 1.0,
        "rescue_ppm": 250.0,
    }

    out = apply_gate_to_predictions(rows, gate)

    assert out[0]["formal_c4_route_rescue_ppm"] == 250.0
    assert out[0]["c4_route_rescue_upper_hit"] == 1
    assert out[1]["formal_c4_route_rescue_ppm"] == 41.0
    assert out[1]["c4_route_rescue_upper_hit"] == 0
