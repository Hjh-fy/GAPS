from run_h2_3_plus_fusion_profile import (
    apply_client_blends,
    combine_h2_3_rows,
    select_client_blend_weights,
)


def test_select_client_blend_weights_prefers_candidate_when_validation_rmse_improves():
    rows = [
        {"client": "C3", "true_class": 1, "true_ppm": 100.0, "anchor_ppm": 110.0, "candidate_ppm": 100.0},
        {"client": "C3", "true_class": 1, "true_ppm": 150.0, "anchor_ppm": 160.0, "candidate_ppm": 150.0},
        {"client": "C3", "true_class": 0, "true_ppm": 25.0, "anchor_ppm": 30.0, "candidate_ppm": 25.0},
    ]

    selected, audit = select_client_blend_weights(
        rows,
        ["C3"],
        [0.0, 0.5, 1.0],
        anchor_key="anchor_ppm",
        candidate_key="candidate_ppm",
        max_nonco_delta=1.0,
    )

    assert selected == {"C3": 1.0}
    chosen = next(row for row in audit if row["client"] == "C3" and row["selected"] == 1)
    assert chosen["weight"] == 1.0
    assert chosen["passes_guard"] == 1


def test_select_client_blend_weights_keeps_anchor_when_nonco_guard_fails():
    rows = [
        {"client": "C5", "true_class": 1, "true_ppm": 200.0, "anchor_ppm": 240.0, "candidate_ppm": 200.0},
        {"client": "C5", "true_class": 1, "true_ppm": 220.0, "anchor_ppm": 250.0, "candidate_ppm": 220.0},
        {"client": "C5", "true_class": 0, "true_ppm": 25.0, "anchor_ppm": 25.0, "candidate_ppm": 80.0},
        {"client": "C5", "true_class": 2, "true_ppm": 25.0, "anchor_ppm": 25.0, "candidate_ppm": 80.0},
    ]

    selected, audit = select_client_blend_weights(
        rows,
        ["C5"],
        [0.0, 0.5, 1.0],
        anchor_key="anchor_ppm",
        candidate_key="candidate_ppm",
        max_nonco_delta=1.0,
    )

    assert selected == {"C5": 0.0}
    assert all(row["selected"] == int(row["weight"] == 0.0) for row in audit)


def test_apply_client_blends_uses_selected_client_weight():
    rows = [
        {"client": "C3", "anchor_ppm": 100.0, "candidate_ppm": 120.0},
        {"client": "C4", "anchor_ppm": 100.0, "candidate_ppm": 120.0},
    ]

    blended = apply_client_blends(
        rows,
        {"C3": 0.25},
        anchor_key="anchor_ppm",
        candidate_key="candidate_ppm",
        output_key="blend_ppm",
    )

    assert blended[0]["blend_ppm"] == 105.0
    assert blended[0]["blend_weight"] == 0.25
    assert blended[1]["blend_ppm"] == 100.0
    assert blended[1]["blend_weight"] == 0.0


def test_combine_h2_3_rows_keeps_only_matching_client_family_rows():
    gate = {
        "pred_classes": "0",
        "max_ppm": 50.0,
        "risk_threshold": 2.0,
        "phase": "recovery",
        "rescue_ppm": 250.0,
    }
    common = {
        "split": "test",
        "true_class": 1,
        "true_ppm": 100.0,
        "pred_class": 1,
        "final_ppm": 100.0,
        "risk_score": 0.0,
        "response_phase": "main_response",
    }
    c3_rows = [
        {**common, "client": "C3", "sample_index": 0, "h2_c3_mlp_ppm": 101.0},
        {**common, "client": "C4", "sample_index": 0},
    ]
    c4_rows = [
        {**common, "client": "C3", "sample_index": 0},
        {**common, "client": "C4", "sample_index": 0, "h2_c4_ridge_ppm": 99.0},
    ]

    combined = combine_h2_3_rows(
        c3_mlp_rows=c3_rows,
        c4_ridge_rows=c4_rows,
        c5_grid_rows=[],
        gate=gate,
    )

    assert [(row["client"], row["sample_index"]) for row in combined] == [("C3", 0), ("C4", 0)]
    assert [row["h2_3_direct_only_ppm"] for row in combined] == [101.0, 99.0]
