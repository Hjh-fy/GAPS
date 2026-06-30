import pytest

from run_profile_qc_coverage_audit import (
    best_profile_rows,
    build_client_hybrid_rows,
    build_profile_rows,
    coverage_sweep_rows,
    post_qc_metric_rows,
)


def test_build_profile_rows_merges_predictions_by_client_split_sample_index():
    qc_rows = [
        {"client": "C3", "split": "test", "sample_index": "0", "true_class": "1", "true_ppm": "100", "qc_decision": "accept"},
    ]
    pred_rows = [
        {"client": "C3", "split": "test", "sample_index": "0", "model_ppm": "99.5"},
    ]

    rows = build_profile_rows(qc_rows, pred_rows, "model_ppm", "profile_ppm")

    assert rows[0]["profile_ppm"] == 99.5
    assert rows[0]["qc_decision"] == "accept"


def test_build_profile_rows_raises_when_prediction_key_is_missing():
    qc_rows = [{"client": "C3", "split": "test", "sample_index": "0"}]

    with pytest.raises(ValueError, match="Missing profile predictions"):
        build_profile_rows(qc_rows, [], "model_ppm", "profile_ppm")


def test_build_client_hybrid_rows_uses_client_specific_prediction_sources():
    qc_rows = [
        {"client": "C3", "split": "test", "sample_index": "0", "true_class": "1", "true_ppm": "100"},
        {"client": "C5", "split": "test", "sample_index": "0", "true_class": "1", "true_ppm": "100"},
    ]
    predictions_by_client = {
        "C3": ([{"client": "C3", "split": "test", "sample_index": "0", "h23p_ppm": "101"}], "h23p_ppm"),
        "C5": ([{"client": "C5", "split": "test", "sample_index": "0", "h8_ppm": "99"}], "h8_ppm"),
    }

    rows = build_client_hybrid_rows(qc_rows, predictions_by_client, "profile_ppm")

    assert [row["profile_ppm"] for row in rows] == [101.0, 99.0]


def test_post_qc_metric_rows_reports_accepted_and_nonreject_coverage():
    rows = [
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_decision": "accept", "profile_ppm": "100"},
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_decision": "review", "profile_ppm": "110"},
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_decision": "reject", "profile_ppm": "200"},
    ]

    metrics = post_qc_metric_rows(rows, "demo", "profile_ppm", ["C3"])
    row = next(item for item in metrics if item["scope"] == "C3")

    assert row["accepted_N"] == 1
    assert row["nonreject_N"] == 2
    assert row["accepted_coverage"] == 1 / 3
    assert row["coverage_review"] == 2 / 3
    assert row["coverage_review_RMSE"] < row["full_RMSE"]


def test_coverage_sweep_rows_selects_lowest_risk_windows_per_client():
    rows = [
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_risk_value": "0.1", "profile_ppm": "100"},
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_risk_value": "0.2", "profile_ppm": "110"},
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_risk_value": "9.0", "profile_ppm": "200"},
        {"client": "C3", "true_class": "1", "true_ppm": "100", "qc_risk_value": "10.0", "profile_ppm": "220"},
    ]

    sweep = coverage_sweep_rows(rows, "demo", "profile_ppm", [0.5], by_client=True)

    assert sweep[0]["client"] == "C3"
    assert sweep[0]["N"] == 2
    assert sweep[0]["threshold"] == 0.2
    assert sweep[0]["RMSE"] < 10.0


def test_best_profile_rows_orders_percent_coverage_numerically():
    rows = [
        {"client": "C3", "target_coverage": "100%", "profile": "a", "RMSE": 2.0, "NRMSE": 2.0},
        {"client": "C3", "target_coverage": "75%", "profile": "a", "RMSE": 1.0, "NRMSE": 1.0},
    ]

    best = best_profile_rows(rows, ["client", "target_coverage"])

    assert [row["target_coverage"] for row in best] == ["75%", "100%"]
