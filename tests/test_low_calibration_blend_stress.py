from run_low_calibration_blend_stress import (
    aggregate_metric_rows,
    sample_rows_by_client,
    stress_repeat_rows,
)


def test_sample_rows_by_client_is_deterministic_and_caps_to_available_rows():
    rows = [
        {"client": "C3", "sample_index": "1"},
        {"client": "C3", "sample_index": "2"},
        {"client": "C5", "sample_index": "3"},
    ]

    first = sample_rows_by_client(rows, ["C3", "C5"], budget=5, seed=11)
    second = sample_rows_by_client(rows, ["C3", "C5"], budget=5, seed=11)

    assert first == second
    assert len(first) == 3


def test_stress_repeat_rows_selects_weights_and_reports_test_metrics():
    val_rows = [
        {"client": "C3", "sample_index": "1", "true_class": "0", "true_ppm": "10", "anchor": "20", "candidate": "10"},
        {"client": "C3", "sample_index": "2", "true_class": "0", "true_ppm": "20", "anchor": "30", "candidate": "20"},
        {"client": "C5", "sample_index": "3", "true_class": "1", "true_ppm": "50", "anchor": "55", "candidate": "50"},
        {"client": "C5", "sample_index": "4", "true_class": "1", "true_ppm": "60", "anchor": "65", "candidate": "60"},
    ]
    test_rows = [
        {"client": "C3", "sample_index": "5", "true_class": "0", "true_ppm": "10", "anchor": "20", "candidate": "10"},
        {"client": "C5", "sample_index": "6", "true_class": "1", "true_ppm": "50", "anchor": "55", "candidate": "50"},
    ]

    selections, metrics = stress_repeat_rows(
        route="demo",
        val_rows=val_rows,
        test_rows=test_rows,
        target_clients=["C3", "C5"],
        budget=2,
        repeat=0,
        seed=7,
        weight_grid=[0, 1],
        anchor_key="anchor",
        candidate_key="candidate",
        max_nonco_delta=0.0,
        min_all_delta=0.0,
    )

    assert {row["selected_weight"] for row in selections} == {1.0}
    all_row = next(row for row in metrics if row["scope"] == "ALL")
    assert all_row["route"] == "demo"
    assert all_row["budget_per_client"] == 2
    assert all_row["repeat"] == 0
    assert all_row["RMSE"] == 0.0


def test_aggregate_metric_rows_reports_mean_std_and_worst_case():
    rows = [
        {"route": "demo", "budget_per_client": 2, "scope": "ALL", "RMSE": 1.0, "NRMSE": 0.1},
        {"route": "demo", "budget_per_client": 2, "scope": "ALL", "RMSE": 3.0, "NRMSE": 0.3},
    ]

    agg = aggregate_metric_rows(rows)

    assert agg == [
        {
            "route": "demo",
            "budget_per_client": 2,
            "scope": "ALL",
            "repeats": 2,
            "RMSE_mean": 2.0,
            "RMSE_std": 1.0,
            "RMSE_min": 1.0,
            "RMSE_max": 3.0,
            "NRMSE_mean": 0.2,
            "NRMSE_std": 0.09999999999999999,
            "NRMSE_min": 0.1,
            "NRMSE_max": 0.3,
        }
    ]
