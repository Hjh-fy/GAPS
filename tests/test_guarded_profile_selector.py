from run_guarded_profile_selector import apply_guarded_profile_choices, guarded_profile_choices


def test_guarded_profile_choices_require_margin_and_stability_before_switching():
    h23_val = [
        {"client": "C4", "sample_index": "1", "true_class": "0", "true_ppm": "0", "h23": "10.0"},
        {"client": "C5", "sample_index": "2", "true_class": "0", "true_ppm": "0", "h23": "10.0"},
        {"client": "C5", "sample_index": "3", "true_class": "1", "true_ppm": "0", "h23": "10.0"},
    ]
    h8_val = [
        {"client": "C4", "sample_index": "1", "true_class": "0", "true_ppm": "0", "h8": "9.6"},
        {"client": "C5", "sample_index": "2", "true_class": "0", "true_ppm": "0", "h8": "9.0"},
        {"client": "C5", "sample_index": "3", "true_class": "1", "true_ppm": "0", "h8": "9.0"},
    ]
    stability_rows = [
        {"route": "oracle-route", "budget_per_client": "96", "client": "C4", "H8_C4_rate": "1.0"},
        {"route": "oracle-route", "budget_per_client": "96", "client": "C5", "H8_C4_rate": "0.8"},
    ]

    choices = guarded_profile_choices(
        h23_val,
        h8_val,
        ["C4", "C5"],
        route="oracle-route",
        stability_rows=stability_rows,
        stability_budget=96,
        h23_key="h23",
        h8_key="h8",
        min_rmse_margin=0.5,
        max_nrmse_delta=0.0,
        max_nonco_delta=0.0,
        min_h8_stability=0.7,
    )

    by_client = {row["client"]: row for row in choices}
    assert by_client["C4"]["selected_profile"] == "H2.3+"
    assert by_client["C4"]["passes_rmse_margin"] == 0
    assert by_client["C4"]["passes_stability"] == 1
    assert by_client["C5"]["selected_profile"] == "H8+C4"
    assert by_client["C5"]["passes_rmse_margin"] == 1
    assert by_client["C5"]["passes_stability"] == 1


def test_guarded_profile_choices_reject_unstable_h8_candidate():
    h23_val = [{"client": "C3", "sample_index": "1", "true_class": "0", "true_ppm": "0", "h23": "10.0"}]
    h8_val = [{"client": "C3", "sample_index": "1", "true_class": "0", "true_ppm": "0", "h8": "8.0"}]
    stability_rows = [{"route": "real-route", "budget_per_client": "96", "client": "C3", "H8_C4_rate": "0.6"}]

    choices = guarded_profile_choices(
        h23_val,
        h8_val,
        ["C3"],
        route="real-route",
        stability_rows=stability_rows,
        stability_budget=96,
        h23_key="h23",
        h8_key="h8",
        min_rmse_margin=0.5,
        max_nrmse_delta=0.0,
        max_nonco_delta=0.0,
        min_h8_stability=0.7,
    )

    assert choices[0]["selected_profile"] == "H2.3+"
    assert choices[0]["passes_rmse_margin"] == 1
    assert choices[0]["passes_stability"] == 0


def test_apply_guarded_profile_choices_uses_selected_profile_per_client():
    h23_test = [
        {"client": "C4", "sample_index": "4", "true_class": "0", "true_ppm": "0", "h23": "10"},
        {"client": "C5", "sample_index": "5", "true_class": "1", "true_ppm": "0", "h23": "20"},
    ]
    h8_test = [
        {"client": "C4", "sample_index": "4", "true_class": "0", "true_ppm": "0", "h8": "11"},
        {"client": "C5", "sample_index": "5", "true_class": "1", "true_ppm": "0", "h8": "7"},
    ]
    choices = [
        {"client": "C4", "selected_profile": "H2.3+"},
        {"client": "C5", "selected_profile": "H8+C4"},
    ]

    rows = apply_guarded_profile_choices(
        h23_test,
        h8_test,
        choices,
        h23_key="h23",
        h8_key="h8",
        output_key="guarded_ppm",
    )

    assert [(row["client"], row["selected_profile"], row["guarded_ppm"]) for row in rows] == [
        ("C4", "H2.3+", 10.0),
        ("C5", "H8+C4", 7.0),
    ]
