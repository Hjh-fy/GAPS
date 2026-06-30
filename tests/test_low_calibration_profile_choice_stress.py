from run_low_calibration_profile_choice_stress import profile_choice_repeat_rows


def test_profile_choice_repeat_rows_selects_profile_per_client_from_validation_subset():
    h23_val = [
        {"client": "C3", "sample_index": "1", "true_class": "0", "true_ppm": "10", "h23": "10"},
        {"client": "C3", "sample_index": "2", "true_class": "0", "true_ppm": "20", "h23": "20"},
        {"client": "C5", "sample_index": "3", "true_class": "1", "true_ppm": "50", "h23": "70"},
        {"client": "C5", "sample_index": "4", "true_class": "1", "true_ppm": "60", "h23": "80"},
    ]
    h8_val = [
        {"client": "C3", "sample_index": "1", "true_class": "0", "true_ppm": "10", "h8": "30"},
        {"client": "C3", "sample_index": "2", "true_class": "0", "true_ppm": "20", "h8": "40"},
        {"client": "C5", "sample_index": "3", "true_class": "1", "true_ppm": "50", "h8": "50"},
        {"client": "C5", "sample_index": "4", "true_class": "1", "true_ppm": "60", "h8": "60"},
    ]
    h23_test = [
        {"client": "C3", "sample_index": "5", "true_class": "0", "true_ppm": "10", "h23": "10"},
        {"client": "C5", "sample_index": "6", "true_class": "1", "true_ppm": "50", "h23": "70"},
    ]
    h8_test = [
        {"client": "C3", "sample_index": "5", "true_class": "0", "true_ppm": "10", "h8": "30"},
        {"client": "C5", "sample_index": "6", "true_class": "1", "true_ppm": "50", "h8": "50"},
    ]

    selections, metrics = profile_choice_repeat_rows(
        route="demo",
        h23_val_rows=h23_val,
        h8_val_rows=h8_val,
        h23_test_rows=h23_test,
        h8_test_rows=h8_test,
        target_clients=["C3", "C5"],
        budget=2,
        repeat=0,
        seed=5,
        h23_key="h23",
        h8_key="h8",
    )

    assert [(row["client"], row["selected_profile"]) for row in selections] == [
        ("C3", "H2.3+"),
        ("C5", "H8+C4"),
    ]
    all_row = next(row for row in metrics if row["scope"] == "ALL")
    assert all_row["RMSE"] == 0.0
