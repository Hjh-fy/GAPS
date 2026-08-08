from scripts.run_iotj_canonical_v1_r84 import expected_counts, special_slices


def test_canonical_r84_uses_frozen_actual_calibration_counts() -> None:
    assert expected_counts("C3") == {
        "calibration": 678,
        "test": 2677,
        "per_class": {0: 170, 1: 168, 2: 170, 3: 170},
    }
    assert expected_counts("C4")["calibration"] == 320
    assert expected_counts("C5")["test"] == 1360


def test_c5_methane_225_repeat1_slice_uses_gas_identity_not_numeric_class() -> None:
    rows = [
        {
            "gas": "methane",
            "true_class": 3,
            "true_ppm": 225.0,
            "repeat_id": 1,
            "pred_84d_h1_ppm": 220.0,
            "route_correct": 1,
        }
    ]
    result = special_slices("C5", rows)
    assert result[0]["slice"] == "methane_225ppm_repeat1"
    assert result[0]["N"] == 1
