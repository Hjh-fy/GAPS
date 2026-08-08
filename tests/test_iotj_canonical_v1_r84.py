from scripts.run_iotj_canonical_v1_r84 import expected_counts


def test_canonical_r84_uses_frozen_actual_calibration_counts() -> None:
    assert expected_counts("C3") == {
        "calibration": 678,
        "test": 2677,
        "per_class": {0: 170, 1: 168, 2: 170, 3: 170},
    }
    assert expected_counts("C4")["calibration"] == 320
    assert expected_counts("C5")["test"] == 1360
