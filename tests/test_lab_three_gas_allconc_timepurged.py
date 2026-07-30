from __future__ import annotations

from scripts.lab_three_gas_3class.build_allconcentration_timepurged_dataset import (
    CALIBRATION_INDICES,
    MAIN_INDICES,
    PURGED_INDICES,
)


def test_timepurged_index_contract() -> None:
    assert CALIBRATION_INDICES == (3, 11, 19)
    assert PURGED_INDICES == (2, 4, 10, 12, 18, 20)
    assert len(MAIN_INDICES) == 14
    assert (
        set(CALIBRATION_INDICES)
        | set(PURGED_INDICES)
        | set(MAIN_INDICES)
    ) == set(range(23))


def test_active_windows_do_not_overlap_raw_time() -> None:
    window_s = 100
    stride_s = 50
    for calibration_index in CALIBRATION_INDICES:
        calibration_start = calibration_index * stride_s
        calibration_end = calibration_start + window_s
        for main_index in MAIN_INDICES:
            main_start = main_index * stride_s
            main_end = main_start + window_s
            assert max(calibration_start, main_start) >= min(
                calibration_end, main_end
            )
