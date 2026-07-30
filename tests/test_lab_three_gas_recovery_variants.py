from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.lab_three_gas_3class.build_allconcentration_timepurged_dataset import (
    resolve_main_indices,
)
from scripts.lab_three_gas_3class.build_fivefold_dataset import (
    BuildConfig,
    read_raw_csv,
    signal_domain,
)
from scripts.lab_three_gas_3class.evaluate_source_target_run import (
    select_source_row,
)


def test_relative_conductance_uses_reciprocal_raw_resistance() -> None:
    resistance = np.asarray([[10.0], [5.0]], dtype=np.float64)
    conductance = signal_domain(resistance, "relative_conductance")
    baseline = conductance[0]
    relative = (conductance[1] - baseline) / baseline

    assert np.allclose(conductance[:, 0], [0.1, 0.2])
    assert np.allclose(relative, [1.0])


def test_raw_reader_honors_selected_channel_contract(tmp_path: Path) -> None:
    data = np.arange(3 * 19, dtype=np.float64).reshape(3, 19) + 1.0
    data[:, 0] = np.arange(3, dtype=np.float64)
    path = tmp_path / "1.csv"
    np.savetxt(path, data, delimiter=",")

    time, signals = read_raw_csv(path, selected_channels=(1, 4, 9))

    assert np.array_equal(time, data[:, 0])
    assert np.array_equal(signals, data[:, [1, 4, 9]])


def test_stable_main_scope_starts_at_150_seconds() -> None:
    config = BuildConfig(raw_root="unused", output_root="unused")
    indices = resolve_main_indices(config, 150.0)

    assert indices == (5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 21, 22)
    assert len(indices) == 12


def test_last_round_selection_ignores_saturated_source_ties() -> None:
    rows = [
        {
            "round": 2,
            "source_validation_exposure_macro_f1": 1.0,
            "source_validation_window_macro_f1": 1.0,
        },
        {
            "round": 25,
            "source_validation_exposure_macro_f1": 1.0,
            "source_validation_window_macro_f1": 1.0,
        },
    ]

    selected, rule = select_source_row(rows, "last_round")

    assert selected["round"] == 25
    assert "source calibration is monitoring only" in rule
