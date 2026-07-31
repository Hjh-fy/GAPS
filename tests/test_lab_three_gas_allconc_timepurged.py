from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

import scripts.lab_three_gas_3class.build_allconcentration_timepurged_dataset as builder
from scripts.lab_three_gas_3class.build_allconcentration_timepurged_dataset import (
    CALIBRATION_INDICES,
    EARLY_INDICES,
    FULL_INDICES,
    MAIN_INDICES,
    PURGED_INDICES,
    STABLE_INDICES,
    assemble_direction_records,
    build_dataset,
    parse_args,
    resolve_direction,
    write_direction_dataset,
)
from scripts.lab_three_gas_3class.build_fivefold_dataset import BuildConfig


def test_timepurged_index_contract() -> None:
    assert CALIBRATION_INDICES == (3, 11, 19)
    assert PURGED_INDICES == (2, 4, 10, 12, 18, 20)
    assert len(MAIN_INDICES) == 14
    assert (
        set(CALIBRATION_INDICES)
        | set(PURGED_INDICES)
        | set(MAIN_INDICES)
    ) == set(range(23))
    assert EARLY_INDICES == (0, 1)
    assert STABLE_INDICES == (
        5,
        6,
        7,
        8,
        9,
        13,
        14,
        15,
        16,
        17,
        21,
        22,
    )
    assert FULL_INDICES == EARLY_INDICES + STABLE_INDICES


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


def test_crossboard_direction_contract() -> None:
    assert resolve_direction("P2_to_P3") == ((2,), 3)
    assert resolve_direction("P2_to_P1") == ((2,), 1)
    assert resolve_direction("P1_to_P3") == ((1,), 3)
    assert resolve_direction("P12_to_P3") == ((1, 2), 3)
    assert resolve_direction("P3_to_P1") == ((3,), 1)


def _record(platform: int) -> dict:
    exposure_id = f"P{platform}_E00"
    return {
        "platform": platform,
        "gas_label": 0,
        "features": np.full((23, 100, 6), platform, dtype=np.float32),
        "window_rows": [
            {
                "exposure_id": exposure_id,
                "platform": platform,
                "base_window_index": index,
            }
            for index in range(23)
        ],
    }


def test_p2_to_p1_record_roles_and_scopes() -> None:
    parts = assemble_direction_records(
        [_record(1), _record(2), _record(3)],
        direction="P2_to_P1",
        primary_indices=STABLE_INDICES,
    )

    assert tuple(parts["source_train"]) == (2,)
    assert len(parts["source_train"][2][0]["features"]) == 12
    assert parts["target_client"] == 1
    assert len(parts["target_calibration"][0]["features"]) == 3
    assert len(parts["target_primary"][0]["features"]) == 12
    assert len(parts["target_stable"][0]["features"]) == 12
    assert len(parts["target_early"][0]["features"]) == len(EARLY_INDICES) == 2
    assert len(parts["target_full"][0]["features"]) == 14


def test_a1_full_primary_has_independent_nonduplicated_diagnostic_scopes() -> None:
    parts = assemble_direction_records(
        [_record(1), _record(2), _record(3)],
        direction="P3_to_P1",
        primary_indices=FULL_INDICES,
    )

    assert tuple(parts["source_train"]) == (3,)
    assert len(parts["source_train"][3][0]["features"]) == 14
    assert parts["target_client"] == 1
    assert len(parts["target_primary"][0]["features"]) == 14
    assert len(parts["target_early"][0]["features"]) == 2
    assert len(parts["target_stable"][0]["features"]) == 12
    assert len(parts["target_full"][0]["features"]) == 14
    assert [
        row["base_window_index"]
        for row in parts["target_full"][0]["window_rows"]
    ] == list(FULL_INDICES)


def test_write_p1_to_p3_dataset_with_named_target_scopes(
    tmp_path: Path,
) -> None:
    records = [
        {
            **_record(platform),
            "gas_label": exposure_index % 3,
            "features": np.full(
                (23, 100, 6),
                platform * 100 + exposure_index,
                dtype=np.float32,
            ),
            "window_rows": [
                {
                    "exposure_id": f"P{platform}_E{exposure_index:02d}",
                    "platform": platform,
                    "gas_label": exposure_index % 3,
                    "base_window_index": index,
                }
                for index in range(23)
            ],
        }
        for platform in (1, 2, 3)
        for exposure_index in range(30)
    ]
    config = BuildConfig(
        raw_root="unused",
        output_root=str(tmp_path),
        normalization_clients=(1,),
        selected_channels=(1, 2, 4, 6, 8, 9),
    )

    summary = write_direction_dataset(
        records,
        config=config,
        output_root=tmp_path,
        direction="P1_to_P3",
        primary_indices=STABLE_INDICES,
    )

    target = tmp_path / "fold_1" / "client_3"
    assert summary["source_clients"] == [1]
    assert summary["target_client"] == 3
    assert len(np.load(target / "test_features.npy")) == 360
    assert len(np.load(target / "early_features.npy")) == 60
    assert len(np.load(target / "stable_features.npy")) == 360
    assert len(np.load(target / "full_features.npy")) == 420
    stats = np.load(tmp_path / "fold_1" / "norm_stats.npz")
    assert stats["normalization_clients"].tolist() == [1]


def test_cli_accepts_crossboard_direction(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["builder", "--direction", "P2_to_P1"],
    )

    assert parse_args().direction == "P2_to_P1"


def test_build_dataset_forwards_direction_to_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = [
        {
            **_record(platform),
            "gas_label": exposure_index % 3,
            "features": np.full(
                (23, 100, 6),
                platform * 100 + exposure_index,
                dtype=np.float32,
            ),
            "window_rows": [
                {
                    "exposure_id": f"P{platform}_E{exposure_index:02d}",
                    "platform": platform,
                    "gas_label": exposure_index % 3,
                    "base_window_index": index,
                }
                for index in range(23)
            ],
        }
        for platform in (1, 2, 3)
        for exposure_index in range(30)
    ]
    monkeypatch.setattr(builder, "discover_sessions", lambda *_: [object()])
    monkeypatch.setattr(builder, "load_boundaries", lambda *_: {})
    monkeypatch.setattr(
        builder,
        "build_exposure_records",
        lambda *_: (
            records,
            [{"exposure_id": "synthetic"}],
            [{"exposure_id": "synthetic"}],
        ),
    )
    config = BuildConfig(
        raw_root=str(tmp_path / "raw"),
        output_root=str(tmp_path / "out"),
        normalization_clients=(1,),
        selected_channels=(1, 2, 4, 6, 8, 9),
    )

    summary = build_dataset(config, direction="P1_to_P3")

    assert summary["direction"] == "P1_to_P3"
    assert summary["source_clients"] == [1]
    assert summary["target_client"] == 3
