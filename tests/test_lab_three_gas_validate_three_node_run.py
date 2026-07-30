from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.lab_three_gas_3class.validate_three_node_run import (
    expected_target_scope,
)


def _write_split(
    client_dir: Path,
    split: str,
    exposure_counts: dict[str, int],
) -> None:
    exposure_ids = [
        exposure_id
        for exposure_id, count in exposure_counts.items()
        for _ in range(count)
    ]
    np.save(
        client_dir / f"{split}_classification_labels.npy",
        np.zeros(len(exposure_ids), dtype=np.int64),
    )
    with (client_dir / f"{split}_window_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["exposure_id"])
        writer.writeheader()
        writer.writerows(
            {"exposure_id": exposure_id} for exposure_id in exposure_ids
        )


def test_expected_target_scope_is_derived_from_dataset(tmp_path: Path) -> None:
    _write_split(
        tmp_path,
        "calibration",
        {f"cal_{index}": 3 for index in range(30)},
    )
    _write_split(
        tmp_path,
        "test",
        {f"test_{index}": 14 for index in range(30)},
    )

    assert expected_target_scope(tmp_path) == {
        "calibration": {"n_windows": 90, "n_exposures": 30},
        "test": {"n_windows": 420, "n_exposures": 30},
    }
