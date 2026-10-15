from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_iotj_b5_regression_multiseed import (
    PRIOR_KEYS,
    SEEDS,
    VARIANTS,
    add_variant_features,
    paired_rg1_rg2,
    require_empty_output,
)


def test_variant_contract_is_exact() -> None:
    assert tuple(VARIANTS) == (
        "RG0_RICH_ONLY",
        "RG1_FEDERATED_H1",
        "RG2_ALL_PRIOR",
    )
    assert VARIANTS["RG0_RICH_ONLY"] == ()
    assert VARIANTS["RG1_FEDERATED_H1"] == (PRIOR_KEYS[0],)
    assert VARIANTS["RG2_ALL_PRIOR"] == PRIOR_KEYS


def test_target_dimensions_are_104_105_107() -> None:
    base = {f"f{i}": float(i) for i in range(104)}
    row = {"feature_dict": base, **dict.fromkeys(PRIOR_KEYS, 1.0)}
    assert [
        len(add_variant_features([row], variant)[0]["feature_dict"])
        for variant in VARIANTS
    ] == [104, 105, 107]


def test_refuses_nonempty_output(tmp_path: Path) -> None:
    path = tmp_path / "out"
    path.mkdir()
    (path / "evidence.json").write_text("{}")
    with pytest.raises(FileExistsError):
        require_empty_output(path)


def test_paired_summary_uses_all_five_seeds() -> None:
    rows = []
    for seed, delta in zip(SEEDS, (-1.0, -0.5, 0.0, 0.5, 1.0)):
        rows.extend(
            [
                {
                    "seed": seed,
                    "variant": "RG1_FEDERATED_H1",
                    "S_CC_RMSE": 10.0 + delta,
                },
                {
                    "seed": seed,
                    "variant": "RG2_ALL_PRIOR",
                    "S_CC_RMSE": 10.0,
                },
            ]
        )
    paired, summary = paired_rg1_rg2(rows)
    assert len(paired) == 5
    assert summary["N"] == 5
    assert np.isclose(summary["mean_delta_RG1_minus_RG2"], 0.0)
    assert summary["RG1_wins"] == 2
    assert summary["RG2_wins"] == 2


def test_lock_boolean_round_trip_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "calibration_selection_lock.json"
    payload = {
        "seed_set": list(SEEDS),
        "test_opened_after_lock": False,
        "test_used_for_fit_select_or_refit": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["test_opened_after_lock"] is False
