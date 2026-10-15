from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_iotj_a0t_vs_a4_qc import (
    H23_EXPECTED_SHA256,
    frozen_qc_asset_paths,
    load_frozen_thresholds,
    qc_summary_rows,
)


def test_frozen_qc_assets_exist_and_h23_hash_is_locked() -> None:
    assets = frozen_qc_asset_paths("C5")
    assert all(path.is_file() for path in assets.values())
    assert H23_EXPECTED_SHA256 == "18b6c14373018474807eec2bd19a0b508b75adfbf994b0821a786a11def9c263"


def test_thresholds_are_loaded_from_lock_without_refit() -> None:
    rows = load_frozen_thresholds("C5")
    assert len(rows) == 13
    assert next(row for row in rows if float(row["target_coverage"]) == 0.9)["threshold"] == "0.6378019435579839"
    assert all(row["target_test_used_for_selection"] == "False" for row in rows)


def test_qc_summary_reports_requested_populations() -> None:
    records = [
        {"true_ppm": 10.0, "pred_84d_h1_ppm": 11.0, "true_class": 0,
         "HC90_decision": "accepted", "HC95_decision": "accepted"},
        {"true_ppm": 20.0, "pred_84d_h1_ppm": 24.0, "true_class": 0,
         "HC90_decision": "review", "HC95_decision": "accepted"},
        {"true_ppm": 30.0, "pred_84d_h1_ppm": 39.0, "true_class": 0,
         "HC90_decision": "reject", "HC95_decision": "reject"},
    ]
    rows = qc_summary_rows("A0T", "C5", records)
    hc90 = {row["population"]: row for row in rows if row["workpoint"] == "HC90"}
    assert hc90["accepted"]["coverage"] == pytest.approx(1 / 3)
    assert hc90["accepted+review"]["RMSE"] == pytest.approx((17 / 2) ** 0.5)
    assert hc90["reject"]["reject_rate"] == pytest.approx(1 / 3)


def test_qc_script_does_not_contain_threshold_refit() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "evaluate_iotj_a0t_vs_a4_qc.py").read_text(encoding="utf-8")
    assert "np.quantile" not in source
    assert "fit_target_qc_thresholds" not in source
