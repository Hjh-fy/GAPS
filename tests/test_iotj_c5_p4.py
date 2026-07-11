import subprocess
import sys
from pathlib import Path

import pytest

from scripts.select_iotj_c5_p4 import merge_streams, select_threshold


REPO_ROOT = Path(__file__).resolve().parents[1]


def row(index: int, split: str, risk: float, h23: float, h8: float, true_ppm: float) -> tuple[dict, dict]:
    base = {
        "client": "C5",
        "split": split,
        "sample_index": index,
        "true_class": 1,
        "pred_class": 1,
        "route_class": 1,
        "true_ppm": true_ppm,
        "risk_score": risk,
        "h23_plus_ppm": h23,
    }
    specialist = {
        "client": "C5",
        "split": split,
        "sample_index": index,
        "true_class": 1,
        "target_ridge_plus_source_preds_ppm": h8,
    }
    return base, specialist


def test_p4_selection_uses_calibration_rows_and_prefers_high_risk_h8() -> None:
    pairs = [
        row(0, "calibration", 0.1, 100.0, 70.0, 100.0),
        row(1, "calibration", 0.9, 50.0, 100.0, 100.0),
    ]
    merged = merge_streams(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs], expected_split="calibration"
    )

    threshold, audit = select_threshold(merged)

    assert 0.1 < threshold < 0.9
    assert any(candidate["feasible"] for candidate in audit)
    assert all(candidate["nonCO_delta_vs_H23"] is None for candidate in audit)


def test_p4_selection_rejects_test_rows() -> None:
    base, specialist = row(0, "test", 0.5, 90.0, 100.0, 100.0)
    merged = merge_streams([base], [specialist], expected_split="test")

    with pytest.raises(ValueError, match="calibration-validation"):
        select_threshold(merged)


def test_p4_merge_rejects_non_c5_rows() -> None:
    base, specialist = row(0, "calibration", 0.5, 90.0, 100.0, 100.0)
    base["client"] = "C4"

    with pytest.raises(ValueError, match="unexpected H2.3 row role"):
        merge_streams([base], [specialist], expected_split="calibration")


def test_p4_script_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/select_iotj_c5_p4.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "C5-only P4 risk gate" in result.stdout
