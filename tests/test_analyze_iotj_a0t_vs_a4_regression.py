from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.analyze_iotj_a0t_vs_a4_regression import (
    DEFAULT_OUTPUT,
    _high_concentration_rows,
    regression_decision,
    routing_row,
)


def test_routing_and_regression_gaps_follow_frozen_formula() -> None:
    row = routing_row(
        "A0T", "C5",
        {"S_ALL": 12.0, "S_CC": 9.0, "Oracle_ALL": 8.0, "Oracle_CC": 8.5},
    )
    assert row["routing_gap"] == pytest.approx(3.0)
    assert row["regression_gap"] == pytest.approx(1.0)
    assert row["paired_regression_gap"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("c5_delta", "pooled_delta", "expected"),
    [
        (-1.0, -0.5, "REGRESSION_ADVANTAGE_SUPPORTED"),
        (0.1, -0.5, "REGRESSION_ADVANTAGE_NOT_SUPPORTED"),
        (-1.0, 0.1, "REGRESSION_ADVANTAGE_NOT_SUPPORTED"),
        (0.0, -0.1, "REGRESSION_ADVANTAGE_NOT_SUPPORTED"),
    ],
)
def test_dual_gate_requires_strict_c5_and_pooled_improvement(
    c5_delta: float, pooled_delta: float, expected: str
) -> None:
    assert regression_decision(c5_delta, pooled_delta) == expected


def test_direct_cli_imports_repo_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "analyze_iotj_a0t_vs_a4_regression.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_c5_high_concentration_rows_use_raw_gas_codes() -> None:
    rows = _high_concentration_rows(DEFAULT_OUTPUT)
    assert {(row["method"], row["gas"]) for row in rows} == {
        ("A0T", "CO"), ("A0T", "Methane"), ("A4", "CO"), ("A4", "Methane")
    }
    assert all(row["N"] > 0 for row in rows)
