from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.audit_iotj_a0t_vs_a4_regression import (
    verify_decision,
    verify_hash_index,
)


def test_hash_index_rejects_prediction_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "prediction.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    index = {"prediction.csv": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    assert verify_hash_index(tmp_path, index) == []
    artifact.write_text("a,b\n1,3\n", encoding="utf-8")
    assert verify_hash_index(tmp_path, index) == ["prediction.csv"]


def test_decision_verification_rejects_changed_gate() -> None:
    assert verify_decision(-1.0, -0.5, "REGRESSION_ADVANTAGE_SUPPORTED")
    with pytest.raises(RuntimeError, match="dual-gate"):
        verify_decision(-1.0, 0.5, "REGRESSION_ADVANTAGE_SUPPORTED")
