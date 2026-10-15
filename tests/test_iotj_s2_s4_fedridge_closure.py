from __future__ import annotations

from scripts.run_iotj_s2_s4_fedridge_closure import (
    SOURCE_ALPHA_GRID,
    audit_frozen_h1_preprocessing,
    audit_s4_source_protocol,
    source_clients_for_pool,
)
from scripts.finalize_iotj_regression_qc_closure import run as run_final_closure
import pytest


def test_s4_fedridge_source_pool_excludes_c5():
    assert source_clients_for_pool("S4") == ("C1", "C2", "C3", "C4")
    assert "C5" not in source_clients_for_pool("S4")


def test_s2_and_s4_use_same_registered_source_only_alpha_grid():
    assert source_clients_for_pool("S2") == ("C1", "C2")
    assert SOURCE_ALPHA_GRID == (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def test_s4_role_view_is_frozen_and_c5_rng_is_never_accessed():
    audit = audit_s4_source_protocol()
    assert audit["status"] == "PASS"
    assert audit["source_clients"] == ["C1", "C2", "C3", "C4"]
    assert audit["target_clients"] == ["C5"]
    assert audit["c5_rng_access"] is False
    assert audit["source_test_used_for_fit_or_selection"] is False
    assert audit["c5_used_for_source_fit_or_selection"] is False


def test_legacy_h1_100x8_blocks_canonical_s2_s4_execution():
    audit = audit_frozen_h1_preprocessing()
    assert audit["status"] == "HARD_FAIL_LEGACY_CANONICAL_MIX"
    assert audit["frozen_h1_source_shape"] == [100, 8]
    assert audit["canonical_shape"] == [50, 8]
    assert audit["phase4_execution_authorized"] is False


def test_final_closure_fails_before_writing_when_h1_is_legacy(tmp_path):
    with pytest.raises(RuntimeError, match="HARD_FAIL_LEGACY_CANONICAL_MIX"):
        run_final_closure(tmp_path / "must_not_exist", repeats=1)
    assert not (tmp_path / "must_not_exist").exists()
