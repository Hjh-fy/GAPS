import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_iotj_canonical_v1_strict_nonoverlap import (
    allocate_balanced_quotas,
    assert_strict_nonoverlap,
    choose_calibration_file,
    evenly_spaced_indices,
)


def test_balanced_quotas_preserve_exact_frozen_calibration_budget():
    cells = [(str(cls), str(ppm)) for cls in range(4) for ppm in range(10)]
    quotas = allocate_balanced_quotas(cells, 678)
    assert sum(quotas.values()) == 678
    assert set(quotas.values()) == {16, 17}


def test_evenly_spaced_selection_is_deterministic_and_unique():
    assert evenly_spaced_indices(21, 8) == evenly_spaced_indices(21, 8)
    assert len(set(evenly_spaced_indices(21, 8))) == 8
    assert min(evenly_spaced_indices(21, 8)) >= 0
    assert max(evenly_spaced_indices(21, 8)) < 21


def test_calibration_repeat_cycles_for_phase_coverage_without_test_metrics():
    files = ["R1.txt", "R2.txt", "R3.txt", "R4.txt"]
    assert [choose_calibration_file(3, files, index) for index in range(4)] == files
    assert [choose_calibration_file(4, files[:2], index) for index in range(2)] == files[:2]
    assert choose_calibration_file(5, files[:2], 0) == "R2.txt"


def test_strict_audit_rejects_shared_raw_file():
    rows = [
        {"client_id": 5, "filename": "same.txt", "role": "calibration", "physical_identity": "a", "window_start_s": 60, "window_end_s": 70},
        {"client_id": 5, "filename": "same.txt", "role": "test", "physical_identity": "b", "window_start_s": 90, "window_end_s": 100},
    ]
    with pytest.raises(RuntimeError, match="raw-file overlap"):
        assert_strict_nonoverlap(rows)


def test_strict_audit_rejects_raw_time_overlap_even_across_distinct_identity():
    rows = [
        {"client_id": 3, "filename": "f.txt", "role": "calibration", "physical_identity": "a", "window_start_s": 60, "window_end_s": 70},
        {"client_id": 3, "filename": "f.txt", "role": "test", "physical_identity": "b", "window_start_s": 65, "window_end_s": 75},
    ]
    with pytest.raises(RuntimeError, match="raw-file overlap"):
        assert_strict_nonoverlap(rows)


def test_strict_audit_accepts_disjoint_files_and_time():
    rows = [
        {"client_id": 4, "filename": "r1.txt", "role": "calibration", "physical_identity": "a", "window_start_s": 60, "window_end_s": 70},
        {"client_id": 4, "filename": "r2.txt", "role": "test", "physical_identity": "b", "window_start_s": 60, "window_end_s": 70},
    ]
    summary = assert_strict_nonoverlap(rows)
    assert summary["exact_window_overlap_count"] == 0
    assert summary["raw_file_overlap_count"] == 0
    assert summary["raw_time_overlap_seconds"] == 0.0


def test_strict_preflight_cli_imports_from_direct_script_execution():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools/preflight_iotj_canonical_v1_strict_nonoverlap.py"), "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
