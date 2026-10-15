from pathlib import Path

import numpy as np
import pytest

from tools.preflight_iotj_canonical_v1 import (
    assert_finite_arrays,
    assert_no_target_overlap,
    verify_dataset_hashes,
)


def test_preflight_rejects_target_overlap():
    rows = [
        {"client_id": "3", "physical_identity": "same", "role": "calibration"},
        {"client_id": "3", "physical_identity": "same", "role": "test"},
    ]
    with pytest.raises(RuntimeError, match="calibration/test overlap"):
        assert_no_target_overlap(rows)


def test_preflight_rejects_nonfinite_features(tmp_path: Path):
    path = tmp_path / "test_features.npy"
    np.save(path, np.asarray([[[np.nan]]], dtype=np.float32))
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        assert_finite_arrays([path])


def test_hash_verifier_rejects_tamper(tmp_path: Path):
    (tmp_path / "x.txt").write_text("after", encoding="utf-8")
    manifest = {"files": {"x.txt": "0" * 64}, "aggregate_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="SHA256"):
        verify_dataset_hashes(tmp_path, manifest)
