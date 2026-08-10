from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_iotj_a0t_vs_a4_regression import (
    EXPECTED_DATASET_SHA256,
    EndpointSpec,
    audit_checkpoint,
    audit_endpoint_pair,
    audit_inputs,
    endpoint_specs,
    frozen_alphas,
)


def test_exact_six_endpoints_and_checkpoint_is_only_method_factor() -> None:
    specs = endpoint_specs()
    assert [(spec.method, spec.target) for spec in specs] == [
        ("A0T", "C3"),
        ("A0T", "C4"),
        ("A0T", "C5"),
        ("A4", "C3"),
        ("A4", "C4"),
        ("A4", "C5"),
    ]
    for target in ("C3", "C4", "C5"):
        a0t = next(spec for spec in specs if spec.method == "A0T" and spec.target == target)
        a4 = next(spec for spec in specs if spec.method == "A4" and spec.target == target)
        assert audit_endpoint_pair(a0t, a4) == {
            "status": "PASS",
            "target": target,
            "varying_fields": ["checkpoint", "checkpoint_sha256", "classification_manifest", "completion_marker", "experiment_id", "method"],
        }


def test_pair_audit_rejects_non_checkpoint_protocol_drift() -> None:
    a0t, a4 = [spec for spec in endpoint_specs() if spec.target == "C5"]
    drifted = replace(a4, split_protocol="changed")
    with pytest.raises(RuntimeError, match="held-constant drift"):
        audit_endpoint_pair(a0t, drifted)


def test_frozen_alpha_table_is_exact_and_method_independent() -> None:
    assert frozen_alphas() == {
        "C3": {0: 100.0, 1: 0.0, 2: 0.1, 3: 0.1},
        "C4": {0: 1.0, 1: 10.0, 2: 0.1, 3: 10.0},
        "C5": {0: 1.0, 1: 0.01, 2: 10.0, 3: 0.1},
    }
    assert EXPECTED_DATASET_SHA256 == "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"


def test_endpoint_paths_resolve_to_existing_frozen_assets() -> None:
    for spec in endpoint_specs():
        assert isinstance(spec, EndpointSpec)
        assert Path(spec.checkpoint).is_file()
        assert Path(spec.classification_manifest).is_file()
        assert Path(spec.completion_marker).is_file()


def test_audit_inputs_writes_sealed_freeze_and_registry(tmp_path: Path) -> None:
    result = audit_inputs(tmp_path / "study")
    assert result["status"] == "PASS"
    assert result["endpoint_count"] == 6
    assert result["target_test_state"] == "SEALED"
    assert result["alpha_selection_performed"] is False
    assert (tmp_path / "study" / "PRE_RUN_FREEZE.json").is_file()
    assert (tmp_path / "study" / "experiment_registry.csv").is_file()


def test_checkpoint_audit_rejects_wrong_registered_hash() -> None:
    spec = endpoint_specs()[0]
    tampered = replace(spec, checkpoint_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="checkpoint SHA256 differs"):
        audit_checkpoint(tampered)
