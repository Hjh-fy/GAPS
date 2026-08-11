from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _write_endpoint(root: Path, method: str, source_fingerprint: str) -> None:
    directory = root / method
    directory.mkdir(parents=True)
    checkpoint = directory / "endpoint.pth"
    checkpoint.write_bytes(method.encode("utf-8"))
    digest = hashlib.sha256(method.encode("utf-8")).hexdigest()
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "source_state_fingerprint": source_fingerprint,
        "calibration_manifest_sha256": "calibration",
        "calibration_count": 320,
        "steps": 100,
        "lr": 5e-4,
        "seed": 42,
        "target_test_opened": False,
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "fixed_endpoint_complete.json").write_text(
        json.dumps(
            {
                "step": 100,
                "checkpoint_sha256": digest,
                "source_state_fingerprint": source_fingerprint,
                "target_test_opened": False,
            }
        ),
        encoding="utf-8",
    )


def test_gate_b_protocol_is_fixed_and_contains_no_search() -> None:
    from scripts.run_iotj_method_breakthrough_gate_b import gate_b_protocol

    protocol = gate_b_protocol()
    assert protocol["source_set"] == "S2"
    assert protocol["target"] == "C5"
    assert protocol["calibration_count"] == 320
    assert protocol["steps"] == 100
    assert protocol["optimizer"] == "Adam"
    assert protocol["lr"] == 5e-4
    assert protocol["seed"] == 42
    assert protocol["adapter_rank"] == 4
    assert protocol["hyperparameter_search"] is False
    assert protocol["checkpoint_selection"] == "fixed_step_100"


def test_gate_b_requires_both_new_locked_endpoints_before_test_open(tmp_path: Path) -> None:
    from scripts.run_iotj_method_breakthrough_gate_b import verify_new_endpoint_locks

    _write_endpoint(tmp_path, "classifier_only", "same-source")
    with pytest.raises(RuntimeError, match="missing endpoint"):
        verify_new_endpoint_locks(tmp_path)
    _write_endpoint(tmp_path, "rank4_adapter", "same-source")
    locked = verify_new_endpoint_locks(tmp_path)
    assert set(locked) == {"classifier_only", "rank4_adapter"}

    marker = tmp_path / "rank4_adapter/fixed_endpoint_complete.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["target_test_opened"] = True
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="test opened"):
        verify_new_endpoint_locks(tmp_path)


def test_gate_b_decision_selects_simplest_sufficient_path() -> None:
    from scripts.run_iotj_method_breakthrough_gate_b import decide_gate_b

    result = decide_gate_b(
        full_f1=0.980,
        candidates={
            "classifier_only": {"macro_f1": 0.976, "trainable_parameters": 260},
            "projection_head": {"macro_f1": 0.979, "trainable_parameters": 3396},
            "rank4_adapter": {"macro_f1": 0.981, "trainable_parameters": 772},
        },
        full_trainable_parameters=22765,
    )
    assert result["decision"] == "LIGHTWEIGHT_PERSONALIZATION_SUPPORTED"
    assert result["selected_method"] == "classifier_only"


def test_gate_b_decision_requires_full_when_no_localized_path_is_close() -> None:
    from scripts.run_iotj_method_breakthrough_gate_b import decide_gate_b

    result = decide_gate_b(
        full_f1=0.980,
        candidates={
            "classifier_only": {"macro_f1": 0.90, "trainable_parameters": 260},
            "projection_head": {"macro_f1": 0.91, "trainable_parameters": 3396},
            "rank4_adapter": {"macro_f1": 0.92, "trainable_parameters": 772},
        },
        full_trainable_parameters=22765,
    )
    assert result["decision"] == "FULL_ADAPTATION_REQUIRED"
    assert result["selected_method"] == "a0t_full"


def test_gate_b_runner_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_iotj_method_breakthrough_gate_b.py", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Gate B" in result.stdout
