"""Protocol tests for Phase-2 DG-to-commissioning bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_phase2_matrix_has_six_endpoints_and_only_i0_b20_reuse() -> None:
    from scripts.run_iotj_dg_commissioning_bridge import phase2_specs

    specs = phase2_specs()
    assert len(specs) == 6
    assert {(row["identity"], row["budget"]) for row in specs} == {
        ("I0", 20), ("I0", 5), ("I1", 20), ("I1", 5), ("I2", 20), ("I2", 5)
    }
    reused = [row for row in specs if row["execution"] == "reuse"]
    assert [(row["identity"], row["budget"]) for row in reused] == [("I0", 20)]


def test_phase2_budget_inputs_are_nested_calibration_only() -> None:
    from scripts.run_iotj_dg_commissioning_bridge import audit_budget_inputs

    audit = audit_budget_inputs()
    assert audit["status"] == "PASS"
    assert audit["counts"] == {"20": 320, "5": 80}
    assert audit["per_stratum"] == {"20": 8, "5": 2}
    assert audit["nested"] is True
    assert audit["calibration_test_identity_overlap"] == 0
    assert audit["test_arrays_available_to_adaptation"] is False


def test_phase2_source_identities_are_exact_frozen_round25_states() -> None:
    from scripts.run_iotj_dg_commissioning_bridge import audit_source_identities

    identities = audit_source_identities()
    assert set(identities) == {"I0", "I1", "I2"}
    assert all(row["round"] == 25 for row in identities.values())
    assert all(row["seed"] == 42 for row in identities.values())
    assert all(row["target_access"] == "NONE" for row in identities.values())
    assert len({row["checkpoint_state_fingerprint"] for row in identities.values()}) == 3


def test_phase2_i0_b20_reuse_requires_exact_full_a0t_protocol() -> None:
    from scripts.run_iotj_dg_commissioning_bridge import audit_i0_b20_reuse

    reuse = audit_i0_b20_reuse()
    assert reuse["status"] == "PASS"
    assert reuse["identity"] == "I0"
    assert reuse["budget"] == 20
    assert reuse["steps"] == 100
    assert reuse["lr"] == 5e-4
    assert reuse["seed"] == 42
    assert reuse["method"] == "a0t_full"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({("I0", 20): .95, ("I0", 5): .90, ("I1", 20): .96, ("I1", 5): .91, ("I2", 20): .98, ("I2", 5): .93}, "DG_TO_COMMISSIONING_SUPPORTED"),
        ({("I0", 20): .95, ("I0", 5): .90, ("I1", 20): .96, ("I1", 5): .91, ("I2", 20): .965, ("I2", 5): .925}, "DG_LOW_BUDGET_VALUE_SUPPORTED"),
        ({("I0", 20): .94, ("I0", 5): .89, ("I1", 20): .95, ("I1", 5): .90, ("I2", 20): .955, ("I2", 5): .905}, "DG_ZERO_SHOT_ONLY"),
        ({("I0", 20): .90, ("I0", 5): .85, ("I1", 20): .93, ("I1", 5): .86, ("I2", 20): .92, ("I2", 5): .85}, "SOURCE_DIVERSITY_ONLY"),
    ],
)
def test_phase2_registered_decision_rules(values: dict[tuple[str, int], float], expected: str) -> None:
    from scripts.run_iotj_dg_commissioning_bridge import decide_dg_commissioning

    result = decide_dg_commissioning(values, seed42_zero_shot_dg_gain=0.0749899433041799)
    assert result["decision"] == expected
    assert result["next_action"] == "ENTER_PHASE3_REGISTERED_SELECTION"


def test_phase2_lock_gate_rejects_test_open_or_wrong_source(tmp_path: Path) -> None:
    from scripts.run_iotj_dg_commissioning_bridge import verify_new_endpoint_locks

    expected = {}
    for identity, budget in (("I0", 5), ("I1", 20), ("I1", 5), ("I2", 20), ("I2", 5)):
        directory = tmp_path / identity / f"B{budget:02d}"
        directory.mkdir(parents=True)
        checkpoint = directory / "adapted.pth"
        checkpoint.write_bytes(f"{identity}-{budget}".encode())
        source_fingerprint = f"source-{identity}"
        expected[identity] = source_fingerprint
        import hashlib
        manifest = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_state_fingerprint": source_fingerprint,
            "target_test_opened": False,
            "steps": 100,
            "lr": 5e-4,
            "seed": 42,
            "budget": budget,
        }
        (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "fixed_endpoint_complete.json").write_text(
            json.dumps({"step": 100, "target_test_opened": False}), encoding="utf-8"
        )
    assert len(verify_new_endpoint_locks(tmp_path, expected)) == 5
    bad = tmp_path / "I2/B05/run_manifest.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["target_test_opened"] = True
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="target test"):
        verify_new_endpoint_locks(tmp_path, expected)
