from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gaps_flower.task import create_model, make_config


def _manifest(path: Path, *, role: str = "calibration", count: int = 2) -> Path:
    rows = [
        {
            "physical_identity": f"C5|fixture|{index}",
            "client_id": 5,
            "role": role,
            "classification_label": index % 4,
            "phase_label": index % 3,
        }
        for index in range(count)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_calibration_manifest_rejects_test_role(tmp_path: Path) -> None:
    from gaps_flower.posthoc_commissioning import load_calibration_identity_manifest

    path = _manifest(tmp_path / "manifest.json", role="test")
    with pytest.raises(ValueError, match="calibration"):
        load_calibration_identity_manifest(path, expected_client=5, expected_count=2)


def test_calibration_manifest_has_unique_calibration_identities(tmp_path: Path) -> None:
    from gaps_flower.posthoc_commissioning import load_calibration_identity_manifest

    path = _manifest(tmp_path / "manifest.json")
    result = load_calibration_identity_manifest(path, expected_client=5, expected_count=2)
    assert result == ("C5|fixture|0", "C5|fixture|1")


def test_target_head_updates_only_projection_and_classifier() -> None:
    from gaps_flower.posthoc_commissioning import configure_trainable_parameters

    model = create_model(make_config(device="cpu", local_epochs=1, batch_size=32, seed=42))
    names = configure_trainable_parameters(model, "target_head")
    assert names
    assert all(name.startswith(("feat_proj.", "classifier.")) for name in names)
    assert any(name.startswith("feat_proj.") for name in names)
    assert any(name.startswith("classifier.") for name in names)
    assert not any(parameter.requires_grad for parameter in model.tcn.parameters())
    assert not any(parameter.requires_grad for parameter in model.self_attn.parameters())
    assert not any(parameter.requires_grad for parameter in model.attn_linear.parameters())


def test_full_methods_update_all_parameters() -> None:
    from gaps_flower.posthoc_commissioning import configure_trainable_parameters

    for method in ("a0t_full", "a4"):
        model = create_model(make_config(device="cpu", local_epochs=1, batch_size=32, seed=42))
        names = configure_trainable_parameters(model, method)
        assert names == [name for name, _ in model.named_parameters()]
        assert all(parameter.requires_grad for parameter in model.parameters())


def test_ordered_state_fingerprint_is_container_serialization_invariant(tmp_path: Path) -> None:
    from gaps_flower.posthoc_commissioning import ordered_state_fingerprint

    state = {"a": torch.arange(3), "b": torch.tensor([[1.0, 2.0]])}
    first = {"round": 25, "model_state": state, "run_name": "one"}
    second = {"run_name": "two", "model_state": state, "round": 25}
    torch.save(first, tmp_path / "first.pth")
    torch.save(second, tmp_path / "second.pth")
    assert ordered_state_fingerprint(first["model_state"]) == ordered_state_fingerprint(second["model_state"])


def test_posthoc_request_rejects_any_test_manifest(tmp_path: Path) -> None:
    from gaps_flower.posthoc_commissioning import PosthocRequest

    calibration = _manifest(tmp_path / "calibration.json")
    test = _manifest(tmp_path / "test.json", role="test")
    with pytest.raises(ValueError, match="test manifest"):
        PosthocRequest(
            source_checkpoint=tmp_path / "source.pth",
            calibration_manifest=calibration,
            target_test_manifest=test,
            method="a0t_full",
        ).validate_static_boundary()


def test_posthoc_request_contains_no_flower_round_configuration(tmp_path: Path) -> None:
    from gaps_flower.posthoc_commissioning import PosthocRequest

    request = PosthocRequest(
        source_checkpoint=tmp_path / "source.pth",
        calibration_manifest=_manifest(tmp_path / "calibration.json"),
        method="a0t_full",
    )
    fields = set(request.__dataclass_fields__)
    assert "rounds" not in fields
    assert "server_address" not in fields
    assert "client_commands" not in fields


def test_gate1_requires_all_three_locked_endpoints(tmp_path: Path) -> None:
    from scripts.run_iotj_posthoc_commissioning_g1 import verify_adaptation_gate

    with pytest.raises(RuntimeError, match="missing endpoint"):
        verify_adaptation_gate(tmp_path)


def test_gate1_decision_thresholds_are_pre_registered() -> None:
    from scripts.run_iotj_posthoc_commissioning_g1 import decide_gate1

    decision = decide_gate1(
        source_f1=0.50,
        a0t_f1=0.98,
        a4_f1=0.99,
        head_f1=0.977,
        a0t_retention=-0.10,
        a4_retention=-0.08,
        head_retention=-0.02,
        a0t_seconds=10.0,
        a4_seconds=20.0,
        head_seconds=5.0,
        a0t_trainable=100,
        head_trainable=20,
        historical_a0t=0.994,
        historical_a4=0.994,
    )
    assert decision["lifecycle"] == "POSTHOC_LIFECYCLE_SUPPORTED"
    assert decision["a4"] == "KEEP"
    assert decision["target_head"] == "PROMISING"
    assert decision["interleaved_dependency_risk"] is False


def test_gate1_flags_interleaved_dependency_without_budget_search() -> None:
    from scripts.run_iotj_posthoc_commissioning_g1 import decide_gate1

    decision = decide_gate1(
        source_f1=0.40,
        a0t_f1=0.80,
        a4_f1=0.81,
        head_f1=0.79,
        a0t_retention=-0.20,
        a4_retention=-0.20,
        head_retention=-0.10,
        a0t_seconds=10.0,
        a4_seconds=20.0,
        head_seconds=5.0,
        a0t_trainable=100,
        head_trainable=20,
        historical_a0t=0.99,
        historical_a4=0.99,
    )
    assert decision["interleaved_dependency_risk"] is True
    assert "2500" not in json.dumps(decision)
