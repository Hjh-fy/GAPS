from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _checkpoint(path: Path, *, offset: float = 0.0) -> tuple[Path, OrderedDict[str, torch.Tensor]]:
    state = OrderedDict(
        (
            ("linear.weight", torch.tensor([[1.0 + offset, 2.0]], dtype=torch.float32)),
            ("linear.bias", torch.tensor([0.25], dtype=torch.float32)),
        )
    )
    torch.save(
        {
            "round": 25,
            "run_name": "CAN-V1-CRRQ-C0-SOURCE",
            "parameter_keys": list(state),
            "model_state": state,
        },
        path,
    )
    return path, state


def _complete_context_inputs(tmp_path: Path) -> dict:
    checkpoint, state = _checkpoint(tmp_path / "source_round25.pth")
    return {
        "round_id": 25,
        "run_name": "CAN-V1-CRRQ-C0-SOURCE",
        "checkpoint_path": checkpoint,
        "checkpoint_state": state,
        "semantic_protos": {
            "0,1": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "1,2": torch.tensor([3.0, 4.0], dtype=torch.float32),
        },
        "semantic_proto_vars": {
            "0,1": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "1,2": torch.tensor([0.3, 0.4], dtype=torch.float32),
        },
        "client_mus": [
            {(0, 1): torch.tensor([1.1, 2.1])},
            {(0, 1): torch.tensor([0.9, 1.9])},
        ],
        "client_counts": [{(0, 1): 7}, {(0, 1): 5}],
        "client_ids": [2, 1],
        "client_residuals": [torch.tensor([0.2, -0.1]), torch.tensor([0.1, -0.2])],
        "client_weights": torch.tensor([0.6, 0.4]),
    }


def test_round25_context_roundtrips_complete_a4_inputs_in_client_order(tmp_path: Path) -> None:
    """Catches dropping client statistics or retaining Flower arrival order."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        load_final_adaptation_context,
        write_final_adaptation_context,
    )

    inputs = _complete_context_inputs(tmp_path)
    payload = build_final_adaptation_context(**inputs)
    path = write_final_adaptation_context(tmp_path / "context.json", payload)
    restored = load_final_adaptation_context(path, inputs["checkpoint_path"])

    assert restored["round_id"] == 25
    assert restored["client_ids"] == [1, 2]
    assert torch.equal(restored["client_weights"], torch.tensor([0.4, 0.6]))
    assert torch.equal(restored["client_mus"][0][(0, 1)], torch.tensor([0.9, 1.9]))
    assert restored["client_counts"] == [{(0, 1): 5}, {(0, 1): 7}]
    assert torch.equal(restored["client_residuals"][1], torch.tensor([0.2, -0.1]))
    assert torch.equal(restored["semantic_protos"]["0,1"], torch.tensor([1.0, 2.0]))
    assert torch.equal(restored["semantic_proto_vars"]["1,2"], torch.tensor([0.3, 0.4]))
    assert restored["loss_input_availability"] == {
        "client_prototypes": True,
        "client_residuals": True,
        "semantic_prototypes": True,
        "two_client_prototypes": True,
    }


def test_context_json_is_deterministic_and_contains_no_target_data(tmp_path: Path) -> None:
    """Catches nondeterministic evidence or accidental target arrays/labels in source context."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        write_final_adaptation_context,
    )

    payload = build_final_adaptation_context(**_complete_context_inputs(tmp_path))
    first = write_final_adaptation_context(tmp_path / "first.json", payload)
    second = write_final_adaptation_context(tmp_path / "second.json", payload)

    assert first.read_bytes() == second.read_bytes()
    serialized = json.loads(first.read_text(encoding="utf-8"))
    assert serialized["target_access"] == {
        "target_x": False,
        "target_class": False,
        "target_phase": False,
        "target_concentration": False,
        "target_test": False,
    }
    assert "target_arrays" not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_round", "round25"),
        ("empty_clients", "client payload"),
        ("inconsistent_lengths", "client payload lengths"),
        ("nonfinite", "finite"),
    ],
)
def test_context_rejects_invalid_source_state(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Catches malformed final-only state before any target calibration is opened."""
    from gaps_flower.final_adaptation_context import build_final_adaptation_context

    inputs = _complete_context_inputs(tmp_path)
    if mutation == "wrong_round":
        inputs["round_id"] = 24
    elif mutation == "empty_clients":
        inputs.update(client_mus=[], client_counts=[], client_ids=[], client_residuals=[], client_weights=torch.tensor([]))
    elif mutation == "inconsistent_lengths":
        inputs["client_counts"] = inputs["client_counts"][:1]
    elif mutation == "nonfinite":
        inputs["semantic_protos"]["0,1"] = torch.tensor([float("nan"), 2.0])

    with pytest.raises((ValueError, RuntimeError), match=message):
        build_final_adaptation_context(**inputs)


def test_context_rejects_ordered_checkpoint_fingerprint_mismatch(tmp_path: Path) -> None:
    """Catches adapting a checkpoint other than the one bound to the context."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        load_final_adaptation_context,
        write_final_adaptation_context,
    )

    inputs = _complete_context_inputs(tmp_path)
    context = write_final_adaptation_context(
        tmp_path / "context.json", build_final_adaptation_context(**inputs)
    )
    wrong_checkpoint, _state = _checkpoint(tmp_path / "wrong.pth", offset=10.0)

    with pytest.raises(RuntimeError, match="ordered state fingerprint"):
        load_final_adaptation_context(context, wrong_checkpoint)


def test_a4_context_gate_rejects_missing_nonzero_loss_input(tmp_path: Path) -> None:
    """Catches silently disabling prototype/residual A4 losses in final-only adaptation."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        validate_a4_context_loss_inputs,
    )

    inputs = _complete_context_inputs(tmp_path)
    inputs["client_residuals"] = [None, None]
    payload = build_final_adaptation_context(**inputs)
    with pytest.raises(RuntimeError, match="device_residual"):
        validate_a4_context_loss_inputs(
            payload,
            configured_weights={
                "proto_anchor": 0.3,
                "proto_loss": 0.05,
                "consistency": 2.0,
                "device_residual": 0.1,
                "proto_mmd": 0.2,
            },
            expected_context_availability={
                "client_prototypes": True,
                "client_residuals": True,
                "semantic_prototypes": True,
                "two_client_prototypes": True,
            },
        )


def test_a4_context_gate_reports_all_context_conditioned_terms_available(tmp_path: Path) -> None:
    """Catches an incomplete preflight report despite a complete source context."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        validate_a4_context_loss_inputs,
    )

    report = validate_a4_context_loss_inputs(
        build_final_adaptation_context(**_complete_context_inputs(tmp_path)),
        configured_weights={
            "proto_anchor": 0.3,
            "proto_loss": 0.05,
            "consistency": 2.0,
            "device_residual": 0.1,
            "proto_mmd": 0.2,
        },
        expected_context_availability={
            "client_prototypes": True,
            "client_residuals": True,
            "semantic_prototypes": True,
            "two_client_prototypes": True,
        },
    )
    assert report["status"] == "PASS"
    assert report["missing_nonzero_inputs"] == []


def test_a4_context_gate_preserves_baseline_unavailable_residual(tmp_path: Path) -> None:
    """Catches treating a configured-but-baseline-inactive loss as a new requirement."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        validate_a4_context_loss_inputs,
    )

    inputs = _complete_context_inputs(tmp_path)
    inputs["client_residuals"] = [None, None]
    report = validate_a4_context_loss_inputs(
        build_final_adaptation_context(**inputs),
        configured_weights={"device_residual": 0.1},
        expected_context_availability={
            "client_prototypes": True,
            "client_residuals": False,
            "semantic_prototypes": True,
            "two_client_prototypes": True,
        },
    )
    assert report["status"] == "PASS"
    residual = next(row for row in report["terms"] if row["loss_name"] == "device_residual")
    assert residual["input_available"] is False
    assert residual["baseline_input_available"] is False


def test_gaps_strategy_captures_exact_round25_client_payload(tmp_path: Path) -> None:
    """Catches a source run that saves weights but drops the final DA payload."""
    from gaps_flower.final_adaptation_context import load_final_adaptation_context
    from gaps_flower.strategy import GapsStrategy

    checkpoint, state = _checkpoint(tmp_path / "server_round_025.pth")
    strategy = object.__new__(GapsStrategy)
    strategy.output_dir = tmp_path
    strategy.run_name = "CAN-V1-CRRQ-C0-SOURCE"
    strategy.final_adaptation_context_round = 25
    strategy.semantic_protos = {"0,1": torch.tensor([1.0, 2.0])}
    strategy.semantic_proto_vars = {"0,1": torch.tensor([0.1, 0.2])}
    results = []
    for client_id, prototype, count in ((2, [1.2, 2.2], 7), (1, [0.8, 1.8], 5)):
        metrics = {
            "client_id": client_id,
            "prototype_json": json.dumps({"0,1": prototype}),
            "class_phase_counts_json": json.dumps({"0,1": count}),
            "device_residual_json": json.dumps([]),
        }
        results.append((SimpleNamespace(cid=str(client_id)), SimpleNamespace(metrics=metrics)))

    observed = strategy._capture_final_adaptation_context(
        server_round=25,
        aggregated_state=state,
        checkpoint_path=checkpoint,
        results=results,
        weights=torch.tensor([0.6, 0.4]),
    )
    restored = load_final_adaptation_context(observed, checkpoint)

    assert observed == tmp_path / "final_adaptation_context_round_025.json"
    assert restored["client_ids"] == [1, 2]
    assert torch.equal(restored["client_weights"], torch.tensor([0.4, 0.6]))
    assert restored["loss_input_availability"]["client_residuals"] is False


def test_gaps_strategy_does_not_capture_before_registered_round(tmp_path: Path) -> None:
    """Catches an early or checkpoint-selected adaptation context."""
    from gaps_flower.strategy import GapsStrategy

    strategy = object.__new__(GapsStrategy)
    strategy.output_dir = tmp_path
    strategy.final_adaptation_context_round = 25
    observed = strategy._capture_final_adaptation_context(
        server_round=24,
        aggregated_state=OrderedDict(),
        checkpoint_path=tmp_path / "unused.pth",
        results=[],
        weights=torch.tensor([]),
    )
    assert observed is None
    assert list(tmp_path.glob("final_adaptation_context*.json")) == []


def test_a4_final_invocation_uses_complete_context_and_frozen_protocol(tmp_path: Path) -> None:
    """Catches legacy post-hoc empty statistics or a changed A4 optimizer budget."""
    from gaps_flower.final_adaptation_context import (
        build_final_adaptation_context,
        load_final_adaptation_context,
        write_final_adaptation_context,
    )
    from gaps_flower.posthoc_commissioning import (
        A4_C0_A_CONTEXT_AVAILABILITY,
        build_a4_final_invocation,
    )

    inputs = _complete_context_inputs(tmp_path)
    inputs["client_residuals"] = [None, None]
    context_path = write_final_adaptation_context(
        tmp_path / "context.json", build_final_adaptation_context(**inputs)
    )
    context = load_final_adaptation_context(context_path, inputs["checkpoint_path"])
    invocation = build_a4_final_invocation(context)

    assert invocation["num_steps"] == 100
    assert invocation["optimizer"] == "Adam"
    assert invocation["lr"] == 5e-4
    assert invocation["seed"] == 42
    assert invocation["batch_size"] == 32
    assert invocation["client_ids"] == [1, 2]
    assert invocation["client_counts"] == [{(0, 1): 5}, {(0, 1): 7}]
    assert invocation["client_residuals"] == [None, None]
    assert len(invocation["semantic_protos"]) == 2
    assert invocation["context_availability"] == A4_C0_A_CONTEXT_AVAILABILITY
    assert invocation["hyperparams"]["SERVER_OPT_LR"] == 5e-4
    assert invocation["hyperparams"]["ABLATION_VARIANT"] == "A4"


def test_a4_final_invocation_rejects_availability_different_from_c0a(tmp_path: Path) -> None:
    """Catches a final schedule that silently activates or disables a C0-A input."""
    from gaps_flower.final_adaptation_context import build_final_adaptation_context
    from gaps_flower.posthoc_commissioning import build_a4_final_invocation

    payload = build_final_adaptation_context(**_complete_context_inputs(tmp_path))
    with pytest.raises(RuntimeError, match="device_residual"):
        build_a4_final_invocation(payload)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_c0_interleaved_metrics_uses_formal_scope_column() -> None:
    """Catches ignoring the real canonical classification CSV schema."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        target_from_metric_row,
    )

    assert target_from_metric_row({"scope": "C3", "router_target": "C3"}) == "C3"
    assert target_from_metric_row({"scope": "ALL", "router_target": "target_specific_A4"}) == ""


def test_c0_remote_hash_gate_rejects_any_source_manifest_mismatch() -> None:
    """Catches a canonical path whose actual data bytes differ on one machine."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        validate_remote_source_hashes,
    )

    expected = {"client_1/train_experiment_info.json": "a" * 64}
    assert validate_remote_source_hashes(expected, dict(expected), host="pi")["status"] == "PASS"
    with pytest.raises(RuntimeError, match="pi.*train_experiment_info"):
        validate_remote_source_hashes(
            expected,
            {"client_1/train_experiment_info.json": "b" * 64},
            host="pi",
        )


def test_c0_source_commands_are_target_free_a4_source_protocol() -> None:
    """Catches target access or a changed source training budget in rounds 1-25."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        build_c0_source_commands,
    )

    commands = build_c0_source_commands()
    server = commands["server"]
    assert _option(server, "--rounds") == "25"
    assert _option(server, "--seed") == "42"
    assert _option(server, "--strategy") == "gaps"
    assert _option(server, "--profile") == "ce_stats"
    assert _option(server, "--ablation-variant") == "A4"
    assert _option(server, "--use-selective-agg") == "false"
    assert _option(server, "--use-proto-mmd") == "true"
    assert _option(server, "--use-domain-adapt") == "false"
    assert _option(server, "--final-adaptation-context-round") == "25"
    assert "--server-calib-data" not in server
    assert "--server-val-data" not in server
    for role in ("client_c1", "client_c2"):
        command = commands[role]
        assert _option(command, "--profile") == "ce_stats"
        assert _option(command, "--local-epochs") == "1"
        assert _option(command, "--batch-size") == "32"
        assert _option(command, "--seed") == "42"
    assert _option(commands["client_c1"], "--data-root") == (
        "/home/gaps/GAPS/flower_runtime/dataset/iotj_canonical_v1"
    )
    assert _option(commands["client_c2"], "--data-root") == (
        "/root/GAPS/confirmation_c2_data/iotj_canonical_v1"
    )
    serialized = json.dumps(commands)
    assert "client_3" not in serialized
    assert "client_4" not in serialized
    assert "client_5" not in serialized
    assert commands["protocol"]["target_information_in_source_api"] is False
    assert commands["protocol"]["optimizer"] == "Adam"
    assert commands["protocol"]["optimizer_lr"] == 5e-4


def test_c0_decision_requires_each_target_within_margin() -> None:
    """Catches pooled averaging that hides a failed target."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import decide_c0

    supported = decide_c0(
        {"C3": 0.9980, "C4": 0.9970, "C5": 0.9900},
        {"C3": 0.9985, "C4": 0.9977, "C5": 0.9941},
    )
    assert supported["decision"] == "V1_FINAL_ADAPT_SUPPORTED"
    assert supported["all_targets_pass"] is True

    retained = decide_c0(
        {"C3": 0.9980, "C4": 0.9910, "C5": 0.9940},
        {"C3": 0.9985, "C4": 0.9977, "C5": 0.9941},
    )
    assert retained["decision"] == "V1_INTERLEAVED_RETAINED"
    assert retained["targets"]["C4"]["pass"] is False


def test_c0_test_gate_requires_all_three_fixed_step100_markers(tmp_path: Path) -> None:
    """Catches opening any target test before every adaptation endpoint is locked."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        verify_final_adaptation_endpoints,
    )

    for target in ("C3", "C4"):
        directory = tmp_path / f"final_adapt_{target}"
        directory.mkdir()
        (directory / "fixed_endpoint_complete.json").write_text(
            json.dumps({"target": target, "step": 100, "target_test_opened": False}),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="C5"):
        verify_final_adaptation_endpoints(tmp_path)


def test_c0_test_gate_rejects_nonfixed_or_preopened_endpoint(tmp_path: Path) -> None:
    """Catches early stopping or target-test access before the common endpoint."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        verify_final_adaptation_endpoints,
    )

    for target in ("C3", "C4", "C5"):
        directory = tmp_path / f"final_adapt_{target}"
        directory.mkdir()
        (directory / "fixed_endpoint_complete.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "step": 99 if target == "C4" else 100,
                    "target_test_opened": target == "C5",
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="fixed step100|test opened"):
        verify_final_adaptation_endpoints(tmp_path)
