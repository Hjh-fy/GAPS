from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_c0_lock_only_preexecution_failure_is_preserved_for_retry(tmp_path: Path) -> None:
    """Catches deleting the sole evidence when no process/round ever started."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        prepare_lock_only_source_retry,
    )

    run = tmp_path / "source_fl"
    run.mkdir()
    expected = {"server": ["python", "server"], "protocol": {"rounds": 25}}
    (run / "locked_run_spec.json").write_text(json.dumps(expected), encoding="utf-8")
    archived = prepare_lock_only_source_retry(run, expected)
    assert archived == tmp_path / "source_fl_preexecution_failure_001"
    assert not run.exists()
    assert json.loads((archived / "locked_run_spec.json").read_text(encoding="utf-8")) == expected


def test_c0_lock_only_retry_rejects_any_execution_artifact(tmp_path: Path) -> None:
    """Catches retrying a partial or completed scientific endpoint."""
    from scripts.run_iotj_canonical_regression_reconstruction_c0 import (
        prepare_lock_only_source_retry,
    )

    run = tmp_path / "source_fl"
    run.mkdir()
    expected = {"server": ["python", "server"]}
    (run / "locked_run_spec.json").write_text(json.dumps(expected), encoding="utf-8")
    (run / "server.stderr.log").write_text("round 1", encoding="utf-8")
    with pytest.raises(FileExistsError, match="execution artifacts"):
        prepare_lock_only_source_retry(run, expected)


def test_canonical_quantitative_feature_operator_is_50x8_and_83_plus_21() -> None:
    """Catches resizing, a changed H1 schema, or loss of row identity binding."""
    from gaps_flower.canonical_quantitative_features import extract_canonical_features

    window = np.arange(50 * 8, dtype=np.float32).reshape(50, 8) / 100.0
    record = extract_canonical_features(
        window,
        phase=1,
        metadata={
            "physical_identity": "C3|methane|225|repeat1|60.0|70.0",
            "filename": "methane_225_repeat1.csv",
            "window_start_s": 60.0,
            "window_end_s": 70.0,
            "response_phase": "main_response",
            "phase_label": "middle",
        },
        client="C3",
        split="calibration",
        sample_index=7,
    )

    assert record.sensor83.shape == (83,)
    assert record.h1.shape == (104,)
    assert record.h1_feature_names[:83] != record.sensor_feature_names  # H1 is globally ordered.
    assert set(record.sensor_feature_names).issubset(record.h1_feature_names)
    assert record.identity["physical_identity"].startswith("C3|")
    assert record.identity["window_start_s"] == 60.0
    assert record.provenance["window_shape"] == [50, 8]
    assert record.provenance["sampling_rate_hz"] == 5
    assert record.provenance["dynamic_descriptor_interpretation"] == (
        "fixed-5-Hz discrete per-sample descriptors"
    )
    assert record.provenance["sampling_rate_invariant_claim"] is False


def test_canonical_quantitative_feature_operator_rejects_legacy_window() -> None:
    """Catches a legacy 10-Hz/100x8 array entering canonical R0."""
    from gaps_flower.canonical_quantitative_features import extract_canonical_features

    with pytest.raises(ValueError, match="50x8"):
        extract_canonical_features(
            np.zeros((100, 8), dtype=np.float32),
            phase=0,
            metadata={},
            client="C1",
            split="train",
            sample_index=0,
        )


def test_canonical_feature_cache_manifest_rejects_legacy_or_mixed_provenance() -> None:
    """Catches accepting a cache without the complete canonical content binding."""
    from gaps_flower.canonical_quantitative_features import validate_cache_manifest

    valid = {
        "study_id": "CAN-V1-CRRQ-20260811",
        "sampling_rate_hz": 5,
        "window_shape": [50, 8],
        "dataset_aggregate_sha256": "a" * 64,
        "source_array_sha256": "b" * 64,
        "metadata_sha256": "c" * 64,
        "extractor_file_sha256": "d" * 64,
        "ordered_h1_feature_names_sha256": "e" * 64,
        "ordered_sensor_feature_names_sha256": "f" * 64,
        "h1_dimensions": 104,
        "sensor_dimensions": 83,
        "created_from_canonical_arrays": True,
        "legacy_cache_reused": False,
    }
    validate_cache_manifest(valid, expected_dataset_sha256="a" * 64)
    for key, value in (
        ("window_shape", [100, 8]),
        ("sampling_rate_hz", 10),
        ("legacy_cache_reused", True),
        ("extractor_file_sha256", ""),
    ):
        broken = dict(valid)
        broken[key] = value
        with pytest.raises(RuntimeError, match="canonical cache provenance"):
            validate_cache_manifest(broken, expected_dataset_sha256="a" * 64)


def test_canonical_fedridge_uses_population_scaler_and_unregularized_intercept() -> None:
    """Catches ddof drift, scale-floor drift, or regularizing the intercept."""
    from gaps_flower.canonical_fedridge import (
        client_feature_moments,
        client_normal_equations,
        server_aggregate_scaler,
        server_reconstruct_ridge,
    )

    x1 = np.asarray([[1.0, 5.0], [3.0, 5.0]])
    x2 = np.asarray([[5.0, 5.0], [7.0, 5.0]])
    y1 = np.asarray([2.0, 4.0])
    y2 = np.asarray([6.0, 8.0])
    moments = [
        client_feature_moments("C1", 0, "train", x1),
        client_feature_moments("C2", 0, "train", x2),
    ]
    scaler = server_aggregate_scaler(moments)
    assert np.allclose(scaler.mean, [4.0, 5.0])
    assert np.allclose(scaler.scale, [np.sqrt(5.0), 1.0])
    equations = [
        client_normal_equations("C1", 0, "train", x1, y1, scaler),
        client_normal_equations("C2", 0, "train", x2, y2, scaler),
    ]
    model = server_reconstruct_ridge(equations, scaler, ["x0", "x1"], alpha=1000.0)
    assert model.coef[0] == pytest.approx(5.0, abs=1e-12)
    assert sum(item.y_y for item in equations) == pytest.approx(float(y1 @ y1 + y2 @ y2))


def test_canonical_fedridge_exactly_recovers_pooled_reference() -> None:
    """Catches non-additive statistics or a different pseudoinverse convention."""
    from gaps_flower.canonical_fedridge import federated_fit, pooled_fit

    x1 = np.asarray([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]], dtype=np.float64)
    x2 = np.asarray([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]], dtype=np.float64)
    y1 = np.asarray([1.0, 5.0, 9.0])
    y2 = np.asarray([3.0, 7.0, 11.0])
    federated, _stats = federated_fit(
        {"C1": (x1, y1), "C2": (x2, y2)},
        gas_id=2,
        role="train_plus_calibration_refit",
        feature_names=["x0", "x1"],
        alpha=0.1,
    )
    pooled = pooled_fit(
        np.vstack([x1, x2]),
        np.concatenate([y1, y2]),
        gas_id=2,
        role="train_plus_calibration_refit",
        feature_names=["x0", "x1"],
        alpha=0.1,
    )
    probe = np.asarray([[1.5, 1.0], [4.5, 1.0]])
    assert np.max(np.abs(federated.mean - pooled.mean)) <= 1e-10
    assert np.max(np.abs(federated.coef - pooled.coef)) <= 1e-8
    assert np.max(np.abs(federated.predict_matrix(probe) - pooled.predict_matrix(probe))) <= 1e-6


def test_canonical_source_alpha_selection_is_source_calibration_only_and_tie_stable() -> None:
    """Catches target/test leakage or an unregistered alpha/tie-break rule."""
    from gaps_flower.canonical_fedridge import select_source_alpha

    train = {
        "C1": (np.asarray([[0.0], [1.0]]), np.asarray([0.0, 1.0])),
        "C2": (np.asarray([[2.0], [3.0]]), np.asarray([2.0, 3.0])),
    }
    calibration = {
        "C1": (np.asarray([[0.5]]), np.asarray([0.5])),
        "C2": (np.asarray([[2.5]]), np.asarray([2.5])),
    }
    selected, audit = select_source_alpha(
        train,
        calibration,
        gas_id=0,
        feature_names=["x0"],
        alphas=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        train_role="source_train",
        validation_role="source_calibration",
    )
    assert selected == 0.0
    assert [row["alpha"] for row in audit] == [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    with pytest.raises(RuntimeError, match="source-only"):
        select_source_alpha(
            train,
            calibration,
            gas_id=0,
            feature_names=["x0"],
            alphas=[0.0],
            train_role="source_train",
            validation_role="C5_test",
        )


def test_r0_execution_plan_locks_source_model_before_any_test_label_access() -> None:
    """Catches source/target test labels entering feature or alpha construction."""
    from scripts.run_iotj_canonical_regression_reconstruction_r0 import (
        build_r0_execution_plan,
    )

    plan = build_r0_execution_plan()
    assert plan.index("write_source_alpha_and_model_lock") < plan.index("open_source_test_labels")
    assert plan.index("write_source_alpha_and_model_lock") < plan.index("build_target_x_only_caches")
    assert "open_target_test_labels" not in plan


def test_r0_exact_recovery_gate_has_no_practical_fallback() -> None:
    """Catches accepting a merely close reconstruction outside the frozen tolerances."""
    from scripts.run_iotj_canonical_regression_reconstruction_r0 import (
        decide_exact_recovery,
    )

    passed = decide_exact_recovery(
        scaler_difference=1e-11,
        coefficient_difference=1e-9,
        prediction_difference_ppm=1e-7,
    )
    assert passed["status"] == "PASS"
    failed = decide_exact_recovery(
        scaler_difference=1e-11,
        coefficient_difference=1.1e-8,
        prediction_difference_ppm=1e-7,
    )
    assert failed["status"] == "FAIL_CLOSED"
    assert failed["practical_equivalence_fallback"] is False
