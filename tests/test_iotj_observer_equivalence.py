from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import numpy as np
import torch
from flwr.common import Code, FitIns, FitRes, Status, ndarrays_to_parameters
from gaps_flower.observability import JsonlObserver, ObserverIdentity

import scripts.run_iotj_observer_equivalence_gate as gate_module

from scripts.run_iotj_observer_equivalence_gate import (
    VOLATILE_JSON_PATHS,
    _create_frozen_initial_checkpoint,
    _flower_trace_fingerprint,
    _fixed_adapted_logits,
    _normalized_timing_scalar_mapping,
    _run_config_argument_types,
    _validate_observer_sidecars,
    compare_fingerprints,
    json_fingerprint,
    run_formal_topology_gate,
    run_local_gate,
    tensor_fingerprint,
)


def test_equivalence_gate_module_contract_is_importable() -> None:
    assert callable(tensor_fingerprint)
    assert callable(json_fingerprint)
    assert callable(compare_fingerprints)
    assert callable(run_local_gate)


def _save_checkpoint(path: Path, state: OrderedDict[str, torch.Tensor]) -> Path:
    torch.save(
        {
            "round": 2,
            "model_state": state,
            "semantic_protos": OrderedDict(
                [("0,0", torch.tensor([1.0, 2.0], dtype=torch.float32))]
            ),
        },
        path,
    )
    return path


def _base_json() -> dict[str, object]:
    return {
        "run_config": {
            "args": {
                "observer_context": None,
                "observer_events": None,
                "stable": 42,
            }
        },
        "metrics": {
            "fit_seconds": 1.25,
            "evaluate_seconds": 2.5,
            "prototype": [1.0, 2.0],
            "prototype_count": 1,
            "global_stat": 0.25,
        },
        "provenance": {
            "wall_time_utc": "2026-07-16T00:00:00Z",
            "pid": 123,
            "path": "off-a/output",
        },
        "flower": {"config": {"server_round": 1}, "metrics": {"loss": 0.5}},
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_volatile_allowlist_is_exact_and_frozen() -> None:
    assert VOLATILE_JSON_PATHS == {
        ("run_config", "args", "observer_context"),
        ("run_config", "args", "observer_events"),
        ("metrics", "fit_seconds"),
        ("metrics", "evaluate_seconds"),
        ("provenance", "wall_time_utc"),
        ("provenance", "pid"),
        ("provenance", "path"),
    }


def test_tensor_fingerprint_detects_one_changed_tensor_byte(tmp_path: Path) -> None:
    left = _save_checkpoint(
        tmp_path / "left.pth",
        OrderedDict(
            [
                ("weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
                ("bias", torch.tensor([3.0], dtype=torch.float32)),
            ]
        ),
    )
    changed = _save_checkpoint(
        tmp_path / "changed.pth",
        OrderedDict(
            [
                ("weight", torch.tensor([[1.0, 2.0000002]], dtype=torch.float32)),
                ("bias", torch.tensor([3.0], dtype=torch.float32)),
            ]
        ),
    )

    left_fp = tensor_fingerprint(left)
    changed_fp = tensor_fingerprint(changed)

    assert left_fp["key_order"] == [
        "model_state.weight",
        "model_state.bias",
        "semantic_protos.0,0",
    ]
    assert left_fp["tensors"]["model_state.weight"]["dtype"] == "torch.float32"
    assert left_fp["tensors"]["model_state.weight"]["shape"] == [1, 2]
    assert left_fp["content_sha256"] != changed_fp["content_sha256"]
    result = compare_fingerprints(
        {"checkpoint": left_fp},
        {"checkpoint": changed_fp},
        {"checkpoint": left_fp},
    )
    assert result["status"] == "observer_path_mutation"
    assert result["max_abs_delta"] > 0.0


@pytest.mark.parametrize(
    "right_state",
    [
        OrderedDict(
            [
                ("bias", torch.tensor([3.0], dtype=torch.float32)),
                ("weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ]
        ),
        OrderedDict(
            [
                ("weight", torch.tensor([[1.0, 2.0]], dtype=torch.float64)),
                ("bias", torch.tensor([3.0], dtype=torch.float32)),
            ]
        ),
        OrderedDict(
            [
                ("weight", torch.tensor([1.0, 2.0], dtype=torch.float32)),
                ("bias", torch.tensor([3.0], dtype=torch.float32)),
            ]
        ),
    ],
)
def test_tensor_fingerprint_preserves_key_order_dtype_and_shape(
    tmp_path: Path, right_state: OrderedDict[str, torch.Tensor]
) -> None:
    base_state = OrderedDict(
        [
            ("weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("bias", torch.tensor([3.0], dtype=torch.float32)),
        ]
    )
    left = tensor_fingerprint(_save_checkpoint(tmp_path / "left.pth", base_state))
    right = tensor_fingerprint(_save_checkpoint(tmp_path / "right.pth", right_state))
    assert left["content_sha256"] != right["content_sha256"]


def test_json_fingerprint_ignores_only_exact_volatile_leaf_values(
    tmp_path: Path,
) -> None:
    off = _base_json()
    on = _base_json()
    on["run_config"]["args"]["observer_context"] = "context.json"
    on["run_config"]["args"]["observer_events"] = "events.jsonl"
    on["metrics"]["fit_seconds"] = 99.0
    on["metrics"]["evaluate_seconds"] = 88.0
    on["provenance"] = {
        "wall_time_utc": "later",
        "pid": 999,
        "path": "on/output",
    }
    off_fp = json_fingerprint(
        _write_json(tmp_path / "off.json", off), VOLATILE_JSON_PATHS
    )
    on_fp = json_fingerprint(
        _write_json(tmp_path / "on.json", on), VOLATILE_JSON_PATHS
    )
    assert off_fp["artifact_sha256"] != on_fp["artifact_sha256"]
    assert off_fp["content_sha256"] == on_fp["content_sha256"]
    assert off_fp["comparison"] == on_fp["comparison"]

    nested = _base_json()
    nested["wrapper"] = {"metrics": {"fit_seconds": 9.0}}
    nested_changed = _base_json()
    nested_changed["wrapper"] = {"metrics": {"fit_seconds": 10.0}}
    assert json_fingerprint(
        _write_json(tmp_path / "nested.json", nested), VOLATILE_JSON_PATHS
    )["content_sha256"] != json_fingerprint(
        _write_json(tmp_path / "nested-changed.json", nested_changed),
        VOLATILE_JSON_PATHS,
    )["content_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["metrics"]["prototype"].__setitem__(0, 1.25),
        lambda value: value["metrics"].__setitem__("prototype_count", 2),
        lambda value: value["metrics"].__setitem__("global_stat", 0.5),
        lambda value: value["flower"]["config"].__setitem__("observer", True),
        lambda value: value["flower"]["metrics"].__setitem__("observer_ns", 1),
    ],
)
def test_json_fingerprint_rejects_stat_change_or_new_flower_key(
    tmp_path: Path, mutation
) -> None:
    left = _base_json()
    right = _base_json()
    mutation(right)
    left_fp = json_fingerprint(
        _write_json(tmp_path / "left.json", left), VOLATILE_JSON_PATHS
    )
    right_fp = json_fingerprint(
        _write_json(tmp_path / "right.json", right), VOLATILE_JSON_PATHS
    )
    assert left_fp["content_sha256"] != right_fp["content_sha256"]


def test_timing_normalization_preserves_keys_and_scalar_types(tmp_path: Path) -> None:
    payload = _base_json()
    fingerprint = json_fingerprint(
        _write_json(tmp_path / "payload.json", payload), VOLATILE_JSON_PATHS
    )
    normalized = fingerprint["comparison"]
    assert set(normalized["metrics"]) == set(payload["metrics"])
    assert type(normalized["metrics"]["fit_seconds"]) is float
    assert type(normalized["metrics"]["evaluate_seconds"]) is float
    assert normalized["metrics"]["fit_seconds"] == 0.0
    assert normalized["metrics"]["evaluate_seconds"] == 0.0


def test_compare_reports_environment_nondeterminism_before_observer_mutation() -> None:
    result = compare_fingerprints(
        {"artifact": {"comparison": {"value": 1}}},
        {"artifact": {"comparison": {"value": 9}}},
        {"artifact": {"comparison": {"value": 2}}},
    )
    assert result["status"] == "environment_nondeterminism"
    assert result["equivalent"] is False
    assert result["off_pair_equal"] is False


def test_compare_reports_observer_path_mutation_when_off_pair_is_equal() -> None:
    result = compare_fingerprints(
        {"artifact": {"comparison": {"value": 1}}},
        {"artifact": {"comparison": {"value": 2}}},
        {"artifact": {"comparison": {"value": 1}}},
    )
    assert result["status"] == "observer_path_mutation"
    assert result["equivalent"] is False
    assert result["off_pair_equal"] is True


def test_compare_equivalent_result_is_deterministic_and_carries_hashes() -> None:
    triplet = {
        "artifact": {
            "artifact_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "comparison": {"value": 1},
        }
    }
    first = compare_fingerprints(triplet, triplet, triplet)
    second = compare_fingerprints(triplet, triplet, triplet)
    assert first == second
    assert first["status"] == "equivalent"
    assert first["equivalent"] is True
    assert first["max_abs_delta"] == 0.0
    assert first["artifact_hashes"]["off_a"]["artifact"] == "a" * 64


def test_compare_requires_raw_checkpoint_sha_equality() -> None:
    off = {
        "final_checkpoint_raw": {
            "artifact_sha256": "a" * 64,
            "content_sha256": "a" * 64,
            "comparison": {"raw_file_sha256": "a" * 64},
        }
    }
    on = {
        "final_checkpoint_raw": {
            "artifact_sha256": "b" * 64,
            "content_sha256": "b" * 64,
            "comparison": {"raw_file_sha256": "b" * 64},
        }
    }
    result = compare_fingerprints(off, on, off)
    assert result["status"] == "observer_path_mutation"


def test_json_object_insertion_order_is_not_a_value_difference() -> None:
    left = {"artifact": {"comparison": {"a": 1, "b": 2}}}
    right = {"artifact": {"comparison": {"b": 2, "a": 1}}}
    assert compare_fingerprints(left, right, left)["status"] == "equivalent"


def test_cli_help_and_invalid_group_contract() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "scripts.run_iotj_observer_equivalence_gate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--group" in help_result.stdout
    assert "--output-root" in help_result.stdout

    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_iotj_observer_equivalence_gate",
            "--group",
            "B1",
            "--output-root",
            "unused",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode != 0


def test_run_local_gate_refuses_existing_output_without_deleting_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "belongs-to-user.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must not already exist"):
        run_local_gate(output, "B2")

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_fingerprints_fail_closed_on_symlink_input(tmp_path: Path) -> None:
    target = _write_json(tmp_path / "target.json", _base_json())
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink/reparse"):
        json_fingerprint(link, VOLATILE_JSON_PATHS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_tensor_fingerprint_rejects_nonfinite_tensor(
    tmp_path: Path, bad: float
) -> None:
    checkpoint = _save_checkpoint(
        tmp_path / "bad.pth",
        OrderedDict([("weight", torch.tensor([bad], dtype=torch.float32))]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        tensor_fingerprint(checkpoint)


def _fit_messages(extra_config: bool = False, extra_metrics: bool = False):
    parameters = ndarrays_to_parameters(
        [
            torch.tensor([[1.0, 2.0]], dtype=torch.float32).numpy(),
            torch.tensor([3.0], dtype=torch.float32).numpy(),
        ]
    )
    config = {"server_round": 1, "stable": "yes"}
    metrics = {
        "client_id": 1,
        "fit_seconds": 1.25,
        "prototype_json": '{"0,0":[1.0,2.0]}',
    }
    if extra_config:
        config["observer_new_key"] = True
    if extra_metrics:
        metrics["observer_new_key"] = 1
    return (
        FitIns(parameters=parameters, config=config),
        FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=parameters,
            num_examples=8,
            metrics=metrics,
        ),
    )


def test_common_trace_normalizes_only_existing_timing_value_and_preserves_keys_types() -> None:
    fit_ins, fit_res = _fit_messages()
    first = _flower_trace_fingerprint(
        fit_ins=fit_ins,
        fit_res=fit_res,
        parameter_keys=["weight", "bias"],
    )
    fit_res.metrics["fit_seconds"] = 999.0
    second = _flower_trace_fingerprint(
        fit_ins=fit_ins,
        fit_res=fit_res,
        parameter_keys=["weight", "bias"],
    )

    assert first["comparison"] == second["comparison"]
    assert first["fit_res"]["metrics_keys"] == list(fit_res.metrics)
    assert first["fit_res"]["metrics_types"]["fit_seconds"] == "float"
    assert first["fit_res"]["normalized_application_message_bytes"] > 0
    assert len(first["fit_res"]["normalized_application_message_sha256"]) == 64


def test_aggregate_trace_normalizes_existing_timings_without_dropping_keys_or_types() -> None:
    metrics = OrderedDict(
        [
            ("accuracy", 0.75),
            ("fit_seconds", 9.25),
            ("evaluate_seconds", 3),
            ("stable_label", "kept"),
        ]
    )
    result = _normalized_timing_scalar_mapping(metrics)

    assert result["key_order"] == list(metrics)
    assert result["keys"] == list(metrics)
    assert result["types"] == {
        "accuracy": "float",
        "fit_seconds": "float",
        "evaluate_seconds": "int",
        "stable_label": "str",
    }
    assert result["values"]["fit_seconds"] == {"type": "float", "value": 0.0}
    assert result["values"]["evaluate_seconds"] == {"type": "int", "value": 0}
    assert result["values"]["accuracy"]["value"] == 0.75
    assert metrics["fit_seconds"] == 9.25


def test_run_config_type_mirror_ignores_only_exact_observer_leaf_types() -> None:
    off = {
        "observer_context": None,
        "observer_events": None,
        "stable": 42,
        "run_name": "same",
    }
    on = {
        "observer_context": "context.json",
        "observer_events": "events.jsonl",
        "stable": 42,
        "run_name": "same",
    }
    changed_nonallowlisted = {**on, "stable": 42.0}

    assert _run_config_argument_types(off) == _run_config_argument_types(on)
    assert _run_config_argument_types(off) != _run_config_argument_types(
        changed_nonallowlisted
    )


@pytest.mark.parametrize("where", ["config", "metrics"])
def test_common_trace_real_message_new_key_is_not_normalized_away(where: str) -> None:
    base_ins, base_res = _fit_messages()
    changed_ins, changed_res = _fit_messages(
        extra_config=where == "config", extra_metrics=where == "metrics"
    )
    base = _flower_trace_fingerprint(
        fit_ins=base_ins,
        fit_res=base_res,
        parameter_keys=["weight", "bias"],
    )
    changed = _flower_trace_fingerprint(
        fit_ins=changed_ins,
        fit_res=changed_res,
        parameter_keys=["weight", "bias"],
    )
    result = compare_fingerprints(
        {"common_trace": base},
        {"common_trace": changed},
        {"common_trace": base},
    )
    assert result["status"] == "observer_path_mutation"


def test_per_client_round_trace_prevents_aggregate_cancellation() -> None:
    off = {
        "common_trace": {
            "comparison": {
                "rounds": {
                    "1": {
                        "fit_res": {"C1": "a", "C2": "b"},
                        "aggregate": "same",
                        "returned": "same",
                        "adapted_logits": "same",
                    }
                }
            }
        }
    }
    on = {
        "common_trace": {
            "comparison": {
                "rounds": {
                    "1": {
                        "fit_res": {"C1": "b", "C2": "a"},
                        "aggregate": "same",
                        "returned": "same",
                        "adapted_logits": "same",
                    }
                }
            }
        }
    }
    assert compare_fingerprints(off, on, off)["status"] == "observer_path_mutation"


def test_one_frozen_initial_checkpoint_is_reused_and_fingerprinted(tmp_path: Path) -> None:
    result = _create_frozen_initial_checkpoint(tmp_path / "initial.pth", "B2")
    assert result["path"] == str(tmp_path / "initial.pth")
    assert len(result["raw_sha256"]) == 64
    assert len(result["tensor_content_sha256"]) == 64
    assert result["training_seed"] == 42
    assert result["loaded_by_modes"] == ["off_a", "on", "off_b"]


def test_fixed_logits_observation_preserves_all_process_rng_states(
    tmp_path: Path,
) -> None:
    checkpoint = _create_frozen_initial_checkpoint(tmp_path / "initial.pth", "B2")
    random.seed(8123)
    np.random.seed(8123)
    torch.manual_seed(8123)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    result = _fixed_adapted_logits(Path(checkpoint["path"]), "B2")

    assert len(result["content_sha256"]) == 64
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_sidecar_validator_derives_zero_for_off_and_rejects_missing_on(
    tmp_path: Path,
) -> None:
    off = tmp_path / "off"
    off.mkdir()
    assert _validate_observer_sidecars(off, enabled=False) == {
        "enabled": False,
        "event_files": 0,
        "close_summaries": 0,
        "producers": {},
    }
    on = tmp_path / "on"
    on.mkdir()
    with pytest.raises(ValueError, match="server.*C1.*C2|missing"):
        _validate_observer_sidecars(on, enabled=True)


def test_formal_topology_gate_uses_strict_frozen_binding_and_injected_runner(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "summary" / "confirmation_protocol_manifest.json"
    protocol.parent.mkdir()
    protocol.write_text("{}", encoding="utf-8")
    calls = []

    def fake_loader(path: Path):
        assert path == protocol
        return {
            "protocol_manifest_sha256": "a" * 64,
            "source_archive_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "regular_members_sha256": "d" * 64,
            "confirmation_commit": "e" * 40,
            "group_id": "B2",
            "seed": 42,
            "command_manifest_sha256": "f" * 64,
            "archive_sha256": "b" * 64,
        }

    def fake_runner(binding, output_root, group_id):
        calls.append((binding, output_root, group_id))
        return {
            "status": "equivalent",
            "equivalent": True,
            "max_abs_delta": 0.0,
            "modes": ["off", "on"],
            "resource_samples_per_client_round": 1,
        }

    output = tmp_path / "formal-output"
    report = run_formal_topology_gate(
        protocol,
        output,
        "B2",
        frozen_loader=fake_loader,
        runner=fake_runner,
    )
    assert report["status"] == "equivalent"
    assert calls == [(fake_loader(protocol), output, "B2")]
    assert report["binding"]["command_manifest_sha256"] == "f" * 64


def test_real_flower_chain_detects_injected_on_fitins_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IOTJ_GATE_TEST_INJECT_ON_KEY", "1")
    monkeypatch.setenv("IOTJ_GATE_INJECT_POST_AUDIT_METRICS_KEY", "1")
    report = run_local_gate(tmp_path / "injected-real-chain", "B2")

    assert report["status"] == "observer_path_mutation"
    assert report["off_pair_equal"] is True
    assert report["on_equal_to_off"] is False
    assert any(
        "observer_injected_regression_key" in mismatch
        for mismatch in report["mismatches"]
    )
    assert any(
        "observer_injected_post_audit_metrics_key" in mismatch
        for mismatch in report["mismatches"]
    )
    assert len(set(report["final_checkpoint_sha256"].values())) == 1


def test_observer_message_audit_must_match_independent_common_trace() -> None:
    validator = getattr(gate_module, "_cross_validate_message_audits", None)
    assert callable(validator), "missing observer/common-trace cross-validator"
    common = {
        "raw_messages": [
            {
                "round": 1,
                "direction": "downlink",
                "client_id": "C1",
                "application_message_bytes": 11,
                "application_message_sha256": "a" * 64,
                "logical": {"logical_downlink_total_bytes": 10},
            }
        ]
    }
    observer = [
        {
            "round": 1,
            "direction": "downlink",
            "client_id": "C1",
            "application_message_bytes": 11,
            "application_message_sha256": "b" * 64,
            "logical": {"logical_downlink_total_bytes": 10},
        }
    ]
    with pytest.raises(ValueError, match="common trace"):
        validator(observer, common)


def _resource_identity(client_id: str) -> ObserverIdentity:
    return ObserverIdentity(
        run_id="c12_to_c5__b2__s42",
        attempt_id="c12_to_c5__b2__s42__a999",
        group_id="B2",
        training_seed=42,
        client_id=client_id,
        host_id="pc" if client_id == "C2" else "pi",
        producer="resource_sampler",
        confirmation_commit="a" * 40,
        source_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        algorithm_config_sha256="d" * 64,
    )


def _training_identity(producer: str, client_id: str | None) -> ObserverIdentity:
    return ObserverIdentity(
        run_id="c12_to_c5__b2__s42",
        attempt_id="c12_to_c5__b2__s42__a999",
        group_id="B2",
        training_seed=42,
        client_id=client_id,
        host_id="server" if producer == "server" else str(client_id).lower(),
        producer=producer,
        confirmation_commit="a" * 40,
        source_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        algorithm_config_sha256="d" * 64,
    )


def _expected_training_binding() -> dict:
    server = _training_identity("server", None)
    return {
        "schema_version": "iotj.confirmation.observability.v1",
        "run_id": server.run_id,
        "attempt_id": server.attempt_id,
        "group_id": server.group_id,
        "training_seed": server.training_seed,
        "confirmation_commit": server.confirmation_commit,
        "source_archive_sha256": server.source_archive_sha256,
        "dataset_manifest_sha256": server.dataset_manifest_sha256,
        "algorithm_config_sha256": server.algorithm_config_sha256,
        "producers": {
            "server": {"host_id": "server", "producer": "server", "client_id": None},
            "C1": {"host_id": "c1", "producer": "client", "client_id": "C1"},
            "C2": {"host_id": "c2", "producer": "client", "client_id": "C2"},
        },
    }


def _expected_resource_binding(client_id: str) -> dict:
    identity = _resource_identity(client_id)
    return {
        "schema_version": "iotj.confirmation.observability.v1",
        "run_id": identity.run_id,
        "attempt_id": identity.attempt_id,
        "group_id": identity.group_id,
        "training_seed": identity.training_seed,
        "client_id": identity.client_id,
        "host_id": identity.host_id,
        "producer": identity.producer,
        "confirmation_commit": identity.confirmation_commit,
        "source_archive_sha256": identity.source_archive_sha256,
        "dataset_manifest_sha256": identity.dataset_manifest_sha256,
        "algorithm_config_sha256": identity.algorithm_config_sha256,
    }


def _rewrite_event_rows(path: Path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    close_path = path.with_suffix(".close.json")
    close = json.loads(close_path.read_text(encoding="utf-8"))
    close["observer_event_bytes_written"] = path.stat().st_size
    close_path.write_text(json.dumps(close), encoding="utf-8")


def _write_training_sidecars(root: Path, *, duplicate_c1_fitres: bool = False) -> None:
    root.mkdir()
    downlink_audit = {
        "logical": {
            "logical_downlink_model_value_bytes": 10,
            "logical_downlink_parameter_blob_bytes": 10,
            "logical_downlink_semantic_proto_utf8_bytes": 0,
            "logical_downlink_other_config_value_bytes": 1,
            "logical_downlink_total_bytes": 11,
        },
        "application_message_bytes": 20,
        "application_message_sha256": "1" * 64,
    }
    uplink_audit = {
        "logical": {
            "logical_uplink_model_value_bytes": 10,
            "logical_uplink_parameter_blob_bytes": 10,
            "logical_uplink_prototype_utf8_bytes": 1,
            "logical_uplink_prototype_var_utf8_bytes": 1,
            "logical_uplink_statistics_utf8_bytes": 1,
            "logical_uplink_diagnostic_value_bytes": 1,
            "logical_uplink_total_bytes": 14,
        },
        "application_message_bytes": 30,
        "application_message_sha256": "2" * 64,
    }
    server = JsonlObserver(
        _training_identity("server", None), root / "server_events.jsonl"
    )
    for round_idx in (1, 2):
        server.emit(
            "fit_round_start", round_idx=round_idx, client_id=None,
            status="started", payload={}
        )
        for proxy_id in ("p1", "p2"):
            server.emit(
                "flower_fitins_prepared", round_idx=round_idx,
                client_id=proxy_id, status="succeeded",
                payload={"proxy_id": proxy_id, "downlink_audit": downlink_audit},
            )
        server.emit(
            "server_aggregate_start", round_idx=round_idx, client_id=None,
            status="started", payload={}
        )
        for proxy_id, client_id in (("p1", "C1"), ("p2", "C2")):
            effective = (
                "C1"
                if duplicate_c1_fitres and round_idx == 2 and client_id == "C2"
                else client_id
            )
            server.emit(
                "flower_fitres_available", round_idx=round_idx,
                client_id=effective, status="succeeded",
                payload={"proxy_id": proxy_id, "uplink_audit": uplink_audit},
            )
        server.emit(
            "server_da_start", round_idx=round_idx, client_id=None,
            status="started", payload={}
        )
        server.emit(
            "server_da_end", round_idx=round_idx, client_id=None,
            status="succeeded", payload={"server_da_total_ns": 1}
        )
        server.emit(
            "server_aggregate_end", round_idx=round_idx, client_id=None,
            status="succeeded",
            payload={
                "server_aggregate_fit_total_ns": 2,
                "server_da_total_ns": 1,
                "server_aggregate_non_da_ns": 1,
                "da_executed": True,
            },
        )
        server.emit(
            "fit_round_end", round_idx=round_idx, client_id=None,
            status="succeeded",
            payload={
                "server_aggregate_fit_total_ns": 2,
                "server_da_total_ns": 1,
                "server_aggregate_non_da_ns": 1,
                "da_executed": True,
                "fit_round_wall_ns": 3,
            },
        )
    server.close()
    for client_id in ("C1", "C2"):
        client = JsonlObserver(
            _training_identity("client", client_id),
            root / f"client_{client_id.lower()}_events.jsonl",
        )
        for round_idx in (1, 2):
            client.emit(
                "client_fit_start", round_idx=round_idx, client_id=client_id,
                status="started", payload={}
            )
            client.emit(
                "client_train_start", round_idx=round_idx, client_id=client_id,
                status="started", payload={}
            )
            client.emit(
                "client_train_end", round_idx=round_idx, client_id=client_id,
                status="succeeded", payload={"client_train_core_ns": 1}
            )
            client.emit(
                "client_fit_end", round_idx=round_idx, client_id=client_id,
                status="succeeded", payload={"client_fit_callback_ns": 2}
            )
        client.close()


def test_training_sidecar_validator_rejects_duplicate_round_client_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    _write_training_sidecars(root, duplicate_c1_fitres=True)
    with pytest.raises(ValueError, match="matrix|C2"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_binds_close_bytes_to_jsonl_size(
    tmp_path: Path,
) -> None:
    root = tmp_path / "close-bytes"
    _write_training_sidecars(root)
    close_path = root / "server_events.close.json"
    close = json.loads(close_path.read_text(encoding="utf-8"))
    close["observer_event_bytes_written"] += 1
    close_path.write_text(json.dumps(close), encoding="utf-8")
    with pytest.raises(ValueError, match="byte"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_invalid_application_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-audit"
    _write_training_sidecars(root)
    event_path = root / "server_events.jsonl"
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    target = next(
        row for row in rows if row["event_type"] == "flower_fitins_prepared"
    )
    target["payload"]["downlink_audit"]["application_message_bytes"] = -1
    event_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    close_path = root / "server_events.close.json"
    close = json.loads(close_path.read_text(encoding="utf-8"))
    close["observer_event_bytes_written"] = event_path.stat().st_size
    close_path.write_text(json.dumps(close), encoding="utf-8")

    with pytest.raises(ValueError, match="application.*audit|message byte"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_string_close_integer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "string-close"
    _write_training_sidecars(root)
    close_path = root / "server_events.close.json"
    close = json.loads(close_path.read_text(encoding="utf-8"))
    close["observer_event_count"] = str(close["observer_event_count"])
    close_path.write_text(json.dumps(close), encoding="utf-8")

    with pytest.raises(ValueError, match="type|integer|close"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_string_overhead_integer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "string-overhead"
    _write_training_sidecars(root)
    event_path = root / "server_events.jsonl"

    def mutate(rows: list[dict]) -> None:
        overhead = next(row for row in rows if row["event_type"] == "observer_overhead")
        value = overhead["payload"]["observer_event_count"]
        overhead["payload"]["observer_event_count"] = str(value)

    _rewrite_event_rows(event_path, mutate)
    with pytest.raises(ValueError, match="type|integer|overhead"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_detected_reparse_event_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reparse-event"
    _write_training_sidecars(root)
    event_path = root / "server_events.jsonl"
    original_detector = gate_module._is_reparse_or_link
    monkeypatch.setattr(
        gate_module,
        "_is_reparse_or_link",
        lambda path: Path(path) == event_path or original_detector(Path(path)),
    )

    with pytest.raises(ValueError, match="symlink|reparse"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_consistent_wrong_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wrong-binding"
    _write_training_sidecars(root)
    for event_path in root.glob("*events.jsonl"):
        _rewrite_event_rows(
            event_path,
            lambda rows: [
                row.__setitem__("confirmation_commit", "f" * 40) for row in rows
            ],
        )

    with pytest.raises(ValueError, match="binding|confirmation_commit"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_float_training_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "float-training-seed"
    _write_training_sidecars(root)
    for event_path in root.glob("*events.jsonl"):
        _rewrite_event_rows(
            event_path,
            lambda rows: [row.__setitem__("training_seed", 42.0) for row in rows],
        )

    with pytest.raises(ValueError, match="type|training_seed|binding"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def test_training_sidecar_validator_rejects_float_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "float-sequence"
    _write_training_sidecars(root)
    _rewrite_event_rows(
        root / "server_events.jsonl",
        lambda rows: [
            row.__setitem__("sequence", float(row["sequence"])) for row in rows
        ],
    )

    with pytest.raises(ValueError, match="type|integer|sequence"):
        _validate_observer_sidecars(
            root, enabled=True, expected_binding=_expected_training_binding()
        )


def _resource_payload(start_ns: int, end_ns: int, *, rss: int = 1024) -> dict:
    return {
        "root_pid": 101,
        "sampler_pid_excluded": 202,
        "pids": [101],
        "process_identities": [
            {"pid": 101, "create_time": 1.0, "identity_available": True}
        ],
        "rss_tree_bytes": rss,
        "rss_tree_peak_bytes": max(rss, 1024),
        "process_count_tree": 1,
        "thread_count_tree": 1,
        "cpu_time_tree_seconds": 1.0,
        "cpu_time_tree_delta_seconds": 0.1,
        "cpu_percent_tree_one_core_scale": 10.0,
        "cpu_percent_tree_host_scale": 2.5,
        "logical_cpu_count": 4,
        "sample_interval_start_monotonic_ns": start_ns,
        "sample_interval_end_monotonic_ns": end_ns,
        "sample_interval_wall_ns": end_ns - start_ns,
        "sample_errors": [],
        "cpu_temperature_c": 45.0,
        "cpu_temperature_available": True,
        "cpu_temperature_source": "sysfs",
        "vcgencmd_available": False,
        "throttled_raw": None,
        "throttled_bits": None,
        "throttled_available": False,
        "thermal_errors": [],
    }


def _write_resource_sidecar(root: Path, *, rss: int = 1024) -> None:
    root.mkdir()
    observer = JsonlObserver(_resource_identity("C2"), root / "resource.jsonl")
    observer.emit(
        "resource_sample",
        round_idx=None,
        client_id="C2",
        status="succeeded",
        payload=_resource_payload(100, 200, rss=rss),
    )
    observer.emit(
        "resource_sample",
        round_idx=None,
        client_id="C2",
        status="succeeded",
        payload=_resource_payload(200, 300, rss=rss),
    )
    observer.emit(
        "resource_sampler_end",
        round_idx=None,
        client_id="C2",
        status="succeeded",
        payload={
            "root_pid": 101,
            "sampler_pid": 202,
            "shutdown_reason": "stop_file",
            "shutdown_error": None,
            "sample_count": 2,
            "sampler_cpu_user_seconds": 0.1,
            "sampler_cpu_system_seconds": 0.1,
            "sampler_rss_peak_bytes": 4096,
            "sampler_rss_peak_available": True,
            "sampler_rss_peak_method": "test",
            "sampler_rss_peak_error": None,
            "observer_cost_values_scope": "before_resource_sampler_end_emit",
            "observer_event_encode_ns": 1,
            "observer_io_write_ns": 1,
            "observer_fsync_ns": 1,
            "observer_event_bytes_written": 1,
            "observer_event_count": 2,
            "observer_close_summary_path": "resource.close.json",
            "observer_close_summary_is_authoritative": True,
        },
    )
    observer.close()


def test_formal_resource_sidecar_requires_exact_schema_finite_nonnegative_values(
    tmp_path: Path,
) -> None:
    validator = getattr(gate_module, "_validate_formal_resource_sidecar", None)
    assert callable(validator), "missing strict formal resource-sidecar validator"
    good = tmp_path / "good"
    _write_resource_sidecar(good)
    result = validator(good, "C2", expected_binding=_expected_resource_binding("C2"))
    assert result["sample_count"] == 2
    bad = tmp_path / "bad"
    _write_resource_sidecar(bad, rss=-1)
    with pytest.raises(ValueError, match="nonnegative|rss"):
        validator(bad, "C2", expected_binding=_expected_resource_binding("C2"))


def test_formal_resource_sidecar_rejects_consistent_wrong_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resource-wrong-binding"
    _write_resource_sidecar(root)
    _rewrite_event_rows(
        root / "resource.jsonl",
        lambda rows: [
            row.__setitem__("confirmation_commit", "f" * 40) for row in rows
        ],
    )

    with pytest.raises(ValueError, match="binding|confirmation_commit"):
        gate_module._validate_formal_resource_sidecar(
            root, "C2", expected_binding=_expected_resource_binding("C2")
        )


def test_formal_resource_sidecar_rejects_float_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resource-float-sequence"
    _write_resource_sidecar(root)
    _rewrite_event_rows(
        root / "resource.jsonl",
        lambda rows: [
            row.__setitem__("sequence", float(row["sequence"])) for row in rows
        ],
    )
    with pytest.raises(ValueError, match="type|integer|sequence"):
        gate_module._validate_formal_resource_sidecar(
            root, "C2", expected_binding=_expected_resource_binding("C2")
        )


def test_formal_resource_sidecar_rejects_later_row_binding_type_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resource-later-binding-drift"
    _write_resource_sidecar(root)

    def mutate(rows: list[dict]) -> None:
        assert rows[0]["training_seed"] == 42
        rows[1]["training_seed"] = 42.0

    _rewrite_event_rows(root / "resource.jsonl", mutate)
    with pytest.raises(ValueError, match="type|identity|training_seed"):
        gate_module._validate_formal_resource_sidecar(
            root, "C2", expected_binding=_expected_resource_binding("C2")
        )


def test_formal_resource_sidecar_rejects_float_overhead_total(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resource-float-overhead-total"
    _write_resource_sidecar(root)

    def mutate(rows: list[dict]) -> None:
        overhead = next(row for row in rows if row["event_type"] == "observer_overhead")
        total = overhead["payload"]["observer_total_ns"]
        overhead["payload"]["observer_total_ns"] = float(total)

    _rewrite_event_rows(root / "resource.jsonl", mutate)
    with pytest.raises(ValueError, match="type|integer|overhead"):
        gate_module._validate_formal_resource_sidecar(
            root, "C2", expected_binding=_expected_resource_binding("C2")
        )


def test_formal_resource_sidecar_rejects_float_sampler_end_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resource-float-end-count"
    _write_resource_sidecar(root)

    def mutate(rows: list[dict]) -> None:
        sampler_end = next(
            row for row in rows if row["event_type"] == "resource_sampler_end"
        )
        sampler_end["payload"]["sample_count"] = float(
            sampler_end["payload"]["sample_count"]
        )

    _rewrite_event_rows(root / "resource.jsonl", mutate)
    with pytest.raises(ValueError, match="type|integer|sample_count"):
        gate_module._validate_formal_resource_sidecar(
            root, "C2", expected_binding=_expected_resource_binding("C2")
        )


def test_portable_evidence_path_rejects_absolute_or_escaping_paths(
    tmp_path: Path,
) -> None:
    portable = getattr(gate_module, "_portable_evidence_path", None)
    assert callable(portable), "missing portable evidence-path normalizer"
    root = tmp_path / "attempt"
    root.mkdir()
    inside = root / "raw" / "events.jsonl"
    inside.parent.mkdir()
    assert portable(root, inside) == "raw/events.jsonl"
    with pytest.raises(ValueError, match="outside|escape"):
        portable(root, tmp_path / "outside.jsonl")


def test_formal_capture_includes_raw_checkpoint_sha_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "formal"
    training = attempt / "raw" / "ecs" / "training"
    training.mkdir(parents=True)
    for name in ("server_latest.pth", "server_latest_adapted.pth"):
        (training / name).write_bytes(name.encode("ascii"))
    (attempt / "raw" / "ecs" / "common_trace.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    for round_idx in (1, 2):
        for stem in ("prototype_stats", "semantic_protos"):
            (training / f"{stem}_round_{round_idx:03d}.json").write_text(
                "{}", encoding="utf-8"
            )
        (training / f"client_stats_round_{round_idx:03d}.json").write_text(
            json.dumps({"clients": [{"client_id": 1}, {"client_id": 2}]}),
            encoding="utf-8",
        )
        (training / f"domain_adapt_round_{round_idx:03d}.json").write_text(
            "{}", encoding="utf-8"
        )
    monkeypatch.setattr(
        gate_module,
        "tensor_fingerprint",
        lambda path: {
            "artifact_sha256": ("a" if "adapted" not in path.name else "b") * 64,
            "content_sha256": "c" * 64,
            "comparison": {"path_kind": path.name},
        },
    )
    monkeypatch.setattr(
        gate_module,
        "json_fingerprint",
        lambda *_args, **_kwargs: {
            "artifact_sha256": "d" * 64,
            "content_sha256": "e" * 64,
            "comparison": {},
        },
    )
    monkeypatch.setattr(
        gate_module,
        "_common_trace_fingerprint",
        lambda _path: {
            "artifact_sha256": "f" * 64,
            "content_sha256": "1" * 64,
            "comparison": {},
        },
    )
    monkeypatch.setattr(
        gate_module,
        "_fixed_adapted_logits",
        lambda *_args, **_kwargs: {
            "content_sha256": "2" * 64,
            "comparison": {},
        },
    )
    artifacts = gate_module._capture_formal_artifacts(attempt)
    assert artifacts["final_aggregated_checkpoint_raw"]["comparison"] == {
        "raw_file_sha256": "a" * 64
    }
    assert artifacts["final_adapted_checkpoint_raw"]["comparison"] == {
        "raw_file_sha256": "b" * 64
    }
