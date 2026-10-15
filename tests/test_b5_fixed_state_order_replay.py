from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import scripts.diagnose_b5_fixed_state_order_replay as diagnostic
import scripts.run_iotj_formal_off_repeat_probe as formal_repeat
from gaps_flower.strategy import canonicalize_fit_results


class _ReplayFakeTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = nn.Linear(2, 1, bias=False)
        self.semantic_protos = nn.ParameterDict(
            {"0,0": nn.Parameter(torch.tensor([1.0, 2.0]))}
        )
        self.device_residuals = nn.ParameterDict()

    def run_adaptation(self, **kwargs):
        client_ids = [int(value) for value in kwargs["client_ids"]]
        for client_id, residual in zip(client_ids, kwargs["client_residuals"]):
            self.device_residuals[str(client_id)] = nn.Parameter(residual.clone())
        with torch.no_grad():
            self.model.weight.add_(0.25)
        return self.model, {"num_steps": int(kwargs["num_steps"])}


def test_one_step_da_replay_records_exact_state_and_rng_boundary() -> None:
    clients = {
        1: {
            "mus": {(0, 0): torch.tensor([1.0, 2.0])},
            "counts": {(0, 0): 2},
            "residual": torch.tensor([0.1, 0.2]),
            "num_examples": 3,
        },
        2: {
            "mus": {(0, 0): torch.tensor([2.0, 3.0])},
            "counts": {(0, 0): 4},
            "residual": torch.tensor([0.3, 0.4]),
            "num_examples": 5,
        },
    }

    first = diagnostic.run_one_step_da_replay(
        _ReplayFakeTrainer, clients, [2, 1], seed=42
    )
    repeat = diagnostic.run_one_step_da_replay(
        _ReplayFakeTrainer, clients, [2, 1], seed=42
    )

    assert first == repeat
    assert first["client_order"] == [2, 1]
    assert first["diagnostics"] == {"num_steps": 1}
    assert first["rng"]["pre_adaptation"]["torch_cpu"]["raw_sha256"]
    assert first["final_parameters"]["model"]["content_sha256"]
    assert first["final_parameters"]["device_residuals"]["key_order"] == ["2", "1"]


def test_formal_off_repeat_probe_uses_separate_noncanonical_attempt_id() -> None:
    run_id = "c12_to_c5__b5__s42"

    assert formal_repeat.noncanonical_attempt_id(run_id, 997) == (
        "c12_to_c5__b5__s42__a997"
    )
    with pytest.raises(ValueError, match="900.*997"):
        formal_repeat.noncanonical_attempt_id(run_id, 998)
    with pytest.raises(ValueError, match="run_id"):
        formal_repeat.noncanonical_attempt_id("bad", 997)


def test_fit_results_are_canonicalized_by_uploaded_client_id() -> None:
    c1 = (SimpleNamespace(cid="proxy-z"), SimpleNamespace(metrics={"client_id": 1}))
    c2 = (SimpleNamespace(cid="proxy-a"), SimpleNamespace(metrics={"client_id": 2}))

    ordered = canonicalize_fit_results([c2, c1])

    assert ordered == [c1, c2]
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_fit_results([c1, c1])
    with pytest.raises(ValueError, match="client_id"):
        canonicalize_fit_results(
            [(SimpleNamespace(cid="proxy-x"), SimpleNamespace(metrics={}))]
        )


def test_fixed_state_order_replay_module_exists() -> None:
    assert importlib.util.find_spec(
        "scripts.diagnose_b5_fixed_state_order_replay"
    ) is not None, "missing fixed-state B5 order-replay diagnostic"


def _typed(value: object) -> dict[str, object]:
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    raise TypeError(type(value).__name__)


def _metric_mapping(
    client_id: int,
    *,
    residual: list[float] | None = None,
    include_residual_metric: bool = True,
) -> dict:
    values = OrderedDict(
        [
            ("client_id", _typed(client_id)),
            ("num_examples", _typed(2360)),
            ("prototype_json", _typed(json.dumps({"0,0": [float(client_id)]}))),
            ("class_phase_counts_json", _typed(json.dumps({"0,0": 2}))),
            ("fit_seconds", _typed(0.0)),
        ]
    )
    if include_residual_metric:
        values["device_residual_json"] = _typed(json.dumps(residual or []))
    keys = list(values)
    return {
        "key_order": keys,
        "keys": keys,
        "types": {key: value["type"] for key, value in values.items()},
        "values": values,
    }


def _fit_res_row(
    client_id: int,
    *,
    round_idx: int = 2,
    residual=None,
    include_residual_metric: bool = True,
) -> dict:
    return {
        "record_type": "fit_res",
        "round": round_idx,
        "proxy_id": f"proxy-{client_id}",
        "client_id": f"C{client_id}",
        "trace": {
            "comparison": {
                "status": {"code": 0, "message": ""},
                "num_examples": 2360,
                "parameters": {"key_order": [], "tensors": []},
                "metrics": _metric_mapping(
                    client_id,
                    residual=residual,
                    include_residual_metric=include_residual_metric,
                ),
                "logical": {},
                "normalized_application_message_bytes": 1,
                "normalized_application_message_sha256": "a" * 64,
            }
        },
    }


def _write_trace(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_direct_script_help_bootstraps_repo_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "diagnose_b5_fixed_state_order_replay.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--reference-root" in result.stdout


def test_round_one_absent_residual_metric_uses_production_empty_default(
    tmp_path: Path,
) -> None:
    traces = []
    for name in ("off", "on"):
        traces.append(
            diagnostic.read_round_fit_res(
                _write_trace(
                    tmp_path / f"{name}.jsonl",
                    [
                        _fit_res_row(
                            1,
                            round_idx=1,
                            include_residual_metric=False,
                        ),
                        _fit_res_row(
                            2,
                            round_idx=1,
                            include_residual_metric=False,
                        ),
                    ],
                ),
                round_idx=1,
            )
        )

    result = diagnostic._validate_residual_transition(
        traces[0],
        traces[1],
        {"device_residual_count": 0, "residual_loss": 0.0},
        {"device_residual_count": 0, "residual_loss": 0.0},
    )

    assert result["round_1_fit_res_residuals_empty"] is True
    assert result["round_1_fit_res_residual_metric_presence"] == {
        "off": {"1": False, "2": False},
        "on": {"1": False, "2": False},
    }


def test_typed_metric_unwrap_preserves_exact_scalar_types() -> None:
    assert diagnostic.unwrap_typed_scalar({"type": "bool", "value": True}) is True
    assert diagnostic.unwrap_typed_scalar({"type": "int", "value": 7}) == 7
    assert type(diagnostic.unwrap_typed_scalar({"type": "int", "value": 7})) is int
    assert diagnostic.unwrap_typed_scalar({"type": "float", "value": 7.0}) == 7.0
    assert diagnostic.unwrap_typed_scalar({"type": "str", "value": "x"}) == "x"
    assert diagnostic.unwrap_typed_scalar(
        {"type": "bytes", "value_base64": "AAE="}
    ) == b"\x00\x01"
    with pytest.raises(ValueError, match="type|finite"):
        diagnostic.unwrap_typed_scalar({"type": "float", "value": float("nan")})


def test_round_fit_res_reader_ignores_post_records_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    valid = _write_trace(
        tmp_path / "valid.jsonl",
        [
            _fit_res_row(2, residual=[0.0] * 64),
            {**_fit_res_row(2, residual=[0.0] * 64), "record_type": "fit_res_post_observer"},
            _fit_res_row(1, residual=[0.0] * 64),
        ],
    )
    parsed = diagnostic.read_round_fit_res(valid, round_idx=2)
    assert parsed["arrival_order"] == [2, 1]
    assert list(parsed["clients"]) == [2, 1]
    assert parsed["clients"][2]["num_examples"] == 2360
    assert len(json.loads(parsed["clients"][2]["metrics"]["device_residual_json"])) == 64

    duplicate = _write_trace(
        tmp_path / "duplicate.jsonl",
        [_fit_res_row(2), _fit_res_row(2), _fit_res_row(1)],
    )
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic.read_round_fit_res(duplicate, round_idx=2)


def test_off_on_cross_validation_rejects_value_mismatch_and_binds_order(
    tmp_path: Path,
) -> None:
    off = diagnostic.read_round_fit_res(
        _write_trace(
            tmp_path / "off.jsonl",
            [_fit_res_row(2, residual=[2.0] * 64), _fit_res_row(1, residual=[1.0] * 64)],
        ),
        round_idx=2,
    )
    on = diagnostic.read_round_fit_res(
        _write_trace(
            tmp_path / "on.jsonl",
            [_fit_res_row(1, residual=[1.0] * 64), _fit_res_row(2, residual=[2.0] * 64)],
        ),
        round_idx=2,
    )
    result = diagnostic.cross_validate_fit_res(off, on)
    assert result["off_arrival_order"] == [2, 1]
    assert result["on_arrival_order"] == [1, 2]
    assert result["per_client_typed_values_equal"] is True

    changed = json.loads(json.dumps(on))
    changed_value = json.dumps({"0,0": [99.0]})
    changed["clients"]["1"]["metrics"]["prototype_json"] = changed_value
    changed["clients"]["1"]["typed_metrics"]["values"]["prototype_json"][
        "value"
    ] = changed_value
    with pytest.raises(ValueError, match="client.*differ|mismatch"):
        diagnostic.cross_validate_fit_res(off, changed)


def test_semantic_reconstruction_matches_float32_production_order_and_new_key() -> None:
    round_one = {
        "proto_ema_alpha": 0.8,
        "semantic_protos": OrderedDict([("0,0", [1.0, 2.0])]),
    }
    round_two = {
        "global_prototypes": OrderedDict(
            [("0,0", [3.0, 6.0]), ("1,0", [5.0, 7.0])]
        )
    }
    checkpoint = OrderedDict([("0,0", torch.tensor([1.0, 2.0]))])

    reconstructed = diagnostic.reconstruct_semantic_protos(
        round_one,
        round_two,
        config_alpha=0.8,
        checkpoint_semantic_protos=checkpoint,
    )

    old = torch.tensor([1.0, 2.0], dtype=torch.float32)
    new = torch.tensor([3.0, 6.0], dtype=torch.float32)
    assert torch.equal(reconstructed["0,0"], 0.8 * old + (1.0 - 0.8) * new)
    assert torch.equal(reconstructed["1,0"], torch.tensor([5.0, 7.0]))
    with pytest.raises(ValueError, match="alpha"):
        diagnostic.reconstruct_semantic_protos(
            round_one,
            round_two,
            config_alpha=0.7,
            checkpoint_semantic_protos=checkpoint,
        )


def test_recursive_manifest_and_guard_revalidate_success_and_failure(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "a.bin").write_bytes(b"alpha")
    before = diagnostic.recursive_manifest(reference)
    assert before == diagnostic.recursive_manifest(reference)

    result, guarded_before, guarded_after = diagnostic.run_with_reference_guard(
        reference, lambda: "ok"
    )
    assert result == "ok"
    assert guarded_before == guarded_after == before

    def injected_failure() -> None:
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        diagnostic.run_with_reference_guard(reference, injected_failure)
    assert diagnostic.recursive_manifest(reference) == before

    def mutate_then_fail() -> None:
        (reference / "a.bin").write_bytes(b"changed")
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="reference root changed"):
        diagnostic.run_with_reference_guard(reference, mutate_then_fail)


def test_root_validation_rejects_existing_linked_and_overlapping_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    sibling = tmp_path / "output"
    diagnostic.validate_roots(reference, sibling)

    sibling.mkdir()
    with pytest.raises((FileExistsError, ValueError), match="exist"):
        diagnostic.validate_roots(reference, sibling)

    with pytest.raises(ValueError, match="overlap|ancestor"):
        diagnostic.validate_roots(reference, reference / "child")

    linked = tmp_path / "linked-output"
    monkeypatch.setattr(
        diagnostic,
        "path_is_link_or_reparse",
        lambda path: Path(path) == linked,
    )
    with pytest.raises(ValueError, match="link|reparse"):
        diagnostic.validate_roots(reference, linked)


def test_nonfinite_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        diagnostic.finite_json({"nested": [1.0, float("inf")]})


def _replay_record(
    *, proto_hex: str, gradient_hex: str, multiset: str = "m", rng_unchanged: bool = True
) -> dict:
    return {
        "losses": {
            "proto_loss": {"raw_hex": proto_hex},
            "mmd_proto_loss": {"raw_hex": "00"},
            "residual_loss": {"raw_hex": "00"},
        },
        "gradients": {
            "semantic_protos": {"0,0": {"raw_hex": gradient_hex}},
            "device_residuals": {"1": {"raw_hex": "00"}},
        },
        "input_fingerprints": {"paired_multiset_sha256": multiset},
        "rng": {"unchanged": rng_unchanged},
    }


@pytest.mark.parametrize(
    ("replays", "expected"),
    [
        (
            {
                "order_21_a": _replay_record(proto_hex="01", gradient_hex="aa"),
                "order_21_b": _replay_record(proto_hex="01", gradient_hex="aa"),
                "order_12": _replay_record(proto_hex="02", gradient_hex="aa"),
            },
            "reconstructed_initial_proto_loss_order_sensitive",
        ),
        (
            {
                "order_21_a": _replay_record(proto_hex="01", gradient_hex="aa"),
                "order_21_b": _replay_record(proto_hex="01", gradient_hex="aa"),
                "order_12": _replay_record(proto_hex="01", gradient_hex="aa"),
            },
            "order_not_causal_at_proto_loss_stage",
        ),
        (
            {
                "order_21_a": _replay_record(proto_hex="01", gradient_hex="aa"),
                "order_21_b": _replay_record(proto_hex="02", gradient_hex="aa"),
                "order_12": _replay_record(proto_hex="01", gradient_hex="aa"),
            },
            "unresolved_fail_closed",
        ),
    ],
)
def test_classification_is_exact_and_three_way(replays: dict, expected: str) -> None:
    assert diagnostic.classify_replays(replays)["classification"] == expected


def test_production_proto_loss_detects_order_sensitive_reduction_without_rng() -> None:
    semantic = OrderedDict(
        (key, torch.zeros(1, dtype=torch.float32))
        for key in ("0,0", "1,0", "2,0")
    )
    clients = {
        1: {
            "mus": OrderedDict(
                [
                    ((1, 0), torch.tensor([1.0], dtype=torch.float32)),
                    ((2, 0), torch.tensor([1.0], dtype=torch.float32)),
                ]
            ),
            "counts": {(1, 0): 2, (2, 0): 2},
            "residual": torch.zeros(1, dtype=torch.float32),
            "num_examples": 1,
        },
        2: {
            "mus": OrderedDict(
                [((0, 0), torch.tensor([4096.0], dtype=torch.float32))]
            ),
            "counts": {(0, 0): 2},
            "residual": torch.zeros(1, dtype=torch.float32),
            "num_examples": 1,
        },
    }
    loss_weights = {"proto": 0.05, "proto_mmd": 0.0, "residual": 0.1}

    order_21_a = diagnostic.run_proto_loss_replay(
        semantic, clients, [2, 1], loss_weights=loss_weights
    )
    order_21_b = diagnostic.run_proto_loss_replay(
        semantic, clients, [2, 1], loss_weights=loss_weights
    )
    order_12 = diagnostic.run_proto_loss_replay(
        semantic, clients, [1, 2], loss_weights=loss_weights
    )

    assert order_21_a["rng"]["unchanged"] is True
    assert order_21_a["losses"] == order_21_b["losses"]
    assert order_21_a["gradients"] == order_21_b["gradients"]
    assert (
        order_21_a["input_fingerprints"]["paired_multiset_sha256"]
        == order_12["input_fingerprints"]["paired_multiset_sha256"]
    )
    assert (
        order_21_a["input_fingerprints"]["ordered_list_sha256"]
        != order_12["input_fingerprints"]["ordered_list_sha256"]
    )
    classification = diagnostic.classify_replays(
        {
            "order_21_a": order_21_a,
            "order_21_b": order_21_b,
            "order_12": order_12,
        }
    )
    assert (
        classification["classification"]
        == "reconstructed_initial_proto_loss_order_sensitive"
    )


def test_proto_replay_reports_production_gradient_clipping_order() -> None:
    semantic = OrderedDict(
        (key, torch.zeros(1, dtype=torch.float32))
        for key in ("0,0", "1,0", "2,0")
    )
    clients = {
        1: {
            "mus": OrderedDict([((0, 0), torch.tensor([4096.0]))]),
            "counts": {(0, 0): 2},
            "residual": torch.zeros(1),
            "num_examples": 1,
        },
        2: {
            "mus": OrderedDict([((1, 0), torch.tensor([1.0]))]),
            "counts": {(1, 0): 2},
            "residual": torch.zeros(1),
            "num_examples": 1,
        },
    }

    replay = diagnostic.run_proto_loss_replay(
        semantic,
        clients,
        [2, 1],
        loss_weights={"proto": 0.05, "proto_mmd": 0.0, "residual": 0.1},
    )

    clipping = replay["gradient_clipping"]
    assert clipping["max_norm"] == 5.0
    assert clipping["parameter_order"] == [
        "semantic_protos.0,0",
        "semantic_protos.1,0",
        "semantic_protos.2,0",
        "device_residuals.2",
        "device_residuals.1",
    ]
    assert clipping["pre_clip_total_norm"]["dtype"] == "torch.float32"
    assert set(clipping["post_clip_gradients"]) == set(clipping["parameter_order"])
