from __future__ import annotations

import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from scripts.run_iotj_observer_equivalence_gate import (
    VOLATILE_JSON_PATHS,
    compare_fingerprints,
    json_fingerprint,
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
