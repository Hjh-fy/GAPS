from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch


def _state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        [
            ("encoder.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("classifier.bias", torch.tensor([3.0], dtype=torch.float32)),
        ]
    )


def test_ordered_state_fingerprint_matches_semantic_content_across_torch_containers(
    tmp_path: Path,
) -> None:
    from gaps_flower.state_fingerprint import (
        checkpoint_provenance,
        ordered_state_content_fingerprint,
    )

    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    torch.save({"round": 25, "model_state": _state(), "note": "first"}, first)
    torch.save({"round": 25, "model_state": _state(), "note": "second"}, second)

    first_info = checkpoint_provenance(first)
    second_info = checkpoint_provenance(second)

    assert first_info["ordered_state_content_fingerprint"] == second_info[
        "ordered_state_content_fingerprint"
    ]
    assert first_info["ordered_state_content_fingerprint"] == ordered_state_content_fingerprint(
        _state()
    )
    assert first_info["whole_file_sha256"] != second_info["whole_file_sha256"]
    assert first_info["equality_basis"] == "ordered_state_content_fingerprint"
    assert first_info["whole_file_sha256_role"] == "provenance_only"


def test_ordered_state_fingerprint_changes_with_order_dtype_shape_or_content() -> None:
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint

    base = _state()
    reversed_state = OrderedDict(reversed(list(base.items())))
    dtype_state = OrderedDict(base)
    dtype_state["encoder.weight"] = dtype_state["encoder.weight"].double()
    shape_state = OrderedDict(base)
    shape_state["encoder.weight"] = shape_state["encoder.weight"].reshape(2, 1)
    content_state = OrderedDict(base)
    content_state["encoder.weight"] = content_state["encoder.weight"] + 1.0

    fingerprints = {
        ordered_state_content_fingerprint(value)
        for value in (base, reversed_state, dtype_state, shape_state, content_state)
    }
    assert len(fingerprints) == 5


def test_ordered_array_fingerprint_rejects_length_mismatch_and_nonfinite() -> None:
    from gaps_flower.state_fingerprint import ordered_array_content_fingerprint

    with pytest.raises(RuntimeError, match="length mismatch"):
        ordered_array_content_fingerprint(["a"], [])
    with pytest.raises(RuntimeError, match="non-finite"):
        ordered_array_content_fingerprint(["a"], [np.array([np.nan], dtype=np.float32)])


def test_import_checkpoint_verifies_ordered_content_and_preserves_source(
    tmp_path: Path,
) -> None:
    from scripts.prepare_iotj_final_classification_inputs import import_checkpoint

    source = tmp_path / "source.pth"
    destination = tmp_path / "inputs" / "source_round25.pth"
    torch.save(
        {
            "round": 25,
            "model_state": _state(),
            "parameter_keys": list(_state()),
            "run_name": "P0A",
        },
        source,
    )
    before = source.read_bytes()

    manifest = import_checkpoint(source, destination)

    assert source.read_bytes() == before
    assert destination.is_file()
    assert manifest["source"]["ordered_state_content_fingerprint"] == manifest["copy"][
        "ordered_state_content_fingerprint"
    ]
    assert manifest["equality_verified"] is True
    assert manifest["formal_round"] == 25
