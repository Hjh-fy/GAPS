"""Serialization-independent fingerprints for ordered model state content."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _update_array(digest: "hashlib._Hash", key: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise RuntimeError(f"FAIL_CLOSED non-finite tensor content for {key}")
    digest.update(str(key).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def ordered_array_content_fingerprint(
    keys: Sequence[str], arrays: Iterable[np.ndarray]
) -> str:
    """Hash ordered key/dtype/shape/content without container metadata."""
    values = list(arrays)
    if len(keys) != len(values):
        raise RuntimeError("FAIL_CLOSED ordered state key/value length mismatch")
    if len(set(keys)) != len(keys):
        raise RuntimeError("FAIL_CLOSED duplicate ordered state key")
    digest = hashlib.sha256()
    for key, value in zip(keys, values):
        _update_array(digest, key, np.asarray(value))
    return digest.hexdigest()


def ordered_state_content_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a mapping in its iteration order; order is part of identity."""
    keys = list(state.keys())
    arrays = [value.detach().cpu().numpy() for value in state.values()]
    return ordered_array_content_fingerprint(keys, arrays)


def whole_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_state(path: str | Path) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), Mapping):
        raise RuntimeError("FAIL_CLOSED checkpoint does not contain model_state mapping")
    return payload["model_state"], payload


def checkpoint_provenance(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    state, payload = load_checkpoint_state(checkpoint_path)
    parameter_keys = payload.get("parameter_keys")
    if parameter_keys is not None and list(parameter_keys) != list(state.keys()):
        raise RuntimeError("FAIL_CLOSED checkpoint parameter_keys do not match model_state order")
    return {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "formal_round": int(payload.get("round", -1)),
        "ordered_state_content_fingerprint": ordered_state_content_fingerprint(state),
        "whole_file_sha256": whole_file_sha256(checkpoint_path),
        "equality_basis": "ordered_state_content_fingerprint",
        "whole_file_sha256_role": "provenance_only",
        "parameter_count": len(state),
    }
