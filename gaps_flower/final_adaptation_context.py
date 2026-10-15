"""Fail-closed round25 source context for canonical final-only A4 adaptation.

The context contains source-side state only.  Target calibration arrays and
labels are deliberately outside this API and are opened by the later target
adaptation stage.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from gaps_flower.posthoc_commissioning import ordered_state_fingerprint, sha256_file


SCHEMA_VERSION = "iotj.canonical_v1.final_adaptation_context.v1"
TARGET_ACCESS_NONE = {
    "target_x": False,
    "target_class": False,
    "target_phase": False,
    "target_concentration": False,
    "target_test": False,
}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), Mapping):
        raise RuntimeError("FAIL_CLOSED source checkpoint lacks model_state")
    return payload


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _encode_tensor(value: Any, *, name: str) -> dict[str, Any]:
    tensor = _tensor(value, name=name)
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def _decode_tensor(payload: Mapping[str, Any], *, name: str) -> torch.Tensor:
    if payload.get("dtype") != "float32":
        raise RuntimeError(f"FAIL_CLOSED {name} dtype must be float32")
    tensor = _tensor(payload.get("values"), name=name)
    declared_shape = tuple(int(item) for item in payload.get("shape", ()))
    if tuple(tensor.shape) != declared_shape:
        raise RuntimeError(f"FAIL_CLOSED {name} shape does not match its declaration")
    return tensor


def _prototype_key(value: Any) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        return f"{int(value[0])},{int(value[1])}"
    text = str(value).strip().replace("_", ",")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid class-phase prototype key: {value!r}")
    return f"{int(parts[0])},{int(parts[1])}"


def _runtime_key(value: str) -> tuple[int, int]:
    first, second = value.split(",", 1)
    return int(first), int(second)


def _encode_prototypes(values: Mapping[Any, Any], *, name: str) -> dict[str, Any]:
    encoded: dict[str, Any] = {}
    for key, tensor in values.items():
        canonical = _prototype_key(key)
        if canonical in encoded:
            raise ValueError(f"duplicate canonical prototype key: {canonical}")
        encoded[canonical] = _encode_tensor(tensor, name=f"{name}[{canonical}]")
    return {key: encoded[key] for key in sorted(encoded)}


def _encode_counts(values: Mapping[Any, Any]) -> dict[str, int]:
    encoded: dict[str, int] = {}
    for key, count in values.items():
        canonical = _prototype_key(key)
        observed = int(count)
        if observed <= 0:
            raise ValueError(f"class-phase count must be positive: {canonical}")
        encoded[canonical] = observed
    return {key: encoded[key] for key in sorted(encoded)}


def build_final_adaptation_context(
    *,
    round_id: int,
    run_name: str,
    checkpoint_path: Path,
    checkpoint_state: Mapping[str, torch.Tensor],
    semantic_protos: Mapping[Any, Any],
    semantic_proto_vars: Mapping[Any, Any],
    client_mus: Sequence[Mapping[Any, Any]],
    client_counts: Sequence[Mapping[Any, Any]],
    client_ids: Sequence[int],
    client_residuals: Sequence[Any | None],
    client_weights: Any,
) -> dict[str, Any]:
    """Build the source-only context bound to one exact round25 state."""

    if int(round_id) != 25:
        raise ValueError("final adaptation context must bind the fixed round25 endpoint")
    lengths = {
        len(client_mus),
        len(client_counts),
        len(client_ids),
        len(client_residuals),
        int(torch.as_tensor(client_weights).numel()),
    }
    if 0 in lengths:
        raise ValueError("final adaptation context requires a nonempty client payload")
    if len(lengths) != 1:
        raise ValueError("final adaptation client payload lengths must match")
    if len(set(int(value) for value in client_ids)) != len(client_ids):
        raise ValueError("final adaptation client IDs must be unique")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    if int(checkpoint.get("round", -1)) != 25:
        raise RuntimeError("FAIL_CLOSED source checkpoint is not round25")
    supplied_state = OrderedDict((str(key), value) for key, value in checkpoint_state.items())
    stored_state = OrderedDict((str(key), value) for key, value in checkpoint["model_state"].items())
    supplied_fingerprint = ordered_state_fingerprint(supplied_state)
    stored_fingerprint = ordered_state_fingerprint(stored_state)
    if supplied_fingerprint != stored_fingerprint:
        raise RuntimeError("FAIL_CLOSED supplied ordered state fingerprint differs from checkpoint")

    weights = _tensor(client_weights, name="client_weights").reshape(-1)
    if bool((weights < 0).any()) or not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("client_weights must be nonnegative and sum to one")

    clients: list[dict[str, Any]] = []
    order = sorted(range(len(client_ids)), key=lambda index: int(client_ids[index]))
    for index in order:
        prototypes = _encode_prototypes(client_mus[index], name=f"client_{client_ids[index]}_prototypes")
        counts = _encode_counts(client_counts[index])
        if set(prototypes) != set(counts):
            raise ValueError("client prototype/count keys must match")
        residual = client_residuals[index]
        clients.append(
            {
                "client_id": int(client_ids[index]),
                "prototypes": prototypes,
                "counts": counts,
                "residual": (
                    None
                    if residual is None
                    else _encode_tensor(residual, name=f"client_{client_ids[index]}_residual")
                ),
                "weight": float(weights[index].item()),
            }
        )

    encoded_semantic = _encode_prototypes(semantic_protos, name="semantic_protos")
    encoded_vars = _encode_prototypes(semantic_proto_vars, name="semantic_proto_vars")
    availability = {
        "client_prototypes": any(bool(item["prototypes"]) for item in clients),
        "client_residuals": any(item["residual"] is not None for item in clients),
        "semantic_prototypes": bool(encoded_semantic),
        "two_client_prototypes": sum(bool(item["prototypes"]) for item in clients) >= 2,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "round_id": 25,
        "run_name": str(run_name),
        "source_checkpoint": {
            "path_at_capture": str(checkpoint_path),
            "file_sha256_provenance": sha256_file(checkpoint_path),
            "ordered_state_fingerprint": stored_fingerprint,
            "parameter_keys": list(stored_state),
        },
        "semantic_protos": encoded_semantic,
        "semantic_proto_vars": encoded_vars,
        "clients": clients,
        "loss_input_availability": availability,
        "target_access": dict(TARGET_ACCESS_NONE),
    }


def write_final_adaptation_context(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_final_adaptation_context(path: Path, checkpoint_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or int(payload.get("round_id", -1)) != 25:
        raise RuntimeError("FAIL_CLOSED invalid round25 final adaptation context")
    if payload.get("target_access") != TARGET_ACCESS_NONE:
        raise RuntimeError("FAIL_CLOSED source context records target access")
    checkpoint = _load_checkpoint(Path(checkpoint_path))
    observed_state = OrderedDict(
        (str(key), value) for key, value in checkpoint["model_state"].items()
    )
    observed_fingerprint = ordered_state_fingerprint(observed_state)
    expected_fingerprint = payload["source_checkpoint"]["ordered_state_fingerprint"]
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError("FAIL_CLOSED ordered state fingerprint mismatch")

    semantic = {
        key: _decode_tensor(value, name=f"semantic_protos[{key}]")
        for key, value in payload["semantic_protos"].items()
    }
    semantic_vars = {
        key: _decode_tensor(value, name=f"semantic_proto_vars[{key}]")
        for key, value in payload["semantic_proto_vars"].items()
    }
    client_ids: list[int] = []
    client_mus: list[dict[tuple[int, int], torch.Tensor]] = []
    client_counts: list[dict[tuple[int, int], int]] = []
    client_residuals: list[torch.Tensor | None] = []
    weights: list[float] = []
    for client in payload["clients"]:
        client_ids.append(int(client["client_id"]))
        client_mus.append(
            {
                _runtime_key(key): _decode_tensor(value, name=f"client_{client['client_id']}_prototypes[{key}]")
                for key, value in client["prototypes"].items()
            }
        )
        client_counts.append(
            {_runtime_key(key): int(value) for key, value in client["counts"].items()}
        )
        client_residuals.append(
            None
            if client["residual"] is None
            else _decode_tensor(client["residual"], name=f"client_{client['client_id']}_residual")
        )
        weights.append(float(client["weight"]))
    return {
        **payload,
        "semantic_protos": semantic,
        "semantic_proto_vars": semantic_vars,
        "client_ids": client_ids,
        "client_mus": client_mus,
        "client_counts": client_counts,
        "client_residuals": client_residuals,
        "client_weights": torch.tensor(weights, dtype=torch.float32),
        "checkpoint_file_sha256_observed": sha256_file(Path(checkpoint_path)),
        "checkpoint_file_sha256_matches_capture": (
            sha256_file(Path(checkpoint_path))
            == payload["source_checkpoint"]["file_sha256_provenance"]
        ),
    }


def validate_a4_context_loss_inputs(
    payload: Mapping[str, Any],
    *,
    configured_weights: Mapping[str, float],
    expected_context_availability: Mapping[str, bool],
) -> dict[str, Any]:
    availability = dict(payload.get("loss_input_availability", {}))
    requirements = {
        "proto_anchor": ("semantic_prototypes",),
        "proto_loss": ("semantic_prototypes", "client_prototypes"),
        "consistency": ("semantic_prototypes",),
        "device_residual": ("client_residuals",),
        "proto_mmd": ("two_client_prototypes",),
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    required_flags = {
        item for needed in requirements.values() for item in needed
    }
    if set(expected_context_availability) != required_flags:
        raise ValueError(
            "expected_context_availability must declare the complete C0-A baseline"
        )
    for loss_name, needed in requirements.items():
        weight = float(configured_weights.get(loss_name, 0.0))
        present = all(bool(availability.get(name, False)) for name in needed)
        baseline_present = all(
            bool(expected_context_availability.get(name, False)) for name in needed
        )
        parity = all(
            bool(availability.get(name, False))
            == bool(expected_context_availability.get(name, False))
            for name in needed
        )
        if weight != 0.0 and not parity:
            missing.append(loss_name)
        rows.append(
            {
                "loss_name": loss_name,
                "configured_weight": weight,
                "required_context_inputs": list(needed),
                "input_available": present,
                "baseline_input_available": baseline_present,
                "availability_parity": parity,
            }
        )
    if missing:
        raise RuntimeError(
            "FAIL_CLOSED nonzero A4 loss differs from C0-A context availability: "
            + ",".join(missing)
        )
    return {"status": "PASS", "missing_nonzero_inputs": [], "terms": rows}
