"""Fail-closed OFF-A/ON/OFF-B numerical equivalence Gate.

This command owns only a deterministic local, synthetic two-client/two-round
Flower fixture.  It never reads a project dataset or a formal checkpoint.  The
real server/client CLIs are launched as subprocesses; observability is the sole
switch between the three attempts.
"""

from __future__ import annotations

import argparse
import copy
import base64
import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import AbstractSet, Any, Callable

import numpy as np
import torch
from flwr.common import FitIns, FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from gaps_flower.flower_message_audit import (
    audit_fit_ins,
    audit_fit_res,
    canonical_fit_ins_bytes,
    canonical_fit_res_bytes,
)


VOLATILE_JSON_PATHS = {
    ("run_config", "args", "observer_context"),
    ("run_config", "args", "observer_events"),
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
    ("provenance", "wall_time_utc"),
    ("provenance", "pid"),
    ("provenance", "path"),
}

_GROUPS = ("B2", "B5")
_MODES = ("off_a", "on", "off_b")
_TIMING_PATHS = {
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
}
_OBSERVER_PATHS = {
    ("run_config", "args", "observer_context"),
    ("run_config", "args", "observer_events"),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes  # type: ignore[attr-defined]
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _require_no_link_ancestors(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if os.path.lexists(ancestor) and _is_reparse_or_link(ancestor):
            raise ValueError(f"symlink/reparse path component is forbidden: {ancestor}")


def _require_regular_file(path: Path) -> Path:
    _require_no_link_ancestors(path.parent)
    if _is_reparse_or_link(path):
        raise ValueError(f"symlink/reparse input is forbidden: {path}")
    if not path.is_file():
        raise ValueError(f"required regular file is missing: {path}")
    return path


def _portable_evidence_path(root: Path, path: Path) -> str:
    """Return a report-stable POSIX path and reject evidence outside its root."""

    evidence_root = Path(root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = evidence_root / candidate
    candidate = candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(
            f"evidence path is outside root or escapes it: {candidate}"
        ) from exc
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError(f"evidence path escapes root: {candidate}")
    return relative.as_posix()


def _finite_json(value: Any, path: tuple[Any, ...] = ()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, path + (index,))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _walk_tensors(
    value: Any,
    *,
    prefix: tuple[str, ...] = (),
    found: list[tuple[str, torch.Tensor]] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if found is None:
        found = []
    if isinstance(value, torch.Tensor):
        found.append((".".join(prefix), value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _walk_tensors(item, prefix=prefix + (str(key),), found=found)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_tensors(item, prefix=prefix + (str(index),), found=found)
    return found


def tensor_fingerprint(checkpoint: Path) -> dict[str, Any]:
    """Fingerprint every checkpoint tensor in insertion order and raw bytes."""

    path = _require_regular_file(Path(checkpoint))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    walked = _walk_tensors(payload)
    if not walked:
        raise ValueError(f"checkpoint contains no tensors: {path}")

    records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    comparison_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for key, tensor in walked:
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"checkpoint contains non-finite tensor: {key}")
        raw = _tensor_bytes(tensor)
        record = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "raw_bytes": len(raw),
            "raw_sha256": _sha256_bytes(raw),
            "_raw": raw,
        }
        records[key] = record
        comparison_records[key] = {
            name: item for name, item in record.items() if name != "_raw"
        }
    comparison = {
        "kind": "tensor_checkpoint",
        "key_order": list(records),
        "tensors": comparison_records,
    }
    return {
        "kind": "tensor_checkpoint",
        "artifact_sha256": _sha256_file(path),
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
        "key_order": list(records),
        "tensors": records,
        "comparison": comparison,
    }


def _scalar_record(value: Any) -> dict[str, Any]:
    """Preserve the exact Flower Scalar type and value in a JSON-safe record."""

    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Flower scalar is non-finite")
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "value_base64": base64.b64encode(value).decode("ascii"),
        }
    raise TypeError(f"unsupported Flower scalar type: {type(value).__name__}")


def _scalar_mapping_record(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key_order": list(values),
        "keys": list(values),
        "types": {str(key): _scalar_record(value)["type"] for key, value in values.items()},
        "values": {str(key): _scalar_record(value) for key, value in values.items()},
    }


def _parameters_record(
    parameters: Parameters, parameter_keys: list[str]
) -> dict[str, Any]:
    arrays = parameters_to_ndarrays(parameters)
    if len(arrays) != len(parameter_keys):
        raise ValueError(
            "Flower parameter count differs from the frozen parameter key order: "
            f"{len(arrays)} != {len(parameter_keys)}"
        )
    tensors: list[dict[str, Any]] = []
    for index, (key, array) in enumerate(zip(parameter_keys, arrays)):
        contiguous = np.ascontiguousarray(array)
        if np.issubdtype(contiguous.dtype, np.inexact) and not bool(
            np.isfinite(contiguous).all()
        ):
            raise ValueError(f"Flower parameter contains non-finite values: {key}")
        raw = contiguous.tobytes(order="C")
        tensors.append(
            {
                "index": index,
                "key": str(key),
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "numel": int(contiguous.size),
                "raw_bytes": len(raw),
                "raw_sha256": _sha256_bytes(raw),
            }
        )
    comparison = {"key_order": list(parameter_keys), "tensors": tensors}
    return {
        **comparison,
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
    }


def _normalize_fit_ins_message(ins: FitIns) -> FitIns:
    # FitIns currently has no timing allowlist.  Copying is deliberate: the
    # gate must never retain or alter a live Flower object.
    return copy.deepcopy(ins)


def _normalize_fit_res_message(res: FitRes) -> FitRes:
    normalized = copy.deepcopy(res)
    metrics = normalized.metrics if normalized.metrics is not None else {}
    for key in ("fit_seconds", "evaluate_seconds"):
        if key not in metrics:
            continue
        value = metrics[key]
        if type(value) is float:
            metrics[key] = 0.0
        elif type(value) is int:
            metrics[key] = 0
        else:
            raise ValueError(f"timing scalar has invalid type: {key}")
    return normalized


def _normalized_timing_scalar_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(values))
    for key in ("fit_seconds", "evaluate_seconds"):
        if key not in normalized:
            continue
        value = normalized[key]
        if type(value) is float:
            normalized[key] = 0.0
        elif type(value) is int:
            normalized[key] = 0
        else:
            raise ValueError(f"timing scalar has invalid type: {key}")
    return _scalar_mapping_record(normalized)


def _fit_ins_trace(ins: FitIns, parameter_keys: list[str]) -> dict[str, Any]:
    snapshot = copy.deepcopy(ins)
    normalized = _normalize_fit_ins_message(snapshot)
    raw_application = canonical_fit_ins_bytes(snapshot)
    normalized_application = canonical_fit_ins_bytes(normalized)
    audit = audit_fit_ins(snapshot)
    raw_sha256 = _sha256_bytes(raw_application)
    if (
        audit.application_message_bytes != len(raw_application)
        or audit.application_message_sha256 != raw_sha256
    ):
        raise ValueError("FitIns independent audit differs from canonical bytes")
    config = _scalar_mapping_record(snapshot.config)
    comparison = {
        "parameters": _parameters_record(snapshot.parameters, parameter_keys),
        "config": config,
        "logical": audit.logical,
        "normalized_application_message_bytes": len(normalized_application),
        "normalized_application_message_sha256": _sha256_bytes(normalized_application),
    }
    return {
        "raw_application_message_bytes": len(raw_application),
        "raw_application_message_sha256": raw_sha256,
        "logical": audit.logical,
        "config_keys": config["keys"],
        "config_types": config["types"],
        "normalized_application_message_bytes": len(normalized_application),
        "normalized_application_message_sha256": _sha256_bytes(normalized_application),
        "comparison": comparison,
    }


def _fit_res_trace(res: FitRes, parameter_keys: list[str]) -> dict[str, Any]:
    snapshot = copy.deepcopy(res)
    normalized = _normalize_fit_res_message(snapshot)
    raw_application = canonical_fit_res_bytes(snapshot)
    normalized_application = canonical_fit_res_bytes(normalized)
    audit = audit_fit_res(snapshot)
    raw_sha256 = _sha256_bytes(raw_application)
    if (
        audit.application_message_bytes != len(raw_application)
        or audit.application_message_sha256 != raw_sha256
    ):
        raise ValueError("FitRes independent audit differs from canonical bytes")
    metrics = _scalar_mapping_record(snapshot.metrics or {})
    normalized_metrics = _scalar_mapping_record(normalized.metrics or {})
    status = {
        "code": int(snapshot.status.code.value),
        "message": snapshot.status.message,
    }
    comparison = {
        "status": status,
        "num_examples": int(snapshot.num_examples),
        "parameters": _parameters_record(snapshot.parameters, parameter_keys),
        "metrics": normalized_metrics,
        "logical": audit.logical,
        "normalized_application_message_bytes": len(normalized_application),
        "normalized_application_message_sha256": _sha256_bytes(normalized_application),
    }
    return {
        "raw_application_message_bytes": len(raw_application),
        "raw_application_message_sha256": raw_sha256,
        "logical": audit.logical,
        "metrics_keys": metrics["keys"],
        "metrics_types": metrics["types"],
        "normalized_application_message_bytes": len(normalized_application),
        "normalized_application_message_sha256": _sha256_bytes(normalized_application),
        "comparison": comparison,
    }


def _flower_trace_fingerprint(
    *, fit_ins: FitIns, fit_res: FitRes, parameter_keys: list[str]
) -> dict[str, Any]:
    """Fingerprint actual FitIns/FitRes objects without using the observer."""

    ins_trace = _fit_ins_trace(fit_ins, parameter_keys)
    res_trace = _fit_res_trace(fit_res, parameter_keys)
    comparison = {
        "fit_ins": ins_trace["comparison"],
        "fit_res": res_trace["comparison"],
    }
    return {
        "kind": "flower_common_trace",
        "artifact_sha256": _sha256_bytes(
            _canonical_bytes(
                {
                    "fit_ins_raw": ins_trace["raw_application_message_sha256"],
                    "fit_res_raw": res_trace["raw_application_message_sha256"],
                }
            )
        ),
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
        "fit_ins": ins_trace,
        "fit_res": res_trace,
        "comparison": comparison,
    }


def _normalize_json(
    value: Any,
    volatile_paths: AbstractSet[tuple],
    path: tuple[Any, ...] = (),
) -> Any:
    if path in volatile_paths:
        if path in _TIMING_PATHS:
            if type(value) is float:
                return 0.0
            if type(value) is int:
                return 0
            raise ValueError(f"timing scalar has invalid type at {path!r}")
        return {"__ignored_exact_volatile_leaf__": ".".join(map(str, path))}
    if isinstance(value, Mapping):
        return {
            key: _normalize_json(item, volatile_paths, path + (key,))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_json(item, volatile_paths, path + (index,))
            for index, item in enumerate(value)
        ]
    return value


def json_fingerprint(
    path: Path, volatile_paths: AbstractSet[tuple]
) -> dict[str, Any]:
    """Fingerprint JSON after normalizing only exact allowlisted leaf paths."""

    input_path = _require_regular_file(Path(path))
    raw = input_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except Exception as exc:
        raise ValueError(f"invalid JSON input {input_path}: {exc}") from exc
    _finite_json(value)
    normalized = _normalize_json(value, volatile_paths)
    content = _canonical_bytes(normalized)
    return {
        "kind": "json",
        "artifact_sha256": _sha256_bytes(raw),
        "content_sha256": _sha256_bytes(content),
        "comparison": normalized,
    }


def _comparison_view(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "comparison" in value and (
            "content_sha256" in value or "artifact_sha256" in value
        ):
            return value["comparison"]
        return {key: _comparison_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_comparison_view(item) for item in value]
    return value


def _first_mismatches(
    left: Any, right: Any, path: tuple[Any, ...] = (), limit: int = 50
) -> list[str]:
    if type(left) is not type(right):
        return [f"{'.'.join(map(str, path)) or '<root>'}: type differs"]
    if isinstance(left, Mapping):
        if set(left) != set(right):
            left_only = sorted(set(left) - set(right), key=str)
            right_only = sorted(set(right) - set(left), key=str)
            return [
                f"{'.'.join(map(str, path)) or '<root>'}: keys differ; "
                f"left_only={left_only!r}; right_only={right_only!r}"
            ]
        output: list[str] = []
        for key in sorted(left, key=str):
            output.extend(_first_mismatches(left[key], right[key], path + (key,), limit))
            if len(output) >= limit:
                break
        return output[:limit]
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{'.'.join(map(str, path)) or '<root>'}: list length differs"]
        output = []
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            output.extend(_first_mismatches(l_item, r_item, path + (index,), limit))
            if len(output) >= limit:
                break
        return output[:limit]
    if left != right:
        return [f"{'.'.join(map(str, path)) or '<root>'}: value differs"]
    return []


def _artifact_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, Mapping) and isinstance(item.get("artifact_sha256"), str):
            output[str(key)] = str(item["artifact_sha256"])
    return output


def _dtype_to_numpy(dtype: str) -> np.dtype[Any] | None:
    mapping = {
        "torch.float16": np.dtype("float16"),
        "torch.float32": np.dtype("float32"),
        "torch.float64": np.dtype("float64"),
        "torch.int8": np.dtype("int8"),
        "torch.uint8": np.dtype("uint8"),
        "torch.int16": np.dtype("int16"),
        "torch.int32": np.dtype("int32"),
        "torch.int64": np.dtype("int64"),
        "torch.bool": np.dtype("bool"),
    }
    return mapping.get(dtype)


def _max_tensor_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    maximum = 0.0
    for name in sorted(set(left) & set(right)):
        l_item, r_item = left[name], right[name]
        if not isinstance(l_item, Mapping) or not isinstance(r_item, Mapping):
            continue
        l_tensors, r_tensors = l_item.get("tensors"), r_item.get("tensors")
        if not isinstance(l_tensors, Mapping) or not isinstance(r_tensors, Mapping):
            continue
        for key in set(l_tensors) & set(r_tensors):
            l_record, r_record = l_tensors[key], r_tensors[key]
            if not isinstance(l_record, Mapping) or not isinstance(r_record, Mapping):
                continue
            if l_record.get("raw_sha256") == r_record.get("raw_sha256"):
                continue
            dtype = _dtype_to_numpy(str(l_record.get("dtype")))
            if dtype is None or dtype != _dtype_to_numpy(str(r_record.get("dtype"))):
                maximum = max(maximum, 1.0)
                continue
            l_raw, r_raw = l_record.get("_raw"), r_record.get("_raw")
            if not isinstance(l_raw, bytes) or not isinstance(r_raw, bytes):
                maximum = max(maximum, 1.0)
                continue
            l_array = np.frombuffer(l_raw, dtype=dtype)
            r_array = np.frombuffer(r_raw, dtype=dtype)
            if l_array.shape != r_array.shape:
                maximum = max(maximum, 1.0)
                continue
            if np.issubdtype(dtype, np.number):
                delta = np.max(np.abs(l_array.astype(np.float64) - r_array.astype(np.float64)), initial=0.0)
                maximum = max(maximum, float(delta))
            else:
                maximum = max(maximum, 1.0)
    return maximum


def compare_fingerprints(
    off_a: Mapping[str, Any],
    on: Mapping[str, Any],
    off_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify OFF drift before checking whether ON mutated the numerical path."""

    off_a_view = _comparison_view(off_a)
    on_view = _comparison_view(on)
    off_b_view = _comparison_view(off_b)
    off_mismatches = _first_mismatches(off_a_view, off_b_view)
    on_mismatches = _first_mismatches(off_a_view, on_view)
    if off_mismatches:
        status = "environment_nondeterminism"
        selected = off_mismatches
        max_abs_delta = _max_tensor_delta(off_a, off_b)
    elif on_mismatches:
        status = "observer_path_mutation"
        selected = on_mismatches
        max_abs_delta = _max_tensor_delta(off_a, on)
    else:
        status = "equivalent"
        selected = []
        max_abs_delta = 0.0
    result = {
        "status": status,
        "equivalent": status == "equivalent",
        "off_pair_equal": not off_mismatches,
        "on_equal_to_off": not on_mismatches,
        "max_abs_delta": float(max_abs_delta),
        "mismatches": selected,
        "artifact_hashes": {
            "off_a": _artifact_hashes(off_a),
            "on": _artifact_hashes(on),
            "off_b": _artifact_hashes(off_b),
        },
        "content_set_sha256": {
            "off_a": _sha256_bytes(_canonical_bytes(off_a_view)),
            "on": _sha256_bytes(_canonical_bytes(on_view)),
            "off_b": _sha256_bytes(_canonical_bytes(off_b_view)),
        },
    }
    return result


def _create_frozen_initial_checkpoint(path: Path, group_id: str) -> dict[str, Any]:
    """Create one seed-42 model checkpoint reused by every gate mode."""

    group = str(group_id).upper()
    if group not in _GROUPS:
        raise ValueError(f"group_id must be one of {_GROUPS}, got {group_id!r}")
    target = Path(path)
    _require_no_link_ancestors(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(f"initial checkpoint already exists: {target}")
    from gaps_flower.task import create_model, make_config

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        config = make_config(
            device="cpu",
            local_epochs=1,
            batch_size=4,
            profile="proto_replay",
            seed=42,
        )
        model = create_model(config)
        state = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )
    torch.save(
        {
            "schema_version": "iotj.observer_equivalence.initial.v1",
            "group_id": group,
            "training_seed": 42,
            "model_state": state,
            "parameter_keys": list(state),
        },
        target,
    )
    fingerprint = tensor_fingerprint(target)
    return {
        "path": str(target),
        "raw_sha256": fingerprint["artifact_sha256"],
        "tensor_content_sha256": fingerprint["content_sha256"],
        "parameter_keys": list(state),
        "training_seed": 42,
        "loaded_by_modes": list(_MODES),
    }


def _write_fixture(root: Path) -> dict[str, str]:
    data_root = root / "fixture_data"
    data_root.mkdir()
    rng = np.random.RandomState(7001)
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    phases = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    regression = np.zeros((8, 4), dtype=np.float32)
    for client_id in (1, 2, 5):
        client_dir = data_root / f"client_{client_id}"
        client_dir.mkdir()
        for split in ("train", "test", "calibration"):
            features = rng.standard_normal((8, 100, 8)).astype(np.float32)
            np.save(client_dir / f"{split}_features.npy", features, allow_pickle=False)
            np.save(client_dir / f"{split}_classification_labels.npy", labels, allow_pickle=False)
            np.save(client_dir / f"{split}_phase_labels.npy", phases, allow_pickle=False)
            np.save(client_dir / f"{split}_regression_labels.npy", regression, allow_pickle=False)
    hashes = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(data_root.rglob("*.npy"))
    }
    manifest = root / "fixture_manifest.json"
    manifest.write_bytes(_canonical_bytes({"seed": 7001, "files": hashes}) + b"\n")
    hashes[str(manifest.relative_to(root)).replace("\\", "/")] = _sha256_file(manifest)
    return hashes


def _reserve_port(used: set[int]) -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in used:
            used.add(port)
            return port
    raise RuntimeError("unable to reserve a unique loopback port")


def _observer_context(
    *, group_id: str, producer: str, client_id: str | None, fixture_sha: str
) -> dict[str, Any]:
    group_lower = group_id.lower()
    return {
        "run_id": f"c12_to_c5__{group_lower}__s42",
        "attempt_id": f"c12_to_c5__{group_lower}__s42__a001",
        "group_id": group_id,
        "training_seed": 42,
        "client_id": client_id,
        "host_id": "local-loopback",
        "producer": producer,
        "confirmation_commit": "0" * 40,
        "source_archive_sha256": fixture_sha,
        "dataset_manifest_sha256": fixture_sha,
        "algorithm_config_sha256": _sha256_bytes(
            _canonical_bytes({"group": group_id, "rounds": 2, "seed": 42})
        ),
    }


def _write_context(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _server_command(
    *, group_id: str, port: int, data_root: Path, output_dir: Path
) -> list[str]:
    b5 = group_id == "B5"
    source = f"{data_root / 'client_1'},{data_root / 'client_2'}"
    target = str(data_root / "client_5")
    return [
        sys.executable,
        "-m",
        "gaps_flower.server_app",
        "--server-address",
        f"127.0.0.1:{port}",
        "--rounds",
        "2",
        "--min-clients",
        "2",
        "--output-dir",
        str(output_dir),
        "--run-name",
        f"observer_equivalence_{group_id.lower()}",
        "--seed",
        "42",
        "--strategy",
        "gaps",
        "--profile",
        "proto_replay",
        "--save-history",
        "true",
        "--use-selective-agg",
        "true",
        "--use-proto-mmd",
        "false",
        "--da-preset",
        "none",
        "--use-domain-adapt",
        "true",
        "--server-val-data",
        source,
        "--server-calib-data",
        target,
        "--domain-adapt-steps",
        "1",
        "--domain-adapt-warmup",
        "0",
        "--da-use-coral",
        _bool(b5),
        "--da-use-mmd",
        "true",
        "--da-use-adversarial",
        _bool(b5),
        "--da-mmd-objective",
        "mmd2",
        "--da-stage-alignment",
        "cross_domain_same_class_phase",
        "--da-adv-feature-objective",
        "wasserstein_min",
        "--da-coral-class-conditional",
        "true",
        "--strict-calibration-split",
        "true",
        "--da-device",
        "cpu",
        "--use-adapted-as-global",
        "true",
        "--da-lambda-coral",
        "0.5" if b5 else "0.0",
        "--da-lambda-global-mmd",
        "0.5",
        "--da-lambda-class-mmd",
        "0.5",
        "--da-lambda-proto-anchor",
        "0.3",
        "--da-lambda-adv",
        "0.5" if b5 else "0.0",
        "--da-lambda-target-ce",
        "0.0",
        "--da-lambda-proto",
        "0.05",
        "--da-lambda-consistency",
        "2.0",
        "--da-lambda-residual",
        "0.1",
        "--da-lambda-proto-mmd",
        "0.0",
        "--da-lambda-stage-mmd",
        "0.2" if b5 else "0.0",
        "--da-target-ce-label-smoothing",
        "0.0",
        "--da-target-ce-class-balanced",
        "false",
        "--da-server-opt-lr",
        "0.0005",
    ]


def _client_command(*, client_id: int, port: int, data_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gaps_flower.client_app",
        "--server-address",
        f"127.0.0.1:{port}",
        "--client-id",
        str(client_id),
        "--data-root",
        str(data_root),
        "--device",
        "cpu",
        "--local-epochs",
        "1",
        "--batch-size",
        "4",
        "--profile",
        "proto_replay",
        "--seed",
        "42",
    ]


def _trace_json_projection(value: Any) -> dict[str, Any]:
    _finite_json(value)
    return {
        "content_sha256": _sha256_bytes(_canonical_bytes(value)),
        "comparison": value,
    }


def _fixed_adapted_logits(checkpoint: Path, group_id: str) -> dict[str, Any]:
    from gaps_flower.task import create_model, make_config

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"adapted checkpoint has no model_state: {checkpoint}")
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7001)
            model = create_model(
                make_config(
                    device="cpu",
                    local_epochs=1,
                    batch_size=4,
                    profile="proto_replay",
                    seed=42,
                )
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            inputs = torch.from_numpy(
                np.random.RandomState(7001)
                .standard_normal((8, 100, 8))
                .astype(np.float32)
            )
            with torch.no_grad():
                output = model(inputs)
                logits = output[0] if isinstance(output, (tuple, list)) else output
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"adapted logits are non-finite for {group_id}")
    raw = _tensor_bytes(logits)
    comparison = {
        "input_seed": 7001,
        "group_id": group_id,
        "dtype": str(logits.dtype),
        "shape": list(logits.shape),
        "raw_sha256": _sha256_bytes(raw),
    }
    return {
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
        "comparison": comparison,
    }


def _append_common_trace(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()


def _run_traced_server(
    trace_output: Path,
    initial_checkpoint: Path,
    delegated_argv: list[str],
) -> int:
    """Run the real server while independently snapshotting live Flower objects."""

    trace_path = Path(trace_output)
    initial_path = _require_regular_file(Path(initial_checkpoint))
    _require_no_link_ancestors(trace_path.parent)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("xb"):
        pass
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    initial_state = initial.get("model_state")
    initial_keys = initial.get("parameter_keys")
    if not isinstance(initial_state, Mapping) or list(initial_state) != initial_keys:
        raise ValueError("frozen initial checkpoint state/key binding is invalid")
    initial_fp = tensor_fingerprint(initial_path)

    import gaps_flower.server_app as server_app
    import gaps_flower.strategy as strategy_module
    from gaps_flower.strategy import GapsStrategy

    original_create_model = server_app.create_model
    original_configure_fit = GapsStrategy.configure_fit
    original_aggregate_fit = GapsStrategy.aggregate_fit
    original_strategy_audit_fit_res = strategy_module.audit_fit_res

    if os.environ.get("IOTJ_GATE_INJECT_POST_AUDIT_METRICS_KEY") == "1":
        def injected_audit_fit_res(fit_res: FitRes) -> Any:
            audit = original_strategy_audit_fit_res(fit_res)
            if fit_res.metrics is None:
                fit_res.metrics = {}
            fit_res.metrics["observer_injected_post_audit_metrics_key"] = "sentinel"
            return audit

        strategy_module.audit_fit_res = injected_audit_fit_res

    def create_model_from_frozen(config: Any) -> torch.nn.Module:
        model = original_create_model(config)
        if list(model.state_dict()) != list(initial_keys):
            raise ValueError("frozen initial checkpoint schema differs from server model")
        model.load_state_dict(initial_state, strict=True)
        return model

    def traced_configure_fit(
        strategy: Any, server_round: int, parameters: Parameters, client_manager: Any
    ) -> Any:
        configured = original_configure_fit(
            strategy, server_round, parameters, client_manager
        )
        if os.environ.get("IOTJ_GATE_INJECT_CONFIG_KEY") == "1":
            for _proxy, fit_ins in configured:
                fit_ins.config["observer_injected_regression_key"] = True
        for proxy, fit_ins in configured:
            _append_common_trace(
                trace_path,
                {
                    "record_type": "fit_ins",
                    "round": int(server_round),
                    "proxy_id": str(proxy.cid),
                    "trace": _fit_ins_trace(fit_ins, list(strategy.parameter_keys)),
                },
            )
        return configured

    def traced_aggregate_fit(
        strategy: Any,
        server_round: int,
        results: list[Any],
        failures: list[Any],
    ) -> Any:
        fit_res_rows: dict[str, Any] = {}
        selector_inputs: dict[str, Any] = {}
        for proxy, fit_res in results:
            raw_client_id = (fit_res.metrics or {}).get("client_id")
            client_id = (
                f"C{int(raw_client_id)}"
                if raw_client_id is not None
                else str(proxy.cid)
            )
            trace = _fit_res_trace(fit_res, list(strategy.parameter_keys))
            fit_res_rows[client_id] = trace
            selector_inputs[client_id] = trace["comparison"]["metrics"]
            _append_common_trace(
                trace_path,
                {
                    "record_type": "fit_res",
                    "round": int(server_round),
                    "proxy_id": str(proxy.cid),
                    "client_id": client_id,
                    "trace": trace,
                },
            )
        returned_parameters, aggregated_metrics = original_aggregate_fit(
            strategy, server_round, results, failures
        )
        for proxy, fit_res in results:
            raw_client_id = (fit_res.metrics or {}).get("client_id")
            client_id = (
                f"C{int(raw_client_id)}"
                if raw_client_id is not None
                else str(proxy.cid)
            )
            _append_common_trace(
                trace_path,
                {
                    "record_type": "fit_res_post_observer",
                    "round": int(server_round),
                    "proxy_id": str(proxy.cid),
                    "client_id": client_id,
                    "trace": _fit_res_trace(fit_res, list(strategy.parameter_keys)),
                },
            )
        if returned_parameters is None:
            raise ValueError("formal trace expected returned aggregate parameters")
        output_dir = Path(strategy.output_dir)
        plain_path = output_dir / f"server_round_{int(server_round):03d}.pth"
        adapted_path = output_dir / f"server_round_{int(server_round):03d}_adapted.pth"
        event = copy.deepcopy(strategy._round_event(int(server_round)))
        round_comparison = {
            "round": int(server_round),
            "fit_res_by_client": {
                key: fit_res_rows[key]["comparison"] for key in sorted(fit_res_rows)
            },
            "plain_aggregate": tensor_fingerprint(plain_path)["comparison"],
            "returned_parameters": _parameters_record(
                returned_parameters, list(strategy.parameter_keys)
            ),
            "selector_inputs": selector_inputs,
            "selector_decision": event.get("selective_agg"),
            "aggregated_metrics": _normalized_timing_scalar_mapping(
                aggregated_metrics or {}
            ),
            "prototype_stats": json_fingerprint(
                output_dir / f"prototype_stats_round_{int(server_round):03d}.json",
                VOLATILE_JSON_PATHS,
            )["comparison"],
            "semantic_protos": json_fingerprint(
                output_dir / f"semantic_protos_round_{int(server_round):03d}.json",
                VOLATILE_JSON_PATHS,
            )["comparison"],
        }
        client_stats = json.loads(
            (output_dir / f"client_stats_round_{int(server_round):03d}.json").read_text(
                encoding="utf-8"
            )
        )
        round_comparison["client_stats"] = {
            "round": client_stats.get("round"),
            "global_summary": client_stats.get("global_summary"),
            "clients": [
                _normalize_json({"metrics": row}, VOLATILE_JSON_PATHS)["metrics"]
                for row in client_stats.get("clients", [])
            ],
        }
        if adapted_path.is_file():
            round_comparison["adapted_checkpoint"] = tensor_fingerprint(adapted_path)[
                "comparison"
            ]
            da_path = output_dir / f"domain_adapt_round_{int(server_round):03d}.json"
            da_value = json.loads(da_path.read_text(encoding="utf-8"))
            da_value.pop("semantic_protos_after_da", None)
            round_comparison["domain_adapt_diagnostics"] = _normalize_json(
                {"metrics": da_value}, VOLATILE_JSON_PATHS
            )["metrics"]
        _append_common_trace(
            trace_path,
            {
                "record_type": "round_summary",
                "round": int(server_round),
                "content_sha256": _sha256_bytes(_canonical_bytes(round_comparison)),
                "comparison": round_comparison,
            },
        )
        return returned_parameters, aggregated_metrics

    server_app.create_model = create_model_from_frozen
    GapsStrategy.configure_fit = traced_configure_fit
    GapsStrategy.aggregate_fit = traced_aggregate_fit
    _append_common_trace(
        trace_path,
        {
            "record_type": "initial_checkpoint",
            "raw_sha256": initial_fp["artifact_sha256"],
            "tensor_content_sha256": initial_fp["content_sha256"],
            "parameter_keys": list(initial_keys),
            "comparison": initial_fp["comparison"],
        },
    )
    if delegated_argv[:2] != ["-m", "gaps_flower.server_app"]:
        raise ValueError("trace child must delegate to gaps_flower.server_app")
    previous_argv = sys.argv
    try:
        sys.argv = ["gaps_flower.server_app", *delegated_argv[2:]]
        server_app.main()
    finally:
        sys.argv = previous_argv
        server_app.create_model = original_create_model
        GapsStrategy.configure_fit = original_configure_fit
        GapsStrategy.aggregate_fit = original_aggregate_fit
        strategy_module.audit_fit_res = original_strategy_audit_fit_res
    return 0


def _popen_flags() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()


def _tail(path: Path, limit: int = 8000) -> str:
    if not path.is_file():
        return "<missing>"
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _wait_server(port: int, process: subprocess.Popen[Any], timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                probe.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.1)
    raise TimeoutError("server readiness timeout")


def _launch_attempt(
    *,
    root: Path,
    fixture_root: Path,
    group_id: str,
    mode: str,
    port: int,
    fixture_sha: str,
    initial_checkpoint: Path,
) -> dict[str, Any]:
    attempt = root / mode
    attempt.mkdir()
    # Keep the runtime output path identical across OFF-A/ON/OFF-B.  The
    # adapted checkpoint contains a diagnostic path, so mode-specific runtime
    # paths would make the raw checkpoint bytes differ despite identical
    # tensors.  Each active directory is still freshly and exclusively made,
    # then moved into its immutable attempt evidence directory after exit.
    active_server_output = root / "_active_server_output"
    if os.path.lexists(active_server_output):
        raise FileExistsError(f"stale active server output: {active_server_output}")
    active_server_output.mkdir()
    server_output = attempt / "server_output"
    commands = attempt / "commands.json"
    server_command = _server_command(
        group_id=group_id,
        port=port,
        data_root=fixture_root,
        output_dir=active_server_output,
    )
    common_trace = attempt / "common_trace.jsonl"
    server_command = [
        server_command[0],
        "-m",
        "scripts.run_iotj_observer_equivalence_gate",
        "--trace-child-role",
        "server",
        "--trace-output",
        str(common_trace),
        "--trace-initial-checkpoint",
        str(initial_checkpoint),
        "--",
        *server_command[1:],
    ]
    client_commands = [
        _client_command(client_id=client_id, port=port, data_root=fixture_root)
        for client_id in (1, 2)
    ]
    contexts: list[tuple[Path, Path]] = []
    if mode == "on":
        server_context = attempt / "server_context.json"
        server_events = attempt / "server_events.jsonl"
        _write_context(
            server_context,
            _observer_context(
                group_id=group_id,
                producer="server",
                client_id=None,
                fixture_sha=fixture_sha,
            ),
        )
        server_command.extend(
            ["--observer-context", str(server_context), "--observer-events", str(server_events)]
        )
        contexts.append((server_context, server_events))
        for index, client_id in enumerate((1, 2)):
            context = attempt / f"client_c{client_id}_context.json"
            events = attempt / f"client_c{client_id}_events.jsonl"
            _write_context(
                context,
                _observer_context(
                    group_id=group_id,
                    producer="client",
                    client_id=f"C{client_id}",
                    fixture_sha=fixture_sha,
                ),
            )
            client_commands[index].extend(
                ["--observer-context", str(context), "--observer-events", str(events)]
            )
            contexts.append((context, events))
    commands.write_bytes(
        _canonical_bytes(
            {
                "server": server_command,
                "clients": client_commands,
                "environment": {
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            }
        )
        + b"\n"
    )

    env = os.environ.copy()
    env.update(
        {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    if mode == "on" and os.environ.get("IOTJ_GATE_TEST_INJECT_ON_KEY") == "1":
        env["IOTJ_GATE_INJECT_CONFIG_KEY"] = "1"
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    repo_root = Path(__file__).resolve().parents[1]
    try:
        server_stdout = (attempt / "server.stdout.log").open("xb")
        server_stderr = (attempt / "server.stderr.log").open("xb")
        handles.extend((server_stdout, server_stderr))
        server = subprocess.Popen(
            server_command,
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=server_stdout,
            stderr=server_stderr,
            **_popen_flags(),
        )
        processes.append(server)
        _wait_server(port, server)
        clients: list[subprocess.Popen[Any]] = []
        for client_id, command in zip((1, 2), client_commands):
            stdout = (attempt / f"client_c{client_id}.stdout.log").open("xb")
            stderr = (attempt / f"client_c{client_id}.stderr.log").open("xb")
            handles.extend((stdout, stderr))
            client = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                **_popen_flags(),
            )
            clients.append(client)
            processes.append(client)
        for client_id, client in zip((1, 2), clients):
            try:
                return_code = client.wait(timeout=240)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"client C{client_id} timeout") from exc
            if return_code != 0:
                raise RuntimeError(f"client C{client_id} exit {return_code}")
        try:
            server_code = server.wait(timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("server timeout") from exc
        if server_code != 0:
            raise RuntimeError(f"server exit {server_code}")
        active_server_output.rename(server_output)
    except Exception as exc:
        for handle in handles:
            handle.flush()
        detail = {
            "server_stdout": _tail(attempt / "server.stdout.log"),
            "server_stderr": _tail(attempt / "server.stderr.log"),
            "client_c1_stderr": _tail(attempt / "client_c1.stderr.log"),
            "client_c2_stderr": _tail(attempt / "client_c2.stderr.log"),
        }
        raise RuntimeError(f"{mode} loopback failed: {exc}; logs={detail}") from exc
    finally:
        for process in reversed(processes):
            _stop_process(process)
        for handle in handles:
            handle.close()

    local_identity = _observer_context(
        group_id=group_id,
        producer="server",
        client_id=None,
        fixture_sha=fixture_sha,
    )
    local_binding = _expected_training_binding(
        run_id=local_identity["run_id"],
        attempt_id=local_identity["attempt_id"],
        group_id=local_identity["group_id"],
        training_seed=local_identity["training_seed"],
        confirmation_commit=local_identity["confirmation_commit"],
        source_archive_sha256=local_identity["source_archive_sha256"],
        dataset_manifest_sha256=local_identity["dataset_manifest_sha256"],
        algorithm_config_sha256=local_identity["algorithm_config_sha256"],
        server_host_id="local-loopback",
        c1_host_id="local-loopback",
        c2_host_id="local-loopback",
    )
    sidecars = _validate_observer_sidecars(
        attempt,
        enabled=mode == "on",
        expected_binding=local_binding,
    )
    return {
        "attempt": attempt,
        "server_output": server_output,
        "port": port,
        "server_address": f"127.0.0.1:{port}",
        "group_id": group_id,
        "common_trace": common_trace,
        "sidecars": sidecars,
    }


def _json_projection(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    projection_path = path.with_suffix(path.suffix + ".gate.json")
    projection_path.write_bytes(_canonical_bytes(payload) + b"\n")
    return json_fingerprint(projection_path, VOLATILE_JSON_PATHS)


def _run_config_argument_types(args: Mapping[str, Any]) -> dict[str, str]:
    types = {str(key): type(item).__name__ for key, item in args.items()}
    for key in ("observer_context", "observer_events"):
        if key in types:
            types[key] = "__ignored_exact_volatile_leaf_type__"
    return types


def _run_config_fingerprint(path: Path, attempt: Mapping[str, Any]) -> dict[str, Any]:
    config_path = _require_regular_file(path)
    raw = config_path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    _finite_json(value)
    args = value.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run_config args must be an object")
    if args.get("server_address") != attempt["server_address"]:
        raise ValueError("run_config server_address differs from reserved binding")
    expected_active_output = Path(attempt["attempt"]).parent / "_active_server_output"
    if args.get("output_dir") != str(expected_active_output):
        raise ValueError("run_config output_dir differs from the exact gate binding")
    normalized_args = _normalize_json(
        {"run_config": {"args": dict(args)}}, VOLATILE_JSON_PATHS
    )["run_config"]["args"]
    normalized_args["server_address"] = {
        "type": "str",
        "host": "127.0.0.1",
        "port_binding": "unique_dynamic_port",
    }
    argument_types = _run_config_argument_types(args)
    comparison = {
        "args": normalized_args,
        "args_key_order": list(args),
        "args_types": argument_types,
        "orchestration_contract": {
            "host": "127.0.0.1",
            "unique_dynamic_port": True,
            "actual_port_type": type(attempt["port"]).__name__,
        },
    }
    return {
        "kind": "run_config",
        "artifact_sha256": _sha256_bytes(raw),
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
        "actual_binding": {
            "server_address": args["server_address"],
            "output_dir": args["output_dir"],
            "run_name": args["run_name"],
            "port": attempt["port"],
        },
        "comparison": comparison,
    }


def _capture_artifacts(attempt: Mapping[str, Any]) -> dict[str, Any]:
    attempt_root = Path(attempt["attempt"])
    output = Path(attempt["server_output"])
    artifacts: OrderedDict[str, Any] = OrderedDict()
    artifacts["final_aggregated_checkpoint"] = tensor_fingerprint(output / "server_latest.pth")
    artifacts["final_adapted_checkpoint"] = tensor_fingerprint(output / "server_latest_adapted.pth")
    for label in ("final_aggregated_checkpoint", "final_adapted_checkpoint"):
        raw_sha = artifacts[label]["artifact_sha256"]
        artifacts[f"{label}_raw"] = {
            "kind": "raw_checkpoint",
            "artifact_sha256": raw_sha,
            "content_sha256": raw_sha,
            "comparison": {"raw_file_sha256": raw_sha},
        }

    artifacts["run_config"] = _run_config_fingerprint(
        output / "run_config.json", attempt
    )
    artifacts["common_trace"] = _common_trace_fingerprint(
        Path(attempt["common_trace"])
    )
    for round_idx in (1, 2):
        artifacts[f"adapted_logits_round_{round_idx}"] = _fixed_adapted_logits(
            output / f"server_round_{round_idx:03d}_adapted.pth",
            str(attempt.get("group_id", "local")),
        )
        for stem in ("prototype_stats", "semantic_protos"):
            source = output / f"{stem}_round_{round_idx:03d}.json"
            artifacts[f"{stem}_round_{round_idx}"] = json_fingerprint(
                source, VOLATILE_JSON_PATHS
            )
        client_stats_path = output / f"client_stats_round_{round_idx:03d}.json"
        client_stats = json.loads(client_stats_path.read_text(encoding="utf-8"))
        for client in client_stats["clients"]:
            client_id = int(client["client_id"])
            artifacts[f"client_stats_round_{round_idx}_c{client_id}"] = _json_projection(
                attempt_root / f"client_stats_round_{round_idx}_c{client_id}.json",
                {"metrics": client},
            )
        diagnostics_path = output / f"domain_adapt_round_{round_idx:03d}.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        semantic_path = diagnostics.pop("semantic_protos_after_da", None)
        artifacts[f"domain_adapt_round_{round_idx}"] = _json_projection(
            attempt_root / f"domain_adapt_round_{round_idx}.json",
            {
                "metrics": diagnostics,
                "provenance": {
                    "wall_time_utc": "not-collected",
                    "pid": 0,
                    "path": semantic_path,
                },
            },
        )
    return artifacts


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(
            line,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        _finite_json(row)
    return rows


def _validate_application_audit(audit: Any, *, direction: str) -> None:
    if direction == "downlink":
        logical_keys = {
            "logical_downlink_model_value_bytes",
            "logical_downlink_parameter_blob_bytes",
            "logical_downlink_semantic_proto_utf8_bytes",
            "logical_downlink_other_config_value_bytes",
            "logical_downlink_total_bytes",
        }
        total_key = "logical_downlink_total_bytes"
        component_keys = logical_keys - {
            total_key,
            "logical_downlink_model_value_bytes",
        }
    elif direction == "uplink":
        logical_keys = {
            "logical_uplink_model_value_bytes",
            "logical_uplink_parameter_blob_bytes",
            "logical_uplink_prototype_utf8_bytes",
            "logical_uplink_prototype_var_utf8_bytes",
            "logical_uplink_statistics_utf8_bytes",
            "logical_uplink_diagnostic_value_bytes",
            "logical_uplink_total_bytes",
        }
        total_key = "logical_uplink_total_bytes"
        component_keys = logical_keys - {
            total_key,
            "logical_uplink_model_value_bytes",
        }
    else:
        raise ValueError(f"unknown application audit direction: {direction}")
    if not isinstance(audit, Mapping) or set(audit) != {
        "logical",
        "application_message_bytes",
        "application_message_sha256",
    }:
        raise ValueError(f"{direction} application audit schema mismatch")
    logical = audit["logical"]
    if not isinstance(logical, Mapping) or set(logical) != logical_keys:
        raise ValueError(f"{direction} application audit logical schema mismatch")
    for key, value in logical.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{direction} application audit has invalid message byte field: {key}")
    if logical[total_key] != sum(logical[key] for key in component_keys):
        raise ValueError(f"{direction} application audit logical total mismatch")
    application_bytes = audit["application_message_bytes"]
    if type(application_bytes) is not int or application_bytes <= 0:
        raise ValueError(f"{direction} application audit has invalid message bytes")
    sha256 = audit["application_message_sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or sha256 != sha256.lower()
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"{direction} application audit SHA-256 is invalid")


_BINDING_COMMON_KEYS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "group_id",
    "training_seed",
    "confirmation_commit",
    "source_archive_sha256",
    "dataset_manifest_sha256",
    "algorithm_config_sha256",
}


def _require_lower_hex_binding(field: str, value: Any, length: int) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"expected binding {field} is not lowercase {length}-hex")


def _validate_binding_common(binding: Mapping[str, Any]) -> None:
    if binding["schema_version"] != "iotj.confirmation.observability.v1":
        raise ValueError("expected binding schema_version is invalid")
    if type(binding["training_seed"]) is not int or binding["training_seed"] not in {
        42,
        43,
        44,
        45,
        46,
    }:
        raise ValueError("expected binding training_seed is invalid")
    group = binding["group_id"]
    if group not in _GROUPS:
        raise ValueError("expected binding group_id is invalid")
    expected_run = f"c12_to_c5__{str(group).lower()}__s{binding['training_seed']}"
    if binding["run_id"] != expected_run:
        raise ValueError("expected binding run_id differs from group/seed")
    attempt_id = binding["attempt_id"]
    if (
        not isinstance(attempt_id, str)
        or not attempt_id.startswith(f"{expected_run}__a")
        or len(attempt_id) != len(expected_run) + 6
        or not attempt_id[-3:].isdigit()
    ):
        raise ValueError("expected binding attempt_id differs from run_id")
    _require_lower_hex_binding(
        "confirmation_commit", binding["confirmation_commit"], 40
    )
    for field in (
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
    ):
        _require_lower_hex_binding(field, binding[field], 64)


def _validate_expected_training_binding(
    expected_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(expected_binding, Mapping) or set(expected_binding) != (
        _BINDING_COMMON_KEYS | {"producers"}
    ):
        raise ValueError("training sidecars require an exact expected binding")
    binding = copy.deepcopy(dict(expected_binding))
    _validate_binding_common(binding)
    producers = binding["producers"]
    if not isinstance(producers, Mapping) or set(producers) != {"server", "C1", "C2"}:
        raise ValueError("training expected binding producer matrix is invalid")
    for logical_name, endpoint in producers.items():
        if not isinstance(endpoint, Mapping) or set(endpoint) != {
            "host_id",
            "producer",
            "client_id",
        }:
            raise ValueError("training expected binding endpoint schema is invalid")
        expected_producer = "server" if logical_name == "server" else "client"
        expected_client = None if logical_name == "server" else logical_name
        if (
            endpoint["producer"] != expected_producer
            or endpoint["client_id"] != expected_client
            or not isinstance(endpoint["host_id"], str)
            or not endpoint["host_id"]
        ):
            raise ValueError("training expected binding endpoint identity is invalid")
    return binding


def _validate_expected_resource_binding(
    expected_binding: Mapping[str, Any] | None,
    client_id: str,
) -> dict[str, Any]:
    expected_keys = _BINDING_COMMON_KEYS | {"client_id", "host_id", "producer"}
    if not isinstance(expected_binding, Mapping) or set(expected_binding) != expected_keys:
        raise ValueError("resource sidecar requires an exact expected binding")
    binding = copy.deepcopy(dict(expected_binding))
    _validate_binding_common(binding)
    if (
        binding["client_id"] != client_id
        or binding["producer"] != "resource_sampler"
        or not isinstance(binding["host_id"], str)
        or not binding["host_id"]
    ):
        raise ValueError("resource expected binding endpoint identity is invalid")
    return binding


def _expected_training_binding(
    *,
    run_id: str,
    attempt_id: str,
    group_id: str,
    training_seed: int,
    confirmation_commit: str,
    source_archive_sha256: str,
    dataset_manifest_sha256: str,
    algorithm_config_sha256: str,
    server_host_id: str,
    c1_host_id: str,
    c2_host_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "iotj.confirmation.observability.v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "group_id": group_id,
        "training_seed": training_seed,
        "confirmation_commit": confirmation_commit,
        "source_archive_sha256": source_archive_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "algorithm_config_sha256": algorithm_config_sha256,
        "producers": {
            "server": {
                "host_id": server_host_id,
                "producer": "server",
                "client_id": None,
            },
            "C1": {"host_id": c1_host_id, "producer": "client", "client_id": "C1"},
            "C2": {"host_id": c2_host_id, "producer": "client", "client_id": "C2"},
        },
    }


def _expected_resource_binding(
    training_binding: Mapping[str, Any], *, client_id: str, host_id: str
) -> dict[str, Any]:
    return {
        key: training_binding[key]
        for key in _BINDING_COMMON_KEYS
    } | {
        "client_id": client_id,
        "host_id": host_id,
        "producer": "resource_sampler",
    }


def _validate_observer_sidecars(
    attempt_dir: Path,
    *,
    enabled: bool,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate actual JSONL/close sidecars and derive counts from disk."""

    root = Path(attempt_dir)
    event_files = sorted(root.rglob("*events.jsonl"))
    close_files = sorted(root.rglob("*events.close.json"))
    if not enabled:
        if event_files or close_files:
            raise ValueError(
                "observer-disabled attempt contains sidecars: "
                f"events={event_files}, close={close_files}"
            )
        return {
            "enabled": False,
            "event_files": 0,
            "close_summaries": 0,
            "producers": {},
        }
    if len(event_files) != 3 or len(close_files) != 3:
        raise ValueError(
            "missing required server/C1/C2 observer sidecars: "
            f"event_files={len(event_files)}, close_summaries={len(close_files)}"
        )
    binding = _validate_expected_training_binding(expected_binding)
    event_files = [_require_regular_file(path) for path in event_files]
    close_files = [_require_regular_file(path) for path in close_files]

    required_event_keys = {
        "schema_version",
        "event_id",
        "event_type",
        "run_id",
        "attempt_id",
        "group_id",
        "training_seed",
        "round",
        "client_id",
        "host_id",
        "producer",
        "process_instance_id",
        "sequence",
        "wall_time_utc",
        "monotonic_ns",
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
        "status",
        "payload",
    }
    required_close_keys = {
        "schema_version",
        "run_id",
        "attempt_id",
        "host_id",
        "producer",
        "process_instance_id",
        "observer_flower_serialize_ns",
        "observer_event_encode_ns",
        "observer_io_write_ns",
        "observer_fsync_ns",
        "observer_total_ns",
        "observer_event_bytes_written",
        "observer_event_count",
        "observer_reporting_tail_bytes",
    }
    producer_rows: dict[str, dict[str, Any]] = {}
    substantive_by_producer: dict[str, list[dict[str, Any]]] = {}
    global_event_ids: set[str] = set()
    common_identity: dict[str, Any] | None = None
    identity_keys = (
        "schema_version",
        "run_id",
        "attempt_id",
        "group_id",
        "training_seed",
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
    )
    for event_path in event_files:
        rows = _read_events(event_path)
        if not rows:
            raise ValueError(f"empty observer event file: {event_path}")
        first = rows[0]
        if common_identity is None:
            common_identity = {key: first[key] for key in identity_keys}
        for index, row in enumerate(rows, start=1):
            if set(row) != required_event_keys:
                raise ValueError(f"observer event schema mismatch: {event_path}:{index}")
            if row["schema_version"] != "iotj.confirmation.observability.v1":
                raise ValueError(f"observer schema version mismatch: {event_path}")
            if type(row["sequence"]) is not int or row["sequence"] != index:
                raise ValueError(f"non-contiguous observer sequence: {event_path}")
            if type(row["monotonic_ns"]) is not int or row["monotonic_ns"] < 0:
                raise ValueError(f"observer monotonic_ns must be a nonnegative integer: {event_path}")
            if row["round"] is not None and (
                type(row["round"]) is not int or row["round"] not in {1, 2}
            ):
                raise ValueError(f"observer round must be null or frozen integer 1/2: {event_path}")
            if any(row[key] != common_identity[key] for key in identity_keys):
                raise ValueError(f"observer identity mismatch: {event_path}:{index}")
            for key in identity_keys:
                if (
                    type(row[key]) is not type(binding[key])
                    or row[key] != binding[key]
                ):
                    raise ValueError(
                        f"observer binding mismatch for {key}: {event_path}:{index}"
                    )
            expected_suffix = f"/{row['process_instance_id']}/{index}"
            if not str(row["event_id"]).endswith(expected_suffix):
                raise ValueError(f"event_id/sequence mismatch: {event_path}:{index}")
            if row["event_id"] in global_event_ids:
                raise ValueError(f"duplicate observer event_id: {row['event_id']}")
            global_event_ids.add(str(row["event_id"]))
        process_values = {str(row["process_instance_id"]) for row in rows}
        host_values = {str(row["host_id"]) for row in rows}
        producer_values = {str(row["producer"]) for row in rows}
        if len(process_values) != 1 or len(host_values) != 1 or len(producer_values) != 1:
            raise ValueError(f"mixed process/host/producer identity: {event_path}")

        substantive = [row for row in rows if row["event_type"] != "observer_overhead"]
        overhead = [row for row in rows if row["event_type"] == "observer_overhead"]
        observed = [str(row["payload"].get("observed_event_id")) for row in overhead]
        substantive_ids = [str(row["event_id"]) for row in substantive]
        if sorted(observed) != sorted(substantive_ids) or len(observed) != len(set(observed)):
            raise ValueError(f"observer overhead pairing mismatch: {event_path}")
        for row in overhead:
            payload = row["payload"]
            expected_overhead_keys = {
                "observed_event_id",
                "observer_flower_serialize_ns",
                "observer_event_encode_ns",
                "observer_io_write_ns",
                "observer_fsync_ns",
                "observer_total_ns",
                "observer_event_bytes_written",
                "observer_event_count",
            }
            if set(payload) != expected_overhead_keys:
                raise ValueError(f"observer overhead schema mismatch: {event_path}")
            numeric_fields = expected_overhead_keys - {"observed_event_id"}
            if any(
                type(payload[field]) is not int or payload[field] < 0
                for field in numeric_fields
            ):
                raise ValueError(
                    f"observer overhead integer type/nonnegative mismatch: {event_path}"
                )
            components = [
                payload["observer_flower_serialize_ns"],
                payload["observer_event_encode_ns"],
                payload["observer_io_write_ns"],
                payload["observer_fsync_ns"],
            ]
            if payload["observer_total_ns"] != sum(components):
                raise ValueError(f"observer overhead total mismatch: {event_path}")
            if (
                payload["observer_event_bytes_written"] <= 0
                or payload["observer_event_count"] <= 0
            ):
                raise ValueError(f"invalid observer overhead accounting: {event_path}")
        client_ids = {
            str(row["client_id"])
            for row in substantive
            if row["client_id"] is not None
        }
        producer = str(first["producer"])
        if producer == "server":
            logical_name = "server"
        elif client_ids == {"C1"}:
            logical_name = "C1"
        elif client_ids == {"C2"}:
            logical_name = "C2"
        else:
            raise ValueError(f"observer producer is not server/C1/C2: {event_path}")
        if logical_name in producer_rows:
            raise ValueError(f"duplicate observer producer: {logical_name}")
        endpoint = binding["producers"][logical_name]
        if first["host_id"] != endpoint["host_id"] or first["producer"] != endpoint["producer"]:
            raise ValueError(f"observer producer binding mismatch: {logical_name}")
        expected_client_id = endpoint["client_id"]
        if logical_name != "server" and client_ids != {expected_client_id}:
            raise ValueError(f"observer client binding mismatch: {logical_name}")

        close_path = event_path.with_suffix(".close.json")
        if close_path not in close_files:
            raise ValueError(f"missing close summary for {logical_name}: {close_path}")
        close = json.loads(
            close_path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
        _finite_json(close)
        if set(close) != required_close_keys:
            raise ValueError(f"observer close schema mismatch: {close_path}")
        for key in ("schema_version", "run_id", "attempt_id", "host_id", "producer", "process_instance_id"):
            if close[key] != first[key]:
                raise ValueError(f"observer close identity mismatch: {close_path}")
        close_numeric_fields = required_close_keys - {
            "schema_version",
            "run_id",
            "attempt_id",
            "host_id",
            "producer",
            "process_instance_id",
        }
        if any(
            type(close[field]) is not int or close[field] < 0
            for field in close_numeric_fields
        ):
            raise ValueError(
                f"observer close integer type/nonnegative mismatch: {close_path}"
            )
        if close["observer_event_count"] != len(rows):
            raise ValueError(f"observer close event count mismatch: {close_path}")
        if close["observer_event_bytes_written"] != event_path.stat().st_size:
            raise ValueError(
                f"observer close byte count differs from JSONL size: {close_path}"
            )
        close_components = [
            close["observer_flower_serialize_ns"],
            close["observer_event_encode_ns"],
            close["observer_io_write_ns"],
            close["observer_fsync_ns"],
        ]
        if close["observer_total_ns"] != sum(close_components):
            raise ValueError(f"observer close total mismatch: {close_path}")

        type_counts: dict[str, int] = {}
        for row in substantive:
            event_type = str(row["event_type"])
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            payload = row["payload"]
            if not isinstance(payload, Mapping):
                raise ValueError(f"observer event payload is not an object: {event_path}")
            for key, value in payload.items():
                if str(key).endswith("_ns") and (
                    type(value) is not int or value < 0
                ):
                    raise ValueError(
                        f"observer event has invalid nonnegative timing: {event_path}/{key}"
                    )
        producer_rows[logical_name] = {
            "event_file": _portable_evidence_path(root, event_path),
            "close_summary": _portable_evidence_path(root, close_path),
            "events": len(rows),
            "substantive_events": len(substantive),
            "overhead_events": len(overhead),
            "event_type_counts": type_counts,
            "observer_total_ns": close["observer_total_ns"],
            "observer_event_bytes_written": close["observer_event_bytes_written"],
        }
        substantive_by_producer[logical_name] = substantive

    if set(producer_rows) != {"server", "C1", "C2"}:
        raise ValueError(f"missing required server/C1/C2 producers: {sorted(producer_rows)}")
    for client in ("C1", "C2"):
        counts = producer_rows[client]["event_type_counts"]
        expected = {
            "client_fit_start": 2,
            "client_train_start": 2,
            "client_train_end": 2,
            "client_fit_end": 2,
        }
        if counts != expected:
            raise ValueError(f"{client} event counts differ from two-round contract: {counts}")
        client_rows = substantive_by_producer[client]
        observed_client_matrix = {
            (row["round"], str(row["event_type"]), str(row["client_id"]))
            for row in client_rows
        }
        expected_client_matrix = {
            (round_idx, event_type, client)
            for round_idx in (1, 2)
            for event_type in expected
        }
        if (
            len(observed_client_matrix) != len(client_rows)
            or observed_client_matrix != expected_client_matrix
        ):
            raise ValueError(f"{client} round/client event matrix is invalid")
    server_counts = producer_rows["server"]["event_type_counts"]
    expected_server_counts = {
        "fit_round_start": 2,
        "flower_fitins_prepared": 4,
        "server_aggregate_start": 2,
        "flower_fitres_available": 4,
        "server_da_start": 2,
        "server_da_end": 2,
        "server_aggregate_end": 2,
        "fit_round_end": 2,
    }
    if server_counts != expected_server_counts:
        raise ValueError(f"server event counts differ from two-round contract: {server_counts}")
    server_rows = substantive_by_producer["server"]
    fit_res_rows = [
        row for row in server_rows if row["event_type"] == "flower_fitres_available"
    ]
    proxy_clients: dict[tuple[int, str], str] = {}
    fit_res_matrix: list[tuple[int, str]] = []
    for row in fit_res_rows:
        _validate_application_audit(row["payload"].get("uplink_audit"), direction="uplink")
        round_idx = row["round"]
        client_id = str(row["client_id"])
        proxy_id = str(row["payload"].get("proxy_id"))
        key = (round_idx, proxy_id)
        if key in proxy_clients or client_id not in {"C1", "C2"}:
            raise ValueError("server FitRes round/client matrix has invalid proxy binding")
        proxy_clients[key] = client_id
        fit_res_matrix.append((round_idx, client_id))
    expected_message_matrix = [
        (round_idx, client_id)
        for round_idx in (1, 2)
        for client_id in ("C1", "C2")
    ]
    if sorted(fit_res_matrix) != expected_message_matrix:
        raise ValueError("server FitRes round/client matrix lacks C1 or C2")
    fit_ins_matrix: list[tuple[int, str]] = []
    for row in server_rows:
        if row["event_type"] != "flower_fitins_prepared":
            continue
        _validate_application_audit(
            row["payload"].get("downlink_audit"), direction="downlink"
        )
        round_idx = row["round"]
        client_id = proxy_clients.get(
            (round_idx, str(row["payload"].get("proxy_id")))
        )
        if client_id is None:
            raise ValueError("server FitIns round/client matrix lacks proxy binding")
        fit_ins_matrix.append((round_idx, client_id))
    if sorted(fit_ins_matrix) != expected_message_matrix:
        raise ValueError("server FitIns round/client matrix lacks C1 or C2")
    return {
        "enabled": True,
        "event_files": len(event_files),
        "close_summaries": len(close_files),
        "producers": producer_rows,
        "binding_sha256": _sha256_bytes(_canonical_bytes(binding)),
    }


def _common_trace_fingerprint(path: Path) -> dict[str, Any]:
    trace_path = _require_regular_file(Path(path))
    rows = _read_events(trace_path)
    initial_rows = [row for row in rows if row.get("record_type") == "initial_checkpoint"]
    fit_ins_rows = [row for row in rows if row.get("record_type") == "fit_ins"]
    fit_res_rows = [row for row in rows if row.get("record_type") == "fit_res"]
    post_fit_res_rows = [
        row for row in rows if row.get("record_type") == "fit_res_post_observer"
    ]
    summary_rows = [row for row in rows if row.get("record_type") == "round_summary"]
    if not (
        len(initial_rows) == 1
        and len(fit_ins_rows) == 4
        and len(fit_res_rows) == 4
        and len(post_fit_res_rows) == 4
        and len(summary_rows) == 2
    ):
        raise ValueError(
            "common trace record count mismatch: "
            f"initial={len(initial_rows)}, FitIns={len(fit_ins_rows)}, "
            f"FitRes={len(fit_res_rows)}, post-FitRes={len(post_fit_res_rows)}, "
            f"rounds={len(summary_rows)}"
        )
    proxy_clients: dict[tuple[int, str], str] = {}
    for row in fit_res_rows:
        key = (int(row["round"]), str(row["proxy_id"]))
        client_id = str(row["client_id"])
        if key in proxy_clients or client_id not in {"C1", "C2"}:
            raise ValueError("common trace proxy/client binding is invalid")
        proxy_clients[key] = client_id

    fit_ins: dict[str, dict[str, Any]] = {"1": {}, "2": {}}
    fit_res: dict[str, dict[str, Any]] = {"1": {}, "2": {}}
    post_fit_res: dict[str, dict[str, Any]] = {"1": {}, "2": {}}
    raw_messages: list[dict[str, Any]] = []
    for row in fit_ins_rows:
        round_idx = int(row["round"])
        client_id = proxy_clients.get((round_idx, str(row["proxy_id"])))
        if client_id is None or client_id in fit_ins[str(round_idx)]:
            raise ValueError("common trace FitIns/client binding is invalid")
        fit_ins[str(round_idx)][client_id] = row["trace"]["comparison"]
        raw_messages.append(
            {
                "round": round_idx,
                "direction": "downlink",
                "client_id": client_id,
                "application_message_bytes": row["trace"][
                    "raw_application_message_bytes"
                ],
                "application_message_sha256": row["trace"][
                    "raw_application_message_sha256"
                ],
                "logical": row["trace"]["logical"],
            }
        )
    for row in fit_res_rows:
        round_idx = int(row["round"])
        client_id = str(row["client_id"])
        if client_id in fit_res[str(round_idx)]:
            raise ValueError("duplicate common trace FitRes/client binding")
        fit_res[str(round_idx)][client_id] = row["trace"]["comparison"]
        raw_messages.append(
            {
                "round": round_idx,
                "direction": "uplink",
                "client_id": client_id,
                "application_message_bytes": row["trace"][
                    "raw_application_message_bytes"
                ],
                "application_message_sha256": row["trace"][
                    "raw_application_message_sha256"
                ],
                "logical": row["trace"]["logical"],
            }
        )
    for row in post_fit_res_rows:
        round_idx = int(row["round"])
        client_id = str(row["client_id"])
        if client_id in post_fit_res[str(round_idx)]:
            raise ValueError("duplicate common trace post-observer FitRes/client binding")
        post_fit_res[str(round_idx)][client_id] = row["trace"]["comparison"]
    for round_idx in (1, 2):
        if set(fit_ins[str(round_idx)]) != {"C1", "C2"} or set(
            fit_res[str(round_idx)]
        ) != {"C1", "C2"} or set(post_fit_res[str(round_idx)]) != {"C1", "C2"}:
            raise ValueError(f"common trace round {round_idx} lacks C1/C2")

    initial = initial_rows[0]
    initial_tensors = initial["comparison"]["tensors"]
    for client_id in ("C1", "C2"):
        sent = fit_ins["1"][client_id]["parameters"]["tensors"]
        for item in sent:
            initial_key = f"model_state.{item['key']}"
            if (
                initial_key not in initial_tensors
                or f"torch.{item['dtype']}" != initial_tensors[initial_key]["dtype"]
                or item["shape"] != initial_tensors[initial_key]["shape"]
                or item["raw_sha256"] != initial_tensors[initial_key]["raw_sha256"]
            ):
                raise ValueError(
                    f"round-1 FitIns does not match frozen initial checkpoint: {client_id}/{item['key']}"
                )
    summaries = {str(int(row["round"])): row["comparison"] for row in summary_rows}
    if set(summaries) != {"1", "2"}:
        raise ValueError("common trace round summaries are incomplete")
    comparison = {
        "initial_tensor_content_sha256": initial["tensor_content_sha256"],
        "initial_parameter_keys": initial["parameter_keys"],
        "fit_ins": fit_ins,
        "fit_res": fit_res,
        "fit_res_post_observer": post_fit_res,
        "rounds": summaries,
    }
    raw_messages.sort(key=lambda row: (row["round"], row["direction"], row["client_id"]))
    return {
        "kind": "flower_common_trace",
        "artifact_sha256": _sha256_file(trace_path),
        "content_sha256": _sha256_bytes(_canonical_bytes(comparison)),
        "raw_messages": raw_messages,
        "comparison": comparison,
    }


def _message_fingerprints(on_attempt: Mapping[str, Any]) -> list[dict[str, Any]]:
    attempt_root = Path(on_attempt["attempt"])
    server_candidates: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in sorted(attempt_root.rglob("*events.jsonl")):
        rows = _read_events(path)
        if rows and {row.get("producer") for row in rows} == {"server"}:
            server_candidates.append((path, rows))
    if len(server_candidates) != 1:
        raise ValueError(
            "expected exactly one server observer event file, got "
            f"{[str(path) for path, _rows in server_candidates]}"
        )
    _server_path, events = server_candidates[0]
    proxy_clients: dict[tuple[int, str], str] = {}
    for event in events:
        if event.get("event_type") != "flower_fitres_available":
            continue
        round_idx = int(event["round"])
        client_id = str(event.get("client_id"))
        proxy_id = str(event["payload"]["proxy_id"])
        key = (round_idx, proxy_id)
        if key in proxy_clients or client_id not in {"C1", "C2"}:
            raise ValueError("observer FitRes proxy/client binding is invalid")
        proxy_clients[key] = client_id

    messages: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("event_type")
        if event_type == "flower_fitins_prepared":
            audit = event["payload"]["downlink_audit"]
            direction = "downlink"
            client_id = proxy_clients.get(
                (int(event["round"]), str(event["payload"]["proxy_id"]))
            )
            if client_id is None:
                raise ValueError("observer FitIns proxy/client binding is missing")
        elif event_type == "flower_fitres_available":
            audit = event["payload"]["uplink_audit"]
            direction = "uplink"
            client_id = str(event.get("client_id"))
        else:
            continue
        messages.append(
            {
                "round": int(event["round"]),
                "direction": direction,
                "client_id": client_id,
                "application_message_bytes": int(audit["application_message_bytes"]),
                "application_message_sha256": str(audit["application_message_sha256"]),
                "logical": audit["logical"],
            }
        )
    messages.sort(
        key=lambda item: (
            item["round"], item["direction"], item["client_id"] or "", item["application_message_sha256"]
        )
    )
    if len(messages) != 8:
        raise RuntimeError(f"expected 8 audited FitIns/FitRes messages, got {len(messages)}")
    if sum(item["direction"] == "downlink" for item in messages) != 4:
        raise RuntimeError("expected exactly four FitIns message fingerprints")
    if sum(item["direction"] == "uplink" for item in messages) != 4:
        raise RuntimeError("expected exactly four FitRes message fingerprints")
    return messages


def _cross_validate_message_audits(
    observer_messages: list[dict[str, Any]], common_trace: Mapping[str, Any]
) -> dict[str, Any]:
    """Require observer message audits to equal the independent live-object trace."""

    common_messages = common_trace.get("raw_messages")
    if not isinstance(common_messages, list):
        raise ValueError("common trace lacks raw message audits")
    key = lambda item: (
        int(item["round"]),
        str(item["direction"]),
        str(item["client_id"]),
    )
    observer_sorted = sorted(copy.deepcopy(observer_messages), key=key)
    common_sorted = sorted(copy.deepcopy(common_messages), key=key)
    if observer_sorted != common_sorted:
        raise ValueError(
            "observer message audits differ from independent common trace: "
            f"observer_sha256={_sha256_bytes(_canonical_bytes(observer_sorted))}, "
            f"common_trace_sha256={_sha256_bytes(_canonical_bytes(common_sorted))}"
        )
    return {
        "message_count": len(observer_sorted),
        "content_sha256": _sha256_bytes(_canonical_bytes(observer_sorted)),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary report path already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prepare_output_root(output_root: Path) -> Path:
    root = Path(output_root)
    if os.path.lexists(root):
        raise FileExistsError(f"output_root must not already exist: {root}")
    _require_no_link_ancestors(root.parent)
    parent = root.parent.resolve(strict=True)
    resolved = parent / root.name
    resolved.mkdir()
    return resolved


def run_local_gate(output_root: Path, group_id: str) -> dict[str, Any]:
    """Run real B2/B5 Flower CLIs OFF-A, ON, OFF-B on synthetic data."""

    group = str(group_id).upper()
    if group not in _GROUPS:
        raise ValueError(f"group_id must be one of {_GROUPS}, got {group_id!r}")
    root = _prepare_output_root(Path(output_root))
    fixture_hashes = _write_fixture(root)
    fixture_root = root / "fixture_data"
    fixture_sha = _sha256_bytes(_canonical_bytes(fixture_hashes))
    initial_checkpoint = _create_frozen_initial_checkpoint(
        root / "frozen_initial_checkpoint.pth", group
    )
    ports: set[int] = set()
    attempts: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    try:
        for mode in _MODES:
            attempts[mode] = _launch_attempt(
                root=root,
                fixture_root=fixture_root,
                group_id=group,
                mode=mode,
                port=_reserve_port(ports),
                fixture_sha=fixture_sha,
                initial_checkpoint=Path(initial_checkpoint["path"]),
            )
            artifacts[mode] = _capture_artifacts(attempts[mode])
        comparison = compare_fingerprints(
            artifacts["off_a"], artifacts["on"], artifacts["off_b"]
        )
        messages = _message_fingerprints(attempts["on"])
        try:
            message_cross_validation = _cross_validate_message_audits(
                messages, artifacts["on"]["common_trace"]
            )
            message_cross_validation["status"] = "matched"
        except ValueError as exc:
            message_cross_validation = {
                "status": "observer_path_mutation",
                "error": str(exc),
            }
            comparison["status"] = "observer_path_mutation"
            comparison["equivalent"] = False
            comparison["on_equal_to_off"] = False
            comparison["mismatches"] = [
                *comparison["mismatches"],
                f"message_common_trace_cross_validation: {exc}",
            ]
        content_hashes = comparison["content_set_sha256"]
        portable_initial_checkpoint = {
            **initial_checkpoint,
            "path": _portable_evidence_path(
                root, Path(initial_checkpoint["path"])
            ),
        }
        report = {
            "schema_version": "iotj.observer_equivalence.v1",
            "group_id": group,
            "fixture": {
                "numpy_seed": 7001,
                "training_seed": 42,
                "rounds": 2,
                "clients": ["C1", "C2"],
                "local_epochs": 1,
                "batch_size": 4,
                "window_shape": [100, 8],
                "rows_per_source": 8,
                "input_hashes": fixture_hashes,
                "input_set_sha256": fixture_sha,
                "frozen_initial_checkpoint": portable_initial_checkpoint,
            },
            "execution_order": ["OFF-A", "ON", "OFF-B"],
            "status": comparison["status"],
            "equivalent": comparison["equivalent"],
            "max_abs_delta": comparison["max_abs_delta"],
            "off_pair_equal": comparison["off_pair_equal"],
            "on_equal_to_off": comparison["on_equal_to_off"],
            "mismatches": comparison["mismatches"],
            "artifact_hashes": comparison["artifact_hashes"],
            "artifact_content_set_sha256": content_hashes,
            "final_checkpoint_sha256": {
                mode: artifacts[mode]["final_adapted_checkpoint"]["artifact_sha256"]
                for mode in _MODES
            },
            "final_checkpoint_raw_sha256": {
                mode: {
                    "aggregated": artifacts[mode]["final_aggregated_checkpoint"]["artifact_sha256"],
                    "adapted": artifacts[mode]["final_adapted_checkpoint"]["artifact_sha256"],
                }
                for mode in _MODES
            },
            "final_checkpoint_tensor_content_sha256": {
                mode: {
                    "aggregated": artifacts[mode]["final_aggregated_checkpoint"]["content_sha256"],
                    "adapted": artifacts[mode]["final_adapted_checkpoint"]["content_sha256"],
                }
                for mode in _MODES
            },
            "message_fingerprints": messages,
            "message_fingerprint_sha256": _sha256_bytes(_canonical_bytes(messages)),
            "message_common_trace_cross_validation": message_cross_validation,
            "observer_sidecars": {
                mode: attempts[mode]["sidecars"] for mode in _MODES
            },
            "orchestration_bindings": {
                mode: {
                    "server_address": attempts[mode]["server_address"],
                    "port": attempts[mode]["port"],
                }
                for mode in _MODES
            },
            "boundaries": {
                "dataset": "synthetic-only; no C5 test or project dataset opened",
                "topology": "local loopback; formal ECS/Pi/PC smoke is a later gate",
                "message_capture": "FitIns/FitRes common traces are captured from live Flower objects independently of JsonlObserver in OFF-A/ON/OFF-B; ON sidecars are separately audited",
            },
        }
    except Exception as exc:
        report = {
            "schema_version": "iotj.observer_equivalence.v1",
            "group_id": group,
            "status": "gate_execution_error",
            "equivalent": False,
            "max_abs_delta": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fixture_input_hashes": fixture_hashes,
            "frozen_initial_checkpoint": initial_checkpoint,
        }
    _atomic_write_json(root / "observer_equivalence_report.json", report)
    return report


def _load_formal_frozen_binding(protocol_path: Path, group_id: str) -> dict[str, Any]:
    from scripts.freeze_iotj_confirmation_protocol import canonical_sha256
    from scripts.run_iotj_confirmation_observability import load_frozen_inputs

    protocol = _require_regular_file(Path(protocol_path))
    summary_root = protocol.parent
    summary_name = summary_root.name
    if not summary_name.endswith("_summary"):
        raise ValueError("formal protocol manifest must be inside the frozen *_summary root")
    base_name = summary_name[: -len("_summary")]
    experiment_root = summary_root.parent / base_name
    command_root = summary_root.parent / f"{base_name}_commands"
    source_manifest = summary_root / "source_archive_manifest.json"
    dataset_manifest = summary_root / "dataset_manifest.json"
    archive = experiment_root / "source" / "confirmation_source.tar"
    frozen = load_frozen_inputs(
        protocol,
        source_manifest,
        dataset_manifest,
        command_root,
        archive,
    )
    group = str(group_id).upper()
    candidates = [
        run for run in frozen.runs if run.group_id == group and run.seed == 42
    ]
    if len(candidates) != 1:
        raise ValueError(f"frozen input has no unique {group}/seed-42 run")
    frozen_run = candidates[0]
    return {
        "protocol_manifest_sha256": str(
            frozen.protocol["protocol_manifest_sha256"]
        ),
        "source_archive_sha256": str(
            frozen.source_manifest["source_archive_sha256"]
        ),
        "dataset_manifest_sha256": str(
            frozen.dataset_manifest["dataset_manifest_sha256"]
        ),
        "regular_members_sha256": str(
            frozen.source_manifest["regular_members_sha256"]
        ),
        "confirmation_commit": str(frozen.protocol["confirmation_commit"]),
        "group_id": group,
        "seed": 42,
        "command_manifest_sha256": canonical_sha256(frozen_run.manifest),
        "archive_sha256": _sha256_file(frozen.archive_path),
        "_frozen": frozen,
        "_frozen_run": frozen_run,
    }


def _capture_formal_artifacts(attempt_root: Path) -> dict[str, Any]:
    output = attempt_root / "raw" / "ecs" / "training"
    trace = attempt_root / "raw" / "ecs" / "common_trace.jsonl"
    artifacts: OrderedDict[str, Any] = OrderedDict()
    artifacts["final_aggregated_checkpoint"] = tensor_fingerprint(
        output / "server_latest.pth"
    )
    artifacts["final_adapted_checkpoint"] = tensor_fingerprint(
        output / "server_latest_adapted.pth"
    )
    for label in ("final_aggregated_checkpoint", "final_adapted_checkpoint"):
        raw_sha = artifacts[label]["artifact_sha256"]
        artifacts[f"{label}_raw"] = {
            "kind": "raw_checkpoint",
            "artifact_sha256": raw_sha,
            "content_sha256": raw_sha,
            "comparison": {"raw_file_sha256": raw_sha},
        }
    artifacts["common_trace"] = _common_trace_fingerprint(trace)
    for round_idx in (1, 2):
        artifacts[f"adapted_logits_round_{round_idx}"] = _fixed_adapted_logits(
            output / f"server_round_{round_idx:03d}_adapted.pth",
            "formal",
        )
        for stem in ("prototype_stats", "semantic_protos"):
            artifacts[f"{stem}_round_{round_idx}"] = json_fingerprint(
                output / f"{stem}_round_{round_idx:03d}.json", VOLATILE_JSON_PATHS
            )
        client_stats = json.loads(
            (output / f"client_stats_round_{round_idx:03d}.json").read_text(
                encoding="utf-8"
            )
        )
        for client in client_stats["clients"]:
            client_id = int(client["client_id"])
            normalized = _normalize_json(
                {"metrics": client}, VOLATILE_JSON_PATHS
            )
            artifacts[f"client_stats_round_{round_idx}_c{client_id}"] = {
                "kind": "json_projection",
                "artifact_sha256": _sha256_bytes(_canonical_bytes(client)),
                "content_sha256": _sha256_bytes(_canonical_bytes(normalized)),
                "comparison": normalized,
            }
        diagnostics = json.loads(
            (output / f"domain_adapt_round_{round_idx:03d}.json").read_text(
                encoding="utf-8"
            )
        )
        diagnostics.pop("semantic_protos_after_da", None)
        normalized_diagnostics = _normalize_json(
            {"metrics": diagnostics}, VOLATILE_JSON_PATHS
        )
        artifacts[f"domain_adapt_round_{round_idx}"] = {
            "kind": "json_projection",
            "artifact_sha256": _sha256_bytes(_canonical_bytes(diagnostics)),
            "content_sha256": _sha256_bytes(
                _canonical_bytes(normalized_diagnostics)
            ),
            "comparison": normalized_diagnostics,
        }
    return artifacts


def _validate_formal_resource_sidecar(
    host_root: Path,
    client_id: str,
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate one formal host's resource sampler evidence."""

    root = Path(host_root)
    expected_client = str(client_id)
    if expected_client not in {"C1", "C2"}:
        raise ValueError(f"resource sidecar client must be C1 or C2: {client_id!r}")
    binding = _validate_expected_resource_binding(
        expected_binding, expected_client
    )
    event_path = _require_regular_file(root / "resource.jsonl")
    close_path = _require_regular_file(root / "resource.close.json")
    rows = _read_events(event_path)
    if not rows:
        raise ValueError(f"empty resource sidecar: {event_path}")

    event_keys = {
        "schema_version",
        "event_id",
        "event_type",
        "run_id",
        "attempt_id",
        "group_id",
        "training_seed",
        "round",
        "client_id",
        "host_id",
        "producer",
        "process_instance_id",
        "sequence",
        "wall_time_utc",
        "monotonic_ns",
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
        "status",
        "payload",
    }
    identity_keys = (
        "schema_version",
        "run_id",
        "attempt_id",
        "group_id",
        "training_seed",
        "client_id",
        "host_id",
        "producer",
        "process_instance_id",
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
    )
    first = rows[0]
    identity = {key: first.get(key) for key in identity_keys}
    if (
        identity["schema_version"] != "iotj.confirmation.observability.v1"
        or identity["client_id"] != expected_client
        or identity["producer"] != "resource_sampler"
    ):
        raise ValueError(f"resource sidecar identity mismatch: {event_path}")
    for key, expected_value in binding.items():
        if (
            type(identity[key]) is not type(expected_value)
            or identity[key] != expected_value
        ):
            raise ValueError(
                f"resource binding mismatch for {key}: {event_path}"
            )
    event_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if set(row) != event_keys:
            raise ValueError(f"resource event exact schema mismatch: {event_path}:{index}")
        if any(row.get(key) != identity[key] for key in identity_keys):
            raise ValueError(f"resource event identity drift: {event_path}:{index}")
        if type(row["sequence"]) is not int or row["sequence"] != index:
            raise ValueError(f"resource event sequence is not contiguous: {event_path}")
        expected_suffix = f"/{identity['process_instance_id']}/{index}"
        if not str(row["event_id"]).endswith(expected_suffix):
            raise ValueError(f"resource event_id/sequence mismatch: {event_path}:{index}")
        if row["event_id"] in event_ids:
            raise ValueError(f"duplicate resource event_id: {row['event_id']}")
        event_ids.add(str(row["event_id"]))
        if type(row["monotonic_ns"]) is not int or row["monotonic_ns"] < 0:
            raise ValueError(f"resource event monotonic_ns must be nonnegative: {event_path}")

    substantive = [row for row in rows if row["event_type"] != "observer_overhead"]
    overhead = [row for row in rows if row["event_type"] == "observer_overhead"]
    if any(row["event_type"] not in {"resource_sample", "resource_sampler_end"} for row in substantive):
        raise ValueError(f"unexpected resource event type: {event_path}")
    observed_ids = [str(row["payload"].get("observed_event_id")) for row in overhead]
    substantive_ids = [str(row["event_id"]) for row in substantive]
    if sorted(observed_ids) != sorted(substantive_ids) or len(observed_ids) != len(set(observed_ids)):
        raise ValueError(f"resource observer overhead pairing mismatch: {event_path}")
    overhead_keys = {
        "observed_event_id",
        "observer_flower_serialize_ns",
        "observer_event_encode_ns",
        "observer_io_write_ns",
        "observer_fsync_ns",
        "observer_total_ns",
        "observer_event_bytes_written",
        "observer_event_count",
    }
    for row in overhead:
        payload = row["payload"]
        if not isinstance(payload, Mapping) or set(payload) != overhead_keys:
            raise ValueError(f"resource observer overhead exact schema mismatch: {event_path}")
        numeric_fields = overhead_keys - {"observed_event_id"}
        if any(
            type(payload[field]) is not int or payload[field] < 0
            for field in numeric_fields
        ):
            raise ValueError(
                f"resource observer overhead integer type/nonnegative mismatch: {event_path}"
            )
        components = [
            payload["observer_flower_serialize_ns"],
            payload["observer_event_encode_ns"],
            payload["observer_io_write_ns"],
            payload["observer_fsync_ns"],
        ]
        if payload["observer_total_ns"] != sum(components):
            raise ValueError(f"resource observer overhead total mismatch: {event_path}")
        if payload["observer_event_bytes_written"] <= 0 or payload["observer_event_count"] <= 0:
            raise ValueError(f"resource observer overhead accounting is invalid: {event_path}")

    sample_payload_keys = {
        "root_pid",
        "sampler_pid_excluded",
        "pids",
        "process_identities",
        "rss_tree_bytes",
        "rss_tree_peak_bytes",
        "process_count_tree",
        "thread_count_tree",
        "cpu_time_tree_seconds",
        "cpu_time_tree_delta_seconds",
        "cpu_percent_tree_one_core_scale",
        "cpu_percent_tree_host_scale",
        "logical_cpu_count",
        "sample_interval_start_monotonic_ns",
        "sample_interval_end_monotonic_ns",
        "sample_interval_wall_ns",
        "sample_errors",
        "cpu_temperature_c",
        "cpu_temperature_available",
        "cpu_temperature_source",
        "vcgencmd_available",
        "throttled_raw",
        "throttled_bits",
        "throttled_available",
        "thermal_errors",
    }
    integer_fields = {
        "root_pid",
        "sampler_pid_excluded",
        "rss_tree_bytes",
        "rss_tree_peak_bytes",
        "process_count_tree",
        "thread_count_tree",
        "logical_cpu_count",
        "sample_interval_start_monotonic_ns",
        "sample_interval_end_monotonic_ns",
        "sample_interval_wall_ns",
    }
    numeric_fields = integer_fields | {
        "cpu_time_tree_seconds",
        "cpu_time_tree_delta_seconds",
        "cpu_percent_tree_one_core_scale",
        "cpu_percent_tree_host_scale",
    }
    samples = [row for row in substantive if row["event_type"] == "resource_sample"]
    if not samples:
        raise ValueError(f"resource sidecar contains no samples: {event_path}")
    valid_intervals: list[tuple[int, int]] = []
    root_pid: int | None = None
    sampler_pid: int | None = None
    last_peak = 0
    last_end: int | None = None
    for row in samples:
        if row["round"] is not None or row["client_id"] != expected_client or row["status"] != "succeeded":
            raise ValueError(f"resource sample event identity/status is invalid: {event_path}")
        payload = row["payload"]
        if not isinstance(payload, Mapping) or set(payload) != sample_payload_keys:
            raise ValueError(f"resource sample exact schema mismatch: {event_path}")
        for field in numeric_fields:
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"resource {field} must be finite and nonnegative")
            if field in integer_fields and type(value) is not int:
                raise ValueError(f"resource {field} must be a nonnegative integer")
        if payload["root_pid"] <= 0 or payload["sampler_pid_excluded"] <= 0:
            raise ValueError("resource root/sampler PID must be positive")
        if root_pid is None:
            root_pid = int(payload["root_pid"])
            sampler_pid = int(payload["sampler_pid_excluded"])
        elif payload["root_pid"] != root_pid or payload["sampler_pid_excluded"] != sampler_pid:
            raise ValueError("resource root/sampler PID identity drift")
        pids = payload["pids"]
        if (
            not isinstance(pids, list)
            or any(type(pid) is not int or pid <= 0 for pid in pids)
            or pids != sorted(set(pids))
            or payload["sampler_pid_excluded"] in pids
            or payload["root_pid"] not in pids
            or payload["process_count_tree"] != len(pids)
        ):
            raise ValueError("resource pids/process_count_tree identity is invalid")
        identities = payload["process_identities"]
        if not isinstance(identities, list) or len(identities) != len(pids):
            raise ValueError("resource process_identities count mismatch")
        for pid, process_identity in zip(pids, identities):
            if not isinstance(process_identity, Mapping) or set(process_identity) != {
                "pid", "create_time", "identity_available"
            }:
                raise ValueError("resource process identity exact schema mismatch")
            if process_identity["pid"] != pid or process_identity["identity_available"] is not True:
                raise ValueError("resource process identity is unavailable or mismatched")
            create_time = process_identity["create_time"]
            if isinstance(create_time, bool) or not isinstance(create_time, (int, float)) or not math.isfinite(create_time) or create_time < 0:
                raise ValueError("resource process create_time must be finite and nonnegative")
        if payload["rss_tree_peak_bytes"] < payload["rss_tree_bytes"] or payload["rss_tree_peak_bytes"] < last_peak:
            raise ValueError("resource rss_tree_peak_bytes is invalid")
        last_peak = int(payload["rss_tree_peak_bytes"])
        start = int(payload["sample_interval_start_monotonic_ns"])
        end = int(payload["sample_interval_end_monotonic_ns"])
        if end < start or payload["sample_interval_wall_ns"] != end - start:
            raise ValueError("resource sample interval is invalid")
        if last_end is not None and start < last_end:
            raise ValueError("resource sample intervals overlap backwards")
        if row["monotonic_ns"] < end:
            raise ValueError("resource sample event clock precedes interval end")
        last_end = end
        valid_intervals.append((start, end))
        if payload["sample_errors"] != []:
            raise ValueError("resource sample contains sampling errors")
        if payload["thermal_errors"] != []:
            raise ValueError("resource sample contains thermal errors")
        for flag in ("cpu_temperature_available", "vcgencmd_available", "throttled_available"):
            if type(payload[flag]) is not bool:
                raise ValueError(f"resource {flag} must be boolean")
        temperature = payload["cpu_temperature_c"]
        if payload["cpu_temperature_available"]:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
                raise ValueError("resource CPU temperature must be finite and nonnegative")
            if payload["cpu_temperature_source"] not in {"sysfs", "vcgencmd"}:
                raise ValueError("resource CPU temperature source is invalid")
        elif temperature is not None or payload["cpu_temperature_source"] is not None:
            raise ValueError("unavailable resource CPU temperature must be null")
        if payload["throttled_available"]:
            if type(payload["throttled_bits"]) is not int or payload["throttled_bits"] < 0 or not isinstance(payload["throttled_raw"], str):
                raise ValueError("resource throttled state is invalid")
        elif payload["throttled_bits"] is not None or payload["throttled_raw"] is not None:
            raise ValueError("unavailable resource throttled state must be null")

    sampler_ends = [row for row in substantive if row["event_type"] == "resource_sampler_end"]
    if len(sampler_ends) != 1:
        raise ValueError(f"resource sidecar requires exactly one sampler end: {event_path}")
    end_event = sampler_ends[0]
    if end_event["round"] is not None or end_event["client_id"] != expected_client or end_event["status"] != "succeeded":
        raise ValueError(f"resource sampler end identity/status is invalid: {event_path}")
    end_payload = end_event["payload"]
    end_payload_keys = {
        "root_pid",
        "sampler_pid",
        "shutdown_reason",
        "shutdown_error",
        "sample_count",
        "sampler_cpu_user_seconds",
        "sampler_cpu_system_seconds",
        "sampler_rss_peak_bytes",
        "sampler_rss_peak_available",
        "sampler_rss_peak_method",
        "sampler_rss_peak_error",
        "observer_cost_values_scope",
        "observer_event_encode_ns",
        "observer_io_write_ns",
        "observer_fsync_ns",
        "observer_event_bytes_written",
        "observer_event_count",
        "observer_close_summary_path",
        "observer_close_summary_is_authoritative",
    }
    if not isinstance(end_payload, Mapping) or set(end_payload) != end_payload_keys:
        raise ValueError("resource sampler end exact schema mismatch")
    if (
        end_payload["root_pid"] != root_pid
        or end_payload["sampler_pid"] != sampler_pid
        or end_payload["sample_count"] != len(samples)
        or end_payload["shutdown_error"] is not None
        or end_payload["shutdown_reason"] not in {"stop_file", "target_exited"}
    ):
        raise ValueError("resource sampler end identity/count/error mismatch")
    exact_integer_fields = (
        "root_pid", "sampler_pid", "sample_count", "observer_event_encode_ns",
        "observer_io_write_ns", "observer_fsync_ns",
        "observer_event_bytes_written", "observer_event_count",
    )
    for field in exact_integer_fields:
        value = end_payload[field]
        if type(value) is not int or value < 0:
            raise ValueError(
                f"resource sampler end {field} must have exact nonnegative integer type"
            )
    for field in ("sampler_cpu_user_seconds", "sampler_cpu_system_seconds"):
        value = end_payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"resource sampler end {field} must be finite and nonnegative"
            )
    if end_payload["sampler_rss_peak_available"] is not True:
        raise ValueError("resource sampler RSS peak must be available")
    peak = end_payload["sampler_rss_peak_bytes"]
    if type(peak) is not int or peak < 0 or end_payload["sampler_rss_peak_error"] is not None:
        raise ValueError("resource sampler RSS peak is invalid")
    if not isinstance(end_payload["sampler_rss_peak_method"], str) or not end_payload["sampler_rss_peak_method"]:
        raise ValueError("resource sampler RSS peak method is invalid")
    if (
        end_payload["observer_cost_values_scope"] != "before_resource_sampler_end_emit"
        or end_payload["observer_close_summary_is_authoritative"] is not True
        or Path(str(end_payload["observer_close_summary_path"])).name != "resource.close.json"
    ):
        raise ValueError("resource sampler observer-cost boundary is invalid")

    close = json.loads(
        close_path.read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    _finite_json(close)
    close_keys = {
        "schema_version", "run_id", "attempt_id", "host_id", "producer",
        "process_instance_id", "observer_flower_serialize_ns",
        "observer_event_encode_ns", "observer_io_write_ns", "observer_fsync_ns",
        "observer_total_ns", "observer_event_bytes_written", "observer_event_count",
        "observer_reporting_tail_bytes",
    }
    if set(close) != close_keys:
        raise ValueError("resource close summary exact schema mismatch")
    for key in ("schema_version", "run_id", "attempt_id", "host_id", "producer", "process_instance_id"):
        if close[key] != identity[key]:
            raise ValueError("resource close summary identity mismatch")
    close_components = [
        close["observer_flower_serialize_ns"], close["observer_event_encode_ns"],
        close["observer_io_write_ns"], close["observer_fsync_ns"],
    ]
    close_numeric_fields = close_keys - {
        "schema_version", "run_id", "attempt_id", "host_id", "producer",
        "process_instance_id",
    }
    if any(
        type(close[field]) is not int or close[field] < 0
        for field in close_numeric_fields
    ):
        raise ValueError(
            "resource close summary integers must have exact nonnegative type"
        )
    if close["observer_total_ns"] != sum(close_components):
        raise ValueError("resource close summary overhead total mismatch")
    if close["observer_event_count"] != len(rows):
        raise ValueError("resource close summary event count mismatch")
    if close["observer_event_bytes_written"] != event_path.stat().st_size:
        raise ValueError("resource close summary byte count mismatch")
    if type(close["observer_reporting_tail_bytes"]) is not int or close["observer_reporting_tail_bytes"] < 0:
        raise ValueError("resource close summary tail bytes must be nonnegative")
    return {
        "sample_count": len(samples),
        "valid_intervals": valid_intervals,
        "event_file": _portable_evidence_path(root, event_path),
        "close_summary": _portable_evidence_path(root, close_path),
        "event_sha256": _sha256_file(event_path),
        "close_sha256": _sha256_file(close_path),
        "binding_sha256": _sha256_bytes(_canonical_bytes(binding)),
    }


def _formal_resource_samples_per_client_round(
    on_root: Path, resource_validations: Mapping[str, Mapping[str, Any]]
) -> int:
    minimum: int | None = None
    for client_id, host_root in (
        ("C1", on_root / "raw" / "pi"),
        ("C2", on_root / "raw" / "pc"),
    ):
        client_events = _read_events(
            _require_regular_file(host_root / "events.jsonl")
        )
        resource_validation = resource_validations.get(client_id)
        if not isinstance(resource_validation, Mapping):
            raise ValueError(f"formal resource validation is missing for {client_id}")
        valid_intervals = resource_validation["valid_intervals"]
        for round_idx in (1, 2):
            starts = [
                row
                for row in client_events
                if row.get("event_type") == "client_fit_start"
                and row.get("client_id") == client_id
                and row.get("round") == round_idx
            ]
            ends = [
                row
                for row in client_events
                if row.get("event_type") == "client_fit_end"
                and row.get("client_id") == client_id
                and row.get("round") == round_idx
            ]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError(f"{client_id} round {round_idx} fit interval is invalid")
            start_ns = starts[0]["monotonic_ns"]
            end_ns = ends[0]["monotonic_ns"]
            overlapping = 0
            for sample_start, sample_end in valid_intervals:
                if sample_start <= end_ns and sample_end >= start_ns:
                    overlapping += 1
            minimum = overlapping if minimum is None else min(minimum, overlapping)
    if minimum is None or minimum < 1:
        raise ValueError("formal smoke lacks one resource sample per client round")
    return minimum


def _run_formal_bound_gate(
    binding: Mapping[str, Any], output_root: Path, group_id: str
) -> dict[str, Any]:
    from scripts.run_iotj_confirmation_observability import (
        DEFAULT_PC_RUNTIME_ROOT,
        FormalSmokeConfig,
        ProductionRuntime,
        REPO_ROOT,
        build_production_hooks,
        run_noncanonical_smoke_attempt,
    )

    frozen = binding.get("_frozen")
    frozen_run = binding.get("_frozen_run")
    if frozen is None or frozen_run is None:
        raise ValueError("formal runner requires validated frozen input objects")
    root = _prepare_output_root(Path(output_root))
    initial = _create_frozen_initial_checkpoint(
        root / "frozen_initial_checkpoint.pth", group_id
    )
    runtime = ProductionRuntime(
        frozen=frozen,
        frozen_run=frozen_run,
        deployments={},
        ecs_host=os.environ.get("IOTJ_ECS_HOST", "root@121.40.139.213"),
        pi_host=os.environ.get("IOTJ_PI_HOST", "gaps@192.168.31.184"),
        validator=REPO_ROOT / "scripts" / "validate_iotj_confirmation_attempt.py",
        poll_seconds=1.0,
        timeout_seconds=1800.0,
        pc_runtime_root=Path(
            os.environ.get("IOTJ_PC_RUNTIME_ROOT", str(DEFAULT_PC_RUNTIME_ROOT))
        ),
    )
    attempts: dict[str, Path] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    sidecars: dict[str, Any] = {}
    resource_sidecars: dict[str, Any] = {}
    for mode in ("off", "on"):
        smoke = FormalSmokeConfig(
            observer_enabled=mode == "on",
            mode=mode,
            trace_output="{ecs_raw}/common_trace.jsonl",
            initial_checkpoint="{ecs_root}/frozen_initial_checkpoint.pth",
            initial_checkpoint_source=Path(initial["path"]),
            initial_checkpoint_sha256=initial["raw_sha256"],
        )
        attempt_root = root / mode
        smoke_attempt = run_noncanonical_smoke_attempt(
            attempt_root,
            run_id=frozen_run.run_id,
            mode=mode,
            provenance=frozen_run.provenance,
            hooks=build_production_hooks(runtime, smoke=smoke),
        )
        attempts[mode] = attempt_root
        provenance = frozen_run.provenance
        training_binding = _expected_training_binding(
            run_id=smoke_attempt.run_id,
            attempt_id=smoke_attempt.attempt_id,
            group_id=frozen_run.group_id,
            training_seed=frozen_run.seed,
            confirmation_commit=provenance.confirmation_commit,
            source_archive_sha256=provenance.source_archive_sha256,
            dataset_manifest_sha256=provenance.dataset_manifest_sha256,
            algorithm_config_sha256=provenance.algorithm_config_sha256,
            server_host_id="ecs",
            c1_host_id="pi-c1",
            c2_host_id="pc-c2",
        )
        sidecars[mode] = _validate_observer_sidecars(
            attempt_root,
            enabled=mode == "on",
            expected_binding=training_binding,
        )
        resource_sidecars[mode] = {
            "C1": _validate_formal_resource_sidecar(
                attempt_root / "raw" / "pi",
                "C1",
                expected_binding=_expected_resource_binding(
                    training_binding, client_id="C1", host_id="pi-c1"
                ),
            ),
            "C2": _validate_formal_resource_sidecar(
                attempt_root / "raw" / "pc",
                "C2",
                expected_binding=_expected_resource_binding(
                    training_binding, client_id="C2", host_id="pc-c2"
                ),
            ),
        }
        artifacts[mode] = _capture_formal_artifacts(attempt_root)
    # Formal topology has one OFF and one ON.  Reuse OFF as the deterministic
    # reference in the three-way classifier; local Gate separately proves the
    # OFF-A/OFF-B environment repeatability contract.
    comparison = compare_fingerprints(
        artifacts["off"], artifacts["on"], artifacts["off"]
    )
    resource_minimum = _formal_resource_samples_per_client_round(
        attempts["on"], resource_sidecars["on"]
    )
    messages = _message_fingerprints({"attempt": attempts["on"]})
    try:
        message_cross_validation = _cross_validate_message_audits(
            messages, artifacts["on"]["common_trace"]
        )
        message_cross_validation["status"] = "matched"
    except ValueError as exc:
        message_cross_validation = {
            "status": "observer_path_mutation",
            "error": str(exc),
        }
        comparison["status"] = "observer_path_mutation"
        comparison["equivalent"] = False
        comparison["on_equal_to_off"] = False
        comparison["mismatches"] = [
            *comparison["mismatches"],
            f"message_common_trace_cross_validation: {exc}",
        ]
    return {
        "status": comparison["status"],
        "equivalent": comparison["equivalent"],
        "max_abs_delta": comparison["max_abs_delta"],
        "modes": ["off", "on"],
        "resource_samples_per_client_round": resource_minimum,
        "message_fingerprints": messages,
        "message_common_trace_cross_validation": message_cross_validation,
        "observer_sidecars": sidecars,
        "resource_sidecars": resource_sidecars,
        "frozen_initial_checkpoint": {
            **initial,
            "path": _portable_evidence_path(root, Path(initial["path"])),
        },
        "artifact_hashes": comparison["artifact_hashes"],
        "mismatches": comparison["mismatches"],
    }


def run_formal_topology_gate(
    protocol_manifest: Path,
    output_root: Path,
    group_id: str,
    *,
    frozen_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    runner: Callable[[Mapping[str, Any], Path, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind and execute the noncanonical ECS/Pi/PC OFF/ON smoke lifecycle."""

    group = str(group_id).upper()
    if group not in _GROUPS:
        raise ValueError(f"group_id must be one of {_GROUPS}, got {group_id!r}")
    loader = frozen_loader or (lambda path: _load_formal_frozen_binding(path, group))
    bound = dict(loader(Path(protocol_manifest)))
    required = {
        "protocol_manifest_sha256": 64,
        "source_archive_sha256": 64,
        "dataset_manifest_sha256": 64,
        "regular_members_sha256": 64,
        "confirmation_commit": 40,
        "command_manifest_sha256": 64,
        "archive_sha256": 64,
    }
    for key, length in required.items():
        value = bound.get(key)
        if not isinstance(value, str) or len(value) != length or any(
            character not in "0123456789abcdef" for character in value.lower()
        ) or value != value.lower():
            raise ValueError(f"formal frozen binding has invalid {key}")
    if bound["archive_sha256"] != bound["source_archive_sha256"]:
        raise ValueError("formal source archive bytes differ from the frozen binding")
    if bound.get("group_id") != group or bound.get("seed") != 42:
        raise ValueError("formal frozen binding differs from requested group/seed-42")
    execute = runner or _run_formal_bound_gate
    result = dict(execute(bound, Path(output_root), group))
    report = {
        "schema_version": "iotj.observer_equivalence.formal.v1",
        "topology": "ECS server + Raspberry Pi C1 + PC C2",
        "group_id": group,
        "binding": {key: value for key, value in bound.items() if not key.startswith("_")},
        **result,
    }
    if runner is None:
        _atomic_write_json(Path(output_root) / "formal_observer_equivalence_report.json", report)
    return report


def main() -> None:
    if "--trace-child-role" in sys.argv[1:]:
        trace_parser = argparse.ArgumentParser(add_help=False)
        trace_parser.add_argument("--trace-child-role", choices=("server",), required=True)
        trace_parser.add_argument("--trace-output", type=Path, required=True)
        trace_parser.add_argument("--trace-initial-checkpoint", type=Path, required=True)
        trace_parser.add_argument("delegated", nargs=argparse.REMAINDER)
        trace_args = trace_parser.parse_args()
        delegated = list(trace_args.delegated)
        if delegated and delegated[0] == "--":
            delegated = delegated[1:]
        raise SystemExit(
            _run_traced_server(
                trace_args.trace_output,
                trace_args.trace_initial_checkpoint,
                delegated,
            )
        )
    parser = argparse.ArgumentParser(
        description="Run the local or formal-topology observer equivalence Gate"
    )
    parser.add_argument("--group", required=True, choices=_GROUPS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-topology", action="store_true")
    parser.add_argument("--protocol-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.formal_topology:
            if args.protocol_manifest is None:
                parser.error("--formal-topology requires --protocol-manifest")
            report = run_formal_topology_gate(
                args.protocol_manifest, args.output_root, args.group
            )
        else:
            if args.protocol_manifest is not None:
                parser.error("--protocol-manifest requires --formal-topology")
            report = run_local_gate(args.output_root, args.group)
    except Exception as exc:
        print(f"observer equivalence Gate refused to run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.get("status") == "equivalent" else 2)


if __name__ == "__main__":
    main()
