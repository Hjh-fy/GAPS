"""Replay the preserved B5 round-2 prototype loss with fixed client order.

This diagnostic is deliberately local and read-only.  It reconstructs the
initial round-2 prototype-loss state from preserved evidence, invokes the
production ``ServerDomainAdaptation._compute_server_proto_losses`` method, and
writes only one new sibling report directory.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.domain_adaptation import ServerDomainAdaptation


EXPECTED_COMMIT = "7ec77e35d57797ad1fc42d153221c3f3dff81644"
EXPECTED_SOURCE_SHA256 = (
    "c96fd135c159f0a4b28ecbc40008436510f79bb3adab347536d76a0305b01e1f"
)
EXPECTED_DATASET_SHA256 = (
    "fb8946da138bea5aa829dd1f5b733561a443083beb77a873e7173cbc95fcd430"
)
REPORT_NAME = "b5_order_replay_report.json"


def _canonical_bytes(value: Any) -> bytes:
    finite_json(value)
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


def finite_json(value: Any, path: tuple[Any, ...] = ()) -> None:
    """Reject values which cannot be represented by strict finite JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string JSON key at {path!r}: {key!r}")
            finite_json(item, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_json(item, path + (index,))


def path_is_link_or_reparse(path: Path) -> bool:
    candidate = Path(path)
    if candidate.is_symlink():
        return True
    try:
        metadata = candidate.lstat()
    except (FileNotFoundError, OSError):
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _require_no_link_ancestors(path: Path) -> None:
    candidate = Path(path).absolute()
    for ancestor in (candidate, *candidate.parents):
        if os.path.lexists(ancestor) and path_is_link_or_reparse(ancestor):
            raise ValueError(f"linked/reparse path component is forbidden: {ancestor}")


def _require_regular_file(path: Path) -> Path:
    candidate = Path(path)
    _require_no_link_ancestors(candidate.parent)
    if path_is_link_or_reparse(candidate):
        raise ValueError(f"linked/reparse file is forbidden: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"required regular file is missing: {candidate}")
    return candidate


def validate_roots(reference_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Validate immutable reference and new sibling output before any write."""

    reference = Path(reference_root)
    output = Path(output_root)
    _require_no_link_ancestors(reference)
    if path_is_link_or_reparse(reference) or not reference.is_dir():
        raise ValueError(f"reference root must be a real directory: {reference}")
    if path_is_link_or_reparse(output):
        raise ValueError(f"output root must not be a link/reparse point: {output}")
    _require_no_link_ancestors(output.parent)
    if not output.parent.is_dir():
        raise ValueError(f"output parent must be an existing directory: {output.parent}")
    if os.path.lexists(output):
        raise FileExistsError(f"output root already exists: {output}")

    resolved_reference = reference.resolve(strict=True)
    resolved_output = output.parent.resolve(strict=True) / output.name
    if (
        resolved_reference == resolved_output
        or resolved_reference in resolved_output.parents
        or resolved_output in resolved_reference.parents
    ):
        raise ValueError("reference/output overlap or ancestor relationship is forbidden")
    if resolved_reference.parent != resolved_output.parent:
        raise ValueError("diagnostic output must be a sibling of the reference root")
    return resolved_reference, resolved_output


def _manifest_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            path = Path(child.path)
            if path_is_link_or_reparse(path):
                raise ValueError(f"reference contains a link/reparse point: {path}")
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"reference contains a non-regular entry: {path}")
            relative = path.relative_to(root).as_posix()
            entries.append(
                {
                    "relative_path": relative,
                    "file_type": "regular_file",
                    "byte_length": int(metadata.st_size),
                    "sha256": _sha256_file(path),
                }
            )

    visit(root)
    return sorted(entries, key=lambda row: row["relative_path"])


def recursive_manifest(reference_root: Path) -> dict[str, Any]:
    root = Path(reference_root)
    _require_no_link_ancestors(root)
    if path_is_link_or_reparse(root) or not root.is_dir():
        raise ValueError(f"reference root must be a real directory: {root}")
    entries = _manifest_entries(root)
    return {
        "definition": "relative_path+file_type+byte_length+sha256",
        "file_count": len(entries),
        "files": entries,
        "manifest_sha256": _sha256_bytes(_canonical_bytes(entries)),
    }


def run_with_reference_guard(
    reference_root: Path, operation: Callable[[], Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run an operation and verify the recursive manifest even on failure."""

    before = recursive_manifest(reference_root)
    result: Any = None
    failure: BaseException | None = None
    try:
        result = operation()
    except BaseException as exc:  # revalidation is mandatory for every failure
        failure = exc
    after = recursive_manifest(reference_root)
    if after != before:
        raise RuntimeError("reference root changed during fixed-state replay") from failure
    if failure is not None:
        raise failure
    return result, before, after


def _load_json(path: Path) -> Any:
    source = _require_regular_file(Path(path))
    value = json.loads(source.read_text(encoding="utf-8"))
    finite_json(value)
    return value


def unwrap_typed_scalar(record: Mapping[str, Any]) -> Any:
    """Invert the exact typed Flower scalar record used by the common trace."""

    if not isinstance(record, Mapping) or type(record.get("type")) is not str:
        raise ValueError("typed scalar record has no exact type")
    scalar_type = record["type"]
    if scalar_type == "bytes":
        if set(record) != {"type", "value_base64"} or type(
            record.get("value_base64")
        ) is not str:
            raise ValueError("typed bytes scalar has invalid schema")
        try:
            return base64.b64decode(record["value_base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("typed bytes scalar has invalid base64") from exc
    if set(record) != {"type", "value"}:
        raise ValueError("typed scalar has invalid schema")
    value = record["value"]
    expected = {"bool": bool, "int": int, "float": float, "str": str}.get(
        scalar_type
    )
    if expected is None or type(value) is not expected:
        raise ValueError(f"typed scalar value does not match type {scalar_type!r}")
    if scalar_type == "float" and not math.isfinite(value):
        raise ValueError("typed float scalar must be finite")
    return value


def _unwrap_typed_mapping(record: Mapping[str, Any]) -> OrderedDict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {
        "key_order",
        "keys",
        "types",
        "values",
    }:
        raise ValueError("typed metric mapping has invalid schema")
    key_order = record["key_order"]
    keys = record["keys"]
    types = record["types"]
    values = record["values"]
    if (
        not isinstance(key_order, list)
        or not isinstance(keys, list)
        or key_order != keys
        or any(type(key) is not str for key in keys)
        or len(keys) != len(set(keys))
        or not isinstance(types, Mapping)
        or not isinstance(values, Mapping)
        or set(types) != set(keys)
        or set(values) != set(keys)
    ):
        raise ValueError("typed metric mapping key order is invalid")
    output: OrderedDict[str, Any] = OrderedDict()
    for key in key_order:
        scalar = values[key]
        if not isinstance(scalar, Mapping) or scalar.get("type") != types[key]:
            raise ValueError(f"typed metric type mirror differs for {key}")
        output[key] = unwrap_typed_scalar(scalar)
    return output


def read_round_fit_res(trace_path: Path, *, round_idx: int) -> dict[str, Any]:
    """Read only primary FitRes rows, retaining typed and unwrapped metrics."""

    path = _require_regular_file(Path(trace_path))
    selected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank common-trace row at line {line_number}")
            row = json.loads(line)
            finite_json(row)
            if row.get("record_type") == "fit_res" and row.get("round") == round_idx:
                selected.append(row)
    clients: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for row in selected:
        client_label = row.get("client_id")
        if client_label not in {"C1", "C2"}:
            raise ValueError("FitRes row has invalid client identity")
        client_id = int(client_label[1:])
        if client_id in clients:
            raise ValueError(f"duplicate FitRes row for client {client_id}")
        try:
            comparison = row["trace"]["comparison"]
            typed_metrics = comparison["metrics"]
            num_examples = comparison["num_examples"]
        except (KeyError, TypeError) as exc:
            raise ValueError("FitRes row lacks typed comparison values") from exc
        if type(num_examples) is not int or num_examples <= 0:
            raise ValueError("FitRes num_examples must be an exact positive integer")
        metrics = _unwrap_typed_mapping(typed_metrics)
        if metrics.get("client_id") != client_id or type(metrics.get("client_id")) is not int:
            raise ValueError("FitRes typed client_id differs from row identity")
        if metrics.get("num_examples") != num_examples or type(
            metrics.get("num_examples")
        ) is not int:
            raise ValueError("FitRes typed num_examples differs from message value")
        clients[client_id] = {
            "client_id": client_id,
            "num_examples": num_examples,
            "typed_metrics": typed_metrics,
            "metrics": metrics,
        }
    if set(clients) != {1, 2} or len(selected) != 2:
        raise ValueError(f"round {round_idx} must contain exactly one FitRes for C1/C2")
    return {
        "round": int(round_idx),
        "arrival_order": list(clients),
        "clients": clients,
        "source_sha256": _sha256_file(path),
    }


def _integer_client_mapping(value: Mapping[Any, Any]) -> dict[int, Any]:
    output: dict[int, Any] = {}
    for key, item in value.items():
        try:
            client_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid client mapping key: {key!r}") from exc
        if client_id in output:
            raise ValueError("duplicate normalized client mapping key")
        output[client_id] = item
    return output


def cross_validate_fit_res(off: Mapping[str, Any], on: Mapping[str, Any]) -> dict[str, Any]:
    if off.get("arrival_order") != [2, 1]:
        raise ValueError("OFF round-2 FitRes arrival order must be [2,1]")
    if on.get("arrival_order") != [1, 2]:
        raise ValueError("ON round-2 FitRes arrival order must be [1,2]")
    off_clients = _integer_client_mapping(off.get("clients", {}))
    on_clients = _integer_client_mapping(on.get("clients", {}))
    if set(off_clients) != {1, 2} or set(on_clients) != {1, 2}:
        raise ValueError("OFF/ON client sets differ")
    per_client_hashes: dict[str, str] = {}
    for client_id in (1, 2):
        off_view = {
            "num_examples": off_clients[client_id]["num_examples"],
            "typed_metrics": off_clients[client_id]["typed_metrics"],
        }
        on_view = {
            "num_examples": on_clients[client_id]["num_examples"],
            "typed_metrics": on_clients[client_id]["typed_metrics"],
        }
        if off_view != on_view:
            raise ValueError(f"client {client_id} OFF/ON typed FitRes values differ")
        per_client_hashes[str(client_id)] = _sha256_bytes(_canonical_bytes(off_view))
    return {
        "off_arrival_order": [2, 1],
        "on_arrival_order": [1, 2],
        "per_client_typed_values_equal": True,
        "per_client_typed_sha256": per_client_hashes,
    }


def reconstruct_semantic_protos(
    round_one: Mapping[str, Any],
    round_two_stats: Mapping[str, Any],
    *,
    config_alpha: float,
    checkpoint_semantic_protos: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    alpha = round_one.get("proto_ema_alpha")
    if type(alpha) is not float or not math.isfinite(alpha) or alpha != config_alpha:
        raise ValueError("semantic proto alpha differs from run_config")
    old_values = round_one.get("semantic_protos")
    new_values = round_two_stats.get("global_prototypes")
    if not isinstance(old_values, Mapping) or not isinstance(new_values, Mapping):
        raise ValueError("semantic prototype payload is malformed")
    if set(old_values) != set(checkpoint_semantic_protos):
        raise ValueError("round-1 semantic JSON/checkpoint keys differ")
    semantic: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, vector in old_values.items():
        tensor = torch.tensor(vector, dtype=torch.float32)
        checkpoint_tensor = checkpoint_semantic_protos[key].detach().cpu().float()
        if not torch.equal(tensor, checkpoint_tensor):
            raise ValueError(f"round-1 semantic JSON/checkpoint tensor differs: {key}")
        semantic[str(key)] = tensor
    for key, vector in new_values.items():
        new_tensor = torch.tensor(vector, dtype=torch.float32)
        if str(key) in semantic:
            semantic[str(key)] = (
                alpha * semantic[str(key)] + (1.0 - alpha) * new_tensor
            )
        else:
            semantic[str(key)] = new_tensor
    return semantic


def _tensor_raw(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError("tensor contains non-finite values")
    return value.numpy().tobytes(order="C")


def _tensor_record(tensor: torch.Tensor, *, include_raw: bool = True) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = _tensor_raw(value)
    result = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "raw_sha256": _sha256_bytes(raw),
    }
    if include_raw:
        result["raw_hex"] = raw.hex()
    return result


def _tensor_mapping_fingerprint(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    tensors = OrderedDict(
        (str(key), _tensor_record(tensor, include_raw=False))
        for key, tensor in values.items()
    )
    return {
        "key_order": list(tensors),
        "tensors": tensors,
        "content_sha256": _sha256_bytes(_canonical_bytes(tensors)),
    }


def _client_pair_record(
    client_id: int, client: Mapping[str, Any], weight: torch.Tensor
) -> dict[str, Any]:
    mus = client["mus"]
    counts = client["counts"]
    key_order = [f"{int(key[0])},{int(key[1])}" for key in mus]
    mu_records = OrderedDict(
        (f"{int(key[0])},{int(key[1])}", _tensor_record(value))
        for key, value in mus.items()
    )
    count_records = OrderedDict(
        (f"{int(key[0])},{int(key[1])}", int(counts[key])) for key in mus
    )
    payload = {
        "client_id": int(client_id),
        "prototype_key_order": key_order,
        "mus": mu_records,
        "counts": count_records,
        "residual": _tensor_record(client["residual"]),
        "weight": _tensor_record(weight),
        "num_examples": int(client["num_examples"]),
    }
    return {
        **payload,
        "paired_sha256": _sha256_bytes(_canonical_bytes(payload)),
    }


def run_proto_loss_replay(
    semantic_protos: Mapping[str, torch.Tensor],
    clients: Mapping[int, Mapping[str, Any]],
    order: Sequence[int],
    *,
    loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Invoke only the production prototype-loss method with an object.__new__ harness."""

    normalized_order = [int(client_id) for client_id in order]
    if len(normalized_order) != len(set(normalized_order)) or set(normalized_order) != set(
        clients
    ):
        raise ValueError("replay order must be an exact client permutation")
    if set(loss_weights) != {"proto", "proto_mmd", "residual"}:
        raise ValueError("loss weight schema is not exact")
    for value in loss_weights.values():
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("loss weights must be exact finite floats")

    examples = [int(clients[client_id]["num_examples"]) for client_id in normalized_order]
    if any(value <= 0 for value in examples):
        raise ValueError("client examples must be positive")
    total_examples = sum(examples)
    weights = torch.tensor(
        [value / total_examples for value in examples], dtype=torch.float32
    )
    pair_records = [
        _client_pair_record(client_id, clients[client_id], weights[index])
        for index, client_id in enumerate(normalized_order)
    ]
    pair_hashes = [record["paired_sha256"] for record in pair_records]

    trainer = object.__new__(ServerDomainAdaptation)
    trainer.device = torch.device("cpu")
    trainer.hp = {
        "USE_PROTO_DECOUPLING": loss_weights["residual"] > 0.0,
        "USE_PROTO_MMD": loss_weights["proto_mmd"] > 0.0,
    }
    trainer.semantic_protos = nn.ParameterDict(
        OrderedDict(
            (str(key), nn.Parameter(value.detach().cpu().float().clone()))
            for key, value in semantic_protos.items()
        )
    )
    trainer.device_residuals = nn.ParameterDict()
    trainer._set_round_client_statistics(
        client_mus=[clients[client_id]["mus"] for client_id in normalized_order],
        client_counts=[clients[client_id]["counts"] for client_id in normalized_order],
        client_weights=weights,
        client_ids=normalized_order,
        client_residuals=[clients[client_id]["residual"] for client_id in normalized_order],
    )
    initial_parameters = {
        "semantic_protos": _tensor_mapping_fingerprint(trainer.semantic_protos),
        "device_residuals": _tensor_mapping_fingerprint(trainer.device_residuals),
    }
    rng_before = torch.random.get_rng_state().clone()
    proto_loss, mmd_proto_loss, residual_loss = trainer._compute_server_proto_losses()
    objective = (
        loss_weights["proto"] * proto_loss
        + loss_weights["proto_mmd"] * mmd_proto_loss
        + loss_weights["residual"] * residual_loss
    )
    objective.backward()
    rng_after = torch.random.get_rng_state().clone()

    def gradients(parameters: Mapping[str, nn.Parameter]) -> OrderedDict[str, Any]:
        output: OrderedDict[str, Any] = OrderedDict()
        for key in sorted(parameters):
            gradient = parameters[key].grad
            if gradient is None:
                output[str(key)] = {"present": False}
            else:
                output[str(key)] = {"present": True, **_tensor_record(gradient)}
        return output

    pre_clip_gradients = {
        "semantic_protos": gradients(trainer.semantic_protos),
        "device_residuals": gradients(trainer.device_residuals),
    }
    ordered_parameters = [
        *(
            (f"semantic_protos.{key}", parameter)
            for key, parameter in trainer.semantic_protos.items()
        ),
        *(
            (f"device_residuals.{key}", parameter)
            for key, parameter in trainer.device_residuals.items()
        ),
    ]
    pre_clip_total_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _name, parameter in ordered_parameters],
        max_norm=5.0,
    )
    post_clip_gradients = OrderedDict()
    for name, parameter in ordered_parameters:
        gradient = parameter.grad
        if gradient is None:
            post_clip_gradients[name] = {"present": False}
        else:
            post_clip_gradients[name] = {
                "present": True,
                **_tensor_record(gradient),
            }

    return {
        "client_order": normalized_order,
        "input_fingerprints": {
            "paired_clients": pair_records,
            "paired_multiset_sha256": _sha256_bytes(
                _canonical_bytes(sorted(pair_hashes))
            ),
            "ordered_list_sha256": _sha256_bytes(_canonical_bytes(pair_hashes)),
            "semantic_protos": _tensor_mapping_fingerprint(semantic_protos),
            "initial_parameters": initial_parameters,
        },
        "losses": {
            "proto_loss": _tensor_record(proto_loss),
            "mmd_proto_loss": _tensor_record(mmd_proto_loss),
            "residual_loss": _tensor_record(residual_loss),
            "fixed_backward_objective": _tensor_record(objective),
        },
        "gradients": pre_clip_gradients,
        "gradient_clipping": {
            "max_norm": 5.0,
            "parameter_order": [name for name, _parameter in ordered_parameters],
            "pre_clip_total_norm": _tensor_record(pre_clip_total_norm),
            "post_clip_gradients": post_clip_gradients,
        },
        "rng": {
            "before": _tensor_record(rng_before),
            "after": _tensor_record(rng_after),
            "unchanged": torch.equal(rng_before, rng_after),
        },
        "data_loader_batches": "not_consumed_at_this_stage",
    }


def run_one_step_da_replay(
    trainer_factory: Callable[[], ServerDomainAdaptation],
    clients: Mapping[int, Mapping[str, Any]],
    order: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run one production DA step from a freshly seeded trainer boundary."""

    normalized_order = [int(client_id) for client_id in order]
    if len(normalized_order) != len(set(normalized_order)) or set(
        normalized_order
    ) != set(clients):
        raise ValueError("replay order must be an exact client permutation")
    if type(seed) is not int or seed < 0:
        raise ValueError("replay seed must be a nonnegative integer")
    examples = [int(clients[client_id]["num_examples"]) for client_id in normalized_order]
    if any(value <= 0 for value in examples):
        raise ValueError("client examples must be positive")
    total_examples = sum(examples)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trainer = trainer_factory()
        weights = torch.tensor(
            [value / total_examples for value in examples],
            dtype=torch.float32,
            device=trainer.device,
        )
        rng_before = torch.random.get_rng_state().clone()
        adapted_model, diagnostics = trainer.run_adaptation(
            num_steps=1,
            client_mus=[clients[client_id]["mus"] for client_id in normalized_order],
            client_counts=[
                clients[client_id]["counts"] for client_id in normalized_order
            ],
            client_weights=weights,
            client_ids=normalized_order,
            client_residuals=[
                clients[client_id]["residual"] for client_id in normalized_order
            ],
        )
        rng_after = torch.random.get_rng_state().clone()

        replay = {
            "client_order": normalized_order,
            "weights": _tensor_record(weights),
            "rng": {
                "pre_adaptation": {"torch_cpu": _tensor_record(rng_before)},
                "post_adaptation": {"torch_cpu": _tensor_record(rng_after)},
            },
            "diagnostics": dict(diagnostics),
            "final_parameters": {
                "model": _tensor_mapping_fingerprint(adapted_model.state_dict()),
                "semantic_protos": _tensor_mapping_fingerprint(
                    trainer.semantic_protos
                ),
                "device_residuals": _tensor_mapping_fingerprint(
                    trainer.device_residuals
                ),
            },
        }
        batch_records = getattr(trainer, "_diagnostic_batch_records", None)
        if isinstance(batch_records, Mapping):
            replay["data_loader_batches"] = {
                str(name): list(records)
                for name, records in batch_records.items()
            }
        return replay


class _RecordingLoader:
    """Read-only DataLoader proxy recording exact tensors consumed per iterator."""

    def __init__(self, loader: Any, name: str) -> None:
        self._loader = loader
        self.name = str(name)
        self.dataset = loader.dataset
        self.records: list[dict[str, Any]] = []
        self._iterator_count = 0

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        self._iterator_count += 1
        iterator_id = self._iterator_count
        for batch_index, batch in enumerate(iter(self._loader), start=1):
            tensors = []
            for value in batch:
                if not isinstance(value, torch.Tensor):
                    raise ValueError("DA diagnostic batch contains a non-tensor field")
                tensors.append(_tensor_record(value, include_raw=False))
            record = {
                "loader": self.name,
                "iterator": iterator_id,
                "batch": batch_index,
                "tensors": tensors,
            }
            record["content_sha256"] = _sha256_bytes(_canonical_bytes(record))
            self.records.append(record)
            yield batch


def _make_recording_da_loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray], name: str
) -> _RecordingLoader:
    from federated_dataset import GasSensorWindowDataset
    from torch.utils.data import DataLoader, Subset

    features, cls_labels, phase_labels = arrays
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=np.zeros((len(features), 4), dtype=np.float32),
        classification_labels=cls_labels,
        phase_labels=phase_labels,
        normalize=False,
        mean_std=None,
    )
    sample_count = min(len(dataset), 500)
    indices = np.random.RandomState(42).choice(
        len(dataset), size=sample_count, replace=False
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    return _RecordingLoader(loader, name)


def _b5_hyperparams(args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "USE_DEEP_CORAL": bool(args["da_use_coral"]),
        "USE_MMD_ALIGNMENT": bool(args["da_use_mmd"]),
        "USE_ADVERSARIAL_DOMAIN": bool(args["da_use_adversarial"]),
        "MMD_OBJECTIVE": str(args["da_mmd_objective"]),
        "STAGE_ALIGNMENT": str(args["da_stage_alignment"]),
        "ADV_FEATURE_OBJECTIVE": str(args["da_adv_feature_objective"]),
        "LAMBDA_DEEP_CORAL": float(args["da_lambda_coral"]),
        "LAMBDA_GLOBAL_MMD": float(args["da_lambda_global_mmd"]),
        "LAMBDA_CLASS_MMD": float(args["da_lambda_class_mmd"]),
        "LAMBDA_PROTO_ANCHOR": float(args["da_lambda_proto_anchor"]),
        "LAMBDA_ADV_DOMAIN": float(args["da_lambda_adv"]),
        "LAMBDA_TARGET_CE": float(args["da_lambda_target_ce"]),
        "LAMBDA_PROTO": float(args["da_lambda_proto"]),
        "LAMBDA_CONSISTENCY": float(args["da_lambda_consistency"]),
        "LAMBDA_RES": float(args["da_lambda_residual"]),
        "LAMBDA_PROTO_MMD": float(args["da_lambda_proto_mmd"]),
        "LAMBDA_STAGE_MMD": float(args["da_lambda_stage_mmd"]),
        "USE_ALIGN_REG_LEGACY": bool(args["da_use_align_reg_legacy"]),
        "LAMBDA_ALIGN_REG_LEGACY": float(args["da_lambda_align_reg_legacy"]),
        "USE_CONTRASTIVE_CONSISTENCY": True,
        "USE_PROTO_MMD": float(args["da_lambda_proto_mmd"]) > 0.0,
        "USE_PROTO_DECOUPLING": float(args["da_lambda_residual"]) > 0.0,
        "TARGET_CE_LABEL_SMOOTHING": float(
            args["da_target_ce_label_smoothing"]
        ),
        "TARGET_CE_CLASS_BALANCED": bool(args["da_target_ce_class_balanced"]),
        "SERVER_OPT_LR": float(args["da_server_opt_lr"]),
        "HIDDEN_DIM2": 64,
        "NUM_CLASSES": 4,
        "MAX_VAL_BATCHES": 10,
        "ADV_DOMAIN_LR": 0.001,
        "ADV_CRITIC_ITERS": 3,
        "ADV_GRADIENT_PENALTY": 10.0,
        "ADV_CLASS_CONDITIONAL": True,
        "CORAL_CLASS_CONDITIONAL": bool(args["da_coral_class_conditional"]),
        "DA_LEARN_SEMANTIC_PROTOS": True,
    }


def _make_b5_one_step_trainer_factory(
    *,
    model_state: Mapping[str, torch.Tensor],
    semantic_protos: Mapping[str, torch.Tensor],
    source_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    hyperparams: Mapping[str, Any],
) -> Callable[[], ServerDomainAdaptation]:
    def factory() -> ServerDomainAdaptation:
        from gaps_flower.task import create_model, make_config

        source_loader = _make_recording_da_loader(source_arrays, "source")
        target_loader = _make_recording_da_loader(target_arrays, "target")
        config = make_config(device="cpu", local_epochs=1, batch_size=32)
        model = create_model(config)
        model.load_state_dict(model_state, strict=False)
        trainer = ServerDomainAdaptation(
            model=model,
            val_loader=source_loader,
            calib_loader=target_loader,
            semantic_protos={
                str(key): tensor.detach().cpu().float().clone()
                for key, tensor in semantic_protos.items()
            },
            device=torch.device("cpu"),
            hyperparams=dict(hyperparams),
        )
        trainer._diagnostic_batch_records = {
            "source": source_loader.records,
            "target": target_loader.records,
        }
        return trainer

    return factory


def classify_replays(replays: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(replays) != {"order_21_a", "order_21_b", "order_12"}:
        raise ValueError("replay classification requires the exact three variants")
    first = replays["order_21_a"]
    repeat = replays["order_21_b"]
    reverse = replays["order_12"]
    same_order_losses = first.get("losses") == repeat.get("losses")
    same_order_gradients = first.get("gradients") == repeat.get("gradients")
    cross_order_losses = first.get("losses") == reverse.get("losses")
    cross_order_gradients = first.get("gradients") == reverse.get("gradients")
    multiset_hashes = {
        replay.get("input_fingerprints", {}).get("paired_multiset_sha256")
        for replay in replays.values()
    }
    input_multisets_match = len(multiset_hashes) == 1 and None not in multiset_hashes
    rng_unchanged = all(
        replay.get("rng", {}).get("unchanged") is True for replay in replays.values()
    )
    repeatable = same_order_losses and same_order_gradients
    if (
        repeatable
        and input_multisets_match
        and rng_unchanged
        and (not cross_order_losses or not cross_order_gradients)
    ):
        classification = "reconstructed_initial_proto_loss_order_sensitive"
    elif (
        repeatable
        and input_multisets_match
        and rng_unchanged
        and cross_order_losses
        and cross_order_gradients
    ):
        classification = "order_not_causal_at_proto_loss_stage"
    else:
        classification = "unresolved_fail_closed"
    return {
        "classification": classification,
        "same_order_losses_exact": same_order_losses,
        "same_order_gradients_exact": same_order_gradients,
        "cross_order_losses_exact": cross_order_losses,
        "cross_order_gradients_exact": cross_order_gradients,
        "input_multisets_match": input_multisets_match,
        "rng_unchanged": rng_unchanged,
    }


def _parse_proto_key(key: Any) -> tuple[int, int]:
    text = str(key).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    text = text.replace("_", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid prototype key: {key!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid prototype key: {key!r}") from exc


def _load_metric_json(client: Mapping[str, Any], key: str) -> Any:
    metrics = client["metrics"]
    value = metrics.get(key)
    if type(value) is not str:
        raise ValueError(f"client metric {key} must be a JSON string")
    parsed = json.loads(value)
    finite_json(parsed)
    return parsed


def _build_clients(off_round_two: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    source_clients = _integer_client_mapping(off_round_two["clients"])
    clients: dict[int, dict[str, Any]] = {}
    for client_id in (1, 2):
        source = source_clients[client_id]
        prototypes = _load_metric_json(source, "prototype_json")
        counts_json = _load_metric_json(source, "class_phase_counts_json")
        residual_json = _load_metric_json(source, "device_residual_json")
        if not isinstance(prototypes, Mapping) or not isinstance(counts_json, Mapping):
            raise ValueError("client prototype/count metrics must be JSON objects")
        if not isinstance(residual_json, list) or len(residual_json) != 64:
            raise ValueError("round-2 device residual must contain exactly 64 elements")
        mus: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        counts: OrderedDict[tuple[int, int], int] = OrderedDict()
        for text_key, vector in prototypes.items():
            parsed_key = _parse_proto_key(text_key)
            if not isinstance(vector, list):
                raise ValueError("client prototype vector must be a JSON list")
            tensor = torch.tensor(vector, dtype=torch.float32)
            count = counts_json.get(text_key, counts_json.get(f"{parsed_key[0]},{parsed_key[1]}"))
            if type(count) is not int or count <= 0:
                raise ValueError("client prototype count must be a positive integer")
            mus[parsed_key] = tensor
            counts[parsed_key] = count
        clients[client_id] = {
            "mus": mus,
            "counts": counts,
            "residual": torch.tensor(residual_json, dtype=torch.float32),
            "num_examples": int(source["num_examples"]),
        }
    return clients


def _validate_residual_transition(
    off_r1: Mapping[str, Any],
    on_r1: Mapping[str, Any],
    off_diag: Mapping[str, Any],
    on_diag: Mapping[str, Any],
) -> dict[str, Any]:
    metric_presence: dict[str, dict[str, bool]] = {}
    for mode, fit_res, diagnostics in (
        ("off", off_r1, off_diag),
        ("on", on_r1, on_diag),
    ):
        if type(diagnostics.get("device_residual_count")) is not int or diagnostics.get(
            "device_residual_count"
        ) != 0:
            raise ValueError(f"{mode} round-1 device residual count is not zero")
        loss = diagnostics.get("residual_loss")
        if type(loss) not in {int, float} or isinstance(loss, bool) or loss != 0:
            raise ValueError(f"{mode} round-1 residual loss is not exact zero")
        mode_presence: dict[str, bool] = {}
        for client_id, client in _integer_client_mapping(fit_res["clients"]).items():
            metrics = client.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{mode} round-1 client metrics are missing")
            present = "device_residual_json" in metrics
            mode_presence[str(client_id)] = present
            if present and _load_metric_json(client, "device_residual_json") != []:
                raise ValueError(f"{mode} round-1 FitRes residual is not empty")
        metric_presence[mode] = mode_presence
    return {
        "off_round_1_device_residual_count": 0,
        "on_round_1_device_residual_count": 0,
        "off_round_1_residual_loss": 0.0,
        "on_round_1_residual_loss": 0.0,
        "round_1_fit_res_residuals_empty": True,
        "round_1_fit_res_residual_metric_presence": metric_presence,
        "round_2_is_first_residual_creation": True,
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    value = result.stdout.strip()
    if value != EXPECTED_COMMIT:
        raise ValueError(f"current HEAD differs from frozen commit: {value}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "gaps_flower/domain_adaptation.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("production domain_adaptation.py has uncommitted changes")
    return value


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    source = _require_regular_file(path)
    value = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {source}")
    return value


def _require_binding(formal_report: Mapping[str, Any], current_head: str) -> dict[str, Any]:
    binding = formal_report.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("formal report binding is missing")
    expected = {
        "confirmation_commit": EXPECTED_COMMIT,
        "source_archive_sha256": EXPECTED_SOURCE_SHA256,
        "archive_sha256": EXPECTED_SOURCE_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_SHA256,
        "group_id": "B5",
        "seed": 42,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"formal binding differs for {key}")
    for key in (
        "protocol_manifest_sha256",
        "command_manifest_sha256",
        "regular_members_sha256",
    ):
        value = binding.get(key)
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"formal binding has invalid {key}")
    if current_head != binding["confirmation_commit"]:
        raise ValueError("current HEAD differs from formal binding")
    return dict(binding)


def _effective_config(off_config: Mapping[str, Any], on_config: Mapping[str, Any]) -> dict[str, Any]:
    off_args = off_config.get("args")
    on_args = on_config.get("args")
    if not isinstance(off_args, Mapping) or not isinstance(on_args, Mapping):
        raise ValueError("run_config args are missing")
    fields = (
        "profile",
        "seed",
        "domain_adapt_steps",
        "da_lambda_proto",
        "da_lambda_proto_mmd",
        "da_lambda_residual",
        "use_selective_agg",
        "selective_warmup",
        "proto_ema_alpha",
        "da_device",
    )
    effective = {field: off_args.get(field) for field in fields}
    if effective != {field: on_args.get(field) for field in fields}:
        raise ValueError("OFF/ON effective B5 DA configuration differs")
    required = {
        "profile": "proto_replay",
        "seed": 42,
        "domain_adapt_steps": 100,
        "da_lambda_proto": 0.05,
        "da_lambda_proto_mmd": 0.0,
        "da_lambda_residual": 0.1,
        "use_selective_agg": True,
        "selective_warmup": 3,
        "proto_ema_alpha": 0.8,
        "da_device": "cpu",
    }
    if effective != required:
        raise ValueError(f"effective B5 DA configuration is not frozen: {effective!r}")
    return {
        **effective,
        "USE_PROTO_DECOUPLING": True,
        "USE_PROTO_MMD": False,
        "loss_weights": {"proto": 0.05, "proto_mmd": 0.0, "residual": 0.1},
    }


def _validate_history(history: Mapping[str, Any]) -> dict[str, Any]:
    rounds = history.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("history rounds are missing")
    selected = [row for row in rounds if row.get("round") == 2]
    if len(selected) != 1:
        raise ValueError("history has no unique round 2")
    selective = selected[0].get("selective_agg")
    if not isinstance(selective, Mapping) or selective.get("selective_active") is not False:
        raise ValueError("round-2 selective aggregation must be inactive")
    return {"round": 2, "selective_active": False, "reason": selective.get("reason")}


def _comparison_delta(first: Mapping[str, Any], reverse: Mapping[str, Any]) -> dict[str, Any]:
    first_proto = torch.frombuffer(
        bytes.fromhex(first["losses"]["proto_loss"]["raw_hex"]), dtype=torch.float32
    ).clone()
    reverse_proto = torch.frombuffer(
        bytes.fromhex(reverse["losses"]["proto_loss"]["raw_hex"]), dtype=torch.float32
    ).clone()
    gradient_mismatches: list[str] = []
    for group in ("semantic_protos", "device_residuals"):
        keys = set(first["gradients"][group]) | set(reverse["gradients"][group])
        for key in sorted(keys):
            if first["gradients"][group].get(key) != reverse["gradients"][group].get(key):
                gradient_mismatches.append(f"{group}.{key}")
    return {
        "proto_loss_equal": first["losses"]["proto_loss"]
        == reverse["losses"]["proto_loss"],
        "proto_loss_delta_order_12_minus_order_21": float(
            (reverse_proto - first_proto).item()
        ),
        "gradient_mismatch_count": len(gradient_mismatches),
        "gradient_mismatch_paths": gradient_mismatches,
    }


def _diagnostic_payload(reference_root: Path, repo_root: Path) -> dict[str, Any]:
    off_root = reference_root / "off" / "raw" / "ecs"
    on_root = reference_root / "on" / "raw" / "ecs"
    off_training = off_root / "training"
    on_training = on_root / "training"
    selected_paths = OrderedDict(
        [
            ("formal_report", reference_root / "formal_observer_equivalence_report.json"),
            ("off_common_trace", off_root / "common_trace.jsonl"),
            ("on_common_trace", on_root / "common_trace.jsonl"),
            ("off_round_1_semantic", off_training / "semantic_protos_round_001.json"),
            ("on_round_1_semantic", on_training / "semantic_protos_round_001.json"),
            ("off_round_2_prototype_stats", off_training / "prototype_stats_round_002.json"),
            ("on_round_2_prototype_stats", on_training / "prototype_stats_round_002.json"),
            ("off_round_1_adapted_checkpoint", off_training / "server_round_001_adapted.pth"),
            ("on_round_1_adapted_checkpoint", on_training / "server_round_001_adapted.pth"),
            ("off_round_2_plain_checkpoint", off_training / "server_round_002.pth"),
            ("on_round_2_plain_checkpoint", on_training / "server_round_002.pth"),
            ("off_round_1_da_diagnostics", off_training / "domain_adapt_round_001.json"),
            ("on_round_1_da_diagnostics", on_training / "domain_adapt_round_001.json"),
            ("off_run_config", off_training / "run_config.json"),
            ("on_run_config", on_training / "run_config.json"),
            ("off_history", off_training / "history.json"),
            ("production_domain_adaptation", repo_root / "gaps_flower/domain_adaptation.py"),
        ]
    )
    source_hashes = {
        name: {
            "relative_path": (
                path.relative_to(reference_root).as_posix()
                if reference_root in path.parents
                else path.relative_to(repo_root).as_posix()
            ),
            "raw_sha256": _sha256_file(_require_regular_file(path)),
        }
        for name, path in selected_paths.items()
    }
    current_head = _git_head(repo_root)
    formal_report = _load_json(selected_paths["formal_report"])
    binding = _require_binding(formal_report, current_head)
    off_config = _load_json(selected_paths["off_run_config"])
    on_config = _load_json(selected_paths["on_run_config"])
    effective = _effective_config(off_config, on_config)
    history_evidence = _validate_history(_load_json(selected_paths["off_history"]))

    off_r1 = read_round_fit_res(selected_paths["off_common_trace"], round_idx=1)
    on_r1 = read_round_fit_res(selected_paths["on_common_trace"], round_idx=1)
    off_r2 = read_round_fit_res(selected_paths["off_common_trace"], round_idx=2)
    on_r2 = read_round_fit_res(selected_paths["on_common_trace"], round_idx=2)
    fit_res_cross_validation = cross_validate_fit_res(off_r2, on_r2)
    residual_transition = _validate_residual_transition(
        off_r1,
        on_r1,
        _load_json(selected_paths["off_round_1_da_diagnostics"]),
        _load_json(selected_paths["on_round_1_da_diagnostics"]),
    )

    clients = _build_clients(off_r2)
    if [clients[client_id]["num_examples"] for client_id in (1, 2)] != [2360, 2360]:
        raise ValueError("round-2 client examples must be exactly 2360/2360")
    production_weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    expected_weights = torch.tensor(
        [2360 / 4720, 2360 / 4720], dtype=torch.float32
    )
    if not torch.equal(production_weights, expected_weights):
        raise ValueError("round-2 float32 base weights are not exact [0.5,0.5]")

    off_round_one = _load_json(selected_paths["off_round_1_semantic"])
    on_round_one = _load_json(selected_paths["on_round_1_semantic"])
    if off_round_one != on_round_one:
        raise ValueError("OFF/ON round-1 semantic JSON differs")
    off_round_two_stats = _load_json(selected_paths["off_round_2_prototype_stats"])
    on_round_two_stats = _load_json(selected_paths["on_round_2_prototype_stats"])
    if off_round_two_stats.get("global_prototypes") != on_round_two_stats.get(
        "global_prototypes"
    ):
        raise ValueError("OFF/ON round-2 global prototypes differ")

    off_r1_checkpoint = _load_checkpoint(selected_paths["off_round_1_adapted_checkpoint"])
    on_r1_checkpoint = _load_checkpoint(selected_paths["on_round_1_adapted_checkpoint"])
    off_checkpoint_semantic = off_r1_checkpoint.get("semantic_protos")
    on_checkpoint_semantic = on_r1_checkpoint.get("semantic_protos")
    if not isinstance(off_checkpoint_semantic, Mapping) or not isinstance(
        on_checkpoint_semantic, Mapping
    ):
        raise ValueError("round-1 adapted checkpoint lacks semantic prototypes")
    off_checkpoint_fingerprint = _tensor_mapping_fingerprint(off_checkpoint_semantic)
    on_checkpoint_fingerprint = _tensor_mapping_fingerprint(on_checkpoint_semantic)
    if off_checkpoint_fingerprint != on_checkpoint_fingerprint:
        raise ValueError("OFF/ON round-1 checkpoint semantic prototypes differ")
    semantic = reconstruct_semantic_protos(
        off_round_one,
        off_round_two_stats,
        config_alpha=effective["proto_ema_alpha"],
        checkpoint_semantic_protos=off_checkpoint_semantic,
    )

    off_plain = _load_checkpoint(selected_paths["off_round_2_plain_checkpoint"])
    on_plain = _load_checkpoint(selected_paths["on_round_2_plain_checkpoint"])
    off_model_state = off_plain.get("model_state")
    on_model_state = on_plain.get("model_state")
    if not isinstance(off_model_state, Mapping) or not isinstance(on_model_state, Mapping):
        raise ValueError("round-2 plain checkpoint lacks model_state")
    off_model_fingerprint = _tensor_mapping_fingerprint(off_model_state)
    on_model_fingerprint = _tensor_mapping_fingerprint(on_model_state)
    if off_model_fingerprint != on_model_fingerprint:
        raise ValueError("OFF/ON round-2 post-aggregate model differs")

    from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays

    data_root = (
        repo_root
        / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    )
    source_dirs = (data_root / "client_1", data_root / "client_2")
    target_dirs = (data_root / "client_5",)
    source_arrays = load_domain_adaptation_arrays(source_dirs, strict=True)
    target_arrays = load_domain_adaptation_arrays(target_dirs, strict=True)
    off_args = off_config["args"]
    if not isinstance(off_args, Mapping):
        raise ValueError("OFF run_config args are missing")
    full_da_factory = _make_b5_one_step_trainer_factory(
        model_state=off_model_state,
        semantic_protos=semantic,
        source_arrays=source_arrays,
        target_arrays=target_arrays,
        hyperparams=_b5_hyperparams(off_args),
    )
    full_da_replays = {
        "order_21_a": run_one_step_da_replay(
            full_da_factory, clients, [2, 1], seed=42
        ),
        "order_21_b": run_one_step_da_replay(
            full_da_factory, clients, [2, 1], seed=42
        ),
        "order_12": run_one_step_da_replay(
            full_da_factory, clients, [1, 2], seed=42
        ),
    }
    full_first = full_da_replays["order_21_a"]
    full_repeat = full_da_replays["order_21_b"]
    full_reverse = full_da_replays["order_12"]
    full_da_classification = {
        "same_order_replay_exact": full_first == full_repeat,
        "same_order_batches_exact": full_first.get("data_loader_batches")
        == full_repeat.get("data_loader_batches"),
        "cross_order_batches_exact": full_first.get("data_loader_batches")
        == full_reverse.get("data_loader_batches"),
        "cross_order_model_exact": full_first["final_parameters"]["model"]
        == full_reverse["final_parameters"]["model"],
        "cross_order_semantic_protos_exact": full_first["final_parameters"][
            "semantic_protos"
        ]
        == full_reverse["final_parameters"]["semantic_protos"],
        "cross_order_device_residual_values_exact": sorted(
            full_first["final_parameters"]["device_residuals"]["tensors"].items()
        )
        == sorted(
            full_reverse["final_parameters"]["device_residuals"]["tensors"].items()
        ),
        "cross_order_diagnostics_exact": full_first["diagnostics"]
        == full_reverse["diagnostics"],
    }

    loss_weights = effective["loss_weights"]
    replays = {
        "order_21_a": run_proto_loss_replay(
            semantic, clients, [2, 1], loss_weights=loss_weights
        ),
        "order_21_b": run_proto_loss_replay(
            semantic, clients, [2, 1], loss_weights=loss_weights
        ),
        "order_12": run_proto_loss_replay(
            semantic, clients, [1, 2], loss_weights=loss_weights
        ),
    }
    classification = classify_replays(replays)
    return {
        "schema_version": "iotj.b5_fixed_state_order_replay.v1",
        "diagnostic_only": True,
        "classification": classification["classification"],
        "classification_evidence": classification,
        "binding": {
            **binding,
            "current_head": current_head,
            "production_domain_adaptation_sha256": source_hashes[
                "production_domain_adaptation"
            ]["raw_sha256"],
        },
        "source_artifacts": source_hashes,
        "effective_b5_da_config": effective,
        "history_weight_evidence": history_evidence,
        "production_weights": {
            "num_examples_by_client": {"1": 2360, "2": 2360},
            "dtype": "torch.float32",
            "client_id_order_12": [0.5, 0.5],
            "client_id_order_21": [0.5, 0.5],
            "selective_active": False,
        },
        "residual_state_transition": residual_transition,
        "fit_res_cross_validation": fit_res_cross_validation,
        "reconstructed_pre_da_state": {
            "round_2_plain_model": off_model_fingerprint,
            "round_1_after_da_semantic": off_checkpoint_fingerprint,
            "round_2_pre_da_semantic": _tensor_mapping_fingerprint(semantic),
            "alpha": effective["proto_ema_alpha"],
            "alpha_bound_to": [
                "off/round_1_semantic.proto_ema_alpha",
                "off/run_config.args.proto_ema_alpha",
            ],
        },
        "replays": replays,
        "same_order_repeatability": {
            "losses_exact": classification["same_order_losses_exact"],
            "gradients_exact": classification["same_order_gradients_exact"],
        },
        "cross_order_delta": _comparison_delta(
            replays["order_21_a"], replays["order_12"]
        ),
        "full_b5_da_one_step": {
            "definition": (
                "production ServerDomainAdaptation.run_adaptation(num_steps=1) "
                "from the preserved round-2 plain model and reconstructed "
                "round-2 semantic prototypes, with exact local C1+C2 source "
                "and C5 calibration arrays"
            ),
            "data_root": data_root.relative_to(repo_root).as_posix(),
            "source_rows": int(len(source_arrays[0])),
            "target_rows": int(len(target_arrays[0])),
            "classification": full_da_classification,
            "replays": full_da_replays,
        },
        "formal_topology_off_repeatability": "unproven_by_this_loss_stage_replay",
        "freeze_record_created": False,
        "formal_25_round_runs_started": False,
    }


def run_diagnostic(reference_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    reference, output = validate_roots(reference_root, output_root)
    payload, manifest_before, manifest_after = run_with_reference_guard(
        reference, lambda: _diagnostic_payload(reference, repo_root)
    )
    report = {
        **payload,
        "reference_root_manifest": {
            "before_sha256": manifest_before["manifest_sha256"],
            "after_sha256": manifest_after["manifest_sha256"],
            "unchanged": manifest_before == manifest_after,
            "file_count": manifest_before["file_count"],
            "definition": manifest_before["definition"],
        },
    }
    finite_json(report)
    output.mkdir()
    report_path = output / REPORT_NAME
    with report_path.open("xb") as handle:
        handle.write(_canonical_bytes(report) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    smoke_root = repo_root / "results/iotj_main_confirmation_observability_20260715/smoke"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=smoke_root / "b5")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=smoke_root / "b5_fixed_state_order_replay_7ec77e3",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_diagnostic(args.reference_root, args.output_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
