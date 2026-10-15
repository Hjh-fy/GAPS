"""Logical and serialized-size accounting for legacy Flower messages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping

from flwr.common import FitIns, FitRes, Scalar, parameters_to_ndarrays, serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


UPLINK_PROTOTYPE_KEYS = {"prototype_json"}
UPLINK_PROTOTYPE_VAR_KEYS = {"prototype_var_json"}
UPLINK_STATISTIC_KEYS = {
    "class_phase_counts_json",
    "global_feature_json",
    "device_residual_json",
}
DOWNLINK_SEMANTIC_KEYS = {"semantic_protos_json"}


@dataclass(frozen=True)
class MessageAudit:
    """Logical payload sizes and canonical serialized application-message data."""

    logical: dict[str, int]
    application_message_bytes: int
    application_message_sha256: str
    flower_serialize_ns: int


def canonical_fit_ins_bytes(ins: FitIns) -> bytes:
    """Serialize a FitIns inside Flower's legacy ServerMessage wrapper."""

    message = ServerMessage(fit_ins=serde.fit_ins_to_proto(ins))
    return message.SerializeToString(deterministic=True)


def canonical_fit_res_bytes(res: FitRes) -> bytes:
    """Serialize a FitRes inside Flower's legacy ClientMessage wrapper."""

    message = ClientMessage(fit_res=serde.fit_res_to_proto(res))
    return message.SerializeToString(deterministic=True)


def _scalar_value_bytes(value: Scalar) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 8
    if isinstance(value, float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    raise TypeError(f"unsupported Flower scalar type: {type(value).__name__}")


def _group_value_bytes(values: Mapping[str, Scalar], keys: set[str]) -> int:
    return sum(
        _scalar_value_bytes(value) for key, value in values.items() if key in keys
    )


def _other_value_bytes(values: Mapping[str, Scalar], excluded: set[str]) -> int:
    return sum(
        _scalar_value_bytes(value)
        for key, value in values.items()
        if key not in excluded
    )


def _model_value_bytes(ins_or_res: FitIns | FitRes) -> int:
    return sum(
        array.nbytes for array in parameters_to_ndarrays(ins_or_res.parameters)
    )


def _parameter_blob_bytes(ins_or_res: FitIns | FitRes) -> int:
    return sum(len(blob) for blob in ins_or_res.parameters.tensors)


def _message_audit(
    logical: dict[str, int], application_bytes: bytes, elapsed_ns: int
) -> MessageAudit:
    return MessageAudit(
        logical=logical,
        application_message_bytes=len(application_bytes),
        application_message_sha256=hashlib.sha256(application_bytes).hexdigest(),
        flower_serialize_ns=elapsed_ns,
    )


def audit_fit_ins(ins: FitIns) -> MessageAudit:
    """Audit a server-to-client FitIns without modifying it."""

    model_value_bytes = _model_value_bytes(ins)
    parameter_blob_bytes = _parameter_blob_bytes(ins)
    semantic_bytes = _group_value_bytes(ins.config, DOWNLINK_SEMANTIC_KEYS)
    other_bytes = _other_value_bytes(ins.config, DOWNLINK_SEMANTIC_KEYS)
    logical = {
        "logical_downlink_model_value_bytes": model_value_bytes,
        "logical_downlink_parameter_blob_bytes": parameter_blob_bytes,
        "logical_downlink_semantic_proto_utf8_bytes": semantic_bytes,
        "logical_downlink_other_config_value_bytes": other_bytes,
        "logical_downlink_total_bytes": (
            parameter_blob_bytes + semantic_bytes + other_bytes
        ),
    }

    started_ns = perf_counter_ns()
    application_bytes = canonical_fit_ins_bytes(ins)
    elapsed_ns = perf_counter_ns() - started_ns
    return _message_audit(logical, application_bytes, elapsed_ns)


def audit_fit_res(res: FitRes) -> MessageAudit:
    """Audit a client-to-server FitRes without modifying it."""

    metrics = res.metrics if res.metrics is not None else {}
    model_value_bytes = _model_value_bytes(res)
    parameter_blob_bytes = _parameter_blob_bytes(res)
    prototype_bytes = _group_value_bytes(metrics, UPLINK_PROTOTYPE_KEYS)
    prototype_var_bytes = _group_value_bytes(metrics, UPLINK_PROTOTYPE_VAR_KEYS)
    statistic_bytes = _group_value_bytes(metrics, UPLINK_STATISTIC_KEYS)
    semantic_keys = (
        UPLINK_PROTOTYPE_KEYS | UPLINK_PROTOTYPE_VAR_KEYS | UPLINK_STATISTIC_KEYS
    )
    diagnostic_bytes = _other_value_bytes(metrics, semantic_keys)
    logical = {
        "logical_uplink_model_value_bytes": model_value_bytes,
        "logical_uplink_parameter_blob_bytes": parameter_blob_bytes,
        "logical_uplink_prototype_utf8_bytes": prototype_bytes,
        "logical_uplink_prototype_var_utf8_bytes": prototype_var_bytes,
        "logical_uplink_statistics_utf8_bytes": statistic_bytes,
        "logical_uplink_diagnostic_value_bytes": diagnostic_bytes,
        "logical_uplink_total_bytes": (
            parameter_blob_bytes
            + prototype_bytes
            + prototype_var_bytes
            + statistic_bytes
            + diagnostic_bytes
        ),
    }

    started_ns = perf_counter_ns()
    application_bytes = canonical_fit_res_bytes(res)
    elapsed_ns = perf_counter_ns() - started_ns
    return _message_audit(logical, application_bytes, elapsed_ns)
