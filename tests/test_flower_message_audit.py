import hashlib

import numpy as np
import pytest
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    serde,
)
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage

from gaps_flower.flower_message_audit import (
    audit_fit_ins,
    audit_fit_res,
    canonical_fit_ins_bytes,
    canonical_fit_res_bytes,
)


def _fixed_arrays() -> list[np.ndarray]:
    return [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([3], dtype=np.int64),
    ]


def test_fit_res_audit_matches_full_client_message_and_does_not_mutate() -> None:
    arrays = _fixed_arrays()
    metrics = {
        "client_id": 1,
        "prototype_json": '{"0,0":[1.0,2.0]}',
        "prototype_var_json": '{"0,0":[0.1,0.2]}',
        "class_phase_counts_json": '{"0,0":7}',
        "global_feature_json": "[0.5,0.6]",
        "device_residual_json": "[0.01]",
        "fit_seconds": 1.25,
    }
    fit_res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(arrays),
        num_examples=7,
        metrics=dict(metrics),
    )
    parameters_id = id(fit_res.parameters)
    metrics_id = id(fit_res.metrics)
    before = canonical_fit_res_bytes(fit_res)

    audit = audit_fit_res(fit_res)

    after = canonical_fit_res_bytes(fit_res)
    nested = serde.fit_res_to_proto(fit_res).SerializeToString(deterministic=True)
    expected = ClientMessage(fit_res=serde.fit_res_to_proto(fit_res)).SerializeToString(
        deterministic=True
    )

    assert before == after == expected
    assert len(expected) > len(nested)
    assert audit.application_message_bytes == len(expected)
    assert audit.application_message_sha256 == hashlib.sha256(expected).hexdigest()
    assert audit.logical == {
        "logical_uplink_model_value_bytes": 16,
        "logical_uplink_parameter_blob_bytes": sum(
            len(blob) for blob in fit_res.parameters.tensors
        ),
        "logical_uplink_prototype_utf8_bytes": len(
            metrics["prototype_json"].encode("utf-8")
        ),
        "logical_uplink_prototype_var_utf8_bytes": len(
            metrics["prototype_var_json"].encode("utf-8")
        ),
        "logical_uplink_statistics_utf8_bytes": sum(
            len(metrics[key].encode("utf-8"))
            for key in (
                "class_phase_counts_json",
                "global_feature_json",
                "device_residual_json",
            )
        ),
        "logical_uplink_diagnostic_value_bytes": 16,
        "logical_uplink_total_bytes": (
            sum(len(blob) for blob in fit_res.parameters.tensors)
            + len(metrics["prototype_json"].encode("utf-8"))
            + len(metrics["prototype_var_json"].encode("utf-8"))
            + sum(
                len(metrics[key].encode("utf-8"))
                for key in (
                    "class_phase_counts_json",
                    "global_feature_json",
                    "device_residual_json",
                )
            )
            + 16
        ),
    }
    assert audit.flower_serialize_ns >= 0
    assert id(fit_res.parameters) == parameters_id
    assert id(fit_res.metrics) == metrics_id
    assert fit_res.metrics == metrics


def test_fit_ins_audit_excludes_semantic_json_from_other_config_bytes() -> None:
    arrays = _fixed_arrays()
    config = {
        "semantic_protos_json": '{"0,0":[1.0,2.0],"label":"水泵"}',
        "enabled": True,
        "server_round": 3,
        "temperature": 0.5,
        "note": "诊断✓",
        "payload": b"\x00\x01\x02",
    }
    fit_ins = FitIns(
        parameters=ndarrays_to_parameters(arrays),
        config=dict(config),
    )
    parameters_id = id(fit_ins.parameters)
    config_id = id(fit_ins.config)
    before = canonical_fit_ins_bytes(fit_ins)

    audit = audit_fit_ins(fit_ins)

    after = canonical_fit_ins_bytes(fit_ins)
    nested = serde.fit_ins_to_proto(fit_ins).SerializeToString(deterministic=True)
    expected = ServerMessage(fit_ins=serde.fit_ins_to_proto(fit_ins)).SerializeToString(
        deterministic=True
    )
    semantic_bytes = len(config["semantic_protos_json"].encode("utf-8"))
    other_bytes = 1 + 8 + 8 + len(config["note"].encode("utf-8")) + 3
    parameter_blob_bytes = sum(len(blob) for blob in fit_ins.parameters.tensors)

    assert before == after == expected
    assert len(expected) > len(nested)
    assert audit.application_message_bytes == len(expected)
    assert audit.application_message_sha256 == hashlib.sha256(expected).hexdigest()
    assert audit.logical == {
        "logical_downlink_model_value_bytes": 16,
        "logical_downlink_parameter_blob_bytes": parameter_blob_bytes,
        "logical_downlink_semantic_proto_utf8_bytes": semantic_bytes,
        "logical_downlink_other_config_value_bytes": other_bytes,
        "logical_downlink_total_bytes": (
            parameter_blob_bytes + semantic_bytes + other_bytes
        ),
    }
    assert audit.flower_serialize_ns >= 0
    assert id(fit_ins.parameters) == parameters_id
    assert id(fit_ins.config) == config_id
    assert fit_ins.config == config


@pytest.mark.parametrize("invalid", [None, [1], {"nested": "mapping"}, object()])
def test_invalid_scalar_type_raises_type_error(invalid: object) -> None:
    fit_ins = FitIns(
        parameters=ndarrays_to_parameters(_fixed_arrays()),
        config={"invalid": invalid},
    )

    with pytest.raises(TypeError, match="scalar"):
        audit_fit_ins(fit_ins)


def test_real_serde_serialization_exception_leaves_original_unchanged() -> None:
    out_of_range_sint64 = -(1 << 63) - 1
    fit_ins = FitIns(
        parameters=ndarrays_to_parameters(_fixed_arrays()),
        config={"server_round": out_of_range_sint64},
    )
    parameters_id = id(fit_ins.parameters)
    config_id = id(fit_ins.config)
    tensors_before = tuple(fit_ins.parameters.tensors)
    tensor_type_before = fit_ins.parameters.tensor_type
    config_before = dict(fit_ins.config)

    with pytest.raises(ValueError, match="out of range"):
        audit_fit_ins(fit_ins)

    assert id(fit_ins.parameters) == parameters_id
    assert id(fit_ins.config) == config_id
    assert tuple(fit_ins.parameters.tensors) == tensors_before
    assert fit_ins.parameters.tensor_type == tensor_type_before
    assert fit_ins.config == config_before
