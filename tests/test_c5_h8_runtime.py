from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


def test_serialized_ridge_predicts_and_clips() -> None:
    from gaps_deploy.c5_h8_runtime import SerializedRidge

    head = SerializedRidge.from_json(
        {"feature_names": ["x"], "mean": [2.0], "scale": [2.0], "coef": [10.0, 4.0], "clip_min": 5.0, "clip_max": 12.0}
    )

    assert head.predict({"x": 3.0}) == pytest.approx(12.0)
    assert head.predict({"x": float("nan")}) == pytest.approx(10.0)


def test_serialized_mlp_applies_relu_hidden_layer_and_clip() -> None:
    from gaps_deploy.c5_h8_runtime import SerializedMLP

    head = SerializedMLP.from_json(
        {
            "feature_names": ["x"], "mean": [0.0], "scale": [1.0],
            "coefs": [[[2.0]], [[3.0]]], "intercepts": [[-1.0], [1.0]],
            "activation": "relu", "out_activation": "identity", "clip_min": 0.0, "clip_max": 5.0,
        }
    )

    assert head.predict({"x": 1.0}) == pytest.approx(4.0)
    assert head.predict({"x": -1.0}) == pytest.approx(1.0)


def test_serialized_shared_mlp_appends_predicted_route_one_hot() -> None:
    from gaps_deploy.c5_h8_runtime import SerializedMLP

    head = SerializedMLP.from_json(
        {
            "feature_names": ["x"], "mean": [0.0] * 5, "scale": [1.0] * 5,
            "coefs": [[[1.0], [0.0], [2.0], [0.0], [0.0]]], "intercepts": [[0.0]],
            "activation": "relu", "out_activation": "identity", "clip_min": -10.0, "clip_max": 10.0,
        }
    )

    assert head.predict({"x": 3.0, "route_class": 1}) == pytest.approx(5.0)


def test_runtime_contract_strict_loads_frozen_b5_classifier() -> None:
    from gaps_deploy.c5_h8_runtime import C5H8Runtime

    root = Path(__file__).resolve().parents[1]
    contract = root / "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v2/runtime_contract.json"
    runtime = C5H8Runtime.from_runtime_contract(contract)
    logits, predicted = runtime.classify(np.zeros((1, 100, 8), dtype=np.float32))

    assert logits.shape == (1, 4)
    assert predicted.shape == (1,)
    assert int(predicted[0]) in (0, 1, 2, 3)
