from __future__ import annotations

import numpy as np
import pytest
import csv
import json
from pathlib import Path


def test_serialized_ridge_predicts_and_clips() -> None:
    from gaps_deploy.c5_h8_runtime import SerializedRidge

    head = SerializedRidge.from_json(
        {"feature_names": ["x"], "mean": [2.0], "scale": [2.0], "coef": [10.0, 4.0], "clip_min": 5.0, "clip_max": 12.0}
    )

    assert head.predict({"x": 3.0}) == pytest.approx(12.0)
    with pytest.raises(ValueError, match="not finite"):
        head.predict({"x": float("nan")})
    with pytest.raises(ValueError, match="missing"):
        head.predict({})


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
    contract = root / "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json"
    runtime = C5H8Runtime.from_runtime_contract(contract)
    logits, predicted = runtime.classify(np.zeros((1, 100, 8), dtype=np.float32))

    assert logits.shape == (1, 4)
    assert predicted.shape == (1,)
    assert int(predicted[0]) in (0, 1, 2, 3)


def test_fixed_h8_route_uses_predicted_class_for_all_heads() -> None:
    from gaps_deploy.c5_h8_runtime import FixedH8Policy

    ridge = {"feature_names": ["x"], "mean": [0], "scale": [1], "coef": [0, 1], "clip_min": -99, "clip_max": 99}
    mlp = {"feature_names": ["x"], "mean": [0], "scale": [1], "coefs": [[[1]]], "intercepts": [[0]], "activation": "relu", "out_activation": "identity", "clip_min": -99, "clip_max": 99}
    policy = FixedH8Policy.from_json({"source_heads": {"ridge_per_gas": [{**ridge, "class_id": i} for i in range(4)], "mlp_per_gas": [{**mlp, "class_id": i} for i in range(4)], "shared_mlp": {**mlp, "gas": "shared"}}, "models": [{**ridge, "class_id": i, "client": "C5"} for i in range(4)]})

    assert policy.predict({"x": 2.0}, predicted_class=3) == pytest.approx(2.0)
    assert set(policy.predict_components({"x": 2.0}, predicted_class=3)) == {
        "H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm",
        "H3_source_shared_mlp_ppm", "target_ridge_plus_source_preds_ppm",
    }


def test_h23_policy_uses_predicted_class_and_frozen_blend() -> None:
    from gaps_deploy.c5_h8_runtime import H23Policy

    mlp = lambda class_id: {
        "class_id": class_id, "feature_names": ["x"], "mean": [0], "scale": [1],
        "coefs": [[[0]]], "intercepts": [[class_id]], "activation": "relu",
        "out_activation": "identity", "clip_min": -99, "clip_max": 99,
    }
    ridge = lambda class_id: {
        "class_id": class_id, "feature_names": ["x"], "mean": [0], "scale": [1],
        "coef": [10 + class_id, 0], "clip_min": -99, "clip_max": 99,
    }
    policy = H23Policy.from_json({
        "anchor": "per_gas_mlp", "secondary": "regfeat_ridge", "target_client": "C5", "blend_weight": 0.5,
        "mlp_models": [mlp(i) for i in range(4)], "ridge_models": [ridge(i) for i in range(4)],
    })

    assert policy.predict_components({"x": 1.0}, 3) == {
        "h23_anchor_ppm": pytest.approx(3.0),
        "h23_weak_ridge_ppm": pytest.approx(13.0),
        "h23_plus_ppm": pytest.approx(8.0),
    }


def test_first_formal_row_replays_b5_h23_and_r4() -> None:
    from gaps_deploy.c5_h8_runtime import C5H8Runtime

    root = Path(__file__).resolve().parents[1]
    contract_path = root / "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    windows = np.load(contract["inputs"]["features"]["path"], mmap_mode="r")
    metadata = json.loads(Path(contract["inputs"]["metadata"]["path"]).read_text(encoding="utf-8"))
    phases = np.load(contract["inputs"]["phase_labels"]["path"], mmap_mode="r")
    with Path(contract["references"]["HC95"]["path"]).open(encoding="utf-8", newline="") as handle:
        reference = next(csv.DictReader(handle))

    row = C5H8Runtime.from_runtime_contract(contract_path).infer_experts(windows[:1], metadata[:1], phases[:1])[0]

    assert row["pred_class"] == int(reference["pred_class"])
    assert row["h23_plus_ppm"] == pytest.approx(float(reference["h23_plus_ppm"]), abs=2e-3)
    assert row["target_ridge_plus_source_preds_ppm"] == pytest.approx(
        float(reference["target_ridge_plus_source_preds_ppm"]), abs=2e-3
    )
