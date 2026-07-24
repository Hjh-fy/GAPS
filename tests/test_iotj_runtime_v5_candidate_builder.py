from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_iotj_runtime_v5_candidate import (
    candidate_identity,
    load_ridge_models,
    require_calibration_lock,
    validate_client_provenance,
    write_calibration_lock,
)


def _model(feature_dimension: int) -> dict[str, object]:
    return {
        "alpha": 0.01,
        "feature_names": [f"f{i}" for i in range(feature_dimension)],
        "mean": [0.0] * feature_dimension,
        "scale": [1.0] * feature_dimension,
        "coef": [2.0] + [0.0] * feature_dimension,
        "clip_min": 0.0,
        "clip_max": 10.0,
    }


def test_load_ridge_models_requires_four_heads_and_exact_dimension(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"models": {str(i): _model(104) for i in range(4)}}),
        encoding="utf-8",
    )
    models = load_ridge_models(path, expected_dimension=104)
    assert set(models) == {0, 1, 2, 3}
    assert models[1].predict([{"feature_dict": {f"f{i}": 3.0 for i in range(104)}}]).tolist() == [2.0]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"].pop("3")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="four gas heads"):
        load_ridge_models(path, expected_dimension=104)


def test_load_ridge_models_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    payload = {"models": {str(i): _model(104) for i in range(4)}}
    payload["models"]["2"]["coef"][1] = np.nan
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        load_ridge_models(path, expected_dimension=104)


def test_calibration_lock_detects_target_manifest_drift(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"models": {str(i): _model(105) for i in range(4)}}), encoding="utf-8")
    lock = tmp_path / "calibration_lock.json"
    h1 = tmp_path / "h1.json"
    h1.write_text("{}", encoding="utf-8")
    write_calibration_lock(lock, {"target_ridge": target, "federated_h1": h1}, {"calibration_validation_RMSE": 15.0})
    require_calibration_lock(lock, {"target_ridge": target, "federated_h1": h1})

    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash differs"):
        require_calibration_lock(lock, {"target_ridge": target, "federated_h1": h1})


def test_calibration_lock_detects_h1_drift(tmp_path: Path) -> None:
    target = tmp_path / "target.json"; target.write_text("{}", encoding="utf-8")
    h1 = tmp_path / "h1.json"; h1.write_text("{}", encoding="utf-8")
    lock = tmp_path / "lock.json"
    assets = {"target_ridge": target, "federated_h1": h1}
    write_calibration_lock(lock, assets, {})
    h1.write_text("{\"changed\":true}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="federated_h1 hash differs"):
        require_calibration_lock(lock, assets)


def test_calibration_lock_rejects_extra_fields_or_opened_test(tmp_path: Path) -> None:
    target = tmp_path / "target.json"; target.write_text("{}", encoding="utf-8")
    lock = tmp_path / "lock.json"
    assets = {"target_ridge": target}
    write_calibration_lock(lock, assets, {})
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema differs"):
        require_calibration_lock(lock, assets)
    payload.pop("unexpected")
    payload["test_opened"] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="after test opening"):
        require_calibration_lock(lock, assets)


def test_client_provenance_requires_explicit_false_and_own_directory() -> None:
    valid = {"client_id": "C1", "allowed_dataset_directory": "/data/client_1", "other_source_client_opened": False}
    validate_client_provenance(valid, 1)
    with pytest.raises(RuntimeError, match="other source"):
        validate_client_provenance({k: v for k, v in valid.items() if k != "other_source_client_opened"}, 1)
    with pytest.raises(RuntimeError, match="dataset directory"):
        validate_client_provenance({**valid, "allowed_dataset_directory": "/data/client_2"}, 1)
