from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_iotj_h1_federated_ridge_equivalence import (
    EXPECTED_FROZEN_HASHES,
    LocalFeatureMoments,
    client_feature_moments,
    client_normal_equations,
    fit_federated_h1,
    fit_pooled_h1,
    frozen_hashes,
    run,
    server_aggregate_scaler,
    server_reconstruct_ridge,
    server_validation_rmse,
)
from run_regression_head_ablation import fit_ridge


FEATURES = ("a", "b")


def rows(client: str, offset: float = 0.0):
    return [
        {
            "client": client,
            "split": "train",
            "sample_index": index,
            "true_class": 0,
            "true_ppm": 3.0 + 2.0 * value,
            "feature_dict": {"a": value + offset, "b": value * value},
        }
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0))
    ]


def test_server_apis_reject_raw_rows_and_raw_x_y():
    raw = rows("C1")
    with pytest.raises(TypeError):
        server_aggregate_scaler(raw)
    with pytest.raises(TypeError):
        server_reconstruct_ridge(raw, object(), FEATURES, 1.0)
    with pytest.raises(TypeError):
        server_validation_rmse(raw)
    with pytest.raises(TypeError):
        server_aggregate_scaler([np.ones((2, 2))])


def test_client_statistics_have_independent_provenance_and_isolation():
    c1 = rows("C1")
    c2 = rows("C2", 10.0)
    all_rows = [*c1, *c2]
    s1 = client_feature_moments("C1", 0, "train", all_rows, FEATURES)
    s2 = client_feature_moments("C2", 0, "train", all_rows, FEATURES)
    assert s1.client_id != s2.client_id
    assert s1.provenance_sha256 != s2.provenance_sha256

    modified = [dict(row) for row in all_rows]
    modified[0] = {
        **modified[0],
        "feature_dict": {**modified[0]["feature_dict"], "a": 99.0},
    }
    s1_changed = client_feature_moments("C1", 0, "train", modified, FEATURES)
    s2_unchanged = client_feature_moments("C2", 0, "train", modified, FEATURES)
    assert not np.array_equal(s1.sum_x, s1_changed.sum_x)
    np.testing.assert_array_equal(s2.sum_x, s2_unchanged.sum_x)
    assert s2.provenance_sha256 == s2_unchanged.provenance_sha256


def test_server_reconstructs_central_ridge_from_statistics_only():
    all_rows = [*rows("C1"), *rows("C2", 0.5)]
    moments = [
        client_feature_moments(client, 0, "train", all_rows, FEATURES)
        for client in ("C1", "C2")
    ]
    scaler = server_aggregate_scaler(moments)
    equations = [
        client_normal_equations(client, 0, "train", all_rows, FEATURES, scaler)
        for client in ("C1", "C2")
    ]
    reconstructed = server_reconstruct_ridge(equations, scaler, FEATURES, 0.1)
    centralized = fit_ridge(all_rows, FEATURES, 0.1)
    np.testing.assert_allclose(reconstructed.mean, centralized.mean, atol=1e-12)
    np.testing.assert_allclose(reconstructed.scale, centralized.scale, atol=1e-12)
    np.testing.assert_allclose(reconstructed.coef, centralized.coef, atol=1e-10)
    np.testing.assert_allclose(
        reconstructed.predict(all_rows), centralized.predict(all_rows), atol=1e-10
    )


def test_duplicate_client_provenance_is_rejected():
    all_rows = [*rows("C1"), *rows("C2")]
    c1 = client_feature_moments("C1", 0, "train", all_rows, FEATURES)
    with pytest.raises(ValueError):
        server_aggregate_scaler([c1, replace(c1)])


def test_training_apis_have_no_target_or_test_inputs():
    for function in (fit_pooled_h1, fit_federated_h1):
        parameters = set(inspect.signature(function).parameters)
        assert not {
            "test",
            "test_rows",
            "c5_calibration",
            "target_calibration",
            "source_test",
        }.intersection(parameters)


def test_formal_pipeline_freezes_gate_before_opening_test():
    source = inspect.getsource(run)
    freeze = source.index('write_json(output / "equivalence_decision.json"')
    open_test = source.index('prepare_split_rows(\n        data_root, "test"')
    assert freeze < open_test
    assert '"C5_calibration_used_for_source_H1_training": False' in source
    assert '"C1_C2_source_test_used_for_H1_train_or_select": False' in source


def test_six_frozen_runtime_qc_hashes_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert frozen_hashes(root) == EXPECTED_FROZEN_HASHES
