from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gaps_deploy.c5_federated_source_ridge_bundle import (
    FederatedSourceRidgeBundleError,
    load_federated_source_ridge_bundle,
)
from gaps_deploy.c5_federated_source_ridge_runtime import (
    C5FederatedSourceRidgeRuntime,
    SerializedRidgeV5,
)


def _ridge(names: list[str], intercept: float) -> SerializedRidgeV5:
    return SerializedRidgeV5(
        feature_names=names,
        mean=np.zeros(len(names)),
        scale=np.ones(len(names)),
        coef=np.asarray([intercept] + [0.0] * len(names)),
        clip_min=0.0,
        clip_max=300.0,
        alpha=0.0,
    )


def test_bundle_rejects_legacy_or_h2_h3_assets(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "iotj.c5_federated_source_ridge_bundle.v1",
        "status": "ready",
        "assets": {"classifier": {}, "federated_h1": {}, "target_ridge": {}, "h2_reference": {}},
    }), encoding="utf-8")
    with pytest.raises(FederatedSourceRidgeBundleError, match="schema differs|exactly"):
        load_federated_source_ridge_bundle(tmp_path)


def test_bundle_rejects_extra_top_level_dependency_declaration(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "iotj.c5_federated_source_ridge_bundle.v1",
        "status": "ready",
        "method": "x", "build_commit": "0" * 40, "assets": {},
        "feature_schema": {}, "route_schema": {}, "output_schema": {},
        "calibration_lineage": {}, "dependency_contract": {},
        "h2_policy": {},
    }), encoding="utf-8")
    with pytest.raises(FederatedSourceRidgeBundleError, match="top-level schema"):
        load_federated_source_ridge_bundle(tmp_path)


class _Classifier(torch.nn.Module):
    def forward(self, values: torch.Tensor):
        n = len(values)
        logits = torch.tensor([[0.0, 2.0, 0.0, 0.0]], device=values.device).repeat(n, 1)
        features = torch.zeros((n, 64), device=values.device)
        return logits, features, features


def test_runtime_routes_h1_and_105d_target_without_qc() -> None:
    rich_names = [f"f{i}" for i in range(104)]
    target_names = rich_names + ["srcpred_H1_source_ridge_ppm"]
    runtime = C5FederatedSourceRidgeRuntime(
        _Classifier(),
        {gas: _ridge(rich_names, 10.0 + gas) for gas in range(4)},
        {gas: _ridge(target_names, 20.0 + gas) for gas in range(4)},
        feature_extractor=lambda _window, _meta: {name: 0.0 for name in rich_names},
    )
    rows = runtime.infer(np.zeros((2, 100, 8), dtype=np.float32), [{}, {}], np.asarray([0, 1]))
    assert [row["pred_class"] for row in rows] == [1, 1]
    assert [row["source_h1_ppm"] for row in rows] == [11.0, 11.0]
    assert [row["prediction_ppm"] for row in rows] == [21.0, 21.0]
    assert all(row["qc_status"] == "disabled_pending_dependency_audit" for row in rows)


def test_runtime_rejects_nonfinite_or_misaligned_inputs() -> None:
    rich_names = [f"f{i}" for i in range(104)]
    target_names = rich_names + ["srcpred_H1_source_ridge_ppm"]
    runtime = C5FederatedSourceRidgeRuntime(
        _Classifier(),
        {gas: _ridge(rich_names, 10.0) for gas in range(4)},
        {gas: _ridge(target_names, 20.0) for gas in range(4)},
        feature_extractor=lambda _window, _meta: {name: 0.0 for name in rich_names},
    )
    bad = np.zeros((1, 100, 8), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        runtime.infer(bad, [{}], np.asarray([0]))
    with pytest.raises(RuntimeError, match="aligned"):
        runtime.infer(np.zeros((1, 100, 8), dtype=np.float32), [], np.asarray([0]))
