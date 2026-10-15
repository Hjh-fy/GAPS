from __future__ import annotations

import numpy as np
import torch

from gaps_deploy.c5_federated_source_ridge_qc_runtime import C5FederatedSourceRidgeQCRuntime
from gaps_deploy.c5_federated_source_ridge_runtime import C5FederatedSourceRidgeRuntime, SerializedRidgeV5
from gaps_deploy.runtime_v5_qc import RuntimeV5QCPolicy


def _ridge(names: list[str], intercept: float) -> SerializedRidgeV5:
    return SerializedRidgeV5(names, np.zeros(len(names)), np.ones(len(names)), np.asarray([intercept] + [0.0] * len(names)), 0.0, 300.0, 0.0)


class _Classifier(torch.nn.Module):
    def forward(self, values: torch.Tensor):
        n = len(values)
        logits = torch.tensor([[4.0, 0.0, 0.0, 0.0]], device=values.device).repeat(n, 1)
        representation = torch.zeros((n, 64), device=values.device)
        return logits, representation, representation


def _policy() -> RuntimeV5QCPolicy:
    distributions = {key: [0.0, 1.0] for key in ("entropy", "inverse_margin", "prototype_distance", "support_distance", "normalized_regression_disagreement")}
    return RuntimeV5QCPolicy.from_payload({
        "schema_version": "iotj.runtime_v5_qc_policy.v1", "status": "locked", "selected_candidate": "QC3", "epsilon": 1e-8,
        "feature_reference": {"feature_dimension": 64, "classes": {"0": {"mean": [0.0] * 64, "scale": [1.0] * 64, "support": [[0.0] * 64], "n": 1}}},
        "regression_consistency_scale": {"per_predicted_gas": {"0": {"scale": 10.0}}},
        "component_distributions": distributions,
        "workpoints": {"HC95": {"accept_threshold": 1.0, "reject_threshold": 1.0}, "HC90": {"accept_threshold": 0.0, "reject_threshold": 0.5}},
        "decision_semantics": {"auto_output_only_for_accept": True},
    })


def test_qc_runtime_emits_accept_only_auto_output_and_all_risk_components() -> None:
    rich = [f"f{i}" for i in range(104)]
    target = rich + ["srcpred_H1_source_ridge_ppm"]
    base = C5FederatedSourceRidgeRuntime(
        _Classifier(), {gas: _ridge(rich, 10.0) for gas in range(4)},
        {gas: _ridge(target, 20.0) for gas in range(4)},
        feature_extractor=lambda _window, _meta: {name: 0.0 for name in rich},
    )
    runtime = C5FederatedSourceRidgeQCRuntime(base, _policy(), "HC95")
    row = runtime.infer(np.zeros((1, 100, 8), dtype=np.float32), [{}], np.asarray([0]))[0]
    assert row["qc_decision"] == "accept"
    assert row["auto_output_ppm"] == row["prediction_ppm"] == 20.0
    assert set(row["raw_risk_components"]) == {"entropy", "inverse_margin", "prototype_distance", "support_distance", "normalized_regression_disagreement"}
