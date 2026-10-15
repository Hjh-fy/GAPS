from __future__ import annotations

import json

import numpy as np
import pytest

import gaps_deploy.qc_policy as qc_policy
from gaps_deploy.final_runtime import FinalDeployRuntime
from gaps_deploy.inference import DeployResult
from gaps_deploy.qc_policy import (
    QCPolicy,
    RiskScoreComputer,
    TwoThresholdDecider,
)


def _classifier_policy(**overrides) -> QCPolicy:
    values = {
        "policy_name": "classifier_uncertainty_test",
        "scores": ["classifier_uncertainty"],
        "thresholds": {"classifier_uncertainty": 0.5},
        "low_ratio": 0.9,
        "high_ratio": 1.1,
        "group": "ALL",
    }
    values.update(overrides)
    return QCPolicy(**values)


def _response_policy() -> QCPolicy:
    return QCPolicy(
        policy_name="response_test",
        scores=["composite_response_risk"],
        thresholds={"composite_response_risk": 1.0},
        low_ratio=0.9,
        high_ratio=1.1,
        group="ALL",
    )


def test_no_policy_rejects_without_numeric_risk_ratio() -> None:
    decision = TwoThresholdDecider().decide({"classifier_uncertainty": 0.01})

    assert decision.decision == "reject"
    assert decision.risk_ratio is None
    assert decision.risk_reasons == ["qc_policy_missing"]
    assert decision.policy_name == "no_policy"


@pytest.mark.parametrize(
    ("scores", "reason"),
    [
        ({}, "qc_score_missing:classifier_uncertainty"),
        ({"classifier_uncertainty": float("nan")}, "qc_score_nonfinite:classifier_uncertainty"),
        ({"classifier_uncertainty": float("inf")}, "qc_score_nonfinite:classifier_uncertainty"),
    ],
)
def test_required_score_must_be_available(scores: dict[str, float], reason: str) -> None:
    decider = TwoThresholdDecider()
    decider.load_policy(_classifier_policy())

    decision = decider.decide(scores)

    assert decision.decision == "reject"
    assert decision.risk_ratio is None
    assert decision.risk_reasons == [reason]


@pytest.mark.parametrize(
    "policy",
    [
        _classifier_policy(scores=[]),
        _classifier_policy(thresholds={}),
        _classifier_policy(thresholds={"classifier_uncertainty": 0.0}),
        _classifier_policy(thresholds={"classifier_uncertainty": float("inf")}),
        _classifier_policy(low_ratio=1.1, high_ratio=1.1),
        _classifier_policy(low_ratio=-0.1),
        _classifier_policy(scores=["not_registered"], thresholds={"not_registered": 1.0}),
    ],
)
def test_malformed_policy_is_rejected_when_loaded(policy: QCPolicy) -> None:
    with pytest.raises(ValueError, match="QC policy"):
        TwoThresholdDecider().load_policy(policy)


def test_corrupted_in_memory_policy_still_fails_closed() -> None:
    decider = TwoThresholdDecider()
    policy = _classifier_policy()
    decider.load_policy(policy)
    policy.thresholds.clear()

    decision = decider.decide({"classifier_uncertainty": 0.1})

    assert decision.decision == "reject"
    assert decision.risk_ratio is None
    assert decision.risk_reasons == ["qc_threshold_invalid:classifier_uncertainty"]


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.1, "accept"), (0.5, "review"), (0.6, "reject")],
)
def test_valid_policy_keeps_two_threshold_behavior(score: float, expected: str) -> None:
    decider = TwoThresholdDecider()
    decider.load_policy(_classifier_policy())

    decision = decider.decide({"classifier_uncertainty": score})

    assert decision.decision == expected
    assert decision.risk_ratio == pytest.approx(score / 0.5)


def test_response_scores_are_unavailable_without_response_references() -> None:
    scores = RiskScoreComputer(calib_refs={}).compute(
        logits=np.asarray([3.0, 1.0, 0.0, -1.0]),
        pred_ppm=50.0,
        class_id=0,
        features=np.ones((100, 8), dtype=np.float32),
    )

    assert "classifier_uncertainty" in scores
    assert "response_signature_norm" not in scores
    assert "route_response_risk" not in scores
    assert "composite_response_risk" not in scores

    decider = TwoThresholdDecider()
    decider.load_policy(_response_policy())
    decision = decider.decide(scores)
    assert decision.decision == "reject"
    assert decision.risk_reasons == ["qc_score_missing:composite_response_risk"]


def test_response_reference_validation_requires_every_class() -> None:
    ref = {
        "center": [0.0] * 8,
        "scale": [1.0] * 8,
        "z_sigs": [[0.0] * 8],
        "loocv_p90": 1.0,
        "rows": [{"concentration": 50.0}],
    }

    with pytest.raises(ValueError, match="classes"):
        qc_policy.validate_calibration_refs({0: ref}, num_classes=4)


def test_invalid_response_reference_cannot_produce_a_low_risk_score() -> None:
    invalid_ref = {
        "center": [0.0] * 8,
        "scale": [0.0] * 8,
        "z_sigs": [[0.0] * 8],
        "loocv_p90": 1.0,
        "rows": [{"concentration": 50.0}],
    }
    refs = {class_id: invalid_ref for class_id in range(4)}

    scores = RiskScoreComputer(refs).compute(
        logits=np.asarray([3.0, 1.0, 0.0, -1.0]),
        pred_ppm=50.0,
        class_id=0,
        features=np.ones((100, 8), dtype=np.float32),
    )

    assert "composite_response_risk" not in scores


def test_legacy_reference_without_ranking_cannot_synthesize_zero_risk() -> None:
    refs = {
        class_id: {
            "center": [0.0] * 8,
            "scale": [1.0] * 8,
            "z_sigs": [[0.0] * 8],
            "loocv_p90": 1.0,
            "rows": [{"concentration": 50.0}],
        }
        for class_id in range(4)
    }
    scores = RiskScoreComputer(refs).compute(
        logits=np.asarray([3.0, 1.0, 0.0, -1.0]),
        pred_ppm=50.0,
        class_id=0,
        features=np.ones((100, 8), dtype=np.float32),
        extra_info={
            "class_response_rank_risk": 0.0,
            "class_response_margin_risk": 0.0,
        },
    )

    assert "response_signature_norm" in scores
    assert "class_response_rank_risk" not in scores
    assert "class_response_margin_risk" not in scores
    assert "route_response_risk" not in scores
    assert "composite_response_risk" not in scores

    decider = TwoThresholdDecider()
    decider.load_policy(_response_policy())
    decision = decider.decide(scores)
    assert decision.decision == "reject"
    assert decision.risk_reasons == ["qc_score_missing:composite_response_risk"]


def test_final_runtime_never_auto_outputs_reject_and_emits_json_null() -> None:
    result = DeployResult(
        pred_gas="CO",
        pred_class=1,
        confidence=0.9,
        base_r3ak16_raw_ppm=40.0,
        routed_pred_ppm=42.0,
        final_ppm=42.0,
        qc_status="reject",
        risk_score=None,
    )

    row = FinalDeployRuntime._public_row(result, 42.0)

    assert row["auto_output_ppm"] == ""
    assert row["risk_score"] is None
    assert '"risk_score": null' in json.dumps(row)


def test_unavailable_qc_evidence_skips_residual_postprocessing() -> None:
    class ExplodingResidual:
        enabled = True

        def apply(self, *args, **kwargs):
            raise AssertionError("residual correction must not run without QC evidence")

    runtime = FinalDeployRuntime.__new__(FinalDeployRuntime)
    runtime.rich_residual = ExplodingResidual()
    runtime.co_params = {}
    runtime.client_id = "C5"
    result = DeployResult(final_ppm=42.0, qc_status="reject", risk_score=None)

    corrected = runtime._artifact_corrected_ppm(
        np.ones((100, 8), dtype=np.float32),
        result,
    )

    assert corrected == pytest.approx(42.0)
