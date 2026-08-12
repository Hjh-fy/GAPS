import numpy as np
import pytest


def test_q0_coverage_grid_is_frozen():
    from gaps_flower.canonical_qc_evaluation import COVERAGE_GRID

    assert len(COVERAGE_GRID) == 51
    assert COVERAGE_GRID[0] == pytest.approx(0.50)
    assert COVERAGE_GRID[-1] == pytest.approx(1.00)
    assert np.diff(COVERAGE_GRID) == pytest.approx(np.full(50, 0.01))


def test_q0_confidence_risk_uses_max_probability():
    from gaps_flower.canonical_qc_evaluation import classification_confidence_risk

    probs = np.asarray([[0.1, 0.7, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    assert classification_confidence_risk(probs) == pytest.approx([0.3, 0.75])


def test_q0_retention_has_deterministic_identity_ties():
    from gaps_flower.canonical_qc_evaluation import retained_indices

    risk = np.asarray([0.2, 0.1, 0.1, 0.3])
    identities = np.asarray(["d", "b", "a", "c"])
    assert retained_indices(risk, identities, 0.5).tolist() == [2, 1]


def test_q0_random_reference_is_fixed_seed_and_equal_count():
    from gaps_flower.canonical_qc_evaluation import random_reference_metrics

    truth = np.arange(10, dtype=float)
    prediction = truth + 1.0
    gas_range = np.full(10, 10.0)
    first = random_reference_metrics(truth, prediction, gas_range, 0.6, 20, 42)
    second = random_reference_metrics(truth, prediction, gas_range, 0.6, 20, 42)
    assert first == second
    assert first["retained_n"] == 6
    assert first["repetitions"] == 20


def test_q0_grouped_dispersion_is_calibration_grouped_and_normalized():
    from gaps_flower.canonical_qc_evaluation import grouped_model_dispersion

    x = np.arange(60, dtype=float).reshape(20, 3)
    truth = np.linspace(0, 19, 20)
    groups = np.asarray([f"f{i // 2}" for i in range(20)])
    test_x = np.arange(18, dtype=float).reshape(6, 3)
    score, audit = grouped_model_dispersion(
        x, truth, groups, test_x, alpha=1.0, gas_range=100.0, n_folds=5
    )
    assert score.shape == (6,)
    assert np.isfinite(score).all()
    assert (score >= 0).all()
    assert audit["n_models"] == 5
    assert audit["group_overlap"] is False


def test_q0_equal_mean_unavailable_cannot_be_substituted():
    from gaps_flower.canonical_qc_evaluation import audit_equal_mean_availability

    status = audit_equal_mean_availability({"confidence", "regression_uncertainty"})
    assert status["decision"] == "Q4_CANONICAL_INPUTS_UNAVAILABLE"
    assert status["available"] is False


def test_q0_decision_rules_are_frozen():
    from gaps_flower.canonical_qc_evaluation import decide_qc_necessity

    assert decide_qc_necessity(None, 0.08, 0.10, 0.09) == "MULTISIGNAL_QC_NOT_ESTABLISHED"
    assert decide_qc_necessity(0.07, 0.08, 0.10, 0.09) == "MULTISIGNAL_QC_SUPPORTED"
    assert decide_qc_necessity(0.09, 0.08, 0.10, 0.09) == "CONFIDENCE_QC_PREFERRED"


def test_q0_curve_reports_frozen_metrics():
    from gaps_flower.canonical_qc_evaluation import aurc, risk_coverage_curve

    truth = np.asarray([0.0, 10.0, 20.0, 30.0])
    pred = np.asarray([0.0, 12.0, 30.0, 70.0])
    rows = risk_coverage_curve(
        truth, pred, np.full(4, 100.0), np.asarray([0, 1, 2, 3]),
        np.asarray([0, 1, 1, 3]), np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray(["a", "b", "c", "d"]), "CONFIDENCE", (0.5, 1.0),
    )
    assert [row["retained_n"] for row in rows] == [2, 4]
    assert rows[-1]["misroute_rate"] == pytest.approx(0.25)
    assert rows[-1]["error_ge_40ppm_rate"] == pytest.approx(0.25)
    assert aurc(rows, "RMSE") > 0


def test_q0_runner_inspect_is_test_free_and_r84_frozen():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/run_iotj_canonical_q0_qc.py"
    spec = importlib.util.spec_from_file_location("q0_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.inspect()
    assert result["regression_backend"] == "R84_CONCAT"
    assert result["target_test_opened"] is False
    assert result["q4_status"] == "Q4_CANONICAL_INPUTS_UNAVAILABLE"
