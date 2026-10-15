import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/run_iotj_canonical_regression_reconstruction_q1.py"
    spec = importlib.util.spec_from_file_location("q1_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q1_gate_requires_q0_insufficiency_and_useful_regression_uncertainty():
    module = _module()
    assert module.q1_trigger("MULTISIGNAL_QC_NOT_ESTABLISHED", 0.08, 0.07, 0.06, 0.05)
    assert not module.q1_trigger("CONFIDENCE_QC_PREFERRED", 0.08, 0.07, 0.06, 0.05)
    assert not module.q1_trigger("MULTISIGNAL_QC_NOT_ESTABLISHED", 0.08, 0.09, 0.06, 0.05)


def test_q1_group_split_is_deterministic_and_has_no_overlap():
    module = _module()
    groups = np.asarray([f"raw-{index // 2}" for index in range(20)])
    first = module.group_aware_split(groups)
    second = module.group_aware_split(groups)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not (set(groups[first[0]]) & set(groups[first[1]]))
    assert first[0].any() and first[1].any()


def test_q1_absolute_residual_interval_uses_fixed_90_percent_level():
    module = _module()
    residuals = np.arange(1.0, 11.0)
    radius = module.conformal_radius(residuals, coverage=0.90)
    assert radius == pytest.approx(10.0)
    lower, upper = module.empirical_interval(np.asarray([20.0, 30.0]), radius)
    assert lower == pytest.approx([10.0, 20.0])
    assert upper == pytest.approx([30.0, 40.0])


def test_q1_empirical_interval_accepts_shape_compatible_per_sample_radius():
    module = _module()
    lower, upper = module.empirical_interval(
        np.asarray([20.0, 30.0]), np.asarray([2.0, 5.0])
    )
    assert lower == pytest.approx([18.0, 25.0])
    assert upper == pytest.approx([22.0, 35.0])
    with pytest.raises(ValueError, match="broadcast-compatible"):
        module.empirical_interval(np.asarray([20.0, 30.0]), np.asarray([1.0, 2.0, 3.0]))


def test_q1_calibration_ecdf_and_equal_mean_are_fixed_without_weight_search():
    module = _module()
    calibration = np.asarray([0.1, 0.2, 0.4, 0.8])
    normalized = module.calibration_ecdf(calibration, np.asarray([0.05, 0.2, 1.0]))
    assert normalized == pytest.approx([0.0, 0.5, 1.0])
    combined = module.equal_mean_risk(np.asarray([0.2, 0.8]), np.asarray([0.6, 0.4]))
    assert combined == pytest.approx([0.4, 0.6])


def test_q1_calibration_ecdf_is_unchanged_by_evaluation_values():
    module = _module()
    calibration = np.asarray([0.1, 0.2, 0.4, 0.8])
    before = calibration.copy()
    module.calibration_ecdf(calibration, np.asarray([-100.0, 100.0]))
    assert calibration == pytest.approx(before)


def test_q1_decision_requires_five_percent_gain_on_c5_and_pooled():
    module = _module()
    assert module.decide_q1(0.10, 0.094, 0.20, 0.18) == "CONFORMAL_AUGMENTED_QC_SUPPORTED"
    assert module.decide_q1(0.10, 0.096, 0.20, 0.18) == "CONFIDENCE_QC_FINAL"
    assert module.decide_q1(0.10, 0.094, 0.20, 0.191) == "CONFIDENCE_QC_FINAL"


def test_q1_runner_inspect_does_not_open_target_test():
    result = _module().inspect()
    assert result["triggered"] is True
    assert result["regression_backend"] == "R84_CONCAT"
    assert result["nominal_interval_coverage"] == pytest.approx(0.90)
    assert result["target_test_opened"] is False
    assert result["formal_root_exists"] is False
