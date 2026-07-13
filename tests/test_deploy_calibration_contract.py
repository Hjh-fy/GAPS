from __future__ import annotations

import math

import numpy as np
import pytest

from gaps_deploy.calibration import RegressionCalibrator


def _routing() -> dict:
    return {
        "selected_modes": {str(class_id): "none" for class_id in range(4)},
        "affine_params": {},
        "phase_affine_params": {},
    }


def _affine(*, a: float = 1.0, b: float = 0.0, mode: str = "affine_only") -> dict:
    return {"a": a, "b": b, "mode": mode}


def _phase_affine() -> dict:
    return {
        "num_phases": 3,
        "phase_calibrators": {
            str(phase): _affine(a=1.0 + phase, b=float(phase))
            for phase in range(3)
        },
    }


def test_routing_config_requires_every_class_exactly_once() -> None:
    with pytest.raises(ValueError, match="selected_modes"):
        RegressionCalibrator(num_classes=4).load_routing_config(
            {"selected_modes": {"0": "none"}}
        )


def test_routing_config_rejects_colliding_class_keys() -> None:
    config = _routing()
    config["selected_modes"][0] = "none"

    with pytest.raises(ValueError, match="colliding"):
        RegressionCalibrator(num_classes=4).load_routing_config(config)


@pytest.mark.parametrize("mode", ["unknown", "affine_only", "phase_affine_only"])
def test_selected_calibration_mode_requires_known_complete_parameters(mode: str) -> None:
    config = _routing()
    config["selected_modes"]["0"] = mode

    with pytest.raises(ValueError, match="mode|params"):
        RegressionCalibrator(num_classes=4).load_routing_config(config)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "not-a-number"])
def test_affine_parameters_must_be_finite_numbers(value) -> None:
    config = _routing()
    config["selected_modes"]["0"] = "affine_only"
    config["affine_params"]["0"] = _affine(a=value)

    with pytest.raises(ValueError, match="finite|numeric"):
        RegressionCalibrator(num_classes=4).load_routing_config(config)


def test_phase_affine_requires_exact_phase_coverage() -> None:
    config = _routing()
    config["selected_modes"]["0"] = "phase_affine_only"
    config["phase_affine_params"]["0"] = _phase_affine()
    del config["phase_affine_params"]["0"]["phase_calibrators"]["2"]

    with pytest.raises(ValueError, match="phase_calibrators"):
        RegressionCalibrator(num_classes=4, num_phases=3).load_routing_config(config)


def test_valid_parameterized_routes_load_and_calibrate() -> None:
    config = _routing()
    config["selected_modes"]["0"] = "bias_only"
    config["affine_params"]["0"] = _affine(a=1.0, b=2.0, mode="bias_only")
    config["selected_modes"]["1"] = "phase_affine_only"
    config["phase_affine_params"]["1"] = _phase_affine()
    calibrator = RegressionCalibrator(num_classes=4, num_phases=3)

    calibrator.load_routing_config(config)
    result = calibrator.calibrate(
        np.asarray([20.0, 30.0]),
        np.asarray([0, 1]),
        np.asarray([0, 2]),
    )

    assert result.tolist() == pytest.approx([22.0, 92.0])
