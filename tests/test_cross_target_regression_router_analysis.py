import math

from scripts.analyze_gaps_cross_target_regression import regression_metrics


def test_regression_metrics_preserve_scope_and_class_ranges():
    rows = [
        {"true_ppm": 10.0, "true_class": 0, "prediction": 20.0},
        {"true_ppm": 100.0, "true_class": 1, "prediction": 80.0},
    ]
    observed = regression_metrics(rows, "prediction")
    assert observed["N"] == 2
    assert math.isclose(observed["RMSE"], math.sqrt(250.0))
    assert math.isclose(observed["MAE"], 15.0)
    assert math.isclose(
        observed["NRMSE"],
        math.sqrt(((10.0 / 112.5) ** 2 + (20.0 / 225.0) ** 2) / 2),
    )


def test_regression_metrics_reject_empty_scope():
    try:
        regression_metrics([], "prediction")
    except RuntimeError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty scope must fail closed")
