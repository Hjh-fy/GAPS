from __future__ import annotations

import numpy as np
import pytest


def test_sensor_shift_statistics_are_per_channel_and_feature_independent() -> None:
    from scripts.evaluate_iotj_final_classification_le1 import (
        sensor_channel_shift_rows,
        sensor_covariance_diagnostics,
    )

    source = np.array(
        [
            [[0.0, 10.0], [2.0, 14.0]],
            [[4.0, 18.0], [6.0, 22.0]],
        ],
        dtype=np.float64,
    )
    target = source.copy()
    target[:, :, 0] += 2.0
    target[:, :, 1] *= 2.0

    rows = sensor_channel_shift_rows(source, target, target_id="C5")
    assert [row["channel"] for row in rows] == [0, 1]
    assert rows[0]["mean_shift"] == pytest.approx(2.0)
    assert rows[0]["median_shift"] == pytest.approx(2.0)
    assert rows[0]["std_shift"] == pytest.approx(0.0)
    assert rows[0]["iqr_shift"] == pytest.approx(0.0)
    assert rows[0]["q05_shift"] == pytest.approx(2.0)
    assert rows[0]["q95_shift"] == pytest.approx(2.0)
    assert rows[0]["standardized_mean_difference"] == pytest.approx(
        2.0 / np.std(source[:, :, 0].reshape(-1), ddof=0)
    )

    covariance = sensor_covariance_diagnostics(source, target, target_id="C5")
    assert covariance["target_id"] == "C5"
    assert covariance["num_channels"] == 2
    assert covariance["covariance_frobenius_shift"] > 0.0
    assert np.isfinite(covariance["covariance_relative_frobenius_shift"])


def test_sensor_shift_rejects_non_window_arrays() -> None:
    from scripts.evaluate_iotj_final_classification_le1 import sensor_channel_shift_rows

    with pytest.raises(ValueError, match=r"\[windows, time, channels\]"):
        sensor_channel_shift_rows(np.ones((3, 2)), np.ones((3, 2)), target_id="C3")


def test_source_target_f1_gap_uses_one_combined_source_value_for_every_target() -> None:
    from scripts.evaluate_iotj_final_classification_le1 import add_source_target_f1_gaps

    rows = [
        {"method": "FedAvg", "target_id": "C3", "macro_f1": 0.70},
        {"method": "FedAvg", "target_id": "C4", "macro_f1": 0.65},
        {"method": "FedAvg", "target_id": "C5", "macro_f1": 0.80},
    ]
    result = add_source_target_f1_gaps(rows, source_macro_f1=0.90)

    assert [row["source_macro_f1"] for row in result] == [0.90, 0.90, 0.90]
    assert [row["source_target_f1_gap"] for row in result] == pytest.approx(
        [0.20, 0.25, 0.10]
    )


def test_classification_metrics_have_fixed_four_class_order_and_15_bin_ece() -> None:
    from scripts.evaluate_iotj_final_classification_le1 import classification_metrics

    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    probabilities = np.eye(4, dtype=np.float64) * 0.9 + 0.1 / 4.0
    metrics = classification_metrics(probabilities, labels, num_classes=4, ece_bins=15)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == np.eye(4, dtype=int).tolist()
    assert metrics["ece_bins"] == 15
    assert metrics["nll"] == pytest.approx(-np.log(0.925))
