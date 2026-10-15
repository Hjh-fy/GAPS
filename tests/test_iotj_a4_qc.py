from __future__ import annotations

import pytest

from scripts.finalize_iotj_a4_qc import (
    COVERAGE_TARGETS,
    annotate_operating_point,
    evaluate_qc_curve,
    fit_qc_thresholds,
    random_reference,
)


def _test_records() -> list[dict[str, float | int]]:
    rows = []
    for index in range(100):
        true_class = index % 4
        rows.append(
            {
                "sample_index": index,
                "true_class": true_class,
                "pred_class": true_class if index < 90 else (true_class + 1) % 4,
                "true_ppm": 100.0,
                "pred_84d_h1_ppm": 100.0 + index / 2,
                "classification_uncertainty_risk": index / 100.0,
                "regression_disagreement_risk": index / 200.0,
                "source_prior_disagreement_risk": index / 50.0,
            }
        )
    return rows


def test_qc_thresholds_are_fit_from_label_free_calibration_fields() -> None:
    calibration = [
        {
            "classification_uncertainty_risk": index / 100.0,
            "regression_disagreement_risk": index / 200.0,
            "source_prior_disagreement_risk": index / 50.0,
        }
        for index in range(100)
    ]
    thresholds = fit_qc_thresholds(calibration, COVERAGE_TARGETS)

    assert len(thresholds) == 13
    assert thresholds[0]["target_coverage"] == pytest.approx(0.70)
    assert thresholds[-1]["target_coverage"] == pytest.approx(1.0)
    assert thresholds[-1]["threshold"] == float("inf")
    assert all(row["selection_split"] == "C5_calibration_x_only_risk" for row in thresholds)
    assert all(row["risk_formula"] == "equal_mean_of_calibration_p95_normalized_components" for row in thresholds)


def test_qc_curve_reports_capture_and_hc_operating_points() -> None:
    thresholds = fit_qc_thresholds(_test_records(), COVERAGE_TARGETS)
    rows = _test_records()
    curve = evaluate_qc_curve(rows, thresholds)
    hc90 = next(row for row in curve if row["target_coverage"] == pytest.approx(0.90))
    hc95_records = annotate_operating_point(rows, thresholds, 0.95)

    assert len(curve) == 13
    assert hc90["accepted_N"] == 90
    assert hc90["misroute_capture_rate"] == pytest.approx(1.0)
    assert sum(int(row["accepted"]) for row in hc95_records) == 95


def test_random_reference_is_seeded_and_matches_retained_counts() -> None:
    thresholds = fit_qc_thresholds(_test_records(), COVERAGE_TARGETS)
    curve = evaluate_qc_curve(_test_records(), thresholds)
    first = random_reference(_test_records(), curve, repeats=20, seed=20260804)
    second = random_reference(_test_records(), curve, repeats=20, seed=20260804)

    assert first == second
    assert len(first) == 13
    assert {row["repeats"] for row in first} == {20}
