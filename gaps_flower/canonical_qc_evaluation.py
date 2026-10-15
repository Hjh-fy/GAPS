"""Frozen primitives for canonical-v1 Q0 QC necessity evaluation."""
from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from gaps_flower.canonical_r1_v1 import (
    assign_balanced_group_folds,
    fit_ridge_model,
    predict_ridge_model,
)


COVERAGE_GRID = np.round(np.arange(0.50, 1.001, 0.01), 2)
RANDOM_REPETITIONS = 5000
RANDOM_SEED = 42
EQUAL_MEAN_REQUIRED_COMPONENTS = frozenset(
    {"classification_uncertainty", "regression_disagreement", "source_prior_disagreement"}
)


def classification_confidence_risk(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("class probabilities must have shape N x 4")
    if not np.isfinite(probabilities).all():
        raise ValueError("class probabilities must be finite")
    return 1.0 - probabilities.max(axis=1)


def retained_indices(risk, physical_identities, coverage: float):
    risk = np.asarray(risk, dtype=np.float64)
    identities = np.asarray(physical_identities, dtype=str)
    if len(risk) != len(identities):
        raise ValueError("risk and identity lengths differ")
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage outside (0, 1]")
    retained_n = min(len(risk), max(1, int(math.ceil(len(risk) * float(coverage)))))
    order = np.lexsort((identities, risk))
    return order[:retained_n]


def _basic_metrics(truth, prediction, gas_range):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    gas_range = np.asarray(gas_range, dtype=np.float64)
    error = prediction - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nrmse_range": float(np.sqrt(np.mean((error / gas_range) ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


def random_reference_metrics(truth, prediction, gas_range, coverage: float,
                             repetitions: int = RANDOM_REPETITIONS,
                             seed: int = RANDOM_SEED):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    gas_range = np.broadcast_to(np.asarray(gas_range, dtype=np.float64), truth.shape)
    retained_n = min(len(truth), max(1, int(math.ceil(len(truth) * float(coverage)))))
    rng = np.random.default_rng(int(seed))
    samples = [_basic_metrics(truth[idx], prediction[idx], gas_range[idx])
               for idx in (rng.choice(len(truth), retained_n, replace=False)
                           for _ in range(int(repetitions)))]
    return {
        "retained_n": retained_n,
        "repetitions": int(repetitions),
        "seed": int(seed),
        **{f"{metric}_{suffix}": float(function([row[metric] for row in samples]))
           for metric in ("rmse", "nrmse_range", "mae")
           for suffix, function in (("mean", np.mean), ("sd", lambda x: np.std(x, ddof=1)))},
    }


def grouped_model_dispersion(calibration_x, calibration_truth, calibration_groups,
                             evaluation_x, alpha: float, gas_range: float,
                             n_folds: int = 5):
    x = np.asarray(calibration_x, dtype=np.float64)
    truth = np.asarray(calibration_truth, dtype=np.float64)
    groups = np.asarray(calibration_groups, dtype=str)
    evaluation_x = np.asarray(evaluation_x, dtype=np.float64)
    if len(np.unique(groups)) < n_folds:
        raise ValueError("insufficient calibration groups")
    folds = assign_balanced_group_folds(groups, n_folds=n_folds)
    predictions = []
    overlap = False
    fold_groups = []
    for fold in range(n_folds):
        held = folds == fold
        train_groups = set(groups[~held])
        held_groups = set(groups[held])
        overlap = overlap or bool(train_groups & held_groups)
        train_truth = truth[~held]
        model = fit_ridge_model(
            x[~held], train_truth, float(alpha),
            float(train_truth.min()), float(train_truth.max())
        )
        predictions.append(predict_ridge_model(model, evaluation_x))
        fold_groups.append({"fold": fold, "train_groups": len(train_groups), "held_groups": len(held_groups)})
    score = np.std(np.asarray(predictions), axis=0, ddof=0) / float(gas_range)
    return score, {"n_models": n_folds, "group_overlap": overlap, "folds": fold_groups}


def audit_equal_mean_availability(available_components: Iterable[str]):
    available_components = frozenset(available_components)
    missing = sorted(EQUAL_MEAN_REQUIRED_COMPONENTS - available_components)
    return {
        "available": not missing,
        "missing_components": missing,
        "decision": "Q4_CANONICAL_INPUTS_AVAILABLE" if not missing else "Q4_CANONICAL_INPUTS_UNAVAILABLE",
    }


def decide_qc_necessity(q4_aurc_nrmse, confidence_aurc_nrmse: float,
                        random_aurc_nrmse: float, regression_aurc_nrmse: float):
    if q4_aurc_nrmse is None:
        return "MULTISIGNAL_QC_NOT_ESTABLISHED"
    q4 = float(q4_aurc_nrmse)
    confidence = float(confidence_aurc_nrmse)
    random = float(random_aurc_nrmse)
    if q4 <= 0.95 * confidence and q4 <= 0.95 * random:
        return "MULTISIGNAL_QC_SUPPORTED"
    if confidence <= q4 or confidence < random:
        return "CONFIDENCE_QC_PREFERRED"
    return "QC_CORE_NOT_SUPPORTED"


def risk_coverage_curve(truth, prediction, gas_range, true_class, predicted_class,
                        risk, identities, policy: str, coverages=COVERAGE_GRID):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    gas_range = np.asarray(gas_range, dtype=np.float64)
    true_class = np.asarray(true_class, dtype=int)
    predicted_class = np.asarray(predicted_class, dtype=int)
    rows = []
    for coverage in coverages:
        idx = retained_indices(risk, identities, float(coverage))
        error = prediction[idx] - truth[idx]
        rows.append({
            "policy": policy,
            "target_coverage": float(coverage),
            "actual_coverage": len(idx) / len(truth),
            "retained_n": len(idx),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "NRMSE_range": float(np.sqrt(np.mean((error / gas_range[idx]) ** 2))),
            "MAE": float(np.mean(np.abs(error))),
            "misroute_rate": float(np.mean(true_class[idx] != predicted_class[idx])),
            "error_ge_40ppm_rate": float(np.mean(np.abs(error) >= 40.0)),
            "P90_absolute_error": float(np.percentile(np.abs(error), 90)),
        })
    return rows


def aurc(curve, metric: str):
    ordered = sorted(curve, key=lambda row: float(row["actual_coverage"]))
    return float(np.trapz(
        [float(row[metric]) for row in ordered],
        [float(row["actual_coverage"]) for row in ordered],
    ) / (float(ordered[-1]["actual_coverage"]) - float(ordered[0]["actual_coverage"])))
