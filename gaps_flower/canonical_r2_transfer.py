"""Frozen primitives for the canonical-v1 transfer-safe R2 gate."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from gaps_flower.canonical_r1_v1 import (CVResult, assign_balanced_group_folds,
    fit_ridge_model, predict_ridge_model, select_grouped_cv_alpha)


BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def residual_transfer_prediction(source_prior, target_residual):
    return np.asarray(source_prior, dtype=np.float64) + np.asarray(target_residual, dtype=np.float64)


def shrinkage_transfer_prediction(target_83d, source_prior, beta: float):
    if float(beta) not in BETAS:
        raise ValueError("beta is outside the frozen grid")
    return (1.0 - float(beta)) * np.asarray(target_83d, dtype=np.float64) + float(beta) * np.asarray(source_prior, dtype=np.float64)


def select_shrinkage_beta(truth, target_83d, source_prior, betas: Sequence[float] = BETAS):
    truth = np.asarray(truth, dtype=np.float64)
    scores = {}
    for beta in betas:
        pred = shrinkage_transfer_prediction(target_83d, source_prior, float(beta))
        scores[float(beta)] = float(np.sqrt(np.mean((pred - truth) ** 2)))
    selected = min(range(len(betas)), key=lambda i: (scores[float(betas[i])], i))
    return float(betas[selected]), scores


def select_grouped_residual_alpha(x, truth, source_prior, groups, alphas, n_folds: int = 5) -> CVResult:
    residual = np.asarray(truth, dtype=np.float64) - np.asarray(source_prior, dtype=np.float64)
    return select_grouped_cv_alpha(x, residual, groups, alphas, n_folds=n_folds)


def grouped_shrinkage_oof_predictions(x, truth, source_prior, groups, alphas,
                                      n_folds: int = 5):
    x=np.asarray(x,dtype=np.float64); truth=np.asarray(truth,dtype=np.float64)
    groups=np.asarray(groups,dtype=str); folds=assign_balanced_group_folds(groups,n_folds)
    oof=np.empty(len(truth),dtype=np.float64)
    for fold in range(n_folds):
        valid=folds==fold
        cv=select_grouped_cv_alpha(x[~valid],truth[~valid],groups[~valid],alphas,n_folds=min(n_folds,len(np.unique(groups[~valid]))))
        model=fit_ridge_model(x[~valid],truth[~valid],cv.alpha,float(truth.min()),float(truth.max()))
        oof[valid]=predict_ridge_model(model,x[valid])
    return oof,{g:int(folds[np.flatnonzero(groups==g)[0]]) for g in np.unique(groups)}


def select_grouped_shrinkage_beta(truth, target_83d_oof, source_prior, groups,
                                  betas: Sequence[float] = BETAS, n_folds: int = 5):
    truth = np.asarray(truth, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    folds = assign_balanced_group_folds(groups, n_folds=n_folds)
    scores = {}
    for beta in betas:
        pred = shrinkage_transfer_prediction(target_83d_oof, source_prior, float(beta))
        scores[float(beta)] = float(np.sqrt(np.mean((pred - truth) ** 2)))
    selected = min(range(len(betas)), key=lambda i: (scores[float(betas[i])], i))
    return {
        "selected_beta": float(betas[selected]),
        "pooled_rmse": scores,
        "fold_by_group": {g: int(folds[np.flatnonzero(groups == g)[0]]) for g in np.unique(groups)},
    }


def decide_transfer_candidate(r84_rmse: float, candidate_rmse: float, bootstrap_ci_high: float,
                              gas_relative_degradation: Mapping[str, float]):
    improvement = (float(r84_rmse) - float(candidate_rmse)) / float(r84_rmse)
    gates = {
        "pooled_rmse_improvement_at_least_3pct": improvement >= 0.03,
        "no_gas_degradation_above_5pct": all(float(v) <= 0.05 for v in gas_relative_degradation.values()),
        "paired_grouped_bootstrap_ci_entirely_below_zero": float(bootstrap_ci_high) < 0.0,
    }
    return {"retained": all(gates.values()), "improvement_fraction": improvement, "gates": gates}
