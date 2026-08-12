import numpy as np
import inspect

from gaps_flower.canonical_r2_transfer import (
    residual_transfer_prediction,
    shrinkage_transfer_prediction,
    select_shrinkage_beta,
    decide_transfer_candidate,
    select_grouped_residual_alpha,
    select_grouped_shrinkage_beta,
    grouped_shrinkage_oof_predictions,
)


def test_transfer_formulas():
    source = np.array([10.0, 20.0])
    residual = np.array([1.5, -2.0])
    target = np.array([14.0, 18.0])
    assert np.allclose(residual_transfer_prediction(source, residual), [11.5, 18.0])
    assert np.allclose(shrinkage_transfer_prediction(target, source, 0.25), [13.0, 18.5])


def test_shrinkage_beta_tie_uses_registered_order():
    y = np.array([1.0, 2.0])
    pred = np.array([1.0, 2.0])
    chosen, scores = select_shrinkage_beta(y, pred, pred, [0.0, 0.25, 0.5])
    assert chosen == 0.0
    assert list(scores) == [0.0, 0.25, 0.5]


def test_decision_requires_all_three_registered_gates():
    assert decide_transfer_candidate(10.0, 9.6, -0.01, {"Et": 0.0})["retained"] is True
    assert decide_transfer_candidate(10.0, 9.8, -0.01, {"Et": 0.0})["retained"] is False
    assert decide_transfer_candidate(10.0, 9.6, 0.0, {"Et": 0.0})["retained"] is False
    assert decide_transfer_candidate(10.0, 9.6, -0.01, {"Et": 0.051})["retained"] is False


def test_grouped_residual_cv_keeps_filename_groups_whole_and_ties_by_alpha_order():
    x = np.zeros((10, 1))
    y = np.arange(10.0)
    source = y.copy()
    groups = np.repeat([f"f{i}" for i in range(5)], 2)
    result = select_grouped_residual_alpha(x, y, source, groups, [0.0, 1.0])
    assert result.alpha == 0.0
    assert set(result.fold_by_group) == set(groups)
    assert len(set(result.fold_by_group.values())) == 5


def test_grouped_shrinkage_selector_has_no_test_argument_and_deterministic_tie():
    y = np.arange(10.0)
    groups = np.repeat([f"f{i}" for i in range(5)], 2)
    result = select_grouped_shrinkage_beta(y, y, y, groups, [0.0, 0.25])
    assert result["selected_beta"] == 0.0
    assert set(result["fold_by_group"]) == set(groups)
    assert "test" not in inspect.signature(select_grouped_shrinkage_beta).parameters
    assert "test" not in inspect.signature(select_grouped_residual_alpha).parameters


def test_protocol_has_only_two_candidates_and_fixed_grids():
    import json
    from pathlib import Path
    manifest = json.loads(Path("docs/experiments/iotj_canonical_v1_final/canonical_r2_transfer_safe_20260812/protocol_manifest.json").read_text())
    assert set(manifest["candidates"]) == {"RESIDUAL_TRANSFER", "SHRINKAGE_TRANSFER"}
    assert manifest["beta_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert manifest["alpha_grid"] == [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    assert manifest["retention_rule"]["pooled_s_all_rmse_improvement_min"] == 0.03


def test_shrinkage_selection_uses_held_out_oof_predictions():
    x = np.arange(10.0)[:, None]
    y = np.arange(10.0)
    source = np.zeros(10)
    groups = np.repeat([f"f{i}" for i in range(5)], 2)
    oof, folds = grouped_shrinkage_oof_predictions(x, y, source, groups, [0.0], 5)
    assert oof.shape == (10,)
    assert len(set(folds.values())) == 5


def test_oof_clip_bounds_do_not_use_held_out_truth(monkeypatch):
    import gaps_flower.canonical_r2_transfer as module
    observed = []
    real_fit = module.fit_ridge_model
    def spy(x, y, alpha, clip_min, clip_max):
        observed.append((float(clip_min), float(clip_max), float(np.min(y)), float(np.max(y))))
        return real_fit(x, y, alpha, clip_min, clip_max)
    monkeypatch.setattr(module, "fit_ridge_model", spy)
    x = np.arange(10.0)[:, None]
    y = np.arange(10.0)
    groups = np.repeat([f"f{i}" for i in range(5)], 2)
    grouped_shrinkage_oof_predictions(x, y, np.zeros(10), groups, [0.0], 5)
    assert all((lo, hi) == (train_lo, train_hi) for lo, hi, train_lo, train_hi in observed)
