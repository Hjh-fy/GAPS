"""Contracts for cached-router x shared quantitative-regression evaluation."""
from collections import Counter
import numpy as np
import pytest


def test_matrix_contains_36_routes_and_only_four_shared_regression_banks():
    from gaps_flower.fedongas_regression_cross import matrix
    rows=matrix()
    assert len(rows)==len({r['experiment_id'] for r in rows})==36
    assert Counter(r['classifier'] for r in rows)=={'TO':12,'FT':12,'CLASS':12}
    assert len({r['regression_bank_id'] for r in rows})==4
    assert len({r['classifier_run'] for r in rows})==18
    assert all(r['neural_training'] is False for r in rows)


def test_concentration_targets_reject_classifier_placeholder_zeros():
    from gaps_flower.fedongas_regression_cross import validate_targets
    cls=np.array([0,1,2,3]); meta=[{'class_id':g,'concentration':10.+g} for g in range(4)]
    y=np.diag([10.,11.,12.,13.])
    np.testing.assert_array_equal(validate_targets(y,cls,meta),[10,11,12,13])
    with pytest.raises(ValueError,match='concentration'):
        validate_targets(np.zeros((4,4)),cls,meta)


def test_cached_routes_are_ordered_and_duplicate_rows_fail():
    from gaps_flower.fedongas_regression_cross import routes_from_cache
    rows=[{'experiment_id':'X','window_index':1,'true_class':1,'predicted_class':0},
          {'experiment_id':'X','window_index':0,'true_class':0,'predicted_class':1}]
    np.testing.assert_array_equal(routes_from_cache(rows,'X',np.array([0,1])),[1,0])
    with pytest.raises(ValueError,match='identity'):
        routes_from_cache(rows+[rows[0]],'X',np.array([0,1]))


def test_feature_extractor_does_not_encode_concentration_or_true_class():
    from gaps_flower.fedongas_regression_cross import quantitative_features
    x=np.arange(400,dtype=float).reshape(1,50,8)/400
    a={'filename':'A','class_id':0,'concentration':12.5}
    b={**a,'class_id':3,'concentration':250.}
    sa,ha=quantitative_features(x,np.array([2]),[a])
    sb,hb=quantitative_features(x,np.array([2]),[b])
    assert sa.shape==(1,83) and ha.shape==(1,104)
    np.testing.assert_array_equal(sa,sb); np.testing.assert_array_equal(ha,hb)


def test_shared_bank_fit_uses_calibration_only_and_saves_group_folds():
    from gaps_flower.fedongas_regression_cross import fit_bank, predict_bank
    cls=np.repeat(np.arange(4),10); y=10+np.tile(np.arange(10),4)
    sensor=np.column_stack([y,y*y]); prior=np.column_stack([y+g for g in range(4)])
    groups=np.array([f'gas{c}-file{i}' for c in range(4) for i in range(10)])
    bank, audit=fit_bank(sensor,prior,y,cls,groups,'R84_CONCAT')
    assert len(bank)==len(audit)==4
    assert all(len(r['fold_by_group'])==10 for r in audit)
    assert all(len(bank[g]['mean'])==3 for g in range(4))
    pred=predict_bank(bank,sensor,prior,'R84_CONCAT')
    assert pred.shape==(40,4) and np.all(np.isfinite(pred))


def test_s_all_keeps_wrong_routes_and_uses_true_gas_ranges():
    from gaps_flower.fedongas_regression_cross import scope_rows
    truth=np.array([10.,20.,100.,200.]); cls=np.array([0,0,1,1]); route=np.array([0,1,1,1])
    predictions=np.array([[10,100],[20,200],[10,100],[20,200]],float)
    overall,gas=scope_rows(truth,cls,route,predictions,{0:10.,1:100.})
    main=next(r for r in overall if r['scope']=='S_ALL')
    oracle=next(r for r in overall if r['scope']=='Oracle_ALL')
    assert main['n']==4 and main['RMSE']==90. and main['NRMSE_range']==9.
    assert oracle['RMSE']==0 and oracle['n']==4
    assert next(r for r in overall if r['scope']=='S_CC')['n']==3
    assert len([r for r in gas if r['scope']=='S_ALL'])==2


def test_paired_results_preserve_negative_zero_and_match_classifier_seed():
    from gaps_flower.fedongas_regression_cross import paired_rows
    rows=[]
    for b in (5,20):
        for s in (42,43,44):
            for c in ('TO','FT','CLASS'):
                for reg,v in (('TARGET_RIDGE_83D',10.),('R84_CONCAT',11. if c=='TO' else 9.)):
                    rows.append({'classifier':c,'regressor':reg,'budget_pct':b,'seed':s,
                                 'RMSE':v,'MAE':v/2,'NRMSE_range':v/100})
    paired=paired_rows(rows)
    assert len(paired)==66
    assert all(r['RMSE_delta']==1 for r in paired if r['comparison']=='TO:R84_CONCAT-TO:TARGET_RIDGE_83D')
    assert all(r['RMSE_delta']==0 for r in paired if r['comparison']=='CLASS:TARGET_RIDGE_83D-FT:TARGET_RIDGE_83D')
