from pathlib import Path
import json
import numpy as np
import pytest


def test_six_classification_tasks_include_ft_and_no_full():
    from gaps_flower.fedongas_extension import classification_matrix
    rows=classification_matrix()
    assert len(rows)==len({r['experiment_id'] for r in rows})==6
    assert [(r['target'],r['method']) for r in rows]==[(t,m) for t in ('C3','C4') for m in ('TO','FT','CLASS')]
    assert all(r['seed']==42 and r['target_updates']==2500 for r in rows)
    assert [r['rounds'] for r in rows]==[0,0,25,0,0,25]
    assert [r['calibration_n'] for r in rows]==[678]*3+[320]*3


def test_class_commands_keep_real_target_and_class_only_protocol():
    from gaps_flower.fedongas_extension import classification_matrix,build_class_commands
    for row in classification_matrix():
        if row['method']!='CLASS':continue
        c=build_class_commands(row);s=c['server']
        assert s[s.index('--server-calib-data')+1].endswith('/class/client_'+row['target'][1:])
        assert s[s.index('--da-lambda-stage-mmd')+1]=='0.0'
        assert s[s.index('--da-lambda-target-ce')+1]=='0.0'
        assert c['protocol']['target']==row['target']
        assert c['protocol']['adaptation_target_fields']==['x','class']
        assert c['protocol']['prototype_scope']=='class_only'
        for role in ('client_c1','client_c2'):
            assert c[role][c[role].index('--local-epochs')+1]=='1'
            assert c[role][c[role].index('--data-root')+1].endswith('/class')


def test_target_reader_accepts_c3_678_and_never_needs_phase_or_regression(tmp_path):
    from gaps_flower.fedongas_extension import target_xy
    np.save(tmp_path/'calibration_features.npy',np.zeros((678,50,8),np.float32))
    np.save(tmp_path/'calibration_classification_labels.npy',np.arange(678)%4)
    x,y=target_xy(tmp_path,'C3')
    assert x.shape==(678,50,8) and y.shape==(678,)
    with pytest.raises(ValueError,match='count'):
        target_xy(tmp_path,'C4')


def test_regression_matrix_only_adds_10_15_four_banks():
    from gaps_flower.fedongas_extension import regression_matrix,load_budget
    rows=regression_matrix()
    assert len(rows)==4 and {r['budget_pct'] for r in rows}=={10,15}
    a=load_budget(10);b=load_budget(15)
    assert len(a['truth'])==160 and len(b['truth'])==240
    assert all(a['truth']>0) and all(b['truth']>0)
    assert {m['physical_identity'] for m in a['metadata']} < {m['physical_identity'] for m in b['metadata']}


def test_endpoint_reuse_rejects_mismatched_method_or_failed_status(tmp_path):
    from gaps_flower.fedongas_extension import match_endpoint,classification_matrix
    from scripts.run_fedongas_regression_cross import sha
    row=classification_matrix()[0];cp=tmp_path/'final.bin';cp.write_bytes(b'fixed model')
    e={'run':row,'status':'PASS','target_updates':2500,'target_test_opened':False,
       'checkpoint':'final.bin','checkpoint_sha256':sha(cp),'freeze_commit':'abc'}
    (tmp_path/'FINAL_ENDPOINT.json').write_text(json.dumps(e))
    assert match_endpoint(tmp_path,row,'abc')['status']=='PASS'
    e['run']={**row,'method':'FT'};(tmp_path/'FINAL_ENDPOINT.json').write_text(json.dumps(e))
    with pytest.raises(RuntimeError,match='identity'):match_endpoint(tmp_path,row,'abc')
    e['run']=row;e['status']='FAILED';(tmp_path/'FINAL_ENDPOINT.json').write_text(json.dumps(e))
    with pytest.raises(RuntimeError,match='status'):match_endpoint(tmp_path,row,'abc')


def test_regression_test_gate_fails_before_all_four_banks_locked(tmp_path):
    from scripts.run_fedongas_budget_extension import release_saved_test
    with pytest.raises(RuntimeError,match='locked'):release_saved_test(tmp_path)
    assert not (tmp_path/'TEST_RELEASE.json').exists()


def test_target_initialization_rejects_source_for_to_and_random_for_ft():
    from gaps_flower.fedongas_extension_target import validate_initial
    validate_initial({'initialization':'random','seed':42},'TO')
    validate_initial({'round':25},'FT')
    with pytest.raises(RuntimeError):validate_initial({'round':25},'TO')
    with pytest.raises(RuntimeError):validate_initial({'initialization':'random','seed':42},'FT')


def test_classification_evaluation_stays_sealed_without_six_endpoints(tmp_path):
    from scripts.evaluate_fedongas_extension import all_endpoint_gate
    with pytest.raises(RuntimeError,match='missing endpoint'):all_endpoint_gate(tmp_path,'fixed')
    assert not (tmp_path/'evaluation').exists()


def test_device_comparisons_keep_negative_and_zero_gains():
    from scripts.evaluate_fedongas_extension import paired_device_rows
    rows=[{'target':t,'method':m,'accuracy':v,'macro_f1':v}
          for t in ('C3','C4','C5') for m,v in [('TO',.9),('FT',.85),('CLASS',.9)]]
    pairs=paired_device_rows(rows)
    assert len(pairs)==9
    assert all(abs(r['accuracy_delta']+.05)<1e-12 for r in pairs if r['comparison']=='FT-TO')
    assert all(r['accuracy_delta']==0 for r in pairs if r['comparison']=='CLASS-TO')
