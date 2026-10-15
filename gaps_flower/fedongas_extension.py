"""Frozen scope for C5 regression budgets and C3/C4 TO/FT/CLASS validation."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from scripts.run_fedongas_regression_cross import read_json,read_csv,sha,load_payload

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/experiments/c5_budget_c34_class_validation_20261015_v2'
CLASS_STUDY=ROOT/'results/fedongas_c34_class_validation_20261015'
REG_STUDY=ROOT/'results/fedongas_c5_regression_budget_extension_20261015'
OLD=ROOT/'results/fedongas_c5_budget_multiseed_20261014'
OLDREG=ROOT/'results/fedongas_c5_regression_cross_20261015'
DATA=ROOT/'dataset/iotj_canonical_v1'
BUDGET=ROOT/'results/iotj_canonical_v1_c5_budget_20260810'
REMOTE_DATA='/root/GAPS/fong_c34_inputs_20261015'
PI_DATA='/home/gaps/GAPS/fong_c34_inputs_20261015'
HOSTS=('root@121.40.139.213','gaps@192.168.137.172','root@114.55.171.63')
COUNTS={'C3':(678,2677),'C4':(320,1360)}


def classification_matrix():
    return [{'experiment_id':f'XVAL-{t}-{m}-MAIN-S42','target':t,'method':m,'seed':42,
             'budget_pct':20,'calibration_n':COUNTS[t][0],'test_n':COUNTS[t][1],
             'initialization_run':'FONG-SRC-S42' if m=='FT' else 'random-S42',
             'rounds':25 if m=='CLASS' else 0,'target_updates':2500}
            for t in ('C3','C4') for m in ('TO','FT','CLASS')]


def regression_matrix():
    return [{'experiment_id':f'XVAL-C5-{r}-B{b}','budget_pct':b,'regressor':r,
             'calibration_n':b*16,'test_n':1360,'final_gas_fits':4}
            for b in (10,15) for r in ('TARGET_RIDGE_83D','R84_CONCAT')]


def load_budget(budget):
    if budget not in (5,10,15,20):raise ValueError('Unregistered calibration budget')
    directory=BUDGET/f'budget_data/client_5_budget_{budget:02d}'
    expected=read_json(BUDGET/'c5_calibration_budget_manifest_sha256.json')['files']
    for f in directory.glob('calibration_*'):
        if sha(f)!=expected[f.relative_to(BUDGET).as_posix()]:raise RuntimeError('Frozen budget changed')
    rows=read_csv(BUDGET/f'c5_calibration_budget_{budget:02d}pct.csv')
    data=load_payload(directory,'calibration')
    if len(rows)!=budget*16 or [r['canonical_identity'] for r in rows]!=[m['physical_identity'] for m in data['metadata']]:
        raise RuntimeError('Budget identity/count mismatch')
    return data


def target_xy(directory,target):
    if target not in COUNTS:raise ValueError('Unregistered target')
    directory=Path(directory)
    x=np.load(directory/'calibration_features.npy',allow_pickle=False)
    y=np.load(directory/'calibration_classification_labels.npy',allow_pickle=False)
    if x.shape!=(COUNTS[target][0],50,8) or y.shape!=(len(x),):raise ValueError('Target calibration count/shape mismatch')
    if not np.isfinite(x).all() or not np.all((y>=0)&(y<=3)):raise ValueError('Target calibration invalid')
    return x.astype(np.float32),y.astype(np.int64)


def build_class_commands(row):
    from scripts.run_iotj_canonical_v1_classification import build_canonical_commands
    from gaps_flower.fedongas27_protocol import option
    if row not in classification_matrix() or row['method']!='CLASS':raise ValueError('Unregistered CLASS run')
    commands=copy.deepcopy(build_canonical_commands(row['target']))
    old_id=commands['protocol']['experiment_id']
    for role in ('server','client_c1','client_c2'):
        commands[role]=[v.replace(old_id,row['experiment_id']) for v in commands[role]]
        mod=commands[role].index('-m')+1
        commands[role][mod]='gaps_flower.fedongas27_entry'
        commands[role].insert(mod+1,'server' if role=='server' else 'client')
        base=PI_DATA if role=='client_c1' else REMOTE_DATA
        commands[role].extend(['--fong-method','CLASS','--initial-state',base+'/initial/random-S42.pth'])
        option(commands[role],'--seed',42)
        if role!='server':option(commands[role],'--data-root',base+'/class')
    option(commands['server'],'--server-val-data',REMOTE_DATA+'/class/client_1,'+REMOTE_DATA+'/class/client_2')
    option(commands['server'],'--server-calib-data',REMOTE_DATA+'/class/client_'+row['target'][1:])
    option(commands['server'],'--da-lambda-stage-mmd',0.0)
    commands['protocol'].update(row)
    commands['protocol'].update(study_id='FEDONGAS-C34-CLASS-VALIDATION-20261015',
        classifier_router='CLASS',prototype_scope='class_only',initialization='common_random_seed42',
        adaptation_target_fields=['x','class'])
    return commands


def match_endpoint(directory,row,freeze_commit):
    directory=Path(directory);e=read_json(directory/'FINAL_ENDPOINT.json')
    if e.get('run')!=row or e.get('freeze_commit')!=freeze_commit:raise RuntimeError('Endpoint identity/protocol mismatch')
    if e.get('status')!='PASS':raise RuntimeError('Endpoint status not PASS')
    if e.get('target_updates')!=2500 or e.get('target_test_opened') is not False:raise RuntimeError('Endpoint updates/test access differs')
    if sha(directory/e['checkpoint'])!=e['checkpoint_sha256']:raise RuntimeError('Endpoint checkpoint changed')
    return e
