"""Register six C3/C4 endpoints and four C5 Ridge banks without training."""
from __future__ import annotations
import sys,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from gaps_flower.fedongas_extension import *
from scripts.run_fedongas_regression_cross import write_json,write_csv,sha,read_json,PRIOR,PRIOR_SHA


def inspect():
    from tools.verify_iotj_canonical_v1_hashes import verify
    from gaps_flower.canonical_r1_v1 import validate_r0_bundle,validate_evidence_bundle
    dataset=verify(DATA)
    if dataset['status']!='PASS':raise RuntimeError('Canonical data changed')
    hashes={(DATA/k).relative_to(ROOT).as_posix():sha(DATA/k) for k in read_json(DATA/'dataset_sha256.json')['files']}
    identities=[]
    for b in (5,10,15,20):
        p=load_budget(b);identities.append({m['physical_identity'] for m in p['metadata']})
        for f in (BUDGET/f'budget_data/client_5_budget_{b:02d}').glob('calibration_*'):
            hashes[f.relative_to(ROOT).as_posix()]=sha(f)
        f=BUDGET/f'c5_calibration_budget_{b:02d}pct.csv';hashes[f.relative_to(ROOT).as_posix()]=sha(f)
    if not identities[0]<identities[1]<identities[2]<identities[3]:raise RuntimeError('Budgets not nested')
    validate_r0_bundle(PRIOR,expected_index_sha256=PRIOR_SHA)
    previous=read_json(OLDREG/'FINAL_EVIDENCE_SHA256.json')['files']
    for rel,h in previous.items():
        if sha(OLDREG/rel)!=h:raise RuntimeError('Old regression evidence changed')
        hashes[(OLDREG/rel).relative_to(ROOT).as_posix()]=h
    prior_model=PRIOR/'model_lock.json';hashes[prior_model.relative_to(ROOT).as_posix()]=sha(prior_model)
    # The C5 rows/predictions must still match the already frozen input lock,
    # not merely acquire a new hash from whatever is on disk today.
    inherited=read_json(OLDREG/'input_lock.json')['input_hashes']
    for rel,h in inherited.items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Inherited C5/source asset changed: '+rel)
        hashes[rel]=h
    old_r1=ROOT/'results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812'
    validate_evidence_bundle(old_r1)
    for name in ('canonical_regression_comparison.csv','canonical_regression_per_gas.csv','sha256_index.json'):
        hashes[(old_r1/name).relative_to(ROOT).as_posix()]=sha(old_r1/name)
    src=read_json(OLD/'runs/FONG-SRC-S42/FINAL_ENDPOINT.json')
    cp=OLD/'runs/FONG-SRC-S42'/src['checkpoint']
    if src['status']!='PASS' or src['run']['method']!='SRC' or src['run']['rounds']!=25 or sha(cp)!=src['checkpoint_sha256']:
        raise RuntimeError('Matching SRC42 unavailable')
    initial=OLD/'inputs/data/initial/random-S42.pth'
    for f in (cp,initial,OLD/'runs/FONG-SRC-S42/FINAL_ENDPOINT.json'):
        hashes[f.relative_to(ROOT).as_posix()]=sha(f)
    class5=OLD/'runs/FONG-CLASS-B20-S42';e=read_json(class5/'FINAL_ENDPOINT.json')
    if e['status']!='PASS' or sha(class5/e['checkpoint'])!=e['checkpoint_sha256']:raise RuntimeError('C5 CLASS endpoint changed')
    for f in [class5/'FINAL_ENDPOINT.json',OLD/'evaluation/target_results_24.csv']+[class5/'remote_server'/f'server_round_{r:03d}_adapted.pth' for r in range(1,26)]:
        hashes[f.relative_to(ROOT).as_posix()]=sha(f)
    return {'status':'PASS','dataset':dataset,'input_hashes':hashes,
            'common_initial_fingerprint':read_json(OLD/'PREFLIGHT.json')['initial_fingerprints']['42'],
            'source_checkpoint':cp.relative_to(ROOT).as_posix(),'source_state_fingerprint':src['state_fingerprint'],
            'source_checkpoint_sha256':src['checkpoint_sha256'],
            'reused_classification_endpoints':[],
            'reuse_note':'No matching C3/C4 TO/FT/CLASS endpoints located; legacy A4/A0T/100-step DA excluded',
            'C3_counts':[678,2677],'C4_counts':[320,1360],'c5_nested_counts':[80,160,240,320]}


def prepare():
    if CLASS_STUDY.exists() or (DOC/'protocol_manifest.json').exists():raise FileExistsError('Preserve prepared study')
    from scripts.prepare_fedongas27 import copy_source_view
    info=inspect()
    root=CLASS_STUDY/'inputs/data';root.mkdir(parents=True)
    for c in (1,2):copy_source_view(DATA/f'client_{c}',root/f'class/client_{c}',class_only=True)
    for target in ('C3','C4'):
        source=DATA/f'client_{target[1:]}'
        for scope in ('class','target_xy'):
            dest=root/scope/f'client_{target[1:]}';dest.mkdir(parents=True)
            for name in ('features','classification_labels'):shutil.copy2(source/f'calibration_{name}.npy',dest/f'calibration_{name}.npy')
            if scope=='class':
                np.save(dest/'calibration_phase_labels.npy',np.zeros(COUNTS[target][0],np.int64))
                np.save(dest/'calibration_regression_labels.npy',np.zeros((COUNTS[target][0],4),np.float32))
    initial=root/'initial';initial.mkdir()
    shutil.copy2(OLD/'inputs/data/initial/random-S42.pth',initial/'random-S42.pth')
    shutil.copy2(ROOT/info['source_checkpoint'],initial/'source-S42.pth')
    data_hashes={f.relative_to(root).as_posix():sha(f) for f in root.rglob('*') if f.is_file()}
    write_json(root/'DATA_SHA256.json',data_hashes)
    archive=CLASS_STUDY/'inputs/data.tar'
    with tarfile.open(archive,'w') as tar:
        for f in sorted(root.rglob('*')):
            if f.is_file():tar.add(f,arcname=f.relative_to(root).as_posix())
    info['data_archive_sha256']=sha(archive)
    DOC.mkdir(parents=True,exist_ok=True)
    spec={'study_id':'FEDONGAS-C34-CLASS-BUDGET-EXTENSION-20261015','authorization':'2026-10-15 user: TO FT CLASS per C3/C4 seed42; no FULL',
          'classification_matrix':classification_matrix(),'regression_matrix':regression_matrix(),
          'final_candidate':'CLASS','classification_seed':42,'source_rounds':25,'local_epochs':1,
          'batch_size':32,'optimizer':'Adam','lr':5e-4,'target_updates':2500,'target_blocks':25,
          'target_optimizer_reset_every':100,'CLASS_target_CE':0,'TO_FT_source_CE':False,
          'CLASS_prototypes':4,'CLASS_true_phase_access':False,'QC':'none','hyperparameter_search':False,
          'regression_alpha_candidates':[0,.01,.1,1,10,100,1000],'regression_CV':'5fold raw filename grouped calibration only',
          'regression_source_prior':'unchanged frozen canonical R0-v2 H1, including existing clipping',
          'regression_primary_scope':'Oracle_ALL; all1360test and340pergas','curve':'CLASS main-calibration seed42 round1..25 diagnostic, fixed endpoint25',
          'test_selection':False,'test_release':'all six classification endpoints validated; all four new Ridge banks locked',
          'inputs':info}
    write_json(DOC/'protocol_manifest.json',spec)
    write_csv(DOC/'CLASSIFICATION_MATRIX.csv',classification_matrix())
    write_csv(DOC/'REGRESSION_MATRIX.csv',regression_matrix())
    write_json(CLASS_STUDY/'PREFLIGHT.json',info)
    write_json(CLASS_STUDY/'protocol_manifest.json',spec)
    write_json(CLASS_STUDY/'RUN_PROGRESS.json',{'status':'PREPARED_NOT_TRAINING','total':6,'completed':[]})
    print('PREPARED: six classification endpoints; four new regression banks; zero formal training so far',flush=True)


if __name__=='__main__':prepare()
