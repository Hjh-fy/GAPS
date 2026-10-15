"""Fixed cached-router x canonical quantitative Ridge study; no neural training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from gaps_flower.fedongas_regression_cross import (
    matrix, validate_targets, routes_from_cache, quantitative_features, fit_bank,
    predict_bank, scope_rows, paired_rows, summarize, ALPHAS, GASES, RANGES,
)
from gaps_flower.canonical_r1_v1 import validate_r0_bundle, predict_ridge_model
from gaps_flower.canonical_quantitative_features import H1_FEATURE_NAMES

STUDY='fedongas_c5_regression_cross_20261015'
OUT=ROOT/'results'/STUDY
DOC=ROOT/'docs/experiments'/STUDY
OLD=ROOT/'results/fedongas_c5_budget_multiseed_20261014'
BUDGET=ROOT/'results/iotj_canonical_v1_c5_budget_20260810'
DATA=ROOT/'dataset/iotj_canonical_v1/client_5'
PRIOR=ROOT/'results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812'
PRIOR_SHA='04e62b0a2b363fcb79cbeb60ad6c2c6c9a3dedef380b2b0f7c2aa808732d4d82'
CACHE_SHA='150694bdf5df22c164d939a754460b68a2ea3ee7c5ad1d9be186332c1ba7c5de'
CRITICAL=[f'gaps_flower/fedongas_regression_cross.py',f'scripts/run_fedongas_regression_cross.py',
          'gaps_flower/canonical_r1_v1.py','gaps_flower/canonical_quantitative_features.py',
          'run_regression_head_ablation.py',f'docs/experiments/{STUDY}/protocol_manifest.json',
          f'docs/experiments/{STUDY}/EXPERIMENT_MATRIX.csv',f'docs/experiments/{STUDY}/EXPERIMENT_PLAN.md']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path,rows):
    if not rows: raise ValueError('Empty result table')
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


def write_text(path,text):
    with Path(path).open('w',encoding='utf-8',newline='\n') as f: f.write(text)


def protocol_spec():
    return {'study_id':STUDY,'neural_training':False,'pipelines':36,'shared_banks':4,'final_gas_fits':16,
            'classifiers':['TO','FT','CLASS'],'regressors':['TARGET_RIDGE_83D','R84_CONCAT'],
            'budgets':[5,20],'calibration_n':[80,320],'seeds':[42,43,44],'test_n':1360,
            'alpha_candidates':list(ALPHAS),'alpha_selection':'fivefold_raw_filename_grouped_calibration_pooled_RMSE_first_tie',
            'source_prior_index_sha256':PRIOR_SHA,'prediction_cache_sha256':CACHE_SHA,
            'source_prior':'frozen_C1_C2_canonical_H1_FedRidge; reused across all classifier seeds',
            'features':'fresh_fixed_5Hz_50x8_sensor83; H1_104_with_protocol_metadata',
            'solver':'numpy.linalg.pinv; float64; fold-local standardization; unregularized intercept',
            'clipping':'per-gas target calibration min/max; same rule for both regressors',
            'primary_scope':'S_ALL predicted-class routing, all test samples',
            'diagnostic_scopes':['S_CC','Oracle_ALL','Oracle_CC'],'gas_ranges':{str(g):v for g,v in RANGES.items()},
            'overall_NRMSE':'sqrt(mean((prediction_error/range_true_gas)^2))',
            'QC':'none','sample_sd_ddof':1,'significance_claim':False,
            'training_concentrations':'original canonical budget regression_labels, not classifier zero placeholders',
            'stopping_rule':'evaluate exactly 36 fixed pipelines; report all positive/zero/negative outcomes; stop'}


def load_payload(directory,split):
    directory=Path(directory)
    x=np.load(directory/f'{split}_features.npy',allow_pickle=False)
    classes=np.load(directory/f'{split}_classification_labels.npy',allow_pickle=False)
    phases=np.load(directory/f'{split}_phase_labels.npy',allow_pickle=False)
    y4=np.load(directory/f'{split}_regression_labels.npy',allow_pickle=False)
    metadata=read_json(directory/f'{split}_experiment_info.json')
    truth=validate_targets(y4,classes,metadata)
    if x.shape!=(len(classes),50,8) or not np.isfinite(x).all(): raise ValueError('Canonical input mismatch')
    return {'x':x,'classes':classes,'phases':phases,'truth':truth,'metadata':metadata}


def load_calibration(budget):
    if budget not in (5,20): raise ValueError('Unregistered budget')
    payload=load_payload(BUDGET/f'budget_data/client_5_budget_{budget:02d}','calibration')
    rows=read_csv(BUDGET/f'c5_calibration_budget_{budget:02d}pct.csv')
    if [m['physical_identity'] for m in payload['metadata']] != [r['canonical_identity'] for r in rows]:
        raise RuntimeError('Budget row identity mismatch')
    if len(rows)!=(80 if budget==5 else 320): raise RuntimeError('Budget size differs')
    return payload


def inspect_inputs():
    from scripts.prepare_fedongas27 import inspect_inputs as original_inspect
    base=original_inspect()
    hashes=dict(base['input_hashes'])
    for budget in (5,20):
        load_calibration(budget)
        for p in (BUDGET/f'budget_data/client_5_budget_{budget:02d}').glob('calibration_*'):
            hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in DATA.glob('test_*'):
        if p.is_file(): hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    models=validate_r0_bundle(PRIOR,expected_index_sha256=PRIOR_SHA)
    if any(list(m['feature_names'])!=list(H1_FEATURE_NAMES) for m in models.values()):
        raise RuntimeError('Frozen H1 feature order differs')
    for name in ('sha256_index.json','model_lock.json','source_alpha_lock.json'):
        hashes[(PRIOR/name).relative_to(ROOT).as_posix()]=sha(PRIOR/name)
    cache=OLD/'evaluation/predictions.csv'
    if sha(cache)!=CACHE_SHA: raise RuntimeError('Classifier prediction cache hash mismatch')
    previous_index=read_json(OLD/'evaluation/sha256_index.json')
    targets=OLD/'evaluation/target_results_24.csv'
    if sha(targets)!=previous_index['target_results_24.csv']: raise RuntimeError('Classifier metrics changed')
    hashes[cache.relative_to(ROOT).as_posix()]=sha(cache)
    hashes[targets.relative_to(ROOT).as_posix()]=sha(targets)
    classifier_lock={}
    metrics={r['experiment_id']:r for r in read_csv(targets)}
    for runid in sorted({r['classifier_run'] for r in matrix()}):
        endpoint=OLD/'runs'/runid/'FINAL_ENDPOINT.json'; e=read_json(endpoint)
        checkpoint=endpoint.parent/e['checkpoint']; digest=sha(checkpoint)
        if e['status']!='PASS' or digest!=e['checkpoint_sha256'] or digest!=metrics[runid]['checkpoint_sha256']:
            raise RuntimeError(f'Classifier endpoint mismatch: {runid}')
        hashes[checkpoint.relative_to(ROOT).as_posix()]=digest
        hashes[endpoint.relative_to(ROOT).as_posix()]=sha(endpoint)
        classifier_lock[runid]={'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'sha256':digest,
                                'ordered_state_fingerprint':e['state_fingerprint'],
                                'endpoint':endpoint.relative_to(ROOT).as_posix(),
                                'endpoint_sha256':sha(endpoint)}
    return {'status':'PASS','input_hashes':hashes,'classifiers':classifier_lock,
            'budget_nesting':base['status'],'test_n':1360}


def prepare():
    if (DOC/'protocol_manifest.json').exists(): raise FileExistsError('Protocol already frozen; preserve it')
    check=inspect_inputs()
    DOC.mkdir(parents=True,exist_ok=True)
    write_csv(DOC/'EXPERIMENT_MATRIX.csv',matrix())
    write_json(DOC/'protocol_manifest.json',{**protocol_spec(),'inputs':check})
    print(json.dumps({'status':'PREPARED','pipelines':36,'classifier_endpoints':len(check['classifiers'])}))


def assert_hashes(hashes):
    for rel,digest in hashes.items():
        if sha(ROOT/rel)!=digest: raise RuntimeError(f'Frozen input changed: {rel}')


def release_test(out):
    out=Path(out)
    if not all((out/n).is_file() for n in ('target_model_lock.json','input_lock.json','target_alpha_lock.json')):
        raise RuntimeError('Target test remains locked: missing locks')
    lock=read_json(out/'target_model_lock.json'); inputs=read_json(out/'input_lock.json')
    expected={r['regression_bank_id'] for r in matrix()}
    if set(lock.get('banks',{}))!=expected or len(inputs.get('classifiers',{}))!=18:
        raise RuntimeError('Target test remains locked: incomplete models/classifiers')
    alphas=read_json(out/'target_alpha_lock.json')
    if set(alphas)!=expected: raise RuntimeError('Target test remains locked: alpha coverage')
    for bank in lock['banks'].values():
        if set(bank)!={'0','1','2','3'}: raise RuntimeError('Target test remains locked: gas coverage')
    assert_hashes(inputs['input_hashes'])
    receipt={'locks_sha256':{n:sha(out/n) for n in ('target_model_lock.json','target_alpha_lock.json','input_lock.json')},
             'test_opened_after_all_locks':True,'new_test_feature_loads':1,
             'note':'test classification was previously evaluated; no regression test values used for fitting/selection'}
    write_json(out/'TEST_RELEASE.json',receipt)
    return load_payload(DATA,'test')


def make_report(out,main,summary,gas_summary,paired_summary,oracle):
    lines=['# C5 分类器×回归方式交叉评价：36/36完成','',
           '主结果：预测类别路由、全部1360个测试窗口；mean ± sample SD（3个分类训练seed）。RMSE/MAE单位ppm，NRMSE无量纲。',
           '4套回归bank共享，不是36次回归独立重复，更没有新增神经网络训练。','',
           '|预算|分类器|回归|分类Acc %|RMSE|MAE|NRMSE|',
           '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        selected=[m for m in main if all(m[k]==r[k] for k in ('budget_pct','classifier','regressor'))]
        acc=np.array([m['classification_accuracy'] for m in selected])*100
        lines.append(f"|{r['budget_pct']}%|{r['classifier']}|{r['regressor']}|{acc.mean():.3f} ± {acc.std(ddof=1):.3f}|"
                     + '|'.join(f"{r[k+'_mean']:.5f} ± {r[k+'_sd']:.5f}" for k in ('RMSE','MAE','NRMSE_range'))+'|')
    lines+=['','## 预注册配对差值','',
            '左减右；负值=误差降低。均值与sample SD仅描述三个分类训练seed，不证明统计显著。','',
            '|预算|比较|ΔRMSE ppm|ΔMAE ppm|ΔNRMSE|','|---|---|---:|---:|---:|']
    for r in paired_summary:
        lines.append(f"|{r['budget_pct']}%|{r['comparison']}|"+'|'.join(
            f"{r[k+'_delta_mean']:+.5f} ± {r[k+'_delta_sd']:.5f}" for k in ('RMSE','MAE','NRMSE_range'))+'|')
    lines+=['','## 对研究问题的直接回答','']
    for b in (5,20):
        rows=[r for r in paired_summary if r['budget_pct']==b]
        source=[r for r in rows if ':R84_CONCAT-' in r['comparison'] and r['comparison'].split('-')[0].split(':')[0]==r['comparison'].split('-')[1].split(':')[0]]
        improved=[r['comparison'].split(':')[0] for r in source if r['RMSE_delta_mean']<0]
        lines.append(f"- {b}%预算：同一分类器下，R84总体RMSE改善的分类器为 {', '.join(improved) or '无'}（{len(improved)}/3）。")
        for reg in ('TARGET_RIDGE_83D','R84_CONCAT'):
            subset=[r for r in summary if r['budget_pct']==b and r['regressor']==reg]
            # Mean classification rank, not the best seed.
            acc_order=sorted(subset,key=lambda r:-np.mean([m['classification_accuracy'] for m in main if m['classifier']==r['classifier'] and m['regressor']==reg and m['budget_pct']==b]))
            rmse_order=sorted(subset,key=lambda r:r['RMSE_mean'])
            lines.append(f"- {b}% / {reg}：平均Accuracy排序 {' > '.join(r['classifier'] for r in acc_order)}；平均RMSE由低到高 {' < '.join(r['classifier'] for r in rmse_order)}。")
        for name in ('FT:R84_CONCAT-TO:TARGET_RIDGE_83D','CLASS:R84_CONCAT-TO:TARGET_RIDGE_83D'):
            d=next(r for r in rows if r['comparison']==name)
            lines.append(f"- {b}% / {name}：ΔRMSE={d['RMSE_delta_mean']:+.5f} ppm；{'支持该固定协议下总体误差改善' if d['RMSE_delta_mean']<0 else '不支持相对TO+目标Ridge的总体误差优势'}。")
    lines+=['','## 按气体结果','',
            '|预算|分类器|回归|气体|RMSE ppm|MAE ppm|NRMSE|','|---|---|---|---|---:|---:|---:|']
    for r in gas_summary:
        lines.append(f"|{r['budget_pct']}%|{r['classifier']}|{r['regressor']}|{r['gas']}|"+'|'.join(
            f"{r[k+'_mean']:.5f} ± {r[k+'_sd']:.5f}" for k in ('RMSE','MAE','NRMSE_range'))+'|')
    lines+=['','## 路由与回归映射的边界','',
            '同预算/回归方式的Oracle_ALL在18个分类路由中对应值严格相同。分类器差异只改变路由，不改变共享回归映射。',
            'S_CC与Oracle_CC完全相同（正确路由子集）；不同分类器的S_CC子集不同，不能据此声称回归映射改善。',
            'Oracle_ALL仅4个唯一bank，不能把重复的oracle结果作为3个独立回归seed计算显著性。',
            'routing_excess_MSE = MSE(S_ALL) − MSE(Oracle_ALL) 是同一全集上的诊断；可能为负，错误路由偶然数值接近不等于语义正确。',
            '浓度评价为预测气体通路的标量ppm误差，不惩罚气体身份本身；应同时阅读Accuracy与浓度误差，不以浓度误差取代分类正确性。','',
            '## 限制和停止','',
            '冻结source prior使用104D H1（包括21维协议/phase元数据）；源增强效应不能单独归因于纯浓度知识。',
            '5%/20%固定窗口协议与原论文一致，未新增独立raw-file/设备划分；CV虽按raw filename分组，不改变最终测试独立性局限。',
            '三seed只改变分类训练，回归bank及其alpha固定共享；没有新增bootstrap/p值、FULL、QC、其他target、训练或算法搜索。',
            '结果保留负收益与气体异质性，不宣称完整FedOnGas/FULL优越；完成后停止。','']
    write_text(out/'CLASSIFIER_REGRESSION_REPORT.md','\n'.join(lines))


def run(freeze_commit):
    if OUT.exists(): raise FileExistsError(f'Preserve existing endpoint: {OUT}; use verify, never rerun')
    subprocess.run(['git','cat-file','-e',freeze_commit+'^{commit}'],cwd=ROOT,check=True)
    diff=subprocess.check_output(['git','diff',freeze_commit,'--',*CRITICAL],cwd=ROOT)
    if diff: raise RuntimeError('Critical code/protocol differs from pre-run freeze')
    registered=read_json(DOC/'protocol_manifest.json')
    if {k:registered[k] for k in protocol_spec()}!=protocol_spec(): raise RuntimeError('Protocol mismatch')
    check=inspect_inputs()
    if check!=registered['inputs']: raise RuntimeError('Registered immutable input lock differs')
    source=validate_r0_bundle(PRIOR,expected_index_sha256=PRIOR_SHA)
    OUT.mkdir(parents=True)
    start=time.perf_counter()
    write_text(OUT/'.gitattributes','*.csv -text\n*.json -text\n*.md -text\n')
    write_json(OUT/'input_lock.json',check)
    write_json(OUT/'PRE_RUN_FREEZE.json',{'commit':freeze_commit,'protocol_sha256':sha(DOC/'protocol_manifest.json'),
               'critical_code_sha256':{rel:sha(ROOT/rel) for rel in CRITICAL},'python':sys.version,'numpy':np.__version__,
               'platform':platform.platform(),'neural_training':False})
    write_json(OUT/'RUN_PROGRESS.json',{'status':'FITTING_CALIBRATION','completed':0,'total':36})
    banks,alpha_lock,feature_audit={},{},{}
    for b in (5,20):
        payload=load_calibration(b)
        sensor,h1=quantitative_features(payload['x'],payload['phases'],payload['metadata'])
        prior=np.column_stack([predict_ridge_model(source[g],h1) for g in range(4)])
        np.savez_compressed(OUT/f'calibration_B{b:02d}_features.npz',sensor83=sensor,h1=h1,source_prior=prior)
        feature_audit[str(b)]={'n':len(sensor),'sensor_dimensions':83,'H1_dimensions':104,
                              'truth_min':float(payload['truth'].min()),'truth_max':float(payload['truth'].max())}
        groups=np.array([m['filename'] for m in payload['metadata']])
        for reg in ('TARGET_RIDGE_83D','R84_CONCAT'):
            bank_id=f'{reg}-B{b:02d}'
            banks[bank_id],alpha_lock[bank_id]=fit_bank(sensor,prior,payload['truth'],payload['classes'],groups,reg)
            print(f'FITTED {bank_id}: '+str([r['alpha'] for r in alpha_lock[bank_id]]),flush=True)
    write_json(OUT/'target_alpha_lock.json',alpha_lock)
    write_json(OUT/'target_model_lock.json',{'banks':banks,'shared_across_classifier_and_seed':True,'fits':16})
    write_json(OUT/'FEATURE_AUDIT.json',feature_audit)
    test=release_test(OUT)
    sensor,h1=quantitative_features(test['x'],test['phases'],test['metadata'],split='test')
    prior=np.column_stack([predict_ridge_model(source[g],h1) for g in range(4)])
    np.savez_compressed(OUT/'test_features.npz',sensor83=sensor,h1=h1,source_prior=prior)
    bank_predictions={k:predict_bank(v,sensor,prior,k.rsplit('-B',1)[0]) for k,v in banks.items()}
    cached=read_csv(OLD/'evaluation/predictions.csv')
    old_metrics={r['experiment_id']:r for r in read_csv(OLD/'evaluation/target_results_24.csv')}
    all_scope,gas_scope,main,gas_main,window_rows,oracle=[],[],[],[],[],[]
    for r in matrix():
        route=routes_from_cache(cached,r['classifier_run'],test['classes'])
        acc=float(np.mean(route==test['classes']))
        if acc!=float(old_metrics[r['classifier_run']]['accuracy']): raise RuntimeError('Cached classification differs')
        pred=bank_predictions[r['regression_bank_id']]
        scopes,gases=scope_rows(test['truth'],test['classes'],route,pred)
        meta={**r,'classification_accuracy':acc,
              'checkpoint_sha256':check['classifiers'][r['classifier_run']]['sha256'],
              'classifier_prediction_sha256':CACHE_SHA,'source_model_sha256':sha(PRIOR/'model_lock.json'),
              'target_model_lock_sha256':sha(OUT/'target_model_lock.json')}
        sr=[{**meta,**s} for s in scopes]; gr=[{**meta,**g} for g in gases]
        all_scope+=sr; gas_scope+=gr
        s=next(x for x in sr if x['scope']=='S_ALL'); o=next(x for x in sr if x['scope']=='Oracle_ALL')
        s['routing_excess_MSE']=s['RMSE']**2-o['RMSE']**2
        main.append(s); gas_main += [g for g in gr if g['scope']=='S_ALL']
        if not any(x['regression_bank_id']==r['regression_bank_id'] for x in oracle):
            oracle.append({'regression_bank_id':r['regression_bank_id'],'budget_pct':r['budget_pct'],
                           'regressor':r['regressor'],**{k:o[k] for k in ('n','RMSE','MAE','NRMSE_range')}})
        tc=test['classes']
        for i,m in enumerate(test['metadata']):
            window_rows.append({'experiment_id':r['experiment_id'],'window_index':i,
                'physical_identity':m['physical_identity'],'filename':m['filename'],'true_class':int(tc[i]),
                'predicted_class':int(route[i]),'truth':float(test['truth'][i]),'gas_range':RANGES[int(tc[i])],
                'prediction':float(pred[i,route[i]]),'oracle_prediction':float(pred[i,tc[i]]),
                **{f'route_{g}':float(pred[i,g]) for g in range(4)}})
    summary=summarize(main,['budget_pct','classifier','regressor'])
    gs=summarize(gas_main,['budget_pct','classifier','regressor','gas'])
    pairs=paired_rows(main); ps=summarize(pairs,['budget_pct','comparison'],tuple(k+'_delta' for k in ('RMSE','MAE','NRMSE_range')))
    gp=[{'gas':GASES[g],**r} for g in range(4) for r in paired_rows([v for v in gas_main if v['gas_id']==g])]
    gps=summarize(gp,['budget_pct','gas','comparison'],tuple(k+'_delta' for k in ('RMSE','MAE','NRMSE_range')))
    for name,rows in [('regression_results_36.csv',main),('regression_summary_12.csv',summary),
                      ('per_gas_results_144.csv',gas_main),('per_gas_summary_48.csv',gs),
                      ('routing_scope_summary.csv',all_scope),('per_gas_routing_scopes.csv',gas_scope),
                      ('paired_deltas_66.csv',pairs),('paired_summary_22.csv',ps),
                      ('per_gas_paired_deltas.csv',gp),('per_gas_paired_summary.csv',gps),
                      ('oracle_unique_banks.csv',oracle),('predictions.csv',window_rows)]: write_csv(OUT/name,rows)
    make_report(OUT,main,summary,gs,ps,oracle)
    assert_hashes(check['input_hashes'])
    write_json(OUT/'RUN_PROGRESS.json',{'status':'EVALUATED','completed':36,'total':36,'classifier_training_runs':0,
               'final_gas_ridge_fits':16,'wall_seconds':time.perf_counter()-start})
    write_json(OUT/'sha256_index.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    audit=verify()
    write_json(OUT/'COMPLETE.json',{'status':'PASS','pipelines':36,'freeze_commit':freeze_commit,
               'evidence_index_sha256':sha(OUT/'sha256_index.json'),'audit':audit})
    print(json.dumps(audit),flush=True)


def verify():
    index=read_json(OUT/'sha256_index.json')
    for rel,digest in index.items():
        if sha(OUT/rel)!=digest: raise RuntimeError(f'Result hash mismatch: {rel}')
    inputs=read_json(OUT/'input_lock.json'); assert_hashes(inputs['input_hashes'])
    receipt=read_json(OUT/'TEST_RELEASE.json')
    if not receipt['test_opened_after_all_locks']: raise RuntimeError('Test chronology invalid')
    for n,digest in receipt['locks_sha256'].items():
        if sha(OUT/n)!=digest: raise RuntimeError('Post-release lock mutation')
    main=read_csv(OUT/'regression_results_36.csv'); scopes=read_csv(OUT/'routing_scope_summary.csv')
    pred=read_csv(OUT/'predictions.csv')
    if len(main)!=36 or {r['experiment_id'] for r in main}!={r['experiment_id'] for r in matrix()}:
        raise RuntimeError('Result matrix incomplete')
    if len(scopes)!=144 or len(pred)!=48960: raise RuntimeError('Prediction/scope counts differ')
    oracle_values={}
    for row in main:
        block=[r for r in pred if r['experiment_id']==row['experiment_id']]
        if [int(r['window_index']) for r in block]!=list(range(1360)): raise RuntimeError('Prediction identity/order mismatch')
        truth=np.array([float(r['truth']) for r in block]); tc=np.array([int(r['true_class']) for r in block]); pc=np.array([int(r['predicted_class']) for r in block])
        mx=np.array([[float(r[f'route_{g}']) for g in range(4)] for r in block])
        expected,gases=scope_rows(truth,tc,pc,mx)
        for e in expected:
            o=next(r for r in scopes if r['experiment_id']==row['experiment_id'] and r['scope']==e['scope'])
            for k in ('n','RMSE','MAE','NRMSE_range','R2','Bias'):
                if not np.isclose(float(o[k]),e[k],rtol=1e-12,atol=1e-12,equal_nan=True): raise RuntimeError('Metric recomputation mismatch')
            if e['scope']=='Oracle_ALL':
                key=row['regression_bank_id']; value=(e['RMSE'],e['MAE'],e['NRMSE_range'])
                if key in oracle_values and oracle_values[key]!=value: raise RuntimeError('Shared oracle differs across classifiers/seeds')
                oracle_values[key]=value
    return {'status':'PASS','pipelines':36,'predictions':len(pred),'scope_rows':144,
            'unique_oracle_banks':len(oracle_values),'input_hashes':'PASS','metric_recomputation':'PASS',
            'neural_training':False,'new_source_fitting':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','verify'])
    parser.add_argument('--freeze-commit')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='verify': print(json.dumps(verify(),indent=2))
    else:
        if not args.freeze_commit: parser.error('--freeze-commit required')
        run(args.freeze_commit)
