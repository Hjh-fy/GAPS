"""Fit only C5 10/15% Ridge banks, reuse 5/20%, retain true-class scope."""
from __future__ import annotations
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from gaps_flower.fedongas_extension import *
from gaps_flower.fedongas_regression_cross import fit_bank,predict_bank,quantitative_features,scope_rows
from gaps_flower.canonical_r1_v1 import validate_r0_bundle,predict_ridge_model
from scripts.run_fedongas_regression_cross import write_json,write_csv,write_text,read_json,read_csv,sha,assert_hashes,PRIOR,PRIOR_SHA


def release_saved_test(out):
    out=Path(out)
    if not all((out/n).exists() for n in ('target_model_lock.json','target_alpha_lock.json','input_lock.json')):
        raise RuntimeError('Regression test remains locked')
    lock=read_json(out/'target_model_lock.json');expected={r['experiment_id'] for r in regression_matrix()}
    if set(lock)!=expected or any(set(v)!={'0','1','2','3'} for v in lock.values()):raise RuntimeError('Regression test remains locked: banks')
    assert_hashes(read_json(out/'input_lock.json'))
    write_json(out/'TEST_RELEASE.json',{'all_four_banks_locked':True,'test_used_for_selection':False,
        'lock_sha256':{n:sha(out/n) for n in ('target_model_lock.json','target_alpha_lock.json','input_lock.json')},
        'cache_reuse':'existing fixed canonical C5 test features; no original test tensor reload'})
    z=np.load(OLDREG/'test_features.npz',allow_pickle=False)
    rows=read_csv(OLDREG/'predictions.csv');name=rows[0]['experiment_id'];rows=[r for r in rows if r['experiment_id']==name]
    if [int(r['window_index']) for r in rows]!=list(range(1360)):raise RuntimeError('Test cache order differs')
    return z,np.array([int(r['true_class']) for r in rows]),np.array([float(r['truth']) for r in rows])


def run(commit):
    if REG_STUDY.exists():raise FileExistsError('Preserve regression endpoint; do not refit')
    from subprocess import check_output
    critical=['gaps_flower/fedongas_extension.py',__file__,'gaps_flower/fedongas_regression_cross.py',str(DOC/'protocol_manifest.json')]
    if check_output(['git','diff',commit,'--',*critical],cwd=ROOT):raise RuntimeError('Regression frozen code differs')
    info=read_json(DOC/'protocol_manifest.json')['inputs'];assert_hashes(info['input_hashes'])
    source=validate_r0_bundle(PRIOR,expected_index_sha256=PRIOR_SHA)
    REG_STUDY.mkdir(parents=True);start=time.perf_counter()
    write_json(REG_STUDY/'PRE_RUN_FREEZE.json',{'freeze_commit':commit,'protocol_sha256':sha(DOC/'protocol_manifest.json')})
    write_json(REG_STUDY/'input_lock.json',info['input_hashes'])
    banks,alphas,activity={},{},[]
    for b in (10,15):
        payload=load_budget(b);sensor,h1=quantitative_features(payload['x'],payload['phases'],payload['metadata'])
        prior=np.column_stack([predict_ridge_model(source[g],h1) for g in range(4)])
        np.savez_compressed(REG_STUDY/f'calibration_B{b}_features.npz',sensor83=sensor,h1=h1,source_prior=prior)
        for r in [r for r in regression_matrix() if r['budget_pct']==b]:
            key=r['experiment_id'];banks[key],alphas[key]=fit_bank(sensor,prior,payload['truth'],payload['classes'],
                np.array([m['filename'] for m in payload['metadata']]),r['regressor'])
            for g in range(4):
                own=prior[payload['classes']==g,g]
                activity.append({'budget_pct':b,'regressor':r['regressor'],'gas_id':g,'source_prior_sd':float(own.std()),
                    'source_prior_unique':len(np.unique(own)),'selected_alpha':banks[key][g]['alpha'],
                    'prior_coefficient':banks[key][g]['coef'][-1] if r['regressor']=='R84_CONCAT' else ''})
            print('FITTED',key,flush=True)
    write_json(REG_STUDY/'target_model_lock.json',banks);write_json(REG_STUDY/'target_alpha_lock.json',alphas)
    z,tc,y=release_saved_test(REG_STUDY)
    overall,gas,predictions=[],[],[]
    for r in regression_matrix():
        mx=predict_bank(banks[r['experiment_id']],z['sensor83'],z['source_prior'],r['regressor'])
        scopes,gs=scope_rows(y,tc,tc,mx)
        meta={**r,'calculation_status':'recomputed_new_fixed_bank'}
        overall.append({**meta,**next(v for v in scopes if v['scope']=='Oracle_ALL')})
        gas += [{**meta,**v} for v in gs if v['scope']=='Oracle_ALL']
        for i in range(1360):predictions.append({'experiment_id':r['experiment_id'],'window_index':i,'true_class':int(tc[i]),
            'truth':float(y[i]),'prediction':float(mx[i,tc[i]]),**{f'route_{g}':float(mx[i,g]) for g in range(4)}})
    for r in read_csv(OLDREG/'oracle_unique_banks.csv'):
        overall.append({'experiment_id':r['regression_bank_id'],'budget_pct':int(r['budget_pct']),'regressor':r['regressor'],
            'calibration_n':int(r['budget_pct'])*16,'test_n':1360,'scope':'Oracle_ALL','n':int(r['n']),
            **{k:float(r[k]) for k in ('RMSE','MAE','NRMSE_range')},'calculation_status':'reported_reused_5_20_no_refit'})
    for r in read_csv(OLDREG/'per_gas_routing_scopes.csv'):
        if r['scope']=='Oracle_ALL' and r['classifier']=='TO' and r['seed']=='42':
            gas.append({'experiment_id':r['regression_bank_id'],'budget_pct':int(r['budget_pct']),'regressor':r['regressor'],
                'calibration_n':int(r['budget_pct'])*16,'test_n':1360,'scope':'Oracle_ALL','n':int(r['n']),
                'gas_id':int(r['gas_id']),'gas':r['gas'],**{k:float(r[k]) for k in ('RMSE','MAE','NRMSE_range')},
                'calculation_status':'reported_reused_5_20_no_refit'})
    overall.sort(key=lambda x:(x['budget_pct'],x['regressor']));gas.sort(key=lambda x:(x['budget_pct'],x['gas_id'],x['regressor']))
    write_csv(REG_STUDY/'c5_regression_budget_overall_8.csv',overall);write_csv(REG_STUDY/'c5_regression_budget_per_gas_32.csv',gas)
    write_csv(REG_STUDY/'new_bank_predictions.csv',predictions);write_csv(REG_STUDY/'source_prior_activity.csv',activity)
    cross=[];oldr1=ROOT/'results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812'
    for r in read_csv(oldr1/'canonical_regression_comparison.csv'):
        if r['target'] in ('C3','C4','C5') and r['scope']=='Oracle_ALL' and r['method'] in ('TARGET_ONLY_83D_RIDGE','R84_CONCAT'):
            cross.append({**r,'calculation_status':'reported_reused_canonical_R1_no_refit'})
    write_csv(REG_STUDY/'cross_device_main_calibration_regression.csv',cross)
    lines=['# C5四档预算真实类别回归','',
        '10%/15%新增4bank/16最终gas fits；5%/20%只复用；无分类/source回归重训。Oracle_ALL全部1360，逐gas340。',
        '回归器与classifier seed无关，不对复制的Oracle计算seed SD或显著性。','',
        '|预算|回归|RMSE ppm|NRMSE_range|来源|','|---|---|---:|---:|---|']
    for r in overall:lines.append(f"|{r['budget_pct']}%|{r['regressor']}|{r['RMSE']:.6f}|{r['NRMSE_range']:.6f}|{r['calculation_status']}|")
    lines+=['','## 源增强差值（R84−83D，负值更好）','']
    for b in (5,10,15,20):
        a=next(r for r in overall if r['budget_pct']==b and r['regressor']=='TARGET_RIDGE_83D')
        c=next(r for r in overall if r['budget_pct']==b and r['regressor']=='R84_CONCAT')
        lines.append(f"- {b}%：ΔRMSE={c['RMSE']-a['RMSE']:+.6f} ppm；ΔNRMSE={c['NRMSE_range']-a['NRMSE_range']:+.6f}。")
    lines+=['','逐gas负收益和非单调趋势均保留；固定源参考含H1协议/phase信息及原有clipping，已知饱和未在本轮修复。',
            'C3/C4主标定回归直接复用既有R1，不将旧A4分类路由当作CLASS。结果不支持对独立设备/不同划分的普适声明。']
    write_text(REG_STUDY/'RESULT_ANALYSIS.md','\n'.join(lines)+'\n')
    assert_hashes(info['input_hashes'])
    write_json(REG_STUDY/'RUN_PROGRESS.json',{'status':'COMPLETE','new_banks':4,'final_gas_fits':16,'overall_budget_rows':8,'wall_seconds':time.perf_counter()-start})
    write_json(REG_STUDY/'sha256_index.json',{p.name:sha(p) for p in REG_STUDY.iterdir() if p.is_file()})
    write_json(REG_STUDY/'COMPLETE.json',{'status':'PASS','freeze_commit':commit,'new_banks':4,'index_sha256':sha(REG_STUDY/'sha256_index.json')})
    print('REGRESSION_COMPLETE: four new banks, four budget points',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-commit',required=True);run(p.parse_args().freeze_commit)
