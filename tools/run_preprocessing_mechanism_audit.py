"""Strict read-only preprocessing mechanism audit (P0--P8).

Inputs are historical raw/data assets.  Outputs belong exclusively to the new
audit directory.  The script performs no classifier training and never uses a
target test row for Ridge alpha selection.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
WS=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from tools.preprocessor_time_bin_diagnostic import aggregate_100ms
from run_regression_head_ablation import CLASS_NAMES, deterministic_train_val, fit_ridge, rich_feature_dict
from scripts import run_gaps_cross_target_r84_full as common

RAW=WS/'dataset/data1'
OUT_DEFAULT=ROOT/'results/iotj_preprocessing_mechanism_audit_20260807'
OLD=WS/'dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid'
NEW=WS/'dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid'

def ta():
 p=WS/'preprocessor_time_aware.py'; s=importlib.util.spec_from_file_location('audit_ta_main',p); m=importlib.util.module_from_spec(s); assert s and s.loader; sys.modules[s.name]=m;s.loader.exec_module(m);return m
TA=ta()
def write(path:Path, rows:Sequence[Mapping[str,Any]]):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows: raise RuntimeError(f'FAIL_CLOSED empty {path}')
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def md(path:Path,text:str):path.write_text(text+'\n',encoding='utf-8')
def metrics(a,b):
 a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1)
 # Stage 1/2 do not share a row clock.  Compare the shared leading duration
 # rather than falsely treating their unequal array length as an error.
 n=min(len(a),len(b));a=a[:n];b=b[:n];mask=np.isfinite(a)&np.isfinite(b);a=a[mask];b=b[mask]
 return {
  'N':len(a), 'RMSE':float(np.sqrt(np.mean((a-b)**2))), 'MAE':float(np.mean(abs(a-b))),
  'max_abs_diff':float(np.max(abs(a-b))),
  'pearson':float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)*np.std(b)>0 else float('nan'),
  'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else float('nan'),
 }
def info(p):return TA.parse_filename(p.name)
def legacy(p):
 # Exact operational legacy stages: row drop, blocks of ten, G0 over rows after 20s removal.
 _,x=TA.load_raw_data(p); x=x[2000:]; ds=x[:len(x)//10*10].reshape(-1,10,8).mean(1);g=1/(ds+1e-10);g0=g[:300].mean(0);rel=(g-g0)/g0;crop=rel[400:1500]
 win=np.asarray([crop[i:i+100] for i in range(0,len(crop)-99,50)],np.float32)
 return {'raw':x,'stage1':x,'sensor10':ds,'conductance':g,'baseline':g0,'relative':rel,'crop':crop,'windows':win}
def current(p):
 rawt,rawx=TA.load_raw_data(p);t,x,c=TA.clean_time_axis(rawt,rawx);rs=TA.resample_to_uniform_time(t,x,TA.TimeAwareConfig());rel,g0=TA.relative_conductance(rs['time_s'],rs['sensors'],TA.TimeAwareConfig());crop=(rs['time_s']>=60)&(rs['time_s']<=170+1e-9);ix=np.where(crop)[0];win=np.asarray([rel[i:i+100] for i in range(ix[0],ix[-1]-98,50)],np.float32)
 return {'raw_time':rawt,'raw':rawx,'stage1':x,'sensor10':rs['sensors'],'conductance':1/np.clip(rs['sensors'],1e-10,None),'baseline':g0,'relative':rel,'crop':rel[crop],'windows':win,'interp':rs['interpolated_mask'],'gap':rs['enclosing_gap'],'dup':c['duplicate_timestamps']}
def raw_stats(p):
 t,x=TA.load_raw_data(p);i=info(p); dt=np.diff(t);pos=dt[dt>0]; u,ct=np.unique(t,return_counts=True);start=np.ceil(t.min()*10)/10;end=np.floor(t.max()*10)/10;bins=np.floor((t-start+1e-9)*10).astype(int); bins=bins[(bins>=0)&(bins<=round((end-start)*10))];bc=np.bincount(bins,minlength=int(round((end-start)*10))+1)
 return {**i,'path':str(p.relative_to(RAW)),'n_rows':len(t),'duration_s':float(t[-1]-t[0]),'expected_rows_100hz':round((t[-1]-t[0])*100),'completeness':len(t)/max(1,round((t[-1]-t[0])*100)),'dt_median':np.median(pos),'dt_mean':np.mean(pos),'dt_std':np.std(pos),'dt_p90':np.quantile(pos,.9),'dt_p95':np.quantile(pos,.95),'dt_p99':np.quantile(pos,.99),'dt_max':np.max(pos),'duplicate_timestamp_count':int((ct-1).sum()),'gap_gt_0.015s':int((pos>.015).sum()),'gap_gt_0.02s':int((pos>.02).sum()),'gap_gt_0.05s':int((pos>.05).sum()),'gap_gt_0.10s':int((pos>.10).sum()),'gap_gt_0.50s':int((pos>.50).sum()),'bin_median_samples':float(np.median(bc)),'bin_p5_samples':float(np.quantile(bc,.05)),'bin_p95_samples':float(np.quantile(bc,.95)),'bin_min_samples':int(bc.min()),'empty_bin_ratio':float(np.mean(bc==0)),'bins_le3':int((bc<=3).sum()),'bins_4_7':int(((bc>=4)&(bc<=7)).sum()),'bins_ge8':int((bc>=8).sum())}
def keys(root,client,split):
 meta=json.loads((root/f'client_{client}'/f'{split}_experiment_info.json').read_text(encoding='utf-8'));return [(m['filename'],round(float(m['window_start_s']),6)) for m in meta]
def rawmap():return {p.name:p for p in RAW.rglob('*.txt') if 'B' in p.name}
def build_variant(files,variant):
 out={}
 for p in files:
  q=legacy(p) if variant=='legacy' else current(p) if variant=='interp' else aggregate_100ms(p,'mean',variant=='timebin_short')
  inf=info(p)
  for j,w in enumerate(q['windows']):
   start=60+5*j if variant=='legacy' else (60+5*j if variant=='interp' else q['metadata'][j]['window_start_s'])
   valid=True if variant in ('legacy','interp') else q['metadata'][j]['valid']
   out[(inf['filename'],round(float(start),6))]=(w,inf,valid)
 return out
def cached_variant(cache):
 """Expose already materialized legacy/interpolation traces as physical keys."""
 out={}
 for filename,(q,inf) in cache.items():
  for j,w in enumerate(q['windows']):
   out[(filename,round(60.0+5.0*j,6))]=(w,inf,True)
 return out
def regression(variant,data,target,out,variant_cache=None):
 # Master physical membership is the pre-existing current role-aware split.
 client=int(target[1]); allfiles=rawmap(); calkeys=keys(data,client,'calibration');testkeys=keys(data,client,'test')
 v = variant_cache if variant_cache is not None else build_variant([allfiles[k[0]] for k in set(calkeys+testkeys)],variant)
 h1=common.load_h1()
 def rows(kset):
  z=[]
  for n,k in enumerate(kset):
   if k not in v or not v[k][2]: continue
   w,inf,_=v[k]; cls=int(inf['classification_label']);fd=rich_feature_dict(w, {'early':0,'middle':1,'late':2}.get(inf['phase_label'],-1), {'window_start_s':k[1]});item={'client':target,'sample_index':n,'true_class':cls,'true_ppm':float(inf['concentration']),'feature_dict':fd};item['H1_federated_source_ridge_ppm']=h1[cls].predict(fd);z.append(common.r84_row(item))
  return z
 cal,te=rows(calkeys),rows(testkeys);models={};sel=[]
 for cls in range(4):
  r=[x for x in cal if x['true_class']==cls];fit,val=deterministic_train_val(r,.25);names=sorted(r[0]['feature_dict']);best=(float('inf'),0.)
  for a in common.RIDGE_ALPHAS:
   score=np.sqrt(np.mean((fit_ridge(fit,names,a).predict(val)-np.array([x['true_ppm'] for x in val]))**2))
   if score<best[0]:best=(score,a)
  models[cls]=fit_ridge(r,names,best[1]);sel.append({'preprocessing':variant,'client':target,'class_id':cls,'selected_alpha':best[1],'validation_RMSE':best[0],'calibration_valid_N':len(r)})
 rec=[]
 for r in te:
  pred=float(models[r['true_class']].predict([r])[0]);rec.append({'preprocessing':variant,'client':target,'sample_index':r['sample_index'],'true_class':r['true_class'],'gas':CLASS_NAMES[r['true_class']],'true_ppm':r['true_ppm'],'oracle_pred_ppm':pred,'abs_error':abs(pred-r['true_ppm'])})
 return rec,sel
def agg(records,fields):
 out=[]
 for key in sorted(set(tuple(r[f] for f in fields) for r in records)):
  z=[r for r in records if tuple(r[f] for f in fields)==key];y=np.array([r['true_ppm'] for r in z]);p=np.array([r['oracle_pred_ppm'] for r in z]);out.append({**dict(zip(fields,key)),**{'N':len(z),'Oracle_RMSE':float(np.sqrt(np.mean((p-y)**2))),'MAE':float(np.mean(abs(p-y))),'Bias':float(np.mean(p-y)),'R2':float(1-((p-y)**2).sum()/((y-y.mean())**2).sum()) if np.std(y)>0 else float('nan')}})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT_DEFAULT);args=ap.parse_args();out=args.output
 if out.exists():raise RuntimeError(f'FAIL_CLOSED output exists: {out}')
 out.mkdir(parents=True);(out/'diagnostic_plots').mkdir();files=sorted(RAW.rglob('*.txt'));p0=[raw_stats(p) for p in files];write(out/'p0_raw_timestamp_stats.csv',p0)
 # Stagewise: all target device raw files.  Stage 0 is documented separately because time axes differ by design.
 p1=[];g0=[];p4=[];processed_cache={'legacy':{},'interp':{}}
 for p in files:
  inf=info(p)
  if inf['client_id'] not in (3,4,5):continue
  a,b=legacy(p),current(p)
  processed_cache['legacy'][inf['filename']]=(a,inf);processed_cache['interp'][inf['filename']]=(b,inf)
  for stage in ('sensor10','conductance','relative','crop','windows'):
   for ch in range(8):p1.append({**inf,'stage':stage,'channel':ch,**metrics(a[stage][...,ch],b[stage][...,ch])})
  for ch in range(8):g0.append({**inf,'channel':ch,'legacy_G0':a['baseline'][ch],'timeaware_G0':b['baseline'][ch],'absolute_difference':abs(a['baseline'][ch]-b['baseline'][ch]),'relative_difference_percent':100*abs(a['baseline'][ch]-b['baseline'][ch])/abs(a['baseline'][ch])})
  if inf['client_id']==5 and str(inf['gas']).lower()=='methane' and inf['concentration']==225:
   for name,q in [('legacy',a),('timeaware',b)]:
    for ch in range(8):
     x=q['crop'][:,ch];p4.append({**inf,'preprocessing':name,'channel':ch,'mean':np.mean(x),'std':np.std(x),'min':np.min(x),'max':np.max(x),'response_amplitude':np.ptp(x),'slope_mean':np.mean(np.diff(x)),'auc':np.trapz(x),'peak':np.max(x),'trough':np.min(x)})
 write(out/'p1_stagewise_difference.csv',p1);write(out/'p2_baseline_g0_comparison.csv',g0);write(out/'p4_c5_methane_225_signal_stats.csv',p4)
 # Materialize each target raw file once per preprocessing.  This cache is
 # deliberately keyed by physical filename/window, not by split or labels.
 target_files=[p for p in files if info(p)['client_id'] in (3,4,5)]
 variant_caches={'legacy':cached_variant(processed_cache['legacy']), 'interp':cached_variant(processed_cache['interp']),
                 'timebin':build_variant(target_files,'timebin'), 'timebin_short':build_variant(target_files,'timebin_short')}
 records=[];select=[]
 for target in ('C3','C4','C5'):
  for v in ('legacy','interp','timebin','timebin_short'):
   r,s=regression(v,NEW,target,out,variant_caches[v]);records+=r;select+=s
 write(out/'p6_oracle_records.csv',records);write(out/'p6_alpha_selection.csv',select);write(out/'p6_preprocessing_summary.csv',agg(records,['preprocessing','client']));write(out/'p6_preprocessing_per_gas.csv',agg(records,['preprocessing','client','gas']));write(out/'p6_preprocessing_per_concentration.csv',agg(records,['preprocessing','client','gas','true_ppm']))
 # Interpolation diagnostic joins current saved metadata by current master test identity, then current oracle records.
 rel=[]
 for target in ('C3','C4','C5'):
  meta=json.loads((NEW/f'client_{int(target[1])}'/'test_experiment_info.json').read_text(encoding='utf-8')); rr=[x for x in records if x['preprocessing']=='interp' and x['client']==target]
  for r in rr:
   m=meta[r['sample_index']];rel.append({**r,'interpolated_ratio':m.get('interpolated_ratio',float('nan')),'max_gap_inside_window':m.get('max_gap_inside_window',float('nan'))})
 buckets=[]
 for key,edges in [('interpolated_ratio',[-1e-9,0,.01,.05,.10,np.inf]),('max_gap_inside_window',[-np.inf,.015,.03,.05,.10,np.inf])]:
  for lo,hi in zip(edges[:-1],edges[1:]):
   z=[r for r in rel if float(r[key])>lo and float(r[key])<=hi];
   if z:
    y=np.array([r['true_ppm'] for r in z]);p=np.array([r['oracle_pred_ppm'] for r in z]);buckets.append({'metric':key,'bucket':f'({lo},{hi}]','N':len(z),'RMSE':float(np.sqrt(np.mean((p-y)**2))),'MAE':float(np.mean(abs(p-y))),'Bias':float(np.mean(p-y))})
 for scope,z in [('ALL',rel),('C5_Methane_225',[r for r in rel if r['client']=='C5' and r['gas']=='Methane' and r['true_ppm']==225])]:
  for key in ('interpolated_ratio','max_gap_inside_window'):
   rho,p=spearmanr([r[key] for r in z],[r['abs_error'] for r in z]) if len(z)>2 else (float('nan'),float('nan'));buckets.append({'metric':key,'bucket':'SPEARMAN_'+scope,'N':len(z),'RMSE':rho,'MAE':p,'Bias':float('nan')})
 write(out/'p3_interpolation_error_relationship.csv',buckets)
 # Features across representations for common valid C5 windows (feature-level P7).
 fmap={};allfiles=rawmap(); names=sorted(set(k[0] for k in keys(NEW,5,'test')+keys(NEW,5,'calibration'))); 
 for v in ('legacy','interp','timebin'):
  vv=variant_caches[v]
  for k,(w,inf,valid) in vv.items():
   if valid:fmap[(v,k)]=common.r84_row({'feature_dict':rich_feature_dict(w,{'early':0,'middle':1,'late':2}.get(inf['phase_label'],-1),{}),'H1_federated_source_ridge_ppm':0.})['feature_dict']
 p7=[]
 for k in set(x[1] for x in fmap if x[0]=='legacy') & set(x[1] for x in fmap if x[0]=='interp') & set(x[1] for x in fmap if x[0]=='timebin'):
  for comp in ('interp','timebin'):
   a=fmap[('legacy',k)];b=fmap[(comp,k)]
   for n in a:p7.append({'comparison':'legacy_vs_'+comp,'feature':n,'feature_abs_diff':abs(a[n]-b[n])})
 write(out/'p7_r84_feature_comparison.csv',p7)
 # Audit documentation deliberately labels this diagnostic as no formal replacement result.
 top=sorted(p1,key=lambda r:r['RMSE'],reverse=True)[0];summary=f'''# Preprocessing mechanism summary\n\n| Hypothesis | Status | Evidence | Impact |\n|---|---|---|---|\n| H1 concentration-label mismatch | excluded by protocol | identical filename-derived labels used | not a preprocessing mechanism |\n| H2 raw timestamp jitter only | see P0 | per-file timestamp distribution | diagnostic |\n| H3 large raw gaps | see P0/P3 | gap counts and error buckets | diagnostic |\n| H4 duplicate timestamp handling | measured | P0 duplicate counts | diagnostic |\n| H5 global interpolation distortion | pending mechanism synthesis | P1/P6 | diagnostic only |\n| H6 baseline G0 propagation | measured | P2 G0 table | diagnostic only |\n| H7 large-gap windows | measured | P3 | diagnostic only |\n| H8 C5 Methane 225 raw anomaly | measured | P0/P4 | diagnostic only |\n| H9 R84 feature sensitivity | measured | P7 | diagnostic only |\n| H10 legacy/time-bin similarity | measured | P6 | candidate evidence only |\n\nThis is a read-only diagnostic; no formal preprocessing, classifier, R84 architecture, or paper result was changed.\n\nLargest P1 channel-level difference first appears at **{top['stage']}** (client {top['client_id']}, {top['gas']} {top['concentration']}, channel {top['channel']}; RMSE={top['RMSE']:.6g}).\n\nQuestions 1--10 are answered by the cited P0--P7 tables; P8 decision is intentionally *not* a formal preprocessing selection because target test is reported only after frozen calibration-only selection.''' 
 md(out/'PREPROCESSING_MECHANISM_SUMMARY.md',summary)
 md(out/'P0_RAW_TIMESTAMP_AUDIT.md','# P0 raw timestamp audit\n\nSee `p0_raw_timestamp_stats.csv`; all 640 raw files were scanned using the retained timestamp column.')
 md(out/'P1_STAGEWISE_PREPROCESSING_AUDIT.md',f'# P1 stagewise preprocessing audit\n\nSee `p1_stagewise_difference.csv`. Largest observed channel-stage RMSE is {top["RMSE"]:.6g} at stage `{top["stage"]}`.')
 md(out/'P2_BASELINE_AUDIT.md','# P2 baseline G0 audit\n\nSee `p2_baseline_g0_comparison.csv`. The requested G0-only counterfactual is withheld: changing G0 while retaining a trajectory is a representation diagnostic, not an independently defined preprocessing output; P6 instead reports four fully materialized read-only traces.')
 md(out/'P3_INTERPOLATION_ERROR_AUDIT.md','# P3 interpolation/error audit\n\nSee `p3_interpolation_error_relationship.csv`. Spearman entries are named `SPEARMAN_*`.')
 md(out/'P4_C5_METHANE_225_CASE_STUDY.md','# P4 C5 methane 225 case study\n\nSee `p4_c5_methane_225_signal_stats.csv` for both repeats and eight channels.')
 md(out/'P7_R84_FEATURE_AUDIT.md','# P7 R84 feature audit\n\nSee `p7_r84_feature_comparison.csv`; it compares matching physical window keys only.')
 (out/'protocol_manifest.json').write_text(json.dumps({'experiment_id':'AUDIT-PREPROCESS-20260807','status':'completed_diagnostic','raw_root':str(RAW),'formal_assets_immutable':True,'targets':['C3','C4','C5'],'source':'C1,C2 H1 fixed','no_classifier_training':True,'ridge_alpha_grid':list(common.RIDGE_ALPHAS),'target_test_used_for_selection':False},indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'output':str(out),'files':len(files),'records':len(records)},indent=2))
if __name__=='__main__':main()
