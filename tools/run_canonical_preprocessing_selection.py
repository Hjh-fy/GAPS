"""Frozen 16-candidate canonical-preprocessing selection; no classifier training."""
from __future__ import annotations
import argparse,csv,hashlib,json,time,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parents[1];sys.path.insert(0,str(ROOT))
from tools import preprocessor_canonical_candidate as pc
from run_regression_head_ablation import CLASS_NAMES,deterministic_train_val,fit_ridge,rich_feature_dict
from scripts import run_gaps_cross_target_r84_full as common

RAW=WS/'dataset/data1';NEW=WS/'dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid';OUT_DEFAULT=ROOT/'results/iotj_canonical_preprocessing_selection_20260808'
CANDS=[(hz,agg,dur) for hz in (1,2,5,10) for agg in ('mean','median') for dur in (10,20)]
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def cid(c):return f'C{c}'
def ident(info,start,end):return (int(info['client_id']),info['filename'],round(float(start),6),round(float(end),6))
def candidate_id(c):return f'HZ{c[0]}_{c[1].upper()}_W{c[2]}S'
def target_roles():
 out={}
 for c in (3,4,5):
  for split,role in [('calibration','target_calibration'),('test','target_test_sealed')]:
   m=json.loads((NEW/f'client_{c}'/f'{split}_experiment_info.json').read_text(encoding='utf-8'))
   for x in m:out[(c,x['filename'],round(float(x['window_start_s']),6))]=role
 return out
def row_from(w,info,meta,role,h1):
 cls=int(info['classification_label']);phase={'early':0,'middle':1,'late':2}.get(info['phase_label'],-1);fd=rich_feature_dict(w,phase,{'window_start_s':meta['physical_window_start_s']});base={'client':cid(info['client_id']),'true_class':cls,'true_ppm':float(info['concentration']),'feature_dict':fd,'sample_index':0};base['H1_federated_source_ridge_ppm']=h1[cls].predict(fd);base=common.r84_row(base);base.update({'role':role,'gas':CLASS_NAMES[cls],**meta});return base
def metrics(rows,pred):
 y=np.array([r['true_ppm'] for r in rows]);e=pred-y;return {'N':len(rows),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(abs(e))),'Bias':float(np.mean(e)),'R2':float(1-(e@e)/np.sum((y-y.mean())**2)) if np.std(y)>0 else float('nan')}
def select_fit_eval(train,valid):
 names=sorted(train[0]['feature_dict']);best=(float('inf'),common.RIDGE_ALPHAS[0])
 for a in common.RIDGE_ALPHAS:
  m=fit_ridge(train,names,a);s=metrics(valid,m.predict(valid))['RMSE']
  if s<best[0]:best=(s,a)
 m=fit_ridge(train,names,best[1]);return m,best
def evaluate_source(rows,candidate):
 summary=[]
 for cls,gas in CLASS_NAMES.items():
  tr=[r for r in rows if r['true_class']==cls and r['role']=='source_train'];va=[r for r in rows if r['true_class']==cls and r['role']=='source_validation']
  if not tr or not va:continue
  model,(_,a)=select_fit_eval(tr,va);summary.append({'candidate_id':candidate,'class_id':cls,'gas':gas,'selected_alpha':a,**metrics(va,model.predict(va))})
 return summary
def evaluate_cal(rows,candidate):
 summary=[]
 for client in ('C3','C4','C5'):
  for cls,gas in CLASS_NAMES.items():
   r=[x for x in rows if x['client']==client and x['true_class']==cls and x['role']=='target_calibration']
   if len(r)<4:continue
   tr,va=deterministic_train_val(r,.25);model,(score,a)=select_fit_eval(tr,va);summary.append({'candidate_id':candidate,'client':client,'class_id':cls,'gas':gas,'selected_alpha':a,'validation_RMSE':score,**{k:v for k,v in metrics(va,model.predict(va)).items() if k!='RMSE'}})
 return summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT_DEFAULT);args=ap.parse_args();out=args.output
 if out.exists():raise RuntimeError(f'FAIL_CLOSED output exists: {out}')
 out.mkdir(parents=True);roles=target_roles();h1=common.load_h1();raw=sorted(p for p in RAW.rglob('*.txt') if 'B' in p.name);allrows={candidate_id(c):[] for c in CANDS};quality=[];manifest=[];cost=defaultdict(lambda:[0.,0])
 # Phase 1: source + calibration only.  No target-test window is materialized.
 for p in raw:
  rt,rx=pc.TA.load_raw_data(p);t,x,dup=pc.TA.clean_time_axis(rt,rx);info=pc.TA.parse_filename(p.name)
  for c in CANDS:
   name=candidate_id(c);tic=time.perf_counter();q=pc.process_arrays(t,x,info,*c,'mean',True,int(dup['duplicate_timestamps']));cost[name][0]+=time.perf_counter()-tic;cost[name][1]+=1
   for w,m in zip(q['windows'],q['metadata']):
    key=(int(info['client_id']),info['filename'],round(m['physical_window_start_s'],6));role=('source_validation' if int(info['repeat_id'])==4 else 'source_train') if int(info['client_id']) in (1,2) else roles.get(key)
    if role is None:continue
    if name==candidate_id(CANDS[0]):manifest.append({'client_id':info['client_id'],'raw_filename':info['filename'],'repeat_id':info['repeat_id'],'gas':info['gas'],'class_id':info['classification_label'],'concentration':info['concentration'],'physical_window_start_s':m['physical_window_start_s'],'physical_window_end_s':m['physical_window_end_s'],'role':role})
    quality.append({'candidate_id':name,**m,'client':cid(info['client_id']),'gas':info['gas'],'concentration':info['concentration'],'duplicate_timestamps':q['duplicate_timestamps'],'baseline_n_raw_samples':q['baseline_n_raw_samples'],'baseline_channel_std_mean':float(np.mean(q['baseline_channel_std']))})
    if role!='target_test_sealed' and m['valid']:allrows[name].append(row_from(w,info,m,role,h1))
 write(out/'master_physical_split_manifest.csv',manifest);src=[];cal=[];features=[];engineering=[];summary=[]
 for c in CANDS:
  name=candidate_id(c);rows=allrows[name];s=evaluate_source(rows,name);v=evaluate_cal(rows,name);src+=s;cal+=v
  q=[x for x in quality if x['candidate_id']==name];usable=sum(x['valid']=='True' or x['valid'] is True for x in q)/len(q);target=float(np.mean([x['validation_RMSE'] for x in v]));source=float(np.mean([x['RMSE'] for x in s]));
  # Feature quality is calibration-only and averaged across per-feature Spearman.
  frows=[x for x in rows if x['role']=='target_calibration'];bygas=defaultdict(list)
  for x in frows:bygas[(x['client'],x['gas'])].append(x)
  for (client,gas),z in bygas.items():
   names=sorted(z[0]['feature_dict']);corr=[abs(float(np.corrcoef([r['feature_dict'][n] for r in z],[r['true_ppm'] for r in z])[0,1])) for n in names if np.std([r['feature_dict'][n] for r in z])>0]
   features.append({'candidate_id':name,'client':client,'gas':gas,'N':len(z),'mean_abs_feature_concentration_corr':float(np.mean(corr)) if corr else float('nan'),'within_concentration_variance':float(np.mean([np.var([r['feature_dict'][n] for r in z]) for n in names]))})
  points=c[0]*c[2];engineering.append({'candidate_id':name,'sampling_rate':c[0],'aggregation':c[1],'window_duration_s':c[2],'points_per_window':points,'approx_input_tensor_bytes':points*8*4,'processing_time_s':cost[name][0],'per_file_latency_s':cost[name][0]/cost[name][1],'relative_compute_proxy':points})
  summary.append({'candidate_id':name,'sampling_rate':c[0],'aggregation':c[1],'window_duration_s':c[2],'usable_window_ratio':usable,'target_calibration_validation_RMSE':target,'source_validation_RMSE':source,'ranking_eligible':usable>=.90,'selection_status':'screened'})
 write(out/'candidate_summary.csv',summary);write(out/'source_validation_summary.csv',src);write(out/'target_calibration_validation_summary.csv',cal);write(out/'feature_quality_summary.csv',features);write(out/'acquisition_quality_summary.csv',quality);write(out/'engineering_cost_summary.csv',engineering)
 eligible=[r for r in summary if r['ranking_eligible']];top=sorted(eligible,key=lambda r:(r['target_calibration_validation_RMSE'],r['source_validation_RMSE'],r['relative_compute_proxy'] if 'relative_compute_proxy' in r else r['sampling_rate']*r['window_duration_s']))[:2]
 # Freeze exactly these IDs before the first target-test materialization.
 frozen={r['candidate_id']:r for r in top};
 for i,r in enumerate(top,1):(out/f'candidate_{i}_preprocessing_manifest.json').write_text(json.dumps({**r,'baseline':'raw_observation_mean','gap_policy':'short_gap_only','frozen_before_target_test':True},indent=2)+'\n',encoding='utf-8')
 # Phase 2: one-time test diagnostic for frozen TOP-2 only.
 testrows={n:[] for n in frozen}
 for p in raw:
  info=pc.TA.parse_filename(p.name)
  if int(info['client_id']) not in (3,4,5):continue
  rt,rx=pc.TA.load_raw_data(p);t,x,dup=pc.TA.clean_time_axis(rt,rx)
  for c in CANDS:
   name=candidate_id(c)
   if name not in frozen:continue
   q=pc.process_arrays(t,x,info,*c,'mean',True,int(dup['duplicate_timestamps']))
   for w,m in zip(q['windows'],q['metadata']):
    if roles.get((int(info['client_id']),info['filename'],round(m['physical_window_start_s'],6)))=='target_test_sealed' and m['valid']:testrows[name].append(row_from(w,info,m,'target_test_sealed',h1))
 testout=[]
 for name in frozen:
  calrows=[r for r in allrows[name] if r['role']=='target_calibration']
  for client in ('C3','C4','C5'):
   for cls,gas in CLASS_NAMES.items():
    cr=[r for r in calrows if r['client']==client and r['true_class']==cls];tr,va=deterministic_train_val(cr,.25);model,_=select_fit_eval(tr,va);model=fit_ridge(cr,sorted(cr[0]['feature_dict']),_ [1]);te=[r for r in testrows[name] if r['client']==client and r['true_class']==cls]
    if te:testout.append({'candidate_id':name,'client':client,'class_id':cls,'gas':gas,**metrics(te,model.predict(te))})
 write(out/'top2_target_test_summary.csv',testout)
 protocol='''# Canonical preprocessing design protocol\n\nFrozen candidates use true timestamps, stable-sort and duplicate merging; raw-observation conductance G0 over [20,50) s; physical-time bins; short-gap-only filling; physical crop [60,170] s; and 10/20-s physical windows. Screening uses source validation plus target calibration internal validation only. Target test is sealed until TOP-2 manifests are written, then used once for diagnostic validation only. No classifier/R84 architecture/loss/QC change or 25-round training is permitted.\n'''
 (out/'PREPROCESSING_DESIGN_PROTOCOL.md').write_text(protocol,encoding='utf-8')
 (out/'CANDIDATE_SCREENING_SUMMARY.md').write_text('# Candidate screening\n\nTOP-2 were frozen using eligibility, calibration validation RMSE, source RMSE, then compute proxy. See CSV evidence; no target test entered selection.\n',encoding='utf-8')
 (out/'FINAL_CANDIDATE_DECISION.md').write_text('# Final candidate decision\n\nTOP-2 target-test diagnostic is reported, but no automatic canonical choice is made from test RMSE. Human/audit review is required before any formal training authorization.\n',encoding='utf-8')
 print(json.dumps({'top2':[r['candidate_id'] for r in top],'screened':len(summary),'target_test_rows':sum(map(len,testrows.values()))},indent=2))
if __name__=='__main__':main()
