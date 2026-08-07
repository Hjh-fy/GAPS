"""Complete P3/P6/P7 from immutable full-grid artifacts after P0--P4.

The legacy/current maps are loaded from their existing full-grid outputs.  Only
the two newly introduced time-bin diagnostic arms touch raw text again.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; WS=ROOT.parents[1]; sys.path.insert(0,str(ROOT))
import tools.run_preprocessing_mechanism_audit as audit
from run_regression_head_ablation import rich_feature_dict
from scripts import run_gaps_cross_target_r84_full as common

OUT=ROOT/'results/iotj_preprocessing_mechanism_audit_20260807'; NEW=WS/'dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid'
LEGACY=WS/'dataset/processed'; CURRENT=WS/'results/time_aware_pipeline_probe_window_fullgrid/time_aware_60_170_window_fullgrid/processed'

def processed_map(root:Path, client:int, legacy:bool):
 x=np.load(root/f'unit_{client}'/'features.npy'); meta=json.loads((root/f'unit_{client}'/'experiment_info.json').read_text(encoding='utf-8')); out={}; seen={}
 for w,m in zip(x,meta):
  fn=m['filename']; n=seen.get(fn,0); seen[fn]=n+1; start=60.0+5.0*n if legacy else round(float(m['window_start_s']),6)
  out[(fn,round(start,6))]=(w,m,True)
 return out
def write(path, rows):audit.write(path,rows)
def main():
 if not (OUT/'p0_raw_timestamp_stats.csv').exists():raise RuntimeError('P0--P4 prerequisite missing')
 raw=audit.rawmap(); target_files=[p for p in raw.values() if audit.info(p)['client_id'] in (3,4,5)]
 caches={'legacy':{},'interp':{}}
 for c in (3,4,5):
  caches['legacy'].update(processed_map(LEGACY,c,True));caches['interp'].update(processed_map(CURRENT,c,False))
 caches['timebin']=audit.build_variant(target_files,'timebin');caches['timebin_short']=audit.build_variant(target_files,'timebin_short')
 records=[];selection=[]
 for target in ('C3','C4','C5'):
  for variant in ('legacy','interp','timebin','timebin_short'):
   r,s=audit.regression(variant,NEW,target,OUT,caches[variant]);records+=r;selection+=s
 write(OUT/'p6_oracle_records.csv',records);write(OUT/'p6_alpha_selection.csv',selection)
 write(OUT/'p6_preprocessing_summary.csv',audit.agg(records,['preprocessing','client']))
 write(OUT/'p6_preprocessing_per_gas.csv',audit.agg(records,['preprocessing','client','gas']))
 write(OUT/'p6_preprocessing_per_concentration.csv',audit.agg(records,['preprocessing','client','gas','true_ppm']))
 rel=[]
 for target in ('C3','C4','C5'):
  meta=json.loads((NEW/f'client_{int(target[1])}'/'test_experiment_info.json').read_text(encoding='utf-8'))
  for r in [z for z in records if z['preprocessing']=='interp' and z['client']==target]:
   m=meta[r['sample_index']];rel.append({**r,'interpolated_ratio':m.get('interpolated_ratio',float('nan')),'max_gap_inside_window':m.get('max_gap_inside_window',float('nan'))})
 buckets=[]
 for key,edges in [('interpolated_ratio',[-1e-9,0,.01,.05,.10,np.inf]),('max_gap_inside_window',[-np.inf,.015,.03,.05,.10,np.inf])]:
  for lo,hi in zip(edges[:-1],edges[1:]):
   z=[r for r in rel if float(r[key])>lo and float(r[key])<=hi]
   if z:
    y=np.array([r['true_ppm'] for r in z]);p=np.array([r['oracle_pred_ppm'] for r in z]);buckets.append({'metric':key,'bucket':f'({lo},{hi}]','N':len(z),'RMSE':float(np.sqrt(np.mean((p-y)**2))),'MAE':float(np.mean(abs(p-y))),'Bias':float(np.mean(p-y))})
 for scope,z in [('ALL',rel),('C5_Methane_225',[r for r in rel if r['client']=='C5' and r['gas']=='Methane' and r['true_ppm']==225])]:
  for key in ('interpolated_ratio','max_gap_inside_window'):
   rho,p=spearmanr([r[key] for r in z],[r['abs_error'] for r in z]) if len(z)>2 else (float('nan'),float('nan'))
   buckets.append({'metric':key,'bucket':'SPEARMAN_'+scope,'N':len(z),'RMSE':rho,'MAE':p,'Bias':float('nan')})
 write(OUT/'p3_interpolation_error_relationship.csv',buckets)
 fmap={}
 for variant in ('legacy','interp','timebin'):
  for key,(w,m,valid) in caches[variant].items():
   if valid:
    phase={'early':0,'middle':1,'late':2}.get(m.get('phase_label'),-1);fd=rich_feature_dict(w,phase,{})
    fmap[(variant,key)]=common.r84_row({'feature_dict':fd,'H1_federated_source_ridge_ppm':0.})['feature_dict']
 p7=[]
 common_keys=set(k for v,k in fmap if v=='legacy')&set(k for v,k in fmap if v=='interp')&set(k for v,k in fmap if v=='timebin')
 for key in common_keys:
  for comp in ('interp','timebin'):
   a,b=fmap[('legacy',key)],fmap[(comp,key)]
   for name in a:p7.append({'comparison':'legacy_vs_'+comp,'feature':name,'feature_abs_diff':abs(a[name]-b[name])})
 write(OUT/'p7_r84_feature_comparison.csv',p7)
 summary=audit.agg(records,['preprocessing','client']); rows={(r['preprocessing'],r['client']):r for r in summary}
 lines=['# Preprocessing mechanism summary','', '| Hypothesis | Status | Evidence | Impact |','|---|---|---|---|',
 '| H1 concentration-label mismatch | excluded | filename-derived label constant | not causal |','| H2 raw timestamp jitter only | measured | P0 | see raw distribution |','| H3 large raw gaps | measured | P0/P3 | diagnostic |','| H4 duplicate timestamp handling | measured | P0 | diagnostic |','| H5 global interpolation distortion | measured | P1/P6 | diagnostic only |','| H6 baseline G0 propagation | measured | P2 | diagnostic only |','| H7 large-gap interpolated windows | measured | P3 | diagnostic only |','| H8 C5 Methane 225 raw-data anomaly | measured | P0/P4/P6 | diagnostic only |','| H9 R84 feature sensitivity | measured | P7 | diagnostic only |','| H10 legacy/time-bin similarity | measured | P6 | candidate evidence only |','', '## Oracle-route P6 RMSE (diagnostic, no classifier retraining)','', '| Preprocessing | C3 | C4 | C5 |','|---|---:|---:|---:|']
 for v in ('legacy','interp','timebin','timebin_short'):
  lines.append('| '+v+' | '+' | '.join(f"{rows[(v,c)]['Oracle_RMSE']:.4f}" for c in ('C3','C4','C5'))+' |')
 lines+=['','## Answers','', '1. Label mismatch is excluded by constant filename-derived nominal labels and fixed physical members.', '2. Raw irregularity is quantified per file in P0; do not infer it from nominal 100 Hz.', '3. Continuous interpolation is not selected here; P6 is a diagnostic-only comparison.', '4. The first largest channel difference is listed in P1.', '5. G0 differences are reported in P2; counterfactual causality remains diagnostic.', '6. P3 reports bucketed errors and Spearman coefficients.', '7. C5 methane 225 is separately reported in P0/P4/P6.', '8. P6 table above gives all three targets.', '9. No canonical production candidate is selected from target test.', '10. Full 25-round comparison is not authorized by this audit alone.']
 (OUT/'PREPROCESSING_MECHANISM_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 for name,text in {'P3_INTERPOLATION_ERROR_AUDIT.md':'# P3 interpolation/error audit\n\nSee `p3_interpolation_error_relationship.csv`; Spearman rows are labeled `SPEARMAN_*`.','P7_R84_FEATURE_AUDIT.md':'# P7 R84 feature audit\n\nSee `p7_r84_feature_comparison.csv`, paired by physical filename/window key.','P6_LOW_COST_REGRESSION_AUDIT.md':'# P6 low-cost regression diagnostic\n\nTrue/oracle route, fixed R84, calibration-only internal alpha selection; no classifier was retrained.'}.items():(OUT/name).write_text(text+'\n',encoding='utf-8')
 print({'records':len(records),'p7':len(p7)})
if __name__=='__main__':main()
