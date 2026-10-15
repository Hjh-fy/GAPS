"""Write transparent evidence documents/plots for the completed diagnostic."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iotj_preprocessing_mechanism_audit_20260807'
def read(n):
 with (OUT/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def f(x):
 try:return float(x)
 except:return float('nan')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as q:
  for b in iter(lambda:q.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p0,p1,p2,p3,p4,p6,p2cf,p7=[read(x) for x in ('p0_raw_timestamp_stats.csv','p1_stagewise_difference.csv','p2_baseline_g0_comparison.csv','p3_interpolation_error_relationship.csv','p4_c5_methane_225_signal_stats.csv','p6_preprocessing_summary.csv','p2_baseline_counterfactual_regression.csv','p7_r84_feature_comparison.csv')]
 top=max(p1,key=lambda r:f(r['RMSE']));c5225=[r for r in p0 if r['client_id']=='5' and r['gas']=='methane' and f(r['concentration'])==225]
 # Plot 1: fixed-membership P6, with N visibly encoded in annotations.
 fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True);clients=['C3','C4','C5'];vars=['legacy','interp','timebin','timebin_short'];colors=['#4c78a8','#f58518','#54a24b','#b279a2'];x=np.arange(3);w=.19
 for i,v in enumerate(vars):
  rr=[next(r for r in p6 if r['preprocessing']==v and r['client']==c) for c in clients];ax.bar(x+(i-1.5)*w,[f(r['Oracle_RMSE']) for r in rr],w,label=v,color=colors[i]);
  for j,r in enumerate(rr):ax.text(x[j]+(i-1.5)*w,f(r['Oracle_RMSE'])+.3,f"N={r['N']}",rotation=90,ha='center',va='bottom',fontsize=7)
 ax.set_xticks(x,clients);ax.set_ylabel('Oracle-route RMSE (ppm)');ax.set_title('P6 diagnostic: fixed physical membership');ax.legend(ncol=2);ax.grid(axis='y',alpha=.25);fig.savefig(OUT/'diagnostic_plots/p6_oracle_rmse.png',dpi=220);plt.close(fig)
 # Plot 2: P3 buckets.
 rr=[r for r in p3 if not r['bucket'].startswith('SPEARMAN') and r['metric']=='interpolated_ratio'];fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);ax.bar(range(len(rr)),[f(r['RMSE']) for r in rr],color='#f58518');ax.set_xticks(range(len(rr)),[r['bucket'] for r in rr],rotation=25,ha='right');ax.set_ylabel('Current interpolation oracle RMSE (ppm)');ax.set_title('P3: error by interpolated-ratio bucket');ax.grid(axis='y',alpha=.25);fig.savefig(OUT/'diagnostic_plots/p3_interpolation_buckets.png',dpi=220);plt.close(fig)
 # Plot 3: G0 relative differences, C5 methane 225 only.
 z=[r for r in p2 if r['client_id']=='5' and r['gas']=='methane' and f(r['concentration'])==225];fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
 for rep in sorted(set(r['repeat_id'] for r in z)):
  q=[r for r in z if r['repeat_id']==rep];ax.plot([int(r['channel']) for r in q],[f(r['relative_difference_percent']) for r in q],marker='o',label=f'repeat {rep}')
 ax.set_xlabel('Sensor channel');ax.set_ylabel('|ΔG0| / |legacy G0| (%)');ax.set_title('C5 methane 225: baseline divergence');ax.legend();ax.grid(alpha=.25);fig.savefig(OUT/'diagnostic_plots/p2_c5_methane225_g0.png',dpi=220);plt.close(fig)
 p0txt=f'''# P0 raw timestamp audit

`p0_raw_timestamp_stats.csv` records all {len(p0)} raw files.  It reports true timestamp deltas, duplicate timestamps, gap thresholds, and observed sample count per real 100-ms bin.  C5 methane 225 has {len(c5225)} repeats; their maximum dt values are {', '.join(f"{f(r['dt_max']):.4g}s" for r in c5225)} and empty-bin ratios are {', '.join(f"{f(r['empty_bin_ratio']):.4g}" for r in c5225)}.  Thus nominal 100 Hz must not be treated as proof of an exact row clock.
'''
 p1txt=f'''# P1 stagewise preprocessing audit

The largest recorded channel-level divergence is at **{top['stage']}**, client {top['client_id']} {top['gas']} {top['concentration']} ppm repeat {top['repeat_id']}, channel {top['channel']}: RMSE={f(top['RMSE']):.5g}, MAE={f(top['MAE']):.5g}, Pearson={f(top['pearson']):.4g}.  The full per-channel table is `p1_stagewise_difference.csv`.  Stage comparisons use shared available duration because the legacy chain defines time by rows and the current chain defines time by retained timestamps.
'''
 p2txt='''# P2 baseline G0 audit

`p2_baseline_g0_comparison.csv` gives legacy/current G0 for every file/channel. `p2_baseline_counterfactual_regression.csv` reports two fixed counterfactuals: (A) time-aware response with legacy G0 and (B) legacy response with time-aware G0. They are diagnostic-only oracle-route Ridge refits; no classifier was retrained and no target test item selected alpha or preprocessing.

| Variant | C3 RMSE | C4 RMSE | C5 RMSE |
|---|---:|---:|---:|
'''+''.join(f"| {v} | "+' | '.join(f"{f(next(r for r in p2cf if r['preprocessing']==v and r['client']==c)['Oracle_RMSE']):.4f}" for c in ['C3','C4','C5'])+' |\n' for v in sorted(set(r['preprocessing'] for r in p2cf)))
 p4txt='''# P4 C5 methane 225 case study

`p4_c5_methane_225_signal_stats.csv` contains both repeats, every sensor channel, and both preprocessing paths with mean, standard deviation, extrema, response amplitude, slope, AUC, peak and trough. The diagnostic plot is `diagnostic_plots/p2_c5_methane225_g0.png`.
'''
 # P7 remains strictly paired: summarize only observed feature differences.
 groups={}
 for r in p7:
  name=r['feature'];family='channel_statistics' if name.startswith('ch') else 'global_statistics'
  groups.setdefault((r['comparison'],family),[]).append(f(r['feature_abs_diff']))
 family=[]
 for (comparison,family_name),x in sorted(groups.items()):family.append({'comparison':comparison,'feature_family':family_name,'N':len(x),'mean_abs_difference':float(np.mean(x)),'p95_abs_difference':float(np.quantile(x,.95)),'max_abs_difference':float(np.max(x))})
 with (OUT/'p7_r84_feature_family_summary.csv').open('w',encoding='utf-8',newline='') as q:
  w=csv.DictWriter(q,fieldnames=family[0].keys());w.writeheader();w.writerows(family)
 p7txt='''# P7 R84 feature audit

`p7_r84_feature_comparison.csv` contains pairwise physical-window feature differences for legacy vs current interpolation and legacy vs time-bin. `p7_r84_feature_family_summary.csv` aggregates the observed differences into global and per-channel statistic families. It intentionally does **not** infer concentration-feature correlation or coefficient-of-variation from the absolute-difference table; those quantities were not persisted per window and are therefore marked unavailable rather than reconstructed from test outcomes.
'''
 (OUT/'P0_RAW_TIMESTAMP_AUDIT.md').write_text(p0txt,encoding='utf-8');(OUT/'P1_STAGEWISE_PREPROCESSING_AUDIT.md').write_text(p1txt,encoding='utf-8');(OUT/'P2_BASELINE_AUDIT.md').write_text(p2txt,encoding='utf-8');(OUT/'P4_C5_METHANE_225_CASE_STUDY.md').write_text(p4txt,encoding='utf-8');(OUT/'P7_R84_FEATURE_AUDIT.md').write_text(p7txt,encoding='utf-8')
 audit='''# Experiment audit — preprocessing mechanism diagnostic

**Verdict: PASS for the stated diagnostic scope; BLOCKED for any formal preprocessing replacement.**

| Check | Status | Evidence |
|---|---|---|
| Existing assets immutable | PASS | all outputs confined to this directory |
| Same physical C3/C4/C5 membership | PASS | filename + window-start master keys |
| Classifier retraining | PASS | none performed |
| Target-test alpha/preprocessing selection | PASS | fixed Ridge grid selected only in calibration internal split; no canonical preprocessing selected |
| Formal-result replacement | BLOCKED | this audit is diagnostic-only |
| Time-bin coverage equality | MAJOR LIMITATION | N is reported in P6; invalid raw-missing windows are not silently filled |

This report does not approve changing the frozen formal preprocessing or `ceb6c78` evidence.
'''
 (OUT/'EXPERIMENT_AUDIT.md').write_text(audit,encoding='utf-8')
 manifest_path=OUT/'protocol_manifest.json'
 manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'experiment_id':'AUDIT-PREPROCESS-20260807','status':'completed_diagnostic','formal_assets_immutable':True,'no_classifier_training':True,'target_test_used_for_selection':False}
 manifest.update({'p2_counterfactual_completed':True,'p6_status':'completed_diagnostic','p7_status':'completed_diagnostic','plots':['p6_oracle_rmse.png','p3_interpolation_buckets.png','p2_c5_methane225_g0.png'],'formal_preprocessing_selection':'none'})
 (OUT/'protocol_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 index=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_index.json':index.append({'path':str(p.relative_to(OUT)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size})
 (OUT/'sha256_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
 print({'files':len(index),'largest_p1_stage':top['stage']})
if __name__=='__main__':main()
