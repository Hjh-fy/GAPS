"""P2 fixed, read-only G0 counterfactual diagnostic (not a formal result)."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WS=ROOT.parents[1];sys.path.insert(0,str(ROOT))
import tools.run_preprocessing_mechanism_audit as audit
import tools.complete_preprocessing_mechanism_audit as completed

OUT=ROOT/'results/iotj_preprocessing_mechanism_audit_20260807'; NEW=WS/'dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid'
def windows(rel, start, end):
 crop=rel[start:end];return [crop[i:i+100].astype(np.float32) for i in range(0,len(crop)-99,50)]
def main():
 raw=audit.rawmap();caches={'A_timeaware_response_legacy_G0':{},'B_legacy_response_timeaware_G0':{}}
 rows=[]
 for p in raw.values():
  inf=audit.info(p)
  if inf['client_id'] not in (3,4,5):continue
  a,b=audit.legacy(p),audit.current(p)
  # A: exactly current 10-Hz response coordinates with legacy G0 inserted.
  ga=1/np.clip(b['sensor10'],1e-10,None);ra=(ga-a['baseline'])/a['baseline']
  # B: exactly legacy row-clock 10-Hz response coordinates with time-aware G0 inserted.
  gb=1/np.clip(a['sensor10'],1e-10,None);rb=(gb-b['baseline'])/b['baseline']
  for label,ww in [('A_timeaware_response_legacy_G0',windows(ra,600,1701)),('B_legacy_response_timeaware_G0',windows(rb,400,1500))]:
   for j,w in enumerate(ww):caches[label][(inf['filename'],round(60+5*j,6))]=(w,inf,True)
  for ch in range(8):rows.append({**inf,'channel':ch,'timeaware_response_legacy_G0_relative_rmse':audit.metrics(b['relative'][:,ch],ra[:,ch])['RMSE'],'legacy_response_timeaware_G0_relative_rmse':audit.metrics(a['relative'][:,ch],rb[:,ch])['RMSE']})
 rec=[];sel=[]
 for target in ('C3','C4','C5'):
  for label,cache in caches.items():
   r,s=audit.regression(label,NEW,target,OUT,cache);rec+=r;sel+=s
 audit.write(OUT/'p2_baseline_counterfactual_signal_difference.csv',rows);audit.write(OUT/'p2_baseline_counterfactual_regression_records.csv',rec);audit.write(OUT/'p2_baseline_counterfactual_alpha_selection.csv',sel)
 audit.write(OUT/'p2_baseline_counterfactual_regression.csv',audit.agg(rec,['preprocessing','client']))
 print({'records':len(rec)})
if __name__=='__main__':main()
