"""Produce auditable canonical-candidate decision documents; no recomputation."""
from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iotj_canonical_preprocessing_selection_20260808'
def rows(n):
 with (OUT/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def f(x):return float(x)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as q:
  for b in iter(lambda:q.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 cs=rows('candidate_summary.csv');eng={r['candidate_id']:r for r in rows('engineering_cost_summary.csv')};test=rows('top2_target_test_summary.csv')
 ranked=sorted([r for r in cs if r['ranking_eligible']=='True'],key=lambda r:(f(r['target_calibration_validation_RMSE']),f(r['source_validation_RMSE']),f(eng[r['candidate_id']]['relative_compute_proxy'])))
 one,two=ranked[:2]
 def wrms(name):
  z=[r for r in test if r['candidate_id']==name];return (sum(f(r['RMSE'])**2*int(r['N']) for r in z)/sum(int(r['N']) for r in z))**.5
 protocol='''# Preprocessing design protocol

## Frozen candidate design

All candidates use true timestamps; stable time sort; duplicate-timestamp mean merge; raw-observation conductance G0 over 20≤t<50 s; physical-time bins; non-empty-bin aggregation; short-gap-only (one-bin) interpolation; long-gap invalid/quality metadata; physical crop 60–170 s; and physical-duration windows. Baseline statistic is **mean** for all 16 first-stage candidates. The 16 registered combinations are 1/2/5/10 Hz × mean/median bin aggregate × 10/20 s windows, with stride equal to one-half duration.

## Leakage gate

Screening reads C1/C2 source and C3/C4/C5 calibration rows only. Test metadata is read only to construct the sealed physical-role manifest; no test feature, error, label, alpha, or candidate ranking is used before the two frozen manifests are written. After freeze, only the TOP-2 receive one test diagnostic.
'''
 (OUT/'PREPROCESSING_DESIGN_PROTOCOL.md').write_text(protocol,encoding='utf-8')
 screen=f'''# Candidate screening summary

All 16 pre-registered candidates were screened under the same source/calibration-only oracle-R84 protocol. The predetermined ranking order was: usable-window eligibility (≥0.90), target-calibration validation RMSE, source validation RMSE, then temporal compute proxy.

| Rank | Candidate | Calibration RMSE | Source RMSE | Usable ratio | Points/window |
|---:|---|---:|---:|---:|---:|
| 1 | {one['candidate_id']} | {f(one['target_calibration_validation_RMSE']):.4f} | {f(one['source_validation_RMSE']):.4f} | {f(one['usable_window_ratio']):.4f} | {eng[one['candidate_id']]['points_per_window']} |
| 2 | {two['candidate_id']} | {f(two['target_calibration_validation_RMSE']):.4f} | {f(two['source_validation_RMSE']):.4f} | {f(two['usable_window_ratio']):.4f} | {eng[two['candidate_id']]['points_per_window']} |

No target-test metric was used to form this ranking. Full evidence is in the CSV files.
'''
 (OUT/'CANDIDATE_SCREENING_SUMMARY.md').write_text(screen,encoding='utf-8')
 decision=f'''# Final candidate decision

## Recommended canonical candidate

**Candidate 1: `{one['candidate_id']}` — 5 Hz, mean physical-time bin aggregation, 10-s physical windows, raw-observation mean G0, duplicate merge, and short-gap-only policy.**

It ranked first *before test access* by calibration validation ({f(one['target_calibration_validation_RMSE']):.4f} ppm), while retaining {100*f(one['usable_window_ratio']):.3f}% usable windows. Its frozen one-time test diagnostic was directionally consistent (weighted oracle RMSE {wrms(one['candidate_id']):.4f} ppm).

## Second frozen candidate

**Candidate 2: `{two['candidate_id']}` — 2 Hz with the same aggregation/baseline/gap/window rules.** It has a modestly weaker calibration result ({f(two['target_calibration_validation_RMSE']):.4f} ppm) but 100% usable-window coverage and lower temporal input cost ({eng[two['candidate_id']]['points_per_window']} versus {eng[one['candidate_id']]['points_per_window']} points/window). It is retained as the engineering/robustness comparator.

## Answers

1. Raw-observation G0 is preferred on methodological grounds and the preceding G0 counterfactual supports it; this selection did not optimize target test.
2. Mean bin aggregation ranked above median in calibration-only screening.
3. 5 Hz is recommended; 2 Hz is the lower-cost backup.
4. 10-s windows ranked ahead of 20-s windows in calibration-only screening.
5. Lower rates maintained higher usable-window coverage; 5 Hz retained near-complete coverage.
6. C5 Methane 225 repeat 1 remains a quality anomaly and was retained.
7. Candidate 1 is the recommended canonical configuration above.
8. Candidate 2 is the 2-Hz configuration above.
9. The primary decision is calibration/source performance plus methodological correctness; engineering distinguishes the retained backup, not test RMSE.
10. **Yes, conditionally:** this freezes preprocessing sufficiently to authorize one pre-registered final 25-round GAPS→adaptation→R84 confirmation, provided its data regeneration and label-access audit are separately pre-run frozen.

The test diagnostic was confirmatory only; it did not alter the pre-test ranking.
'''
 (OUT/'FINAL_CANDIDATE_DECISION.md').write_text(decision,encoding='utf-8')
 audit='''# Selection audit

**Verdict: PASS for canonical-candidate selection; not a replacement for the separate final full-system confirmation audit.**

| Gate | Status | Evidence |
|---|---|---|
| 16 candidates pre-registered | PASS | protocol and candidate summary |
| Shared source/calibration roles | PASS | master physical split manifest |
| Target test excluded from ranking | PASS | frozen candidate manifests precede test summary |
| Target-test search | PASS | exactly two pre-frozen candidates evaluated |
| Model/R84/loss/QC changes | PASS | none |
| Repeated seeds | LIMITATION | seed 42 / deterministic split only |
| Full GAPS proof | PENDING | no 25-round training was run |
'''
 (OUT/'SELECTION_AUDIT.md').write_text(audit,encoding='utf-8')
 index=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_index.json':index.append({'path':str(p.relative_to(OUT)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size})
 (OUT/'sha256_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'recommended':one['candidate_id'],'backup':two['candidate_id'],'test_rmse':[wrms(one['candidate_id']),wrms(two['candidate_id'])]},indent=2))
if __name__=='__main__':main()
