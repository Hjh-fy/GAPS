"""Finalize the read-only C5 audit narrative and hash index from persisted evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/iotj_c5_pipeline_audit_20260807"


def rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def main() -> None:
    p1=rows("p1_window_matching.csv"); p6=rows("p6_2x2_factorial_summary.csv"); iso=rows("p6_membership_preprocessing_factorial.csv"); p10=rows("p10_methane_seed_stability.csv")
    f={(r["experiment"],r["scope"]):r for r in p6}; i={(r["representation"],r["physical_membership"],r["scope"]):r for r in iso}
    cal_overlap=sum(r["old_split"]=="calibration" and r["new_split"]=="calibration" for r in p1)
    def stability(dataset: str, key: str) -> tuple[float,float,float,float]:
        a=np.array([float(r[key]) for r in p10 if r["dataset"]==dataset]); return float(a.mean()),float(a.std(ddof=1)),float(a.min()),float(a.max())
    old_st=stability("OLD","oracle_route_RMSE"); new_st=stability("NEW","oracle_route_RMSE")
    p6_md=f"""# P6 factorial analysis

All four arms independently refit R84 on the corresponding calibration split. No test row enters alpha selection or fitting.

| Arm | Accuracy | Pipeline RMSE | S_CC RMSE | Oracle RMSE |
|---|---:|---:|---:|---:|
| A OLD checkpoint + OLD data | {100*float(f[('A_OLD_CKPT_OLD_DATA','PIPELINE_ALL')]['class_accuracy']):.2f}% | {float(f[('A_OLD_CKPT_OLD_DATA','PIPELINE_ALL')]['RMSE']):.4f} | {float(f[('A_OLD_CKPT_OLD_DATA','S_CC')]['RMSE']):.4f} | {float(f[('A_OLD_CKPT_OLD_DATA','ORACLE_ROUTE')]['RMSE']):.4f} |
| B OLD checkpoint + NEW data | {100*float(f[('B_OLD_CKPT_NEW_DATA','PIPELINE_ALL')]['class_accuracy']):.2f}% | {float(f[('B_OLD_CKPT_NEW_DATA','PIPELINE_ALL')]['RMSE']):.4f} | {float(f[('B_OLD_CKPT_NEW_DATA','S_CC')]['RMSE']):.4f} | {float(f[('B_OLD_CKPT_NEW_DATA','ORACLE_ROUTE')]['RMSE']):.4f} |
| C NEW checkpoint + OLD data | {100*float(f[('C_NEW_CKPT_OLD_DATA','PIPELINE_ALL')]['class_accuracy']):.2f}% | {float(f[('C_NEW_CKPT_OLD_DATA','PIPELINE_ALL')]['RMSE']):.4f} | {float(f[('C_NEW_CKPT_OLD_DATA','S_CC')]['RMSE']):.4f} | {float(f[('C_NEW_CKPT_OLD_DATA','ORACLE_ROUTE')]['RMSE']):.4f} |
| D NEW checkpoint + NEW data | {100*float(f[('D_NEW_CKPT_NEW_DATA','PIPELINE_ALL')]['class_accuracy']):.2f}% | {float(f[('D_NEW_CKPT_NEW_DATA','PIPELINE_ALL')]['RMSE']):.4f} | {float(f[('D_NEW_CKPT_NEW_DATA','S_CC')]['RMSE']):.4f} | {float(f[('D_NEW_CKPT_NEW_DATA','ORACLE_ROUTE')]['RMSE']):.4f} |

Oracle RMSE follows the data/calibration pipeline exactly (OLD 12.0132, NEW 20.9572) and is invariant to checkpoint. S_CC additionally changes with the checkpoint-specific correctness mask; on NEW data it is 15.6551 with OLD checkpoint and 20.2864 with NEW checkpoint.

## Fixed physical membership isolation

| Window representation | Membership | Oracle RMSE | Methane oracle RMSE |
|---|---|---:|---:|
| OLD | OLD | {float(i[('OLD','OLD','ALL')]['oracle_RMSE']):.4f} | {float(i[('OLD','OLD','Methane')]['oracle_RMSE']):.4f} |
| NEW | OLD | {float(i[('NEW','OLD','ALL')]['oracle_RMSE']):.4f} | {float(i[('NEW','OLD','Methane')]['oracle_RMSE']):.4f} |
| OLD | NEW | {float(i[('OLD','NEW','ALL')]['oracle_RMSE']):.4f} | {float(i[('OLD','NEW','Methane')]['oracle_RMSE']):.4f} |
| NEW | NEW | {float(i[('NEW','NEW','ALL')]['oracle_RMSE']):.4f} | {float(i[('NEW','NEW','Methane')]['oracle_RMSE']):.4f} |

Holding OLD physical calibration/test membership does not restore NEW preprocessing: RMSE changes from 12.0131 to 22.4505 (Methane 13.2915 to 40.1372). Conversely, under NEW membership, OLD numerical windows remain much better (12.3685 vs 20.9572). Preprocessing/numerical provenance is therefore primary; membership is secondary here.

Arms B/C are diagnostic-only because crossed tests can contain windows consumed by the checkpoint's original target calibration/adaptation.
"""
    (OUT/"P6_FACTORIAL_ANALYSIS.md").write_text(p6_md,encoding="utf-8")
    summary=f"""# C5 regression provenance and pipeline audit

**Final classification: B. DATA PROVENANCE DIFFERENCE.** Primary evidence is the legacy row-decimation versus time-aware timestamp-clean/interpolation preprocessing difference. Secondary factors are checkpoint-dependent S_CC masking and calibration split sensitivity. A confirmed client-order RNG coupling defect changes membership, but fixed-membership isolation shows it is not the dominant numerical cause.

| Hypothesis | Status | Evidence | Impact |
|---|---|---|---|
| H1 RMSE calculation bug | Rejected | Independent NumPy recomputation matches exactly (max delta 0) | None |
| H2 S_CC filter bug | Rejected | S_CC independently uses `route_correct == 1`; counts reproduce 1339 and 1346 | None |
| H3 row alignment bug | Rejected | Full-array feature/class/regression/phase/metadata and route index checks pass | None |
| H4 classifier checkpoint difference | Confirmed, secondary | Oracle is checkpoint-invariant; NEW-data S_CC changes 15.6551 to 20.2864 with mask | Secondary conditional-subset effect |
| H5 C5 calibration membership difference | Confirmed | Only {cal_overlap}/320 calibration windows overlap physically | Secondary; fixed-membership test does not restore NEW |
| H6 preprocessing version difference | Confirmed, primary | Same raw filenames, different upstream code/hashes; fixed OLD membership 12.0131 vs 22.4505 | Primary |
| H7 float64/float32 only | Rejected | Time-axis handling differs; P95 window RMSE 0.04910 and max 2.3893 exceed cast noise | Not an adequate explanation |
| H8 RNG client-order coupling | Confirmed defect | C5-entry RNG hashes differ; simulated calibration overlap 55/320 | Reproducibility/split defect |
| H9 R84 84-D feature difference | Confirmed downstream | Common builder; matched median 84-D RMSE 0.33865, P95 3.3581 | Propagates preprocessing difference |
| H10 calibration internal split instability | Confirmed, not primary | NEW five-seed oracle mean/std {new_st[0]:.3f}/{new_st[1]:.3f}, all {new_st[2]:.3f}--{new_st[3]:.3f}; OLD {old_st[2]:.3f}--{old_st[3]:.3f} | Adds variance but NEW is consistently worse |
| H11 Ridge alpha selection instability | Consequence, not root cause | NEW Methane alpha=10 wins a uniformly poor fixed validation sweep (45.1366 RMSE) | Regularization reacts to poor calibration geometry |
| H12 Methane-specific domain/data shift | Confirmed | NEW Methane oracle 37.4413; B5_GMe_F090_R1 has max matched-window RMSE 2.3893; 225 ppm dominates validation error | Largest gas-specific degradation |

## Direct answers

1. **11.8 -> 20.3 main cause:** changed processed numerical data/preprocessing provenance, not RMSE math. Fixed-membership oracle isolation worsens 12.0131 -> 22.4505 when only OLD-to-NEW window representation changes. The NEW checkpoint's S_CC mask is secondary.
2. **Code bug:** no evaluator, S_CC, row-alignment, R84-builder, or RMSE bug was found. A real split reproducibility defect exists: global RNG consumption is client-role/order coupled.
3. **Same physical windows:** yes at experiment and nominal window-position identity (1,680/1,680 Hungarian matches), but not the same numerical arrays; zero bit-identical windows.
4. **Does C1--C4 RNG consumption affect C5?** yes. Entry-state hashes differ and the exact simulation shares only 55/320 calibration indices.
5. **Why alpha=10?** it is the least-bad candidate on the frozen NEW Methane validation rows (45.1366 RMSE), chiefly reducing the extreme 225-ppm error versus smaller alpha. It does not indicate healthy generalization.
6. **Does fixed membership restore performance?** no. With OLD membership, OLD/NEW representations give 12.0131/22.4505 overall oracle RMSE and 13.2915/40.1372 Methane RMSE.
7. **Canonical dataset:** yes, before final paper evidence. Freeze one time-aware preprocessing implementation and a client/bucket-keyed split RNG, then regenerate manifests/hashes. Do not select OLD merely because it scores better.
8. **Need 25-round classifier rerun?** after canonical regeneration, yes for strict formal evidence, because GAPS consumes target calibration during adaptation and the split changes. R84 alone is insufficient. No rerun was started by this audit.

## Stability

- OLD Methane oracle seed sensitivity: mean {old_st[0]:.3f}, SD {old_st[1]:.3f}, range {old_st[2]:.3f}--{old_st[3]:.3f}.
- NEW Methane oracle seed sensitivity: mean {new_st[0]:.3f}, SD {new_st[1]:.3f}, range {new_st[2]:.3f}--{new_st[3]:.3f}.

P10 is descriptive sensitivity only; no seed is selected.
"""
    (OUT/"AUDIT_SUMMARY.md").write_text(summary,encoding="utf-8")
    manifest={"schema_version":"iotj.c5_pipeline_audit.v1","status":"COMPLETE_READ_ONLY_AUDIT","seed":42,
              "formal_training_started":False,"target_test_used_for_selection":False,
              "existing_assets_modified":False,"conclusion":"B. DATA PROVENANCE DIFFERENCE",
              "scripts":["tools/audit_c5_window_identity.py","tools/reproduce_c5_split_rng_state.py","tools/run_c5_pipeline_factorial_audit.py","tools/recompute_r84_metrics_independent.py","tools/finalize_c5_pipeline_audit.py"]}
    (OUT/"audit_protocol_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    artifacts=[]
    for path in sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="sha256_index.json"):
        artifacts.append({"path":str(path.relative_to(ROOT)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":sha(path)})
    (OUT/"sha256_index.json").write_text(json.dumps({"schema_version":"iotj.c5_pipeline_audit.sha256.v1","artifacts":artifacts},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
