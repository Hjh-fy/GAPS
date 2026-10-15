# C5 regression provenance and pipeline audit

**Final classification: B. DATA PROVENANCE DIFFERENCE.** Primary evidence is the legacy row-decimation versus time-aware timestamp-clean/interpolation preprocessing difference. Secondary factors are checkpoint-dependent S_CC masking and calibration split sensitivity. A confirmed client-order RNG coupling defect changes membership, but fixed-membership isolation shows it is not the dominant numerical cause.

| Hypothesis | Status | Evidence | Impact |
|---|---|---|---|
| H1 RMSE calculation bug | Rejected | Independent NumPy recomputation matches exactly (max delta 0) | None |
| H2 S_CC filter bug | Rejected | S_CC independently uses `route_correct == 1`; counts reproduce 1339 and 1346 | None |
| H3 row alignment bug | Rejected | Full-array feature/class/regression/phase/metadata and route index checks pass | None |
| H4 classifier checkpoint difference | Confirmed, secondary | Oracle is checkpoint-invariant; NEW-data S_CC changes 15.6551 to 20.2864 with mask | Secondary conditional-subset effect |
| H5 C5 calibration membership difference | Confirmed | Only 70/320 calibration windows overlap physically | Secondary; fixed-membership test does not restore NEW |
| H6 preprocessing version difference | Confirmed, primary | Same raw filenames, different upstream code/hashes; fixed OLD membership 12.0131 vs 22.4505 | Primary |
| H7 float64/float32 only | Rejected | Time-axis handling differs; P95 window RMSE 0.04910 and max 2.3893 exceed cast noise | Not an adequate explanation |
| H8 RNG client-order coupling | Confirmed defect | C5-entry RNG hashes differ; simulated calibration overlap 55/320 | Reproducibility/split defect |
| H9 R84 84-D feature difference | Confirmed downstream | Common builder; matched median 84-D RMSE 0.33865, P95 3.3581 | Propagates preprocessing difference |
| H10 calibration internal split instability | Confirmed, not primary | NEW five-seed oracle mean/std 27.920/5.928, all 24.033--37.441; OLD 12.561--13.291 | Adds variance but NEW is consistently worse |
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

- OLD Methane oracle seed sensitivity: mean 12.853, SD 0.400, range 12.561--13.291.
- NEW Methane oracle seed sensitivity: mean 27.920, SD 5.928, range 24.033--37.441.

P10 is descriptive sensitivity only; no seed is selected.
