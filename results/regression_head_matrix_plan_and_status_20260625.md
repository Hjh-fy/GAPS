# Regression Head Matrix Plan and Status

Date: 2026-06-25

Main rule: performance first. A regression-head or calibration method is promoted only if it improves no-QC full-set target final RMSE / NRMSE, then CO and CO-high RMSE, without obvious nonCO damage. QC is not used for model selection here.

## Current Baselines

| item | status | key result |
| --- | --- | --- |
| R3aK16 + original auto_v2 | stable reference | ALL RMSE 27.34, NRMSE 0.1578, C4 CO high 95.32, C5 CO high 60.00 |
| target direct-head auto_v2 H2.3 | current best mainline | ALL RMSE 18.62, NRMSE 0.1326, C3/C4/C5 CO 16.15/22.02/26.85 |
| H8 stratified direct-head | CO-specialist candidate | ALL RMSE 18.47, C4/C5 CO 19.76/23.69, but nonCO and NRMSE trade-off need caution |

## Matrix A: Existing R3aK16 Structure Ablation

Source: `summarize_r3ak16_structure_ablation.py`

Output:

- `results/r3ak16_structure_ablation_20260625/structure_ablation_summary.csv`
- `results/r3ak16_structure_ablation_20260625/structure_ablation_scope_metrics.csv`
- `results/r3ak16_structure_ablation_20260625/r3ak16_structure_ablation_report.md`

Important note: these artifacts compare structure candidates after existing target-side specialist/calibration output, not pure source-only transfer.

| candidate | meaning | conclusion |
| --- | --- | --- |
| M0 R3aK16 depth4 DCT16 | current structural baseline | best among existing structure candidates: ALL RMSE 23.31 |
| M1 R3aK8 depth4 DCT8 | smaller DCT branch | only tiny parameter saving, ALL/CO worsen |
| M2 MSConv16 | classic local convolution response branch | worsens CO, especially C5 high |
| M3 TCN adapter | response adapter | worsens ALL/CO |
| M4 shared trunk | truly lightweight neural structure | much smaller, but ALL/nonCO worsen too much |
| M5 ratio + DCT | ratio auxiliary branch | helps C4 high, but global/nonCO trade-off too large |

Decision: no existing structural replacement beats M0 under current metric priority.

## Matrix B: Source Lightweight Head + Target Calibration

Existing outputs:

- `results/source_lightweight_regression_head_ablation_20260625_lite/source_lightweight_head_ablation_report.md`
- `results/source_lightweight_target_calibrated_20260625_lite/source_lightweight_target_calibrated_report.md`
- `results/lightweight_fair_auto_v2_experiment_plan_20260625.md`
- `results/source_lightweight_full_auto_v2_20260625_fair/source_lightweight_full_auto_v2_report.md`
- `results/lightweight_fair_matrix_20260625/lightweight_fair_matrix_report.md`
- `results/lightweight_l2_unified_selector_20260625/l2_unified_selector_report.md`

Key reading:

- Source lightweight heads fit C1/C2 well, especially per-gas MLP source test ALL RMSE 5.25.
- Direct transfer to C3/C4/C5 collapses, so source fit quality is not enough.
- Adding target affine calibration rescues a lot, but still underperforms the original baseline:

| mode | ALL RMSE | C4 CO | C5 CO | nonCO |
| --- | ---: | ---: | ---: | ---: |
| R3aK16 baseline final | 27.34 | 56.59 | 46.12 | 19.00 |
| source ridge + target affine | 40.17 | 82.34 | 44.23 | 32.63 |
| source per-gas MLP + target affine | 39.41 | 73.29 | 59.68 | 29.96 |
| source shared MLP + target affine | 36.87 | 60.33 | 52.08 | 31.69 |

Fair follow-up:

- `run_source_lightweight_full_auto_v2_eval.py` now evaluates lightweight source heads with full target residual auto_v2.
- The target calibration split is internally split into calibration-fit/calibration-validation.
- Residual candidates are selected per target client/gas using calibration-validation only.
- Selected candidates are refit on full target calibration.
- Target test is used only once for final reporting.

This changes the conclusion materially:

| mode | ALL RMSE | C3 CO | C4 CO | C5 CO | C4 high | C5 high | nonCO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H8 CO-specialist reference | 18.47 | 14.97 | 19.76 | 23.69 | 32.22 | 27.54 | 18.38 |
| H2.3 mainline reference | 18.62 | 16.15 | 22.02 | 26.85 | 34.24 | 34.82 | 17.83 |
| best diagnostic lightweight forced piecewise | 21.96 | 14.27 | 53.02 | 27.79 | 95.73 | 34.88 | 17.56 |
| source ridge + full residual auto_v2 | 22.62 | 17.04 | 48.35 | 25.83 | 85.87 | 29.53 | 19.54 |
| source shared MLP + full residual auto_v2 | 22.63 | 14.27 | 53.14 | 27.25 | 95.38 | 29.73 | 18.70 |
| source per-gas MLP + full residual auto_v2 | 22.76 | 15.73 | 53.71 | 24.87 | 96.37 | 27.62 | 18.85 |
| R3aK16 baseline final | 27.34 | 33.70 | 56.59 | 46.12 | 95.32 | 60.00 | 19.00 |

Decision: affine-only calibration was not enough, but full residual auto_v2 makes lightweight source heads credible deployment-lite candidates. They now beat the original R3aK16 baseline on ALL RMSE and CO for C3/C5, but still do not reach H2.3/H8 because C4 CO and C4 high-CO remain weak. Do not replace the performance mainline yet; keep lightweight heads as optional candidates for parameter/runtime evaluation and a future unified L2 selector.

L2 selector follow-up:

- `run_lightweight_l2_unified_selector.py` compares B0 baseline, target Ridge direct, target MLP direct, and three L1 lightweight full-auto_v2 candidates using calibration-validation RMSE.
- Per-client/gas selector chooses lightweight candidates in 8/12 cells, but test ALL worsens to 23.58. This indicates cell-level calibration-val selection is unstable with small validation cells.
- Conservative client-level selector chooses lightweight candidates for all three target clients, but reaches only ALL RMSE 22.63, similar to the best L1 single candidate and worse than target MLP direct 21.82.

| mode | ALL RMSE | C3 CO | C4 CO | C5 CO | C4 high | C5 high | nonCO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target MLP direct | 21.82 | 16.15 | 48.85 | 31.03 | 85.05 | 39.41 | 17.61 |
| L2 client-level selector | 22.63 | 14.27 | 53.71 | 24.87 | 96.37 | 27.62 | 18.84 |
| L2 per-client/gas selector | 23.58 | 14.27 | 57.35 | 28.06 | 96.82 | 27.62 | 19.12 |

L2 decision: lightweight heads do contain useful validation signal, but selector-level mixing does not create a new performance mainline. The best use is still deployment-lite evaluation or as optional ingredients inside a carefully guarded selector, not free per-gas switching.

L3 named matrix follow-up:

- `run_l3_lightweight_hybrid_matrix.py` explicitly tests lightweight-base + H2.3/H8/C4-rescue combinations.
- Output: `results/l3_lightweight_hybrid_matrix_20260626/l3_lightweight_hybrid_matrix_report.md`
- No new training is performed; the script combines existing L1 lightweight full-auto_v2 predictions, H2.3/H8 predictions, source-augmented target Ridge predictions, and the formal calibration-selected C4 route-rescue gate.

| mode | ALL RMSE | ALL NRMSE | C4 CO | C4 high | C5 high | nonCO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2.3 reference | 18.62 | 0.1326 | 22.02 | 34.24 | 34.82 | 17.83 |
| H8 + formal C4 rescue | 18.30 | 0.1350 | 17.16 | 26.79 | 27.54 | 18.38 |
| L3 source per-gas MLP CO switch + H2.3 fallback + formal C4 rescue | 18.30 | 0.1339 | 17.98 | 27.38 | 27.42 | 18.20 |
| L3 source shared MLP CO switch + H2.3 fallback + formal C4 rescue | 18.44 | 0.1351 | 17.50 | 26.54 | 29.54 | 18.34 |
| L3 light H2.3 analogy + formal C4 rescue | 19.69 | 0.1411 | 25.19 | 41.16 | 27.62 | 19.26 |
| Best strict lightweight full-auto_v2 base | 22.62 | 0.1492 | 48.35 | 85.87 | 29.53 | 19.54 |

L3 decision: full lightweight replacement is still not good enough, mainly because C4 CO/high-CO remains weak. However, selective lightweight CO switching is much stronger than expected. The per-gas MLP CO-switch plus formal C4 rescue is effectively tied with H8 + formal C4 rescue on ALL RMSE while improving NRMSE/nonCO slightly, so it should be kept as a deployment-lite CO-specialist candidate. The shared-MLP CO switch gives the best C4 high-CO RMSE but loses more on ALL/C5.

## Matrix C: Missing Lightweight Structural Runs

These are worth running next only if we want a neural lightweight replacement rather than a post-hoc direct head.

| candidate | purpose | structure |
| --- | --- | --- |
| M6 depth2 DCT16 | test shallow residual head while keeping response statistics | `--reg-head-depth 2 --reg-response-branch dct --reg-dct-k 16` |
| M7 depth2 none | compact classic MLP-style head | `--reg-head-depth 2 --reg-response-branch none` |
| M8 depth4 none | isolate value of DCT branch | `--reg-head-depth 4 --reg-response-branch none` |

Required evaluation for each:

1. train/federated regression checkpoint on C1/C2;
2. run the same target-side specialist/calibration flow;
3. evaluate no-QC full-set target test;
4. compare against M0 and H2.3/H8.

Stop rule: if a candidate cannot approach M0 after the same target calibration, do not promote it. If it is close to M0 with much lower parameter count, keep it as deployment-lite candidate but not performance mainline.

## Matrix D: Mainline Performance Work

The most valuable line remains target direct-head auto_v2, because it still beats both the R3aK16 baseline and the fair lightweight L1 results by a clear margin.

Formalization status:

1. `summarize_target_direct_head_mainline.py` now consolidates H2.3/H8 metrics, artifact checklist, runtime parity, and reproduction workflow.
2. H2.3 has deployment/runtime parity and remains the balanced mainline.
3. H8 has deployment/runtime parity and remains the CO-specialist candidate.
4. H8 + formal C4 route rescue now has a deployment candidate and runtime parity, so it can be used as a deployable CO-specialist rescue candidate.
5. H8 improves CO/high-CO but worsens ALL NRMSE/nonCO versus H2.3, so it should not silently replace H2.3 as the default.

Reverse-direction C45 -> C123 check:

- `run_c45_c123_optimal_config_analysis.py` analyzes source C4/C5 -> target C1/C2/C3 under the same no-QC full-set priority.
- Output: `results/c45_c123_optimal_config_analysis_20260626/c45_c123_optimal_config_analysis_report.md`
- Target test rows: 8040.

| mode | ALL RMSE | ALL NRMSE | C1 CO | C2 CO | C3 CO | C1 high | C2 high | C3 high | nonCO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline final | 22.94 | 0.1473 | 37.68 | 22.00 | 32.31 | 51.97 | 25.56 | 38.16 | 19.34 |
| target Ridge direct | 15.59 | 0.1123 | 23.77 | 15.55 | 14.68 | 38.30 | 17.09 | 16.37 | 14.50 |
| target MLP direct | 16.46 | 0.1201 | 23.84 | 17.31 | 14.93 | 39.73 | 22.13 | 18.47 | 15.49 |
| source-aug target Ridge | 16.01 | 0.1183 | 24.92 | 15.78 | 10.73 | 40.16 | 17.35 | 11.04 | 15.24 |
| H8-style source-aug CO else Ridge | 16.13 | 0.1192 | 24.70 | 15.78 | 10.73 | 39.70 | 17.35 | 11.04 | 15.44 |

Reverse decision: for C45 -> C123, the clean performance mainline is all-target Ridge direct. The C12 -> C345 H8/source-aug idea improves C3 CO/high-CO in reverse, but it worsens ALL RMSE and nonCO, so it should remain a diagnostic CO-specialist variant rather than the reverse default. C4 route-rescue does not apply because C4 is a source client in this direction.

## Matrix E: C4 CO High Diagnosis

New outputs:

- `diagnose_c4_co_high_mainline.py`
- `run_c4_route_rescue_upper_bound.py`
- `export_h8_formal_c4_rescue_deployment_candidate.py`
- `results/c4_co_high_mainline_diagnosis_20260625/c4_co_high_mainline_diagnosis_report.md`
- `results/c4_route_rescue_upper_bound_20260625/c4_route_rescue_upper_bound_report.md`
- `results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_selector_report.md`
- `results/deployment_h8_formal_c4_rescue_candidate_20260625/rich_residual_candidate.json`
- `results/runtime_validation_h8_formal_c4_rescue_candidate_20260625/runtime_summary.csv`
- `results/equivalence_h8_formal_c4_rescue_candidate_20260625/equivalence_summary.json`

Key diagnosis:

| scope | baseline | H2.3 | source-aug | H8 |
| --- | ---: | ---: | ---: | ---: |
| C4 high overall RMSE | 95.32 | 34.24 | 33.05 | 32.22 |
| C4 high pred_CO RMSE | 20.84 | 19.70 | 15.05 | 15.05 |
| C4 high pred_nonCO RMSE | 228.78 | 71.38 | 73.64 | 71.38 |

Reading:

- H8/source-aug already solves most C4 high cases when the route is predicted as CO.
- The remaining C4 high error is dominated by a small number of route-driven failures where true high CO is predicted as non-CO.
- Existing C4 route rescue already fixes many pred-Ethanol/recovery 250 ppm cases.
- The current large residual failures include one pred-Ethylene recovery 250 ppm case and one pred-Ethanol main-response 200 ppm case.

Test-only upper-bound:

- A diagnostic gate using deployment-visible fields (`client=C4`, `pred_class in {Ethanol, Ethylene}`, `final_ppm <= 20 or 30`, `risk_score >= 4`) with `rescue_ppm=250` hits 15-17 true C4 high cases with zero false hits on test.
- This lowers C4 high RMSE to 14.81 and ALL RMSE to about 18.05 in the upper-bound sweep.
- This is not yet a formal rule because it was selected on test. The next formal step is to run the same gate family on calibration-validation, select there, then evaluate once on test.

Formal calibration-selected C4 route-rescue:

- `run_formal_c4_route_rescue_selector.py` selects the route-rescue gate on target calibration only, then applies it to H8 test predictions.
- Selected gate: `client=C4`, `pred_class=Ethanol`, `final_ppm <= 20`, `risk_score >= 6`, `rescue_ppm=250`.
- Calibration support: 3/24 C4 high calibration hits, zero false hits.
- Test result: 14/102 C4 high test hits, zero false hits.

| mode | ALL RMSE | C4 high RMSE | C4 nonCO RMSE | nonCO ALL RMSE |
| --- | ---: | ---: | ---: | ---: |
| H8 pred-CO source-aug | 18.47 | 32.22 | 8.86 | 18.38 |
| H8 + formal C4 route rescue | 18.30 | 26.79 | 8.86 | 18.38 |

Runtime/export status:

- The formal C4 route-rescue gate is layered on top of the existing H8 route-rescue gate instead of replacing it.
- Runtime parity check compared 5400 target-test rows against the formal analysis output.
- Equivalence result: `num_mismatch=0`, `max_abs_diff=1.99e-13`, `mean_abs_diff=1.15e-14`.

Decision: C4 high is now clearly a route-rescue calibration problem more than a general concentration-head problem. The formal calibration-selected C4 route rescue gives a no-leakage gain with zero false hits, and it has been exported into a runtime-validated deployment candidate. It only recovers the Ethanol-route subset; the Ethylene-route miss remains the main gap between the formal result and the test-only upper bound.

## Current Interpretation

The original regression difficulty is not simply "the R3aK16 head is too complex" or "the head cannot fit source." The evidence says:

- source-domain concentration mapping can be fitted by even lightweight heads;
- target-domain ppm mapping shifts substantially across clients;
- source-only direct transfer is unreliable;
- full residual target calibration can rescue lightweight source heads much more than affine-only calibration;
- target-side direct heads are still the strongest performance mechanism;
- lighter neural structures are interesting as deployment-lite candidates if they keep target-calibrated full-set metrics close to the current best and provide a real parameter/runtime advantage.
- C4 high-CO now appears dominated by a few route-driven non-CO predictions, so the next performance gain should come from calibration-selected route rescue rather than another global regression head.
