# GAPS Final Mainline Summary - 2026-06-26

This document summarizes the current system-level conclusion after the regression, calibration, deployment-runtime, guardrail, and benchmark work.

## Executive Decision

Current GAPS profiles:

| Mode | Current Profile | Status | Use Case |
|---|---|---|---|
| balanced | H2.3 | default mainline | no-QC full-set regression with stable global/nonCO trade-off |
| co_priority | H8 + formal C4 route rescue | guarded runtime-ready specialist | CO/high-CO risk-priority deployment |
| deployment_lite | H2.3 fallback | L1 not established | low-resource mode is pending an exported L1 runtime bundle and benchmark |
| QC | post-hoc reliability layer | separate from model selection | accept/review/reject output governance |

Do not select models by QC-accepted RMSE. Model ability is judged by no-QC full-set `final_ppm` / `co_corrected_ppm` metrics; QC is reported separately as a deployment reliability layer.

## Classification Backbone

The fixed-DA classification/backbone line remains the current classification base. Classification is not the present bottleneck; the recent work focused on target-side regression and calibration. The deployment flow keeps classification as the route into regression:

```text
fixed-DA classifier/backbone
-> source regression base / target profile
-> target-side residual/direct-head correction
-> QC decision
-> runtime output schema
```

## Regression Base: R3aK16

R3aK16 remains the source-domain neural regression base and reference, not the final ppm output by itself.

The current interpretation is:

```text
fixed-DA route
-> R3aK16 source regression base
-> target calibration/direct-head/specialist policy
-> co_corrected_ppm
-> QC / auto_output_ppm
```

R3aK16 is kept because it is the stable source regression reference and the basis for comparison. It is not treated as theoretically final or automatically optimal.

## Forward Direction: C12 -> C345

| Profile | Role | ALL RMSE | ALL NRMSE | CO RMSE by target | CO high RMSE by target | nonCO ALL RMSE |
|---|---|---:|---:|---|---|---:|
| baseline final | baseline | 27.34 | 0.1578 | C3=33.70; C4=56.59; C5=46.12 | C3=41.70; C4=95.32; C5=60.00 | 19.00 |
| H2.3 | balanced mainline | 18.62 | 0.1326 | C3=16.15; C4=22.02; C5=26.85 | C3=20.02; C4=34.24; C5=34.82 | 17.83 |
| H8+C4 | runtime-ready CO specialist | 18.30 | 0.1350 | C3=14.97; C4=17.16; C5=23.69 | C3=19.93; C4=26.79; C5=27.54 | 18.38 |

H2.3 remains the balanced default because it gives the cleanest overall/nonCO balance. H8+C4 is selected only in `co_priority` mode because it improves CO/high-CO metrics while slightly worsening nonCO compared with H2.3.

## H2.3 No-B0 Feature Ablation

A0-A6 tested whether the H2.3 target direct-head profile can be simplified by removing B0/R3aK16/auto_v2 support. The forward C12 -> C345 run used the existing formal target-head outputs and compared the current H2.3 profile with direct-only variants.

Direct-head feature construction was checked against `run_regression_head_ablation.add_target_features`: the target heads use target-window rich statistics and metadata, not B0/R3aK16/auto_v2 ppm fields. The B0-like keys checked were `final_ppm`, `auto_v2_ppm`, `baseline_final_ppm`, `base_r3ak16_raw_ppm`, `routed_pred_ppm`, `risk_score`, and `confidence_margin`; none are direct-head training features.

| Mode | Reading | ALL RMSE | ALL NRMSE | C3 NRMSE | C4 NRMSE | C5 NRMSE | Macro-client NRMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | B0 R3aK16/auto_v2 baseline | 27.34 | 0.1578 | 0.1108 | 0.1520 | 0.2272 | 0.1633 |
| A1 | current H2.3 with B0-dependent C4 rescue/profile layer | 18.62 | 0.1326 | 0.1023 | 0.0713 | 0.2100 | 0.1279 |
| A2-A6 | direct target heads only; no B0-dependent rescue | 22.39 | 0.1436 | 0.1023 | 0.1311 | 0.2100 | 0.1478 |

Conclusion:

- A2-A6 are identical, confirming that B0/R3aK16/auto_v2 ppm values and QC-risk scalars are not target direct-head feature inputs.
- However, no-B0 direct-only is not close enough to current H2.3: macro-client NRMSE degrades from 0.1279 to 0.1478, mainly because C4 NRMSE degrades from 0.0713 to 0.1311.
- The C4 high-CO error confirms the same pattern: current H2.3 gives C4 high-CO RMSE/NRMSE 34.24 / 0.1522, while direct-only gives 96.70 / 0.4298.
- Therefore R3aK16/auto_v2 should not be described as target direct-head feature inputs, but they should remain in the current runtime/mainline as the baseline, fallback, audit, and route-rescue support layer.
- Under the planned decision rule, reverse C45 -> C123 no-B0 ablation was not run because the forward no-B0 version is not close to H2.3.

Entrypoints:

- `run_h2_3_no_b0_feature_ablation.py`
- `results/h2_3_no_b0_feature_ablation_20260629/c12_c345/h2_3_no_b0_feature_ablation_report.md`

## H8+C4 Runtime-Ready Specialist Evidence

H8+C4 now has a complete deployment evidence chain:

```text
calibration-only gate
-> v2 export with max_conf_margin
-> runtime artifact
-> feature schema validation
-> guardrail audit
-> runtime-vs-analysis parity
-> CPU benchmark
-> auto_output_ppm schema
```

Key validation results:

| Check | Result |
|---|---:|
| feature schema | pass |
| guardrail status | pass |
| runtime rows compared | 5400 |
| runtime mismatch rows | 0 |
| runtime max_abs_diff | 1.42e-13 |
| C4 route-rescue hit_N | 1 |
| false hit_N | 0 |
| nonCO hit_N | 0 |
| C4 high-CO recall | 0.0098 |

Important limitation: the formal C4 route-rescue is a high-precision surgical rescue, not broad high-CO coverage. It fixes one high-risk C4 high-CO wrong-route window in the current test split with zero false/nonCO hits.

## Reverse Direction: C45 -> C123

| Profile | Role | ALL RMSE | ALL NRMSE | CO RMSE by target | CO high RMSE by target | nonCO ALL RMSE |
|---|---|---:|---:|---|---|---:|
| baseline final | baseline | 22.94 | 0.1473 | C1=37.68; C2=22.00; C3=32.31 | C1=51.97; C2=25.56; C3=38.16 | 19.34 |
| target Ridge direct | balanced mainline | 15.59 | 0.1123 | C1=23.77; C2=15.55; C3=14.68 | C1=38.30; C2=17.09; C3=16.37 | 14.50 |
| H8-style source-aug CO else Ridge | diagnostic CO specialist | 16.13 | 0.1192 | C1=24.70; C2=15.78; C3=10.73 | C1=39.70; C2=17.35; C3=11.04 | 15.44 |

The reverse direction supports the profile-selection story rather than hard-coded H8+C4. C4 is a source client in C45 -> C123, so the forward C4 rescue rule is not applicable. Target Ridge direct is the clean balanced mainline; H8-style switching stays diagnostic.

## Runtime / Size / Latency

First CPU benchmark: 300 target test windows per client, C3/C4/C5, 3 repeats.

| Profile | Status | Artifact MB | Model MB | Mean ms/window | P90 ms/window | Expected full ALL RMSE |
|---|---|---:|---:|---:|---:|---:|
| H2.3 | ok | 24.0419 | 21.2380 | 2.7344 | 3.2869 | 18.62 |
| H8 | ok | 24.6380 | 21.2380 | 2.6319 | 2.8498 | 18.47 |
| H8+C4 | ok | 24.6414 | 21.2380 | 2.5857 | 2.7215 | 18.30 |
| L1 | missing_bundle | 0.0000 | 0.0000 |  |  | 22.6 |
| B0 | missing_bundle | 0.0000 | 0.0000 |  |  | 27.34 |

H8+C4 does not introduce a meaningful runtime burden compared with H2.3/H8. L1 cannot be called deployment-lite yet because no exported runtime bundle has been benchmarked.

## Low-Calibration Stress

Low-calibration stress was run with stratified target-calibration subsets at 20%, 10%, and 5%. QC was not used. Each ratio refits the target-side heads with calibration-only selection, then evaluates the fixed target test set.

| Calibration | Selector profile | B0 ALL | H2.3 ALL | H8+C4 forced ALL | Selector ALL | H2.3 nonCO | Test gate false/nonCO |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20% | H2.3 fallback | 27.34 | 18.86 | 18.24 | 18.86 | 17.16 | 0 / 0 |
| 10% | H2.3 fallback | 27.34 | 23.23 | 22.60 | 23.23 | 21.21 | 0 / 0 |
| 5% | H2.3 fallback | 27.34 | 27.47 | 26.87 | 27.47 | 24.83 | 0 / 0 |

Reading:

- At 20% and 10%, H2.3 still improves no-QC full-set ALL RMSE over B0; at 5%, the refit becomes too weak and no longer clearly beats B0.
- H8+C4 forced sometimes improves test ALL RMSE, but the calibration-only selector does not enable it because calibration CO RMSE does not beat H2.3. This is the correct conservative behavior: test is not used for selection.
- Low-calibration route-rescue gates had zero test false hits and zero test nonCO hits after enforcing the same gate schema (`max_final`, `min_risk`, `max_conf_margin`) used by the selector.
- Therefore, the runtime-ready H8+C4 specialist remains a full-calibration CO-priority profile, while low-calibration mode should default to H2.3 fallback unless a future selector has stronger calibration evidence.

Entrypoints:

- `run_low_calib_stress_profiles.py`
- `results/low_calib_stress_profiles_20260626/low_calib_stress_report.md`

## QC Post-Hoc Reliability

QC remains a post-hoc deployment reliability layer, not a model-selection criterion.

| Profile | Role | Full RMSE | Accept coverage | Accept RMSE | Review coverage | Reject coverage | Accepted+review RMSE | High-error recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H2.3 | balanced | 18.62 | 0.4233 | 5.92 | 0.3411 | 0.2356 | 9.07 | 0.9089 |
| H8+C4 | co_priority | 18.30 | 0.4233 | 5.48 | 0.3411 | 0.2356 | 8.06 | 0.9033 |

Under the same QC routing, H8+C4 has lower accepted and accepted+review RMSE than H2.3, while both profiles preserve similar high-error recall. These values support deployment reliability reporting only; they do not replace the no-QC full-set model-capability decision.

Entrypoints:

- `summarize_qc_posthoc_profiles.py`
- `results/qc_posthoc_reliability_20260626/qc_posthoc_report.md`

## Runtime Output Schema

Current public runtime fields:

```text
gas_class
gas_name
class_prob
base_r3ak16_raw_ppm
routed_pred_ppm
final_ppm
co_corrected_ppm
auto_output_ppm
qc_decision
risk_score
```

`final_ppm` and `co_corrected_ppm` are retained for audit. `auto_output_ppm` is filled only when `qc_decision == "accept"`; review/reject rows keep model predictions for traceability but do not silently produce automatic ppm output.

## Current Code and Result Entrypoints

| Purpose | File |
|---|---|
| mainline role map | `docs/mainline_entrypoints_20260626.md` |
| next-stage execution plan | `docs/gaps_next_stage_execution_plan_20260626.md` |
| runtime feature contract | `docs/feature_schema_and_runtime_contract_20260626.md` |
| deployment selector | `select_target_profile.py` |
| direction-level summary | `summarize_bidirectional_profile_selection.py` |
| runtime benchmark | `benchmark_runtime_profiles.py` |
| low-calibration stress | `run_low_calib_stress_profiles.py` |
| QC post-hoc reliability | `summarize_qc_posthoc_profiles.py` |
| H2.3 no-B0 ablation | `run_h2_3_no_b0_feature_ablation.py` |
| H8+C4 specialist report | `results/h8_c4_deployable_specialist_validation_20260626.md` |
| profile selector output | `results/target_profile_selector_20260626/selected_profiles.json` |
| bidirectional report | `results/bidirectional_profile_selection_20260626/bidirectional_profile_selection_report.md` |
| low-calibration report | `results/low_calib_stress_profiles_20260626/low_calib_stress_report.md` |
| QC post-hoc report | `results/qc_posthoc_reliability_20260626/qc_posthoc_report.md` |
| H2.3 no-B0 ablation report | `results/h2_3_no_b0_feature_ablation_20260629/c12_c345/h2_3_no_b0_feature_ablation_report.md` |

## Current Limitations

1. H8+C4 route-rescue has very low recall by design: it is a precision guard for one observed C4 wrong-route pattern.
2. Deployment-lite is not established. L1 needs an exported runtime bundle and size/latency benchmark before promotion.
3. Low-calibration 5% is not enough for reliable target profile refitting; 20% and 10% are more credible, with 20% remaining the strongest setting.
4. QC summary should be reported after model selection, not used to select model ability.
5. The reverse direction validates direction-specific selection, but it does not yet have a full runtime-ready artifact chain equivalent to forward H8+C4.
6. H2.3 target direct heads do not use B0/R3aK16/auto_v2 ppm as training features, but the current best forward H2.3 result still depends on the B0/risk-supported C4 rescue/profile layer; direct-only no-B0 is weaker.

## Recommended Final-Stage Work

1. Decide whether L1 is worth exporting; keep it out of mainline unless benchmarked advantage is clear.
2. Keep H8+C4 as a 20% calibration CO-priority specialist; do not generalize the C4 gate beyond calibration-selected target clients.
3. When presenting H2.3, distinguish target direct-head calibration from the runtime support layer: R3aK16/auto_v2 are not direct-head feature inputs, but they remain useful for baseline/fallback/audit/rescue.
4. Convert this summary into thesis/meeting figures and tables.
