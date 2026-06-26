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
| H8+C4 specialist report | `results/h8_c4_deployable_specialist_validation_20260626.md` |
| profile selector output | `results/target_profile_selector_20260626/selected_profiles.json` |
| bidirectional report | `results/bidirectional_profile_selection_20260626/bidirectional_profile_selection_report.md` |

## Current Limitations

1. H8+C4 route-rescue has very low recall by design: it is a precision guard for one observed C4 wrong-route pattern.
2. Deployment-lite is not established. L1 needs an exported runtime bundle and size/latency benchmark before promotion.
3. Low-calibration stress testing is still pending for 20%, 10%, and 5% calibration ratios.
4. QC summary should be reported after model selection, not used to select model ability.
5. The reverse direction validates direction-specific selection, but it does not yet have a full runtime-ready artifact chain equivalent to forward H8+C4.

## Recommended Final-Stage Work

1. Run low-calibration stress for B0/H2.3/H8+C4 selector at 20%, 10%, and 5%.
2. Decide whether L1 is worth exporting; keep it out of mainline unless benchmarked advantage is clear.
3. Produce a QC post-hoc reliability report for H2.3 and H8+C4.
4. Convert this summary into thesis/meeting figures and tables.
