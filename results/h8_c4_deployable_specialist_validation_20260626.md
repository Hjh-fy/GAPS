# H8+C4 Deployable Specialist Validation

This note records the current deployable status of the H8 plus formal C4 route-rescue candidate after the v2 guard fix and `auto_output_ppm` runtime schema update.

## Positioning

H2.3 remains the balanced default mainline for no-QC full-set reporting. H8+C4 is a guarded CO-priority specialist candidate: it is useful when CO/high-CO risk matters more, but it should not replace H2.3 as the default balanced profile.

## Evidence Chain

| Check | Result | Evidence |
|---|---:|---|
| Calibration-only gate selection | pass | `results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_selected_gate.json` |
| Runtime artifact exported | pass | `results/deployment_h8_formal_c4_rescue_candidate_20260625` |
| Route-rescue schema | `c4_route_rescue_policy.v2` | `rich_residual_candidate.json` |
| `max_conf_margin` runtime guard | pass | `runtime_src/gaps_deploy/rich_residual.py` |
| Feature/runtime schema validation | pass | `results/feature_schema_validation_h8_formal_c4_rescue_20260626/feature_schema_validation.json` |
| Guardrail audit | pass | `results/h8_c4_guardrail_audit_20260626/h8_c4_guardrail_summary.json` |
| Runtime-analysis parity | pass | 5400 rows, 0 mismatch, max_abs_diff 1.42e-13 |
| Public output schema | pass | `auto_output_ppm` present in H8+C4 v2 runtime rows |

## Runtime Metrics

The re-exported H8+C4 v2 bundle preserves the previous no-QC model-only results:

| Prediction | Scope | RMSE | NRMSE |
|---|---|---:|---:|
| final_ppm | ALL | 27.3363 | 0.1578 |
| co_corrected_ppm | ALL | 18.3041 | 0.1350 |
| co_corrected_ppm | C4-CO | 17.1603 | 0.0763 |
| co_corrected_ppm | C4-CO high 200-250 | 26.7913 | 0.1191 |

The C4 route-rescue part specifically changes one test window:

| Metric | Value |
|---|---:|
| hit_N | 1 |
| hit_true_C4_high_CO_N | 1 |
| hit_false_N | 0 |
| hit_nonCO_N | 0 |
| C4_high_CO_total_N | 102 |
| C4_high_CO_recall | 0.0098 |

C4 high CO RMSE improves from 32.2169 to 26.7913 for the H8-to-H8+C4 delta, while C4 nonCO and all nonCO remain unchanged.

## Important Limitation

H8+C4 route-rescue is a high-precision surgical rescue, not a broad high-CO recovery model. In the current C12 -> C345 test set, it triggers on only one C4 high-CO window. Its value is that it fixes one extreme, high-risk wrong-route case without nonCO false hits, not that it covers most high-CO windows.

## Runtime/Size/Latency Benchmark

`benchmark_runtime_profiles.py` was added to compare exported deployment bundles. The first run used 300 test windows per client, repeated 3 times on CPU:

| Profile | Status | Artifact MB | Model MB | Mean ms/window | P90 ms/window | Expected full ALL RMSE |
|---|---|---:|---:|---:|---:|---:|
| H2.3 | ok | 24.0419 | 21.2380 | 2.7344 | 3.2869 | 18.62 |
| H8 | ok | 24.6380 | 21.2380 | 2.6319 | 2.8498 | 18.47 |
| H8+C4 | ok | 24.6414 | 21.2380 | 2.5857 | 2.7215 | 18.30 |
| L1 | missing_bundle | 0.0000 | 0.0000 |  |  | 22.6 |
| B0 | missing_bundle | 0.0000 | 0.0000 |  |  | 27.34 |

Because L1 has no exported runtime bundle in this benchmark, it should remain a pending deployment-lite candidate. It should not be described as deployment-lite until it shows a real size or latency advantage.

## Profile Selector Result

`select_target_profile.py` now emits `results/target_profile_selector_20260626/selected_profiles.json`.

Current selections:

| Mode | Selected Profile | Reason |
|---|---|---|
| balanced | H2.3 | balanced no-QC full-set mainline |
| co_priority | H8_plus_formal_C4_route_rescue | guardrail, feature schema, and runtime parity all pass |
| deployment_lite | H2.3 fallback | L1 runtime bundle is missing |

The selector record explicitly sets `test_used_for_selection=false`. Test metrics remain report-only, not selection evidence.

## Deployment Output Schema

`FinalDeployRuntime` now returns:

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

`auto_output_ppm` is set to `co_corrected_ppm` only when `qc_decision == "accept"`; otherwise it is blank. This keeps `final_ppm` and `co_corrected_ppm` available for audit while preventing review/reject windows from being silently treated as automatic system outputs.
