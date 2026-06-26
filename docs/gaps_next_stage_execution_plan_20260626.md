# GAPS Next-Stage Execution Plan — 2026-06-26

This is the execution plan for turning the current GAPS codebase from a collection of strong experiments into a stable, reproducible, deployable system.

## 0. Current decision state

```text
R3aK16 = source-domain neural regression base and fallback
H2.3 = balanced default performance mainline
H8 + formal C4 route rescue = CO/high-CO deployable specialist candidate
L1 = deployment-lite candidate, pending runtime/size/latency proof
QC = post-hoc reliability layer, not model capability selector
```

## 1. P0 tasks: must finish before new model search

### P0.1 Re-export H8+C4 v2 bundle after `max_conf_margin` fix

Reason: the exporter now preserves `max_conf_margin` and patches the exported runtime guard. Existing old bundles do not automatically inherit this fix.

Command:

```bash
python export_h8_formal_c4_rescue_deployment_candidate.py
```

Expected console payload:

```json
{
  "runtime_guard_patched": true,
  "additional_gates": [
    {
      "max_conf_margin": 1.0
    }
  ]
}
```

Acceptance:

- `results/deployment_h8_formal_c4_rescue_candidate_20260625/rich_residual_candidate.json` contains `c4_route_rescue_policy.v2`.
- `additional_gates[0].max_conf_margin` exists.
- exported `runtime_src/gaps_deploy/rich_residual.py` contains a confidence-margin check in `_single_route_rescue_ppm`.

### P0.2 Validate H8+C4 runtime after re-export

Command:

```bash
python validate_rich_residual_runtime_candidate.py \
  --bundle results/deployment_h8_formal_c4_rescue_candidate_20260625 \
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid \
  --clients C3,C4,C5 \
  --output-dir results/runtime_validation_h8_formal_c4_rescue_candidate_20260626
```

Acceptance:

- runtime prediction row count equals expected C3/C4/C5 test rows.
- `runtime_summary.csv` contains both `final_ppm` and `co_corrected_ppm` summaries.
- H8+C4 metrics remain close to previous values:
  - ALL RMSE around 18.30
  - C4 CO RMSE around 17.16
  - C4 high CO RMSE around 26.79
  - nonCO unchanged or not meaningfully worse

### P0.3 Run runtime-vs-analysis equivalence

Use the existing equivalence comparison script or create a small comparator if needed.

Acceptance:

- rows compared = 5400 for C3/C4/C5 test in the current split
- mismatch rows = 0
- max_abs_diff < 1e-8

### P0.4 README cleanup

Replace missing or local-only mainline commands with `docs/mainline_entrypoints_20260626.md`.

Acceptance:

- README does not present missing files as current GitHub mainline commands.
- README links to:
  - `docs/mainline_entrypoints_20260626.md`
  - `docs/gaps_next_stage_execution_plan_20260626.md`
  - `docs/experiment_design_and_acceptance_20260626.md`
  - `docs/feature_schema_and_runtime_contract_20260626.md`

## 2. P1 tasks: formalization and guardrails

### P1.1 H8+C4 guardrail audit

Create or extend a script:

```text
audit_h8_c4_guardrails.py
```

Required statistics:

- total gate hit_N
- true C4 high-CO hit_N
- false hit_N
- nonCO hit_N
- C4 high-CO recall
- C4 high-CO RMSE before/after
- C4 nonCO RMSE before/after
- nonCO ALL RMSE before/after
- hit sample filename/repeat/response_phase distribution

Acceptance:

- false hit_N should be 0 or explicitly justified.
- nonCO hit_N should be 0 for formal C4 rescue.
- If false hits appear after the v2 export fix, do not promote H8+C4 as default; keep H2.3 default.

### P1.2 Feature/runtime schema validation

Create:

```text
validate_feature_schema.py
```

Required checks:

- feature arrays are `(N,100,8)`.
- label arrays match feature length.
- metadata length matches feature length when metadata is required.
- `norm_stats` exists and has finite `mean` and `std`.
- `runtime_config.normalization.enabled` is consistent with the documented feature space.
- rich residual receives raw pre-model window, while model predictor receives normalized window when normalization is enabled.

Acceptance:

- a `feature_schema_validation.json` is emitted for every deployment bundle.
- failed schema checks should block runtime validation.

### P1.3 Profile selector consolidation

Target script:

```text
select_target_profile.py
```

Modes:

```text
balanced        -> default H2.3 unless invalid
co_priority     -> H8/H8+C4 allowed if guardrails pass
deployment_lite -> L1 allowed if runtime/size/latency advantage is proven
```

Acceptance:

- selector input uses calibration or calibration-validation evidence only.
- test metrics are reported only after profile selection.
- selected profile JSON contains:
  - `profile_mode`
  - `selected_profile`
  - `fallback_profile`
  - `selection_data`
  - `test_used_for_selection: false`
  - guardrail pass/fail details

## 3. P2 tasks: robustness and deployment evidence

### P2.1 Bidirectional validation

Current directions:

- C12 -> C345
- C45 -> C123

Acceptance:

- report both direction-level selections.
- specialist/gate should be selected only when calibration-derived guardrails pass.
- failure to select H8+C4 in reverse direction is acceptable if fallback is correct.

### P2.2 Low-calibration stress test

Ratios:

```text
20%, 10%, 5%, 2.5%
```

Candidates:

- B0/R3aK16 baseline
- H2.3
- H8/H8+C4 selector
- L1 lightweight

Acceptance:

- selector remains stable or falls back safely under low calibration.
- no test-only selection.
- report ALL, CO, high-CO, nonCO, NRMSE, and guardrail false hits.

### P2.3 Runtime/size/latency benchmark

Create:

```text
benchmark_runtime_profiles.py
```

Compare:

- B0/R3aK16 bundle
- H2.3 bundle
- H8+C4 bundle
- L1 bundle

Metrics:

- artifact size
- model parameter count
- mean latency/window
- p90 latency/window
- memory footprint if available
- ALL RMSE / nonCO / CO high trade-off

Acceptance for L1 deployment-lite role:

- L1 must show a clear runtime or size benefit, preferably 30%-50% improvement.
- If L1 is not meaningfully lighter, keep it as diagnostic rather than deployment-lite.

## 4. P3 tasks: engineering cleanup

### P3.1 Add `auto_output_ppm`

Modify `gaps_deploy/final_runtime.py` public output:

```text
auto_output_ppm = co_corrected_ppm if qc_decision == "accept" else ""
```

Reason: review/reject rows may retain prediction values for audit, but the system should not silently auto-output them.

### P3.2 Specialist calibration trainable parameter switch

Modify `gaps_flower/specialist_calibration_fit.py`:

```text
--train-response-adapter true/false
```

Default should be false to reduce small-sample overfit. If true, include DCT/response adapter parameters in specialist calibration.

### P3.3 Server/client profile consistency

Modify `gaps_flower/server_app.py`:

```text
--profile gaps_cls
```

Pass the same profile into `make_config()` that clients use.

## 5. Stop conditions

Pause new head/model search until all P0 tasks and at least P1.1/P1.2 are completed.

Do not promote a candidate as a default mainline if any of these are true:

- selector uses test metrics;
- runtime parity fails;
- feature schema validation fails;
- nonCO guardrail fails without clear deployment-mode justification;
- low-calib stress shows unstable false hits;
- QC accepted-only metrics are used as the main model capability metric.

## 6. Recommended near-term order

```text
1. Re-export H8+C4 v2 bundle
2. Validate runtime and equivalence
3. Run H8+C4 guardrail audit
4. Add feature schema validation
5. Clean README
6. Consolidate select_target_profile
7. Bidirectional report cleanup
8. Low-calib stress
9. Runtime/size/latency benchmark
10. auto_output_ppm and engineering refinements
```

## 7. Status update after local validation

Current completed items:

- H8+C4 v2 bundle re-exported with `runtime_guard_patched=true`.
- Runtime validation completed on C3/C4/C5 target test.
- Runtime-vs-analysis equivalence completed: 5400 rows, 0 mismatch, max_abs_diff 1.42e-13.
- H8+C4 guardrail audit completed: hit_N=1, hit_false_N=0, hit_nonCO_N=0, guardrail_status=pass.
- Feature/runtime schema validation completed: status=pass.
- `auto_output_ppm` added to `FinalDeployRuntime` and the H8+C4 v2 exported runtime schema.
- `benchmark_runtime_profiles.py` added and first CPU benchmark generated at `results/runtime_profile_benchmark_20260626/`.
- `select_target_profile.py` added and selector output generated at `results/target_profile_selector_20260626/`.

Current selector state:

```text
balanced        -> H2.3
co_priority     -> H8_plus_formal_C4_route_rescue
deployment_lite -> H2.3 fallback, because L1 has no exported runtime bundle yet
```

Important limitation:

H8+C4 formal route-rescue is a high-precision surgical rescue. It hits one C4 high-CO test window in the current split, with zero false/nonCO hits. It should be reported as a guarded specialist, not as broad high-CO coverage.
