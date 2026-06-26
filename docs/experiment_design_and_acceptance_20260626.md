# GAPS Experiment Design and Acceptance Criteria — 2026-06-26

This document defines the next experiments to run after the current H2.3 / H8+C4 / L1 matrix. The focus is not new model hunting; it is formal selection, runtime validation, robustness, and deployment evidence.

## 0. Reporting principles

1. Report **no-QC full-set target test** as the model capability metric.
2. Report QC metrics separately as deployment reliability.
3. Use calibration or calibration-validation only for selector/gate/profile selection.
4. Never use test metrics to select a candidate.
5. Every deployable candidate must pass runtime artifact validation and runtime-vs-analysis parity.

## 1. Experiment E1 — H8+C4 v2 deployment validation

### Purpose

Validate the fixed H8 + formal C4 route-rescue deployment candidate after preserving `max_conf_margin` in the export/runtime gate.

### Inputs

- Base H8 bundle:
  - `results/deployment_h8_source_aug_candidate_20260625`
- Formal C4 gate:
  - `results/formal_c4_route_rescue_selector_20260625/formal_c4_route_rescue_selected_gate.json`
- Target data root:
  - `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`

### Commands

```bash
python export_h8_formal_c4_rescue_deployment_candidate.py

python validate_rich_residual_runtime_candidate.py \
  --bundle results/deployment_h8_formal_c4_rescue_candidate_20260625 \
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid \
  --clients C3,C4,C5 \
  --output-dir results/runtime_validation_h8_formal_c4_rescue_candidate_20260626
```

### Required outputs

- `results/deployment_h8_formal_c4_rescue_candidate_20260625/rich_residual_candidate.json`
- `results/runtime_validation_h8_formal_c4_rescue_candidate_20260626/runtime_predictions.csv`
- `results/runtime_validation_h8_formal_c4_rescue_candidate_20260626/runtime_summary.csv`
- equivalence summary if comparison script is run

### Metrics

| Scope | Expected approximate target |
|---|---:|
| ALL RMSE | ~18.30 |
| ALL NRMSE | ~0.1350 |
| C4 CO RMSE | ~17.16 |
| C4 high-CO RMSE | ~26.79 |
| C5 high-CO RMSE | ~27.54 |
| nonCO ALL RMSE | ~18.38 |

### Acceptance

- deployment artifact schema is `c4_route_rescue_policy.v2`.
- exported additional gate contains `max_conf_margin`.
- runtime guard enforces `max_conf_margin`.
- runtime-vs-analysis mismatch = 0.
- if metrics shift, explain whether it is due to the stricter margin guard.

## 2. Experiment E2 — H8+C4 guardrail audit

### Purpose

Prove that formal C4 rescue is not a test-only hand-tuned rule and does not harm nonCO.

### Proposed script

```text
audit_h8_c4_guardrails.py
```

### Inputs

- runtime predictions from E1
- selected gate JSON
- metadata JSON with filename/repeat/response_phase

### Required audit fields

| Field | Meaning |
|---|---|
| `hit_N` | number of route-rescue hits |
| `hit_true_C4_high_CO_N` | hits that are true C4 CO >= 200 ppm |
| `hit_false_N` | hits outside true C4 high CO |
| `hit_nonCO_N` | nonCO hit count |
| `C4_high_recall` | true C4 high-CO hit recall |
| `C4_high_RMSE_before/after` | high-CO benefit |
| `C4_nonCO_RMSE_before/after` | nonCO guard |
| `nonCO_ALL_RMSE_before/after` | global nonCO guard |
| `hit_by_file/repeat/phase` | diagnose file-specific overfitting |

### Acceptance

- `hit_nonCO_N = 0` is ideal.
- `hit_false_N = 0` is ideal.
- If false hits occur, keep H8+C4 as optional CO-priority candidate only.
- H2.3 remains default balanced mainline regardless of H8+C4 ALL RMSE unless nonCO guard passes robustly.

## 3. Experiment E3 — Formal profile selector

### Purpose

Stop manual candidate picking. Produce a profile JSON selected from calibration-derived evidence.

### Proposed script or current target

```text
select_target_profile.py
```

### Candidate families

| Candidate | Role |
|---|---|
| B0/R3aK16 baseline | fallback/neural baseline |
| H2.3 | balanced default |
| H8 | CO-specialist |
| H8+C4 | CO/high-CO specialist |
| L1 | deployment-lite candidate |

### Modes

```text
balanced
co_priority
deployment_lite
```

### Selection rules

Balanced:

```text
H2.3 wins unless invalid or missing.
```

CO-priority:

```text
H8/H8+C4 may win only if:
- calibration CO improves versus H2.3;
- nonCO degradation <= threshold;
- false-hit guard passes;
- runtime parity passes.
```

Deployment-lite:

```text
L1 may win only if:
- runtime/size/latency advantage is meaningful;
- ALL RMSE degradation is within allowed budget;
- nonCO does not collapse;
- calibration selector does not overfit tiny cells.
```

### Required output JSON

```json
{
  "direction": "C12_to_C345",
  "profile_mode": "balanced",
  "selected_profile": "H2.3",
  "fallback_profile": "R3aK16_auto_v2",
  "selection_data": "calibration_val_only",
  "test_used_for_selection": false,
  "guardrails": {
    "nonco_guard": "pass",
    "runtime_parity": "pass",
    "false_hit_guard": "pass"
  }
}
```

### Acceptance

- test columns are not used in profile selection.
- output includes fallback and guardrail reasons.
- selected profile can be loaded by deployment export scripts.

## 4. Experiment E4 — Bidirectional validation

### Purpose

Demonstrate that the selector is not direction-specific or C4-specific.

### Directions

| Direction | Source | Target |
|---|---|---|
| forward | C1/C2 | C3/C4/C5 |
| reverse | C4/C5 | C1/C2/C3 |

### Commands/current scripts

```bash
python run_c45_c123_optimal_config_analysis.py
python summarize_bidirectional_profile_selection.py
```

### Required report

| direction | balanced selected | co-priority selected | ALL | CO | high-CO | nonCO | notes |
|---|---|---|---:|---:|---:|---:|---|

### Acceptance

- reverse result does not need to match forward result.
- selector must fall back safely if H8/C4-like specialist is not valid.
- no hard-coded C4 gate should be applied to reverse targets unless selected by reverse calibration.

## 5. Experiment E5 — Low-calibration stress test

### Purpose

Support the thesis claim of few-shot target calibration.

### Calibration ratios

```text
20%, 10%, 5%, 2.5%
```

### Candidates

- B0/R3aK16 baseline
- H2.3
- H8/H8+C4 selector
- L1 lightweight

### Required table

| ratio | candidate | ALL RMSE | NRMSE | CO RMSE | high-CO RMSE | nonCO RMSE | false hits | selector stable |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Acceptance

- H2.3 or fallback remains better than B0 at 10% if possible.
- At 5%/2.5%, safe fallback is more important than forcing H8+C4.
- false hits must be explicitly reported.
- unstable selectors should be downgraded to diagnostic.

## 6. Experiment E6 — Runtime/size/latency benchmark

### Purpose

Determine whether L1 deserves the deployment-lite label.

### Proposed script

```text
benchmark_runtime_profiles.py
```

### Candidates

- B0/R3aK16 bundle
- H2.3 bundle
- H8+C4 bundle
- L1 bundle

### Measurements

| Metric | Notes |
|---|---|
| artifact size | total bundle size on disk |
| parameter count | model parameters if PyTorch checkpoint present |
| mean latency/window | CPU inference |
| p90 latency/window | tail latency |
| correction latency | rich residual overhead |
| memory | optional |

### Acceptance for L1

L1 can be called deployment-lite only if it has a meaningful benefit:

```text
size or latency improvement >= 30% preferred
```

If not, write L1 as an analytical lightweight diagnostic rather than a formal deployment candidate.

## 7. Experiment E7 — QC post-hoc reliability report

### Purpose

Keep QC separate from model capability.

### Candidates

- H2.3
- H8+C4
- L1 if deployment-lite remains valid

### Metrics

| Candidate | accepted coverage | accepted RMSE | accepted+review coverage | accepted+review RMSE | reject RMSE | high-error recall |
|---|---:|---:|---:|---:|---:|---:|

### Acceptance

- no model should be selected based on accepted-only RMSE.
- QC report must include rejected/reviewed distribution by gas/client/high-CO/phase.
- `auto_output_ppm` should be added before final deployment UI/export.

## 8. Final deliverable checklist

A candidate is ready for thesis/system chapter only when it has:

- calibration-only selector record;
- target test full-set metrics;
- runtime artifact;
- runtime parity check;
- guardrail audit;
- feature schema validation;
- QC post-hoc table;
- bidirectional or reverse-direction evidence;
- low-calib stress evidence if it claims few-shot robustness.
