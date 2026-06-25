# Lightweight Regression Head Fair Auto-v2 Experiment Plan

Date: 2026-06-25

Goal: fairly test whether a lightweight source regression head can replace or approximate the current R3aK16 regression base after the same target-domain calibration flow.

## Core Question

Earlier lightweight source-head tests answered only a partial question:

- Source-only transfer collapses on C3/C4/C5.
- Source lightweight head + target affine calibration improves, but still underperforms R3aK16 baseline.

That is not yet a final rejection of lightweight heads, because the lightweight heads have not been evaluated with the same full target auto_v2 candidate selection used by the current mainline.

The fair question is:

Can a lightweight source head, after the same target-side auto_v2 calibration and selector, match or improve the no-QC full-set target metrics of R3aK16 + auto_v2/H2.3?

## Fairness Rules

All candidates must use the same:

- Source clients: C1/C2.
- Target clients: C3/C4/C5.
- Target calibration/test files.
- Fixed-DA classification route and backbone.
- No-QC full-set target-test evaluation.
- Calibration-fit / calibration-validation split for model and profile selection.
- Test split only for final reporting.

The target test split must not be used for:

- choosing head structure;
- choosing alpha/hidden size;
- choosing auto_v2 candidate;
- choosing C4 route-rescue gate;
- choosing per-client profile.

## Baselines

| ID | Candidate | Role | Current evidence |
|---|---|---|---|
| B0 | R3aK16 + original auto_v2 | stable reference | ALL RMSE 27.34, NRMSE 0.1578 |
| B1 | R3aK16 + H2.3 target direct-head auto_v2 | current balanced mainline | ALL RMSE 18.62, NRMSE 0.1326 |
| B2 | R3aK16 + H8 pred-CO source-aug | CO-specialist candidate | ALL RMSE 18.47, improves CO/high-CO but worsens nonCO/NRMSE |

## Lightweight Candidates

### L0: Existing Source Lightweight + Affine

This is the already-run diagnostic, retained only as a reference:

- source Ridge + target affine;
- source per-gas MLP + target affine;
- source shared MLP + target affine.

It is not the final fair comparison because it does not use full auto_v2.

### L1: Source Lightweight + Full Target Residual Auto-v2

Use lightweight source-head prediction as the base ppm, then apply target auto_v2 residual candidates:

- identity / keep source head output;
- affine;
- ridge_basic residual;
- ridge_phase residual;
- piecewise_ridge residual;
- optional C4 route-rescue only if selected by calibration-validation.

Expected value:

- directly tests whether source lightweight ppm can be rescued by the same rich residual calibration that helped R3aK16.

### L2: Source Lightweight + Target Direct-Head Auto-v2

Use lightweight source head as an additional candidate, but allow target direct heads to compete:

- baseline R3aK16 final_ppm;
- source lightweight final_ppm;
- target Ridge direct head;
- target shallow MLP direct head;
- per-client profile selection.

Expected value:

- tests whether lightweight source head provides useful signal beyond the target direct heads.
- If selector never chooses lightweight source head, then lightweight source pretraining is not useful for target final regression.

### L3: Lightweight Neural Base + Same Target Specialist Flow

Train genuinely smaller neural regression branches in the Flower/regression model, then run the same target-side specialist/calibration flow:

- depth2 MLP head without response branch;
- depth1 MLP head;
- shared-private compact CO head;
- response-branch removed or reduced.

Expected value:

- tests deployable lightweight replacement, not just post-hoc sklearn heads.

## First Experimental Matrix

Run this order to control cost and ambiguity:

| Step | ID | Method | Output needed | Promote if |
|---|---|---|---|---|
| 1 | L1-Ridge | source Ridge base + full residual auto_v2 | summary CSV/report | beats source Ridge + affine and approaches B0 |
| 2 | L1-PerGasMLP | source per-gas MLP base + full residual auto_v2 | summary CSV/report | beats source MLP + affine and approaches B0 |
| 3 | L1-SharedMLP | source shared MLP base + full residual auto_v2 | summary CSV/report | beats source shared MLP + affine and approaches B0 |
| 4 | L2-Selector | source bases + target Ridge/MLP direct heads in one selector | selected profile JSON + test report | selector chooses source-lightweight for any stable client/gas |
| 5 | L3-NeuralLite | compact trainable regression branch + same target calibration | model metrics + params | close to B0 with clear parameter reduction |

Do not move to L3 until L1/L2 shows whether lightweight outputs contain useful target signal after full calibration.

## Metrics

Primary:

- ALL RMSE.
- ALL NRMSE.
- C3/C4/C5 CO RMSE.
- C3/C4/C5 CO high RMSE.

Secondary:

- nonCO ALL RMSE.
- per-client ALL RMSE.
- CO bias.
- CO high P90AE.
- parameter count and deployment artifact size.
- runtime latency if a candidate approaches baseline.

## Promotion Criteria

Performance-mainline promotion:

- ALL RMSE <= H2.3 or H8, and
- ALL NRMSE does not worsen, and
- nonCO does not obviously worsen, and
- CO/high-CO improves or stays comparable.

Deployment-lite candidate:

- ALL RMSE is close to B0 or better, and
- parameter count/runtime is materially lower than R3aK16, even if it does not beat H2.3.

Reject or pause:

- ALL RMSE remains > B0 after full target auto_v2.
- nonCO worsens severely.
- calibration-validation improvement does not reproduce on test.

## Implementation Plan

### Script 1: `run_source_lightweight_full_auto_v2_eval.py`

Purpose:

- start from source-trained lightweight predictions;
- split target calibration into fit/validation;
- fit residual auto_v2 candidates on calibration-fit;
- select per client/gas on calibration-validation;
- refit on full calibration;
- evaluate target test.

Inputs:

- `results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv`
- `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`

Outputs:

- `results/source_lightweight_full_auto_v2_20260625/summary.csv`
- `results/source_lightweight_full_auto_v2_20260625/selection_profile.json`
- `results/source_lightweight_full_auto_v2_20260625/source_lightweight_full_auto_v2_report.md`
- per-sample prediction CSVs for each candidate.

### Script 2: `summarize_lightweight_fair_matrix.py`

Purpose:

- combine old affine results, L1 results, H2.3, H8, and B0 into one table.

Outputs:

- `results/lightweight_fair_matrix_20260625/lightweight_fair_matrix_summary.csv`
- `results/lightweight_fair_matrix_20260625/lightweight_fair_matrix_report.md`

## Interpretation Guide

Possible outcomes:

1. Lightweight + full auto_v2 still fails.
   - Conclusion: the issue is not just missing calibration; lightweight source outputs lack stable target-domain signal.

2. Lightweight + full auto_v2 reaches B0 but not H2.3.
   - Conclusion: lightweight is viable for deployment-lite, but not performance mainline.

3. Lightweight + full auto_v2 reaches H2.3/H8.
   - Conclusion: R3aK16 can be replaced or compressed, and runtime schema should support lightweight source-head artifacts.

4. Selector uses lightweight only for specific gas/client.
   - Conclusion: keep it as a per-client/per-gas optional candidate, not a global replacement.

## Current Working Hypothesis

The most likely result is that lightweight source heads will improve over target affine after full residual auto_v2, but still struggle to beat H2.3. If so, the best practical direction is:

- keep H2.3/H8 as performance mainline;
- retain lightweight heads as deployment-lite diagnostics;
- only promote a lightweight branch if it approaches B0 with a clear parameter/runtime advantage.
