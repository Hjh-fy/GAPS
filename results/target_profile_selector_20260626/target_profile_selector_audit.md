# Target Profile Selector Audit

This audit turns the recent regression matrix into an explicit, reproducible profile-selection step.
The selector uses no-QC full-set metrics only; QC accepted quality is intentionally not part of the decision.

## Selection Logic

- Balanced mainline: shortlist candidates within 2% of the best ALL RMSE, then choose the lowest ALL NRMSE, nonCO RMSE, and ALL RMSE.
- CO-specialist candidate: among candidates within 2% ALL RMSE of the balanced mainline, choose the lowest mean/max target CO-high RMSE if it improves over balanced.
- Test-oracle candidates are excluded from selection and only kept for sanity checks.

## Results

### C12_to_C345

- Baseline: `A0_baseline_final` ALL RMSE 27.34, NRMSE 0.1578, CO-high mean 65.67.
- Balanced mainline: `H2_3_R3aK16_current_mainline` ALL RMSE 18.62, NRMSE 0.1326, nonCO RMSE 17.83, CO-high mean 29.69.
- CO-specialist candidate: `H8_plus_formal_C4_rescue` ALL RMSE 18.30, NRMSE 0.1350, nonCO RMSE 18.38, CO-high mean 24.76.

### C45_to_C123

- Baseline: `A0_baseline_final` ALL RMSE 22.94, NRMSE 0.1473, CO-high mean 38.56.
- Balanced mainline: `H1_target_Ridge_direct` ALL RMSE 15.59, NRMSE 0.1123, nonCO RMSE 14.50, CO-high mean 23.92.
- CO-specialist candidate: none selected under the current constraints.

## Interpretation

- C12 -> C345 still supports a two-profile story: H2.3 is the balanced mainline, while a CO-specialist/rescue profile is useful when CO-high is prioritized.
- C45 -> C123 selects a simpler target Ridge direct mainline; source-aug switching remains diagnostic because the overall/nonCO tradeoff is not favorable.
- This supports the method description as direction-specific target profile selection rather than a single hard-coded regression head.
