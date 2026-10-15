# Experiment Audit

## Verdict: PASS for a single-seed supplementary mechanism diagnostic

- All six branches independently reloaded the exact P0A round25 checkpoint.
- C1/C2 source calibration, C5 calibration identities, 100 steps, Adam 5e-4,
  alignment weight 0.5, seed42, and fixed step100 were held constant.
- C5 class labels were intentionally available during adaptation. TCE branches
  used them in target CE; conditional branches used them only for grouping.
- Target phase and concentration were not loaded by the target adaptation API.
- The sealed C5 test was opened only after all six fixed endpoints existed and
  was not used for tuning, stopping, thresholding, or checkpoint selection.
- Existing x-only E2, Simple Target-CE, and GAPS rows were copied read-only.
- Limitation: seed42 only; results are descriptive and do not establish
  statistical significance or replace the canonical unsupervised TABLE II rows.
