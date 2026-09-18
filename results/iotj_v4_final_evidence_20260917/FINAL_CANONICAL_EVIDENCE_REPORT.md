# IoT-J v4 Final Canonical Evidence Report

Generated: 2026-09-18T07:01:23.061539+00:00

## Classification

The main table is copied from the existing canonical fixed-endpoint evidence; no classifier was trained or adapted in this closure. Pooled Accuracy and Macro-F1 use the frozen definition: **merge C3/C4/C5 predictions first, then compute one pooled metric**. Methods from different target-information regimes are shown contextually and are not an equal-supervision total ranking. M0 is excluded from the main table.

| Regime | Method | Target information | C3 Acc. (%) | C4 Acc. (%) | C5 Acc. (%) | Pooled Acc. (%) | C3 Macro-F1 (%) | C4 Macro-F1 (%) | C5 Macro-F1 (%) | Pooled Macro-F1 (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Source-only FL | FedAvg | None (source-only) | 93.986 | 92.353 | 47.868 | 81.953 | 93.924 | 92.115 | 36.861 | 81.873 |
| Source-only FL | FedProx | None (source-only) | 95.928 | 91.838 | 50.294 | 83.398 | 95.906 | 91.734 | 39.308 | 83.372 |
| Source-only FL | SCAFFOLD | None (source-only) | 73.366 | 73.529 | 62.132 | 70.576 | 64.936 | 63.521 | 71.561 | 61.821 |
| Unlabeled DA | MMD | Target x only | 98.282 | 97.500 | 56.324 | 87.512 | 98.279 | 97.474 | 56.411 | 87.589 |
| Label-aware | A0T | Target x + calibration class labels | 99.851 | 99.485 | 99.412 | 99.648 | 99.851 | 99.486 | 99.414 | 99.648 |
| Label-aware | GAPS/A4 | Target x + calibration class and phase labels | 99.851 | 99.779 | 99.412 | 99.722 | 99.851 | 99.779 | 99.413 | 99.722 |

CORAL and DANN remain `EXCLUDED_NONCANONICAL`: their observed C3/C4 endpoint populations are 680/320 rather than 2677/1360. They were not retrained. Figure A directly reuses the existing 25-round Accuracy curves; those retrospective curves use the earlier FCL-E3 endpoint populations and are diagnostic only, not checkpoint-selection evidence for the canonical main table.

## Regression

Exactly six new deterministic Ridge fits were run: R83 and R84 at 15%, 10%, and 5%. The 20% point was **reported/reused**, not refit, from the later sealed R1 (2026-08-12) using the same R0-v2 source prior, feature caches, route, and test population as the reduced-budget runs. The earlier 2026-08-08 Table-10 numbers are not mixed into this curve because they belong to a different frozen provenance chain.

`delta_NRMSE = R84 - R83`; negative values indicate lower error with the source prior.

| Budget (%) | Calibration N | R83 NRMSE | R84 NRMSE | Delta NRMSE | Relative improvement |
|---|---|---|---|---|---|
| 20 | 320 | 0.099898 | 0.101923 | +0.002026 | -2.03% |
| 15 | 240 | 0.129120 | 0.128114 | -0.001006 | +0.78% |
| 10 | 160 | 0.117883 | 0.116044 | -0.001839 | +1.56% |
| 5 | 80 | 0.141353 | 0.140169 | -0.001184 | +0.84% |

### Paired original-file bootstrap

The C5 test metadata reliably recover 80 original-file/exposure groups (20 per gas). The interval uses 1,000 paired group-bootstrap replicates with seed 42 and identical resampled groups for R83/R84. Intervals are conditional on each single frozen calibration subset.

| Budget (%) | Point delta NRMSE | Paired 95% CI |
|---|---|---|
| 20 | +0.002026 | [-0.006320, +0.012806] |
| 15 | -0.001006 | [-0.003815, +0.001584] |
| 10 | -0.001839 | [-0.006896, +0.002288] |
| 5 | -0.001184 | [-0.001810, -0.000650] |

### Per-gas source-prior effect

| Budget (%) | Gas | R83 NRMSE | R84 NRMSE | Delta NRMSE |
|---|---|---|---|---|
| 20 | Ethanol | 0.061401 | 0.094382 | +0.032981 |
| 20 | CO | 0.110865 | 0.112354 | +0.001489 |
| 20 | Ethylene | 0.039578 | 0.041735 | +0.002157 |
| 20 | Methane | 0.149301 | 0.135205 | -0.014096 |
| 15 | Ethanol | 0.095461 | 0.095609 | +0.000148 |
| 15 | CO | 0.129003 | 0.129098 | +0.000095 |
| 15 | Ethylene | 0.039537 | 0.052490 | +0.012953 |
| 15 | Methane | 0.198419 | 0.192588 | -0.005831 |
| 10 | Ethanol | 0.115228 | 0.115228 | +0.000000 |
| 10 | CO | 0.154928 | 0.154507 | -0.000421 |
| 10 | Ethylene | 0.049968 | 0.058076 | +0.008108 |
| 10 | Methane | 0.125733 | 0.115507 | -0.010226 |
| 5 | Ethanol | 0.128281 | 0.128281 | -0.000000 |
| 5 | CO | 0.146568 | 0.146283 | -0.000285 |
| 5 | Ethylene | 0.065863 | 0.065863 | -0.000000 |
| 5 | Methane | 0.194026 | 0.190779 | -0.003247 |

Effects are descriptive and may be negative, zero, positive, or non-monotonic; no subset, alpha rule, prior, or model was changed after observing test results.

## QC

Figure C uses only `CLASSIFICATION_CONFIDENCE_ONLY` rows from the authoritative Q1 risk-coverage CSV at nominal 70%, 75%, 80%, 85%, 90%, 95%, and 100%, plotting actual retained coverage against accepted-set `NRMSE_range`. HC90/HC95 were overlaid because the current-authoritative rows match the same R84 backend, canonical C3/C4/C5 population, confidence-only risk, and full-population NRMSE. The random-retention band was intentionally omitted because it is optional and would add nonessential engineering; no legacy random baseline was mixed into the canonical curve.

The risk-ranked curve is retrospective. Only calibration-locked HC90/HC95 are deployment operating points; individual curve points are not separately commissioned thresholds.

## Robustness limitation

The existing **strict file/exposure-disjoint** sensitivity is preserved as a limitation. It uses 40 calibration and 40 test raw files with zero raw-file/time-support overlap, reduces the test population from 1360 to 840, and yields C5 Accuracy 74.762%, Macro-F1 69.361%, R83 NRMSE 0.3556, and R84 NRMSE 0.3489. Performance collapses materially relative to the window-level protocol. Therefore the evidence does **not** support a claim of robust unseen-exposure generalization, even though the strict-split R84-minus-R83 RMSE effect remains -1.506 ppm.

## Evidence boundary and stop condition

- No FL, DA, backbone, classifier, QC model, threshold, or additional seed experiment was added.
- Existing canonical source assets were treated as read-only.
- Six reduced-budget Ridge fits are the only new model fits.
- All requested tables and Figures A/B/C are present below this result root.
- Experimental work is closed with status `IOTJ_V4_EXPERIMENTS_FROZEN`.
