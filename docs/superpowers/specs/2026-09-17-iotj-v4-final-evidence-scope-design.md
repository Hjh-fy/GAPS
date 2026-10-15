# IoT-J v4 final evidence scope design

Date: 2026-09-17
Status: design frozen for user review; no implementation or run authorized by this file alone

## 1. Objective

Close the remaining IoT-J v4 evidence chain without expanding the algorithm inventory. The final package has three evidence blocks:

1. a canonical classification comparison organized by federated-learning and domain-adaptation information regime;
2. a C5 canonical calibration-budget study that quantifies the incremental effect of the frozen one-dimensional FedRidge source prior in R84 relative to target-only R83;
3. a canonical confidence-QC coverage--NRMSE figure derived from already frozen predictions and risk scores.

After these blocks are audited, one final evidence report is produced and the experimental stage is marked `IOTJ_V4_EXPERIMENTS_FROZEN`.

## 2. Scope decisions

### 2.1 Classification main comparison

The main classification table and figure exclude M0 accuracy. M0 remains a supplementary, legacy-protocol matched semantic-mechanism ablation and is not used as a canonical method-ranking row.

The canonical main comparison is grouped rather than treated as one label-fair ranking:

| Information regime | Methods | Target information |
|---|---|---|
| Source-only FL | FedAvg, FedProx, SCAFFOLD | none |
| Unlabeled DA | CORAL, MMD, DANN | target `x` only |
| Label-aware reference/proposed | A0T, GAPS/A4 | explicit calibration-label access as recorded by each protocol |

Primary outputs are Accuracy and Macro-F1 for C3, C4, C5, and pooled. NLL, ECE, source retention, optimizer details, and checkpoint hashes remain in supplementary provenance. Rows from different information regimes may be displayed together only with the regime column and must not be described as an equal-information total ranking.

No new FL or DA training is planned. Existing canonical fixed endpoints are consolidated and audited into one schema. The six legacy labeled C5 DA probes remain diagnostic and are excluded.

### 2.2 R83/R84 calibration-budget experiment

This is a backend-only C5 regression study. It does not change or retrain the frozen A4 classifier and does not establish end-to-end classification commissioning sample efficiency.

The frozen nested calibration budgets are:

| Budget | Calibration rows |
|---:|---:|
| 20% | 320 |
| 15% | 240 |
| 10% | 160 |
| 5% | 80 |

The 20% R83/R84 pair is reused. Only six new lightweight Ridge fits are planned: R83 and R84 at 15%, 10%, and 5%.

R83 uses exactly the canonical 83-dimensional target statistics. R84 uses those same 83 dimensions plus exactly one prediction column from the immutable R0-v2 FedRidge source model. The source model, source alpha/model locks, feature order, source client order, and prediction semantics are immutable across budgets.

For each target method/gas/budget, target Ridge alpha is selected by the same registered deterministic five-fold raw-filename-grouped calibration-only CV over the fixed grid `(0, 0.01, 0.1, 1, 10, 100, 1000)`, with pooled RMSE and first-in-grid tie break. The selection rule is fixed; the selected numeric alpha is not forced to be identical across budgets or methods and must never be manually tuned after observing test results.

R83 and R84 at a given budget must share the same calibration identities, test identities, routing decisions, folds, metric implementation, and release gate. Target test access occurs only after both model locks for that budget are complete.

### 2.3 Source-prior effect quantities

The primary paired effect is:

`delta_NRMSE = NRMSE_R84 - NRMSE_R83`.

Negative values indicate lower normalized error for R84. Also report:

- `delta_RMSE_ppm = RMSE_R84 - RMSE_R83`;
- `relative_NRMSE_improvement_pct = (NRMSE_R83 - NRMSE_R84) / NRMSE_R83 * 100`;
- RMSE, MAE, R2, and NRMSE_range for each method;
- per-gas paired effects;
- a paired raw-file-grouped bootstrap 95% CI for the test-set difference, conditional on the one frozen calibration subset.

No monotonic trend is assumed. A negative, zero, positive, or non-monotonic effect is reported without changing subsets, alphas, priors, or models.

The figure has two panels: R83/R84 NRMSE versus budget, and `R84 - R83` NRMSE versus budget with a zero reference line.

### 2.4 QC coverage--NRMSE evidence

The primary QC policy is the authoritative `CLASSIFICATION_CONFIDENCE_ONLY` / `CONFIDENCE_QC_FINAL` policy. The plot uses the existing canonical Q1 risk-coverage source and selects target coverage values 70%, 75%, 80%, 85%, 90%, 95%, and 100%.

The x-axis is actual retained coverage and the y-axis is accepted-set `NRMSE_range`. The pooled C3+C4+C5 curve is primary; C3, C4, and C5 curves are secondary panels or supplementary material. The 100% point is the no-QC endpoint. Calibration-locked HC90 and HC95 points are overlaid and kept semantically distinct from the retrospective ranking curve.

A canonical equal-retention random reference mean and interval may be added only as deterministic post-processing of the same authoritative prediction rows after identity audit. Legacy random-retention values may not be overlaid on canonical curves. No QC model, score, threshold, or policy may be retrained.

The curve is a retrospective risk-ranking diagnostic. Only calibration-locked HC90/HC95 are deployment operating points. Individual curve coverage values must not be described as separately commissioned thresholds.

## 3. Protocol amendment and evidence boundaries

The earlier `CALIBRATION_BUDGET_GAP.md` proposed a future 100%/50%/25% backend study. The later frozen manifest contains nested 20%/15%/10%/5% subsets, and the user explicitly approved that grid on 2026-09-17. This design is the scoped amendment for Experiment B and supersedes the earlier proposed grid for this study. The earlier document remains read-only and is cited as historical planning context.

This amendment does not weaken its scientific warning: a backend-only R83/R84 budget study does not establish end-to-end low-label commissioning performance.

Canonical and legacy evidence remain separate. M0 and the six labeled C5 probes cannot be inserted into the canonical FL/DA ranking. Historical QC random baselines cannot be combined numerically with canonical QC curves.

## 4. Data flow and outputs

1. Audit the immutable classification endpoints and build a grouped canonical method table.
2. Audit the nested C5 budget manifest, calibration/test non-overlap, R0-v2 source locks, feature identity, fold construction, and 20% reference pair.
3. Register the six reduced-budget Ridge configurations before execution.
4. Fit R83 and R84 in paired order per budget without opening target test.
5. Lock both models, release the test once per budget, generate predictions and paired statistics, and never select a budget/model from test performance.
6. Extract the seven QC coverage points from the authoritative Q1 curve; optionally compute a canonical equal-retention random band from the same frozen prediction rows.
7. Produce the final result table, figures, provenance index, claim boundaries, and final evidence report.

All new artifacts use new paths beneath `results/iotj_v4_final_experiments_20260914/`; existing results are read-only and never overwritten.

## 5. Fail-closed conditions

Stop the affected work before execution if any of the following occurs:

- the 5%/10%/15%/20% identities are not nested or include target-test rows;
- R83 and R84 at the same budget do not share calibration/test identities;
- the R0-v2 source model or its predictions differ across budgets;
- the 83D feature order or 84th source-prior column differs from canonical R1;
- alpha grid, grouped-CV rule, fold assignment rule, routing, normalization, or metric code differs from the full-budget reference without an approved amendment;
- target test is opened before paired model locks;
- classification endpoints cannot be reconciled to canonical-v1 hashes;
- the QC figure requires threshold refitting, test-label optimization, or mixing legacy and canonical rows.

## 6. Tests and audits

Required pre-run tests cover manifest nesting, exact row identities, no test overlap, feature hashes, source-lock hashes, paired folds, alpha-rule equality, output non-overwrite, and test-release gating.

Required post-run checks cover six-run completeness, finite predictions, paired row order, metric recomputation, bootstrap determinism, no target-test selection, figure-source equality, and machine-readable provenance.

Classification and QC consolidation are post-processing tasks and must reproduce their authoritative source values exactly before a new table or figure is accepted.

## 7. Stop condition

No additional algorithm, backbone, seed matrix, target client, QC model, or deployment rerun is in scope. After classification consolidation, the six reduced-budget fits, QC figure, audit, and final evidence report are complete, set `IOTJ_V4_EXPERIMENTS_FROZEN` and stop unless a later reviewer request explicitly reopens scope.
