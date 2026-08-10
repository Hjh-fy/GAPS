# Canonical-v1 A0T vs GAPS/A4 Regression Commissioning Design

## Objective

Determine whether the primary value of GAPS/A4 appears in quantitative regression commissioning rather than classification accuracy. The study is a fixed-endpoint, seed42, canonical-v1 comparison and does not authorize classifier retraining, algorithm search, preprocessing changes, split changes, R84 changes, QC changes, or hyperparameter tuning.

## Research questions and decision rule

The study answers three mechanism questions:

1. Does A4 reduce classification-induced regression error?
2. Does A4 improve the regression mapping after routing is correct?
3. Is any observed benefit attributable mainly to routing/classification or to quantitative regression commissioning?

The final decision is descriptive fixed-endpoint evidence:

- `REGRESSION_ADVANTAGE_SUPPORTED` requires both:
  - A4 C5 `S_ALL` RMSE is lower than A0T C5 `S_ALL` RMSE; and
  - A4 target-size-weighted pooled C3+C4+C5 `S_ALL` RMSE is lower than the corresponding A0T value.
- Otherwise the decision is `REGRESSION_ADVANTAGE_NOT_SUPPORTED`.

Mechanism attribution is based on the routing and Oracle decompositions, not on the direction of the final decision alone. Seed42 does not support a stability claim or uncertainty estimate.

## Frozen inputs

### Dataset and split

- Dataset: `dataset/iotj_canonical_v1`.
- Dataset aggregate SHA-256: `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Sources: C1/C2, used only through the frozen Federated-H1 source prior.
- Targets: C3/C4/C5.
- Calibration/test counts: C3 678/2677, C4 320/1360, and C5 320/1360, using each target's canonical approximately 20%/80% role-aware split.
- Test: each target's unchanged canonical sealed test split with the exact counts above.
- Test data cannot be used for checkpoint selection, Ridge fitting, alpha selection, QC normalization, QC threshold selection, stopping, or any other configuration decision.

### Classifier checkpoints

The six existing round25 target-adapted checkpoints are immutable inputs. No classifier training or adaptation is run in this study.

| Method | Target | Experiment | Whole-file SHA-256 |
|---|---|---|---|
| A0T | C3 | `CANONICAL-V1-A0T-C3` | `4894be9a943876dc46e219ffcb68d1d7ce0fdb3981ae9255b0aba2ce4e6b5728` |
| A0T | C4 | `CANONICAL-V1-A0T-C4` | `eee28075336170682abc4fb7e17fd01f481776ea06d175c2cf0decada85ec609` |
| A0T | C5 | `CANONICAL-V1-A0T-C5` | `b46d1f5fe9df53b425d207df965af2656ca4290e1fe0cb6f723cdd8f0e007fa5` |
| GAPS/A4 | C3 | `CANONICAL-V1-A4-C3` | `e2364290ffc7fd9748fe86edb3745dca0eac692165f6c8aba1825728ddcd4414` |
| GAPS/A4 | C4 | `CANONICAL-V1-A4-C4` | `422a49f28331e5486d215a8d34bc9a972dc8fc1992f8b5bf27428329143599c3` |
| GAPS/A4 | C5 | `CANONICAL-V1-A4-C5` | `3965ec8618a2d496804bbc141f49e00b451fce05e9edbefde721f0dd4f912b93` |

Checkpoint equality and identity audits use ordered state-content fingerprints; whole-file SHA-256 is retained for provenance.

### R84_FED_H1

- Input: 83 canonical sensor statistics plus one frozen Federated-H1 prediction.
- Federated-H1 manifest SHA-256: `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`.
- Model family, feature ordering, fitting implementation, calibration rows, full-calibration refit rule, and prediction clipping behavior remain unchanged.
- Ridge coefficients are fitted separately for each method/target/gas because the routed H1 commissioning feature depends on the classifier checkpoint.
- Alpha is not searched. Both methods use the same pre-frozen per-target/per-gas alpha values selected by the existing canonical A4 R84 run:

| Target | Ethanol | CO | Ethylene | Methane |
|---|---:|---:|---:|---:|
| C3 | 100 | 0 | 0.1 | 0.1 |
| C4 | 1 | 10 | 0.1 | 10 |
| C5 | 1 | 0.01 | 10 | 0.1 |

No alpha grid is evaluated, logged, ranked, or selected in the new study.

## Executable comparison matrix

Exactly six regression evaluations are permitted:

| Experiment ID | Method | Target | Classifier | Regression |
|---|---|---|---|---|
| `CAN-V1-REG-A0T-C3-S42` | A0T | C3 | frozen A0T C3 round25 adapted | fixed-alpha R84_FED_H1 |
| `CAN-V1-REG-A0T-C4-S42` | A0T | C4 | frozen A0T C4 round25 adapted | fixed-alpha R84_FED_H1 |
| `CAN-V1-REG-A0T-C5-S42` | A0T | C5 | frozen A0T C5 round25 adapted | fixed-alpha R84_FED_H1 |
| `CAN-V1-REG-A4-C3-S42` | GAPS/A4 | C3 | frozen A4 C3 round25 adapted | fixed-alpha R84_FED_H1 |
| `CAN-V1-REG-A4-C4-S42` | GAPS/A4 | C4 | frozen A4 C4 round25 adapted | fixed-alpha R84_FED_H1 |
| `CAN-V1-REG-A4-C5-S42` | GAPS/A4 | C5 | frozen A4 C5 round25 adapted | fixed-alpha R84_FED_H1 |

The only upstream experimental factor is the classifier/adaptation checkpoint. Every other input and implementation choice is held constant.

## Data flow and sealing

For each method/target pair:

1. Verify the dataset, checkpoint, run manifest, fixed endpoint, H1 manifest, alpha lock, target split manifest, and QC assets.
2. Read target calibration and classifier calibration predictions only.
3. Build routed and Oracle 84-D calibration features.
4. Fit the four gas-specific R84 Ridge coefficients with the frozen alpha values on the complete calibration set.
5. Persist and hash the regression calibration lock before reading target test.
6. Open the unchanged target test once for fixed evaluation.
7. Generate deployment-routed and Oracle predictions, metrics, per-gas/per-concentration slices, QC outputs, and hashes.

Failure of any identity, count, order, split, or hash check stops the affected endpoint without substituting another checkpoint or changing the protocol.

## Evaluation scopes

Four scopes are persisted:

- `S_ALL`: all test rows with the classifier-predicted route and corresponding routed H1 feature.
- `S_CC`: the subset of `S_ALL` whose predicted class equals the true class.
- `Oracle_ALL`: all test rows forced through the true gas route, true-class H1 feature, and true gas R84 model.
- `Oracle_CC`: the same sample indices as `S_CC`, evaluated with the Oracle route and mapping.

For `S_ALL`, `S_CC`, `Oracle_ALL`, and `Oracle_CC`, report:

- RMSE in ppm;
- MAE in ppm;
- NRMSE by class range;
- R2;
- Bias in ppm;
- sample count.

For Ethanol, CO, Ethylene, and Methane, report RMSE, MAE, Bias, and sample count for all formal scopes.

Per-concentration outputs include RMSE, MAE, Bias, and sample count. The C5 analysis explicitly records CO, Methane, high-concentration strata, and Methane 225 ppm repeat1.

## Routing and regression decomposition

The requested primary quantities are:

- `routing_gap = RMSE(S_ALL) - RMSE(S_CC)`;
- `regression_gap = RMSE(S_CC) - RMSE(Oracle_ALL)`.

Because `S_CC` and `Oracle_ALL` have different sample populations, the study also reports the paired diagnostic:

- `paired_regression_gap = RMSE(S_CC) - RMSE(Oracle_CC)`.

The primary formula is preserved exactly as requested. Mechanism claims must use the paired diagnostic when sample-scope differences could change the interpretation.

## Frozen QC comparison

QC is conditional on the existing canonical-v1 equal-mean pipeline supporting the generated records without changing its interface or hidden dependencies.

- Use the exact existing target-specific normalization constants and HC90/HC95 threshold locks.
- Do not recompute normalization, quantiles, thresholds, workpoints, auxiliary models, or policy fields for A0T.
- Apply the same frozen policy independently to A0T and A4 predictions.
- Report target and pooled coverage, accepted RMSE, accepted+review RMSE, review rate, reject rate, and sample counts for HC90 and HC95.
- If the exact frozen QC cannot consume one method's records without changing the policy, mark QC `BLOCKED_UNSUPPORTED_INTERFACE`; do not approximate or tune it.

## Outputs

All new artifacts are written under the new destination:

`results/iotj_canonical_v1_final/a0t_vs_a4_regression/`

Required compact outputs:

- `regression_comparison.csv`
- `per_gas_regression_comparison.csv`
- `routing_scope_summary.csv`
- `qc_comparison.csv`
- `A0T_VS_A4_REGRESSION_REPORT.md`
- `ROUTING_VS_REGRESSION_ANALYSIS.md`
- `C5_A0T_VS_A4_REGRESSION.md`
- `A0T_VS_GAPS_FINAL_CONCLUSION.md`
- `protocol_manifest.json`
- `experiment_registry.csv`
- `checkpoint_sha256.json`
- `prediction_sha256.json`
- `test_manifest_sha256.json`
- `STRICT_AUDIT.json`

Per-endpoint calibration locks, model manifests, prediction records, per-concentration records, and endpoint manifests remain available locally for audit. Only compact evidence is committed; checkpoints and large prediction tables are excluded from Git.

## Testing and audit gates

Tests must verify:

- exactly six method/target endpoints;
- classifier training and adaptation entry points are unreachable;
- the checkpoint is the only method-varying input;
- the frozen alpha table is identical across methods and no alpha search path is called;
- calibration locks precede test access;
- all four metric scopes use the intended sample indices;
- Bias and range-normalized NRMSE definitions are fixed;
- C5 Methane 225 ppm repeat1 is present;
- QC consumes exact frozen locks and never refits thresholds;
- checkpoint, prediction, and test-manifest SHA-256 indices match the produced artifacts;
- target test is never used for fitting, selection, or stopping.

Before publication, run the relevant pytest set, `python -m compileall`, canonical dataset hash verification, checkpoint/state fingerprint checks, prediction/test-manifest hash verification, and strict protocol audit.

## Interpretation boundaries and stop rule

- Classification advantage is reported from the already audited A0T/A4 classification comparison; it is not recomputed through classifier training.
- Regression advantage is approved only by the frozen dual-gate rule above.
- Routing versus mapping attribution remains descriptive and is qualified by the Oracle sample scope.
- If classification advantage is unsupported but regression advantage is supported, the allowed interpretation is that semantic-aware commissioning improves quantitative sensing under this fixed canonical-v1 protocol.
- If regression advantage is not supported, the paper positioning shifts to lifecycle framework value rather than algorithmic superiority.
- On completion, stop. Do not start C3/C4 extensions beyond this matrix, new seeds, new budgets, new QC workpoints, R84 variants, or any algorithm/hyperparameter search.
