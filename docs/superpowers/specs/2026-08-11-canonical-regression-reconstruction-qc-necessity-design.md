# Canonical Regression Reconstruction + QC Necessity Validation Design

## Status

`PRE_RUN_DESIGN_FROZEN_NO_EXECUTION`

This document freezes the next canonical-v1 evidence chain. It does not authorize
hyperparameter search, new algorithms, manuscript edits, or any use of legacy
10-Hz/100x8 quantitative assets. Implementation and execution begin only after
this design commit is reviewed.

Base commit: `26d453eed62057fff45cb6abdb96037b48112ba4`

Study ID: `CAN-V1-CRRQ-20260811`

Result root:
`results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/`

Experiment-document root:
`docs/experiments/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/`

## Scientific objective

Close the final canonical quantitative-sensing evidence chain without reopening
classification algorithm search:

1. determine whether the official classification-v1 lifecycle can move target
   adaptation from every federated round to one final commissioning stage;
2. reconstruct H1/FedRidge and all 83D/R84 quantitative assets entirely from
   canonical-v1 5-Hz/50x8 windows;
3. test whether the canonical source prior improves target regression;
4. invoke only the preregistered transfer-safe Ridge alternatives if canonical
   R84 shows device- or gas-specific negative transfer; and
5. decide whether a selective-output QC layer adds useful risk ranking beyond
   random retention and classifier confidence.

The permitted claim is limited to canonical-v1 and its frozen split. The known
raw-file/raw-time relationship between calibration and test remains a limitation;
the fixed target split is not changed in this study.

## Selected design and rejected alternatives

### H1 porting decision

The selected interpretation is
`H1_CANONICAL_PORTABLE_AS_FIXED_RATE_DISCRETE_FEATURE_OPERATOR`.

`rich_feature_dict` contains no fixed 100-sample index and no hard-coded 10-Hz
constant. Its `slope`, `absdiff_mean`, and `absdiff_max` terms are nevertheless
defined per discrete sample step. They are therefore interpreted only as
**fixed-5-Hz discrete dynamic descriptors**. This study does not claim that they
are sampling-rate invariant or numerically equivalent between 10 Hz and 5 Hz.

All source and target feature matrices are recomputed from canonical-v1
5-Hz/50x8 windows with the same extractor implementation. No legacy numerical
H1 or 83D asset is ported.

Rejected alternatives:

- treating legacy 10-Hz features, scalers, alphas, or coefficients as canonical;
- resizing or interpolating 100x8 windows into 50x8 windows;
- changing the derivative terms to per-second units in this repair; and
- defining an H1-v2 feature schema inside this study.

### Classification timing design

C0-A selectively reuses the three already audited official interleaved endpoints.
C0-B uses one new target-independent source trajectory and three independent
final adaptation branches. The earlier G1 post-hoc A4 checkpoint is rejected as
a C0-B endpoint because its audit reports that interleaved client statistics were
unavailable and several A4 loss inputs were inactive.

### Regression design

R0 reuses the established sufficient-statistics mathematics and historical alpha
grid, but regenerates every numerical input from canonical-v1. R1 compares only
source-only canonical FedRidge, target-only 83D Ridge, and 83D plus one canonical
FedRidge prior. No neural regression model is admitted.

R2 is conditional and contains exactly two already specified Ridge constructions:
residual transfer and gas-specific shrinkage. No other rescue experiment is
allowed.

### QC design

Q0 first asks whether QC is necessary. It does not assume that the historical
equal-mean policy remains valid. An exact historical Q4 may be reported only if
every component can be regenerated from assets authorized by C0-R1/R2. It is
forbidden to silently substitute legacy H1/H2/H3 values or to relabel a changed
formula as the historical equal-mean policy.

The primary regression-only uncertainty signal is a preregistered five-fold
raw-file-grouped calibration model-dispersion score: with the final regression
method and its hyperparameters frozen, five auxiliary fold models are fit from
target calibration only, and the per-window standard deviation of their
predictions is divided by the routed gas range. These auxiliary models are QC
diagnostics only and never replace or select the final regressor.

If an exact canonical Q4 is unavailable, Q0 records
`Q4_CANONICAL_INPUTS_UNAVAILABLE`; it may not declare multisignal superiority.
That condition can trigger the authorized Q1 conformal-style comparison rather
than an improvised replacement score.

## Global canonical freeze

| Field | Frozen value |
|---|---|
| Dataset | `canonical-v1` |
| Preprocessing | `HZ5_MEAN_W10S` |
| Sampling | 5 Hz |
| Physical window | 10 s |
| Stride | 5 s |
| Sequence | 50x8 |
| Source | C1 + C2 |
| Targets | C3, C4, C5 where formal endpoint audit passes |
| Primary deployment target | C5 |
| Training seed | 42 only |
| Classification rounds | 25 |
| Classification local epochs | 1 |
| Classification batch size | 32 |
| Classification optimizer | Adam |
| Classification learning rate | 5e-4 |
| Checkpoint selection | fixed round 25 |
| Target calibration | frozen canonical target calibration identities |
| Target test | fixed canonical test; evaluation only |
| Ridge alpha grid | `{0, 0.01, 0.1, 1, 10, 100, 1000}` |
| Bootstrap | paired raw-file grouped, 5,000 replicates, seed 42 |
| Manuscript | impact report only; no body edits |

The following 10-Hz/100x8-derived assets are forbidden inputs to every executable
API and cache loader:

- H1 feature matrices;
- feature scalers;
- sufficient statistics;
- FedRidge coefficients;
- source Ridge alpha values;
- target 83D feature caches;
- R84 coefficients;
- QC normalization scales; and
- QC thresholds.

A cache is accepted only when its manifest contains the canonical source array
SHA256, metadata SHA256, extractor-code SHA256, ordered feature-name SHA256,
window shape `[50, 8]`, sampling rate 5 Hz, and the current study ID. Missing or
mismatched provenance causes rejection and recomputation, never fallback.

## Execution state machine

```text
C0
  -> freeze classifier lifecycle
  -> R0.1 feature-port and 83D cache audit
  -> R0.2/R0.3/R0.4 canonical FedRidge reconstruction
  -> R1 canonical target regression
  -> stop, evaluate registered R1 rule
       SUPPORTED      -> freeze R84; skip R2
       DEVICE_DEPENDENT -> run R2
       NOT_SUPPORTED  -> freeze 83D; skip R2
  -> freeze final regression backend
  -> Q0 QC necessity
  -> run Q1 only when its registered trigger is true
  -> manuscript-impact report
  -> stop
```

No downstream gate may start without the preceding gate's completion marker,
hash index, leakage audit, relevant tests, and `compileall` PASS.

## Gate C0 — classification-v1 final-adapt simplification

### C0-A: V1_INTERLEAVED

Reuse only after checkpoint, protocol, target manifest, round, seed, and state
fingerprint audits pass:

| Target | Checkpoint SHA256 | Existing Macro-F1 |
|---|---|---:|
| C3 | `e2364290ffc7fd9748fe86edb3745dca0eac692165f6c8aba1825728ddcd4414` | 0.9985068849 |
| C4 | `422a49f28331e5486d215a8d34bc9a972dc8fc1992f8b5bf27428329143599c3` | 0.9977941081 |
| C5 | `3965ec8618a2d496804bbc141f49e00b451fce05e9edbefde721f0dd4f912b93` | 0.9941260906 |

The historical target-test metrics are reference evidence only. They do not tune
or select C0-B.

### C0-B: V1_FINAL_ADAPT

Run exactly one new source trajectory:

- fresh seed42 initialization;
- C1/C2 canonical train data only;
- the same 25 rounds, local optimizer, local epoch, batch size, client profile,
  aggregation, state ordering, and diagnostic payload schema as C0-A;
- no target directory or target tensor is accepted by the round-1-to-25 source
  training API;
- no round-level target adaptation is executed; and
- the complete round-25 source class-phase statistics, semantic prototypes,
  prototype variances, device residuals, and teacher/state artifacts that C0-A
  would make available to its server adaptation call are persisted.

From the same ordered-state fingerprint of the round-25 aggregated checkpoint,
start three independent final-adaptation processes for C3/C4/C5. Each process
reloads the original round-25 source state and its exact round-25 client-derived
adaptation inputs. It runs one 100-step A4 server adaptation using the target's
unchanged calibration identities.

The following must equal C0-A's registered per-invocation adaptation settings:

- all configured loss names and coefficients;
- all required loss inputs and their availability;
- optimizer Adam, learning rate 5e-4, batch size 32, and seed42;
- 100 steps;
- source batch convention;
- target fields `x`, class, and phase; and
- no target-test access, early stopping, hyperparameter search, or checkpoint
  selection.

An activity audit records for every loss: `loss_name`, `configured_weight`,
`input_available`, `active_steps`, `mean_raw_loss`, `mean_weighted_loss`, and
`inactive_reason`. A nonzero-weight C0-A loss that becomes unavailable solely
because of the final schedule is a hard comparability failure. The old G1
post-hoc result may not be substituted.

The interleaved lifecycle invokes 100 target steps after each of 25 rounds,
whereas final-only invokes one 100-step stage. This is the intended lifecycle
timing policy and its resulting commissioning-compute reduction; it is not
reported as an optimizer-controlled equal-total-step ablation.

After all three C0-B endpoints are locked, evaluate C3/C4/C5 test once and report
Accuracy, Macro-F1, NLL, ECE, adaptation seconds, and total target-adaptation
steps.

Decision:

- if `MacroF1_final - MacroF1_interleaved >= -0.005` for every target:
  `V1_FINAL_ADAPT_SUPPORTED`;
- otherwise: `V1_INTERLEAVED_RETAINED`.

No extra steps or retuning are allowed. If C0-B cannot satisfy loss-input parity,
the simplification is `NOT_ESTABLISHED` and the official interleaved endpoints
remain frozen for R1.

Required outputs:

- `CLASSIFICATION_V1_FINAL_ADAPT_REPORT.md`
- `classification_v1_final_adapt.csv`
- `classification_loss_activity.csv`
- source and adapted checkpoint fingerprints
- calibration/test access audit and hash index

## Gate R0 — canonical FedRidge reconstruction

### R0.1 feature and cache audit

The ordered H1 schema remains 104D: 83 sensor statistics plus 21 metadata/phase
fields. The ordered target-only schema remains the exact 83 sensor-statistic
subset. Both are recomputed by applying the frozen extractor to canonical 50x8
arrays.

The audit must prove:

- every opened raw feature array has shape `N x 50 x 8`;
- H1 and 83D rows share the same canonical physical identity and window;
- no array is resized or interpolated after canonical preprocessing;
- the 83D cache is newly generated or has a complete matching canonical cache
  manifest;
- the H1 cache is newly generated or has a complete matching canonical cache
  manifest; and
- derivative-like fields are documented only as fixed-5-Hz discrete descriptors.

Any legacy cache is rejected and rebuilt. A changed feature definition, missing
canonical raw array, wrong shape, or source/target feature-schema mismatch stops
R0 before fitting.

### R0.2 sufficient statistics

For each gas and source client, produce local statistics from canonical C1/C2:

1. feature moments `n`, `sum_x`, and `sum_x2`;
2. globally standardized normal-equation terms `X'X`, `X'y`, `y'y`, `y_min`,
   and `y_max`; and
3. local validation SSE/count for each frozen alpha candidate.

Server aggregation APIs reject raw rows, raw X/y, per-sample labels, and
per-sample predictions. Population variance is `sum_x2/n - mean^2`; scales below
`1e-9` become 1.0. A leading intercept is added and is not regularized. Ridge is
solved with the existing pseudoinverse convention.

### R0.3 source alpha

The existing canonical source roles are used exactly:

- C1/C2 `train` selects candidate models;
- C1/C2 `calibration` supplies distributed clipped SSE/count for alpha selection;
- C1/C2 `train + calibration` supplies final refit sufficient statistics; and
- C1/C2 `test` is opened only after alpha/model locks and is evaluation-only.

Select one alpha per gas by minimum pooled source-calibration RMSE, breaking ties
by the earliest value in the frozen historical grid. No target asset is accepted
by source fitting or selection APIs.

### R0.4 exact recovery

Under the same canonical rows, scaler, alpha, intercept convention, clipping, and
float64 implementation, compare the sufficient-statistics model with a pooled
audit-only Ridge reference. The required exact-equivalence tolerances are:

- scaler maximum absolute difference <= `1e-10`;
- coefficient/intercept maximum absolute difference <= `1e-8`; and
- prediction maximum absolute difference <= `1e-6` ppm.

Failure stops R1. Practical-equivalence fallback is not sufficient for this
canonical reconstruction gate.

Required outputs:

- `H1_CANONICAL_PORTING_AUDIT.md`
- `CANONICAL_83D_CACHE_AUDIT.md`
- `canonical_fedridge_models/`
- `canonical_fedridge_sufficient_statistics/`
- `canonical_fedridge_alpha_lock.json`
- `canonical_fedridge_exact_recovery.json`
- `CANONICAL_FEDRIDGE_RECONSTRUCTION_REPORT.md`
- hash and leakage audits

## Gate R1 — canonical target regression

### Methods

- `SOURCE_ONLY_FEDRIDGE`: the canonical global H1/FedRidge prediction;
- `TARGET_ONLY_83D_RIDGE`: canonical target sensor statistics only; and
- `R84_CONCAT`: the same 83D features plus exactly one canonical global
  FedRidge prediction.

The only difference between 83D and R84 is the single source-prior column. The
same target calibration/test rows, routes, grouping, clipping convention, and
metric code are used.

### Target alpha

For each target, method, and gas, choose alpha from the frozen historical grid
using deterministic five-fold raw-filename-grouped CV inside target calibration.
Groups are assigned in stable sorted order with fold balancing; no raw filename
may cross folds. Score is pooled validation RMSE. Ties use the earliest grid
value. Refit on all target calibration after writing the alpha lock. Target test
is unavailable to the selector API.

The frozen canonical split is not changed even though some target calibration
and test windows originate from the same raw file/time neighborhood. This known
limitation must appear in every interpretation report.

### Routes and metrics

The classifier selected by C0 is used for each target:

- `S_ALL`: predicted route, all test rows;
- `S_CC`: predicted route, classifier-correct rows only;
- `Oracle_ALL`: true-class route, all test rows; and
- `Oracle_CC`: true-class route on the same classifier-correct subset.

Report N, RMSE, MAE, NRMSE_range, R2, and bias. NRMSE_range is
`sqrt(mean((prediction - truth)^2 / gas_range^2))`. Report each target, pooled,
each gas, and each exact registered concentration level. Retain a dedicated row
and narrative for C5 Methane 225 ppm repeat1.

### Paired grouped bootstrap

Use 5,000 paired replicates with seed42. Resample `target|raw_filename` groups
with replacement; pooled replicates are stratified by target so target weights
remain fixed. Compute `R84 - 83D` deltas for RMSE, MAE, and NRMSE_range with
percentile 95% intervals for C3/C4/C5, pooled, and every gas.

Decision:

- `CANONICAL_R84_SUPPORTED`: pooled RMSE-delta CI is entirely below zero and no
  target or gas has positive RMSE delta;
- `CANONICAL_R84_DEVICE_DEPENDENT`: pooled point estimate improves but its CI
  crosses zero, or any target/gas has positive RMSE delta; or
- `CANONICAL_R84_NOT_SUPPORTED`: pooled R84 RMSE is not lower than pooled 83D.

Any target RMSE increase above 5% is additionally flagged as severe collapse.
No result changes alpha, features, checkpoints, or the run matrix.

Required outputs:

- `CANONICAL_83D_VS_R84_REPORT.md`
- `canonical_regression_comparison.csv`
- `canonical_regression_bootstrap.csv`
- `canonical_regression_per_gas.csv`
- `canonical_regression_per_concentration.csv`
- prediction, alpha, classifier, and test-manifest hashes

## Gate R2 — conditional transfer-safe regression

R2 runs only when R1 is `CANONICAL_R84_DEVICE_DEPENDENT` or when R84 has a clear
gas-specific negative-transfer flag while retaining a pooled benefit. It is
skipped when R84 is cleanly supported. If R84 is globally not supported, 83D is
frozen directly and R2 is skipped; this gate is not a general R84 rescue search.

The two allowed candidates are:

1. `RESIDUAL_TRANSFER`: per gas,
   `prediction = canonical_source_prior + Ridge_83D(target residual)`;
2. `SHRINKAGE_TRANSFER`: per gas,
   `prediction = (1-beta) * target_83D + beta * canonical_source_prior`, with
   `beta in {0, 0.25, 0.5, 0.75, 1}`.

Alpha or beta selection uses the same five-fold target-calibration raw-file
grouping and no target test. The frozen R1 target-83D and canonical-source inputs
are unchanged.

A candidate is retained only when all conditions hold:

- pooled S_ALL RMSE improves at least 3% relative to R84;
- no gas RMSE degrades more than 5%, unless the preregistered difficult-case
  audit proves the degradation is confined to the documented anomaly; and
- the paired grouped-bootstrap RMSE-delta interval is entirely below zero.

If neither candidate passes, retain R84 for a device-dependent R1 outcome. No
further regression architecture is opened.

Required output: `TRANSFER_SAFE_REGRESSION_REPORT.md` plus comparison,
bootstrap, selection-lock, prediction, and hash files.

## Gate Q0 — QC necessity

Q0 starts only after a single final regression backend is frozen. The primary
decision target is C5; C3/C4 and three-target pooled results are secondary
consistency evidence. Each target uses only its own calibration-derived scales.

Policies:

- `FULL_OUTPUT`: coverage 100%;
- `RANDOM_RETAIN`: equal-count random retention, 5,000 repetitions, seed42;
- `CLASSIFICATION_CONFIDENCE_ONLY`: risk `1 - max(class probability)`;
- `REGRESSION_UNCERTAINTY_ONLY`: five-fold raw-file-grouped calibration model
  prediction dispersion divided by routed gas range; and
- `EXISTING_EQUAL_MEAN_QC`: exact historical formula only if a canonical-input
  availability audit passes. No legacy source-prior ensemble, normalization, or
  threshold may enter. If exact reconstruction is not authorized by completed
  gates, record `Q4_CANONICAL_INPUTS_UNAVAILABLE` and do not replace it silently.

Use the common coverage grid 0.50, 0.51, ..., 1.00 with deterministic
physical-identity tie breaking. Report RMSE, NRMSE_range, MAE, misroute rate,
error >=40 ppm rate, P90 absolute error, AURC_RMSE, and AURC_NRMSE.

At nominal HC90 and HC95 and at the actual Q4 retained counts, compare exactly
the same retained N for random, confidence, regression-only, and Q4 when
available. QC thresholds and normalization are derived from calibration only
and locked before test evaluation.

Decision:

- `MULTISIGNAL_QC_SUPPORTED` only if canonical Q4 is available and beats both
  random and confidence by at least 5% relative NRMSE-AURC on C5 and pooled;
- `CONFIDENCE_QC_PREFERRED` if confidence matches or beats Q4, or if confidence
  is the best available policy and reliably beats matched random;
- `QC_CORE_NOT_SUPPORTED` if no available risk policy reliably beats matched
  random; or
- `MULTISIGNAL_QC_NOT_ESTABLISHED` if Q4 cannot be reconstructed canonically.

The last outcome satisfies the Q1 trigger but is not evidence that the legacy Q4
failed numerically.

Required outputs include risk-coverage curves, AURC table, matched-count table,
QC availability/leakage audit, policy locks, and `QC_NECESSITY_REPORT.md`.

## Gate Q1 — conditional conformal-style uncertainty

Q1 runs only if Q0 returns `MULTISIGNAL_QC_NOT_ESTABLISHED`, or shows that the
available regression uncertainty is scientifically useful but the current QC is
inferior to confidence. It is skipped when Q0 already supports a final simple or
multisignal QC policy.

For each target and gas, split target calibration by raw filename into a
deterministic group-aware fit subset and conformal-calibration subset. No group
crosses the two subsets. Fit the already frozen regression method on the fit
subset without changing its hyperparameters. Use absolute residuals from the
conformal-calibration subset to construct an empirical split-conformal-style
interval. Because the windows are dependent, all reports use the terms
`group-aware conformal-style interval` or `empirical prediction interval`, not an
exact iid coverage guarantee.

Compare once:

- confidence-only;
- interval-width-only; and
- equal mean of calibration-CDF-normalized confidence risk and interval width.

There is no weight search. Report empirical coverage, mean/median width, coverage
by gas and exact concentration, risk-coverage AURC, same-count RMSE, and large
error capture.

If the combined interval signal improves NRMSE-AURC by at least 5% relative to
confidence on both C5 and pooled, decide `CONFORMAL_AUGMENTED_QC_SUPPORTED`;
otherwise decide `CONFIDENCE_QC_FINAL`.

## Leakage, provenance, and fail-closed requirements

Every gate must verify and hash:

- canonical dataset assets used by that gate;
- calibration and test manifests;
- ordered classifier state contents and raw checkpoint files;
- extractor code and ordered feature schemas;
- H1/83D caches;
- sufficient statistics and scalers;
- source and target alpha/beta locks;
- QC/conformal policy locks; and
- saved predictions.

Hard failures include:

- any opened regression array with a non-50x8 shape;
- any unmanifested or legacy feature cache/model/scaler/alpha/QC asset;
- target test entering fit, alpha/beta choice, QC/conformal calibration,
  checkpoint selection, early stopping, or model selection;
- source FedRidge training APIs accepting any target path or tensor;
- group overlap inside target CV or conformal fit/calibration partitions;
- bootstrap splitting a raw-file group; and
- failed exact FedRidge recovery.

Each gate runs relevant pytest tests and `python -m compileall`. Completed endpoint
markers are immutable; a failed endpoint is preserved and is not overwritten or
silently rerun.

## Required tests frozen at design time

At minimum, implementation must add tests for:

- C0 source-round APIs reject target inputs;
- C0 final branches reload the same ordered round-25 state;
- C0 client statistics/prototypes/teacher inputs match interleaved availability;
- C0 loss activity and 100-step fixed endpoint;
- H1/83D cache manifests reject 10-Hz, 100x8, wrong code hash, and missing hashes;
- H1 schema is 104D and target sensor schema is 83D on 50x8 inputs;
- source server APIs reject raw/sample-level payloads and target assets;
- source alpha is selected only from source calibration;
- sufficient-statistics and pooled Ridge meet exact tolerances;
- target alpha folds are raw-filename disjoint and test-free;
- S_ALL/S_CC/Oracle_ALL/Oracle_CC definitions are stable;
- bootstrap preserves paired raw-file groups;
- R2 and Q1 cannot run unless their trigger files authorize them;
- QC scales/thresholds are calibration-only;
- unavailable historical Q4 cannot be silently replaced; and
- target test opens only after all selection locks for its gate exist.

## Commit and stopping protocol

Use separate commits and push after each completed gate:

1. classification-v1 final-adapt validation;
2. canonical FedRidge reconstruction;
3. canonical 83D-versus-R84 validation;
4. transfer-safe regression only if triggered;
5. QC necessity validation; and
6. conformal-style QC only if triggered.

After the final gate, generate `NEXT_STAGE_MANUSCRIPT_IMPACT.md` without editing
the manuscript. Report the required A-G decision summary and stop. Do not start
new classifiers, DG, MME, prototypes, deep regression, source-diversity, routing,
or hyperparameter searches.
