# Canonical FedRidge R0-v2 Numerically Stable Reconstruction Design

## 1. Document status

- Date: 2026-08-12 (Asia/Shanghai)
- Branch: `codex/iotj-final-classification-le1`
- Design base commit: `5e1cdf96d4054c96c85964f1a9fde60b1dfb3f5b`
- Proposed study ID: `CAN-V1-FEDRIDGE-R0V2-20260812`
- Design status: approved in conversation; design-spec commit only
- Formal execution status: not started
- Target access status: no target calibration or test access authorized

This document freezes the design for a new, independently versioned numerical-stability protocol. It does not modify, supersede, rerun, or reinterpret the completed C0 or the failed original R0.

## 2. Frozen prior conclusions

### 2.1 C0

The frozen C0 decision remains:

```text
V1_INTERLEAVED_RETAINED
```

C0 will not be rerun. No new classification algorithm experiment is authorized.

### 2.2 Original R0

The original R0 decision remains:

```text
R0_EXACT_RECOVERY_NOT_ESTABLISHED
```

The original R0 result root, protocol, thresholds, audits, and failure interpretation are immutable. R0-v2 is a new protocol motivated by the already-audited floating-point cancellation and reduction-order mechanism; it is not a relaxed rerun of R0.

### 2.3 Downstream gates

R1, conditional R2, Q0, and conditional Q1 remain blocked. A future R0-v2 PASS may open R1 for a separate authorized action, but it will not start R1 automatically.

## 3. Scientific question and hypotheses

### 3.1 Question

Can the canonical-v1 source-only sufficient-statistics FedRidge implementation reconstruct the same regularized normal-equation problem as pooled Ridge, and produce numerically equivalent predictions within preregistered IEEE-754 floating-point tolerances, when global standardization is reconstructed with stable mergeable central moments?

### 3.2 Hypothesis H-R0V2-ALG

In exact arithmetic, with identical source samples, feature coordinates, population standardization, intercept convention, regularization matrix, and alpha:

```text
sum_k X_k^T X_k = X_pool^T X_pool
sum_k X_k^T y_k = X_pool^T y_pool
```

Therefore, sufficient-statistics FedRidge and pooled Ridge define the same regularized normal equations.

### 3.3 Hypothesis H-R0V2-NUM

In float64 implementation, stable mergeable central moments and deterministic aggregation will produce scaler, normal-equation, residual, and source-test prediction discrepancies within the preregistered numerical gates in Section 10.

### 3.4 Falsification

If any gas fails any hard numerical or access gate, the decision is:

```text
R0_V2_FAILED
```

R1 remains blocked and no tolerance, solver, alpha, feature, or aggregation rule may be changed after observing the result.

## 4. Scope and held constants

### 4.1 Canonical data

- Dataset: `dataset/iotj_canonical_v1`
- Aggregate SHA256: `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`
- Preprocessing: `HZ5_MEAN_W10S`
- Sampling: 5 Hz
- Window: 10 seconds
- Stride: 5 seconds
- Tensor shape: `50 x 8`
- Source clients: ordered `C1`, `C2`
- Source train count: 2360 per client
- Source calibration count: 320 per client
- Source test count: 680 per client
- Per-gas train-plus-calibration refit count: 670 per client, 1340 pooled
- Per-gas source-test count: 170 per client, 340 pooled

### 4.2 Quantitative features

- Sensor-only dimensions: 83
- H1 dimensions: 104 = 83 sensor statistics + 21 metadata/phase descriptors
- Dynamic descriptors remain fixed-5-Hz discrete per-sample descriptors
- No sampling-rate-invariance claim
- No 10-Hz/5-Hz numerical-equivalence claim
- No feature deletion
- No H1-v2 feature definition
- No per-second derivative conversion

### 4.3 Ridge protocol

- Numeric type: float64 throughout
- Population variance: denominator `n`, equivalent to `ddof=0`
- Safe-scale floor: existing frozen `SCALE_FLOOR=1e-9`
- Alpha grid: `(0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)`
- Alpha selection: source train fit and source calibration distributed SSE/count only
- Tie break: first alpha in the registered grid
- Final refit: source train plus source calibration
- Solver: unchanged `numpy.linalg.pinv`
- Intercept: unregularized
- Coefficients: per-gas Ridge
- Target input: forbidden
- QC: unavailable and not executed

## 5. Independent versioning and paths

The original implementation and runner remain untouched:

```text
gaps_flower/canonical_fedridge.py
scripts/run_iotj_canonical_regression_reconstruction_r0.py
results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/R0/
```

R0-v2 uses new files:

```text
gaps_flower/canonical_fedridge_v2.py
scripts/run_iotj_canonical_fedridge_r0_v2.py
tests/test_iotj_canonical_fedridge_r0_v2.py
```

Protocol destination:

```text
docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/
```

Formal result destination, created only after a separate execution authorization:

```text
results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/
```

The pre-run manifest status must be:

```text
DESIGN_FREEZE_READY_FORMAL_NOT_STARTED
```

## 6. Fresh feature-cache policy

R0-v2 will not reuse the original R0 feature cache. It will recompute C1/C2 train and calibration 83D/104D features from the canonical 50x8 arrays into the new R0-v2 result root.

Each cache manifest must bind:

- study ID;
- client and split;
- source array, phase array, metadata, and row-identity hashes;
- canonical dataset aggregate hash;
- extractor file and function-source hashes;
- ordered 83D and 104D feature-name hashes;
- sampling rate and window shape;
- float64 dtype;
- row count and row order;
- `created_from_canonical_arrays=true`;
- `legacy_cache_reused=false`.

No C3/C4/C5 cache will be created under R0-v2.

## 7. Stable global standardization

### 7.1 Local record

For each client, gas, role, and feature coordinate, compute in float64:

```text
n_k
mean_k
M2_k = sum_i (x_i - mean_k)^2
min_k
max_k
```

The implementation must not use `sum_x2/n - mean^2` for canonical R0-v2 variance.

### 7.2 Deterministic merge

Merge blocks A and B using:

```text
delta = mean_B - mean_A
n = n_A + n_B
mean = mean_A + delta * n_B / n
M2 = M2_A + M2_B + delta^2 * n_A * n_B / n
```

Client aggregation order is fixed and recorded as:

```text
C1 -> C2
```

Within each client, the canonical manifest row order is retained. The pooled reference concatenates the same rows in the same C1-then-C2 order.

### 7.3 Population variance and safe scale

```text
variance = max(M2 / n, 0)
raw_scale = sqrt(variance)
canonical_scale = 1.0 if raw_scale < 1e-9 else raw_scale
```

The comparison is strict at the boundary: raw scale equal to `1e-9` is retained, matching the existing `< SCALE_FLOOR` semantics.

## 8. Near-constant feature policy

All 104 H1 features are audited for:

- minimum;
- maximum;
- mean;
- population variance;
- raw scale;
- dynamic range;
- safe-scale applied flag;
- final canonical scale.

`window_len_s` receives an explicit named row in the audit. It is not deleted or redefined. Its canonical-v1 representation-level spread is expected to fall below the existing `1e-9` scale floor, in which case both federated and pooled paths must use scale 1.0.

The `1e-9` threshold is reused from the original frozen protocol. It is not selected from regression performance, original R0 discrepancies, source-test metrics, or target data.

Formal output:

```text
H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv
```

## 9. Normal equations and model reconstruction

Under the single global canonical scaler, each client computes in float64:

```text
A_k = X_k^T X_k
b_k = X_k^T y_k
```

The server aggregates in recorded C1-then-C2 order:

```text
A = A_C1 + A_C2
b = b_C1 + b_C2
```

The pooled reference constructs a design matrix from exactly the same standardized source rows in the same order and directly computes `A_pool` and `b_pool`.

For each registered alpha:

```text
M = A + alpha P
```

where `P[0,0]=0` and all non-intercept diagonal entries are 1. Both paths retain the existing `numpy.linalg.pinv` convention. No alternative solver, explicit inverse, alpha expansion, or target-based selection is allowed.

## 10. Preregistered numerical gates

### 10.1 Constants

```text
epsilon = 2.220446049250313e-16
d = 104
p = 105
n_max = 1340
gamma(m) = m*epsilon / (1 - m*epsilon)
tau_moment = 64*gamma(1340) ~= 1.9043e-11
tau_residual = 128*gamma(105) ~= 2.9843e-12
tau_functional = 1e-6 ppm
```

These rules are frozen before formal output generation. The original R0 observed prediction discrepancy is not an input to the tolerance calculation.

### 10.2 Scaler gate

For feature `j`, define:

```text
S_j = max(1, max_abs_j, dynamic_range_j, abs(mean_pool_j), scale_pool_j)
```

Require:

```text
abs(mean_fed_j - mean_pool_j) <= tau_moment*S_j
abs(scale_fed_j - scale_pool_j) <= tau_moment*S_j
```

The federated and pooled safe-scale masks must be exactly identical.

Report mean and maximum absolute mean/scale discrepancies in addition to the per-feature normalized checks.

### 10.3 Sufficient-statistics gate

Report and require:

```text
relative_A = ||A_fed-A_pool||_F / ||A_pool||_F <= tau_moment
relative_b = ||b_fed-b_pool||_2 / ||b_pool||_2 <= tau_moment
```

Absolute norms are also reported. A zero or nonfinite denominator is a hard failure rather than a silent replacement.

### 10.4 Condition-number gate

For each gas, report:

```text
kappa = cond(A + alpha P)
```

Require:

```text
kappa is finite
kappa*epsilon < 1
```

Condition number is not allowed to change alpha or any threshold.

### 10.5 Linear-system residual gate

For both federated and pooled coefficients:

```text
r = (A + alpha P) beta - b
relative_residual = ||r||_2 / (||A+alpha P||_2*||beta||_2 + ||b||_2)
```

Require:

```text
relative_residual <= tau_residual
```

### 10.6 Coefficient diagnostic

Report:

- maximum absolute beta difference;
- relative L2 beta difference;
- condition-aware forward-error envelope:

```text
beta_forward_envelope = kappa*(2*tau_moment + tau_residual)
```

Report whether the observed relative L2 beta difference falls inside this envelope. This is diagnostic-only and has no cap or independent PASS/FAIL effect.

Coefficient discrepancy is important diagnostic evidence but is not an independent hard-fail gate. It cannot override a failed scaler, sufficient-statistics, residual, condition, or functional gate.

### 10.7 Functional-equivalence gate

After alpha and both final source models are locked, open the same C1/C2 source-test samples and labels. Report and require:

```text
max abs raw prediction difference <= 1e-6 ppm
max abs clipped prediction difference <= 1e-6 ppm
abs(RMSE_fed-RMSE_pool) <= 1e-6 ppm
abs(MAE_fed-MAE_pool) <= 1e-6 ppm
```

### 10.8 Absolute identity and safety gates

- Selected alpha is exactly identical for all four gases.
- Ordered client list is exactly `C1,C2`.
- Feature names, order, dimensions, and hashes are identical.
- All inputs, statistics, coefficients, residuals, and predictions are finite.
- No target path, array, label, calibration, test, or checkpoint is accepted by the API.
- All four gases must pass all hard gates.

## 11. Access sequence and leakage control

The execution plan is fixed as:

```text
verify design freeze and canonical dataset
verify original C0/R0 remain immutable
create fresh C1/C2 train/calibration feature caches
write H1 numerical audit
perform source-only alpha selection
refit source train+calibration Fed/pooled models
write alpha and model locks
open source test only after locks
evaluate functional equivalence
write diagnostics, decision, audit, and SHA256 index
stop
```

Forbidden throughout R0-v2:

- target C3/C4/C5 inputs of any kind;
- target test access;
- C0 execution;
- R1/R2/Q0/Q1 execution;
- QC thresholds or QC execution;
- new regression model;
- feature definition change;
- solver or alpha search;
- tolerance search or post-result adjustment.

## 12. Implementation architecture

### 12.1 `canonical_fedridge_v2.py`

Owns:

- immutable central-moment and normal-equation record types;
- finite float64 validation;
- local central-moment computation;
- deterministic C1/C2 merge with aggregation-order provenance;
- population scaler and safe-scale mask;
- client normal equations;
- deterministic server aggregation;
- source-only alpha selection;
- model reconstruction using the frozen solver;
- numerical-diagnostic helpers;
- R0-v2 gate evaluation.

It does not import or mutate target, QC, classifier, deployment, or original R0 result code.

### 12.2 `run_iotj_canonical_fedridge_r0_v2.py`

Owns:

- preflight and formal-lock verification;
- canonical data/hash checks;
- fresh feature-cache generation;
- source role and label-access sequencing;
- output creation and fail-closed behavior;
- decision and SHA256 audit writing.

The runner must reject a nonempty output root, a changed protocol hash, an unregistered execution commit, any target argument, or an existing completion marker.

The execution audit must record Python, NumPy, platform, dtype, and available BLAS/LAPACK configuration so the registered reduction environment is traceable. These records do not authorize environment-dependent threshold changes.

### 12.3 Test isolation

Synthetic tests exercise the v2 module without opening formal canonical arrays. Protocol tests may read manifest files and dataset count metadata, but may not produce formal numerical results.

## 13. Required TDD coverage

Tests must cover:

1. constant feature gives `M2=0` and canonical scale 1.0;
2. `1e-14`-scale jitter does not create spurious amplification;
3. large offset plus small variance remains stable where raw moments cancel;
4. population variance is retained and `ddof=1` is rejected by expected values;
5. safe-scale behavior below and exactly at `1e-9`;
6. float64 output and NaN/Inf rejection;
7. unsorted input records are deterministically aggregated and recorded as C1 then C2;
8. normal-equation A/b parity on synthetic data;
9. unregularized intercept and frozen alpha grid;
10. coefficient discrepancy remains diagnostic-only;
11. every hard gate independently causes failure;
12. nonfinite condition number or `kappa*epsilon>=1` causes failure;
13. source model lock precedes source-test label access;
14. target inputs are structurally absent/rejected;
15. original C0/R0 paths and files are not write targets;
16. nonempty output, protocol mismatch, or execution-commit mismatch fails closed;
17. decision vocabulary is restricted to the registered PASS/FAIL terms.

## 14. Formal outputs

The future authorized run must generate:

```text
canonical_feature_caches/
H1_CANONICAL_FEATURE_NUMERICAL_AUDIT.csv
r0_v2_scaler_diagnostics.csv
r0_v2_normal_equation_diagnostics.csv
r0_v2_system_diagnostics.csv
r0_v2_functional_equivalence.csv
source_alpha_audit.csv
source_alpha_lock.json
model_lock.json
DATA_ACCESS_AUDIT.md
R0_V2_DECISION.json
R0_V2_EXPERIMENT_AUDIT.md
protocol_manifest_execution.json
sha256_index.json
```

Protocol/design artifacts created before execution:

```text
PROTOCOL.md
protocol_manifest.json
EXPERIMENT_PLAN.md
EXPERIMENT_MATRIX.csv
NEAR_CONSTANT_SCALE_POLICY.md
R0_V2_NUMERICAL_TOLERANCE_JUSTIFICATION.md
FEDRIDGE_NUMERICAL_STABILITY_MANUSCRIPT_NOTE.md
```

The manuscript note proposes only a future minimal implementation clarification. It does not edit the manuscript and does not frame numerical stability as a new algorithmic contribution.

## 15. Decision and claim boundary

If every gas passes every hard gate:

```text
FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED
```

Permitted future wording must distinguish:

```text
algebraically identical normal equations in exact arithmetic
```

from:

```text
numerically equivalent within preregistered floating-point tolerance in implementation
```

Forbidden wording:

```text
bitwise exact recovery
```

If any hard gate fails:

```text
R0_V2_FAILED
```

R1 remains blocked and the project returns to protocol discussion.

## 16. Failure preservation

- Preserve partial outputs, logs, locks, and hashes.
- Do not overwrite, delete, resume, or rerun automatically.
- Do not relax tolerances or substitute practical equivalence.
- Do not change feature, solver, alpha, or aggregation order.
- Do not promote a favorable prediction metric if another hard gate fails.
- Emit a failure audit that identifies the exact gate and unopened downstream assets.

## 17. Two-layer commit and review gate

### 17.1 Design-spec commit

This document is committed alone after completeness, consistency, scope, and ambiguity review. It contains no formal implementation or result.

### 17.2 Pre-run freeze commit

Only after user review of this design:

1. create the implementation plan;
2. implement the independently versioned module and runner with TDD;
3. create the protocol, tolerance, near-constant, plan, matrix, and manuscript-note artifacts;
4. run pytest, compileall, static access audit, and manifest/hash checks;
5. update `代码文件介绍.md` with the new planned stage;
6. commit and push a pre-run freeze with `formal_execution_started=false`;
7. report protocol summary, near-constant policy, numerical gates, and commit hash;
8. wait for separate authorization before formal R0-v2 execution.

## 18. Explicit non-goals

- No C0 rerun
- No classification experiment
- No target commissioning
- No R1/R2 execution
- No QC/Q1 execution
- No target cache
- No new feature or regression model
- No alpha expansion
- No solver comparison
- No hyperparameter search
- No manuscript body edit
- No claim of algorithmic novelty from numerical-stability engineering
