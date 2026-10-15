# Canonical Regression Reconstruction + QC Necessity Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan. In this session the plan is executed inline because the active collaboration policy does not authorize sub-agent delegation.

**Goal:** Reconstruct the canonical-v1 quantitative regression and QC evidence chain while isolating target-adaptation timing in C0 and preventing every legacy 10 Hz/100x8 quantitative artifact from entering the study.

**Architecture:** Add fail-closed, separately testable contracts around (1) round25 A4 adaptation-context capture and final-only adaptation, (2) canonical 50x8 quantitative feature construction and sufficient-statistic recovery, (3) routed regression evaluation and grouped uncertainty, and (4) frozen QC evaluation. Each stage consumes only the immutable output of its predecessor, emits a fixed-endpoint marker plus SHA256 index, and must be committed and pushed before the next stage opens.

**Tech Stack:** Python 3.10+, PyTorch, Flower, NumPy, pandas, scikit-learn, pytest, existing `gaps_flower` model/DA code, existing three-machine SSH runner.

## Global Constraints

- Frozen design: `docs/superpowers/specs/2026-08-11-canonical-regression-reconstruction-qc-necessity-design.md`.
- Frozen manifest: `docs/experiments/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/protocol_manifest.json`.
- Order: C0 -> R0 -> R1 -> conditional R2 -> Q0 -> conditional Q1.
- Dataset: canonical-v1 only, 5 Hz, 50x8 windows, seed42, C1/C2 source, C3/C4/C5 targets, frozen 20% calibration and sealed test identities.
- No algorithm search, target-test selection, checkpoint selection, split change, feature-definition change, R84/QC modification, or rescue run.
- Formal output root: `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811`.
- A completed endpoint is immutable and never rerun. A partial failure is preserved and reported fail-closed.
- Every execution packet ends with targeted pytest, `python -m compileall`, strict audit/hash checks, intentional commit, and push to `codex/iotj-final-classification-le1`.

---

### Task 1: C0 adaptation-context behavioral contract

**Files:**
- Create: `gaps_flower/final_adaptation_context.py`
- Modify: `gaps_flower/strategy.py`
- Test: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Write failing behavioral tests**

Add tests that exercise real serialization/deserialization with hand-built tensors and assert:

- round25 context contains the ordered source checkpoint fingerprint, semantic prototypes, client class-phase prototypes/counts/IDs/residuals, aggregation weights, and loss-input availability;
- serialization is deterministic and rejects NaN/Inf, empty client payloads, inconsistent client lengths, non-round25 endpoints, and fingerprint mismatch;
- client identities remain in canonical numeric order;
- a nonzero A4 loss whose input availability differs from the audited C0-A baseline makes the final-adaptation preflight fail closed (including preserving baseline-unavailable terms as unavailable).

Run:

`python -m pytest tests/test_iotj_canonical_regression_reconstruction.py -q --basetemp=.tmp_pytest_crrq_c0_red`

Expected: FAIL because the context module and capture path do not exist.

**Step 2: Implement the minimal context module**

Provide typed functions:

- `build_final_adaptation_context(...) -> dict[str, Any]`
- `write_final_adaptation_context(path, payload) -> Path`
- `load_final_adaptation_context(path, checkpoint_path) -> dict[str, Any]`
- `validate_a4_context_loss_inputs(payload, configured_weights, expected_context_availability) -> dict[str, Any]`

Use ordered state-content fingerprint for checkpoint equality and file SHA256 only for provenance. Store tensors as JSON numeric lists with explicit dtype/shape metadata; restore them as float32 tensors. The strategy captures the exact final-round payload after semantic-prototype update and before any target adaptation. No target arrays or labels enter this artifact.

**Step 3: Re-run tests and refactor**

Run the same pytest command and verify PASS. Then run existing strategy/ablation tests to catch mutation of the interleaved path.

---

### Task 2: C0 final-only A4 adaptation parity API

**Files:**
- Modify: `gaps_flower/domain_adaptation.py`
- Modify: `gaps_flower/strategy.py`
- Modify: `gaps_flower/posthoc_commissioning.py`
- Create: `scripts/run_iotj_canonical_regression_reconstruction_c0.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Write failing parity tests**

Tests construct a tiny real model/loaders plus a complete context and assert that interleaved and final-only invocation builders produce identical A4 hyperparameters, loss-input availability, client payloads, optimizer/lr, and 100-step fixed endpoint. The only declared difference must be lifecycle timing. Also assert three target branches reload the same original round25 ordered state rather than adapting sequentially.

Expected mutation caught: empty prototypes/residuals or carrying C3-adapted weights into C4 makes a test fail.

**Step 2: Refactor one shared A4 invocation**

Extract a shared, side-effect-free invocation builder used by `GapsStrategy._run_domain_adapt` and the final-only runner. Extend the post-hoc function with an explicit required context path for this study; keep the legacy G1 behavior separate and clearly non-canonical. Do not change loss equations or registered coefficients.

**Step 3: Add fixed-endpoint outputs**

The C0 script supports `--stage freeze`, `--stage run-source`, `--stage adapt-targets`, and `--stage evaluate` and emits:

- `C0_PRE_EXECUTION_AUDIT.json`
- `C0_SOURCE_ROUND25_CONTEXT.json`
- independent C3/C4/C5 adapted checkpoints and diagnostics
- `C0_CLASSIFICATION_COMPARISON.csv`
- `C0_LOSS_ACTIVITY.csv`
- `C0_DECISION.json`
- `C0_EXPERIMENT_AUDIT.md`
- `C0_SHA256_INDEX.json`
- per-stage completion markers.

Target test is opened only by `evaluate`, after all three adaptations complete.

---

### Task 3: C0 three-machine controller and formal execution

**Files:**
- Modify: `scripts/run_iotj_canonical_regression_reconstruction_c0.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`
- Modify: `docs/experiments/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/EXPERIMENT_REGISTRY.csv`

**Step 1: Test command construction and resume safety**

Assert literal protocol values: 25 rounds, LE1, batch32, seed42, Adam 5e-4, `ce_stats`, GAPS aggregation, A4 prototype/stat payloads enabled, target adaptation disabled during source FL, target paths absent from every source command, and round25-only completion. Test that completed endpoints are skipped and any partial directory fails closed.

**Step 2: Run pre-execution gates**

Run targeted pytest, compileall, dataset identity/hash checks, remote canonical-v1 parity checks, target-label/test API audit, and command-lock audit. Write `C0_PRE_EXECUTION_AUDIT.json` only if all gates pass.

**Step 3: Commit/push executable freeze**

Commit code/tests/plan/audit inputs and push before remote execution so all machines run the same commit.

**Step 4: Execute and monitor C0**

Launch one source FL trajectory. Preserve round logs, endpoint locks, stderr, and remote process state. After round25, copy the exact checkpoint/context, run three independent 100-step final adaptations, then evaluate C3/C4/C5 once.

**Step 5: Decide C0**

For every target calculate `Macro-F1_final_only - Macro-F1_interleaved`. Final-only timing is supported only when absolute drop is no greater than 0.005 on all three targets and all loss-activity parity gates pass. Otherwise retain interleaved A4. Commit/push C0 evidence before R0.

---

### Task 4: R0 canonical 83D/H1 feature-space contract

**Files:**
- Create: `gaps_flower/canonical_quantitative_features.py`
- Create: `scripts/run_iotj_canonical_regression_reconstruction_r0.py`
- Test: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Write failing feature-provenance tests**

Use a literal 50x8 window fixture. Assert that 83D and H1 are computed from the supplied window, include the source filename/window identity, record `(50, 8)` and 5 Hz, and refuse 100x8 input, legacy cache provenance, missing canonical preprocessing hash, or mixed source/target extractor versions.

**Step 2: Implement canonical extraction/cache metadata**

Wrap the existing extractor without changing its formulas. H1 slope/absdiff fields are labeled fixed-5-Hz discrete descriptors. Create content-addressed caches only under the study output root. No legacy cache is read.

**Step 3: Audit full dataset extraction**

Recompute C1/C2 and C3/C4/C5 train/calibration/test features from canonical windows. Emit `R0_FEATURE_PROVENANCE.csv`, schema manifests, cache hashes, and per-client shape/finite-value audits.

---

### Task 5: R0 sufficient-statistic reconstruction and exact recovery

**Files:**
- Create: `gaps_flower/canonical_fedridge.py`
- Modify: `scripts/run_iotj_canonical_regression_reconstruction_r0.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Write failing numerical tests**

Against hand-computed small matrices assert population-variance scaling, 1e-9 scale floor, unregularized intercept, additive sufficient statistics, source-calibration alpha selection over `[0, .01, .1, 1, 10, 100, 1000]`, pseudoinverse solution, and train+cal refit. Tests independently check scaler, coefficient, and predictions.

**Step 2: Implement canonical FedRidge**

Keep alpha selection source-only and group audit metadata explicit. Reject target/test inputs in the selection API.

**Step 3: Run exact-recovery gate**

Compare centralized and federated reconstruction using tolerances: scaler `1e-10`, coefficient `1e-8`, prediction `1e-6 ppm`. If any fails, stop the study; no practical fallback. Emit `R0_EXACT_RECOVERY.json`, source metrics, and hashes. Commit/push R0 before R1.

---

### Task 6: R1 routed regression and grouped uncertainty

**Files:**
- Create: `gaps_flower/canonical_regression_evaluation.py`
- Create: `scripts/run_iotj_canonical_regression_reconstruction_r1.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Write failing evaluation tests**

With literal predictions assert S_ALL, S_CC, Oracle_ALL, and Oracle_CC routing, per-gas/per-concentration summaries, RMSE/MAE/NRMSE_range/R2/Bias, high-concentration slices, and the C5 Methane 225 ppm repeat1 identity. Assert target Ridge alpha is selected only by 5-fold raw-filename-grouped CV inside calibration.

**Step 2: Implement the evaluation core**

Evaluate source-only FedRidge, target 83D, and R84 using the frozen C0 classifier selected by the C0 decision. Produce row-level prediction artifacts before summaries and bind each to checkpoint/feature/test-manifest hashes.

**Step 3: Implement paired grouped bootstrap**

Use 5000 replicates, seed42, resampling unit `target|raw_filename`, and target-stratified pooled resampling. Emit paired deltas/CIs for all registered comparisons.

**Step 4: Decide and freeze R1**

Apply the preregistered supported/device-dependent/not-supported rules and >5% device-collapse flag. Commit/push R1. Open R2 only when the R1 decision meets its trigger.

---

### Task 7: Conditional R2 residual/shrinkage repair

**Files:**
- Create: `scripts/run_iotj_canonical_regression_reconstruction_r2.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Test the gate before any model code**

Assert R2 refuses to run unless R1 reports pooled benefit plus device/gas-specific negative transfer. Clean R84 support skips R2; global non-support freezes 83D.

**Step 2: Implement only registered repairs**

Evaluate residual transfer and shrinkage beta `[0, .25, .5, .75, 1]` using calibration-only selection and fixed test evaluation. No other feature/model/alpha search is permitted.

**Step 3: Freeze the regression endpoint**

Emit the R2 decision or explicit skip artifact, then commit/push before Q0.

---

### Task 8: Q0 frozen-QC necessity evaluation

**Files:**
- Create: `gaps_flower/canonical_qc_evaluation.py`
- Create: `scripts/run_iotj_canonical_regression_reconstruction_q0.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Write failing QC tests**

Assert exact coverage grid 0.50-1.00 by 0.01, full/random/confidence/regression-only routes, 5000 random references seed42, primary C5 and secondary target-stratified pooled scoring, and fail-closed handling of unavailable canonical equal-mean inputs. Verify frozen thresholds cannot be recalculated from test data.

**Step 2: Implement uncertainty and QC scoring**

Regression uncertainty is calibration 5-fold raw-filename-grouped model prediction dispersion normalized by routed gas range. If all canonical inputs for historical equal-mean QC are available, replay the exact formula; otherwise emit `Q4_CANONICAL_INPUTS_UNAVAILABLE` without substitution.

**Step 3: Decide QC necessity**

Emit coverage-RMSE/NRMSE curves, AURC summaries, random-reference distribution, necessity decision, hashes, and audit. Commit/push Q0. Open Q1 only when the frozen trigger is met.

---

### Task 9: Conditional Q1 combined uncertainty QC

**Files:**
- Create: `scripts/run_iotj_canonical_regression_reconstruction_q1.py`
- Modify: `tests/test_iotj_canonical_regression_reconstruction.py`

**Step 1: Test the gate and group isolation**

Assert Q1 cannot open without Q0 insufficiency plus useful regression uncertainty. Verify calibration groups never cross conformal folds and test rows never enter fitting/normalization/weighting.

**Step 2: Implement the single registered method**

Construct group-aware absolute-residual intervals, transform confidence/width by calibration empirical CDFs, and combine by equal mean. Do not search weights.

**Step 3: Apply the frozen decision**

Support Q1 only with at least 5% NRMSE-AURC improvement on both C5 and pooled. Otherwise retain frozen Q0 QC. Emit evidence or explicit skip, then commit/push.

---

### Task 10: Final claim/evidence closure

**Files:**
- Create: `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/FINAL_RESULT_MASTER_TABLE.csv`
- Create: `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/FINAL_CLAIM_EVIDENCE_MATRIX.md`
- Create: `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/FINAL_EXPERIMENT_STATUS.md`
- Create: `results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/FINAL_SUBMISSION_AUDIT.md`
- Modify: `docs/experiments/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/EXPERIMENT_REGISTRY.csv`

**Step 1: Reconcile the registry**

Every registered row must end as completed, conditionally skipped with a machine-readable reason, or failed closed. Link commands, data manifests, checkpoints, predictions, summaries, and SHA256 indexes.

**Step 2: Run complete verification**

Run all study tests, affected existing tests, `python -m compileall gaps_flower scripts tests`, hash-index verification, checkpoint ordered-state fingerprints, prediction/test-manifest hashes, leakage audit, and a clean intended-diff review.

**Step 3: Write scientific conclusion without overclaiming**

State C0 timing decision, canonical regression support status, device/gas limitations, QC necessity, and any conditional method decision. Claims must use only audited evidence and must not imply 5 Hz/10 Hz feature equivalence.

**Step 4: Final commit and push**

Stage only intended compact evidence and source/test/docs changes; preserve unrelated watcher logs and temporary directories. Push the final branch and stop all algorithm execution.
