# GAPS Post-hoc DG/SSDA Method Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate a canonical-v1 lifecycle consisting of immutable C1/C2 source FL, one-time C5 commissioning, source-only prototype DG, and C5 5%-labeled/15%-unlabeled SSDA without target-test-driven tuning.

**Architecture:** Gate 0 freezes immutable provenance and information access. Gate 1 adds a Flower-free post-hoc API and evaluates three target-personalization methods only after all step-100 endpoints are locked. Gate 2 reuses the existing source-only Flower prototype exchange with no C5 access. Gate 3 starts all methods from the same Gate-1 source endpoint, masks unlabeled labels at the dataset/API boundary, selects the bounded SSDA configuration only by deterministic labeled-calibration folds, and opens the target test once after final endpoints are locked.

**Tech Stack:** Python 3, PyTorch, Flower, NumPy, pandas/CSV, pytest, SHA-256 provenance.

## Global Constraints

- Dataset and preprocessing: `canonical-v1`, `HZ5_MEAN_W10S`, input `50x8`.
- Source clients: C1 and C2 only; target: C5 only.
- Source FL: 25 rounds, local epochs 1, batch size 32, seed 42, canonical backbone.
- C5 calibration: frozen 320-window pool; target test identities remain frozen.
- No target-test tuning, preprocessing changes, split changes, anomaly removal, new baseline expansion, C3/C4 expansion, or result-driven algorithm search.
- Each Gate gets its own verification, evidence hashes, commit, and push.
- G2/G3 run sequentially after G1 analysis. G2 stops prototype-DG expansion if unsupported. G3 keeps its fixed bounded selection rule and does not expand tuning.
- Do not enter G4 or G5 automatically.

---

### Task 1: Gate 0 provenance and protocol freeze

**Files:**
- Create: `docs/experiments/iotj_canonical_v1_final/POSTHOC_DG_SSDA_PROTOCOL_AUDIT.md`
- Create: `docs/experiments/iotj_canonical_v1_final/method_redesign/EXPERIMENT_REGISTRY.csv`
- Create: `docs/experiments/iotj_canonical_v1_final/method_redesign/EXPERIMENT_PLAN.md`
- Create: `docs/experiments/iotj_canonical_v1_final/method_redesign/EXPERIMENT_MATRIX.csv`

**Interfaces:**
- Consumes: canonical dataset hash manifest, source FedAvg locked run spec, round-25/latest checkpoints, existing A0T/A4 configs.
- Produces: immutable checkpoint/data/test identities and method information-access contracts consumed by all later Gates.

- [ ] Verify whole-file SHA-256 and ordered state-content fingerprint for both round-25 and latest source checkpoints.
- [ ] Verify the source training commands name only client 1 and client 2 and declare target X/Y unavailable.
- [ ] Audit response phase as acquisition-time metadata; if not observable, freeze class-only semantics for G3.
- [ ] Freeze all Gate experiment IDs and unique result destinations before code execution.
- [ ] Record any unavailable interleaved-only A4 state as an interpretation limitation, never synthesize it.

### Task 2: Gate 1 post-hoc API with test-first safety gates

**Files:**
- Create: `gaps_flower/posthoc_commissioning.py`
- Create: `scripts/run_iotj_posthoc_commissioning_g1.py`
- Create: `tests/test_iotj_posthoc_commissioning.py`

**Interfaces:**
- Consumes: `source_checkpoint: Path`, target calibration manifest and array paths, source calibration arrays for A4 source losses, fixed `steps=100`, `lr=5e-4`, `batch_size=32`, `seed=42`.
- Produces: one personalized checkpoint plus completion manifest per method; the API never starts Flower or opens test arrays.

- [ ] Write failing tests proving target-test identities are rejected from adaptation manifests, each method reloads the same source content fingerprint, the API exposes no Flower process path, and target-head trains only `feat_proj` plus `classifier`.
- [ ] Run the tests and confirm failures are caused by the missing API.
- [ ] Implement A0T-full as all-parameter Adam CE adaptation for exactly 100 steps.
- [ ] Implement A4 using the registered coefficients and existing server adaptation implementation; record activity for losses whose interleaved client statistics are unavailable.
- [ ] Implement Target-head by freezing TCN and attention and training `feat_proj` plus `classifier` only.
- [ ] Save step diagnostics, trainable/total parameter counts, wall-clock, peak RSS, checkpoint bytes, and relative parameter displacement.
- [ ] Run focused tests to green.

### Task 3: Gate 1 locked endpoints, one-time evaluation, and decision

**Files:**
- Create under: `results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc/`
- Create: `POSTHOC_COMMISSIONING_COMPARISON.csv`
- Create: `POSTHOC_COMMISSIONING_PER_CLASS.csv`
- Create: `POSTHOC_G1_ANALYSIS.md`
- Create: `EXPERIMENT_AUDIT.md`
- Create: `sha256_index.json`

**Interfaces:**
- Consumes: three step-100 completion locks plus frozen source checkpoint.
- Produces: C1/C2/C1+C2/C5 predictions and metrics, retention deltas, historical interleaved comparison, and G1 decision tokens.

- [ ] Run adaptation-only for A0T-full, A4, and Target-head; fail if any output destination exists.
- [ ] Verify all three completion locks and endpoint hashes before loading any test split.
- [ ] Evaluate the four checkpoints on C1, C2, merged C1+C2, and C5.
- [ ] Compute Accuracy, Macro-F1, NLL, ECE, per-class Recall/F1, retention deltas, and historical interleaved gaps.
- [ ] Emit exactly one lifecycle verdict, one A4 verdict, one Target-head verdict, and the interleaved-dependency risk flag when applicable.
- [ ] Run pytest, compileall, hash audit, commit, and push.

### Task 4: Gate 2 source-only GAPS-DG-P

**Files:**
- Modify only if tests require a missing safety hook: `gaps_flower/task.py`, `gaps_flower/strategy.py`, `client.py`
- Create: `scripts/run_iotj_source_dg_g2.py`
- Create: `tests/test_iotj_source_dg_g2.py`
- Create under: `results/iotj_canonical_v1_method_redesign_20260811/gate2_source_dg/`

**Interfaces:**
- Consumes: C1/C2 training arrays only, canonical backbone, registered `LAMBDA_ALIGN=0.05`, existing class-phase prototype upload/weighted aggregation, 25-round endpoint.
- Produces: GAPS-DG-P checkpoint, round diagnostics, semantic prototypes, C1/C2/C1+C2/C5 zero-shot predictions, representation analysis.

- [ ] Write failing tests proving no target directory/manifest is accepted, round 1 is CE-only, round 2 onward receives persisted global prototypes, and client alignment uses the registered normalized class-phase prototype distance.
- [ ] Verify source phase derives only from acquisition-time metadata; otherwise lock class-only fallback before running.
- [ ] Freeze the single GAPS-DG-P run: strategy GAPS with selective aggregation, replay, server DA, and target calibration all disabled; profile `align_only`; prototype EMA remains registered at 0.8.
- [ ] Run exactly one seed-42 25-round C1/C2 source FL endpoint.
- [ ] After endpoint lock, evaluate FedAvg and GAPS-DG-P on C1/C2/C1+C2/C5 once.
- [ ] Compute inter-client prototype distance, within-class C1-C2 feature distance, and between-class margin.
- [ ] Emit `SOURCE_DG_SUPPORTED` or `SOURCE_DG_NOT_SUPPORTED`; do not search lambda.
- [ ] Run verification, commit, and push.

### Task 5: Gate 3 masked-label C5 SSDA

**Files:**
- Create: `gaps_flower/ssda.py`
- Create: `scripts/run_iotj_c5_ssda_g3.py`
- Create: `tests/test_iotj_c5_ssda_g3.py`
- Create under: `results/iotj_canonical_v1_method_redesign_20260811/gate3_ssda/`

**Interfaces:**
- Consumes: Gate-1 source checkpoint; 80 labeled identities (2/stratum); disjoint 240 unlabeled identities (6/stratum); frozen source prototypes; fixed test identities.
- Produces: A0T-5L, MME-5L15U, and GAPS-SSDA-5L15U endpoints plus selection/audit diagnostics.

- [x] Write failing tests proving the unlabeled training batch type has no `y_true` field, labeled/unlabeled/test identity sets are pairwise disjoint, and evaluation labels cannot reach training or selection functions.
- [x] Freeze deterministic two-fold labeled-calibration selection and the maximum six-item grid `tau in {0.90,0.95}`, `lambda_u in {0.25,0.5,1.0}`; fix EMA alpha 0.99 and prototype weight 0.05.
- [x] Audit MME feasibility and implement either exact MME or explicitly named MME-compatible head with identical source checkpoint, identities, backbone, and adaptation budget.
- [x] Run bounded selection using only the labeled folds; lock the selected configuration before final training.
- [x] Independently reload the same source checkpoint for all three final methods and train on the complete 80L/240U calibration allocation.
- [x] Verify all endpoint hashes, then open target test once and evaluate Accuracy, Macro-F1, NLL, ECE, per-class Precision/Recall/F1.
- [x] Compute pseudo-label acceptance, hidden-label accuracy only post hoc, per-class coverage, and confidence distribution.
- [x] Emit the G3 decision without expanding search; run verification, commit, and push.

### Task 6: Cross-Gate scientific handoff and stop

**Files:**
- Create: `docs/experiments/iotj_canonical_v1_final/method_redesign/GAPS_METHOD_REDESIGN_DECISION.md`
- Update: `PROJECT_STATUS.md`, `NEXT_ACTIONS.md`

- [x] Select the permitted Story A/B/C/D only from audited G1/G2/G3 results.
- [x] Preserve the strict non-overlap limitation and avoid experiment-independent generalization claims.
- [x] Record G4/G5 as not started and stop without launching them.
