# IoT-J Final Classification LE1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, preflight-audit, freeze, execute, and analyze the approved seed42 IoT-J final classification experiment matrix without target-test leakage or post-start protocol changes.

**Architecture:** Add focused protocol, SCAFFOLD, canonical-UDA, activity-audit, orchestration, evaluation, and strict-audit modules around the existing Flower client/server. Existing FedAvg/FedProx/GAPS paths remain behavior-compatible; new functionality is opt-in through explicit CLI/config fields. Formal execution is staged and fail-closed: immutable input import and tests precede a pre-run freeze commit, and sealed target tests are unlocked only for final fixed-endpoint evaluation.

**Tech Stack:** Python 3, PyTorch, Flower, NumPy, pandas, scikit-learn, matplotlib, pytest, PowerShell/SSH orchestration, Git.

## Global Constraints

- Branch: `codex/iotj-final-classification-le1`; base audit commit: `11cfbba`.
- Source clients C1+C2; targets C3/C4/C5; rounds25; LE1; batch32; seed42 only.
- FedAvg, FedProx, GAPS: Adam lr5e-4. FedProx mu0.01. SCAFFOLD: canonical SGD lr5e-4, no momentum or scheduler.
- SCAFFOLD SGD lr5e-4 is a preregistered fixed value, not an optimizer-equivalent fairness match to Adam lr5e-4.
- E2: target x only, 100 steps, Adam lr5e-4, coefficient0.5, no conditional losses, no target class/phase/concentration, no search.
- Full GAPS: method-specific target calibration x/class/phase access; concentration unavailable; target CE weight0; selective warm-up rounds1-5 FedAvg and round6 onward selective; min scale0.3.
- Target test labels/features are sealed during train/adapt/select/stop phases. Any access outside final fixed-endpoint evaluation is a hard failure.
- Ordered state-content fingerprint establishes checkpoint equality. Whole-file SHA-256 is provenance only.
- No formal training starts until tests, compileall, source-only SCAFFOLD numerical gate, and pre-execution strict audit pass and a `pre-run freeze` commit exists.
- After formal training starts: no hyperparameter changes, matrix changes, checkpoint selection, or target-test-driven decisions.

---

### Task 1: Apply the four protocol amendments

**Files:**
- Modify: `docs/plans/2026-08-04-iotj-final-classification-le1-design.md`
- Modify: `docs/experiments/iotj_final_classification_le1_20260804/EXPERIMENT_PLAN.md`
- Modify: `docs/experiments/iotj_final_classification_le1_20260804/ABLATION_PLAN.md`
- Modify: `docs/experiments/iotj_final_classification_le1_20260804/PRE_EXECUTION_AUDIT.md`
- Create: `docs/experiments/iotj_final_classification_le1_20260804/TARGET_INFORMATION_POLICY.md`

**Interfaces:**
- Consumes: approved design commit `ec9cb22` and the four user amendments.
- Produces: exact policy text used by manifests, tests, and audit.

- [ ] **Step 1: Patch the written protocol**

Record exact SCAFFOLD source-only numerical gates, method-specific target fields, A0-A6 activity columns, round1-5/round6 warm-up semantics, ordered fingerprint rule, E0 sensor-space statistics, and E1 source-target F1 gaps.

- [ ] **Step 2: Add the target information table**

Use columns `method`, `split`, `x`, `class`, `phase`, `concentration`, `purpose`, `selection_allowed`. Register E2 x-only and GAPS calibration x/class/phase with concentration false. Register target test as final-evaluation only.

- [ ] **Step 3: Verify documentation consistency**

Run:

```powershell
rg -n "round1-5|round6|ordered state-content|source-target F1 gap|ablation_loss_activity|concentration" docs/plans/2026-08-04-iotj-final-classification-le1-design.md docs/experiments/iotj_final_classification_le1_20260804
```

Expected: every amendment appears in the design and experiment protocol; no statement says all target calibration labels are globally forbidden.

### Task 2: Ordered checkpoint fingerprint and immutable input gate

**Files:**
- Create: `gaps_flower/state_fingerprint.py`
- Create: `tests/test_iotj_final_classification_protocol.py`
- Create: `scripts/prepare_iotj_final_classification_inputs.py`

**Interfaces:**
- Consumes: `Mapping[str, torch.Tensor]` or ordered `(keys, arrays)`.
- Produces: `ordered_state_content_fingerprint(state) -> str`, `ordered_array_content_fingerprint(keys, arrays) -> str`, and an immutable input manifest.

- [ ] **Step 1: Write failing fingerprint tests**

Tests must prove identical ordered tensor content fingerprints match across separate `torch.save` containers, key-order changes produce a different fingerprint or a fail-closed error, dtype/shape/content changes differ, and whole-file SHA is stored only as provenance.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_iotj_final_classification_protocol.py -k fingerprint -q
```

Expected: import/function failures because the module does not exist.

- [ ] **Step 3: Implement minimal fingerprint module**

Hash, in order, UTF-8 key, NUL, dtype, NUL, int64 shape bytes, and contiguous tensor bytes. Reject duplicate keys, non-finite floating tensors, and length mismatch.

- [ ] **Step 4: Implement input importer**

Copy the read-only P0A round25 checkpoint into `results/iotj_final_classification_le1_20260804/inputs/`, preserve source path and whole-file SHA, load both copies, compute ordered state fingerprints, and hard fail if they differ. Never modify the source result.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest tests/test_iotj_final_classification_protocol.py -k fingerprint -q
```

Expected: all fingerprint tests pass.

### Task 3: Canonical SCAFFOLD transport, local update, and server strategy

**Files:**
- Create: `gaps_flower/scaffold.py`
- Modify: `gaps_flower/task.py`
- Modify: `gaps_flower/client_app.py`
- Modify: `gaps_flower/server_app.py`
- Modify: `gaps_flower/strategy.py`
- Create: `tests/test_scaffold_canonical.py`

**Interfaces:**
- Produces `pack_control_variates(keys, arrays) -> bytes`, `unpack_control_variates(payload, expected_keys, reference)`, `scaffold_train_one_round(model, loader, client_control, server_control, lr, device, local_epochs) -> ScaffoldLocalResult`, and `ScaffoldStrategy`.
- `ScaffoldLocalResult` contains model arrays, updated client control, control delta, K, CE trajectory, accuracy, grad norms, parameter norms, and finite-status fields.

- [ ] **Step 1: Write the five required failing tests**

Create exactly named tests:

```python
def test_scaffold_uses_sgd(): ...
def test_scaffold_gradient_contains_control_variate_correction(): ...
def test_scaffold_client_control_variate_persists(): ...
def test_scaffold_server_control_variate_updates(): ...
def test_scaffold_no_adam_state_present(): ...
```

The gradient test uses a one-parameter model and verifies the numerical update equals `w - eta*(grad+c-c_i)`. Persistence test performs two calls on one client instance. Server update checks `c_new=c+sum(delta_c_i)/N`.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_scaffold_canonical.py -q
```

Expected: missing SCAFFOLD symbols.

- [ ] **Step 3: Implement ndarray transport and local update**

Use `io.BytesIO` plus `numpy.savez` with deterministic ordered keys. Initialize `c_i` once per long-lived client, never per round. For K actual minibatch updates, use SGD with no momentum and apply `param.grad += c - c_i` before `optimizer.step()`.

- [ ] **Step 4: Implement canonical control update**

After local training:

```python
c_i_new = c_i - c + (w_global - w_local) / (K * eta)
delta_c_i = c_i_new - c_i
```

Keep control variates only for trainable named parameters; model aggregation continues over the complete state dict.

- [ ] **Step 5: Implement Flower integration**

Add `--optimizer {adam,scaffold_sgd}` to the client and `--strategy scaffold` to the server. Send server `c` in fit config bytes; return `delta_c_i` in metrics bytes. `ScaffoldStrategy` aggregates model deltas with server step size1.0 and sample weights, updates `c` by the N=2 client mean, persists per-round state and fingerprints, and rejects missing/duplicate client updates.

- [ ] **Step 6: Run GREEN and regression tests**

```powershell
python -m pytest tests/test_scaffold_canonical.py tests/test_flower_classification_contract.py tests/test_confirmation_flower_integration.py -q
```

Expected: all pass.

### Task 4: Source-only SCAFFOLD numerical validity gate

**Files:**
- Create: `gaps_flower/source_numerical_gate.py`
- Modify: `scripts/run_iotj_final_classification_le1.py`
- Modify: `tests/test_scaffold_canonical.py`

**Interfaces:**
- Produces `evaluate_source_gate(client_diagnostics, source_eval, class_prior) -> SourceGateVerdict` and `scaffold_source_validity_gate.json`.

- [ ] **Step 1: Write failing gate tests**

Test hard failure for NaN/Inf, zero/non-finite gradients, non-finite/exploding parameter norm, absent CE decrease, and source accuracy not exceeding the majority-class prior. Test a finite improving trajectory passes. Verify the gate never accepts target paths or target metrics.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_scaffold_canonical.py -k source_gate -q
```

- [ ] **Step 3: Implement fixed gate**

Use only C1/C2 train and source test. Predetermine: all values finite; `0 < max_grad_norm < 1e4`; `0 < max_parameter_norm < 1e4`; final-quarter mean CE < first-quarter mean CE; combined source accuracy > observed majority-class prior. The one-round dry-run is discarded and formal SCAFFOLD reloads the original initialization. No retry with another lr.

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests/test_scaffold_canonical.py -k source_gate -q
```

### Task 5: Method-specific target information gate and canonical E2

**Files:**
- Create: `gaps_flower/target_information.py`
- Create: `gaps_flower/canonical_uda.py`
- Create: `tests/test_canonical_uda_and_target_gate.py`
- Create: `scripts/run_iotj_canonical_uda_references.py`

**Interfaces:**
- Produces `TargetInformationPolicy`, `TargetAccessLedger`, `load_target_calibration_x`, `unlock_target_test_for_final_evaluation`, and `run_canonical_uda(method, source_model, source_loader, target_x_loader, ...)`.

- [ ] **Step 1: Write failing policy and hard-fail tests**

Test E2 policies allow only calibration x; GAPS allows calibration x/class/phase and denies concentration; any target test feature/label access during train/adapt/select/stop raises `TargetTestLeakageError`; final-evaluation access requires a one-purpose unlock token and writes the ledger.

- [ ] **Step 2: Write failing E2 tests**

Test function signature has no target-label parameter, target loader yields tensor only, CORAL uses global covariance, MMD uses global MMD², DANN uses unconditional GRL+BCE, all forbidden conditional/pseudo/target-CE terms are unavailable, and all nine runs require the same ordered source fingerprint.

- [ ] **Step 3: Run RED**

```powershell
python -m pytest tests/test_canonical_uda_and_target_gate.py -q
```

- [ ] **Step 4: Implement policies and ledger**

Persist fields `timestamp`, `method`, `stage`, `split`, `x`, `class`, `phase`, `concentration`, `purpose`, `allowed`, and `reason`. Refuse target test during any non-final stage.

- [ ] **Step 5: Implement canonical UDA methods**

Each method independently deep-copies/reloads the P0A state, trains source CE plus exactly one registered alignment loss for 100 steps, uses Adam5e-4 and coefficient0.5, and saves step diagnostics. DANN uses a standard binary discriminator/GRL objective rather than the GAPS Wasserstein conditional path.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest tests/test_canonical_uda_and_target_gate.py -q
```

### Task 6: A0-A6 loss-activity audit

**Files:**
- Create: `gaps_flower/loss_activity.py`
- Modify: `client.py`
- Modify: `gaps_flower/task.py`
- Modify: `gaps_flower/domain_adaptation.py`
- Modify: `gaps_flower/strategy.py`
- Create: `tests/test_ablation_loss_activity.py`

**Interfaces:**
- Produces `LossActivityAccumulator.record(loss_name, configured_weight, input_available, raw_loss, active, inactive_reason)` and CSV rows with the exact requested columns plus experiment/variant/scope/round.

- [ ] **Step 1: Write failing accumulator tests**

Verify active steps count only genuinely computed terms, raw and weighted means are separate, zero-weight terms are inactive with `configured_weight_zero`, unavailable inputs are inactive with a specific reason, and schema contains `loss_name,configured_weight,input_available,active_steps,mean_raw_loss,mean_weighted_loss,inactive_reason`.

- [ ] **Step 2: Write failing A4/A5 tests**

Construct DA summaries showing which server terms activate with `ce_stats` versus `proto_replay`. Assert A4 client semantic/replay terms inactive; server global CORAL/MMD/adversarial can activate from x; class/stage terms require calibration class/phase; prototype/residual terms require uploaded client statistics. Assert A5 reports its actual active set rather than assuming all configured terms ran.

- [ ] **Step 3: Run RED**

```powershell
python -m pytest tests/test_ablation_loss_activity.py -q
```

- [ ] **Step 4: Implement client and server instrumentation**

Record existing forward-pass losses without additional optimization passes. Extend server DA diagnostics with per-term configured weight, input-availability predicate, active-step count, mean raw/weighted loss, and inactive reason. Do not change loss values or optimizer updates.

- [ ] **Step 5: Aggregate A0-A6 CSV**

Create `ablation_loss_activity.csv` from client round diagnostics and server DA diagnostics. Fail audit if a variant's declared active set disagrees with observed activity.

- [ ] **Step 6: Run GREEN and regression tests**

```powershell
python -m pytest tests/test_ablation_loss_activity.py tests/test_flower_da_v3_corrections.py tests/test_flower_classification_contract.py -q
```

### Task 7: Selective aggregation warm-up boundary

**Files:**
- Modify: `gaps_flower/strategy.py`
- Modify: `gaps_flower/server_app.py`
- Create: `tests/test_selective_warmup_boundary.py`

**Interfaces:**
- Produces `selective_phase_for_round(server_round, warmup_rounds) -> "fedavg_warmup" | "selective"` and complete per-round aggregation logs.

- [ ] **Step 1: Write failing boundary tests**

Verify warmup5 returns FedAvg for rounds1-5 and selective for round6+. Verify logs contain base weight, similarity, scale, final weight, active flag, and phase even when semantic prototypes are unavailable.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_selective_warmup_boundary.py -q
```

- [ ] **Step 3: Implement explicit boundary helper and logging**

Replace implicit comparison text with the helper. For rounds1-5 final weight equals base weight. From round6, selective is required when registered semantic inputs exist; missing inputs are recorded and become an audit failure for Full GAPS/A3/A6.

- [ ] **Step 4: Run GREEN**

```powershell
python -m pytest tests/test_selective_warmup_boundary.py tests/test_confirmation_flower_integration.py -q
```

### Task 8: E0 sensor shifts and E1 source-target gaps

**Files:**
- Create: `scripts/evaluate_iotj_final_classification_le1.py`
- Create: `tests/test_iotj_final_classification_evaluation.py`

**Interfaces:**
- Produces sensor-space CSVs, classification metrics, per-window predictions, `source_target_f1_gap`, discrepancy summaries, cost summaries, and figure source data.

- [ ] **Step 1: Write failing E0 tests**

On synthetic 3D windows, verify per-channel mean/std/median/IQR/quantile shift, standardized mean difference, and covariance diagnostics are computed per target and never require learned features.

- [ ] **Step 2: Write failing E1 gap tests**

Verify for each method and target `source_target_f1_gap = source_macro_f1 - target_macro_f1`, with source macro-F1 computed once from the combined registered source-test population.

- [ ] **Step 3: Run RED**

```powershell
python -m pytest tests/test_iotj_final_classification_evaluation.py -q
```

- [ ] **Step 4: Implement evaluation and outputs**

Use fixed labels/order, accuracy, macro-F1, per-class scores, NLL, 15-bin ECE, confusion matrices, per-window probabilities, source retention, per-channel sensor shifts, embedding discrepancies, training/commissioning time, and communication bytes. Target test loading must go through the final-evaluation unlock gate.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest tests/test_iotj_final_classification_evaluation.py -q
```

### Task 9: Resumable suite runner and strict pre-run audit

**Files:**
- Create: `scripts/run_iotj_final_classification_le1.py`
- Create: `scripts/audit_iotj_final_classification_le1.py`
- Create: `tests/test_iotj_final_classification_runner.py`
- Create: `docs/experiments/iotj_final_classification_le1_20260804/PROTOCOL.md`
- Create: `docs/experiments/iotj_final_classification_le1_20260804/LABEL_ACCESS_AUDIT.md`

**Interfaces:**
- Runner subcommands: `prepare`, `source-gate`, `preflight`, `run --experiment-id`, `evaluate`, `analyze`, `audit`.
- Audit returns nonzero unless all blocking findings pass.

- [ ] **Step 1: Write failing runner tests**

Verify 21 registered configs, 10 new full runs, 9 E2 branches, exact commands, optimizer fields, warmup5, no target-test paths in train/adapt commands, immutable completion markers, resume behavior, and matrix freeze hash.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_iotj_final_classification_runner.py -q
```

- [ ] **Step 3: Implement command generation and stage locks**

Generate three-host Flower commands from the existing distributed runner conventions. Create `formal_training_started.lock` containing the matrix/protocol hash; after it exists, reject changes to registered configs. Store stdout/stderr, process IDs, round checkpoints, history, timing, communication and completion metadata.

- [ ] **Step 4: Implement strict audit**

Pre-run audit checks immutable P0A import, ordered fingerprint, dataset counts/fingerprints, SCAFFOLD tests/source gate, target policies, E2 x-only, warm-up boundary, A0-A6 activity schema, optimizer disclosure, absence of target-test train paths, and output destinations. Post-run audit adds completeness, fixed endpoints, predictions, metrics, figures and SHA index.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest tests/test_iotj_final_classification_runner.py -q
```

### Task 10: Full verification and pre-run freeze commit

**Files:**
- Modify: `docs/experiments/iotj_final_classification_le1_20260804/PRE_EXECUTION_AUDIT.md`
- Create: `docs/experiments/iotj_final_classification_le1_20260804/protocol_manifest.json`
- Create: `docs/experiments/iotj_final_classification_le1_20260804/sha256_index.json`

**Interfaces:**
- Produces an auditable pre-run PASS and immutable Git commit.

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests/test_scaffold_canonical.py tests/test_canonical_uda_and_target_gate.py tests/test_ablation_loss_activity.py tests/test_selective_warmup_boundary.py tests/test_iotj_final_classification_evaluation.py tests/test_iotj_final_classification_runner.py -q
```

- [ ] **Step 2: Run relevant regressions**

```powershell
python -m pytest tests/test_flower_classification_contract.py tests/test_flower_da_v3_corrections.py tests/test_confirmation_flower_integration.py tests/test_iotj_r1_m2_baselines.py tests/test_iotj_p0i_adaptation_timing.py -q
```

- [ ] **Step 3: Run compileall**

```powershell
python -m compileall -q gaps_flower scripts tests
```

- [ ] **Step 4: Run immutable input preparation and source gate**

```powershell
python -m scripts.run_iotj_final_classification_le1 prepare
python -m scripts.run_iotj_final_classification_le1 source-gate
```

Expected: P0A ordered fingerprint match and source-only SCAFFOLD numerical gate PASS. Failure stops without lr search.

- [ ] **Step 5: Run strict preflight**

```powershell
python -m scripts.audit_iotj_final_classification_le1 --stage pre-run --strict
```

Expected: PASS with zero blocking findings.

- [ ] **Step 6: Create pre-run freeze commit**

Stage only this task's code/docs/tests/manifests and commit with message `freeze: lock IoT-J final classification pre-run protocol`. Record the commit hash in `protocol_manifest.json` without amending execution semantics.

### Task 11: Execute the frozen formal matrix

**Files:**
- Create under: `results/iotj_final_classification_le1_20260804/`

**Interfaces:**
- Consumes: pre-run freeze commit and approved 21-row matrix.
- Produces: immutable run directories and completion markers.

- [ ] **Step 1: Start formal execution lock**

```powershell
python -m scripts.run_iotj_final_classification_le1 start-formal
```

- [ ] **Step 2: Run E0 and E1**

Run E0 diagnostics, reused FedAvg evaluation staging, new FedProx and SCAFFOLD through the existing three-machine topology. Never retry with changed config.

- [ ] **Step 3: Run E2**

Execute CORAL/MMD/DANN × C3/C4/C5 from the same ordered P0A fingerprint. Complete all 100 steps before any target test evaluation.

- [ ] **Step 4: Run E3**

Execute Full GAPS C3/C4/C5 sequentially with warmup rounds1-5 and selective round6+.

- [ ] **Step 5: Run E4**

Reuse A0/A6 only after exact audit and execute A1-A5. Save `ablation_loss_activity.csv` inputs for every variant.

- [ ] **Step 6: Open sealed tests for final evaluation**

Only after each method's fixed endpoint completion marker, invoke the one-purpose evaluation unlock and write the target-access ledger.

### Task 12: Analyze, audit, commit, and push

**Files:**
- Create/modify under: `results/iotj_final_classification_le1_20260804/`
- Create/modify under: `docs/experiments/iotj_final_classification_le1_20260804/`

**Interfaces:**
- Produces all required CSV/Parquet/JSON/Markdown/Fig.1-9 artifacts and final audited Evidence package.

- [ ] **Step 1: Build unified tables and figures**

Generate the main comparison with optimizer fields, C5 hierarchy, loss activity, source-target F1 gaps, sensor/embedding shifts, source retention, costs, predictions, embeddings, and Fig.1-9 source data/plots.

- [ ] **Step 2: Write result analysis**

Answer the nine registered research questions without overstating seed42. Include the mandatory SCAFFOLD optimizer paragraph verbatim.

- [ ] **Step 3: Run strict post-run audit**

```powershell
python -m scripts.audit_iotj_final_classification_le1 --stage post-run --strict
python -m compileall -q gaps_flower scripts tests
python -m pytest tests/test_scaffold_canonical.py tests/test_canonical_uda_and_target_gate.py tests/test_ablation_loss_activity.py tests/test_selective_warmup_boundary.py tests/test_iotj_final_classification_evaluation.py tests/test_iotj_final_classification_runner.py -q
```

- [ ] **Step 4: Commit and push**

Commit code/results/audit artifacts on `codex/iotj-final-classification-le1` and push to `origin`. Stop; do not launch further optimization.
