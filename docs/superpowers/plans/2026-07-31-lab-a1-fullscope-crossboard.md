# Lab A1 Full-Scope Cross-Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, deploy, execute, and audit six seed-42 laboratory three-gas cross-board runs that compare A1 full-response training with an A4 matched local-epoch control.

**Architecture:** Extend the existing time-purged dataset builder so primary train/test scope is independent from the always-available early60, stable360, and full420 diagnostic splits. Extend the three-node controller so the Raspberry Pi can run logical client P1 or P3 while server A remains Flower Server and Server DA. Generate content-addressed source and direction-specific datasets locally, synchronize immutable copies to the three machines, then run one fail-closed sequential queue.

**Tech Stack:** Python 3, NumPy, PyTorch, Flower, pytest, PowerShell, SSH/SCP, JSON/CSV manifests.

## Global Constraints

- Run exactly six configurations in the order frozen by `docs/superpowers/specs/2026-07-31-lab-a1-fullscope-crossboard-design.md`.
- Use seed 42, 25 federated rounds, 1 local epoch, 100 Server DA steps, batch size 32, `proto_replay`, `corrected_b2`, target CE weight 0, and fixed round 25.
- A1 full-response source train and target primary test contain 420 windows per board; target calibration contains 90 windows.
- Always materialize independent `early`=60, `stable`=360, and `full`=420 diagnostic splits without duplicated indices.
- Fit normalization only on the declared source train clients for each direction.
- P3→P1 runs logical client 3 on the Raspberry Pi; server A holds P1 target calibration/test; cloud B is idle.
- Never overwrite existing A4 datasets, checkpoints, results, or content-addressed runtimes.
- Target test data never participates in training, DA, tuning, or round selection.
- Treat all results as seed-42 descriptive evidence.

---

### Task 1: Correct primary and diagnostic split semantics

**Files:**
- Modify: `scripts/lab_three_gas_3class/build_allconcentration_timepurged_dataset.py`
- Modify: `tests/test_lab_three_gas_allconc_timepurged.py`

**Interfaces:**
- Produces: `EARLY_INDICES`, `STABLE_INDICES`, `FULL_INDICES` constants.
- Produces: `assemble_direction_records(records, direction, primary_indices)` with independent `target_primary`, `target_early`, `target_stable`, and `target_full` records.
- Preserves: `build_dataset(config, overwrite, main_min_offset_s, direction)`.

- [ ] **Step 1: Write failing split-contract tests**

Add tests asserting:

```python
assert EARLY_INDICES == (0, 1)
assert len(STABLE_INDICES) == 12
assert FULL_INDICES == EARLY_INDICES + STABLE_INDICES

full = assemble_direction_records(
    records, direction="P2_to_P3", primary_indices=FULL_INDICES
)
assert len(full["source_train"][2]) == 420
assert len(full["target_primary"]) == 420
assert len(full["target_early"]) == 60
assert len(full["target_stable"]) == 360
assert len(full["target_full"]) == 420
```

Also assert that no record key is duplicated inside `target_full` and that the A4 primary scope remains 360 when `primary_indices=STABLE_INDICES`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_lab_three_gas_allconc_timepurged.py -q --basetemp .tmp_pytest_lab_a1_split_red
```

Expected: failure because the current A1 `target_full` repeats early indices and `target_stable` follows the primary scope.

- [ ] **Step 3: Implement independent scopes and P3→P1 direction**

Use:

```python
EARLY_INDICES = (0, 1)
STABLE_INDICES = tuple(index for index in MAIN_INDICES if index not in EARLY_INDICES)
FULL_INDICES = EARLY_INDICES + STABLE_INDICES

DIRECTION_ROLES["P3_to_P1"] = ((3,), 1)
```

Change record assembly so source train and target primary use `primary_indices`; named diagnostic splits always use the three fixed constants. Save target `test` from `target_primary`, plus `early`, `stable`, and `full` named arrays and manifests. Record all four index lists and the primary scope in `fold_config.json`.

- [ ] **Step 4: Verify GREEN and legacy behavior**

Run:

```powershell
python -m pytest tests/test_lab_three_gas_allconc_timepurged.py -q --basetemp .tmp_pytest_lab_a1_split_green
```

Expected: all tests pass, including prior P2→P3/A4 direction tests.

- [ ] **Step 5: Commit Task 1**

```powershell
git add scripts/lab_three_gas_3class/build_allconcentration_timepurged_dataset.py tests/test_lab_three_gas_allconc_timepurged.py
git commit -m "feat(lab): add independent A1 response scopes"
```

---

### Task 2: Evaluate stable, early, and full named splits consistently

**Files:**
- Modify: `scripts/lab_three_gas_3class/evaluate_exposure_checkpoint.py`
- Modify: `scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py`
- Modify: `tests/test_lab_three_gas_crossboard_scopes.py`

**Interfaces:**
- Consumes: target `stable_*`, `early_*`, and `full_*` arrays from Task 1.
- Produces: scope mapping `stable360 -> stable`, `early60 -> early`, `full420 -> full`.

- [ ] **Step 1: Write a failing evaluator mapping test**

```python
assert SCOPE_SPLITS == {
    "stable360": "stable",
    "early60": "early",
    "full420": "full",
}
```

Add a loader test showing `make_named_loader(..., "stable", ...)` reads the stable arrays without normalization a second time.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_lab_three_gas_crossboard_scopes.py -q --basetemp .tmp_pytest_lab_a1_scope_red
```

Expected: failure because stable360 currently aliases `test` and `stable` is not an accepted named split.

- [ ] **Step 3: Implement the named stable split**

Allow `stable` in the evaluator CLI and route `stable`, `early`, and `full` through `make_named_loader`. Change `SCOPE_SPLITS["stable360"]` to `stable`.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/test_lab_three_gas_crossboard_scopes.py -q --basetemp .tmp_pytest_lab_a1_scope_green
```

- [ ] **Step 5: Commit Task 2**

```powershell
git add scripts/lab_three_gas_3class/evaluate_exposure_checkpoint.py scripts/lab_three_gas_3class/evaluate_crossboard_scopes.py tests/test_lab_three_gas_crossboard_scopes.py
git commit -m "fix(lab): evaluate named response scopes"
```

---

### Task 3: Support logical P3 on the Raspberry Pi

**Files:**
- Modify: `scripts/lab_three_gas_3class/run_lab_three_node_fold.ps1`
- Modify: `scripts/lab_three_gas_3class/validate_three_node_run.py`
- Modify: `tests/test_lab_three_gas_crossboard_controller.py`
- Modify: `tests/test_lab_three_gas_validate_three_node_run.py`

**Interfaces:**
- Produces: direction `P3_to_P1 -> sources [3], target 1`.
- Produces: controller contract fields `pi_client_id` and `launch_pi`.

- [ ] **Step 1: Write failing controller tests**

Add parameterized assertions:

```python
assert resolve_run_roles("P3_to_P1") == ([3], 1)
```

Run the PowerShell `-ContractOnly` mode and assert P3→P1 returns source `[3]`, target `1`, `launch_pi=true`, `pi_client_id=3`, and `launch_cloud_b=false`.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_lab_three_gas_crossboard_controller.py tests/test_lab_three_gas_validate_three_node_run.py -q --basetemp .tmp_pytest_lab_a1_controller_red
```

- [ ] **Step 3: Implement dynamic Pi identity**

Add `P3_to_P1` to both direction maps. Derive:

```powershell
$piClientId = if ($sourceClients -contains 3) { 3 } elseif ($sourceClients -contains 1) { 1 } else { $null }
$launchPi = $null -ne $piClientId
```

Use `$piClientId` in Pi preflight, launch, process lookup, and stop commands. Keep the Raspberry Pi Python path and runtime root unchanged. Continue using server A data for the target client.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/test_lab_three_gas_crossboard_controller.py tests/test_lab_three_gas_validate_three_node_run.py -q --basetemp .tmp_pytest_lab_a1_controller_green
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add scripts/lab_three_gas_3class/run_lab_three_node_fold.ps1 scripts/lab_three_gas_3class/validate_three_node_run.py tests/test_lab_three_gas_crossboard_controller.py tests/test_lab_three_gas_validate_three_node_run.py
git commit -m "feat(lab): run logical P3 on Raspberry Pi"
```

---

### Task 4: Create a fail-closed six-run queue

**Files:**
- Create: `scripts/lab_three_gas_3class/run_a1_full_crossboard_queue.ps1`
- Create: `tests/test_lab_three_gas_a1_full_queue.py`

**Interfaces:**
- Consumes: `run_lab_three_node_fold.ps1` and six immutable dataset names.
- Produces: one sequential queue with explicit `Rounds=25`, `LocalEpochs=1`, `DaSteps=100`, and `Seed=42`.

- [ ] **Step 1: Write a failing queue contract test**

Parse the PowerShell source and assert the ordered direction/protocol list is:

```python
[
    ("A1", "P2_to_P3"),
    ("A4", "P2_to_P3"),
    ("A1", "P1_to_P3"),
    ("A1", "P12_to_P3"),
    ("A1", "P2_to_P1"),
    ("A1", "P3_to_P1"),
]
```

Assert `25`, `1`, `100`, `42`, `last_round`, and fail-on-nonzero behavior are explicit.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_lab_three_gas_a1_full_queue.py -q --basetemp .tmp_pytest_lab_a1_queue_red
```

- [ ] **Step 3: Implement the queue**

Create six records with unique experiment IDs, dataset names, run labels, and result namespaces. For every record, run controller `-PreflightOnly`, then the formal run. Stop the queue immediately on a non-zero exit. Write queue start/end timestamps and completed experiment IDs to a new controller evidence directory.

- [ ] **Step 4: Verify GREEN and PowerShell parsing**

```powershell
python -m pytest tests/test_lab_three_gas_a1_full_queue.py -q --basetemp .tmp_pytest_lab_a1_queue_green
powershell -NoProfile -Command "[void][scriptblock]::Create((Get-Content -Raw scripts/lab_three_gas_3class/run_a1_full_crossboard_queue.ps1)); 'PARSE_OK'"
```

- [ ] **Step 5: Commit Task 4**

```powershell
git add scripts/lab_three_gas_3class/run_a1_full_crossboard_queue.ps1 tests/test_lab_three_gas_a1_full_queue.py
git commit -m "feat(lab): queue six A1 cross-board runs"
```

---

### Task 5: Build and validate five A1 datasets

**Files:**
- Create: five directories under `dataset/client_data_lab_3gas_a1_full_crossboard_<direction>_v1`
- Create: `results/lab_3gas_a1_full_crossboard_protocol_20260731/dataset_validation_summary.json`

**Interfaces:**
- Consumes: Task 1 builder and `dataset/Dataset_self` raw data.
- Produces: immutable direction datasets for P2→P3, P1→P3, P1+P2→P3, P2→P1, and P3→P1.

- [ ] **Step 1: Run the complete local test suite for modified components**

```powershell
python -m pytest tests/test_lab_three_gas_allconc_timepurged.py tests/test_lab_three_gas_crossboard_scopes.py tests/test_lab_three_gas_crossboard_controller.py tests/test_lab_three_gas_validate_three_node_run.py tests/test_lab_three_gas_a1_full_queue.py -q --basetemp .tmp_pytest_lab_a1_release
```

- [ ] **Step 2: Build each A1 dataset into a new destination**

Invoke `build_allconcentration_timepurged_dataset.py` with `--main-min-offset-s 0`, the exact direction, and its new output root. Do not use `--overwrite` if the destination already contains files; choose an `_r2` destination after auditing the existing path.

- [ ] **Step 3: Validate dataset contracts**

For each dataset assert finite arrays, shape `[420,100,6]` for every source train and target test, `[90,100,6]` for target calibration, `[60,100,6]` early, `[360,100,6]` stable, and `[420,100,6]` full. Verify normalization-client identity and exact source/target roles from `fold_config.json` rather than directory names.

- [ ] **Step 4: Freeze hashes and validation summary**

Hash every manifest and array, write the new validation summary, and ensure no A4 path changed.

---

### Task 6: Create and synchronize an immutable runtime

**Files:**
- Create: new source archive and manifest under `results/lab_3gas_a1_full_crossboard_runtime_20260731_r1/`
- Create remotely: content-addressed runtime roots derived from the new archive SHA.

**Interfaces:**
- Consumes: committed source from Tasks 1–4.
- Produces: byte-identical runtime on server A, cloud B, and Raspberry Pi.

- [ ] **Step 1: Verify the committed source tree**

Run the targeted tests again from committed HEAD and record the exact commit SHA. Refuse to archive uncommitted task code.

- [ ] **Step 2: Build the source archive and manifest**

Use the existing laboratory runtime packaging convention. Include source code and tests; exclude datasets, checkpoints, logs, caches, and unrelated dirty files.

- [ ] **Step 3: Synchronize source and role-specific data**

Upload the source archive to all three machines. Upload P2 source datasets to cloud B; upload P1 and P3 source datasets required by each direction to the Raspberry Pi; upload all target calibration/test datasets to server A. It is acceptable for a machine to hold extra read-only direction data, but each run must load only the declared client directory.

- [ ] **Step 4: Verify remote hashes and paths**

On every machine compare the archive SHA and required dataset manifest hashes with the local manifest. Stop on any mismatch.

---

### Task 7: Preflight and execute the six-run overnight queue

**Files:**
- Create: `results/lab_3gas_a1_full_crossboard_seed42_20260731_controller/<run_id>/`
- Create remotely: `results/lab_3gas_a1_full_crossboard_seed42_20260731/<run_id>/`

**Interfaces:**
- Consumes: immutable runtime SHA and validated datasets.
- Produces: 25 base checkpoints, 25 adapted checkpoints, formal evaluation, and postflight audit per run.

- [ ] **Step 1: Run direction-specific contract-only checks**

Verify all six controller mappings. Specifically verify P3→P1 reports Pi logical client 3 and no cloud-B client.

- [ ] **Step 2: Run three-node preflight for every unique topology**

Preflight P2-only, Pi-P1-only, P1+P2, and Pi-P3-only topologies. Confirm Python environments, data roots, input dimension 6, free disk, no stale same-run processes, and Raspberry Pi temperature/throttling status.

- [ ] **Step 3: Launch the sequential queue**

Start one queue process with durable stdout/stderr logs. Do not launch six independent controllers in parallel. Persist the queue PID, source SHA, experiment list, and start time.

- [ ] **Step 4: Monitor fail-closed progress**

Check checkpoint count, last completed round, active process identity, log freshness, Raspberry Pi temperature, and throttling. If a run fails, preserve its directory and stop before the next run.

- [ ] **Step 5: Run scope evaluation and postflight audit**

For every fixed round-25 adapted checkpoint, evaluate named stable, early, and full splits. Confirm the formal primary scope is full420 for A1 and stable360 for the A4 control. Require audit status `valid`.

---

### Task 8: Summarize and audit the results

**Files:**
- Create: `scripts/lab_three_gas_3class/summarize_a1_full_crossboard_results.py`
- Create: `tests/test_lab_three_gas_a1_full_summary.py`
- Create: `results/lab_3gas_a1_full_crossboard_final_20260731_r1/combined_summary.json`
- Create: `results/lab_3gas_a1_full_crossboard_final_20260731_r1/combined_metrics.csv`
- Create: `docs/experiments/lab_3gas_a1_full_crossboard_final_analysis_20260731.zh.md`
- Create: `docs/experiments/lab_3gas_a1_full_crossboard_final_audit_20260731.zh.md`

**Interfaces:**
- Consumes: six valid postflight audits and scope summaries plus existing A1/A4 P2→P3 baselines.
- Produces: six-run tables and the P2→P3 2×2 time-range/local-epoch comparison.

- [ ] **Step 1: Write failing summary-contract tests**

Require six unique experiment IDs, fixed round 25, seed 42, exact direction roles, primary-scope identity, and metric equality between scope summaries and audits.

- [ ] **Step 2: Implement the minimal summarizer and verify GREEN**

Write new output only; refuse overwrite. Report correct/total, Accuracy, Macro-F1, exposure Accuracy, confusion matrices, unadapted/adapted deltas, and source/evaluator hashes.

- [ ] **Step 3: Produce the formal analysis and audit**

Separate confirmed values, descriptive comparisons, and unknowns. Explicitly retain single-seed, overlapping-window, nominal-boundary, all-concentration calibration, and unmatched P1+P2 budget limitations.

- [ ] **Step 4: Verify and publish lightweight evidence**

Run tests, compileall, JSON/CSV consistency checks, and `git diff --check`. Commit only code, tests, reports, manifests, audits, and lightweight JSON/CSV; exclude raw datasets, model checkpoints, logs, archives, and caches.
