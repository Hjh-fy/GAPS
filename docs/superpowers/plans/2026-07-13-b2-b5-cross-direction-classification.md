# B2/B5 Cross-Direction Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run paired B2/B5 classification experiments for C1-to-C5, C5-to-C1, and C4/C5-to-C1 on Alibaba Cloud ECS plus the physical Raspberry Pi/PC, then produce leakage-safe paired evidence for whether B5's extra alignment losses add stable value.

**Architecture:** Preserve the hard-guarded primary C1/C2-to-C5 generator and controller. Add a separate manifest-driven cross-direction generator and controller that reuse the existing Flower runtime and frozen B2/B5 definitions, then extend checkpoint evaluation to accept an explicit target client and add a paired comparison layer over saved per-window streams.

**Tech Stack:** Python 3, PyTorch, Flower, NumPy, SciPy, pytest, PowerShell, SSH/SCP, Alibaba Cloud ECS, physical Raspberry Pi, Windows PC.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-13-b2-b5-cross-direction-classification.md` exactly.
- Keep the primary `C1,C2 -> C5` generator/controller guards unchanged.
- Use only B2 and B5 with their frozen v3 weights and corrected objective modes.
- Run reportable training only on ECS plus physical Pi/PC clients; never use local simulation.
- Reuse the existing balanced datasets unless active-file validation fails.
- Execute seed-42 runs sequentially in the approved six-run order.
- Do not select methods, weights, checkpoints, or thresholds using target test labels.
- Update `docs/experiments/iotj_system_experiment_notebook.md` after implementation, preflight, each recovered run, evaluation, and decision gate.

---

### Task 1: Freeze Cross-Direction Manifests

**Files:**
- Create: `configs/iotj_b2_b5_cross_direction_20260713.json`
- Create: `scripts/generate_iotj_cross_direction_commands.py`
- Create: `tests/test_iotj_cross_direction_classification.py`

**Interfaces:**
- Produces `load_direction_specs(path: Path) -> tuple[DirectionSpec, ...]`
- Produces `build_run_manifest(direction: DirectionSpec, group_id: str, seed: int, repo_root: Path, results_root: str) -> dict[str, Any]`
- Produces `generate_manifests(...) -> list[dict[str, Any]]`
- Produces manifest `commands.clients` as a list of client ID, executor, command, and script filename records.

- [ ] **Step 1: Write failing direction and manifest tests**

```python
def test_frozen_directions_and_device_assignments():
    specs = load_direction_specs(CONFIG)
    assert [(s.direction_id, s.source_clients, s.target_client) for s in specs] == [
        ("F1_C1_TO_C5", (1,), 5),
        ("R1_C5_TO_C1", (5,), 1),
        ("R2_C45_TO_C1", (4, 5), 1),
    ]
    assert specs[1].executors == {5: "pi"}
    assert specs[2].executors == {4: "pi", 5: "pc"}

def test_b2_b5_manifest_keeps_frozen_losses_and_budget(tmp_path):
    b2 = build_run_manifest(DIRECTION, "B2", 42, tmp_path, RESULTS)
    b5 = build_run_manifest(DIRECTION, "B5", 42, tmp_path, RESULTS)
    assert b2["training"]["rounds"] == 25
    assert b2["server_adaptation"]["lambda_global_mmd"] == 0.5
    assert b2["server_adaptation"]["lambda_coral"] == 0.0
    assert b5["server_adaptation"]["lambda_coral"] == 0.5
    assert b5["server_adaptation"]["lambda_stage_mmd"] == 0.2
    assert b5["server_adaptation"]["lambda_adv"] == 0.5
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_iotj_cross_direction_classification.py -q --basetemp .tmp_pytest_iotj_cross_direction_manifest_red`

Expected: FAIL because the config and generator do not exist.

- [ ] **Step 3: Implement the frozen config, manifest builder, validation, and LF-only command files**

Validation must reject unknown clients, C3/C4 targets, unsupported groups, duplicate executors, target/source overlap, non-25-round schedules, altered B weights, missing arrays, incorrect counts, and a command whose client ID/data root differs from its manifest.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_iotj_cross_direction_classification.py tests/test_flower_classification_contract.py -q --basetemp .tmp_pytest_iotj_cross_direction_manifest_green`

Expected: PASS with the existing primary manifest tests unchanged.

- [ ] **Step 5: Commit the manifest implementation**

```powershell
git add configs/iotj_b2_b5_cross_direction_20260713.json scripts/generate_iotj_cross_direction_commands.py tests/test_iotj_cross_direction_classification.py
git commit -m "feat: add B2 B5 cross-direction manifests"
```

### Task 2: Add A Manifest-Driven Real-Topology Controller

**Files:**
- Create: `scripts/run_iotj_cross_direction_cloud_edge.py`
- Create: `tests/test_iotj_cross_direction_controller.py`

**Interfaces:**
- Consumes the Task 1 manifest schema.
- Produces `load_ordered_manifests(command_root: Path, seed: int) -> list[tuple[Path, dict[str, Any]]]`.
- Produces `active_executors(manifest) -> dict[str, tuple[int, ...]]`.
- Launches zero or one Pi source and zero or one PC source per approved manifest.

- [ ] **Step 1: Write failing order, topology, and refusal tests**

```python
def test_seed42_queue_uses_approved_order(command_root):
    manifests = load_ordered_manifests(command_root, 42)
    assert [(m["direction_id"], m["group_id"]) for _, m in manifests] == [
        ("F1_C1_TO_C5", "B2"), ("F1_C1_TO_C5", "B5"),
        ("R1_C5_TO_C1", "B2"), ("R1_C5_TO_C1", "B5"),
        ("R2_C45_TO_C1", "B2"), ("R2_C45_TO_C1", "B5"),
    ]

def test_controller_refuses_partial_remote_run():
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        assert_remote_run_is_fresh(running=False, rounds=3, has_files=True)
```

- [ ] **Step 2: Verify controller tests fail**

Run: `python -m pytest tests/test_iotj_cross_direction_controller.py -q --basetemp .tmp_pytest_iotj_cross_direction_controller_red`

Expected: FAIL because the controller module is absent.

- [ ] **Step 3: Implement manifest loading, code/data sync, preflight, tunnel lifecycle, dynamic client launch, progress polling, and artifact recovery**

The preflight checks exact active files on each machine, ECS idle state, Python imports, data provenance, TCP tunnels, Pi temperature/throttling, and output freshness. A completed run requires history, round-25 client stats, final checkpoints, and DA diagnostics before recovery.

- [ ] **Step 4: Verify controller and legacy contracts pass**

Run: `python -m pytest tests/test_iotj_cross_direction_controller.py tests/test_iotj_cloud_edge_controller.py -q --basetemp .tmp_pytest_iotj_cross_direction_controller_green`

Expected: PASS; the legacy controller remains hard-guarded to C1/C2-to-C5.

- [ ] **Step 5: Commit the controller**

```powershell
git add scripts/run_iotj_cross_direction_cloud_edge.py tests/test_iotj_cross_direction_controller.py
git commit -m "feat: run cross-direction classification on real edges"
```

### Task 3: Generalize Target Evaluation And Add Paired Statistics

**Files:**
- Modify: `scripts/summarize_iotj_classification_ablation.py`
- Create: `scripts/summarize_iotj_cross_direction_classification.py`
- Modify: `tests/test_iotj_classification_summary.py`
- Create: `tests/test_iotj_cross_direction_summary.py`

**Interfaces:**
- Changes `evaluate_checkpoint_stream(..., target_client: int)` while preserving target C5 as the CLI default.
- Produces paired direction rows with deltas, exact McNemar p-value, and fixed-seed bootstrap intervals.

- [ ] **Step 1: Write failing explicit-target and paired-comparison tests**

```python
def test_paired_comparison_uses_identical_row_keys():
    result = compare_streams(B2_ROWS, B5_ROWS, bootstrap_seed=20260713)
    assert result["N"] == len(B2_ROWS)
    assert result["accuracy_delta_pp"] == pytest.approx(25.0)
    assert 0.0 <= result["mcnemar_exact_p"] <= 1.0

def test_paired_comparison_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="row keys"):
        compare_streams(B2_ROWS, list(reversed(B5_ROWS)))
```

- [ ] **Step 2: Verify summary tests fail**

Run: `python -m pytest tests/test_iotj_classification_summary.py tests/test_iotj_cross_direction_summary.py -q --basetemp .tmp_pytest_iotj_cross_direction_summary_red`

Expected: FAIL because explicit targets and paired comparison are missing.

- [ ] **Step 3: Implement target-aware evaluation and deterministic paired statistics**

Persist calibration/test probabilities, predictions, row keys, metrics, confusion matrices, and comparison JSON/CSV. Use exact McNemar on discordant correctness counts and paired bootstrap resampling for accuracy and macro-F1 differences.

- [ ] **Step 4: Run summary tests and verify GREEN**

Run: `python -m pytest tests/test_iotj_classification_summary.py tests/test_iotj_cross_direction_summary.py -q --basetemp .tmp_pytest_iotj_cross_direction_summary_green`

Expected: PASS.

- [ ] **Step 5: Commit evaluation changes**

```powershell
git add scripts/summarize_iotj_classification_ablation.py scripts/summarize_iotj_cross_direction_classification.py tests/test_iotj_classification_summary.py tests/test_iotj_cross_direction_summary.py
git commit -m "feat: compare B2 B5 classification by direction"
```

### Task 4: Generate Commands And Preflight Three Machines

**Files:**
- Generate: `results/iotj_b2_b5_cross_direction_20260713_commands`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- Consumes Tasks 1-3.
- Produces exactly six seed-42 manifests and a successful dry-run/preflight record.

- [ ] **Step 1: Generate and audit the seed-42 queue**

Run: `python scripts/generate_iotj_cross_direction_commands.py --config configs/iotj_b2_b5_cross_direction_20260713.json --seed 42 --output-root results/iotj_b2_b5_cross_direction_20260713_commands --results-root results/iotj_b2_b5_cross_direction_20260713`

Expected: six manifests in the approved order with no local-simulation command.

- [ ] **Step 2: Dry-run controller resolution**

Run: `python scripts/run_iotj_cross_direction_cloud_edge.py --command-root results/iotj_b2_b5_cross_direction_20260713_commands --seed 42 --dry-run`

Expected: the exact six-run queue and approved Pi/PC mapping.

- [ ] **Step 3: Preflight and synchronize only missing active data**

Verify ECS `root@121.40.139.213`, Pi `gaps@192.168.31.184`, and the local PC. Upload only the runtime, manifests, metadata, and active client directories required by the six runs. Re-run hashes and sample counts after transfer.

- [ ] **Step 4: Record preflight evidence in the experiment notebook**

Record connectivity, code revision, data hashes/counts, free storage, ECS idle state, Pi `get_throttled`, and command-index hash.

### Task 5: Run And Recover The Six Seed-42 Experiments

**Files:**
- Generate: `results/iotj_b2_b5_cross_direction_20260713`
- Generate: `results/iotj_b2_b5_cross_direction_20260713_local_logs`
- Modify after every pair: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- Consumes the frozen queue from Task 4.
- Produces six complete round-25 runs.

- [ ] **Step 1: Run the F1 C1-to-C5 B2/B5 pair and review artifacts**

Launch the controller for the first pair only. Require 25 rounds, finite metrics/losses, expected client count one, final adapted checkpoint, empty fatal-error scan, and no Pi throttling before advancing.

- [ ] **Step 2: Run the R1 C5-to-C1 B2/B5 pair and review artifacts**

Use Pi C5 exactly. Apply the same completion gates and record target C1 counts 680/2680.

- [ ] **Step 3: Run the R2 C4/C5-to-C1 B2/B5 pair and review artifacts**

Use Pi C4 and PC C5. Require both client logs and two-client round statistics.

- [ ] **Step 4: Recover and hash all six immutable result directories**

Refuse to treat a partial directory as complete. Record wall time, checkpoint SHA-256, artifact counts, and Pi health for every run.

### Task 6: Evaluate Seed 42 And Make The Confirmation Decision

**Files:**
- Generate: `results/iotj_b2_b5_cross_direction_20260713_summary`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`
- Modify: `代码文件介绍.md`

**Interfaces:**
- Consumes six recovered checkpoints.
- Produces per-direction metrics, per-window streams, paired statistics, complexity/timing table, and a bounded interpretation.

- [ ] **Step 1: Evaluate each run against its frozen target**

Run the target-aware summary separately for C5 and C1 result roots. Assert N=1360 for F1 and N=2680 for R1/R2.

- [ ] **Step 2: Build paired B2-minus-B5 tables**

Generate accuracy, macro-F1, NLL, ECE, worst recall, confusion, McNemar, bootstrap intervals, wall time, and communication columns.

- [ ] **Step 3: Apply the predeclared 0.5-point decision rule**

Classify every direction as B2 non-inferior, inconclusive, or B5-favored. Do not convert seed-42 screening into a final superiority claim.

- [ ] **Step 4: Update documentation and commit the seed-42 evidence**

Document actual results, failures, limitations, artifact paths, and the seeds 43-46 confirmation queue. Update the code guide with the new entrypoints.

### Task 7: Run Paired Confirmation Seeds 43-46

**Files:**
- Extend: `results/iotj_b2_b5_cross_direction_20260713_commands`
- Extend: `results/iotj_b2_b5_cross_direction_20260713`
- Extend: `results/iotj_b2_b5_cross_direction_20260713_summary`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- Produces five paired seeds per method/direction and the final generalization claim.

- [ ] **Step 1: Generate seeds 43-46 without changing seed-42 manifests**

Require identical data hashes, method weights, budgets, target IDs, and physical device mapping.

- [ ] **Step 2: Execute each seed sequentially with pair-level review gates**

Do not parallelize reportable training on the shared topology.

- [ ] **Step 3: Aggregate paired seeds and freeze the final conclusion**

Report mean, sample standard deviation, paired seed differences, bootstrap intervals, and direction-specific exceptions. Promote B2 only if the approved non-inferiority rule is satisfied; otherwise retain B5 or report direction dependence.
