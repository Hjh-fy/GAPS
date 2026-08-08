# Canonical v1 Final Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze `dataset/iotj_canonical_v1/`, then run one from-scratch canonical GAPS classification/adaptation/R84/QC pipeline and publish audited final evidence.

**Architecture:** A deterministic dataset builder materializes HZ5_MEAN_W10S windows and independent per-client/class/concentration splits. A fail-closed preflight validates hashes, identities, roles, labels, alignment, and finite values before a separate formal runner may launch training. Classification, regression, QC/engineering, and evidence finalization consume immutable manifests and write only to `results/iotj_canonical_v1/`.

**Tech Stack:** Python 3, NumPy/Pandas/PyTorch, Flower/GAPS existing runners, pytest, SHA256 manifests, Markdown/CSV/JSON evidence.

## Global Constraints

- Canonical preprocessing is exactly 5 Hz mean physical-time bins, raw-observation mean G0 over 20–50 s, short-gap-only, 60–170 s crop, 10 s window, 5 s stride, 50 points.
- Source clients are C1/C2; target clients are C3/C4/C5 with 0% train, 20% calibration, 80% sealed test.
- Each target client/class/concentration uses an independent deterministic RNG stream; client order cannot change identities.
- C5 Methane 225 ppm repeat 1 remains included with acquisition-quality metadata.
- Formal classification is 25 rounds, local epochs 1, seed 42, frozen GAPS architecture/optimizer/aggregation/semantic/server adaptation protocol.
- Historical checkpoints cannot be reused; all checkpoint-reuse flags must be false.
- Target test cannot select preprocessing, model, endpoint, Ridge alpha, QC threshold, or checkpoint.
- R84_FED_H1, Ridge alpha candidates, internal calibration split, QC policy, and model structures remain frozen.
- Existing datasets/results/checkpoints are read-only. New assets write only to `dataset/iotj_canonical_v1/`, `docs/experiments/iotj_canonical_v1/`, and `results/iotj_canonical_v1/`.

---

### Task 1: Canonical dataset builder and independent split

**Files:**
- Create: `tools/build_iotj_canonical_v1.py`
- Create: `tests/test_iotj_canonical_v1_dataset.py`
- Create: `docs/experiments/iotj_canonical_v1/EXPERIMENT_PLAN.md`
- Create: `docs/experiments/iotj_canonical_v1/EXPERIMENT_MATRIX.csv`
- Create: `docs/experiments/iotj_canonical_v1/ABLATION_PLAN.md`
- Reuse: `tools/preprocessor_canonical_candidate.py`

**Interfaces:**
- Consumes: raw files under `dataset/data1`, frozen candidate manifest from `results/iotj_canonical_preprocessing_selection_20260808/`.
- Produces: `build_canonical_dataset(raw_root: Path, output: Path, seed: int = 42) -> dict`, canonical `.npy` arrays, metadata, and required manifests.

- [ ] **Step 1: Write failing split invariance and reuse-flag tests**

```python
def test_split_is_client_order_invariant():
    a = assign_target_roles(identities, [3, 4, 5], seed=42)
    b = assign_target_roles(identities, [5, 3, 4], seed=42)
    assert identities_by_client(a) == identities_by_client(b)

def test_checkpoint_reuse_is_forbidden():
    manifest = canonical_preprocessing_manifest()
    assert manifest["reuse_historical_checkpoint"] is False
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_dataset.py`
Expected: FAIL because the canonical builder and split API do not exist.

- [ ] **Step 3: Implement deterministic physical identities and HZ5 windows**

Use SHA256-derived RNG seeds from `seed|client|class|concentration`; write `canonical_preprocessing_manifest.json`, `raw_file_manifest.csv`, `raw_sha256.json`, `processing_manifest.csv`, `window_identity_manifest.csv`, `split_manifest.csv`, and `dataset_sha256.json`. Materialize per-client source train/validation/test and target calibration/test arrays without historical checkpoint assets.

- [ ] **Step 4: Run GREEN tests and a miniature rebuild parity test**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_dataset.py`
Expected: PASS; same physical identities and SHA256 across client ordering.

- [ ] **Step 5: Commit dataset builder**

```powershell
git add tools/build_iotj_canonical_v1.py tests/test_iotj_canonical_v1_dataset.py docs/experiments/iotj_canonical_v1
git commit -m "feat: add canonical v1 dataset builder"
```

### Task 2: Preflight and reproducible freeze

**Files:**
- Create: `tools/preflight_iotj_canonical_v1.py`
- Create: `tests/test_iotj_canonical_v1_preflight.py`
- Create: `docs/experiments/iotj_canonical_v1/CANONICAL_DATASET_FREEZE.md`
- Create: `docs/experiments/iotj_canonical_v1/CANONICAL_PREPROCESSING_FREEZE.md`
- Create: `docs/experiments/iotj_canonical_v1/CANONICAL_SPLIT_FREEZE.md`

**Interfaces:**
- Consumes: canonical dataset directory and manifests from Task 1.
- Produces: `run_preflight(dataset_root: Path) -> dict`, `results/iotj_canonical_v1/DATASET_PREFLIGHT.md`, and `preflight.json` with `status=PASS` only after all hard gates.

- [ ] **Step 1: Write failing overlap, NaN, alignment, and hash tests**

```python
def test_preflight_rejects_target_overlap(tmp_path):
    dataset = fixture_dataset(tmp_path, overlap=True)
    with pytest.raises(RuntimeError, match="calibration/test overlap"):
        run_preflight(dataset)

def test_preflight_rejects_nonfinite_features(tmp_path):
    dataset = fixture_dataset(tmp_path, feature_value=np.nan)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        run_preflight(dataset)
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_preflight.py`
Expected: FAIL because preflight gates do not exist.

- [ ] **Step 3: Implement fail-closed checks**

Check raw completeness and hashes; parameter freeze; label/metadata/feature row alignment; finite arrays; source/target role rules; unique physical identities; no target overlap; class×concentration coverage; actual C3/C4/C5 counts; repeat-1 retention; and full dataset SHA rebuild parity.

- [ ] **Step 4: Run GREEN tests and full dataset build/preflight**

Run: `python tools/build_iotj_canonical_v1.py --output dataset/iotj_canonical_v1`
Run: `python tools/preflight_iotj_canonical_v1.py --dataset dataset/iotj_canonical_v1 --output results/iotj_canonical_v1`
Expected: PASS before any training process starts.

- [ ] **Step 5: Commit frozen dataset manifests and preflight code**

```powershell
git add tools/preflight_iotj_canonical_v1.py tests/test_iotj_canonical_v1_preflight.py docs/experiments/iotj_canonical_v1
git add -f dataset/iotj_canonical_v1 results/iotj_canonical_v1/DATASET_PREFLIGHT.md results/iotj_canonical_v1/preflight.json
git commit -m "data: freeze canonical v1 dataset"
```

### Task 3: From-scratch formal classification and calibration-only adaptation

**Files:**
- Create: `tools/run_iotj_canonical_v1_classification.py`
- Create: `tests/test_iotj_canonical_v1_classification_protocol.py`
- Reuse: `scripts/run_iotj_final_classification_le1.py`
- Reuse: `scripts/evaluate_iotj_final_classification_le1.py`
- Reuse: `gaps_flower/server_app.py`, `gaps_flower/client_app.py`, `gaps_flower/strategy.py`

**Interfaces:**
- Consumes: passing preflight, canonical split arrays, frozen GAPS runtime.
- Produces: `results/iotj_canonical_v1/classification/`, pre/post-adaptation checkpoints, hashes, configs, identities, confusion CSV, and `classification_summary.csv`.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_formal_run_is_from_scratch():
    cfg = canonical_classification_config()
    assert cfg.rounds == 25 and cfg.local_epochs == 1 and cfg.seed == 42
    assert cfg.reuse_checkpoint is False

def test_target_test_is_not_an_adaptation_input():
    cfg = canonical_classification_config()
    assert cfg.adaptation_splits == ("calibration",)
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_classification_protocol.py`
Expected: FAIL because canonical run config/guard does not exist.

- [ ] **Step 3: Implement wrapper and guards**

Require preflight PASS, an empty formal run directory, explicit `reuse_checkpoint=False`, and canonical data paths. Configure C1/C2 source clients, C3/C4/C5 calibration adaptation, 25 rounds, local epochs 1, seed 42, frozen optimizer/profile/DA, and pre/post checkpoint SHA256 logging.

- [ ] **Step 4: Run GREEN protocol tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_classification_protocol.py tests/test_iotj_final_classification_protocol.py`
Expected: PASS without launching the long run.

- [ ] **Step 5: Execute one formal classification/adaptation run**

Run: `python tools/run_iotj_canonical_v1_classification.py --dataset dataset/iotj_canonical_v1 --output results/iotj_canonical_v1/classification --device cuda`
Expected: 25/25 rounds, no historical checkpoint load, sealed test evaluated only before/after fixed adaptation endpoint.

- [ ] **Step 6: Commit runner and immutable run evidence**

```powershell
git add tools/run_iotj_canonical_v1_classification.py tests/test_iotj_canonical_v1_classification_protocol.py
git add -f results/iotj_canonical_v1/classification results/iotj_canonical_v1/classification_summary.csv
git commit -m "exp: run canonical v1 classification"
```

### Task 4: Frozen R84 regression and quality slices

**Files:**
- Create: `tools/run_iotj_canonical_v1_r84.py`
- Create: `tests/test_iotj_canonical_v1_r84.py`
- Reuse: `scripts/run_gaps_cross_target_r84_full.py`
- Reuse: `run_regression_head_ablation.py`

**Interfaces:**
- Consumes: canonical adapted classification checkpoint, calibration identities, canonical windows/quality metadata, fixed Federated-H1.
- Produces: `results/iotj_canonical_v1/r84/`, `regression_summary.csv`, `per_client_summary.csv`, `per_gas_summary.csv`, `per_concentration_summary.csv`, `c5_methane_summary.csv`, and `quality_stratified_summary.csv`.

- [ ] **Step 1: Write failing R84 contract tests**

```python
def test_r84_contract_is_frozen():
    cfg = canonical_r84_config()
    assert cfg.feature_profile == "R84_FED_H1"
    assert tuple(cfg.alpha_grid) == (0.0, .01, .1, 1.0, 10.0, 100.0, 1000.0)
    assert cfg.target_test_selection is False
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_r84.py`
Expected: FAIL because canonical R84 guard/output schema does not exist.

- [ ] **Step 3: Implement fixed R84 evaluation**

Fit/select only from target calibration internal splits; evaluate target test once. Persist S_ALL, S_CC, oracle-route metrics with N/RMSE/MAE/Bias/R2/NRMSE_range across ALL/client/gas/concentration. Persist C5 Methane 25–250 ppm and separate 225-ppm repeat rows. Create Q0–Q3 read-only strata from empty-bin ratio, max missing run, and observed ratio without deleting samples.

- [ ] **Step 4: Run GREEN tests and R84 job**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_r84.py tests/test_gaps_cross_target_r84_full.py`
Run: `python tools/run_iotj_canonical_v1_r84.py --dataset dataset/iotj_canonical_v1 --classification results/iotj_canonical_v1/classification --output results/iotj_canonical_v1/r84`
Expected: all required scopes/slices populated; repeat 1 retained.

- [ ] **Step 5: Commit R84 evidence**

```powershell
git add tools/run_iotj_canonical_v1_r84.py tests/test_iotj_canonical_v1_r84.py
git add -f results/iotj_canonical_v1/r84 results/iotj_canonical_v1/*summary.csv
git commit -m "exp: evaluate canonical v1 R84"
```

### Task 5: Frozen QC and engineering measurement

**Files:**
- Create: `tools/run_iotj_canonical_v1_postrun.py`
- Create: `tests/test_iotj_canonical_v1_postrun.py`
- Reuse: `scripts/finalize_iotj_a4_qc.py`
- Reuse: `scripts/evaluate_iotj_runtime_v5_qc.py`

**Interfaces:**
- Consumes: raw canonical prediction records and frozen QC manifest/policy.
- Produces: `qc_summary.csv`, `engineering_metrics.csv`, pipeline latency/memory records, and the three-way preprocessing diagnostic table.

- [ ] **Step 1: Write failing QC/communication semantics tests**

```python
def test_qc_thresholds_are_frozen():
    cfg = canonical_qc_config()
    assert cfg.search_enabled is False

def test_input_reduction_does_not_change_parameter_payload_claim():
    row = engineering_claims(points_old=100, points_new=50, parameter_bytes=1234)
    assert row["temporal_input_reduction"] == .5
    assert row["parameter_communication_reduction"] == 0.0
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_postrun.py`
Expected: FAIL because canonical QC/engineering contracts do not exist.

- [ ] **Step 3: Implement frozen postrun**

Apply the frozen QC policy to canonical predictions only; report accepted/accepted+review/reject, coverage, RMSE, and MAE. Benchmark preprocessing, classifier, R84, full pipeline, peak memory, input bytes, and calibration time. Explicitly keep parameter payload unchanged while reporting 50% temporal-input reduction. Add a diagnostic-only Legacy/current interpolation/canonical table with provenance caveat.

- [ ] **Step 4: Run GREEN tests and postrun**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_postrun.py tests/test_runtime_v5_qc_policy.py`
Run: `python tools/run_iotj_canonical_v1_postrun.py --results results/iotj_canonical_v1`
Expected: canonical QC and engineering CSVs generated without threshold search.

- [ ] **Step 5: Commit postrun evidence**

```powershell
git add tools/run_iotj_canonical_v1_postrun.py tests/test_iotj_canonical_v1_postrun.py
git add -f results/iotj_canonical_v1/qc_summary.csv results/iotj_canonical_v1/engineering_metrics.csv
git commit -m "exp: close canonical v1 QC and engineering"
```

### Task 6: Final evidence audit, reproducibility, and publication

**Files:**
- Create: `tools/finalize_iotj_canonical_v1.py`
- Create: `tests/test_iotj_canonical_v1_finalization.py`
- Create: `results/iotj_canonical_v1/FINAL_CANONICAL_SUMMARY.md`
- Create: `results/iotj_canonical_v1/reproducibility_manifest.json`

**Interfaces:**
- Consumes: immutable dataset, classification, R84, QC, and engineering evidence.
- Produces: final 12-question summary, SHA256 closure, and audit verdict.

- [ ] **Step 1: Write failing final-evidence completeness tests**

```python
def test_final_summary_requires_all_twelve_answers(tmp_path):
    with pytest.raises(RuntimeError, match="missing canonical evidence"):
        finalize(tmp_path)

def test_reproducibility_manifest_rejects_hash_drift(tmp_path):
    bundle = complete_fixture(tmp_path)
    tamper(bundle / "classification_summary.csv")
    with pytest.raises(RuntimeError, match="SHA256"):
        verify_reproducibility(bundle)
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest -q tests/test_iotj_canonical_v1_finalization.py`
Expected: FAIL because final evidence gate does not exist.

- [ ] **Step 3: Implement finalizer and experiment audit**

Fail closed on missing metrics, mixed dataset hashes, checkpoint reuse, split overlap, target-test selection, QC search, or hash drift. Write the 12 required answers at the top of `FINAL_CANONICAL_SUMMARY.md`, label any blocker, and preserve diagnostics as non-algorithmic evidence.

- [ ] **Step 4: Run full verification**

Run: `python -m compileall tools gaps_flower scripts`
Run: `python -m pytest -q tests/test_iotj_canonical_v1_*.py`
Run: `python tools/finalize_iotj_canonical_v1.py --dataset dataset/iotj_canonical_v1 --results results/iotj_canonical_v1`
Run: `python tools/preflight_iotj_canonical_v1.py --dataset dataset/iotj_canonical_v1 --verify-only`
Expected: all tests pass, dataset SHA rebuild is reproducible, and final audit has no unreported blocker.

- [ ] **Step 5: Commit and push**

```powershell
git add tools/finalize_iotj_canonical_v1.py tests/test_iotj_canonical_v1_finalization.py
git add -f results/iotj_canonical_v1 docs/experiments/iotj_canonical_v1 dataset/iotj_canonical_v1
git commit -m "exp: finalize canonical v1 evidence"
git push origin codex/iotj-final-classification-le1
```
