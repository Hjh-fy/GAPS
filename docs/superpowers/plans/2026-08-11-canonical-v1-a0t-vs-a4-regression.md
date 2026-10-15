# Canonical-v1 A0T vs GAPS/A4 Regression Commissioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute an audited six-endpoint comparison that changes only the frozen A0T versus GAPS/A4 classifier checkpoint while holding canonical-v1 calibration/test data, R84_FED_H1, fixed Ridge alphas, metric definitions, and QC policy constant.

**Architecture:** A focused evaluator validates immutable inputs, fits gas-specific R84 coefficients with a pre-frozen alpha table, locks calibration before test access, and emits four regression scopes. A separate QC/analysis module consumes those fixed predictions, applies exact existing HC90/HC95 locks, produces the requested mechanism reports, and a strict auditor verifies hashes and the dual-gate conclusion.

**Tech Stack:** Python 3, NumPy, PyTorch, existing GAPS classifier evaluation helpers, serialized Ridge models, CSV/JSON/Markdown, pytest, SHA-256, Git.

## Global Constraints

- Dataset is exactly `dataset/iotj_canonical_v1`, aggregate SHA-256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Targets are exactly C3/C4/C5 with calibration/test counts 678/2677, 320/1360, and 320/1360.
- Classifiers are the six existing round25 target-adapted A0T/A4 checkpoints; no classifier training or adaptation is callable.
- R84 is exactly 83 sensor statistics plus frozen Federated-H1; H1 manifest SHA-256 is `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`.
- Ridge alpha is read from the frozen per-target/per-gas table; no grid evaluation, ranking, or search is permitted.
- QC uses the exact existing target-specific equal-mean normalization constants and HC90/HC95 threshold locks; no threshold refit is permitted.
- Target test is opened only after all calibration locks for the six endpoints are persisted and verified.
- Output is a new directory: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/`.
- Seed is 42; results are descriptive fixed-endpoint evidence, not stability evidence.
- Stop after this matrix; no new methods, seeds, budgets, targets, R84 variants, QC workpoints, or hyperparameter search.

---

### Task 1: Frozen protocol, registry, and input audit

**Files:**
- Create: `scripts/run_iotj_a0t_vs_a4_regression.py`
- Create: `tests/test_iotj_a0t_vs_a4_regression.py`
- Create after audit: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/PRE_RUN_FREEZE.json`
- Create after audit: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/experiment_registry.csv`

**Interfaces:**
- Produces: `EndpointSpec`, `endpoint_specs() -> tuple[EndpointSpec, ...]`, `frozen_alphas() -> dict[str, dict[int, float]]`, `audit_inputs(output: Path) -> dict[str, Any]`.
- Consumes: six classification run manifests and completion markers, canonical dataset hash index, H1 manifest, canonical A4 alpha-selection CSVs, and frozen QC threshold locks.

- [ ] **Step 1: Write failing identity and no-search tests**

```python
def test_exact_six_endpoints_and_checkpoint_is_only_method_factor():
    specs = endpoint_specs()
    assert [(s.method, s.target) for s in specs] == [
        ("A0T", "C3"), ("A0T", "C4"), ("A0T", "C5"),
        ("A4", "C3"), ("A4", "C4"), ("A4", "C5"),
    ]
    for target in ("C3", "C4", "C5"):
        a0t, a4 = [s for s in specs if s.target == target]
        assert a0t.held_constants == a4.held_constants
        assert a0t.checkpoint != a4.checkpoint

def test_frozen_alpha_table_is_shared_and_has_no_search_api():
    assert frozen_alphas()["C5"] == {0: 1.0, 1: 0.01, 2: 10.0, 3: 0.1}
    source = Path(MODULE.__file__).read_text(encoding="utf-8")
    assert "RIDGE_ALPHAS" not in source
    assert "best_alpha" not in source
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/test_iotj_a0t_vs_a4_regression.py -q`

Expected: import failure because `scripts/run_iotj_a0t_vs_a4_regression.py` does not exist.

- [ ] **Step 3: Implement immutable endpoint and alpha declarations**

```python
@dataclass(frozen=True)
class EndpointSpec:
    experiment_id: str
    method: str
    target: str
    checkpoint: Path
    checkpoint_sha256: str
    classification_manifest: Path
    completion_marker: Path
    held_constants: tuple[str, ...]

FROZEN_ALPHAS = {
    "C3": {0: 100.0, 1: 0.0, 2: 0.1, 3: 0.1},
    "C4": {0: 1.0, 1: 10.0, 2: 0.1, 3: 10.0},
    "C5": {0: 1.0, 1: 0.01, 2: 10.0, 3: 0.1},
}
```

`audit_inputs` must fail closed unless every manifest identifies the expected experiment, `formal_round == 25`, `target_test_opened == false`, the checkpoint SHA and ordered state fingerprint verify, dataset counts match, A4 historical alpha CSVs exactly equal `FROZEN_ALPHAS`, and all QC locks hash to their existing canonical files.

- [ ] **Step 4: Add audit failure tests**

Use temporary copied manifests to assert rejection of checkpoint SHA mismatch, wrong target, non-round25 provenance, opened test marker, alpha drift, missing QC lock, and changed dataset aggregate hash.

- [ ] **Step 5: Run Task 1 tests and commit**

Run: `python -m pytest tests/test_iotj_a0t_vs_a4_regression.py -q`

Expected: all Task 1 tests pass.

Commit: `git commit -m "feat: freeze A0T versus A4 regression protocol"`

---

### Task 2: Fixed-alpha R84 fitting and four-scope predictions

**Files:**
- Modify: `scripts/run_iotj_a0t_vs_a4_regression.py`
- Modify: `tests/test_iotj_a0t_vs_a4_regression.py`

**Interfaces:**
- Produces: `fit_fixed_alpha_models(target: str, oracle_rows: Sequence[Mapping[str, Any]]) -> dict[int, SerializedRidge]`.
- Produces: `apply_scope_models(rows, models, scope) -> list[dict[str, Any]]` for `S_ALL`, `S_CC`, `Oracle_ALL`, and `Oracle_CC`.
- Produces: `summarize_scope(rows, prediction_key="pred_ppm") -> dict[str, Any]` with `N`, `RMSE`, `MAE`, `NRMSE_range`, `R2`, and `Bias`.

- [ ] **Step 1: Write failing fixed-alpha and scope tests**

```python
def test_fixed_alpha_fit_uses_exact_value(monkeypatch):
    calls = []
    monkeypatch.setattr(MODULE, "fit_ridge", lambda rows, names, alpha: calls.append(alpha) or FakeRidge())
    fit_fixed_alpha_models("C5", synthetic_oracle_rows())
    assert calls == [1.0, 0.01, 10.0, 0.1]

def test_oracle_cc_uses_same_indices_as_s_cc():
    scopes = build_four_scopes(synthetic_deployment_rows(), synthetic_oracle_rows(), fake_models())
    assert [r["sample_index"] for r in scopes["Oracle_CC"]] == [
        r["sample_index"] for r in scopes["S_CC"]
    ]
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python -m pytest tests/test_iotj_a0t_vs_a4_regression.py -q -k "fixed_alpha or oracle_cc"`

Expected: missing functions.

- [ ] **Step 3: Implement calibration-only fixed-alpha fitting**

For each true class, build Oracle calibration R84 features, fit once on all class calibration rows with `FROZEN_ALPHAS[target][class_id]`, and serialize the model plus exact alpha. Do not create a train/validation split and do not evaluate any alternative alpha.

- [ ] **Step 4: Implement all four scopes**

`S_ALL` uses predicted route, routed H1, and predicted-route R84 model. `S_CC` filters `S_ALL` by route correctness. `Oracle_ALL` uses true route, true-class H1, and true-route model for every row. `Oracle_CC` filters `Oracle_ALL` to the exact `S_CC` sample-index set.

- [ ] **Step 5: Implement complete metrics and slices**

```python
def summarize_scope(rows):
    truth = np.asarray([float(r["true_ppm"]) for r in rows])
    pred = np.asarray([float(r["pred_ppm"]) for r in rows])
    error = pred - truth
    ranges = np.asarray([CLASS_RANGES[int(r["true_class"])] for r in rows])
    return {
        "N": len(rows),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "NRMSE_range": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "R2": r2_score_without_external_selection(truth, pred),
        "Bias": float(np.mean(error)),
    }
```

Add per-gas and per-concentration summaries and require a non-empty C5 Methane 225 ppm repeat1 slice.

- [ ] **Step 6: Test calibration-before-test access**

Inject read hooks and assert the six regression calibration locks exist and validate before the first test loader call. Assert test fields never enter fitting or lock construction.

- [ ] **Step 7: Run Task 2 tests and commit**

Run: `python -m pytest tests/test_iotj_a0t_vs_a4_regression.py -q`

Commit: `git commit -m "feat: add fixed-alpha four-scope R84 evaluation"`

---

### Task 3: Frozen equal-mean QC adapter

**Files:**
- Create: `scripts/evaluate_iotj_a0t_vs_a4_qc.py`
- Create: `tests/test_iotj_a0t_vs_a4_qc.py`

**Interfaces:**
- Consumes: fixed regression records, existing canonical R83 models, H1/H2/H3 auxiliary policies, and exact target QC threshold-lock CSVs.
- Produces: `enrich_qc_components(...)`, `load_frozen_qc_lock(target)`, `apply_frozen_qc(records, lock)`, and `summarize_frozen_qc(records) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing no-refit QC tests**

```python
def test_qc_loads_exact_lock_without_quantile_refit(monkeypatch):
    monkeypatch.setattr(np, "quantile", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("refit")))
    lock = load_frozen_qc_lock("C5")
    assert threshold_at(lock, 0.90)["threshold"] == pytest.approx(0.6378019435579839)
    assert threshold_at(lock, 0.95)["threshold"] == pytest.approx(0.6422456682734112)
```

- [ ] **Step 2: Run QC tests and confirm RED**

Run: `python -m pytest tests/test_iotj_a0t_vs_a4_qc.py -q`

- [ ] **Step 3: Implement exact frozen QC enrichment and decisions**

Reuse `combined_equal_mean_risk`, `classify_qc_decision`, and the existing R83/H2/H3 definitions. Read the p95 scales and q90/q95/q97.5 thresholds from the frozen lock; never call threshold-fitting functions.

- [ ] **Step 4: Implement HC90/HC95 summaries**

For each method, target, workpoint, and population (`accepted`, `accepted+review`, `reject`), emit coverage, review rate, reject rate, N, RMSE, MAE, NRMSE_range, R2, and Bias. Add target-size-weighted pooled rows by concatenating target records rather than averaging target metrics.

- [ ] **Step 5: Add fail-closed unsupported-interface behavior**

If an exact auxiliary asset or required record field is absent, write no QC metric row and return `BLOCKED_UNSUPPORTED_INTERFACE` with the missing dependency. Do not substitute approximate risks.

- [ ] **Step 6: Run QC tests and commit**

Run: `python -m pytest tests/test_iotj_a0t_vs_a4_qc.py -q`

Commit: `git commit -m "feat: compare A0T and A4 with frozen QC"`

---

### Task 4: Comparison tables, mechanism reports, and decision

**Files:**
- Create: `scripts/analyze_iotj_a0t_vs_a4_regression.py`
- Create: `tests/test_analyze_iotj_a0t_vs_a4_regression.py`

**Interfaces:**
- Produces: `routing_rows(scope_rows)`, `pooled_scope_rows(records)`, `regression_decision(rows) -> str`, and `write_reports(output)`.
- Produces the required compact CSV and Markdown files in the study root.

- [ ] **Step 1: Write failing decomposition and dual-gate tests**

```python
def test_routing_and_regression_gaps_follow_frozen_formula():
    row = routing_row({"S_ALL": 12.0, "S_CC": 9.0, "Oracle_ALL": 8.0, "Oracle_CC": 8.5})
    assert row["routing_gap"] == pytest.approx(3.0)
    assert row["regression_gap"] == pytest.approx(1.0)
    assert row["paired_regression_gap"] == pytest.approx(0.5)

def test_dual_gate_requires_c5_and_pooled_improvement():
    assert regression_decision(c5_delta=-1.0, pooled_delta=-0.5) == "REGRESSION_ADVANTAGE_SUPPORTED"
    assert regression_decision(c5_delta=0.1, pooled_delta=-0.5) == "REGRESSION_ADVANTAGE_NOT_SUPPORTED"
    assert regression_decision(c5_delta=-1.0, pooled_delta=0.1) == "REGRESSION_ADVANTAGE_NOT_SUPPORTED"
```

- [ ] **Step 2: Run analysis tests and confirm RED**

Run: `python -m pytest tests/test_analyze_iotj_a0t_vs_a4_regression.py -q`

- [ ] **Step 3: Implement compact comparison tables**

Write `regression_comparison.csv`, `per_gas_regression_comparison.csv`, `routing_scope_summary.csv`, and `qc_comparison.csv`. Preserve method, target, scope, N, units, checkpoint hash, prediction hash, seed, and calculation status on every row.

- [ ] **Step 4: Implement required reports**

Generate:

- `ROUTING_VS_REGRESSION_ANALYSIS.md` with A0T/A4 routing, requested regression, and paired regression gaps;
- `C5_A0T_VS_A4_REGRESSION.md` with CO/Methane gas tables, per-concentration curves, high-concentration errors, and Methane 225 ppm repeat1;
- `A0T_VS_GAPS_FINAL_CONCLUSION.md` answering classification, regression, QC, and positioning questions;
- `A0T_VS_A4_REGRESSION_REPORT.md` as the complete evidence index and limitations statement.

- [ ] **Step 5: Run analysis tests and commit**

Run: `python -m pytest tests/test_analyze_iotj_a0t_vs_a4_regression.py -q`

Commit: `git commit -m "feat: analyze A0T versus A4 regression mechanism"`

---

### Task 5: Pre-run freeze, six evaluations, and one-time analysis

**Files:**
- Create at execution: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/`
- Create at execution: six endpoint subdirectories under `endpoints/`

**Interfaces:**
- Consumes the code and tests from Tasks 1-4.
- Produces immutable calibration locks, models, predictions, manifests, comparison tables, reports, and QC records.

- [ ] **Step 1: Run pre-execution audit only**

Run: `python scripts/run_iotj_a0t_vs_a4_regression.py --audit-only`

Expected: `PRE_RUN_FREEZE.json` status `PASS`, six unique endpoints, exact checkpoint/H1/dataset/QC hashes, fixed alpha table, output test state `SEALED`, and no classifier training command.

- [ ] **Step 2: Run the six fixed evaluations once**

Run: `python scripts/run_iotj_a0t_vs_a4_regression.py --execute --device cpu --batch-size 32`

Expected: six calibration locks precede test opening; all six endpoint manifests finish `COMPLETE`; no alpha-search trace exists.

- [ ] **Step 3: Apply frozen QC once**

Run: `python scripts/evaluate_iotj_a0t_vs_a4_qc.py`

Expected: HC90/HC95 results for both methods or a fail-closed `BLOCKED_UNSUPPORTED_INTERFACE` audit without threshold changes.

- [ ] **Step 4: Generate final analysis once**

Run: `python scripts/analyze_iotj_a0t_vs_a4_regression.py`

Expected: all required CSV/Markdown outputs and one dual-gate decision.

---

### Task 6: Strict audit, compact evidence commit, and push

**Files:**
- Create: `scripts/audit_iotj_a0t_vs_a4_regression.py`
- Create: `tests/test_audit_iotj_a0t_vs_a4_regression.py`
- Create after execution: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/checkpoint_sha256.json`
- Create after execution: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/prediction_sha256.json`
- Create after execution: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/test_manifest_sha256.json`
- Create after execution: `results/iotj_canonical_v1_final/a0t_vs_a4_regression/STRICT_AUDIT.json`

**Interfaces:**
- Produces: `audit_study(study: Path) -> dict[str, Any]` and exits nonzero on any identity, leakage, scope, hash, or stop-rule defect.

- [ ] **Step 1: Write failing tamper tests**

Tests copy compact fixture manifests and assert rejection after changing a checkpoint hash, prediction byte, test-manifest hash, alpha, scope sample index, QC threshold, or dual-gate decision.

- [ ] **Step 2: Implement strict audit and hash indices**

Verify six endpoint identities, six calibration-before-test locks, dataset/H1/QC hashes, checkpoint whole-file hashes and ordered state fingerprints, prediction file hashes, test manifest hashes, row counts, S_CC/Oracle_CC index equality, C5 special slice presence, no alpha-grid trace, and exact conclusion recomputation.

- [ ] **Step 3: Run all required verification**

Run:

```text
python -m pytest tests/test_iotj_a0t_vs_a4_regression.py tests/test_iotj_a0t_vs_a4_qc.py tests/test_analyze_iotj_a0t_vs_a4_regression.py tests/test_audit_iotj_a0t_vs_a4_regression.py -q
python -m compileall -q scripts/run_iotj_a0t_vs_a4_regression.py scripts/evaluate_iotj_a0t_vs_a4_qc.py scripts/analyze_iotj_a0t_vs_a4_regression.py scripts/audit_iotj_a0t_vs_a4_regression.py
python tools/verify_iotj_canonical_v1_hashes.py dataset/iotj_canonical_v1
python scripts/audit_iotj_a0t_vs_a4_regression.py
```

Expected: required pytest passes, compileall exits zero, dataset reports 71 checked files and `PASS`, strict audit reports `PASS`.

- [ ] **Step 4: Stage only intended compact evidence**

Include code, tests, protocol/registry, summary CSVs, reports, compact endpoint manifests, alpha locks, hash indices, and strict audit. Exclude checkpoints, full prediction tables, temporary test directories, runner logs, and unrelated dirty files.

- [ ] **Step 5: Commit and push**

Commit: `git commit -m "results: publish A0T versus A4 regression evidence"`

Push: `git push origin codex/iotj-final-classification-le1`

- [ ] **Step 6: Stop**

Report A0T/A4 regression metrics, A4-A0T RMSE differences, C5 Methane difference, QC difference, hashes, commit, branch, and the final scientific decision. Do not launch follow-up experiments.
