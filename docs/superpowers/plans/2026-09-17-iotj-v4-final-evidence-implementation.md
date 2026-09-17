# IoT-J v4 Final Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the frozen canonical classification table, six reduced-budget C5 Ridge fits and paired statistics, canonical confidence-QC coverage figures, and a final audited evidence report without adding algorithms or changing canonical assets.

**Architecture:** Existing canonical results remain read-only. New code first builds a fail-closed provenance audit, then fits only the six registered reduced-budget R83/R84 configurations from immutable calibration identities, and finally performs deterministic post-processing into tables, figures, registry records, and the final report. Pure validation/statistics live in a focused module; command-line scripts create new immutable result directories with exclusive writes.

**Tech Stack:** Python 3, NumPy, Matplotlib (Agg backend), pytest, existing `gaps_flower.canonical_r1_v1` and `gaps_flower.canonical_quantitative_features` contracts.

**Spec:** `docs/superpowers/specs/2026-09-17-iotj-v4-final-evidence-scope-design.md`

## Global Constraints

- Scope authority is commit `5f554cdfa05af272a4d496c7063f317174bc0d81`.
- Existing `dataset/`, `results/`, checkpoints, manifests, CSV, JSON, and Markdown evidence are read-only.
- Write all new evidence below `results/iotj_v4_final_evidence_20260917/`; refuse to overwrite any existing destination.
- Classification is post-processing only; do not train or adapt a classifier.
- M0 and the six legacy labeled-DA probes are excluded from the classification main table and main figure.
- Cross-regime numbers are contextual; do not emit equal-supervision deltas or “fairly outperforms” wording across regimes.
- Reuse the canonical pooled classification rule: concatenate C3/C4/C5 predictions, then compute one Accuracy and one global Macro-F1.
- Reuse the existing 20%/320-window R83/R84 result; fit only 15%/240, 10%/160, and 5%/80 for R83 and R84.
- R84 adds exactly one immutable R0-v2 FedRidge prediction column to the same 83D sensor features.
- Alpha selection is deterministic five-fold raw-filename-grouped calibration-only CV over `(0, 0.01, 0.1, 1, 10, 100, 1000)`, pooled RMSE, first tie.
- Bootstrap is paired by `C5|filename`, 5,000 replicates, seed 42; never fall back to IID-window bootstrap.
- Test-side intervals are conditional on the frozen calibration subset and do not represent alternative-calibration-subset variability.
- QC uses `CLASSIFICATION_CONFIDENCE_ONLY`, the canonical R84 backend, the same C3/C4/C5 test population, and the 70%–100% grid in 5-point increments.
- Do not smooth or force monotonicity in any curve.
- Preserve the strict raw-file/exposure-disjoint limitation verbatim as a limitation, not a positive generalization result.
- Do not add FedBN, FedProto, MOON, FedDyn, a new backbone, new DA, multi-seed runs, a new QC model, or new hardware experiments.

## Frozen file structure

- Create `gaps_flower/iotj_v4_final_evidence.py`: pure contracts for identity matching, metric recomputation, paired grouped bootstrap, and provenance decisions.
- Create `scripts/audit_iotj_v4_final_pre_run.py`: read-only audit and candidate registry producer.
- Create `scripts/run_iotj_v4_c5_budget_ridge.py`: the only formal model-fitting entry point; six registered configurations only.
- Create `scripts/analyze_iotj_v4_calibration_budget.py`: merge the existing 20% result with reduced-budget predictions and produce aggregate/per-gas statistics.
- Create `scripts/build_iotj_v4_classification_main_table.py`: canonical classification consolidation with target-information regimes.
- Create `scripts/build_iotj_v4_qc_coverage.py`: QC extraction, compatible HC points, and optional same-population random-retention band.
- Create `scripts/plot_iotj_v4_final_figures.py`: publication-ready budget and QC figures from generated CSV only.
- Create `scripts/finalize_iotj_v4_canonical_evidence.py`: final report, evidence index, registry closure, and freeze marker.
- Create one focused `tests/test_*.py` file for each script/module above.
- Do not modify the existing untracked `scripts/build_iotj_v4_*.py`, `tests/test_iotj_v4_*.py`, M0 code, watcher logs, or historical results.

---

### Task 1: Pure evidence contracts

**Files:**
- Create: `gaps_flower/iotj_v4_final_evidence.py`
- Create: `tests/test_iotj_v4_final_evidence_contract.py`

**Interfaces:**
- Consumes: confirmed row dictionaries and NumPy arrays; no filesystem writes.
- Produces: `pooled_classification_metrics(rows)`, `match_budget_identities(budget_rows, canonical_rows)`, `paired_grouped_interval(...)`, `classify_provenance(checks)`, and constants `BUDGET_COUNTS`, `RIDGE_ALPHAS`, `BOOTSTRAP_SEED`, `BOOTSTRAP_REPLICATES`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_pooled_metrics_merge_predictions_before_macro_f1():
    rows = [
        {"target": "C3", "true_class": 0, "predicted_class": 0},
        {"target": "C3", "true_class": 1, "predicted_class": 0},
        {"target": "C4", "true_class": 1, "predicted_class": 1},
        {"target": "C5", "true_class": 1, "predicted_class": 1},
    ]
    result = pooled_classification_metrics(rows)
    assert result["aggregation"] == "merged_predictions_global_metric"
    assert result["N"] == 4
    assert result["accuracy"] == 0.75


def test_grouped_bootstrap_fails_when_filename_is_missing():
    with pytest.raises(RuntimeError, match="grouping key"):
        paired_grouped_interval(
            truth=np.array([0.0, 1.0]),
            pred_r83=np.array([0.0, 1.0]),
            pred_r84=np.array([0.0, 1.0]),
            gas_range=np.ones(2),
            groups=np.array(["", ""]),
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_iotj_v4_final_evidence_contract.py -q --basetemp .tmp_pytest_iotj_v4_contract_20260917`

Expected: import failure because `gaps_flower.iotj_v4_final_evidence` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

```python
BUDGET_COUNTS = {20: 320, 15: 240, 10: 160, 5: 80}
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 5000


def paired_grouped_interval(truth, pred_r83, pred_r84, gas_range, groups):
    groups = np.asarray(groups, dtype=str)
    if groups.size == 0 or any(not value for value in groups):
        raise RuntimeError("FAIL_CLOSED bootstrap grouping key is missing")
    output = bootstrap_paired_group_deltas(
        truth, pred_r83, pred_r84, groups, gas_range,
        replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    return summarize_bootstrap(
        "C5", output,
        point=_paired_points(truth, pred_r83, pred_r84, gas_range),
        n_groups=len(np.unique(groups)),
        n_replicates=BOOTSTRAP_REPLICATES,
    )
```

Use stable identity matching on `physical_identity`; reject duplicates, missing identities, reordered labels, non-finite values, unexpected budgets, and any method outside `R83_TARGET_ONLY` and `R84_CONCAT`.

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run: `python -m pytest tests/test_iotj_v4_final_evidence_contract.py -q --basetemp .tmp_pytest_iotj_v4_contract_20260917`

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add gaps_flower/iotj_v4_final_evidence.py tests/test_iotj_v4_final_evidence_contract.py
git commit -m "feat: add IoT-J v4 evidence contracts"
```

### Task 2: Final pre-run audit and candidate registry

**Files:**
- Create: `scripts/audit_iotj_v4_final_pre_run.py`
- Create: `tests/test_audit_iotj_v4_final_pre_run.py`

**Interfaces:**
- Consumes: scope commit, canonical classification CSV/predictions, CORAL/DANN final-evaluation JSON, R1 evidence bundle, R0-v2 lock, budget manifests, Q1-v2 bundle, and current-authoritative confidence-QC tables.
- Produces: `pre_run/FINAL_PRE_RUN_AUDIT.md`, `pre_run/final_pre_run_audit.json`, `pre_run/qc_provenance_audit.json`, `pre_run/experiment_registry.csv`, and `pre_run/input_sha256.json`.

- [ ] **Step 1: Write failing audit tests**

```python
def test_unknown_critical_field_blocks_only_affected_component():
    verdict = final_verdict({"budget_grouping_key": "unknown", "classification": "PASS"})
    assert verdict["regression_budget"] == "BLOCKED"
    assert verdict["classification"] == "PASS"


def test_mismatched_test_population_excludes_comparator():
    item = audit_comparator_population("CORAL", {"C3": 680, "C4": 320, "C5": 1360},
                                       {"C3": 2677, "C4": 1360, "C5": 1360})
    assert item["status"] == "EXCLUDED_NONCANONICAL"
```

- [ ] **Step 2: Run the audit tests and verify RED**

Run: `python -m pytest tests/test_audit_iotj_v4_final_pre_run.py -q --basetemp .tmp_pytest_iotj_v4_audit_20260917`

Expected: import failure because the audit script does not exist.

- [ ] **Step 3: Implement the fail-closed audit**

The audit must establish and serialize these exact facts:

```python
POOLED_RULE = "concatenate C3/C4/C5 predictions, then compute one Accuracy and one global Macro-F1"
INFORMATION_NOTE = (
    "Cross-regime numbers are reported for contextual comparison and must not "
    "be interpreted as an equal-supervision ranking."
)
GROUPING_KEY = "C5|filename"
```

It must verify 320/240/160/80 nested identities, 80 non-empty C5 test filename groups, 20 groups per gas, no calibration/test exact-identity overlap, 83D/104D schema hashes, R0-v2 model-lock hash, alpha rule, R1 test identity hash, and Q1/current-authoritative compatibility. Mark CORAL/DANN `EXCLUDED_NONCANONICAL` because their observed C3/C4 N values are 680/320 rather than 2677/1360; do not retrain them. Mark historical `05_qc_final.csv` and legacy random-retention evidence `HISTORICAL_SUPERSEDED`; authorize overlays only from `FINAL_CONFIDENCE_QC_METRICS.csv` and `FINAL_CONFIDENCE_QC_THRESHOLDS.csv` after matching backend, population, risk definition, and authority chain.

- [ ] **Step 4: Run tests and a dry audit**

Run: `python -m pytest tests/test_audit_iotj_v4_final_pre_run.py -q --basetemp .tmp_pytest_iotj_v4_audit_20260917`

Run: `python scripts/audit_iotj_v4_final_pre_run.py --dry-run --scope-commit 5f554cdfa05af272a4d496c7063f317174bc0d81`

Expected: tests pass; dry-run prints a JSON PASS/BLOCKED breakdown and writes nothing.

- [ ] **Step 5: Commit the audit implementation**

```bash
git add scripts/audit_iotj_v4_final_pre_run.py tests/test_audit_iotj_v4_final_pre_run.py
git commit -m "feat: add IoT-J v4 final pre-run audit"
```

### Task 3: Six-fit reduced-budget Ridge runner

**Files:**
- Create: `scripts/run_iotj_v4_c5_budget_ridge.py`
- Create: `tests/test_run_iotj_v4_c5_budget_ridge.py`

**Interfaces:**
- Consumes: PASS regression section of `final_pre_run_audit.json`, R1 C5 calibration/test caches, budget identities and labels, R0-v2 source models, and frozen R1 C5 predicted classes.
- Produces: six experiment directories, `calibration_model_locks.json`, `target_test_release_receipt.json`, `predictions.csv`, `run_manifest.json`, and `COMPLETE.json` under `regression_budget/`.

- [ ] **Step 1: Write failing runner tests**

```python
def test_registered_runs_are_exactly_six():
    assert registered_experiment_ids() == (
        "IOTJ-V4-C5-B15-R83-S42", "IOTJ-V4-C5-B15-R84-S42",
        "IOTJ-V4-C5-B10-R83-S42", "IOTJ-V4-C5-B10-R84-S42",
        "IOTJ-V4-C5-B05-R83-S42", "IOTJ-V4-C5-B05-R84-S42",
    )


def test_test_access_requires_all_six_model_locks():
    with pytest.raises(RuntimeError, match="all six calibration locks"):
        release_test_access({"IOTJ-V4-C5-B15-R83-S42": {"locked": True}})
```

- [ ] **Step 2: Run the runner tests and verify RED**

Run: `python -m pytest tests/test_run_iotj_v4_c5_budget_ridge.py -q --basetemp .tmp_pytest_iotj_v4_runner_20260917`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement calibration-only fitting and locked evaluation**

Map each budget identity into the immutable 320-row R1 C5 calibration cache. For every budget/method/gas, call existing `select_grouped_cv_alpha`, then `fit_ridge_model`; R83 consumes 83 sensor columns and R84 consumes the same 83 columns plus the gas-specific frozen R0-v2 prediction. Write all six model locks before reading the frozen C5 test arrays/predictions. Then evaluate exactly 1,360 C5 test rows using the existing R1 classifier route, never test truth for alpha/model selection.

The CLI exposes no method, budget, seed, alpha, or bootstrap override. Its only mutable parameters are authenticated input/audit paths and a new output directory.

- [ ] **Step 4: Run unit tests and dry-run**

Run: `python -m pytest tests/test_run_iotj_v4_c5_budget_ridge.py -q --basetemp .tmp_pytest_iotj_v4_runner_20260917`

Run: `python scripts/run_iotj_v4_c5_budget_ridge.py --dry-run --audit results/iotj_v4_final_evidence_20260917/pre_run/final_pre_run_audit.json --output results/iotj_v4_final_evidence_20260917/regression_budget`

Expected: tests pass; dry-run reports six configurations and zero test access.

- [ ] **Step 5: Commit the runner**

```bash
git add scripts/run_iotj_v4_c5_budget_ridge.py tests/test_run_iotj_v4_c5_budget_ridge.py
git commit -m "feat: add frozen C5 Ridge budget runner"
```

### Task 4: Calibration-budget statistics

**Files:**
- Create: `scripts/analyze_iotj_v4_calibration_budget.py`
- Create: `tests/test_analyze_iotj_v4_calibration_budget.py`

**Interfaces:**
- Consumes: existing 20% R1 predictions/bootstrap and the six new prediction files.
- Produces: `analysis/calibration_budget_83d_84d.csv`, `analysis/calibration_budget_per_gas.csv`, `analysis/calibration_budget_bootstrap.csv`, and `analysis/calibration_budget_analysis.json`.

- [ ] **Step 1: Write failing analysis tests**

```python
def test_delta_direction_is_r84_minus_r83():
    row = summarize_pair(np.array([0.0, 2.0]), np.array([1.0, 3.0]),
                         np.array([0.5, 2.5]), np.array([1.0, 1.0]))
    assert row["delta_RMSE"] < 0
    assert row["relative_NRMSE_improvement_percent"] > 0


def test_interval_declares_conditional_calibration_boundary():
    assert "conditional on the frozen calibration subset" in INTERVAL_INTERPRETATION
```

- [ ] **Step 2: Run the analysis tests and verify RED**

Run: `python -m pytest tests/test_analyze_iotj_v4_calibration_budget.py -q --basetemp .tmp_pytest_iotj_v4_analysis_20260917`

Expected: import failure because the analysis script does not exist.

- [ ] **Step 3: Implement deterministic aggregate and per-gas analysis**

Use these column semantics:

```python
delta_rmse = rmse_r84 - rmse_r83
delta_nrmse = nrmse_r84 - nrmse_r83
relative_improvement = 100.0 * (nrmse_r83 - nrmse_r84) / nrmse_r83
```

The aggregate CSV has one row for each budget 5, 10, 15, 20 and includes calibration N, test N, both RMSE/NRMSE values, both deltas, relative improvement, paired NRMSE CI lower/upper, seed, replicates, grouping key, and unique-group count. The per-gas CSV repeats the paired calculation for Ethanol, CO, Ethylene, and Methane. Reuse the existing 20% model predictions; model fitting at 20% is forbidden. Do not assert improvement direction or monotonicity.

- [ ] **Step 4: Run the analysis tests**

Run: `python -m pytest tests/test_analyze_iotj_v4_calibration_budget.py -q --basetemp .tmp_pytest_iotj_v4_analysis_20260917`

Expected: all tests pass, including bootstrap determinism and row-order mismatch failure.

- [ ] **Step 5: Commit the analysis**

```bash
git add scripts/analyze_iotj_v4_calibration_budget.py tests/test_analyze_iotj_v4_calibration_budget.py
git commit -m "feat: analyze paired C5 budget effects"
```

### Task 5: Canonical classification main table

**Files:**
- Create: `scripts/build_iotj_v4_classification_main_table.py`
- Create: `tests/test_build_iotj_v4_classification_main_table.py`

**Interfaces:**
- Consumes: canonical classification comparison/predictions and pre-run comparator verdicts.
- Produces: `classification/classification_main_table.csv`, `classification/classification_main_table.md`, and `classification/classification_manifest.json`.

- [ ] **Step 1: Write failing table tests**

```python
def test_main_table_excludes_noncanonical_and_mechanism_rows():
    methods = {row["Method"] for row in build_rows(fixture_inputs())}
    assert "M0" not in methods
    assert "CORAL" not in methods
    assert "DANN" not in methods
    assert methods == {"FedAvg", "FedProx", "SCAFFOLD", "MMD", "A0T", "GAPS/A4"}


def test_main_table_has_no_cross_regime_delta_column():
    assert not any("delta" in name.lower() for name in MAIN_COLUMNS)
```

- [ ] **Step 2: Run the table tests and verify RED**

Run: `python -m pytest tests/test_build_iotj_v4_classification_main_table.py -q --basetemp .tmp_pytest_iotj_v4_classification_20260917`

Expected: import failure because the builder does not exist.

- [ ] **Step 3: Implement canonical-only consolidation**

Create three ordered groups: Source-only FL (`FedAvg`, `FedProx`, `SCAFFOLD`), Unlabeled DA (`MMD`), and Label-aware reference/proposed (`A0T`, `GAPS/A4`). Columns are Target information, C3/C4/C5 Accuracy, pooled Accuracy, C3/C4/C5 Macro-F1, pooled Macro-F1, endpoint, and seed. Recompute every metric from the saved canonical predictions and require exact agreement with the reported CSV before writing. Put the cross-regime interpretation sentence and pooled definition in both Markdown and manifest.

- [ ] **Step 4: Run the table tests**

Run: `python -m pytest tests/test_build_iotj_v4_classification_main_table.py -q --basetemp .tmp_pytest_iotj_v4_classification_20260917`

Expected: all tests pass and fixture output contains no cross-regime superiority wording.

- [ ] **Step 5: Commit the table builder**

```bash
git add scripts/build_iotj_v4_classification_main_table.py tests/test_build_iotj_v4_classification_main_table.py
git commit -m "feat: build canonical classification main table"
```

### Task 6: Canonical QC coverage dataset

**Files:**
- Create: `scripts/build_iotj_v4_qc_coverage.py`
- Create: `tests/test_build_iotj_v4_qc_coverage.py`

**Interfaces:**
- Consumes: PASS `qc_provenance_audit.json`, Q1-v2 confidence curve, current-authoritative confidence thresholds/metrics, and R1 R84 predictions.
- Produces: `qc/qc_coverage_nrmse.csv`, `qc/qc_locked_operating_points.csv`, optional `qc/qc_random_retention_band.csv`, and `qc/qc_coverage_manifest.json`.

- [ ] **Step 1: Write failing QC tests**

```python
def test_exact_requested_grid_is_selected_without_smoothing():
    rows = select_coverage_grid(curve_fixture())
    assert [row["target_coverage"] for row in rows] == [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    assert rows[0]["NRMSE_range"] == curve_fixture()[0]["NRMSE_range"]


def test_superseded_multisignal_hc_is_rejected():
    with pytest.raises(RuntimeError, match="CONFIDENCE_QC_FINAL"):
        authorize_operating_points({"policy": "historical_multisignal_equal_mean"})
```

- [ ] **Step 2: Run the QC tests and verify RED**

Run: `python -m pytest tests/test_build_iotj_v4_qc_coverage.py -q --basetemp .tmp_pytest_iotj_v4_qc_20260917`

Expected: import failure because the QC builder does not exist.

- [ ] **Step 3: Implement extraction and compatible overlays**

Filter `CLASSIFICATION_CONFIDENCE_ONLY` for C3, C4, C5, and target-stratified POOLED at exactly seven registered coverage targets. Use `actual_coverage` as x and `NRMSE_range` as y. Overlay only the calibration-locked confidence-only nominal 0.90/0.95 rows from the current-authoritative tables. If the audit confirms exact R1 prediction identity/backend alignment, generate a target-stratified equal-retention random NRMSE mean and 2.5/97.5 percentile band with 1,000 deterministic draws at seed 42; otherwise omit the file and record the precise failed check. Never read the legacy random baseline as a numeric source.

- [ ] **Step 4: Run the QC tests**

Run: `python -m pytest tests/test_build_iotj_v4_qc_coverage.py -q --basetemp .tmp_pytest_iotj_v4_qc_20260917`

Expected: all tests pass; the 100% row equals the full-population NRMSE exactly.

- [ ] **Step 5: Commit the QC builder**

```bash
git add scripts/build_iotj_v4_qc_coverage.py tests/test_build_iotj_v4_qc_coverage.py
git commit -m "feat: build canonical QC coverage evidence"
```

### Task 7: Publication figures

**Files:**
- Create: `scripts/plot_iotj_v4_final_figures.py`
- Create: `tests/test_plot_iotj_v4_final_figures.py`

**Interfaces:**
- Consumes: generated analysis/QC CSV only.
- Produces: PDF, SVG, and 600-DPI PNG for `fig_calibration_budget`, `fig_qc_coverage_nrmse`, and `fig_qc_coverage_nrmse_by_target`, plus `figures/figure_source_manifest.json`.

- [ ] **Step 1: Write failing figure tests**

```python
def test_budget_panel_b_uses_signed_r84_minus_r83_delta():
    figure = build_budget_figure(budget_fixture())
    assert list(figure.axes[1].lines[1].get_ydata()) == [-0.01, 0.0, 0.02, -0.03]


def test_qc_x_axis_uses_actual_coverage():
    figure = build_qc_figure(qc_fixture(), [], None)
    assert list(figure.axes[0].lines[0].get_xdata()) == [0.71, 0.76]
```

- [ ] **Step 2: Run the figure tests and verify RED**

Run: `python -m pytest tests/test_plot_iotj_v4_final_figures.py -q --basetemp .tmp_pytest_iotj_v4_figures_20260917`

Expected: import failure because the plotting script does not exist.

- [ ] **Step 3: Implement publication-safe figures**

Use Matplotlib Agg, Okabe–Ito colors plus distinct markers/linestyles, vector export, 600-DPI PNG, readable 7–9 pt final-size fonts, no interpolation, and no smoothing. Budget Panel A plots R83/R84 NRMSE at 5/10/15/20%; Panel B plots signed `R84 - R83` with a zero line. The QC primary plot uses pooled actual coverage and NRMSE; locked HC90/HC95 are distinct point markers, and the random band appears only when its CSV exists. The auxiliary plot contains C3/C4/C5 panels.

- [ ] **Step 4: Run tests and render fixture figures**

Run: `python -m pytest tests/test_plot_iotj_v4_final_figures.py -q --basetemp .tmp_pytest_iotj_v4_figures_20260917`

Expected: all tests pass; PDF/SVG/PNG exist, have non-zero size, and the source manifest hashes every plotted CSV.

- [ ] **Step 5: Commit the plotting implementation**

```bash
git add scripts/plot_iotj_v4_final_figures.py tests/test_plot_iotj_v4_final_figures.py
git commit -m "feat: plot IoT-J v4 final evidence"
```

### Task 8: Formal execution, final report, and freeze

**Files:**
- Create: `scripts/finalize_iotj_v4_canonical_evidence.py`
- Create: `tests/test_finalize_iotj_v4_canonical_evidence.py`

**Interfaces:**
- Consumes: every output from Tasks 2–7.
- Produces: `FINAL_CANONICAL_EVIDENCE_REPORT.md`, `RESULT_ANALYSIS.md`, `EXPERIMENT_AUDIT.md`, final `experiment_registry.csv`, `evidence_sha256_index.json`, and `IOTJ_V4_EXPERIMENTS_FROZEN.json`.

- [ ] **Step 1: Write failing finalizer tests**

```python
def test_freeze_requires_every_registered_artifact(tmp_path):
    with pytest.raises(RuntimeError, match="required artifact"):
        finalize(tmp_path)


def test_report_preserves_robustness_limitation():
    report = render_report(valid_fixture())
    assert "strict file/exposure-disjoint" in report
    assert "does not represent variability across alternative calibration subsets" in report
```

- [ ] **Step 2: Run the finalizer tests and verify RED**

Run: `python -m pytest tests/test_finalize_iotj_v4_canonical_evidence.py -q --basetemp .tmp_pytest_iotj_v4_finalize_20260917`

Expected: import failure because the finalizer does not exist.

- [ ] **Step 3: Implement report/audit/freeze gates**

The report answers only: classification performance under explicit target-information regimes; paired historical source-prior effects by calibration budget; and the confidence ranking coverage–NRMSE tradeoff. It labels copied values `reported` and newly calculated values `recomputed`, retains anomalies and negative transfer, and contains no cross-regime fairness claim. The freeze marker is written last and only when every hash, registry status, row count, metric recomputation, bootstrap contract, figure source, and limitation check passes.

- [ ] **Step 4: Run all focused tests before formal execution**

Run: `python -m pytest tests/test_iotj_v4_final_evidence_contract.py tests/test_audit_iotj_v4_final_pre_run.py tests/test_run_iotj_v4_c5_budget_ridge.py tests/test_analyze_iotj_v4_calibration_budget.py tests/test_build_iotj_v4_classification_main_table.py tests/test_build_iotj_v4_qc_coverage.py tests/test_plot_iotj_v4_final_figures.py tests/test_finalize_iotj_v4_canonical_evidence.py -q --basetemp .tmp_pytest_iotj_v4_final_20260917`

Expected: all focused tests pass.

- [ ] **Step 5: Commit the finalizer, then record the authorized code commit**

```bash
git add scripts/finalize_iotj_v4_canonical_evidence.py tests/test_finalize_iotj_v4_canonical_evidence.py
git commit -m "feat: finalize IoT-J v4 canonical evidence"
git rev-parse HEAD
```

- [ ] **Step 6: Run the formal pre-run audit**

```powershell
python scripts/audit_iotj_v4_final_pre_run.py --output results/iotj_v4_final_evidence_20260917/pre_run --scope-commit 5f554cdfa05af272a4d496c7063f317174bc0d81 --authorized-code-commit "$(git rev-parse HEAD)"
```

Expected: `FINAL_PRE_RUN_AUDIT.md` reports PASS for the regression budget, canonical classification sources, bootstrap grouping, and QC provenance; CORAL/DANN are explicitly excluded rather than silently mixed.

- [ ] **Step 7: Run exactly the six Ridge configurations**

```powershell
python scripts/run_iotj_v4_c5_budget_ridge.py --audit results/iotj_v4_final_evidence_20260917/pre_run/final_pre_run_audit.json --output results/iotj_v4_final_evidence_20260917/regression_budget --authorized-code-commit "$(git rev-parse HEAD)"
```

Expected: exactly six experiment IDs complete; no B20 fit and no unregistered configuration exists.

- [ ] **Step 8: Build statistics, tables, QC data, and figures**

```powershell
python scripts/analyze_iotj_v4_calibration_budget.py --run-root results/iotj_v4_final_evidence_20260917/regression_budget --output results/iotj_v4_final_evidence_20260917/analysis
python scripts/build_iotj_v4_classification_main_table.py --audit results/iotj_v4_final_evidence_20260917/pre_run/final_pre_run_audit.json --output results/iotj_v4_final_evidence_20260917/classification
python scripts/build_iotj_v4_qc_coverage.py --audit results/iotj_v4_final_evidence_20260917/pre_run/qc_provenance_audit.json --output results/iotj_v4_final_evidence_20260917/qc
python scripts/plot_iotj_v4_final_figures.py --root results/iotj_v4_final_evidence_20260917 --output results/iotj_v4_final_evidence_20260917/figures
```

Expected: the two requested main figures, the auxiliary target QC figure, and all requested CSV files are produced without modifying source assets.

- [ ] **Step 9: Finalize and independently verify the immutable evidence bundle**

```powershell
python scripts/finalize_iotj_v4_canonical_evidence.py --root results/iotj_v4_final_evidence_20260917
python scripts/finalize_iotj_v4_canonical_evidence.py --root results/iotj_v4_final_evidence_20260917 --audit-only
git diff --check
```

Expected: `IOTJ_V4_EXPERIMENTS_FROZEN.json` has status `IOTJ_V4_EXPERIMENTS_FROZEN`; audit-only recomputation returns PASS; no additional experiment is launched.

- [ ] **Step 10: Audit the final Git scope without staging unrelated assets**

```bash
git status --short
git diff --name-only 5f554cdfa05af272a4d496c7063f317174bc0d81 HEAD
```

Expected: committed changes are limited to this implementation plan plus the new source/test files listed above; watcher logs, M0 files, historical evidence, and unrelated dirty files remain unstaged and unchanged.
