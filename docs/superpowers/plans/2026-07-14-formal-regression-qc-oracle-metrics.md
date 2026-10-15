# Formal Regression QC Oracle Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fully recomputed H8 oracle-route predictions and Accepted/Nonreject actual/oracle RMSE and NRMSE to the A6/B5/B2 formal C5 QC evidence.

**Architecture:** Extend the existing H8 evaluator to emit a second 1,360-row test stream whose complete source-head and target-head route is forced to `true_class`. The QC evaluator joins that stream one-to-one without changing risk scoring or workpoint decisions, calculates subset metrics, and the formal summarizer exposes the new fields. Unit tests cover each contract before Alibaba Cloud ECS regenerates the formal artifacts.

**Tech Stack:** Python 3, NumPy, scikit-learn-compatible existing MLP/Ridge helpers, CSV/JSON, pytest, Alibaba Cloud ECS.

## Global Constraints

- Do not retrain or mutate A6/B5/B2 classification checkpoints.
- Do not change H8 fitting grids, C5 calibration split, risk components, score-family selection, or HC95/HC90 thresholds.
- `Nonreject` means `accept + review`; only `reject` is excluded.
- Forced oracle-route keeps the workpoint subset and recomputes all H8 source/final heads using `route_class=true_class`.
- Oracle FULL must contain all 1,360 C5 test windows and must not be labeled `S_CC`.
- Formal H8 refits and result regeneration run on Alibaba Cloud ECS; local execution is limited to unit/contract tests and read-only result analysis.
- Preserve existing actual-route output columns and values byte-for-value where practical, numerically within `1e-12` otherwise.

---

### Task 1: Emit a Complete H8 Oracle-Route Stream

**Files:**
- Modify: `run_source_augmented_target_ridge_eval.py:99-110,286-470`
- Modify: `tests/test_iotj_c5_regression_suite.py`

**Interfaces:**
- Produces: `force_oracle_routes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]`
- Produces: `target_predictions_plus_source_preds_oracle_route.csv`
- Produces prediction key: `target_ridge_plus_source_preds_oracle_route_ppm`
- Consumes existing `attach_source_predictions`, `add_pred_features`, and `apply_client_models`.

- [ ] **Step 1: Write failing oracle-route copy tests**

```python
from run_source_augmented_target_ridge_eval import force_oracle_routes


def test_force_oracle_routes_copies_rows_and_replaces_only_route() -> None:
    source = [{"client": "C5", "sample_index": 7, "pred_class": 1, "true_class": 3, "route_class": 1}]
    result = force_oracle_routes(source)
    assert result[0]["route_class"] == 3
    assert result[0]["pred_class"] == 1
    assert source[0]["route_class"] == 1
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `python -m pytest tests/test_iotj_c5_regression_suite.py -q --basetemp .tmp_pytest_qc_oracle_h8_red`

Expected: collection/import failure because `force_oracle_routes` is not defined.

- [ ] **Step 3: Add the copy helper and oracle inference branch**

```python
def force_oracle_routes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        true_class = inum(item.get("true_class"), -1)
        if true_class not in CLASS_NAMES:
            raise ValueError(f"invalid true_class for oracle route: {true_class}")
        item["route_class"] = true_class
        output.append(item)
    return output
```

After fitting `aug_models`, derive and evaluate the second stream:

```python
target_test_oracle = force_oracle_routes(target_test)
target_test_oracle_with_src = attach_source_predictions(
    target_test_oracle, ridge_models, mlp_models, shared_model
)
target_test_oracle_aug = add_pred_features(target_test_oracle_with_src, pred_keys)
target_oracle_aug = apply_client_models(
    target_test_oracle_aug,
    aug_models,
    "target_ridge_plus_source_preds_oracle_route",
)
write_csv(
    out / "target_predictions_plus_source_preds_oracle_route.csv",
    [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_oracle_aug],
)
```

Add the oracle filename and count to `manifest.json` while leaving existing fields intact.

- [ ] **Step 4: Run H8 contract tests and confirm GREEN**

Run: `python -m pytest tests/test_iotj_c5_regression_suite.py -q --basetemp .tmp_pytest_qc_oracle_h8_green`

Expected: all tests pass.

- [ ] **Step 5: Commit the H8 oracle stream**

```powershell
git add -- run_source_augmented_target_ridge_eval.py tests/test_iotj_c5_regression_suite.py
git commit -m "feat: emit H8 oracle-route predictions"
```

### Task 2: Calculate Accepted and Nonreject Actual/Oracle Metrics

**Files:**
- Modify: `scripts/evaluate_iotj_high_coverage_qc.py:369-450,608-747`
- Modify: `tests/test_iotj_high_coverage_qc.py`

**Interfaces:**
- Produces: `attach_oracle_prediction(rows, oracle_rows, oracle_source_key, oracle_output_key) -> list[dict[str, Any]]`
- Extends: `evaluate_workpoint(..., oracle_pred_key: str | None = None) -> dict[str, Any]`
- Consumes: `target_predictions_plus_source_preds_oracle_route.csv` from Task 1.

- [ ] **Step 1: Write failing metric and join tests**

```python
def test_workpoint_reports_nonreject_and_oracle_metrics() -> None:
    rows = [
        {"client": "C5", "split": "test", "sample_index": 0, "true_class": 0, "pred_class": 1,
         "true_ppm": 10.0, "pred_ppm": 30.0, "oracle_ppm": 12.0, "qc_decision": "accept"},
        {"client": "C5", "split": "test", "sample_index": 1, "true_class": 0, "pred_class": 0,
         "true_ppm": 20.0, "pred_ppm": 24.0, "oracle_ppm": 21.0, "qc_decision": "review"},
        {"client": "C5", "split": "test", "sample_index": 2, "true_class": 0, "pred_class": 0,
         "true_ppm": 30.0, "pred_ppm": 90.0, "oracle_ppm": 31.0, "qc_decision": "reject"},
    ]
    report = evaluate_workpoint(rows, "pred_ppm", oracle_pred_key="oracle_ppm", n_random=0)
    assert report["nonreject_N"] == 2
    assert report["nonreject_metrics"]["RMSE"] == pytest.approx((208.0 / 2.0) ** 0.5)
    assert report["oracle_accept_metrics"]["RMSE"] == pytest.approx(2.0)
    assert report["oracle_nonreject_metrics"]["RMSE"] == pytest.approx((5.0 / 2.0) ** 0.5)
```

Add duplicate, missing, and non-finite oracle prediction tests. Each must raise `ValueError` before evaluation.

- [ ] **Step 2: Run focused QC tests and confirm RED**

Run: `python -m pytest tests/test_iotj_high_coverage_qc.py -q --basetemp .tmp_pytest_qc_oracle_metrics_red`

Expected: `evaluate_workpoint` rejects the new keyword or omits the new fields.

- [ ] **Step 3: Implement strict one-to-one oracle attachment**

```python
def attach_oracle_prediction(rows, oracle_rows, oracle_source_key, oracle_output_key):
    by_key = {}
    for row in oracle_rows:
        key = _row_key(row)
        if key in by_key:
            raise ValueError(f"duplicate oracle row: {key}")
        value = _to_float(row.get(oracle_source_key), np.nan)
        if not np.isfinite(value):
            raise ValueError(f"non-finite oracle prediction: {key}")
        by_key[key] = value
    output = []
    for row in rows:
        key = _row_key(row)
        if key not in by_key:
            raise ValueError(f"missing oracle prediction: {key}")
        item = dict(row)
        item[oracle_output_key] = by_key[key]
        output.append(item)
    if len(by_key) != len(output):
        raise ValueError("oracle/test row sets are not identical")
    return output
```

- [ ] **Step 4: Extend workpoint metrics without changing decisions**

```python
nonreject_mask = accept_mask | review_mask
report.update({
    "nonreject_N": int(nonreject_mask.sum()),
    "nonreject_metrics": _regression_metrics(take(nonreject_mask), pred_key),
    "oracle_accept_metrics": _regression_metrics(take(accept_mask), oracle_pred_key),
    "oracle_nonreject_metrics": _regression_metrics(take(nonreject_mask), oracle_pred_key),
})
```

Add required CLI argument `--h8-test-oracle`, attach its prediction after the existing H8 merge, and record its path/key in the manifest. Assert FULL has `accept_N=nonreject_N=1360` and matching Accepted/Nonreject metrics for both prediction keys.

- [ ] **Step 5: Run QC tests and confirm GREEN**

Run: `python -m pytest tests/test_iotj_high_coverage_qc.py tests/test_iotj_c5_regression_suite.py -q --basetemp .tmp_pytest_qc_oracle_metrics_green`

Expected: all tests pass and existing risk truth-invariance tests remain green.

- [ ] **Step 6: Commit QC metric support**

```powershell
git add -- scripts/evaluate_iotj_high_coverage_qc.py tests/test_iotj_high_coverage_qc.py
git commit -m "feat: report QC nonreject and oracle metrics"
```

### Task 3: Wire the Formal Suite and Summary Table

**Files:**
- Modify: `scripts/run_iotj_c5_regression_suite.py:12-128`
- Modify: `scripts/summarize_iotj_c5_formal_regression.py:68-139`
- Modify: `tests/test_iotj_c5_regression_suite.py`

**Interfaces:**
- Passes CLI: `--h8-test-oracle <h8_no_rescue/target_predictions_plus_source_preds_oracle_route.csv>`
- Adds CSV columns defined in the design specification.

- [ ] **Step 1: Extend failing command and flattening tests**

```python
def test_suite_passes_oracle_h8_stream_to_qc(tmp_path: Path) -> None:
    commands = build_suite_commands(
        classifier_checkpoint=Path("classifier.pth"),
        regression_checkpoint=Path("regression.pt"),
        data_root=Path("dataset"),
        output_root=tmp_path,
        device="cpu",
        seed=42,
        n_random=1000,
    )
    qc = commands[3]
    oracle = qc[qc.index("--h8-test-oracle") + 1]
    assert oracle.endswith("target_predictions_plus_source_preds_oracle_route.csv")
```

Extend the existing `flatten_operational_qc` fixture to assert exact values for `nonreject_N`, `nonreject_RMSE`, `nonreject_NRMSE`, `oracle_accept_RMSE`, `oracle_accept_NRMSE`, `oracle_nonreject_RMSE`, and `oracle_nonreject_NRMSE`.

- [ ] **Step 2: Run suite tests and confirm RED**

Run: `python -m pytest tests/test_iotj_c5_regression_suite.py -q --basetemp .tmp_pytest_qc_oracle_summary_red`

Expected: missing CLI path and flattened fields.

- [ ] **Step 3: Wire the oracle file and flatten nested metrics**

```python
nonreject = item["nonreject_metrics"]
oracle_accept = item["oracle_accept_metrics"]
oracle_nonreject = item["oracle_nonreject_metrics"]
row.update({
    "nonreject_N": item["nonreject_N"],
    "nonreject_RMSE": nonreject["RMSE"],
    "nonreject_NRMSE": nonreject["NRMSE"],
    "oracle_accept_RMSE": oracle_accept["RMSE"],
    "oracle_accept_NRMSE": oracle_accept["NRMSE"],
    "oracle_nonreject_RMSE": oracle_nonreject["RMSE"],
    "oracle_nonreject_NRMSE": oracle_nonreject["NRMSE"],
})
```

Update the Markdown Operational QC table to show actual Accepted, actual Nonreject, oracle Accepted, and oracle Nonreject RMSE/NRMSE side by side.

- [ ] **Step 4: Run all formal regression/QC contract tests**

Run: `python -m pytest tests/test_iotj_c5_regression_suite.py tests/test_iotj_high_coverage_qc.py -q --basetemp .tmp_pytest_qc_oracle_summary_green`

Expected: all tests pass.

- [ ] **Step 5: Commit suite and summary wiring**

```powershell
git add -- scripts/run_iotj_c5_regression_suite.py scripts/summarize_iotj_c5_formal_regression.py tests/test_iotj_c5_regression_suite.py
git commit -m "feat: expose formal QC oracle metrics"
```

### Task 4: Regenerate and Audit A6/B5/B2 on Alibaba Cloud ECS

**Files:**
- Regenerate: `results/iotj_c5_formal_regression_20260713_v2/{A6,B5,B2}/h8_no_rescue`
- Regenerate: `results/iotj_c5_formal_regression_20260713_v2/{A6,B5,B2}/high_coverage_qc`
- Regenerate: `results/iotj_c5_formal_regression_20260713_v2_summary`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- Consumes the frozen ECS inputs already recorded in each `suite_manifest.json`.
- Produces the expanded formal CSV and Markdown report.

- [ ] **Step 1: Sync the three changed runtime scripts to ECS**

Run the repository's established SSH/SCP path to place these files under `/root/GAPS`:

```text
run_source_augmented_target_ridge_eval.py
scripts/evaluate_iotj_high_coverage_qc.py
scripts/run_iotj_c5_regression_suite.py
scripts/summarize_iotj_c5_formal_regression.py
```

Verify on ECS: `cd /root/GAPS && /root/gaps_env/bin/python -m pytest tests/test_iotj_c5_regression_suite.py tests/test_iotj_high_coverage_qc.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Re-run only deterministic H8 and QC stages for A6/B5/B2**

For each classifier, reuse the paths from its frozen `suite_manifest.json`; execute H8 first and QC second with the additional `--h8-test-oracle` argument. Do not rerun classifier training or input extraction.

Expected per classifier:

```text
h8_no_rescue/target_predictions_plus_source_preds.csv: 1360 rows, unchanged actual prediction
h8_no_rescue/target_predictions_plus_source_preds_oracle_route.csv: 1360 rows
high_coverage_qc/operational_summary.json: FULL, HC95, HC90 with all new metrics
```

- [ ] **Step 3: Pull the regenerated derived artifacts back to the local workspace**

Retrieve only the three `h8_no_rescue`, three `high_coverage_qc`, and updated suite manifests. Preserve unrelated local result files.

- [ ] **Step 4: Rebuild the consolidated summary locally from ECS outputs**

Run:

```powershell
python scripts/summarize_iotj_c5_formal_regression.py --run-root results/iotj_c5_formal_regression_20260713_v2 --classifiers A6,B5,B2 --output-dir results/iotj_c5_formal_regression_20260713_v2_summary
```

Expected: 9 QC rows and a report containing the expanded columns.

- [ ] **Step 5: Verify frozen-result and oracle invariants**

Check:

```text
A6/B5/B2 FULL actual RMSE remains 28.0143907617 / 17.4473315022 / 14.6563580788.
FULL actual Accepted equals actual Nonreject for RMSE and NRMSE.
FULL oracle Accepted equals oracle Nonreject and N=1360.
HC95/HC90 accept/review/reject counts remain unchanged.
All oracle values are finite and every oracle CSV joins one-to-one to test_scored.csv.
```

- [ ] **Step 6: Update the experiment notebook with methods, results, and interpretation**

Record the exact ECS command provenance, FULL/HC95/HC90 table, the distinction between `S_CC` and forced oracle-route, and the fact that QC decisions remain deployment-visible and unchanged.

- [ ] **Step 7: Run final verification and commit derived evidence**

Run:

```powershell
python -m pytest tests/test_iotj_c5_regression_suite.py tests/test_iotj_high_coverage_qc.py -q --basetemp .tmp_pytest_qc_oracle_final
git diff --check
```

Expected: all tests pass; no whitespace errors. Commit only the intended code, tests, notebook, and compact summary artifacts, leaving unrelated dirty files untouched.
