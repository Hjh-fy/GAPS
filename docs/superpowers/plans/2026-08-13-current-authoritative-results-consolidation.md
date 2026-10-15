# Current Authoritative Results Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a new, provenance-indexed canonical summary of the current authoritative regression and confidence-QC evidence without changing historical result roots or executing any model training.

**Architecture:** A single deterministic consolidation script reads only sealed R1/Q0/Q1/R2 prediction and summary artifacts, recomputes final metrics from authorized R1 predictions, locks HC90/HC95 from calibration confidence alone, and writes a new versioned result directory. A focused pytest module exercises the pure metric, threshold, and status-separation helpers; the runtime script rejects an existing destination.

**Tech Stack:** Python 3, NumPy, CSV/JSON, pytest.

## Global Constraints

- Read existing results, manifests, locks, dataset, checkpoints, and historical assets only; never delete, rename, overwrite, or train.
- The only new result root is `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/`.
- Final regression is R84_CONCAT from authoritative R1 predictions; old `04_regression_final.csv` and `FINAL_RESULT_MASTER_TABLE.csv` are comparison-only.
- Final QC is fixed `CONFIDENCE_QC_FINAL`, risk `1-max_g p(g|x)`, ACCEPT iff risk is at or below a calibration-only threshold.
- HC90/HC95 use deterministic stable empirical quantile locking from calibration identities before test statistics are read.
- Keep `HISTORICAL_SUPERSEDED` distinct from `INVALID_DO_NOT_CITE`.
- No new bootstrap, no model inference, no algorithm/QC policy search, and no use of test rows for threshold selection.

---

### Task 1: Authoritative-only consolidation helpers and tests

**Files:**
- Create: `scripts/consolidate_current_authoritative_results.py`
- Create: `tests/test_current_authoritative_results_consolidation.py`

**Interfaces:**
- Produces `metric_row(truth, prediction, gas_range, ...) -> dict`, `calibration_threshold(risks, identities, nominal) -> dict`, and `summarize_selective(...) -> dict`.
- Produces `main()` with an explicit new output directory and fail-closed no-overwrite behavior.

- [ ] **Step 1: Write failing tests** for recomputed RMSE/MAE/NRMSE/R2/Bias/absolute-error percentiles, deterministic calibration-only threshold ties, ACCEPT/REJECT semantics, micro vs macro pooled identity, and status separation.
- [ ] **Step 2: Run the focused pytest module** and verify RED because the module is absent.
- [ ] **Step 3: Implement only the helpers and artifact readers** needed for authoritative R1/Q0/Q1/R2 and historical comparison extraction.
- [ ] **Step 4: Run focused pytest** and verify GREEN.
- [ ] **Step 5: Commit** the code and tests with `feat: consolidate current authoritative regression and QC metrics`.

### Task 2: Produce the immutable summary and document evidence status

**Files:**
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/CURRENT_AUTHORITATIVE_RESULTS_SUMMARY_20260812.md`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/CURRENT_AUTHORITATIVE_EVIDENCE_INVENTORY.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_R84_REGRESSION_METRICS.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_83D_VS_R84_COMPARISON.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/HISTORICAL_VS_AUTHORITATIVE_REGRESSION.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_CONFIDENCE_QC_THRESHOLDS.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_CONFIDENCE_QC_METRICS.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_QC_RISK_COVERAGE_SUMMARY.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/HISTORICAL_QC_EVIDENCE_INDEX.csv`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/RESULT_PROVENANCE_INDEX.json`
- Create: `results/iotj_canonical_v1_final/current_authoritative_summary_20260812/sha256_index.json`

- [ ] **Step 1: Confirm the destination does not exist** and record SHA256 values for every source artifact.
- [ ] **Step 2: Run the consolidator exactly once**; it must write only the new summary root.
- [ ] **Step 3: Verify** source artifact hashes are unchanged, output index covers all new files, all values trace to source paths, and Q1 interval coverage is clearly separate from selective-output coverage.
- [ ] **Step 4: Run focused pytest and compileall.**
- [ ] **Step 5: Commit only source/test/docs metadata** (formal summary remains an ignored result asset) with `docs: summarize canonical final and historical evidence`, then push the requested branch.

## Spec coverage review

- S0 inventory: Task 2 inventory and provenance JSON.
- S1/S2 regression statistics and existing bootstrap reuse: Task 1 readers and Task 2 CSVs.
- S3/S8 historical preservation: Task 2 comparison/index records only source paths and statuses.
- S4-S7 confidence policy, calibration-only HC locks, test evaluation, risk summary: Task 1 helpers and Task 2 CSV/report.
- S9/S10 final report, audit, and SHA index: Task 2.
- No placeholder or unregistered algorithm work is included.
