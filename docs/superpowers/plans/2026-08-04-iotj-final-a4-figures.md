# IoT-J Final A4 Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build audited C5 A4 regression, QC, deployment evidence and publication-ready Fig. 5--Fig. 8 without retraining classification or changing frozen results.

**Architecture:** A fail-closed replay module consumes the frozen A4 checkpoint, calibration/test arrays, and source-prior manifests, then emits canonical window records and summaries. A separate QC module fits calibration-only risk thresholds and random references. A plotting module consumes only the new CSVs plus frozen budget/system tables.

**Tech Stack:** Python 3, PyTorch inference, NumPy, CSV/JSON, Matplotlib, pytest, existing GAPS runtime and evaluation helpers.

## Global Constraints

- Seed 42 only for the final replay; random QC seed is 20260804.
- No source Flower retraining, classifier retraining, hyperparameter search, or target-test checkpoint/model/threshold selection.
- Existing assets are read-only; output is a new non-empty-protected directory.
- C3/C4 A4 identity remains blocked unless an immutable matching endpoint is found.
- Figure exports are PDF and 600-DPI PNG with source-data CSVs.

---

### Task 1: Freeze classifier identity and replay contracts

**Files:**
- Create: `scripts/finalize_iotj_a4_end_to_end.py`
- Test: `tests/test_iotj_a4_end_to_end.py`

**Interfaces:**
- Consumes: `FCL-E4-A4/locked_run_spec.json`, `final_evaluation_C5.json`, `server_round_025_adapted.pth`.
- Produces: `checkpoint_state_fingerprint(path) -> str`, `build_classifier_manifest(...) -> dict`.

- [ ] Write tests asserting ordered state-content fingerprint stability, C5 protocol fields, C3/C4 blocked status, and refusal to overwrite a non-empty destination.
- [ ] Run `pytest tests/test_iotj_a4_end_to_end.py -v` and confirm failure because the module is missing.
- [ ] Implement the minimal identity and output guards.
- [ ] Run the targeted tests and confirm they pass.

### Task 2: Implement the 83-D A4 regression replay

**Files:**
- Modify: `scripts/finalize_iotj_a4_end_to_end.py`
- Test: `tests/test_iotj_a4_end_to_end.py`

**Interfaces:**
- Produces: `fit_final_regressors(...)`, `apply_final_regressors(...)`, `summarize_regression_records(...)`.
- Output columns include three fixed prediction keys and all required route/error/risk inputs.

- [ ] Add failing fixture tests for 83/84/86 input dimensions, calibration-only alpha selection, route-correct versus end-to-end masks, canonical row alignment, and non-finite rejection.
- [ ] Run the targeted tests and confirm each new behavior fails for the intended reason.
- [ ] Implement the three fixed variants by reusing the existing feature/source-prior helpers.
- [ ] Run targeted tests and `python -m compileall scripts/finalize_iotj_a4_end_to_end.py`.
- [ ] Execute the C5 calibration lock and test replay into `results/iotj_final_end_to_end_a4_20260804/regression`.

### Task 3: Implement QC curves and random reference

**Files:**
- Create: `scripts/finalize_iotj_a4_qc.py`
- Test: `tests/test_iotj_a4_qc.py`

**Interfaces:**
- Consumes: final calibration/test record CSVs.
- Produces: `qc_coverage_curve.csv`, `qc_random_reference.csv`, `qc_operating_points.csv`, and policy-annotated HC90/HC95 records.

- [ ] Add failing tests for 13 exact coverage values, calibration-only thresholds, 1,000 deterministic random repetitions, NRMSE normalization, capture-rate definitions, and HC90/HC95 semantics.
- [ ] Run the targeted QC tests and confirm the expected failures.
- [ ] Implement the fixed risk score, calibration threshold application, and random reference.
- [ ] Run targeted tests and compileall.
- [ ] Execute QC into `results/iotj_final_end_to_end_a4_20260804/qc`.

### Task 4: Build deployment manifests and Fig. 5--Fig. 8

**Files:**
- Create: `scripts/plot_iotj_final_a4_figures.py`
- Test: `tests/test_plot_iotj_final_a4_figures.py`

**Interfaces:**
- Consumes: new regression/QC CSVs and frozen calibration-budget/system benchmark CSVs.
- Produces: `fig05_concentration_estimation`, `fig06_source_prior_calibration_budget`, `fig07_qc_coverage_random_reference`, `fig08_communication_pi5_physical_validation` in PDF/PNG with source-data CSVs.

- [ ] Add failing tests for input-schema validation, no mixed-protocol pooling, expected panel/source-data files, and non-empty PDF/PNG exports.
- [ ] Run the plotting tests and confirm failure because the module is missing.
- [ ] Implement IEEE-consistent plotting and deployment manifest/package summaries.
- [ ] Run targeted tests and generate all four figures.
- [ ] Visually inspect the four PNGs at final size and correct only presentation defects without changing data or protocol.

### Task 5: Audit, analysis, hashes, and publication

**Files:**
- Create under the new result root: `EXPERIMENT_AUDIT.md`, `RESULT_ANALYSIS.md`, `protocol_manifest.json`, `sha256_index.json`, `FIGURE_CAPTIONS.md`.

- [ ] Run all new targeted tests, relevant existing regression/QC/system tests, strict audit, and compileall.
- [ ] Verify existing classification result hashes and git-tracked contents are unchanged except watcher log timestamps already present before this work.
- [ ] Record reported versus recomputed metrics and all blocked/unknown fields.
- [ ] Stage only the new scripts, tests, docs, and final result root; exclude pre-existing watcher log modifications.
- [ ] Commit and push `codex/iotj-final-classification-le1`.

