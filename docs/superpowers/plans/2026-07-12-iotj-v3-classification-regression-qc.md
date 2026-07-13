# IoT-J V3 Classification, Regression, and High-Coverage QC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the four server-adaptation definition problems without rewriting the frozen v2 evidence, run a causal B1-B5 classification screen on the real cloud-edge topology, rebuild formal C5 regression from A6 and B5, and demonstrate that deployment-visible QC catches concentrated failures while retaining about 90% to 95% automatic coverage.

**Architecture:** Preserve A0-A7 v2 as immutable historical run identities. Treat A6 as the semantic adaptation reference, define B1-B4 as one-correction-at-a-time additions, and predefine B5 as the full corrected combination before any B-run test result is opened. Rebuild identical C5 calibration/test streams for A6 and B5, fit target-personalized regression on Alibaba Cloud ECS, then fit and freeze a calibration-only QC policy whose primary operating point targets 95% acceptance.

**Tech Stack:** Python 3, PyTorch, Flower, NumPy, scikit-learn Ridge/MLP, pytest, Alibaba Cloud ECS, physical Raspberry Pi, Windows PC, SSH/SCP.

## Global Constraints

- Primary protocol is exactly source `C1,C2` to target `C5`; C3/C4 are not target domains.
- Dataset remains `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` with the advisor-approved window/class/concentration stratified split.
- C5 contains exactly 320 calibration and 1360 sealed test windows. Test labels never select losses, checkpoints, experts, risk features, or thresholds.
- Classification training remains 25 rounds, 5 local epochs, batch size 32, client LR `5e-4`, 100 server adaptation steps, and server LR `5e-4`.
- Reportable classification training runs only on Alibaba Cloud ECS plus the physical Pi C1 and PC C2 clients. No local simulated training.
- Regression fitting and QC fitting run on ECS because they are server-side target calibration stages. Final inference is replayed on PC and Pi.
- Existing `results/iotj_classification_ablation_20260711_v2r1` artifacts are read-only. Corrected runs use new `v3` names and directories.
- B5 is fixed before execution as `A6 semantic core + CORAL + conventional MMD-squared + cross-domain same-class/same-phase MMD-squared + corrected Wasserstein feature objective`. It is not assembled after inspecting test results.
- Pairwise L2 between detached client prototypes is excluded from all B groups by setting its DA weight to zero. It remains an offline diagnostic, not a training contribution.
- The primary QC point targets 95% automatic acceptance; 90% is a stronger-filter secondary point and 100% is the no-QC system baseline. A 98% point is diagnostic only because the 80-row calibration-validation set has coarse tail resolution.
- Every reported selective result includes N, realized coverage, RMSE, NRMSE, MAE, P90AE, route-error recall, high-error recall, and false flag rate among class-correct rows.
- Update `docs/experiments/iotj_system_experiment_notebook.md` after every implementation, launch, recovery, evaluation, and decision gate.

---

### Task 1: Version and Test Corrected Server Adaptation

**Files:**
- Modify: `utils.py`
- Modify: `gaps_flower/domain_adaptation.py`
- Modify: `gaps_flower/strategy.py`
- Modify: `gaps_flower/server_app.py`
- Create: `tests/test_flower_da_v3_corrections.py`

**Interfaces:**
- Produces: `compute_mmd2(features1, features2, seed=42) -> Tensor`
- Produces DA modes: `mmd_objective={legacy_quartic,mmd2}`, `stage_alignment={legacy_intra_domain,cross_domain_same_class_phase}`, and `adv_feature_objective={legacy_grl_plus,wasserstein_min}`
- Keeps all legacy defaults unchanged so an old v2 manifest still means the original implementation.

- [x] **Step 1: Write failing MMD tests**

Add tests proving that `compute_mmd2` returns the kernel discrepancy itself, has finite nonzero gradients under a shifted distribution, and does not change `torch.random.get_rng_state()`.

- [x] **Step 2: Verify the MMD tests fail for the missing API**

Run: `python -m pytest tests/test_flower_da_v3_corrections.py -q --basetemp .tmp_pytest_iotj_v3_mmd`

Expected: FAIL because `compute_mmd2` is not defined.

- [x] **Step 3: Implement `compute_mmd2` without global RNG mutation**

Use the existing Gaussian-kernel biased empirical MMD-squared estimator. For more than 1000 rows, sample with a local `torch.Generator`; do not call `torch.manual_seed`. Leave `compute_mmd` intact for frozen legacy semantics.

- [x] **Step 4: Write failing stage-alignment tests**

Construct source and target tensors with two classes and phases. Assert the corrected loss pairs only source `(class,phase)` with target `(class,phase)`, ignores unmatched cells, and backpropagates to both domains.

- [x] **Step 5: Implement corrected stage alignment**

For every class and phase in the source/target intersection, append `compute_mmd2(source[c,p], target[c,p])` when both sides contain at least two rows, then average valid terms. Preserve the old within-domain cross-phase implementation behind `legacy_intra_domain`.

- [x] **Step 6: Write failing adversarial-direction tests**

With a frozen one-dimensional critic, take one SGD step on source and target feature parameters and assert that `D(source)-D(target)` decreases under `wasserstein_min`. Also assert critic parameters receive no feature-step gradients.

- [x] **Step 7: Implement the corrected Wasserstein feature objective**

Keep critic optimization as minimization of `-(D_s-D_t)+GP`. For the encoder step, temporarily freeze critic parameters, do not apply GRL, and return `D_s-D_t` for minimization. Preserve `legacy_grl_plus` for v2 replay only.

- [x] **Step 8: Prove the detached prototype pair term is gradient-free**

Add a test showing that changing its numeric weight changes logged total loss but not any model, semantic-prototype, or residual gradient. Add an explicit diagnostic flag in summaries and require B manifests to set the weight to zero.

- [x] **Step 9: Run focused and existing Flower contracts**

Run: `python -m pytest tests/test_flower_da_v3_corrections.py tests/test_flower_classification_contract.py -q --basetemp .tmp_pytest_iotj_v3_da`

Expected: PASS with no failed tests.

### Task 2: Generate the Predeclared B1-B5 Classification Suite

**Files:**
- Modify: `scripts/generate_iotj_classification_ablation_commands.py`
- Modify: `scripts/run_iotj_classification_cloud_edge.py`
- Modify: `scripts/summarize_iotj_classification_ablation.py`
- Modify: `tests/test_flower_classification_contract.py`
- Modify: `tests/test_iotj_classification_summary.py`

**Interfaces:**
- Produces manifests under `results/iotj_classification_ablation_20260712_v3_commands`
- Produces training outputs under `results/iotj_classification_ablation_20260712_v3`

- [x] **Step 1: Write failing B-suite manifest tests**

Assert the following exact groups and weights:

| ID | Additions to A6 semantic core | Corrected modes |
|---|---|---|
| B1 | CORAL `0.5` | pair-L2 `0` |
| B2 | global MMD `0.5`, class MMD `0.5` | `mmd2`; pair-L2 `0` |
| B3 | stage MMD `0.2` | `cross_domain_same_class_phase`; pair-L2 `0` |
| B4 | adversarial `0.5` | `wasserstein_min`; pair-L2 `0` |
| B5 | B1+B2+B3+B4 | all corrected modes; pair-L2 `0` |

All B groups keep A6's prototype anchor `0.3`, prototype fit `0.05`, consistency `2.0`, residual `0.1`, selective aggregation, and `proto_replay` client profile.

- [x] **Step 2: Verify manifest tests fail**

Run: `python -m pytest tests/test_flower_classification_contract.py -q --basetemp .tmp_pytest_iotj_v3_manifest`

Expected: FAIL because B1-B5 are absent.

- [x] **Step 3: Add versioned B specs and CLI propagation**

Keep the v2 default generation list unchanged. Add an explicit v3 suite selector and record corrected-mode fields in `command_manifest.json` and `run_config.json`.

- [x] **Step 4: Extend summary parsing for B IDs**

The evaluator must parse A0/A0T/A4S and B1-B5 without silently omitting any group. Add an expected-group option that fails when a requested B run is missing.

- [x] **Step 5: Generate and audit seed-42 commands**

Run: `python scripts/generate_iotj_classification_ablation_commands.py --suite v3 --output-root results/iotj_classification_ablation_20260712_v3_commands --results-root results/iotj_classification_ablation_20260712_v3`

Expected: five scheduled B runs, all C1/C2 to C5, with no A7 leave-one-out jobs.

- [x] **Step 6: Dry-run the real controller**

Run: `python scripts/run_iotj_classification_cloud_edge.py --command-root results/iotj_classification_ablation_20260712_v3_commands --results-root results/iotj_classification_ablation_20260712_v3 --local-results-root results/iotj_classification_ablation_20260712_v3 --local-log-root results/iotj_classification_ablation_20260712_v3_local_client_logs --groups B1,B2,B3,B4,B5 --seed 42 --dry-run`

Expected: exactly B1-B5 resolve to frozen C12-to-C5 manifests.

### Task 3: Run and Review B1-B5 on the Real Topology

**Files:**
- Read/write results only: `results/iotj_classification_ablation_20260712_v3*`
- Modify after recovery: `docs/experiments/iotj_system_experiment_notebook.md`

**Interfaces:**
- Consumes the Task 2 command root and corrected runtime
- Produces five round-25 checkpoints plus a classification summary

- [x] **Step 1: Preflight all three machines**

Verify ECS has C5 calibration arrays and no active Flower server, Pi `gaps@192.168.31.184` has C1 train arrays, PC has C2 train arrays, and all machines import the synchronized corrected modes.

- [x] **Step 2: Run B1-B5 sequentially**

Run the Task 2 controller command without `--dry-run`. Do not run jobs in parallel on the same three-machine topology.

- [x] **Step 3: Recover and validate each run before advancing**

Require round 25 history, `server_latest_adapted.pth`, run config, domain-adaptation diagnostics, and zero partial-run overwrite. Record wall time and checkpoint SHA-256.

- [x] **Step 4: Summarize the corrected screen**

Run: `python -m scripts.summarize_iotj_classification_ablation --run-root results/iotj_classification_ablation_20260712_v3 --output-root results/iotj_classification_ablation_20260712_v3_summary --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --expected-groups B1,B2,B3,B4,B5`

Report A6, A7-v2, and B1-B5 together, but label A7-v2 as legacy. B5 remains the predefined full corrected method regardless of its test rank.

### Task 4: Rebuild Formal A6 and B5 C5 Regression on ECS

**Files:**
- Modify: `gaps_flower/evaluate_regression_pipeline.py`
- Modify: `scripts/build_iotj_c5_regression_inputs.py`
- Create: `scripts/run_iotj_c5_regression_suite.py`
- Create: `scripts/run_iotj_c5_regression_cloud.py`
- Create: `tests/test_iotj_c5_regression_suite.py`

**Interfaces:**
- Consumes A6 and B5 classifier checkpoints plus the frozen C1/C2 R3aK16 source reference
- Produces aligned A6/B5 C5 calibration/test streams and R0-R7 tables

- [x] **Step 1: Write failing risk-schema and row-contract tests**

Require exact C5 counts `320/1360`, unique `(split,sample_index)`, predicted-route normalization for deployment-risk fields, explicit legacy-risk names, and no generic `risk_score` alias from the leaked historical composite.

- [x] **Step 2: Add deployment-visible pipeline fields**

Keep true-class range only for evaluation metrics. Compute deployment-risk normalization with predicted `route_class` and emit names prefixed `deployment_risk_`; preserve the old value only as `legacy_true_range_composite_risk`.

- [x] **Step 3: Build one regression-suite orchestrator**

For each classifier ID, execute input building, H2.3+, H8 with `--disable-c4-rescue`, row merging, simple predicted-CO gate, and oracle diagnostic. Hyperparameters are selected only from concentration-stratified calibration fit/validation rows.

- [x] **Step 4: Emit the exact regression ladder**

`R0` source R3aK16, `R1` C5 rich Ridge, `R2` C5 H2.3 MLP, `R3` H2.3+, `R4` fixed H8, `R5` predicted-CO gate, `R6` deployment-visible calibrated selector, and `R7` per-row oracle diagnostic.

- [x] **Step 5: Run formal regression on ECS**

Use the cloud runner to synchronize exact scripts, run A6 first, then B5, and recover outputs. Do not fit these models locally. Report `S_ALL`, `S_CC`, classification-wrong, per-gas, N, and coverage for every R group.

### Task 5: Build High-Coverage Deployment-Visible QC

**Files:**
- Create: `scripts/evaluate_iotj_high_coverage_qc.py`
- Create: `tests/test_iotj_high_coverage_qc.py`
- Reuse: `gaps_deploy/evaluate_qc_budget_curve.py`

**Interfaces:**
- Produces `fit_risk_reference(calibration_rows, calibration_features) -> RiskReference`
- Produces `score_deployment_rows(rows, features, reference) -> rows`
- Produces `fit_workpoints(calibration_validation_rows, score) -> policy.json`
- Produces test-only evaluation tables without test-driven threshold selection

- [x] **Step 1: Write truth-invariance tests**

Score the same deployment row twice after changing `true_class`, `true_ppm`, `class_correct`, `route_correct`, and error columns. Assert every `deployment_risk_*` value and QC decision is identical.

- [x] **Step 2: Implement risk components**

Use only predicted class probabilities, confidence margin/entropy, predicted-class/phase diagonal-Mahalanobis prototype distance, leave-one-out calibration support distance, H2.3/H8 disagreement normalized by predicted-class range, and spread among source-reference predictions. Robustly percentile-normalize components on calibration only.

- [x] **Step 3: Predeclare score ablations**

Compare confidence-only, prototype/support-only, expert-disagreement-only, and the full mean-of-three-calibrated-family-risks composite. The mean was frozen after the historical F2 smoke exposed ECDF saturation under a raw maximum, before formal A6/B5 evaluation. Select the score family using calibration-validation only with the lexicographic objective: maximize route/high-error capture at 5% review budget, then minimize accepted P90AE, then minimize false flags among class-correct rows.

- [x] **Step 4: Freeze high-coverage workpoints**

Create three-state policies from calibration-validation quantiles:

| Workpoint | Target accept | Target review | Target reject | Role |
|---|---:|---:|---:|---|
| Full | 100% | 0% | 0% | no-QC baseline |
| HC95 | 95% | 4% | 1% | primary system point |
| HC90 | 90% | 8% | 2% | stronger-filter secondary point |

Test reports realized rather than forced coverage. If the small calibration tail causes tied thresholds, use deterministic `(risk,sample_index)` ordering and record the realized calibration counts.

- [x] **Step 5: Evaluate two distinct evidence views**

Deployment view applies frozen HC95/HC90 thresholds to test. Ranking view evaluates exact test coverages `100,98,95,92.5,90` only to measure ordering quality and is labeled non-operational.

- [x] **Step 6: Compare matched random rejection**

At every realized flagged N, run 1000 seeded random selections. Report accepted RMSE distribution, route-error recall, high-error recall, and enrichment of the proposed risk over random.

- [x] **Step 7: Report the paper-facing high-coverage story**

At 100% coverage report actual-route `S_ALL` and independent correct-route `S_CC`. At HC95 and HC90 report accept/review/reject N, automatic yield, nonreject coverage, accepted errors, fraction of all route errors captured by review+reject, and false flag rate among class-correct windows.

### Task 6: Multi-Seed Confirmation and Downstream Stability

**Files:**
- Modify: `scripts/generate_iotj_classification_ablation_commands.py`
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`

- [ ] **Step 1: Run seeds 43-46 for the minimum confirmation set**

Run A0, A0T, A6, and B5 with paired seeds 43-46 on the same real topology. Do not run A4/A4S unless selective aggregation is retained as an independent paper contribution.

- [ ] **Step 2: Summarize paired uncertainty**

Report five seed values, mean, sample SD, and paired B5-minus-A6/B5-minus-A0T differences. Do not claim strong significance from five seeds.

- [ ] **Step 3: Repeat downstream regression/QC for B5 seeds**

Run the formal regression suite and frozen QC construction for B5 seeds 43-46. Use the same risk schema and workpoint targets; report variability in realized coverage, accepted RMSE, route-error capture, and automatic yield.

### Task 7: Edge Replay, Documentation, and Evidence Freeze

**Files:**
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`
- Modify: `docs/paper/iotj_system_methodology_20260711.zh.md`
- Modify: `代码文件介绍.md`
- Modify: `scripts/validate_final_deployment_bundle.py`

- [ ] **Step 1: Replay the frozen selected pipeline on Pi and PC**

Require per-window class, regression, risk, and QC parity with ECS outputs. Report p50/p95 latency, throughput, peak RSS, model/policy size, and communication overhead.

- [ ] **Step 2: Freeze the two-line result story**

Capability line: classification metrics plus 100%-coverage `S_CC` regression. System line: 100%-coverage actual-route `S_ALL`, followed by HC95 primary and HC90 secondary selective results.

- [ ] **Step 3: Update documentation and claim boundaries**

Document corrected formulas, exact B definitions, the distinction between operational thresholds and fixed-coverage ranking curves, and the fact that QC detects rather than repairs risky windows.

- [ ] **Step 4: Freeze lightweight evidence**

Store manifests, hashes, CSV summaries, policy JSON, plots, and claim-to-evidence mapping. Do not commit datasets, checkpoints, or raw large logs.

## Execution Order and Review Gates

1. Tasks 1-2 must pass before any remote code synchronization.
2. Task 3 runs B1-B5 sequentially and stops on the first incomplete or non-finite run.
3. Task 4 starts only after B5 round 25 is recovered; A6 uses the already frozen round-25 checkpoint.
4. Task 5 starts only after aligned A6/B5 regression validation/test streams exist and leakage tests pass.
5. Task 6 begins after the seed-42 classification/regression/QC review, so expensive repetitions preserve only the final claims.
6. Task 7 is required before calling the pipeline deployment-ready or freezing paper tables.

## Predeclared Interpretation

- If B5 consistently exceeds A6 and A0T, claim complementarity between semantic memory and corrected distribution alignment.
- If B5 is close to A6, use A6 as the compact method and B1-B5 as evidence that extra DA complexity is unnecessary.
- If A0T matches A6/B5, state that target label budget explains most classification gain and move the main contribution toward target-personalized regression, high-coverage QC, and real cloud-edge closure.
- QC is successful only when HC95 captures route/high-regression-risk windows substantially better than matched random rejection while retaining approximately 90% to 95% automatic yield. Lower-error results obtained by rejecting a large fraction are appendix diagnostics, not the headline.
