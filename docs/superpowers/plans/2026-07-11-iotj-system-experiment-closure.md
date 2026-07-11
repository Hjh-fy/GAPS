# IoT-J System Experiment Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze, ablate, validate, and deploy the current GAPS mainline as a reliability-aware personalized federated IoT gas sensing system with a complete classification, regression, selective-QC, source-target generalization, and cloud-edge evidence chain.

**Architecture:** Keep the current four-stage mainline: Flower semantic classification, calibration-assisted server adaptation, target-personalized dual-stream regression, and reliability-aware expert/QC decisions. Use one frozen C12-to-C5 protocol for all main ablations, use source-count C5 matrix runs only for generalization, and report model capability and end-to-end selective performance as separate result lines.

**Tech Stack:** Python 3, PyTorch, Flower, scikit-learn Ridge/MLP, NumPy, pandas-compatible CSV artifacts, pytest, PowerShell controller, Alibaba Cloud ECS, Raspberry Pi, Windows PC.

## Global Constraints

- Primary data root: `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`; only active client directories C1, C2, and C5 belong to the primary protocol.
- Primary source clients: `C1,C2`; the only primary target client is `C5`. C3 and C4 must never be loaded as target calibration/test clients or included in primary target metrics.
- Keep the advisor-approved window-level stratified split unchanged: target calibration/test = `20%/80%`, balanced by gas class and concentration; calibration-fit/calibration-validation = `75%/25%` inside calibration.
- Freeze data split seed at `42`; vary training seed only over `42,43,44,45,46`.
- Classification training is fixed at 25 global rounds, 5 local epochs, batch size 32, Adam learning rate `5e-4`, TCN input `(100,8)`, server DA 100 steps/round, server optimizer learning rate `5e-4`. Target CE is zero for all proposed GAPS groups; the explicitly labeled A0T calibration-supervised baseline alone uses target CE weight 1.0.
- Use final round 25 for primary reporting. Test performance must not select checkpoints, heads, blend weights, expert thresholds, or QC thresholds.
- Main classifier metrics: accuracy, macro-F1, per-class recall, NLL, ECE, and confusion matrix.
- Main regression metrics: RMSE, class-range NRMSE, MAE, P90AE, bias, R2, N, and slice coverage. R2 is undefined and reported as blank for constant-label slices.
- Keep three primary regression slices separate: `S_ALL`, `S_CC={route_class=true_class}`, and `S_AR={qc_decision in accept,review}`. `S_AR intersect S_CC` is diagnostic only.
- R3aK16 is a source-regression reference, not the final regression mainline. C5 H2.3+, fixed C5 H8, the simple predicted-CO gate, and P4 are predeclared candidates. The final deployable policy must be chosen with calibration-validation only; test performance cannot decide which candidate becomes the main method. C4 rescue is outside the primary protocol.
- Label server adaptation as `calibration-assisted` because target class/phase labels are used by class-conditional alignment even when target CE weight is zero.
- Never call `server_latest.pth` from a `use_adapted_as_global=true` run a no-DA baseline. A no-DA baseline must be trained in an independent run with server adaptation disabled.
- Every run writes `run_config.json`, immutable input hashes, code revision, seed, wall time, checkpoint hashes, and evaluation artifact paths.
- Existing user changes and historical result folders must not be modified or deleted.
- Maintain `docs/experiments/iotj_system_experiment_notebook.md` after every implementation, training, evaluation, and review step. Record commands, immutable inputs, outputs, observations, failures, decisions, and next actions before a task is marked complete.
- All reportable training runs must use the real cloud-edge topology: Alibaba Cloud ECS Flower server and physical Raspberry Pi/PC source clients. Local execution is limited to unit tests, frozen-artifact analysis, command generation, and numerical parity checks; local simulated training cannot enter paper performance tables.

---

### Task 1: Freeze Inputs and Recover Cloud Matrix Artifacts

**Files:**
- Create: `scripts/audit_iotj_experiment_inputs.py`
- Create: `tests/test_iotj_experiment_input_audit.py`
- Create: `results/iotj_experiment_freeze_20260711/input_manifest.json`

**Interfaces:**
- Consumes: primary C12-to-C5 dataset, F2 checkpoint/config, and later C5-only H2.3+/H8/P4 streams
- Produces: `audit_inputs(paths: Sequence[Path]) -> dict`, local cloud-artifact inventory, SHA-256 manifest, missing-artifact report

- [ ] **Step 1: Write failing manifest tests**

Assert that the audit records resolved path, existence, byte size, SHA-256, dataset role, and refuses a matrix run that lacks `server_latest_adapted.pth` or `run_config.json`.

- [ ] **Step 2: Run the focused tests**

Run: `python -m pytest tests/test_iotj_experiment_input_audit.py -q`

Expected: FAIL because `audit_iotj_experiment_inputs.py` does not exist.

- [ ] **Step 3: Implement read-only input audit**

Use `hashlib.sha256` with chunked reads. Do not rewrite source artifacts. Record the primary split counts by client, class, concentration, and calibration/test role.

- [ ] **Step 4: Recover the clean matrix artifacts**

Copy `/root/GAPS/results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25/` to local `results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25/`, preserving layout and timestamps. Recover F1/F3/F4 only for source-count appendix analysis. Do not use F5 because it treats C3/C4 as targets; do not use reverse R1-R4 as primary-protocol evidence.

- [ ] **Step 5: Generate and verify the frozen manifest**

Run: `python scripts/audit_iotj_experiment_inputs.py --output results/iotj_experiment_freeze_20260711/input_manifest.json`

Expected: primary C12-to-C5 dataset and F2 run complete; C5 calibration/test counts and protocol metadata validated; every unavailable required artifact explicitly marked `missing`.

### Task 2: Correct the Metric Slice Contract

**Files:**
- Modify: `run_final_metric_consolidation_20260709.py`
- Modify: `tests/test_final_metric_consolidation.py`
- Reuse: `scripts/analyze_matrix_correct_class_regression.py`

**Interfaces:**
- Consumes: `threshold_guard_test_outputs.csv`
- Produces: `build_regression_slice_table(rows, pred_keys) -> list[dict]` with independent `all`, `class_correct`, `class_wrong`, `accepted_review`, and `accepted_review_class_correct` rows

- [ ] **Step 1: Replace the old intersection-only test**

Create four fixture rows so the test proves that `class_correct` is selected before QC and has a larger N than `accepted_review_class_correct`.

- [ ] **Step 2: Verify the test fails against current behavior**

Run: `python -m pytest tests/test_final_metric_consolidation.py::test_classification_correct_slice_is_independent_of_qc -q`

Expected: FAIL because current code computes `S_AR intersect S_CC` only.

- [ ] **Step 3: Implement independent slices using the shared metric definition**

Reuse `run_regression_head_ablation.metrics` so NRMSE remains the RMS of error divided by the true class concentration range `{112.5,225,112.5,225}`.

- [ ] **Step 4: Regenerate a contract-check table without promoting the old protocol**

Use a deterministic fixture or the superseded F6 output only to prove the slice implementation. Label every F6-derived number historical. Do not publish or freeze new primary values until Task 5 has produced aligned F2 C12-to-C5 classifier/regression streams for C5 only.

- [ ] **Step 5: Run focused and matrix-slice tests**

Run: `python -m pytest tests/test_final_metric_consolidation.py tests/test_matrix_correct_class_regression_analysis.py -q`

Expected: PASS.

### Task 3: Add Reproducible Classification Ablation Profiles

**Files:**
- Modify: `gaps_flower/task.py`
- Modify: `gaps_flower/client_app.py`
- Modify: `gaps_flower/server_app.py`
- Modify: `scripts/remote_launch_flower_client_clean.py`
- Modify: `scripts/remote_launch_flower_server_clean.py`
- Create: `scripts/generate_iotj_classification_ablation_commands.py`
- Modify: `tests/test_flower_classification_contract.py`

**Interfaces:**
- Produces causal client profiles `ce_only`, `align_only`, `replay_only`, `align_replay`, and `proto_replay`; legacy `proto_only` remains supported but is not used in the primary v2 matrix
- Produces CLI option `--seed` that changes model/training randomness without changing the frozen dataset split
- Produces one command manifest per ablation ID and seed

- [ ] **Step 1: Add profile contract tests**

Assert exact switches: `ce_only=(0,0,0)`, `align_only=(1,0,0)`, `replay_only=(0,1,0)`, `align_replay=(1,1,0)`, and `proto_replay=(1,1,1)` for `(align,replay,device-residual statistics)`, with regression loss disabled in every profile.

- [ ] **Step 2: Implement profiles and seed propagation**

Keep `smoke` and `strong_cls` backward compatible. Record the effective switches and seed in client metrics and server `run_config.json`.
Replay distillation must be inactive in round 1 because no previous server model exists. Cache the round-1 incoming server state and activate the frozen previous-round teacher from round 2 onward.
CE-only and replay-only profiles must skip post-training prototype/variance passes and omit prototype JSON from Flower metrics. Alignment profiles upload class/phase prototypes but do not compute unused device-residual statistics. Device residuals are enabled only for semantic-DA groups that consume them; otherwise runtime and communication comparisons would be artificially inflated.

- [ ] **Step 3: Generate the independent comparison baselines**

Keep these out of the causal ablation table: `Local-C1`, `Local-C2`, pooled-source centralized TCN as a non-private upper reference, `FedAvg`, and `FedProx`. For FedProx, select `mu` from `0.001,0.01,0.1` using source validation only, then freeze it before target testing. Use the same backbone, batch size, client learning rate, local epochs, global rounds, source clients, and training seeds as C0/C7. Do not add SCAFFOLD unless its standard control-variate state and communication overhead can be implemented and audited exactly.

- [ ] **Step 4: Generate the seed-42 screening matrix**

Use these exact groups:

| ID | Client loss | Server adaptation |
|---|---|---|
| A0 | FedAvg CE only | disabled |
| A0T | CE only with FedAvg-equivalent parameter aggregation | source rehearsal CE + C5 calibration target CE 1.0; all alignment/semantic/stage terms disabled |
| A1 | GAPS aggregation, CE only | disabled; aggregation contract control only, not a full paper run |
| A2 | CE + class/phase prototype contrastive alignment | disabled; FedAvg-equivalent parameter aggregation |
| A3 | CE + previous-round feature distillation | disabled |
| A4 | CE + prototype alignment + distillation | disabled; FedAvg-equivalent parameter aggregation |
| A4S | A4 + selective aggregation | disabled |
| A5 | A4S | distribution DA: CORAL 0.5 + global MMD 0.5 + class MMD 0.5 + ADV 0.5; semantic/stage terms 0 |
| A6 | A4S + device-residual statistics | semantic DA: proto anchor 0.3 + proto fit 0.05 + consistency 2.0 + residual 0.1 + proto MMD 0.2; distribution/stage terms 0 |
| A7 | A4S | full distribution + semantic + stage-MMD fixed-DA configuration |

`A2-A4` must set `use_selective_agg=false`; otherwise client-loss effects are confounded with server aggregation. Prototype-MMD diagnostics are disabled in timing-comparable primary runs because they do not affect model updates. A separate offline diagnostics pass may compute them from frozen artifacts.

- [ ] **Step 5: Generate leave-one-group-out appendix runs**

Before full cloud training, compare A0 and A1 on identical synthetic client updates as a unit-level aggregation contract. If a short training smoke check is still required, run it through the real Alibaba Cloud ECS plus physical Raspberry Pi/PC Flower topology, not a local simulation. If aggregated model tensors are identical within `1e-7`, omit A1 from all full training, result tables, and seed repetitions. Record the equivalence result in the experiment notebook.

Generate `A7-noCORAL`, `A7-noMMD` (global, class, stage, and proto MMD all zero), `A7-noADV`, `A7-noSemantic` (anchor, fit, consistency, residual, and proto-MMD zero while stage-MMD remains), and `A7-noStage`. Mark all as `appendix_conditional`; run them only if the seed-42 core screen supports A7.

- [ ] **Step 6: Generate five-seed confirmation runs**

Run seeds `42,43,44,45,46` only for A0, A0T, A4, A4S, A5, and A7. Screening-only rows remain seed 42 and are identified as such in tables. Execute in three gates: nine core seed-42 groups first, five-seed confirmation second, conditional leave-one-group-out appendix last.

- [ ] **Step 7: Verify generated commands**

Each command must freeze rounds 25, local epochs 5, batch 32, client LR `5e-4`, DA steps 100, DA LR `5e-4`, source clients C1/C2, target calibration client C5 only, and the primary C12-to-C5 data root. Target CE must equal 0 except in the separately named A0T equal-label-budget baseline, where it is exactly 1.0.

### Task 4: Evaluate Classification Ablations and Generalization

**Files:**
- Modify: `gaps_flower/evaluate_checkpoint.py`
- Create: `scripts/summarize_iotj_classification_ablation.py`
- Create: `tests/test_iotj_classification_summary.py`

**Interfaces:**
- Consumes: final checkpoints and logits from Task 3
- Produces: per-run and mean/std CSVs for accuracy, macro-F1, per-class recall, NLL, ECE; convergence curves at rounds 5/10/15/20/25

- [ ] **Step 1: Add true CE/NLL evaluation**

Do not use Flower's current `1-accuracy` pseudo-loss. Compute cross-entropy from logits and labels, and preserve the existing accuracy output.

- [ ] **Step 2: Add summary tests**

Verify seed aggregation, missing-seed failure, final-round selection, and that a strong-DA `server_latest.pth` row is labeled `pre_final_DA`, not `no_DA`.

- [ ] **Step 3: Evaluate A0-A7 and appendix groups**

Primary table rows: A0, A0T, A2, A3, A4, A4S, A5, A6, A7. A0T is required because A5-A7 consume C5 class labels through class-conditional alignment even with target CE zero. Keep A1 only in the implementation-contract audit, with no target-performance claim. Never describe A5-A7 as unsupervised DA.

Create a separate comparison table for Local-C1, Local-C2, centralized pooled-source, FedAvg, FedProx, and A7. Report extra FedProx proximal-state bytes and never describe the centralized reference as privacy preserving.

- [ ] **Step 4: Report statistical uncertainty**

For A0/A0T/A4/A4S/A5/A7 report mean, sample standard deviation, all five paired seed values, and paired seed-wise differences against A7. A bootstrap CI over only five seeds may be shown as descriptive uncertainty but must not be presented as strong significance evidence. Do not claim significance from seed-42 screening rows; use repeat/file-clustered bootstrap as a robustness appendix for window-level metrics.

- [ ] **Step 5: Reuse the clean matrix as external generalization evidence**

Main paper: F1/C1-to-C5 and F2/C12-to-C5 source-count comparison, with F2 as the primary protocol. Appendix: F3/C123-to-C5 and F4/C1234-to-C5 may test extra source domains while C5 remains the only target. Exclude F5 and reverse R1-R4 from primary-protocol claims. Keep existing matrix `base/adapted` wording and do not reinterpret `base` as an independently trained no-DA model.

### Task 5: Build the Unified Regression and Expert-Selection Ablation

**Files:**
- Create: `scripts/evaluate_iotj_regression_ablation.py`
- Create: `tests/test_iotj_regression_ablation.py`
- Reuse: `run_formal_target_ridge_auto_v2_eval.py`
- Reuse: `run_formal_target_mlp_auto_v2_eval.py`
- Reuse: `run_h2_3_plus_fusion_profile.py`
- Reuse: `run_source_augmented_target_ridge_eval.py`

**Interfaces:**
- Consumes: each confirmed A7 C12-to-C5 classifier seed, C5 calibration/test streams, frozen C1/C2 source regression artifacts
- Produces: one aligned C5 prediction CSV and one metric CSV for R0-R7 on identical rows

- [ ] **Step 1: Add row-alignment and leakage tests**

Fail on duplicate `(client,split,sample_index)` keys, missing predictions, any threshold selected from test labels, or any calibration row appearing in test.

- [ ] **Step 2: Freeze head-selection settings**

Ridge alpha grid: `0,0.01,0.1,1,10,100,1000`. MLP solver: LBFGS, ReLU, max_iter 800. MLP hidden grid: `16;32;64;32,16`; alpha grid: `0.001,0.01,0.1,1`. H2.3+ blend grid: `0,0.1,0.25,0.5,0.75,1`; nonCO validation degradation limit: `1.0 ppm`.

- [ ] **Step 3: Evaluate the exact method ladder**

| ID | Regression method |
|---|---|
| R0 | source R3aK16 reference |
| R1 | C5 per-gas target Ridge |
| R2 | C5 per-gas expanded-grid target MLP; this is the C5 H2.3 anchor |
| R3 | C5 H2.3+ balanced blend of the MLP anchor and weak reg-feature Ridge |
| R4 | fixed C5 H8 source-augmented target Ridge applied to every row; no C4 rescue |
| R5 | simple class gate: predicted CO uses C5 H8, otherwise C5 H2.3+ |
| R6 | C5 P4 risk-threshold gate |
| R7 | per-sample oracle expert, diagnostic ceiling only |

- [ ] **Step 4: Add specialist component ablations**

For C5 H8 report target-rich Ridge only, plus source Ridge prediction, plus source per-gas MLP prediction, plus source shared-MLP prediction, and all source predictions. Do not include C4 rescue. For C5 H2.3+ report the expanded-grid MLP anchor versus reg-feature blend.

- [ ] **Step 5: Report the two required lines**

Capability line: `S_CC` before QC for R0-R7. End-to-end line: `S_ALL` under the actual predicted route, followed by Accept-only and Accepted+Review after QC. Always report N and coverage beside errors. The paper may emphasize the two requested story axes, but must not merge `S_CC` with a QC slice.

### Task 6: Evaluate QC and Reliability-Constrained Selection

**Files:**
- Create: `scripts/evaluate_iotj_selective_system.py`
- Create: `tests/test_iotj_selective_system.py`
- Reuse: `gaps_deploy/qc_policy.py`
- Reuse: `gaps_deploy/evaluate_qc_against_random.py`

**Interfaces:**
- Consumes: C5 R3/R4/R5/R6 aligned outputs and deployment-visible risk fields
- Produces: risk-coverage curves, AURC, fixed-coverage table, high-error detection table, and one frozen calibration-selected policy

- [ ] **Step 1: Add calibration-only threshold tests**

Threshold selection may use calibration-validation labels only. Test labels are accepted only by the final metric function.

- [ ] **Step 2: Evaluate QC component groups**

Compare no QC, confidence/margin only, response-signature risks only, full current multi-risk QC, and random rejection matched to every method's coverage.

- [ ] **Step 3: Evaluate fixed coverage points**

Report RMSE, NRMSE, MAE, P90AE, classification-error recall, false rejection of class-correct windows, and normalized-high-error recall at coverage `70%,80%,90%,95%,100%`. Define normalized high error as `abs_error / true_class_range > 0.10`.

- [ ] **Step 4: Compare selector complexity**

Compare C5 H2.3+ only, C5 H8 only, simple predicted-CO gate, one C5 risk threshold, current C5 P4 threshold, and reliability-constrained C5 threshold.

- [ ] **Step 5: Select the constrained policy**

Choose among the fixed H2.3+, fixed H8, simple predicted-CO gate, P4, and reliability-constrained gate using calibration-validation only. Select the highest-coverage feasible candidate satisfying `nonCO RMSE degradation <= 1.0 ppm` and the predeclared normalized-high-error risk bound. If no gated candidate passes, fall back to the better predeclared fixed expert according to calibration-validation. Freeze the selected method and threshold before opening test metrics.

- [ ] **Step 6: Keep conformal risk control as an appendix candidate**

Calibrate the existing nested risk threshold using the same calibration-validation partition and report empirical validity. Do not claim a distribution-free guarantee unless exchangeability assumptions and the exact finite-sample procedure are satisfied.

### Task 7: Complete C5 Source-Count and Low-Calibration Validation

**Files:**
- Reuse: `scripts/replay_matrix_p4_streams.py`
- Reuse: `scripts/analyze_matrix_correct_class_regression.py`
- Reuse: `run_real_route_selector_low_cal_stress.py`
- Create: `scripts/summarize_iotj_matrix_and_budget.py`

**Interfaces:**
- Produces: C5 source-count `S_CC` regression, selected-policy Accepted+Review results, and paired calibration-budget uncertainty

- [ ] **Step 1: Dry-run recovered matrix artifacts**

Run: `python scripts/replay_matrix_p4_streams.py --dry-run --fail-fast`

Expected: F1 and F2 each resolve a checkpoint, the C5-only data root, and a C5 calibration split. F3/F4 are optional appendix rows. No C3/C4 target or C4-rescue artifact is required.

- [ ] **Step 2: Replay all C5-only candidate-policy streams**

Run: `python scripts/replay_matrix_p4_streams.py --reuse-existing --fail-fast`

- [ ] **Step 3: Generate the C5 correct-class decomposition**

Run: `python scripts/analyze_matrix_correct_class_regression.py --reuse-existing --fail-fast`

- [ ] **Step 4: Repeat the calibration-budget experiment**

Use total-window budgets `12,24,48,80,full`, 50 paired repeats, master seed `20260706`, class/concentration-stratified sampling, identical sampled calibration keys for every compared method, and fixed test predictions. This experiment measures target regression/QC calibration efficiency only because the classifier checkpoint has already used the full C5 calibration partition during server adaptation. Do not call it end-to-end system calibration efficiency. If an end-to-end claim is needed, rerun classifier DA separately at each budget.

- [ ] **Step 5: Add statistical summaries**

Report paired median gain, 95% bootstrap CI, positive-gain rate, and Wilcoxon signed-rank p-value with Holm correction over budgets/clients. Add repeat/file clustered bootstrap only as a robustness appendix; do not change the primary window-level split.

### Task 8: Integrate the Frozen Calibration-Selected Policy Into the Final Runtime and Measure the IoT System

**Files:**
- Modify: `gaps_deploy/final_runtime.py`
- Modify: `scripts/build_final_deployment_package.py`
- Modify: `scripts/validate_final_deployment_bundle.py`
- Modify: `scripts/benchmark_edge_and_calibration.py`
- Modify: `scripts/benchmark_system_overhead.py`
- Create: `tests/test_final_runtime_threshold_guard.py`

**Interfaces:**
- Consumes: C5 H2.3+ stream artifact, C5 H8 stream artifact, C5-only frozen selection policy, QC policy
- Produces: one runtime row with selected profile, selected ppm, QC decision, auto output, and timing; runtime/offline parity report

- [ ] **Step 1: Write failing selected-policy runtime tests**

Cover the C5 threshold, nonCO fallback, unknown-client fallback, exact threshold equality, reject output semantics, and absence of true labels at runtime. The final primary package must not contain C3/C4 target profiles.

- [ ] **Step 2: Load and apply `threshold_guard_policy.json`**

Expose `selected_profile`, `selected_ppm`, and `auto_output_ppm`; keep old public fields for backward compatibility.

- [ ] **Step 3: Build and validate the final bundle**

Require both expert artifacts and the frozen calibration-selected policy, even when the selected method is a fixed expert. Fail package validation if the build silently falls back to the old R3aK16-only default.

- [ ] **Step 4: Verify per-window numerical parity**

Compare runtime against the frozen selected-policy outputs on all 1360 C5 primary test windows for each confirmed seed. Acceptance: zero profile mismatches and maximum absolute ppm difference `<=1e-6`.

- [ ] **Step 5: Run edge benchmarks**

On PC and Raspberry Pi, benchmark batch 1 and batch 32, 30 warm-up batches and 100 measured repetitions. Report model-load time, calibration-fit time, p50/p95/p99 latency per window, throughput, peak RSS, artifact size, CPU model, OS, and Python/PyTorch versions.

- [ ] **Step 6: Run communication experiments**

Compare 25-round full training, 10-round training, and 10-round training plus server-only post-DA. Report uplink/downlink bytes, prototype/diagnostic bytes, wall time, target accuracy/F1/NLL/ECE, and downstream `S_CC` regression.

- [ ] **Step 7: Run availability stress**

Use a separate robustness configuration with minimum clients 1: normal two-client run, C2 absent, and C2 disconnected during rounds 8-12 then rejoined. Do not mix these runs with the accuracy ablation table.

### Task 9: Build the Frozen Paper Evidence Pack

**Files:**
- Create: `scripts/build_iotj_evidence_pack.py`
- Create: `tests/test_iotj_evidence_pack.py`
- Create: `results/iotj_evidence_pack_20260711/`
- Modify: `代码文件介绍.md`

**Interfaces:**
- Produces: paper-ready CSV tables, figure source data, limitations, claim-to-evidence map, and reproduction manifest

- [ ] **Step 1: Define mandatory tables**

Table 1 system/data/hardware; Table 2 classification baselines and ablations; Table 3 `S_CC` regression ladder; Table 4 end-to-end QC coverage; Table 5 source-target/budget robustness; Table 6 latency/communication/resource overhead.

- [ ] **Step 2: Define mandatory figures**

Figure 1 cloud-edge architecture; Figure 2 training and inference workflow; Figure 3 wrong-route error amplification; Figure 4 risk-coverage curves; Figure 5 calibration-budget and source-target generalization; Figure 6 edge latency/communication trade-off.

- [ ] **Step 3: Add evidence-pack contract tests**

Fail if a headline metric lacks N/coverage, a table mixes historical protocols, the selected policy lacks runtime parity, a method uses test-selected parameters, or a privacy claim says all target raw calibration data remains local.

- [ ] **Step 4: Generate the final pack**

Run: `python scripts/build_iotj_evidence_pack.py --freeze-manifest results/iotj_experiment_freeze_20260711/input_manifest.json --output-dir results/iotj_evidence_pack_20260711`

- [ ] **Step 5: Update the code guide**

Document whichever calibration-selected candidate passes Task 8 as the final policy; do not promote P4 merely because it is more complex. Mark old H2.3/H8 reports as component evidence, old R3aK16 packages as references, and list one command for reproducing every paper table.

## Execution Order and Gates

1. Tasks 1-2 are blocking code-contract work. Task 2 primary-number regeneration is gated on the new F2/C5 streams from Task 5.
2. Tasks 3-4 establish the classification contribution before any new regression claim.
3. Tasks 5-7 establish personalized quantification and reliability evidence.
4. Task 8 is required before calling the system deployment-ready.
5. Task 9 freezes the paper evidence only after all mandatory gates pass.

Do not proceed to a later gate when an earlier task reports missing artifacts, label leakage, runtime mismatch, or incomplete seeds.
