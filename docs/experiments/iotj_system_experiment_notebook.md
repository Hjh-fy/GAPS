# IoT-J System Experiment Notebook

Last updated: 2026-07-15

## Purpose

This is the durable engineering and research record for the IoT-J system experiment closure. Update it after every implementation, training, evaluation, and review step. Paper claims must be traceable to an entry here and to an immutable result artifact.

## Frozen Main Protocol

- Primary direction: C1,C2 -> C5. C5 is the only target; C3/C4 are excluded from target calibration, target testing, and primary target metrics.
- Data root: `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`; only client directories C1, C2, and C5 are active in the primary protocol.
- Target split: advisor-approved window-level class/concentration-stratified calibration/test split, 20%/80%.
- Calibration inner split: fit/validation = 75%/25%.
- Data split seed: 42.
- Training seeds: 42,43,44,45,46.
- Flower training: 25 rounds, 5 local epochs, batch size 32, client Adam LR 5e-4.
- Server adaptation: 100 steps/round, LR 5e-4, target CE weight 0.
- Primary classification checkpoint: final round 25.
- Primary regression slices: S_ALL, S_CC, S_AR. S_AR intersect S_CC is diagnostic only.
- Training topology: Alibaba Cloud ECS Flower server plus physical Raspberry Pi/PC source clients. Local runs are analysis/test/parity only and are not paper training evidence.

## Decision Log

| Date | Decision | Reason | Evidence/owner |
|---|---|---|---|
| 2026-07-11 | Use an IoT system-method paper story | Best match to current Flower, target calibration, QC, and cloud-edge runtime work | Advisor direction and code audit |
| 2026-07-11 | Keep the window-level stratified calibration/test split | Advisor-approved protocol; primary split is not being redesigned | User confirmation |
| 2026-07-11 | Treat R3aK16 as a source regression reference | It is not the current best target ppm path | `代码文件介绍.md` and result audit |
| 2026-07-11 | Report regression capability on S_CC separately from QC | Existing consolidation incorrectly mixed S_CC with Accepted+Review | `run_final_metric_consolidation_20260709.py` audit |
| 2026-07-11 | Downgrade A1 to an aggregation contract check | With CE-only clients and no DA, GAPS and FedAvg parameter aggregation should be equivalent; prototype diagnostics alone do not justify a full experiment | Pending deterministic equivalence test |
| 2026-07-11 | Require real cloud-edge topology for every reportable training run | Preserve deployment realism; local simulation is permitted only for tests and frozen-artifact analysis | User instruction |
| 2026-07-11 | Change the primary protocol to C1/C2 source and C5-only target | C3/C4 must not participate as target domains; incompatible C12-to-C345 evidence becomes historical only | User correction |
| 2026-07-11 | Continue with the existing `c1234src_c5tgt` data root | The user accepted the shared-root preprocessing; the Flower classification loaders explicitly use `normalize=False`, so `norm_stats.npz` is not consumed by this training/evaluation path | User decision plus `gaps_flower/task.py` and `federated_dataset.py` code check |
| 2026-07-11 | Keep C5-only H8 as a primary regression candidate and P4 as an ablation until new-run confirmation | Historical F2 replay shows the calibration-selected risk gate does not generalize as well as H8 on test | `results/iotj_c5_p4_f2_historical_smoke_20260711` |
| 2026-07-11 | Refreeze classification ablations as v2 before Pi training | A2-A4 previously mixed client losses with selective aggregation, so their effects were not causally attributable | Generator/code review and v2 contract tests |
| 2026-07-11 | Do not predeclare P4 as the final runtime method | Calibration-validation must select among fixed H2.3+, fixed H8, simple gate, and risk gates before test metrics are opened | Historical P4 negative smoke plus leakage review |
| 2026-07-11 | Scope the low-calibration experiment to regression/QC calibration | The frozen adapted classifier has already consumed the full C5 calibration partition; fixed classifier outputs cannot support an end-to-end calibration-budget claim | Pre-training protocol review |
| 2026-07-11 | Add A0T as an equal-target-label-budget baseline | A5-A7 use C5 class labels in class-conditional losses even though target CE is zero; source-only FedAvg alone is not a sufficient fairness baseline | DA loss/data-flow audit |
| 2026-07-11 | Describe the method from executable code and frozen manifests, not from the 2026-07-08 story draft | The old draft is mojibake and still assumes C3/C4/C5, F6, H8+C4, and a preselected P4; all are incompatible with the corrected protocol | `docs/paper/iotj_system_methodology_20260711.zh.md` |
| 2026-07-11 | Keep the running v2r1 queue immutable despite newly identified DA-definition risks | Changing MMD or stage alignment after A0/A0T started would make the core ablation internally incomparable | Method/code audit; post-core correction gate added |
| 2026-07-11 | Treat current stage-MMD as an optional phase-invariance term, not source-target phase-conditioned alignment | Code compares different phases inside each domain; it does not compare the same class/phase across source and target | `gaps_flower/domain_adaptation.py::_compute_stage_mmd_loss` |
| 2026-07-11 | Report H2.3 and H8 feature/training roles exactly | C5 H2.3 anchor is a 104-D rich-feature MLP; its weak Ridge uses rich+64-D reg features; H8 ends in a C5 Ridge but consumes source Ridge/per-gas MLP/shared-MLP predictions | Regression implementation audit |
| 2026-07-11 | Disqualify the historical P4 risk stream from deployable evidence | Its upstream composite risk normalizes by `true_class`, so wrong-route rows indirectly expose test truth; the threshold itself being calibration-selected does not repair a leaked input feature | Regression/QC audit |
| 2026-07-11 | Require a newly named deployment-visible selector risk before formal P4 | Offline composite risk, QC composite risk, and runtime normalized risk ratio previously shared `risk_score` despite different meanings | Risk schema audit |
| 2026-07-11 | Treat the current adversarial term as a legacy under-audit factor | WGAN critic and GRL feature updates appear to increase the same signed Wasserstein statistic; v2 remains immutable, and any sign fix must receive a new group ID | Gradient-direction audit |
| 2026-07-12 | Freeze a separately named B1-B5 v3 correction suite | v2 results must remain reproducible; B1-B4 isolate corrected CORAL/MMD/stage/adversarial factors and B5 is their predeclared full combination | `docs/superpowers/plans/2026-07-12-iotj-v3-classification-regression-qc.md` |
| 2026-07-12 | Use A6 and B5 as the two formal C5 regression backbones | A6 is the strongest compact semantic baseline already available; B5 tests whether corrected distribution alignment improves the downstream system | User decision and v3 execution plan |
| 2026-07-13 | Replay B2 downstream only as a post-screen exploratory candidate | B2 was selected after opening the seed-42 classification test ranking, so its regression/QC results can guide final confirmation but cannot replace the predeclared A6/B5 comparison | `CLS-V3-EVAL-001`, `REG-FORMAL-B2-001` |
| 2026-07-13 | Freeze a paired B2/B5 cross-direction classification study | C1-to-C5, C5-to-C1, and C4/C5-to-C1 test whether the compact MMD2 method generalizes and whether B5's extra CORAL/stage/adversarial stack adds stable value; this is appendix evidence and does not replace the C1/C2-to-C5 primary protocol | `docs/superpowers/specs/2026-07-13-b2-b5-cross-direction-classification.md` |
| 2026-07-13 | Use Pi C1 for C1-to-C5, Pi C5 for C5-to-C1, and Pi C4 plus PC C5 for C4/C5-to-C1 | User-approved real-device mapping; B2 and B5 use identical hardware within every direction | User confirmation and cross-direction specification |
| 2026-07-13 | Harden deployment and legacy Flower paths without changing frozen experiment semantics | Deployment claims require fail-closed QC, exact assets, deployable specialist routing, immutable aggregation weights, and clean-checkout execution; B1-B5/R0-R7 results must remain untouched | `docs/superpowers/specs/2026-07-13-system-safety-hardening-design.md`; commits `6b6d930`-`dfb43f1` plus Git closure |
| 2026-07-15 | Freeze a single new-conversation handoff and track all lightweight seed-42 summaries in Git | A new GPT must be able to recover protocol, results, paths, limitations, and next gates without relying on untracked local results or stale paper prose | `docs/experiments/iotj_latest_handoff_20260715.zh.md`; evidence archive and refreshed paper draft |
| 2026-07-12 | Make HC95 the primary QC operating point, HC90 secondary, and coverage 1 the no-QC baseline | The paper should show that a small 5%-10% review/reject budget concentrates obvious failures, rather than obtaining low RMSE by discarding many windows | User/advisor reporting direction |
| 2026-07-12 | Separate operational QC thresholds from exact fixed-coverage ranking curves | Deployment thresholds are frozen on calibration-validation and produce realized test coverage; exact test coverage is only a non-operational ranking diagnostic | Leakage-safe QC design review |
| 2026-07-12 | Correct the prototype pair-L2 status | `--use-proto-mmd=false` disables only GAPS diagnostic files; DA still evaluates pair-L2 when `da_lambda_proto_mmd>0`, but both uploaded endpoints are detached so the term changes logged loss and no trainable gradient | `gaps_flower/strategy.py`, `gaps_flower/domain_adaptation.py`, A6 round-25 diagnostics |
| 2026-07-12 | Aggregate the three calibrated QC component families by their mean instead of their maximum | The historical F2 smoke showed that taking a maximum of empirical-CDF-normalized components saturated many deployment scores at one and caused static HC95/HC90 thresholds to reject far more than their intended budgets; this change was frozen before formal A6/B5 test evaluation | `scripts/evaluate_iotj_high_coverage_qc.py`; historical F2 smoke only |

## Stage Ledger

| Stage | Status | Entry criteria | Exit evidence |
|---|---|---|---|
| 1. Input freeze and metric contract | complete | Main plan approved | Frozen manifest, corrected S_CC tables, clean tests |
| 2. Classification ablation | B1-B5 seed-42 real-topology screen complete; confirmation seeds pending | Dataset/input contract frozen | Paired confirmation seeds for B2/B5 and required references |
| 3. Regression and expert selection | formal A6/B5 plus exploratory B2 seed-42 ECS closure complete | A6, B5, and post-screen B2 classifier outputs | Multi-seed B2/B5 downstream stability |
| 4. QC and low-calibration reliability | A6/B5/B2 FULL/HC95/HC90 complete; low-calibration stress pending | Aligned A6/B5/B2 regression streams | Low-calibration stress plus frozen runtime policy parity |
| 5. Cross-direction simplification | F1 C1-to-C5, R1 C5-to-C1, and R2 C4/C5-to-C1 paired seed-42 analyses complete | Frozen B2/B5 definitions; balanced F1/R1/R2 datasets | Use confirmation seeds 43-46 to test the observed direction dependence |
| 5. C5 source-count generalization | pending | F1/F2 cloud artifacts recovered | C5-only classification and regression source-count table; F3/F4 appendix optional |
| 6. Runtime and IoT system validation | pending | Final P4 policy frozen | Runtime parity, edge latency, memory, communication |
| 7. Paper evidence freeze | seed-42 handoff, method draft, paper draft, and lightweight summaries current; final freeze pending | Stages 1-6 complete | Multi-seed claim-to-evidence map and submission-ready tables/figures |

## Experiment Register

| ID | Date | Hypothesis | Immutable inputs | Command/config | Output | Status | Headline observation |
|---|---|---|---|---|---|---|---|
| AUDIT-001 | 2026-07-11 | Current artifacts can be frozen under the corrected C12-to-C5 protocol | C12-to-C5 dataset and F2 cloud run | `python scripts/audit_iotj_experiment_inputs.py --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --run-dir results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25 --output results/iotj_experiment_freeze_20260711/input_manifest.json` | `results/iotj_experiment_freeze_20260711/input_manifest.json` | complete | 228 artifacts hashed; active clients are exactly C1/C2/C5; C5 calibration/test is 320/1360; canonical F2 is complete with 183 files. |
| METRIC-001 | 2026-07-11 | Pure S_CC must be computed independently of QC | Historical F6/P4 outputs used only as a contract fixture | `python run_final_metric_consolidation_20260709.py --output-dir results/final_metric_consolidation_20260711_historical_contract --docs-report results/final_metric_consolidation_20260711_historical_contract/historical_story_report.zh.md` | `regression_slice_table.csv` | complete | Slice code now emits S_ALL, S_CC, S_CW, S_AR, and S_AR intersect S_CC with explicit parent N/coverage. Historical parity: S_CC N=5301, RMSE=8.3269; S_AR N=4228, RMSE=5.8497. New primary C5 values remain pending. |
| CONTRACT-A01 | 2026-07-11 | CE-only GAPS aggregation equals FedAvg aggregation when selective aggregation, prototype diagnostics, and DA are disabled | Two deterministic synthetic FitRes with 30/70 sample weights | `python -m pytest tests/test_flower_classification_contract.py -q --basetemp .tmp_pytest_cls_equivalence_20260711` | Contract audit | complete | All tensors match FedAvg within absolute tolerance `1e-7`; A1 is not scheduled for full training. |
| CLS-CONFIG-001 | 2026-07-11 | A0-A7 can be generated without role/config drift | Existing C12-to-C5 data root; seeds 42-46 | `python scripts/generate_iotj_classification_ablation_commands.py --include-confirmation-seeds` | `results/iotj_classification_ablation_20260711_commands` | complete | 24 manifests generated: 23 scheduled real training runs and one A1 contract-only row; protocol validation found zero invalid manifests. |
| CLS-CONFIG-002 | 2026-07-11 | Client modules, selective aggregation, server DA, and equal-label-budget adaptation can be separated before real training | Same frozen C12-to-C5 protocol | `python scripts/generate_iotj_classification_ablation_commands.py --include-confirmation-seeds` | `results/iotj_classification_ablation_20260711_v2_commands` | complete | v2 adds A0T/A4S, disables selective aggregation in A2-A4, disables timing-only proto-MMD diagnostics, and materializes five conditional leave-one-group-out rows. 39 manifests: 9 core seed-42 groups, 24 additional confirmation runs, 5 conditional appendix runs, and A1 contract-only. CLS-CONFIG-001 is superseded and must not be launched. |
| CLS-RUN-001 | 2026-07-11 | Seed-42 screening can start on the real ECS/Pi/PC topology | Superseded v1 command manifests | Connectivity preflight | No training outputs | superseded-paused | No partial training run started. Controller PID 28880 was stopped before connectivity returned so it cannot launch the confounded v1 queue. |
| CLS-RUN-002 | 2026-07-11 | The refrozen v2 core screen can execute on the exact cloud-edge runtime | Runtime commit `2424071`; frozen v2 manifests carry the same full revision | `python scripts/run_iotj_classification_cloud_edge.py --pi-hosts gaps@192.168.31.184` | `results/iotj_classification_ablation_20260711_v2r1_controller` and `results/iotj_classification_ablation_20260711_v2r1` | complete | All nine seed-42 groups A0/A0T/A2/A3/A4/A4S/A5/A6/A7 completed 25 rounds on ECS + physical Pi/PC and were recovered locally. Controller ended at 22:48 with “All requested classification runs completed.” |
| METHOD-DOC-001 | 2026-07-11 | A code-aligned method chapter can replace the obsolete F6/H8+C4 story | Commit `2424071`, frozen manifests, current C5 regression smoke artifacts, source files listed in the implementation map | Manual source/manifests/formula audit | `docs/paper/iotj_system_methodology_20260711.zh.md`; updated `代码文件介绍.md` | complete | Document separates verified implementation, pending claims, mathematical definitions, causal ablations, two metric lines, known risks, and result-dependent paper-story branches. |
| CLS-EVAL-001 | 2026-07-11 | One evaluator can produce true classification metrics and aligned C5 streams | Historical F2 adapted checkpoint used only for numerical parity | `evaluate_checkpoint_stream(..., split='test')` | In-memory 1360-row check plus evaluator tests | complete | Exact parity with prior F2 summary: N=1360, accuracy=0.9882353, macro-F1=0.9882675, NLL=0.1000582, ECE=0.0111189. The same code will evaluate new A-runs. |
| CLS-EVAL-002 | 2026-07-11 | The seed-42 core screen identifies which mechanism families merit confirmation | Nine recovered round-25 checkpoints; fixed 1360-row C5 test | `python -m scripts.summarize_iotj_classification_ablation --run-root results/iotj_classification_ablation_20260711_v2r1 ...` | `results/iotj_classification_ablation_20260711_v2r1_summary` | complete-screening | A0 26.54%, A0T 98.24%, A2 29.71%, A3 29.12%, A4 29.34%, A4S 31.62%, A5 73.01%, A6 98.01%, A7 98.60% accuracy. A7 also has macro-F1 98.60%, NLL 0.1132, ECE 0.0118. A6 replaces A5's distribution DA with semantic DA rather than incrementally adding it; the result supports a method-family contrast, while A7 only modestly exceeds A6 and A0T. Single-seed values are screening evidence, not final significance claims. |
| REG-INPUT-001 | 2026-07-11 | The confirmed C5 classifier can feed the regression candidates without C3/C4 leakage | Historical F2 classifier plus C1/C2 source regression reference, smoke only | `python scripts/build_iotj_c5_regression_inputs.py ...` | `results/iotj_c5_regression_inputs_f2_historical_smoke_20260711` | complete | Contract produced exactly C5 calibration/test 320/1360 rows and exported aligned backbone/source-reference features; no new training was performed. |
| REG-H23-001 | 2026-07-11 | A C5-only H2.3+ anchor/blend can be selected on calibration-validation only | REG-INPUT-001 historical smoke stream | `python scripts/run_iotj_c5_h23_plus.py ...` | `results/iotj_c5_h23_plus_f2_historical_smoke_20260711` | complete | Historical test RMSE 21.2182; S_CC N=1344, RMSE 12.3807. Selected blend weight was zero, so the expanded H2.3+ collapsed to its MLP anchor in this replay. |
| REG-H8-001 | 2026-07-11 | H8 can be evaluated for C5 without the obsolete C4 rescue path | REG-INPUT-001 historical smoke stream | `python run_source_augmented_target_ridge_eval.py ... --disable-c4-rescue` | `results/iotj_c5_h8_no_rescue_f2_historical_smoke_20260711` | complete | Historical H8 test RMSE 16.6166; S_CC N=1344, RMSE 11.5028. Manifest records `c4_rescue_enabled=false`. |
| REG-P4-001 | 2026-07-11 | Calibration-selected risk routing improves over both fixed experts | Historical H2.3+/H8 streams, smoke only | `python scripts/select_iotj_c5_p4.py ...` | `results/iotj_c5_p4_f2_historical_smoke_20260711` | invalid-for-deployment | Threshold 0.00904 used H8 for 5.22% of test rows and gave RMSE 17.3559, worse than fixed H8 16.6166. More importantly, the upstream risk used `true_class` to select ppm range, so this row is now leakage-diagnostic only and cannot support any deployable P4 claim. |
| PLAN-V3-001 | 2026-07-12 | A versioned correction and high-coverage reliability plan can close the current method-definition and paper-story gaps without mutating v2 | Frozen v2r1 artifacts, A6/A7 audit, C5 F2 smoke, user/advisor reporting constraints | Plan self-review and placeholder scan | `docs/superpowers/plans/2026-07-12-iotj-v3-classification-regression-qc.md` | complete | Execution order is fixed as DA correction tests, B1-B5 seed 42, A6/B5 formal regression on ECS, HC95/HC90 QC, paired seeds, then Pi/PC parity and evidence freeze. B5 is predeclared before test inspection. |
| DA-V3-IMPL-001 | 2026-07-12 | Conventional MMD-squared, cross-domain class-phase alignment, and corrected Wasserstein feature minimization can be added without changing legacy v2 defaults | Current `utils.py` and Flower DA runtime | `python -m pytest tests/test_flower_da_v3_corrections.py tests/test_flower_classification_contract.py tests/test_iotj_classification_summary.py tests/test_iotj_cloud_edge_controller.py -q --basetemp .tmp_pytest_iotj_v3_preflight_contract` | Versioned DA runtime and focused contracts | complete | 62 tests passed. Legacy defaults remain `legacy_quartic`, `legacy_intra_domain`, and `legacy_grl_plus`; corrected modes are explicit CLI/manifest fields. DA summaries now state that detached prototype pair-L2 is non-trainable. |
| CLS-V3-CONFIG-001 | 2026-07-12 | B1-B5 can isolate four corrected factors and a predeclared full combination | DA-V3-IMPL-001; frozen C12-to-C5 protocol | `python scripts/generate_iotj_classification_ablation_commands.py --suite v3 --output-root results/iotj_classification_ablation_20260712_v3_commands --results-root results/iotj_classification_ablation_20260712_v3` | `results/iotj_classification_ablation_20260712_v3_commands` | complete | Exactly five seed-42 manifests generated. Every B group uses A6 semantic weights, corrected modes, selective aggregation, and prototype pair-L2 weight zero; controller dry-run resolves B1-B5 only. |
| CLS-V3-RUN-001 | 2026-07-12/13 | The corrected screen can run sequentially on ECS plus physical Pi/PC | CLS-V3-CONFIG-001; all three machine preflights passed after Pi recovery | Controller PID `50220`; stdout `results/iotj_classification_ablation_20260712_v3_controller/controller.stdout.log` | `results/iotj_classification_ablation_20260712_v3` | complete | B1-B5 each completed 25 rounds and were recovered by 04:38 on 2026-07-13. Every run has 25 client-stat files, 25 adapted checkpoints, 25 DA diagnostics, history/config/final adapted checkpoint; controller stderr is empty and Pi remained `throttled=0x0`. |
| CLS-V3-B1-001 | 2026-07-12 | Corrected CORAL on the A6 semantic core can retain strong C5 classification | Frozen B1 seed-42 manifest; real ECS + Pi C1 + PC C2 topology | Controller-managed 25-round run; live summary over final adapted checkpoint | `results/iotj_classification_ablation_20260712_v3/B1_proto_replay_corrected_server_da_c12_to_c5_s42_r25`; `results/iotj_classification_ablation_20260712_v3_summary_live` | complete-screening | All 25 round artifacts and DA diagnostics are present. Checkpoint SHA-256 is `7F4CE4C311E9FAD444968445D847EE34A035274827E7CB975C10938F9173A485`; C5 N=1360, accuracy=0.9875, macro-F1=0.987534, NLL=0.098826, ECE=0.012027. This is single-seed screening evidence only. |
| REG-SUITE-IMPL-001 | 2026-07-12/13 | A single versioned command chain can rebuild classifier-aligned C5 inputs, H2.3+, H8 without C4 rescue, and leakage-safe QC | A6/B5/B2 checkpoints; frozen C12-to-C5 data | `python -m pytest tests/test_iotj_c5_regression_suite.py tests/test_iotj_high_coverage_qc.py -q` | `scripts/run_iotj_c5_regression_suite.py`, cloud runner, ladder, and QC runtime | complete | ECS sync/launch/recovery, R0-R7 finite-value validation, and append-safe cloud manifests are implemented. A6/B5/B2 have all been fitted on ECS and recovered. |
| QC-IMPL-001 | 2026-07-12 | Calibration-only deployment-visible risk can support HC95/HC90 without using target truth at inference | Historical F2 streams used only as a smoke fixture | `python scripts/evaluate_iotj_high_coverage_qc.py ...` plus truth-invariance tests | `results/iotj_c5_high_coverage_qc_f2_historical_smoke_20260712_v2` | complete-smoke | Risk is invariant to changes in true class/ppm and correctness fields; FULL, static HC95/HC90, exact-coverage ranking curves, and 1000 matched-random controls are emitted. Historical realized acceptance was 92.94%/87.87%, confirming that formal results must report realized coverage rather than relabel thresholds as exact test coverage. |
| CONTRACT-V3-ALL-001 | 2026-07-12 | All corrected classification, regression, and QC interfaces remain mutually compatible | Current v3 source and tests | `python -m pytest tests/test_flower_da_v3_corrections.py tests/test_flower_classification_contract.py tests/test_iotj_classification_summary.py tests/test_iotj_cloud_edge_controller.py tests/test_iotj_c5_regression_inputs.py tests/test_iotj_c5_regression_suite.py tests/test_iotj_high_coverage_qc.py -q --basetemp .tmp_pytest_iotj_v3_all` | Test report | complete | 76 tests passed with two dependency deprecation warnings and no failures. |
| CLS-V3-EVAL-001 | 2026-07-13 | The corrected factors have separable effects and the predeclared full method need not be the best single-seed classifier | Five recovered seed-42 final adapted checkpoints; fixed C5 N=1360 test | `python -m scripts.summarize_iotj_classification_ablation ... --expected-groups B1,B2,B3,B4,B5` | `results/iotj_classification_ablation_20260712_v3_summary` | complete-screening | B1/B2/B3/B4/B5 accuracy is 98.75/99.2647/98.8971/98.9706/98.8971%. B2 also has the best macro-F1 99.2657%, NLL 0.0690, and ECE 0.00668. B5 does not show additive accuracy gain; retain it as the predeclared full-mechanism backbone for formal regression while treating B2 as the performance candidate. All conclusions remain seed-42 screening only. |
| REG-PREFLIGHT-001 | 2026-07-13 | Formal A6/B5 regression has all required immutable checkpoints and ECS dependencies | A6, B5, and R3aK16 checkpoints; C5 data root | Local SHA-256 audit plus ECS SSH import/data probe | Preflight record | complete | A6/B5/R3aK16 SHA-256 values start `60191614`, `D497BBA2`, and `790FC6FF`; ECS imports torch 2.12.0+cu130, NumPy, and scikit-learn and can read C5 calibration data. No infrastructure blocker remains. |
| REG-CLOSURE-AUDIT-001 | 2026-07-13 | A file-complete regression suite also has a numerically complete R0-R7 ladder | First ECS output `results/iotj_c5_formal_regression_20260713/A6` | Immediate post-recovery metric audit | First output retained as diagnostic only | invalid-superseded | R1 metrics were empty because H8 rich-only predictions were written to a separate CSV and were absent from the QC/ladder merge. B5 was stopped before completion. Added key-aligned rich-only attachment and fail-closed finite checks for R0-R4; 20 focused tests passed before rerun. |
| REG-FORMAL-001 | 2026-07-13 | Corrected B5 classification features improve downstream C5 personalization relative to A6 | Frozen A6/B5 seed-42 checkpoints, identical C5 calibration/test, identical grids and source reference | `python scripts/run_iotj_c5_regression_cloud.py ... --remote-output-base ..._v2 --device cpu --n-random 1000` | `results/iotj_c5_formal_regression_20260713_v2`; consolidated `..._v2_summary` | complete-seed42 | All required manifests exist; each classifier has 1360 test rows and 56 finite R0-R7 metric rows. B5 coverage-1 S_ALL RMSE: R1 19.3245, R2/R3 20.1082, R4 17.4473, R5 17.7724, R6 18.1699, R7 oracle 15.7940. A6 R4 is 28.0144 and R5 is 27.7283. B5 S_CC N=1345: R4 RMSE 11.3890; A6 S_CC N=1333: R4 RMSE 11.3890. The large S_ALL gain therefore comes mainly from fewer/less damaging route errors, while correct-route numerical capability is nearly unchanged. |
| QC-FORMAL-001 | 2026-07-13 | Deployment-visible calibration-only risk concentrates obvious B5 failures while preserving high automatic yield | Formal B5 R4 stream; score family and thresholds selected on calibration-validation only | Formal QC stage inside REG-FORMAL-001 | `results/iotj_c5_formal_regression_20260713_v2/B5/high_coverage_qc` | complete-seed42 | FULL RMSE 17.4473. HC95 realized accept/review/reject is 1309/33/18, automatic yield 96.25%, nonreject coverage 98.68%, accepted RMSE 15.9075; it flags 7/15 route errors and 20/135 high-error rows versus matched-random mean recalls 4.04% and 3.75%. HC90 realizes 88.24% automatic yield and 98.68% nonreject coverage with accepted RMSE 15.3599. Report realized coverage and do not relabel HC90 as exact 90% test acceptance. |
| REG-FORMAL-B2-001 | 2026-07-13 | The best seed-42 classification screen candidate also improves downstream actual-route performance under the unchanged formal pipeline | Frozen B2 checkpoint SHA-256 `8FA3123F...FF2FDD`; same C5 split, R3aK16 source reference, grids, seed, and 1000 random controls as A6/B5 | `python scripts/run_iotj_c5_regression_cloud.py --classifier B2=... --remote-output-base ..._v2 --device cpu --seed 42 --n-random 1000` | `results/iotj_c5_formal_regression_20260713_v2/B2` | complete-exploratory-seed42 | B2 has 10 route errors. R4 is the best deployable coverage-1 point estimator: S_ALL RMSE 14.6564, NRMSE 0.1059, MAE 7.4099; S_CC N=1350 and RMSE 11.3288. R5/R6 are worse at 15.0080/15.4945. R7 oracle is 12.6393 and is not deployable. Relative to B5, the S_CC R4 difference is only 0.0602 ppm, so most S_ALL gain comes from fewer and less damaging route errors, not a materially stronger correct-route regressor. |
| QC-FORMAL-B2-001 | 2026-07-13 | The unchanged deployment-visible QC can retain about 95% automatic yield while enriching B2 route and high-error failures | Formal B2 R4 stream; full risk and thresholds selected only on 80 calibration-validation rows | Formal QC stage inside REG-FORMAL-B2-001 | `results/iotj_c5_formal_regression_20260713_v2/B2/high_coverage_qc` | complete-exploratory-seed42 | HC95 accept/review/reject is 1301/35/24: automatic yield 95.66%, nonreject coverage 98.24%, accepted RMSE 12.6729, false-flag rate among class-correct rows 3.85%. It flags 7/10 route errors and 23/132 high-error rows versus matched-random mean recalls 4.02% and 4.26%. HC90 yields 89.49%, accepted RMSE 11.5866, and flags 8/10 route errors. QC detects risk; it does not repair reviewed/rejected predictions. |
| REG-EVIDENCE-001 | 2026-07-13 | Formal regression/QC claims can be traced to compact immutable summaries | Valid formal v2 suite manifests and summaries | `python scripts/summarize_iotj_c5_formal_regression.py --classifiers A6,B5,B2 ...` | `results/iotj_c5_formal_regression_20260713_v2_summary` | complete | Consolidation validated 168 ladder rows and nine operational QC rows, then recorded SHA-256 for each A6/B5/B2 source summary, QC JSON, and suite manifest. |
| CLS-XDIR-IMPL-001 | 2026-07-13 | A separate manifest-driven path can test B2/B5 across directions without weakening the primary C1/C2-to-C5 guards | Approved F1/R1/R2 specification; frozen v3 B2/B5 definitions | `python -m pytest tests/test_iotj_cross_direction_classification.py tests/test_iotj_cross_direction_controller.py tests/test_iotj_cross_direction_summary.py tests/test_iotj_cloud_edge_controller.py tests/test_iotj_classification_summary.py tests/test_flower_classification_contract.py -q` | New generator, controller, target-aware evaluator, and paired summary | complete | Integrated verification is 72 passed with two dependency deprecation warnings. The implementation caught and fixed two preflight defects before training: ECS source adaptation requires source calibration arrays, and direct script entrypoints require an explicit repository import path. The primary controller remains hard-guarded to C1/C2-to-C5. |
| CLS-XDIR-CONFIG-001 | 2026-07-13 | The approved six-run seed-42 queue can be materialized without protocol or topology drift | Config SHA-256 `68339EB0...B5B353`; code revision `4565587` | `python scripts/generate_iotj_cross_direction_commands.py --seed 42 ...`; controller `--dry-run` | `results/iotj_b2_b5_cross_direction_20260713_commands` | complete | Exactly six manifests resolve in order F1 B2/B5, R1 B2/B5, R2 B2/B5. Pi assignments are C1, C1, C5, C5, C4, C4; PC is active only as C5 for the R2 pair. Command-index SHA-256 is `F9BDCDFB...D58E1E`. |
| CLS-XDIR-PREFLIGHT-001 | 2026-07-13 | All three real machines can accept the frozen cross-direction queue | Formal command root and active-file SHA-256 map | Controller `--preflight-only`; independent ECS/Pi probes | Preflight record | complete | After a temporary network outage, Pi recovered at `192.168.31.184`. ECS/Pi active-file SHA-256 checks required zero copies, ECS was idle with 23 GB available and PyTorch 2.12.0+cu130, and Pi had 46 GB available at 50.5 C with `throttled=0x0`. All source-train, source-calibration, and target-calibration inputs match the manifests. |
| CLS-XDIR-RUN-F1-001 | 2026-07-13 | Compact B2 remains competitive with full B5 for C1-to-C5 under a one-source physical-edge topology | Frozen F1 B2/B5 seed-42 manifests; Pi C1; ECS target C5 calibration | Controller PID `21920`; stdout `results/iotj_b2_b5_cross_direction_20260713_controller/f1_pair.stdout.log` | `results/iotj_b2_b5_cross_direction_20260713`; paired summary and per-run evaluation | complete-seed42 | B2 and B5 both completed 25 rounds and were recovered with 25 client-stat and 25 DA diagnostic files each. B2/B5 target C5 accuracy is 0.9889706/0.9830882, macro-F1 is 0.9889540/0.9831310, NLL is 0.100814/0.1322, and ECE is 0.010652/0.0150. B2 has 15 errors versus B5's 23 and is noninferior by the predeclared rule, but paired McNemar p=0.0963 does not support a significant superiority claim from seed 42 alone. |
| CLS-XDIR-RUN-R1-001 | 2026-07-14 | Test whether the compact B2 configuration remains competitive when C5 is the physical source and C1 is the target | Frozen R1 B2/B5 seed-42 manifests at code revision `4565587`; Pi C5; ECS target C1 calibration | Retry controller PID `42308`; stdout `results/iotj_b2_b5_cross_direction_20260713_controller/r1_pair_retry_20260714.stdout.log`; corrected paired evaluator with manifest-enforced calibration/test counts and 2000 class-stratified bootstrap replicates | `results/iotj_b2_b5_cross_direction_20260713`; `results/iotj_b2_b5_cross_direction_20260714_r1_summary` | complete-seed42 | Both runs completed 25/25 rounds and recovered 157 artifacts each. On the correct C1 target split (calibration N=680, test N=2680), B2/B5 accuracy is 0.976493/0.983582, macro-F1 is 0.976525/0.983605, NLL is 0.276859/0.171755, and ECE is 0.023724/0.015614. B5 has 44 errors versus B2's 63; B2-minus-B5 accuracy is -0.7090 pp with stratified bootstrap 95% CI [-1.1940, -0.2612] pp. B5-only correct/B2-only correct is 30/11 and exact McNemar p=0.00432. Thus R1 provides significant seed-42 paired evidence for B5, while F1 favors B2 descriptively; mechanism value is direction-dependent rather than universally additive or universally redundant. |
| CLS-XDIR-RUN-R2-001 | 2026-07-14/15 | Test B2 against B5 when two heterogeneous physical sources jointly adapt to C1 | Frozen R2 B2/B5 seed-42 manifests at code revision `4565587`; Pi C4 plus PC C5; ECS target C1 calibration | B2/B5 controller logs `r2_b2.stdout.log` and `r2_b5.stdout.log`; manifest-enforced N=2680 paired evaluator; 2000 class-stratified bootstrap replicates | `results/iotj_b2_b5_cross_direction_20260713`; `results/iotj_b2_b5_cross_direction_20260715_r2_summary` | complete-seed42 | Both runs completed 25/25 rounds, recovered 157 formal artifacts each, contain 25 two-client stat files with client IDs 4 and 5, and have 25 DA diagnostics. B2/B5 accuracy is 0.989552/0.991418, macro-F1 is 0.989565/0.991436, NLL is 0.105859/0.089356, and ECE is 0.009554/0.008249. B5 has 23 errors versus B2's 28 and improves worst-class recall from 0.983582 to 0.991045. B2-minus-B5 accuracy is -0.1866 pp with 95% CI [-0.4851, 0.1119] pp; B5-only/B2-only correct is 11/6 and exact McNemar p=0.3323. The predeclared three-metric 0.5 pp rule labels R2 `B5_favored` because the worst-recall delta is -0.7463 pp, while the paired accuracy difference is not statistically significant. Together with F1/R1, this supports direction-dependent robustness rather than universal additive gain. |
| REG-QC-ORACLE-001 | 2026-07-14 | Separate classifier routing loss from the common H8 regressor while reporting both automatic-output and human-review operating scopes | Frozen A6/B5/B2 formal v2 inputs; same ECS H8/QC fit; no test-label use in the operational policy | Rebuild H8 with a second, diagnostic-only stream that forces `route_class=true_class`; strict key join into unchanged FULL/HC95/HC90 decisions; execution commit `7106e5c`; summary-contract commit `dc50495`; 40 focused tests on both local PC and ECS, 55 combined evidence tests locally | `results/iotj_c5_formal_regression_20260713_v2_summary/qc_operational_comparison.csv`; `formal_regression_report.md`; schema-v2 `manifest.json` | complete-seed42 | Original H8 prediction and risk-policy SHA-256 values are unchanged, as are all previous accept/review/reject counts and actual-route metrics. The table now reports Accepted and Nonreject (=Accept+Review) RMSE/NRMSE plus oracle-route values on the exact same frozen QC subsets. FULL uses all 1360 rows and gives the same oracle RMSE/NRMSE 11.9082/0.0690 for A6, B5, and B2, confirming a common regressor and isolating classification routing as the main source of their coverage-1 gap. This is an offline forced-true-class routing diagnostic, not deployable performance. `S_CC` is different: it drops misclassified rows, whereas this oracle diagnostic retains and re-routes every row. The extension manifest records the execution revision plus nine key input/output hashes per classifier, separately from the original base-suite manifest; the summarizer now also rejects extra workpoints, non-1360 N, nonconserved decisions, and metric-cardinality mismatches. |
| SYS-HARDEN-001 | 2026-07-13 | The reviewed P0/P1 deployment and legacy training failures can be closed without changing frozen B1-B5/R0-R7 semantics | Approved hardening design; current production paths and regression fixtures | Focused TDD suites, frozen-contract gate, tracked-only checkout, `compileall`, independent review | Commits `6b6d930`-`dfb43f1`; current Git closure | complete | QC and assets fail closed, including unavailable ranking evidence; phase paths agree; production DA enforces strict calibration inputs and handles no shared classes; specialist selection uses predicted routes and disjoint validation; FedAvg uses checkpoint `n_samples`; clean-index import/CLIs pass. Verification: deploy 88, Flower 62, specialist/regression 44, frozen contracts 111, all tracked tests 322; only two existing protobuf deprecation warnings in suites that import Flower. Independent re-review found no remaining Critical/Important issue. |
| REPORT-ADVISOR-001 | 2026-07-13 | A concise advisor briefing can present the completed system loop without overstating single-seed or still-running evidence | Frozen A0-A7/B1-B5 summaries, formal A6/B5/B2 regression and QC summaries, F1 B2 partial result | DOCX build audit plus 12-page Word-to-PDF raster inspection | `docs/paper/GAPS_IoTJ_advisor_progress_report_20260713.zh.md`; `docs/paper/GAPS_IoTJ_advisor_progress_report_20260713.zh.docx` | complete | The briefing provides an 11-slide PPT structure, exact classification/regression/QC tables, speaking notes, advisor Q&A, paper contribution framing, evidence boundaries, and the next experiment gates. All 12 rendered document pages were inspected with no clipping, overlap, missing glyphs, or orphaned key figures/tables. |

## Review Findings and Risks

| Date | Severity | Finding | Resolution/status |
|---|---|---|---|
| 2026-07-11 | High | Current final metric script labels S_AR intersect S_CC as classification-correct regression | Stage 1 correction planned |
| 2026-07-11 | High | P4 policy is not loaded by the default final runtime bundle | Stage 6 integration planned |
| 2026-07-11 | Medium | Flower evaluation loss is 1-accuracy rather than CE/NLL | Classification evaluation correction planned |
| 2026-07-11 | Medium | Main P4 test behavior is close to a simple predicted-CO expert switch | Add simple switch baseline and constrained selector |
| 2026-07-11 | Medium | Repeated historical inspection may have indirectly adapted design to the test set | Freeze configuration and run final locked evaluation |
| 2026-07-11 | Medium | Initial matrix inventory used short aliases beside recovered full directory names, creating six false missing rows | Resolved: priority aliases now map to the six canonical full directory names; regenerated manifest has six complete rows and no missing artifacts. |
| 2026-07-11 | High | F6/H2.3+/P4 primary evidence was generated with C3/C4/C5 as targets | Reclassify as historical diagnostic; rebuild C5 regression, expert selector, and QC from the F2 C12-to-C5 classifier |
| 2026-07-11 | High | Task 1 audit used name-based matrix heuristics, weak dataset validation, and unsafe output handling | Resolved with explicit F2 run auditing, structured `split_info.json` validation, F2-only completion, deterministic payload separation, and protected-output checks; 28 focused tests pass. |
| 2026-07-11 | Low | Shared data root `norm_stats.npz` was fitted more broadly than C1/C2 | Accepted for this experiment after code verification: Flower `load_client_loaders` passes `normalize=False` for both source training and client testing, so the file does not affect classifier tensors. Keep this statement scoped to the Flower path. |
| 2026-07-11 | Blocking environment | Raspberry Pi SSH is unreachable from the Codex execution environment at the current address `192.168.31.184`; old addresses are retired | v2 controller PID 33856 is active and retries only `gaps@192.168.31.184`. It will sync and preflight Pi before launching A0. Do not substitute local simulation. |
| 2026-07-11 | Medium | P4 threshold selection improved historical calibration-validation but underperformed fixed H8 on test | Keep P4 as an ablation/negative result; do not promote it without multi-seed new-run evidence and a stronger, predeclared selector. |
| 2026-07-11 | High | Original A2/A3/A4 simultaneously enabled selective aggregation, confounding client-loss attribution | Resolved in v2: A2-A4 use FedAvg-equivalent parameter aggregation; A4S isolates selective aggregation. |
| 2026-07-11 | Medium | The old `proto_only` profile mixed alignment with device-residual statistic extraction even when no server loss consumed the residual | Resolved in v2: A2 uses `align_only`; A4/A4S/A5 use `align_replay`; residual statistics activate only in A6/A7 semantic-DA groups. |
| 2026-07-11 | Medium | Flower client evaluation reported `1-accuracy` through the field named loss | Resolved before training: client evaluation now reports mean cross-entropy/NLL and keeps accuracy as a separate metric. |
| 2026-07-11 | High | Task 8 assumed P4 would be final despite historical calibration/test reversal | Resolved in plan: calibration-validation selects among all predeclared fixed/gated candidates, then the selected method is frozen for test/runtime parity. |
| 2026-07-11 | Medium | Fixed-classifier low-budget replay could be mislabeled as end-to-end calibration efficiency | Resolved in plan: primary budget claim is target regression/QC calibration efficiency; end-to-end claims require rerunning DA at each budget. |
| 2026-07-11 | Medium | Unified regression summaries omitted R2, leaving requested table cells blank | Resolved in the shared metric function; constant-label slices explicitly return blank R2. |
| 2026-07-11 | High | A5-A7 consume target calibration class labels but lacked an equal-label-budget supervised baseline | Resolved in v2 design: A0T uses the same 100 steps/round and C5 calibration labels with source rehearsal CE + target CE, while all proposed alignment terms remain off. |
| 2026-07-13 | Resolved environment | Physical Pi temporarily disappeared from the local 192.168.31.0/24 network before the B2/B5 cross-direction preflight | Pi recovered at `192.168.31.184`; full hash/code/health preflight passed before F1 training. No local simulation or partial run was substituted during the outage. |
| 2026-07-14 | Resolved experiment provenance | Starting R1 from the later safety-hardening branch synchronized a new `server_app.py` import without its new module, while the six cross-direction manifests freeze code revision `4565587` | The server failed before Pi launch and produced no training artifacts. Its log was archived, the partial run directory was moved out of the formal result root, and the exact manifest revision was restored to ECS/Pi with SHA-256 verification. R1 now runs with `--skip-code-sync --skip-data-sync`; do not mix the later safety-hardening runtime into F1/R1/R2. |
| 2026-07-14 | Resolved evaluation provenance | The first R1 paired summary evaluated the recovered checkpoints against the default C1/C2-to-C5 data root, yielding only N=680 instead of the R1 manifest's N=2680 C1 test | Archived the invalid summary under `results/iotj_b2_b5_cross_direction_20260714_r1_summary_invalid_wrong_data_root`; added a fail-closed manifest count contract and direction filter to the cross-direction summarizer; regenerated the canonical R1 summary on `client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` with calibration/test N=680/2680. Never cite the archived 680-row metrics. |
| 2026-07-11 | High | Replay distillation installed the current server model as a teacher in round 1 despite being defined as previous-round replay | Resolved before training: round 1 only caches the incoming state; the frozen teacher activates from round 2. |
| 2026-07-11 | High | CE-only FedAvg still computed and uploaded unused prototype statistics, invalidating runtime/communication comparison | Resolved before training: CE-only and replay-only profiles skip the extra statistics pass and JSON payload; alignment profiles upload prototypes, while device residuals are limited to semantic-DA groups. |
| 2026-07-11 | Blocking code | Cloud-edge controller synchronized runtime/manifests to Pi but assumed ECS already had the same code and command root | Resolved before training: controller checks ECS idle, uploads committed root/server runtime plus the exact v2 command directory, and reruns code preflight before waiting for Pi. |
| 2026-07-11 | Blocking code | First A0 launch stopped before clients because Windows-generated shell scripts contained CRLF and Bash parsed `pipefail\r` | No training occurred. Resolved with LF-only command generation plus remote pre-launch normalization; focused controller/profile tests pass. |
| 2026-07-11 | High method-definition | `compute_mmd` already returns the biased empirical MMD-squared kernel discrepancy, while global/class/stage DA applies `**2` again | Freeze v2 as implemented and describe it as a squared kernel-discrepancy term. After the core review, add a separately named conventional-MMD correction if MMD groups merit continuation. Do not silently relabel v2 as standard MMD. |
| 2026-07-11 | High conceptual | Current stage-MMD compares pairs of phases within source and within target, rather than matching the same `(class, phase)` across domains | Use A6 vs A7 as the immediate diagnostic. Predeclare a possible cross-domain same-class/same-phase correction after the v2 core; do not claim stage-preserving alignment from current A7. |
| 2026-07-11 | Medium scale-sensitivity | Server prototype fit multiplies each squared error by both client weight and absolute class-phase count, then averages over terms rather than total effective weight | Record the exact formula in the method chapter; consider a normalized-weight ablation only after the current queue so lambda scale is not changed mid-matrix. |
| 2026-07-11 | Medium efficiency | Classification checkpoints serialize 22,765 parameters, but a backward-use audit found 19,557 active and 3,208 inactive legacy `channel_attn`/`feat_proj` parameters | Keep architecture unchanged for v2 comparability. After model selection, prune only with exact logits/checkpoint conversion parity and repeat Pi latency/communication benchmarks. |
| 2026-07-11 | Medium terminology | H2.3 C5 MLP anchor uses 104 rich features, while only its weak Ridge branch consumes the additional 64-D `reg_feat` | Corrected in the method chapter and code guide; avoid the inaccurate phrase “H2.3 MLP over rich+backbone features.” |
| 2026-07-11 | High gradient-direction | Critic minimizes `-(D_source-D_target)+GP`, while the feature term returns `+(D_source-D_target)` through GRL; both effective updates appear to increase the signed statistic | Do not claim Wasserstein alignment from v2. Keep queue immutable, inspect A7-noADV if justified, and create separately named adversarial-sign corrected groups after core review. |
| 2026-07-11 | High leakage | Historical `composite_response_risk` divides route/calibration deltas by a range chosen from `true_class`, then builder aliases it to `risk_score` for P4 | Historical P4 is invalid for deployment claims. Before formal regression, compute a selector risk from predicted route/logits/window/calibration references only and add a test that changing true class cannot change the selector risk. |
| 2026-07-11 | High schema | `risk_score` refers to three incompatible quantities: offline composite, QC composite, and final normalized risk ratio | Introduce explicit names and provenance in the unified evaluator; forbid silent legacy fallback in the final policy artifact. |
| 2026-07-11 | High integration | New C5 H2.3+/H8/selector artifacts are not loaded by current `gaps_deploy/final_runtime.py` | Final runtime integration and 1360-window numerical parity remain mandatory Stage 6 gates. |
| 2026-07-11 | High safety, code-resolved 2026-07-13 | QC returned accept when no policy matched and treated missing score/threshold as zero/infinity | Resolved in `6b6d930`: missing/malformed policy inputs reject, and runtime emits no `auto_output_ppm`. Package/checkpoint validation is also strict. Final C5 runtime parity remains a separate deployment-readiness gate. |
| 2026-07-11 | Medium feature-space | Uploaded class-phase prototypes are means of unnormalized `reg_feat`; InfoNCE/cosine paths normalize them, but raw L2 anchor/fit paths do not | Corrected method formula. Log prototype norms and consider normalized-anchor or separate prototype-space ablations after v2. |
| 2026-07-11 | Medium no-gradient term | DA `mmd_proto_loss` is pairwise squared L2 between detached uploaded prototypes and contains no trainable variable | It cannot affect model/prototype/residual updates despite lambda 0.2. Treat as a diagnostic in v2 and do not claim it as an optimized regularizer. |
| 2026-07-13 | High evaluation completeness | First formal regression assembly silently emitted empty R1 because the H8 augmented stream did not carry the separately written rich-only prediction | Resolved and rerun under a new v2 output root. H8 now attaches the column by unique row key; R0-R4 missing/non-finite values fail the suite. The first directory is invalid diagnostic evidence. |
| 2026-07-13 | Low interoperability | Detailed prediction CSVs retain both `H1_source_ridge_ppm` and lowercase `h1_source_ridge_ppm`; Python reads both but PowerShell treats them as duplicate members | Formal compact summaries are clean and unaffected. Remove the legacy lowercase alias from paper/runtime export schemas before spreadsheet delivery. |
| 2026-07-13 | High selection bias | B2 downstream replay was requested after the seed-42 B1-B5 classification test ranking was opened | Label every B2 regression/QC table as exploratory post-screen evidence. Use paired seeds 43-46 and a frozen B2/B5 downstream protocol before promoting B2 to the final method; do not present B2-vs-B5 seed-42 as a confirmatory test. |
| 2026-07-11 | High evaluation completeness | Classification summary run-name parsing accepted only `A0`-style IDs and silently omitted A0T/A4S | Resolved with a shared run-name pattern covering A0T/A4S, confirmation groups updated to match the frozen plan, and parameterized tests. Regenerated summary contains all nine core groups. |

## Task 2 Metric Slice Contract (2026-07-11)

- Implementation: `build_regression_slice_table` now calculates `S_ALL`, `S_CC`, classification-wrong, `S_AR`, and `S_AR intersect S_CC` independently. `S_CC` is selected before QC; the intersection uses `S_CC` as its explicit parent.
- Tests: `python -m pytest tests/test_final_metric_consolidation.py tests/test_matrix_correct_class_regression_analysis.py -q --basetemp .tmp_pytest_metric_matrix_20260711` -> `6 passed`.
- Historical parity only: F6/P4 `S_CC N=5301, RMSE=8.3268605`; `S_AR N=4228, RMSE=5.8496597`; classification-wrong `N=99, RMSE=152.8612`. These values validate implementation and are not promoted to the C12-to-C5 paper table.
- New primary regeneration gate: run this table on the new C5-only aligned regression output after classification and regression stages finish.

## Task 1 Input Freeze, Current Protocol (2026-07-11)

- Primary contract: active source clients C1/C2 and target C5 only. The dataset folder also contains C3/C4 source arrays, but they are inactive and must not appear in the primary run configuration or target metrics.
- Recovered cloud input: `/root/GAPS/results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25` copied read-only to `results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25`.
- Recovery verification: 183 files and 11,522,104 bytes; `run_config.json`, `history.json`, and `server_latest_adapted.pth` are present. Cloud `run_config.json` records rounds 25, strategy `gaps`, profile `strong_cls`, source validation clients C1/C2 only, and target calibration client C5 only.
- Dataset metadata: `split_info.json` records target clients `[5]`, seed 42, target calibration/test `0.2/0.8`, and stratification by client/class/concentration. Its free-text `protocol` field is stale, so the audit must validate structured fields and record the stale label as a warning rather than treating it as authority.
- Audit command: `python scripts/audit_iotj_experiment_inputs.py --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --run-dir results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25 --output results/iotj_experiment_freeze_20260711/input_manifest.json`.
- Manifest result: `complete`; 228/228 artifacts are present and hashed, validation errors are zero, and the one warning marks the stale free-text `protocol` while accepting the valid structured fields.
- Active dataset result: clients exactly C1/C2/C5. C5 calibration/test is `320/1360`; calibration class counts are `80/80/80/80`, test class counts are `340/340/340/340`, and class/concentration counts are frozen in the manifest. C3/C4 do not appear as primary clients or artifact paths.
- Required-run result: exactly one F2 row, `complete`, with 183 files and 11,522,104 bytes. The validated config records rounds 25, strategy `gaps`, profile `strong_cls`, source validation C1/C2 only, target calibration C5 only, DA enabled for 100 steps, target CE 0, and adapted weights used globally.
- Focused tests: `python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest/task1-c12-c5` -> `28 passed`.
- Status: complete. No local simulated training was run; work was limited to unit tests and frozen-artifact analysis.

## Superseded Task 1 Snapshot

- Command: `python scripts/audit_iotj_experiment_inputs.py --output results/iotj_experiment_freeze_20260711/input_manifest.json`
- Output: `results/iotj_experiment_freeze_20260711/input_manifest.json` with 1,207 entries: all `present`; overall status `complete`.
- Historical finding only: the old audit treated C1,C2 -> C3,C4,C5 as primary. Its C3 calibration/test counts were `680/2680`, C4 `320/1360`, and C5 `320/1360`. These rows are not valid primary-protocol evidence after the C5-only correction.
- Failure: the default pytest system temporary directory is not readable in this worker session (`WinError 5` during `tmp_path` setup). `python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest` passes with a workspace-local temporary base.
- Matrix recovery and canonical-directory fix: the controller recovered the six clean-matrix directories. The audit now maps the priority aliases to `F4_C1234_to_C5_fixed_da_strong_r25`, `F5_C1_to_C2345_fixed_da_strong_r25`, `R1_C5_to_C1_fixed_da_strong_r25`, `R2_C45_to_C1_fixed_da_strong_r25`, `R3_C345_to_C1_fixed_da_strong_r25`, and `R4_C2345_to_C1_fixed_da_strong_r25`. The regenerated manifest contains exactly these six complete matrix rows and zero short-name rows.
- Fix verification: `python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest` reports `4 passed`; the new regression fixture creates all six full-name directories with `run_config.json` and `server_latest_adapted.pth`.
- Supersession reason: this manifest and its six-run clean-matrix requirement were created before the C5-only protocol correction and cannot close current Task 1.

## Superseded Historical Snapshot

- The following numbers came from the superseded C12-to-C345/F6 route and are not current primary-paper results.
- Historical C5 classification under F6: 95.96%.
- P4 S_CC: RMSE 8.3269, NRMSE 0.0443, MAE 4.6884, coverage 98.17%.
- P4 Accepted+Review: RMSE 5.8497, NRMSE 0.0339, coverage 78.30%.
- Classification-wrong P4 rows: N=99, RMSE 152.861.
- QC rejected all 99 classification-wrong rows in the current test, plus 1073 class-correct rows; rejected class-correct RMSE 14.412.

## Next Actions

1. Preserve the completed seed-42 v2r1 artifacts and hashes. Treat A0T, A6, and A7 as the main causal comparison; A2-A4/A4S show that client-only mechanisms do not bridge the C1/C2->C5 shift under the current setting.
2. Run seeds 43-46 for the preregistered A0/A0T/A4/A4S/A5/A7 groups on the same ECS/Pi/PC topology. Because A7 only modestly exceeds A0T/A6, do not claim superiority until mean, standard deviation, and paired-seed uncertainty are available.
3. Add a separately named semantic-focused correction screen after confirmation: at minimum A6/A7 without the legacy adversarial term, conventional MMD-squared without the second square, and cross-domain same-class/same-phase alignment. Never overwrite v2 identities.
4. Before formal P4, replace the leaked legacy risk with a newly named deployment-visible selector risk and add invariance/leakage tests. Then rebuild C5-only Ridge/MLP/H2.3+/H8/simple-gate/P4 streams from the selected confirmed classifier, select one candidate on calibration-validation only, and lock C5 test evaluation.
5. Complete QC risk-coverage/random-rejection controls, low-calibration regression/QC stress, Pi/PC latency-memory-communication benchmarks, and availability stress before freezing the paper evidence pack.

## Method and Paper Audit (2026-07-11)

- New paper-method draft: `docs/paper/iotj_system_methodology_20260711.zh.md`.
- Code guide updated: `代码文件介绍.md` now opens with the C1/C2->C5-only protocol, current ablation/controller/evaluator entrypoints, and the correct regression candidate roles.
- Classification architecture audit: 100x8 input; DS-TCN 8->32->48->48 with dilation 1/2/4; 4-head self-attention; learned attention pooling; 64-D normalized classification embedding; 22,765 serialized parameters and 19,557 parameters active in the current forward path.
- Client loss audit: standard CE plus optional class-phase InfoNCE (`lambda=0.05`, `tau=0.1`) and previous-round feature MSE (`lambda=2.0`, active from round 2). No Flower regression loss.
- Aggregation audit: FedAvg sample weighting; EMA prototype alpha 0.8; selective minimum scale 0.3; actual selective warmup is 3 because v2 commands do not override the server CLI default.
- Server audit: every DA group includes source rehearsal CE. A0T alone sets target CE to 1.0. A5-A7 set target CE to 0 but still consume C5 class/phase labels in class-conditional terms, so they remain calibration-assisted.
- Regression audit: C5 target heads include both Ridge and MLP. H2.3 is the expanded-grid per-gas MLP anchor over 104 rich features; H2.3+ may blend a 168-D rich+reg-feature Ridge. H8 is a C5 target Ridge over 104 rich features plus three C1/C2 source-head predictions. C4 rescue is disabled.
- Adversarial audit: the current critic/GRL signs do not justify a domain-confusion claim. The v2 result measures the legacy implementation; a corrected sign requires a new run identity.
- Selector-risk audit: the historical P4 input risk leaks `true_class` through ppm-range normalization and is disqualified from deployment evidence. The formal selector must use a new deployment-visible risk schema.
- Runtime/QC audit: the checked-in runtime now fails closed when policy/schema fields are absent, and package/checkpoint assets are strict. The available final bundle still serves the old C12->C345 policy, so C5-only artifact integration and 1360-row numerical parity remain blocking before a “closed-loop deployed system” claim.
- Paper packaging rule: contribution claims follow the core result. A7 is not automatically the proposed method, P4 is not automatically the final policy, and a negative DA result shifts the story toward target personalization, reliability, and real cloud-edge deployment rather than being hidden.
- Verification: `python -m pytest tests/test_flower_classification_contract.py tests/test_iotj_c5_regression_inputs.py tests/test_iotj_c5_p4.py tests/test_final_metric_consolidation.py tests/test_iotj_classification_summary.py -q --basetemp .tmp_pytest_method_docs_20260711` -> `51 passed, 2 dependency deprecation warnings`.

## C5 Regression Closure Smoke (2026-07-11)

- Scope: engineering and historical-artifact parity only. These numbers do not replace the pending real cloud-edge A-run results.
- The new input builder enforces source C1/C2, target C5, and exact C5 counts of 320 calibration plus 1360 test windows. It exports aligned classifier backbone features and the existing R3aK16 C1/C2 source-reference prediction as candidate inputs.
- H2.3+ uses a calibration fit/validation split of 75%/25%, expanded MLP/Ridge grids, and a constrained blend selected without test labels. Historical selection chose blend weight 0; test RMSE was 21.2182 and S_CC RMSE was 12.3807.
- H8 was rerun with `--disable-c4-rescue`, making the target contract genuinely C5-only. Its historical test RMSE was 16.6166 and S_CC RMSE was 11.5028.
- P4 searches risk thresholds only on 80 calibration-validation rows. The selected threshold 0.0090404 routed 71/1360 test windows (5.22%) to H8 and produced test RMSE 17.3559. The fixed H8 expert remained better at 16.6166, while the unattainable per-window oracle was 14.8724.
- Interpretation: the fixed C5 target Ridge with source-reference features is currently the strongest regression candidate. `risk_score` alone does not provide a stable expert selector in this replay. The new classifier seeds must confirm this ordering before the paper method is frozen.
- Verification: `python -m pytest tests/test_iotj_c5_regression_inputs.py tests/test_iotj_c5_h23_plus.py tests/test_iotj_c5_p4.py -q`; direct CLI execution is covered for P4 so pytest path injection cannot mask import failures.

## Spec A 确认可观测候选冻结（2026-07-17，Task 10）

### 结论和声明边界

- Spec A 到当前修订只实现了 **Confirmation Experiment Observability Framework（确认实验可观测框架）**。B2/B5 的正式十运行分类确认尚未开始；本节的测试、合成 Gate 和文档冻结不能写成正式确认结果。
- 冻结方向仍是 `C1/C2 -> C5`，候选为 B2/B5，种子为 42--46，每次正式运行 25 轮。真实 C5 target-test 从未在本轮工作中打开；ECS/Pi 正式 smoke、预检和十个 25 轮运行也均未启动。
- 历史 `feaa75b` 主方向 seed-42 结果与跨方向 seed-42 结果只属于 screening/appendix；它们不进入五种子确认均值、sample std 或显著性结论。B2 仍标记为 post-screen exploratory，B5 为 predeclared full method。
- 通信必须分三层报告：Layer 1 是逻辑 payload 分量，Layer 2 是确定性序列化后的 Flower application message 字节数与 SHA-256，Layer 3 传输层明确为 `transport_status=not_collected`。未采集传输字节绝不写成 `0`，也不从应用层字节推算。

### 经过审计的 Git 起点和 Task 1--9 完整链

Task 10 修改文档前，分支为 `codex/iotj-confirmation-observability`，HEAD 精确为 `12b3bc45dd8ceff7098e543cf94d789a2eb338d7`，且 `a920ecdbdbea250220343d63926cb370178cdc5e` 是其祖先。tracked worktree 和 index 均为空；仅保留 brief 允许的未跟踪 `.tmp*`、`.t9` 和旧 Gate 证据。批准链如下（按祖先到后继顺序）：

```text
de23322 docs: specify IoTJ confirmation observability
c3dc7fa docs: plan IoTJ confirmation observability
6d01259 feat: add confirmation event observer
cc982cb fix: fail closed in confirmation observer
0b31e30 feat: audit Flower application messages
4e797a7 feat: observe confirmation Flower phases
5394b41 feat: sample confirmation training resources
0381597 fix: stabilize resource sampler identity and peak RSS
8f54e66 feat: freeze confirmation protocol manifests
9ed42f5 fix: make confirmation freeze transactional
6aec05c feat: orchestrate immutable confirmation attempts
95d5faf fix: close confirmation controller safety gaps
18effa3 fix: harden formal confirmation runtime
d352435 fix: secure remote confirmation lifecycle
7cac204 fix: enforce confirmation lifecycle invariants
d6412c3 fix: complete confirmation failure handling
5e497b7 feat: validate confirmation attempt evidence
0b386c4 fix: harden confirmation evidence coverage
b861f06 fix: deduplicate resource sampling points
d06cee1 fix: bind transport status in confirmation protocol
3ce17dc feat: summarize confirmation system evidence
e6dbbd9 fix: seal confirmation summary inputs
d210c39 fix: extend sealed summary transaction
b84b76c fix: finalize summary publication transaction
8827461 test: gate observer numerical equivalence
8b0bca7 fix: harden observer equivalence evidence
3005e61 fix: bind observer evidence identity
929cd99 fix: enforce exact observer numeric types
b7ba98c fix: validate every resource identity row
b7106d4 fix: harden observer equivalence contracts
12b3bc4 fix: bind formal resource recovery paths
```

训练关键路径保护审计：

```text
git diff a920ecdbdbea250220343d63926cb370178cdc5e -- config.py client.py model.py federated_dataset.py gaps_flower/task.py gaps_flower/domain_adaptation.py
```

输出为空，说明 Spec A 没有修改这六个训练关键文件。`scripts/run_iotj_confirmation_observability.py`、`scripts/run_iotj_observer_equivalence_gate.py` 及其两个测试模块的四文件 `py_compile` 通过；`git diff --check` 通过；静态审计后 tracked worktree/index 仍为空。

### Task 10 fresh verification

- 九文件 Tasks 1--9 related suite，系统临时目录唯一短 basetemp：`355 passed, 4 skipped, 2 warnings in 215.39s`，exit 0。
- 完整 `python -m pytest -q`，另一系统临时目录唯一短 basetemp：`700 passed, 4 skipped, 2 warnings in 295.37s`，exit 0。
- 这组 fresh 结果补充并确认 Task 9 批准前的 Gate `59 passed, 1 skipped`、controller `134 passed, 1 skipped`、related `355 passed, 4 skipped`、full `700 passed, 4 skipped`。

### Task 9 最终本地 Gate 证据

- 最终目录：`.tmp_iotj_observer_gate_b2_task9_final_v10/` 和 `.tmp_iotj_observer_gate_b5_task9_final_v10/`；二者都是 synthetic、local-only、unstaged evidence，不是正式拓扑或真实 C5 证据。
- B2/B5 都为 `status=equivalent`、`max_abs_delta=0`、无 mismatch；OFF-A/ON/OFF-B 的冻结初始点、每轮张量/统计、最终 checkpoint 及独立 live trace/audit 绑定一致，跨比较精确匹配 8 条真实 Flower application message。
- B2 report SHA-256：`1191d766e932360c8ed2e83b9258c3e18c284010ba5dbe5df249e6de8ea48646`。
- B5 report SHA-256：`a7d8a437e87d6703b8255d9431a0d47472a278bbe1d603b5658cbe9eda5d7d96`。

### 失败审查所保留的工程教训

- 观测代码必须保存并恢复 Python、NumPy、PyTorch CPU/CUDA 的 RNG 状态；即使 logits 相同，observer 引入的随机数消费也会污染后续正式训练。
- FitRes 必须在 observer audit 后再次取样并比较；只审计 observer 看到的对象会漏掉 post-observer mutation。
- 不能让 observer 自己证明自己：独立 common live trace 要与 events sidecar 的 message bytes、SHA、round/client/proxy identity 做双向绑定。
- 文件集合、字段集合、schema、JSON numeric type、regular-file 类型、run/attempt/round/client/host/producer identity 和恢复路径都必须精确；“内容大致相同”不构成 Gate。
- Pi 端先生成的 Linux resource 路径恢复到 Windows 后必须重新绑定为 attempt 内精确 regular-file 路径，不能接受旧绝对路径或旁路文件。
- DA diagnostics 不只比较路径字符串；要比较引用 JSON 的字节 SHA 和规范化语义，同时保留 allowlist 之外的全部 key。
- failed/invalid/aborted attempt 不删除、不覆盖、不改写为 canonical。它们保留 immutable status chain、controller log、raw sidecar 和 audit，既防止 metric-driven rerun，也为故障复盘提供证据。

### 后续授权与停止条件

用户于 2026-07-17 预授权：仅在 Tasks 10--12 完成，且 candidate freeze、三主机 preflight、B2/B5 formal smoke、archive/hash/schema/runtime identity 与十运行 dry-run queue Gate 全部 fail-closed 通过后，可无需再次人工批准而启动冻结的十个 25 轮运行。任一 Gate 失败仍必须立即停止并保留证据。Task 10 本身只生成本地 archive/manifests，不执行 preflight、formal smoke、训练或 C5 evaluation。

操作入口：安全本地 Gate 使用 `python -m scripts.run_iotj_observer_equivalence_gate --group B2|B5 --output-root <new-empty-root>`；本地冻结使用 `python -m scripts.freeze_iotj_confirmation_protocol --confirmation-commit <FULL_HEAD> --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --archive-output results/iotj_main_confirmation_observability_20260715/source/confirmation_source.tar --command-root results/iotj_main_confirmation_observability_20260715_commands --summary-root results/iotj_main_confirmation_observability_20260715_summary`。formal topology、preflight、`--validate-inputs-only` dry-run queue、sealed summarizer 和正式 controller 的完整可复制命令及其网络边界见 `docs/experiments/iotj_latest_handoff_20260715.zh.md` 第 11.3 节。

## 论文指标闭环优先与初步证据包（2026-07-17）

### 优先级调整

- 当前停止继续扩展 Spec A 内部诊断范围；不新增 Observer 功能，不继续追逐非阻塞审计项。
- 正式 10×25 confirmation 仍未启动。B2 formal-topology OFF/ON 已满足 `status=equivalent`、`max_abs_delta=0`；B5 在 round 2 保留真实 parity 失败，因此仍禁止生成 confirmation freeze record 或把任何新运行写成正式 confirmation。
- 当前论文闭环按“分类 -> 回归 -> QC -> 真实拓扑系统 pilot”组织。历史 `feaa75b` 主方向 seed-42 只作 screening/historical；三个跨方向 seed-42 只作 appendix/generalization；二者都不进入未来 confirmation mean/std。

### 已生成的统一入口

- 结果根：`results/iotj_preliminary_paper_metrics_20260717/`
- 审计工作簿：`iotj_preliminary_paper_metrics_20260717.xlsx`
- 轻量表：`paper_kpis.csv`、`classification_directional.csv`、`classification_paired_comparison.csv`、`system_pilot_round_metrics.csv`、`system_pilot_resource_summary.csv`
- 证据与边界：`evidence_sources.csv`、`claim_boundary.md`、`preliminary_metrics_manifest.json`
- 工作簿含 Overview、主方向分类、跨方向配对、正式回归、正式 QC、消息、时延、资源、Observer 开销、claim boundary 和源文件 SHA-256。公式检查无错误；首页轻量快照视觉检查通过。artifact-tool 原生 render 在 Windows 发生可重复 native crash，Excel COM 又被 Office RPC 拒绝，因此视觉检查使用只读 openpyxl+Pillow fallback，并在 `overview.render_report.json` 中明确记录。

### 当前可用于论文分析的初步指标

- 主方向 seed-42：B2 Accuracy/Macro-F1 为 `0.992647/0.992657`；B5 为 `0.988971/0.988990`；B2-B5 Accuracy 为 `+0.3676 pp`。这只是历史单种子描述，不是 confirmation 结论。
- 正式 C5 回归：B2 R4 FULL actual-route RMSE/NRMSE 为 `14.6564/0.1059`；B5 为 `17.4473/0.1352`。
- 正式 C5 QC：B2 HC90 Yield 为 `0.8949`，accepted RMSE/NRMSE 为 `11.5866/0.0747`；B5 HC90 Yield 为 `0.8824`，accepted RMSE/NRMSE 为 `15.3599/0.1151`。
- B2 真实 ECS + Pi C1 + PC C2 两轮 pilot：serialized application messages 合计 `1,395,868 bytes = 1.3312 MiB`；logical payload 合计 `1,391,310 bytes`；transport bytes 未采集。
- 两轮平均 round wall 为 `193.0714 s`，平均 server DA 为 `168.5313 s`，DA 约占 round wall 的 `87.3%`。当前系统瓶颈首先是 ECS server DA，而不是消息体大小。
- Pi/PC 平均 local train core 为 `11.1061/20.4351 s`；training-overlap peak RSS 为 `511.91/426.21 MiB`。Pi training-overlap 平均/峰值主机 CPU 为 `80.50%/90.63%`，平均/峰值温度为 `55.0/58.4 °C`。只有两轮，不能据此宣称 Pi 持续快于 PC。
- Observer 编码/序列化/I/O 自测总开销为 `24.8501 ms`；与 386 秒级两轮 wall time 相比很小，但仍只属于 pilot 口径。

### 下一步最小实验原则

1. 不再为了完善 Spec A 而推迟论文指标闭环。
2. 先使用本包更新论文主表/系统表草稿，并把 B2 两轮系统值明确标为 preliminary。
3. 低校准当前只完成 budgets、sample-key、stratified sampling、统计合同和工具校验，不生成正式批量结果；正式结果必须等待十个 confirmation run 完成并冻结 classifier prediction stream。
4. B5 Gate 未解决前不启动正式 10×25；如需系统初步值，继续使用已完成的 B2 real-topology pilot，不把它伪装为 25-round total。

### 论文导向分析与剩余证据矩阵（2026-07-17）

- 新增 `results/iotj_preliminary_paper_metrics_20260717/iotj_preliminary_results_analysis.md`：将工作簿严格定位为“已有算法证据与新系统 pilot 的论文初步指标整合”，不是新的完整训练结果。
- 新增 `results/iotj_preliminary_paper_metrics_20260717/iotj_paper_evidence_gap_matrix.csv`：共 10 项，P0/P1/P2 分别为 3/5/2 项，逐项记录论文价值、最低实验范围、前置条件、计算/工程成本、主文角色和 fail-closed 停止条件。
- 两个新增产物的 bytes/SHA-256 已写入 `preliminary_metrics_manifest.json`；本次只整理与分析已有证据，没有新增训练、打开新的 test 排名或扩展 Observer/Gate。
- 初步系统结论收紧为：B2 两轮真实拓扑 pilot 中，server-side DA 是当前测得的主导耗时组成（约占总 round wall 的 87.3%），Pi local training 不是主要时间瓶颈；transport latency 未独立采集，不能表述为已排除网络瓶颈。
- 修订后的最小路线为：P0 先做 B5 第一处分叉的最小修复，再运行同一冻结 revision 下的 B2/B5 × seeds 42--46 十个 25-round canonical runs，并同步收集正式通信/时延/资源证据；P1 才是正式 low-calibration、bundle、1360-row parity 与 Pi/PC inference；P2 为 availability 与至少 1 h 稳定性，详细长稳结果优先进入 appendix。
- 下一项最值得投入计算资源的实验是 P0 十个正式 25-round confirmation runs；按当前 B2 pilot 的顺序运行粗略下限约 13.4 h，实际应预留 14--24 h。它同时补齐分类 mean/sample std、正式 25-round 系统表，并冻结后续 low-calibration/deployment 所需 prediction stream。

## Regression Provenance Audit（2026-07-17，只读）

### 审计范围与产物

- 本轮只读取现有训练入口、正式 R0--R7 汇总脚本、正式 H8/QC 结果清单和当前部署代码；没有修改模型、loss、数据协议、训练超参数、回归头或正式结果，也没有启动任何新的回归训练。
- 审计产物位于 `results/iotj_regression_provenance_audit_20260717/`：
  - `regression_provenance_map.csv`
  - `regression_dependency_graph.md`
  - `regression_federated_boundary_audit.md`
  - `regression_source_head_followup_plan.md`

### 已确认的训练方式

- Source Ridge（H1）：Ethanol、CO、Ethylene、Methane 都先把 C1/C2 的 source-train 窗口读入同一进程，再按 gas 过滤并 pooled fitting；alpha 由合并后的 C1/C2 calibration 选择，最终在 pooled train+calibration 上重拟合。不存在 Flower 消息、FedAvg 或按客户端参数聚合。
- Source per-gas MLP（H2）：四个 gas 各有一个 MLP，但每个 MLP 都使用同一进程中的 C1+C2 pooled source data 集中训练；不存在联邦训练。
- Source shared MLP（H3）：C1/C2 四类 source data 合并后，加入 predicted/oracle gas route one-hot 特征，集中训练一个 shared MLP；不存在联邦训练。
- 上述 source heads 读取的是已生成的 window feature/label/metadata 数组。它们代表 source 窗口数据在一个进程中的集中拟合，不能表述成“source raw/window data 从未集中”。
- R0 的 R3aK16 source checkpoint 属于另一条 offline FedAvg-style reference：各客户端本地训练后按样本数加权平均 checkpoint；现有实现是单机/文件式模拟，不构成真实 Flower 网络回归训练。R0 不进入 R4 的数值计算，但当前 formal input builder 与 legacy runtime 仍保留该 artifact plumbing。

### R4/H8、QC 与部署依赖边界

- 正式固定 R4/H8 数值链路为：classifier logits -> predicted gas route -> H1 source Ridge prediction + H2 source per-gas MLP prediction + H3 source shared MLP prediction -> C5 calibration 上拟合的 per-gas augmented target Ridge -> R4 ppm prediction。
- 因此 R4 的数值推断强依赖 H1/H2/H3 三个 source heads，任一缺失都不能视作同一个正式 R4。R4 本身不依赖 target per-gas MLP（R2/H23）或 R0 R3aK16 prediction。
- 正式 high-coverage QC 还使用 classifier confidence、H23/R4 disagreement 和 H1/H2/H3 source-head spread。因此 target MLP 虽不进入固定 R4 数值路径，却进入当前正式 QC 的风险特征；三个 source heads 同时影响 R4 prediction 与 QC risk。
- 现有 Git-tracked runtime 仍是 legacy C12 -> C3/C4/C5 路径，并带有 predicted-CO 分支/回退语义；它不等于已经冻结的 formal C5 R4/H8 + high-coverage QC runtime。最终 C5 deployment bundle 及 1360-row parity 尚未完成，论文不能写成“正式 R4 runtime 已部署并验证”。

### 论文联邦边界与后续顺序

- 当前可安全称为 federated 的核心是分类训练；R0 最多称为 offline FedAvg source-regression reference。H1/H2/H3 应称为 `centrally pooled multi-source regression references`，C5 Ridge/MLP/H8/QC 应称为 `target-personalized calibration and decision pipeline`。
- 安全总表述：GAPS combines real-device federated classification with centrally pooled multi-source regression references and target-personalized calibration/QC。禁止笼统写成“端到端全流程联邦”或“source regression data 从未集中”。
- 后续执行顺序冻结为：P0 完成 -> final B2/B5 classifier checkpoints/prediction streams 冻结 -> Source-head dependency ablation -> 决定是否需要 distributed Ridge -> 正式 12/24/48/80/Full low-calibration batch。
- 最有价值的回归后续实验是：首先在同一冻结 prediction stream 和同一 C5 calibration keys 下成组执行 source Ridge/per-gas MLP/shared MLP removal 以及 source-head-free baseline；其中 Experiment A（去掉 source Ridge）与 Experiment D（只保留 target Ridge/MLP）最直接回答 final dependency 和论文方法边界。
- 现在不建议直接实现 distributed sufficient-statistics Ridge。只有 Experiment A 显示 source Ridge 对 R4/QC 有稳定且重要的贡献时，才执行 C1/C2 本地计算 `X^T X`、`X^T y` 并由 server 聚合求解的 equivalence experiment；否则该实现对投稿的边际证据价值不足。
- 产物完整性：`regression_provenance_map.csv` 为 `12395 bytes / e1656bedbd3c8441d9e2253d69fae29169b934be05d1e87eb11769901bec70e5`；`regression_dependency_graph.md` 为 `6762 / 6285d4280df5e8240071a21513d614b8ce93ea40bd520974f16057cd34c1851a`；`regression_federated_boundary_audit.md` 为 `5983 / eb4c911e9d0ace44e030d34336989d707effff87e0ac1968e10afe6d18e1db53`；`regression_source_head_followup_plan.md` 为 `4838 / 0ca725610dde0504d3188b42c9df96bee4371f77a62cd538be09a9f7156c2478`。

### P0 B5 blocker 当前证据边界

- P0 优先级未改变，也未启动任何正式 25-round run。B5 formal OFF/ON 报告继续保留为 `observer_path_mutation`，`max_abs_delta=0.01269597525242716`；失败证据未被覆盖。
- 第一处分叉审计目前只能证明：round 2 的 FitRes arrival order 为 OFF `[C2,C1]`、ON `[C1,C2]`；每客户端 normalized FitRes、post-aggregate/pre-DA checkpoint 均一致，最早捕获到的数值分叉位于 server DA。B2 round 2 顺序一致且 `max_abs_delta=0`。
- 固定状态 replay 证明：相同顺序精确等价、RNG 状态未改变；逆序只改变初始 prototype-loss scalar `0.00390625`，但该 replay 的梯度 mismatch count 为 0。因此当前证据不足以把“排序 FitRes”宣布为已证实根因，也不足以授权补丁。
- focused replay test 为 `14 passed in 37.64s`。本轮新增未跟踪的只读诊断脚本 `scripts/diagnose_b5_fixed_state_order_replay.py` 与 focused test `tests/test_b5_fixed_state_order_replay.py`，但没有修改训练/runtime 数值路径，也没有形成候选修复。由于尚未获得 formal OFF-A/OFF-B 的独立重复性证据，没有创建新 candidate commit，也没有生成 confirmation freeze record。

### P0 B5 blocker 根因确认与最小修复（2026-07-17）

- 独立 formal OFF-B 使用保留的 `7ec77e3` archive、相同 frozen checkpoint/命令/数据与新的非 canonical `a997` 身份完成两轮。旧 OFF-A 与新 OFF-B 的比较为 `status=environment_nondeterminism`、`max_abs_delta=0.02182745933532715`；因此原 OFF/ON 差异不能归因于 Observer。repeat report SHA-256 为 `6be58896b1fabd5425538b84dd28d907facf3d3978e7a86e47501c745d3b3fd7`。
- 第一处真实分叉已前移到 round 1：frozen initial checkpoint、C1/C2 FitIns、两个客户端 FitRes 张量/指标以及 plain parameter aggregate 都精确相同；旧 OFF-A 的 FitRes arrival order 为 `[C2,C1]`，新 OFF-B 为 `[C1,C2]`。到达顺序未经规范化进入 float32 prototype/statistics reduction，首先使 `semantic_protos["0,0"][0]` 相差 `1.0728836059570312e-05`，round-1 semantic prototype 最大绝对差为 `1.5351921319961548e-05`，随后传播至 DA 和 round 2。
- 结构化证据位于 `results/iotj_main_confirmation_observability_20260715/smoke/b5_formal_off_repeat_a997_7ec77e3_v2/b5_first_divergence_report.json`。旧 B5 OFF/ON 与所有失败 probe 目录均未覆盖或删除。
- 最小修复仅在 `gaps_flower/strategy.py` 增加 `canonicalize_fit_results()`，要求上传的正整数 `client_id` 唯一，并在 `GapsStrategy.aggregate_fit()` 的所有 float32 aggregation/statistics/DA 路径前按该身份排序。未修改模型、loss、B2/B5 配置、数据协议、训练超参数或 server DA 数学定义。
- TDD/验证：新排序合同先 RED 后 GREEN；focused B5 诊断 `18 passed`，Observer Gate tests `72 passed, 1 skipped`；修复后的本地 B2/B5 OFF-A/ON/OFF-B 均为 `status=equivalent`、`max_abs_delta=0`。confirmation 其余模块短路径验证为 `290 passed, 3 skipped`，summary 在 `D:\itj7` 为 `39 passed, 1 skipped`；较深 basetemp 的失败均由 Windows `MAX_PATH` 复现，不属于代码回归。
- 当前仍未创建新 candidate commit/archive/freeze record，也未启动正式 25-round run。下一步是把最小修复与诊断/文档形成新 candidate commit，重建唯一 archive 和 hashes，重跑三主机 preflight 与同一 revision 的 B2/B5 formal smoke；只有二者都精确等价才可进入十运行队列。

### P0 confirmation freeze 已完成（2026-07-17）

- 最终算法源码 revision 冻结为 `2ef7aea77b9dfabdd09da4f38742907a37c58c30`。唯一 source archive 位于 `results/c2e/source/confirmation_source.tar`，SHA-256 为 `52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`；dataset manifest SHA-256 为 `fb8946da138bea5aa829dd1f5b733561a443083beb77a873e7173cbc95fcd430`。
- 三机 preflight 已通过，ECS、Raspberry Pi、PC 均绑定相同 archive/protocol/dataset。首次使用长 candidate 路径生成 archive 时因 Windows `MAX_PATH` 在 staging 文件名处失败；该失败目录被保留，随后只缩短输出根为 `results/c2e*`，未改变归档内容或算法 revision。
- B2 formal report：`status=equivalent`、`max_abs_delta=0`、message/trace `matched`、每客户端每轮最少 12 个资源采样、0 mismatch；report SHA-256 `1f395b7209b94abc049ab16679c483e3d89565212e2266f868506dbbb1ad92d1`。
- B5 formal report：`status=equivalent`、`max_abs_delta=0`、message/trace `matched`、每客户端每轮最少 12 个资源采样、0 mismatch；report SHA-256 `36016723cfa253f1e7df1292964dbd4c4ef0960a5e4fe2e0a8c75f8088f8207f`。
- `results/c2e_summary/confirmation_freeze_record.json` 已将 commit、archive、regular members、dataset、protocol 及两个 smoke report 哈希绑定；`target_test_opened=false`。冻结输入 validator 返回规定的 10-run 交替队列。
- 下一步是从该 archive 启动正式 10×25 canonical queue。所有训练侧 application bytes、round/local train/server aggregation/server DA/wall time、Pi/PC RSS/CPU 与 Pi 温度随运行同步采集；formal smoke 的两轮指标仍只作 Gate/diagnostic，不进入正式论文 25-round 汇总。

### 正式 10×25 confirmation 已启动（2026-07-17 22:04 Asia/Shanghai）

- 后台 controller PID：`1664`；stdout/stderr：`results/c2e_runs/controller.stdout.log` 与 `results/c2e_runs/controller.stderr.log`；raw/runtime 根：`results/c2e_runs/raw`、`results/c2e_runs/runtime`。
- 首项为 `c12_to_c5__b2__s42 / c12_to_c5__b2__s42__a001`，已记录 `preflight_passed` 并进入 `state=running`。attempt provenance 精确绑定 `confirmation_commit=2ef7aea77b9dfabdd09da4f38742907a37c58c30`、archive `52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`、dataset `fb8946da138bea5aa829dd1f5b733561a443083beb77a873e7173cbc95fcd430`。
- PC `resource.jsonl` 已开始写入，说明正式训练侧系统采样与 run 同步启动。controller 将完成当前 attempt 的 validator 后才按冻结顺序进入下一项；failed/invalid attempt 会保留且不计入 confirmation。
- 运行时监控只读入口：检查 PID `1664`、`attempt_status.json`、`controller.stderr.log` 和最新 `resource.jsonl/events.jsonl`；禁止在队列运行中重新打包源码或手工并行启动相同 run。

### 断网失败与正式队列恢复（2026-07-18）

- 首次 `B2 seed-42 / a001` 于 2026-07-17 22:25 Asia/Shanghai 因本地到 ECS 的 SSH 状态查询连续 30 秒无响应而 fail closed：`subprocess.TimeoutExpired`。Controller 正确写入 `state=failed / reason=process_failure` 后退出；ECS、Pi、PC 未发现残留该 attempt 的训练或采样进程。
- `a001` 已完成 round-1 并收到 round-2 FitIns，但未完成 25 轮且未经过 validator，因此永久排除在 confirmation mean/std 之外。其 PC `resource.close.json` 与事件证据保留在原目录，不覆盖、不删除。
- 2026-07-18 13:09 重新执行相同冻结 archive 的三机 preflight，10-run queue、commit/archive/dataset hashes 全部通过。Controller/validator/训练代码与冻结 `2ef7aea` 无差异，没有重新冻结算法 revision。
- 2026-07-18 13:10 使用相同 `results/c2e_runs/raw` 恢复队列，新 attempt 为 `c12_to_c5__b2__s42__a002`；新 Controller PID `25268`，日志为 `results/c2e_runs/controller_restart_20260718_131057.stdout.log`、`controller_restart_20260718_131057.stderr.log`。`a002` 已进入 `preflight_passed / running`，PC resource sampler 已开始写入，stderr 为 0 bytes。
- 后续监控应读取 `results/c2e_runs/latest_controller_launch.json` 获取最新 PID/日志，不再硬编码旧 PID `1664`。网络再次中断时仍按 fail-closed 保留 attempt，并从下一 attempt ID 恢复，禁止把 failed attempt 改写为 canonical。

### 第二次断网后的安全清理与 a003 重启（2026-07-18）

- `B2 seed-42 / a002` 在 2026-07-18 17:12 Asia/Shanghai 因 ECS SSH port 22 connection timeout 被 Controller 标记为 `failed / process_failure`。PC 已完成 20 个完整 `client_fit_end`，round 21 的 C2 local training 已结束，但 server 未完成 25 rounds/validator，故不能计入正式结果或系统统计。
- 网络恢复后发现 Controller 无法在断网时清理的 ECS supervisor/child（`791300/791301`）仍在运行；Pi、PC 无残留进程。通过 `server.registration.json` 的 label、launch token、PID/PGID、start ticks 逐项验证身份后，仅终止该受控 process group；注册记录被远端清理逻辑归档为 `.cleaned`。
- 在删除远端 runtime attempt 目录前，a001/a002 的 ECS 与 Pi 原始目录均已回收到本地。保留位置为 `results/c2e_runs/raw/c12_to_c5__b2__s42/`：a001 约 `7,574,722 bytes`，a002 约 `91,623,842 bytes`，均包含 PC/ECS/Pi 子目录。之后仅删除 ECS/Pi 上这两个精确 attempt 的 runtime dirs，未删除本地证据、冻结 archive、协议或命令 manifest。
- 清理后 ECS/Pi 无残留 `a001/a002` process/attempt 目录；同一 archive 的三机 preflight 再次通过。2026-07-18 18:25 启动 `c12_to_c5__b2__s42__a003`，Controller PID `47056`；已达到 `preflight_passed / running`，新 stderr 为空，PC resource sampler 已开始写入。训练算法、数据、超参数和 confirmation archive 均未修改。

### Controller deadline 终止、证据回收与 a004 重跑（2026-07-19）

- `B2 seed-42 / a003` 不是数值、checkpoint、schema 或 parity 失败：Controller stderr 明确为 `TimeoutError: server process exceeded formal timeout`。该 Controller 的默认 `--run-timeout-seconds=18_000`（5 h）在 2026-07-18 15:26 UTC 终止了尚未完成的 server process；最后完整的 C2 `client_fit_end` 为 round 22。它未完成 25 rounds/validator，故永久作为 `failed / process_failure` evidence，不能续跑第 23 轮，也不能计入 confirmation 或系统汇总。
- a003 的逐阶段事件给出了真实耗时边界：22 个完成 round 的 mean/median wall time 为 `803.56/810.89 s`；PC C2 local train 为 `647.97/658.46 s`，ECS server DA 为 `152.68/149.69 s`，Pi C1 local train 为 `41.84/42.25 s`。这说明此正式 CPU 条件下的关键路径是 PC C2 local train 加上 server DA，不是 Observer 或消息字节；PC resource sampler 的全程 observer self-cost 为 `5.238 s`（约 5 h 墙钟时间），不能解释该数量级。此前从两轮 pilot 推出的 14--24 h 十运行粗略预估已被该实测证据取代：B2 单个完整 25-round run 应先按约 5.6 h 加 validator/recovery 余量规划，B5 可能因其额外 DA 分支更慢，正式 10-run 串行队列应按数天而非一天估计。
- a003 与此前 B2 two-round smoke/pilot 的只读配置/资源对照已完成。两者均为 CPU、C2 train/test windows 均为 `2360/680`，但 smoke 的 `local_epochs=1`，正式 confirmation 固定为 `local_epochs=5`；这是预期的首要 5x 差别。仍有未归因的性能差：smoke 每 epoch 为 `14.67/26.20 s`，而 a003 折算每 epoch mean 为 `129.59 s`（总 local train 增幅约 32x，而非仅 5x）。a003 C2 training 时 process-tree CPU 为 6 logical CPUs 的 mean `23.66%`（约 1.42 cores），低于 smoke 的 `37.76%`（约 2.27 cores）；这支持 PC 当时有效并行度/主机状态不同的判断，但当前采样不能把原因进一步归因为 power plan、后台竞争、线程池或 OS 状态。不得在 canonical a004 中途修改 epochs、device、线程、模型或算法；若要改变这些条件以缩短时间，必须停止该队列、重新冻结并重新 smoke，不能作为同一 confirmation evidence。

### a003 慢点诊断与双轨证据设计（2026-07-19，只读/未执行）

- a003 的 ECS/Pi/PC 原始证据已完整保留在本地；它在 round 22 后被 5 h Controller deadline 终止，继续保持 `failed`，不进入任何算法或系统正式统计。新产物为 `results/iotj_a003_timing_diagnosis_20260719/a003_round_timing_diagnosis.csv` 和 `a003_vs_b2_pilot_timing_analysis.md`，逐轮覆盖 wall、C1/C2 train/fit、aggregate/DA/non-DA、同步残差、Pi/PC RSS/CPU 与 Pi temperature。
- 22 轮实测：round wall mean `803.56 s`；PC C2 train `647.97 s`（`80.6%` wall，pilot 的 `31.71x`）；Pi C1 train `41.82 s`（pilot `3.77x`）；ECS DA `152.68 s`（pilot `0.91x`）；waiting/synchronization 合并残差 mean `2.82 s`。因此当前主要 slowdown 是 PC C2 local training；DA 仍是第二大绝对耗时，但不是本次变慢来源；无随 round 单调恶化的 wall/PC-train 证据。
- `a004` 已在本条决策前完成 25 rounds，但 C2 resource coverage `0.938536 < 0.95` 被 validator 标为 `invalid`；它同样不可进入统计。不得启动 a005 或后续 B5，直到改变执行策略并完成新的 preflight。
- 依据双轨决策，已核对并停止旧 confirmation Controller PID `34712`；a004 的 `attempt_audit.json` 已绑定 ECS/Pi/PC 的 events、resource JSONL 与 close summaries，本地证据保留，Controller 不会继续自动分配 a005。后续只允许在新的 Track-A 或 Track-B execution manifest 审核通过后启动对应队列。
- 双轨设计（未执行）位于 `results/iotj_a003_timing_diagnosis_20260719/track_a_track_b_execution_design.md`：Track A 用同一算法 archive/数据/25r/5epoch/DA，在较快环境的两个独立 logical Flower clients 上获取 B2/B5 五种子算法统计；Track B 仅保留 B2/B5 各一个预声明真实 ECS+Pi+PC 25-round representative run，用于通信/时延/资源表。二者必须分别冻结 execution-topology manifest，且论文表格/claim 不得混淆。
- 先回收再清理：a003 的本地 `raw/` 已保留 PC 8、ECS 142、Pi 11 个文件（共 161 文件、`112,270,916` bytes）。ECS/Pi 原始 attempt 目录分别约 11 MiB/50 MiB；文件数核对一致后，才删除远端这两个精确的 stale runtime directories。检查确认没有残留 a003 训练进程；本地失败证据、archive、command manifests 和先前 failed attempts 均未删除或覆盖。
- 2026-07-19 00:33 Asia/Shanghai，使用完全相同的 `2ef7aea / 52bdbf...` archive、protocol/dataset manifests 和 10-run queue 完成新的三机 preflight。唯一运行控制变更是将 Controller 单 attempt deadline 显式设置为 `--run-timeout-seconds 172800`（48 h）；它不进入模型数值路径，不改变模型、loss、数据协议、25 global rounds、local epochs、batch size、learning rate 或 server-DA steps。
- 2026-07-19 00:35 Asia/Shanghai 已启动新 Controller PID `34712`，日志为 `results/c2e_runs/controller_restart_20260719_003506.stdout.log` / `.stderr.log`，身份入口仍为 `results/c2e_runs/latest_controller_launch.json`。它自动创建 `c12_to_c5__b2__s42__a004`，该 attempt 必须从 round 1 重跑完整 25 rounds 并通过 validator；成功后才按已冻结的交替顺序继续后续九项。
