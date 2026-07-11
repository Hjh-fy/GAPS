# IoT-J System Experiment Notebook

Last updated: 2026-07-11

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

## Stage Ledger

| Stage | Status | Entry criteria | Exit evidence |
|---|---|---|---|
| 1. Input freeze and metric contract | complete | Main plan approved | Frozen manifest, corrected S_CC tables, clean tests |
| 2. Classification ablation | in progress | Dataset/input contract frozen | Seed-42 screen, five-seed key groups, classification report |
| 3. Regression and expert selection | in progress | Frozen classifier outputs | C5-only input builder and candidate evaluators are ready; new A-run streams pending |
| 4. QC and low-calibration reliability | pending | Aligned P4 streams | Risk-coverage, fixed-coverage, budget statistics |
| 5. C5 source-count generalization | pending | F1/F2 cloud artifacts recovered | C5-only classification and regression source-count table; F3/F4 appendix optional |
| 6. Runtime and IoT system validation | pending | Final P4 policy frozen | Runtime parity, edge latency, memory, communication |
| 7. Paper evidence freeze | pending | Stages 1-6 complete | Claim-to-evidence map and paper-ready tables/figures |

## Experiment Register

| ID | Date | Hypothesis | Immutable inputs | Command/config | Output | Status | Headline observation |
|---|---|---|---|---|---|---|---|
| AUDIT-001 | 2026-07-11 | Current artifacts can be frozen under the corrected C12-to-C5 protocol | C12-to-C5 dataset and F2 cloud run | `python scripts/audit_iotj_experiment_inputs.py --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --run-dir results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25 --output results/iotj_experiment_freeze_20260711/input_manifest.json` | `results/iotj_experiment_freeze_20260711/input_manifest.json` | complete | 228 artifacts hashed; active clients are exactly C1/C2/C5; C5 calibration/test is 320/1360; canonical F2 is complete with 183 files. |
| METRIC-001 | 2026-07-11 | Pure S_CC must be computed independently of QC | Historical F6/P4 outputs used only as a contract fixture | `python run_final_metric_consolidation_20260709.py --output-dir results/final_metric_consolidation_20260711_historical_contract --docs-report results/final_metric_consolidation_20260711_historical_contract/historical_story_report.zh.md` | `regression_slice_table.csv` | complete | Slice code now emits S_ALL, S_CC, S_CW, S_AR, and S_AR intersect S_CC with explicit parent N/coverage. Historical parity: S_CC N=5301, RMSE=8.3269; S_AR N=4228, RMSE=5.8497. New primary C5 values remain pending. |
| CONTRACT-A01 | 2026-07-11 | CE-only GAPS aggregation equals FedAvg aggregation when selective aggregation, prototype diagnostics, and DA are disabled | Two deterministic synthetic FitRes with 30/70 sample weights | `python -m pytest tests/test_flower_classification_contract.py -q --basetemp .tmp_pytest_cls_equivalence_20260711` | Contract audit | complete | All tensors match FedAvg within absolute tolerance `1e-7`; A1 is not scheduled for full training. |
| CLS-CONFIG-001 | 2026-07-11 | A0-A7 can be generated without role/config drift | Existing C12-to-C5 data root; seeds 42-46 | `python scripts/generate_iotj_classification_ablation_commands.py --include-confirmation-seeds` | `results/iotj_classification_ablation_20260711_commands` | complete | 24 manifests generated: 23 scheduled real training runs and one A1 contract-only row; protocol validation found zero invalid manifests. |
| CLS-CONFIG-002 | 2026-07-11 | Client modules, selective aggregation, server DA, and equal-label-budget adaptation can be separated before real training | Same frozen C12-to-C5 protocol | `python scripts/generate_iotj_classification_ablation_commands.py --include-confirmation-seeds` | `results/iotj_classification_ablation_20260711_v2_commands` | complete | v2 adds A0T/A4S, disables selective aggregation in A2-A4, disables timing-only proto-MMD diagnostics, and materializes five conditional leave-one-group-out rows. 39 manifests: 9 core seed-42 groups, 24 additional confirmation runs, 5 conditional appendix runs, and A1 contract-only. CLS-CONFIG-001 is superseded and must not be launched. |
| CLS-RUN-001 | 2026-07-11 | Seed-42 screening can start on the real ECS/Pi/PC topology | Superseded v1 command manifests | Connectivity preflight | No training outputs | superseded-paused | No partial training run started. Controller PID 28880 was stopped before connectivity returned so it cannot launch the confounded v1 queue. |
| CLS-EVAL-001 | 2026-07-11 | One evaluator can produce true classification metrics and aligned C5 streams | Historical F2 adapted checkpoint used only for numerical parity | `evaluate_checkpoint_stream(..., split='test')` | In-memory 1360-row check plus evaluator tests | complete | Exact parity with prior F2 summary: N=1360, accuracy=0.9882353, macro-F1=0.9882675, NLL=0.1000582, ECE=0.0111189. The same code will evaluate new A-runs. |
| REG-INPUT-001 | 2026-07-11 | The confirmed C5 classifier can feed the regression candidates without C3/C4 leakage | Historical F2 classifier plus C1/C2 source regression reference, smoke only | `python scripts/build_iotj_c5_regression_inputs.py ...` | `results/iotj_c5_regression_inputs_f2_historical_smoke_20260711` | complete | Contract produced exactly C5 calibration/test 320/1360 rows and exported aligned backbone/source-reference features; no new training was performed. |
| REG-H23-001 | 2026-07-11 | A C5-only H2.3+ anchor/blend can be selected on calibration-validation only | REG-INPUT-001 historical smoke stream | `python scripts/run_iotj_c5_h23_plus.py ...` | `results/iotj_c5_h23_plus_f2_historical_smoke_20260711` | complete | Historical test RMSE 21.2182; S_CC N=1344, RMSE 12.3807. Selected blend weight was zero, so the expanded H2.3+ collapsed to its MLP anchor in this replay. |
| REG-H8-001 | 2026-07-11 | H8 can be evaluated for C5 without the obsolete C4 rescue path | REG-INPUT-001 historical smoke stream | `python run_source_augmented_target_ridge_eval.py ... --disable-c4-rescue` | `results/iotj_c5_h8_no_rescue_f2_historical_smoke_20260711` | complete | Historical H8 test RMSE 16.6166; S_CC N=1344, RMSE 11.5028. Manifest records `c4_rescue_enabled=false`. |
| REG-P4-001 | 2026-07-11 | Calibration-selected risk routing improves over both fixed experts | Historical H2.3+/H8 streams, smoke only | `python scripts/select_iotj_c5_p4.py ...` | `results/iotj_c5_p4_f2_historical_smoke_20260711` | complete-negative | Threshold 0.00904 used H8 for 5.22% of test rows. Test RMSE 17.3559: better than H2.3+ (21.2182), worse than fixed H8 (16.6166). P4 is not supported as the main method by this replay. |

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
| 2026-07-11 | Blocking environment | Raspberry Pi SSH is unreachable at the current address `192.168.31.184`; old `172.31.139.224` is retired | Controller PID 28880 was stopped for the v2 preflight review. No automatic retry or training is active. Do not substitute local simulation. |
| 2026-07-11 | Medium | P4 threshold selection improved historical calibration-validation but underperformed fixed H8 on test | Keep P4 as an ablation/negative result; do not promote it without multi-seed new-run evidence and a stronger, predeclared selector. |
| 2026-07-11 | High | Original A2/A3/A4 simultaneously enabled selective aggregation, confounding client-loss attribution | Resolved in v2: A2-A4 use FedAvg-equivalent parameter aggregation; A4S isolates selective aggregation. |
| 2026-07-11 | Medium | The old `proto_only` profile mixed alignment with device-residual statistic extraction even when no server loss consumed the residual | Resolved in v2: A2 uses `align_only`; A4/A4S/A5 use `align_replay`; residual statistics activate only in A6/A7 semantic-DA groups. |
| 2026-07-11 | Medium | Flower client evaluation reported `1-accuracy` through the field named loss | Resolved before training: client evaluation now reports mean cross-entropy/NLL and keeps accuracy as a separate metric. |
| 2026-07-11 | High | Task 8 assumed P4 would be final despite historical calibration/test reversal | Resolved in plan: calibration-validation selects among all predeclared fixed/gated candidates, then the selected method is frozen for test/runtime parity. |
| 2026-07-11 | Medium | Fixed-classifier low-budget replay could be mislabeled as end-to-end calibration efficiency | Resolved in plan: primary budget claim is target regression/QC calibration efficiency; end-to-end claims require rerunning DA at each budget. |
| 2026-07-11 | Medium | Unified regression summaries omitted R2, leaving requested table cells blank | Resolved in the shared metric function; constant-label slices explicitly return blank R2. |
| 2026-07-11 | High | A5-A7 consume target calibration class labels but lacked an equal-label-budget supervised baseline | Resolved in v2 design: A0T uses the same 100 steps/round and C5 calibration labels with source rehearsal CE + target CE, while all proposed alignment terms remain off. |
| 2026-07-11 | High | Replay distillation installed the current server model as a teacher in round 1 despite being defined as previous-round replay | Resolved before training: round 1 only caches the incoming state; the frozen teacher activates from round 2. |
| 2026-07-11 | High | CE-only FedAvg still computed and uploaded unused prototype statistics, invalidating runtime/communication comparison | Resolved before training: CE-only and replay-only profiles skip the extra statistics pass and JSON payload; alignment profiles upload prototypes, while device residuals are limited to semantic-DA groups. |
| 2026-07-11 | Blocking code | Cloud-edge controller synchronized runtime/manifests to Pi but assumed ECS already had the same code and command root | Resolved before training: controller checks ECS idle, uploads committed root/server runtime plus the exact v2 command directory, and reruns code preflight before waiting for Pi. |

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

1. Complete the v2 preflight review and dry-run only; keep the Pi controller paused until the v2 manifests and paper metric contract are signed off.
2. After approval, run only the nine v2 seed-42 core groups: A0/A0T/A2/A3/A4/A4S/A5/A6/A7. Review target metrics and diagnostics before any confirmation or appendix run.
3. If the core screen is coherent, run seeds 43-46 for A0/A0T/A4/A4S/A5/A7. Run `A7-no*` only if A7 merits detailed attribution.
4. Rebuild C5-only Ridge/MLP/H2.3+/H8/gated streams from the confirmed classifier; select the deployable candidate on calibration-validation only and do not filter old F6 predictions into the new main table.

## C5 Regression Closure Smoke (2026-07-11)

- Scope: engineering and historical-artifact parity only. These numbers do not replace the pending real cloud-edge A-run results.
- The new input builder enforces source C1/C2, target C5, and exact C5 counts of 320 calibration plus 1360 test windows. It exports aligned classifier backbone features and the existing R3aK16 C1/C2 source-reference prediction as candidate inputs.
- H2.3+ uses a calibration fit/validation split of 75%/25%, expanded MLP/Ridge grids, and a constrained blend selected without test labels. Historical selection chose blend weight 0; test RMSE was 21.2182 and S_CC RMSE was 12.3807.
- H8 was rerun with `--disable-c4-rescue`, making the target contract genuinely C5-only. Its historical test RMSE was 16.6166 and S_CC RMSE was 11.5028.
- P4 searches risk thresholds only on 80 calibration-validation rows. The selected threshold 0.0090404 routed 71/1360 test windows (5.22%) to H8 and produced test RMSE 17.3559. The fixed H8 expert remained better at 16.6166, while the unattainable per-window oracle was 14.8724.
- Interpretation: the fixed C5 target Ridge with source-reference features is currently the strongest regression candidate. `risk_score` alone does not provide a stable expert selector in this replay. The new classifier seeds must confirm this ordering before the paper method is frozen.
- Verification: `python -m pytest tests/test_iotj_c5_regression_inputs.py tests/test_iotj_c5_h23_plus.py tests/test_iotj_c5_p4.py -q`; direct CLI execution is covered for P4 so pytest path injection cannot mask import failures.
