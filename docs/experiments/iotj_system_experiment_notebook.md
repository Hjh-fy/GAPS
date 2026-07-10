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

## Stage Ledger

| Stage | Status | Entry criteria | Exit evidence |
|---|---|---|---|
| 1. Input freeze and metric contract | in progress | Main plan approved | Frozen manifest, corrected S_CC tables, clean tests |
| 2. Classification ablation | pending | Stage 1 complete | Seed-42 screen, five-seed key groups, classification report |
| 3. Regression and expert selection | pending | Frozen classifier outputs | R0-R7 aligned S_CC and real-route tables |
| 4. QC and low-calibration reliability | pending | Aligned P4 streams | Risk-coverage, fixed-coverage, budget statistics |
| 5. C5 source-count generalization | pending | F1/F2 cloud artifacts recovered | C5-only classification and regression source-count table; F3/F4 appendix optional |
| 6. Runtime and IoT system validation | pending | Final P4 policy frozen | Runtime parity, edge latency, memory, communication |
| 7. Paper evidence freeze | pending | Stages 1-6 complete | Claim-to-evidence map and paper-ready tables/figures |

## Experiment Register

| ID | Date | Hypothesis | Immutable inputs | Command/config | Output | Status | Headline observation |
|---|---|---|---|---|---|---|---|
| AUDIT-001 | 2026-07-11 | Current artifacts can be frozen under the corrected C12-to-C5 protocol | C12-to-C5 dataset and F2 cloud run | `python scripts/audit_iotj_experiment_inputs.py --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --run-dir results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25 --output results/iotj_experiment_freeze_20260711/input_manifest.json` | `results/iotj_experiment_freeze_20260711/input_manifest.json` | complete | 228 artifacts hashed; active clients are exactly C1/C2/C5; C5 calibration/test is 320/1360; canonical F2 is complete with 183 files. |
| METRIC-001 | 2026-07-11 | Pure S_CC differs from S_AR intersect S_CC | P4 test outputs | Task 2 pending | Final metric consolidation | pending | Manual audit: P4 S_CC N=5301, RMSE=8.3269; S_AR N=4228, RMSE=5.8497 |
| CONTRACT-A01 | 2026-07-11 | CE-only GAPS aggregation equals FedAvg aggregation when DA and client prototype use are disabled | Synthetic identical FitRes; any training smoke must use the real cloud-edge topology | Task 3 pending | Contract audit | pending | Pending |

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

1. Rebuild S_CC/S_AR metrics from the F2 classifier and new C5-only regression streams; do not merely filter old F6 predictions.
2. Implement and verify the A0/A1 aggregation equivalence contract before scheduling real cloud-edge ablations.
