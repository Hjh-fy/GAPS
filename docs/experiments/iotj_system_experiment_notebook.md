# IoT-J System Experiment Notebook

Last updated: 2026-07-11

## Purpose

This is the durable engineering and research record for the IoT-J system experiment closure. Update it after every implementation, training, evaluation, and review step. Paper claims must be traceable to an entry here and to an immutable result artifact.

## Frozen Main Protocol

- Primary direction: C1,C2 -> C3,C4,C5.
- Data root: `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`.
- Target split: advisor-approved window-level class/concentration-stratified calibration/test split, 20%/80%.
- Calibration inner split: fit/validation = 75%/25%.
- Data split seed: 42.
- Training seeds: 42,43,44,45,46.
- Flower training: 25 rounds, 5 local epochs, batch size 32, client Adam LR 5e-4.
- Server adaptation: 100 steps/round, LR 5e-4, target CE weight 0.
- Primary classification checkpoint: final round 25.
- Primary regression slices: S_ALL, S_CC, S_AR. S_AR intersect S_CC is diagnostic only.

## Decision Log

| Date | Decision | Reason | Evidence/owner |
|---|---|---|---|
| 2026-07-11 | Use an IoT system-method paper story | Best match to current Flower, target calibration, QC, and cloud-edge runtime work | Advisor direction and code audit |
| 2026-07-11 | Keep the window-level stratified calibration/test split | Advisor-approved protocol; primary split is not being redesigned | User confirmation |
| 2026-07-11 | Treat R3aK16 as a source regression reference | It is not the current best target ppm path | `代码文件介绍.md` and result audit |
| 2026-07-11 | Report regression capability on S_CC separately from QC | Existing consolidation incorrectly mixed S_CC with Accepted+Review | `run_final_metric_consolidation_20260709.py` audit |
| 2026-07-11 | Downgrade C1 to an aggregation contract check | With CE-only clients and no DA, GAPS and FedAvg parameter aggregation should be equivalent; prototype diagnostics alone do not justify a full experiment | Pending deterministic equivalence test |

## Stage Ledger

| Stage | Status | Entry criteria | Exit evidence |
|---|---|---|---|
| 1. Input freeze and metric contract | in progress | Main plan approved | Frozen manifest, corrected S_CC tables, clean tests |
| 2. Classification ablation | pending | Stage 1 complete | Seed-42 screen, five-seed key groups, classification report |
| 3. Regression and expert selection | pending | Frozen classifier outputs | R0-R8 aligned S_CC and real-route tables |
| 4. QC and low-calibration reliability | pending | Aligned P4 streams | Risk-coverage, fixed-coverage, budget statistics |
| 5. Matrix generalization | pending | Cloud artifacts recovered | Forward/reverse classification and regression matrix |
| 6. Runtime and IoT system validation | pending | Final P4 policy frozen | Runtime parity, edge latency, memory, communication |
| 7. Paper evidence freeze | pending | Stages 1-6 complete | Claim-to-evidence map and paper-ready tables/figures |

## Experiment Register

| ID | Date | Hypothesis | Immutable inputs | Command/config | Output | Status | Headline observation |
|---|---|---|---|---|---|---|---|
| AUDIT-001 | 2026-07-11 | Current artifacts can be frozen without retraining | Primary C12-to-C345 dataset, frozen H2.3+/H8+C4 streams, P4 policy, and local matrix root | `python scripts/audit_iotj_experiment_inputs.py --output results/iotj_experiment_freeze_20260711/input_manifest.json` | `results/iotj_experiment_freeze_20260711/input_manifest.json` | complete with external matrix gap | 97 local artifacts hashed; primary split is complete at seed 42; F4, F5, R1-R4 are explicitly missing their cloud-only checkpoint and config. |
| METRIC-001 | 2026-07-11 | Pure S_CC differs from S_AR intersect S_CC | P4 test outputs | Task 2 pending | Final metric consolidation | pending | Manual audit: P4 S_CC N=5301, RMSE=8.3269; S_AR N=4228, RMSE=5.8497 |
| CONTRACT-C01 | 2026-07-11 | CE-only GAPS aggregation equals FedAvg aggregation when DA and client prototype use are disabled | Synthetic identical FitRes and deterministic smoke inputs | Task 3 pending | Contract audit | pending | Pending |

## Review Findings and Risks

| Date | Severity | Finding | Resolution/status |
|---|---|---|---|
| 2026-07-11 | High | Current final metric script labels S_AR intersect S_CC as classification-correct regression | Stage 1 correction planned |
| 2026-07-11 | High | P4 policy is not loaded by the default final runtime bundle | Stage 6 integration planned |
| 2026-07-11 | Medium | Flower evaluation loss is 1-accuracy rather than CE/NLL | Classification evaluation correction planned |
| 2026-07-11 | Medium | Main P4 test behavior is close to a simple predicted-CO expert switch | Add simple switch baseline and constrained selector |
| 2026-07-11 | Medium | Repeated historical inspection may have indirectly adapted design to the test set | Freeze configuration and run final locked evaluation |
| 2026-07-11 | Medium | Local clean-matrix root is absent, so F4, F5, R1, R2, R3, and R4 cannot pass the required checkpoint/config gate | Controller must copy `/root/GAPS/results/source_target_classification_matrix_20260708_clean/` to `results/source_target_classification_matrix_20260708_clean/`; Task 1 audit then reruns locally. |

## Task 1 Input Freeze (2026-07-11)

- Command: `python scripts/audit_iotj_experiment_inputs.py --output results/iotj_experiment_freeze_20260711/input_manifest.json`
- Output: `results/iotj_experiment_freeze_20260711/input_manifest.json` with 109 entries: 97 `present`, 12 `missing`; overall status `incomplete` only because the matrix root is unavailable.
- Findings: the primary C1,C2 -> C3,C4,C5 input root is complete under split seed 42. Target split counts are C3 calibration/test `680/2680`, C4 `320/1360`, and C5 `320/1360`. The manifest records per-split class and concentration counts, every local file's resolved path, byte size, SHA-256, role, and status.
- Failure: the default pytest system temporary directory is not readable in this worker session (`WinError 5` during `tmp_path` setup). `python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest` passes with a workspace-local temporary base.
- Next action: controller-owned cloud transfer only; recover the clean matrix directory, preserve its timestamps and run layout, then rerun the same audit command. No SSH/SCP action was attempted by this worker.

## Current Results Snapshot

- Classification: C3 99.07%, C4 98.60%, C5 95.96%, overall 98.17% (5301/5400).
- P4 S_CC: RMSE 8.3269, NRMSE 0.0443, MAE 4.6884, coverage 98.17%.
- P4 Accepted+Review: RMSE 5.8497, NRMSE 0.0339, coverage 78.30%.
- Classification-wrong P4 rows: N=99, RMSE 152.861.
- QC rejected all 99 classification-wrong rows in the current test, plus 1073 class-correct rows; rejected class-correct RMSE 14.412.

## Next Actions

1. Controller: recover the clean matrix artifacts and rerun the Task 1 audit manifest.
2. Correct the S_CC/S_AR metric contract and regenerate the evidence table.
3. Implement and verify the C0/C1 aggregation equivalence contract before scheduling cloud ablations.
