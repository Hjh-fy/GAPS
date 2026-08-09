# GAPS IoT-J canonical-v1 conversation handoff

Date: 2026-08-09 (Asia/Shanghai)  
From skill: `experiment-registry`  
Suggested receiving skill: `gaps-research-orchestrator`, followed by `number-consistency-audit` when the v7 manuscript is supplied.

## 1. New-dialog starting point

Use this as the opening instruction in the next conversation:

> Continue from `codex/iotj-final-classification-le1` at commit `83cb87c`. Read `docs/experiments/iotj_canonical_v1_final/CONVERSATION_HANDOFF_20260809.md` and `FINAL_EVIDENCE_INDEX.md` first. Treat `dataset/iotj_canonical_v1/`, all canonical result CSV/JSON/checkpoints, A4, R84_FED_H1, and frozen equal-mean QC as read-only. Do not restart preprocessing/model/QC searches. First validate the branch, dataset hash, checkpoint hashes, and current blockers. Then work only on the explicitly requested writing, canonical figure regeneration, manuscript consistency, or separately authorized blocker resolution.

## 2. Repository and immutable starting state

- Worktree: `D:/A Python learning/Federated Learning/TRAE SOLO/.worktrees/iotj-final-classification-le1`
- Branch: `codex/iotj-final-classification-le1`
- Latest pushed commit at handoff: `83cb87c`
- Remote: `https://github.com/Hjh-fy/GAPS.git`
- Canonical result root: `results/iotj_canonical_v1_final_20260808/`
- Git-tracked evidence bundle: `docs/experiments/iotj_canonical_v1_final/`
- Canonical dataset: `dataset/iotj_canonical_v1/`
- Dataset aggregate SHA256: `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`
- Preprocessing candidate freeze commit: `d6d28d5`
- Classification pre-run freeze commit: `f60f28b`
- Evidence derivation code commit: `02e4ae5` plus executable fix `e7c942f`
- Portable runtime commit: `f3d1577`
- Formal deployment archive SHA256: `52328c9cd9f8c9d9eba2f700a35f20f488070df2919fba6fa94e8a77a5dc1c31`

Existing unrelated dirty logs and `.tmp_pytest*` directories belong to earlier work and were deliberately left untouched. Do not clean, reset, stage, or delete them as part of manuscript work.

## 3. Current research stage and stop rule

The project is no longer in algorithm search. Canonical classification, regression, QC, model-size audit, deployment packaging, and Pi 5 benchmarking are complete. The next stage is writing and evidence presentation, subject to the blockers in Section 10.

Do not:

- change preprocessing, A4, R84, or the QC formula;
- tune against target test;
- remove C5 methane 225 ppm repeat1;
- promote historical/legacy datasets or figures to canonical evidence;
- infer a 50% latency or FL communication reduction from the 50% temporal-input reduction;
- launch additional experiments unless the user explicitly authorizes one of the registered blockers.

## 4. Canonical data and preprocessing

Canonical preprocessing ID: `HZ5_MEAN_W10S`.

- Stable real-time sort.
- Duplicate timestamps merged by mean.
- Baseline is the raw-observation mean conductance G0 over 20-50 s.
- Sampling/physical binning: 5 Hz, 0.2 s mean bins.
- Gap policy: interpolate at most one missing bin; no long-gap continuous interpolation.
- Physical crop: 60-170 s.
- Window: 10 s duration, 5 s stride.
- Model input: `50 x 8`, FP32 raw tensor size 1,600 bytes/window.
- Source devices: C1, C2.
- Independent target devices: C3, C4, C5.
- Frozen seed: 42.

Frozen role counts:

| Client | Role | Train | Calibration | Test |
|---|---|---:|---:|---:|
| C1 | source | 2360 | 320 | 680 |
| C2 | source | 2360 | 320 | 680 |
| C3 | target | 0 | 678 | 2677 |
| C4 | target | 0 | 320 | 1360 |
| C5 | target | 0 | 320 | 1360 |

C3 has more physical windows, so its approximately 20% calibration split contains 678 windows. C4/C5 each contain 320 calibration windows. The role split is keyed by frozen physical identity, not by forcing the same absolute count across devices.

The earlier `CANONICAL_SPLIT_FREEZE.md` statement that calibration/test overlap is zero refers to exact physical-window identity. A later raw-time audit found overlapping underlying time intervals because 10 s windows use a 5 s stride; see Section 10.

## 5. Canonical classification protocol

Formal IDs: `CANONICAL-V1-A4-C3`, `CANONICAL-V1-A4-C4`, `CANONICAL-V1-A4-C5`.

- Architecture: `FedGasBaseModel`, TCN/attention path, four gas classes, eight sensors.
- Router/method: frozen A4 GAPS, `ce_stats` profile.
- Source FL clients: C1 and C2.
- Each target is commissioned independently.
- Fresh seed-42 random initialization for every target run; no historical checkpoint reuse.
- Federated rounds: 25.
- Local epochs: **1**.
- Batch size: 32.
- Optimizer: Adam, learning rate `5e-4`.
- Aggregation: ordinary FedAvg throughout; selective aggregation disabled.
- Client semantic/replay paths: disabled in the frozen A4 protocol.
- Server adaptation: 100 fixed steps, learning rate `5e-4`, beginning without DA warm-up.
- Adaptation consumes target calibration `x`, class, and phase. Target concentration is not used.
- Target CE weight is **0**. Class/phase labels support conditional alignment and prototype/stage objectives; this is not Simple Target-CE.
- Enabled server mechanisms include CORAL, global/class MMD, adversarial alignment, prototype/proto-MMD, consistency/residual, and same-class-phase stage alignment according to the locked command.
- Target test is unavailable to training, tuning, stopping, and checkpoint selection.
- Checkpoint is the fixed round-25 adapted endpoint.

Checkpoint SHA256:

| Target | SHA256 |
|---|---|
| C3 | `e2364290ffc7fd9748fe86edb3745dca0eac692165f6c8aba1825728ddcd4414` |
| C4 | `422a49f28331e5486d215a8d34bc9a972dc8fc1992f8b5bf27428329143599c3` |
| C5 | `3965ec8618a2d496804bbc141f49e00b451fce05e9edbefde721f0dd4f912b93` |

Formal classification metrics:

| Target | N | Accuracy | Macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| C3 | 2677 | 0.998506 | 0.998507 | 0.015451 | 0.001343 |
| C4 | 1360 | 0.997794 | 0.997794 | 0.018446 | 0.001953 |
| C5 | 1360 | 0.994118 | 0.994126 | 0.065940 | 0.006261 |
| ALL | 5397 | 0.997221 | 0.997221 | 0.028929 | 0.002566 |

## 6. Canonical regression protocol and results

Formal regression: `R84_FED_H1` using each target's corresponding A4 checkpoint.

- Per-gas Ridge routing uses the predicted A4 class for formal S_ALL evaluation.
- 83D base: target sensor/window statistical features.
- 84th input: frozen federated source-side H1 prediction.
- Calibration only is used to fit/select Ridge.
- Frozen alpha grid: `0, 0.01, 0.1, 1, 10, 100, 1000`.
- Calibration-internal validation fraction for the 83D/84D closure: 0.25.
- No test-based alpha/model selection and no R84 tuning after results.
- S_ALL is the deployable predicted-route result; S_CC is route-correct subset diagnostics; oracle-route is diagnostic only.

| Target | S_ALL RMSE (ppm) | S_ALL NRMSE | S_CC RMSE (ppm) | S_CC NRMSE |
|---|---:|---:|---:|---:|
| C3 | 9.3327 | 0.05567 | 8.8479 | 0.05333 |
| C4 | 13.8080 | 0.06831 | 10.2452 | 0.05291 |
| C5 | 18.4765 | 0.09465 | 14.3340 | 0.07041 |

Canonical 83D versus 84D overall S_ALL:

| Scope | 83D RMSE | 84D RMSE | Absolute gain | Relative RMSE reduction |
|---|---:|---:|---:|---:|
| C3 | 10.5084 | 9.3327 | 1.1757 | 11.19% |
| C4 | 13.6137 | 13.8080 | -0.1942 | -1.43% |
| C5 | 18.9330 | 18.4765 | 0.4565 | 2.41% |
| ALL | 13.8494 | 13.3144 | 0.5350 | 3.86% |

The correct claim is mixed: H1 improves aggregate, C3, and C5 RMSE, but slightly worsens C4 RMSE. Do not claim universal per-device improvement.

## 7. Frozen QC and quality robustness

QC formula: equal mean of three calibration-p95-normalized risk components, clipped to `[0,1]`:

1. classification uncertainty;
2. R83/R84 regression disagreement;
3. H1/H2/H3 source-prior disagreement.

Thresholds are target-specific empirical quantiles computed from canonical calibration only.

- HC90: accept through q90, review q90-q95, reject above q95.
- HC95: accept through q95, review q95-q97.5, reject above q97.5.
- Random reference: same accepted N, 1,000 repetitions, seed 20260804.
- No target-test threshold search.

Aggregate accepted-only results:

| Workpoint | Achieved coverage | RMSE | NRMSE | Random mean RMSE | QC gain vs random |
|---|---:|---:|---:|---:|---:|
| HC90 | 0.858996 | 12.1793 | 0.06424 | 13.3040 | 1.1248 ppm |
| HC95 | 0.907727 | 12.2650 | 0.06484 | 13.3027 | 1.0377 ppm |

Quality strata are read-only diagnostics; no sample was deleted. Most windows are Q0. Q1/Q2 are sparse and Q3 is empty, so the data do not establish severe-stratum robustness. C5 methane 225 ppm repeat1 remains a documented anomaly:

- repeat1: N=17, classification accuracy 0.88235, S_ALL RMSE 70.9693 ppm;
- repeat2: N=17, classification accuracy 1.0, S_ALL RMSE 20.8510 ppm.

## 8. Deployment and Raspberry Pi 5

Formal runtime pipeline:

`canonical preprocessing contract -> target-specific A4 -> R83/R84_FED_H1 -> frozen equal-mean QC`

The deployment package is self-contained. Earlier failed packages were preserved with `failed_nonportable`, `failed_hidden_dependency`, or `failed_package_init` names and must not be deployed.

- Formal package directory: `results/iotj_canonical_v1_final_20260808/deployment_package/`
- Formal archive: `deployment_package_f3d1577.tar.gz`
- Runtime status: `FINAL_DEPLOYED_RUNTIME`
- Classifier state tensor count: 80. This is not parameter count.
- Total/trainable scalar parameters: 22,765.
- FP32 parameter bytes: 91,060.
- Complete package bytes: 1,455,987.
- Canonical input tensor bytes: 1,600/window.

Pi 5 protocol: C5 package, batch 1, four threads, 200 warm-up windows, 10,000 measured windows, governor `ondemand`, Python 3.13.5, PyTorch 2.12.1+cpu.

| Metric | Value |
|---|---:|
| P50 total latency | 3.1487 ms |
| P95 total latency | 3.1925 ms |
| P99 total latency | 4.9241 ms |
| Throughput | 295.929 windows/s |
| Peak RSS | 258.922 MiB |

## 9. Historical evidence: retain but do not mix with canonical-v1

### 9.1 Old-preprocessing fixed classification matrix

Root: `results/iotj_final_classification_le1_20260804/`. This matrix is useful historical/algorithm evidence but is not the canonical-v1 final numerical source.

Seed42 fixed endpoints include FedAvg, FedProx, canonical SGD SCAFFOLD, nine x-only E2 references (CORAL/MMD/DANN x C3/C4/C5), GAPS C3/C4/C5, and C5 A1-A5/A4 ablations. Main Macro-F1 triplets C3/C4/C5:

| Method | C3 | C4 | C5 |
|---|---:|---:|---:|
| FedAvg, Adam | 0.9499 | 0.8950 | 0.2352 |
| FedProx, Adam | 0.9749 | 0.9314 | 0.3202 |
| SCAFFOLD, canonical SGD | 0.6338 | 0.6244 | 0.4451 |
| CORAL, x-only | 0.9304 | 0.8514 | 0.2779 |
| MMD, x-only | 0.9779 | 0.9591 | 0.5512 |
| DANN, x-only | 0.9511 | 0.8863 | 0.4166 |
| GAPS | 0.9897 | 0.9906 | 0.9845 |

SCAFFOLD uses SGD `5e-4` with canonical control variates; FedAvg/FedProx/GAPS use Adam `5e-4`. This is an algorithm-level comparison, not an optimizer-controlled single-factor ablation.

### 9.2 Zero-label and adaptation-timing studies

These are historical C5 mechanism studies from the P0A source checkpoint, not canonical-v1 final results.

- Source-only Macro-F1: 0.2352.
- U1 unconditional x-only alignment, 100 steps: 0.5921.
- U2 threshold-0.90 pseudo-label CE, 100 steps: 0.2416; post-hoc pseudo-label precision 35.60% despite 99.60% mean selected confidence.
- Simple Target-CE reference: 0.9765.
- Post-hoc UDA2500: 0.4901; interleaved 25x100 endpoint: 0.4927.

The frozen interpretation is that short x-only global alignment helps direct transfer, naive self-training is unsafe, and extending/interleaving UDA does not guarantee a better fixed endpoint. Do not automatically optimize these methods.

### 9.3 Historical 83D/91D/104D metadata study

The historical metadata asset made the additional metadata columns constant, so 83D, 91D, and 104D were numerically equivalent. This supports simplifying the deployed feature contract, but not a claim that real online metadata has no predictive value. It is not canonical-v1 quantitative evidence.

## 10. Unresolved fields and submission blockers

1. **Canonical equal-label A0T: BLOCKED_NOT_RUN.** A preregistered config exists at `results/.../a0t_equal_label/A0T_PRE_RUN_FREEZE.json`, freeze commit `0db9dac`. It requires C3/C4/C5, 25 rounds, LE1, Adam `5e-4`, 100 target-CE steps/round, the same calibration labels as A4, and all non-CE adaptation losses disabled. Execution was not authorized in the previous session. Do not report A0T metrics until a new explicit authorization and completed audit exist.
2. **Raw-time overlap robustness: BLOCKER.** Exact identity overlap is zero, but shared raw-time union ratios are C3 0.2937, C4 0.2852, C5 0.2898. The proposal exists at `evidence_closure/overlap/STRICT_NON_OVERLAP_ROBUSTNESS_PROTOCOL_PROPOSAL.md`. Do not modify the canonical main results; if explicitly authorized, run a separate grouped/non-overlap supplementary robustness study.
3. **Final canonical figures: NEEDS_REGEN.** Fig.1 is legacy-only; Fig.2/3/5/6/7/8 need regeneration or canonical verification; Fig.4's historical ablation must not be relabeled canonical. See `FIGURE_TABLE_PANEL_TRACKER.md`.
4. **Manuscript v7 consistency: BLOCKED.** No v7 `main.tex` was found. Available v5/v6 files are historical. Once supplied, scan Methods/Results for `10 Hz`, `100x8`, `SEQ_LEN=100`, `local_epochs=5`, old C5/QC/deployment values, and continuous interpolation.
5. **Calibration-budget evidence: MISSING_CANONICAL_EVIDENCE.** See `CALIBRATION_BUDGET_GAP.md`. This is claim-dependent and did not block QC/deployment closure.

Largest current evidence gap for a fairness claim: canonical equal-label A0T. Largest split-validity concern: raw-time overlap robustness.

## 11. Code map

Canonical data/preprocessing:

- `tools/run_preprocessing_mechanism_audit.py`
- `tools/run_canonical_preprocessing_selection.py`
- `tools/build_iotj_canonical_v1.py`
- `tools/preflight_iotj_canonical_v1.py`
- `tools/verify_iotj_canonical_v1_hashes.py`

Classification:

- `scripts/run_iotj_canonical_v1_classification.py`
- `scripts/evaluate_iotj_canonical_v1_classification.py`
- `gaps_flower/server_app.py`
- `gaps_flower/client_app.py`
- `model.py`

Regression and evidence:

- `scripts/run_iotj_canonical_v1_r84.py`
- `run_regression_head_ablation.py`
- `scripts/finalize_iotj_canonical_v1_evidence.py`
- `scripts/build_iotj_canonical_v1_submission_bundle.py`

Deployment:

- `gaps_deploy/canonical_v1_runtime.py`
- `gaps_deploy/canonical_serialized.py`
- `scripts/build_iotj_canonical_v1_deployment.py`
- `scripts/benchmark_iotj_canonical_v1_pi5.py`

Registered but unexecuted A0T:

- `scripts/run_iotj_canonical_v1_a0t.py`

Historical baseline/mechanism code:

- `scripts/run_iotj_final_classification_le1.py`
- `scripts/evaluate_iotj_final_classification_le1.py`
- `scripts/finalize_iotj_final_classification_le1.py`
- `scripts/audit_iotj_final_classification_le1.py`
- `scripts/run_iotj_p0_zero_label_commissioning.py`
- `scripts/run_iotj_p0i_adaptation_timing.py`
- `scripts/evaluate_iotj_feature_metadata_ablation.py`

Tests:

- `tests/test_iotj_canonical_v1_dataset.py`
- `tests/test_iotj_canonical_v1_preflight.py`
- `tests/test_iotj_canonical_v1_classification_protocol.py`
- `tests/test_iotj_canonical_v1_classification_evaluation.py`
- `tests/test_iotj_canonical_v1_r84.py`
- `tests/test_iotj_canonical_v1_evidence_closure.py`
- `tests/test_iotj_canonical_v1_deployment.py`
- `tests/test_iotj_canonical_v1_a0t.py`
- `tests/test_iotj_canonical_v1_submission_bundle.py`

## 12. Primary evidence files

Read these first:

1. `docs/experiments/iotj_canonical_v1_final/FINAL_EVIDENCE_INDEX.md`
2. `docs/experiments/iotj_canonical_v1_final/12_reproducibility_manifest.json`
3. `docs/experiments/iotj_canonical_v1_final/FINAL_SUBMISSION_READINESS.md`
4. `docs/experiments/iotj_canonical_v1_final/FINAL_CLAIM_EVIDENCE_MATRIX.md`
5. `docs/experiments/iotj_canonical_v1_final/FIGURE_TABLE_PANEL_TRACKER.md`
6. `results/iotj_canonical_v1_final_20260808/FINAL_EXPERIMENT_STATE.json`

Large checkpoints/predictions remain local and are indexed by path/hash; they were intentionally not force-added to Git.

## 13. Completed checks

- Canonical dataset three-machine consistency and 71-file hash verification: PASS.
- Classification protocol audit and one-time sealed test evaluation: PASS.
- Regression target-test selection audit: PASS.
- QC calibration-only threshold audit: PASS.
- Deployment package preflight and package SHA verification: PASS.
- Pi 5 formal benchmark: COMPLETE.
- Latest closure test set: 17/17 PASS.
- `python -m compileall`: PASS.
- Final evidence/package/source SHA256 verification: PASS for 71 checked files.

## 14. Requested next action

The next conversation should first ask what asset is available:

- If manuscript v7 is supplied: run a read-only number/claim consistency audit, then propose text edits; do not silently edit.
- If the user wants figures: regenerate only from canonical-v1 evidence and update the panel tracker.
- If the user explicitly authorizes A0T: validate the preregistered freeze and execute exactly that single baseline, with no search.
- If the user explicitly authorizes split robustness: execute the separate strict non-overlap proposal without replacing canonical main results.

Files that must remain read-only: canonical dataset, all formal checkpoints, result CSV/JSON, QC thresholds, deployment package, protocol freezes, and historical result roots.
