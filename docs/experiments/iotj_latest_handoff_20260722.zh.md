# GAPS IoT-J Latest Handoff — 2026-07-22（2026-07-23 runtime closure）

> `iotj_latest_handoff_20260715.zh.md` 是 **PREVIOUS HANDOFF / HISTORICAL BASELINE**。本文件是后续 Codex 的唯一恢复入口；以仓库、checkpoint 和结果产物为准。

## 0. Five-Minute Recovery

| Item | Value |
|---|---|
| Previous handoff / commit | 20260715 / `a920ecd` |
| Current branch | `codex/iotj-confirmation-observability` |
| Audited code / evidence baseline | code `5ff301c`; evidence `b7598c7` |
| Current worktree | `D:/A Python learning/Federated Learning/TRAE SOLO/.worktrees/iotj-confirmation-observability` |
| Current task | B5/C5 runtime parity closure complete; no active runtime implementation gap |
| Mainline | C1/C2 -> C5, B5 classifier + canonical R4; no C3/C4/H8+C4/P4/R3aK16 runtime |
| Best reusable classifier | B5 round-25 adapted checkpoint, listed below |
| Running experiment | NONE (Pi classifier->R4 sweep completed) |
| Current result root | `results/iotj_b5_c5_deployment_p1_20260722/` |
| Immediate next step | none for runtime parity; do not retrain. A full-chain Pi latency run is a separate future measurement only if explicitly requested |

## 1. Changes Since `a920ecd`

### Code committed after the previous baseline

| Commit | Purpose |
|---|---|
| `bc49990` / `a5dd01f` | canonical runtime-asset capture specification and plan |
| `f289cd8` | export exact H2.3 runtime reference rather than refitting a runtime surrogate |
| `5c151c4` | export canonical all-class R4 runtime policy with source heads |
| `621bde1` | fail-closed canonical B5 replay verifier |
| `43d2c86`–`5419857` | strict bundle/contract, serialized heads, B5 loading and fixed predicted-class R4 route |
| `9acae23` / `3ef2210` | formal 1,360-row map and required phase-label binding |
| `92c0f0b` / `8e36183` | exact H2.3/R4 expert replay and frozen risk/HC policy |
| `bcd53ab` / `5ff301c` | strict six-field validator and non-overwriting formal parity runner |
| `b7598c7` | v4 contract, v2/v3 audit notes, HC95/HC90 runtime rows and parity reports |

Key files:

- `scripts/run_iotj_c5_h23_plus.py`: optional `--runtime-reference-output`; exports exact H2.3 assets.
- `run_source_augmented_target_ridge_eval.py`: optional `--runtime-policy-output`; exports R4 policy/source heads.
- `scripts/run_iotj_c5_regression_suite.py`: wires the above exports into the C5 suite.
- `scripts/verify_iotj_b5_c5_canonical_replay.py`: checks 1360 rows, QC assets and forbidden legacy tokens.
- `scripts/prepare_iotj_b5_c5_bundle_inputs.py` -> `inspect_b5_c5_deployment_inputs.py` -> `build_iotj_b5_c5_deployment_bundle.py`: audited immutable bundle build.
- `scripts/benchmark_iotj_b5_classifier_r4_preliminary.py`: **UNCOMMITTED, USED** Pi/PC preliminary B5 classifier->R4 benchmark; it explicitly excludes HC90 QC.
- `gaps_deploy/c5_h8_runtime.py`: formal versioned B5 -> H2.3/R4 -> risk -> HC runtime; no legacy fallback.
- `scripts/run_iotj_b5_c5_h8_parity.py`: non-overwriting formal parity execution and provenance report.
- `scripts/validate_iotj_b5_c5_runtime_parity.py`: preserves legacy validation and adds strict C5/H8 six-field parity.

## 2. Current code workflow

```text
C1/C2 data + Flower classification
  -> B5 adapted classifier checkpoint
  -> C5 H2.3/R4 regression suite
  -> H2.3 reference + R4 policy + QC assets
  -> audited bundle
  -> classifier -> R4 Pi benchmark (done; preliminary/no QC)
  -> B5 -> H2.3 risk expert + fixed R4/H8 -> risk -> HC95/HC90 runtime/parity (done)
```

## 3. Reusable models, results, and evidence

| Role | Path | Status / use |
|---|---|---|
| B5 classifier checkpoint | `results/iotj_ecs_c2_representative_20260720/raw/c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training/server_round_025_adapted.pth` | CURRENT classifier asset; 25-round representative B5 run |
| B5/C5 bundle | `results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/` | VERIFIED asset contract; use for runtime work |
| canonical replay root | `results/iotj_b5_c5_deployment_p1_20260722/` | VERIFIED `runtime_rows=1360` source-artifact replay |
| formal runtime contract | `results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/` | SOLE FORMAL contract; windows + metadata + phase labels + references hash-bound |
| HC95 parity report | `results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/parity_report.json` | PRIMARY; `equivalent`, 1,360 rows |
| HC90 parity report | `results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/parity_report.json` | STRICTER SECONDARY; `equivalent`, 1,360 rows |
| parity registry | `docs/experiments/iotj_c5_h8_runtime_parity_registry_20260723.csv` | two independent audited records with provenance |
| closure audit | `docs/experiments/iotj_c5_h8_runtime_parity_closure_20260723.zh.md` | contract history, statistics, reuse boundary and handoff |
| HC95 offline reference | `results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc95_records.csv` | FORMAL primary QC reference |
| HC90 offline reference | `results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc90_records.csv` | FORMAL offline QC reference; runtime comparison target |
| HC90 policy | `.../bundle_candidate/assets/qc_risk_policy.json` | frozen: accept `0.7333333333`, reject `0.8458333333` |
| Pi full classifier->R4 evidence | `.../benchmark_pi_fulltest_preliminary_classifier_r4.json` | PRELIMINARY, 1360 windows, no QC |

## 4. Completed results and claim boundaries

- Existing B5 historical seed-42 classification: Accuracy **98.8971%**; screening/historical only, never five-seed mean/std.
- Existing formal C5 B5 R4/H8 offline regression: FULL RMSE **17.4473 ppm**; HC90 accepted RMSE **15.3599 ppm**, yield **88.24%**. Do not describe the regression chain as end-to-end federated.
- B5 representative 25-round real topology evidence: 16.7586 MiB serialized application messages; mean round wall 237.29 s; server DA 161.34 s (~68%); Pi training RSS peak 518.38 MiB; Pi temperature peak 62.25 C.
- Pi 1360-window real preliminary inference, B5 classifier->R4 only: mean 3.698 ms, p50/p95/p99 3.680/3.718/3.835 ms, throughput 264.24 window/s, RSS peak 239.52 MiB, temperature 53.45 C.
- These Pi results do **not** include H2.3/risk/HC90 decision and are not final 1360-row deployment parity.
- Formal runtime parity is complete for both workpoints: zero class, calibrated-risk, QC-decision and `auto_output_ppm` mismatches. Maximum H8 delta is `6.252776074688882e-13 ppm`; maximum calibrated-risk delta is `0`.
- HC95 runtime decisions: accept/review/reject = `1323/33/4` (`97.28%/2.43%/0.29%`). HC90: `1235/107/18` (`90.81%/7.87%/1.32%`). These are deployment-parity counts, not new training metrics.

## 5. Experiment status

| Experiment | Status | Re-run? | Boundary |
|---|---|---|---|
| B5 s42 representative 25-round system run | DONE | NO | single representative system evidence |
| B2 canonicalized recovery | DONE | NO | original controller failure remains recorded |
| formal C5 regression/QC | DONE | NO | reuse outputs/checkpoints |
| Pi classifier->R4 1360 sweep | DONE | NO unless hardware changes | preliminary/no QC |
| B2/B5 five-seed confirmation | TODO | YES after confirmation freeze | formal algorithm statistics |
| HC95 runtime 1360 parity | DONE | NO | primary deployment workpoint; `equivalent` |
| HC90 runtime 1360 parity | DONE | NO | stricter secondary workpoint; `equivalent` |

## 6. Running / monitoring

No local, ECS, Pi, PowerShell, tmux, screen, or Python training task is currently required for monitoring. Old `monitor_b2_s42_a005.ps1` concerns B2 seed 42 attempt a005 and is no longer needed for active work.

## 7. Do not repeat / do not reintroduce

- **DO NOT REINTRODUCE:** C3/C4 targets, H8+C4, P4 leakage version, old R3aK16-only runtime.
- **DO NOT RETRAIN:** completed B5 representative 25-round run, formal regression/QC, or Pi classifier->R4 sweep merely to recreate summaries.
- **DO NOT CLAIM:** classifier->R4 Pi metrics as full deployment/QC/parity.
- **DIRECTLY REUSE:** B5 adapted checkpoint, bundle candidate, v4 runtime contract, HC95/HC90 references and parity reports.
- **DO NOT USE:** v2 runtime contract (wrong data root) or v3 contract (phase labels not hash-bound).

## 8. Minimal reading order

1. this file;
2. `docs/experiments/iotj_c5_h8_runtime_parity_closure_20260723.zh.md`;
3. `results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json`;
4. `results/iotj_b5_c5_deployment_p1_20260722/bundle_candidate/manifest.json`;
5. both `c5_h8_runtime_parity_hc95_v1/parity_report.json` and `c5_h8_runtime_parity_hc90_v1/parity_report.json`;
6. `docs/experiments/iotj_system_experiment_notebook.md`.

## 9. Resume commands

```powershell
cd 'D:/A Python learning/Federated Learning/TRAE SOLO/.worktrees/iotj-confirmation-observability'
git switch codex/iotj-confirmation-observability
python scripts/verify_iotj_b5_c5_canonical_replay.py --root results/iotj_b5_c5_deployment_p1_20260722
python -m pytest tests/test_c5_h8_bundle.py tests/test_c5_h8_runtime.py tests/test_validate_iotj_b5_c5_runtime_parity.py tests/test_run_iotj_b5_c5_h8_parity.py -q
```

The existing parity output directories are intentionally non-overwriting. Do not rerun the parity command into them; choose a new versioned output directory only when assets, code, hardware, or the declared purpose changes.

Pi benchmark environment (already prepared): `gaps@192.168.137.172`, `/home/gaps/b5_c5_bench_venv`, `/home/gaps/b5_c5_preliminary_benchmark`.
