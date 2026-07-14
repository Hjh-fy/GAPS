# IoTJ Evidence Archive Index (2026-07-14)

## Scope

This index freezes the lightweight, paper-facing evidence for the seed-42 C5 formal regression/QC study and the completed F1/R1 B2-versus-B5 cross-direction classification study.

- Branch: `codex/system-safety-hardening`
- Oracle/QC execution revision: `7106e5c`
- Strict summary-contract revision: `dc50495`
- Dataset protocol: C1/C2 source, C5 target for formal regression; F1 C1-to-C5 and R1 C5-to-C1 for the completed cross-direction pairs
- Training topology: Alibaba Cloud ECS plus physical Raspberry Pi/PC clients; no local simulation was used for reportable training

## Tracked Lightweight Evidence

- `results/iotj_c5_formal_regression_20260713_v2_summary/manifest.json`
- `results/iotj_c5_formal_regression_20260713_v2_summary/qc_operational_comparison.csv`
- `results/iotj_c5_formal_regression_20260713_v2_summary/formal_regression_report.md`
- `results/iotj_b2_b5_cross_direction_20260713_f1_summary/classification_per_run.csv`
- `results/iotj_b2_b5_cross_direction_20260713_f1_summary/classification_group_summary.csv`
- `results/iotj_b2_b5_cross_direction_20260714_r1_summary/classification_per_run.csv`
- `results/iotj_b2_b5_cross_direction_20260714_r1_summary/paired_direction_comparison.csv`
- `results/iotj_b2_b5_cross_direction_20260714_r1_summary/paired_direction_comparison.json`

The schema-v2 regression manifest records the execution revision and SHA-256 plus byte size for nine key H8/QC files per classifier (27 files total). It distinguishes the original base-suite manifest from the later oracle-route/QC extension.

## Storage Boundary

The row-level H8 and QC files total approximately 108 MiB and remain outside Git by design. They are retained in both locations below:

- Local: `results/iotj_c5_formal_regression_20260713_v2/{A6,B5,B2}`
- ECS: `/root/GAPS/results/iotj_c5_formal_regression_20260713_v2/{A6,B5,B2}`

The tracked manifest can verify these files but cannot recreate them by itself. A GitHub Release, Git LFS object, or durable object-storage bundle is still required before the raw evidence can be called independently archived.

## Frozen Findings

- Formal FULL actual-route H8 RMSE/NRMSE: A6 `28.0144/0.2276`, B5 `17.4473/0.1352`, exploratory B2 `14.6564/0.1059`.
- FULL forced-true-class H8 routing uses all 1360 test windows and is identical across A6/B5/B2: `11.9082/0.0690`.
- The oracle columns are an offline counterfactual under frozen actual-route QC masks, not deployable performance.
- F1 C1-to-C5 favors compact B2 descriptively (`98.8971%` versus `98.3088%`, McNemar `p=0.0963`).
- R1 C5-to-C1 reverses direction and favors B5 on the correct 2680-row C1 test (`98.3582%` versus `97.6493%`, exact McNemar `p=0.00432`; B2-minus-B5 accuracy 95% bootstrap CI `[-1.1940, -0.2612]` pp).
- R2 C4/C5-to-C1 and confirmation seeds 43-46 remain pending; the latest R2 preflight was blocked by Raspberry Pi SSH timeout.

## Verification

- Focused formal regression/QC tests: `40 passed` on both the local PC and ECS.
- Combined regression/QC plus cross-direction evidence tests: `55 passed` locally.
- Original actual-route H8 and `risk_policy.json` hashes remained unchanged after the oracle/QC extension.
- The first 680-row R1 summary used the wrong default data root and is retained only under `results/iotj_b2_b5_cross_direction_20260714_r1_summary_invalid_wrong_data_root`; the canonical tracked R1 summary is count-validated at calibration/test N=680/2680.
