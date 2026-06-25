# Regression Head Matrix Plan and Status

Date: 2026-06-25

Main rule: performance first. A regression-head or calibration method is promoted only if it improves no-QC full-set target final RMSE / NRMSE, then CO and CO-high RMSE, without obvious nonCO damage. QC is not used for model selection here.

## Current Baselines

| item | status | key result |
| --- | --- | --- |
| R3aK16 + original auto_v2 | stable reference | ALL RMSE 27.34, NRMSE 0.1578, C4 CO high 95.32, C5 CO high 60.00 |
| target direct-head auto_v2 H2.3 | current best mainline | ALL RMSE 18.62, NRMSE 0.1326, C3/C4/C5 CO 16.15/22.02/26.85 |
| H8 stratified direct-head | CO-specialist candidate | ALL RMSE 18.47, C4/C5 CO 19.76/23.69, but nonCO and NRMSE trade-off need caution |

## Matrix A: Existing R3aK16 Structure Ablation

Source: `summarize_r3ak16_structure_ablation.py`

Output:

- `results/r3ak16_structure_ablation_20260625/structure_ablation_summary.csv`
- `results/r3ak16_structure_ablation_20260625/structure_ablation_scope_metrics.csv`
- `results/r3ak16_structure_ablation_20260625/r3ak16_structure_ablation_report.md`

Important note: these artifacts compare structure candidates after existing target-side specialist/calibration output, not pure source-only transfer.

| candidate | meaning | conclusion |
| --- | --- | --- |
| M0 R3aK16 depth4 DCT16 | current structural baseline | best among existing structure candidates: ALL RMSE 23.31 |
| M1 R3aK8 depth4 DCT8 | smaller DCT branch | only tiny parameter saving, ALL/CO worsen |
| M2 MSConv16 | classic local convolution response branch | worsens CO, especially C5 high |
| M3 TCN adapter | response adapter | worsens ALL/CO |
| M4 shared trunk | truly lightweight neural structure | much smaller, but ALL/nonCO worsen too much |
| M5 ratio + DCT | ratio auxiliary branch | helps C4 high, but global/nonCO trade-off too large |

Decision: no existing structural replacement beats M0 under current metric priority.

## Matrix B: Source Lightweight Head + Target Calibration

Existing outputs:

- `results/source_lightweight_regression_head_ablation_20260625_lite/source_lightweight_head_ablation_report.md`
- `results/source_lightweight_target_calibrated_20260625_lite/source_lightweight_target_calibrated_report.md`

Key reading:

- Source lightweight heads fit C1/C2 well, especially per-gas MLP source test ALL RMSE 5.25.
- Direct transfer to C3/C4/C5 collapses, so source fit quality is not enough.
- Adding target affine calibration rescues a lot, but still underperforms the original baseline:

| mode | ALL RMSE | C4 CO | C5 CO | nonCO |
| --- | ---: | ---: | ---: | ---: |
| R3aK16 baseline final | 27.34 | 56.59 | 46.12 | 19.00 |
| source ridge + target affine | 40.17 | 82.34 | 44.23 | 32.63 |
| source per-gas MLP + target affine | 39.41 | 73.29 | 59.68 | 29.96 |
| source shared MLP + target affine | 36.87 | 60.33 | 52.08 | 31.69 |

Decision: lightweight source heads are useful diagnostics, but not ready to replace R3aK16. The bottleneck is cross-client transfer/calibration, not just regression head capacity.

## Matrix C: Missing Lightweight Structural Runs

These are worth running next only if we want a neural lightweight replacement rather than a post-hoc direct head.

| candidate | purpose | structure |
| --- | --- | --- |
| M6 depth2 DCT16 | test shallow residual head while keeping response statistics | `--reg-head-depth 2 --reg-response-branch dct --reg-dct-k 16` |
| M7 depth2 none | compact classic MLP-style head | `--reg-head-depth 2 --reg-response-branch none` |
| M8 depth4 none | isolate value of DCT branch | `--reg-head-depth 4 --reg-response-branch none` |

Required evaluation for each:

1. train/federated regression checkpoint on C1/C2;
2. run the same target-side specialist/calibration flow;
3. evaluate no-QC full-set target test;
4. compare against M0 and H2.3/H8.

Stop rule: if a candidate cannot approach M0 after the same target calibration, do not promote it. If it is close to M0 with much lower parameter count, keep it as deployment-lite candidate but not performance mainline.

## Matrix D: Mainline Performance Work

The most valuable line remains target direct-head auto_v2, because it already beats the R3aK16 baseline by a large margin.

Next formalization:

1. turn target direct-head Ridge/MLP/profile selection into one reproducible mainline script;
2. export selected profiles and parameters with runtime schema;
3. verify deployment/runtime parity on PC;
4. only then revisit Raspberry Pi deployment for the new bundle.

## Current Interpretation

The original regression difficulty is not simply "the R3aK16 head is too complex" or "the head cannot fit source." The evidence says:

- source-domain concentration mapping can be fitted by even lightweight heads;
- target-domain ppm mapping shifts substantially across clients;
- source-only direct transfer is unreliable;
- target-side calibration/direct heads are currently the strongest improvement mechanism;
- lighter neural structures are interesting only if they keep target-calibrated full-set metrics close to the current best.
