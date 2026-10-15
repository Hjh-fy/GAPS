# GAPS IoT-J low-calibration sensitivity 结果（2026-07-25）

## 结论

本实验在固定 B5 seed42、real-topology sufficient-statistics Federated H1 与 C5 105D per-gas Ridge 下，仅改变 C5 calibration budget。最终描述性状态为 `HIGH_CALIBRATION_SENSITIVITY`；该状态不改变最终方法、runtime 或 QC。

| Nominal budget | Actual rows range | S_CC RMSE mean±std | S_ALL RMSE mean±std | CO RMSE mean±std | CO-high RMSE mean±std |
|---:|---:|---:|---:|---:|---:|
| 320 | 320–320 | 11.3416 ± 0.0000 | 25.6490 ± 0.0000 | 22.6650 ± 0.0000 | 35.0212 ± 0.0000 |
| 160 | 160–160 | 23.9156 ± 5.2550 | 33.5489 ± 4.4420 | 32.3168 ± 1.6253 | 45.5564 ± 3.6188 |
| 80 | 80–80 | 30.4799 ± 4.3262 | 37.5764 ± 3.1809 | 38.9094 ± 4.5892 | 52.1114 ± 7.8845 |
| 40 | 40–40 | 36.4992 ± 2.7156 | 41.1598 ± 1.8896 | 43.3038 ± 3.7815 | 62.4105 ± 5.2100 |

相对 320 reference，160/80/40 的 mean S_CC RMSE 相对变化分别为 110.87%、168.74% 和 221.82%。160 的 S_CC subset standard deviation 为 5.2550 ppm，80 为 4.3262 ppm，40 为 2.7156 ppm。

## 分气体结果

| Nominal budget | Gas | RMSE mean±std (ppm) |
|---:|---|---:|
| 320 | Ethanol | 26.0658 ± 0.0000 |
| 320 | CO | 22.6650 ± 0.0000 |
| 320 | Ethylene | 35.5052 ± 0.0000 |
| 320 | Methane | 13.3317 ± 0.0000 |
| 160 | Ethanol | 27.3123 ± 5.1884 |
| 160 | CO | 32.3168 ± 1.6253 |
| 160 | Ethylene | 35.5548 ± 0.6534 |
| 160 | Methane | 36.3059 ± 14.5197 |
| 80 | Ethanol | 29.0143 ± 1.9904 |
| 80 | CO | 38.9094 ± 4.5892 |
| 80 | Ethylene | 33.5089 ± 2.9637 |
| 80 | Methane | 45.9392 ± 8.9608 |
| 40 | Ethanol | 33.4972 ± 5.4079 |
| 40 | CO | 43.3038 ± 3.7815 |
| 40 | Ethylene | 28.2007 ± 1.5451 |
| 40 | Methane | 54.1985 ± 5.1439 |

Methane 与 CO/CO-high 是随预算缩减退化最明显的部分；Ethylene 的均值并非严格单调，说明 sensitivity 具有气体依赖性。所有 replicate 均保留，没有根据结果删除或替换 subset。

## Target Ridge 校准耗时

| Nominal budget | Total p50 (s) | Total p95 (s) | Timing N |
|---:|---:|---:|---:|
| 320 | 0.2743 | 0.3397 | 10 |
| 160 | 0.4979 | 0.5575 | 50 |
| 80 | 0.3481 | 0.4154 | 50 |
| 40 | 0.2213 | 0.2934 | 50 |

160/80/40 使用 group-aware 5-fold alpha selection，而 320 保留冻结的 240/80 holdout，因此耗时不要求随 calibration rows 单调变化。计时覆盖 rich feature、H1 prediction、fold preparation、alpha search、final refit 与 serialization；详细分阶段统计见 `low_calibration_timing_summary.csv`。

## 协议与统计边界

- 40 ⊆ 80 ⊆ 160 ⊆ 320；同一 filename 的 calibration 行不会被拆分。
- 160/80/40 各 5 个确定性 balanced subset replicates；320 为完整 calibration 单次参考。
- 低预算 alpha 仅由 group-aware 5-fold calibration-only selection 决定；320 保持冻结 240/80 语义。
- 冻结 seed/算法的独立重放审计覆盖全部 15 个低预算组合，filename fold leakage 为 0。
- 计时使用 PC 高精度单调时钟，每个 budget/replicate 10 次；重复计时的模型 numeric SHA 必须一致。
- primary metric 为固定 1360-row test 的 S_CC RMSE，未使用 QC accepted RMSE。

## Evidence boundary

1. Low-calibration is a sensitivity analysis under a frozen method.
2. The same historical C5 test split has been used in prior method confirmation.
3. Calibration subsets are selected without access to test labels or errors.
4. Filename grouping is used where possible to reduce within-calibration leakage.
5. Historical calibration/test splitting remains window-level.
6. No model or threshold selection is performed using low-calibration test results.

完整统计见 `low_calibration_summary.csv`、`low_calibration_per_gas_summary.csv`、`low_calibration_alpha_summary.csv` 与 `low_calibration_timing_summary.csv`；论文表格和图分别位于 `paper_tables/` 与 `paper_figures/`。

当前结果具备进入 paper evidence freeze 的条件：协议、subset、模型、calibration lock、test evaluation、表图和报告均有 SHA256 provenance；但论文表述必须保留 `HIGH_CALIBRATION_SENSITIVITY` 与 historical-test evidence boundary。
