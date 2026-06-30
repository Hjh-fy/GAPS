# Regression-Aware Fusion P2 Balanced Findings

日期：2026-06-30

## 当前进度

P2 balanced fusion 已完成。实验目标是验证：

- H2.3 current r25 继续作为 balanced anchor。
- P1 中表现最稳的 A3-style `rich + reg_feat` 只作为弱增量信号。
- 不改变 C4 formal route-rescue gate 的语义。
- per-client blend 权重只用 calibration-validation 选择，test 只做最终报告。

结果目录：

- `results/h2_3_plus_fusion_profile_20260630/r25_balanced/`

新增脚本：

- `run_h2_3_plus_fusion_profile.py`

## 实验实现

P2 使用如下 profile：

- `A0_baseline_final`：F6 r25 B0/R3aK16/auto_v2 baseline。
- `H2_3_direct_only_r25_refit`：C3 MLP、C4 Ridge、C5 grid-MLP direct-only。
- `H2_3_current_r25_refit_anchor`：direct-only 加同一个 formal C4 route-rescue gate。
- `H2_3_plus_reg_feat_ridge_rescue`：`rich + reg_feat` Ridge，并应用同一个 C4 route-rescue gate。
- `H2_3_plus_blend_r25`：`anchor + lambda * (reg_feat_candidate - anchor)`。
- `H2_3_current_r25_reference`：既有 H2.3 current 产物，用于对齐检查。

每个 client 的 lambda 从 `{0, 0.1, 0.25, 0.5, 0.75, 1}` 中选取。选择约束：

- validation ALL RMSE 必须优于 anchor。
- validation nonCO RMSE 不能比 anchor 高超过 1.0。
- C4 route-rescue gate 不重新学习；anchor 与 reg_feat candidate 都使用同一个 formal C4 gate。

## 质量审计

第一次运行时发现 `fusion_profile_predictions.csv` 有 16200 行，是应有 5400 行的 3 倍。根因是 H2.3 anchor 的 C3/C4/C5 family 输出中保留了非目标 client 的未预测行，组合时重复计入。

修复方式：

- `combine_h2_3_rows()` 只保留与 family 匹配的 client。
- 对重复 `(client, split, sample_index)` key 做防御检查。
- 新增单测 `test_combine_h2_3_rows_keeps_only_matching_client_family_rows()`。

修复后审计：

- test prediction rows：5400。
- C4 rescue hits：10。
- C4 high-CO rescue hits：10。
- C4 nonCO rescue hits：0。
- refit anchor 与 reference H2.3 current 完全对齐。

## Calibration-Validation 选择

选出的 per-client lambda：

- C3：0.50。
- C4：0.50。
- C5：0.10。

validation 读数：

- C3 anchor ALL RMSE 7.16 -> blend 5.92，nonCO 5.18 -> 3.71。
- C4 anchor ALL RMSE 11.33 -> blend 10.71，nonCO 7.82 -> 6.76。
- C5 anchor ALL RMSE 33.12 -> blend 33.09，nonCO 36.99 -> 37.05。

C5 只允许 0.10 的小权重是合理的：更大的权重会开始提高 nonCO 风险，且 ALL RMSE 也不再改善。

## Test 结果

核心对比：

- `H2_3_current_r25_refit_anchor`：ALL RMSE 22.94，ALL NRMSE 0.1789，macro-client NRMSE 0.1640，nonCO_ALL RMSE 23.79。
- `H2_3_plus_reg_feat_ridge_rescue`：ALL RMSE 23.34，ALL NRMSE 0.1747，macro-client NRMSE 0.1589，nonCO_ALL RMSE 23.62。
- `H2_3_plus_blend_r25`：ALL RMSE 22.43，ALL NRMSE 0.1743，macro-client NRMSE 0.1570，nonCO_ALL RMSE 23.12。

P2 blend 相比 H2.3 current：

- ALL RMSE 改善 0.51。
- ALL NRMSE 改善 0.0046。
- macro-client NRMSE 改善 0.0069。
- nonCO_ALL RMSE 改善 0.67。

per-client：

- C3 NRMSE：0.0978 -> 0.0858，明显改善。
- C4 NRMSE：0.0734 -> 0.0663，改善。
- C5 NRMSE：0.3207 -> 0.3190，小幅改善。

CO/high-CO：

- C4 high-CO：36.05 -> 35.81，基本持平略好。
- C5 high-CO：30.80 -> 30.21，小幅改善。
- C3 high-CO：20.02 -> 22.13，略有退化。

## 判断

P2 balanced 是有效阳性。

与 P1 的结论相比，P2 的关键进步是：不是让 `rich + reg_feat` Ridge 直接替代 H2.3，而是把它作为 H2.3 current 的 calibration-selected 弱增量。这样既保留了 C4 route-rescue 的主线收益，又让 backbone `reg_feat` 修正了 C3/C4 和 nonCO 的一部分误差。

当前可以把 `H2_3_plus_blend_r25` 作为 balanced profile 候选，但还不建议立刻宣布替代主线。原因：

- test 读数已经优于 H2.3 current，但需要低校准压力和 selector audit 检查稳定性。
- C5 权重很小，说明 C5 仍然需要 guard，不能把 reg_feat 全量打开。
- C3 high-CO 略退化，后续报告中需要保留这个 trade-off。

## 下一步建议

建议进入 P2b/P3 前置检查：

1. 做 low-calibration stress：用 10%、25%、50%、100% calibration 比较 H2.3 current 与 H2.3+ blend，确认 lambda 是否稳定。
2. 做 selector audit：把 `balanced -> H2_3_plus_blend_r25` 的选择边界写成 calibration-only profile rule。
3. 再进入 CO-priority：在 H8 + formal C4 route rescue 中测试 A6/A7-style signal，但必须继续使用 C5 guard。

如果 low-calibration 下 H2.3+ blend 仍优于或接近 H2.3 current，则第三层 target profile calibration 可以升级为：

- balanced：H2.3+ reg_feat weak-blend。
- CO-priority：H8 + formal C4 rescue，后续再测 backbone/risk-gated rescue。
