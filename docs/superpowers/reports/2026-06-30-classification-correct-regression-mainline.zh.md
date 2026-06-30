# 分类正确下的回归主线补充

本轮补充的目标是把老师提出的“分类正确下的回归性能指标”单独拉成主线，而不是继续让分类错误、route 错误和回归 profile 能力混在同一个 full test 指标里。

## 实验口径

本轮使用 `results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/inputs/target_layer_predictions_oracle_route.csv` 作为 oracle-route 输入。该输入把 `pred_class/route_class` 对齐到 `true_class`，并保持原有 profile replay 所需的 ppm/risk/confidence 字段，用于回答一个更干净的问题：如果分类路线正确，H2.3 / H2.3+ / H8+C4 的回归误差分别如何。

同时保留同一份 deployment `qc_test_records.csv` 做 Accepted / Accepted+Review 子集读数。这里要注意：QC 决策仍来自实际部署分类器，因此 Accepted+Review 更适合作为“同一 QC 子集上的回归表现”补充表，而不是纯 oracle 分类覆盖率。

## per-client blend weight 的含义

`per-client blend weight` 不是样本权重，也不是源域/目标域客户端占比；它是每个目标客户端在 calibration-validation 上选择的 profile 混合系数：

```text
blend_ppm = (1 - w_client) * H2.3_anchor_ppm + w_client * candidate_profile_ppm
```

因此它必须由当前分类基座、当前目标域 calibration split、当前候选 profile 共同决定，不应固定成经验常数。本轮 oracle-route H2.3+ 重新选择出的权重为：

| client | selected weight |
|---|---:|
| C3 | 0.50 |
| C4 | 0.50 |
| C5 | 0.25 |

这和 real-route 里的 `C3=0.5、C4=0.5、C5=0.1` 不完全一致，说明你前面担心的“目标域和源域状态变化会影响权重”是成立的。后续报告里应写成“calibration-validation 选择的 per-client lambda”，而不是固定策略。

## 分类正确 full 指标主表

| profile | ALL RMSE / NRMSE | C3 RMSE / NRMSE | C4 RMSE / NRMSE | C5 RMSE / NRMSE |
|---|---:|---:|---:|---:|
| H2.3 oracle-route | 10.57 / 0.0564 | 9.63 / 0.0540 | 9.28 / 0.0527 | 13.23 / 0.0643 |
| H2.3+ oracle-route weak-blend | 9.86 / 0.0515 | 9.16 / 0.0487 | 8.50 / 0.0482 | 12.18 / 0.0596 |
| H8+C4 oracle-route | 9.10 / 0.0511 | 9.13 / 0.0522 | 8.55 / 0.0502 | 9.57 / 0.0499 |
| Oracle client selector C34 H2.3+ / C5 H8+C4 | 9.11 / 0.0489 | 9.16 / 0.0487 | 8.50 / 0.0482 | 9.57 / 0.0499 |

主结论：

- H2.3+ 在分类正确条件下确实有效：ALL NRMSE 从 `0.0564` 降到 `0.0515`，并且 C3/C4/C5 三个客户端都比 H2.3 低。
- H8+C4 的 ALL RMSE 最低，主要来自 C5 大幅改善：C5 从 H2.3 的 `13.23 / 0.0643` 降到 `9.57 / 0.0499`。
- client selector 的 ALL RMSE 与 H8+C4 几乎持平，但 ALL NRMSE 更低：`0.0489`，说明按客户端 profile 分流后，跨客户端归一化尺度更稳。
- C3 在 full RMSE 上 H8+C4 略低于 H2.3+，但 H2.3+ 的 NRMSE 更低；C4 明确偏 H2.3+；C5 明确偏 H8+C4。

## Accepted+Review 补充表

| profile | ALL RMSE / NRMSE | C3 RMSE / NRMSE | C4 RMSE / NRMSE | C5 RMSE / NRMSE |
|---|---:|---:|---:|---:|
| H2.3 oracle-route | 7.98 / 0.0425 | 6.89 / 0.0390 | 7.09 / 0.0379 | 10.41 / 0.0521 |
| H2.3+ oracle-route weak-blend | 7.00 / 0.0376 | 5.65 / 0.0328 | 6.32 / 0.0343 | 9.53 / 0.0478 |
| H8+C4 oracle-route | 6.48 / 0.0371 | 5.76 / 0.0352 | 6.54 / 0.0359 | 7.63 / 0.0415 |
| Oracle client selector C34 H2.3+ / C5 H8+C4 | 6.38 / 0.0356 | 5.65 / 0.0328 | 6.32 / 0.0343 | 7.63 / 0.0415 |

Accepted+Review coverage 固定来自同一份 QC records：ALL `75.13%`，C3 `75.19%`，C4 `73.90%`，C5 `76.25%`。

这张表支持和 full 指标相同的结构性判断：C3/C4 用 H2.3+ 更好，C5 用 H8+C4 更好，组合后的 selector 在 ALL Accepted+Review 上达到 `6.38 / 0.0356`，是当前最稳的 post-QC 读数。

## 覆盖率 sweep 观察

逐客户端 coverage sweep 显示：

- C3：75%-90% 覆盖下 H2.3+ 同时是 RMSE/NRMSE 最优；95%-100% 时 H8+C4 的 RMSE 略低，但 H2.3+ 的 NRMSE 仍低。
- C4：75%-100% 覆盖下 H2.3+ 都是最优。
- C5：75%-100% 覆盖下 H8+C4 都是最优。
- 全局 75%-90% 覆盖下 client selector 同时是 RMSE/NRMSE 最优；95%-100% 时 H8+C4 的 RMSE 略低，但 selector 的 NRMSE 仍最优。

这说明后续不应该只选一个全局 profile。更稳的路线是把 profile selection 写成目标客户端层面的 calibration policy：C3/C4 走 balanced H2.3+，C5 走 CO-priority H8+C4。

## 后续推进方向

论文/汇报主表建议采用“classification-correct oracle-route full”作为主指标，因为它最直接回答老师的问题：分类正确时回归头和 profile calibration 能做到什么程度。

部署相关补充表保留 Accepted / Accepted+Review，因为它回答的是另一个问题：在当前 QC policy 放行或复核的样本上，回归误差是否可用。这个表不要替代分类正确主表。

下一步实验建议按这个顺序推进：

1. 固化 oracle-route 主表：H2.3、H2.3+、H8+C4、client selector，报告 ALL/C3/C4/C5 的 RMSE/NRMSE。
2. 把 real-route full 与 oracle-route full 拆开报告，用 gap 量化分类/route 错误对回归指标的污染。
3. 对 C3/C4/C5 分别做低 calibration 量 stress test，确认 `C34 -> H2.3+、C5 -> H8+C4` 是否在小样本 calibration 下仍稳定。
4. 将 per-client blend weight 选择规则写成方法：只允许使用 calibration-validation，不看 test；权重随当前分类基座和目标域 split 重新选择。
5. 如果后续还要优化，优先优化 C5 的 CO-priority route/rescue 和 C3 高覆盖尾部，而不是再盲目增强一个全局 profile。

## 产物

- H2.3+ oracle-route replay: `results/h2_3_plus_fusion_profile_20260630/r25_oracle_route_replay_gate/`
- oracle profile QC audit: `results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/profile_qc_coverage_audit/profile_qc_coverage_audit_report.md`
- oracle profile config: `configs/oracle_route_profile_qc_profiles_20260630.json`
