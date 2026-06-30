# Profile QC Coverage Client Audit

日期：2026-06-30

## per-client blend 权重是什么意思

P2 中的 per-client blend 权重不是样本权重，也不是源域/目标域整体权重。它是每个目标客户端自己的预测融合系数：

`final_ppm = H2.3_anchor + lambda * (reg_feat_candidate - H2.3_anchor)`

含义：

- `lambda=0`：完全使用 H2.3 current。
- `lambda=1`：完全使用 `rich + reg_feat` candidate。
- `0 < lambda < 1`：只让 backbone `reg_feat` 对 H2.3 做弱修正。

本轮 calibration-validation 选出：

- C3：0.50。
- C4：0.50。
- C5：0.10。

这说明 C3/C4 可以较强地接受 `reg_feat` 修正；C5 只能开很小的权重，因为 C5 nonCO / tail risk 更敏感。

这个权重必须随 target client、source/base classifier、calibration split 重新选择，不能跨实验默认复用。

## 本轮新增实验

为了回应“每个目标域客户端单独看”的问题，本轮新增统一 post-profile QC 审计：

- `run_profile_qc_coverage_audit.py`
- 输出目录：`results/f6_fixed_da_strong_r25_profile_replay_20260630/profile_qc_coverage_audit/`

审计口径：

- QC 决策和风险分数固定使用正式 deployment QC：`qc_test_records.csv`。
- 只替换 profile ppm 输出，再重新计算 RMSE / NRMSE。
- Accepted：`qc_decision=accept`。
- Accepted+Review：`qc_decision in {accept, review}`。
- coverage sweep：每个目标客户端内部按 `qc_risk_value` 从低到高排序，取最低风险 75%、80%、85%、90%、95%、100% window。

纳入比较的 profile：

- H2.3 target direct-head。
- H2.3+ reg-feat weak-blend。
- H8 + formal C4 route rescue。
- Client selector：C3/C4 使用 H2.3+，C5 使用 H8+C4。

## 正式 QC 下的 Accepted+Review

ALL：

- H2.3：7.98 / 0.0425。
- H2.3+：7.16 / 0.0383。
- H8+C4：6.48 / 0.0371。
- Client selector：6.38 / 0.0356。

Per-client：

- C3 Accepted+Review：H2.3 6.89 / 0.0390，H2.3+ 5.65 / 0.0328，H8+C4 5.76 / 0.0352。
- C4 Accepted+Review：H2.3 7.09 / 0.0379，H2.3+ 6.32 / 0.0343，H8+C4 6.54 / 0.0359。
- C5 Accepted+Review：H2.3 10.41 / 0.0521，H2.3+ 9.99 / 0.0500，H8+C4 7.63 / 0.0415。

因此：

- C3/C4：H2.3+ 比 H8+C4 更适合。
- C5：H8+C4 明显更适合。
- 全局 Accepted+Review：client selector 最好。

这修正了前一轮“H8+C4 在每个客户端都最好”的结论。那个结论在没有加入 H2.3+ profile 时成立；加入 H2.3+ 后，C3/C4 的最佳 profile 发生了变化。

## 每客户端 coverage sweep

C3：

- 75%：H2.3+ 5.655 / 0.0328，H8+C4 5.760 / 0.0352。
- 80%：H2.3+ 5.823 / 0.0338，H8+C4 5.880 / 0.0359。
- 85%：H2.3+ 6.012 / 0.0352，H8+C4 6.275 / 0.0384。
- 90%：H2.3+ 6.584 / 0.0375，H8+C4 6.930 / 0.0408。
- 95%：H8+C4 RMSE 略好，H2.3+ NRMSE 更好。
- 100%：H2.3+ 12.223 / 0.0858，H8+C4 12.285 / 0.0897。

C4：

- 75% 到 100%：H2.3+ 在 RMSE 和 NRMSE 上全部优于 H8+C4。
- 90%：H2.3+ 11.777 / 0.0584，H8+C4 12.325 / 0.0659。
- 100%：H2.3+ 12.725 / 0.0663，H8+C4 13.046 / 0.0718。

C5：

- 75% 到 90%：H8+C4 在 RMSE 和 NRMSE 上全部明显优于 H2.3+。
- 95%：H8+C4 RMSE 更好，H2.3+ NRMSE 极小幅更好。
- 100%：H8+C4 RMSE 更好，H2.3+ NRMSE 更好。

这说明 C5 的 RMSE/NRMSE tradeoff 仍然存在，不能简单用一个全局 profile 处理。

## Client selector 结果

构造规则：

- C3：H2.3+ reg-feat weak-blend。
- C4：H2.3+ reg-feat weak-blend。
- C5：H8 + formal C4 route rescue。

正式 QC：

- Accepted RMSE / NRMSE：5.11 / 0.0293。
- Accepted+Review RMSE / NRMSE：6.38 / 0.0356。
- Full RMSE / NRMSE：22.38 / 0.1771。

全局 coverage sweep：

- 75%：6.206 / 0.0350。
- 80%：6.483 / 0.0366。
- 85%：6.745 / 0.0385。
- 90%：8.749 / 0.0461。
- 95%：9.435 / 0.0496。
- 100%：22.376 / 0.1771。

75%-95% coverage 区间内，client selector 在 RMSE 和 NRMSE 上都是全局最优。100% full-set 下，client selector 的 RMSE 最好，H2.3+ 的 NRMSE 最好。

## Oracle-route 对照

已有 oracle-route replay 显示：如果强制 `route_class=true_class`，Full-set 明显改善：

- H2.3 oracle-route ALL：10.57 / 0.0564。
- H8+C4 oracle-route ALL：9.10 / 0.0511。

但 Accepted 和 Accepted+Review 指标基本不变，因为这组 oracle-route 实验沿用了同一份正式 QC 风险分数和 QC 决策。

这说明：

- Full-set 的大头问题仍然是 route/classification tail error。
- Post-profile QC 已经把大量 route error 挡在 reject 区域。
- Accepted+Review 区间里，profile/head 选择开始成为主要差异来源。

## 当前判断

第三层 target profile calibration 不应只做方向级选择，而应至少升级成 client-level profile：

- C3：balanced 使用 H2.3+ weak-blend。
- C4：balanced 使用 H2.3+ weak-blend；CO-priority 仍可保留 H8+C4 rescue 作为对照。
- C5：Accepted+Review / deployable coverage 优先使用 H8+C4；Full-set NRMSE 指标需要保留 H2.3+ tradeoff 注释。

这也解释了 per-client blend 权重的重要性：它不是为了构造一个全域统一线性融合，而是为 profile selector 提供每个目标客户端的校准证据。

## 下一步建议

1. 将 client selector 变成正式 selector artifact：`C3/C4 -> H2.3+`，`C5 -> H8+C4`。
2. 对 client selector 做 low-calibration stress，检查 10%、25%、50%、100% calibration 下 C3/C4 的 `lambda=0.5` 和 C5 的 H8+C4 选择是否稳定。
3. 对 C5 单独做 RMSE/NRMSE tradeoff audit，明确论文主指标优先 RMSE、NRMSE，还是 Accepted+Review deployability。
4. 后续如果源域/目标域换了，必须重新跑：F6 base replay -> P2 blend selection -> profile QC coverage audit，不能直接复用当前权重。
