# Regression-Aware Fusion P1 Findings

日期：2026-06-30

## 当前进度

当前已经完成计划里的 P0/P1 主线：

- P0：使用 F6 r25 final adapted checkpoint 导出 C3/C4/C5 calibration/test backbone 特征。
- P0 diagnostic：额外导出 F6 r19 adapted checkpoint 特征，仅作为诊断线，不进入官方读数。
- P1：在 H2.3 的 target direct-head/Ridge calibration 框架中做 backbone feature ablation。

对应产物：

- `results/f6_r25_backbone_feature_export_20260630/`
- `results/f6_r19_backbone_feature_export_20260630_diagnostic/`
- `results/h2_3_backbone_feature_ablation_20260630/r25/feature_ablation_report.md`

## 实验设置

官方读数只看 F6 r25 final adapted checkpoint。

P1 消融使用同一套 target predictions 和 calibration/test 切分，不使用 test 指标做特征选择。对比组：

- A0：rich-only baseline。
- A1：rich + classification confidence/probability。
- A2：rich + classification embedding。
- A3：rich + regression-head embedding。
- A4：rich + B0 prior。
- A5：rich + source prior。
- A6：rich + all backbone signals。
- A7：rich + all priors/signals。

## 核心结果

结论分类是 `backbone-positive`，但不是“可以直接替换主线”的强阳性。

最值得看的改善来自 A3：

- A0 rich-only：ALL RMSE 25.61，ALL NRMSE 0.1863，macro-client NRMSE 0.1795，nonCO_ALL RMSE 24.49。
- A3 rich + reg_feat：ALL RMSE 25.48，ALL NRMSE 0.1805，macro-client NRMSE 0.1742，nonCO_ALL RMSE 23.62。

也就是说，regression-head embedding 对整体归一化误差和 nonCO 稳定性有真实贡献：

- ALL NRMSE 改善约 0.0058。
- macro-client NRMSE 改善约 0.0053。
- nonCO_ALL RMSE 改善约 0.87。

但是，当前 direct-head feature ablation 的绝对水平仍弱于上一轮 H2.3 current r25 主线读数：

- H2.3 current r25：ALL RMSE 22.94，ALL NRMSE 0.1789。
- P1 最优 A3：ALL RMSE 25.48，ALL NRMSE 0.1805。

因此这轮实验的含义是：backbone embedding 可以作为 profile calibration 的辅助信号，但不应该把 H2.3 target direct-head 直接替换成 embedding-only 或 rich+embedding 简化头。

## CO 与高浓度段反馈

A6/A7 对 C4 high-CO 有改善：

- A0 C4-CO_high_200_250：82.64。
- A6 C4-CO_high_200_250：78.21。
- A7 C4-CO_high_200_250：77.70。

但它们同时明显伤害 C5 high-CO：

- A0 C5-CO_high_200_250：20.85。
- A6 C5-CO_high_200_250：33.11。
- A7 C5-CO_high_200_250：34.62。

这说明 all-backbone/all-prior 信号对 C4 route rescue 有价值，但不能无条件全局打开。它更像是 CO-priority 模式下的局部 rescue 信号，而不是 balanced 模式下的默认特征。

C5 nonCO wrong-route audit：

- C5 nonCO N：1020。
- C5 nonCO predicted as CO N：33。
- rate：3.24%。

这个比例不高，但结合 C5 high-CO 被伤害的结果，下一步必须继续保留 C5 guard，不能只按 C4 high-CO 改善来选择配置。

## 当前判断

第三层 target profile calibration 的方向仍然成立：

- balanced 模式：H2.3 target direct-head 继续作为主干。
- CO-priority 模式：H8 + formal C4 route rescue 继续作为候选策略。
- P1 新增信息：F6 r25 的 reg_feat / all-backbone signal 可以进入 profile selector，但应该以 gated/conditional 的方式进入。

更具体地说：

- A3 是最适合进入 balanced profile 的候选辅助信号，因为它改善 ALL NRMSE、macro-client NRMSE 和 nonCO_ALL。
- A6/A7 是最适合进入 CO-priority route rescue 的候选信号，因为它们改善 C4 high-CO，但必须加 C5 guard。
- A4/A5 prior-only 的收益很弱，不能单独作为下一层 profile 的核心依据。

## 下一步计划

建议按下面顺序推进：

1. 做 H2.3+profile fusion 小实验：以 H2.3 current r25 为锚点，只允许 A3-style reg_feat 作为弱增量，不改 route/rescue 逻辑。
2. 做 CO-priority 局部 rescue 小实验：只在 C4 high-CO/formal C4 route rescue 分支中引入 A6/A7-style signal，并显式检查 C5 high-CO 和 C5 nonCO。
3. 做 selector audit：比较 balanced 与 CO-priority 两个 profile 的选择边界，确认是否能用 calibration-only 规则决定何时打开 rescue。
4. 如果第 1/2 步不能同时守住 H2.3 current r25 的 ALL NRMSE 和 C5 guard，就暂停更重的 regression-aware encoder 训练，把 backbone signal 保留为诊断特征。

阶段性推进标准：

- balanced profile 必须不劣于 H2.3 current r25 的 ALL NRMSE 0.1789，且不能显著恶化 nonCO_ALL。
- CO-priority profile 可以牺牲少量 ALL RMSE，但必须明确换来 C4 high-CO 改善，并且 C5 high-CO/C5 nonCO 不出现新的崩塌。
- 任何新 profile 不能使用 test 指标做选择，只能用 calibration/audit 规则决定。
