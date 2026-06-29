# 回归感知融合后续实验设计

## 背景

当前回归主线的性能已经比较强，但方法叙事上有一个结构性风险：最好的 H2.3 / H8+C4 结果仍然强依赖 target calibration direct-head 和 B0-dependent route-rescue/profile 层。这样会让联邦分类 backbone 看起来主要只是在做气体路由，而没有真正参与连续 ppm 回归。

最近的本地证据把这个问题变得更清楚：

- 2026-06-26 的 profile selector 审计仍然把 `H2_3_R3aK16_current_mainline` 作为 C12_to_C345 的 balanced 主线，把 `H8_plus_formal_C4_rescue` 作为 CO-priority profile。
- F6 fixed-DA strong classification 运行产生了正式最终 adapted checkpoint：第 25 轮的 `server_latest_adapted.pth`；同时第 19 轮的 `server_round_019_adapted.pth` 是一个有价值的机制诊断 checkpoint。
- 第 25 轮是正式报告主线。第 19 轮只能用于 best-checkpoint 或机制诊断，不能作为主结果，除非后面再设计 calibration-only checkpoint selector。
- 现有 H2.3 no-B0 消融说明：当前 direct-head 的 `feature_dict` 只包含 target window rich statistics，不包含 B0/R3aK16/QC-risk ppm 特征。去掉 B0-dependent rescue/profile 行为后，C4 high-CO 会明显退化。

## 设计目标

建立一条干净的后续实验路线，验证最新分类 backbone 是否能通过 route confidence 和 embedding 特征直接帮助回归；然后基于验证结果构建 H2.3+ feature fusion，以及可选的 H8+ CO risk-gated specialist。

## 非目标

- 第一阶段不重训新的 regression-aware encoder。
- 不把第 19 轮提升为正式 checkpoint。
- 不用 test 指标选择 profile、gate 或 checkpoint。
- 不把 QC accepted-only 指标混入模型能力指标。
- 在 feature ablation 结论明确之前，不把 H2.3/H8+C4 替换成复杂 MoE。

## 架构

后续实验保持现有 pipeline 边界：

1. Flower classification backbone 输出 logits、probabilities、confidence metrics、`cls_feat` 和 `reg_feat`。
2. 现有 auto_v2/R3aK16 runtime prediction 提供 B0/source prior ppm 列和 route context。
3. target calibration direct-head 训练小型 per-client、per-gas Ridge 或 ElasticNet。
4. 评估优先报告 no-QC full-set target test 指标，并分开报告 CO/high-CO/nonCO。
5. 可选 specialist gate 只使用 calibration-validation evidence。

方法可描述为：

```text
window x
  -> official F6 r25 adapted classification backbone
  -> route confidence + cls_feat + reg_feat
  -> existing B0/source ppm priors + rich response statistics
  -> target profile adapter
      -> H2.3+ balanced fusion head
      -> optional H8+ CO risk-gated specialist
  -> no-QC ppm report
  -> QC report only as deployment reliability
```

## 实验 P0：正式 backbone 特征导出

为正式 F6 最后一轮 adapted checkpoint 建立特征导出步骤：

- Checkpoint: `results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/server_latest_adapted.pth`
- Data root: `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`
- Clients: C3, C4, C5
- Splits: calibration 和 test
- 对齐主键：`(client, split, sample_index)`

导出列：

- `pred_class_f6_r25`
- `prob_0` 到 `prob_3`
- `confidence`
- `margin`
- `entropy`
- `cls_feat_000` 到 `cls_feat_063`
- `reg_feat_000` 到 `reg_feat_063`

同时为 `server_round_019_adapted.pth` 导出同样 schema，但必须写到明确标注为 diagnostic 的目录中。第 19 轮输出的 manifest 必须包含 `diagnostic_only=true`。

验收标准：

- 导出行数与 C3/C4/C5 的 calibration 和 test split 样本数一致。
- 主键能与现有 target prediction CSV 精确对齐。
- manifest 记录 checkpoint path、round、adaptive flag、clients、splits、feature dimensions，以及该导出是 official 还是 diagnostic。

## 实验 P1：H2.3 backbone 特征消融

使用与 formal H2.3/H1/H2 脚本一致的 calibration split 策略训练和评估 target direct-head。唯一变量是输入特征组。

特征组：

- `A0_rich_only`：当前 rich response statistics。
- `A1_rich_plus_confidence`：rich stats 加 confidence、margin、entropy、predicted class one-hot 和 probability vector。
- `A2_rich_plus_cls_feat`：rich stats 加 `cls_feat`。
- `A3_rich_plus_reg_feat`：rich stats 加 `reg_feat`。
- `A4_rich_plus_b0`：rich stats 加 aligned prediction CSV 中的 B0/final ppm prior 列。
- `A5_rich_plus_source_priors`：rich stats 加可用的 source/layer prediction priors。
- `A6_rich_plus_all_backbone`：rich stats 加 confidence、probabilities、`cls_feat` 和 `reg_feat`。
- `A7_rich_plus_all_priors`：rich stats 加 confidence、probabilities、`cls_feat`、`reg_feat`、B0/final ppm 和 source/layer priors。

模型：

- Ridge 是第一个必须完成的 head。
- ElasticNet 只在 Ridge 输出完成后作为可选扩展。
- P1 不做 shallow MLP。

指标：

- ALL RMSE 和 NRMSE。
- Macro-client RMSE 和 NRMSE。
- C3/C4/C5 RMSE 和 NRMSE。
- C3/C4/C5 CO RMSE。
- C3/C4/C5 high-CO RMSE。
- nonCO_ALL RMSE。
- C5 nonCO wrong-route audit，尤其是 nonCO 被预测成 CO 的样本。

验收标准：

- 如果 A2/A3/A6/A7 相比 A0 改善 ALL 或 macro-client NRMSE，并且 nonCO_ALL RMSE 增幅不超过 1.0，则说明 backbone 对回归有直接价值。
- 如果只有 A4/A5/A7 改善，则说明主要有用信号来自 source prior ppm，而不是 backbone embedding。
- 如果没有任何组改善，则当前分类 backbone 在论文里应被视为 route-only，regression-aware encoder retraining 作为单独后续任务。

## 实验 P2：H2.3+ balanced fusion profile

使用 P1 中最好的特征组定义 H2.3+。

必需候选：

- `H2_3_current_r25`：在 official r25 predictions 上重新计算的现有 formal H2.3。
- `H2_3_plus_ridge_r25`：基于选中特征组的 Ridge。
- `H2_3_plus_elasticnet_r25`：如果 Ridge 相比 A0 有改善，或者 Ridge 持平但 per-client 更稳定，则追加 ElasticNet。

选择规则：

- 候选选择只使用 calibration-validation。
- test metrics 只做最终报告。
- H2.3+ 只有在改善 ALL 或 macro-client NRMSE，并且不明显伤害 nonCO_ALL 时，才能替代 H2.3 balanced。

验收标准：

- 强成功：official r25 上 ALL RMSE 低于 18.0，并且 nonCO_ALL 相比 H2.3 current 退化不超过 1.0 RMSE。
- 中等成功：H2.3+ 优于 H2.3 direct-only，并缩小 direct-only 与 H2.3 current 的差距，证明融合特征恢复了部分 B0/profile-layer 收益。
- 失败：H2.3+ 差于 current H2.3，且不改善 direct-only head；此时保留 H2.3 作为 balanced mainline，后续再考虑 regression-aware encoder。

## 实验 P3：H8+ CO risk-gated specialist

用 calibration-validation 学到的 CO-risk gate 替代硬 C4 rescue 叙事。

Gate 候选：

- logistic gate over risk features。
- Ridge-style linear score，并在 calibration-validation 上选 threshold。

Gate 输入特征：

- predicted class 和 CO probability。
- confidence、margin、entropy。
- B0/final ppm。
- H2.3 current ppm。
- 如果可用，H2.3+ ppm。
- H8 或 CO-specialist ppm。
- balanced 与 specialist ppm 的 disagreement。
- response phase 和 phase id。
- client id。

Gate 范围：

- 第一版可以限制在 predicted CO windows。
- C4-specific 行为必须来自 calibration-validation evidence，并写入 manifest。

验收标准：

- CO/high-CO RMSE 匹配或优于 H8+C4。
- `hit_nonCO_N` 为 0，或者非常小且有明确解释。
- nonCO_ALL RMSE 相比 H2.3 current 退化不超过 1.0。
- 除非所有 guardrails 都稳健通过，否则报告中必须把它标为 CO-priority specialist，而不是 balanced default。

## 实验 P4：calibration size curve

使用 P1/P2 输出，测试融合特征是否能降低 target calibration 成本。

Calibration modes：

- 20 percent。
- 10 percent。
- 5 percent。
- 2.5 percent。
- split 支持时，每个 gas/concentration one shot。
- split 支持时，每个 gas/concentration two shots。
- 移除 high-CO calibration 做压力测试。

候选：

- B0/R3aK16 baseline。
- H2.3 current。
- H2.3+ fused profile。
- H8+C4 current specialist。
- 如果 P3 通过，加入 H8+ learned gate。
- no-backbone direct-head control。

验收标准：

- 在 10 percent calibration 下，H2.3 或 H2.3+ 应仍然优于 B0。
- 在 5 percent 和 2.5 percent 下，安全 fallback 优先于强行启用 specialist。
- 如果 H2.3+ 比 rich-only direct-head 在更少 calibration 样本下仍更稳定，论文可以主张 backbone/source-prior fusion 降低了 target calibration 成本。

## 报告规则

- 正式 headline table 只使用 F6 r25 final adapted checkpoint。
- 第 19 轮可以进入 appendix 或 diagnostic table，但必须标注为 `best-checkpoint diagnostic`。
- 模型能力指标使用 no-QC full-set target test metrics。
- QC accepted-only metrics 单独报告。
- calibration-validation 用于选择 alpha、feature、profile 和 gate。
- test metrics 绝不用于选择 feature、profile、gate 或 checkpoint。

## 交付物

- `results/f6_r25_backbone_feature_export_*/backbone_features_calibration.csv`
- `results/f6_r25_backbone_feature_export_*/backbone_features_test.csv`
- `results/h2_3_backbone_feature_ablation_*/feature_ablation_summary.csv`
- `results/h2_3_backbone_feature_ablation_*/feature_ablation_report.md`
- `results/h2_3_plus_fusion_profile_*/fusion_profile_summary.csv`
- `results/h2_3_plus_fusion_profile_*/fusion_profile_report.md`
- `results/h8_plus_co_risk_gate_*/co_gate_audit.csv`
- `results/h8_plus_co_risk_gate_*/co_gate_report.md`
- `results/calibration_size_curve_fusion_*/calibration_size_curve_report.md`

## 风险与缓解

- 风险：r25 分类准确率高，但 C5 nonCO 通过 CO wrong route 拉坏回归。
  缓解：报告 C5 nonCO wrong-route audit，避免 ALL RMSE 掩盖 route-specific failure。
- 风险：高维 embedding 在小 target calibration 上过拟合。
  缓解：先用 Ridge，标准化特征，在 calibration-validation 上选 alpha，并与 rich-only control 对比。
- 风险：source priors 主导了表面上的 backbone gain。
  缓解：在 P1 中分离 embedding-only 组与 B0/source-prior 组。
- 风险：r19 比 r25 好，诱导 checkpoint cherry-picking。
  缓解：r19 一律标记为 diagnostic-only，直到设计 calibration-only checkpoint selector。
- 风险：learned CO gate 变成另一个不透明规则。
  缓解：导出 feature coefficients 或 threshold audit、hit counts、false-hit counts 和 nonCO guard metrics。

## 已确认取舍

正式后续实验使用 F6 r25 final adapted checkpoint 作为主线。第 19 轮只保留为诊断对照。
