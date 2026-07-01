# 当前基座回归主线优化设计

## 背景

当前阶段已经完成了 F6 fixed-DA strong r25 分类基座、H2.3+ target direct-head weak-blend、H8+C4 rescue、real-route vs oracle-route gap、low calibration stress，以及 guarded profile selector。老师提出的关键口径是：重点报告“分类正确下的回归性能指标”。因此后续不应继续把分类错误、route 错误和回归头能力混在一个 real-route full-set 指标里解释。

同时，最近一轮证据说明 R3aK16/auto_v2 不再适合作为每轮优化都重复训练的回归主线。H2.3+ 的收益主要来自目标域 calibration direct-head 和 reg_feat weak-blend；H8 的收益主要来自 C5 CO-priority rescue 和 source/target augmented head。R3aK16 仍然有 baseline、fallback、runtime context 价值，但不应继续占据后续实验的训练预算。

## 设计目标

在当前 F6 分类基座上，建立一条固定、可复现、可汇报的回归主线：

- 主报告聚焦 classification-correct / oracle-route full-set。
- 部署补充聚焦同一 QC policy 下的 Accepted+Review。
- real-route full-set 只作为 route/classification error impact 的风险说明。
- R3aK16 冻结为 baseline/context，不再进入每轮 profile calibration 的重训循环。
- 后续优化集中到 C5 CO-priority calibration/rescue，而不是继续训练全局神经回归头。

## 非目标

- 当前阶段不扩展新的源域/目标域组合。
- 当前阶段不重新训练 R3aK16。
- 当前阶段不设计新的 regression-aware encoder。
- 不用 test set 反推 profile、blend weight、guard threshold。
- 不把 QC accepted-only 或 Accepted+Review 替代 oracle-route full-set 主指标。
- 不把 strict guard 作为唯一结论；strict guard 仅作为保守边界，practical guard 作为当前主线候选。

## 冻结基座

后续实验默认读取当前已有基座产物：

- F6 分类基座：`results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/server_latest_adapted.pth`
- F6 backbone features：`results/f6_r25_backbone_feature_export_20260630/`
- H2.3+ real-route：`results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/`
- H2.3+ oracle-route：`results/h2_3_plus_fusion_profile_20260630/r25_oracle_route_replay_gate/`
- H8+C4 real-route：`results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector/`
- H8+C4 oracle-route：`results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/formal_c4_route_rescue_selector/`
- Guarded practical selector：`results/guarded_profile_selector_nonco05_20260630/`
- QC records：`results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/qc_test_records.csv`

R3aK16/auto_v2 只保留为：

- 历史 baseline。
- 部署兼容字段来源。
- gate context，例如 `final_ppm`、risk、confidence。
- 需要导出部署包时的 fallback/runtime 对照。

只有当数据 split、分类基座、正式 baseline 口径或部署包格式发生变化时，才重新生成 R3aK16 相关产物。

## 当前证据

### Oracle-route Full 主表

| profile | ALL RMSE / NRMSE | C3 | C4 | C5 |
|---|---:|---:|---:|---:|
| H2.3 | 10.57 / 0.0564 | 9.63 / 0.0540 | 9.28 / 0.0527 | 13.23 / 0.0643 |
| H2.3+ | 9.86 / 0.0515 | 9.16 / 0.0487 | 8.50 / 0.0482 | 12.18 / 0.0596 |
| H8+C4 | 9.10 / 0.0511 | 9.13 / 0.0522 | 8.55 / 0.0502 | 9.57 / 0.0499 |
| Guarded practical | 9.11 / 0.0489 | 9.16 / 0.0487 | 8.50 / 0.0482 | 9.57 / 0.0499 |

解释：

- H2.3+ 是 C3/C4 的稳定 balanced direct-head。
- H8+C4 的主要收益来自 C5。
- Guarded practical 保留 C5 的 H8 收益，同时避免 C3/C4 被 H8 的 NRMSE 退化拖累。

### Oracle-route Accepted+Review

| profile | ALL RMSE / NRMSE | C3 | C4 | C5 |
|---|---:|---:|---:|---:|
| H2.3+ | 6.997 / 0.0376 | 5.653 / 0.0328 | 6.317 / 0.0343 | 9.528 / 0.0478 |
| H8+C4 | 6.476 / 0.0371 | 5.756 / 0.0352 | 6.540 / 0.0359 | 7.632 / 0.0415 |
| Guarded practical | 6.375 / 0.0356 | 5.653 / 0.0328 | 6.317 / 0.0343 | 7.632 / 0.0415 |

解释：

- Accepted+Review 固定来自同一份 QC records，适合作为部署口径补充。
- Guarded practical 在保持 C3/C4 H2.3+ 优势的同时，拿到 C5 H8 rescue 收益。

### Route Gap

real-route full-set 与 oracle-route full-set 的差距主要来自分类/route 错误，尤其集中在 C5：

- H2.3+ ALL real-oracle RMSE gap 约 12.57。
- H2.3+ C5 real-oracle RMSE gap 约 27.09。
- H8+C4 C5 real-oracle RMSE gap 约 29.56。

因此 real-route full-set 不适合作为 profile 本身优劣的主判断口径。

## 方法设计

### 主线 profile policy

当前基座上的主线 policy 为：

```text
C3 -> H2.3+ target direct-head weak-blend
C4 -> H2.3+ target direct-head weak-blend
C5 -> H8+C4 CO-priority rescue
```

该 policy 由 guarded practical selector 支撑。H2.3+ 是默认 profile；H8+C4 只有在 validation RMSE margin、validation NRMSE、low-cal stability 和 practical nonCO guard 通过时才切换。

当前 guard 参数：

- `min_rmse_margin = 0.5`
- `max_nrmse_delta = 0.0`
- `min_h8_stability = 0.7`
- `max_nonco_delta = 0.5`
- `stability_budget = 96`

### 汇总产物

新增一个当前基座主线汇总脚本，负责读取已有 CSV，而不是重新训练模型。它应输出：

- `current_base_regression_mainline_summary.csv`
- `current_base_regression_post_qc_summary.csv`
- `current_base_regression_route_gap_summary.csv`
- `current_base_regression_low_cal_summary.csv`
- `current_base_regression_story.zh.md`
- `manifest.json`

报告必须包含：

- 冻结基座说明。
- R3aK16 停止重复训练的理由。
- oracle-route full 主表。
- Accepted+Review 补充表。
- real-route vs oracle-route gap 摘要。
- low calibration 稳定性摘要。
- 下一步 C5 优化计划。

### 后续 C5 优化方向

当前基座内的后续优化只围绕 C5：

1. C5 CO-specific calibration：重新审计 C5 CO low/mid/high bin 的 residual pattern。
2. C5 H8 rescue stability：在 low calibration budget 下复核 H8 profile selection 与 C5 CO-bin 性能。
3. C5 route-risk guard：把 real-route 下的 C5 route 错误影响拆成 route-confidence bins，明确哪些样本应该交给 review/reject。

这些实验只读取冻结基座和已有 profile predictions，不重训 R3aK16。

## 测试与验证

新增汇总脚本需要最小测试覆盖：

- 能从简化的 oracle/full/post-QC/gap/stress 输入表中抽取指定 profile 与 scope。
- 能正确生成 guarded practical 的主线行。
- 缺少必要 profile 或列时抛出明确错误。
- report writer 输出包含 oracle full、Accepted+Review、route gap、R3aK16 freeze 四个关键段落。

回归验证需要运行：

```powershell
python -m pytest tests/test_current_base_regression_story.py tests/test_guarded_profile_selector.py tests/test_profile_qc_coverage_audit.py -q
python -m pytest tests/test_low_calibration_profile_choice_stress.py tests/test_low_calibration_blend_stress.py tests/test_route_gap_audit.py -q
```

如果实现只新增汇总脚本和报告，不需要重新跑重训练流程。

## 成功标准

- 主报告中可以用一张 oracle-route full 表回答“分类正确下回归性能如何”。
- Accepted+Review 表能作为部署补充，而不混淆主指标。
- real-route gap 明确说明分类/route 错误污染了 full-set。
- R3aK16 的新定位清晰：保留 baseline/context，不再每轮重训。
- 后续优化计划收敛到 C5 CO-priority，而不是继续扩大全局 profile 搜索。

## 风险与处理

- 风险：strict guard 与 practical guard 结论不同。
  处理：报告中同时保留 strict guard 作为保守边界，主线使用 practical guard，并解释 C5 nonCO 验证波动只有 0.374 ppm，CO/full/Accepted+Review 收益更大。

- 风险：real-route full-set 看起来没有明显优于 H2.3+。
  处理：把 real-route full-set 放入 route-error impact section，不作为回归 profile 主指标。

- 风险：后续换源域/目标域时当前 policy 不一定成立。
  处理：当前 spec 只覆盖 C12 source 到 C345 target 的当前基座；跨域验证作为下一阶段独立 spec。

## 交付顺序

1. 写汇总脚本和单元测试。
2. 运行脚本生成当前基座主线汇总结果。
3. 写正式中文 story 报告。
4. 跑相关测试并提交。
5. 等当前基座故事稳定后，再进入跨源域/目标域泛化验证。
