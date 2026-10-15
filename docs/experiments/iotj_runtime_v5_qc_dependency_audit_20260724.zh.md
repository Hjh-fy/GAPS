# IoT-J Runtime v5 QC 依赖审计

## 审计结论

结论为 `PATH_B_RISK_SEMANTICS_INCOMPATIBLE`。现有 runtime v4 的 HC95/HC90
风险语义不能直接复用于 B5 + Federated-H1 Runtime v5，因此本阶段没有复制
阈值、没有临时发明新风险模型，也没有构建 v5 HC95/HC90。

## 依赖证据

v4 的正式 `deployment_risk_full` 由三组 percentile 分量平均得到：

1. classifier confidence：entropy 与 margin；
2. feature distance：B5 64D classifier feature 的 prototype/support distance；
3. regression disagreement：expert disagreement 与 source spread。

其中第三组与 v5 不兼容：

- `raw_risk_expert_disagreement` 需要 `h23_plus_ppm` 与
  `target_ridge_plus_source_preds_ppm`，两者都属于 v4 回归结构；
- `raw_risk_source_spread` 同时需要 H1、H2 per-gas MLP、H3 shared MLP；
- v4 HC95/HC90 阈值是在 calibration-validation 的上述完整风险分布上冻结的，
  不能解释为只依赖分类置信度或 B5 feature distance。

虽然 confidence、prototype、support 三个分量本身可以由 v5 计算，但删除
disagreement 分量会改变风险分数的定义与分布，因而不属于“语义兼容后重新校准”。

## 决策边界

- v5 回归 runtime 保持 QC disabled；
- v4 runtime、HC95、HC90 继续作为只读历史正式基线；
- 下一步必须先冻结独立的 v5 QC protocol，仅用 C5 calibration 选择风险结构和
  HC95/HC90 阈值，再一次性打开 test；
- 在 v5 QC 闭环前不进入 Pi benchmark。

机器可读审计见
`results/iotj_b5_c5_runtime_v5_candidate_20260724/qc/qc_dependency_manifest.json`。
