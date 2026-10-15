# Regression Source-Head Follow-up Plan

状态：**等待 P0 与 final classifier/prediction stream 冻结后执行**。
本文件只定义实验；本轮不得运行 Experiment A--E，不得实现 federated Ridge/MLP。

## 1. 强制执行顺序

1. 完成 B5 blocker 最小修复。
2. B2/B5 formal smoke 均达到 `status=equivalent`、`max_abs_delta=0`。
3. 冻结 final confirmation revision/archive/protocol。
4. 完成 B2/B5 × seeds 42--46 的十个 25-round canonical runs。
5. 冻结 final classifier checkpoints 与逐窗口 prediction streams。
6. 运行 source-head dependency ablation A--D。
7. 根据 A--D 决定是否执行/实现 Experiment E。
8. 最后才进入正式 12/24/48/80/Full low-calibration batch。

## 2. A--D 共用冻结合同

- classifier checkpoint、`pred_class`、class logits/probabilities、C5 calibration/test sample keys、240/80 calibration fit/validation split全部固定；任何 ablation 不得重新训练或选择 classifier。
- 所有方法共享完全一致的 C5 calibration fit/validation/test rows、Ridge/MLP grid、seed 和 routing stream。
- 删除 source head 时，重新拟合缺少该 feature 的 C5 target Ridge；禁止把缺失 prediction 静默填 0，因为这会制造不可解释的分布外输入。
- test 仅在全部 ablation 配置、输出 schema 与 hash 冻结后打开一次。
- 主指标：S_ALL RMSE/NRMSE、S_CC RMSE/NRMSE/N、S_CW、四 gas RMSE/NRMSE/N；同时记录 fit time、artifact size 与 QC FULL/HC95/HC90 yield/coverage/error。
- 结果按 B2/B5 与正式 seeds 配对；报告全部 seed 值、mean、sample std 和 paired delta。

## 3. Candidate experiments

### Experiment A — remove source Ridge

固定 final classifier prediction stream，从 R4 feature schema 删除 `H1_source_ridge_ppm`，只保留 H2/H3，使用相同 C5 calibration split 重新选择/refit target Ridge。

比较：S_ALL、S_CC、RMSE、NRMSE、per-gas RMSE，以及 QC workpoints。若明显退化，说明 H1 是真实 final dependency；若不退化，优先删除它而不是立即分布式重写。

### Experiment B — remove source per-gas MLP

删除 `H2_source_per_gas_mlp_ppm`，只保留 H1/H3；其余合同同 A。

### Experiment C — remove source shared MLP

删除 `H3_source_shared_mlp_ppm`，只保留 H1/H2；其余合同同 A。

### Experiment D — source-head-free baseline

删除全部 H1/H2/H3 prediction features，仅保留 target Ridge/MLP 路径。至少报告：

- R1 rich-only target Ridge；
- R2 target MLP；
- calibration-selected target-only Ridge/MLP candidate；
- 与完整 R4/H8 的 paired difference。

该实验直接回答“source heads 是否整体带来超出 target calibration 的增益”，是判断后续系统复杂度是否合理的最关键 baseline。

### Experiment E — conditional distributed sufficient-statistics Ridge

仅当 Experiment A 显示 source Ridge 对 R4/H8 或 QC 有稳定、实质性贡献时执行。

候选协议：

1. C1/C2 对每个 gas、每个 split 在本地计算 `n`、`sum(X)`、`sum(X^2)`、`X^T X`、`X^T y`、`sum(y)`、`min(y)`、`max(y)`；不上传逐窗口 X/y。
2. server 聚合 sufficient statistics，恢复 pooled mean/scale，并把 raw normal equations 转换为与当前 standardized Ridge 相同的 `Z^T Z`、`Z^T y`；intercept 不惩罚。
3. 每个 alpha 的模型下发给 C1/C2；客户端只返回 validation SSE/N，server 以 pooled RMSE 选择同一 alpha。
4. 使用 train+calibration aggregated statistics refit final model。
5. 与当前 pooled Ridge 比较：selected alpha、mean/scale、coef、clip bounds 和 C1/C2/C5 固定 prediction vectors。

Equivalence Gate：固定 float64/BLAS/runtime 后，selected alpha 必须完全一致；参数与 prediction 的 `max_abs_delta <= 1e-10`、`rtol <= 1e-12`。任一 gas 不通过则 fail closed，不生成“distributed Ridge equivalent”论文结论。

## 4. 优先级建议

最值得做的第一项不是单独 A，而是把 A/B/C/D 作为一个共享冻结输入的 paired dependency batch：它一次回答三种 source head 的边际贡献和全部 source heads 的整体贡献。

第二项仅在 A 证明 H1 Ridge 必不可少时做 E。若 A 无稳定退化，直接简化 final bundle 的论文价值高于实现 distributed Ridge；若 A 稳定退化，E 才能同时解决 privacy/federated-boundary 与 pooled-equivalence 两个问题。

## 5. Stop rules

- P0 未完成或 classifier stream 未冻结：不运行 A--E。
- 任一 ablation 重新训练 classifier、改变 route/sample keys 或使用 test 选配置：整批 invalid。
- A--D 证明 source heads 无稳定贡献：停止 E，优先 source-head-free bundle。
- A 证明 Ridge 重要但 E 不等价：保留 pooled Ridge 的诚实边界，不以近似结果冒充 distributed equivalence。
