# 实验室三气体 P1+P2→P3 Fold 1 结果分析

## Input contract and provenance

- Experiment IDs：
  - `LAB3GAS-P12-P3-F1-S42-R25LE3-BASE`
  - `LAB3GAS-P12-P3-F1-S42-R25LE3-DA`
- 训练源码 SHA：`4354e9f3a4a6a20eeefd2f54180b3962cb4e9e7a8ce5cae80365dd8f79846e60`
- postflight evaluator SHA：`f3f674df45a071240fda5fe1958b9eb70b5b46b86aac8460b56aba0eec89c22a`
- 数据：`client_data_lab_3gas_5fold_nominal_v1/fold_1`
- 划分：train exposure groups 3/4/5；calibration group 2；test group 1。
- P3 test 是最低保留浓度组，共 6 个独立 exposures、138 个重叠窗口：
  - 乙醛：0.399 / 0.325 ppm；
  - 甲烷：4000 / 3260 ppm；
  - 乙酸：28.52 / 23.2 ppm。
- round 选择：P1/P2 calibration exposure Macro-F1 优先，window Macro-F1
  次级，最后取更早 round；选中 round 8。
- 下表直接读取 summary 的值标记为 `reported`；差值、正确窗口数和
  per-class precision/recall/F1 标记为 `recomputed`。

## Descriptive statistics

### Selected round 与 calibration

| Scope | Variant | Accuracy | Macro-F1 | N |
|---|---|---:|---:|---:|
| P1/P2 calibration windows | round 8 base | 1.0000 | 1.0000 | 276 windows |
| P1/P2 calibration exposures | round 8 base | 1.0000 | 1.0000 | 12 exposures |
| P3 calibration windows | base | 0.9710 | 0.9710 | 138 windows |
| P3 calibration windows | adapted | 1.0000 | 1.0000 | 138 windows |
| P3 calibration exposures | base/adapted | 1.0000 | 1.0000 | 6 exposures |

### P3 test 结果

| Metric | Base | Adapted | Adapted−Base | Calculation |
|---|---:|---:|---:|---|
| Window Accuracy | 0.6522 | 0.6957 | +0.0435 | reported / delta recomputed |
| Window Macro-F1 | 0.5992 | 0.6306 | +0.0314 | reported / delta recomputed |
| Correct windows | 90/138 | 96/138 | +6 | recomputed |
| Exposure Accuracy | 0.6667 | 0.6667 | 0 | reported |
| Exposure Macro-F1 | 0.5556 | 0.5556 | 0 | reported |
| Correct exposures | 4/6 | 4/6 | 0 | recomputed |

### P3 test window-level per-class 结果

| Variant | Gas | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Base | 乙醛 | 0.9556 | 0.9348 | 0.9451 | 46 |
| Base | 甲烷 | 0.5000 | 0.1304 | 0.2069 | 46 |
| Base | 乙酸 | 0.5062 | 0.8913 | 0.6457 | 46 |
| Adapted | 乙醛 | 0.9783 | 0.9783 | 0.9783 | 46 |
| Adapted | 甲烷 | 0.8571 | 0.1304 | 0.2264 | 46 |
| Adapted | 乙酸 | 0.5294 | 0.9783 | 0.6870 | 46 |

## Assumptions, comparisons, effect sizes, and corrections

base/adapted 使用同一 round 和同一目标样本，是配对描述。DA 在窗口层面多判对
6 行，Accuracy 增加 4.35 个百分点、Macro-F1 增加 3.14 个百分点；但 138
个窗口来自 6 个 exposures，窗口高度相关，不能把它们当作 138 个独立重复做
显著性检验。暴露级只有 6 个样本，且 base/adapted 都是 4/6，因此当前没有
证据说明 DA 改善了最终暴露级决策。

单 fold、单 seed 不支持 SD、95% CI、跨 seed 效应量或统计等效判断。本记录
不执行显著性检验，也不做多重比较。

## Anomalies and sensitivity analysis

主要异常是最低浓度甲烷的系统性混淆：

- base 与 adapted 都只正确识别 6/46 个甲烷窗口，recall 仅 0.1304；
- 两者均把 40/46 个甲烷窗口判为乙酸；
- 两个甲烷 exposures 在暴露聚合后都被判为乙酸；
- DA 提升主要来自乙醛和乙酸，未修复甲烷暴露级错误。

P3 calibration group 2 在 adapted 后达到 100%，但最低浓度 test group 1
下降到 window Accuracy 69.57%、exposure Accuracy 66.67%。这说明模型对
calibration 浓度组拟合良好，但向更低浓度外推时甲烷类边界失效。可能原因包括
低浓度响应幅度接近、P3 域偏移、名义时间边界误差或 target-inclusive
normalization；当前单 fold 不能区分这些原因。

另一个现象是 source calibration exposure Macro-F1 从 round 2 起已为 1.0，
多个 round 的 window Macro-F1 也达到 1.0。round 8 只是按锁定 tie-break
得到的最早全满分 round，不应根据本次 P3 test 再改选后续 round。

## Proposed paper tables and figures

在完成无泄漏五折前不建议进入论文主表。内部诊断可暂时保留：

1. 三类 window confusion matrix（base vs adapted）；
2. 每 fold 的 exposure confusion matrix；
3. 五折完成后的 per-gas recall 箱线/点图；
4. 以浓度组为横轴的 per-gas exposure Accuracy，用于判断低浓度退化是否稳定。

## Unknowns, conflicts, and audit handoff

- source-only normalization 下的结果未知，当前数字不可作为严格跨平台最终值。
- 精确通气边界尚未确认，当前属于 nominal-boundary screening。
- 尚缺 folds 2–5 和 P2→P3，不能判断增加 P1 是否有帮助。
- 审计结论：执行完整但最终 Evidence `blocked`；详见同日完成审计。
