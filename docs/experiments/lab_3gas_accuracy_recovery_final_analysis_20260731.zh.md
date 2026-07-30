# 实验室三气体准确率恢复：完整结果分析

## Input contract and provenance

- 实验方向：P2/C2 → P3/C3，三分类（0=乙醛、1=甲烷、2=乙酸）。
- 数据协议：`all_concentration_timepurged_p2_to_p3_v1`；P2 train 420 窗口，P3 calibration 90 窗口，完整 P3 test 420 窗口。
- 训练配置：seed 42，25 轮联邦训练，本地 3 epoch，固定选择第 25 轮；P3 test 不参与选轮。
- REC-A2-TCE-RETRY2 于 2026-07-31 00:21:15 完成。25/25 轮均有 base/adapted checkpoint；每轮 `fit_failures=0`、`evaluate_failures=0`，postflight audit 为 `valid`。
- REC-A2 实际 runtime 源码 archive 为
  `6af5aa66d729bb65a0e31f4c8ea6497b6af478fb92da1a98386b00593232a88c`。
  相对 REC-A1 的 archive，源文件差异仅为客户端 evaluate 异常日志和 strategy
  evaluate failure detail 记录；训练与推理算法未改变。
- 正式指标均从相应 `formal_evaluation_summary.json` 复制，标记为 `reported`。
  表中的正确数、类别召回率、百分点评价和损失占比为本次从已确认 JSON 计算，标记为
  `recomputed`。

## Descriptive statistics

| Metric ID | n/seeds | Accuracy | Macro-F1 | 正确数 | 暴露级 Accuracy | Scope | Source |
|---|---:|---:|---:|---:|---:|---|---|
| REC-A0-LR25 | 1 | 94.52% | 94.50% | 397/420 | 100% | 完整 test | reported |
| REC-A1-CB2 | 1 | 94.52% | 94.50% | 397/420 | 100% | 完整 test | reported |
| REC-A2-TCE-RETRY2 | 1 | 94.52% | 94.51% | 397/420 | 100% | 完整 test | reported |
| REC-A3-COND | 1 | 93.57% | 93.60% | 393/420 | 100% | 完整 test | reported |
| REC-A5-NOCH2 | 1 | 93.81% | 93.83% | 394/420 | 100% | 完整 test | reported |
| A1 checkpoint 稳定段诊断 | 1 | 96.67% | 96.67% | 348/360 | 100% | A4 common scope | recomputed diagnostic |
| REC-A4-STABLE150 | 1 | 99.72% | 99.72% | 359/360 | 100% | 稳定 test，coverage=85.71% | reported |

只有一个 seed，且同一 exposure 内窗口相关，不能把 420 或 360 个窗口当作独立实验重复。
因此跨 seed 的 SD、95% CI 和显著性检验均为 `unknown`。

## Assumptions, comparisons, effect sizes, and corrections

### REC-A2：固定权重 P3 目标监督没有提高完整时段准确率

REC-A1 与 REC-A2 的完整测试范围、数据、通道、模型、DA、轮数、local epoch、seed
和第 25 轮 checkpoint 规则一致；唯一预定算法变量是
`lambda_target_ce: 0 → 1.0`。

| 实验 | 乙醛召回率 | 甲烷召回率 | 乙酸召回率 | 混淆矩阵 |
|---|---:|---:|---:|---|
| REC-A1 | 140/140=100% | 119/140=85.00% | 138/140=98.57% | `[[140,0,0],[0,119,21],[2,0,138]]` |
| REC-A2 | 140/140=100% | 121/140=86.43% | 136/140=97.14% | `[[140,0,0],[0,121,19],[4,0,136]]` |

- Accuracy 差值：0.00 个百分点，正确窗口数相同，均为 397/420。
- Macro-F1 差值：+0.0083 个百分点，仅为描述性微差。
- REC-A2 少了 2 个“甲烷→乙酸”错误，但多了 2 个“乙酸→乙醛”错误，属于错误转移，
  不是净提升。
- REC-A2 第 25 轮 unadapted 与 adapted Accuracy 也同为 94.52%；最终一次 DA 没有带来
  Accuracy 增益。

目标监督未解决瓶颈的诊断证据：

1. P3 calibration 在第 25 轮 unadapted 和 adapted 上都已 90/90 正确，监督信号已经饱和。
2. 第 1 轮 `weighted_target_ce_loss=0.7932`，而 `total_loss=1342.1021`，
   目标 CE 仅占约 0.0591%；第 25 轮其占比约为
   `6.75e-6 / 3613.03 = 1.87e-9`。
3. calibration 窗口位于每次暴露的 150–250、550–650、950–1050 s，
   没有给最难的 0–150 s 早期响应提供直接监督。

所以当前结果不支持继续把固定 `target_ce_weight=1.0` 作为主要提准手段。若要再次研究
target CE，应先改变它的样本覆盖或损失尺度，而不是只重复同一权重。

### 完整 420 窗口的单变量比较

- REC-A3 相对 REC-A1 少正确 4 个窗口，Accuracy 下降 0.95 个百分点。
- REC-A5 相对 REC-A1 少正确 3 个窗口，Accuracy 下降 0.71 个百分点。
- REC-A2 与 REC-A1 正确数完全相同。
- 因此当前 seed 下，精确相对电导、去除 CH2、固定权重目标 CE 均未超过
  “6 通道相对电阻 + corrected B2”的完整时段基线。

### 稳定段 REC-A4

- A1 原 checkpoint 仅在相同 360 稳定窗口上评估时为 348/360=96.67%。
- REC-A4 重新使用稳定段 P2 train 和对应 P2-only 归一化后为 359/360=99.72%。
- 相同评估范围下多正确 11 个窗口，描述性差值为 +3.06 个百分点。
- 但 REC-A4 同时改变 P2 训练窗口组成和归一化统计，不能把 +3.06 个百分点完全归因于
  “删除早期测试窗口”。
- REC-A4 coverage 为 360/420=85.71%，不能表述为全时段 99.72%。

## Anomalies and sensitivity analysis

- REC-A2 初次 attempt 在第 1 轮 evaluate 后失去客户端；retry1 又因遗留 SSH 反向隧道占用
  18080 端口而在训练前失败。两次失败均保留为 blocked 记录。
- REC-A2 retry2 使用新 run ID 和新内容寻址 runtime，越过原故障点并完成 25 轮；完成后
  Server A 与 Cloud B 均无该 run 的遗留训练进程。
- A1 在被 A4 排除的 60 个早期窗口上为 49/60=81.67%，明显低于稳定 360 窗口的
  348/360=96.67%。这支持“早期动态响应是主要困难区”，但精确进气边界尚未复核。
- 所有完整实验的暴露级 Accuracy 都为 30/30=100%；当前差异集中于 exposure 内局部窗口，
  不是整次气体暴露被判错。

## Proposed paper tables and figures

- 完整 420 窗口消融表：A1/A2/A3/A5，附类别召回率和暴露级 Accuracy。
- 覆盖率—准确率表：A1 完整、A1 stable common-scope、A4 stable protocol。
- 后续保存逐窗口预测后，绘制“相对目标气体起点时间—错误率/置信度”曲线。

## Unknowns, conflicts, and audit handoff

- 跨 seed 稳定性、独立采集批次泛化能力为 `unknown`。
- 当前 P3 test 已被多次查看，消融结论属于 post-hoc 探索。
- 精确进气和结束边界尚未审核，当前时间段为名义边界。
- 与公共数据集 98–99% 不构成同任务公平对照：类别、设备域、采集流程和样本独立性不同。
- 可用于内部决策的当前排序是：完整覆盖保留 A1 设置；稳定段使用 A4 协议；A2/A3/A5
  暂不组合。该排序不得升级为论文级确认性结论。
