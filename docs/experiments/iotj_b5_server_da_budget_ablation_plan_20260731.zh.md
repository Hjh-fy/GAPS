# B5 Server-DA 计算预算敏感性实验

## 研究问题

在 B5、seed 42、C1/C2→C5、25 个联邦轮以及客户端每轮 1 个本地
epoch 全部固定时，将服务器端域适配从每轮 100 steps 降至
80、50 和 30 steps，是否仍能保持目标 C5 的分类性能，并减少训练耗时？

## 可证伪假设

`H-DA-BUDGET-01`：至少一个低于 100 steps/round 的配置能够同时满足：

- round-25 Accuracy 相对 DA100 的绝对下降不超过 0.5 个百分点；
- round-25 Macro-F1 相对 DA100 的绝对下降不超过 0.5 个百分点；
- 没有单一类别 recall 的明显集中退化；
- 实际 server-DA 时间及总 wall time 减少。

上述 0.5 个百分点是工程性能保持容差，不构成统计非劣声明。

## 固定实验矩阵

| 配置 | local epochs/round | DA steps/round | 25轮 DA总步数 | 身份 |
|---|---:|---:|---:|---|
| LE1_DA100 | 1 | 100 | 2500 | 已完成、只读复用 |
| LE1_DA80 | 1 | 80 | 2000 | 新运行 1 |
| LE1_DA50 | 1 | 50 | 1250 | 新运行 2 |
| LE1_DA30 | 1 | 30 | 750 | 新运行 3 |

严格按 DA80→DA50→DA30 顺序运行。当前组只有在
`canonical / validator_accepted / POSTFLIGHT_PASS` 后才允许启动下一组。

## 保持不变

B5 方法、seed 42、数据及 320/1360 历史 split、C1/C2→C5 角色、三机真实
拓扑、25 rounds、客户端 1 epoch/round、batch size 32、Adam、client/server
学习率、全部 DA 损失项及权重、校准数据、checkpoint 取 round 25。

唯一允许变化的算法字段为 `server_adaptation.steps`。

## 指标

主指标为 round-25 C5 Accuracy。辅助报告 Macro-F1、NLL、ECE、error count、
per-class recall、混淆矩阵、每轮训练曲线、server-DA 时间和总 wall time。

## 证据与 test 边界

该 test universe 已在历史冻结研究中使用。本实验是在配置预先固定后追加的
post-freeze、single-seed 计算预算敏感性分析。test 不参与拟合、早停、
checkpoint 选择、步数搜索或正式 B5 重选；结果不替换五种子 B5、runtime、
回归、QC 或冻结论文证据。
