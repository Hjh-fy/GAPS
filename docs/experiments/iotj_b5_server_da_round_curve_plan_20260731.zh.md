# B5 Server-DA 逐轮目标域曲线诊断计划

## 研究问题

在固定 B5、seed 42、C1/C2→C5、25 个联邦轮次及本地训练
1 epoch/round 的条件下，服务器适配预算为 100、80、50、30
steps/round 时，C5 目标域分类指标如何随联邦轮次变化？

## 假设

`H-RC1`：服务器适配预算会影响目标域准确率的收敛速度、后期波动和最终值；
该关系不预设为单调。

## 固定协议

- 只读取已经完成的四组 `server_round_001_adapted.pth` 至
  `server_round_025_adapted.pth`，不训练、不修改 checkpoint。
- 数据角色、模型、seed、split、预处理和分类评估代码保持一致。
- 每个 checkpoint 仅计算冻结 C5 test 的 Accuracy、Macro-F1、NLL、ECE、
  error count 和 per-class recall。
- 输出只保存逐轮聚合指标和 checkpoint SHA，不保存新的逐行预测副本。
- DA100、DA80、DA50 为 canonical；DA30 始终标记为
  `blocked_observability_contract_technical_result_only`。

## Test-access 边界

这是历史 test 已经打开后的回顾性训练轨迹诊断。逐轮 test 指标不得用于：

- 早停；
- checkpoint 选择；
- DA 步数选择；
- 超参数调整；
- 替换 round-25 正式结果；
- 修改冻结 B5、runtime、QC 或论文证据。

因此“最佳轮次”“首次达到阈值”等只能作为描述性诊断，不形成模型选择依据。

## 输出

新目录：

`results/iotj_b5_server_da_round_curve_20260731/`

包含：

- `protocol_manifest.json`
- `per_round_c5_metrics.csv`
- `round_curve_summary.json`
- `b5_server_da_round_curve.png`
- `b5_server_da_round_curve.svg`
- `result_analysis.zh.md`

## 停止条件

任一组缺少 1–25 轮 checkpoint、checkpoint 内 round 与文件名不一致、
严格加载失败、C5 test 行数不是 1360、出现 NaN/Inf 或冻结资产变化时
fail closed。不得用相邻轮或其他配置 checkpoint 替代。
