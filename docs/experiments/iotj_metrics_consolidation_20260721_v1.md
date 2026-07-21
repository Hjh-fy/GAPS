# IoT-J 回归、系统与部署指标汇总（2026-07-21，v1）

此文档只汇总已有、可追溯的实验数字；没有启动新训练、重新拟合回归头、改变模型或进行性能优化。

完整的机器可读指标清单、结果分析、实验审计和 claim–evidence map 位于：
`results/iotj_metrics_consolidation_20260721_v1/`。

## 当前可汇报指标

| 证据层级 | 指标 | 当前数值 | 可安全表述的范围 |
|---|---:|---:|---|
| 历史分类 screening | B2 / B5 accuracy（seed 42） | 99.2647% / 98.8971% | 单 seed screening；不得写为 five-seed mean/std。 |
| B5 C5 回归 | R4 RMSE / NRMSE | 26.0250 ppm / 0.2073 | 固定 B5 prediction stream 的 1,360 个 C5 test 窗口。 |
| B5 C5 QC | HC90 yield / accepted RMSE | 90.8088% / 15.8328 ppm | accepted RMSE 仅针对 1,235 个自动接受窗口。 |
| 真实三机系统 | application communication | 16.7586 MiB / 25 rounds | serialized Flower application 层；非 transport/wire bytes。 |
| 真实三机系统 | mean round / server DA | 237.29 s / 161.34 s | 一次 canonical B5 seed-42 ECS + Pi + ECS-C2 run。 |
| Pi 训练资源 | peak RSS / peak temperature | 518.38 MiB / 62.25 C | 训练侧；未观察到 throttling。 |
| 部署 candidate v2 | runtime asset size | 2.8378 MiB | 10 个运行资产；仍非最终发布包。 |

该三机 run 中，server-side DA 占平均 round wall time 的 67.99%，而 Pi local training 为 41.96 s/round。因此当前可写为一个**限定范围内的初步系统观察**：在该 B5 真实云边拓扑中，服务器端 DA 是主要已量化时间组成，而不是 Pi 本地训练。

## 不得提升的结论

- B2 a006 已恢复的 25 轮记录仍是 failed diagnostic，不能作为 canonical 或 paired confirmation 结果。
- B1–B5 seed-42 不能支持稳定性、显著性或 five-seed 算法优势结论。
- bundle v2 已通过输入/哈希审计，但尚无独立 runtime 生成的 1,360 行 parity report；Pi/PC batch-1 latency、p50/p95/p99、inference RSS/CPU 均为 `unknown`。
- 训练侧 RSS、CPU 和温度不能替代最终部署 runtime 的资源指标。

## 已完成的只读/完整性检查

1. 23 条指标记录的 schema 与 metric ID 唯一性检查通过。
2. B5 bundle v2 的 10 个 runtime asset SHA-256 与 manifest 一致。
3. 1,360 行 parity reference 被保留为 bundle 外部证据，未复制进 `assets/`。
4. B5 canonical system JSON、R4 FULL 和 HC90 JSON 与指标清单的关键数值交叉核对通过。

## 下一项测试（不做优化）

完成 B5-only runtime 的离线 1,360 行 parity：类别、profile、QC decision 必须逐行一致，ppm 最大绝对误差不超过 `1e-6`。只有该 gate 通过后，才测试 Pi/PC 的真实推理 latency、RSS 与 CPU；之后才考虑正式低校准预算实验。
