# B5→C5 当前部署量化指标（2026-07-21）

## 结论边界

本记录整合的是一个 canonical B5、seed 42、C1/C2→C5 的真实 ECS + Pi + ECS-C2 训练运行，以及其绑定的 C5 deployment rebuild。它可用于说明当前 B5 候选在真实拓扑中的系统成本和 C5 离线回归/QC 表现；它**不是** B5 的五种子 confirmation，也不是已经在 Pi 上完成的最终 runtime benchmark。

分类 checkpoint 为 `server_round_025_adapted.pth`，SHA-256 为 `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`。所有 C5 回归资产仅使用 C1/C2 source 或 C5 calibration 拟合；C5 test 的 1360 行仅用于离线评估。

## 当前可报告的算法与回归/QC 指标

| 类别 | 指标 | 数值 | 范围与状态 |
|---|---:|---:|---|
| 分类 | C5 test accuracy | 98.0147% (1333/1360) | B5 canonical deployment prediction stream；单 seed |
| 分类 | Ethanol / CO / Ethylene / Methane recall | 96.7647% / 99.1176% / 96.7647% / 99.4118% | 同一 1360 行 prediction stream |
| 回归 R4 | RMSE / NRMSE / MAE | 26.0250 ppm / 0.2073 / 9.4860 ppm | C5 1360 test；预测 route；无 C4 rescue |
| QC FULL | automatic yield | 100.00% | 仅作为无拒绝参照 |
| QC HC90 | automatic yield | 90.8088% (1235/1360) | `deployment_risk_full` policy |
| QC HC90 | accepted RMSE | 15.8328 ppm | 仅对自动接受的 1235 行 |
| QC HC90 | nonreject coverage | 98.6765% (1342/1360) | accept + review，不等于 automatic yield |
| QC HC90 | wrong-route recall | 81.4815% (22/27) | 当前 QC 检出的错路由比例 |

目前尚未从该 prediction stream 得到可审计的 NLL/ECE；不能以训练端或历史 screening 表中的数值代替。

## 真实三机训练侧系统指标

| 指标 | 数值 | 证据范围 |
|---|---:|---|
| Flower serialized application communication（25 rounds total） | 16.7586 MiB | logical/serialized application 层；不含 transport bytes |
| 平均 round wall time | 237.294 s | 25-round B5 canonical run |
| total wall time | 5932.341 s（约 98.9 min） | 同上 |
| server DA mean / p95 | 161.336 s / 163.301 s | 占平均 round wall 67.99% |
| Pi C1 local training mean | 41.960 s/round | 真实 Pi 训练进程 |
| ECS-C2 local training mean | 74.746 s/round | 真实 ECS-C2 训练进程 |
| Pi RSS active mean / peak | 514.074 / 518.375 MiB | 资源采样覆盖率 97.55% |
| ECS-C2 RSS active mean / peak | 508.538 / 513.707 MiB | 资源采样覆盖率 97.72% |
| Pi CPU mean / peak | 84.44% / 90.89% | host CPU percent |
| ECS-C2 CPU mean / peak | 49.94% / 58.80% | host CPU percent |
| Pi temperature mean / peak | 57.42°C / 62.25°C | 未观察到 throttling |
| Observer total overhead | 5969.05 ms | 约为总 round wall 的 0.1006% |

该表支持的初步系统观察是：在此单次真实拓扑运行中，server-side DA 仍为最大的可量化 round-time 组成，而不是 application message bytes。它不能外推到其他 seed、B2 或最终 Pi inference。

## 当前可导出的 deployment assets（尚非最终 bundle）

| 资产 | 大小 | 用途 |
|---|---:|---|
| B5 classifier checkpoint | 184,201 B（0.1757 MiB） | C1/C2 federated classifier；SHA 见上 |
| C5 R4 policy | 498,562 B（0.4755 MiB） | H1/H2/H3 source references + C5 target Ridge；四类预测路由 |
| C5 H2.3 reference | 794,125 B（0.7573 MiB） | C5 calibration-only MLP/Ridge blend reference |
| HC90 risk policy | 679 B | frozen decision-policy metadata |
| HC90 component calibrator | 11,009 B | QC component calibration |
| HC90 feature reference | 1,466,440 B（1.3985 MiB） | QC feature reference |
| HC90 risk selection | 16,723 B | QC selection metadata |
| 当前核心资产合计 | 2,971,739 B（2.8341 MiB） | 不含 feature schema、class map、normalization、bundle manifest 和 SHA-256 清单；不能称为 final bundle size |

## 已完成与下一步

已完成：B5 真实 25-round canonical run、通信/时间/RSS/CPU/温度采集、C5 R4 与 H2.3 calibration-only assets 重建、HC90 operational QC、R4 运行时四类路由加载合同。

下一步（按顺序）：

1. 将 classifier、R4、H2.3、QC、feature schema、class map 和 normalization 封装为不可变 bundle，并计算逐文件及整体 SHA-256。
2. 以 C5 1360 test 行生成 runtime stream，执行 class/profile/QC 完全一致且 ppm 最大绝对误差不超过 `1e-6` 的 parity gate。
3. parity 通过后，才在 Pi 和 PC 做 batch=1 / batch=32、30 warm-up、至少 100 次测量的 p50/p95/p99 推理延迟与 runtime RSS/CPU 实测。
4. final classifier/prediction stream 冻结后，再启动 12/24/48/80/Full 的正式 low-calibration 回归/QC batch；当前不得把它与本记录混合。

## 机器可读证据

- `results/iotj_ecs_c2_b5_canonical_analysis_20260721/b5_canonical_system_metrics.json`
- `results/iotj_b5_c5_deployment_p1_20260721/rebuilt_suite/B5canonical/inputs/c5_target_layer_predictions.csv`
- `results/iotj_b5_c5_deployment_p1_20260721/rebuilt_suite/B5canonical/r0_r7/r0_r7_summary.csv`
- `results/iotj_b5_c5_deployment_p1_20260721/rebuilt_suite/B5canonical/high_coverage_qc/operational_summary.json`
- `results/iotj_b5_c5_deployment_p1_20260721/rebuilt_assets_pending/r4_policy.json`
- `results/iotj_b5_c5_deployment_p1_20260721/rebuilt_assets_pending/h23_reference.json`
