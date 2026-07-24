# IoT-J B5/C5 Federated-H1 Runtime v5 Candidate 结果

## 最终状态

`RUNTIME_V5_REGRESSION_READY_QC_PENDING`

分类与回归链路、真实三机 H1、seed42 RG1 重放以及 320/1360 行 runtime parity
全部通过；旧 QC 语义与 v5 不兼容，因此没有构建 v5 HC95/HC90，也没有晋级为
带自动输出决策的正式 runtime。

## A. 真实三机 Federated H1

- C1：Raspberry Pi，仅读取 C1 source train/calibration；
- C2：ECS client，仅读取 C2 source train/calibration；
- server：Alibaba Cloud ECS，只接收 feature moments、normal equations、候选模型的
  clipped validation SSE/count；
- server 未接收 raw rows、raw X/y、sample predictions 或 sample labels；
- global H1 SHA256：
  `1ca10939f09e744fdddc0dce6f5fd959ccf769e9b78142030aa7e50aa6b2f3d4`。

允许的证据表述是：source raw samples remain local and only aggregated sufficient
statistics are used to reconstruct the global source Ridge reference。这里没有实现或
声称 secure aggregation、DP 或 cryptographic privacy。

## B. H1 等价性

| Gate | 结果 | 门槛 |
|---|---:|---:|
| 四气体 alpha 一致 | 4/4 | 4/4 |
| C5 H1 prediction max diff | 6.2532e-08 ppm | <=1e-6 |
| Ridge+H1 S_ALL RMSE diff | 9.4460e-11 ppm | <=0.01 |
| Ridge+H1 S_CC RMSE diff | 6.7040e-12 ppm | <=0.01 |

结论为 `PRACTICAL_EQUIVALENCE`。

## C. seed42 RG1 精确重放

target Ridge 使用固定的 C5 calibration 240/80、104D rich + 1D Federated H1，
并在持久化 calibration lock 后才打开 test。

| 指标 | 正式重算值 |
|---|---:|
| calibration-validation RMSE | 15.3943240430 ppm |
| test S_ALL RMSE | 25.6489781431 ppm |
| test S_ALL MAE | 9.3837478583 ppm |
| test S_ALL NRMSE | 0.2042947562 |
| test S_CC N / RMSE | 1333 / 11.3415985730 ppm |
| CO RMSE | 22.6649997607 ppm |
| CO-high 200–250 ppm RMSE | 35.0212127843 ppm |

target Ridge 模型 SHA256 为
`2039d049776e7dfe0e8c4e6405dff2ae56a6e09b63f60ff2627ac0975aa075de`。

## D. Runtime v5 dependency

新模块为 `gaps_deploy/c5_federated_source_ridge_runtime.py`，bundle 只包含：

- canonical B5 seed42 classifier；
- real-topology Federated H1；
- C5 105D per-gas target Ridge。

不存在 H2/H3、R3aK16、C3/C4、H8+C4、P4、test label 或 legacy fallback。
runtime contract SHA256 为
`bca1471198f0505d4536fba71100e87279156a0c69fdd54d300ffad991b36482`。

## E. Offline/runtime parity

| Split | N | class/route mismatch | rich feature max diff | H1 max diff | final ppm max diff | nonfinite/missing |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 320 | 0 | 0 | 8.2423e-13 | 8.2423e-13 | 0 |
| test | 1360 | 0 | 0 | 1.2790e-12 | 8.2423e-13 | 0 |

两套 parity 均为 `PASS`；正式 ppm 门槛为 `1e-6`。

## F–H. QC 与 promotion

v4 QC 依赖 H2/H3、H2.3/v4 all-prior prediction 和对应 calibration risk
distribution，审计结论为 PATH B。没有构建 v5 HC95/HC90；v5 当前只能作为
回归可用、QC 待闭环的 candidate。runtime v4 及 HC95/HC90 六个冻结 SHA 均未变化。

## I. 下一步与停止边界

下一步是独立冻结并执行 v5 QC 专项闭环；不是 Pi benchmark。此次工作到此停止，
未启动 Pi/PC latency、low-calibration、新算法实验或新训练 seed。
