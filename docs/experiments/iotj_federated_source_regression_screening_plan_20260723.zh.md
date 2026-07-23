# IoT-J Federated Source Regression Single-Seed Screening Plan

## 研究目标与证据边界

- 本阶段仅执行 seed 42 的 RS0–RS4 source-regression-prior 筛选。
- 证据等级仅为 in-process/checkpoint FedAvg 算法筛选，不代表真实网络联邦回归部署。
- 不修改 runtime v4，不接入 QC，不运行 Pi benchmark、multi-seed 或 Flower regression network。

## 假设

| ID | 可证伪假设 | 基线 | 干预 | 主指标 | 判据 |
|---|---|---|---|---|---|
| FSRP-H1 | centralized pooled source predictions 相对 rich-only 有增益 | RS4 | RS0 | C5 test ALL/per-gas RMSE | 只报告 generalization，不参与 selection |
| FSRP-H2 | C1/C2 device-specific local experts 提供互补信息 | RS4 | RS1 | C5 calibration-validation ALL/per-gas RMSE | RS1 RMSE < RS4 |
| FSRP-H3 | FedAvg global regression prior 独立提供信息 | RS4 | RS2 | C5 calibration-validation ALL/per-gas RMSE | RS2 RMSE < RS4 |
| FSRP-H4 | 不集中混合 raw source data 时，Local + FedAvg 接近 pooled baseline | RS0 | RS3 | C5 calibration-validation ALL RMSE | RS3 优于 RS0，或退化不超过 5% |

## 冻结协议

- Source clients：C1、C2，各自只读取本地 train rows。
- Target client：C5；frozen calibration/test 为 320/1360。
- Calibration 内部固定 75% fit / 25% validation，即 240/80。
- Classifier：frozen B5 seed42 adapted checkpoint。
- 分类路由必须通过 runtime v4 contract 调用 `C5H8Runtime.from_runtime_contract` 加载正式 `FedGasBaseModel`；禁止复用 source-regression model 附带的分类头。
- 正式运行前 smoke 必须显示 frozen RS0 validation/test route 零差异（80/80、1360/1360）。
- Source local model：相同 DCT16 architecture、完全相同初始化并记录 hash。
- `source_steps=100` 表示每个 client 整个实验共 100 optimizer steps；3 rounds 分配为 34/33/33。C1=100、C2=100，总计 200；FedAvg 执行 3 次。
- FedAvg 仅聚合已审计的 109 个 regression-specific tensors，按 source sample count 加权；classifier/backbone 保持冻结。
- Target model：per-gas C5 Ridge；alpha 仅由 calibration-validation 选择，之后在 full calibration refit。
- C5 test 不参与 fit、select 或 refit。
- QC 关闭；runtime v4 只读。
- 本机 PyTorch 无 CUDA，因此正式命令使用 `--device cpu`。这只改变执行设备，不改变 seed、数据、结构、步数、聚合范围或 selection protocol。
- 本阶段为 single-seed，不提供跨 seed 稳定性或显著性结论。

## Candidate selection gate

必须先落盘并回读 `decision_gate.json`，之后才允许首次加载 C5 test。

- RS3 优于 RS0：`candidate`，选择 RS3。
- RS3 退化不超过 5%：`paper_preferred_candidate`，选择 RS3。
- RS3 退化位于 5%–10%：`inconclusive`，不冻结最终方法。
- RS3 退化超过 10%：`no_promotion`，不晋级。

下一阶段建议只使用 calibration-validation 证据：

- `PROMOTE_FOR_CONFIRMATION`：RS3 优于/接近 RS0，且 RS1/RS2 至少一个优于 RS4，同时收益不只依赖单一气体。
- `INCONCLUSIVE`：RS3 退化位于 5%–10%，或收益高度依赖单一气体。
- `STOP_FEDERATED_REGRESSION`：其余情况。

C5 test 只用于一次性 generalization evaluation，禁止按 test 排名更换 variant。

## Fail-closed 停止条件

任一情况立即停止：本地与 origin commit 不一致、冻结 SHA 改变、相关测试失败、output root 非空、C1/C2 初始化不一致、聚合范围不是 109 tensors、classifier/backbone 状态改变、row identity 不一致、B5 route parity 非零、test 在 decision gate 之前加载、NaN/Inf。
