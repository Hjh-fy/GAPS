# IoT-J RS0–RS4 Source-Regression-Prior 正式筛选结果

## 1. 结论

冻结建议为 **`STOP_FEDERATED_REGRESSION`**。

RS3 在 C5 calibration-validation 上相对 RS0 退化 1.799%，因此满足“退化不超过 5%”并被标记为 `paper_preferred_candidate`。但晋级还要求 RS1 或 RS2 至少一个相对 RS4 显示明确、整体的 source-prior 信息价值；本次 RS1 比 RS4 差 2.644%，RS2 比 RS4 差 0.019%。因此不能晋级真实 Flower regression network closure，也不创建 runtime v5。

本结果只支持 in-process/checkpoint FedAvg algorithm screening，不支持“已经完成真实网络联邦回归部署”的表述。

## 2. 正式协议与审计

| 项目 | 正式值 |
|---|---|
| Formal run commit | `109a095343a5e83aa401e1f52cb8b99cd975a95c` |
| Branch | `codex/iotj-confirmation-observability` |
| Classifier | frozen B5 seed42 adapted checkpoint |
| Canonical route loader | runtime v4 `C5H8Runtime.from_runtime_contract` / `FedGasBaseModel` |
| Source / target | C1、C2 / C5 |
| C5 split | calibration 320 / test 1360 |
| Calibration selection split | fit 240 / validation 80；每气体 60/20 |
| Source optimizer steps | C1=100、C2=100；总计 200 |
| Round step allocation | 34/33/33 per client |
| FedAvg | 3 次 |
| Aggregation scope | 109 regression-specific tensors；411,850 parameters |
| Device | CPU（本机 PyTorch 无 CUDA） |
| Tests | 77 passed |

Fail-closed 审计通过：

- local HEAD 与 origin HEAD 在正式运行前一致；
- manifest 与 decision gate 都绑定正式 commit；
- 80 行 calibration-validation 与 1360 行 test 的 canonical B5 route 均与 frozen RS0 reference 零差异；
- test 在 `decision_gate.json` 落盘并回读之后才打开；
- test 未用于 fit、selection 或 refit；
- 1360 个 test row key 全部唯一；
- 三轮 trace 的实际步数为 34/33/33，FedAvg 次数为 3；
- runtime v4、row map、HC95/HC90 parity report/runtime rows 六个冻结 SHA 在运行前后完全一致；
- 未出现 NaN/Inf，未修改 runtime v4 或 QC。

## 3. Calibration-validation selection

| Variant | RMSE (ppm) | 相对 RS4 | 解释 |
|---|---:|---:|---|
| RS4 rich-only | 17.2221 | 0.000% | 无 source prediction 的控制组 |
| RS2 FedAvg prior | 17.2254 | +0.019% | 几乎持平，但没有严格优于 RS4 |
| RS0 pooled source | 17.3428 | +0.701% | centralized pooled reference |
| RS3 local + FedAvg | 17.6547 | +2.512% | 相对 RS0 为 +1.799% |
| RS1 local experts | 17.6776 | +2.644% | 没有整体 local-expert 增益 |

冻结 gate：

- `selection_status=paper_preferred_candidate`
- `selected_candidate=RS3_local_plus_fedavg`
- `RS3_relative_delta_vs_RS0_percent=1.798835`
- `RS1_better_than_RS4=false`
- `RS2_better_than_RS4=false`
- `next_stage_recommendation=STOP_FEDERATED_REGRESSION`

这里的 `selected_candidate` 只表示 RS3 落入预声明的“不超过 5% 退化”区间；它不覆盖晋级建议中的 source-value 必要条件。

## 4. 一次性 test generalization evaluation

| Variant | Test RMSE (ppm) | 相对 RS4 | 相对 RS0 |
|---|---:|---:|---:|
| RS4 rich-only | 25.8985 | 0.000% | −0.486% |
| RS2 FedAvg prior | 25.9841 | +0.331% | −0.157% |
| RS0 pooled source | 26.0250 | +0.488% | 0.000% |
| RS1 local experts | 26.5726 | +2.603% | +2.104% |
| RS3 local + FedAvg | 26.6691 | +2.976% | +2.475% |

这些 test 数值只用于泛化描述，没有改变 calibration-validation 已冻结的 candidate 或 gate。

## 5. 重点问题回答

### 5.1 RS0 是否优于 RS4

整体上没有。RS0 test RMSE 比 RS4 高 0.488%，所以不能说当前 centralized pooled source predictions 提供了整体增益。

它存在明显的气体特异性：相对 RS4，RS0 在 Ethanol、CO、Methane 上的 test RMSE 分别改善 2.13%、16.74%、4.46%，CO high 200–250 ppm 改善 11.90%；但 Ethylene 退化 15.15%，抵消了整体收益。由于只有一个 seed，这些是描述性差异，不是显著性结论。

### 5.2 RS1 是否优于 RS4

没有。RS1 calibration-validation 整体退化 2.644%，test 退化 2.603%。Calibration-validation 只在 Ethanol 改善 1.13%，但 Methane 退化 26.74%；test 只在 Ethylene 改善 0.64%。C1/C2 local experts 确实产生不同预测，但没有形成稳定的整体互补价值。

### 5.3 RS2 是否优于 RS4

没有严格优于。RS2 calibration-validation 只差 0.019%，可以描述为近似持平，但按冻结规则 `RS2_better_than_RS4=false`。它在 calibration-validation 的 Ethanol/Methane 分别改善 2.12%/1.23%，而 CO/Ethylene 略差；test 仅 Ethylene 改善 1.13%，其余气体略差。FedAvg prior 的价值较弱且气体依赖，不足以作为独立晋级证据。

### 5.4 RS3 是否接近 RS0

在 selection 指标上接近：RS3 相对 RS0 退化 1.799%，落入不超过 5% 的 `paper_preferred_candidate` 区间。但 RS3 仍比 RS4 差 2.512%，且 test 比 RS0 差 2.475%、比 RS4 差 2.976%。因此“接近 pooled baseline”成立于单次 calibration-validation 阈值意义，不等于新的 prior 已产生有效增益。

## 6. Source prediction 相关性与分歧

| Scope | Pair | Pearson r | Mean absolute disagreement (ppm) | RMS disagreement (ppm) |
|---|---|---:|---:|---:|
| Calibration 320 | C1–C2 | 0.9241 | 21.4529 | 24.9261 |
| Calibration 320 | C1–FedAvg | 0.9819 | 11.4991 | 13.3031 |
| Calibration 320 | C2–FedAvg | 0.9761 | 10.7012 | 13.1074 |
| Test 1360 | C1–C2 | 0.9293 | 20.8686 | 24.2582 |
| Test 1360 | C1–FedAvg | 0.9832 | 11.2401 | 13.1500 |
| Test 1360 | C2–FedAvg | 0.9777 | 10.4634 | 12.5493 |

三模型平均 spread 为 calibration 21.8266 ppm、test 21.2861 ppm。FedAvg 与两个 local expert 高度相关，表现为平滑的共享 prior；local-local 的绝对偏移仍然较大。但 target Ridge 没有把这些差异转化为稳健的整体改善，说明“预测存在差异”不能等价为“存在可用的互补信息”。

## 7. 分气体与 CO high

Test RMSE（ppm）：

| Variant | Ethanol | CO | Ethylene | Methane | CO high 200–250 |
|---|---:|---:|---:|---:|---:|
| RS0 | 27.2128 | 22.9142 | 35.4527 | 13.6641 | 35.6278 |
| RS4 | 27.8052 | 27.5201 | 30.7880 | 14.3018 | 40.4410 |
| RS1 | 29.6766 | 27.8759 | 30.5925 | 15.1908 | 41.2793 |
| RS2 | 28.4093 | 27.5547 | 30.4398 | 14.4138 | 40.5532 |
| RS3 | 29.6743 | 28.1418 | 30.5398 | 15.4845 | 41.8793 |

FedAvg prior 没有集中表现为某个稳定受益气体：calibration-validation 的小幅收益在 Ethanol/Methane，test 的小幅收益转移到 Ethylene；CO 与 CO high 均未受益。该模式支持“气体依赖且不稳定”，不支持晋级。

## 8. 异常、限制与下一步

- Pre-formal smoke 曾发现若误用 regression model 附带分类头，B5 route 无法复现冻结 reference。正式代码已改为 runtime v4 canonical B5 loader，并通过 80/1360 行零差异门槛；错误 smoke 未作为证据保留。
- 推荐命令中的 CUDA 因本机不可用改为 CPU。协议、seed、步数、模型、数据和聚合范围未改变。
- 单 seed 不支持稳定性、置信区间或显著性声明；窗口级样本不能替代独立 seed/client 重复。
- 按冻结建议，本阶段停止。保留 runtime v4；后续可直接进入用户另行授权的 Pi benchmark、multi-seed 或 low-calibration 工作，但本次不启动。
