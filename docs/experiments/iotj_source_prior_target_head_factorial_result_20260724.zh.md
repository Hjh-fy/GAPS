# IoT-J Source-Prior × Target-Head E1/E2 正式结果

## 1. 决策结论

Calibration-validation 冻结 gate 为：

- `selection_status=NEW_CANDIDATE_PENDING_CONFIRMATION`
- `selected_candidate=E1_MLP_PRIOR`
- MLP+prior 相对当前 Ridge+prior 的 calibration-validation RMSE 改善 **19.143%**，超过预注册的严格 `>5%` 门槛。
- `runtime_action=none`

这只表示 MLP+prior 获得后续确认资格，不表示可以替换 runtime v4。一次性 test generalization 出现反转：MLP+prior RMSE 为 27.275，当前 Ridge+prior 为 26.025，MLP+prior 反而差 **4.805%**。按照预注册协议，test 不能撤销 calibration 已冻结的 candidate 标签，但该反转是阻止 runtime 修改的重要风险证据。

最终操作：**保持 runtime v4 与 QC 不变，不创建 runtime v5；本阶段停止。**

## 2. 协议与审计

| 项目 | 正式值 |
|---|---|
| Formal run commit | `ead9f93d76994c1b653b17820fc92071314aa5b6` |
| Classifier | frozen B5 seed42 / runtime-v4 `FedGasBaseModel` |
| C5 split | calibration 320（fit 240 / validation 80）；test 1360 |
| Rich feature dimension | 104 |
| Prior feature dimension | H1/H2/H3 各1维；all-prior 总维度107 |
| Ridge alpha grid | 0, 0.01, 0.1, 1, 10, 100, 1000 |
| MLP hidden grid | 16；32；64；32,16 |
| MLP alpha grid | 0.001, 0.01, 0.1, 1 |
| MLP protocol | ReLU、LBFGS、max_iter=800、no early stopping、seed42 |
| QC/runtime | 关闭 / 未修改 |
| Tests | 81 passed |

Fail-closed 审计通过：

- local/origin 在正式运行前一致；
- calibration-validation 80个唯一row key，test 1360个唯一row key；
- 28条per-gas记录、28条per-class selection记录、8条factorial effect记录齐全；
- canonical B5 route 与正式 reference 在80/1360行均为零差异；
- H1/H2/H3 replay 最大绝对差异约 `1.0e-12`；
- test 未用于 fit、select 或 refit，gate 先于 test metrics 落盘；
- 所有正式 runtime、HC95/HC90、R4、H2.3资产 SHA 在运行前后相同；
- 未重新训练 B5 或 H1/H2/H3，未修改 runtime v4 或 QC。

## 3. Calibration-validation

| Variant | Head | Prior | Input dim | Parameters | RMSE |
|---|---|---|---:|---:|---:|
| E1_MLP_PRIOR | MLP | H1+H2+H3 | 107 | 19,700 | **14.0229** |
| E2_RIDGE_H1 | Ridge | H1 | 105 | 424 | 15.3943 |
| E2_RIDGE_H3 | Ridge | H3 | 105 | 424 | 15.9700 |
| E2_RIDGE_H2 | Ridge | H2 | 105 | 424 | 17.1834 |
| E1_RIDGE_RICH | Ridge | none | 104 | 420 | 17.2221 |
| E1_RIDGE_PRIOR | Ridge | H1+H2+H3 | 107 | 432 | 17.3428 |
| E1_MLP_RICH | MLP | none | 104 | 18,660 | 17.3521 |

MLP+prior 各气体选择：

| Gas | Hidden | Alpha | Validation RMSE | Per-head parameters |
|---|---|---:|---:|---:|
| Ethanol | 32,16 | 1.0 | 5.8011 | 4,001 |
| CO | 64 | 1.0 | 25.3493 | 6,977 |
| Ethylene | 64 | 1.0 | 7.8456 | 6,977 |
| Methane | 16 | 1.0 | 6.9838 | 1,745 |

## 4. Test generalization metrics

| Variant | S_ALL RMSE | MAE | NRMSE | S_CC RMSE | CO RMSE | CO-high RMSE | Params | Dim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge rich | 25.8985 | 10.9809 | 0.19684 | 14.2019 | 27.5201 | 40.4410 | 420 | 104 |
| Ridge + H1 | **25.6490** | 9.3837 | 0.20429 | 11.3416 | 22.6650 | 35.0212 | 424 | 105 |
| Ridge + H2 | 25.9256 | 10.5768 | **0.19670** | 13.7034 | 27.8190 | 41.5098 | 424 | 105 |
| Ridge + H3 | 26.8699 | 9.9083 | 0.21498 | 11.8246 | 22.8849 | 34.5919 | 424 | 105 |
| Ridge + H1/H2/H3 | 26.0250 | 9.4860 | 0.20729 | **11.3129** | 22.9142 | 35.6278 | 432 | 107 |
| MLP rich | 28.1175 | 9.8123 | 0.22570 | 12.0645 | 23.9952 | 37.0663 | 18,660 | 104 |
| MLP + H1/H2/H3 | 27.2754 | **9.2783** | 0.22014 | 11.5891 | **22.3600** | **34.2755** | 19,700 | 107 |

Test S_CC 使用1333个B5分类正确样本；S_ALL使用全部1360行。

## 5. E1 factorial effects

正值表示 intervention 更好：

| Effect | Calibration improvement | Test improvement |
|---|---:|---:|
| Source prior gain for Ridge | −0.701% | −0.488% |
| Source prior gain for MLP | **+19.186%** | **+2.995%** |
| MLP vs Ridge without prior | −0.755% | −8.568% |
| MLP vs Ridge with prior | **+19.143%** | **−4.805%** |

解释：

- H1/H2/H3 对 MLP 有一致正增益：test四种气体相对 MLP-rich 均改善，Ethanol/CO/Ethylene/Methane 分别改善1.80%/6.81%/2.67%/1.44%。
- 但 MLP 本身的跨split泛化弱于 Ridge。MLP+prior 虽在MAE、CO、CO-high上优于 Ridge+prior，S_ALL RMSE、NRMSE和S_CC RMSE均更差。
- MLP+prior 参数量是 Ridge+prior 的约45.6倍，320行 calibration 下存在明显的小样本复杂度风险。

## 6. E2 component ablation

相对 Ridge rich-only：

| Component | Calibration improvement | Test improvement | 结论 |
|---|---:|---:|---|
| H1 only | **+10.613%** | **+0.963%** | 最稳定的单component |
| H2 only | +0.225% | −0.105% | 基本中性 |
| H3 only | +7.270% | −3.751% | calibration收益未泛化 |
| H1+H2+H3 | −0.701% | −0.488% | 全部加入反而略差 |

H1 对 CO/CO-high 和正确路由子集最有价值，但会使 Ethylene test RMSE 从30.788升至35.505。H3也改善 CO，却同样损害 Ethylene。三个预测同时进入 Ridge 后没有叠加收益，符合冗余、共线性或小样本系数不稳定的解释。

## 7. 分气体 test RMSE

| Variant | Ethanol | CO | Ethylene | Methane |
|---|---:|---:|---:|---:|
| Ridge rich | 27.8052 | 27.5201 | 30.7880 | 14.3018 |
| Ridge + H1 | 26.0658 | 22.6650 | 35.5052 | 13.3317 |
| Ridge + H2 | 27.7062 | 27.8190 | 30.7882 | 14.1103 |
| Ridge + H3 | 29.7170 | 22.8849 | 35.6901 | 14.4003 |
| Ridge all prior | 27.2128 | 22.9142 | 35.4527 | 13.6641 |
| MLP rich | 34.7586 | 23.9952 | 34.2956 | 14.2219 |
| MLP all prior | 34.1333 | **22.3600** | **33.3805** | 14.0176 |

MLP+prior 相对 Ridge+prior 在 CO/Ethylene 改善2.42%/5.85%，但 Ethanol/Methane 退化25.43%/2.59%，最终导致S_ALL RMSE更差。

## 8. Evidence boundary 与停止状态

- Candidate 标签只由预注册 calibration gate 产生。
- Test generalization 不用于改选，但明确显示 candidate 尚不稳定。
- 单seed、单个C5 client、80行validation不足以支持稳定性或显著性声明。
- 不自动修改runtime，不创建runtime v5。
- 本阶段已停止；未启动Pi、multi-seed、low-calibration或新的联邦回归实验。
