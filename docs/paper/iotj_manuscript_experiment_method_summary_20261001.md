# GAPS 论文实验、方法与系统流程汇总（2026-10-01）

> 用途：论文撰写、数字核对和新对话交接。
>
> 当前分支：`codex/iotj-final-classification-le1`；本文整理时本地与远端基线均为 `b05076a78feaf204af70de7d35e86c06360014af`。
>
> 本文只汇总既有冻结证据，不训练、不推断、不重选模型、不修改论文正文。主数值来自 `docs/paper/iscas2027_commissioning_20260921/UNIFIED_MAIN_TABLES.md`，其状态为 `PASS_NUMERIC_CONSOLIDATION_WITH_DEPLOYMENT_VERSION_LIMITATION`，已完成 148 项数值核对和 19 项来源完整性核对。

---

## 1. 论文当前最稳妥的主线

GAPS 应被描述为一个面向跨设备电子鼻的 **source federated learning → target commissioning → gas routing → quantitative regression → quality control → edge deployment** 生命周期框架，而不是单纯依靠分类准确率取胜的单一算法。

当前证据支持以下层次化表述：

1. Source-only FL 在设备偏移较大的 C5 上明显不足；使用目标节点 calibration 信息进行 commissioning 后，A0T 和 GAPS/A4 均可显著恢复分类性能。
2. 在相同目标标签预算下，GAPS/A4 相对 A0T 的分类增益很小：pooled Accuracy 仅高 0.074 个百分点，C5 Accuracy 两者相同。因此不能把核心贡献写成“大幅超越 target-CE 适配”。
3. matched M0 消融表明语义知识组织在固定 C5、seed42 条件下存在额外收益，但增量只有 Accuracy +0.0735 个百分点、Macro-F1 +0.0707 个百分点；这是一项限定性机制证据。
4. 定量回归是当前更有价值的 commissioning 证据：pooled source-only regressor 到 R84 的 RMSE 从 66.947 ppm 降到 13.991 ppm。不过该差异同时包含目标回归标定和模型变化，不能全部归因于 1D source prior。
5. R84 相对 R83 的净增益为 pooled RMSE −0.793 ppm（5.37%），但具有明显设备和气体依赖性；C5 的 NRMSE 反而略差，且区间跨零。因此正式结论是 `RETAIN_R84_DEVICE_DEPENDENT`，不是 universal transfer-safe gain。
6. confidence-only QC 能以一定 reject rate 换取更低的 accepted-set error，但 coverage 是“保留输出比例”，不是预测区间 coverage。
7. strict raw-file/exposure-disjoint sensitivity 下 C5 Accuracy 降至 74.762%、R84 NRMSE 升至 0.3489。这限制了任何“稳健泛化到完全未见 exposure”的强表述。

---

## 2. 数据、角色与冻结实验配置

### 2.1 Canonical-v1 数据合同

| 项目 | 冻结设置 |
|---|---|
| 数据根 | `dataset/iotj_canonical_v1/` |
| 预处理 ID | `HZ5_MEAN_W10S` |
| 时间处理 | 稳定排序；重复时间戳取均值 |
| 基线 | 原始观测 20–50 s 的均值电导 G0 |
| 重采样 | 5 Hz；0.2 s physical bin 内取均值 |
| 缺口处理 | 只允许补一个 bin；不连续填补长缺口 |
| 时间范围 | 60–170 s |
| 窗口 | 10 s，步长 5 s |
| 分类输入 | 50×8 时序张量 |
| source clients | C1、C2 |
| target clients | C3、C4、C5 |
| seed | 42 |
| 数据 aggregate SHA256 | `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6` |

### 2.2 样本角色

| Client | Source train | Source calibration | Source test | Target calibration | Target sealed test |
|---|---:|---:|---:|---:|---:|
| C1 | 2360 | 320 | 680 | — | — |
| C2 | 2360 | 320 | 680 | — | — |
| C3 | — | — | — | 678 | 2677 |
| C4 | — | — | — | 320 | 1360 |
| C5 | — | — | — | 320 | 1360 |

目标 test 在训练、适配、阈值选择、超参数选择和 checkpoint 选择阶段不可访问；只在所有固定端点完成后用于一次性正式评价。

### 2.3 分类主协议

| 配置 | 值 |
|---|---|
| backbone | TCN + self-attention classifier |
| source FL rounds | 25 |
| local epochs / round | 1 |
| batch size | 32 |
| seed | 42 |
| FedAvg/FedProx/GAPS optimizer | Adam |
| client learning rate | 5×10⁻⁴ |
| SCAFFOLD optimizer | canonical SGD control-variate update |
| SCAFFOLD learning rate | 5×10⁻⁴，预注册单点，不搜索 |
| FedProx μ | 0.01 |
| checkpoint | fixed round 25 |
| GAPS client profile | `ce_stats` |
| GAPS aggregation | 普通 FedAvg；`use_selective_agg=false` |
| server adaptation | 每轮 100 steps，Adam 5×10⁻⁴ |
| adapted checkpoint | `use_adapted_as_global=true` |
| hyperparameter search | 无 |
| historical checkpoint reuse | 无；各 target 从 seed42 重新初始化 |

### 2.4 各方法使用的目标信息

| 方法 | Target calibration x | class | phase | concentration | Target CE |
|---|---:|---:|---:|---:|---:|
| FedAvg | 否 | 否 | 否 | 否 | 否 |
| FedProx | 否 | 否 | 否 | 否 | 否 |
| SCAFFOLD | 否 | 否 | 否 | 否 | 否 |
| canonical MMD | 是 | 否 | 否 | 否 | 否 |
| A0T | 是 | 是 | 否 | 否 | 是 |
| GAPS/A4 | 是 | 是 | 是 | 否 | 否，权重 0 |

因此主表是不同 commissioning 信息条件下的算法级参照，不是完全等监督、等优化器的单因素排名。

---

## 3. 分类实验结果

### 3.1 Canonical 分类主表

Accuracy 和 Macro-F1 均为百分数；pooled 指先合并 C3/C4/C5 逐窗口预测再计算，不是三个设备指标的算术平均。

| 信息条件 | 方法 | C3 Acc./F1 | C4 Acc./F1 | C5 Acc./F1 | Pooled Acc./F1 |
|---|---|---:|---:|---:|---:|
| Source-only FL | FedAvg | 93.986 / 93.924 | 92.353 / 92.115 | 47.868 / 36.861 | 81.953 / 81.873 |
| Source-only FL | FedProx | 95.928 / 95.906 | 91.838 / 91.734 | 50.294 / 39.308 | 83.398 / 83.372 |
| Source-only FL | SCAFFOLD | 73.366 / 64.936 | 73.529 / 63.521 | 62.132 / 71.561 | 70.576 / 61.821 |
| Unlabeled DA | MMD | 98.282 / 98.279 | 97.500 / 97.474 | 56.324 / 56.411 | 87.512 / 87.589 |
| Label-aware | A0T | 99.851 / 99.851 | 99.485 / 99.486 | 99.412 / 99.414 | 99.648 / 99.648 |
| Label/phase-aware | GAPS/A4 | 99.851 / 99.851 | 99.779 / 99.779 | 99.412 / 99.413 | 99.722 / 99.722 |

写作解释：

- C5 是最明显的跨设备困难目标：FedAvg Accuracy 47.868%，MMD 56.324%，而有标签 commissioning 后达到 99.4%。
- A4 相对 A0T 的 pooled Accuracy/Macro-F1 增量均约 0.074 个百分点；C5 基本无差异，增量主要来自 C4。
- SCAFFOLD 使用 canonical SGD-style control variates，而 FedAvg、FedProx、GAPS 使用冻结的 Adam。该表代表标准算法级基线，不是 optimizer-controlled 单因素消融。
- CORAL/DANN 的既有 C3/C4 population 为 680/320，不是 canonical 2677/1360，故未拼入 canonical 主表。

### 3.2 Matched M0 语义消融（补充机制证据）

该结果属于固定 C5、seed42 的 matched semantic ablation，不应替代 canonical 主表，也不能扩展成多设备或统计显著性结论。

| 方法 | Accuracy | Macro-F1 | Error rate |
|---|---:|---:|---:|
| M0：matched target-aware FL without semantic knowledge | 98.3824% | 98.3835% | 1.6176% |
| Full GAPS | 98.4559% | 98.4543% | 1.5441% |
| GAPS − M0 | +0.0735 pp | +0.0707 pp | — |

相对分类错误率下降 4.545%。可写成：“在冻结的 C5/seed42 matched 设置中，语义机制提供了小幅额外收益。”不可写成“显著、稳定或跨设备普遍提升”。

### 3.3 C5 有标签/类条件 DA 诊断（legacy supplementary）

该研究使用 legacy P0A round-25 source checkpoint、C5 calibration/test=320/1360、100 adaptation steps、Adam 5×10⁻⁴；它是 mechanism diagnostic，不是 canonical-v1 主表的一部分。

| 方法 | 固定目标函数 | Accuracy | Macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|
| CORAL+TCE | source CE + 0.5 global CORAL + 1.0 target CE | 96.838% | 96.840% | 0.1437 | 0.0181 |
| MMD+TCE | source CE + 0.5 global MMD² + 1.0 target CE | 96.471% | 96.480% | 0.1744 | 0.0192 |
| DANN+TCE | source CE + 0.5 global domain BCE/GRL + 1.0 target CE | 96.765% | 96.773% | 0.1556 | 0.0171 |
| CC-CORAL | source CE + 0.5 true-class conditional CORAL | 38.382% | 27.544% | 4.6570 | 0.6022 |
| Class-MMD | source CE + 0.5 true-class conditional MMD² | 88.382% | 88.514% | 0.5188 | 0.0673 |
| CC-DANN | source CE + 0.5 true-class conditional DANN | 35.735% | 29.180% | 4.3158 | 0.6088 |

该诊断说明：target CE 是主要增益来源；在固定权重下，额外 global alignment 没有超过 Simple Target-CE。仅用真实类别做条件分组而不加 target CE 时，Class-MMD 有一定效果，CC-CORAL/CC-DANN 不稳定。

#### 3.3.1 C3/C4 +TCE 补充（2026-10-08）

新增六组沿用上述legacy P0A source checkpoint和100-step配置；各组独立重载source模型。
C3 calibration/test=320/680；C4=160/320；C5旧结果直接复用。
下表Accuracy/Macro-F1为百分数，C3/C4为recomputed，C5为reported_read_only。

| Method | C3 Acc./F1 | C4 Acc./F1 | C5 Acc./F1（已有） |
|---|---:|---:|---:|
| CORAL+TCE | 99.5588 / 99.5584 | 99.0625 / 99.0624 | 96.8382 / 96.8400 |
| MMD+TCE | 99.4118 / 99.4118 | 99.3750 / 99.3769 | 96.4706 / 96.4800 |
| DANN+TCE | 99.2647 / 99.2625 | 99.3750 / 99.3769 | 96.7647 / 96.7730 |

六组均完成100steps，15项既有实现测试通过，3000条新预测重算指标与CSV完全一致，
33个执行资产hash与全部输入hash检查通过。NLL/ECE、source F1、耗时及逐样本预测保留在：

```text
results/c34_tce_da_extension_20261008/c345_tce_comparison.csv
results/c34_tce_da_extension_20261008/tce_vs_xonly_deltas.csv
docs/experiments/c34_tce_da_extension_20261008/RESULT_ANALYSIS.md
```

这组补充进一步表明C3/C4也能通过普通目标标签监督达到99%以上。
相比旧GAPS，C3三项Accuracy均更高，C4 MMD/DANN+TCE更高，C5仍是GAPS更高。
比较仅作legacy单seed上下文解释；不同适配时机/总训练预算不能解释为纯算法机制增量。
论文不得据此宣称GAPS在所有target统一优于有标签DA，且不可与canonical主表混用。

### 3.4 Canonical A4 逐轮准确率与稳定性

该结果对 C3/C4/C5 各 25 个既有 `server_round_XXX_adapted.pth` 做统一事后评价，共 75 个 target-round 端点和 134,925 条逐窗口预测。评价使用 canonical-v1 test population（C3/C4/C5 分别为 2677/1360/1360）、CPU、batch size 128、4 threads、logits argmax。全过程未重新训练、未调参、未选择 checkpoint，也未执行 early stopping；正式端点始终是预先冻结的 Round 25。

| Target | 首次 Accuracy≥99% | 从该轮起持续≥99% | Round 25 Accuracy | Round 21–25 mean | min–max | range |
|---|---:|---:|---:|---:|---:|---:|
| C3 | Round 6 | Round 6 | 99.851% | 99.828% | 99.813–99.851% | 0.037 pp |
| C4 | Round 1 | Round 9 | 99.779% | 99.618% | 99.412–99.779% | 0.368 pp |
| C5 | Round 7 | Round 16 | 99.412% | 99.279% | 99.118–99.412% | 0.294 pp |

结果表明：C3 最早形成持续稳定的 ≥99% 性能；C4 虽然 Round 1 已短暂超过 99%，但到 Round 9 才持续保持；最困难的 C5 到 Round 16 才进入持续 ≥99% 区间。三个目标在最后五轮均维持较高准确率，波动范围分别为 0.037、0.368 和 0.294 个百分点。

论文中可将该图描述为 **retrospective fixed-checkpoint trajectory** 或“固定 25 轮协议的事后训练动态可视化”。不能将“首次/持续达到 99%”解释为经过验证的提前停止轮次，因为同一 sealed target test 被用于事后逐轮描述，轮次也不是相互独立的统计重复；该单 seed 曲线不提供置信区间或显著性检验。

结果与图文件：

```text
results/canonical_round_curve_20260928/round_accuracy.csv
results/canonical_round_curve_20260928/round_predictions.csv
results/canonical_round_curve_20260928/round_summary.csv
results/canonical_round_curve_20260928/canonical_accuracy_vs_round_paper.png
results/canonical_round_curve_20260928/canonical_accuracy_vs_round_paper.pdf
results/canonical_round_curve_20260928/canonical_accuracy_vs_round_paper.svg
results/canonical_round_curve_20260928/canonical_accuracy_vs_round_grayscale.png
results/canonical_round_curve_20260928/evaluation_audit.json
results/canonical_round_curve_20260928/preflight.json
```

---

## 4. 回归实验结果

### 4.1 三种回归器的含义

- `SOURCE_ONLY_FEDRIDGE`：只使用 C1/C2 构建的 per-gas source regressor，不进行目标浓度标定；但分类路由仍使用冻结 A4，所以“source-only”只描述回归器。
- `R83`：使用目标 calibration concentration 拟合 83D canonical sensor-statistics Ridge。
- `R84`：在相同 83D 特征上拼接 1D 冻结 source-prior prediction，再使用相同目标 calibration 拟合 Ridge。
- 所有主表行使用实际分类预测路由 `S_ALL`；S_CC/Oracle 只用于诊断，不进入主结果。

Ridge 目标函数为：

```text
min_{β,b} Σ_i (y_i - x_i^T β - b)^2 + α ||β||²
```

intercept 不正则化。R1 固定 alpha grid 为 `[0, 0.01, 0.1, 1, 10, 100, 1000]`，只在 target calibration 上做 deterministic five-fold raw-filename-grouped CV，以 pooled RMSE 选 alpha；并列时取 grid 中第一个值。target test 不参与 alpha 选择。

### 4.2 正式 S_ALL 结果

| Target | Method | N | RMSE ppm | MAE ppm | R² | NRMSE_range |
|---|---|---:|---:|---:|---:|---:|
| C3 | Source-only FedRidge | 2677 | 56.532 | 41.199 | 0.275137 | 0.325733 |
| C3 | R83 | 2677 | 10.795 | 6.017 | 0.973571 | 0.057853 |
| C3 | R84 | 2677 | 9.445 | 4.950 | 0.979767 | 0.052918 |
| C4 | Source-only FedRidge | 1360 | 88.542 | 68.537 | -0.780023 | 0.466671 |
| C4 | R83 | 1360 | 13.544 | 6.450 | 0.958347 | 0.068042 |
| C4 | R84 | 1360 | 13.281 | 6.006 | 0.959952 | 0.065844 |
| C5 | Source-only FedRidge | 1360 | 60.462 | 45.225 | 0.169985 | 0.445418 |
| C5 | R83 | 1360 | 21.320 | 10.787 | 0.896791 | 0.099898 |
| C5 | R84 | 1360 | 20.611 | 10.457 | 0.903544 | 0.101923 |
| Pooled | Source-only FedRidge | 5397 | 66.947 | 49.103 | -0.017102 | 0.396865 |
| Pooled | R83 | 5397 | 14.784 | 7.328 | 0.950399 | 0.073086 |
| Pooled | R84 | 5397 | 13.991 | 6.604 | 0.955579 | 0.071409 |

Source-only → R84 的 RMSE 相对下降为 C3 83.29%、C4 85.00%、C5 65.91%、pooled 79.10%。这反映 target regression commissioning 的总体价值，不等于 1D source prior 的独立贡献。

### 4.3 R84 相对 R83 的增量

`Δ = R84 − R83`，误差指标负值代表改善。CI 为 5000 次 paired `target|raw_filename` grouped bootstrap，seed42，不是跨 seed CI。

| Target | ΔRMSE ppm | RMSE 改善 | 95% CI | ΔNRMSE | NRMSE 改善 | 95% CI |
|---|---:|---:|---|---:|---:|---|
| C3 | -1.350 | 12.50% | [-1.956, -0.849] | -0.004935 | 8.53% | [-0.007182, -0.002999] |
| C4 | -0.263 | 1.95% | [-0.397, -0.122] | -0.002198 | 3.23% | [-0.004126, -0.000735] |
| C5 | -0.709 | 3.33% | [-1.862, +0.676] | +0.002026 | -2.03% | [-0.006268, +0.012784] |
| Pooled | -0.793 | 5.37% | [-1.241, -0.317] | -0.001677 | 2.29% | [-0.004652, +0.001907] |

### 4.4 分气体 RMSE

| Target | Gas | R83 | R84 | ΔRMSE ppm |
|---|---|---:|---:|---:|
| C3 | Ethanol | 5.398 | 5.476 | +0.078 |
| C3 | CO | 19.007 | 15.973 | -3.033 |
| C3 | Ethylene | 6.430 | 6.326 | -0.104 |
| C3 | Methane | 5.955 | 5.693 | -0.262 |
| C4 | Ethanol | 7.032 | 7.051 | +0.019 |
| C4 | CO | 23.986 | 23.754 | -0.232 |
| C4 | Ethylene | 4.296 | 2.783 | -1.513 |
| C4 | Methane | 9.517 | 9.155 | -0.362 |
| C5 | Ethanol | 6.908 | 10.618 | +3.710 |
| C5 | CO | 24.945 | 25.280 | +0.335 |
| C5 | Ethylene | 4.452 | 4.695 | +0.243 |
| C5 | Methane | 33.593 | 30.421 | -3.172 |
| Pooled | Ethanol | 6.239 | 7.470 | +1.230 |
| Pooled | CO | 21.935 | 20.738 | -1.197 |
| Pooled | Ethylene | 5.492 | 5.231 | -0.261 |
| Pooled | Methane | 18.017 | 16.440 | -1.577 |

R84 对 C3 CO、C4 Ethylene、C5 Methane 帮助最明显，但使 C5 Ethanol 明显恶化。这正是“device/gas dependent”的直接依据。

### 4.5 C5 calibration budget

| Budget | N_cal | R83 RMSE / NRMSE | R84 RMSE / NRMSE | ΔNRMSE | 95% CI |
|---|---:|---:|---:|---:|---|
| 5% | 80 | 28.533 / 0.141353 | 28.236 / 0.140169 | -0.001184 | [-0.001810, -0.000650] |
| 10% | 160 | 23.532 / 0.117883 | 22.884 / 0.116044 | -0.001839 | [-0.006896, +0.002288] |
| 15% | 240 | 27.252 / 0.129120 | 26.795 / 0.128114 | -0.001006 | [-0.003815, +0.001584] |
| 20% | 320 | 21.320 / 0.099898 | 20.611 / 0.101923 | +0.002026 | [-0.006320, +0.012806] |

预算曲线非单调；5/10/15% 是单个冻结 nested subset，20% 复用 sealed R1。不能据此宣称误差随标注比例严格单调，也不能把 window 数直接等同于独立 exposure 数。

---

## 5. QC 与部署结果

### 5.1 Final confidence-only QC

风险分数：

```text
r_cls(x) = 1 - max_g p(g | x)
```

各 target 的 HC90/HC95 阈值仅由 calibration 锁定；test 只做固定策略评价。

| Target | Workpoint | Accept / Reject | Actual coverage | Accepted RMSE ppm | Accepted MAE ppm | Accepted NRMSE |
|---|---|---:|---:|---:|---:|---:|
| C3 | HC90 | 2517 / 160 | 94.02% | 6.762 | 4.513 | 0.044052 |
| C3 | HC95 | 2573 / 104 | 96.12% | 6.843 | 4.545 | 0.044049 |
| C4 | HC90 | 1264 / 96 | 92.94% | 11.242 | 5.582 | 0.053981 |
| C4 | HC95 | 1302 / 58 | 95.74% | 11.111 | 5.501 | 0.053406 |
| C5 | HC90 | 1226 / 134 | 90.15% | 14.282 | 8.670 | 0.076482 |
| C5 | HC95 | 1270 / 90 | 93.38% | 14.678 | 8.981 | 0.077547 |
| Pooled | HC90 | 5007 / 390 | 92.77% | 10.239 | 5.801 | 0.056066 |
| Pooled | HC95 | 5145 / 252 | 95.33% | 10.384 | 5.882 | 0.056361 |

100% 全保留时 pooled R84 NRMSE 为 0.071409；HC90/HC95 的 accepted-set NRMSE 约为 0.056。该结果支持 selective reporting/人工复核价值，但不代表被拒绝样本已经被正确修复，也不等于 interval coverage。

### 5.2 Raspberry Pi 5 现有基准

| Component | Mean ms/window | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| preprocessing | 0.013062 | 0.012907 | 0.014204 | 0.014722 |
| classifier | 2.496779 | 2.441152 | 2.483004 | 4.217013 |
| R84 | 0.617340 | 0.613876 | 0.641897 | 0.653063 |
| QC | 0.077749 | 0.077037 | 0.079779 | 0.101095 |
| total pipeline | 3.204929 | 3.148741 | 3.192528 | 4.924147 |

补充统计：分类器 22,765 个标量参数，FP32 参数载荷 91,060 bytes；10,000 windows、batch=1、4 threads 下吞吐率 295.929 windows/s，peak RSS 271,499,264 bytes（258.92 MiB）。25 轮、2 client、双向纯参数理论载荷为 9,106,000 bytes。

版本限制：被测 runtime package SHA256 为 `52328c9cd9f8c9d9eba2f700a35f20f488070df2919fba6fa94e8a77a5dc1c31`，runtime commit 为 `f3d15775f3c3372dc1589989746d864fa0c332d3`；尚未证明它与后来 sealed R1/confidence-only QC 全部资产版本完全等价。因此只能写成“现有被测包的性能”，不能写成“最新最终系统精确端到端性能”。

---

## 6. 客户端与服务器损失函数

### 6.1 Canonical A4 客户端本地目标

canonical A4 使用 `ce_stats` profile。客户端真正用于优化的目标只有 source classification CE：

```text
L_client^A4 = L_CE^source
            = -(1/B) Σ_i log p_θ(y_i | x_i)
```

客户端设置：Adam，lr=5×10⁻⁴，local_epochs=1，batch_size=32，gradient clipping max-norm=5。回归损失、客户端 semantic alignment 和 replay distillation 均禁用；客户端仍上传 class/phase prototype statistics，供服务器构造语义原型与条件对齐。

这一点必须与旧 Full-GAPS 区分：历史 `proto_replay` profile 可以包含客户端语义对齐和 replay distillation，但它不是 canonical-v1 A4 主表的本地损失。

### 6.2 FedAvg、FedProx、SCAFFOLD

FedAvg 本地目标：

```text
L_FedAvg,k = L_CE,k
```

FedProx 本地目标：

```text
L_FedProx,k = L_CE,k + (μ/2) ||θ_k - θ_global||²,  μ=0.01
```

SCAFFOLD 使用 SGD-style control-variate 更新，而不是“Adam 加一个修正损失”：

```text
θ_k ← θ_k - η [∇L_CE,k(θ_k) + c - c_k]
```

其中 server control variate `c` 和 per-client `c_k` 跨轮持久化，并通过 client model delta/control-variate delta 更新；η=5×10⁻⁴。它与 Adam 5×10⁻⁴ 不应解释为优化器等价公平值。

### 6.3 A0T 服务器适配目标

A0T 在每轮 source FL 聚合后执行 100 个固定 adaptation steps：

```text
L_A0T = L_CE^source + 1.0 L_CE^target
```

目标 CE 只使用 target calibration class label；不访问 target test，不使用 target concentration，不使用 A4 的 class/phase/prototype alignment 项。服务器优化器为 Adam，lr=5×10⁻⁴。

### 6.4 Canonical A4 服务器适配目标

每轮聚合后，服务器在合并的 C1/C2 source validation/calibration batch 与对应 target calibration 上执行 100 steps：

```text
L_A4 = L_CE^source
     + 0.5 L_CORAL^class
     + 0.5 L_MMD^global
     + 0.5 L_MMD^class
     + 0.5 L_adv^Wasserstein-min
     + 0.3 L_proto-anchor
     + 0.05 L_proto-fit
     + 2.0 L_consistency
     + 0.2 L_proto-MMD
     + 0.2 L_stage-MMD
```

其中：

- `L_CORAL^class`：按 target calibration class 条件分组的 covariance alignment；
- `L_MMD^global`：无条件 source-target global MMD²；
- `L_MMD^class`：同类 source-target MMD²；
- `L_adv`：Wasserstein-min feature objective；domain discriminator 使用独立 Adam，默认 lr=10⁻³；
- `L_proto-anchor`、`L_proto-fit`、`L_consistency`、`L_proto-MMD`：服务器语义原型相关项；
- `L_stage-MMD`：`cross_domain_same_class_phase` 对齐；
- 主模型/原型优化器为 Adam，lr=5×10⁻⁴，gradient clipping max-norm=5；
- target CE 权重固定为 0；不存在用 target classification CE 直接训练 A4 的情况；
- target concentration 对分类完全不可见。

实际 C5 round25 的 100 steps 中，source CE、CORAL、global/class MMD、adversarial、prototype anchor/fit/consistency/proto-MMD 和 stage-MMD 均 active。以下两项虽然在配置中保留权重，但未激活：

| 项 | 配置权重 | 实际状态 | 原因 |
|---|---:|---|---|
| target CE | 0.0 | inactive | configured weight zero |
| device residual | 0.1 | inactive | client residual unavailable |
| legacy align regularizer | 0.05 | inactive | module disabled |

不同损失的原始数值尺度差异很大，不能通过比较 raw loss 大小直接判断模块重要性；损失权重为冻结协议值，没有根据 target test 调整。

### 6.5 Canonical x-only MMD 基线

```text
L_MMD-baseline = L_CE^source + 0.5 L_MMD²_global
```

它从同一 canonical FedAvg round25 checkpoint 独立开始，使用 target calibration x only，100 steps、Adam 5×10⁻⁴、固定 step100；不使用 class、phase、concentration、pseudo label 或 target CE。

---

## 7. 整体系统运行流程

```text
原始约100 Hz公共电子鼻数据
  ↓ 稳定时间排序 / 重复均值 / G0 / 5 Hz physical bins
canonical-v1 50×8 windows + immutable role manifests + SHA256
  ↓
C1、C2 source clients：每轮接收 global model
  ↓ local CE, Adam 5e-4, 1 local epoch
上传 model update + class/phase statistics
  ↓
Flower server：按样本数 FedAvg 聚合
  ↓
服务器 target commissioning：C3/C4/C5 calibration，100-step A4 DA
  ↓ adapted state 成为下一轮 global；共25轮
fixed round25 classifier/router
  ↓
从同一 canonical windows 重算83D传感统计特征
  ↓
R83：target-calibrated Ridge
R84：83D + frozen 1D source-prior → target-calibrated Ridge
  ↓
按预测气体类别选择 per-gas regressor，输出浓度
  ↓
confidence-only QC：ACCEPT / REVIEW(REJECT)
  ↓
序列化部署包 → Raspberry Pi 5 离线/边缘推理基准
```

### 7.1 三机 Flower 拓扑

| 角色 | SSH | 路径 | Python |
|---|---|---|---|
| Flower server + server DA | `root@121.40.139.213` | `/root/GAPS` | `/root/gaps_env/bin/python` |
| C1 Raspberry Pi | `gaps@192.168.137.172` | `/home/gaps/GAPS` | `/home/gaps/GAPS/gaps_rpi_env/bin/python` |
| C2 ECS | `root@114.55.171.63` | `/root/GAPS` | `/root/gaps_c2_cpu_env/bin/python` |
| Windows controller | 当前 worktree | orchestration/evaluation/audit | 项目 Python |

server 一般监听 `121.40.139.213:8080`；Windows runner 建立 loopback-only SSH tunnels，两个 client 通过各自 `127.0.0.1:18080` 接入。正式 runner 部署由 freeze commit 生成的 source archive，而不是默认信任远端 checkout。

连通性与专用环境检查：

```powershell
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@121.40.139.213 "date -u; hostname"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 gaps@192.168.137.172 "date -u; hostname"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@114.55.171.63 "date -u; hostname"

ssh -n root@121.40.139.213 "/root/gaps_env/bin/python -c 'import torch,flwr; print(torch.__version__,flwr.__version__)'"
ssh -n gaps@192.168.137.172 "/home/gaps/GAPS/gaps_rpi_env/bin/python -c 'import torch,flwr; print(torch.__version__,flwr.__version__)'"
ssh -n root@114.55.171.63 "/root/gaps_c2_cpu_env/bin/python -c 'import torch,flwr; print(torch.__version__,flwr.__version__)'"
```

### 7.2 主要控制器与代码

| 文件 | 功能 |
|---|---|
| `tools/build_iotj_canonical_v1.py` | 从 raw 数据构建 versioned canonical 数据集 |
| `gaps_flower/client_app.py` / `client.py` | source client 本地训练、CE/FedProx/SCAFFOLD、统计量上传 |
| `gaps_flower/server_app.py` | Flower server、轮次控制、server DA、checkpoint |
| `gaps_flower/strategy.py` | 聚合和服务器状态管理 |
| `gaps_flower/domain_adaptation.py` | A4 server-side DA 与语义/phase losses |
| `gaps_flower/canonical_uda.py` | x-only CORAL/MMD/DANN |
| `gaps_flower/canonical_quantitative_features.py` | 从 canonical 50×8 重算 83D/104D 特征 |
| `gaps_flower/canonical_fedridge_v2.py` | source sufficient-statistics FedRidge 数值等价实现 |
| `gaps_flower/canonical_r1_v1.py` | R83/R84 评价与 grouped bootstrap |
| `gaps_flower/canonical_qc_evaluation.py` | risk–coverage 和 QC 评价 |
| `gaps_deploy/canonical_v1_runtime.py` | edge runtime 主入口 |
| `scripts/run_iotj_canonical_v1_classification.py` | canonical A4 C3/C4/C5 编排 |
| `scripts/run_iotj_canonical_v1_comparators.py` | FedAvg/FedProx/SCAFFOLD/MMD 基线 |
| `scripts/run_iotj_canonical_v1_a0t.py` | equal-label A0T |
| `scripts/run_iotj_canonical_r1_v1.py` | R1 83D vs R84 |
| `scripts/run_iotj_canonical_q0_qc.py` | Q0 QC necessity |
| `scripts/run_iotj_canonical_regression_reconstruction_q1.py` | Q1 interval/QC 验证 |

这些正式端点已经完成并封存。可运行 `--help` 阅读参数，但不要在原 output root 重跑：

```powershell
python scripts/run_iotj_canonical_v1_classification.py --help
python scripts/run_iotj_canonical_v1_comparators.py --help
python scripts/run_iotj_canonical_v1_a0t.py --help
python scripts/run_iotj_canonical_r1_v1.py --help
python scripts/run_iotj_canonical_q0_qc.py --help
```

---

## 8. 论文中建议采用的结论强度

| 论断 | 当前状态 | 推荐写法 |
|---|---|---|
| GAPS 在 canonical 三目标分类上达到约 99.72% pooled Accuracy | 支持 | 可直接报告固定端点结果 |
| Round 25 前三个目标均已进入 ≥99% 区间且最后五轮较稳定 | 描述性支持 | 仅作为固定 checkpoint 的事后轨迹；不得用于 early stopping 论证 |
| GAPS 分类显著优于相同标签预算 A0T | 不支持 | A4 仅小幅领先；C5 无优势 |
| 语义知识组织有额外作用 | 限定支持 | 仅 C5/seed42 matched M0，效应很小，无显著性 |
| target commissioning 改善 quantitative sensing | 支持 | source-only regressor → target-calibrated R83/R84 的大幅下降 |
| 1D source prior 普遍改善所有设备/气体 | 不支持 | 结论必须写 device/gas dependent |
| confidence QC 可降低 accepted-set error | 支持 | 同时报告 coverage/reject rate，不称 interval coverage |
| 对完全未见 exposure 具有稳健泛化 | 不支持 | strict split 性能明显下降 |
| 已测 Pi 5 包就是最新完整 final runtime | 尚未闭环 | 必须保留 package-version limitation |

推荐总论：

> The evidence supports GAPS primarily as a lifecycle-oriented target commissioning framework that integrates federated source learning, target-aware routing, quantitative calibration, confidence-based quality control, and edge deployment. Its strongest evidence lies in the recovery of cross-device classification and quantitative sensing after target commissioning, while the incremental benefit of semantic organization and source priors is modest and device/gas dependent.

---

## 9. 必须保留的局限

1. 主实验多数为 seed42；不能宣称充分的跨 seed 稳定性。
2. canonical 主协议是 window-level 20/80 commissioning split；相邻窗口可能来自同一 raw exposure。
3. strict exposure-disjoint 测试中：C5 Accuracy 74.762%、Macro-F1 69.361%、R83 NRMSE 0.3556、R84 NRMSE 0.3489；这必须作为主要局限报告。
4. C3/C4/C5 calibration 数量并不完全相同，C3 为 678，C4/C5 为 320；它们均为各自 canonical pool 的约 20%。
5. 21D metadata 不是 21 个新传感器通道；R84 也不是 104D，而是 83D + 1D source prior。
6. `slope`、`absdiff_mean/max` 只定义为 fixed-5-Hz discrete descriptors，不具备跨采样率等价声明。
7. 公共数据模型来自约 100 Hz 原始数据经 canonical 5 Hz 处理；它不能直接作为未来实验室 1 Hz 板卡的正式模型。
8. 目前未完成实验室 1 Hz 正式模型、真实在线推理验收、新版 MCU timestamp/sequence/CRC 协议、OTA 和经正式验证的全自动 QC 决策。

---

## 10. 权威文件与结果位置

### 10.1 统一主表和最终证据

```text
docs/paper/iscas2027_commissioning_20260921/UNIFIED_MAIN_TABLES.md
docs/paper/iscas2027_commissioning_20260921/UNIFIED_MAIN_TABLES_DATA.json
docs/paper/iscas2027_commissioning_20260921/UNIFIED_MAIN_TABLES_CHECKS.json
results/iotj_v4_final_evidence_20260917/FINAL_CANONICAL_EVIDENCE_REPORT.md
results/iotj_v4_final_evidence_20260917/IOTJ_V4_EXPERIMENTS_FROZEN.json
```

### 10.2 分类

```text
results/iotj_v4_final_evidence_20260917/classification/canonical_classification_table.csv
results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C3/
results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C4/
results/iotj_canonical_v1_final_20260808/classification/CANONICAL-V1-A4-C5/
results/canonical_round_curve_20260928/
results/iotj_v4_final_experiments_20260914/M0_vs_GAPS_final_comparison.json
results/iotj_v4_final_experiments_20260914/Experiment_A_claims_evidence.md
```

### 10.3 回归和 QC

```text
results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/
results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812/
results/iotj_canonical_v1_final/canonical_q0_qc_necessity_20260812/
results/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812/
results/iotj_canonical_v1_final/current_authoritative_summary_20260812/
results/iotj_v4_final_evidence_20260917/calibration_budget/
results/iotj_v4_final_evidence_20260917/qc/
```

### 10.4 部署

```text
docs/experiments/iotj_canonical_v1_final/07_pi5_benchmark.csv
docs/experiments/iotj_canonical_v1_final/08_model_size.json
docs/experiments/iotj_canonical_v1_final/SYSTEM_CLAIM_VALIDATION.md
```

---

## 11. 写论文时的数字核对规则

- 分类 Accuracy/F1 用百分数；回归 RMSE/MAE 用 ppm；NRMSE_range 无量纲。
- pooled Macro-F1 必须从合并预测计算，不能平均 C3/C4/C5 Macro-F1。
- pooled RMSE/NRMSE 是 micro 合并样本定义，不能平均三个设备值。
- `Δ=R84−R83`；负误差差值表示改善。
- bootstrap CI 是 grouped resampling uncertainty，不是跨 seed 置信区间。
- 逐轮 Accuracy 是对同一 target test 和同一 seed 的 25 个冻结 checkpoint 的事后重复评价；轮次不是独立重复，不能据此做显著性检验或重新选择停止轮次。
- HC90/HC95 是 calibration nominal 工作点名称，不保证 test coverage 恰好为 90%/95%。
- 结果目录名含 `final` 不代表仍然 authoritative；以本文件第 10 节来源链为准。
- 旧 TABLE II、旧 9.333/13.808/18.477 ppm 回归、旧 multi-signal QC 均不可与 canonical-v1 数值拼接。

---

## 12. 一句话交接

```text
论文主线使用 canonical-v1 + fixed round25 A4 + R84 + confidence-only QC；
分类侧强调 target commissioning 恢复跨设备能力，不夸大 A4 对 A0T 的小增量；
回归侧强调目标浓度标定的总体价值，并将 source prior 写成 device/gas-dependent；
所有泛化和部署结论都保留 strict-exposure 与 package-version 限制。
```
