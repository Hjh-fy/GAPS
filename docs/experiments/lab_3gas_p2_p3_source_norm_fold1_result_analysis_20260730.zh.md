# 实验室三气体 P2→P3 Source-only Normalization Fold 1 结果分析

## Input contract and provenance

- Experiment ID: `LAB3GAS-P2P3-SRCNORM-F1-S42-R25LE3-20260730`
- 方向：P2 为唯一源客户端，P3 为目标客户端；树莓派/P1 不参与本次训练。
- 数据集：`dataset/client_data_lab_3gas_5fold_nominal_p2src_v2`
- 数据划分：fold 1；train groups=`3,4,5`，source calibration group=`2`，target test group=`1`。
- 归一化：仅使用 P2 train 拟合 Z-score，`normalization_fit_scope=source_clients_train_only`，
  `normalization_fit_clients=[2]`。
- 训练配置：seed=42，25 个联邦轮次，本地 3 epochs，batch size=32，
  `fixed_da_strong`，每轮服务器侧 DA 100 steps。
- 源码 archive SHA256：
  `aaa6de1c9b119102bab82e1cac854edadb33956fa56ba9c735a638790ff1abba`
- 数据 manifest SHA256：
  `b4fa1d0b083a62a87653bce8e720be4eacb5b2e186c642a46cd8058b07a85561`
- 协议 manifest SHA256：
  `eb1d2a35de1f2686d0f798b02b7f826fb1d41c9e6904217834edcaacdc5ce364`
- 结果：
  `results/lab_three_gas_p2src_norm_three_node_r25le3_20260730_controller/lab3gas_nominal_p2srcnorm_v2_P2_to_P3_fold1_s42_r25le3/formal_evaluation_summary.json`
- 审计：
  `results/lab_three_gas_p2src_norm_three_node_r25le3_20260730_controller/lab3gas_nominal_p2srcnorm_v2_P2_to_P3_fold1_s42_r25le3/postflight_attempt_audit.json`
- postflight 状态：`valid`；25 个基础检查点和 25 个 adapted 检查点完整，客户端失败数为 0。
- 证据边界：`preliminary_nominal_boundary_screening`。气体精确起止时间尚未复核，不能作为最终实验室性能。

现有 JSON 中直接复制的值标记为 `reported`；本报告计算的百分点差、错误减少率和类别召回率
标记为 `recomputed`。

## 数据与部署完整性

1. 本地五折验证 `ok=true`，所有客户端每折均为 414/138/138 个
   train/calibration/test 窗口，三类各为 138/46/46 个窗口。
2. 新旧数据集的 137 个分类标签、阶段标签和窗口 manifest 文件 SHA256 完全一致；
   本次只改变归一化数值，不改变标签、暴露分组和五折划分。
3. fold 1 的 P2 train 六通道均值约为 0、标准差为 1；P3 train 均值约为
   `[-0.537, 1.211, -0.060, 0.264, 0.813, -0.369]`，证明没有使用 P3 train
   重新居中。
4. 云 A 完整数据包 SHA256 为
   `1572068ec000b5ad356b45e3a831e03197fb924ce048a3a035a6f2ac6a52d1eb`；
   云 B 的 P2-only 数据包 SHA256 为
   `c05114a665457287a6b0de9c57505458457f5bd44a571cd55a7f312c43840e1b`。
   两端上传后均通过 `sha256sum -c`。
5. 远端实际加载结果为 P2 train=414、calibration=138，P3 train=414、
   calibration=138，首个 batch shape=`[32,100,6]`。

因此，没有证据表明低准确率来自服务器文件损坏、上传错包、标签错位、样本数异常或
Flower 通信失败。旧数据的主要问题是严格实验协议不满足：Z-score 曾使用所有客户端 train
拟合，包含 P3 train 的无标签分布信息；这是 target-feature leakage，不是数据文件损坏。

## Descriptive statistics

选择规则只使用 P2 calibration：先最大化 exposure Macro-F1，再比较 window Macro-F1，
最后选择最早轮次。第 2 轮被锁定；第 2–25 轮的 P2 calibration window/exposure
Macro-F1 均为 1.0，说明源域选轮指标已经饱和。

| Metric ID | 状态 | n | 值 | Scope | Source |
|---|---|---:|---:|---|---|
| `M-P2P3-F1-BASE-WACC` | reported | 138 windows | 0.8188 | P3 test, unadapted | summary |
| `M-P2P3-F1-BASE-WF1` | reported | 138 windows | 0.8098 | P3 test, unadapted | summary |
| `M-P2P3-F1-DA-WACC` | reported | 138 windows | 0.9130 | P3 test, adapted | summary |
| `M-P2P3-F1-DA-WF1` | reported | 138 windows | 0.9113 | P3 test, adapted | summary |
| `M-P2P3-F1-BASE-EACC` | reported | 6 exposures | 1.0000 | P3 test, unadapted | summary |
| `M-P2P3-F1-DA-EACC` | reported | 6 exposures | 1.0000 | P3 test, adapted | summary |
| `M-P2P3-F1-DA-GAIN-ACC` | recomputed | same 138 windows | +9.42 pp | adapted minus unadapted | summary |
| `M-P2P3-F1-DA-GAIN-F1` | recomputed | same 138 windows | +10.15 pp | adapted minus unadapted | summary |
| `M-P2P3-F1-DA-ERROR-REDUCTION` | recomputed | 25→12 errors | 52.0% | window errors | confusion matrices |

暴露级 6/6 正确的 Clopper–Pearson 95% 区间为约 `[0.541, 1.000]`
（recomputed）。该区间只说明样本很少，且仍未处理同一板子和相邻浓度之间的相关性。
不能根据 6/6 推断总体准确率接近 100%。

## 类别错误

类别顺序为：0=乙醛，1=甲烷，2=乙酸。

| 模型 | 乙醛召回率 | 甲烷召回率 | 乙酸召回率 | 主要错误 |
|---|---:|---:|---:|---|
| unadapted | 46/46=1.000 | 40/46=0.870 | 27/46=0.587 | 乙酸→乙醛 11，乙酸→甲烷 8 |
| adapted | 46/46=1.000 | 45/46=0.978 | 35/46=0.761 | 乙酸→乙醛 9，乙酸→甲烷 2 |

DA 的主要收益不是乙醛，而是恢复甲烷并减少乙酸混淆。adapted 仍有 12 个窗口错误，
其中 11 个来自乙酸；后续应优先检查 P3 最低浓度乙酸的时序边界、稳态区间和通道响应，
而不是继续针对甲烷调参。

## 为什么之前准确率低

1. **fold 1 是最低保留浓度测试。** 模型使用较高浓度训练，在目标板 P3 上向最低浓度
   外推，信号幅度更弱，板间基线和灵敏度差异相对更突出。
2. **板间域偏移是真实存在的。** P2-only 归一化后 P3 train 并非 0/1 分布；
   这不是异常，而是严格 source-only 条件下应保留的目标域偏移。
3. **旧 P1+P2→P3 fold 1 的主要崩溃是甲烷→乙酸。** 旧 adapted 甲烷召回率仅
   6/46=0.130，40/46 个甲烷窗口被判为乙酸；本次 P2→P3 adapted 甲烷召回率为
   45/46=0.978。该变化反驳了“服务器数据整体损坏”的解释。
4. **加入 P1 可能带来负迁移，但目前不能下结论。** 旧 P1+P2→P3 与本次 P2→P3
   同时改变了源客户端集合和归一化边界，不能把 `69.57%→91.30%` 的描述性差异
   归因于任一单因素。需要在相同 source-only normalization 下完成配对五折。
5. **源域 calibration 太容易。** P2 calibration 从第 2 轮起长期 100%，无法区分
   哪个轮次更适合 P3；它能防止读取 P3 test 选轮次，却不能预测跨板最难浓度的性能。
6. **名义时间切割仍可能引入边界误差。** 精确通气起止时间未冻结，低浓度和恢复拖尾
   更容易受到切割偏差影响。

## Assumptions, comparisons, effect sizes, and corrections

- 只有 fold 1、seed 42，没有跨 fold/seed 的均值、SD 或置信区间。
- 138 个窗口来自 6 个暴露且窗口重叠，不能当作 138 个独立观测进行显著性检验。
- adapted 与 unadapted 的 +9.42 pp / +10.15 pp 是同一折的描述性配对差，不是经统计
  检验确认的总体效应。
- 旧 P1+P2→P3 与本次 P2→P3 不是单变量公平消融；不报告 p 值或因果效应量。
- 本报告没有删除异常点，也没有以本次 test 结果重新选轮次或调参。

## Proposed paper tables and figures

- 在 folds 1–5 全部完成后，汇总每折 P3 test 的 window/exposure Accuracy 与 Macro-F1，
  同时报 paired fold mean、SD 和置信区间。
- 做 `P1+P2→P3` 与 `P2→P3` 的配对折线图，保持 source-only normalization、
  seed、轮数和 DA 参数一致。
- 做三类混淆矩阵及按气体/浓度的 exposure-level 错误表，重点呈现乙酸的剩余混淆。

## Unknowns, conflicts, and audit handoff

- folds 2–5：unknown。
- 多 seed 稳定性：unknown。
- 精确气体边界复核后的性能：unknown。
- P1 是否导致负迁移：unknown，需 matched five-fold ablation。
- source-only normalization 的独立因果增益：unknown；缺少同一 P2→P3 配置的
  all-client-normalization 对照，且后者只能作为泄漏敏感性分析，不能作为正式 baseline。
- 当前结果可用于实验筛查和后续设计，不应写入冻结的 IoTJ 最终主张。
