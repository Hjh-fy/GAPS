# 实验室三气体全浓度 P2→P3 准确率差距分析

## Input contract and provenance

- 分析 ID：`LAB3GAS-ALLCONC-P2P3-ACCURACY-GAP-20260730`
- 正式实验：P2 为唯一源客户端，P3 为目标客户端，25 轮、本地 3 epochs、seed 42。
- 实验室数据：`client_data_lab_3gas_allconc_timepurged_p2src_v1`。
- 公共数据对照：`PUBLIC-C1-C5-B2-S42`，仅用于方法和协议差异审查，不作为公平性能基线。
- 实验室正式结果来源：`results/lab_3gas_allconc_accuracy_gap_20260730/input/formal_evaluation_summary.json`。
- 诊断结果：`results/lab_3gas_allconc_accuracy_gap_20260730/diagnostics_v2.json`，
  SHA256=`a82694b1626718dc077fe6c0c02c8474bece4354a239942af9ab7f59710d2ffc`。
- 正式 JSON 中直接复制的数值标记为 `reported`；本报告中新计算的距离、比例和线性探针结果标记为
  `recomputed diagnostic`。
- 线性探针不加载 GAPS checkpoint，不是新的 GAPS 准确率，也不能替代正式联邦实验。

## Descriptive statistics

### 正式结果

| Metric ID | 状态 | n | 值 | Scope |
|---|---|---:|---:|---|
| `LAB-WACC` | reported | 420 windows | 0.9190 | P3 test, round-2 adapted |
| `LAB-WF1` | reported | 420 windows | 0.9191 | P3 test, round-2 adapted |
| `LAB-EACC` | reported | 30 exposures | 1.0000 | P3 test, majority vote |
| `PUBLIC-B2-WACC` | reported | 1360 windows | 0.9890 | C5 test, final round-25 adapted |

实验室 adapted 混淆矩阵为：

```text
              predicted
true          乙醛  甲烷  乙酸
乙醛           136    0     4
甲烷             2  120    18
乙酸             7    3   130
```

甲烷召回率最低，为 `120/140=85.71%`；最大单向错误是甲烷→乙酸，共 18 个窗口。
但 30 个完整暴露段全部分类正确，说明错误更像同一暴露内部的局部时间窗口问题，而不是整段气体身份不可分。

## 与公共数据方法设置的关键区别

| 因素 | 实验室全浓度实验 | 公共 B2 |
|---|---|---|
| 气体类别 | 3 | 4 |
| 输入通道 | 6 | 8 |
| 源训练量 | 420 windows / 30 exposures | 2360 windows / 160 files |
| 目标 calibration | 90 windows | 320 windows |
| 目标 test | 420 windows | 1360 windows |
| 时间范围 | 名义通气后 0–1200 s 全段 | time-aware 60–170 s 响应区 |
| phase | 单一 whole-exposure phase | early/middle/late 三阶段 |
| calibration/test | 原始时间交集为 0 | 按窗口随机 20/80 划分 |
| 原始文件共享 | 同 exposure、但时间净化 | C5 calibration/test 完全共享 80/80 个文件 |
| checkpoint | P2 calibration 饱和后选最早 round 2 | 固定 final round 25 |
| 目标测试相对 calibration 的最近邻中位距离 | 0.1078/维 | 0.0205/维 |

公共 test 到 calibration 的标准化最近邻中位距离只有实验室的约 `19%`，且 calibration/test 来自完全相同的
80 个原始文件。因此公共 98–99% 是明显更接近同文件窗口内插的任务，不能直接作为实验室跨板、
时间净化任务的模型上限。

## 主要原因定位

### 1. 最早 0–150 s 窗口与 calibration 存在时间覆盖缺口

当前 P3 calibration 的最早窗口是 base index 3，即名义通气后 150–250 s；P3 test 却包含：

- index 0：0–100 s；
- index 1：50–150 s。

这 60 个早期窗口占 test 的 `60/420=14.29%`。诊断性目标域线性探针结果如下：

| 特征 | index 0 | index 1 | 其余 12 个 test 位置 |
|---|---:|---:|---:|
| 当前相对电阻 | 33.33% | 53.33% | 全部 100% |
| 近似相对电导 | 66.67% | 90.00% | 全部 100% |

该结果不能证明 GAPS 的 34 个错误全部来自早期窗口，因为正式评估没有保存逐窗口预测流；但它提供了很强的
可证伪假设：名义气体边界、输气延迟和早期响应分布缺口是当前窗口错误的主要来源之一。

### 2. 预处理实现使用相对电阻，而原方法合同使用相对电导

当前构建代码实际计算：

```text
(R - R0) / |R0|
```

而公共方法和原始 EDA 推荐的是：

```text
G = 1 / R
(G - G0) / G0
```

当气体使电阻明显下降时，相对电阻被压缩到 `-1` 附近。P2 train 中：

- 11.89% 的逐点值低于 `-0.8`；
- 7.42% 的逐点值低于 `-0.9`。

把已保存窗口分析性近似转换为相对电导后，P3 calibration→test 线性探针由 91.90% 增至 96.90%。
该转换是在平滑之后完成，只能作为筛查证据；正式验证必须从原始 R 先转成 G，再平滑、切窗和
P2-only Z-score。

### 3. P2→P3 板间偏移大于甲烷/乙酸的类间距离

在 P2 train 标准化的响应描述空间中：

- 甲烷与乙酸在 P3 的类中心距离：0.221/维；
- 甲烷 P2→P3 类中心偏移：0.489/维；
- 乙酸 P2→P3 类中心偏移：0.260/维。

即甲烷的板间偏移已经超过甲烷与乙酸本身的间距，与正式混淆矩阵中
`甲烷→乙酸=18` 完全一致。整体 P3 相对 P2 的通道均值偏移还包括：

- CH2：`+1.09σ`；
- CH8：`+0.76σ`；
- CH1：`-0.55σ`。

### 4. CH2 在当前方向中低判别、高漂移

CH2 的 P3 类间/类内方差比只有 0.038，为六通道最低；其各类板间偏移约为
`+1.53σ/+1.35σ/+0.38σ`。在近似相对电导目标域探针中：

- 六通道：96.90%；
- 去掉 CH2：98.81%；
- 仅 CH2：55.71%。

这只是同 exposure 内的诊断筛查，不能直接声称正式 GAPS 会达到 98.81%；但 CH2 是优先级很高的预注册通道消融。

### 5. 当前 DA 配置不是公共结果所用的 corrected B2

实验室运行使用旧 `fixed_da_strong`：

- CORAL=true，权重 0.5；
- adversarial=true，权重 0.5；
- `legacy_quartic` MMD；
- `legacy_intra_domain` stage；
- proto/stage MMD 权重均为 0.2。

公共 B2 使用：

- CORAL=false；
- adversarial=false；
- `mmd2`；
- `cross_domain_same_class_phase`；
- proto/stage MMD 权重均为 0。

因此“训练代码流程一样”只在模型骨干和客户端 `proto_replay` 别名层面成立，服务端 DA 并不相同。
旧 strong 栈中还包含已知的 legacy 目标定义，不能假定比 corrected B2 更适合小样本三气体数据。

### 6. checkpoint 选择规则已经失去区分能力

P2 calibration 从 round 2 到 round 25 的 exposure/window Macro-F1 均为 1.0，规则只能机械选择最早的
round 2。公共 98.90% 使用固定 round 25。当前尚不知道实验室 round 25 在 P3 上更好还是更差；
但两者 checkpoint 策略不同，必须单独消融，不能把差距全部归因于数据。

## 可达到的提升空间：诊断性证据

| Probe | Accuracy | Macro-F1 | 解释边界 |
|---|---:|---:|---|
| P2 train→P3 test，相对电阻 | 54.76% | 45.95% | 源域直接迁移很差 |
| P3 calibration→P3 test，相对电阻 | 91.90% | 92.01% | 当前目标校准的线性可分性 |
| P2+P3 calibration→P3 test，相对电阻 | 95.00% | 94.99% | 目标校准可明显修复板间偏移 |
| P3 calibration→P3 test，近似相对电导 | 96.90% | 96.88% | 转换方向值得正式重建 |
| 上一行去掉 CH2 | 98.81% | 98.81% | 通道消融值得正式验证 |

这些探针使用同一 exposure 的不重叠时间窗口，不是独立暴露泛化；它们的作用是判断“数据是否仍有可提取信息”，
而不是生成论文性能数字。结果表明，91.90% 并非明显的数据可分性硬上限，预处理、通道和目标校准利用方式仍有提升空间。

## 建议的提升优先级

1. **先固定评价合同**：使用预声明的 round 25，修复动态样本数审计，并输出逐窗口预测流；不根据 test 选轮次。
2. **从原始 R 正确重建相对电导**：先 `G=1/R`，再基线化、平滑、切窗和 P2-only Z-score。
3. **处理早期瞬态**：先做固定 `150 s` 延迟消融；随后用只依赖原始信号的 onset 检测替代名义边界。
   如果部署必须实时输出，应报告“150 s 后开始分类”的延迟与覆盖率，而不是静默删除困难窗口。
4. **预注册去 CH2 消融**：保持所有其他参数不变；不要在 test 上逐组合搜索通道。
5. **换用 corrected B2 服务端 DA**：关闭 legacy CORAL/adversarial/stage/proto 项，使用 `mmd2`。
6. **透明使用目标标签预算**：当前 class-conditional DA 已读取 P3 calibration 标签但
   `lambda_target_ce=0`。增加一个预注册的 target-CE 对照或只微调分类头，可直接利用已经允许的目标校准资源。
7. **最终增加独立测试批次**：当前 test 已被反复查看；后续最佳组合在本数据上只能算 post-hoc 开发结果。
   冻结配置后应使用新的测试日期/重复实验确认。

## Assumptions, comparisons, effect sizes, and corrections

- 只有 seed 42，不报告跨 seed 均值、标准差或显著性。
- 420 个窗口来自 30 个 exposure，不能当作 420 个独立实验单位。
- 公共数据与实验室数据类别、板子、文件数、通道、时间协议和划分均不同；6.99 pp 差仅为描述性差值。
- 去除早期窗口会改变评价覆盖率；必须同时报告准确率、保留窗口比例、等待时间和 exposure 指标。
- 本报告不删除任何异常窗口，也未使用 target test 重新选择正式 checkpoint。

## Unknowns, conflicts, and audit handoff

- GAPS 错误按 base-window index、浓度和 v1/v2 的精确分布：`unknown`，因为没有逐窗口预测流。
- 从原始 R 正确构建相对电导后的正式 GAPS 准确率：`unknown`。
- round 25 在当前 P3 test 的 post-hoc 指标：`unknown`。
- 精确通气 onset：`unknown`，当前仍为名义 1800/1200/1800 s 边界。
- 当前运行的旧审计器期待 138/6，和全浓度数据的 90/30、420/30 冲突；需使用动态数据合同重新审计。
