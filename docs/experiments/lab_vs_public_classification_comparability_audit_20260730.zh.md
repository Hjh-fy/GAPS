# 实验室自测数据与公共数据集分类准确率可比性审计

## Audit scope and intended claim

拟审计的说法是：

> 实验室 P2→P3 fold 1 adapted window Accuracy 为 91.30%，低于公共数据集约
> 98%–99%，因此当前模型或联邦系统性能发生了约 7–8 个百分点的退化。

审计目标是判断两个数字能否直接比较，以及现有证据能否定位下降原因。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Target calibration | QC | Seed | Provenance |
|---|---|---|---|---|---|---|---|---|
| `LAB3GAS-P2P3-SRCNORM-F1-S42` | concentration-group-aware fold 1；整个最低保留浓度组为 P3 test | `strong_cls`，3 类，6 通道 | source-calibration 选中的 round 2 adapted | fixed strong，100 steps/round | P3 单个相邻浓度组，138 windows/6 exposures | none | 42 | `formal_evaluation_summary.json`，postflight=`valid` |
| `PUBLIC-C1-C5-B2-S42` | C5 calibration/test 按 class×concentration 分层 20/80；test 覆盖全部 10 个浓度 | B2，4 类，8 通道 | frozen adapted | B2 | C5 320 windows，覆盖 4 类×10 浓度 | none | 42 | Accuracy=98.8971%，N=1360 |
| `PUBLIC-C45-C1-B5-S42` | C1 target test；test 覆盖全部 10 个浓度 | B5，4 类，8 通道 | frozen adapted | B5 | target calibration 覆盖全部 10 个浓度 | none | 42 | Accuracy=99.1418%，N=2680 |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| `CMP-01` | blocking | 相同研究问题与 split | 公共 test 在每类全部 10 个浓度中分层抽样；实验室 fold 1 将最低浓度整组留出 | 前者主要是已见浓度网格内插值，后者是跨板且向未见最低浓度外推 | 分开命名任务；不得用 Accuracy 差直接声称模型退化 | open |
| `CMP-02` | blocking | 相同数据与类别 | 公共数据为 Ethanol/CO/Ethylene/Methane 四类；实验室为乙醛/甲烷/乙酸三类 | 类间传感响应可分性不同；类别数相同与否不能校正这种差异 | 只在同一数据集内做模型比较 | open |
| `CMP-03` | major | 训练数据量 | 公共 C1/C2 每个源客户端约 2360 train windows、10 个浓度；实验室 P2 仅 414 train windows、18 个独立训练 exposures | 实验室覆盖更窄，估计方差更大，对板间变化更敏感 | 扩展独立测试次数/日期/板子，而不只是增加重叠窗口 | open |
| `CMP-04` | major | 目标校准覆盖 | 公共 target calibration 覆盖全部 10 个浓度；实验室 P3 calibration 只有 test 之上的一个相邻浓度组 | 公共 DA 已看到与 test 相同浓度网格，实验室 DA 必须外推 | 增加预先冻结的多浓度 target calibration 消融；不得使用 test 选方案 | open |
| `CMP-05` | major | 输入维度 | 公共数据使用 8 个传感通道；实验室当前只使用 6 个有效通道 | 可用判别信息与噪声结构不同 | 做实验室 6 通道内部的通道消融；不能用公共 8 通道数字作上限 | open |
| `CMP-06` | major | 预处理确定性 | 实验室仍使用约 1800/1200/1800 s 的名义边界，精确通气起止时间未冻结 | 过渡段、恢复拖尾和低浓度稳态污染可直接增加窗口错误 | 复核精确边界后重建并复跑 | open |
| `CMP-07` | informational | 指标层级 | 实验室 window Accuracy=91.30%，但 6 个 exposure 多数投票全部正确 | 12 个错误窗口没有导致一次完整暴露判错 | 同时报告 window 与 exposure 指标 | resolved |
| `CMP-08` | major | 统计完整性 | 实验室只有 fold 1、seed 42、P3 的 6 个 test exposures；公共比较也多为单 seed，但 test windows 更多 | 当前 7–8 pp 差没有跨折/跨 seed 不确定性 | 等 folds 1–5 完成后报告 paired fold mean/SD/CI | open |
| `CMP-09` | informational | 系统/数据完整性 | 实验室上传 SHA、样本形状、类别平衡、有限值、25/25 checkpoints 和 postflight 均通过 | 不支持“服务器数据损坏导致下降” | 保留哈希与审计链 | resolved |

## Leakage assessment

- 当前实验室 P2→P3 已修复为只用 P2 train 拟合归一化。
- 轮次只按 P2 calibration 选择；P3 test 在第 2 轮锁定后才打开。
- P3 calibration 参与服务器 DA，符合当前预声明协议；但其浓度覆盖与公共数据集不同。
- 公共数据集的 20/80 target split 是按 client、class、concentration 分层，因此
  calibration 和 test 共享相同的浓度水平。这不是同一协议内部的泄漏，但使其任务明显
  不同于“整浓度组外推”。

## Baseline, completeness, and reproducibility assessment

实验室 fold 1 的 91.3043% 与公共 C1→C5 B2 的 98.8971% 相差 7.5928 pp，
与公共 C4+C5→C1 B5 的 99.1418% 相差 7.8375 pp。这些仅是描述性差值。

现有证据不能把差值归因于：

- 联邦学习代码退化；
- source-only normalization 单独造成下降；
- P1 是否应加入；
- 模型结构不适合实验室数据。

现有证据能够支持：

- 当前实验室任务比公共数据任务包含更强的浓度外推要求；
- 实验室独立数据量、目标校准浓度覆盖和通道数都更少；
- 名义时间边界与板间域偏移是当前实验室结果特有的风险；
- 当前剩余错误集中于乙酸窗口，而不是随机遍布三类。

## Verdict: blocked

将 `91.30%` 与 `98%–99%` 直接解释成“模型性能下降约 7–8 pp”的比较被阻断。
两个数字可以并列作为不同任务的描述性结果，但不能构成公平的性能回归判断。

更合适的表述是：

> 公共数据集在覆盖全部浓度水平的分层 target test 上达到约 98%–99%；实验室数据在
> 跨板、整组最低浓度留出的更严格 fold 1 上达到 91.30% window Accuracy，同时 6/6
> exposure 均被正确识别。实验室五折与精确边界复核尚未完成。

## Unknowns and handoff

- 实验室 folds 2–5 汇总：运行中。
- 跨 seed 稳定性：unknown。
- 精确气体边界重建后的准确率：unknown。
- 同一实验室 split 上不同模型/DA 配置的公平消融：unknown。
- 若要判断模型本身是否退化，必须在同一实验室数据、相同 split、相同归一化和相同
  target calibration 下比较公共数据阶段的旧模型配置与当前配置。
