# 实验室 A4 完整时段评估与跨板方向扩展设计

## 1. 目标

本轮工作回答两个相互独立的问题：

1. 保持现有 REC-A4-STABLE150 的训练数据、P2-only 归一化、P3 全浓度校准、
   域适配配置和第 25 轮 checkpoint 完全不变，仅把 P3 测试范围从稳定段
   360 窗口扩展到包含 0–150 s 的完整 420 窗口，测量现有 A4 模型的真实
   完整时段分类能力。
2. 在相同 A4 稳定段协议下新增 `P2→P1`、`P1→P3` 和 `P1+P2→P3`
   三个 seed-42 跨板方向，检查反向迁移、源板选择和加入第二源板后的综合收益。

本轮不改变模型结构，不研究浓度回归、分类 QC、早期窗口增强训练、等预算双源
对照或多 seed 稳定性。

## 2. 已冻结基线

- 基线方向：`P2/C2 → P3/C3`。
- 模型：`proto_replay`，三分类，输入 `[B, 100, 6]`。
- 输入：相对电阻，物理通道 `CH1/2/4/6/8/9`。
- DA：`corrected_b2`，target CE 权重 `0`。
- 训练：25 个 Flower 轮次，本地 3 epoch，batch size 32，seed 42。
- checkpoint：固定第 25 轮；source calibration 仅监控，不选轮。
- 目标域校准：每次暴露基础索引 `3/11/19`，共 90 窗口、30 次暴露，
  覆盖全部保留浓度。
- 稳定测试：360 窗口，已报告 `359/360 = 99.72%`。
- 暴露级稳定测试：`30/30 = 100%`。
- 证据边界：single-seed、nominal-boundary、post-hoc exploratory。

现有 A4 数据、checkpoint、正式 summary、postflight audit 和部署包全部只读，
新工作不得覆盖这些资产。

## 3. 时间窗口协议

每次目标气体暴露生成 23 个基础窗口，窗口长度 100 s，步长 50 s。

| 角色 | 基础窗口索引 | 每次暴露窗口数 | 30 次暴露总数 |
|---|---|---:|---:|
| calibration | `3, 11, 19` | 3 | 90 |
| purge | `2, 4, 10, 12, 18, 20` | 6 | 不参与 |
| early test | `0, 1` | 2 | 60 |
| stable test/train | `5–9, 13–17, 21–22` | 12 | 360 |
| full test | early + stable | 14 | 420 |

calibration 与 test 之间继续保持原始时间 purge。早期窗口只进入扩展评估，不进入
A4 source train、target calibration、checkpoint 选择、停止或超参数选择。

## 4. 实验矩阵

| ID | 源域训练客户端 | 目标客户端 | 归一化拟合范围 | 是否训练 | 目标 |
|---|---|---|---|---|---|
| `A4-XB-E0-FULL420` | 复用 C2/P2 | C3/P3 | 复用冻结 P2 stable train stats | 否 | 测量现有 A4 完整时段性能 |
| `A4-XB-E1-P2P1-S42` | C2/P2 | C1/P1 | 仅 P2 stable train | 是 | 检查反向跨板迁移 |
| `A4-XB-E2-P1P3-S42` | C1/P1 | C3/P3 | 仅 P1 stable train | 是 | 与既有 P2→P3 比较源板差异 |
| `A4-XB-E3-P12P3-S42` | C1/P1；C2/P2 | C3/P3 | P1+P2 stable train pooled stats | 是 | 检查加入第二源板后的综合收益 |

P1 的双戊烯暴露始终排除，仅使用乙醛、甲烷和乙酸。每个单源客户端提供
360 个 stable train 窗口和 90 个 source-calibration 窗口；双源实验使用完整
P1 与 P2 数据，因此其训练样本量约为单源的两倍。

`P1+P2→P3` 的结果只能解释为“加入第二个源板后的综合收益”，不能单独归因于
源板多样性，因为本轮不增加 matched-budget 对照。

## 5. 统一训练和评价规则

三个新训练方向保持以下变量不变：

- seed 42；
- 25 轮、本地 3 epoch；
- `proto_replay + corrected_b2`；
- 六通道相对电阻；
- target CE 权重 0；
- server DA steps 100；
- 固定第 25 轮；
- 目标板全部保留浓度参与 calibration；
- normalization 仅拟合允许的 source stable train；
- 目标 test 不参与训练、DA、选轮、停止或阈值选择。

每个方向使用同一个最终 checkpoint 输出三个互斥/嵌套范围：

1. `stable360`：主指标；
2. `early60`：0–150 s 困难区诊断；
3. `full420`：完整时段补充指标，等于 early 与 stable 的合并范围。

每个范围报告：

- window Accuracy；
- window Macro-F1；
- 每类 precision、recall、F1；
- window confusion matrix；
- exposure-level Accuracy、Macro-F1 与 confusion matrix；
- 正确数/总数，而不仅是百分比。

由于只有一个 seed，所有跨方向差异均为描述性结果，不报告显著性或确认性优越。

## 6. 假设与判读规则

### H-A4-XB-01：冻结 A4 的完整时段能力

对现有 A4 checkpoint 的 `early60` 和 `full420` 指标为未知。E0 只测量结果，
不设置成功阈值，也不重新训练。

### H-A4-XB-02：源板选择影响 P3

在完全相同的 P3 calibration/test 上，比较 E2 `P1→P3` 与既有
`P2→P3`。若预测计数或混淆结构不同，则记录源板相关差异；不得从单 seed
宣称稳定优越。

### H-A4-XB-03：加入第二源板的综合收益

在完全相同的 P3 calibration/test 上，比较 E3 `P1+P2→P3` 与既有
`P2→P3`：

- stable 不退化条件：正确数至少 `359/360`；
- full 改善条件：E3 的 full420 正确数高于 E0；
- early 改善条件：E3 的 early60 正确数高于 E0。

满足条件时只能称为 seed-42 描述性候选收益。未满足时保留完整结果，不删除或改选轮。

### H-A4-XB-04：反向迁移可行性

E1 `P2→P1` 只用于描述方向不对称性。由于目标板从 P3 变为 P1，不能与
P3-target 行做严格的模型优越排名。

## 7. 三机角色映射

| 方向 | 树莓派 C1/P1 | 云服务器 B C2/P2 | 云服务器 A |
|---|---|---|---|
| E0 | 不训练 | 不训练 | 加载冻结 checkpoint，评估 P3 |
| P2→P1 | 不训练 | 唯一 source client | Flower Server + P1 target DA/eval |
| P1→P3 | 唯一 source client | 不训练 | Flower Server + P3 target DA/eval |
| P1+P2→P3 | source client | source client | Flower Server + P3 target DA/eval |

“目标客户端”表示数据/模型语义，不要求目标板所在物理机器参加 Flower fit。
目标 calibration/test 的只读副本由云服务器 A 用于 server-side DA 和最终评估。

## 8. 最小代码修改设计

### 8.1 数据构建

新增方向参数化的数据构建入口，复用现有 session 发现、边界、窗口和保存函数。
入口必须显式接收 source client 集合与 target client，生成：

- 各 source client 的 stable train 与 source calibration；
- target client 的 calibration、stable test、early test 和 full test；
- source-only normalization；
- exposure、boundary、split、class schema 和 dataset manifests。

每个方向使用新的 dataset root；禁止 `--overwrite` 指向既有 A4 数据集。

### 8.2 三机控制器

在保持既有 `P2_to_P3` 行为不变的前提下支持：

- `P2_to_P1`；
- `P1_to_P3`；
- `P12_to_P3`。

控制器从方向映射得到 `source_clients` 和 `target_client`，动态选择：

- 需要启动的 source client；
- server target data 路径；
- server launcher 的 `--target-clients`；
- evaluator 的 `--target-client`；
- postflight 的目标客户端和目标数据目录；
- 残留进程清理与健康检查范围。

### 8.3 评价与审计

评价器和 postflight 不得固定 `target_client=3`。同一个 final checkpoint 分别读取
stable、early 和 full split，并验证：

- `stable + early = full` 的样本 ID 与计数关系；
- calibration/test 原始时间不重叠；
- normalization manifest 只包含 source train；
- selected round 精确为 25；
- 每轮 source fit/evaluate 客户端数符合方向；
- 没有 per-round target-test 文件。

## 9. 执行顺序与失败策略

1. 构建四个评价/训练所需数据视图并执行本地验证。
2. E0 在冻结 A4 checkpoint 上执行 stable/early/full 一次性评估。
3. 三端同步新代码和内容寻址数据，不覆盖已有 runtime。
4. 对 E1/E2/E3 分别执行 preflight。
5. 按 E1 → E2 → E3 顺序运行，避免三机端口和资源冲突。
6. 每个 run 完成后回收 formal summary 与 postflight audit。
7. 只有 `postflight=valid` 的第 25 轮结果进入汇总。

若某一方向失败：

- 立即停止当前精确 run-tag 的 server/client 和专用隧道；
- 保留失败日志与 attempt 身份；
- 不覆盖目录、不复用不完整 checkpoint；
- 修复后使用新 run ID 重试；
- 不阻止其他已通过 preflight 的独立方向，但正式汇总必须标明缺失行。

## 10. 测试要求

在远端运行前至少通过：

1. 三个方向的数据计数、类别平衡、浓度覆盖和 exposure ID 检查；
2. early/stable/full 集合关系检查；
3. source-only normalization 检查；
4. P1 双戊烯排除检查；
5. 动态 source/target 命令渲染检查；
6. 既有 `P2_to_P3` 控制器回归测试；
7. evaluator/postflight 的 target-client 参数化测试；
8. 三机逐方向 preflight；
9. E0 稳定段必须复现冻结 A4 的 `359/360`，否则停止全部新训练。

## 11. 输出和边界

新结果必须进入独立的日期化目录，并记录：源码 archive SHA、dataset manifest SHA、
protocol manifest SHA、运行拓扑、Python/Torch、seed、轮数、本地 epoch、模型配置、
checkpoint 和目标范围。

最终汇总包含：

- 既有 P2→P3 A4 stable 基线；
- E0 的 early/full 扩展指标；
- E1/E2/E3 的 stable/early/full 指标；
- 同目标 P3 行的描述性比较；
- P2→P1 的独立方向性结果；
- single-seed、重叠窗口、名义边界和 target-test post-hoc 限制。

本轮结果不改变以下既有结论：A4 的 `99.72%` 只表示所有保留浓度参与校准时的
稳定段分类能力，不代表完整通气过程、未见浓度外推或真实 STM32 端到端准确率。
