# GAPS 最新进度与证据交接（2026-07-15）

> 用途：新开对话、让新的 GPT/Codex 从 GitHub 接手时，先阅读本文件，再进入代码和详细实验账本。
>
> Git 分支：`codex/system-safety-hardening`。最低完整证据提交为 `02dc259`；本文件及其后续更新以所在提交为准。不要只读取默认分支后假定它代表当前实验状态。

## 1. 一句话状态

GAPS 已完成 seed-42 的真实 ECS + 树莓派 + PC 分类消融、A6/B5/B2 正式 C5 回归、FULL/HC95/HC90 QC、oracle-route 诊断以及 B2/B5 三方向六组分类验证；当前主要缺口是 seeds 43-46、低校准压力、最终部署 bundle parity 和端侧/通信开销。

## 2. 冻结主协议

- 主论文方向：C1/C2 source -> C5 target。
- C5 calibration/test：窗口级按类别和浓度分层切分，20%/80%，由导师确认；主协议不改成文件级或 repeat 级切分。
- C3/C4 不进入 C5 主目标指标；C4 只在跨方向 appendix 的 C4+C5 -> C1 中作为源设备。
- 报告训练必须使用真实阿里云 ECS、树莓派和 PC；本地运行只允许测试、审计和冻结产物评估。
- 分类训练：25 轮，客户端每轮 5 local epochs，batch size 32，Adam LR `5e-4`。
- 服务器适配：每轮 100 steps，LR `5e-4`；A5-A7/B1-B5 使用目标 calibration 标签，应称为 calibration-assisted DA。
- 回归和 QC 只用 calibration/validation 拟合或选阈值；C5 test 只用于冻结后的最终评估。

## 3. 系统闭环与状态

| 层 | 当前实现 | 状态 | 首要证据 |
|---|---|---|---|
| 数据合同 | time-aware 60-170 s 窗口、行键对齐、C5 320/1360 | 完成 | `docs/experiments/iotj_system_experiment_notebook.md` |
| 联邦分类 | Flower 客户端分类骨干、FedAvg/GAPS、服务器 calibration-assisted DA | seed-42 完成 | v2r1 与 v3 分类 summary |
| 个性化回归 | C5 Ridge/MLP、H2.3+、H8、R0-R7 阶梯 | A6/B5/B2 完成 | `formal_regression_report.md` |
| 可靠性 QC | deployment-visible risk、FULL/HC95/HC90、accept/review/reject | 完成 | `qc_operational_comparison.csv` |
| 部署安全 | fail-closed policy/assets、严格 checkpoint/schema、预测类别路由 | 代码加固完成 | `docs/superpowers/specs/2026-07-13-system-safety-hardening-design.md` |
| 真实部署性能 | Pi/PC 延迟、RSS、实际通信、正式 bundle parity | 待补 | 实验笔记本 Stage 6 |

## 4. 分类实验

### 4.1 v2r1 因果筛选：C1/C2 -> C5

| 组别 | 核心含义 | Accuracy | Macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|
| A0 | source-only FedAvg | 26.5441% | 17.9349% | 11.2816 | 0.7275 |
| A0T | 相同标签预算 target CE | 98.2353% | 98.2358% | 0.1742 | 0.0158 |
| A5 | distribution DA family | 73.0147% | 74.1456% | 1.0907 | 0.2001 |
| A6 | semantic DA family | 98.0147% | 98.0235% | 0.1384 | 0.0178 |
| A7 | legacy combined family | 98.6029% | 98.6048% | 0.1132 | 0.0118 |

完整 A0/A0T/A2/A3/A4/A4S/A5/A6/A7 表：`results/iotj_classification_ablation_20260711_v2r1_summary/classification_per_run.csv`。

### 4.2 v3 修正版：C1/C2 -> C5

| 组别 | Accuracy | Macro-F1 | NLL | ECE | 解释 |
|---|---:|---:|---:|---:|---|
| B1 | 98.7500% | 98.7534% | 0.0988 | 0.0120 | corrected CORAL 基线 |
| B2 | **99.2647%** | **99.2657%** | **0.0690** | **0.0067** | corrected global/class MMD2，seed-42 最佳 |
| B3 | 98.8971% | 98.8980% | 0.1022 | 0.0108 | B2 + stage |
| B4 | 98.9706% | 98.9714% | 0.0835 | 0.0081 | B2 + adversarial |
| B5 | 98.8971% | 98.8990% | 0.0704 | 0.0093 | B2 + CORAL + stage + adversarial |

B2 是打开 seed-42 test 排名后选出的性能候选，因此其后续回归/QC必须标为 post-screen exploratory；B5 是预声明完整修正版。B5 没有在主方向表现出简单模块叠加增益。

### 4.3 三方向 B2/B5 六运行

| 源域 -> 目标域 | 模型 | Accuracy | Macro-F1 | NLL | ECE |
|---|---|---:|---:|---:|---:|
| C1 -> C5 | B2 | **98.8971%** | **98.8954%** | **0.1008** | **0.0107** |
| C1 -> C5 | B5 | 98.3088% | 98.3131% | 0.1322 | 0.0150 |
| C5 -> C1 | B2 | 97.6493% | 97.6525% | 0.2769 | 0.0237 |
| C5 -> C1 | B5 | **98.3582%** | **98.3605%** | **0.1718** | **0.0156** |
| C4+C5 -> C1 | B2 | 98.9552% | 98.9565% | 0.1059 | 0.0096 |
| C4+C5 -> C1 | B5 | **99.1418%** | **99.1436%** | **0.0894** | **0.0082** |

- C1 -> C5：B2-B5 `+0.5882 pp`，McNemar `p=0.0963`，B2 满足预声明 non-inferiority，但不能声称显著优于。
- C5 -> C1：B2-B5 `-0.7090 pp`，`p=0.0043`，seed-42 同窗口下 B5 显著更优。
- C4+C5 -> C1：B2-B5 `-0.1866 pp`，`p=0.3323`；三指标 0.5 pp 规则因最差类别召回差 `-0.7463 pp` 判 `B5_favored`，但准确率差异不显著。
- 总结：B2 是高性价比核心，B5 在困难反向迁移和异构多源场景更稳健；当前只能主张方向依赖，不能主张任何一方跨种子普遍胜出。

## 5. 正式 C5 个性化回归

所有回归模型与策略拟合均在阿里云 ECS 完成。R4 是固定 H8 的最佳可部署 coverage-1 点估计；R7 使用 test 真值逐行选专家，只是不可部署上界。

| 分类器 | 分类错误 | R4 S_ALL RMSE/NRMSE | R4 S_CC N/RMSE | FULL forced-true-route RMSE/NRMSE |
|---|---:|---:|---:|---:|
| A6 | 27 | 28.0144 / 0.2276 | 1333 / 11.3890 | 11.9082 / 0.0690 |
| B5 | 15 | 17.4473 / 0.1352 | 1345 / 11.3890 | 11.9082 / 0.0690 |
| B2 | 10 | **14.6564 / 0.1059** | 1350 / **11.3288** | 11.9082 / 0.0690 |

主要解释：三组 forced-true-route 完全相同，S_CC 也接近，因此 coverage-1 实际路由差异主要来自分类错路由的数量和破坏程度，不是三套不同 H8 回归器。B5/B2 的 R3 自动选择退化为 R2，未得到 H2.3+ 融合增益；R5/R6 也没有超过固定 R4。

## 6. QC 结果

HC95 是主工作点，HC90 是次工作点，FULL 是 coverage-1 对照。Yield 只计算 accept；Nonreject 合并 accept+review。

| 分类器 | HC95 A/R/R | Yield | Nonreject | Accepted RMSE/NRMSE | Nonreject RMSE/NRMSE | 错路由召回 |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 1233/86/41 | 90.66% | 96.99% | 21.1455 / 0.1731 | 22.3047 / 0.1839 | 66.67% |
| B5 | 1309/33/18 | 96.25% | 98.68% | 15.9075 / 0.1197 | 16.0983 / 0.1213 | 46.67% |
| B2 | 1301/35/24 | 95.66% | 98.24% | **12.6729 / 0.0857** | **12.8614 / 0.0858** | **70.00%** |

B2-HC95 只拒绝 24/1360 个窗口，并把 35 个窗口转入 review；它筛出 7/10 个分类错误。匹配随机拒绝的错路由平均召回约 4.02%，因此 QC 风险排序有实际信息，但 QC 没有“修复”被 review/reject 的输出。

## 7. 论文证据边界

当前可以主张：

- 目标 calibration 是跨设备分类恢复的必要系统资源；source-only A0 明显失效。
- corrected MMD2 核心 B2 在主方向具有很强的性能/复杂度比。
- 完整 B5 的收益依赖方向和源域异质性，在 C5 -> C1 有显著 seed-42 同窗口优势。
- H8 coverage-1 回归优于 H2.3+，端到端尾部主要受分类路由影响。
- deployment-visible QC 能在约 95% 自动收益率下集中识别高风险窗口。

当前不能主张：

- B2 或 B5 已在 seeds 42-46 上稳定显著优于另一方。
- 窗口级切分等价于文件级独立泛化。
- C3/C4 是 C5 主协议目标域。
- R7 oracle、forced-true-route 或 S_CC 是部署性能。
- 联邦学习本身提供差分隐私或安全聚合保证。
- Pi 端到端延迟、内存和实际通信成本已经完成正式测量。

## 8. 证据和保存路径

### 8.1 GitHub 已跟踪的轻量证据

- v2r1 分类：`results/iotj_classification_ablation_20260711_v2r1_summary/`
- v3 分类：`results/iotj_classification_ablation_20260712_v3_summary/`
- 正式回归/QC：`results/iotj_c5_formal_regression_20260713_v2_summary/`
- 跨方向 F1：`results/iotj_b2_b5_cross_direction_20260715_f1_summary/`
- 跨方向 R1：`results/iotj_b2_b5_cross_direction_20260714_r1_summary/`
- 跨方向 R2：`results/iotj_b2_b5_cross_direction_20260715_r2_summary/`
- 六运行中文表：`docs/experiments/iotj_b2_b5_cross_direction_results_20260715.zh.md`

### 8.2 原始产物边界

- 本地分类 checkpoint/每轮统计：`results/iotj_classification_ablation_20260712_v3/`、`results/iotj_b2_b5_cross_direction_20260713/`。
- 本地正式回归行流和模型：`results/iotj_c5_formal_regression_20260713_v2/{A6,B5,B2}`。
- ECS 镜像路径：`/root/GAPS/results/iotj_c5_formal_regression_20260713_v2/{A6,B5,B2}`，跨方向和分类结果位于 `/root/GAPS/results/` 下的同名实验根。
- 控制器日志：`results/iotj_b2_b5_cross_direction_20260713_controller/` 与 `results/iotj_b2_b5_cross_direction_20260714_controller/`。
- 大型 checkpoint、逐窗口行流和完整日志默认不进入 Git；GitHub 只冻结可审计 summary、manifest 和报告。不能因为 summary 已跟踪就声称原始证据已独立归档。

## 9. 新对话推荐阅读顺序

1. `docs/experiments/iotj_latest_handoff_20260715.zh.md`，即本文件。
2. `代码文件介绍.md`，确认主线代码入口与 legacy 边界。
3. `docs/experiments/iotj_system_experiment_notebook.md`，查看逐实验决策、失败和风险账本。
4. `docs/paper/iotj_system_methodology_20260711.zh.md`，理解损失函数、回归特征和 QC 数学原理。
5. `docs/paper/GAPS_IoTJ_paper_draft_20260711.zh.md`，查看当前论文故事和结果章节。
6. `docs/experiments/iotj_evidence_archive_20260714.md`，核对 Git/raw artifact 的保存边界。
7. 再读上述六个 `results/*_summary/` 目录中的 CSV/JSON/Markdown。

## 10. 后续计划优先级

1. 冻结 B2/B5 confirmation manifests，按相同真实拓扑顺序运行 seeds 43-46，报告 mean、sample std、配对 seed 差和方向例外。
2. 做回归/QC 低校准压力实验，明确它只改变目标回归/QC calibration，不冒充端到端分类标签预算实验。
3. 生成最终部署 bundle，完成全部 1360 个 C5 test 窗口 offline/runtime 逐值 parity。
4. 在 Pi/PC 测量分类、回归、QC 延迟、RSS、模型字节、实际通信和掉线恢复。
5. 完成论文图表、claim-to-evidence map 和最终 IoTJ 写作，不再用旧 F2/P4 或 C3/C4 目标结果填补当前主表。
