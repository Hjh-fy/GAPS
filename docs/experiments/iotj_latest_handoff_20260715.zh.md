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

## 11. Spec A 确认可观测冻结交接（2026-07-17）

### 11.1 当前状态

当前仅完成 Confirmation Experiment Observability Framework 和候选冻结代码审计，**正式十运行分类确认尚未开始**。批准的 Task 1--9 代码头为 `12b3bc45dd8ceff7098e543cf94d789a2eb338d7`，包含基线 `a920ecdbdbea250220343d63926cb370178cdc5e`；Task 10 修改文档前 tracked worktree/index 均干净。fresh audit 为 related `355 passed, 4 skipped, 2 warnings in 215.39s`、full `700 passed, 4 skipped, 2 warnings in 295.37s`，训练关键六文件相对基线零 diff，四文件 `py_compile` 与 `git diff --check` 通过。

冻结 schedule 顺序必须精确为：

```text
B2:42, B5:42, B5:43, B2:43, B2:44,
B5:44, B5:45, B2:45, B2:46, B5:46
```

C5 calibration/test 必须为 `320/1360`。到本交接点尚未打开 C5 test，尚未连接 ECS/Pi 执行 formal smoke，也没有启动官方 25-round queue。

### 11.2 Task 9 可复核证据

- `.tmp_iotj_observer_gate_b2_task9_final_v10/`：synthetic/local-only/unstaged，`equivalent`，`max_abs_delta=0`，8-message cross matched，report SHA-256 `1191d766e932360c8ed2e83b9258c3e18c284010ba5dbe5df249e6de8ea48646`。
- `.tmp_iotj_observer_gate_b5_task9_final_v10/`：synthetic/local-only/unstaged，`equivalent`，`max_abs_delta=0`，8-message cross matched，report SHA-256 `a7d8a437e87d6703b8255d9431a0d47472a278bbe1d603b5658cbe9eda5d7d96`。
- 以上只证明本地 OFF-A/ON/OFF-B 观测等价；不能替代正式拓扑 smoke 或五种子确认。

### 11.3 精确命令和网络边界

先定义冻结路径；`$P/$S/$D/$C/$A` 必须来自同一次 Task 10 冻结：

```powershell
$P = 'results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json'
$S = 'results/iotj_main_confirmation_observability_20260715_summary/source_archive_manifest.json'
$D = 'results/iotj_main_confirmation_observability_20260715_summary/dataset_manifest.json'
$C = 'results/iotj_main_confirmation_observability_20260715_commands'
$A = 'results/iotj_main_confirmation_observability_20260715/source/confirmation_source.tar'
$R = 'results/iotj_main_confirmation_observability_20260715/raw'
```

安全本地 synthetic Gate；只使用新建空目录，不连接 ECS/Pi/PC，不读取项目 C5 test：

```powershell
python -m scripts.run_iotj_observer_equivalence_gate --group B2 --output-root .tmp_iotj_observer_gate_b2_new
python -m scripts.run_iotj_observer_equivalence_gate --group B5 --output-root .tmp_iotj_observer_gate_b5_new
```

协议/source archive/数据/十条 command manifest 的本地 dry-run queue Gate；当前 CLI 的正确参数是 `--validate-inputs-only`，不是旧计划中的 `--dry-run`，且该模式不做 transport 或 process action：

```powershell
python -m scripts.run_iotj_confirmation_observability --protocol-manifest $P --source-archive-manifest $S --dataset-manifest $D --command-root $C --source-archive $A --raw-root $R --validate-inputs-only
```

三主机 preflight；会等待/连接 ECS、Pi，并验证 PC runtime，部署并核对 archive/runtime identity，但不启动训练进程：

```powershell
python -m scripts.run_iotj_confirmation_observability --protocol-manifest $P --source-archive-manifest $S --dataset-manifest $D --command-root $C --source-archive $A --raw-root $R --preflight-only
```

非 canonical formal-topology OFF/ON smoke；以下两条都会接触 ECS/Pi/PC，输出必须使用新目录，并且不会占用 canonical attempt registry：

```powershell
python -m scripts.run_iotj_observer_equivalence_gate --formal-topology --protocol-manifest $P --group B2 --output-root results/iotj_main_confirmation_observability_20260715/smoke/b2
python -m scripts.run_iotj_observer_equivalence_gate --formal-topology --protocol-manifest $P --group B5 --output-root results/iotj_main_confirmation_observability_20260715/smoke/b5
```

正式十运行 controller CLI；会连接 ECS/Pi/PC 并启动冻结队列，Task 10/11 Gate 未全部通过前禁止执行：

```powershell
python -m scripts.run_iotj_confirmation_observability --protocol-manifest $P --source-archive-manifest $S --dataset-manifest $D --command-root $C --source-archive $A --raw-root $R
```

sealed summarizer；只有精确十个 validator-accepted canonical attempts 与全部输入 SHA 绑定通过后才打开 C5 test，否则在测试密封线之前 fail closed：

```powershell
python -m scripts.summarize_iotj_confirmation_observability --raw-root $R --protocol-manifest $P --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --output-root results/iotj_main_confirmation_observability_20260715_summary_final --device auto --batch-size 32
```

从 exact docs commit 生成本地 source archive/manifests；此命令不连接 ECS/Pi/PC，也不打开 C5 test，但会读取冻结的 C1/C2/C5 dataset 文件并校验 C5 `320/1360`：

```powershell
$commit = git rev-parse HEAD
python -m scripts.freeze_iotj_confirmation_protocol --confirmation-commit $commit --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --archive-output results/iotj_main_confirmation_observability_20260715/source/confirmation_source.tar --command-root results/iotj_main_confirmation_observability_20260715_commands --summary-root results/iotj_main_confirmation_observability_20260715_summary
```

### 11.4 证据与声明纪律

- logical payload、serialized Flower application message 和 transport 是三层不同口径；`transport_status=not_collected` 永远不能显示为 0。
- 每个 run 只允许一个 canonical attempt。failed/invalid/aborted attempt 必须原地保留，不能删除、覆盖或根据 metric 选择重跑。
- summary 只从 immutable status chain 发现 canonical attempt，绑定 `confirmation_commit`、source archive、dataset、algorithm config、protocol、audit、checkpoint 与实际消费 sidecar SHA；禁止 metric-driven selection。
- 历史 `feaa75b` 和跨方向 seed-42 继续只作 screening/appendix。真实 C5 evaluation、ECS/Pi formal smoke 和十运行训练在本交接点均未发生。
- 2026-07-17 用户预授权只在 Tasks 10--12 与 candidate freeze、preflight、B2/B5 formal smoke、hash/schema/runtime identity、dry-run queue Gate 全通过后生效；届时可不再人工确认而启动精确十个 25-round runs。任一 Gate 失败仍立即停止并保留证据。

## 12. 论文指标闭环优先状态（2026-07-17）

- 当前不再继续扩展 Spec A。正式 10×25 未启动，B5 formal-topology round-2 parity 失败继续作为 fail-closed blocker；B2 formal-topology 已精确等价。
- 已建立统一初步指标入口：`results/iotj_preliminary_paper_metrics_20260717/`。优先阅读 `iotj_preliminary_paper_metrics_20260717.xlsx` 与 `claim_boundary.md`；源文件 SHA-256、输出哈希和证据角色记录在 `preliminary_metrics_manifest.json`。
- 现阶段可形成的论文闭环是：主方向 seed-42 screening 分类 + 三方向 seed-42 appendix + 正式 C5 回归/QC + B2 真实 ECS/Pi/PC 两轮系统 pilot。不得把该闭环改写成多种子 confirmation 或 25-round 系统结论。
- 核心初步数值：主方向 B2/B5 Accuracy `0.992647/0.988971`；B2/B5 R4 FULL actual RMSE `14.6564/17.4473 ppm`；B2 HC90 Yield `0.8949`、accepted RMSE `11.5866 ppm`；B2 两轮 application traffic `1.3312 MiB`，平均 round wall/server DA `193.0714/168.5313 s`，Pi peak RSS/temperature `511.91 MiB/58.4 °C`。
- 当前最重要的系统判断是 server DA 占平均 round wall 约 `87.3%`。因此后续系统实验应优先验证该瓶颈在更多轮次下是否稳定，而不是继续增加 Observer 字段。
- 下一执行优先级：先把本包用于论文初步结果分析；随后只做 B5 Gate 的最小修复，并在新冻结 revision 的 B2/B5 formal smoke 均通过后运行十个正式 25-round confirmation。低校准当前只准备冻结 sample-key 协议和工具，正式批量等待 classifier prediction stream；其后再做 final bundle、1360 parity、Pi/PC inference benchmark、availability 与长稳。

### 12.1 论文导向分析已完成（2026-07-17）

- 分析入口：`results/iotj_preliminary_paper_metrics_20260717/iotj_preliminary_results_analysis.md`。
- 缺口矩阵：`results/iotj_preliminary_paper_metrics_20260717/iotj_paper_evidence_gap_matrix.csv`，含 10 项缺口，P0/P1/P2 为 3/5/2 项；新增文件 hashes 已纳入 `preliminary_metrics_manifest.json`。
- 本次只整合、分析和固化已有证据，没有新增 Spec A 功能，没有执行新训练，也没有打开新的 test 排名。
- 当前最有价值的 preliminary system finding 是：B2 两轮真实拓扑 pilot 的 application-layer traffic 为 `1.3312 MiB`，平均 round wall/server DA 为 `193.07/168.53 s`，DA 约占 `87.3%`；Pi local train 为约 `10--12 s/round`，training-overlap peak RSS 为 `511.91 MiB`，peak temperature 为 `58.4 °C`；Observer 自测总开销为 `24.85 ms`。这些值不代表 B5 或正式 25-round，且 transport latency 未独立测量。
- 最小剩余路线修订为：P0 = B5 blocker 最小修复 + B2/B5 五种子正式 confirmation + 同步 25-round communication/time/resource；P1 = low-calibration、final bundle、1360-row parity、Pi/PC inference benchmark；P2 = availability/rejoin 与至少 1 h stability，6 h 长稳优先作为 supplement/reviewer-driven 扩展。
- 下一项计算资源应投入同一冻结 revision 的十个 25-round confirmation runs，但只有 B5 最小修复、B2/B5 formal smoke 与全部 fail-closed Gate 通过后才可启动。低校准现在只冻结 budgets/sample-key/sampling/statistics contract；正式批量必须等待最终 classifier prediction stream。

### 12.2 Regression Provenance Audit（2026-07-17，只读）

- 产物根：`results/iotj_regression_provenance_audit_20260717/`，包含 `regression_provenance_map.csv`、`regression_dependency_graph.md`、`regression_federated_boundary_audit.md` 和 `regression_source_head_followup_plan.md`。
- H1 source Ridge：四种 gas 分别使用同一进程中合并的 C1+C2 source windows 进行 pooled fitting；H2 source per-gas MLP：每种 gas 一个模型，但同样为 C1+C2 pooled centralized fitting；H3 source shared MLP：C1+C2 四类合并并加入 gas one-hot 后集中训练一个模型。H1/H2/H3 均无 Flower/FedAvg/客户端参数聚合。
- R0 R3aK16 是单机/文件式 offline FedAvg source-regression reference，不是真实 Flower 回归通信；它不进入 R4 数值路径，但当前 formal input builder/legacy runtime 仍有 artifact plumbing。
- 正式固定 R4/H8 同时依赖 classifier predicted route、H1、H2、H3 与 C5 calibration 上拟合的 augmented target Ridge；target per-gas MLP 不进入 R4 数值计算，但通过 H23/R4 disagreement 进入正式 QC。三个 source heads 还通过 source-head spread 进入 QC。
- 当前 source heads 必须称为 `centrally pooled multi-source regression references`，C5 阶段称为 `target-personalized calibration/QC`；只能把分类主线称为真实设备联邦训练。禁止把当前系统概括成“端到端全流程联邦”或“source data 从未集中”。
- 当前 tracked runtime 仍是 legacy C12 -> C3/C4/C5 路径，不等价于正式 C5 R4/H8 + high-coverage QC。final C5 bundle 与 1360-row parity 尚未完成，不能宣称正式 R4 runtime 已冻结部署。
- 严格后续顺序：P0 -> 冻结 final classifier checkpoints/prediction streams -> source-head dependency ablation -> 依据 Experiment A 决定是否做 distributed sufficient-statistics Ridge -> 正式 low-calibration batch。本轮未修改训练代码，也未启动回归训练。
- 四个审计产物 SHA-256：provenance CSV `e1656bedbd3c8441d9e2253d69fae29169b934be05d1e87eb11769901bec70e5`；dependency graph `6285d4280df5e8240071a21513d614b8ce93ea40bd520974f16057cd34c1851a`；boundary audit `eb4c911e9d0ace44e030d34336989d707effff87e0ac1968e10afe6d18e1db53`；follow-up plan `0ca725610dde0504d3188b42c9df96bee4371f77a62cd538be09a9f7156c2478`。

### 12.3 P0 B5 blocker 当前定位（2026-07-17）

- P0 顺序保持不变，正式 10×25 仍未启动。B5 formal report 保持 `observer_path_mutation` 与 `max_abs_delta=0.01269597525242716`。
- round 2 最早捕获到的结构差异是 FitRes arrival order：OFF `[C2,C1]`、ON `[C1,C2]`；每客户端 normalized FitRes 与 post-aggregate/pre-DA checkpoint 已证明一致，首个已捕获数值分叉在 server DA。B2 对应顺序一致并精确等价。
- fixed-state replay 显示相同顺序精确等价且 RNG 未变化；逆序改变 prototype-loss scalar `0.00390625`，但梯度仍精确一致。该证据尚不能证明 arrival order 是正式 checkpoint 分叉的根因，故没有提交未经证实的排序修复。
- focused replay tests 为 `14 passed in 37.64s`；新增未跟踪的只读诊断脚本 `scripts/diagnose_b5_fixed_state_order_replay.py` 与 focused test `tests/test_b5_fixed_state_order_replay.py`，没有修改训练/runtime 数值路径，也没有形成候选修复。formal OFF-A/OFF-B 独立重复性仍未证明，本轮没有创建新 candidate commit、source archive 或 confirmation freeze record。

### 12.4 B5 formal OFF repeat 与已验证最小修复（2026-07-17）

- 新的非 canonical `a997` formal OFF-B 已完成，复用 `7ec77e3`、archive `c96fd135...`、相同数据/命令与 frozen initial checkpoint。旧 OFF-A 与新 OFF-B 不相等：`status=environment_nondeterminism`、`max_abs_delta=0.02182745933532715`，repeat report SHA-256 `6be58896b1fabd5425538b84dd28d907facf3d3978e7a86e47501c745d3b3fd7`。这推翻了“Observer 是根因”的旧临时分类。
- 第一处真实分叉发生在 round 1 server semantic-prototype reduction：上游 initial/FitIns/逐客户端 FitRes/plain aggregate 全部精确相同，但旧 OFF-A 到达顺序 `[2,1]`、新 OFF-B `[1,2]`。首字段为 `semantic_protos["0,0"][0]`，绝对差 `1.0728836059570312e-05`；完整报告见 `.../b5_formal_off_repeat_a997_7ec77e3_v2/b5_first_divergence_report.json`。
- 最小修复是 `gaps_flower.strategy.canonicalize_fit_results()`：按上传 `client_id` 唯一、正整数、升序规范化 `GapsStrategy.aggregate_fit()` 输入。它不改变模型、loss、配置、数据、超参数或 DA 公式，只消除 edge timing 对 float32 reduction order 的选择。
- 修复后本地真实 Flower B2/B5 OFF-A/ON/OFF-B 都为 `equivalent`、`max_abs_delta=0`。focused/相关验证为 `18 passed`、`72 passed, 1 skipped`、`290 passed, 3 skipped`、summary `39 passed, 1 skipped`；深目录产生的 MAX_PATH 失败已在 `D:\itj7` 短 basetemp 重跑排除。
- 仍未创建新 confirmation commit/archive/freeze record，十个 25-round runs 仍未启动。严格下一步：candidate commit -> 唯一 source archive/protocol hashes -> 三主机 preflight -> 新 revision B2 formal smoke -> 新 revision B5 formal smoke -> 两者精确等价后自动进入预授权的正式队列。

### 12.5 最终 confirmation freeze（2026-07-17）

- 冻结算法 commit：`2ef7aea77b9dfabdd09da4f38742907a37c58c30`；source archive：`results/c2e/source/confirmation_source.tar`；archive SHA-256：`52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`。
- manifests/commands：`results/c2e_summary/` 与 `results/c2e_commands/`。dataset manifest SHA-256：`fb8946da138bea5aa829dd1f5b733561a443083beb77a873e7173cbc95fcd430`；protocol manifest SHA-256：`ba289bf87a7d4526f3e3a3639e6e04881c75aececc84bdd8fd918b89b054f57b`。
- 三机 preflight 已通过。B2 与 B5 formal-topology OFF/ON 均为 `equivalent`、`max_abs_delta=0`、message/trace `matched`、0 mismatch；两个 report 分别位于 `results/c2e_smoke_b2/`、`results/c2e_smoke_b5/`。
- `results/c2e_summary/confirmation_freeze_record.json` 已通过哈希交叉验证；冻结队列顺序为 B2-s42、B5-s42、B5-s43、B2-s43、B2-s44、B5-s44、B5-s45、B2-s45、B2-s46、B5-s46。
- 正式队列必须继续使用上述 archive，而不是后续仅记录 freeze evidence 的文档 commit。运行中同步采集通信、时延、RSS/CPU/温度；不得打开新的 test 排名，失败 attempt 保留但不纳入 confirmation。

### 12.6 正式 confirmation 运行状态（2026-07-17 22:04 Asia/Shanghai）

- 10×25 controller 已作为隐藏后台进程启动，PID `1664`；日志根 `results/c2e_runs/`。
- 当前首项 `c12_to_c5__b2__s42__a001` 已通过 attempt preflight，`state=running`；其 provenance 与 freeze record 的 commit/archive/dataset hashes 完全一致。
- 后续由同一 controller 严格按冻结的 10-run 顺序串行推进，无需再次人工批准。只读监控不得改动 attempt 目录；任一运行若失败或 validator 拒绝，controller 将保留状态/原始证据并 fail closed，不得把该 attempt 纳入正式 mean/std。

### 12.7 断网后的队列恢复（2026-07-18）

- 昨晚 Controller PID `1664` 已因到 ECS 的 SSH 状态查询 30 秒超时退出；仅创建 `B2 seed-42 / a001`，状态已封存为 `failed`，其余九项未启动。三台主机均无残留进程。
- `a001` 到达 round 2，但没有完成 25 轮或通过 validator，不能作为算法结果或正式系统统计；所有本地/远端证据继续保留。
- 当前网络恢复后，同一 `2ef7aea / 52bdbf...` archive 已重新通过三机 preflight。恢复 Controller PID 为 `25268`，从 `B2 seed-42 / a002` 开始；状态 `preflight_passed / running`，PC resource sampler 正常写入，restart stderr 为空。
- 最新 Controller 身份与日志入口记录在 `results/c2e_runs/latest_controller_launch.json`；恢复没有修改训练代码、模型、配置、数据或冻结 revision。

### 12.8 第二次断网、受控清理与 a003（2026-07-18）

- `a002` 在本机到 ECS 的 SSH 22 端口再次 timeout 后 fail closed；它到达 round 21 但未完成 canonical validation。`a001/a002` 永久保留为 failed evidence，不进入 mean/std。
- 断网使 ECS 的 a002 server process group 未能被自动清理。网络恢复后按远端注册记录验证其 launch token、PID/PGID/start ticks，安全终止该 group；Pi/PC 无残留进程。
- 两次失败的 ECS/Pi raw evidence 已复制至本地 `results/c2e_runs/raw/` 后，才删除远端的两个 stale attempt directories。冻结 archive 与十个 command manifests 未受影响。
- 清理后的三机 preflight 通过；当前 Controller PID `47056` 已从 `B2-s42 / a003` 恢复，状态 `preflight_passed / running`。动态监控继续只读 `results/c2e_runs/latest_controller_launch.json`，不要使用旧 PID。

### 12.9 Controller deadline 终止与 a004 重跑（2026-07-19）

- `a003` 并非训练数值失败：Controller stderr 的唯一终止原因是 `TimeoutError: server process exceeded formal timeout`。默认 18,000 s（5 h）在 C2 完成 round 22 后、25-round canonical validation 前终止该 attempt；因此它只能保留为 failed evidence，绝不能从 round 23 续跑或进入任何正式汇总。
- a003 的 22 个完成轮次显示 mean/median round wall 为 `803.56/810.89 s`：PC C2 local train `647.97/658.46 s`，ECS server DA `152.68/149.69 s`，Pi C1 local train `41.84/42.25 s`。因此当前正式 CPU 拓扑的关键路径是 PC C2 training + ECS DA；不是 Observer（PC sampler 五小时自测总开销仅 `5.238 s`）或已测 application message。此前由两轮 pilot 得出的 14--24 h 十运行估计作废；B2 单个 25-round run 暂按约 5.6 h 加验证余量，B5 可能更慢，10-run 串行队列应按数天安排。
- pilot 并非同一 local workload：其 PC C2 为 CPU、2360 train windows、`local_epochs=1`；formal confirmation 同为 CPU/2360 windows，但按冻结协议为 `local_epochs=5`。该差异解释首要的 5x 成本，但不能解释全部约 32x local-train 增幅：pilot 每 epoch `14.67/26.20 s`，a003 折算每 epoch mean `129.59 s`。a003 training 时 process-tree CPU mean 为 6 logical CPUs 的 `23.66%`，低于 smoke 的 `37.76%`，表明 PC 的有效并行度/系统状态也不同；现有只读证据不足以在 power plan、后台竞争、线程池与 OS 状态之间归因。不可在 a004 中途改 epoch/device/threads；任何改变都将要求停止、重新冻结、重新 smoke 并从头执行 confirmation。

### 12.10 a003 慢点诊断与双轨最小证据设计（2026-07-19）

- a003 ECS/Pi/PC 证据已完整留在本地，并生成 `results/iotj_a003_timing_diagnosis_20260719/a003_round_timing_diagnosis.csv`、`a003_vs_b2_pilot_timing_analysis.md`。22 个完整轮次 mean：wall `803.56 s`、PC C2 train `647.97 s`、Pi C1 train `41.82 s`、server DA `152.68 s`、waiting/synchronization 合并残差 `2.82 s`。PC C2 是主 slowdown（`80.6%` wall、相对 pilot `31.71x`）；DA 比 pilot 更短，不能归为本次变慢的来源；无 wall 或 PC-train 随 round 单调恶化证据。
- a004 虽完成 25 rounds，但 C2 resource coverage `0.938536` 未达到 `0.95` Gate，已判 `invalid / validator_rejected`；不进入算法/系统结果，且在诊断和执行策略变更前不得启动新的 attempt。
- 旧 confirmation Controller PID `34712` 已在确认 a004 `invalid` 后受控停止；a004 三端 events/resource/close summaries 与 `attempt_audit.json` 均已回收在本地。不得自动分配 a005；任何后续运行须先审核新的 Track-A 或 Track-B execution manifest。
- 推荐的未执行双轨设计位于 `track_a_track_b_execution_design.md`：Track A 用两个独立 logical Flower clients 的快速共置/快速执行拓扑完成 B2/B5 五种子算法确认；Track B 仅执行预声明 B2/B5 各一个真实 ECS+Pi+PC 25-round run 形成系统表。保留同一算法 archive、数据和训练配置，但为 host placement/controller orchestration 单独冻结 execution-topology manifests；Track A 不能称为真实异构边缘部署，Track B 两条 run 不能称为算法多种子稳定性。
- a003 的远端原始证据已先回收到本地 `results/c2e_runs/raw/c12_to_c5__b2__s42/c12_to_c5__b2__s42__a003/raw/`：PC/ECS/Pi 文件数分别为 8/142/11（总 161 文件、112,270,916 bytes）。核对后才删除 ECS/Pi 上这一个 stale runtime directory；两台设备均确认无残留训练进程。
- 同一冻结 archive 再次通过三机 preflight。新 Controller PID `34712` 于 2026-07-19 00:35 Asia/Shanghai 启动并创建 `B2-s42 / a004`；唯一控制层调整是显式 `--run-timeout-seconds 172800`（48 h），不改变算法、数据、模型、loss 或训练超参数。应继续以 `latest_controller_launch.json` 获取实时 PID/log；a004 必须从 round 1 完整重跑、通过 validator 后才可成为 B2 seed-42 的 canonical candidate。
