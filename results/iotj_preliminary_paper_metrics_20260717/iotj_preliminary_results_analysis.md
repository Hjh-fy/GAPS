# GAPS IoT-J 论文导向的初步结果分析

生成日期：2026-07-17

证据入口：`iotj_preliminary_paper_metrics_20260717.xlsx`
定位：**已有算法证据与新系统 pilot 的论文初步指标整合**，不是一次新的完整训练，也不是正式 multi-seed confirmation 结果。

## 1. 执行摘要

当前证据已经形成一条可审计但尚未完全闭合的论文链条：

1. 历史 seed-42 分类结果说明当前 C1/C2 -> C5 主协议下，B1–B5 均能达到较高的单次性能，并提供了 B2/B5 的候选依据；
2. 正式 C5 R0–R7 regression 和 FULL/HC95/HC90 operational QC 已经可以进入论文结果分析；
3. Spec A 首次提供了 B2 在真实 ECS + Raspberry Pi + PC 拓扑上的两轮 Flower 系统 pilot，包括应用层通信量、分阶段时延、训练侧资源和 Observer 开销；
4. 正式 B2/B5 五种子 confirmation、正式 25-round 系统统计、low-calibration、最终 bundle/parity 和 edge inference benchmark 仍是投稿前的主要缺口。

当前最有价值的系统初步发现是：

> 在 B2 的两轮真实云边联邦 pilot 中，ECS server-side DA 平均耗时约 168.53 s，占平均 round wall time 的约 87.3%；Raspberry Pi 本地训练约 10–12 s/round。因此，当前测得的主要时间瓶颈位于 server-side DA，而不是 Raspberry Pi local training。

这是 **preliminary observation**。应用层消息量已经测量，但 transport latency 没有独立测量；因此论文不能把该结果写成“已经完全排除网络传输瓶颈”，更不能从两轮数据线性外推 25-round total 或代表 B5。

## 2. 工作簿证据定位

| 证据类别 | 当前内容 | 允许的论文用途 | 禁止或延后的用途 |
|---|---|---|---|
| 历史算法基座 | B1–B5 seed-42、25-round 主方向分类；F1/R1/R2 三方向 seed-42 B2/B5 | 方法筛选、候选合理性、appendix/generalization 描述 | 不进入正式五种子 mean/sample std；不宣称方向稳定性或 B2/B5 的普遍优劣 |
| 已完成正式算法结果 | C5 R0–R7 regression；FULL/HC95/HC90 operational QC | 正式回归性能、actual-route 误差、QC yield/coverage/error trade-off | R7 和 forced-true-class QC oracle 不能写成可部署结果 |
| Spec A 系统 pilot | B2 两轮真实 ECS + Pi + PC Flower；应用层字节、时延、RSS/CPU/温度、Observer 开销 | preliminary system table、瓶颈假设、后续实验设计依据 | 不外推 25 轮；不代表 B5；无 transport Layer 3、tail latency、availability 或 long-run claim |

工作簿的 `Overview`、`Main Classification`、`Regression`、`Operational QC`、`System Messages`、`System Timing`、`System Resources` 和 `Claim Boundaries` 已按上述定位分开。当前描述应始终使用“整合”或“preliminary evidence package”，不能描述为“新完成的一组完整训练结果”。

## 3. 当前已经可以支持的论文结论

### 3.1 正式 C5 regression/QC 是当前最成熟的主文证据

- B2 R4 FULL actual-route RMSE/NRMSE 为 **14.6564 ppm / 0.1059**；
- B5 R4 FULL actual-route RMSE/NRMSE 为 **17.4473 ppm / 0.1352**；
- B2 HC90 的 Yield 为 **0.8949**，accepted RMSE/NRMSE 为 **11.5866 ppm / 0.0747**；
- B5 HC90 的 Yield 为 **0.8824**，accepted RMSE/NRMSE 为 **15.3599 ppm / 0.1151**。

这些结果支持以下正式叙述：

1. 分类路由质量会传导到 target regression 的 coverage-1 误差；
2. 冻结的 operational QC 能以 yield/coverage 为代价降低 accepted subset 的误差；
3. 在当前正式 C5 actual-route 结果中，B2 路由对应的回归/QC 指标优于 B5。

边界：

- R7 是 offline per-row oracle；
- QC oracle 列使用 true class 强制路由，同时保留 actual-route QC mask；
- 二者只能用于误差上界或路由差距诊断，不能作为部署性能。

### 3.2 历史分类结果只能支持候选筛选，不支持正式优劣结论

主方向 historical seed-42：

- B2 Accuracy/Macro-F1：**0.992647 / 0.992657**；
- B5 Accuracy/Macro-F1：**0.988971 / 0.988990**；
- B2-B5 Accuracy：**+0.3676 percentage points**。

这只能写成“历史单种子 screening 中 B2 略高”，不能写成 B2 正式优于 B5。

三个跨方向 seed-42 进一步表明排序具有方向依赖：

- F1 C1 -> C5：B2-B5 Accuracy **+0.5882 pp**，McNemar `p=0.0963`；
- R1 C5 -> C1：B2-B5 Accuracy **-0.7090 pp**，McNemar `p=0.00432`；
- R2 C4/C5 -> C1：B2-B5 Accuracy **-0.1866 pp**，McNemar `p=0.3323`。

这组结果适合 appendix，用于说明“单一方向 seed-42 不足以确立稳定排名”，并强化主方向五种子 confirmation 的必要性。

## 4. B2 两轮真实系统 pilot

### 4.1 通信

- serialized Flower application messages 两轮合计：**1,395,868 bytes = 1.3312 MiB**；
- logical payload 两轮合计：**1,391,310 bytes = 1.3269 MiB**；
- round 1 / round 2 application bytes：**682,174 / 713,694 bytes**；
- transport bytes：**未采集**。

当前可以比较 logical payload 和 serialized application message 的封装差异，但不能推断 TCP/TLS/gRPC transport 总字节，也不能由字节量单独推断网络传输时间。

### 4.2 时延分解

| 指标 | Round 1 | Round 2 | 平均/汇总 |
|---|---:|---:|---:|
| Round wall time | 192.50 s | 193.65 s | **193.07 s** |
| Server DA | 169.92 s | 167.14 s | **168.53 s** |
| DA / round wall | 88.27% | 86.31% | **约 87.3%** |
| Pi local train core | 10.56 s | 11.65 s | **11.11 s** |
| PC local train core | 14.67 s | 26.20 s | **20.44 s** |

两轮的 DA 绝对时间和占比接近，说明“server DA dominates”不是由单个异常 round 单独造成。Pi local training 只占 round wall 的小部分，明显不是当前两轮 pilot 的主要时间瓶颈。

但仍需保留三项限制：

1. 只有两轮，不能评估 25 轮稳态、warm-up、缓存或后期漂移；
2. PC round 2 local train 明显高于 round 1，样本不足以比较 Pi 与 PC 谁更快；
3. round wall 中除 aggregate/DA 以外的约 24.47 s/round 同时包含调度、客户端训练、消息往返和其他等待，当前没有独立 transport latency。

因此推荐论文措辞为：

> In a two-round real-topology B2 pilot, server-side domain adaptation accounted for approximately 87.3% of round wall time, while Raspberry Pi local training required about 10–12 s per round. This preliminary decomposition identifies server-side DA as the dominant measured component; transport latency was not measured separately.

### 4.3 资源与 Observer 开销

- Pi training-overlap peak RSS：**511.91 MiB**；
- Pi training-overlap mean/peak host CPU：**80.50% / 90.63%**；
- Pi mean/peak temperature：**55.0°C / 58.4°C**；
- PC training-overlap peak RSS：**426.21 MiB**；
- Observer encoding/serialization/I/O 总自测开销：**24.8501 ms**。

Observer 开销相当于两轮总 wall time 的约 **0.0064%**，在这个 pilot 中很小。该比例是 Observer 自身计量口径，只能说明当前事件量和实现下没有观察到显著 wall-time 开销，不能替代正式 ON/OFF 等价 Gate；B5 Gate 仍然失败。

## 5. 投稿前关键证据缺口

最高价值缺口不是新增更多 observability 字段，而是：

1. **统计确认缺口**：没有 B2/B5 × seeds 42–46 的正式 mean、sample std 和 paired-seed difference；
2. **完整系统训练缺口**：没有与 10 个 canonical run 同步获得的 25-round communication/time/resource summary；
3. **校准效率缺口**：没有最终 classifier stream 下 12/24/48/80/Full 的正式 paired low-calibration 结果；
4. **部署闭环缺口**：没有 final C5 bundle、1360-row parity 和 Pi/PC inference p50/p95/p99；
5. **鲁棒性缺口**：没有 disconnect/rejoin 和至少 1 h stability。

详细优先级、成本和论文价值见 `iotj_paper_evidence_gap_matrix.csv`。

## 6. 最小剩余实验路线

### P0：投稿有效性与最高证据回报

1. **B5 blocker 最小修复**

   只定位第一处分叉、增加最小回归测试、做最小修复；不得扩展 event schema、transport Layer 3 或 Observer 指标。

2. **B2/B5 五种子 formal confirmation**

   同一最终 commit/archive，按冻结顺序完成 10 个 25-round canonical runs。

3. **同步生成正式 25-round 系统统计**

   通信、round decomposition、Pi/PC local training、RSS/CPU 与温度必须随 10 个 run 同步采集。只要 instrumentation 已冻结，这部分几乎不增加训练计算成本，却同时补齐算法和系统两类主文证据。

按 B2 pilot 的 193.07 s/round 粗略估算，10×25×193.07 s 的顺序执行下限约 **13.4 h**；考虑 B5 可能更慢、主机准备和审计，建议为无失败队列预留 **14–24 h** 三机 wall time。该估算不是正式运行时间承诺。

### P1：部署与校准效率闭环

1. 现在完成 low-calibration 的 budgets、sample-key manifest、stratified sampling、多 seed contract、评估脚本和统计模板；
2. 最终 classifier checkpoint/prediction stream 冻结后，再执行正式 12/24/48/80/Full paired batch；
3. 构建 final C5 bundle；
4. 完成 1360-row offline/runtime parity；
5. 运行 Pi/PC batch=1 主、batch=32 辅的 inference benchmark。

P1 应进入主文或主要补充材料，因为它把算法结果转换为“可部署、可重复、资源可接受”的 IoT-J 系统证据。

### P2：鲁棒性与长期运行

- availability/disconnect/rejoin：建议至少把 round 8–12 断开、round 13 rejoin 场景做完；它对真实联邦系统可信度最有价值；
- C2 全程缺席：适合 appendix，不必占主文大量篇幅；
- stability：至少 1 h 可作为投稿前最低证据，6 h 更适合作为 appendix、supplement 或 reviewer-driven 扩展。

P2 不必全部进入主文。主文可保留一行 robustness summary；详细 event counts、RSS drift 和温度曲线进入 appendix。若时间或算力紧张，优先级为：rejoin > 1 h stability > C2 全程缺席 > 6 h stability。

## 7. 下一项最值得投入计算资源的实验

**工程上的下一步**是 B5 blocker 的最小修复；它本身不是为了增加论文指标，而是解除正式实验的安全阻塞。

**修复后最值得投入计算资源的实验**是同一冻结 revision 下的 **B2/B5 × seeds 42–46 共 10 个 25-round canonical confirmation runs**。原因是：

1. 直接补齐论文最关键的分类统计不确定性；
2. 同步产生正式 25-round communication/time/resource evidence；
3. 冻结最终 classifier prediction stream，为正式 low-calibration、bundle 和 parity 解锁；
4. 单次三机投入同时服务算法主表、系统表和后续部署实验，单位计算成本的论文证据价值最高。

在该 prediction stream 冻结前，不建议花计算资源生成正式 low-calibration batch；现在只完成协议、sample-key 和工具准备，避免后续整体重跑。
