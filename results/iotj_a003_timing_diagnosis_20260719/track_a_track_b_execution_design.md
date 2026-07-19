# IoT-J 最小双轨证据设计（未执行）

## 决策与边界

本设计将算法多种子确认与真实异构系统测量解耦；它不修改模型、loss、optimizer、C1/C2/C5 data split、25 global rounds、5 local epochs、batch size、learning rate、server DA 或 B2/B5 配置。`2ef7aea` 算法 source archive 与其 dataset/algorithm-config hashes 继续作为共同算法输入。host placement 与 controller orchestration 另行冻结并明确记录，不能伪装为算法 archive 的一部分。

当前 `results/c2e_summary/confirmation_protocol_manifest.json` 定义了共同的 C1/C2 -> C5 算法协议，但 `results/c2e_commands/*/command_manifest.json` 还把 C1 绑定到 Pi、C2 绑定到 PC。因此它**不能原样**同时充当 Track A 与 Track B 的执行清单；需要在不改算法 archive 的前提下，为两条轨道各自建立新的 execution-topology manifest、command manifests、host/data hashes 与 controller revision。此变动是 execution-topology change，不是 algorithm change。

## Track A — Algorithm multi-seed confirmation

### 目的与允许声明

- 目标：B2/B5 × seeds 42--46，输出 Accuracy、Macro-F1、NLL、ECE、per-class recall、mean ± sample std 与按相同 seed 的 B2-B5 paired differences。
- 允许声明：在固定数据、算法和随机种子下，算法结果在五个种子上的稳定性。
- 禁止声明：真实 Pi 异构边缘部署的时延、资源、温度、通信成本或可用性。

### 最小执行拓扑

- 在一台较快的 PC/执行环境上启动 **三个独立进程**：Flower server、logical C1 client、logical C2 client；两个 client 必须有不同的 client ID、独立进程、独立 RNG 和各自数据根。
- C1/C2 继续读取原 C1/C2 数据，预先以 dataset manifest 验证输入 hashes；server 继续使用同一 C5 calibration 输入且 strict calibration split 保持启用。
- 保持 CPU device、5 local epochs、batch 32、25 rounds、相同 seed、B2/B5 DA 参数和同一 server/client source archive。不得因共置而改成 GPU、缩短 rounds 或合并两个逻辑 client。
- Track A 可不收集 Pi 温度/RSS 等系统侧样本，也不应受 real-topology resource-coverage Gate 阻塞；仍需记录 archive hash、execution-topology manifest hash、command manifest hash、attempt state、checkpoint hash 和 C5 test evaluation stream hash。

### 风险与 Gate

- 同机双 client 仍可能 CPU 竞争。先进行一个非正式 topology smoke，记录两个 client 的独立连接、client IDs、每轮 FitRes 数量、数据 hashes 和数值有限性；它不进入五种子统计。
- Track A 的 10 个 canonical runs 必须使用完全同一 Track-A execution manifest，不可按指标换 host、换 device、换线程或选择更快 attempt。
- 它是新的 confirmation track，不能混入已失败/invalid 的 a001--a004，也不能用历史 `feaa75b` seed-42 代替。

## Track B — Real-system representative evaluation

### 目的与允许声明

- 保留 Alibaba ECS + Raspberry Pi C1 + PC C2 的真实三机拓扑。
- 预声明仅执行 B2 × 1 seed 与 B5 × 1 seed（建议均为 seed 42，原因是固定于队列首项而非任何 test metric；最终须在 Track-B manifest 中写死）。
- 输出 serialized Flower application communication、per-round wall time、server DA、Pi/PC local training、Pi/PC RSS/CPU 与 Pi temperature。
- 允许声明：该真实云边拓扑可运行，且在这两个预声明代表性配置下测得特定系统成本。禁止把两条 run 写成 B2/B5 五种子算法优劣、尾延迟总体或长期稳定性结论。

### 前置 Gate

1. 针对 C2 resource coverage 93.85% < 95% 的根因进行最小只观测修复，并在 noncanonical smoke 中验证 C1/C2 coverage >= 95%、事件/消息合同完整、数值路径与 observer OFF 等价。
2. 若 observer/controller 代码改变，生成新的 **system-observability revision/archive** 与对应 hashes；算法 archive、algorithm config 和训练超参数不变。不得把该修复伪装为 a004 的补丁或用 a004 进入系统表。
3. 两个 Track-B runs 使用相同 Track-B execution manifest，串行运行并各自通过 validator；failed/invalid attempts保留但不选择性重跑。

## 论文口径

建议写为：

> GAPS algorithm stability was evaluated over five seeds using two independent logical Flower clients on a controlled fast execution topology. Separately, representative B2 and B5 runs on an ECS–Raspberry Pi–PC topology quantified application-layer communication, phase timing, and edge resource use.

不得写为“Track A 是真实 heterogeneous edge deployment”，也不得写为“Track B 的两次运行证明五种子算法稳定性”。

## 推荐 P0 最小执行顺序（未启动）

1. 归档 a003/a004 failed/invalid evidence，保持所有先前 attempt 状态不可变；完成 a003 timing diagnosis。
2. 写入并审核 Track-A/Track-B execution manifests；执行 topology/hash/dataset preflight，不运行正式训练。
3. 启动 Track A 的 10 个 25-round canonical runs，先获得算法 mean/std 和冻结的 prediction streams。
4. 并行或随后以最小修复完成 Track-B resource-coverage smoke；通过后运行 B2-s42 与 B5-s42 各一个真实三机 25-round representative system run。
5. 将 Track A 分类统计与 Track B 系统表分开呈现；其后才冻结 downstream regression/low-calibration 的 classifier prediction stream 使用规则。

## 成本判断

- 当前真实三机 a003 的实测 B2 下界约 5.58 h/25 rounds，主因是 PC C2 local training；Track B 两个代表性 run 约需至少 12--16 h 加 recovery/validation 余量，B5 可能更慢。
- Track A 的可行性高，但其实际提速必须由 noncanonical fast-topology smoke 量化；共置两 client 可消除 Pi 的环境限制，却可能出现同机 CPU 竞争。因此不在未测 smoke 前承诺具体小时数。
