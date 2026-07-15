# GAPS IoT-J 主方向 Confirmation Observability 规格（Spec A）

**状态：** 架构已于 2026-07-15 获得认可；本文档待最终审阅后进入实现与冻结

**基线分支：** `codex/system-safety-hardening`

**基线提交：** `a920ecdbdbea250220343d63926cb370178cdc5e`

**正式方向：** `C1/C2 -> C5`

**正式方法与种子：** B2、B5 × seeds 42、43、44、45、46，共 10 个运行

## 1. 目标与结论边界

本规格定义一套只观测、不改变算法数值路径的 Confirmation Experiment Observability Framework。它必须在 10 个主方向确认运行发生的同时，采集可审计的 Flower 应用层通信量、训练与服务端阶段时延、Raspberry Pi/PC 训练侧资源占用，并把代码、数据、运行、尝试和事件绑定到统一证据链。

本阶段回答两个问题：

1. 在同一冻结代码和协议下，B2 与 B5 在 `C1/C2 -> C5` 的五个配对种子上表现如何；
2. 这些真实 Flower 训练运行的应用层通信、每轮时延分解和 Pi/PC 训练资源成本是多少。

以下内容不属于 Spec A：

- 低校准预算的正式批量结果；
- final C5 deployment bundle；
- 1360 行 offline/runtime parity；
- Pi/PC 正式推理 benchmark 与最终 runtime RSS；
- calibration fit overhead；
- availability、掉线恢复和长稳测试；
- 三个跨方向的新增多种子训练。

这些内容分别进入后续独立规格。Spec A 只为低校准规格导出冻结的 classifier checkpoint、prediction stream、sample-key 输入接口及其 SHA-256，不提前生成低校准正式结果。

## 2. 证据分层与统计资格

### 2.1 正式 confirmation evidence

正式 confirmation mean、sample std 和 B2-B5 配对差只允许使用从最终 confirmation commit 生成的同一个 source archive 运行出的以下 10 个结果：

| Seed | 第一个运行 | 第二个运行 |
|---:|---|---|
| 42 | B2 | B5 |
| 43 | B5 | B2 |
| 44 | B2 | B5 |
| 45 | B5 | B2 |
| 46 | B2 | B5 |

B2 继续明确标记为 **post-screen exploratory**；B5 标记为 **predeclared full method**。该标签不影响两者使用相同运行纪律和配对统计，但限制论文中对 B2 的确认性表述。

### 2.2 不进入正式 mean/std 的历史证据

- `feaa75b` 的 B2/B5 seed-42 是 screening/historical evidence，不得并入新 confirmation mean/std；
- 现有三个跨方向 seed-42 共六个运行保留为 appendix/generalization evidence，不扩展到五种子；
- 任何 failed、aborted、incomplete 或 audit-invalid attempt 均保留原始记录，但不得进入算法或系统汇总。

### 2.3 Test 封存

10 个训练 attempt 全部通过结构、通信、资源和 provenance 完整性 Gate 以前，不运行、不打开、不排序新的 C5 target-test 指标。训练阶段只允许使用冻结协议已经授权的 source validation 和 C5 calibration。10 个训练均完整后，才由固定批处理命令一次性评估 10 个 checkpoint，并生成 Accuracy、Macro-F1、NLL、ECE 和 per-class recall。

## 3. 不可变协议与源码冻结

### 3.1 派生关系

最终 confirmation commit 必须从 `a920ecdbdbea250220343d63926cb370178cdc5e` 派生，只加入本规格允许的观测、校验、汇总和必要测试。不得改变：

- 模型结构、参数命名或初始化；
- loss 定义、权重或计算顺序；
- 数据生成、划分、窗口、normalization 或 sample keys；
- Flower 轮数、local epochs、batch size、optimizer、learning rate、gradient clipping；
- 聚合、semantic prototype、selector 或 server DA 数值逻辑；
- B2/B5 已冻结的算法配置。

Confirmation controller 必须使用正向 allowlist：方向只能是 `C1/C2 -> C5`，group 只能是 B2/B5，seed 只能是 42–46。不得直接沿用会展开 A0/A0T/A6/B1–B5 或跨方向任务的宽泛生成器。新 seed-42 必须与 seeds 43–46 一样从最终 confirmation commit 重跑；不得复制历史 `feaa75b` 指标填充。

### 3.2 Source archive

实现与测试通过后，从干净 confirmation commit 只生成一次 `git archive --format=tar` 源码归档。控制端计算该 tar 的 SHA-256，然后把同一字节文件分发到 ECS、Pi 和 PC。不得在各主机分别重新生成 tar。

每个正式 attempt 启动前都必须验证：

- full confirmation commit SHA；
- source archive SHA-256；
- archive 展开后的 tracked-file manifest SHA-256；
- dataset manifest SHA-256；
- 算法配置 canonical JSON SHA-256；
- Python、Flower、NumPy、PyTorch 和操作系统/CPU 环境记录。

三台主机的 Flower 精确版本必须在冻结前确定并保持一致；只满足当前 `flwr>=1.11.0` 的宽松范围不构成正式环境冻结。平台相关的 PyTorch 构建允许不同，但必须逐主机记录精确版本和 wheel/build 标识。

### 3.3 Dataset manifest

数据 manifest 必须列出所有实际读取文件的相对路径、字节数和 SHA-256，并生成基于 canonical JSON 的聚合 `dataset_manifest_sha256`。同时断言：

- source clients 仅为 C1、C2；target 仅为 C5；
- C5 calibration 为 320 个窗口，C5 test 为 1360 个窗口；
- active train/calibration/test 文件集合与冻结协议一致；
- 未出现 legacy C3/C4 target、H8+C4、P4 leakage 或旧 R3aK16 主线输入。

任一哈希或协议断言失败时 fail closed，不启动训练。

## 4. 只观测架构

框架采用 hybrid observer，由四部分组成：

1. **Flower message accountant：** 在现有 `FitIns` 已完成配置后，以及 `FitRes` 已到达服务端、尚未进入聚合前，对现有对象的只读副本做逻辑分量计数和官方 Flower protobuf 序列化；
2. **Phase timer：** 在不拆分、不重排算法语句的前提下，为 client fit/local train、server aggregate、server DA 和 fit-round wall 添加单调时钟边界；
3. **Local JSONL sidecar：** server、C1 client、C2 client 和 controller 分别落盘，不通过 Flower config/metrics/message 回传 observer 自身数据；
4. **External resource sampler：** 独立进程采集 Pi/PC 训练进程树的 RSS/CPU，并单独记录 sampler 自身资源和 I/O。

Observer 不得：

- 向 `FitIns.config` 或 `FitRes.metrics` 添加任何 observability 字段；
- 修改、排序、替换或重新赋值训练正在使用的 Parameters、ndarray、tensor、config 或 metrics；
- 把 observer event、计数、时延或资源样本塞入 Flower 上下行消息；
- 在训练线程中读取 target-test 数据；
- 因记录失败而静默继续并生成可发表结果。

Observer 允许对已有消息构建独立 protobuf 副本并序列化；副本不得反向写入训练对象。

## 5. 通信字节三层合同

所有论文表格和 CSV 列名必须带层级前缀，禁止把三层数字混称为“网络流量”。正式主口径是第二层 serialized Flower application message；第一层用于解释组成，第三层是可选的传输侧辅助证据。

### 5.1 Layer 1：Logical payload

Logical payload 表示算法语义内容，不含 protobuf 字段标签、长度前缀、map key、gRPC framing、HTTP/2、TCP/IP、SSH tunnel、TLS 或重传。

下行每 client/round 至少记录：

- `logical_downlink_model_value_bytes`：所有全局模型 ndarray 的 `sum(array.nbytes)`；
- `logical_downlink_parameter_blob_bytes`：Flower `Parameters.tensors` 中现有 ndarray blob 的字节数之和；
- `logical_downlink_semantic_proto_utf8_bytes`：现有 `semantic_protos_json` 值的 UTF-8 字节数；
- `logical_downlink_other_config_value_bytes`：其余现有 config 值按冻结的 scalar logical-size 规则计数；
- `logical_downlink_total_bytes`：上述 logical 分量之和。

上行每 client/round 至少记录：

- `logical_uplink_model_value_bytes`；
- `logical_uplink_parameter_blob_bytes`；
- `logical_uplink_prototype_utf8_bytes`，含已有 `prototype_json`；
- `logical_uplink_prototype_var_utf8_bytes`，含已有 `prototype_var_json`；
- `logical_uplink_statistics_utf8_bytes`，含已有 class/phase counts、global feature summary 和 device residual JSON；
- `logical_uplink_diagnostic_value_bytes`，含其余已有诊断 scalar/string/bytes；
- `logical_uplink_total_bytes`。

Scalar logical-size 规则冻结为：bool=1 byte，signed/unsigned integer=8 bytes，float=8 bytes，string=UTF-8 长度，bytes=原始长度。Key 名、容器和编码开销不进入 logical payload。

`model_value_bytes` 与 `parameter_blob_bytes` 是两种视角，均需单列；为避免重复，`logical_*_total_bytes` 使用 `parameter_blob_bytes` 而不再次加入 `model_value_bytes`。Logical 分量不要求与第二层 serialized message 相加相等。

### 5.2 Layer 2：Serialized Flower application message

该层定义为实际协议栈发送的 Flower protobuf 应用消息本体：

- 下行：完整 `flwr.proto.ServerMessage(fit_ins=...)` 的 deterministic protobuf serialization 长度；
- 上行：完整 `flwr.proto.ClientMessage(fit_res=...)` 的 deterministic protobuf serialization 长度。

构建过程必须使用冻结环境中 Flower 官方 `serde.fit_ins_to_proto` / `serde.fit_res_to_proto` 和实际 `ServerMessage` / `ClientMessage` 类型，再调用 deterministic `SerializeToString`。记录字段为：

- `application_downlink_message_bytes`；
- `application_uplink_message_bytes`；
- `application_round_total_bytes`；
- `application_25round_total_bytes`。

该层包含 protobuf wrapper、field tags、length prefixes、status、tensor type、parameter blob、num_examples、config/metrics map entries及其 key/value 编码；不包含 gRPC 的 5-byte message prefix、压缩、HTTP/2 frame、TLS record、TCP/IP、SSH tunnel 或重传。

Observer 还必须记录 serialized message 的 SHA-256，以支持同一对象在不同观测点的交叉核对。若运行所用 Flower transport 的真实发送对象不是上述 legacy `ServerMessage/ClientMessage`，实现必须在实际发送序列化边界取数并更新测试；不得继续使用“重建近似值”冒充实际应用消息。

### 5.3 Layer 3：Transport bytes

Transport bytes 是被动采集点看到的实际链路字节，可能包含 gRPC framing、HTTP/2、TLS、TCP/IP、SSH tunnel、ACK 和重传。由于当前 Pi/PC 经隧道连接，transport 数字必须同时记录：

- capture host 与 network interface；
- 方向、端口/filter；
- capture 起止单调时间；
- 是否位于 TLS/SSH 加密前或后；
- 是否包含 ACK、重传和同端口非 Flower 流量；
- capture coverage 与缺口。

只有在 filter 能隔离本 attempt 且 coverage 完整时，才生成 `transport_uplink_bytes`、`transport_downlink_bytes` 和 `transport_total_bytes`。Transport 层不能精确归因到 prototype/statistics 等逻辑分量，也不能用 `application bytes + 固定头部` 推算。

Spec A 的有效性强制要求 Layer 1 和 Layer 2 完整；Layer 3 是 secondary evidence。未采集 Layer 3 必须显式写 `transport_status=not_collected`，不得写 0，也不影响应用层 confirmation 资格。

## 6. Observer 自身开销合同

Observer 开销不从原始 round wall、client fit 或 server aggregate 时间中扣除。原始观测时间和 observer 开销分别报告，读者可看到测量扰动上界。

每个 message/event 至少记录：

- `observer_flower_serialize_ns`：官方 Flower serde + deterministic protobuf serialization；
- `observer_event_encode_ns`：canonical JSON event 编码；
- `observer_io_write_ns`：JSONL write + flush；
- `observer_fsync_ns`：只在 round/end/failure durability boundary 执行的 fsync；
- `observer_total_ns`：observer 入口到返回的总单调时间；
- `observer_event_bytes_written`；
- `observer_event_count`。

写盘策略冻结为 UTF-8、每行一个 canonical compact JSON、行尾 `\n`。每个事件执行 write + userspace flush；每个 `round_end`、`run_end`、`failure` 执行 fsync。各阶段计时使用 `time.perf_counter_ns()`，wall-clock 仅用于跨主机可读时间戳。

为避免“把一条记录自身的最终写盘耗时写回同一条记录”的因果循环，I/O 计量采用一事件延迟合同：Flower serialization 耗时可写入当前 domain event；该 event 的 JSON encode、write、flush/fsync 耗时由紧随其后的 `observer_overhead` event 通过 `observed_event_id` 引用。Producer 关闭时另写累计 close summary，并明确记录 close-summary 本身的 `observer_reporting_tail_bytes`；该尾部报告写入不冒充已完整计量的训练期 I/O。汇总必须同时给出逐事件已配对开销、累计开销和 reporting tail，不能静默丢弃未配对尾项。

外部 resource sampler 还必须单独记录自己的：

- process PID、CPU user/system time、RSS peak；
- sample encode/write/fsync time；
- samples 与 bytes written；
- 启停时间和异常。

正式汇总输出 observer 开销的 per-event、per-round、per-attempt total，以及 `observer_total_ns / fit_round_wall_ns`。该比例是观测扰动披露，不是校正后的性能指标。

## 7. 统一运行与事件合同

### 7.1 标识符

- `run_id` 标识逻辑实验单元，格式固定为 `c12_to_c5__{b2|b5}__s{42|43|44|45|46}`；
- `attempt_id` 标识一次不可覆盖的执行，格式固定为 `{run_id}__aNNN`，从 `a001` 单调递增；
- `round` 对训练事件为 1–25，对 run/attempt 级事件为 JSON `null`；
- `client_id` 字段始终存在，client 事件为 `C1` 或 `C2`，server/controller/sampler 非客户端事件为 JSON `null`；
- `event_id` 格式为 `{attempt_id}/{host_id}/{producer}/{process_instance_id}/{sequence:08d}`，并在全 attempt 内唯一。

同一个 `run_id` 只能有一个 canonical attempt。Canonical attempt 是 attempt 序号最小且通过全部完整性 Gate 的执行；一旦存在 canonical attempt，不得因指标不理想重跑。

### 7.2 公共事件字段

所有 JSONL 事件必须包含：

```text
schema_version = "iotj.confirmation.observability.v1"
event_id
event_type
run_id
attempt_id
group_id            # "B2" or "B5"
training_seed       # 42..46
round               # 1..25 or null
client_id           # "C1", "C2", or null
host_id
producer            # controller/server/client/resource_sampler
process_instance_id
sequence
wall_time_utc       # RFC 3339 UTC
monotonic_ns
confirmation_commit
source_archive_sha256
dataset_manifest_sha256
algorithm_config_sha256
status              # started/succeeded/failed/aborted
payload              # event-specific object
```

每个 producer 的 `sequence` 从 1 连续递增；parser 必须拒绝重复、倒退或缺少必填字段的事件。跨主机阶段配对以 round、client 和事件类型为主，以 wall time 为辅；禁止直接比较不同主机的 `monotonic_ns`。

### 7.3 事件类型与最低基数

必须支持并验证以下事件：

- controller：`attempt_start`、`preflight_passed`、`attempt_end`、`attempt_failure`；
- server：`fit_round_start`、`flower_fitins_prepared`、`flower_fitres_available`、`server_aggregate_start`、`server_da_start`、`server_da_end`、`server_aggregate_end`、`fit_round_end`；
- client：`client_fit_start`、`client_train_start`、`client_train_end`、`client_fit_end`；
- sampler：`resource_sample`、`resource_sampler_end`；
- all producers：`observer_overhead`、`producer_failure`。

`flower_fitins_prepared` 表示 strategy 返回前已经完成全部现有 config 写入的对象；它不宣称 socket 已完成发送。`flower_fitres_available` 表示 `aggregate_fit` 入口已经拿到的对象；它不宣称该时刻就是最后一个网络字节到达时刻。

每个有效 attempt 必须有 25 个 round，每轮恰有 C1、C2 各一个有效 `flower_fitins_prepared` 和 `flower_fitres_available`，并能与相同 round/client 的 client fit 事件配对。重复、缺失或身份不一致均使 attempt audit-invalid。

## 8. 阶段时延定义

所有阶段使用同一进程内 `perf_counter_ns()` 差值：

- `client_train_core_ns`：紧贴现有 `train_one_round` 调用前后，不含 observer serialization/I/O；
- `client_fit_callback_ns`：现有 client `fit` 入口到返回前，包含参数装载、local train、统计构建和原有 fit callback 工作；
- `server_aggregate_fit_total_ns`：`aggregate_fit` 入口到返回前，包含聚合、原有统计/历史/checkpoint I/O 和 DA；
- `server_da_total_ns`：现有 `_run_domain_adapt` 调用入口到返回；未执行 DA 的 round 明确记录 0 和 `da_executed=false`；
- `server_aggregate_non_da_ns`：`server_aggregate_fit_total_ns - server_da_total_ns`，名称不得缩写为纯 aggregation math；
- `fit_round_wall_ns`：strategy `configure_fit` 入口到同一 round `aggregate_fit` 返回，包含下发等待、客户端并发训练、上行等待和服务端聚合/DA。

客户端并行运行，因此 C1/C2 local time、server time 与 round wall 不构成可直接相加的串行分解。汇总必须同时给出两客户端时间和 critical-path 最大值，并明确此限制。

## 9. Pi/PC 训练资源合同

Pi 和 PC 上各运行一个独立 sampler，采样周期固定为 1.0 s。采集目标是 client 主进程及其递归子进程树，按 PID 去重后记录：

- `rss_tree_bytes` 与 `rss_tree_peak_bytes`；
- `cpu_percent_tree_one_core_scale`，其中单核满载为 100%，多核可超过 100%；
- `cpu_percent_tree_host_scale = cpu_percent_tree_one_core_scale / logical_cpu_count`；
- process/thread count；
- Pi 的 CPU temperature 和 throttling state（可读取时为必填）；
- 被采样 PID 集、样本覆盖区间和采样错误。

每个 client/round 的 active fit 区间至少有一个有效资源样本；每个 attempt 的预期采样点覆盖率至少为 95%。sampler 异常、进程身份漂移或覆盖不足使系统资源证据无效，并使该 attempt 不能成为 canonical confirmation attempt。

Sampler 自身进程不能并入训练进程树 RSS/CPU；它的开销按第 6 节单独汇总。

## 10. Observer ON/OFF 数值等价性 Gate

任何正式 confirmation commit 冻结前，必须通过三层 Gate。Gate 使用相同初始 checkpoint、数据顺序、seed、B2/B5 配置和目标设备拓扑；只切换 observability 开关。执行顺序为 OFF-A、ON、OFF-B，避免把环境本身的不稳定误归因给 Observer。

### 10.1 纯函数不变性

对固定 `FitIns`/`FitRes` fixture：

- Observer 调用前后的 parameters key/order、dtype、shape 和每个字节完全相同；
- config/metrics key 集完全相同；
- 除允许的既有 volatile timing 值外，canonical protobuf 完全相同；
- deterministic protobuf serialized length 完全相同；
- Observer 不持有可在调用后修改训练对象的共享可变引用。

### 10.2 两轮确定性集成

用两客户端、两轮的固定小型真实训练 fixture 分别执行 OFF-A、ON、OFF-B。三者必须满足：

- 每轮客户端返回参数与服务端返回参数逐 tensor bitwise equal，`max_abs_delta=0`；
- 最终 checkpoint SHA-256 完全相同；
- prototype、prototype variance、counts、global statistics 和 selector/aggregation 输入完全相同；
- server DA 后参数、loss 诊断和预测 logits 完全相同；
- 训练所见 config/metrics key 集完全相同，没有新增 observer 字段；
- 清零允许的原有 volatile timing scalar 后，FitIns/FitRes deterministic protobuf bytes 完全相同。

允许忽略的差异只限既有 wall-clock、path、PID、`fit_seconds` 等明确列入测试 allowlist 的非数值路径字段。Allowlist 必须在测试中硬编码并进入 review；不得用通配符忽略整个 metrics 或 diagnostics 对象。

### 10.3 正式拓扑 smoke gate

在 ECS + Pi + PC 的正式解包环境运行最小 OFF/ON smoke，验证进程启动、事件基数、应用消息计数、sidecar 落盘和资源采样。数值 fingerprint 仍要求精确一致；时延和资源值只检查有限、非负和 schema 合法。

若 OFF-A 与 OFF-B 自身不一致，视为环境确定性 blocker；若仅 ON 不一致，视为 Observer 改变数值路径。两种情况都必须停止冻结和正式运行，不得改用容差、四舍五入或“差异很小”放行。

## 11. Attempt 生命周期与失败纪律

每个 attempt 依次经过：

1. 创建只增不改的 attempt 目录；
2. 写入 run/attempt/protocol/provenance manifest；
3. 三主机 preflight 与哈希核对；
4. 启动 server、两侧 sampler 和两个真实 client；
5. 完成 25 轮训练和同步观测；
6. 回收所有主机 sidecar、日志、checkpoint 和 sampler summary；
7. 运行结构/audit completeness validator；
8. 标记 `canonical`、`invalid`、`failed` 或 `aborted`，永不覆盖。

允许因可客观验证的基础设施故障创建新 attempt，例如 SSH/tunnel 中断、进程崩溃、磁盘写失败、事件缺失或哈希不一致。禁止因为 Accuracy、Macro-F1、loss、某类 recall 或 B2-B5 差值不理想而重跑。

最低 fail-closed 条件包括：

- source、dataset、config 或 confirmation commit 不一致；
- Observer ON/OFF Gate 未通过；
- 任一轮/客户端消息或阶段事件缺失、重复、非有限或负数；
- 不是 25 轮或不是 C1/C2 两个真实 client；
- Layer 1/Layer 2 通信记录不完整；
- Pi/PC resource coverage 不足；
- 训练过程读取 target-test 或发生 test-driven selection；
- checkpoint/schema 不一致；
- observer producer 失败但 controller 仍试图生成正式汇总。

## 12. 汇总与统计合同

全部 10 个 canonical attempt 产生后，统一评估并输出：

- 每 run 的 Accuracy、Macro-F1、NLL、ECE、四类 recall、worst-class recall；
- B2、B5 各自在五个种子上的全部值、mean、sample std（`ddof=1`）；
- 每个 seed 的 B2-B5 paired difference，以及 paired difference 的 mean/sample std；
- 每 round/client 的 logical 与 application uplink/downlink bytes；
- 每 run 的 application 25-round total；
- client train、client fit、server aggregate、server DA、fit-round wall 的 per-round 原始值和 p50/p95/mean/total；
- Pi/PC RSS peak、steady/active mean RSS、CPU mean/peak、温度/节流；
- Observer serialization、event encoding、I/O、fsync 和 total overhead。

通信主表使用真实 serialized application message bytes。Logical payload 作为组成解释表；transport 若完整则单列，不得混入 application total。

本阶段不执行基于目标测试结果的 early stopping、超参数调整、模型筛选或异常值剔除。统计脚本必须按 run manifest 自动纳入全部且仅全部 canonical attempt。

## 13. 产物与归档合同

本地/ECS 原始根目录固定为：

`results/iotj_main_confirmation_observability_20260715/`

每个 attempt 使用：

`raw/{run_id}/{attempt_id}/{controller|server|C1|C2}/`

原始目录包含 JSONL events、resource samples、stdout/stderr、checkpoints、Flower history、原始 prediction stream 和逐轮审计文件。大型产物继续保存在本地/ECS，不提交 GitHub。

GitHub 轻量 summary 根目录固定为：

`results/iotj_main_confirmation_observability_20260715_summary/`

至少包含：

- `confirmation_protocol_manifest.json`；
- `source_archive_manifest.json`；
- `dataset_manifest.json`；
- `attempt_registry.csv`；
- `classification_per_run.csv`；
- `classification_multiseed_summary.csv`；
- `flower_communication_per_round.csv`；
- `flower_communication_summary.csv`；
- `flower_round_time_breakdown.csv`；
- `training_resource_summary.csv`；
- `observer_overhead_summary.csv`；
- `claim_boundary.md`；
- `claim_to_evidence_map.md`。

Summary、manifest、report 和轻量验证证据应纳入 GitHub；若命令 manifest 受 `.gitignore` 影响，必须通过明确的 ignore exception 或审查过的 force-add 纳入，不得让正式命令只存在于远端临时目录。

## 14. 实现顺序与完成定义

Spec A 审阅通过后才进入实现，顺序固定为：

1. 从当前批准规格提交创建干净 worktree；
2. 先写 Observer contract、三层 byte accountant、event schema、ON/OFF Gate 的失败测试；
3. 实现只观测 message accountant、phase timer、JSONL writer 和 resource sampler；
4. 实现 attempt validator、summary builder 和 controller 集成；
5. 运行单元、集成、现有回归和 OFF-A/ON/OFF-B 等价性 Gate；
6. 在正式拓扑完成 smoke gate；
7. 审查 diff，确认无训练算法、数据或超参数变化；
8. 冻结 confirmation commit、source archive 和全部 manifest；
9. 按第 2.1 节顺序运行 10 个 confirmation run；
10. 全部结构审计通过后一次性打开 target-test，生成五种子统计。

Spec A 的“完成”不是 instrumentation 代码可运行，而是同时满足：

- 代码审查确认只观测；
- Observer ON/OFF 三层 Gate 通过；
- confirmation commit 与 source archive 已冻结；
- 10 个运行均来自同一 source archive SHA-256；
- 10 个 canonical attempt 的算法和系统证据完整；
- 历史 seed-42 与跨方向证据未混入正式 mean/std；
- summary、claim boundary、notebook 和 latest handoff 已更新；
- 没有任何 fail-closed 条件被豁免。
