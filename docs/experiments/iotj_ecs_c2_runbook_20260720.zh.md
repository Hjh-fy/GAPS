# IoT-J ECS-C2 + Pi 代表性实验运行手册（2026-07-20）

## 1. 这次实验在做什么

当前执行拓扑为：

- 原阿里云 ECS `121.40.139.213`：Flower server 与 server-side DA；
- Raspberry Pi `192.168.137.172`：逻辑客户端 C1；
- 新阿里云 ECS `114.55.171.63`：逻辑客户端 C2；
- 本地 Windows：只运行 controller，负责检查、启动、监控、回收与验证，不参与模型训练。

算法仍使用冻结的 `C1/C2 -> C5` 协议。B2/B5 均为 25 rounds、每轮 5 local epochs、batch size 32。更换 C2 的 host placement 不改变 C2 数据、模型、loss、optimizer、server DA 或 B2/B5 配置，因此只能作为 execution-topology change；ECS-C2 结果不能冒充原 Pi+PC 拓扑结果。

固定身份：

- algorithm confirmation commit：`2ef7aea77b9dfabdd09da4f38742907a37c58c30`
- controller commit：`351a4e61133922af6705e6d276de24bec87c9bff`
- source archive SHA-256：`52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`
- dataset manifest SHA-256：`fb8946da138bea5aa829dd1f5b733561a443083beb77a873e7173cbc95fcd430`

## 2. 从哪里运行

所有命令都在下面的独立 worktree 中执行：

```powershell
Set-Location 'D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-confirmation-observability'
```

不要从默认 `main` 或仓库根目录的旧分支运行。

## 3. 标准执行顺序

### 步骤 A：检查三机连接与残留进程

```powershell
ssh root@121.40.139.213 "date -u; ps -eo pid,args | grep -E '[g]aps_flower\.(server_app|client_app)' || true"
ssh gaps@192.168.137.172 "date -u; ps -eo pid,args | grep -E '[g]aps_flower\.(server_app|client_app)' || true"
ssh root@114.55.171.63 "date -u; ps -eo pid,args | grep -E '[g]aps_flower\.(server_app|client_app)' || true"
```

如果看到不属于当前 attempt 的训练进程，先查清其身份；不要直接删除历史结果目录。

### 步骤 B：只做三机 preflight

`preflight` 不执行训练，只检查三端连接、冻结源码、数据 hash、依赖版本和算法配置。完整命令：

```powershell
D:\anaconda3\python.exe -m scripts.run_iotj_confirmation_observability `
  --protocol-manifest results/c2e_summary/confirmation_protocol_manifest.json `
  --source-archive-manifest results/c2e_summary/source_archive_manifest.json `
  --dataset-manifest results/c2e_summary/dataset_manifest.json `
  --command-root results/c2e_commands `
  --source-archive results/c2e/source/confirmation_source.tar `
  --raw-root results/iotj_ecs_c2_representative_20260720/raw `
  --runs B2:42 `
  --ecs-host root@121.40.139.213 `
  --pi-hosts gaps@192.168.137.172 `
  --wait-for-pi-minutes 30 `
  --pi-retry-seconds 10 `
  --c2-host root@114.55.171.63 `
  --c2-python /root/gaps_c2_cpu_env/bin/python `
  --c2-data-root /root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid `
  --c2-dataset-subset-manifest results/c2e_ecs_c2_topology/c2_dataset_subset_manifest.json `
  --execution-topology-manifest results/c2e_ecs_c2_topology/execution_topology_manifest.json `
  --run-timeout-seconds 172800 `
  --poll-seconds 30 `
  --preflight-only
```

成功标志是：

```text
{"queue": ["c12_to_c5__b2__s42"], "status": "preflighted"}
```

### 步骤 C：启动 B2-s42

本次实际入口已经保存为：

```powershell
Start-Process -FilePath 'C:\Windows\System32\cmd.exe' `
  -ArgumentList '/d /c "D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-confirmation-observability\results\iotj_ecs_c2_representative_20260720\launch_b2_s42_a005.cmd"' `
  -WindowStyle Hidden
```

启动脚本中的 48 小时是 controller 的最长容错窗口，不是预计训练时间，也不会改变单轮训练配置。

### 步骤 D：实时监控

单次快照：

```powershell
powershell -ExecutionPolicy Bypass -File results/iotj_ecs_c2_representative_20260720/monitor_b2_s42_a005.ps1 -Once
```

每 30 秒持续刷新：

```powershell
powershell -ExecutionPolicy Bypass -File results/iotj_ecs_c2_representative_20260720/monitor_b2_s42_a005.ps1
```

只查看本地状态链：

```powershell
Get-Content -Raw results/iotj_ecs_c2_representative_20260720/raw/c12_to_c5__b2__s42/c12_to_c5__b2__s42__a005/attempt_status.json
```

`state=running` 且 `event_type=preflight_passed` 表示三机已通过检查并进入运行；只有最终 `state=canonical` 和 validator `status=valid` 才算有效完成。`failed`、`aborted`、`invalid` 都必须保留，不能纳入论文表。

### 步骤 E：B2 完成后再运行 B5

B5 不会由当前 B2 脚本自动启动。先确认 B2 的证据已回收、validator 通过，再把同一标准命令中的 `--runs B2:42` 改成 `--runs B5:42`，使用新的日志文件和新 attempt。不要同时运行 B2 与 B5，以免三台主机资源竞争。

## 4. 自动记录了哪些系统指标

训练过程中同步记录：

- 每轮 C1/C2 的 `client_train_core` 与 `client_fit_callback`；
- server aggregate、server DA、server non-DA 与 round wall time；
- 每个客户端的 logical payload 与 serialized Flower application uplink/downlink bytes；
- Pi 与 ECS-C2 的进程树 RSS、peak RSS、CPU time、单核/整机 CPU、进程数和线程数；
- Pi CPU temperature 与 throttling（可用时）；
- Observer 的 serialization、JSON encoding、I/O、fsync 与写入字节开销；
- run/attempt/round/client/host/producer 身份和所有冻结 hash。

原始远端流在训练结束后由 controller 只追加式回收到本地 attempt 的 `raw/ecs`、`raw/pi` 与 `raw/ecs_c2`。validator 要求 25 rounds、50 FitIns、50 FitRes，并验证每个客户端训练区间的 1 Hz 资源覆盖率至少为 95%。

## 5. 断网或失败时怎么处理

- 不要手动改写 `attempt_status.json`，不要覆盖远端或本地 attempt 目录；
- 先保存 controller stdout/stderr、三端日志和当前进程身份；
- 断网后的 attempt 默认按失败处理，不能从中间轮次续作成 canonical；
- 只有完成证据回收、失败归因且旧进程清理后，才允许分配新的 attempt 从 round 1 重跑；
- 不要只看某一台机器的 `round 25` 日志判断当前实验成功，必须核对 attempt_id、owner binding、三端当前时间戳和最终 validator。

## 6. 关键代码位置

- `scripts/run_iotj_confirmation_observability.py`：正式 controller、远端唯一目录、生命周期、回收和 validator 调用；
- `scripts/run_iotj_classification_cloud_edge.py`：SSH/Pi 可达性等待与重试；
- `scripts/validate_iotj_confirmation_attempt.py`：PC-C2 / ECS-C2 显式证据合同、消息矩阵、资源覆盖率与审计；
- `scripts/sample_iotj_process_resources.py`：1 Hz 训练进程树资源采样；
- `gaps_flower/observability.py`：结构化事件与 Observer 开销；
- `gaps_flower/flower_message_audit.py`：logical/application message bytes 与 SHA-256；
- `gaps_flower/strategy.py`：server round、aggregation、DA 与通信事件；
- `gaps_flower/client_app.py`：客户端训练/fit 时延事件。

## 7. 2026-07-21 当前运行与夜间顺序门控

`B2-s42/a005` 已固定为 failed evidence：它完成 23 rounds 后，本地 Controller 同时失去 server ECS 与 ECS-C2 的 SSH 可达性。三端证据已回收，失败目录不得覆盖。当前重新从 round 1 运行的是 `B2-s42/a006`。

通用一次快照：

```powershell
powershell -ExecutionPolicy Bypass -File results/iotj_ecs_c2_representative_20260720/monitor_confirmation_attempt.ps1 `
  -RunId c12_to_c5__b2__s42 `
  -AttemptId c12_to_c5__b2__s42__a006 `
  -Once
```

持续刷新时去掉 `-Once`。B2 的最终成功条件不是日志出现 round 25，而是 `attempt_status.json` 同时满足 `state=canonical`、`event_type=attempt_end`、`reason=validator_accepted` 且 `audit_sha256` 为 64 位十六进制值。

本次夜间自动接续入口是 `overnight_b2_then_b5.ps1`。它使用独占 lock，先等待 a006 的上述成功条件；成功后写一次性 gate，运行 B5 standalone preflight，再启动 `B5-s42/a001`。B2 失败、等待超时、controller 文件 SHA 改变、B5 目录已存在或 B5 preflight 失败时都会停止，不会机械重跑，也不会删除证据。
