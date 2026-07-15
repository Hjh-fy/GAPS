# IoTJ Main-Direction Confirmation Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 GAPS 模型、loss、数据协议或训练超参数的前提下，为 B2/B5 × seeds 42–46 的 10 个 `C1/C2 -> C5` confirmation run 构建可审计的 Flower 应用层通信、阶段时延、Pi/PC 训练资源、attempt 生命周期和数值等价性框架。

**Architecture:** 新增纯观测核心和 Flower message accountant，由 client/server 入口以可关闭的 sidecar observer 调用；资源采样在独立进程运行，controller 只部署同一个 source archive 并管理不可覆盖 attempt。Validator 先验证事件、哈希和资源完整性，只有 10 个 canonical attempt 全部通过后，summarizer 才允许读取 C5 test 并生成五种子统计。

**Tech Stack:** Python 3、Flower 1.23.0 legacy `ServerMessage/ClientMessage` protobuf、protobuf 4.25.8、PyTorch、NumPy、psutil 7.0.0、pytest、JSONL、CSV、SHA-256、SSH/SCP。

## Global Constraints

- 基线必须是 `a920ecdbdbea250220343d63926cb370178cdc5e` 的后代；Spec A 与本计划文档提交可以位于该派生链上。
- 正式方向只能是 `C1/C2 -> C5`；正式 group 只能是 B2、B5；正式 seed 只能是 42、43、44、45、46。
- 旧 `feaa75b` seed-42 只保留为 historical screening evidence；新 seed-42 必须从最终 confirmation commit 重跑。
- 10 个正式运行必须使用同一个 `git archive --format=tar` 文件和同一个 source archive SHA-256。
- Observer 不得向 `FitIns.config` 或 `FitRes.metrics` 增加字段，不得修改模型数组、训练配置或聚合输入。
- 不得修改 `config.py`、`client.py`、`model.py`、`federated_dataset.py`、`gaps_flower/task.py` 或 `gaps_flower/domain_adaptation.py` 的训练数值逻辑。
- 训练固定为 25 Flower rounds、5 local epochs、batch 32、client Adam LR `5e-4`、gradient clipping 5、server DA 100 steps、server LR `5e-4`。
- B2/B5 运行顺序固定为：s42 B2→B5，s43 B5→B2，s44 B2→B5，s45 B5→B2，s46 B2→B5。
- 10 个 canonical training attempt 全部通过前，不得运行或打开新的 C5 target-test 排名。
- 任一 hash、schema、numerical equivalence、event completeness、resource coverage 或 runtime identity Gate 失败时 fail closed。
- 大型 JSONL、checkpoints、逐窗口 stream 保存在本地/ECS；GitHub 只纳入 summary、manifest、report 和轻量证据。
- 所有代码任务使用 TDD：先看到目标测试按预期失败，再写最小实现，再运行目标测试和相关回归，再提交。

---

## File and Responsibility Map

| File | Change | Single responsibility |
|---|---|---|
| `requirements-confirmation.txt` | Create | 冻结 Flower/protobuf/psutil 的 confirmation 观测依赖 |
| `gaps_flower/observability.py` | Create | 事件 identity、schema、canonical JSONL writer、NullObserver、observer overhead |
| `gaps_flower/flower_message_audit.py` | Create | Logical payload 与完整 Flower protobuf application message 计数 |
| `gaps_flower/client_app.py` | Modify | client fit/train 事件和 sidecar 生命周期接入 |
| `gaps_flower/strategy.py` | Modify | FitIns/FitRes、aggregate、DA、fit-round wall 事件接入 |
| `gaps_flower/server_app.py` | Modify | observer CLI、构造和关闭；算法参数保持原样 |
| `scripts/sample_iotj_process_resources.py` | Create | 1 Hz 跨平台进程树 RSS/CPU 与 Pi thermal sampler |
| `scripts/freeze_iotj_confirmation_protocol.py` | Create | source archive、dataset/config hash、精确 10-run schedule 和 protocol manifest |
| `scripts/run_iotj_confirmation_observability.py` | Create | archive 部署、preflight、attempt 分配、真实 ECS+Pi+PC 生命周期 |
| `scripts/validate_iotj_confirmation_attempt.py` | Create | schema、消息基数、时延、resource coverage 和 canonical 资格验证 |
| `scripts/summarize_iotj_confirmation_observability.py` | Create | test seal、五种子分类/通信/时延/资源/overhead 汇总 |
| `scripts/run_iotj_observer_equivalence_gate.py` | Create | OFF-A/ON/OFF-B 两轮确定性 Gate 与正式拓扑 smoke 校验入口 |
| `tests/test_confirmation_observability.py` | Create | event contract、sequence、delayed overhead、close summary |
| `tests/test_flower_message_audit.py` | Create | 三层字段中的 Layer 1/2 和消息不变性 |
| `tests/test_confirmation_flower_integration.py` | Create | client/server 只观测接入与无 Flower 字段变化 |
| `tests/test_confirmation_resource_sampler.py` | Create | 进程树、CPU/RSS、thermal 和 sampler self-cost |
| `tests/test_iotj_confirmation_protocol.py` | Create | 10-run allowlist、hash、archive、dataset counts |
| `tests/test_iotj_confirmation_controller.py` | Create | attempt 不覆盖、部署 SHA、顺序、失败保留 |
| `tests/test_iotj_confirmation_validator.py` | Create | 25×2 基数、hash、coverage 和 fail-closed |
| `tests/test_iotj_confirmation_summary.py` | Create | canonical-only、test seal、ddof=1 和 paired differences |
| `tests/test_iotj_observer_equivalence.py` | Create | artifact fingerprint 与 volatile allowlist 比较 |
| `docs/experiments/iotj_system_experiment_notebook.md` | Modify | 命令、失败、hash、输出和下一步 |
| `docs/experiments/iotj_latest_handoff_20260715.zh.md` | Modify | confirmation commit、证据边界和执行状态 |

### Shared interfaces frozen by this plan

`gaps_flower.observability` produces these exact public names and signatures:

- `SCHEMA_VERSION = "iotj.confirmation.observability.v1"`.
- `DURABLE_EVENT_TYPES = {"round_end", "fit_round_end", "attempt_end", "attempt_failure", "producer_failure", "resource_sampler_end"}`.
- Frozen dataclass `ObserverIdentity(run_id: str, attempt_id: str, group_id: str, training_seed: int, client_id: str | None, host_id: str, producer: str, confirmation_commit: str, source_archive_sha256: str, dataset_manifest_sha256: str, algorithm_config_sha256: str)`.
- `NullObserver.emit(event_type: str, *, round_idx: int | None, client_id: str | None, status: str, payload: Mapping[str, Any], flower_serialize_ns: int = 0) -> None` and `NullObserver.close() -> None`.
- `JsonlObserver.emit(event_type: str, *, round_idx: int | None, client_id: str | None, status: str, payload: Mapping[str, Any], flower_serialize_ns: int = 0) -> str` and `JsonlObserver.close() -> None`.
- `load_observer(context_path: str | None, events_path: str | None) -> NullObserver | JsonlObserver`.

`gaps_flower.flower_message_audit` produces frozen dataclass `MessageAudit(logical: dict[str, int], application_message_bytes: int, application_message_sha256: str, flower_serialize_ns: int)` and these exact functions:

- `audit_fit_ins(ins: FitIns) -> MessageAudit`;
- `audit_fit_res(res: FitRes) -> MessageAudit`;
- `canonical_fit_ins_bytes(ins: FitIns) -> bytes`;
- `canonical_fit_res_bytes(res: FitRes) -> bytes`.

The controller/validator/summarizer share:

```python
CONFIRMATION_SCHEDULE = (
    ("B2", 42), ("B5", 42),
    ("B5", 43), ("B2", 43),
    ("B2", 44), ("B5", 44),
    ("B5", 45), ("B2", 45),
    ("B2", 46), ("B5", 46),
)

def confirmation_run_id(group_id: str, seed: int) -> str:
    return f"c12_to_c5__{group_id.lower()}__s{seed}"
```

---

### Task 1: Freeze observer dependencies and implement the event writer

**Files:**
- Create: `requirements-confirmation.txt`
- Create: `gaps_flower/observability.py`
- Create: `tests/test_confirmation_observability.py`

**Interfaces:**
- Consumes: only Python standard library and a context JSON file.
- Produces: `ObserverIdentity`, `NullObserver`, `JsonlObserver`, `load_observer`, `canonical_json_bytes`.

- [ ] **Step 1: Write failing event-contract tests**

Create tests covering valid context loading, exact common fields, per-process sequence continuity, event-id format, delayed overhead reference, durable fsync selection, NullObserver no-op, and close summary.

```python
def test_jsonl_observer_emits_contract_and_delayed_cost(tmp_path):
    identity = ObserverIdentity(
        run_id="c12_to_c5__b2__s42",
        attempt_id="c12_to_c5__b2__s42__a001",
        group_id="B2",
        training_seed=42,
        client_id=None,
        host_id="ecs",
        producer="server",
        confirmation_commit="a" * 40,
        source_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        algorithm_config_sha256="d" * 64,
    )
    events = tmp_path / "events.jsonl"
    observer = JsonlObserver(identity, events)
    first_id = observer.emit(
        "fit_round_start",
        round_idx=1,
        client_id=None,
        status="started",
        payload={"fit_round_wall_ns": 0},
    )
    observer.emit(
        "flower_fitins_prepared",
        round_idx=1,
        client_id=None,
        status="succeeded",
        payload={"proxy_id": "proxy-1"},
        flower_serialize_ns=17,
    )
    observer.close()

    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == "iotj.confirmation.observability.v1"
    assert rows[0]["event_id"] == first_id
    assert rows[0]["sequence"] == 1
    assert rows[1]["event_type"] == "observer_overhead"
    assert rows[1]["payload"]["observed_event_id"] == first_id
    assert all(row["run_id"] == identity.run_id for row in rows)
    assert (tmp_path / "events.close.json").is_file()


def test_identity_rejects_non_confirmation_scope():
    with pytest.raises(ValueError, match="run_id"):
        ObserverIdentity(
            run_id="c5_to_c1__b2__s42",
            attempt_id="c5_to_c1__b2__s42__a001",
            group_id="B2",
            training_seed=42,
            client_id=None,
            host_id="ecs",
            producer="server",
            confirmation_commit="a" * 40,
            source_archive_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            algorithm_config_sha256="d" * 64,
        )
```

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_confirmation_observability.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'gaps_flower.observability'`.

- [ ] **Step 3: Implement the minimal writer**

Implement:

- strict `run_id` regex `^c12_to_c5__(b2|b5)__s(42|43|44|45|46)$`;
- strict matching `attempt_id = run_id + "__aNNN"`;
- RFC 3339 UTC timestamp and process-local `perf_counter_ns`;
- `event_id = attempt_id/host_id/producer/process_instance_id/sequence`;
- every event contains exactly the required common fields `schema_version`, `event_id`, `event_type`, `run_id`, `attempt_id`, `group_id`, `training_seed`, `round`, `client_id`, `host_id`, `producer`, `process_instance_id`, `sequence`, `wall_time_utc`, `monotonic_ns`, `confirmation_commit`, `source_archive_sha256`, `dataset_manifest_sha256`, `algorithm_config_sha256`, `status`, and `payload`;
- compact canonical UTF-8 JSON with `sort_keys=True` and `separators=(",", ":")`;
- write + flush for every event, fsync only for `DURABLE_EVENT_TYPES`;
- one-event-delayed `observer_overhead` keyed by `observed_event_id`;
- accumulated encode/write/fsync/bytes/event counters in `events.close.json`;
- exact close-summary field `observer_reporting_tail_bytes` for the final reporting record that cannot self-account its own I/O latency;
- no write and no mutable state in `NullObserver`.

Every `observer_overhead` payload must expose the exact fields `observer_flower_serialize_ns`, `observer_event_encode_ns`, `observer_io_write_ns`, `observer_fsync_ns`, `observer_total_ns`, `observer_event_bytes_written` and `observer_event_count`.

`requirements-confirmation.txt` must contain exactly:

```text
-r requirements.txt
flwr==1.23.0
protobuf==4.25.8
psutil==7.0.0
```

- [ ] **Step 4: Run the targeted tests**

Run: `pytest -q tests/test_confirmation_observability.py`

Expected: all tests in the file pass; no warning about an unclosed file.

- [ ] **Step 5: Commit**

```powershell
git add requirements-confirmation.txt gaps_flower/observability.py tests/test_confirmation_observability.py
git commit -m "feat: add confirmation event observer"
```

---

### Task 2: Implement logical and serialized Flower message accounting

**Files:**
- Create: `gaps_flower/flower_message_audit.py`
- Create: `tests/test_flower_message_audit.py`

**Interfaces:**
- Consumes: Flower `FitIns`, `FitRes` and `Parameters` objects without mutation.
- Produces: `MessageAudit`, `audit_fit_ins`, `audit_fit_res`, canonical protobuf byte helpers.

- [ ] **Step 1: Write failing message-accounting tests**

Use fixed arrays, UTF-8 semantic JSON, prototype/statistic metrics and a diagnostic scalar.

```python
def test_fit_res_audit_matches_full_client_message_and_does_not_mutate():
    arrays = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([3], dtype=np.int64),
    ]
    metrics = {
        "client_id": 1,
        "prototype_json": '{"0,0":[1.0,2.0]}',
        "prototype_var_json": '{"0,0":[0.1,0.2]}',
        "class_phase_counts_json": '{"0,0":7}',
        "global_feature_json": "[0.5,0.6]",
        "device_residual_json": "[0.01]",
        "fit_seconds": 1.25,
    }
    fit_res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(arrays),
        num_examples=7,
        metrics=dict(metrics),
    )
    before = canonical_fit_res_bytes(fit_res)
    audit = audit_fit_res(fit_res)
    after = canonical_fit_res_bytes(fit_res)
    expected = ClientMessage(
        fit_res=serde.fit_res_to_proto(fit_res)
    ).SerializeToString(deterministic=True)

    assert before == after == expected
    assert audit.application_message_bytes == len(expected)
    assert audit.application_message_sha256 == hashlib.sha256(expected).hexdigest()
    assert audit.logical["logical_uplink_model_value_bytes"] == 16
    assert audit.logical["logical_uplink_parameter_blob_bytes"] == sum(
        len(blob) for blob in fit_res.parameters.tensors
    )
    assert audit.logical["logical_uplink_total_bytes"] >= (
        audit.logical["logical_uplink_parameter_blob_bytes"]
    )
    assert fit_res.metrics == metrics
```

Also test:

- downlink semantic JSON is excluded from `other_config_value_bytes`;
- bool=1, int=8, float=8, string=UTF-8, bytes=raw length;
- full wrapper length is larger than the nested `FitIns/FitRes` payload where expected;
- invalid scalar type raises `TypeError`;
- a serialization exception leaves the original object unchanged.

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_flower_message_audit.py`

Expected: collection fails because `gaps_flower.flower_message_audit` does not exist.

- [ ] **Step 3: Implement exact Layer 1 and Layer 2 accounting**

Use these frozen key groups:

```python
UPLINK_PROTOTYPE_KEYS = {"prototype_json"}
UPLINK_PROTOTYPE_VAR_KEYS = {"prototype_var_json"}
UPLINK_STATISTIC_KEYS = {
    "class_phase_counts_json",
    "global_feature_json",
    "device_residual_json",
}
DOWNLINK_SEMANTIC_KEYS = {"semantic_protos_json"}
```

Serialize with the actual Flower legacy wrapper:

```python
def canonical_fit_ins_bytes(ins: FitIns) -> bytes:
    message = ServerMessage(fit_ins=serde.fit_ins_to_proto(ins))
    return message.SerializeToString(deterministic=True)


def canonical_fit_res_bytes(res: FitRes) -> bytes:
    message = ClientMessage(fit_res=serde.fit_res_to_proto(res))
    return message.SerializeToString(deterministic=True)
```

Compute `model_value_bytes` with `parameters_to_ndarrays` and `sum(array.nbytes)`, `parameter_blob_bytes` from existing `Parameters.tensors`, and totals with parameter blobs only so numeric values are not double-counted. Measure only official serde + deterministic serialization inside `flower_serialize_ns`.

`audit_fit_ins(ins).logical` must contain the five logical fields below; the FitIns event builder adds the two application fields from the generic `MessageAudit.application_message_bytes` and `MessageAudit.application_message_sha256` attributes:

```text
logical_downlink_model_value_bytes
logical_downlink_parameter_blob_bytes
logical_downlink_semantic_proto_utf8_bytes
logical_downlink_other_config_value_bytes
logical_downlink_total_bytes
application_downlink_message_bytes
application_downlink_message_sha256
```

`audit_fit_res(res).logical` must contain the seven logical fields below; the FitRes event builder adds the two application fields from the same generic `MessageAudit` attributes:

```text
logical_uplink_model_value_bytes
logical_uplink_parameter_blob_bytes
logical_uplink_prototype_utf8_bytes
logical_uplink_prototype_var_utf8_bytes
logical_uplink_statistics_utf8_bytes
logical_uplink_diagnostic_value_bytes
logical_uplink_total_bytes
application_uplink_message_bytes
application_uplink_message_sha256
```

The summarizer derives `application_round_total_bytes` and `application_25round_total_bytes` from these per-message values.

- [ ] **Step 4: Run message and existing Flower contract tests**

Run:

```powershell
pytest -q tests/test_flower_message_audit.py
pytest -q tests/test_flower_classification_contract.py
```

Expected: both commands exit 0.

- [ ] **Step 5: Commit**

```powershell
git add gaps_flower/flower_message_audit.py tests/test_flower_message_audit.py
git commit -m "feat: audit Flower application messages"
```

---

### Task 3: Connect the observer to client and server without changing Flower fields

**Files:**
- Modify: `gaps_flower/client_app.py:38-145,173-212`
- Modify: `gaps_flower/strategy.py:25-43,320-423,434-567`
- Modify: `gaps_flower/server_app.py:90-255`
- Create: `tests/test_confirmation_flower_integration.py`

**Interfaces:**
- Consumes: `load_observer`, `audit_fit_ins`, `audit_fit_res`.
- Produces: client/server JSONL events; returned Flower arrays/config/metrics remain algorithmically identical.

- [ ] **Step 1: Write failing client integration tests**

Construct OFF and ON clients with the same monkeypatched `train_one_round` result. Assert:

```python
off_arrays, off_n, off_metrics = off_client.fit(input_arrays, {"server_round": 1})
on_arrays, on_n, on_metrics = on_client.fit(input_arrays, {"server_round": 1})

for left, right in zip(off_arrays, on_arrays):
    np.testing.assert_array_equal(left, right)
assert off_n == on_n
assert set(off_metrics) == set(on_metrics)
assert not any(key.startswith("observer") for key in on_metrics)
assert {
    row["event_type"]
    for row in read_jsonl(on_events)
} >= {
    "client_fit_start", "client_train_start",
    "client_train_end", "client_fit_end",
}
```

Hard-code the volatile allowlist as:

```python
VOLATILE_FLOWER_FIELDS = {
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
}
```

No wildcard path is allowed.

- [ ] **Step 2: Write failing server integration tests**

Use a fake `ClientProxy` with stable `cid` and two `FitRes` objects with `client_id` 1 and 2. Assert:

- each configured FitIns is audited only after Gaps semantic config is complete;
- `flower_fitins_prepared` stores `proxy_id` and downlink audit;
- `flower_fitres_available` stores resolved `C1/C2` and uplink audit;
- aggregate output tensors are bitwise equal OFF versus ON;
- aggregate metrics key/value pairs are equal OFF versus ON;
- DA start/end appear only when `_run_domain_adapt` executes;
- `server_aggregate_fit_total_ns >= server_da_total_ns >= 0`;
- `fit_round_wall_ns >= server_aggregate_fit_total_ns`.

- [ ] **Step 3: Run integration tests and verify the red state**

Run: `pytest -q tests/test_confirmation_flower_integration.py`

Expected: tests fail because constructors and CLI do not accept observer inputs.

- [ ] **Step 4: Add client observer lifecycle**

Add optional constructor parameter:

```python
def __init__(
    self,
    client_id: int,
    data_root: str,
    device: str,
    local_epochs: int,
    batch_size: int,
    profile: str = "smoke",
    seed: int = 42,
    observer=None,
):
    self.observer = observer or NullObserver()
```

In `fit`:

1. emit `client_fit_start` after reading `server_round`;
2. emit `client_train_start` immediately before capturing `train_start_ns`;
3. call existing `train_one_round` without changing arguments;
4. capture `train_end_ns` before any observer call;
5. emit `client_train_end` with `client_train_core_ns`;
6. preserve the existing `metrics.update` keys exactly;
7. emit `client_fit_end` after metrics are complete and before return.

Add CLI options `--observer-context` and `--observer-events`. In `main`, build the observer before the client and close it in `finally` after `start_numpy_client` returns or raises.

- [ ] **Step 5: Add server observer lifecycle and timing**

Add `observer=None` to `CheckpointFedAvg.__init__` and store `NullObserver` when absent. Do not override `CheckpointFedAvg.configure_fit`, preserving the existing FedAvg contract.

In `GapsStrategy.configure_fit`:

1. capture round start and emit `fit_round_start`;
2. call the existing `super().configure_fit`;
3. add the existing semantic config exactly as before;
4. only when observer is enabled, audit each final FitIns and emit `flower_fitins_prepared`.

In `GapsStrategy.aggregate_fit`:

1. capture aggregate start at the first executable line;
2. emit `server_aggregate_start`;
3. audit each received FitRes before `parameters_to_ndarrays`;
4. wrap only the existing `_run_domain_adapt` call with DA timers;
5. leave all tensor operations and their order unchanged;
6. emit `server_aggregate_end` and `fit_round_end` immediately before the existing return.

In `server_app.py` add `--observer-context` and `--observer-events`, pass the observer through `strategy_kwargs`, and close in `finally`. Observer CLI values may appear in `run_config.json` but must be excluded from `algorithm_config_sha256`.

- [ ] **Step 6: Run focused and regression tests**

Run:

```powershell
pytest -q tests/test_confirmation_flower_integration.py
pytest -q tests/test_flower_classification_contract.py tests/test_flower_da_v3_corrections.py
```

Expected: all commands exit 0; existing Flower metrics and tensor assertions remain unchanged.

- [ ] **Step 7: Commit**

```powershell
git add gaps_flower/client_app.py gaps_flower/strategy.py gaps_flower/server_app.py tests/test_confirmation_flower_integration.py
git commit -m "feat: observe confirmation Flower phases"
```

---

### Task 4: Implement the external Pi/PC resource sampler

**Files:**
- Create: `scripts/sample_iotj_process_resources.py`
- Create: `tests/test_confirmation_resource_sampler.py`

**Interfaces:**
- Consumes: target PID, observer context, 1.0-second interval, optional stop file.
- Produces: `resource_sample` and `resource_sampler_end` JSONL events without joining the training process tree.

- [ ] **Step 1: Write failing sampler tests**

Tests must cover:

- recursive child PID de-duplication;
- summed RSS and process/thread counts;
- one-core CPU scale and host-normalized scale;
- sampler PID exclusion;
- Linux thermal value conversion from millidegrees;
- missing `vcgencmd` represented as `null` plus an explicit availability flag;
- target process exit produces a successful `resource_sampler_end`.

Use the current pytest process as the target for a one-sample smoke:

```python
def test_collect_process_tree_sample_excludes_sampler():
    sample = collect_process_tree_sample(
        root_pid=os.getpid(),
        sampler_pid=os.getpid() + 100000,
        previous=None,
        now_ns=time.perf_counter_ns(),
    )
    assert sample.payload["rss_tree_bytes"] > 0
    assert os.getpid() in sample.payload["pids"]
    assert sample.payload["cpu_percent_tree_one_core_scale"] >= 0.0
    assert sample.payload["cpu_percent_tree_host_scale"] >= 0.0
```

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_confirmation_resource_sampler.py`

Expected: collection fails because the sampler module does not exist.

- [ ] **Step 3: Implement sampler and self-overhead**

Implement CLI:

```text
python -m scripts.sample_iotj_process_resources
  --pid 12345
  --observer-context results/confirmation_context.C1.json
  --observer-events results/resource_samples.C1.jsonl
  --interval-seconds 1.0
  --stop-file results/c12_to_c5__b2__s42__a001/C1.stop
```

Use `psutil.Process(pid).children(recursive=True)`, deduplicate by PID, and compute CPU percentage from process user+system CPU deltas divided by monotonic wall delta. Record both scales. Read Pi temperature from `/sys/class/thermal/thermal_zone0/temp` first and `vcgencmd measure_temp` second; read throttling with `vcgencmd get_throttled` when present.

At shutdown record sampler process CPU user/system time, RSS peak, sample count, encode/write/fsync time and bytes written. Never add sampler PID to the target process tree.

- [ ] **Step 4: Run sampler tests**

Run: `pytest -q tests/test_confirmation_resource_sampler.py`

Expected: all tests pass in less than 10 seconds; no test sleeps for the formal 1.0-second interval.

- [ ] **Step 5: Commit**

```powershell
git add scripts/sample_iotj_process_resources.py tests/test_confirmation_resource_sampler.py
git commit -m "feat: sample confirmation training resources"
```

---

### Task 5: Build the immutable protocol, dataset, algorithm and source manifests

**Files:**
- Create: `scripts/freeze_iotj_confirmation_protocol.py`
- Create: `tests/test_iotj_confirmation_protocol.py`

**Interfaces:**
- Consumes: clean Git commit, primary dataset root, existing B2/B5 `build_run_manifest`.
- Produces: one tar archive, `confirmation_protocol_manifest.json`, `source_archive_manifest.json`, `dataset_manifest.json` and 10 command manifests.

- [ ] **Step 1: Write failing schedule and manifest tests**

Assert exact schedule and identities:

```python
def test_confirmation_schedule_is_exact_and_alternating():
    assert CONFIRMATION_SCHEDULE == (
        ("B2", 42), ("B5", 42),
        ("B5", 43), ("B2", 43),
        ("B2", 44), ("B5", 44),
        ("B5", 45), ("B2", 45),
        ("B2", 46), ("B5", 46),
    )
    assert [confirmation_run_id(group, seed) for group, seed in CONFIRMATION_SCHEDULE] == [
        "c12_to_c5__b2__s42", "c12_to_c5__b5__s42",
        "c12_to_c5__b5__s43", "c12_to_c5__b2__s43",
        "c12_to_c5__b2__s44", "c12_to_c5__b5__s44",
        "c12_to_c5__b5__s45", "c12_to_c5__b2__s45",
        "c12_to_c5__b2__s46", "c12_to_c5__b5__s46",
    ]
```

Dataset tests create C1/C2 train/calibration/test and C5 calibration/test fixtures, then assert:

- only C1/C2 are sources and only C5 is target;
- C5 calibration/test counts are exactly 320/1360;
- all actual input files have relative path, size and SHA-256;
- canonical aggregate hash is stable across two calls;
- adding C3/C4 files does not add them to the active manifest;
- a missing active file or wrong count raises before writing.

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_iotj_confirmation_protocol.py`

Expected: collection fails because the freeze module does not exist.

- [ ] **Step 3: Implement canonical manifest builders**

Implement these exact functions:

- `sha256_file(path: Path) -> str`;
- `canonical_sha256(payload: Mapping[str, Any]) -> str`;
- `build_dataset_manifest(data_root: Path) -> dict[str, Any]`;
- `build_algorithm_manifest(repo_root: Path, group_id: str, seed: int) -> dict[str, Any]`;
- `build_protocol_manifest(repo_root: Path, data_root: Path, confirmation_commit: str, archive_path: Path) -> dict[str, Any]`;
- `create_source_archive(repo_root: Path, confirmation_commit: str, output: Path) -> dict[str, Any]`.

`build_algorithm_manifest` must select only `protocol`, `training`, `causal_factors` and `server_adaptation` from the existing B2/B5 run manifest, then compute `algorithm_config_sha256` from canonical JSON. It must reject all other groups, seeds and directions.

`create_source_archive` must:

1. require `git rev-parse HEAD` to equal `confirmation_commit`;
2. require `git status --porcelain --untracked-files=no` to be empty;
3. invoke `subprocess.run(["git", "archive", "--format=tar", "--output", str(output), confirmation_commit], cwd=repo_root, check=True)` once;
4. hash the tar bytes;
5. enumerate tar members and hash each regular file;
6. write the exact Flower/protobuf/psutil versions into the source manifest.

- [ ] **Step 4: Generate attempt-independent command manifests**

For every scheduled run, save the unchanged algorithm commands from `build_run_manifest(group_id, seed, repo_root=repo_root, results_root=results_root)` after `group_id` has passed the exact B2/B5 allowlist, plus:

- `run_id`;
- `historical_seed42_included=false`;
- `b2_claim_status=post_screen_exploratory` or `b5_claim_status=predeclared_full_method`;
- protocol/dataset/source/config SHA-256;
- `transport_status=not_collected`;
- a statement that observer CLI is controller-local and excluded from algorithm config.

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest -q tests/test_iotj_confirmation_protocol.py
pytest -q tests/test_flower_classification_contract.py::test_ablation_manifests_freeze_c12_to_c5_protocol
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

```powershell
git add scripts/freeze_iotj_confirmation_protocol.py tests/test_iotj_confirmation_protocol.py
git commit -m "feat: freeze confirmation protocol manifests"
```

---

### Task 6: Implement the dedicated confirmation controller and attempt lifecycle

**Files:**
- Create: `scripts/run_iotj_confirmation_observability.py`
- Create: `tests/test_iotj_confirmation_controller.py`

**Interfaces:**
- Consumes: frozen protocol manifest and the one source tar.
- Produces: immutable raw attempt directories, exact host contexts, process logs and attempt registry.

- [ ] **Step 1: Write failing allowlist and attempt-allocation tests**

```python
def test_allocate_attempt_never_overwrites_and_stops_after_canonical(tmp_path):
    first = allocate_attempt(tmp_path, "c12_to_c5__b2__s42")
    second = allocate_attempt(tmp_path, "c12_to_c5__b2__s42")
    assert first.attempt_id.endswith("__a001")
    assert second.attempt_id.endswith("__a002")
    mark_attempt(first.path, "canonical", audit_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="canonical"):
        allocate_attempt(tmp_path, "c12_to_c5__b2__s42")


@pytest.mark.parametrize(
    ("group", "seed"),
    [("A6", 42), ("B1", 42), ("B2", 41), ("B5", 47)],
)
def test_controller_rejects_out_of_scope_runs(group, seed):
    with pytest.raises(ValueError, match="allowlist"):
        validate_requested_run(group, seed)
```

Also assert:

- the default queue equals `CONFIRMATION_SCHEDULE`;
- `--skip-ecs-sync` and `--skip-pi-sync` are absent;
- archive mismatch aborts before launching server;
- partial/failed attempt is retained;
- controller never inspects classification metrics to decide rerun;
- source deployment uses the same tar bytes on all hosts, not per-file sync.

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_iotj_confirmation_controller.py`

Expected: collection fails because the dedicated controller does not exist.

- [ ] **Step 3: Implement exact attempt state**

Use:

```python
@dataclass(frozen=True)
class Attempt:
    run_id: str
    attempt_id: str
    path: Path


VALID_ATTEMPT_STATES = {"running", "failed", "aborted", "invalid", "canonical"}


def allocate_attempt(raw_root: Path, run_id: str) -> Attempt:
    validate_run_id(run_id)
    run_root = raw_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    if any((path / "attempt_status.json").is_file() and
           json.loads((path / "attempt_status.json").read_text(encoding="utf-8"))["state"] == "canonical"
           for path in run_root.glob(f"{run_id}__a[0-9][0-9][0-9]")):
        raise RuntimeError(f"canonical attempt already exists for {run_id}")
    used = sorted(
        int(path.name.rsplit("__a", 1)[1])
        for path in run_root.glob(f"{run_id}__a[0-9][0-9][0-9]")
    )
    number = (used[-1] + 1) if used else 1
    attempt_id = f"{run_id}__a{number:03d}"
    path = run_root / attempt_id
    path.mkdir(parents=False, exist_ok=False)
    return Attempt(run_id, attempt_id, path)
```

Every state change writes a new immutable `status_events/status_NNN.json` and updates `attempt_status.json` atomically. The status payload includes reason, UTC time, confirmation commit and all three hashes. It never includes classification accuracy or loss.

- [ ] **Step 4: Implement archive deployment and preflight**

Reuse only transport/process helpers `_run`, `_ssh`, `_remote_python`, `_wait_for_pi`, `_start_tunnels` and `_terminate_processes` from `scripts/run_iotj_classification_cloud_edge.py`. Do not call its per-file sync functions.

Deploy the exact tar to:

- ECS path expression: `f"/root/GAPS/confirmation_runtime/{source_sha256}/source.tar"`;
- Pi path expression: `f"/home/gaps/GAPS/confirmation_runtime/{source_sha256}/source.tar"`;
- PC path expression: `Path("results/iotj_main_confirmation_observability_20260715/runtime") / source_sha256 / "source.tar"`.

Hash before and after transfer, extract into a new `src` directory, then hash extracted tracked files against `source_archive_manifest.json`. Preflight must also assert:

```text
flwr==1.23.0
protobuf==4.25.8
psutil==7.0.0
dataset_manifest_sha256 matches
algorithm_config_sha256 matches
no existing server/client process for this attempt
```

- [ ] **Step 5: Implement process and sampler lifecycle**

For each attempt:

1. write controller `attempt_start` and `preflight_passed`;
2. launch ECS server from extracted archive with server observer context/events;
3. start tunnels;
4. launch Pi C1 from extracted archive, then its sampler against the returned PID;
5. launch PC C2 from extracted archive, then its sampler against the returned PID;
6. monitor server completion without reading target metrics;
7. create stop files and wait for samplers;
8. copy ECS server and Pi C1 raw evidence back without overwriting;
9. invoke validator;
10. mark canonical only when validator returns success.

On every exception, write `attempt_failure`, retain all directories and logs, stop owned processes, and set state `failed` or `aborted`. A new attempt is allowed only after an objective infrastructure/audit failure.

- [ ] **Step 6: Run controller tests and legacy controller regression**

Run:

```powershell
pytest -q tests/test_iotj_confirmation_controller.py
pytest -q tests/test_iotj_cloud_edge_controller.py tests/test_iotj_cross_direction_controller.py
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit**

```powershell
git add scripts/run_iotj_confirmation_observability.py tests/test_iotj_confirmation_controller.py
git commit -m "feat: orchestrate immutable confirmation attempts"
```

---

### Task 7: Implement fail-closed attempt validation

**Files:**
- Create: `scripts/validate_iotj_confirmation_attempt.py`
- Create: `tests/test_iotj_confirmation_validator.py`

**Interfaces:**
- Consumes: one attempt directory, protocol manifest and host JSONL files.
- Produces: deterministic `attempt_audit.json` with `status=valid|invalid` and reason list.

- [ ] **Step 1: Write failing validator tests**

Build a compact 25-round fixture generator and test one valid attempt plus independent failures for:

- duplicate event ID;
- sequence gap;
- hash mismatch;
- missing C2 FitRes in round 17;
- negative or non-finite byte/timing value;
- missing application message SHA;
- resource sample absent from one active client/round;
- `resource_coverage < 0.95`, while canonical eligibility requires `resource_coverage >= 0.95`;
- unpaired observer overhead;
- `transport_status` omitted instead of `not_collected`.

```python
def test_validator_requires_exact_25_by_2_message_matrix(valid_attempt):
    audit = validate_attempt(valid_attempt.path, valid_attempt.protocol)
    assert audit["status"] == "valid"
    assert audit["counts"]["rounds"] == 25
    assert audit["counts"]["fitins"] == 50
    assert audit["counts"]["fitres"] == 50
    assert audit["resource"]["C1"]["coverage"] >= 0.95
    assert audit["resource"]["C2"]["coverage"] >= 0.95
```

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_iotj_confirmation_validator.py`

Expected: collection fails because the validator module does not exist.

- [ ] **Step 3: Implement parsing and common schema checks**

Implement these exact functions:

- `read_events(path: Path) -> list[dict[str, Any]]`;
- `validate_common_fields(events: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[str]`;
- `resolve_proxy_clients(events: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], str]`;
- `validate_message_matrix(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], list[str]]`;
- `validate_phase_times(events: Sequence[Mapping[str, Any]]) -> list[str]`;
- `validate_resource_coverage(events_by_host: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[dict[str, Any], list[str]]`;
- `validate_observer_overhead(events: Sequence[Mapping[str, Any]]) -> list[str]`;
- `validate_attempt(attempt_dir: Path, protocol: Mapping[str, Any]) -> dict[str, Any]`.

Validate sequence within each `(host_id, producer, process_instance_id)`. Map downlink `proxy_id` to C1/C2 using same-round FitRes events. Do not compare monotonic clocks across hosts; use each client’s own event and sampler clock for resource interval joins.

- [ ] **Step 4: Write deterministic audit output**

Sort reasons and count maps before canonical JSON serialization. Include input file SHA-256 values in the audit so the status cannot be detached from the reviewed JSONL files. Exit code 0 only for `status=valid`; exit code 2 for invalid evidence.

- [ ] **Step 5: Run validator tests**

Run: `pytest -q tests/test_iotj_confirmation_validator.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add scripts/validate_iotj_confirmation_attempt.py tests/test_iotj_confirmation_validator.py
git commit -m "feat: validate confirmation attempt evidence"
```

---

### Task 8: Implement sealed evaluation and five-seed system summaries

**Files:**
- Create: `scripts/summarize_iotj_confirmation_observability.py`
- Create: `tests/test_iotj_confirmation_summary.py`

**Interfaces:**
- Consumes: exact 10 canonical audits/checkpoints, C5 dataset only after Gate.
- Produces: classification, communication, timing, resource, observer-overhead and claim-boundary outputs.

- [ ] **Step 1: Write failing test-seal and statistics tests**

Assert evaluation refuses 9/10 canonical attempts and refuses a historical revision:

```python
def test_test_gate_refuses_incomplete_or_mixed_revision(canonical_rows):
    with pytest.raises(RuntimeError, match="10 canonical"):
        assert_test_gate(canonical_rows[:-1])
    mixed = copy.deepcopy(canonical_rows)
    mixed[-1]["confirmation_commit"] = "f" * 40
    with pytest.raises(RuntimeError, match="confirmation_commit"):
        assert_test_gate(mixed)
```

For complete rows assert:

- exactly B2/B5 × 42–46;
- all seed values retained;
- mean and `np.std(ddof=1)`;
- paired `B2 - B5` difference for accuracy, Macro-F1, NLL, ECE and recall 0–3;
- B2 claim label is `post_screen_exploratory`;
- B5 claim label is `predeclared_full_method`;
- old `feaa75b` and cross-direction paths are rejected.

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_iotj_confirmation_summary.py`

Expected: collection fails because the summarizer does not exist.

- [ ] **Step 3: Implement canonical-only classification evaluation**

After `assert_test_gate` passes, resolve each round-25 adapted checkpoint and call existing `evaluate_checkpoint_stream` for C5 test. Save raw stream under the local raw attempt, not the GitHub summary. Build `classification_per_run.csv` and `classification_multiseed_summary.csv` with:

```text
run_id,group_id,seed,claim_status,N,accuracy,macro_f1,nll,ece,
recall_0,recall_1,recall_2,recall_3,worst_class_recall,
confirmation_commit,source_archive_sha256,dataset_manifest_sha256,
algorithm_config_sha256,checkpoint_sha256
```

- [ ] **Step 4: Implement system summaries**

Write:

- `flower_communication_per_round.csv`: run/attempt/round/client and all logical/application columns;
- `flower_communication_summary.csv`: per-run uplink/downlink/round/25-round totals;
- `flower_round_time_breakdown.csv`: both client times, critical-path max, aggregate, DA, non-DA and round wall;
- `training_resource_summary.csv`: Pi/PC mean/peak RSS, both CPU scales, temperature/throttling and coverage;
- `observer_overhead_summary.csv`: serialization/encode/write/fsync/total/tail and ratio to round wall;
- `attempt_registry.csv`;
- `claim_boundary.md` and `claim_to_evidence_map.md`.

Never add client parallel times to server time as if serial. Use `transport_status=not_collected` unless a complete passive capture manifest exists.

- [ ] **Step 5: Run summary and existing metric tests**

Run:

```powershell
pytest -q tests/test_iotj_confirmation_summary.py
pytest -q tests/test_iotj_classification_summary.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

```powershell
git add scripts/summarize_iotj_confirmation_observability.py tests/test_iotj_confirmation_summary.py
git commit -m "feat: summarize confirmation system evidence"
```

---

### Task 9: Implement the OFF-A/ON/OFF-B numerical equivalence Gate

**Files:**
- Create: `scripts/run_iotj_observer_equivalence_gate.py`
- Create: `tests/test_iotj_observer_equivalence.py`

**Interfaces:**
- Consumes: deterministic artifacts or a local two-client/two-round fixture.
- Produces: `observer_equivalence_report.json`; exit 0 only for exact numerical equality.

- [ ] **Step 1: Write failing artifact comparison tests**

Freeze the only volatile paths:

```python
VOLATILE_JSON_PATHS = {
    ("run_config", "args", "observer_context"),
    ("run_config", "args", "observer_events"),
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
    ("provenance", "wall_time_utc"),
    ("provenance", "pid"),
    ("provenance", "path"),
}
```

Tests must prove:

- one changed tensor byte fails;
- one changed prototype/count/stat value fails;
- one new Flower config/metrics key fails;
- a volatile value change is ignored only at the exact path;
- OFF-A differing from OFF-B reports environment nondeterminism;
- ON alone differing reports observer path mutation.

- [ ] **Step 2: Run tests and verify the red state**

Run: `pytest -q tests/test_iotj_observer_equivalence.py`

Expected: collection fails because the Gate module does not exist.

- [ ] **Step 3: Implement deterministic fingerprints**

Implement these exact functions:

- `tensor_fingerprint(checkpoint: Path) -> dict[str, Any]`;
- `json_fingerprint(path: Path, volatile_paths: AbstractSet[tuple]) -> dict[str, Any]`;
- `compare_fingerprints(off_a: Mapping[str, Any], on: Mapping[str, Any], off_b: Mapping[str, Any]) -> dict[str, Any]`;
- `run_local_gate(output_root: Path, group_id: str) -> dict[str, Any]`.

Tensor comparison must check key order, dtype, shape, raw bytes and `max_abs_delta=0`. JSON comparison removes only exact allowlisted leaf paths. Canonicalized FitIns/FitRes tests zero only `fit_seconds` and `evaluate_seconds` values while preserving all keys and types.

- [ ] **Step 4: Implement the real two-round local fixture**

Create deterministic C1/C2 train/test and source/target calibration arrays with:

- 8 train rows per source, two rows per class;
- shape `(100, 8)`, dtype `float32`;
- fixed NumPy seed 7001;
- classification labels `[0,0,1,1,2,2,3,3]`;
- phase labels `[0,1,0,1,0,1,0,1]`;
- regression labels all zeros.

Run actual loopback Flower server and two clients for 2 rounds, local epoch 1, batch 4, CPU, once for B2 and once for B5. For each group run OFF-A, ON, OFF-B with:

```text
PYTHONHASHSEED=0
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
seed=42
server DA steps=1
server DA device=cpu
```

Use separate ports and fresh output directories. Capture final aggregated/adapted checkpoints, prototype stats, client stats, DA diagnostics and message fingerprints.

- [ ] **Step 5: Run unit and local integration Gates**

Run:

```powershell
pytest -q tests/test_iotj_observer_equivalence.py
python -m scripts.run_iotj_observer_equivalence_gate --group B2 --output-root .tmp_iotj_observer_gate_b2
python -m scripts.run_iotj_observer_equivalence_gate --group B5 --output-root .tmp_iotj_observer_gate_b5
```

Expected: each command exits 0; both reports contain `status="equivalent"`, `max_abs_delta=0.0` and identical final checkpoint SHA-256 across OFF-A/ON/OFF-B.

- [ ] **Step 6: Commit**

```powershell
git add scripts/run_iotj_observer_equivalence_gate.py tests/test_iotj_observer_equivalence.py
git commit -m "test: gate observer numerical equivalence"
```

---

### Task 10: Run the full safety audit and freeze the confirmation candidate

**Files:**
- Modify: `docs/experiments/iotj_system_experiment_notebook.md`
- Modify: `docs/experiments/iotj_latest_handoff_20260715.zh.md`
- Generate locally: `results/iotj_main_confirmation_observability_20260715_commands/`
- Generate locally: `results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json`

**Interfaces:**
- Consumes: Tasks 1–9 and the approved Spec A.
- Produces: clean candidate commit, local source archive and local protocol evidence ready for topology smoke.

- [ ] **Step 1: Run all targeted tests together**

Run:

```powershell
pytest -q tests/test_confirmation_observability.py tests/test_flower_message_audit.py tests/test_confirmation_flower_integration.py tests/test_confirmation_resource_sampler.py tests/test_iotj_confirmation_protocol.py tests/test_iotj_confirmation_controller.py tests/test_iotj_confirmation_validator.py tests/test_iotj_confirmation_summary.py tests/test_iotj_observer_equivalence.py
```

Expected: exit 0 with no failures or errors.

- [ ] **Step 2: Run the full tracked regression suite**

Run: `pytest -q`

Expected: exit 0 with no failures or errors; record the exact pass count in the notebook.

- [ ] **Step 3: Prove training-critical files did not change**

Run:

```powershell
git diff a920ecdbdbea250220343d63926cb370178cdc5e -- config.py client.py model.py federated_dataset.py gaps_flower/task.py gaps_flower/domain_adaptation.py
```

Expected: no output.

Also run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended tracked implementation/docs changes are present in the worktree.

- [ ] **Step 4: Update notebook and handoff**

Record:

- every test/Gate command and result;
- selected exact dependency versions;
- candidate confirmation commit;
- observer ON/OFF report paths and SHA-256;
- any failed attempt retained during implementation;
- no target-test evaluation performed;
- next action is formal topology smoke, not the 10-run queue.

- [ ] **Step 5: Commit the candidate**

```powershell
git add docs/experiments/iotj_system_experiment_notebook.md docs/experiments/iotj_latest_handoff_20260715.zh.md
git commit -m "docs: record confirmation observer freeze gates"
```

- [ ] **Step 6: Create the candidate source archive and protocol manifests**

Run:

```powershell
$commit = git rev-parse HEAD
python -m scripts.freeze_iotj_confirmation_protocol --confirmation-commit $commit --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --archive-output results/iotj_main_confirmation_observability_20260715/source/confirmation_source.tar --command-root results/iotj_main_confirmation_observability_20260715_commands --summary-root results/iotj_main_confirmation_observability_20260715_summary
```

Expected: exit 0; output reports 10 scheduled runs, C5 counts 320/1360, one source archive SHA-256 and one dataset manifest SHA-256.

Do not commit generated evidence yet; the candidate commit must remain the exact archive source.

---

### Task 11: Run formal-topology smoke and designate the immutable confirmation commit

**Files:**
- Read only: candidate archive and manifests from Task 10
- Generate locally/ECS/Pi: smoke attempt evidence under `results/iotj_main_confirmation_observability_20260715/smoke/`
- Modify only if smoke fails: the smallest relevant implementation/test file, followed by repeating Tasks 9–11

**Interfaces:**
- Consumes: candidate source archive.
- Produces: topology smoke report and the final `confirmation_commit`/archive SHA pair.

- [ ] **Step 1: Preflight all hosts without launching training**

Run:

```powershell
python -m scripts.run_iotj_confirmation_observability --protocol-manifest results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json --preflight-only
```

Expected: ECS, Pi and PC each report matching source archive, extracted tracked files, dataset/config hashes and `flwr=1.23.0 protobuf=4.25.8 psutil=7.0.0`.

- [ ] **Step 2: Run OFF/ON formal-topology smoke**

Run:

```powershell
python -m scripts.run_iotj_observer_equivalence_gate --formal-topology --protocol-manifest results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json --group B2 --output-root results/iotj_main_confirmation_observability_20260715/smoke/b2
python -m scripts.run_iotj_observer_equivalence_gate --formal-topology --protocol-manifest results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json --group B5 --output-root results/iotj_main_confirmation_observability_20260715/smoke/b5
```

Expected: both commands exit 0; each report has exact tensor/stat equivalence, valid application byte records, complete sidecars and at least one resource sample per client round.

- [ ] **Step 3: Fail closed on any mismatch**

If either smoke command fails:

1. do not run a formal seed;
2. preserve the smoke directories and mark them failed;
3. diagnose the failing contract;
4. add a reproducing test;
5. implement the minimal observer-only correction;
6. rerun Tasks 9–11 from a new commit and a newly generated archive.

- [ ] **Step 4: Designate and record the final freeze**

When both smoke Gates pass, construct `confirmation_freeze_record.json` directly from already validated manifests:

```python
record = {
    "status": "frozen",
    "direction": "C1/C2 -> C5",
    "groups": ["B2", "B5"],
    "seeds": [42, 43, 44, 45, 46],
    "confirmation_commit": protocol["confirmation_commit"],
    "source_archive_sha256": source["source_archive_sha256"],
    "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
    "observer_gate": "passed",
    "target_test_opened": False,
}
```

The writer must reject commit/hash values that are not exact 40/64-hex strings.

- [ ] **Step 5: Force-add lightweight freeze evidence only**

```powershell
git add -f results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json results/iotj_main_confirmation_observability_20260715_summary/source_archive_manifest.json results/iotj_main_confirmation_observability_20260715_summary/dataset_manifest.json results/iotj_main_confirmation_observability_20260715_summary/confirmation_freeze_record.json results/iotj_main_confirmation_observability_20260715_commands
git commit -m "docs: freeze IoTJ confirmation revision"
```

Do not add the tar, checkpoints, JSONL or raw logs.

---

### Task 12: Formal execution handoff

**Files:**
- Read: `confirmation_freeze_record.json`
- Write during later experiment execution: raw attempt tree and lightweight summaries defined by Spec A

**Interfaces:**
- Consumes: immutable freeze record.
- Produces: an executable, reviewable queue; this task does not silently start multi-hour runs.

- [ ] **Step 1: Verify the formal queue without target evaluation**

Run:

```powershell
python -m scripts.run_iotj_confirmation_observability --protocol-manifest results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json --dry-run
```

Expected order:

```text
c12_to_c5__b2__s42
c12_to_c5__b5__s42
c12_to_c5__b5__s43
c12_to_c5__b2__s43
c12_to_c5__b2__s44
c12_to_c5__b5__s44
c12_to_c5__b5__s45
c12_to_c5__b2__s45
c12_to_c5__b2__s46
c12_to_c5__b5__s46
```

- [ ] **Step 2: Confirm readiness**

The implementation phase is ready for formal runs only when:

- the worktree is clean;
- full tests and both local equivalence Gates pass;
- formal topology B2/B5 smoke passes;
- freeze record hashes match the deployed archive;
- no C5 target-test output exists in the new confirmation summary root;
- the queue contains exactly the ten lines above.

Stop at this checkpoint for explicit confirmation before starting the first 25-round formal attempt.

---

## Plan Self-Review Checklist

- Spec Sections 1–4: Tasks 1–6 define scope, provenance and only-observational integration.
- Spec Section 5: Task 2 implements logical/application bytes; Task 8 preserves transport as explicit secondary evidence.
- Spec Section 6: Task 1 implements delayed per-event overhead; Task 4 implements sampler self-cost; Task 8 summarizes both.
- Spec Section 7: Tasks 1, 6 and 7 implement identity/event/attempt contracts.
- Spec Section 8: Task 3 implements all phase timers; Task 8 preserves concurrent-time interpretation.
- Spec Section 9: Task 4 collects resource data; Task 7 enforces per-round and 95% coverage.
- Spec Section 10: Tasks 2, 3 and 9 implement non-mutation and OFF-A/ON/OFF-B Gates.
- Spec Section 11: Tasks 6 and 7 preserve failed attempts and reject metric-driven reruns.
- Spec Sections 12–13: Task 8 produces every required lightweight summary and raw/archive boundary.
- Spec Section 14: Tasks 10–12 enforce regression, smoke, freeze and execution ordering.
- Interface names in later tasks match the shared interfaces at the top of this plan.
- No task authorizes low-calibration results, deployment bundle, parity, inference benchmark, availability or long-run work.
