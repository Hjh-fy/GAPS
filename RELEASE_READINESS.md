# GAPS Release Readiness

最后审计：2026-07-26
范围：最终系统代码、正式证据与部署可复现性；不包含任何新实验。

## 当前结论

| 维度 | 状态 | 说明 |
|---|---|---|
| Paper evidence | `COMPLETE` | evidence freeze、protocol close、claim–evidence 与正式数字已经冻结。 |
| Code import/tests | `COMPLETE_FOR_AUDITED_SCOPE` | canonical imports、CLI help、compileall 及 42 项合同/runtime/bundle 测试通过。 |
| External assets | `REQUIRED` | 正式 classifier、runtime bundle、policy、reference 和部分数据不全部随 Git 分发，必须按 manifest/SHA 恢复。 |
| Runtime v5 CLI | `MISSING` | Runtime v5 regression core 目前提供 Python API；benchmark CLI 不是部署 inference CLI。 |
| Clean-checkout deployment | `NOT_COMPLETE` | 干净 Git checkout 尚不能仅凭 tracked bytes 完成正式 v4/v5 部署。 |

因此当前状态是：

```text
PAPER_EVIDENCE_READY
CODE_CONTRACT_READY
RELEASE_ARTIFACT_CLOSURE_PENDING
```

不能表述为 `CLEAN_CHECKOUT_DEPLOYMENT_READY`。

## 1. 已完成

- 最终 classifier：B5，seeds 42–46 稳定性证据已冻结；
- 最终 regression：C1/C2 sufficient-statistics Federated H1 + C5 105D
  per-gas target Ridge；
- Runtime v5 regression core 的 calibration/test parity 已通过；
- Runtime v4 保留为 formal C5 selective-output baseline；
- Runtime v5 QC2 保留为 `VALID_CANDIDATE_NOT_PROMOTED`；
- PC/Pi steady-state、cold-start、资源和通信证据已形成；
- 当前投稿范围不再需要新实验。

## 2. 代码入口身份

| 对象 | 身份 |
|---|---|
| `gaps_deploy.c5_h8_runtime.C5H8Runtime` | formal C5 Runtime-v4 baseline API |
| `gaps_deploy/final_runtime.py` | maintained legacy C12→C345 package wrapper |
| `C5FederatedSourceRidgeRuntime` | final Runtime v5 regression-core Python API |
| `C5FederatedSourceRidgeQCRuntime` | v5 QC2 candidate Python API |
| `scripts/benchmark_iotj_final_runtime.py` | evidence benchmark tool；不是部署 CLI |

## 3. 干净检出仍缺什么

正式运行至少需要恢复 manifest 绑定的：

- B5 classifier checkpoint；
- Runtime-v4 bundle、H2.3/R4/QC references 与 policy；
- Runtime-v5 Federated H1 和 105D target Ridge；
- Runtime-v5 QC2 policy（仅在复现 candidate 时）；
- C5 输入数据或等价的正式输入 package。

恢复后必须逐项核对 bytes 和 SHA256。不能因为文件名相同就视为同一资产，也不能在
缺失时回退到随机初始化、legacy rescue 或其他 checkpoint。

## 4. Release 阻塞项

1. 建立只读、版本化的 external asset release archive；
2. 将 archive 内资产与现有 manifest/SHA index 双向绑定；
3. 增加 Runtime v5 独立 inference CLI，明确输入/输出 schema 与 fail-closed
   行为；
4. 在全新 checkout + 恢复后的 release archive 上完成 import、bundle load 和
   synthetic smoke contract；
5. 生成 clean-checkout deployment receipt。

以上是工程发布动作，不要求重训、重评估、重跑 benchmark 或重新打开 test。

## 5. Benchmark provenance

既有命令的只读重建见：

`docs/system/benchmark_command_manifest_20260725.json`

该 manifest 明确区分已记录参数与未知的原始 Python executable、shell quoting 和
working directory。它不替代正式 benchmark 结果，也不声称完整原始 shell invocation
曾被捕获。

## 6. 冻结边界

发布整理不得修改：

- 冻结模型或 checkpoint；
- 正式实验结果与论文数字；
- Runtime-v4 HC95/HC90；
- Runtime-v5 QC2 policy、threshold 或 decision gate；
- calibration/test protocol；
- claim–evidence 关系。
