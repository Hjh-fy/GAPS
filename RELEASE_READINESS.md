# GAPS Release Readiness

最后审计：2026-07-26
范围：最终系统代码、正式证据与部署可复现性；不包含任何新实验。

## 当前结论

| 维度 | 状态 | 说明 |
|---|---|---|
| Paper evidence | `COMPLETE` | evidence freeze、protocol close、claim–evidence 与正式数字已经冻结。 |
| Code import/tests | `COMPLETE_FOR_AUDITED_SCOPE` | canonical imports、CLI help、compileall、既有合同测试及 portable release 相关测试通过。 |
| Release provenance | `CLOSED` | 35 项外部资产已按版本化 manifest、bytes 和 SHA256 锁定；不代表 archive 已生成。 |
| External archive | `V5_CORE_COMPLETE` | Runtime-v5 core archive、SHA256SUMS、portable binding 与 synthetic examples 已形成；不覆盖 v4/QC。 |
| Runtime v5 CLI | `AVAILABLE_FOR_CORE` | `python -m gaps_deploy.runtime_v5_cli` 支持 verify、describe 与 inference。 |
| Clean-checkout deployment | `RUNTIME_V5_CORE_READY` | 固定 commit 的 fresh checkout、archive restore、load 和 synthetic smoke 已通过；full system 未闭合。 |

因此当前状态是：

```text
PAPER_EVIDENCE_READY
CODE_CONTRACT_READY
RELEASE_PROVENANCE_CLOSED
CLEAN_CHECKOUT_RUNTIME_V5_CORE_READY
```

不能表述为 `CLEAN_CHECKOUT_FULL_SYSTEM_READY` 或
`FULL_V4_V5_QC_REPRODUCTION_READY`。

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
| `gaps_deploy.runtime_v5_cli` | portable Runtime-v5 core thin inference CLI |
| `gaps_deploy.runtime_v5_portable` | strict relative-path portable binding loader |
| `C5FederatedSourceRidgeQCRuntime` | v5 QC2 candidate Python API |
| `scripts/benchmark_iotj_final_runtime.py` | evidence benchmark tool；不是部署 CLI |

## 3. Runtime-v5 core portable closure

以下内容已经闭合：

- `release/gaps_runtime_v5_core_20260726.zip`
- archive SHA256：
  `740e8237384041523e51969b88795c27e43e88650c73e1a5209092880cf547de`
- portable binding schema：`gaps.runtime_v5.portable_binding.v1`
- B5 classifier、Federated H1、105D target Ridge、calibration lock；
- SHA256SUMS、archive manifest、provenance mapping；
- 不含正式 C5 test、HC95/HC90 records、offline predictions 或 QC policy；
- fresh checkout receipt：
  `docs/system/iotj_runtime_v5_clean_checkout_receipt_20260726.json`。

因此 Runtime-v5 regression core 已能从 tracked archive 在新的 checkout 中恢复并运行
synthetic inference。

## 4. 完整系统仍缺什么

正式运行至少需要恢复 manifest 绑定的：

- Runtime-v4 bundle、H2.3/R4/QC references 与 policy；
- Runtime-v5 QC2 policy（仅在复现 candidate 时）；
- C5 输入数据或等价的正式输入 package。

恢复后必须逐项核对 bytes 和 SHA256。不能因为文件名相同就视为同一资产，也不能在
缺失时回退到随机初始化、legacy rescue 或其他 checkpoint。

外部资产身份与 loader 依赖审计见：

- `docs/system/iotj_release_provenance_manifest_20260726.json`
- `docs/system/GAPS_RELEASE_PROVENANCE_20260726.zh.md`
- `scripts/verify_iotj_release_provenance.py`

当前 manifest 已覆盖 35 项资产。必须注意：v4 loader 会在启动时强制校验 1360 行
offline parity reference；现有 v4/v5 contracts 还含冻结绝对路径。因此 raw copy
不是 portable deployment，不能据此宣称 clean-checkout ready。

## 5. Full-system 阻塞项（不属于本轮范围）

1. Runtime-v4 portable loader/package；
2. Runtime-v5 QC2 candidate 的独立 reproduction closure；
3. 完整正式输入 package 的分发策略。

这些未完成项不影响 `CLEAN_CHECKOUT_RUNTIME_V5_CORE_READY`，但阻止声明 full system
ready。本轮按约束不继续开发它们。

## 6. Benchmark provenance

既有命令的只读重建见：

`docs/system/benchmark_command_manifest_20260725.json`

该 manifest 明确区分已记录参数与未知的原始 Python executable、shell quoting 和
working directory。它不替代正式 benchmark 结果，也不声称完整原始 shell invocation
曾被捕获。

## 7. 冻结边界

发布整理不得修改：

- 冻结模型或 checkpoint；
- 正式实验结果与论文数字；
- Runtime-v4 HC95/HC90；
- Runtime-v5 QC2 policy、threshold 或 decision gate；
- calibration/test protocol；
- claim–evidence 关系。
