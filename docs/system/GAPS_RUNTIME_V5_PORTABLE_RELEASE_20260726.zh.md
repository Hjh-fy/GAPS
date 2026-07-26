# GAPS Runtime-v5 Core Portable Release 收口

日期：2026-07-26

最终状态：

```text
CLEAN_CHECKOUT_RUNTIME_V5_CORE_READY
```

本状态只覆盖 B5 classifier → sufficient-statistics Federated H1 → C5 105D
per-gas target Ridge 的 Runtime-v5 regression core。

不得扩展解释为：

```text
FULL_V4_V5_QC_REPRODUCTION_READY
CLEAN_CHECKOUT_FULL_SYSTEM_READY
```

## 1. 冻结边界

本轮以 commit
`4d0e6b84341142a80ffd265e2e95dcda06fd1c72` 为冻结起点，没有修改：

- 原 frozen Runtime-v5 contract；
- 原 frozen bundle manifest；
- B5 checkpoint；
- Federated H1 或 target Ridge；
- calibration/test protocol；
- Runtime-v4、HC95/HC90；
- Runtime-v5 QC2 policy、threshold 或 decision gate；
- 正式实验结果、论文数字和 paper evidence。

没有运行训练、正式 test、评价或 benchmark。

## 2. Protocol closeout safety

`scripts/close_iotj_manuscript_protocol.py` 已改为默认 `--verify-only`。

默认执行结果：

| 文件 | SHA256 |
|---|---|
| protocol-closed HTML | `3e23158f3772865da6e804d5799ad5a6988de7cb367637abc44c0e67d61881fa` |
| legacy ablation table | `74171741d904b98579dfa78b636b353f91ed2d901b35be001d1ad2fc442ff7b5` |
| protocol closeout index | `5d53ed23816830bd4b11678d4f04c2adbff594f1decd443436f394b138d41196` |

验证报告 `files_written=0`。生成模式必须显式提供：

```powershell
python scripts/close_iotj_manuscript_protocol.py --generate `
  --output-html <new-output.html> `
  --table <new-table.csv> `
  --index <new-index.json>
```

任一目标已存在时返回 `REFUSE_TO_OVERWRITE`。本轮没有重新生成或覆盖正式
protocol-closed 文件。

## 3. Portable binding

新增 schema：

```text
gaps.runtime_v5.portable_binding.v1
```

binding 使用相对于 release root 的 POSIX paths，只绑定以下四项：

| 资产 | Bytes | SHA256 |
|---|---:|---|
| B5 classifier | 184,201 | `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c` |
| Federated H1 | 50,226 | `1ca10939f09e744fdddc0dce6f5fd959ccf769e9b78142030aa7e50aa6b2f3d4` |
| C5 target Ridge 105D | 46,151 | `2039d049776e7dfe0e8c4e6405dff2ae56a6e09b63f60ff2627ac0975aa075de` |
| calibration lock | 2,266 | `4edf75222e41d8bf43097625a076964e9338493478edc76c9a04a08794d5affe` |

同时绑定原证据身份：

- frozen Runtime-v5 contract SHA256：
  `bca1471198f0505d4536fba71100e87279156a0c69fdd54d300ffad991b36482`
- frozen bundle manifest SHA256：
  `f239c3b4929d1748574ec4d2fe4f61e09030087c2104b8c726046b5a39bffb1e`

portable binding 是独立工程派生物，不修改、不覆盖也不替代原 frozen contract。

## 4. Release archive

目录：

```text
release/gaps_runtime_v5_core_20260726/
```

确定性 ZIP：

```text
release/gaps_runtime_v5_core_20260726.zip
```

| 字段 | 值 |
|---|---|
| archive bytes | 298,228 |
| archive SHA256 | `740e8237384041523e51969b88795c27e43e88650c73e1a5209092880cf547de` |
| portable binding SHA256 | `4a3c678d629359ac40df98496f13226bfe2199414c795d483bbbd514828b973d` |
| payload tree SHA256 | `e6fa5cfdf93f4e42ffc5cd429854e40abaf0944c975c68b4cb00882a2e5ca8e5` |
| SHA-verified file count | 12 |

archive 包含：

- portable binding；
- 四项冻结资产；
- provenance mapping；
- archive manifest；
- `SHA256SUMS`；
- README；
- 一组 1×100×8 合成输入、metadata、phase 和 expected output schema。

archive 不包含：

- C5 formal test features、metadata 或 phase labels；
- test labels；
- HC95/HC90 test records；
- offline formal predictions；
- Runtime-v4 assets；
- Runtime-v5 QC policy。

## 5. Thin inference CLI

新增入口：

```powershell
python -m gaps_deploy.runtime_v5_cli
```

验证 binding 与四项资产：

```powershell
python -m gaps_deploy.runtime_v5_cli `
  --contract release/gaps_runtime_v5_core_20260726/portable_binding.json `
  --verify-only
```

描述合同：

```powershell
python -m gaps_deploy.runtime_v5_cli `
  --contract release/gaps_runtime_v5_core_20260726/portable_binding.json `
  --describe-contract
```

推理：

```powershell
python -m gaps_deploy.runtime_v5_cli `
  --contract <portable_binding.json> `
  --input <N_by_100_by_8.npy> `
  --metadata <metadata.json> `
  --phase-file <phase.npy> `
  --output <new-output.json> `
  --device cpu
```

CLI 只通过 `C5FederatedSourceRidgeRuntime` 运行现有分类、H1 和 target Ridge
逻辑。没有复制或重写模型、特征或回归预测算法。

下列情况均 fail closed：

- binding 或资产缺失；
- bytes/SHA256 不符；
- 非相对路径或路径逃逸；
- 非 `N×100×8` 输入；
- metadata/phase 未对齐；
- phase 非整数 0–2；
- NaN/Inf；
- 输出 schema 或数值非法；
- 输出文件已经存在。

Runtime-v5 core 的 `qc_status` 保持
`disabled_pending_dependency_audit`，`auto_output_ppm` 必须为 `null`。

## 6. Fresh-checkout synthetic smoke

受测代码 commit：

```text
adb3c7103af9e9fb24609fda0d10d5e01aa314a9
```

正式回执：

```text
docs/system/iotj_runtime_v5_clean_checkout_receipt_20260726.json
```

验证环境：

- Windows 11；
- Python 3.12.4；
- NumPy 1.26.4；
- PyTorch 2.5.1；
- CPU device。

验证步骤全部通过：

| Check | 结果 |
|---|---|
| fresh fixed-commit checkout | PASS |
| archive SHA256 + sidecar | PASS |
| restore to new directory | PASS |
| SHA256SUMS + archive manifest | PASS |
| runtime import | PASS |
| portable binding verify | PASS |
| runtime load | PASS |
| one-row synthetic inference | PASS |
| output schema | PASS |
| checkout clean after smoke | PASS |

合成输出为 1 行，预测 route 为 class 3；该数值只证明执行链路与 schema 可用，
不是性能评价，也不得写入论文结果。

首次尝试在原长路径下 checkout 时触发 Windows filename-too-long 并在 runtime
加载前停止；没有生成正式回执。随后使用新的短临时路径、相同 commit 和相同 archive
重新执行并通过。该异常不涉及资产、算法或实验协议变化。

## 7. Evidence boundary

本轮正式边界为：

```text
formal test accessed = false
training run = false
evaluation run = false
benchmark run = false
frozen results modified = false
runtime-v4 portability claimed = false
runtime-v5 QC reproduction claimed = false
full-system ready claimed = false
```

fresh checkout 不包含 Git 忽略的 dataset/results 外部资产；restore archive 的文件
membership、明确禁止路径和命令输入均已审计。正式 1360 行 test、HC95/HC90
records 与 offline predictions 没有被打开。

## 8. 最终结论

Runtime-v5 regression core 已完成最小 portable release closure。工程开发在此停止，
后续转入中文论文投稿候选稿，不继续 v4 portable、QC、REST、GUI、Docker、ONNX、
量化、benchmark 或实验开发。
