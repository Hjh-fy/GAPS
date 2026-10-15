# GAPS 最终系统开放问题（2026-07-26）

## 结论摘要

12 个问题中，8 个已由代码或 Git 证据回答，4 个存在工程可复现性/命名层面的待处理项。所有待处理项都不要求新训练或重新打开 test。

## 逐项回答

### 1. 哪个脚本正式构建 Runtime v5 bundle？

**已确认。**

`scripts/build_iotj_runtime_v5_candidate.py build-bundle`

其 `build_bundle()` 只打包：

- canonical B5 seed42 checkpoint；
- real-topology Federated H1 JSON；
- C5 `target_ridge_105d_manifest.json`。

输出 `runtime_v5/bundle_manifest.json` 和 `runtime_v5/runtime_contract_v5.json`。

### 2. 哪个脚本把 federated_h1_manifest 转成 runtime asset？

**已确认，但名称需区分。**

- `scripts/materialize_iotj_federated_h1_topology.py` 生成 real-topology `global_h1_model.json`；
- `scripts/build_iotj_runtime_v5_candidate.py build-bundle --real-h1 <global_h1_model.json>` 将其复制为 `runtime_v5/assets/federated_h1.json`。

旧审计文件 `results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json` 是 equivalence reference，不是最终 runtime 中实际打包的 real-topology H1 字节。

### 3. 哪个脚本构建 target_ridge_105d.json？

**已确认。**

- `build_iotj_runtime_v5_candidate.py freeze-calibration` 拟合并写入 `target_ridge/target_ridge_105d_manifest.json`；
- `build-bundle` 将其复制为 `runtime_v5/assets/target_ridge_105d.json`。

### 4. Runtime v5 是否已有正式 CLI？

**没有独立 inference CLI。**

`gaps_deploy/c5_federated_source_ridge_runtime.py` 只有 Python class API：

```python
runtime = C5FederatedSourceRidgeRuntime.from_runtime_contract(contract, device="cpu")
rows = runtime.infer(windows, metadata, phases)
```

benchmark 脚本可加载它，但 benchmark CLI 不是通用部署 inference CLI。

- missing evidence：无 `main()`/argparse 单窗或 batch runtime-v5 CLI；
- recommended owner：部署工程维护者；
- next inspection action：若投稿后需要交付 CLI，新增只读 wrapper，并用现有 parity tests 锁定输出；当前 scope 不实施。

### 5. 最终 PC/Pi benchmark 脚本是什么？

**已确认。**

- package：`scripts/prepare_iotj_final_benchmark_package.py`
- steady-state：`scripts/benchmark_iotj_final_runtime.py`
- cold start：`scripts/probe_iotj_runtime_cold_start.py`
- evidence 汇总：`scripts/build_iotj_final_system_evidence.py` 与 `scripts/finalize_iotj_final_system_evidence.py`

正式 code commit：`4ccfc489821410ddacb6ad36180694bb953311f1`。

### 6. Runtime v4 是否可从 clean checkout 完整执行？

**否，代码可导入但正式资产不完整随 Git 分发。**

`gaps_deploy/c5_h8_runtime.py` 和相关合同代码被跟踪，但大型 classifier、bundle/reference/policy 资产采用 SHA 索引绑定，部分只在正式本地/远端结果位置。clean checkout 需要恢复被 manifest 绑定的 external assets 后才能完整执行正式 v4。

- missing evidence：没有一个仅依赖 Git tracked bytes 的 v4 formal bundle；
- recommended owner：artifact/release 维护者；
- next inspection action：发布时制作只读 release archive，验证全部 manifest SHA 后再声明 clean-checkout runnable。

### 7. `gaps_deploy/final_runtime.py` 应标为 v4/legacy 还是重命名？

**标为 maintained legacy C12→C345 package wrapper，不建议当前重命名。**

它的内部 R3aK16/AutoV2 schema 和 package contract 是历史实现。正式 C5
Runtime-v4 selective-output baseline API 是
`gaps_deploy.c5_h8_runtime.C5H8Runtime`，不能把 `final_runtime.py` 本身写成正式
C5 v4 唯一入口。重命名会破坏历史命令与 import；当前通过文档显式标记即可。

### 8. README 中哪些旧命令已不存在？

**部分文件仅存在于历史工作树或从未进入当前 tracked tree。**

README 旧章节引用的 `diagnostics/` 目录不是当前 tracked 顶层目录；若干 `run_time_aware_*`、`audit_time_aware_*` 名称也不是当前分支的 tracked entrypoint。旧命令不得复制为正式命令，完整正式命令只看 command cookbook。

### 9. 论文系统框图与实际 code call graph 是否一致？

**protocol-closed 稿在角色层面一致。**

其 B5 → server DA → sufficient-statistics H1 → 105D target Ridge → runtime/QC 结构与代码一致。边界是：

- runtime v4 和 v5 QC2 是两条并列部署证据，不是 v5 QC2 已替代 v4；
- H1 是充分统计量重构，不是 Flower regression transport；
- 不提供 secure aggregation/DP。

旧 `iotj_system_methodology_20260711.zh.md` 的 H2.3/H8 图不再代表最终 call graph。

### 10. protocol-close 本地文件是否都进入 Git？

**五个正式 protocol-close 文件已进入 Git。**

commit：`2e0fe7c985fc2f487863f65a21f957a4c33e82bd`

该提交精确包含 protocol-closed HTML、closeout index、legacy table、scope lock 和 number audit。当前另有 untracked minimal-gap/submission-preparation 文件，它们不属于上述五文件 closeout commit。

### 11. 是否存在只在本地、没有 Git provenance 的关键命令？

**原始 shell 捕获仍有边界，但 provenance 缺口已在 2026-07-26 文档收口中处理。**

- B5 seeds 43–46 controller launch `.cmd` 已进入 Git；
- server/client argv 已进入 command manifests 和 run_config；
- H1/runtime-v5/benchmark 的代码与结果 manifest 已进入 Git；
- 最终 benchmark 原始逐平台 shell invocation 没有被完整捕获；现已新增
  `docs/system/benchmark_command_manifest_20260725.json`，明确以 argparse、
  protocol、result records 和保留的 Pi package 做只读重建，并保留未知字段；
- `scripts/close_iotj_manuscript_protocol.py` 已通过静态安全/provenance 审查，
  本轮决定纳入 Git；它只做冻结 SHA 守卫和确定性文档生成。

recommended owner：实验 registry / release 维护者。
next inspection action：release 阶段补齐 external asset archive 与 clean-checkout
receipt；不重新运行 benchmark。

### 12. 是否存在两个不同 checkpoint 被称为 final B5？

**是，必须持续区分。**

- historical corrected B5 screen：旧单种子筛选，Accuracy 约 `0.988971`；
- canonical final B5 seed42：SHA256 `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`，Accuracy `0.980147`；
- final multi-seed 还包含 seeds 43–46 各自 checkpoint，但 runtime 固定复用 seed42，不按 test 选择最佳 seed。

后续文档必须写 `B5 (v3 screen)` 或 `Final B5 (seed42, SHA...)`，不得只写“B5 checkpoint”。
