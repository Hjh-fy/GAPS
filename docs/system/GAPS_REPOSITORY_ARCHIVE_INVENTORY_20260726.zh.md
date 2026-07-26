# GAPS 仓库目录归档清单（2026-07-26）

## 1. 使用方式

本清单用于整理当前 worktree，不授权自动删除。分类优先级：

| 分类 | 含义 | 建议 |
|---|---|---|
| `KEEP_CANONICAL` | 当前系统、论文或正式 SHA 链必需 | 原位保留并纳入备份 |
| `KEEP_REPRODUCIBILITY` | 历史/legacy 复现和失败审计所需 | 移入只读 archive，不要散放主目录 |
| `ARCHIVE_EXTERNAL_FIRST` | 大型 checkpoint/raw/log；可能是唯一原始证据 | 先复制到外部归档并生成 SHA 清单，验证后再精简工作树 |
| `REGENERATABLE_DELETE_CANDIDATE` | pytest/cache/smoke 临时产物 | 确认无进程占用后可删除 |
| `MANUAL_REVIEW` | 未跟踪源码、未提交文档或有本地修改 | 逐个决定 commit/archive，禁止批量删除 |

## 2. 必须保留的顶层代码

### `KEEP_CANONICAL`

- `gaps_flower/`：B5 classification、server DA、observability；
- `gaps_deploy/`：v4/v5 runtime、bundle、QC；
- `scripts/`：正式 manifest/controller/evaluation/bundle/parity/benchmark/reporting；
- `tests/`：contract/runtime/bundle/parity 回归测试；
- `configs/`；
- `model.py`、`config.py`、`federated_dataset.py`、`utils.py`；
- `README.md`、`代码文件介绍.md`；
- `docs/system/`、`docs/paper/`、`docs/paper_evidence_freeze/`、`docs/experiments/`。

### `KEEP_REPRODUCIBILITY`

以下顶层旧入口不应再当主线，但对历史复现仍有价值：

- `exp_improved.py`；
- `preprocessor.py`；
- `gaps_flower/regression_task.py`、`regression_client.py`、`regression_server.py`；
- 根目录 `run_*H2.3/H8/H8+C4/AutoV2*` 相关脚本；
- 2026-06-25 至 2026-07-15 的 legacy result summaries。

建议未来统一移动到 `archive/legacy_code/` 或独立 Git tag；本轮不移动。

## 3. 当前正式结果根

### `KEEP_CANONICAL`

| 目录 | 作用 |
|---|---|
| `results/iotj_b5_multiseed_20260724/` | B5 seeds 42–46 checkpoints、routes、分类汇总 |
| `results/iotj_h1_federated_ridge_equivalence_20260724/` | H1 practical-equivalence 审计 |
| `results/iotj_b5_regression_multiseed_20260724/` | RG0/RG1/RG2 五种子与最终 regression gate |
| `results/iotj_b5_c5_runtime_v5_candidate_20260724/` | real-topology H1、105D target Ridge、v5 bundle/parity |
| `results/iotj_b5_c5_runtime_v5_qc_20260725/` | v5 QC2 OOF、lock、test 与 parity |
| `results/iotj_final_system_benchmark_20260725/` | PC/Pi benchmark、参数、包大小和通信 |
| `results/iotj_low_calibration_sensitivity_20260725/` | 冻结方法 calibration-budget sensitivity |
| `results/iotj_calibration_protocol_harmonization_20260726/` | post-freeze protocol harmonization |
| `results/iotj_b5_c5_deployment_p1_20260722/` | Runtime v4 frozen contract/HC assets |
| `results/c2e_summary/`、`results/c2e_commands/`、`results/c2e_ecs_c2_topology/` | 分类冻结协议、命令和拓扑 |

这些目录中的轻量 index/summary 已有 Git provenance，但大型 checkpoint/raw/log 可能被 `.gitignore` 排除。不要仅因 `git status` 不显示就认为它们已上传 GitHub。

### `ARCHIVE_EXTERNAL_FIRST`

本 worktree 当前可见的大型正式/历史 raw：

- `results/iotj_b5_multiseed_20260724/`：约 194 MB 可见文件；
- `results/iotj_ecs_c2_representative_20260720/`：约 143 MB；
- `results/iotj_main_confirmation_observability_20260715/`：约 293 MB；
- `results/iotj_b5_c5_deployment_p1_20260722/`：约 66 MB。

归档步骤应为：

1. 复制整个 result root 到外部只读存储；
2. 保留目录结构；
3. 对所有 regular files 生成 SHA256 index；
4. 验证外部副本；
5. 在仓库保留 Git-tracked summary/index/README；
6. 最后才考虑移除本地 ignored raw/checkpoints。

## 4. 可重建临时目录

### `REGENERATABLE_DELETE_CANDIDATE`

当前未跟踪、名称和内容均对应测试/smoke scratch：

- `.tmp_b5_regression_multiseed_smoke/`
- `.tmp_h1_federated_ridge_equivalence_smoke/`
- `.tmp_pytest_lowcal_*`
- `.tmp_runtime_v5_qc_smoke*`
- `.tmp_v5_h1_topology_smoke*`
- `.p1_pytest_bundle_final/`
- `.pytest_cache/`
- `__pycache__/`

当前最大的重复组是：

- `.tmp_iotj_observer_gate_b2_*`：多版约 6.6–7.4 MB/目录；
- `.tmp_iotj_observer_gate_b5_*`：多版约 6.6–7.4 MB/目录。

这些目录是 synthetic/local-only/unstaged gate 产物。最终 v10 的结论已经写入历史文档，但若仍需保留一次本地复盘证据，可只外部归档 `*_task9_final_v10/`，其余版本作为删除候选。删除前仍应确认没有未提交引用。

### 失败 manuscript 临时产物

以下不是正式 paper freeze：

- `.paper_evidence_freeze_failed_backup*.html`
- `.paper_evidence_freeze_failed_manuscript*.html`
- `.paper_evidence_freeze_failed_validation*/`

正式 evidence-frozen 与 protocol-closed 文件已经有 Git SHA。若不再需要失败复盘，这组可归入删除候选；若需要保留故障史，压缩到 `archive/failed_paper_freeze_20260726/`。

### Benchmark staging

`.final_benchmark_pi_staging/` 约 13.8 MB，是 Pi 便携 benchmark staging。正式 benchmark 结果已冻结，但该目录可能用于发布包复核，建议先归档，不直接删除。

## 5. 必须人工处理的未跟踪文件

### `MANUAL_REVIEW_KEEP_OR_COMMIT`

- `scripts/close_iotj_manuscript_protocol.py`：已提交 protocol-close 文档的生成器，建议单独审查后纳入 Git；
- `scripts/audit_iotj_minimal_experiment_gap.py`；
- `docs/experiments/iotj_minimal_experiment_gap_audit_20260726.zh.md`；
- `docs/experiments/iotj_minimal_experiment_gap_audit_index_20260726.json`；
- `docs/submission_preparation/`：投稿准备文档；
- `scripts/benchmark_iotj_b5_classifier_r4_preliminary.py`：仅 preliminary/diagnostic，建议移入 legacy archive；
- `scripts/diagnose_iotj_source_tree_manifest.py`：诊断脚本，建议移入 diagnostic archive；
- `docs/experiments/iotj_handoff_20260722.zh.md`：已有 `iotj_latest_handoff_20260722.zh.md`，先对比后归档。

### 已有本地修改，禁止覆盖

- `results/iotj_a003_timing_diagnosis_20260719/a003_vs_b2_pilot_timing_analysis.md`
- `results/iotj_advisor_metrics_20260721/build_advisor_workbook_v3.mjs`

这两个文件属于既有用户改动，不纳入本轮文档提交。

## 6. Legacy result 归档建议

以下结果仍有论文历史、消融或故障复盘价值，但不应与 final evidence 混放：

- `*_20260625`、`*_20260626` 的 H2.3/H8/H8+C4/AutoV2/L1/L2/L3 结果；
- `iotj_classification_ablation_20260711_v2r1_summary/`；
- `iotj_classification_ablation_20260712_v3_summary/`；
- cross-direction seed42 summaries；
- `iotj_preliminary_paper_metrics_20260717/`；
- `iotj_a003_timing_diagnosis_20260719/`。

建议逻辑归档目录：

```text
archive/
  legacy_algorithms_20260625_20260715/
  failed_and_diagnostic_attempts/
  preliminary_reports/
  external_large_artifacts/
```

不要在未建立 SHA relocation manifest 前实际移动 Git-tracked 文件；移动会造成大量历史路径失效。更稳妥的是先用 Git tag 固定当前树，再在新整理分支做 `git mv` 和链接索引。

## 7. 建议的两阶段整理

### 第一阶段：低风险清理

1. 备份当前 worktree；
2. 清除 `.pytest_cache`、`__pycache__`、确认无用的 `.tmp_pytest_*`；
3. observer gate 只保留外部归档的一份 v10；
4. 处理 failed paper freeze 临时文件；
5. 不碰任何 `results/iotj_b5_*`、runtime v4/v5 或 paper freeze。

### 第二阶段：结构化归档

1. 对未跟踪脚本逐个决定 commit 或 archive；
2. 为大型 results 建外部 SHA 索引；
3. 将 legacy 文档/代码通过 Git history-preserving move 整理；
4. 更新所有链接；
5. 运行文档链接和 contract tests；
6. 再删除已验证可恢复的本地 raw 副本。

## 8. 当前不能称为“无用”的内容

以下即使不参与最终算法，也不能直接删除：

- failed/invalid real-device attempts；
- checkpoint SHA 索引指向的本地模型；
- calibration/test locks；
- parity row streams；
- source sufficient-statistics payload；
- runtime v4 六个冻结资产；
- legacy ablation summaries；
- protocol-close 与 paper evidence freeze。

它们分别承担失败审计、可复现性、论文边界或最终系统对照角色。
