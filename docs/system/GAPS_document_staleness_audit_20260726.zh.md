# GAPS 文档陈旧性审计（2026-07-26）

## 1. 审计范围与结论

- 审计状态：`PASS_WITH_STALE_ENTRY_DOCUMENTS`
- 当前实验范围：`NO_FURTHER_EXPERIMENTS_REQUIRED_FOR_CURRENT_SCOPE`
- 当前权威系统说明：`docs/system/GAPS_SYSTEM_OVERVIEW_20260726.zh.md`
- 当前权威代码图谱：`docs/system/GAPS_CODE_MAP_20260726.zh.md`
- 当前权威命令手册：`docs/system/GAPS_COMMAND_COOKBOOK_20260726.zh.md`
- 当前权威性能账本：`docs/system/GAPS_PERFORMANCE_LEDGER_20260726.zh.md`
- 本审计只读取代码、manifest、run config、result index 和 SHA 记录；没有训练、推理、重评估、benchmark 或 test 打开。

## 2. 逐文档审计

| 文件 | Git 最后更新 | 所代表的系统故事 | 状态 | 冲突或缺口 | 本轮动作 |
|---|---|---|---|---|---|
| `README.md` | 2026-07-15 | B2/B5 筛选、H2.3/H8、R3aK16/AutoV2、旧 QC | `STALE_ENTRY_UPDATED_WITH_CANONICAL_BANNER` | 把 H8 写成当前最佳主线；仍列旧 runtime 故事；分支指向旧工作树 | 顶部加入 current guide、最终角色和新文档链接；历史正文保留 |
| `代码文件介绍.md` | 2026-07-21 | 真实 Flower + H2.3/H8 + Spec A/代表性运行 | `STALE_ENTRY_UPDATED_WITH_CANONICAL_BANNER` | 写有“缺 seeds 43–46”“C5-only runtime 尚未完成”等已关闭缺口 | 顶部更新为 2026-07-26 canonical 状态，并链接代码图谱与归档清单 |
| `docs/mainline_entrypoints_20260626.md` | 2026-06-26 | R3aK16 → H2.3/H8/H8+C4 → AutoV2/QC | `LEGACY_SUPERSEDED` | 将 H2.3 定义为 default mainline，将 H8+C4 定义为部署候选 | 顶部加入 `LEGACY / SUPERSEDED`，正文仅作历史导航 |
| `docs/paper/iotj_system_methodology_20260711.zh.md` | 2026-07-15 | 修正版分类 + H2.3/H8/R4 + 旧 runtime 待闭环 | `LEGACY_METHOD_DEVELOPMENT_RECORD` | 写有 seeds 43–46、low-calibration、runtime parity 未完成；算法 2 仍是待实现合同 | 不改历史正文；禁止作为最终系统状态入口 |
| `docs/experiments/iotj_system_experiment_notebook.md` | 2026-07-23 | 逐阶段实验、失败、恢复和决策日志 | `CANONICAL_HISTORY_NOT_CURRENT_SUMMARY` | 末尾停在 source-regression screening；没有覆盖 7 月 24–26 最终闭环 | 保留为不可删实验账本；当前状态以新 system docs 和冻结 indexes 为准 |
| `docs/experiments/iotj_latest_handoff_20260715.zh.md` | 2026-07-21 | seed42、旧 H8/QC、Spec A 和代表性运行交接 | `LEGACY_SUPERSEDED` | 一句话状态仍称 seeds/runtime/benchmark 未完成；分支和下一步均已过时 | 顶部加入 superseded 标记和新入口；正文保留 |
| `gaps_deploy/final_runtime.py` docstring | 2026-07-14 | C12→C345、R3aK16、AutoV2、旧 QC wrapper | `CANONICAL_V4_BASELINE_CODE_WITH_LEGACY_INTERNALS` | 文件名容易被误认为“最终 v5”；docstring 未说明其现在是 v4 baseline | 不改 runtime 代码；在代码图谱中固定标为 Runtime v4 formal selective-output baseline |
| `docs/paper/GAPS_IoTJ_protocol_closed_20260726.zh.html` | 2026-07-26 | B5 + Federated H1 + 105D Ridge；v4 baseline；v5 QC2 未晋级 | `CANONICAL_PROTOCOL_CLOSED_MANUSCRIPT` | 未发现与当前系统角色冲突 | 保持只读 |
| `docs/experiments/iotj_manuscript_protocol_scope_lock_20260726.zh.md` | 2026-07-26 | calibrated-target held-out-window protocol 与实验范围关闭 | `CANONICAL_SCOPE_LOCK` | 未发现冲突 | 保持只读 |

## 3. 主要陈旧冲突

### DSTALE-01：旧回归主线覆盖新最终方法

旧文档将 `R3aK16 -> auto_v2 -> H2.3/H8 -> QC` 写成当前系统。当前正式角色已经变为：

```text
B5 classifier
-> calibration-assisted server DA
-> sufficient-statistics Federated H1
-> C5 104D rich + 1D H1
-> C5 105D per-gas Ridge
```

旧 H2.3/H8/H8+C4 仍有历史复现价值，但不能作为唯一主线。

### DSTALE-02：已完成事项仍被写成缺口

以下缺口已经关闭：

- B5 seeds 42–46；
- regression multi-seed；
- Federated H1 practical equivalence；
- Runtime v5 regression core；
- 320/1360 runtime parity；
- Runtime v5 QC2 audit；
- PC/Pi benchmark；
- low-calibration sensitivity；
- paper evidence freeze 与 protocol close。

### DSTALE-03：Runtime 版本角色混淆

- `gaps_deploy/final_runtime.py`：Runtime v4 的历史 package wrapper，不是 v5。
- `gaps_deploy/c5_h8_runtime.py`：当前正式 v4 selective-output baseline 的 C5 H8 runtime。
- `gaps_deploy/c5_federated_source_ridge_runtime.py`：最终简化 regression core。
- `gaps_deploy/c5_federated_source_ridge_qc_runtime.py`：v5 QC2 candidate wrapper；有效但未晋级。

## 4. 文档优先级

发生冲突时按以下顺序解释：

1. `docs/paper/GAPS_IoTJ_protocol_closeout_index_20260726.json`
2. `docs/experiments/iotj_manuscript_protocol_scope_lock_20260726.zh.md`
3. `docs/system/GAPS_SYSTEM_OVERVIEW_20260726.zh.md`
4. `docs/system/GAPS_CODE_MAP_20260726.zh.md`
5. `docs/system/GAPS_COMMAND_COOKBOOK_20260726.zh.md`
6. `docs/system/GAPS_PERFORMANCE_LEDGER_20260726.zh.md`
7. 各正式 result index / protocol manifest / SHA index
8. 旧 handoff、旧 methodology、旧 mainline entrypoints

## 5. Verdict

当前代码与冻结证据能够支持最终系统说明；主要问题是入口文档陈旧，而不是正式结果缺失。本轮通过醒目标记和新权威文档消除入口歧义，不修改历史记录或冻结数字。
