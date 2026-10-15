# B5/C5 固定 H8 Runtime Parity 收口审计 — 2026-07-23

## 结论

正式链路已经实现为独立、版本化、fail-closed 的 C5/H8 runtime：

```text
B5 frozen classifier
  -> predicted class
  -> H2.3 deployment-visible risk expert
  -> fixed R4/H8 C5 Ridge
  -> frozen component-percentile risk
  -> manifest-selected HC95 or HC90
  -> accept / review / reject
  -> auto_output_ppm only for accept
```

HC95 是主工作点，HC90 是更严格的次工作点。两者的正式 1,360 行 parity 均为 `equivalent`。

| Workpoint | accept | review | reject | class mismatch | risk mismatch | QC mismatch | auto-output mismatch | max H8 delta | max calibrated-risk delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HC95 | 1323 | 33 | 4 | 0 | 0 | 0 | 0 | `6.252776074688882e-13 ppm` | `0` |
| HC90 | 1235 | 107 | 18 | 0 | 0 | 0 | 0 | `6.252776074688882e-13 ppm` | `0` |

原始风险分量存在 float32 计算差异，最大值为 `3.7956109789794024e-05`；冻结分位数校准后的三个风险分量和 `deployment_risk_full` 全部逐行一致，因此没有改变任何 QC 决策。

## 正式资产与版本边界

- v2：`blocked`，绑定了错误的 C5 数据根。
- v3：`superseded`，修正了数据根，但未哈希绑定 numeric phase labels。
- v4：唯一正式合同；绑定窗口、metadata、phase labels、HC95/HC90 reference 和 bundle manifest。
- v4 contract SHA-256：`54a42bb9f622c441a889a36fb1e585cb437e04c11128eb0578cfef6fd7711c3c`。
- row-map SHA-256：`7c37cc00d7fdb47e53130d5eeadea913ae96b88aeb8bfe3c6d081d9683a5fd35`。
- B5 checkpoint SHA-256：`9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`。
- parity 生成代码：`5ff301cee904e8eb7f4f3d605fc38898a83d325e`。
- parity evidence commit：`b7598c7`。

## 复用方式

同一 schema 下更换兼容基座时，只需生成新的非覆盖 bundle/runtime contract 并更新资产哈希，不需要修改 runtime 算法代码。以下变化必须使用新 schema 或单独适配器，不能放宽当前合同：窗口形状、类别映射、H2.3/R4 语义、风险校准语义或 QC policy schema。

v4 是本机绝对路径绑定的执行合同。换工作树或机器时应通过 `scripts/prepare_iotj_b5_c5_runtime_contract.py` 生成新的版本化合同；不得手工改写路径或哈希。

已有 parity 输出目录是非覆盖证据。不要在原目录重跑；仅当资产、代码、硬件或验证目的发生变化时，才创建新的版本目录。

## Evidence boundary

本次结果只证明冻结 offline reference 与部署 runtime 的数值/决策一致性。它没有重新训练或 refit 模型，没有产生新的算法性能结论，也不把旧 Pi classifier->R4、无 H2.3/risk/QC 的 preliminary 延迟数据提升为完整部署性能。

## Skill handoff

- `from_skill`: `experiment-registry`
- `to_skill`: `result-analysis`（仅在未来需要基于已确认 runtime CSV 做新的描述性分析时）
- completed checks: unique experiment IDs; source/target; checkpoint; dataset; seed; QC; paths; hashes; 1,360-row counts; parity status
- unresolved `unknown` / `conflict`: none for runtime parity
- blocking Evidence Gap: none for runtime parity; full-chain Pi latency remains a separate, unrequested measurement
- files that must remain read-only: bundle assets, v4 contract/map, HC95/HC90 references, parity runtime CSVs and reports
- requested next action: none; do not retrain or overwrite evidence
