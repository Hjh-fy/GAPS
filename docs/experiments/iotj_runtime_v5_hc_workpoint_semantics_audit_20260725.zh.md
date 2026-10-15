# Runtime v5 HC95/HC90 工作点语义审计

历史 HC95/HC90 名称表示 calibration 风险分布上的目标工作点，不保证 test 实现
精确 95% 或 90% automatic yield。可复用的只有命名、输出 schema 和决策语义：

- HC95：risk 不高于 calibration q95 为 accept；高于 q98.75 为 reject；中间 review。
- HC90：risk 不高于 calibration q90 为 accept；高于 q97.5 为 reject；中间 review。
- automatic yield 只计算 accept；nonreject 为 accept + review。
- `auto_output_ppm` 只允许在 accept 行出现。
- 阈值必须在 calibration 上冻结，test 只用于一次性 generalization evaluation。

不可复用：v4 的 risk 数值、component calibrator、feature reference、分位分布、
disagreement 定义和具体阈值。v4 regression disagreement 依赖 H2/H3/H2.3，与仅含
Federated H1 的 v5 语义不兼容。本次 v5 所有风险资产必须重新由 320 calibration
行按 amendment v2 的 group-aware OOF 协议建立。
