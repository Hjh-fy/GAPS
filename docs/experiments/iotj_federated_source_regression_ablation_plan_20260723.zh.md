# IoT-J Federated Source Regression Ablation Plan

| Hypothesis ID | 因素 | Levels | 固定项 | 主指标 | 证据用途 | 停止规则 |
|---|---|---|---|---|---|---|
| FSRP-H1 | source prior presence | RS4 rich-only / RS0 pooled | B5、C5 split、Ridge grid、route | test ALL/per-gas RMSE | pooled-source generalization benefit；不参与 selection | 只报告 |
| FSRP-H2 | local expert prior | RS4 / RS1 | source architecture/init、C1/C2 isolation | calibration-validation ALL/per-gas RMSE | device-specific complementarity | RS1 不优于 RS4 则无 local-value evidence |
| FSRP-H3 | FedAvg prior | RS4 / RS2 | 109-tensor aggregation、sample weighting | calibration-validation ALL/per-gas RMSE | global prior independent value | RS2 不优于 RS4 则无 FedAvg-value evidence |
| FSRP-H4 | source topology | RS0 pooled / RS3 Local+FedAvg | B5、C5、Ridge grid、selection split | calibration-validation ALL RMSE | non-pooled parity | >10% 退化停止晋级；5%–10% 为 inconclusive |

## 必需基线

- RS0：冻结 centralized pooled H8/R4 source predictions。
- RS4：不含任何 source regression prediction 的 rich-features-only control。

## 执行顺序与资源预算

1. 运行相关测试。
2. C1/C2 各执行 100 total optimizer steps，round allocation 为 34/33/33。
3. 执行 3 次 regression-only FedAvg。
4. 通过 runtime v4 contract 加载 canonical B5 route，并要求冻结 RS0 validation/test route parity 为 80/80、1360/1360。
5. 只用 C5 calibration 内部 fit/validation 完成 alpha 与 candidate selection。
6. 写出并回读 `decision_gate.json`。
7. 一次性打开 C5 test 做 generalization evaluation。
8. 输出审计、机制和分气体分析后停止；不进入 QC/runtime/Pi/multi-seed/Flower network。

## 统计边界

本阶段只有一个 seed。窗口行不是独立 client，因此不把逐窗差异包装为跨 seed 显著性或稳定性证据；只报告描述性 RMSE、MAE、相对差异、source prediction Pearson correlation 与 disagreement。
