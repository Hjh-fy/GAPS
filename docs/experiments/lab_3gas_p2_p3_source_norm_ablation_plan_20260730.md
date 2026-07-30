# P2→P3 Normalization Boundary Ablation

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| `H-LAB-NORM-01` | Z-score fit clients | historical `[1,2,3]` vs corrected `[2]` | raw sessions、fold groups、windowing、model、DA、seed、rounds、LE、batch | P3 exposure Macro-F1 | corrected protocol result；historical row仅作敏感性参考 | verify `norm_stats` equals P2 train moments and differs from all-client moments | fail closed on any identity, leakage or completion defect |

## Required baselines

旧 `[1,2,3]` 结果只用于说明协议修正影响，不作为严格公平 baseline，也不用于
选择阈值、round 或后续配置。

## Resource budget and execution order

1. 构建、验证、冻结 `client_data_lab_3gas_5fold_nominal_p2src_v2`；
2. 上传云 A 全数据和云 B client 2；
3. fold 1 preflight；
4. fold 1 25×3；
5. postflight/audit；
6. 在不改参数的前提下继续 folds 2–5。

## Unknown or conflicting protocol fields

- 精确通气边界：unknown；当前明确标记 nominal screening。
- source-only normalization 对准确率的方向：unknown。
