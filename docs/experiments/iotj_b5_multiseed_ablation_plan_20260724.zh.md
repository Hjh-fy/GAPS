# B5 multi-seed regression ablation plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| MS-B5-CLS | classifier training seed | 42,43,44,45,46 | dataset、split、topology、25 rounds、5 epochs、batch32、Adam 5e-4、B5/DA flags | Accuracy/Macro-F1/NLL/ECE | five-seed descriptive stability | normalized command manifest 除 seed/派生标识外必须一致 | 任一正式 seed 不完整则不生成五种子汇总 |
| MS-RG-H1 | target regression prior | RG0 rich; RG1 fed-H1; RG2 all-prior | 同 seed B5 route、C5 calibration/test、Ridge grid/refit | S_CC RMSE | paired five-seed RG1 vs RG2 | 同 seed 共享 route 与 S_CC subset | 按预注册 M4 gate 二选一 |
| MS-COMPONENT-FREEZE | source-head reuse | fixed fed-H1; fixed pooled H1/H2/H3 | C1/C2 source data/split/assets | source asset SHA | five seeds 使用同一 hash | 禁止按 classifier seed 重训 source heads | hash drift 即停止 |

## Required baselines

- seed42 canonical B5；
- RG0_RICH_ONLY；
- RG2_ALL_PRIOR；
- RG1_FEDERATED_H1。

## Resource budget and execution order

M0 → seed43 → validator → seed44 → validator → seed45 → validator → seed46
→ validator → M2 → M3 → M4 → stop。

seed42 实测训练 1.65 h；每个新 seed 预算约 2 h，四个顺序运行预计约 8 h，
预留 10 h。禁止并行。

## Unknown or conflicting protocol fields

- 启动时三机可达性：`unknown`，由 preflight 解决；
- 旧 command manifest 的 `client_c2_pc` 是历史模板字段；正式 seed42 实际
  topology 证据为 ECS-C2。新运行必须由 controller 的 `--c2-host`、
  `--c2-python`、`--c2-data-root` 覆盖到同一 ECS-C2。
