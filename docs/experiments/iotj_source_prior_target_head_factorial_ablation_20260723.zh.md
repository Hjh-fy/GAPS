# IoT-J Source-Prior × Target-Head Ablation Plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Confound check | Stopping rule |
|---|---|---|---|---|---|---|
| E1-H1 | source prior under Ridge | absent / H1+H2+H3 | B5、split、Ridge grid、route | calibration-validation ALL RMSE | Ridge prior branch must replay formal R4 | report only |
| E1-H2 | source prior under MLP | absent / H1+H2+H3 | B5、split、MLP grid/regularization/seed | calibration-validation ALL RMSE | only input dimension changes 104→107 | report only |
| E1-H3 | target head without prior | Ridge / MLP | 104D schema、split、route | calibration-validation ALL RMSE | same row keys | report only |
| E1-H4 | target head with prior | Ridge / MLP | 107D schema、formal H1/H2/H3 | calibration-validation ALL RMSE | no test selection | >5% marks candidate only |
| E2-H1 | source component | none / H1 / H2 / H3 / all | Ridge、B5、split、route、alpha grid | calibration-validation ALL/per-gas RMSE | component columns only | report only |

## 执行顺序

1. 验证 runtime v4、R4 policy、H2.3 protocol 和 frozen SHA。
2. 仅加载 C5 calibration，生成 oracle-route fit features 与 predicted-route validation features。
3. 对7个唯一 target-head配置完成 calibration selection/refit。
4. 写出并回读 gate。
5. 一次性加载 C5 test，计算 generalization metrics。
6. 审计、分析、提交结果索引与文档后停止。

## 统计边界

本实验只有一个 seed。逐窗数据不作为独立 seed/client 重复，不进行伪显著性检验；报告描述性 RMSE、MAE、NRMSE、相对改善和分气体差异。
