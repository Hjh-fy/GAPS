# GAPS IoT-J calibration-protocol harmonization audit（2026-07-26）

## 结论

描述性判定为 `SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT`。该判定不修改既有 `HIGH_CALIBRATION_SENSITIVITY` 记录、最终方法、runtime 或 QC。

| Calibration rows | Historical S_CC RMSE | Group-aware S_CC RMSE | G−H (ppm) |
|---:|---:|---:|---:|
| 320 | 11.3416 ± 0.0000 | 10.8724 ± 0.4391 | -0.4692 |
| 160 | 17.8621 ± 1.5586 | 23.9156 ± 5.2550 | 6.0536 |
| 80 | 28.4545 ± 3.3698 | 30.4799 ± 4.3262 | 2.0254 |
| 40 | 43.2442 ± 10.9745 | 36.4992 ± 2.7156 | -6.7449 |

历史 240/80 审计：fit 240 行、validation 80 行；fit filename=80，
validation filename=61，跨池 filename overlap=61。
因此历史轨迹是 window-level holdout，不能声称 original-file independent 或 group-aware。

Group-aware 320 的标准差来自 fold/alpha-selection variability；group-aware 160/80/40 来自
subset + fold variability；historical 160/80/40 来自 holdout subset variability；historical
320 是固定单次 reference。

## Evidence boundary

本工作是 post-freeze calibration-protocol harmonization audit。两条轨迹使用同一冻结 B5、
Federated H1、105D schema、Ridge family 与 alpha grid。相同且此前已经使用过的 C5 1360 行
test 仅作描述性评估；没有根据 test 选择 protocol、subset、fold、alpha 或模型。
