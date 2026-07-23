# IoT-J Source-Prior × Target-Head Factorial Plan

## 研究范围

- 在 frozen B5 seed42 classifier 和 C5 calibration/test 320/1360 split 下执行一次轻量 E1/E2 消融。
- runtime v4、QC、B5 classifier、正式 pooled-source H1/H2/H3 均为只读。
- 只训练新的 C5 target Ridge/MLP heads；不重新训练 classification 或 source models。
- 本实验是单 seed 回归结构完整性消融，不构成 runtime v5 或真实联邦回归证据。

## 假设

| ID | 可证伪假设 | Baseline | Intervention | Primary metric | Acceptance criterion |
|---|---|---|---|---|---|
| E1-H1 | H1/H2/H3 prior 对 Ridge 有增益 | Ridge rich | Ridge rich+prior | calibration-validation ALL RMSE | prior RMSE 更低 |
| E1-H2 | H1/H2/H3 prior 对 MLP 有增益 | MLP rich | MLP rich+prior | calibration-validation ALL RMSE | prior RMSE 更低 |
| E1-H3 | MLP 在无 prior 时优于 Ridge | Ridge rich | MLP rich | calibration-validation ALL RMSE | MLP RMSE 更低 |
| E1-H4 | MLP+prior 相对当前 Ridge+prior 改善超过 5% | Ridge rich+prior | MLP rich+prior | calibration-validation ALL RMSE | 改善严格大于 5% |
| E2-H1 | H1、H2、H3 对 Ridge 的贡献可分离 | Ridge rich | Ridge rich+H1/H2/H3/all | calibration-validation ALL/per-gas RMSE | 报告各 component 的描述性增量 |

## 冻结协议

- Source clients：C1、C2；target client：C5。
- C5 calibration：320；每气体80行，固定60 fit / 20 validation。
- C5 test：1360；只在 calibration gate 落盘并回读后打开。
- 路由：正式 runtime v4 canonical B5 predicted class。
- 104D rich feature schema：复用 `run_regression_head_ablation.rich_feature_dict`。
- H1/H2/H3：从正式 runtime v4 所绑定的 R4 policy 只读推理；禁止 refit source heads。
- Calibration fit 的 source-head route：true gas，与正式 R4 target-head fitting 一致。
- Calibration-validation/test 的 source-head route：B5 predicted gas，与部署可见路径一致。
- Ridge alpha grid：`0, 0.01, 0.1, 1, 10, 100, 1000`。
- MLP hidden grid：`(16), (32), (64), (32,16)`。
- MLP alpha grid：`0.001, 0.01, 0.1, 1`。
- MLP：ReLU、LBFGS、max_iter=800、early_stopping=false、seed=42；per-gas model seed 与正式 H2.3 相同。
- QC：off。

## E1 2×2 factorial

| Target head | Source prior absent | Source prior present |
|---|---|---|
| Ridge | 104D rich | 104D rich + H1/H2/H3 = 107D |
| MLP | 104D rich | 104D rich + H1/H2/H3 = 107D |

## E2 component ablation

- Ridge + rich only：104D。
- Ridge + rich + H1：105D。
- Ridge + rich + H2：105D。
- Ridge + rich + H3：105D。
- Ridge + rich + H1+H2+H3：107D。

## Gate

- `MLP+prior` 相对 `Ridge+prior` calibration-validation ALL RMSE 改善严格超过 5%：标记 `NEW_CANDIDATE_PENDING_CONFIRMATION`。
- 改善不超过 5%：`KEEP_RUNTIME_V4`。
- Test 只做一次 generalization evaluation，禁止改变 gate。
- 无论 gate 结果如何，本实验都不自动修改 runtime。

## 输出指标

- calibration-validation RMSE；
- test S_ALL RMSE/MAE/NRMSE；
- test S_CC RMSE；
- per-gas RMSE、CO RMSE、CO-high 200–250 ppm RMSE；
- target-head trainable parameter count；
- target input dimension；
- E1 四个 factorial effects。

## Fail-closed 条件

正式资产 SHA 改变、formal R4/H2.3 baseline replay 不一致、schema 不是104/105/107、row key 不一致、calibration split 不一致、test 在 gate 前参与 fit/select/refit、非有限预测、输出目录非空、local/origin commit 不一致时停止。
