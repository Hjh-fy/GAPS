# CLS-FlowerExpB-TimeAware2080 / REG-R3aK16-AutoV2-TimeAware2080 冻结手册

更新日期：2026-06-19

本文档用于冻结当前最优主线，避免后续分类、回归、QC、部署实验误用旧数据、旧 checkpoint、旧 routing config 或旧 QC policy。

当前冻结只适用于：

```text
Source clients: C1, C2
Target clients: C3, C4, C5
Data protocol: time-aware 60-170 window-fullgrid
Target protocol: calibration:test = 2:8, target train = 0
```

只要更换 source clients、target clients、target calibration ratio、数据预处理协议或窗口协议，必须重新生成数据、重新训练 Flower 分类器、重新训练/校准回归 package，并重新选择 QC 策略。

---

## 1. 冻结名称

当前开始统一使用下面三个名称：

| 层级 | 冻结名 | 含义 |
|---|---|---|
| 分类基座 | `CLS-FlowerExpB-TimeAware2080` | Flower ExpB strong_cls + strong DA + adapted logits |
| 回归基座 | `REG-R3aK16-AutoV2-TimeAware2080` | R3aK16 source C1/C2 FedAvg + per-client auto_v2 specialist + QC |
| 系统主线 | `GAPS-TimeAware2080-FlowerCLS-R3aK16-AutoV2-QC` | 当前分类、回归、QC 联合最优链路 |

报告里不要再写模糊表述，例如“Exp B 分类器”“昨天回归”“strong classifier”“auto_v2 package”。正式写法应固定为上表名称。

---

## 2. 数据协议与正式目录

正式数据目录：

```text
dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid
```

生成与审计依据：

| 项目 | 当前值 |
|---|---|
| 预处理主线 | `preprocessor_time_aware.py` |
| 数据/split 主入口 | `run_time_aware_target_split_ablation.py` |
| split manifest | `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid/split_protocol_manifest.json` |
| split audit | `results/timeaware_2080_flower_expB/split_audit.md` |
| base protocol | `window_level_fullgrid` |
| formal protocol | `role_aware_target_8_2` |
| source split | train:test:calibration = 7:2:1 |
| target split | train = 0, test:calibration = 8:2 |
| stratify keys | gas + concentration |
| seed | 42 |
| file overlap | allowed, window-level fullgrid |

Target split audit 已通过：

| Client | Calibration N | Test N | Calibration ratio | Test ratio |
|---|---:|---:|---:|---:|
| C3 | 680 | 2680 | 0.2024 | 0.7976 |
| C4 | 320 | 1360 | 0.1905 | 0.8095 |
| C5 | 320 | 1360 | 0.1905 | 0.8095 |

C3/C4/C5 的 calibration/test 均覆盖 4 类 gas 和 10 个浓度水平。C3/C4/C5 的 train split 为 0，不得作为 Flower training client 启动。

---

## 3. 正式预处理主线

必须使用：

```text
preprocessor_time_aware.py
run_time_aware_target_split_ablation.py
```

`preprocessor_time_aware.py` 的作用：

- 读取原始 txt 时间列和 8 路传感器。
- 清理重复 timestamp。
- 按真实秒级时间重采样到 10Hz。
- 计算相对电导 `deltaG/G0`。
- 检测响应事件 `t_onset` / `t_min`。
- 固定裁剪 60-170s。
- 生成 100 x 8 window。
- 记录 window-level metadata，例如 `window_start_s`、`window_end_s`、`window_center_s`、`response_phase`、`crop_mode`、`interpolated_ratio`。

仅作 legacy / 对照：

```text
preprocessor.py
split_dataset.py
scripts/create_c12src_c345tgt_calib20_test80.py
dataset/client_data_c12src_c345tgt_calib20_test80
dataset/client_data_federated_window_fullgrid_src12_tgt345
```

这些不是当前最优主线的数据生成路径。旧目录结果可作工程对照，但不要作为正式主结果继续推进。

---

## 4. 分类冻结条件：CLS-FlowerExpB-TimeAware2080

### 4.1 使用代码

分类主线必须使用：

```text
gaps_flower/server_app.py
gaps_flower/client_app.py
gaps_flower/strategy.py
gaps_flower/domain_adaptation.py
gaps_flower/evaluate_checkpoint.py
model.py
config.py
federated_dataset.py
utils.py
```

当前服务器脚本记录：

```text
results/timeaware_2080_flower_expB/sync/run_expB_server_timeaware_clean_rerun.sh
```

当前 clean rerun 报告：

```text
results/timeaware_2080_flower_expB_strong_cls_clean_rerun_report.md
```

### 4.2 训练设置

| 项目 | 设置 |
|---|---|
| Data | `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid` |
| Server | Alibaba Cloud, `0.0.0.0:8080` |
| Clients | local PC C1/C2 only |
| Strategy | `gaps` |
| Client profile | `strong_cls` |
| Rounds | 25 |
| Min clients | 2 |
| Local epochs | 5 |
| Batch size | 32 |
| Selective aggregation | true |
| Selective warmup | 3 |
| Domain adaptation | true |
| DA warmup | 0 |
| DA steps | 100 |
| CORAL | true |
| Class-conditional CORAL | true |
| MMD | true |
| Adversarial DA | true |
| `da_lambda_coral` | 0.5 |
| `da_lambda_global_mmd` | 0.5 |
| `da_lambda_class_mmd` | 0.5 |
| `da_lambda_proto_anchor` | 0.3 |
| `da_lambda_adv` | 0.5 |
| `da_lambda_target_ce` | 0.0 |
| `da_lambda_proto` | 0.05 |
| `da_lambda_consistency` | 2.0 |
| `da_lambda_residual` | 0.1 |
| `da_lambda_proto_mmd` | 0.2 |
| `da_lambda_stage_mmd` | 0.2 |
| Server optimizer lr | 0.0005 |
| Use adapted as global | true |
| Strict calibration split | true |
| Legacy align-reg | false |

Target usage:

```text
C3/C4/C5 calibration -> server_calib_data for classification DA
C3/C4/C5 test        -> final classification evaluation
C3/C4/C5 train       -> not used, N=0
```

### 4.3 冻结 checkpoint 与推理口径

必须使用：

```text
Checkpoint: results/timeaware_2080_flower_expB_strong_cls_clean_rerun/server_latest_adapted.pth
Inference: logits
```

不要把 `server_latest.pth` 当作最终分类基座；不要把 `soft_agg` 当作主推理口径。当前 `soft_agg` 精度接近，但 NLL/ECE 明显变差，只能做补充分析。

### 4.4 当前分类指标

Clean rerun 健康状态：

| Item | Value |
|---|---:|
| Rounds completed | 25 |
| Fit failures | 0 |
| Evaluate failures | 0 |
| Adapted checkpoints | 25 |
| Latest DA steps | 100 |
| Semantic prototypes | 12 |
| Trainable semantic prototypes | 12 |
| Legacy align-reg latest | 0.0 |

C3/C4/C5 test，最佳模式 `server_latest_adapted.pth + logits`：

| Metric | Value |
|---|---:|
| Weighted accuracy | 0.989444 |
| Weighted NLL | 0.107643 |
| Weighted ECE | 0.009875 |

Per-client accuracy：

| Client | Accuracy |
|---|---:|
| C3 | 0.997388 |
| C4 | 0.998529 |
| C5 | 0.964706 |

Per-class accuracy：

| Client | Ethanol | CO | Ethylene | Methane |
|---|---:|---:|---:|---:|
| C3 | 1.000000 | 0.997015 | 0.992537 | 1.000000 |
| C4 | 0.997059 | 0.997059 | 1.000000 | 1.000000 |
| C5 | 0.979412 | 0.929412 | 0.967647 | 0.982353 |

结论：分类已经可以冻结。当前主要瓶颈不是分类基座，而是回归映射，尤其 C5-CO。

---

## 5. 回归冻结条件：REG-R3aK16-AutoV2-TimeAware2080

### 5.1 使用代码

回归/QC 主线必须使用：

```text
run_time_aware_raw_calibrated_qc_eval.py
run_regression_mainline_eval.py
run_regression_raw_diagnosis.py
exp_improved.py
model.py
config.py
federated_dataset.py
utils.py
```

当前回归报告：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_report.md
```

当前回归输出目录：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval
```

当前主要 scheme：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/flower_expB_adapted_logits
```

### 5.2 训练与校准设置

| 项目 | 设置 |
|---|---|
| Classifier | `CLS-FlowerExpB-TimeAware2080` |
| Classifier checkpoint | `server_latest_adapted.pth` |
| Routing inference | logits |
| Regressor | R3aK16 |
| Source training clients | C1, C2 |
| Training style | source C1/C2 FedAvg |
| Steps | 2000 |
| LR | 0.001 |
| Batch size | 64 |
| Huber delta | 0.2 |
| R4A | false |
| Weighted SmoothL1 | false |
| New model structures | false |

R3aK16 / model shape:

```text
reg_head_depth = 4
reg_response_branch = dct
reg_dct_k = 16
shared_trunk = false
ratio_branch = false
```

Target calibration 必须 per-client：

```text
C3 calibration -> C3 auto_v2 package -> C3 test
C4 calibration -> C4 auto_v2 package -> C4 test
C5 calibration -> C5 auto_v2 package -> C5 test
```

不要把 C3+C4+C5 混成一个全局回归 calibration / routing config。分类 DA 可以混合 target calibration；浓度回归 calibration 必须按设备单独做。

Auto_v2 specialist 当前设置：

| 项目 | 设置 |
|---|---|
| Routing mode | `auto_v2` |
| Specialist classes | 1, 2 |
| Specialist classes meaning | CO, Ethylene |
| Specialist steps | 80 |
| Full steps | 50 |
| LR | 0.001 |
| Split by | class_concentration |
| Gate metric | NRMSE_range |
| Refit affine full calib | true |

QC 口径：

```text
accept  -> automatic output
review  -> non-reject, needs review
reject  -> no output
```

正式报告必须同时给出：

```text
Full test
Accepted-only
Coverage+Review / nonreject
```

### 5.3 当前回归指标

C3/C4/C5 target test：

| Client | Raw RMSE | Raw MAE | Raw NRMSE | QC Accepted RMSE | QC Accepted Coverage | Coverage+Review RMSE | Coverage+Review MAE | Coverage+Review NRMSE | Coverage+Review P90AE | Coverage+Review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24.670 | 13.394 | 0.1492 | 6.668 | 0.465 | 12.487 | 8.111 | 0.0797 | 17.707 | 0.754 |
| C3 | 22.025 | 12.699 | 0.1215 | 6.641 | 0.462 | 10.968 | 8.205 | 0.0769 | 17.627 | 0.750 |
| C4 | 18.815 | 11.142 | 0.1128 | 5.720 | 0.478 | 9.276 | 6.639 | 0.0616 | 16.268 | 0.726 |
| C5 | 33.265 | 17.017 | 0.2159 | 7.576 | 0.459 | 16.943 | 9.288 | 0.0976 | 21.987 | 0.788 |

与昨日 role-aware 8:2 reference 的 Coverage+Review 对比：

| Scope | Yesterday RMSE | Current RMSE | Delta |
|---|---:|---:|---:|
| ALL | 12.986 | 12.487 | -0.499 |
| C3 | 10.830 | 10.968 | +0.138 |
| C4 | 12.120 | 9.276 | -2.844 |
| C5 | 16.800 | 16.943 | +0.143 |

结论：R3aK16 + per-client auto_v2/QC 已经成功迁移到 Flower ExpB adapted classifier 上，整体效果基本复现并略优于昨日 reference。

### 5.4 当前必须保留的限制说明

旧版导出记录中，`raw_ppm`、`auto_v2_ppm`、`final_ppm` 曾经完全一致：

```text
raw_auto_v2_match_ratio = 1.0
raw_auto_v2_mean_abs_diff = 0.0
raw_auto_v2_max_abs_diff = 0.0
```

解释：旧版 `raw_ppm` 实际复制自 deployment `pred_ppm`，而 `pred_ppm` 已经经过当前选择的 neural routing 路径；后续 scalar affine/bias 没有进一步改变 `calibrated_ppm`。因此旧版 `Raw all` 不能解读为纯 base R3aK16，旧版 `Final all` 也不能解读为“显著数值校准后”的结果。

C5-CO 是当前保留弱点：

| Metric | Value |
|---|---:|
| Coverage+Review | 0.547 |
| RMSE | 34.432 |
| MAE | 22.617 |
| NRMSE | 0.1530 |
| P90AE | 48.770 |

源域 C1-CO 也已经显示出 oracle-route 下仍然较差，说明 CO 问题不是单纯分类路由错误，而是 CO 回归映射本身较弱。后续改进应专门做 CO-specific residual / affine / monotonic calibration。

2026-06-18 已补充三层 ppm 审计：

```text
base_r3ak16_raw_ppm: pure base R3aK16 hard-route output
routed_pred_ppm: auto_v2/full/specialist routing output
final_ppm: scalar affine/bias/phase-affine final output before QC
```

新增代码：

```text
audit_timeaware_ppm_layers_and_co_bins.py
```

新增输出：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/ppm_layer_co_audit/layer_metrics.csv
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/ppm_layer_co_audit/co_bin_metrics.csv
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/ppm_layer_co_audit/layer_diff_audit.csv
```

CO test 层级结果：

| Client | Layer | CO RMSE | CO MAE | CO Bias | CO Slope |
|---|---|---:|---:|---:|---:|
| ALL | base | 71.023 | 51.339 | -35.360 | 0.5 |
| ALL | routed | 39.233 | 25.610 | -9.264 | 0.7 |
| ALL | final | 39.233 | 25.610 | -9.264 | 0.7 |
| C3 | base | 53.770 | 40.951 | -22.550 | 0.6 |
| C3 | routed | 38.299 | 23.783 | -13.836 | 0.7 |
| C4 | base | 55.532 | 40.640 | -14.221 | 0.7 |
| C4 | routed | 29.866 | 19.382 | 6.743 | 0.8 |
| C5 | base | 106.055 | 82.507 | -81.744 | 0.2 |
| C5 | routed | 48.262 | 35.438 | -16.260 | 0.7 |
| C5 | final | 48.262 | 35.438 | -16.260 | 0.7 |

层间结论：

```text
base -> routed: CO mean absolute change is large, especially C5-CO = 66.746 ppm.
routed -> final: match ratio = 1.0, mean_abs_diff = 0.0.
```

因此 CO 的当前问题不是 auto_v2 完全没生效。auto_v2/specialist 已经显著修正 base R3aK16，但 final scalar calibration 没有继续生效；C5-CO 高浓度仍存在系统性低估。

C5-CO 分桶结果：

| Bin | Layer | RMSE | MAE | Bias | Slope |
|---|---|---:|---:|---:|---:|
| low 25-100 | base | 37.210 | 27.360 | -25.451 | 0.403 |
| low 25-100 | routed | 31.591 | 22.606 | 0.858 | 0.895 |
| mid 125-175 | base | 98.828 | 87.868 | -87.868 | 0.496 |
| mid 125-175 | routed | 45.992 | 38.526 | -7.117 | 0.999 |
| high 200-250 | base | 160.870 | 150.676 | -150.676 | 0.241 |
| high 200-250 | routed | 65.713 | 49.458 | -48.226 | 0.643 |

下一步 CO correction 应优先针对 `C5-CO high 200-250` 做 monotonic / piecewise affine / high-concentration residual correction。

---

## 6. CO-specific 修正规则

CO-specific correction 只能作为 `REG-R3aK16-AutoV2-TimeAware2080` 的后处理增强层，不允许改变 `CLS-FlowerExpB-TimeAware2080` 分类基座，也不允许改变 R3aK16 全局回归 backbone。

硬约束：

```text
Do not retrain classifier.
Do not replace server_latest_adapted.pth + logits.
Do not change source/target split.
Do not switch back to mixed target calibration.
Do not treat CO correction as a new global regression model.
Fit correction on target calibration only.
Report final metrics on target test only.
```

拟合与评估口径：

| 项目 | 规则 |
|---|---|
| Fit data | C3/C4/C5 target calibration split only |
| Test data | C3/C4/C5 target test split only |
| Primary weak cell | C5-CO high concentration 200-250 ppm |
| First input feature | `routed_pred_ppm` |
| First output | corrected ppm written as an additional post-processing layer |
| Must preserve | `base_r3ak16_raw_ppm`、`routed_pred_ppm`、`final_ppm` 三层审计字段 |
| Must not degrade | C5-CO low/mid bins, C5 other gas classes, ALL Coverage+Review |

优先顺序：

1. `C5-CO high-bin piecewise affine`
2. `C5-CO monotonic / isotonic calibration`
3. `C5-CO high-concentration residual ridge`
4. 若前 3 步有效，再扩展到 `C3-CO` / `C4-CO`
5. 若前 4 步仍无法解决 CO high-bin 系统偏差，再开启 source-side CO training ablation

### 6.1 第一层：C5-CO High-Bin Piecewise Affine

先做最简单、可解释、低风险的修正。

适用范围：

```text
client == C5
true/predicted class == CO
high concentration region == 200-250 ppm
```

建议从两种门控各跑一版：

| Gate | 含义 |
|---|---|
| true-bin diagnostic gate | 仅用于离线诊断，按 `true_ppm` 判断 high bin，不能部署 |
| deployable pred-bin gate | 用 `routed_pred_ppm` 或 corrected pre-score 判断 high bin，可部署 |

拟合：

```text
corrected_ppm = a * routed_pred_ppm + b
```

目标：

```text
C5-CO high 200-250 RMSE 明显低于 65.713
C5-CO high 200-250 Bias 从 -48.226 明显回到接近 0
不明显伤害 C5-CO low/mid
不明显伤害 C5 Ethanol / Ethylene / Methane
```

### 6.2 第二层：Monotonic / Isotonic Calibration

如果 piecewise affine 有效但高浓度仍有压缩，则做 monotonic calibration。

输入输出：

```text
input: routed_pred_ppm
output: monotonic corrected ppm
constraint: corrected ppm is non-decreasing with routed_pred_ppm
scope: per-client, per-class
```

该方法适合当前 CO 问题，因为 C5-CO high bin 呈现高浓度系统性低估和预测压缩。

### 6.3 第三层：CO Residual Correction

只有在 piecewise affine / monotonic 不足时，再引入响应特征拟合 residual。

候选形式：

```text
residual = true_ppm - routed_pred_ppm
features = routed_pred_ppm + response stats + response_phase + interpolated_ratio + DCT low-frequency features
corrected_ppm = routed_pred_ppm + residual_model(features)
```

复杂 residual correction 必须保持可审计，不得覆盖三层 ppm 字段。新增字段建议命名：

```text
co_corrected_ppm
co_correction_delta
co_correction_mode
co_correction_applied
```

### 6.4 每一步必须报告的指标

每个 CO correction 实验必须报告：

```text
Full test
Accepted-only
Coverage+Review / nonreject
C5-CO low/mid/high bins
C5 Ethanol / CO / Ethylene / Methane
C3-CO / C4-CO / C5-CO
ALL overall
base -> routed -> final -> corrected layer audit
```

Guardrail：

| Scope | 要求 |
|---|---|
| C5-CO high | RMSE / Bias 必须明显改善 |
| C5-CO low/mid | 不得明显恶化 |
| C5 other classes | 不得明显恶化 |
| ALL Coverage+Review | 不得明显恶化 |
| Accepted-only | 不得用过低 coverage 制造虚高指标 |
| Deployability | 最终部署版本不能依赖 `true_ppm` gate |

当前阶段只允许在以上规则内推进 CO-specific correction，不再泛泛调整回归模型。

### 6.5 当前 CO Correction 验证状态

2026-06-18 已完成 Round 1 / Round 2 验证。

结果文件：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/co_correction_round1_report.md
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/co_correction_round2_guarded_gate_report.md
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/summary_metrics.csv
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/round2_best_candidate_metrics.csv
```

Round 1 结论：

```text
diag_true_high_piecewise_affine can strongly fix C5-CO high,
but it uses true_ppm >= 200 as the gate and is not deployable.
```

Round 2 新增 deployable guard：

```text
pred_threshold_nonreject = C5 + predicted CO + routed_pred_ppm >= threshold + QC accept/review
```

当前最安全的候选：

```text
ridge_highfit_pred_ge_175_nonreject_alpha_1p0
```

该候选不改变分类器、不重训 R3aK16，仅作为后处理修正：

| Subset | Scope | Baseline RMSE | Candidate RMSE | 结论 |
|---|---|---:|---:|---|
| Accepted | ALL | 6.668 | 6.634 | 小幅改善 |
| Accepted | C5-CO high | 26.564 | 24.515 | 小幅改善 |
| Coverage+Review | ALL | 12.487 | 12.379 | 小幅改善 |
| Coverage+Review | C5-CO | 34.432 | 33.572 | 小幅改善 |
| Coverage+Review | C5-CO high | 53.821 | 50.728 | 小幅改善 |

Guardrail：

```text
C5-Ethanol Full RMSE: 29.433 -> 29.433
C5-Ethylene Full RMSE: 28.353 -> 28.353
C5-Methane Full RMSE: 20.661 -> 20.661
```

当前判断：

```text
ridge_highfit_pred_ge_175_nonreject_alpha_1p0 is deployable-safe,
but the gain is still modest.
Do not freeze it as the final CO correction yet.
```

被拒绝候选：

```text
deploy_pred_ge_125_nonreject_piecewise_affine:
  C5-CO high Coverage+Review RMSE improves 53.821 -> 44.491
  but Accepted ALL worsens 6.668 -> 8.082
  and Coverage+Review ALL worsens 12.487 -> 13.532

c5_co_isotonic_all_pred_class_nonreject:
  Coverage+Review ALL improves 12.487 -> 12.332
  C5-CO high Coverage+Review improves 53.821 -> 48.170
  but Accepted ALL worsens 6.668 -> 7.223
```

Round 2 scatter：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c5_all_ridge175_nonreject_a1.png
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c5_co_ridge175_nonreject_a1.png
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c5_nonreject_ridge175_nonreject_a1.png
```

Client extension 验证：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/client_extension/co_correction_client_extension_report.md
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/client_extension/client_extension_grid_summary.csv
```

当前扩展结论：

```text
Apply guarded high-CO residual ridge to C3 and C5.
Do not apply it to C4.
```

最佳扩展候选：

```text
C3+C5 ridge_highfit_pred_ge_175_nonreject_alpha_1p0
```

关键指标：

| Scope | Baseline Coverage+Review RMSE | C3+C5 Candidate Coverage+Review RMSE |
|---|---:|---:|
| ALL | 12.487 | 12.192 |
| C3-CO high | 20.256 | 12.351 |
| C4-CO high | 14.256 | 14.256 |
| C5-CO high | 53.821 | 50.728 |

Accepted-only ALL 基本持平：

```text
6.668 -> 6.671
```

C4 不纳入扩展，原因：

```text
C4-only correction worsens C4-CO high Coverage+Review RMSE: 14.256 -> 15.317
```

下一步：

```text
Do not start source-side retraining yet.
Use C3+C5 guarded ridge as the current correction candidate.
Do not include C4.
Improve C5-CO high correction strength under the same nonreject guard.
Then tune CO-specific QC thresholds.
Only if target-side guarded correction remains insufficient, start source-side CO ablation.
```

QC threshold tuning 已完成：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/qc_threshold_tuning/co_specific_qc_threshold_tuning_report.md
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/qc_threshold_tuning/equal_coverage_key_table.csv
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/qc_threshold_tuning/deployment_visible_fields_audit.csv
```

当前推荐工作点：

```text
Coverage target: 0.75 per client
Gate: review_only
Correction: C3+C5 high-CO residual ridge, routed_pred_ppm>=175, alpha=1.0
```

核心效果：

| Metric | Baseline | Corrected review-only |
|---|---:|---:|
| Coverage+Review ALL RMSE | 11.740 | 11.441 |
| Coverage+Review ALL MAE | 7.922 | 7.720 |
| Coverage+Review ALL P90AE | 17.577 | 17.156 |
| C3-CO high RMSE | 20.256 | 12.807 |
| C5-CO high RMSE | 45.136 | 41.644 |
| Accepted ALL RMSE | 6.668 | 6.668 |
| Reject rate | 0.249 | 0.249 |

推荐 `review_only` 而不是 `nonreject` 的原因：

```text
review_only keeps accepted automatic outputs unchanged.
nonreject gives slightly stronger high-bin correction, but touches accepted samples.
```

部署 gate 只允许使用：

```text
client
pred_class
routed_pred_ppm
qc_risk_value / risk_score
qc_tuned_decision
```

不允许使用：

```text
true_class
true_ppm
test split label
oracle route
```

推荐工作点散点图：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c3_co_c3c5_reviewonly_cov75.png
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c5_co_c3c5_reviewonly_cov75.png
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c3_nonreject_c3c5_reviewonly_cov75.png
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/figures/scatter_c5_nonreject_c3c5_reviewonly_cov75.png
```

### 6.6 第一版可部署 CO 增强层

当前可冻结为第一版候选增强层：

```text
CO-GuardedRidge-C3C5-ReviewOnly-Cov75
```

系统名写法：

```text
GAPS-TimeAware2080-FlowerCLS-R3aK16-AutoV2-QC
+ CO-GuardedRidge-C3C5-ReviewOnly-Cov75
```

部署候选目录：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/deployment_candidate
```

关键文件：

```text
co_guarded_ridge_params.json
co_specific_correction_manifest.json
final_four_layer_test_outputs.csv
metric_snapshot_cov75.csv
CO_GuardedRidge_C3C5_ReviewOnly_Cov75_README.md
```

运行时规则：

```text
client in {C3, C5}
pred_class == CO
routed_pred_ppm >= 175
qc_tuned_decision == review
```

运行时触发审计：

```text
test rows = 5400
correction_applied = 80
applied_by_client = C3: 52, C5: 28
applied_decision = review: 80
accepted_applied = 0
non_co_applied = 0
```

输出字段规则：

```text
Do not overwrite final_ppm.
Write CO-enhanced output to co_corrected_ppm.
Keep base_r3ak16_raw_ppm, routed_pred_ppm, final_ppm, co_corrected_ppm for audit.
```

该增强层是后处理增强，不是新回归 backbone。若后续要替换它，必须重新通过：

```text
deployment-visible field audit
accepted-only invariant check
equal-coverage Coverage+Review check
non-CO no-touch check
```

### 6.7 源域训练是否处理

需要保留源域 CO 问题作为诊断线索，但不要把它作为当前第一优先级。

已知现象：

| Source client | Class | Route | RMSE | MAE | NRMSE |
|---|---|---|---:|---:|---:|
| C1 | CO | hard-route | 55.06 | 46.19 | 0.245 |
| C1 | CO | oracle-route | 55.06 | 46.19 | 0.245 |
| C2 | CO | hard-route | 24.42 | 18.66 | 0.109 |
| C2 | CO | oracle-route | 24.42 | 18.66 | 0.109 |

解释：

```text
C1-CO hard-route == oracle-route and both are poor.
Therefore C1-CO is not mainly a classification routing problem.
It is a weak source-side CO response-to-concentration mapping.
```

当前决策：

```text
Do not unfreeze source regression training yet.
Do not retrain global R3aK16 before target-side CO correction.
Use C1-CO weakness as evidence for why C5-CO may need CO-specific correction.
```

源域训练只在下面条件满足时开启：

```text
C5-CO high-bin piecewise affine is insufficient
AND monotonic/isotonic calibration is insufficient
AND high-concentration residual correction is insufficient
OR source-side CO weakness blocks all target-side correction from generalizing
```

若需要处理源域训练，必须作为受控 ablation，不得直接替换冻结主线。优先顺序：

1. `Source CO audit`：重新报告 C1/C2 per-class、per-bin、hard/oracle metrics。
2. `C2-only CO head diagnostic`：只作为诊断，看去掉 C1 后 CO head 是否改善。
3. `Source-client-weighted CO training`：降低 C1-CO 对 CO head 的负面影响，不能影响其他 gas。
4. `CO high-bin weighted source loss`：只增强 CO high concentration，不泛化调整全局 loss。
5. `Source-specific CO specialist`：保留全局 R3aK16，只给 CO head 增加 source-aware/specialist 分支。

每个 source-side ablation 必须同时报告：

```text
Source C1/C2 CO hard/oracle metrics
Target C3/C4/C5 CO low/mid/high bins
C5-CO high-bin RMSE/Bias
C5 other classes
ALL Coverage+Review
```

只有当 source-side ablation 在 target test 上显著改善 C5-CO high，同时不伤害 ALL 和其他 gas，才允许考虑形成新的 `REG-R3aK16-COEnhanced-TimeAware2080` 候选。否则当前 `REG-R3aK16-AutoV2-TimeAware2080` 继续保持冻结。

---

## 7. 当前可视化与诊断输出

散点图脚本：

```text
plot_timeaware_regression_scatter_comparison.py
```

散点图输出：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/scatter_comparison
```

关键文件：

```text
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/scatter_comparison/visual_index.md
results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/scatter_comparison/scatter_metric_summary.csv
```

旧 scatter 图像口径：

| Panel | 含义 |
|---|---|
| Raw all | 旧版 `raw_ppm`，来自 deployment `pred_ppm`，不是纯 base R3aK16 |
| Final all | 旧版 `final_ppm`，当时与旧版 `raw_ppm` 一致 |
| QC accepted | `qc_decision == accept` |
| Accepted + review | `qc_decision in {accept, review}` |

后续三层散点图应基于 `ppm_layer_co_audit/target_layer_predictions.csv` 重新生成，分别画 `base_r3ak16_raw_ppm`、`routed_pred_ppm` 和 `final_ppm`。

---

## 8. 严禁混用清单

不要在当前主线中混用：

```text
旧数据目录 dataset/client_data_c12src_c345tgt_calib20_test80
旧数据目录 dataset/client_data_federated_window_fullgrid_src12_tgt345
Exp A safe DA checkpoint
Exp C legacy align-reg checkpoint
server_latest.pth 作为最终分类基座
soft_agg 作为主推理口径
任意未通过 split audit 的数据目录
跨 target client 复用 auto_v2 routing_config
跨 target client 复用 selected_policy.json
跨 source/target 组合复用 norm_stats.npz、classifier checkpoint 或 regression package
```

以下代码/结果仅作 legacy 或对照，不是当前最优主入口：

```text
gaps_flower/regression_client.py
gaps_flower/regression_server.py
gaps_flower/evaluate_regression_pipeline.py
scripts/create_c12src_c345tgt_calib20_test80.py
run_time_aware_split_backbone_gain_isolation.py
audit_time_aware_auto_v2_noop.py
run_regression_training_strength_ablation.py
run_weighted_regression_loss_ablation.py
```

这些文件仍有诊断价值，不建议删除，但正式主实验不要从这些入口启动。

---

## 9. 推荐复现实验顺序

后续主线只按下面顺序运行：

1. 用 `preprocessor_time_aware.py` / `run_time_aware_target_split_ablation.py` 生成 time-aware 60-170 window-fullgrid 数据。
2. 运行 split audit，确认 C3/C4/C5 calibration:test = 2:8，且每类 gas/浓度覆盖完整。
3. 阿里云启动 Flower server，本地 PC 只启动 C1/C2 clients。
4. 训练 `CLS-FlowerExpB-TimeAware2080`，25 rounds，strong_cls，strong DA。
5. 用 `gaps_flower.evaluate_checkpoint` 评估 `server_latest_adapted.pth + logits`。
6. 用 `run_time_aware_raw_calibrated_qc_eval.py` 训练 R3aK16 source C1/C2 FedAvg。
7. 分别用 C3/C4/C5 calibration 生成 per-client auto_v2 package。
8. 分别选择 per-client QC policy。
9. 在 C3/C4/C5 test 上报告 Full / accepted-only / Coverage+Review。
10. 生成 scatter plot 和 per-client-per-class 表。
11. 在冻结主线上按第 6 节顺序推进 CO-specific correction。

---

## 10. 当前项目状态

已完成：

```text
[x] time-aware 数据协议确定
[x] 分类主线 CLS-FlowerExpB-TimeAware2080 冻结
[x] 回归主线 REG-R3aK16-AutoV2-TimeAware2080 冻结
[x] 三层 ppm 输出口径拆清
[x] Coverage+Review / accepted-only 口径明确
[x] C5-CO 弱点定位到 high concentration 系统性低估
[x] CO-specific Round 1 / Round 2 guarded-gate 验证完成
[x] CO-specific C3/C4/C5 client extension 验证完成，当前只建议 C3+C5
[x] CO-specific QC threshold tuning 完成，推荐 0.75 per-client coverage + review_only
[x] Round 2 最佳安全候选 scatter 出图
[x] CO-GuardedRidge-C3C5-ReviewOnly-Cov75 第一版可部署增强层候选冻结
[x] CO-specific deployment manifest / ridge params / final four-layer outputs 导出完成
```

待完成：

```text
[ ] Source-side CO training ablation, only if target-side CO correction is insufficient
[ ] PC runtime 验证
[ ] 树莓派离线推理
[ ] 树莓派 Flower client 通信/轻量训练
```

---

## Fixed-DA 修复后 C45->C123 结果

本节只适用于修复 DA 逻辑后的新源/目标组合：

```text
Source clients: C4, C5
Target clients: C1, C2, C3
Data protocol: time-aware 60-170 window-fullgrid
Target protocol: calibration:test = 2:8, target train = 0
Data root: dataset/client_data_c45src_c123tgt_2080_timeaware_60_170_window_fullgrid
```

旧 C45->C123 失败结果作废。旧结果中的 target accuracy 约 0.754851、Ethylene/class 2 崩溃，不再作为源/目标组合不可迁移的证据；该失败由 DA 实现路径退化导致。

修复点：

```text
gaps_flower/domain_adaptation.py
- source CE loader 与 source alignment loader 分离，避免 C4/C5 小源域 loader 被 CE 循环提前耗尽。
- semantic prototype key 兼容 compact / underscore / tuple-string 写法。
- consistency loss fallback 正确读取语义 prototype。

gaps_flower/strategy.py
- server DA calibration loader 显式加载 calibration_phase_labels.npy。
- target phase_labels 正确传入 GasSensorWindowDataset。
```

修复后分类结果：

| Scope | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| C1/C2/C3 target test | 0.994403 | 0.043628 | 0.005401 |
| C4/C5 source test | 1.000000 | 0.000034 | 0.000034 |

修复后回归 / QC 结果：

| Stage | N | RMSE | MAE | NRMSE | P90AE |
|---|---:|---:|---:|---:|---:|
| Raw R3aK16 hard-route | 8040 | 40.079 | 29.353 | 0.2535 | 61.594 |
| Auto_v2 routed | 8040 | 23.325 | 13.170 | 0.1521 | 30.940 |
| Final calibrated | 8040 | 22.935 | 12.851 | 0.1473 | 28.819 |
| Final calibrated route-correct | 7995 | 18.821 | 12.066 | 0.1031 | 28.173 |
| QC accepted | 4183 | 8.025 | 5.792 | 0.0543 | 12.658 |
| Coverage+Review | 6573 | 13.761 | 9.042 | 0.0579 | 20.870 |

正式部署统计以 `qc_decision` 为准，不以预测文件中的在线 `qc_status` 为准。`qc_status=accept` 有 5709 条，而正式 `qc_decision=accept` 有 4183 条。

主要剩余弱点：

| Client | Class / Bin | RMSE | MAE | Bias | Slope |
|---|---|---:|---:|---:|---:|
| C1 | CO full | 37.684 | 27.087 | -8.233 | 0.768 |
| C3 | CO full | 32.305 | 21.165 | -2.897 | 0.846 |
| C1 | CO high 200-250 | 51.971 | 36.826 | -34.707 | 0.294 |
| C3 | CO high 200-250 | 38.163 | 24.713 | -18.103 | 0.406 |

结论：修复后的 C4/C5 -> C1/C2/C3 全流程已经通过。分类 DA 恢复后，target accuracy 从 0.754851 提升到 0.994403，Ethylene/class 2 崩溃消失；R3aK16 + auto_v2 将全量 RMSE 从 raw 40.079 降到 final 22.935，QC accepted RMSE 为 8.025，Coverage+Review RMSE 为 13.761。说明之前失败不是源/目标组合天然不可迁移，而是 DA 实现路径退化导致。当前主要剩余瓶颈是 C1/C3 的高浓度 CO 斜率压缩和低估，后续应沿用 CO-specific review-only guarded correction 思路继续优化。

下一步建议：

1. 对 C45->C123 做 CO high correction。优先复用 C3/C5 的 guarded ridge 思路，但 client 改成 `C1-CO high` 和 `C3-CO high`。
2. 候选 gate：`client in {C1, C3}`、`class == CO`、`routed_pred_ppm` 或 `final_ppm >= threshold`、`qc_decision == review`。
3. 候选修正：residual ridge / monotonic correction。目标是不改变 accepted-only，降低 Coverage+Review，修复 C1/C3 CO high bias 和 slope compression。
4. 做 fixed-DA 版 C12->C345 sanity rerun。DA bug 修复可能影响原主线，应使用修复后代码短流程或完整流程确认原主线仍接近 `classification acc ~0.989`、`Coverage+Review RMSE ~12.5`。
5. 如果 sanity rerun 通过，当前主线升级为 `fixed-DA final frozen version`。

2026-06-19 已完成本地 CPU 5-round fixed-DA short sanity：

```text
results/timeaware_2080_c12src_c345tgt_fixed_da_sanity_report.md
results/timeaware_2080_c12src_c345tgt_fixed_da_sanity/expB_strong_da_fixed_da_sanity_r5
```

健康检查通过：

| Check | Result |
|---|---:|
| Fit 0-failure rounds | 5 / 5 |
| Evaluate 0-failure rounds | 5 / 5 |
| Latest DA steps | 100 |
| Semantic prototypes | 12 |
| Checkpoint changed tensors | 76 / 80 |
| Latest DA total_loss | 2718.347 |

5-round adapted target accuracy 为 `0.947407`。该结果只证明 fixed-DA 代码路径健康，不替代 25-round final sanity；完整 C12->C345 fixed-DA rerun 仍需在阿里云或更快环境上完成。

一句话判断：C45->C123 修复后已经从失败案例变成第二组成功泛化验证；下一步不是重训分类，而是对 C1/C3-CO high 做和之前 C5-CO 类似的 guarded CO-specific correction，并用修复版代码回头 sanity check 原 C12->C345 主线。

---

## 11. 当前阶段性结论

当前最佳分类和回归路线已经可以在固定 C1/C2 -> C3/C4/C5、time-aware 60-170 window-fullgrid、target calibration:test = 2:8 协议下初步冻结。

分类采用 `CLS-FlowerExpB-TimeAware2080`，即 Flower ExpB strong_cls + strong DA + adapted logits，在 C3/C4/C5 test 上达到 0.989444 weighted accuracy，NLL 0.107643，ECE 0.009875。

回归采用 `REG-R3aK16-AutoV2-TimeAware2080`，即 R3aK16 source C1/C2 FedAvg + per-client auto_v2 specialist + Coverage/Review QC，在 C3/C4/C5 test 上达到 Coverage+Review ALL RMSE 12.487、MAE 8.111、NRMSE 0.0797，基本复现并略优于昨日 role-aware 8:2 reference。

下一步不要再重训或替换分类基座。应在当前冻结主线上，专门推进 CO-specific residual / affine / monotonic calibration，优先处理 C5-CO，同时保留 accepted-only 与 Coverage+Review 两套部署口径。
---

## Fixed-DA C12->C345 25-round sanity rerun, 2026-06-21

Artifacts:

```text
results/timeaware_2080_c12src_c345tgt_fixed_da_r25/expB_strong_da_fixed_da_r25
results/timeaware_2080_c12src_c345tgt_fixed_da_r25_analysis.md
results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval
```

Classification with `server_latest_adapted.pth + logits`:

| Scope | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| C3+C4+C5 weighted | 0.982593 | 0.153917 | 0.014551 |
| C3 | 0.992164 | 0.043242 | 0.005599 |
| C4 | 0.983088 | 0.183244 | 0.014588 |
| C5 | 0.963235 | 0.342685 | 0.032156 |
| C1+C2 source weighted | 1.000000 | 0.000062 | 0.000062 |

Fixed-DA health checks passed: 25/25 rounds completed, fit/eval failures 0, DA json files 25, adapted checkpoints 25, latest DA steps 100, latest DA total_loss 2201.904, semantic prototypes 12, checkpoint changed tensors 76/80.

Acceptance decision: this run is acceptable but not a direct `fixed-DA final frozen` classifier upgrade. ALL accuracy `0.982593` is below the `0.985` direct-freeze threshold, while C5 accuracy `0.963235`, weighted ECE `0.014551`, and non-zero DA loss satisfy sanity requirements. Do not tune DA parameters further in this round; use the result for regression/QC validation.

R3aK16 + auto_v2 regression/QC validation from the fixed-DA backbone:

| Stage | Scope | RMSE | MAE | NRMSE | Coverage |
|---|---|---:|---:|---:|---:|
| Raw R3aK16 hard-route | ALL | 27.336 | 12.999 | 0.1578 | - |
| Auto_v2 calibrated route-correct | ALL | 19.895 | 11.230 | 0.1078 | - |
| QC accepted | ALL | 5.723 | 3.989 | 0.0372 | 42.33% |
| Coverage+Review | ALL | 11.828 | 7.456 | 0.0716 | 76.44% |

Conclusion: fixed-DA C12->C345 full sanity is valid and the combined regression/QC pipeline is strong. It should be recorded as an acceptable fixed-DA sanity rerun with better-than-expected Coverage+Review RMSE, but not renamed to final frozen solely on classification because ALL accuracy remains below 0.985.

---

## Meeting report visual entry, 2026-06-23

Do not keep adding one-off plotting scripts for meeting figures. Use one unified entry:

```bash
python scripts/build_meeting_report_visuals.py
```

Fast table-only check:

```bash
python scripts/build_meeting_report_visuals.py --tables-only
```

Reference plotting convention:

```text
plot_timeaware_regression_scatter_comparison.py
```

The unified entry follows the same regression scatter structure: Raw, Auto_v2, QC accepted, and accepted+review panels. The metric box reports RMSE, MAE, NRMSE, P90AE, Bias, and Coverage.

Required inputs per direction:

```text
classification_json:
  evaluate_checkpoint output with weighted_accuracy / weighted_nll / weighted_ece

qc_test_records.csv:
  true_ppm, true_class, client or client_id, raw_ppm, final_ppm, qc_decision

optional CO correction CSV:
  true_ppm, true_class, client, final_ppm, co_corrected_ppm,
  qc_decision or qc_tuned_decision
```

Current configured report inputs:

```text
C12->C345:
  results/timeaware_2080_c12src_c345tgt_fixed_da_r25/expB_strong_da_fixed_da_r25/eval_c345_test_logits.json
  results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/fixed_da_r25/qc_test_records.csv

C45->C123:
  results/timeaware_2080_c45src_c123tgt_flower_r3ak16_auto_v2_eval_after_fix/classifier_eval_c123_test_logits.json
  results/timeaware_2080_c45src_c123tgt_flower_r3ak16_auto_v2_eval_after_fix/pipeline_eval/flower_expB_timeaware_c45src_c123tgt_after_fix/qc_test_records.csv

C12 CO correction candidate:
  results/timeaware_2080_flower_expB_r3ak16_auto_v2_eval/co_specific_correction/deployment_candidate/final_four_layer_test_outputs.csv
```

Generated report directory:

```text
results/meeting_report_visuals_20260623
```

Main report files:

```text
tables/main_overview.csv
tables/rejected_risk_summary.csv
tables/co_high_bias_summary.csv
required_files_manifest.md
visual_index.md
```

Figure groups:

```text
src12_tgt345/scatter_src12_tgt345_all_targets_stages.png
src12_tgt345/scatter_src12_tgt345_c3_stages.png
src12_tgt345/scatter_src12_tgt345_c4_stages.png
src12_tgt345/scatter_src12_tgt345_c5_stages.png
src12_tgt345/scatter_src12_tgt345_qc_decisions.png
src12_tgt345/scatter_src12_tgt345_co_high_correction.png

src45_tgt123/scatter_src45_tgt123_all_targets_stages.png
src45_tgt123/scatter_src45_tgt123_c1_stages.png
src45_tgt123/scatter_src45_tgt123_c2_stages.png
src45_tgt123/scatter_src45_tgt123_c3_stages.png
src45_tgt123/scatter_src45_tgt123_qc_decisions.png

stage_rmse_overview.png
```

Reporting convention:

```text
Primary deployment metric: accepted+review.
QC accepted: automatic trusted output only.
Rejected set: report reject_RMSE, reject_P90AE, and reject_high_error_rate to show rejected samples are high risk rather than random.
CO high: report Bias because the main physical failure mode is high-concentration underestimation.
C45->C123 CO high is diagnostic only for now; do not claim correction until a guarded C1/C3 CO high method is validated.
C12 CO correction row is an existing guarded candidate and should be presented as an auxiliary comparison, not as a newly retrained fixed-DA r25 correction.

---

## QC v2 deployment replay decision, 2026-06-23

Scope:

```text
C12 -> C345 fixed-DA r25
R3aK16 + per-client auto_v2 packages
PC batch replay
```

Archive paths:

```text
results/qc_deploy_v2_c12_c345_fixed_da_r25/auto_v2_packages
results/qc_deploy_v2_c12_c345_fixed_da_r25/replay_test
results/qc_deploy_v2_c12_c345_fixed_da_r25/qc_policy_comparison_summary.md
results/qc_deploy_v2_c12_c345_fixed_da_r25/qc_policy_comparison_summary.csv
results/flower_training_regression_qc_workflow_20260623.md
```

Main conclusion:

```text
QC v2 completed deployment validation for the 40-D response descriptor,
class-wise response ranking, and margin risk. The enhanced risk signals can be
generated, serialized, loaded, and replayed on PC.

However, under the current single composite/route threshold policies, QC v2
does not yet satisfy the deployment guardrails and cannot replace the old v1 QC.
The old v1 QC remains the main deployment QC baseline for C12->C345 fixed-DA r25.
QC v2 is retained only as a candidate risk signal and diagnostic module.
```

Key numbers:

```text
old_qc_v1_8d_policy:
  accepted coverage = 0.7257
  accepted RMSE = 9.924
  accepted+review coverage = 0.7994
  accepted+review RMSE = 11.562
  reject RMSE = 56.508

new_qc_v2_40d_old_policy:
  accepted+review coverage = 0.5615
  accepted+review RMSE = 13.119

new_qc_v2_40d_composite_calib20:
  accepted+review coverage = 0.8267
  accepted+review RMSE = 20.309

new_qc_v2_40d_route_calib20:
  accepted+review coverage = 0.8239
  accepted+review RMSE = 19.911
```

Interpretation:

```text
QC v2 is not a code failure. It is a policy-selection failure.

The old v1 threshold is too strict for the new 40-D risk scale.
Calibration-fitted single composite/route thresholds recover coverage but do
not separate high-error windows well enough.
```

Current mainline decision:

```text
Use old_qc_v1_8d_policy for the C12->C345 fixed-DA r25 report and deployment
baseline.

Do not present QC v2 as better than v1.

Next step should be guardrail-constrained policy selection that adds v2 risk
terms to the stable v1 baseline, rather than replacing v1 with one 40-D
composite threshold.
```
```
