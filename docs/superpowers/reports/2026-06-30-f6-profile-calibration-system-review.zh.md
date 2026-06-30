# F6 Strong-DA Backbone 与 Profile Calibration 阶段性系统梳理

日期：2026-06-30  
当前工作分支：`codex/regression-aware-fusion`  
数据协议：`C12 -> C345`, time-aware 60-170s window-fullgrid, target calibration:test = 2:8

这份文档接在 `CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md` 后面，用来冻结最近一轮 Stage-A Flower 分类矩阵、F6 strong-DA 分类基座、R3aK16 回归基座、第三层 target profile calibration、QC coverage sweep 和 oracle-route 对照的阶段性状态。

## 0. GitHub 同步状态

当前本地检查结果：

- 当前分支：`codex/regression-aware-fusion`
- 当前分支没有 upstream：`NO_UPSTREAM`
- `origin`：`https://github.com/Hjh-fy/GAPS.git`
- 本地存在大量 `??` 未跟踪文件，包括新的报告、诊断脚本、QC/回归工具和部分结果索引。
- 因此，不能认为当前使用的全部代码和文档已经上传到 GitHub。

建议先不要直接 `git add .`，因为 `results/` 下有大量 CSV/模型/中间文件，可能很大。更稳妥的做法是先提交代码、配置和轻量报告，再按需提交结果摘要。

检查命令：

```powershell
git status -sb
git branch -vv
git log --oneline --decorate -n 12
git rev-parse --abbrev-ref --symbolic-full-name '@{u}'
git ls-remote --heads origin codex/regression-aware-fusion
```

最小上传当前已提交分支：

```powershell
git push -u origin codex/regression-aware-fusion
```

如果要把本轮阶段性文档与关键配置也提交：

```powershell
git add `
  CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md `
  docs/superpowers/reports/2026-06-30-f6-profile-calibration-system-review.zh.md `
  docs/superpowers/reports/2026-06-30-regression-aware-fusion-p1-findings.zh.md `
  docs/superpowers/reports/2026-06-30-regression-aware-fusion-p2-balanced.zh.md `
  docs/superpowers/reports/2026-06-30-profile-qc-coverage-client-audit.zh.md `
  docs/superpowers/reports/2026-06-30-classification-correct-regression-mainline.zh.md `
  docs/superpowers/reports/2026-06-30-real-oracle-gap-and-low-cal-stress.zh.md `
  docs/superpowers/reports/2026-06-30-h8-validation-profile-choice-stress.zh.md `
  configs/oracle_route_profile_qc_profiles_20260630.json

git commit -m "Document F6 profile calibration stage review"
git push -u origin codex/regression-aware-fusion
```

如果要把关键代码脚本也纳入 Git，需要先确认每个脚本是否属于当前主线。建议用下面命令逐个加入，而不是全量加入：

```powershell
git add `
  generate_flower_matrix_commands.py `
  validate_matrix_command_config.py `
  gaps_flower/server_app.py `
  docs/flower_matrix_stage_a_runbook_20260627.md `
  run_time_aware_raw_calibrated_qc_eval.py `
  audit_timeaware_ppm_layers_and_co_bins.py `
  run_formal_target_ridge_auto_v2_eval.py `
  run_formal_target_mlp_auto_v2_eval.py `
  run_source_augmented_target_ridge_eval.py `
  run_co_only_source_aug_hybrid_eval.py `
  run_formal_c4_route_rescue_selector.py `
  run_l3_lightweight_hybrid_matrix.py `
  run_profile_qc_coverage_audit.py

git commit -m "Add F6 profile calibration replay tooling"
git push
```

如果某个脚本不存在或不是本轮主线，用 `git add` 时删掉对应行即可。

## 1. 当前冻结名称

| 层级 | 当前名称 | 说明 |
|---|---|---|
| 分类基座 | `F6_C12_to_C345_fixed_da_strong_r25` | C1/C2 source -> C3/C4/C5 target, Flower Stage-A, strong fixed-DA |
| 回归基座 | `F6 strongDA R3aK16 auto_v2` | 使用 F6 adapted classifier 输出，训练 R3aK16 回归与 per-client auto_v2/QC |
| 第三层 balanced profile | `H2.3 target direct-head` | C3=target MLP, C4=target Ridge, C5=C5-grid target MLP |
| 第三层 CO-priority profile | `H8 + formal C4 route rescue` | H2.3 fallback + source-augmented CO specialist + calibration-selected C4 rescue |
| 部署 QC | `qc_test_records.qc_decision` | 使用正式 `qc_decision`，post-profile 只替换 ppm 后重算误差 |

主数据目录：

```text
dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid
```

最新本地 replay 根目录：

```text
results/f6_fixed_da_strong_r25_profile_replay_20260630
results/f6_fixed_da_strong_r25_profile_oracle_route_20260630
```

远端阿里云同步目录：

```text
/root/GAPS/results/f6_fixed_da_strong_r25_profile_replay_20260630
/root/GAPS/results/f6_fixed_da_strong_r25_profile_oracle_route_20260630
```

## 2. Stage-A Flower 分类矩阵

### 2.1 防错修复

本轮已经修复 strong DA 配置防错问题：

- `gaps_flower/server_app.py`
  - server 启动后保存 `run_config.json`
  - 记录完整 argparse 参数
  - 支持 `--da-preset none/default/fixed_da_strong`
- `generate_flower_matrix_commands.py`
  - command manifest 中写入 `expected_da_config`
  - F6 strong run command 包含 `--da-preset fixed_da_strong`
- `validate_matrix_command_config.py`
  - 校验 F6 strong run 是否包含关键 strong-DA 参数
- `docs/flower_matrix_stage_a_runbook_20260627.md`
  - 记录 Stage-A matrix runbook

关键验证命令：

```powershell
python -m py_compile gaps_flower/server_app.py generate_flower_matrix_commands.py validate_matrix_command_config.py
python validate_matrix_command_config.py results/source_target_classification_matrix_20260627_commands/F6_C12_to_C345/command_manifest.json
git diff --check
pytest tests/test_flower_classification_contract.py
```

### 2.2 网络/终端拓扑

本地到阿里云 server：

```powershell
ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 root@121.40.139.213
```

树莓派反向隧道：

```bash
ssh -N -R 127.0.0.1:18080:127.0.0.1:18080 gaps@172.31.139.224
```

生成 F6 strong command 的模板：

```powershell
python generate_flower_matrix_commands.py `
  --runs F6_C12_to_C345 `
  --remote-project-dir /root/GAPS `
  --remote-data-root /root/GAPS/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --remote-output-root /root/GAPS/results/source_target_classification_matrix_20260627 `
  --server-public-address 127.0.0.1:18080 `
  --local-data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --da-preset fixed_da_strong `
  --run-suffix fixed_da_strong_r25
```

### 2.3 F6 分类结果

正式 run：

```text
F6_C12_to_C345_fixed_da_strong_r25
```

核心结果：

| 指标 | 数值 |
|---|---:|
| final adapted target accuracy | 0.9817 |
| final macro-F1 | 0.9817 |
| final ECE | 0.0143 |
| final NLL | 0.1051 |
| best adapted round | 19 |
| best adapted accuracy | 0.9869 |

结果路径：

```text
/root/GAPS/results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/target_summary/
results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/target_summary/
```

关键图：

```text
target_client_accuracy_curves.png
```

## 3. F6 回归基座：R3aK16 auto_v2

主脚本：

```text
run_time_aware_raw_calibrated_qc_eval.py
```

辅助审计：

```text
audit_timeaware_ppm_layers_and_co_bins.py
```

输入：

```text
F6 adapted classifier checkpoint
dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid
source clients C1,C2
target clients C3,C4,C5
```

输出根目录：

```text
results/regression_f6_fixed_da_strong_r25_20260630
/root/GAPS/results/regression_f6_fixed_da_strong_r25_20260630
```

主结果：

| Scope | Full RMSE / NRMSE | Accepted Coverage | Accepted RMSE / NRMSE | Coverage+Review | Coverage+Review RMSE / NRMSE |
|---|---:|---:|---:|---:|---:|
| ALL | 28.20 / 0.1901 | 46.76% | 7.46 / 0.0511 | 75.13% | 12.30 / 0.0763 |
| C3 | 19.55 / 0.1131 | 51.79% | 8.48 / 0.0600 | 75.19% | 12.26 / 0.0792 |
| C4 | 27.93 / 0.1447 | 43.53% | 4.74 / 0.0324 | 73.90% | 9.94 / 0.0635 |
| C5 | 40.31 / 0.3120 | 40.07% | 7.06 / 0.0424 | 76.25% | 14.28 / 0.0817 |

关键 CSV：

```text
results/regression_f6_fixed_da_strong_r25_20260630/analysis/mainline_comparison_corrected.csv
results/regression_f6_fixed_da_strong_r25_20260630/analysis/nonreject_comparison_corrected.csv
results/regression_f6_fixed_da_strong_r25_20260630/pipeline_eval/ppm_layer_co_audit/target_layer_predictions.csv
```

## 4. 第三层 Profile Calibration Replay

### 4.1 H2.3 target direct-head

H2.3 不是重新训练分类器，而是在 target calibration split 上训练小型 target direct head，然后对 test split 输出新的 ppm。

组合逻辑：

| Client | Head |
|---|---|
| C3 | target MLP + C4 rescue |
| C4 | target Ridge + C4 rescue |
| C5 | C5-grid target MLP + C4 rescue |

相关脚本：

```text
run_formal_target_ridge_auto_v2_eval.py
run_formal_target_mlp_auto_v2_eval.py
run_regression_head_ablation.py
```

replay 命令：

```powershell
python run_formal_target_ridge_auto_v2_eval.py `
  --target-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/target_layer_predictions.csv `
  --output-dir results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_target_ridge

python run_formal_target_mlp_auto_v2_eval.py `
  --target-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/target_layer_predictions.csv `
  --output-dir results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_target_mlp `
  --hidden-grid 32 `
  --alphas 0.001,0.01

python run_formal_target_mlp_auto_v2_eval.py `
  --target-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/target_layer_predictions.csv `
  --target-clients 5 `
  --output-dir results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_target_mlp_c5_grid `
  --hidden-grid "16;32;64;32,16" `
  --alphas 0.001,0.01,0.1,1
```

H2.3 合成输出：

```text
results/f6_fixed_da_strong_r25_profile_replay_20260630/h2_3_profile_replay/h2_3_profile_predictions.csv
```

### 4.2 H8 + formal C4 route rescue

H8 逻辑：

```text
if predicted route is CO:
    use source-augmented target Ridge CO specialist
else:
    use H2.3 ppm
then apply formal C4 route rescue selected on calibration
```

相关脚本：

```text
run_source_augmented_target_ridge_eval.py
run_co_only_source_aug_hybrid_eval.py
run_formal_c4_route_rescue_selector.py
```

replay 命令：

```powershell
python run_source_augmented_target_ridge_eval.py `
  --target-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/target_layer_predictions.csv `
  --output-dir results/f6_fixed_da_strong_r25_profile_replay_20260630/source_augmented_target_ridge

python run_co_only_source_aug_hybrid_eval.py `
  --h23-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/h2_3_profile_replay/h2_3_profile_predictions.csv `
  --h23-key client_hybrid_mlp_c3_ridge_c4_c5grid_ppm `
  --source-aug-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/source_augmented_target_ridge/target_predictions_plus_source_preds_plus_c4_rescue.csv `
  --output-dir results/f6_fixed_da_strong_r25_profile_replay_20260630/co_only_h8_replay

python run_formal_c4_route_rescue_selector.py `
  --target-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/target_layer_predictions.csv `
  --h8-test-predictions results/f6_fixed_da_strong_r25_profile_replay_20260630/co_only_h8_replay/co_only_source_aug_hybrid_predictions.csv `
  --output-dir results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector
```

selected C4 gate：

```json
{
  "class_label": "ethanol",
  "pred_classes": "0",
  "phase": "any",
  "max_final": 20.0,
  "min_risk": 6.0,
  "max_conf_margin": 1.0,
  "rescue_ppm": 250.0,
  "hit_N": 2,
  "true_c4_high_hits": 2,
  "false_hits": 0,
  "calib_c4_high_N": 24,
  "calib_c4_high_recall": 0.0833
}
```

输出：

```text
results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_predictions.csv
```

## 5. Post-profile QC 结果

QC 口径：

- 使用正式 deployment QC 文件：`qc_test_records.csv`
- 使用字段：`qc_decision`
- H2.3/H8+C4 只替换 ppm 输出，再重新计算误差
- Accepted：`qc_decision=accept`
- Accepted+Review：`qc_decision in {accept, review}`

结果：

| Profile | Scope | Full RMSE / NRMSE | Accepted Coverage | Accepted RMSE / NRMSE | Coverage+Review | Coverage+Review RMSE / NRMSE |
|---|---|---:|---:|---:|---:|---:|
| H2.3 | ALL | 22.94 / 0.1789 | 46.76% | 6.20 / 0.0341 | 75.13% | 7.98 / 0.0425 |
| H8+C4 | ALL | 22.44 / 0.1786 | 46.76% | 5.21 / 0.0307 | 75.13% | 6.48 / 0.0371 |
| H2.3 | C3 | 13.31 / 0.0978 | 51.79% | 5.84 / 0.0341 | 75.19% | 6.89 / 0.0390 |
| H8+C4 | C3 | 12.29 / 0.0897 | 51.79% | 5.03 / 0.0314 | 75.19% | 5.76 / 0.0352 |
| H2.3 | C4 | 13.50 / 0.0734 | 43.53% | 5.95 / 0.0326 | 73.90% | 7.09 / 0.0379 |
| H8+C4 | C4 | 13.05 / 0.0718 | 43.53% | 5.40 / 0.0306 | 73.90% | 6.54 / 0.0359 |
| H2.3 | C5 | 39.48 / 0.3207 | 40.07% | 7.25 / 0.0358 | 76.25% | 10.41 / 0.0521 |
| H8+C4 | C5 | 39.14 / 0.3250 | 40.07% | 5.44 / 0.0288 | 76.25% | 7.63 / 0.0415 |

文件：

```text
results/f6_fixed_da_strong_r25_profile_replay_20260630/post_profile_qc/profile_post_qc_metrics.csv
results/f6_fixed_da_strong_r25_profile_replay_20260630/post_profile_qc/profile_post_qc_report.md
```

## 6. QC 风险分数与 Coverage sweep

正式 QC 中 C3/C4/C5 使用的 `qc_score` 均为：

```text
composite_response_risk
```

因此 sweep 使用的 `qc_risk_value` 实际是每个 window 的 `composite_response_risk`。

风险分数主要依赖：

| 风险项 | 依赖 |
|---|---|
| `classifier_uncertainty` | 分类器 top1 softmax 置信度 |
| `margin_risk` | top1/top2 置信度差距 |
| `response_signature_norm` | 当前窗口响应形状与 calibration 响应原型的距离 |
| `response_conc_gap_norm` | 预测 ppm 与最近 calibration 浓度的差距 |
| `class_response_rank_risk` | 响应形状排序是否支持预测类别 |
| `class_response_margin_risk` | 预测类别响应匹配度与最佳类别匹配度差距 |
| `route_response_risk` | route 与响应一致性的综合风险 |
| `composite_response_risk` | 多个风险项的综合/max |

相关代码：

```text
gaps_deploy/qc_policy.py
run_time_aware_raw_calibrated_qc_eval.py
scripts/evaluate_coverage_review_qc.py
```

H8+C4 的 threshold multiplier sweep：

| Setting | Coverage+Review | RMSE / NRMSE |
|---:|---:|---:|
| 1.0x | 75.13% | 6.476 / 0.0371 |
| 1.1x | 78.59% | 6.575 / 0.0377 |
| 1.2x | 81.17% | 6.714 / 0.0386 |
| 1.35x | 83.96% | 6.902 / 0.0396 |
| 1.5x | 86.87% | 7.422 / 0.0442 |
| 2.0x | 92.11% | 9.096 / 0.0501 |

建议部署候选：

- 稳健版：`1.2x`
- 高覆盖版：`1.35x`
- `1.5x` 后误差开始明显上升

文件：

```text
results/f6_fixed_da_strong_r25_profile_replay_20260630/qc_threshold_sweep/profile_qc_threshold_sweep.csv
results/f6_fixed_da_strong_r25_profile_replay_20260630/qc_threshold_sweep/profile_qc_threshold_sweep_by_client.csv
results/f6_fixed_da_strong_r25_profile_replay_20260630/qc_threshold_sweep/coverage_review_rmse_curve.png
```

## 7. Oracle-route 对照

目的：回答“如果分类/route 全部正确，回归 head 本身能达到什么水平？”

构造：

```text
pred_class = true_class
route_class = true_class
pred_gas = true_gas
```

注意：本实验没有重新生成完美分类器的 logits/confidence/risk score，因此 Accepted/Coverage+Review coverage 沿用正式 QC，主要观察回归误差变化。

结果：

| Profile | Scope | Full RMSE / NRMSE | Accepted Coverage | Accepted RMSE / NRMSE | Coverage+Review | Coverage+Review RMSE / NRMSE |
|---|---|---:|---:|---:|---:|---:|
| H2.3 oracle-route | ALL | 10.57 / 0.0564 | 46.76% | 6.20 / 0.0341 | 75.13% | 7.98 / 0.0425 |
| H8+C4 oracle-route | ALL | 9.10 / 0.0511 | 46.76% | 5.21 / 0.0307 | 75.13% | 6.48 / 0.0371 |
| H2.3 oracle-route | C3 | 9.63 / 0.0540 | 51.79% | 5.84 / 0.0341 | 75.19% | 6.89 / 0.0390 |
| H8+C4 oracle-route | C3 | 9.13 / 0.0522 | 51.79% | 5.03 / 0.0314 | 75.19% | 5.76 / 0.0352 |
| H2.3 oracle-route | C4 | 9.28 / 0.0527 | 43.53% | 5.95 / 0.0326 | 73.90% | 7.09 / 0.0379 |
| H8+C4 oracle-route | C4 | 8.55 / 0.0502 | 43.53% | 5.40 / 0.0306 | 73.90% | 6.54 / 0.0359 |
| H2.3 oracle-route | C5 | 13.23 / 0.0643 | 40.07% | 7.25 / 0.0358 | 76.25% | 10.41 / 0.0521 |
| H8+C4 oracle-route | C5 | 9.57 / 0.0499 | 40.07% | 5.44 / 0.0288 | 76.25% | 7.63 / 0.0415 |

解释：

- Full-set 的主要污染来自 route/classification tail error。
- 当前 QC 已经将很多高风险 route error 放到 reject 区域。
- Accepted+Review 子集里，profile/head 选择成为主要差异来源。

文件：

```text
results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/post_profile_qc/oracle_route_post_qc_metrics.csv
results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/post_profile_qc/oracle_route_post_qc_report.md
```

## 8. 当前判断

1. 分类基座 F6 strong-DA 已经恢复到高水平，可作为当前阶段分类 backbone。
2. 单纯 F6 R3aK16 auto_v2 回归 full-set 不够好，主要被 route/classification tail error 和 C5 尾部问题污染。
3. 第三层 profile calibration 明显改善 post-QC 读数。
4. H2.3 更像 balanced/base profile；H8+C4 在当前 replay 下对 Accepted+Review 更强。
5. 如果加入 H2.3+ / client selector，已有报告显示 C3/C4 更适合 H2.3+，C5 更适合 H8+C4；这应作为下一步正式 selector artifact 的方向。
6. 论文/组会中应拆开报告：
   - real-route full：真实部署端到端能力
   - post-profile QC：部署可用读数
   - oracle-route full：分类正确时回归 head 的理论能力
   - coverage sweep：风险阈值与可用覆盖率的 tradeoff

## 9. 复盘索引

核心旧手册：

```text
CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md
```

本轮新增/相关报告：

```text
docs/superpowers/reports/2026-06-30-regression-aware-fusion-p1-findings.zh.md
docs/superpowers/reports/2026-06-30-regression-aware-fusion-p2-balanced.zh.md
docs/superpowers/reports/2026-06-30-profile-qc-coverage-client-audit.zh.md
docs/superpowers/reports/2026-06-30-classification-correct-regression-mainline.zh.md
docs/superpowers/reports/2026-06-30-real-oracle-gap-and-low-cal-stress.zh.md
docs/superpowers/reports/2026-06-30-h8-validation-profile-choice-stress.zh.md
docs/superpowers/reports/2026-06-30-f6-profile-calibration-system-review.zh.md
```

核心结果目录：

```text
results/regression_f6_fixed_da_strong_r25_20260630
results/f6_fixed_da_strong_r25_profile_replay_20260630
results/f6_fixed_da_strong_r25_profile_oracle_route_20260630
results/h2_3_plus_fusion_profile_20260630
```

后续建议：

1. 把 `run_profile_qc_coverage_audit.py`、H2.3+ replay、client selector 固化成正式入口脚本。
2. 将大结果文件留在 `results/`，GitHub 只提交轻量 summary/report/manifest。
3. 对低 calibration 量做 stress test，确认 `C3/C4 -> H2.3+`, `C5 -> H8+C4` 是否稳定。
4. 在论文主表中明确区分 real-route、oracle-route、Accepted-only、Coverage+Review，避免单一 full RMSE 混淆分类错误和回归 head 能力。
