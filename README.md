# GAPS 项目代码导览

> 2026-07-15 状态提示：当前权威主线已经更新为真实云边 Flower 分类、C5 个性化 H8 回归和 deployment-visible QC。下面较早的 R3aK16/AutoV2 章节保留作历史代码导航，不能替代最新实验结论。
>
> 新对话或从 GitHub 接手时，请先阅读 `docs/experiments/iotj_latest_handoff_20260715.zh.md`，再阅读 `代码文件介绍.md` 和实验经验笔记本。当前证据位于分支 `codex/system-safety-hardening`。

本文档用于说明主要代码文件之间的关系。当前正式系统名和论文主线以最新交接文档及 `代码文件介绍.md` 为准。

当前主线可以概括为：

```text
time-aware 60-170 s C1/C2/C5 window 数据
-> 真实 ECS + Pi + PC Flower 联邦分类
-> C5 calibration-assisted server adaptation
-> C5 target-personalized Ridge/MLP/H8 回归
-> deployment-visible QC: accept / review / reject
-> runtime parity、系统开销和论文证据
```

正式命名：

```text
分类主线: IoTJ v2r1 + corrected v3 B1-B5
回归主线: C5 R0-R7，固定 H8 为当前最佳可部署点估计
系统主线: GAPS calibration-assisted cloud-edge sensing + reliable output
```

## 目录关系

当前核心代码不只在 `gaps_flower/` 和 `gaps_deploy/` 下，但这两个目录确实是训练与部署逻辑的主体。

```text
.
├── gaps_flower/      # Flower 分类、server-side DA、回归任务构建、auto_v2 拟合
├── gaps_deploy/      # 部署 package、推理、QC policy、risk score、运行时输出
├── scripts/          # 固定实验入口、报告图表生成、部署包校验
├── diagnostics/      # 诊断和 ablation 脚本，通常不作为正式主入口
├── dataset/          # 本地数据目录，不建议提交 Git
├── results/          # 本地结果、报告、图表，不建议提交 Git
├── model.py          # 分类/回归共享模型定义
├── config.py         # 全局配置和实验参数
├── federated_dataset.py
├── utils.py
└── run_*.py          # 顶层实验、评估、诊断入口
```

注意：目录名是 `gaps_flower`，不是 `gaps_flwoer`。

## 主线一：数据生成与 split

主要文件：

```text
preprocessor_time_aware.py
run_time_aware_target_split_ablation.py
scripts/audit_target_split.py
```

功能关系：

```text
原始 txt 文件
-> preprocessor_time_aware.py
-> 10 Hz 重采样、60-170 s 裁剪、100 x 8 window、窗口元数据
-> run_time_aware_target_split_ablation.py
-> source/target role-aware split
-> split manifest 和 split audit
```

当前正式数据协议是 time-aware 60-170 window-fullgrid。旧入口如 `preprocessor.py`、`split_dataset.py`、`scripts/create_c12src_c345tgt_calib20_test80.py` 主要作为 legacy 或对照，不建议作为正式主线继续推进。

## 主线二：Flower 分类与 fixed-DA

主要文件：

```text
gaps_flower/client_app.py
gaps_flower/server_app.py
gaps_flower/strategy.py
gaps_flower/domain_adaptation.py
gaps_flower/task.py
gaps_flower/evaluate_checkpoint.py
scripts/run_fixed_da_source_target_experiment.py
```

功能关系：

```text
scripts/run_fixed_da_source_target_experiment.py
-> 启动 fixed source-target 实验

gaps_flower/server_app.py
-> Flower server 入口
-> 使用 GapsStrategy 聚合客户端参数

gaps_flower/client_app.py
-> Flower client 入口
-> 只启动 source clients，例如 C12->C345 时只启动 C1/C2

gaps_flower/strategy.py
-> FedAvg / selective aggregation
-> 收集 prototypes、semantic prototypes、device residuals
-> 调用 server-side DA
-> 保存 server_round_xxx.pth 和 server_round_xxx_adapted.pth

gaps_flower/domain_adaptation.py
-> server-side DA
-> source CE、CORAL、MMD、adversarial DA、prototype consistency 等
-> fixed-DA 版本中 source CE loader 和 source alignment loader 已分离

gaps_flower/evaluate_checkpoint.py
-> 评估 server_latest_adapted.pth + logits
-> 输出 target accuracy / NLL / ECE
```

分类训练和回归训练是解耦的。分类主线中 `USE_REG_LOSS = False`，分类器主要提供 gas class route 和 logits。

当前分类推理口径应优先使用：

```text
server_latest_adapted.pth + logits
```

不要把 `server_latest.pth` 当作最终分类基座，也不要把 `soft_agg` 当作正式主推理口径。

## 主线三：R3aK16 回归、auto_v2 与 QC

主要文件：

```text
run_time_aware_raw_calibrated_qc_eval.py
gaps_flower/regression_task.py
gaps_flower/specialist_calibration_fit.py
gaps_deploy/build_per_client_packages.py
gaps_deploy/calibration.py
gaps_deploy/inference.py
gaps_deploy/qc_policy.py
gaps_deploy/build_qc_policy_from_predictions.py
```

功能关系：

```text
run_time_aware_raw_calibrated_qc_eval.py
-> 当前回归/QC 总入口之一
-> 训练 source-domain R3aK16 回归
-> 构建 per-client auto_v2 package
-> 运行部署推理
-> 拟合或应用 QC policy
-> 导出 full / accepted-only / accepted+review 指标

gaps_flower/regression_task.py
-> 构建回归模型和 source regression dataloader
-> 定义只训练/聚合 regression branch 的逻辑

gaps_flower/specialist_calibration_fit.py
-> 在 target calibration split 上拟合 per-client auto_v2
-> 比较 none、bias_only、affine_only、phase_affine_only、full、specialist、specialist_full 等模式
-> 生成 routing_config 和 calibration references

gaps_deploy/build_per_client_packages.py
-> 将 classifier、regressor、calibration、QC 所需文件打成 per-client package

gaps_deploy/inference.py
-> deployment 推理主流程
-> classifier forward
-> predicted class route
-> R3aK16 regression
-> auto_v2/full/specialist route
-> scalar calibration
-> risk score
-> QC decision
-> optional CO/R4A correction

gaps_deploy/qc_policy.py
-> risk score 计算和 accept/review/reject 双阈值决策

gaps_deploy/build_qc_policy_from_predictions.py
-> 从 calibration 或 prediction CSV 中选择风险列并生成 qc_policy.json
```

正式 ppm 字段口径：

```text
base_r3ak16_raw_ppm: 纯 base R3aK16 hard-route 输出
routed_pred_ppm: auto_v2/full/specialist route 后输出
final_ppm: scalar calibration 后、QC 前输出
co_corrected_ppm: CO-specific 增强层输出，不覆盖 final_ppm
```

`gaps_flower/regression_client.py` 和 `gaps_flower/regression_server.py` 仍有参考价值，但当前主线更多通过 `run_time_aware_raw_calibrated_qc_eval.py`、`gaps_flower/regression_task.py` 和 `gaps_deploy/*` 串起回归/QC 流程。

## 主线四：部署运行时与最终包

主要文件：

```text
gaps_deploy/final_runtime.py
scripts/build_final_deployment_package.py
scripts/validate_final_deployment_bundle.py
build_deployment_output_package.py
validate_package_runtime.py
```

功能关系：

```text
gaps_deploy/final_runtime.py
-> 面向最终部署包的运行时封装

scripts/build_final_deployment_package.py
-> 构建最终部署包入口

scripts/validate_final_deployment_bundle.py
-> 校验部署包字段、配置和运行时可用性

build_deployment_output_package.py
-> 汇总部署输出、报告或交付材料

validate_package_runtime.py
-> 运行时 package smoke test / sanity check
```

部署原则：CO correction 不应覆盖 `final_ppm`，应额外写入 `co_corrected_ppm`，以便保留 base、routed、final、corrected 四层审计口径。

## 主线五：报告、图表与会议材料

主要文件：

```text
scripts/build_meeting_report_visuals.py
plot_timeaware_regression_scatter_comparison.py
plot_co_correction_scatter.py
summarize_mainline_evidence.py
summarize_final_gaps_evidence.py
```

当前统一会议图表入口：

```bash
python scripts/build_meeting_report_visuals.py
```

快速只生成表格：

```bash
python scripts/build_meeting_report_visuals.py --tables-only
```

输出目录：

```text
results/meeting_report_visuals_20260623
```

报告中推荐同时给出：

```text
Full test
QC accepted
accepted+review
Rejected risk summary
CO high Bias / RMSE / P90AE
```

其中 `accepted+review` 是当前主要部署口径，`QC accepted` 表示自动可信输出，`reject` 集合应单独说明其高风险特征。

## 当前 CO high 相关诊断

CO high 是当前主要瓶颈之一。相关文件包括：

```text
audit_timeaware_ppm_layers_and_co_bins.py
diagnose_co_high_tail.py
run_co_specific_correction_eval.py
run_co_specific_client_extension_eval.py
run_co_specific_qc_threshold_tuning.py
run_c45_c123_co_high_guarded_correction.py
results/co_tail_rank_ablation_fixed_da_r25/co_tail_rank_ablation_report.md
```

当前 `co_tail_rank_ablation_report.md` 的结论是：loss-side tail/rank 改动还不是 CO high compression 的干净修复。可以把 `R1_tail2` 或 `R4_tail2_rank005` 保留为候选，但不建议直接用 `tail=3` 替换 baseline。下一步更适合沿 C4/CO high recovery calibration、显式 window/phase statistics 或 guarded CO-specific correction 方向推进。

## 不建议作为正式主入口的旧代码

以下文件或目录仍可用于诊断、对照或历史复现，但不要混入当前正式主线：

```text
preprocessor.py
split_dataset.py
scripts/create_c12src_c345tgt_calib20_test80.py
run_time_aware_split_backbone_gain_isolation.py
audit_time_aware_auto_v2_noop.py
run_regression_training_strength_ablation.py
run_weighted_regression_loss_ablation.py
dataset/client_data_c12src_c345tgt_calib20_test80
dataset/client_data_federated_window_fullgrid_src12_tgt345
gaps_flower - 1版本/
历史版本代码备份归档/
```

也不要跨 source/target 组合复用以下内容：

```text
norm_stats.npz
classifier checkpoint
regression package
auto_v2 routing_config
selected_policy.json
qc_policy.json
```

## 建议阅读顺序

如果是第一次接手当前代码，建议按下面顺序读：

```text
1. README.md
2. results/flower_training_regression_qc_workflow_20260623.md
3. CLS_ExpB_StrongDA_2080_Classification_Backbone_Freeze_Manual.md
4. gaps_flower/strategy.py
5. gaps_flower/domain_adaptation.py
6. gaps_flower/regression_task.py
7. gaps_flower/specialist_calibration_fit.py
8. gaps_deploy/inference.py
9. gaps_deploy/qc_policy.py
10. scripts/build_meeting_report_visuals.py
```

## Git 与文件管理

本仓库建议只提交核心代码和轻量文档。以下内容通常不要提交 Git：

```text
dataset/
results/
checkpoints
generated figures
large deployment bundles
临时诊断输出
```

如果需要共享大文件、模型 checkpoint 或完整结果目录，应单独打包或通过外部存储传输。
## Current Mainline Entry Points (2026-06-26)

The current reproducible mainline is documented here:

- [Mainline Entrypoints](docs/mainline_entrypoints_20260626.md)
- [Next-Stage Execution Plan](docs/gaps_next_stage_execution_plan_20260626.md)
- [Experiment Design and Acceptance Criteria](docs/experiment_design_and_acceptance_20260626.md)
- [Feature Schema and Runtime Contract](docs/feature_schema_and_runtime_contract_20260626.md)
- [Final Mainline Summary](results/gaps_final_mainline_summary_20260626.md)
- [H8+C4 Deployable Specialist Validation](results/h8_c4_deployable_specialist_validation_20260626.md)
- [Bidirectional Profile Selection](results/bidirectional_profile_selection_20260626/bidirectional_profile_selection_report.md)
- [Runtime Profile Benchmark](results/runtime_profile_benchmark_20260626/runtime_profile_benchmark_report.md)
- [Target Profile Selector](results/target_profile_selector_20260626/target_profile_selector_report.md)

Use these documents as the current entry point for classification, regression,
target profile selection, deployment runtime, and acceptance criteria. The older
README sections are kept as a historical code map and may mention legacy or
diagnostic scripts.
