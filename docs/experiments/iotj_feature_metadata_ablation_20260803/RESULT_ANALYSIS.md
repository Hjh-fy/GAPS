# 83D / 91D / 104D 特征消融结果分析

## 结论摘要

本次五种子消融在冻结的 IoT-J 正式协议上完成。就当前数据资产而言，83D 纯传感统计、91D（83D + 8 个在线安全字段）和 104D 完整模式的回归结果在浮点精度内相同。该结果支持的窄结论是：**删除当前正式数据中21个无变化的辅助字段不会降低性能**。它不支持“在线安全元数据具有预测价值”或“104D 模型利用了丰富协议元数据”的主张。

## 实验范围

- 目标：比较 M83_SENSOR、M91_ONLINE_SAFE、M104_FULL。
- 正式种子：42、43、44、45、46；每个种子均使用自己的冻结分类路由。
- 目标划分：C5 calibration 320，test 1360。
- 回归器：按气体独立的 Ridge；校准集内以 60/20 选择 alpha，锁定后在80个样本上重拟合，再打开测试集。
- H1：同时报告不使用 H1 和追加冻结联邦源先验 H1 的结果。
- 主指标：S_CC NRMSE；数值下表换算为百分数。

## 五种子汇总

| 特征模式 | 最终输入维数（无 H1 / 有 H1） | S_CC NRMSE 无 H1，mean ± SD | S_CC NRMSE 有 H1，mean ± SD | 有 H1 的描述性 95% t-CI |
|---|---:|---:|---:|---:|
| M83_SENSOR | 83 / 84 | 7.8322% ± 0.1295% | 6.8324% ± 0.1443% | [6.6533%, 7.0116%] |
| M91_ONLINE_SAFE | 91 / 92 | 7.8322% ± 0.1295% | 6.8324% ± 0.1443% | [6.6533%, 7.0116%] |
| M104_FULL | 104 / 105 | 7.8322% ± 0.1295% | 6.8324% ± 0.1443% | [6.6533%, 7.0116%] |

补充指标方面，三组有 H1 的 S_ALL NRMSE 均为 13.8233%，无 H1 均为 14.4294%；同样仅有浮点舍入差异。

## 配对差异

- M83_SENSOR + H1 相对 M104_FULL + H1：平均 S_CC NRMSE 差为 `4.03e-11`（NRMSE 原始比例），相对差 `5.90e-10`。这不是具有实际意义的性能差异。
- M91_ONLINE_SAFE + H1 相对 M104_FULL + H1：平均差为 `1.18e-11`，相对差 `1.73e-10`。
- H1 在三个特征模式中的平均作用一致：S_CC NRMSE 下降约 0.0099976，即 0.9998 个百分点；相对下降约 12.77%，5/5 种子均改善。
- 预注册的 M83 对 M104 等效门槛通过：平均相对退化不超过5%，且没有任何气体在至少3个种子中恶化超过10%。

由于三种模式的设计矩阵只相差常数列，针对 M83/M91/M104 做显著性检验没有解释价值；`10^-11` 量级差异来自数值求解与舍入，不应描述为某一模式“胜出”。H1 的区间也应理解为五个冻结训练种子的描述性不确定性，而不是对独立物理设备总体的抽样推断。

## 元数据变化性审计

正式数据根目录的 C5 `calibration_experiment_info.json` 和 `test_experiment_info.json` 没有保存窗口时间、起始响应、插值比例、窗口内最大间隙或响应阶段等字段。特征构造器因此将缺失数值字段填为0，并把响应阶段编码为 unknown；数据中的 `phase_label` 又全部为 late。最终21列在 C5 calibration 和 test 中都没有变化：

- 恒为0的数值列：`center_minus_onset`、`center_minus_t_min`、`interpolated_ratio`、`max_gap_inside_window`、`t_min`、`t_onset`、`window_center_s`、`window_end_s`、`window_len_s`、`window_start_s`。
- 恒为1的类别列：`phase_id_2`、`phase_label_late`、`response_phase_unknown`。
- 恒为0的类别列：其余 `phase_id_*`、`phase_label_*` 和 `response_phase_*` 列。

按气体分别拟合且带截距的 Ridge 无法从这些常数列获得额外信息。因此，当前所谓104D模式在有效信息上等价于83D；追加 H1 后分别等价于84D有效输入。

## 论文使用建议

1. 当前最稳妥的正式表述是“目标回归器使用83维窗口传感统计，并可追加1维冻结的 H1 源先验”。
2. 若仍保留104字段实现描述，应明确它是接口/模式的维数，而冻结正式数据中的21个辅助字段未被填充、没有有效方差。
3. 不应将本次结果写成“8维在线安全元数据足以替代完整元数据”，因为这8列在本次数据中同样是常数。
4. 若要检验真实的91D在线安全模型，需要重新生成包含因果窗口元数据的数据资产，并从校准锁定开始重跑完整五种子证据链。

## 证据文件

- 汇总结果：`results/iotj_feature_metadata_ablation_20260803_r2/multiseed_summary.csv`
- 配对差异：`results/iotj_feature_metadata_ablation_20260803_r2/paired_profile_differences.csv`
- 分气体结果：`results/iotj_feature_metadata_ablation_20260803_r2/per_gas_multiseed_summary.csv`
- 决策记录：`results/iotj_feature_metadata_ablation_20260803_r2/feature_metadata_ablation_decision.json`
- 校准选择锁：`results/iotj_feature_metadata_ablation_20260803_r2/calibration_selection_lock.json`
- 文件哈希索引：`results/iotj_feature_metadata_ablation_20260803_r2/sha256_index.json`

