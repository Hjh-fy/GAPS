# 六组联合评价

| 模型 | Accuracy | Macro-F1 | Ethanol Recall | CO Recall | Ethylene Recall | Methane Recall | 分类异常 | 回归不可用 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 34.17% | 0.2519 | 16.19% | 0.00% | 20.48% | 100.00% | 0 | 0 |
| B1 | 83.81% | 0.8332 | 93.33% | 81.43% | 60.48% | 100.00% | 0 | 0 |
| B2 | 32.26% | 0.2353 | 18.57% | 0.00% | 10.48% | 100.00% | 0 | 0 |
| B3 | 81.90% | 0.8159 | 56.67% | 90.48% | 80.48% | 100.00% | 0 | 0 |
| B4 | 80.36% | 0.7839 | 90.00% | 37.62% | 93.81% | 100.00% | 0 | 0 |
| B5 | 79.88% | 0.7774 | 90.00% | 35.71% | 93.81% | 100.00% | 0 | 0 |

Accuracy、Recall 与 Macro-F1 使用 840 条全流量。四类混淆矩阵和预测类别的回归调用分母另列。[分类全表](evaluation/CLASSIFICATION.csv) · [4×4 混淆矩阵](evaluation/CONFUSION.json) · [预测类别调用及有效率](evaluation/PREDICTED_GAS_AVAILABILITY.csv) · [状态](evaluation/STATUS.csv)

浓度误差仅用分类正确且回归有效的窗口。各气体分别报告 raw/clipped MAE、RMSE、暴露等权 NRMSE 和分母。NRMSE 按对应回归支持跨度归一化，Ethanol/Ethylene 为 112.5 ppm，CO/Methane 为 225 ppm。暴露等权先对每个入选暴露的入选窗计算，再在入选暴露间等权。窗口覆盖使用该真实气体全部窗口作分母，暴露覆盖使用该气体全部独立暴露作分母。各组可比较的子集随分类而变。[逐气体误差与覆盖](evaluation/CONDITIONAL_BY_GAS.csv) · [浓度档位](evaluation/CONDITIONAL_BY_CONCENTRATION.csv) · [逐暴露](evaluation/CONDITIONAL_BY_EXPOSURE.csv)

错路由按真实气体、预测气体及真实浓度汇总。15 个同窗配对给出分类差异、调用头差异及同头 ppm 差异。无跨气体整体 NRMSE、联合成功率或系统排名。[错路由分解](evaluation/WRONG_ROUTE_BY_CONCENTRATION.csv) · [同窗配对](evaluation/PAIRED_ROUTING.csv)

六组均有 840 条数值有效的浓度输出。B0 和 B2 的 CO 正确路由覆盖为 0/210，条件浓度误差未定义。其中 CO→Methane 分别为 207 和 201 窗。六组 Methane 均覆盖 210 窗和 10 次暴露，其 clipped 暴露等权 NRMSE 为 10.25%。同窗配对的不同头调用为 20–579 窗。同头配对的 raw/clip 不一致数合计 0。

| 模型 | 旧 logits 超容差窗 | argmax 变化窗 | 最大 logits 绝对差 | 旧批量 Accuracy | 新单窗 Accuracy |
|---|---:|---:|---:|---:|---:|
| B0 | 5 | 0 | 0.00043582916 | 34.17% | 34.17% |
| B1 | 9 | 0 | 0.0004234314 | 83.81% | 83.81% |
| B2 | 0 | 0 | 0.00016403198 | 32.26% | 32.26% |
| B3 | 3 | 0 | 0.0001437664 | 81.90% | 81.90% |
| B4 | 22 | 0 | 0.00065612793 | 80.36% | 80.36% |
| B5 | 22 | 0 | 0.0013928413 | 79.88% | 79.88% |

旧参考仅用于 Legacy Comparison。旧容差超出保留为超出，Phase3.8 裁决继续 BLOCKED。旧分类参考来自 Phase3.5 C5 批量输出，其版本差异不影响单窗执行封存。[Legacy 全表](evaluation/LEGACY.csv)
