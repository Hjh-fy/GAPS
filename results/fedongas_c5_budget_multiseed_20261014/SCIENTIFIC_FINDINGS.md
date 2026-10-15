# C5 27 项预算/三种子研究：科学解读

来源：evaluation/target_results_24.csv、method_budget_summary_8.csv、paired_deltas_18.csv及paired_summary_6.csv。
数值为reported；收尾独立recomputed核算确认一致。范围为C5固定1360 test窗口，训练seed42/43/44，每method/budget n=3；SD为sample SD（ddof=1）。
未做显著性检验或跨exposure CI，不把1360相关窗口当作1360独立设备或训练重复。

## 结果表

|方法|5% Accuracy (%)|5% Macro-F1 (%)|20% Accuracy (%)|20% Macro-F1 (%)|
|---|---:|---:|---:|---:|
|TO|96.9118±0.1471|96.9100±0.1146|99.6324±0.1471|99.6323±0.1470|
|FT|97.6225±0.7142|97.6396±0.6995|99.0931±0.0425|99.0940±0.0437|
|CLASS|97.0098±1.2134|97.0140±1.2095|99.3382±0.3370|99.3392±0.3353|
|FULL|96.4216±0.7546|96.4342±0.7473|99.3137±0.3627|99.3138±0.3621|

NLL/ECE、逐类F1/recall、混淆矩阵和逐样本预测均保存在evaluation，不只保留有利指标。

## 预定义问题

1. **FT−TO：source初始化作用并非普遍正收益。**
   5%预算Accuracy配对差值+0.7108±0.6254 pp，三个seed均为正；20%为−0.5392±0.1123 pp，三个seed均为负。
   低预算可观察到source初始化收益；该证据不能扩展成所有预算或所有目标都受益。

2. **FULL−FT：完整interleaved方案没有稳定分类优势。**
   5%为−1.2010±0.0425 pp，三个seed均负；20%为+0.2206±0.3205 pp，两seed正、一seed负。
   20%的小幅平均改善不构成跨预算稳定优越性或显著性结论；也不能据此称FULL优于所有目标监督方案，因为20% TO的平均Acc/F1最高。

3. **FULL−CLASS：没有支持源历史phase组织的分类增益。**
   5%为−0.5882±0.7091 pp，三个seed均负；20%为−0.0245±0.1531 pp，一seed正、两seed负。
   20%数值很接近，而5%平均下降更明显；不能将极小差值表述为等效性已证明。
   C5全部为late，这只检验既定源phase组织和stage项对late目标的表现，不是target多phase研究。

## 描述性决策（不是通用算法定理）

- LOW_BUDGET_SOURCE_INITIALIZATION_BENEFIT_OBSERVED：仅本协议5% FT−TO。
- FULL_CLASSIFICATION_SUPERIORITY_NOT_SUPPORTED：本协议未获得跨预算稳定优于FT/TO的证据。
- SOURCE_PHASE_CLASSIFICATION_GAIN_NOT_SUPPORTED：FULL−CLASS未显示正的总体分类增益。
- WINDOW_PROTOCOL_ONLY_NO_SIGNIFICANCE_CLAIM：三个训练seed，不声称统计显著或未见exposure泛化。

这些结果应如实进入论文的边界和讨论：在条件覆盖完整的固定window划分下，纯监督目标训练是强基线，完整方法的分类精度优势不能预设。
阶段组织是否对quantitative regression/QC/lifecycle有其他价值不由本次分类研究回答。本轮完成后停止，不自动追加优化、回归、QC或新实验。

## 过程异常

FULL20 seed43的两个基础设施失败尝试全部保留并在MONITOR_FINAL_AUDIT.md说明。成功端点仍从原始随机初始化完整25轮产生，seed未替换，固定终点未按test性能选择。
