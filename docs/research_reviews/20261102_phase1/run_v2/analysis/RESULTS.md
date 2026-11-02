# Phase1 严格回归强基线结果

状态：12配置/48目标头完成；本轮仅Oracle_ALL。主指标是单窗口误差、暴露等权；暴露均值融合单列辅助，未运行预测路由。

|目标/协议|预算每气体|模型|主NRMSE %|主MAE ppm|辅助均值融合NRMSE %|裁剪比例|
|---|---:|---|---:|---:|---:|---:|
|C3 / C3_three_relative_dates_8_14_17|2|TargetRidge71|28.5628|41.7635|28.4774|0.000|
|C3 / C3_three_relative_dates_8_14_17|2|TargetSVR71|20.2826|23.7747|19.2715|0.000|
|C3 / C3_three_relative_dates_8_14_17|4|TargetRidge71|26.9504|39.1183|26.8206|0.000|
|C3 / C3_three_relative_dates_8_14_17|4|TargetSVR71|18.0119|21.1566|16.8658|0.004|
|C4 / C4_two_relative_dates_3_9_retrospective|2|TargetRidge71|28.5532|41.2537|28.4807|0.000|
|C4 / C4_two_relative_dates_3_9_retrospective|2|TargetSVR71|17.6591|21.2984|15.3683|0.007|
|C4 / C4_two_relative_dates_3_9_retrospective|4|TargetRidge71|27.2394|38.7408|27.1696|0.000|
|C4 / C4_two_relative_dates_3_9_retrospective|4|TargetSVR71|17.8881|21.4681|16.7052|0.017|
|C5 / C5_two_relative_dates_18_22|2|TargetRidge71|29.1473|42.9319|29.0786|0.000|
|C5 / C5_two_relative_dates_18_22|2|TargetSVR71|19.9296|25.7499|18.7265|0.000|
|C5 / C5_two_relative_dates_18_22|4|TargetRidge71|27.5163|40.4727|27.4014|0.000|
|C5 / C5_two_relative_dates_18_22|4|TargetSVR71|18.8327|22.1706|17.8634|0.035|

SVR在6个设备×预算比较中有6个主NRMSE更低。这是描述性计数，不能将不同日期协议合并为同一个泛化结论。CI见PAIRED_SVR_VS_RIDGE.csv，仅为给定模型和单个test日的条件性重采样，不能代表跨日期或跨种子稳定。

预算变化见BUDGET_DELTAS.csv：A2→A4同时增加样本数量和浓度档，不能单独归因于特征或覆盖。所有气体/浓度/已见未见/早晚时间片结果见evaluation/GAS_CONCENTRATION_SLICES.csv；核覆盖诊断不用于调参。

源参数只由C1 Day4、C2 Day1/7选择。Development未用于目标选择；C4源晚于其首日标定，只作回顾性source-before-test。历史test已被观察，本结果不是前瞻未触碰测试。

CPU成本为本机软件回归调用，输入71D已计算；不含原始采样、特征、分类、QC、通信或Pi硬件。训练/序列化/模型大小及逐窗调用时延分别见TRAINING_COSTS.csv与evaluation/INFERENCE_COSTS.csv。

没有额外调参、浓度重选、72D、KRR、残差、联合网络、Flower训练或部署。创新方向解释另见本轮审阅文档，不能从一次比较直接宣称新方法有效。
