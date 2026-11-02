# FedOnGas Phase 1｜严格71D回归强基线交付

依据只读v5与Phase0.5固定提交a6aa42c91e2e2b562371517f3f512f6a251206b9。本轮在既有独立managed worktree中完成，正式结果目录为run_v2；初次run/只包含准备阶段被沙箱阻断的记录，0拟合，不是另一组调参结果。

## 已完成与主要数值

12配置、48个目标气体回归头、10,080条逐窗预测、960行暴露级错误表（含raw/clip）、完整气体/浓度/seen-unseen/早晚时间片与成本均保存。结果均为**Oracle_ALL、单窗口预测误差的暴露等权汇总**；主指标不是暴露平均预测。Development没有参与拟合、参数选择或早停；没有运行旧分类checkpoint、预测类别路由、KRR、残差、联合网络、Flower或Pi部署。

|目标/日期协议|每气体标定|Ridge NRMSE %|SVR NRMSE %|SVR−Ridge pp|
|---|---:|---:|---:|---:|
|C3 / 三日期|2|28.5628|20.2826|-8.2802|
|C3 / 三日期|4|26.9504|18.0119|-8.9385|
|C4 / 两日期替代、回顾性|2|28.5532|17.6591|-10.8941|
|C4 / 两日期替代、回顾性|4|27.2394|17.8881|-9.3513|
|C5 / 两日期替代|2|29.1473|19.9296|-9.2176|
|C5 / 两日期替代|4|27.5163|18.8327|-8.6836|

主NRMSE=100×sqrt(mean_exposure(mean_window((error/RANGE_g)^2)))。每配置40测试暴露、840有效窗口；本例各暴露均21窗，所以与全窗池化数值相同，但统计单位/实现仍按暴露定义。mean_window_prediction的误差只在aux_mean_fusion列中作为辅助结果。全部未裁剪预测与指标同时保留。

## 四个研究问题的结论

1. **在本次固定源侧选参流程内，SVR在6/6设备×预算比较中均低于Ridge；不能扩展为“充分优化的SVR普遍优于充分优化的Ridge”。** 6个条件性配对bootstrap区间均落在0以下，但只反映给定模型、单个测试日的40次暴露重采样，不是跨日/跨种子验证，也没有作多重比较显著性主张。
2. **瓶颈有明显气体和浓度差异。** C3 A2主要是Ethylene（29.0101%），50/37.5/62.5ppm较差；A4下降到21.9956%，但Ethanol仍21.4225%。C4主要是Ethanol高浓度（A2 25.5384%，A4 23.6465%），125/112.5ppm突出。C5 A2以CO（25.2087%）和Ethanol（23.2552%）为主；A4仍是Ethanol（23.8944%）和CO（22.0711%）。125ppm Ethanol、250ppm CO端点已在标定中却仍有较高误差，所以不能把问题全部归为“测试浓度未见过”。
3. **更多标定并不单调改善每种气体/设备。** SVR整体A4−A2：C3 −2.2707pp、C4 +0.2290pp、C5 −1.0970pp。C4 Ethylene从14.8099%升至19.1087%，抵消了Ethanol/CO的改善。新增样本量、浓度覆盖、目标训练scaler和有效正则强度同时变化，不能仅凭此判定71D特征失效或浓度档最优。Oracle仍有17.66–20.28%量程NRMSE，说明即便真类已知，浓度适配本身也未解决；不能将当前误差归咎于错分类。
4. **KRR和71D残差均为条件性下一步，不在本轮追加。** 首先补强少样本正则迁移的公平控制，再考虑一次低成本RBF-KRR损失对照。残差方向须先建立合法Day7 SourceRidge71并检验Source-only先验；复用已归档机制和rho=0回退，不能复用时间越界权重或把既有模块算新创新。详见INNOVATION_AND_STOP_GO.md。

## 必须公开的正则迁移限制

源侧两折选择的Ridge alpha四气体全部为预设网格上界1000。源折训练每气体210/420窗，而目标A2/A4每气体仅4/8窗。原Ridge最小化sum squared loss+alpha×penalty；换成平均损失表述，lambda=alpha/n，即目标250/125，源折约4.76/2.38。源侧绝对alpha直接迁移导致目标有效惩罚明显增强，是本轮Ridge接近均值收缩的重要实现机制和比较限制。SVR的有效正则也依赖样本数，源侧选参规模与目标few-shot任务不匹配需单独控制。

我们没有在看到测试后更改alpha、重开网格或重跑。当前数值应完整报告为“预注册source-only超参数迁移基线”，不包装成Ridge能力上限。下一轮若获准，应只在合法源侧模拟A2/A4任务或比较alpha/n归一化规则，提前注册后再评估；这是公平基线补强，不是原创算法。

## 数据、模型与执行合同

- 源仅C1 Day4、C2 Day1/7共120暴露/2520窗；源侧leave-client-out2折，7个Ridge alpha、9个SVR(C,gamma)候选。目标为原A2/A4档位，cal固定[60,70)、[150,160)两窗，未重新选浓度。
- 71D按v5严格64通道+7全局原公式和顺序；96个标定特征行与Phase0.5探针精确一致。所有训练scaler只用自身训练数据，两方法同标签归一化、支持区间和测试掩码。
- 源参数锁定→注册12配置→fit48头→ALL_MODELS_LOCKED→TEST_RELEASE→一次性Oracle评价；完整事件链在run_v2/ACCESS_EVENTS.jsonl。身份/真值元数据和历史测试早已观察，不虚称全新前瞻测试。
- 正式模型fit为128次源CV+48目标=176；冒烟保存2模型并有1个ppm/归一化等价参考拟合，共3次合成fit，总179。完整计数说明在任何fit前写入PRE_SMOKE_COUNT_CLARIFICATION.json，不是追加参数候选。
- 首次准备的Git只读状态检查被沙箱阻断，0模型fit；保留run/，正式run_v2重新构建。初始单测系统temp路径受限后只改为本worktree内临时夹具。没有修改冻结代码、预处理、数据或结果来绕过科学门禁。

C3为Day8/14/17三日期；C4/C5 cal/dev同日、test次日的两日期替代分别报告，C4 source晚于首日cal，只能作回顾性source-before-test。绝对日期未知。标签是每次气体暴露的设定浓度，本轮不验证时间连续变化的瞬时浓度真值或传感器物理响应时间。

## 如何查阅和复现

优先读run_v2/analysis/RESULTS.md、INNOVATION_AND_STOP_GO.md、EXPERIMENT_AUDIT.md。

|内容|文件|
|---|---|
|预执行定义/实现|CONTRACT.md；run_phase1.py；phase1_common.py；analyze_phase1.py；test_phase1.py|
|源输入/实际数据|run_v2/data/*.npz及metadata.json；ROLES_A2/A4.csv；PHASE05_QUALITY_MASK.csv|
|合同和源选参|PRE_RUN_CONTRACT.json；SOURCE_CV_RECORDS.json；FROZEN_PARAMETERS.json|
|注册表|EXPERIMENT_REGISTRY_REGISTERED.csv；EXPERIMENT_REGISTRY_COMPLETED.csv；EXPERIMENT_REGISTRY_AUDITED.csv|
|模型|run_v2/models/*.joblib（48正式头）；MODEL_INDEX.json；ALL_MODELS_LOCKED.json|
|窗口/暴露/切片|evaluation/WINDOW_PREDICTIONS.csv；EXPOSURE_ERRORS.csv；GAS_CONCENTRATION_SLICES.csv；OVERALL_METRICS.csv|
|成本|PREPARATION_QA.json；FROZEN_PARAMETERS.json wall_seconds；TRAINING_COSTS.csv；evaluation/INFERENCE_COSTS.csv|
|分析/异常/验证|analysis/PAIRED_SVR_VS_RIDGE.csv；BUDGET_DELTAS.csv；GAS_BOTTLENECKS.csv；KERNEL_COVERAGE_DIAGNOSTICS.csv；ANOMALIES.json；FINAL_VERIFICATION.json|

阶段命令为python run_phase1.py prepare / smoke / select / fit / evaluate，随后python analyze_phase1.py和python verify_phase1.py；这些阶段在已有输出处拒绝覆盖。此正式run_v2已完成，不要原地重跑。新复现须使用新的研究目录及记录化配置，不能直接改锁定代码或参数。所有原始/v5资产只读，选择后的数据/程序/模型/结果均有SHA。

CPU成本只是本机暖模型、预计算71D后的逐窗回归调用，不含特征、分类、QC、采样或网络，更不是Pi实测。本轮两方法都很轻，不能用这些数字宣称物理传感器响应速度。

## 最终核验与停止

独立核验：640原始SHA未变；v5选用数值源码/独立ZIP未变；原工程跟踪状态未变；96个已保存目标头窗口重放精确一致；24组指标从逐窗CSV复算一致。0新增超参数轮、0旧分类checkpoint、0KRR/残差/神经网络/Flower/Pi运行。

本轮到12配置停止。结论是获得了可追溯的严格Oracle基线及明确限制，而不是证明了新复杂方法、真实路由性能或长期动态维护。精简GitHub审阅副本会保留主表、负结果、参数边界和固定commit链接，模型/原始信号/特征数值留本地。
