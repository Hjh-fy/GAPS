# 代码与现有结果复用决定

|资产|状态/复用|边界|
|---|---|---|
|v5 strict_data.parse_raw_name/process_arrays/calibration_indices|原字节导入并实际用于本轮640身份/320目标QA|角色选择另建，未改原处理算法|
|v5 canonical_quantitative_features + rich_feature_dict|71D同名同序，142个黄金行精确一致；96个标定窗探针|不把旧窗口角色搬为新角色|
|v5 ablation_core.py/svr_core.py/frozen numerics|复用特征索引和fit/predict/原数值；本轮不fit|新输入、参数选择来源与新模型索引要另建|
|旧SVR71/72 5%/20% 48头及12指标|历史冻结复现证据保留，不重算新比较|window-level Oracle诊断；已见过新test候选暴露，不能复用权重做严格test|
|旧SourceRidge71|保留历史参考|源canonical train覆盖R1–R4，非Day7队列；新72D需新合法source bank|
|C5严格少标定与系统成本20261020|已有7分类端点、文件/跨日隔离、L8/L16和正负结果，禁止同配置重复训练|其source R1–R3、完整源重训、83D/84D工程头与新71D计划不匹配；不是未见过test或TAC全矩阵|
|ResidualSVR/源贡献rho/联合v1v2/动态维护|reference_only，直接复用既有实现、账本与失败记录|不重新开发、不自动当新增创新或可靠增益|
|已有部署、串口、UI、旧Pi记录|保存为系统实现参考|本轮未部署、未重新计时；不可作为新SVR71端侧结果|

已存在严格C5端点逐项信息见evidence/EXISTING_STRICT_C5_RUNS.csv（只读原EXPERIMENT_REGISTRY.json生成，含真实ID/预算/目标CE/sourceCE/MMD/seed/原checkpoint位置）。原条目evidence_status=draft，完成训练不等于本轮重新审定全部科学主张。新增报告没有改写任何旧指标。

CHECKPOINT_REUSE_CANDIDATES.csv另将7个最终权重映射到v5实际副本，全部SHA一致，本轮未加载/推理。原C5协议下可条件复用它们及已存预测来研究路由，须先核对encoder/预处理/类别顺序/物理窗口身份与固定端点；共同Day7协议下不能直接复用其更晚源队列。C3/C4合法早期分类权重在本轮范围内尚未确认，不作全仓库不存在的断言。

既有C5源码strict_matrix明确：A0源FedAvg；B0目标独立；B1服务器SourceCE+TargetCE；A5服务器SourceCE+ClassMMD，7个端点。旧科学解读已记录较大预算不单调、方法有负结果以及仍需完整源重训；本轮保留这些边界，不能借重新命名变成历史模型免源重训或动态闭环证据。
