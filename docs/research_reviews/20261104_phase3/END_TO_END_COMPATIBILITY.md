# 分类—71D回归兼容性审计

当前结论：共同Day7主协议的预测类别路由未释放；本轮0分类推理、0端到端指标。evidence/CHECKPOINT_COMPATIBILITY是7个历史严格C5权重的SHA核对，并非推理验收。不能把来源未闭合统称目标test已泄漏，也不能把shape匹配当作来源闭合。

|检查项|新回归合同|分类候选与门禁|
|---|---|---|
|采样与形状|5Hz、50×8、10s、stride5s|旧严格C5同称TCN50×8/64D；ONNX实际轴N×50×8或N×8×50和dtype未解析，需逐包schema|
|预处理|strict_data：按时间排序、重复时间合并，20–50s原始电导基线g0；5Hz分箱，单空bin线性补；(1/R−g0)/g0；crop60–170，float32有效窗|旧canonical与strict不能凭5Hz推等价；须固定相同baseline/单位/缺失掩码并以行身份/数值核对|
|特征|同预处理窗生成71D=64通道统计+7global，固定名字/顺序和scaler|分类encoder64D不是这71D；原分类checkpoint/分类ONNX不接收71D回归向量|
|标签|0Ethanol、1CO、2Ethylene、3Methane；气体专用ppm支持|逐权重class_map/输出头4类及temperature等完整SHA，不默认FLConfig或ONNX索引一致|
|训练时间/身份|SourceDay7；C3cal8/test17，C4cal3/test9，C5cal18/test22|旧严格C5源R1–R3，targetcal18/test22，可能满足原C5目标互斥，但不满足共同Day7主比较|
|模型祖先|新源与回归头索引有训练SHA|分类预训练/每轮targetCE/原型/DA输入/祖先checkpoint都不能含目标test暴露，不能只检查最后一轮train目录|
|当前模型|F/N-SVR/KRR及合法Source/残差均固定|旧窗口分类可能cal/test共享暴露；旧严格C5是条件候选，尚需初始祖先、导出manifest和原始窗等价；C3/C4未确认合法主协议分类器|

57个历史ONNX字节SHA和大小已核对，含诊断/优化图及classifier候选；当前onnx/onnxruntime均未安装，本轮不安装、不执行图，不猜其input shape/dtype/opset。具体graph结构和PC/Pi输出一致性仍pending；这是部署解析的明确前置，不代表模型不可运行。历史真实Pi案例、串口与UI可复用，不代表新组合已上Pi。

旧严格C5 checkpoint若完整证明祖先/输入互斥和预处理一致，可预注册**原C5源R1–R3的补充路由试验**，必须单列该source合同，不改名共同Day7主结果。当前未通过这些前置，不生成预测路由数值；不得用历史J0准确率与本阶段回归NRMSE拼表冒充系统指标。

未来路由必须先由分类器输出预测gas j，选择第j气体头、支持与单位；不能按真实gas挑头。错分气体的ppm单位/气体含义可能不同，主报告应同时给类别准确率、gas-concentration联合成功率（阈值事先源侧定）、错分类成本矩阵；Oracle_ALL仅作浓度诊断。若计算端到端归一化浓度误差，必须明确跨气体含义及真实gas仅用于离线评分，不用于运行路由。按true-correct-only子集得到的回归不代表全流量效果。

失败语义：source失败可用已批准的同预测gas F-SVR独立头回退，意味着整套回退系统仍依赖目标cal；分类器缺失/NaN/非法类别不能借Oracle补类别，应输出不可用状态，不给伪浓度。原始预测非有限值要在clip前拒绝，不让clip把Inf变成有限“正常”输出。不得根据test设备效果选择Source-only还是残差；模型策略必须固定或仅由源/合法cal规则决定并另注册。
