# 浓度研究收敛

REGRESSION_CONSOLIDATED.csv包含Phase1/1.5/2B的72个结果视图、144条raw/clip主指标，统一71D、Oracle_ALL、A2/A4独立暴露和单窗口暴露等权评价。reused_alias指出Phase2B复用的F/N-SVR/KRR，不能把72视图说成72套独立训练。Source-only4个全源头跨目标/预算复用；源alpha1000、F/N参数、KRR与rho选择均取各阶段原锁。

Phase1 TargetRidge/TargetSVR是原源侧大样本选参；Phase1.5 F以4/8窗源任务选参，N以n_ref315换算。原Ridge平均27.995%到N-Ridge20.921%的修正是正则迁移局限的证据，不等于Ridge理论上不适用。KRR无稳定大增益已停止扩网格。统一表保留raw、clipped和辅助均值融合，不能把融合当部署单窗性能。

Phase2B源侧固定rho A2=.75/A4=1，两伪任务同日cal/validation、元参数复用整个Day7池，不独立确认。固定残差5/6总体改善，C4 A4 F-SVR17.6055→17.8624%负结果保留；各gas/浓度负迁移文件及困难高浓度切片SHA见evidence/REGRESSION_EVIDENCE_LINKS.json。C3/C5 Source-only分别12.2137/14.7792%，优于残差，但不得据此事后给每设备挑最优部署策略。当前共享rho方向停止扩展，H-CONSISTENCY/复杂门控均不实现。

## Source-only的标签和裁剪依赖

实际Phase2B study.py中SourceOnly调用FULL源头clip=False，随后所有输出统一np.clip(p,LOW[g],LOW[g]+SPAN[g])；这些常量是源侧已定义的gas-specific物理浓度范围，不是从目标test估计。因A2/A4预定两端点相同，它也恰等于原cal支持；Source-only运行代码本身没有读取目标cal数值拟合或决定clip，不能错误宣称它偷偷用了目标标定拟合。

但当前实验是**Oracle真实类别g路由、已知气体集合/量程/ppm语义下的Source-only浓度诊断**。实际系统仍需合法分类器、可用baseline/采样/传感单位、预测类别和验证过的输出支持；若加入目标F-SVR故障回退，整个系统就依赖该目标cal训练。A2/A4相同支持下的Source-only两行并不证明无标定新量程泛化。raw/clip都保留，不能以裁剪掩盖超界。故不把这些点估计写成无需标定系统已完成或跨设备最优策略。

336故障场景/672raw-clip断言包含rho0不访问source、非零源失败改走同一独立F头，以及源身份/索引成员异常；旧240场景及额外96场景都保存。本轮仅引用和哈希核对，不重新推理或改旧结果。均为回顾性；C4cal3晚用source7早于test9，不称即时Day3共同源部署。历史640暴露的确认性缺口继续保留。
