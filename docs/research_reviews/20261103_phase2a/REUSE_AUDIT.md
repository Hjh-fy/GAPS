# 历史算法与源资产复用审计

REUSE_INVENTORY.csv逐方法给出维度、源训练、标定、缩放、损失、参数、气体路由、裁剪和协议；evidence_v2/V5_REUSE_PROVENANCE.csv保存28项v5字节SHA，STATIC_CODE_REFERENCES.csv保存函数起止行/片段SHA。源码生成commit未记录时保持unknown，不用当前worktree HEAD冒充生产版本。

## 已确认可以直接复用

1. ablation_core.source_module(71)已有维度绑定器；frozen_source_numerics的local_central_moments、merge_central_moments、local_normal_equations_v2、aggregate_normal_equations_v2、reconstruct_ridge_v2和pooled参考；不是重新实现FedRidge。
2. 同源的SourceRidge71预测器、state/schema检查，以及Phase1 strict71缓存、固定LOW/SPAN、TargetSVR71 fit/predict；F/N-SVR和KRR冻结参数/权重作为未来参照，不本轮推理。
3. alpha_core.base已有LOW+rho(fS−LOW)及0/1精确分支；Centered prior已有offset=(1−rho)LOW，训练(y−rho*fS−offset)/SPAN、预测rho*fS+offset+SPAN*h；choose_rho可复用为源侧已计算分数的排序器。

## 不可直接搬用的部分

旧71D银行alpha=.1/.01/.1/.1，源canonical train→source calibration选alpha、train+cal重拟合，覆盖所有R1–R4。104D源银行alpha=.1/.01/.01/.01，本阶段通过原legacy.validate_r0_bundle校验原index/model_lock并确认四头JSON等于v5副本；execution manifest明确source C1/C2、target空。104D元数据白名单在旧实验成立，不代表这些上下文已进入71D或实际部署可观测。

ResidualSVR用target83/source104，固定C10、gamma .1/83、epsilon .01，M0标签(y−LOW)/SPAN，M1标签(y−fS_clipped)/SPAN；训练自己的gas/cal scaler，先clip源、再clip最终物理范围。1%16窗/5%80窗与1360测试窗均属旧C5 canonical协议；标定分别16/80个原始文件同时出现在test中，只有精确窗口互斥，不能当独立暴露结果。

源贡献alpha旧LOEO以目标标定标签选共享/逐气体系数；Centered AdaptiveSourcePrior旧4折以目标标定OOF选共享rho，历史选中1%rho1、5%rho.25。它们是已实现先行机制，不是本轮新发明，也不适合直接继承到源侧选参的新比较。A2每gas只有2个暴露，旧LOEO要求至少3、4折要求至少4，两个入口均正确拒绝A2。A4可分4折也不授权用目标Development/Test重选参数，本方案不用目标cal选rho。

三个旧拟合入口都硬编码83D，本阶段已验证它们拒绝71D，不能直接调用，更不能补零12列假装复用。未来只能以原SVR71数值核、原base/Centered代数构建薄兼容边界；本阶段的test_interfaces仅是合成接口探针，不是可运行的生产迁移器。

## rho0：数值退化与运行回退

历史未经中心化版本曾不能复现M0：最大差.083503/.184622ppm，修正版保留失败现场，通过固定offset恢复原训练标签数值路径。不能仅因自由截距下代数等价，就认为有限容差SMO输出逐点相同。

本阶段2预算×4gas的71D探针复用原base及原SVR71，以相同合成样本分别拟合直接标签和rho0调整标签，再交给原Centered预测接口；训练标签、scaler、dual/intercept及raw/clip预测逐元素相同，最大差0ppm。没有改变历史83D入口，也没有在真实cal上训练。

同时确认旧predict_head无rho0短路，仍算0*source；source为NaN时结果NaN。未来回退必须显式返回**同一冻结F-SVR checkpoint**，不依赖源模型或源输入成功，不能仅rho=0后继续要求source可用。所有非零rho仍需完整源维度/单位/有限性合同。

## 审计结果

数值部件可复用；历史权重/历史rho/历史切分不能用于严格主比较。合法新输入合同通过，真实SourceRidge71模型及完整71D残差兼容接口尚未实例化验收，目标迁移训练未释放。没有新增神经网络、门控、Flower、部署或真实数据拟合。
