# 更新预算：实际循环计数与不能满足的部分

## 数据流与原Client循环

真实导出loader只遍历、不优化：C1 840→53批(52×16+8)，C2 1680→105批，C5 32→2批×16，drop_last=False。合成Tiny模型运行**原Client.train_one_round**及原CE/禁用辅助损失函数，计数optimizer.step及实际CE标签行数：C1一轮53/840；C2一轮105/1680；C5每调用50epochs、调用25次→2500/40000。原Adam每次调用重建，LR5e-4；reg权重取0测试替身只作用于已关闭、返回0的回归项，不影响目标CE。

完成的证据在checks_v2/CLIENT_GRADIENT_BUDGET.json；这证明原循环与batch16的计数关系，未证明未来六组已消费同一采样序列，也未训练真实源/目标模型。PROPOSED_TARGET_BATCH_SCHEDULE固定2500个16索引批，每窗1250次、合计40000；hash锁在SAMPLER_LOCK_PROPOSAL，但尚未绑定真实runner。未来必须记录实际索引消费逐批匹配；不能只用shuffle seed或名义epochs推定。

## 服务器实际反例

原GapsStrategy._init_domain_adapt_loaders与原ServerDomainAdaptation.run_adaptation在合成100×8输入、Tiny模型、零辅助损失测试替身上实际跑25×100次Adam更新。采用100是隔离原验证器默认schema；没有把研究数据50填充成100。真实研究入口在schema验证时即会拒绝50，其后潜在预算冲突仍可在合成合法shape上复现。

结果：2500次optimizer.step、目标80000次（每步32）；计划要求40000。不能通过减成1250步、改变A4数据或复制成不同“独立样本”解决，否则又违反固定2500步/16暴露合同。最小修补是未来薄入口显式batch16和共同采样序列，然后重新计数，而不是改冻结算法。

原server另把2520源窗RandomState42无放回截取500；每步SourceCE取10批，另1批源特征forward，即便所有辅助loss系数0也仍执行。25次适配调用观察到859600源forward呈现，其中按循环结构与批大小分解CE781400、附加forward78200。前者为参与sourceCE的重复标签呈现，后者只forward；都不是新独立标签。500源子集尚未登记身份/浓度覆盖，不能静默把完整源池当全量服务器CE。下一轮先明确去封顶或固定合法源子集身份及source每步批数，不根据目标Test效果选择；B3/B4/B5必须一致。

## TAC和比较范围

真实CheckpointFedAvg.aggregate_fit合成3客户端840、1680、32，只有target参数置1，其余0，输出每个state值=32/2552=0.01253918495（约1.254%），与预期一致；上传num_examples来自dataset长度，不是1600/round呈现数，不设1/3。旧严格/匹配runner仍拒绝第三客户端，不能用旧runner已完成宣称TAC合同已经闭合。

B1只目标本地更新，B2三个客户端联合平均，B3–B5两个源本地更新+服务器混合CE。全组只计划匹配共同S*与目标监督预算，不是完全等算力、等目标函数单因素比较。真正单因素仅B3/B4反馈、B4/B5参数权重。全部保留源CE、本地/服务器计算、重复标签呈现与成本差异。
