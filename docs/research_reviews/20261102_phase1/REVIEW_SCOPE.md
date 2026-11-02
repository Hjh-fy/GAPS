# Phase1 外部审阅范围

主结论与限制见README.md，创新预判见INNOVATION_AND_STOP_GO.md。当前只完成Oracle单窗矩阵；均值融合为辅助，预测分类路由NOT_RUN。不要把source-only参数迁移下的6/6方向性胜出解读为充分调校Ridge普遍弱于SVR：alpha1000网格边界和alpha/n差异是必须保留的major限制。

本包包含完整12配置主指标(raw/clip)、全部480行主暴露误差、全部648行主气体/浓度切片、源CV候选分数与折身份、成本、注册/审计/异常和哈希。逐窗结果提供96行按文件/时间固定抽样的审阅样例；完整10080行、完整960行raw/clip暴露表及全部1296行切片均在本地保存，ARTIFACT_SHA256.csv给出SHA。

code_review为实际执行脚本的文本审阅副本，仅替换本机路径及规范换行；不能直接当成跨机器可执行包。FROZEN_BASELINE_V5、PHASE1_WORKTREE和ORIGINAL_WORKSPACE是本地路径别名。原始信号、特征数值、模型权重/scaler和备份ZIP未上传。模型索引只含来源/参数/标定文件/文件SHA，不含系数。

本地运行合同和evaluation完成记录中的SHA指向完整本地原始制品。发布后的路径别名、紧凑投影与UTF8-LF换行使副本SHA不同，PUBLICATION_MANIFEST.json分别记录二者；副本不冒充原字节冻结资产。历史Phase0.5角色参见../20261102_phase05/evidence/ACQUISITION_ROLES.csv。

本轮没有新增分类器/KRR/残差/Flower/Pi，也没有Development调参或依据test重新选浓度/参数。sourceCV128+target48+synthetic3=179实际拟合调用，其中2个保存的smoke模型加1个归一化等价参考拟合，计数说明在任何fit前保存。首次run/准备失败0拟合记录保留，正式run_v2有效。

建议外部agent重点检查：1)主指标是否真正单窗口误差而非均值融合；2)绝对alpha源→fewshot迁移是否削弱Ridge；3)标签预算、source/cohort和test释放顺序是否一致；4)C3与C4/C5范围是否分开；5)KRR/残差下一步是否应先补强源侧fewshot参数选择。
