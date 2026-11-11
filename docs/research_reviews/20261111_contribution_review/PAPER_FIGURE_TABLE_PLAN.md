# 论文图表计划

本文件只规划，未重评分、绘制新性能曲线或选择模型。所有表按原实验顺序保留负结果，旧新协议不共用排名色标。公开稿只使用许可摘要与SHA，完整窗口身份和模型不发布。

|编号|拟图/表|研究作用与必要口径|精确固定来源|适用主线|
|---|---|---|---|---|
|图1|物理暴露—日期—角色时间轴|C3三日期、C4/C5替代、C4回顾性；窗口不充作独立样本|PROTOCOL_AND_VERSION_MAP.csv，[P1/README.md](https://github.com/Hjh-fy/GAPS/blob/772c0e9087435b9deb8ea4ac50b7c24814f6b345/docs/research_reviews/20261102_phase1/README.md)|A/B|
|图2|分类/71D/预测SVR双路径与版本状态|真实gas只评分；64D encoder不作71D；PC PASS/Pi严格BLOCKED分开|[P3.7/API_CONTRACT.md](https://github.com/Hjh-fy/GAPS/blob/ba7a143677404155e93083d0c43e15e9e72b4658/docs/research_reviews/20261106_phase37/API_CONTRACT.md)，[P3.10/README.md](https://github.com/Hjh-fy/GAPS/blob/ae2a614249443edaeae78d421851a3bdc57a92d6/docs/research_reviews/20261109_phase310/README.md)|A/C|
|图3|目标效果—源保持—监督成本并列面板|无事后综合分、不宣布B1/B3赢家；原CI原样保留|[P3.5/RESULT_ANALYSIS.md](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/RESULT_ANALYSIS.md)、[P3.5/COMPUTE_AND_COMMUNICATION.csv](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/COMPUTE_AND_COMMUNICATION.csv)|A|
|图4|源知识六配置完整对照|F/N-SVR、KRR、SourceOnly、残差，raw/clip分面，SourceOnly重复视图注明|[P2B/README.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/README.md)，本地run/OVERALL_METRICS.csv|A|
|图5|Calibration预测风险与开发有限步CE|按气体/快照分面；独立暴露计数；不把多候选当独立样本|[GATE3/RELIABILITY_TRANSFER_ANALYSIS.md](https://github.com/Hjh-fy/GAPS/blob/93c2f9fe445b0c8079cba8d52bae8f6279dd089e/docs/research_reviews/20261110_gate3/RELIABILITY_TRANSFER_ANALYSIS.md)，本地CALIBRATION_RISK_VS_DEVELOPMENT.csv|A/B|
|图6|四焦点固定alpha曲线与暴露分布|0/.25/.5/1全部保留；零带不变；不得把割线作起点梯度|[POSTMORTEM/FIXED_SCALE_ANALYSIS.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/FIXED_SCALE_ANALYSIS.md)、[POSTMORTEM/FAILURE_CASE_AUDIT.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/FAILURE_CASE_AUDIT.md)|B|
|图7|Pi分段成本及独立数值状态|latest联合P99与旧0.413不算速度比；重复工作负载不是独立暴露|[P3.12/PERFORMANCE.md](https://github.com/Hjh-fy/GAPS/blob/b4f6f664d7d6b25877f257d08b347387961c433d/docs/research_reviews/20261109_phase312/PERFORMANCE.md)、[P3.11/NUMERICAL_AND_RECOVERY.md](https://github.com/Hjh-fy/GAPS/blob/33dff2d8b34e033d6bd69458d4401719ad1fc577/docs/research_reviews/20261109_phase311/NUMERICAL_AND_RECOVERY.md)|A/C|
|表1|协议/版本映射简表|Oracle、预测路由、PC、Pi性能、历史在线/重放分列|PROTOCOL_AND_VERSION_MAP.csv|全部|
|表2|71D少样本正则迁移和强基线合同|原alpha1000、F/N规则及无目标选参边界|[P1.5/FAIRNESS_AUDIT.md](https://github.com/Hjh-fy/GAPS/blob/4988abb3160c0b0b9230a48a817f2d7baed137d2/docs/research_reviews/20261102_phase15/FAIRNESS_AUDIT.md)|A|
|表3|六组真实训练条件|同S*、目标预算、额外源监督、反馈/原型单因素对照|FEDERATED_FAIRNESS_REASSESSMENT.md及其固定引用|A|
|表4|迁移主张分层|5/6胜F、2/6胜SourceOnly、1/6同时胜；不得混称|SOURCE_KNOWLEDGE_STRONG_BASELINES.md及其固定引用|A|
|表5|逐气体混淆与条件回归覆盖|所有六分类器；各正确路由子集不一，不比较混合ppm|[P3.10/RESULTS.md](https://github.com/Hjh-fy/GAPS/blob/ae2a614249443edaeae78d421851a3bdc57a92d6/docs/research_reviews/20261109_phase310/RESULTS.md)，本地逐气体条件回归摘要|A/C|
|表6|最近邻机制与双维贡献矩阵|区分机制重叠、分析差异及证据等级；无原创首创暗示|NOVELTY_NEAREST_NEIGHBOR_MATRIX.csv、CONTRIBUTION_EVIDENCE_MATRIX.csv|全部|
|表7|系统验收版本边界|已实现、真实设备重放、最新离线、同版在线缺口分列|SYSTEM_VERSION_BOUNDARIES.md与HISTORICAL_SOURCE_SUMMARY.csv|A/C|
|表8|会议扩展增量及未解决项|本地稿SHA、实际上传身份UNAVAILABLE、0.413预算冲突|ISCAS_EXTENSION_AND_REUSE_AUDIT.md|B/扩展声明|

图5/6不得添加未经预注册的风险阈值或“最优alpha”。若以后需要新图，直接从封存表作可复核描述，仍不得因图形效果改变指标定义。
