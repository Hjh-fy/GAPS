# Phase1 实验审计

审计对象：注册的12个Oracle配置及48目标头。审计范围为已约定source-only参数迁移流程，非所有算法调参策略的最优性。

|检查|结果|证据|
|---|---|---|
|源队列/时间|PASS；仅Day4和Day1/7；C4按回顾性例外解释|PRE_RUN_CONTRACT、source_metadata、源CV两客户端折|
|目标角色/预算|PASS；A2/A4原身份和固定档位；8/16暴露16/32窗|冻结ROLES与fit metadata；样本不按结果变化|
|特征/标准化/单位|PASS；71D同顺序；96探针精确；train-only scalers；相同标签尺度|PREPARATION_QA；MODEL_INDEX；SMOKE_QA|
|Development/test选参|未发生；只有源CV；48头锁定后释放test数值特征|FROZEN_PARAMETERS、ACCESS_EVENTS、TEST_RELEASE|
|主指标语义|PASS；暴露等权的单窗口误差；mean fusion另列|独立24公式复算、4单测、WINDOW/EXPOSURE CSV|
|模型与预测绑定|PASS；48头均有SHA，96固定窗口重放差0|MODEL_REPLAY_QA|
|完整性|12配置/48头/10080预测，正负结果未删|completed registry、evaluation complete、ANOMALIES|
|旧资产保护|640raw与v5数值/ZIP及原Git状态未变|FINAL_VERIFICATION与输入SHA|

### Major限制

M1：Ridge alpha源侧上界1000及source/target样本规模差异，限制“SVR普遍优于充分调优Ridge”的主张。当前只允许描述这组固定source-only流程的结果，不追加任何target选参。

M2：C4/C5非三日期，C4非即时双源上线；不能和C3混成同一种强泛化证据。

M3：每target仅一个test日、固定cal方案、确定性拟合；bootstrap是条件性暴露重采样，不能替代日期/种子稳定性。历史test已被观察，非前瞻确认。

M4：仅Oracle；没有合法预测分类路由合同，不能声称端到端浓度性能。均值融合不是当前单窗部署任务。

Verdict：**所声明的运行协议与数据/结果完整性核验PASS；方法最优性、真实路由、长期闭环与新创新增益均未验证。** 原registered与completed表保留，另生成audited表；此为程序性自审，不冒充外部同行评审。
