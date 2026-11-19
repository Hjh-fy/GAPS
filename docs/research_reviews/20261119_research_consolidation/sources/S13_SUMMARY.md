# Sprint13｜独立确认设计

日期：2026-11-19。**最终状态：DATASET_NOT_ELIGIBLE（本次审查的两个候选）。** 方法冻结完成；未取得可执行的独立联合确认队列。本目录只交付五个设计文件，不包含运行授权。

## 冻结什么

Per-gas record-LOEO保持12B原样：同一当前分类器、Source与当前Bias二选一、每气体2/4条记录、均值raw-NMSE折差小于−1e−12才接受。KEEP选择Source，必要时恢复参考版本；不沿用上轮Bias。分类器仍按每步当前标定记录更新，不由未来性能决定。完整数学、代码/模型来源及SHA见`METHOD_FREEZE_MANIFEST.json`。

12B的34.84%/43.65%及Joint的33.53%/40.02%仅引用封存摘要，本轮没有重新评分。既有八步不能重新成为确认数据；12B探索性GO不等于确认成功。

## 数据结论

|候选|核实结果|本任务判定|
|---|---|---|
|2025 Long-Term Drift|有按日测量与样品类别；但配制浓度属于液相，气相ppm真值不可用|不满足气体—气相浓度J10S|
|2022 Transfer Learning|有混合气各组分的气相浓度标签；无可直接沿用的单一真实气体类别，预切分也不能认作后续日期确认|不满足单类预测选头联合任务|

2025论文Methods的“Preparation of the sample concentrations”明确写道：
> All indicated concentrations for every substance refer to the concentration in the solution.

依据：[作者论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508210/)。因此，Sprint8 `EXTERNAL_DATA_NEXT.md`、Sprint10外部草案及12A `NEXT_VALIDATION_PLAN.md`中按文件名ppm认定可做气相回归的建议，**在本轮后不作为联合确认依据**。历史文件及加载器保持不动，旧字段`ppm`不能赋予气相物理含义。Day14缺失则已由论文Data Records解释为协议偏离后的剔除，无需补造该日。

2022任务定义依据：[官方数据说明](https://zenodo.org/records/6821340)、[作者论文§2.1–2.2](https://www.mdpi.com/2073-4433/13/10/1614)。不能把主组分/最大浓度临时定义成真实gas，也不能把多输出回归、随机训练/测试划分或传感器制造批次改称本次时间联合确认。

逐气体浓度档、每日文件容量、官方数量差异、历史访问与UNAVAILABLE项在`DATASET_ELIGIBILITY.csv`。本轮只查官方文字和既有身份索引，没有打开新原始信号、预提取特征或MAT数据。

## 确认框架与下一步

保留Joint、Classifier-only、Per-gas三项主组；Static/Regression-only作固定背景对照；Oracle-J只作事后上限。主预算为4条、2条为预定次预算。J10S继续用真实气相浓度及源跨度10%，共同正确路由raw NRMSE另列。配对差值、时间相关性、风险、版本和完整标签成本规则见`INDEPENDENT_PROTOCOL.md`；确认门禁见`CONFIRMATION_GO_NO_GO.md`。

当前不指定任何候选的源/标定/确认日，不生成训练命令。最小后续工作是取得带合法时间、单一气体身份和可追溯气相浓度的独立队列；或取得现有数据缺失的物理真值和身份依据后重新审查。液相浓度回归或多组分回归可以成为另一个任务，但不冒充本次指标的独立确认，也不在本轮启动。
