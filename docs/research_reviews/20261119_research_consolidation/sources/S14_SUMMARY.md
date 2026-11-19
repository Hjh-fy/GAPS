# Sprint14｜独立数据可行性

2026-11-19。**最终状态：DATA_GAP_CONFIRMED。** 当前没有满足冻结五气体时间联合确认任务且全部前置条件闭合的数据。没有训练、预测、查看模型性能或重新评分12B；没有访问Twin Day22、连接Pi或修改旧资产。

|候选|实际文件核查|资格|
|---|---|---|
|Methane/Propane|17个日期；1536个四传感器CSV视图；按日期/气体/浓度/序号合并384组，其中甲烷96、丙烷96、空气192|STRUCTURAL_REPLICATION_ONLY，条件性两气体时间任务|
|UCI1081 Low-concentration|90行×9002列；6气体×3档×5条；无日期/批次字段|STRUCTURAL_REPLICATION_ONLY，非时间任务|
|UCI308 Flow Modulation|58个exp、928个传感器行；5会话/4相对日；纯气体、混合物与空气并存|NOT_ELIGIBLE，原气相浓度联合任务不适用|

## 关键发现

1. **甲烷/丙烷最值得保留。** 作者标注40/100/200 ppm，17个真实日期目录可访问。四个sensor文件共享同一序号时只计一组；每CSV内成对温度/电阻行及连续一分钟周期均不增加标签记录数。多数日期每气体6组，但2019-01-31与2019-02-20只有3组，A4不足。文件组是否对应各自复位后的独立暴露仍未证实；数据许可、浓度计量来源及逐文件完整SHA也未闭合，不能直接READY。第一份CSV仅取64KiB结构前缀，前两行各600列，未下载整个原始信号库。
2. **Low-concentration数量够，时间不够。** 每气体15条完整记录，标签是50/100/200 ppb；可以设计跨记录的三角色任务，但不能用行序、发表日或下载日造出跨日期维护。6类与原5类集合不同；单位换算ppb→ppm可行，不能解决时间身份缺失。气相标签的独立计量溯源尚未核验。
3. **Flow Modulation有时间分组，但没有原含义的气相浓度目标。** `ace_conc/eth_conc`对应配制溶液vol.%；混合样品也不能变成单一真实气体。即使只留纯气体，也仅4天，且后续单会话多不足4条/气体；不构成八步/长期时间维护确认。原始16行是同一个exp的16个传感器，不能算16次测量。

以上两份UCI数据为CC BY4.0；作者GitHub/Drive未查得明确数据再利用许可证，标UNSPECIFIED，公开可下载不自动等于已授予再分发许可。原作者的随机切分/示例模型没有执行、没有读取其性能输出。

## 交付与下一步

- `DATASET_ELIGIBILITY.csv`：三个候选的单位、范围、预算、历史使用、许可、缺口与资格。
- `DATASET_RECORD_COUNTS.csv`：248行分层计数；`aggregation_level`不同的行不能重复相加；`independent_physical_count=UNAVAILABLE`不填成文件数。
- `PROPOSED_INDEPENDENT_PROTOCOL.md`：与原确认合同分开的两个结构性任务草案；当前均未授权运行。
- `LAB_COLLECTION_FALLBACK.md`：五气体、可追溯气相真值、完整暴露级补采；最小140次、8日期的先导与满足现有时间推断下限的扩展成本分开列出。

优先补齐甲烷/丙烷的许可与暴露身份，作为**两气体结构性时间复现**候选；它不能替代原五气体确认。若坚持原联合确认主张，则需补采或新的合格五气体队列，不增加算法规则来补数据缺口。

## 证据和边界

官方来源：[作者发布页](https://lcpssm.github.io/)、[固定作者代码](https://github.com/Lcpssm/Lcpssm.github.io/tree/da0ba0bbf4ab7646a617e8ae0b06b2e4247cab34)、[UCI1081](https://archive.ics.uci.edu/dataset/1081/gas+sensor+array+low-concentration)、[UCI308](https://archive.ics.uci.edu/dataset/308/gas+sensor+array+under+flow+modulation)。文件结构和计数来自本地下载/目录清单，不是仅按网页估计。

`evidence/fetch1/FETCH_RECEIPT.json`、`drive1/DIRECTORY_RECEIPT.json`、`drive1/FILE_IDENTITY_INDEX.json`、`uci_structure1/STRUCTURE_COUNTS.json`及`COUNTS_RECEIPT.json`保全来源和SHA。Drive188个目录页均成功，未进入merged数据；UCI只转换标签/身份，信号数值未转换成分析特征。历史检索发现UCI1081曾出现在Sprint8缓存Crossref的参考文献（327/330行），未发现这三项用于本项目数据拟合或调参的证据；不能据此保证不存在未记录访问。本轮已查看的标签元数据不再称盲态。

新目录含证据与复现计数脚本；未合并main。Sprint13的2022/2025不适用结论未重新审计。12B方法和Sprint13原确认合同保持冻结，完整性结果见`evidence/FINAL_VERIFICATION.json`。
