# 静态实验审计

## 审计范围与比较

仅重评冻结研究的可写主张，不重评分。逐阶段协议、模型、预算、统计单位、路由与固定版本在PROTOCOL_AND_VERSION_MAP.csv，参数/模型锁的精确文件与SHA在MODEL_AND_CONTRACT_REFERENCES.csv。

|发现|严重度|证据|影响与处置|
|---|---|---|---|
|旧窗口划分与新独立暴露不同|blocking for cross-protocol ranking|协议映射；ISCAS稿blocks41–43|不准跨协议数值排名或单因素归因|
|同S*与同目标预算不等总计算|major|P35 COMPUTE_AND_COMMUNICATION.csv|B1/B3只能分析目标/源保持/成本权衡|
|SourceOnly是必要直接强基线|blocking for broad residual claim|P2B README六行固定结果|撤回全面迁移优势，保留C3/C5强源结果|
|新PC和旧batch128合同不同|blocking for merged acceptance|P38/P39/P310固定报告|旧失败不改，新PC独立PASS|
|Pi严格logits超差但独立性能达标|blocking for full numeric PASS|P311/P312固定报告|数值BLOCKED与性能PASS分开|
|历史0.413ms预算5%/20%冲突|major/conflict|稿block65与实际benchmark manifest|旧稿保持只读，后续引用20%真实来源|
|风险稳定不保证开发保护|blocking for trustworthy constraint claim|Gate3与postmortem|不接入新Flower算法，不调QP|
|实际投稿文件与STM32固件未取得|unknown|本轮可用性记录|标记UNAVAILABLE，不推断独立验证|

## 泄漏与完整性

本轮没有读取原始信号、模型反序列化、测试推理或选参。历史已观察数据仍为回顾性；没有把旧源/C3缓存重新命名盲测。996条模型文件引用与原SHA匹配，仅确认字节可用。训练/恢复有效性沿用已审计固定报告，不把本轮静态检查升级为重新运行。

## 判定

E1–E6的窄范围实现/分析主张 approved；其限制构成主张的一部分。原创算法优越性、跨设备普适效果、最新完整在线闭环和严格Pi数值PASS为 blocked。历史0.413标定关联为conflict。不同主张不得共享无条件“全部通过”标签。

## 交接

experiment-audit → claim-evidence，输入为27行协议、固定阶段验收和来源SHA；已核对公平范围、基线完整性、负结果及版本边界；未解为精确平台提交身份、机制分解与新确认队列。仅授权窄范围论文论证，历史所有资产保持只读。
