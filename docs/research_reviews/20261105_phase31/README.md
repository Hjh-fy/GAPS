# FedOnGas Phase3.1：目标感知联邦更新运行前预检

结论：**预检工作完成，六组正式pilot全部BLOCKED，不批准启动训练。** 依据v5、Phase0.5及Phase3固定3794b71666358a597d5ed3cea65fadead8ff5719。本轮独立worktree完成身份导出、真实CLI解析、原数值循环合成计数和Flower接口合成版本链；真实数据梯度0、S*预训练0、六组pilot0、目标Test输入读取/推理0、网络训练0、部署0。

|门禁|结果|实际证据|
|---|---|---|
|合法源/C5 A4输入|PASS|C1Day4+C2Day1/7，120暴露2520窗；C5Day18 16暴露32窗；复用原预处理/QC|
|合法共同S*|BLOCKED|2203个v5权重后缀条目普查、7重点端点安全元数据检查；无完整Day7祖先合同；未用旧权重顶替|
|六组CLI/profile|BLOCKED|实际parser接受的参数已保存，但通用入口缺S*加载/SEQ50绑定，严格入口拒绝新ID|
|目标客户端预算|PASS合成计数|32窗、batch16、50epochs×25调用=2500更新/40000呈现；仅Tiny合成模型|
|服务器预算|BLOCKED|原loader+原DA循环2500更新实际80000目标呈现；硬编码batch32；源还被截到500窗|
|反馈链|PASS合成接口|原fit/configure_fit/aggregate_fit/_run_domain_adapt/EMA/序列化；B3返回aggregated，B4返回adapted；数值训练用明确替身|
|B4/B5原型条件|BLOCKED生产路径|注入相同原型的合成隔离检查通过；真实ce_only不上传原型，B5不会激活|

阅读：[S*与源预训练方案](SSTAR_AUDIT.md) → [有效参数及真实命令](EFFECTIVE_CONFIG_AND_COMMANDS.md) → [数据/标签访问](DATA_AND_LABEL_AUDIT.md) → [预算实测与差异](UPDATE_BUDGET_AUDIT.md) → [版本链](VERSION_CHAIN_AUDIT.md) → [成本与继续条件](NEXT_DECISION.md)。逐组判定PILOT_GATE_MATRIX.csv；原生计数为反例证据，不是准许运行配置。

关键新增认识：TCN本体可处理50×8，实测22765参数、80个state张量。Flower传输全部state的张量载荷144692B，不能继续用91060B参数字节代替总state载荷。旧模型结构能严格加载，也不能证明祖先数据合法。纯CE、原型上传、原型损失、参数选择性加权是独立语义，当前通用profile没有把它们完全解耦。

只建议下一轮最小入口/合同补齐及一次源预训练，完成后重新预检；不扩大六组，不自动开始72分支。不以局部门禁通过替代整体批准。所有目标Test已观察，后续pilot仍为回顾性研究。
