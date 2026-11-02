# Flower Target CE、客户端与跨轮反馈静态审计

结论：实现可复用，新的严格协议运行配置尚未闭合。本轮仅AST/源代码检查，没有启动服务、客户端或训练。证据为FLOWER_ARG_DEFAULTS.json与evidence/FLOWER_STATIC_EVIDENCE.csv，路径/行号可追溯至v5。

|机制|实际门控/默认|必须满足的边界|状态|
|---|---|---|---|
|服务器策略|server --strategy默认fedavg|需确认真正实例化GapsStrategy；开DA参数不等于fedavg具有相同适配流程|已静态核对|
|服务器适配|--use-domain-adapt=False|显式开启、源/目标合法输入目录、warmup/steps/optimizer有效|配置待新协议绑定|
|Target CE|--da-lambda-target-ce=0.0；strategy同为0|正权重、目标logits与labels存在、进入总损失；不能从profile名称推断|已静态核对|
|跨轮反馈|--use-adapted-as-global=False；strategy同为False|DA实际运行且返回适配后参数，下一轮客户端接收到同一版本|静态可达；逐轮张量证据未跑|
|fixed_da_strong|预置steps100/warmup0/反馈True等；未把TargetCE设为正，也未单靠preset开启DA主开关|显式CLI覆盖preset，保存effective config|不能把preset当所有开关已生效|
|数据持有|server-val-data与server-calib-data供服务器DA使用|目标监督样本/标签到服务器，须报告数据持有边界；不能宣称标签天然仅留客户端|必须按新角色重导出|
|Target-as-Client|普通客户端可按client_id加载train loader；默认num_clients=4|C5需合法num_clients/数据目录/目标train=仅标定、采样权重、更新步数和服务端接纳一致|缺端到端配置证据|
|严格旧研究|TimedStrategy显式只接受两个客户端且client_ids=[1,2]|B1为服务器SourceCE+TargetCE，B0为Target-only；均不是目标作为普通第三客户端|已有7端点不补造TAC身份|
|客户端验证|--eval-split默认test|禁止读取目标test的逐轮结果进行早停/调参；新协议应使用源侧验证、禁用目标逐轮test，或声明额外development标签|执行前blocking|

需逐运行保存：训练/标定/development/test文件SHA、数据持有方、有效loss权重、训练轮数/目标更新数、聚合样本权重、服务器DA开关、实际返回参数哈希、客户端下轮接收哈希和原型/encoder版本。不能用“feedback=True”代替跨轮闭环发生的证据；联邦通信轮不等于一次新物理暴露。

后续若开展最小反馈对照，架构、初始模型、source cohort、目标标定、优化器、目标梯度步数、评估端点均相同，仅比较：(i)源模型、(ii)Target-as-Client且无服务器DA、(iii)服务器TargetCE且反馈False、(iv)同TargetCE且反馈True。TAC数据接口/权重/更新数未闭合时停止其行；不为凑齐矩阵实现新复杂网络。MMD/原型/维护均已有实现与负结果，暂不叠加。
