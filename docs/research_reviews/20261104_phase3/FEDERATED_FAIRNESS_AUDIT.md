# 联邦分类已有证据与同名冲突

本轮静态审计不启动Flower。evidence/V5_ASSET_PROVENANCE记录79项v5文件字节SHA，FLOWER_CODE_EVIDENCE记录实际路径/行号；STRICT_CLASSIFICATION_RUNS逐项抄录7端点有效配置，MATCHED_MECHANISM_MATRIX保留3组已有匹配消融。历史记录中的缺项不补猜。

## B0–B5必须带实验命名空间

|命名空间|真实语义|预算/更新/来源|复用范围|
|---|---|---|---|
|202607 v3 B0|已检查handoff未给出独立B0运行合同，unknown|不能把A0或别处B0自动补到该行|停止跨表自动映射|
|202607 v3 B1|修正CORAL|旧C5窗口cal320/test1360；25轮、local5epochs、batch32、Adam5e-4；server100steps/round|历史监督适配；非Target-as-Client|
|202607 v3 B2|global/class MMD2|同旧窗口；源起点SHA/实际开关须回到run-spec核对|已知test排名后筛出的探索性候选|
|202607 v3 B3|B2+stage|同旧窗口与目标cal标签|不是“仅反馈”的对照|
|202607 v3 B4|B2+adversarial|同旧窗口与目标cal标签|不是“仅原型”的对照|
|202607 v3 B5|B2+CORAL+stage+adversarial|同旧窗口；模块叠加收益方向相关|不能等同新主方法|
|202610 strict A0|source-only FedAvg|源R1–R3、固定round25、server0、target labels0|非共同Day7源；C5严格日期条件内参考|
|202610 strict B0|目标独立微调|同架构22765参数；L8/L16=16/32窗；2500updates、25blocks、batch32、lr5e-4|不是目标作为客户端参与联邦|
|202610 strict B1|server SourceCE+TargetCE|25轮×100server步=2500、local1epoch、选择性聚合+EMA.8|接纳两个源客户端，不是普通第三目标客户端|
|202610 strict A5|server SourceCE+ClassMMD.5|TargetCE=False；预算L8/L16提供target输入|不能因有target_y目录就说用了TargetCE|

202607分类 B1–B5在handoff声明使用目标cal标签，属于calibration-assisted DA；单组最终effective CLI与初始源checkpoint SHA并未在本轮逐run闭合，因此该处保留unknown，不以描述名证明逐步训练合同。已完成的严格7端点在evidence中列出target_x_n、target_y_n、loss开关、步数、模型/权重SHA；初始化SHA在该registry缺失，同样不能宣称全部组共享起点已被本轮证实。

## 已有最有价值的匹配消融

20261025 C5 M0/J0与M1仅关闭模型反馈：每轮仍100步适配，仍评价adapted模型；下一轮客户端M1接收aggregated、M0接收adapted。原报告记录M0=97.353%、M1=93.897%，固定25轮差3.456pp；训练链验收225个状态指纹、150条客户端接收链。**这是旧单seed/窗口协议的限定证据**，不是新Day7独立暴露或跨seed结论。

M2仅去Class-MMD，终点98.529%高于M0；M3仅去模型参数原型加权而保留EMA，终点97.868%亦高于M0，但源节点保持及后期平均存在不同权衡。不能预设Class-MMD或原型加权都带来目标增益，不能把后5轮当5个独立重复。以上数值直接引用原RESULT_INTERPRETATION，未重算替代指标。

## 当前代码的实际门控

server默认strategy=fedavg、use-domain-adapt=False、da-lambda-target-ce=0、use-adapted-as-global=False；client默认eval-split=test。fixed_da_strong预置不自动开启全部DA/正TargetCE，CLI覆盖必须记录effective_config。domain_adaptation.py的TargetCE需正权重、target logits/labels齐全并加入总loss。strategy.py反馈须实际返回adapted参数给下一轮，不是仅保存文件；任务模块USE_REG_LOSS=False，profile再设置align等，不得把gaps_cls当纯CE。

选择性参数聚合利用客户端原型与EMA语义参考；维护原型、加入原型损失、参数加权是三种不同开关。其原型主要来自参与客户端上传，不能未审计就称直接按新目标原型加权。目标感知可能通过适配参数回传改变下一轮encoder及原型间接产生，需逐轮版本证据检验。下一轮B4/B5保留同EMA、同原型上传、关闭原型损失，仅改参数权重，否则不是单因素。

服务器Target CE意味着服务器可访问指定标定输入/标签；不是标签天然仅留端侧。Target-as-Client用target train loader与普通FedAvg接纳，需第三客户端ID、标签角色、调度/权重、步数全部闭合。旧TimedStrategy固定两源客户端，不能把它的B1改名TAC。所有新目标test在锁定最终轮前禁止进入客户端/服务器evaluate回调。
