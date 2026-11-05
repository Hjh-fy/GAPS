# 修复差分与算法边界

|Phase3.1阻塞|Phase3.2最小改动|复用的原实现|
|---|---|---|
|SEQ metadata100与输入50冲突|新spec强制50×8，TCN原结构不变；canonical shape/label/原SHA校验|原process_raw输出和TCN|
|无法加载共同S*|新随机源/继续训练分开；预加载原state原dtype/形状/有限值后strict load；祖先锚与25轮原制品核验|原完整state schema|
|服务器batch32导致80000|覆盖DA loader为固定16索引批，每轮100；每次实际yield写索引|原run_adaptation和Adam|
|源500截断|完整ConcatDataset840+1680；不改每步10SourceCE批+1特征forward|原SourceCE目标函数|
|ce_only无法上传原型|ce_only后仅独立开UPLOAD_PROTO_STATS，关闭原对齐/回放/回归/DP/增强损失|原Client统计均值/方差/计数和global feature函数|
|TAC/独立微调受旧ID限制|新B1 target action；B2普通第三客户端且num_examples32|原Client循环和FedAvg样本权重|
|隐藏Test评价|无eval loader，fraction_evaluate0；evaluate直接拒绝；数据包只允许已锁3文件|新入口边界，不改旧默认值|
|只检查返回但不检查分发链|原fit接收SHA、上传/encoder/原型内容SHA；服务器检查初始/前轮返回、顺序与人数|原Flower fit/aggregate/DA/EMA|

新validator不是关闭旧验证：原通用DA validator面向calibration_*与默认100；本入口使用Phase31锁定Source/Calibration角色的train_*精简包，逐字节canonicalSHA、身份、形状、类序、phase0验证更严格。无train/test fallback，无随机窗口混角色。非预注册ID/字段、额外Test/Dev文件、未知类别、非有限值、未知模型或错误版本均停止。

B3/B4除实验身份仅feedback不同；B4/B5除身份仅selective不同。共同初始原型为空、EMA.8、上传规则、warmup3/minscale.3、所有辅助loss系数0、SourceCE=TargetCE=1；普通初始网络feature noise.01/dropout保留为共同结构。维护原型、原型损失、参数权重分开：无原型损失不等于无统计。真实轨迹后期原型可因encoder反馈不同而自然变化，不强制不同轨迹数值一致。

独立review发现并修复：不能以自报digest链证明S*；绑定已注册随机初始lock及真实25轮receipt/checkpoint/SHA/源码/Source spec；在load_state_dict前检查原dtype；每气体精确210/420计数而非仅总和；检验64D方差/globalfeature及prototype_content_sha。最终审阅无critical/important未解决项。
