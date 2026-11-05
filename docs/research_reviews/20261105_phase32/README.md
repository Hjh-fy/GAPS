# FedOnGas Phase3.2：薄入口修复与重新预检

**接口与数据流预检通过，可单独批准一次S*源FedAvg预训练；本轮未训练。六组真实pilot仍全部BLOCKED_NO_SSTAR。** 依据v5、Phase3固定3794b71666358a597d5ed3cea65fadead8ff5719、Phase3.1固定b05c139d757a79994f8fc2fc983f1475052bdd45。

独立managed研究worktree新增contract.py、runtime.py、build.py及测试，不改旧strict/matched白名单或v5数值算法。最薄入口直接import原TCN/Client、Flower客户端fit、CheckpointFedAvg、GapsStrategy、ServerDA；不再用AST或数值训练替身完成最终优化计数。

|检查|结果/边界|
|---|---|
|7份spec与真实CLI effective_config|PASS；6pilot+1SOURCE；同50×8/4类/seed42/Adam5e-4|
|B1/B2目标循环|PASS：每组2500真实合成optimizer.step、40000呈现|
|B3/B4/B5服务器原DA循环|PASS：每组2500/40000，消费同一目标索引序列|
|完整源2520窗|PASS无梯度数据流；4.032MB特征，去除旧500截断|
|纯CE与原型统计|PASS：独立UPLOAD，phase0中性键；B5过warmup实际加权，B4不加权|
|反馈/encoder/下一轮接收|PASS：另做各分支两轮真实合成本地fit+原DA，无RPC训练替身|
|TAC权重|PASS：32/2552，不按重复呈现膨胀|
|真实S*与六组训练|BLOCKED：真实S*未生成；无授权文件、未启动任何真实训练|

阅读：[修复差分](REPAIR_DIFF.md) → [合同/命令](CONTRACT_AND_COMMANDS.md) → [测试与资源](TEST_AND_COST.md) → [一次源预训练](SOURCE_PRETRAIN_PLAN.md) → [恢复](RECOVERY.md) → [继续/停止](NEXT_DECISION.md)。EXECUTION_REGISTRY_LOCK为最终参数入口，cli_plans为真实解析输出；contracts/*_effective为早期构建草稿保留，不作为最终指纹。

本轮没有分类精度、端到端Test结果、网络隔离或Pi测量。旧结果/v5及其ZIP保持只读；470个Phase1–3历史payload与145个Phase3.1payload重新核验。所有合成模型/源码/失败日志和逐批收据留本地，GitHub只发审阅副本。
