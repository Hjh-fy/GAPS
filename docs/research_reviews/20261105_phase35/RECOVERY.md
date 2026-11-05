# 恢复与不可重跑约束

主要代码：entry.py为入口，launch.py为一次性协调器，audit.py为无训练验收，evaluate.py为全部端点锁定后离线评价；冻结Phase33R runtime.py/contract.py和v5原client/strategy/DA为数值内核。registrations/Bx.json含真实argv，specs/authorizations分别独立注册。Python/依赖继承Phase33R ENVIRONMENT_LOCK.json。

主要输出：runs/Bx/server/server_round_025.pth（B0/B2）；runs/B1/target/target_update_2500.pth；runs/B3、B4、B5/server/server_round_025_adapted.pth。锁和SHA见locks/Bx.json、MODEL_VERSION_INDEX。observer有全部client25轮训练后state、真实梯度步数和索引。server有全部聚合及DA模型、原型、EMA、统计。

恢复推理：按对应锁校验文件SHA，使用r.model()、r.validate_state(model,ck['model_state'])、strictload，再model.eval()输入float32[N,50,8]；分类logits形状[N,4]，沿类别轴argmax(1)，标签顺序Ethanol/CO/Ethylene/Methane。完整输入需原采样/预处理合同，不能用任意71D回归特征直接输入分类器。

每个授权max_launches1，started/MATRIX_STARTED阻止重复；不要通过删除标记重跑。失败恢复须保留当前目录并单独批准新attempt，不沿用已消费授权。公开副本不含权重、原始数据或输入特征，无法独立复训，须本地归档。v5、旧Phase及Sstar来源仍只读。
