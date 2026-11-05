# 单次执行及证据

正式SOURCE启动次数1、完整训练完成次数0、原生Adam.step次数0。ONCE_STARTED记录唯一启动与完全原argv；AUTHORIZATION_RECEIPT记录用户SOURCE专用批准；EXECUTION_FAILED记录三个exit1、现场栈与12.9897705秒墙钟。没有修改已锁spec/runtime/contract或任何v5/Phase0–3.2既有payload；新授权和runs是本次明确允许的新文件。

三个进程均在observer_ready后运行，Torch线程1。只读观察器包装原Adam.step/初始化、RandomSampler/DataLoader和原Client.train_one_round，不改变参数、损失、随机种子、排序或输出。observer启动异常会立即exit78，避免静默失去记录。正式server/client1/client2各报告0优化步、0完整round，观察到的12次数字文件打开均为合法C1/C2的6个source文件（hash读取+np.load）；总36次源数值打开，目标数值访问0。两个guard selftest的故意目标打开被拒绝、没有成功读取；与正式进程零拒绝/零目标访问分开记录。

源输入120暴露/2520窗合法性保持：C1Day4 840、C2Day1/7 1680，50×8/4类/phase0，原始预处理与有效mask由canonical SHA固定。训练进程只把source包作为数据输入，Python层数值白名单拒绝其它npy/npz，并非声称实现容器或隐私隔离。来源元数据/清单用于身份审计，没有目标数值输入或监督更新。

runs/SOURCE恰好3个effective_config.json，与最终SOURCE注册effective SHA一致；没有训练receipt、server_round_001、其余25轮模型或SSTAR文件。SOURCE_ROUND_STATUS按50个planned client-round条目记录实际0，不伪造63k呈现或25轮链。状态表中840/1680与1/3、2/3都是已注册数据量/计划权重，不是实际聚合参与证据。

成本：约13秒为失败启动总墙钟，非训练耗时。逐进程峰值RSS约379MB见PROCESS_COSTS；异常分支未持久化coordinator的并发峰值，不能将单进程峰值相加冒充实测并发峰值。没有fit模型参数收发，wire握手/错误流量未计量；不能声称已测出模型训练通信成本。没有S*，训练后strict load/合法源输入恢复推理均NOT_RUN，不用随机模型或旧权重补齐。

独立read-only审阅确认configure keyword失败、实际aggregate位置调用、零step范围和冻结代码SHA。诊断期间无额外模型训练，也未读取真实目标Test。所有权重/数值数据继续留本地；本轮没有新训练权重可公开。
