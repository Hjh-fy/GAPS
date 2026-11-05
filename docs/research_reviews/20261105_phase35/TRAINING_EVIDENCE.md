# 训练与成本证据

成功锁定6/6，各组25轮/块全state在runs及observer；原生phase32_receipt逐轮记录收到/上传/聚合/返回，encoder+prototype SHA绑定。audit各组ROUND_LEDGER、CHECKPOINTS、FEEDBACK_AND_WEIGHTS给出实际计数与独立聚合参考，容差4eps32×max(1,absreference)。最终state有限、strict load与两次加载固定4合法Calibration窗输出精确一致，不作精度选择。

source每轮53+105步、2520呈现，25轮3950步63000呈现。B1/B2target每轮100步1600呈现，总2500/40000；B2样本聚合32/2552，与重复呈现无关。B3–B5target同2500/40000，另有源客户端3950步、源服务器398800标签呈现、40000额外源forward与126000统计no_grad forward；serverSourceCE共25000批次，partialbatch如实计数。

通信以真实FitIns/FitRes protobuf.ByteSize记录（包含参数、config、stats），同时原始receipt记录statepayload；这不等于包含gRPC/TCP/TLS帧的物理链路字节。初始get_parameters/握手/控制消息不并入该Fit账本；不隐藏为完整链路总量。B1本地无联邦通信。墙钟含进程启动训练，独立audit用时未并入训练完成账本；RSS按.2s采样、子进程观察器.1s，非精确连续峰值。

B1/B2/B3计算位置及额外源更新不同，不能主张等算力单因素。只有B3/B4反馈、B4/B5选择性权重满足预锁机制差异。phase=0为中性类别统计键，不是测得的物理响应阶段。

所有异常均保留，缺支不放Test、不选剩余赢家；完整状态和资源成本见CSV/原日志。
