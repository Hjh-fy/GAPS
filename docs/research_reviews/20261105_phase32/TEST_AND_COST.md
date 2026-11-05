# 测试证据、差异与计算成本

五条目标路径实际原optimizer循环全部2500/40000，索引digest均183ff95ee79712c78a5c9c63ded9748f1dac832a1e7b27b165623fd304d55baf。B1/B2第一次完成的收据保留并复用；第一次在进入DA前因原序列化使用"0,0"而非"(0,0)"停止。修正新guard后，只运行尚未完成的DA，不重复完整B1/B2。失败日志未删除。

25轮DA数值使用真实原ServerDomainAdaptation/损失/optimizer，Tiny64合成输入，无数值替身。为隔离服务器机制，该长循环复用了第一次真实合成source fit生成的上传fixture，因此不把它单独说成真实逐轮客户端训练。integration又各做B3/B4/B5两轮真实合成source Client.fit+原DA+下一轮fit：校验返回/接收/encoder/prototype版本和stale拒绝，不用RPC训练替身。真实Flower网络未启动。

B5从第4轮实际selective_active且非退化相似度给出不同权重，B4不触发；受控相同upload下三组EMA一致。独立两轮实际fit允许模型反馈后统计自然变化。纯CE客户端统计调用原raw64D均值/方差及normalized global feature，都是训练后无梯度；不新增MMD/对抗/原型损失。

Source完整2520×50×8float32仅4.032MB，实际无梯度loader遍历840/1680/32窗成功。每DA组25轮实测27500源批、438800源forward行；其中SourceCE398800、附加source feature forward40000。源每轮本地CE总2520/158步，25轮63000/3950。另统计上传执行两次source pass，每DA组额外126000无梯度forward，必须单列。成本详COST_MATRIX；B1本地、B2第三客户端、B3–B5服务器混合CE并非完全等算力单因素对照。

模型完整state144692B；两源25轮上下行张量约14.4692MB，三客户端21.7038MB。新增记录FitIns/FitRes实际protobuf序列化字节，integration有合成实测；不是TCP/TLS wire字节、不是实际TCN训练时间。CPU无CUDA，源训练墙钟/峰值内存仍未测，不能用Tiny时间兑现。

测试：11项本阶段unit全部通过，旧Flower split兼容2项通过；红/绿日志均保存。没有运行可能读取历史真实Test/训练的全仓库实验套件，不声称全仓库测试通过。故意错误预算、shape/class/phase/nonfinite、未知模型、自造random/S*链、原dtype隐式转换、错误类计数/原型内容、广播版本均fail closed。临时目录第一次受Windows沙箱拒绝，之后全部fixture放当前worktree，不触碰旧结果。
