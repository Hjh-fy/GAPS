# FedOnGas Phase3.4：合法S*源分类预训练完成

**SOURCE attempt2成功，合法S*已生成并验收；25轮训练与恢复证据完整。六组pilot初始化/配置复检通过，可以申请下一轮独立授权，本轮没有启动pilot。** 依据Phase3.3R固定b593c2f4b7498049d928d98981a448489b27bd3d；唯一注册P33R-SOURCE-DAY7-S42-ATTEMPT2，科学spec730e5d3e95c7176e605badd32eb2b309c90046f65bcd059987c73a42498a87a7。

|验收项|结果|
|---|---|
|来源|仅C1Day4、C2Day1/7，120暴露/2520窗；原随机初始state未重生成|
|真实训练|25轮/50client-round；3950更新、63000呈现，逐行索引每轮恰好覆盖各自源集一次|
|模型版本|25servercheckpoint、50clientstate快照、接收/上传/聚合/返回SHA链及源/代码/环境合同全部通过|
|恢复|固定12个合法源窗口，原round25→S*封装→再次保存恢复的输出逐元素完全一致|
|六组门禁|同一合法S* strict load，initialSHA一致；旧目标2500/40000及反馈/原型语义不变；未运行训练|
|隔离/范围|目标train/Development/Test数值读取0，未计算目标准确率；attempt1现场与旧冻结制品未改|

训练总墙钟72.990秒，三进程RSS求和采样峰值1,292,713,984B；FitIns/FitRes protobuf合计15,569,000B，模型tensor载荷14,469,200B。详[成本与版本链](TRAINING_EVIDENCE.md)，这些是本地多进程与审计开销下的观测，不是Pi/真实跨设备网络性能。

阅读：[训练/成本证据](TRAINING_EVIDENCE.md) → [S*来源及恢复](SSTAR_AND_RESTORE.md) → [六组门禁](PILOT_GATE.md) → [恢复路径](RECOVERY.md) → [继续/停止](NEXT_DECISION.md)。SSTAR_ACCEPTANCE_LOCK是本轮完整验收，bank_metadata/SSTAR_LOCKED保留原生锁；完整权重及输入仅本地。
