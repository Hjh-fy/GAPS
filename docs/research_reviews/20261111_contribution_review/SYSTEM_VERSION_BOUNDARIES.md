# 系统实现与验证版本边界

本轮静态读取实现、历史收据与摘要，并核对哈希，没有连接Pi、启动进程或重放输入。`HISTORICAL_SOURCE_SUMMARY.csv`提供26项历史来源的路径标识、SHA和可用性；旧Pi审计清单72项全部匹配。原文件绝对路径仅在本地`SOURCE_EVIDENCE_INDEX.csv`。本轮取得的文件可用不等于重新实测，历史设备证据均按原验收范围引用。

|版本/链路|已经实现或已有实测|不得扩大的结论|当前论文写法|
|---|---|---|---|
|STM32/UART/旧上位机|serial_worker.py、frame_parser_v20.py、edge_ui_app.py具备串口线程、43字节协议解析、曲线/日志；edge_ai_worker/runtime具备包加载和推理接口|审计范围内未取得STM32固件.c/.h，不能声称已审计板端固件；有串口代码不等于本版真实采样到GUI已验收|既有采集与上位机工程模块；固件来源UNAVAILABLE|
|July Flower云端/参数管理|gaps_flower客户端/服务器、部署合同和ECS/Pi/PC研究日志存在；初期记录也有双PC模拟客户端|不能把模拟客户端记作Pi实机，不能把旧25轮5epoch/batch32当新1epoch/batch16|区分代码实现、日志报告与实机收据|
|July runtime v5 UI|FinalB5→FederatedH1→105D TargetRidge；320窗PC/Pi及Qt worker/card重放，hash破坏拒绝记录|旧105D不是当前71D/SVR；原始未标记STM32输入有replay-only拒绝门禁|真实设备软件链重放，非已证明在线物理气体闭环|
|独立lab3gas UI|100×6、1Hz三气体实验，PC/Pi/cloud相同包；359/360分类正确，PC/Pi预测一致；真实Wayland Qt路径有历史验收|仅分类，ppm为空；rawADC到电阻/稳定特征映射未闭合；不是50×8四气体版本|已有独立实验与设备UI重放证据，保留其范围|
|旧ISCAS Pi benchmark|ONNX分类+71D/sourceRidge+72D/targetRidge，C5 J0 B20；pipeline P95=0.4127459ms|不称“只测分类”；也不当新SVR、传感器到GUI或5%模型成本；71D选择历史Test使用已披露|旧固定软件栈联合离线延迟，4线程，20%标定|
|Phase3.7/3.9/3.10 PC|最新六分类器+71D+四F-SVR71预测路由；Calibration golden、异常恢复、独立进程与5040条固定单窗评价完成|PC通过不撤销旧batch128失败；条件回归子集不同，不能跨气体混合ppm|PC-F32-CPU-SINGLE-v1范围内的联合推理证据|
|Phase3.11 Pi|新模型实机5040条完成，argmax和路由未变，特征一致、ppm差很小|30条logits超差，严格数值合同仍BLOCKED，不能用argmax一致改判|硬件执行成功，与数值严格验收未通过分开|
|Phase3.12 Pi性能|Calibration/合成独立工程表征，联合P99=3.936–3.992ms，最大RSS317.27MiB，最高55.10℃|不撤销3.11；不是新Day22性能或完整在线链路；不能与旧0.413直接算加速比|真实硬件最新FP32联合离线成本|

历史精确文件见`HISTORICAL_SOURCE_SUMMARY.csv`。旧lab包SHA `88a4a4d75c6bd2f1a917a1e771712a0568e9c284bee599f4f83ab26257fb0877`，旧0.413ms benchmark manifest SHA `7e0ccaa4ae6e2a025796b9682c6547e0eefc5dd391e3c9861ba3e48fa0e5d102`，以本报告固定版本封存其来源摘要。历史Qt/云端收据文件可取得，但本轮没有重新启动或独立验证远程设备。

最新PC、Pi来源分别为 [P3.10/RESULTS.md](https://github.com/Hjh-fy/GAPS/blob/ae2a614249443edaeae78d421851a3bdc57a92d6/docs/research_reviews/20261109_phase310/RESULTS.md)、[P3.11/NUMERICAL_AND_RECOVERY.md](https://github.com/Hjh-fy/GAPS/blob/33dff2d8b34e033d6bd69458d4401719ad1fc577/docs/research_reviews/20261109_phase311/NUMERICAL_AND_RECOVERY.md)、[P3.12/PERFORMANCE.md](https://github.com/Hjh-fy/GAPS/blob/b4f6f664d7d6b25877f257d08b347387961c433d/docs/research_reviews/20261109_phase312/PERFORMANCE.md)。旧batch128验收 [P3.8/NUMERIC_CONSISTENCY.md](https://github.com/Hjh-fy/GAPS/blob/4dbf732df9ed693c88cb6486099930a428acc90b/docs/research_reviews/20261106_phase38/NUMERIC_CONSISTENCY.md) 继续BLOCKED；PC新合同来源 [P3.9/README.md](https://github.com/Hjh-fy/GAPS/blob/cb850dadfbef986702e3155372fef225a818f2b1/docs/research_reviews/20261107_phase39/README.md)。不得把不同合同合并为单个系统PASS。

最新FP32版本还缺同版真实传感器输入标定映射、窗口时序/QC、真实通信与GUI、远程候选包下发与健康快照保持、长期在线恢复的完整闭环证据。旧edge_ai_worker的加载异常分支可能清空runtime；不能把新恢复测试能力反向赋给全部旧版本。现有模块应复用，不能重新表述成全未开发。若需要上线，应另立同版验收合同，本轮不修改实现。
