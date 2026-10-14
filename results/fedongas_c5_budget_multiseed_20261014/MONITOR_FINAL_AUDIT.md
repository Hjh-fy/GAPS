# 27 项实验最终监控核验

2026-10-15，Asia/Shanghai。研究 FEDONGAS-C5-BUDGET-MULTISEED-20261014。
冻结代码 `dc73318636335e0d2317613dff00ef73e6c798f1`；控制器结果提交 `36b6f9aa1f51b7d98f26173dcb06e017539c8b10`，GitHub指定分支ls-remote已确认一致。

## 核验结果

- 27/27独立正式端点PASS：SRC3，TO/FT/CLASS/FULL各6；seed42/43/44，预算5%80与20%320，冻结test1360。
- 27 checkpoint整文件SHA256及ordered state-content fingerprint PASS；实际round25/update2500、方法和seed校验PASS。
- 12项CLASS/FULL共300份逐轮审计PASS：每轮100steps；CLASS四原型/no true phase、stage MMD关闭；FULL最多12个class-phase原型；两组target CE关闭。
- protocol及原canonical/budget/test输入hash PASS；未更改预处理、窗口清单、目标划分、模型、超参数、loss权重或run矩阵。
- 统一评价目录36项SHA256 PASS；36720行缓存预测=27×1360，逐项Accuracy、macro-F1及4×4混淆矩阵独立核算与保存值一致（未重新执行model inference）。
- 24项目标结果、8组mean/sample SD、18项seed配对差值、6组配对汇总齐全；每组3seed，SD采用ddof=1。
- SEALED_TEST_OPEN记录all_27_complete=true；训练侧不以C5 test选checkpoint/阈值/超参数。未重新开启评价。
- 最终相关pytest 46/46 PASS；compileall PASS。两个protobuf依赖deprecation warning不影响运行。
- 完整仓库pytest历史结果仍为228 passed/1 failed/2 errors（maxfail=3停止）：旧protocol index hash不一致、两个tracked checkout fixture Windows路径过长。详见repository_pytest.log和启动ledger，不能声称全仓测试全绿。
- 控制器已退出，三机无本研究训练进程；不启动任何进一步实验。

## 失败与恢复

FULL20 seed43存在两个失败尝试：attempt01在round4附近Pi gRPC stream/socket关闭，attempt02因C2反向隧道18080地址占用在client训练前退出。
两个现场和对应日志/局部checkpoint完整保存在failed/，每个有sha256_index.json和RECOVERY_REPORT.md；FAILURES.jsonl保留原事件。
恢复时仅终止确认属于本任务的残留PID，三隧道连接测试PASS后从同一random-S43初始化完整重启25轮；未续训失败checkpoint，未重跑其余25项完成端点。
seed43成功端点已通过同样的固定round25核验。精确初始断连触发因素unknown，不虚构根因。

## 快照说明

原COMPACT_EVIDENCE_SHA256.json对应结果commit36b6f9a的提交前快照；其中RUN_PROGRESS.json为EVALUATING状态。
push成功后控制器将运行状态写为COMPLETE_PUSHED，因此这一动态状态文件的当前hash与旧快照不同；其余旧快照资产一致。
不改写该历史index或模型结果；新增FINAL_CLOSURE_SHA256.json封存当前完成状态、原快照及新增收尾报告，并提交当前RUN_PROGRESS.json。
result_commit继续引用原正式结果commit36b6f9a；后续收尾提交只补文档/状态/哈希，没有训练或结果数值改变。

## 科学边界

核验支持的是此已冻结window-level比较的结果完整性，不代表未见exposure/节点泛化或统计显著性。
三个seed反映训练随机性。FULL−CLASS是源历史phase组织加已定义stage损失对C5 late目标的整体比较，不是target跨phase泛化，也不能单独归因于prototype一个损失项。
TO/FT为target CE，CLASS/FULL为无target CE的语义/域适配；matched更新数不等于matched FLOPs或完全单因素消融。
本次不验证regression、QC或在线deployment，不据分类结果代替这些领域的科学结论。
