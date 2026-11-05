# FedOnGas Phase3.5 C5 A4 seed42 六组pilot

依据Phase3.4固定8ceeebcbd5e9e493c8dec5e726f030e5f9ec2fcc；共同合法Sstar state 431fa41ecc0f86d948286f12374a3b12a00eee58c97ae358000ac7199e5cc673。仅授权B0–B5，各一次、固定25轮；B1–B5使用相同32目标标定窗及预锁索引，B0训练不使用目标数据。数值代码来自未修改Phase33R runtime/contract及v5。

当前训练终点锁定6/6；离线评价完成。实际状态见TRAINING_PASS_BLOCKED.csv，版本见MODEL_VERSION_INDEX.csv，预算/通信见COMPUTE_AND_COMMUNICATION.csv。

|组|目标Accuracy|Macro-F1|暴露等权错误率|
|---|---:|---:|---:|
|B0|0.341667|0.251914|0.658333|
|B1|0.838095|0.833213|0.161905|
|B2|0.322619|0.235264|0.677381|
|B3|0.819048|0.815875|0.180952|
|B4|0.803571|0.783889|0.196429|
|B5|0.798810|0.777418|0.201190|


B4−B3目标Accuracy变化-1.548个百分点，B5−B4变化-0.476个百分点。目标配对区间均含0；这些是单seed、单目标的观察差异，不是跨设备或独立确认性证据。

反馈未取得目标准确率增益，应保留负结果，停止本配置的优势主张。

原型加权未取得目标准确率增益，停止该配置的增益主张，不追加搜索。

阅读：SCIENTIFIC_CONTRACT.md、TRAINING_EVIDENCE.md、RESULT_ANALYSIS.md、NEXT_DECISION.md、RECOVERY.md、FINAL_VERIFICATION.json。本地保留所有25轮模型、client快照、原型/encoder版本、索引与逐窗预测；GitHub只发布精简脱敏审阅副本。

研究边界：回顾性探索，seed42/C5A4，不是首次盲测、跨设备普遍有效或真实在线验证。无额外seed/72分支/预测类别回归/Pi部署；无测试选轮、事后超参或策略选择。
