# 联邦分类公平性重评

以下值均转录固定结果，不重新推理、评分或抽样。对象只限合法 Day7 S*、C5 Day18 A4、seed42 的六组 pilot。所有组完整加载同一个 S*，state SHA256 `431fa41ecc0f86d948286f12374a3b12a00eee58c97ae358000ac7199e5cc673`，统一重置优化器。B0也继续执行25轮源训练，并非直接评价原S*。来源 [P3.4/SSTAR_AND_RESTORE.md](https://github.com/Hjh-fy/GAPS/blob/8ceeebcbd5e9e493c8dec5e726f030e5f9ec2fcc/docs/research_reviews/20261105_phase34/SSTAR_AND_RESTORE.md)、[P3.5/SCIENTIFIC_CONTRACT.md](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/SCIENTIFIC_CONTRACT.md)。

|组|机制与固定端点|源本地更新|目标更新 / 呈现|服务器源CE呈现|墙钟s|Fit收发字节合计|
|---|---|---:|---:|---:|---:|---:|
|B0|仅源FedAvg，aggregated25|3950|0 / 0|0|79.158|15569000|
|B1|目标独立微调，update2500|0|2500 / 40000|0|60.056|0|
|B2|目标作为普通第三客户端，aggregated25|3950|2500 / 40000|0|72.972|23353425|
|B3|源FedAvg后服务器SourceCE+TargetCE，适配不反馈，adapted25|3950|2500 / 40000|398800|495.794|16261586|
|B4|同B3，adapted反馈到下一轮，adapted25|3950|2500 / 40000|398800|501.665|16258352|
|B5|同B4，增加原型选择性参数加权，adapted25|3950|2500 / 40000|398800|504.370|16257967|

B2目标聚合权重32/2552，来自32个实际训练窗口，不按40000次重复呈现膨胀。B3–B5还各有40000行额外源forward、126000行源统计forward，不能遗漏。三组源本地CE均63000次，故各组总CE标签呈现分别为63000、40000、103000、501800、501800、501800。共同S*成本不重复计入。Fit字节为已保存protobuf消息载荷，不是完整TCP/gRPC流量。B1的0只表示无联邦Fit通信。墙钟受执行条件影响，不能用小差值证明算法速度优势。逐字段来源 [P3.5/COMPUTE_AND_COMMUNICATION.csv](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/COMPUTE_AND_COMMUNICATION.csv)、[P3.6/FIXED_RESULTS_AND_COST.md](https://github.com/Hjh-fy/GAPS/blob/4bf929fbbaa456e0a21599ad39fd01e8145a0406/docs/research_reviews/20261106_phase36/FIXED_RESULTS_AND_COST.md)。

同起点不等于同监督预算，同目标预算不等于同总计算。B1移除了源监督与联邦更新；B2将目标梯度放在客户端且权重很小；B3–B5在服务器联合源目标CE。B1/B3适合呈现目标适应—源保持—成本权衡，不能称等算力单因素。B0/B2显示既定普通样本权重下加入目标客户端的效果，不是对所有Target-as-Client实现的否定。

|组|C5 Accuracy %|Macro-F1|暴露等权错误率 %|C1 Accuracy %|C2 Accuracy %|
|---|---:|---:|---:|---:|---:|
|B0|34.1667|0.251914|65.8333|97.14|100.00|
|B1|83.8095|0.833213|16.1905|66.43|75.00|
|B2|32.2619|0.235264|67.7381|92.50|100.00|
|B3|81.9048|0.815875|18.0952|94.52|99.76|
|B4|80.3571|0.783889|19.6429|97.50|99.76|
|B5|79.8810|0.777418|20.1190|91.67|98.93|

S*源保持参考为C1 82.26%、C2 99.88%，与B0的继续源训练端点不同。来源 [P3.5/RESULT_ANALYSIS.md](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/RESULT_ANALYSIS.md)。只有B3/B4的反馈开关、B4/B5的原型权重机制，经过共同配置、目标索引、模型收发与统计上传核实，可作本协议下的成对机制比较。B3与B4都完成相同服务器适配；B3下一轮广播aggregated，B4广播adapted。B5在warmup后实际激活22轮，并非未启用的空对照。来源 [P3.6/FEEDBACK_PROTOTYPE_AUDIT.md](https://github.com/Hjh-fy/GAPS/blob/4bf929fbbaa456e0a21599ad39fd01e8145a0406/docs/research_reviews/20261106_phase36/FEEDBACK_PROTOTYPE_AUDIT.md)。

B4−B3目标Accuracy为−1.548个百分点，B5−B4为−0.476个百分点。原暴露等权错误率差的95%配对区间分别为[−4.289,+6.905]和[−0.476,+1.310]个百分点，均包含0。不能宣称确定改善，也不能据此证明普遍损害。B3→B4的CO Recall从90.48%降至37.62%，其他类别部分补偿，不将其归因为已确认的梯度冲突或物理老化。来源 [P3.5/RESULT_ANALYSIS.md](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/RESULT_ANALYSIS.md)、[P3.6/CO_DIAGNOSIS.md](https://github.com/Hjh-fy/GAPS/blob/4bf929fbbaa456e0a21599ad39fd01e8145a0406/docs/research_reviews/20261106_phase36/CO_DIAGNOSIS.md)。

结论限单设备、单预算、单种子、历史已观察Test。40次目标物理暴露不是840个独立样本，同一训练轨迹的多个轮次也不是重复训练。旧ISCAS的Class-MMD和窗口划分与本协议不同，旧正向反馈不能与新负向结果合并为单因素因果结论。来源 [P3.6/OLD_NEW_PROTOCOL_COMPARISON.md](https://github.com/Hjh-fy/GAPS/blob/4bf929fbbaa456e0a21599ad39fd01e8145a0406/docs/research_reviews/20261106_phase36/OLD_NEW_PROTOCOL_COMPARISON.md)。
