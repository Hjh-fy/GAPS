# FedOnGas研究贡献重评与方向收敛

本轮结论是收敛写作，正式停止当前新增联邦算法研发。现有工作可形成结构完整、证据清楚的硕士论文；尚无已充分确立的原创算法增益，不能把成熟机制实现或局部QP数值通过包装成新算法性能证明。IoT-J扩展需要进一步证明科学增量，当前不能认定已具备方法稿的充分证据。

## 审阅范围与访问

先建立PROTOCOL_AND_VERSION_MAP.csv，再做贡献重评。映射含Phase1–3.12及反馈/开发/复盘共22阶段，另加5条历史系统/会议协议。996条已归档模型文件引用及旧Pi审计72项哈希匹配，未加载模型、原始信号或输入数组。只有静态文献/代码/收据/已封存结果读取。本文转录原始固定指标，不重新推理、评分、bootstrap或训练。精确固定来源见SOURCE_EVIDENCE_INDEX.csv和各专题内链接。

## 三个决定性事实

第一，严格联邦pilot同S*且B1–B5目标预算相同，但计算并不相同。B3–B5额外源监督明显多于B1/B2。合法成对机制对照是B3/B4和B4/B5，本次目标准确率差为−1.548、−0.476个百分点，原区间包含0，反馈/原型增益未获支持。来源 [P3.5/RESULT_ANALYSIS.md](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/RESULT_ANALYSIS.md)、[P3.5/COMPUTE_AND_COMMUNICATION.csv](https://github.com/Hjh-fy/GAPS/blob/df2e61878a8902cb1fae95431013bbf3514798f7/docs/research_reviews/20261105_phase35/COMPUTE_AND_COMMUNICATION.csv)。详见FEDERATED_FAIRNESS_REASSESSMENT.md。

第二，残差胜F-SVR的5/6不能替代胜Source-only。源模型在C3/C5四个预算视图更强；残差同时优于这两个直接强基线仅C4A2。Oracle、支持裁剪及C4回顾性边界保留。来源 [P2B/README.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/README.md)。详见SOURCE_KNOWLEDGE_STRONG_BASELINES.md。

第三，反馈链路正确、A/B数值可行，并不保证类别保护。独立暴露开发出现第5轮标定稳定方向的漏报，现有有限步曲线无法区分代表性与非线性余项。后期预测不确定不等于开发CE近零，64条中55条实际变化超过原零带。来源 [GATE3/README.md](https://github.com/Hjh-fy/GAPS/blob/93c2f9fe445b0c8079cba8d52bae8f6279dd089e/docs/research_reviews/20261110_gate3/README.md)、[POSTMORTEM/HISTORICAL_CLARIFICATION.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/HISTORICAL_CLARIFICATION.md)、[POSTMORTEM/IDENTIFIABILITY.md](https://github.com/Hjh-fy/GAPS/blob/a41b93fc2db2c8de76310af8256e990443e0be76/docs/research_reviews/20261111_gate3_postmortem/IDENTIFIABILITY.md)。

## 系统贡献与论文范围

历史STM32/UART、上位机、Flower云端和模型管理代码及旧实机重放证据已存在，应作为工作量保留。最新PC预测路由合同通过，Pi最新联合模型可完整执行且独立性能满足原目标；严格跨平台logits仍BLOCKED，同版完整在线链路未验收。旧0.413ms确是ONNX+71D/72D Ridge联合pipeline，真实模型预算20%，本地ThreePass稿写5%存在来源范围冲突，必须如实记录。详见SYSTEM_VERSION_BOUNDARIES.md、ISCAS_EXTENSION_AND_REUSE_AUDIT.md；最新固定来源 [P3.10/README.md](https://github.com/Hjh-fy/GAPS/blob/ae2a614249443edaeae78d421851a3bdc57a92d6/docs/research_reviews/20261109_phase310/README.md)、[P3.11/README.md](https://github.com/Hjh-fy/GAPS/blob/33dff2d8b34e033d6bd69458d4401719ad1fc577/docs/research_reviews/20261109_phase311/README.md)、[P3.12/PERFORMANCE.md](https://github.com/Hjh-fy/GAPS/blob/b4f6f664d7d6b25877f257d08b347387961c433d/docs/research_reviews/20261109_phase312/PERFORMANCE.md)。

本地会议候选已取得并与v5字节一致，实际平台提交身份UNAVAILABLE。新论文需披露旧稿并突出严格协议、反证与数值/版本链增量。原型、梯度投影、层级冲突及Ridge统计量已有最近邻，详见NOVELTY_NEAREST_NEIGHBOR_MATRIX.csv。双维状态见CONTRIBUTION_EVIDENCE_MATRIX.csv。

## 最终六项回答

1. **不加训练可以写什么？** 六个受限证据包：独立暴露及少样本公平性分析；合法源银行和回退实现；同起点同目标预算的分类/源保持/成本及负结果；风险跨暴露失效和可识别性分析；冻结预测类别路由与PC恢复；实际Pi最新FP32成本及分版本验收。每包均有固定来源和明确限制，不以写作删负结果。

2. **方法创新有多少？** 当前充分确立的原创算法创新为0项。六个可写证据包按本报告合并口径计，3项严格实验/研究分析，3项系统或成熟方法实现。E4具有区别性的分析问题，但不计为原创算法。A/B是2个数值原型，不是2项已证实方法贡献；六包数量也不是学校创新条数或毕业条件判定。

3. **最有价值的未解问题？** 少量物理暴露上的稳定Calibration风险，为什么不能预测标定外有限更新损失？代表性与完整位移非线性各占多少？源知识被少量适配破坏亦有研究价值，但当前不能据此选一种门控。

4. **最小必要补充？** 毕业写作优先完成协议/数字/文献和实际ISCAS提交身份核对，采用本轮图表计划，无需新增训练。若要恢复算法研究，最小新信息为预先独立批准的四案例开发起点梯度诊断，之后才讨论独立日期/设备证据；不是自动增加种子或网络。最新在线系统则另需同版硬件输入和恢复闭环，不能用旧系统替代验收。本轮均未启动。

5. **完全不增实验如何定位？** 硕士论文优先采用THESIS_STORYLINES.md的A线，以受限方法实现、严格比较、负迁移/反馈失效分析及实际硬件工程证据组织。可开始完整写作，但是否满足学校毕业硬性规则需正式要求。IoT-J优先评估B线的风险可迁移性问题；零新实验时只能定位为有限范围的实证/系统边界研究候选，不主张新可靠反馈算法、跨设备普适改善或期刊录用就绪。

6. **是否停止新联邦算法研发？** 是。当前停止QP改造、可靠性门控拟合、复杂模块和追加联邦矩阵。保留已有实现，收敛毕业论文。只有关键机制可识别、数据角色与独立证据闭合，并得到新授权后才重新立项。该决定不否定历史已实现系统，也不把未知机制强行归因。
