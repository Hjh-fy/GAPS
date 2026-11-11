# ISCAS提交版本、扩展增量与复用边界

本轮实际读取v5封存和原工作区同名`FedOnGas_ISCAS2027_Final_ThreePass_Checked.docx`，两者SHA均为`239cb846238c1acc96acb87b69025161f8f254d636b2519504dbdc5aeb62f91a`。只通过ZIP/XML提取88个段落/表块，不改正文。`MANUSCRIPT_SOURCE_AUDIT.json`保存来源和提取SHA，PRIVATE_ISCAS_0_TEXT.json的block编号从0开始。用户提供的“已提交”视为研究状态信息，但平台提交回执及实际上传字节身份为UNAVAILABLE。因此下表审阅的是已找到的本地冻结候选，不冒称已核验平台最终文件。

|内容|ISCAS本地冻结候选的已有内容|后续真实增量/限制|
|---|---|---|
|方法|blocks11–13、33–37已有目标感知聚合、服务器适配反馈、71D源Ridge+72D目标Ridge；充分统计量已引既有文献|不能把这些作为扩展论文全新贡献；71D源预测仿射追加没有新增独立传感信息|
|分类|blocks41–50为外层窗口划分、5%标定、Class-MMD开启；旧C3/C4/C5 Acc99.14/97.43/98.53%，旧反馈差4.63pp|新合法Day7 S*、独立暴露A4、纯CE、同目标监督预算，源/目标/成本及负结果是新增证据；不能跨协议排名或将正负反转归为单一因素|
|回归|blocks43、59为旧窗口级TargetRidge与source-aug Ridge；相对NRMSE降低9.08/4.65/3.55%，不是百分点|新71D强SVR/KRR、正则迁移、公平源银行/残差/回退和Source-only强基线为新增；不能继续沿用旧普遍适配收益叙事|
|持续接入|旧论文已有75轮连续接入|新阶段没有同协议完整顺序接入证据；不得把旧75轮认作最新严格版本验证|
|系统|blocks63–65已有独立三气体实验及Pi重放/延迟|最新预测路由、单窗数值合同、FP32 Pi逐层/资源与失败边界是增量；“首次实现Pi”不成立|
|反馈风险|旧稿无四快照暴露级风险迁移验证|Gate0–3与静态可识别性分析是真实新增，但没有训练收益或独立确认性数据，不能包装成已验证新算法|

## 确认的一处范围冲突

ThreePass候选block65把0.413ms对应代表模型写作5%标定；旧routing-v5候选同段写20%。实际原始benchmark `package/MANIFEST.json`为`ISCAS27-EDGE-J0-C5-B20-71x72-S42`，`budget_pct=20`、320cal窗、batch32、Class-MMD。`LATENCY_SUMMARY.csv`明确pipeline P95=0.4127459ms，分类P95=0.2123008ms。72项原审计哈希全部匹配。故数值本身属于旧完整ONNX+71D/72D Ridge离线pipeline，问题是标定预算关联错误，不能静默改成新SVR性能，也不能误称只测分类。历史Test参与71D选择的限制在旧FINAL_ACCEPTANCE.json中披露，边缘一致性不证明无偏泛化。

本轮记录冲突并保留所有原稿，不修改已提交论文。后续写作应以真实manifest锁定20%范围；正式平台文件仍需作者核对回执。来源精确SHA见HISTORICAL_SOURCE_SUMMARY.csv及MANUSCRIPT_SOURCE_AUDIT.json。

## 扩展论文政策与文本复用

IoT-J官方作者指南允许具有显著扩展的相关会议版本；其链接的扩展说明要求实质新信息、披露并引用旧论文、标明新内容及提交旧文供审阅。该说明没有给出适用于本案的固定“30%/35%”门槛，不能套用其他期刊标准。见[官方作者指南](https://ieee-iotj.org/guidelines-for-authors/)及[官方扩展说明](https://iot-journal.weebly.com/uploads/1/8/8/0/18809834/guidelines_for_expanding_conference_papers_iot-j.pdf)。另据[IEEE投稿与同行评审政策](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)，投稿时应披露关联稿件和解释差异。当前会议接收/出版状态未知，应在真实投稿前核对，不能假定会议稿已出版或期刊允许并行。

建议建立逐节增量表。旧架构/预处理/数据背景可简要引用并明确来源，避免整段复制作为新贡献；旧表图需标明来自/改编自会议稿并核查出版许可。新增正文应围绕严格协议、反证和版本数值链重新组织，不能仅换模型名字复写旧摘要/结论。本文未进行两稿逐字相似率检测，因为尚无扩展稿，不报告虚构复用百分比。

当前增量足以支持有实质内容的论文重构与硕士论文章节，但“有新增实验”不自动满足IoT-J的新颖性和系统价值要求。最近邻重叠、单设备单种子和最新在线闭环未验收仍是主要风险。新旧协议来源 [P3.6/OLD_NEW_PROTOCOL_COMPARISON.md](https://github.com/Hjh-fy/GAPS/blob/4bf929fbbaa456e0a21599ad39fd01e8145a0406/docs/research_reviews/20261106_phase36/OLD_NEW_PROTOCOL_COMPARISON.md)；严格新结果 [P2B/README.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/README.md)、[P3.10/README.md](https://github.com/Hjh-fy/GAPS/blob/ae2a614249443edaeae78d421851a3bdc57a92d6/docs/research_reviews/20261109_phase310/README.md)、[P3.12/README.md](https://github.com/Hjh-fy/GAPS/blob/b4f6f664d7d6b25877f257d08b347387961c433d/docs/research_reviews/20261109_phase312/README.md)。
