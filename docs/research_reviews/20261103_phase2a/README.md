# FedOnGas Phase2A：71D历史浓度知识迁移资产审计

日期2026-11-03，基于v5、Phase0.5、[Phase1固定772c0e9](https://github.com/Hjh-fy/GAPS/blob/772c0e9087435b9deb8ea4ac50b7c24814f6b345/docs/research_reviews/20261102_phase1/README.md)、[Phase1.5固定4988abb](https://github.com/Hjh-fy/GAPS/blob/4988abb3160c0b0b9230a48a817f2d7baed137d2/docs/research_reviews/20261102_phase15/README.md)。本阶段只读审计、合成单元测试及接口检查；**真实源训练0、目标迁移训练0、门控训练0**。

## 主要结论

- 历史SourceRidge71和104权重不满足新的Day7/71D合同，旧ResidualSVR及AdaptiveSourcePrior用target83/source104，不能直接搬模型或历史rho。可直接复用71D维度绑定、Ridge矩统计/求解、SVR71、base和Centered预测代数/排序器；没有重开发成熟算法。
- 合法新源输入已闭合：120暴露/2520窗，每gas630窗，来自C1 Day4和C2 Day1/7。旧源未来队列排除，目标文件/数值不参与源拟合方案或source-only参数来源；C4仍只作Day3cal/Day9test下的回顾性共同Day7源解释。
- 7项检查通过，24次合成拟合。4gas源充分统计量与pooled预测最大差1.14e-13ppm，符合既有1e-6ppm门槛；2预算×4gas的rho0探针raw/clip严格相同，最大差0。原83D fit仍拒绝71D，旧rho0遇source NaN仍失败，需要下一阶段薄兼容边界和不访问源的同F-SVR回退分支。
- 数学上源Ridge可充分统计量汇总；当前研究实际为集中主机读取源缓存的逻辑客户端模拟，不是已执行的纯联邦隐私训练。
- 640/640暴露均在已检查历史元数据中，没有确认未触及的独立目标确认集。全阶段仍是回顾性，不能把unused/reserved或窗口重切当新盲测。

## 阅读入口

1. [复用审计](REUSE_AUDIT.md)及[逐方法规格清单](REUSE_INVENTORY.csv)。
2. [71D合法输入与时间合同](SOURCE71_INPUT_AND_TIME_CONTRACT.md)及SOURCE71_MODEL_CONTRACT_PLAN.json（计划，真实权重未建立）。
3. [最小实验计划](EXPERIMENT_PLAN.md)、[30行矩阵](EXPERIMENT_MATRIX.csv)、[消融/停止条件](ABLATION_PLAN.md)。
4. [创新性与负迁移](INNOVATION_AND_NEGATIVE_TRANSFER.md)及[LITERATURE_NOVELTY_MATRIX.csv](LITERATURE_NOVELTY_MATRIX.csv)：成熟机制不包装成新贡献，仅提出1条端点浓度增量矛盾的拒绝迁移假设，未实现/训练。
5. [确认性数据审计](CONFIRMATORY_DATA_AUDIT.md)、[研究审计](EXPERIMENT_AUDIT.md)、[Phase2B门禁](PHASE2B_DECISION.md)、[CLAIMS_EVIDENCE.csv](CLAIMS_EVIDENCE.csv)。

## 下一阶段规模与门禁

主要比较固定F-SVR71=rho0，保留N-SVR/KRR参考；新候选仅Source-only与源侧固定rho残差。三个目标×两预算×五视图30行，所有行仅计划。最多4个全源头+8个伪源头及12个pooled参考、80个源侧rho残差、24个新目标残差，共128次真实数据拟合上限，rho0/源负迁移停止可减少。Source-only4头跨预算共享，F/N/KRR不重复训练。

**可以继续源侧构建与合同验收；现在不释放目标迁移训练。** 新真实SourceRidge71权重、真实数据一致性和71D生产兼容路径尚待验收；通过后再固定rho、锁目标模型并回顾性评价，不能从test选择rho或策略。新增门控/神经网络/部署不在范围。

## 可复现证据与保留限制

checks/保存7项完整输出、24次合成fit账本、rho0/维度/NaN等结果；evidence_v2/保存合法源120暴露清单、源cache/metadata/身份hash、28项v5来源SHA及函数起止行、历史104源JSON原index核验、640暴露历史覆盖。没有加载pickle/joblib作新推理，没有打开目标数值矩阵。读旧JSON只是来源/结构核对。

初次只读审计把10秒元数据浮点差要求成精确相等而停止；最大偏差1.42e-14秒。已依据冻结strict既有1e-6秒身份容差改审计断言，保留原evidence/与checks/INITIAL_AUDIT_DIAGNOSTIC，正式结果在evidence_v2/。没有改变数据/算法，也没有重复24次合成拟合。Phase1的125项和Phase1.5的185项封存负载均保持原SHA。

实际入口：audit_common.py（只读导入）、run_interface_checks.py（合成测试）、collect_audit.py（身份/静态核验）、write_deliverables.py和write_review.py（生成报告），后续verify_delivery.py做不拟合复核。所有输出拒绝覆盖；独立重现需复制脚本/PLAN到新的research同级目录并使用全新输出目录，不在本阶段目录重跑覆盖。

本地完整制品和总ARTIFACT_SHA256.csv封存；GitHub只发布精简审阅副本，原始信号、源71D输入、模型系数/scaler、v5及ZIP不上传。公开副本的路径别名/UTF8-LF使其hash不同，由PUBLICATION_MANIFEST区分来源与副本。
