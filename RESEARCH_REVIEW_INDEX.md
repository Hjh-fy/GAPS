# GAPS 研究审阅入口

审阅分支`codex/research-reviews`，各阶段另附固定commit。

最新：[反馈冲突与 A/B 门禁](docs/research_reviews/20261110_feedback_gates/README.md)

冻结链路与 QP 数值验证通过。有限步结果不支持直接进入硬约束训练，研究决定 REDESIGN。

上阶段：[Phase3.12 Pi 5 数值诊断与性能表征](docs/research_reviews/20261109_phase312/README.md)

标定与合成输入定位到首块 TCN pointwise 的首处字节差异。实机联合 P99 3.936–3.992 ms。Phase3.11 严格数值门禁仍 BLOCKED。

上阶段：[Phase3.11 Pi 5 离线联合推理](docs/research_reviews/20261109_phase311/README.md) · [数值与恢复](docs/research_reviews/20261109_phase311/NUMERICAL_AND_RECOVERY.md)

Pi 5 完成六视图 5,040 条推理，类别与选头一致。30 条 logits 超容差，完整数值门禁 BLOCKED，性能未运行。

上阶段：[Phase3.10 单窗联合评价](docs/research_reviews/20261109_phase310/README.md) · [实机决定](docs/research_reviews/20261109_phase310/DECISION.md)

六视图 5040 条预测完成封存与评分。旧 Phase3.8 BLOCKED 单独保留。

上阶段：[Phase3.9 PC 单窗验收](docs/research_reviews/20261107_phase39/README.md) · [决定矩阵](docs/research_reviews/20261107_phase39/DECISION.md)

新 PC 单窗合同通过双进程验收，历史 Phase3.8 继续 BLOCKED。Day22 和 Pi 仍需各自授权。

上阶段：[Phase3.8R数值差异定位](docs/research_reviews/20261106_phase38r/README.md) · [batch诊断](docs/research_reviews/20261106_phase38r/BATCH_DIAGNOSIS.md) · [后端/逐层](docs/research_reviews/20261106_phase38r/BACKEND_AND_LAYER_EVIDENCE.md) · [PC提案](docs/research_reviews/20261106_phase38r/PC_CONTRACT_PROPOSAL.md) · [继续/停止](docs/research_reviews/20261106_phase38r/NEXT_DECISION.md)。

Calibration/合成主3020+上下文304次；原容差均通过，byte差异保留。定位TCN pointwise跨batch与Attention no_grad/inference路径差，未证明历史5次超容差完整根因。旧P38仍BLOCKED；新PC合同仅提案，Day22/Pi未授权。

上阶段：[Phase3.8联合离线评价：BLOCKED](docs/research_reviews/20261106_phase38/README.md) · [执行审计](docs/research_reviews/20261106_phase38/EXECUTION_AUDIT.md) · [数值一致性](docs/research_reviews/20261106_phase38/NUMERIC_CONSISTENCY.md) · [继续/停止](docs/research_reviews/20261106_phase38/NEXT_DECISION.md)。

唯一执行已消费，1376/5040条；B2完成840，其他五组logits超锁定容差停止，已处理argmax均与P35一致。全矩阵未锁，不评分、不挑赢家、不重试。训练0、refit0、Pi0，先审计数值来源。

上阶段：[Phase3.7严格数值桥接](docs/research_reviews/20261106_phase37/README.md) · [接口](docs/research_reviews/20261106_phase37/API_CONTRACT.md) · [数值验收](docs/research_reviews/20261106_phase37/NUMERIC_ACCEPTANCE.md) · [未来六视图合同](docs/research_reviews/20261106_phase37/FUTURE_OFFLINE_EVALUATION_CONTRACT.md) · [下一步](docs/research_reviews/20261106_phase37/NEXT_DECISION.md)。

16Day18暴露/32窗Calibration桥接PASS，信号/71D逐位一致，四SVR与原实现差0，六分类argmax一致，35项异常测试通过。训练0、Day22评价0、Pi0；下一轮必须独立授权。

上阶段：[Phase3.6机制诊断与贡献收敛](docs/research_reviews/20261106_phase36/README.md) · [CO诊断](docs/research_reviews/20261106_phase36/CO_DIAGNOSIS.md) · [机制审计](docs/research_reviews/20261106_phase36/FEEDBACK_PROTOTYPE_AUDIT.md) · [贡献矩阵](docs/research_reviews/20261106_phase36/CONTRIBUTION_REVISION.md) · [路由合同](docs/research_reviews/20261106_phase36/ROUTING_COMPATIBILITY.md) · [继续/停止](docs/research_reviews/20261106_phase36/NEXT_DECISION.md)。

本轮训练/推理0；复用最终预测定位CO−111窗与其他类+98窗的权衡。反馈/原型优势未获支持；身份/71D静态合同通过，数值路由未执行。所有新候选须另行授权。

上阶段：[Phase3.5 C5A4六组pilot](docs/research_reviews/20261105_phase35/README.md) · [实际预算/成本](docs/research_reviews/20261105_phase35/TRAINING_EVIDENCE.md) · [结果与边界](docs/research_reviews/20261105_phase35/RESULT_ANALYSIS.md) · [继续/停止](docs/research_reviews/20261105_phase35/NEXT_DECISION.md)。

训练锁定6/6；评价完成=True。固定Sstar、seed42/C5A4、25轮；不扩实验、不调参。完整权重/输入仅本地。


上阶段：[Phase3.4合法Sstar完成](docs/research_reviews/20261105_phase34/README.md) · [训练/成本](docs/research_reviews/20261105_phase34/TRAINING_EVIDENCE.md) · [Sstar/恢复](docs/research_reviews/20261105_phase34/SSTAR_AND_RESTORE.md) · [六组门禁](docs/research_reviews/20261105_phase34/PILOT_GATE.md) · [继续/停止](docs/research_reviews/20261105_phase34/NEXT_DECISION.md)。

SOURCE attempt2真实25轮完成且验收，3950更新/63000呈现，合法Day7源身份、25server+50client状态及完整版本链通过；固定12合法源窗恢复输出完全一致。合法共同Sstar已生成。六组初始化/配置复检PASS，可以申请下一轮独立授权，未启动pilot；目标数据/Day22Test读取0。新授权仅SOURCE，旧attempt1失败保留。

历史：[Phase3.3R固定b593c2f](https://github.com/Hjh-fy/GAPS/blob/b593c2f4b7498049d928d98981a448489b27bd3d/docs/research_reviews/20261105_phase33r/README.md) · [Phase3.3失败](docs/research_reviews/20261105_phase33/README.md) · [Phase3.2](docs/research_reviews/20261105_phase32/README.md) · [Phase3.1](docs/research_reviews/20261105_phase31/README.md) · [Phase3](docs/research_reviews/20261104_phase3/README.md) · [Phase2B](docs/research_reviews/20261104_phase2b/README.md) · [Phase2A](docs/research_reviews/20261103_phase2a/README.md) · [Phase1.5](docs/research_reviews/20261102_phase15/README.md) · [Phase1](docs/research_reviews/20261102_phase1/README.md) · [Phase0.5](docs/research_reviews/20261102_phase05/README.md)。

[发布约定](docs/research_reviews/PUBLISHING_POLICY.md)：原始数据/模型权重/v5/ZIP仅本地；不自动合并main。
