# 历史浓度知识的强基线与主张边界

转录 [P2B/README.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/README.md) 的Oracle_ALL、单窗误差暴露等权、clipped NRMSE%，不是新的评分。四个全源头共享A2/A4视图；重复行不代表两个独立源模型或两次独立验证。C3采用三日期；C4/C5为两日期替代。C4共同Day7源模型晚于Day3标定，只能作回顾性解释。

|目标|每气体标定暴露|F-SVR71|固定rho残差|Source-only71|N-SVR71|KRR|
|---|---:|---:|---:|---:|---:|---:|
|C3|2|21.0686|18.1665|12.2137|19.2257|19.8443|
|C3|4|19.9339|17.6483|12.2137|17.4610|19.6454|
|C4|2|17.6522|12.4106|19.1383|17.6287|17.7653|
|C4|4|17.6055|17.8624|19.1383|18.3148|17.4875|
|C5|2|20.6068|17.1497|14.7792|19.7669|20.1712|
|C5|4|17.6940|16.8023|14.7792|19.9686|18.3857|

三个主张必须拆开。残差优于冻结F-SVR为5/6；残差优于Source-only为2/6，均在C4；同时优于这两个直接强基线只有C4 A2，即1/6。C3/C5的Source-only同时优于F-SVR和残差，不能隐藏。上述计数是固定六行的描述，不是跨任务独立显著性，也不是对所有N-SVR/KRR的全面优势。C4 A4残差总体退步，逐气体/浓度负迁移保留。

Source-only的raw函数没有使用目标浓度拟合，但clipped结果仍依赖已知Calibration物理ppm支持，且Oracle路由知道真实气体。不能把它写成无需任何标定或类别识别的完整系统。应并列raw、clipped、支持来源和路由条件，不据Test改选设备策略。源Ridge来自120源暴露2520窗的逻辑充分统计量，alpha1000；数值等价pooled参考不代表已实现网络隔离或隐私保护。来源 [P2B/REVIEW_SCOPE.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/REVIEW_SCOPE.md)、[P2B/RUNTIME_SUPPLEMENT.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/RUNTIME_SUPPLEMENT.md)。

rho=.75/A2与1/A4由两个合法源伪任务预先选择；任务与Phase1/1.5源侧选参队列复用，不是独立确认。rho=0调用同一F-SVR，非零残差源故障调用独立F回退头。240故障场景/480 raw-clip检查及数值一致性是可靠实现贡献，不能代替迁移泛化优势。来源 [P2B/README.md](https://github.com/Hjh-fy/GAPS/blob/675124ab6985cf32199e0b270e24b62d0e6f1621/docs/research_reviews/20261104_phase2b/README.md)。

Phase1的Ridge alpha1000从210/420源窗口迁至每气体4/8目标窗口，存在损失规模与正则强度不匹配。Phase1.5提供few-shot源选参与损失归一化对照后，不能再用原始弱Ridge证明SVR普遍优越。F/N-SVR、KRR是不同参数合同，不重新排名或删去不利对照。来源 [P1.5/FAIRNESS_AUDIT.md](https://github.com/Hjh-fy/GAPS/blob/4988abb3160c0b0b9230a48a817f2d7baed137d2/docs/research_reviews/20261102_phase15/FAIRNESS_AUDIT.md)、[P1.5/RESEARCH_DECISIONS.md](https://github.com/Hjh-fy/GAPS/blob/4988abb3160c0b0b9230a48a817f2d7baed137d2/docs/research_reviews/20261102_phase15/RESEARCH_DECISIONS.md)。

研究决定继续保留源知识有用和适配可能破坏源关系的证据，停止共享rho的扩张及事后策略选择。不因5/6优于Target-only而推出残差是最佳迁移方案。
