# FedOnGas Phase2B：合法源模型与固定残差迁移最小验证

2026-11-04，依据Phase2A固定4955a3243ffe05a5f39087c4c8189c59ee13d53c。全部结果回顾性，不是首次盲测；C3三日期，C4/C5两日期替代，C4源Day7晚于calDay3、早于testDay9。逻辑客户端统计量实验，不代表网络隔离或隐私保护。

## 门禁与结果

Gate1 PASS：12个71D源头及12个pooled参考，alpha1000；训练身份、矩、scaler、系数、raw/clip逐窗对照和模型均保存。最大raw差4.547473508864641e-13ppm。来源合法C1Day4/C2Day1/7，共120暴露2520窗。

Gate2 PASS：原SVR71/base/Centered薄适配，240个故障场景、480次raw/clip检查与同一独立F-SVR头逐元素一致。rho0不访问源；非零残差源失效调用独立F头，不改变残差rho。原始cal支持最终仅clip一次，调整标签clip字段不用于浓度支持。源侧固定5值选择A2 rho=.75、A4 rho=1，两个任务均不劣于rho0。全部源候选评分和逐窗预测保留。

Gate3 PASS：24个新目标残差头锁定后评价30配置，共25200逐窗预测。F/N-SVR/KRR只读复用；SourceOnly4个全源头共享目标/预算，A2/A4是重复视图而不是独立训练。

主指标为Oracle_ALL、单窗口误差暴露等权NRMSE%，裁剪结果如下；raw、MAE、暴露/气体/浓度/seen/unseen完整保存。

| target | budget | F_SVR_rho0 | FixedRhoResidual71 | KRR_reference | N_SVR_reference | SourceOnly71 | residual_minus_F_points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | 2 | 21.0686 | 18.1665 | 19.8443 | 19.2257 | 12.2137 | -2.9021 |
| C3 | 4 | 19.9339 | 17.6483 | 19.6454 | 17.4610 | 12.2137 | -2.2857 |
| C4 | 2 | 17.6522 | 12.4106 | 17.7653 | 17.6287 | 19.1383 | -5.2416 |
| C4 | 4 | 17.6055 | 17.8624 | 17.4875 | 18.3148 | 19.1383 | 0.2569 |
| C5 | 2 | 20.6068 | 17.1497 | 20.1712 | 19.7669 | 14.7792 | -3.4571 |
| C5 | 4 | 17.6940 | 16.8023 | 18.3857 | 19.9686 | 14.7792 | -0.8917 |

## 研究判断

固定残差较F-SVR总体改善5/6，但C4 A4退步，逐气体/浓度仍有负迁移；不能宣布可靠、跨目标预算一致的迁移保证。Source-only在C3/C5明显强于标定后的残差，说明少量校准可能破坏已有浓度关系，不能只看残差赢F-SVR就断言适配必要。按预注册要求停止扩展当前固定共享rho方向，保留源知识有用的探索性证据，不调rho/阈值、不追加复杂门控。H-CONSISTENCY未实现。

源伪任务与Phase1/1.5选参复用同Day7源队列，cal/validation同日、文件互斥；继承元参数由整个Day7池选择，不能当作当Day1/4已有参数或独立确认。目标test历史已观察，当前640暴露中无已确认的新独立盲测集，不能据条件bootstrap推广跨日显著性。

## 制品与复现

run/SOURCE_INDEX及source/保存12源头、12pooled头、矩与SHA；SOURCE_MODELS_LOCKED、RHO_PARAMETERS_LOCKED、TARGET_MODELS_LOCKED给出三阶段锁定。SOURCE_TRAINING_IDENTITIES与SOURCE_PARITY_WINDOW_PREDICTIONS保留全部身份和7560个源对照。SOURCE_RHO_SCORES及SOURCE_RHO_WINDOW_PREDICTIONS保留80次选参拟合的所有候选。WINDOW_PREDICTIONS、OVERALL_METRICS、EXPOSURE_ERRORS、GAS_CONCENTRATION_ERRORS、DIFFICULT_CONCENTRATIONS、NEGATIVE_TRANSFER_SLICES与PAIRED_UNCERTAINTY是完整结果。暴露均值融合仅辅助，不能当单窗部署性能。

实际入口study.py，依次gate1→gate2→gate3，verify_report.py不拟合复核。使用现有Python3.12/numpy1.26/sklearn1.8/joblib1.4/scipy1.13/threadpoolctl，单线程CPU。输出拒绝覆盖；完整重现需复制到新research同级目录，保持依赖路径并新建run，不在本目录重跑。

首次Gate1在首个源/pooled头已保存后因审计dict重复gas_id停止；只修正元数据构造，原3文件SHA保留，直接加载继续，无重拟合。初始代码SHA及修正版SHA在PRE_RUN_CONTRACT/GATE1_RECOVERY，实际合计128fit，不隐瞒中断。

验证：60组指标独立公式核对；5040条新残差逐窗保存模型回放；15120条旧F/N/KRR预测逐元素复现；历史资产SHA不变。完整源/目标权重、特征与矩留本地，GitHub只发布精简报告、锁/索引摘要和数值证据。
