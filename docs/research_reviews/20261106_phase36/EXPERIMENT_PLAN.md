# 未来实验候选合同（未授权、不可直接运行）

H36-R：同一合法Sstar和数据/样本预算下，B3/B4/B5的目标/源差异是否对更新随机性敏感？若论文需要稳定性证据，可限seed43/44六组补充；不是为挽救负结果。固定源预训练seed42的Sstar，仅改pilot更新seed，保持同一目标消费索引表及每seed各组一致的Source shuffle/dropout初始化规则。不能叫端到端全流程seed43/44，因为未重训Sstar。

资源参考本次每DA组约8.3分钟，六组约50分钟CPU、38700总优化步、15000目标承载更新/240000次目标呈现。只是同宿主实测外推，I/O/审核另计；保存全部25轮与同样版本链，失败不重跑。

明确阻塞：Phase33R spec/config、PhaseClient.seed、source_loader(42+round)、共同target schedule等写死42；现有entry没有真实可用seed43/44开关。必须先另行批准薄入口/合同扩展及无Test预检，并锁新代码/配置/采样合同；不能伪造CLI或只改日志seed。当前无可执行新训练命令，所有candidate NOT_AUTHORIZED。

分析预定：保留已有seed42及两个更新seed，按每seed配对B4−B3/B5−B4报告C5/C1/C2的Accuracy、MacroF1、逐类Recall、暴露错误率和成本，三点mean/SD只衡量更新随机性；不把40暴露与3seed相乘做独立n，不以n3声称稳健显著。无目标Test逐轮callback，只固定25。方向混合则明确不稳定；新两次均无改善则停止继续为该机制做增益验证；新两次改善但seed42负仍不能宣称普遍优势。任何结果均不触发调参/追加seed/重训Sstar。

H36-T：C3 A4若单独批准，先只比较B3/B4以检验反馈条件迁移；C3Day8Calibration、Day17Test、源截至Day7，独立Dev不参与选择。Sstar只能源Day7，不能用C5Day18模型初始化。C3 Test历史已被回归研究观察，不是新盲测。当前data validator只认target5、PhaseClient/B1/DA target binding硬编码5，必须先解决真实C3身份合同，不能把C3冒名client5。B5不是默认扩展分支。

先批准哪个需依据论文问题及资源，不能按C5Test最优值决定跨设备策略。优先低成本的C5更新随机性诊断，C3列后；也可以在不需要稳定性主张时直接停止训练、完成论文边界与系统合同。本轮不执行任何候选。
