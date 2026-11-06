# Phase3.8R Calibration数值诊断实施计划

用户本轮指令为执行依据；旧Phase3.8固定提交4dbf732df9ed693c88cb6486099930a428acc90b，仅允许静态读取公开摘要/五条封存差异。新目录不复用旧授权。复用当前外部管理的独立worktree，不修改任何旧Phase或v5。

目标：区分输入/state/序列化/接口差异、batch形态数值差异及后端开关影响，原容差atol1e-4/rtol1e-5不变，旧失败不解除。不选部署模型，不训练/refit，不读Day22。

## 固定设计与文件接口

- common.py：标准库身份、独占写入；core.py：固定indices及逐分量描述；test_core.py：容差相对项、重复位置、拒绝非Calibration输入。先RED再GREEN。
- register.py：注册六模型文件/state SHA、P37代码/容差、Calibration两个输入pack/metadata SHA、旧五事件SHA、各新代码SHA，固定运行ID P38R-CAL-BATCH-BACKEND-v1。
- diagnose.py：只加载P37原FrozenAssets；原Calibration guard之外额外禁止旧evaluation、Day22及P38完整逐窗文件，无SHA旁路；固定32窗模型输入，核对16暴露Day18身份和P37保存信号SHA，不读取Test。原SVR资产仅核验原loader，不调用回归推理或fit。
- summarize.py：仅分析保存Calibration/合成输出，写逐模型矩阵、原容差结论、根因确认等级、PC合同提案和继续/停止；不会调用模型。
- 最后独立代码/证据审阅、旧已许可资产SHA复核、新目录封存，精简GitHub发布与索引更新。保留计划/源码/测试/失败，不自动本地提交或合并main。

## 预定比较

H-I：同模型/同输入/同batch，不同接口严格相同；失败停止接口结论。P35的main包含Test访问不调用，只复刻并校核其中已冻结的forward表达式。P37/P38公开接口只支持batch1，真实调用两者32次；batch16/32/128不存在旧adapter API，用显式标注的新诊断batch shim和P35来源等价loop比较，绝不冒称旧接口。

固定六形态：batch1×32；batch16×2；batch32；batch32顺序roll7；batch128 tile(0..31,4)；batch128 repeat每窗4次。记录同输入在批次位置和重复位置差异。batch128只有32个独特窗口，16次暴露，不是128次独立数据。

H-B：其余不变跨batch是否超过原容差；未超过也报告最大差/相对差/逐分量和整窗通过率、argmax、Top1/2间隔。ref固定各模式自身batch1，不用Test选择batch。

H-K：独立2×2后端诊断，先保留原模式作为基线，再固定比较MKLDNN关闭、MHA fastpath关闭、两者关闭；线程1、float32、CPU、eval/inference_mode不变。全部模式都是诊断，不改生产默认模式。每模式均跑六形态，Calibration32与固定seed20261106合成32×50×8；不根据结果扩搜索。模式结束恢复原开关，再核对原batch32。

逐层只读forward hooks覆盖实际执行的TCN/Attention/投影/分类头及模型输入/输出。对齐同输入位置，与同模式batch1比较首个非逐位一致层及首个超过描述阈值层。比较有/无hook batch32，避免hook改变数值路径。B0四模式batch1/32记录CPU profiler算子事件（无权重/信号），辅助证明实际算子路径；不将其他模型的算子路径假定相同。

同batch原P35来源等价loop、诊断batch shim、重新strict load模型在原后端下比较；实际P37 Adapter和P38 ObservedClassifier在batch1比较。原no_grad表达式与外层inference_mode及no_grad-only各留独立控制记录。

资源上限：六模型、两输入域、四后端模式、六batch形态；全模型forward调用最多3200次，超限fail closed。不进行梯度更新，不用类别标签选择实现。指标仅数值一致性，不算Accuracy/回归误差或联合性能。

## 审阅重点与停止条件

原容差逐分量相对参考abs值；零参考相对差只作描述，不能改变gate。原state/输入身份不符、非finite、eval/inference关闭、接口同batch不一致、前后state变化或访问未许可数据均阻塞对应结论。后端模式的负结果保留；某模式通过不意味着历史合同通过。固定函数返回的64D encoder仍不能当71D回归输入。

序列化只静态审阅P35保存代码、已允许五条logits故障摘要及合成float32→tolist→JSON→float64→float32往返；不打开完整历史Test CSV或1376条记录。可证明五条摘要中数值可无损表示float32，不能声称已重新验证完整历史CSV行连接。

原Phase3.8严格验收始终BLOCKED。仅在Calibration证据闭合后提出独立版本PC执行合同和验收范围；未授权Day22 attempt2，无有效执行入口/授权，不部署Pi。
