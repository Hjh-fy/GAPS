# Phase1.5 预注册：公平性补强、特征诊断和条件KRR

日期2026-11-02，依据v5、Phase0.5及Phase1固定提交772c0e9087435b9deb8ea4ac50b7c24814f6b345。当前独立worktree内新建本目录；旧数据/模型/结果/代码只读。所有目标已经观察过，全部新比较均为回顾性探索，不能重新称首次盲测。

## A 两条控制规则，先注册后执行

F（主公平比较）：每gas、每预算独立选择，合法源只有两个预定episode。

|episode|训练|验证|日期属性|
|---|---|---|---|
|E1|C1 Day4，A2=F010/F100或A4=F010/F040/F070/F100，各取60/150起点两窗|同C1 Day4其余固定六档F020/030/050/060/080/090全部有效窗|仅文件互斥、同日；日内真实顺序未知|
|E2|C2 Day1，完全相同档位/窗口|C2 Day7全部十档有效窗|文件互斥，跨相对日1→7|

每gas训练4/8窗口，对应2/4次独立暴露；E1每gas验证6暴露126窗，E2每gas10暴露210窗。每episode先按暴露等权取未裁剪归一化MSE，再对两个episode等权平均，最小值选参，同分首候选。两个任务重复使用源队列，不当成多独立数据集或多随机种子。

F-Ridge alpha仍为[0,.01,.1,1,10,100,1000]；F-SVR C=[1,10,100]、gamma_factor=[.1,1,10]，epsilon=.01。无目标Development/Test选参。参数按gas×A2/A4锁定，不事后扩大网格。

N（正则损失归一化机制控制）：不重选Phase1参数，固定n_ref=(210+420)/2=315，仅改变目标惩罚系数。Ridge alpha_new=alpha_Phase1*n_target/315；SVR C_new=C_Phase1*315/n_target；gamma和epsilon不变，n_target=4/8。原Ridge目标为sum squared error+alpha*norm，平均损失下lambda=alpha/n；原SVR相对正则约为1/(C*n)。315仅由源折规模规定，是可复现参考规则，不是由原CV唯一推出或从目标分数选出。P1→N同数据/scaler/核宽，只变alpha或C；P1→F同时改变选参任务，不能单归因为一种因素。

新增A矩阵=3target×2budget×2rule(F/N)×2family(Ridge/SVR)=24配置、96头。Phase1原12配置只读引用、不重新拟合。A参数与模型锁定后才执行KRR源选参。

## C 一次条件性KRR

仅在A身份/参数/模型合同完成后执行，固定F源episodes、episode等权未裁剪MSE。KRR alpha=[.01,.1,1,10]、gamma_factor=[.1,1,10]，共12候选，无扩网格。训练scaler同Ridge/SVR，标签同固定LOW/RANGE。RBF核矩阵训练中心化、标签训练均值中心化并反变换，保留未惩罚截距；用sklearn KernelRidge(kernel=precomputed)。线性核与原Ridge的等价性及逐窗/批量一致性先smoke核对。

KRR只走F规则，新增6配置/24头；与F-Ridge/F-SVR作主比较，N两者作控制，绝不从目标结果选择哪套规则来赢。全部30配置/120头锁定后统一评价目标Test。三日期C3和两日期C4/C5分别标注，C4继续仅作回顾性source-before-test。

## B 预定描述性诊断

重点为C3 Ethylene、C4 Ethanol、C5 CO，保持全部其它gas结果。窗口仍为单个50×8、5Hz、10秒，cal起点60/150，test21个起点60:5:160；早/中/晚仅描述起点{60,65}/{100,105}/{150,155}，不声称已测物理稳态。

- 边际cal/test每71维均值/标准差/训练scaler标准化偏移；浓度组成不同导致的混杂必须标注。
- 同已见浓度、同窗口起点60/150的cal/test配对，避免把浓度构成差异直接称漂移；每浓度每日期只有一次暴露，不据窗口数制造独立重复。
- 保留最小标定标准化距离、SVR/KRR最大RBF相似度、残差随浓度/时间变化；不用于参数或特征选择。
- 已见端点按e_test=e_calibration+(p_test-p_calibration)精确分解，区分校准拟合偏差与模型对输入变化的输出差；不直接归因物理老化。
- Ridge按原系数精确分解预测变化；SVR/KRR按校准到测试的直线路径做integrated gradients（quad_vec epsabs1e-8/epsrel1e-6），记录加和残差，按8通道+global汇总。共线性、路径/参考依赖明确，不称因果重要性。积分误差>1e-3ppm标警告，不修改模型或临时重选参考。

## 固定评价/资源与闸门

同Phase1的Oracle_ALL单窗口误差、暴露等权主NRMSE/MAE；clip到标定支持为主，raw全保存，暴露均值融合仅辅助。目标Development不用。复用相同cal/test缓存、scaler规则和质量掩码。

源F选参256fit（8gas-budget×16候选×2episode）；N无新source fit；KRR源选参192fit（8×12×2）；新target120fit；smoke四次合成fit（Ridge、SVR、RBF-KRR三模型+线性KRR参考），合计572fit上限。均为4/8训练窗的小模型，除N规则不重跑完整源CV。0神经网络/Flower/部署/源贡献重开发。

不论优劣全部保存。任务A若身份、时间或模型/参数合同失败则停止C。哈希、特征、原始暴露互斥、cal预算不符即停止；不能用随机窗修复。拟合次数、序列化往返、模型锁、Test释放、验证与GitHub发布均留痕。条件性bootstrap可沿用Phase1按gas分层2000次，仅作固定模型/单test日描述，不宣称跨日独立显著性。

研究决策先看N相对P1的受控差异，再看F公平比较，最后KRR是否仍有可复现收益。超参数尺度修正与KRR本身不是原创算法；跨设备知识复用只有在合法Source71及负迁移控制成立后才可能继续。本阶段不新增残差拟合。
