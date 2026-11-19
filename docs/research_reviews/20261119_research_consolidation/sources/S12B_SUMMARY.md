# SPRINT12B_SUMMARY

**DONE · GO_TO_INDEPENDENT_CONFIRMATION_DESIGN**。按已冻结合同完成8步、五气体、2/4条测量记录预算的最小回顾性验证；不增加规则。GO只允许申请独立确认设计，不代表已证明跨时间泛化、业务合格或算法创新。

|主要实际策略|2条/气体|4条/气体|
|---|---:|---:|
|Static|12.20%|12.20%|
|JointUpdate|33.53%|40.02%|
|ClassifierOnly|29.13%|36.61%|
|RegressionOnly|6.66%|9.63%|
|PerGasLOEO|34.84%|43.65%|

主指标为40个气体—时间案例等权的J10S-raw：分类正确、raw有限且误差≤10%源浓度跨度；错类/不可用均失败。Per-gas相对Joint为 **+1.31 / +3.63个百分点**，相对Classifier-only为+5.71 / +7.04个百分点。同分类器Oracle-J为37.62% / 45.75%，余差2.78 / 2.10个百分点。Oracle按气体选整头，未参与实际选择，亦非NRMSE最优头。

**增益有限且不均匀**：两预算池化micro仅23.81%→23.95%、26.98%→27.19%。2条预算相对Joint为3案例改善/4退步/33不变，4条为3/2/35。不得写成所有气体稳定改善。分类输出与Joint、Classifier-only逐记录完全一致。

|预算|有害头：Joint→Per / 可评分|累计额外NMSE|最坏NRMSE增幅pp|Per新适配+Source恢复|组件包总字节：Joint→Per|
|---|---:|---:|---:|---:|---:|
|2|12 → 9 / 38|0.1782 → 0.1041|14.69 → 8.90|30 + 7|63,124 → 62,858|
|4|13 → 7 / 39|0.2557 → 0.1525|24.35 → 10.50|32 + 6|63,124 → 62,908|

Per有益更新遗漏为7/2次；无正确路由的2/1案例保持NOT_ASSESSABLE。C_U风险与Oracle_ALL诊断分表。五策略共同正确路由交集C_I为2078/2314记录、25/26个有定义气体—时间案例；Per与Joint的可用案例描述均值为25.24%对25.35%、26.68%对27.02%。这些不是完整五气体宏NRMSE，不能跨预算比较；只有Cal6→Eval7具备完整五类交集，其余严格macro为NA。详见CONCENTRATION_METRICS.csv及CONCENTRATION_MACRO.csv。

**剩余失败**：2条预算Acetaldehyde Cal4→Eval5拒绝了有益头，J10S下降45.65pp；4条Acetone Cal8→Eval9遗漏收益，下降29.49pp。NRMSE有害接受共9/7次，Cal9→Eval10 Ethanol最坏仍达6.44/10.50pp。部分J10S改善来自保留Source，不能认定其校正方向已普遍可预测。完整25条有害/遗漏NRMSE案例保留在PER_GAS_FAILURE_CASES.csv；J10S增退步、样本数及覆盖在JOINT_CASE_DIFFERENCES.csv，二者目标不同。

**标签与成本**：五气体Source共用371条历史有标签记录；完整冻结bank另含74条未使用的第六气体。当前每步10/20条、累计80/160条，B2嵌套B4、联合唯一160条，均是测量记录。拒绝更新仍消费ppm标签，Classifier-only无需当前ppm。额外完整历史评价标签用于决策为0。Per接受30/32次、恢复Source7/6次，实际头变化37/38次而非30/32次。统一schema载荷仅节省266/216字节（约0.42%/0.34%），不主张显著通信或总计算优势；真实网络、在线重拟合、功耗均NOT_MEASURED。Source假定常驻，包元数据及恢复引用均计入。

**执行与验证**：复用17个分类checkpoint、80个当前Bias、240个LOO折；用原单记录兼容Bias适配器核对240折，20条Cal源前向、140条源预测缓存复用；无新基础模型拟合、无新未来批次推理。全部80个决策→80个实际视图锁定→释放评分标签→事后Oracle。104670实际视图行+20934 Oracle行对应10467唯一评价记录，不能扩增统计n。8项定向测试、独立封存行算术复核6074项通过，125个所用历史输入SHA未变。该复核是独立程序，不冒称独立人员审阅。

**下一步**：保留既定Per-gas LOEO作为强基线，先提出未用于选法的数据/任务确认合同；对上述遗漏与有害接受只做封存结果分析。不得在这8步上增加门槛、改tau或选择新规则。不自动开展新训练、外部验证、Twin Day22、Flower或Pi操作。

本次执行命令（已执行，不自动重跑）：
[Local machine path/command omitted in public snapshot; original SHA retained.]

[Local machine path/command omitted in public snapshot; original SHA retained.]

封存行只读复核函数：`verify_delivery.verify(Path('development/fast_sprint12b_20261119/results/run1'))`；CLI会以x模式保存独立收据，已存在时不会覆盖。所有新文件在本目录，冻结资产/main未修改；未提交或合并Git。
