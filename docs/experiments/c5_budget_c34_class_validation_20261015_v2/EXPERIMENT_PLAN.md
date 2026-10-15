# C5回归预算与C3/C4 TO/FT/CLASS验证：正式授权v2

用户2026-10-15最终授权：CLASS为主候选；C3/C4各TO、FT、CLASS，主标定条件，seed42，最多6分类任务；FULL不追加。此前不含FT的draft只保留历史，本文件取代其待确认范围。

## 假设与固定矩阵

- H1：C5源参考增强的真实类别RMSE/NRMSE效应随5/10/15/20%预算变化；保留不单调/负收益，不预设优势。
- H2：TO/FT/CLASS在C3/C4主标定下的分类差值；seed42是描述性验证，不声明显著性或多seed稳定性。
- H3：三目标同协议83D/R84的真实类别回归；复用已有C3/C4结果，不把classifier路由差异叫作共享回归映射改善。
- H4：最终CLASS的逐轮轨迹与对应固定round25端点一致；旧A4曲线不替用。

### P0 C5回归

仅新增10%/15%×83D/R84=4bank、16最终gas Ridge fits；原固定清单160/240窗口、40 strata每格4/6，5⊂10⊂15⊂20，test1360不变。5/20%原结果只复用。
固定canonical-v1 5Hz/50×8；重新提取新增标定83D/H1，复用已核验的同一C1/C2 R0-v2 H1 source prior。保留原source clipping，不能因先前发现饱和而改定义。
Ridge：float64、pinv、fold内标准化、intercept不惩罚；alpha 0/.01/.1/1/10/100/1000，5-fold raw-filename-grouped calibration CV pooled RMSE/first tie；最终按gas calibration min/max截断。没有新增搜索范围。
全部4bank/alpha/input锁定后才读取原已保存test feature/prediction cache。主表Oracle_ALL：每gas340、总体1360，NRMSE逐行按真实gas量程归一化；不虚构分类seed回归SD。
输出四档8行总体、32行逐gas、4新bank系数/alpha、5440新bank逐窗口预测、源参考方差诊断、三目标旧回归复用表及中文分析。

### P1 六分类任务

按C3 TO→FT→CLASS→C4 TO→FT→CLASS顺序，独立保存端点。C3原标定678/test2677；C4原标定320/test1360，不重抽、均为原主标定命名20%。
seed42、batch32、Adam5e-4。TO/FT各25block×100纯目标CE更新，每block重建Adam、encoder/head均更新，不读source CE/原型/target phase或浓度。固定update2500。
TO使用既有FONG random-S42原始初始化；FT使用既有匹配FONG-SRC-S42纯源round25完整模型（C1/C2、25轮LE1），两目标共享该源端点，不重训SRC。
CLASS独立从同一random-S42开始：C1/C2 Flower25轮、LE1，每轮100服务器适配，adapted参数反馈source；固定round25 adapted，不从C5适配端点或SRC末轮直接微调冒充CLASS。
CLASS实现复用已冻结fedongas27_methods/entry：统计、prototype库、语义查询均class-only，4prototype；真实phase在训练input view中被移除；client align/replay关闭、普通FedAvg聚合，target CE0/source CE1/stage MMD0，其他原权重、EMA、优化器、更新步骤完全不改。
源视图原样保留canonical x/class，CLASS phase及regression只放兼容零占位；target worker只收到x/class。原真实浓度只用于回归，不进入分类损失。
训练归档使用新pre-run commit的已提交Python文件，不能混入当前工作区其他未提交核心修改。新输入与输出根隔离。
优先复用匹配端点；本次检索未找到可直接复用的C3/C4 TO/FT/CLASS固定端点，A4/A0T/100步DA或旧数据不作替代。队列遇到已PASS的同协议完成marker校验后跳过，绝不重跑；partial保留并fail closed。

### P2 固定评价和收敛图

全部6个端点验证后一次性正式test评价，输出6行新目标结果、24行逐类指标/混淆矩阵、12111逐窗口结果；复用C5主标定TO/FT/CLASS seed42得到9行三目标表和9组方法差值。C5三seed原结果另保留，不混为三目标均为三seed。
CLASS曲线：C3/C4新25个adapted checkpoint；C5复用FONG-CLASS-B20-S42已有25个。正式端点先锁定；事后对round1..24评价，round25复用已锁定预测。全75原始点，无平滑、不早停、不选最佳轮。
图使用项目既有Python/matplotlib流程、Arial、可编辑SVG/PDF和600dpi PNG。单seed、全部测试窗口，round不是独立重复，无CI/显著性。旧20260928图为A4，不能因C5Accuracy相同就替代CLASS。
逐轮推理只用于描述，不反馈训练或本轮方法选择；不修改论文正文。

## 代码与命令

- prepare_fedongas_extension.py：核验canonical/旧端点/预算；保存6任务+4bank注册和隔离input archive，不训练。
- fedongas_extension.py：冻结矩阵、目标字段边界、CLASS命令；复用旧纯类别方法，不改旧27项脚本。
- fedongas_extension_target.py：C3/C4 TO/FT，仅x/class，调用原train_target_blocks。
- run_fedongas_budget_extension.py：仅4新Ridge bank，5/20不重拟合。
- run_fedongas_c34_extension.py：单控制器顺序队列、三机部署、checkpoint/更新数验收、失败保留。
- evaluate_fedongas_extension.py：6端点release、三目标表、CLASS匹配轨迹/图，已完成评价缓存校验后复用。

在既有worktree运行：
```
D:/anaconda3/python.exe scripts/prepare_fedongas_extension.py
# 新代码、测试和protocol提交为pre-run freeze后：
D:/anaconda3/python.exe scripts/run_fedongas_budget_extension.py --freeze-commit <FULL_SHA>
D:/anaconda3/python.exe scripts/run_fedongas_c34_extension.py --freeze-commit <FULL_SHA>
```

## 资源、进度与停止

server root@121.40.139.213 /root/gaps_env/bin/python；C1 gaps@192.168.137.172 /home/gaps/GAPS/gaps_rpi_env/bin/python；C2 root@114.55.171.63 /root/gaps_c2_cpu_env/bin/python。只读探针已确认三机torch可用、无活动Flower；无CUDA不影响既有三机CPU协议，不使用系统Python。
结果根：results/fedongas_c5_regression_budget_extension_20261015、results/fedongas_c34_class_validation_20261015。
Windows控制器后台隐藏窗口；runner.pid、runner.stdout.log/stderr.log和RUN_PROGRESS.json为进度入口，至多一个控制器。延续用户每20分钟监控要求，状态无变化不重复通知。
不扩C3/C4低预算、不加seed、不重跑C5分类/旧回归/SRC、不加FULL/QC、不优化源参考。保留正/零/负收益；测试、compileall、hash、分析后只提交本任务紧凑证据并push既有分支，然后停止。
