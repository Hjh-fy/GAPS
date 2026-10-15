# Experiment Audit — 最终六项验证

## Audit scope and intended claim

审计范围：C3/C4 TO、FT、CLASS主标定seed42六个新固定端点、复用C5三端点、CLASS75点曲线，以及C5四预算回归证据。冻结commit `20007966527949309a094605aebfb710b57eed2d`；无新增训练、推理或重拟合。

## Compared experiments

| Experiment ID | Split | Model/endpoint | DA/calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|
| XVAL-C3-TO-MAIN-S42 | canonical678/2677 | TCN-Attention update2500 | 仅target CE，random42 | none | 42 | runs/同名目录/FINAL_ENDPOINT.json |
| XVAL-C3-FT-MAIN-S42 | 同上 | update2500 | SRC42 round25后仅target CE | none | 42 | 同上 |
| XVAL-C3-CLASS-MAIN-S42 | 同上 | round25 adapted | 4类别原型，逐轮100step | none | 42 | 同上 |
| XVAL-C4-TO-MAIN-S42 | canonical320/1360 | update2500 | 仅target CE，random42 | none | 42 | 同上 |
| XVAL-C4-FT-MAIN-S42 | 同上 | update2500 | SRC42 round25后仅target CE | none | 42 | 同上 |
| XVAL-C4-CLASS-MAIN-S42 | 同上 | round25 adapted | 4类别原型，逐轮100step | none | 42 | 同上 |
| FONG-TO/FT/CLASS-B20-S42 | canonical320/1360 | 历史固定端点 | 原冻结同名方法 | none | 42 | 既有FONG27 marker及输入hash锁 |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| FIN-01 | informational | 完整性/身份 | 6/6原completion gate；CP ordered-state fingerprint及SHA核验，206项冻结输入hash | 固定端点完整 | 保留所有结果 | PASS |
| FIN-02 | informational | CLASS实际方法 | 50份原loss audit：CLASS、4proto、true_phase_used=false、100steps、sourceCE active、targetCE/stageMMD inactive | 非旧A4改名、非仅关stage weight | 不改变方法 | PASS |
| FIN-03 | informational | 预测/曲线一致性 | 12111新正式预测；75点/134925曲线记录；所有逐轮cache/CP SHA和重新计数Acc/F1一致 | round25与对应CLASS正式端点一致 | 封存兄弟目录曲线资产 | PASS |
| FIN-04 | informational | 回归复用/特征合同 | 14新回归产物hash及COMPLETE一致；5/10/15%对应20%的sensor83/H1/prior最大差均0 | 无跨预算提取器更换证据 | 不重拟合、不改clipping | PASS |
| FIN-05 | minor | 全仓健康 | 本轮相关43passed，compileall0；全仓既有1604passed/84failed/9errors/5skipped | 不支持全仓无缺陷声明 | 明示限制，不扩大修复 | retained |
| FIN-06 | informational | 系统退出 | PID35644已结束，三机均无gaps_flower训练进程 | 未留后台训练 | 不启动新控制器 | PASS |
| FIN-07 | minor | 推断范围 | 单seed、目标原窗口划分、CLASS额外source训练成本 | 不支持广义算法优越性/显著性 | 限定描述性结论 | retained |

## Leakage assessment

所有6个训练marker记录target_test_opened=false，均早于SEALED_TEST_OPEN记录；全部固定终点验收后才统一评价。原hash/input lock保持一致；本次核验只读取预测并计数，没有重新模型推理。曲线为post-hoc描述，不选择checkpoint、调参或提前停止。标定class标签使用按方法定义审计：TO/FT用于CE；CLASS用于类别条件损失及prototype，不使用真实phase或浓度训练分类。

## Baseline, completeness, and reproducibility assessment

目标更新均2500，TO/FT仅目标训练，CLASS有额外25轮C1/C2 LE1训练；因此不是总计算受控单因素消融。FT源模型匹配既有SRC42，不重新训练源。C5只复用匹配seed42的正式端点；三目标不宣称均为三seed。Oracle回归按真实gas路由，不把不同分类器重复当作独立回归实验。

## Verdict: approved

仅批准本冻结协议下的描述性Evidence：CLASS在C3/C4近饱和，优于匹配FT但不在三目标全面优于TO；R84四预算总体降低Oracle误差但存在逐gas负迁移与非单调。无本轮阻断性比较/来源问题。全仓未全绿、单seed及数据划分限制保留；不是正式投稿全项目无缺陷认证。

## Unknowns and handoff

未知：跨更多设备/划分的泛化、统计显著性、源参考饱和的最优改法。按授权停止，不运行FULL、更多seed、其他预算分类、QC或算法搜索。文件级封存在FINAL_EVIDENCE_SHA256.json，未公开的raw/cache/checkpoint只留哈希与本地定位。
