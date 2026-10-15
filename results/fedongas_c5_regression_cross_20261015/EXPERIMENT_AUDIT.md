# 运行后核验

状态：36/36固定流水线已完成，未重训神经网络、未重训source FedRidge、未访问QC。
pre-run freeze：`04f4f3883e248b9f23d61d01bd6a0aa30e68dd24`。

## 已验证

- 原80/320标定窗口清单严格nested，40条件每格2/8，固定1360 test；canonical原始数组及预算哈希不变。
- 18个TO/FT/CLASS分类endpoint、checkpoint文件SHA、ordered state fingerprint、原36720行分类预测缓存均锁定，不加载或修改模型权重。
- 回归浓度读取原budget的真实regression_labels并逐行对齐metadata，不使用分类view零占位。
- 83D/H1从canonical 50×8重新提取，未复用legacy10Hz/100×8缓存；H1特征序与冻结source模型104D名字完全一致。
- source prior包整体通过旧R0-v2 validator，model/alpha/scaler/coeff只读、三个classifier和seed共同使用。
- 4个目标bank、16个最终gas fits；固定7候选alpha、5折filename-grouped calibration CV，fold内standardization和最终calibration min/max clipping。
- `TEST_RELEASE.json`锁定input/classifier/model/alpha文件SHA后才读取本次regression test；正式36组评价仅执行一次。
- 36行主表、12组mean/sample SD、144行主逐gas、144行scope、576行逐gas scope、48960行逐窗口输出。
- 66条按seed配对、22组配对汇总，以及对应逐gas配对；保留零/负收益。
- 复算所有144条总体scope的RMSE、MAE、NRMSE、R2、Bias，与保存结果一致；4个Oracle_ALL bank在各classifier/seed重复值严格相同。
- 另外用保存的逐窗口CSV独立复算全部144条主逐gas RMSE/MAE/NRMSE，并重新聚合12组总体、48组逐gas三seed均值/SD及66条配对差值，全部一致。此步骤只读保存结果，不重新拟合或释放test。
- 从保存的test feature cache与target_model_lock重建4套bank的四路输出，与逐窗口预测一致（无refit、无test原数组重新加载）。
- 输入/checkpoint/预测缓存和结果索引SHA复核通过。`sha256_index.json`为原执行产物索引；后加解释/审查材料由`FINAL_EVIDENCE_SHA256.json`另行覆盖，不改原索引或COMPLETE。

## 测试

运行前、运行后：12项新增测试与17项既有R1数值/审计测试均通过（29 passed, 1 deselected）。
完整选定3个测试文件的首次执行为29 passed/1 failed；该旧失败项`test_runner_inspect_does_not_create_formal_root`要求已存在的历史R1正式结果目录不存在。未删除历史结果、未改旧测试。最终相关测试显式排除该前提已不成立的断言。
新模块、runner和两测试文件`python -m compileall -q`通过。未声称全仓测试通过。

## 判断边界

- 本次改变因素仅为缓存分类路由与回归输入，source prior固定共享；分类器之间没有回归系数差异。
- shared oracle不是独立seed重复；S_CC/Oracle_CC为相同正确路由子集，不能用其差异声称映射改进。
- H1含协议/phase；原窗口级划分、邻近时间窗口相关性限制不变，无统计显著声明。
- 没有用test挑选checkpoint/alpha、改变特征/方法/矩阵；没有看到不理想结果后追加训练或优化。
- unrelated dirty code及历史结果全部保留。本次只提交自身代码/设计和新输出根，不提交旧checkpoint、原始数据或其他任务修改。

结果：可用于本固定协议的描述性比较；不能外推为FULL/阶段机制或普遍算法优越性的证据。
