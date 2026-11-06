# C5分类—71D回归兼容性

门禁结论：身份/模型来源与静态合同PASS，数值桥接、运行路由和端到端性能NOT_EXECUTED，不把metadata通过称为部署通过。

|项|本轮核对|状态|
|---|---|---|
|共同输入身份|Phase35分类与Phase1/1.5回归的C5Day18A4相同16暴露32窗；Day22相同40暴露840窗；文件rawSHA、gas、start/end逐项完全连接；cal/test互斥|PASS metadata|
|源与祖先|分类合法Day7 Sstar及25轮新目标更新；F-SVR源侧选参仅合法源、目标头仅Calibration；模型文件与训练身份SHA固定|PASS archived evidence|
|原始预处理|同v5 strict process_raw：排序、时间重复合并，20–50s原始电导g0，5Hz分箱/孤立缺失插补，响应(1/R−g0)/g0，crop60–170，有效10s50×8窗、stride5s|同代码来源；本轮未生成特征|
|模型输入|分类float32[N,50,8]、logits[N,4]；回归是同窗抽71D、固定71名字/顺序；encoder64D不可替换71D|PASS schema|
|标准化|分类原合同不额外fit全局Test scaler；回归每气体8个Calibration窗的scaler已封装，推理只transform|静态/既有回放PASS；本轮不loadjoblib|
|类别/输出|0Ethanol、1CO、2Ethylene、3Methane；F-SVR四头FONG-P15-C5-A4-F-SVR-20261102，文件及身份SHA核对|PASS metadata|
|物理支持|Ethanol/Ethylene12.5–125ppm，CO/Methane25–250ppm，来自各头Calibration浓度；裁剪支持依预测gas所选头|PASS declared support|
|runtime桥接|Calibration同窗浮点/特征SHA、模型加载、错类也调用相应完整头、NaN/非法class保护、单窗回放|PENDING future approval|

ROUTING_REGRESSOR_HEADS含四个模型的SHA、训练身份、超参和支持。SOURCE model不能因Source-only结果好而事后替换F-SVR。四头不接受分类encoder、不使用真实类别帮忙选头。以真实gas路由是Oracle，只能保留独立诊断。

下一阶段最小路由计划：先仅用已授权Calibration固定fixture连接50×8→71D及冻结头，要求与既有保存特征/预测合同一致；固定模型/特征/scaler/SHA，非法类别或非有限值返回不可用，不Oracle补救，不让clip吞掉Inf。其后若单独批准离线结果，使用同一F-SVR71银行配全部六组已锁分类端点（可复用最终分类CSV及SHA），共6种固定路由840窗，不基于已有Test挑一个分类器。错误类别必须调用预测头；不能只保留分类正确子集。原始与裁剪浓度分开，gas错误与正确类别下浓度误差分开。

主报告同时呈现类别指标、每预测gas的有效/不可用率、分类正确条件下浓度诊断和全部流量错分类方向；若要气体—浓度联合成功率，容差须由应用要求/合法Source/Calibration事先固定，不能本Test反求。跨气体ppm数值不是同一化学任务，不能用单个混合NRMSE掩盖错分成本。高代价错类须先有任务成本依据，本轮不凭频数发明风险权重。

此方案不训练，不作Pi部署，也不在本轮执行任何新预测。该历史Test已看过，未来路由结果同样只能称回顾性系统描述，不能替代独立确认性数据。
