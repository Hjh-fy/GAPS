# Pre-execution audit

范围：36组C5流水线评价，0次新增神经网络训练；18个已有分类端点、4个共享回归bank、16最终gas fits。

- 新模块7项测试与runner5项测试经RED→GREEN；预算真实浓度、嵌套、路由行序、共享拟合、NRMSE和test lock边界已覆盖。
- `prepare`已核验canonical数据、80/320身份、固定test哈希、18个checkpoint与预测缓存、R0-v2源回归包。
- 预检发现新脚本转录source index hash漏两个字符；已与原R1脚本及磁盘SHA逐字核对修复。历史资产未变化，正式拟合未开始。
- 联合既有R1测试：29通过、1失败。失败为旧`test_runner_inspect_does_not_create_formal_root`断言历史正式R1结果目录不存在；该目录早已完成且存在，不能删除历史证据来使其通过。该失败不涉及本次数值函数/预算/runner，不修改旧测试或结果。
- 排除该已知环境/历史状态断言后重新运行全部29项相关测试；记录实际终端验证结果。compileall须通过后方可提交冻结版本。
- 源prior模型/scaler/alpha只读；目标alpha仅calibration grouped CV，所有bank锁定后才读取本次regression test。
- 不修改其他用户未提交的server_app.py、strategy.py、target_information.py或历史watcher日志。

执行命令、输出、解释边界见RUN_GUIDE.md和EXPERIMENT_PLAN.md。执行后保留负/零收益，不新增研究配置。
