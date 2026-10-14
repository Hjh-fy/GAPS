# Execution ledger — plan: docs/superpowers/plans/2026-10-14-fedongas-c5-budget-multiseed.md

2026-10-15：继续已授权27任务，未新增研究范围。
Task 1–3：预算/共同初始化/CLASS/target-only及入口14项测试通过；保留dirty核心文件，不混入runtime。
Task 4–5：顺序控制器4项RED→GREEN；新测试累计18通过。相关旧协议与runner测试共42通过（protobuf依赖2项deprecation warning）。
Task 6：输入准备PASS；5%80/20%320，40strata2/8，nested，无test重叠，test1360，shape50×8，C5phase2；三机SSH/磁盘核验PASS。
Pre-flight shared interfaces：protocol→entry/runner使用共同initial asset；runner→evaluate以27端点/commit/hash为release gate；FT→SRC使用同seed round25精确hash。
Ruling：沿用prototype MMD0.2于CLASS/FULL — 已核实A4有效损失应保留，避免额外变量 — 若认定不同需作为新协议而不是改本次运行。
Ruling：CLASS兼容wire key(c,0)，0是生成占位、不是真phase — 原序列化协议需要二元key — 必须结合直接class统计和无真实phase输入理解。
Ruling：保留固定source calibration cap500/RandomState42取样 — 只改变训练seed，不重新抽样源支持 — 不应解释为全source样本参与每次DA。
Task 7等待全部正式端点：统一评价、保留失败/零负收益、验证、紧凑证据commit/push。
Final review fixed：fresh-process懒加载builder捕获错误 — 先生成所有命令后替换分发函数 — 新独立进程测试RED→GREEN，避免首项SRC递归/StopIteration。
Final review fixed：完成gate只检查身份/hash — 增加PASS、updates、test未开放、checkpoint实际round/updates及ordered fingerprint — 错误status/round测试RED→GREEN。
Reviewer未评价的正式数值收敛/远程执行结果须待实际运行；网络/数据已核验，不预先声称精度或完成。
完整仓库pytest已执行：228 passed，1 failed，2 errors（--maxfail=3停止）。历史失败：test_close_iotj_manuscript_protocol_safety.py::test_default_mode_verifies_formal_files_without_writing 的旧index hash不同；test_deployment_git_closure.py 两项tracked checkout fixture因Windows路径太长失败。详细日志 results/fedongas_c5_budget_multiseed_20261014/repository_pytest.log。不修改历史论文资产或部署模块来使本任务全仓测试变绿。
最终启动验证：新测试22/22及相关协议/runner测试共46/46 PASS；compileall PASS；两项review修复均RED→GREEN。固定输入三机逐文件71项hash PASS。正式训练仅用新freeze commit，不含未提交核心改动。
