# 2026-10-15 运行状态

正式冻结：`20007966527949309a094605aebfb710b57eed2d`。

- C5回归10%/15%：4套新bank、16个最终gas拟合完成；5%/20%未重跑。14个输出文件hash与COMPLETE index引用核验通过。
- 新回归结果：`results/fedongas_c5_regression_budget_extension_20261015`。
- C3/C4分类六项队列已在后台启动，控制器PID35644。当前动态状态以`results/fedongas_c34_class_validation_20261015/RUN_PROGRESS.json`为准；首次检查C3 TO完成、C3 FT运行中。
- 每20分钟heartbeat `c3-c4-to-ft-class` 已启用，正常无变化不重复通知；最终核验及GitHub同步后删除。

## 验证范围与未解决限制

本轮及相关FONG测试43 passed，新增测试9项包含RED→GREEN。7个新Python文件compileall退出0。独立只读review无Critical/Important发现。

全仓检查已真实运行：

```powershell
D:/anaconda3/python.exe -m pytest -q --basetemp=.tmp_fongext_fullsuite_20261015 -ra
```

结果：1604 passed，84 failed，9 errors，5 skipped，2 warnings，478.64秒；**全仓不是全绿**。原始完整输出保留于工作区`.tmp_fongext_fullsuite_20261015.log`，未覆盖或删除。

失败/错误涉及旧manuscript协议、B5消融、R0v2、confirmation、runtime/deployment等测试；已观察到历史资产缺失，例如`results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc95_records.csv`不存在，portable release与tracked-only checkout fixture亦有错误。未逐个归因全部84项，不宣称全由环境导致。本轮43项通过不能代替全仓健康证明。本次不扩大到这些旧模块修复，也不修改现有用户dirty核心文件；远程训练运行已冻结commit的源码归档。

最终曲线封存还需显式覆盖`class_round_curves`目录，不能仅检查`evaluation/sha256_index.json`。该项在独立review中列为Minor，正式训练与首次评价不受影响。
