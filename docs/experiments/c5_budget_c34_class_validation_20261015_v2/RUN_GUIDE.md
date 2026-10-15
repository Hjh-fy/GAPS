# 本轮运行与交接

工作目录：`D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1`。
Python：`D:\anaconda3\python.exe`。分支：`codex/iotj-final-classification-le1`。

## 冻结后入口

```powershell
D:/anaconda3/python.exe scripts/run_fedongas_budget_extension.py --freeze-commit <本次完整冻结commit>
D:/anaconda3/python.exe -u scripts/run_fedongas_c34_extension.py --freeze-commit <本次完整冻结commit>
```

分类控制器应在隐藏后台启动，stdout/stderr、runner.pid写入分类结果根目录。文件锁防止重复控制器；仍须先核对PID与命令行，不能仅凭受限进程视图认定退出。

## 队列与进展

- 分类：C3 TO → FT → CLASS → C4 TO → FT → CLASS，seed42；不运行FULL。
- 进展：`results/fedongas_c34_class_validation_20261015/RUN_PROGRESS.json`。
- 每项最终凭证：`runs/<experiment_id>/FINAL_ENDPOINT.json`；已完成项绝不重跑。
- 分类有效配置、日志、checkpoint：对应`runs/<experiment_id>`，远程回收资产在`remote_server`。
- 六项固定终点全部完成后，控制器自动执行`evaluate_fedongas_extension.py`一次；正式结果在`evaluation`。
- CLASS逐轮结果在`class_round_curves`，来源必须是CLASS，不使用历史A4曲线。
- 回归：`results/fedongas_c5_regression_budget_extension_20261015`；新建10/15%的4套bank，5/20%与C3/C4主条件按哈希核对后复用。

## 远程环境

```powershell
ssh root@121.40.139.213
# /root/gaps_env/bin/python
ssh gaps@192.168.137.172
# /home/gaps/GAPS/gaps_rpi_env/bin/python
ssh root@114.55.171.63
# /root/gaps_c2_cpu_env/bin/python
```

三节点上述venv均已验证安装PyTorch，可执行CPU训练；不能因system Python没有torch或无CUDA就判断不可用。新输入根：服务器/C2 `/root/GAPS/fong_c34_inputs_20261015`，Pi `/home/gaps/GAPS/fong_c34_inputs_20261015`。

异常时先核对三机进程、日志更新时间与已完成远程终点；保留失败证据。只修复基础设施或实现错误，不改变预算/初始化/损失/迭代数；不能维持协议则停止报告。勿覆盖/清理历史实验。结束后核验所有结果与曲线hash、测试、compileall，精确提交紧凑证据并push；不要提交raw输入、代码tar或checkpoint大文件。
