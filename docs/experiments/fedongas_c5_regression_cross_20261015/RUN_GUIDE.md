# 36组分类路由×定量回归评价

工作目录：`D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1`。
分支：`codex/iotj-final-classification-le1`。本机CPU执行，不使用SSH或Flower、不新增神经网络训练。

## 已冻结输入

- 20261014研究的18个TO/FT/CLASS端点及`evaluation/predictions.csv`。分类器只决定路由。
- 原canonical-v1预算80/320及1360 test，读取原真实浓度数组，禁止用分类view的零占位标签。
- 已PASS的canonical R0-v2 source FedRidge。冻结source alpha/scaler/coeff不重训。
- 新提取83D/H1；目标Ridge的固定CV规则见`EXPERIMENT_PLAN.md`。4banks、16最终gas fits，36组流水线评价。

## 命令顺序

以下命令均在上述工作目录运行，PowerShell：

```powershell
D:\anaconda3\python.exe -m pytest -q tests/test_fedongas_regression_cross.py tests/test_fedongas_regression_cross_runner.py tests/test_iotj_canonical_r1_v1.py --basetemp=.tmp_fongreg_verify
# prepare只允许一次，已生成清单后不要重复执行：
D:\anaconda3\python.exe scripts/run_fedongas_regression_cross.py prepare
# 将新代码、测试、设计和protocol_manifest提交，使用该pre-run commit：
D:\anaconda3\python.exe scripts/run_fedongas_regression_cross.py run --freeze-commit <PRE_RUN_COMMIT>
# 已存在结果时只能核验，run会拒绝覆盖；不能为了查看结果重新拟合或测试：
D:\anaconda3\python.exe scripts/run_fedongas_regression_cross.py verify
D:\anaconda3\python.exe -m compileall -q gaps_flower/fedongas_regression_cross.py scripts/run_fedongas_regression_cross.py tests/test_fedongas_regression_cross.py tests/test_fedongas_regression_cross_runner.py
```

## 输出和停止

结果根：`results/fedongas_c5_regression_cross_20261015/`。

- `regression_results_36.csv` / `regression_summary_12.csv`：主S_ALL结果和三seed均值/sample SD。
- `per_gas_results_144.csv` / `per_gas_summary_48.csv`：按真实gas切片。
- `routing_scope_summary.csv` / `per_gas_routing_scopes.csv`：S_ALL/S_CC/Oracle_ALL/Oracle_CC。
- `paired_deltas_66.csv` / `paired_summary_22.csv`：按预算、seed配对误差差值，左减右，负值更好。
- `per_gas_paired_*`：相同配对的逐gas结果。
- `oracle_unique_banks.csv`：仅4个独立bank，不把重复Oracle当作独立随机回归实验。
- `CLASSIFIER_REGRESSION_REPORT.md`：中文结果、解读和局限，不预设收益。
- `predictions.csv`：36×1360逐窗口路由/浓度输出，用于复算，不用于再次选择模型。
- `input_lock.json` / `target_alpha_lock.json` / `target_model_lock.json` / `TEST_RELEASE.json`：拟合/选择与test访问边界。
- `PRE_RUN_FREEZE.json` / `FEATURE_AUDIT.json` / `sha256_index.json` / `COMPLETE.json`：代码、输入和输出来源核验。

共享回归器与三个分类seed无关，因此同预算/回归方式的Oracle相同；S_CC样本集合可能不同。
不新增FULL、其他target、QC、神经网络训练、源回归训练或其他算法。完成后commit+push并停止。
