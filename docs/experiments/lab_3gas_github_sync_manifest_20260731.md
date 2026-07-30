# 实验室三气体 GitHub 同步清单

## 同步范围

本次提交保存实验室自测三气体分类从数据构建到三机运行、正式评估和结果审计的可复现
代码及轻量证据。

### 代码

- `federated_dataset.py`
- `gaps_flower/client_app.py`
- `gaps_flower/evaluate_checkpoint.py`
- `gaps_flower/server_app.py`
- `gaps_flower/strategy.py`
- `gaps_flower/task.py`
- `scripts/lab_three_gas_3class/`
- `scripts/remote_launch_flower_client_clean.py`
- `scripts/remote_launch_flower_server_clean.py`
- 对应 `tests/test_lab_three_gas_*.py`
- `tests/test_remote_launch_flower_server_clean.py`

### 实验设计与分析

- `docs/experiments/lab_*.md`
- 实验 matrix、registry 和 retry registry CSV
- 最终结果分析：
  `docs/experiments/lab_3gas_accuracy_recovery_final_analysis_20260731.zh.md`
- 最终实验审计：
  `docs/experiments/lab_3gas_accuracy_recovery_final_audit_20260731.zh.md`

### 机器可读结果

- REC-A1/A2/A3/A4/A5 的 `formal_evaluation_summary.json`
- REC-A1/A2/A3/A4/A5 的 `postflight_attempt_audit.json`
- 固定协议的 `protocol_manifest.json`、`dataset_manifest.json` 和
  `source_manifest.json`
- A1 stable common-scope 与 A2 round-1 故障诊断 JSON
- 全浓度准确率差距分析输入与诊断 JSON

## 有意排除

- `dataset/Dataset_self` 原始实验数据；
- 生成数据集中的 `.npy` 特征和标签；
- `.pth`、`.pt`、`.ckpt` checkpoint；
- SSH 隧道、stdout/stderr、控制器运行日志和 PID；
- 内容寻址 source tar 包及远端 runtime 缓存；
- `__pycache__`、pytest 临时目录及其他项目的脏工作树文件。

这些文件体积较大，或包含机器本地运行状态，不适合直接进入普通 Git 历史。数据身份、
文件哈希、样本数、通道、划分、归一化边界和源码 archive SHA 已由协议 manifest、
正式结果 JSON 与审计报告保留。

## 当前结果边界

- P2→P3，三分类，全浓度 time-purged，seed 42；
- 25 个联邦轮次，本地 3 epoch，固定使用第 25 轮；
- 完整 420 窗口最佳当前结果为 94.52%；
- 稳定段 REC-A4 为 359/360=99.72%，coverage=85.71%；
- 结论属于 nominal-boundary、single-seed、post-hoc 探索结果，不作为冻结论文证据。
