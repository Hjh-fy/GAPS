# 27 项 C5 实验运行与交接

范围已获本对话授权，设计日期 2026-10-14；实际准备/启动日期 2026-10-15。
工作目录：`D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1`。
分支：`codex/iotj-final-classification-le1`。不要修改其他未提交的旧核心文件。

## 固定队列

SRC seed42/43/44（3项）→ TO 5%/20%×42/43/44（6项）→ FT（6项）→ CLASS（6项）→ FULL（6项）。
共27项；不是历史结果复用。每个seed FT从本轮匹配SRC的round25初始化。
源训练25轮、LE1；目标2500updates。仅在27个正式端点全部验证后统一评价1360个C5 test窗口。
任务身份与配置见 `EXPERIMENT_MATRIX.csv`、`EXPERIMENT_PLAN.md`。

## 入口及职责

- `scripts/prepare_fedongas27.py`：核验旧预算清单/数据哈希，建立严格nested80/320窗口视图和共同随机初始化资产；不重新预处理。
- `gaps_flower/fedongas27_protocol.py`：27项矩阵与固定三机命令。
- `gaps_flower/fedongas27_methods.py`：直接class统计、class-only语义查询、独立target-only CE训练。
- `gaps_flower/fedongas27_entry.py`：加载共同初始化并封装原Flower入口，保存每轮损失/prototype审计。
- `gaps_flower/fedongas27_target.py`：TO/FT 25×100次目标CE，整模型更新，每block重建Adam。
- `scripts/run_fedongas27.py`：逐任务三机编排、输入部署、完成验证、失败停队列、最终评价与紧凑证据commit/push。
- `scripts/evaluate_fedongas27.py`：24项目标指标、8组mean/SD、18项配对差值、6组配对汇总及逐类/混淆矩阵。

## 命令

仅新研究首次准备（已准备后不要重复）：

```powershell
python scripts/prepare_fedongas27.py prepare
python -m pytest -q tests/test_fedongas27.py tests/test_fedongas27_inputs.py tests/test_fedongas27_runner.py
python scripts/run_fedongas27.py run --freeze-commit <完整冻结commit>
```

实际正式启动为Windows隐藏后台进程，pid保存在结果目录 `runner.pid`；不要同时再执行上述run命令。
控制器终止后若存在partial endpoint必须先保留日志诊断，不能删除目录假装新任务重跑。
全部完成端点可验证后跳过，不会再次训练；评价目录已存在时禁止再次自动评价。

## 查看进度和结果

```powershell
Get-Content results/fedongas_c5_budget_multiseed_20261014/RUN_PROGRESS.json
Get-Content results/fedongas_c5_budget_multiseed_20261014/runner.stdout.log -Tail 20
Get-Content results/fedongas_c5_budget_multiseed_20261014/runner.stderr.log -Tail 20
```

结果根：`results/fedongas_c5_budget_multiseed_20261014/`。
每任务：`runs/<experiment_id>/`；正式checkpoint和服务器每轮材料在 `remote_server/`。
正式完成判据是 `FINAL_ENDPOINT.json` 校验通过，不能仅凭目录或latest文件判断。
当前轮次查看 `runs/<current_run>/server.stderr.log`，客户端分别 `client_c1/client_c2.stderr.log`。
最终指标：`evaluation/target_results_24.csv`、`method_budget_summary_8.csv`、`paired_deltas_18.csv`、`paired_summary_6.csv`、`RESULT_ANALYSIS.md`。
初始化和训练数据本地保存但不上传GitHub；紧凑配置/哈希/端点证据/分析上传GitHub。

## 三机连接

```powershell
ssh root@121.40.139.213
ssh gaps@192.168.137.172
ssh root@114.55.171.63
```

server Python `/root/gaps_env/bin/python`；Pi C1 `/home/gaps/GAPS/gaps_rpi_env/bin/python`；C2 `/root/gaps_c2_cpu_env/bin/python`。
都使用已有专用PyTorch/Flower环境；不要使用服务器系统Python来判断能否训练。
Pi需保持开机和USB/局域网连接，本机控制器也需保持运行，不能休眠。
本轮新输入根 `/root/GAPS/fedongas27_inputs_20261014`（ECS），`/home/gaps/GAPS/fedongas27_inputs_20261014`（Pi）。
runtime只从冻结git commit生成，不混入当前未提交旧核心文件。

## 局限

仅既定window-level划分，三个seed反映训练随机性；不自动称统计显著。C5都是late，FULL−CLASS是源历史阶段知识组织作用。
不扩展其他target/预算/模型/QC/超参数。零收益、负收益、失败证据都保留。
