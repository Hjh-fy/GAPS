# GAPS 新对话交接（2026-08-24）

> 本文件是新对话的第一入口。先读本文件，再按需读仓库根目录的《代码文件介绍.md》。
>
> 本文件只汇总当前已封存证据，不授权训练、重跑、调参或创建新算法。

## 1. 仓库与 Git 状态

```text
Repository: https://github.com/Hjh-fy/GAPS.git
Worktree:   D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1
Branch:     codex/iotj-final-classification-le1
Authoritative results baseline: 2846c6dcdbe2f2c022535546971a7357b30bc128
Handoff document commit:       use `git log -1 --oneline` after checkout
```

进入正确工作树：

```powershell
Set-Location 'D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1'
git status --short --branch
git log -8 --oneline
```

工作树中长期存在 watcher log 修改、pytest 临时目录及 preprocessing log。它们不是本轮交接资产，不要清理、回滚或提交。正式提交必须精确列出目标文件。

## 2. 当前研究阶段与停止条件

当前阶段是：**canonical-v1 authoritative evidence 已闭环，进入投稿文稿/图表/数字一致性维护阶段**。

当前没有正在运行的正式训练或实验。默认停止：

- 不再进行分类、回归或 QC 算法探索；
- 不重跑 C0、R0、R0-v2、R1、R2、Q0、Q1；
- 不修改 canonical-v1 preprocessing、target split、A4、R84、Ridge alpha、QC 公式或阈值；
- 不根据 target test 重新选择方法、checkpoint、阈值或 stopping point；
- 不删除 historical/superseded/invalid 目录。

若后续需要论文工作，优先级是：

1. 使用 current authoritative summary 更新论文表格、图和 claim–evidence；
2. 对摘要、正文、表格、图注、结论做 number-consistency audit；
3. 只在发现明确的证据缺口且用户重新授权时，才设计新实验。

## 3. 四个冻结科学结论

| 研究侧 | 当前正式结论 | 解释边界 |
|---|---|---|
| Classification | `V1_INTERLEAVED_RETAINED` | 保留每轮 target adaptation 的 A4；不主张任意 final-only adaptation 等价 |
| FedRidge reconstruction | `FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED` | R0-v2 建立数值/代数等价；不等于 bitwise identity 或性能提升 |
| Regression | `CANONICAL_R84_DEVICE_DEPENDENT`；`RETAIN_R84_DEVICE_DEPENDENT` | R84 整体保留，但收益随设备/气体变化；不主张 transfer-safe universal gain |
| QC | `MULTISIGNAL_QC_NOT_ESTABLISHED`；`CONFIDENCE_QC_FINAL` | 最终策略是 confidence-only；regression uncertainty 仅为诊断，不重新选作 final |

最终科学定位应偏向 **lifecycle / commissioning / observability framework**，而不是笼统的“所有算法指标均优于所有基线”。

## 4. canonical-v1 冻结协议

| 项目 | 冻结值 |
|---|---|
| 数据集 | `dataset/iotj_canonical_v1` |
| 数据聚合 SHA256 | `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6` |
| preprocessing | `HZ5_MEAN_W10S` |
| sampling/window | 5 Hz，10 s，50×8，step 5 s |
| source | C1、C2 |
| target | C3、C4、C5 |
| target split | 约 20% calibration / 80% sealed test，身份由 manifest 冻结 |
| seed | seed42（除已注册 multi-seed 研究） |
| classification | A4 interleaved，25 rounds，local epochs=1，batch=32，Adam |
| regression | final `R84_CONCAT`，83D canonical sensor statistics + 1D frozen H1 source prior |
| QC | `r_cls = 1 - max_g p(g|x)`；每个 target 用 calibration-only HC90/HC95 threshold |

数据核验：

```powershell
python tools/verify_iotj_canonical_v1_hashes.py dataset/iotj_canonical_v1
```

## 5. 当前 authoritative 结果

### 5.1 Classification

| Scope | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| C3 | 2677 | 0.998506 | 0.998507 |
| C4 | 1360 | 0.997794 | 0.997794 |
| C5 | 1360 | 0.994118 | 0.994126 |
| pooled | 5397 | 0.997221 | 0.997221 |

主要来源：

```text
results/iotj_canonical_v1_scientific_validation_20260809/classification_comparison/
results/iotj_canonical_v1_final_20260808/classification_evaluation/
```

### 5.2 Final R84 regression（S_ALL）

所有 NRMSE 使用 canonical row-wise registered gas range，由 sealed R1 `y_true/y_pred` 重算。

| Target | RMSE (ppm) | MAE (ppm) | NRMSE_range | R² |
|---|---:|---:|---:|---:|
| C3 | 9.444772 | 4.949892 | 0.052918 | 0.979767 |
| C4 | 13.280947 | 6.006006 | 0.065844 | 0.959952 |
| C5 | 20.611231 | 10.457268 | 0.101923 | 0.903544 |
| pooled micro | 13.990912 | 6.603838 | 0.071409 | 0.955579 |

83D→R84 的 paired grouped bootstrap RMSE 差（R84−83D，5000 replicates，seed42）：

| Scope | ΔRMSE (ppm) | 95% CI |
|---|---:|---|
| C3 | -1.349818 | [-1.956467, -0.849457] |
| C4 | -0.263446 | [-0.396692, -0.121824] |
| C5 | -0.709216 | [-1.861830, 0.675739] |
| pooled | -0.793243 | [-1.241396, -0.316673] |

解释：C3、C4 与 pooled RMSE 支持 R84 改善；C5 CI 跨 0，且不同气体存在正/负迁移，因此正式结论是 `RETAIN_R84_DEVICE_DEPENDENT`，不是 universal superiority。

### 5.3 Final confidence QC

以下 coverage 是 **selective output accepted proportion**，不是 Q1 prediction-interval coverage。

| Scope | HC90 coverage / accepted RMSE / NRMSE | HC95 coverage / accepted RMSE / NRMSE |
|---|---|---|
| C3 | 94.02% / 6.762 / 0.04405 | 96.12% / 6.843 / 0.04405 |
| C4 | 92.94% / 11.242 / 0.05398 | 95.74% / 11.111 / 0.05341 |
| C5 | 90.15% / 14.282 / 0.07648 | 93.38% / 14.678 / 0.07755 |
| pooled micro | 92.77% / 10.239 / 0.05607 | 95.33% / 10.384 / 0.05636 |

NRMSE-AURC：

| Scope | confidence | regression uncertainty | conformal equal-mean |
|---|---:|---:|---:|
| C5 | 0.07935383 | 0.07191279 | 0.09154944 |
| pooled | 0.05681674 | 0.05241224 | 0.06430592 |

较低的 regression-uncertainty AURC 只说明它具有诊断信息；冻结规则不允许在 post-test 阶段把 final policy 改为它。Q1 的 C5 90% nominal prediction interval empirical coverage 为约 92.13%，必须写成 `prediction_interval_empirical_coverage`，不能写成 QC accepted coverage。

## 6. 当前唯一汇总入口

本地生成结果：

```text
results/iotj_canonical_v1_final/current_authoritative_summary_20260812/
```

关键文件：

```text
CURRENT_AUTHORITATIVE_RESULTS_SUMMARY_20260812.md
CURRENT_AUTHORITATIVE_EVIDENCE_INVENTORY.csv
FINAL_R84_REGRESSION_METRICS.csv
FINAL_83D_VS_R84_COMPARISON.csv
HISTORICAL_VS_AUTHORITATIVE_REGRESSION.csv
FINAL_CONFIDENCE_QC_THRESHOLDS.csv
FINAL_CONFIDENCE_QC_METRICS.csv
FINAL_QC_RISK_COVERAGE_SUMMARY.csv
HISTORICAL_QC_EVIDENCE_INDEX.csv
RESULT_PROVENANCE_INDEX.json
sha256_index.json
```

生成器与测试已提交并推送：

```text
scripts/consolidate_current_authoritative_results.py
tests/test_current_authoritative_results_consolidation.py
commit 2846c6dcdbe2f2c022535546971a7357b30bc128
```

注意：仓库 `.gitignore` 忽略 `results/`。因此上述生成结果目前是本地 evidence；GitHub 上可获得生成器、测试、协议文档和已跟踪状态文件，但不能假定所有 formal/result CSV 已推送。新对话需要先检查本地结果根是否存在。

生成器 fail-closed：目标目录已存在时拒绝覆盖。当前目录已经存在，**不要再次运行同名生成命令**。若用户授权生成新版，必须使用新的 versioned output root。

安全验证：

```powershell
python -m pytest -q tests/test_current_authoritative_results_consolidation.py
python -m compileall -q scripts/consolidate_current_authoritative_results.py
Get-Content -Raw 'results/iotj_canonical_v1_final/current_authoritative_summary_20260812/sha256_index.json'
```

## 7. Authoritative、historical、invalid 必须分开

### 7.1 Authoritative

```text
results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/
results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812/
results/iotj_canonical_v1_final/canonical_r2_transfer_safe_v2_20260812/
results/iotj_canonical_v1_final/canonical_q0_qc_necessity_20260812/
results/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812/
```

### 7.2 Historical superseded（保留、可比较、不能作为当前 final claim）

- 旧 A4 regression：C3 9.333、C4 13.808、C5 18.477、pooled 13.314 ppm；
- 旧 multi-signal HC90：coverage 87.87%，RMSE 10.811，NRMSE 0.06765；
- 旧 multi-signal HC95：coverage 94.78%，RMSE 11.107，NRMSE 0.06762。

统一标签：`HISTORICAL_SUPERSEDED`。

### 7.3 Invalid（保留取证、不得引用）

- R2-v1：code identity / OOF leakage 边界不合规；
- Q1-v1：vector interval radius 实现错误并产生 partial root。

统一标签：`INVALID_DO_NOT_CITE`。不得把 invalid 与 historical superseded 合并。

## 8. 主要代码入口

| 文件 | 功能与当前使用边界 |
|---|---|
| `gaps_flower/client_app.py` | Flower source client，本地训练与上传 |
| `gaps_flower/server_app.py` | Flower server、聚合、server adaptation 与 checkpoint |
| `gaps_flower/canonical_quantitative_features.py` | canonical 50×8→83D/104D quantitative feature 与 cache provenance |
| `gaps_flower/canonical_fedridge.py` | 原 R0 实现；保留历史，不由 v2 覆盖 |
| `gaps_flower/canonical_fedridge_v2.py` | R0-v2 stable central moments、FedRidge reconstruction 与 gates |
| `gaps_flower/canonical_r1_v1.py` | R1 83D/R84 metrics、grouped bootstrap 与 decision 原语 |
| `gaps_flower/canonical_r2_transfer.py` | R2 transfer-safe candidate 与 retention decision 原语 |
| `gaps_flower/canonical_qc_evaluation.py` | Q0 risk–coverage、AURC、random reference 原语 |
| `scripts/run_iotj_canonical_fedridge_r0_v2.py` | R0-v2 controller；formal endpoint 已完成，禁止重跑 |
| `scripts/run_iotj_canonical_r1_v1.py` | R1 controller；formal endpoint 已完成，禁止重跑 |
| `scripts/run_iotj_canonical_r2_transfer.py` | R2-v2 controller；formal endpoint已完成，禁止重跑 |
| `scripts/run_iotj_canonical_q0_qc.py` | Q0 controller；formal endpoint 已完成，禁止重跑 |
| `scripts/run_iotj_canonical_regression_reconstruction_q1.py` | Q1-v2 conformal-style diagnostic 与 final QC decision；禁止重跑 |
| `scripts/consolidate_current_authoritative_results.py` | 只读汇总 sealed evidence，不做 checkpoint inference 或训练 |

使用入口前先执行 `python <script> --help`；已完成 experiment ID 的 `run/preflight/audit` 命令只供理解，不构成执行授权。

## 9. 三机拓扑与远程命令

| 角色 | SSH | repo/runtime | 专用 Python |
|---|---|---|---|
| Flower server + server DA | `root@121.40.139.213` | `/root/GAPS` | `/root/gaps_env/bin/python` |
| C1 Pi source client | `gaps@192.168.137.172` | `/home/gaps/GAPS` | `/home/gaps/GAPS/gaps_rpi_env/bin/python` |
| C2 ECS source client | `root@114.55.171.63` | `/root/GAPS` | `/root/gaps_c2_cpu_env/bin/python` |
| Windows | 当前 linked worktree | orchestration/evaluation/audit | 当前项目 Python |

连通性：

```powershell
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@121.40.139.213 "date -u; hostname"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 gaps@192.168.137.172 "date -u; hostname"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@114.55.171.63 "date -u; hostname"
```

专用环境：

```powershell
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@121.40.139.213 "/root/gaps_env/bin/python -c 'import sys,torch,flwr; print(sys.executable); print(torch.__version__); print(flwr.__version__)'"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 gaps@192.168.137.172 "/home/gaps/GAPS/gaps_rpi_env/bin/python -c 'import sys,torch,flwr; print(sys.executable); print(torch.__version__); print(flwr.__version__)'"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@114.55.171.63 "/root/gaps_c2_cpu_env/bin/python -c 'import sys,torch,flwr; print(sys.executable); print(torch.__version__); print(flwr.__version__)'"
```

残留 Flower 进程：

```powershell
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@121.40.139.213 "pgrep -af 'gaps_flower.(server_app|client_app)' || true"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 gaps@192.168.137.172 "pgrep -af 'gaps_flower.(server_app|client_app)' || true"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 root@114.55.171.63 "pgrep -af 'gaps_flower.(server_app|client_app)' || true"
```

2026-08-24 00:57（Asia/Shanghai）只读实测：三台机器均可 SSH 到达，主机名分别为 `iZbp13efwf75yj81x6zuh7Z`、`gaps-pi`、`iZbp12lqy3b1orgiy3x1j0Z`，未发现匹配的 Flower server/client 残留进程。Pi 地址是局域网地址，未来网络变化时可能改变；只能在新 protocol/manifest 中更新，不能修改已冻结证据。

## 10. 本地验证与查看结果命令

```powershell
Set-Location 'D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1'

# Git 与数据
git status --short --branch
git log -8 --oneline
python tools/verify_iotj_canonical_v1_hashes.py dataset/iotj_canonical_v1

# 当前汇总器
python -m pytest -q tests/test_current_authoritative_results_consolidation.py
python -m compileall -q scripts/consolidate_current_authoritative_results.py

# 读取最终表
Import-Csv 'results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_R84_REGRESSION_METRICS.csv' | Where-Object scope -eq 'S_ALL' | Format-Table
Import-Csv 'results/iotj_canonical_v1_final/current_authoritative_summary_20260812/FINAL_CONFIDENCE_QC_METRICS.csv' | Format-Table
Get-Content -Raw 'results/iotj_canonical_v1_final/current_authoritative_summary_20260812/RESULT_PROVENANCE_INDEX.json'
```

不要因为 `.tmp_pytest_*` 很多而递归删除工作树；这些属于既有脏项，除非用户明确授权清理。

## 11. 已知局限与论文建议

- seed42 是绝大多数冻结正式比较的主协议；不能声称充分的跨 seed statistical robustness。
- R84 pooled RMSE 改善受支持，但 C5 RMSE CI 跨 0，且 gas 级存在 negative transfer；必须保留 device-dependent 限定。
- Q0/Q1 表明 regression uncertainty 有排序信息，但 final policy 仍为 confidence-only；不能在看过 test 后改政策。
- C5 是 hardest target，R84 S_ALL RMSE 20.611 ppm；不要用 historical 18.477 ppm 替代 canonical final。
- historical quantitative evidence 的具体 legacy sampling provenance 在部分原始 artifact 中未完全建立；需要写 `PROVENANCE_NOT_ESTABLISHED`，不要凭路径猜测。
- 当前 summary result root 被 `.gitignore` 忽略；新对话必须区分“本地存在”与“GitHub 已跟踪”。
- 下一步更适合做论文 claim–evidence 与数字一致性闭环，不建议再扩算法。

## 12. 给新对话的首条提示词

复制以下内容：

```text
请先只读检查并完整阅读：
1. docs/handoff/CURRENT_HANDOFF_20260824.md
2. 仓库根目录《代码文件介绍.md》
3. results/iotj_canonical_v1_final/current_authoritative_summary_20260812/RESULT_PROVENANCE_INDEX.json
4. 同目录 FINAL_R84_REGRESSION_METRICS.csv、FINAL_83D_VS_R84_COMPARISON.csv、FINAL_CONFIDENCE_QC_METRICS.csv、FINAL_QC_RISK_COVERAGE_SUMMARY.csv

正确工作树：D:\A Python learning\Federated Learning\TRAE SOLO\.worktrees\iotj-final-classification-le1
分支：codex/iotj-final-classification-le1
当前 authoritative results 基线提交：2846c6dcdbe2f2c022535546971a7357b30bc128

当前冻结结论：
- Classification: V1_INTERLEAVED_RETAINED
- FedRidge: FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED
- Regression: CANONICAL_R84_DEVICE_DEPENDENT / RETAIN_R84_DEVICE_DEPENDENT
- QC: MULTISIGNAL_QC_NOT_ESTABLISHED / CONFIDENCE_QC_FINAL

当前没有正在运行的正式实验。不要重跑、训练、调参、修改 canonical-v1/A4/R84/QC，也不要删除历史结果。
先核对 Git、canonical dataset hash、本地 authoritative summary 是否存在、三机 SSH 连通性和残留进程；只汇报当前状态与论文证据缺口。未经我明确授权，不启动任何 formal endpoint 或新算法研究。
```

## 13. 交接协议

```text
from_skill: gaps-research-orchestrator + experiment-registry + claim-evidence + number-consistency-audit
to_skill: number-consistency-audit（优先）；随后 research-writing-skill
input_version: commit 2846c6d + current_authoritative_summary_20260812
completed_checks: authoritative/historical/invalid 分层；核心数值与 source path；Git/remote；三机只读连通性
largest_evidence_gap: 论文正文、表格、图注与当前 authoritative values 的一致性尚未在本轮完成
requested_next_action: 先做 manuscript number/claim consistency；不要启动实验
read_only: dataset/iotj_canonical_v1、全部 formal result roots、checkpoints、protocol manifests、historical/invalid roots
```
