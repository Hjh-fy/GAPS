# FedOnGas C5 三种子预算研究 Implementation Plan

**Goal:** 执行用户指定27个训练任务并提供可追溯的固定终点比较。
**Spec:** docs/experiments/fedongas_c5_budget_multiseed_20261014/EXPERIMENT_PLAN.md。
**Execution:** 在既有隔离worktree中逐项执行；用户完整规格已授权设计和正式执行。
**Architecture:** 新增独立protocol、methods、entry和runner；继承现有Flower/A4实现，禁止覆盖dirty核心文件。
**Tech Stack:** Python、PyTorch、Flower、SSH、NumPy/CSV/JSON。

## Tasks

1. Protocol与数据：先写nested/矩阵/同seed初始化测试，验证RED，实现27-row矩阵和输入哈希/无phase CLASS view，验证GREEN。
2. Methods：先写target-only CE全模型梯度/Adam重建、class统计/查询phase不变性测试，验证RED；实现独立目标函数与CLASS adapter，保留FULL基类行为。
3. Entry与commands：测试SRC无DA、CLASS/FULL反馈/损失/seed/输出路径；封装server/client入口和初始化fingerprint日志，冻结配置和代码。
4. Controller：复用既有3机Flower编排和loopback SSH隧道；TO/FT在同server环境执行；每任务独立marker、失败停队列；不覆盖旧结果。
5. Evaluation：固定所有端点后一次性打开test，24-row原始结果、8-rowmean/SD、per-class/confusion、18-rowpaired deltas及6-row均值/SD。
6. Pre-run：新测试、相关A4原实现测试、compileall、runtime/三机预检；提交freeze commit，部署并后台启动27任务队列。
7. Closure：所有端点完成后完整评价/分析/审计，精确提交紧凑证据并push。未完成时保留RUN_PROGRESS和运行日志供继续检查。

## Review focus

- CLASS不得先按phase裁剪/平均再合并；必须直接从同类全部样本统计。
- TO/FT不得调用含source CE的旧run_adaptation，encoder/classifier都必须有CE梯度。
- FULL当前seed不被辅助make_config默认42覆盖；共同初始化以state fingerprint为准。
- 20%不是320的20%；两个预算都用冻结清单，test1360一致。
- 完成marker不得仅凭文件名判定；检查27任务身份、步数、种子、hash及状态，失败不重跑已完成端点。
