# B5 multi-seed M0 protocol audit

## Audit scope and intended claim

审计 seed42 是否能作为 seeds43–46 的唯一正式协议基线，并确认后续五种子
分类/回归比较只改变 classifier training seed。

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | QC | Seeds | Provenance |
|---|---|---|---|---|---|---|---|---|
| B5-MS-CLS-S42 | frozen C1/C2→C5 | B5 proto_replay | canonical round25 adapted | full corrected B5 server DA | C5 calibration only | off | 42 | canonical attempt audit valid |
| B5-MS-CLS-S43–46 | same frozen split | same B5 | pending | identical server DA | identical | off | 43–46 | frozen command manifests; pending real runs |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| M0-01 | informational | seed42 checkpoint | path、SHA、valid attempt audit | establishes baseline | preserve read-only | resolved |
| M0-02 | informational | numerical protocol | normalized B5 command manifests | only seed and derived identity fields vary | bind hashes in manifest | resolved |
| M0-03 | major | actual topology | seed42 execution topology + launch manifest | historical `client_c2_pc` label could mislead | force ECS-C2 controller overrides | resolved in plan |
| M0-04 | informational | H1 seed coupling | H1 reads fixed C1/C2 train/calibration only | source H1 need not retrain | freeze audited fed-H1 hash | resolved |
| M0-05 | informational | H2/H3 seed coupling | source-head fit reads fixed C1/C2 data; serialized R4 policy | source H2/H3 need not retrain | freeze R4 policy hash | resolved |
| M0-06 | blocking-at-launch | three-host readiness | no live preflight in M0 planning | cannot launch until checked | run fail-closed preflight per seed | open |
| M0-07 | informational | existing seeds43–46 inventory | user confirmed absent | no reusable checkpoint search | train four new seeds | resolved |

## Leakage assessment

- C5 test 只用于 M2/M3 evaluation；
- C5 test 不参与 checkpoint、seed、target Ridge alpha 或 refit；
- source H1/H2/H3 不读取 C5 test；
- M4 只选择 regression prior 表达，不重新选择 classifier。

## Baseline, completeness, and reproducibility assessment

seed42 的 checkpoint、training commit、source archive、dataset manifest、
command manifest、真实 topology、25-round attempt audit 和实测时间完整。
seeds43–46 的冻结 commands 已存在且算法字段可复现。正式完整性仍取决于四个
新 attempt 均通过 validator。

## Verdict: approved for preflight; training not yet launched

M0 协议审计通过，可以进入三机 preflight。当前没有授权或执行 M1 长训练。
只读 M0 validator 返回 `ready_for_preflight`、0 errors；M0、protocol、
controller、validator 与 cloud-edge 相关测试最终为 `230 passed, 2 skipped`。
首次使用较长 pytest basetemp 时出现 11 个 Windows path-length failures；改用
短 basetemp 后同一测试集全部通过，故这些失败不属于训练协议或代码缺陷。

## Unknowns and handoff

- `unknown`：启动时三机在线/资源/残留进程状态；
- handoff：M1 controller 必须逐 seed 运行预声明命令，并在任一失败时停止；
- read-only：seed42 attempt、checkpoint、runtime v4、HC95/HC90、fed-H1
  正式审计结果、R4 policy。
