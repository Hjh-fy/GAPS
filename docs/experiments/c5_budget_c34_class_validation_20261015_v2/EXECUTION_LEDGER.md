# Execution ledger — plan: docs/experiments/c5_budget_c34_class_validation_20261015_v2/EXPERIMENT_PLAN.md

授权：用户明确运行C3/C4 TO/FT/CLASS seed42，CLASS主候选；并保留C5 10/15%回归补齐。
Ruling: v1 draft缺FT，v2按最新用户要求新增FT-C3/FT-C4；不改任何方法或超参数，FT共享已有匹配SRC42，成本仅2个目标2500-step训练，不新增源训练。
Pre-flight interfaces: 预算原数组→Ridge真实浓度，不能读分类零占位；SRC42→FT精确checkpoint SHA；class-only source/target view→原4prototype worker；newtarget reader→C3实际678行，不能沿用C5的80/320硬限制。
Task1: 三机SSH/既有venv只读核验PASS；均无活动Flower。torch版本server2.12.0+cu130（无CUDA）、Pi2.12.1CPU、C2 2.12.0CPU；使用原环境不换优化器或协议。
Task2: 新矩阵/真实目标路径/字段隔离/嵌套预算/reuse gate/test gate/初始化/配对差值测试经RED→GREEN。新9项及相关原有测试合计43 passed（18.17s）；全仓测试另行记录，不等同全仓通过。
Task3: pre-run freeze、4bank、6分类队列尚待启动；完成项依持久marker与Git记录，不凭聊天记忆重跑。
旧3个核心文件与watcher日志均为用户已有未提交修改，不修改或纳入归档。
Pre-run review: 独立只读复核无Critical/Important问题，确认可冻结启动。Minor：最终证据封存须另外包含class_round_curves目录，evaluation目录自身index不覆盖该兄弟目录；不影响首轮训练/评价。完成时必须核验曲线资产，不以evaluation的复用分支代替全证据核验。
