# 实际运行、恢复与来源

实际入口：本阶段run_phase15.py；公共71D、Ridge/SVR由phase15_common.py只读导入Phase1 phase1_common.py及v5原函数，未复制改造算法。新增仅源episode、调度、中心化KRR及审计/诊断。输入为Phase1 run_v2/data的NPZ和metadata、Phase0.5角色/质量掩码和特征合同；原始640文件只读校验。

环境Python3.12.4/numpy1.26.4/scikit-learn1.8.0/scipy1.13.1/joblib1.4.2/threadpoolctl，Windows CPU，threadpool_limits(1)，不需要GPU/Flower。确切环境和所有执行前脚本SHA见run/PRE_RUN_CONTRACT.json。数学参考文档可能跟随stable版本更新，实际可复现数值依据是冻结代码和本地包版本。

本次依次实际执行以下命令（仓库根目录，设置PYTHONDONTWRITEBYTECODE=1及PYTHONIOENCODING=utf-8）：

```powershell
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/run_phase15.py prepare
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/run_phase15.py smoke
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/run_phase15.py task_a
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/run_phase15.py task_c
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/run_phase15.py evaluate
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/analyze_phase15.py
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/verify_phase15.py
PYTHON_EXECUTABLE research/fedongas_phase15_20261102/write_reports.py
```

四个unittest在本阶段目录运行`python -m unittest test_phase15 -v`，覆盖独立暴露预算/角色、归一化规则、中心化KRR单窗/批量一致性及RBF解析梯度。首次实现前因缺少模块而红，完成后四项通过；smoke另做4次合成fit及保存往返、线性核等价，未增加数据拟合。

上述制品拒绝覆盖；不能直接在已有run/重复执行整套命令。若需要独立重现，应将本阶段脚本和EXPERIMENT_PLAN.md原样复制到新的research同级独立目录，以保持P1路径关系，在新的run/运行；先核验SHA和依赖，不复制/覆盖旧run。该完整训练重现本轮未额外执行，以遵守拟合资源上限。

已执行的恢复证据：joblib.load全部120新头，使用原71D缓存完整回放25200逐窗原始预测，与保存值逐条一致（容差1e-10ppm）；每头首末窗单独预测与批量一致，最大8.53e-14ppm。60套主/辅助公式独立重算。原始640文件、Phase1清单125文件、v5使用代码和独立ZIP逐一比对。没有重训恢复、ZIP写入或原资产移动。

run/ACCESS_EVENTS.jsonl记录prepare→smoke→A参数锁→A模型锁→KRR参数锁→全部模型锁→本轮test特征释放→评价→分析；锁JSON关联SHA。历史test早已观察过，锁序列仅证明本轮流程，不构成全历史的盲测证据。代码原封不动的入口/特征/角色/模型/参数/结果哈希见PRE_RUN_CONTRACT、MODEL_INDEX、EVALUATION_COMPLETE及总ARTIFACT_SHA256.csv。

公开审阅副本剔除信号、71D数值、模型权重/训练scaler/ZIP，仅发布来源SHA、紧凑指标、锁定记录、参数、诊断汇总和脚本文本；路径别名与UTF8-LF转换后的SHA由PUBLICATION_MANIFEST另记录，不能混同原文件SHA。完整本地模型和输入依赖不随GitHub报告分发；外部agent可审阅，若没有原始本地资产不能仅凭报告重新训练。
