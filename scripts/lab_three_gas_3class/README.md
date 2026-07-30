# 实验室三气体分类数据管线

本目录只放新代码，不修改 `Dataset_self` 中的原始 CSV 和师兄脚本。

## 当前任务

- 三分类：乙醛 `0`、甲烷 `1`、乙酸 `2`
- 三个客户端：Platform1、Platform2、Platform3
- 通道：CH1、CH2、CH4、CH6、CH8、CH9
- 不做浓度回归
- 不划分 early/middle/late
- 双戊烯不参与三分类
- 每个 v1/v2 的第一个不稳定暴露不进入模型

## 构建数据

在项目根目录运行：

```powershell
python scripts/lab_three_gas_3class/build_fivefold_dataset.py
```

默认输出：

```text
dataset/client_data_lab_3gas_5fold_nominal_v1/
```

默认使用名义时间表：

```text
1800 s 空气
1200 s 目标气体
1800 s 恢复
重复 6 次
```

输出中的 `boundary_manifest.csv` 包含全部 18 个正式 Session、每个
Session 的 6 次暴露边界。获得精确阀门时间后，复制并编辑该文件，再重建：

```powershell
python scripts/lab_three_gas_3class/build_fivefold_dataset.py `
  --boundaries-csv path/to/edited_boundaries.csv `
  --output-root dataset/client_data_lab_3gas_5fold_exact_v1
```

边界文件至少需要以下列：

```text
session_id,exposure_index,gas_start_s,gas_end_s
```

## 划分

每个 `平台 × 气体 × v1/v2` 有 5 个有效 Exposure。相同保留序号的
v1/v2 Exposure 构成一个 Fold Group。

每一折、每个客户端包含：

- 训练：18 Exposure
- 验证：6 Exposure
- 测试：6 Exposure

现有 GAPS 代码使用 `calibration_*` 文件名，因此本数据中的验证集保存为
`calibration_*`；它的科学用途仍然是验证集，不是目标域浓度校准集。

窗口来自同一次 Exposure 时始终属于同一 Split，禁止随机窗口划分。

## 特征

默认处理为：

1. 根据 Time 列切割；
2. 使用通气前 300 s 中位数作为局部空气基线；
3. 计算相对变化 `(R - R0) / |R0|`；
4. 重采样到 1 Hz；
5. 5 s 移动平均；
6. 100点窗口、50点步长；
7. 仅由当前 Fold 全部客户端的训练集拟合逐通道 Z-score；
8. 同一统计量应用于验证和测试。

输出特征已经完成 Z-score。现有加载器应使用 `normalize=False`，避免重复归一化。

## 验证

```powershell
python scripts/lab_three_gas_3class/validate_fivefold_dataset.py
```

验证内容包括：

- 形状必须为 `(N, 100, 6)`；
- 标签只能为 `0/1/2`；
- 每个客户端每折为 `18/6/6` 个 Exposure；
- Exposure 不得跨 Split 泄漏；
- 三类 Exposure 数量平衡；
- 训练集 Z-score 均值约为0、标准差约为1；
- 数据中没有 NaN/Inf。

## 中心化基线

先用中心化训练检查切割、标签和模型接口：

```powershell
python scripts/lab_three_gas_3class/train_centralized_baseline.py `
  --fold 1 --epochs 20 --device auto
```

该脚本使用现有 GAPS TCN 分类骨干，但显式设置：

```text
NUM_CLASSES=3
INPUT_DIM=6
NUM_CLIENTS=3
NUM_PHASES=1
USE_REG_LOSS=False
```

保存窗口级和 Exposure 级 Accuracy、Macro-F1、混淆矩阵。Exposure 级结果
通过同一次暴露全部窗口的平均概率得到，应作为主要结果。

完成五折后可以汇总：

```powershell
python scripts/lab_three_gas_3class/summarize_fivefold_results.py `
  --results-root results/lab_three_gas_centralized_smoke
```

汇总文件会明确标记为 `smoke_baseline_only`，防止将名义边界、单随机种子和
短训练结果误当成正式联邦实验结论。

## 接入 Flower

服务器和三个客户端必须使用完全相同的结构参数：

```text
--num-classes 3 --input-dim 6 --num-clients 3 --num-phases 1
```

第一阶段只运行 `--strategy fedavg --profile smoke`，即纯交叉熵分类。当前数据
没有浓度回归标签，也没有 early/middle/late 阶段，不应先启用回归或阶段域适配。

服务器示例：

```powershell
python -m gaps_flower.server_app `
  --server-address 0.0.0.0:8080 `
  --rounds 10 --min-clients 3 `
  --strategy fedavg --profile smoke `
  --num-classes 3 --input-dim 6 --num-clients 3 --num-phases 1 `
  --output-dir results/lab_3gas_flower_fold1
```

客户端示例（分别把 `--client-id` 改为1、2、3）：

```powershell
python -m gaps_flower.client_app `
  --server-address 127.0.0.1:8080 `
  --client-id 1 `
  --data-root dataset/client_data_lab_3gas_5fold_nominal_v1/fold_1 `
  --profile smoke --local-epochs 5 --batch-size 32 `
  --num-classes 3 --input-dim 6 --num-clients 3 --num-phases 1
```

检查点会保存上述结构参数。评估程序优先从新检查点读取这些参数，也允许使用
相同的命令行参数显式覆盖。

本机一轮端到端烟雾测试可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts/lab_three_gas_3class/run_flower_smoke.ps1 -Fold 1
```

脚本会在后台启动一个服务器和三个客户端，完成后检查进程退出码和
`server_latest.pth`。日志与检查点写入
`results/lab_3gas_flower_smoke_fold1/`。

Flower检查点的窗口级和Exposure级评估：

```powershell
python scripts/lab_three_gas_3class/evaluate_exposure_checkpoint.py `
  --checkpoint results/lab_3gas_flower_smoke_fold1/server_latest.pth `
  --data-root dataset/client_data_lab_3gas_5fold_nominal_v1/fold_1 `
  --client-ids 1,2,3 --split test `
  --output results/lab_3gas_flower_smoke_fold1/exposure_evaluation.json
```

## 单一兼容 Phase

生成的 `phase_labels` 全为0，含义是 `whole_target_gas_exposure`。它只是为了
兼容现有四元组数据接口，并不代表早期、中期或晚期划分。
