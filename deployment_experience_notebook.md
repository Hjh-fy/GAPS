# 联邦部署经验笔记

更新时间: 2026-06-03

本笔记用于记录阿里云服务端、本地电脑模拟边缘客户端、后续树莓派迁移相关的实验过程、终端证据、代码问题、修复结论和下一步规划。每次实验建议按以下固定字段追加，便于论文方法描述、实验复盘和部署回放。

## 2026-06-03 阶段4 云边联邦分类通信验证

### 实验目标

验证阿里云 ECS 作为 Flower 服务端、本地电脑模拟源域客户端 `client_1` 和 `client_2` 时，是否能完成真实网络通信下的联邦分类训练闭环。

### 运行配置

- 服务端: 阿里云 `121.40.139.213:8080`
- 客户端: 本地电脑两个 PowerShell 进程，分别模拟 `client_id=1` 和 `client_id=2`
- 数据集: `dataset/client_data_federated_window_fullgrid_src12_tgt345`
- 训练轮数: `--rounds 2`
- 本地训练: `--local-epochs 1`, `--batch-size 32`, `--device cpu`
- 策略: `--strategy gaps`

### 关键命令

```bash
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.server_app --server-address 0.0.0.0:8080 --rounds 2 --min-clients 2 --strategy gaps --use-selective-agg false --use-proto-mmd false --output-dir results/flower_cloud_src12_smoke --run-name aliyun_src12_8080_smoke
```

### 结果证据

- 两轮训练均显示 `aggregate_fit: received 2 results and 0 failures`
- 两轮评估均显示 `aggregate_evaluate: received 2 results and 0 failures`
- 服务端生成:
  - `server_round_001.pth`
  - `server_round_002.pth`
  - `server_latest.pth`
  - `client_stats_round_001/002.json`
  - `prototype_stats_round_001/002.json`
  - `semantic_protos_round_001/002.json`

### 结论

阶段4云边联邦分类通信闭环已跑通。该流程保留单机模拟中的模型、数据划分和本地训练逻辑，但将原进程内模拟通信替换为 Flower gRPC 网络通信。服务端不直接访问客户端训练集，客户端本地训练后上传参数和统计信息，更接近真实联邦学习部署。

## 2026-06-03 阶段4.4 服务端域适应触发与修复

### 实验目标

在 GAPS 联邦聚合后，让阿里云服务端读取已上传的源域/目标域校准数据，执行服务端域适应，并保存 adapted checkpoint 与诊断日志。

### 数据上传状态

阿里云已存在完整数据集:

```text
dataset/client_data_federated_window_fullgrid_src12_tgt345/
  client_1/
  client_2/
  client_3/
  client_4/
  client_5/
  calibration/
  global_test/
  norm_stats.npz
  split_info.json
```

确认命令:

```bash
cd ~/GAPS && find dataset/client_data_federated_window_fullgrid_src12_tgt345 -maxdepth 2 -name "calibration_features.npy" -o -name "train_features.npy"
```

### 关键命令

```bash
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.server_app --server-address 0.0.0.0:8080 --rounds 2 --min-clients 2 --strategy gaps --use-domain-adapt true --server-val-data "dataset/client_data_federated_window_fullgrid_src12_tgt345/client_1,dataset/client_data_federated_window_fullgrid_src12_tgt345/client_2" --server-calib-data "dataset/client_data_federated_window_fullgrid_src12_tgt345/client_3" --domain-adapt-warmup 0 --domain-adapt-steps 10 --da-device cpu --output-dir results/flower_cloud_src12_da_smoke --run-name aliyun_src12_da_smoke
```

### 发现的问题

1. `--strategy gaps` 初始没有触发服务端域适应。
   - 原因: `_run_domain_adapt()` 写在 `CheckpointFedAvg.aggregate_fit()` 中，而实际运行路径是 `GapsStrategy.aggregate_fit()`。
   - 修复: 将阶段4.4域适应调用接入 `GapsStrategy.aggregate_fit()` 保存聚合 checkpoint 之后。

2. class-wise MMD 出现 mask 维度错误。
   - 报错: `mask [32,4]` 索引 `feat [32,64]`。
   - 原因: 条件 MMD 需要一维类别标签 `y: [B]`，但某处传入了二维标签矩阵。
   - 修复: 在生成 `src_mask/tgt_mask` 前，将二维标签通过 `argmax(dim=1)` 转为 class id。

3. 热修过程中出现 Python 缩进错误。
   - 报错: `IndentationError: expected an indented block after 'for' statement`
   - 修复: 恢复 class-wise MMD 循环体结构:

```text
for c in range(num_classes):
    y_s_ids = ...
    y_t_ids = ...
    src_mask = ...
    tgt_mask = ...
    if src_mask.sum() > 1 and tgt_mask.sum() > 1:
        c_mmd += ...
        class_count += 1
```

### 结果证据

编译检查通过:

```bash
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m py_compile gaps_flower/domain_adaptation.py gaps_flower/strategy.py
```

域适应产物已生成:

```text
domain_adapt_latest.json
domain_adapt_round_001.json
domain_adapt_round_002.json
```

`history.json` 已记录 adapted checkpoint:

```text
domain_adapt: results/flower_cloud_src12_da_smoke/server_round_001_adapted.pth
domain_adapt: results/flower_cloud_src12_da_smoke/server_round_002_adapted.pth
```

`domain_adapt_latest.json`:

```json
{
  "val_loss": 139.24795532226562,
  "coral_loss": 1.0670629535525222e-06,
  "mmd_global": 0.001360361697152257,
  "mmd_class": 0.00034009042428806424,
  "adv_loss": 0.0,
  "proto_anchor": 464.15704345703125,
  "total_loss": 139.24795532226562,
  "num_steps": 10
}
```

### 技术解释

服务端域适应当前使用源域校准 split 与目标域 calibration split。设源域特征为 `F_s in R^{B_s x d}`，目标域特征为 `F_t in R^{B_t x d}`，当前分类特征维度约为 `d=64`。全局 MMD 对齐目标是:

```text
MMD^2(P_s, P_t) = || E[phi(F_s)] - E[phi(F_t)] ||_H^2
```

class-wise MMD 进一步在标签空间一致的前提下，对每一类气体计算:

```text
L_CMMD = (1 / C) * sum_c MMD^2(F_s^c, F_t^c)
```

因此条件 MMD 的前提是源域和目标域标签空间一致，且 `y_s/y_t` 必须是一维类别标签。如果传入 `[B,4]` 的回归标签或 one-hot 标签，必须先转为 `[B]` 类别 ID，否则 mask 维度不匹配。

CORAL 对齐二阶统计量:

```text
L_CORAL = || Cov(F_s) - Cov(F_t) ||_F^2 / (4 d^2)
```

其假设是源/目标域主要差异可由特征均值和协方差漂移刻画。对于气体传感器漂移，这对应传感器响应幅值、基线和通道相关性发生缓慢变化的物理假设。

### 需要继续确认

`total_loss == val_loss`，而 `coral_loss/mmd/proto_anchor` 只出现在日志中。这可能意味着:

- 域适应损失权重当前为 0，组件只作为诊断记录；
- 或者 `total_loss` 保存的是主任务 loss，没有记录加权后的实际优化目标；
- 或者 alignment/proto loss 在代码中计算但未加入反向传播。

下一步必须审计 `domain_adaptation.py` 中 `total_loss` 的构造:

```text
total_loss = val_loss + lambda_coral * coral_loss + lambda_mmd * mmd_loss + lambda_proto * proto_anchor + lambda_adv * adv_loss
```

并记录各 `lambda` 的实际值。否则不能声称 CORAL/MMD/prototype anchor 已经参与优化，只能说它们已被计算和记录。

## 后续规划

1. 审计 `domain_adaptation.py` 的损失权重和反向传播路径，确认 CORAL/MMD/prototype anchor 是否真正参与模型更新。
2. 将域适应诊断 JSON 扩展为包含 `lambda_*`、`weighted_*_loss`、`grad_norm`、`nan_flag`。
3. 为 `domain_adaptation.py` 添加最小单元测试:
   - `y: [B]` 标签输入
   - `y: [B, C]` 标签输入
   - 某类别在源域或目标域样本数不足时跳过 class-wise MMD
4. 固化云端运行脚本，区分:
   - `src12_tgt3`
   - `src12_tgt345`
   - `src45_tgt123`
5. 树莓派购买前，继续用本地电脑模拟多客户端；树莓派迁移主要验证 ARM 环境、CPU batch size、实时窗口数据适配。

## 2026-06-03 下一阶段验证规划

### 当前判断

在推进回归模型 B、目标端回归校准和 QC 层之前，优先验证完整分类任务的联邦域适应是否在目标域有效。原因是当前部署链路中分类模型 A 是回归路由器，若目标域分类路由不稳定，后续回归头选择、响应锚定和 QC 风险解释都会被上游分类错误污染。

### 必做验证

1. 评估聚合 checkpoint 与 domain-adapted checkpoint 在目标域 `client_3,client_4,client_5` 的 test split 准确率。
2. 同时保留源域 `client_1,client_2` 的 test split 准确率，检测域适应是否牺牲源域性能。
3. 对比 `server_round_002.pth` 与 `server_round_002_adapted.pth`:
   - 若目标域准确率提升且源域下降可控，则阶段4.4有效。
   - 若目标域无提升或下降，则先审计 domain adaptation loss 权重和 checkpoint 是否真的采用 adapted state。
4. 分类校准先不直接启用训练式校准，先评估置信度校准指标:
   - Accuracy
   - NLL
   - ECE
   - confidence margin 分布
   - confusion matrix

### 需要的小代码改动

`gaps_flower/evaluate_checkpoint.py` 当前使用 `create_client_full_test_loader()`，会合并 train/test/calibration。用于通信烟测可以，但不适合作为目标域分类准确率报告。下一步应增加:

```text
--split test|calibration|full
--metrics accuracy,nll,ece,confusion
```

默认使用 `--split test`，避免 calibration split 同时参与域适应和最终评估。

### 分类是否需要校准

分类校准不应先验默认加入训练链路。建议采用两阶段判断:

1. 若目标域 `accuracy` 已高，但 `ECE/NLL` 差，说明类别判别对但置信度不可信。优先做 temperature scaling，因为它只学习一个温度参数，不改变 argmax 路由。
2. 若目标域 per-class accuracy 差，尤其 CO/Ethylene 混淆明显，则 temperature scaling 无法修复类别边界，应回到域适应或少样本分类头校准。

温度缩放公式:

```text
p_k = softmax(z_k / T)
```

其中 `T > 1` 会降低过度自信，通常不改变分类 argmax；因此适合作为部署置信度校准和 QC 风险分数校准，而不是分类精度修复器。

### 进入回归阶段的门槛

满足以下条件后再推进回归:

- `client_3,4,5` test accuracy 达到或接近单机模拟分类水平。
- adapted checkpoint 不劣于 plain aggregation checkpoint。
- 混淆矩阵中没有系统性错误路由，例如某一气体大面积被错分到另一气体。
- calibration split 不被混入最终 test 评估。

### 推荐实验顺序

1. 修 `evaluate_checkpoint.py` 的 split 和 ECE/NLL 指标。
2. 在阿里云或本地离线评估:
   - `server_round_002.pth`
   - `server_round_002_adapted.pth`
   - `server_latest.pth`
3. 对目标域 `3,4,5` 做 per-client/per-class 报告。
4. 根据分类结果决定:
   - 分类稳定: 进入回归模型 B 与目标端校准。
   - 分类准确但置信度不稳: 加 temperature scaling，不改 argmax。
   - 分类本身不稳: 回到域适应 loss 权重、目标校准数据来源和选择性聚合策略。

## 参考基础

- MMD: Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
- CORAL: Sun and Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV Workshops 2016.
- Domain adversarial training: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.

## 2026-06-03 分类 Flower + DA 复现前代码审计

### 本轮修正

1. `gaps_flower/domain_adaptation.py`
   - 修正 DA 监督分类损失的标签索引: `GasSensorWindowDataset` 输出为 `(x, y_cls, y_reg, phase)`，因此分类标签必须取 `batch[1]`。之前误用 `batch[2]`，实际是 4 维回归浓度标签，会导致 `CrossEntropyLoss` 和 class-wise MMD 的数学对象错误。
   - class-wise MMD mask 统一转换为 `(B,)` class id，避免 one-hot/probability label 触发二维 mask。
   - class-wise MMD 按实际参与的类别数 `class_count` 平均，不再固定除以 `NUM_CLASSES`。
   - prototype anchor 从固定 `"cls,0"` 改为同一 class 下所有 phase semantic prototype 的均值，适配当前 server calibration loader 没有 phase 标签的事实。
   - DA 诊断 JSON 增加 `weighted_*` 和 `lambda_*`，用于确认 CORAL/MMD/prototype anchor/adv 是否真实进入 `total_loss`。

2. `gaps_flower/task.py`
   - Flower 客户端训练统一使用 `normalize=False`，与单机已验证主线保持同一输入尺度。
   - Flower 客户端 evaluate 改为 `create_client_test_only_loader()`，不再把 train/calibration 混入评估。

3. `gaps_flower/evaluate_checkpoint.py`
   - 默认 `--split test`，严格读取 `test_features.npy`。
   - 增加 `--split calibration|full` 显式开关。
   - 输出 `accuracy / macro_accuracy / NLL / ECE / mean_confidence / mean_margin / per_class_accuracy / confusion_matrix`。

### 本地检查结果

```text
python -m py_compile gaps_flower/domain_adaptation.py gaps_flower/strategy.py gaps_flower/task.py gaps_flower/evaluate_checkpoint.py gaps_flower/client_app.py gaps_flower/server_app.py
```

通过。

```text
client_1 train split: 2360 samples
client_3 test split: 680 samples
test batch: x=(32,100,8), y_cls=(32,), y_reg=(32,4), phase=(32,)
class id range: 0..3
```

### 云端复现实验判据

1. 先做 2 轮 smoke，确认阿里云 server 与本地 client 1/2 通讯、聚合、DA checkpoint、评估脚本全部连通。
2. 再做 10 轮正式分类链路验证，使用 source `client_1,client_2` 训练，target `client_3,client_4,client_5` 的 calibration split 做服务端 DA，target test split 做最终评估。
3. 必须同时报告 plain checkpoint 和 adapted checkpoint:
   - `server_latest.pth`
   - `server_latest_adapted.pth`
4. 通过门槛:
   - target `client_3,4,5` test accuracy 接近单机分类水平。
   - adapted 不显著劣化 source `client_1,2` test accuracy。
   - confusion matrix 中没有系统性路由错误。
   - ECE/NLL 只作为是否需要 temperature scaling 的依据，不作为 argmax accuracy 的替代。

### 推荐一行命令

本地同步代码到阿里云:

```text
cd "D:\A Python learning\Federated Learning\TRAE SOLO"; ssh root@121.40.139.213 "mkdir -p ~/GAPS/gaps_flower_upload_tmp"; scp .\federated_dataset.py .\gaps_flower\domain_adaptation.py .\gaps_flower\task.py .\gaps_flower\evaluate_checkpoint.py .\gaps_flower\strategy.py .\gaps_flower\server_app.py .\gaps_flower\client_app.py root@121.40.139.213:~/GAPS/gaps_flower_upload_tmp/
```

阿里云移动同步文件:

```text
cd ~/GAPS && mkdir -p gaps_flower_upload_tmp && cp gaps_flower_upload_tmp/federated_dataset.py ./federated_dataset.py && cp gaps_flower_upload_tmp/domain_adaptation.py gaps_flower/domain_adaptation.py && cp gaps_flower_upload_tmp/task.py gaps_flower/task.py && cp gaps_flower_upload_tmp/evaluate_checkpoint.py gaps_flower/evaluate_checkpoint.py && cp gaps_flower_upload_tmp/strategy.py gaps_flower/strategy.py && cp gaps_flower_upload_tmp/server_app.py gaps_flower/server_app.py && cp gaps_flower_upload_tmp/client_app.py gaps_flower/client_app.py
```

阿里云语法检查:

```text
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m py_compile gaps_flower/domain_adaptation.py gaps_flower/strategy.py gaps_flower/task.py gaps_flower/evaluate_checkpoint.py gaps_flower/client_app.py gaps_flower/server_app.py
```

阿里云启动正式 server:

```text
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.server_app --server-address 0.0.0.0:8080 --rounds 10 --min-clients 2 --strategy gaps --use-selective-agg true --selective-warmup 3 --use-proto-mmd true --use-domain-adapt true --server-val-data "dataset/client_data_federated_window_fullgrid_src12_tgt345/client_1,dataset/client_data_federated_window_fullgrid_src12_tgt345/client_2" --server-calib-data "dataset/client_data_federated_window_fullgrid_src12_tgt345/client_3,dataset/client_data_federated_window_fullgrid_src12_tgt345/client_4,dataset/client_data_federated_window_fullgrid_src12_tgt345/client_5" --domain-adapt-warmup 3 --domain-adapt-steps 30 --da-use-coral true --da-use-mmd true --da-use-adversarial false --da-device cpu --output-dir results/flower_cloud_src12_tgt345_cls_da_v5 --run-name aliyun_src12_tgt345_cls_da_v5
```

本地客户端 1:

```text
cd "D:\A Python learning\Federated Learning\TRAE SOLO"; python -m gaps_flower.client_app --server-address 121.40.139.213:8080 --client-id 1 --data-root dataset/client_data_federated_window_fullgrid_src12_tgt345 --device cpu --local-epochs 5 --batch-size 32
```

本地客户端 2:

```text
cd "D:\A Python learning\Federated Learning\TRAE SOLO"; python -m gaps_flower.client_app --server-address 121.40.139.213:8080 --client-id 2 --data-root dataset/client_data_federated_window_fullgrid_src12_tgt345 --device cpu --local-epochs 5 --batch-size 32
```

阿里云评估 target plain:

```text
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.evaluate_checkpoint --checkpoint results/flower_cloud_src12_tgt345_cls_da_v5/server_latest.pth --data-root dataset/client_data_federated_window_fullgrid_src12_tgt345 --client-ids 3,4,5 --split test --device cpu --batch-size 64 --output results/flower_cloud_src12_tgt345_cls_da_v5/eval_target345_plain_test.json
```

阿里云评估 target adapted:

```text
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.evaluate_checkpoint --checkpoint results/flower_cloud_src12_tgt345_cls_da_v5/server_latest_adapted.pth --data-root dataset/client_data_federated_window_fullgrid_src12_tgt345 --client-ids 3,4,5 --split test --device cpu --batch-size 64 --output results/flower_cloud_src12_tgt345_cls_da_v5/eval_target345_adapted_test.json
```

阿里云评估 source adapted:

```text
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.evaluate_checkpoint --checkpoint results/flower_cloud_src12_tgt345_cls_da_v5/server_latest_adapted.pth --data-root dataset/client_data_federated_window_fullgrid_src12_tgt345 --client-ids 1,2 --split test --device cpu --batch-size 64 --output results/flower_cloud_src12_tgt345_cls_da_v5/eval_source12_adapted_test.json
```

## 2026-06-03 src12 -> target345 分类 Flower 结果复盘

### 已观察到的结果

`results/flower_cloud_src12_tgt345_cls_da_v5/server_latest.pth` 和
`server_latest_adapted.pth` 在 target test split 上输出完全一致:

```text
target clients 3,4,5 weighted_accuracy = 0.7901515
weighted_nll = 1.8898403
weighted_ece = 0.2033591
```

分客户端:

```text
client_3 accuracy = 0.9750, ECE = 0.0229
client_4 accuracy = 0.8750, ECE = 0.1182
client_5 accuracy = 0.3125, ECE = 0.6720
```

source adapted:

```text
client_1 accuracy = 0.9941
client_2 accuracy = 0.9985
source weighted_accuracy = 0.9963
```

### 判断

1. Flower 通讯、10 轮分类训练、checkpoint 保存、test split 评估链路已经连通。
2. source clients 学得很好，说明本地客户端训练和云端聚合基本有效。
3. target 端没有达到之前单机 `src12_tgt345` 约 `0.984` 的分类水平，主要被 `client_5` 拉低。
4. plain 与 adapted 指标完全一致，必须优先确认 `server_latest_adapted.pth` 权重是否真的不同于 `server_latest.pth`。如果权重完全相同，说明 DA 没有实际更新 checkpoint；如果权重不同但指标完全相同，说明当前 DA 步长/目标不足以改变分类边界。
5. `client_5` 是系统性错分，不是简单置信度校准问题:
   - true class 0 大量被预测为 class 2
   - true class 2 大量被预测为 class 3
   - mean confidence 约 0.979，ECE 约 0.672，表示模型在错误域上高度自信

### 下一步排查命令

检查云端是否是修正后的 DA 代码:

```text
cd ~/GAPS && grep -n "y_cls = batch\\[1\\]" gaps_flower/domain_adaptation.py && grep -n "weighted_coral_loss" gaps_flower/domain_adaptation.py
```

查看 DA 诊断:

```text
cd ~/GAPS && cat results/flower_cloud_src12_tgt345_cls_da_v5/domain_adapt_latest.json && grep -n "domain_adapt" results/flower_cloud_src12_tgt345_cls_da_v5/history.json
```

比较 plain/adapted 权重差异:

```text
cd ~/GAPS && source ~/gaps_env/bin/activate && python -c "import torch; p='results/flower_cloud_src12_tgt345_cls_da_v5'; a=torch.load(p+'/server_latest.pth',map_location='cpu'); b=torch.load(p+'/server_latest_adapted.pth',map_location='cpu'); diffs=[(k,(a['model_state'][k].float()-b['model_state'][k].float()).abs().max().item(),(a['model_state'][k].float()-b['model_state'][k].float()).abs().mean().item()) for k in a['model_state']]; nz=[x for x in diffs if x[1]>0]; print('changed_tensors',len(nz),'/',len(diffs),'max_diff',max([x[1] for x in diffs] or [0])); print(sorted(nz,key=lambda x:x[1],reverse=True)[:10])"
```

### 继续排查后的根因

云端输出:

```text
changed_tensors 0 / 80
max_diff 0.0
```

说明 `server_latest_adapted.pth` 与 `server_latest.pth` 完全相同。代码根因在
`gaps_flower/strategy.py` 的 `_run_domain_adapt()`:

```text
self._da_trainer.model = da_model
```

替换了当前轮模型，但 `ServerDomainAdaptation.optimizer` 仍然绑定到首次创建 DA trainer 时的旧模型参数。于是后续轮次中:

```text
loss.backward()  -> 梯度写到新模型
optimizer.step() -> 更新旧模型参数
torch.save()     -> 保存未被更新的新模型
```

因此 adapted checkpoint 与 plain checkpoint 完全一致。

另一个诊断异常是:

```text
val_loss = 227.1157
weighted_proto_anchor = 227.0704
total_loss = 227.1157
```

按当前修正后的公式，`total_loss` 应为:

```text
L = L_CE(source) + λ_coral L_CORAL + λ_mmd_g L_MMD_g + λ_mmd_c L_MMD_c + λ_proto L_anchor + λ_adv L_adv
```

因此 `total_loss` 应大于 `val_loss`。复跑前必须确认云端 `domain_adaptation.py` 的损失组合行已经完整同步。

### 已在本地修复

1. `ServerDomainAdaptation.reset_round_state(model, semantic_protos)`
   - 每轮绑定新模型。
   - 重建 Adam optimizer，使 optimizer 参数引用当前轮模型。
   - 刷新 semantic prototypes。
2. `GapsStrategy._run_domain_adapt()`
   - 在已有 DA trainer 的 else 分支调用 `reset_round_state()`。
3. `domain_adapt_latest.json`
   - 增加 `checkpoint_changed_tensors`
   - 增加 `checkpoint_max_abs_delta`
   - 增加 `checkpoint_mean_abs_delta`

### 与单机命令的关键差异

单机命令:

```text
python exp_improved.py --rounds 25 --local_epochs 5 --train_clients 1,2 --test_clients 3,4,5 --no_time_drift --eval_test_only --regression_mode joint --no_use_reg_loss --no_target_cls_calibration --use_adversarial_domain --use_deep_coral --coral_calib_clients 5 --output_dir results/
```

当前 Flower v5 差异:

```text
rounds: Flower 10 vs single 25
local_epochs: both 5
data: both no_time_drift/raw train_features.npy
eval: both test-only after task.py/evaluate_checkpoint.py fix
classification/regression: both classification-only/no regression loss
target cls calibration: both disabled
Deep CORAL: both enabled
adversarial DA: Flower disabled, single enabled
target DA calibration clients: Flower 3,4,5 vs single 5
DA loss weights: Flower hardcoded coral=0.1/adv=0.1, single default config coral=0.5/adv=0.5
server/client algorithm: Flower deployment path is simplified and does not yet fully reproduce gaps_full learnable aggregation/replay distill/server optimizer behavior
```

下一轮建议先做两步:

1. 修 bug 后先跑 5 轮 smoke，确认:
   - `checkpoint_changed_tensors > 0`
   - `total_loss > val_loss` when weighted losses are positive
   - plain/adapted 权重不再完全相同
2. 再跑接近单机配置的 25 轮正式分类实验:
   - `--rounds 25`
   - `--da-use-adversarial true`
   - `--server-calib-data client_5`

### optimizer fix smoke 结果

`results/flower_cloud_src12_da_optimizer_fix_smoke` 已确认 DA checkpoint 不再等于 plain checkpoint:

```text
checkpoint_changed_tensors = 76 / 80
checkpoint_max_abs_delta = 0.000972
checkpoint_mean_abs_delta = 0.000424
```

这说明 optimizer 绑定问题已经修复，当前轮模型确实被 DA 更新。

诊断中仍出现:

```text
val_loss = 224.0813
weighted_proto_anchor = 223.8044
total_loss = 224.0813
```

这不是 weighted losses 未参与优化，而是 `total_loss = val_loss; total_loss += ...`
导致 `val_loss` 和 `total_loss` 引用同一个 Tensor，后续 in-place `+=` 把日志里的
`val_loss` 也改成了完整 loss。按数值反推，纯 CE 约为:

```text
raw_ce ≈ 224.0813 - 223.8044 - 0.1539 - 0.0097 = 0.1133
```

本地已修复为非 in-place:

```text
total_loss = val_loss + weighted_coral + weighted_global_mmd + weighted_class_mmd + weighted_proto + weighted_adv
```

下一次 smoke 中预期:

```text
val_loss ≈ 0.1 左右
total_loss ≈ val_loss + weighted_* 项
checkpoint_changed_tensors > 0
```

### optimizer fix smoke 的 plain/adapted 评估

`results/flower_cloud_src12_da_optimizer_fix_smoke` 在 target test split 上:

```text
plain weighted_accuracy   = 0.8182
adapted weighted_accuracy = 0.8303

plain weighted_nll        = 0.6323
adapted weighted_nll      = 0.5659

plain weighted_ece        = 0.1481
adapted weighted_ece      = 0.1384
```

分客户端:

```text
client_3: acc 0.9706 -> 0.9721, nll 0.1511 -> 0.1486
client_4: acc 0.9063 -> 0.9031, nll 0.3402 -> 0.3255
client_5: acc 0.4063 -> 0.4563, nll 1.9471 -> 1.6930, ece 0.4366 -> 0.3782
```

判断:

1. adapted checkpoint 已经在目标域产生可观测差异，DA 链路从训练、保存到评估已经打通。
2. 2 轮 smoke 中 `client_5` 有方向性改善，但仍是主要瓶颈。
3. `client_5` 的 class 0 仍然 0/80，全被系统性推向 class 2，因此正式实验仍需重点观察 class 0/2 边界。
4. 当前 smoke 不用于和单机最终性能比较，只用于证明 DA 生效。

## 2026-06-03 formal_v1 结果复盘

配置:

```text
rounds=25
local_epochs=5
train_clients=1,2
target_eval_clients=3,4,5
server_calib_data=client_5
da_use_coral=true
da_use_mmd=true
da_use_adversarial=true
target_cls_calibration=false
```

### 指标

Target plain:

```text
weighted_accuracy = 0.824242
weighted_nll = 2.322778
weighted_ece = 0.171938
```

Target adapted:

```text
weighted_accuracy = 0.825000
weighted_nll = 2.069819
weighted_ece = 0.171959
```

Source adapted:

```text
weighted_accuracy = 0.997794
weighted_nll = 0.011715
weighted_ece = 0.002404
```

DA diagnostics:

```text
val_loss = 0.025137
weighted_proto_anchor = 298.692322
total_loss = 298.915741
checkpoint_changed_tensors = 76 / 80
checkpoint_max_abs_delta = 0.002783
checkpoint_mean_abs_delta = 0.001005
```

### 判断

1. Flower 分类联邦训练链路正常，source test accuracy 接近 1.0。
2. DA checkpoint 确认真实更新，且诊断日志正常。
3. Adapted 相比 plain:
   - target weighted accuracy 仅 `+0.000758`
   - target NLL 明显下降 `2.3228 -> 2.0698`
   - ECE 基本不变
4. `client_5` 有局部改善:
   - accuracy `0.4719 -> 0.4938`
   - class 0 `0.0375 -> 0.1000`
   - class 2 `0.8250 -> 0.8500`
5. `client_3/4` 略有牺牲，说明只用 `client_5` 做 DA calibration 会把最终 adapted checkpoint 朝 hard target 偏移。
6. 该结果仍明显低于单机 `src12_tgt345` 分类水平，当前不能声称云端 Flower 已复现单机分类性能。

### 需要继续验证的假设

1. 当前 Flower DA 是 post-aggregation checkpoint adaptation，未作为下一轮全局模型继续下发；单机服务端域适应可能在训练循环中直接更新 global model。
2. Flower 部署路径是简化分类链路，尚未完全迁移 `gaps_full` 中的 learnable aggregation、server optimizer、replay distill 等机制。
3. `weighted_proto_anchor` 远大于其他 DA 项，可能导致 DA 目标被 prototype anchor 主导，需要做 λ_proto 消融。
4. `client_5` class 0/1 仍系统性错分，temperature scaling 无法修复 argmax，需要边界校正或带标签 target CE。

## 2026-06-04 Flower DA feedback patch

### Code changes

Priority P0:

```text
gaps_flower/strategy.py
```

Added `--use-adapted-as-global` support through `GapsStrategy`. When enabled, the server returns the domain-adapted parameter arrays to Flower after `aggregate_fit`, so the next client round receives the adapted global model. This makes the deployment loop closer to the single-machine loop:

```text
client fit -> server aggregate -> server DA -> adapted global -> next client fit
```

The previous loop was:

```text
client fit -> server aggregate -> save adapted checkpoint only -> next client fit still uses plain aggregate
```

Priority P1:

```text
gaps_flower/server_app.py
gaps_flower/strategy.py
```

Exposed DA loss weights from CLI:

```text
--da-lambda-coral
--da-lambda-global-mmd
--da-lambda-class-mmd
--da-lambda-proto-anchor
--da-lambda-adv
--da-lambda-target-ce
--da-server-opt-lr
```

This is needed because `weighted_proto_anchor` dominated formal_v1. The next controlled experiment should reduce `lambda_proto_anchor` from `0.3` to `0.05`, while keeping MMD/CORAL/adversarial terms visible in diagnostics.

Priority P2:

```text
gaps_flower/domain_adaptation.py
```

Added optional supervised target calibration loss:

```text
L_target_ce = CE(f_theta(x_target_calib), y_target_calib)
```

This is off by default (`--da-lambda-target-ce 0.0`). If enabled, it is no longer strictly equivalent to the single-machine `--no_target_cls_calibration` setting; it becomes a supervised target classifier boundary calibration experiment.

### Verification

```text
python -m py_compile gaps_flower/domain_adaptation.py gaps_flower/strategy.py gaps_flower/server_app.py gaps_flower/task.py gaps_flower/evaluate_checkpoint.py
python -m gaps_flower.server_app --help
```

Both checks passed locally. `git diff --check` only reports an unrelated trailing whitespace in `split_dataset.py`.

### Next experiments

Smoke first:

```text
rounds=3
use_adapted_as_global=true
lambda_proto_anchor=0.05
lambda_target_ce=0.0
```

If smoke history contains `"returned_parameters": "adapted"` and checkpoint deltas remain nonzero, run formal_v2 with 25 rounds. If target accuracy is still capped near 0.82, run formal_v3 with `lambda_target_ce=0.5` as a supervised target calibration ablation.

## 2026-06-04 Experiment C DA feedback result

Setting:

```text
strategy=gaps
use_domain_adapt=true
use_adapted_as_global=true
domain_adapt_steps=40
lambda_coral=0.5
lambda_global_mmd=0.5
lambda_class_mmd=0.5
lambda_proto_anchor=0.3
lambda_adv=0.5
lambda_target_ce=0.0
server_opt_lr=5e-4
server_calib_data=client_5
```

History check:

```text
returned_parameters="adapted" for recorded rounds
checkpoint_changed_tensors=76/80
checkpoint_max_abs_delta=0.0162265
checkpoint_mean_abs_delta=0.0024682
```

DA diagnostics:

```text
val_loss=0.001526
weighted_coral_loss=0.0000009
weighted_mmd_global=0.000381
weighted_mmd_class=0.000969
weighted_adv_loss=0.032226
weighted_proto_anchor=307.066071
total_loss=307.101196
```

Target test comparison against Experiment A:

```text
A no DA target weighted_accuracy=0.804545, nll=2.736836, ece=0.190734
C server_latest target weighted_accuracy=0.956061, nll=0.431083, ece=0.042583
C server_latest_adapted target weighted_accuracy=0.954545, nll=0.518280, ece=0.045698
```

Per-client C result:

```text
plain/latest: client_3=0.977941, client_4=0.921875, client_5=0.943750
adapted/latest: client_3=0.977941, client_4=0.893750, client_5=0.965625
```

Interpretation:

1. The adapted-feedback loop is effective; target weighted accuracy improves by about +15.15 percentage points over A.
2. Final `server_latest.pth` is slightly better than `server_latest_adapted.pth` on weighted target accuracy because final adaptation improves client_5 but hurts client_4.
3. `weighted_proto_anchor` still dominates the DA objective, so a lower `lambda_proto_anchor` run remains useful only if we need to reduce client_4 degradation.
4. The pasted evaluation reports `round=20`; before treating this as the final 25-round result, verify whether the C server run was launched with 20 or 25 rounds and whether it finished all intended rounds.

## 2026-06-04 Experiment D lower proto-anchor result

Setting:

```text
strategy=gaps
use_domain_adapt=true
use_adapted_as_global=true
domain_adapt_steps=40
lambda_coral=0.5
lambda_global_mmd=0.5
lambda_class_mmd=0.5
lambda_proto_anchor=0.05
lambda_adv=0.5
lambda_target_ce=0.0
server_opt_lr=5e-4
server_calib_data=client_5
rounds=20
```

History check:

```text
returned_parameters="adapted" for all recorded rounds
checkpoint_changed_tensors=76/80
checkpoint_max_abs_delta=0.0217886
checkpoint_mean_abs_delta=0.0028896
```

DA diagnostics:

```text
val_loss=0.000049
weighted_coral_loss=0.000001
weighted_mmd_global=0.002524
weighted_mmd_class=0.013643
weighted_adv_loss=0.118642
weighted_proto_anchor=50.942352
total_loss=51.077217
```

Target test comparison:

```text
C latest target weighted_accuracy=0.956061, nll=0.431083, ece=0.042583
C adapted target weighted_accuracy=0.954545, nll=0.518280, ece=0.045698
D latest target weighted_accuracy=0.937121, nll=0.664377, ece=0.060711
D adapted target weighted_accuracy=0.926515, nll=0.872289, ece=0.073045
```

Per-client D result:

```text
plain/latest: client_3=0.969118, client_4=0.862500, client_5=0.943750
adapted/latest: client_3=0.952941, client_4=0.828125, client_5=0.968750
```

Interpretation:

1. Reducing `lambda_proto_anchor` from `0.3` to `0.05` reduced the scalar dominance of proto-anchor, but did not improve the multi-target weighted result.
2. D keeps client_5 strong, especially in the final adapted checkpoint, but it further hurts client_3/client_4.
3. For single-target deployment to client_5, both C-adapted and D-adapted are viable; for balanced target clients 3/4/5, C-latest is better.
4. Current recommendation: use C-style `lambda_proto_anchor=0.3` as the deployment classification DA default, and prefer `server_latest.pth` over `server_latest_adapted.pth` when evaluating mixed targets.

## 2026-06-04 Experiment E target CE patch

Goal:

```text
Test whether a labeled target calibration CE term can push the target-device
classification route closer to the single-machine level without changing the
already validated Flower DA feedback loop.
```

Code changes:

```text
gaps_flower/domain_adaptation.py
gaps_flower/strategy.py
gaps_flower/server_app.py
```

Added E-specific target CE controls:

```text
--da-lambda-target-ce
--da-target-ce-label-smoothing
--da-target-ce-class-balanced
```

Implementation details:

```text
L_targetCE = CE(logits_t, y_t; optional class weights, optional label smoothing)
```

Diagnostics now include:

```text
target_ce_loss
target_ce_acc
weighted_target_ce_loss
target_ce_label_smoothing
target_ce_class_balanced
target_ce_class_weights
```

Expected interpretation:

1. If E improves client_5 while preserving client_3/client_4, use E as the supervised target-device classification calibration setting.
2. If E only improves client_5 but hurts mixed-target metrics, use it only for single-target deployment.
3. If E does not improve over C, keep C as the classification DA default and avoid adding direct target CE.

## 2026-06-04 Experiment E target CE result

Setting:

```text
strategy=gaps
use_domain_adapt=true
use_adapted_as_global=true
domain_adapt_steps=40
lambda_coral=0.5
lambda_global_mmd=0.5
lambda_class_mmd=0.5
lambda_proto_anchor=0.3
lambda_adv=0.5
lambda_target_ce=0.5
target_ce_label_smoothing=0.05
target_ce_class_balanced=true
server_calib_data=client_5
rounds=20
```

History check:

```text
returned_parameters="adapted" for all recorded rounds
checkpoint_changed_tensors=76/80
checkpoint_max_abs_delta=0.0202004
checkpoint_mean_abs_delta=0.0024794
```

DA diagnostics:

```text
target_ce_loss=0.463004
target_ce_acc=0.997656
weighted_target_ce_loss=0.231502
weighted_proto_anchor=308.788940
total_loss=309.049652
target_ce_class_weights=[1.0, 1.0, 1.0, 1.0]
```

Target test comparison:

```text
C latest target weighted_accuracy=0.956061, nll=0.431083, ece=0.042583
C adapted target weighted_accuracy=0.954545, nll=0.518280, ece=0.045698
E latest target weighted_accuracy=0.936364, nll=0.594564, ece=0.062028
E adapted target weighted_accuracy=0.951515, nll=0.455936, ece=0.047977
```

Per-client E result:

```text
latest: client_3=0.969118, client_4=0.900000, client_5=0.903125
adapted: client_3=0.969118, client_4=0.906250, client_5=0.959375
```

Interpretation:

1. Target CE did not improve over C. The best E checkpoint is E-adapted, but it remains below C-latest and C-adapted on weighted accuracy.
2. `target_ce_acc=0.997656` on calibration batches but client_5 test accuracy is lower than C-adapted, indicating calibration-set CE is not the missing mechanism.
3. Since class weights are all 1.0, the client_5 calibration split is balanced; class-balanced CE has no practical effect here.
4. Current classification DA default should remain C-style: `lambda_proto_anchor=0.3`, `lambda_target_ce=0.0`, use adapted feedback, and prefer `server_latest.pth` for mixed-target evaluation.

## 2026-06-04 Regression/QC deployment linkage patch

Goal:

```text
Move from classification-only Flower validation to a deployable inference chain:
classification route -> regression ppm -> target calibration -> QC decision -> CSV/JSON output.
```

Code changes:

```text
gaps_deploy/inference.py
gaps_deploy/calibration.py
gaps_deploy/predict_client_file.py
gaps_deploy/build_package.py
```

Key fixes:

1. `DeployPredictor.predict_batch()` now uses the regression model B to extract its own `reg_feat` before `forward_reg()`. The classifier model A is used only for route selection.
2. `predict_batch()` accepts per-window phase arrays, so `test_phase_labels.npy` can be used without collapsing a batch to a single majority phase.
3. `predict_client_file.py` automatically loads sidecar labels next to `test_features.npy` / `calibration_features.npy`:

```text
{split}_classification_labels.npy
{split}_regression_labels.npy
{split}_phase_labels.npy
```

It then writes `true_class`, `true_ppm`, `abs_error`, and summary `classification_accuracy`, `R2`, `MAE`, `P90AE`.

4. `RegressionCalibrator.load_routing_config()` now routes `bias_only` params into `bias_params`; previously they were loaded as affine params but `calibrate()` looked in `bias_params`.
5. Added `gaps_deploy/build_package.py` to create the standard deploy package layout consumed by `DeployPredictor.from_package()`.
6. `gaps_flower/calibration_fit.py` now prefers `calibration_*.npy` for target calibration fitting/metrics and only falls back to `train_*.npy` for legacy layouts.
7. `gaps_flower/regression_task.py` now calls `create_train_loader(..., normalize=False)` explicitly, matching the Flower classification path and the single-machine simulation input scale.
8. `gaps_deploy/inference.py` now treats `calibration_stats.json` as optional response-reference data only if it contains class-keyed `center/scale/z_sigs`; plain calibration metric files are skipped safely.
9. `gaps_deploy/predict_client_file.py` prints `R2` instead of `R²`, avoiding Windows GBK console failures during local dry-run.
10. `gaps_flower/calibration_fit.py` preserves original sample order when computing overall calibrated metrics; previous per-class metrics were valid, but overall calibrated R2/MAE could be wrong because calibrated predictions were concatenated by class.

Verification:

```text
python -m py_compile gaps_deploy/inference.py gaps_deploy/calibration.py gaps_deploy/predict_client_file.py gaps_deploy/build_package.py
python -c "from gaps_deploy import DeployPredictor, DeployConfig; from gaps_deploy.build_package import build_package; print('ok')"
git diff --check -- gaps_deploy/inference.py gaps_deploy/calibration.py gaps_deploy/predict_client_file.py gaps_deploy/build_package.py
python -m py_compile gaps_flower/calibration_fit.py gaps_flower/regression_task.py gaps_flower/regression_client.py gaps_flower/regression_server.py
temporary package dry-run: build dummy checkpoints -> build deploy package -> DeployPredictor.from_package(...).predict_batch(...)
temporary CLI dry-run: python -m gaps_deploy.predict_client_file with sidecar labels -> CSV/JSON summary with classification_accuracy and regression metrics
```

All checks passed locally.

Cloud regression/QC smoke result on client_5:

```text
classification_accuracy=0.94375
test_R2=0.414661
test_MAE=37.8400 ppm
test_P90AE=81.6191 ppm
QC accept/review/reject=313/0/7
```

Calibration v2 fixed the overall calibrated metrics on the client_5 calibration split:

```text
calibration split raw: R2=-0.1063, MAE=53.8656
calibration split affine: R2=0.5467, MAE=34.6072
```

The deployed client_5 test metrics stayed identical to v1 because the fitted affine parameters did not change; only the overall calibrated metric calculation order was corrected.

## 2026-06-04 Calibration Mode Comparison Runner

Added:

```text
gaps_deploy/evaluate_calibration_modes.py
```

Purpose:

```text
Run none / bias_only / affine_only through the same deployment path:
calibration_fit -> build_package -> predict_client_file -> comparison CSV/JSON.
```

Outputs:

```text
calibration_mode_comparison.csv
calibration_mode_comparison.json
<mode>/calibration/
<mode>/package/
<mode>/test_outputs.csv
<mode>/test_summary.json
```

Verification:

```text
python -m py_compile gaps_deploy/evaluate_calibration_modes.py
python -m gaps_deploy.evaluate_calibration_modes --help
temporary dummy dry-run with modes none,bias_only
```

Cloud client_5 comparison:

```text
none:        test_R2=-0.207626, MAE=55.4291, MedAE=47.4270, P90AE=117.3607, Bias=-40.1234
bias_only:   test_R2= 0.377560, MAE=38.9931, MedAE=33.7201, P90AE= 80.6476, Bias=  0.9830
affine_only: test_R2= 0.414661, MAE=37.8400, MedAE=27.2923, P90AE= 81.6191, Bias=  1.0079
```

Interpretation:

```text
Calibration is necessary. The uncalibrated regressor strongly underestimates client_5 concentration
(Bias=-40.12 ppm). Both bias_only and affine_only remove the systematic bias. affine_only is the
current default because it has the best R2, MAE, and median absolute error on client_5 test; bias_only
has a marginally smaller P90AE but the difference is only about 1 ppm.
```

Follow-up patch:

```text
gaps_deploy/predict_client_file.py now writes per_true_class_evaluation in test_summary.json.
gaps_deploy/evaluate_calibration_modes.py now expands per-gas metrics into comparison CSV columns:
class{c}_R2, class{c}_MAE, class{c}_P90AE, class{c}_Bias, class{c}_classification_accuracy.
```

Purpose:

```text
Identify which gas class dominates regression error before porting auto_v2_specialist.
```

Cloud per-gas result on client_5:

```text
affine_only overall: R2=0.414661, MAE=37.8400, MedAE=27.2923, P90AE=81.6191

Ethanol:  R2= 0.5352, MAE=18.4686, P90AE= 40.9371, Bias=  5.4701, cls_acc=0.9750
CO:       R2=-0.1842, MAE=64.6341, P90AE=123.4866, Bias=-10.6172, cls_acc=0.8875
Ethylene: R2=-0.3620, MAE=33.3142, P90AE= 53.7174, Bias=  4.2511, cls_acc=0.9375
Methane:  R2= 0.6410, MAE=34.9431, P90AE= 57.9184, Bias=  4.9278, cls_acc=0.9750
```

Interpretation:

```text
CO is the dominant regression bottleneck and also has the lowest classification routing accuracy.
Ethylene still has negative R2 but moderate MAE/P90AE, suggesting compressed dynamic range rather than
a pure bias problem. Specialist migration should start with CO, then Ethylene.
```

## 2026-06-04 Gated Specialist Deployment Support

Added:

```text
gaps_flower/specialist_calibration_fit.py
```

Updated:

```text
gaps_deploy/build_package.py supports --specialist-dir and copies models/specialists/*.pth.
gaps_deploy/inference.py loads models/specialists/class_{id}.pth and applies them when routing_config selected_modes[class_id] is specialist_full.
```

Design:

```text
The specialist path is gated. It starts from the affine baseline, trains specialist candidates only for requested classes, and writes specialist_full only if validation score improves by min_delta. If the gate rejects a class, the deploy package silently falls back to affine_only.
```

Initial target:

```text
classes=1,2 (CO, Ethylene)
gate_metric=R2
class_weight=2.0
steps=80
```

Cloud gated specialist result on client_5:

```text
gate accepted: class_1 CO, class_2 Ethylene

overall affine_only baseline:
R2=0.414661, MAE=37.8400, MedAE=27.2923, P90AE=81.6191, Bias=1.0079

overall specialist_gated:
R2=0.430256, MAE=30.0064, MedAE=14.5135, P90AE=70.9412, Bias=-4.2457
```

Per-gas comparison against affine_only:

```text
Ethanol:  MAE 18.47 -> 16.44, R2 0.535 -> 0.628
CO:       MAE 64.63 -> 57.15, R2 -0.184 -> -0.229, Bias -10.62 -> -25.28
Ethylene: MAE 33.31 -> 15.64, R2 -0.362 -> 0.144
Methane:  MAE 34.94 -> 30.79, R2 0.641 -> 0.589
```

Interpretation:

```text
Ethylene specialist is clearly useful. CO specialist improves MAE/P90AE but worsens R2 and bias, so
R2-only gating on a 12-sample validation split is not reliable enough for CO. Next specialist selection
should use a composite gate or per-class metric preference: Ethylene by R2, CO by MAE/P90AE with bias
guardrail.
```

## 2026-06-05 Specialist Guarded Gate

Added to `gaps_flower/specialist_calibration_fit.py`:

```text
--gate-mode metric|guarded
--p90-max-worsen
--bias-max-worsen
--refit-full-calib
--refit-steps
```

Reason:

```text
The first specialist experiment accepted CO and Ethylene using an R2-only gate. Ethylene improved
consistently, but CO showed MAE/P90AE improvement with much worse Bias. The guarded gate keeps the
primary metric check, then rejects a specialist if P90AE or absolute Bias worsens beyond explicit
limits. This is closer to deployment QC logic: a lower average error is not enough if systematic
bias becomes unacceptable.
```

Next validation:

```text
Use gate_metric=MAE, min_delta=5, p90_max_worsen=0, bias_max_worsen=10 on classes 1,2.
Expected behavior from the previous validation numbers: class_2 Ethylene should be accepted; class_1
CO should likely be rejected because its validation Bias worsened by about 20 ppm. The resulting
package tests whether "affine for CO + specialist for Ethylene" is safer than accepting both.
```

## 2026-06-05 E2 Result And Metric Priority Update

E2 result on client_5:

```text
selected_specialists: [2]
overall: R2=0.423287, MAE=31.8394, P90AE=80.9974, classification_accuracy=0.9437

Ethanol:  MAE=16.3545, R2=0.6295
CO:       MAE=66.7696, R2=-0.3667
Ethylene: MAE=14.3104, R2=0.3688
Methane:  MAE=29.9231, R2=0.6467
```

Interpretation after switching priority to MAE/RMSE/NRMSE:

```text
E2 is not the best deployment candidate under the main metric priority. It rejects CO because the
validation Bias guard fails, but the earlier CO+Ethylene specialist package had lower overall MAE
(30.006 vs 31.839) and lower CO MAE on test (57.15 vs 66.77). Bias is useful as a diagnostic, but
it should not be the primary hard gate if deployment reporting focuses on MAE/RMSE/NRMSE.
```

Code adjustment:

```text
gaps_deploy/predict_client_file.py now reports RMSE and NRMSE_range overall and per gas.
gaps_flower/specialist_calibration_fit.py now supports gate_metric=RMSE and gate_metric=NRMSE_range.
P90 and Bias guards are explicit switches instead of always-on guarded constraints.
Added --refit-affine-full-calib so validation split is used for gate selection, then the final
general affine fallback can be refit on the full target calibration set.
```

## 2026-06-05 E3 NRMSE-Gated Specialist Result

Command summary:

```text
gate_metric=NRMSE_range
gate_mode=metric
min_delta=0.01
refit_affine_full_calib=true
refit_full_calib=true
classes=1,2
```

Gate result:

```text
class_1 CO:       baseline NRMSE=0.2904, specialist NRMSE=0.2337, accepted=true
class_2 Ethylene: baseline NRMSE=0.2855, specialist NRMSE=0.1196, accepted=true
loaded specialists: class_1, class_2
```

Client_5 test result:

```text
overall: R2=0.474803, MAE=30.0958, RMSE=48.0949, NRMSE_range=0.2595, P90AE=62.5
classification_accuracy=0.9437
QC: accept=313/320 (97.8%), reject=7/320 (2.2%)

Ethanol:  MAE=17.0865, RMSE=22.0305, NRMSE=0.1958, R2=0.6235
CO:       MAE=51.8296, RMSE=74.4463, NRMSE=0.3309, R2=-0.0749
Ethylene: MAE=15.0576, RMSE=31.2403, NRMSE=0.2777, R2=0.2429
Methane:  MAE=36.4096, RMSE=47.4231, NRMSE=0.2108, R2=0.5638
```

Interpretation:

```text
E3 is the best current regression deployment candidate by the main priority metrics. Compared with
affine_only, overall MAE decreases from 37.84 to 30.10. Compared with E2, overall MAE decreases from
31.84 to 30.10 and CO MAE decreases from 66.77 to 51.83. This supports using MAE/RMSE/NRMSE-driven
specialist selection rather than Bias/P90 hard guards.
```

## 2026-06-05 Next Cross-Client Deployment Check

Added:

```text
gaps_deploy/compare_deploy_summaries.py
```

Purpose:

```text
After one deployment package is evaluated on client_3/client_4/client_5, collect the JSON summaries
into a single CSV/table with overall MAE, RMSE, NRMSE_range, classification accuracy, accept rate,
and per-gas metrics. This avoids manual comparison errors and makes later package selection easier.
```

Next validation question:

```text
E3 was calibrated on client_5. Evaluate the same E3 package on client_3, client_4, and client_5.
If client_5 is clearly best while client_3/client_4 degrade, the deployment rule should be
"one target calibration set -> one target-domain package". If all target clients remain acceptable,
then a shared target package may be feasible.
```

## 2026-06-05 E3 Cross-Client Evaluation

Package:

```text
results/deploy_pkg_client5_specialist_E3_nrmse
calibrated/specialized on client_5 calibration split
loaded specialists: class_1 CO, class_2 Ethylene
```

Cross-client summary:

```text
client_3 with C5 package: MAE=51.9441, RMSE=69.9875, NRMSE=0.4248, cls_acc=0.9779, accept=99.26%
client_4 with C5 package: MAE=53.3676, RMSE=70.0865, NRMSE=0.4249, cls_acc=0.9219, accept=99.06%
client_5 with C5 package: MAE=30.0958, RMSE=48.0949, NRMSE=0.2595, cls_acc=0.9437, accept=97.81%
```

Interpretation:

```text
The client_5-specific package does not generalize well to client_3/client_4. C3 and C4 retain high
classification accuracy, especially C3, but their regression error nearly doubles in normalized
terms compared with C5. The degradation is concentrated in Ethanol, CO, and Ethylene; Methane remains
relatively stable. This supports the deployment rule: use one calibration/specialist package per
target device/domain, rather than one shared package for all target clients.
```

## 2026-06-05 Next Target-Specific Package Validation

Next experiment:

```text
Build E3-style packages separately for client_3 and client_4, using each client's own calibration
split, then evaluate each package on its matching test split. Compare:

C3_with_C3pkg
C4_with_C4pkg
C5_with_C5pkg
```

Expected decision rule:

```text
If C3_with_C3pkg and C4_with_C4pkg recover toward C5_with_C5pkg-level MAE/RMSE/NRMSE, deployment
should use shared classifier/source-regression checkpoints plus per-device target calibration
artifacts. If they remain poor, the bottleneck is no longer package sharing but the source regression
model or target calibration set quality for those devices.
```

## 2026-06-05 Target-Specific Package Result

E3-style target-specific packages:

```text
C3 package: accepted specialists class_1 CO and class_2 Ethylene
C4 package: accepted specialists class_1 CO and class_2 Ethylene
C5 package: accepted specialists class_1 CO and class_2 Ethylene
```

Matched-client summary:

```text
C3_with_C3pkg: MAE=22.6890, RMSE=35.1208, NRMSE=0.1902, cls_acc=0.9779, accept=99.26%
C4_with_C4pkg: MAE=32.7349, RMSE=53.6193, NRMSE=0.2692, cls_acc=0.9219, accept=99.06%
C5_with_C5pkg: MAE=30.0958, RMSE=48.0949, NRMSE=0.2595, cls_acc=0.9437, accept=97.81%
```

Compared with using the C5 package on all clients:

```text
C3: MAE 51.9441 -> 22.6890, RMSE 69.9875 -> 35.1208, NRMSE 0.4248 -> 0.1902
C4: MAE 53.3676 -> 32.7349, RMSE 70.0865 -> 53.6193, NRMSE 0.4249 -> 0.2692
```

Interpretation:

```text
The target-specific package strategy is validated. Shared classifier and source-regression
checkpoints can be reused, but calibration routing, affine parameters, specialists, and later QC
policy should be generated per target device/domain. C4 remains the hardest device, mainly because
CO has low classification accuracy (0.725) and high regression error even after the CO specialist.
```

## 2026-06-05 Flower Deployment vs Single-Machine Regression Gap

Single-machine `src12_tgt345` summary from `classconc_final_qc_pipeline_20260601`:

```text
Calibrated:
C3: MAE=12.8104, RMSE=20.6550, NRMSE=0.0870
C4: MAE=12.2662, RMSE=18.7723, NRMSE=0.0790
C5: MAE=18.5910, RMSE=34.4924, NRMSE=0.1452

QC accepted:
C3: MAE=12.0432, RMSE=19.3275, NRMSE=0.0814, coverage=88.68%
C4: MAE=10.4035, RMSE=15.2526, NRMSE=0.0642, coverage=88.13%
C5: MAE=12.1255, RMSE=17.1534, NRMSE=0.0722, coverage=83.13%
```

Current Flower deployment with target-specific E3 packages:

```text
C3_with_C3pkg: MAE=22.6890, RMSE=35.1208, NRMSE=0.1902, cls_acc=0.9779
C4_with_C4pkg: MAE=32.7349, RMSE=53.6193, NRMSE=0.2692, cls_acc=0.9219
C5_with_C5pkg: MAE=30.0958, RMSE=48.0949, NRMSE=0.2595, cls_acc=0.9437
```

Interpretation:

```text
The deployment pipeline is connected, but it has not reproduced the single-machine regression
performance. The gap is largest on C4. In single-machine results, C4 classification was essentially
perfect and CO MAE was 16.69. In Flower deployment, C4 CO classification accuracy is 0.725 and CO
MAE is 64.98. Therefore the next diagnostic should decompose C4 CO into route-correct and
route-wrong subsets before changing the regression specialist.
```

Added:

```text
gaps_deploy/analyze_route_errors.py
```

Purpose:

```text
Read deployment prediction CSV files and report metrics by true class, predicted class,
route_correct/route_wrong, and target-class wrong-route destinations. This separates classification
routing errors from regression-head calibration errors.
```

## 2026-06-05 C4 Route Error Diagnosis

C4 target-specific E3 package:

```text
overall: MAE=32.7349, RMSE=53.6193, NRMSE=0.2692
route_correct windows: n=295, MAE=23.4032, RMSE=33.0357, NRMSE=0.1887
route_wrong windows:   n=25,  MAE=142.8488, RMSE=154.6687, NRMSE=0.7126
```

C4 CO decomposition:

```text
true CO all:           n=80, MAE=64.9792, RMSE=92.0278, NRMSE=0.4090
true CO route_correct: n=58, MAE=31.1794, RMSE=40.3298, NRMSE=0.1792
true CO route_wrong:   n=22, MAE=154.0879, RMSE=162.8152, NRMSE=0.7236

wrong destinations:
CO -> Ethanol:  n=16, MAE=143.8165
CO -> Ethylene: n=6,  MAE=181.4782
```

C4 Ethylene decomposition:

```text
true Ethylene all:           n=80, MAE=7.3489, RMSE=10.7155, NRMSE=0.0952
true Ethylene route_correct: n=78, MAE=6.6098, RMSE=8.9066, NRMSE=0.0792
true Ethylene route_wrong:   n=2,  MAE=36.1727
```

Checkpoint comparison:

```text
server_latest / server_round_020:
C3 acc=0.9779, C4 acc=0.9219, C5 acc=0.9438
C4 CO acc=0.725, C5 CO acc=0.8875

server_latest_adapted / server_round_020_adapted:
C3 acc=0.9779, C4 acc=0.8938, C5 acc=0.9656
C4 CO acc=0.625, C5 CO acc=0.9625
```

Interpretation:

```text
The adapted classifier checkpoint is biased toward client_5 and hurts client_4 CO routing. For C4,
the plain latest classifier is the better shared classifier, but still far below the single-machine
C4 CO route accuracy. C4 CO error has two layers: wrong routing is the biggest source of catastrophic
errors, while route-correct CO regression is still worse than the single-machine CO result
(31.18 MAE vs 16.69 MAE). Next work should prioritize classifier checkpoint/route guard before
further CO specialist tuning.
```

## 2026-06-05 A-Test: Single-Machine Classifier In Deployment

Classifier checkpoint evaluation:

```text
T1b server_checkpoint:
C3 acc=0.9882, C4 acc=0.9969, C5 acc=0.9531, weighted_acc=0.9818
C4 per-class acc: Ethanol=1.0, CO=1.0, Ethylene=1.0, Methane=0.9875

T1b target_cls_client4:
C4 acc=1.0, all per-class acc=1.0
```

C4 deployment package with single-machine C4 classifier + Flower regression/calibration:

```text
C4_flower_cls: MAE=32.7349, RMSE=53.6193, NRMSE=0.2692, cls_acc=0.9219
C4_single_cls: MAE=24.6711, RMSE=34.3878, NRMSE=0.1945, cls_acc=1.0000
```

Route diagnosis after replacing classifier:

```text
all windows route_correct: n=320/320
CO route_accuracy: 1.0, MAE=33.1820, RMSE=42.2013, NRMSE=0.1876
Ethylene route_accuracy: 1.0, MAE=7.2775, RMSE=10.2607, NRMSE=0.0912
```

Interpretation:

```text
Replacing only the classifier removes C4 wrong-route errors and reduces overall MAE by 24.6%.
This confirms the first deployment bottleneck was Flower classifier routing, especially C4 CO.
However, the result is still behind single-machine calibrated C4 (MAE=12.2662, RMSE=18.7723,
NRMSE=0.0790). The remaining gap is now mainly regression/checkpoint/calibration mismatch rather
than classification routing.
```

## 2026-06-05 Experience Summary And Next Design

Main experience:

```text
The deployment chain should be decomposed into two independent error sources:

1. Classification route error:
   The classifier decides which gas-specific calibration/specialist branch is used. A wrong route
   changes the regression target semantics, so its error is not a small additive perturbation. On
   client_4 CO, wrong-route windows produced MAE above 150 ppm.

2. Route-correct regression/calibration error:
   After replacing the classifier with the single-machine C4 classifier, all C4 windows were routed
   correctly, but overall MAE was still 24.67 ppm. This means the remaining error comes from the
   Flower regression checkpoint and target calibration/specialist artifacts.
```

Design implication:

```text
The classifier checkpoint can be selected independently from the regression checkpoint. For the
current deployment goal, the single-machine T1b global classifier is a strong candidate shared
router because it reaches C3=0.9882, C4=0.9969, C5=0.9531 on target test. The target-specific C4
classifier is useful as an upper-bound diagnostic, but it is not the first choice for a shared
deployment package unless each target device is allowed to have its own supervised classifier
calibration.
```

Next validation design:

```text
Step A2:
Use the T1b global classifier checkpoint as the shared classification router, keep the current
Flower source regression checkpoint, and rebuild target-specific E3 calibration packages for
client_3, client_4, and client_5.

Purpose:
Measure how much of the deployment gap is removed by using the stronger shared classifier without
introducing target-specific classifier overfitting.

Decision:
If A2 improves C3/C4/C5 consistently, freeze the classification router as T1b global for the
deployment package and move all optimization effort to regression checkpoint alignment.
```

Regression follow-up:

```text
After A2, compare Flower regression against the single-machine regression/calibration path. The
most likely remaining mismatch is not QC, because current QC accepts nearly all windows, but one or
more of:

1. regression checkpoint is not the same as the single-machine source-domain regression checkpoint;
2. deploy calibration implements only part of class_concentration + auto_v2_specialist;
3. specialist training/gating differs from exp_improved.py;
4. calibration artifacts are selected by different validation metrics or splits.
```

## 2026-06-05 A2 Planned Experiment: Shared T1b Classifier

Principle:

```text
Change only one variable at a time. In A2, replace the Flower classifier with the single-machine
T1b global classifier, but keep the current Flower regression checkpoint and each target client's
own E3 calibration/specialist artifacts. This avoids mixing classifier improvement with regression
checkpoint changes.
```

Inputs:

```text
classifier checkpoint:
results/T1b_cls_src12_tgt345/checkpoints/server_checkpoint.pth

regression checkpoint:
results/flower_regression_src12_Cbase/regression_fedavg_global.pt

target-specific calibration/specialist dirs:
results/flower_regression_src12_Cbase/calibration_client3_specialist_E3_nrmse
results/flower_regression_src12_Cbase/calibration_client4_specialist_E3_nrmse
results/flower_regression_src12_Cbase/calibration_client5_specialist_E3_nrmse
```

Expected interpretation:

```text
If C3/C4/C5 all improve compared with the Flower-classifier packages, freeze T1b global classifier
as the deployment router and stop spending time on Flower classifier DA for now. If only C4 improves,
then the benefit mainly comes from fixing C4 CO routing, and C5 may still prefer the adapted Flower
classifier or a target-specific classifier. If no client improves, the current error is dominated by
regression/calibration and classifier replacement should not be the main deployment direction.
```

Acceptance rule:

```text
Use MAE, RMSE, and NRMSE_range as primary metrics. Classification accuracy is a diagnostic upstream
metric. QC accept rate is secondary in A2 because the current QC policy accepts almost all windows.
```

## 2026-06-05 A2 Result: Shared T1b Classifier

Comparison:

```text
C3_flowercls: MAE=22.6890, RMSE=35.1208, NRMSE=0.1902, cls_acc=0.9779
C3_t1bcls:    MAE=21.9429, RMSE=32.9109, NRMSE=0.1816, cls_acc=0.9882

C4_flowercls: MAE=32.7349, RMSE=53.6193, NRMSE=0.2692, cls_acc=0.9219
C4_t1bcls:    MAE=24.7434, RMSE=34.6664, NRMSE=0.1955, cls_acc=0.9969

C5_flowercls: MAE=30.0958, RMSE=48.0949, NRMSE=0.2595, cls_acc=0.9437
C5_t1bcls:    MAE=29.1884, RMSE=44.5326, NRMSE=0.2468, cls_acc=0.9531
```

Route analysis:

```text
C3:
route_correct n=672/680, MAE=21.5215
route_wrong   n=8/680,   MAE=57.3411
CO route_acc=0.9882, Ethylene route_acc=0.9706

C4:
route_correct n=319/320, MAE=24.3676
CO route_acc=1.0000, Ethylene route_acc=1.0000

C5:
route_correct n=305/320, MAE=26.6448
route_wrong   n=15/320,  MAE=80.9095
CO route_acc=0.9125, Ethylene route_acc=0.9500
```

Interpretation:

```text
The shared T1b global classifier improves all three target clients under the main deployment metrics.
The improvement is small but consistent on C3/C5, and large on C4. C4 confirms the earlier diagnosis:
most of the catastrophic CO error came from classifier routing. After T1b routing, C4 CO and Ethylene
are both routed correctly, but the route-correct regression MAE remains much higher than the
single-machine calibrated result.

Decision:
Freeze T1b server_checkpoint as the current shared deployment classification router. Further Flower
classification DA tuning is not the priority unless the deployment target specifically requires an
end-to-end Flower-trained classifier. The next productive line is regression checkpoint/calibration
alignment against exp_improved.py.
```

Remaining gap to single-machine calibrated results:

```text
Single-machine calibrated:
C3 MAE=12.8104, RMSE=20.6550, NRMSE=0.0870
C4 MAE=12.2662, RMSE=18.7723, NRMSE=0.0790
C5 MAE=18.5910, RMSE=34.4924, NRMSE=0.1452

A2 deployment:
C3 MAE=21.9429, RMSE=32.9109, NRMSE=0.1816
C4 MAE=24.7434, RMSE=34.6664, NRMSE=0.1955
C5 MAE=29.1884, RMSE=44.5326, NRMSE=0.2468

Therefore the remaining error is no longer mainly a classifier route problem. The next audit should
compare the regression checkpoint and calibration/specialist implementation against the single-machine
class_concentration + auto_v2_specialist pipeline.
```

## 2026-06-05 Regression-Side Audit After A2

Experience summary:

```text
A2 freezes the classification router decision: T1b global classifier is better than the current
Flower classifier for all target clients. The remaining gap must be audited on the regression side.

The single-machine T2d result is much stronger than A2:
single-machine C3/C4/C5 calibrated MAE = 12.81 / 12.27 / 19.19
A2 deployment C3/C4/C5 MAE = 21.94 / 24.74 / 29.19
```

Code-level difference:

```text
Single-machine exp_improved.py uses auto_v2_specialist:

1. split target calibration by class_concentration;
2. train/evaluate five general candidates:
   none, bias_only, affine_only, phase_affine_only, full;
3. select candidate per class by validation score;
4. train specialist_full candidates for selected classes;
5. optionally refit accepted specialists on full calibration split.

Current deployment specialist_calibration_fit.py is only a subset:

1. split target calibration only by class;
2. fit affine_only as the general fallback;
3. train specialist candidates for requested classes;
4. gate specialists by validation metric;
5. no full candidate, no phase_affine candidate, no per-class auto_v2 candidate selection.
```

Artifact audit:

```text
Local single-machine artifacts exist:
results/T2d_reg_classconc_split_src12_tgt345/checkpoints/separate_regression/
  separate_regression_client3.pth
  separate_regression_client4.pth
  separate_regression_client5.pth
  separate_regression_metrics.json

Each client checkpoint contains:
model_state, classifier_state, metrics

However, the metrics routing_config often selects "full" and "specialist_full", while the checkpoint
only saves client_model.state_dict(). The runtime full_model and specialist_models are not saved as
separate deployable model files in this artifact directory. Therefore directly using
separate_regression_client{cid}.pth as the deployment regression checkpoint would not faithfully
reproduce the single-machine auto_v2_specialist pipeline.
```

Design decision:

```text
Do not run a large batch of direct "single-machine checkpoint as deploy regression model" experiments
as if they were definitive. That test would mix artifact incompleteness with model quality and could
produce misleading results.

The next useful step is to port the missing deployment capability:

1. add deploy-side full_model route support;
2. add an auto_v2_specialist calibration fitter that saves routing_config, full_model.pth, and
   specialists/class_{id}.pth;
3. rerun target-specific calibration packages with the same T1b global classifier and the current
   Flower source regression checkpoint.
```

Next minimal experiment after code support:

```text
B2:
classifier = T1b global server_checkpoint
regression base = Flower source regression checkpoint
target calibration = deploy auto_v2_specialist with full/phase/specialist support
clients = C3, C4, C5

Purpose:
Isolate whether the A2 gap mainly comes from incomplete deployment calibration logic. If B2 approaches
the single-machine T2d metrics, the regression checkpoint is acceptable and the missing piece was
auto_v2/full/specialist packaging. If B2 remains far from T2d, the source regression checkpoint or
training config differs from the single-machine source regression path.
```

## 2026-06-06 B2 Code Support And Smoke Test

Implemented:

```text
gaps_deploy/deploy_config.py
- added optional full_model_checkpoint.

gaps_deploy/build_package.py
- added --full-model-ckpt.
- copies models/full_model.pth when provided.

gaps_deploy/inference.py
- loads optional full_model.pth.
- if routing_config selected_modes[class] == "full", routes that class through full_model.
- specialist_full still overrides through models/specialists/class_{id}.pth.

gaps_flower/specialist_calibration_fit.py
- added --calibration-mode auto_v2_specialist.
- added --split-by class_concentration.
- auto_v2 candidates: none, bias_only, affine_only, phase_affine_only, full.
- per-class validation selection by gate metric.
- specialist_full gate for selected classes.
- exports routing_config.json, calibration_stats.json, full_model.pth, specialists/class_{id}.pth.
```

Important audit finding:

```text
Local single-machine T2d regression client checkpoints cannot be directly used as deployment
regression checkpoints for the current Flower deploy model. Their regression head dimensions differ:
T2d reg_heads use wider layers such as [256,128,64,32], while the current deploy regression model
uses [128,64,32,1]. This explains why direct load raises size mismatch errors.

Therefore direct "use T2d client checkpoint in deploy package" is not a valid experiment unless the
deploy model config is also reconstructed from the T2d architecture.
```

Smoke test:

```text
Created a temporary same-architecture regression checkpoint:
results/_codex_smoke_base_reg.pt

Ran:
specialist_calibration_fit --calibration-mode auto_v2_specialist --steps 1 --full-steps 1

Produced:
results/_codex_smoke_autov2/routing_config.json
results/_codex_smoke_autov2/full_model.pth
results/_codex_smoke_autov2/specialists/class_1.pth

Built package and ran predict_client_file successfully on client_3 test split.
The smoke metrics are not meaningful because the regression checkpoint was random; the purpose was
only to verify code flow, artifact layout, and inference compatibility.
```

Next real experiment:

```text
B2-cloud:
Use the real Flower regression checkpoint on Aliyun, T1b global classifier, and the new
auto_v2_specialist deployment fitter. This isolates whether the A2 gap comes from incomplete
deployment calibration logic rather than the regression base checkpoint.
```

## 2026-06-06 B2-Cloud Result

Comparison against A2:

```text
C3_A2_E3:     MAE=21.9429, RMSE=32.9109, NRMSE=0.1816, cls_acc=0.9882
C3_B2_autov2: MAE=22.2935, RMSE=34.0630, NRMSE=0.1974, cls_acc=0.9882

C4_A2_E3:     MAE=24.7434, RMSE=34.6664, NRMSE=0.1955, cls_acc=0.9969
C4_B2_autov2: MAE=28.6804, RMSE=42.6337, NRMSE=0.2124, cls_acc=0.9969

C5_A2_E3:     MAE=29.1884, RMSE=44.5326, NRMSE=0.2468, cls_acc=0.9531
C5_B2_autov2: MAE=25.8192, RMSE=42.0572, NRMSE=0.2446, cls_acc=0.9531
```

Per-gas interpretation:

```text
C3:
Ethanol improves 14.08 -> 9.44, Methane improves 27.67 -> 21.24.
CO worsens 34.07 -> 39.67, Ethylene worsens 11.95 -> 18.82.
Overall slightly worsens.

C4:
Ethanol improves 25.03 -> 13.01.
CO worsens 33.18 -> 55.66, Ethylene worsens 7.28 -> 12.44.
Overall clearly worsens despite near-perfect classification.

C5:
Overall improves, mainly from Methane 36.68 -> 22.27 and small Ethanol improvement.
CO and Ethylene do not improve; CO 47.78 -> 48.29, Ethylene 15.33 -> 16.68.
```

Decision:

```text
B2 code path is valid, but B2 calibration policy should not replace A2/E3 globally.
The new auto_v2/full candidate is useful for some gases and clients, especially Ethanol/Methane and
C5 overall, but it harms CO/Ethylene on C3/C4. Therefore the remaining gap is not simply "missing
full_model support"; it is candidate selection/gating instability under a small calibration split.

The current B2 uses validation R2 as the gate metric, while deployment priority is MAE/RMSE/NRMSE.
This mismatch likely explains why validation-selected full/specialist routes can worsen test MAE.
```

Next controlled experiment:

```text
B3:
Keep T1b global classifier and Flower regression base.
Run auto_v2_specialist again, but gate by NRMSE_range or RMSE instead of R2.
Then compare A2_E3 vs B3_NRMSE. If B3 keeps the C5 improvement without hurting C3/C4, the issue was
metric mismatch. If B3 still hurts C3/C4, use a hybrid selection policy: retain A2/E3 for C3/C4 and
use B2/B3 only for C5 or only for classes whose validation and test trends agree.
```

## 2026-06-06 B3 Result: NRMSE-Gated AutoV2

Comparison:

```text
C3_A2:       MAE=21.9429, RMSE=32.9109, NRMSE=0.1816
C3_B2_R2:    MAE=22.2935, RMSE=34.0630, NRMSE=0.1974
C3_B3_NRMSE: MAE=22.2935, RMSE=34.0630, NRMSE=0.1974

C4_A2:       MAE=24.7434, RMSE=34.6664, NRMSE=0.1955
C4_B2_R2:    MAE=28.6804, RMSE=42.6337, NRMSE=0.2124
C4_B3_NRMSE: MAE=28.6804, RMSE=42.6337, NRMSE=0.2124

C5_A2:       MAE=29.1884, RMSE=44.5326, NRMSE=0.2468
C5_B2_R2:    MAE=25.8192, RMSE=42.0572, NRMSE=0.2446
C5_B3_NRMSE: MAE=25.8192, RMSE=42.0572, NRMSE=0.2446
```

Interpretation:

```text
B3 is numerically identical to B2. Changing the gate metric from R2 to NRMSE_range did not change
the final deployed routing decision. Therefore the degradation on C3/C4 is not solved by a simple
metric swap. The auto_v2/full route itself is client/class dependent: it helps some classes
(Ethanol/Methane, especially C5 overall) but hurts CO/Ethylene on C3/C4.

Decision:
Do not continue broad auto_v2 metric sweeps. The evidence supports a conservative hybrid deployment
policy:

1. keep A2/E3 as the default package for C3 and C4;
2. use B2/B3 auto_v2 package only for C5 if the target is client_5-like;
3. before enabling full/phase/full-specialist on any future target, require validation improvement
   and a route-level sanity check on CO/Ethylene.
```

Current best deployment candidates:

```text
C3: A2/E3 package, MAE=21.9429, RMSE=32.9109, NRMSE=0.1816
C4: A2/E3 package, MAE=24.7434, RMSE=34.6664, NRMSE=0.1955
C5: B2/B3 auto_v2 package, MAE=25.8192, RMSE=42.0572, NRMSE=0.2446

All use T1b global classifier as the shared route model.
```

Open issues:

```text
1. Regression gap to single-machine remains:
   single-machine calibrated C3/C4/C5 MAE = 12.81 / 12.27 / 19.19
   current best deploy C3/C4/C5 MAE = 21.94 / 24.74 / 25.82

2. Current Flower regression checkpoint/config does not match the stronger single-machine T2d
   regression setup. Local T2d checkpoints have wider regression heads and cannot be directly loaded
   by the current deploy model architecture.

3. AutoV2 candidate selection is unstable on small target calibration splits. Full/phase candidates
   can improve one class while damaging CO/Ethylene.

4. QC is still mostly lenient and accepts almost all windows, so current regression metrics are
   essentially pre-QC deployment metrics.

5. model_version still logs as unknown for T1b classifier checkpoints unless wrapped with a
   model_version field.
```

## 2026-06-07 Regression Architecture Alignment

Finding:

```text
The regression architecture mismatch is now localized.

Single-machine config.py default:
REG_HEAD_DEPTH = 4

Flower deployment regression_task.py previously forced:
REG_HEAD_DEPTH = 3

T2d checkpoint shapes:
reg_heads.0.fc1.weight = [256, 64]
reg_heads.0.fc2.weight = [128, 256]
reg_heads.0.fc3.weight = [64, 128]
reg_heads.0.fc4.weight = [32, 64]
reg_heads.0.fc5.weight = [1, 32]

Flower deploy depth=3 shapes:
reg_heads.0.fc1.weight = [128, 64]
reg_heads.0.fc2.weight = [64, 128]
reg_heads.0.fc3.weight = [32, 64]
reg_heads.0.fc4.weight = [1, 32]
```

Code update:

```text
gaps_deploy/build_package.py
- added --reg-head-depth.
- writes model_config["reg_head_depth"].

gaps_deploy/inference.py
- maps model_config["reg_head_depth"] to FLConfig.REG_HEAD_DEPTH before creating regression models.

gaps_flower/specialist_calibration_fit.py
- added --reg-head-depth, applied before creating the regression model.
```

Smoke validation:

```text
Built a deploy package using:
regression checkpoint = results/T2d_reg_classconc_split_src12_tgt345/checkpoints/separate_regression/separate_regression_client3.pth
reg_head_depth = 4

The package loads and runs predict_client_file successfully, so depth=4 deployment construction is
valid. However, the smoke metric is not a valid reproduction of T2d because the saved client
checkpoint contains only model_state/classifier_state/metrics. The single-machine runtime full_model
and specialist_models were not saved as separate deployable files.
```

Next controlled experiment:

```text
B4:
Use the single-machine source regression base checkpoint rather than the already-calibrated T2d
client checkpoint:

classifier = T1b global classifier
regression base = results/T2c_source_reg_base_src12_tgt345/checkpoints/separate_regression/separate_regression_source.pth
reg_head_depth = 4
target calibration = deploy auto_v2_specialist

Purpose:
Test whether the remaining deployment gap is mainly caused by the weaker Flower depth=3 regression
base. If B4 approaches T2d, the deployment inference/calibration chain is validated and the Flower
regression training should be changed to depth=4. If B4 still lags, the remaining gap is likely in
auto_v2_specialist training details or unsaved single-machine runtime artifacts.
```

## 2026-06-07 B4 Depth-4 Source Regression Deployment

Experiment:

```text
B4 uses:
classifier = T1b global classifier
regression base = T2c single-machine source regression checkpoint
reg_head_depth = 4
calibration = deploy auto_v2_specialist
gate_metric = R2
split_by = class_concentration
```

Result:

```text
C3_B4_T2cDepth4:
R2=0.8871, MAE=14.5185, RMSE=22.3020, NRMSE=0.1401, cls_acc=0.9882

C4_B4_T2cDepth4:
R2=0.9185, MAE=11.6044, RMSE=18.9464, NRMSE=0.1075, cls_acc=0.9969

C5_B4_T2cDepth4:
R2=0.7066, MAE=19.8041, RMSE=35.9468, NRMSE=0.2161, cls_acc=0.9531
```

Comparison against previous best deploy packages:

```text
C3: MAE 21.9429 -> 14.5185, RMSE 32.9109 -> 22.3020, NRMSE 0.1816 -> 0.1401
C4: MAE 24.7434 -> 11.6044, RMSE 34.6664 -> 18.9464, NRMSE 0.1955 -> 0.1075
C5: MAE 25.8192 -> 19.8041, RMSE 42.0572 -> 35.9468, NRMSE 0.2446 -> 0.2161
```

Interpretation:

```text
B4 is the strongest deployment result so far. It validates that the deploy package can correctly
load a depth-4 regression base, apply target-side auto_v2_specialist calibration, route through the
T1b classifier, and export stable JSON/CSV predictions.

The main previous regression gap was not caused by the classification route or the inference/QC
wrapper. It was primarily caused by using a weaker/mismatched Flower depth-3 regression base.

Compared with the single-machine calibrated reference:
C4 is already essentially at the single-machine MAE/RMSE level.
C5 is close in MAE, but RMSE/NRMSE still leave room because CO remains hard.
C3 improves strongly but still lags the single-machine result, mainly in CO.
```

Decision:

```text
Use B4 as the current best deployment package for C3/C4/C5.

Stop spending time on Flower depth-3 regression packages. The next useful migration step is to make
Flower regression training produce depth-4 source checkpoints by default, or expose reg_head_depth in
the Flower regression server/client CLI so future cloud-edge regression checkpoints match the
single-machine T2c/T2d architecture.

Run only targeted follow-up checks:
1. route-level CO/Ethylene error analysis for B4;
2. depth-4 Flower regression smoke/full run;
3. deployment summary table against the single-machine reference.
```

## 2026-06-07 Flower Regression Depth-4 Alignment Patch

Problem:

```text
B4 proved that a depth-4 source regression checkpoint can recover much of the single-machine
regression performance after deployment calibration. However, the Flower regression client/server
entry points still created regression models through make_regression_config() with an older forced
REG_HEAD_DEPTH=3. That means future cloud-edge regression checkpoints would again be architecture
mismatched unless depth is explicitly aligned.
```

Code update:

```text
gaps_flower/regression_task.py
- make_regression_config(..., reg_head_depth=None)
- default now follows config.py / FLConfig.REG_HEAD_DEPTH, currently 4
- explicit reg_head_depth can still reproduce old depth=3 runs

gaps_flower/regression_client.py
- added --reg-head-depth
- logs REG_HEAD_DEPTH
- saves model_config.reg_head_depth in local client checkpoints

gaps_flower/regression_server.py
- added --reg-head-depth
- creates the global aggregation model with the same depth
- checks local checkpoint model_config.reg_head_depth before FedAvg
- saves model_config.reg_head_depth in regression_fedavg_global.pt and regressor.pt
```

Validation:

```text
python -m py_compile gaps_flower/regression_task.py gaps_flower/regression_client.py gaps_flower/regression_server.py

make_regression_config() default REG_HEAD_DEPTH = 4
make_regression_config(reg_head_depth=3) override REG_HEAD_DEPTH = 3
```

Next experiment:

```text
Train Flower source regression with depth=4 on source clients 1/2, aggregate it, then repeat the B4
deployment calibration/evaluation using this Flower-produced depth-4 regression checkpoint.

Purpose:
separate two hypotheses:
1. deployment package and target calibration are already valid (supported by B4);
2. Flower source regression training can produce a depth-4 checkpoint close to the single-machine
   T2c source checkpoint (next verification).
```

## 2026-06-07 T3a Flower Depth-4 Regression Result

Experiment:

```text
classifier = T1b global classifier
regression base = results/T3a_flower_reg_depth4_src12/regression_fedavg_global.pt
reg_head_depth = 4
source clients = 1,2
target calibration = auto_v2_specialist, same as B4
```

Result:

```text
B4 single-machine T2c source checkpoint:
C3: MAE=14.5185, RMSE=22.3020, NRMSE=0.1401
C4: MAE=11.6044, RMSE=18.9464, NRMSE=0.1075
C5: MAE=19.8041, RMSE=35.9468, NRMSE=0.2161

T3a Flower depth-4 source checkpoint:
C3: MAE=18.8482, RMSE=30.7114, NRMSE=0.1780
C4: MAE=19.8897, RMSE=27.4120, NRMSE=0.1728
C5: MAE=27.4660, RMSE=43.1250, NRMSE=0.2571
```

Interpretation:

```text
T3a verifies syntax, checkpoint shape, deployment packaging, and calibration compatibility for a
Flower-produced depth-4 regression checkpoint. However, it does not reproduce B4 performance.

The remaining gap is now localized to Flower regression source training quality, not deployment
inference, classification routing, or target-side calibration loading.

The degradation is strongest on CO:
C3 CO MAE=31.58 vs B4 CO MAE=23.80
C4 CO MAE=27.37 vs B4 CO MAE=15.13
C5 CO MAE=53.82 vs B4 CO MAE=35.50

This suggests the source regression base has a weaker gas-specific concentration mapping before
target calibration. The target calibration can correct affine/class-specific bias, but it cannot
fully compensate for a weaker learned regression representation.
```

Decision:

```text
Keep B4 as the current best deployable package.

Do not tune target calibration first. The same calibration recipe performs well with the T2c source
checkpoint, so the bottleneck is upstream source regression training.

Next step should compare Flower regression training details against exp_improved/T2c:
1. number of source regression optimization steps;
2. whether the encoder is actually frozen or allowed to update;
3. exact trainable parameter set;
4. class weights / per-class Huber settings;
5. whether the source regression checkpoint was trained with multi-round FedAvg or a stronger
   centralized/separate-regression routine.
```

## 2026-06-07 Regression Trainable-Parameter Bug Fix

Finding:

```text
train_regression_local() froze all parameters first, then collected regression-module parameters
with:

p for p in module.parameters() if p.requires_grad is not False

Because all parameters had already been set to requires_grad=False, this filter excluded
reg_proj/reg_transformer/reg_attn/reg_attn_linear/reg_stats_proj from the optimizer. In practice,
T3a likely trained mainly reg_heads and prototype scalar parameters. This explains why depth=4
architecture was aligned but the source regression representation stayed weaker than B4/T2c.
```

Code update:

```text
gaps_flower/regression_task.py
- collect all parameters from regression modules before setting requires_grad=True.
- log trainable regression parameter count at the start of local regression training.
- FedAvg denominator now uses only clients that actually provide local checkpoints.

gaps_flower/specialist_calibration_fit.py
- --reg-head-depth default changed from 3 to None, so calibration follows config.py/FLConfig
  unless explicitly overridden.
```

Validation:

```text
python -m py_compile gaps_flower/regression_task.py gaps_flower/regression_client.py gaps_flower/regression_server.py gaps_flower/specialist_calibration_fit.py

make_regression_config() default REG_HEAD_DEPTH = 4
make_regression_config(reg_head_depth=3) override REG_HEAD_DEPTH = 3
```

Next experiment:

```text
T3b-small:
Use the same setup as T3a, keep client steps=500, and only apply the trainable-parameter bug fix.

Interpretation:
If T3b-small improves clearly over T3a, the main source-regression bottleneck was the optimizer
parameter set.
If T3b-small is still far from B4, move to T3b-full: more steps plus T2c loss/class-weight alignment.
```

## 2026-06-07 T3b-Small Training/Aggregation Status

Observation:

```text
T3b-small source regression client training completed with the trainable-parameter fix.

client1:
REG_HEAD_DEPTH=4
trainable regression params=403209
steps=500
avg_loss=0.037964

client2:
REG_HEAD_DEPTH=4
trainable regression params=403209
steps=500
avg_loss=0.031645

server FedAvg:
clients=1,2
sample_counts={1:2360, 2:2360}
saved results/T3bsmall_flower_reg_depth4_src12/regression_fedavg_global.pt
```

Interpretation:

```text
The optimizer parameter-set bug is fixed operationally: regression training now includes the full
regression branch rather than mainly reg_heads. The next required step is target calibration and
deployment evaluation on clients 3/4/5. Only after that can T3b-small be compared with T3a/B4.
```

## 2026-06-07 T3b-Small Deployment Result

Experiment:

```text
T3b-small = T3a setup + regression trainable-parameter bug fix
client steps = 500
reg_head_depth = 4
target calibration = auto_v2_specialist, same as B4/T3a
```

Result:

```text
B4:
C3 MAE=14.5185, RMSE=22.3020, NRMSE=0.1401
C4 MAE=11.6044, RMSE=18.9464, NRMSE=0.1075
C5 MAE=19.8041, RMSE=35.9468, NRMSE=0.2161

T3a:
C3 MAE=18.8482, RMSE=30.7114, NRMSE=0.1780
C4 MAE=19.8897, RMSE=27.4120, NRMSE=0.1728
C5 MAE=27.4660, RMSE=43.1250, NRMSE=0.2571

T3b-small:
C3 MAE=14.2441, RMSE=24.4676, NRMSE=0.1468
C4 MAE=14.0871, RMSE=20.7681, NRMSE=0.1165
C5 MAE=20.0544, RMSE=37.2393, NRMSE=0.2326
```

Interpretation:

```text
The trainable-parameter bug fix is confirmed as a major cause of the T3a gap.

Compared with T3a:
C3 MAE improves by 4.6041 ppm and RMSE by 6.2438 ppm.
C4 MAE improves by 5.8026 ppm and RMSE by 6.6439 ppm.
C5 MAE improves by 7.4116 ppm and RMSE by 5.8857 ppm.

Compared with B4:
C3 has slightly lower MAE than B4 but higher RMSE/NRMSE, meaning the average error is competitive
but some larger errors remain.
C5 is very close to B4 in MAE, with RMSE still slightly worse.
C4 remains behind B4, mainly because CO is still worse.
```

Per-gas notes:

```text
C3 T3b-small:
CO MAE=24.70, close to B4 CO MAE=23.80.
Methane improves strongly compared with T3a.

C4 T3b-small:
CO MAE=22.03, still worse than B4 CO MAE=15.13.
Ethylene is good, MAE=6.66.

C5 T3b-small:
CO MAE=35.38, close to B4 CO MAE=35.50.
Ethylene degrades versus B4, with R2 below 0. This should be checked before making T3b-small a
general replacement for B4.
```

Decision:

```text
T3b-small proves Flower depth-4 regression can approach the single-machine T2c deployment baseline
once the full regression branch is trained.

Keep B4 as the current best deployment package because it still has the best RMSE/NRMSE on all
clients except C3 MAE. Use T3b-small as the best Flower-produced regression checkpoint so far.

Next step should not return to broad calibration sweeps. The remaining gap is narrower and mostly
class/client specific:
1. C4 CO still needs source regression improvement.
2. C5 Ethylene has a negative R2 after T3b-small and needs route-level diagnosis.
3. A T3b-full run with more steps and source-side class weighting/per-class Huber is now justified.
```

## 2026-06-07 T3b-Full Deployment Result

Experiment:

```text
T3b-full = T3b-small + longer source regression training
client steps = 2000
reg_head_depth = 4
lr = 0.0005
target calibration = auto_v2_specialist, same recipe as B4/T3a/T3b-small
```

Source training:

```text
client1 avg_loss: 0.037964 (T3b-small, 500 steps) -> 0.019043 (T3b-full, 2000 steps)
client2 avg_loss: 0.031645 (T3b-small, 500 steps) -> 0.015158 (T3b-full, 2000 steps)
```

Deployment comparison:

```text
B4:
C3 MAE=14.5185, RMSE=22.3020, NRMSE=0.1401
C4 MAE=11.6044, RMSE=18.9464, NRMSE=0.1075
C5 MAE=19.8041, RMSE=35.9468, NRMSE=0.2161

T3b-small:
C3 MAE=14.2441, RMSE=24.4676, NRMSE=0.1468
C4 MAE=14.0871, RMSE=20.7681, NRMSE=0.1165
C5 MAE=20.0544, RMSE=37.2393, NRMSE=0.2326

T3b-full:
C3 MAE=11.6609, RMSE=19.5145, NRMSE=0.1252
C4 MAE=12.7326, RMSE=19.8474, NRMSE=0.1145
C5 MAE=18.0579, RMSE=33.8927, NRMSE=0.2200
```

Interpretation:

```text
T3b-full confirms that source regression training strength is a real limiting factor.

Compared with T3b-small:
C3 improves in MAE/RMSE/NRMSE.
C4 improves in MAE/RMSE/NRMSE.
C5 improves in MAE/RMSE, while NRMSE remains slightly higher than B4.

Compared with B4:
C3 is now better than B4 on MAE/RMSE/NRMSE.
C5 is better than B4 on MAE/RMSE, but slightly worse on NRMSE.
C4 is close but still behind B4.
```

Per-gas notes:

```text
C3 T3b-full:
CO MAE=18.26, much better than B4 CO MAE=23.80.
Methane MAE=9.86, also better than B4.

C4 T3b-full:
CO MAE=16.79, close to B4 CO MAE=15.13.
Ethylene MAE=5.55, better than B4 Ethylene MAE=6.03.
Methane remains the main residual at MAE=16.02.

C5 T3b-full:
Overall MAE/RMSE beats B4, but CO remains high at MAE=36.05 and Ethylene still has negative R2.
The negative Ethylene R2 suggests error distribution/variance mismatch even though MAE is moderate.
```

Decision:

```text
T3b-full is now the best Flower-produced regression checkpoint and a credible replacement candidate
for B4 on C3 and C5.

Current best by client:
C3: T3b-full
C4: B4 remains slightly better overall, but T3b-full is close.
C5: T3b-full is better on MAE/RMSE; B4 slightly better on NRMSE.

Next step should be targeted rather than broad:
1. keep T3b-full as the new Flower regression baseline;
2. diagnose C4 Methane and C5 CO/Ethylene;
3. only then consider class-specific loss weighting or per-class Huber.
```

## 2026-06-07 T3b-Full Route Error Analysis

Experiment:

```text
Analyze T3b-full deployment prediction CSVs for C4 and C5.
Goal: separate classification-route errors from route-correct regression/calibration errors.
```

C4:

```text
overall: MAE=12.73, RMSE=19.85, Bias=-3.86
route-correct only: n=319/320, MAE=12.29, RMSE=17.95, Bias=-3.39

Ethanol: route_acc=1.0000, MAE=12.56, Bias=-4.99
CO: route_acc=1.0000, MAE=16.79, Bias=+1.14
Ethylene: route_acc=1.0000, MAE=5.55, Bias=-0.95
Methane: route_acc=0.9875, all MAE=16.02, route-correct MAE=14.29, Bias=-8.83
```

C4 interpretation:

```text
C4 is not a classification-route problem. Only one window is misrouted.
Remaining error is regression/calibration error under correct routing.
Methane has a clear negative bias, indicating underprediction after calibration.
CO is close to B4 but still not better.
```

C5:

```text
overall: MAE=18.06, RMSE=33.89, Bias=+4.75
route-correct only: n=305/320, MAE=14.80, RMSE=24.54, NRMSE=0.1336
route-wrong only: n=15/320, MAE=84.40, RMSE=110.73, NRMSE=0.8185

Ethanol: route_acc=0.9750, all MAE=12.99, route-correct MAE=11.70
CO: route_acc=0.9125, all MAE=36.05, route-correct MAE=32.54, route-wrong MAE=72.60
Ethylene: route_acc=0.9500, all R2=-0.0226, all MAE=12.34, route-correct R2=0.9328, route-correct MAE=5.74
Methane: route_acc=0.9750, all MAE=10.86, route-correct MAE=10.10
```

C5 interpretation:

```text
C5 is materially affected by classification routing errors.
When the route is correct, overall MAE drops from 18.06 to 14.80 and RMSE drops from 33.89 to 24.54.

The negative Ethylene R2 is not a weak Ethylene regressor problem. It is caused by four Ethylene
windows routed as CO. Under correct routing, Ethylene R2=0.9328 and MAE=5.74.

CO remains hard even when route-correct: route-correct MAE=32.54. This is a true regression/calibration
residual, while the wrong-route subset further amplifies the tail error.
```

Decision:

```text
Do not tune C5 Ethylene regression. The Ethylene issue is primarily routing, not regression.

For C5, improving or calibrating the classifier route/confidence policy will produce larger gains
than changing Ethylene regression calibration. CO still needs regression/calibration attention.

For C4, route correction will not help much. The next useful intervention is class-specific
calibration/bias correction for Methane and possibly CO.
```

## 2026-06-07 Route-Confidence QC Diagnostic Tool

Motivation:

```text
T3b-full route analysis showed two different residual mechanisms:

C4:
route accuracy is almost perfect, so confidence-based route QC will not materially improve most
errors. C4 needs class-specific regression/calibration diagnosis.

C5:
15/320 windows are misrouted. These wrong-route windows have MAE=84.40 and RMSE=110.73, while
route-correct windows have MAE=14.80 and RMSE=24.54. Therefore a route-confidence review/reject
policy could reduce deployed tail errors if it can identify a reasonable fraction of wrong routes
without flagging too many correct windows.
```

Code update:

```text
Added gaps_deploy/analyze_route_confidence_thresholds.py

Inputs:
- prediction CSV from predict_client_file
- confidence thresholds
- confidence_margin thresholds

Outputs:
- JSON/CSV sweep report
- for each threshold rule:
  flag_rate
  route_wrong_recall
  route_correct_flag_rate
  kept-window MAE/RMSE/R2
```

Validation:

```text
python -m py_compile gaps_deploy/analyze_route_confidence_thresholds.py
python -m gaps_deploy.analyze_route_confidence_thresholds --help
```

Next experiment:

```text
Run the confidence-threshold sweep on C5 T3b-full first.

If low confidence or low margin catches most wrong routes with flag_rate <= 10-15%, then we can add
a deployment QC policy that sends low-route-confidence windows to review.

If confidence/margin cannot identify wrong routes, then C5 needs classifier calibration/target-domain
classification adaptation rather than a simple QC threshold.
```

## 2026-06-07 Route-Confidence Threshold Sweep Result

Experiment:

```text
Sweep confidence and confidence_margin thresholds on T3b-full deployment predictions for C5 and C4.
Criterion:
flag_rate <= 10-15%, high route_wrong_recall, and clearly lower kept_RMSE means route-confidence QC
is useful.
```

C5 result:

```text
baseline:
route_wrong_total=15/320
MAE=18.0579
RMSE=33.8927
route_accuracy=0.9531

best reported low-flag rule:
confidence < 0.98
flag_rate=0.0219
route_wrong_recall=0.2667
route_correct_flag_rate=0.0098
kept_MAE=17.6299
kept_RMSE=33.5597
kept_route_acc=0.9649

confidence < 0.90 or margin < 0.80:
flag_rate=0.0156
route_wrong_recall=0.2000
route_correct_flag_rate=0.0066
kept_MAE=17.6959
kept_RMSE=33.5624
kept_route_acc=0.9619
```

C5 interpretation:

```text
Simple softmax confidence/margin thresholds do not catch enough wrong-route windows.
The best rule catches only 4/15 wrong routes and reduces RMSE by only about 0.33 ppm.

This fails the decision criterion. Route-confidence QC is not sufficient as a standalone deployment
guard for C5.
```

C4 result:

```text
baseline:
route_wrong_total=1/320
MAE=12.7326
RMSE=19.8474
route_accuracy=0.9969

All scanned confidence/margin thresholds up to confidence<0.98 and margin<0.8 flag 0 windows.
```

C4 interpretation:

```text
C4 predictions are extremely confident and almost always routed correctly. Confidence-threshold QC is
irrelevant for C4; remaining error is regression/calibration.
```

Decision:

```text
Do not implement a simple confidence/margin QC rule as the next deployment change.

For C5, the classifier is overconfident on many wrong-route windows. The next useful direction is:
1. target-side classification calibration or adaptation for C5;
2. response-signature/range-risk QC rather than softmax-only QC;
3. inspect wrong-route windows directly to see whether they cluster by phase or concentration.

For C4, focus on class-specific regression calibration, especially Methane bias.
```

## 2026-06-07 Gate Mechanisms and Generic QC Risk Sweep

Gate mechanism definition:

```text
There are two gates in the current deployment logic.

1. Calibration candidate gate:
auto_v2_specialist trains multiple candidate mappings/models for each class and target client
(none, bias_only, affine_only, phase_affine_only, full, specialist where enabled). A candidate is
selected only if it improves the chosen validation metric. This prevents a specialist/full model
from being enabled when it overfits the calibration subset or damages a class.

2. Deployment QC gate:
After prediction and calibration, each window is assigned risk scores. The QC policy decides whether
the window is automatically accepted, sent to review, or rejected. This gate does not change the
prediction; it controls trust and deployment action.
```

Rationale:

```text
The confidence-threshold sweep showed that softmax confidence/margin alone is not sufficient for C5.
However, the current QC layer also contains response- and route-risk scores based on sensor response
signatures and calibration references. These are more physically meaningful for gas sensors because
wrong routes may still have high neural-network confidence but inconsistent response morphology.
```

Code update:

```text
Added gaps_deploy/analyze_qc_risk_sweep.py

The script sweeps existing CSV columns:
- risk_score
- risk_classifier_uncertainty
- risk_margin_risk
- risk_response_signature_norm
- risk_response_conc_gap_norm
- risk_response_mean_conc_gap_norm
- risk_class_response_rank_risk
- risk_class_response_margin_risk
- risk_route_response_risk
- risk_composite_response_risk

For each score and flag_rate, it reports:
- kept-window MAE/RMSE/R2
- route_wrong_recall
- high_error_recall
- route_correct_flag_rate
```

Validation:

```text
python -m py_compile gaps_deploy/analyze_qc_risk_sweep.py
python -m gaps_deploy.analyze_qc_risk_sweep --help
```

Next experiment:

```text
Run the generic risk sweep on T3b-full C3/C4/C5 prediction CSVs.

Decision rule:
If a risk score reduces kept RMSE substantially at <=10-15% flag_rate and has acceptable
high_error/route_wrong recall, it can be considered for a deployment QC policy.

If a score helps only one client, keep it as client-specific policy evidence rather than global QC.

If no score catches C5 wrong-route/high-error windows, then the system needs stronger response
anchoring or target-side classification adaptation.
```

## 2026-06-07 Generic QC Risk Sweep Result

Experiment:

```text
Run analyze_qc_risk_sweep.py on T3b-full C3/C4/C5 prediction CSVs.
Flag rates tested: 2%, 5%, 10%, 15%, 20%.
High-error windows are defined as top 10% absolute-error windows for each client.
```

Result summary:

```text
C3 baseline:
MAE=11.6609, RMSE=19.5145, route_wrong=8/680

C3 at 10% flag rate:
risk_score / risk_classifier_uncertainty / risk_route_response_risk / risk_composite_response_risk
kept_MAE=10.3589, kept_RMSE=15.3810
route_wrong_recall=1.0000
high_error_recall=0.2794
route_correct_flag_rate=0.0893

C4 baseline:
MAE=12.7326, RMSE=19.8474, route_wrong=1/320

C4 at 10% flag rate:
risk_score / risk_classifier_uncertainty / risk_route_response_risk / risk_composite_response_risk
kept_MAE=11.6325, kept_RMSE=16.7200
route_wrong_recall=1.0000
high_error_recall=0.1875
route_correct_flag_rate=0.0972

C5 baseline:
MAE=18.0579, RMSE=33.8927, route_wrong=15/320

C5 at 10% flag rate:
risk_score kept_MAE=15.5804, kept_RMSE=28.6409, route_wrong_recall=0.6667
risk_margin_risk kept_MAE=15.3024, kept_RMSE=27.6289, route_wrong_recall=0.6000

C5 at 15% flag rate:
risk_score kept_MAE=13.6921, kept_RMSE=20.8627, route_wrong_recall=0.8000
high_error_recall=0.4375
route_correct_flag_rate=0.1180
```

Interpretation:

```text
The generic QC sweep is positive: sending the highest-risk 10-15% windows to review can materially
reduce accepted-window RMSE on all clients.

C5 benefits the most:
RMSE=33.89 baseline -> 28.64 at 10% flag -> 20.86 at 15% flag.
This confirms that QC is useful for controlling the C5 tail-error problem.

C3/C4 also improve, but their baseline is already strong. A 10% review rate is a reasonable
conservative operating point.
```

Important caveat:

```text
The best scores are numerically identical or nearly identical across:
risk_score
risk_classifier_uncertainty
risk_route_response_risk
risk_composite_response_risk

This means the current deployed risk signal is still dominated by classifier uncertainty/margin.
Response-signature risk is not yet contributing strongly, likely because the deployment package does
not include full response reference statistics or those scores are zero/weak in current outputs.

Therefore this experiment validates the QC-layer concept, but not yet the full response-risk design.
```

Decision:

```text
Use this result as evidence for a two-threshold QC policy candidate:

Conservative global candidate:
flag top 10% by risk_score or risk_margin_risk for review.

Client-specific C5 candidate:
flag top 15% by risk_score/risk_margin_risk for review because C5 has stronger wrong-route and
tail-error risk.

Do not finalize this as the only QC policy yet. Next, inspect whether response reference files can be
exported into the deploy package so response_signature/rank/margin risks can be evaluated properly.
```

## 2026-06-07 Response Reference Export for QC

Finding:

```text
DeployPredictor already has a hook to load response references from calibration_stats.json, and
RiskScoreComputer can compute response_signature_norm / response_conc_gap_norm when refs are present.

However, specialist_calibration_fit.py did not export response_refs. As a result, deployed packages
logged that calibration_stats did not contain response refs, and response-based QC risks were skipped
or effectively zero. The previous QC sweep therefore mainly validated classifier-risk QC, not the
full response-risk design.
```

Code update:

```text
gaps_flower/specialist_calibration_fit.py
- added _build_response_refs(train_loader)
- computes per-class time-mean response signature references:
  center, scale, z_sigs, loocv_p90, rows(concentration, phase)
- writes response_refs into calibration_stats.json for both auto_v2_specialist and specialist_gated

gaps_deploy/inference.py
- _load_calibration_refs now supports calibration_stats["response_refs"]
- keeps backward compatibility with old top-level {class_id: ref} format
```

Validation:

```text
python -m py_compile gaps_flower/specialist_calibration_fit.py gaps_deploy/inference.py

Smoke-loaded a temporary calibration_stats.json with response_refs and verified:
class id parsed, center shape parsed, z_sigs shape parsed.
```

Next experiment:

```text
Re-run target calibration and deployment package build for T3b-full using the updated
specialist_calibration_fit.py and inference.py. Then re-run prediction and QC risk sweep.

Expected check:
deployment logs should say that calibration reference data are loaded, not skipped.

Purpose:
compare classifier-only QC risk against response-enabled QC risk and decide whether response-risk
should be part of the final deployment QC policy.
```

## 2026-06-07 Response-Enabled QC Risk Sweep Result

Experiment:

```text
Rebuilt T3b-full deployment packages with calibration_stats["response_refs"].
Prediction metrics are expected to stay unchanged because response refs only affect QC risk scores,
not classification/regression/calibration outputs.
```

Validation:

```text
Deployment logs now show:
加载校准参考数据: .../calibration/calibration_stats.json

The previous "calibration_stats does not contain response refs" skip message is gone.
```

Base prediction metrics:

```text
C3: MAE=11.6609, RMSE=19.5145, NRMSE=0.1252, cls_acc=0.9882
C4: MAE=12.7326, RMSE=19.8474, NRMSE=0.1145, cls_acc=0.9969
C5: MAE=18.0579, RMSE=33.8927, NRMSE=0.2200, cls_acc=0.9531
```

QC sweep highlights:

```text
C3:
baseline RMSE=19.5145
top 10% risk_score/composite: kept_RMSE=15.3810, route_wrong_recall=1.0000
top 15% response_conc_gap_norm: kept_RMSE=14.6471, high_error_recall=0.5294
top 20% response_conc_gap_norm: kept_MAE=9.0269, kept_RMSE=14.5305, high_error_recall=0.6324

C4:
baseline RMSE=19.8474
top 10% risk_score/composite: kept_RMSE=15.8503, high_error_recall=0.3125
top 15% risk_score/composite: kept_RMSE=15.1659, high_error_recall=0.5312
top 20% risk_score/composite: kept_RMSE=13.6291, high_error_recall=0.7812

C5:
baseline RMSE=33.8927
top 10% risk_score/composite: kept_RMSE=19.1620, route_wrong_recall=0.8000, high_error_recall=0.5938
top 15% risk_score/composite: kept_RMSE=16.3718, route_wrong_recall=0.8667, high_error_recall=0.8125
top 20% risk_score/composite: kept_RMSE=15.6776, route_wrong_recall=0.8667, high_error_recall=0.9062
```

Interpretation:

```text
Response refs are now active and materially improve QC risk ranking, especially for C4/C5.

Compared with classifier-only QC, C5 improves strongly:
previous top 15% classifier-risk kept_RMSE ≈ 20.86
response-enabled top 15% risk_score/composite kept_RMSE = 16.37

C4 also benefits:
previous top 20% classifier-risk kept_RMSE ≈ 15.80
response-enabled top 20% composite kept_RMSE = 13.63

C3 benefits as well, but the best signal is more mixed:
classifier/routing risk catches all route errors, while response_conc_gap_norm catches more high-error
windows and gives lower MAE/RMSE at higher review rates.
```

Decision:

```text
Response-enabled QC is validated as useful. It should remain part of the deployment design.

Recommended next policy candidates:

Global conservative:
top 10% by risk_score/composite_response_risk -> review.

Client-specific:
C5 top 15% by risk_score/composite_response_risk -> review, because it cuts kept_RMSE from 33.89
to 16.37 and catches 86.7% wrong-route windows.

C4 can use 10-15% global review, or 20% if deployment allows a higher review burden.

Do not use response_conc_gap_norm alone as the final policy yet. It can reduce MAE strongly, but it
does not consistently catch route errors. It is better as one component of composite risk.
```

## 2026-06-08 Deployable QC Policy Tools

Purpose:

```text
Convert the response-enabled QC sweep result into an actual deployment policy, then verify the
policy through real qc_status outputs instead of another offline-only sweep.
```

Code added:

```text
gaps_deploy/build_qc_policy_from_predictions.py
- reads prediction CSV risk columns
- computes quantile thresholds for target review rates
- writes qc/selected_policy.json-compatible policies
- uses runtime score names, e.g. CSV risk_composite_response_risk -> policy composite_response_risk

gaps_deploy/evaluate_qc_status_metrics.py
- reads deployed prediction CSVs with qc_status
- reports all-window, accept-only, and flagged-window MAE/RMSE/NRMSE
- reports route_wrong_recall and high_error_recall for review/reject windows
```

Validation:

```text
python -m py_compile gaps_deploy/build_qc_policy_from_predictions.py gaps_deploy/evaluate_qc_status_metrics.py gaps_deploy/qc_policy.py gaps_deploy/predict_client_file.py
python -m gaps_deploy.build_qc_policy_from_predictions --help
python -m gaps_deploy.evaluate_qc_status_metrics --help
```

Next verification design:

```text
Policy candidate:
ALL: top 10% risk_composite_response_risk -> review
C5: top 15% risk_composite_response_risk -> review
reject disabled for this validation by high_ratio=999.0, because the current stage is to verify
manual-review routing before automatic rejection.

Expected behavior:
raw MAE/RMSE will not change because QC does not repair predictions.
accept-only MAE/RMSE should match the previous kept_MAE/kept_RMSE from the sweep.
review_rate should be close to 10% for C3/C4 and 15% for C5.
```
