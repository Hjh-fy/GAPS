# GAPS IoT-J 最终分类实验闭环设计

日期：2026-08-04
状态：用户已批准设计，等待设计文档复核
分支：`codex/iotj-final-classification-le1`
审计基线：`11cfbba4531a5ab92382e08d5b17fa5a22e936f8`
冻结 source 训练来源：`P0A-PURE-FEDAVG-LE1-S42`

## 1. 目标与边界

本任务为 IEEE IoT-J 分类实验建立完整、可审计的证据闭环，回答以下问题：

1. C1+C2 到 C3/C4/C5 存在多大域偏移；
2. FedAvg、FedProx、canonical SCAFFOLD 的跨设备泛化差异；
3. canonical、无监督、非条件化的 CORAL/MMD/DANN 能否改善固定 FedAvg source checkpoint；
4. Full GAPS 相对于标准联邦与标准域适配基线的收益；
5. C5 上语义、回放、选择性聚合和 server DA 的分层贡献；
6. 性能收益是否伴随域差异下降，并以何种计算、通信与 source-retention 代价获得。

本任务不包含回归、QC、多随机种子、阈值搜索、超参数搜索、数据重划分或根据 target test 选择 checkpoint。所有正式结论只限 seed 42 的描述性证据，不宣称跨种子稳定性。

## 2. 选择性复用策略

采用方案 A：审计后选择性复用。

- 复用 P0A FedAvg LE1 round25 checkpoint，不重新训练 source Flower。
- 旧 FedProx 采用 LE5，与冻结 LE1 不同，不能复用。
- 旧 C5 Full GAPS 的 selective warmup 为 3，与本协议 warmup 5 不同，不能复用。
- 新增 10 个 25 轮完整联邦运行：FedProx 1、SCAFFOLD 1、Full GAPS 3、C5 A1-A5 5。
- 新增 9 个 100-step canonical UDA 适配分支。
- E0、统一评价、分析、绘图与审计不计为训练运行。

P0A checkpoint 是只读外部输入。执行前复制到新结果根的 `inputs/`，不修改旧结果。checkpoint equality 由按 `state_dict` 顺序计算的 ordered state-content fingerprint 判定；whole-file SHA-256 只用于文件来源记录，因为两个语义相同的 `torch.save` 容器不保证字节相同。已核验源文件：

- 相对语义路径：`results/iotj_p0_routing_simplification_20260803/P0A_PURE_FEDAVG_LE1_S42/remote_server/server_round_025.pth`
- SHA-256：`4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c`
- 训练协议：FedAvg、Adam、lr 5e-4、25 rounds、LE1、batch 32、seed 42、CE only、无 target access。

若复制后 ordered key/tensor/shape/dtype/content fingerprint 不一致，所有依赖该 checkpoint 的 E1 FedAvg 评价和 E2 均 fail closed。whole-file SHA 不一致本身只触发 provenance 复核，不替代 tensor-content equality。

## 3. 固定数据与评价协议

- Dataset：`dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`
- Source clients：C1、C2。
- Target clients：C3、C4、C5。
- Source train：仅用于 25 轮联邦训练。
- Source calibration：C1+C2，供 E2 的共同 source batches 使用。
- Target calibration：C3/C4/C5 的既有 calibration features；用途由各方法协议限定。
- Target test：sealed；训练、适配、停止、阈值、超参数或 checkpoint 选择阶段访问 target test features/labels 均为绝对 HARD FAIL。只有对应方法的固定 round25/step100 completion marker 产生后，final-evaluation gate 才能一次性解封该方法和 target 的 test x/class label。
- Target train：即使数据目录存在，也不进入本任务。
- 冻结样本计数：C1/C2 各 train 2360、calibration 320、test 680；C3 calibration 320、test 680；C4 calibration 160、test 320；C5 calibration 320、test 1360。执行前必须用 manifest/fingerprint 再核验这些计数。
- Formal checkpoint：round25 或 adaptation step100 固定端点。
- Seed：42 only。
- Metrics：accuracy、macro_f1、per-class precision/recall/F1、confusion matrix、NLL、ECE、commissioning/training seconds、source retention、通信量，以及每个 C3/C4/C5 的 `source_target_f1_gap = source_macro_f1 - target_macro_f1`。
- 逐窗口输出：sample id、client、split、method、true label、predicted label、class probabilities；只在 sealed test 开启后生成。

## 4. E0：无训练域偏移诊断

E0 使用 C1+C2 calibration 与各 target calibration，不打开 target test：

- 原始 8 通道逐通道均值、标准差、中位数、IQR、分位数差、标准化均值差和协方差统计；
- 使用冻结 source normalization 后的通道统计；
- 冻结 FedAvg round25 encoder embedding 的 global MMD²、CORAL covariance distance、mean-vector norm 和 covariance norm；
- PCA/UMAP 仅用于可视化，不用于选择方法或超参数。

输出包括三个机器可读 CSV 和 Fig. 1。E0 是描述性诊断，不用于选择 E2 loss 权重。

## 5. E1：联邦算法基线

### FedAvg

复用 P0A round25，不重新训练。统一 evaluator 在 C1/C2 source test 和 C3/C4/C5 sealed test 上评价。

### FedProx

- Adam，lr 5e-4；
- `mu=0.01`，复用历史正式值，不搜索；
- rounds 25、LE1、batch 32、seed 42；
- 其余与 FedAvg 相同。

### Canonical SCAFFOLD

- SGD，lr 5e-4，无 momentum、无 scheduler；该值在运行前预注册，不搜索；
- SGD lr 5e-4 不解释为与 Adam lr 5e-4 等价或优化器控制下的公平值；它只表示单一预注册 canonical SCAFFOLD 配置；
- 本地每个 minibatch 执行 `w <- w - eta * (grad L + c - c_i)`；
- server 持久化 control variate `c`；每个 source client 跨轮持久化 `c_i`；
- 对 K 个实际本地更新，采用 canonical Option II：
  `c_i_new = c_i - c + (w_global - w_local) / (K * eta)`；
- client 上传 model delta 和 `delta_c_i = c_i_new - c_i`；
- 全部 N=2 source clients 每轮参与时，server 更新
  `c_new = c + (1/N) * sum(delta_c_i)`；server model step size 固定为 1.0，模型 delta 按样本量聚合；由于 C1/C2 train 均为 2360，正式运行中的权重相等；
- 保存每轮 `c`、`c_i`、delta norms、状态 fingerprint；client 状态丢失或出现 Adam state 时 fail closed。

正式训练前增加一次丢弃式 source-only numerical validity gate：只使用 C1/C2 train 与 source test，从原始初始化运行一个完整本地轮次，检查 CE 首末四分位下降、combined source accuracy 高于观测 majority-class prior、所有 loss/gradient/parameter 为 finite、`0 < max_grad_norm < 1e4` 且 `0 < max_parameter_norm < 1e4`。该 gate 不读取任何 C3/C4/C5 target test，且结果不作为模型初始化或 checkpoint。若固定 lr5e-4 无法通过，则 fail closed、报告诊断并停止 SCAFFOLD，不自动尝试其他 lr。

SCAFFOLD 的优化器差异将在主表和分析中显式标注，不能解释为单因素 optimizer-controlled ablation。

## 6. E2：Canonical Domain-Adaptation Reference Study

E2 由 CORAL、MMD、DANN × C3/C4/C5 的 9 个独立分支组成。每个分支都从同一个原始 P0A round25 checkpoint 独立重载。

共同协议：

- target adaptation API 只接收 `x_target`，不接收 `y_cls`、`y_phase` 或包含这些字段的 batch；
- source batches 统一来自 C1+C2 calibration，batch 32；
- 100 adaptation steps，固定端点；
- 模型 optimizer=Adam、lr=5e-4；
- alignment coefficient=0.5，预注册、不搜索；
- target test 不参与 early stopping、checkpoint selection、threshold selection 或超参数选择；
- 每个分支 step100 完成后才开启对应 sealed test。

方法定义：

1. CORAL：`source CE + 0.5 * unconditional global covariance alignment`。
2. MMD：`source CE + 0.5 * unconditional global MMD^2`。
3. DANN：标准 gradient reversal + binary domain classification；`source CE + 0.5 * domain BCE`。encoder/classifier/discriminator 使用同一 Adam lr=5e-4 约定。

E2 明确禁止：target CE、target calibration class/phase/concentration、class-conditional CORAL、class-wise MMD、phase-conditioned MMD、same-class-phase stage MMD、conditional adversarial mask、target prototype anchor、pseudo labels，以及任何 GAPS-internal 条件化语义匹配。这里的 label prohibition 只适用于正式定义为 x-only 的 E2，不泛化到 Full GAPS。

静态测试检查函数签名和配置；运行时 audit 记录 label-free batch schema、loss terms、固定 endpoint 与 checkpoint reload hash。仅仅把受禁 loss weight 设为 0 不合格。

## 7. E3：Full GAPS 跨目标正式实验

分别对 C3、C4、C5 完成三次完整 25 轮目标特定运行：

- source clients C1+C2；target calibration 为当前目标；
- Adam lr 5e-4、rounds25、LE1、batch32、seed42；
- client classification：CE + semantic module + replay，回归 loss 禁用；
- semantic module 定义为现有 prototype alignment + prototype decoupling，避免隐藏组件变化；
- selective aggregation：warmup=5、min_scale=0.3、target-informed aggregation disabled；
- warmup=5 的轮次语义固定为 round1-5 完整执行 sample-weighted FedAvg，round6 起才允许 selective scaling；代码、日志、测试和 Fig.7 必须使用同一边界；
- 每轮保存 base weight、similarity、scale、final normalized weight；
- server DA 使用已冻结 full configuration，100 steps/round；
- server-adapted model 必须成为下一轮 global model；
- target calibration x/class/phase 可进入 GAPS 明确注册的 server-DA 分支；target concentration 不加载，target CE 固定为 0，任何 target test 信息不得进入训练或选择。每次访问均写入 method-specific ledger。

每个 target 的正式结果均为 round25，C5 逐轮 target 曲线仅作为 post-hoc diagnostic。

## 8. E4：C5 分层消融

采用累积式、非全因子消融：

| ID | Client profile | Selective aggregation | Server DA | 复用 |
|---|---|---:|---:|---|
| A0 | CE only | no | no | E1 FedAvg |
| A1 | CE + semantic (`proto_only`) | no | no | 新运行 |
| A2 | CE + semantic + replay (`proto_replay`) | no | no | 新运行 |
| A3 | CE + semantic + replay | yes | no | 新运行 |
| A4 | CE-only loss + stats upload (`ce_stats`) | no | full | 新运行 |
| A5 | CE + semantic + replay | no | full | 新运行 |
| A6 | CE + semantic + replay | yes | full | E3 C5 |

A4 的 stats upload 是 server DA 输入/观测，不是客户端 semantic loss。A1/A2 使用完整现有 semantic 定义，因此不会在 A6 中突然加入未标注的 prototype decoupling。

A1-A3 的训练不读取 target calibration；其 semantic/replay/选择性聚合输入来自 C1/C2 client statistics。A4-A6 的 server DA 使用 C5 calibration x/class/phase，不使用 concentration。每个 A0-A6 variant 必须汇总 `loss_name, configured_weight, input_available, active_steps, mean_raw_loss, mean_weighted_loss, inactive_reason` 到 `ablation_loss_activity.csv`。尤其 A4/A5 不能根据配置推断激活状态，而要从逐步运行诊断确认 server DA 的 global、class/phase conditional、prototype和 residual 项实际是否具有输入并产生有效 step。

## 9. 运行架构、恢复与失败策略

- 新结果根：`results/iotj_final_classification_le1_20260804/`。
- 新文档根：`docs/experiments/iotj_final_classification_le1_20260804/`。
- runner 分阶段执行：preflight → E0 → E1 → E2 → E3 → E4 → evaluation → analysis/figures → strict audit。
- 每个 run 具有唯一 experiment ID、不可变 manifest、命令快照、git commit、dataset fingerprint、日志和 completion marker。
- 已完成且 hash/manifest 一致的 run 可跳过；不完整 run 只能从明确 checkpoint 恢复，不能覆盖完成目录。
- 三机 Flower 拓扑保持现有 server/C1/C2 分工；每次只运行一个正式联邦配置，避免 GPU/端口/日志混淆。
- 独立实验失败时记录 `failed` 并允许后续独立实验继续；共同输入、split、checkpoint 或 label-access audit 失败时整阶段 fail closed。

## 10. 输出与主表

必须生成：

- `classification_main_comparison.csv`
- `classification_hierarchical_ablation_c5.csv`
- `domain_shift_summary.csv`
- `domain_discrepancy_summary.csv`
- `source_retention_summary.csv`
- `communication_compute_summary.csv`
- `per_window_predictions.parquet`
- FedAvg/GAPS embeddings
- Fig. 1–9 的源数据和成图
- `PROTOCOL.md`
- `RESULT_ANALYSIS.md`
- `EXPERIMENT_AUDIT.md`
- `LABEL_ACCESS_AUDIT.md`
- `protocol_manifest.json`
- `experiment_registry.csv`
- `sha256_index.json`

`classification_main_comparison.csv` 与 `PROTOCOL.md` 必须包含：`optimizer`、`optimizer_lr`、`optimizer_note`。

`RESULT_ANALYSIS.md` 必须原样包含：

> SCAFFOLD is implemented with its canonical SGD-style control-variate update, whereas FedAvg, FedProx, and GAPS use the frozen Adam optimizer adopted by the experimental system. Therefore, the comparison represents standard algorithm-level baselines rather than an optimizer-controlled single-factor ablation.

## 11. 验证与审计门禁

实现阶段先写失败测试，再写代码。至少包含用户指定的五项 SCAFFOLD 测试，以及：

- E2 target batch/API 不含 class/phase label；
- CORAL/MMD/DANN 为 unconditional/global；
- E2 禁用 target CE、conditional losses、pseudo labels；
- 三种 E2 方法重载同一 checkpoint hash；
- fixed step100 且无 test-based selection；
- GAPS selective warmup=5、min_scale=0.3；
- adapted-as-global lineage；
- round25 formal endpoint；
- source/test/calibration split isolation；
- artifact schema、manifest、SHA index 完整。

最终门禁：目标测试隔离审计通过、strict audit 通过、目标测试集只在允许阶段打开、全部要求文件存在、CSV/JSON schema 校验通过、`pytest` 通过、`compileall` 通过、工作树只包含本任务变更。

## 12. 停止条件

本任务完成已批准的固定矩阵后停止。不因结果高低搜索 SCAFFOLD lr、FedProx mu、E2 loss 权重、GAPS 参数或新增无监督方法；不自动扩展多 seed。任何后续优化需新协议和用户批准。
