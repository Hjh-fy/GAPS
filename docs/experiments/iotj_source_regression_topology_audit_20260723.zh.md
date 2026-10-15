# IoT-J Source Regression Topology Audit — 2026-07-23

## 1. 审计结论

本次审计只读检查现有 pooled-source H8/R4、`gaps_flower` 回归训练/聚合/推理代码，并新增独立实验入口和 smoke 验证；没有重新训练 B5 classifier，没有修改 C5 runtime v4、HC95/HC90 policy 或 parity evidence。

六个问题的直接答案如下。

| 问题 | 结论 | 证据与边界 |
|---|---|---|
| H1/H2/H3 是否使用 pooled C1+C2 raw source rows？ | **是** | `fit_source_heads()` 分别调用 `build_oracle_rows(data_root, ["C1","C2"], "train"/"calibration")`，随后只按 gas class 分组，不保留 client 隔离。H1 per-gas Ridge、H2 per-gas MLP、H3 shared MLP 均是 centralized pooled-source training。 |
| 真正的 source regression FedAvg 链路是否完整可执行？ | **算法链路可执行；标准 Flower 网络链路未接入** | `train_federated_source_regression()` 可完成同一 global state 下的 C1/C2 local training、sample-count weighted regression-only FedAvg 和多轮回写；`regression_client.py` + `regression_server.py` 可通过 checkpoint 文件完成分离训练/聚合。但标准 `client_app.py/server_app.py` 仍是分类链路，当前回归不是经 Flower transport 传参的在线联邦作业。 |
| C1/C2 local model 能否分别保存并独立推理？ | **能，但旧路径缺少严格同源合同** | client 脚本分别写出 `regression_source_client1_local.pth`、`regression_source_client2_local.pth`；完整 state 可由同构模型严格加载并推理。旧 checkpoint 没有强制记录 classifier SHA、initial regression SHA 和非回归 state SHA。新增实验入口补齐这些字段。 |
| global FedAvg model 能否独立对 C5 windows 输出 ppm？ | **能** | `regression_server.py` 写出 `regression_fedavg_global.pt`；`evaluate_regression_pipeline.py` 能 strict-load regression checkpoint，以 B5 predicted class 路由，调用 `forward_reg()` 并按 gas range 反归一化为 ppm。 |
| local 与 FedAvg 是否保证完全相同架构和初始化？ | **进程内训练是；旧分离式 client/server 仅假定、未证明** | 进程内函数从同一 global state `deepcopy` 各 client，满足同构同初始化。分离 client CLI 在相同 seed/config 下通常一致，但旧产物不写 initial-state hash；server 也未比较两端 initial hash。新增入口在训练前 fail-closed 比较 109 个回归 state tensors 的 SHA256。 |
| 是否存在参数遗漏、错误聚合、backbone 更新、泄漏或 test selection？ | **未发现当前默认 DCT16 回归参数遗漏或 target-test selection；存在若干 P1 合同风险** | 动态审计确认当前默认模型 109/109 个回归 parameter tensors 全部进入聚合集合，共 411,850 parameters，未包含 backbone。现有 H8 target Ridge 只在 calibration fit/validation 选 alpha 并在 full calibration refit，test 只评估。风险见第 4 节。 |

## 2. 当前 pooled-source H8/R4 的准确数据流

```text
C1 train rows + C2 train rows
  -> centralized merge
  -> H1 per-gas Ridge / H2 per-gas MLP / H3 shared MLP

C1 calibration rows + C2 calibration rows
  -> centralized merge
  -> source head hyperparameter selection
  -> refit with pooled source train + calibration

B5 seed42 predicted class on C5
  + C5 rich response features
  + pooled H1/H2/H3 ppm
  -> per-gas C5 Ridge
  -> alpha selected on C5 calibration-validation
  -> selected Ridge refit on full C5 calibration
  -> one-time C5 test evaluation
```

因此 RS0 的性能是有效 centralized-source baseline，但不能称为“source regression knowledge federated”。正式 runtime v4 的 R4 仍保持冻结，本实验不改变其语义。

## 3. 新 federated source-prior 数据流

```text
frozen B5 seed42 classifier
  ├─ identical regression initialization -> C1 train only -> R_C1
  └─ identical regression initialization -> C2 train only -> R_C2
                                      │
                     regression-only, sample-count weighted FedAvg
                                      ↓
                                  R_FedAvg

same C5 window + B5 predicted route
  ├─ R_C1 -> pred_C1
  ├─ R_C2 -> pred_C2
  └─ R_FedAvg -> pred_FedAvg

C5 calibration rich features + frozen source-prior columns
  -> RS1 / RS2 / RS3 target Ridge
  -> calibration-fit / calibration-validation selection only
  -> full-calibration refit
  -> frozen C5 test evaluated once
```

C1 与 C2 raw rows 在 RS1–RS3 中从不合并。只有参数通过 FedAvg 聚合。第一阶段不接 QC，也不修改 runtime。

## 4. 风险登记

### P0

没有发现阻断本次协议冻结的 P0。新增入口在以下条件下直接失败：B5 checkpoint 非预期键、回归 state 范围不完整、C1/C2 loader 缭乱、初始化哈希不同、backbone/non-reg state 改变、C5 行数或 RS0 row identity 不一致、selection 输入含 test、输出目录非空、冻结 evidence SHA 改变。

### P1

1. **回归尚未接入标准 Flower transport。** 当前可执行的是进程内 FedAvg 或 checkpoint-mediated aggregation。论文可称 regression-only FedAvg simulation / file-mediated federation，不能把它描述为已验证的在线 Flower regression deployment。
2. **旧 `load_classifier_weights()` 使用 `strict=False`。** 它记录 missing/unexpected keys，却不强制“missing 只能等于回归专属键集合”。新入口改为严格白名单；旧入口本身仍保留历史语义。
3. **旧分离 client/server 未绑定同一初始化。** 相同 seed/config 是约定，不是证据。新入口记录 B5 SHA、initial regression SHA、local/global SHA 和每轮 before/after trace。
4. **旧 `fedavg_regression_states()` 缺少友好的 shape/dtype/key-set 预检。** 缺键通常以 `KeyError`/tensor error 失败，但合同信息不完整。新入口先审计完整聚合 key set，并由 strict state load 和 hash trace封闭。
5. **冻结 backbone 仍处于 `model.train()` 前向模式。** 参数与 state 经 smoke hash 证明不变，但 dropout 等训练态行为可能使 regression feature sampling 随机。seed42 使协议可复现；若后续要求“冻结行为等于 B5 eval feature”，应另立消融，不能静默改变本协议。
6. **旧 server 对架构字段的检查不完整等价于完整模型合同。** 部分 gamma/dropout 或未来新增结构字段可能未被逐项比较。新入口用 exact state compatibility、完整 regression state keys 和 manifest 补强。
7. **数据根必须绑定正式 C5 root。** runtime v4/R4 正式 reference 使用 `client_data_c1234src_c5tgt_...`。初次 smoke 若误用历史 `c12src_c345tgt`，row identity 会与 RS0 reference 不一致；新增入口现已将前者设为默认并逐行 fail-closed 比较 true class/ppm。

## 5. RS0–RS4 冻结矩阵

| Variant | Source raw topology | C5 Ridge 输入 | Source models | pooled raw source |
|---|---|---|---:|---|
| RS4_rich_only | 无 source regressor | rich | 0 | 否 |
| RS0_pooled_source | C1+C2 centralized pooled | rich + H1 + H2 + H3 | 3 | **是** |
| RS1_local_experts | C1、C2 独立 | rich + pred_C1 + pred_C2 | 2 | 否 |
| RS2_fedavg_prior | C1/C2 local + regression-only FedAvg | rich + pred_FedAvg | 1 inference prior | 否 |
| RS3_local_plus_fedavg | 同一轮 local + FedAvg 产物 | rich + pred_C1 + pred_C2 + pred_FedAvg | 3 inference priors | 否 |

共同冻结项：

- B5 seed42 adapted classifier，SHA256 `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`；
- C1/C2 各 2,360 train windows；C5 calibration 320、test 1,360；
- DCT16 regression branch、depth 4、sigmoid output，当前允许聚合 411,850 parameters；
- source clients 使用相同 architecture、B5 state 和 regression initialization；
- 3 rounds、每 client 总计 100 local steps、batch 32、AdamW lr 1e-3；
- C5 calibration 内 deterministic 75% fit / 25% validation；
- alpha grid `0, 0.01, 0.1, 1, 10, 100, 1000`；
- predicted-class route；无 QC；test 不进入 fit/select/refit。

## 6. 指标、晋级规则与计算量

输出指标固定为 ALL RMSE、ALL MAE、range-normalized NRMSE、S_CC RMSE、per-gas RMSE、CO RMSE、CO high 200–250 ppm RMSE。另记录 source parameter/model counts、pooled raw source requirement、target calibration parameter count 与 inference feature dimension。

RS1–RS3 共享一次 C1/C2 local training 与一次 FedAvg，不是三套重复训练。正式单 seed 预算约为：

- source optimization：2 clients × 100 steps = 200 local optimizer steps，分 3 rounds；
- source inference：3 regressors × (320 calibration + 1,360 test) = 5,040 window-model evaluations；
- B5 routing：1,680 windows；
- target Ridge：4 个新 variants × 4 gases × 7 alpha candidates，加 16 个 full-calibration refits；计算量相对神经前向很小；
- 参数/存储：每个 source regressor 411,850 trainable regression parameters；三个完整 checkpoints 约 5–6 MiB，表格通常小于数 MiB；
- 1-step CPU smoke 的主要成本已由三路 C5 神经前向主导；据此正式单 seed 预计同机 CPU 约 12–20 分钟，单 GPU 约 2–5 分钟。正式 multi-seed 约按 seed 数线性扩展。

晋级规则保持预注册：

1. RS3 test RMSE 优于 RS0：进入 final regression candidate；
2. RS3 相对 RS0 退化不超过 5%：标记 paper-preferred candidate，等待 multi-seed；
3. 退化超过 10%：不得替换 RS0；
4. RS2 解释 FedAvg prior 本身；RS1 解释 local complementarity；RS4 解释 source prior 增益。

注意：规则只在所有 calibration 选择冻结后用于最终方法判定，不能据 test 反向调参。

## 7. 最小验证结果

- 25 个新旧相关单元测试通过；
- B5 strict compatibility：80 个 checkpoint tensors；missing 恰好为 109 个 regression-only tensors；unexpected 为 0；
- C1/C2 initial regression SHA 相同：`c92ebdd8e7df14eac6cef2c59d42640603a9c74b512a62f2dfe17bd61343650b`；
- aggregation scope：109/109 parameter tensors，411,850 parameters，无 backbone key；
- 1 round × 1 step smoke 使用独立 C1/C2 loaders，生成 local C1、local C2、FedAvg global checkpoints；
- 三个模型对相同 C5 windows 均输出有限 ppm；
- RS1/RS2/RS3 feature schema 固定；
- selection API 拒绝 test rows，test-label mutation 不改变 selection signature；
- runtime v4 与 HC95/HC90 六个冻结文件 smoke 前后 SHA256 完全一致。

Smoke 只验证执行性，不是性能 evidence，不得用于 RS3 晋级。

## 8. 推荐正式命令与是否进入正式实验

推荐先在当前 commit 上运行一次 contract check：

```powershell
python scripts/evaluate_iotj_federated_source_regression_prior.py
```

正式单 seed 命令：

```powershell
python scripts/evaluate_iotj_federated_source_regression_prior.py `
  --formal-run `
  --device cuda `
  --source-rounds 3 `
  --source-steps 100 `
  --output-dir results/iotj_federated_source_regression_prior_20260723
```

建议：**可以进入一次正式 RS0–RS4 单 seed 实验，但应在本次代码/文档 commit 审阅后另行授权运行。** 单 seed 只用于 topology feasibility 与候选筛选；若 RS3 达到前两条晋级规则，再冻结 multi-seed 计划。当前阶段不替换 runtime v4，不接 QC，不作正式论文主结果声明。
