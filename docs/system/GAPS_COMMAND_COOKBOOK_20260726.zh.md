# GAPS 最终系统命令手册（2026-07-26）

> **重要：这是已完成实验的可审计命令记录，不是待执行队列。**
>
> 当前状态为 `NO_FURTHER_EXPERIMENTS_REQUIRED_FOR_CURRENT_SCOPE`。标为 `can rerun after evidence freeze: no` 的命令不得在当前范围执行。

## 0. 命令来源

命令只来自：

- `results/c2e_commands/*/command_manifest.json`；
- `results/iotj_b5_multiseed_20260724/commands/launch_seed*.cmd`；
- formal attempt `run_config.json`；
- 各脚本 argparse / `--help`；
- protocol/result manifests；
- Git commit identity。

占位符只表示需要从相邻 manifest 读取的路径，不代表猜测默认值。

## A. Environment / preflight

```powershell
python -m scripts.run_iotj_confirmation_observability `
  --protocol-manifest results/c2e_summary/confirmation_protocol_manifest.json `
  --source-archive-manifest results/c2e_summary/source_archive_manifest.json `
  --dataset-manifest results/c2e_summary/dataset_manifest.json `
  --command-root results/c2e_commands `
  --source-archive results/c2e/source/confirmation_source.tar `
  --validate-inputs-only
```

| Field | Value |
|---|---|
| machine | Local controller |
| type | audit |
| reads test | no |
| expected outputs | stdout validation only |
| approximate runtime | minutes；正式耗时未单独持久化 |
| identity | formal frozen manifests |
| can rerun after evidence freeze | yes，纯静态输入核验 |

三机 preflight 使用同一命令再加入正式 topology/host 参数和 `--preflight-only`。当前不应重新连接主机。

## B. Classification manifest generation

```powershell
python -m scripts.freeze_iotj_confirmation_protocol `
  --confirmation-commit 2ef7aea77b9dfabdd09da4f38742907a37c58c30 `
  --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid `
  --archive-output results/c2e/source/confirmation_source.tar `
  --command-root results/c2e_commands `
  --summary-root results/c2e_summary
```

| machine | type | reads test | outputs | runtime | formal | rerun |
|---|---|---|---|---|---|---|
| Local | manifest/archive build | no labels；会 fingerprint data | c2e archive、command/protocol manifests | unknown | formal source identity | no；已有 freeze 不得覆盖 |

## C. ECS server — frozen B5 seed42 argv

来源：`results/c2e_commands/c12_to_c5__b5__s42/command_manifest.json`。

```bash
/root/gaps_env/bin/python -m gaps_flower.server_app \
  --server-address 0.0.0.0:8080 \
  --rounds 25 --min-clients 2 --strategy gaps --profile proto_replay \
  --seed 42 \
  --run-name B5_proto_replay_corrected_full_da_c12_to_c5_s42_r25 \
  --output-dir results/iotj_main_confirmation_observability_20260715/B5_proto_replay_corrected_full_da_c12_to_c5_s42_r25 \
  --save-history true --use-selective-agg true --use-proto-mmd false \
  --da-preset none --use-domain-adapt true \
  --server-val-data dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_1,dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_2 \
  --server-calib-data dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5 \
  --domain-adapt-steps 100 --domain-adapt-warmup 0 \
  --da-use-coral true --da-use-mmd true --da-use-adversarial true \
  --da-mmd-objective mmd2 \
  --da-stage-alignment cross_domain_same_class_phase \
  --da-adv-feature-objective wasserstein_min \
  --da-coral-class-conditional true --strict-calibration-split true \
  --da-device cpu --use-adapted-as-global true \
  --da-lambda-coral 0.5 --da-lambda-global-mmd 0.5 \
  --da-lambda-class-mmd 0.5 --da-lambda-proto-anchor 0.3 \
  --da-lambda-adv 0.5 --da-lambda-target-ce 0.0 \
  --da-lambda-proto 0.05 --da-lambda-consistency 2.0 \
  --da-lambda-residual 0.1 --da-lambda-proto-mmd 0.0 \
  --da-lambda-stage-mmd 0.2 \
  --da-target-ce-label-smoothing 0.0 \
  --da-target-ce-class-balanced false \
  --da-server-opt-lr 0.0005
```

| machine | type | reads test | outputs | runtime | formal | rerun |
|---|---|---|---|---|---|---|
| Alibaba ECS | training | no；C5 calibration only | 25-round checkpoints/run_config | seed42 recorded 5932 s | formal B5 | no |

正式 attempts 还由 controller 注入 `--observer-context` 和 `--observer-events`；不要手写这些 attempt-specific paths。

## D. Pi C1 client — frozen B5 seed42 argv

```bash
/home/gaps/GAPS/gaps_rpi_env/bin/python -m gaps_flower.client_app \
  --server-address 127.0.0.1:18080 \
  --client-id 1 \
  --data-root /home/gaps/GAPS/flower_runtime/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid \
  --device cpu --local-epochs 5 --batch-size 32 \
  --profile proto_replay --seed 42
```

| machine | type | reads test | inputs | outputs | formal | rerun |
|---|---|---|---|---|---|---|
| Raspberry Pi C1 | training | no | C1 local source data | Flower FitRes/local stats | formal | no |

## E. C2 client

冻结 command manifest 的 PC argv：

```powershell
python -m gaps_flower.client_app `
  --server-address 127.0.0.1:18080 `
  --client-id 2 `
  --data-root "D:\A Python learning\Federated Learning\TRAE SOLO\dataset\client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid" `
  --device cpu --local-epochs 5 --batch-size 32 `
  --profile proto_replay --seed 42
```

最终 five-seed 使用 ECS C2。其路径/observer/tunnel 由 controller 根据
`results/c2e_ecs_c2_topology/execution_topology_manifest.json` 重写；没有独立发布的
manual ECS-C2 shell command。正式使用必须走 controller，不能用上面的 PC argv
冒充 ECS C2。

| machine | type | reads test | formal | rerun |
|---|---|---|---|---|
| PC C2 command / ECS C2 controller-managed replacement | training | no | PC argv 是冻结基础；最终 topology 是 ECS C2 | no |

## F. Sequential controller

seed43 的完整正式 controller invocation 已冻结在：

`results/iotj_b5_multiseed_20260724/commands/launch_seed43.cmd`

核心调用如下；seeds 44–46 只改变 seed、run/output 和派生 manifest：

```powershell
python -m scripts.run_iotj_confirmation_observability `
  --protocol-manifest results/c2e_summary/confirmation_protocol_manifest.json `
  --source-archive-manifest results/c2e_summary/source_archive_manifest.json `
  --dataset-manifest results/c2e_summary/dataset_manifest.json `
  --command-root results/c2e_commands `
  --source-archive results/c2e/source/confirmation_source.tar `
  --raw-root results/iotj_b5_multiseed_20260724/seed43/raw `
  --runs B5:43 `
  --ecs-host root@121.40.139.213 `
  --pi-hosts gaps@192.168.137.172 `
  --wait-for-pi-minutes 30 --pi-retry-seconds 10 `
  --c2-host root@114.55.171.63 `
  --c2-python /root/gaps_c2_cpu_env/bin/python `
  --c2-data-root /root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid `
  --c2-dataset-subset-manifest results/c2e_ecs_c2_topology/c2_dataset_subset_manifest.json `
  --execution-topology-manifest results/c2e_ecs_c2_topology/execution_topology_manifest.json `
  --run-timeout-seconds 172800 --poll-seconds 30
```

| machine | type | reads test | output | runtime | formal | rerun |
|---|---|---|---|---|---|---|
| Local controller + three hosts | training orchestration | no | canonical attempt | seed43 6151 s；43–46 range 6100–6304 s | formal | no |

## G. Classification evaluation

CLI contract：

```powershell
python -m scripts.evaluate_iotj_b5_multiseed_seed `
  --seed <42..46> `
  --checkpoint <server_round_025_adapted.pth> `
  --data-root <frozen_dataset_root> `
  --row-map <frozen_row_map> `
  --runtime-contract <frozen_runtime_contract> `
  --output-dir <seedXX/classification_evaluation> `
  --device cpu --batch-size 64
```

| machine | type | reads test | outputs | runtime | formal | rerun |
|---|---|---|---|---|---|---|
| Local | evaluation | yes，1360 rows | metrics、confusion、route CSV | unknown | completed formal | no |

路径必须从 seed manifest/row-map/runtime contract 解析，不能选择其他 checkpoint。

## H. H1 sufficient-statistics build / audit

真实 topology 的六阶段 CLI：

```text
python -m scripts.materialize_iotj_federated_h1_topology client-moments ...
python -m scripts.materialize_iotj_federated_h1_topology server-scalers ...
python -m scripts.materialize_iotj_federated_h1_topology client-equations ...
python -m scripts.materialize_iotj_federated_h1_topology server-candidates ...
python -m scripts.materialize_iotj_federated_h1_topology client-scores ...
python -m scripts.materialize_iotj_federated_h1_topology server-model ...
```

客户端模式是唯一允许接收 `--data-root` 的模式；server 模式只接收 JSON statistics。
具体 inputs/outputs 已冻结在
`results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/`。

equivalence audit CLI：

```powershell
python -m scripts.evaluate_iotj_h1_federated_ridge_equivalence `
  --data-root <frozen_dataset_root> `
  --runtime-contract <v4_runtime_contract> `
  --h8-validation-rich <validation_rich> `
  --h8-validation-prior <validation_prior> `
  --h8-test-rich <test_rich> `
  --h8-test-prior <test_prior> `
  --output-dir results/iotj_h1_federated_ridge_equivalence_20260724 `
  --seed 42 --batch-size 64 --device cpu --formal-run
```

| machine | type | reads test | runtime | formal | rerun |
|---|---|---|---|---|---|
| C1/C2/server staged + Local audit | statistics build/evaluation | formal audit yes；test 不参与 fit/select | unknown | completed | no |

## I. Target Ridge multi-seed evaluation

```powershell
python -m scripts.evaluate_iotj_b5_regression_multiseed `
  --formal-run `
  --device cpu --batch-size 64 `
  --data-root <frozen_dataset_root> `
  --multiseed-root results/iotj_b5_multiseed_20260724 `
  --runtime-contract <frozen_v4_runtime_contract> `
  --h1-manifest results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json `
  --r4-policy <frozen_r4_policy> `
  --output-dir results/iotj_b5_regression_multiseed_20260724
```

| machine | type | reads test | output | runtime | formal | rerun |
|---|---|---|---|---|---|---|
| Local/ECS analysis host | evaluation | yes only after calibration lock | RG0/RG1/RG2 + decision | unknown | formal | no |

## J. Runtime-v5 bundle build

同一 output root 的正式顺序：

```powershell
python -m scripts.build_iotj_runtime_v5_candidate freeze-calibration <common-args>
python -m scripts.build_iotj_runtime_v5_candidate evaluate-test <common-args>
python -m scripts.build_iotj_runtime_v5_candidate materialize-outputs <common-args>
python -m scripts.build_iotj_runtime_v5_candidate build-bundle <common-args>
python -m scripts.build_iotj_runtime_v5_candidate calibration-parity <common-args>
python -m scripts.build_iotj_runtime_v5_candidate runtime-parity <common-args>
python -m scripts.build_iotj_runtime_v5_candidate finalize-candidate <common-args>
```

`<common-args>` 精确字段：

```text
--data-root <frozen_dataset_root>
--multiseed-root results/iotj_b5_multiseed_20260724
--real-h1 results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/global_h1_model.json
--audited-h1 results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json
--output-dir results/iotj_b5_c5_runtime_v5_candidate_20260724
--batch-size 64
--device cpu
```

| machine | type | reads test | output | formal | rerun |
|---|---|---|---|---|---|
| Local | build + parity | evaluate-test/runtime-parity yes | v5 three-asset bundle/contract/parity | completed | no |

## K. Runtime-v5 single/batch inference

**没有正式 CLI。** Python API：

```python
from gaps_deploy.c5_federated_source_ridge_runtime import C5FederatedSourceRidgeRuntime

runtime = C5FederatedSourceRidgeRuntime.from_runtime_contract(
    "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json",
    device="cpu",
)
rows = runtime.infer(windows, metadata, phases)
```

| machine | type | reads test | formal | rerun |
|---|---|---|---|---|
| Local/Pi/PC | inference API | depends on caller | canonical core API | no current-scope rerun |

## L. Runtime-v4 inference

历史 package CLI：

```powershell
python -m gaps_deploy.final_runtime `
  --bundle <v4_bundle> --client-id C5 `
  --input <windows.npy> --metadata-file <metadata.json> `
  --phase-file <phase.npy> --device cpu --output <rows.json>
```

当前正式 C5 v4 baseline 的代码对象是 `C5H8Runtime`；`final_runtime.py` CLI 仍是
C12→C345 package wrapper，不能当成 v5 CLI。

## M. QC runtime

v5 QC2 Python API：

```python
from gaps_deploy.c5_federated_source_ridge_qc_runtime import C5FederatedSourceRidgeQCRuntime

runtime = C5FederatedSourceRidgeQCRuntime.from_runtime_contract(contract, device="cpu")
rows = runtime.infer(windows, metadata, phases)
```

正式 QC 构建/evaluation 的 staged CLI 是：

```text
python -m scripts.evaluate_iotj_runtime_v5_qc calibrate-lock ...
python -m scripts.evaluate_iotj_runtime_v5_qc evaluate-test ...
```

该实验已完成，`can rerun after evidence freeze: no`。

## N. PC/Pi benchmark

脚本参数合同：

```powershell
python -m scripts.benchmark_iotj_final_runtime `
  --runtime <RUNTIME_V4_FULL|RUNTIME_V5_REGRESSION_CORE|RUNTIME_V5_QC2_CANDIDATE> `
  --contract <matching_runtime_contract> `
  --data-root <C5_data_root> `
  --platform-id <PC|Pi> `
  --device cpu --threads 1 --batch-size 1 --warmup 50 --runs 500 `
  --output <new_result.json> --rows-output <new_rows.csv>
```

2026-07-25 正式运行的已知参数、结果路径和未知原始 shell 字段，见只读重建
`docs/system/benchmark_command_manifest_20260725.json`。该文件没有重跑 benchmark，
也不声称恢复了未被记录的 Python executable、working directory 或 shell quoting。

| machine | type | reads test | inputs | output | formal | rerun |
|---|---|---|---|---|---|---|
| PC/Pi | benchmark + inference | yes，固定 1360 universe 中前 500 次 | matching contract/data | benchmark JSON/rows | commit `4ccfc489...` | no |

cold-start：

```powershell
python -m scripts.probe_iotj_runtime_cold_start `
  --runtime <runtime_kind> --contract <contract> `
  --data-root <C5_data_root> --output <new_result.json>
```

## O. Parity validation

v5 parity 由 J 的 `calibration-parity` 和 `runtime-parity` 阶段执行。v4 行流 validator：

```powershell
python -m scripts.validate_iotj_b5_c5_runtime_parity `
  --reference <offline_reference.csv> `
  --runtime <runtime_rows.csv> `
  --output <new_parity_report.json>
```

| machine | type | reads test | formal | rerun |
|---|---|---|---|---|
| Local | audit | depends on supplied rows；不做新 inference | formal validator | yes，仅对已有行流写新 sibling report，不覆盖 |

## P. Paper table regeneration

正式 evidence builder：

```powershell
python -m scripts.build_iotj_final_system_evidence `
  --output-dir results/iotj_final_system_benchmark_20260725
```

finalizer：

```powershell
python -m scripts.finalize_iotj_final_system_evidence `
  --result-root results/iotj_final_system_benchmark_20260725 `
  --report-path docs/experiments/iotj_final_system_benchmark_result_20260725.zh.md `
  --index-path docs/experiments/iotj_final_system_benchmark_result_index_20260725.json
```

当前输出已冻结，因此 `can rerun after evidence freeze: no`。后续论文排版应复制已冻结
table/figure，不覆盖 canonical result root。

## 命令身份的最后边界

- classification/server/client 命令必须由 matching command manifest 驱动；
- seed 只能匹配其 checkpoint/attempt；
- v5 runtime bundle 只允许 B5 seed42 + real H1 + 105D target Ridge；
- test-reading 命令在 evidence freeze 后全部禁止重跑；
- benchmark 参数必须是 batch 1、warmup 50、500 runs、CPU single-thread；
- 没有 command manifest 的本地命令不得被补写成“当时正式执行的 exact shell command”。
