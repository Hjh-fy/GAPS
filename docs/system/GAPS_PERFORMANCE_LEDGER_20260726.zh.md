# GAPS 最终系统性能账本（2026-07-26）

## 1. 口径规则

本账本只复制 frozen canonical assets 中已经存在的数值，没有重新评估。每个指标保留 experiment、seed、sample scope、QC/routing 状态和 source SHA。不同 scope 不合并。

## 2. B5 five-seed classification

来源：

- `results/iotj_b5_multiseed_20260724/per_seed_b5_classification_metrics.csv`
- SHA256 `aef1a8c10936693d7444ab14fd727cf473296d5d859236be7b3f323ae0193e51`
- summary SHA256 `11adaa5202268a206ac01f49fe8ca515716d2e80b9cfc88c5877f3ef6ef925c8`

| Metric | Value | Scope | Role |
|---|---:|---|---|
| Accuracy | `0.9891176471 ± 0.0059826016` | seeds 42–46；C5 test N=1360/seed | main |
| Macro-F1 | `0.9891344644 ± 0.0059603199` | seeds 42–46 | main |
| NLL | `0.1002370480 ± 0.0378334431` | seeds 42–46 | appendix/system calibration |
| ECE | `0.0106065767 ± 0.0057738394` | seeds 42–46 | appendix/system calibration |
| Error count | `14.8 ± 8.1363` | seeds 42–46 | diagnostic |
| Accuracy range | `[0.9801470588, 0.9948529412]` | seeds 42–46 | stability |

## 3. Runtime classifier identity

| Item | Value | Source | Role |
|---|---|---|---|
| Runtime classifier | Final B5 seed42 | `results/iotj_b5_c5_runtime_v5_candidate_20260724/lineage_manifest.json` | deployment identity |
| Checkpoint SHA256 | `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c` | same | main |
| Seed42 Accuracy | `0.9801470588` | per-seed classification CSV | deployment seed；not five-seed mean |
| Seed42 errors | `27/1360` | per-seed classification CSV | diagnostic |

不要用 historical B5 screen 的约 `0.988971` 替代该 checkpoint 指标。

## 4. RG0/RG1/RG2 five-seed regression

来源：

- `results/iotj_b5_regression_multiseed_20260724/regression_multiseed_summary.csv`
- SHA256 `5f6e3078bb82652d5268fda9e626c85dac07976e385bebf76ccb6fbaa4541378`
- per-seed SHA256 `6c01a8c403e0ce105de47505cf83fe72abb33e7b5feaf58e50fd43fc6e2b1a3a`

| Variant | Input | S_CC RMSE mean ± sample std (ppm) | S_ALL RMSE mean ± sample std (ppm) | Role |
|---|---:|---:|---:|---|
| RG0_RICH_ONLY | 104D | `14.454631 ± 0.262100` | `20.466785 ± 3.457738` | ablation |
| RG1_FEDERATED_H1 | 105D | `11.633858 ± 0.314203` | `18.508025 ± 4.321091` | final simplified regression |
| RG2_ALL_PRIOR | 107D | `11.520766 ± 0.314776` | `18.598173 ± 4.419164` | all-prior reference |

RG1 相对 RG2 的 mean S_CC 退化为 `0.981637%`，满足预注册 `≤1%` 非劣 gate；绝对精度方向仍是 RG2 在 5/5 seeds 更低。该选择是简化非劣，不是 RG1 绝对更优。

## 5. Final seed42 Runtime v5 regression

来源：

- `results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/equivalence_decision.json`
- SHA256 `34b9125cba3e46bf4703ba8c518ff442c87a2456ed53bc1761b1c6e11c21ccee`

| Metric | Value | Scope | Role |
|---|---:|---|---|
| S_ALL RMSE | `25.6489781431 ppm` | seed42；all 1360 rows | deployment end-to-end |
| S_ALL MAE | `9.3837478583 ppm` | seed42；all 1360 | main/appendix |
| S_ALL NRMSE | `0.2042947562` | seed42；all 1360 | appendix |
| S_CC RMSE | `11.3415985730 ppm` | seed42；1333 correct-route rows | regression capability |
| CO RMSE | `22.6649997607 ppm` | seed42；all CO rows | appendix |
| CO-high RMSE | `35.0212127843 ppm` | seed42；102 rows | appendix |
| target Ridge parameters | `424` | four per-gas 105D heads | model-only |

该 `S_CC` 不是自动输出 RMSE，也不是 five-seed mean。

## 6. H1 equivalence

两项 evidence 不混写：

| Evidence | Value | Source / SHA256 | Role |
|---|---:|---|---|
| in-process sufficient-statistics H1 prediction max diff | `2.0809125e-08 ppm` | `results/iotj_h1_federated_ridge_equivalence_20260724/equivalence_decision.json` / `e8f30ec1a64c294afa9ef69494f15fad51f6061e7a754a7feeae3e7e1d19101d` | diagnostic equivalence |
| real-topology H1 prediction max diff | `6.2532195e-08 ppm` | runtime-v5 candidate equivalence / `34b912...` | deployment lineage |
| real-topology RG1 S_ALL RMSE diff | `9.44596e-11 ppm` | same | diagnostic |
| real-topology RG1 S_CC RMSE diff | `6.70397e-12 ppm` | same | diagnostic |

两者结论均为 `PRACTICAL_EQUIVALENCE`，但来自不同阶段和文件。

## 7. Runtime efficiency

来源：

- `results/iotj_final_system_benchmark_20260725/benchmarks/benchmark_summary.csv`
- SHA256 `528f60a0f87d94ecc748001c580e32d5ae67de3bbeb78b2848a1efe5baaf654c`
- package size SHA256 `0346b757f7ac7df130f8d4e10ad527f8d2f1a0a4f86a9e4028ea1fe9ed523c22`

| Runtime | Regression params | Bundle bytes | PC p50/p95 (ms) | PC peak RSS (MiB) | Pi p50/p95 (ms) | Pi peak RSS (MiB) | Pi peak °C | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v4 full | 28,737 | 2,971,538 | 5.4337 / 13.9448 | 199.848 | 4.57137 / 4.62118 | 237.891 | 54.00 | formal baseline |
| v5 regression core | 844 | 289,916 | 4.0716 / 14.7863 | 195.711 | 3.72454 / 3.76499 | 234.234 | 52.35 | final simplified regression |
| v5 QC2 | 844 | 1,065,632 | 5.47655 / 19.3115 | 197.895 | 4.52254 / 4.57112 | 235.141 | 54.55 | valid candidate, not promoted |

三个对象均包含 22,765 个 classifier parameters。表中的 regression params 不包括非训练 QC reference/policy bytes。Pi 三次均为 `throttled=0x0`。

## 8. QC quality–coverage

v5 source：

- `results/iotj_b5_c5_runtime_v5_qc_20260725/qc_test_summary.csv`
- SHA256 `9901a8adf53f59b847497d19349d2ec9bd789fd141cb692919a9da4a1d8eb08e`

v4 comparison：

- `results/iotj_b5_c5_runtime_v5_qc_20260725/comparison_vs_runtime_v4.json`
- SHA256 `49f8a9b40a73bdcb601094e7ad17d6667c692c7791b0cfe636a01a32cf92f0a2`

| Runtime/workpoint | Accept/Review/Reject | Yield | Accepted RMSE (ppm) | Scope | Role |
|---|---:|---:|---:|---|---|
| v4 HC95 | 1323/33/4 | 97.2794% | 18.8520 | accept subset | formal baseline |
| v4 HC90 | 1235/107/18 | 90.8088% | 15.8328 | accept subset | formal baseline |
| v5 QC2 HC95 | 1275/41/44 | 93.7500% | 13.9178 | accept subset | candidate |
| v5 QC2 HC90 | 1183/113/64 | 86.9853% | 12.7723 | accept subset | candidate |

v5 accepted RMSE 更低但 coverage 更低；HC90 CO yield/accepted-RMSE guard 失败，因此不能按 RMSE 单项晋级。

## 9. Communication

### B5 Flower

来源 `results/iotj_final_system_benchmark_20260725/system_metrics/b5_fl_communication_summary.csv`，SHA256 `72832672a95192bd3789955c79812f70a2481de7ab96c321c7e7accb64303c7f`。

| Metric | Value | Scope |
|---|---:|---|
| measured application downlink | 8,764,300 bytes | seed42, 25 rounds, C1/C2 |
| measured application uplink | 8,808,350 bytes | same |
| measured application total | 17,572,650 bytes | same |
| transport bytes | not collected | 不得称为 wire traffic |

### Federated H1

来源 `results/iotj_final_system_benchmark_20260725/system_metrics/federated_h1_communication_summary.csv`，SHA256 `3b0a2d886870fb80de146f0bbce88fd9daf97f0035e2ed62113f6a0d1fd905ce`。

- one-shot sufficient-statistics theoretical serialized exchange：`7,710,128 bytes`；
- selected global H1 model：`50,226 bytes`；
- raw source rows/X/y、sample predictions/labels：未传输；
- secure aggregation / DP：未提供。

## 10. Calibration sensitivity

来源：

- `results/iotj_low_calibration_sensitivity_20260725/low_calibration_summary.csv`
- SHA256 `b110cadf495b0e3a93e61401b3e8d0186fe30223127610ed257a221cac23aa0b`

| Nominal rows | S_CC RMSE mean ± std (ppm) | Protocol role |
|---:|---:|---|
| 320 | `11.3416 ± 0.0000` | historical frozen 240/80 reference |
| 160 | `23.9156 ± 5.2550` | group-aware five replicates |
| 80 | `30.4799 ± 4.3262` | group-aware five replicates |
| 40 | `36.4992 ± 2.7156` | group-aware five replicates |

post-freeze harmonization 的 group-aware 320 为 `10.8724 ± 0.4391 ppm`，它是另一条 protocol sensitivity，不得替换 historical seed42 `11.3416`。来源：

- `results/iotj_calibration_protocol_harmonization_20260726/track_groupaware/groupaware_budget_summary.csv`
- SHA256 `63608d778f815360375271defe868d36be4d60309b78a1bf4a834c0818b70743`
- 用途：appendix/limitation。

## 11. 不能合并的“系统性能”

以下问题必须分别回答：

1. 分类稳定性：B5 five-seed Accuracy/Macro-F1；
2. 回归器能力：five-seed或 seed42 `S_CC`；
3. 端到端部署误差：seed42 `S_ALL`；
4. 选择性输出：指定 runtime/workpoint 的 accepted RMSE + yield；
5. 效率：指定平台、runtime、batch/thread/warmup/runs 的 latency/RSS；
6. 校准依赖：指定 calibration protocol/budget 的 sensitivity。

任何只写“GAPS RMSE = X”而不带上述 identity 的表述都不合格。
