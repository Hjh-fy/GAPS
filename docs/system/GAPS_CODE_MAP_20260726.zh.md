# GAPS 最终系统代码图谱（2026-07-26）

> 本图谱以当前 canonical call graph 为准。`safe to delete=否` 表示不能因“不是最终模型”就删除；`归档后可移出主线` 仍需 Git/SHA 可恢复。

## 1. Data contract

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `federated_dataset.py` | dataset/loaders | `gaps_flower.task`、regression scripts | client split arrays/metadata | PyTorch loaders | active | canonical shared | 否 |
| `gaps_flower/task.py` | `make_config`, `create_model`, `load_client_loaders`, `train_one_round`, `evaluate` | Flower client/server | config、client data | model/parameters/metrics | active | canonical | 否 |
| `scripts/audit_iotj_experiment_inputs.py` | input audit CLI | formal experiment preparation | dataset/config/checkpoints | audit JSON | active audit | canonical | 否 |
| `scripts/build_iotj_c2_dataset_subset_manifest.py` | C2 subset manifest builder | topology preparation | C2 data root | subset manifest | active | canonical topology | 否 |
| `scripts/generate_iotj_ecs_c2_topology_manifest.py` | topology manifest builder | controller preflight | host/data identities | execution topology manifest | active | canonical topology | 否 |
| `scripts/build_iotj_b5_c5_runtime_row_map.py` | row-map builder | v4 bundle preparation | 320/1360 metadata | deterministic row map | frozen | canonical v4 | 否 |
| `gaps_deploy/rich_residual.py` | `target_ridge_features` and policy helpers | v4/v5 runtime、benchmark | `(100,8)` window + metadata | 104D rich feature dict | active | canonical shared feature contract | 否 |

## 2. B5 classification

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `gaps_flower/client_app.py` | `GapsFlowerClient`, `main` | controller / direct client command | C1/C2 local data、round config | FitRes、local statistics | frozen training implementation | canonical B5 | 否 |
| `gaps_flower/server_app.py` | `main`, `fit_config` | controller / direct server command | command manifest、C5 calibration | round checkpoints、run_config | frozen training implementation | canonical B5 | 否 |
| `gaps_flower/strategy.py` | `canonicalize_fit_results`, `GapsStrategy` | `server_app` | FitRes、prototypes、stats | FedAvg result、DA checkpoints | active frozen semantics | canonical B5 | 否 |
| `gaps_flower/evaluate_checkpoint.py` | checkpoint evaluation helpers | evaluation scripts | frozen checkpoint、C5 split | logits/routes/metrics | active, test-reading | canonical evaluation | 否 |
| `scripts/freeze_iotj_confirmation_protocol.py` | manifest/archive builders | operator before training | commit、dataset、source tree | `results/c2e_*` | completed | canonical provenance | 否 |
| `scripts/run_iotj_confirmation_observability.py` | `load_frozen_inputs`, `preflight_frozen_run`, `run_confirmation_attempt`, `main` | local controller | frozen manifests/archive/topology | remote attempts、audit evidence | completed formal controller | canonical | 否 |
| `scripts/validate_iotj_confirmation_attempt.py` | attempt validator | controller/postflight | attempt sidecars/checkpoints | `attempt_audit.json` | active validation | canonical | 否 |
| `scripts/evaluate_iotj_b5_multiseed_seed.py` | `main` | per-seed postflight | checkpoint、data、row map | metrics + 1360 route CSV | completed, test-reading | canonical evaluation | 否 |
| `scripts/summarize_iotj_b5_multiseed_classification.py` | `main` | final classification closeout | five per-seed outputs | `results/iotj_b5_multiseed_20260724/` summaries | completed | canonical evidence | 否 |

## 3. Server DA and observability

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `gaps_flower/domain_adaptation.py` | `ServerDomainAdaptation`, `cross_domain_same_class_phase_mmd2`, `wasserstein_feature_objective` | `GapsStrategy` | aggregate model、C5 calibration | adapted global model | frozen | canonical B5 | 否 |
| `gaps_flower/domain_adaptation_inputs.py` | DA input contracts | server DA | calibration/source loaders | validated batches | active | canonical | 否 |
| `gaps_flower/observability.py` | observer/event API | client/server/controller | Flower events/resources | JSONL/sidecars | completed formal evidence | canonical diagnostics | 否 |
| `gaps_flower/flower_message_audit.py` | application payload audit | strategy/client | Flower messages | logical/application byte records | completed | canonical system evidence | 否 |
| `scripts/sample_iotj_process_resources.py` | resource sampler CLI | controller | process identity | resource JSONL | completed | canonical system evidence | 否 |

## 4. H1 sufficient statistics

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `scripts/materialize_iotj_federated_h1_topology.py` | `client_moments`, `server_scalers`, `client_equations`, `server_candidates`, `client_scores`, `server_model` | staged C1/C2/server workflow | local C1/C2 data or statistics JSON | `results/iotj_b5_c5_runtime_v5_candidate_20260724/federated_h1/` | completed real topology | canonical H1 builder | 否 |
| `scripts/evaluate_iotj_h1_federated_ridge_equivalence.py` | local statistics classes、`fit_federated_h1`, `run` | formal equivalence audit | C1/C2 data、v4 reference assets | `results/iotj_h1_federated_ridge_equivalence_20260724/` | completed | canonical audit | 否 |
| `results/.../federated_h1/global_h1_model.json` | serialized per-gas Ridge | runtime-v5 builder | sufficient statistics | H1 runtime asset | frozen large/local asset | canonical asset | 否；先外部归档 |
| `gaps_flower/regression_task.py` | legacy regression FedAvg helpers | historical scripts | source data/checkpoints | regression model | not current H1 path | legacy/diagnostic | 归档后可移出主线 |
| `gaps_flower/regression_client.py` / `regression_server.py` | file-mediated regression helpers | historical topology audit | checkpoint files | local/global checkpoint | no real Flower regression closure | diagnostic | 归档后可移出主线 |

## 5. Target 105D Ridge and multi-seed decision

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `scripts/evaluate_iotj_b5_regression_multiseed.py` | `fit_seed_calibration`, `apply_models`, `final_gate`, `run` | formal RG0/RG1/RG2 evaluation | five B5 routes、H1/H2/H3、C5 calibration | `results/iotj_b5_regression_multiseed_20260724/` | completed, test-reading after lock | canonical selection | 否 |
| `scripts/build_iotj_runtime_v5_candidate.py` | `freeze_calibration`, `evaluate_test`, `materialize_required_outputs`, `build_bundle`, parity/finalize | staged v5 closeout | B5 seed42、real/audited H1、C5 data | target Ridge、bundle、parity | completed | canonical v5 builder | 否 |
| `results/.../target_ridge/target_ridge_105d_manifest.json` | four per-gas Ridge models | v5 builder/runtime | 105D feature schema | target model asset | frozen | canonical asset | 否；先外部归档 |

## 6. Bundle build and validation

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `gaps_deploy/c5_federated_source_ridge_bundle.py` | `FederatedSourceRidgeBundle`, `load_federated_source_ridge_bundle` | v5 runtime/builder | manifest + three assets | verified asset paths | active | canonical v5 | 否 |
| `scripts/build_iotj_runtime_v5_candidate.py` | `build_bundle` | operator | classifier/H1/target Ridge | `runtime_v5/{assets,bundle_manifest,runtime_contract}` | completed | exact v5 builder | 否 |
| `scripts/build_iotj_b5_c5_deployment_bundle.py` | `build_bundle` | older P1/v4 flow | P1 input audit | v4 C5 H8 bundle | frozen historical builder | canonical v4 provenance | 否 |
| `scripts/prepare_iotj_b5_c5_runtime_contract.py` | `prepare_runtime_contract` | v4 P1 flow | bundle、C5 arrays、HC refs | v4 runtime contract | completed | canonical v4 | 否 |
| `scripts/validate_iotj_b5_c5_runtime_parity.py` | `validate_parity`, `validate_c5_h8_parity` | v4 parity flow | offline/runtime rows | parity report | completed | canonical v4 audit | 否 |

## 7. Runtime v5

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `gaps_deploy/c5_federated_source_ridge_runtime.py` | `SerializedRidgeV5`, `C5FederatedSourceRidgeRuntime` | tests、benchmark、QC wrapper | runtime contract、windows、metadata、phase | B5 route、H1 ppm、target ppm | final simplified regression | canonical v5 core | 否 |
| `gaps_deploy/c5_federated_source_ridge_qc_runtime.py` | `C5FederatedSourceRidgeQCRuntime` | benchmark/tests | v5 core + QC2 contract | ppm、risk、decision、auto output | valid candidate not promoted | canonical candidate | 否 |
| `scripts/build_iotj_runtime_v5_candidate.py` | `calibration_runtime_parity`, `runtime_parity` | v5 closeout | offline reference + runtime | 320/1360 parity reports | completed | canonical parity | 否 |

Runtime v5 当前没有独立通用 inference CLI；正式 API 是 Python class。不要把 benchmark CLI 当作部署 CLI。

## 8. Runtime v4

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `gaps_deploy/c5_h8_bundle.py` | v4 manifest/asset loader | C5H8Runtime | v4 bundle manifest | verified assets | formal baseline | canonical v4 | 否 |
| `gaps_deploy/c5_h8_runtime.py` | `C5H8Runtime` | benchmark/parity | B5/H1/H2/H3/H2.3/QC assets | v4 ppm + decision | formal selective-output baseline | canonical v4 | 否 |
| `gaps_deploy/final_runtime.py` | `FinalDeployRuntime`, CLI `main` | historical package users | C12→C345 package | nine-field legacy output | maintained historical wrapper | v4/legacy internals | 否 |
| `gaps_deploy/inference.py` | `DeployPredictor` | `final_runtime.py` | legacy client package | R3aK16/AutoV2 result | not v5 | legacy v4 dependency | 否，除非 v4 retired |

## 9. QC

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `gaps_deploy/qc_policy.py` | legacy/v4 QC policy | v4 runtime | deployment-visible risk | accept/review/reject | formal v4 | canonical v4 | 否 |
| `gaps_deploy/runtime_v5_qc.py` | fold/reference/ECDF/MAD/decision policy | v5 QC evaluator/runtime | calibration-only references | QC2 policy + risk | valid candidate | canonical v5 QC | 否 |
| `gaps_deploy/runtime_v5_qc_bundle.py` | `RuntimeV5QCBundle`, loader | v5 QC runtime | QC bundle manifest | verified policy/assets | active candidate | canonical v5 QC | 否 |
| `scripts/evaluate_iotj_runtime_v5_qc.py` | `freeze_protocol`, `calibrate_and_lock`, `evaluate_test` | staged QC closeout | v5 core、C5 calibration/test | `results/iotj_b5_c5_runtime_v5_qc_20260725/` | completed; no rerun | canonical evidence | 否 |

## 10. Benchmark and reporting

| Path | Public function/class | Caller | Inputs | Outputs / result directory | Status | Identity | Safe to delete |
|---|---|---|---|---|---|---|---|
| `scripts/prepare_iotj_final_benchmark_package.py` | `prepare` | local release prep | frozen runtime assets | portable Pi package | completed | canonical benchmark support | 否 |
| `scripts/benchmark_iotj_final_runtime.py` | `benchmark`, `latency_statistics` | PC/Pi operator | runtime contract + C5 test windows | per-runtime JSON/row timings | completed | exact final steady-state benchmark | 否 |
| `scripts/probe_iotj_runtime_cold_start.py` | `parent`, `child` | PC/Pi operator | runtime/contract/data | cold-start JSON | completed | canonical benchmark | 否 |
| `scripts/build_iotj_final_system_evidence.py` | `verify_frozen_assets`, `build` | final evidence builder | frozen results | system tables/figures | completed | canonical reporting | 否 |
| `scripts/finalize_iotj_final_system_evidence.py` | `finalize` | closeout | benchmark root | report/index/SHA | completed | canonical reporting | 否 |
| `scripts/freeze_iotj_paper_evidence.py` | evidence/table/figure/manuscript builders | paper freeze | approved results | `docs/paper_evidence_freeze/` + frozen HTML | completed | canonical paper evidence | 否 |

## 11. Tests

| Test family | Protects | Status | Safe to delete |
|---|---|---|---|
| `tests/test_iotj_b5_multiseed_protocol.py` | B5 seed protocol/route identities | active | 否 |
| `tests/test_iotj_h1_federated_ridge_equivalence.py` | H1 statistics equivalence | active | 否 |
| `tests/test_iotj_runtime_v5_candidate_builder.py` | v5 staged builder and locks | active | 否 |
| `tests/test_runtime_v5_qc_policy.py` / `test_runtime_v5_qc_runtime.py` | v5 QC policy/runtime | active | 否 |
| `tests/test_iotj_final_system_benchmark.py` | benchmark contracts | active | 否 |
| `tests/test_validate_iotj_b5_c5_runtime_parity.py` | v4 parity | active | 否 |

测试产生的 `.tmp_*`、`.pytest_cache`、`__pycache__` 不属于代码资产，完成验证后可作为可重建删除候选。
