# Flower Matrix Stage-A Runbook

This runbook defines how to execute the Stage-A source-target classification matrix with Alibaba Cloud/ECS as the Flower server and local PC processes as source clients.

## 1. Core Principle

Official matrix runs should use the current Flower migration path:

```text
gaps_flower.server_app
gaps_flower.client_app
gaps_flower.strategy
gaps_flower.task
gaps_flower.domain_adaptation
```

Do not fall back to the old single-machine pipeline for official matrix results.

## 2. Dataset Role Rule

The dataset root must match source/target roles. A split root named like:

```text
client_data_c12src_c345tgt_...
```

means C1/C2 use source-style train/calibration/test splits and C3/C4/C5 use target-style calibration/test splits.

The command generator parses tags like `c12src_c345tgt` and warns if a run uses clients outside the encoded role sets. Subset reuse is allowed. For example, `c12src_c345tgt` is appropriate for `F1_C1_to_C5`, `F2_C12_to_C5`, and `F6_C12_to_C345`, but it is not appropriate for reverse-direction runs where C3/C4/C5 must be source clients.

## 3. Matrix Config

The core source-target combinations are defined in:

```text
configs/source_target_matrix_20260627.json
```

Recommended groups:

| Group | Runs | Required role-aware data root |
|---|---|---|
| current forward reference | F1, F2, F6 | `c12src_c345tgt` is enough |
| C5 source-count scaling | F1, F2, F3, F4 | prefer `c1234src_c5tgt` |
| reverse C1 source-count scaling | R1, R2, R3, R4 | prefer `c2345src_c1tgt` |
| single-source multi-target | F5 | require `c1src_c2345tgt` |

## 4. Script Files

| File | Purpose |
|---|---|
| `configs/source_target_matrix_20260627.json` | source-target matrix config |
| `generate_flower_matrix_commands.py` | generate per-run server/client commands and manifest |
| `run_local_flower_matrix_clients.py` | launch local source clients for one run |
| `validate_matrix_command_config.py` | validate fixed-DA strong command manifests before long runs |
| `summarize_flower_matrix_classification.py` | summarize saved target classification metrics |

## 5. First Reference Run: F6_C12_to_C345 Fixed-DA Strong R25

Use F6 first because it matches the current mainline direction and the existing role-aware split.

### 5.1 Generate Commands On Local PC

```powershell
python generate_flower_matrix_commands.py `
  --matrix-config configs/source_target_matrix_20260627.json `
  --runs F6_C12_to_C345 `
  --remote-project-dir /root/GAPS `
  --remote-python "source ~/gaps_env/bin/activate && python" `
  --remote-data-root /root/GAPS/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --remote-output-root results/source_target_classification_matrix_20260627 `
  --server-public-address <ECS_IP>:8080 `
  --local-python python `
  --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" `
  --local-device cuda `
  --da-preset fixed_da_strong `
  --run-suffix _fixed_da_strong_r25
```

Expected output directory:

```text
results/source_target_classification_matrix_20260627_commands/F6_C12_to_C345_fixed_da_strong_r25/
```

### 5.2 Validate The Manifest

```powershell
python validate_matrix_command_config.py results/source_target_classification_matrix_20260627_commands/F6_C12_to_C345_fixed_da_strong_r25/command_manifest.json
```

The validator should print `OK`. The manifest should include:

```text
--da-preset fixed_da_strong
--domain-adapt-steps 100
--domain-adapt-warmup 0
--da-use-adversarial true
--da-lambda-coral 0.5
--da-lambda-adv 0.5
--da-server-opt-lr 0.0005
--use-adapted-as-global true
```

### 5.3 Start Server On ECS

Copy and run:

```text
results/source_target_classification_matrix_20260627_commands/F6_C12_to_C345_fixed_da_strong_r25/server_command.sh
```

The ECS output path should be:

```text
/root/GAPS/results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/
```

### 5.4 Start Local Source Clients

```powershell
python run_local_flower_matrix_clients.py `
  --matrix-config configs/source_target_matrix_20260627.json `
  --run-id F6_C12_to_C345 `
  --server-address <ECS_IP>:8080 `
  --local-project-dir "D:/A Python learning/Federated Learning/TRAE SOLO" `
  --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" `
  --local-python python `
  --device cuda `
  --wait
```

Run one server and its matching client set at a time. Do not run multiple servers on port 8080 concurrently.

## 6. Acceptance Criteria

For each successful run, the ECS output directory should contain:

```text
history.json
run_config.json
server_latest.pth
server_latest_adapted.pth
client_stats_round_*.json
prototype_stats_round_*.json
domain_adapt_round_*.json
```

The server log should show every round with the expected number of source-client results and `0 failures`.

For F6 fixed-DA strong R25, `run_config.json` should record:

```text
da_preset == fixed_da_strong
use_domain_adapt == true
domain_adapt_warmup == 0
rounds > warmup
strict_calibration_split == true
```

## 7. Target Summary

After a run finishes, evaluate target held-out metrics:

```powershell
python summarize_flower_matrix_classification.py `
  --run-dir results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25 `
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --target-clients 3,4,5
```

Expected outputs:

```text
target_summary/per_round_target_metrics.csv
target_summary/final_target_metrics.csv
target_summary/classification_matrix_report.md
```

Report the official final checkpoint first. If a best-round checkpoint is used for oracle analysis, label it explicitly as best-checkpoint analysis rather than official final performance.
