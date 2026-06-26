# Flower Matrix Stage-A Runbook — Split-Aware Version

This runbook defines how to execute the Stage-A source-target classification matrix with Alibaba Cloud/ECS as the Flower server and local PC processes as source clients.

## 1. Core principle

The matrix must use the current Flower migration code path:

```text
gaps_flower.server_app
gaps_flower.client_app
gaps_flower.strategy
gaps_flower.task
gaps_flower.domain_adaptation
```

Do not fall back to the old single-machine pipeline for official matrix results.

## 2. Dataset role rule

The dataset root must match source/target roles, but it can be reused when the current run is a subset of the encoded role split.

The project convention is:

```text
source clients: 7:2:1 train/calibration/test split
target clients: 2:8 calibration/test split
```

Therefore, a root named like:

```text
client_data_c1234src_c5tgt_...
```

means:

```text
C1/C2/C3/C4 use source-style 7:2:1 split
C5 uses target-style 2:8 split
```

It can be reused for all runs whose source set is a subset of `{1,2,3,4}` and target set is a subset of `{5}`:

```text
F1_C1_to_C5
F2_C12_to_C5
F3_C123_to_C5
F4_C1234_to_C5
```

Likewise:

```text
client_data_c2345src_c1tgt_...
```

can be reused for:

```text
R1_C5_to_C1
R2_C45_to_C1
R3_C345_to_C1
R4_C2345_to_C1
```

And:

```text
client_data_c1src_c2345tgt_...
```

is needed for:

```text
F5_C1_to_C2345
```

The existing `client_data_c12src_c345tgt_...` root is appropriate for:

```text
F1_C1_to_C5
F2_C12_to_C5
F6_C12_to_C345
```

but it is not appropriate for `F3_C123_to_C5`, `F4_C1234_to_C5`, or reverse-direction runs because C3/C4 are target-style in that root, not source-style.

## 3. Current matrix config

The 10 core combinations are defined in:

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

## 4. Script files

| File | Purpose |
|---|---|
| `configs/source_target_matrix_20260627.json` | 10-run source-target matrix config |
| `generate_flower_matrix_commands.py` | generate per-run server/client commands |
| `run_local_flower_matrix_clients.py` | launch local source clients for one run |

The command generator parses split tags like `c12src_c345tgt` and warns if the run's source/target clients are outside the encoded role sets. It allows subset reuse.

## 5. First reference run: F6_C12_to_C345

Use this first because it matches the current mainline direction.

### 5.1 Generate commands on local PC

```bash
python generate_flower_matrix_commands.py ^
  --matrix-config configs/source_target_matrix_20260627.json ^
  --runs F6_C12_to_C345 ^
  --remote-project-dir /root/GAPS ^
  --remote-python "source ~/gaps_env/bin/activate && python" ^
  --remote-data-root /root/GAPS/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid ^
  --remote-output-root results/source_target_classification_matrix_20260627 ^
  --server-public-address <ECS_IP>:8080 ^
  --local-python python ^
  --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" ^
  --local-device cuda
```

### 5.2 Start server on ECS

Copy and run:

```text
results/source_target_classification_matrix_20260627_commands/F6_C12_to_C345/server_command.sh
```

The command should include:

```text
--profile strong_cls
--min-clients 2
--server-val-data ...client_1,...client_2
--server-calib-data ...client_3,...client_4,...client_5
--use-domain-adapt true
--use-adapted-as-global true
```

### 5.3 Start local source clients

```bash
python run_local_flower_matrix_clients.py ^
  --matrix-config configs/source_target_matrix_20260627.json ^
  --run-id F6_C12_to_C345 ^
  --server-address <ECS_IP>:8080 ^
  --local-project-dir "D:/A Python learning/Federated Learning/TRAE SOLO" ^
  --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" ^
  --local-python python ^
  --device cuda ^
  --wait
```

## 6. C5 source-count scaling group

After the reference run passes, prepare or use:

```text
client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid
```

Then generate commands for:

```text
F1_C1_to_C5,F2_C12_to_C5,F3_C123_to_C5,F4_C1234_to_C5
```

Example:

```bash
python generate_flower_matrix_commands.py ^
  --matrix-config configs/source_target_matrix_20260627.json ^
  --runs F1_C1_to_C5,F2_C12_to_C5,F3_C123_to_C5,F4_C1234_to_C5 ^
  --remote-project-dir /root/GAPS ^
  --remote-python "source ~/gaps_env/bin/activate && python" ^
  --remote-data-root /root/GAPS/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid ^
  --remote-output-root results/source_target_classification_matrix_20260627 ^
  --server-public-address <ECS_IP>:8080 ^
  --local-python python ^
  --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid" ^
  --local-device cuda
```

Run each server/client pair sequentially. Do not run multiple servers on port 8080 at the same time.

## 7. Reverse C1 source-count scaling group

Prepare or use:

```text
client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid
```

Then run:

```text
R1_C5_to_C1
R2_C45_to_C1
R3_C345_to_C1
R4_C2345_to_C1
```

## 8. Single-source multi-target group

Prepare or use:

```text
client_data_c1src_c2345tgt_2080_timeaware_60_170_window_fullgrid
```

Then run:

```text
F5_C1_to_C2345
```

## 9. Stage-A acceptance criteria

For each successful run, the ECS output directory should contain at least:

```text
history.json
server_latest.pth
server_latest_adapted.pth
client_stats_round_*.json
prototype_stats_round_*.json
```

If `server_latest_adapted.pth` is missing, check:

```text
--use-domain-adapt true
--domain-adapt-warmup < rounds
--server-val-data paths exist
--server-calib-data paths exist
--strict-calibration-split true can find calibration files
```

## 10. Next required script

This runbook only gets training/checkpoints/logs. The next needed script is:

```text
summarize_flower_matrix_classification.py
```

It should evaluate each saved checkpoint on target client test sets and output:

```text
per_round_target_metrics.csv
final_target_metrics.csv
source_count_scaling.csv
classification_matrix_report.md
```

Required metrics:

```text
before/after DA target accuracy
DA gain
macro-F1
ECE
NLL
per-class accuracy
confusion matrix
```
