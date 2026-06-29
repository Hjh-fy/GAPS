# Communication Compression Experiment

Goal: verify whether fewer Flower communication rounds plus strong server-side DA can approach the 25-round strong-DA classification baseline.

## Experiment Matrix

| Profile | Flower communication rounds | Per-round DA steps | Final post-DA steps | Purpose |
|---|---:|---:|---:|---|
| `25R_strong_DA` | 25 | 100 | 0 | baseline |
| `10R_strong_DA` | 10 | 100 | 0 | compressed communication |
| `10R_postDA300` | 10 | 100 | 300 | compressed + server-only final adaptation |
| `10R_postDA500` | 10 | 100 | 500 | stronger server-only final adaptation |

Only the first two profiles require Flower client-server training. The `postDA300` and `postDA500` profiles start from the 10R final adapted checkpoint and run on the server only, using source calibration and target calibration splits.

## Data And Roles

Use the current F6 direction:

```text
source clients: C1 + C2
target clients: C3 + C4 + C5
data root: client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid
```

Recommended system split:

```text
Alibaba Cloud: Flower server and post-DA
Raspberry Pi: C1 source client
PC: C2 source client
```

## Generate Server Commands

Run on PC:

```powershell
python generate_flower_matrix_commands.py `
  --matrix-config configs/communication_compression_matrix_20260630.json `
  --runs all `
  --remote-project-dir /root/GAPS `
  --remote-python "source ~/gaps_env/bin/activate && python" `
  --remote-data-root /root/GAPS/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --remote-output-root results/communication_compression_20260630 `
  --server-public-address <ECS_IP>:8080 `
  --local-python python `
  --local-data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" `
  --local-device cuda `
  --da-preset fixed_da_strong `
  --output-dir results/communication_compression_20260630_commands
```

Validate each generated manifest:

```powershell
python validate_matrix_command_config.py results/communication_compression_20260630_commands/CC_F6_C12_to_C345_R25_STRONG_DA/command_manifest.json
python validate_matrix_command_config.py results/communication_compression_20260630_commands/CC_F6_C12_to_C345_R10_STRONG_DA/command_manifest.json
```

## Run One Profile

For each profile, start the Alibaba Cloud server first. Example for 10R:

```bash
cd /root/GAPS
bash results/communication_compression_20260630_commands/CC_F6_C12_to_C345_R10_STRONG_DA/server_command.sh
```

Start C1 on Raspberry Pi:

```bash
cd ~/GAPS
source ~/GAPS/gaps_rpi_env/bin/activate
python -m gaps_flower.client_app \
  --server-address <ECS_IP>:8080 \
  --client-id 1 \
  --data-root ~/GAPS/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid \
  --device cpu \
  --local-epochs 5 \
  --batch-size 32 \
  --profile strong_cls
```

Start C2 on PC:

```powershell
python -m gaps_flower.client_app `
  --server-address <ECS_IP>:8080 `
  --client-id 2 `
  --data-root "D:/A Python learning/Federated Learning/TRAE SOLO/dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid" `
  --device cuda `
  --local-epochs 5 `
  --batch-size 32 `
  --profile strong_cls
```

Acceptance criteria for every Flower run:

```text
all rounds: 2 fit results / 0 failures
all rounds: 2 evaluate results / 0 failures
server_latest.pth exists
server_latest_adapted.pth exists
domain_adapt_round_*.json exists for every round
run_config.json records da_preset == fixed_da_strong
```

## Run Final Post-DA On Alibaba Cloud

After `CC_F6_C12_to_C345_R10_STRONG_DA` finishes, run:

```bash
cd /root/GAPS
source ~/gaps_env/bin/activate

python run_final_post_da.py \
  --checkpoint results/communication_compression_20260630/CC_F6_C12_to_C345_R10_STRONG_DA/server_latest_adapted.pth \
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid \
  --source-client-ids 1,2 \
  --target-client-ids 3,4,5 \
  --steps 300 \
  --output-checkpoint results/communication_compression_20260630/CC_F6_C12_to_C345_R10_STRONG_DA/server_latest_postda300.pth \
  --output-diagnostics results/communication_compression_20260630/CC_F6_C12_to_C345_R10_STRONG_DA/domain_adapt_postda300.json \
  --device cuda

python run_final_post_da.py \
  --checkpoint results/communication_compression_20260630/CC_F6_C12_to_C345_R10_STRONG_DA/server_latest_adapted.pth \
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid \
  --source-client-ids 1,2 \
  --target-client-ids 3,4,5 \
  --steps 500 \
  --output-checkpoint results/communication_compression_20260630/CC_F6_C12_to_C345_R10_STRONG_DA/server_latest_postda500.pth \
  --output-diagnostics results/communication_compression_20260630/CC_F6_C12_to_C345_R10_STRONG_DA/domain_adapt_postda500.json \
  --device cuda
```

If CUDA is not available on the server, use `--device cpu`.

## Summarize Target Generalization

Run on the machine that has the result folders and dataset:

```powershell
python summarize_comm_compression_classification.py `
  --matrix-config configs/communication_compression_matrix_20260630.json `
  --profiles-config configs/communication_compression_profiles_20260630.json `
  --results-root results/communication_compression_20260630 `
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --output-dir results/communication_compression_20260630_summary `
  --device cuda
```

Outputs:

```text
per_round_target_metrics.csv
final_profile_metrics.csv
profile_client_metrics.csv
communication_compression_report.md
```

The main claim is supported if `10R_strong_DA`, `10R_postDA300`, or `10R_postDA500` keeps target accuracy and macro-F1 close to `25R_strong_DA` while reducing communication rounds from 25 to 10.
