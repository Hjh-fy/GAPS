# GAPS Flower Cloud-Edge Deployment Notes

## Verified Minimal Deployment

Date: 2026-06-01

Topology:
- Cloud: Alibaba Cloud ECS, Flower server, public IP `121.40.139.213`, port `8080`.
- Edge: Local PC simulating two edge clients, `client_id=1` and `client_id=2`.
- Data: Local PC only. Raw `.npy` client data was not uploaded to ECS.
- Framework: Flower minimal server/client API with a custom checkpointing FedAvg strategy.

## Verified Results

Communication and training:
- 2 clients connected to the ECS Flower server over the public network.
- 3 global rounds completed successfully.
- Each round completed fit and evaluate with `0 failures`.
- The cloud server received both client updates in every round.

Cloud checkpointing:
- Server checkpoints were saved on ECS:
  - `results/flower_server/server_round_001.pth`
  - `results/flower_server/server_round_002.pth`
  - `results/flower_server/server_round_003.pth`
  - `results/flower_server/server_latest.pth`
- `server_latest.pth` was copied back to the local PC and loaded successfully.
- The returned checkpoint contains `round=3` and `80` model tensors.

Local checkpoint evaluation:
- Checkpoint: `server_latest.pth`
- Evaluation data root: `dataset/client_data_federated_window_fullgrid_src12_tgt345`
- Evaluated clients: `1,2`

| Client | Examples | Accuracy |
|---:|---:|---:|
| 1 | 3360 | 0.5077 |
| 2 | 3360 | 0.5262 |
| Weighted | 6720 | 0.5170 |

Interpretation:
- The accuracy is not intended to match the paper's best result because this is a 3-round, 1-local-epoch deployment smoke test.
- The important verified capability is that cloud-generated model artifacts can be persisted, transferred, loaded, and evaluated locally.

## ECS Server Command

```bash
cd ~/GAPS && source ~/gaps_env/bin/activate && python -m gaps_flower.server_app --server-address 0.0.0.0:8080 --rounds 3 --min-clients 2 --output-dir results/flower_server
```

## Local Client Commands

```powershell
cd "D:\A Python learning\Federated Learning\TRAE SOLO"; python -m gaps_flower.client_app --server-address 121.40.139.213:8080 --client-id 1 --data-root "dataset/client_data_federated_window_fullgrid_src12_tgt345" --device cpu --local-epochs 1
```

```powershell
cd "D:\A Python learning\Federated Learning\TRAE SOLO"; python -m gaps_flower.client_app --server-address 121.40.139.213:8080 --client-id 2 --data-root "dataset/client_data_federated_window_fullgrid_src12_tgt345" --device cpu --local-epochs 1
```

## Copy Checkpoint Back To Local PC

```powershell
scp root@121.40.139.213:~/GAPS/results/flower_server/server_latest.pth "D:\A Python learning\Federated Learning\TRAE SOLO\server_latest.pth"
```

## Local Checkpoint Evaluation

```powershell
cd "D:\A Python learning\Federated Learning\TRAE SOLO"; python -m gaps_flower.evaluate_checkpoint --checkpoint server_latest.pth --data-root "dataset/client_data_federated_window_fullgrid_src12_tgt345" --client-ids 1,2 --device cpu --output results/flower_server_eval.json
```

## Current Deployment Boundary

This deployment layer currently verifies the minimal federated classification path:
- Flower server/client communication.
- Local PC as edge client.
- FedAvg aggregation.
- Cloud checkpoint persistence.
- Local checkpoint reuse.

The following are intentionally not yet migrated into Flower:
- Full paper-level experiment orchestration from `exp_improved.py`.
- T5/T6/T7/T8 QC and response anchoring post-processing.
- Regression model B and route/client-aware reliability output.
- Server-side domain adaptation and prototype aggregation.
- Raspberry Pi hardware runtime.

## Next Engineering Steps

1. Keep the current Flower layer as the deployment smoke-test baseline.
2. Add optional checkpoint evaluation on each server round if cloud-side test data is intentionally provided.
3. Add command-line switches for `with_reg_head` and a regression-capable deployment mode.
4. Migrate custom aggregation only after the FedAvg cloud-edge path remains stable.
5. Keep QC/risk-accepted response logic as a post-training deployment inference module unless real-time QC feedback becomes necessary.
6. When Raspberry Pi arrives, copy only the edge runtime files and local data shard; do not copy `results/`, paper figures, or intermediate analysis scripts.

## Raspberry Pi Migration Target Command

```bash
python -m gaps_flower.client_app --server-address 121.40.139.213:8080 --client-id 1 --data-root /home/pi/gaps_data --device cpu --local-epochs 1
```
