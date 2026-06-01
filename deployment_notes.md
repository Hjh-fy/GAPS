# GAPS Flower Cloud-Edge Deployment Notes

## Verified Minimal Deployment

Date: 2026-06-01

Topology:
- Cloud: Alibaba Cloud ECS, Flower server, public IP `121.40.139.213`, port `8080`.
- Edge: Local PC simulating two edge clients, `client_id=1` and `client_id=2`.
- Data: Local PC only. Raw `.npy` client data was not uploaded to ECS.

Verified results:
- 2 clients connected to the ECS Flower server over the public network.
- 3 global rounds completed successfully.
- Each round completed fit and evaluate with `0 failures`.
- Server checkpoints were saved on ECS:
  - `results/flower_server/server_round_001.pth`
  - `results/flower_server/server_round_002.pth`
  - `results/flower_server/server_round_003.pth`
  - `results/flower_server/server_latest.pth`
- `server_latest.pth` was copied back to the local PC and loaded successfully.

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
