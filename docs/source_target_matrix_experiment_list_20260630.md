# Source-Target Classification Matrix 2026-06-30

This matrix continues after the completed F6 baseline. All runs use Flower Stage-A classification with `fixed_da_strong`, 25 rounds, local epochs 5, batch size 32, and target calibration data on the server only.

## Completed Reference

| ID | Source | Target | Data Root | Status |
| --- | --- | --- | --- | --- |
| F6 | C1,C2 | C3,C4,C5 | `client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid` | completed |

## Remaining Runs

| ID | Source | Target | Data Root | Role |
| --- | --- | --- | --- | --- |
| F1 | C1 | C5 | `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` | forward C5 target |
| F2 | C1,C2 | C5 | `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` | forward C5 target |
| F3 | C1,C2,C3 | C5 | `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` | forward C5 target |
| F4 | C1,C2,C3,C4 | C5 | `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` | forward leave-one-target |
| F5 | C1 | C2,C3,C4,C5 | `client_data_c1src_c2345tgt_2080_timeaware_60_170_window_fullgrid` | single-source broad target |
| R1 | C5 | C1 | `client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` | reverse C1 target |
| R2 | C4,C5 | C1 | `client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` | reverse C1 target |
| R3 | C3,C4,C5 | C1 | `client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` | reverse C1 target |
| R4 | C2,C3,C4,C5 | C1 | `client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid` | reverse leave-one-target |

## Data Protocol

The same data root can be reused only when the run's source clients are a subset of the root's encoded source clients and target clients are a subset of the encoded target clients. For example, `client_data_c1234src_c5tgt_*` can run F1-F4 because C1/C2/C3/C4 are all source-style and C5 is target-style.

Do not use `client_data_c12src_c345tgt_*` for F3, because C3 is target-style in that root and would be incorrectly used as a source client.

## Launch Topology

- Server: Alibaba Cloud, `/root/GAPS`
- Clients: Raspberry Pi only, `/home/gaps/GAPS/flower_runtime`
- Client connection: `127.0.0.1:18080` on the Pi
- Tunnel chain: Pi `127.0.0.1:18080` -> PC `127.0.0.1:18080` -> ECS `127.0.0.1:8080`
- Result root: `/root/GAPS/results/source_target_classification_matrix_20260630`
- Pi client logs: `/home/gaps/GAPS/flower_runtime/results/source_target_classification_matrix_20260630_local_client_logs`

