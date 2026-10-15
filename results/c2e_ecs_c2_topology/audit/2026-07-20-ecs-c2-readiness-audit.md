# ECS-C2 execution-topology readiness audit

**Date:** 2026-07-20
**Scope:** Read-only readiness audit for the proposed `Pi C1 + ECS-hosted C2 + existing ECS server/DA` topology. This is not experiment evidence and does not approve a canonical run.

## Inputs inspected

- Frozen algorithm commit: `2ef7aea77b9dfabdd09da4f38742907a37c58c30`.
- Immutable source archive SHA-256: `52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`.
- `results/c2e_summary/confirmation_protocol_manifest.json`: ten distinct frozen `run_id -> algorithm_config_sha256` entries.
- `results/c2e_summary/dataset_manifest.json`: 26 files across multiple client roles, including C1 and C2.
- New C2 host `root@114.55.171.63`, checked read-only by SSH.
- Current controller and validator source: `scripts/run_iotj_confirmation_observability.py` and `scripts/validate_iotj_confirmation_attempt.py`.

## Environment finding

The new C2 host has Python 3.10.12, `torch==2.12.0+cpu`, `flwr==1.23.0`, and `psutil==7.0.0` in `/root/gaps_c2_cpu_env`; it has 1.6 GiB RAM, an active 2 GiB swapfile, and 24 GiB free root-disk space at audit time. This is sufficient to proceed to an isolated smoke-preflight implementation, not evidence that a 25-round run will complete within a particular time.

An earlier CUDA-oriented environment at `/root/gaps_c2_env` is an unused provisioning attempt. It is not an approved runtime and must never be selected silently.

## Blocking findings

| ID | Finding | Evidence | Required minimal resolution |
|---|---|---|---|
| B1 | The controller deploys and launches C2 only as local PC state. | `deploy_source_archive`, `ProductionRuntime.pc_runtime_root`, `launch_pc_client`, `launch_pc_sampler` in `scripts/run_iotj_confirmation_observability.py`. | Add an explicit `ecs_c2` placement while retaining legacy PC behaviour unchanged. |
| B2 | The validator requires `pc-c2` evidence under `raw/pc`. | `REQUIRED_CONTEXTS` and C2 aggregation in `scripts/validate_iotj_confirmation_attempt.py`. | Permit `ecs-c2` plus `raw/ecs_c2` only when a validated ECS-C2 topology manifest is present. |
| B3 | Existing tunnel creation publishes the local server tunnel only to the Pi reverse tunnel. | `_start_tunnels` in `scripts/run_iotj_classification_cloud_edge.py`. | The confirmation controller must own a second loopback-only reverse tunnel for C2 before remote C2 starts; do not expose Flower port 8080 publicly. |
| B4 | The full dataset manifest cannot serve as a C2-only host file-presence gate. | `dataset_manifest.json` includes C1 and C2 files; current preflight sends that complete manifest to every host. | Generate a C2 subset manifest with only `client_2/**` entries and bind its hash in the topology manifest. |
| B5 | A single algorithm-config hash is insufficient. | Protocol manifest contains ten different configuration hashes. | Bind and validate the complete ten-entry `run_id -> algorithm_config_sha256` map, both in the manifest generator and controller. |

## Major risks (not blockers to implementation)

| ID | Risk | Mitigation / decision gate |
|---|---|---|
| M1 | C2 has only 2 vCPU and 1.6 GiB RAM. | Record C2 sampler RSS/CPU; use B2 then B5 two-round smoke to determine actual viability before any 25-round queue. |
| M2 | CPU PyTorch differs from historical Windows placement. | Treat it as execution-topology evidence; preserve identical frozen source/config/data and never pool timing with the PC topology. |
| M3 | Pi may be connected through the PC hotspot. | Record this transport condition in host contexts and smoke summary; it is not interchangeable with the laboratory Wi-Fi pilot. |

## Audit decision

**Status: blocked for archive/data transfer and smoke execution until B1--B5 have tests and pass.** The frozen algorithm remains intact, and the proposed placement is admissible as an execution-topology change once the controller/validator contracts are implemented. No model, loss, data protocol, optimizer, B2/B5 setting, or server-DA definition needs to change.

## Evidence and claim boundary

Any future B2/B5 two-round output is a noncanonical topology smoke. It may support only operational readiness and diagnostic timing/resource observations. It cannot be a five-seed algorithm result, a PC-edge deployment result, or a replacement for the prior real `ECS + Pi + Windows PC` system pilot.
