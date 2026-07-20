# ECS-C2 + Pi-C1 execution-topology design

**Date:** 2026-07-20
**Status:** user-approved design; implementation requires this document review.
**Scope:** replace the unstable Windows-PC placement of logical client C2 with a
separate Alibaba Cloud ECS host. This is an execution-topology change only.

## Objective

Run the frozen C1/C2 -> C5 B2/B5 confirmation protocol with physical Raspberry
Pi C1 and a separate ECS-hosted C2. The existing ECS remains the Flower server
and executes server-side DA. The new C2 host is `root@114.55.171.63`.

This removes the observed Windows C2 process instability from the algorithm
confirmation path without changing the classifier algorithm.

## Non-negotiable invariants

- Keep the immutable algorithm source archive
  `52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`
  and algorithm commit `2ef7aea77b9dfabdd09da4f38742907a37c58c30` unchanged.
- Keep C1 and C2 datasets, sample identities, C5 evaluation split, B2/B5
  profile, model, loss, optimizer, rounds (25), local epochs (5), batch size,
  learning-rate configuration, and server DA unchanged.
- Preserve all existing PC-topology attempts as failed or diagnostic evidence;
  never overwrite, rename as canonical, or aggregate them with new topology
  results.
- Do not reuse PC-topology formal-smoke evidence as validation of the new C2
  host. New topology requires its own preflight and B2/B5 smoke evidence.

## Alternatives considered

1. **Recommended: separate ECS C2 and separate ECS server.** It removes Windows
   host contention while retaining two independent logical Flower clients and a
   physical Pi edge client. It is appropriate for algorithm multi-seed evidence,
   but its deployment wording is `Pi C1 + ECS-hosted C2`.
2. **Co-locate C2 on the existing server ECS.** Rejected: CPU, memory and timing
   become confounded with aggregation and DA.
3. **Retain Windows PC C2.** Retained only for prior diagnostic evidence; it is
   blocked by non-repeatable multi-minute C2 stalls and controller SSH failures.

## Deployment design

The new host is Ubuntu 22.04 with 2 vCPU, 2 GiB RAM, 35 GiB free disk and no
Python ML dependencies at preflight. It will receive:

- an isolated C2-only virtual environment;
- CPU-compatible PyTorch plus the frozen confirmation dependencies
  (`flwr==1.23.0`, `protobuf==4.25.8`, `psutil==7.0.0`, NumPy and scikit-learn);
- a 2--4 GiB swapfile, recorded as execution-environment metadata;
- the content-addressed immutable source archive and only C2 source data;
- generated C2 command/context/manifest files.

Dependency versions, Python version, CPU/RAM/disk and dataset/archive hashes
will be recorded before any training starts. Installation must not alter the
frozen archive or datasets.

## Controller and evidence design

The controller will gain a distinct `remote C2` host placement option rather
than relabeling the PC. It will separately launch, own, sample and recover:

- ECS server / server DA;
- Pi C1;
- ECS-hosted C2.

The topology manifest binds C2 host identity, IP-independent host label,
archive and dataset hashes, runtime dependency versions, and resource sampler
metadata. It is separate from the algorithm manifest.  Crucially, it must bind
the complete `run_id -> algorithm_config_sha256` mapping for all ten frozen
runs, rather than a single configuration hash: B2/B5 and each seed have
distinct frozen configuration hashes. Result labels must use `ecs-c2` (never
`pc-c2`) and record whether the Pi uses the PC hotspot.

The controller machine retains the existing loopback-only server tunnel and
creates a separate loopback reverse tunnel to both remote clients. Thus C1 and
ECS-C2 each connect to `127.0.0.1:18080` on their own host; the Flower port is
not newly exposed on either ECS public interface. C2 receives only its own
immutable `client_2` dataset subset and a subset manifest. The full multi-client
dataset manifest remains an immutable protocol input, but cannot be used as the
remote-C2 file-presence check.

## Gates and stopping rules

1. SSH, archive, dataset-count/hash, Python-import, free-space, RAM/swap, and
   no-active-process preflight must all pass.
2. Run two-round B2 and B5 ECS-C2 + Pi-C1 smoke attempts. Verify complete
   application-message, event, resource and process-ownership evidence.
3. Check numerical results against the applicable observer OFF/ON requirement
   and ensure no schema/checkpoint/data mismatch. A failed smoke is preserved
   and blocks the formal queue.
4. Only after both smokes pass, freeze an execution-topology manifest and run
   the sequential B2/B5 x seeds 42--46 queue.

## Claim boundary

Multi-seed results under this topology test algorithm stability with two
separate logical Flower clients. They do not demonstrate a PC edge device.
System timing/resource tables must explicitly report the topology and must not
pool results with the prior `ECS + Pi + Windows PC` pilot. The physical Pi still
supports an edge-device observation; the C2 ECS represents a server-hosted
client execution environment.
