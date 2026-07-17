# IoT-J Preliminary Paper Metrics Claim Boundary

Generated: 2026-07-17 (Asia/Shanghai)

This package closes the currently available classification -> regression -> QC -> system-pilot evidence chain. It does not reopen Spec A or authorize formal confirmation runs.

## Main C1/C2 -> C5 classification

- Usable: Seed-42 25-round descriptive screening performance for B1-B5.
- Deferred/forbidden: Do not include feaa75b seed-42 in future confirmation mean/std; no variance claim.

## Three cross directions

- Usable: Seed-42 appendix/generalization evidence for F1, R1 and R2.
- Deferred/forbidden: Do not claim direction-stable multi-seed significance.

## Formal C5 regression

- Usable: Coverage-1 R0-R7 actual-route results on all 1360 C5 test windows.
- Deferred/forbidden: R7 is an offline per-row oracle, not deployable.

## Operational QC

- Usable: FULL/HC95/HC90 actual-route yield, nonreject coverage and errors.
- Deferred/forbidden: Oracle QC columns use test truth for forced routing and are diagnostic only.

## B2 real-topology system pilot

- Usable: Measured logical/application bytes, timing and training-side resource use for a real ECS+Pi+PC 2-round run.
- Deferred/forbidden: No 25-round total, transport-layer byte count, tail latency, or long-run stability claim.

## B2 Observer Gate

- Usable: OFF/ON exact numerical equivalence for the B2 two-round formal topology smoke (max_abs_delta=0).
- Deferred/forbidden: Does not validate B5 or the full 10-run confirmation queue.

## B5 Observer Gate

- Usable: A preserved round-2 parity failure exists and blocks formal confirmation.
- Deferred/forbidden: Do not generate a freeze record or label any new B5 run as formal confirmation.

## Formal multi-seed confirmation

- Usable: Not available yet.
- Deferred/forbidden: 10x25 queue remains unstarted; do not report confirmation mean/std.
