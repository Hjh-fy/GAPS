# STRICT_GROUPED_NON_OVERLAP robustness protocol

Status before training: `FROZEN_SPLIT_PASS`.

This is a supplementary sensitivity protocol; it does not replace canonical-v1. HZ5_MEAN_W10S, the 5 Hz samples, physical 10 s windows, 50x8 features, C1/C2 source arrays, A4, 25 rounds, local_epochs=1, Adam 5e-4, R84_FED_H1, seed42, and fixed-endpoint evaluation remain unchanged. Only target calibration/test membership changes.

Within each target and class×concentration cell, complete raw files—not overlapping windows—are assigned to one role. C3/C4 cycle the calibration repeat deterministically by ordered class×concentration cell so calibration and test both retain all phases available on that device. C5 uses repeat2 for calibration and repeat1 for test; therefore the C5 methane 225 ppm repeat1 anomaly remains in the sealed test. No split seed or target-test metric was used.

Calibration-only raw files are deterministically midpoint-subsampled to retain the frozen calibration budgets: C3=678, C4=320, C5=320. Unselected windows from those calibration files are excluded rather than reassigned to test. All other raw files enter test. Consequently C4/C5 test N falls to 840 because their two-repeat design cannot simultaneously provide raw-file disjointness and an 80% test ratio. C3 test N is 2515.

Pre-execution audit:

- exact-window overlap: 0;
- raw-file overlap: 0;
- raw-time overlap: 0 s;
- four classes and all 40 class×concentration cells: present in calibration and test for every target;
- all phases available for a target: present in both roles;
- C1/C2 assets: byte-identical to canonical-v1;
- strict dataset aggregate SHA256: `881de29938460ad1a7564aca1f01a2b3f41cdc4820284397a05a0b3b218816c4`;
- parent canonical-v1 aggregate SHA256: `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.

The test set remains sealed until all three fixed round-25 A4 endpoints and R84 calibration locks are complete. No result-triggered tuning is permitted.
