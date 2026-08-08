# Canonical v1 Experiment Plan

## Registered hypothesis

`CAN-V1-H1`: A from-scratch GAPS pipeline using frozen `HZ5_MEAN_W10S` can be executed without target-test selection or historical checkpoint reuse and will yield complete classification, R84, QC, quality-stratified, and engineering evidence. The hypothesis fails operationally if any dataset/preflight/provenance gate fails; numerical weakness is reported and does not reopen preprocessing search.

## Roles and protocol

- Source: C1/C2, exact frozen source train/calibration/test physical identities.
- Target: C3/C4/C5, 0% train, frozen 20% calibration and 80% sealed-test physical identities.
- Classification: final A4 GAPS router (`ce_stats`, no client semantic/replay, no selective aggregation, frozen A4 server DA), TCN+attention, Adam 5e-4, 25 rounds, local epochs 1, batch size 32, seed 42. Only preprocessing and the explicit 50-point input contract differ from the prior frozen training protocol.
- Adaptation: current frozen GAPS server adaptation, calibration inputs only, fixed 100-step endpoint.
- Regression: R84_FED_H1, frozen alpha grid and calibration-internal split.
- QC: frozen final equal-mean policy; no threshold search.

## Acceptance gates

Dataset SHA reproducibility, identity uniqueness, row alignment, finite features, role correctness, split disjointness, class×concentration coverage, repeat-1 retention, and all checkpoint reuse flags must pass before training. Target test cannot enter preprocessing, fitting, adaptation, checkpoint, alpha, or QC selection.

## Evidence

The approved evidence destination is `results/iotj_canonical_v1/`. Single-seed deterministic results are formal system evidence, not repeated-seed stability evidence. Historical Legacy/current-interpolation rows remain preprocessing diagnostics only.
