# P2 split RNG audit

- Exact splitter family: `np.random.default_rng(seed)` created once, then consumed sequentially while iterating `unit_1` ... `unit_5`.
- The legacy `split_dataset.py` independently shows the same structural risk with one global `np.random.seed(seed)` and sequential bucket/final-array shuffles. The executable replay below uses the actual time-aware generator behind NEW, because OLD and NEW do not share one processed Unit5 representation.
- Case A C5-entry RNG-state SHA256: `bf7e8b1f33eb8961c06850323988288c383b051301f1fe78093a0f6a12c15099`.
- Case B C5-entry RNG-state SHA256: `632d57e0f8552439e7d71f31e7556f55a23189aacda02b07c0d309f1a2c13424`.
- States equal: **False**.
- C5 calibration membership overlap: 55/320; symmetric difference: 530 windows.
- `same seed=42` does **not** guarantee the same C5 split when earlier clients have different source/target roles, because their ratio-specific final split-array shuffles consume different amounts of RNG state.

`RNG_CLIENT_ORDER_COUPLING = TRUE`

This audit only reproduces and records behavior; it does not modify the splitter.
