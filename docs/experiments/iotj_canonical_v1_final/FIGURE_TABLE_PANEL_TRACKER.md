# Figure/table panel tracker

- **Fig.1 all - LEGACY_ONLY**: cloud-edge-sensor architecture; source `N/A`; script `MISSING`. Must be checked against canonical 5 Hz, 50x8, LE1 protocol.
- **Fig.2 all - NEEDS_REGEN**: device-domain shift; source `canonical sensor-shift evidence not bundled`; script `MISSING`. Regenerate from canonical-v1 only.
- **Fig.3 all - NEEDS_REGEN**: cross-target classification; source `03_classification_final.csv`; script `MISSING`. A4, LE1, round25, seed42.
- **Fig.4 all - LEGACY_ONLY**: C5 classification and A0-A6; source `canonical A4 exists; canonical ablation table absent`; script `MISSING`. Do not relabel historical ablation as canonical.
- **Fig.5 all - NEEDS_REGEN**: concentration/per-gas; source `04_regression_final.csv + regression/cross_target_r84_per_gas.csv`; script `MISSING`. Canonical A4+R84_FED_H1.
- **Fig.6 all - NEEDS_REGEN**: 83D/84D plus calibration budget; source `10_fedridge_83d_84d.csv`; script `MISSING`. Budget panel is not yet supported by canonical evidence.
- **Fig.7 all - NEEDS_REGEN**: QC curve/random/HC90/HC95; source `05_qc_final.csv + qc_random_reference.csv`; script `MISSING`. Frozen equal-mean QC; same-budget random seed 20260804.
- **Fig.8 all - NEEDS_REGEN**: communication/Pi5/physical validation; source `07_pi5_benchmark.csv + 06_quality_robustness.csv`; script `MISSING`. Pi5 formal package f3d1577; communication must not infer 50% reduction.
- **Main table classification/regression/QC - READY**: formal numerical results; source `FINAL_RESULT_MASTER_TABLE.csv`; script `N/A`. Canonical values only.
