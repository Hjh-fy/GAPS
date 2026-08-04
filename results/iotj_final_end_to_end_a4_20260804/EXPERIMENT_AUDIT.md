# Experiment audit

Status: **PASS WITH DECLARED BOUNDARIES**

- Existing P0/classification assets were read only; no source Flower or classifier was retrained.
- A4 equality is based on the ordered state-content fingerprint. Whole-file SHA-256 is provenance only.
- Regression alpha selection used C5 calibration only (60 fit/20 validation per gas), was persisted, and was read back before C5 test loading.
- C5 test was used only for the fixed final classification/regression/QC evaluation; it did not select a checkpoint, Ridge alpha, or QC threshold.
- QC component scales and thresholds use label-free calibration fields. The excluded first QC attempt is explicitly non-formal.
- C3/C4 same-protocol A4 evidence is unavailable and remains blocked.
- Fig. 6 visibly separates the seed-42 A4 prior ablation from the historical five-replicate group-aware budget study.
- Fig. 8 distinguishes measured application payload from theoretical serialization and does not claim transport-byte measurement or a hardware photograph.
- Final replay is seed 42 only; QC random reference uses seed 20260804 for 1,000 matched selections.
