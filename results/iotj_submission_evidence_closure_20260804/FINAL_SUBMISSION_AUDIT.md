# Final submission audit

Status: **PASS WITH EXPLICIT BOUNDARIES**

- Frozen result baseline: `ceb6c78`; no training, hyperparameter search, model change, or QC-formula change was performed.
- Runtime: `FINAL_DEPLOYED_RUNTIME` = A4 round-25 classifier + `R84_FED_H1` + final equal-mean QC.
- Pi 5 benchmark: 5,000 measured windows, batch 1, CPU single-thread, P50 3.758529 ms, P95 3.791003 ms, P99 4.006750 ms, throughput 252.120 windows/s, peak RSS 238.812 MiB; throttling `throttled=0x0` → `throttled=0x0`.
- Parameter semantics corrected without model modification: `state_tensor_count=80`, `total_parameter_count=22765`, `trainable_parameter_count=22765`, `fp32_model_bytes=91060`.
- Full 1,360-row local parity: route, R83/R84 predictions, source-prior risks, and HC90 decisions were invariant. Cross-device classifier uncertainty differed by at most 8.60095e-5 and final risk by 4.00939e-6 due to CPU floating-point execution; no formula or decision was changed.
- Fig. 1 is a schematic and therefore correctly has no CSV/checkpoint. Fig. 2–Fig. 8 panels have explicit source, hash/asset, script, and caption records in `FINAL_FIGURE_MANIFEST.csv`.
- Fig. 5–Fig. 8 numeric tables are written under `paper_ready_tables/`; manuscript source was not modified.
- Boundary: single seed 42; benchmark generalizes only to the recorded Raspberry Pi 5 hardware/software environment.
