# Final figure captions

**Fig. 5. Concentration estimation with the frozen A4 router.** (a) End-to-end ($S_{ALL}$) and route-correct ($S_{CC}$) RMSE for 83-D sensor statistics, 84-D sensor statistics plus federated H1, and 86-D statistics plus H1/H2/H3. (b) End-to-end per-gas RMSE. All results use the fixed C5 round-25 A4 endpoint and calibration-only Ridge selection (seed 42).

**Fig. 6. Source-prior and calibration-budget evidence.** (a) Fixed-A4 seed-42 NRMSE ablation for the three target-regression inputs. (b) Frozen group-aware calibration-budget study (five replicates; mean and sample SD). The panels use distinct protocols and are not pooled as a single-factor ablation.

**Fig. 7. Label-free quality-control trade-off.** (a) Test NRMSE versus retained coverage using calibration-only p95-normalized risk thresholds, with 1,000 matched random selections (mean and sample SD). HC90/HC95 annotations report actual test coverage obtained from thresholds targeting 90%/95% calibration coverage. (b) Fractions of misroutes, errors at least 40 ppm, and top-decile errors captured among rejected outputs.

**Fig. 8. Communication and deployment validation.** (a) Measured 25-round Flower application payload and theoretical one-shot federated-H1 serialized exchange (different evidence types). (b,c) Raspberry Pi 5 latency, throughput, and peak RSS for the audited runtime variants. (d) Completed three-machine Flower A4 execution and fixed-endpoint audit; no hardware photograph or transport-byte measurement is implied.
