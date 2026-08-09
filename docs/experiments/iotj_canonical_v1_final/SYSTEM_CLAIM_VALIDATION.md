# Canonical-v1 system-claim validation

Status: `PASS_WITH_LIMITATION`.

## Deployment evidence chain

The formal Raspberry Pi 5 benchmark reports `FINAL_DEPLOYED_RUNTIME` and the archive SHA256 `52328c9cd9f8c9d9eba2f700a35f20f488070df2919fba6fa94e8a77a5dc1c31`. Its package manifest SHA256 is `7cf667af89f01de42217d11625d9b72607f238f9115a48070f681b24ed2fad44` and records the same runtime commit (`f3d1577`) as the Pi result.

| Component | Canonical provenance |
|---|---|
| Dataset | aggregate SHA256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6` |
| Preprocessing manifest | `6c33f0a1586653b2bfa5a43f43ab502c5bdaa3474c24ac03015e36ddd40c2c41` |
| A4 classifiers | C3 `e2364290...4414`; C4 `422a49f2...99c3`; C5 `3965ec86...2b93` |
| R84 artifacts | C3 `562ae2e4...67d2`; C4 `4afd3fd8...febf`; C5 `d2bac602...729` |
| Federated H1 manifest | `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc` |
| QC policy | `7da42eb54d32f6fedcc67b2e636fecf6406204490d56a569237002070a47a04b` |
| Deployment archive | `52328c9cd9f8c9d9eba2f700a35f20f488070df2919fba6fa94e8a77a5dc1c31` |

The package manifest links each packaged asset to its source hash. The Pi benchmark's archive hash therefore closes the chain from canonical preprocessing, A4, R84/H1, and QC to the measured runtime package.

## Model and Pi 5 measurements

- `state_tensor_count = 80`; this is not a scalar parameter count.
- Total/trainable scalar parameters: 22,765 / 22,765.
- FP32 parameter payload: 91,060 bytes.
- P50/P95/P99 total latency: 3.149/3.193/4.924 ms.
- Throughput: 295.93 windows/s.
- Peak RSS: 258.92 MiB.

## Communication scope

For two participating source clients, 25 rounds, and two directions per round, the model-only FP32 payload implied by the canonical parameter count is `22,765 x 4 x 2 x 2 x 25 = 9,106,000` bytes. A state-tensor payload accounting gives 14,469,200 bytes. These are analytical payload estimates, not canonical wire measurements.

An earlier run with the identical model topology measured 17,572,650 application-layer bytes (transport bytes were not collected). This is retained only as a historical reference and must not be relabeled as a canonical-v1 measurement.

The frozen federated-H1 sufficient-statistics protocol has a one-shot theoretical serialized exchange of 7,710,128 bytes. Canonical-v1 reuses the exact H1 manifest by hash, so this value documents the reused artifact protocol; it is not a new canonical wire measurement.

The evidence supports “50% temporal input-length reduction” when comparing 5 Hz/50-step input with the historical 10 Hz/100-step representation. It does **not** support “50% FL communication reduction.”

## Timing limitations

The Pi package supplies component inference timings (including R84 and QC). Existing logs do not isolate canonical R84 calibration-fit time, QC calibration time, or target-adaptation time as comparable standalone measurements; those fields remain unreported rather than inferred from whole-run wall time.
