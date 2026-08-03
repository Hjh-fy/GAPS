# P0-I Frozen Protocol

- Dataset: `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`
- Source: C1 and C2; target: C5; calibration: 320; sealed test: 1,360.
- Seed: 42 only. Classification TCN, 25 rounds, LE1, batch 32, Adam lr 5e-4, CE-only, sample-weighted FedAvg, FedProx 0.
- I2 source checkpoint: P0A round 25, required SHA-256 `4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c`.
- Shared UDA: source CE + 0.5 unconditional CORAL + 0.5 global MMD2 + 0.5 unconditional Wasserstein adversarial loss; Adam model lr 5e-4.
- Target adaptation API is C5 calibration x only. Target class/phase labels, target CE, conditional CORAL/MMD, stage MMD, prototypes, semantic matching, pseudo-labels, and label-conditioned sampling are unavailable.
- I2 runs continuously for 2,500 steps and saves steps 0/100/250/500/1000/1500/2000/2500 without opening C5 test. Formal endpoint is step 2,500.
- I3 reruns Flower from the same seed initialization. Every round is C1/C2 LE1 CE-only, FedAvg PRE checkpoint, 100-step x-only UDA, POST checkpoint, then POST broadcast as the next global state. Formal endpoint is round 25 POST.
- During training, C5 test metrics are forbidden. A separate evaluator may open the sealed test only after both training procedures complete and must evaluate every requested checkpoint without selection.
- I3 lineage is proved using deterministic state-content fingerprints over ordered key, dtype, shape, and tensor bytes reported by both clients. Rounds 2–25 fail closed unless both received fingerprints equal the preceding POST fingerprint.
- Existing P0 and P0-U files are read-only.
