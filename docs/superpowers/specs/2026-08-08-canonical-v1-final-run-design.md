# Canonical v1 Final Run Design

## Scope

Build one immutable `dataset/iotj_canonical_v1/` from raw observations and run one final formal GAPS classification, target adaptation, R84 regression, QC, and deployment measurement sequence. The dataset configuration is frozen before any training.

## Frozen preprocessing

`HZ5_MEAN_W10S`: stable real-time sort; duplicate-timestamp mean merge; raw-observation conductance mean G0 from 20–50 s; 5 Hz (0.2 s) mean physical-time bins; only one-bin short-gap interpolation; no long-gap interpolation; 60–170 s crop; 10 s windows / 5 s stride / 50 points. C5 Methane 225 ppm repeat 1 remains included with quality metadata. `HZ2_MEAN_W10S` is an engineering fallback only and receives no second formal training.

## Data and split architecture

Raw files, labels, code provenance, and output hashes are recorded in the canonical dataset manifests. Physical identity is `(client_id, raw_filename, repeat_id, gas, class_id, concentration, physical_window_start_s, physical_window_end_s)`. C1/C2 use the frozen source protocol. C3/C4/C5 have 0% target train, 20% calibration, 80% sealed test. Existing frozen physical identities are reused where compatible; otherwise each client/class/concentration receives an independent deterministic RNG stream. No split may depend on client ordering and calibration/test overlap is a hard failure.

## Execution architecture

1. Build dataset and run preflight/hash reproducibility validation.
2. Train a new 25-round, local-epoch-1 GAPS classifier from scratch; all checkpoint reuse flags are false. This preserves the prior frozen training protocol so preprocessing is the main changed factor. Adapt only using target calibration; test remains sealed.
3. Run fixed R84_FED_H1 from canonical representations with frozen Ridge candidates and calibration internal split, then evaluate test once in S_ALL, S_CC, and oracle-route scopes.
4. Run the frozen QC policy and engineering measurements without tuning QC thresholds or model settings.

## Required evidence

Dataset manifests, source/target counts, checkpoint hashes, adaptation configuration and identities, classification before/after metrics and confusion data, R84 overall/client/gas/concentration/C5-repeat summaries, quality strata, QC summary, latency/memory/input-byte measurements, and a reproducibility manifest are written only under `results/iotj_canonical_v1/`.

## Safety and stopping rules

Target-test labels, features, metrics, or checkpoints cannot choose preprocessing, model hyperparameters, adaptation endpoint, Ridge alpha, or QC thresholds. Existing datasets, checkpoints, and paper results are read-only. Any raw completeness, label alignment, NaN/Inf, role, split overlap, identity uniqueness, coverage, or hash-rebuild failure blocks training. No additional preprocessing search or fallback switching is permitted after freeze.
