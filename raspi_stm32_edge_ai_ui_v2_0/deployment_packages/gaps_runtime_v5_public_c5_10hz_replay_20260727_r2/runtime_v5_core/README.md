# GAPS Runtime-v5 Core Portable Release

Status: `CANDIDATE_FOR_CLEAN_CHECKOUT_SMOKE`

This archive contains only the final B5 classifier, sufficient-statistics
Federated H1, C5 105D target Ridge, calibration lineage lock, a strict
relative-path portable binding, provenance records, and synthetic inputs.

It contains no C5 formal test windows, labels, HC95/HC90 records, offline
formal predictions, Runtime-v4 assets, or Runtime-v5 QC policy.

Verify assets:

```powershell
python -m gaps_deploy.runtime_v5_cli --contract portable_binding.json --verify-only
```

Describe the binding:

```powershell
python -m gaps_deploy.runtime_v5_cli --contract portable_binding.json --describe-contract
```

Run the synthetic example:

```powershell
python -m gaps_deploy.runtime_v5_cli `
  --contract portable_binding.json `
  --input synthetic/input.npy `
  --metadata synthetic/metadata.json `
  --phase-file synthetic/phase.npy `
  --output synthetic/output.json `
  --device cpu
```

The CLI refuses missing or mismatched assets, invalid shapes, NaN/Inf, malformed
metadata/phases, and an existing output path.
