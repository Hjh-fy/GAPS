# Deployment package example

This directory is a schema example only. Add these generated artifacts from the
GAPS training/export pipeline before loading it in the UI:

- `model.ts`: TorchScript model returning either `(logits, ppm)` or a dict with
  `logits`, `ppm`, and optional `risk_score`.
- `norm_stats.npz`: required only when `normalization.enabled=true`; mean/std
  must be broadcastable to the package-defined `[window_size, channels]`.
- `manifest.json`: channel mapping, preprocessing, window and QC settings.

Do not copy a raw `.pth` checkpoint directly to the Raspberry Pi UI. Export a
frozen deployment model so the UI does not depend on the full training codebase.

Schema v2 defines timing in seconds instead of silently assuming a sampling
rate. `Baseline Start`, `Exposure Start`, and `Recovery Start` drive the runtime
state machine when `phase_control.mode=event_driven`. The example uses
`relative_adc` only as a valid engineering template. A formal
`relative_conductance` package must provide one verified load resistance per
sensor channel.
