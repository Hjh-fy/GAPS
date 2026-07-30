# REC-A4 classification-only replay candidate

This package validates the laboratory three-gas A4 model inside the UI v2.2
runtime.  It accepts only explicitly marked, precomputed relative-resistance
features and intentionally rejects raw STM32 serial frames.

There is no concentration output.  `ppm_*` values remain null and QC remains
Unavailable because no target-validated classification QC policy exists.

The reported 359/360 (99.72%) accuracy covers 360/420 (85.71%) stable windows.
Do not present it as full-time accuracy.  Live serial inference remains blocked
until the six-channel ADC-to-resistance and channel-mapping contract is frozen.
