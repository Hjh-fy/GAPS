# Preprocessing design protocol

## Frozen candidate design

All candidates use true timestamps; stable time sort; duplicate-timestamp mean merge; raw-observation conductance G0 over 20≤t<50 s; physical-time bins; non-empty-bin aggregation; short-gap-only (one-bin) interpolation; long-gap invalid/quality metadata; physical crop 60–170 s; and physical-duration windows. Baseline statistic is **mean** for all 16 first-stage candidates. The 16 registered combinations are 1/2/5/10 Hz × mean/median bin aggregate × 10/20 s windows, with stride equal to one-half duration.

## Leakage gate

Screening reads C1/C2 source and C3/C4/C5 calibration rows only. Test metadata is read only to construct the sealed physical-role manifest; no test feature, error, label, alpha, or candidate ranking is used before the two frozen manifests are written. After freeze, only the TOP-2 receive one test diagnostic.
