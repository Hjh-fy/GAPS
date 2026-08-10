# Gate 2 Source-phase Safety Audit

## Observed definition

The canonical manifests store `phase_label` alongside each physical window's `window_start_s`, `window_end_s`, and acquisition filename. The model training loader reads the phase array belonging to the source window. Phase is therefore derived from source acquisition-time protocol metadata, not from C5 data, target predictions, or target-test outcomes.

## Gate-2 information boundary

- Allowed: C1/C2 source X, class, response phase, and per-cell sample counts.
- Forbidden and absent from commands: C5 X, class, phase, concentration, calibration path, and test path.
- Round 1: source CE and upload of C1/C2 class×phase prototypes.
- Rounds 2-25: source CE plus squared distance between normalized local feature and normalized server class×phase prototype, weight 0.05.
- Server aggregation: local cell means are weighted by valid cell sample counts and then maintained with the registered prototype EMA alpha 0.8.

## Verdict

`PASS_SOURCE_PHASE_OBSERVABLE`. Class-only fallback is not required for G2. This verdict does not automatically authorize target phase use in G3, where real online observability is audited separately.
