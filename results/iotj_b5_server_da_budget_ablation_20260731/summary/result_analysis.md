# Result Analysis

## Input contract and provenance

- Experiment IDs: `IOTJ-B5-LE1-DA100/80/50/30-S42`
- Comparison identity: B5, seed 42, C1/C2→C5, 25 rounds, one local
  epoch per round; only `server_adaptation.steps` changes.
- Sample scope: frozen C5 test, 1360 held-out windows.
- DA100 is the existing LE1 reference. DA80 and DA50 passed the canonical
  validator and postflight. DA30 completed training and evaluation but did not
  pass the observability validator.
- Values copied from the four classification metric JSON files are `reported`.
  Deltas, disagreement counts and wall-time reductions in the summary are
  `recomputed`.

## Descriptive comparison

| Configuration | Accuracy | Δ Accuracy vs DA100 | Macro-F1 | Δ Macro-F1 | NLL | ECE | Errors | Wall time | Wall reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DA100 | 0.989706 | 0 | 0.989712 | 0 | 0.076054 | 0.008967 | 14 | 1.338 h | 0% |
| DA80 | 0.985294 | −0.004412 | 0.985303 | −0.004409 | 0.105687 | 0.012612 | 20 | 1.127 h | 15.74% |
| DA50 | 0.975000 | −0.014706 | 0.975087 | −0.014625 | 0.145886 | 0.019382 | 34 | 0.791 h | 40.84% |
| DA30† | 0.986029 | −0.003676 | 0.986053 | −0.003659 | 0.072129 | 0.007259 | 19 | 0.537 h | 59.83% |

† DA30 is a non-canonical technical result. Its C2 resource-sampling coverage
was 0.948214 against the locked 0.95 minimum.

## Comparisons and practical interpretation

- DA80 is the only newly run configuration that both passed the formal
  observability/postflight gates and stayed within the pre-set 0.5 percentage
  point retention tolerance for Accuracy and Macro-F1. Its calibration metrics
  nevertheless worsened: NLL increased by 0.029632 and ECE by 0.003645.
- DA50 failed both retention checks and produced 20 more errors than DA100. It
  should not be treated as an equivalent compute reduction.
- DA30 numerically stayed inside the engineering tolerance and reduced wall time
  by 59.83%, but the locked validator did not accept the run. This numerical
  result cannot override the audit status.
- The trajectory is non-monotonic: DA50 is worse than DA30. With only one seed,
  the experiment does not establish a smooth dose-response relation between DA
  steps and accuracy.
- Relative to DA100, prediction disagreement counts were 13 rows for DA80, 26
  for DA50 and 17 for DA30.

## Statistical boundary

Each configuration has one training seed. The 1360 windows are not independent
training replicates and cannot replace multiple seeds. No significance or
statistical non-inferiority claim is made. The 0.5 percentage point threshold is
an engineering performance-retention tolerance only.

## Proposed use

- Keep DA100 as the frozen formal configuration.
- DA80 may be retained as a compute-saving candidate for a future, explicitly
  authorized multi-seed confirmation; it is not automatically promoted.
- DA30 may only motivate a clean same-protocol rerun if canonical observability
  evidence is later required. The 0.95 validator threshold must not be relaxed
  retroactively.
