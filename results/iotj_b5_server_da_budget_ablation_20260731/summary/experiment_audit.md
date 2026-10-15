# Experiment Audit

## Audit scope and intended claim

Audit the post-freeze, single-seed sensitivity of the frozen B5 classification
training pipeline to reducing server domain-adaptation steps per round from 100
to 80, 50 and 30 while fixing all other registered fields.

## Compared experiments

| Experiment ID | Split | Model | DA steps/round | Local epochs | QC | Seed | Provenance |
|---|---|---|---:|---:|---|---:|---|
| IOTJ-B5-LE1-DA100-S42 | C5 frozen 320/1360 protocol | B5 | 100 | 1 | off | 42 | existing LE1 reference |
| IOTJ-B5-LE1-DA80-S42 | same | B5 | 80 | 1 | off | 42 | canonical / validator accepted |
| IOTJ-B5-LE1-DA50-S42 | same | B5 | 50 | 1 | off | 42 | canonical / validator accepted |
| IOTJ-B5-LE1-DA30-S42 | same | B5 | 30 | 1 | off | 42 | validator rejected |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| F01 | informational | DA80 completeness | 25 rounds, 2000 DA steps, 50 fit instructions/results, 1360 predictions | Formal single-seed comparison available | none | closed |
| F02 | informational | DA50 launch history | First sampler-registration attempt failed before Flower training; preserved, then identical locked configuration reran | Does not alter the canonical trained result | retain failure logs | closed |
| F03 | blocking for formal DA30 evidence | Observability coverage | C2 coverage 0.948214 < locked 0.95 | DA30 cannot be approved as canonical evidence | keep blocked or perform a separately authorized clean rerun | open |
| F04 | informational | DA30 training integrity | 25 rounds, 50 fit instructions/results, 750 DA steps, strict checkpoint evaluation, 1360 unique row keys | Supports technical inspection only | label non-canonical | closed |
| F05 | major | Seed coverage | one seed per compute budget | No stability/significance claim | retain single-seed boundary | open |
| F06 | informational | Frozen assets | Runtime contract, bundle manifest, HC95 and HC90 SHA256 unchanged | No runtime/QC drift | none | closed |
| F07 | minor | Postflight self-identification | DA80 and DA50 postflight JSON payloads do not include the DA step count and are byte-identical | Standalone copies cannot identify their level without path/summary context | bind by protocol, directory and training-summary SHA | mitigated |

## Leakage assessment

The C5 test labels were not used for training, early stopping, checkpoint
selection, step-budget selection or rerunning a poor numerical result. This is a
post-freeze sensitivity analysis on the historical held-out-window test
universe, not a new prospective test.

## Baseline, completeness, and reproducibility assessment

The held constants, algorithm hashes, data roles, topology and output paths are
registered in `protocol_lock.json`. DA80 and DA50 are complete and canonical.
DA30 is training-complete but formally blocked solely by its observability
coverage gate. The audit does not lower the threshold or relabel that attempt.

## Verdict: blocked

The four-row table is valid as a transparently qualified engineering analysis,
but the DA30 row is blocked from approved experimental evidence. DA80 and DA50
retain their canonical identities independently.

## Unknowns and handoff

Multi-seed behavior of the reduced DA budgets is unknown. No additional run is
authorized or required by this audit.
