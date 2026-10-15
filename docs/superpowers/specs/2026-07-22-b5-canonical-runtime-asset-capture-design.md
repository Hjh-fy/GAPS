# B5 Canonical Runtime Asset Capture Design

## Objective

Produce a B5 C1/C2-to-C5 deployment bundle that can reproduce the B5 canonical
R4 and HC90 stream exactly, before measuring real PC and Raspberry Pi inference
performance.

## Confirmed Cause

The historic B5 canonical regression suite saved prediction streams but did not
persist the exact in-memory fitted R4 and H23 objects used to generate them.
Re-fitting later on ECS recovers the same calibration protocol and model
selection, but changes floating-point coefficients slightly.  The observed
first-row R4 delta is `1.929072750499472e-4 ppm`, above the required
`1e-6 ppm` parity tolerance.  This is an asset-provenance problem, not a model,
dataset, classifier, or feature-schema discrepancy: the 104 frozen R4 window
features were verified identical.

## Alternatives Considered

1. Relax the ppm tolerance or accept the re-fit asset. Rejected: it would hide
   a real provenance difference and invalidate the exact runtime parity claim.
2. Put the historical 1360-row prediction stream into the bundle. Rejected:
   this would be a test-row lookup, not an executable deployment runtime.
3. Export the fitted deployment objects inside the canonical regression process.
   Selected: it preserves the existing protocol, does not use C5 test labels for
   fitting, and removes post-hoc re-fit drift.

## Design

The source-augmented R4 evaluation receives one optional deployment-export
destination and the classifier SHA-256 binding.  After it has fitted source
heads and C5 calibration-only Ridge heads, it serializes those exact objects
before emitting the existing prediction streams.  The regular regression
outputs remain unchanged when the optional export arguments are absent.

The H23 evaluation similarly emits its exact fitted reference models from the
same process.  The existing QC evaluation then consumes only the new canonical
R4/H23 streams and writes its existing HC90 policies.  A bundle builder binds
these assets, hashes them, and creates an external 1360-row reference.  The
runtime must pass class, selected-profile, QC-decision, and final-ppm parity
with `max_abs_delta <= 1e-6` before any device benchmark begins.

## Constraints

- B5 is the only deployment mainline; B2 stays separate system evidence.
- The chain is classifier -> H1/H2/H3 -> C5-calibrated R4 -> H23/QC HC90.
- C3/C4, R3aK16, H8+C4 rescue, P4, altered data splits, altered model/loss, and
  C5 test fitting are forbidden.
- Existing evidence is immutable; new outputs use a fresh, dated directory.
- A parity failure is terminal for the benchmark stage.

## Verification and Evidence

1. Unit tests show optional export serializes the same object used by the
   current-process prediction function, and no optional export leaves existing
   output contracts unchanged.
2. On the original ECS, rerun only the B5 regression/QC evaluation against the
   frozen B5 classifier and unchanged source/C5 calibration data; this is not a
   Flower training run and does not reselect an algorithm.
3. Build the new bundle and run the 1360-row B5 C5 parity gate.
4. Only on `status=equivalent`, run PC and Pi batch=1/32 benchmarks with 30
   warm-up and at least 100 measurements, collecting latency, RSS, CPU, model
   size, and platform metadata.

## Out of Scope

No new model structure, algorithm comparison, low-calibration result,
availability test, long-run stability test, or observability schema expansion
is included in this change.
