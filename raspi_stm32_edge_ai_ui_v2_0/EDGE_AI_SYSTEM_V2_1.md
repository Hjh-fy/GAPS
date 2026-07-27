# GAPS EdgeSense v2.1 system contract

## Positioning

This directory is no longer treated as only a waveform UI. It is the edge
orchestration layer between a physical STM32 acquisition stream and a frozen,
auditable inference package:

```text
STM32 frame stream
  -> protocol and value validation
  -> experiment phase state
  -> device-specific feature transform
  -> time-defined baseline and window
  -> frozen model backend
  -> calibration and fail-closed QC
  -> linked raw/event/prediction audit files
```

The implementation supports two explicit backends:

- `torchscript` for schema-v1/v2 packages;
- `gaps_runtime_v5` for the schema-v3 portable
  `Final B5 -> Federated H1 -> C5 105D target Ridge -> Runtime v5 core`.

Runtime v5 is loaded from its verified portable binding and matching hash-locked
Python code bundle. It is not silently approximated by the TorchScript path.
The Runtime-v5 core keeps QC disabled and never emits an automatic ppm output.

## Two datasets, two model profiles

The public dataset and future laboratory dataset must not share sensor
semantics, normalization statistics or model weights by default.

### Public benchmark profile

```text
source data: public gas-drift dataset
original timestamps: about 100 Hz
model input rate: 10 Hz after time-aware resampling
channels: 8 public resistance-response channels
role: algorithm benchmark and Raspberry Pi replay/performance evidence
```

The model is a **10 Hz input model**, even though the source files were sampled
at about 100 Hz.

### Laboratory physical profile

```text
source data: laboratory STM32 acquisition board
runtime rate: expected 1 Hz
channels: laboratory sensor/ADC mapping
role: real acquisition, calibration and online edge inference
```

Window duration and baseline duration are selected in seconds. Candidate
laboratory windows should be compared at 30 s, 60 s and 100 s rather than
copying the public `100 points` setting without regard to time.

## Schema v2/v3 invariants

A loadable package declares:

- `dataset_profile` and `device_profile`;
- the exact ordered `sensor_fields`;
- runtime input and target sampling rates;
- unstable, baseline, window and stride durations in seconds;
- the physical feature transform and all required load resistances;
- whether normalization is enabled;
- automatic or event-driven experiment phase control;
- allowed inference phases;
- model class names and QC workpoint;
- model SHA-256 and a runtime-derived package fingerprint.

The runtime rejects:

- duplicate or missing sensor fields;
- invalid timing/rate configuration;
- non-finite normalization arrays;
- missing or invalid load resistances;
- package path traversal;
- a model whose SHA-256 does not match the manifest;
- non-finite or shape-inconsistent model outputs.

Schema v1 remains readable for existing prototypes, with its original
sample-count semantics and normalization default.

Schema v3 adds:

- `model_backend=gaps_runtime_v5`;
- a relative portable binding with frozen asset identities;
- a runtime code manifest that locks every packaged Python file;
- exactly 8 channels and a 100-sample Runtime-v5 window;
- a fixed target drift phase contract;
- fail-closed `disabled_pending_dependency_audit` QC semantics;
- a replay-only `precomputed` input mode that rejects unmarked raw serial frames.

The validated public C5 replay package is intentionally not a live STM32 model.
See `RUNTIME_V5_INTEGRATION_20260727.zh.md`.

## Event-driven state machine

For `phase_control.mode=event_driven`:

```text
unmarked
  -> Baseline Start
  -> sensor stabilization
  -> clean baseline collection
  -> baseline ready
  -> Exposure Start
  -> window collection/inference
  -> Recovery Start
  -> recovery window collection/inference
```

`Baseline Start` resets baseline and window. `Exposure Start` and
`Recovery Start` keep the accepted baseline but create a new model-window
boundary.

Inference is not permitted before a valid baseline or outside the package's
allowed phases.

## Stream safety

The AI window is reset when:

- the serial input disconnects;
- an implausible frame is received;
- timestamps are non-monotonic;
- the inter-frame gap exceeds `max_gap_s`;
- the experiment phase changes;
- the operator explicitly resets the window or baseline.

Raw acquisition retains suspicious frames for diagnosis, while edge inference
fails closed and does not consume them.

For production firmware, the 43-byte protocol should still be upgraded with a
frame sequence number, STM32 monotonic timestamp and CRC. Host arrival time and
header/tail framing alone cannot prove that no full frame or payload bit was
lost.

## Audit linkage

Every prediction carries:

- package fingerprint and model/backend identity;
- dataset and device profile;
- normalization state and experiment phase;
- inference ID;
- start/end stream-frame index and timestamp;
- connection ID;
- whether every frame in the model window belonged to one raw recording
  session;
- the recording session identifier.

An AI row is written to an experiment only when the entire input window was
covered by the same `raw.csv` recording session. This prevents delayed
background inference from being attached to the wrong experiment.

## Backend roadmap

### Implemented

- serial/HC-04 and USB input;
- experiment/event/raw logging;
- schema-v1/v2 TorchScript packages;
- schema-v3 verified Runtime-v5 portable packages;
- streaming baseline/window preprocessing;
- event-driven experiment phases;
- simple packaged calibration and two-threshold QC;
- package integrity and window-level audit;
- public C5 10 Hz precomputed stream replay on Windows and Raspberry Pi;
- Raspberry Pi Wayland worker-to-result UI validation with Runtime-v5.

### Remaining production edge-system work

- freeze the laboratory 1 Hz sensor/ADC/physical-unit contract;
- collect and split laboratory data without crossing experiment-file groups;
- train and validate a separate laboratory model;
- implement only a QC policy that has been formally promoted;
- validate the physical HC-04/USB link, firmware sequence/CRC and long-run
  acquisition behavior.

## Acceptance gates

Before claiming a complete edge AI system:

1. Replay an approved parity set through PC and Raspberry Pi runtimes; do not
   reopen a locked formal test merely for software diagnostics.
2. Require identical class/QC decisions and numerically bounded ppm/logit error.
3. Measure P50/P90/P95 latency, RSS and model load time.
4. Run hardware fault injection for disconnect, frame corruption, long gaps,
   low disk and application shutdown.
5. Run at least a 6-hour acquisition test with frame sequence/CRC evidence.
6. Freeze package, firmware and preprocessing hashes in experiment metadata.
