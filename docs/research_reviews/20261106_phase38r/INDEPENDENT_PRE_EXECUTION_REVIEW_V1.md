# Phase3.8R independent pre-execution review V1

Reviewed at 2026-11-06 23:12 +08:00. Reviewer was a separate agent restricted to code and registration inspection. No models, checkpoints, signal arrays, Day22 data, historical Test/evaluation outputs, or old Phase3.8 run1 records were loaded. No inference or tests were executed by this reviewer. Only this new report was written.

## Disposition

No confirmed critical or important launch blocker was found in the reviewed V1. The implementer will freeze a V2 before launch to broaden hook controls; this report does not certify that unreviewed revision or any execution result. Historical Phase3.8 strict acceptance remains BLOCKED.

Reviewed V1 registration SHA256: `7fae15ab09d512e96f97b4db6ab5df94e100bf221e7bf901549b17386e9807bf`.

## Evidence checked

- Read the new EXPERIMENT_PLAN.md, diagnose.py, core.py, p38r_common.py, REGISTRATION_FINAL.json, and test_core.py.
- Read only explicitly permitted frozen source: Phase3.7 assets.py, Phase3.8 inference.py, Phase3.5 p35_common.py, Phase3.3R runtime.py, and the archived model.py/gaps_flower/task.py. These were static code reads, not execution of their loaders or functions.
- Independently rehashed all eight entries in V1 `code_sha256`: all matched. Input bytes and model bytes were deliberately not opened by this reviewer. The implementer's independent identity gate is outside the proof established by this code-only review.

## Confirmed structure

1. The new audit hook is installed before FrozenAssets is constructed. It has no HASH_ONLY bypass. Nonallowlisted NPY/NPZ and dataset/data1 paths, Phase3.5 evaluation paths, and the old Phase3.8 run1 tree are rejected before Python opens them. The original Phase3.7 guard remains installed as an additional gate.
2. FrozenAssets loads locked models and SVR assets through the original loader; the new diagnosis never calls bridge, feature extraction, regressor prediction, fit, or an optimizer. Static inspection of p35_common.load_model and runtime.load_checkpoint found no model forward in initialization. The model pre-hook counts every subsequent diagnostic forward, before its body executes, and rejects calls after 3200.
3. Expected V1 total is 2924 calls: 1824 main matrix calls, 1092 interface/adapter/no-hook/no-grad/restore calls, and eight B0 profiler calls. Each six-regime pass uses 38 calls. This is below the fixed 3200 limit.
4. b1, b16, b32, roll7, tile4, and repeat4 mappings preserve the fixed 32-window identity space. Comparisons select the same original window from the same-mode b1 reference. Repeated 128-row batches remain 32 unique windows and 16 Calibration exposures. The two duplicate layouts compare all four occurrences of each original window.
5. The actual frozen classifier is the TCN path with batch_first MultiheadAttention. Executed module tensor outputs have batch axis zero; capture concatenation and reference indexing are consistent with this architecture. Nonexecuted modules are not misrepresented as observed hook outputs. Functional operations between module boundaries remain outside direct hook localization.
6. Original/Calibration source-equivalent, diagnostic-shim, and strict-reload controls execute after Capture hooks are removed for all six regimes. Actual Phase3.7 and Phase3.8 adapters are only represented at their supported batch1 interface. New batch shims are explicitly named as new diagnostics. The P35 source string is inspected without invoking its Test-reading main function.
7. Component gates use `abs(candidate-reference) <= 1e-4 + 1e-5*abs(reference)`. Hidden-layer thresholds are explicitly descriptive. Numeric batch/backend findings cannot release the historical strict-PC failure or establish a 71D regression input contract for a 64D encoder.
8. Original backend flags are captured, restored before per-domain original-b32 checks, and also restored in a per-model finally block. State and scaler digests are compared after both domains. No deployment settings are written.

## Nonblocking observations and proposed closure

- Hook equivalence in V1 is broad for original/Calibration but has no independent full-model control for other modes and synthetic inputs. The implementer proposed adding a no-Capture b1-first-window and b32 comparison for every model/domain/mode. That adds 96 calls, for an expected V2 total of 3020 and 180 calls of remaining headroom. The frozen V2 must be re-reviewed and its SHA checked before relying on this closure.
- `core.compare` names dtype-matched `np.array_equal` as `bitwise_equal`. This treats +0.0 and -0.0 as equal. It proves exact finite numeric equality, not necessarily identical float32 bytes. Either use byte/digest equality for that field or qualify the final claim accordingly. This does not change the original tolerance gate.
- Access receipts and unchanged-state receipts are execution evidence still to be audited. Static review cannot independently establish zero actual Day22 reads, final state identity, or restored backend values.

## Remaining review

Recheck the final registered code/registration SHA after the planned pre-execution edit. After the authorized run, audit only the new Calibration/synthetic evidence, budget ledger, access receipts, hook controls, state versions, restored backend controls, and final summaries. Do not inspect old Test/evaluation or Phase3.8 run1 records.
