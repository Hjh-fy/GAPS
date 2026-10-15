# Post-hoc DG/SSDA Protocol Audit

## Audit scope and intended claim

This Gate-0 audit determines whether the existing canonical source-only checkpoint can support a real new-node lifecycle study and freezes the information-access boundaries for G1-G3. It does not approve any performance claim before the new endpoints and predictions exist.

## Compared experiments

| Experiment ID | Split | Model | Checkpoint | DA | Calibration | Seeds | Provenance |
|---|---|---|---|---|---|---|---|
| CAN-V1-CMP-FEDAVG | canonical-v1 C1/C2 source-only | canonical TCN, 50x8 | round 25 / latest | none | none | 42 | locked spec, completion marker, run manifest, source archive |
| CAN-V1-MR-G1-A0T-FULL | canonical C5 320 calibration, frozen test | same checkpoint copy | pending | target CE | full labels | 42 | registered before execution |
| CAN-V1-MR-G1-A4 | canonical C5 320 calibration, frozen test | same checkpoint copy | pending | registered A4 | class and phase | 42 | registered before execution |
| CAN-V1-MR-G1-HEAD | canonical C5 320 calibration, frozen test | same checkpoint copy | pending | target CE | full labels | 42 | registered before execution |

## Source checkpoint provenance

- Producer experiment: `CAN-V1-CMP-FEDAVG`.
- Producer freeze commit: `3ba128529eeebd452106dc98a24267afd4e95573`.
- Current evidence commit before redesign: `4acc79dc10981bbeb3169665c463461d3c1da67d`.
- Whole-file SHA-256, `server_latest.pth`: `2d114a8ae23fcdea574d1e7c64e638620f60e49560da594397187bd5de1505fa`.
- Whole-file SHA-256, `server_round_025.pth`: `5a4b95f60f594135a0226d562029bcbd7a1c01b0a40452fa4c058d2beaf1818e`.
- Ordered state-content SHA-256 for both files: `cad6726ec29fb574314a5f2a45ed9800d1d90906b81cbd3ba8f8efb48a0df5d7`; all 80 tensors are exactly equal. Whole-file SHA differs only because the checkpoint containers are separate serializations.
- Model state contains 36,173 scalar parameters/buffers as serialized; actual trainable/total parameter counts will be reported from the instantiated model.
- Dataset aggregate SHA-256: `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Source archive SHA-256: `5161d657ba500d4296022c988ae8e6790406ab353de930c69a86213415869f22`.
- Training config: C1 and C2 only, FedAvg, Adam 5e-4, 25 rounds, LE=1, batch size 32, seed 42, `ce_only`, fixed round-25 endpoint.
- The locked run spec has no C3/C4/C5 client command or server target loader. It explicitly records `target_information_regime=source_only`, `target_x=false`, `target_y=false`, `target_test_selection=false`, and completion with `target_test_opened=false`.

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| G0-F01 | informational | Source identity | C1/C2 are the only client commands | Valid source-only endpoint | Reuse; do not retrain | closed |
| G0-F02 | informational | Endpoint identity | round25/latest states are tensor-identical | Fixed endpoint is unambiguous | Use latest path plus ordered fingerprint | closed |
| G0-F03 | informational | Dataset/config | canonical hash, 50x8 DA shape, LE1, 25 rounds, seed42 | Matches global freeze | Preserve exactly | closed |
| G0-F04 | major | True post-hoc boundary | historical A0T/A4 are interleaved; the old helper is not a formal API | Cannot call historical endpoints post-hoc evidence | Added `gaps_flower.posthoc_commissioning` with no rounds/server/client fields; calibration-only and test-manifest rejection tests pass | closed |
| G0-F05 | major | A4 state availability | the source-only FedAvg endpoint has no interleaved A4 semantic prototype/client-residual state | Some configured A4 terms can be unavailable in true post-hoc use | Keep registered coefficients, record per-loss input availability/activity, do not fabricate state | accepted limitation |
| G0-F06 | informational | Target-head ownership | `feat_proj` is the projection after TCN and attention pooling and before the classifier | It is a personalization head, not the shared temporal encoder | Train `feat_proj` and `classifier`; no classifier-only expansion required | closed |
| G0-F07 | informational | Phase observability | canonical metadata stores phase beside acquisition window start/end; source-DG uses source labels/phase only | No target information is needed for G2 | Keep class-phase for G2; G3 phase use remains subject to deployment observability audit | closed for G2 |

## Leakage assessment

- G1 may use only C5 calibration X/class and, for A4, calibration phase. Concentration is unused. Target test is inaccessible until all endpoints are locked.
- G2 training may use C1/C2 X/class/phase only and has no target argument.
- G3 labeled training may use 80 C5 X/class windows. The 240 unlabeled windows must be represented by a training type that contains X and identity only. Hidden truth is permitted only after endpoint lock for a clearly labeled post-hoc diagnostic.
- The canonical protocol is window-level and has known strict non-overlap limitations; none of these results may be described as strict experiment-independent generalization.

## Baseline, completeness, and reproducibility assessment

The source endpoint is approved for reuse. G1 has the minimum required baseline and holds all endpoint inputs fixed except the intended adaptation mechanism. G2 changes one source-local loss family. G3 uses a shared source checkpoint, identity allocation, backbone, and final update budget. Seed 42 is the frozen scope, not a multi-seed uncertainty claim.

## Verdict: approved

Checkpoint, data provenance, and the true post-hoc API boundary pass. No source retraining is authorized or needed. Gate 1 may execute the pre-registered matrix.

## Unknowns and handoff

- Exact G1 post-hoc performance and active A4 losses are unknown until execution.
- Exact MME architecture identity is unknown pending the Gate-3 feasibility audit; it must be labeled exact or compatible, never inferred.
- Existing result/checkpoint assets remain read-only. New evidence is written only to `results/iotj_canonical_v1_method_redesign_20260811/`.
