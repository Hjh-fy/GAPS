# Experiment Record Contract

Use one row per executable experiment configuration. Never infer values from directory names alone.

| Field | Meaning |
|---|---|
| `experiment_id` | Stable identifier such as `EXP-031`; never reuse it. |
| `source_clients` | Ordered source-client set, for example `C1;C2`. |
| `target_clients` | Ordered target-client set, for example `C3;C4;C5`. |
| `split_protocol` | Explicit protocol such as `8:2` or `7:2:1`, including role semantics. |
| `model` | Exact model/profile name. |
| `checkpoint` | Exact checkpoint path or immutable identifier. |
| `DA` | Domain-adaptation mode, including `none`, `fixed`, or `strong` when verified. |
| `calibration` | Calibration mode such as `none`, `bias`, `affine`, `full`, `specialist`, or `auto_v2`. |
| `QC` | QC policy/version and accepted/review/reject scope. |
| `seed` | Integer seed or explicit seed set. |
| `result_path` | Repository-relative result path. |
| `metrics` | Metric record IDs or a machine-readable summary reference. |
| `status` | Workflow status defined below. |
| `notes` | Limitations, unresolved ambiguity, or operator context. |

Recommended provenance fields: `code_commit`, `config_path`, `dataset_path`, `created_at`, `evidence_status`, `provenance`.

Allowed workflow statuses: `draft`, `registered`, `completed`, `audited`, `approved`, `blocked`, `conflict`. Use `unknown` as a field value when evidence is absent. Use `conflict` when two traceable sources disagree; preserve both sources in `notes` or `provenance`.
