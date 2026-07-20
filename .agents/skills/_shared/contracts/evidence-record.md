# Evidence Record Contract

Each Evidence item contains `evidence_id`, `experiment_ids`, `metric_ids`, `comparison`, `source_paths`, `audit_status`, `support_strength`, `claim_ids`, `limitations`, and `provenance`.

Experimental Evidence may become `approved` only after `experiment-audit` reports no blocking comparability or provenance issue. Keep unaudited Evidence as `draft` or `blocked`; never promote it because the number looks plausible.
