# Target Profile Selector

- test_used_for_selection: `False`
- guardrail_status: `pass`
- feature_schema_status: `pass`
- runtime_parity_num_mismatch: `0`

| mode | selected_profile | fallback | reason |
| --- | --- | --- | --- |
| balanced | H2.3 | R3aK16/B0 | H2.3 remains the balanced no-QC full-set mainline. |
| co_priority | H8_plus_formal_C4_route_rescue | H2.3 | H8+C4 selected because guardrail, feature schema, and runtime parity all pass. |
| deployment_lite | H2.3 | H2.3 | L1 has no exported runtime bundle or benchmark pass yet. |

Limitations:
- H8+C4 is a guarded CO-priority specialist, not the balanced default.
- L1 is pending until a real deployment bundle proves a size/latency advantage.
