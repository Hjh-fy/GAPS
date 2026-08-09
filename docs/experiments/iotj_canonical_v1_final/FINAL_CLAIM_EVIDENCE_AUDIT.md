# Final claim-evidence audit

## Verdict before new execution

| Claim | Existing canonical evidence | Finding | Severity | Required action |
|---|---|---|---|---|
| Standard FL is insufficient under target shift | A4 only; FedAvg/FedProx/SCAFFOLD are legacy preprocessing | `CANONICAL_COMPARATOR_MISSING` | blocking | Run minimal canonical comparators |
| Unlabeled alignment and labeled commissioning are distinct regimes | No canonical MMD/A0T | `CANONICAL_COMPARATOR_MISSING` | blocking | Run MMD and A0T with explicit information table |
| GAPS benefit is not only target-label access | A0T preregistered, no endpoint | `SUBMISSION_BLOCKER_P0` | blocking | Run frozen equal-label A0T |
| Routing errors propagate to regression | Canonical prediction and S_ALL/S_CC/oracle artifacts exist | reusable | informational | Read-only routing analysis |
| Federated H1 contributes to regression | Matched canonical 83D/84D predictions exist | reusable with uncertainty gap | major | Raw-file-grouped paired bootstrap |
| QC identifies risk beyond reduced coverage | HC90/HC95 and same-budget random exist | reusable with capture/AURC gap | major | Read-only risk/capture analysis |
| Strict non-overlap conclusion holds | Exact identity overlap 0; raw-time overlap about 29% | `SUBMISSION_BLOCKER_P0` | blocking | Separate strict grouped robustness run |
| Edge claims match the deployed package | Package/Pi/model-size hashes exist | reusable | informational | Hash and communication audit |

The historical comparator root uses `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid` and cannot populate the canonical table. Existing canonical assets remain read-only. Exactly 12 missing executable configurations are frozen; no other algorithm is authorized.
