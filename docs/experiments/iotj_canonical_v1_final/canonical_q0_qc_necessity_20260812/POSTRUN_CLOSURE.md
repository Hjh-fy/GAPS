# Canonical-v1 Q0 post-run closure

Status: `FORMAL_COMPLETE_AUDIT_PASS`

Formal result root:
`results/iotj_canonical_v1_final/canonical_q0_qc_necessity_20260812/`

The frozen final regression backend was `R84_CONCAT`. Q0 completed with the
decision `MULTISIGNAL_QC_NOT_ESTABLISHED` because the exact historical
equal-mean Q4 could not be reconstructed from fully authorized canonical
inputs. It was recorded as `Q4_CANONICAL_INPUTS_UNAVAILABLE`; no substitute
multisignal formula was introduced.

Primary C5 NRMSE-AURC was `0.0793538309` for classification confidence and
`0.0719127937` for regression uncertainty. Target-stratified pooled NRMSE-AURC
was `0.0568167369` and `0.0524122410`, respectively. These values show that the
registered regression-dispersion signal is informative in C5 and pooled
ranking, but they do not establish historical equal-mean multisignal QC.

The registered Q1 trigger is satisfied. Q1 was not started by this closure.
Formal Q0 result bytes and their SHA256 index were not modified.
