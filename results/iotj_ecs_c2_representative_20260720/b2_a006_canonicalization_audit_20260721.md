# B2-s42/a006 recovery canonicalization audit

## Scope

This append-only audit records the explicit authorization of `c12_to_c5__b2__s42__a006` as a **canonicalized representative real-system run**. It does not edit or supersede the immutable controller status in `raw/.../attempt_status.json`, which remains `failed / process_failure`.

## Basis

| Check | Evidence | Result |
|---|---|---|
| Training completion | Manual recovery system metrics | 25 completed rounds |
| Flower message trace | Corrected manual validation | 50 FitIns and 50 FitRes |
| Resource trace completeness | Corrected manual validation | C1 97.274%; C2 97.464% |
| Recovery integrity | Manual recovery system metrics | remote originals preserved; selected remote/local SHA-256 values match |
| Numerical configuration identity | Source status and validation | commit `2ef7aea`; archive `52bdbf965680`; dataset `fb8946da138b`; protocol `ba289bf87a7` |

## Authorization and use boundary

The user authorized this promotion on 2026-07-21. The derived label is `canonicalized_recovery_authorized` and can be used only in the B2/B5 representative ECS + Raspberry Pi + ECS-C2 system-cost table (communication, timing, resources, and Observer measurements).

It does **not** make the original attempt controller-canonical, and it does **not** authorize B2 to enter any future five-seed algorithm mean/std, paired-difference, or significance claim. Historical seed-42 classification metrics remain screening evidence.

## Verdict

**Approved with a scoped recovery exception:** canonical representative real-system evidence only.
