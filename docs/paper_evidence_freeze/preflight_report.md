# IoT-J paper evidence freeze preflight

- Status: PASS
- Branch: `codex/iotj-confirmation-observability`
- Local/origin HEAD at preflight: `4a846fc36512bb93f5e310a0e5789ce17eb21968`
- Canonical source uniquely identified: `shared-root/docs/paper/GAPS_IoTJ_traditional_draft_20260720.zh.html`
- Identification basis: it is the only IoT-J HTML in the shared project root; the active worktree contained no competing HTML manuscript.
- Canonical source Git state: untracked in the shared root, therefore imported by SHA and backed up byte-for-byte.
- Runtime v4 six frozen SHA status: PASS.
- Required classification, regression, H1, runtime v5, QC, benchmark, low-calibration, and harmonization evidence: present.
- Existing unrelated modifications were observed and left untouched:
  - `results/iotj_a003_timing_diagnosis_20260719/a003_vs_b2_pilot_timing_analysis.md`
  - `results/iotj_advisor_metrics_20260721/build_advisor_workbook_v3.mjs`
- Temporary directories were not deleted or modified.
- No training, inference evaluation, benchmark, or test reopening was run.

## Working-tree snapshot

```text
 M results/iotj_a003_timing_diagnosis_20260719/a003_vs_b2_pilot_timing_analysis.md
 M results/iotj_advisor_metrics_20260721/build_advisor_workbook_v3.mjs
?? .final_benchmark_pi_staging/
?? .p/
?? .p1_pytest_bundle_final/
?? .paper_evidence_freeze_failed_backup2_20260726.html
?? .paper_evidence_freeze_failed_backup_20260726.html
?? .paper_evidence_freeze_failed_manuscript2_20260726.html
?? .paper_evidence_freeze_failed_manuscript_20260726.html
?? .paper_evidence_freeze_failed_validation2_20260726/
?? .paper_evidence_freeze_failed_validation_20260726/
?? .tmp_b5_regression_multiseed_smoke/
?? .tmp_h1_federated_ridge_equivalence_smoke/
?? .tmp_iotj_observer_gate_b2_canonical_sort_fix/
?? .tmp_iotj_observer_gate_b2_task11_refreeze_7ec77e3/
?? .tmp_iotj_observer_gate_b2_task11_refreeze_955a853/
?? .tmp_iotj_observer_gate_b2_task9_final_v10/
?? .tmp_iotj_observer_gate_b2_task9_final_v4/
?? .tmp_iotj_observer_gate_b2_task9_final_v5/
?? .tmp_iotj_observer_gate_b2_task9_final_v6/
?? .tmp_iotj_observer_gate_b2_task9_final_v7/
?? .tmp_iotj_observer_gate_b2_task9_final_v8/
?? .tmp_iotj_observer_gate_b2_task9_final_v9/
?? .tmp_iotj_observer_gate_b2_task9_review_fix/
?? .tmp_iotj_observer_gate_b2_task9_review_fix_v2/
?? .tmp_iotj_observer_gate_b2_task9_review_fix_v3/
?? .tmp_iotj_observer_gate_b5_canonical_sort_fix/
?? .tmp_iotj_observer_gate_b5_task11_refreeze_7ec77e3/
?? .tmp_iotj_observer_gate_b5_task11_refreeze_955a853/
?? .tmp_iotj_observer_gate_b5_task9_final_v10/
?? .tmp_iotj_observer_gate_b5_task9_final_v4/
?? .tmp_iotj_observer_gate_b5_task9_final_v5/
?? .tmp_iotj_observer_gate_b5_task9_final_v6/
?? .tmp_iotj_observer_gate_b5_task9_final_v7/
?? .tmp_iotj_observer_gate_b5_task9_final_v8/
?? .tmp_iotj_observer_gate_b5_task9_final_v9/
?? .tmp_iotj_observer_gate_b5_task9_review_fix/
?? .tmp_iotj_preliminary_metrics/
?? .tmp_pytest_lowcal_final_full/
?? .tmp_pytest_lowcal_finalize_fix/
?? .tmp_pytest_lowcal_fold_audit/
?? .tmp_pytest_lowcal_prerun/
?? .tmp_runtime_v5_qc_smoke/
?? .tmp_runtime_v5_qc_smoke_v2/
?? .tmp_runtime_v5_qc_smoke_v2_fix1/
?? .tmp_v5_h1_topology_smoke/
?? .tmp_v5_h1_topology_smoke2/
?? docs/experiments/iotj_handoff_20260722.zh.md
?? scripts/benchmark_iotj_b5_classifier_r4_preliminary.py
?? scripts/diagnose_iotj_source_tree_manifest.py
?? scripts/freeze_iotj_paper_evidence.py

```
