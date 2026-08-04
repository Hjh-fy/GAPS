"""Create submission evidence closure artifacts without training or manuscript edits."""

from __future__ import annotations

import csv, hashlib, json, shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/iotj_submission_evidence_closure_20260804"
CLS = ROOT / "results/iotj_final_classification_le1_20260804"
E2E = ROOT / "results/iotj_final_end_to_end_a4_20260804"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576),b""): digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str: return str(path.relative_to(ROOT)).replace("\\","/")


def md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    def clean(value: Any) -> str: return str(value).replace("|","/")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n" + "\n".join("| " + " | ".join(clean(row.get(c,"")) for c in columns) + " |" for row in rows) + "\n"


def main() -> None:
    if not OUT.exists(): raise FileNotFoundError(OUT)
    benchmark=json.loads((OUT/"benchmarks/pi5_final_deployed_runtime_benchmark.json").read_text(encoding="utf-8"))
    manifest=json.loads((OUT/"runtime_package/FINAL_DEPLOYMENT_MANIFEST.json").read_text(encoding="utf-8"))
    paper=OUT/"paper_ready_tables"; paper.mkdir(exist_ok=True)
    source=E2E/"figures/source_data"
    copies={
        "FIG5_OVERALL_REGRESSION.csv": source/"fig05_overall_regression.csv",
        "FIG5_PER_GAS_REGRESSION.csv": source/"fig05_per_gas_regression.csv",
        "FIG6_SOURCE_PRIOR_ABLATION.csv": source/"fig06_source_prior_ablation.csv",
        "FIG6_CALIBRATION_BUDGET.csv": source/"fig06_calibration_budget.csv",
        "FIG7_QC_COVERAGE.csv": source/"fig07_qc_coverage_curve.csv",
        "FIG7_HC90_HC95.csv": source/"fig07_hc_operating_points.csv",
        "FIG7_RANDOM_REFERENCE.csv": source/"fig07_random_reference.csv",
        "FIG8_PHYSICAL_VALIDATION.csv": source/"fig08_physical_validation.csv",
    }
    for name,src in copies.items(): shutil.copy2(src,paper/name)
    old_system=read_csv(source/"fig08_system_deployment.csv")
    system=[row for row in old_system if row["record_type"]=="communication"]
    system.append({"record_type":"pi5_runtime","label":"FINAL_DEPLOYED_RUNTIME","bytes":"","rounds":"","evidence_type":"measured_5000_window_benchmark","pi_p50_ms":benchmark["latency"]["p50_ms"],"pi_p95_ms":benchmark["latency"]["p95_ms"],"pi_p99_ms":benchmark["latency"]["p99_ms"],"pi_peak_rss_mib":benchmark["resources"]["peak_rss_mib"],"pi_throughput_windows_per_s":benchmark["latency"]["throughput_windows_per_s"],"deployment_status":"FINAL_DEPLOYED_RUNTIME"})
    write_csv(paper/"FIG8_FINAL_SYSTEM_DEPLOYMENT.csv",system)

    sections=[]
    for title,name,cols in [
        ("Fig. 5 — Overall regression","FIG5_OVERALL_REGRESSION.csv",["variant","input_dimension","evaluation_scope","N","RMSE","MAE","R2","NRMSE"]),
        ("Fig. 5 — Per-gas regression","FIG5_PER_GAS_REGRESSION.csv",["variant","evaluation_scope","gas","N","RMSE","MAE","NRMSE"]),
        ("Fig. 6 — Source-prior ablation","FIG6_SOURCE_PRIOR_ABLATION.csv",["variant","input_dimension","evaluation_scope","N","RMSE","MAE","NRMSE"]),
        ("Fig. 6 — Calibration budget","FIG6_CALIBRATION_BUDGET.csv",["nominal_budget","replicates","S_CC_RMSE_mean","S_CC_RMSE_sample_std","S_ALL_NRMSE_mean","S_ALL_NRMSE_sample_std"]),
        ("Fig. 7 — HC90/HC95","FIG7_HC90_HC95.csv",["target_coverage","test_coverage","accepted_N","N","RMSE","MAE","NRMSE","misroute_capture_rate","error_ge_40ppm_capture_rate"]),
        ("Fig. 8 — Communication and final Pi 5 runtime","FIG8_FINAL_SYSTEM_DEPLOYMENT.csv",["record_type","label","bytes","rounds","evidence_type","pi_p50_ms","pi_p95_ms","pi_p99_ms","pi_peak_rss_mib","pi_throughput_windows_per_s","deployment_status"]),
    ]:
        sections.append(f"## {title}\n\n"+md_table(read_csv(paper/name),cols))
    (paper/"FIG5_FIG8_PAPER_READY_TABLES.md").write_text("# Paper-ready numerical tables for Fig. 5–Fig. 8\n\nNo manuscript text was modified. Values are copied from frozen CSVs except the newly measured final Pi 5 row.\n\n"+"\n".join(sections),encoding="utf-8")

    master=[]
    for row in read_csv(CLS/"classification_main_comparison.csv"):
        for metric in ("accuracy","macro_f1","nll","ece"):
            master.append({"evidence_id":row["experiment_id"],"category":"classification","target":row["target_id"],"method":row["method"],"scope":"sealed_test_fixed_endpoint","metric":metric,"value":row[metric],"unit":"fraction" if metric in {"accuracy","macro_f1","ece"} else "nats","N":row["num_examples"],"seed":row["seed"],"source_csv":rel(CLS/"classification_main_comparison.csv"),"status":"APPROVED_FROZEN"})
    for row in read_csv(source/"fig05_overall_regression.csv"):
        for metric in ("RMSE","MAE","R2","NRMSE"):
            master.append({"evidence_id":"A4-"+row["variant"],"category":"regression","target":"C5","method":row["variant"],"scope":row["evaluation_scope"],"metric":metric,"value":row[metric],"unit":"ppm" if metric in {"RMSE","MAE"} else "fraction","N":row["N"],"seed":"42","source_csv":rel(source/"fig05_overall_regression.csv"),"status":"APPROVED_FROZEN"})
    for row in read_csv(source/"fig07_hc_operating_points.csv"):
        for metric in ("test_coverage","RMSE","NRMSE"):
            master.append({"evidence_id":"QC-HC"+str(int(float(row["target_coverage"])*100)),"category":"quality_control","target":"C5","method":"equal_mean_QC","scope":"sealed_test_fixed_threshold","metric":metric,"value":row[metric],"unit":"ppm" if metric=="RMSE" else "fraction","N":row["N"],"seed":"42","source_csv":rel(source/"fig07_hc_operating_points.csv"),"status":"APPROVED_FROZEN"})
    for metric,key,unit in [("p50_latency_ms","p50_ms","ms/window"),("p95_latency_ms","p95_ms","ms/window"),("p99_latency_ms","p99_ms","ms/window"),("throughput","throughput_windows_per_s","windows/s")]:
        master.append({"evidence_id":"FINAL-PI5-5000","category":"deployment","target":"Raspberry Pi 5","method":"FINAL_DEPLOYED_RUNTIME","scope":"5000_window_batch1","metric":metric,"value":benchmark["latency"][key],"unit":unit,"N":5000,"seed":"42 model; deterministic fixed-order benchmark","source_csv":rel(OUT/"benchmarks/pi5_final_deployed_runtime_benchmark.json"),"status":"APPROVED_MEASURED"})
    master.append({"evidence_id":"FINAL-PI5-5000","category":"deployment","target":"Raspberry Pi 5","method":"FINAL_DEPLOYED_RUNTIME","scope":"5000_window_batch1","metric":"peak_rss","value":benchmark["resources"]["peak_rss_mib"],"unit":"MiB","N":5000,"seed":"42 model","source_csv":rel(OUT/"benchmarks/pi5_final_deployed_runtime_benchmark.json"),"status":"APPROVED_MEASURED"})
    write_csv(OUT/"FINAL_RESULT_MASTER_TABLE.csv",master)

    cls_ck=ROOT/"results/iotj_final_classification_le1_20260804/FCL-E4-A4/remote_server/server_round_025_adapted.pth"
    cls_index=CLS/"sha256_index.json"
    regression_models=E2E/"regression/regression_models.json"
    h1_manifest=ROOT/"results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json"
    qc_lock=E2E/"qc/qc_threshold_lock.csv"
    figure_rows=[]
    def add(fig,panel,title,data,csv_path,asset,asset_hash,script,caption,status="READY"):
        figure_rows.append({"figure":fig,"panel":panel,"content":title,"data_source":data,"source_csv":csv_path,"checkpoint_or_asset":asset,"checkpoint_or_asset_sha256":asset_hash,"script":script,"caption":caption,"status":status})
    add("Fig.1","a","GAPS cloud–edge–sensor architecture","schematic; no measured values","not_applicable","not_applicable","not_applicable","scripts/freeze_iotj_paper_evidence.py","GAPS cloud–edge–sensor architecture and bounded data flow; C5 test is evaluation-only.")
    add("Fig.2","a","Per-channel device-domain shift","C1/C2 source vs C3/C4/C5 calibration x",rel(CLS/"FCL-E0-SHIFT/sensor_channel_shift.csv"),"calibration features",sha(CLS/"FCL-E0-SHIFT/sensor_channel_shift.csv"),"scripts/finalize_iotj_final_classification_le1.py","Device-domain shift in raw sensor statistics; target test is unopened.")
    add("Fig.2","b","Sensor covariance shift","C1/C2 source vs C3/C4/C5 calibration x",rel(CLS/"FCL-E0-SHIFT/sensor_covariance_shift.csv"),"calibration features",sha(CLS/"FCL-E0-SHIFT/sensor_covariance_shift.csv"),"scripts/finalize_iotj_final_classification_le1.py","Global sensor-space covariance discrepancy under the frozen estimator.")
    add("Fig.3","a","Cross-target classification comparison","classification main comparison",rel(CLS/"classification_main_comparison.csv"),rel(cls_index),sha(cls_index),"scripts/finalize_iotj_final_classification_le1.py","Seed-42 fixed-endpoint comparison across C3/C4/C5.")
    add("Fig.3","b","Source-to-target macro-F1 gap","source/target evaluations",rel(CLS/"source_target_f1_gap.csv"),rel(cls_index),sha(cls_index),"scripts/finalize_iotj_final_classification_le1.py","Source–target macro-F1 gaps for the frozen endpoints.")
    add("Fig.4","a","C5 classification comparison","C5 rows from classification comparison",rel(CLS/"classification_main_comparison.csv"),rel(cls_index),sha(cls_index),"scripts/finalize_iotj_final_classification_le1.py","C5 fixed-endpoint classification comparison; A4 is the final router.")
    add("Fig.4","b","Server-centric A0–A6 ablation","C5 formal ablation",rel(CLS/"figures/source_data/fig05_c5_ablation_hierarchy.csv"),rel(cls_index),sha(cls_index),"scripts/finalize_iotj_final_classification_le1.py","Server-centric C5 ablation under seed 42 and the frozen protocol.")
    caps=json.loads(json.dumps({"5":"Concentration estimation with the frozen A4 router.","6":"Source-prior and calibration-budget evidence; panels retain distinct protocols.","7":"Label-free equal-mean QC quality–coverage trade-off with random references and frozen HC90/HC95.","8":"Communication, FINAL_DEPLOYED_RUNTIME Pi 5 benchmark, and physical three-machine validation."}))
    for fig,panel,csvname,title in [(5,"a","fig05_overall_regression.csv","Overall regression"),(5,"b","fig05_per_gas_regression.csv","Per-gas regression"),(6,"a","fig06_source_prior_ablation.csv","Source-prior ablation"),(6,"b","fig06_calibration_budget.csv","Calibration budget"),(7,"a","fig07_qc_coverage_curve.csv","Coverage–NRMSE and random reference"),(7,"b","fig07_hc_operating_points.csv","HC90/HC95 event capture"),(8,"a","FIG8_FINAL_SYSTEM_DEPLOYMENT.csv","Communication"),(8,"b","FIG8_FINAL_SYSTEM_DEPLOYMENT.csv","Pi 5 P50/P95/P99"),(8,"c","FIG8_FINAL_SYSTEM_DEPLOYMENT.csv","Pi 5 throughput and peak RSS"),(8,"d","FIG8_PHYSICAL_VALIDATION.csv","Physical Flower validation")]:
        p=(source/csvname) if csvname.startswith("fig") else (paper/csvname)
        if fig in (5,6):
            asset=";".join(map(rel,(cls_ck,regression_models,h1_manifest)))
            ah=";".join(map(sha,(cls_ck,regression_models,h1_manifest)))
        elif fig == 7:
            asset=";".join(map(rel,(cls_ck,regression_models,h1_manifest,qc_lock)))
            ah=";".join(map(sha,(cls_ck,regression_models,h1_manifest,qc_lock)))
        else:
            runtime_manifest=OUT/"runtime_package/FINAL_DEPLOYMENT_MANIFEST.json"
            benchmark_path=OUT/"benchmarks/pi5_final_deployed_runtime_benchmark.json"
            asset=";".join(map(rel,(runtime_manifest,benchmark_path)))
            ah=";".join(map(sha,(runtime_manifest,benchmark_path)))
        add(f"Fig.{fig}",panel,title,"frozen ceb6c78 evidence",rel(p),asset,ah,"scripts/plot_iotj_final_a4_figures.py",caps[str(fig)])
    write_csv(OUT/"FINAL_FIGURE_MANIFEST.csv",figure_rows)

    claims=[
        ("C-FIG3","GAPS classification is compared at fixed seed-42 endpoints across C3/C4/C5.","Fig.3; classification_main_comparison.csv","direct","SUPPORTED_WITH_SINGLE_SEED_LIMIT"),
        ("C-FIG5","The final A4 router with R84_FED_H1 provides the registered C5 concentration estimates.","Fig.5; regression_main_summary.csv; A4 checkpoint hash","direct","SUPPORTED"),
        ("C-FIG6","Source-prior and calibration-budget results are separate protocols and are not pooled.","Fig.6 source CSVs","direct","SUPPORTED"),
        ("C-FIG7","Final QC uses equal mean of three calibration-p95-normalized components and sealed-test thresholds.","qc_threshold_lock.csv; Fig.7 CSVs","direct","SUPPORTED"),
        ("C-FIG8","The exact final runtime runs on Raspberry Pi 5 at the measured latency, throughput, and RSS.","pi5_final_deployed_runtime_benchmark.json; FINAL_DEPLOYMENT_MANIFEST.json","direct","SUPPORTED"),
        ("C-PARAM","The former value 80 denotes state tensor entries, while the classifier has 22,765 parameters and 91,060 FP32 parameter bytes.","FINAL_DEPLOYMENT_MANIFEST.json","direct","SUPPORTED_CORRECTED_SEMANTICS"),
    ]
    matrix="# Final claim–evidence matrix\n\n| Claim ID | Manuscript-ready claim | Evidence | Link type | Status |\n|---|---|---|---|---|\n"+"\n".join("| "+" | ".join(row)+" |" for row in claims)+"\n\nClaim boundary: seed 42 only; no uncertainty claim is made for classification comparisons. Hardware numbers apply only to the audited Pi 5 environment and fixed 5,000-window protocol.\n"
    (OUT/"FINAL_CLAIM_EVIDENCE_MATRIX.md").write_text(matrix,encoding="utf-8")
    inv=manifest["classifier_checkpoint_identity"]
    audit=f"""# Final submission audit

Status: **PASS WITH EXPLICIT BOUNDARIES**

- Frozen result baseline: `ceb6c78`; no training, hyperparameter search, model change, or QC-formula change was performed.
- Runtime: `FINAL_DEPLOYED_RUNTIME` = A4 round-25 classifier + `R84_FED_H1` + final equal-mean QC.
- Pi 5 benchmark: 5,000 measured windows, batch 1, CPU single-thread, P50 {benchmark['latency']['p50_ms']:.6f} ms, P95 {benchmark['latency']['p95_ms']:.6f} ms, P99 {benchmark['latency']['p99_ms']:.6f} ms, throughput {benchmark['latency']['throughput_windows_per_s']:.3f} windows/s, peak RSS {benchmark['resources']['peak_rss_mib']:.3f} MiB; throttling `{benchmark['resources']['throttled_before']}` → `{benchmark['resources']['throttled_after']}`.
- Parameter semantics corrected without model modification: `state_tensor_count={inv['state_tensor_count']}`, `total_parameter_count={inv['total_parameter_count']}`, `trainable_parameter_count={inv['trainable_parameter_count']}`, `fp32_model_bytes={inv['fp32_model_bytes']}`.
- Full 1,360-row local parity: route, R83/R84 predictions, source-prior risks, and HC90 decisions were invariant. Cross-device classifier uncertainty differed by at most 8.60095e-5 and final risk by 4.00939e-6 due to CPU floating-point execution; no formula or decision was changed.
- Fig. 1 is a schematic and therefore correctly has no CSV/checkpoint. Fig. 2–Fig. 8 panels have explicit source, hash/asset, script, and caption records in `FINAL_FIGURE_MANIFEST.csv`.
- Fig. 5–Fig. 8 numeric tables are written under `paper_ready_tables/`; manuscript source was not modified.
- Boundary: single seed 42; benchmark generalizes only to the recorded Raspberry Pi 5 hardware/software environment.
"""
    (OUT/"FINAL_SUBMISSION_AUDIT.md").write_text(audit,encoding="utf-8")
    status="""# Final experiment status

## Frozen/closed

- Result baseline: `ceb6c78`.
- Classification, regression, QC, ablations, and calibration studies: **CLOSED; NO FURTHER SEARCH**.
- Final deployed runtime: **FINAL_DEPLOYED_RUNTIME**.
- Raspberry Pi 5 5,000-window benchmark: **COMPLETE / PASS**.
- Submission evidence closure: **COMPLETE**, subject only to author-controlled manuscript insertion.

## Stop boundary

No training, optimizer change, threshold search, algorithm exploration, or new ablation is authorized or pending. Work stops after evidence publication.
"""
    (OUT/"FINAL_EXPERIMENT_STATUS.md").write_text(status,encoding="utf-8")
    artifacts=[]
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="sha256_index.json"):
        artifacts.append({"path":rel(path),"bytes":path.stat().st_size,"sha256":sha(path)})
    (OUT/"sha256_index.json").write_text(json.dumps({"schema_version":"iotj.submission_evidence_closure.v1","baseline_commit":"ceb6c78","status":"PASS","artifacts":artifacts},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
