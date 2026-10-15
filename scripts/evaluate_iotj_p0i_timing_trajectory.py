"""Post-hoc evaluator for completed P0-I training; this is the only C5-test entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.evaluate_checkpoint import evaluate_classification, load_checkpoint_model, make_loader
from scripts.evaluate_iotj_p0_roundwise_routing import enrich
from utils import compute_mmd2, deep_coral_loss

MILESTONES=(0,100,250,500,1000,1500,2000,2500)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise RuntimeError(f"empty output {path}")
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metrics(checkpoint: Path, loader, device) -> dict:
    model,_,_=load_checkpoint_model(str(checkpoint),device,32)
    return enrich(evaluate_classification(model,loader,device,4,15))


def features(checkpoint: Path, loader, device) -> np.ndarray:
    model,_,_=load_checkpoint_model(str(checkpoint),device,32); model.eval(); values=[]
    with torch.no_grad():
        for batch in loader:
            x=batch[0] if isinstance(batch,(tuple,list)) else batch
            _logits,feat,_=model(x.to(device)); values.append(feat.cpu())
    return torch.cat(values).numpy()


def discrepancy(checkpoint: Path, source_loader, target_loader, device) -> dict:
    source=torch.from_numpy(features(checkpoint,source_loader,device)); target=torch.from_numpy(features(checkpoint,target_loader,device))
    return {"global_mmd2":float(compute_mmd2(source,target)),"coral_discrepancy":float(deep_coral_loss(source,target)),
            "source_feature_norm":float(source.norm(dim=1).mean()),"target_feature_norm":float(target.norm(dim=1).mean())}


def row(round_id,role,checkpoint,target_loader,device):
    m=metrics(checkpoint,target_loader,device)
    return {"round":round_id,"checkpoint_role":role,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"],"nll":m["nll"],"ece":m["ece"],"num_examples":m["num_examples"],"selection_role":"posthoc_diagnostic_only"}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--result-root",default="results/iotj_p0_adaptation_timing_20260803"); parser.add_argument("--data-root",default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"); parser.add_argument("--docs-root",default="docs/experiments/iotj_p0_adaptation_timing_20260803"); parser.add_argument("--device",default="cpu"); args=parser.parse_args()
    root=Path(args.result_root).resolve(); docs=Path(args.docs_root).resolve(); docs.mkdir(parents=True,exist_ok=True); evaluation=root/"evaluation"; evaluation.mkdir(exist_ok=False)
    post_dir=root/"P0I_POSTHOC_UDA2500_S42"; inter_root=root/"P0I_INTERLEAVED_UDA25X100_S42"
    i2_manifest=json.loads((post_dir/"protocol_manifest.json").read_text()); i3_manifest=json.loads((inter_root/"protocol_manifest.json").read_text())
    if i2_manifest.get("status")!="training_completed_test_unopened" or i3_manifest.get("status")!="training_completed_test_unopened":
        raise RuntimeError("FAIL_CLOSED sealed C5 test cannot open before both training runs complete")
    data=Path(args.data_root).resolve(); device=torch.device(args.device)
    target_loader=make_loader(data,5,"test",32); c1_loader=make_loader(data,1,"test",32); c2_loader=make_loader(data,2,"test",32)
    source_cal=np.concatenate([np.load(data/f"client_{c}"/"calibration_features.npy",allow_pickle=False).astype(np.float32) for c in (1,2)])
    target_cal=np.load(data/"client_5"/"calibration_features.npy",allow_pickle=False).astype(np.float32)
    source_x_loader=DataLoader(TensorDataset(torch.from_numpy(source_cal)),batch_size=32)
    target_x_loader=DataLoader(TensorDataset(torch.from_numpy(target_cal)),batch_size=32)
    pure_dir=REPO_ROOT/"results/iotj_p0_routing_simplification_20260803/P0A_PURE_FEDAVG_LE1_S42/remote_server"
    inter_dir=inter_root/"remote_server"
    inter_rows=[]; retention=[]; discrepancies=[]; gaps=[]; pure_target={}
    for round_id in range(1,26):
        checkpoints=[("pure_source_reference",pure_dir/f"server_round_{round_id:03d}.pth"),("interleaved_pre_uda",inter_dir/f"server_round_{round_id:03d}_pre_uda.pth"),("interleaved_post_uda",inter_dir/f"server_round_{round_id:03d}_post_uda.pth")]
        for role,checkpoint in checkpoints:
            target=row(round_id,role,checkpoint,target_loader,device)
            if role!="pure_source_reference": inter_rows.append(target)
            else: pure_target[round_id]=target
            c1=metrics(checkpoint,c1_loader,device); c2=metrics(checkpoint,c2_loader,device); source_mean=(c1["macro_f1"]+c2["macro_f1"])/2
            retention.append({"round":round_id,"checkpoint_role":role,"C1_accuracy":c1["accuracy"],"C1_macro_f1":c1["macro_f1"],"C2_accuracy":c2["accuracy"],"C2_macro_f1":c2["macro_f1"],"source_mean_macro_f1":source_mean})
            disc=discrepancy(checkpoint,source_x_loader,target_x_loader,device); discrepancies.append({"round":round_id,"checkpoint_role":role,**disc})
            gaps.append({"round":round_id,"checkpoint_role":role,"source_mean_macro_f1":source_mean,"target_macro_f1":target["macro_f1"],"generalization_gap":source_mean-target["macro_f1"]})
    milestone_rows=[]
    for step in MILESTONES:
        m=metrics(post_dir/f"step_{step:04d}.pth",target_loader,device)
        milestone_rows.append({"step":step,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"],"nll":m["nll"],"ece":m["ece"],"selection_role":"posthoc_diagnostic_only","formal_endpoint":step==2500})
    refs={
      "I0_source_only":pure_dir/"server_round_025.pth",
      "I1_posthoc_uda100":REPO_ROOT/"results/iotj_p0_zero_label_commissioning_20260803/U1_UNSUPERVISED_GLOBAL_ALIGNMENT/u1_adapted.pth",
      "I2_posthoc_uda2500":post_dir/"step_2500.pth",
      "I3_interleaved_uda_25x100":inter_dir/"server_round_025_post_uda.pth",
      "S1_target_ce100":REPO_ROOT/"results/iotj_p0_routing_simplification_20260803/P0B_ROUNDWISE_COMMISSIONING_S42/round25_simple_ce_adapted.pth",
    }
    p0u={r["method"]:r for r in csv.DictReader((REPO_ROOT/"docs/experiments/iotj_p0_zero_label_commissioning_20260803/zero_label_commissioning_comparison.csv").open(encoding="utf-8"))}
    comparison=[]
    configs={"I0_source_only":(0,"none",0.0),"I1_posthoc_uda100":(100,"posthoc_after_round25",float(p0u["unsupervised_global_alignment"]["commissioning_seconds"])),"I2_posthoc_uda2500":(2500,"posthoc_after_round25",i2_manifest["uda_wall_seconds"]),"I3_interleaved_uda_25x100":(2500,"100_after_each_round",sum(json.loads((inter_dir/"interleaved_uda_diagnostics.json").read_text())[r*100]["uda_wall_seconds_per_round"] for r in range(25))),"S1_target_ce100":(100,"posthoc_after_round25",float(p0u["simple_target_ce"]["commissioning_seconds"]))}
    for method,checkpoint in refs.items():
        m=metrics(checkpoint,target_loader,device); c1=metrics(checkpoint,c1_loader,device); c2=metrics(checkpoint,c2_loader,device); steps,schedule,seconds=configs[method]
        comparison.append({"method":method,"target_label_access":"C5_calibration_labels" if method.startswith("S1") else "none_x_only" if steps else "none","federated_rounds":25,"local_epochs":1,"total_uda_steps":steps,"adaptation_schedule":schedule,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"],"nll":m["nll"],"ece":m["ece"],"source_mean_macro_f1":(c1["macro_f1"]+c2["macro_f1"])/2,"total_uda_seconds":seconds,"seed":42,"calculation_status":"recomputed_posthoc"})
    outputs={"interleaved_roundwise_target_metrics.csv":inter_rows,"source_retention_vs_round.csv":retention,"domain_discrepancy_vs_round.csv":discrepancies,"source_target_generalization_gap.csv":gaps,"posthoc_uda2500_target_trajectory.csv":milestone_rows,"adaptation_timing_comparison.csv":comparison}
    for name,rows in outputs.items(): write_csv(evaluation/name,rows); write_csv(docs/name,rows)
    # Four minimum diagnostic figures; all are descriptive and non-selective.
    rounds=np.arange(1,26)
    plt.figure(); plt.plot(rounds,[pure_target[r]["macro_f1"] for r in rounds],label="Pure FedAvg"); plt.plot(rounds,[r["macro_f1"] for r in inter_rows if r["checkpoint_role"]=="interleaved_pre_uda"],label="Interleaved PRE"); plt.plot(rounds,[r["macro_f1"] for r in inter_rows if r["checkpoint_role"]=="interleaved_post_uda"],label="Interleaved POST"); plt.xlabel("Round"); plt.ylabel("Target Macro-F1"); plt.legend(); plt.tight_layout(); plt.savefig(docs/"fig_target_macro_f1_timing_vs_round.png",dpi=200); plt.close()
    plt.figure();
    for role in ("pure_source_reference","interleaved_pre_uda","interleaved_post_uda"): plt.plot(rounds,[r["global_mmd2"] for r in discrepancies if r["checkpoint_role"]==role],label=role)
    plt.xlabel("Round"); plt.ylabel("Global MMD2"); plt.legend(); plt.tight_layout(); plt.savefig(docs/"fig_domain_discrepancy_vs_round.png",dpi=200); plt.close()
    plt.figure();
    for role in ("pure_source_reference","interleaved_pre_uda","interleaved_post_uda"): plt.plot(rounds,[r["generalization_gap"] for r in gaps if r["checkpoint_role"]==role],label=role)
    plt.xlabel("Round"); plt.ylabel("Source-target Macro-F1 gap"); plt.legend(); plt.tight_layout(); plt.savefig(docs/"fig_source_target_gap_vs_round.png",dpi=200); plt.close()
    plt.figure(); plt.plot([r["step"] for r in milestone_rows],[r["macro_f1"] for r in milestone_rows],marker="o"); plt.xlabel("Post-hoc UDA step"); plt.ylabel("C5 Macro-F1 (post-hoc diagnostic)"); plt.tight_layout(); plt.savefig(docs/"fig_posthoc_uda2500_trajectory.png",dpi=200); plt.close()
    (evaluation/"evaluation_manifest.json").write_text(json.dumps({"status":"completed","sealed_test_opened_after_training":True,"selection_performed":False,"formal_I2_step":2500,"formal_I3_endpoint":"round25_post_uda","target_rows":len(inter_rows),"source_retention_rows":len(retention),"discrepancy_rows":len(discrepancies)},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"completed","comparison":comparison},sort_keys=True))


if __name__=="__main__": main()
