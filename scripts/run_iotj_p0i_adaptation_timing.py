"""Run frozen P0-I post-hoc or three-host interleaved UDA training."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays
from gaps_flower.evaluate_checkpoint import load_checkpoint_model
from gaps_flower.p0i_adaptation import feature_only_loader, run_frozen_u1
from scripts.run_iotj_confirmation_observability import _start_ecs_c2_tunnels, _terminate_processes
from scripts.run_iotj_r1_m2_distributed_baselines import deploy_archive, ensure_idle, process_command, run, sha256_file, ssh

SEED = 42
ROUNDS = 25
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
CLIENT_LR = 5e-4
UDA_LR = 5e-4
SOURCE_SHA256 = "4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c"
MILESTONES = {0, 100, 250, 500, 1000, 1500, 2000, 2500}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def source_loader(data_root: Path) -> DataLoader:
    arrays = load_domain_adaptation_arrays([data_root / "client_1", data_root / "client_2"], strict=True)
    x, y, _phase = arrays
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    indices = np.random.RandomState(SEED).choice(len(dataset), size=500, replace=False)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)


def save_model_checkpoint(path: Path, source_checkpoint: Path, model: torch.nn.Module, step: int) -> None:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    payload.update({
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "role": "posthoc_uda_milestone", "uda_step": int(step), "formal_endpoint": step == 2500,
        "target_test_opened_during_training": False, "source_checkpoint_sha256": SOURCE_SHA256,
    })
    torch.save(payload, path)


def run_posthoc(args) -> None:
    checkpoint = Path(args.source_checkpoint).resolve(); data_root = Path(args.data_root).resolve()
    if sha256_file(checkpoint) != SOURCE_SHA256:
        raise RuntimeError("FAIL_CLOSED source checkpoint SHA-256 mismatch")
    output = Path(args.output_root).resolve() / "P0I_POSTHOC_UDA2500_S42"
    output.mkdir(parents=True, exist_ok=False)
    model, _config, _payload = load_checkpoint_model(str(checkpoint), torch.device(args.device), BATCH_SIZE)
    def milestone(step: int, adapted: torch.nn.Module) -> None:
        if step in MILESTONES:
            save_model_checkpoint(output / f"step_{step:04d}.pth", checkpoint, adapted, step)
    adapted, diagnostics, seconds = run_frozen_u1(
        model, source_loader(data_root), feature_only_loader(data_root / "client_5", shuffle=True),
        torch.device(args.device), num_steps=2500, seed=SEED, milestone_callback=milestone,
    )
    del adapted
    write_csv(output / "posthoc_uda2500_diagnostics.csv", diagnostics)
    windows = [(1,100),(101,500),(501,1000),(1001,1500),(1501,2000),(2001,2500)]
    summaries = []
    for lo, hi in windows:
        rows = [row for row in diagnostics if lo <= row["step"] <= hi]
        for metric in ("source_ce","coral_loss","global_mmd2","adversarial_loss","total_loss"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summaries.append({"step_start":lo,"step_end":hi,"metric":metric,"mean":values.mean(),"median":np.median(values),"start":values[0],"end":values[-1]})
    write_csv(output / "posthoc_uda2500_summary.csv", summaries)
    manifest = {
        "schema_version":"iotj.p0i.posthoc.v1","status":"training_completed_test_unopened","seed":SEED,
        "source_checkpoint":str(checkpoint),"source_checkpoint_sha256":SOURCE_SHA256,"uda_steps":2500,
        "uda_lr":UDA_LR,"formal_endpoint_step":2500,"milestones":sorted(MILESTONES),
        "target_calibration_api":"x_only","target_labels_loaded":False,"target_test_opened_during_training":False,
        "objective":{"source_ce":1.0,"unconditional_coral":0.5,"global_mmd2":0.5,"unconditional_wasserstein_adversarial":0.5},
        "uda_wall_seconds":seconds,"model_selection":False,"early_stopping":False,"hyperparameter_search":False,
    }
    (output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


def server_argv(remote_output: str, server_data: str) -> list[str]:
    return [
        "/root/gaps_env/bin/python","-m","gaps_flower.server_app","--server-address","0.0.0.0:8080",
        "--rounds",str(ROUNDS),"--min-clients","2","--seed",str(SEED),"--run-name","P0I-INTERLEAVED-UDA25X100-S42",
        "--output-dir",remote_output,"--save-history","true","--strategy","p0i_interleaved","--profile","ce_only",
        "--p0i-source-calibration-dirs",f"{server_data}/client_1,{server_data}/client_2",
        "--p0i-target-calibration-dir",f"{server_data}/client_5","--p0i-uda-steps-per-round","100","--p0i-uda-device","cpu",
    ]


def client_argv(python: str, client_id: int, data_root: str) -> list[str]:
    return [python,"-m","gaps_flower.client_app","--server-address","127.0.0.1:18080","--client-id",str(client_id),
            "--data-root",data_root,"--device","cpu","--local-epochs","1","--batch-size","32","--profile","ce_only",
            "--seed","42","--proximal-mu","0"]


def run_interleaved(args) -> None:
    archive = Path(args.source_archive).resolve()
    if not archive.is_file(): raise FileNotFoundError(archive)
    archive_hash = sha256_file(archive); short_hash = archive_hash[:12]
    output = Path(args.output_root).resolve() / "P0I_INTERLEAVED_UDA25X100_S42"
    output.mkdir(parents=True, exist_ok=False)
    runtimes = {"ecs":f"/root/GAPS/p0i_runtime/{short_hash}","pi":f"/home/gaps/GAPS/p0i_runtime/{short_hash}","c2":f"/root/GAPS/p0i_runtime/{short_hash}"}
    remote_output = f"/root/GAPS/p0i_runs/P0I_INTERLEAVED_UDA25X100_S42_{short_hash}"
    hosts = [args.ecs_host,args.pi_host,args.c2_host]; ensure_idle(hosts)
    if ssh(args.ecs_host, f"if test -e {shlex.quote(remote_output)}; then echo EXISTS; fi").strip():
        raise FileExistsError(remote_output)
    for host, runtime in ((args.ecs_host,runtimes["ecs"]),(args.pi_host,runtimes["pi"]),(args.c2_host,runtimes["c2"])):
        deploy_archive(host, archive, runtime)
    server_data = "/root/GAPS/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    pi_data = "/home/gaps/GAPS/flower_runtime/dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    c2_data = "/root/GAPS/confirmation_c2_data/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    for host,path,clients in ((args.ecs_host,server_data,"1 2 5"),(args.pi_host,pi_data,"1"),(args.c2_host,c2_data,"2")):
        ssh(host, f"for c in {clients}; do test -d {shlex.quote(path)}/client_$c || exit 17; done")
    commands = {"server":server_argv(remote_output,server_data),"client_c1":client_argv("/home/gaps/GAPS/gaps_rpi_env/bin/python",1,pi_data),"client_c2":client_argv("/root/gaps_c2_cpu_env/bin/python",2,c2_data)}
    (output / "locked_commands.json").write_text(json.dumps(commands,indent=2)+"\n",encoding="utf-8")
    processes=[]; tunnels=[]; handles=[]; started=time.perf_counter()
    try:
        for label in ("server","client_c1","client_c2"):
            handles.extend([(output/f"{label}.stdout.log").open("w",encoding="utf-8"),(output/f"{label}.stderr.log").open("w",encoding="utf-8")])
        processes.append(subprocess.Popen(process_command(args.ecs_host,runtimes["ecs"],commands["server"]),stdout=handles[0],stderr=handles[1],text=True))
        time.sleep(5)
        if processes[0].poll() is not None: raise RuntimeError("server exited before clients")
        tunnels=list(_start_ecs_c2_tunnels(args.ecs_host,args.pi_host,args.c2_host))
        processes.append(subprocess.Popen(process_command(args.pi_host,runtimes["pi"],commands["client_c1"]),stdout=handles[2],stderr=handles[3],text=True))
        processes.append(subprocess.Popen(process_command(args.c2_host,runtimes["c2"],commands["client_c2"]),stdout=handles[4],stderr=handles[5],text=True))
        deadline=time.monotonic()+args.timeout_hours*3600
        while any(p.poll() is None for p in processes):
            if time.monotonic()>deadline: raise TimeoutError("P0-I3 exceeded timeout")
            if any(p.poll() not in (None,0) for p in processes): raise RuntimeError(f"process failure: {[p.poll() for p in processes]}")
            time.sleep(10)
        if any(p.returncode != 0 for p in processes): raise RuntimeError(f"non-zero process codes: {[p.returncode for p in processes]}")
    finally:
        _terminate_processes(processes); _terminate_processes(tunnels)
        for handle in handles: handle.close()
    remote_copy=output/"remote_server"; run(["scp","-r",f"{args.ecs_host}:{remote_output}",str(remote_copy)],timeout=600)
    for round_id in range(1,26):
        for role in ("pre_uda","post_uda"):
            if not (remote_copy/f"server_round_{round_id:03d}_{role}.pth").is_file(): raise RuntimeError(f"FAIL_CLOSED missing round {round_id} {role}")
    lineage=json.loads((remote_copy/"interleaved_lineage.json").read_text(encoding="utf-8"))
    if len(lineage)!=25 or not all(row["parent_match"] for row in lineage): raise RuntimeError("FAIL_CLOSED lineage audit")
    manifest={"schema_version":"iotj.p0i.interleaved.v1","status":"training_completed_test_unopened","seed":SEED,
              "rounds":25,"local_epochs":1,"batch_size":32,"client_lr":CLIENT_LR,"total_uda_steps":2500,
              "uda_steps_per_round":100,"uda_lr":UDA_LR,"adapted_as_global":True,"target_calibration_api":"x_only",
              "target_labels_loaded":False,"target_test_opened_during_training":False,"formal_endpoint":"round25_post_uda",
              "source_archive_sha256":archive_hash,"total_wall_seconds":time.perf_counter()-started,"lineage_status":"PASS",
              "objective":{"source_ce":1.0,"unconditional_coral":0.5,"global_mmd2":0.5,"unconditional_wasserstein_adversarial":0.5},
              "model_selection":False,"early_stopping":False,"hyperparameter_search":False}
    (output/"protocol_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,sort_keys=True))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage",choices=("posthoc-uda","interleaved"),required=True)
    parser.add_argument("--source-checkpoint"); parser.add_argument("--source-archive"); parser.add_argument("--data-root",default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--output-root",default="results/iotj_p0_adaptation_timing_20260803"); parser.add_argument("--device",default="cpu")
    parser.add_argument("--ecs-host",default="root@121.40.139.213"); parser.add_argument("--pi-host",default="gaps@192.168.137.172"); parser.add_argument("--c2-host",default="root@114.55.171.63"); parser.add_argument("--timeout-hours",type=float,default=4.0)
    args=parser.parse_args()
    if args.stage=="posthoc-uda":
        if not args.source_checkpoint: parser.error("posthoc-uda requires --source-checkpoint")
        run_posthoc(args)
    else:
        if not args.source_archive: parser.error("interleaved requires --source-archive")
        run_interleaved(args)


if __name__ == "__main__": main()
