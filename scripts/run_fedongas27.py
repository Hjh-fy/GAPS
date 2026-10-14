"""Sequential, durable controller for the authorized 27 endpoints; no search."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from gaps_flower.fedongas27_protocol import STUDY, REMOTE_DATA, PI_DATA, run_matrix, build_commands
from scripts.prepare_fedongas27 import sha, write_json
from scripts import run_iotj_final_classification_le1 as frozen

HOSTS=('root@121.40.139.213','gaps@192.168.137.172','root@114.55.171.63')
REMOTE_PARENT='/root/GAPS/iotj_final_classification_le1_20260804'


def pending_runs(study,rows,verify):
    pending=[]
    for row in rows:
        directory=Path(study)/'runs'/row['experiment_id']
        if (directory/'FINAL_ENDPOINT.json').is_file(): verify(row)
        elif directory.exists(): raise RuntimeError(f'partial endpoint preserved: {directory}')
        else: pending.append(row)
    return pending


def install_fl_builder(rows):
    # Materialize while the historical modules still capture their original builder.
    # Replacing the global before their lazy import creates recursive dispatch.
    commands={r['experiment_id']:build_commands(r) for r in rows if r['method'] in ('SRC','CLASS','FULL')}
    frozen.build_flower_commands=lambda experiment_id: commands[experiment_id]


def validate_target_updates(rows):
    if len(rows)!=2500: raise RuntimeError('Expected exactly 2500 target updates')
    for i,r in enumerate(rows):
        if int(r['update'])!=i+1 or int(r['block'])!=i//100+1 or int(r['step'])!=i%100+1:
            raise RuntimeError('Target update order differs')
        if str(r['source_ce_enabled']).lower()!='false' or r['loss_name']!='target_ce':
            raise RuntimeError('Target-only source loss contamination')
        if i%100==0 and int(r['optimizer_state_before_first_update'])!=0:
            raise RuntimeError('Adam was not reset at block boundary')
        for k in ('encoder_grad_norm','classifier_grad_norm','raw_loss'):
            if not math.isfinite(float(r[k])): raise RuntimeError('Nonfinite target diagnostics')


def validate_da_audit(value,method,seed):
    if value.get('seed')!=seed or value.get('prototype_scope')!=method or value.get('steps')!=100:
        raise RuntimeError('DA method/seed/steps mismatch')
    if int(value.get('changed_tensors') or 0)<=0: raise RuntimeError('DA did not update model')
    count=value.get('prototype_count',0)
    if (method=='CLASS' and count!=4) or (method=='FULL' and not 4<=count<=12):
        raise RuntimeError('Wrong prototype count')
    if value.get('true_phase_used')!=(method=='FULL'): raise RuntimeError('True-phase access differs')
    losses={r['loss_name']:r for r in value['loss_activity']}
    if losses.get('source_ce',{}).get('active_steps')!=100: raise RuntimeError('source CE inactive')
    if losses.get('target_ce',{}).get('active_steps',0)!=0: raise RuntimeError('target CE unexpectedly active')
    if method=='CLASS' and losses.get('stage_mmd',{}).get('active_steps',0)!=0:
        raise RuntimeError('CLASS stage loss unexpectedly active')


def code_archive(commit):
    path=STUDY/'inputs'/f'code_{commit[:12]}.tar'
    if not path.exists(): frozen._run(['git','archive','--format=tar','-o',str(path),commit,'--','*.py'],timeout=300)
    return path,sha(path)


def ensure_idle(hosts):
    for host in hosts:
        out=frozen._ssh(host,"pgrep -af '[p]ython.*[g]aps_flower[.](server_app|client_app|fedongas27_entry|fedongas27_target)' || true")
        if out.strip(): raise RuntimeError(f'Residual training process on {host}: {out}')


def deploy_inputs():
    archive=STUDY/'inputs/data.tar'; digest=sha(archive)
    expected=json.loads((STUDY/'PREFLIGHT.json').read_text())['data_archive_sha256']
    if digest!=expected: raise RuntimeError('Input archive changed')
    for host,directory in zip(HOSTS,(REMOTE_DATA,PI_DATA,REMOTE_DATA)):
        marker=directory+'/.data_archive_sha256'
        observed=frozen._ssh(host,f'if test -f {marker}; then cat {marker}; elif test -e {directory}; then echo PARTIAL; fi').strip()
        if observed:
            if observed!=digest: raise RuntimeError(f'Preserve mismatched/partial input deployment on {host}')
        else:
            remote=f'/tmp/fong27_data_{digest}.tar'
            frozen._run(['scp','-p',str(archive),f'{host}:{remote}'],timeout=600)
            frozen._ssh(host,f'echo "{digest}  {remote}" | sha256sum -c - && mkdir {directory} && tar -xf {remote} -C {directory} && printf "%s\\n" {digest} > {marker}',timeout=600)
        py='/home/gaps/GAPS/gaps_rpi_env/bin/python' if host==HOSTS[1] else '/root/gaps_env/bin/python' if host==HOSTS[0] else '/root/gaps_c2_cpu_env/bin/python'
        check="import pathlib,json,hashlib; p=pathlib.Path("+repr(directory)+"); d=json.loads((p/'DATA_SHA256.json').read_text()); assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in d.items()); print('INPUT_HASH_PASS',len(d))"
        print(host,frozen._ssh(host,f'{py} -c {shlex.quote(check)}').strip(),flush=True)


def progress(status,completed,current=None,**extra):
    write_json(STUDY/'RUN_PROGRESS.json',{'status':status,'total':27,'completed':completed,
        'completed_count':len(completed),'current_run':current,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'target_test_used_for_selection':False,**extra})


def verify_marker(row,commit):
    from scripts.evaluate_fedongas27 import completion_gate
    return completion_gate(STUDY,[row],commit)[0][2]


def accept_endpoint(row,commit,wall):
    import torch
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
    directory=STUDY/'runs'/row['experiment_id']; remote=directory/'remote_server'
    target=row['method'] in ('TO','FT')
    name='update_2500.pth' if target else ('server_round_025.pth' if row['method']=='SRC' else 'server_round_025_adapted.pth')
    cp=remote/name; payload=torch.load(cp,map_location='cpu',weights_only=False)
    preflight=json.loads((STUDY/'PREFLIGHT.json').read_text())
    initial=preflight['initial_fingerprints'][str(row['seed'])]
    if target:
        with (remote/'update_diagnostics.csv').open() as f: validate_target_updates(list(csv.DictReader(f)))
        complete=json.loads((remote/'COMPLETE.json').read_text())
        if payload['updates']!=2500 or payload['method']!=row['method'] or payload['seed']!=row['seed']:
            raise RuntimeError('Target checkpoint endpoint differs')
        if row['method']=='FT':
            source_row=next(r for r in run_matrix() if r['experiment_id']==row['initialization_run'])
            source=verify_marker(source_row,commit)
            initial=source['state_fingerprint']
            cfg=json.loads((remote/'effective_config.json').read_text())
            if cfg['initial_checkpoint_sha256']!=source['checkpoint_sha256']: raise RuntimeError('FT source hash differs')
        if complete['initial_fingerprint']!=initial or complete['test_opened']:
            raise RuntimeError('Target initialization/test access differs')
    else:
        if payload['round']!=25: raise RuntimeError('Not round25')
        for role in ('server','client_c1','client_c2'):
            logs=(directory/f'{role}.stdout.log').read_text(encoding='utf-8',errors='replace')
            events=[json.loads(line) for line in logs.splitlines() if line.startswith('{') and 'COMMON_RANDOM_INITIALIZATION' in line]
            if not events or any(e['ordered_fingerprint']!=initial or e['seed']!=row['seed'] for e in events):
                raise RuntimeError(f'Common initialization not verified: {role}')
        if row['method']!='SRC':
            for round_id in range(1,26):
                audit=json.loads((remote/f'fong_loss_audit_round_{round_id:03d}.json').read_text())
                validate_da_audit(audit,row['method'],row['seed'])
    marker={'run':row,'freeze_commit':commit,'checkpoint':cp.relative_to(directory).as_posix(),
            'checkpoint_sha256':sha(cp),'state_fingerprint':ordered_state_content_fingerprint(payload['model_state']),
            'initial_fingerprint':initial,'wall_seconds':wall,'target_updates':row['target_updates'],
            'target_test_opened':False,'status':'PASS'}
    write_json(directory/'FINAL_ENDPOINT.json',marker)


def execute_target(row,commit):
    runid=row['experiment_id']; directory=STUDY/'runs'/runid
    ensure_idle([HOSTS[0]])
    remote=REMOTE_PARENT+'/'+runid
    if frozen._ssh(HOSTS[0],f'if test -e {remote}; then echo EXISTS; fi').strip():
        raise RuntimeError(f'Preserve existing remote endpoint {remote}')
    archive,digest=code_archive(commit); runtime='/root/GAPS/iotj_final_runtime/'+digest[:12]
    frozen._deploy_archive(HOSTS[0],archive,runtime)
    init=f'{REMOTE_DATA}/initial/random-S{row["seed"]}.pth' if row['method']=='TO' else f'{REMOTE_PARENT}/{row["initialization_run"]}/server_round_025.pth'
    argv=['/root/gaps_env/bin/python','-u','-m','gaps_flower.fedongas27_target','--method',row['method'],
          '--seed',str(row['seed']),'--data',f'{REMOTE_DATA}/target_xy/budget_{row["budget_pct"]:02d}',
          '--initial-state',init,'--output',remote]
    directory.mkdir(parents=True)
    write_json(directory/'locked_run_spec.json',{'run':row,'command':argv,'freeze_commit':commit})
    with (directory/'server.stdout.log').open('w') as stdout, (directory/'server.stderr.log').open('w') as stderr:
        subprocess.run(frozen._process_command(HOSTS[0],runtime,argv),cwd=ROOT,stdout=stdout,stderr=stderr,check=True,timeout=43200)
    frozen._run(['scp','-r',f'{HOSTS[0]}:{remote}',str(directory/'remote_server')],timeout=1200)


def verify_inputs(freeze):
    for rel,value in freeze['input_hashes'].items():
        if sha(ROOT/rel)!=value: raise RuntimeError(f'Frozen input changed: {rel}')
    if sha(STUDY/'protocol_manifest.json')!=freeze['protocol_sha256']: raise RuntimeError('Protocol changed')


def finish(commit):
    # Never release the test before every fixed endpoint has passed acceptance.
    from scripts.evaluate_fedongas27 import completion_gate
    completion_gate(STUDY,run_matrix(),commit)
    subprocess.run([sys.executable,'scripts/evaluate_fedongas27.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_fedongas27.py','tests/test_fedongas27_inputs.py',
                    'tests/test_fedongas27_runner.py','--basetemp=.tmp_fong27_postrun_20261015'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','compileall','-q','gaps_flower/fedongas27_entry.py','gaps_flower/fedongas27_methods.py',
                    'gaps_flower/fedongas27_protocol.py','gaps_flower/fedongas27_target.py','scripts/run_fedongas27.py',
                    'scripts/prepare_fedongas27.py','scripts/evaluate_fedongas27.py'],cwd=ROOT,check=True)
    stage=[]
    for p in STUDY.rglob('*'):
        if p.is_file() and 'inputs' not in p.relative_to(STUDY).parts and p.suffix in ('.json','.csv','.md'):
            stage.append(p.relative_to(ROOT).as_posix())
    index={rel:sha(ROOT/rel) for rel in stage}
    write_json(STUDY/'COMPACT_EVIDENCE_SHA256.json',index)
    stage.append((STUDY/'COMPACT_EVIDENCE_SHA256.json').relative_to(ROOT).as_posix())
    existing=frozen._run(['git','diff','--cached','--name-only']).stdout.strip()
    if existing: raise RuntimeError('Unrelated staged changes; do not commit automatically')
    for i in range(0,len(stage),50): frozen._run(['git','add','--',*stage[i:i+50]])
    frozen._run(['git','commit','-m','results: close FedOnGas C5 27-run budget and multiseed study'])
    frozen._run(['git','push','origin','HEAD:codex/iotj-final-classification-le1'],timeout=300)
    progress('COMPLETE_PUSHED',[r['experiment_id'] for r in run_matrix()],result_commit=frozen._run(['git','rev-parse','HEAD']).stdout.strip())


def run(commit):
    freeze_path=STUDY/'PRE_RUN_FREEZE.json'
    if frozen._run(['git','rev-parse',commit]).stdout.strip()!=commit: raise RuntimeError('Use full freeze commit')
    archive,digest=code_archive(commit)
    if freeze_path.exists():
        freeze=json.loads(freeze_path.read_text())
        if freeze['freeze_commit']!=commit or freeze['code_archive_sha256']!=digest: raise RuntimeError('Freeze differs')
    else:
        preflight=json.loads((STUDY/'PREFLIGHT.json').read_text())
        freeze={'freeze_commit':commit,'code_archive_sha256':digest,'protocol_sha256':sha(STUDY/'protocol_manifest.json'),
                'input_hashes':preflight['input_hashes'],'data_archive_sha256':preflight['data_archive_sha256']}
        write_json(freeze_path,freeze)
    verify_inputs(freeze)
    rows=run_matrix(); completed=[]; current=None
    try:
        pending=pending_runs(STUDY,rows,lambda r:verify_marker(r,commit))
        completed=[r['experiment_id'] for r in rows if r not in pending]
        progress('DEPLOYING',completed)
        ensure_idle(list(HOSTS)); deploy_inputs()
        frozen.RESULT_ROOT=STUDY/'runs'
        frozen._source_archive=code_archive
        frozen._ensure_idle=ensure_idle
        install_fl_builder(rows)
        for row in pending:
            verify_inputs(freeze)
            current=row['experiment_id']; progress('RUNNING',completed,current)
            print(datetime.now(timezone.utc).isoformat(),'START',current,flush=True)
            started=time.perf_counter()
            if row['method'] in ('TO','FT'): execute_target(row,commit)
            else:
                frozen.execute_full_fl(current,protocol_hash=freeze['protocol_sha256'],lock_payload={'freeze_commit':commit},
                    ecs_host=HOSTS[0],pi_host=HOSTS[1],c2_host=HOSTS[2],timeout_hours=12.)
            accept_endpoint(row,commit,time.perf_counter()-started)
            completed.append(current); progress('RUNNING',completed,current)
            print('COMPLETE',current,flush=True)
        current=None; progress('EVALUATING',completed)
        finish(commit)
    except BaseException as exc:
        progress('FAILED_CLOSED',completed,current,error=repr(exc))
        with (STUDY/'FAILURES.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'run':current,'error':repr(exc),'traceback':traceback.format_exc()})+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('run',))
    parser.add_argument('--freeze-commit',required=True)
    args=parser.parse_args(); run(args.freeze_commit)
