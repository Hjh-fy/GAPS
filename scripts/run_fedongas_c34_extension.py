"""Single sequential controller: C3/C4 TO, FT, CLASS; no extra algorithms."""
from __future__ import annotations
import argparse,csv,json,os,shlex,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from gaps_flower.fedongas_extension import *
from scripts.run_fedongas_regression_cross import write_json,read_json,sha,assert_hashes
from scripts import run_iotj_final_classification_le1 as frozen
from scripts.run_fedongas27 import validate_target_updates,validate_da_audit


def progress(status,completed,current=None,**extra):
    write_json(CLASS_STUDY/'RUN_PROGRESS.json',{'status':status,'completed':completed,'completed_count':len(completed),
        'total':6,'current_run':current,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'target_test_selection':False,**extra})


def ensure_idle(hosts):
    for host in hosts:
        text=frozen._ssh(host,"pgrep -af '[p]ython.*[g]aps_flower[.](server_app|client_app|fedongas27_entry|fedongas27_target|fedongas_extension_target)' || true")
        if text.strip():raise RuntimeError('Active training process; preserve and stop: '+host+' '+text)


def code_archive(commit):
    path=CLASS_STUDY/'inputs'/f'code_{commit[:12]}.tar'
    if not path.exists():frozen._run(['git','archive','--format=tar','-o',str(path),commit,'--','*.py'],timeout=300)
    return path,sha(path)


def deploy():
    info=read_json(CLASS_STUDY/'PREFLIGHT.json');archive=CLASS_STUDY/'inputs/data.tar'
    if sha(archive)!=info['data_archive_sha256']:raise RuntimeError('Input archive changed')
    for host,base in zip(HOSTS,(REMOTE_DATA,PI_DATA,REMOTE_DATA)):
        marker=base+'/.data_archive_sha256'
        observed=frozen._ssh(host,f'if test -f {marker}; then cat {marker}; elif test -e {base}; then echo PARTIAL; fi').strip()
        if observed:
            if observed!=info['data_archive_sha256']:raise RuntimeError('Unmatched remote data archive; preserve')
        else:
            remote='/tmp/fong_c34_'+info['data_archive_sha256']+'.tar'
            frozen._run(['scp','-p',str(archive),host+':'+remote],timeout=600)
            frozen._ssh(host,f'echo "{info["data_archive_sha256"]}  {remote}" | sha256sum -c - && mkdir {base} && tar -xf {remote} -C {base} && echo {info["data_archive_sha256"]} > {marker}',timeout=600)
        py='/home/gaps/GAPS/gaps_rpi_env/bin/python' if host==HOSTS[1] else '/root/gaps_env/bin/python' if host==HOSTS[0] else '/root/gaps_c2_cpu_env/bin/python'
        check=f"import pathlib,json,hashlib; p=pathlib.Path({base!r}); d=json.loads((p/'DATA_SHA256.json').read_text()); assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in d.items()); print('INPUT_HASH_PASS',len(d))"
        print(host,frozen._ssh(host,py+' -c '+shlex.quote(check)),flush=True)


def execute_target(row,commit):
    ensure_idle([HOSTS[0]]);runid=row['experiment_id'];directory=CLASS_STUDY/'runs'/runid
    remote='/root/GAPS/iotj_final_classification_le1_20260804/'+runid
    if frozen._ssh(HOSTS[0],f'if test -e {remote}; then echo EXISTS; fi').strip():raise RuntimeError('Preserve existing remote endpoint: '+remote)
    archive,digest=code_archive(commit);runtime='/root/GAPS/iotj_final_runtime/'+digest[:12]
    frozen._deploy_archive(HOSTS[0],archive,runtime)
    name='source-S42.pth' if row['method']=='FT' else 'random-S42.pth'
    initial_sha=sha(CLASS_STUDY/'inputs/data/initial'/name)
    argv=['/root/gaps_env/bin/python','-u','-m','gaps_flower.fedongas_extension_target','--target',row['target'],
        '--method',row['method'],'--data',REMOTE_DATA+'/target_xy/client_'+row['target'][1:],
        '--initial-state',REMOTE_DATA+'/initial/'+name,'--initial-sha256',initial_sha,'--output',remote]
    directory.mkdir(parents=True)
    write_json(directory/'locked_run_spec.json',{'run':row,'command':argv,'freeze_commit':commit})
    with (directory/'server.stdout.log').open('w') as stdout,(directory/'server.stderr.log').open('w') as stderr:
        subprocess.run(frozen._process_command(HOSTS[0],runtime,argv),cwd=ROOT,stdout=stdout,stderr=stderr,check=True,timeout=43200)
    frozen._run(['scp','-r',HOSTS[0]+':'+remote,str(directory/'remote_server')],timeout=1200)


def accept(row,commit,wall):
    import torch
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
    d=CLASS_STUDY/'runs'/row['experiment_id'];r=d/'remote_server';info=read_json(CLASS_STUDY/'PREFLIGHT.json')
    initial=info['source_state_fingerprint'] if row['method']=='FT' else info['common_initial_fingerprint']
    cp=r/('server_round_025_adapted.pth' if row['method']=='CLASS' else 'update_2500.pth')
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    if row['method']=='CLASS':
        if payload.get('round')!=25:raise RuntimeError('CLASS endpoint not round25')
        for role in ('server','client_c1','client_c2'):
            logs=(d/f'{role}.stdout.log').read_text(encoding='utf-8',errors='replace')
            events=[json.loads(l) for l in logs.splitlines() if l.startswith('{') and 'COMMON_RANDOM_INITIALIZATION' in l]
            if not events or any(e['ordered_fingerprint']!=initial for e in events):raise RuntimeError('Common initial state differs')
        for i in range(1,26):validate_da_audit(read_json(r/f'fong_loss_audit_round_{i:03d}.json'),'CLASS',42)
    else:
        with (r/'update_diagnostics.csv').open() as f:validate_target_updates(list(csv.DictReader(f)))
        cfg=read_json(r/'effective_config.json');done=read_json(r/'COMPLETE.json')
        if payload.get('updates')!=2500 or payload.get('target')!=row['target'] or payload.get('method')!=row['method'] or payload.get('seed')!=42:
            raise RuntimeError('Target endpoint identity differs')
        if done['initial_fingerprint']!=initial or done['test_opened']:raise RuntimeError('Initial fingerprint/test access differs')
        name='source-S42.pth' if row['method']=='FT' else 'random-S42.pth'
        if cfg['initial_sha256']!=sha(CLASS_STUDY/'inputs/data/initial'/name):raise RuntimeError('TO/FT initialization asset differs')
    write_json(d/'FINAL_ENDPOINT.json',{'run':row,'freeze_commit':commit,'checkpoint':cp.relative_to(d).as_posix(),
        'checkpoint_sha256':sha(cp),'state_fingerprint':ordered_state_content_fingerprint(payload['model_state']),
        'initial_fingerprint':initial,'wall_seconds':wall,'target_updates':2500,'target_test_opened':False,'status':'PASS'})


def run(commit):
    import msvcrt
    guard=(CLASS_STUDY/'controller.lock').open('a+b');guard.write(b'0');guard.flush();guard.seek(0)
    try:msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another controller holds this study lock')
    spec=read_json(CLASS_STUDY/'protocol_manifest.json');info=spec['inputs'];assert_hashes(info['input_hashes'])
    if spec['classification_matrix']!=classification_matrix():raise RuntimeError('Run matrix differs')
    archive,digest=code_archive(commit)
    freeze={'freeze_commit':commit,'code_archive_sha256':digest,'protocol_sha256':sha(CLASS_STUDY/'protocol_manifest.json'),
        'data_archive_sha256':info['data_archive_sha256'],'input_hashes':info['input_hashes']}
    freeze_path=CLASS_STUDY/'PRE_RUN_FREEZE.json'
    if freeze_path.exists() and read_json(freeze_path)!=freeze:raise RuntimeError('Frozen protocol differs')
    if not freeze_path.exists():write_json(freeze_path,freeze)
    # Build commands before replacing the historical global builder (fresh-process regression).
    commands={r['experiment_id']:build_class_commands(r) for r in classification_matrix() if r['method']=='CLASS'}
    frozen.RESULT_ROOT=CLASS_STUDY/'runs';frozen._source_archive=code_archive;frozen._ensure_idle=ensure_idle
    frozen.build_flower_commands=lambda runid:commands[runid]
    completed=[];current=None
    try:
        for row in classification_matrix():
            d=CLASS_STUDY/'runs'/row['experiment_id']
            if (d/'FINAL_ENDPOINT.json').exists():match_endpoint(d,row,commit);completed.append(row['experiment_id'])
            elif d.exists():raise RuntimeError('Preserve incomplete local endpoint: '+str(d))
        if len(completed)<6:
            ensure_idle(HOSTS);progress('DEPLOYING',completed);deploy()
        for row in classification_matrix():
            if row['experiment_id'] in completed:continue
            assert_hashes(info['input_hashes']);current=row['experiment_id'];progress('RUNNING',completed,current)
            print(datetime.now(timezone.utc).isoformat(),'START',current,flush=True);start=time.perf_counter()
            if row['method']=='CLASS':
                frozen.execute_full_fl(current,protocol_hash=freeze['protocol_sha256'],lock_payload={'freeze_commit':commit},
                    ecs_host=HOSTS[0],pi_host=HOSTS[1],c2_host=HOSTS[2],timeout_hours=12.)
            else:execute_target(row,commit)
            accept(row,commit,time.perf_counter()-start);completed.append(current);progress('RUNNING',completed,current)
            print('COMPLETE',current,flush=True)
        progress('EVALUATING',completed)
        subprocess.run([sys.executable,'scripts/evaluate_fedongas_extension.py'],cwd=ROOT,check=True)
        progress('COMPLETE_EVALUATED',completed)
    except BaseException as exc:
        progress('FAILED_CLOSED',completed,current,error=repr(exc))
        write_json(CLASS_STUDY/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'),{'run':current,'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:guard.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-commit',required=True);run(p.parse_args().freeze_commit)
