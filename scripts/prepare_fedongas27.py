"""Prepare registered identities and isolated input views, never resample budgets."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from gaps_flower.fedongas27_protocol import DATA, BUDGET, STUDY, DOC, run_matrix, validate_budget_rows, initial_model
from gaps_flower.state_fingerprint import ordered_state_content_fingerprint


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty result table')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect_inputs():
    from tools.verify_iotj_canonical_v1_hashes import verify
    dataset = verify(DATA)
    if dataset['status'] != 'PASS':
        raise RuntimeError('Canonical dataset hash failed')
    expected = json.loads((BUDGET/'c5_calibration_budget_manifest_sha256.json').read_text())['files']
    for rel, value in expected.items():
        if sha(BUDGET/rel) != value:
            raise RuntimeError(f'Frozen budget hash changed: {rel}')
    with (BUDGET/'c5_calibration_budget_05pct.csv').open(encoding='utf-8') as f:
        small = list(csv.DictReader(f))
    with (BUDGET/'c5_calibration_budget_20pct.csv').open(encoding='utf-8') as f:
        large = list(csv.DictReader(f))
    metadata = json.loads((DATA/'client_5/test_experiment_info.json').read_text())
    check = validate_budget_rows(small, large, {r['physical_identity'] for r in metadata})
    if len(metadata) != 1360 or len({r['physical_identity'] for r in metadata}) != 1360:
        raise RuntimeError('Frozen test identities differ')
    hashes = {str((DATA/n).relative_to(ROOT)): sha(DATA/n) for n in ('dataset_sha256.json','canonical_preprocessing_manifest.json')}
    for n in ('test_features.npy','test_classification_labels.npy','test_experiment_info.json'):
        hashes[f'dataset/iotj_canonical_v1/client_5/{n}'] = sha(DATA/'client_5'/n)
    phases = set()
    for budget, rows in ((5,small),(20,large)):
        directory = BUDGET/f'budget_data/client_5_budget_{budget:02d}'
        info = json.loads((directory/'calibration_experiment_info.json').read_text())
        if [r['physical_identity'] for r in info] != [r['canonical_identity'] for r in rows]:
            raise RuntimeError('CSV and budget arrays row identity differ')
        x = np.load(directory/'calibration_features.npy', allow_pickle=False)
        y = np.load(directory/'calibration_classification_labels.npy', allow_pickle=False)
        p = np.load(directory/'calibration_phase_labels.npy', allow_pickle=False)
        if x.shape != (len(rows),50,8) or not np.all(np.isfinite(x)):
            raise RuntimeError('Canonical calibration shape/finite gate failed')
        if list(y) != [int(r['class_id']) for r in rows]:
            raise RuntimeError('Budget class/identity order differs')
        phases.update(np.unique(p).tolist())
        for name in ('features','classification_labels','phase_labels'):
            f = directory/f'calibration_{name}.npy'
            hashes[str(f.relative_to(ROOT))] = sha(f)
        f = BUDGET/f'c5_calibration_budget_{budget:02d}pct.csv'
        hashes[str(f.relative_to(ROOT))] = sha(f)
    if phases != {2}:
        raise RuntimeError(f'C5 phases no longer late: {phases}')
    return {'status':'PASS', **check, 'test_n':1360, 'target_phases':sorted(phases),
            'dataset':dataset, 'input_hashes':hashes}


def copy_source_view(source, destination, *, class_only):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for split in ('train','calibration','test'):
        for name in ('features','classification_labels'):
            shutil.copy2(source/f'{split}_{name}.npy', destination/f'{split}_{name}.npy')
        count = len(np.load(source/f'{split}_classification_labels.npy', allow_pickle=False))
        if class_only:
            np.save(destination/f'{split}_phase_labels.npy', np.zeros(count,np.int64))
        else:
            shutil.copy2(source/f'{split}_phase_labels.npy', destination/f'{split}_phase_labels.npy')
        np.save(destination/f'{split}_regression_labels.npy', np.zeros((count,4),np.float32))


def prepare():
    if STUDY.exists():
        raise FileExistsError(f'Preserve existing study: {STUDY}')
    check = inspect_inputs()
    STUDY.mkdir(parents=True)
    initial = STUDY/'inputs/data/initial'
    initial.mkdir(parents=True)
    fingerprints = {}
    for seed in (42,43,44):
        model = initial_model(seed)
        fp = ordered_state_content_fingerprint(model.state_dict())
        fingerprints[str(seed)] = fp
        torch.save({'model_state':model.state_dict(), 'seed':seed, 'initialization':'random',
                    'ordered_fingerprint':fp}, initial/f'random-S{seed}.pth')
    root = STUDY/'inputs/data'
    for scope in ('full','class'):
        for client in (1,2):
            copy_source_view(DATA/f'client_{client}', root/scope/f'client_{client}', class_only=scope=='class')
        for budget in (5,20):
            src = BUDGET/f'budget_data/client_5_budget_{budget:02d}'
            dest = root/scope/f'budget_{budget:02d}'
            dest.mkdir(parents=True)
            count = 80 if budget==5 else 320
            for name in ('features','classification_labels'):
                shutil.copy2(src/f'calibration_{name}.npy', dest/f'calibration_{name}.npy')
            if scope=='class':
                np.save(dest/'calibration_phase_labels.npy', np.zeros(count,np.int64))
            else:
                shutil.copy2(src/'calibration_phase_labels.npy', dest/'calibration_phase_labels.npy')
            # Compatibility placeholder; not a concentration feature/label.
            np.save(dest/'calibration_regression_labels.npy', np.zeros((count,4),np.float32))
    for budget in (5,20):
        directory=root/'target_xy'/f'budget_{budget:02d}'
        directory.mkdir(parents=True)
        src=BUDGET/f'budget_data/client_5_budget_{budget:02d}'
        for name in ('features','classification_labels'):
            shutil.copy2(src/f'calibration_{name}.npy', directory/f'calibration_{name}.npy')
    data_hashes={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'DATA_SHA256.json',data_hashes)
    archive=STUDY/'inputs/data.tar'
    with tarfile.open(archive,'w') as tar:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                tar.add(p,arcname=p.relative_to(root).as_posix())
    check.update(initial_fingerprints=fingerprints,data_archive_sha256=sha(archive),data_hashes=data_hashes)
    write_json(STUDY/'PREFLIGHT.json',check)
    write_json(STUDY/'protocol_manifest.json',{
        'study_id':'FEDONGAS-C5-BUDGET-MULTISEED-20261014','seeds':[42,43,44],
        'budgets':[5,20],'matrix':run_matrix(),'preflight':check,
        'target_updates':2500,'blocks':25,'steps':100,'lr':5e-4,'batch_size':32,
        'client_local_epochs':1,'source_rounds':25,'prototype_ema':.8,
        'full_stage_weight':.2,'class_stage_weight':0.,'prototype_mmd_weight':.2,
        'classification_concentration_access':False,'target_test_selection':False,
        'evaluation_release':'all_27_fixed_endpoints','sample_std_ddof':1,
        'scope':'existing_window_level_protocol; C5_late; no_significance_claim'})
    rows=[]
    for r in run_matrix():
        rows.append({**r,'source_clients':'C1;C2' if r['method']!='TO' else '',
                     'target_clients':'C5' if r['method']!='SRC' else '',
                     'initial_fingerprint':fingerprints[str(r['seed'])] if r['method']!='FT' else 'from_matching_SRC',
                     'status':'registered','result_path':f'results/fedongas_c5_budget_multiseed_20261014/runs/{r["experiment_id"]}',
                     'config_path':'results/fedongas_c5_budget_multiseed_20261014/protocol_manifest.json',
                     'split_protocol':'frozen_C5_80_or_320_calibration__1360_test',
                     'model':'original_TCN_Attention_50x8_4class','QC':'none','metrics':'accuracy;macro_f1;per_class_f1;recall;confusion'})
    write_csv(STUDY/'EXPERIMENT_MATRIX.csv',rows)
    write_csv(DOC/'EXPERIMENT_MATRIX.csv',rows)
    write_json(STUDY/'RUN_PROGRESS.json',{'status':'PREPARED_NOT_TRAINING','total':27,'completed':[]})
    print(json.dumps({'status':'PASS','study':str(STUDY),'tasks':27,'initial_fingerprints':fingerprints},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('inspect','prepare'))
    args=parser.parse_args()
    print(json.dumps(inspect_inputs(),indent=2)) if args.command=='inspect' else prepare()
