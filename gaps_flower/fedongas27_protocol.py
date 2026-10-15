"""Registered 27-run matrix, shared initialization and frozen command surfaces."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/fedongas_c5_budget_multiseed_20261014'
DOC = ROOT / 'docs/experiments/fedongas_c5_budget_multiseed_20261014'
DATA = ROOT / 'dataset/iotj_canonical_v1'
BUDGET = ROOT / 'results/iotj_canonical_v1_c5_budget_20260810'
REMOTE_DATA = '/root/GAPS/fedongas27_inputs_20261014'
PI_DATA = '/home/gaps/GAPS/fedongas27_inputs_20261014'
SEEDS = (42, 43, 44)
BUDGETS = (5, 20)
METHODS = ('TO', 'FT', 'CLASS', 'FULL')


def run_matrix():
    rows = []
    for seed in SEEDS:
        rows.append({'experiment_id': f'FONG-SRC-S{seed}', 'method': 'SRC', 'seed': seed,
                     'budget_pct': 0, 'calibration_n': 0, 'initialization_run': f'random-S{seed}',
                     'rounds': 25, 'target_updates': 0})
    for method in METHODS:
        for budget in BUDGETS:
            for seed in SEEDS:
                rows.append({'experiment_id': f'FONG-{method}-B{budget:02d}-S{seed}',
                             'method': method, 'seed': seed, 'budget_pct': budget,
                             'calibration_n': 80 if budget == 5 else 320,
                             'initialization_run': f'FONG-SRC-S{seed}' if method == 'FT' else f'random-S{seed}',
                             'rounds': 25 if method in ('CLASS','FULL') else 0, 'target_updates': 2500})
    return rows


def validate_budget_rows(small, large, test_ids):
    ids5 = {r['canonical_identity'] for r in small}
    ids20 = {r['canonical_identity'] for r in large}
    if len(small) != 80 or len(ids5) != 80 or len(large) != 320 or len(ids20) != 320:
        raise ValueError('Budget rows or duplicate identities differ')
    if not ids5 < ids20:
        raise ValueError('Budgets must be strictly nested')
    if ids20 & set(test_ids):
        raise ValueError('Budget/test identity overlap')
    a = Counter((int(r['class_id']), float(r['concentration'])) for r in small)
    b = Counter((int(r['class_id']), float(r['concentration'])) for r in large)
    if set(a) != set(b) or len(a) != 40 or set(a.values()) != {2} or set(b.values()) != {8}:
        raise ValueError('40 strata must have exact 2/8 quotas')
    return {'nested': True, 'strata': 40, 'counts': {'5': 80, '20': 320}, 'test_overlap': 0}


def initial_model(seed):
    from gaps_flower.task import make_config, create_model
    return create_model(make_config(device='cpu', local_epochs=1, batch_size=32, profile='ce_stats', seed=seed))


def option(command, name, value):
    if name in command:
        command[command.index(name)+1] = str(value)
    else:
        command.extend([name, str(value)])


def build_commands(row):
    from scripts.run_iotj_canonical_v1_classification import build_canonical_commands
    from scripts.run_iotj_canonical_v1_comparators import build_source_fl_commands
    method, seed, runid = row['method'], row['seed'], row['experiment_id']
    if method not in ('SRC','CLASS','FULL'):
        raise ValueError('TO/FT use independent target-only entry')
    commands = copy.deepcopy(build_source_fl_commands('FedAvg') if method == 'SRC' else build_canonical_commands('C5'))
    old_id = commands['protocol']['experiment_id']
    scope = 'class' if method == 'CLASS' else 'full'
    for role in ('server','client_c1','client_c2'):
        commands[role] = [v.replace(old_id, runid) for v in commands[role]]
        option(commands[role], '--seed', seed)
        module_idx = commands[role].index('-m')+1
        commands[role][module_idx] = 'gaps_flower.fedongas27_entry'
        commands[role].insert(module_idx+1, 'server' if role=='server' else 'client')
        data_base = PI_DATA if role=='client_c1' else REMOTE_DATA
        commands[role].extend(['--fong-method', method, '--initial-state', f'{data_base}/initial/random-S{seed}.pth'])
        if role != 'server':
            option(commands[role], '--data-root', f'{data_base}/{scope}')
    if method != 'SRC':
        option(commands['server'], '--server-val-data', f'{REMOTE_DATA}/{scope}/client_1,{REMOTE_DATA}/{scope}/client_2')
        option(commands['server'], '--server-calib-data', f'{REMOTE_DATA}/{scope}/budget_{row["budget_pct"]:02d}')
        option(commands['server'], '--da-lambda-stage-mmd', '0.0' if method=='CLASS' else '0.2')
    commands['protocol'].update(row)
    commands['protocol'].update({
        'study_id': 'FEDONGAS-C5-BUDGET-MULTISEED-20261014',
        'initialization': f'common_random_seed{seed}',
        'prototype_scope': 'class_only' if method=='CLASS' else 'class_phase' if method=='FULL' else 'none',
        'adaptation_target_fields': ['x','class','phase'] if method=='FULL' else ['x','class'] if method=='CLASS' else [],
    })
    return commands
