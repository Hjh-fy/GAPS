"""The prepared data contract must reject budget swaps and true-stage CLASS views."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_preflight_binds_existing_budget_identities_and_test_population():
    from scripts.prepare_fedongas27 import inspect_inputs
    check = inspect_inputs()
    assert check['status'] == 'PASS'
    assert check['counts'] == {'5': 80, '20': 320}
    assert check['strata'] == 40 and check['nested'] is True
    assert check['test_n'] == 1360 and check['test_overlap'] == 0
    assert check['target_phases'] == [2]


def test_class_view_derives_tags_without_reading_real_phase_or_concentration(tmp_path, monkeypatch):
    from scripts.prepare_fedongas27 import copy_source_view
    src = tmp_path / 'original'
    src.mkdir()
    for split in ('train','calibration','test'):
        np.save(src / f'{split}_features.npy', np.zeros((4, 50, 8), np.float32))
        np.save(src / f'{split}_classification_labels.npy', np.arange(4))
    # Intentionally absent phase/concentration files: CLASS must still construct.
    dest = tmp_path / 'class'
    copy_source_view(src, dest, class_only=True)
    for split in ('train','calibration','test'):
        assert np.array_equal(np.load(dest/f'{split}_phase_labels.npy'), np.zeros(4, np.int64))
        assert np.array_equal(np.load(dest/f'{split}_regression_labels.npy'), np.zeros((4,4), np.float32))


def test_completion_gate_rejects_missing_or_tampered_final_asset(tmp_path):
    from scripts.evaluate_fedongas27 import completion_gate
    from gaps_flower.fedongas27_protocol import run_matrix
    with pytest.raises(RuntimeError, match='missing'):
        completion_gate(tmp_path, run_matrix(), 'freeze')


def test_completion_gate_rejects_failed_marker_even_with_valid_file_hash(tmp_path):
    from scripts.evaluate_fedongas27 import completion_gate
    from gaps_flower.fedongas27_protocol import run_matrix
    from scripts.prepare_fedongas27 import sha
    row=run_matrix()[0]; directory=tmp_path/'runs'/row['experiment_id']; directory.mkdir(parents=True)
    cp=directory/'checkpoint.pth'; cp.write_bytes(b'incomplete')
    marker={'run':row,'freeze_commit':'freeze','checkpoint':cp.name,'checkpoint_sha256':sha(cp),
            'status':'FAILED','target_updates':0,'target_test_opened':False}
    (directory/'FINAL_ENDPOINT.json').write_text(json.dumps(marker))
    with pytest.raises(RuntimeError,match='status'):
        completion_gate(tmp_path,[row],'freeze')


def test_completion_gate_rejects_wrong_round_with_matching_content_hash(tmp_path):
    import torch
    from scripts.evaluate_fedongas27 import completion_gate
    from gaps_flower.fedongas27_protocol import run_matrix
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
    from scripts.prepare_fedongas27 import sha
    row=run_matrix()[0]; directory=tmp_path/'runs'/row['experiment_id']; directory.mkdir(parents=True)
    state={'w':torch.zeros(2)}; cp=directory/'checkpoint.pth'; torch.save({'round':24,'model_state':state},cp)
    marker={'run':row,'freeze_commit':'freeze','checkpoint':cp.name,'checkpoint_sha256':sha(cp),
            'status':'PASS','target_updates':0,'target_test_opened':False,
            'state_fingerprint':ordered_state_content_fingerprint(state)}
    (directory/'FINAL_ENDPOINT.json').write_text(json.dumps(marker))
    with pytest.raises(RuntimeError,match='round25'):
        completion_gate(tmp_path,[row],'freeze')


def test_summary_preserves_paired_negative_and_zero_deltas():
    from scripts.evaluate_fedongas27 import paired_comparisons
    rows = []
    for budget in (5, 20):
        for seed in (42, 43, 44):
            for method, value in (('TO',.9),('FT',.9),('CLASS',.95),('FULL',.92)):
                rows.append({'budget_pct': budget, 'seed': seed, 'method': method,
                             'accuracy': value, 'macro_f1': value})
    pairs, summaries = paired_comparisons(rows)
    assert len(pairs) == 18 and len(summaries) == 6
    assert all(r['accuracy_delta'] == 0 for r in pairs if r['comparison']=='FT-TO')
    assert all(r['accuracy_delta'] < 0 for r in pairs if r['comparison']=='FULL-CLASS')
    assert all(r['n_seeds']==3 for r in summaries)
