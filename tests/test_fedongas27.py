"""Behavioral gates for the authorized 27-run C5 study."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest


def test_command_metadata_tracks_current_seed_and_class_only_information():
    from gaps_flower.fedongas27_protocol import build_commands, run_matrix
    row=next(r for r in run_matrix() if r['method']=='CLASS' and r['seed']==44)
    protocol=build_commands(row)['protocol']
    assert protocol['initialization']=='common_random_seed44'
    assert protocol['adaptation_target_fields']==['x','class']
    assert protocol['prototype_scope']=='class_only'
import torch
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 3)
        self.classifier = torch.nn.Linear(3, 4)

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat), torch.nn.functional.normalize(feat, dim=1), feat


def test_matrix_27_distinct_tasks_and_correct_dependencies():
    from gaps_flower.fedongas27_protocol import run_matrix
    rows = run_matrix()
    assert len(rows) == len({r['experiment_id'] for r in rows}) == 27
    assert Counter(r['method'] for r in rows) == {'SRC': 3, 'TO': 6, 'FT': 6, 'CLASS': 6, 'FULL': 6}
    for row in rows:
        assert row['seed'] in (42, 43, 44)
        if row['method'] == 'FT':
            assert row['initialization_run'] == f"FONG-SRC-S{row['seed']}"
        else:
            assert row['initialization_run'] == f"random-S{row['seed']}"


def budget_rows(quota):
    return [{'canonical_identity': f'{c}/{level}/{i}', 'class_id': str(c),
             'concentration': str(level)}
            for c in range(4) for level in range(10) for i in range(quota)]


def test_budget_gate_requires_nested_and_exact_stratum_quotas():
    from gaps_flower.fedongas27_protocol import validate_budget_rows
    assert validate_budget_rows(budget_rows(2), budget_rows(8), {'sealed'})['nested']
    corrupted = budget_rows(2)
    corrupted[0] = dict(corrupted[0], canonical_identity='not-in-20')
    with pytest.raises(ValueError, match='nested'):
        validate_budget_rows(corrupted, budget_rows(8), {'sealed'})
    with pytest.raises(ValueError, match='test'):
        validate_budget_rows(budget_rows(2), budget_rows(8), {'0/0/0'})


def test_random_initialization_same_seed_matches_and_other_seed_differs():
    from gaps_flower.fedongas27_protocol import initial_model
    a, b, c = initial_model(42), initial_model(42), initial_model(43)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())


def test_target_ce_updates_encoder_head_and_rebuilds_adam_each_block():
    from gaps_flower.fedongas27_methods import train_target_blocks
    torch.manual_seed(11)
    model = TinyModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
                                     torch.arange(4)), batch_size=4)
    rows = train_target_blocks(model, loader, seed=42, blocks=2, steps=2, formal=False)
    assert len(rows) == 4 and rows[-1]['update'] == 4
    assert all(r['loss_name'] == 'target_ce' and r['source_ce_enabled'] is False for r in rows)
    assert [r['optimizer_state_before_first_update'] for r in rows if r['step'] == 1] == [0, 0]
    assert all(r['encoder_grad_norm'] > 0 and r['classifier_grad_norm'] > 0 for r in rows)
    assert not torch.equal(before['encoder.weight'], model.encoder.weight)
    assert not torch.equal(before['classifier.weight'], model.classifier.weight)


def test_target_training_rejects_phase_or_concentration_payload():
    from gaps_flower.fedongas27_methods import train_target_blocks
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.arange(4), torch.ones(4)), batch_size=4)
    with pytest.raises(ValueError, match='exactly x/class'):
        train_target_blocks(TinyModel(), loader, seed=42, blocks=1, steps=1, formal=False)


def test_class_statistics_pool_samples_before_clipping_and_ignore_phase():
    from gaps_flower.fedongas27_methods import class_feature_statistics
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x, x, x
    model = IdentityModel()
    x = torch.tensor([[2., 4.], [4., 8.], [10., 12.]])
    y = torch.tensor([0, 0, 1])
    first = DataLoader(TensorDataset(x, y, torch.zeros(3, 4), torch.tensor([0, 2, 1])), batch_size=3)
    second = DataLoader(TensorDataset(x, y, torch.zeros(3, 4), torch.tensor([2, 1, 0])), batch_size=3)
    mus, counts, variances = class_feature_statistics(model, first, torch.device('cpu'), True, .01)
    mus2, counts2, _ = class_feature_statistics(model, second, torch.device('cpu'), True, .01)
    assert counts == counts2 == {(0, 0): 2, (1, 0): 1}
    assert torch.equal(mus[(0, 0)], torch.tensor([3., 6.]))
    assert torch.equal(variances[(0, 0)], torch.tensor([1., 4.]))
    assert all(torch.equal(mus[k], mus2[k]) for k in mus)


def test_class_semantic_query_and_consistency_ignore_true_phase():
    from gaps_flower.fedongas27_methods import ClassDomainAdaptation
    loader = DataLoader(TensorDataset(torch.randn(4, 2), torch.arange(4)), batch_size=4)
    protos = {f'{c},0': torch.ones(3) * (c+1) for c in range(4)}
    adapter = ClassDomainAdaptation(TinyModel(), loader, loader, protos, torch.device('cpu'),
                                   {'USE_ADVERSARIAL_DOMAIN': False, 'LAMBDA_STAGE_MMD': 0})
    assert torch.equal(adapter._semantic_proto((1, 2)), adapter._semantic_proto((1, 0)))
    feats, labels = torch.randn(4, 3), torch.arange(4)
    a = adapter._compute_consistency_loss(feats, labels, torch.tensor([0, 1, 2, 0]))
    b = adapter._compute_consistency_loss(feats, labels, torch.tensor([2, 0, 1, 2]))
    assert torch.equal(a, b) and a.requires_grad
    assert len(adapter.get_semantic_protos()) == 4
    assert adapter._compute_stage_mmd_loss(None, None, object(), None, None, object()).item() == 0


def test_full_adapter_keeps_original_class():
    from gaps_flower.fedongas27_methods import adapter_class
    from gaps_flower.domain_adaptation import ServerDomainAdaptation
    assert adapter_class('FULL') is ServerDomainAdaptation


def test_commands_keep_full_losses_and_class_only_stage_change():
    from gaps_flower.fedongas27_protocol import run_matrix, build_commands
    rows = run_matrix()
    full = next(r for r in rows if r['method'] == 'FULL' and r['seed'] == 43 and r['budget_pct'] == 5)
    cls = next(r for r in rows if r['method'] == 'CLASS' and r['seed'] == 43 and r['budget_pct'] == 5)
    for row in (full, cls):
        cmd = build_commands(row)
        server = cmd['server']
        assert server[server.index('--seed')+1] == '43'
        assert server[server.index('--domain-adapt-steps')+1] == '100'
        assert server[server.index('--da-lambda-target-ce')+1] == '0.0'
        assert server[server.index('--da-lambda-proto-mmd')+1] == '0.2'
        assert server[server.index('--use-adapted-as-global')+1] == 'true'
        assert server[server.index('--use-selective-agg')+1] == 'false'
        assert all(cmd[k][cmd[k].index('--profile')+1] == 'ce_stats' for k in ('client_c1','client_c2'))
    s = build_commands(cls)['server']
    assert s[s.index('--da-lambda-stage-mmd')+1] == '0.0'
    s = build_commands(full)['server']
    assert s[s.index('--da-lambda-stage-mmd')+1] == '0.2'


def test_src_commands_disable_target_adaptation():
    from gaps_flower.fedongas27_protocol import run_matrix, build_commands
    cmd = build_commands(next(r for r in run_matrix() if r['method']=='SRC'))
    assert '--use-domain-adapt' not in cmd['server']
    assert '--server-calib-data' not in cmd['server']
    assert cmd['server'][cmd['server'].index('--strategy')+1] == 'fedavg'
    assert cmd['server'][cmd['server'].index('--rounds')+1] == '25'
