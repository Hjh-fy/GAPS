"""Per-process hooks around the original server/client entrypoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from gaps_flower.fedongas27_methods import ClassOnlyClient, ClassDomainAdaptation
from gaps_flower.state_fingerprint import ordered_state_content_fingerprint


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('role', choices=('server','client'))
    parser.add_argument('--fong-method', required=True, choices=('SRC','CLASS','FULL'))
    parser.add_argument('--initial-state', required=True)
    args, original_args = parser.parse_known_args()
    seed = int(original_args[original_args.index('--seed')+1])
    initial = torch.load(args.initial_state, map_location='cpu', weights_only=False)
    if initial['seed'] != seed or initial['initialization'] != 'random':
        raise RuntimeError('Initial random state identity differs')
    if args.role == 'server':
        from gaps_flower import server_app as app
        from gaps_flower.strategy import GapsStrategy
        from gaps_flower.task import make_config, create_model

        class SeededStrategy(GapsStrategy):
            def _build_da_model(self, aggregated_state, device):
                config = make_config(device=str(device), local_epochs=1, batch_size=32, seed=seed)
                model = create_model(config)
                model.load_state_dict(aggregated_state, strict=True)
                return model.to(device)

            def _run_domain_adapt(self, *a, **kw):
                result = super()._run_domain_adapt(*a, **kw)
                _, diag, _ = result
                count = len(self.semantic_protos)
                if args.fong_method == 'CLASS' and count != 4:
                    raise RuntimeError(f'CLASS must have 4 prototypes, got {count}')
                if args.fong_method == 'FULL' and not (4 <= count <= 12):
                    raise RuntimeError(f'FULL prototype count invalid: {count}')
                audit = {'prototype_scope': args.fong_method, 'prototype_count': count,
                         'true_phase_used': args.fong_method == 'FULL', 'seed': seed,
                         'loss_activity': diag.get('loss_activity', []),
                         'changed_tensors': diag.get('checkpoint_changed_tensors'),
                         'steps': diag.get('num_steps')}
                round_idx = int(a[0] if a else kw['server_round'])
                (self.output_dir / f'fong_loss_audit_round_{round_idx:03d}.json').write_text(
                    json.dumps(audit, indent=2), encoding='utf-8')
                return result

        app.GapsStrategy = SeededStrategy
        if args.fong_method == 'CLASS':
            from gaps_flower import domain_adaptation
            domain_adaptation.ServerDomainAdaptation = ClassDomainAdaptation
    else:
        from gaps_flower import client_app as app
        if args.fong_method == 'CLASS':
            from gaps_flower import task
            task.Client = ClassOnlyClient
    original_create = app.create_model

    def initialized_model(config):
        model = original_create(config)
        model.load_state_dict(initial['model_state'], strict=True)
        fp = ordered_state_content_fingerprint(model.state_dict())
        if fp != initial['ordered_fingerprint']:
            raise RuntimeError('Loaded initial state fingerprint differs')
        print(json.dumps({'event': 'COMMON_RANDOM_INITIALIZATION', 'role': args.role,
                          'method': args.fong_method, 'seed': seed, 'ordered_fingerprint': fp}), flush=True)
        return model

    app.create_model = initialized_model
    sys.argv = [sys.argv[0], *original_args]
    app.main()


if __name__ == '__main__':
    main()
