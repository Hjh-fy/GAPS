"""Isolated class-only prototypes and target-only CE for the registered study."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from client import Client
from gaps_flower.domain_adaptation import ServerDomainAdaptation

ORIGINAL_ADAPTER = ServerDomainAdaptation


def class_feature_statistics(model, loader, device, return_var=True, min_var=.01):
    """Pool raw per-sample features by class before any prototype clipping."""
    model.eval()
    sums, squares, counts = {}, {}, {}
    with torch.no_grad():
        for batch in loader:
            x, labels = batch[0].to(device), batch[1].to(device)
            _, normalized, raw = model(x)
            features = raw if return_var else normalized
            for i, label in enumerate(labels):
                key = (int(label.item()), 0)  # wire compatibility tag, NOT a true phase
                if key not in sums:
                    sums[key] = torch.zeros_like(features[i])
                    squares[key] = torch.zeros_like(features[i])
                    counts[key] = 0
                sums[key] += features[i]
                squares[key] += features[i].square()
                counts[key] += 1
    means = {k: sums[k] / counts[k] for k in sums}
    if not return_var:
        return means, counts
    variances = {k: (squares[k]/counts[k] - means[k].square()).clamp(min=min_var) for k in sums}
    return means, counts, variances


class ClassOnlyClient(Client):
    def _compute_feature_means(self, return_var=False):
        return class_feature_statistics(
            self.model, self.train_loader, self.device, return_var,
            float(getattr(self.config, 'MAHALANOBIS_MIN_VAR', .01)))


class ClassDomainAdaptation(ORIGINAL_ADAPTER):
    """Four class prototypes; true stage labels never select semantic entries."""
    def _semantic_proto(self, key):
        parsed = self._parse_proto_key(key)
        if parsed is None:
            return None
        return super()._semantic_proto((parsed[0], 0))

    def _compute_consistency_loss(self, feats, labels, phases):
        if labels is None:
            return torch.tensor(0., device=self.device)
        # Preserve the original effective consistency operator and raw feature space.
        return super()._compute_consistency_loss(feats, labels, torch.zeros_like(labels))

    def _compute_stage_mmd_loss(self, *args, **kwargs):
        return torch.tensor(0., device=self.device)


def adapter_class(method):
    if method == 'FULL':
        return ORIGINAL_ADAPTER
    if method == 'CLASS':
        return ClassDomainAdaptation
    raise ValueError(method)


def train_target_blocks(model, loader, *, seed, blocks=25, steps=100, formal=True):
    """Only target CE; rebuilding Adam never adds source batches or source CE."""
    if formal and (blocks != 25 or steps != 100 or seed not in (42, 43, 44)):
        raise ValueError('Fixed protocol requires 25x100 updates and seed42/43/44')
    from utils import set_random_seed
    set_random_seed(seed)
    model.train()
    rows, iterator = [], iter(loader)
    for block in range(1, blocks+1):
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        for step in range(1, steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if len(batch) != 2:
                raise ValueError('Target training receives exactly x/class')
            device = next(model.parameters()).device
            x, labels = batch[0].to(device), batch[1].to(device).long()
            state_size = len(optimizer.state)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite target CE')
            loss.backward()
            enc, head = [], []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    (head if name.startswith('classifier.') else enc).append(param.grad.norm().square())
            if not enc or not head:
                raise RuntimeError('Encoder/classifier CE gradient missing')
            enc_norm, head_norm = torch.stack(enc).sum().sqrt(), torch.stack(head).sum().sqrt()
            if not torch.isfinite(enc_norm + head_norm):
                raise RuntimeError('Nonfinite target CE gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            rows.append({'block': block, 'step': step, 'update': len(rows)+1,
                         'loss_name': 'target_ce', 'raw_loss': float(loss.detach()),
                         'configured_weight': 1., 'source_ce_enabled': False,
                         'encoder_grad_norm': float(enc_norm), 'classifier_grad_norm': float(head_norm),
                         'optimizer_state_before_first_update': state_size if step == 1 else None})
    return rows
