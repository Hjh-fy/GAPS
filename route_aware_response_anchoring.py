import argparse
import csv
import json
from pathlib import Path

import numpy as np


GAS_NAMES = ['Ethanol', 'CO', 'Ethylene', 'Methane']


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def gas_name(cls):
    cls = inum(cls)
    return GAS_NAMES[cls] if 0 <= cls < len(GAS_NAMES) else f'Class{cls}'


def client_sort_key(name):
    text = str(name).strip()
    if text.startswith('C'):
        text = text[1:]
    try:
        return int(float(text))
    except ValueError:
        return 10**9


def parse_list(text, cast=float):
    return [cast(x.strip()) for x in str(text).split(',') if x.strip()]


def empty_metric(n):
    return {'n': int(n), 'R2': None, 'RMSE': None, 'MAE': None, 'MedAE': None,
            'P90AE': None, 'P95AE': None, 'Bias': None, 'class_acc': None}


def metric(rows, pred_key='pred_ppm'):
    if not rows:
        return empty_metric(0)
    y = np.asarray([fnum(r.get('true_ppm')) for r in rows], dtype=np.float64)
    p = np.asarray([fnum(r.get(pred_key)) for r in rows], dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size < 2:
        return empty_metric(int(y.size))
    err = p - y
    ae = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None
    return {
        'n': int(y.size),
        'R2': r2,
        'RMSE': float(np.sqrt(np.mean(err ** 2))),
        'MAE': float(np.mean(ae)),
        'MedAE': float(np.median(ae)),
        'P90AE': float(np.percentile(ae, 90)),
        'P95AE': float(np.percentile(ae, 95)),
        'Bias': float(np.mean(err)),
        'class_acc': float(np.mean([inum(r.get('class_correct')) for r in rows])),
    }


def improvement(base, corr):
    return {
        'delta_R2': None if corr['R2'] is None or base['R2'] is None else corr['R2'] - base['R2'],
        'delta_MAE': None if corr['MAE'] is None or base['MAE'] is None else corr['MAE'] - base['MAE'],
        'delta_P90AE': None if corr['P90AE'] is None or base['P90AE'] is None else corr['P90AE'] - base['P90AE'],
        'delta_P95AE': None if corr['P95AE'] is None or base['P95AE'] is None else corr['P95AE'] - base['P95AE'],
    }


def route_key(row):
    return row.get('client', ''), inum(row.get('pred_class'))


def route_name(client, pred_cls):
    return f'{client}_pred{pred_cls}_{gas_name(pred_cls)}'


def clamp_by_pred_class(value, pred_cls):
    ranges = {0: (12.5, 125.0), 1: (25.0, 250.0), 2: (12.5, 125.0), 3: (25.0, 250.0)}
    lo, hi = ranges.get(inum(pred_cls), (0.0, 250.0))
    return float(min(max(float(value), lo), hi))


def candidate_reference(row, ref_key):
    for key in [ref_key, 'best_response_mean_nearest_calib_conc',
                'best_response_nearest_calib_conc', 'nearest_mean_calib_conc',
                'nearest_calib_conc']:
        value = fnum(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def select_threshold(rows, score, trigger_rate):
    vals = np.asarray([fnum(r.get(score)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.inf
    trigger_rate = min(max(float(trigger_rate), 0.0), 1.0)
    return float(np.percentile(vals, 100.0 * (1.0 - trigger_rate)))


def none_policy():
    return {'policy_type': 'none', 'score': 'none', 'trigger_rate': 0.0,
            'risk_threshold': np.inf, 'alpha': 0.0, 'ref_key': 'none',
            'require_rank_mismatch': 0, 'min_margin': 0.0,
            'max_margin': None, 'max_amp_all': None}


def is_triggered(row, policy):
    if policy.get('policy_type') == 'none':
        return False
    allowed = policy.get('allowed_pred_classes') or []
    if allowed and inum(row.get('pred_class')) not in {inum(x) for x in allowed}:
        return False
    if fnum(row.get(policy['score']), -np.inf) < float(policy['risk_threshold']):
        return False
    if int(policy.get('require_rank_mismatch', 0)) and inum(row.get('class_response_rank'), 1) <= 1:
        return False
    if fnum(row.get('class_response_margin_risk'), 0.0) < float(policy.get('min_margin', 0.0)):
        return False
    max_margin = policy.get('max_margin')
    if max_margin is not None and fnum(row.get('class_response_margin_risk'), np.inf) > float(max_margin):
        return False
    max_amp = policy.get('max_amp_all')
    if max_amp is not None and fnum(row.get('feature_amp_all'), np.inf) > float(max_amp):
        return False
    return True


def apply_policy(rows, policy, policy_name='route'):
    out = []
    for row in rows:
        r = dict(row)
        pred = fnum(row.get('pred_ppm'))
        ref = candidate_reference(row, policy.get('ref_key', 'nearest_mean_calib_conc'))
        triggered = is_triggered(row, policy)
        if triggered and np.isfinite(pred) and np.isfinite(ref):
            alpha = float(policy.get('alpha', 0.0))
            corrected = (1.0 - alpha) * pred + alpha * ref
            corrected = clamp_by_pred_class(corrected, row.get('pred_class'))
        else:
            corrected = pred
        r['t5_policy'] = policy_name
        r['t5_triggered'] = int(bool(triggered))
        r['t5_reference_ppm'] = ref
        r['t5_pred_ppm'] = corrected
        r['t5_error_ppm'] = corrected - fnum(row.get('true_ppm'))
        r['t5_abs_error_ppm'] = abs(r['t5_error_ppm'])
        out.append(r)
    return out


def policy_key(policy):
    if policy.get('policy_type') == 'none':
        return 'none'
    return '{}|rate={:.2f}|thr={:.4f}|alpha={:.2f}|ref={}|rank={}|min_margin={:.2f}|max_margin={}|max_amp={}'.format(
        policy['score'], policy['trigger_rate'], policy['risk_threshold'], policy['alpha'],
        policy['ref_key'], policy['require_rank_mismatch'], policy['min_margin'],
        policy.get('max_margin'), policy.get('max_amp_all'))


def candidate_policies(rows, pred_cls, args):
    yield none_policy()
    max_margins = [None] + args.max_margins
    max_amps = [None]
    if args.mode == 'safe' and pred_cls == 1:
        max_margins = [2.0]
        max_amps = [None, 0.08]
    for score in args.scores:
        for rate in args.trigger_rates:
            threshold = select_threshold(rows, score, rate)
            for alpha in args.alphas:
                for ref_key in args.ref_keys:
                    for require_rank in args.require_rank_options:
                        for min_margin in args.min_margins:
                            for max_margin in max_margins:
                                for max_amp in max_amps:
                                    yield {'policy_type': 'anchor', 'score': score,
                                           'trigger_rate': float(rate), 'risk_threshold': float(threshold),
                                           'alpha': float(alpha), 'ref_key': ref_key,
                                           'require_rank_mismatch': int(bool(require_rank)),
                                           'min_margin': float(min_margin), 'max_margin': max_margin,
                                           'max_amp_all': max_amp,
                                           'allowed_pred_classes': sorted(args.allowed_pred_classes)}


def policy_selection_ok(base, corr, args):
    if corr['R2'] is not None and base['R2'] is not None:
        if corr['R2'] < base['R2'] - args.r2_guard:
            return False
    if corr['MAE'] is not None and base['MAE'] is not None:
        if corr['MAE'] > base['MAE'] + args.max_mae_worse:
            return False
    p90_gain = 0.0 if corr['P90AE'] is None or base['P90AE'] is None else base['P90AE'] - corr['P90AE']
    mae_gain = 0.0 if corr['MAE'] is None or base['MAE'] is None else base['MAE'] - corr['MAE']
    if p90_gain < args.min_selected_p90_gain and mae_gain < args.min_selected_mae_gain:
        return False
    return p90_gain >= args.min_p90_gain or mae_gain >= args.min_mae_gain


def select_route_policy(calib_rows, client, pred_cls, args):
    label = route_name(client, pred_cls)
    base = metric(calib_rows, 'pred_ppm')
    sweep = []
    if args.allowed_pred_classes and pred_cls not in args.allowed_pred_classes:
        selected = {'selection_group': label, 'client': client, 'pred_class': pred_cls,
                    'pred_gas': gas_name(pred_cls), 'reason': 'pred_class_not_allowed',
                    **none_policy(), 'base_R2': base['R2'], 'base_MAE': base['MAE'],
                    'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE']}
        return none_policy(), selected, sweep
    if args.auto_hard_clients and client not in args.hard_clients:
        selected = {'selection_group': label, 'client': client, 'pred_class': pred_cls,
                    'pred_gas': gas_name(pred_cls), 'reason': 'easy_client_protected',
                    **none_policy(), 'base_R2': base['R2'], 'base_MAE': base['MAE'],
                    'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE']}
        return none_policy(), selected, sweep
    hard_by_r2 = base['R2'] is not None and base['R2'] <= args.hard_route_max_base_r2
    hard_by_mae = base['MAE'] is not None and base['MAE'] >= args.hard_route_min_base_mae
    hard_by_p90 = base['P90AE'] is not None and base['P90AE'] >= args.hard_route_min_base_p90
    if args.require_hard_route and not (hard_by_r2 or hard_by_mae or hard_by_p90):
        selected = {'selection_group': label, 'client': client, 'pred_class': pred_cls,
                    'pred_gas': gas_name(pred_cls), 'reason': 'easy_route_protected',
                    **none_policy(), 'base_R2': base['R2'], 'base_MAE': base['MAE'],
                    'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE']}
        return none_policy(), selected, sweep
    if len(calib_rows) < args.min_route_samples:
        selected = {'selection_group': label, 'client': client, 'pred_class': pred_cls,
                    'pred_gas': gas_name(pred_cls), 'reason': 'too_few_samples',
                    **none_policy(), 'base_R2': base['R2'], 'base_MAE': base['MAE'],
                    'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE']}
        return none_policy(), selected, sweep
    best = None
    for policy in candidate_policies(calib_rows, pred_cls, args):
        corrected = apply_policy(calib_rows, policy, label)
        corr = metric(corrected, 't5_pred_ppm')
        delta = improvement(base, corr)
        row = {'selection_group': label, 'client': client, 'pred_class': pred_cls,
               'pred_gas': gas_name(pred_cls), 'config_key': policy_key(policy), **policy,
               'triggered_fraction': float(np.mean([inum(r.get('t5_triggered')) for r in corrected])) if corrected else 0.0,
               'base_R2': base['R2'], 'base_MAE': base['MAE'],
               'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE'],
               'R2': corr['R2'], 'MAE': corr['MAE'], 'P90AE': corr['P90AE'],
               'P95AE': corr['P95AE'], **delta}
        sweep.append(row)
        if policy.get('policy_type') == 'none' or not policy_selection_ok(base, corr, args):
            continue
        key = (fnum(row.get('P90AE'), np.inf), fnum(row.get('MAE'), np.inf), -fnum(row.get('R2'), -np.inf))
        if best is None or key < best[0]:
            best = (key, row, policy)
    if best is None:
        row = dict([r for r in sweep if r.get('policy_type') == 'none'][0])
        row['reason'] = 'no_safe_gain'
        return none_policy(), row, sweep
    row = dict(best[1])
    row['reason'] = 'selected'
    return best[2], row, sweep


def select_client_policy(calib_rows, client, args):
    label = '{}_client'.format(client)
    base = metric(calib_rows, 'pred_ppm')
    sweep = []
    if args.auto_hard_clients and client not in args.hard_clients:
        selected = {'selection_group': label, 'client': client, 'pred_class': '',
                    'pred_gas': 'ALL', 'reason': 'easy_client_protected',
                    **none_policy(), 'base_R2': base['R2'], 'base_MAE': base['MAE'],
                    'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE']}
        return none_policy(), selected, sweep
    if len(calib_rows) < args.min_route_samples:
        selected = {'selection_group': label, 'client': client, 'pred_class': '',
                    'pred_gas': 'ALL', 'reason': 'too_few_samples',
                    **none_policy(), 'base_R2': base['R2'], 'base_MAE': base['MAE'],
                    'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE']}
        return none_policy(), selected, sweep
    best = None
    for policy in candidate_policies(calib_rows, -1, args):
        corrected = apply_policy(calib_rows, policy, label)
        corr = metric(corrected, 't5_pred_ppm')
        delta = improvement(base, corr)
        row = {'selection_group': label, 'client': client, 'pred_class': '',
               'pred_gas': 'ALL', 'config_key': policy_key(policy), **policy,
               'triggered_fraction': float(np.mean([inum(r.get('t5_triggered')) for r in corrected])) if corrected else 0.0,
               'base_R2': base['R2'], 'base_MAE': base['MAE'],
               'base_P90AE': base['P90AE'], 'base_P95AE': base['P95AE'],
               'R2': corr['R2'], 'MAE': corr['MAE'], 'P90AE': corr['P90AE'],
               'P95AE': corr['P95AE'], **delta}
        sweep.append(row)
        if policy.get('policy_type') == 'none' or not policy_selection_ok(base, corr, args):
            continue
        key = (fnum(row.get('P90AE'), np.inf), fnum(row.get('MAE'), np.inf), -fnum(row.get('R2'), -np.inf))
        if best is None or key < best[0]:
            best = (key, row, policy)
    if best is None:
        row = dict([r for r in sweep if r.get('policy_type') == 'none'][0])
        row['reason'] = 'no_safe_gain'
        return none_policy(), row, sweep
    row = dict(best[1])
    row['reason'] = 'selected'
    return best[2], row, sweep


def group_routes(rows):
    routes = {}
    for row in rows:
        routes.setdefault(route_key(row), []).append(row)
    return routes


def apply_route_policies(test_rows, policies):
    out = []
    for key, rows in group_routes(test_rows).items():
        client, pred_cls = key
        policy = policies.get(key, none_policy())
        out.extend(apply_policy(rows, policy, route_name(client, pred_cls)))
    out.sort(key=lambda r: (client_sort_key(r.get('client')), inum(r.get('sample_index'))))
    return out


def apply_client_policies(test_rows, policies):
    out = []
    clients = sorted({r.get('client') for r in test_rows if r.get('client')}, key=client_sort_key)
    for client in clients:
        rows = [r for r in test_rows if r.get('client') == client]
        policy = policies.get(client, none_policy())
        out.extend(apply_policy(rows, policy, '{}_client'.format(client)))
    out.sort(key=lambda r: (client_sort_key(r.get('client')), inum(r.get('sample_index'))))
    return out


def add_summary(rows, level, group, cls, base_rows, corr_rows):
    base = metric(base_rows, 'pred_ppm')
    corr = metric(corr_rows, 't5_pred_ppm')
    rows.append({'level': level, 'group': group, 'true_class': '' if cls is None else cls,
                 'true_gas': '' if cls is None else gas_name(cls), 'n': base['n'],
                 'base_R2': base['R2'], 'R2': corr['R2'],
                 'base_MAE': base['MAE'], 'MAE': corr['MAE'],
                 'base_P90AE': base['P90AE'], 'P90AE': corr['P90AE'],
                 'base_P95AE': base['P95AE'], 'P95AE': corr['P95AE'],
                 'triggered_fraction': float(np.mean([inum(r.get('t5_triggered')) for r in corr_rows])) if corr_rows else 0.0,
                 'class_acc': base['class_acc'], **improvement(base, corr)})


def summarize_groups(base_rows, corr_rows):
    rows = []
    clients = sorted({r.get('client') for r in base_rows if r.get('client')}, key=client_sort_key)
    add_summary(rows, 'ALL', 'ALL', None, base_rows, corr_rows)
    for client in clients:
        b = [r for r in base_rows if r.get('client') == client]
        c = [r for r in corr_rows if r.get('client') == client]
        add_summary(rows, 'client', client, None, b, c)
    for client in clients:
        for cls in range(len(GAS_NAMES)):
            b = [r for r in base_rows if r.get('client') == client and inum(r.get('true_class')) == cls]
            c = [r for r in corr_rows if r.get('client') == client and inum(r.get('true_class')) == cls]
            if b:
                add_summary(rows, 'client_true_class', client, cls, b, c)
    for (client, pred_cls), b in sorted(group_routes(base_rows).items(), key=lambda x: (client_sort_key(x[0][0]), x[0][1])):
        c = [r for r in corr_rows if route_key(r) == (client, pred_cls)]
        add_summary(rows, 'client_pred_route', route_name(client, pred_cls), pred_cls, b, c)
    return rows


def select_hard_clients(calib_rows, args):
    hard = set()
    client_rows = []
    clients = sorted({r.get('client') for r in calib_rows if r.get('client')}, key=client_sort_key)
    for client in clients:
        rows = [r for r in calib_rows if r.get('client') == client]
        m = metric(rows, 'pred_ppm')
        is_hard = False
        if m['R2'] is not None and m['R2'] <= args.hard_client_max_base_r2:
            is_hard = True
        if m['MAE'] is not None and m['MAE'] >= args.hard_client_min_base_mae:
            is_hard = True
        if m['P90AE'] is not None and m['P90AE'] >= args.hard_client_min_base_p90:
            is_hard = True
        if is_hard:
            hard.add(client)
        client_rows.append({'client': client, 'hard_client': int(is_hard), 'R2': m['R2'],
                            'MAE': m['MAE'], 'P90AE': m['P90AE'], 'P95AE': m['P95AE'],
                            'n': m['n'], 'class_acc': m['class_acc']})
    return hard, client_rows



def fmt(value):
    if value is None:
        return ''
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        return f'{float(value):.4f}'
    return str(value)


def write_markdown(path, selected_rows, summary_rows, args):
    lines = ['# T5 Route-Aware Automatic QC', '']
    lines.append(f'- calibration records: `{args.calibration_records_csv}`')
    lines.append(f'- test records: `{args.test_records_csv}`')
    lines.append(f'- mode: `{args.mode}`')
    lines.append('- selection unit: client + predicted gas route')
    lines.append('- protocol: single-window prediction; no file-level aggregation')
    if getattr(args, 'hard_clients', set()):
        hard_text = ', '.join(sorted(args.hard_clients, key=client_sort_key))
        lines.append(f'- auto hard clients: `{hard_text}`')
    lines.append('')
    cols = ['selection_group', 'reason', 'policy_type', 'score', 'trigger_rate', 'alpha', 'ref_key',
            'require_rank_mismatch', 'min_margin', 'max_margin', 'max_amp_all', 'base_P90AE',
            'P90AE', 'delta_P90AE', 'base_MAE', 'MAE', 'delta_MAE', 'base_R2', 'R2', 'delta_R2']
    lines.append('## Selected Route Policies')
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
    for row in selected_rows:
        lines.append('| ' + ' | '.join(fmt(row.get(c)) for c in cols) + ' |')
    lines.append('')
    eval_cols = ['level', 'group', 'true_gas', 'n', 'base_R2', 'R2', 'delta_R2', 'base_MAE',
                 'MAE', 'delta_MAE', 'base_P90AE', 'P90AE', 'delta_P90AE', 'triggered_fraction', 'class_acc']
    lines.append('## Frozen Test Evaluation')
    lines.append('| ' + ' | '.join(eval_cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(eval_cols)) + '|')
    for row in summary_rows:
        lines.append('| ' + ' | '.join(fmt(row.get(c)) for c in eval_cols) + ' |')
    lines.append('')
    lines.append('## Reading Guide')
    lines.append('- `none` means calibration did not validate a safe correction for that route.')
    lines.append('- A useful route should reduce MAE/P90AE on test while keeping R2 stable or improved.')
    lines.append('- Easy routes should become no-op automatically; that is route protection.')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description='T5 route-aware calibration-selected response anchoring')
    parser.add_argument('--calibration_records_csv', required=True)
    parser.add_argument('--test_records_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mode', choices=['free', 'safe'], default='safe')
    parser.add_argument('--selection_level', choices=['route', 'client'], default='route')
    parser.add_argument('--scores', default='response_mean_conc_gap_norm,composite_response_risk,response_signature_norm,class_response_rank_risk,class_response_margin_risk,route_response_risk')
    parser.add_argument('--trigger_rates', default='0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50')
    parser.add_argument('--alphas', default='0.10,0.20,0.30,0.40,0.50,0.70')
    parser.add_argument('--ref_keys', default='nearest_mean_calib_conc,nearest_calib_conc,best_response_mean_nearest_calib_conc,best_response_nearest_calib_conc')
    parser.add_argument('--require_rank_options', default='0,1')
    parser.add_argument('--min_margins', default='0,0.25,0.5,1.0')
    parser.add_argument('--max_margins', default='1.0,2.0,3.0,5.0')
    parser.add_argument('--min_route_samples', type=int, default=20)
    parser.add_argument('--r2_guard', type=float, default=0.03)
    parser.add_argument('--max_mae_worse', type=float, default=0.0)
    parser.add_argument('--min_p90_gain', type=float, default=1.0)
    parser.add_argument('--min_mae_gain', type=float, default=0.25)
    parser.add_argument('--min_selected_p90_gain', type=float, default=0.0)
    parser.add_argument('--min_selected_mae_gain', type=float, default=0.0)
    parser.add_argument('--allowed_pred_classes', default='')
    parser.add_argument('--require_hard_route', action='store_true')
    parser.add_argument('--hard_route_max_base_r2', type=float, default=0.85)
    parser.add_argument('--hard_route_min_base_mae', type=float, default=20.0)
    parser.add_argument('--hard_route_min_base_p90', type=float, default=45.0)
    parser.add_argument('--auto_hard_clients', action='store_true')
    parser.add_argument('--hard_client_max_base_r2', type=float, default=0.85)
    parser.add_argument('--hard_client_min_base_mae', type=float, default=18.0)
    parser.add_argument('--hard_client_min_base_p90', type=float, default=45.0)
    return parser


def normalize_args(args):
    args.scores = parse_list(args.scores, str)
    args.trigger_rates = parse_list(args.trigger_rates, float)
    args.alphas = parse_list(args.alphas, float)
    args.ref_keys = parse_list(args.ref_keys, str)
    args.require_rank_options = [bool(int(x)) for x in parse_list(args.require_rank_options, str)]
    args.min_margins = parse_list(args.min_margins, float)
    args.max_margins = parse_list(args.max_margins, float)
    args.allowed_pred_classes = set(parse_list(args.allowed_pred_classes, int)) if str(args.allowed_pred_classes).strip() else set()
    args.hard_clients = set()
    return args


def main():
    args = normalize_args(build_parser().parse_args())
    calib_rows = read_csv(args.calibration_records_csv)
    test_rows = read_csv(args.test_records_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_client_rows = []
    if args.auto_hard_clients:
        args.hard_clients, hard_client_rows = select_hard_clients(calib_rows, args)

    policies = {}
    selected_rows = []
    sweep_rows = []
    if args.selection_level == 'client':
        clients = sorted({r.get('client') for r in calib_rows if r.get('client')}, key=client_sort_key)
        for client in clients:
            rows = [r for r in calib_rows if r.get('client') == client]
            policy, selected, sweep = select_client_policy(rows, client, args)
            policies[client] = policy
            selected_rows.append(selected)
            sweep_rows.extend(sweep)
        corrected = apply_client_policies(test_rows, policies)
        corrected_calib = apply_client_policies(calib_rows, policies)
    else:
        for (client, pred_cls), rows in sorted(group_routes(calib_rows).items(), key=lambda x: (client_sort_key(x[0][0]), x[0][1])):
            policy, selected, sweep = select_route_policy(rows, client, pred_cls, args)
            policies[(client, pred_cls)] = policy
            selected_rows.append(selected)
            sweep_rows.extend(sweep)
        corrected = apply_route_policies(test_rows, policies)
        corrected_calib = apply_route_policies(calib_rows, policies)
    summary_rows = summarize_groups(test_rows, corrected)
    write_csv(out_dir / 't5_selected_route_policies.csv', selected_rows)
    write_csv(out_dir / 't5_policy_sweep.csv', sweep_rows)
    write_csv(out_dir / 't5_corrected_calibration.csv', corrected_calib)
    write_csv(out_dir / 't5_corrected_test.csv', corrected)
    write_csv(out_dir / 't5_test_summary.csv', summary_rows)
    if hard_client_rows:
        write_csv(out_dir / 't5_hard_clients.csv', hard_client_rows)
    payload = {'settings': {k: str(v) for k, v in vars(args).items()},
               'hard_clients': sorted(args.hard_clients, key=client_sort_key),
               'selected_policies': selected_rows, 'test_summary': summary_rows}
    (out_dir / 't5_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    write_markdown(out_dir / 't5_summary.md', selected_rows, summary_rows, args)
    print((out_dir / 't5_summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
