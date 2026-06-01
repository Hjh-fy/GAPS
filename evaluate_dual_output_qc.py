import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    return GAS_NAMES[cls] if 0 <= cls < len(GAS_NAMES) else 'Class{}'.format(cls)


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


def metric(rows, pred_key):
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
    return {'n': int(y.size), 'R2': r2, 'MAE': float(np.mean(ae)),
            'MedAE': float(np.median(ae)), 'P90AE': float(np.percentile(ae, 90)),
            'P95AE': float(np.percentile(ae, 95)), 'Bias': float(np.mean(err)),
            'class_acc': float(np.mean([inum(r.get('class_correct')) for r in rows]))}


def empty_metric(n):
    return {'n': int(n), 'R2': None, 'MAE': None, 'MedAE': None,
            'P90AE': None, 'P95AE': None, 'Bias': None, 'class_acc': None}


def delta(base, item):
    return {'delta_R2': diff(item.get('R2'), base.get('R2')),
            'delta_MAE': diff(item.get('MAE'), base.get('MAE')),
            'delta_P90AE': diff(item.get('P90AE'), base.get('P90AE')),
            'delta_P95AE': diff(item.get('P95AE'), base.get('P95AE'))}


def diff(a, b):
    if a is None or b is None:
        return None
    return a - b


def risk_value(row, score, pred_key):
    if score == 'corrected_abs_delta':
        return abs(fnum(row.get(pred_key)) - fnum(row.get('pred_ppm')))
    if score == 'corrected_composite':
        return max(fnum(row.get('composite_response_risk'), 0.0),
                   abs(fnum(row.get(pred_key)) - fnum(row.get('pred_ppm'))) / 25.0)
    return fnum(row.get(score), np.nan)


def build_thresholds(calib_rows, scores, coverages, pred_key, group_mode):
    groups = {'ALL': calib_rows}
    if group_mode in ('client', 'client_and_all'):
        for client in sorted({r.get('client') for r in calib_rows if r.get('client')}, key=client_sort_key):
            groups[client] = [r for r in calib_rows if r.get('client') == client]
    rows = []
    thresholds = {}
    for group, group_rows in groups.items():
        for score in scores:
            vals = np.asarray([risk_value(r, score, pred_key) for r in group_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            for cov in coverages:
                thr = float(np.percentile(vals, 100.0 * float(cov)))
                thresholds[(group, score, float(cov))] = thr
                rows.append({'group': group, 'score': score, 'target_coverage': float(cov),
                             'risk_threshold': thr, 'calib_n': int(vals.size)})
    return thresholds, rows


def threshold_for(row, thresholds, score, cov):
    client = row.get('client')
    key = (client, score, float(cov))
    if key in thresholds:
        return thresholds[key]
    return thresholds.get(('ALL', score, float(cov)), np.inf)


def group_rows(rows):
    groups = {'ALL': rows}
    for client in sorted({r.get('client') for r in rows if r.get('client')}, key=client_sort_key):
        groups[client] = [r for r in rows if r.get('client') == client]
    for client in sorted({r.get('client') for r in rows if r.get('client')}, key=client_sort_key):
        for cls in range(len(GAS_NAMES)):
            sub = [r for r in rows if r.get('client') == client and inum(r.get('true_class')) == cls]
            if sub:
                groups['{}_{}'.format(client, gas_name(cls))] = sub
    return groups


def evaluate_acceptance(test_rows, thresholds, scores, coverages, pred_key, high_error_ppm):
    out = []
    full_by_group = {g: metric(rows, pred_key) for g, rows in group_rows(test_rows).items()}
    for score in scores:
        for cov in coverages:
            annotated = []
            for row in test_rows:
                r = dict(row)
                risk = risk_value(r, score, pred_key)
                thr = threshold_for(r, thresholds, score, cov)
                r['dual_risk_score'] = risk
                r['dual_risk_threshold'] = thr
                r['dual_accepted'] = int(np.isfinite(risk) and risk <= thr)
                r['dual_abs_error'] = abs(fnum(r.get(pred_key)) - fnum(r.get('true_ppm')))
                annotated.append(r)
            for group, rows in group_rows(annotated).items():
                accepted = [r for r in rows if inum(r.get('dual_accepted')) == 1]
                rejected = [r for r in rows if inum(r.get('dual_accepted')) == 0]
                base = full_by_group[group]
                acc = metric(accepted, pred_key)
                rej = metric(rejected, pred_key)
                rejected_high = [r for r in rejected if fnum(r.get('dual_abs_error')) >= high_error_ppm]
                full_high = [r for r in rows if fnum(r.get('dual_abs_error')) >= high_error_ppm]
                out.append({'score': score, 'target_coverage': float(cov), 'group': group,
                            'accepted_n': acc['n'], 'total_n': len(rows),
                            'accepted_coverage': acc['n'] / max(1, len(rows)),
                            'full_R2': base['R2'], 'accepted_R2': acc['R2'],
                            'full_MAE': base['MAE'], 'accepted_MAE': acc['MAE'],
                            'full_P90AE': base['P90AE'], 'accepted_P90AE': acc['P90AE'],
                            'full_P95AE': base['P95AE'], 'accepted_P95AE': acc['P95AE'],
                            'rejected_n': len(rejected), 'rejected_MAE': rej['MAE'],
                            'rejected_P90AE': rej['P90AE'],
                            'full_high_error_rate': len(full_high) / max(1, len(rows)),
                            'rejected_high_error_rate': len(rejected_high) / max(1, len(rejected)),
                            **delta(base, acc)})
    return out


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        return '{:.4f}'.format(float(value))
    return str(value)


def best_rows(curve_rows, coverage):
    out = []
    groups = sorted({r['group'] for r in curve_rows if r['group'] == 'ALL' or '_' not in r['group']},
                    key=lambda x: -1 if x == 'ALL' else client_sort_key(x))
    for group in groups:
        rows = [r for r in curve_rows if r['group'] == group and abs(fnum(r.get('target_coverage')) - coverage) < 1e-9]
        if rows:
            out.append(min(rows, key=lambda r: fnum(r.get('accepted_P90AE'), np.inf)))
    return out


def select_workpoints_from_calibration(calib_curve_rows, test_curve_rows, select_coverages):
    selected = []
    groups = sorted({r['group'] for r in calib_curve_rows if r['group'] == 'ALL' or '_' not in r['group']},
                    key=lambda x: -1 if x == 'ALL' else client_sort_key(x))
    for coverage in select_coverages:
        for group in groups:
            candidates = [r for r in calib_curve_rows
                          if r['group'] == group and abs(fnum(r.get('target_coverage')) - coverage) < 1e-9]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda r: (fnum(r.get('accepted_P90AE'), np.inf),
                                                    fnum(r.get('accepted_MAE'), np.inf),
                                                    -fnum(r.get('accepted_R2'), -np.inf)))
            matches = [r for r in test_curve_rows
                       if r['group'] == group
                       and r['score'] == chosen['score']
                       and abs(fnum(r.get('target_coverage')) - coverage) < 1e-9]
            if not matches:
                continue
            test = matches[0]
            selected.append({
                'group': group,
                'target_coverage': coverage,
                'selected_score': chosen['score'],
                'calib_accepted_coverage': chosen['accepted_coverage'],
                'calib_R2': chosen['accepted_R2'],
                'calib_MAE': chosen['accepted_MAE'],
                'calib_P90AE': chosen['accepted_P90AE'],
                'test_accepted_coverage': test['accepted_coverage'],
                'test_full_R2': test['full_R2'],
                'test_accepted_R2': test['accepted_R2'],
                'test_delta_R2': test['delta_R2'],
                'test_full_MAE': test['full_MAE'],
                'test_accepted_MAE': test['accepted_MAE'],
                'test_delta_MAE': test['delta_MAE'],
                'test_full_P90AE': test['full_P90AE'],
                'test_accepted_P90AE': test['accepted_P90AE'],
                'test_delta_P90AE': test['delta_P90AE'],
                'test_rejected_n': test['rejected_n'],
                'test_rejected_high_error_rate': test['rejected_high_error_rate'],
            })
    return selected


def write_markdown(path, baseline_rows, curve_rows, threshold_rows, selected_rows, args):
    lines = ['# Dual-Output QC Evaluation', '']
    lines.append(f'- calibration_csv: `{args.calibration_csv}`')
    lines.append(f'- test_csv: `{args.test_csv}`')
    lines.append(f'- pred_key: `{args.pred_key}`')
    lines.append('- thresholds are selected on calibration risk distributions only')
    lines.append('- protocol: single-window output; no file-level aggregation')
    lines.append('')
    cols = ['group', 'n', 'R2', 'MAE', 'P90AE', 'P95AE', 'class_acc']
    lines.append('## Full-Coverage Baseline')
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
    for row in baseline_rows:
        if row['group'] == 'ALL' or '_' not in row['group']:
            lines.append('| ' + ' | '.join(fmt(row.get(c)) for c in cols) + ' |')
    lines.append('')
    sel_cols = ['group', 'target_coverage', 'selected_score', 'test_accepted_coverage',
                'test_full_R2', 'test_accepted_R2', 'test_delta_R2',
                'test_full_MAE', 'test_accepted_MAE', 'test_delta_MAE',
                'test_full_P90AE', 'test_accepted_P90AE', 'test_delta_P90AE',
                'test_rejected_n', 'test_rejected_high_error_rate']
    lines.append('## Calibration-Selected Frozen Test Workpoints')
    lines.append('| ' + ' | '.join(sel_cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(sel_cols)) + '|')
    for row in selected_rows:
        lines.append('| ' + ' | '.join(fmt(row.get(c)) for c in sel_cols) + ' |')
    lines.append('')
    wp_cols = ['group', 'score', 'accepted_coverage', 'accepted_R2', 'delta_R2',
               'accepted_MAE', 'delta_MAE', 'accepted_P90AE', 'delta_P90AE',
               'rejected_n', 'rejected_high_error_rate']
    for cov in [0.95, 0.90, 0.80]:
        lines.append('## Best {}% Accepted Workpoints'.format(int(round(cov * 100))))
        lines.append('| ' + ' | '.join(wp_cols) + ' |')
        lines.append('|' + '|'.join(['---'] * len(wp_cols)) + '|')
        for row in best_rows(curve_rows, cov):
            lines.append('| ' + ' | '.join(fmt(row.get(c)) for c in wp_cols) + ' |')
        lines.append('')
    lines.append('## Interpretation')
    lines.append('- Full-coverage metrics answer: how good is the system when every window must return a value?')
    lines.append('- Accepted metrics answer: how good is the system when high-risk windows are flagged for retest or warning?')
    lines.append('- The calibration-selected table is the strict paper-facing result; the best-workpoint tables are diagnostic views.')
    lines.append('- A useful deployment risk score should lower accepted MAE/P90AE and enrich high-error samples in the rejected set.')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_curves(curve_rows, out_dir):
    out_dir = Path(out_dir)
    groups = sorted({r['group'] for r in curve_rows if r['group'] == 'ALL' or '_' not in r['group']},
                    key=lambda x: -1 if x == 'ALL' else client_sort_key(x))
    scores = sorted({r['score'] for r in curve_rows})
    for metric_name, ylabel in [('accepted_MAE', 'Accepted MAE'), ('accepted_P90AE', 'Accepted P90AE')]:
        fig, axes = plt.subplots(1, len(groups), figsize=(3.9 * len(groups), 3.5), sharex=True)
        if len(groups) == 1:
            axes = [axes]
        for ax, group in zip(axes, groups):
            for score in scores:
                rows = sorted([r for r in curve_rows if r['group'] == group and r['score'] == score],
                              key=lambda r: r['accepted_coverage'])
                if not rows:
                    continue
                ax.plot([r['accepted_coverage'] for r in rows], [r[metric_name] for r in rows],
                        marker='o', linewidth=1.4, markersize=3.2, label=score)
            ax.set_title(group)
            ax.set_xlabel('Accepted coverage')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / 'dual_output_{}.png'.format(metric_name), dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Dual-output QC evaluation for single-window deployment')
    parser.add_argument('--calibration_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--pred_key', default='t5_pred_ppm')
    parser.add_argument('--scores', default='composite_response_risk,response_mean_conc_gap_norm,class_response_margin_risk,route_response_risk,corrected_abs_delta,corrected_composite')
    parser.add_argument('--coverages', default='1.0,0.95,0.9,0.85,0.8,0.7')
    parser.add_argument('--select_coverages', default='0.95,0.9,0.8')
    parser.add_argument('--threshold_group', choices=['all', 'client', 'client_and_all'], default='client')
    parser.add_argument('--high_error_ppm', type=float, default=40.0)
    args = parser.parse_args()

    calib_rows = read_csv(args.calibration_csv)
    test_rows = read_csv(args.test_csv)
    scores = parse_list(args.scores, str)
    coverages = parse_list(args.coverages, float)
    select_coverages = parse_list(args.select_coverages, float)
    thresholds, threshold_rows = build_thresholds(calib_rows, scores, coverages, args.pred_key, args.threshold_group)
    calib_curve_rows = evaluate_acceptance(calib_rows, thresholds, scores, coverages, args.pred_key, args.high_error_ppm)
    curve_rows = evaluate_acceptance(test_rows, thresholds, scores, coverages, args.pred_key, args.high_error_ppm)
    selected_rows = select_workpoints_from_calibration(calib_curve_rows, curve_rows, select_coverages)
    baseline_rows = []
    for group, rows in group_rows(test_rows).items():
        baseline_rows.append({'group': group, **metric(rows, args.pred_key)})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'dual_output_thresholds.csv', threshold_rows)
    write_csv(out_dir / 'dual_output_calibration_curves.csv', calib_curve_rows)
    write_csv(out_dir / 'dual_output_curves.csv', curve_rows)
    write_csv(out_dir / 'dual_output_selected_workpoints.csv', selected_rows)
    write_csv(out_dir / 'dual_output_baseline.csv', baseline_rows)
    payload = {'settings': vars(args), 'baseline': baseline_rows,
               'selected_workpoints': selected_rows,
               'best_90pct': best_rows(curve_rows, 0.90)}
    (out_dir / 'dual_output_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    write_markdown(out_dir / 'dual_output_summary.md', baseline_rows, curve_rows, threshold_rows, selected_rows, args)
    plot_curves(curve_rows, out_dir)
    print((out_dir / 'dual_output_summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
