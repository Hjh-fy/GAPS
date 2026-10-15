"""All-27-endpoint release, target results and seed-paired descriptive summaries."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from gaps_flower.fedongas27_protocol import STUDY, DATA, run_matrix
from scripts.prepare_fedongas27 import sha, write_json, write_csv


def completion_gate(study, matrix, freeze_commit):
    complete=[]
    for row in matrix:
        directory=Path(study)/'runs'/row['experiment_id']
        marker=directory/'FINAL_ENDPOINT.json'
        if not marker.is_file(): raise RuntimeError(f'missing endpoint: {row["experiment_id"]}')
        item=json.loads(marker.read_text())
        if item['run']!=row or item['freeze_commit']!=freeze_commit:
            raise RuntimeError('Endpoint identity/protocol differs')
        if item.get('status')!='PASS': raise RuntimeError('Endpoint status is not PASS')
        if item.get('target_updates')!=row['target_updates'] or item.get('target_test_opened') is not False:
            raise RuntimeError('Endpoint update count/test access differs')
        cp=directory/item['checkpoint']
        if sha(cp)!=item['checkpoint_sha256']: raise RuntimeError('Endpoint checkpoint hash differs')
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        if row['method'] in ('TO','FT'):
            if payload.get('updates')!=2500 or payload.get('seed')!=row['seed'] or payload.get('method')!=row['method']:
                raise RuntimeError('Endpoint is not matching update2500')
        elif payload.get('round')!=25:
            raise RuntimeError('Endpoint is not round25')
        from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
        if ordered_state_content_fingerprint(payload['model_state'])!=item['state_fingerprint']:
            raise RuntimeError('Endpoint state-content fingerprint differs')
        complete.append((row,directory,item,cp))
    return complete


def paired_comparisons(rows):
    by={(int(r['budget_pct']),int(r['seed']),r['method']):r for r in rows}
    comparisons=(('FT','TO'),('FULL','FT'),('FULL','CLASS'))
    pairs,summaries=[],[]
    for budget in (5,20):
        for left,right in comparisons:
            values=[]
            for seed in (42,43,44):
                a,b=by[(budget,seed,left)],by[(budget,seed,right)]
                item={'budget_pct':budget,'seed':seed,'comparison':f'{left}-{right}',
                      'accuracy_delta':float(a['accuracy'])-float(b['accuracy']),
                      'macro_f1_delta':float(a['macro_f1'])-float(b['macro_f1'])}
                pairs.append(item); values.append(item)
            summary={'budget_pct':budget,'comparison':f'{left}-{right}','n_seeds':3}
            for metric in ('accuracy_delta','macro_f1_delta'):
                arr=np.array([v[metric] for v in values])
                summary[metric+'_mean']=float(arr.mean())
                summary[metric+'_sd']=float(arr.std(ddof=1))
            summaries.append(summary)
    return pairs,summaries


def evaluate():
    freeze=json.loads((STUDY/'PRE_RUN_FREEZE.json').read_text())
    endpoints=completion_gate(STUDY,run_matrix(),freeze['freeze_commit'])
    out=STUDY/'evaluation'
    if out.exists(): raise FileExistsError('Preserve one-time existing evaluation')
    for rel,h in freeze['input_hashes'].items():
        if sha(ROOT/rel)!=h: raise RuntimeError(f'Frozen input changed: {rel}')
    out.mkdir()
    write_json(out/'SEALED_TEST_OPEN.json',{'all_27_complete':True,'selection':False,'freeze_commit':freeze['freeze_commit']})
    from scripts.run_iotj_c5_labeled_conditional_da import checkpoint_model, predict_probabilities, classification_metrics
    from federated_dataset import create_client_test_only_loader
    loader=create_client_test_only_loader(DATA/'client_5',batch_size=32,num_workers=0)
    targets,source,per_class,predictions=[],[],[],[]
    for row,directory,item,cp in endpoints:
        model,_=checkpoint_model(cp,torch.device('cpu'))
        probabilities,labels=predict_probabilities(model,loader,torch.device('cpu'))
        metrics=classification_metrics(probabilities,labels)
        if len(labels)!=1360: raise RuntimeError('Frozen test population differs')
        values={**row,**{k:metrics[k] for k in ('accuracy','macro_f1','nll','ece','num_examples')},
                'wall_seconds':item['wall_seconds'],'checkpoint_sha256':item['checkpoint_sha256']}
        (source if row['method']=='SRC' else targets).append(values)
        write_json(out/f'{row["experiment_id"]}_metrics.json',metrics)
        for classrow in metrics['per_class']: per_class.append({**row,**classrow})
        for i,(y,p) in enumerate(zip(labels,probabilities)):
            predictions.append({'experiment_id':row['experiment_id'],'window_index':i,'true_class':int(y),
                                'predicted_class':int(np.argmax(p)),**{f'prob_{k}':float(p[k]) for k in range(4)}})
    write_csv(out/'target_results_24.csv',targets)
    write_csv(out/'source_results_3.csv',source)
    write_csv(out/'per_class_metrics.csv',per_class)
    write_csv(out/'predictions.csv',predictions)
    means=[]
    for method in ('TO','FT','CLASS','FULL'):
        for budget in (5,20):
            group=[r for r in targets if r['method']==method and r['budget_pct']==budget]
            mean={'method':method,'budget_pct':budget,'n_seeds':len(group)}
            for metric in ('accuracy','macro_f1','nll','ece'):
                arr=np.array([g[metric] for g in group]);mean[metric+'_mean']=float(arr.mean());mean[metric+'_sd']=float(arr.std(ddof=1))
            means.append(mean)
    pairs,summaries=paired_comparisons(targets)
    write_csv(out/'method_budget_summary_8.csv',means)
    write_csv(out/'paired_deltas_18.csv',pairs)
    write_csv(out/'paired_summary_6.csv',summaries)
    lines=['# FedOnGas C5 预算与三种子结果','',
           '固定window-level protocol；以下SD为3个训练seed的sample SD（ddof=1），不代表exposure泛化或显著性。','',
           '|Method|Budget|Accuracy mean±SD (%)|Macro-F1 mean±SD (%)|','|---|---:|---:|---:|']
    for r in means: lines.append(f"|{r['method']}|{r['budget_pct']}%|{100*r['accuracy_mean']:.4f}±{100*r['accuracy_sd']:.4f}|{100*r['macro_f1_mean']:.4f}±{100*r['macro_f1_sd']:.4f}|")
    lines.extend(['','## Seed配对差值','', '|Comparison|Budget|ΔAcc mean±SD (pp)|ΔF1 mean±SD (pp)|','|---|---:|---:|---:|'])
    for r in summaries: lines.append(f"|{r['comparison']}|{r['budget_pct']}%|{100*r['accuracy_delta_mean']:+.4f}±{100*r['accuracy_delta_sd']:.4f}|{100*r['macro_f1_delta_mean']:+.4f}±{100*r['macro_f1_delta_sd']:.4f}|")
    lines.extend(['','零收益和负收益全部保留。FULL−CLASS是历史source阶段组织对late目标的比较，不能扩展为target跨阶段泛化。',
                  'TO/FT仅目标CE；CLASS/FULL interleaved feedback预算固定2500 updates，source FL均25 rounds。'])
    (out/'RESULT_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'sha256_index.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write_json(STUDY/'FINAL_EXPERIMENT_STATUS.json',{'status':'TRAINING_EVALUATION_COMPLETE','trained_runs':27,
               'target_rows':len(targets),'summary_rows':len(means),'paired_rows':len(pairs),'test_selection':False})


if __name__=='__main__': evaluate()
