"""Fixed endpoints, seed42 cross-device table and matched CLASS trajectories."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from gaps_flower.fedongas_extension import *
from scripts.run_fedongas_regression_cross import read_json,read_csv,write_json,write_csv,write_text,sha,assert_hashes


def all_endpoint_gate(root,commit):
    from scripts.evaluate_fedongas27 import completion_gate
    return completion_gate(root,classification_matrix(),commit)


def paired_device_rows(rows):
    by={(r['target'],r['method']):r for r in rows}
    return [{'target':t,'seed':42,'comparison':a+'-'+b,
             **{m+'_delta':float(by[t,a][m])-float(by[t,b][m]) for m in ('accuracy','macro_f1')}}
            for t in ('C3','C4','C5') for a,b in (('FT','TO'),('CLASS','TO'),('CLASS','FT'))]


def plot_class_curve(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.5,2.65),layout='constrained')
    for t,color,marker in [('C3','#0072B2','o'),('C4','#D55E00','s'),('C5','#009E73','^')]:
        r=[r for r in rows if r['target']==t]
        ax.plot([r['round'] for r in r],[100*r['accuracy'] for r in r],label=t,color=color,marker=marker,
            markersize=2.5,linewidth=1.1,markerfacecolor='white')
    ax.set(xlabel='Communication round',ylabel='Target-test accuracy (%)',xlim=(.5,25.5),ylim=(0,101))
    ax.set_xticks([1,5,10,15,20,25]);ax.legend(frameon=False,ncol=3,loc='lower right');ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'CLASS_accuracy_vs_round.svg')
    fig.savefig(out/'CLASS_accuracy_vs_round.pdf')
    fig.savefig(out/'CLASS_accuracy_vs_round.png',dpi=600)
    plt.close(fig)


def curves(final_rows,all_predictions,freeze):
    from scripts.summarize_iotj_classification_ablation import evaluate_checkpoint_stream
    curve=CLASS_STUDY/'class_round_curves';curve.mkdir(exist_ok=True)
    rows=[];windows=[];hashes={}
    for t in ('C3','C4','C5'):
        runid=f'XVAL-{t}-CLASS-MAIN-S42' if t!='C5' else 'FONG-CLASS-B20-S42'
        d=CLASS_STUDY/'runs'/runid if t!='C5' else OLD/'runs'/runid
        marker=read_json(d/'FINAL_ENDPOINT.json')
        if marker['run']['method']!='CLASS':raise RuntimeError('Curve is not CLASS')
        endpoint=next(r for r in final_rows if r['target']==t and r['method']=='CLASS')
        for round_id in range(1,26):
            cp=d/'remote_server'/f'server_round_{round_id:03d}_adapted.pth';hashes[cp.relative_to(ROOT).as_posix()]=sha(cp)
            cache=curve/f'{t}_round{round_id:02d}.json';samples=curve/f'{t}_round{round_id:02d}_predictions.csv'
            if cache.exists():
                saved=read_json(cache)
                if saved['checkpoint_sha256']!=sha(cp) or saved['prediction_sha256']!=sha(samples):raise RuntimeError('Trajectory cache changed')
                point=saved['point'];rr=read_csv(samples)
            else:
                if round_id==25:
                    if sha(cp)!=marker['checkpoint_sha256']:raise RuntimeError('Curve endpoint differs from final checkpoint')
                    rr=[{'sample_index':r['window_index'],'true_class':r['true_class'],'pred_class':r['predicted_class']}
                        for r in all_predictions if r['experiment_id']==runid]
                    metrics=endpoint
                else:
                    rr,metrics=evaluate_checkpoint_stream(cp,data_root=DATA,target_client=int(t[1:]),split='test',
                        device=torch.device('cpu'),batch_size=32,expected_endpoint=('round',round_id))
                point={'target':t,'method':'CLASS','seed':42,'round':round_id,'n':len(rr),
                       'accuracy':float(metrics['accuracy']),'macro_f1':float(metrics['macro_f1']),
                       'checkpoint_sha256':sha(cp),'analysis':'posthoc_no_selection'}
                write_csv(samples,[{k:r[k] for k in ('sample_index','true_class','pred_class')} for r in rr])
                write_json(cache,{'point':point,'checkpoint_sha256':sha(cp),'prediction_sha256':sha(samples)})
            expected=2677 if t=='C3' else 1360
            if point['n']!=expected:raise RuntimeError('Trajectory sample count differs')
            rows.append(point)
            windows += [{'target':t,'round':round_id,**r} for r in rr]
    write_csv(curve/'round_accuracy.csv',rows);write_csv(curve/'round_predictions.csv',windows)
    write_json(curve/'curve_audit.json',{'method':'CLASS','seed':42,'all_75_points':True,'round25_reuses_fixed_endpoint_predictions':True,
        'training_performed':False,'checkpoint_selection':False,'input_checkpoint_sha256':hashes,
        'note':'Not the historical A4 curve; single-seed descriptive trajectory; no CI or early stopping'})
    plot_class_curve(rows,curve)
    write_text(curve/'CAPTION.md','CLASS post-adaptation classification accuracy across25communication rounds for C3/C4/C5, canonical main calibration, seed42. All raw points shown; no smoothing, error bars, test-based selection or early stopping. Round25 reuses locked formal endpoint predictions. Curves are not independent replicates.\n')
    return rows


def evaluate():
    freeze=read_json(CLASS_STUDY/'PRE_RUN_FREEZE.json');commit=freeze['freeze_commit']
    endpoints=all_endpoint_gate(CLASS_STUDY,commit);assert_hashes(freeze['input_hashes'])
    out=CLASS_STUDY/'evaluation'
    if (out/'COMPLETE.json').exists():
        index=read_json(out/'sha256_index.json')
        if any(sha(out/n)!=h for n,h in index.items()):raise RuntimeError('Completed classification evidence changed')
        print('Existing complete evaluation verified; not repeated',flush=True);return
    out.mkdir(exist_ok=True)
    write_json(out/'SEALED_TEST_OPEN.json',{'all_six_fixed_endpoints_pass':True,'freeze_commit':commit,'selection':False})
    from scripts.run_iotj_c5_labeled_conditional_da import checkpoint_model,predict_probabilities,classification_metrics
    from federated_dataset import create_client_test_only_loader
    torch.set_num_threads(4)
    target_rows=[];per_class=[];predictions=[]
    for row,directory,marker,cp in endpoints:
        runid=row['experiment_id'];cache=out/(runid+'_COMPLETE.json');predfile=out/(runid+'_predictions.csv')
        if cache.exists():
            cached=read_json(cache)
            if cached['checkpoint_sha256']!=sha(cp) or cached['prediction_sha256']!=sha(predfile):raise RuntimeError('Cached endpoint evaluation changed')
            values=cached['values'];metrics=read_json(out/(runid+'_metrics.json'));rr=read_csv(predfile)
        else:
            loader=create_client_test_only_loader(DATA/f'client_{row["target"][1:]}',batch_size=32,num_workers=0)
            model,_=checkpoint_model(cp,torch.device('cpu'));probs,labels=predict_probabilities(model,loader,torch.device('cpu'))
            if len(labels)!=row['test_n']:raise RuntimeError('Wrong target test population')
            metrics=classification_metrics(probs,labels)
            values={**row,**{k:metrics[k] for k in ('accuracy','macro_f1','nll','ece','num_examples')},
                    'checkpoint_sha256':sha(cp),'calculation_status':'new_fixed_endpoint','wall_seconds':marker['wall_seconds']}
            rr=[{'experiment_id':runid,'target':row['target'],'window_index':i,'true_class':int(y),'predicted_class':int(np.argmax(prob)),
                  **{f'prob_{g}':float(prob[g]) for g in range(4)}} for i,(y,prob) in enumerate(zip(labels,probs))]
            write_json(out/(runid+'_metrics.json'),metrics);write_csv(predfile,rr)
            write_json(cache,{'values':values,'checkpoint_sha256':sha(cp),'prediction_sha256':sha(predfile)})
        target_rows.append(values);predictions+=rr;per_class += [{**row,**v} for v in metrics['per_class']]
        print('EVALUATED',runid,values['accuracy'],values['macro_f1'],flush=True)
    combined=target_rows[:]
    for r in read_csv(OLD/'evaluation/target_results_24.csv'):
        if r['method'] in ('TO','FT','CLASS') and r['seed']=='42' and r['budget_pct']=='20':
            combined.append({**r,'target':'C5','calculation_status':'reported_reused_C5_no_retraining'})
    old_predictions=read_csv(OLD/'evaluation/predictions.csv')
    write_csv(out/'classification_results_6.csv',target_rows);write_csv(out/'cross_device_seed42_results_9.csv',combined)
    write_csv(out/'per_class_metrics.csv',per_class);write_csv(out/'predictions.csv',predictions)
    pairs=paired_device_rows(combined);write_csv(out/'paired_device_differences_9.csv',pairs)
    curve_rows=curves(combined,predictions+old_predictions,freeze)
    lines=['# C3/C4最小跨设备分类验证','',
        'CLASS为主候选；TO/FT/CLASS均seed42、固定2500目标更新，CLASS另有25轮LE1源联邦训练。C5主标定seed42只复用。',
        '每目标原canonical主标定split；C3 calibration678/test2677，C4/C5 calibration320/test1360。单seed无SD/CI/显著性。','',
        '|目标|方法|Accuracy %|Macro-F1 %|来源|','|---|---|---:|---:|---|']
    for r in combined:lines.append(f"|{r['target']}|{r['method']}|{100*float(r['accuracy']):.5f}|{100*float(r['macro_f1']):.5f}|{r['calculation_status']}|")
    lines+=['','## 方法差值（百分点）','']
    for r in pairs:lines.append(f"- {r['target']} {r['comparison']}: ΔAcc={100*r['accuracy_delta']:+.5f}, ΔMacro-F1={100*r['macro_f1_delta']:+.5f}")
    lines+=['','所有正/零/负收益保留；不会因test结果改变CLASS主候选、增加FULL或改训练预算。',
        '匹配的CLASS75个逐轮点另存class_round_curves；旧A4曲线保持历史归属。回归主指标为独立的Oracle_ALL，不能将分类差异归因于共享Ridge映射改变。']
    write_text(out/'RESULT_ANALYSIS.md','\n'.join(lines)+'\n')
    assert_hashes(freeze['input_hashes'])
    write_json(out/'sha256_index.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out/'COMPLETE.json',{'status':'PASS','new_classification_endpoints':6,'cross_device_rows':9,'CLASS_curve_points':len(curve_rows),
        'index_sha256':sha(out/'sha256_index.json'),'freeze_commit':commit})
    print('CLASSIFICATION_AND_CURVES_COMPLETE',flush=True)


if __name__=='__main__':evaluate()
