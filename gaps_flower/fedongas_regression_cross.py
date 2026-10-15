"""Cached classifier routes crossed with classifier-independent Ridge banks."""
from __future__ import annotations
import numpy as np
from gaps_flower.canonical_r1_v1 import (
    fit_ridge_model, predict_ridge_model, select_grouped_cv_alpha, compute_route_metrics,
)

CLASSIFIERS=('TO','FT','CLASS')
REGRESSORS=('TARGET_RIDGE_83D','R84_CONCAT')
BUDGETS=(5,20)
SEEDS=(42,43,44)
ALPHAS=(0.,.01,.1,1.,10.,100.,1000.)
GASES={0:'Ethanol',1:'CO',2:'Ethylene',3:'Methane'}
RANGES={0:112.5,1:225.,2:112.5,3:225.}
METRICS=('RMSE','MAE','NRMSE_range')


def matrix():
    return [{'experiment_id':f'FONGREG-{c}-{reg}-B{b:02d}-S{s}',
             'classifier':c,'regressor':reg,'budget_pct':b,'seed':s,
             'classifier_run':f'FONG-{c}-B{b:02d}-S{s}',
             'regression_bank_id':f'{reg}-B{b:02d}',
             'calibration_n':80 if b==5 else 320,'test_n':1360,
             'neural_training':False,'source_clients':'C1;C2','target_clients':'C5',
             'split_protocol':'frozen_nested_80_320_calibration__1360_test','QC':'none'}
            for c in CLASSIFIERS for reg in REGRESSORS for b in BUDGETS for s in SEEDS]


def validate_targets(y4,labels,metadata):
    y4=np.asarray(y4,dtype=np.float64); labels=np.asarray(labels,dtype=np.int64)
    if y4.shape!=(len(labels),4) or len(metadata)!=len(labels) or not np.isfinite(y4).all():
        raise ValueError('concentration target shape/finite mismatch')
    if not np.all((labels>=0)&(labels<4)): raise ValueError('target class mismatch')
    truth=y4[np.arange(len(labels)),labels]
    expected=np.array([float(m['concentration']) for m in metadata])
    classes=np.array([int(m['class_id']) for m in metadata])
    if not np.array_equal(classes,labels) or not np.array_equal(truth,expected) or np.any(truth<=0):
        raise ValueError('concentration target/metadata mismatch; classifier placeholders forbidden')
    return truth


def routes_from_cache(rows,runid,truth_classes):
    selected=[r for r in rows if r['experiment_id']==runid]
    selected.sort(key=lambda r:int(r['window_index']))
    if len(selected)!=len(truth_classes) or [int(r['window_index']) for r in selected]!=list(range(len(truth_classes))):
        raise ValueError('Cached window identity/count differs')
    if not np.array_equal([int(r['true_class']) for r in selected],truth_classes):
        raise ValueError('Cached classifier truth order differs')
    route=np.array([int(r['predicted_class']) for r in selected],np.int64)
    if np.any(route<0) or np.any(route>3): raise ValueError('Cached route outside four gases')
    return route


def quantitative_features(windows,phases,metadata,*,split='calibration'):
    from gaps_flower.canonical_quantitative_features import extract_canonical_features
    if len(windows)!=len(phases) or len(windows)!=len(metadata): raise ValueError('Feature row mismatch')
    records=[extract_canonical_features(x,phase=int(p),metadata=m,client='C5',split=split,sample_index=i)
             for i,(x,p,m) in enumerate(zip(windows,phases,metadata))]
    return np.stack([r.sensor83 for r in records]),np.stack([r.h1 for r in records])


def fit_bank(sensor,prior,truth,classes,groups,regressor):
    if regressor not in REGRESSORS: raise ValueError('Unregistered regressor')
    bank,audit={},[]
    for gas in range(4):
        selected=np.asarray(classes)==gas
        x=sensor[selected] if regressor=='TARGET_RIDGE_83D' else np.column_stack([sensor[selected],prior[selected,gas]])
        y=truth[selected]; g=np.asarray(groups)[selected]
        if len(np.unique(g))<5: raise ValueError('Fewer than five calibration file groups')
        cv=select_grouped_cv_alpha(x,y,g,ALPHAS,n_folds=5)
        bank[gas]=fit_ridge_model(x,y,cv.alpha,float(y.min()),float(y.max()))
        audit.append({'gas_id':gas,'alpha':cv.alpha,'fold_by_group':dict(cv.fold_by_group),
                      'pooled_rmse':dict(cv.pooled_rmse),'fit_n':int(selected.sum()),
                      'input_dimensions':int(x.shape[1]),'selection_scope':'calibration_only'})
    return bank,audit


def predict_bank(bank,sensor,prior,regressor):
    return np.column_stack([predict_ridge_model(bank[g] if g in bank else bank[str(g)],
        sensor if regressor=='TARGET_RIDGE_83D' else np.column_stack([sensor,prior[:,g]])) for g in range(4)])


def scope_rows(truth,classes,route,predictions,ranges=RANGES):
    truth,classes,route,predictions=map(np.asarray,(truth,classes,route,predictions))
    def rows(mask):
        return [{'scope':scope,'n':m.n,'RMSE':m.rmse,'MAE':m.mae,'NRMSE_range':m.nrmse_range,
                 'R2':m.r2,'Bias':m.bias}
                for scope,m in compute_route_metrics(truth[mask],classes[mask],route[mask],predictions[mask],ranges).items()]
    overall=rows(np.ones(len(truth),dtype=bool))
    gas=[{'gas_id':int(g),'gas':GASES[int(g)],**r} for g in sorted(set(classes.tolist()))
         for r in rows(classes==g)]
    return overall,gas


def paired_rows(rows):
    by={(int(r['budget_pct']),int(r['seed']),r['classifier'],r['regressor']):r for r in rows}
    contrasts=[(c,'R84_CONCAT',c,'TARGET_RIDGE_83D') for c in CLASSIFIERS]
    contrasts.extend((left,reg,right,reg) for reg in REGRESSORS for left,right in (('FT','TO'),('CLASS','TO'),('CLASS','FT')))
    contrasts.extend((c,'R84_CONCAT','TO','TARGET_RIDGE_83D') for c in ('FT','CLASS'))
    result=[]
    for budget in BUDGETS:
        for left,lr,right,rr in contrasts:
            for seed in SEEDS:
                a,b=by[(budget,seed,left,lr)],by[(budget,seed,right,rr)]
                result.append({'budget_pct':budget,'seed':seed,'comparison':f'{left}:{lr}-{right}:{rr}',
                               **{m+'_delta':float(a[m])-float(b[m]) for m in METRICS}})
    return result


def summarize(rows,keys,metrics=METRICS):
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for group,values in sorted(groups.items()):
        if len(values)!=3 or {int(r['seed']) for r in values}!=set(SEEDS):
            raise ValueError('Summary requires exactly three paired classifier seeds')
        item={**dict(zip(keys,group)),'n_seeds':3}
        for m in metrics:
            arr=np.array([float(r[m]) for r in values])
            item[m+'_mean']=float(arr.mean()); item[m+'_sd']=float(arr.std(ddof=1))
        result.append(item)
    return result
