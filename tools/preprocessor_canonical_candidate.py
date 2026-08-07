"""Candidate canonical preprocessing: raw-observation G0 and physical-time bins.

This diagnostic/candidate module intentionally leaves preprocessor_time_aware.py
unchanged.  It keeps true time, uses only raw observations for G0, aggregates
non-empty physical bins, and only bridges a single missing bin.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import importlib.util, sys
import numpy as np
import pandas as pd

def _ta():
    p=Path(__file__).resolve().parents[3]/'preprocessor_time_aware.py'
    s=importlib.util.spec_from_file_location('canonical_candidate_ta',p);m=importlib.util.module_from_spec(s)
    assert s and s.loader;sys.modules[s.name]=m;s.loader.exec_module(m);return m
TA=_ta()

def max_run(mask:np.ndarray)->int:
    best=cur=0
    for x in mask:
        cur=cur+1 if bool(x) else 0;best=max(best,cur)
    return best

def process_file(path:str|Path, sampling_hz:int, aggregation:str, window_duration_s:int,
                 baseline_stat:str='mean', short_gap_only:bool=True)->dict[str,Any]:
    rt,rx=TA.load_raw_data(path);t,x,dup=TA.clean_time_axis(rt,rx)
    return process_arrays(t,x,TA.parse_filename(Path(path).name),sampling_hz,aggregation,window_duration_s,baseline_stat,short_gap_only,int(dup['duplicate_timestamps']))

def process_arrays(t:np.ndarray,x:np.ndarray,info:dict[str,Any],sampling_hz:int,aggregation:str,window_duration_s:int,
                   baseline_stat:str='mean',short_gap_only:bool=True,duplicate_timestamps:int=0)->dict[str,Any]:
    if sampling_hz not in (1,2,5,10) or aggregation not in ('mean','median') or baseline_stat not in ('mean','median'):
        raise ValueError('unregistered candidate parameter')
    # G0 precedes any resampling and therefore cannot be polluted by interpolation.
    bm=(t>=20.0)&(t<50.0);graw=1.0/np.clip(x,1e-10,None);base_raw=graw[bm]
    if len(base_raw)<3:raise RuntimeError('insufficient raw baseline')
    g0=np.mean(base_raw,axis=0) if baseline_stat=='mean' else np.median(base_raw,axis=0)
    dt=1.0/sampling_hz;start=np.ceil(t[0]*sampling_hz)/sampling_hz;end=np.floor(t[-1]*sampling_hz)/sampling_hz
    grid=np.arange(start,end+dt*.5,dt);bid=np.floor((t-start+1e-9)*sampling_hz).astype(int);keep=(bid>=0)&(bid<len(grid))
    valid_bins=bid[keep];valid_x=x[keep]
    counts=np.bincount(valid_bins,minlength=len(grid)).astype(int)
    values=np.full((len(grid),8),np.nan)
    if aggregation=='mean':
        sums=np.zeros((len(grid),8),float);np.add.at(sums,valid_bins,valid_x)
        np.divide(sums,counts[:,None],out=values,where=counts[:,None]>0)
    else:
        # Pandas groupby median is vectorized in compiled kernels; this avoids
        # a Python loop over ~6,000 physical bins per raw file.
        frame=pd.DataFrame(valid_x);frame['_bin']=valid_bins
        med=frame.groupby('_bin',sort=False).median(numeric_only=True)
        values[med.index.to_numpy(dtype=int)]=med.to_numpy(dtype=float)
    empty=counts==0;filled=np.zeros(len(grid),bool)
    if short_gap_only:
        for i in range(1,len(grid)-1):
            if empty[i] and not empty[i-1] and not empty[i+1]: values[i]=(values[i-1]+values[i+1])/2;filled[i]=True
    rel=(1.0/np.clip(values,1e-10,None)-g0)/g0
    crop=np.where((grid>=60.0)&(grid<=170.0+1e-9))[0];points=round(window_duration_s*sampling_hz);stride=round(points/2)
    windows=[];meta=[]
    for s in range(int(crop[0]),int(crop[-1])-points+2,stride):
        ind=np.arange(s,s+points);w=rel[ind];missing=empty[ind] & ~filled[ind]
        windows.append(w.astype(np.float32));meta.append({'physical_window_start_s':float(grid[ind[0]]),'physical_window_end_s':float(grid[ind[-1]]+dt),'quality_flag':'valid' if not missing.any() else 'invalid_long_gap','valid':bool(not missing.any()),'samples_per_bin_mean':float(np.mean(counts[ind])),'empty_bin_ratio':float(np.mean(empty[ind])),'observed_ratio':float(np.mean(~empty[ind])),'short_gap_interpolated_ratio':float(np.mean(filled[ind])),'max_missing_run':max_run(empty[ind])})
    return {'info':info,'windows':windows,'metadata':meta,'baseline':g0,'baseline_n_raw_samples':int(len(base_raw)),'baseline_duration':30.0,'baseline_channel_std':np.std(base_raw,axis=0),'duplicate_timestamps':int(duplicate_timestamps),'sampling_completeness':float(np.mean(~empty)),'runtime_time_s':len(grid)*0.0}
