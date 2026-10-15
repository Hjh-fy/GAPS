"""TO/FT in the server environment; receives only target x/class and one init asset."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gaps_flower.fedongas27_methods import train_target_blocks
from gaps_flower.task import make_config, create_model
from gaps_flower.state_fingerprint import ordered_state_content_fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True,choices=('TO','FT'))
    parser.add_argument('--seed',required=True,type=int,choices=(42,43,44))
    parser.add_argument('--data',required=True)
    parser.add_argument('--initial-state',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(f'Preserve existing target endpoint: {out}')
    config=make_config(device='cpu',local_epochs=1,batch_size=32,profile='ce_stats',seed=args.seed)
    model=create_model(config)
    init=torch.load(args.initial_state,map_location='cpu',weights_only=False)
    if args.method=='TO' and (init.get('initialization')!='random' or init['seed']!=args.seed):
        raise RuntimeError('TO requires matching random state, not trained source model')
    if args.method=='FT' and init.get('round')!=25:
        raise RuntimeError('FT requires source round25')
    model.load_state_dict(init['model_state'],strict=True)
    initial_fp=ordered_state_content_fingerprint(model.state_dict())
    directory=Path(args.data)
    x=np.load(directory/'calibration_features.npy',allow_pickle=False)
    y=np.load(directory/'calibration_classification_labels.npy',allow_pickle=False)
    if x.shape[1:]!=(50,8) or len(x) not in (80,320) or len(x)!=len(y):
        raise RuntimeError('Unexpected target-only input')
    loader=DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32)),torch.from_numpy(y.astype(np.int64))),
                      batch_size=32,shuffle=True,num_workers=0)
    out.mkdir(parents=True)
    (out/'effective_config.json').write_text(json.dumps({**vars(args),'initial_fingerprint':initial_fp,
        'fields':['x','class'],'source_ce':False,'optimizer':'Adam','lr':5e-4,'blocks':25,'steps_per_block':100,
        'initial_checkpoint_sha256':hashlib.sha256(Path(args.initial_state).read_bytes()).hexdigest()},indent=2))
    start=time.perf_counter()
    rows=train_target_blocks(model,loader,seed=args.seed)
    with (out/'update_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cp=out/'update_2500.pth'
    torch.save({'model_state':model.state_dict(),'method':args.method,'seed':args.seed,'updates':2500,
                'initial_fingerprint':initial_fp},cp)
    (out/'COMPLETE.json').write_text(json.dumps({'status':'PASS','updates':len(rows),
        'initial_fingerprint':initial_fp,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
        'final_fingerprint':ordered_state_content_fingerprint(model.state_dict()),
        'wall_seconds':time.perf_counter()-start,'test_opened':False},indent=2))
    print(json.dumps({'status':'PASS','method':args.method,'updates':len(rows),'wall_seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':
    main()
