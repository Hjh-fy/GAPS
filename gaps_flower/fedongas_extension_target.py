"""C3/C4 TO/FT, identical CE update implementation to the frozen C5 study."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from gaps_flower.fedongas27_methods import train_target_blocks
from gaps_flower.fedongas_extension import target_xy
from gaps_flower.task import make_config,create_model
from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
from scripts.run_fedongas_regression_cross import write_json,write_csv,sha


def validate_initial(payload,method):
    if method=='TO':
        if payload.get('initialization')!='random' or payload.get('seed')!=42:raise RuntimeError('TO requires random seed42')
    elif method=='FT':
        if payload.get('round')!=25:raise RuntimeError('FT requires frozen source round25')
    else:raise RuntimeError('Unregistered target method')


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=('TO','FT'),required=True)
    p.add_argument('--target',choices=('C3','C4'),required=True);p.add_argument('--data',required=True)
    p.add_argument('--initial-state',required=True);p.add_argument('--initial-sha256',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve existing target endpoint')
    if sha(args.initial_state)!=args.initial_sha256:raise RuntimeError('Initial checkpoint hash changed')
    init=torch.load(args.initial_state,map_location='cpu',weights_only=False);validate_initial(init,args.method)
    model=create_model(make_config(device='cpu',local_epochs=1,batch_size=32,profile='ce_stats',seed=42))
    model.load_state_dict(init['model_state'],strict=True);fp=ordered_state_content_fingerprint(model.state_dict())
    x,y=target_xy(args.data,args.target)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=32,shuffle=True,num_workers=0)
    out.mkdir(parents=True);start=time.perf_counter()
    write_json(out/'effective_config.json',{**vars(args),'seed':42,'initial_fingerprint':fp,'fields':['x','class'],
        'source_ce':False,'optimizer':'Adam','lr':5e-4,'blocks':25,'steps_per_block':100})
    rows=train_target_blocks(model,loader,seed=42)
    write_csv(out/'update_diagnostics.csv',rows)
    cp=out/'update_2500.pth';torch.save({'model_state':model.state_dict(),'method':args.method,'target':args.target,
        'seed':42,'updates':2500,'initial_fingerprint':fp},cp)
    write_json(out/'COMPLETE.json',{'status':'PASS','updates':2500,'initial_fingerprint':fp,
        'checkpoint_sha256':sha(cp),'final_fingerprint':ordered_state_content_fingerprint(model.state_dict()),
        'wall_seconds':time.perf_counter()-start,'test_opened':False})
    print('COMPLETE',args.target,args.method,2500,flush=True)


if __name__=='__main__':main()
