"""Build a fail-closed C5 runtime row map from frozen B5 probability signatures."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from gaps_deploy.c5_h8_runtime import C5H8Runtime

EXPECTED_ROWS = 1360
MAX_PROB_DELTA = 5e-5

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def match_probability_signatures(runtime: np.ndarray, reference: np.ndarray) -> list[tuple[int, int, float]]:
    """Return a deterministic bijection or reject ambiguous/unmatched signatures."""
    runtime = np.asarray(runtime, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if runtime.ndim != 2 or runtime.shape != reference.shape or runtime.shape[1] != 4:
        raise ValueError("probability signature arrays must have matching (N,4) shapes")
    if not np.isfinite(runtime).all() or not np.isfinite(reference).all():
        raise ValueError("probability signatures must be finite")
    if len({tuple(row) for row in runtime}) != len(runtime) or len({tuple(row) for row in reference}) != len(reference):
        raise ValueError("duplicate probability signatures are ambiguous")
    distances = np.max(np.abs(runtime[:, None, :] - reference[None, :, :]), axis=2)
    runtime_indices, reference_indices = linear_sum_assignment(distances)
    matches = [
        (int(i), int(j), float(distances[i, j]))
        for i, j in zip(runtime_indices, reference_indices)
    ]
    if len(matches) != len(runtime) or any(delta > MAX_PROB_DELTA for _, _, delta in matches):
        raise ValueError("probability signatures have no bijection within tolerance")
    return matches

def build_row_map(contract: Path, output: Path) -> dict:
    c=json.loads(Path(contract).read_text(encoding='utf8'))
    x_path=Path(c['inputs']['features']['path']); m_path=Path(c['inputs']['metadata']['path']); r_path=Path(c['references']['HC95']['path'])
    x=np.load(x_path); meta=json.loads(m_path.read_text(encoding='utf8'))
    with r_path.open(newline='',encoding='utf8') as f: ref=list(csv.DictReader(f))
    if len(x)!=EXPECTED_ROWS or len(meta)!=EXPECTED_ROWS or len(ref)!=EXPECTED_ROWS: raise ValueError('row count must be 1360')
    rt=C5H8Runtime.from_runtime_contract(contract); logits,_=rt.classify(x)
    p=np.exp(logits-logits.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    groups={}; refs={}
    for i,v in enumerate(meta): groups.setdefault((v['filename'],str(v['repeat_id'])),[]).append(i)
    for i,v in enumerate(ref): refs.setdefault((v['filename'],str(v['repeat_id'])),[]).append(i)
    if set(groups)!=set(refs): raise ValueError('metadata/reference key sets differ')
    rows=[]
    for k,idxs in groups.items():
        js=refs[k]
        if len(idxs)!=len(js): raise ValueError(f'key multiplicity differs: {k}')
        reference_probabilities = np.asarray(
            [[float(ref[j][f'prob_{z}']) for z in range(4)] for j in js], dtype=np.float64
        )
        for ii, q, delta in match_probability_signatures(p[idxs], reference_probabilities):
            rows.append({'runtime_index':idxs[ii],'reference_index':int(ref[js[q]]['sample_index']),'filename':k[0],'repeat_id':k[1],'max_abs_probability_delta':delta})
    if len(rows)!=EXPECTED_ROWS or len({r['runtime_index'] for r in rows})!=EXPECTED_ROWS or len({r['reference_index'] for r in rows})!=EXPECTED_ROWS: raise ValueError('row map is not bijective')
    output=Path(output)
    if output.exists(): raise FileExistsError(f'refusing to overwrite row map: {output}')
    payload={'schema_version':'iotj.c5_h8_row_map.v1','status':'ready','contract_sha256':_sha256(Path(contract)),'row_count':EXPECTED_ROWS,'max_probability_delta':max(r['max_abs_probability_delta'] for r in rows),'rows':sorted(rows,key=lambda r:r['reference_index'])}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    return payload

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--contract',type=Path,required=True); a.add_argument('--output',type=Path,required=True); z=a.parse_args()
    result = build_row_map(z.contract,z.output)
    print(json.dumps({key: result[key] for key in ('schema_version', 'status', 'row_count', 'max_probability_delta')}))
