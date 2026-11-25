#!/usr/bin/env python3
"""Smoke test only: a small balanced subset of the real data, shortest clips first, in the HF layout."""
import collections, json, os, sys, urllib.request as U
from concurrent.futures import ThreadPoolExecutor
split_dir, hf_meta, out = sys.argv[1:4]
B = 'https://huggingface.co/datasets/hding2455/TACTUS/resolve/main/'
def pick(d, per):
    by = collections.defaultdict(list)
    for u, e in d.items(): by[e['tool'] + e['action']].append((e['end'] - e['start'], u))
    return {u: d[u] for k, v in by.items() for _, u in sorted(v)[:per]}
os.makedirs(f'{out}/split', exist_ok=True); sets = {}
for s, per in (('train', 8), ('val', 3), ('test', 3)):
    sets[s] = pick(json.load(open(f'{split_dir}/index_gold_{s}.json')), per); json.dump(sets[s], open(f'{out}/split/index_gold_{s}.json', 'w'))
sil = pick(json.load(open(f'{hf_meta}/index_silver.json')), 3); json.dump(sil, open(f'{out}/index_silver.json', 'w'))
files = [e[k] for d in list(sets.values()) + [sil] for e in d.values() for k in ('clip_dir', 'anno_dir', 'prompt_dir')]
def get(f):
    p = f'{out}/hf/{f}'; os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p): U.urlretrieve(B + f, p)
with ThreadPoolExecutor(8) as ex: list(ex.map(get, files))
print({k: len(v) for k, v in sets.items()}, 'silver', len(sil), 'files', len(files))
