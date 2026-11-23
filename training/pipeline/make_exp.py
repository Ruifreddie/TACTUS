#!/usr/bin/env python3
"""Lay out one experiment as a complete UniTAP repo, so the recipes' relative paths work with no edits to their code.

$WORK/exp/<E>/{tools,third_party,experiments,data/raw_case32,data/case32/v1}
v1 links to the shared export. Only the split lists, meta and token links belong to the experiment.
A: train = Gold train. B: train = Gold train + Silver, annotations as published. C: train = Silver only. Val and test are the same Gold clips in all.
"""
import json, os, shutil, sys
E, U, EXPORT, TOK, RAW, DST = sys.argv[1:7]
for d in ('tools', 'third_party'):
    if not os.path.exists(f'{DST}/{d}'): shutil.copytree(f'{U}/{d}', f'{DST}/{d}', symlinks=True)
v1 = f'{DST}/data/case32/v1'; os.makedirs(v1, exist_ok=True); os.makedirs(f'{DST}/experiments/analysis', exist_ok=True)
rawl = f'{DST}/data/raw_case32'
if os.path.lexists(rawl): os.remove(rawl)
os.symlink(RAW, rawl)
META = json.load(open(f'{EXPORT}/meta.json'))
by = {s: [m['sid'] for m in META if m['split'] == s] for s in ('train', 'val', 'test', 'silver')}
train = by['silver'] if E == 'C' else by['train'] + (by['silver'] if E == 'B' else [])
if E in ('B', 'C') and not by['silver']: raise SystemExit(f'experiment {E} needs the Silver export')
lists = {'train': train, 'val': by['val'], 'test': by['test']}
for s, ids in lists.items():
    open(f'{v1}/{s}.txt', 'w').write(''.join(i + '\n' for i in ids))
keep = set(train) | set(by['val']) | set(by['test']); out = []
for m in META:
    if m['sid'] not in keep: continue
    m = dict(m); m['split'] = 'train' if m['split'] == 'silver' else m['split']; out.append(m)
json.dump(out, open(f'{v1}/meta.json', 'w'))
for e in os.listdir(EXPORT):
    if not os.path.isdir(f'{EXPORT}/{e}') or e == 'vjepa_token_cache': continue
    t = f'{v1}/{e}'
    if not os.path.lexists(t): os.symlink(f'{EXPORT}/{e}', t)
shutil.copy(f'{EXPORT}/labels.json', f'{v1}/labels.json')
for s, ids in lists.items():
    d = f'{v1}/vjepa_token_cache/{s}'; os.makedirs(d, exist_ok=True)
    for i in ids:
        t = f'{d}/{i}.npy'
        if not os.path.lexists(t): os.symlink(f'{TOK}/all/{i}.npy', t)
import collections
key = lambda m: next((str(m[k]) for k in ('combo', 'combo_name', 'pair') if k in m), f"{m.get('tool', m.get('tool_name'))}|{m.get('action', m.get('action_name'))}")
pair = {m['sid']: key(m) for m in META}
print(E, {k: len(v) for k, v in lists.items()}, 'train pairs:', dict(collections.Counter(pair.get(i) for i in train)))
