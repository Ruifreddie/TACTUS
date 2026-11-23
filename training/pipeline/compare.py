#!/usr/bin/env python3
"""Final table, Experiment A vs B on the same Test clips. Keypoint ADE gets a 95% interval from a paired bootstrap over videos."""
import glob, json, os, sys, collections, numpy as np
A, B, out = sys.argv[1:4]
J = lambda r, f: json.load(open(f'{r}/results/{f}')) if os.path.exists(f'{r}/results/{f}') else {}
def flat(d, pre=''):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, f'{pre}{k}.'))
        elif isinstance(v, (int, float)): o[pre + k] = float(v)
    return o
rows = []
for name, f, keys in (('Tool / action / combo accuracy', 'cls_test.json', ('metrics.tool_acc', 'metrics.action_acc', 'metrics.combo_acc')), ('Interval tIoU', 'interval_test.json', ('metrics.range_tiou',)),
                      ('Contact vs human', 'contact_human_test.json', ('summary.region_hull_dice.mean',)), ('Contact vs hindsight', 'contact_hindsight_test.json', ('dice_mean',))):
    a, b = flat(J(A, f)), flat(J(B, f))
    rows += [(k.replace('metrics.', '').replace('summary.', ''), name, a.get(k), b.get(k)) for k in keys]
def per_clip(r):
    d = {}
    for f in glob.glob(f'{r}/results/pipeline_test/*.npz'):
        z = np.load(f); key = 'pred_xy' if 'pred_xy' in z.files else None
        if key and 'gt_xy' in z.files and 'gt_vis' in z.files:
            v = z['gt_vis'] > .5
            if v.any(): d[os.path.basename(f)[:-4]] = float(np.linalg.norm((z[key] - z['gt_xy']) * 384, axis=-1)[v].mean())
    return d
pa, pb = per_clip(A), per_clip(B); ci = ''
meta = {m['sid']: m['case'] for m in json.load(open(f'{A}/data/case32/v1/meta.json'))}; common = sorted(set(pa) & set(pb))
if common:
    vids = collections.defaultdict(list)
    for s in common: vids[meta[s]].append(pa[s] - pb[s])
    keys = sorted(vids); r = np.random.RandomState(0); boots = [np.mean([x for k in r.choice(keys, len(keys)) for x in vids[k]]) for _ in range(2000)]
    ci = f'Tool ADE @384: A {np.mean([pa[s] for s in common]):.2f}, B {np.mean([pb[s] for s in common]):.2f}; A minus B {np.mean([pa[s] - pb[s] for s in common]):.2f}, 95% CI [{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}] over {len(keys)} videos, {len(common)} clips.\n'
fmt = lambda v: 'n/a' if v is None else f'{v:.4f}'
t = '| Metric | Group | A: Gold train | B: Gold train + Silver |\n|---|---|---:|---:|\n' + ''.join(f'| {k} | {n} | {fmt(a)} | {fmt(b)} |\n' for k, n, a, b in rows)
os.makedirs(os.path.dirname(out), exist_ok=True); open(out, 'w').write('# Test set, checkpoints picked on Val\n\n' + t + '\n' + ci); print(t + ci)
