#!/usr/bin/env python3
"""Crop check: one clip per video, through the real exporter. Shows the sampled frame with the crop box, and the frame V-JEPA receives.
usage: audit_crop.py --split-dir --hf-meta --out --code U  (fetches one short clip per video from Hugging Face)"""
import argparse, collections, json, os, subprocess, sys, time, urllib.request as U
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--split-dir', required=True); ap.add_argument('--hf-meta', required=True); ap.add_argument('--out', required=True); ap.add_argument('--code', required=True); a = ap.parse_args()
B = 'https://huggingface.co/datasets/hding2455/surgical-action-affordance/resolve/main/'; out = a.out; os.makedirs(f'{out}/split', exist_ok=True)
STATUS = json.load(open(f'{a.hf_meta}/review_status.json')); N = 12
def pick(d, only=None):
    """Per video: N clips spread over it for the box, the short one among them goes through the exporter. Silver: clips that hold points."""
    by = collections.defaultdict(list)
    for u, e in d.items():
        if only is None or STATUS.get(u, {}).get('status') == only: by[e['task']].append((e['start'], u))
    keep = {}; extra = []
    for t, v in by.items():
        v.sort(); sel = [v[int(i)][1] for i in np.linspace(0, len(v) - 1, min(N, len(v)))]; extra += [d[u] for u in sel]
        c = sorted(sel, key=lambda u: d[u]['end'] - d[u]['start'])[len(sel) // 4]; keep[c] = d[c]
    return keep, extra
sets = {}; more = []
for s in ('train', 'val', 'test'): sets[s], x = pick(json.load(open(f'{a.split_dir}/index_gold_{s}.json'))); more += x
sil, x = pick(json.load(open(f'{a.hf_meta}/index_silver.json')), 'success'); more += x
for s, d in sets.items(): json.dump(d, open(f'{out}/split/index_gold_{s}.json', 'w'))
json.dump(sil, open(f'{out}/index_silver.json', 'w'))
SIZE = {x['rfilename']: x.get('size') for x in json.load(U.urlopen('https://huggingface.co/api/datasets/hding2455/surgical-action-affordance?blobs=true', timeout=300))['siblings']}
def get(f):
    p = f'{out}/hf/{f}'; os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p) == SIZE.get(f): return
    for k in range(6):
        try:
            U.urlretrieve(B + f, p + '.part'); os.replace(p + '.part', p); return
        except OSError: time.sleep(5 + 10 * k)
    raise RuntimeError('download failed: ' + f)
files = [e[k] for d in list(sets.values()) + [sil] for e in d.values() for k in ('clip_dir', 'anno_dir')] + [e['clip_dir'] for e in more]
with ThreadPoolExecutor(8) as ex: list(ex.map(get, files))
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, f'{here}/video_boxes.py', '--raw', f'{out}/hf', '--out', f'{out}/video_boxes.json', '--workers', '6'], check=True)
subprocess.run([sys.executable, f'{here}/export_clips.py', '--boxes', f'{out}/video_boxes.json', '--unitap', a.code, '--raw', f'{out}/hf', '--out', f'{out}/export', '--split-dir', f'{out}/split', '--silver-index', f'{out}/index_silver.json', '--workers', '6'], check=True)
subprocess.run([sys.executable, f'{here}/crop_sheet.py', '--raw', f'{out}/hf', '--export', f'{out}/export', '--boxes', f'{out}/video_boxes.json', '--out', out], check=True)
