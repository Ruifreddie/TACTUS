#!/usr/bin/env python3
"""Fills the persistent directory GV; needs no GPU. Raw data, weights and code go in before any GPU machine is started.
One task at a time: download, check sizes against Hugging Face, tar, copy to the volume, delete. So the local disk only ever holds one video.
Re-run after any interruption; finished tasks have an .ok marker. Object storage has no atomic rename, hence markers rather than renames.
usage: GV=/workspace python fill_volume.py [--workers 8]   optional env: HF_TOKEN (higher Hugging Face rate limit)"""
import argparse, collections, hashlib, os, re, shutil, subprocess, sys, time
import requests
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
GV = os.environ.get('GV', '/workspace'); TMP = os.environ.get('DL', '/root/dl'); REPO = 'hding2455/surgical-action-affordance'; BASE = f'https://huggingface.co/datasets/{REPO}/resolve/main/'
AUTH = {'Authorization': 'Bearer ' + os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}
V_NAME = 'vjepa2_1_vitl_dist_vitG_384.pt'; V_SHA = '7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6'; V_URL = os.environ.get('VJEPA_URL', 'https://dl.fbaipublicfiles.com/vjepa2/' + V_NAME)
os.makedirs(f'{GV}/raw', exist_ok=True); os.makedirs(f'{GV}/weights', exist_ok=True); os.makedirs(TMP, exist_ok=True)
sib = HfApi().dataset_info(REPO, files_metadata=True).siblings; by = collections.defaultdict(list); top = []
for s in sib:
    (by[s.rfilename.split('/')[0]].append(s) if '/' in s.rfilename else top.append(s))
if os.environ.get('ONLY_TASKS'): by = {t: by[t] for t in os.environ['ONLY_TASKS'].split(',')}      # local test only
print('tasks', len(by), 'files', len(sib), 'GB', round(sum(s.size or 0 for s in sib) / 1e9, 1), flush=True)
def get(s, root):
    """One resolver request per file (hf_hub_download spends two), and a rate limit waits for its window instead of counting as a failure."""
    p = f'{root}/{s.rfilename}'
    if os.path.exists(p) and (s.size is None or os.path.getsize(p) == s.size): return
    os.makedirs(os.path.dirname(p), exist_ok=True); fails = 0
    while fails < 12:
        try:
            with requests.get(BASE + s.rfilename, stream=True, timeout=(30, 120), headers=AUTH) as r:
                if r.status_code == 429:
                    w = re.search(r';t=(\d+)', r.headers.get('RateLimit', '')); time.sleep(int(w.group(1)) + 1 if w else 30); continue
                r.raise_for_status()
                with open(p + '.part', 'wb') as f:
                    for c in r.iter_content(1 << 20): f.write(c)
            if s.size is None or os.path.getsize(p + '.part') == s.size: os.replace(p + '.part', p); return
            raise IOError('size differs')
        except Exception as ex:
            fails += 1; print('retry', s.rfilename, repr(ex)[:120], flush=True); time.sleep(min(300, 15 * 2 ** fails))
    raise RuntimeError('cannot fetch ' + s.rfilename)
for s in top:
    if not os.path.exists(f'{GV}/raw/{s.rfilename}.ok') and not s.rfilename.startswith('.'):
        get(s, TMP); shutil.copyfile(f'{TMP}/{s.rfilename}', f'{GV}/raw/{s.rfilename}'); open(f'{GV}/raw/{s.rfilename}.ok', 'w').close()
for n, (t, files) in enumerate(sorted(by.items()), 1):
    if os.path.exists(f'{GV}/raw/{t}.tar.ok'): continue
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(lambda s: get(s, TMP), files))
    tar = f'{TMP}/{t}.tar'; subprocess.run(['tar', '-cf', tar, '-C', TMP, t], check=True); shutil.copyfile(tar, f'{GV}/raw/{t}.tar')
    if os.path.getsize(f'{GV}/raw/{t}.tar') != os.path.getsize(tar): raise SystemExit('bad copy ' + t)
    open(f'{GV}/raw/{t}.tar.ok', 'w').close(); os.remove(tar); shutil.rmtree(f'{TMP}/{t}'); print(f'[{n}/{len(by)}] {t} {len(files)} files', flush=True)
w = f'{GV}/weights/{V_NAME}'
if not os.path.exists(w + '.ok'):
    tmp = f'{TMP}/{V_NAME}'; subprocess.run(['curl', '-fL', '--retry', '5', '-o', tmp, V_URL], check=True); h = hashlib.sha256()
    with open(tmp, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    if h.hexdigest() != V_SHA: raise SystemExit('V-JEPA 2.1 weight hash differs')
    shutil.copyfile(tmp, w); open(w + '.ok', 'w').close(); os.remove(tmp)
if os.path.exists('code.tar'): shutil.copyfile('code.tar', f'{GV}/code.tar')
missing = [t for t in by if not os.path.exists(f'{GV}/raw/{t}.tar.ok')]
if missing or not os.path.exists(f'{GV}/code.tar'): raise SystemExit(f'not ready: missing tasks {missing[:5]}, code.tar present: {os.path.exists(f"{GV}/code.tar")}')
open(f'{GV}/raw/READY', 'w').write(f'{len(by)} tasks, {len(sib)} files\n'); print('volume ready', flush=True)
