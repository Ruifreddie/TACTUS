#!/usr/bin/env python3
"""Copy what a new machine needs to continue from the local disk (WORK) to persistent storage (GV), and back. Plain copies with no rename, since object storage has no atomic rename.
Saved: step markers, best checkpoints, prediction dumps as one tar, results, logs. save|load E EXP DST"""
import glob, os, shutil, subprocess, sys
op, E, exp, dst = sys.argv[1:5]; dst = os.path.abspath(dst); os.makedirs(dst, exist_ok=True); os.chdir(exp)
if op == 'save':
    keep = [p for pat in ('.done', 'results', 'logs', 'experiments/analysis', 'experiments/runs/*/out/checkpoints/best.pt', 'experiments/runs/*/checkpoints/best.pt', 'experiments/runs/*/pixref/best.pt', 'experiments/runs/*/epoch_metrics.jsonl', 'experiments/runs/*/train.log', 'experiments/runs/*/out/epoch_metrics.jsonl') for p in glob.glob(pat)]
    if keep:
        subprocess.run(['tar', '-cf', f'{dst}/state.tar.new', *keep], check=True)
        shutil.copyfile(f'{dst}/state.tar.new', f'{dst}/state.tar'); os.remove(f'{dst}/state.tar.new')
elif not glob.glob('.done/*'):
    # state.tar.new survives only if the machine died while saving: then one of the two is whole, and tar fails on the truncated one.
    tars = [t for t in (f'{dst}/state.tar', f'{dst}/state.tar.new') if os.path.exists(t)]
    if tars and not any(subprocess.run(['tar', '-xf', t]).returncode == 0 for t in tars): raise SystemExit('no readable state in ' + dst)
    if tars: print('restored', E, 'state from', dst)
    for m in ('zoom_build', 'zoom_tok', 'labels'):      # these rebuild on the new machine from the restored dumps; zoom tokens come back from shards
        if os.path.exists(f'.done/{m}'): os.remove(f'.done/{m}')
