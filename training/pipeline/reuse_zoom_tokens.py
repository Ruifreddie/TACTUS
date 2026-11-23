#!/usr/bin/env python3
"""B's Gold train clips get the same zoom boxes as A's (same seed and order), so their tokens are reused. Checked box by box; any mismatch recomputes."""
import json, os, sys
a, b = sys.argv[1:3]; ba = json.load(open(f'{a}/boxes.json'))['boxes']; bb = json.load(open(f'{b}/boxes.json'))['boxes']
os.makedirs(f'{b}/vjepa_token_cache/train', exist_ok=True); n = bad = 0
for i in open(f'{a}/train.txt'):
    i = i.strip()
    if not i: continue
    if ba.get(i) != bb.get(i): bad += 1; continue
    t = f'{b}/vjepa_token_cache/train/{i}.npy'
    if not os.path.lexists(t): os.symlink(os.path.abspath(f'{a}/vjepa_token_cache/train/{i}.npy'), t)
    n += 1
print('reused', n, 'recompute', bad)
