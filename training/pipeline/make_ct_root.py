#!/usr/bin/env python3
"""data/case32/v1_zoom_ct: contact tracks in the tool slots, pt_role 1,1,1,1, same zoom frames and tokens. Used by recipe C004 (human contact regions)."""
import json, os
Z, C = 'data/case32/v1_zoom', 'data/case32/v1_zoom_ct'; os.makedirs(C, exist_ok=True)
pairs = {'gt_tool_xy': 'gt_ct_xy', 'gt_tool_vis': 'gt_ct_vis'}
for e in os.listdir(Z):
    if e in ('meta.json',) or e in pairs: continue
    t = f'{C}/{e}'
    if not os.path.lexists(t): os.symlink(os.path.abspath(f'{Z}/{e}'), t)
for a, b in pairs.items():
    t = f'{C}/{a}'
    if not os.path.lexists(t): os.symlink(os.path.abspath(f'{Z}/{b}'), t)
m = json.load(open(f'{Z}/meta.json'))
for r in m: r['pt_role'] = [1, 1, 1, 1]; r['topology'] = '4C'
json.dump(m, open(f'{C}/meta.json', 'w'))
