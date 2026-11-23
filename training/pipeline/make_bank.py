#!/usr/bin/env python3
"""Pixel refiner's error bank: stage-1 errors on Val, [N,3,2] px @384. Only frames with all three points visible."""
import glob, sys, numpy as np
out = []
for f in sorted(glob.glob(f'{sys.argv[1]}/*.npz')):
    z = np.load(f); e = (z['pred_xy'] - z['gt_xy']) * 384.0; v = z['gt_vis'] > 0.5
    out += [e[t] for t in range(e.shape[0]) if v[t].all()]
out = np.stack(out).astype(np.float32); np.save(sys.argv[2], out); print('bank', out.shape, 'mean err px', float(np.linalg.norm(out, axis=-1).mean()))
