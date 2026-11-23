"""Evaluate a contact dump (box-normalized pred_xy [16,4,2]) against the HINDSIGHT hotspot labels
(data/case32/v1_zoom_hs/region_mask [16,48,48], box grid): per-frame Dice between the rasterized convex hull of the
predicted 4 points and the hotspot (>0.5), frames with a non-empty hotspot only. Also the centroid-inside rate."""
import argparse, glob, os, json
import numpy as np
from PIL import Image, ImageDraw

def hull_poly(pts):
    P = sorted(set(map(tuple, np.asarray(pts, float).round(4))))
    if len(P) <= 2: return P
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    for q in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    up = []
    for q in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0: up.pop()
        up.append(q)
    return lo[:-1] + up[:-1]

def quad_mask(pts_box, side=48):
    im = Image.new('L', (side, side), 0); d = ImageDraw.Draw(im); h = [(x * side, y * side) for x, y in hull_poly(pts_box)]
    if len(h) >= 3: d.polygon(h, fill=255)
    for x, y in h: d.ellipse([x - 0.75, y - 0.75, x + 0.75, y + 0.75], fill=255)
    return np.asarray(im) > 0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dump', required=True); ap.add_argument('--labels', default='data/case32/v1_zoom_hs/region_mask'); ap.add_argument('--save', default=None); a = ap.parse_args()
    meta = {m['sid']: m for m in json.load(open('data/case32/v1/meta.json'))}
    D, CIN, by = [], [], {}
    for f in sorted(glob.glob(os.path.join(a.dump, '*.npz'))):
        sid = os.path.basename(f)[:-4]; z = np.load(f); rm = np.load(os.path.join(a.labels, f'{sid}.npy'))
        pred = z['pred_xy'].astype(float)
        for t in range(16):
            g = rm[t] > 0.5
            if not g.any(): continue
            p = quad_mask(pred[t]); d = 2 * (p & g).sum() / max(p.sum() + g.sum(), 1); D.append(d); by.setdefault(meta[sid]['combo'], []).append(d)
            c = (pred[t].mean(0) * 48).astype(int).clip(0, 47); CIN.append(bool(g[c[1], c[0]]))
    r = dict(n_frames=len(D), dice_mean=float(np.mean(D)), dice_median=float(np.median(D)), frac_ge_0p5=float(np.mean(np.array(D) >= 0.5)), centroid_inside=float(np.mean(CIN)), by_combo={k: float(np.mean(v)) for k, v in by.items()})
    print(json.dumps(r, indent=1))
    if a.save: json.dump(r, open(a.save, 'w'), indent=1)

if __name__ == '__main__':
    main()
