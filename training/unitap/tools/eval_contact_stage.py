"""Order-free evaluation of contact-track predictions (4 slots) produced by the tool path on the zoom dataset.

Predictions (dump from tools/dump_val_predictions.py on data/case32/v1_zoom_ct) are box-normalized; they are mapped
back to the original frame through data/case32/v1_zoom/boxes.json and compared with the ORIGINAL contact GT
(data/case32/v1/gt_ct_xy, gt_ct_vis) in the same 384-px metric as the tool keypoints.

Reported (visible GT slots only):
  clip_perm : one permutation per clip (best of 4!), i.e. slot identity consistent over time   (training objective)
  frame_hung: Hungarian assignment per frame (upper bound, order-free per frame)
  region    : per-frame Dice between Gaussian-blob unions (sigma px) of predicted vs GT points on a 384 grid
"""
import argparse, glob, itertools, json, os
import numpy as np

PERMS4 = None


def perms(P):
    return np.array(list(itertools.permutations(range(P))))


def blob_union(pts, sigma, size=96):
    """pts [N,2] in 384-px coords -> soft mask on a size x size grid (cell = 384/size px)."""
    g = np.zeros((size, size), dtype=np.float32)
    if len(pts) == 0:
        return g
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) * (384.0 / size)
    for p in pts:
        g = np.maximum(g, np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * sigma ** 2)))
    return g


def dice(a, b):
    return float((2 * (a * b).sum() + 1e-6) / (a.sum() + b.sum() + 1e-6))


def convex_hull(pts):
    """Andrew monotone chain; pts [N,2] -> hull vertices in order (N>=1)."""
    P = sorted(set(map(tuple, np.asarray(pts, dtype=np.float64).round(4))))
    if len(P) <= 2: return np.array(P)
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for q in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0: lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0: upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def hull_mask(pts, size=384, min_radius=6.0):
    """Rasterized convex hull of the points (384-px coords). Degenerate hulls (<3 distinct pts / tiny area) are
    dilated to a disc/segment of radius min_radius so that a 2-point GT still defines a region."""
    from PIL import Image, ImageDraw
    im = Image.new('L', (size, size), 0); d = ImageDraw.Draw(im); h = convex_hull(pts)
    if len(h) >= 3:
        d.polygon([tuple(v) for v in h], fill=255)
    for v in h:  # dilate corners a little so thin regions are not empty
        d.ellipse([v[0] - min_radius, v[1] - min_radius, v[0] + min_radius, v[1] + min_radius], fill=255)
    if len(h) == 2: d.line([tuple(h[0]), tuple(h[1])], fill=255, width=int(2 * min_radius))
    return np.asarray(im) > 0


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dump', required=True); ap.add_argument('--boxes', default='data/case32/v1_zoom/boxes.json')
    ap.add_argument('--src', default='data/case32/v1'); ap.add_argument('--frame-coords', action='store_true', help='dump pred_xy (or --key) is already normalized to the native frame (no box mapping)'); ap.add_argument('--key', default='pred_xy'); ap.add_argument('--slots', default='', help='a:b slice of predicted slots (joint 7-point dumps: contact = 3:7)'); ap.add_argument('--sigma', type=float, default=12.0); ap.add_argument('--save', default=None); a = ap.parse_args()
    boxes = json.load(open(a.boxes))['boxes']; meta = {m['sid']: m for m in json.load(open(f'{a.src}/meta.json'))}
    E_clip, E_frame, D, per_clip = [], [], [], {}; n_out = 0; n_pts = 0; IOU = []; CIN = []; per_clip_iou = {}; HD = []; HD_meta = []
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        linear_sum_assignment = None
    for f in sorted(glob.glob(os.path.join(a.dump, '*.npz'))):
        sid = os.path.basename(f)[:-4]; z = np.load(f); bx0, by0, L = boxes[sid]; WH = np.array([float(meta[sid]['W']), float(meta[sid]['H'])])
        pxy = z[a.key].astype(np.float64)
        if a.slots: lo, hi = [int(x) for x in a.slots.split(':')]; pxy = pxy[:, lo:hi]
        pred = (pxy * 384) if a.frame_coords else ((pxy * L + np.array([bx0, by0])) / WH * 384)          # [T,P,2] 384-px original frame
        gt = np.load(f'{a.src}/gt_ct_xy/{sid}.npy').astype(np.float64) * 384; vis = np.load(f'{a.src}/gt_ct_vis/{sid}.npy') > 0.5
        T, P = gt.shape[:2]
        gt_nat = gt / 384 * WH; rel = (gt_nat - np.array([bx0, by0])) / L; n_out += int((vis & ~((rel >= 0) & (rel < 1)).all(-1)).sum()); n_pts += int(vis.sum())
        # clip-level best permutation (visibility-weighted L1 over frames)
        best = None
        for pm in perms(P):
            e = np.linalg.norm(pred - gt[:, pm], axis=-1); c = (np.abs(pred - gt[:, pm]).sum(-1) * vis[:, pm]).sum()
            if best is None or c < best[0]: best = (c, e[vis[:, pm]])
        E_clip.append(best[1])
        # per-frame Hungarian
        for t in range(T):
            vk = np.where(vis[t])[0]
            if len(vk) == 0: continue
            C = np.linalg.norm(pred[t][:, None, :] - gt[t][vk][None, :, :], axis=-1)   # [P, nvis]
            if linear_sum_assignment is not None:
                r, c = linear_sum_assignment(C); E_frame.append(C[r, c])
            else:
                bestf = min((C[list(pm[:len(vk)]), range(len(vk))].sum(), pm) for pm in perms(P)); E_frame.append(C[list(bestf[1][:len(vk)]), range(len(vk))])
            D.append(dice(blob_union(pred[t], a.sigma), blob_union(gt[t][vk], a.sigma)))
            if len(vk) >= 2:
                mg = hull_mask(gt[t][vk]); mp = hull_mask(pred[t]); IOU.append(iou(mp, mg)); per_clip_iou.setdefault(sid, []).append(IOU[-1])
                HD.append(float(2 * (mp & mg).sum() / max(mp.sum() + mg.sum(), 1))); HD_meta.append((meta[sid]['combo'], len(vk)))
                pc = pred[t].mean(0).round().astype(int); CIN.append(bool(0 <= pc[0] < 384 and 0 <= pc[1] < 384 and mg[pc[1], pc[0]]))
        per_clip[sid] = float(best[1].mean()) if len(best[1]) else None
    def m(E):
        e = np.concatenate(E); return dict(n=int(len(e)), ADE=float(e.mean()), med=float(np.median(e)), p90=float(np.percentile(e, 90)), **{f'PCK{t}': float((e < t).mean()) for t in (4, 8, 12, 16)}, frac_gt32=float((e > 32).mean()))
    IOU = np.array(IOU)
    HD = np.array(HD); by_combo = {}; by_nvis = {}
    for (cb, nv), d in zip(HD_meta, HD): by_combo.setdefault(cb, []).append(d); by_nvis.setdefault(nv, []).append(d)
    r = dict(clip_perm=m(E_clip), frame_hung=m(E_frame),
             region_hull_dice=dict(mean=float(HD.mean()), median=float(np.median(HD)), frac_dice_ge_0p5=float((HD >= 0.5).mean()), by_combo={k: float(np.mean(v)) for k, v in by_combo.items()}, by_nvis={int(k): float(np.mean(v)) for k, v in by_nvis.items()}, mean_nvis_ge3=float(np.mean([d for (cb, nv), d in zip(HD_meta, HD) if nv >= 3]))),
             region_hull_iou=dict(n_frames=int(len(IOU)), mean=float(IOU.mean()), median=float(np.median(IOU)), frac_iou_ge_0p3=float((IOU >= 0.3).mean()), frac_iou_ge_0p5=float((IOU >= 0.5).mean()), frac_iou_zero=float((IOU == 0).mean()), pred_centroid_inside_gt_region=float(np.mean(CIN)), clip_mean_iou_median=float(np.median([np.mean(v) for v in per_clip_iou.values()]))),
             region_dice_mean=float(np.mean(D)), region_dice_median=float(np.median(D)), gt_points_outside_box=n_out / max(n_pts, 1), sigma_px=a.sigma)
    print(json.dumps(r, indent=1))
    if a.save: json.dump(dict(summary=r, per_clip=per_clip), open(a.save, 'w'), indent=1)


if __name__ == '__main__':
    main()
