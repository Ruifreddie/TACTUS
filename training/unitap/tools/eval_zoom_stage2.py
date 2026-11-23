"""Evaluate a stage-2 (zoom dataset) prediction dump in the ORIGINAL frame coordinates against the original GT.

The zoom dump's pred_xy is normalized to the per-clip box (data/case32/v1_zoom/boxes.json); it is mapped back to
native px (bx0 + u*L, by0 + v*L), then to the original normalized frame (/896, /665) so that the metric is exactly
the one used for stage-1 (||d_norm * 384||, original-visible points only).  Optionally compares with a stage-1 dump.
"""
import argparse, glob, json, os
import numpy as np

# native frame size varies per clip (W 895/896, H 613-720): read from meta W/H


def metrics(E):
    e = np.concatenate(E); return dict(n=int(len(e)), ADE=float(e.mean()), med=float(np.median(e)), p90=float(np.percentile(e, 90)),
                                      **{f'PCK{t}': float((e < t).mean()) for t in (4, 8, 12, 16)}, frac_gt32=float((e > 32).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--zoom-dump', required=True); ap.add_argument('--boxes', default='data/case32/v1_zoom/boxes.json')
    ap.add_argument('--src', default='data/case32/v1'); ap.add_argument('--stage1-dump', default=None); ap.add_argument('--save', default=None)
    ap.add_argument('--slots', default='', help='a:b slice of predicted slots to evaluate (joint 7-point dumps: tool = 0:3)'); ap.add_argument('--export-dump', default=None, help='write per-sid npz (pred_xy/gt_xy/gt_vis in ORIGINAL frame coords) so tools/pixel_refiner.py eval can refine stage-2 outputs'); a = ap.parse_args()
    if a.export_dump: os.makedirs(a.export_dump, exist_ok=True)
    boxes = json.load(open(a.boxes))['boxes']; meta = {m['sid']: m for m in json.load(open(f'{a.src}/meta.json'))}
    E2, E1, by_tool, outside = [], [], {}, 0; n_pts = 0; per_clip = {}
    for f in sorted(glob.glob(os.path.join(a.zoom_dump, '*.npz'))):
        sid = os.path.basename(f)[:-4]; z = np.load(f); bx0, by0, L = boxes[sid]; WH = np.array([float(meta[sid]['W']), float(meta[sid]['H'])])
        gt = np.load(f'{a.src}/gt_tool_xy/{sid}.npy').astype(np.float64); vis = np.load(f'{a.src}/gt_tool_vis/{sid}.npy') > 0.5
        pxy = z['pred_xy'].astype(np.float64)
        if a.slots: lo, hi = [int(x) for x in a.slots.split(':')]; pxy = pxy[:, lo:hi]
        pred_nat = pxy * L + np.array([bx0, by0]); pred_norm = pred_nat / WH
        e2 = np.linalg.norm((pred_norm - gt) * 384, axis=-1)[vis]; E2.append(e2); by_tool.setdefault(meta[sid]['tool'], []).append(e2)
        if a.export_dump: np.savez(os.path.join(a.export_dump, f'{sid}.npz'), pred_xy=pred_norm.astype(np.float32), gt_xy=gt.astype(np.float32), gt_vis=vis.astype(np.float32))
        gt_nat = gt * WH; rel = (gt_nat - np.array([bx0, by0])) / L
        outside += int((vis & ~((rel >= 0) & (rel < 1)).all(-1)).sum()); n_pts += int(vis.sum())
        per_clip[sid] = dict(stage2=float(e2.mean()) if len(e2) else None)
        if a.stage1_dump and os.path.isfile(os.path.join(a.stage1_dump, f'{sid}.npz')):
            z1 = np.load(os.path.join(a.stage1_dump, f'{sid}.npz')); e1 = np.linalg.norm((z1['pred_xy'].astype(np.float64) - gt) * 384, axis=-1)[vis]; E1.append(e1)
            per_clip[sid]['stage1'] = float(e1.mean()) if len(e1) else None
    r = dict(stage2=metrics(E2), gt_points_outside_box=outside / max(n_pts, 1), by_tool={k: float(np.concatenate(v).mean()) for k, v in by_tool.items()})
    if E1: r['stage1'] = metrics(E1)
    print(json.dumps(r, indent=1))
    if a.save: json.dump(dict(summary=r, per_clip=per_clip), open(a.save, 'w'), indent=1)


if __name__ == '__main__':
    main()
