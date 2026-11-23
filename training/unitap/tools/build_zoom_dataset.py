"""Build a per-clip zoomed dataset root (E009 stage-2): square box around the tool trajectory, cropped from the
native 896x665 frames, resized to 384x384, with GT re-normalized to the box.

Boxes:
  train  : from GT visible points over all 16 frames, plus random jitter (centre shift / scale) so the training
           box distribution mimics boxes derived from stage-1 predictions.
  val/test: from stage-1 predictions in a dump directory (pred_xy of all 16x3 points, robust 5/95 percentiles).
Box side L = clamp(max(w, h) * margin_scale + margin_add, lmin, lmax) native px (square), centre = bbox centre.

Output root mirrors data/case32/v1: regenerated video_16_npy (384x384 uint8, BGR kept), gt_tool_xy, gt_tool_vis,
gt_ct_xy, gt_ct_vis (points outside the box -> invisible), boxes.json, meta.json (+zoom_box), split txts; every
other file/dir of the source root is symlinked.  Frames are read whole then dropped from the page cache.
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool

OUT = 384   # native frame size varies per clip (W 895/896, H 613..720): always taken from meta W/H


def read_npy_dontneed(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
    return arr


def box_from_points(pts, margin_scale, margin_add, lmin, lmax, robust=False, W=896, H=665):
    """pts [N,2] native px -> (cx, cy, L)."""
    if len(pts) == 0:
        return W / 2.0, H / 2.0, float(lmax)
    if robust and len(pts) >= 8:
        x0, x1 = np.percentile(pts[:, 0], [5, 95]); y0, y1 = np.percentile(pts[:, 1], [5, 95])
    else:
        x0, x1 = pts[:, 0].min(), pts[:, 0].max(); y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    L = max(x1 - x0, y1 - y0) * margin_scale + margin_add
    L = float(np.clip(L, lmin, lmax))
    return float((x0 + x1) / 2), float((y0 + y1) / 2), L


def jitter_box(cx, cy, L, rng, shift=0.08, scale=(0.9, 1.25), lmin=256, lmax=720):
    cx += rng.uniform(-shift, shift) * L; cy += rng.uniform(-shift, shift) * L
    L = float(np.clip(L * rng.uniform(*scale), lmin, lmax))
    return cx, cy, L


def finalize_box(cx, cy, L):
    L = int(round(L)); bx0 = int(round(cx - L / 2)); by0 = int(round(cy - L / 2))
    return bx0, by0, L


def photometric_jitter(frames, rng, strength=0.25):
    """Per-clip brightness/contrast/saturation jitter (same params for all frames), uint8 BGR in/out."""
    import cv2
    b = rng.uniform(1 - strength, 1 + strength); c = rng.uniform(1 - strength, 1 + strength); sat = rng.uniform(1 - strength, 1 + strength)
    out = np.empty_like(frames)
    for t in range(frames.shape[0]):
        f = frames[t].astype(np.float32)
        hsv = cv2.cvtColor(frames[t], cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255); f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        m = f.mean(); f = (f - m) * c + m; f = f * b
        out[t] = np.clip(f, 0, 255).astype(np.uint8)
    return out


def crop_resize(frames, bx0, by0, L):
    import cv2
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]; out = np.zeros((T, OUT, OUT, 3), dtype=np.uint8)
    xs, ys = max(bx0, 0), max(by0, 0); xe, ye = min(bx0 + L, W), min(by0 + L, H)
    interp = cv2.INTER_AREA if L > OUT else cv2.INTER_LINEAR
    for t in range(T):
        pad = np.zeros((L, L, 3), dtype=np.uint8)
        if xe > xs and ye > ys:
            pad[ys - by0:ye - by0, xs - bx0:xe - bx0] = frames[t, ys:ye, xs:xe]
        out[t] = cv2.resize(pad, (OUT, OUT), interpolation=interp)
    return out


def transform_points(xy_norm, vis, bx0, by0, L, W, H):
    """xy_norm [...,2] normalized to the native frame -> normalized to the box; vis updated for points outside."""
    nat = xy_norm * np.array([W, H], dtype=np.float32)
    rel = (nat - np.array([bx0, by0], dtype=np.float32)) / float(L)
    inside = (rel[..., 0] >= 0) & (rel[..., 0] < 1) & (rel[..., 1] >= 0) & (rel[..., 1] < 1)
    vis_new = (vis > 0.5) & inside
    return rel.astype(np.float32), vis_new.astype(np.float32), inside


def work(job):
    src, dst, sid, box, write_video, W, H, aug = job
    bx0, by0, L = box
    out_sid = sid + (aug.get('suffix', '') if aug else '')
    gt = np.load(f'{src}/gt_tool_xy/{sid}.npy').astype(np.float32); vis = np.load(f'{src}/gt_tool_vis/{sid}.npy').astype(np.float32)
    rel, vnew, inside = transform_points(gt, vis, bx0, by0, L, W, H)
    if aug is not None and aug.get('flip'):
        rel = rel.copy(); rel[..., 0] = 1.0 - rel[..., 0]
        roles = list(aug.get('roles', [])) if aug.get('roles') else None
        if roles and len(roles) == rel.shape[1]:
            pair = [i for i in range(len(roles)) if roles.count(roles[i]) == 2]
            if len(pair) == 2: rel[:, pair] = rel[:, pair[::-1]]; vnew[:, pair] = vnew[:, pair[::-1]]
    np.save(f'{dst}/gt_tool_xy/{out_sid}.npy', rel); np.save(f'{dst}/gt_tool_vis/{out_sid}.npy', vnew)
    lost = int(((vis > 0.5) & ~inside).sum()); nvis = int((vis > 0.5).sum())
    if os.path.isfile(f'{src}/gt_ct_xy/{sid}.npy'):
        ct = np.load(f'{src}/gt_ct_xy/{sid}.npy').astype(np.float32); cv = np.load(f'{src}/gt_ct_vis/{sid}.npy').astype(np.float32)
        crel, cvnew, _ = transform_points(ct, cv, bx0, by0, L, W, H)
        if aug is not None and aug.get('flip'): crel = crel.copy(); crel[..., 0] = 1.0 - crel[..., 0]
        np.save(f'{dst}/gt_ct_xy/{out_sid}.npy', crel); np.save(f'{dst}/gt_ct_vis/{out_sid}.npy', cvnew)
    if write_video and not os.path.isfile(f'{dst}/video_16_npy/{out_sid}.npy'):
        frames = read_npy_dontneed(f'{src}/video_16_npy/{sid}.npy')
        assert frames.shape[1] == H and frames.shape[2] == W, (sid, frames.shape, W, H)
        if aug is not None:
            frames = photometric_jitter(frames, np.random.RandomState(aug['seed'] * 100003 + int(sid)), aug['strength'])
        out = crop_resize(frames, bx0, by0, L)
        if aug is not None and aug.get('flip'):
            out = out[:, :, ::-1].copy()
        tmp = f'{dst}/video_16_npy/{out_sid}.npy.tmp.npy'; np.save(tmp, out); os.replace(tmp, f'{dst}/video_16_npy/{out_sid}.npy')
    return sid, lost, nvis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/case32/v1'); ap.add_argument('--dst', default='data/case32/v1_zoom')
    ap.add_argument('--dump', action='append', default=[], help='split=dump_dir for prediction-derived boxes (val/test)')
    ap.add_argument('--margin-scale', type=float, default=1.5); ap.add_argument('--margin-add', type=float, default=96.0)
    ap.add_argument('--lmin', type=float, default=256); ap.add_argument('--lmax', type=float, default=720)
    ap.add_argument('--jitter-shift', type=float, default=0.08); ap.add_argument('--jitter-scale', type=float, nargs=2, default=(0.9, 1.25))
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--splits', default='train,val,test'); ap.add_argument('--stats-only', action='store_true'); ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--boxes-from', default=None, help='reuse the boxes of an existing zoom root (boxes.json) instead of computing new ones (augmented copies share GT)')
    ap.add_argument('--photometric', type=float, default=0.0, help='per-clip brightness/contrast/saturation jitter strength (0 = off)')
    ap.add_argument('--flip', action='store_true', help='horizontal flip of the crops (GT x -> 1-x; symmetric tool pair swapped via meta pt_role)')
    ap.add_argument('--sid-suffix', default='', help='write outputs under sid+suffix (augmented copies living next to the base clips)')
    ap.add_argument('--sids-file', default=None, help='restrict to the sids listed in this file (one per line); split labels still come from --splits')
    a = ap.parse_args()
    src, dst = a.src, a.dst
    meta = json.load(open(f'{src}/meta.json')); msid = {m['sid']: m for m in meta}
    dumps = dict(kv.split('=', 1) for kv in a.dump)
    rng = np.random.RandomState(a.seed)
    boxes = {}; stats = {}
    fixed = json.load(open(f'{a.boxes_from}/boxes.json'))['boxes'] if a.boxes_from else None
    for split in a.splits.split(','):
        sids = [l.strip() for l in open(f'{src}/{split}.txt') if l.strip()]
        if a.sids_file:
            keep = {l.strip() for l in open(a.sids_file) if l.strip()}; sids = [x for x in sids if x in keep]
        st = dict(n=0, inside_gt_pts=0, gt_pts=0, L=[], zoom=[], off=[], ratio=[])
        for sid in sids:
            W, H = int(msid[sid]['W']), int(msid[sid]['H'])
            gt = np.load(f'{src}/gt_tool_xy/{sid}.npy').astype(np.float32) * [W, H]; vis = np.load(f'{src}/gt_tool_vis/{sid}.npy') > 0.5
            gcx, gcy, gL = box_from_points(gt[vis], a.margin_scale, a.margin_add, a.lmin, a.lmax, W=W, H=H)
            if split in dumps:
                z = np.load(f'{dumps[split]}/{sid}.npz'); pred = z['pred_xy'].astype(np.float32).reshape(-1, 2) * [W, H]
                cx, cy, L = box_from_points(pred, a.margin_scale, a.margin_add, a.lmin, a.lmax, robust=True, W=W, H=H)
                st['off'].append(np.hypot(cx - gcx, cy - gcy) / gL); st['ratio'].append(L / gL)
            else:
                cx, cy, L = jitter_box(gcx, gcy, gL, rng, a.jitter_shift, tuple(a.jitter_scale), a.lmin, a.lmax)
                st['off'].append(np.hypot(cx - gcx, cy - gcy) / gL); st['ratio'].append(L / gL)
            bx0, by0, Li = finalize_box(cx, cy, L)
            if fixed is not None: bx0, by0, Li = fixed[sid]
            boxes[sid] = [bx0, by0, Li]
            rel = (gt[vis] - [bx0, by0]) / Li; inside = ((rel >= 0) & (rel < 1)).all(-1)
            st['n'] += 1; st['gt_pts'] += int(vis.sum()); st['inside_gt_pts'] += int(inside.sum()); st['L'].append(Li); st['zoom'].append(W / Li)
        stats[split] = dict(clips=st['n'], gt_points_inside=st['inside_gt_pts'] / max(st['gt_pts'], 1), L_median=float(np.median(st['L'])), L_p10_p90=[float(np.percentile(st['L'], 10)), float(np.percentile(st['L'], 90))],
                            zoom_median=float(np.median(st['zoom'])), zoom_p10_p90=[float(np.percentile(st['zoom'], 10)), float(np.percentile(st['zoom'], 90))],
                            centre_offset_over_L_median=float(np.median(st['off'])), centre_offset_over_L_p90=float(np.percentile(st['off'], 90)),
                            L_ratio_to_gt_median=float(np.median(st['ratio'])), L_ratio_p10_p90=[float(np.percentile(st['ratio'], 10)), float(np.percentile(st['ratio'], 90))])
        print(split, json.dumps(stats[split]))
    if a.stats_only:
        return
    for d in ['video_16_npy', 'gt_tool_xy', 'gt_tool_vis', 'gt_ct_xy', 'gt_ct_vis']:
        os.makedirs(f'{dst}/{d}', exist_ok=True)
    for e in os.listdir(src):
        if e in ('video_16_npy', 'gt_tool_xy', 'gt_tool_vis', 'gt_ct_xy', 'gt_ct_vis', 'meta.json', 'vjepa_token_cache', 'boxes.json') or e.startswith('vjepa_token_cache'):
            continue
        tgt = f'{dst}/{e}'
        if not os.path.lexists(tgt):
            os.symlink(os.path.abspath(f'{src}/{e}'), tgt)
    meta_out = []
    for m in meta:
        if m['sid'] in boxes:
            mm = dict(m); mm['zoom_box'] = boxes[m['sid']]; mm['sid'] = m['sid'] + a.sid_suffix; mm['base_sid'] = m['sid']
            mm['aug'] = dict(photometric=a.photometric, flip=bool(a.flip), seed=a.seed); meta_out.append(mm)
    # merge with what an earlier (chunked) run already wrote into dst
    if os.path.isfile(f'{dst}/meta.json'):
        prev = {m['sid']: m for m in json.load(open(f'{dst}/meta.json'))}; prev.update({m['sid']: m for m in meta_out}); meta_out = sorted(prev.values(), key=lambda m: m['sid'])
    new_boxes = {k + a.sid_suffix: v for k, v in boxes.items()}
    if os.path.isfile(f'{dst}/boxes.json'):
        prev_b = json.load(open(f'{dst}/boxes.json')).get('boxes', {}); prev_b.update(new_boxes); new_boxes = prev_b
    json.dump(meta_out, open(f'{dst}/meta.json', 'w')); json.dump(dict(boxes=new_boxes, args=vars(a), stats=stats), open(f'{dst}/boxes.json', 'w'))
    aug = (dict(seed=a.seed, strength=a.photometric, flip=bool(a.flip), suffix=a.sid_suffix) if (a.photometric > 0 or a.flip or a.sid_suffix) else None)
    jobs = [(src, dst, sid, boxes[sid], not a.no_video, int(msid[sid]['W']), int(msid[sid]['H']), (dict(aug, roles=list(msid[sid].get('pt_role', []))) if aug else None)) for sid in boxes]
    t0 = time.time(); lost_tot = 0; vis_tot = 0
    with Pool(a.workers) as pool:
        for i, (sid, lost, nvis) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            lost_tot += lost; vis_tot += nvis
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f'[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s  gt points outside box so far {lost_tot}/{vis_tot} ({100 * lost_tot / max(vis_tot, 1):.2f}%)', flush=True)
    print('done', dst)


if __name__ == '__main__':
    main()
