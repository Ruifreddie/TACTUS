"""V1 hindsight contact labels (static-tissue approximation) for the ZOOM dataset:
per sampled frame t: hotspot = union of tool-TIP positions at all raw-annotation frames t' >= t inside the action range
(fallback: all action frames), transformed with the dataset crop transform and then into the clip's zoom box.
Outputs (data/case32/v1_zoom_hs): region_mask/{sid}.npy [16,48,48] float32 soft mask (Gaussian blobs, sigma_px @384),
gt_tool_xy/{sid}.npy [16,4,2] pseudo-corners (hotspot centroid +- 1.5 sigma principal axes) box-normalized, gt_tool_vis [16,4],
meta.json with pt_role [1,1,1,1]; everything else symlinked from data/case32/v1_zoom_ct."""
import json, os, sys, numpy as np, collections
sys.path.insert(0, 'tools'); sys.path.insert(0, 'third_party/vitpose_dataset_tools')
import export_case32 as EX; EX._init_worker(); B = EX.builder
SRC = os.environ.get('HS_SRC', 'data/case32/v1'); RAW = os.environ.get('HS_RAW', 'data/raw_case32'); ZOOM = os.environ.get('HS_ZOOM', 'data/case32/v1_zoom'); CT = os.environ.get('HS_CT', 'data/case32/v1_zoom_ct'); DST = os.environ.get('HS_DST', 'data/case32/v1_zoom_hs')
SIGMA384 = 14.0; SIDE = 48
meta = {m['sid']: m for m in json.load(open(f'{SRC}/meta.json'))}; boxes = json.load(open(f'{ZOOM}/boxes.json'))['boxes']
for d in ('region_mask', 'gt_tool_xy', 'gt_tool_vis'): os.makedirs(f'{DST}/{d}', exist_ok=True)
for e in os.listdir(CT):
    if e in ('gt_tool_xy', 'gt_tool_vis', 'meta.json', 'region_mask'): continue
    if not os.path.exists(f'{CT}/{e}'): continue
    t = f'{DST}/{e}'
    if not os.path.lexists(t): os.symlink(os.path.abspath(f'{CT}/{e}') if not os.path.islink(f'{CT}/{e}') else os.readlink(f'{CT}/{e}'), t)
def tips_crop(m):
    d = json.load(open(f"{RAW}/{m['ann_relpath']}")); annW, annH = d['imageDimensions']['width'], d['imageDimensions']['height']
    fr = collections.defaultdict(dict)
    for p in d['points']:
        if p['type'] == 'tool_tip' and p.get('vis', True): fr[int(p['frameIndex'])][int(p['uid'])] = (p['x'] / annW, p['y'] / annH)
    out = {}
    for f, uids in fr.items():
        xy = np.array(list(uids.values()), dtype=np.float32).reshape(1, -1, 2)
        out[f] = np.asarray(B.transform_points_to_crop(xy, annW, annH, m['frame_W'], m['frame_H'], tuple(m['crop_bbox_xyxy']))).reshape(-1, 2).mean(0)  # crop-normalized
    return out, (int(m['action_start_abs']), int(m['action_end_abs']))
yy, xx = np.mgrid[0:SIDE, 0:SIDE]; cell = np.stack([xx, yy], -1).astype(np.float32) + 0.5
n_ok = 0; n_empty = 0; meta_out = []
for sid, m in meta.items():
    bx0, by0, L = boxes[sid]; W, H = m['W'], m['H']; samp = np.load(f'{SRC}/sampled_abs_frames/{sid}.npy').reshape(-1).astype(int)
    tips, (a0, a1) = tips_crop(m); act = sorted(f for f in tips if a0 <= f <= a1)
    mask = np.zeros((16, SIDE, SIDE), np.float32); corners = np.zeros((16, 4, 2), np.float32); vis = np.zeros((16, 4), np.float32)
    if act:
        n_ok += 1
        for i, t in enumerate(samp):
            pts = np.array([tips[f] for f in act if f >= t] or [tips[f] for f in act])  # crop-normalized
            nat = pts * [W, H]; box = (nat - [bx0, by0]) / L                                # box-normalized
            sig_cells = SIGMA384 / 384.0 * (W / L) * SIDE / 1.0 * (384.0 / 384.0)          # sigma in box cells: px@384 -> native px (W/384) -> box units (/L) -> cells (*SIDE)
            sig_cells = SIGMA384 * (W / 384.0) / L * SIDE
            for p in box:
                d2 = ((cell - p * SIDE) ** 2).sum(-1); mask[i] = np.maximum(mask[i], np.exp(-d2 / (2 * sig_cells ** 2)))
            c = box.mean(0); cov = np.cov(box.T) + np.eye(2) * (sig_cells / SIDE) ** 2 if len(box) > 1 else np.eye(2) * (sig_cells / SIDE) ** 2
            w, v = np.linalg.eigh(cov); r = 1.5 * np.sqrt(np.maximum(w, 1e-8))
            corners[i] = np.stack([c + v[:, 0] * r[0], c + v[:, 1] * r[1], c - v[:, 0] * r[0], c - v[:, 1] * r[1]]).clip(0, 1); vis[i] = 1.0
    else:
        n_empty += 1
    np.save(f'{DST}/region_mask/{sid}.npy', mask); np.save(f'{DST}/gt_tool_xy/{sid}.npy', corners); np.save(f'{DST}/gt_tool_vis/{sid}.npy', vis)
    mm = dict(m); mm['pt_role'] = [1, 1, 1, 1]; mm['topology'] = '4C'; mm['tool_kpt_source'] = 'hindsight_tip_hotspot_v1'; meta_out.append(mm)
json.dump(meta_out, open(f'{DST}/meta.json', 'w')); print('hindsight labels:', n_ok, 'clips with action tips,', n_empty, 'without (empty masks)')
