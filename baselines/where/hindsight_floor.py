#!/usr/bin/env python3
"""Floor of Tab. 3 (c): the hindsight target of eq. (hindsight), built from the TRUE tool tips, scored against the human
contact regions. No learning: it shows how close tip-derived targets come to the human regions.

Hotspot definition (UniTAP tools/build_hindsight_labels.py, the targets of TactusNet's contact decoder, run C006):
  A_t  = the raw annotated frames of the action interval at or after the sampled frame t (all action frames if t lies
         after the interval); frames whose tips are all invisible do not count;
  u_i  = mean of the visible annotated tool tips of frame i, in content-crop coordinates (UniTAP's crop transform);
  M_t(g) = max_{i in A_t} exp(-|g - u_i|^2 / (2 sigma^2)), sigma = 14 px at 384 along the frame width, i.e. 14 W / 384
         native pixels, isotropic in native pixels (UniTAP's SIGMA384 * W / 384); the hotspot is M_t > 0.5.
UniTAP rasterizes M_t in the 48x48 grid of the clip's zoom box. Here it is rasterized in the 384x384 content-crop grid
of the metric (whole frame, no box clipping): sigma is then 14 px along x and 14 W / H px along y. Pixel (u, v) sits at
(u, v) in 384-px units, the convention of PIL's polygons in common.hull.hull_mask, so the hotspot and the human region
live in the same grid. A clip without any visible tip inside the action interval has an empty hotspot (UniTAP: empty
mask, no loss), so its scored frames get Dice 0.

Outputs (standard kpt files, scored by baselines.common.score):
  <out>/hindsight_floor/pred_<split>.npz          ct_frame_dice of the hotspot mask (the Tab. 3 row)
  <out>/hindsight_floor_corners/pred_<split>.npz  ct_xy = the 4 pseudo-corners (hotspot centre +- 1.5 sigma along the
                                                  principal axes, UniTAP's point target), clipped to the frame
--check-exp EXP: agreement with UniTAP's own labels in EXP/data/case32/v1_zoom_hs (see check_unitap).

  python -m baselines.where.hindsight_floor --out OUT [--splits val,test] [--check-exp EXP] [--tip-source raw|export]
"""
import argparse, collections, json, time
from pathlib import Path

import numpy as np

from ..common import config
from ..common import metrics as M
from ..common import preds as P
from ..common.data import get_export

SIGMA384 = 14.0      # UniTAP build_hindsight_labels.py
BOX_SIDE = 48        # UniTAP's box grid
THR = 0.5            # UniTAP eval_hindsight.py: hotspot = region_mask > 0.5
R2 = 2.0 * np.log(2.0)   # exp(-d^2 / (2 s^2)) > 0.5  <=>  d^2 < 2 ln 2 s^2


# ---------------------------------------------------------------- tips and A_t (exactly as UniTAP)
def raw_tips(ex, sid):
    """{raw frame index: mean visible tip, crop-normalized float32}, (action start, action end) in absolute frames.
    Same arithmetic as UniTAP tips_crop() with transform_points_to_crop (USE_PIXEL_COORDS=False)."""
    m = ex.meta[sid]; d = json.load(open(ex.hf_raw / m["ann_relpath"]))
    annW, annH = d["imageDimensions"]["width"], d["imageDimensions"]["height"]
    fr = collections.defaultdict(dict)
    for p in d["points"]:
        if p["type"] == "tool_tip" and p.get("vis", True): fr[int(p["frameIndex"])][int(p["uid"])] = (p["x"] / annW, p["y"] / annH)
    x0, y0, x1, y1 = m["crop_bbox_xyxy"]; cw, ch = float(max(1, int(x1 - x0))), float(max(1, int(y1 - y0)))
    out = {}
    for f, uids in fr.items():
        xy = np.array(list(uids.values()), dtype=np.float32).reshape(1, -1, 2).copy()
        xy[..., 0] = xy[..., 0] * float(m["frame_W"]) - float(x0); xy[..., 1] = xy[..., 1] * float(m["frame_H"]) - float(y0)
        xy[..., 0] /= cw; xy[..., 1] /= ch
        out[f] = xy.astype(np.float32).reshape(-1, 2).mean(0)
    return out, (int(m["action_start_abs"]), int(m["action_end_abs"]))


def export_tips(ex, sid):
    """Fallback without raw annotations: the 16 protocol frames only (mean visible tip), action interval of the export."""
    xy, vis = ex.tool_points(sid); ti = ex.tip_index(sid); samp = ex.sampled_abs(sid); out = {}
    for t, f in enumerate(samp):
        v = [k for k in ti if vis[t, k] > 0.5]
        if v: out[int(f)] = xy[t, v].mean(0).astype(np.float32)
    m = ex.meta[sid]
    return out, (int(m["action_start_abs"]), int(m["action_end_abs"]))


def hotspot_points(ex, sid, tip_source="raw"):
    """For each of the 16 sampled frames, the tip positions u_i, i in A_t (crop-normalized [n,2]); None if A is empty."""
    tips, (a0, a1) = raw_tips(ex, sid) if tip_source == "raw" else export_tips(ex, sid)
    act = sorted(f for f in tips if a0 <= f <= a1)
    if not act: return None
    return [np.array([tips[f] for f in act if f >= t] or [tips[f] for f in act]) for t in ex.sampled_abs(sid)]


# ---------------------------------------------------------------- rasterization
def hotspot_mask_384(pts, W, H, size=config.METRIC_SIZE, iso384=False):
    """Boolean [size,size] hotspot M_t > 0.5 in the content-crop grid; pts crop-normalized [n,2]."""
    S = float(size); m = np.zeros((size, size), bool)
    if iso384:   # sensitivity variant: isotropic 14 px in the squashed 384 grid
        sx = sy = SIGMA384 * S / 384.0
    else:        # UniTAP: isotropic 14 W / 384 native px -> 14 px along x, 14 W / H px along y at 384
        sx = SIGMA384 * S / 384.0; sy = SIGMA384 * (float(W) / 384.0) * S / float(H)
    rx, ry = sx * np.sqrt(R2), sy * np.sqrt(R2)
    for x, y in np.unique(np.round(np.asarray(pts, np.float64) * S, 6), axis=0):
        u0, u1 = max(0, int(np.floor(x - rx))), min(size - 1, int(np.ceil(x + rx)))
        v0, v1 = max(0, int(np.floor(y - ry))), min(size - 1, int(np.ceil(y + ry)))
        if u0 > u1 or v0 > v1: continue
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        m[v0:v1 + 1, u0:u1 + 1] |= ((uu - x) / sx) ** 2 + ((vv - y) / sy) ** 2 < R2
    return m


def unitap_box_soft(pts_t, W, H, box, side=BOX_SIDE):
    """UniTAP's own soft mask on the zoom-box grid, line for line (for the reproduction check)."""
    bx0, by0, L = box; yy, xx = np.mgrid[0:side, 0:side]; cell = np.stack([xx, yy], -1).astype(np.float32) + 0.5
    out = np.zeros((len(pts_t), side, side), np.float32)
    for i, pts in enumerate(pts_t):
        nat = pts * [W, H]; bxy = (nat - [bx0, by0]) / L; sig = SIGMA384 * (W / 384.0) / L * side
        for p in bxy:
            d2 = ((cell - p * side) ** 2).sum(-1); out[i] = np.maximum(out[i], np.exp(-d2 / (2 * sig ** 2)))
    return out


def corners_frame(pts, W, H):
    """UniTAP's pseudo-corners computed in native pixels (box-free), normalized to the content crop and clipped to it."""
    nat = np.asarray(pts, np.float64) * [W, H]; s = SIGMA384 * W / 384.0; c = nat.mean(0)
    cov = np.cov(nat.T) + np.eye(2) * s ** 2 if len(nat) > 1 else np.eye(2) * s ** 2
    w, v = np.linalg.eigh(cov); r = 1.5 * np.sqrt(np.maximum(w, 1e-8))
    q = np.stack([c + v[:, 0] * r[0], c + v[:, 1] * r[1], c - v[:, 0] * r[0], c - v[:, 1] * r[1]])
    return np.clip(q / [W, H], 0, 1)


# ---------------------------------------------------------------- the floor
def floor_clip(ex, sid, tip_source="raw", iso384=False):
    """-> per-frame Dice [16] vs the human region, pseudo-corners [16,4,2], hotspot area [16], empty flag."""
    m = ex.meta[sid]; W, H = float(m["W"]), float(m["H"]); gt, vis = ex.contact_points(sid)
    pts = hotspot_points(ex, sid, tip_source)
    if pts is None:   # no target: empty mask; the corners collapse onto the frame's corner (a 6-px disc, about Dice 0)
        masks = np.zeros((16, config.METRIC_SIZE, config.METRIC_SIZE), bool); corners = np.zeros((16, 4, 2), np.float32)
    else:
        masks = np.stack([hotspot_mask_384(p, W, H, iso384=iso384) for p in pts])
        corners = np.stack([corners_frame(p, W, H) for p in pts]).astype(np.float32)
    d = M.contact_frame_dice(gt, vis, pred_masks=masks)
    return d, corners, masks.reshape(16, -1).sum(1), pts is None


def run_floor(ex, splits, out, tip_source="raw", iso384=False, tag=""):
    res = {}
    for split in splits:
        sids = ex.sids(split); t0 = time.time(); D, C, empty, area = [], [], 0, []
        for s in sids:
            d, c, a, e = floor_clip(ex, s, tip_source, iso384); D.append(d); C.append(c); empty += int(e); area.append(a)
        D = np.stack(D); C = np.stack(C)
        info = {"method": "hindsight_floor" + tag, "split": split, "conditioning": "true tool tips and true action interval",
                "target": "UniTAP build_hindsight_labels.py hotspot (M_t > 0.5), sigma 14 px at 384 along x"
                          + (" (isotropic at 384)" if iso384 else ", isotropic in native px"),
                "grid": "384x384 content crop, no zoom box", "tip_source": tip_source, "clips_without_action_tips": empty}
        p1 = P.save(Path(out) / f"hindsight_floor{tag}" / f"pred_{split}.npz", "kpt", sids, info, ct_frame_dice=D)
        p2 = P.save(Path(out) / f"hindsight_floor{tag}_corners" / f"pred_{split}.npz", "kpt", sids,
                    dict(info, method="hindsight_floor_corners" + tag, target="UniTAP pseudo-corners (centre +- 1.5 sigma, principal axes)"), ct_xy=C)
        ok = ~np.isnan(D); res[split] = {"clips": len(sids), "frames_scored": int(ok.sum()), "dice_pooled": float(D[ok].mean()),
                                         "clips_without_action_tips": empty, "seconds": round(time.time() - t0, 1)}
        print(f"[floor{tag}] {split}: {len(sids)} clips, pooled Dice {D[ok].mean():.4f} over {ok.sum()} frames, "
              f"{empty} clips without action tips -> {p1.parent.name}, {p2.parent.name}", flush=True)
    return res


# ---------------------------------------------------------------- agreement with UniTAP's labels
def check_unitap(ex, exp, tip_source="raw"):
    """(a) UniTAP's formula re-run on its own box grid vs its region_mask (max |diff|, frames whose hotspot flag differs);
    (b) its box-grid mask (> 0.5) mapped to the 384 grid through the zoom box vs our 384-grid hotspot: Dice inside the
        box (frames where either is non-empty), and the share of our hotspot area that falls outside the box;
    (c) its pseudo-corners (gt_tool_xy, box-normalized) mapped to the frame vs ours (Hungarian, px at 384), corners that
        UniTAP did not clip to the box only;
    (d) Where Dice vs the human regions of its box-limited mask vs our full-frame hotspot, per split."""
    from scipy.optimize import linear_sum_assignment
    E = Path(exp); hs = E / "data/case32/v1_zoom_hs"; boxes = json.load(open(E / "data/case32/v1_zoom/boxes.json"))["boxes"]
    S = config.METRIC_SIZE; uu, vv = np.meshgrid(np.arange(S), np.arange(S))
    sids = sorted(set(boxes) & set(ex.meta)); a_max = 0.0; a_flag = 0; a_frames = 0; dice_in, out_share, cerr = [], [], []
    d_split = collections.defaultdict(lambda: {"ours": [], "unitap_box": []}); split_of = {s: sp for sp in ("train", "val", "test") for s in ex.sids(sp)}
    for sid in sids:
        m = ex.meta[sid]; W, H = float(m["W"]), float(m["H"]); box = boxes[sid]; bx0, by0, L = box
        rm = np.load(hs / "region_mask" / f"{sid}.npy"); pts = hotspot_points(ex, sid, tip_source)
        soft = np.zeros_like(rm) if pts is None else unitap_box_soft(pts, W, H, box)
        a_max = max(a_max, float(np.abs(soft - rm).max())); a_flag += int(((soft.reshape(16, -1).max(1) > THR) != (rm.reshape(16, -1).max(1) > THR)).sum()); a_frames += 16
        # 384-grid pixel (u, v) -> native px (u W / 384, v H / 384) -> box cell
        bx = (uu / S * W - bx0) / L; by = (vv / S * H - by0) / L; inside = (bx >= 0) & (bx < 1) & (by >= 0) & (by < 1)
        ci = np.clip((by * BOX_SIDE).astype(int), 0, BOX_SIDE - 1); cj = np.clip((bx * BOX_SIDE).astype(int), 0, BOX_SIDE - 1)
        gt, gvis = ex.contact_points(sid); ours_all = []; theirs_all = []
        for t in range(16):
            ours = hotspot_mask_384(pts[t], W, H) if pts is not None else np.zeros((S, S), bool)
            theirs = (rm[t] > THR)[ci, cj] & inside; ours_all.append(ours); theirs_all.append(theirs)
            oi = ours & inside
            if oi.any() or theirs.any(): dice_in.append(2 * (oi & theirs).sum() / max(oi.sum() + theirs.sum(), 1))
            if ours.any(): out_share.append(float((ours & ~inside).sum() / ours.sum()))
        if pts is not None:
            qt = np.load(hs / "gt_tool_xy" / f"{sid}.npy").astype(np.float64)
            for t in range(16):
                keep = ((qt[t] > 1e-6) & (qt[t] < 1 - 1e-6)).all(1)
                if not keep.any(): continue
                theirs_f = (qt[t] * L + [bx0, by0]) / [W, H] * S; ours_f = corners_frame(pts[t], W, H) * S
                C = np.linalg.norm(theirs_f[keep][:, None] - ours_f[None], axis=-1); r, c = linear_sum_assignment(C); cerr.extend(C[r, c].tolist())
        if split_of.get(sid) in ("val", "test"):
            d_split[split_of[sid]]["ours"].append(M.contact_frame_dice(gt, gvis, pred_masks=np.stack(ours_all)))
            d_split[split_of[sid]]["unitap_box"].append(M.contact_frame_dice(gt, gvis, pred_masks=np.stack(theirs_all)))
    dice_in = np.array(dice_in); cerr = np.array(cerr)
    res = {"clips": len(sids), "a_box_grid_max_abs_diff": a_max, "a_frames_hotspot_flag_differs": a_flag, "a_frames": a_frames,
           "b_frames": int(dice_in.size), "b_dice_inside_box_mean": float(dice_in.mean()), "b_dice_inside_box_median": float(np.median(dice_in)),
           "b_share_frames_dice_ge_0p9": float((dice_in >= 0.9).mean()), "b_share_frames_dice_ge_0p8": float((dice_in >= 0.8).mean()),
           "b_hotspot_area_outside_box_mean": float(np.mean(out_share)), "b_share_frames_with_area_outside_box": float(np.mean(np.array(out_share) > 0)),
           "c_corners": int(cerr.size), "c_corner_err_px384_mean": float(cerr.mean()) if cerr.size else None,
           "c_corner_err_px384_max": float(cerr.max()) if cerr.size else None,
           "d_where_dice_vs_human": {sp: {k: float(np.nanmean(np.concatenate(v))) for k, v in dd.items()} for sp, dd in d_split.items()}}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(config.OUT)); ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tip-source", default="raw", choices=("raw", "export")); ap.add_argument("--iso384", action="store_true")
    ap.add_argument("--check-exp", help="UniTAP experiment dir with data/case32/v1_zoom_hs (agreement check)")
    ap.add_argument("--export"); ap.add_argument("--hf-raw"); a = ap.parse_args(argv); ex = get_export(a.export, a.hf_raw)
    tag = ("_iso384" if a.iso384 else "") + ("_exporttips" if a.tip_source == "export" else "")
    res = {"floor": run_floor(ex, a.splits.split(","), a.out, a.tip_source, a.iso384, tag)}
    if a.check_exp:
        res["unitap_agreement"] = check_unitap(ex, a.check_exp, a.tip_source)
        print("[check] " + json.dumps(res["unitap_agreement"], indent=1))
    p = Path(a.out) / f"hindsight_floor{tag}" / "report.json"; p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p, "w"), indent=1); print("wrote", p)


if __name__ == "__main__":
    main()
