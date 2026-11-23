#!/usr/bin/env python3
"""Export the case32 dataset (v1) with the ViTPose builder's crop logic + uid-based contact GT + hard visibility.

Per clip in data/case32/split_temporal.json (non-null split):
  1. builder.extract_base_sample(...)  -> frames (video_16_npy), crop bbox (black bars + UI removed, point-aware
     expansion), gt_tool_xy (crop-normalized), raw keypoint_vis, sampled_abs_frames, action_range_abs/mask, ...
  2. uid post-process: contact points are rebuilt as 4 uid-consistent tracks (top-4 by visible frames in the
     16-frame window), transformed with the SAME crop transform; hard 0/1 visibility; per-frame existence.
     The builder's geometric-order versions are kept as gt_ct_xy_geom / gt_ct_vis_soft for comparison.
  3. gt_tool_vis becomes the raw binary visibility (builder's soft weights kept as gt_tool_vis_soft).
  4. action_range_16/<sid>.npz = features [16,4] (active, progress, duration_frac, signed_dist_frac) + inside_action_range.
Writes meta.json / labels.json / {train,val,test}.txt / stats.json into data/case32/v1. Reads only raw_case32.
"""
from __future__ import annotations
import argparse, dataclasses, json, os, sys, time, traceback, collections
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("UNITAP_ROOT", Path(__file__).resolve().parents[1]))   # pipeline/export_clips.py overrides RAW, OUT, TOOLS_DIR
RAW = ROOT / "data" / "raw_case32"
SPLIT_JSON = ROOT / "data" / "case32" / "split_temporal.json"
OUT = ROOT / "data" / "case32" / "v1"
TOOLS_DIR = ROOT / "third_party" / "vitpose_dataset_tools"
N_CONTACT_SLOTS = 4

builder = None  # set in worker init


def _init_worker():
    global builder
    import cv2  # noqa
    cv2.setNumThreads(1)
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    import build_phase4p5_delta_aug as b  # type: ignore
    b.OUTPUT_MODE = "tool_kpt_only"
    b._refresh_output_mode_dependent_config()
    b.ENFORCE_HOOK_DISSECT_MIN_STEPS = False
    builder = b


def make_dirs(out_root: Path):
    names = {
        "MOTION_DIR": "motions", "ANCHOR_TIP_DIR": "anchor_tips", "COND_POINTS_DIR": "cond_points",
        "VIDEO16_DIR": "video_16_npy", "KEYPOINT_DIR": "keypoints", "KEYPOINT_VIS_DIR": "keypoint_vis",
        "CONTACT_VIS_DIR": "contact_vis_raw", "FRAME_INDEX_DIR": "sampled_abs_frames",
        "ACTION_RANGE_ABS_DIR": "action_range_abs", "ACTION_RANGE_MASK_DIR": "action_range_mask",
        "GT_TOOL_XY_DIR": "gt_tool_xy", "GT_TOOL_VIS_DIR": "gt_tool_vis", "GT_CT_XY_DIR": "gt_ct_xy",
        "GT_CT_VIS_DIR": "gt_ct_vis", "COMBO_ID_DIR": "combo_ids", "GT_TOOL_XY_PSEUDO_DIR": "gt_tool_xy_pseudo",
        "GT_TOOL_PSEUDO_W_DIR": "gt_tool_pseudo_w", "GT_CT_XY_PSEUDO_DIR": "gt_ct_xy_pseudo",
        "GT_CT_PSEUDO_W_DIR": "gt_ct_pseudo_w", "PT_ROLE_DIR": "pt_roles", "TOOL_ID_DIR": "tool_ids",
        "ACT_ID_DIR": "act_ids", "IMAGE_DIR": "images", "CONTACT_DIR": "contacts", "TEXT_DIR": "texts",
    }
    dirs = {k: out_root / v for k, v in names.items()}
    extra = ["gt_ct_xy_geom", "gt_ct_vis_soft", "gt_tool_vis_soft", "gt_ct_exist", "contact_uids", "action_range_16"]
    for p in list(dirs.values()) + [out_root / e for e in extra]:
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def uid_contact_tracks(data, sampled_abs, ann_W, ann_H, frame_W, frame_H, bbox):
    """Return xy [16,4,2] crop-normalized, vis [16,4] {0,1}, slot uids, n_tracks_total, n_tracks_visible."""
    T = len(sampled_abs)
    fidx = {int(f): i for i, f in enumerate(sampled_abs)}
    tracks = collections.defaultdict(dict)  # uid -> frame -> (x, y, vis)
    for p in data.get("points", []):
        if p.get("type") != "contact":
            continue
        tracks[int(p["uid"])][int(p["frameIndex"])] = (float(p["x"]), float(p["y"]), bool(p.get("vis", False)))
    summary = []
    for uid, fr in tracks.items():
        vis_in = [i for f, i in fidx.items() if f in fr and fr[f][2]]
        first_vis = min(vis_in) if vis_in else T
        summary.append((-len(vis_in), first_vis, uid))
    summary.sort()
    n_total = len(summary)
    n_visible = sum(1 for s in summary if -s[0] > 0)
    chosen = [s[2] for s in summary[:N_CONTACT_SLOTS]]
    # slot order: first visible frame, then angle around the mean position of the chosen tracks
    means = {}
    for uid in chosen:
        pts = [tracks[uid][f][:2] for f in sampled_abs if int(f) in tracks[uid]]
        means[uid] = np.mean(np.asarray(pts, dtype=np.float64), axis=0) if pts else np.zeros(2)
    center = np.mean(np.stack([means[u] for u in chosen]), axis=0) if chosen else np.zeros(2)

    def slot_key(uid):
        s = next(x for x in summary if x[2] == uid)
        ang = np.arctan2(means[uid][1] - center[1], means[uid][0] - center[0])
        return (s[1], ang)
    chosen.sort(key=slot_key)
    xy = np.zeros((T, N_CONTACT_SLOTS, 2), dtype=np.float32)
    vis = np.zeros((T, N_CONTACT_SLOTS), dtype=np.float32)
    all_frames = sorted(set().union(*[set(tr.keys()) for tr in tracks.values()])) if tracks else []
    for k, uid in enumerate(chosen):
        tr = tracks[uid]
        for i, f in enumerate(sampled_abs):
            f = int(f)
            if f in tr:
                x, y, v = tr[f]
            else:  # nearest annotated frame (should not happen: uids exist in every frame)
                fn = min(tr.keys(), key=lambda g: abs(g - f))
                x, y, _ = tr[fn]
                v = False
            xy[i, k] = (x / ann_W, y / ann_H)
            vis[i, k] = 1.0 if v else 0.0
    xy_crop = builder.transform_points_to_crop(xy, ann_W, ann_H, frame_W, frame_H, tuple(bbox))
    xy_crop = np.clip(xy_crop, 0.0, 1.0).astype(np.float32)
    return xy_crop, vis, chosen, n_total, n_visible


def process_one(task):
    import cv2
    sid, clip_key, split = task
    case, stem = clip_key.split("/")
    dirs = make_dirs(OUT)
    stats = collections.Counter()
    try:
        ann_path = RAW / case / "annotation" / f"{stem}.json"
        item = builder.extract_base_sample(ann_path=ann_path, root_dir=RAW, sid=sid, dirs=dirs, stats=stats,
                                           start_offset_frames=0)
        if item is None:
            return {"sid": sid, "clip_key": clip_key, "ok": False, "stats": dict(stats), "error": "extract_base_sample returned None"}
        data = json.loads(ann_path.read_text())
        ann_W, ann_H = int(data["imageDimensions"]["width"]), int(data["imageDimensions"]["height"])
        video_path = RAW / case / "clips" / f"{stem}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        frame_W, frame_H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        sampled_abs = np.load(dirs["FRAME_INDEX_DIR"] / f"{sid}.npy").reshape(-1)
        bbox = item.crop_bbox_xyxy
        # keep the builder's geometric-order contact GT and soft visibilities for comparison
        os.replace(dirs["GT_CT_XY_DIR"] / f"{sid}.npy", OUT / "gt_ct_xy_geom" / f"{sid}.npy")
        os.replace(dirs["GT_CT_VIS_DIR"] / f"{sid}.npy", OUT / "gt_ct_vis_soft" / f"{sid}.npy")
        os.replace(dirs["GT_TOOL_VIS_DIR"] / f"{sid}.npy", OUT / "gt_tool_vis_soft" / f"{sid}.npy")
        # uid-based contact tracks, hard visibility, existence
        xy, vis, uids, n_total, n_visible = uid_contact_tracks(data, sampled_abs, ann_W, ann_H, frame_W, frame_H, bbox)
        np.save(dirs["GT_CT_XY_DIR"] / f"{sid}.npy", xy)
        np.save(dirs["GT_CT_VIS_DIR"] / f"{sid}.npy", vis)
        np.save(OUT / "gt_ct_exist" / f"{sid}.npy", (vis.max(axis=1) > 0).astype(np.float32))
        (OUT / "contact_uids" / f"{sid}.json").write_text(json.dumps(
            {"slots": [int(u) for u in uids], "n_tracks_total": n_total, "n_tracks_visible_in_window": n_visible}))
        # hard tool visibility = raw binary annotation visibility
        kv_raw = np.load(dirs["KEYPOINT_VIS_DIR"] / f"{sid}.npy").astype(np.float32)
        np.save(dirs["GT_TOOL_VIS_DIR"] / f"{sid}.npy", (kv_raw > 0.5).astype(np.float32))
        # action range features
        s, e = int(item.action_start_abs), int(item.action_end_abs)
        win = max(1, int(item.abs_end) - int(item.abs_start))
        f = sampled_abs.astype(np.float64)
        active = ((f >= s) & (f <= e)).astype(np.float32)
        progress = np.clip((f - s) / max(1, e - s), 0.0, 1.0).astype(np.float32) * active
        duration = np.full_like(active, (e - s) / win, dtype=np.float32)
        signed = np.where(f < s, f - s, np.where(f > e, f - e, 0.0)) / win
        feats = np.stack([active, progress, duration, signed.astype(np.float32)], axis=1).astype(np.float32)
        np.savez(OUT / "action_range_16" / f"{sid}.npz", features=feats, inside_action_range=active)
        row = dataclasses.asdict(item)
        row.update({"split": split, "clip_key": clip_key, "frame_W": frame_W, "frame_H": frame_H,
                    "ann_W": ann_W, "ann_H": ann_H, "contact_slots_uid": [int(u) for u in uids],
                    "contact_tracks_total": n_total, "contact_tracks_visible": n_visible})
        return {"sid": sid, "clip_key": clip_key, "ok": True, "stats": dict(stats), "row": row}
    except Exception as ex:  # noqa
        return {"sid": sid, "clip_key": clip_key, "ok": False, "stats": dict(stats),
                "error": f"{ex}\n{traceback.format_exc()}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    split = json.loads(SPLIT_JSON.read_text())["split"]
    keys = sorted(k for k, v in split.items() if v)
    # deterministic sids: sorted by (case, affordance start) == sorted clip_key order (stem = clip_<start>_<end>)
    keys.sort(key=lambda k: (k.split("/")[0], int(k.split("/")[1].split("_")[1]), int(k.split("/")[1].split("_")[2])))
    if args.limit:
        keys = keys[: args.limit]
    tasks = [(f"{i:06d}", k, split[k]) for i, k in enumerate(keys)]
    OUT.mkdir(parents=True, exist_ok=True)
    make_dirs(OUT)
    log = open(OUT / "export.log", "a")
    t0 = time.time()
    rows, failures, stats = [], [], collections.Counter()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for n, r in enumerate(pool.imap_unordered(process_one, tasks, chunksize=4), 1):
            stats.update(r["stats"])
            if r["ok"]:
                rows.append(r["row"])
            else:
                failures.append({"sid": r["sid"], "clip_key": r["clip_key"], "error": r["error"]})
            if n % 100 == 0 or n == len(tasks):
                msg = f"[{n}/{len(tasks)}] ok={len(rows)} fail={len(failures)} elapsed={time.time()-t0:.0f}s"
                print(msg, flush=True)
                log.write(msg + "\n"); log.flush()
    rows.sort(key=lambda r: r["sid"])
    (OUT / "meta.json").write_text(json.dumps(rows, indent=1))
    _init_worker()
    labels = {
        "tool2id": builder.TOOL2ID, "act2id": builder.ACT2ID, "combo2id_6class": builder.COMBO2ID,
        "tool2topology": builder.TOOL2TOPO, "output_mode": "uid_contact_v1", "target_frames": 16,
        "contact_slots": N_CONTACT_SLOTS, "visibility": "hard_binary", "fps": 60,
        "split_protocol": json.loads(SPLIT_JSON.read_text())["protocol"],
        "note": "gt_ct_xy = uid-consistent contact tracks (top-4 visible in window); gt_ct_xy_geom = builder geometric order; *_soft = builder soft weights",
    }
    (OUT / "labels.json").write_text(json.dumps(labels, indent=1))
    for s in ("train", "val", "test"):
        (OUT / f"{s}.txt").write_text("\n".join(r["sid"] for r in rows if r["split"] == s) + "\n")
    (OUT / "stats.json").write_text(json.dumps({"builder_stats": dict(stats), "n_ok": len(rows), "n_fail": len(failures),
                                                "failures": failures, "elapsed_s": time.time() - t0}, indent=1))
    print(f"done: ok={len(rows)} fail={len(failures)} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
