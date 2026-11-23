#!/usr/bin/env python
"""Offline PRE-ANNOTATION of surgical clips (human-in-the-loop workflow):
the model predicts, for the 16 sampled frames of a clip, the 3 tool keypoints and the 4-point interaction region
(hindsight definition), with a per-frame confidence and a suggested best frame; a human picks one frame and propagates
it with CoTracker (optional --track-from / --auto-track here).

Manifest (JSON lines), one clip per line:
  {"video": ".../clip.mp4", "clip_start": 0, "clip_end": 83, "action_start": 22, "action_end": 66, "tool": "hook", "action": "dissect", "id": "any"}
  frame indices are LOCAL to the video file. Alternatively --split test|val runs case32 clips from the raw mp4s (validation).

Output per clip: <out>/<id>.json in the raw-annotation style:
  imageDimensions, tool, action, clip_range, action_range, content_bbox, sampled_frames, suggested_frame,
  points: [{id, uid, type: tool_anchor|tool_tip|contact, frameIndex, x, y, vis, conf, inside_action}], (+ tracks if tracked)
Coordinates are pixels of the ORIGINAL video frame.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "tools")); sys.path.insert(0, str(WS / "third_party/vitpose_dataset_tools")); sys.path.insert(0, str(WS / "third_party/t2b")); sys.path.insert(0, str(WS / "third_party/t2b/kpt_combo"))
import infer_pipeline as IP  # noqa: E402
import build_phase4p5_delta_aug as BLD  # noqa: E402

TOPO = {"1A2T": ["tool_anchor", "tool_tip", "tool_tip"], "2A1T": ["tool_anchor", "tool_anchor", "tool_tip"]}


class Annotator(IP.Pipeline):
    """Pipeline + per-point confidences (sigmoid of the heads' conf logits)."""

    @torch.no_grad()
    def head_with_conf(self, name, tok, combo_id):
        cid = torch.tensor([combo_id], dtype=torch.long, device=self.device); tool_id, action_id = IP._derive_tool_action_id_from_combo_id(cid)
        m = self.heads[name]; o = m(video=None, tokens_grid=tok, token_order="cached_t_major_then_h_w", tool_id=tool_id, action_id=action_id, combo_id=cid)
        xy = o["pred_tool_xy"][0].float().cpu().numpy()
        cl = getattr(getattr(m, "tool_prob_head", None), "last_conf_logit", None)
        conf = torch.sigmoid(cl[0].float()).cpu().numpy() if torch.is_tensor(cl) else np.full(xy.shape[:2], np.nan, np.float32)
        return xy, conf

    @torch.no_grad()
    def run_clip_conf(self, frames_bgr_u8, tool, action):
        T, H, W = frames_bgr_u8.shape[:3]; WH = np.array([W, H], float); combo_id = int(self.combo2id[f"{tool}|{action}"]); t0 = time.time()
        tok_full = self.tokens(frames_bgr_u8); s1, c1 = self.head_with_conf("stage1", tok_full, combo_id)
        s1_nat = s1.astype(float) * WH
        cx, cy, L = IP.box_from_points(s1_nat.reshape(-1, 2), 1.5, 96.0, 256, 720, robust=True, W=W, H=H); bx0, by0, L = IP.finalize_box(cx, cy, L)
        tok_zoom = self.tokens(IP.crop_resize(frames_bgr_u8, bx0, by0, L))
        s2, c2 = self.head_with_conf("stage2", tok_zoom, combo_id); ct, cc = self.head_with_conf("contact", tok_zoom, combo_id)
        s2_nat = s2.astype(float) * L + [bx0, by0]; ct_nat = ct.astype(float) * L + [bx0, by0]
        s3_nat = self.refine(frames_bgr_u8, s2_nat, tool)
        return dict(tool_xy=s3_nat / WH, tool_conf=c2, contact_xy=ct_nat / WH, contact_conf=cc, box=(bx0, by0, L), seconds=time.time() - t0)


def track_with_cotracker(video, local_frames, query_frame_idx, pts_px, device):
    """Bidirectional CoTracker3 (offline) from the chosen frame: pts_px [N,2] in video px -> tracks [T,N,2], vis [T,N]."""
    import cv2
    cap = cv2.VideoCapture(str(video)); frames = []
    for i in local_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if not ok: raise RuntimeError(f"cannot read frame {i}")
        frames.append(f[..., ::-1].copy())
    cap.release()
    vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(device)  # [1,T,3,H,W] RGB 0-255
    model = torch.hub.load(str(WS / "research/multiscale_refs/repos/co-tracker"), "cotracker3_offline", source="local").to(device)
    q = torch.tensor([[float(query_frame_idx), float(x), float(y)] for x, y in pts_px], device=device)[None]
    tracks, vis = model(vid, queries=q)          # [1,T,N,2], [1,T,N]
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=None); ap.add_argument("--split", default=None, help="validation: run case32 clips of this split from the raw mp4s")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--contact-ckpt", default=None)
    ap.add_argument("--conf-vis-thresh", type=float, default=0.3, help="tool point vis flag = conf >= thresh"); ap.add_argument("--auto-track", action="store_true", help="propagate the suggested frame with CoTracker to every frame of the clip")
    ap.add_argument("--track-from", type=int, default=None, help="single-clip mode: local frame index chosen by the human; propagate with CoTracker"); a = ap.parse_args()
    ann = Annotator(ckpts=({"contact": a.contact_ckpt} if a.contact_ckpt else None), device=a.device); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.split:
        meta = json.load(open(WS / "data/case32/v1/meta.json"))
        for m in meta:
            if m["split"] != a.split: continue
            vid = WS / "data/raw_case32" / m["case"] / "clips" / (Path(m["ann_relpath"]).stem + ".mp4"); b = m["video_base_abs"]
            jobs.append(dict(id=m["sid"], video=str(vid), clip_start=m["abs_start"] - b, clip_end=m["abs_end"] - b, action_start=m["action_start_abs"] - b, action_end=m["action_end_abs"] - b, tool=m["tool"], action=m["action"]))
    else:
        jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.limit: jobs = jobs[: a.limit]
    summ = []
    for j in jobs:
        cs, ce = int(j["clip_start"]), int(j["clip_end"]); idx = BLD.sample_fixed_abs_indices(cs, ce, 16); frames = BLD.read_video_frames_at_local_indices(Path(j["video"]), [int(i) for i in idx])
        if any(f is None for f in frames): print("skip (unreadable frames):", j.get("id")); continue
        bbox = BLD.detect_content_bbox(frames[0]); crop = np.stack([BLD.crop_frame(f, bbox) for f in frames]); x0, y0, x1, y1 = bbox; cw, ch = x1 - x0, y1 - y0
        r = ann.run_clip_conf(crop, j["tool"], j["action"]); Hf, Wf = frames[0].shape[:2]
        tool_px = np.asarray(r["tool_xy"]) * [cw, ch] + [x0, y0]; ct_px = np.asarray(r["contact_xy"]) * [cw, ch] + [x0, y0]
        tc, cc = np.nan_to_num(r["tool_conf"], nan=0.5), np.nan_to_num(r["contact_conf"], nan=0.5)
        inside = [bool(j.get("action_start") is not None and int(j["action_start"]) <= int(i) <= int(j["action_end"])) for i in idx]
        frame_score = tc.mean(1) + cc.mean(1) + 0.5 * np.array(inside, float); sug = int(np.argmax(frame_score))
        roles = TOPO[ann.tool2topology.get(j["tool"], "1A2T")]; pts = []; pid = 0
        for ti, fi in enumerate(idx):
            for k in range(3):
                pts.append(dict(id=pid, uid=k, type=roles[k], frameIndex=int(fi), x=float(tool_px[ti, k, 0]), y=float(tool_px[ti, k, 1]), vis=bool(tc[ti, k] >= a.conf_vis_thresh), conf=float(tc[ti, k]), inside_action=inside[ti])); pid += 1
            for k in range(4):
                pts.append(dict(id=pid, uid=10 + k, type="contact", frameIndex=int(fi), x=float(ct_px[ti, k, 0]), y=float(ct_px[ti, k, 1]), vis=True, conf=float(cc[ti, k]), inside_action=inside[ti])); pid += 1
        rec = dict(id=j.get("id"), video=j["video"], imageDimensions=dict(width=int(Wf), height=int(Hf)), tool=j["tool"], action=j["action"], clip_range=[cs, ce], action_range=[j.get("action_start"), j.get("action_end")],
                   content_bbox=[int(v) for v in bbox], sampled_frames=[int(i) for i in idx], suggested_frame=int(idx[sug]), frame_scores=[float(v) for v in frame_score], points=pts, seconds=round(r["seconds"], 3))
        tf = a.track_from if a.track_from is not None else (int(idx[sug]) if a.auto_track else None)
        if tf is not None:
            ti = int(np.argmin(np.abs(idx - tf))); q = np.concatenate([tool_px[ti], ct_px[ti]], 0)
            tracks, vis = track_with_cotracker(j["video"], list(range(cs, ce + 1)), tf - cs, q, ann.device)
            rec["tracks"] = dict(from_frame=int(idx[ti]), frames=list(range(cs, ce + 1)), point_types=roles + ["contact"] * 4, xy=tracks.tolist(), vis=vis.tolist())
        json.dump(rec, open(out / f"{j.get('id', Path(j['video']).stem)}.json", "w")); summ.append((j.get("id"), r["seconds"]))
        print(f"{j.get('id')}: {len(pts)} points, suggested frame {int(idx[sug])}, {r['seconds']:.2f}s", flush=True)
    print(f"done {len(summ)} clips -> {out}")


if __name__ == "__main__":
    main()
