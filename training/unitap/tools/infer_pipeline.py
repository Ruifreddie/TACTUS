#!/usr/bin/env python
"""UniTAP single-clip inference WITHOUT token caches (for the per-case pipeline).

Input : 16 frames uint8 [16,H,W,3] in BGR (native size, e.g. 896x665), tool name and action name (labels.json vocab).
Stages: V-JEPA 2.1 tokens of the squashed full frame -> stage-1 head (E007k) -> 3 tool points
        -> square zoom box around the predicted trajectory -> crop+resize 384 -> V-JEPA tokens
        -> stage-2 head (E009) -> 3 tool points (mapped back)    and    contact head (C004) -> 4 contact points
        -> pixel refiner (E008b, native crops) on the stage-2 tool points.
Output: dict of normalized xy (relative to the native frame) per stage + the box; see `run_clip`.

Demo/validation:  python tools/infer_pipeline.py --demo-sids 000010 000188 --device cuda
   runs dataset clips end to end and reports ADE @384 vs GT for each stage (should match the cached-token numbers up to
   fp16 rounding of the cache).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

WS = Path(__file__).resolve().parent.parent
KC = WS / "third_party/t2b/kpt_combo"; sys.path.insert(0, str(KC)); sys.path.insert(0, str(KC.parent)); sys.path.insert(0, str(WS / "tools"))
from train import TrainableEMA, _derive_tool_action_id_from_combo_id, build_model, load_trainable_state_dict  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from build_zoom_dataset import box_from_points, finalize_box, crop_resize  # noqa: E402
import pixel_refiner as PR  # noqa: E402
sys.path.insert(0, str(WS / "third_party/vitpose_dataset_tools"))

def _pick(final, evaluated):
    """Prefer the DEPLOYMENT checkpoint (train+val+test, *_all) when it exists, else the train+val one (*_tv), else the
    case32-evaluated one."""
    final = Path(final); alt = Path(str(final).replace("_tv/", "_all/"))
    return str(alt) if alt.exists() else (str(final) if final.exists() else str(evaluated))

DEFAULT_CKPTS = dict(stage1=_pick(WS / "experiments/runs/F1_stage1_tv/out/checkpoints/best.pt", WS / "experiments/runs/E007k/out/checkpoints/best.pt"),
                     stage2=_pick(WS / "experiments/runs/F2_zoom_tv/out/checkpoints/best.pt", WS / "experiments/runs/E009/out/checkpoints/best.pt"),
                     contact=_pick(WS / "experiments/runs/F3_contact_hs_tv/out/checkpoints/best.pt", WS / "experiments/runs/C006/out/checkpoints/best.pt"),  # hindsight contact targets (TactusNet default)
                     refiner=_pick(WS / "experiments/runs/F4_refiner_tv/pixref/best.pt", WS / "experiments/runs/E008b/pixref/best.pt"))
LABELS = json.load(open(WS / "data/case32/v1/labels.json"))


def preprocess(frames_bgr_u8: np.ndarray, size: int = 384) -> torch.Tensor:
    """[T,H,W,3] uint8 BGR -> [1,T,3,size,size] float32, exactly like the token converter / dataset."""
    rgb = np.ascontiguousarray(frames_bgr_u8[..., ::-1])
    v = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
    v = (v - IMAGENET_MEAN) / IMAGENET_STD
    return v.unsqueeze(0)


class Pipeline:
    def __init__(self, ckpts=None, device="cuda", refine_decode="full"):
        ck = dict(DEFAULT_CKPTS); ck.update(ckpts or {})
        self.device = torch.device(device); self.refine_decode = refine_decode
        self.heads = {}; shared_vjepa = None
        for name in ("stage1", "stage2", "contact"):
            c = torch.load(ck[name], map_location="cpu"); args = argparse.Namespace(**dict(c["args"])); args.device = device
            m = build_model(args, num_combo_classes=len(LABELS["combo2id_6class"]))
            load_trainable_state_dict(m, c["modules"]); m.eval()
            if "ema" in c:
                ema = TrainableEMA(model=m, decay=float(getattr(args, "ema_decay", 0.999))); ema.load_state_dict(c["ema"], model=m); ema.apply_shadow(m)
            if shared_vjepa is None:
                shared_vjepa = m.vjepa
            else:
                m.vjepa = shared_vjepa  # one frozen encoder for all heads
            self.heads[name] = m.to(self.device)
        self.vjepa = shared_vjepa.to(self.device).eval()
        r = torch.load(ck["refiner"], map_location="cpu"); self.refiner = PR.PixelRefiner(w=r["args"].get("width", 32)); self.refiner.load_state_dict(r["model"]); self.refiner.to(self.device).eval()
        self.combo2id = LABELS["combo2id_6class"]; self.tool2topology = LABELS["tool2topology"]
        self.topology_roles = {"1A2T": [0, 1, 1], "2A1T": [0, 0, 1]}  # anchor=0, tip=1 (matches meta pt_role of the training data)

    @torch.no_grad()
    def tokens(self, frames_bgr_u8: np.ndarray) -> torch.Tensor:
        x = preprocess(frames_bgr_u8).to(self.device)
        out = self.vjepa(x)
        tok = out[0] if isinstance(out, (tuple, list)) else out
        return tok.float()  # [1,8,24,24,1024]

    @torch.no_grad()
    def head(self, name: str, tok: torch.Tensor, combo_id: int) -> np.ndarray:
        cid = torch.tensor([combo_id], dtype=torch.long, device=self.device); tool_id, action_id = _derive_tool_action_id_from_combo_id(cid)
        o = self.heads[name](video=None, tokens_grid=tok, token_order="cached_t_major_then_h_w", tool_id=tool_id, action_id=action_id, combo_id=cid)
        return o["pred_tool_xy"][0].float().cpu().numpy()  # [16,P,2] normalized to the input frame (full frame or zoom box)

    @torch.no_grad()
    def refine(self, frames_bgr_u8: np.ndarray, pts_nat: np.ndarray, tool: str) -> np.ndarray:
        """pts_nat [16,3,2] native px -> refined [16,3,2] native px (E008b crop refiner, 'full' soft-argmax decode)."""
        tool_id = PR.TOOLS.index(tool); roles = self.topology_roles.get(self.tool2topology.get(tool, ""), [0, 1, 1])
        xs, orgs, idx = [], [], []
        for t in range(pts_nat.shape[0]):
            for k in range(pts_nat.shape[1]):
                s = PR.make_sample(frames_bgr_u8[t], pts_nat[t].astype(np.float32), k, tool_id, int(roles[k]))  # refiner was trained directly on the npy byte order (BGR) -> feed the same
                xs.append(s["x"]); orgs.append(s["origin"]); idx.append((t, k))
        x = torch.from_numpy(np.stack(xs)).to(self.device); logits, trust = self.refiner(x); full, loc = self.refiner.decode(logits)
        dec = (full if self.refine_decode == "full" else loc).cpu().numpy() + np.stack(orgs)
        out = pts_nat.copy()
        for (t, k), p in zip(idx, dec): out[t, k] = p
        return out

    @torch.no_grad()
    def run_clip(self, frames_bgr_u8: np.ndarray, tool: str, action: str) -> dict:
        T, H, W = frames_bgr_u8.shape[:3]; WH = np.array([W, H], dtype=np.float64)
        combo = f"{tool}|{action}"; combo_id = int(self.combo2id[combo])
        t0 = time.time()
        tok_full = self.tokens(frames_bgr_u8); s1 = self.head("stage1", tok_full, combo_id)              # [16,3,2] norm (full frame)
        s1_nat = s1.astype(np.float64) * WH
        cx, cy, L = box_from_points(s1_nat.reshape(-1, 2), 1.5, 96.0, 256, 720, robust=True, W=W, H=H); bx0, by0, L = finalize_box(cx, cy, L)
        crop = crop_resize(frames_bgr_u8, bx0, by0, L)                                                    # [16,384,384,3] BGR
        tok_zoom = self.tokens(crop)
        s2 = self.head("stage2", tok_zoom, combo_id); ct = self.head("contact", tok_zoom, combo_id)          # box-normalized
        s2_nat = s2.astype(np.float64) * L + np.array([bx0, by0]); ct_nat = ct.astype(np.float64) * L + np.array([bx0, by0])
        s3_nat = self.refine(frames_bgr_u8, s2_nat, tool)
        return dict(tool_xy=s3_nat / WH, tool_xy_stage2=s2_nat / WH, tool_xy_stage1=s1, contact_xy=ct_nat / WH,
                    tool_xy_native=s3_nat, contact_xy_native=ct_nat, box=(bx0, by0, L), seconds=time.time() - t0)


def _ade(pred_norm, gt_norm, vis):
    e = np.linalg.norm((pred_norm - gt_norm) * 384, axis=-1)[vis > 0.5]; return float(e.mean()) if len(e) else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--demo-sids", nargs="*", default=[])
    ap.add_argument("--clip-npy", default=None, help="run on a [16,H,W,3] uint8 BGR npy"); ap.add_argument("--tool", default=None); ap.add_argument("--action", default=None)
    ap.add_argument("--out", default=None, help="json output path"); ap.add_argument("--refine-decode", default="full", choices=["full", "loc"])
    ap.add_argument("--split", default=None, help="run every clip of a dataset split online and write per-clip npz + summary metrics"); ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--video", default=None, help="annotation mode: mp4 of the clip (or full video); frames are sampled like training (16 uniform over [clip-start, clip-end])")
    ap.add_argument("--clip-start", type=int, default=None, help="first frame (affordance/clip range) as a LOCAL index in --video"); ap.add_argument("--clip-end", type=int, default=None)
    ap.add_argument("--action-start", type=int, default=None, help="annotated action range (local indices); only marks which output frames are inside the action"); ap.add_argument("--action-end", type=int, default=None)
    ap.add_argument("--contact-ckpt", default=None, help="contact head checkpoint: default C006 (hindsight interaction site = where the tool acts within the clip); experiments/runs/C005/out/checkpoints/best.pt = human-annotated 4-corner region")
    a = ap.parse_args()
    pipe = Pipeline(ckpts=({"contact": a.contact_ckpt} if a.contact_ckpt else None), device=a.device, refine_decode=a.refine_decode)
    if a.split:
        meta = {m["sid"]: m for m in json.load(open(WS / "data/case32/v1/meta.json"))}
        sids = [l.strip() for l in open(WS / f"data/case32/v1/{a.split}.txt") if l.strip()]
        dd = Path(a.dump_dir or (WS / f"experiments/analysis/pipeline_{a.split}")); dd.mkdir(parents=True, exist_ok=True)
        os.chdir(WS)  # PR.read_clip_frames uses a relative ROOT
        E = {k: [] for k in ("stage1", "stage2", "refined")}; secs = []
        for i, sid in enumerate(sids):
            m = meta[sid]; frames = PR.read_clip_frames(sid); r = pipe.run_clip(frames, m["tool"], m["action"]); secs.append(r["seconds"])  # read + drop from page cache
            gt = np.load(WS / f"data/case32/v1/gt_tool_xy/{sid}.npy"); vis = np.load(WS / f"data/case32/v1/gt_tool_vis/{sid}.npy") > 0.5
            for k, pr in (("stage1", r["tool_xy_stage1"]), ("stage2", r["tool_xy_stage2"]), ("refined", r["tool_xy"])):
                E[k].append(np.linalg.norm((pr - gt) * 384, axis=-1)[vis])
            np.savez_compressed(dd / f"{sid}.npz", pred_xy=r["tool_xy"].astype(np.float32), pred_xy_stage2=r["tool_xy_stage2"].astype(np.float32), pred_xy_stage1=r["tool_xy_stage1"].astype(np.float32),
                                contact_xy=r["contact_xy"].astype(np.float32), box=np.array(r["box"]), gt_xy=gt.astype(np.float32), gt_vis=vis.astype(np.float32))
            if (i + 1) % 50 == 0: print(f"[{i + 1}/{len(sids)}] mean {np.mean(secs):.2f} s/clip", flush=True)
        summ = {}
        for k, v in E.items():
            e = np.concatenate(v); summ[k] = dict(ADE=float(e.mean()), med=float(np.median(e)), **{f"PCK{t}": float((e < t).mean()) for t in (4, 8, 12, 16)}, frac_gt32=float((e > 32).mean()))
        summ["seconds_per_clip_mean"] = float(np.mean(secs)); summ["n_clips"] = len(sids)
        print(json.dumps(summ, indent=1)); json.dump(summ, open(dd / "summary_tool.json", "w"), indent=1)
    elif a.demo_sids:
        meta = {m["sid"]: m for m in json.load(open(WS / "data/case32/v1/meta.json"))}
        rows = []
        for sid in a.demo_sids:
            m = meta[sid]; frames = np.load(WS / f"data/case32/v1/video_16_npy/{sid}.npy")
            r = pipe.run_clip(frames, m["tool"], m["action"])
            gt = np.load(WS / f"data/case32/v1/gt_tool_xy/{sid}.npy"); vis = np.load(WS / f"data/case32/v1/gt_tool_vis/{sid}.npy")
            gct = np.load(WS / f"data/case32/v1/gt_ct_xy/{sid}.npy"); cvis = np.load(WS / f"data/case32/v1/gt_ct_vis/{sid}.npy")
            ref = {}
            for name, d in (("cache_E007k", "pred_E007k_best_val"), ("cache_E009", "pred_E009_best_val_orig")):
                p = WS / f"experiments/analysis/{d}/{sid}.npz"
                if p.exists(): ref[name] = _ade(np.load(p)["pred_xy"], gt, vis)
            row = dict(sid=sid, combo=m["combo"], box=r["box"], sec=round(r["seconds"], 2), ADE_stage1=_ade(r["tool_xy_stage1"], gt, vis), ADE_stage2=_ade(r["tool_xy_stage2"], gt, vis),
                       ADE_refined=_ade(r["tool_xy"], gt, vis), contact_ADE_uid_order=_ade(r["contact_xy"], gct, cvis), **{k: round(v, 2) for k, v in ref.items()})
            print(json.dumps(row)); rows.append(row)
        if a.out: json.dump(rows, open(a.out, "w"), indent=1)
    elif a.video:
        import cv2, build_phase4p5_delta_aug as BLD
        cs, ce = int(a.clip_start), int(a.clip_end); idx = BLD.sample_fixed_abs_indices(cs, ce, 16)
        frames = BLD.read_video_frames_at_local_indices(Path(a.video), [int(i) for i in idx])
        if any(f is None for f in frames): raise SystemExit(f"could not read all 16 frames from {a.video} (indices {idx.tolist()})")
        bbox = BLD.detect_content_bbox(frames[0])                      # black bars + UI bars, same as the dataset builder (no point-aware expansion at inference)
        crop = np.stack([BLD.crop_frame(f, bbox) for f in frames])     # [16,h,w,3] BGR
        r = pipe.run_clip(crop, a.tool, a.action); x0, y0, x1, y1 = bbox; cw, ch = x1 - x0, y1 - y0
        def to_video_px(xy_norm): return (np.asarray(xy_norm) * [cw, ch] + [x0, y0]).tolist()
        act = [bool(a.action_start is not None and a.action_end is not None and a.action_start <= int(i) <= a.action_end) for i in idx]
        out = dict(video=a.video, clip_range=[cs, ce], action_range=[a.action_start, a.action_end], content_bbox=[int(v) for v in bbox], sampled_frames=[int(i) for i in idx], inside_action=act,
                   tool_xy_video_px=to_video_px(r["tool_xy"]), contact_xy_video_px=to_video_px(r["contact_xy"]), tool_xy_norm_crop=r["tool_xy"].tolist(), contact_xy_norm_crop=r["contact_xy"].tolist(),
                   zoom_box_crop_px=[int(v) for v in r["box"]], seconds=r["seconds"])
        print(json.dumps({k: out[k] for k in ("content_bbox", "sampled_frames", "inside_action", "seconds")}))
        if a.out: json.dump(out, open(a.out, "w"))
    elif a.clip_npy:
        frames = np.load(a.clip_npy); r = pipe.run_clip(frames, a.tool, a.action)
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        print(json.dumps({k: out[k] for k in ("box", "seconds")}))
        if a.out: json.dump(out, open(a.out, "w"))
    else:
        print("nothing to do: pass --demo-sids or --clip-npy")


if __name__ == "__main__":
    main()
