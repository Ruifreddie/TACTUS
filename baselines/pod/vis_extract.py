#!/usr/bin/env python3
"""Data for the supplement's qualitative comparison: the Gold labels and every Table 3 method's predictions on a few
random test clips, plus the shown frame of each clip.

The clips are drawn at random (fixed seed) with at most one clip per pair and one per source, among test clips that
have a sampled frame inside the action interval with all three Gold tool points and at least three Gold contact points
visible. That frame, the one nearest the middle of the interval, is the shown frame.

Read-only on the volume. Everything comes from the prediction files that score.py scores (collect_v4.rows), except
SegFormer's region: its file keeps only per-frame Dice, so its best.pt (state/baselines/segformer) runs again on the clip (CPU, fp32). The
re-run's per-frame Dice is checked against the stored one. Writes OUT/data.json, OUT/<sid>.jpg (the shown frame, content
crop) and OUT/<sid>_segformer.png (its mask on the 384x384 metric grid), then OUT.tgz and OUT.b64.

  python -m baselines.pod.vis_extract --bl-out $GV/results/baselines/runs --raw $GV/raw --out /work/vis [--k 4] [--seed 20261107]
"""
import argparse, base64, hashlib, io, json, tarfile
from pathlib import Path

import numpy as np

from baselines.common import config
from baselines.common import metrics as M
from baselines.common import preds as P
from baselines.common.data import get_export
from baselines.pod.collect_v4 import rows

NAMES = {"T3a/tsn": "TSN", "T3a/tsm": "TSM", "T3a/videomae": "VideoMAE", "T3a/xclip": "X-CLIP", "T3a/endofm": "Endo-FM",
         "T3a/tactusnet": "TactusNet", "T3b/actionformer_s8": "ActionFormer", "T3b/tridet_s8": "TriDet",
         "T3b/qddetr_s8": "QD-DETR", "T3b/mstcn_s32": "MS-TCN", "T3b/tactusnet": "TactusNet", "T3c/kpt_hrnet": "DARK",
         "T3c/kpt_vitpose": "ViTPose", "T3c/kpt_dsta": "DSTA", "T3c/segformer": "SegFormer", "T3c/tactusnet": "TactusNet"}


def window_pos(ex, sid):
    sab = ex.sampled_abs(sid).astype(np.float64)
    return (sab - sab[0]) / max(sab[-1] - sab[0], 1e-9)


def shown_frame(ex, sid):
    """The sampled frame inside the interval, nearest its middle, with 3 visible Gold tool points and >= 3 contact points."""
    s, e, ok = ex.interval(sid)
    if ok < 0.5 or e <= s: return None
    pos = window_pos(ex, sid); txy, tvis = ex.tool_points(sid); cxy, cvis = ex.contact_points(sid)
    cand = [t for t in range(16) if s <= pos[t] <= e and tvis[t].sum() == 3 and cvis[t].sum() >= 3]
    return min(cand, key=lambda t: abs(pos[t] - (s + e) / 2)) if cand else None


def pick(ex, k, seed, have):
    """First pass: at most one clip per pair and per source. If that leaves fewer than k clips (few clipper and scissors
    clips show all three tool points), a second pass over the same random order drops the source rule."""
    test = [s for s in ex.sids("test") if all(s in h for h in have)]
    order = np.random.RandomState(seed).permutation(len(test)); out, pairs, srcs = [], set(), set()
    for one_per_source in (True, False):
        for j in order:
            sid = test[j]; p, src = ex.pair_id(sid), ex.source(sid)
            if p in pairs or (one_per_source and src in srcs) or any(sid == o for o, _ in out): continue
            t = shown_frame(ex, sid)
            if t is None: continue
            out.append((sid, t)); pairs.add(p); srcs.add(src)
            if len(out) == k: return out
    return out


def fetch_clip(ex, sid, raw):
    """Copy the clip's mp4 out of <raw>/<task>.tar into the HF layout that Export.clip_path expects."""
    dst = ex.clip_path(sid)
    if dst.is_file(): return dst
    task, stem = ex.meta[sid]["clip_key"].split("/")
    with tarfile.open(Path(raw) / f"{task}.tar") as tf:
        mem = next(m for m in tf if m.name.endswith(f"/clips/{stem}.mp4"))
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(tf.extractfile(mem).read())
    return dst


def segformer_masks(ex, sid, ckpt, weights):
    """Masks [16,384,384] of SegFormer's best.pt on the clip, exactly as segformer.predict reads them (fp32 on CPU)."""
    import torch
    import torch.nn.functional as F
    from baselines.where import segformer as SF
    model = SF.SegFormerWhere(SF.ensure_weights(weights), len(ex.pairs))
    st = torch.load(ckpt, map_location="cpu", weights_only=False); model.load_state_dict(st.get("model", st)); model.eval()
    x = torch.from_numpy(np.stack([SF.test_image(f) for f in ex.frames(sid)])); S = config.METRIC_SIZE
    with torch.no_grad():
        zs = [model(x[c:c + 2], torch.full((len(x[c:c + 2]),), ex.pair_id(sid), dtype=torch.long)) for c in range(0, 16, 2)]
        z = F.interpolate(torch.cat(zs)[:, None].float(), size=(S, S), mode="bilinear", align_corners=False)[:, 0]
    return (z > 0).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-out", required=True); ap.add_argument("--ao"); ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--weights", default="/work/weights/mit-b2")
    ap.add_argument("--seg-ckpt", default="/workspace-global/state/baselines/segformer/best.pt")
    a = ap.parse_args(argv)
    bl = Path(a.bl_out); ao = Path(a.ao) if a.ao else bl / "ablate"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ex = get_export()
    preds = {k: P.load(p) for k, p in rows(bl, ao) if k.startswith("T3") and p.is_file()}
    missing = [k for k, p in rows(bl, ao) if k.startswith("T3") and not p.is_file()]
    print("prediction files:", len(preds), "missing:", missing, flush=True)
    idx = {k: {s: i for i, s in enumerate(v["sids"])} for k, v in preds.items()}
    clips = pick(ex, a.k, a.seed, list(idx.values()))
    print("clips:", [(s, ex.source(s), ex.pairs[ex.pair_id(s)], t) for s, t in clips], flush=True)
    import cv2
    data = {"seed": a.seed, "rule": "random test clips, at most one per pair and one per source; shown frame = sampled frame inside "
            "the interval nearest its middle with 3 visible Gold tool points and >= 3 visible contact points", "clips": []}
    for sid, t in clips:
        fetch_clip(ex, sid, a.raw); fr = ex.frames(sid, rgb=False); H, W = fr.shape[1:3]
        cv2.imwrite(str(out / f"{sid}.jpg"), fr[t], [cv2.IMWRITE_JPEG_QUALITY, 92])
        gs, ge, _ = ex.interval(sid); txy, tvis = ex.tool_points(sid); cxy, cvis = ex.contact_points(sid)
        c = {"sid": sid, "source": ex.source(sid), "pair": ex.pairs[ex.pair_id(sid)], "frame": int(t), "size": [int(W), int(H)],
             "window_pos": window_pos(ex, sid).round(4).tolist(), "gold": {"interval": [gs, ge], "tool_xy": txy[t].tolist(),
             "tool_vis": tvis[t].tolist(), "ct_xy": cxy[t].tolist(), "ct_vis": cvis[t].tolist()}, "methods": {}}
        for k, pr in preds.items():
            i = idx[k][sid]; m = {"name": NAMES.get(k, k)}
            if pr["kind"] == "cls":
                m["pair_pred"] = ex.pairs[int(pr["pair_logits"][i].argmax())]
            elif pr["kind"] == "when":
                ps, pe = float(pr["start"][i]), float(pr["end"][i]); m.update(interval=[ps, pe], tiou=M.tiou(ps, pe, gs, ge))
            else:
                if "tool_xy" in pr:
                    e = M.tool_errors(pr["tool_xy"][i], txy, tvis); m.update(tool_xy=pr["tool_xy"][i][t].tolist(), ade_clip=float(e.mean()),
                                                                              ade_frame=float(M.tool_errors(pr["tool_xy"][i][t:t + 1], txy[t:t + 1], tvis[t:t + 1]).mean()))
                if "ct_xy" in pr:
                    d = M.contact_frame_dice(cxy, cvis, pred_ct_xy=pr["ct_xy"][i]); m.update(ct_xy=pr["ct_xy"][i][t].tolist(),
                                                                                           dice_clip=float(np.nanmean(d)), dice_frame=float(d[t]))
                if "ct_frame_dice" in pr:
                    d = pr["ct_frame_dice"][i].astype(np.float64); m.update(dice_clip=float(np.nanmean(d)), dice_frame=float(d[t]))
                    masks = segformer_masks(ex, sid, a.seg_ckpt, a.weights)
                    d2 = M.contact_frame_dice(cxy, cvis, pred_masks=masks); m["dice_rerun_max_abs_diff"] = float(np.nanmax(np.abs(d2 - d)))
                    cv2.imwrite(str(out / f"{sid}_segformer.png"), masks[t].astype(np.uint8) * 255)
                    print(sid, "SegFormer re-run vs stored per-frame Dice, max |diff|:", m["dice_rerun_max_abs_diff"], flush=True)
            c["methods"][k] = m
        data["clips"].append(c)
    (out / "data.json").write_text(json.dumps(data, indent=1))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(out.iterdir()): tf.add(str(f), arcname=f"vis/{f.name}")
    raw = buf.getvalue(); Path(str(out) + ".tgz").write_bytes(raw); s = base64.b64encode(raw).decode()
    Path(str(out) + ".b64").write_text("\n".join(s[i:i + 100] for i in range(0, len(s), 100)) + "\n")
    print("WROTE", str(out) + ".b64", len(raw), "bytes, md5", hashlib.md5(raw).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
