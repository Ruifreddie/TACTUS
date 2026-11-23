#!/usr/bin/env python3
"""Convert TactusNet's own dumps (UniTAP experiment dirs) into a standard kpt prediction file, so its Tab. 3 row is scored
by the same score.py as every baseline, and its predicted tips can feed the fixed-quad floor.

  How:   <exp>/results/pipeline_test/<sid>.npz          pred_xy [16,3,2], normalized to the content crop (full pipeline)
  Where: <exp>/experiments/analysis/pred_C006_best_test/<sid>.npz   pred_xy [16,4,2], normalized to the zoom box
         + <exp>/data/case32/v1_zoom/boxes.json          {sid: [bx0, by0, L]} in content-crop pixels
         mapped exactly like UniTAP tools/eval_contact_stage.py: (pxy * L + [bx0, by0]) / [W, H]

  python -m baselines.common.tactus --exp /work/exp/A --out OUT/tactusnet/pred_test.npz [--contact-dump DIR]
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

from . import preds as P
from .data import get_export


STAGE_KEYS = {"refined": "pred_xy", "stage2": "pred_xy_stage2", "stage1": "pred_xy_stage1"}


def load_pipeline(d, stage="refined"):
    return {Path(f).stem: np.load(f)[STAGE_KEYS[stage]].astype(np.float32) for f in sorted(glob.glob(os.path.join(d, "*.npz")))}


def _box_from_points(pts, margin_scale, margin_add, lmin, lmax, robust=False, W=896, H=665):
    """Verbatim from UniTAP tools/build_zoom_dataset.py (box_from_points): pts [N,2] native px -> (cx, cy, L)."""
    if len(pts) == 0:
        return W / 2.0, H / 2.0, float(lmax)
    if robust and len(pts) >= 8:
        x0, x1 = np.percentile(pts[:, 0], [5, 95]); y0, y1 = np.percentile(pts[:, 1], [5, 95])
    else:
        x0, x1 = pts[:, 0].min(), pts[:, 0].max(); y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    L = max(x1 - x0, y1 - y0) * margin_scale + margin_add
    L = float(np.clip(L, lmin, lmax))
    return float((x0 + x1) / 2), float((y0 + y1) / 2), L


def zoom_boxes_from_stage1(dump, ex, sids):
    """The zoom boxes of val/test, rebuilt exactly as build_zoom_dataset.py does for prediction-derived splits
    (defaults: margin 1.5, +96 px, L in [256, 720], robust percentiles), from the stage-1 dump pred_E007k_best_<split>."""
    out = {}
    for sid in sids:
        W, H = int(ex.meta[sid]["W"]), int(ex.meta[sid]["H"])
        pred = np.load(os.path.join(dump, f"{sid}.npz"))["pred_xy"].astype(np.float32).reshape(-1, 2) * [W, H]
        cx, cy, L = _box_from_points(pred, 1.5, 96.0, 256, 720, robust=True, W=W, H=H)
        L = int(round(L)); out[sid] = [int(round(cx - L / 2)), int(round(cy - L / 2)), L]
    return out


def load_contact(dump, boxes, ex):
    out = {}
    for f in sorted(glob.glob(os.path.join(dump, "*.npz"))):
        sid = Path(f).stem
        if sid not in boxes: continue
        bx0, by0, L = boxes[sid]; m = ex.meta[sid]; WH = np.array([float(m["W"]), float(m["H"])])
        out[sid] = ((np.load(f)["pred_xy"].astype(np.float64) * L + np.array([bx0, by0])) / WH).astype(np.float32)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True, help="UniTAP experiment dir, e.g. /work/exp/A")
    ap.add_argument("--split", default="test"); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="refined", choices=list(STAGE_KEYS), help="which tool points of the pipeline dump (ablation)")
    ap.add_argument("--pipeline-dir"); ap.add_argument("--contact-dump"); ap.add_argument("--boxes", help="boxes.json; rebuilt from --stage1-dump when missing")
    ap.add_argument("--stage1-dump"); ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--export"); ap.add_argument("--hf-raw"); a = ap.parse_args(argv); ex = get_export(a.export, a.hf_raw); E = Path(a.exp)
    sids = ex.sids(a.split)
    pipe = load_pipeline(a.pipeline_dir or E / f"results/pipeline_{a.split}", a.stage)
    arrays = {"tool_xy": None}
    if not a.no_contact:
        bj = Path(a.boxes or E / "data/case32/v1_zoom/boxes.json")
        if bj.exists():
            boxes = json.load(open(bj))["boxes"]; src = str(bj)
        else:
            s1 = a.stage1_dump or E / f"experiments/analysis/pred_E007k_best_{a.split}"
            boxes = zoom_boxes_from_stage1(s1, ex, sids); src = f"rebuilt from {s1}"
        cont = load_contact(a.contact_dump or E / f"experiments/analysis/pred_C006_best_{a.split}", boxes, ex)
        arrays["ct_xy"] = cont
    miss = [s for s in sids if s not in pipe or (not a.no_contact and s not in arrays["ct_xy"])]
    if miss: raise SystemExit(f"{len(miss)} clips missing from the dumps, e.g. {miss[:5]}")
    out = {"tool_xy": np.stack([pipe[s] for s in sids])}
    if not a.no_contact: out["ct_xy"] = np.stack([arrays["ct_xy"][s] for s in sids])
    P.save(a.out, "kpt", sids, {"method": "tactusnet", "exp": str(E), "split": a.split, "tool_stage": a.stage,
                                "boxes": None if a.no_contact else src}, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
