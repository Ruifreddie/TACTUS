#!/usr/bin/env python3
"""TactusNet's tool-point outputs of the ablations -> standard kpt prediction files (scored by baselines.common.score).

  dump    a UniTAP prediction dump of a full-frame head (tools/dump_val_predictions.py: <dir>/<sid>.npz, pred_xy [16,3,2]
          normalized to the content crop), e.g. the stage-1 tool decoder retrained on another encoder's tokens.
  refine  stage-1 points refined by TactusNet's pixel refiner WITHOUT the zoom stage (row "stage 1 + refiner, no zoom"):
          exactly tools/infer_pipeline.py Pipeline.refine ('full' soft-argmax decode, native BGR crops, every point of
          every sampled frame), applied to the stage-1 dump instead of the stage-2 points.

  python -m baselines.ablate.convert dump   --dump <exp>/experiments/analysis/pred_E007k_short_best_test --out OUT/pred_test.npz
  python -m baselines.ablate.convert refine --exp <exp A> --dump <exp A>/results/pipeline_test --key pred_xy_stage1 --out OUT/pred_test.npz
"""
import argparse, glob, json, os, sys
from pathlib import Path

import numpy as np

from baselines.common import preds as P
from baselines.common.data import get_export


def load_dump(d, key="pred_xy"):
    out = {Path(f).stem: np.load(f)[key].astype(np.float32) for f in sorted(glob.glob(os.path.join(d, "*.npz")))}
    if not out: raise SystemExit(f"no dump files in {d}")
    return out


def save_tool(ex, split, pts, out, info):
    sids = ex.sids(split); miss = [s for s in sids if s not in pts]
    if miss: raise SystemExit(f"{len(miss)} {split} clips missing, e.g. {miss[:5]}")
    bad = [s for s in sids if pts[s].shape != (16, 3, 2)]
    if bad: raise SystemExit(f"tool points must be [16,3,2], e.g. {bad[0]}: {pts[bad[0]].shape}")
    P.save(out, "kpt", sids, info, tool_xy=np.stack([pts[s] for s in sids])); print("wrote", out)


def cmd_dump(a):
    ex = get_export(a.export, a.hf_raw)
    save_tool(ex, a.split, load_dump(a.dump), a.out, {"method": "tactusnet", "source": str(a.dump), "split": a.split, **dict(kv.split("=", 1) for kv in a.info)})


def cmd_refine(a):
    import torch
    E = Path(a.exp); sys.path.insert(0, str(E / "tools")); import pixel_refiner as PR   # noqa: E402
    ex = get_export(a.export, a.hf_raw); dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    ck = Path(a.ckpt or E / "experiments/runs/E008b/pixref/best.pt"); r = torch.load(ck, map_location="cpu", weights_only=False)
    net = PR.PixelRefiner(w=r["args"].get("width", 32)); net.load_state_dict(r["model"]); net.to(dev).eval()
    topo = json.load(open(ex.root / "labels.json"))["tool2topology"]; roles_of = {"1A2T": [0, 1, 1], "2A1T": [0, 0, 1]}
    s1 = load_dump(a.dump, a.key); sids = ex.sids(a.split); out = {}
    with torch.no_grad():
        for i, sid in enumerate(sids):
            frames = ex.frames(sid, rgb=False)                                   # the refiner was trained on the BGR npy bytes
            tool = ex.meta[sid]["tool"]; tid = PR.TOOLS.index(tool); roles = roles_of.get(topo.get(tool, ""), [0, 1, 1])
            H, W = frames.shape[1:3]; WH = np.array([W, H], np.float64); pts = s1[sid].astype(np.float64) * WH
            xs, orgs, idx = [], [], []
            for t in range(pts.shape[0]):
                for k in range(pts.shape[1]):
                    smp = PR.make_sample(frames[t], pts[t].astype(np.float32), k, tid, int(roles[k]))
                    xs.append(smp["x"]); orgs.append(smp["origin"]); idx.append((t, k))
            logits, _ = net(torch.from_numpy(np.stack(xs)).to(dev)); full, _loc = net.decode(logits)
            dec = full.cpu().numpy() + np.stack(orgs); ref = pts.copy()
            for (t, k), p in zip(idx, dec): ref[t, k] = p
            out[sid] = (ref / WH).astype(np.float32)
            if (i + 1) % 200 == 0: print(f"[refine] {i + 1}/{len(sids)}", flush=True)
    save_tool(ex, a.split, out, a.out, {"method": "tactusnet", "row": "stage 1 + pixel refiner, no zoom", "stage1_dump": f"{a.dump} [{a.key}]",
                                         "refiner": str(ck), "decode": "full", "split": a.split})


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--dump", required=True); d.add_argument("--info", nargs="*", default=[], help="key=value pairs for info")
    r = sub.add_parser("refine"); r.add_argument("--exp", required=True); r.add_argument("--dump", required=True); r.add_argument("--ckpt")
    r.add_argument("--key", default="pred_xy", help="points to refine: pred_xy of a stage-1 dump, or pred_xy_stage1 of the pipeline dumps")
    r.add_argument("--device", default="cuda")
    for p in (d, r):
        p.add_argument("--out", required=True); p.add_argument("--split", default="test"); p.add_argument("--export"); p.add_argument("--hf-raw")
    a = ap.parse_args(argv)
    {"dump": cmd_dump, "refine": cmd_refine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
