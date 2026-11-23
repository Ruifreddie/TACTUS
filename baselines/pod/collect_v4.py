#!/usr/bin/env python3
"""Collect the numbers of Tables 3 and 4 and of the supplement's nine-video comparison from the prediction files on the pod.

Every file is scored with baselines.common.score.score_file (the code of every row of the paper), twice: on all test clips,
and without the two test videos that were in Endo-FM's pretraining data (the other nine test videos). Prints one line per
row and writes everything to a JSON file. Missing files are listed, not fatal, so the script can run while jobs finish.

  python -m baselines.pod.collect_v4 [--bl-out $BL_OUT] [--ao $BL_OUT/ablate] [--out $BL_OUT/tables_v4.json] [--boot]
"""
import argparse, json, os, sys
from pathlib import Path

from baselines.common import preds as P
from baselines.common.data import get_export
from baselines.common.score import score_file

ENDOFM_PRETRAIN_TASKS = ("cholec80_13", "cholec80_22")   # in Endo-FM's pretraining data (cls/run.sh, CLS_ENDOFM_EXCLUDE)


def rows(bl, ao):
    """(key, prediction file) for every row of Tables 3 and 4 that comes from a prediction file."""
    r = []
    for m in ("tsn", "tsm", "videomae", "xclip", "endofm"):
        r.append((f"T3a/{m}", bl / "cls" / m / "pred_test.npz"))
    r.append(("T3a/tactusnet", ao / "enc/ours/cls/pred_test.npz"))                 # Pair mAcc of TactusNet
    for m in ("actionformer_s8", "tridet_s8", "qddetr_s8", "mstcn_s32"):   # the official config ("main"), never the b64 sweep entry
        r.append((f"T3b/{m}", bl / "when" / m / "main" / "seed0" / "pred_test.npz"))
    r.append(("T3b/tactusnet", ao / "enc/ours/when/pred_test.npz"))
    for m in ("kpt_hrnet", "kpt_vitpose", "kpt_dsta"):
        r.append((f"T3c/{m}", bl / m / "pred_test.npz"))
    r.append(("T3c/segformer", bl / "segformer" / "pred_test.npz"))
    r.append(("T3c/tactusnet", ao / "silver/A/pred_test.npz"))                      # experiment A, tool and contact points
    for bb in ("videomae_l224", "endofm_b224", "vjepa2_l256"):
        for g in ("cls", "when", "stage1"):
            r.append((f"T4a/{bb}/{g}", ao / "enc" / bb / g / "pred_test.npz"))
    r.append(("T4b/stage1", ao / "components/stage1/pred_test.npz"))               # the ours row of T4a (How, stage 1)
    r.append(("T4d/pred/when", ao / "cond/pred/when/pred_test.npz"))
    r.append(("T4d/pred/spatial", ao / "cond/cascade/pred/pred_test.npz"))
    r.append(("T4d/true/spatial_online", ao / "cond/cascade/gt/pred_test.npz"))
    return r


def heads_acc(pred_path, ex, exclude=()):
    """Which/What from the separate tool and action heads (heads.py's <pred>.heads.npz), as TactusNet's 96.2 / 96.5."""
    import numpy as np
    h = Path(str(pred_path).replace(".npz", ".heads.npz"))
    if not h.is_file(): return None
    z = np.load(h); idx = {str(s): i for i, s in enumerate(z["sids"])}
    keep = [s for s in ex.sids("test") if ex.task(s) not in exclude]
    miss = [s for s in keep if s not in idx]
    if miss: raise SystemExit(f"{h}: {len(miss)} test clips missing")
    rows = [idx[s] for s in keep]
    which = float(np.mean(z["tool_logits"][rows].argmax(1) == np.array([ex.tool_id(s) for s in keep])))
    what = float(np.mean(z["action_logits"][rows].argmax(1) == np.array([ex.act_id(s) for s in keep])))
    return {"which_heads": which, "what_heads": what, "n": len(keep)}


def fmt(kind, s):
    if kind == "cls":
        return f"Which {100 * s['which']:.1f}  What {100 * s['what']:.1f}  Pair {100 * s['pair']:.1f}  mAcc {100 * s['pair_macc']:.1f}"
    if kind == "when":
        return f"tIoU {s['tiou']:.3f}  R@0.7 {100 * s['r7']:.1f}"
    t = []
    if "how" in s: t.append(f"ADE {s['how']['ade']:.2f}  >32px {100 * s['how']['gt32']:.1f}  median {s['how']['median_px']:.1f}")
    if "where" in s: t.append(f"Dice {s['where']['dice']:.3f}")
    return "  ".join(t)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-out", default=os.environ.get("BL_OUT", "/work/bl")); ap.add_argument("--ao")
    ap.add_argument("--out"); ap.add_argument("--boot", action="store_true", help="also 95%% bootstrap CIs (slower)")
    a = ap.parse_args(argv)
    bl = Path(a.bl_out); ao = Path(a.ao) if a.ao else bl / "ablate"; ex = get_export(); res = {}
    for key, p in rows(bl, ao):
        if not p.is_file():
            print(f"{key:34s} MISSING  {p}", flush=True); res[key] = None; continue
        pred = P.load(p); out = {"file": str(p), "kind": pred["kind"]}
        for sub, excl in (("all", ()), ("nine_videos", ENDOFM_PRETRAIN_TASKS)):
            out[sub] = score_file(pred, ex, "test", excl, boot=a.boot)["subsets"]["all"]
            if pred["kind"] == "cls":
                hd = heads_acc(p, ex, excl)
                if hd: out[sub].update(hd)
        res[key] = out
        extra = ""
        if pred["kind"] == "cls" and "which_heads" in out["all"]:
            extra = f"   [separate heads: Which {100 * out['all']['which_heads']:.1f} What {100 * out['all']['what_heads']:.1f}]"
        print(f"{key:34s} {fmt(pred['kind'], out['all'])}{extra}   | nine videos: {fmt(pred['kind'], out['nine_videos'])}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float)); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
