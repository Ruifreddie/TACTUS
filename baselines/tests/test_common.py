#!/usr/bin/env python3
"""Checks of baselines/common on a real export. Run it on your export before any training:

  BL_EXPORT=<export> BL_HF_RAW=<hf> python -m baselines.tests.test_common [--exp <UniTAP exp dir, e.g. exp/A>]

1. the export reads and its labels are consistent;
2. raw_frames() decodes the same pixels as the exported protocol frames;
3. score.py reproduces UniTAP's own numbers for TactusNet: How (pipeline dump, pooled ADE / median / share > 32 px)
   and Where (contact dump + zoom boxes, region_hull_dice mean), when --exp is given;
4. floors, prediction files, subsets and the paired bootstrap run end to end.
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np

from baselines.common import floors, metrics as M, preds as P, score, tactus
from baselines.common.data import get_export


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg); assert cond, msg


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--exp"); a = ap.parse_args(argv)
    ex = get_export(); n = {s: len(ex.sids(s)) for s in ("train", "val", "test")}; print("clips", n)
    check(all(v > 0 for v in n.values()), "train/val/test lists are not empty")
    for s in ex.sids("test")[:50]:
        m = ex.meta[s]; assert ex.pairs[ex.pair_id(s)] == m["combo"] and ex.tool_id(s) == m["tool_id"] and ex.act_id(s) == m["act_id"], s
    check(True, "pair, tool and action ids agree with meta.json")

    for s in ex.sids("test")[:3]:
        d = np.abs(ex.raw_frames(s, ex.sampled_abs(s)).astype(int) - ex.frames(s).astype(int)).max()
        check(d == 0, f"raw_frames == exported frames for {s} (max diff {d})")
    s0 = ex.sids("test")[0]; nb = ex.raw_frames(s0, ex.sampled_abs(s0)[[0, 0, 1]] + np.array([-2, 0, 1]))
    check(nb.shape[0] == 3, "raw_frames clamps and returns neighbors")

    with tempfile.TemporaryDirectory() as tmp:
        floors.main(["--which", "majority,whole_window,pair_mean_interval", "--out", tmp])
        for f in ("majority", "whole_window", "pair_mean_interval"):
            r = score.score_file(P.load(f"{tmp}/{f}/pred_test.npz"), ex, boot=True)["subsets"]["all"]
            print("    ", f, {k: v for k, v in r.items() if k in ("pair", "which", "what", "tiou", "r7", "pair_macc")})
        r = score.score_file(P.load(f"{tmp}/whole_window/pred_test.npz"), ex)["subsets"]["all"]
        check(0 < r["tiou"] <= 1 and r["r7"] <= r["r5"] <= r["r3"], "whole-window tIoU and recalls are consistent")
        cmp = score.main(["--pred", f"{tmp}/pair_mean_interval/pred_test.npz", "--compare", f"{tmp}/whole_window/pred_test.npz"])
        check(True, "paired bootstrap runs")

        if a.exp:
            E = Path(a.exp); out = f"{tmp}/tactusnet/pred_test.npz"
            tactus.main(["--exp", str(E), "--out", out]); pred = P.load(out)
            res = score.score_file(pred, ex)["subsets"]["all"]
            # How: UniTAP's pipeline summary pools all visible points ('refined')
            summ = _pipeline_summary(E)
            errs = np.concatenate([M.tool_errors(pred["tool_xy"][i], *ex.tool_points(s)) for i, s in enumerate(pred["sids"])])
            check(abs(errs.mean() - summ["ADE"]) < 1e-3, f"pooled How ADE {errs.mean():.4f} == UniTAP {summ['ADE']:.4f}")
            check(abs(np.median(errs) - summ["med"]) < 1e-3 and abs((errs > 32).mean() - summ["frac_gt32"]) < 1e-9,
                  "median and share > 32 px match UniTAP")
            check(abs(res["how"]["gt32"] - summ["frac_gt32"]) < 1e-9, "score.py share > 32 px matches UniTAP")
            print(f"     How ADE (clip, then clips) = {res['how']['ade']:.3f}  pooled = {errs.mean():.3f}")
            ref = json.load(open(E / "results/contact_human_test.json"))["summary"]["region_hull_dice"]["mean"]
            check(abs(res["where"]["dice"] - ref) < 1e-9, f"Where Dice {res['where']['dice']:.6f} == UniTAP region_hull_dice {ref:.6f}")
            floors.main(["--which", "fixed_quad", "--out", tmp, "--tip-pred", out])
            r = score.score_file(P.load(f"{tmp}/fixed_quad/pred_test.npz"), ex, boot=False)["subsets"]["all"]["where"]
            print(f"     fixed quad at TactusNet's tip: Dice {r['dice']:.4f}")
    print("all checks passed")


def _pipeline_summary(E):
    p = E / "results/pipeline_test/summary_tool.json"
    if p.exists(): return json.load(open(p))["refined"]
    txt = open(E / "results/pipeline_test.log").read(); return json.loads(txt[txt.index("{\n \"stage1\""):])["refined"]


if __name__ == "__main__":
    sys.exit(main())
