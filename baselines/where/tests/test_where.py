#!/usr/bin/env python3
"""Checks of the Where family on a real export (run them before training):

  python -m baselines.where.tests.test_where [--exp <UniTAP exp dir with data/case32/v1_zoom_hs>] [--preds <BL_OUT>]

1. the mask path of the metric: GT hull masks given as pred_masks score Dice 1 on every scored frame, and score.py reads
   ct_frame_dice files like ct_xy files;
2. canonical_order: a permutation of the 4 slots, idempotent, and the Where Dice of contact points is unchanged by it;
3. hindsight_floor: with --exp, its hotspots reproduce UniTAP's own box-grid labels exactly;
4. with --preds, every Where prediction file present there is complete and scores.
"""
import argparse, sys, tempfile
from pathlib import Path

import numpy as np

from baselines.common import config, metrics as M, preds as P, score
from baselines.common.data import get_export
from baselines.common.hull import hull_mask
from baselines.where import canonical_order as CO
from baselines.where import hindsight_floor as HF


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg); assert cond, msg


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--exp"); ap.add_argument("--preds"); a = ap.parse_args(argv)
    ex = get_export(); sids = ex.sids("test"); S = config.METRIC_SIZE

    fd = []
    for s in sids:
        gt, vis = ex.contact_points(s)
        masks = np.stack([hull_mask(gt[t][vis[t] > 0.5] * S) if (vis[t] > 0.5).sum() >= 1 else np.zeros((S, S), bool) for t in range(16)])
        fd.append(M.contact_frame_dice(gt, vis, pred_masks=masks))
    fd = np.stack(fd); ok = ~np.isnan(fd)
    check(np.allclose(fd[ok], 1.0), f"GT hull masks as predictions give Dice 1 on all {ok.sum()} scored test frames")
    with tempfile.TemporaryDirectory() as tmp:
        P.save(f"{tmp}/gt.npz", "kpt", sids, {"method": "gt"}, ct_frame_dice=fd)
        P.save(f"{tmp}/gtxy.npz", "kpt", sids, {"method": "gt"}, ct_xy=np.stack([ex.contact_points(s)[0] for s in sids]))
        r1 = score.score_file(P.load(f"{tmp}/gt.npz"), ex, boot=False)["subsets"]["all"]["where"]
        r2 = score.score_file(P.load(f"{tmp}/gtxy.npz"), ex, boot=False)["subsets"]["all"]["where"]
        check(abs(r1["dice"] - 1) < 1e-9 and r1["n"] == r2["n"], "score.py reads ct_frame_dice files (Dice 1, same frames as ct_xy)")

    n_changed = 0
    for s in ex.sids("train") + ex.sids("val") + sids:
        cxy, cvis = ex.contact_points(s); txy, tvis = ex.tool_points(s)
        perm = CO.canonical_export(ex, s); assert sorted(perm.tolist()) == [0, 1, 2, 3], perm
        n_changed += int(list(perm) != [0, 1, 2, 3])
        cxy2, cvis2 = CO.apply(perm, cxy, cvis)
        assert list(CO.canonical(cxy2, cvis2, txy, tvis, tip_index=tuple(ex.tip_index(s)))) == [0, 1, 2, 3], s
        d1, d2 = M.contact_frame_dice(cxy, cvis, pred_ct_xy=cxy[:, ::-1]), M.contact_frame_dice(cxy, cvis, pred_ct_xy=cxy2)
        assert np.allclose(np.nan_to_num(d1), np.nan_to_num(d2)), s
    check(True, f"canonical order is a permutation, idempotent, Dice-invariant ({n_changed} Gold clips re-ordered)")

    if a.exp:
        r = HF.check_unitap(ex, a.exp)
        check(r["a_box_grid_max_abs_diff"] < 1e-6 and r["a_frames_hotspot_flag_differs"] == 0,
              f"hindsight hotspots reproduce UniTAP's region_mask on {r['clips']} clips (max |diff| {r['a_box_grid_max_abs_diff']:.2e})")
        check(r["c_corner_err_px384_max"] < 1e-3, f"pseudo-corners match UniTAP's (max {r['c_corner_err_px384_max']:.1e} px)")
        print(f"     384-grid vs box-grid hotspot: Dice {r['b_dice_inside_box_mean']:.3f} inside the box, "
              f"{100 * r['b_hotspot_area_outside_box_mean']:.2f}% of the area outside it")

    if a.preds:
        for m in ("segformer", "tactus_human", "hindsight_floor", "hindsight_floor_corners"):
            for split in ("val", "test"):
                p = Path(a.preds) / m / f"pred_{split}.npz"
                if not p.exists(): print(f"skip {p}"); continue
                r = score.score_file(P.load(p), ex, split=split, boot=False)["subsets"]["all"]["where"]
                check(0 <= r["dice"] <= 1, f"{m} {split}: Where Dice {r['dice']:.4f} over {int(r['n']['dice'])} frames")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
