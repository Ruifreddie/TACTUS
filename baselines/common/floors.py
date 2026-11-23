#!/usr/bin/env python3
"""Floors of Tab. 3: no learning beyond training-set statistics. Each writes a standard prediction file.

  majority            the most frequent training pair for every clip                       (cls)
  whole_window        the whole affordance window as the interval                          (when)
  pair_mean_interval  the mean training interval of the true pair                          (when)
  fixed_quad          the mean training contact quadrilateral of the true pair, placed at a predicted tip
                      (needs --tip-pred: a kpt prediction file with tool_xy, e.g. TactusNet's)   (kpt: ct_xy)

  python -m baselines.common.floors --which all --out OUT [--tip-pred OUT/tactusnet/pred_test.npz]
"""
import argparse
from pathlib import Path

import numpy as np

from . import config
from . import preds as P
from .data import get_export


def majority(ex, split):
    tr = [ex.pair_id(s) for s in ex.sids("train")]; maj = int(np.bincount(tr, minlength=len(ex.pairs)).argmax())
    sids = ex.sids(split); logits = np.zeros((len(sids), len(ex.pairs)), np.float32); logits[:, maj] = 1
    return sids, dict(pair_logits=logits), {"method": "majority", "pair": ex.pairs[maj]}


def whole_window(ex, split):
    sids = ex.sids(split)
    return sids, dict(start=np.zeros(len(sids)), end=np.ones(len(sids))), {"method": "whole_window"}


def pair_mean_interval(ex, split):
    acc = {p: [] for p in range(len(ex.pairs))}
    for s in ex.sids("train"):
        gs, ge, v = ex.interval(s)
        if v > 0.5: acc[ex.pair_id(s)].append((gs, ge))
    mean = {p: np.mean(v, 0) if v else np.array([0.0, 1.0]) for p, v in acc.items()}
    sids = ex.sids(split); se = np.stack([mean[ex.pair_id(s)] for s in sids])
    return sids, dict(start=se[:, 0], end=se[:, 1]), {"method": "pair_mean_interval", "conditioning": "true pair",
                                                     "mean": {ex.pairs[p]: [float(x) for x in m] for p, m in mean.items()}}


def quad_templates(ex):
    """Mean offsets (384-px) of the 4 contact points from the mean tip, per pair, over Gold-train frames where every
    contact point and every tip point is visible."""
    S = config.METRIC_SIZE; acc = {p: [] for p in range(len(ex.pairs))}
    for s in ex.sids("train"):
        txy, tvis = ex.tool_points(s); cxy, cvis = ex.contact_points(s); ti = ex.tip_index(s)
        ok = (tvis[:, ti] > 0.5).all(1) & (cvis > 0.5).all(1)
        if ok.any():
            tip = txy[ok][:, ti].mean(1, keepdims=True) * S; acc[ex.pair_id(s)].append((cxy[ok] * S - tip).mean(0))
    return {p: (np.mean(v, 0) if v else np.zeros((4, 2))) for p, v in acc.items()}


def fixed_quad(ex, split, tip_pred):
    S = config.METRIC_SIZE; tmpl = quad_templates(ex); tp = P.load(tip_pred); idx = {s: i for i, s in enumerate(tp["sids"])}
    sids = ex.sids(split); out = np.zeros((len(sids), 16, 4, 2), np.float32)
    for i, s in enumerate(sids):
        tip = tp["tool_xy"][idx[s]][:, ex.tip_index(s)].mean(1) * S                 # [16,2] predicted tip, 384-px
        out[i] = np.clip((tip[:, None, :] + tmpl[ex.pair_id(s)][None]) / S, 0, 1)
    return sids, dict(ct_xy=out), {"method": "fixed_quad", "conditioning": "true pair", "tip_source": str(tip_pred)}


FLOORS = {"majority": ("cls", majority), "whole_window": ("when", whole_window), "pair_mean_interval": ("when", pair_mean_interval),
          "fixed_quad": ("kpt", fixed_quad)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="all"); ap.add_argument("--out", default=str(config.OUT))
    ap.add_argument("--split", default="test"); ap.add_argument("--tip-pred"); ap.add_argument("--export"); ap.add_argument("--hf-raw")
    a = ap.parse_args(argv); ex = get_export(a.export, a.hf_raw)
    names = list(FLOORS) if a.which == "all" else a.which.split(",")
    for n in names:
        kind, fn = FLOORS[n]
        if n == "fixed_quad":
            if not a.tip_pred: print("skip fixed_quad: needs --tip-pred"); continue
            sids, arr, info = fn(ex, a.split, a.tip_pred)
        else:
            sids, arr, info = fn(ex, a.split)
        info["split"] = a.split; p = P.save(Path(a.out) / n / f"pred_{a.split}.npz", kind, sids, info, **arr); print("wrote", p)


if __name__ == "__main__":
    main()
