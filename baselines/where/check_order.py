#!/usr/bin/env python3
"""Do the 4 contact slots of the export have a consistent spatial meaning across clips and frames?

Keypoint baselines (baselines/kpt) use the 4 contact slots as fixed heatmap channels, so slot k has to mean the same
corner of the region in every clip and, for per-frame models, in every frame. The export orders the tracks by
(first visible protocol frame, angle of the track's mean around the mean of the 4 track means); canonical_order.py
holds the candidate re-orderings. For the export order and for each candidate rule this script measures, on Gold clips:

  tie        share of clips whose visible slots all become visible in the same frame (only then the angle decides)
  sector     per slot, the share of clips whose slot mean lies in each 90-degree sector around the clip centre
             (image axes: up-left, up-right, down-right, down-left), and the same relative to the tool direction
  R          per slot, the resultant length of the clip-level angle (1 = always the same direction, 0 = uniform)
  frame_id   share of frames with >= 3 visible slots whose angular order around the frame centre (same rule, same
             cut) equals the slot order, i.e. slot identity can be read from the single frame
  frame_cyc  ... equals the slot order up to a cyclic shift (same winding, the cut is the only disagreement)
  near_cut   share of clips with a slot mean within 15 degrees of the rule's cut direction (unstable start)
Positions are native content-crop pixels (the angular order is the same in normalized coordinates).

  python -m baselines.where.check_order [--splits train,val,test] [--out OUT/where/check_order.json]
"""
import argparse, json
from pathlib import Path

import numpy as np

from ..common import config
from ..common.data import get_export
from . import canonical_order as CO

SECTORS = ("up-left", "up-right", "down-right", "down-left")   # ascending atan2 from -pi with y pointing down
TOOL_SECTORS = ("back-left", "front-left", "front-right", "back-right")  # same, relative to the tool direction


def _sector(a):
    return int(np.clip((np.mod(a + np.pi, 2 * np.pi)) // (np.pi / 2), 0, 3))


def _wrap(a):
    return float(np.mod(a + np.pi, 2 * np.pi) - np.pi)


def export_order_reproduced(cxy, cvis):
    """The export's own key, recomputed: (first visible frame, angle of the all-frame track mean around their centre)."""
    vis = cvis > 0.5; T = len(cxy)
    fv = [int(np.where(vis[:, k])[0][0]) if vis[:, k].any() else T for k in range(cxy.shape[1])]
    means = cxy.mean(0); c = means.mean(0); ang = np.arctan2(means[:, 1] - c[1], means[:, 0] - c[0])
    return sorted(range(cxy.shape[1]), key=lambda k: (fv[k], ang[k]))


def clip_stats(ex, sid, rule):
    m = ex.meta[sid]; WH = np.array([float(m["W"]), float(m["H"])])
    cxy, cvis = ex.contact_points(sid); txy, tvis = ex.tool_points(sid); ti = tuple(ex.tip_index(sid))
    perm = CO.canonical(cxy, cvis, txy, tvis, rule=rule, tip_index=ti) if rule != "export" else np.arange(4)
    cxy, cvis = cxy[:, perm].astype(np.float64) * WH, cvis[:, perm] > 0.5; txy = txy.astype(np.float64) * WH
    seen = cvis.any(0); fv = [int(np.where(cvis[:, k])[0][0]) for k in range(4) if seen[k]]
    rec = {"sid": sid, "n_seen": int(seen.sum()), "tie": len(set(fv)) <= 1}
    means = np.stack([cxy[cvis[:, k], k].mean(0) if seen[k] else cxy[:, k].mean(0) for k in range(4)])
    c = means[seen].mean(0) if seen.any() else means.mean(0)
    ang = np.arctan2(means[:, 1] - c[1], means[:, 0] - c[0])
    phi = CO.tool_direction(txy, tvis, ti)
    rec["ang"] = [float(a) if seen[k] else None for k, a in enumerate(ang)]
    rec["ang_tool"] = [(_wrap(a - phi) if seen[k] else None) for k, a in enumerate(ang)] if phi is not None else None
    cut = CO.cut_of(rule if rule != "export" else "image_left", txy, tvis, ti)
    d_cut = [abs(_wrap(a - cut)) for k, a in enumerate(ang) if seen[k]]
    rec["near_cut"] = bool(d_cut and min(d_cut) < np.deg2rad(15))
    # per-frame: does the frame's own angular order (same rule) give the slot order?
    fid = fcyc = nfr = 0
    for t in range(len(cxy)):
        vk = np.where(cvis[t])[0]
        if len(vk) < 3: continue
        if rule == "tool":
            pt = CO.tool_direction(txy[t:t + 1], tvis[t:t + 1], ti); fcut = (pt + np.pi) if pt is not None else cut
        else:
            fcut = cut
        o = vk[CO.order_by_angle(cxy[t, vk], np.ones(len(vk), bool), fcut)]
        nfr += 1; fid += int(list(o) == list(vk))
        fcyc += int(any(list(np.roll(o, s)) == list(vk) for s in range(len(vk))))
    rec.update(frames=nfr, frame_id=fid, frame_cyc=fcyc)
    if rule == "export":
        rec["export_key_reproduced"] = export_order_reproduced(ex.contact_points(sid)[0].astype(np.float64) * WH,
                                                               ex.contact_points(sid)[1]) == [0, 1, 2, 3]
    return rec


def summarize(recs):
    out = {"clips": len(recs), "clips_4_slots_seen": int(sum(r["n_seen"] == 4 for r in recs)),
           "tie": float(np.mean([r["tie"] for r in recs])), "near_cut": float(np.mean([r["near_cut"] for r in recs])),
           "frames": int(sum(r["frames"] for r in recs))}
    nf = max(out["frames"], 1)
    out["frame_id"] = sum(r["frame_id"] for r in recs) / nf; out["frame_cyc"] = sum(r["frame_cyc"] for r in recs) / nf
    if "export_key_reproduced" in recs[0]: out["export_key_reproduced"] = float(np.mean([r["export_key_reproduced"] for r in recs]))
    for key, names in (("ang", SECTORS), ("ang_tool", TOOL_SECTORS)):
        per = {}
        for k in range(4):
            a = np.array([r[key][k] for r in recs if r[key] is not None and r[key][k] is not None])
            if a.size == 0: continue
            h = np.bincount([_sector(x) for x in a], minlength=4) / a.size; z = np.exp(1j * a).mean()
            per[f"slot{k}"] = {"n": int(a.size), "R": float(abs(z)), "mean_deg": float(np.degrees(np.angle(z))),
                               "sectors": {n: round(float(v), 3) for n, v in zip(names, h)},
                               "top_sector": names[int(h.argmax())], "top_share": float(h.max())}
        out["image_axes" if key == "ang" else "tool_relative"] = per
    return out


def verdict(s):
    """The export order is consistent when the angle decides it (tie), single frames agree with it, and each slot
    sits in its own dominant sector."""
    ia = s["image_axes"]; tops = [ia[k]["top_sector"] for k in ia]
    ok = s["tie"] >= 0.95 and s["frame_id"] >= 0.9 and min(ia[k]["top_share"] for k in ia) >= 0.6 and len(set(tops)) == 4
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--rules", default=",".join(CO.RULES)); ap.add_argument("--with-silver", action="store_true")
    ap.add_argument("--out", default=str(config.OUT / "where" / "check_order.json")); ap.add_argument("--export"); ap.add_argument("--hf-raw")
    a = ap.parse_args(argv); ex = get_export(a.export, a.hf_raw)
    groups = {"gold": [s for sp in a.splits.split(",") for s in ex.sids(sp)]}
    if a.with_silver: groups["silver"] = ex.sids("silver")
    res = {}
    for g, sids in groups.items():
        for rule in a.rules.split(","):
            recs = [clip_stats(ex, s, rule) for s in sids]; res[f"{g}/{rule}"] = summarize(recs)
            if rule == "export": res[f"{g}/{rule}"]["consistent"] = bool(verdict(res[f"{g}/{rule}"]))
    for k, s in res.items():
        ia = s["image_axes"]; tie = f"tie {s['tie']:.3f}" if k.endswith("/export") else " " * 9
        print(f"{k:22s} clips {s['clips']:5d}  {tie}  frame_id {s['frame_id']:.3f}  frame_cyc {s['frame_cyc']:.3f}  "
              f"near_cut {s['near_cut']:.3f}  " + "  ".join(f"s{i}:{ia[f'slot{i}']['top_sector']}/{ia[f'slot{i}']['top_share']:.2f}/R{ia[f'slot{i}']['R']:.2f}"
                                                         for i in range(4) if f"slot{i}" in ia))
        tr = s.get("tool_relative") or {}
        if tr: print(" " * 23 + "tool-relative: " + "  ".join(f"s{i}:{tr[f'slot{i}']['top_sector']}/{tr[f'slot{i}']['top_share']:.2f}/R{tr[f'slot{i}']['R']:.2f}"
                                                            for i in range(4) if f"slot{i}" in tr))
    v = res["gold/export"]["consistent"] if "gold/export" in res else None
    print("verdict: export slot order is", "CONSISTENT" if v else "NOT consistent", "(rule: tie >= 0.95, frame_id >= 0.9, each slot >= 60% in its own sector)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump({"verdict_export_consistent": v, "results": res}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
