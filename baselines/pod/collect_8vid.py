#!/usr/bin/env python3
"""Score the rows of Table 3 on the eight test videos outside SurgVU, next to all test clips.

No development step of TactusNet's Spatial Head and no tool-point head of the Silver generator saw these eight videos
(Supp. E.4), so they check the Table 3 ranking on an unaffected subset. Every file is scored with
baselines.common.score.score_file (the code of every row of the paper). In addition, TactusNet is compared with every
other learned row of its block by a paired bootstrap over the eight videos (metrics.paired_bootstrap).

  python -m baselines.pod.collect_8vid --bl-out $GV/results/baselines/runs --out $GV/results/baselines/tables_8vid.json
"""
import argparse, json, os
from pathlib import Path

from baselines.common import metrics as M
from baselines.common import preds as P
from baselines.common.data import get_export
from baselines.common.score import score_file
from baselines.pod.collect_v4 import fmt, rows

REF = {"T3a": "T3a/tactusnet", "T3b": "T3b/tactusnet", "T3c": "T3c/tactusnet"}
KEYS = {"T3a": ("which", "what", "pair"), "T3b": ("tiou", "r7"), "T3c": ("how_ade", "how_gt32", "where_dice")}


def accs_on(pred, ex, sids):
    """The Acc objects of the main metrics on the given clips (score.accs_for, restricted to sids)."""
    idx = {str(s): i for i, s in enumerate(pred["sids"])}
    miss = [s for s in sids if s not in idx]
    if miss: raise SystemExit(f"{len(miss)} clips missing from the prediction file")
    r = [idx[s] for s in sids]; k = pred["kind"]
    if k == "cls": return M.score_cls(ex, sids, pred["pair_logits"][r].argmax(1))[0]
    if k == "when": return M.score_when(ex, sids, pred["start"][r], pred["end"][r])
    out = {}
    if "tool_xy" in pred: out.update({f"how_{n}": v for n, v in M.score_how(ex, sids, pred["tool_xy"][r])[0].items()})
    if "ct_xy" in pred or "ct_frame_dice" in pred:
        w = M.score_where(ex, sids, ct_xy=pred["ct_xy"][r] if "ct_xy" in pred else None,
                          frame_dice=pred["ct_frame_dice"][r] if "ct_xy" not in pred else None)
        out.update({f"where_{n}": v for n, v in w.items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl-out", default=os.environ.get("BL_OUT", "/work/bl")); ap.add_argument("--ao")
    ap.add_argument("--out", required=True); ap.add_argument("--no-boot", action="store_true")
    a = ap.parse_args(argv)
    bl = Path(a.bl_out); ao = Path(a.ao) if a.ao else bl / "ablate"; ex = get_export()
    test = [str(s) for s in ex.sids("test")]; eight = [s for s in test if ex.source(s) != "SurgVU"]
    print(f"test clips {len(test)}, outside SurgVU {len(eight)} from {len({ex.task(s) for s in eight})} videos", flush=True)
    res = {"n_test": len(test), "n_eight": len(eight), "videos_eight": sorted({ex.task(s) for s in eight}), "rows": {}, "paired": {}}
    accs = {}
    for key, p in rows(bl, ao):
        if not key.startswith("T3"): continue
        if not p.is_file():
            print(f"{key:20s} MISSING  {p}", flush=True); res["rows"][key] = None; continue
        pred = P.load(p); sc = score_file(pred, ex, "test", (), boot=not a.no_boot)["subsets"]
        res["rows"][key] = {"file": str(p), "kind": pred["kind"], "all": sc["all"], "non_surgvu": sc["non_surgvu"]}
        for sub in ("all", "non_surgvu"): print(f"{key:20s} {sub:10s} {fmt(pred['kind'], sc[sub])}", flush=True)
        accs[key] = accs_on(pred, ex, eight)
    for fam, ref in REF.items():
        if ref not in accs: continue
        for key in accs:
            if not key.startswith(fam + "/") or key == ref: continue
            pb = {k: M.paired_bootstrap(accs[ref][k], accs[key][k]) for k in KEYS[fam] if k in accs[ref] and k in accs[key]}
            res["paired"][f"{ref} - {key}"] = pb
            print(f"paired {ref} - {key}: " + "  ".join(f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in pb.items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
