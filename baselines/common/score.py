#!/usr/bin/env python3
"""Score baseline or TactusNet predictions on an exported partition.

  python -m baselines.common.score --pred OUT/videomae/pred_test.npz [--out metrics.json] [--exclude-tasks t1,t2]
  python -m baselines.common.score --pred A.npz --compare B.npz        # paired bootstrap of A - B over test videos

Reports point estimates for the selected split, non-SurgVU clips, each source and each pair.
The all and non-SurgVU summaries include 95% video-bootstrap intervals for ratio metrics.
Both input files must cover every clip in the selected export list.
"""
import argparse, json, sys

import numpy as np

from . import metrics as M
from . import preds as P
from .data import get_export, SOURCES


def subsets(ex, sids, exclude=()):
    keep = [s for s in sids if ex.task(s) not in exclude]
    out = {"all": keep, "non_surgvu": [s for s in keep if ex.source(s) != "SurgVU"]}
    for src in SOURCES: out[f"source/{src}"] = [s for s in keep if ex.source(s) == src]
    for p, name in enumerate(ex.pairs): out[f"pair/{name}"] = [s for s in keep if ex.pair_id(s) == p]
    return {k: v for k, v in out.items() if v}


def _summ(accs, boot=True, extra=None):
    r = {k: a.value() for k, a in accs.items()}; r["n"] = {k: a.n() for k, a in accs.items()}
    if boot: r["ci95"] = M.bootstrap(accs)
    if extra: r.update(extra)
    return r


def score_file(pred, ex, split="test", exclude=(), boot=True):
    sids_eval = ex.sids(split); idx = {s: i for i, s in enumerate(pred["sids"])}
    missing = [s for s in sids_eval if s not in idx]
    if missing: raise SystemExit(f"{len(missing)} evaluation clips have no prediction, e.g. {missing[:5]}")
    kind = pred["kind"]; out = {"kind": kind, "info": pred["info"], "split": split, "n_clips": len(sids_eval), "subsets": {}}
    for name, ss in subsets(ex, sids_eval, exclude).items():
        rows = [idx[s] for s in ss]; r = {}
        if kind == "cls":
            pp = pred["pair_logits"][rows].argmax(1); acc, per_pair = M.score_cls(ex, ss, pp)
            r = _summ(acc, boot and name in ("all", "non_surgvu"))
            vals = [a.value() for a in per_pair.values() if a.n() > 0]
            r["pair_macc"] = float(np.mean(vals)) if vals else float("nan")
            r["per_pair_acc"] = {ex.pairs[p]: a.value() for p, a in per_pair.items() if a.n() > 0}
        elif kind == "when":
            acc = M.score_when(ex, ss, pred["start"][rows], pred["end"][rows]); r = _summ(acc, boot and name in ("all", "non_surgvu"))
            if "fallback" in pred: r["fallback_rate"] = float(np.mean(pred["fallback"][rows]))
        elif kind == "kpt":
            if "tool_xy" in pred:
                acc, med = M.score_how(ex, ss, pred["tool_xy"][rows]); r["how"] = _summ(acc, boot and name in ("all", "non_surgvu"), {"median_px": med})
            if "ct_xy" in pred or "ct_frame_dice" in pred:
                acc = M.score_where(ex, ss, ct_xy=pred["ct_xy"][rows] if "ct_xy" in pred else None,
                                    frame_dice=pred["ct_frame_dice"][rows] if "ct_xy" not in pred else None)
                r["where"] = _summ(acc, boot and name in ("all", "non_surgvu"))
        out["subsets"][name] = r
    return out


def accs_for(pred, ex, split="test"):
    """The Acc objects of the main metrics on all test clips (for paired comparisons)."""
    sids = ex.sids(split); idx = {s: i for i, s in enumerate(pred["sids"])}; rows = [idx[s] for s in sids]
    if pred["kind"] == "cls": return M.score_cls(ex, sids, pred["pair_logits"][rows].argmax(1))[0]
    if pred["kind"] == "when": return M.score_when(ex, sids, pred["start"][rows], pred["end"][rows])
    out = {}
    if "tool_xy" in pred: out.update({f"how_{k}": v for k, v in M.score_how(ex, sids, pred["tool_xy"][rows])[0].items()})
    if "ct_xy" in pred or "ct_frame_dice" in pred:
        out.update({f"where_{k}": v for k, v in M.score_where(ex, sids, ct_xy=pred["ct_xy"][rows] if "ct_xy" in pred else None,
                                                             frame_dice=pred["ct_frame_dice"][rows] if "ct_xy" not in pred else None).items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--out"); ap.add_argument("--compare"); ap.add_argument("--exclude-tasks", default="")
    ap.add_argument("--no-boot", action="store_true"); ap.add_argument("--export"); ap.add_argument("--hf-raw")
    a = ap.parse_args(argv); ex = get_export(a.export, a.hf_raw); pred = P.load(a.pred)
    if a.compare:
        other = P.load(a.compare); A, B = accs_for(pred, ex, a.split), accs_for(other, ex, a.split)
        res = {k: M.paired_bootstrap(A[k], B[k]) for k in A if k in B}
    else:
        res = score_file(pred, ex, a.split, tuple(t for t in a.exclude_tasks.split(",") if t), boot=not a.no_boot)
    txt = json.dumps(res, indent=1)
    if a.out: open(a.out, "w").write(txt)
    print(txt if a.compare else json.dumps({k: v for k, v in res["subsets"]["all"].items() if k != "ci95"}, indent=1)[:3000])


if __name__ == "__main__":
    sys.exit(main())
