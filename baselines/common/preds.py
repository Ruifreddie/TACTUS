"""One prediction file per method and split: <OUT>/<method>/pred_<split>.npz. score.py reads only these files.

kind "cls":  pair_logits [N,6]      scores over the 6 pairs (ids of labels.json combo2id_6class)
kind "when": start [N], end [N]     interval on the normalized window axis; fallback [N] bool marks whole-window outputs
kind "kpt":  tool_xy [N,16,3,2]     tool points normalized to the content crop (How), optional
             ct_xy [N,16,4,2]       contact points normalized to the content crop (Where = their convex hull), optional
             ct_frame_dice [N,16]   for mask-based Where: per-frame Dice from metrics.contact_frame_dice(pred_masks=...)
Every file also stores sids [N] and a JSON "info" dict (method, split, input, conditioning, checkpoint, ...).
"""
import json
from pathlib import Path

import numpy as np

KINDS = {"cls": ("pair_logits",), "when": ("start", "end"), "kpt": ()}
OPTIONAL = {"cls": (), "when": ("fallback",), "kpt": ("tool_xy", "ct_xy", "ct_frame_dice")}


def save(path, kind, sids, info=None, **arrays):
    if kind not in KINDS: raise ValueError(kind)
    missing = [k for k in KINDS[kind] if k not in arrays]
    if missing: raise ValueError(f"{kind} predictions need {missing}")
    if kind == "kpt" and not any(k in arrays for k in OPTIONAL["kpt"]): raise ValueError("kpt predictions need tool_xy, ct_xy or ct_frame_dice")
    extra = set(arrays) - set(KINDS[kind]) - set(OPTIONAL[kind])
    if extra: raise ValueError(f"unknown arrays for {kind}: {sorted(extra)}")
    n = len(sids)
    for k, v in arrays.items():
        if len(v) != n: raise ValueError(f"{k} has {len(v)} rows, sids has {n}")
    shapes = {"pair_logits": (6,), "tool_xy": (16, 3, 2), "ct_xy": (16, 4, 2), "ct_frame_dice": (16,)}
    for k, s in shapes.items():
        if k in arrays and tuple(np.shape(arrays[k])[1:]) != s: raise ValueError(f"{k} must be [N,{','.join(map(str, s))}]")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, kind=kind, sids=np.asarray(sids, dtype="U16"), info=json.dumps(info or {}),
                        **{k: np.asarray(v, dtype=np.float32 if k != "fallback" else bool) for k, v in arrays.items()})
    tmp.replace(path)
    return path


def load(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["kind"] = str(out["kind"]); out["info"] = json.loads(str(out["info"])); out["sids"] = [str(s) for s in out["sids"]]
    return out
