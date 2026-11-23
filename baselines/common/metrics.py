"""The paper's metrics, implemented exactly as TactusNet computes them (UniTAP train.py / infer_pipeline.py /
eval_contact_stage.py), plus subsets and a bootstrap over test videos.

Every metric is a ratio of sums, so it can be accumulated per test video and resampled by video:
  Which / What / Pair accuracy   = correct clips / clips
  When tIoU, R@theta             = sum of tIoU (or hits) / clips with a valid interval
  How ADE                        = sum of per-clip mean errors / clips with >= 1 visible tool point  (clip, then clips)
  How share > 32 px, median      = pooled over all visible tool points
  Where Dice                     = sum of per-frame Dice / frames with >= 2 visible GT contact points (pooled frames)
"""
from collections import defaultdict

import numpy as np

from . import config
from .hull import hull_mask

RANGE_THRESHOLDS = (0.3, 0.5, 0.7)
PCK_THRESHOLDS = (4, 8, 16)


# ---------------------------------------------------------------- per-clip primitives
def tiou(ps, pe, gs, ge):
    """Continuous tIoU on the normalized window axis (UniTAP _compute_range_metrics_per_sample)."""
    ps = float(np.clip(ps, 0, 1)); pe = max(float(np.clip(pe, 0, 1)), ps)
    gs = float(np.clip(gs, 0, 1)); ge = max(float(np.clip(ge, 0, 1)), gs)
    inter = max(0.0, min(pe, ge) - max(ps, gs)); union = max(1e-6, max(pe, ge) - min(ps, gs))
    return inter / union


def tool_errors(pred_xy, gt_xy, gt_vis):
    """Pixel errors (384-px metric) of the visible tool points; pred/gt [16,3,2] normalized to the content crop."""
    e = np.linalg.norm((np.asarray(pred_xy, np.float64) - np.asarray(gt_xy, np.float64)) * config.METRIC_SIZE, axis=-1)
    return e[np.asarray(gt_vis) > 0.5]


def contact_frame_dice(gt_ct_xy, gt_ct_vis, pred_ct_xy=None, pred_masks=None):
    """Per-frame Dice [16] (NaN where the frame is not scored, i.e. < 2 visible GT contact points).
    The prediction is either 4 points per frame (their convex hull, like TactusNet) or a boolean 384x384 mask per frame."""
    S = config.METRIC_SIZE; out = np.full(len(gt_ct_xy), np.nan)
    for t in range(len(gt_ct_xy)):
        vk = np.where(np.asarray(gt_ct_vis[t]) > 0.5)[0]
        if len(vk) < 2: continue
        mg = hull_mask(np.asarray(gt_ct_xy[t], np.float64)[vk] * S)
        if pred_masks is not None:
            mp = np.asarray(pred_masks[t], bool)
            if mp.shape != mg.shape: raise ValueError(f"pred mask must be {mg.shape}, got {mp.shape}")
        else:
            mp = hull_mask(np.asarray(pred_ct_xy[t], np.float64) * S)
        out[t] = float(2 * (mp & mg).sum() / max(mp.sum() + mg.sum(), 1))
    return out


# ---------------------------------------------------------------- accumulation by video
class Acc:
    """numerator / denominator per unit (test video), for ratio metrics and the bootstrap."""
    def __init__(self):
        self.num = defaultdict(float); self.den = defaultdict(float)

    def add(self, unit, num, den=1.0):
        self.num[unit] += float(num); self.den[unit] += float(den)

    def value(self, units=None):
        units = list(self.num) if units is None else units
        d = sum(self.den[u] for u in units)
        return float(sum(self.num[u] for u in units) / d) if d > 0 else float("nan")

    def n(self): return float(sum(self.den.values()))


def bootstrap(accs, n_boot=2000, seed=0, units=None):
    """95% CI of each metric (and of a - b for paired comparisons, see paired_bootstrap), resampling test videos."""
    rng = np.random.default_rng(seed)
    units = sorted(set().union(*[set(a.num) | set(a.den) for a in accs.values()])) if units is None else units
    vals = {k: [] for k in accs}
    for _ in range(n_boot):
        samp = [units[i] for i in rng.integers(0, len(units), len(units))]
        for k, a in accs.items(): vals[k].append(_ratio(a, samp))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def _ratio(a, units):
    d = sum(a.den[u] for u in units)
    return sum(a.num[u] for u in units) / d if d > 0 else np.nan


def paired_bootstrap(acc_a, acc_b, n_boot=2000, seed=0):
    """95% CI of value(a) - value(b), resampling the shared test videos."""
    rng = np.random.default_rng(seed); units = sorted(set(acc_a.den) & set(acc_b.den)); diffs = []
    for _ in range(n_boot):
        samp = [units[i] for i in rng.integers(0, len(units), len(units))]
        diffs.append(_ratio(acc_a, samp) - _ratio(acc_b, samp))
    return {"diff": acc_a.value(units) - acc_b.value(units), "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
            "n_units": len(units)}


# ---------------------------------------------------------------- family scorers (return Acc objects per metric)
def score_cls(ex, sids, pred_pair):
    acc = {k: Acc() for k in ("which", "what", "pair")}
    per_pair = {p: Acc() for p in range(len(ex.pairs))}
    for sid, pp in zip(sids, pred_pair):
        u = ex.task(sid); g = ex.pair_id(sid); pp = int(pp)
        acc["which"].add(u, ex.pair_tool[pp] == ex.pair_tool[g]); acc["what"].add(u, ex.pair_act[pp] == ex.pair_act[g])
        acc["pair"].add(u, pp == g); per_pair[g].add(u, pp == g)
    return acc, per_pair


def score_when(ex, sids, start, end):
    acc = {"tiou": Acc(), **{f"r{int(t * 10)}": Acc() for t in RANGE_THRESHOLDS}}
    for sid, s, e in zip(sids, start, end):
        gs, ge, valid = ex.interval(sid)
        if valid <= 0.5: continue
        u = ex.task(sid); v = tiou(s, e, gs, ge); acc["tiou"].add(u, v)
        for t in RANGE_THRESHOLDS: acc[f"r{int(t * 10)}"].add(u, v >= t)
    return acc


def score_how(ex, sids, tool_xy):
    acc = {"ade": Acc(), "gt32": Acc(), **{f"pck{t}": Acc() for t in PCK_THRESHOLDS}}; pooled = []
    for sid, p in zip(sids, tool_xy):
        gt, vis = ex.tool_points(sid); e = tool_errors(p, gt, vis)
        if e.size == 0: continue
        u = ex.task(sid); acc["ade"].add(u, e.mean()); acc["gt32"].add(u, (e > 32).sum(), e.size)
        for t in PCK_THRESHOLDS: acc[f"pck{t}"].add(u, (e < t).sum(), e.size)
        pooled.append(e)
    med = float(np.median(np.concatenate(pooled))) if pooled else float("nan")
    return acc, med


def score_where(ex, sids, ct_xy=None, frame_dice=None):
    """Either ct_xy [N,16,4,2] (hull of the points) or frame_dice [N,16] computed with contact_frame_dice(pred_masks=...)."""
    acc = {"dice": Acc(), "dice_ge50": Acc()}
    for i, sid in enumerate(sids):
        gt, vis = ex.contact_points(sid)
        d = contact_frame_dice(gt, vis, pred_ct_xy=ct_xy[i]) if frame_dice is None else np.asarray(frame_dice[i], np.float64)
        ok = ~np.isnan(d)
        if not ok.any(): continue
        u = ex.task(sid); acc["dice"].add(u, d[ok].sum(), ok.sum()); acc["dice_ge50"].add(u, (d[ok] >= 0.5).sum(), ok.sum())
    return acc
