"""Data of the When baselines: pooled frozen V-JEPA 2.1 features, the true pair, the interval, and the time axes.

Window axis (the metric's, `ex.interval`): 0 = first protocol frame, 1 = last. The features come from F frames spread
uniformly over the window, end frames included (F = 16 protocol frames for 8 steps, F = 64 frames for 32 steps), encoded
with tubelet 2, so step k covers frames 2k and 2k+1 and is centred at x_k = (2k + 0.5) / (F - 1).

Coordinates used by the methods (all affine in x):
  grid  g  ActionFormer / TriDet feature grid, step k at g = k        x = (2g + 0.5) / (F - 1)
  unit  u  QD-DETR's normalized clip axis, step k spans [k/T, (k+1)/T]  x = (F u - 0.5) / (F - 1)
Every prediction is mapped back to x and clipped to [0, 1] (metrics.tiou clips as well).

Features on disk (written by features.py): <root>/s8/<sid>.npy and <root>/s32/<sid>.npy, fp16 [T, 1024]; the root is
--feat-root, else $BL_WHEN_FEATS, else $BL_OUT/when/feats.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from baselines.common import config

STEPS = (8, 32)
DIM = 1024


def n_frames(steps: int) -> int:
    """Frames behind a T-step sequence (tubelet 2)."""
    if steps not in STEPS: raise ValueError(f"steps must be one of {STEPS}, got {steps}")
    return 2 * steps


def token_centers(steps: int) -> np.ndarray:
    F = n_frames(steps); return (2 * np.arange(steps) + 0.5) / (F - 1)


def win_to_grid(x, steps):
    F = n_frames(steps); return ((F - 1) * np.asarray(x, np.float64) - 0.5) / 2


def grid_to_win(g, steps):
    F = n_frames(steps); return (2 * np.asarray(g, np.float64) + 0.5) / (F - 1)


def win_to_unit(x, steps):
    F = n_frames(steps); return ((F - 1) * np.asarray(x, np.float64) + 0.5) / F


def unit_to_win(u, steps):
    F = n_frames(steps); return (F * np.asarray(u, np.float64) - 0.5) / (F - 1)


def feat_root(path=None) -> Path:
    return Path(path or os.environ.get("BL_WHEN_FEATS") or (config.OUT / "when" / "feats"))


def feat_dir(root, steps) -> Path:
    n_frames(steps); return Path(root) / f"s{steps}"


def standin_from_8(a8, steps):
    """SMOKE ONLY: an 8-step sequence linearly interpolated to the step centres of `steps` (clamped at the ends). Used by
    --smoke when real 32-step features exist for only a few clips locally; never for a real run."""
    x8, xs = token_centers(8), token_centers(steps); a8 = a8.astype(np.float32)
    return np.stack([np.interp(xs, x8, a8[:, j]) for j in range(a8.shape[1])], 1).astype(np.float16)


def load_feats(root, steps, sids, workers=16, standin=False):
    """fp16 [N, T, 1024] for the given sids; raises with the command to run if any file is missing.
    standin=True (smoke runs only) replaces missing 32-step files by standin_from_8; returns (array, n_standins)."""
    d = feat_dir(root, steps); missing = [s for s in sids if not (d / f"{s}.npy").is_file()]
    use_standin = bool(missing) and standin and steps != 8
    if missing and not use_standin:
        cmd = "pool" if steps == 8 else "extract32"
        raise SystemExit(f"{len(missing)} of {len(sids)} {steps}-step features missing in {d} (e.g. {missing[:3]}); "
                         f"run: python -m baselines.when.features {cmd} --feat-root {Path(root)}")
    miss = set(missing)

    def one(s):
        if s in miss: return standin_from_8(np.load(feat_dir(root, 8) / f"{s}.npy"), steps)
        a = np.load(d / f"{s}.npy")
        if a.shape != (steps, DIM): raise ValueError(f"{d / s}.npy has shape {a.shape}, expected {(steps, DIM)}")
        return a.astype(np.float16, copy=False)

    with ThreadPoolExecutor(workers) as pool:
        arr = np.stack(list(pool.map(one, sids))) if sids else np.zeros((0, steps, DIM), np.float16)
    return arr, len(miss)


class Split:
    """One split in memory: features, true pairs and intervals on the window axis (valid = annotated interval)."""

    def __init__(self, ex, root, steps, split, limit=None, standin=False):
        self.name, self.steps = split, steps
        self.sids = ex.sids(split)[:limit] if limit else ex.sids(split)
        self.x, self.n_standin = load_feats(root, steps, self.sids, standin=standin)  # fp16 [N, T, D]
        self.pair = np.array([ex.pair_id(s) for s in self.sids], np.int64)
        iv = np.array([ex.interval(s) for s in self.sids], np.float64).reshape(-1, 3)
        self.seg = iv[:, :2].astype(np.float32); self.valid = iv[:, 2] > 0.5

    def __len__(self): return len(self.sids)
