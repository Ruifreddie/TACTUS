"""Shared pieces of the keypoint baselines: output channels, the contact-order hook, full-frame transforms, frame lists.

Output channels (all three models): 36 = 12 tool + 24 contact.
  tool    channel 3 * tool_id + k,       k = 0..2  (hook: anchor, anchor, tip; jawed tools: anchor, tip, tip)
  contact channel 12 + 4 * pair_id + k,  k = 0..3  (the 4 contact tracks, in the order set by the contact-order hook)
The loss covers only the true tool's 3 channels and the true pair's 4 channels (x visibility), and the test reads the same
7 channels (`group_channels`).

Input: the full content frame (ex.frames, or ex.raw_frames for DSTA's neighbours) resized to the model's input, a plain
rescale (no box, no padding, no aspect fix: TactusNet also squashes the content crop). Training adds each official
recipe's random scale and rotation around the frame centre; no flip. Points are normalized to the content crop
(baselines/common/data.py); the transforms work in OpenCV pixel-centre coordinates, p = xy * (W, H) - 0.5, so the
untransformed map equals cv2.resize and predictions map back by a plain rescale.
"""
import importlib, inspect, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

N_TOOLS, N_PAIRS, N_TOOL_PTS, N_CT_PTS = 4, 6, 3, 4
N_OUT = N_TOOLS * N_TOOL_PTS + N_PAIRS * N_CT_PTS          # 36
N_SEL = N_TOOL_PTS + N_CT_PTS                              # 7 channels read per sample


def group_channels(tool_id, pair_id):
    """The 7 output channels of a clip: its tool's 3 points, then its pair's 4 contact points."""
    return np.array([3 * tool_id + k for k in range(N_TOOL_PTS)] + [12 + 4 * pair_id + k for k in range(N_CT_PTS)], np.int64)


# ---------------------------------------------------------------- frames
_CLIPS = {}                      # per process (each DataLoader worker has its own): the last few clips of ex.frames


def frame(ex, sid, t, keep=8):
    """Protocol frame t, RGB uint8 [H,W,3] at native resolution, read only through ex.frames(sid) (the export's
    video_16_npy, or the raw clip decoded when it is absent). Items of one clip are adjacent (FullFrames), so a small
    per-process cache loads each clip about once per pass."""
    key = (str(ex.root), sid)
    if key in _CLIPS: _CLIPS[key] = _CLIPS.pop(key)
    else:
        _CLIPS[key] = ex.frames(sid)
        while len(_CLIPS) > keep: _CLIPS.pop(next(iter(_CLIPS)))
    return _CLIPS[key][t]


def frame_size(ex, sid):
    m = ex.meta[sid]; return float(m["W"]), float(m["H"])


def frames_1fps(ex, sids, cache=None, workers=16):
    """{sid: ex.frames_1fps(sid)}, the about-1-fps training frames of the image models. ex.fps opens every mp4 once, so
    the lists are computed in parallel and cached as JSON (prep_kpt writes the shared cache)."""
    cache = Path(cache) if cache else None; have = {}; key = str(Path(ex.root).resolve())
    if cache and cache.exists():
        c = json.loads(cache.read_text()); have = c.get("frames", {}) if c.get("export") == key else {}
    todo = [s for s in sids if s not in have]
    if todo:
        with ThreadPoolExecutor(workers) as pool:
            for s, f in zip(todo, pool.map(ex.frames_1fps, todo)): have[s] = [int(t) for t in f]
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True); tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps({"export": key, "frames": have})); tmp.replace(cache)
    return {s: have[s] for s in sids}


# ---------------------------------------------------------------- points and the contact-order hook
def load_contact_order(spec):
    """Contact-target order hook. 'identity' (default) keeps the export's uid order. 'module:function' (for example
    'baselines.where.canonical_order:canonical_order') is called as fn(ct_xy [16,4,2], ct_vis [16,4], **kw) with the
    keyword arguments it accepts among ex, sid, tool_xy, tool_vis, tip_index. It returns (ct_xy, ct_vis), or an index
    array [4] (one order per clip) or [16,4] (one per frame) that permutes the 4 slots."""
    if spec in (None, "", "identity"): return None
    mod, _, name = spec.partition(":")
    fn = getattr(importlib.import_module(mod), name or "canonical_order")
    params = inspect.signature(fn).parameters
    takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())

    def apply(ex, sid, cxy, cvis, txy, tvis):
        kw = dict(ex=ex, sid=sid, tool_xy=txy, tool_vis=tvis, tip_index=ex.tip_index(sid))
        r = fn(cxy.copy(), cvis.copy(), **{k: v for k, v in kw.items() if takes_all or k in params})
        if isinstance(r, tuple) and len(r) == 2:
            return np.asarray(r[0], np.float32).reshape(cxy.shape), np.asarray(r[1], np.float32).reshape(cvis.shape)
        perm = np.asarray(r, np.int64)
        if perm.shape == (N_CT_PTS,): return cxy[:, perm], cvis[:, perm]
        if perm.shape == cvis.shape: return np.take_along_axis(cxy, perm[..., None], 1), np.take_along_axis(cvis, perm, 1)
        raise ValueError(f"contact order {spec} returned shape {perm.shape}")
    apply.spec = spec
    return apply


def clip_points(ex, sid, contact_order=None):
    """Targets of one clip: xy [16,7,2] normalized (3 tool + 4 contact points), vis [16,7], channels [7]."""
    txy, tvis = ex.tool_points(sid); cxy, cvis = ex.contact_points(sid)
    if contact_order is not None: cxy, cvis = contact_order(ex, sid, cxy, cvis, txy, tvis)
    return (np.concatenate([txy, cxy], 1).astype(np.float32), np.concatenate([tvis, cvis], 1).astype(np.float32),
            group_channels(ex.tool_id(sid), ex.pair_id(sid)))


# ---------------------------------------------------------------- full-frame transforms
def random_scale_rotation(rng, cfg):
    """Official TopDownGetRandomScaleRotation (HRNet / DarkPose / mmpose 0.x / DCPose-DSTA): scale
    s = clip(N(1, sf), 1 - sf, 1 + sf) of the box (here the whole frame), rotation clip(N(0, rf), -2 rf, 2 rf) degrees
    with probability rot_prob."""
    sf, rf = cfg["scale_factor"], cfg["rot_factor"]
    s = float(np.clip(rng.standard_normal() * sf + 1, 1 - sf, 1 + sf))
    r = float(np.clip(rng.standard_normal() * rf, -2 * rf, 2 * rf)) if rng.random() <= cfg.get("rot_prob", 0.6) else 0.0
    return s, r


def frame_affine(W, H, out_wh, scale=1.0, rot=0.0):
    """2x3 affine from frame pixels (W x H) to the input (out_wh), OpenCV pixel-centre convention. scale=1, rot=0 is the
    plain rescale of cv2.resize; scale > 1 zooms out (a larger 'box', as in the official pipelines); the rotation is
    about the frame centre, applied after the rescale."""
    ow, oh = out_wh; th = np.deg2rad(rot); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = R @ np.diag([ow / W, oh / H]) / scale
    uc, uo = np.array([W / 2, H / 2]), np.array([ow / 2, oh / 2])
    return np.concatenate([A, (uo - 0.5 - A @ (uc - 0.5))[:, None]], 1)


def affine_points(M, p):
    """Apply a 2x3 affine to points [..., 2]."""
    p = np.asarray(p, np.float64); return p @ M[:, :2].T + M[:, 2]


def invert_affine(M):
    A = M[:, :2]; Ai = np.linalg.inv(A); return np.concatenate([Ai, (-Ai @ M[:, 2])[:, None]], 1)


def norm_to_px(xy, W, H):
    return np.asarray(xy, np.float64) * np.array([W, H]) - 0.5


def px_to_norm(p, W, H):
    return (np.asarray(p, np.float64) + 0.5) / np.array([W, H])


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
