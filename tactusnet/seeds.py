"""Seed-frame selection and prompt assembly: the hand-over from the spatial layer to the tracking layer.

These are the rules of the batch annotation run that placed prompts with this model, pulled out as pure functions."""
from typing import Any, Dict, List, Tuple

import numpy as np

from .schema import N_SAMPLED, Seeds, SpatialOut


def inside_mask(sampled_local: np.ndarray, start_local: int, end_local: int) -> np.ndarray:
    idx = np.asarray(sampled_local)
    return (idx >= int(start_local)) & (idx <= int(end_local))


def choose_seeds(sampled_local: np.ndarray, tool_conf: np.ndarray, contact_conf: np.ndarray, start_local: int, end_local: int) -> Seeds:
    """Tool and contact points are seeded separately, each on the sampled frame with the highest mean confidence INSIDE the action range.

    The two confidences are not comparable (the contact head is trained on a different target and stays around 0.03-0.2), so they are
    never mixed.  If no sampled frame falls inside a very short range, both seeds use the sampled frame nearest to the range."""
    idx = np.asarray(sampled_local)
    tc, cc = np.asarray(tool_conf, float).mean(axis=1), np.asarray(contact_conf, float).mean(axis=1)
    inside = inside_mask(idx, start_local, end_local)
    if inside.any():
        ft = int(np.argmax(np.where(inside, tc, -np.inf)))
        fc = int(np.argmax(np.where(inside, cc, -np.inf)))
        return Seeds(tool_local=int(idx[ft]), contact_local=int(idx[fc]), fallback=False, inside_n=int(inside.sum()))
    dist = np.maximum(0, np.maximum(int(start_local) - idx, idx - int(end_local)))
    f = int(np.argmin(dist))
    return Seeds(tool_local=int(idx[f]), contact_local=int(idx[f]), fallback=True, inside_n=0)


def build_prompt(spatial: SpatialOut, seeds: Seeds, a0: int) -> List[Dict[str, Any]]:
    """7 prompt points in the annotation server's format: absolute frameIndex, x/y normalised to the frame and clamped to [0, 1]."""
    w, h = spatial.image_wh
    pos = {int(f): i for i, f in enumerate(spatial.sampled_local)}
    it, ic = pos[seeds.tool_local], pos[seeds.contact_local]

    def pt(frame_local: int, xy: np.ndarray, typ: str) -> Dict[str, Any]:
        return dict(frameIndex=int(a0 + frame_local), x=float(min(1.0, max(0.0, xy[0] / w))), y=float(min(1.0, max(0.0, xy[1] / h))), type=typ)

    out = [pt(seeds.tool_local, spatial.tool_xy[it, k], spatial.roles[k]) for k in range(3)]
    out += [pt(seeds.contact_local, spatial.contact_xy[ic, k], "contact") for k in range(4)]
    return out


def range_to_local_frames(start_t: float, end_t: float, n: int, t_steps: int = N_SAMPLED) -> Tuple[int, int]:
    """Map the head's 16-step time axis to frame indices inside the clip: frame = round(t / (T - 1) * (n - 1))."""
    last = max(t_steps - 1, 1)
    s = int(round(min(max(start_t, 0.0), last) / last * (n - 1)))
    e = int(round(min(max(end_t, 0.0), last) / last * (n - 1)))
    return s, max(s, e)
