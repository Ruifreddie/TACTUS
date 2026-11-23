"""Canonical order of the 4 contact slots, for training targets of methods that use the slots as fixed channels.

The export orders a clip's 4 contact tracks by (first visible protocol frame, angle of the track's mean position around
the mean of the 4 track means) (UniTAP tools/export_case32.py, uid_contact_tracks). The first key wins whenever the
tracks do not all start in the same frame, so slot k then has no fixed spatial meaning. check_order.py measures this.

canonical() keeps the tracks whole (one permutation per clip, so video methods still see one identity per slot) and
uses only the ground truth, so it is meant for training targets. Dice is order-free: evaluation does not change.

Rule (DEFAULT_RULE = "image_left"):
  1. each slot's mean position over the frames where it is visible (all 16 frames if it is never visible);
  2. the centre is the mean of these slot means (slots visible at least once only);
  3. slots are sorted by the angle of their mean around the centre, measured from the image's left direction and
     increasing clockwise on screen (y points down), i.e. ascending atan2(dy, dx) starting at -pi;
  4. slots that are never visible go last, in export order.
So slot 0 is usually the up-left corner of the region, then up-right, down-right, down-left. The angular order is the
same in normalized and in pixel coordinates (an axis scaling keeps the order of directions and maps the left
direction to itself), so the rule runs directly on the normalized export arrays.

Other rules, kept for check_order.py: "image_up" (cut at the up direction) and "tool" (cut at the backward tool
direction, tip -> anchor, so slot 0 is the first corner clockwise after the tool shaft).

  from baselines.where.canonical_order import canonical, apply
  perm = canonical(ct_xy, ct_vis, tool_xy, tool_vis)      # [4] int, new slot j = old slot perm[j]
  ct_xy, ct_vis = apply(perm, ct_xy, ct_vis)
"""
import numpy as np

DEFAULT_RULE = "image_left"
RULES = ("export", "image_left", "image_up", "tool")
_CUT = {"image_left": np.pi, "image_up": -np.pi / 2}


def _slot_means(ct_xy, ct_vis):
    xy = np.asarray(ct_xy, np.float64); vis = np.asarray(ct_vis) > 0.5
    seen = vis.any(0)
    means = np.stack([xy[vis[:, k], k].mean(0) if seen[k] else xy[:, k].mean(0) for k in range(xy.shape[1])])
    return means, seen


def tool_direction(tool_xy, tool_vis, tip_index=(2,), anchor_index=(0,)):
    """Mean unit direction anchor -> tip over the frames where both are visible (angle in radians), or None.
    Point 0 is always an anchor and point 2 always a tip (hook: anchor, anchor, tip; jawed: anchor, tip, tip)."""
    if tool_xy is None or tool_vis is None: return None
    xy = np.asarray(tool_xy, np.float64); vis = np.asarray(tool_vis) > 0.5
    ti, ai = list(tip_index), list(anchor_index); acc = np.zeros(2)
    for t in range(len(xy)):
        tv = [k for k in ti if vis[t, k]]; av = [k for k in ai if vis[t, k]]
        if not tv or not av: continue
        d = xy[t, tv].mean(0) - xy[t, av].mean(0); n = np.linalg.norm(d)
        if n > 1e-9: acc += d / n
    return float(np.arctan2(acc[1], acc[0])) if np.linalg.norm(acc) > 1e-9 else None


def order_by_angle(points, valid, cut):
    """Indices of the valid points sorted clockwise on screen starting just after the direction `cut` (radians,
    image axes, y down), then the invalid ones in their original order."""
    pts = np.asarray(points, np.float64); valid = np.asarray(valid, bool)
    if valid.sum() == 0: return np.arange(len(pts))
    c = pts[valid].mean(0); ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    key = np.mod(ang - cut, 2 * np.pi)
    vi = [int(i) for i in np.where(valid)[0]]; vi.sort(key=lambda i: (key[i], i))
    return np.array(vi + [int(i) for i in np.where(~valid)[0]])


def cut_of(rule, tool_xy=None, tool_vis=None, tip_index=(2,)):
    if rule in _CUT: return _CUT[rule]
    if rule == "tool":
        phi = tool_direction(tool_xy, tool_vis, tip_index)
        return _CUT["image_left"] if phi is None else phi + np.pi
    raise ValueError(rule)


def canonical(ct_xy, ct_vis, tool_xy=None, tool_vis=None, rule=DEFAULT_RULE, tip_index=(2,)):
    """ct_xy [16,4,2], ct_vis [16,4] (normalized export arrays), tool_xy [16,3,2], tool_vis [16,3] -> permutation [4]
    such that the canonical slot j is the export slot perm[j]. A fixed rule on the ground truth only."""
    n = np.asarray(ct_xy).shape[1]
    if rule == "export": return np.arange(n)
    means, seen = _slot_means(ct_xy, ct_vis)
    return order_by_angle(means, seen, cut_of(rule, tool_xy, tool_vis, tip_index))


def apply(perm, ct_xy, ct_vis):
    """Re-order the slots of [T,4,2] / [T,4] arrays (or [N,T,4,...] with one perm per clip applied by the caller)."""
    perm = np.asarray(perm)
    return np.asarray(ct_xy)[:, perm], np.asarray(ct_vis)[:, perm]


def canonical_export(ex, sid, rule=DEFAULT_RULE):
    """Convenience: the canonical permutation of a clip of baselines.common.data.Export (uses the clip's tip index)."""
    cxy, cvis = ex.contact_points(sid); txy, tvis = ex.tool_points(sid)
    return canonical(cxy, cvis, txy, tvis, rule=rule, tip_index=tuple(ex.tip_index(sid)))
