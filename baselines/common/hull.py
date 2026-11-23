"""Contact-region rasterization, copied verbatim from UniTAP tools/eval_contact_stage.py (convex_hull, hull_mask).

TactusNet's Where Dice (paper Tab. 3) is computed with exactly these two functions, so every baseline must use them too.
Do not edit them; change the caller instead.
"""
import numpy as np


def convex_hull(pts):
    """Andrew monotone chain; pts [N,2] -> hull vertices in order (N>=1)."""
    P = sorted(set(map(tuple, np.asarray(pts, dtype=np.float64).round(4))))
    if len(P) <= 2: return np.array(P)
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for q in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0: lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0: upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def hull_mask(pts, size=384, min_radius=6.0):
    """Rasterized convex hull of the points (384-px coords). Degenerate hulls (<3 distinct pts / tiny area) are
    dilated to a disc/segment of radius min_radius so that a 2-point GT still defines a region."""
    from PIL import Image, ImageDraw
    im = Image.new('L', (size, size), 0); d = ImageDraw.Draw(im); h = convex_hull(pts)
    if len(h) >= 3:
        d.polygon([tuple(v) for v in h], fill=255)
    for v in h:  # dilate corners a little so thin regions are not empty
        d.ellipse([v[0] - min_radius, v[1] - min_radius, v[0] + min_radius, v[1] + min_radius], fill=255)
    if len(h) == 2: d.line([tuple(h[0]), tuple(h[1])], fill=255, width=int(2 * min_radius))
    return np.asarray(im) > 0
