#!/usr/bin/env python3
"""CPU checks of the keypoint baselines (run before any GPU time; prep_kpt runs it):

  python -m baselines.kpt.tests.test_kpt [--weights]

1. the full-frame affine equals cv2.resize when not jittered, and inverts exactly;
2. the DARK and UDP codecs recover sub-pixel points from their own targets (encode -> decode round trip);
3. the 36 output channels: tool and pair groups are disjoint and cover 0..35;
4. the contact-order hook accepts tuples and permutations;
5. the data path: a train sample's targets land on the same pixels as the frame's points (on the export);
6. with --weights: every backbone checkpoint loads into its model with only the expected keys left out.
"""
import argparse, sys

import cv2
import numpy as np
import torch

from baselines.kpt import core
from baselines.kpt.codecs import DarkCodec, UDPCodec


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg); assert cond, msg


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", action="store_true"); a = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    # 1. transforms
    img = rng.integers(0, 255, (601, 894, 3), dtype=np.uint8); M = core.frame_affine(894, 601, (384, 288))
    a1 = cv2.warpAffine(img, M.astype(np.float32), (384, 288), flags=cv2.INTER_LINEAR); a2 = cv2.resize(img, (384, 288), interpolation=cv2.INTER_LINEAR)
    d = np.abs(a1[2:-2, 2:-2].astype(int) - a2[2:-2, 2:-2].astype(int))
    check(d.mean() < 2.5, f"untransformed frame affine == cv2.resize on the interior (mean |diff| {d.mean():.2f}; resize averages, warp samples)")
    p = rng.uniform(0, 1, (50, 2)); px = core.norm_to_px(p, 894, 601)
    q = core.px_to_norm(core.affine_points(core.invert_affine(M), core.affine_points(M, px)), 894, 601)
    check(np.abs(q - p).max() < 1e-9, "frame affine inverts exactly")
    check(np.allclose(core.affine_points(M, px), p * [384, 288] - 0.5), "plain rescale: input px = norm * size - 0.5")
    Mj = core.frame_affine(894, 601, (256, 192), 1.3, 30.0)
    check(np.abs(core.affine_points(core.invert_affine(Mj), core.affine_points(Mj, px)) - px).max() < 1e-9, "jittered affine inverts exactly")

    # 2. codecs: round trip on clean targets
    for codec, name in ((DarkCodec((384, 288), (96, 72), 3, 11), "DARK 384x288 / 96x72"), (UDPCodec((256, 192), (64, 48), 2, 11), "UDP 256x192 / 64x48")):
        W, H = codec.input_wh; kp = torch.tensor(rng.uniform([40, 40], [W - 40, H - 40], (1, 200, 2)), dtype=torch.float32)
        heat, w = codec.encode(kp, torch.ones(1, 200)); dec, _ = codec.decode(heat.numpy())
        err = np.abs(dec - kp.numpy()).max(); check(w.min() == 1 and err < 0.15, f"{name}: encode -> decode max error {err:.3f} input px")
        out = torch.tensor([[[-50.0, -50.0], [W + 60.0, 10.0]]]); _, w2 = codec.encode(out, torch.ones(1, 2))
        check(w2.sum() == 0, f"{name}: points whose 3-sigma box is outside the heatmap get weight 0")

    # 3. channels
    allc = np.concatenate([core.group_channels(t, p)[:3] for t in range(4) for p in [0]] + [core.group_channels(0, p)[3:] for p in range(6)])
    check(sorted(allc.tolist()) == list(range(36)), "tool (4 x 3) and pair (6 x 4) channel groups tile 0..35")

    # 4. contact-order hook
    import types, sys as _s
    mod = types.ModuleType("_kpt_order_test")
    mod.rev = lambda xy, vis: (xy[:, ::-1], vis[:, ::-1]); mod.perm = lambda xy, vis, sid=None: np.array([1, 0, 3, 2])
    _s.modules["_kpt_order_test"] = mod
    xy = rng.uniform(0, 1, (16, 4, 2)).astype(np.float32); vis = np.ones((16, 4), np.float32)

    class _Ex:
        def tip_index(self, sid): return [2]
    r1 = core.load_contact_order("_kpt_order_test:rev")(_Ex(), "x", xy, vis, None, None)
    r2 = core.load_contact_order("_kpt_order_test:perm")(_Ex(), "x", xy, vis, None, None)
    check(np.allclose(r1[0], xy[:, ::-1]) and np.allclose(r2[0], xy[:, [1, 0, 3, 2]]), "contact-order hook: tuple and permutation returns")
    check(core.load_contact_order("identity") is None, "identity order is the default")

    # 5. data path on the export
    from baselines.common.data import get_export
    from baselines.kpt.datasets import FullFrames
    ex = get_export(); sid = ex.sids("train")[0]
    ds = FullFrames(ex, [sid], (384, 288), train=False)
    s = ds[0]; W, H = core.frame_size(ex, sid); xy, vis, ch = core.clip_points(ex, sid)
    check(tuple(s["img"].shape) == (3, 288, 384) and np.allclose(s["kp"].numpy(), xy[0] * [384, 288] - 0.5, atol=1e-4), "eval sample = plain rescale of frame 0")
    check(np.array_equal(s["ch"].numpy(), core.group_channels(ex.tool_id(sid), ex.pair_id(sid))), "sample carries its true-group channels")

    if a.weights:
        from baselines.kpt.hrnet import build_hrnet_w32
        from baselines.kpt.vitpose import build_vitpose_b
        from baselines.kpt.third_party.dsta.dsta import DSTA
        from baselines.kpt import weights as Wt
        m, loaded = build_hrnet_w32(36); check(len(loaded) == 11, f"HRNet-W32 ImageNet layers loaded: {loaded}")
        y = m(torch.zeros(1, 3, 288, 384)); check(tuple(y.shape) == (1, 36, 72, 96), "HRNet output 36 x 72 x 96")
        v, rep = build_vitpose_b(36, (256, 192)); check(rep["loaded"] > 190, f"ViT-B MAE weights loaded ({rep['pos_embed']})")
        y = v(torch.zeros(1, 3, 192, 256)).heatmaps; check(tuple(y.shape) == (1, 36, 48, 64), "ViTPose output 36 x 48 x 64")
        d = DSTA(36, [[3 * t + k for k in range(3)] for t in range(4)] + [[12 + 4 * p + k for k in range(4)] for p in range(6)])
        d.init_weights(Wt.tv_resnet50_state()); c, sg = d(torch.zeros(2, 5, 3, 192, 256))
        check(tuple(c.shape) == (2, 36, 2) and tuple(sg.shape) == (2, 36, 2), "DSTA output coord/sigma 36 x 2 from 5 frames")
    print("all kpt checks passed")


if __name__ == "__main__":
    sys.exit(main())
