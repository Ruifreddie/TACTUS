#!/usr/bin/env python3
"""CPU checks of the classification harness (no GPU, no training). The GPU smoke runs are smoke_cls in ../run.sh.

  python -m baselines.cls.tests.test_cls

1. the pack.py cache holds exactly data.squash(ex.frames(sid)) and matches this export;
2. every official train augmentation returns a finite normalized [16,3,224,224] clip and is reproducible per sample;
3. the test view is the whole frame (no crop) and the epoch sampler is deterministic per (seed, epoch);
4. Mixup/CutMix targets are distributions; every LR schedule hits its official landmarks;
5. one X-CLIP prompt per pair.
"""
import math, sys

import numpy as np
import torch

from baselines.common.data import get_export
from baselines.cls import augment as A, data as D, models as M
from baselines.cls.train import group_lr


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg); assert cond, msg


def main():
    ex = get_export()
    for size, split in ((224, "test"), (256, "train"), (320, "train")):
        d = D.cache_dir(size)
        if not D.cache_valid(d, ex): print(f"skip cache check s{size}: no valid cache in {d}"); continue
        for s in ex.sids(split)[:2]:
            if not D.cache_ok(d / f"{s}.npy", size): continue
            a = np.load(d / f"{s}.npy"); b = D.squash(ex.frames(s), size)
            check(a.shape == (16, size, size, 3) and np.array_equal(a, b), f"cache s{size}/{s} == squash(frames)")

    sid = ex.sids("train")[0]
    for name, r in M.RECIPES.items():
        ds = D.ClipSet(ex, [sid], r["pre"], train_tf=A.TRAIN[name], norm=r["norm"], num_sample=r["num_sample"])
        x1, y, _ = ds[(0, 123)]; x2, _, _ = ds[(0, 123)]; x3, _, _ = ds[(0, 124)]
        shape = (r["num_sample"], 16, 3, 224, 224) if r["num_sample"] > 1 else (16, 3, 224, 224)
        check(tuple(x1.shape) == shape and torch.isfinite(x1.float()).all(), f"{name}: train view {tuple(x1.shape)} at {r['pre']}px")
        check(torch.equal(x1, x2) and not torch.equal(x1, x3), f"{name}: same seed -> same view, other seed -> other view")
    ev = D.ClipSet(ex, [sid], 224, norm=A.IMAGENET)[0][0].float()
    ref = A.normalize(A.to_tensor(D.squash(ex.frames(sid), 224)), A.IMAGENET)
    check(torch.allclose(ev, ref, atol=2e-3), "test view = whole frame squashed to 224, normalized")
    s1 = D.EpochSampler(10, 25, 3); s1.set_epoch(2); s2 = D.EpochSampler(10, 25, 3); s2.set_epoch(2)
    check(list(s1) == list(s2) and len(list(s1)) == 25, "epoch sampler deterministic, fixed length")

    rng = np.random.default_rng(0); y = torch.tensor([0, 1, 2, 3])
    for mix in (A.TimmMixup(6), A.XClipBlend(6)):
        for _ in range(4):
            x, t = mix(torch.randn(4, 16, 3, 32, 32), y, rng)
            check(t.shape == (4, 6) and torch.allclose(t.sum(1), torch.ones(4)) and x.shape == (4, 16, 3, 32, 32),
                  f"{type(mix).__name__}: soft targets sum to 1")

    spe = 100
    def lr(name, t, scale=1.0):
        r = M.RECIPES[name]; peak = r["lr"] * 64 / r["lr_batch"]
        return group_lr(r, peak, scale, t, spe, r["epochs"], peak), peak
    v, p = lr("tsm", 9 * spe); check(math.isclose(v, p), "tsm: lr 1e-3 before epoch 10")
    v, p = lr("tsm", 10 * spe); check(math.isclose(v, p * 0.1), "tsm: x0.1 at epoch 10")
    v, p = lr("tsm", 20 * spe, 5.0); check(math.isclose(v, p * 0.01 * 5), "tsm: x0.01 at epoch 20, fc x5")
    v, p = lr("tsn", 40 * spe); check(math.isclose(v, p * 0.01), "tsn: x0.01 at epoch 40")
    v, p = lr("videomae", 0); check(math.isclose(v, 2.5e-7), "videomae: warmup starts at 1e-6 x 64/256")
    v, p = lr("videomae", 5 * spe); check(math.isclose(v, p), "videomae: peak 1.25e-4 after 5 warmup epochs")
    v, p = lr("videomae", 30 * spe - 1); check(abs(v - 2.5e-7) < 1e-8, "videomae: ends at min_lr 1e-6 x 64/256")
    v, p = lr("xclip", 0, 10.0); check(v == 0.0, "xclip: warmup from 0")
    v, p = lr("xclip", 30 * spe, 10.0); check(math.isclose(v, p / 100), "xclip: 10x group ends at the absolute LR/100")
    v, p = lr("mvitv2", 0); check(math.isclose(v, 4e-4), "mvitv2: warmup starts at 1e-4 x 64/16")
    v, p = lr("mvitv2", 3 * spe); check(math.isclose(v, p), "mvitv2: peak 0.01 after 3 warmup epochs")
    v, p = lr("mvitv2", 100 * spe); check(math.isclose(v, 4e-6), "mvitv2: ends at 1e-6 x 64/16")
    v, p = lr("endofm", 10 * spe + 50); check(math.isclose(v, p * 0.5), "endofm: per-epoch cosine, half at epoch 10")
    check(sorted(M.PROMPTS) == sorted(ex.pairs), "one X-CLIP prompt per pair")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
