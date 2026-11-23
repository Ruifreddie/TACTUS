"""Clips for the classification baselines: the 16 protocol frames of a clip (ex.frames: RGB, native content crop), the
whole crop squashed to a square (TACTUS adaptation (b)), read from the pre-resized cache when present (pack.py), else
resized on the fly with the same function.

  train: frames squashed to the method's pre-crop size P (256 or 320), then the method's official augmentation
         (augment.TRAIN[model]); `num_sample` views per clip (VideoMAE's repeated augmentation);
  val / test: frames squashed to 224, normalized: the whole frame is the single test view.

Each sample's augmentation is seeded from (seed, epoch, position in the epoch), so it does not depend on the number of
workers or on a resume. Samples are returned as float16, normalized.
"""
import hashlib, json, os, random
from pathlib import Path

import numpy as np
import torch

from baselines.common import config
from . import augment as A

CACHE = Path(os.environ.get("BL_CLS_CACHE") or Path(os.environ.get("BL_OUT", str(config.OUT))) / "cls_cache")  # local disk
RESIZE = "whole content crop squashed to size x size (aspect not kept, no center crop), PIL bilinear with antialias"
EVAL_SIZE = A.CROP


def squash(frames, size):
    """uint8 [T,H,W,3] -> uint8 [T,size,size,3]: the whole content crop resized to a square (PIL bilinear, antialiased
    when shrinking, like torchvision's Resize on PIL images). pack.py and the on-the-fly path both call this."""
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f)).resize((size, size), Image.Resampling.BILINEAR))
                     for f in frames])


# ---------------------------------------------------------------- cache
def export_key(ex):
    """Identifies the export a cache was built from (sha1 of its meta.json)."""
    return hashlib.sha1((Path(ex.root) / "meta.json").read_bytes()).hexdigest()


def cache_dir(size, root=None):
    return Path(root or CACHE) / f"s{size}"


def cache_bytes(size, n_frames=config.N_FRAMES):
    """File size of a complete cache entry: .npy v1.0 header (as np.save writes it) + uint8 payload."""
    import io
    shape = (n_frames, size, size, 3); b = io.BytesIO()
    np.lib.format.write_array_header_1_0(b, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
                                             "fortran_order": False, "shape": shape})
    return b.tell() + int(np.prod(shape))


def cache_ok(path, size):
    try: return path.stat().st_size == cache_bytes(size)
    except FileNotFoundError: return False


def cache_valid(d, ex):
    m = Path(d) / "MANIFEST.json"
    if not m.is_file(): return False
    try: return json.loads(m.read_text()).get("export_key") == export_key(ex)
    except (ValueError, OSError): return False


# ---------------------------------------------------------------- dataset
def sample_seed(seed, epoch, k):
    return (seed * 1_000_003 + epoch * 100_003 + k) % (2**31 - 1)


class ClipSet(torch.utils.data.Dataset):
    """Items are row indices (eval) or (row, sample seed) pairs from EpochSampler (train). Returns (float16 clip
    [16,3,224,224] or [num_sample,16,3,224,224], pair id, row)."""

    def __init__(self, ex, sids, size, train_tf=None, norm=A.IMAGENET, num_sample=1, cache_root=None, use_cache=True):
        self.ex, self.sids, self.size, self.tf, self.norm, self.k = ex, list(sids), int(size), train_tf, norm, int(num_sample)
        self.labels = np.array([ex.pair_id(s) for s in self.sids], dtype=np.int64)
        d = cache_dir(size, cache_root)
        self.cache = d if use_cache and cache_valid(d, ex) else None
        self.n_cached = sum(cache_ok(d / f"{s}.npy", size) for s in self.sids) if self.cache is not None else 0

    def __len__(self): return len(self.sids)

    def load(self, sid):
        if self.cache is not None:
            p = self.cache / f"{sid}.npy"
            if cache_ok(p, self.size): return np.load(p)
        return squash(self.ex.frames(sid), self.size)

    def __getitem__(self, item):
        i, s = item if isinstance(item, tuple) else (item, None)
        f = self.load(self.sids[i])
        if self.tf is None:
            x = A.test_view(f, self.norm)
        else:
            random.seed(s); np.random.seed(s)          # the streams the official-style transforms draw from
            views = [self.tf(f) for _ in range(self.k)]
            x = torch.stack(views) if self.k > 1 else views[0]
        return x.half(), int(self.labels[i]), i


class EpochSampler(torch.utils.data.Sampler):
    """A fixed number of (row, sample seed) items per epoch, reshuffled with seed + epoch (a resumed run sees the same
    order and the same augmentation)."""

    def __init__(self, n, n_per_epoch, seed):
        self.n, self.k, self.seed, self.epoch = n, n_per_epoch, seed, 0

    def set_epoch(self, e): self.epoch = e
    def __len__(self): return self.k

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 100003 + self.epoch); out = []
        while len(out) < self.k: out.extend(rng.permutation(self.n).tolist())
        return iter([(i, sample_seed(self.seed, self.epoch, k)) for k, i in enumerate(out[:self.k])])
