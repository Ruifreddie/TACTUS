"""Full-frame samples for the keypoint baselines (no boxes: the whole content frame is the input).

FullFrames (HRNet, ViTPose): one sample = protocol frame t of a clip. Train items are the about-1-fps frames of each
  Gold-train clip (ex.frames_1fps); eval items are all 16 protocol frames.
FullFrameClips (DSTA): one item = K key frames of one clip; key frame t brings the 5 consecutive raw frames
  ex.raw_frames(sid, ex.sampled_abs(sid)[t] + [-2, -1, 0, 1, 2]), all warped with the key frame's transform. The raw
  frames of the K key frames are decoded in one pass over the clip. Train: each epoch draws `kf_per_clip` of the 16 key
  frames per clip (seeded by epoch); eval: all 16.

Every sample returns the input (uint8, CHW), the 7 target points in input pixels, their visibility, the 7 output
channels, and the 2x3 frame -> input affine, so predictions map back to content-crop-normalized coordinates.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import core

NEIGHBORS = np.arange(-2, 3)    # DSTA: key frame +- 2 consecutive raw frames


class _Base(Dataset):
    def __init__(self, ex, sids, input_wh, train, jitter=None, contact_order=None, seed=0):
        self.ex, self.sids, self.input_wh, self.train = ex, list(sids), tuple(input_wh), train
        self.jitter = jitter if train else None; self.seed, self.epoch = seed, 0
        self.pts = {s: core.clip_points(ex, s, contact_order) for s in self.sids}    # xy [16,7,2], vis [16,7], ch [7]
        self.size = {s: core.frame_size(ex, s) for s in self.sids}
        self.row = {s: i for i, s in enumerate(self.sids)}

    def set_epoch(self, epoch): self.epoch = int(epoch)

    def _affine(self, sid, rng):
        W, H = self.size[sid]; s, r = core.random_scale_rotation(rng, self.jitter) if self.jitter else (1.0, 0.0)
        return core.frame_affine(W, H, self.input_wh, s, r)

    def _sample(self, sid, t, imgs, M):
        """imgs: RGB frames (1 for image models, 5 for DSTA). Returns inputs [F,3,H,W] uint8 and the targets."""
        W, H = self.size[sid]; Mf = M.astype(np.float32)
        x = np.stack([cv2.warpAffine(im, Mf, self.input_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                      for im in imgs])
        xy, vis, ch = self.pts[sid]
        kp = core.affine_points(M, core.norm_to_px(xy[t], W, H)).astype(np.float32)
        return x.transpose(0, 3, 1, 2), kp, vis[t].copy(), ch, Mf


class FullFrames(_Base):
    """Items (sid, t): train frames from `train_frames` {sid: [t, ...]}, else all 16 frames. Frames come from ex.frames.
    Train order: clips shuffled each epoch (seeded), a clip's frames adjacent, so each clip is read about once per epoch
    even when ex.frames has to decode the raw clip (the loader does not reshuffle: `grouped`)."""
    grouped = True

    def __init__(self, ex, sids, input_wh, train, jitter=None, contact_order=None, seed=0, train_frames=None):
        super().__init__(ex, sids, input_wh, train, jitter, contact_order, seed)
        self.frames_of = {s: [int(t) for t in (train_frames[s] if train else range(16))] for s in self.sids}
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch); order = self.sids
        if self.train: order = [self.sids[i] for i in np.random.default_rng((self.seed, 104729, self.epoch)).permutation(len(self.sids))]
        self.items = [(s, t) for s in order for t in self.frames_of[s]]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        sid, t = self.items[i]; rng = np.random.default_rng((self.seed, self.epoch, i, np.random.randint(1 << 30)))
        x, kp, vis, ch, M = self._sample(sid, t, [core.frame(self.ex, sid, t)], self._affine(sid, rng))
        return {"img": torch.from_numpy(x[0]), "kp": torch.from_numpy(kp), "vis": torch.from_numpy(vis), "ch": torch.from_numpy(ch),
                "M": torch.from_numpy(M), "row": self.row[sid], "t": t}


class FullFrameClips(_Base):
    def __init__(self, ex, sids, input_wh, train, jitter=None, contact_order=None, seed=0, kf_per_clip=8, kf_per_item=4):
        super().__init__(ex, sids, input_wh, train, jitter, contact_order, seed)
        self.kf_per_clip, self.kf_per_item = kf_per_clip, kf_per_item
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if not self.train: self.items = [(s, list(range(16))) for s in self.sids]; return
        rng = np.random.default_rng((self.seed, 7919, self.epoch)); items = []
        for s in self.sids:
            ks = rng.permutation(16)[:self.kf_per_clip]
            items += [(s, sorted(ks[i:i + self.kf_per_item].tolist())) for i in range(0, len(ks), self.kf_per_item)]
        self.items = [items[i] for i in rng.permutation(len(items))]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        sid, ts = self.items[i]; sab = self.ex.sampled_abs(sid)
        raw = self.ex.raw_frames(sid, (sab[ts][:, None] + NEIGHBORS[None]).reshape(-1))
        raw = raw.reshape(len(ts), len(NEIGHBORS), *raw.shape[1:])
        rng = np.random.default_rng((self.seed, self.epoch, i, np.random.randint(1 << 30)))
        out = [self._sample(sid, t, list(raw[j]), self._affine(sid, rng)) for j, t in enumerate(ts)]
        st = lambda k: torch.from_numpy(np.stack([o[k] for o in out]))
        return {"img": st(0), "kp": st(1), "vis": st(2), "ch": st(3), "M": st(4),
                "row": torch.full((len(ts),), self.row[sid], dtype=torch.long), "t": torch.tensor(ts, dtype=torch.long)}


def collate_clips(items):
    """Flatten the key frames of several clip items into one batch."""
    return {k: torch.cat([it[k] for it in items]) for k in items[0]}
