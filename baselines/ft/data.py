"""Batches of the fine-tuning runs: one sampled stream of Gold training clips feeds every head of a run.

Fairness: the clip order depends only on the seed and the epoch (the same multinomial draw as the trainers'
WeightedRandomSampler, with the pair weights of their reweighting), so every run (frozen control, joint, each head alone)
sees the same clips in the same order and in the same batches.

Video: the full frame is the trainers' preprocessing (SurgKptComboDataset._prepare_video: BGR -> RGB, /255, bilinear
squash to 384 without antialias, ImageNet mean/std). The zoom crop is cut on the fly from the same native frames with
UniTAP's crop_resize (tools/build_zoom_dataset.py) and the boxes of experiment A (data/case32/v1_zoom/boxes.json), so it
equals the stored video_16_npy of the zoom root byte for byte; the zoom root of a restored experiment needs no video.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def prepare_video(frames_bgr_u8, size=384):
    """[T,H,W,3] uint8 BGR -> [T,3,size,size] float32, exactly SurgKptComboDataset._prepare_video (both snapshots)."""
    if frames_bgr_u8.ndim != 4 or frames_bgr_u8.shape[-1] != 3:
        raise ValueError(f"video must be [T,H,W,3], got {frames_bgr_u8.shape}")
    rgb = np.ascontiguousarray(frames_bgr_u8[..., ::-1])
    v = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().float() / 255.0
    v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
    return (v - IMAGENET_MEAN) / IMAGENET_STD


class Zoom:
    """Zoom crops of experiment A: train boxes = true points plus the jitter drawn once per clip, val/test boxes = stage-1
    predictions of A, as build_zoom_dataset.py wrote them into boxes.json."""

    def __init__(self, exp):
        exp = Path(exp)
        spec = importlib.util.spec_from_file_location("ft_build_zoom_dataset", exp / "tools/build_zoom_dataset.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.crop_resize = mod.crop_resize
        self.boxes = json.load(open(exp / "data/case32/v1_zoom/boxes.json"))["boxes"]

    def __contains__(self, sid):
        return sid in self.boxes

    def __call__(self, sid, frames):
        bx0, by0, L = (int(v) for v in self.boxes[sid])
        return self.crop_resize(frames, bx0, by0, L)


def load_frames(video_root, sid):
    return np.load(Path(video_root) / "video_16_npy" / f"{sid}.npy")


class JointTrain(Dataset):
    """Item: sid, video_full [16,3,384,384] (if a full-frame head is active), video_zoom (if a zoom head trains on this
    clip), and items[head] = that head's own dataset item (targets only, load_video=False) for every head whose training
    split holds the clip."""

    def __init__(self, sids, video_root, heads, zoom=None):
        self.sids = list(sids)
        self.video_root = Path(video_root)
        self.heads = {h.name: (h.ds_train, h.train_index, h.tokens) for h in heads}
        self.need_full = any(h.tokens == "full" for h in heads)
        self.zoom = zoom
        if any(h.tokens == "zoom" for h in heads) and zoom is None:
            raise ValueError("zoom heads need the zoom boxes")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        sid = self.sids[i]
        frames = load_frames(self.video_root, sid)
        out = {"sid": sid, "items": {}}
        if self.need_full:
            out["video_full"] = prepare_video(frames)
        zoom_needed = False
        for name, (ds, index, kind) in self.heads.items():
            if sid in index and (kind == "full" or sid in self.zoom):
                out["items"][name] = ds[index[sid]]
                zoom_needed |= kind == "zoom"
        if zoom_needed:
            out["video_zoom"] = prepare_video(self.zoom(sid, frames))
        return out


class Collate:
    """A micro-batch: sids, video_full [b,...] or None, video_zoom [bz,...] or None, and per head (sub_batch, rows):
    the head's items collated as its trainer's DataLoader does (default_collate), and the rows of its clips in the
    video tensor it reads (full or zoom)."""

    def __init__(self, heads):
        self.kind = {h.name: h.tokens for h in heads}

    def __call__(self, samples):
        sids = [s["sid"] for s in samples]
        vf = torch.stack([s["video_full"] for s in samples]) if "video_full" in samples[0] else None
        zpos = [i for i, s in enumerate(samples) if "video_zoom" in s]
        vz = torch.stack([samples[i]["video_zoom"] for i in zpos]) if zpos else None
        zrow = {i: r for r, i in enumerate(zpos)}
        heads = {}
        for name, kind in self.kind.items():
            pos = [i for i, s in enumerate(samples) if name in s["items"]]
            if not pos:
                continue
            sub = default_collate([samples[i]["items"][name] for i in pos])
            rows = pos if kind == "full" else [zrow[i] for i in pos]
            heads[name] = (sub, torch.tensor(rows, dtype=torch.long))
        return {"sids": sids, "video_full": vf, "video_zoom": vz, "heads": heads}


def epoch_plan(weights, batch, micro, seed, epoch, limit=0):
    """The clip stream of one epoch: len(weights) draws with replacement (WeightedRandomSampler's torch.multinomial),
    cut into optimizer steps of `batch` clips (the last step may be smaller, drop_last=False as in the trainers) and each
    step into micro-batches of at most `micro` clips. Returns (micro_batches, step_of_micro)."""
    g = torch.Generator().manual_seed(int(seed) * 1000 + int(epoch))
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=torch.double)
    n = len(weights) if not limit else min(int(limit), len(weights))
    idx = torch.multinomial(w, len(weights), replacement=True, generator=g).tolist()[:n]
    micros, step_of = [], []
    for k, s in enumerate(range(0, n, batch)):
        st = idx[s:s + batch]
        for j in range(0, len(st), micro):
            micros.append(st[j:j + micro])
            step_of.append(k)
    return micros, step_of


def train_loader(ds, micros, heads, workers, prefetch=2):
    return DataLoader(ds, batch_sampler=micros, collate_fn=Collate(heads), num_workers=workers, pin_memory=True,
                      prefetch_factor=(prefetch if workers > 0 else None), persistent_workers=False)


# --------------------------------------------------------------------------------------------- token caches
class _Clips(Dataset):
    def __init__(self, sids, video_root, zoom=None):
        self.sids, self.video_root, self.zoom = list(sids), Path(video_root), zoom

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        sid = self.sids[i]
        f = load_frames(self.video_root, sid)
        if self.zoom is not None:
            f = self.zoom(sid, f)
        return sid, prepare_video(f)


def _collate_clips(b):
    return [x[0] for x in b], torch.stack([x[1] for x in b])


def _save(path, arr):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


@torch.no_grad()
def fp32_mode():
    """How plain fp32 runs. 'converter' (default): PyTorch's own defaults, as TactusNet's token converter, trainers and
    pipeline ran (none sets a TF32 flag; on torch 2.8 cuDNN convolutions, e.g. the patch embedding, use TF32 and matmuls do
    not). 'strict': TF32 off everywhere."""
    m = os.environ.get("FT_FP32", "converter")
    if m not in ("converter", "strict"):
        raise SystemExit(f"FT_FP32={m}: converter | strict")
    return m


def set_fp32_flags(mode=None):
    if (mode or fp32_mode()) == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def write_tokens(enc, sids, video_root, dst, zoom=None, batch=8, workers=6, amp=True, device="cuda", skip_existing=False):
    """Encode clips into a token cache dir in TactusNet's format: dst/<sid>.npy, fp16 [8,24,24,1024].
    amp=True: bf16 autocast; amp=False: plain fp32 as TactusNet's token converter (fp32_mode)."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sids if not (skip_existing and (dst / f"{s}.npy").is_file())]
    if not todo:
        return 0
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    if not amp:
        set_fp32_flags()
    was_training = enc.training
    enc.eval()
    try:
        loader = DataLoader(_Clips(todo, video_root, zoom), batch_size=batch, shuffle=False, num_workers=workers,
                            collate_fn=_collate_clips, pin_memory=True)
        for sids_b, v in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
                tok = enc(v.to(device, non_blocking=True))
            tok = tok.float()
            if not torch.isfinite(tok).all():
                raise RuntimeError(f"non-finite tokens in {sids_b}")
            arr = tok.cpu().numpy().astype(np.float16)
            for s, t in zip(sids_b, arr):
                _save(dst / f"{s}.npy", t)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        enc.train(was_training)
    return len(todo)
