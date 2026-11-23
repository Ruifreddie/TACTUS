"""Each method's OFFICIAL training augmentation, re-implemented on the 16 protocol frames (no third-party code copied;
the logic follows the cited files line by line). Every function takes uint8 frames [T,P,P,3] (RGB, the whole content
crop squashed to the method's pre-crop size P) and returns float32 [T,3,224,224], normalized with the method's mean/std.
Randomness comes from the global `random` / `np.random` streams, as in the originals; data.ClipSet seeds both once per
sample, so a sample's augmentation depends only on (seed, epoch, position), not on workers or resumes.

TACTUS adaptation (b), shared by all: the official pipelines first resize a video's short side (TSM, X-CLIP: 256;
VideoMAE: decode to 320x256; Endo-FM, MViTv2: jitter 256-320); here the frame is the whole content crop squashed to a
P x P square, P = 256 or 320. At test, the official short-side resize + center crop is replaced by the whole frame
squashed to 224 x 224 (a center crop would cut tool tips at the borders).

  tsn, tsm   TSM repo ops/transforms.py: GroupMultiScaleCrop(224, [1, .875, .75, .66], max_distort 1, 13 fixed offsets,
             PIL bilinear) + GroupRandomHorizontalFlip (on unless the dataset is Something/Jester); ImageNet mean/std.
  videomae   VideoMAE kinetics.py _aug_frame: RandAugment rand-m7-n4-mstd0.5-inc1 (bicubic, translate_const 100) on the
             frames, random resized crop scale 0.08-1 / aspect 0.75-1.333 to 224 (torch bilinear), flip (on unless SSV2),
             random erasing p 0.25 'pixel' 1 box (same box on all frames); ImageNet mean/std. Mixup/CutMix in the loop.
  xclip      X-CLIP datasets/build.py train pipeline: MultiScaleCrop(224, (1, .875, .75, .66), max_wh_scale_gap 1,
             5 fixed crops) -> resize 224 (cv2 bilinear) -> flip 0.5 -> ColorJitter p 0.8 (per frame torchvision
             ColorJitter(0.4, 0.4, 0.2, 0.1)) -> GrayScale p 0.2; ImageNet mean/std. Mixup/CutMix in the loop.
  endofm     Endo-FM datasets/ucf101.py (eval_finetune.py): short side jitter U[256, 320], random crop 224, flip 0.5;
             mean 0.45 / std 0.225.
  mvitv2     PySlowFast utils.aug_frame with configs/SSv2/MVITv2_S_16x4.yaml: RandAugment rand-m7-n4-mstd0.5-inc1
             (bicubic, translate_const 0.45 x frame size), short side jitter 1/U[1/320, 1/256] (inverse uniform),
             random crop 224, flip (the SSv2 config turns it off only for SSv2's left/right classes), random erasing
             p 0.25 'pixel' 1 box per clip; mean 0.45 / std 0.225. Mixup/CutMix in the loop.
"""
import math, random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
KINETICS = ((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
CROP = 224


def to_tensor(frames):
    """uint8 [T,H,W,3] -> float32 [T,3,H,W] in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x, norm):
    m = torch.tensor(norm[0]).view(1, 3, 1, 1); s = torch.tensor(norm[1]).view(1, 3, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------- TSM / TSN (ops/transforms.py)
def _tsm_fix_offsets(image_w, image_h, crop_w, crop_h):
    """GroupMultiScaleCrop.fill_fix_offset(more_fix_crop=True): 13 positions."""
    w_step = (image_w - crop_w) // 4; h_step = (image_h - crop_h) // 4
    return [(0, 0), (4 * w_step, 0), (0, 4 * h_step), (4 * w_step, 4 * h_step), (2 * w_step, 2 * h_step),
            (0, 2 * h_step), (4 * w_step, 2 * h_step), (2 * w_step, 4 * h_step), (2 * w_step, 0),
            (w_step, h_step), (3 * w_step, h_step), (w_step, 3 * h_step), (3 * w_step, 3 * h_step)]


def tsm_train(frames, flip=True):
    imgs = [Image.fromarray(f) for f in frames]; image_w, image_h = imgs[0].size
    base = min(image_w, image_h); sizes = [int(base * x) for x in (1, .875, .75, .66)]
    crop_h = [CROP if abs(x - CROP) < 3 else x for x in sizes]; crop_w = [CROP if abs(x - CROP) < 3 else x for x in sizes]
    pairs = [(w, h) for i, h in enumerate(crop_h) for j, w in enumerate(crop_w) if abs(i - j) <= 1]
    cw, ch = random.choice(pairs); ow, oh = random.choice(_tsm_fix_offsets(image_w, image_h, cw, ch))
    imgs = [im.crop((ow, oh, ow + cw, oh + ch)).resize((CROP, CROP), Image.Resampling.BILINEAR) for im in imgs]
    if flip and random.random() < 0.5: imgs = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in imgs]
    return normalize(to_tensor(np.stack([np.asarray(im) for im in imgs])), IMAGENET)


# ---------------------------------------------------------------- RandAugment applied to a whole clip
_RA = {}


def _rand_augment(translate_const):
    """timm's rand_augment_transform('rand-m7-n4-mstd0.5-inc1') ops (same op list and level functions as the copies in
    VideoMAE and PySlowFast), with bicubic resampling as both configs set."""
    if translate_const not in _RA:
        from timm.data.auto_augment import rand_augment_transform
        _RA[translate_const] = rand_augment_transform("rand-m7-n4-mstd0.5-inc1",
                                                      {"translate_const": translate_const, "interpolation": Image.BICUBIC})
    return _RA[translate_const]


def video_rand_augment(imgs, translate_const):
    """RandAugment.__call__ + AugmentOp.__call__ of the video versions: the n ops, their application coin, magnitude
    noise and sign are drawn once per clip and applied to every frame."""
    ra = _rand_augment(translate_const)
    ops = np.random.choice(ra.ops, ra.num_layers, replace=ra.choice_weights is None, p=ra.choice_weights)
    for op in ops:
        if op.prob < 1.0 and random.random() > op.prob: continue
        mag = op.magnitude
        if op.magnitude_std > 0:
            mag = random.uniform(0, mag) if op.magnitude_std == float("inf") else random.gauss(mag, op.magnitude_std)
        mag = max(0.0, min(mag, op.magnitude_max or 10.0))
        args = op.level_fn(mag, op.hparams) if op.level_fn is not None else tuple()
        imgs = [op.aug_fn(im, *args, **op.kwargs) for im in imgs]
    return imgs


def random_erasing_cube(x, prob=0.25, min_area=0.02, max_area=1 / 3, min_aspect=0.3, count=1):
    """RandomErasing(prob, mode='pixel', max_count=1, cube=True) of VideoMAE / PySlowFast on a normalized clip
    [T,C,H,W]: one box for all frames, per-pixel N(0,1) fill drawn per frame."""
    if random.random() > prob: return x
    T, C, H, W = x.shape; area = H * W; log_ar = (math.log(min_aspect), math.log(1 / min_aspect))
    g = torch.Generator().manual_seed(random.getrandbits(32))      # noise from the seeded `random` stream
    for _ in range(count):
        for _ in range(100):
            ta = random.uniform(min_area, max_area) * area / count; ar = math.exp(random.uniform(*log_ar))
            h = int(round(math.sqrt(ta * ar))); w = int(round(math.sqrt(ta / ar)))
            if w < W and h < H:
                top = random.randint(0, H - h); left = random.randint(0, W - w)
                for t in range(T): x[t, :, top:top + h, left:left + w] = torch.empty((C, h, w)).normal_(generator=g)
                break
    return x


# ---------------------------------------------------------------- VideoMAE (kinetics.py _aug_frame)
def _crop_params(scale, ratio, height, width):
    """video_transforms._get_param_spatial_crop (10 tries, log-uniform aspect, central fallback)."""
    for _ in range(10):
        ta = random.uniform(*scale) * height * width; ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w = int(round(math.sqrt(ta * ar))); h = int(round(math.sqrt(ta / ar)))
        if 0 < w <= width and 0 < h <= height:
            return random.randint(0, height - h), random.randint(0, width - w), h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio): w = width; h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio): h = height; w = int(round(h * max(ratio)))
    else: w = width; h = height
    return (height - h) // 2, (width - w) // 2, h, w


def videomae_train(frames, flip=True):
    imgs = video_rand_augment([Image.fromarray(f) for f in frames], translate_const=int(CROP * 0.45))
    x = to_tensor(np.stack([np.asarray(im) for im in imgs]))
    i, j, h, w = _crop_params((0.08, 1.0), (0.75, 1.3333), x.shape[2], x.shape[3])
    x = F.interpolate(x[:, :, i:i + h, j:j + w], size=(CROP, CROP), mode="bilinear", align_corners=False)
    if flip and np.random.uniform() < 0.5: x = x.flip(-1)
    return random_erasing_cube(normalize(x, IMAGENET))


# ---------------------------------------------------------------- X-CLIP (datasets/build.py, pipeline.py)
_CJ = None


def xclip_train(frames):
    import cv2, torchvision
    global _CJ
    H, W = frames.shape[1:3]; base = min(H, W); sizes = [int(base * s) for s in (1, 0.875, 0.75, 0.66)]
    cand = [[w, h] for i, h in enumerate(sizes) for j, w in enumerate(sizes) if abs(i - j) <= 1]
    cw, ch = random.choice(cand); cw = CROP if abs(cw - CROP) < 3 else cw; ch = CROP if abs(ch - CROP) < 3 else ch
    ws, hs = (W - cw) // 4, (H - ch) // 4
    ox, oy = random.choice([(0, 0), (4 * ws, 0), (0, 4 * hs), (4 * ws, 4 * hs), (2 * ws, 2 * hs)])
    out = [cv2.resize(np.ascontiguousarray(f[oy:oy + ch, ox:ox + cw]), (CROP, CROP), interpolation=cv2.INTER_LINEAR) for f in frames]
    if np.random.rand() < 0.5: out = [np.ascontiguousarray(f[:, ::-1]) for f in out]   # mmaction Flip(flip_ratio=0.5)
    if random.random() < 0.8:                                                             # ColorJitter(p=0.8), per frame
        if _CJ is None: _CJ = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        with torch.random.fork_rng(devices=[]):      # torchvision draws from torch's RNG: seed it from `random`, locally
            torch.manual_seed(random.getrandbits(32)); out = [np.asarray(_CJ(Image.fromarray(f))) for f in out]
    if random.random() < 0.2:                                                             # GrayScale(p=0.2)
        out = [np.asarray(Image.fromarray(f).convert("L").convert("RGB")) for f in out]
    return normalize(to_tensor(np.stack(out)), IMAGENET)


# ---------------------------------------------------------------- Endo-FM and MViTv2 (SlowFast-style spatial sampling)
def _short_side_jitter(x, lo, hi, inverse_uniform):
    """transform.random_short_side_scale_jitter on a square clip [T,C,P,P]: torch bilinear, align_corners False."""
    size = int(round(1.0 / np.random.uniform(1.0 / hi, 1.0 / lo))) if inverse_uniform else int(round(np.random.uniform(lo, hi)))
    if x.shape[-1] == size: return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _random_crop(x, size):
    """transform.random_crop: offsets np.random.randint(0, side - size) (0 when the side equals the size)."""
    H, W = x.shape[-2:]
    y = int(np.random.randint(0, H - size)) if H > size else 0; xo = int(np.random.randint(0, W - size)) if W > size else 0
    return x[..., y:y + size, xo:xo + size]


def endofm_train(frames):
    x = normalize(to_tensor(frames), KINETICS)
    x = _random_crop(_short_side_jitter(x, 256, 320, inverse_uniform=False), CROP)
    if np.random.uniform() < 0.5: x = x.flip(-1)
    return x


def mvitv2_train(frames):
    imgs = video_rand_augment([Image.fromarray(f) for f in frames], translate_const=int(min(frames.shape[1:3]) * 0.45))
    x = normalize(to_tensor(np.stack([np.asarray(im) for im in imgs])), KINETICS)
    x = _random_crop(_short_side_jitter(x, 256, 320, inverse_uniform=True), CROP)
    if np.random.uniform() < 0.5: x = x.flip(-1)
    return random_erasing_cube(x)


TRAIN = {"tsn": tsm_train, "tsm": tsm_train, "videomae": videomae_train, "xclip": xclip_train,
         "endofm": endofm_train, "mvitv2": mvitv2_train}


def test_view(frames, norm):
    """uint8 [T,224,224,3] (whole content crop squashed to 224) -> normalized float32 [T,3,224,224]."""
    return normalize(to_tensor(frames), norm)


# ---------------------------------------------------------------- batch mixing (applied in the training loop)
def one_hot(y, n, on, off):
    return torch.full((y.shape[0], n), off, device=y.device).scatter_(1, y.view(-1, 1), on)


class TimmMixup:
    """Mixup/CutMix as in VideoMAE's mixup.py and PySlowFast's MixUp (timm logic, mode 'batch'): one lambda per batch,
    partner = the batch reversed, CutMix box on the last two dims with lambda correction, label smoothing on both
    targets. (timm 1.0's own Mixup indexes x[:, :, y, x] and is wrong for 5-D video, hence this copy of the logic.)"""

    def __init__(self, n_cls, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, smoothing=0.1):
        self.n, self.ma, self.ca, self.prob, self.sw, self.sm = n_cls, mixup_alpha, cutmix_alpha, prob, switch_prob, smoothing

    def __call__(self, x, y, rng):
        lam, cut = 1.0, False
        if rng.random() < self.prob:
            cut = rng.random() < self.sw
            lam = float(rng.beta(self.ca, self.ca) if cut else rng.beta(self.ma, self.ma))
        if lam != 1.0:
            if cut:
                H, W = x.shape[-2:]; r = math.sqrt(1 - lam); ch, cw = int(H * r), int(W * r)
                cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
                yl, yh = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
                xl, xh = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
                x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
                lam = 1.0 - (yh - yl) * (xh - xl) / float(H * W)
            else:
                x.mul_(lam).add_(x.flip(0).mul_(1.0 - lam))
        off = self.sm / self.n; on = 1.0 - self.sm + off
        t = one_hot(y, self.n, on, off) * lam + one_hot(y.flip(0), self.n, on, off) * (1.0 - lam)
        return x, t


class XClipBlend:
    """X-CLIP datasets/blending.py CutmixMixupBlending(smoothing 0.1, mixup_alpha 0.8, cutmix_alpha 1, switch_prob
    0.5): always mixes, partner = a random permutation, CutMix box from X-CLIP's rand_bbox with lambda correction."""

    def __init__(self, n_cls, smoothing=0.1, mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5):
        self.n, self.sm, self.ma, self.ca, self.sw = n_cls, smoothing, mixup_alpha, cutmix_alpha, switch_prob

    def __call__(self, x, y, rng):
        off = self.sm / self.n; on = 1.0 - self.sm + off; t = one_hot(y, self.n, on, off)
        perm = torch.from_numpy(rng.permutation(x.shape[0])).to(x.device)
        if rng.random() < self.sw:
            lam = float(rng.beta(self.ca, self.ca)); H, W = x.shape[-2:]; r = math.sqrt(1.0 - lam)
            cw, ch = int(W * r), int(H * r); cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
            x1, y1 = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
            x2, y2 = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
            x[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
            lam = 1.0 - (x2 - x1) * (y2 - y1) / float(W * H)
        else:
            lam = float(rng.beta(self.ma, self.ma)); x = lam * x + (1 - lam) * x[perm]
        return x, lam * t + (1 - lam) * t[perm]
