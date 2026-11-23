#!/usr/bin/env python3
"""SegFormer-B2 (Xie et al., NeurIPS 2021) for Where: segments the contact region of a frame.

SegFormer's OFFICIAL configuration is kept (NVlabs/SegFormer local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py
and its _base_ files: dataset ade20k_repeat, schedule 160k, mmseg 0.x transforms), see OFFICIAL below. Only what TACTUS
requires changes; every change is listed in ADAPTATIONS and written to info.json:

 init     baselines rule: public pretrained BACKBONE only. MiT-B2 encoder = ImageNet-1k weights of Hugging Face
          nvidia/mit-b2 (pinned revision); the all-MLP decoder (768 ch) and the classifier start from random init,
          which is also how SegFormer was trained (the official config loads pretrained/mit_b2.pth into the backbone).
 (a)      6 output channels (one sigmoid map per tool-action pair) instead of 150 softmax classes; the loss and the
          readout use only the true pair's channel; loss = binary CE + soft Dice on that channel (--loss bce drops Dice),
          padded pixels ignored like the official seg_pad_val 255.
 (b)      inputs are protocol frames: training images are the Gold-train protocol frames nearest to each whole second
          (ex.frames_1fps, about one frame per second) that have >= 2 visible contact points; the target is the
          metric's own region (common.hull.hull_mask of the visible GT points at 384), resized (nearest) with the image.
          Testing uses all 16 protocol frames of every clip.
 (c)      schedule length: the official 160k iterations are kept (--max-iters cuts it if the budget requires; the poly
          shape, the 1500-iteration warmup and 10 evaluations over the schedule are kept).
 (d)      batch: the official total batch is 16 (8 GPUs x 2); here one GPU runs --batch (default 16) and the LR scales
          linearly (6e-5 x batch / 16), with iterations and warmup scaled by 16 / batch.
 test     official test scale (keep ratio, short side 512, whole image); the logits of the true pair are upsampled
          (bilinear) to the 384x384 metric grid, then sigmoid > 0.5, then common.metrics.contact_frame_dice.
 other    one GPU: BatchNorm instead of SyncBN in the decoder; bf16 autocast (official fp32; --no-amp for fp32);
          best.pt by the val Dice at the 10 evaluation points (the official run keeps its last checkpoint).

  python -m baselines.where.segformer --out $BL_OUT/segformer [--smoke] [--batch 16] [--max-iters N]
"""
import argparse, json, math, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from ..common import config
from ..common import metrics as M
from ..common import preds as P
from ..common.data import get_export
from ..common.hull import hull_mask
from ..common.train_utils import Checkpointer, JsonlLog, set_seed

HF_ID = "nvidia/mit-b2"                                 # MiT-B2 encoder, ImageNet-1k pretrained (no decoder)
HF_REV = "3bb39e8739149c3777d0325349b2a6c32c6413db"     # pinned revision (2022-08-06), pytorch_model.bin 98,978,917 bytes
WEIGHTS_NAME = f"{HF_ID}@{HF_REV}"
OFFICIAL = {   # segformer.b2.512x512.ade.160k.py (+ _base_/datasets/ade20k_repeat.py, _base_/models/segformer.py)
    "backbone": "mit_b2 (drop_path_rate 0.1), pretrained pretrained/mit_b2.pth (ImageNet-1k)",
    "decode_head": "SegFormerHead, embed_dim 768, dropout 0.1, SyncBN, CrossEntropyLoss over 150 classes",
    "train_pipeline": "Resize(img_scale=(2048, 512), ratio_range=(0.5, 2.0)) -> RandomCrop(512x512, cat_max_ratio=0.75) -> "
                      "RandomFlip(0.5) -> PhotoMetricDistortion -> Normalize(mean 123.675/116.28/103.53, std 58.395/57.12/57.375, to_rgb) -> "
                      "Pad(512x512, pad_val 0, seg_pad_val 255)",
    "test_pipeline": "Resize(img_scale=(2048, 512), keep_ratio) -> whole-image inference -> logits resized to the image",
    "optimizer": "AdamW lr 6e-5, betas (0.9, 0.999), weight_decay 0.01, paramwise custom_keys: pos_block decay_mult 0, "
                 "norm decay_mult 0, head lr_mult 10",
    "lr_config": "poly, power 1.0, min_lr 0, by_epoch False, linear warmup 1500 iters, warmup_ratio 1e-6",
    "runner": "IterBasedRunner, max_iters 160000; evaluation every 16000 iters", "batch": "samples_per_gpu 2 x 8 GPUs = 16",
    "train_images": "ADE20K train, 20,210 images"}
CROP = 512; IMG_SCALE = (2048, 512); RATIO_RANGE = (0.5, 2.0); CAT_MAX_RATIO = 0.75; IGNORE = 255
MAX_ITERS, WARMUP_ITERS, EVAL_ITERS, BASE_BATCH, BASE_LR = 160000, 1500, 16000, 16, 6e-5
MEAN = np.array([123.675, 116.28, 103.53], np.float32); STD = np.array([58.395, 57.12, 57.375], np.float32)
ADAPTATIONS = [
    "init: encoder from nvidia/mit-b2 (ImageNet-1k), decode head and classifier randomly initialized (baselines rule; same as the official mit_b2.pth init)",
    "(a) 6 sigmoid output channels, one per tool-action pair, instead of 150 softmax classes; loss and readout on the true pair's channel only; "
    "loss = binary cross-entropy + soft Dice (official: softmax cross-entropy), padded pixels ignored",
    "(b) training images = Gold-train protocol frames at about 1 fps (ex.frames_1fps) with >= 2 visible contact points; target = the metric's "
    "region (hull_mask of the visible GT contact points at 384), resized with the image (nearest); test = all 16 protocol frames",
    "(c) official 160k-iteration schedule kept unless --max-iters is given (then the poly shape, warmup and 10 evaluations are kept)",
    "(d) one GPU at --batch with linear LR scaling (6e-5 x batch / 16); iterations and warmup scaled by 16 / batch",
    "no RandomFlip (TACTUS rule for keypoint and contact methods)",
    "test: official test scale (short side 512, keep ratio), logits of the true pair upsampled to the 384x384 metric grid, sigmoid > 0.5",
    "one GPU: BatchNorm instead of SyncBN; bf16 autocast unless --no-amp; best.pt = best val Dice over the 10 evaluation points"]


# ---------------------------------------------------------------- weights and model
def ensure_weights(path=None):
    """Download the pinned checkpoint into $BL_WEIGHTS once (idempotent); returns the local dir."""
    p = Path(path or config.WEIGHTS / "mit-b2")
    if not (p / "config.json").exists() or not (p / "pytorch_model.bin").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(HF_ID, revision=HF_REV, local_dir=str(p), allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"])
    return p


def build_model(path, n_pairs):
    """SegFormer-B2 for semantic segmentation: encoder weights from the MiT-B2 checkpoint, decode head (all-MLP decoder
    and classifier) newly initialized. Checks that exactly the encoder was loaded and only the decode head is new."""
    from transformers import SegformerForSemanticSegmentation
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
    model, li = SegformerForSemanticSegmentation.from_pretrained(
        str(path), num_labels=n_pairs, id2label={i: str(i) for i in range(n_pairs)}, label2id={str(i): i for i in range(n_pairs)},
        output_loading_info=True)
    missing = sorted(li.get("missing_keys", [])); mism = li.get("mismatched_keys", [])
    bad = [k for k in missing if not k.startswith("decode_head.")]
    if bad or mism or not any(n.startswith("segformer.encoder.") for n, _ in model.named_parameters()):
        raise RuntimeError(f"unexpected load of {path}: missing {bad[:5]}, mismatched {mism[:5]}")
    c = model.config
    model.load_report = {"weights": WEIGHTS_NAME, "encoder_tensors_loaded": len([n for n in model.state_dict() if n.startswith("segformer.")]),
                         "decode_head_tensors_new": len(missing), "checkpoint_tensors_unused": sorted(li.get("unexpected_keys", [])),
                         "decoder_hidden_size": c.decoder_hidden_size, "classifier_dropout": c.classifier_dropout_prob, "drop_path_rate": c.drop_path_rate}
    return model


class SegFormerWhere(nn.Module):
    def __init__(self, path, n_pairs):
        super().__init__(); self.net = build_model(path, n_pairs); self.load_report = self.net.load_report

    def forward(self, x, pair):
        """x [B,3,H,W] normalized, pair [B] -> logits of the true pair's channel at 1/4 resolution [B,H/4,W/4]."""
        z = self.net(pixel_values=x).logits
        return z[torch.arange(z.shape[0], device=z.device), pair]


def param_groups(model, lr, wd):
    """mmcv DefaultOptimizerConstructor with the official custom_keys: keys sorted by name then by length (descending),
    the first key contained in the parameter name sets lr_mult / decay_mult; biases keep weight decay (no bias rule)."""
    keys = {"pos_block": {"decay_mult": 0.0}, "norm": {"decay_mult": 0.0}, "head": {"lr_mult": 10.0}}
    order = sorted(sorted(keys), key=len, reverse=True); groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        k = next((k for k in order if k in n), None); cfg = keys.get(k, {})
        g_lr, g_wd = lr * cfg.get("lr_mult", 1.0), wd * cfg.get("decay_mult", 1.0)
        g = groups.setdefault((k, g_lr, g_wd), {"params": [], "lr": g_lr, "base_lr": g_lr, "weight_decay": g_wd, "name": k or "default"})
        g["params"].append(p)
    return list(groups.values())


def lr_factor(it, total, warmup, power=1.0, warmup_ratio=1e-6):
    """mmcv PolyLrUpdaterHook (min_lr 0) with linear warmup: the warmup scales the poly LR by 1 - (1 - it/W)(1 - ratio)."""
    f = (1 - it / total) ** power
    if it < warmup: f *= 1 - (1 - it / warmup) * (1 - warmup_ratio)
    return f


# ---------------------------------------------------------------- official transforms (mmseg 0.x), flips removed
def rescale_size(h, w, scale):
    """mmcv.rescale_size with a (long, short) scale tuple: keep ratio, fit inside it; returns (new_h, new_w)."""
    f = min(max(scale) / max(h, w), min(scale) / min(h, w))
    return int(h * f + 0.5), int(w * f + 0.5)


def random_crop(img, seg, rng):
    """mmseg RandomCrop(crop_size 512, cat_max_ratio 0.75): up to 10 draws until no class covers >= 75% of the crop."""
    def bbox():
        mh, mw = max(img.shape[0] - CROP, 0), max(img.shape[1] - CROP, 0)
        y, x = int(rng.integers(0, mh + 1)), int(rng.integers(0, mw + 1)); return y, y + CROP, x, x + CROP
    b = bbox()
    for _ in range(10):
        labels, cnt = np.unique(seg[b[0]:b[1], b[2]:b[3]], return_counts=True); cnt = cnt[labels != IGNORE]
        if len(cnt) > 1 and cnt.max() / cnt.sum() < CAT_MAX_RATIO: break
        b = bbox()
    return img[b[0]:b[1], b[2]:b[3]], seg[b[0]:b[1], b[2]:b[3]]


def photometric_distortion(img, rng):
    """mmseg PhotoMetricDistortion (brightness delta 32, contrast 0.5-1.5, saturation 0.5-1.5, hue delta 18), uint8 RGB."""
    import cv2
    conv = lambda x, a=1.0, b=0.0: np.clip(x.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.integers(2): img = conv(img, b=rng.uniform(-32, 32))
    mode = rng.integers(2)
    if mode == 1 and rng.integers(2): img = conv(img, a=rng.uniform(0.5, 1.5))
    if rng.integers(2):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); hsv[..., 1] = conv(hsv[..., 1], a=rng.uniform(0.5, 1.5)); img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if rng.integers(2):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); hsv[..., 0] = (hsv[..., 0].astype(int) + int(rng.integers(-18, 18))) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if mode == 0 and rng.integers(2): img = conv(img, a=rng.uniform(0.5, 1.5))
    return img


def normalize(img_u8):
    return ((img_u8.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)


def train_sample(frame_rgb, mask384, rng):
    """Official train pipeline without RandomFlip: returns image [3,512,512] float32 and seg [512,512] uint8 (0, 1, 255)."""
    import cv2
    h, w = frame_rgb.shape[:2]; r = rng.uniform(*RATIO_RANGE)
    nh, nw = rescale_size(h, w, (int(IMG_SCALE[0] * r), int(IMG_SCALE[1] * r)))
    img = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    seg = cv2.resize(mask384.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    img, seg = random_crop(img, seg, rng); img = photometric_distortion(np.ascontiguousarray(img), rng)
    x = np.zeros((3, CROP, CROP), np.float32); y = np.full((CROP, CROP), IGNORE, np.uint8)   # Pad after Normalize
    x[:, :img.shape[0], :img.shape[1]] = normalize(img); y[:seg.shape[0], :seg.shape[1]] = seg
    return x, y


def test_image(frame_rgb):
    """Official test pipeline: keep-ratio resize into (2048, 512), i.e. short side 512 for video frames."""
    import cv2
    h, w = frame_rgb.shape[:2]; nh, nw = rescale_size(h, w, IMG_SCALE)
    return normalize(cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR))


# ---------------------------------------------------------------- data
def metric_region(ex, sid, t):
    gt, vis = ex.contact_points(sid); vk = vis[t] > 0.5
    return hull_mask(gt[t][vk].astype(np.float64) * config.METRIC_SIZE)


def train_frames(ex, sids):
    """(sid, t) of the 1-fps protocol frames with >= 2 visible contact points."""
    out = []
    for s in sids:
        _, vis = ex.contact_points(s)
        out += [(s, int(t)) for t in ex.frames_1fps(s) if (vis[t] > 0.5).sum() >= 2]
    return out


class StreamFrames(Dataset):
    """Sample j of the training stream: pass j // n over the n training images in a per-pass shuffled order (seeded),
    augmented with a per-sample seed, so a resumed run sees exactly the same stream."""
    def __init__(self, ex, items, seed, length):
        self.ex, self.items, self.seed, self.length = ex, items, seed, length; self._perm = (-1, None)

    def __len__(self): return self.length

    def __getitem__(self, j):
        n = len(self.items); p, pos = divmod(j, n)
        if self._perm[0] != p: self._perm = (p, np.random.default_rng([self.seed, p]).permutation(n))
        sid, t = self.items[int(self._perm[1][pos])]
        f = self.ex.frames(sid)[t]     # RGB, native content crop; ex.frames decodes the raw clip when video_16_npy is absent
        x, y = train_sample(np.ascontiguousarray(f), metric_region(self.ex, sid, t), np.random.default_rng([self.seed, 7, j]))
        return torch.from_numpy(x), torch.from_numpy(y), self.ex.pair_id(sid)


class RangeSampler(Sampler):
    def __init__(self, a, b): self.a, self.b = a, b
    def __iter__(self): return iter(range(self.a, self.b))
    def __len__(self): return self.b - self.a


class EvalClips(Dataset):
    def __init__(self, ex, sids): self.ex, self.sids = ex, sids
    def __len__(self): return len(self.sids)
    def __getitem__(self, i):
        fr = self.ex.frames(self.sids[i])
        return torch.from_numpy(np.stack([test_image(f) for f in fr])), self.ex.pair_id(self.sids[i]), i


# ---------------------------------------------------------------- train / eval
def loss_fn(z, y, kind="bce_dice"):
    """z logits [B,h,w] of the true pair (upsampled here to the crop, as mmseg does), y [B,512,512] in {0,1,255}."""
    z = F.interpolate(z[:, None].float(), size=y.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    valid = (y != IGNORE).float(); t = (y == 1).float()
    bce = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
    if kind == "bce": return bce, bce.detach(), torch.zeros((), device=z.device)
    p = torch.sigmoid(z) * valid
    dice = (1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()
    return bce + dice, bce.detach(), dice.detach()


@torch.no_grad()
def predict(model, ex, sids, device, amp, workers, chunk=4):
    """Per-frame Dice [N,16] (NaN on unscored frames): all 16 protocol frames, true pair's channel, logits upsampled
    to the 384x384 metric grid, sigmoid > 0.5. Frames go through the model `chunk` at a time (the decoder's four
    768-channel maps at 1/4 of 512x762 take about 0.75 GB per frame, next to the training state)."""
    model.eval(); S = config.METRIC_SIZE; out = np.full((len(sids), 16), np.nan)
    dl = DataLoader(EvalClips(ex, sids), batch_size=1, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    for x, pair, idx in dl:
        x = x[0].to(device, non_blocking=True); pr = pair.repeat(x.shape[0]).to(device); i = int(idx); zs = []
        for c in range(0, x.shape[0], chunk):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                zc = model(x[c:c + chunk], pr[c:c + chunk])
            zs.append(F.interpolate(zc[:, None].float(), size=(S, S), mode="bilinear", align_corners=False)[:, 0])
        z = torch.cat(zs)
        gt, vis = ex.contact_points(sids[i]); out[i] = M.contact_frame_dice(gt, vis, pred_masks=(z > 0).cpu().numpy())
    model.train(); return out


def pooled(fd):
    ok = ~np.isnan(fd); return float(fd[ok].mean()) if ok.any() else float("nan")


def wait_for_gpu(min_free_mb, poll=60, max_wait=4 * 3600):
    """Shared local GPU: wait until nvidia-smi reports enough free memory (no-op when min_free_mb <= 0)."""
    if min_free_mb <= 0: return
    t0 = time.time()
    while True:
        try:
            free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().split()[0])
        except Exception: return
        if free >= min_free_mb or time.time() - t0 > max_wait: return
        print(f"[segformer] GPU has {free} MiB free < {min_free_mb}, waiting", flush=True); time.sleep(poll)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=BASE_BATCH, help="per-step batch on one GPU (official total 16); LR scales linearly")
    ap.add_argument("--max-iters", type=int, default=0, help="0 = the official 160k iterations, scaled by 16 / batch")
    ap.add_argument("--loss", default="bce_dice", choices=("bce_dice", "bce"))
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--weights")
    ap.add_argument("--wait-gpu-mb", type=int, default=0, help="wait for this much free GPU memory before starting (shared GPU)")
    ap.add_argument("--export"); ap.add_argument("--hf-raw"); a = ap.parse_args(argv)
    if a.smoke: a.batch, a.max_iters, a.workers = min(a.batch, 2), a.max_iters or 6, min(a.workers, 2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); set_seed(a.seed); torch.backends.cudnn.benchmark = True
    ex = get_export(a.export, a.hf_raw); wpath = ensure_weights(a.weights); wait_for_gpu(a.wait_gpu_mb)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = (not a.no_amp) and dev.type == "cuda"
    scale = BASE_BATCH / a.batch; T = a.max_iters or int(round(MAX_ITERS * scale)); lr = BASE_LR * a.batch / BASE_BATCH
    W = max(1, min(int(round(WARMUP_ITERS * scale)), T // 10)); I = max(1, math.ceil(T / (MAX_ITERS // EVAL_ITERS)))
    if a.smoke: I = max(I, T // 2)                       # two evaluations in a smoke run
    items = train_frames(ex, ex.sids("train"))
    if not items: raise SystemExit("no training frames (1-fps frames with >= 2 visible contact points)")
    model = SegFormerWhere(wpath, len(ex.pairs)).to(dev)
    opt = torch.optim.AdamW(param_groups(model, lr, 0.01), lr=lr, betas=(0.9, 0.999))
    ck = Checkpointer(out, metric="dice", mode="max"); k0 = ck.resume(model, opt); log = JsonlLog(out / "log.jsonl")
    i0 = min(k0 * I, T); val_sids = ex.sids("val")
    print(f"[segformer] {len(items)} training images (1-fps frames of {len({s for s, _ in items})} Gold-train clips); "
          f"{T} iterations at batch {a.batch} ({T * a.batch / max(len(items), 1):.1f} passes), lr {lr:.2e}, warmup {W}, "
          f"eval every {I}; start at iteration {i0}; device {dev}", flush=True)
    ds = StreamFrames(ex, items, a.seed, T * a.batch); t_start = time.time()
    dl = DataLoader(ds, batch_size=a.batch, sampler=RangeSampler(i0 * a.batch, T * a.batch), num_workers=a.workers,
                    pin_memory=dev.type == "cuda", drop_last=False, persistent_workers=False)
    it = i0; acc = np.zeros(3); n = 0; t0 = time.time(); model.train()
    for x, y, pair in (dl if i0 < T else []):
        f = lr_factor(it, T, W)
        for g in opt.param_groups: g["lr"] = g["base_lr"] * f
        x, y, pair = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), pair.to(dev)
        for attempt in range(10):           # the local GPU is shared: on OOM wait and retry the same step
            try:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    z = model(x, pair)
                loss, bce, dice = loss_fn(z, y, a.loss); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); print("[segformer] OOM, retry in 60 s", flush=True); time.sleep(60)
        else:
            raise SystemExit("out of GPU memory 10 times in a row")
        it += 1; acc += [float(loss.detach()), float(bce), float(dice)]; n += 1
        if it % 50 == 0: print(f"  it {it}/{T} loss {acc[0] / n:.4f} lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / n:.3f} s/it", flush=True)
        if it % I == 0 or it == T:
            t_train = time.time() - t0; fd = predict(model, ex, val_sids, dev, amp, a.workers); vd = pooled(fd); k = math.ceil(it / I) - 1
            better = ck.step(k, {"dice": vd}, {"model": model.state_dict(), "optimizer": opt.state_dict(), "iter": it})
            log.write(eval=k, iter=it, train_loss=acc[0] / n, train_bce=acc[1] / n, train_dice_loss=acc[2] / n, val_dice=vd, best=better,
                      lr_encoder=opt.param_groups[0]["lr"], s_per_iter=round(t_train / n, 4), val_s=round(time.time() - t0 - t_train, 1))
            print(f"[segformer] iter {it}/{T}: loss {acc[0] / n:.4f}  val Dice {vd:.4f}{'  (best)' if better else ''}  "
                  f"{t_train / n:.3f} s/it, val {time.time() - t0 - t_train:.0f}s", flush=True)
            acc[:] = 0; n = 0; t0 = time.time()
        if it >= T: break
    s = ck.load_best(model); wall = time.time() - t_start
    info = {"method": "segformer_b2", "paper": "Xie et al., SegFormer, NeurIPS 2021", "weights": WEIGHTS_NAME, "load_report": model.load_report,
            "official_config": OFFICIAL, "adaptations": ADAPTATIONS,
            "train_frames": f"{len(items)} 1-fps protocol frames (ex.frames_1fps) with >= 2 visible contact points, from {len({s_ for s_, _ in items})} Gold-train clips",
            "test_frames": "all 16 protocol frames", "input": "train: 512x512 crops of the rescaled frame (ratio 0.5-2.0 of short side 512); test: short side 512, keep ratio",
            "init": f"encoder {WEIGHTS_NAME} (ImageNet-1k); decode head (all-MLP, 768 ch, dropout 0.1) and 6-channel classifier random",
            "trainable": "fine-tuned end to end", "conditioning": "true pair: its channel is trained and read",
            "loss": {"bce_dice": "binary CE + soft Dice", "bce": "binary CE"}[a.loss] + " on the true pair's channel, logits upsampled to the crop, pad ignored",
            "max_iters": T, "official_iters_kept": a.max_iters == 0, "batch": a.batch, "lr": lr, "lr_rule": "6e-5 x batch / 16", "head_lr_mult": 10.0,
            "weight_decay": 0.01, "optimizer": "AdamW (0.9, 0.999), mmcv custom_keys (norm/pos_block: no decay, head: lr x10)",
            "schedule": f"poly power 1 to 0 over {T} iters, linear warmup {W} iters (ratio 1e-6)", "eval_interval": I,
            "passes_over_train_frames": T * a.batch / max(len(items), 1), "augmentation": "official minus RandomFlip",
            "amp": "bf16" if amp else "fp32", "seed": a.seed, "smoke": a.smoke, "best_eval": int(s.get("best_epoch", -1)),
            "best_iter": int(s.get("iter", -1)), "best_val_dice": float(s.get("best", float("nan"))),
            "wall_clock_s_this_run": round(wall, 1), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    for split in ("val", "test"):
        sids = ex.sids(split); fd = predict(model, ex, sids, dev, amp, a.workers)
        p = P.save(out / f"pred_{split}.npz", "kpt", sids, dict(info, split=split, checkpoint=str(out / "best.pt")), ct_frame_dice=fd)
        info[f"{split}_dice"] = pooled(fd); print(f"[segformer] {split}: pooled Dice {pooled(fd):.4f} -> {p}", flush=True)
    json.dump(info, open(out / "info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
