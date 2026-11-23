"""ActionFormer and TriDet: anchor-free temporal action detection heads on the frozen V-JEPA 2.1 sequence.

Both keep their official THUMOS14 configuration (configs/thumos_i3d.yaml + libs/core/config.py of each repo): the
architecture, AdamW with linear warmup and cosine, the losses (sigmoid focal + DIoU, TriDet's IoU-weighted focal loss
and Trident head), centre sampling (radius 1.5), the random crop augmentation (crop_ratio [0.9, 1.0], trunc_thresh 0.5),
the model EMA used for evaluation, and the decoding filters. What TACTUS requires changes (adaptations() lists them):
  (a) 6 classes, one per valid tool-action pair; the focal loss sees only the true pair's channel of each clip (the
      other five are masked) and the test reads only that channel (score, and with TriDet its Trident boundary bins);
  (b) max_seq_len = T (8 or 32 steps): FPN levels cut to what T supports (default 3: strides 1, 2, 4), the official
      regression ranges cut to those levels ([0,4], [4,8], [8,inf) at T = 8, x4 at T = 32), ActionFormer's local
      attention window 19 > T becomes global attention, TriDet's 16 Trident bins become 4 at T = 8 (16 at T = 32);
  (c) the official number of epochs, so the iterations follow from our data size (6,221 iterations per epoch);
  (d) the default keeps the official batch 2 and LR 1e-4; the sweep entry `b64` (batch 64, LR scaled linearly to
      3.2e-3) is an extra config that val may choose.
Decoding: the top-1 segment of the true channel after the official filters (score > 0.001, duration > 0.05 steps). With
the official per-class soft-NMS (multiclass_nms: True, which disables voting) the top-scoring segment is never changed,
so this equals the official top-1; no NMS is run. If nothing passes the filters, the whole window is output (fallback).

Time mapping: step k of a T-step sequence sits at grid position g = k (the official points, use_offset=False) and at
x = (2g + 0.5) / (F - 1) on the window axis (F = 2T frames). The interval [s, e] becomes [g(s), g(e)] on the grid; a
predicted grid segment is mapped back and clipped to [0, 1]. The crop offset of ActionFormer's truncation is its
feat_offset = 0.5 * num_frames / feat_stride = 0.5 * 2 / 2 frames.
"""
import numpy as np
import torch

from ..data import grid_to_win, win_to_grid
from .base import Method, adapt, batch_adapt, finish, range_ladder

# official THUMOS14 values shared by the two repos (train_cfg / test_cfg / opt of configs/thumos_i3d.yaml + defaults)
TRAIN_CFG = dict(center_sample="radius", center_sample_radius=1.5, loss_weight=1.0, cls_prior_prob=0.01, init_loss_norm=100,
                 clip_grad_l2norm=1.0, dropout=0.0, droppath=0.1, label_smoothing=0.0)
TEST_CFG = dict(pre_nms_thresh=0.001, pre_nms_topk=2000, iou_threshold=0.1, min_score=0.001, max_seg_num=200, nms_method="soft",
                nms_sigma=0.5, duration_thresh=0.05, multiclass_nms=True, voting_thresh=0.7)
CROP = dict(crop_ratio=[0.9, 1.0], trunc_thresh=0.5)


class _TAL(Method):
    sched_per = "iter"
    arch_name = ""
    package = None
    truncate_offset = None    # ActionFormer passes feat_offset to truncate_feats, TriDet has no offset

    @classmethod
    def derived(cls, steps, r):
        d = super().derived(steps, r)
        d["regression_range"] = range_ladder(int(r["fpn_levels"]), steps)
        return d

    def _cfg(self):
        raise NotImplementedError

    def build(self):
        pkg = self.package()
        self.model = pkg.make_meta_arch(self.arch_name, **self._cfg()).to(self.device)
        self.ema = pkg.ModelEma(self.model, decay=float(self.recipe["ema_decay"]))
        self._pkg = pkg
        import importlib
        self._truncate = importlib.import_module(pkg.__name__ + ".data_utils").truncate_feats

    def total_epochs(self):
        return int(self.recipe["epochs"]) + int(self.recipe["warmup_epochs"])       # official: max_epochs = epochs + warmup

    def make_optimizer(self, iters_per_epoch):
        r = self.recipe
        opt = self._pkg.make_optimizer(self.model, {"type": "AdamW", "learning_rate": r["lr"], "weight_decay": r["weight_decay"], "momentum": 0.9})
        sched = self._pkg.LinearWarmupCosineAnnealingLR(opt, int(r["warmup_epochs"]) * iters_per_epoch, self.total_epochs() * iters_per_epoch,
                                                        eta_min=float(r["eta_min"]))
        return opt, sched

    def grad_clip(self):
        v = float(self.recipe["clip_grad_l2norm"]); return v if v > 0 else None

    def _crop(self, x, pair, g):
        """The official training crop (truncate_feats, called verbatim per window), then padding to max_seq_len = T with
        masks, as the official preprocessing does. Returns feats [B,D,T], masks [B,1,T], segments and labels lists."""
        B, T, D = x.shape; r = self.recipe; st = np.zeros(B, np.int64); ln = np.full(B, T, np.int64); segs, labels = [], []
        for i in range(B):
            d = {"feats": torch.arange(T, dtype=torch.float32)[None], "segments": torch.as_tensor(g[i:i + 1], dtype=torch.float32),
                 "labels": pair[i:i + 1].cpu()}
            args = (d, T, r["trunc_thresh"]) + ((self.truncate_offset,) if self.truncate_offset is not None else ()) + (r["crop_ratio"],)
            d = self._truncate(*args)
            st[i], ln[i] = int(d["feats"][0, 0]), d["feats"].shape[1]
            segs.append(d["segments"].to(self.device)); labels.append(d["labels"].to(self.device))
        t = torch.arange(T, device=self.device)
        idx = (torch.as_tensor(st, device=self.device)[:, None] + t[None]).clamp(max=T - 1)            # [B, T]
        valid = t[None] < torch.as_tensor(ln, device=self.device)[:, None]
        xc = torch.gather(x, 1, idx[:, :, None].expand(-1, -1, D)) * valid[:, :, None]
        return xc.transpose(1, 2).contiguous(), valid[:, None, :], segs, labels

    def loss(self, x, pair, seg, rng):
        g = win_to_grid(seg.detach().cpu().numpy(), self.steps).astype(np.float32)                     # [B, 2] grid units
        feats, masks, segs, labels = self._crop(x, pair, g)
        out = self.model(feats, masks, segs, labels, cls_channel=pair)
        return out["final_loss"], {k: float(v.detach()) for k, v in out.items()}

    def after_step(self):
        self.ema.update(self.model)

    @torch.no_grad()
    def predict(self, x, pair):
        m = self.ema.module; m.eval()
        feats = x.transpose(1, 2).contiguous()
        masks = torch.ones(feats.shape[0], 1, feats.shape[2], dtype=torch.bool, device=feats.device)
        seg, score, found = m.decode_top1(*m(feats, masks), cls_channel=pair)
        seg = seg.cpu().numpy()
        return finish(grid_to_win(seg[:, 0], self.steps), grid_to_win(seg[:, 1], self.steps), found.cpu().numpy())

    def state(self):
        return {"model": self.model.state_dict(), "ema": self.ema.module.state_dict(),
                "loss_normalizer": float(self.model.loss_normalizer), "ema_loss_normalizer": float(self.ema.module.loss_normalizer)}

    def load_state(self, s):
        self.model.load_state_dict(s["model"]); self.ema.module.load_state_dict(s["ema"])
        self.model.loss_normalizer = s.get("loss_normalizer", self.model.loss_normalizer)
        self.ema.module.loss_normalizer = s.get("ema_loss_normalizer", self.ema.module.loss_normalizer)

    @classmethod
    def _common_adaptations(cls, steps, r, official_levels=6):
        L = int(r["fpn_levels"])
        return [
            adapt("num_classes", cls.OFFICIAL["num_classes"], 6, "(a) one class per valid tool-action pair"),
            adapt("classification loss", "sigmoid focal loss on all classes", "sigmoid focal loss on the true pair's channel only",
                  "(a) the true pair is given; recognition errors are kept out of When"),
            adapt("max_seq_len", cls.OFFICIAL["max_seq_len"], steps, "(b) one window = T feature steps"),
            adapt("FPN levels (backbone_arch)", f"{official_levels} levels {cls.OFFICIAL['backbone_arch']}", f"{L} levels (2, 2, {L - 1})",
                  f"(b) T = {steps} supports strides up to {steps // 2}; a window spans {steps - 0.5:.1f} steps, so levels whose "
                  "regression range starts beyond it never get positives"),
            adapt("regression_range", cls.OFFICIAL["regression_range"], r["regression_range"],
                  "(b) the official ladder cut to the levels used, last range open-ended" + (" (x4 at 32 steps)" if steps == 32 else "")),
            batch_adapt(r, unit="videos"),
            adapt("schedule", f"{r['warmup_epochs']} warmup + {r['epochs']} cosine epochs (per-iteration schedule)",
                  "same epochs; iterations per epoch = training windows / batch", "(c) official epochs over our training windows"),
            adapt("training crop", "truncate_feats crop_ratio [0.9, 1.0], trunc_thresh 0.5", "unchanged (official); at T = 8 it drops "
                  "0 or 1 step", "kept (official)"),
            adapt("decoding", "per-class soft-NMS (no voting with multiclass_nms), top-k segments",
                  "top-1 segment of the true pair's channel after the official score/duration filters; whole window if none",
                  "(a) one interval per window from the true channel; soft-NMS never changes the top-1 segment"),
            adapt("input features", cls.OFFICIAL["features"], f"V-JEPA 2.1 ViT-L/384 (frozen, TactusNet), 1024-d, {steps} steps",
                  "the shared TACTUS features"),
        ]


def _af():
    from ..third_party import actionformer
    return actionformer


def _td():
    from ..third_party import tridet
    return tridet


class ActionFormer(_TAL):
    name, title = "actionformer", "ActionFormer"
    paper = "ActionFormer: Localizing Moments of Actions with Transformers, ECCV 2022 (Zhang, Wu, Li)"
    code_url = "https://github.com/happyharrycn/actionformer_release"
    official_source = "configs/thumos_i3d.yaml + libs/core/config.py @ 61ea7eb9"
    arch_name = "LocPointTransformer"
    package = staticmethod(_af)
    truncate_offset = 0.5          # feat_offset = 0.5 * num_frames / feat_stride with 2-frame steps
    OFFICIAL = dict(features="I3D (2048-d, stride 4, 16 frames)", num_classes=20, max_seq_len=2304, backbone_arch=(2, 2, 5),
                    n_mha_win_size=19, regression_range=[[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]], batch=2, lr=1e-4)
    DEFAULTS = dict(backbone_type="convTransformer", fpn_type="identity", fpn_levels=3, scale_factor=2, n_head=4, n_mha_win_size=-1,
                    embd_kernel_size=3, embd_dim=512, embd_with_ln=True, fpn_dim=512, fpn_with_ln=True, fpn_start_level=0, head_dim=512,
                    head_num_layers=3, head_kernel_size=3, head_with_ln=True, use_abs_pe=False, use_rel_pe=False,
                    max_buffer_len_factor=6.0, **TRAIN_CFG, **TEST_CFG, **CROP,
                    base_lr=1e-4, base_batch=2, batch=2, lr=None, weight_decay=0.05, eta_min=1e-8, epochs=30, warmup_epochs=5,
                    ema_decay=0.999)
    CONFIGS = {
        "main": {},                                                    # 3 levels, the official batch 2 and lr 1e-4
        "b64": {"batch": 64},                                          # batch 64, lr scaled linearly to 3.2e-3
        "l2": {"fpn_levels": 2},                                       # 2 levels, ranges [0,4],[4,inf) (optional)
    }
    SWEEP = ["main", "b64"]
    SMOKE = dict(epochs=1, warmup_epochs=1)
    conditioning = ("6 sigmoid classes (one per valid pair); focal loss only on the true pair's channel, the other channels "
                    "masked; class-agnostic offsets as in the original; test reads the true pair's channel")
    decoding = ("top-1 point of the true channel over all FPN levels after the official filters (score > 0.001, duration > 0.05 "
                "steps) = the official top-1 (per-class soft-NMS keeps it); else whole window")

    @classmethod
    def adaptations(cls, steps, r):
        return cls._common_adaptations(steps, r) + [
            adapt("n_mha_win_size", 19, "global (-1)", f"(b) the local window (19) is longer than the {steps}-step sequence"),
        ]

    def _cfg(self):
        r = self.recipe; L = int(r["fpn_levels"])
        return dict(backbone_type=r["backbone_type"], fpn_type=r["fpn_type"], backbone_arch=(2, 2, L - 1), scale_factor=r["scale_factor"],
                    input_dim=1024, max_seq_len=self.steps, max_buffer_len_factor=r["max_buffer_len_factor"], n_head=r["n_head"],
                    n_mha_win_size=r["n_mha_win_size"], embd_kernel_size=r["embd_kernel_size"], embd_dim=r["embd_dim"],
                    embd_with_ln=r["embd_with_ln"], fpn_dim=r["fpn_dim"], fpn_with_ln=r["fpn_with_ln"], fpn_start_level=r["fpn_start_level"],
                    head_dim=r["head_dim"], regression_range=r["regression_range"], head_num_layers=r["head_num_layers"],
                    head_kernel_size=r["head_kernel_size"], head_with_ln=r["head_with_ln"], use_abs_pe=r["use_abs_pe"],
                    use_rel_pe=r["use_rel_pe"], num_classes=self.n_pairs,
                    train_cfg={**{k: r[k] for k in TRAIN_CFG}, "head_empty_cls": []}, test_cfg={k: r[k] for k in TEST_CFG})


class TriDet(_TAL):
    name, title = "tridet", "TriDet"
    paper = "TriDet: Temporal Action Detection with Relative Boundary Modeling, CVPR 2023 (Shi et al.)"
    code_url = "https://github.com/dingfengshi/TriDet"
    official_source = "configs/thumos_i3d.yaml + libs/core/config.py @ 764b8165"
    arch_name = "TriDet"
    package = staticmethod(_td)
    truncate_offset = None
    OFFICIAL = dict(features="I3D (2048-d, stride 4, 16 frames)", num_classes=20, max_seq_len=2304, backbone_arch=(2, 2, 5),
                    regression_range=[[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]], num_bins=16, batch=2, lr=1e-4)
    DEFAULTS = dict(backbone_type="SGP", fpn_type="identity", fpn_levels=3, scale_factor=2, n_sgp_win_size=1, k=5, init_conv_vars=0,
                    sgp_mlp_dim=768, downsample_type="max", embd_kernel_size=3, embd_dim=512, embd_with_ln=True, fpn_dim=512,
                    fpn_with_ln=True, head_dim=512, head_num_layers=3, head_kernel_size=3, boudary_kernel_size=3, head_with_ln=True,
                    use_abs_pe=False, num_bins=None, iou_weight_power=0.2, input_noise=0.0005, use_trident_head=True,
                    max_buffer_len_factor=6.0, **TRAIN_CFG, **{**TEST_CFG, "max_seg_num": 2000}, **CROP,
                    base_lr=1e-4, base_batch=2, batch=2, lr=None, weight_decay=0.025, eta_min=1e-8, epochs=20, warmup_epochs=20,
                    ema_decay=0.999)
    CONFIGS = {
        "main": {},                                                    # 3 levels, 4 bins, the official batch 2 and lr 1e-4
        "b64": {"batch": 64},                                          # batch 64, lr scaled linearly to 3.2e-3
        "l2": {"fpn_levels": 2},                                       # 2 levels (optional)
    }
    SWEEP = ["main", "b64"]
    SMOKE = dict(epochs=1, warmup_epochs=1)
    conditioning = ("6 sigmoid classes (one per valid pair); focal loss only on the true pair's channel, the other channels "
                    "masked; the Trident start/end bins of the true channel; test reads the true pair's channel")
    decoding = ("top-1 point of the true channel over all FPN levels, offsets = expectation over the true channel's Trident "
                "bins, after the official filters (score > 0.001, duration > 0.05 steps) = the official top-1; else whole window")

    @classmethod
    def derived(cls, steps, r):
        d = super().derived(steps, r)
        if r.get("num_bins") is None: d["num_bins"] = 4 * (steps // 8)
        return d

    @classmethod
    def adaptations(cls, steps, r):
        return cls._common_adaptations(steps, r) + [
            adapt("num_bins (Trident head)", 16, r["num_bins"],
                  f"(b) boundary bins beyond the {steps}-step window would only see padding; 4 bins cover the [0,4] / [4,8] ranges "
                  "at strides 1 / 2" if steps == 8 else "(b) unchanged at 32 steps"),
        ]

    def _cfg(self):
        r = self.recipe; L = int(r["fpn_levels"])
        return dict(backbone_type=r["backbone_type"], fpn_type=r["fpn_type"], backbone_arch=(2, 2, L - 1), scale_factor=r["scale_factor"],
                    input_dim=1024, max_seq_len=self.steps, max_buffer_len_factor=r["max_buffer_len_factor"], n_sgp_win_size=r["n_sgp_win_size"],
                    embd_kernel_size=r["embd_kernel_size"], embd_dim=r["embd_dim"], embd_with_ln=r["embd_with_ln"], fpn_dim=r["fpn_dim"],
                    sgp_mlp_dim=r["sgp_mlp_dim"], fpn_with_ln=r["fpn_with_ln"], head_dim=r["head_dim"], regression_range=r["regression_range"],
                    head_num_layers=r["head_num_layers"], head_kernel_size=r["head_kernel_size"], boudary_kernel_size=r["boudary_kernel_size"],
                    head_with_ln=r["head_with_ln"], use_abs_pe=r["use_abs_pe"], num_bins=int(r["num_bins"]), iou_weight_power=r["iou_weight_power"],
                    downsample_type=r["downsample_type"], input_noise=r["input_noise"], k=r["k"], init_conv_vars=r["init_conv_vars"],
                    use_trident_head=r["use_trident_head"], num_classes=self.n_pairs,
                    train_cfg={**{k: r[k] for k in TRAIN_CFG}, "head_empty_cls": []}, test_cfg={k: r[k] for k in TEST_CFG})
