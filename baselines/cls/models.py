"""Models and OFFICIAL recipes of the classification baselines: each method keeps its
official configuration (architecture, input size, optimizer, LR schedule shape, augmentation, loss); only four things
change, listed per model in `adaptations` and copied to info.json:
  (a) the output: 6 pair classes (labels.json combo2id_6class), a new head trained from scratch (X-CLIP: 6 prompts);
  (b) the input: the 16 protocol frames of the clip, each the whole content crop squashed to a square; the official
      train augmentation runs on that square at the official pre-crop size, and the test view is the whole frame at
      the official input size (224) instead of the official short-side resize + center crop;
  (c) the schedule length: the official number of epochs and schedule shape, run on our 12,372 Gold train clips
      (so the number of iterations scales with our data size);
  (d) the batch: an effective batch of 64 clips per optimizer step (gradient accumulation on one GPU), the official
      LR scaled linearly from the official batch.
Backbones start from the authors' public weights. Which and What are read from the predicted pair by score.py.

Each wrapper takes x float [B,16,3,224,224] (normalized as the recipe says) and returns logits [B,6]; `param_groups`
returns optimizer groups with an `lr_scale` and their own weight decay.
"""
import types
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import augment as A
from . import weights as W

COMMON = ["(a) 6 pair outputs; new head from scratch", "(b) 16 protocol frames, whole content crop squashed to a square; "
          "train augmentation on the P x P square; test view = whole frame at 224 (no center crop)",
          "(c) official epochs and schedule shape on 12,372 clips", "(d) effective batch 64 by gradient accumulation, "
          "LR scaled linearly from the official batch", "bf16 autocast (official: fp32 / fp16 AMP)"]

# lr: official LR at the official batch lr_batch (scaled linearly to the effective batch); pre: pre-crop square size.
RECIPES = {
    "tsn": dict(
        name="TSN", paper="Wang et al., ECCV 2016; TSM repo implementation (TSN = TSM code without --shift)",
        backbone="ResNet-50, 16 segments, average consensus", input=224, pre=256, norm=A.IMAGENET, frames_official=8,
        init="ImageNet-1k ResNet-50 (torchvision IMAGENET1K_V1 = TSM repo --pretrain imagenet); the TSM repo has no "
             "Something-Something TSN checkpoint", pretrain_data="ImageNet-1k (supervised)",
        official="TSM README, training from ImageNet on Something-Something: main.py something RGB --arch resnet50 "
                 "--num_segments 8 --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 --batch-size 64 --dropout 0.5 "
                 "--consensus_type=avg --npb (without --shift = TSN); opts.py defaults: momentum 0.9, wd 5e-4",
        optimizer="sgd", momentum=0.9, nesterov=False, weight_decay=5e-4, lr=0.01, lr_batch=64,
        schedule=dict(kind="step", steps=(20, 40), gamma=0.1), epochs=50, clip_grad=20.0, dropout=0.5, partial_bn=False,
        loss="ce", label_smoothing=0.0, mix=None, num_sample=1, batch=8,
        augmentation="GroupMultiScaleCrop(224, [1, .875, .75, .66], 13 fixed offsets) + horizontal flip (the repo turns "
                     "flip off only for Something/Jester)",
        adaptations=COMMON[:1] + ["(b) 16 segments instead of 8"] + COMMON[1:]),
    "tsm": dict(
        name="TSM", paper="Lin, Gan, Han, ICCV 2019", backbone="ResNet-50 + temporal shift (shift_div 8, blockres), 16 "
        "segments, average consensus", input=224, pre=256, norm=A.IMAGENET, frames_official=16,
        init="TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth (fc re-initialized)",
        pretrain_data="ImageNet-1k, then Something-Something V2 (supervised)",
        official="TSM README fine-tuning example: main.py ucf101 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.001 "
                 "--lr_steps 10 20 --epochs 25 --batch-size 64 --dropout 0.8 --consensus_type=avg --shift --shift_div=8 "
                 "--shift_place=blockres --tune_from=<checkpoint> (partial BN, fc x5/x10 for a new dataset)",
        optimizer="sgd", momentum=0.9, nesterov=False, weight_decay=5e-4, lr=0.001, lr_batch=64,
        schedule=dict(kind="step", steps=(10, 20), gamma=0.1), epochs=25, clip_grad=20.0, dropout=0.8, partial_bn=True,
        loss="ce", label_smoothing=0.0, mix=None, num_sample=1, batch=8,
        augmentation="GroupMultiScaleCrop(224, [1, .875, .75, .66], 13 fixed offsets) + horizontal flip",
        adaptations=COMMON[:1] + ["(b) 16 segments (the SSv2 checkpoint's own count) instead of the example's 8"] + COMMON[1:]),
    "videomae": dict(
        name="VideoMAE", paper="Tong et al., NeurIPS 2022", backbone="ViT-B/16, tubelet 2, mean pooling", input=224,
        pre=256, norm=A.IMAGENET, frames_official=16, init="MCG-NJU/videomae-base-finetuned-ssv2 (classifier re-initialized)",
        pretrain_data="Something-Something V2 (self-supervised 2400 epochs, then supervised fine-tuning)",
        official="VideoMAE scripts/ssv2/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/finetune.sh "
                 "(--opt adamw --lr 5e-4 --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs 30 --num_sample 2) and "
                 "run_class_finetuning.py defaults (layer_decay 0.75, drop_path 0.1, warmup 5 epochs, warmup_lr and "
                 "min_lr 1e-6, smoothing 0.1, mixup 0.8, cutmix 1.0, switch 0.5, aa rand-m7-n4-mstd0.5-inc1, reprob "
                 "0.25, init_scale 0.001); LR = lr x batch / 256",
        optimizer="adamw", betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, lr=5e-4, lr_batch=256, layer_decay=0.75,
        schedule=dict(kind="videomae", warmup_epochs=5, warmup_lr=1e-6, min_lr=1e-6), epochs=30, clip_grad=None,
        drop_path=0.1, head_init_scale=0.001, loss="soft", label_smoothing=0.1, mix="timm", num_sample=2, batch=4,
        augmentation="RandAugment rand-m7-n4-mstd0.5-inc1 + random resized crop (0.08-1, 3/4-4/3) + flip + random "
                     "erasing 0.25; Mixup 0.8 / CutMix 1.0; repeated augmentation x2",
        adaptations=COMMON[:2] + ["(b) flip on: VideoMAE disables it only for SSV2 (left/right classes)"] + COMMON[2:]),
    "xclip": dict(
        name="X-CLIP", paper="Ni et al., ECCV 2022", backbone="CLIP ViT-B/16 + cross-frame communication, MIT, "
        "video-specific prompts", input=224, pre=256, norm=A.IMAGENET, frames_official=16,
        init="microsoft/xclip-base-patch16-16-frames", pretrain_data="CLIP WIT-400M image-text, then Kinetics-400 (supervised)",
        official="X-CLIP configs/k400/16_16.yaml (batch 4 x 8 GPUs x accumulation 8 = 256) with utils/config.py "
                 "defaults (30 epochs, warmup 5, wd 0.001, lr 8e-6, smoothing 0.1, mixup 0.8, cutmix 1.0, switch 0.5, "
                 "color jitter 0.8, gray 0.2, FIX_TEXT True, drop path 0), utils/optimizer.py (AdamW 0.9/0.98, eps 1e-8, "
                 "10x LR for prompts/mit/message_, cosine to lr/100), datasets/build.py train pipeline",
        optimizer="adamw", betas=(0.9, 0.98), eps=1e-8, weight_decay=0.001, lr=8e-6, lr_batch=256, new_module_lr_mult=10.0,
        schedule=dict(kind="timm_cosine", warmup_epochs=5, min_ratio=0.01), epochs=30, clip_grad=None, drop_path=0.0,
        fix_text=True, loss="soft", label_smoothing=0.1, mix="xclip", num_sample=1, batch=8,
        augmentation="MultiScaleCrop(224, (1, .875, .75, .66), 5 fixed crops) + flip + ColorJitter p 0.8 (per frame) + "
                     "GrayScale p 0.2; Mixup 0.8 / CutMix 1.0",
        adaptations=["(a) 6 classes = 6 fixed prompts, one per pair ('a video of a hook dissecting tissue', ...); the text "
                     "side stays frozen as in the official FIX_TEXT"] + COMMON[1:]),
    "endofm": dict(
        name="Endo-FM", paper="Wang et al., MICCAI 2023", backbone="ViT-B/16 TimeSformer (divided space-time attention)",
        input=224, pre=320, norm=A.KINETICS, frames_official=8,
        init="Endo-FM pretrained checkpoint, teacher backbone (Google Drive link of the Endo-FM README)",
        pretrain_data="endoscopy videos (Endo-FM's 33K clips incl. CholecTriplet, self-supervised; initialized from SVT "
                      "Kinetics-400 self-supervised)",
        official="Endo-FM scripts/eval_finetune_polypdiag.sh (--lr 0.001 --epochs 20 --batch_size_per_gpu 4, 1 GPU) and "
                 "eval_finetune.py (SGD momentum 0.9, no weight decay, lr x batch / 256, CosineAnnealingLR per epoch to 0, "
                 "cross-entropy, linear head on the last CLS token), config TimeSformer_divST_8x32_224.yaml (8 frames, "
                 "224, drop path 0.1), datasets/ucf101.py train sampling (short side 256-320, random crop 224, flip; "
                 "mean 0.45, std 0.225)",
        optimizer="sgd", momentum=0.9, nesterov=False, weight_decay=0.0, lr=0.001, lr_batch=256,
        schedule=dict(kind="cosine_epoch"), epochs=20, clip_grad=None, drop_path=0.1, loss="ce", label_smoothing=0.0,
        mix=None, num_sample=1, batch=4,
        augmentation="short side jitter U[256, 320] + random crop 224 + flip",
        adaptations=COMMON[:1] + ["(b) 16 frames instead of the official 8: Endo-FM's own forward nearest-interpolates "
                                  "its 8 time embeddings to 16, as for its 16-frame global views in pretraining"] + COMMON[1:]),
    "mvitv2": dict(
        name="MViTv2-S", paper="Li et al., CVPR 2022", backbone="MViTv2-S (16x4)", input=224, pre=320, norm=A.KINETICS,
        frames_official=16, init="torchvision MViT_V2_S_Weights.KINETICS400_V1 (head re-initialized)",
        pretrain_data="Kinetics-400 (supervised)",
        official="PySlowFast configs/SSv2/MVITv2_S_16x4.yaml, the official fine-tuning of the K400 model (SGD momentum "
                 "0.9, Nesterov (default), wd 1e-4 without 1-D/bias, BASE_LR 0.0025 at batch 16, warmup 3 epochs from "
                 "1e-4, cosine after warmup to 1e-6, 100 epochs, drop path 0.4, dropout 0.5, RandAugment "
                 "rand-m7-n4-mstd0.5-inc1, random erasing 0.25, mixup 0.8, cutmix 1.0, smoothing 0.1, soft "
                 "cross-entropy, short side 256-320 inverse-uniform + crop 224)",
        optimizer="sgd", momentum=0.9, nesterov=True, weight_decay=1e-4, lr=0.0025, lr_batch=16,
        schedule=dict(kind="slowfast_cosine", warmup_epochs=3, warmup_start=1e-4, end=1e-6), epochs=100, clip_grad=None,
        drop_path=0.4, loss="soft", label_smoothing=0.1, mix="timm", num_sample=1, batch=8,
        augmentation="RandAugment rand-m7-n4-mstd0.5-inc1 + short side jitter 1/U[1/320, 1/256] + random crop 224 + "
                     "flip + random erasing 0.25; Mixup 0.8 / CutMix 1.0",
        adaptations=COMMON[:2] + ["(b) flip on: the SSv2 config turns RANDOM_FLIP off only for SSv2's left/right classes"]
        + COMMON[2:]),
}

# One fixed prompt per pair for X-CLIP (keys = labels.json pair names).
PROMPTS = {
    "hook|dissect": "a video of a hook dissecting tissue",
    "grasper|grasp": "a video of a grasper grasping tissue",
    "clipper|clip": "a video of a clipper applying a clip",
    "scissors|cut": "a video of scissors cutting tissue",
    "scissors|dissect": "a video of scissors dissecting tissue",
    "grasper|dissect": "a video of a grasper dissecting tissue",
}


def _no_wd(n, p):
    return p.ndim <= 1 or n.endswith(".bias")


def _groups(model, layer_of, n_layers, decay, no_wd=_no_wd, wd=0.0):
    """lr_scale = decay ** (n_layers - layer_of(name)) (BEiT/VideoMAE layer decay; decay 1 = one LR)."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lid = layer_of(n); w = 0.0 if no_wd(n, p) else wd
        g = groups.setdefault((lid, w), {"params": [], "lr_scale": decay ** (n_layers - lid), "weight_decay": w, "name": f"layer{lid}"})
        g["params"].append(p)
    return list(groups.values())


# ---------------------------------------------------------------- TSN / TSM (ResNet-50, 16 segments, logit averaging)
class TSN(nn.Module):
    """TSM repo's TSN class: torchvision ResNet-50 (temporal shift on every residual branch when shift=True), fc
    replaced by dropout + a new fc, per-frame logits averaged over the 16 segments (consensus avg). partial_bn: like
    TSN.train(), every BatchNorm but the first stays in eval mode with frozen affine parameters."""

    def __init__(self, n_cls, shift, n_segment=16, shift_div=8, dropout=0.5, partial_bn=False):
        super().__init__()
        import torchvision
        base = torchvision.models.resnet50(weights=None)
        if shift:
            from .vendor.tsm.temporal_shift import make_temporal_shift
            make_temporal_shift(base, n_segment, n_div=shift_div, place="blockres")
        base.fc = nn.Dropout(p=dropout); base.avgpool = nn.AdaptiveAvgPool2d(1)
        self.base_model, self.n_segment, self.partial_bn = base, n_segment, partial_bn
        self.new_fc = nn.Linear(2048, n_cls); nn.init.normal_(self.new_fc.weight, 0, 0.001); nn.init.constant_(self.new_fc.bias, 0)
        self._frozen_bn = [m for m in self.base_model.modules() if isinstance(m, nn.BatchNorm2d)][1:] if partial_bn else []
        for m in self._frozen_bn: m.weight.requires_grad = False; m.bias.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for m in self._frozen_bn: m.eval()
        return self

    def forward(self, x):
        B, T = x.shape[:2]; assert T == self.n_segment, (T, self.n_segment)
        y = self.base_model(x.flatten(0, 1).contiguous(memory_format=torch.channels_last))
        return self.new_fc(y).view(B, T, -1).mean(1)

    def param_groups(self, r):
        """TSN.get_optim_policies: conv weights (lr 1, wd 1), BN (lr 1, wd 0; first BN only with partial BN), new fc
        weight (lr 5, wd 1) and bias (lr 10, wd 0) when tuning to a new dataset (fc_lr5)."""
        wd = r["weight_decay"]; g = {"conv": [], "bn": [], "fc_w": [], "fc_b": []}
        for m in self.modules():
            if isinstance(m, nn.Conv2d): g["conv"] += [p for p in m.parameters() if p.requires_grad]
            elif isinstance(m, nn.BatchNorm2d): g["bn"] += [p for p in m.parameters() if p.requires_grad]
            elif isinstance(m, nn.Linear): g["fc_w"].append(m.weight); g["fc_b"].append(m.bias)
            elif len(m._modules) == 0 and len(list(m.parameters(recurse=False))) > 0: raise ValueError(f"no policy for {type(m)}")
        return [dict(params=g["conv"], lr_scale=1.0, weight_decay=wd, name="conv"),
                dict(params=g["bn"], lr_scale=1.0, weight_decay=0.0, name="bn"),
                dict(params=g["fc_w"], lr_scale=5.0, weight_decay=wd, name="fc_weight_lr5"),
                dict(params=g["fc_b"], lr_scale=10.0, weight_decay=0.0, name="fc_bias_lr10")]


def build_tsn(r, n_cls, shift, **_):
    m = TSN(n_cls, shift=shift, dropout=r["dropout"], partial_bn=r["partial_bn"])
    if shift:   # main.py --tune_from: strip "module.", drop every "fc" key (new dataset), keep the rest
        sd = torch.load(W.path("tsm"), map_location="cpu", weights_only=True)["state_dict"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if "fc" not in k}
    else:       # TSN._prepare_base_model: torchvision resnet50(pretrained=True)
        sd = {f"base_model.{k}": v for k, v in torch.load(W.path("tsn"), map_location="cpu", weights_only=True).items()
              if not k.startswith("fc.")}
    res = m.load_state_dict(sd, strict=False)
    if res.unexpected_keys or set(res.missing_keys) != {"new_fc.weight", "new_fc.bias"}:
        raise RuntimeError(f"TSN/TSM init mismatch: missing {res.missing_keys[:8]}, unexpected {res.unexpected_keys[:8]}")
    return m.to(memory_format=torch.channels_last)


# ---------------------------------------------------------------- VideoMAE (Hugging Face port)
class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1.0 - self.p; mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self): return f"p={self.p}"


def _videomae_layer_forward(self, hidden_states):
    """VideoMAELayer.forward with stochastic depth on both residual branches (VideoMAE's Block; HF has no drop path)."""
    h = hidden_states + self.drop_path(self.attention(self.layernorm_before(hidden_states)))
    out = self.output.dropout(self.output.dense(self.intermediate(self.layernorm_after(h))))
    return h + self.drop_path(out)


class VideoMAE(nn.Module):
    def __init__(self, n_cls, drop_path=0.1, head_init_scale=0.001, grad_ckpt=False):
        super().__init__()
        from transformers import VideoMAEForVideoClassification
        id2label = {i: str(i) for i in range(n_cls)}
        self.m = VideoMAEForVideoClassification.from_pretrained(
            str(W.path("videomae")), num_labels=n_cls, id2label=id2label, label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True, attn_implementation="sdpa", dtype=torch.float32)
        layers = self.m.videomae.encoder.layer
        for layer, p in zip(layers, torch.linspace(0, drop_path, len(layers)).tolist()):
            layer.drop_path = DropPath(p); layer.forward = types.MethodType(_videomae_layer_forward, layer)
        c = self.m.classifier                      # VideoMAE: trunc_normal_(std=.02) then weight *= init_scale, bias 0
        nn.init.trunc_normal_(c.weight, std=0.02); c.weight.data.mul_(head_init_scale); nn.init.zeros_(c.bias)
        if grad_ckpt: self.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.n_blocks = len(layers)

    def forward(self, x):
        return self.m(pixel_values=x).logits

    def param_groups(self, r):
        """optim_factory.get_parameter_groups with LayerDecayValueAssigner: embeddings 0, block i -> i + 1, rest 13;
        no weight decay for 1-D parameters and biases."""
        nb = self.n_blocks
        def layer_of(n):
            if ".embeddings." in n: return 0
            if ".encoder.layer." in n: return int(n.split(".encoder.layer.")[1].split(".")[0]) + 1
            return nb + 1
        return _groups(self, layer_of, nb + 1, r["layer_decay"], wd=r["weight_decay"])


# ---------------------------------------------------------------- X-CLIP (Hugging Face port)
class XCLIP(nn.Module):
    def __init__(self, n_cls, pairs, fix_text=True, grad_ckpt=False):
        super().__init__()
        from transformers import AutoTokenizer, XCLIPModel
        path = str(W.path("xclip"))
        # transformers 5.1 declares no SDPA support for X-CLIP (_supports_sdpa = False): eager attention, as released
        self.m = XCLIPModel.from_pretrained(path, dtype=torch.float32); self.attn = "eager"
        self.prompts = [PROMPTS[p] for p in pairs]; assert len(self.prompts) == n_cls
        t = AutoTokenizer.from_pretrained(path)(self.prompts, padding=True, return_tensors="pt")
        self.register_buffer("input_ids", t["input_ids"], persistent=False)
        self.register_buffer("attention_mask", t["attention_mask"], persistent=False)
        self.fix_text = fix_text
        if fix_text:   # X-CLIP utils/optimizer.py fix_text(): only visual., mit and prompts* parameters are trained
            for n, p in self.m.named_parameters():
                if not (n.startswith(("vision_model.", "visual_projection.")) or n.split(".")[0] == "mit" or "prompts" in n):
                    p.requires_grad = False
        if grad_ckpt: self.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, x):
        return self.m(input_ids=self.input_ids, attention_mask=self.attention_mask, pixel_values=x).logits_per_video

    def param_groups(self, r):
        """X-CLIP build_optimizer: names with prompts / mit / message_ at 10x LR; set_weight_decay: no WD on 1-D, bias and
        position embeddings (no_weight_decay_keywords = positional_embedding)."""
        out = {}
        for n, p in self.m.named_parameters():
            if not p.requires_grad: continue
            new = any(k in n for k in ("prompts", "message_")) or n.split(".")[0] == "mit"
            nwd = p.ndim <= 1 or n.endswith(".bias") or "position_embedding" in n
            g = out.setdefault((new, nwd), dict(params=[], lr_scale=r["new_module_lr_mult"] if new else 1.0,
                                                weight_decay=0.0 if nwd else r["weight_decay"],
                                                name=("new_x10" if new else "clip") + ("_nowd" if nwd else "")))
            g["params"].append(p)
        return list(out.values())


# ---------------------------------------------------------------- Endo-FM (vendored TimeSformer)
def _sdpa_attention_forward(self, x, return_attn=False):
    """Endo-FM Attention.forward with F.scaled_dot_product_attention (same math, less memory)."""
    if return_attn or not self.with_qkv: return type(self).forward(self, x, return_attn)
    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


class EndoFM(nn.Module):
    """get_vit_base_patch16_224(cfg, no_head=True) of Endo-FM (TimeSformer_divST_8x32_224: 8 time embeddings, 224,
    drop path 0.1) + eval_finetune.py's LinearClassifier on the final CLS token (N(0, 0.01) init)."""

    def __init__(self, n_cls, drop_path=0.1, grad_ckpt=False):
        super().__init__()
        from .vendor.endofm.timesformer import Attention, VisionTransformer
        self.vit = VisionTransformer(img_size=224, num_classes=0, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                                     attn_drop_rate=0., drop_path_rate=drop_path, num_frames=8,
                                     attention_type="divided_space_time")
        self.vit.head = None
        sd = torch.load(W.endofm_backbone(), map_location="cpu", weights_only=True)   # teacher backbone, "backbone." stripped
        res = self.vit.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"Endo-FM init mismatch: missing {res.missing_keys[:8]}, unexpected {res.unexpected_keys[:8]}")
        self.time_embed_len = int(self.vit.time_embed.shape[1])
        for mod in self.vit.modules():
            if isinstance(mod, Attention): mod.forward = types.MethodType(_sdpa_attention_forward, mod)
        if grad_ckpt:
            from torch.utils.checkpoint import checkpoint
            for blk in self.vit.blocks:
                f = blk.forward
                blk.forward = (lambda f: lambda x, B, T, W_: checkpoint(f, x, B, T, W_, use_reentrant=False)
                               if torch.is_grad_enabled() else f(x, B, T, W_))(f)
        self.head = nn.Linear(768, n_cls); self.head.weight.data.normal_(0.0, 0.01); self.head.bias.data.zero_()

    def forward(self, x):
        return self.head(self.vit.forward_features(x.permute(0, 2, 1, 3, 4)))   # [B,3,T,H,W] -> CLS [B,768]

    def param_groups(self, r):
        """eval_finetune.py: one SGD group for the backbone and one for the classifier, same LR, weight decay 0."""
        return [dict(params=[p for p in self.vit.parameters() if p.requires_grad], lr_scale=1.0, weight_decay=0.0, name="backbone"),
                dict(params=list(self.head.parameters()), lr_scale=1.0, weight_decay=0.0, name="linear_classifier")]


# ---------------------------------------------------------------- MViTv2-S (torchvision)
class MViT(nn.Module):
    def __init__(self, n_cls, drop_path=0.4):
        super().__init__()
        from torchvision.models.video import mvit_v2_s
        self.m = mvit_v2_s(weights=None, stochastic_depth_prob=drop_path)        # head: Dropout(0.5) + Linear
        self.m.load_state_dict(torch.load(W.path("mvitv2"), map_location="cpu", weights_only=True))
        lin = nn.Linear(self.m.head[1].in_features, n_cls); nn.init.trunc_normal_(lin.weight, std=0.02); nn.init.zeros_(lin.bias)
        self.m.head[1] = lin

    def forward(self, x):
        return self.m(x.permute(0, 2, 1, 3, 4))   # [B,3,16,224,224]

    def param_groups(self, r):
        """PySlowFast optimizer.get_param_groups with ZERO_WD_1D_PARAM (1-D and bias: no WD), LAYER_DECAY 1,
        ZERO_DECAY_POS_CLS False (so position and class tokens keep their weight decay)."""
        return _groups(self, lambda n: 0, 0, 1.0, wd=r["weight_decay"])


def build(name, n_cls, pairs, grad_ckpt=False):
    r = RECIPES[name]
    if name in ("tsn", "tsm"): return build_tsn(r, n_cls, shift=name == "tsm")
    if name == "videomae": return VideoMAE(n_cls, r["drop_path"], r["head_init_scale"], grad_ckpt)
    if name == "xclip": return XCLIP(n_cls, pairs, r["fix_text"], grad_ckpt)
    if name == "endofm": return EndoFM(n_cls, r["drop_path"], grad_ckpt)
    if name == "mvitv2": return MViT(n_cls, r["drop_path"])
    raise KeyError(name)
