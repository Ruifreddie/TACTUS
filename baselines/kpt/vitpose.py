#!/usr/bin/env python3
"""ViTPose-B (Xu et al., NeurIPS 2022) on the full content frame.

  python -m baselines.kpt.vitpose --out $BL_OUT/kpt_vitpose [--smoke]

Official configuration: ViTPose configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py
(github.com/ViTAE-Transformer/ViTPose). Kept: plain ViT-B/16 (patch padding 2, qkv bias, LayerNorm eps 1e-6, drop path
0.3) + classic decoder (2 x deconv 4x4/2 with 256 channels, BN, ReLU, then 1x1 conv), backbone initialized from the MAE
ImageNet-1k release with the position embeddings interpolated bicubically (ViTPose's load_checkpoint; MAE's final norm and
cls token are not loaded, as in ViTPose), decoder from scratch (deconv N(0, 0.001), BN (1, 0), final conv N(0, 0.001));
192x256 input pixel budget, 48x64 heatmaps, UDP targets sigma 2, JointsMSELoss with target weights; AdamW lr 5e-4 for the
total batch 8 x 64 = 512, betas (0.9, 0.999), weight decay 0.1 (not on 1-D, bias and position-embedding parameters),
layer-wise lr decay 0.75 over 12 layers, grad clip 1.0; linear warmup 500 iterations from 0.001; x0.1 at epochs 170 and
200 of 210; random scale 0.5 and rotation 40 (p 0.6); UDP + DARK-UDP decoding (kernel 11).
The model is Hugging Face transformers' VitPoseForPoseEstimation (the same ViTPose-B architecture; its port of the
official vitpose-b.pth reproduces the official outputs), built from config with no pose weights, plus drop path, which
the inference-only port lacks.
TACTUS adaptations (engine.py): 36 outputs with the loss and readout on the true group; full frame turned landscape
(256 x 192, a 16 x 12 patch grid); about-1-fps training frames; total batch 512 = micro-batch 64 x 8 accumulation steps;
no flip, no half-body crop, no flip test.
"""
import math, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core, engine, weights as Wt
from .codecs import UDPCodec, joints_mse, select
from .third_party.dsta.dsta import DropPath

NUM_LAYERS = 12


def _layer_with_drop_path():
    from transformers.models.vitpose_backbone.modeling_vitpose_backbone import VitPoseBackboneLayer

    class VitPoseLayerDP(VitPoseBackboneLayer):
        """ViTPose Block: x = x + drop_path(attn(norm1(x))); x = x + drop_path(mlp(norm2(x)))."""
        def forward(self, hidden_states, dataset_index=None):
            h = hidden_states + self.drop_path(self.attention(self.layernorm_before(hidden_states)))
            return h + self.drop_path(self.mlp(self.layernorm_after(h)))
    return VitPoseLayerDP


def mae_to_vitpose(sd, grid_hw, dim=768, depth=NUM_LAYERS):
    """Official MAE ViT-B keys -> transformers VitPose backbone keys; pos_embed interpolated (bicubic, align_corners
    False) from 14x14 to grid_hw as ViTPose's mmcv_custom/checkpoint.py; MAE's cls_token and final norm are left out."""
    new = {"backbone.embeddings.patch_embeddings.projection.weight": sd["patch_embed.proj.weight"],
           "backbone.embeddings.patch_embeddings.projection.bias": sd["patch_embed.proj.bias"]}
    pe = sd["pos_embed"]; n = int(round(math.sqrt(pe.shape[1] - 1)))
    grid = pe[:, 1:].reshape(1, n, n, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=tuple(grid_hw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).flatten(1, 2)
    new["backbone.embeddings.position_embeddings"] = torch.cat([pe[:, :1], grid], 1)
    for i in range(depth):
        p, q = f"blocks.{i}.", f"backbone.encoder.layer.{i}."
        for a, b in (("norm1", "layernorm_before"), ("norm2", "layernorm_after"), ("attn.proj", "attention.output.dense"),
                     ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            new[q + b + ".weight"] = sd[p + a + ".weight"]; new[q + b + ".bias"] = sd[p + a + ".bias"]
        w, bias = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]
        for j, nm in enumerate(("query", "key", "value")):
            new[q + f"attention.attention.{nm}.weight"] = w[j * dim:(j + 1) * dim]
            new[q + f"attention.attention.{nm}.bias"] = bias[j * dim:(j + 1) * dim]
    return new


def build_vitpose_b(num_out, input_wh, drop_path_rate=0.3, mae=True):
    from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation
    W, H = input_wh
    bb = VitPoseBackboneConfig(image_size=[H, W], patch_size=[16, 16], num_channels=3, hidden_size=768, num_hidden_layers=NUM_LAYERS,
                               num_attention_heads=12, mlp_ratio=4, qkv_bias=True, layer_norm_eps=1e-6, hidden_act="gelu",
                               out_indices=[NUM_LAYERS])
    cfg = VitPoseConfig(backbone_config=bb, use_simple_decoder=False, scale_factor=4, num_labels=num_out)
    m = VitPoseForPoseEstimation(cfg)
    m.backbone.config._attn_implementation = "sdpa"
    LayerDP = _layer_with_drop_path(); dpr = [x.item() for x in torch.linspace(0, drop_path_rate, NUM_LAYERS)]
    for i, layer in enumerate(m.backbone.encoder.layer):
        layer.__class__ = LayerDP; layer.add_module("drop_path", DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity())
    hd = m.head                                                   # mmpose 0.x TopdownHeatmapSimpleHead.init_weights
    for d in (hd.deconv1, hd.deconv2): nn.init.normal_(d.weight, std=0.001)
    for bn in (hd.batchnorm1, hd.batchnorm2): nn.init.ones_(bn.weight); nn.init.zeros_(bn.bias)
    nn.init.normal_(hd.conv.weight, std=0.001); nn.init.zeros_(hd.conv.bias)
    report = {}
    if mae:
        grid = (H // 16, W // 16); new = mae_to_vitpose(Wt.load_state("mae_vit_base"), grid)
        r = m.load_state_dict(new, strict=False)
        rest = sorted(k for k in r.missing_keys if not k.startswith("head."))
        if r.unexpected_keys or rest != ["backbone.layernorm.bias", "backbone.layernorm.weight"]:
            raise RuntimeError(f"MAE load: unexpected {r.unexpected_keys[:5]}, missing {rest[:5]}")
        report = {"loaded": len(new), "pos_embed": f"14x14 -> {grid[0]}x{grid[1]} (bicubic)", "not_loaded": ["cls_token", "norm (ViTPose last_norm starts at 1, 0)"]}
    return m, report


def layer_id(name):
    """ViTPose get_num_layer_for_vit with num_layers = 12 + 2: embeddings 0, block i -> i + 1, the rest 13."""
    if name.startswith("backbone.embeddings."): return 0
    if name.startswith("backbone.encoder.layer."): return int(name.split(".")[3]) + 1
    return NUM_LAYERS + 1


class ViTPoseSpec(engine.Spec):
    name, paper_row = "vitpose", "ViTPose-B"
    frames_desc = "train: about-1-fps protocol frames (ex.frames_1fps); test: all 16 protocol frames, each on its own"
    input_wh, channels_last = (256, 192), False
    jitter = {"scale_factor": 0.5, "rot_factor": 40.0, "rot_prob": 0.6}
    official = {"source": "ViTPose configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py", "input_wh": [192, 256],
                "heatmap_wh": [48, 64], "sigma": 2, "batch": 512, "per_gpu": 64, "gpus": 8, "lr": 5e-4, "optimizer": "AdamW",
                "betas": [0.9, 0.999], "weight_decay": 0.1, "layer_decay": 0.75, "drop_path": 0.3, "grad_clip": 1.0, "epochs": 210,
                "lr_step": [170, 200], "warmup": "linear, 500 iters, ratio 0.001", "rot_factor": 40, "scale_factor": 0.5, "flip": True,
                "half_body": [0.3, 8], "flip_test": True, "decode": "UDP + DARK-UDP, kernel 11", "init": "MAE ViT-B mae_pretrain_vit_base.pth"}
    schedule = {"warmup_iters": 500, "warmup_ratio": 0.001, "milestones": (170 / 210, 200 / 210), "gamma": 0.1, "grad_clip": 1.0}
    defaults = {"epochs": 210, "workers": 8, "eval_batch": 128, "val_every": 10}
    WEIGHT_DECAY, LAYER_DECAY = 0.1, 0.75

    def build_model(self, args):
        m, self.init_report = build_vitpose_b(core.N_OUT, self.input_wh)
        W, H = self.input_wh; self.codec = UDPCodec(self.input_wh, (W // 4, H // 4), sigma=2, kernel=11)
        return m

    def param_groups(self, model, lr):
        groups = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            nd = p.ndim == 1 or n.endswith(".bias") or "position_embeddings" in n; lid = layer_id(n)
            g = groups.setdefault((lid, nd), {"params": [], "weight_decay": 0.0 if nd else self.WEIGHT_DECAY,
                                              "lr": lr * self.LAYER_DECAY ** (NUM_LAYERS + 1 - lid), "layer": lid})
            g["params"].append(p)
        return list(groups.values())

    def optimizer(self, groups, lr): return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=self.WEIGHT_DECAY)

    def loss(self, model, img, kp, vis, ch):
        tgt, w = self.codec.encode(kp, vis); return joints_mse(model(img).heatmaps, ch, tgt, w, core.N_OUT, half=False)

    def predict(self, model, img, ch): return self.codec.decode(select(model(img).heatmaps, ch))[0]

    def recipe(self, args):
        return {"model": "ViTPose-B: transformers VitPoseForPoseEstimation (classic decoder) built from config, 36 output channels, drop path 0.3 added",
                "backbone_init": {**Wt.describe("mae_vit_base"), **self.init_report},
                "pretraining": "MAE self-supervised on ImageNet-1k only (no labels, no pose data)", **self.codec.describe(),
                "loss": "JointsMSELoss(use_target_weight): mean over the 36 outputs of MSE(pred x w, gt x w)",
                "optimizer_cfg": {"type": "AdamW", "betas": [0.9, 0.999], "weight_decay": self.WEIGHT_DECAY, "layer_decay": self.LAYER_DECAY,
                                  "no_weight_decay": "1-D params, biases, position embeddings", "grad_clip": 1.0},
                "augmentation": {**self.jitter, "flip": False, "half_body": False, "note": "random scale/rotation of the full frame"},
                "test_time": "single full frame, no flip test, UDP + DARK-UDP decoding",
                "adaptations": ["36 outputs (12 tool + 24 contact), loss and readout on the true group",
                                "full content frame, input 256x192 (official 192x256 turned landscape), 16x12 patch grid, heatmap 64x48",
                                "train frames: ex.frames_1fps; test: all 16 protocol frames", "total batch 512 = 64 x 8 accumulation steps on one GPU",
                                "no flip / half-body / flip test", "bf16 autocast, SDPA attention"]}


def main(argv=None):
    return engine.main(ViTPoseSpec(), argv)


if __name__ == "__main__":
    main(); sys.exit(0)
