#!/usr/bin/env python3
"""DSTA ResNet-50 (He and Yang, CVPR 2024): video pose regression from neighbouring raw frames, on the full content frame.

  python -m baselines.kpt.dsta --out $BL_OUT/kpt_dsta [--smoke]

Official configuration: DSTA configs/posetimation/DSTA/posetrack17/configs-resnet50.yaml + Base_PoseTrack17.yaml +
posetimation/config/defaults.py (github.com/zgspose/DSTA, commit cb683c1e). Kept: DSTA_STD_ResNet50 (ResNet-50 per frame,
joint-centric feature decoder with RLE's normalized linear layer, 4 spatial + 4 temporal transformer blocks of width 32,
weighted-mean aggregation, coordinate and sigma heads, RealNVP flow with prior N(0.5, I)), its init_weights, the RLE loss
with target weights, 192x256 input pixel budget, targets = joints / input size, AdamW lr 5e-5 for batch 32 (one GPU),
betas (0.9, 0.999), weight decay 0.01, MultiStep x0.1 at epochs 10 and 15 of 20, no warmup, random scale 0.35 and
rotation 45 (p 0.6), ImageNet normalization.
Ported to plain PyTorch 2.x: third_party/dsta/dsta.py (notice of changes there).
TACTUS adaptations: 36 joints (tool and pair groups as the local-awareness groups) with the loss and readout on the true
group; 5 consecutive raw frames per key frame (+-2; the released config uses +-1); full frame turned landscape
(256 x 192); backbone from torchvision ImageNet ResNet-50 (the released config starts from a COCO RLE checkpoint); each
epoch draws 8 of the 16 protocol frames per clip as key frames; no flip, no half-body crop.
"""
import sys

import torch

from . import core, engine, weights as Wt
from .third_party.dsta.dsta import DSTA, rle_loss

TOOL_GROUPS = [[3 * t + k for k in range(core.N_TOOL_PTS)] for t in range(core.N_TOOLS)]
PAIR_GROUPS = [[12 + 4 * p + k for k in range(core.N_CT_PTS)] for p in range(core.N_PAIRS)]


class DSTASpec(engine.Spec):
    name, paper_row = "dsta", "DSTA"
    frames_desc = ("key frames: 8 of the 16 protocol frames per clip per epoch (all 16 at test); each brings the 5 consecutive raw "
                   "frames sampled_abs[t] + [-2..2] (ex.raw_frames, clamped at the clip ends)")
    input_wh, clips, channels_last = (256, 192), True, False
    jitter = {"scale_factor": 0.35, "rot_factor": 45.0, "rot_prob": 0.6}
    official = {"source": "DSTA configs/posetimation/DSTA/posetrack17/configs-resnet50.yaml (+ Base_PoseTrack17.yaml, defaults.py)",
                "input_wh": [192, 256], "batch": 32, "per_gpu": 32, "gpus": 1, "lr": 5e-5, "optimizer": "AdamW", "betas": [0.9, 0.999],
                "weight_decay": 0.01, "epochs": 20, "milestones": [10, 15], "gamma": 0.1, "warmup": "none", "rot_factor": 45,
                "scale_factor": 0.35, "flip": True, "half_body": [0.3, 8], "frames": "3 (key frame +- 1)", "bbox_enlarge": 1.25,
                "init": "COCO RLE ResNet-50 (coco-laplace-rle.pth)"}
    schedule = {"warmup_iters": 0, "milestones": (10 / 20, 15 / 20), "gamma": 0.1, "grad_clip": None}
    defaults = {"epochs": 20, "workers": 8, "eval_batch": 32, "val_every": 1}
    extra_args = [(("--kf-per-clip",), dict(type=int, default=8, help="key frames drawn per clip per epoch")),
                  (("--kf-per-item",), dict(type=int, default=4, help="key frames decoded together per data item"))]

    def build_model(self, args):
        m = DSTA(core.N_OUT, TOOL_GROUPS + PAIR_GROUPS, num_frame=5)
        m.init_weights(Wt.tv_resnet50_state())
        return m

    def param_groups(self, model, lr): return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    def optimizer(self, groups, lr): return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    def _sel(self, model, img, ch):
        coord, sigma = model(img); idx = ch[..., None].expand(-1, -1, 2)
        return coord.gather(1, idx), sigma.gather(1, idx)

    def loss(self, model, img, kp, vis, ch):
        coord, sigma = self._sel(model, img, ch)
        size = torch.tensor(self.input_wh, dtype=torch.float32, device=kp.device)
        tgt = kp / size                                                  # DSTA _generate_target: joints / image_size
        inside = (kp[..., 0] >= 0) & (kp[..., 0] <= size[0] - 1) & (kp[..., 1] >= 0) & (kp[..., 1] <= size[1] - 1)
        w = (vis * inside.float())[..., None].expand(-1, -1, 2)
        with torch.autocast(kp.device.type, enabled=False):              # the flow and the log-likelihood in fp32
            return rle_loss(coord, sigma, tgt, w, model.flow)

    def predict(self, model, img, ch):
        coord, _ = self._sel(model, img, ch)
        return (coord.float() * torch.tensor(self.input_wh, dtype=torch.float32, device=coord.device)).cpu().numpy()

    def recipe(self, args):
        return {"model": "DSTA_STD_ResNet50 ported to PyTorch 2.x (third_party/dsta), 36 joints, 5 frames",
                "backbone_init": {**Wt.describe("tv_resnet50"), "loaded": "ResNet-50 conv1..layer4 (fc dropped)",
                                  "new": "joint-centric decoder, spatial/temporal blocks, aggregation, heads, RealNVP flow (DSTA init_weights)"},
                "pretraining": "ImageNet-1k classification only (no pose data)",
                "local_groups": {"tool": TOOL_GROUPS, "pair": PAIR_GROUPS},
                "loss": "RLE (residual log-likelihood with RealNVP, Laplace residual term), target weight = visibility x inside-input x group",
                "optimizer_cfg": {"type": "AdamW", "betas": [0.9, 0.999], "weight_decay": 0.01},
                "key_frames": {"per_clip_per_epoch": args.kf_per_clip, "per_item": args.kf_per_item, "neighbors": [-2, -1, 0, 1, 2]},
                "augmentation": {**self.jitter, "flip": False, "half_body": False, "note": "the same random scale/rotation for the 5 frames"},
                "test_time": "single pass on the 5 frames, no flip; the sigma head is not used",
                "adaptations": ["36 joints (12 tool + 24 contact); local groups = 4 tool groups of 3 and 6 pair groups of 4; loss and readout on the true group",
                                "5 consecutive raw frames (+-2) instead of the released +-1", "full content frame, input 256x192 (official 192x256 turned landscape)",
                                "backbone: torchvision ImageNet ResNet-50 instead of the released COCO RLE checkpoint",
                                "8 of 16 key frames per clip per epoch, 20 epochs (official), milestones 10/15", "no flip / half-body", "bf16 autocast (loss in fp32)"]}


def main(argv=None):
    return engine.main(DSTASpec(), argv)


if __name__ == "__main__":
    main(); sys.exit(0)
