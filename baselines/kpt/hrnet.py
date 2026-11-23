#!/usr/bin/env python3
"""HRNet-W32 + DARK (Sun et al., CVPR 2019; Zhang et al., CVPR 2020) on the full content frame.

  python -m baselines.kpt.hrnet --out $BL_OUT/kpt_hrnet [--smoke]

Official configuration: DarkPose experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml (github.com/ilovepose/DarkPose,
commit 0185b427), the DARK release of the official HRNet code. Kept: pose HRNet-W32 (official code, vendored), ImageNet
init hrnet_w32-36af842e.pth with the official PRETRAINED_LAYERS (the head is new, N(0, 0.001)), 288x384 input pixel
budget, heatmaps at 1/4, sigma 3, unbiased Gaussian targets, JointsMSELoss with target weights, Adam lr 1e-3 for the total
batch 4 x 28 = 112, no weight decay, MultiStep x0.1 at epochs 170 and 200 of 210, random scale 0.35 and rotation 45
(p 0.6), DARK decoding with blur kernel 11, ImageNet normalization.
TACTUS adaptations (engine.py): 36 outputs with the loss and readout on the true group; full frame turned landscape
(384 x 288); about-1-fps training frames; total batch 112 = micro-batch 28 x 4 accumulation steps; no flip, no half-body
crop, no flip test.
"""
import sys

import torch

from . import core, engine, weights as Wt
from .codecs import DarkCodec, joints_mse, select
from .third_party.hrnet.pose_hrnet import PoseHighResolutionNet

PRETRAINED_LAYERS = ["conv1", "bn1", "conv2", "bn2", "layer1", "transition1", "stage2", "transition2", "stage3", "transition3", "stage4"]


def hrnet_w32_cfg(num_joints):
    st = lambda nb, nc, nm: {"NUM_MODULES": nm, "NUM_BRANCHES": len(nb), "BLOCK": "BASIC", "NUM_BLOCKS": nb,
                             "NUM_CHANNELS": nc, "FUSE_METHOD": "SUM"}
    return {"MODEL": {"NUM_JOINTS": num_joints, "EXTRA": {
        "PRETRAINED_LAYERS": PRETRAINED_LAYERS, "FINAL_CONV_KERNEL": 1, "STAGE2": st([4, 4], [32, 64], 1),
        "STAGE3": st([4, 4, 4], [32, 64, 128], 4), "STAGE4": st([4, 4, 4, 4], [32, 64, 128, 256], 3)}}}


def build_hrnet_w32(num_joints, imagenet=True):
    """Official pose HRNet-W32; init_weights() (conv N(0, 0.001), BN (1, 0)), then the ImageNet file filtered by the
    official PRETRAINED_LAYERS, as PoseHighResolutionNet.init_weights(pretrained) does."""
    m = PoseHighResolutionNet(hrnet_w32_cfg(num_joints)); m.init_weights()
    loaded = []
    if imagenet:
        sd = Wt.load_state("hrnet_w32_imagenet"); own = m.state_dict()
        keep = {k: v for k, v in sd.items() if k.split(".")[0] in PRETRAINED_LAYERS and k in own}
        bad = [k for k, v in keep.items() if v.shape != own[k].shape]
        if bad: raise RuntimeError(f"HRNet ImageNet shapes differ: {bad[:5]}")
        r = m.load_state_dict(keep, strict=False)
        if set(r.missing_keys) != {"final_layer.weight", "final_layer.bias"}:
            raise RuntimeError(f"HRNet ImageNet load left out {sorted(set(r.missing_keys) - {'final_layer.weight', 'final_layer.bias'})[:5]}")
        loaded = sorted({k.split(".")[0] for k in keep})
    return m, loaded


class HRNetSpec(engine.Spec):
    name, paper_row = "hrnet", "HRNet-W32 + DARK"
    frames_desc = "train: about-1-fps protocol frames (ex.frames_1fps); test: all 16 protocol frames, each on its own"
    input_wh, channels_last = (384, 288), True
    jitter = {"scale_factor": 0.35, "rot_factor": 45.0, "rot_prob": 0.6}
    official = {"source": "DarkPose experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml (commit 0185b427)", "input_wh": [288, 384],
                "heatmap_wh": [72, 96], "sigma": 3, "batch": 112, "per_gpu": 28, "gpus": 4, "lr": 1e-3, "optimizer": "Adam",
                "epochs": 210, "lr_step": [170, 200], "lr_factor": 0.1, "warmup": "none", "rot_factor": 45, "scale_factor": 0.35,
                "flip": True, "half_body": [0.3, 8], "flip_test": True, "blur_kernel": 11, "init": "hrnet_w32-36af842e.pth (ImageNet)"}
    schedule = {"warmup_iters": 0, "milestones": (170 / 210, 200 / 210), "gamma": 0.1, "grad_clip": None}
    defaults = {"epochs": 210, "workers": 8, "eval_batch": 64, "val_every": 10}

    def build_model(self, args):
        m, self.loaded = build_hrnet_w32(core.N_OUT)
        W, H = self.input_wh; self.codec = DarkCodec(self.input_wh, (W // 4, H // 4), sigma=3, kernel=11)
        return m

    def param_groups(self, model, lr): return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    def optimizer(self, groups, lr): return torch.optim.Adam(groups, lr=lr)

    def loss(self, model, img, kp, vis, ch):
        tgt, w = self.codec.encode(kp, vis); return joints_mse(model(img), ch, tgt, w, core.N_OUT, half=True)

    def predict(self, model, img, ch): return self.codec.decode(select(model(img), ch))[0]

    def recipe(self, args):
        return {"model": "pose HRNet-W32 (official code, vendored third_party/hrnet) with a new 1x1 head of 36 channels",
                "backbone_init": {**Wt.describe("hrnet_w32_imagenet"), "loaded_layers": self.loaded, "new": "final_layer (36 x 32 x 1 x 1), N(0, 0.001)"},
                "pretraining": "ImageNet-1k classification only (no pose data)", **self.codec.describe(),
                "loss": "JointsMSELoss(use_target_weight): mean over the 36 outputs of 0.5 x MSE(pred x w, gt x w)",
                "optimizer_cfg": {"type": "Adam", "weight_decay": 0, "betas": [0.9, 0.999]},
                "augmentation": {**self.jitter, "flip": False, "half_body": False, "note": "random scale/rotation of the full frame"},
                "test_time": "single full frame, no flip test, DARK decoding",
                "adaptations": ["36 outputs (12 tool + 24 contact), loss and readout on the true group", "full content frame, input 384x288 (official 288x384 turned landscape), heatmap 96x72",
                                "train frames: ex.frames_1fps; test: all 16 protocol frames", "total batch 112 = 28 x 4 accumulation steps on one GPU",
                                "no flip / half-body / flip test", "bf16 autocast", "val every 10 epochs (DarkPose validates every epoch)"]}


def main(argv=None):
    return engine.main(HRNetSpec(), argv)


if __name__ == "__main__":
    main(); sys.exit(0)
