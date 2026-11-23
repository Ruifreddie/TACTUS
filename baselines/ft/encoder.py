"""The V-JEPA 2.1 ViT-L/16 that the fine-tuning runs train, built through TactusNet's own wrapper and weights.

The wrapper is UniTAP's kpt_combo/vjepa_wrapper.py, the same class that produced every token cache of TactusNet, so the
frozen encoder here reproduces the cached tokens (tests/test_ft.py checks this). With `trainable=True` the encoder gets
the parts of a fine-tuning recipe that the frozen one does not need: activation checkpointing (memory only, the values
are unchanged) and stochastic depth.

Recipe source: VideoMAE (MCG-NJU/VideoMAE, scripts/kinetics/videomae_vit_large_patch16_224_tubemasking_ratio_0.9_epoch_1600/
finetune.sh and run_class_finetuning.py / optim_factory.py), the official fine-tuning of a self-supervised video ViT-L:
AdamW (0.9, 0.999), weight decay 0.05 except 1-D parameters, biases and embeddings, layer-wise LR decay 0.75, drop path
0.1, LR 2e-3 per 256 clips, warmup 5 of 35 epochs from 1e-6, cosine to 1e-6 (both also per 256 clips), no gradient
clipping. train.py scales the schedule to the run's length and the LR to its batch.
"""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

NUM_LAYERS = 24                       # ViT-L
_EMBED = ("cls_token", "mask_token", "pos_embed", "img_mod_embed", "video_mod_embed")


def _wrapper_module(unitap):
    kc = Path(unitap) / "third_party/t2b/kpt_combo"
    spec = importlib.util.spec_from_file_location("ft_vjepa_wrapper", kc / "vjepa_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Encoder(nn.Module):
    """tokens = Encoder(video): video [B,16,3,384,384] (normalized, the trainers' dataset layout) -> [B,8,24,24,1024]
    float32, the layout of TactusNet's token cache (time-major, then rows, then columns)."""

    def __init__(self, unitap, ckpt, trainable=True, drop_path=0.1, act_ckpt=True):
        super().__init__()
        if not Path(ckpt).is_file():
            raise SystemExit(f"V-JEPA 2.1 weight not found: {ckpt}")
        W = _wrapper_module(unitap)
        self.w = W.VJEPAWrapper(vjepa_repo=str(Path(unitap) / "third_party/t2b/vjepa2"), arch="vjepa2_1_vit_large_384",
                                img_size=384, num_frames=16, patch_size=16, tubelet_size=2, hf_repo="",
                                pretrained_from_hub=False, checkpoint_path=str(ckpt), freeze=not trainable)
        self.trainable = bool(trainable)
        net = self.w.encoder
        if len(net.blocks) != NUM_LAYERS:
            raise RuntimeError(f"expected a ViT-L with {NUM_LAYERS} blocks, got {len(net.blocks)}")
        self.drop_path = 0.0
        if self.trainable:
            net.use_activation_checkpointing = bool(act_ckpt)
            if drop_path > 0:
                from app.vjepa_2_1.models.utils.modules import DropPath   # the encoder's own class (its repo is on sys.path)
                rates = torch.linspace(0, float(drop_path), NUM_LAYERS).tolist()   # the stochastic depth rule of the ViT
                for blk, r in zip(net.blocks, rates):
                    blk.drop_path = DropPath(r) if r > 0 else nn.Identity()
                self.drop_path = float(drop_path)
        else:
            net.use_activation_checkpointing = False
            for p in net.parameters():
                p.requires_grad_(False)
        self.train(self.trainable)

    @property
    def net(self):
        return self.w.encoder

    def train(self, mode=True):
        return super().train(bool(mode) and self.trainable)     # a frozen encoder always runs in eval mode

    def forward(self, video):
        x = video.permute(0, 2, 1, 3, 4).contiguous()          # [B,3,T,H,W], as the wrapper feeds the ViT
        flat = self.w.encoder(x)
        grid, _, _ = self.w._tokens_flat_to_grid(flat, video)
        return grid

    def state(self):
        """The ViT weights alone (keys of the wrapper's encoder), which UniTAP's pipeline can load into its own encoder."""
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}


class EncoderEMA:
    """Exponential moving average of the encoder weights (decay as the heads' EMA). Validation and test use it."""

    def __init__(self, enc, decay=0.999):
        self.decay = float(decay)
        self.model = copy.deepcopy(enc)
        self.model.net.use_activation_checkpointing = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self._src = [p for p in enc.parameters()]
        self._dst = [p for p in self.model.parameters()]
        if len(self._src) != len(self._dst):
            raise RuntimeError("EMA copy has a different parameter list")

    @torch.no_grad()
    def update(self):
        torch._foreach_mul_(self._dst, self.decay)
        torch._foreach_add_(self._dst, [p.detach() for p in self._src], alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "weights": self.model.state()}

    def load_state_dict(self, sd):
        self.decay = float(sd["decay"])
        self.model.net.load_state_dict(sd["weights"], strict=True)


def layer_id(name):
    """VideoMAE's get_num_layer_for_vit: embeddings 0, block N -> N + 1, everything else (final norms) L + 1."""
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    if name.startswith("patch_embed") or name in _EMBED:
        return 0
    return NUM_LAYERS + 1


def param_groups(net, lr, layer_decay, weight_decay):
    """Parameter groups of VideoMAE's optim_factory.get_parameter_groups with its LayerDecayValueAssigner: layer i gets
    lr * layer_decay ** (L + 1 - i); 1-D parameters, biases and embeddings get no weight decay."""
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        i = layer_id(n)
        no_wd = p.ndim == 1 or n.endswith(".bias") or n in _EMBED
        key = (i, no_wd)
        if key not in groups:
            groups[key] = dict(params=[], names=[], layer=i, lr_scale=float(layer_decay) ** (NUM_LAYERS + 1 - i),
                               weight_decay=0.0 if no_wd else float(weight_decay), lr=float(lr))
        groups[key]["params"].append(p)
        groups[key]["names"].append(n)
    return [groups[k] for k in sorted(groups)]


def videomae_schedule(base, total_steps, warmup_steps, warmup_value, final_value):
    """utils.cosine_scheduler of VideoMAE: linear warmup from warmup_value to base over warmup_steps, then cosine to
    final_value at the last step. Returns one value per optimizer step."""
    warm = np.linspace(warmup_value, base, warmup_steps) if warmup_steps > 0 else np.array([])
    iters = np.arange(total_steps - warmup_steps)
    cos = final_value + 0.5 * (base - final_value) * (1 + np.cos(np.pi * iters / max(len(iters), 1)))
    out = np.concatenate((warm, cos))
    if len(out) != total_steps:
        raise RuntimeError(f"schedule has {len(out)} steps, expected {total_steps}")
    return out
