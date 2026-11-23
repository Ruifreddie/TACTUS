"""Frozen encoders of the backbone-swap ablation. TactusNet's heads are retrained on the tokens of each encoder.

Every encoder maps the 16 protocol frames of a clip (uint8 RGB [16,H,W,3]: content crop, native resolution, ex.frames)
to one token grid in TactusNet's cache layout, so the UniTAP trainers read it unchanged:

    [8, h, w, D] float16, time-major then rows then columns; time step k covers protocol frames 2k and 2k+1.

Preprocessing is TactusNet's (token converter convert_video16_to_vjepa_tokens_one_by_one.py): RGB, /255, bilinear squash
to size x size (align_corners=False, no antialias, in float32), ImageNet mean/std. Only the encoder and its input size
change. Layout per encoder:

  name             encoder (weights)                                    input  tokens      how the grid is made
  vjepa21_l384     V-JEPA 2.1 ViT-L/16, distilled from ViT-G (ours)     384    8x24x24x1024 tubelet 2 x 16 x 16 (native)
  vjepa21_l256     the same model at 256 px (resolution ablation)       256    8x16x16x1024 RoPE interpolated to 16x16
  vjepa2_l256      V-JEPA 2 ViT-L/16, fpc64-256 (the predecessor)       256    8x16x16x1024 tubelet 2 x 16 x 16 (native)
  dinov3_l16_384   DINOv3 ViT-L/16, LVD-1689M (image model)             384    8x24x24x1024 per frame, CLS + 4 registers
                                                                                            dropped, frames 2k, 2k+1 averaged
  dinov2_l14_336   DINOv2 ViT-L/14, LVD-142M (image model, fallback)    336    8x24x24x1024 per frame, CLS dropped, averaged
  lemonfm_384      LemonFM ConvNeXt-L (surgical image model, LEMON)     384    8x24x24x768  per frame, stage-3 map (stride 16)
                                                                                            after the pretrained LayerNorm
                                                                                            of the next downsampling, averaged
  videomae_b224    VideoMAE ViT-B/16, self-supervised on K400           224    8x14x14x768  tubelet 2 x 16 x 16 (native)
  videomae_l224    VideoMAE ViT-L/16, self-supervised on K400 (Tab. 4a) 224    8x14x14x1024 tubelet 2 x 16 x 16 (native)
  endofm_b224      Endo-FM ViT-B/16 TimeSformer, endoscopy (Tab. 4a)    224    8x14x14x768  16 frames, divided space-time
                                                                                            attention, frames 2k, 2k+1 averaged;
                                                                                            Endo-FM's own mean 0.45 / std 0.225

Per-frame image models give 16 maps; consecutive frames are averaged so that every encoder hands the heads the same 8
time steps over the same frames as a V-JEPA tubelet (the head decodes 2 sub-frames per step, as for V-JEPA). The heads read
the grid size from the tokens; D only sets the width of the head's input layers (stub encoder, see launch.py).
All weights are pinned to a revision and downloaded into $HF_HOME (on the pod: the volume or $BL_WEIGHTS).
"""
import importlib.util, os, sys
from pathlib import Path

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> spec. stub = --vjepa-arch the trainer builds (never run) so that its heads get the right input width.
SPECS = {
    "vjepa21_l384": dict(kind="vjepa_repo", arch="vjepa2_1_vit_large_384", size=384, grid=24, dim=1024, per_frame=False,
                         stub="vjepa2_1_vit_large_384", weights="$VJEPA_CKPT (vjepa2_1_vitl_dist_vitG_384.pt)",
                         desc="V-JEPA 2.1 ViT-L/16 (distilled from ViT-G), 16 frames at 384 px"),
    "vjepa21_l256": dict(kind="vjepa_repo", arch="vjepa2_1_vit_large_384", size=256, grid=16, dim=1024, per_frame=False,
                         stub="vjepa2_1_vit_large_384", weights="$VJEPA_CKPT (vjepa2_1_vitl_dist_vitG_384.pt)",
                         desc="V-JEPA 2.1 ViT-L/16 run at 256 px (RoPE interpolation)"),
    "vjepa2_l256": dict(kind="vjepa2_hf", repo="facebook/vjepa2-vitl-fpc64-256", revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
                        size=256, grid=16, dim=1024, per_frame=False, stub="vjepa2_1_vit_large_384", license="mit",
                        desc="V-JEPA 2 ViT-L/16 (fpc64-256), 16 frames at 256 px"),
    "dinov3_l16_384": dict(kind="timm_vit", model="vit_large_patch16_dinov3.lvd1689m", repo="timm/vit_large_patch16_dinov3.lvd1689m",
                           revision="30c1109559f65dea34316b0d4842d35c5771fe11", size=384, grid=24, dim=1024, per_frame=True,
                           stub="vjepa2_1_vit_large_384", license="DINOv3 License",
                           desc="DINOv3 ViT-L/16 (LVD-1689M), each frame at 384 px, frame pairs averaged"),
    "dinov2_l14_336": dict(kind="timm_vit", model="vit_large_patch14_dinov2.lvd142m", repo="timm/vit_large_patch14_dinov2.lvd142m",
                           revision="4741e1cafbf45415e77074bb0cb42dba76c8684a", size=336, grid=24, dim=1024, per_frame=True,
                           stub="vjepa2_1_vit_large_384", license="apache-2.0",
                           desc="DINOv2 ViT-L/14 (LVD-142M), each frame at 336 px, frame pairs averaged"),
    "lemonfm_384": dict(kind="lemonfm", repo="visurg/LemonFM", revision="4c58064eabf69b0ab148d30933549bf8a1c5bdcc", file="lemonfm.pth",
                        size=384, grid=24, dim=768, per_frame=True, stub="vjepa2_1_vit_base_384", license="apache-2.0",
                        desc="LemonFM ConvNeXt-L (LEMON surgical videos), stage-3 map of each frame at 384 px, frame pairs averaged"),
    "videomae_b224": dict(kind="videomae_hf", repo="MCG-NJU/videomae-base", revision="dc740ceda42fce44faed2ea03c6d447db72f6af9",
                          size=224, grid=14, dim=768, per_frame=False, stub="vjepa2_1_vit_base_384", license="cc-by-nc-4.0",
                          desc="VideoMAE ViT-B/16 (self-supervised on Kinetics-400), 16 frames at 224 px"),
    "videomae_l224": dict(kind="videomae_hf", repo="MCG-NJU/videomae-large", revision="12da269a02d3e1fbbb7011e610e591fea8061dca",
                          size=224, grid=14, dim=1024, per_frame=False, stub="vjepa2_1_vit_large_384", license="cc-by-nc-4.0",
                          desc="VideoMAE ViT-L/16 (self-supervised on Kinetics-400, pretrain only), 16 frames at 224 px"),
    "endofm_b224": dict(kind="endofm", size=224, grid=14, dim=768, per_frame=False, stub="vjepa2_1_vit_base_384",
                        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225), license="Apache-2.0 (code); weights: Endo-FM release",
                        weights="$BL_WEIGHTS/cls/endofm/endo_fm_teacher_backbone.pth (baselines.cls.weights.endofm_backbone)",
                        desc="Endo-FM ViT-B/16 TimeSformer (self-supervised on endoscopy videos), 16 frames at 224 px, frame pairs averaged"),
}


def unitap_dir():
    """UniTAP code (has third_party/t2b/kpt_combo and third_party/t2b/vjepa2): $U, else the local smoke copy."""
    for c in (os.environ.get("U"), os.path.join(os.environ.get("BL_ROOT", ""), "smoke/code/unitap")):
        if c and Path(c, "third_party/t2b/vjepa2").is_dir(): return Path(c)
    raise SystemExit("set U to the UniTAP code dir (third_party/t2b/vjepa2 is needed for V-JEPA 2.1)")


def _hf_file(repo, filename, revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, revision=revision)


class Encoder:
    """encode(list of uint8 RGB [16,H,W,3]) -> list of float16 [8, grid, grid, dim] (numpy)."""

    def __init__(self, name, device="cuda", amp="none", vjepa_ckpt=None, tf32=False):
        import torch
        if name not in SPECS: raise SystemExit(f"unknown backbone {name}; choose from {sorted(SPECS)}")
        self.name, self.spec, self.device = name, dict(SPECS[name]), torch.device(device)
        self.amp = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]
        # plain fp32 like the token converter (TF32 is not faster than fp32 on an RTX 4090 and shifts tokens by ~0.3 %)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32); torch.backends.cudnn.allow_tf32 = bool(tf32); self.tf32 = bool(tf32)
        # ImageNet mean/std as TactusNet's converter, unless the encoder was pretrained with its own (Endo-FM: 0.45 / 0.225)
        self.mean = torch.tensor(self.spec.get("mean", IMAGENET_MEAN), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.spec.get("std", IMAGENET_STD), device=self.device).view(1, 3, 1, 1)
        k = self.spec["kind"]
        if k == "vjepa_repo": self._build_vjepa_repo(vjepa_ckpt)
        elif k == "vjepa2_hf": self._build_hf("VJEPA2Model")
        elif k == "videomae_hf": self._build_hf("VideoMAEModel")
        elif k == "timm_vit": self._build_timm()
        elif k == "lemonfm": self._build_lemonfm()
        elif k == "endofm": self._build_endofm()
        self.net.eval().to(self.device)
        for p in self.net.parameters(): p.requires_grad_(False)

    # ------------------------------------------------------------------ builders
    def _build_vjepa_repo(self, ckpt):
        """TactusNet's own wrapper (UniTAP kpt_combo/vjepa_wrapper.py), exactly as the token converter builds it."""
        U = unitap_dir(); kc = U / "third_party/t2b/kpt_combo"
        spec = importlib.util.spec_from_file_location("ablate_vjepa_wrapper", kc / "vjepa_wrapper.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        ckpt = ckpt or os.environ.get("VJEPA_CKPT")
        if not ckpt or not Path(ckpt).is_file(): raise SystemExit(f"V-JEPA 2.1 weight not found: {ckpt!r} (set VJEPA_CKPT)")
        self.spec["weights"] = str(ckpt)
        self.net = mod.VJEPAWrapper(vjepa_repo=str(U / "third_party/t2b/vjepa2"), arch=self.spec["arch"], img_size=self.spec["size"],
                                    num_frames=16, patch_size=16, tubelet_size=2, hf_repo="", pretrained_from_hub=False,
                                    checkpoint_path=str(ckpt), freeze=True)

    def _build_hf(self, cls):
        import transformers
        self.net = getattr(transformers, cls).from_pretrained(self.spec["repo"], revision=self.spec["revision"])
        if cls == "VJEPA2Model": self.net = self.net.encoder          # what TactusNet's wrapper uses for V-JEPA 2 (hf backend)

    def _build_timm(self):
        import timm
        f = _hf_file(self.spec["repo"], "model.safetensors", self.spec["revision"])
        self.net = timm.create_model(self.spec["model"], pretrained=True, num_classes=0, img_size=self.spec["size"],
                                     pretrained_cfg_overlay=dict(file=f))

    def _build_lemonfm(self):
        """LemonFM = torchvision ConvNeXt-L with the DINO teacher weights of lemonfm.pth (the repo's model_loader.py)."""
        import torch, torchvision
        net = torchvision.models.convnext_large()
        sd = torch.load(_hf_file(self.spec["repo"], self.spec["file"], self.spec["revision"]), map_location="cpu", weights_only=False)["teacher"]
        sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
        msg = net.load_state_dict(sd, strict=False)
        bad = [k for k in msg.missing_keys if not k.startswith("classifier.")]
        if bad: raise SystemExit(f"LemonFM weights miss backbone keys: {bad[:5]}")
        self.load_report = {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}
        self.net = torch.nn.Sequential(*net.features[:6], net.features[6][0])   # stem .. stage 3, then the LayerNorm of downsample 3

    def _build_endofm(self):
        """Endo-FM's TimeSformer ViT-B/16 (divided space-time attention, 8 time embeddings, interpolated to the 16 frames) with
        the teacher backbone of the Endo-FM checkpoint, built as the classification baseline builds it (cls/models.py, EndoFM)."""
        import torch
        from functools import partial
        from baselines.cls import weights as W
        from baselines.cls.vendor.endofm.timesformer import VisionTransformer
        net = VisionTransformer(img_size=224, num_classes=0, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0.,
                                drop_path_rate=0., num_frames=8, attention_type="divided_space_time")
        net.head = None
        f = W.endofm_backbone(); self.spec["weights"] = str(f)
        res = net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise SystemExit(f"Endo-FM weights: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}")
        self.net = net

    # ------------------------------------------------------------------ forward
    def _prep(self, frames_u8):
        """uint8 RGB [16,H,W,3] numpy -> [16,3,S,S] float32 normalized (TactusNet's converter, on the GPU)."""
        import torch, torch.nn.functional as F
        S = self.spec["size"]
        v = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        v = F.interpolate(v, size=(S, S), mode="bilinear", align_corners=False)
        return (v - self.mean) / self.std

    def _forward(self, x):
        """x [B,16,3,S,S] -> [B,8,g,g,D] float32."""
        import torch
        B, T = x.shape[:2]; g, D, k = self.spec["grid"], self.spec["dim"], self.spec["kind"]
        if k == "vjepa_repo":
            tok = self.net(x)[0]                                         # [B,8,g,g,D]
        elif k == "vjepa2_hf":
            tok = self.net(pixel_values_videos=x).last_hidden_state.reshape(B, T // 2, g, g, D)
        elif k == "videomae_hf":
            tok = self.net(pixel_values=x).last_hidden_state.reshape(B, T // 2, g, g, D)
        elif k == "timm_vit":
            y = self.net.forward_features(x.flatten(0, 1))               # [B*16, prefix + g*g, D], final norm applied
            tok = y[:, self.net.num_prefix_tokens:].reshape(B, T, g, g, D)
            tok = tok.reshape(B, T // 2, 2, g, g, D).mean(2)
        elif k == "endofm":
            y = self.net.forward_features(x.permute(0, 2, 1, 3, 4), get_all=True)   # [B, 1 + g*g*T, D]: CLS, then '(h w t)'
            tok = y[:, 1:].reshape(B, g, g, T, D).permute(0, 3, 1, 2, 4)            # [B,T,g,g,D]
            tok = tok.reshape(B, T // 2, 2, g, g, D).mean(2)
        elif k == "lemonfm":
            y = self.net(x.flatten(0, 1))                                # [B*16, 768, g, g]
            tok = y.permute(0, 2, 3, 1).reshape(B, T, g, g, D).reshape(B, T // 2, 2, g, g, D).mean(2)
        if tuple(tok.shape[1:]) != (T // 2, g, g, D): raise RuntimeError(f"{self.name}: tokens {tuple(tok.shape)}, expected [B,{T // 2},{g},{g},{D}]")
        return tok.float()

    def encode(self, clips):
        import torch
        x = torch.stack([self._prep(f) for f in clips])
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.amp or torch.float32, enabled=self.amp is not None and self.device.type == "cuda"):
            tok = self._forward(x)
        if not torch.isfinite(tok).all(): raise RuntimeError(f"{self.name}: non-finite tokens")
        return [t.cpu().numpy().astype(np.float16) for t in tok]

    def describe(self):
        s = {k: v for k, v in self.spec.items()}
        s.update(name=self.name, layout="[8,h,w,D] fp16, t-major then rows then columns; step k = protocol frames 2k, 2k+1",
                 preprocessing="RGB, /255, bilinear squash to size x size (align_corners=False, float32), ImageNet mean/std",
                 precision=("fp32" + (" (TF32 matmuls)" if self.tf32 else "")) if self.amp is None else str(self.amp))
        if hasattr(self, "load_report"): s["load_report"] = self.load_report
        return s
