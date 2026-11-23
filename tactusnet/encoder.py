"""The shared, frozen video encoder.  Every layer reads the tokens it returns.

All implementations take 16 BGR uint8 frames [16, H, W, 3] (already cropped to the content area) and return float32 tokens
[1, 8, 24, 24, 1024]: V-JEPA 2.1 ViT-L/384, final layer, tubelet 2 -> 8 time tokens, patch 16 at 384 px -> 24 x 24.
"""
from typing import Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn.functional as F

TOKEN_SHAPE: Tuple[int, int, int, int] = (8, 24, 24, 1024)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TokenEncoder(Protocol):
    def encode(self, frames_bgr_u8: np.ndarray) -> torch.Tensor: ...


def preprocess_bgr(frames_bgr_u8: np.ndarray, size: int = 384) -> torch.Tensor:
    """[16,H,W,3] uint8 BGR -> [1,16,3,size,size] float32; the same recipe as UniTAP's `preprocess` and the token converter."""
    if frames_bgr_u8.ndim != 4 or frames_bgr_u8.shape[-1] != 3 or frames_bgr_u8.dtype != np.uint8:
        raise ValueError(f"expected uint8 BGR frames [T,H,W,3], got {frames_bgr_u8.dtype} {frames_bgr_u8.shape}")
    rgb = np.ascontiguousarray(frames_bgr_u8[..., ::-1])
    v = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
    return ((v - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def check_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 5 or tuple(tokens.shape[1:]) != TOKEN_SHAPE:
        raise ValueError(f"expected tokens [B, {', '.join(map(str, TOKEN_SHAPE))}], got {tuple(tokens.shape)}")
    if not torch.isfinite(tokens).all():
        raise ValueError("tokens contain NaN or Inf")
    return tokens


class VJEPAEncoder:
    """Standalone encoder built on the V-JEPA wrapper in `_vendored_vjepa_wrapper.py` (vendored byte for byte).

    Needs a local V-JEPA 2.1 source checkout (`vjepa_repo`, e.g. training/unitap/third_party/t2b/vjepa2 in this repository) and the 5.2 GB weight
    file `vjepa2_1_vitl_dist_vitG_384.pt`.  Not exercised by the tests (they use FakeEncoder); load it once on a free GPU to check.
    """

    def __init__(self, vjepa_repo: str, checkpoint: str, device: str = "cuda") -> None:
        from ._vendored_vjepa_wrapper import VJEPAWrapper
        self.device = torch.device(device)
        self.net = VJEPAWrapper(vjepa_repo=vjepa_repo, arch="vjepa2_1_vit_large_384", img_size=384, num_frames=16, patch_size=16, tubelet_size=2,
                                hf_repo="", pretrained_from_hub=False, checkpoint_path=checkpoint, freeze=True).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, frames_bgr_u8: np.ndarray) -> torch.Tensor:
        out = self.net(preprocess_bgr(frames_bgr_u8).to(self.device))
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        return check_tokens(tokens.float())


class UniTAPTokenEncoder:
    """Reuse the encoder UniTAP's Pipeline already holds, so the whole pyramid loads V-JEPA once (see spatial.UniTAPSpatial)."""

    def __init__(self, annotator) -> None:
        self.annotator = annotator

    @torch.inference_mode()
    def encode(self, frames_bgr_u8: np.ndarray) -> torch.Tensor:
        return check_tokens(self.annotator.tokens(frames_bgr_u8))


class FakeEncoder:
    """Deterministic random tokens with the right shape; for tests and for checking the plumbing without a GPU."""

    def __init__(self, seed: int = 0, device: str = "cpu") -> None:
        self.seed, self.device = seed, torch.device(device)

    def encode(self, frames_bgr_u8: np.ndarray) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + int(frames_bgr_u8.sum() % 9973))
        return torch.randn(1, *TOKEN_SHAPE, generator=g).to(self.device)
