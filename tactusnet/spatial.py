"""Spatial layer (How / Where): UniTAP, wrapped behind a small interface.

    crop(frames)                              -> content-area crop (black bars and UI strips removed) + its bbox
    predict(cropped, bbox, image_wh, tool, action, sampled_local, tokens=None) -> SpatialOut   (3 tool + 4 contact points for each of the 16 frames)

UniTAP itself (stage-1 head -> zoom box -> second encoding -> stage-2 head + contact head -> pixel refiner) lives in training/unitap
(tools/ and third_party/). `UniTAPSpatial` imports it from `unitap_root`, a directory that also holds data/case32/v1/labels.json and the
head checkpoints (an experiment directory of the training pipeline). tactusnet/patches/unitap_deploy.patch re-points the V-JEPA paths
stored in the checkpoints to this checkout; apply it when the checkpoints were trained on another machine (see README.md).
"""
import os
import sys
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np
import torch

from .encoder import UniTAPTokenEncoder
from .schema import N_SAMPLED, SpatialOut


class SpatialLayer(Protocol):
    def crop(self, frames_bgr_u8: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]: ...
    def predict(self, cropped: np.ndarray, bbox: Tuple[int, int, int, int], image_wh: Tuple[int, int], tool: str, action: str,
                sampled_local: np.ndarray, tokens: Optional[torch.Tensor] = None) -> SpatialOut: ...


class UniTAPSpatial:
    def __init__(self, unitap_root: str, device: str = "cuda", vjepa_ckpt: Optional[str] = None, ckpts: Optional[dict] = None) -> None:
        root = Path(unitap_root)
        if vjepa_ckpt:
            os.environ["VJEPA_CKPT"] = str(vjepa_ckpt)           # read by the patched Pipeline.__init__
        sys.path.insert(0, str(root / "tools"))
        import annotate_clips as AC                              # puts the rest of UniTAP's code on sys.path
        self.AC = AC

        class SharedTokenAnnotator(AC.Annotator):
            """Annotator whose first `tokens()` call can return tokens that were already computed, so the full-frame clip is encoded once."""
            _preset = None

            def tokens(self, frames_bgr_u8):
                if self._preset is not None:
                    tok, self._preset = self._preset, None
                    return tok
                return super().tokens(frames_bgr_u8)

        self.ann = SharedTokenAnnotator(ckpts=ckpts, device=device)

    def token_encoder(self) -> UniTAPTokenEncoder:
        """The encoder UniTAP already holds, for the semantic and temporal layers to reuse (one V-JEPA in memory)."""
        return UniTAPTokenEncoder(self.ann)

    def crop(self, frames_bgr_u8: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        BLD = self.AC.BLD
        bbox = BLD.detect_content_bbox(frames_bgr_u8[0])
        return np.stack([BLD.crop_frame(f, bbox) for f in frames_bgr_u8]), tuple(int(v) for v in bbox)

    @torch.inference_mode()
    def predict(self, cropped, bbox, image_wh, tool, action, sampled_local, tokens=None) -> SpatialOut:
        x0, y0, x1, y1 = bbox
        self.ann._preset = tokens
        r = self.ann.run_clip_conf(cropped, tool, action)
        self.ann._preset = None
        scale, off = [x1 - x0, y1 - y0], [x0, y0]
        return SpatialOut(
            sampled_local=np.asarray(sampled_local), tool_xy=np.asarray(r["tool_xy"]) * scale + off, tool_conf=np.nan_to_num(np.asarray(r["tool_conf"]), nan=0.5),
            contact_xy=np.asarray(r["contact_xy"]) * scale + off, contact_conf=np.nan_to_num(np.asarray(r["contact_conf"]), nan=0.5),
            roles=list(self.AC.TOPO[self.ann.tool2topology.get(tool, "1A2T")]), image_wh=tuple(image_wh))


class MockSpatial:
    """Deterministic stand-in with UniTAP's output shapes, for tests and plumbing checks (no GPU, no weights)."""

    def __init__(self, roles=("tool_anchor", "tool_anchor", "tool_tip")) -> None:
        self.roles = list(roles)

    def crop(self, frames_bgr_u8):
        h, w = frames_bgr_u8.shape[1:3]
        return frames_bgr_u8, (0, 0, w, h)

    def predict(self, cropped, bbox, image_wh, tool, action, sampled_local, tokens=None) -> SpatialOut:
        w, h = image_wh
        t = np.linspace(0.3, 0.7, N_SAMPLED)[:, None]
        tool_xy = np.stack([np.concatenate([t * w, t * h], 1) + [dx, dx] for dx in (0.0, 8.0, 16.0)], axis=1)           # [16,3,2]
        contact_xy = np.stack([np.concatenate([t * w, t * h], 1) + [dx, -dx] for dx in (-20.0, -10.0, 10.0, 20.0)], axis=1)  # [16,4,2]
        peak = np.exp(-0.5 * ((np.arange(N_SAMPLED) - 8) / 3.0) ** 2)
        return SpatialOut(sampled_local=np.asarray(sampled_local), tool_xy=tool_xy, tool_conf=np.repeat(peak[:, None], 3, 1) * 0.8 + 0.1,
                          contact_xy=contact_xy, contact_conf=np.repeat(peak[:, None], 4, 1) * 0.1 + 0.01, roles=self.roles, image_wh=(w, h))
