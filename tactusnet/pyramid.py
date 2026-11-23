"""TactusNet: the pyramid that turns one clip into every annotation.

    frames -> crop -> encode once -> Which/What (semantic) -> When (temporal, conditioned on tool+action)
                                  -> How/Where (spatial, UniTAP) -> seeds -> prompt -> tracks (server CoTracker)

Each layer's output is the next layer's condition; any of them can be supplied by a person instead of predicted
(`tool`/`action` -> skip the semantic layer, `action_range` -> skip the temporal layer), which is how the Silver batch runs.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoder import TokenEncoder, check_tokens
from .labels import combo_id_of, split_combo, tool_action_ids
from .schema import N_SAMPLED, Annotation, ClipSpec, SemanticOut, TemporalOut
from .seeds import build_prompt, choose_seeds
from .semantic import SemanticHead
from .spatial import SpatialLayer
from .temporal import TemporalLayer
from .tracking import JobRef, TrackingLayer


class TactusNet(nn.Module):
    def __init__(self, encoder: TokenEncoder, semantic: SemanticHead, temporal: TemporalLayer, spatial: SpatialLayer, tracker: Optional[TrackingLayer] = None) -> None:
        super().__init__()
        self.encoder, self.semantic, self.temporal, self.spatial, self.tracker = encoder, semantic, temporal, spatial, tracker

    @torch.inference_mode()
    def annotate(self, frames_bgr_u8: np.ndarray, clip: ClipSpec, *, tool: Optional[str] = None, action: Optional[str] = None,
                 action_range: Optional[Tuple[int, int]] = None, job: Optional[JobRef] = None, track: bool = False) -> Annotation:
        """frames_bgr_u8: the 16 sampled frames [16,H,W,3] (indices `clip.sampled_local()`), uncropped.
        action_range: (start, end) as frame indices inside the clip, if a person supplies it."""
        if frames_bgr_u8.shape[0] != N_SAMPLED:
            raise ValueError(f"expected {N_SAMPLED} frames, got {frames_bgr_u8.shape[0]}")
        if (tool is None) != (action is None):
            raise ValueError("give both tool and action, or neither")
        self.eval()
        image_wh = (int(frames_bgr_u8.shape[2]), int(frames_bgr_u8.shape[1]))
        cropped, bbox = self.spatial.crop(frames_bgr_u8)
        tokens = check_tokens(self.encoder.encode(cropped))                                   # the one shared full-frame encoding

        if tool is None:                                                                      # Which / What
            sem = self.semantic.predict(tokens)
        else:
            cid = combo_id_of(tool, action)
            sem = SemanticOut(combo_id=cid, combo=f"{tool}|{action}", tool=tool, action=action, conf=1.0, probs=[float(i == cid) for i in range(6)], source="given")

        if action_range is None:                                                              # When
            tmp = self.temporal.predict(tokens, sem.combo_id, clip.n)
        else:
            s, e = int(action_range[0]), int(action_range[1])
            if not 0 <= s <= e < clip.n:
                raise ValueError(f"action_range {action_range} must satisfy 0 <= start <= end < {clip.n}")
            tmp = TemporalOut(start_t=s / max(clip.n - 1, 1) * (N_SAMPLED - 1), end_t=e / max(clip.n - 1, 1) * (N_SAMPLED - 1), start_local=s, end_local=e, score=1.0, source="given")

        sampled = clip.sampled_local()                                                        # How / Where
        spatial = self.spatial.predict(cropped, bbox, image_wh, sem.tool, sem.action, sampled, tokens=tokens)
        seeds = choose_seeds(spatial.sampled_local, spatial.tool_conf, spatial.contact_conf, tmp.start_local, tmp.end_local)
        prompt = build_prompt(spatial, seeds, clip.a0)

        tracks: Optional[Dict[str, Any]] = None
        if track:
            if self.tracker is None or job is None:
                raise ValueError("track=True needs a tracker and a JobRef")
            tracks = self.tracker.submit(job, prompt)
        return Annotation(clip=clip, semantic=sem, temporal=tmp, spatial=spatial, seeds=seeds, prompt=prompt, tracks=tracks)
