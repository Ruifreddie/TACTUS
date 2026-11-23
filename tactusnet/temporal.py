"""Temporal layer (When): where, inside the 16-step clip, the action starts and ends.

tokens [B,8,24,24,1024] -> per-step features [B,16,d] -> SoLA-style adapter -> point-based range head conditioned on (tool, action).
The adapter and the head are the original classes (vendored verbatim); `range_temporal_adapter` / `range_head` / `range_temporal_pool`
keep their `kpt_combo` names.  The feature builder is a simplified version of the original (see TemporalFeatureBuilder).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendored_heads import TemporalRangeOffsetHead, TemporalSoLaAdapter, decode_range_from_points
from .labels import ACTIONS, COMBOS, TOOLS, split_combo, tool_action_ids
from .schema import N_SAMPLED, TemporalOut
from .seeds import range_to_local_frames


def temporal_interpolate(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """[B,T,D] -> [B,target_t,D], linear along time (same as the original `_temporal_interpolate_dim1`)."""
    if x.shape[1] == target_t:
        return x
    return F.interpolate(x.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2).contiguous()


class TemporalFeatureBuilder(nn.Module):
    """Token grid -> one feature vector per time step.

    Per tubelet: LayerNorm -> Linear(1024->d) -> GELU on every spatial token, then mean and max over space, concatenated and projected
    (`range_temporal_pool`), then interpolated from 8 to 16 steps.  The original also ran a depth-wise conv refinement and a temporal
    mean residual before pooling, and froze the trunk in the interval run (its launcher passes no initial weights - check on the
    machine that trained it); here the trunk is an ordinary trainable block and the two extra modules are left out."""

    def __init__(self, in_dim: int = 1024, dim: int = 512, out_steps: int = N_SAMPLED) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.shared_trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU())
        self.range_temporal_pool = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t, h, w, d = tokens.shape
        x = self.shared_trunk(tokens.reshape(b, t, h * w, d))                      # [B,T,S,dim]
        f = self.range_temporal_pool(torch.cat((x.mean(dim=2), x.amax(dim=2)), dim=-1))   # [B,T,dim]
        return temporal_interpolate(f, self.out_steps)


class TemporalLayer(nn.Module):
    def __init__(self, in_dim: int = 1024, dim: int = 512, head_mode: str = "trident", trident_num_bins: int = 15, proposal_mode: str = "bmn",
                 quality_mode: str = "dgqp", adapter_kernel: int = 5, convention: str = "kpt_combo") -> None:
        super().__init__()
        self.convention = convention
        self.features = TemporalFeatureBuilder(in_dim, dim)
        self.range_temporal_adapter = TemporalSoLaAdapter(dim=dim, kernel_size=adapter_kernel, proj_dim=max(64, dim // 2))
        self.range_head = TemporalRangeOffsetHead(in_dim=dim, num_tool_classes=len(TOOLS), num_action_classes=len(ACTIONS), num_combo_classes=len(COMBOS),
                                                  condition_mode="tool_action", head_mode=head_mode, trident_num_bins=trident_num_bins,
                                                  proposal_mode=proposal_mode, quality_mode=quality_mode)

    def forward(self, tokens: torch.Tensor, tool_id: torch.Tensor, action_id: torch.Tensor, combo_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        f = self.features(tokens)
        a = self.range_temporal_adapter(f)
        out = self.range_head(a["adapted"], tool_id=tool_id, action_id=action_id, combo_id=combo_id)
        out["sola_projected"] = a["projected"]          # only the training-time similarity loss reads this
        return out

    @torch.inference_mode()
    def predict(self, tokens: torch.Tensor, combo_id: int, n_frames: int) -> TemporalOut:
        if tokens.shape[0] != 1:
            raise ValueError("predict() takes one clip at a time")
        tool_id, action_id = tool_action_ids(combo_id, self.convention)
        dev = tokens.device
        out = self.forward(tokens, torch.tensor([tool_id], device=dev), torch.tensor([action_id], device=dev), torch.tensor([combo_id], device=dev))
        quality = out.get("range_quality_logits")
        dec = decode_range_from_points(out["range_actionness_logits"], out["range_offsets"], quality)
        score = torch.sigmoid(out["range_actionness_logits"].float())
        if torch.is_tensor(quality):
            score = score * torch.sigmoid(quality.float())
        st, et = float(dec["pred_action_start_t"][0]), float(dec["pred_action_end_t"][0])
        s_local, e_local = range_to_local_frames(st, et, n_frames)
        return TemporalOut(start_t=st, end_t=et, start_local=s_local, end_local=e_local, score=float(score.max()), source="predicted")
