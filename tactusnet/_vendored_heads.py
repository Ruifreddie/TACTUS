"""VERBATIM EXTRACT -- do not edit by hand; regenerate with tactusnet/tools/extract_vendored.py.

Source : UniTAP_project.zip::third_party/t2b/kpt_combo/model.py  (sha256 22327746823c95561ad5b81e9e4339ad2c1c83ac90450600ba8b0d0fcf488a03)
Blocks : _pick_attn_heads, _normalize_class_ids, _build_trajectory_condition, CrossAttentionPooler, LinearClassifierHead, ToolConditionedActionPrior, ComboHead, TemporalSoLaAdapter, TemporalRangeOffsetHead, KeypointComboNet._decode_range_from_points -> decode_range_from_points
Only the module-level `_decode_range_from_points` staticmethod was un-indented, de-decorated and renamed `decode_range_from_points`.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_attn_heads(embed_dim: int) -> int:
    for h in (16, 12, 8, 4, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


def _normalize_class_ids(
    ids: Optional[torch.Tensor],
    *,
    name: str,
    bsz: int,
    num_classes: int,
    ref_tensor: torch.Tensor,
) -> torch.Tensor:
    if num_classes <= 0:
        raise ValueError(f"`num_classes` for {name} must be > 0, got {num_classes}")
    if ids is None:
        return ref_tensor.new_zeros((bsz,), dtype=torch.long)
    ids = ids.long().to(device=ref_tensor.device)
    if ids.ndim != 1 or int(ids.shape[0]) != int(bsz):
        raise ValueError(f"`{name}` must be [B], got {tuple(ids.shape)}")
    if bool((ids < 0).any().item()) or bool((ids >= num_classes).any().item()):
        raise ValueError(
            f"`{name}` out of range [0,{num_classes - 1}] "
            f"(min={int(ids.min().item())}, max={int(ids.max().item())})"
        )
    return ids


def _build_trajectory_condition(
    *,
    condition_mode: str,
    ref_tensor: torch.Tensor,
    tool_embed: nn.Embedding,
    num_tool_classes: int,
    tool_id: Optional[torch.Tensor] = None,
    action_embed: Optional[nn.Embedding] = None,
    num_action_classes: int = 0,
    action_id: Optional[torch.Tensor] = None,
    combo_embed: Optional[nn.Embedding] = None,
    num_combo_classes: int = 0,
    combo_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz = int(ref_tensor.shape[0])
    mode = str(condition_mode).lower()
    if mode == "combo":
        if combo_embed is None:
            raise RuntimeError("combo conditioning requested but combo_embed is None.")
        combo_id_in = _normalize_class_ids(
            combo_id,
            name="combo_id",
            bsz=bsz,
            num_classes=int(num_combo_classes),
            ref_tensor=ref_tensor,
        )
        return combo_embed(combo_id_in)

    tool_id_in = _normalize_class_ids(
        tool_id,
        name="tool_id",
        bsz=bsz,
        num_classes=int(num_tool_classes),
        ref_tensor=ref_tensor,
    )
    cond = tool_embed(tool_id_in)
    if mode == "tool":
        return cond
    if mode == "tool_action":
        if action_embed is None:
            raise RuntimeError("tool_action conditioning requested but action_embed is None.")
        action_id_in = _normalize_class_ids(
            action_id,
            name="action_id",
            bsz=bsz,
            num_classes=int(num_action_classes),
            ref_tensor=ref_tensor,
        )
        return cond + action_embed(action_id_in)
    raise ValueError(f"Unsupported trajectory condition mode: {condition_mode}")


class CrossAttentionPooler(nn.Module):
    """Single-query cross-attention pooler (probe-style)."""

    def __init__(self, in_dim: int, num_heads: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        heads = _pick_attn_heads(int(in_dim)) if num_heads is None else int(num_heads)
        if int(in_dim) % heads != 0:
            raise ValueError(f"`in_dim` must be divisible by `num_heads`, got in_dim={in_dim}, heads={heads}")
        self.query = nn.Parameter(torch.zeros(1, 1, int(in_dim)))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.q_norm = nn.LayerNorm(int(in_dim))
        self.kv_norm = nn.LayerNorm(int(in_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(in_dim),
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, query_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,N,D]
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,N,D], got {tuple(x.shape)}")
        bsz = int(x.shape[0])
        q = self.query.expand(bsz, -1, -1)  # [B,1,D]
        if query_bias is not None:
            if query_bias.ndim == 2:
                query_bias = query_bias.unsqueeze(1)
            if query_bias.ndim != 3 or int(query_bias.shape[1]) != 1 or int(query_bias.shape[2]) != int(q.shape[2]):
                raise ValueError(
                    f"`query_bias` must be [B,D] or [B,1,D], got {tuple(query_bias.shape)} "
                    f"for query shape {tuple(q.shape)}"
                )
            if int(query_bias.shape[0]) != bsz:
                raise ValueError(
                    f"`query_bias` batch mismatch: expected {bsz}, got {int(query_bias.shape[0])}"
                )
            q = q + query_bias.to(dtype=q.dtype, device=q.device)
        q = self.q_norm(q)
        kv = self.kv_norm(x)
        pooled, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        return pooled.squeeze(1)  # [B,D]


class LinearClassifierHead(nn.Module):
    """Minimal classifier head: LayerNorm + Linear."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class ToolConditionedActionPrior(nn.Module):
    """Tool-conditioned action-token bias at query stage.

    In dual-attn probe mode:
      q_action' = q_action + alpha * Linear(softmax(tool_logits))
    """

    def __init__(
        self,
        num_tool_classes: int,
        query_dim: int,
        init_alpha: float = 0.0,
        detach_tool_prob: bool = True,
    ) -> None:
        super().__init__()
        self.num_tool_classes = int(num_tool_classes)
        self.query_dim = int(query_dim)
        self.detach_tool_prob = bool(detach_tool_prob)
        if self.num_tool_classes <= 0:
            raise ValueError(f"`num_tool_classes` must be > 0, got {self.num_tool_classes}")
        if self.query_dim <= 0:
            raise ValueError(f"`query_dim` must be > 0, got {self.query_dim}")

        self.tool_to_query = nn.Linear(self.num_tool_classes, self.query_dim, bias=False)
        # Start from no prior; model learns if tool-conditioned query helps.
        nn.init.zeros_(self.tool_to_query.weight)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def forward(self, tool_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tool_logits.ndim != 2:
            raise ValueError(f"`tool_logits` must be [B,Nt], got {tuple(tool_logits.shape)}")
        if int(tool_logits.shape[1]) != self.num_tool_classes:
            raise ValueError(
                f"`tool_logits` class dim mismatch: expected {self.num_tool_classes}, got {int(tool_logits.shape[1])}"
            )
        tool_prob = torch.softmax(tool_logits, dim=-1)
        if self.detach_tool_prob:
            tool_prob = tool_prob.detach()
        cond = self.tool_to_query(tool_prob)
        delta = self.alpha.to(dtype=cond.dtype, device=cond.device) * cond
        return delta, tool_prob


class ComboHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        p = float(dropout)
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"`dropout` for ComboHead must be in [0,1), got {p}")
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSoLaAdapter(nn.Module):
    """SoLa-style lightweight temporal adapter for frozen [B,T,D] features."""

    def __init__(self, dim: int, kernel_size: int = 5, proj_dim: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.proj_dim = int(proj_dim)
        if self.dim <= 0:
            raise ValueError(f"`dim` must be > 0, got {self.dim}")
        if self.kernel_size <= 0 or (self.kernel_size % 2) != 1:
            raise ValueError(
                f"`kernel_size` must be a positive odd integer, got {self.kernel_size}"
            )
        if self.proj_dim <= 0:
            raise ValueError(f"`proj_dim` must be > 0, got {self.proj_dim}")
        self.norm = nn.LayerNorm(self.dim)
        self.in_proj = nn.Linear(self.dim, self.dim)
        self.temporal_conv = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.sim_norm = nn.LayerNorm(self.dim)
        self.sim_proj = nn.Linear(self.dim, self.proj_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"`x` feature dim mismatch: expected {self.dim}, got {int(x.shape[-1])}"
            )
        if int(x.shape[1]) <= 0:
            projected = self.sim_proj(self.sim_norm(x))
            return {
                "adapted": x,
                "projected": projected,
            }
        h = self.in_proj(self.norm(x))
        h = self.act(h)
        h = self.temporal_conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        adapted = x + self.out_proj(h)
        projected = self.sim_proj(self.sim_norm(adapted))
        return {
            "adapted": adapted,
            "projected": projected,
        }


class TemporalRangeOffsetHead(nn.Module):
    """Point-based temporal range head (single-event): actionness + left/right offsets."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_tool_classes: int = 0,
        num_action_classes: int = 0,
        num_combo_classes: int = 0,
        condition_mode: str = "none",
        head_mode: str = "scalar",
        trident_num_bins: int = 15,
        proposal_mode: str = "none",
        quality_mode: str = "none",
        task_decouple_mode: str = "none",
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_tool_classes = int(num_tool_classes)
        self.num_action_classes = int(num_action_classes)
        self.num_combo_classes = int(num_combo_classes)
        self.condition_mode = str(condition_mode).lower()
        self.head_mode = str(head_mode).lower()
        self.trident_num_bins = int(trident_num_bins)
        self.proposal_mode = str(proposal_mode).lower()
        self.quality_mode = str(quality_mode).lower()
        self.task_decouple_mode = str(task_decouple_mode).lower()
        if self.condition_mode not in ("none", "tool", "tool_action", "combo"):
            raise ValueError(
                f"Unsupported range condition_mode={condition_mode}, expected one of: none,tool,tool_action,combo"
            )
        if self.head_mode not in ("scalar", "trident"):
            raise ValueError(
                f"Unsupported range head_mode={head_mode}, expected one of: scalar,trident"
            )
        if self.proposal_mode not in ("none", "bmn"):
            raise ValueError(
                f"Unsupported proposal_mode={proposal_mode}, expected one of: none,bmn"
            )
        if self.quality_mode not in ("none", "dgqp"):
            raise ValueError(
                f"Unsupported quality_mode={quality_mode}, expected one of: none,dgqp"
            )
        if self.task_decouple_mode not in ("none", "score_loc", "score_loc_align"):
            raise ValueError(
                f"Unsupported task_decouple_mode={task_decouple_mode}, expected one of: none,score_loc,score_loc_align"
            )
        if self.head_mode == "trident" and self.trident_num_bins <= 0:
            raise ValueError(f"`trident_num_bins` must be > 0 for trident head, got {self.trident_num_bins}")
        if self.quality_mode == "dgqp" and self.head_mode != "trident":
            raise ValueError("`quality_mode=dgqp` requires `head_mode=trident`.")
        if self.condition_mode != "none":
            if self.num_tool_classes <= 0:
                raise ValueError(f"`num_tool_classes` must be > 0, got {self.num_tool_classes}")
            if self.condition_mode == "tool_action" and self.num_action_classes <= 0:
                raise ValueError(f"`num_action_classes` must be > 0, got {self.num_action_classes}")
            if self.condition_mode == "combo" and self.num_combo_classes <= 0:
                raise ValueError(f"`num_combo_classes` must be > 0, got {self.num_combo_classes}")
            self.tool_embed = nn.Embedding(self.num_tool_classes, self.in_dim)
            self.action_embed = nn.Embedding(max(self.num_action_classes, 1), self.in_dim)
            self.combo_embed = nn.Embedding(max(self.num_combo_classes, 1), self.in_dim)
            nn.init.trunc_normal_(self.tool_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.action_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.combo_embed.weight, std=0.02)
            self.cond_scale = nn.Linear(self.in_dim, self.in_dim)
            self.cond_shift = nn.Linear(self.in_dim, self.in_dim)
        else:
            self.tool_embed = None
            self.action_embed = None
            self.combo_embed = None
            self.cond_scale = None
            self.cond_shift = None
        if self.task_decouple_mode in ("score_loc", "score_loc_align"):
            self.pre_cls = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            )
            self.pre_loc = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            )
            self.pre = None
            if self.task_decouple_mode == "score_loc_align":
                self.align_norm = nn.LayerNorm(hidden_dim * 2)
                self.align_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                nn.init.zeros_(self.align_gate.weight)
                nn.init.constant_(self.align_gate.bias, math.log(0.1 / 0.9))
            else:
                self.align_norm = None
                self.align_gate = None
        else:
            self.pre = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            )
            self.pre_cls = None
            self.pre_loc = None
            self.align_norm = None
            self.align_gate = None
        self.actionness_proj = nn.Linear(hidden_dim, 1)
        if self.head_mode == "trident":
            self.offset_proj = nn.Linear(hidden_dim, 2 * (self.trident_num_bins + 1))
        else:
            self.offset_proj = nn.Linear(hidden_dim, 2)
        self.start_boundary_proj = nn.Linear(hidden_dim, 1)
        self.end_boundary_proj = nn.Linear(hidden_dim, 1)
        if self.proposal_mode == "bmn":
            proposal_in_dim = hidden_dim * 3 + 1
            self.proposal_proj = nn.Sequential(
                nn.LayerNorm(proposal_in_dim),
                nn.Linear(proposal_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.proposal_proj = None
        if self.quality_mode == "dgqp":
            quality_stats_dim = 6
            self.quality_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim + quality_stats_dim),
                nn.Linear(hidden_dim + quality_stats_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.quality_proj = None

    @staticmethod
    def _boundary_neighbour_windows(
        boundary_logits: torch.Tensor,
        num_bins: int,
        *,
        pad_left: bool,
    ) -> torch.Tensor:
        if boundary_logits.ndim != 2:
            raise ValueError(f"`boundary_logits` must be [B,T], got {tuple(boundary_logits.shape)}")
        bsz, t_len = boundary_logits.shape
        if t_len <= 0:
            return boundary_logits.new_zeros((bsz, 0, int(num_bins) + 1))
        x = boundary_logits.unsqueeze(1)  # [B,1,T]
        if pad_left:
            x = F.pad(x, (int(num_bins), 0), mode="constant", value=0.0)
        else:
            x = F.pad(x, (0, int(num_bins)), mode="constant", value=0.0)
        return x.unfold(dimension=2, size=int(num_bins) + 1, step=1).squeeze(1)  # [B,T,K]

    def _decode_trident_offsets(
        self,
        offset_logits: torch.Tensor,
        range_start_logits: torch.Tensor,
        range_end_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if offset_logits.ndim != 3:
            raise ValueError(f"`offset_logits` must be [B,T,C], got {tuple(offset_logits.shape)}")
        bsz, t_len, chans = offset_logits.shape
        k = int(self.trident_num_bins) + 1
        if chans != 2 * k:
            raise ValueError(
                f"`offset_logits` channel dim must be {2 * k} for trident head, got {chans}"
            )
        if t_len <= 0:
            z = offset_logits.new_zeros((bsz, 0))
            return offset_logits.new_zeros((bsz, 0, 2)), z.unsqueeze(-1), z.unsqueeze(-1)
        base_logits = offset_logits.view(bsz, t_len, 2, k)
        left_base = base_logits[:, :, 0, :]
        right_base = base_logits[:, :, 1, :]

        left_neighbours = self._boundary_neighbour_windows(
            range_start_logits,
            self.trident_num_bins,
            pad_left=True,
        )
        left_neighbours = torch.flip(left_neighbours, dims=[-1])
        right_neighbours = self._boundary_neighbour_windows(
            range_end_logits,
            self.trident_num_bins,
            pad_left=False,
        )

        left_dist_logits = left_base + left_neighbours
        right_dist_logits = right_base + right_neighbours

        bin_values = torch.arange(k, device=offset_logits.device, dtype=torch.float32).view(1, 1, k)
        left_probs = torch.softmax(left_dist_logits.float(), dim=-1)
        right_probs = torch.softmax(right_dist_logits.float(), dim=-1)
        left_offsets = (left_probs * bin_values).sum(dim=-1).to(offset_logits.dtype)
        right_offsets = (right_probs * bin_values).sum(dim=-1).to(offset_logits.dtype)
        range_offsets = torch.stack((left_offsets, right_offsets), dim=-1)
        return range_offsets, left_dist_logits, right_dist_logits

    @staticmethod
    def _distribution_guided_quality_stats(
        left_dist_logits: torch.Tensor,
        right_dist_logits: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if left_dist_logits.ndim != 3 or right_dist_logits.ndim != 3:
            raise ValueError(
                "Distribution-guided quality stats require left/right logits with shape [B,T,K]."
            )
        left_prob = torch.softmax(left_dist_logits.float(), dim=-1)
        right_prob = torch.softmax(right_dist_logits.float(), dim=-1)
        left_top2 = torch.topk(left_prob, k=min(2, int(left_prob.shape[-1])), dim=-1).values
        right_top2 = torch.topk(right_prob, k=min(2, int(right_prob.shape[-1])), dim=-1).values
        left_top1 = left_top2[..., :1]
        right_top1 = right_top2[..., :1]
        left_margin = left_top1 - (
            left_top2[..., 1:2] if int(left_top2.shape[-1]) > 1 else torch.zeros_like(left_top1)
        )
        right_margin = right_top1 - (
            right_top2[..., 1:2] if int(right_top2.shape[-1]) > 1 else torch.zeros_like(right_top1)
        )
        left_entropy = -(left_prob * left_prob.clamp_min(float(eps)).log()).sum(dim=-1, keepdim=True)
        right_entropy = -(right_prob * right_prob.clamp_min(float(eps)).log()).sum(dim=-1, keepdim=True)
        return torch.cat(
            (left_top1, right_top1, left_margin, right_margin, left_entropy, right_entropy),
            dim=-1,
        ).to(dtype=left_dist_logits.dtype)

    @staticmethod
    def _interval_mean_features(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,H], got {tuple(x.shape)}")
        bsz, t_len, hid = x.shape
        if t_len <= 0:
            return x.new_zeros((bsz, 0, 0, hid))
        prefix = torch.cat([x.new_zeros((bsz, 1, hid)), x.cumsum(dim=1)], dim=1)
        idx = torch.arange(t_len, device=x.device, dtype=torch.long)
        start_idx = idx.view(1, t_len, 1).expand(bsz, t_len, t_len)
        end_idx = (idx.view(1, 1, t_len).expand(bsz, t_len, t_len) + 1)
        start_pref = torch.gather(
            prefix,
            dim=1,
            index=start_idx.reshape(bsz, -1).unsqueeze(-1).expand(-1, -1, hid),
        ).view(bsz, t_len, t_len, hid)
        end_pref = torch.gather(
            prefix,
            dim=1,
            index=end_idx.reshape(bsz, -1).unsqueeze(-1).expand(-1, -1, hid),
        ).view(bsz, t_len, t_len, hid)
        lengths = (
            idx.view(1, 1, t_len).expand(bsz, t_len, t_len)
            - idx.view(1, t_len, 1).expand(bsz, t_len, t_len)
            + 1
        ).clamp_min(1).to(dtype=x.dtype)
        return (end_pref - start_pref) / lengths.unsqueeze(-1)

    def _build_proposal_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.proposal_proj is None:
            raise RuntimeError("proposal_proj is required for proposal_mode != none")
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,H], got {tuple(x.shape)}")
        bsz, t_len, _ = x.shape
        if t_len <= 0:
            return x.new_zeros((bsz, 0, 0))
        start_feat = x.unsqueeze(2).expand(-1, -1, t_len, -1)
        end_feat = x.unsqueeze(1).expand(-1, t_len, -1, -1)
        interval_feat = self._interval_mean_features(x)
        idx = torch.arange(t_len, device=x.device, dtype=x.dtype)
        duration = (idx.view(1, 1, t_len) - idx.view(1, t_len, 1)).clamp_min(0.0)
        if t_len > 1:
            duration = duration / float(t_len - 1)
        duration = duration.expand(bsz, -1, -1).unsqueeze(-1)
        pair_feat = torch.cat((start_feat, end_feat, interval_feat, duration), dim=-1)
        proposal_logits = self.proposal_proj(pair_feat).squeeze(-1)
        valid_mask = torch.triu(torch.ones((t_len, t_len), device=x.device, dtype=torch.bool), diagonal=0)
        return proposal_logits.masked_fill(~valid_mask.unsqueeze(0), -20.0)

    def _build_condition(
        self,
        f_t: torch.Tensor,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.condition_mode == "none":
            return None
        return _build_trajectory_condition(
            condition_mode=self.condition_mode,
            ref_tensor=f_t.mean(dim=1),
            tool_embed=self.tool_embed,
            num_tool_classes=self.num_tool_classes,
            tool_id=tool_id,
            action_embed=self.action_embed,
            num_action_classes=self.num_action_classes,
            action_id=action_id,
            combo_embed=self.combo_embed,
            num_combo_classes=self.num_combo_classes,
            combo_id=combo_id,
        )

    def _apply_condition(self, f_t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return f_t
        assert self.cond_scale is not None and self.cond_shift is not None
        scale = 0.5 * torch.tanh(self.cond_scale(cond)).unsqueeze(1)
        shift = self.cond_shift(cond).unsqueeze(1)
        return f_t * (1.0 + scale) + shift

    def forward(
        self,
        f_t: torch.Tensor,
        score_f_t: Optional[torch.Tensor] = None,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # f_t: [B,T,D]
        score_ref = score_f_t if torch.is_tensor(score_f_t) else f_t
        cond = self._build_condition(
            score_ref,
            tool_id=tool_id,
            action_id=action_id,
            combo_id=combo_id,
        )
        if self.task_decouple_mode in ("score_loc", "score_loc_align"):
            loc_f_t = self._apply_condition(f_t, cond)
            cls_f_t = self._apply_condition(score_ref, cond)
            assert self.pre_cls is not None and self.pre_loc is not None
            x_cls = self.pre_cls(cls_f_t)
            x_loc = self.pre_loc(loc_f_t)
        else:
            loc_f_t = self._apply_condition(f_t, cond)
            assert self.pre is not None
            x_cls = self.pre(loc_f_t)
            x_loc = x_cls
        if self.task_decouple_mode == "score_loc_align":
            assert self.align_norm is not None and self.align_gate is not None
            align_in = torch.cat((x_loc, x_cls), dim=-1)
            align_gate = torch.sigmoid(self.align_gate(self.align_norm(align_in)))
            x_rank = x_loc + align_gate * (x_cls - x_loc)
        else:
            align_gate = None
            x_rank = x_cls
        actionness_logits = self.actionness_proj(x_rank).squeeze(-1)  # [B,T]
        range_start_logits = self.start_boundary_proj(x_loc).squeeze(-1)  # [B,T]
        range_end_logits = self.end_boundary_proj(x_loc).squeeze(-1)  # [B,T]
        proposal_logits = self._build_proposal_logits(x_cls) if self.proposal_mode != "none" else None
        if self.head_mode == "trident":
            offset_logits = self.offset_proj(x_loc)
            range_offsets, left_dist_logits, right_dist_logits = self._decode_trident_offsets(
                offset_logits=offset_logits,
                range_start_logits=range_start_logits,
                range_end_logits=range_end_logits,
            )
            out = {
                "range_actionness_logits": actionness_logits,
                "range_offsets": range_offsets,
                "range_start_logits": range_start_logits,
                "range_end_logits": range_end_logits,
                "range_left_dist_logits": left_dist_logits,
                "range_right_dist_logits": right_dist_logits,
            }
            if self.quality_proj is not None:
                quality_stats = self._distribution_guided_quality_stats(
                    left_dist_logits=left_dist_logits,
                    right_dist_logits=right_dist_logits,
                )
                quality_in = torch.cat((x_rank, quality_stats), dim=-1)
                out["range_quality_logits"] = self.quality_proj(quality_in).squeeze(-1)
            if proposal_logits is not None:
                out["range_proposal_logits"] = proposal_logits
            if torch.is_tensor(align_gate):
                out["range_task_align_gate"] = align_gate
            return out
        # Offsets are distances to start/end around the current point; must stay non-negative.
        range_offsets = F.softplus(self.offset_proj(x_loc))  # [B,T,2]
        out = {
            "range_actionness_logits": actionness_logits,
            "range_offsets": range_offsets,
            "range_start_logits": range_start_logits,
            "range_end_logits": range_end_logits,
        }
        if proposal_logits is not None:
            out["range_proposal_logits"] = proposal_logits
        if torch.is_tensor(align_gate):
            out["range_task_align_gate"] = align_gate
        return out


def decode_range_from_points(
    range_actionness_logits: torch.Tensor,
    range_offsets: torch.Tensor,
    range_quality_logits: Optional[torch.Tensor] = None,
    combo_duration_prior_mean: Optional[torch.Tensor] = None,
    combo_duration_prior_std: Optional[torch.Tensor] = None,
    combo_duration_prior_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    # range_actionness_logits: [B,T], range_offsets: [B,T,2]
    if range_actionness_logits.ndim != 2:
        raise ValueError(f"`range_actionness_logits` must be [B,T], got {tuple(range_actionness_logits.shape)}")
    if range_offsets.ndim != 3 or int(range_offsets.shape[-1]) != 2:
        raise ValueError(f"`range_offsets` must be [B,T,2], got {tuple(range_offsets.shape)}")
    bsz, t_len = range_actionness_logits.shape
    if int(range_offsets.shape[0]) != bsz or int(range_offsets.shape[1]) != t_len:
        raise ValueError(
            "Mismatched shape between range_actionness_logits and range_offsets: "
            f"{tuple(range_actionness_logits.shape)} vs {tuple(range_offsets.shape)}"
        )
    if t_len <= 0:
        z = range_actionness_logits.new_zeros((bsz,))
        return {
            "pred_action_start_norm": z,
            "pred_action_end_norm": z,
            "pred_action_start_t": z,
            "pred_action_end_t": z,
        }

    score = torch.sigmoid(range_actionness_logits.float())
    if torch.is_tensor(range_quality_logits):
        if range_quality_logits.shape != range_actionness_logits.shape:
            raise ValueError(
                f"`range_quality_logits` must match actionness shape {tuple(range_actionness_logits.shape)}, "
                f"got {tuple(range_quality_logits.shape)}"
            )
        score = score * torch.sigmoid(range_quality_logits.float())
    if torch.is_tensor(combo_duration_prior_mean) and torch.is_tensor(combo_duration_prior_std):
        if combo_duration_prior_mean.shape != (bsz,) or combo_duration_prior_std.shape != (bsz,):
            raise ValueError(
                "`combo_duration_prior_mean/std` must be [B], got "
                f"{tuple(combo_duration_prior_mean.shape)} and {tuple(combo_duration_prior_std.shape)}"
            )
        if t_len > 1:
            pred_duration_norm = range_offsets.float().sum(dim=-1) / float(t_len - 1)
        else:
            pred_duration_norm = range_offsets.float().sum(dim=-1).new_zeros((bsz, t_len))
        mu = combo_duration_prior_mean.float().unsqueeze(1)
        sigma = combo_duration_prior_std.float().clamp_min(1e-4).unsqueeze(1)
        z = (pred_duration_norm - mu) / sigma
        duration_prior = torch.exp(-0.5 * z.square()).clamp_min(1e-6)
        score = score * duration_prior.pow(float(max(combo_duration_prior_weight, 0.0)))
    point_idx = score.argmax(dim=1)  # [B]
    gather_idx = point_idx.view(-1, 1, 1).expand(-1, 1, 2)
    picked_offsets = range_offsets.gather(dim=1, index=gather_idx).squeeze(1)  # [B,2]
    left = picked_offsets[:, 0]
    right = picked_offsets[:, 1]

    point_t = point_idx.to(range_offsets.dtype)
    start_t = point_t - left
    end_t = point_t + right
    if t_len > 1:
        max_t = float(t_len - 1)
        start_t = start_t.clamp(0.0, max_t)
        end_t = end_t.clamp(0.0, max_t)
        end_t = torch.maximum(end_t, start_t)
        start_norm = start_t / max_t
        end_norm = end_t / max_t
    else:
        start_t = start_t.new_zeros(start_t.shape)
        end_t = end_t.new_zeros(end_t.shape)
        start_norm = start_t
        end_norm = end_t

    return {
        "pred_action_start_norm": start_norm.clamp(0.0, 1.0),
        "pred_action_end_norm": end_norm.clamp(0.0, 1.0),
        "pred_action_start_t": start_t,
        "pred_action_end_t": end_t,
    }
