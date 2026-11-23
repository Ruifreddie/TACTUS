"""Affordance model and its transitive dependencies, extracted unchanged."""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

def parse_float_csv(value: str | None) -> list[float]:
    if value is None:
        return []
    vals: list[float] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals

def parse_int_csv(value: str | None) -> list[int]:
    if value is None:
        return []
    vals: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals

def token_affinity_variant_count(extra_top_fracs: str | None, extra_size_scales: str | None) -> int:
    return 1 + max(len(parse_float_csv(extra_top_fracs)), len(parse_float_csv(extra_size_scales)))

def parse_template_sizes_px(value: str | None, mask_res: int = 64) -> list[tuple[float, float]]:
    if value is None:
        return []
    sizes: list[tuple[float, float]] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"template size must be WxH pixels, got {part!r}")
        a, b = part.split("x", 1)
        sizes.append((float(a) / float(mask_res), float(b) / float(mask_res)))
    return sizes

def topology_contact_template_count(sizes_px: str | None, offsets_px: str | None, mask_res: int = 64) -> int:
    sizes = parse_template_sizes_px(sizes_px, mask_res=mask_res)
    offsets = parse_float_csv(offsets_px)
    if not sizes or not offsets:
        return 0
    # Hook: hook-tip templates at all offsets plus a perpendicular zero-offset
    # family. Non-hook tools: tip-pair, shaft, and two endpoint families.
    hook_count = 2 * len(sizes) * (len(offsets) + 1)
    nonhook_count = 2 * len(sizes) * (2 * len(offsets) + 6)
    return max(hook_count, nonhook_count)

def parse_combo_index_allowlist(value: str | None, count: int, num_combos: int = 17) -> torch.Tensor | None:
    if value is None or not str(value).strip() or int(count) <= 0:
        return None
    raw = str(value).strip()
    path = Path(raw)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)
    mask = torch.zeros((int(num_combos), int(count)), dtype=torch.bool)

    def _indices(obj: Any) -> list[int]:
        if isinstance(obj, dict) and "indices" in obj:
            obj = obj["indices"]
        return [int(x) for x in obj]

    if isinstance(payload, dict):
        default_indices = payload.get("default")
        if default_indices is not None:
            idx = [i for i in _indices(default_indices) if 0 <= i < int(count)]
            if idx:
                mask[:, idx] = True
        combo_payload = payload.get("combo_indices", payload)
        if isinstance(combo_payload, dict):
            for combo_key, combo_indices in combo_payload.items():
                if str(combo_key) in {"default", "combo_indices", "meta"}:
                    continue
                combo = int(combo_key)
                if not 0 <= combo < int(num_combos):
                    continue
                idx = [i for i in _indices(combo_indices) if 0 <= i < int(count)]
                if idx:
                    mask[combo, :] = False
                    mask[combo, idx] = True
    elif isinstance(payload, list):
        idx = [i for i in _indices(payload) if 0 <= i < int(count)]
        if idx:
            mask[:, idx] = True
    else:
        raise ValueError(f"Unsupported combo allowlist payload: {type(payload).__name__}")

    empty = ~mask.any(dim=1)
    if bool(empty.any()):
        mask[empty, :] = True
    return mask

def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(dim=-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1).clamp(0.0, 1.0)

def xyxy_to_cxcywh_torch(box: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = box.unbind(dim=-1)
    return torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0), (y1 - y0)], dim=-1).clamp(0.0, 1.0)

def quad_to_xyxy(quad: torch.Tensor, min_size: float = 0.02) -> torch.Tensor:
    x0 = quad[..., 0].min(dim=-1).values
    y0 = quad[..., 1].min(dim=-1).values
    x1 = quad[..., 0].max(dim=-1).values
    y1 = quad[..., 1].max(dim=-1).values
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (x1 - x0).clamp_min(float(min_size))
    h = (y1 - y0).clamp_min(float(min_size))
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1).clamp(0.0, 1.0)

def roll_quad_start_min_sum(quad: torch.Tensor) -> torch.Tensor:
    """Rotate cyclic corner order so the first point is roughly top-left."""
    if quad.shape[-2:] != (4, 2):
        return quad
    start = quad.sum(dim=-1).argmin(dim=-1)
    base = torch.arange(4, device=quad.device, dtype=start.dtype)
    order = (start.unsqueeze(-1) + base.view(*([1] * start.ndim), 4)) % 4
    return quad.gather(-2, order.unsqueeze(-1).expand(*order.shape, 2))

def oriented_rect_to_quad(center: torch.Tensor, size: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert center/size/axis to a cyclic ordered normalized quad."""
    axis = F.normalize(axis, dim=-1, eps=1e-6)
    perp = torch.stack([-axis[..., 1], axis[..., 0]], dim=-1)
    half_u = axis * size[..., :1] * 0.5
    half_v = perp * size[..., 1:2] * 0.5
    quad = torch.stack(
        [
            center - half_u - half_v,
            center + half_u - half_v,
            center + half_u + half_v,
            center - half_u + half_v,
        ],
        dim=-2,
    )
    return roll_quad_start_min_sum(quad).clamp(0.0, 1.0)

class H161RoiCondLite(nn.Module):
    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.region_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.register_buffer("grid_xy", self._make_grid(24, 24), persistent=False)

    @staticmethod
    def _make_grid(h: int, w: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / h, 1.0 - 0.5 / h, h),
            torch.linspace(0.5 / w, 1.0 - 0.5 / w, w),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=0)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # tokens: B,T=8,H=24,W=24,C=1024
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = torch.cat(feats, dim=2).reshape(b * t, -1, h, w)
        y = self.spatial(y)
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled)).clamp(0.02, 0.95)
        box8_cxcywh = torch.cat([centers8, size8], dim=-1)
        box16_cxcywh = F.interpolate(box8_cxcywh.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        box16_cxcywh = box16_cxcywh.clamp(0.0, 1.0)
        return {
            "boxes_cxcywh": box16_cxcywh,
            "boxes_xyxy": cxcywh_to_xyxy(box16_cxcywh),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "region_logits_8": self.region_head(y).view(b, t, h, w),
        }

def make_rgb_sidecar_encoder(hidden_dim: int) -> nn.Sequential:
    c = int(hidden_dim)
    rgb_c1 = max(32, c // 4)
    rgb_c2 = max(64, c // 2)
    return nn.Sequential(
        nn.Conv2d(3, rgb_c1, 3, stride=2, padding=1),
        nn.GroupNorm(4, rgb_c1),
        nn.SiLU(),
        nn.Conv2d(rgb_c1, rgb_c2, 3, stride=2, padding=1),
        nn.GroupNorm(8, rgb_c2),
        nn.SiLU(),
        nn.Conv2d(rgb_c2, c, 3, padding=1),
        nn.GroupNorm(8, c),
        nn.SiLU(),
    )

def encode_rgb_sidecar_8(
    rgb_encoder: nn.Module,
    rgb_16: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rgb_16 is None:
        raise ValueError("RGB sidecar fusion is enabled but rgb_16 was not provided")
    b = rgb_16.shape[0]
    rgb = rgb_16.to(dtype=dtype).reshape(b, 8, 2, 3, rgb_16.shape[-2], rgb_16.shape[-1]).mean(dim=2)
    y = rgb_encoder(rgb.reshape(b * 8, 3, rgb.shape[-2], rgb.shape[-1]))
    if y.shape[-2:] != (24, 24):
        y = F.interpolate(y, size=(24, 24), mode="bilinear", align_corners=False)
    return y.view(b, 8, -1, 24, 24)

class H161QuadTemporalRegion(nn.Module):
    """Predict an ordered 4-point operation region, then derive the Stage1 ROI.

    This keeps the output compatible with the bbox metric while giving the head
    an explicit intra-frame region structure and a 16-frame temporal sequence.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        tool_mask_spatial: bool = True,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_mask_spatial = bool(use_tool_mask) and bool(tool_mask_spatial)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if self.use_tool_mask_spatial else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        self.support_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        seq_in_dim = 3 * c + 4 + (c if self.use_tool_mask else 0)
        self.seq_in = nn.Sequential(
            nn.LayerNorm(seq_in_dim),
            nn.Linear(seq_in_dim, c),
            nn.SiLU(),
        )
        self.box_delta_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        self.point_offset_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 8),
        )
        nn.init.zeros_(self.box_delta_head[-1].weight)
        nn.init.zeros_(self.box_delta_head[-1].bias)
        nn.init.zeros_(self.point_offset_head[-1].weight)
        nn.init.zeros_(self.point_offset_head[-1].bias)
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        # Return [cx, cy, area, sx, sy] per 8 pooled frame.
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mask.sum(dim=(-1, -2)) <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask_spatial:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        corner_logits_8 = self.corner_heatmap_head(y).view(b, t, 4, h, w)
        support_logits_8 = self.support_head(y).view(b, t, h, w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.03, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)

        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        seq_parts = [visual16, role16 + cond16, kpt16]
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8)
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            seq_parts.append(self.tool_mask_geom_mlp(mask_geom16))
        seq_parts.append(coarse16.detach())
        seq = self.seq_in(torch.cat(seq_parts, dim=-1))
        seq = self.temporal(seq)

        box_delta = self.box_delta_head(seq)
        dc = torch.tanh(box_delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(box_delta[..., 2:]) * 0.5
        center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.03, 0.95)

        corner_offsets = 0.35 * torch.tanh(self.point_offset_head(seq).view(b, 16, 4, 2))
        base = self.base_corners.to(device=seq.device, dtype=seq.dtype).view(1, 1, 4, 2)
        quad = center[:, :, None, :] + (base + corner_offsets) * size[:, :, None, :]
        quad = quad.clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "corner_logits_8": corner_logits_8,
            "support_logits_8": support_logits_8,
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "quad_xy": quad,
        }

class H191QuadLocalPointTemporalRegion(nn.Module):
    """H174-style quad stage followed by noisy local point refinement.

    Stage 1 predicts a structured quad/box with the existing quad-temporal head.
    Stage 2 treats the four predicted vertices as point queries, samples local
    V-JEPA/tool-mask features around each vertex, and refines them with
    same-frame and same-point temporal attention.  Training-time reference noise
    sits between the two stages so the refiner cannot learn a trivial copy.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        dn_query_noise_scale: float = 0.0,
        dn_query_groups: int = 1,
        dn_query_noise_min_scale: float = 1.0,
        use_proposal_oracle_refiner: bool = False,
        proposal_oracle_refiner_noise_scale: float = 0.0,
        stage2_bridge_noise_mode: str = "local_only",
        use_tool_trajectory_bias: bool = False,
        use_tool_trajectory_attn: bool = False,
        use_tool_corner_query_bias: bool = False,
        stage1_drop_tool_keypoints: bool = False,
        detach_stage1_refiner: bool = False,
        local_grid_refine: bool = False,
        local_grid_refine_grid_size: int = 5,
        local_grid_refine_scale: float = 1.35,
        local_grid_refine_temperature: float = 0.20,
        local_grid_refine_prior_scale: float = 6.0,
        local_grid_refine_prior_sigma: float = 0.28,
        local_grid_refine_detach_base: bool = False,
        local_grid_final_refine: bool = True,
        local_grid_confidence_score: bool = False,
        local_grid_confidence_scale: float = 0.35,
        local_grid_confidence_detach: bool = False,
        candidate_mask_quality: bool = False,
        candidate_mask_grid_size: int = 7,
        candidate_mask_score_scale: float = 0.0,
        candidate_mask_score_detach: bool = False,
        use_rgb_sidecar: bool = False,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.use_rgb_sidecar = bool(use_rgb_sidecar)
        self.use_tool_trajectory_bias = bool(use_tool_trajectory_bias) and self.use_tool_keypoints
        self.use_tool_trajectory_attn = bool(use_tool_trajectory_attn) and self.use_tool_keypoints
        self.use_tool_corner_query_bias = bool(use_tool_corner_query_bias) and self.use_tool_keypoints
        self.stage1_drop_tool_keypoints = bool(stage1_drop_tool_keypoints)
        self.detach_stage1_refiner = bool(detach_stage1_refiner)
        self.local_grid_refine = bool(local_grid_refine)
        self.local_grid_refine_grid_size = max(int(local_grid_refine_grid_size), 3)
        if self.local_grid_refine_grid_size % 2 == 0:
            self.local_grid_refine_grid_size += 1
        self.local_grid_refine_scale = max(float(local_grid_refine_scale), 0.25)
        self.local_grid_refine_temperature = max(float(local_grid_refine_temperature), 1e-3)
        self.local_grid_refine_prior_scale = float(local_grid_refine_prior_scale)
        self.local_grid_refine_prior_sigma = max(float(local_grid_refine_prior_sigma), 1e-3)
        self.local_grid_refine_detach_base = bool(local_grid_refine_detach_base)
        self.local_grid_final_refine = bool(local_grid_final_refine)
        stage1_use_tool_keypoints = self.use_tool_keypoints and not self.stage1_drop_tool_keypoints
        self.stage1 = H161QuadTemporalRegion(
            hidden_dim=c,
            use_tool_mask=self.use_tool_mask,
            use_tool_keypoints=stage1_use_tool_keypoints,
            residual_scale=float(residual_scale),
            tool_mask_spatial=self.use_tool_mask,
        )
        self.residual_scale = float(residual_scale)
        self.decoder_layers = max(int(decoder_layers), 1)
        self.query_noise_scale = float(query_noise_scale)
        self.dn_query_noise_scale = float(dn_query_noise_scale)
        self.dn_query_groups = max(int(dn_query_groups), 1)
        self.dn_query_noise_min_scale = float(dn_query_noise_min_scale)
        self.use_proposal_oracle_refiner = bool(use_proposal_oracle_refiner)
        self.proposal_oracle_refiner_noise_scale = float(proposal_oracle_refiner_noise_scale)
        if stage2_bridge_noise_mode not in {"local_only", "replace"}:
            raise ValueError(f"unknown stage2_bridge_noise_mode={stage2_bridge_noise_mode!r}")
        self.stage2_bridge_noise_mode = str(stage2_bridge_noise_mode)

        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_traj_mlp = nn.Sequential(nn.LayerNorm(16 * 9), nn.Linear(16 * 9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_traj_token_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_corner_ref_mlp = nn.Sequential(nn.Linear(2, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.rgb_encoder = make_rgb_sidecar_encoder(c) if self.use_rgb_sidecar else None
        spatial_in_ch = (
            c
            + (c if self.use_rgb_sidecar else 0)
            + 2
            + (1 if self.use_tool_mask else 0)
            + (3 if self.use_tool_keypoints else 0)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(spatial_in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )

        self.point_index = nn.Parameter(torch.zeros(1, 1, 4, c))
        self.time_pos = nn.Parameter(torch.zeros(1, 16, 1, c))
        self.tool_time_pos = nn.Parameter(torch.zeros(1, 16, c))
        self.point_ref_mlp = nn.Sequential(nn.Linear(2, c), nn.SiLU(), nn.Linear(c, c))
        self.query_norm = nn.LayerNorm(c)
        self.local_point_proj = nn.Sequential(
            nn.LayerNorm(3 * c),
            nn.Linear(3 * c, c),
            nn.SiLU(),
            nn.Linear(c, c),
        )
        if self.local_grid_refine:
            self.local_grid_point_proj = nn.Sequential(
                nn.LayerNorm(3 * c + 2),
                nn.Linear(3 * c + 2, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.local_grid_corner_embed = nn.Parameter(torch.zeros(1, 1, 4, c))
            self.local_grid_corner_head = nn.Sequential(
                nn.LayerNorm(2 * c + 2),
                nn.Linear(2 * c + 2, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.trunc_normal_(self.local_grid_corner_embed, std=0.02)
            nn.init.zeros_(self.local_grid_corner_head[-1].weight)
            nn.init.zeros_(self.local_grid_corner_head[-1].bias)
        self.frame_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.frame_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.tool_query_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.tool_memory_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.tool_traj_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Dropout(0.10), nn.Linear(4 * c, c))
            for _ in range(self.decoder_layers)
        ])
        self.point_delta_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
            for _ in range(self.decoder_layers)
        ])
        for head in self.point_delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.trunc_normal_(self.point_index, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.tool_time_pos, std=0.02)
        sample_offsets = torch.tensor(
            [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("point_sample_offsets", sample_offsets, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)
        self.point_sample_radius = 1.25 / 24.0

    def _tool_anchor_quad(self, tool_kpt_vec_16: torch.Tensor) -> torch.Tensor:
        b = tool_kpt_vec_16.shape[0]
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
        base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
        base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
        fallback = torch.zeros_like(base_axis)
        fallback[..., 0] = 1.0
        axis_norm = base_axis.norm(dim=-1, keepdim=True)
        base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
        kpt_dist = (kpt_xy - base_center[:, :, None, :]).norm(dim=-1)
        tool_extent = (kpt_dist * kpt_vis.squeeze(-1)).sum(dim=2) / vis_sum.squeeze(-1)
        base_size = torch.stack(
            [
                (0.75 * tool_extent + 0.060).clamp(0.045, 0.24),
                (0.45 * tool_extent + 0.045).clamp(0.035, 0.18),
            ],
            dim=-1,
        )
        return oriented_rect_to_quad(base_center, base_size, base_axis).clamp(0.0, 1.0)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def _sample_point_features(self, feat16: torch.Tensor, ref_xy: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feat16.shape
        p = ref_xy.shape[2]
        offsets = (
            self.point_sample_offsets.to(device=feat16.device, dtype=feat16.dtype)
            * float(self.point_sample_radius)
        )
        pts = ref_xy[:, :, :, None, :] + offsets.view(1, 1, 1, -1, 2)
        grid = (pts.reshape(b * t, p * offsets.shape[0], 1, 2) * 2.0 - 1.0).clamp(-1.15, 1.15)
        sampled = F.grid_sample(
            feat16.reshape(b * t, c, h, w).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).to(feat16.dtype)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, p, offsets.shape[0], c)
        center = sampled[:, :, :, 0]
        mean = sampled.mean(dim=3)
        maxv = sampled.max(dim=3).values
        return torch.cat([center, mean, maxv], dim=-1)

    def _local_grid_points(
        self,
        quad: torch.Tensor,
        grid_size: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a rotated local search grid around an ordered quad."""
        orig_shape = quad.shape[:-2]
        q = quad.reshape(-1, 4, 2)
        center = q.mean(dim=-2)
        dirs, lens = quad_axes(q)
        coords = torch.linspace(
            -0.5 * float(scale),
            0.5 * float(scale),
            int(grid_size),
            device=quad.device,
            dtype=quad.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rel = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        pts = (
            center[:, None, :]
            + dirs[:, None, 0, :] * (rel[None, :, 0:1] * lens[:, None, 0:1])
            + dirs[:, None, 1, :] * (rel[None, :, 1:2] * lens[:, None, 1:2])
        )
        pts = pts.clamp(0.0, 1.0)
        return pts.reshape(*orig_shape, grid_size * grid_size, 2), rel.reshape(1, 1, grid_size * grid_size, 2)

    def _local_grid_refine_quad(
        self,
        feat16_map: torch.Tensor,
        base_quad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t = base_quad.shape[:2]
        grid_size = int(self.local_grid_refine_grid_size)
        search_quad = base_quad.detach() if self.local_grid_refine_detach_base else base_quad
        grid_pts, rel = self._local_grid_points(
            search_quad,
            grid_size=grid_size,
            scale=float(self.local_grid_refine_scale),
        )
        p = grid_pts.shape[2]
        point_feat = self._sample_point_features(feat16_map, grid_pts)
        rel_feat = rel.to(device=point_feat.device, dtype=point_feat.dtype).expand(b, t, -1, -1)
        grid_feat = self.local_grid_point_proj(torch.cat([point_feat, rel_feat], dim=-1))
        corner_feat = self.local_grid_corner_embed.to(device=grid_feat.device, dtype=grid_feat.dtype)
        corner_feat = corner_feat.expand(b, t, -1, -1)
        corner_grid_feat = grid_feat[:, :, None, :, :].expand(-1, -1, 4, -1, -1)
        corner_token = corner_feat[:, :, :, None, :].expand(-1, -1, -1, p, -1)
        grid_xy = grid_pts[:, :, None, :, :].expand(-1, -1, 4, -1, -1)
        logits = self.local_grid_corner_head(
            torch.cat([corner_grid_feat, corner_token, grid_xy], dim=-1)
        ).squeeze(-1)
        if self.local_grid_refine_prior_scale != 0.0:
            prior_quad = search_quad.to(device=grid_pts.device, dtype=grid_pts.dtype)
            _dirs, lens = quad_axes(prior_quad.reshape(-1, 4, 2))
            side = lens.mean(dim=-1).reshape(b, t, 1, 1).clamp_min(1e-4)
            prior_dist2 = (grid_xy - prior_quad[:, :, :, None, :]).square().sum(dim=-1)
            sigma = float(self.local_grid_refine_prior_sigma) * side
            corner_prior = -prior_dist2 / (2.0 * sigma.square().clamp_min(1e-8))
            logits = logits + float(self.local_grid_refine_prior_scale) * corner_prior
        probs = F.softmax(logits / float(self.local_grid_refine_temperature), dim=-1)
        quad = (probs[..., None] * grid_xy).sum(dim=-2).clamp(0.0, 1.0)
        return quad, logits, grid_pts

    def _maybe_perturb_points(self, ref_xy: torch.Tensor, step_scale: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.query_noise_scale <= 0.0:
            return ref_xy
        noise = (torch.rand_like(ref_xy) * 2.0 - 1.0) * self.query_noise_scale
        return (ref_xy + noise * step_scale.detach()).clamp(0.0, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        dn_quad_xy: torch.Tensor | None = None,
        rgb_16: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stage1_kwargs: dict[str, torch.Tensor] = {
            "tokens": tokens,
            "role_features_16": role_features_16,
            "tool_mask_8": tool_mask_8,
            "tool_kpt_heatmaps_8": tool_kpt_heatmaps_8,
            "tool_kpt_vec_16": tool_kpt_vec_16,
            "tool_id": tool_id,
            "act_id": act_id,
            "combo_id": combo_id,
        }
        if getattr(self.stage1, "use_rgb_sidecar", False):
            stage1_kwargs["rgb_16"] = rgb_16
        stage1 = self.stage1(**stage1_kwargs)
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        spatial_parts = [x]
        if self.use_rgb_sidecar:
            if self.rgb_encoder is None:
                raise RuntimeError("RGB sidecar is enabled without an encoder")
            spatial_parts.append(encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype))
        spatial_parts.append(grid)
        if self.use_tool_mask:
            spatial_parts.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            spatial_parts.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(spatial_parts, dim=2).reshape(b * t8, -1, h, w))
        mem8 = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        mem16_chw = F.interpolate(mem8, size=(16, h, w), mode="trilinear", align_corners=True)
        feat16_map = mem16_chw.permute(0, 2, 1, 3, 4).contiguous()
        mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b, 16, h * w, -1)

        query_base = (
            self.role_mlp(role_features_16)
            + z[:, None, :].expand(-1, 16, -1)
        )
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            query_base = query_base + self.tool_mask_geom_mlp(mask_geom16)
        if self.use_tool_keypoints:
            query_base = query_base + self.tool_kpt_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype))
            if self.use_tool_trajectory_bias:
                traj_bias = self.tool_traj_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype).reshape(b, 16 * 9))
                query_base = query_base + traj_bias[:, None, :]
        tool_memory = None
        if self.use_tool_trajectory_attn:
            tool_memory = self.tool_traj_token_mlp(
                torch.cat(
                    [
                        tool_kpt_vec_16.to(device=device, dtype=dtype),
                        role_features_16.to(device=device, dtype=dtype),
                    ],
                    dim=-1,
                )
            )
            tool_memory = tool_memory + z[:, None, :] + self.tool_time_pos.to(device=device, dtype=dtype)
        corner_query_bias = 0.0
        if self.use_tool_corner_query_bias:
            tool_anchor_quad = self._tool_anchor_quad(tool_kpt_vec_16.to(device=device, dtype=dtype))
            corner_query_bias = self.tool_corner_ref_mlp(tool_anchor_quad)
        corner_query_bias_all = (
            corner_query_bias[:, :, None, :, :] if isinstance(corner_query_bias, torch.Tensor) else 0.0
        )

        ref_clean = stage1["quad_xy"].clamp(0.0, 1.0)
        initial_box_xyxy = stage1["boxes_xyxy"]
        if self.detach_stage1_refiner:
            ref_clean = ref_clean.detach()
            initial_box_xyxy = initial_box_xyxy.detach()
        initial_wh = xyxy_to_cxcywh_torch(initial_box_xyxy)[..., 2:].detach()
        step_scale = (self.residual_scale * initial_wh[:, :, None, :]).clamp(0.01, 0.50)

        if self.stage2_bridge_noise_mode == "replace":
            # H193/H194: corrupt the actual stage1-to-stage2 bridge.  This is
            # kept as an option because it is useful diagnostically, but the
            # default follows H192: clean main query plus noisy local sampling.
            ref = self._maybe_perturb_points(ref_clean, step_scale)
        else:
            ref = ref_clean
        q = (
            query_base[:, :, None, :]
            + self.point_index.to(device=device, dtype=dtype)
            + self.time_pos.to(device=device, dtype=dtype)
            + self.point_ref_mlp(ref)
            + corner_query_bias
        )
        q = self.query_norm(q)

        layer_boxes = []
        layer_xyxy = []
        layer_quads = []
        mem_bt = mem16.reshape(b * 16, h * w, -1)
        for i in range(self.decoder_layers):
            sample_ref = ref
            if self.stage2_bridge_noise_mode == "local_only":
                sample_ref = self._maybe_perturb_points(ref, step_scale)
            q = q + self.local_point_proj(self._sample_point_features(feat16_map, sample_ref))

            q_frame = q.reshape(b * 16, 4, -1)
            frame_out, _ = self.frame_attn[i](
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                need_weights=False,
            )
            q = (q_frame + frame_out).reshape(b, 16, 4, -1)

            q_tem = q.permute(0, 2, 1, 3).reshape(b * 4, 16, -1)
            tem_out, _ = self.temporal_attn[i](
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                need_weights=False,
            )
            q = (q_tem + tem_out).reshape(b, 4, 16, -1).permute(0, 2, 1, 3)

            if tool_memory is not None:
                q_tool = q.reshape(b, 16 * 4, -1)
                tool_out, _ = self.tool_traj_attn[i](
                    self.tool_query_norms[i](q_tool),
                    self.tool_memory_norms[i](tool_memory),
                    self.tool_memory_norms[i](tool_memory),
                    need_weights=False,
                )
                q = (q_tool + tool_out).reshape(b, 16, 4, -1)

            q_ref = q + self.point_ref_mlp(ref)
            cross_q = self.cross_norms[i](q_ref).reshape(b * 16, 4, -1)
            mem_norm = self.memory_norms[i](mem_bt)
            cross_out, _ = self.cross_attn[i](cross_q, mem_norm, mem_norm, need_weights=False)
            q = q + cross_out.reshape(b, 16, 4, -1)
            q = q + self.ffns[i](self.ffn_norms[i](q))

            delta = self.point_delta_heads[i](q)
            ref_new = (ref + torch.tanh(delta) * step_scale).clamp(0.0, 1.0)
            layer_quads.append(ref_new)
            box_xyxy = quad_to_xyxy(ref_new, min_size=0.02)
            layer_xyxy.append(box_xyxy)
            layer_boxes.append(xyxy_to_cxcywh_torch(box_xyxy))
            ref = ref_new.detach() if i + 1 < self.decoder_layers else ref_new

        final_quad = layer_quads[-1]
        final_xyxy = layer_xyxy[-1]
        local_grid_corner_logits = None
        local_grid_points = None
        if self.local_grid_refine and self.local_grid_final_refine:
            final_quad, local_grid_corner_logits, local_grid_points = self._local_grid_refine_quad(
                feat16_map,
                final_quad,
            )
            final_xyxy = quad_to_xyxy(final_quad, min_size=0.02)
        out = dict(stage1)
        # Keep optional Stage1 OICR logits only when they still line up with
        # the proposal bank used by proposal losses. This is the normal
        # non-recollect path; stale tensors are dropped defensively.
        if (
            "proposal_base_logits" in out
            and "proposal_logits" in out
            and out["proposal_base_logits"].shape != out["proposal_logits"].shape
        ):
            out.pop("proposal_base_logits", None)
        if (
            "proposal_oicr_logits" in out
            and "proposal_logits" in out
            and out["proposal_oicr_logits"].shape != out["proposal_logits"].shape
        ):
            out.pop("proposal_oicr_logits", None)
        out["boxes_xyxy"] = final_xyxy
        out["boxes_cxcywh"] = xyxy_to_cxcywh_torch(final_xyxy)
        out["coarse_xyxy"] = initial_box_xyxy
        out["coarse_cxcywh"] = xyxy_to_cxcywh_torch(initial_box_xyxy)
        out["decoder_layer_xyxy"] = torch.stack(layer_xyxy, dim=0)
        out["decoder_layer_cxcywh"] = torch.stack(layer_boxes, dim=0)
        out["quad_xy"] = final_quad
        if local_grid_corner_logits is not None and local_grid_points is not None:
            out["local_grid_corner_logits"] = local_grid_corner_logits
            out["local_grid_points"] = local_grid_points
            out["local_grid_base_quad"] = layer_quads[-1]
        if dn_quad_xy is not None and self.training and self.dn_query_noise_scale > 0.0:
            gt_ref = dn_quad_xy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            gt_xyxy = quad_to_xyxy(gt_ref, min_size=0.02)
            gt_wh = xyxy_to_cxcywh_torch(gt_xyxy)[..., 2:].detach()
            dn_groups = max(int(self.dn_query_groups), 1)
            dn_b = b * dn_groups
            if dn_groups > 1:
                gt_ref_dn = (
                    gt_ref[:, None]
                    .expand(b, dn_groups, 16, 4, 2)
                    .reshape(dn_b, 16, 4, 2)
                    .contiguous()
                )
                gt_wh_dn = (
                    gt_wh[:, None]
                    .expand(b, dn_groups, 16, 2)
                    .reshape(dn_b, 16, 2)
                    .contiguous()
                )
                noise_scales = torch.linspace(
                    float(self.dn_query_noise_min_scale),
                    1.0,
                    steps=dn_groups,
                    device=device,
                    dtype=dtype,
                ).clamp_min(0.0)
                noise_scales = noise_scales.view(1, dn_groups, 1, 1, 1).expand(b, dn_groups, 16, 4, 2)
                noise_scales = noise_scales.reshape(dn_b, 16, 4, 2)
                query_base_dn = (
                    query_base[:, None]
                    .expand(b, dn_groups, 16, -1)
                    .reshape(dn_b, 16, -1)
                    .contiguous()
                )
                feat16_map_dn = (
                    feat16_map[:, None]
                    .expand(b, dn_groups, 16, -1, h, w)
                    .reshape(dn_b, 16, -1, h, w)
                    .contiguous()
                )
                mem_bt_dn = (
                    mem16[:, None]
                    .expand(b, dn_groups, 16, h * w, -1)
                    .reshape(dn_b * 16, h * w, -1)
                    .contiguous()
                )
                if tool_memory is not None:
                    tool_memory_dn = (
                        tool_memory[:, None]
                        .expand(b, dn_groups, 16, -1)
                        .reshape(dn_b, 16, -1)
                        .contiguous()
                    )
                else:
                    tool_memory_dn = None
                if isinstance(corner_query_bias, torch.Tensor):
                    corner_query_bias_dn = (
                        corner_query_bias[:, None]
                        .expand(b, dn_groups, 16, 4, -1)
                        .reshape(dn_b, 16, 4, -1)
                        .contiguous()
                    )
                else:
                    corner_query_bias_dn = 0.0
            else:
                gt_ref_dn = gt_ref
                gt_wh_dn = gt_wh
                noise_scales = 1.0
                query_base_dn = query_base
                feat16_map_dn = feat16_map
                mem_bt_dn = mem_bt
                tool_memory_dn = tool_memory
                corner_query_bias_dn = corner_query_bias
            dn_step_scale = (self.residual_scale * gt_wh_dn[:, :, None, :]).clamp(0.01, 0.50)
            dn_noise = (
                (torch.rand_like(gt_ref_dn) * 2.0 - 1.0)
                * float(self.dn_query_noise_scale)
                * noise_scales
            )
            dn_ref = (gt_ref_dn + dn_noise * dn_step_scale.detach()).clamp(0.0, 1.0)
            dn_q = (
                query_base_dn[:, :, None, :]
                + self.point_index.to(device=device, dtype=dtype)
                + self.time_pos.to(device=device, dtype=dtype)
                + self.point_ref_mlp(dn_ref)
                + corner_query_bias_dn
            )
            dn_q = self.query_norm(dn_q)
            dn_layer_boxes = []
            dn_layer_xyxy = []
            dn_layer_quads = []
            for i in range(self.decoder_layers):
                dn_q = dn_q + self.local_point_proj(self._sample_point_features(feat16_map_dn, dn_ref))

                dn_q_frame = dn_q.reshape(dn_b * 16, 4, -1)
                dn_frame_out, _ = self.frame_attn[i](
                    self.frame_norms[i](dn_q_frame),
                    self.frame_norms[i](dn_q_frame),
                    self.frame_norms[i](dn_q_frame),
                    need_weights=False,
                )
                dn_q = (dn_q_frame + dn_frame_out).reshape(dn_b, 16, 4, -1)

                dn_q_tem = dn_q.permute(0, 2, 1, 3).reshape(dn_b * 4, 16, -1)
                dn_tem_out, _ = self.temporal_attn[i](
                    self.temporal_norms[i](dn_q_tem),
                    self.temporal_norms[i](dn_q_tem),
                    self.temporal_norms[i](dn_q_tem),
                    need_weights=False,
                )
                dn_q = (dn_q_tem + dn_tem_out).reshape(dn_b, 4, 16, -1).permute(0, 2, 1, 3)

                if tool_memory_dn is not None:
                    dn_q_tool = dn_q.reshape(dn_b, 16 * 4, -1)
                    dn_tool_out, _ = self.tool_traj_attn[i](
                        self.tool_query_norms[i](dn_q_tool),
                        self.tool_memory_norms[i](tool_memory_dn),
                        self.tool_memory_norms[i](tool_memory_dn),
                        need_weights=False,
                    )
                    dn_q = (dn_q_tool + dn_tool_out).reshape(dn_b, 16, 4, -1)

                dn_q_ref = dn_q + self.point_ref_mlp(dn_ref)
                dn_cross_q = self.cross_norms[i](dn_q_ref).reshape(dn_b * 16, 4, -1)
                mem_norm = self.memory_norms[i](mem_bt_dn)
                dn_cross_out, _ = self.cross_attn[i](dn_cross_q, mem_norm, mem_norm, need_weights=False)
                dn_q = dn_q + dn_cross_out.reshape(dn_b, 16, 4, -1)
                dn_q = dn_q + self.ffns[i](self.ffn_norms[i](dn_q))

                dn_delta = self.point_delta_heads[i](dn_q)
                dn_ref_new = (dn_ref + torch.tanh(dn_delta) * dn_step_scale).clamp(0.0, 1.0)
                dn_layer_quads.append(dn_ref_new)
                dn_xyxy = quad_to_xyxy(dn_ref_new, min_size=0.02)
                dn_layer_xyxy.append(dn_xyxy)
                dn_layer_boxes.append(xyxy_to_cxcywh_torch(dn_xyxy))
                dn_ref = dn_ref_new.detach() if i + 1 < self.decoder_layers else dn_ref_new
            out["dn_refiner_quad"] = dn_layer_quads[-1]
            out["dn_refiner_xyxy"] = dn_layer_xyxy[-1]
            out["dn_refiner_cxcywh"] = dn_layer_boxes[-1]
            out["dn_refiner_layer_xyxy"] = torch.stack(dn_layer_xyxy, dim=0)
            out["dn_refiner_layer_cxcywh"] = torch.stack(dn_layer_boxes, dim=0)
            out["dn_refiner_group_count"] = torch.tensor(dn_groups, device=device)
        if (
            self.use_proposal_oracle_refiner
            and dn_quad_xy is not None
            and self.training
            and "proposal_quads" in stage1
        ):
            gt_ref = dn_quad_xy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            proposal_quads = stage1["proposal_quads"].to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
            with torch.no_grad():
                gt_prop = gt_ref[:, :, None, :, :].expand_as(proposal_quads)
                prop_quad_iou = quad_iou_approx(
                    proposal_quads.reshape(-1, 4, 2).float(),
                    gt_prop.reshape(-1, 4, 2).float(),
                    grid_size=24,
                ).reshape(proposal_quads.shape[:-2])
                best_idx = prop_quad_iou.argmax(dim=-1)
            gather_idx = best_idx[:, :, None, None, None].expand(-1, -1, 1, 4, 2)
            oracle_ref = proposal_quads.gather(2, gather_idx).squeeze(2).clamp(0.0, 1.0)
            oracle_start = oracle_ref.detach()
            oracle_xyxy = quad_to_xyxy(oracle_ref, min_size=0.02)
            oracle_wh = xyxy_to_cxcywh_torch(oracle_xyxy)[..., 2:].detach()
            oracle_step_scale = (self.residual_scale * oracle_wh[:, :, None, :]).clamp(0.01, 0.50)
            if self.proposal_oracle_refiner_noise_scale > 0.0:
                oracle_noise = (
                    (torch.rand_like(oracle_ref) * 2.0 - 1.0)
                    * float(self.proposal_oracle_refiner_noise_scale)
                )
                oracle_ref = (oracle_ref + oracle_noise * oracle_step_scale.detach()).clamp(0.0, 1.0)

            oracle_q = (
                query_base[:, :, None, :]
                + self.point_index.to(device=device, dtype=dtype)
                + self.time_pos.to(device=device, dtype=dtype)
                + self.point_ref_mlp(oracle_ref)
                + corner_query_bias
            )
            oracle_q = self.query_norm(oracle_q)
            oracle_layer_boxes = []
            oracle_layer_xyxy = []
            oracle_layer_quads = []
            for i in range(self.decoder_layers):
                oracle_q = oracle_q + self.local_point_proj(self._sample_point_features(feat16_map, oracle_ref))

                oracle_q_frame = oracle_q.reshape(b * 16, 4, -1)
                oracle_frame_out, _ = self.frame_attn[i](
                    self.frame_norms[i](oracle_q_frame),
                    self.frame_norms[i](oracle_q_frame),
                    self.frame_norms[i](oracle_q_frame),
                    need_weights=False,
                )
                oracle_q = (oracle_q_frame + oracle_frame_out).reshape(b, 16, 4, -1)

                oracle_q_tem = oracle_q.permute(0, 2, 1, 3).reshape(b * 4, 16, -1)
                oracle_tem_out, _ = self.temporal_attn[i](
                    self.temporal_norms[i](oracle_q_tem),
                    self.temporal_norms[i](oracle_q_tem),
                    self.temporal_norms[i](oracle_q_tem),
                    need_weights=False,
                )
                oracle_q = (oracle_q_tem + oracle_tem_out).reshape(b, 4, 16, -1).permute(0, 2, 1, 3)

                if tool_memory is not None:
                    oracle_q_tool = oracle_q.reshape(b, 16 * 4, -1)
                    oracle_tool_out, _ = self.tool_traj_attn[i](
                        self.tool_query_norms[i](oracle_q_tool),
                        self.tool_memory_norms[i](tool_memory),
                        self.tool_memory_norms[i](tool_memory),
                        need_weights=False,
                    )
                    oracle_q = (oracle_q_tool + oracle_tool_out).reshape(b, 16, 4, -1)

                oracle_q_ref = oracle_q + self.point_ref_mlp(oracle_ref)
                oracle_cross_q = self.cross_norms[i](oracle_q_ref).reshape(b * 16, 4, -1)
                mem_norm = self.memory_norms[i](mem_bt)
                oracle_cross_out, _ = self.cross_attn[i](oracle_cross_q, mem_norm, mem_norm, need_weights=False)
                oracle_q = oracle_q + oracle_cross_out.reshape(b, 16, 4, -1)
                oracle_q = oracle_q + self.ffns[i](self.ffn_norms[i](oracle_q))

                oracle_delta = self.point_delta_heads[i](oracle_q)
                oracle_ref_new = (oracle_ref + torch.tanh(oracle_delta) * oracle_step_scale).clamp(0.0, 1.0)
                oracle_layer_quads.append(oracle_ref_new)
                oracle_ref_xyxy = quad_to_xyxy(oracle_ref_new, min_size=0.02)
                oracle_layer_xyxy.append(oracle_ref_xyxy)
                oracle_layer_boxes.append(xyxy_to_cxcywh_torch(oracle_ref_xyxy))
                oracle_ref = oracle_ref_new.detach() if i + 1 < self.decoder_layers else oracle_ref_new
            out["proposal_oracle_refiner_start_quad"] = oracle_start
            out["proposal_oracle_refiner_quad"] = oracle_layer_quads[-1]
            out["proposal_oracle_refiner_xyxy"] = oracle_layer_xyxy[-1]
            out["proposal_oracle_refiner_cxcywh"] = oracle_layer_boxes[-1]
            out["proposal_oracle_refiner_layer_xyxy"] = torch.stack(oracle_layer_xyxy, dim=0)
            out["proposal_oracle_refiner_layer_cxcywh"] = torch.stack(oracle_layer_boxes, dim=0)
        return out

class H210ProposalOrientedStage1(nn.Module):
    """H151-style proposal/rerank reader for the H161 proxy.

    It generates a small set of spatial proposals from V-JEPA tokens, lets
    candidates interact within each frame and along their own 16-frame temporal
    tracks, then selects a differentiable proposal.  The selected proposal is an
    oriented rectangle so the next point-token stage starts from a structured
    tight-region hypothesis rather than from a single direct quad regression.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        proposal_count: int,
        proposal_selection_mode: str = "train_soft_eval_hard",
        proposal_selection_temperature: float = 1.0,
        proposal_selection_topk: int = 0,
        proposal_selection_straight_through: bool = False,
        proposal_selection_train_logit_noise_std: float = 0.0,
        proposal_selection_train_drop_prob: float = 0.0,
        proposal_geom_temporal_fusion: bool = False,
        proposal_geom_temporal_fusion_scale: float = 0.30,
        proposal_geom_temporal_center_scale: float = 0.18,
        proposal_geom_temporal_source_penalty: float = 4.0,
        proposal_clip_context_fusion: bool = False,
        proposal_clip_context_scale: float = 0.22,
        proposal_clip_context_dropout: float = 0.10,
        proposal_frame_anchor_memory: bool = False,
        proposal_frame_anchor_scale: float = 0.20,
        proposal_frame_anchor_topk: int = 2,
        proposal_frame_anchor_temperature: float = 0.12,
        proposal_frame_anchor_candidate_temperature: float = 0.45,
        proposal_final_fusion: bool = False,
        proposal_final_fusion_scale: float = 0.18,
        proposal_quality_readout: bool = False,
        proposal_quality_readout_detach: bool = False,
        proposal_quality_readout_replace_logits: bool = False,
        proposal_quality_readout_traj_context: bool = False,
        proposal_quality_readout_points: str = "center_corners",
        proposal_quality_readout_pool: str = "mean",
        proposal_support_overlap_score: bool = False,
        proposal_support_overlap_scale: float = 0.0,
        proposal_support_overlap_grid_size: int = 24,
        proposal_support_overlap_detach: bool = False,
        proposal_support_geom_score: bool = False,
        proposal_support_geom_scale: float = 0.0,
        proposal_support_geom_grid_size: int = 24,
        proposal_support_geom_center_tau: float = 0.07,
        proposal_support_geom_area_tau: float = 0.75,
        proposal_support_geom_detach: bool = False,
        proposal_support_moment_proposal: bool = False,
        proposal_support_moment_top_frac: float = 0.035,
        proposal_support_moment_size_scale: float = 1.8,
        proposal_support_moment_temperature: float = 0.16,
        proposal_support_moment_detach: bool = False,
        proposal_decoupled_selector: bool = False,
        proposal_decoupled_selector_detach: bool = False,
        proposal_decoupled_selector_replace_logits: bool = False,
        proposal_decoupled_selector_scale: float = 1.0,
        proposal_decoupled_selector_base_scale: float = 0.0,
        proposal_decoupled_selector_readout: bool = False,
        proposal_oicr_refine_head: bool = False,
        proposal_oicr_refine_use_for_selection: bool = False,
        proposal_pointset_refine: bool = False,
        proposal_pointset_refine_scale: float = 0.18,
        proposal_pointset_refine_grad_mul: float = 0.25,
        proposal_pointset_quality_scale: float = 1.0,
        proposal_pointset_refine_update_token: bool = False,
        proposal_set_refiner: bool = False,
        proposal_set_refiner_topk: int = 0,
        proposal_set_refiner_scale: float = 0.22,
        proposal_set_refiner_temperature: float = 0.90,
        proposal_set_refiner_detach: bool = False,
        proposal_local_corr_score: bool = False,
        proposal_local_corr_scale: float = 0.45,
        proposal_local_corr_grid_size: int = 3,
        proposal_local_corr_temperature: float = 0.08,
        proposal_local_corr_detach: bool = False,
        proposal_active_tip_selector: bool = False,
        proposal_active_tip_selector_scale: float = 1.0,
        proposal_active_tip_selector_aux_only: bool = False,
        proposal_active_tip_selector_eval_fuse_scale: float = 0.0,
        proposal_active_tip_feature_fusion: bool = False,
        proposal_active_tip_feature_scale: float = 0.35,
        proposal_trajectory_rerank: bool = False,
        proposal_trajectory_rerank_scale: float = 0.45,
        proposal_trajectory_refine: bool = False,
        proposal_trajectory_refine_scale: float = 0.30,
        condition_prior_proposal: bool = False,
        h259_prior_proposal: bool = False,
        trajectory_query_proposal_count: int = 0,
        trajectory_query_output_mode: str = "oriented_rect",
        trajectory_readout_bias: bool = False,
        condition_prior_readout_bias: bool = False,
        condition_prior_readout_scale: float = 0.45,
        trajectory_anchor_logit_bias_scale: float = 0.0,
        trajectory_anchor_logit_bias_radius: float = 0.22,
        trajectory_anchor_axis_bias_weight: float = 0.0,
        corner_heatmap_proposal: bool = False,
        corner_heatmap_proposal_temperature: float = 0.35,
        corner_heatmap_proposal_detach: bool = False,
        proposal_source_calibration: bool = False,
        proposal_source_calibration_scale: float = 1.0,
        proposal_handoff_source_calibration: bool = False,
        proposal_handoff_source_calibration_scale: float = 1.0,
        proposal_handoff_source4_init: float = 0.0,
        proposal_handoff_source_router: bool = False,
        proposal_handoff_source_router_scale: float = 1.0,
        proposal_handoff_source_router_candidate_select_scale: float = 0.0,
        proposal_handoff_source5_readout: bool = False,
        proposal_handoff_source5_readout_scale: float = 1.0,
        proposal_handoff_source5_readout_grid_size: int = 3,
        proposal_handoff_source5_readout_sources: str = "5",
        proposal_handoff_source5_readout_detach: bool = False,
        proposal_handoff_mask_quality: bool = False,
        proposal_handoff_mask_quality_scale: float = 0.50,
        proposal_handoff_mask_grid_size: int = 7,
        proposal_handoff_mask_expand_scale: float = 1.25,
        proposal_handoff_mask_refine_scale: float = 0.0,
        proposal_handoff_mask_refine_size_scale: float = 2.0,
        proposal_handoff_mask_refine_min_size_px: float = 2.0,
        proposal_handoff_mask_refine_max_ratio: float = 1.05,
        proposal_handoff_mask_quality_sources: str = "5,7",
        proposal_handoff_mask_detach: bool = False,
        proposal_handoff_mask_score_detach: bool = False,
        proposal_handoff_rank_adaptive_scorer: bool = False,
        proposal_handoff_rank_adaptive_scale: float = 0.45,
        proposal_handoff_rank_adaptive_max_rank: int = 512,
        proposal_handoff_rank_adaptive_detach: bool = False,
        proposal_handoff_sparse_set_scorer: bool = False,
        proposal_handoff_sparse_set_scorer_scale: float = 0.35,
        proposal_handoff_sparse_set_scorer_sources: str = "0,5,7",
        proposal_handoff_sparse_set_scorer_detach: bool = False,
        proposal_handoff_source5_refiner: bool = False,
        proposal_handoff_source5_refiner_topk: int = 12,
        proposal_handoff_source5_refiner_temperature: float = 0.75,
        proposal_handoff_source5_refiner_scale: float = 0.35,
        proposal_handoff_source5_refiner_logit_scale: float = 0.65,
        proposal_handoff_source5_refiner_init_logit_bias: float = 0.0,
        proposal_handoff_source5_refiner_detach: bool = False,
        proposal_handoff_source5_refiner_source_id: int = 5,
        proposal_handoff_source5_refiner_output_source_id: int = 6,
        proposal_handoff_topk_query_refiner: bool = False,
        proposal_handoff_topk_query_sources: str = "0,5",
        proposal_handoff_topk_query_topk_per_source: int = 12,
        proposal_handoff_topk_query_grid_size: int = 3,
        proposal_handoff_topk_query_scale: float = 0.30,
        proposal_handoff_topk_query_logit_scale: float = 0.75,
        proposal_handoff_topk_query_output_source_id: int = 6,
        proposal_handoff_topk_query_replace_logits: bool = False,
        proposal_handoff_topk_query_detach: bool = False,
        proposal_handoff_final_fusion: bool = False,
        proposal_handoff_final_fusion_scale: float = 0.18,
        proposal_handoff_detach_selection_geometry: bool = False,
        proposal_handoff_source_representatives_only: bool = False,
        proposal_handoff_source5_topk: int = 0,
        proposal_handoff_router_gated_source: bool = False,
        proposal_handoff_router_normalized_select: bool = False,
        proposal_handoff_router_normalized_source_scale: float = 1.0,
        proposal_handoff_router_normalized_within_scale: float = 1.0,
        token_affinity_proposal: bool = False,
        token_affinity_proposal_count: int = 1,
        token_affinity_proposal_temperature: float = 0.08,
        token_affinity_proposal_top_frac: float = 0.055,
        token_affinity_proposal_size_scale: float = 3.6,
        token_affinity_proposal_extra_top_fracs: str = "",
        token_affinity_proposal_extra_size_scales: str = "",
        token_affinity_proposal_detach: bool = True,
        topology_contact_template_proposal: bool = False,
        topology_contact_template_sizes_px: str = "4x3,6x4,8x4,10x5",
        topology_contact_template_offsets_px: str = "-2,0,2",
        topology_contact_template_logit_bias: float = 0.0,
        topology_contact_template_index_embedding: bool = False,
        topology_contact_template_combo_bias: bool = False,
        topology_contact_template_combo_bias_scale: float = 1.0,
        topology_contact_template_combo_allowlist: str = "",
        local_grid_refine: bool = False,
        local_grid_refine_grid_size: int = 5,
        local_grid_refine_scale: float = 1.35,
        local_grid_refine_temperature: float = 0.20,
        local_grid_refine_prior_scale: float = 6.0,
        local_grid_refine_prior_sigma: float = 0.28,
        local_grid_refine_detach_base: bool = False,
        local_grid_final_refine: bool = True,
        local_grid_confidence_score: bool = False,
        local_grid_confidence_scale: float = 0.35,
        local_grid_confidence_detach: bool = False,
        candidate_mask_quality: bool = False,
        candidate_mask_grid_size: int = 7,
        candidate_mask_score_scale: float = 0.0,
        candidate_mask_score_detach: bool = False,
        use_rgb_sidecar: bool = False,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.use_rgb_sidecar = bool(use_rgb_sidecar)
        self.condition_prior_proposal = bool(condition_prior_proposal)
        self.h259_prior_proposal = bool(h259_prior_proposal) and self.use_tool_keypoints
        self.residual_scale = float(residual_scale)
        self.proposal_count = max(int(proposal_count), 2)
        self.proposal_selection_mode = str(proposal_selection_mode)
        self.proposal_selection_temperature = max(float(proposal_selection_temperature), 1e-3)
        self.proposal_selection_topk = max(int(proposal_selection_topk), 0)
        self.proposal_selection_straight_through = bool(proposal_selection_straight_through)
        self.proposal_selection_train_logit_noise_std = max(
            float(proposal_selection_train_logit_noise_std),
            0.0,
        )
        self.proposal_selection_train_drop_prob = float(
            np.clip(proposal_selection_train_drop_prob, 0.0, 0.95)
        )
        self.proposal_geom_temporal_fusion = bool(proposal_geom_temporal_fusion)
        self.proposal_geom_temporal_fusion_scale = max(float(proposal_geom_temporal_fusion_scale), 0.0)
        self.proposal_geom_temporal_center_scale = max(float(proposal_geom_temporal_center_scale), 1e-3)
        self.proposal_geom_temporal_source_penalty = max(float(proposal_geom_temporal_source_penalty), 0.0)
        self.proposal_clip_context_fusion = bool(proposal_clip_context_fusion)
        self.proposal_clip_context_scale = max(float(proposal_clip_context_scale), 0.0)
        self.proposal_clip_context_dropout = float(np.clip(proposal_clip_context_dropout, 0.0, 0.5))
        self.proposal_frame_anchor_memory = bool(proposal_frame_anchor_memory)
        self.proposal_frame_anchor_scale = max(float(proposal_frame_anchor_scale), 0.0)
        self.proposal_frame_anchor_topk = max(int(proposal_frame_anchor_topk), 0)
        self.proposal_frame_anchor_temperature = max(float(proposal_frame_anchor_temperature), 1e-3)
        self.proposal_frame_anchor_candidate_temperature = max(
            float(proposal_frame_anchor_candidate_temperature),
            1e-3,
        )
        self.proposal_final_fusion = bool(proposal_final_fusion)
        self.proposal_final_fusion_scale = max(float(proposal_final_fusion_scale), 0.0)
        self.proposal_quality_readout = bool(proposal_quality_readout)
        self.proposal_quality_readout_detach = bool(proposal_quality_readout_detach)
        self.proposal_quality_readout_replace_logits = bool(proposal_quality_readout_replace_logits)
        self.proposal_quality_readout_traj_context = (
            bool(proposal_quality_readout_traj_context) and self.use_tool_keypoints
        )
        if proposal_quality_readout_pool not in {"mean", "attention"}:
            proposal_quality_readout_pool = "mean"
        self.proposal_quality_readout_pool = str(proposal_quality_readout_pool)
        if proposal_quality_readout_points not in {
            "center_corners",
            "center_corners_edges",
            "aligned_grid_3x3",
            "aligned_grid_5x5",
        }:
            proposal_quality_readout_points = "center_corners"
        self.proposal_quality_readout_points = str(proposal_quality_readout_points)
        self.proposal_support_overlap_score = bool(proposal_support_overlap_score)
        self.proposal_support_overlap_scale = float(proposal_support_overlap_scale)
        self.proposal_support_overlap_grid_size = max(int(proposal_support_overlap_grid_size), 8)
        self.proposal_support_overlap_detach = bool(proposal_support_overlap_detach)
        self.proposal_support_geom_score = bool(proposal_support_geom_score)
        self.proposal_support_geom_scale = float(proposal_support_geom_scale)
        self.proposal_support_geom_grid_size = max(int(proposal_support_geom_grid_size), 8)
        self.proposal_support_geom_center_tau = max(float(proposal_support_geom_center_tau), 1e-4)
        self.proposal_support_geom_area_tau = max(float(proposal_support_geom_area_tau), 1e-4)
        self.proposal_support_geom_detach = bool(proposal_support_geom_detach)
        self.proposal_support_moment_proposal = bool(proposal_support_moment_proposal)
        self.proposal_support_moment_top_frac = float(np.clip(proposal_support_moment_top_frac, 0.01, 0.35))
        self.proposal_support_moment_size_scale = max(float(proposal_support_moment_size_scale), 0.25)
        self.proposal_support_moment_temperature = max(float(proposal_support_moment_temperature), 1e-3)
        self.proposal_support_moment_detach = bool(proposal_support_moment_detach)
        self.proposal_decoupled_selector = bool(proposal_decoupled_selector)
        self.proposal_decoupled_selector_detach = bool(proposal_decoupled_selector_detach)
        self.proposal_decoupled_selector_replace_logits = bool(
            proposal_decoupled_selector_replace_logits
        )
        self.proposal_decoupled_selector_scale = float(proposal_decoupled_selector_scale)
        self.proposal_decoupled_selector_base_scale = float(proposal_decoupled_selector_base_scale)
        self.proposal_decoupled_selector_readout = (
            bool(proposal_decoupled_selector_readout) and self.proposal_decoupled_selector
        )
        self.proposal_oicr_refine_head_enabled = bool(proposal_oicr_refine_head)
        self.proposal_oicr_refine_use_for_selection = bool(proposal_oicr_refine_use_for_selection)
        self.proposal_pointset_refine = bool(proposal_pointset_refine)
        self.proposal_pointset_refine_scale = float(proposal_pointset_refine_scale)
        self.proposal_pointset_refine_grad_mul = float(np.clip(proposal_pointset_refine_grad_mul, 0.0, 1.0))
        self.proposal_pointset_quality_scale = float(proposal_pointset_quality_scale)
        self.proposal_pointset_refine_update_token = bool(proposal_pointset_refine_update_token)
        self.proposal_set_refiner = bool(proposal_set_refiner)
        self.proposal_set_refiner_topk = max(int(proposal_set_refiner_topk), 0)
        self.proposal_set_refiner_scale = max(float(proposal_set_refiner_scale), 0.0)
        self.proposal_set_refiner_temperature = max(float(proposal_set_refiner_temperature), 1e-3)
        self.proposal_set_refiner_detach = bool(proposal_set_refiner_detach)
        self.proposal_local_corr_score = bool(proposal_local_corr_score)
        self.proposal_active_tip_selector = bool(proposal_active_tip_selector) and self.use_tool_keypoints
        self.proposal_active_tip_selector_scale = float(proposal_active_tip_selector_scale)
        self.proposal_active_tip_selector_aux_only = bool(proposal_active_tip_selector_aux_only)
        self.proposal_active_tip_selector_eval_fuse_scale = float(proposal_active_tip_selector_eval_fuse_scale)
        self.proposal_active_tip_feature_fusion = (
            bool(proposal_active_tip_feature_fusion) and self.use_tool_keypoints
        )
        self.proposal_active_tip_feature_scale = float(proposal_active_tip_feature_scale)
        self.proposal_trajectory_rerank = bool(proposal_trajectory_rerank) and self.use_tool_keypoints
        self.proposal_trajectory_rerank_scale = float(proposal_trajectory_rerank_scale)
        self.proposal_trajectory_refine = bool(proposal_trajectory_refine) and self.use_tool_keypoints
        self.proposal_trajectory_refine_scale = float(proposal_trajectory_refine_scale)
        self.trajectory_query_proposal_count = (
            max(int(trajectory_query_proposal_count), 0) if self.use_tool_keypoints else 0
        )
        self.trajectory_query_output_mode = str(trajectory_query_output_mode)
        if self.trajectory_query_output_mode not in {"oriented_rect", "direct_quad"}:
            raise ValueError(f"Unsupported trajectory_query_output_mode={self.trajectory_query_output_mode!r}")
        self.trajectory_readout_bias = bool(trajectory_readout_bias) and self.use_tool_keypoints
        self.condition_prior_readout_bias = bool(condition_prior_readout_bias)
        self.condition_prior_readout_scale = float(condition_prior_readout_scale)
        self.condition_prior_enabled = self.condition_prior_proposal or self.condition_prior_readout_bias
        self.trajectory_anchor_logit_bias_scale = (
            float(trajectory_anchor_logit_bias_scale) if self.use_tool_keypoints else 0.0
        )
        self.trajectory_anchor_logit_bias_radius = max(float(trajectory_anchor_logit_bias_radius), 1e-3)
        self.trajectory_anchor_axis_bias_weight = float(trajectory_anchor_axis_bias_weight)
        self.corner_heatmap_proposal = bool(corner_heatmap_proposal)
        self.corner_heatmap_proposal_temperature = max(float(corner_heatmap_proposal_temperature), 1e-3)
        self.corner_heatmap_proposal_detach = bool(corner_heatmap_proposal_detach)
        self.proposal_source_calibration = bool(proposal_source_calibration)
        self.proposal_source_calibration_scale = float(proposal_source_calibration_scale)
        self.proposal_handoff_source_router = bool(proposal_handoff_source_router)
        self.proposal_handoff_source_router_scale = float(proposal_handoff_source_router_scale)
        self.proposal_handoff_source_router_candidate_select_scale = float(
            proposal_handoff_source_router_candidate_select_scale
        )
        self.proposal_handoff_router_normalized_select = bool(
            proposal_handoff_router_normalized_select
        )
        self.proposal_handoff_router_normalized_source_scale = float(
            proposal_handoff_router_normalized_source_scale
        )
        self.proposal_handoff_router_normalized_within_scale = float(
            proposal_handoff_router_normalized_within_scale
        )
        self.token_affinity_proposal = bool(token_affinity_proposal)
        self.token_affinity_proposal_count = max(int(token_affinity_proposal_count), 1)
        self.token_affinity_proposal_temperature = max(float(token_affinity_proposal_temperature), 1e-3)
        self.token_affinity_proposal_top_frac = float(np.clip(token_affinity_proposal_top_frac, 0.01, 0.35))
        self.token_affinity_proposal_size_scale = max(float(token_affinity_proposal_size_scale), 0.25)
        extra_top_fracs = parse_float_csv(token_affinity_proposal_extra_top_fracs)
        extra_size_scales = parse_float_csv(token_affinity_proposal_extra_size_scales)
        variant_n = max(len(extra_top_fracs), len(extra_size_scales))
        self.token_affinity_proposal_variants = [
            (self.token_affinity_proposal_top_frac, self.token_affinity_proposal_size_scale)
        ]
        for i in range(variant_n):
            top_frac = extra_top_fracs[i] if i < len(extra_top_fracs) else self.token_affinity_proposal_top_frac
            size_scale = extra_size_scales[i] if i < len(extra_size_scales) else self.token_affinity_proposal_size_scale
            self.token_affinity_proposal_variants.append(
                (
                    float(np.clip(top_frac, 0.01, 0.35)),
                    max(float(size_scale), 0.25),
                )
            )
        self.token_affinity_proposal_detach = bool(token_affinity_proposal_detach)
        self.topology_contact_template_proposal = bool(topology_contact_template_proposal) and self.use_tool_keypoints
        self.topology_contact_template_sizes = parse_template_sizes_px(
            topology_contact_template_sizes_px,
            mask_res=64,
        )
        self.topology_contact_template_offsets = [
            float(v) / 64.0 for v in parse_float_csv(topology_contact_template_offsets_px)
        ]
        if not self.topology_contact_template_sizes:
            self.topology_contact_template_sizes = parse_template_sizes_px("4x3,6x4,8x4,10x5", mask_res=64)
        if not self.topology_contact_template_offsets:
            self.topology_contact_template_offsets = [-2.0 / 64.0, 0.0, 2.0 / 64.0]
        self.topology_contact_template_count = topology_contact_template_count(
            ",".join(f"{int(round(w * 64))}x{int(round(hh * 64))}" for w, hh in self.topology_contact_template_sizes),
            ",".join(str(v * 64.0) for v in self.topology_contact_template_offsets),
            mask_res=64,
        )
        self.topology_contact_template_logit_bias = float(topology_contact_template_logit_bias)
        self.topology_contact_template_index_embedding = (
            bool(topology_contact_template_index_embedding) and self.topology_contact_template_proposal
        )
        self.topology_contact_template_combo_bias = (
            bool(topology_contact_template_combo_bias) and self.topology_contact_template_proposal
        )
        self.topology_contact_template_combo_bias_scale = float(topology_contact_template_combo_bias_scale)
        allow_mask = parse_combo_index_allowlist(
            topology_contact_template_combo_allowlist,
            int(self.topology_contact_template_count),
            num_combos=17,
        )
        if allow_mask is not None and self.topology_contact_template_proposal:
            self.register_buffer(
                "topology_contact_template_allow_mask",
                allow_mask,
                persistent=False,
            )
        else:
            self.topology_contact_template_allow_mask = None

        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.rgb_encoder = make_rgb_sidecar_encoder(c) if self.use_rgb_sidecar else None
        in_ch = (
            c
            + (c if self.use_rgb_sidecar else 0)
            + 2
            + (1 if self.use_tool_mask else 0)
            + (3 if self.use_tool_keypoints else 0)
        )
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, self.proposal_count, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        self.candidate_embed = nn.Parameter(torch.zeros(1, 1, self.proposal_count, c))
        self.box_ref_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.candidate_in = nn.Sequential(nn.LayerNorm(3 * c + 4), nn.Linear(3 * c + 4, c), nn.SiLU())
        self.frame_norm = nn.LayerNorm(c)
        self.frame_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
        self.temporal_norm = nn.LayerNorm(c)
        self.temporal_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
        self.ffn_norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Dropout(0.10), nn.Linear(4 * c, c))
        self.delta_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 6))
        self.size_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 1))
        if self.proposal_oicr_refine_head_enabled:
            self.proposal_oicr_refine_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 1),
                nn.Linear(c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_oicr_refine_head[-1].weight)
            nn.init.zeros_(self.proposal_oicr_refine_head[-1].bias)
        if self.proposal_final_fusion:
            self.proposal_final_fusion_candidate_mlp = nn.Sequential(
                nn.LayerNorm(c + 8 + 2),
                nn.Linear(c + 8 + 2, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_final_fusion_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 4),
                nn.Linear(c + 8 + 4, c),
                nn.SiLU(),
                nn.Linear(c, 8),
            )
            nn.init.zeros_(self.proposal_final_fusion_head[-1].weight)
            nn.init.zeros_(self.proposal_final_fusion_head[-1].bias)
        if self.proposal_source_calibration:
            self.proposal_source_logit_bias = nn.Embedding(8, 1)
            nn.init.zeros_(self.proposal_source_logit_bias.weight)
        if self.proposal_handoff_source_router:
            self.proposal_handoff_source_router_candidate_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 1),
                nn.Linear(c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            self.proposal_handoff_source_router_context_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 8),
            )
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].bias)
        if self.proposal_geom_temporal_fusion:
            self.proposal_geom_temporal_value = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
            )
            self.proposal_geom_temporal_token_proj = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_geom_temporal_logit_head = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_geom_temporal_token_proj[-1].weight)
            nn.init.zeros_(self.proposal_geom_temporal_token_proj[-1].bias)
            nn.init.zeros_(self.proposal_geom_temporal_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_geom_temporal_logit_head[-1].bias)
        if self.proposal_clip_context_fusion:
            self.proposal_clip_time_embed = nn.Parameter(torch.zeros(1, 16, 1, c))
            self.proposal_clip_source_embed = nn.Embedding(8, c)
            self.proposal_clip_geom_mlp = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_clip_context_norm = nn.LayerNorm(c)
            self.proposal_clip_context_attn = nn.MultiheadAttention(
                c,
                num_heads=4,
                dropout=self.proposal_clip_context_dropout,
                batch_first=True,
            )
            self.proposal_clip_context_ffn_norm = nn.LayerNorm(c)
            self.proposal_clip_context_ffn = nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(self.proposal_clip_context_dropout),
                nn.Linear(4 * c, c),
            )
            self.proposal_clip_context_token_proj = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_clip_context_logit_head = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.trunc_normal_(self.proposal_clip_time_embed, std=0.02)
            nn.init.normal_(self.proposal_clip_source_embed.weight, std=0.02)
            nn.init.zeros_(self.proposal_clip_context_token_proj[-1].weight)
            nn.init.zeros_(self.proposal_clip_context_token_proj[-1].bias)
            nn.init.zeros_(self.proposal_clip_context_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_clip_context_logit_head[-1].bias)
        if self.proposal_frame_anchor_memory:
            self.proposal_frame_anchor_frame_mlp = nn.Sequential(
                nn.LayerNorm(3 * c + 14),
                nn.Linear(3 * c + 14, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_frame_anchor_score_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            self.proposal_frame_anchor_token_proj = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_frame_anchor_logit_head = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_frame_anchor_score_head[-1].weight)
            nn.init.zeros_(self.proposal_frame_anchor_score_head[-1].bias)
            nn.init.zeros_(self.proposal_frame_anchor_token_proj[-1].weight)
            nn.init.zeros_(self.proposal_frame_anchor_token_proj[-1].bias)
            nn.init.zeros_(self.proposal_frame_anchor_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_frame_anchor_logit_head[-1].bias)
        if self.token_affinity_proposal:
            self.token_affinity_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.token_affinity_proposal_quality_head = nn.Sequential(
                nn.LayerNorm(c + 8),
                nn.Linear(c + 8, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.token_affinity_proposal_quality_head[-1].weight)
            nn.init.zeros_(self.token_affinity_proposal_quality_head[-1].bias)
        if self.topology_contact_template_proposal:
            topo_count = max(int(self.topology_contact_template_count), 1)
            self.topology_contact_template_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.topology_contact_template_quality_head = nn.Sequential(
                nn.LayerNorm(c + 8),
                nn.Linear(c + 8, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.topology_contact_template_quality_head[-1].weight)
            nn.init.zeros_(self.topology_contact_template_quality_head[-1].bias)
            if self.topology_contact_template_index_embedding:
                self.topology_contact_template_index_embed = nn.Embedding(topo_count, c)
            if self.topology_contact_template_combo_bias:
                self.topology_contact_template_combo_bias_table = nn.Embedding(17, topo_count)
                nn.init.zeros_(self.topology_contact_template_combo_bias_table.weight)
        if self.proposal_support_moment_proposal:
            self.support_moment_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.support_moment_proposal_quality_head = nn.Sequential(
                nn.LayerNorm(c + 8),
                nn.Linear(c + 8, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.support_moment_proposal_quality_head[-1].weight)
            nn.init.zeros_(self.support_moment_proposal_quality_head[-1].bias)
        if self.corner_heatmap_proposal:
            self.corner_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.corner_proposal_quality_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.corner_proposal_quality_head[-1].weight)
            nn.init.zeros_(self.corner_proposal_quality_head[-1].bias)
        if self.proposal_quality_readout:
            self.proposal_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            if self.proposal_quality_readout_pool == "attention":
                self.proposal_quality_readout_attn = nn.Sequential(
                    nn.LayerNorm(2 * c + 8),
                    nn.Linear(2 * c + 8, c),
                    nn.SiLU(),
                    nn.Linear(c, 1),
                )
            if self.proposal_quality_readout_traj_context:
                self.proposal_quality_traj_context_mlp = nn.Sequential(
                    nn.LayerNorm(13),
                    nn.Linear(13, c),
                    nn.SiLU(),
                    nn.Linear(c, c),
                )
            quality_in_dim = (3 * c + 8) if self.proposal_quality_readout_traj_context else (2 * c + 8)
            self.proposal_quality_readout_head = nn.Sequential(
                nn.LayerNorm(quality_in_dim),
                nn.Linear(quality_in_dim, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_quality_readout_head[-1].weight)
            nn.init.zeros_(self.proposal_quality_readout_head[-1].bias)
        if self.proposal_decoupled_selector_readout:
            self.proposal_decoupled_selector_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
        if self.proposal_decoupled_selector:
            self.proposal_decoupled_selector_source_emb = nn.Embedding(8, c)
            nn.init.normal_(self.proposal_decoupled_selector_source_emb.weight, std=0.02)
            selector_in_dim = 2 * c + 8 + 1
            if self.proposal_decoupled_selector_readout:
                selector_in_dim += c
            self.proposal_decoupled_selector_head = nn.Sequential(
                nn.LayerNorm(selector_in_dim),
                nn.Linear(selector_in_dim, c),
                nn.SiLU(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_decoupled_selector_head[-1].weight)
            nn.init.zeros_(self.proposal_decoupled_selector_head[-1].bias)
        if self.proposal_set_refiner:
            self.proposal_set_refiner_source_emb = nn.Embedding(8, c)
            nn.init.normal_(self.proposal_set_refiner_source_emb.weight, std=0.02)
            self.proposal_set_refiner_item_mlp = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_set_refiner_score_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            self.proposal_set_refiner_delta_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 4),
                nn.Linear(c + 8 + 4, c),
                nn.SiLU(),
                nn.Linear(c, 8),
            )
            nn.init.zeros_(self.proposal_set_refiner_score_head[-1].weight)
            nn.init.zeros_(self.proposal_set_refiner_score_head[-1].bias)
            nn.init.zeros_(self.proposal_set_refiner_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_set_refiner_delta_head[-1].bias)
        if self.proposal_pointset_refine:
            self.proposal_pointset_corner_embed = nn.Parameter(torch.zeros(1, 1, 1, 4, c))
            self.proposal_pointset_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_pointset_delta_head = nn.Sequential(
                nn.LayerNorm(2 * c + 4),
                nn.Linear(2 * c + 4, c),
                nn.SiLU(),
                nn.Linear(c, 2),
            )
            self.proposal_pointset_quality_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            if self.proposal_pointset_refine_update_token:
                self.proposal_pointset_token_proj = nn.Sequential(
                    nn.LayerNorm(c),
                    nn.Linear(c, c),
                    nn.SiLU(),
                    nn.Linear(c, c),
                )
                nn.init.zeros_(self.proposal_pointset_token_proj[-1].weight)
                nn.init.zeros_(self.proposal_pointset_token_proj[-1].bias)
            nn.init.trunc_normal_(self.proposal_pointset_corner_embed, std=0.02)
            nn.init.zeros_(self.proposal_pointset_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_pointset_delta_head[-1].bias)
            nn.init.zeros_(self.proposal_pointset_quality_head[-1].weight)
            nn.init.zeros_(self.proposal_pointset_quality_head[-1].bias)
        active_tip_logits = None
        if self.proposal_active_tip_selector:
            # Candidate-level geometry features: current tip/path distances,
            # tool/motion alignment, frame role, proposal shape, and visibility.
            self.proposal_active_tip_selector_head = nn.Sequential(
                nn.LayerNorm(12),
                nn.Linear(12, c),
                nn.SiLU(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_active_tip_selector_head[-1].weight)
            nn.init.zeros_(self.proposal_active_tip_selector_head[-1].bias)
        if self.proposal_active_tip_feature_fusion:
            self.proposal_active_tip_feature_mlp = nn.Sequential(
                nn.LayerNorm(12),
                nn.Linear(12, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_active_tip_feature_gate = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.proposal_active_tip_feature_mlp[-1].weight)
            nn.init.zeros_(self.proposal_active_tip_feature_mlp[-1].bias)
        if self.proposal_trajectory_rerank or self.proposal_trajectory_refine:
            self.proposal_traj_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
            self.proposal_traj_time_pos = nn.Parameter(torch.zeros(1, 16, c))
            self.proposal_traj_query_norm = nn.LayerNorm(c)
            self.proposal_traj_memory_norm = nn.LayerNorm(c)
            self.proposal_traj_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            if self.proposal_trajectory_refine:
                self.proposal_traj_refine_head = nn.Sequential(
                    nn.LayerNorm(2 * c + 8),
                    nn.Linear(2 * c + 8, c),
                    nn.SiLU(),
                    nn.Linear(c, 6),
                )
                nn.init.zeros_(self.proposal_traj_refine_head[-1].weight)
                nn.init.zeros_(self.proposal_traj_refine_head[-1].bias)
            if self.proposal_trajectory_rerank:
                self.proposal_traj_head = nn.Sequential(
                    nn.LayerNorm(2 * c + 8),
                    nn.Linear(2 * c + 8, c),
                    nn.SiLU(),
                    nn.Linear(c, 1),
                )
                nn.init.zeros_(self.proposal_traj_head[-1].weight)
                nn.init.zeros_(self.proposal_traj_head[-1].bias)
            nn.init.trunc_normal_(self.proposal_traj_time_pos, std=0.02)
        if self.condition_prior_enabled:
            self.condition_prior_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
            self.condition_prior_time_pos = nn.Parameter(torch.zeros(1, 16, c))
            self.condition_prior_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.condition_prior_norm = nn.LayerNorm(c)
            self.condition_prior_ffn_norm = nn.LayerNorm(c)
            self.condition_prior_ffn = nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * c, c),
            )
            self.condition_prior_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 6))
            self.condition_prior_quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 1))
            nn.init.trunc_normal_(self.condition_prior_time_pos, std=0.02)
            nn.init.zeros_(self.condition_prior_head[-1].weight)
            nn.init.zeros_(self.condition_prior_head[-1].bias)
            with torch.no_grad():
                self.condition_prior_head[-1].bias[2:4].fill_(-2.1)
        if self.h259_prior_proposal:
            self.h259_prior = H259TrajectoryPriorTemporalRegion(
                hidden_dim=c,
                residual_scale=min(max(float(residual_scale), 0.10), 0.50),
            )
            self.h259_prior_token_mlp = nn.Sequential(
                nn.LayerNorm(21),
                nn.Linear(21, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.h259_prior_quality_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.h259_prior_quality_head[-1].weight)
            nn.init.zeros_(self.h259_prior_quality_head[-1].bias)
        if self.trajectory_query_proposal_count > 0:
            tq = self.trajectory_query_proposal_count
            traj_out_dim = 8 if self.trajectory_query_output_mode == "direct_quad" else 6
            self.traj_query_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
            self.traj_query_time_pos = nn.Parameter(torch.zeros(1, 16, c))
            self.traj_query_embed = nn.Parameter(torch.zeros(1, 1, tq, c))
            self.traj_query_norm = nn.LayerNorm(c)
            self.traj_memory_norm = nn.LayerNorm(c)
            self.traj_cross_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.traj_frame_norm = nn.LayerNorm(c)
            self.traj_frame_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.traj_temporal_norm = nn.LayerNorm(c)
            self.traj_temporal_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.traj_query_ffn_norm = nn.LayerNorm(c)
            self.traj_query_ffn = nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * c, c),
            )
            self.traj_query_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, traj_out_dim))
            self.traj_query_quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 1))
            nn.init.trunc_normal_(self.traj_query_time_pos, std=0.02)
            nn.init.trunc_normal_(self.traj_query_embed, std=0.02)
            nn.init.zeros_(self.traj_query_head[-1].weight)
            nn.init.zeros_(self.traj_query_head[-1].bias)
            nn.init.zeros_(self.traj_query_quality_head[-1].weight)
            nn.init.zeros_(self.traj_query_quality_head[-1].bias)
            if self.trajectory_query_output_mode == "oriented_rect":
                with torch.no_grad():
                    self.traj_query_head[-1].bias[2:4].fill_(-2.1)
            else:
                with torch.no_grad():
                    self.traj_query_quality_head[-1].bias.fill_(-1.25)
        if (
            self.trajectory_readout_bias
            or self.condition_prior_readout_bias
            or self.proposal_quality_readout
            or self.proposal_pointset_refine
            or self.proposal_local_corr_score
            or self.corner_heatmap_proposal
            or self.token_affinity_proposal
            or self.topology_contact_template_proposal
        ):
            self.traj_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.traj_readout_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.Sigmoid())
            sample_offsets = torch.tensor(
                [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
                dtype=torch.float32,
            )
            self.register_buffer("traj_readout_sample_offsets", sample_offsets, persistent=False)
            self.traj_readout_sample_radius = 1.25 / 24.0
        nn.init.trunc_normal_(self.candidate_embed, std=0.02)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.size_head[-1].weight)
        nn.init.constant_(self.size_head[-1].bias, -2.1)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _proposal_geom_temporal_context(
        self,
        candidate_tokens: torch.Tensor,
        candidate_quads: torch.Tensor,
        proposal_source_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Adjacent-frame proposal memory matched by geometry, not candidate index."""
        b, t, _k, _c = candidate_tokens.shape
        if t < 2:
            return torch.zeros_like(candidate_tokens)
        value = self.proposal_geom_temporal_value(candidate_tokens)
        centers = candidate_quads.detach().mean(dim=-2)
        sources = proposal_source_ids.detach()
        ctx = torch.zeros_like(candidate_tokens)
        count = candidate_tokens.new_zeros((b, t, _k, 1))
        center_scale_sq = self.proposal_geom_temporal_center_scale ** 2
        source_penalty = self.proposal_geom_temporal_source_penalty

        cur_centers = centers[:, 1:]
        prev_centers = centers[:, :-1]
        dist2 = ((cur_centers[:, :, :, None, :] - prev_centers[:, :, None, :, :]) ** 2).sum(dim=-1)
        logits = -dist2 / center_scale_sq
        if source_penalty != 0.0:
            source_mismatch = sources[:, 1:, :, None] != sources[:, :-1, None, :]
            logits = logits - source_penalty * source_mismatch.to(dtype=logits.dtype)
        attn = F.softmax(logits, dim=-1)
        ctx[:, 1:] = ctx[:, 1:] + torch.einsum("btkj,btjc->btkc", attn, value[:, :-1])
        count[:, 1:] = count[:, 1:] + 1.0

        cur_centers = centers[:, :-1]
        next_centers = centers[:, 1:]
        dist2 = ((cur_centers[:, :, :, None, :] - next_centers[:, :, None, :, :]) ** 2).sum(dim=-1)
        logits = -dist2 / center_scale_sq
        if source_penalty != 0.0:
            source_mismatch = sources[:, :-1, :, None] != sources[:, 1:, None, :]
            logits = logits - source_penalty * source_mismatch.to(dtype=logits.dtype)
        attn = F.softmax(logits, dim=-1)
        ctx[:, :-1] = ctx[:, :-1] + torch.einsum("btkj,btjc->btkc", attn, value[:, 1:])
        count[:, :-1] = count[:, :-1] + 1.0
        return ctx / count.clamp_min(1.0)

    def _proposal_clip_context(
        self,
        candidate_tokens: torch.Tensor,
        candidate_quads: torch.Tensor,
        proposal_source_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full-clip proposal context over all frame/candidate selectors."""
        b, t, k, c = candidate_tokens.shape
        device = candidate_tokens.device
        dtype = candidate_tokens.dtype
        geom = candidate_quads.detach().reshape(b, t, k, 8).to(device=device, dtype=dtype)
        source_ids = proposal_source_ids.detach().clamp(0, self.proposal_clip_source_embed.num_embeddings - 1)
        source_embed = self.proposal_clip_source_embed(source_ids).to(device=device, dtype=dtype)
        time_idx = torch.arange(t, device=device).clamp(max=self.proposal_clip_time_embed.shape[1] - 1)
        time_embed = self.proposal_clip_time_embed[:, time_idx].to(device=device, dtype=dtype)
        clip_tokens = (
            candidate_tokens
            + time_embed
            + source_embed
            + self.proposal_clip_geom_mlp(geom)
        )
        flat = clip_tokens.reshape(b, t * k, c)
        norm_flat = self.proposal_clip_context_norm(flat)
        attn_out, _ = self.proposal_clip_context_attn(norm_flat, norm_flat, norm_flat, need_weights=False)
        flat = flat + attn_out
        flat = flat + self.proposal_clip_context_ffn(self.proposal_clip_context_ffn_norm(flat))
        return flat.reshape(b, t, k, c)

    def _proposal_frame_anchor_memory(
        self,
        candidate_tokens: torch.Tensor,
        candidate_quads: torch.Tensor,
        proposal_source_ids: torch.Tensor,
        quality_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restricted essential-frame memory over proposal evidence.

        This follows the RMem lesson at the proposal level: select a compact
        frame memory first, then broadcast it, instead of letting all candidates
        attend to all other candidates.
        """
        b, t, k, c = candidate_tokens.shape
        device = candidate_tokens.device
        dtype = candidate_tokens.dtype
        detached_logits = quality_logits.detach().float()
        candidate_weights = F.softmax(
            detached_logits / self.proposal_frame_anchor_candidate_temperature,
            dim=-1,
        ).to(device=device, dtype=dtype)
        weighted_token = (candidate_tokens * candidate_weights[..., None]).sum(dim=2)
        mean_token = candidate_tokens.mean(dim=2)
        max_token = candidate_tokens.max(dim=2).values

        weighted_quad = (
            candidate_quads.detach().reshape(b, t, k, 8).to(device=device, dtype=dtype)
            * candidate_weights[..., None]
        ).sum(dim=2)
        logit_mean = detached_logits.mean(dim=-1)
        logit_max = detached_logits.max(dim=-1).values
        logit_std = detached_logits.std(dim=-1, unbiased=False)
        entropy = -(
            candidate_weights.float().clamp_min(1e-6)
            * candidate_weights.float().clamp_min(1e-6).log()
        ).sum(dim=-1) / math.log(max(k, 2))
        source_ids = proposal_source_ids.detach().to(device=device)
        source0_mass = (candidate_weights * (source_ids == 0).to(dtype=dtype)).sum(dim=-1)
        source4_mass = (candidate_weights * (source_ids == 4).to(dtype=dtype)).sum(dim=-1)
        stats = torch.stack(
            [
                logit_mean,
                logit_max,
                logit_std,
                entropy.to(device=device, dtype=logit_mean.dtype),
                source0_mass.float(),
                source4_mass.float(),
            ],
            dim=-1,
        ).to(device=device, dtype=dtype)
        frame_token = self.proposal_frame_anchor_frame_mlp(
            torch.cat([weighted_token, mean_token, max_token, weighted_quad, stats], dim=-1)
        )
        frame_logits = self.proposal_frame_anchor_score_head(frame_token).squeeze(-1)
        frame_select_logits = frame_logits / self.proposal_frame_anchor_temperature
        topk = min(int(self.proposal_frame_anchor_topk), t)
        if topk > 0 and topk < t:
            topk_idx = frame_select_logits.topk(topk, dim=-1).indices
            masked_logits = torch.full_like(frame_select_logits, torch.finfo(frame_select_logits.dtype).min)
            frame_select_logits = masked_logits.scatter(
                -1,
                topk_idx,
                frame_select_logits.gather(-1, topk_idx),
            )
        frame_weights = F.softmax(frame_select_logits, dim=-1).to(dtype=dtype)
        memory = (frame_token * frame_weights[..., None]).sum(dim=1)
        context = memory[:, None, None, :].expand(-1, t, k, -1)
        return context, frame_logits, frame_weights

    def _active_tip_candidate_features(
        self,
        candidate_quads: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_id: torch.Tensor,
    ) -> torch.Tensor:
        b, _, k_select = candidate_quads.shape[:3]
        device = candidate_quads.device
        dtype = candidate_quads.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        is_hook = (
            tool_id.to(device=device).reshape(b, 1, 1, 1).to(dtype=torch.long) == 0
        )
        hook_tip_xy = kpt_xy[:, :, 2, :]
        hook_tip_vis = kpt_vis[:, :, 2, 0]
        tip_pair_vis = kpt_vis[:, :, 1:3, :]
        tip_pair_sum = tip_pair_vis.sum(dim=2).clamp_min(1e-4)
        tip_pair_xy = (kpt_xy[:, :, 1:3, :] * tip_pair_vis).sum(dim=2) / tip_pair_sum
        tip_pair_visible = tip_pair_vis.squeeze(-1).amax(dim=2)
        tip_xy = torch.where(is_hook.squeeze(-1), hook_tip_xy, tip_pair_xy)
        tip_vis = torch.where(is_hook.reshape(b, 1, 1).squeeze(-1), hook_tip_vis, tip_pair_visible)
        role = role_features_16.to(device=device, dtype=dtype)
        active = role[..., 0].clamp(0.0, 1.0)
        phase = role[..., 1].clamp(0.0, 1.0)
        dist_to_range = role[..., 3].abs().clamp(0.0, 2.0)

        cand_center = candidate_quads.mean(dim=-2)
        curr_tip_dist = torch.linalg.norm(cand_center - tip_xy[:, :, None, :], dim=-1)

        path_w = (active * tip_vis).clamp_min(0.0)
        fallback_w = tip_vis.clamp_min(0.0)
        has_active = path_w.sum(dim=1, keepdim=True) > 1e-4
        path_w = torch.where(has_active, path_w, fallback_w)
        path_w = path_w / path_w.sum(dim=1, keepdim=True).clamp_min(1e-4)
        active_tip_mean = (tip_xy * path_w[:, :, None]).sum(dim=1)
        mean_tip_dist = torch.linalg.norm(cand_center - active_tip_mean[:, None, None, :], dim=-1)

        dist_to_path = torch.linalg.norm(cand_center[:, :, :, None, :] - tip_xy[:, None, None, :, :], dim=-1)
        tau = 0.050
        path_logits = -dist_to_path / tau + (path_w[:, None, None, :] + 1e-6).log()
        path_alpha = F.softmax(path_logits, dim=-1)
        active_path_dist = (path_alpha * dist_to_path).sum(dim=-1)

        hook_anchor_vis = kpt_vis[:, :, 0:2, :]
        hook_anchor_sum = hook_anchor_vis.sum(dim=2).clamp_min(1e-4)
        hook_anchor_xy = (kpt_xy[:, :, 0:2, :] * hook_anchor_vis).sum(dim=2) / hook_anchor_sum
        nonhook_anchor_xy = kpt_xy[:, :, 0, :]
        anchor_xy = torch.where(is_hook.squeeze(-1), hook_anchor_xy, nonhook_anchor_xy)
        tool_axis = tip_xy - anchor_xy
        fallback_axis = torch.zeros_like(tool_axis)
        fallback_axis[..., 0] = 1.0
        tool_axis_norm = tool_axis.norm(dim=-1, keepdim=True)
        tool_axis = torch.where(
            tool_axis_norm > 1e-4,
            tool_axis / tool_axis_norm.clamp_min(1e-4),
            fallback_axis,
        )
        prev_tip = torch.cat([tip_xy[:, :1], tip_xy[:, :-1]], dim=1)
        next_tip = torch.cat([tip_xy[:, 1:], tip_xy[:, -1:]], dim=1)
        motion = next_tip - prev_tip
        motion_norm = motion.norm(dim=-1, keepdim=True)
        motion_axis = torch.where(motion_norm > 1e-4, motion / motion_norm.clamp_min(1e-4), tool_axis)

        cand_dirs, cand_lens = quad_axes(candidate_quads.reshape(-1, 4, 2))
        cand_axis = cand_dirs[:, 0, :].reshape(b, 16, k_select, 2)
        cand_lens = cand_lens.reshape(b, 16, k_select, 2)
        tool_align = (cand_axis * tool_axis[:, :, None, :]).sum(dim=-1).abs()
        motion_align = (cand_axis * motion_axis[:, :, None, :]).sum(dim=-1).abs()
        cand_area = (cand_lens[..., 0] * cand_lens[..., 1]).clamp(0.0, 1.0)
        cand_aspect = (cand_lens[..., 0] / cand_lens[..., 1].clamp_min(1e-4)).log().clamp(-3.0, 3.0) / 3.0
        active_count = path_w.gt(1e-4).float().sum(dim=1).clamp(0.0, 16.0) / 16.0

        return torch.stack(
            [
                (-curr_tip_dist / 0.25).clamp(-8.0, 0.0),
                (-active_path_dist / 0.25).clamp(-8.0, 0.0),
                (-mean_tip_dist / 0.25).clamp(-8.0, 0.0),
                tool_align,
                motion_align,
                active[:, :, None].expand(-1, -1, k_select),
                phase[:, :, None].expand(-1, -1, k_select),
                (-dist_to_range[:, :, None]).expand(-1, -1, k_select),
                cand_area,
                cand_aspect,
                tip_vis[:, :, None].expand(-1, -1, k_select),
                active_count[:, None, None].expand(-1, 16, k_select),
            ],
            dim=-1,
        )

    def _tool_anchor_quad(self, tool_kpt_vec_16: torch.Tensor) -> torch.Tensor:
        b = tool_kpt_vec_16.shape[0]
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
        base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
        base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
        fallback = torch.zeros_like(base_axis)
        fallback[..., 0] = 1.0
        axis_norm = base_axis.norm(dim=-1, keepdim=True)
        base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
        kpt_dist = (kpt_xy - base_center[:, :, None, :]).norm(dim=-1)
        tool_extent = (kpt_dist * kpt_vis.squeeze(-1)).sum(dim=2) / vis_sum.squeeze(-1)
        base_size = torch.stack(
            [
                (0.75 * tool_extent + 0.060).clamp(0.045, 0.24),
                (0.45 * tool_extent + 0.045).clamp(0.035, 0.18),
            ],
            dim=-1,
        )
        return oriented_rect_to_quad(base_center, base_size, base_axis).clamp(0.0, 1.0)

    def _unit_vec(self, vec: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
        if fallback is None:
            fallback = torch.zeros_like(vec)
            fallback[..., 0] = 1.0
        fb_norm = fallback.norm(dim=-1, keepdim=True)
        fallback = torch.where(fb_norm > 1e-4, fallback / fb_norm.clamp_min(1e-4), torch.zeros_like(fallback))
        fallback[..., 0] = torch.where(fb_norm.squeeze(-1) > 1e-4, fallback[..., 0], torch.ones_like(fallback[..., 0]))
        norm = vec.norm(dim=-1, keepdim=True)
        return torch.where(norm > 1e-4, vec / norm.clamp_min(1e-4), fallback)

    def _topology_contact_template_quads(
        self,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic source7 candidates from the surgical-tool topology.

        Dataset keypoint schema is hook=[anchor0, anchor1, tip0] and
        grasper/scissors/clipper=[anchor0, tip0, tip1].  We generate a fixed
        padded bank so mixed-tool batches keep a stable candidate count.
        """
        b = tool_kpt_vec_16.shape[0]
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        sizes = self.topology_contact_template_sizes
        offsets = self.topology_contact_template_offsets
        endpoint_offsets = [0.0, 2.0 / 64.0, -2.0 / 64.0]

        def append_family(
            out: list[torch.Tensor],
            center: torch.Tensor,
            axis: torch.Tensor,
            family_offsets: list[float],
        ) -> None:
            axis = self._unit_vec(axis)
            for off in family_offsets:
                shifted = (center + float(off) * axis).clamp(0.0, 1.0)
                for long_size, short_size in sizes:
                    size = shifted.new_tensor([float(long_size), float(short_size)]).view(1, 1, 2).expand(b, 16, 2)
                    out.append(oriented_rect_to_quad(shifted, size, axis).clamp(0.0, 1.0))
                    swapped = shifted.new_tensor([float(short_size), float(long_size)]).view(1, 1, 2).expand(b, 16, 2)
                    out.append(oriented_rect_to_quad(shifted, swapped, axis).clamp(0.0, 1.0))

        hook_quads: list[torch.Tensor] = []
        anchor_mid = 0.5 * (kpt_xy[:, :, 0] + kpt_xy[:, :, 1])
        hook_tip = kpt_xy[:, :, 2]
        hook_axis = self._unit_vec(hook_tip - anchor_mid)
        hook_perp = torch.stack([-hook_axis[..., 1], hook_axis[..., 0]], dim=-1)
        append_family(hook_quads, hook_tip, hook_axis, offsets)
        append_family(hook_quads, hook_tip, hook_perp, [0.0])

        nonhook_quads: list[torch.Tensor] = []
        tip_w = kpt_vis[:, :, 1:3].clamp(0.05, 1.0)
        tip_mid = (kpt_xy[:, :, 1:3] * tip_w).sum(dim=2) / tip_w.sum(dim=2).clamp_min(1e-4)
        jaw_axis = self._unit_vec(kpt_xy[:, :, 2] - kpt_xy[:, :, 1], fallback=tip_mid - kpt_xy[:, :, 0])
        shaft_axis = self._unit_vec(tip_mid - kpt_xy[:, :, 0], fallback=jaw_axis)
        append_family(nonhook_quads, tip_mid, jaw_axis, offsets)
        append_family(nonhook_quads, tip_mid, shaft_axis, offsets)
        append_family(nonhook_quads, kpt_xy[:, :, 1], shaft_axis, endpoint_offsets)
        append_family(nonhook_quads, kpt_xy[:, :, 2], shaft_axis, endpoint_offsets)

        max_count = max(len(hook_quads), len(nonhook_quads), int(self.topology_contact_template_count), 1)

        def pad_stack(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if not items:
                zero = kpt_xy[:, :, 0]
                fallback = oriented_rect_to_quad(
                    zero,
                    zero.new_tensor([4.0 / 64.0, 3.0 / 64.0]).view(1, 1, 2).expand(b, 16, 2),
                    self._unit_vec(zero.new_zeros(b, 16, 2)),
                )
                items = [fallback]
            valid_count = len(items)
            while len(items) < max_count:
                items.append(items[0])
            quads = torch.stack(items[:max_count], dim=2).clamp(0.0, 1.0)
            valid = torch.zeros((b, 16, max_count), device=device, dtype=torch.bool)
            valid[:, :, :valid_count] = True
            return quads, valid

        hook_bank, hook_valid = pad_stack(hook_quads)
        nonhook_bank, nonhook_valid = pad_stack(nonhook_quads)
        is_hook = (tool_id.to(device=device).reshape(b).to(dtype=torch.long) == 0).view(b, 1, 1, 1, 1)
        quads = torch.where(is_hook, hook_bank, nonhook_bank)
        valid = torch.where(is_hook.squeeze(-1).squeeze(-1), hook_valid, nonhook_valid)
        return quads.clamp(0.0, 1.0), valid

    def _sample_traj_readout_features(self, feat16: torch.Tensor, ref_xy: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feat16.shape
        p = ref_xy.shape[2]
        offsets = (
            self.traj_readout_sample_offsets.to(device=feat16.device, dtype=feat16.dtype)
            * float(self.traj_readout_sample_radius)
        )
        pts = ref_xy[:, :, :, None, :] + offsets.view(1, 1, 1, -1, 2)
        grid = (pts.reshape(b * t, p * offsets.shape[0], 1, 2) * 2.0 - 1.0).clamp(-1.15, 1.15)
        sampled = F.grid_sample(
            feat16.reshape(b * t, c, h, w).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).to(feat16.dtype)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, p, offsets.shape[0], c)
        center = sampled[:, :, :, 0]
        mean = sampled.mean(dim=3)
        maxv = sampled.max(dim=3).values
        return torch.cat([center, mean, maxv], dim=-1)

    def _decode_support_moment_proposal(self, support_logits_8: torch.Tensor) -> torch.Tensor:
        """Decode a tight support-peak moment into one 16-frame quad candidate."""
        if support_logits_8.ndim == 4:
            b, t8, h, w = support_logits_8.shape
            n = h * w
            logits_flat_in = support_logits_8.reshape(b, t8, n)
        elif support_logits_8.ndim == 3:
            b, t8, n = support_logits_8.shape
            h = int(round(math.sqrt(float(n))))
            w = h
            if h * w != n:
                raise ValueError(f"support logits token count is not square: n={n}")
            logits_flat_in = support_logits_8
        else:
            raise ValueError(f"unsupported support logits shape: {tuple(support_logits_8.shape)}")
        logits_flat = (logits_flat_in.detach() if self.proposal_support_moment_detach else logits_flat_in).float()
        top_n = max(4, min(n, int(round(float(self.proposal_support_moment_top_frac) * float(n)))))
        _top_val, top_idx = logits_flat.topk(top_n, dim=-1)
        masked_logits = torch.full_like(logits_flat, -1.0e4)
        masked_logits.scatter_(-1, top_idx, logits_flat.gather(-1, top_idx))
        weights = F.softmax(
            masked_logits / float(self.proposal_support_moment_temperature),
            dim=-1,
        ).to(dtype=support_logits_8.dtype)

        grid_flat = self.grid_xy.to(device=support_logits_8.device, dtype=support_logits_8.dtype)
        grid_flat = grid_flat.view(2, n).transpose(0, 1)
        center = torch.matmul(weights, grid_flat).clamp(0.0, 1.0)
        diff = grid_flat.view(1, 1, n, 2) - center.unsqueeze(-2)
        ww = weights.unsqueeze(-1)
        var_xx = (ww * diff[..., 0:1] * diff[..., 0:1]).sum(dim=-2).squeeze(-1)
        var_yy = (ww * diff[..., 1:2] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)
        cov_xy = (ww * diff[..., 0:1] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)

        angle = 0.5 * torch.atan2(2.0 * cov_xy, var_xx - var_yy + 1e-8)
        axis = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        delta = torch.sqrt((var_xx - var_yy).square() + 4.0 * cov_xy.square() + 1e-8)
        trace = (var_xx + var_yy).clamp_min(1e-8)
        eig1 = (0.5 * (trace + delta)).clamp_min(1e-8)
        eig2 = (0.5 * (trace - delta)).clamp_min(1e-8)
        token_floor = 2.0 / float(max(h, w))
        size = float(self.proposal_support_moment_size_scale) * torch.stack(
            [torch.sqrt(eig1), torch.sqrt(eig2)],
            dim=-1,
        )
        size = (size + token_floor).clamp(0.03, 0.75)
        quad8 = oriented_rect_to_quad(center, size, axis).clamp(0.0, 1.0)
        quad16 = F.interpolate(
            quad8.permute(0, 2, 3, 1).reshape(b * 4, 2, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, 4, 2, 16).permute(0, 3, 1, 2).contiguous()
        if self.proposal_support_moment_detach:
            quad16 = quad16.detach()
        return quad16.clamp(0.0, 1.0)

    def _decode_token_affinity_proposal(
        self,
        token_feat_8: torch.Tensor,
        center_logits_8: torch.Tensor,
    ) -> torch.Tensor:
        """Decode TokenCut/CutLER-style affinity regions into 16-frame quads.

        TokenCut/LOST/CutLER first use frozen transformer token affinity to
        discover a foreground region, then convert that region into a proposal.
        This lightweight version keeps the same contract but avoids a CPU
        eigensolve in the training loop: learned visual center logits choose
        seed tokens, cosine affinity expands each seed to a compact token
        region, and weighted second moments become an oriented rectangle.
        """
        b, t8, c, h, w = token_feat_8.shape
        n = h * w
        k = center_logits_8.shape[2]
        m = min(max(int(self.token_affinity_proposal_count), 1), int(k))
        tau = float(self.token_affinity_proposal_temperature)

        feat = token_feat_8.detach() if self.token_affinity_proposal_detach else token_feat_8
        feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(b, t8, n, c).float()
        feat_flat = F.normalize(feat_flat, dim=-1, eps=1e-6)

        seed_logits_all = center_logits_8.detach() if self.token_affinity_proposal_detach else center_logits_8
        channel_conf = seed_logits_all.float().amax(dim=-1)
        seed_idx = channel_conf.topk(m, dim=-1).indices
        gather_idx = seed_idx[..., None].expand(-1, -1, -1, n)
        seed_logits = seed_logits_all.float().gather(2, gather_idx)
        seed_prob = F.softmax(seed_logits / tau, dim=-1)
        seed_feat = torch.einsum("btmn,btnc->btmc", seed_prob, feat_flat)
        seed_feat = F.normalize(seed_feat, dim=-1, eps=1e-6)

        grid_flat = self.grid_xy.to(device=token_feat_8.device, dtype=token_feat_8.dtype)
        grid_flat = grid_flat.view(2, n).transpose(0, 1)
        sim = torch.einsum("btnc,btmc->btmn", feat_flat, seed_feat)
        quad_variants = []
        for top_frac, size_scale in self.token_affinity_proposal_variants:
            top_n = max(4, min(n, int(round(float(top_frac) * float(n)))))
            _top_val, top_pos = sim.topk(top_n, dim=-1)
            top_mask = torch.zeros_like(sim, dtype=torch.bool).scatter_(-1, top_pos, True)
            masked_sim = sim.masked_fill(~top_mask, -1.0e4)
            region_w = F.softmax(masked_sim / tau, dim=-1).to(dtype=token_feat_8.dtype)

            center = torch.einsum("btmn,nc->btmc", region_w, grid_flat).clamp(0.0, 1.0)
            diff = grid_flat.view(1, 1, 1, n, 2) - center.unsqueeze(-2)
            ww = region_w.unsqueeze(-1)
            var_xx = (ww * diff[..., 0:1] * diff[..., 0:1]).sum(dim=-2).squeeze(-1)
            var_yy = (ww * diff[..., 1:2] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)
            cov_xy = (ww * diff[..., 0:1] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)

            angle = 0.5 * torch.atan2(2.0 * cov_xy, var_xx - var_yy + 1e-8)
            axis = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            delta = torch.sqrt((var_xx - var_yy).square() + 4.0 * cov_xy.square() + 1e-8)
            trace = (var_xx + var_yy).clamp_min(1e-8)
            eig1 = (0.5 * (trace + delta)).clamp_min(1e-8)
            eig2 = (0.5 * (trace - delta)).clamp_min(1e-8)
            token_floor = 2.0 / float(max(h, w))
            size = float(size_scale) * torch.stack(
                [torch.sqrt(eig1), torch.sqrt(eig2)],
                dim=-1,
            )
            size = (size + token_floor).clamp(0.03, 0.75)
            quad_variants.append(oriented_rect_to_quad(center, size, axis).clamp(0.0, 1.0))

        quad8 = torch.cat(quad_variants, dim=2)
        mv = int(quad8.shape[2])
        quad16 = F.interpolate(
            quad8.permute(0, 2, 3, 4, 1).reshape(b * mv * 4, 2, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, mv, 4, 2, 16).permute(0, 4, 1, 2, 3).contiguous()
        if self.token_affinity_proposal_detach:
            quad16 = quad16.detach()
        return quad16.clamp(0.0, 1.0)

    def _decode_corner_heatmap_proposal(self, corner_logits_8: torch.Tensor) -> torch.Tensor:
        """Decode four ordered corner heatmaps into a differentiable 16-frame quad."""
        b, t8, corners, h, w = corner_logits_8.shape
        if corners != 4:
            raise ValueError(f"Expected 4 corner heatmap channels, got {corners}")
        tau = float(self.corner_heatmap_proposal_temperature)
        logits = corner_logits_8.reshape(b, t8, 4, h * w)
        prob = F.softmax((logits / tau).float(), dim=-1).to(dtype=corner_logits_8.dtype)
        grid_flat = self.grid_xy.to(device=corner_logits_8.device, dtype=corner_logits_8.dtype)
        grid_flat = grid_flat.view(2, h * w).transpose(0, 1)
        quad8 = torch.matmul(prob, grid_flat).clamp(0.0, 1.0)
        quad16 = F.interpolate(
            quad8.permute(0, 2, 3, 1).reshape(b * 4, 2, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, 4, 2, 16).permute(0, 3, 1, 2).contiguous()
        if self.corner_heatmap_proposal_detach:
            quad16 = quad16.detach()
        return quad16.clamp(0.0, 1.0)

    @staticmethod
    def _candidate_aligned_grid_points(candidate_quads: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Sample a regular grid in each candidate quad's own coordinate frame.

        S2A-Net's AlignConv aligns convolution sampling locations to rotated
        anchors.  Here the feature map is a 24x24 V-JEPA token grid, so the
        lightweight equivalent is to bilinearly sample token features at a
        candidate-aligned grid rather than only at global center/corner points.
        """
        grid_size = max(int(grid_size), 2)
        q0 = candidate_quads[..., 0, :]
        q1 = candidate_quads[..., 1, :]
        q2 = candidate_quads[..., 2, :]
        q3 = candidate_quads[..., 3, :]
        coords = torch.linspace(
            0.0,
            1.0,
            grid_size,
            device=candidate_quads.device,
            dtype=candidate_quads.dtype,
        )
        vv, uu = torch.meshgrid(coords, coords, indexing="ij")
        u = uu.reshape(1, 1, 1, grid_size * grid_size, 1)
        v = vv.reshape(1, 1, 1, grid_size * grid_size, 1)
        pts = (
            (1.0 - u) * (1.0 - v) * q0.unsqueeze(-2)
            + u * (1.0 - v) * q1.unsqueeze(-2)
            + u * v * q2.unsqueeze(-2)
            + (1.0 - u) * v * q3.unsqueeze(-2)
        )
        return pts.clamp(0.0, 1.0)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        rgb_16: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        k = self.proposal_count

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias8 = role_bias8 + self.tool_kpt_mlp(kpt8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        feats = [x]
        if self.use_rgb_sidecar:
            if self.rgb_encoder is None:
                raise RuntimeError("RGB sidecar is enabled without an encoder")
            feats.append(encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype))
        feats.append(grid)
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t8, -1, h, w))
        corner_logits_8 = self.corner_heatmap_head(y).view(b, t8, 4, h, w)

        center_logits_8 = self.center_head(y).view(b, t8, k, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=device, dtype=dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        y_flat = y.flatten(2).view(b, t8, 1, -1, h * w)
        pooled8 = (y_flat * prob.unsqueeze(3)).sum(dim=-1)
        role16 = self.role_mlp(role_features_16)
        role8_as16 = role16.reshape(b, 8, 2, -1).mean(dim=2)
        cond8 = z[:, None, :].expand(-1, t8, -1)
        role_cond8 = role8_as16 + cond8
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            role_cond8 = role_cond8 + self.tool_mask_geom_mlp(mask_geom8)

        size8 = torch.sigmoid(self.size_head(pooled8 + self.candidate_embed.to(device=device, dtype=dtype))).clamp(0.02, 0.95)
        boxes8 = torch.cat([centers8, size8], dim=-1)
        q8 = self.candidate_in(
            torch.cat(
                [
                    pooled8,
                    role_cond8[:, :, None, :].expand(-1, -1, k, -1),
                    self.box_ref_mlp(boxes8),
                    boxes8,
                ],
                dim=-1,
            )
        )
        q8 = q8 + self.candidate_embed.to(device=device, dtype=dtype)

        q16 = F.interpolate(
            q8.permute(0, 2, 3, 1).reshape(b * k, -1, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, k, -1, 16).permute(0, 3, 1, 2).contiguous()
        boxes16 = F.interpolate(
            boxes8.permute(0, 2, 3, 1).reshape(b * k, 4, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, k, 4, 16).permute(0, 3, 1, 2).contiguous()

        active_tip_logits = None
        frame_anchor_logits = None
        frame_anchor_weights = None
        cond_quad = None
        cond_xyxy = None
        cond_cxcywh = None
        cond_logit = None
        h259_quad = None
        h259_xyxy = None
        h259_cxcywh = None
        h259_token = None
        h259_logit = None
        token_affinity_proposal_quad = None
        if self.condition_prior_enabled:
            cond_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            cond_tok = self.condition_prior_mlp(cond_in)
            cond_tok = cond_tok + z[:, None, :] + self.condition_prior_time_pos.to(device=device, dtype=dtype)
            cond_attn, _ = self.condition_prior_attn(
                self.condition_prior_norm(cond_tok),
                self.condition_prior_norm(cond_tok),
                self.condition_prior_norm(cond_tok),
                need_weights=False,
            )
            cond_tok = cond_tok + cond_attn
            cond_tok = cond_tok + self.condition_prior_ffn(self.condition_prior_ffn_norm(cond_tok))
            cond_raw = self.condition_prior_head(cond_tok)
            kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
            kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
            base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
            base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
            fallback = torch.zeros_like(base_axis)
            fallback[..., 0] = 1.0
            axis_norm = base_axis.norm(dim=-1, keepdim=True)
            base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
            cond_center = (base_center + 0.35 * torch.tanh(cond_raw[..., :2])).clamp(0.0, 1.0)
            cond_size = torch.sigmoid(cond_raw[..., 2:4]).clamp(0.02, 0.95)
            cond_axis = F.normalize(base_axis + 0.75 * torch.tanh(cond_raw[..., 4:6]), dim=-1, eps=1e-6)
            cond_quad = oriented_rect_to_quad(cond_center, cond_size, cond_axis)
            cond_xyxy = quad_to_xyxy(cond_quad, min_size=0.02)
            cond_cxcywh = xyxy_to_cxcywh_torch(cond_xyxy)
            cond_logit = self.condition_prior_quality_head(cond_tok).squeeze(-1)
        if self.h259_prior_proposal:
            h259_out = self.h259_prior(
                tokens,
                role_features_16,
                tool_mask_8,
                tool_kpt_heatmaps_8,
                tool_kpt_vec_16,
                tool_id,
                act_id,
                combo_id,
            )
            h259_quad = h259_out["trajectory_prior_quad"].to(device=device, dtype=dtype)
            h259_xyxy = h259_out["trajectory_prior_xyxy"].to(device=device, dtype=dtype)
            h259_cxcywh = xyxy_to_cxcywh_torch(h259_xyxy)
            h259_geom = torch.cat(
                [
                    h259_quad.reshape(b, 16, 8),
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            h259_token = self.h259_prior_token_mlp(h259_geom)
            if self.proposal_quality_readout:
                mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
                feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
                feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
                h259_readout = self._sample_traj_readout_features(feat16_map, h259_quad)
                h259_readout = self.traj_readout_proj(h259_readout).mean(dim=2)
                h259_token = h259_token + h259_readout
            h259_logit = self.h259_prior_quality_head(h259_token).squeeze(-1)

            # Reuse the existing condition-prior metric/loss keys for this
            # stronger trajectory-prior proposal when it is enabled. This keeps
            # downstream logging and ablation scripts comparable to H260.
            cond_quad = h259_quad
            cond_xyxy = h259_xyxy
            cond_cxcywh = h259_cxcywh
            cond_logit = h259_logit

        if self.trajectory_readout_bias:
            traj_anchor_quad = self._tool_anchor_quad(tool_kpt_vec_16.to(device=device, dtype=dtype))
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            readout = self._sample_traj_readout_features(feat16_map, traj_anchor_quad)
            readout = self.traj_readout_proj(readout).mean(dim=2)
            readout_gate = self.traj_readout_gate(readout)
            q16 = q16 + 0.35 * readout_gate[:, :, None, :] * readout[:, :, None, :]
        if self.condition_prior_readout_bias and cond_quad is not None:
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            prior_readout = self._sample_traj_readout_features(feat16_map, cond_quad)
            prior_readout = self.traj_readout_proj(prior_readout).mean(dim=2)
            prior_gate = self.traj_readout_gate(prior_readout)
            q16 = q16 + self.condition_prior_readout_scale * prior_gate[:, :, None, :] * prior_readout[:, :, None, :]

        q_frame = q16.reshape(b * 16, k, -1)
        frame_out, _ = self.frame_attn(
            self.frame_norm(q_frame),
            self.frame_norm(q_frame),
            self.frame_norm(q_frame),
            need_weights=False,
        )
        q16 = (q_frame + frame_out).reshape(b, 16, k, -1)

        q_tem = q16.permute(0, 2, 1, 3).reshape(b * k, 16, -1)
        tem_out, _ = self.temporal_attn(
            self.temporal_norm(q_tem),
            self.temporal_norm(q_tem),
            self.temporal_norm(q_tem),
            need_weights=False,
        )
        q16 = (q_tem + tem_out).reshape(b, k, 16, -1).permute(0, 2, 1, 3)
        q16 = q16 + self.ffn(self.ffn_norm(q16))

        delta = torch.tanh(self.delta_head(q16))
        center = (boxes16[..., :2] + delta[..., :2] * self.residual_scale * boxes16[..., 2:].detach()).clamp(0.0, 1.0)
        size = (boxes16[..., 2:] * torch.exp(0.45 * delta[..., 2:4])).clamp(0.02, 0.95)
        base_axis = torch.zeros_like(delta[..., 4:6])
        base_axis[..., 0] = 1.0
        axis = F.normalize(base_axis + 0.75 * delta[..., 4:6], dim=-1, eps=1e-6)
        candidate_quads = oriented_rect_to_quad(center, size, axis)
        candidate_xyxy = quad_to_xyxy(candidate_quads.reshape(b, 16 * k, 4, 2), min_size=0.02).reshape(b, 16, k, 4)
        candidate_cxcywh = xyxy_to_cxcywh_torch(candidate_xyxy)
        if self.proposal_active_tip_feature_fusion:
            tip_feats = self._active_tip_candidate_features(
                candidate_quads,
                tool_kpt_vec_16,
                role_features_16,
                tool_id,
            )
            tip_feature = self.proposal_active_tip_feature_mlp(tip_feats)
            cand_geom = candidate_quads.reshape(b, 16, k, 8)
            tip_gate = self.proposal_active_tip_feature_gate(torch.cat([q16, tip_feature, cand_geom], dim=-1))
            q16 = q16 + self.proposal_active_tip_feature_scale * tip_gate * tip_feature
        quality_logits = self.quality_head(q16).squeeze(-1)
        candidate_tokens = q16
        proposal_source_ids = torch.zeros((b, 16, k), device=device, dtype=torch.long)
        support_logits_8 = torch.logsumexp(center_logits_8.float(), dim=2) - math.log(float(max(k, 1)))

        traj_query_quad = None
        traj_query_xyxy = None
        traj_query_cxcywh = None
        if self.trajectory_query_proposal_count > 0:
            tq = self.trajectory_query_proposal_count
            cond_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            traj_tok = self.traj_query_mlp(cond_in)
            traj_tok = traj_tok + z[:, None, :] + self.traj_query_time_pos.to(device=device, dtype=dtype)
            traj_q = traj_tok[:, :, None, :] + self.traj_query_embed.to(device=device, dtype=dtype)

            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            mem16_chw = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b * 16, h * w, -1)

            traj_q_flat = traj_q.reshape(b * 16, tq, -1)
            traj_cross, _ = self.traj_cross_attn(
                self.traj_query_norm(traj_q_flat),
                self.traj_memory_norm(mem16),
                self.traj_memory_norm(mem16),
                need_weights=False,
            )
            traj_q = (traj_q_flat + traj_cross).reshape(b, 16, tq, -1)

            traj_frame = traj_q.reshape(b * 16, tq, -1)
            traj_frame_out, _ = self.traj_frame_attn(
                self.traj_frame_norm(traj_frame),
                self.traj_frame_norm(traj_frame),
                self.traj_frame_norm(traj_frame),
                need_weights=False,
            )
            traj_q = (traj_frame + traj_frame_out).reshape(b, 16, tq, -1)

            traj_tem = traj_q.permute(0, 2, 1, 3).reshape(b * tq, 16, -1)
            traj_tem_out, _ = self.traj_temporal_attn(
                self.traj_temporal_norm(traj_tem),
                self.traj_temporal_norm(traj_tem),
                self.traj_temporal_norm(traj_tem),
                need_weights=False,
            )
            traj_q = (traj_tem + traj_tem_out).reshape(b, tq, 16, -1).permute(0, 2, 1, 3)
            traj_q = traj_q + self.traj_query_ffn(self.traj_query_ffn_norm(traj_q))

            traj_raw = self.traj_query_head(traj_q)
            kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
            kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
            base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
            base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
            fallback = torch.zeros_like(base_axis)
            fallback[..., 0] = 1.0
            axis_norm = base_axis.norm(dim=-1, keepdim=True)
            base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
            if self.trajectory_query_output_mode == "direct_quad":
                kpt_dist = (kpt_xy - base_center[:, :, None, :]).norm(dim=-1)
                tool_extent = (kpt_dist * kpt_vis.squeeze(-1)).sum(dim=2) / vis_sum.squeeze(-1)
                base_size = torch.stack(
                    [
                        (0.75 * tool_extent + 0.060).clamp(0.045, 0.24),
                        (0.45 * tool_extent + 0.045).clamp(0.035, 0.18),
                    ],
                    dim=-1,
                )
                base_quad = oriented_rect_to_quad(
                    base_center[:, :, None, :].expand(-1, -1, tq, -1),
                    base_size[:, :, None, :].expand(-1, -1, tq, -1),
                    base_axis[:, :, None, :].expand(-1, -1, tq, -1),
                )
                corner_scale = (0.55 * base_size[:, :, None, None, :]).clamp(0.018, 0.16)
                traj_delta = torch.tanh(traj_raw.view(b, 16, tq, 4, 2)) * corner_scale
                traj_query_quad = (base_quad + traj_delta).clamp(0.0, 1.0)
            else:
                traj_center = (base_center[:, :, None, :] + 0.35 * torch.tanh(traj_raw[..., :2])).clamp(0.0, 1.0)
                traj_size = torch.sigmoid(traj_raw[..., 2:4]).clamp(0.02, 0.95)
                traj_axis = F.normalize(
                    base_axis[:, :, None, :] + 0.75 * torch.tanh(traj_raw[..., 4:6]),
                    dim=-1,
                    eps=1e-6,
                )
                traj_query_quad = oriented_rect_to_quad(traj_center, traj_size, traj_axis)
            traj_query_xyxy = quad_to_xyxy(traj_query_quad.reshape(b, 16 * tq, 4, 2), min_size=0.02).reshape(b, 16, tq, 4)
            traj_query_cxcywh = xyxy_to_cxcywh_torch(traj_query_xyxy)
            traj_query_logit = self.traj_query_quality_head(traj_q).squeeze(-1)

            candidate_quads = torch.cat([candidate_quads, traj_query_quad], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, traj_query_xyxy], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, traj_query_cxcywh], dim=2)
            quality_logits = torch.cat([quality_logits, traj_query_logit], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, traj_q], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, tq), 1, device=device, dtype=torch.long),
                ],
                dim=2,
            )

        if self.condition_prior_proposal and cond_quad is not None:
            candidate_quads = torch.cat([candidate_quads, cond_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, cond_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, cond_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, cond_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, cond_tok[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 2, device=device, dtype=torch.long),
                ],
                dim=2,
            )
        if self.h259_prior_proposal and h259_quad is not None:
            candidate_quads = torch.cat([candidate_quads, h259_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, h259_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, h259_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, h259_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, h259_token[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 3, device=device, dtype=torch.long),
                ],
                dim=2,
            )
        corner_proposal_quad = None
        if self.corner_heatmap_proposal:
            corner_proposal_quad = self._decode_corner_heatmap_proposal(corner_logits_8)
            corner_proposal_xyxy = quad_to_xyxy(corner_proposal_quad, min_size=0.02)
            corner_proposal_cxcywh = xyxy_to_cxcywh_torch(corner_proposal_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            corner_readout = self._sample_traj_readout_features(feat16_map, corner_proposal_quad)
            corner_token = self.corner_proposal_token_proj(corner_readout.mean(dim=2))
            corner_logit = self.corner_proposal_quality_head(corner_token).squeeze(-1)
            candidate_quads = torch.cat([candidate_quads, corner_proposal_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, corner_proposal_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, corner_proposal_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, corner_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, corner_token[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 4, device=device, dtype=torch.long),
                ],
                dim=2,
            )
        if self.token_affinity_proposal:
            token_affinity_proposal_quad = self._decode_token_affinity_proposal(x, center_logits_8)
            ta_count = int(token_affinity_proposal_quad.shape[2])
            token_affinity_proposal_xyxy = quad_to_xyxy(
                token_affinity_proposal_quad.reshape(b, 16 * ta_count, 4, 2),
                min_size=0.02,
            ).reshape(b, 16, ta_count, 4)
            token_affinity_proposal_cxcywh = xyxy_to_cxcywh_torch(token_affinity_proposal_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            affinity_readout = self._sample_traj_readout_features(
                feat16_map,
                token_affinity_proposal_quad.reshape(b, 16, ta_count * 4, 2),
            )
            affinity_readout = affinity_readout.reshape(b, 16, ta_count, 4, -1).mean(dim=3)
            affinity_token = self.token_affinity_proposal_token_proj(affinity_readout)
            affinity_geom = token_affinity_proposal_quad.reshape(b, 16, ta_count, 8)
            affinity_logit = self.token_affinity_proposal_quality_head(
                torch.cat([affinity_token, affinity_geom], dim=-1)
            ).squeeze(-1)
            candidate_quads = torch.cat([candidate_quads, token_affinity_proposal_quad], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, token_affinity_proposal_xyxy], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, token_affinity_proposal_cxcywh], dim=2)
            quality_logits = torch.cat([quality_logits, affinity_logit], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, affinity_token], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, ta_count), 5, device=device, dtype=torch.long),
                ],
                dim=2,
            )

        topology_contact_template_quad = None
        if self.topology_contact_template_proposal:
            topology_contact_template_quad, topology_template_valid = self._topology_contact_template_quads(
                tool_kpt_vec_16.to(device=device, dtype=dtype),
                tool_id,
            )
            topo_count = int(topology_contact_template_quad.shape[2])
            topology_contact_template_xyxy = quad_to_xyxy(
                topology_contact_template_quad.reshape(b, 16 * topo_count, 4, 2),
                min_size=0.02,
            ).reshape(b, 16, topo_count, 4)
            topology_contact_template_cxcywh = xyxy_to_cxcywh_torch(topology_contact_template_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            topo_readout = self._sample_traj_readout_features(
                feat16_map,
                topology_contact_template_quad.reshape(b, 16, topo_count * 4, 2),
            )
            topo_readout = topo_readout.reshape(b, 16, topo_count, 4, -1).mean(dim=3)
            topology_token = self.topology_contact_template_token_proj(topo_readout)
            if self.topology_contact_template_index_embedding:
                topo_idx = torch.arange(topo_count, device=device, dtype=torch.long).clamp(
                    max=self.topology_contact_template_index_embed.num_embeddings - 1
                )
                topology_token = topology_token + self.topology_contact_template_index_embed(
                    topo_idx
                ).to(dtype=topology_token.dtype).view(1, 1, topo_count, -1)
            topology_geom = topology_contact_template_quad.reshape(b, 16, topo_count, 8)
            topology_logit = self.topology_contact_template_quality_head(
                torch.cat([topology_token, topology_geom], dim=-1)
            ).squeeze(-1)
            topology_logit = topology_logit + float(self.topology_contact_template_logit_bias)
            if self.topology_contact_template_combo_bias:
                combo_idx = combo_id.to(device=device, dtype=torch.long).clamp(
                    0,
                    self.topology_contact_template_combo_bias_table.num_embeddings - 1,
                )
                combo_bias = self.topology_contact_template_combo_bias_table(combo_idx)[
                    :, :topo_count
                ].to(dtype=topology_logit.dtype)
                topology_logit = topology_logit + float(
                    self.topology_contact_template_combo_bias_scale
                ) * combo_bias[:, None, :]
            allow_mask = getattr(self, "topology_contact_template_allow_mask", None)
            if allow_mask is not None:
                combo_idx = combo_id.to(device=device, dtype=torch.long).clamp(
                    0,
                    allow_mask.shape[0] - 1,
                )
                combo_allow = allow_mask.to(device=device)[combo_idx, :topo_count]
                topology_template_valid = topology_template_valid & combo_allow[:, None, :]
                topology_contact_template_quad = torch.where(
                    topology_template_valid[..., None, None],
                    topology_contact_template_quad,
                    torch.zeros_like(topology_contact_template_quad),
                )
                topology_contact_template_xyxy = quad_to_xyxy(
                    topology_contact_template_quad.reshape(b, 16 * topo_count, 4, 2),
                    min_size=0.02,
                ).reshape(b, 16, topo_count, 4)
                topology_contact_template_cxcywh = xyxy_to_cxcywh_torch(
                    topology_contact_template_xyxy
                )
            topology_logit = torch.where(
                topology_template_valid,
                topology_logit,
                torch.full_like(topology_logit, -50.0),
            )
            candidate_quads = torch.cat([candidate_quads, topology_contact_template_quad], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, topology_contact_template_xyxy], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, topology_contact_template_cxcywh], dim=2)
            quality_logits = torch.cat([quality_logits, topology_logit], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, topology_token], dim=2)
            topology_source_ids = torch.where(
                topology_template_valid,
                torch.full((b, 16, topo_count), 7, device=device, dtype=torch.long),
                torch.full((b, 16, topo_count), -1, device=device, dtype=torch.long),
            )
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    topology_source_ids,
                ],
                dim=2,
            )

        support_moment_proposal_quad = None
        if self.proposal_support_moment_proposal:
            support_moment_proposal_quad = self._decode_support_moment_proposal(support_logits_8)
            support_moment_proposal_xyxy = quad_to_xyxy(support_moment_proposal_quad, min_size=0.02)
            support_moment_proposal_cxcywh = xyxy_to_cxcywh_torch(support_moment_proposal_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            support_readout = self._sample_traj_readout_features(feat16_map, support_moment_proposal_quad)
            support_token = self.support_moment_proposal_token_proj(support_readout.mean(dim=2))
            support_geom = support_moment_proposal_quad.reshape(b, 16, 8)
            support_logit = self.support_moment_proposal_quality_head(
                torch.cat([support_token, support_geom], dim=-1)
            ).squeeze(-1)
            candidate_quads = torch.cat([candidate_quads, support_moment_proposal_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, support_moment_proposal_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, support_moment_proposal_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, support_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, support_token[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 6, device=device, dtype=torch.long),
                ],
                dim=2,
            )

        k_select = candidate_quads.shape[2]
        if self.proposal_trajectory_refine:
            traj_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            traj_mem = self.proposal_traj_mlp(traj_in)
            traj_mem = traj_mem + z[:, None, :] + self.proposal_traj_time_pos.to(device=device, dtype=dtype)
            cand_q = candidate_tokens.reshape(b * 16, k_select, -1)
            traj_kv = traj_mem[:, None, :, :].expand(-1, 16, -1, -1).reshape(b * 16, 16, -1)
            traj_ctx, _ = self.proposal_traj_attn(
                self.proposal_traj_query_norm(cand_q),
                self.proposal_traj_memory_norm(traj_kv),
                self.proposal_traj_memory_norm(traj_kv),
                need_weights=False,
            )
            traj_ctx = traj_ctx.reshape(b, 16, k_select, -1)
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            refine_raw = torch.tanh(
                self.proposal_traj_refine_head(torch.cat([candidate_tokens, traj_ctx, cand_geom], dim=-1))
            )
            cand_center = candidate_quads.mean(dim=-2)
            cand_dirs, cand_lens = quad_axes(candidate_quads.reshape(-1, 4, 2))
            cand_dirs = cand_dirs.reshape(b, 16, k_select, 2, 2)
            cand_lens = cand_lens.reshape(b, 16, k_select, 2)
            refine_scale = float(self.proposal_trajectory_refine_scale)
            center_step = cand_lens.mean(dim=-1, keepdim=True).clamp(0.02, 0.35)
            refined_center = (cand_center + refine_scale * refine_raw[..., :2] * center_step).clamp(0.0, 1.0)
            refined_size = (cand_lens * torch.exp(0.35 * refine_scale * refine_raw[..., 2:4])).clamp(0.02, 0.95)
            refined_axis = F.normalize(
                cand_dirs[..., 0, :] + 0.75 * refine_scale * refine_raw[..., 4:6],
                dim=-1,
                eps=1e-6,
            )
            candidate_quads = oriented_rect_to_quad(refined_center, refined_size, refined_axis)
            candidate_xyxy = quad_to_xyxy(candidate_quads.reshape(b, 16 * k_select, 4, 2), min_size=0.02).reshape(
                b, 16, k_select, 4
            )
            candidate_cxcywh = xyxy_to_cxcywh_torch(candidate_xyxy)

        if self.proposal_pointset_refine:
            # OrientedRepPoints-style handoff: use the current 4 candidate
            # points as the representation, sample local token evidence at
            # those points, then predict a small point-wise residual.  The
            # final task remains free-quad regression; this avoids forcing the
            # GT through a rotated-rectangle surrogate.
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            gm = float(self.proposal_pointset_refine_grad_mul)
            sample_quads = candidate_quads.detach() * (1.0 - gm) + candidate_quads * gm
            point_readout = self._sample_traj_readout_features(
                feat16_map,
                sample_quads.reshape(b, 16, k_select * 4, 2),
            )
            point_readout = point_readout.reshape(b, 16, k_select, 4, -1)
            point_feat = self.proposal_pointset_readout_proj(point_readout)
            point_feat = point_feat + self.proposal_pointset_corner_embed.to(device=device, dtype=dtype)
            point_geom_center = sample_quads.mean(dim=-2, keepdim=True)
            point_geom = torch.cat([sample_quads, sample_quads - point_geom_center], dim=-1)
            cand_tok = candidate_tokens[:, :, :, None, :].expand(-1, -1, -1, 4, -1)
            point_delta = torch.tanh(
                self.proposal_pointset_delta_head(torch.cat([cand_tok, point_feat, point_geom], dim=-1))
            )
            _dirs, cand_lens = quad_axes(sample_quads.reshape(-1, 4, 2))
            cand_lens = cand_lens.reshape(b, 16, k_select, 2)
            point_step = (
                float(self.proposal_pointset_refine_scale)
                * cand_lens.mean(dim=-1).clamp(0.015, 0.35)
            )
            candidate_quads = (candidate_quads + point_delta * point_step[..., None, None]).clamp(0.0, 1.0)
            candidate_xyxy = quad_to_xyxy(candidate_quads.reshape(b, 16 * k_select, 4, 2), min_size=0.02).reshape(
                b, 16, k_select, 4
            )
            candidate_cxcywh = xyxy_to_cxcywh_torch(candidate_xyxy)
            point_context = point_feat.mean(dim=3)
            if self.proposal_pointset_refine_update_token:
                candidate_tokens = candidate_tokens + self.proposal_pointset_token_proj(point_context)
            if self.proposal_pointset_quality_scale != 0.0:
                quality_logits = quality_logits + (
                    self.proposal_pointset_quality_scale
                    * self.proposal_pointset_quality_head(point_context).squeeze(-1)
                )

        decoupled_selector_readout = None

        if self.proposal_quality_readout:
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            cand_center = candidate_quads.mean(dim=-2, keepdim=True)
            if self.proposal_quality_readout_points == "aligned_grid_5x5":
                readout_pts = self._candidate_aligned_grid_points(candidate_quads, grid_size=5)
                points_per_candidate = 25
            elif self.proposal_quality_readout_points == "aligned_grid_3x3":
                readout_pts = self._candidate_aligned_grid_points(candidate_quads, grid_size=3)
                points_per_candidate = 9
            elif self.proposal_quality_readout_points == "center_corners_edges":
                edge_midpoints = 0.5 * (candidate_quads + torch.roll(candidate_quads, shifts=-1, dims=-2))
                readout_pts = torch.cat([cand_center, candidate_quads, edge_midpoints], dim=3)
                points_per_candidate = 9
            else:
                readout_pts = torch.cat([cand_center, candidate_quads], dim=3)
                points_per_candidate = 5
            readout_pts = readout_pts.reshape(b, 16, k_select * points_per_candidate, 2)
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            cand_readout = self._sample_traj_readout_features(feat16_map, readout_pts)
            cand_readout = cand_readout.reshape(b, 16, k_select, points_per_candidate, -1)
            point_readout = self.proposal_readout_proj(cand_readout)
            if self.proposal_quality_readout_pool == "attention":
                attn_token = candidate_tokens[:, :, :, None, :].expand(-1, -1, -1, points_per_candidate, -1)
                attn_geom = cand_geom[:, :, :, None, :].expand(-1, -1, -1, points_per_candidate, -1)
                attn_in = torch.cat([point_readout, attn_token, attn_geom], dim=-1)
                if self.proposal_quality_readout_detach:
                    attn_in = attn_in.detach()
                point_weights = F.softmax(self.proposal_quality_readout_attn(attn_in).squeeze(-1), dim=3)
                cand_readout = (point_readout * point_weights[..., None]).sum(dim=3)
            else:
                cand_readout = point_readout.mean(dim=3)
            readout_tokens = candidate_tokens
            if self.proposal_quality_readout_detach:
                readout_tokens = readout_tokens.detach()
                cand_readout = cand_readout.detach()
                cand_geom = cand_geom.detach()
            quality_parts = [readout_tokens, cand_readout, cand_geom]
            if self.proposal_quality_readout_traj_context:
                traj_context = self.proposal_quality_traj_context_mlp(
                    torch.cat(
                        [
                            tool_kpt_vec_16.to(device=device, dtype=dtype),
                            role_features_16.to(device=device, dtype=dtype),
                        ],
                        dim=-1,
                    )
                )
                quality_parts.insert(2, traj_context[:, :, None, :].expand(-1, -1, k_select, -1))
            quality_delta = self.proposal_quality_readout_head(
                torch.cat(quality_parts, dim=-1)
            ).squeeze(-1)
            if self.proposal_quality_readout_replace_logits:
                quality_logits = quality_delta
            else:
                quality_logits = quality_logits + quality_delta

        if self.proposal_decoupled_selector_readout:
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            readout_quads = candidate_quads
            if self.proposal_decoupled_selector_detach:
                feat16_map = feat16_map.detach()
                readout_quads = readout_quads.detach()
            cand_center = readout_quads.mean(dim=-2, keepdim=True)
            if self.proposal_quality_readout_points == "aligned_grid_5x5":
                readout_pts = self._candidate_aligned_grid_points(readout_quads, grid_size=5)
                points_per_candidate = 25
            elif self.proposal_quality_readout_points == "aligned_grid_3x3":
                readout_pts = self._candidate_aligned_grid_points(readout_quads, grid_size=3)
                points_per_candidate = 9
            elif self.proposal_quality_readout_points == "center_corners_edges":
                edge_midpoints = 0.5 * (readout_quads + torch.roll(readout_quads, shifts=-1, dims=-2))
                readout_pts = torch.cat([cand_center, readout_quads, edge_midpoints], dim=3)
                points_per_candidate = 9
            else:
                readout_pts = torch.cat([cand_center, readout_quads], dim=3)
                points_per_candidate = 5
            readout_pts = readout_pts.reshape(b, 16, k_select * points_per_candidate, 2)
            selector_readout = self._sample_traj_readout_features(feat16_map, readout_pts)
            selector_readout = selector_readout.reshape(b, 16, k_select, points_per_candidate, -1)
            decoupled_selector_readout = self.proposal_decoupled_selector_readout_proj(selector_readout).mean(dim=3)

        if self.proposal_active_tip_selector:
            tip_feats = self._active_tip_candidate_features(
                candidate_quads,
                tool_kpt_vec_16,
                role_features_16,
                tool_id,
            )
            active_tip_delta = self.proposal_active_tip_selector_head(tip_feats).squeeze(-1)
            active_tip_logits = active_tip_delta
            if self.proposal_active_tip_selector_aux_only:
                if (not self.training) and self.proposal_active_tip_selector_eval_fuse_scale != 0.0:
                    quality_logits = (
                        quality_logits
                        + self.proposal_active_tip_selector_eval_fuse_scale * active_tip_delta
                    )
            else:
                quality_logits = quality_logits + self.proposal_active_tip_selector_scale * active_tip_delta

        if self.proposal_trajectory_rerank:
            traj_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            traj_mem = self.proposal_traj_mlp(traj_in)
            traj_mem = traj_mem + z[:, None, :] + self.proposal_traj_time_pos.to(device=device, dtype=dtype)
            cand_q = candidate_tokens.reshape(b * 16, k_select, -1)
            traj_kv = traj_mem[:, None, :, :].expand(-1, 16, -1, -1).reshape(b * 16, 16, -1)
            traj_ctx, _ = self.proposal_traj_attn(
                self.proposal_traj_query_norm(cand_q),
                self.proposal_traj_memory_norm(traj_kv),
                self.proposal_traj_memory_norm(traj_kv),
                need_weights=False,
            )
            traj_ctx = traj_ctx.reshape(b, 16, k_select, -1)
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            traj_delta = self.proposal_traj_head(
                torch.cat([candidate_tokens, traj_ctx, cand_geom], dim=-1)
            ).squeeze(-1)
            quality_logits = quality_logits + self.proposal_trajectory_rerank_scale * traj_delta

        if self.trajectory_anchor_logit_bias_scale != 0.0 and self.use_tool_keypoints:
            kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
            kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
            anchor_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
            cand_center = candidate_quads.mean(dim=-2)
            center_dist = torch.linalg.norm(cand_center - anchor_center[:, :, None, :], dim=-1)
            prior = -center_dist / self.trajectory_anchor_logit_bias_radius
            if self.trajectory_anchor_axis_bias_weight != 0.0:
                base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
                fallback = torch.zeros_like(base_axis)
                fallback[..., 0] = 1.0
                axis_norm = base_axis.norm(dim=-1, keepdim=True)
                base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
                cand_dirs, _cand_lens = quad_axes(candidate_quads.reshape(-1, 4, 2))
                cand_axis = cand_dirs[..., 0, :].reshape(b, 16, k_select, 2)
                axis_align = (cand_axis * base_axis[:, :, None, :]).sum(dim=-1).abs().pow(2)
                prior = prior + self.trajectory_anchor_axis_bias_weight * axis_align
            quality_logits = quality_logits + self.trajectory_anchor_logit_bias_scale * prior

        if self.proposal_source_calibration:
            source_bias = self.proposal_source_logit_bias(proposal_source_ids).squeeze(-1)
            quality_logits = quality_logits + self.proposal_source_calibration_scale * source_bias

        support_logits_8 = torch.logsumexp(center_logits_8.float(), dim=2) - math.log(float(max(k, 1)))
        proposal_support_overlap_scores = None
        proposal_support_geom_scores = None

        handoff_source_router_logits = None
        handoff_source_router_candidate_logits = None
        if self.proposal_handoff_source_router and proposal_source_ids.shape == quality_logits.shape:
            router_geom = candidate_quads.reshape(b, 16, k_select, 8)
            router_in = torch.cat(
                [
                    candidate_tokens,
                    router_geom,
                    quality_logits.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            handoff_source_router_candidate_logits = (
                self.proposal_handoff_source_router_candidate_head(router_in).squeeze(-1)
            )
            handoff_source_ids = proposal_source_ids.detach().clamp(0, 7)
            neg = torch.finfo(handoff_source_router_candidate_logits.dtype).min
            group_logits = []
            group_present = []
            for source_id in range(8):
                source_mask = handoff_source_ids == int(source_id)
                group_present.append(source_mask.any(dim=-1))
                source_scores = handoff_source_router_candidate_logits.masked_fill(~source_mask, neg)
                group_logits.append(source_scores.max(dim=-1).values)
            handoff_source_router_logits = torch.stack(group_logits, dim=-1)
            router_context = role16 + z[:, None, :].expand(-1, 16, -1)
            if self.use_tool_keypoints:
                router_context = router_context + self.tool_kpt_mlp(
                    tool_kpt_vec_16.to(device=device, dtype=dtype)
                )
            if self.use_tool_mask:
                mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
                mask_geom16 = F.interpolate(
                    mask_geom8.transpose(1, 2),
                    size=16,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
                router_context = router_context + self.tool_mask_geom_mlp(mask_geom16)
            handoff_source_router_logits = (
                handoff_source_router_logits
                + self.proposal_handoff_source_router_context_head(router_context).to(
                    dtype=handoff_source_router_logits.dtype
                )
            )
            group_present_t = torch.stack(group_present, dim=-1)
            handoff_source_router_logits = handoff_source_router_logits.masked_fill(
                ~group_present_t,
                neg,
            )
            handoff_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
            quality_logits = quality_logits + float(self.proposal_handoff_source_router_scale) * handoff_bonus.to(
                device=quality_logits.device,
                dtype=quality_logits.dtype,
            )
            if abs(float(self.proposal_handoff_source_router_candidate_select_scale)) > 0.0:
                quality_logits = quality_logits + float(
                    self.proposal_handoff_source_router_candidate_select_scale
                ) * handoff_source_router_candidate_logits.to(
                    device=quality_logits.device,
                    dtype=quality_logits.dtype,
                )
            if self.proposal_handoff_router_normalized_select and (not self.training):
                centered_logits = torch.full_like(quality_logits, -50.0)
                for source_id in range(8):
                    source_mask = handoff_source_ids == int(source_id)
                    if not bool(source_mask.any().detach().cpu()):
                        continue
                    source_count = source_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(
                        dtype=quality_logits.dtype
                    )
                    source_vals = quality_logits.masked_fill(~source_mask, 0.0)
                    source_mean = source_vals.sum(dim=-1, keepdim=True) / source_count
                    source_var = (
                        ((quality_logits - source_mean).masked_fill(~source_mask, 0.0) ** 2).sum(
                            dim=-1,
                            keepdim=True,
                        )
                        / source_count
                    )
                    source_std = torch.sqrt(source_var.clamp_min(1e-4))
                    centered_logits = torch.where(
                        source_mask,
                        (quality_logits - source_mean) / source_std,
                        centered_logits,
                    )
                router_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
                quality_logits = (
                    float(self.proposal_handoff_router_normalized_within_scale)
                    * centered_logits
                    + float(self.proposal_handoff_router_normalized_source_scale)
                    * router_bonus.to(device=quality_logits.device, dtype=quality_logits.dtype)
                )

        if self.proposal_geom_temporal_fusion:
            geom_ctx = self._proposal_geom_temporal_context(
                candidate_tokens,
                candidate_quads,
                proposal_source_ids,
            )
            t_select = candidate_tokens.shape[1]
            cand_geom = candidate_quads.reshape(b, t_select, k_select, 8)
            token_delta = self.proposal_geom_temporal_token_proj(
                torch.cat([candidate_tokens, geom_ctx, cand_geom], dim=-1)
            )
            fusion_scale = float(self.proposal_geom_temporal_fusion_scale)
            candidate_tokens = candidate_tokens + fusion_scale * token_delta
            logit_delta = self.proposal_geom_temporal_logit_head(
                torch.cat(
                    [
                        candidate_tokens,
                        geom_ctx,
                        cand_geom,
                        quality_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            quality_logits = quality_logits + fusion_scale * logit_delta

        if self.proposal_clip_context_fusion:
            clip_ctx = self._proposal_clip_context(
                candidate_tokens,
                candidate_quads,
                proposal_source_ids,
            )
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            clip_scale = float(self.proposal_clip_context_scale)
            token_delta = self.proposal_clip_context_token_proj(
                torch.cat([candidate_tokens, clip_ctx, cand_geom], dim=-1)
            )
            candidate_tokens = candidate_tokens + clip_scale * token_delta
            logit_delta = self.proposal_clip_context_logit_head(
                torch.cat(
                    [
                        candidate_tokens,
                        clip_ctx,
                        cand_geom,
                        quality_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            quality_logits = quality_logits + clip_scale * logit_delta

        if self.proposal_frame_anchor_memory:
            frame_ctx, frame_anchor_logits, frame_anchor_weights = self._proposal_frame_anchor_memory(
                candidate_tokens,
                candidate_quads,
                proposal_source_ids,
                quality_logits,
            )
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            anchor_scale = float(self.proposal_frame_anchor_scale)
            token_delta = self.proposal_frame_anchor_token_proj(
                torch.cat([candidate_tokens, frame_ctx, cand_geom], dim=-1)
            )
            candidate_tokens = candidate_tokens + anchor_scale * token_delta
            logit_delta = self.proposal_frame_anchor_logit_head(
                torch.cat(
                    [
                        candidate_tokens,
                        frame_ctx,
                        cand_geom,
                        quality_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            quality_logits = quality_logits + anchor_scale * logit_delta

        if self.proposal_support_overlap_score and float(self.proposal_support_overlap_scale) != 0.0:
            support_for_score = support_logits_8.detach() if self.proposal_support_overlap_detach else support_logits_8
            support_flat = support_for_score.reshape(b, t8, h * w).transpose(1, 2)
            support16_flat = F.interpolate(
                support_flat,
                size=16,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            support16 = support16_flat.reshape(b, 16, h, w)
            score_grid = int(self.proposal_support_overlap_grid_size)
            if score_grid != h:
                support16 = F.interpolate(
                    support16.reshape(b * 16, 1, h, w),
                    size=(score_grid, score_grid),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, 16, score_grid, score_grid)
            support_prob = support16.sigmoid().to(dtype=candidate_quads.dtype).unsqueeze(2)
            cand_mask = quad_soft_inside_grid(
                candidate_quads.reshape(-1, 4, 2),
                grid_size=score_grid,
                edge_sharpness=96.0,
            ).to(dtype=candidate_quads.dtype).reshape(b, 16, k_select, score_grid, score_grid)
            mask_area = cand_mask.sum(dim=(-1, -2)).clamp_min(1.0)
            support_mean = (cand_mask * support_prob).sum(dim=(-1, -2)) / mask_area
            proposal_support_overlap_scores = torch.logit(support_mean.clamp(1e-4, 1.0 - 1e-4))
            quality_logits = quality_logits + float(self.proposal_support_overlap_scale) * proposal_support_overlap_scores.to(
                dtype=quality_logits.dtype
            )

        if self.proposal_support_geom_score and float(self.proposal_support_geom_scale) != 0.0:
            support_for_geom = support_logits_8.detach() if self.proposal_support_geom_detach else support_logits_8
            support_flat = support_for_geom.reshape(b, t8, h * w).transpose(1, 2)
            support16_flat = F.interpolate(
                support_flat,
                size=16,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            support16 = support16_flat.reshape(b, 16, h, w)
            geom_grid = int(self.proposal_support_geom_grid_size)
            if geom_grid != h:
                support16 = F.interpolate(
                    support16.reshape(b * 16, 1, h, w),
                    size=(geom_grid, geom_grid),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, 16, geom_grid, geom_grid)
            support_prob = support16.sigmoid().to(dtype=candidate_quads.dtype)
            coords = torch.linspace(
                0.0,
                1.0,
                geom_grid,
                device=device,
                dtype=candidate_quads.dtype,
            )
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            grid_xy = torch.stack([xx, yy], dim=-1)
            support_mass = support_prob.sum(dim=(-1, -2)).clamp_min(1e-4)
            support_center = (support_prob[..., None] * grid_xy).sum(dim=(-3, -2)) / support_mass[..., None]
            support_area = (support_mass / float(geom_grid * geom_grid)).clamp_min(1e-5)
            cand_center = candidate_quads.mean(dim=-2)
            cand_area = quad_polygon_area(candidate_quads.reshape(-1, 4, 2)).reshape(b, 16, k_select)
            center_err = (cand_center - support_center[:, :, None, :]).norm(dim=-1)
            log_area_err = torch.abs(torch.log(cand_area.clamp_min(1e-5) / support_area[:, :, None]))
            raw_geom_score = -(
                center_err / float(self.proposal_support_geom_center_tau)
                + log_area_err / float(self.proposal_support_geom_area_tau)
            )
            score_mean = raw_geom_score.mean(dim=-1, keepdim=True)
            score_std = raw_geom_score.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            proposal_support_geom_scores = (raw_geom_score - score_mean) / score_std
            quality_logits = quality_logits + float(self.proposal_support_geom_scale) * proposal_support_geom_scores.to(
                dtype=quality_logits.dtype
            )

        decoupled_selector_logits = None
        pre_decoupled_quality_logits = None
        if self.proposal_decoupled_selector:
            pre_decoupled_quality_logits = quality_logits
            selector_tokens = candidate_tokens
            selector_geom = candidate_quads.reshape(b, 16, k_select, 8)
            selector_base = quality_logits.unsqueeze(-1)
            if self.proposal_decoupled_selector_detach:
                selector_tokens = selector_tokens.detach()
                selector_geom = selector_geom.detach()
                selector_base = selector_base.detach()
            selector_source = proposal_source_ids.detach().clamp(0, 7)
            selector_source_feat = self.proposal_decoupled_selector_source_emb(selector_source).to(
                dtype=selector_tokens.dtype
            )
            selector_parts = [selector_tokens, selector_source_feat, selector_geom, selector_base]
            if decoupled_selector_readout is not None:
                selector_parts.insert(2, decoupled_selector_readout.to(dtype=selector_tokens.dtype))
            selector_in = torch.cat(selector_parts, dim=-1)
            selector_delta = self.proposal_decoupled_selector_head(selector_in).squeeze(-1)
            if self.proposal_decoupled_selector_replace_logits:
                decoupled_selector_logits = (
                    float(self.proposal_decoupled_selector_base_scale)
                    * quality_logits.detach()
                    + float(self.proposal_decoupled_selector_scale) * selector_delta
                )
                quality_logits = decoupled_selector_logits
            else:
                decoupled_selector_logits = quality_logits + (
                    float(self.proposal_decoupled_selector_scale) * selector_delta
                )
                quality_logits = decoupled_selector_logits

        base_quality_logits = quality_logits
        oicr_refine_logits = None
        if self.proposal_oicr_refine_head_enabled:
            # PCL/OICR trains separate refinement classifiers from the base
            # MIL score.  Here the base selector logit is a detached input, and
            # the new head learns a residual refinement score.  Zero init makes
            # the handoff start exactly from the existing selector.
            oicr_in = torch.cat(
                [
                    candidate_tokens,
                    candidate_quads.reshape(b, 16, k_select, 8),
                    base_quality_logits.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            oicr_refine_logits = base_quality_logits + self.proposal_oicr_refine_head(oicr_in).squeeze(-1)
            if self.proposal_oicr_refine_use_for_selection:
                quality_logits = oicr_refine_logits

        select_quality_logits = quality_logits
        if self.training:
            noise_std = float(self.proposal_selection_train_logit_noise_std)
            if noise_std > 0.0:
                select_quality_logits = select_quality_logits + torch.randn_like(select_quality_logits) * noise_std
            drop_prob = float(self.proposal_selection_train_drop_prob)
            if drop_prob > 0.0 and k_select > 1:
                keep = torch.rand_like(select_quality_logits.float()) >= drop_prob
                has_keep = keep.any(dim=-1, keepdim=True)
                fallback_idx = quality_logits.detach().argmax(dim=-1, keepdim=True)
                fallback_keep = torch.zeros_like(keep, dtype=torch.bool).scatter_(-1, fallback_idx, True)
                keep = torch.where(has_keep, keep, fallback_keep)
                select_quality_logits = select_quality_logits.masked_fill(~keep, -50.0)

        use_hard = self.proposal_selection_mode == "hard" or (
            self.proposal_selection_mode == "train_soft_eval_hard" and not self.training
        )
        if use_hard:
            best = select_quality_logits.argmax(dim=-1)
            weights = F.one_hot(best, num_classes=k_select).to(dtype=select_quality_logits.dtype)
        else:
            select_logits = select_quality_logits / self.proposal_selection_temperature
            topk = min(int(self.proposal_selection_topk), k_select)
            if topk > 0 and topk < k_select:
                topk_idx = select_logits.topk(topk, dim=-1).indices
                masked_logits = torch.full_like(select_logits, torch.finfo(select_logits.dtype).min)
                select_logits = masked_logits.scatter(-1, topk_idx, select_logits.gather(-1, topk_idx))
            weights_soft = F.softmax(select_logits, dim=-1)
            if self.proposal_selection_straight_through:
                best = weights_soft.argmax(dim=-1)
                weights_hard = F.one_hot(best, num_classes=k_select).to(dtype=weights_soft.dtype)
                weights = weights_hard - weights_soft.detach() + weights_soft
            else:
                weights = weights_soft
        soft_quad = (candidate_quads * weights[..., None, None]).sum(dim=2).clamp(0.0, 1.0)
        quad = soft_quad
        if self.proposal_final_fusion:
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            quality_stat = torch.tanh(quality_logits)[..., None]
            cand_weight = weights[..., None]
            cand_fusion = self.proposal_final_fusion_candidate_mlp(
                torch.cat([candidate_tokens, cand_geom, cand_weight, quality_stat], dim=-1)
            )
            fused_token = (cand_fusion * cand_weight).sum(dim=2)
            weight_max = weights.max(dim=-1).values
            weight_entropy = -(
                weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(k_select, 2))
            quality_tanh = torch.tanh(quality_logits)
            quality_mean = (weights * quality_tanh).sum(dim=-1)
            quality_max = quality_tanh.max(dim=-1).values
            fusion_stats = torch.stack([weight_max, weight_entropy, quality_mean, quality_max], dim=-1)
            fusion_delta = torch.tanh(
                self.proposal_final_fusion_head(
                    torch.cat([fused_token, soft_quad.reshape(b, 16, 8), fusion_stats], dim=-1)
                )
            ).reshape(b, 16, 4, 2)
            _fusion_dirs, fusion_lens = quad_axes(soft_quad.reshape(-1, 4, 2))
            fusion_step = fusion_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            quad = (soft_quad + self.proposal_final_fusion_scale * fusion_delta * fusion_step).clamp(0.0, 1.0)
        set_refiner_weights = None
        set_refiner_base_quad = None
        set_refiner_scores = None
        if self.proposal_set_refiner:
            set_tokens = candidate_tokens
            set_quads = candidate_quads.reshape(b, 16, k_select, 8)
            set_logits = quality_logits
            if self.proposal_set_refiner_detach:
                set_tokens = set_tokens.detach()
                set_quads = set_quads.detach()
                set_logits = set_logits.detach()
            set_source = proposal_source_ids.detach().clamp(0, 7)
            set_source_feat = self.proposal_set_refiner_source_emb(set_source).to(dtype=set_tokens.dtype)
            set_item = self.proposal_set_refiner_item_mlp(
                torch.cat([set_tokens, set_source_feat, set_quads, set_logits.unsqueeze(-1)], dim=-1)
            )
            set_score_delta = self.proposal_set_refiner_score_head(set_item).squeeze(-1)
            set_scores = set_logits / float(self.proposal_set_refiner_temperature) + set_score_delta
            set_topk = min(int(self.proposal_set_refiner_topk), k_select)
            if set_topk > 0 and set_topk < k_select:
                topk_idx = set_logits.detach().topk(set_topk, dim=-1).indices
                masked_scores = torch.full_like(set_scores, torch.finfo(set_scores.dtype).min)
                set_scores = masked_scores.scatter(-1, topk_idx, set_scores.gather(-1, topk_idx))
            set_refiner_scores = set_scores
            set_refiner_weights = F.softmax(set_scores, dim=-1)
            set_candidate_quads = candidate_quads.detach() if self.proposal_set_refiner_detach else candidate_quads
            set_refiner_base_quad = (
                set_candidate_quads * set_refiner_weights[..., None, None]
            ).sum(dim=2).clamp(0.0, 1.0)
            set_pooled = (set_item * set_refiner_weights[..., None]).sum(dim=2)
            set_entropy = -(
                set_refiner_weights.clamp_min(1e-6)
                * set_refiner_weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(k_select, 2))
            set_stats = torch.stack(
                [
                    set_refiner_weights.max(dim=-1).values,
                    set_entropy,
                    torch.tanh(set_logits).mean(dim=-1),
                    torch.tanh(set_logits).max(dim=-1).values,
                ],
                dim=-1,
            )
            set_delta = torch.tanh(
                self.proposal_set_refiner_delta_head(
                    torch.cat([set_pooled, set_refiner_base_quad.reshape(b, 16, 8), set_stats], dim=-1)
                )
            ).reshape(b, 16, 4, 2)
            _set_dirs, set_lens = quad_axes(set_refiner_base_quad.reshape(-1, 4, 2))
            set_step = set_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            quad = (
                set_refiner_base_quad
                + float(self.proposal_set_refiner_scale) * set_delta * set_step
            ).clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        out = {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.mean(dim=2).view(b, t8, h, w),
            "support_logits_8": support_logits_8.to(dtype=center_logits_8.dtype).view(b, t8, h, w),
            "coarse_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "coarse_xyxy": box_xyxy,
            "quad_xy": quad,
            "proposal_cxcywh": candidate_cxcywh,
            "proposal_xyxy": candidate_xyxy,
            "proposal_quads": candidate_quads,
            "proposal_logits": quality_logits,
            "proposal_source_ids": proposal_source_ids,
            "proposal_selection_weights": weights,
            "corner_logits_8": corner_logits_8,
        }
        if oicr_refine_logits is not None:
            out["proposal_base_logits"] = base_quality_logits
            out["proposal_oicr_logits"] = oicr_refine_logits
        if decoupled_selector_logits is not None:
            out["proposal_decoupled_selector_logits"] = decoupled_selector_logits
        if pre_decoupled_quality_logits is not None:
            out["proposal_pre_decoupled_logits"] = pre_decoupled_quality_logits
        if proposal_support_overlap_scores is not None:
            out["proposal_support_overlap_scores"] = proposal_support_overlap_scores
        if proposal_support_geom_scores is not None:
            out["proposal_support_geom_scores"] = proposal_support_geom_scores
        if self.proposal_final_fusion:
            out["proposal_fusion_base_quad"] = soft_quad
        if set_refiner_weights is not None:
            out["proposal_set_refiner_weights"] = set_refiner_weights
            out["proposal_set_refiner_base_quad"] = set_refiner_base_quad
        if set_refiner_scores is not None:
            out["proposal_set_refiner_scores"] = set_refiner_scores
        if active_tip_logits is not None:
            out["proposal_active_tip_logits"] = active_tip_logits
        if frame_anchor_logits is not None:
            out["proposal_frame_anchor_logits"] = frame_anchor_logits
        if frame_anchor_weights is not None:
            out["proposal_frame_anchor_weights"] = frame_anchor_weights
        if handoff_source_router_logits is not None:
            out["proposal_handoff_source_router_logits"] = handoff_source_router_logits
        if handoff_source_router_candidate_logits is not None:
            out["proposal_handoff_source_router_candidate_logits"] = handoff_source_router_candidate_logits
        if (self.condition_prior_enabled or self.h259_prior_proposal) and cond_quad is not None:
            out["condition_prior_quad"] = cond_quad
            out["condition_prior_xyxy"] = cond_xyxy
            out["condition_prior_cxcywh"] = cond_cxcywh
        if self.h259_prior_proposal and h259_quad is not None:
            out["h259_prior_quad"] = h259_quad
            out["h259_prior_xyxy"] = h259_xyxy
            out["h259_prior_cxcywh"] = h259_cxcywh
        if traj_query_quad is not None and traj_query_xyxy is not None and traj_query_cxcywh is not None:
            out["trajectory_query_quad"] = traj_query_quad
            out["trajectory_query_xyxy"] = traj_query_xyxy
            out["trajectory_query_cxcywh"] = traj_query_cxcywh
        if corner_proposal_quad is not None:
            out["corner_proposal_quad"] = corner_proposal_quad
        if token_affinity_proposal_quad is not None:
            out["token_affinity_proposal_quad"] = token_affinity_proposal_quad
        if support_moment_proposal_quad is not None:
            out["support_moment_proposal_quad"] = support_moment_proposal_quad
        return out

class H210ProposalPointTemporalRegion(H191QuadLocalPointTemporalRegion):
    """Proposal/rerank Stage1 plus CoTracker-style four-point Stage2."""

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        dn_query_noise_scale: float = 0.0,
        dn_query_groups: int = 1,
        dn_query_noise_min_scale: float = 1.0,
        use_proposal_oracle_refiner: bool = False,
        proposal_oracle_refiner_noise_scale: float = 0.0,
        stage2_bridge_noise_mode: str = "local_only",
        use_tool_trajectory_bias: bool = False,
        use_tool_trajectory_attn: bool = False,
        use_tool_corner_query_bias: bool = False,
        stage1_drop_tool_keypoints: bool = False,
        detach_stage1_refiner: bool = False,
        proposal_count: int = 6,
        proposal_selection_mode: str = "train_soft_eval_hard",
        proposal_selection_temperature: float = 1.0,
        proposal_selection_topk: int = 0,
        proposal_selection_straight_through: bool = False,
        proposal_selection_train_logit_noise_std: float = 0.0,
        proposal_selection_train_drop_prob: float = 0.0,
        proposal_geom_temporal_fusion: bool = False,
        proposal_geom_temporal_fusion_scale: float = 0.30,
        proposal_geom_temporal_center_scale: float = 0.18,
        proposal_geom_temporal_source_penalty: float = 4.0,
        proposal_clip_context_fusion: bool = False,
        proposal_clip_context_scale: float = 0.22,
        proposal_clip_context_dropout: float = 0.10,
        proposal_frame_anchor_memory: bool = False,
        proposal_frame_anchor_scale: float = 0.20,
        proposal_frame_anchor_topk: int = 2,
        proposal_frame_anchor_temperature: float = 0.12,
        proposal_frame_anchor_candidate_temperature: float = 0.45,
        proposal_final_fusion: bool = False,
        proposal_final_fusion_scale: float = 0.18,
        proposal_quality_readout: bool = False,
        proposal_quality_readout_detach: bool = False,
        proposal_quality_readout_replace_logits: bool = False,
        proposal_quality_readout_traj_context: bool = False,
        proposal_quality_readout_points: str = "center_corners",
        proposal_quality_readout_pool: str = "mean",
        proposal_support_overlap_score: bool = False,
        proposal_support_overlap_scale: float = 0.0,
        proposal_support_overlap_grid_size: int = 24,
        proposal_support_overlap_detach: bool = False,
        proposal_support_geom_score: bool = False,
        proposal_support_geom_scale: float = 0.0,
        proposal_support_geom_grid_size: int = 24,
        proposal_support_geom_center_tau: float = 0.07,
        proposal_support_geom_area_tau: float = 0.75,
        proposal_support_geom_detach: bool = False,
        proposal_support_moment_proposal: bool = False,
        proposal_support_moment_top_frac: float = 0.035,
        proposal_support_moment_size_scale: float = 1.8,
        proposal_support_moment_temperature: float = 0.16,
        proposal_support_moment_detach: bool = False,
        proposal_decoupled_selector: bool = False,
        proposal_decoupled_selector_detach: bool = False,
        proposal_decoupled_selector_replace_logits: bool = False,
        proposal_decoupled_selector_scale: float = 1.0,
        proposal_decoupled_selector_base_scale: float = 0.0,
        proposal_decoupled_selector_readout: bool = False,
        proposal_oicr_refine_head: bool = False,
        proposal_oicr_refine_use_for_selection: bool = False,
        proposal_handoff_oicr_refine_times: int = 0,
        proposal_handoff_oicr_refine_scale: float = 1.0,
        proposal_handoff_oicr_refine_aggregate: str = "last",
        proposal_handoff_oicr_refine_detach: bool = False,
        proposal_handoff_oicr_refine_use_for_selection: bool = False,
        refined_quality_readout: bool = False,
        refined_quality_readout_detach: bool = False,
        refined_quality_readout_replace_logits: bool = False,
        refined_quality_readout_points: str = "aligned_grid_3x3",
        refiner_aligned_context: bool = False,
        refiner_aligned_context_points: str = "aligned_grid_3x3",
        refiner_aligned_context_scale: float = 0.35,
        proposal_pointset_refine: bool = False,
        proposal_pointset_refine_scale: float = 0.18,
        proposal_pointset_refine_grad_mul: float = 0.25,
        proposal_pointset_quality_scale: float = 1.0,
        proposal_pointset_refine_update_token: bool = False,
        proposal_set_refiner: bool = False,
        proposal_set_refiner_topk: int = 0,
        proposal_set_refiner_scale: float = 0.22,
        proposal_set_refiner_temperature: float = 0.90,
        proposal_set_refiner_detach: bool = False,
        proposal_local_corr_score: bool = False,
        proposal_local_corr_scale: float = 0.45,
        proposal_local_corr_grid_size: int = 3,
        proposal_local_corr_temperature: float = 0.08,
        proposal_local_corr_detach: bool = False,
        proposal_active_tip_selector: bool = False,
        proposal_active_tip_selector_scale: float = 1.0,
        proposal_active_tip_selector_aux_only: bool = False,
        proposal_active_tip_selector_eval_fuse_scale: float = 0.0,
        proposal_active_tip_feature_fusion: bool = False,
        proposal_active_tip_feature_scale: float = 0.35,
        proposal_trajectory_rerank: bool = False,
        proposal_trajectory_rerank_scale: float = 0.45,
        proposal_trajectory_refine: bool = False,
        proposal_trajectory_refine_scale: float = 0.30,
        refine_all_proposals: bool = False,
        refine_proposal_topk: int = 0,
        refine_recollect_stage1: bool = False,
        refine_recollect_refined_base_only: bool = False,
        condition_prior_proposal: bool = False,
        h259_prior_proposal: bool = False,
        trajectory_query_proposal_count: int = 0,
        trajectory_query_output_mode: str = "oriented_rect",
        trajectory_readout_bias: bool = False,
        condition_prior_readout_bias: bool = False,
        condition_prior_readout_scale: float = 0.45,
        trajectory_anchor_logit_bias_scale: float = 0.0,
        trajectory_anchor_logit_bias_radius: float = 0.22,
        trajectory_anchor_axis_bias_weight: float = 0.0,
        corner_heatmap_proposal: bool = False,
        corner_heatmap_proposal_temperature: float = 0.35,
        corner_heatmap_proposal_detach: bool = False,
        proposal_source_calibration: bool = False,
        proposal_source_calibration_scale: float = 1.0,
        proposal_handoff_source_calibration: bool = False,
        proposal_handoff_source_calibration_scale: float = 1.0,
        proposal_handoff_source4_init: float = 0.0,
        proposal_handoff_source_router: bool = False,
        proposal_handoff_source_router_scale: float = 1.0,
        proposal_handoff_source_router_candidate_select_scale: float = 0.0,
        proposal_handoff_source5_readout: bool = False,
        proposal_handoff_source5_readout_scale: float = 1.0,
        proposal_handoff_source5_readout_grid_size: int = 3,
        proposal_handoff_source5_readout_sources: str = "5",
        proposal_handoff_source5_readout_detach: bool = False,
        proposal_handoff_mask_quality: bool = False,
        proposal_handoff_mask_quality_scale: float = 0.50,
        proposal_handoff_mask_grid_size: int = 7,
        proposal_handoff_mask_expand_scale: float = 1.25,
        proposal_handoff_mask_refine_scale: float = 0.0,
        proposal_handoff_mask_refine_size_scale: float = 2.0,
        proposal_handoff_mask_refine_min_size_px: float = 2.0,
        proposal_handoff_mask_refine_max_ratio: float = 1.05,
        proposal_handoff_mask_quality_sources: str = "5,7",
        proposal_handoff_mask_detach: bool = False,
        proposal_handoff_mask_score_detach: bool = False,
        proposal_handoff_rank_adaptive_scorer: bool = False,
        proposal_handoff_rank_adaptive_scale: float = 0.45,
        proposal_handoff_rank_adaptive_max_rank: int = 512,
        proposal_handoff_rank_adaptive_detach: bool = False,
        proposal_handoff_sparse_set_scorer: bool = False,
        proposal_handoff_sparse_set_scorer_scale: float = 0.35,
        proposal_handoff_sparse_set_scorer_sources: str = "0,5,7",
        proposal_handoff_sparse_set_scorer_detach: bool = False,
        proposal_handoff_source5_refiner: bool = False,
        proposal_handoff_source5_refiner_topk: int = 12,
        proposal_handoff_source5_refiner_temperature: float = 0.75,
        proposal_handoff_source5_refiner_scale: float = 0.35,
        proposal_handoff_source5_refiner_logit_scale: float = 0.65,
        proposal_handoff_source5_refiner_init_logit_bias: float = 0.0,
        proposal_handoff_source5_refiner_detach: bool = False,
        proposal_handoff_source5_refiner_source_id: int = 5,
        proposal_handoff_source5_refiner_output_source_id: int = 6,
        proposal_handoff_topk_query_refiner: bool = False,
        proposal_handoff_topk_query_sources: str = "0,5",
        proposal_handoff_topk_query_topk_per_source: int = 12,
        proposal_handoff_topk_query_grid_size: int = 3,
        proposal_handoff_topk_query_scale: float = 0.30,
        proposal_handoff_topk_query_logit_scale: float = 0.75,
        proposal_handoff_topk_query_output_source_id: int = 6,
        proposal_handoff_topk_query_replace_logits: bool = False,
        proposal_handoff_topk_query_detach: bool = False,
        proposal_handoff_final_fusion: bool = False,
        proposal_handoff_final_fusion_scale: float = 0.18,
        proposal_handoff_detach_selection_geometry: bool = False,
        proposal_handoff_source_representatives_only: bool = False,
        proposal_handoff_source5_topk: int = 0,
        proposal_handoff_router_gated_source: bool = False,
        proposal_handoff_router_normalized_select: bool = False,
        proposal_handoff_router_normalized_source_scale: float = 1.0,
        proposal_handoff_router_normalized_within_scale: float = 1.0,
        token_affinity_proposal: bool = False,
        token_affinity_proposal_count: int = 1,
        token_affinity_proposal_temperature: float = 0.08,
        token_affinity_proposal_top_frac: float = 0.055,
        token_affinity_proposal_size_scale: float = 3.6,
        token_affinity_proposal_extra_top_fracs: str = "",
        token_affinity_proposal_extra_size_scales: str = "",
        token_affinity_proposal_detach: bool = True,
        topology_contact_template_proposal: bool = False,
        topology_contact_template_sizes_px: str = "4x3,6x4,8x4,10x5",
        topology_contact_template_offsets_px: str = "-2,0,2",
        topology_contact_template_logit_bias: float = 0.0,
        topology_contact_template_index_embedding: bool = False,
        topology_contact_template_combo_bias: bool = False,
        topology_contact_template_combo_bias_scale: float = 1.0,
        topology_contact_template_combo_allowlist: str = "",
        local_grid_refine: bool = False,
        local_grid_refine_grid_size: int = 5,
        local_grid_refine_scale: float = 1.35,
        local_grid_refine_temperature: float = 0.20,
        local_grid_refine_prior_scale: float = 6.0,
        local_grid_refine_prior_sigma: float = 0.28,
        local_grid_refine_detach_base: bool = False,
        local_grid_final_refine: bool = True,
        local_grid_confidence_score: bool = False,
        local_grid_confidence_scale: float = 0.35,
        local_grid_confidence_detach: bool = False,
        candidate_mask_quality: bool = False,
        candidate_mask_grid_size: int = 7,
        candidate_mask_score_scale: float = 0.0,
        candidate_mask_score_detach: bool = False,
        use_rgb_sidecar: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=use_tool_keypoints,
            residual_scale=residual_scale,
            decoder_layers=decoder_layers,
            query_noise_scale=query_noise_scale,
            dn_query_noise_scale=dn_query_noise_scale,
            dn_query_groups=dn_query_groups,
            dn_query_noise_min_scale=dn_query_noise_min_scale,
            use_proposal_oracle_refiner=use_proposal_oracle_refiner,
            proposal_oracle_refiner_noise_scale=proposal_oracle_refiner_noise_scale,
            stage2_bridge_noise_mode=stage2_bridge_noise_mode,
            use_tool_trajectory_bias=use_tool_trajectory_bias,
            use_tool_trajectory_attn=use_tool_trajectory_attn,
            use_tool_corner_query_bias=use_tool_corner_query_bias,
            stage1_drop_tool_keypoints=stage1_drop_tool_keypoints,
            detach_stage1_refiner=detach_stage1_refiner,
            local_grid_refine=local_grid_refine,
            local_grid_refine_grid_size=local_grid_refine_grid_size,
            local_grid_refine_scale=local_grid_refine_scale,
            local_grid_refine_temperature=local_grid_refine_temperature,
            local_grid_refine_prior_scale=local_grid_refine_prior_scale,
            local_grid_refine_prior_sigma=local_grid_refine_prior_sigma,
            local_grid_refine_detach_base=local_grid_refine_detach_base,
            local_grid_final_refine=local_grid_final_refine,
            use_rgb_sidecar=use_rgb_sidecar,
        )
        stage1_use_tool_keypoints = bool(use_tool_keypoints) and not bool(stage1_drop_tool_keypoints)
        self.stage1 = H210ProposalOrientedStage1(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=stage1_use_tool_keypoints,
            residual_scale=residual_scale,
            proposal_count=proposal_count,
            proposal_selection_mode=proposal_selection_mode,
            proposal_selection_temperature=proposal_selection_temperature,
            proposal_selection_topk=proposal_selection_topk,
            proposal_selection_straight_through=proposal_selection_straight_through,
            proposal_selection_train_logit_noise_std=proposal_selection_train_logit_noise_std,
            proposal_selection_train_drop_prob=proposal_selection_train_drop_prob,
            proposal_geom_temporal_fusion=proposal_geom_temporal_fusion,
            proposal_geom_temporal_fusion_scale=proposal_geom_temporal_fusion_scale,
            proposal_geom_temporal_center_scale=proposal_geom_temporal_center_scale,
            proposal_geom_temporal_source_penalty=proposal_geom_temporal_source_penalty,
            proposal_clip_context_fusion=proposal_clip_context_fusion,
            proposal_clip_context_scale=proposal_clip_context_scale,
            proposal_clip_context_dropout=proposal_clip_context_dropout,
            proposal_frame_anchor_memory=proposal_frame_anchor_memory,
            proposal_frame_anchor_scale=proposal_frame_anchor_scale,
            proposal_frame_anchor_topk=proposal_frame_anchor_topk,
            proposal_frame_anchor_temperature=proposal_frame_anchor_temperature,
            proposal_frame_anchor_candidate_temperature=proposal_frame_anchor_candidate_temperature,
            proposal_final_fusion=proposal_final_fusion,
            proposal_final_fusion_scale=proposal_final_fusion_scale,
            proposal_quality_readout=proposal_quality_readout,
            proposal_quality_readout_detach=proposal_quality_readout_detach,
            proposal_quality_readout_replace_logits=proposal_quality_readout_replace_logits,
            proposal_quality_readout_traj_context=proposal_quality_readout_traj_context,
            proposal_quality_readout_points=proposal_quality_readout_points,
            proposal_quality_readout_pool=proposal_quality_readout_pool,
            proposal_support_overlap_score=proposal_support_overlap_score,
            proposal_support_overlap_scale=proposal_support_overlap_scale,
            proposal_support_overlap_grid_size=proposal_support_overlap_grid_size,
            proposal_support_overlap_detach=proposal_support_overlap_detach,
            proposal_support_geom_score=proposal_support_geom_score,
            proposal_support_geom_scale=proposal_support_geom_scale,
            proposal_support_geom_grid_size=proposal_support_geom_grid_size,
            proposal_support_geom_center_tau=proposal_support_geom_center_tau,
            proposal_support_geom_area_tau=proposal_support_geom_area_tau,
            proposal_support_geom_detach=proposal_support_geom_detach,
            proposal_support_moment_proposal=proposal_support_moment_proposal,
            proposal_support_moment_top_frac=proposal_support_moment_top_frac,
            proposal_support_moment_size_scale=proposal_support_moment_size_scale,
            proposal_support_moment_temperature=proposal_support_moment_temperature,
            proposal_support_moment_detach=proposal_support_moment_detach,
            proposal_decoupled_selector=proposal_decoupled_selector,
            proposal_decoupled_selector_detach=proposal_decoupled_selector_detach,
            proposal_decoupled_selector_replace_logits=proposal_decoupled_selector_replace_logits,
            proposal_decoupled_selector_scale=proposal_decoupled_selector_scale,
            proposal_decoupled_selector_base_scale=proposal_decoupled_selector_base_scale,
            proposal_decoupled_selector_readout=proposal_decoupled_selector_readout,
            proposal_oicr_refine_head=proposal_oicr_refine_head,
            proposal_oicr_refine_use_for_selection=proposal_oicr_refine_use_for_selection,
            proposal_pointset_refine=proposal_pointset_refine,
            proposal_pointset_refine_scale=proposal_pointset_refine_scale,
            proposal_pointset_refine_grad_mul=proposal_pointset_refine_grad_mul,
            proposal_pointset_quality_scale=proposal_pointset_quality_scale,
            proposal_pointset_refine_update_token=proposal_pointset_refine_update_token,
            proposal_set_refiner=proposal_set_refiner,
            proposal_set_refiner_topk=proposal_set_refiner_topk,
            proposal_set_refiner_scale=proposal_set_refiner_scale,
            proposal_set_refiner_temperature=proposal_set_refiner_temperature,
            proposal_set_refiner_detach=proposal_set_refiner_detach,
            proposal_local_corr_score=proposal_local_corr_score,
            proposal_local_corr_scale=proposal_local_corr_scale,
            proposal_local_corr_grid_size=proposal_local_corr_grid_size,
            proposal_local_corr_temperature=proposal_local_corr_temperature,
            proposal_local_corr_detach=proposal_local_corr_detach,
            proposal_active_tip_selector=proposal_active_tip_selector,
            proposal_active_tip_selector_scale=proposal_active_tip_selector_scale,
            proposal_active_tip_selector_aux_only=proposal_active_tip_selector_aux_only,
            proposal_active_tip_selector_eval_fuse_scale=proposal_active_tip_selector_eval_fuse_scale,
            proposal_active_tip_feature_fusion=proposal_active_tip_feature_fusion,
            proposal_active_tip_feature_scale=proposal_active_tip_feature_scale,
            proposal_trajectory_rerank=proposal_trajectory_rerank,
            proposal_trajectory_rerank_scale=proposal_trajectory_rerank_scale,
            proposal_trajectory_refine=proposal_trajectory_refine,
            proposal_trajectory_refine_scale=proposal_trajectory_refine_scale,
            condition_prior_proposal=condition_prior_proposal,
            h259_prior_proposal=h259_prior_proposal,
            trajectory_query_proposal_count=trajectory_query_proposal_count,
            trajectory_query_output_mode=trajectory_query_output_mode,
            trajectory_readout_bias=trajectory_readout_bias,
            condition_prior_readout_bias=condition_prior_readout_bias,
            condition_prior_readout_scale=condition_prior_readout_scale,
            trajectory_anchor_logit_bias_scale=trajectory_anchor_logit_bias_scale,
            trajectory_anchor_logit_bias_radius=trajectory_anchor_logit_bias_radius,
            trajectory_anchor_axis_bias_weight=trajectory_anchor_axis_bias_weight,
            corner_heatmap_proposal=corner_heatmap_proposal,
            corner_heatmap_proposal_temperature=corner_heatmap_proposal_temperature,
            corner_heatmap_proposal_detach=corner_heatmap_proposal_detach,
            proposal_source_calibration=proposal_source_calibration,
            proposal_source_calibration_scale=proposal_source_calibration_scale,
            proposal_handoff_source_router=proposal_handoff_source_router,
            proposal_handoff_source_router_scale=proposal_handoff_source_router_scale,
            proposal_handoff_router_normalized_select=proposal_handoff_router_normalized_select,
            proposal_handoff_router_normalized_source_scale=proposal_handoff_router_normalized_source_scale,
            proposal_handoff_router_normalized_within_scale=proposal_handoff_router_normalized_within_scale,
            token_affinity_proposal=token_affinity_proposal,
            token_affinity_proposal_count=token_affinity_proposal_count,
            token_affinity_proposal_temperature=token_affinity_proposal_temperature,
            token_affinity_proposal_top_frac=token_affinity_proposal_top_frac,
            token_affinity_proposal_size_scale=token_affinity_proposal_size_scale,
            token_affinity_proposal_extra_top_fracs=token_affinity_proposal_extra_top_fracs,
            token_affinity_proposal_extra_size_scales=token_affinity_proposal_extra_size_scales,
            token_affinity_proposal_detach=token_affinity_proposal_detach,
            topology_contact_template_proposal=topology_contact_template_proposal,
            topology_contact_template_sizes_px=topology_contact_template_sizes_px,
            topology_contact_template_offsets_px=topology_contact_template_offsets_px,
            topology_contact_template_logit_bias=topology_contact_template_logit_bias,
            topology_contact_template_index_embedding=topology_contact_template_index_embedding,
            topology_contact_template_combo_bias=topology_contact_template_combo_bias,
            topology_contact_template_combo_bias_scale=topology_contact_template_combo_bias_scale,
            topology_contact_template_combo_allowlist=topology_contact_template_combo_allowlist,
            use_rgb_sidecar=use_rgb_sidecar,
        )
        self.refine_all_proposals = bool(refine_all_proposals)
        self.refine_proposal_topk = max(int(refine_proposal_topk), 0)
        self.refine_recollect_stage1 = bool(refine_recollect_stage1)
        self.refine_recollect_refined_base_only = bool(refine_recollect_refined_base_only)
        self.local_grid_confidence_score = bool(local_grid_confidence_score)
        self.local_grid_confidence_scale = float(local_grid_confidence_scale)
        self.local_grid_confidence_detach = bool(local_grid_confidence_detach)
        self.candidate_mask_quality = bool(candidate_mask_quality)
        self.candidate_mask_grid_size = max(int(candidate_mask_grid_size), 2)
        self.candidate_mask_score_scale = float(candidate_mask_score_scale)
        self.candidate_mask_score_detach = bool(candidate_mask_score_detach)
        self.proposal_local_corr_score = bool(proposal_local_corr_score) and bool(use_tool_keypoints)
        self.proposal_local_corr_scale = float(proposal_local_corr_scale)
        self.proposal_local_corr_grid_size = max(int(proposal_local_corr_grid_size), 2)
        self.proposal_local_corr_temperature = max(float(proposal_local_corr_temperature), 1e-3)
        self.proposal_local_corr_detach = bool(proposal_local_corr_detach)
        self.proposal_handoff_source_calibration = bool(proposal_handoff_source_calibration)
        self.proposal_handoff_source_calibration_scale = float(proposal_handoff_source_calibration_scale)
        self.proposal_handoff_source_router = bool(proposal_handoff_source_router)
        self.proposal_handoff_source_router_scale = float(proposal_handoff_source_router_scale)
        self.proposal_handoff_source_router_candidate_select_scale = float(
            proposal_handoff_source_router_candidate_select_scale
        )
        self.proposal_handoff_source5_readout = bool(proposal_handoff_source5_readout)
        self.proposal_handoff_source5_readout_scale = float(proposal_handoff_source5_readout_scale)
        self.proposal_handoff_source5_readout_grid_size = max(
            int(proposal_handoff_source5_readout_grid_size),
            2,
        )
        readout_sources = sorted(
            {
                max(0, min(7, int(src)))
                for src in parse_int_csv(proposal_handoff_source5_readout_sources)
            }
        )
        if not readout_sources:
            readout_sources = [5]
        self.proposal_handoff_source5_readout_sources = tuple(readout_sources)
        self.proposal_handoff_source5_readout_detach = bool(
            proposal_handoff_source5_readout_detach
        )
        self.proposal_handoff_mask_quality = bool(proposal_handoff_mask_quality)
        self.proposal_handoff_mask_quality_scale = float(proposal_handoff_mask_quality_scale)
        self.proposal_handoff_mask_grid_size = max(int(proposal_handoff_mask_grid_size), 2)
        self.proposal_handoff_mask_expand_scale = max(float(proposal_handoff_mask_expand_scale), 0.25)
        self.proposal_handoff_mask_refine_scale = max(
            float(proposal_handoff_mask_refine_scale),
            0.0,
        )
        self.proposal_handoff_mask_refine_size_scale = max(
            float(proposal_handoff_mask_refine_size_scale),
            0.1,
        )
        self.proposal_handoff_mask_refine_min_size_px = max(
            float(proposal_handoff_mask_refine_min_size_px),
            0.25,
        )
        self.proposal_handoff_mask_refine_max_ratio = max(
            float(proposal_handoff_mask_refine_max_ratio),
            0.1,
        )
        handoff_mask_sources = sorted(
            {
                max(0, min(7, int(src)))
                for src in parse_int_csv(proposal_handoff_mask_quality_sources)
            }
        )
        if not handoff_mask_sources:
            handoff_mask_sources = [5, 7]
        self.proposal_handoff_mask_quality_sources = tuple(handoff_mask_sources)
        self.proposal_handoff_mask_detach = bool(proposal_handoff_mask_detach)
        self.proposal_handoff_mask_score_detach = bool(proposal_handoff_mask_score_detach)
        self.proposal_handoff_rank_adaptive_scorer = bool(
            proposal_handoff_rank_adaptive_scorer
        )
        self.proposal_handoff_rank_adaptive_scale = float(
            proposal_handoff_rank_adaptive_scale
        )
        self.proposal_handoff_rank_adaptive_max_rank = max(
            int(proposal_handoff_rank_adaptive_max_rank),
            2,
        )
        self.proposal_handoff_rank_adaptive_detach = bool(
            proposal_handoff_rank_adaptive_detach
        )
        self.proposal_handoff_sparse_set_scorer = bool(proposal_handoff_sparse_set_scorer)
        self.proposal_handoff_sparse_set_scorer_scale = float(
            proposal_handoff_sparse_set_scorer_scale
        )
        sparse_set_sources = sorted(
            {
                max(0, min(7, int(src)))
                for src in parse_int_csv(proposal_handoff_sparse_set_scorer_sources)
            }
        )
        if not sparse_set_sources:
            sparse_set_sources = [0, 5, 7]
        self.proposal_handoff_sparse_set_scorer_sources = tuple(sparse_set_sources)
        self.proposal_handoff_sparse_set_scorer_detach = bool(
            proposal_handoff_sparse_set_scorer_detach
        )
        self.proposal_handoff_source5_refiner = bool(proposal_handoff_source5_refiner)
        self.proposal_handoff_source5_refiner_topk = max(
            int(proposal_handoff_source5_refiner_topk),
            0,
        )
        self.proposal_handoff_source5_refiner_temperature = max(
            float(proposal_handoff_source5_refiner_temperature),
            1e-3,
        )
        self.proposal_handoff_source5_refiner_scale = max(
            float(proposal_handoff_source5_refiner_scale),
            0.0,
        )
        self.proposal_handoff_source5_refiner_logit_scale = float(
            proposal_handoff_source5_refiner_logit_scale
        )
        self.proposal_handoff_source5_refiner_init_logit_bias = float(
            proposal_handoff_source5_refiner_init_logit_bias
        )
        self.proposal_handoff_source5_refiner_detach = bool(
            proposal_handoff_source5_refiner_detach
        )
        self.proposal_handoff_source5_refiner_source_id = int(
            max(0, min(7, int(proposal_handoff_source5_refiner_source_id)))
        )
        self.proposal_handoff_source5_refiner_output_source_id = int(
            max(0, min(7, int(proposal_handoff_source5_refiner_output_source_id)))
        )
        self.proposal_handoff_topk_query_refiner = bool(
            proposal_handoff_topk_query_refiner
        )
        topk_query_sources = sorted(
            {
                int(max(0, min(7, src)))
                for src in parse_int_csv(proposal_handoff_topk_query_sources)
            }
        )
        if not topk_query_sources:
            topk_query_sources = [0, 5]
        self.proposal_handoff_topk_query_sources = tuple(topk_query_sources)
        self.proposal_handoff_topk_query_topk_per_source = max(
            int(proposal_handoff_topk_query_topk_per_source),
            1,
        )
        self.proposal_handoff_topk_query_grid_size = max(
            int(proposal_handoff_topk_query_grid_size),
            2,
        )
        self.proposal_handoff_topk_query_scale = max(
            float(proposal_handoff_topk_query_scale),
            0.0,
        )
        self.proposal_handoff_topk_query_logit_scale = float(
            proposal_handoff_topk_query_logit_scale
        )
        self.proposal_handoff_topk_query_output_source_id = int(
            max(0, min(7, int(proposal_handoff_topk_query_output_source_id)))
        )
        self.proposal_handoff_topk_query_replace_logits = bool(
            proposal_handoff_topk_query_replace_logits
        )
        self.proposal_handoff_topk_query_detach = bool(
            proposal_handoff_topk_query_detach
        )
        self.proposal_handoff_final_fusion = bool(proposal_handoff_final_fusion)
        self.proposal_handoff_final_fusion_scale = max(float(proposal_handoff_final_fusion_scale), 0.0)
        self.proposal_handoff_detach_selection_geometry = bool(
            proposal_handoff_detach_selection_geometry
        )
        self.proposal_handoff_source_representatives_only = bool(
            proposal_handoff_source_representatives_only
        )
        self.proposal_handoff_source5_topk = max(int(proposal_handoff_source5_topk), 0)
        self.proposal_handoff_router_gated_source = bool(proposal_handoff_router_gated_source)
        self.proposal_handoff_router_normalized_select = bool(
            proposal_handoff_router_normalized_select
        )
        self.proposal_handoff_router_normalized_source_scale = float(
            proposal_handoff_router_normalized_source_scale
        )
        self.proposal_handoff_router_normalized_within_scale = float(
            proposal_handoff_router_normalized_within_scale
        )
        self.proposal_handoff_oicr_refine_times = max(
            int(proposal_handoff_oicr_refine_times),
            0,
        )
        self.proposal_handoff_oicr_refine_scale = float(
            proposal_handoff_oicr_refine_scale
        )
        if proposal_handoff_oicr_refine_aggregate not in {"last", "mean", "mean_base"}:
            proposal_handoff_oicr_refine_aggregate = "last"
        self.proposal_handoff_oicr_refine_aggregate = str(
            proposal_handoff_oicr_refine_aggregate
        )
        self.proposal_handoff_oicr_refine_detach = bool(
            proposal_handoff_oicr_refine_detach
        )
        self.proposal_handoff_oicr_refine_use_for_selection = bool(
            proposal_handoff_oicr_refine_use_for_selection
        )
        self.refined_quality_readout = bool(refined_quality_readout)
        self.refined_quality_readout_detach = bool(refined_quality_readout_detach)
        self.refined_quality_readout_replace_logits = bool(refined_quality_readout_replace_logits)
        self.refiner_aligned_context = bool(refiner_aligned_context)
        if refiner_aligned_context_points not in {"aligned_grid_3x3", "aligned_grid_5x5"}:
            refiner_aligned_context_points = "aligned_grid_3x3"
        self.refiner_aligned_context_points = str(refiner_aligned_context_points)
        self.refiner_aligned_context_scale = float(refiner_aligned_context_scale)
        if refined_quality_readout_points not in {
            "center_corners",
            "center_corners_edges",
            "aligned_grid_3x3",
            "aligned_grid_5x5",
        }:
            refined_quality_readout_points = "aligned_grid_3x3"
        self.refined_quality_readout_points = str(refined_quality_readout_points)
        self.proposal_count = max(int(proposal_count), 2)
        extra_proposal_count = 0
        if bool(condition_prior_proposal):
            extra_proposal_count += 1
        if stage1_use_tool_keypoints:
            if bool(h259_prior_proposal):
                extra_proposal_count += 1
            extra_proposal_count += max(int(trajectory_query_proposal_count), 0)
        if bool(corner_heatmap_proposal):
            extra_proposal_count += 1
        if bool(token_affinity_proposal):
            variant_count = token_affinity_variant_count(
                token_affinity_proposal_extra_top_fracs,
                token_affinity_proposal_extra_size_scales,
            )
            extra_proposal_count += max(int(token_affinity_proposal_count), 1) * variant_count
        if stage1_use_tool_keypoints and bool(topology_contact_template_proposal):
            extra_proposal_count += topology_contact_template_count(
                topology_contact_template_sizes_px,
                topology_contact_template_offsets_px,
                mask_res=64,
            )
        self.refine_candidate_count = self.proposal_count + extra_proposal_count
        if self.refine_all_proposals:
            self.candidate_refine_embed = nn.Parameter(
                torch.zeros(1, 1, self.refine_candidate_count, 1, hidden_dim)
            )
            self.refined_quality_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.refined_quality_readout:
                self.refined_quality_readout_proj = nn.Sequential(
                    nn.LayerNorm(3 * hidden_dim),
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.refined_quality_readout_head = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim + 8),
                    nn.Linear(2 * hidden_dim + 8, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.refined_quality_readout_head[-1].weight)
                nn.init.zeros_(self.refined_quality_readout_head[-1].bias)
            if self.refiner_aligned_context:
                self.refiner_aligned_context_proj = nn.Sequential(
                    nn.LayerNorm(3 * hidden_dim),
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            if self.candidate_mask_quality:
                mask_in_dim = 2 * hidden_dim + 10
                self.candidate_mask_readout_proj = nn.Sequential(
                    nn.LayerNorm(3 * hidden_dim),
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.candidate_mask_point_head = nn.Sequential(
                    nn.LayerNorm(mask_in_dim),
                    nn.Linear(mask_in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.candidate_mask_quality_head = nn.Sequential(
                    nn.LayerNorm(mask_in_dim),
                    nn.Linear(mask_in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.candidate_mask_point_head[-1].weight)
                nn.init.zeros_(self.candidate_mask_point_head[-1].bias)
                nn.init.zeros_(self.candidate_mask_quality_head[-1].weight)
                nn.init.zeros_(self.candidate_mask_quality_head[-1].bias)
            nn.init.trunc_normal_(self.candidate_refine_embed, std=0.02)
        if self.proposal_handoff_source_calibration:
            self.proposal_handoff_source_logit_bias = nn.Embedding(8, 1)
            nn.init.zeros_(self.proposal_handoff_source_logit_bias.weight)
            with torch.no_grad():
                self.proposal_handoff_source_logit_bias.weight[4, 0] = float(proposal_handoff_source4_init)
        if self.proposal_handoff_source_router:
            self.proposal_handoff_source_router_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_source_router_candidate_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1),
                nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.proposal_handoff_source_router_context_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].bias)
        if self.proposal_handoff_source5_readout:
            self.proposal_handoff_source5_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_source5_readout_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1),
                nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.proposal_handoff_source5_readout_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_readout_head[-1].bias)
        if self.proposal_handoff_mask_quality:
            mask_point_in_dim = 3 * hidden_dim + 10
            mask_quality_in_dim = 3 * hidden_dim + 11
            self.proposal_handoff_mask_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_mask_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_mask_point_head = nn.Sequential(
                nn.LayerNorm(mask_point_in_dim),
                nn.Linear(mask_point_in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.proposal_handoff_mask_quality_head = nn.Sequential(
                nn.LayerNorm(mask_quality_in_dim),
                nn.Linear(mask_quality_in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_mask_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_mask_point_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_mask_point_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_mask_quality_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_mask_quality_head[-1].bias)
        if self.proposal_handoff_rank_adaptive_scorer:
            self.proposal_handoff_rank_adaptive_rank_emb = nn.Embedding(
                self.proposal_handoff_rank_adaptive_max_rank,
                hidden_dim,
            )
            self.proposal_handoff_rank_adaptive_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_rank_adaptive_geom_mlp = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_rank_adaptive_logit_mlp = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, hidden_dim),
            )
            self.proposal_handoff_rank_adaptive_score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_rank_adaptive_rank_emb.weight, std=0.02)
            nn.init.normal_(self.proposal_handoff_rank_adaptive_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_rank_adaptive_score_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_rank_adaptive_score_head[-1].bias)
        if self.proposal_handoff_sparse_set_scorer:
            self.proposal_handoff_sparse_set_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_sparse_set_geom_mlp = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_sparse_set_logit_mlp = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_sparse_set_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                dropout=0.10,
                batch_first=True,
            )
            self.proposal_handoff_sparse_set_attn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_sparse_set_ffn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_sparse_set_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.proposal_handoff_sparse_set_score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_sparse_set_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_sparse_set_score_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_sparse_set_score_head[-1].bias)
        if self.proposal_handoff_oicr_refine_times > 0:
            self.proposal_handoff_oicr_refine_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim + 8 + 1),
                        nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(self.proposal_handoff_oicr_refine_times)
                ]
            )
            for head in self.proposal_handoff_oicr_refine_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        if self.proposal_handoff_source5_refiner:
            self.proposal_handoff_source5_refiner_item_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1),
                nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_source5_refiner_score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.proposal_handoff_source5_refiner_delta_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 4),
                nn.Linear(hidden_dim + 8 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            self.proposal_handoff_source5_refiner_logit_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 4),
                nn.Linear(hidden_dim + 8 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.proposal_handoff_source5_refiner_score_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_score_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_delta_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_logit_head[-1].bias)
        if self.proposal_handoff_topk_query_refiner:
            topk_query_in_dim = 3 * hidden_dim + 8 + 1
            self.proposal_handoff_topk_query_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_topk_query_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_topk_query_item_mlp = nn.Sequential(
                nn.LayerNorm(topk_query_in_dim),
                nn.Linear(topk_query_in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_topk_query_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                dropout=0.10,
                batch_first=True,
            )
            self.proposal_handoff_topk_query_attn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_topk_query_ffn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_topk_query_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.proposal_handoff_topk_query_delta_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8),
                nn.Linear(hidden_dim + 8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            self.proposal_handoff_topk_query_logit_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8),
                nn.Linear(hidden_dim + 8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_topk_query_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_topk_query_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_topk_query_delta_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_topk_query_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_topk_query_logit_head[-1].bias)
        if self.proposal_handoff_final_fusion:
            self.proposal_handoff_final_fusion_candidate_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 2),
                nn.Linear(hidden_dim + 8 + 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_final_fusion_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 4),
                nn.Linear(hidden_dim + 8 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            nn.init.zeros_(self.proposal_handoff_final_fusion_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_final_fusion_head[-1].bias)
        if self.proposal_local_corr_score:
            self.proposal_local_corr_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_local_corr_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1 + 12 + 6),
                nn.Linear(hidden_dim + 8 + 1 + 12 + 6, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.proposal_local_corr_head[-1].weight)
            nn.init.zeros_(self.proposal_local_corr_head[-1].bias)

    def _proposal_local_corr_logits(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_logits: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, k_select = candidate_quads.shape[:3]
        device = candidate_quads.device
        dtype = candidate_quads.dtype
        grid_size = int(self.proposal_local_corr_grid_size)
        cand_pts = self.stage1._candidate_aligned_grid_points(candidate_quads, grid_size=grid_size)
        cand_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            cand_pts.reshape(b, 16, k_select * grid_size * grid_size, 2),
        )
        cand_readout = cand_readout.reshape(b, 16, k_select, grid_size * grid_size, -1)
        cand_feat = self.proposal_local_corr_readout_proj(cand_readout)

        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        tool_readout = self.stage1._sample_traj_readout_features(feat16_map, kpt_xy.reshape(b, 16, 3, 2))
        tool_feat = self.proposal_local_corr_readout_proj(tool_readout)

        cand_norm = F.normalize(cand_feat.float(), dim=-1, eps=1e-6)
        tool_norm = F.normalize(tool_feat.float(), dim=-1, eps=1e-6)
        sim = torch.einsum("btkgc,btpc->btkgp", cand_norm, tool_norm)
        vis = kpt_vis[:, :, None, None, :].float()
        sim = sim.masked_fill(vis <= 0.05, -1.0)
        flat = sim.reshape(b, 16, k_select, -1)
        max_sim = flat.max(dim=-1).values
        valid = (flat > -0.999).float()
        valid_count = valid.sum(dim=-1).clamp_min(1.0)
        mean_sim = (flat * valid).sum(dim=-1) / valid_count
        var_sim = (((flat - mean_sim[..., None]) * valid) ** 2).sum(dim=-1) / valid_count
        std_sim = var_sim.clamp_min(0.0).sqrt()
        tau = float(self.proposal_local_corr_temperature)
        prob = F.softmax(flat / tau, dim=-1)
        entropy = -(
            prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()
        ).sum(dim=-1) / math.log(max(int(flat.shape[-1]), 2))
        vis_frac = kpt_vis.mean(dim=-1)[:, :, None].expand(-1, -1, k_select)
        corr_stats = torch.stack(
            [
                max_sim,
                mean_sim,
                std_sim,
                max_sim - mean_sim,
                1.0 - entropy.to(device=device, dtype=max_sim.dtype),
                vis_frac.to(device=device, dtype=max_sim.dtype),
            ],
            dim=-1,
        ).to(device=device, dtype=dtype)

        cand_context = cand_feat.mean(dim=3).to(dtype=dtype)
        tip_feats = self.stage1._active_tip_candidate_features(
            candidate_quads,
            tool_kpt_vec_16,
            role_features_16,
            tool_id,
        )
        corr_tail = torch.cat(
            [
                candidate_quads.reshape(b, 16, k_select, 8),
                candidate_logits.unsqueeze(-1),
                tip_feats,
                corr_stats,
            ],
            dim=-1,
        )
        if self.proposal_local_corr_detach:
            corr_tail = corr_tail.detach()
            cand_context = cand_context.detach()
        corr_logits = self.proposal_local_corr_head(
            torch.cat([cand_context, corr_tail], dim=-1)
        ).squeeze(-1)
        return corr_logits.to(device=device, dtype=dtype), corr_stats

    def _select_refined_proposals(
        self,
        candidate_quads: torch.Tensor,
        quality_logits: torch.Tensor,
        detach_geometry: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = candidate_quads.shape[2]
        use_hard = self.stage1.proposal_selection_mode == "hard" or (
            self.stage1.proposal_selection_mode == "train_soft_eval_hard" and not self.training
        )
        if use_hard:
            best = quality_logits.argmax(dim=-1)
            weights = F.one_hot(best, num_classes=k).to(dtype=quality_logits.dtype)
        else:
            select_logits = quality_logits / self.stage1.proposal_selection_temperature
            topk = min(int(self.stage1.proposal_selection_topk), k)
            if topk > 0 and topk < k:
                topk_idx = select_logits.topk(topk, dim=-1).indices
                masked_logits = torch.full_like(select_logits, torch.finfo(select_logits.dtype).min)
                select_logits = masked_logits.scatter(-1, topk_idx, select_logits.gather(-1, topk_idx))
            weights_soft = F.softmax(select_logits, dim=-1)
            if self.stage1.proposal_selection_straight_through:
                best = weights_soft.argmax(dim=-1)
                weights_hard = F.one_hot(best, num_classes=k).to(dtype=weights_soft.dtype)
                weights = weights_hard - weights_soft.detach() + weights_soft
            else:
                weights = weights_soft
        handoff_quads = candidate_quads.detach() if detach_geometry else candidate_quads
        quad = (handoff_quads * weights[..., None, None]).sum(dim=2).clamp(0.0, 1.0)
        return quad, weights

    def _local_grid_refine_candidate_quads(
        self,
        feat16_map: torch.Tensor,
        base_quads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the local corner search head independently to every proposal."""
        b, t, k = base_quads.shape[:3]
        grid_size = int(self.local_grid_refine_grid_size)
        search_quads = base_quads.detach() if self.local_grid_refine_detach_base else base_quads
        grid_pts, rel = self._local_grid_points(
            search_quads,
            grid_size=grid_size,
            scale=float(self.local_grid_refine_scale),
        )
        p = grid_pts.shape[3]
        point_feat = self._sample_point_features(
            feat16_map,
            grid_pts.reshape(b, t, k * p, 2),
        ).reshape(b, t, k, p, -1)
        rel_feat = rel.to(device=point_feat.device, dtype=point_feat.dtype)
        rel_feat = rel_feat.reshape(1, 1, 1, p, 2).expand(b, t, k, -1, -1)
        grid_feat = self.local_grid_point_proj(torch.cat([point_feat, rel_feat], dim=-1))
        corner_feat = self.local_grid_corner_embed.to(device=grid_feat.device, dtype=grid_feat.dtype)
        corner_feat = corner_feat.reshape(1, 1, 1, 4, -1).expand(b, t, k, -1, -1)
        corner_grid_feat = grid_feat[:, :, :, None, :, :].expand(-1, -1, -1, 4, -1, -1)
        corner_token = corner_feat[:, :, :, :, None, :].expand(-1, -1, -1, -1, p, -1)
        grid_xy = grid_pts[:, :, :, None, :, :].expand(-1, -1, -1, 4, -1, -1)
        logits = self.local_grid_corner_head(
            torch.cat([corner_grid_feat, corner_token, grid_xy], dim=-1)
        ).squeeze(-1)
        if self.local_grid_refine_prior_scale != 0.0:
            prior_quad = search_quads.to(device=grid_pts.device, dtype=grid_pts.dtype)
            _dirs, lens = quad_axes(prior_quad.reshape(-1, 4, 2))
            side = lens.mean(dim=-1).reshape(b, t, k, 1, 1).clamp_min(1e-4)
            prior_dist2 = (grid_xy - prior_quad[:, :, :, :, None, :]).square().sum(dim=-1)
            sigma = float(self.local_grid_refine_prior_sigma) * side
            corner_prior = -prior_dist2 / (2.0 * sigma.square().clamp_min(1e-8))
            logits = logits + float(self.local_grid_refine_prior_scale) * corner_prior
        probs = F.softmax(logits / float(self.local_grid_refine_temperature), dim=-1)
        quads = (probs[..., None] * grid_xy).sum(dim=-2).clamp(0.0, 1.0)
        return quads, logits, grid_pts

    def _local_grid_confidence_logits(self, corner_logits: torch.Tensor) -> torch.Tensor:
        """Convert local-grid corner peak margins into per-candidate scores."""
        logits = corner_logits.float()
        if logits.shape[-1] < 2:
            conf = logits.mean(dim=(-1, -2))
        else:
            top2 = logits.topk(2, dim=-1).values
            conf = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
        mean = conf.mean(dim=-1, keepdim=True)
        std = conf.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (conf - mean) / std

    def _candidate_mask_quality_outputs(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict candidate-local contact masks and an IoU-aware quality logit."""
        b, t, k = candidate_quads.shape[:3]
        grid_size = int(self.candidate_mask_grid_size)
        p_grid = grid_size * grid_size
        mask_pts = self.stage1._candidate_aligned_grid_points(
            candidate_quads,
            grid_size=grid_size,
        )
        mask_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            mask_pts.reshape(b, t, k * p_grid, 2),
        )
        mask_readout = mask_readout.reshape(b, t, k, p_grid, -1)
        point_feat = self.candidate_mask_readout_proj(mask_readout)
        coords = torch.linspace(
            -1.0,
            1.0,
            grid_size,
            device=candidate_quads.device,
            dtype=candidate_quads.dtype,
        )
        vv, uu = torch.meshgrid(coords, coords, indexing="ij")
        uv = torch.stack([uu, vv], dim=-1).reshape(1, 1, 1, p_grid, 2)
        uv = uv.expand(b, t, k, -1, -1).to(device=point_feat.device, dtype=point_feat.dtype)
        geom = candidate_quads.reshape(b, t, k, 8).to(device=point_feat.device, dtype=point_feat.dtype)
        token = candidate_tokens.to(device=point_feat.device, dtype=point_feat.dtype)
        point_token = token[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_geom = geom[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_logits = self.candidate_mask_point_head(
            torch.cat([point_token, point_feat, point_geom, uv], dim=-1)
        ).squeeze(-1)
        prob = torch.sigmoid(point_logits.float()).to(dtype=point_feat.dtype)
        pooled_feat = (point_feat * prob[..., None]).sum(dim=3) / prob.sum(dim=3, keepdim=True).clamp_min(1e-4)
        mask_stats = torch.stack(
            [
                prob.mean(dim=3),
                prob.amax(dim=3),
            ],
            dim=-1,
        ).to(device=point_feat.device, dtype=point_feat.dtype)
        quality_logits = self.candidate_mask_quality_head(
            torch.cat([token, pooled_feat, geom, mask_stats], dim=-1)
        ).squeeze(-1)
        return point_logits, mask_pts, quality_logits

    def _proposal_handoff_mask_quality_outputs(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_source_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict source-conditioned local masks and IoU-quality logits.

        Deformable DETR samples sparse evidence around each reference box.
        Mask Scoring R-CNN then scores a predicted region using both region
        evidence and the predicted mask.  This lightweight version keeps our
        existing proposal bank fixed but gives source5/source7 candidates a
        deployable local evidence head before final handoff ranking.
        """
        b, t, k = candidate_quads.shape[:3]
        grid_size = int(self.proposal_handoff_mask_grid_size)
        p_grid = grid_size * grid_size
        sample_quads = candidate_quads
        expand = float(self.proposal_handoff_mask_expand_scale)
        if abs(expand - 1.0) > 1e-6:
            center = candidate_quads.mean(dim=-2, keepdim=True)
            sample_quads = (center + (candidate_quads - center) * expand).clamp(0.0, 1.0)
        mask_pts = self.stage1._candidate_aligned_grid_points(
            sample_quads,
            grid_size=grid_size,
        )
        mask_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            mask_pts.reshape(b, t, k * p_grid, 2),
        ).reshape(b, t, k, p_grid, -1)
        point_feat = self.proposal_handoff_mask_readout_proj(mask_readout)
        coords = torch.linspace(
            -1.0,
            1.0,
            grid_size,
            device=candidate_quads.device,
            dtype=candidate_quads.dtype,
        )
        vv, uu = torch.meshgrid(coords, coords, indexing="ij")
        uv = torch.stack([uu, vv], dim=-1).reshape(1, 1, 1, p_grid, 2)
        uv = uv.expand(b, t, k, -1, -1).to(device=point_feat.device, dtype=point_feat.dtype)
        source_ids = candidate_source_ids.detach().clamp(0, 7)
        source_emb = self.proposal_handoff_mask_source_emb(source_ids).to(dtype=point_feat.dtype)
        geom = candidate_quads.reshape(b, t, k, 8).to(device=point_feat.device, dtype=point_feat.dtype)
        token = candidate_tokens.to(device=point_feat.device, dtype=point_feat.dtype)
        base_logit = candidate_logits.detach().unsqueeze(-1).to(device=point_feat.device, dtype=point_feat.dtype)
        if self.proposal_handoff_mask_detach:
            point_feat = point_feat.detach()
            geom = geom.detach()
            token = token.detach()
            source_emb = source_emb.detach()
        point_token = token[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_source = source_emb[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_geom = geom[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_logits = self.proposal_handoff_mask_point_head(
            torch.cat([point_token, point_feat, point_source, point_geom, uv], dim=-1)
        ).squeeze(-1)
        prob = torch.sigmoid(point_logits.float()).to(dtype=point_feat.dtype)
        pooled_feat = (point_feat * prob[..., None]).sum(dim=3) / prob.sum(
            dim=3,
            keepdim=True,
        ).clamp_min(1e-4)
        mask_stats = torch.stack(
            [
                prob.mean(dim=3),
                prob.amax(dim=3),
            ],
            dim=-1,
        ).to(device=point_feat.device, dtype=point_feat.dtype)
        quality_logits = self.proposal_handoff_mask_quality_head(
            torch.cat([token, pooled_feat, source_emb, geom, mask_stats, base_logit], dim=-1)
        ).squeeze(-1)
        source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
        for source_id in self.proposal_handoff_mask_quality_sources:
            source_mask = source_mask | (source_ids == int(source_id))
        return point_logits, mask_pts, quality_logits, source_mask

    def _proposal_handoff_mask_refined_quads(
        self,
        base_quads: torch.Tensor,
        mask_points: torch.Tensor,
        mask_logits: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode candidate-local point-mask logits into aligned moment quads."""
        b, t, k = base_quads.shape[:3]
        dtype = base_quads.dtype
        prob = torch.sigmoid(mask_logits.float()).to(dtype=dtype)
        weights = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        center = (weights[..., None] * mask_points.to(dtype=dtype)).sum(dim=-2)

        dirs, lens = quad_axes(base_quads.reshape(-1, 4, 2))
        dirs = dirs.reshape(b, t, k, 2, 2).to(device=base_quads.device, dtype=dtype)
        lens = lens.reshape(b, t, k, 2).to(device=base_quads.device, dtype=dtype)
        delta = mask_points.to(dtype=dtype) - center[..., None, :]
        proj = torch.einsum("btkpd,btkad->btkpa", delta, dirs)
        var = (weights[..., None] * proj.square()).sum(dim=-2).clamp_min(1e-8)
        min_size = float(self.proposal_handoff_mask_refine_min_size_px) / 64.0
        max_size = lens * float(self.proposal_handoff_mask_refine_max_ratio)
        size = (
            float(self.proposal_handoff_mask_refine_size_scale) * torch.sqrt(var)
        ).clamp_min(min_size)
        size = torch.minimum(size, max_size.clamp_min(min_size))
        moment_quad = oriented_rect_to_quad(center, size, dirs[..., 0, :]).to(dtype=dtype)
        refined = (
            base_quads
            + float(self.proposal_handoff_mask_refine_scale) * (moment_quad - base_quads)
        ).clamp(0.0, 1.0)
        return torch.where(source_mask[..., None, None], refined, base_quads)

    def _proposal_handoff_topk_query_outputs(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_source_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Refine a fixed top-k source subset as independent proposal queries."""
        b, t, k = candidate_quads.shape[:3]
        source_ids = candidate_source_ids.detach().clamp(0, 7)
        topk_each = min(int(self.proposal_handoff_topk_query_topk_per_source), k)
        query_indices = []
        query_valid = []
        for source_id in self.proposal_handoff_topk_query_sources:
            source_mask = source_ids == int(source_id)
            source_scores = candidate_logits.masked_fill(~source_mask, -50.0)
            vals, idx = source_scores.topk(topk_each, dim=-1)
            valid = source_mask.gather(-1, idx) & (vals > -49.0)
            query_indices.append(idx)
            query_valid.append(valid)
        topk_idx = torch.cat(query_indices, dim=-1)
        valid_mask = torch.cat(query_valid, dim=-1)
        q = int(topk_idx.shape[-1])

        gather_quad_idx = topk_idx[..., None, None].expand(-1, -1, -1, 4, 2)
        query_quads = candidate_quads.gather(2, gather_quad_idx)
        gather_token_idx = topk_idx[..., None].expand(-1, -1, -1, candidate_tokens.shape[-1])
        query_tokens = candidate_tokens.gather(2, gather_token_idx)
        query_logits = candidate_logits.gather(-1, topk_idx)
        query_sources = source_ids.gather(-1, topk_idx)

        grid_size = int(self.proposal_handoff_topk_query_grid_size)
        p_grid = grid_size * grid_size
        query_pts = self.stage1._candidate_aligned_grid_points(
            query_quads,
            grid_size=grid_size,
        )
        query_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            query_pts.reshape(b, t, q * p_grid, 2),
        ).reshape(b, t, q, p_grid, -1)
        query_readout = self.proposal_handoff_topk_query_readout_proj(
            query_readout.mean(dim=3)
        )
        query_geom = query_quads.reshape(b, t, q, 8)
        query_source_emb = self.proposal_handoff_topk_query_source_emb(query_sources).to(
            dtype=query_tokens.dtype
        )
        item_tokens = query_tokens
        item_geom = query_geom
        item_logits = query_logits
        if self.proposal_handoff_topk_query_detach:
            item_tokens = item_tokens.detach()
            item_geom = item_geom.detach()
            item_logits = item_logits.detach()
            query_readout = query_readout.detach()
            query_source_emb = query_source_emb.detach()
        query_item = self.proposal_handoff_topk_query_item_mlp(
            torch.cat(
                [
                    item_tokens,
                    query_readout.to(dtype=item_tokens.dtype),
                    query_source_emb,
                    item_geom.to(dtype=item_tokens.dtype),
                    item_logits.unsqueeze(-1).to(dtype=item_tokens.dtype),
                ],
                dim=-1,
            )
        )
        flat = query_item.reshape(b * t, q, -1)
        valid_flat = valid_mask.reshape(b * t, q)
        key_padding = ~valid_flat
        has_valid = valid_flat.any(dim=-1)
        if key_padding.numel() > 0:
            key_padding = key_padding.clone()
            key_padding[~has_valid, 0] = False
        attn, _ = self.proposal_handoff_topk_query_attn(
            flat,
            flat,
            flat,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        flat = self.proposal_handoff_topk_query_attn_norm(flat + attn)
        flat = self.proposal_handoff_topk_query_ffn_norm(
            flat + self.proposal_handoff_topk_query_ffn(flat)
        )
        flat = flat * valid_flat.unsqueeze(-1).to(dtype=flat.dtype)
        query_item = flat.reshape(b, t, q, -1)

        delta_in = torch.cat(
            [query_item, query_geom.to(device=query_item.device, dtype=query_item.dtype)],
            dim=-1,
        )
        query_delta = torch.tanh(
            self.proposal_handoff_topk_query_delta_head(delta_in)
        ).reshape(b, t, q, 4, 2)
        _query_dirs, query_lens = quad_axes(query_quads.reshape(-1, 4, 2))
        query_step = query_lens.mean(dim=-1).reshape(b, t, q, 1, 1).clamp(0.015, 0.35)
        refined_quads = (
            query_quads
            + float(self.proposal_handoff_topk_query_scale) * query_delta * query_step
        ).clamp(0.0, 1.0)
        query_logit_delta = self.proposal_handoff_topk_query_logit_head(delta_in).squeeze(-1)
        query_logit_delta = query_logit_delta.to(device=query_logits.device, dtype=query_logits.dtype)
        if self.proposal_handoff_topk_query_replace_logits:
            refined_logits = float(self.proposal_handoff_topk_query_logit_scale) * query_logit_delta
        else:
            refined_logits = query_logits + float(
                self.proposal_handoff_topk_query_logit_scale
            ) * query_logit_delta
        refined_logits = torch.where(
            valid_mask,
            refined_logits,
            torch.full_like(refined_logits, -50.0),
        )
        return refined_quads, query_quads, refined_logits, valid_mask, query_sources, query_item

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        dn_quad_xy: torch.Tensor | None = None,
        rgb_16: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.refine_all_proposals:
            return super().forward(
                tokens=tokens,
                role_features_16=role_features_16,
                tool_mask_8=tool_mask_8,
                tool_kpt_heatmaps_8=tool_kpt_heatmaps_8,
                tool_kpt_vec_16=tool_kpt_vec_16,
                tool_id=tool_id,
                act_id=act_id,
                combo_id=combo_id,
                dn_quad_xy=dn_quad_xy,
                rgb_16=rgb_16,
            )

        stage1_kwargs: dict[str, torch.Tensor] = {
            "tokens": tokens,
            "role_features_16": role_features_16,
            "tool_mask_8": tool_mask_8,
            "tool_kpt_heatmaps_8": tool_kpt_heatmaps_8,
            "tool_kpt_vec_16": tool_kpt_vec_16,
            "tool_id": tool_id,
            "act_id": act_id,
            "combo_id": combo_id,
        }
        if getattr(self.stage1, "use_rgb_sidecar", False):
            stage1_kwargs["rgb_16"] = rgb_16
        stage1 = self.stage1(**stage1_kwargs)
        stage1_full_proposal_quads = stage1["proposal_quads"]
        stage1_full_proposal_xyxy = stage1["proposal_xyxy"]
        stage1_full_proposal_logits = stage1["proposal_logits"]
        topk_indices = None
        if self.refine_proposal_topk > 0 and self.refine_proposal_topk < stage1["proposal_quads"].shape[2]:
            topk = min(self.refine_proposal_topk, stage1["proposal_quads"].shape[2])
            topk_indices = stage1["proposal_logits"].topk(topk, dim=-1).indices

            def gather_candidates(x: torch.Tensor) -> torch.Tensor:
                expand_shape = topk_indices.shape + x.shape[3:]
                gather_idx = topk_indices.reshape(*topk_indices.shape, *([1] * (x.dim() - 3))).expand(expand_shape)
                return x.gather(2, gather_idx)

            stage1 = dict(stage1)
            for key in ("proposal_quads", "proposal_xyxy", "proposal_cxcywh"):
                if key in stage1:
                    stage1[key] = gather_candidates(stage1[key])
            stage1["proposal_logits"] = gather_candidates(stage1["proposal_logits"].unsqueeze(-1)).squeeze(-1)
            if "proposal_source_ids" in stage1:
                stage1["proposal_source_ids"] = gather_candidates(stage1["proposal_source_ids"])
            if "proposal_selection_weights" in stage1:
                stage1["proposal_selection_weights"] = gather_candidates(stage1["proposal_selection_weights"])

        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        k = stage1["proposal_quads"].shape[2]
        p = k * 4

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        spatial_parts = [x]
        if self.use_rgb_sidecar:
            if self.rgb_encoder is None:
                raise RuntimeError("RGB sidecar is enabled without an encoder")
            spatial_parts.append(encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype))
        spatial_parts.append(grid)
        if self.use_tool_mask:
            spatial_parts.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            spatial_parts.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(spatial_parts, dim=2).reshape(b * t8, -1, h, w))
        mem8 = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        mem16_chw = F.interpolate(mem8, size=(16, h, w), mode="trilinear", align_corners=True)
        feat16_map = mem16_chw.permute(0, 2, 1, 3, 4).contiguous()
        mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b, 16, h * w, -1)

        query_base = self.role_mlp(role_features_16) + z[:, None, :].expand(-1, 16, -1)
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            query_base = query_base + self.tool_mask_geom_mlp(mask_geom16)
        if self.use_tool_keypoints:
            query_base = query_base + self.tool_kpt_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype))
            if self.use_tool_trajectory_bias:
                traj_bias = self.tool_traj_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype).reshape(b, 16 * 9))
                query_base = query_base + traj_bias[:, None, :]
        tool_memory = None
        if self.use_tool_trajectory_attn:
            tool_memory = self.tool_traj_token_mlp(
                torch.cat(
                    [
                        tool_kpt_vec_16.to(device=device, dtype=dtype),
                        role_features_16.to(device=device, dtype=dtype),
                    ],
                    dim=-1,
                )
            )
            tool_memory = tool_memory + z[:, None, :] + self.tool_time_pos.to(device=device, dtype=dtype)
        corner_query_bias = 0.0
        if self.use_tool_corner_query_bias:
            tool_anchor_quad = self._tool_anchor_quad(tool_kpt_vec_16.to(device=device, dtype=dtype))
            corner_query_bias = self.tool_corner_ref_mlp(tool_anchor_quad)
        corner_query_bias_all = (
            corner_query_bias[:, :, None, :, :] if isinstance(corner_query_bias, torch.Tensor) else 0.0
        )

        ref_clean = stage1["proposal_quads"].clamp(0.0, 1.0)
        initial_xyxy = stage1["proposal_xyxy"]
        if self.detach_stage1_refiner:
            ref_clean = ref_clean.detach()
            initial_xyxy = initial_xyxy.detach()
        initial_wh = xyxy_to_cxcywh_torch(initial_xyxy)[..., 2:].detach()
        step_scale = (self.residual_scale * initial_wh[:, :, :, None, :]).clamp(0.01, 0.50)
        step_scale_points = step_scale.expand(-1, -1, -1, 4, -1).reshape(b, 16, p, 2)

        if self.stage2_bridge_noise_mode == "replace":
            ref = self._maybe_perturb_points(ref_clean.reshape(b, 16, p, 2), step_scale_points)
            ref = ref.reshape(b, 16, k, 4, 2)
        else:
            ref = ref_clean

        q = (
            query_base[:, :, None, None, :]
            + self.candidate_refine_embed[:, :, :k].to(device=device, dtype=dtype)
            + self.point_index.to(device=device, dtype=dtype)[:, :, None, :, :]
            + self.time_pos.to(device=device, dtype=dtype).unsqueeze(2)
            + self.point_ref_mlp(ref)
            + corner_query_bias_all
        )
        q = self.query_norm(q).reshape(b, 16, p, -1)

        layer_boxes = []
        layer_xyxy = []
        layer_quads = []
        mem_bt = mem16.reshape(b * 16, h * w, -1)
        for i in range(self.decoder_layers):
            sample_ref = ref
            if self.stage2_bridge_noise_mode == "local_only":
                sample_ref = self._maybe_perturb_points(
                    ref.reshape(b, 16, p, 2),
                    step_scale_points,
                ).reshape(b, 16, k, 4, 2)
            if self.refiner_aligned_context:
                grid_size = 5 if self.refiner_aligned_context_points == "aligned_grid_5x5" else 3
                aligned_pts = self.stage1._candidate_aligned_grid_points(sample_ref, grid_size=grid_size)
                aligned_ctx = self.stage1._sample_traj_readout_features(
                    feat16_map,
                    aligned_pts.reshape(b, 16, k * grid_size * grid_size, 2),
                )
                aligned_ctx = aligned_ctx.reshape(b, 16, k, grid_size * grid_size, -1).mean(dim=3)
                aligned_ctx = self.refiner_aligned_context_proj(aligned_ctx)
                aligned_ctx = aligned_ctx[:, :, :, None, :].expand(-1, -1, -1, 4, -1).reshape(b, 16, p, -1)
                q = q + float(self.refiner_aligned_context_scale) * aligned_ctx
            q = q + self.local_point_proj(
                self._sample_point_features(feat16_map, sample_ref.reshape(b, 16, p, 2))
            )

            q_frame = q.reshape(b * 16, p, -1)
            frame_out, _ = self.frame_attn[i](
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                need_weights=False,
            )
            q = (q_frame + frame_out).reshape(b, 16, p, -1)

            q_tem = q.permute(0, 2, 1, 3).reshape(b * p, 16, -1)
            tem_out, _ = self.temporal_attn[i](
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                need_weights=False,
            )
            q = (q_tem + tem_out).reshape(b, p, 16, -1).permute(0, 2, 1, 3)

            if tool_memory is not None:
                q_tool = q.reshape(b, 16 * p, -1)
                tool_out, _ = self.tool_traj_attn[i](
                    self.tool_query_norms[i](q_tool),
                    self.tool_memory_norms[i](tool_memory),
                    self.tool_memory_norms[i](tool_memory),
                    need_weights=False,
                )
                q = (q_tool + tool_out).reshape(b, 16, p, -1)

            ref_flat = ref.reshape(b, 16, p, 2)
            q_ref = q + self.point_ref_mlp(ref_flat)
            cross_q = self.cross_norms[i](q_ref).reshape(b * 16, p, -1)
            mem_norm = self.memory_norms[i](mem_bt)
            cross_out, _ = self.cross_attn[i](cross_q, mem_norm, mem_norm, need_weights=False)
            q = q + cross_out.reshape(b, 16, p, -1)
            q = q + self.ffns[i](self.ffn_norms[i](q))

            delta = self.point_delta_heads[i](q).reshape(b, 16, k, 4, 2)
            ref_new = (ref + torch.tanh(delta) * step_scale).clamp(0.0, 1.0)
            layer_quads.append(ref_new)
            box_xyxy = quad_to_xyxy(ref_new.reshape(b, 16 * k, 4, 2), min_size=0.02).reshape(b, 16, k, 4)
            layer_xyxy.append(box_xyxy)
            layer_boxes.append(xyxy_to_cxcywh_torch(box_xyxy))
            ref = ref_new.detach() if i + 1 < self.decoder_layers else ref_new

        refined_quads = layer_quads[-1]
        candidate_local_grid_logits = None
        candidate_local_grid_points = None
        candidate_local_grid_base_quads = None
        if self.local_grid_refine:
            candidate_local_grid_base_quads = refined_quads
            refined_quads, candidate_local_grid_logits, candidate_local_grid_points = (
                self._local_grid_refine_candidate_quads(
                    feat16_map,
                    refined_quads,
                )
            )
        refined_xyxy = quad_to_xyxy(refined_quads.reshape(b, 16 * k, 4, 2), min_size=0.02).reshape(b, 16, k, 4)
        refined_cxcywh = xyxy_to_cxcywh_torch(refined_xyxy)
        refined_tokens = q.reshape(b, 16, k, 4, -1).mean(dim=3)
        proposal_oracle_refiner_payload: dict[str, torch.Tensor] = {}
        if self.use_proposal_oracle_refiner and dn_quad_xy is not None and self.training:
            gt_ref = dn_quad_xy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            with torch.no_grad():
                gt_prop = gt_ref[:, :, None, :, :].expand_as(ref_clean)
                prop_quad_iou = quad_iou_approx(
                    ref_clean.reshape(-1, 4, 2).float(),
                    gt_prop.reshape(-1, 4, 2).float(),
                    grid_size=24,
                ).reshape(ref_clean.shape[:-2])
                oracle_idx = prop_quad_iou.argmax(dim=-1)
            gather_quad_idx = oracle_idx[:, :, None, None, None].expand(-1, -1, 1, 4, 2)
            gather_box_idx = oracle_idx[:, :, None, None].expand(-1, -1, 1, 4)
            oracle_start = ref_clean.gather(2, gather_quad_idx).squeeze(2).detach()
            oracle_quad = refined_quads.gather(2, gather_quad_idx).squeeze(2)
            oracle_xyxy = refined_xyxy.gather(2, gather_box_idx).squeeze(2)
            oracle_cxcywh = refined_cxcywh.gather(2, gather_box_idx).squeeze(2)
            proposal_oracle_refiner_payload = {
                "proposal_oracle_refiner_start_quad": oracle_start,
                "proposal_oracle_refiner_quad": oracle_quad,
                "proposal_oracle_refiner_xyxy": oracle_xyxy,
                "proposal_oracle_refiner_cxcywh": oracle_cxcywh,
            }
        q_grouped = refined_tokens
        refined_logits = self.refined_quality_head(q_grouped).squeeze(-1)
        if self.refined_quality_readout:
            refined_center = refined_quads.mean(dim=-2, keepdim=True)
            if self.refined_quality_readout_points == "aligned_grid_5x5":
                readout_pts = self.stage1._candidate_aligned_grid_points(refined_quads, grid_size=5)
                points_per_candidate = 25
            elif self.refined_quality_readout_points == "aligned_grid_3x3":
                readout_pts = self.stage1._candidate_aligned_grid_points(refined_quads, grid_size=3)
                points_per_candidate = 9
            elif self.refined_quality_readout_points == "center_corners_edges":
                edge_midpoints = 0.5 * (refined_quads + torch.roll(refined_quads, shifts=-1, dims=-2))
                readout_pts = torch.cat([refined_center, refined_quads, edge_midpoints], dim=3)
                points_per_candidate = 9
            else:
                readout_pts = torch.cat([refined_center, refined_quads], dim=3)
                points_per_candidate = 5

            readout_pts = readout_pts.reshape(b, 16, k * points_per_candidate, 2)
            refined_readout = self.stage1._sample_traj_readout_features(feat16_map, readout_pts)
            refined_readout = refined_readout.reshape(b, 16, k, points_per_candidate, -1).mean(dim=3)
            refined_readout = self.refined_quality_readout_proj(refined_readout)
            refined_geom = refined_quads.reshape(b, 16, k, 8)
            readout_tokens = q_grouped
            if self.refined_quality_readout_detach:
                readout_tokens = readout_tokens.detach()
                refined_readout = refined_readout.detach()
                refined_geom = refined_geom.detach()
            refined_quality_delta = self.refined_quality_readout_head(
                torch.cat([readout_tokens, refined_readout, refined_geom], dim=-1)
            ).squeeze(-1)
            if self.refined_quality_readout_replace_logits:
                refined_logits = refined_quality_delta
            else:
                refined_logits = refined_logits + refined_quality_delta
        candidate_local_grid_confidence = None
        if self.local_grid_confidence_score and candidate_local_grid_logits is not None:
            candidate_local_grid_confidence = self._local_grid_confidence_logits(candidate_local_grid_logits)
            if self.local_grid_confidence_detach:
                candidate_local_grid_confidence = candidate_local_grid_confidence.detach()
            refined_logits = refined_logits + float(self.local_grid_confidence_scale) * candidate_local_grid_confidence.to(
                device=refined_logits.device,
                dtype=refined_logits.dtype,
            )
        candidate_mask_logits = None
        candidate_mask_points = None
        candidate_mask_quality_logits = None
        candidate_mask_base_quads = None
        if self.candidate_mask_quality:
            candidate_mask_base_quads = refined_quads
            (
                candidate_mask_logits,
                candidate_mask_points,
                candidate_mask_quality_logits,
            ) = self._candidate_mask_quality_outputs(
                feat16_map,
                refined_quads,
                refined_tokens,
            )
            mask_quality_score = candidate_mask_quality_logits
            if self.candidate_mask_score_detach:
                mask_quality_score = mask_quality_score.detach()
            if float(self.candidate_mask_score_scale) != 0.0:
                refined_logits = refined_logits + float(self.candidate_mask_score_scale) * mask_quality_score.to(
                    device=refined_logits.device,
                    dtype=refined_logits.dtype,
                )
        _refined_quad, refined_weights = self._select_refined_proposals(refined_quads, refined_logits)
        select_quads = refined_quads
        select_xyxy = refined_xyxy
        select_cxcywh = refined_cxcywh
        select_logits = refined_logits
        select_tokens = refined_tokens
        select_source_ids = stage1.get("proposal_source_ids")
        select_active_tip_logits = stage1.get("proposal_active_tip_logits")
        if self.refine_recollect_stage1:
            recollect_quads = refined_quads
            recollect_xyxy = refined_xyxy
            recollect_cxcywh = refined_cxcywh
            recollect_logits = refined_logits
            recollect_source_ids = stage1.get("proposal_source_ids")
            recollect_active_tip_logits = select_active_tip_logits
            if self.refine_recollect_refined_base_only:
                base_k = min(int(self.stage1.proposal_count), int(refined_quads.shape[2]))
                recollect_quads = recollect_quads[:, :, :base_k]
                recollect_xyxy = recollect_xyxy[:, :, :base_k]
                recollect_cxcywh = recollect_cxcywh[:, :, :base_k]
                recollect_logits = recollect_logits[:, :, :base_k]
                recollect_tokens = refined_tokens[:, :, :base_k]
                if recollect_source_ids is not None:
                    recollect_source_ids = recollect_source_ids[:, :, :base_k]
                if recollect_active_tip_logits is not None:
                    recollect_active_tip_logits = recollect_active_tip_logits[:, :, :base_k]
            else:
                recollect_tokens = refined_tokens
            select_quads = torch.cat([stage1["proposal_quads"], recollect_quads], dim=2)
            select_xyxy = torch.cat([stage1["proposal_xyxy"], recollect_xyxy], dim=2)
            select_cxcywh = torch.cat([stage1["proposal_cxcywh"], recollect_cxcywh], dim=2)
            select_logits = torch.cat([stage1["proposal_logits"], recollect_logits], dim=-1)
            stage1_tokens = refined_tokens[:, :, : stage1["proposal_quads"].shape[2]]
            select_tokens = torch.cat([stage1_tokens, recollect_tokens], dim=2)
            if "proposal_source_ids" in stage1:
                select_source_ids = torch.cat(
                    [stage1["proposal_source_ids"], recollect_source_ids],
                    dim=2,
                )
            if select_active_tip_logits is not None and recollect_active_tip_logits is not None:
                select_active_tip_logits = torch.cat(
                    [select_active_tip_logits, recollect_active_tip_logits],
                    dim=-1,
                )
        proposal_local_corr_logits = None
        proposal_local_corr_stats = None
        if self.proposal_local_corr_score and tool_kpt_vec_16 is not None:
            proposal_local_corr_logits, proposal_local_corr_stats = self._proposal_local_corr_logits(
                feat16_map,
                select_quads,
                select_logits,
                tool_kpt_vec_16,
                role_features_16,
                tool_id,
            )
            select_logits = select_logits + float(self.proposal_local_corr_scale) * proposal_local_corr_logits.to(
                device=select_logits.device,
                dtype=select_logits.dtype,
            )
        handoff_source5_readout_logits = None
        if (
            self.proposal_handoff_source5_readout
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            select_k = int(select_quads.shape[2])
            source5_grid_size = int(self.proposal_handoff_source5_readout_grid_size)
            source5_pts = self.stage1._candidate_aligned_grid_points(
                select_quads,
                grid_size=source5_grid_size,
            )
            source5_readout = self.stage1._sample_traj_readout_features(
                feat16_map,
                source5_pts.reshape(b, 16, select_k * source5_grid_size * source5_grid_size, 2),
            )
            source5_readout = source5_readout.reshape(
                b,
                16,
                select_k,
                source5_grid_size * source5_grid_size,
                -1,
            ).mean(dim=3)
            source5_readout = self.proposal_handoff_source5_readout_proj(source5_readout)
            source5_geom = select_quads.reshape(b, 16, select_k, 8)
            source5_tail = torch.cat(
                [
                    source5_readout,
                    source5_geom,
                    select_logits.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            if self.proposal_handoff_source5_readout_detach:
                source5_tail = source5_tail.detach()
            handoff_source5_readout_logits = (
                self.proposal_handoff_source5_readout_head(source5_tail).squeeze(-1)
            )
            readout_source_ids = select_source_ids.detach().clamp(0, 7)
            source5_mask = torch.zeros_like(readout_source_ids, dtype=torch.bool)
            for source_id in self.proposal_handoff_source5_readout_sources:
                source5_mask = source5_mask | (readout_source_ids == int(source_id))
            select_logits = select_logits + (
                float(self.proposal_handoff_source5_readout_scale)
                * handoff_source5_readout_logits.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
                * source5_mask.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        handoff_mask_logits = None
        handoff_mask_points = None
        handoff_mask_quality_logits = None
        handoff_mask_source_mask = None
        handoff_mask_base_quads = None
        handoff_mask_refined_quads = None
        if (
            self.proposal_handoff_mask_quality
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            handoff_mask_base_quads = select_quads
            (
                handoff_mask_logits,
                handoff_mask_points,
                handoff_mask_quality_logits,
                handoff_mask_source_mask,
            ) = self._proposal_handoff_mask_quality_outputs(
                feat16_map,
                select_quads,
                select_tokens,
                select_logits,
                select_source_ids,
            )
            if float(self.proposal_handoff_mask_refine_scale) > 0.0:
                handoff_mask_refined_quads = self._proposal_handoff_mask_refined_quads(
                    handoff_mask_base_quads,
                    handoff_mask_points,
                    handoff_mask_logits,
                    handoff_mask_source_mask,
                )
                select_quads = handoff_mask_refined_quads
                select_xyxy = quad_to_xyxy(select_quads.reshape(b, 16 * select_quads.shape[2], 4, 2), min_size=0.02).reshape(
                    b,
                    16,
                    select_quads.shape[2],
                    4,
                )
                select_cxcywh = xyxy_to_cxcywh_torch(select_xyxy)
            handoff_mask_score = handoff_mask_quality_logits
            if self.proposal_handoff_mask_score_detach:
                handoff_mask_score = handoff_mask_score.detach()
            select_logits = select_logits + (
                float(self.proposal_handoff_mask_quality_scale)
                * handoff_mask_score.to(device=select_logits.device, dtype=select_logits.dtype)
                * handoff_mask_source_mask.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        handoff_rank_adaptive_delta = None
        handoff_rank_adaptive_rank = None
        if (
            self.proposal_handoff_rank_adaptive_scorer
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            # Rank-DETR reorders queries by current confidence and injects
            # rank-aware query content before later scoring.  We keep the bank
            # fixed, but expose that same detached rank context to a residual
            # score head so high-quality candidates can be calibrated by their
            # current position in the handoff list.
            rank_base_logits = select_logits.detach()
            rank_order = rank_base_logits.argsort(dim=-1, descending=True)
            handoff_rank_adaptive_rank = torch.empty_like(rank_order)
            rank_values = torch.arange(
                int(rank_order.shape[-1]),
                device=rank_order.device,
                dtype=rank_order.dtype,
            )
            handoff_rank_adaptive_rank.scatter_(
                -1,
                rank_order,
                rank_values.view(*([1] * (rank_order.dim() - 1)), -1).expand_as(rank_order),
            )
            rank_idx = handoff_rank_adaptive_rank.clamp(
                max=int(self.proposal_handoff_rank_adaptive_max_rank) - 1
            )
            rank_tokens = select_tokens
            rank_quads = select_quads
            rank_logits = select_logits
            if self.proposal_handoff_rank_adaptive_detach:
                rank_tokens = rank_tokens.detach()
                rank_quads = rank_quads.detach()
                rank_logits = rank_logits.detach()
            rank_source_ids = select_source_ids.detach().clamp(0, 7)
            rank_item = (
                rank_tokens
                + self.proposal_handoff_rank_adaptive_rank_emb(rank_idx).to(
                    dtype=rank_tokens.dtype
                )
                + self.proposal_handoff_rank_adaptive_source_emb(rank_source_ids).to(
                    dtype=rank_tokens.dtype
                )
                + self.proposal_handoff_rank_adaptive_geom_mlp(
                    rank_quads.reshape(b, 16, rank_quads.shape[2], 8)
                ).to(dtype=rank_tokens.dtype)
                + self.proposal_handoff_rank_adaptive_logit_mlp(
                    rank_logits.unsqueeze(-1)
                ).to(dtype=rank_tokens.dtype)
            )
            handoff_rank_adaptive_delta = (
                self.proposal_handoff_rank_adaptive_score_head(rank_item).squeeze(-1)
            )
            select_logits = select_logits + (
                float(self.proposal_handoff_rank_adaptive_scale)
                * handoff_rank_adaptive_delta.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
            )
        handoff_sparse_set_delta = None
        handoff_sparse_set_mask = None
        if (
            self.proposal_handoff_sparse_set_scorer
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            # Sparse R-CNN's dynamic head lets proposal features interact before
            # scoring/regression.  Here we keep the proposal bank fixed but add
            # the same set interaction before final handoff ranking.
            sparse_tokens = select_tokens
            sparse_quads = select_quads
            sparse_logits = select_logits
            if self.proposal_handoff_sparse_set_scorer_detach:
                sparse_tokens = sparse_tokens.detach()
                sparse_quads = sparse_quads.detach()
                sparse_logits = sparse_logits.detach()
            sparse_k = int(sparse_quads.shape[2])
            sparse_source_ids = select_source_ids.detach().clamp(0, 7)
            sparse_geom = sparse_quads.reshape(b, 16, sparse_k, 8)
            sparse_item = (
                sparse_tokens
                + self.proposal_handoff_sparse_set_source_emb(sparse_source_ids).to(
                    dtype=sparse_tokens.dtype
                )
                + self.proposal_handoff_sparse_set_geom_mlp(sparse_geom).to(
                    dtype=sparse_tokens.dtype
                )
                + self.proposal_handoff_sparse_set_logit_mlp(
                    sparse_logits.unsqueeze(-1)
                ).to(dtype=sparse_tokens.dtype)
            )
            sparse_flat = sparse_item.reshape(b * 16, sparse_k, -1)
            sparse_attn, _ = self.proposal_handoff_sparse_set_attn(
                sparse_flat,
                sparse_flat,
                sparse_flat,
                need_weights=False,
            )
            sparse_flat = self.proposal_handoff_sparse_set_attn_norm(sparse_flat + sparse_attn)
            sparse_flat = self.proposal_handoff_sparse_set_ffn_norm(
                sparse_flat + self.proposal_handoff_sparse_set_ffn(sparse_flat)
            )
            handoff_sparse_set_delta = self.proposal_handoff_sparse_set_score_head(
                sparse_flat
            ).reshape(b, 16, sparse_k)
            handoff_sparse_set_mask = torch.zeros_like(sparse_source_ids, dtype=torch.bool)
            for source_id in self.proposal_handoff_sparse_set_scorer_sources:
                handoff_sparse_set_mask = handoff_sparse_set_mask | (
                    sparse_source_ids == int(source_id)
                )
            select_logits = select_logits + (
                float(self.proposal_handoff_sparse_set_scorer_scale)
                * handoff_sparse_set_delta.to(device=select_logits.device, dtype=select_logits.dtype)
                * handoff_sparse_set_mask.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        handoff_source5_refiner_quad = None
        handoff_source5_refiner_base_quad = None
        handoff_source5_refiner_scores = None
        handoff_source5_refiner_weights = None
        handoff_source5_refiner_valid = None
        handoff_source5_refiner_logit = None
        handoff_topk_query_refiner_quads = None
        handoff_topk_query_refiner_base_quads = None
        handoff_topk_query_refiner_logits = None
        handoff_topk_query_refiner_valid = None
        handoff_topk_query_refiner_source_ids = None
        if (
            self.proposal_handoff_source5_refiner
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            source5_refiner_source_id = int(self.proposal_handoff_source5_refiner_source_id)
            source5_ids = select_source_ids.detach().clamp(0, 7)
            source5_mask = source5_ids == source5_refiner_source_id
            source5_valid = source5_mask.any(dim=-1)
            ref_tokens = select_tokens
            ref_quads = select_quads
            ref_logits = select_logits
            if self.proposal_handoff_source5_refiner_detach:
                ref_tokens = ref_tokens.detach()
                ref_quads = ref_quads.detach()
                ref_logits = ref_logits.detach()
            ref_geom = ref_quads.reshape(b, 16, ref_quads.shape[2], 8)
            ref_item = self.proposal_handoff_source5_refiner_item_mlp(
                torch.cat([ref_tokens, ref_geom, ref_logits.unsqueeze(-1)], dim=-1)
            )
            ref_score_delta = self.proposal_handoff_source5_refiner_score_head(ref_item).squeeze(-1)
            ref_scores = (
                ref_logits / float(self.proposal_handoff_source5_refiner_temperature)
                + ref_score_delta.to(device=ref_logits.device, dtype=ref_logits.dtype)
            )
            source5_keep = source5_mask
            ref_topk = min(
                int(self.proposal_handoff_source5_refiner_topk),
                int(ref_scores.shape[-1]),
            )
            if ref_topk > 0 and ref_topk < int(ref_scores.shape[-1]):
                source5_rank_scores = ref_scores.masked_fill(~source5_mask, -50.0)
                source5_top_idx = source5_rank_scores.topk(ref_topk, dim=-1).indices
                source5_keep = torch.zeros_like(source5_mask, dtype=torch.bool)
                source5_keep.scatter_(
                    -1,
                    source5_top_idx,
                    source5_mask.gather(-1, source5_top_idx),
                )
            masked_ref_scores = ref_scores.masked_fill(~source5_keep, -50.0)
            source5_weights = F.softmax(masked_ref_scores, dim=-1)
            source5_weights = source5_weights * source5_valid.to(
                device=source5_weights.device,
                dtype=source5_weights.dtype,
            ).unsqueeze(-1)
            handoff_source5_refiner_scores = masked_ref_scores
            handoff_source5_refiner_weights = source5_weights
            handoff_source5_refiner_valid = source5_valid
            handoff_source5_refiner_base_quad = (
                ref_quads * source5_weights[..., None, None]
            ).sum(dim=2).clamp(0.0, 1.0)
            source5_pooled = (ref_item * source5_weights[..., None]).sum(dim=2)
            source5_entropy = -(
                source5_weights.clamp_min(1e-6)
                * source5_weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(int(ref_scores.shape[-1]), 2))
            ref_logits_tanh = torch.tanh(ref_logits)
            source5_max_logit = ref_logits.masked_fill(~source5_mask, -50.0).max(dim=-1).values
            source5_stats = torch.stack(
                [
                    source5_weights.max(dim=-1).values,
                    source5_entropy.to(device=ref_logits.device, dtype=ref_logits.dtype),
                    (source5_weights * ref_logits_tanh).sum(dim=-1),
                    ref_logits_tanh.masked_fill(~source5_mask, -1.0).max(dim=-1).values,
                ],
                dim=-1,
            )
            source5_tail = torch.cat(
                [
                    source5_pooled,
                    handoff_source5_refiner_base_quad.reshape(b, 16, 8),
                    source5_stats,
                ],
                dim=-1,
            )
            source5_delta = torch.tanh(
                self.proposal_handoff_source5_refiner_delta_head(source5_tail)
            ).reshape(b, 16, 4, 2)
            _source5_dirs, source5_lens = quad_axes(
                handoff_source5_refiner_base_quad.reshape(-1, 4, 2)
            )
            source5_step = source5_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            handoff_source5_refiner_quad = (
                handoff_source5_refiner_base_quad
                + float(self.proposal_handoff_source5_refiner_scale)
                * source5_delta
                * source5_step
            ).clamp(0.0, 1.0)
            ref_logit_delta = self.proposal_handoff_source5_refiner_logit_head(source5_tail).squeeze(-1)
            handoff_source5_refiner_logit = (
                source5_max_logit
                + float(self.proposal_handoff_source5_refiner_init_logit_bias)
                + float(self.proposal_handoff_source5_refiner_logit_scale)
                * ref_logit_delta.to(device=ref_logits.device, dtype=ref_logits.dtype)
            )
            handoff_source5_refiner_logit = torch.where(
                source5_valid,
                handoff_source5_refiner_logit,
                torch.full_like(handoff_source5_refiner_logit, -50.0),
            )
            ref_xyxy = quad_to_xyxy(handoff_source5_refiner_quad, min_size=0.02)
            select_quads = torch.cat([select_quads, handoff_source5_refiner_quad[:, :, None]], dim=2)
            select_xyxy = torch.cat([select_xyxy, ref_xyxy[:, :, None]], dim=2)
            select_cxcywh = torch.cat(
                [select_cxcywh, xyxy_to_cxcywh_torch(ref_xyxy)[:, :, None]],
                dim=2,
            )
            select_logits = torch.cat([select_logits, handoff_source5_refiner_logit[:, :, None]], dim=-1)
            select_tokens = torch.cat([select_tokens, source5_pooled[:, :, None]], dim=2)
            select_source_ids = torch.cat(
                [
                    select_source_ids,
                    torch.full_like(
                        select_source_ids[:, :, :1],
                        int(self.proposal_handoff_source5_refiner_output_source_id),
                    ),
                ],
                dim=2,
            )
            if select_active_tip_logits is not None and select_active_tip_logits.shape[:2] == select_logits.shape[:2]:
                select_active_tip_logits = torch.cat(
                    [select_active_tip_logits, handoff_source5_refiner_logit[:, :, None]],
                    dim=-1,
                )
        if (
            self.proposal_handoff_topk_query_refiner
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            (
                handoff_topk_query_refiner_quads,
                handoff_topk_query_refiner_base_quads,
                handoff_topk_query_refiner_logits,
                handoff_topk_query_refiner_valid,
                handoff_topk_query_refiner_source_ids,
                handoff_topk_query_refiner_tokens,
            ) = self._proposal_handoff_topk_query_outputs(
                feat16_map,
                select_quads,
                select_tokens,
                select_logits,
                select_source_ids,
            )
            topk_query_xyxy = quad_to_xyxy(
                handoff_topk_query_refiner_quads.reshape(-1, 4, 2),
                min_size=0.02,
            ).reshape(b, 16, handoff_topk_query_refiner_quads.shape[2], 4)
            select_quads = torch.cat(
                [select_quads, handoff_topk_query_refiner_quads],
                dim=2,
            )
            select_xyxy = torch.cat([select_xyxy, topk_query_xyxy], dim=2)
            select_cxcywh = torch.cat(
                [select_cxcywh, xyxy_to_cxcywh_torch(topk_query_xyxy)],
                dim=2,
            )
            select_logits = torch.cat(
                [select_logits, handoff_topk_query_refiner_logits],
                dim=-1,
            )
            select_tokens = torch.cat(
                [select_tokens, handoff_topk_query_refiner_tokens],
                dim=2,
            )
            select_source_ids = torch.cat(
                [
                    select_source_ids,
                    torch.full_like(
                        handoff_topk_query_refiner_source_ids,
                        int(self.proposal_handoff_topk_query_output_source_id),
                    ),
                ],
                dim=2,
            )
            if select_active_tip_logits is not None and select_active_tip_logits.shape[:2] == select_logits.shape[:2]:
                select_active_tip_logits = torch.cat(
                    [select_active_tip_logits, handoff_topk_query_refiner_logits],
                    dim=-1,
                )
        handoff_oicr_logits_list = None
        if (
            self.proposal_handoff_oicr_refine_times > 0
            and select_tokens.shape[:3] == select_quads.shape[:3]
            and select_logits.shape == select_quads.shape[:3]
        ):
            # PCL/OICR uses several independent refinement streams, where
            # each stream is supervised from the previous proposal scores.
            # Here we apply the same idea to the final handoff bank so the
            # refinement logits align with the candidates actually selected.
            oicr_tokens = select_tokens
            oicr_quads = select_quads.reshape(b, 16, select_quads.shape[2], 8)
            if self.proposal_handoff_oicr_refine_detach:
                oicr_tokens = oicr_tokens.detach()
                oicr_quads = oicr_quads.detach()
            base_oicr_logits = select_logits
            prev_logits = select_logits
            oicr_logits_seq = []
            for head in self.proposal_handoff_oicr_refine_heads:
                prev_in = prev_logits.detach().unsqueeze(-1)
                oicr_in = torch.cat([oicr_tokens, oicr_quads, prev_in], dim=-1)
                delta = head(oicr_in).squeeze(-1)
                prev_logits = prev_logits + float(self.proposal_handoff_oicr_refine_scale) * delta.to(
                    device=prev_logits.device,
                    dtype=prev_logits.dtype,
                )
                oicr_logits_seq.append(prev_logits)
            handoff_oicr_logits_list = torch.stack(oicr_logits_seq, dim=0)
            if self.proposal_handoff_oicr_refine_use_for_selection:
                if self.proposal_handoff_oicr_refine_aggregate == "mean":
                    select_logits = handoff_oicr_logits_list.mean(dim=0)
                elif self.proposal_handoff_oicr_refine_aggregate == "mean_base":
                    select_logits = torch.cat(
                        [
                            base_oicr_logits.unsqueeze(0),
                            handoff_oicr_logits_list,
                        ],
                        dim=0,
                    ).mean(dim=0)
                else:
                    select_logits = prev_logits
        handoff_source_router_logits = None
        handoff_source_router_candidate_logits = None
        if self.proposal_handoff_source_router and select_source_ids is not None:
            if select_source_ids.shape == select_logits.shape:
                select_k = int(select_quads.shape[2])
                router_pts = self.stage1._candidate_aligned_grid_points(select_quads, grid_size=3)
                router_readout = self.stage1._sample_traj_readout_features(
                    feat16_map,
                    router_pts.reshape(b, 16, select_k * 9, 2),
                )
                router_readout = router_readout.reshape(b, 16, select_k, 9, -1).mean(dim=3)
                router_readout = self.proposal_handoff_source_router_readout_proj(router_readout)
                router_geom = select_quads.reshape(b, 16, select_k, 8)
                router_in = torch.cat(
                    [
                        router_readout,
                        router_geom,
                        select_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
                handoff_source_router_candidate_logits = (
                    self.proposal_handoff_source_router_candidate_head(router_in).squeeze(-1)
                )
                handoff_source_ids = select_source_ids.detach().clamp(0, 7)
                neg = torch.finfo(handoff_source_router_candidate_logits.dtype).min
                group_logits = []
                group_present = []
                for source_id in range(8):
                    source_mask = handoff_source_ids == int(source_id)
                    group_present.append(source_mask.any(dim=-1))
                    source_scores = handoff_source_router_candidate_logits.masked_fill(~source_mask, neg)
                    group_logits.append(source_scores.max(dim=-1).values)
                handoff_source_router_logits = torch.stack(group_logits, dim=-1)
                handoff_source_router_logits = (
                    handoff_source_router_logits
                    + self.proposal_handoff_source_router_context_head(query_base).to(
                        dtype=handoff_source_router_logits.dtype
                    )
                )
                group_present_t = torch.stack(group_present, dim=-1)
                handoff_source_router_logits = handoff_source_router_logits.masked_fill(
                    ~group_present_t,
                    neg,
                )
                handoff_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
                select_logits = select_logits + float(self.proposal_handoff_source_router_scale) * handoff_bonus.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
                if abs(float(self.proposal_handoff_source_router_candidate_select_scale)) > 0.0:
                    select_logits = select_logits + float(
                        self.proposal_handoff_source_router_candidate_select_scale
                    ) * handoff_source_router_candidate_logits.to(
                        device=select_logits.device,
                        dtype=select_logits.dtype,
                    )
        if self.proposal_handoff_source_calibration and select_source_ids is not None:
            if select_source_ids.shape == select_logits.shape:
                handoff_source_ids = select_source_ids.detach().clamp(
                    0,
                    self.proposal_handoff_source_logit_bias.num_embeddings - 1,
                )
                handoff_bias = self.proposal_handoff_source_logit_bias(handoff_source_ids).squeeze(-1)
                select_logits = select_logits + float(self.proposal_handoff_source_calibration_scale) * handoff_bias.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
        if (
            self.proposal_handoff_source5_topk > 0
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            source5_mask = handoff_source_ids == 5
            if bool(source5_mask.any().detach().cpu()):
                k5 = min(int(self.proposal_handoff_source5_topk), int(select_logits.shape[-1]))
                source5_scores = select_logits.masked_fill(~source5_mask, -50.0)
                source5_top_idx = source5_scores.topk(k5, dim=-1).indices
                source5_keep = torch.zeros_like(source5_mask, dtype=torch.bool)
                source5_keep.scatter_(
                    -1,
                    source5_top_idx,
                    source5_mask.gather(-1, source5_top_idx),
                )
                select_logits = torch.where(
                    source5_mask & ~source5_keep,
                    torch.full_like(select_logits, -50.0),
                    select_logits,
                )
        handoff_router_source_gate_mask = None
        if (
            self.proposal_handoff_router_normalized_select
            and (not self.training)
            and handoff_source_router_logits is not None
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            centered_logits = torch.full_like(select_logits, -50.0)
            for source_id in range(8):
                source_mask = handoff_source_ids == int(source_id)
                if not bool(source_mask.any().detach().cpu()):
                    continue
                source_count = source_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(
                    dtype=select_logits.dtype
                )
                source_vals = select_logits.masked_fill(~source_mask, 0.0)
                source_mean = source_vals.sum(dim=-1, keepdim=True) / source_count
                source_var = (
                    ((select_logits - source_mean).masked_fill(~source_mask, 0.0) ** 2).sum(
                        dim=-1,
                        keepdim=True,
                    )
                    / source_count
                )
                source_std = torch.sqrt(source_var.clamp_min(1e-4))
                centered_logits = torch.where(
                    source_mask,
                    (select_logits - source_mean) / source_std,
                    centered_logits,
                )
            router_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
            select_logits = (
                float(self.proposal_handoff_router_normalized_within_scale)
                * centered_logits
                + float(self.proposal_handoff_router_normalized_source_scale)
                * router_bonus.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        if (
            self.proposal_handoff_router_gated_source
            and (not self.training)
            and handoff_source_router_logits is not None
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            router_top_source = handoff_source_router_logits.detach().argmax(dim=-1)
            handoff_router_source_gate_mask = handoff_source_ids == router_top_source.unsqueeze(-1)
            select_logits = torch.where(
                handoff_router_source_gate_mask,
                select_logits,
                torch.full_like(select_logits, -50.0),
                )
        handoff_source_representative_mask = None
        if (
            self.proposal_handoff_source_representatives_only
            and (not self.training)
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            handoff_source_representative_mask = torch.zeros_like(select_logits, dtype=torch.bool)
            neg = torch.full_like(select_logits, -50.0)
            for source_id in range(8):
                source_mask = handoff_source_ids == int(source_id)
                source_present = source_mask.any(dim=-1)
                source_scores = select_logits.masked_fill(~source_mask, -50.0)
                source_best = source_scores.argmax(dim=-1, keepdim=True)
                handoff_source_representative_mask.scatter_(
                    -1,
                    source_best,
                    source_present.unsqueeze(-1),
                )
            select_logits = torch.where(handoff_source_representative_mask, select_logits, neg)
        quad, _weights = self._select_refined_proposals(
            select_quads,
            select_logits,
            detach_geometry=bool(self.proposal_handoff_detach_selection_geometry),
        )
        handoff_fusion_base_quad = None
        if (
            self.proposal_handoff_final_fusion
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            handoff_fusion_base_quad = quad
            handoff_geom = select_quads.reshape(b, 16, select_quads.shape[2], 8)
            handoff_quality = torch.tanh(select_logits)[..., None]
            handoff_weight = _weights[..., None]
            handoff_item = self.proposal_handoff_final_fusion_candidate_mlp(
                torch.cat([select_tokens, handoff_geom, handoff_weight, handoff_quality], dim=-1)
            )
            handoff_token = (handoff_item * handoff_weight).sum(dim=2)
            handoff_entropy = -(
                _weights.clamp_min(1e-6) * _weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(int(select_quads.shape[2]), 2))
            quality_tanh = torch.tanh(select_logits)
            handoff_stats = torch.stack(
                [
                    _weights.max(dim=-1).values,
                    handoff_entropy,
                    (_weights * quality_tanh).sum(dim=-1),
                    quality_tanh.max(dim=-1).values,
                ],
                dim=-1,
            )
            handoff_delta = torch.tanh(
                self.proposal_handoff_final_fusion_head(
                    torch.cat([handoff_token, quad.reshape(b, 16, 8), handoff_stats], dim=-1)
                )
            ).reshape(b, 16, 4, 2)
            _handoff_dirs, handoff_lens = quad_axes(quad.reshape(-1, 4, 2))
            handoff_step = handoff_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            quad = (
                quad + self.proposal_handoff_final_fusion_scale * handoff_delta * handoff_step
            ).clamp(0.0, 1.0)
        local_grid_corner_logits = None
        local_grid_points = None
        local_grid_base_quad = None
        if self.local_grid_refine and self.local_grid_final_refine:
            local_grid_base_quad = quad
            quad, local_grid_corner_logits, local_grid_points = self._local_grid_refine_quad(
                feat16_map,
                quad,
            )
        final_xyxy = quad_to_xyxy(quad, min_size=0.02)

        out = dict(stage1)
        out["boxes_xyxy"] = final_xyxy
        out["boxes_cxcywh"] = xyxy_to_cxcywh_torch(final_xyxy)
        out["coarse_xyxy"] = stage1["boxes_xyxy"]
        out["coarse_cxcywh"] = stage1["boxes_cxcywh"]
        out["decoder_layer_xyxy"] = torch.stack(
            [
                (
                    (cand * refined_weights[..., None]).sum(dim=2)
                    if self.refine_recollect_stage1
                    else (cand * _weights[..., None]).sum(dim=2)
                )
                for cand in layer_xyxy
            ],
            dim=0,
        )
        out["decoder_layer_cxcywh"] = torch.stack(
            [
                xyxy_to_cxcywh_torch(
                    (cand * refined_weights[..., None]).sum(dim=2)
                    if self.refine_recollect_stage1
                    else (cand * _weights[..., None]).sum(dim=2)
                )
                for cand in layer_xyxy
            ],
            dim=0,
        )
        out["quad_xy"] = quad
        out["proposal_quads"] = select_quads
        out["proposal_xyxy"] = select_xyxy
        out["proposal_cxcywh"] = select_cxcywh
        out["proposal_logits"] = select_logits
        if select_source_ids is not None:
            out["proposal_source_ids"] = select_source_ids
        if select_active_tip_logits is not None and select_active_tip_logits.shape == select_logits.shape:
            out["proposal_active_tip_logits"] = select_active_tip_logits
        else:
            out.pop("proposal_active_tip_logits", None)
        if handoff_source_router_logits is not None:
            out["proposal_handoff_source_router_logits"] = handoff_source_router_logits
        if handoff_source_router_candidate_logits is not None:
            out["proposal_handoff_source_router_candidate_logits"] = handoff_source_router_candidate_logits
        if handoff_source_representative_mask is not None:
            out["proposal_handoff_source_representative_mask"] = handoff_source_representative_mask
        if handoff_router_source_gate_mask is not None:
            out["proposal_handoff_router_source_gate_mask"] = handoff_router_source_gate_mask
        if handoff_fusion_base_quad is not None:
            out["proposal_handoff_fusion_base_quad"] = handoff_fusion_base_quad
        if proposal_local_corr_logits is not None:
            out["proposal_local_corr_logits"] = proposal_local_corr_logits
        if proposal_local_corr_stats is not None:
            out["proposal_local_corr_stats"] = proposal_local_corr_stats
        if handoff_source5_readout_logits is not None:
            out["proposal_handoff_source5_readout_logits"] = handoff_source5_readout_logits
        if (
            handoff_mask_logits is not None
            and handoff_mask_points is not None
            and handoff_mask_quality_logits is not None
            and handoff_mask_source_mask is not None
            and handoff_mask_base_quads is not None
        ):
            out["proposal_handoff_mask_logits"] = handoff_mask_logits
            out["proposal_handoff_mask_points"] = handoff_mask_points
            out["proposal_handoff_mask_quality_logits"] = handoff_mask_quality_logits
            out["proposal_handoff_mask_source_mask"] = handoff_mask_source_mask
            out["proposal_handoff_mask_base_quads"] = handoff_mask_base_quads
            if handoff_mask_refined_quads is not None:
                out["proposal_handoff_mask_refined_quads"] = handoff_mask_refined_quads
        if handoff_rank_adaptive_delta is not None:
            out["proposal_handoff_rank_adaptive_delta"] = handoff_rank_adaptive_delta
            out["proposal_handoff_rank_adaptive_rank"] = handoff_rank_adaptive_rank
        if handoff_sparse_set_delta is not None:
            out["proposal_handoff_sparse_set_delta"] = handoff_sparse_set_delta
            out["proposal_handoff_sparse_set_mask"] = handoff_sparse_set_mask
        if handoff_source5_refiner_quad is not None:
            out["proposal_handoff_source5_refiner_quad"] = handoff_source5_refiner_quad
            out["proposal_handoff_source5_refiner_base_quad"] = handoff_source5_refiner_base_quad
            out["proposal_handoff_source5_refiner_scores"] = handoff_source5_refiner_scores
            out["proposal_handoff_source5_refiner_weights"] = handoff_source5_refiner_weights
            out["proposal_handoff_source5_refiner_valid"] = handoff_source5_refiner_valid
            out["proposal_handoff_source5_refiner_logit"] = handoff_source5_refiner_logit
        if handoff_topk_query_refiner_quads is not None:
            out["proposal_handoff_topk_query_refiner_quads"] = (
                handoff_topk_query_refiner_quads
            )
            out["proposal_handoff_topk_query_refiner_base_quads"] = (
                handoff_topk_query_refiner_base_quads
            )
            out["proposal_handoff_topk_query_refiner_logits"] = (
                handoff_topk_query_refiner_logits
            )
            out["proposal_handoff_topk_query_refiner_valid"] = (
                handoff_topk_query_refiner_valid
            )
            out["proposal_handoff_topk_query_refiner_source_ids"] = (
                handoff_topk_query_refiner_source_ids
            )
        if handoff_oicr_logits_list is not None:
            out["proposal_oicr_logits_list"] = handoff_oicr_logits_list
            out["proposal_oicr_logits"] = handoff_oicr_logits_list[-1]
        out["proposal_selection_weights"] = _weights
        if local_grid_corner_logits is not None and local_grid_points is not None:
            out["local_grid_corner_logits"] = local_grid_corner_logits
            out["local_grid_points"] = local_grid_points
            if local_grid_base_quad is not None:
                out["local_grid_base_quad"] = local_grid_base_quad
        if candidate_local_grid_logits is not None and candidate_local_grid_points is not None:
            out["candidate_local_grid_corner_logits"] = candidate_local_grid_logits
            out["candidate_local_grid_points"] = candidate_local_grid_points
            if candidate_local_grid_base_quads is not None:
                out["candidate_local_grid_base_quads"] = candidate_local_grid_base_quads
        if candidate_local_grid_confidence is not None:
            out["candidate_local_grid_confidence_logits"] = candidate_local_grid_confidence
        if (
            candidate_mask_logits is not None
            and candidate_mask_points is not None
            and candidate_mask_quality_logits is not None
        ):
            out["candidate_mask_logits"] = candidate_mask_logits
            out["candidate_mask_points"] = candidate_mask_points
            out["candidate_mask_quality_logits"] = candidate_mask_quality_logits
            if candidate_mask_base_quads is not None:
                out["candidate_mask_base_quads"] = candidate_mask_base_quads
        out["stage1_proposal_quads"] = stage1["proposal_quads"]
        out["stage1_proposal_xyxy"] = stage1["proposal_xyxy"]
        out["stage1_full_proposal_quads"] = stage1_full_proposal_quads
        out["stage1_full_proposal_xyxy"] = stage1_full_proposal_xyxy
        out["stage1_full_proposal_logits"] = stage1_full_proposal_logits
        if topk_indices is not None:
            out["stage1_refine_topk_indices"] = topk_indices
        out.update(proposal_oracle_refiner_payload)
        return out

class H259TrajectoryPriorTemporalRegion(nn.Module):
    """Trajectory/action-range prior without V-JEPA visual tokens.

    This model is a condition-use proof model.  Tool trajectory is not appended
    as a weak side feature; it provides the geometric reference center/axis.
    Action-range role and tool/action IDs gate that trajectory representation.
    A useful checkpoint must degrade under condition ablations.
    """

    def __init__(self, hidden_dim: int, residual_scale: float = 0.35) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.tool_mlp = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.role_mlp = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.traj_global_mlp = nn.Sequential(
            nn.LayerNorm(16 * 13),
            nn.Linear(16 * 13, c),
            nn.SiLU(),
            nn.Linear(c, c),
        )
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.role_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.Sigmoid())
        self.id_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.LayerNorm(4 * c + 2), nn.Linear(4 * c + 2, c), nn.SiLU())
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        self.center_delta = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.size_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.axis_delta = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.corner_offset = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 8))
        self.time_pos = nn.Parameter(torch.zeros(1, 16, c))
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.zeros_(self.center_delta[-1].weight)
        nn.init.zeros_(self.center_delta[-1].bias)
        nn.init.zeros_(self.size_head[-1].weight)
        nn.init.constant_(self.size_head[-1].bias, -2.1)
        nn.init.zeros_(self.axis_delta[-1].weight)
        nn.init.zeros_(self.axis_delta[-1].bias)
        nn.init.zeros_(self.corner_offset[-1].weight)
        nn.init.zeros_(self.corner_offset[-1].bias)

    @staticmethod
    def _tool_anchor_and_axis(tool_kpt_vec_16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tool_kpt_vec_16.shape[0]
        xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).clamp(0.0, 1.0)
        vis = tool_kpt_vec_16[..., 6:9].clamp(0.0, 1.0)
        mass = vis.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        anchor = (xy * vis.unsqueeze(-1)).sum(dim=-2) / mass
        empty = (vis.sum(dim=-1, keepdim=True) <= 0.0).to(xy.dtype)
        anchor = anchor * (1.0 - empty) + 0.5 * empty

        shaft = xy[:, :, 0] - xy[:, :, 2]
        shaft_norm = shaft.norm(dim=-1, keepdim=True)
        default_axis = torch.zeros_like(shaft)
        default_axis[..., 0] = 1.0
        axis = torch.where(shaft_norm > 1e-4, shaft / shaft_norm.clamp_min(1e-6), default_axis)
        return anchor, axis

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tokens, tool_mask_8, tool_kpt_heatmaps_8
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        b = tool_kpt_vec_16.shape[0]
        role = role_features_16.to(device=device, dtype=dtype)
        tool_vec = tool_kpt_vec_16.to(device=device, dtype=dtype)
        anchor, tool_axis = self._tool_anchor_and_axis(tool_vec)

        tool_tok = self.tool_mlp(tool_vec)
        role_tok = self.role_mlp(role)
        cond_global_in = torch.cat([tool_vec, role], dim=-1).reshape(b, 16 * 13)
        global_tok = self.traj_global_mlp(cond_global_in)
        ids = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gate = 0.5 * (self.role_gate(role_tok) + self.id_gate(ids[:, None, :]))
        x = torch.cat(
            [
                tool_tok * (1.0 + gate),
                role_tok,
                global_tok[:, None, :].expand(-1, 16, -1),
                ids[:, None, :].expand(-1, 16, -1),
                anchor,
            ],
            dim=-1,
        )
        x = self.fuse(x) + self.time_pos.to(device=device, dtype=dtype)
        x = self.temporal(x)

        center = (anchor + float(self.residual_scale) * torch.tanh(self.center_delta(x))).clamp(0.0, 1.0)
        size = torch.sigmoid(self.size_head(x)).clamp(0.02, 0.95)
        axis = F.normalize(tool_axis + 0.75 * torch.tanh(self.axis_delta(x)), dim=-1, eps=1e-6)
        perp = torch.stack([-axis[..., 1], axis[..., 0]], dim=-1)
        base = self.base_corners.to(device=device, dtype=dtype).view(1, 1, 4, 2)
        local = base + 0.20 * torch.tanh(self.corner_offset(x).view(b, 16, 4, 2))
        local = local * size[:, :, None, :]
        quad = center[:, :, None, :] + local[..., :1] * axis[:, :, None, :] + local[..., 1:] * perp[:, :, None, :]
        quad = quad.clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        center_logits_8 = torch.zeros((b, 8, 24, 24), device=device, dtype=dtype)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "coarse_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "coarse_xyxy": box_xyxy,
            "center_logits_8": center_logits_8,
            "quad_xy": quad,
            "trajectory_prior_quad": quad,
            "trajectory_prior_xyxy": box_xyxy,
        }

def quad_axes(quad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two sign-agnostic quad axes and their lengths.

    The input is an ordered quad [top-left, top-right, bottom-right,
    bottom-left] in normalized image coordinates.
    """
    p0, p1, p2, p3 = quad.unbind(dim=-2)
    axis_a = 0.5 * ((p1 - p0) + (p2 - p3))
    axis_b = 0.5 * ((p3 - p0) + (p2 - p1))
    len_a = torch.linalg.norm(axis_a, dim=-1).clamp_min(1e-4)
    len_b = torch.linalg.norm(axis_b, dim=-1).clamp_min(1e-4)
    dirs = torch.stack([axis_a / len_a[..., None], axis_b / len_b[..., None]], dim=-2)
    lens = torch.stack([len_a, len_b], dim=-1)
    return dirs, lens

def quad_polygon_area(quad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shoelace polygon area for an ordered free quad in normalized coordinates."""
    x = quad[..., :, 0]
    y = quad[..., :, 1]
    area = 0.5 * torch.abs(
        (x * torch.roll(y, shifts=-1, dims=-1)).sum(dim=-1)
        - (y * torch.roll(x, shifts=-1, dims=-1)).sum(dim=-1)
    )
    return area.clamp_min(float(eps))

def quad_inside_grid(quad: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    q = quad.float().clamp(0.0, 1.0)
    g = max(int(grid_size), 8)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(*([1] * (q.ndim - 2)), 1, g, g, 2)
    a = q
    b = torch.roll(q, shifts=-1, dims=-2)
    edge = b - a
    rel = grid - a[..., None, None, :]
    cross = edge[..., 0, None, None] * rel[..., 1] - edge[..., 1, None, None] * rel[..., 0]
    return (cross >= -1e-6).all(dim=-3) | (cross <= 1e-6).all(dim=-3)

def quad_soft_inside_grid(
    quad: torch.Tensor,
    grid_size: int = 64,
    edge_sharpness: float = 128.0,
) -> torch.Tensor:
    """Differentiable soft polygon rasterizer for an ordered free quad."""
    q = quad.float().clamp(0.0, 1.0)
    g = max(int(grid_size), 8)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(*([1] * (q.ndim - 2)), 1, g, g, 2)
    a = q
    b = torch.roll(q, shifts=-1, dims=-2)
    edge = b - a
    rel = grid - a[..., None, None, :]
    cross = edge[..., 0, None, None] * rel[..., 1] - edge[..., 1, None, None] * rel[..., 0]
    nxt = torch.roll(edge, shifts=-1, dims=-2)
    turn = edge[..., 0] * nxt[..., 1] - edge[..., 1] * nxt[..., 0]
    orient = torch.sign(turn.mean(dim=-1, keepdim=True))
    orient = torch.where(orient == 0.0, torch.ones_like(orient), orient)
    edge_norm = torch.linalg.norm(edge, dim=-1).clamp_min(1e-4)
    signed_dist = cross * orient[..., None, None] / edge_norm[..., None, None]
    min_signed_dist = signed_dist.amin(dim=-3)
    return torch.sigmoid(float(edge_sharpness) * min_signed_dist)

def quad_iou_approx(pred_quad: torch.Tensor, gt_quad: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    pred_m = quad_inside_grid(pred_quad, grid_size=grid_size)
    gt_m = quad_inside_grid(gt_quad, grid_size=grid_size)
    inter = (pred_m & gt_m).sum(dim=(-1, -2)).float()
    union = (pred_m | gt_m).sum(dim=(-1, -2)).float()
    return inter / union.clamp_min(1.0)
