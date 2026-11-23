import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vjepa_wrapper import VJEPAWrapper
except ImportError:
    from vjepa_wrapper import VJEPAWrapper


TEMPLATE_CORR_REFINER_MODES: Tuple[str, ...] = (
    "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
    "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
)
STRICT_S3_CROSSLAYER_REFINER_MODES: Tuple[str, ...] = (
    "s3p2x_finalrefine_l18",
)
SHALLOW_VERIFIER_REFINER_MODES: Tuple[str, ...] = (
    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
)
MULTILAYER_DYNAMIC_CENTROID_REFINER_MODES: Tuple[str, ...] = (
    "m3_layerwise_statecarry_dynamiccentroid",
    "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
    *TEMPLATE_CORR_REFINER_MODES,
)
SINGLELAYER_DYNAMIC_CENTROID_REFINER_MODES: Tuple[str, ...] = (
    "s3p4_singlelayer_statecarry_dynamiccentroid",
)
DYNAMIC_CENTROID_REFINER_MODES: Tuple[str, ...] = (
    *MULTILAYER_DYNAMIC_CENTROID_REFINER_MODES,
    *SINGLELAYER_DYNAMIC_CENTROID_REFINER_MODES,
)
LAYERWISE_STATECARRY_REFINER_MODES: Tuple[str, ...] = (
    "m2_layerwise_statecarry_refine",
) + MULTILAYER_DYNAMIC_CENTROID_REFINER_MODES
SINGLELAYER_STATECARRY_REFINER_MODES: Tuple[str, ...] = (
    "s3p3_singlelayer_statecarry_refine",
    "s3p3x_singlelayer_statecarry_refine_plus1",
    "s3p8_singlelayer_statecarry_fourierbelief",
    "s3p8a_singlelayer_statecarry_residualfourierbelief",
    "s3p9_singlelayer_statecarry_anchoronly",
    "s3p9d_singlelayer_statecarry_anchoronly_detached",
    "s3p9e_singlelayer_statecarry_anchoronly_ramped",
    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
    "s3p10_singlelayer_statecarry_factorizedtoolresidual",
    "s3p5_singlelayer_statecarry_deformablesupport",
    "s3p6_singlelayer_statecarry_lategated_deformablesupport",
    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
) + SINGLELAYER_DYNAMIC_CENTROID_REFINER_MODES
STATECARRY_REFINER_MODES: Tuple[str, ...] = LAYERWISE_STATECARRY_REFINER_MODES + SINGLELAYER_STATECARRY_REFINER_MODES
MULTILAYER_TRAJ_REFINER_MODES: Tuple[str, ...] = (
    "m1_progressive_multilayer_movingrefine",
    "m1_strict_s3_multilayer_mean",
    *STRICT_S3_CROSSLAYER_REFINER_MODES,
    *SHALLOW_VERIFIER_REFINER_MODES,
) + LAYERWISE_STATECARRY_REFINER_MODES

S3P9E_ANCHOR_DELAY_EPOCHS = 10
S3P9E_ANCHOR_RAMP_EPOCHS = 10


def build_grid_xy(h_patches: int, w_patches: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.arange(h_patches, device=device, dtype=dtype)
    cols = torch.arange(w_patches, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    # normalized patch-center coordinates in [0,1]
    grid = torch.stack(((xx + 0.5) / w_patches, (yy + 0.5) / h_patches), dim=-1)
    return grid.reshape(-1, 2)  # [S,2]


def build_time_emb_sincos(t_train: int, dim: int) -> torch.Tensor:
    if int(t_train) <= 0:
        raise ValueError(f"`t_train` must be > 0, got {t_train}")
    if int(dim) <= 0 or (int(dim) % 2) != 0:
        raise ValueError(f"`dim` must be positive even integer, got {dim}")
    pos = torch.arange(int(t_train), dtype=torch.float32).view(-1, 1)
    omega = torch.arange(int(dim // 2), dtype=torch.float32) / float(dim // 2)
    omega = 1.0 / (10000.0 ** omega)
    out = pos * omega.view(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb.unsqueeze(0)


def interpolate_time_emb(time_emb: torch.Tensor, t_now: int) -> torch.Tensor:
    if time_emb.ndim != 3:
        raise ValueError(f"`time_emb` must be [1,T,D], got {tuple(time_emb.shape)}")
    if int(t_now) <= 0:
        raise ValueError(f"`t_now` must be > 0, got {t_now}")
    if int(time_emb.shape[1]) == int(t_now):
        return time_emb
    x = time_emb.float().permute(0, 2, 1)
    x = F.interpolate(x, size=int(t_now), mode="linear", align_corners=False)
    return x.permute(0, 2, 1).to(dtype=time_emb.dtype)


def cotracker_posenc(x: torch.Tensor, min_deg: int, max_deg: int) -> torch.Tensor:
    """Copied from CoTracker3 positional encoding for motion-state inputs."""
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(int(min_deg), int(max_deg))],
        dtype=x.dtype,
        device=x.device,
    )
    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x, four_feat], dim=-1)


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


class KVTokenMaskAugmenter(nn.Module):
    """Train-time KV token corruption for trajectory-only heads.

    Input/Output: [B,T,S,D] (mask only KV tokens; query path is untouched).
    """

    def __init__(
        self,
        in_dim: int,
        *,
        enable: bool = False,
        eval_enable: bool = False,
        strategy: str = "random",
        value_mode: str = "learned_pos",
        p_max: float = 0.10,
        p_early: Optional[float] = None,
        delay_epochs: int = 5,
        warmup_epochs: int = 20,
        late_start_epoch: int = 25,
        p_late: float = 0.15,
        late_warmup_epochs: int = 40,
        keep_k: int = 64,
        mix_801010: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.enable = bool(enable)
        self.eval_enable = bool(eval_enable)
        self.strategy = str(strategy).lower()
        self.value_mode = str(value_mode).lower()
        self.p_max = float(p_max)  # legacy alias for early-ramp target
        self.p_early = float(self.p_max if p_early is None else p_early)
        self.delay_epochs = int(delay_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.late_start_epoch = int(late_start_epoch)
        self.p_late = float(p_late)
        self.late_warmup_epochs = int(late_warmup_epochs)
        self.keep_k = int(keep_k)
        self.mix_801010 = bool(mix_801010)
        self.runtime_epoch = 0

        if self.in_dim <= 0:
            raise ValueError(f"`in_dim` must be > 0, got {self.in_dim}")
        if self.strategy not in ("random", "tube"):
            raise ValueError(f"`strategy` must be one of {{random,tube}}, got {self.strategy}")
        if self.value_mode not in ("learned_pos", "zero"):
            raise ValueError(f"`value_mode` must be one of {{learned_pos,zero}}, got {self.value_mode}")
        if self.p_max < 0.0 or self.p_max >= 1.0:
            raise ValueError(f"`p_max` must be in [0,1), got {self.p_max}")
        if self.p_early < 0.0 or self.p_early >= 1.0:
            raise ValueError(f"`p_early` must be in [0,1), got {self.p_early}")
        if self.delay_epochs < 0:
            raise ValueError(f"`delay_epochs` must be >= 0, got {self.delay_epochs}")
        if self.warmup_epochs < 0:
            raise ValueError(f"`warmup_epochs` must be >= 0, got {self.warmup_epochs}")
        if self.late_start_epoch < 0:
            raise ValueError(f"`late_start_epoch` must be >= 0, got {self.late_start_epoch}")
        if self.p_late < 0.0 or self.p_late >= 1.0:
            raise ValueError(f"`p_late` must be in [0,1), got {self.p_late}")
        if self.late_warmup_epochs < 0:
            raise ValueError(f"`late_warmup_epochs` must be >= 0, got {self.late_warmup_epochs}")
        if self.keep_k <= 0:
            raise ValueError(f"`keep_k` must be > 0, got {self.keep_k}")

        if self.value_mode == "learned_pos":
            hidden = int(min(max(64, self.in_dim // 2), 512))
            self.mask_base = nn.Parameter(torch.zeros(self.in_dim))
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.in_dim),
            )
            # Start close to a constant mask token; let position-conditioning grow during training.
            nn.init.zeros_(self.pos_mlp[-1].weight)
            nn.init.zeros_(self.pos_mlp[-1].bias)
        else:
            self.register_parameter("mask_base", None)
            self.pos_mlp = None

    def set_runtime_epoch(self, epoch: int) -> None:
        self.runtime_epoch = int(epoch)

    def _early_prob(self, ep: int) -> float:
        ep = int(ep)
        if ep < self.delay_epochs:
            return 0.0
        if self.warmup_epochs <= 0:
            return float(self.p_early)
        frac = float(ep - self.delay_epochs) / float(max(self.warmup_epochs, 1))
        frac = max(0.0, min(1.0, frac))
        return float(self.p_early * frac)

    def current_prob(self) -> float:
        ep = int(self.runtime_epoch)
        early_now = float(self._early_prob(ep))
        if ep < self.late_start_epoch:
            return early_now
        late_start_prob = float(self._early_prob(self.late_start_epoch))
        if self.late_warmup_epochs <= 0:
            return float(self.p_late)
        frac = float(ep - self.late_start_epoch) / float(max(self.late_warmup_epochs, 1))
        frac = max(0.0, min(1.0, frac))
        return float(late_start_prob + (self.p_late - late_start_prob) * frac)

    @staticmethod
    def _sample_keep_mask_random(
        bsz: int,
        t_steps: int,
        s_tokens: int,
        keep_n: int,
        device: torch.device,
    ) -> torch.Tensor:
        scores = torch.rand((bsz, t_steps, s_tokens), device=device)
        keep_idx = scores.topk(keep_n, dim=-1, largest=True).indices  # [B,T,keep_n]
        keep_mask = torch.zeros((bsz, t_steps, s_tokens), dtype=torch.bool, device=device)
        keep_mask.scatter_(-1, keep_idx, True)
        return keep_mask

    @staticmethod
    def _sample_keep_mask_tube(
        bsz: int,
        t_steps: int,
        s_tokens: int,
        keep_n: int,
        spatial_hw: Optional[Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        mask_n = int(max(s_tokens - keep_n, 0))
        if mask_n <= 0:
            return torch.ones((bsz, t_steps, s_tokens), dtype=torch.bool, device=device)

        if (spatial_hw is None) or (int(spatial_hw[0]) * int(spatial_hw[1]) != int(s_tokens)):
            # Fallback: tube-random (same mask across time).
            scores = torch.rand((bsz, s_tokens), device=device)
            keep_idx = scores.topk(keep_n, dim=-1, largest=True).indices
            keep_bt = torch.zeros((bsz, s_tokens), dtype=torch.bool, device=device)
            keep_bt.scatter_(-1, keep_idx, True)
            return keep_bt.unsqueeze(1).expand(-1, t_steps, -1).contiguous()

        h, w = int(spatial_hw[0]), int(spatial_hw[1])
        keep_bt = torch.ones((bsz, s_tokens), dtype=torch.bool, device=device)
        for bi in range(bsz):
            mask_flat = torch.zeros((s_tokens,), dtype=torch.bool, device=device)
            block_h = int(round(float(mask_n) ** 0.5))
            block_h = max(1, min(h, block_h))
            block_w = int(round(float(mask_n) / float(max(block_h, 1))))
            block_w = max(1, min(w, block_w))
            r0_max = max(h - block_h, 0)
            c0_max = max(w - block_w, 0)
            r0 = int(torch.randint(0, r0_max + 1, (1,), device=device).item())
            c0 = int(torch.randint(0, c0_max + 1, (1,), device=device).item())

            grid = torch.zeros((h, w), dtype=torch.bool, device=device)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = True
            mask_flat = grid.reshape(-1)

            cur = int(mask_flat.sum().item())
            if cur > mask_n:
                idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
                perm = torch.randperm(cur, device=device)[:mask_n]
                new_mask = torch.zeros_like(mask_flat)
                new_mask[idx[perm]] = True
                mask_flat = new_mask
            elif cur < mask_n:
                idx = torch.nonzero(~mask_flat, as_tuple=False).squeeze(1)
                add = int(min(mask_n - cur, int(idx.numel())))
                if add > 0:
                    perm = torch.randperm(int(idx.numel()), device=device)[:add]
                    mask_flat[idx[perm]] = True

            keep_bt[bi] = ~mask_flat
        return keep_bt.unsqueeze(1).expand(-1, t_steps, -1).contiguous()

    def _build_mask_value(
        self,
        *,
        s_tokens: int,
        spatial_hw: Optional[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.value_mode == "zero":
            return torch.zeros((1, 1, s_tokens, self.in_dim), device=device, dtype=dtype)

        if (self.pos_mlp is None) or (self.mask_base is None):
            return torch.zeros((1, 1, s_tokens, self.in_dim), device=device, dtype=dtype)

        if (spatial_hw is None) or (int(spatial_hw[0]) * int(spatial_hw[1]) != int(s_tokens)):
            return self.mask_base.view(1, 1, 1, self.in_dim).to(device=device, dtype=dtype).expand(1, 1, s_tokens, -1)

        h, w = int(spatial_hw[0]), int(spatial_hw[1])
        grid_xy = build_grid_xy(h_patches=h, w_patches=w, device=device, dtype=self.mask_base.dtype)  # [S,2]
        pos_bias = self.pos_mlp(grid_xy)  # [S,D]
        v = self.mask_base.view(1, self.in_dim) + pos_bias
        return v.to(device=device, dtype=dtype).view(1, 1, s_tokens, self.in_dim)

    def build_mask_plan(
        self,
        x_btsd: torch.Tensor,
        *,
        spatial_hw: Optional[Tuple[int, int]],
        apply_mask: bool,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, object]]:
        if x_btsd.ndim != 4:
            raise ValueError(f"`x_btsd` must be [B,T,S,D], got {tuple(x_btsd.shape)}")
        bsz, t_steps, s_tokens, d = x_btsd.shape
        if int(d) != self.in_dim:
            raise ValueError(f"`x_btsd` dim mismatch: expected D={self.in_dim}, got D={int(d)}")

        p_cur = float(self.current_prob()) if bool(apply_mask) else 0.0
        keep_n = int(round((1.0 - p_cur) * float(s_tokens)))
        keep_n = max(int(self.keep_k), keep_n)
        keep_n = max(1, min(int(s_tokens), keep_n))
        stats: Dict[str, object] = {
            "enabled": bool(self.enable),
            "eval_enabled": bool(self.eval_enable),
            "active": bool(apply_mask),
            "applied": False,
            "epoch": int(self.runtime_epoch),
            "p": float(p_cur),
            "p_early": float(self.p_early),
            "p_late": float(self.p_late),
            "late_start_epoch": int(self.late_start_epoch),
            "late_warmup_epochs": int(self.late_warmup_epochs),
            "strategy": str(self.strategy),
            "value_mode": str(self.value_mode),
            "mix_801010": bool(self.mix_801010),
            "keep_k": int(self.keep_k),
            "keep_n": int(keep_n),
            "mask_ratio": 0.0,
            "masked_tokens": 0,
            "masktok_ratio": 0.0,
            "random_ratio": 0.0,
            "keep_ratio": 0.0,
        }

        if (not apply_mask) or (p_cur <= 0.0) or (int(s_tokens) <= 0) or (int(keep_n) >= int(s_tokens)):
            return None, stats

        if self.strategy == "tube":
            keep_mask = self._sample_keep_mask_tube(
                bsz=int(bsz),
                t_steps=int(t_steps),
                s_tokens=int(s_tokens),
                keep_n=int(keep_n),
                spatial_hw=spatial_hw,
                device=x_btsd.device,
            )
        else:
            keep_mask = self._sample_keep_mask_random(
                bsz=int(bsz),
                t_steps=int(t_steps),
                s_tokens=int(s_tokens),
                keep_n=int(keep_n),
                device=x_btsd.device,
            )
        mask_pos = ~keep_mask
        masked = int(mask_pos.sum().item())
        if masked <= 0:
            return None, stats

        rand_src: Optional[torch.Tensor] = None
        if self.mix_801010:
            u = torch.rand((bsz, t_steps, s_tokens), device=x_btsd.device)
            do_masktok = mask_pos & (u < 0.8)
            do_random = mask_pos & (u >= 0.8) & (u < 0.9)
            do_keep = mask_pos & (u >= 0.9)
        else:
            do_masktok = mask_pos
            do_random = torch.zeros_like(mask_pos)
            do_keep = torch.zeros_like(mask_pos)

        if bool(do_random.any().item()):
            rand_src = torch.randint(0, int(s_tokens), (bsz, t_steps, s_tokens), device=x_btsd.device)
            if int(s_tokens) > 1:
                ar = torch.arange(int(s_tokens), device=x_btsd.device).view(1, 1, int(s_tokens))
                rand_src = torch.where(rand_src == ar, (rand_src + 1) % int(s_tokens), rand_src)

        stats["applied"] = True
        stats["mask_ratio"] = float(masked) / float(max(int(bsz * t_steps * s_tokens), 1))
        stats["masked_tokens"] = int(masked)
        stats["masktok_ratio"] = float(int(do_masktok.sum().item())) / float(masked)
        stats["random_ratio"] = float(int(do_random.sum().item())) / float(masked)
        stats["keep_ratio"] = float(int(do_keep.sum().item())) / float(masked)
        plan = {
            "do_masktok": do_masktok,
            "do_random": do_random,
            "rand_src": torch.empty((0,), dtype=torch.long, device=x_btsd.device) if rand_src is None else rand_src,
            "s_tokens": torch.tensor(int(s_tokens), dtype=torch.long, device=x_btsd.device),
        }
        return plan, stats

    def apply_mask_plan(
        self,
        x_btsd: torch.Tensor,
        *,
        plan: Optional[Dict[str, torch.Tensor]],
        spatial_hw: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        if x_btsd.ndim != 4:
            raise ValueError(f"`x_btsd` must be [B,T,S,D], got {tuple(x_btsd.shape)}")
        bsz, t_steps, s_tokens, d = x_btsd.shape
        if int(d) != self.in_dim:
            raise ValueError(f"`x_btsd` dim mismatch: expected D={self.in_dim}, got D={int(d)}")
        if plan is None:
            return x_btsd
        s_plan = int(plan["s_tokens"].item())
        if s_plan != int(s_tokens):
            raise ValueError(f"Mask plan token count mismatch: plan={s_plan}, x={int(s_tokens)}")

        v_b = self._build_mask_value(
            s_tokens=int(s_tokens),
            spatial_hw=spatial_hw,
            device=x_btsd.device,
            dtype=x_btsd.dtype,
        )
        x_out = x_btsd
        do_masktok = plan["do_masktok"]
        do_random = plan["do_random"]
        rand_src = plan["rand_src"]
        if tuple(do_masktok.shape) != (bsz, t_steps, s_tokens):
            raise ValueError(
                f"`do_masktok` must be [B,T,S]={bsz,t_steps,s_tokens}, got {tuple(do_masktok.shape)}"
            )
        if tuple(do_random.shape) != (bsz, t_steps, s_tokens):
            raise ValueError(
                f"`do_random` must be [B,T,S]={bsz,t_steps,s_tokens}, got {tuple(do_random.shape)}"
            )
        if bool(do_masktok.any().item()):
            x_out = torch.where(do_masktok.unsqueeze(-1), v_b, x_out)
        if bool(do_random.any().item()):
            if tuple(rand_src.shape) != (bsz, t_steps, s_tokens):
                raise ValueError(
                    f"`rand_src` must be [B,T,S]={bsz,t_steps,s_tokens}, got {tuple(rand_src.shape)}"
                )
            src_vec = x_btsd.gather(2, rand_src.unsqueeze(-1).expand(-1, -1, -1, int(d)))
            x_out = torch.where(do_random.unsqueeze(-1), src_vec, x_out)
        return x_out

    def forward(
        self,
        x_btsd: torch.Tensor,
        *,
        spatial_hw: Optional[Tuple[int, int]],
        apply_mask: bool,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        plan, stats = self.build_mask_plan(
            x_btsd,
            spatial_hw=spatial_hw,
            apply_mask=apply_mask,
        )
        return self.apply_mask_plan(x_btsd, plan=plan, spatial_hw=spatial_hw), stats


def _pick_gn_groups(num_channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


class SharedSpatialTrunk(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, proj_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,S,D]
        return self.act(self.fc(self.norm(x)))


class SharedConvRefine(nn.Module):
    """Lightweight YOLO-style local refinement tower on a single feature scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _pick_gn_groups(channels)
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(g, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn1(self.dw1(x)))
        x = self.act(self.gn2(self.dw2(x)))
        x = self.act(self.gn3(self.pw(x)))
        return x


class TemporalMeanResidual(nn.Module):
    """Temporal mean residual refinement on shared feature maps.

    Input shape: [B, T, C, H, W]
    """

    def __init__(self, channels: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.channel_scale = nn.Parameter(torch.ones(int(channels)))
        self.channel_bias = nn.Parameter(torch.zeros(int(channels)))
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"`x` must be [B,T,C,H,W], got {tuple(x.shape)}")
        mean_t = x.mean(dim=1, keepdim=True)
        residual = mean_t - x
        scale = self.channel_scale.view(1, 1, -1, 1, 1)
        bias = self.channel_bias.view(1, 1, -1, 1, 1)
        update = scale * residual + bias
        return x + (self.alpha * self.residual_scale) * update


class SpatialHeatmapOffsetHead(nn.Module):
    """Predict heatmap logits + per-cell sub-grid offsets for each keypoint."""

    def __init__(self, in_dim: int, num_points: int, enable_delta_branch: bool = True) -> None:
        super().__init__()
        self.num_points = num_points
        self.enable_delta_branch = bool(enable_delta_branch)
        self.logits_proj = nn.Conv2d(in_dim, num_points, kernel_size=1, stride=1, padding=0)
        self.delta_proj = nn.Conv2d(in_dim, num_points * 2, kernel_size=1, stride=1, padding=0)
        if not self.enable_delta_branch:
            for p in self.delta_proj.parameters():
                p.requires_grad = False

    def forward(
        self,
        feat_btdhw: torch.Tensor,
        bsz: int,
        t_prime: int,
        h_patches: int,
        w_patches: int,
    ) -> Dict[str, torch.Tensor]:
        # feat_btdhw: [B*T,d,H',W']
        s = int(h_patches * w_patches)
        logits_map = self.logits_proj(feat_btdhw).reshape(bsz, t_prime, self.num_points, h_patches, w_patches)
        logits = logits_map.reshape(bsz, t_prime, self.num_points, s).contiguous()  # [B,T,P,S]

        if self.enable_delta_branch:
            delta_map = self.delta_proj(feat_btdhw).reshape(bsz, t_prime, self.num_points, 2, h_patches, w_patches)
            delta_map = delta_map.permute(0, 1, 2, 4, 5, 3).reshape(bsz, t_prime, self.num_points, s, 2).contiguous()
        else:
            delta_map = logits.new_zeros((bsz, t_prime, self.num_points, s, 2))

        return {
            "logits": logits,
            "delta_raw": delta_map,
        }


class SpatialProbHeatmapHead(nn.Module):
    """Heatmap-only head: configurable deconv upsample + logits."""

    def __init__(
        self,
        in_dim: int,
        num_points: int,
        hidden_dim: Optional[int] = None,
        num_up_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(self.in_dim if hidden_dim is None else hidden_dim)
        self.num_up_layers = int(num_up_layers)
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")
        if self.num_up_layers <= 0:
            raise ValueError(f"`num_up_layers` must be > 0, got {self.num_up_layers}")

        g = _pick_gn_groups(self.hidden_dim)
        up_layers = []
        c_in = self.in_dim
        for _ in range(self.num_up_layers):
            up_layers.extend(
                [
                    nn.ConvTranspose2d(c_in, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.GroupNorm(g, self.hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
            c_in = self.hidden_dim
        self.up = nn.Sequential(*up_layers)
        self.logits_proj = nn.Conv2d(c_in, self.num_points, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        feat_btdhw: torch.Tensor,
        bsz: int,
        t_prime: int,
    ) -> Dict[str, torch.Tensor]:
        # feat_btdhw: [B*T,C,H,W] -> logits: [B,T,P,Hh*Wh]
        feat_up = self.up(feat_btdhw)
        h_hm = int(feat_up.shape[-2])
        w_hm = int(feat_up.shape[-1])
        s_hm = int(h_hm * w_hm)
        logits_map = self.logits_proj(feat_up).reshape(bsz, t_prime, self.num_points, h_hm, w_hm)
        logits = logits_map.reshape(bsz, t_prime, self.num_points, s_hm).contiguous()
        return {
            "logits": logits,
            "h_heatmap": h_hm,
            "w_heatmap": w_hm,
        }


class ResidualRecurrentTrajectoryHead(nn.Module):
    """Residual recurrent tracker:
      - first sub-frame predicts absolute xy
      - remaining sub-frames predict delta xy
    """

    def __init__(
        self,
        in_dim: int,
        num_points: int,
        num_tool_classes: int = 4,
        num_action_classes: int = 4,
        num_combo_classes: int = 6,
        condition_mode: str = "tool",
        max_delta_norm: float = 0.06,
        num_subframes: int = 2,
        output_num_points: Optional[int] = None,
        predict_offsets_from_centroid: bool = False,
        offset_max_norm: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_points = int(num_points)
        self.output_num_points = int(self.num_points if output_num_points is None else output_num_points)
        self.num_tool_classes = int(num_tool_classes)
        self.num_action_classes = int(num_action_classes)
        self.num_combo_classes = int(num_combo_classes)
        self.condition_mode = str(condition_mode).lower()
        self.max_delta_norm = float(max_delta_norm)
        self.num_subframes = int(num_subframes)
        self.predict_offsets_from_centroid = bool(predict_offsets_from_centroid)
        self.offset_max_norm = float(offset_max_norm)
        if self.in_dim <= 0:
            raise ValueError(f"`in_dim` must be > 0, got {self.in_dim}")
        if self.num_points <= 0:
            raise ValueError(f"`num_points` must be > 0, got {self.num_points}")
        if self.output_num_points <= 0:
            raise ValueError(f"`output_num_points` must be > 0, got {self.output_num_points}")
        if self.num_tool_classes <= 0:
            raise ValueError(f"`num_tool_classes` must be > 0, got {self.num_tool_classes}")
        if self.num_action_classes <= 0:
            raise ValueError(f"`num_action_classes` must be > 0, got {self.num_action_classes}")
        if self.num_combo_classes <= 0:
            raise ValueError(f"`num_combo_classes` must be > 0, got {self.num_combo_classes}")
        if self.condition_mode not in ("tool", "tool_action", "combo"):
            raise ValueError(
                f"Unsupported trajectory condition mode={condition_mode}, "
                "expected one of: tool,tool_action,combo"
            )
        if self.max_delta_norm <= 0.0:
            raise ValueError(f"`max_delta_norm` must be > 0, got {self.max_delta_norm}")
        if self.num_subframes <= 0:
            raise ValueError(f"`num_subframes` must be > 0, got {self.num_subframes}")
        if self.predict_offsets_from_centroid and self.num_points != 1:
            raise ValueError(
                "predict_offsets_from_centroid=True requires num_points=1 "
                f"(got num_points={self.num_points})."
            )
        if self.predict_offsets_from_centroid and self.output_num_points <= 1:
            raise ValueError(
                "predict_offsets_from_centroid=True requires output_num_points>1 "
                f"(got output_num_points={self.output_num_points})."
            )
        if self.predict_offsets_from_centroid and self.offset_max_norm <= 0.0:
            raise ValueError(f"`offset_max_norm` must be > 0, got {self.offset_max_norm}")

        heads = _pick_attn_heads(self.in_dim)
        self.base_state = nn.Parameter(torch.zeros(1, self.in_dim))
        nn.init.trunc_normal_(self.base_state, std=0.02)
        self.tool_embed = nn.Embedding(self.num_tool_classes, self.in_dim)
        self.action_embed = nn.Embedding(self.num_action_classes, self.in_dim)
        self.combo_embed = nn.Embedding(self.num_combo_classes, self.in_dim)
        self.point_embed = nn.Embedding(self.num_points, self.in_dim)
        self.subframe_embed = nn.Embedding(self.num_subframes, self.in_dim)
        nn.init.trunc_normal_(self.tool_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.action_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.combo_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.point_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.subframe_embed.weight, std=0.02)

        self.q_norm = nn.LayerNorm(self.in_dim)
        self.kv_norm = nn.LayerNorm(self.in_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.in_dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.abs_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            nn.Linear(self.in_dim, 2),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            nn.Linear(self.in_dim, 2),
        )
        if self.predict_offsets_from_centroid:
            self.offset_head = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, self.in_dim),
                nn.GELU(),
                nn.Linear(self.in_dim, self.output_num_points * 2),
            )
        else:
            self.offset_head = None
        self.state_proj = nn.Linear(self.in_dim, self.in_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(4, self.in_dim),
            nn.GELU(),
            nn.Linear(self.in_dim, self.in_dim),
        )
        self.state_norm = nn.LayerNorm(self.in_dim)

    def _cross_attend(self, tokens_bsd: torch.Tensor, state_bd: torch.Tensor, subframe_idx: int) -> torch.Tensor:
        bsz = int(tokens_bsd.shape[0])
        point_q = self.point_embed.weight.unsqueeze(0).expand(bsz, -1, -1)  # [B,P,D]
        sub_q = self.subframe_embed.weight[int(subframe_idx)].view(1, 1, -1)
        q = state_bd.unsqueeze(1) + point_q + sub_q
        q = self.q_norm(q)
        kv = self.kv_norm(tokens_bsd)
        out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        return out  # [B,P,D]

    def _decode_output_points(
        self,
        point_xy: torch.Tensor,
        point_hidden: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not self.predict_offsets_from_centroid:
            return {
                "pred_xy": point_xy,
                "centroid_xy": None,
                "offsets_xy": None,
            }
        if self.offset_head is None:
            raise RuntimeError("offset_head is required when predict_offsets_from_centroid=True.")
        bsz = int(point_xy.shape[0])
        centroid_xy = point_xy[:, :1, :]  # [B,1,2]
        centroid_hidden = point_hidden[:, 0, :]  # [B,D]
        offsets_xy = torch.tanh(self.offset_head(centroid_hidden)).reshape(bsz, self.output_num_points, 2)
        offsets_xy = offsets_xy * self.offset_max_norm
        # Keep centroid semantics explicit: per-frame offsets are zero-mean across points.
        offsets_xy = offsets_xy - offsets_xy.mean(dim=1, keepdim=True)
        pred_xy = (centroid_xy + offsets_xy).clamp(0.0, 1.0)
        return {
            "pred_xy": pred_xy,
            "centroid_xy": centroid_xy,
            "offsets_xy": offsets_xy,
        }

    def forward(
        self,
        feat_btsd: torch.Tensor,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        # feat_btsd: [B,T_block,S,D]
        _ = spatial_hw
        if feat_btsd.ndim != 4:
            raise ValueError(f"`feat_btsd` must be [B,T,S,D], got {tuple(feat_btsd.shape)}")
        bsz, t_block, _s, d = feat_btsd.shape
        if int(d) != self.in_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.in_dim}, got {int(d)}")
        if int(t_block) <= 0:
            raise ValueError("`t_block` must be > 0")

        traj_cond = _build_trajectory_condition(
            condition_mode=self.condition_mode,
            ref_tensor=feat_btsd,
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
        state = self.base_state.expand(bsz, -1) + traj_cond

        pred_seq = []
        delta_seq = []
        gate_seq = []
        centroid_seq = []
        offset_seq = []
        prev_pred_xy = None

        # block-1 / subframe-0: absolute detection
        h_abs = self._cross_attend(feat_btsd[:, 0], state, subframe_idx=0)  # [B,P,D]
        p_prev2 = torch.sigmoid(self.abs_head(h_abs)).clamp(0.0, 1.0)  # frame 0
        dec_abs = self._decode_output_points(point_xy=p_prev2, point_hidden=h_abs)
        pred_abs = dec_abs["pred_xy"]
        assert pred_abs is not None
        pred_seq.append(pred_abs)
        delta_seq.append(torch.zeros_like(pred_abs))
        gate_seq.append(torch.ones((bsz, self.output_num_points), device=feat_btsd.device, dtype=feat_btsd.dtype))
        prev_pred_xy = pred_abs
        if dec_abs["centroid_xy"] is not None:
            centroid_seq.append(dec_abs["centroid_xy"])
        if dec_abs["offsets_xy"] is not None:
            offset_seq.append(dec_abs["offsets_xy"])

        # block-1 / subframe-1: delta tracking
        h_last = self._cross_attend(feat_btsd[:, 0], state, subframe_idx=1)  # [B,P,D]
        d_prev = torch.tanh(self.delta_head(h_last)) * self.max_delta_norm
        p_prev = (p_prev2 + d_prev).clamp(0.0, 1.0)  # frame 1
        dec_prev = self._decode_output_points(point_xy=p_prev, point_hidden=h_last)
        pred_prev = dec_prev["pred_xy"]
        assert pred_prev is not None
        pred_seq.append(pred_prev)
        if prev_pred_xy is None:
            delta_seq.append(torch.zeros_like(pred_prev))
        else:
            delta_seq.append(pred_prev - prev_pred_xy)
        gate_seq.append(torch.ones((bsz, self.output_num_points), device=feat_btsd.device, dtype=feat_btsd.dtype))
        prev_pred_xy = pred_prev
        if dec_prev["centroid_xy"] is not None:
            centroid_seq.append(dec_prev["centroid_xy"])
        if dec_prev["offsets_xy"] is not None:
            offset_seq.append(dec_prev["offsets_xy"])
        h_state_prev = h_last.mean(dim=1)  # [B,D]

        for bi in range(1, int(t_block)):
            vel_prev = p_prev - p_prev2  # [B,P,2]
            pos_feat = torch.cat([p_prev, vel_prev], dim=-1)  # [B,P,4]
            pos_emb = self.pos_embed(pos_feat).mean(dim=1)  # [B,D]
            state = self.state_norm(state + self.state_proj(h_state_prev) + pos_emb + traj_cond)

            h_even = self._cross_attend(feat_btsd[:, bi], state, subframe_idx=0)
            d_even = torch.tanh(self.delta_head(h_even)) * self.max_delta_norm
            p_even = (p_prev + d_even).clamp(0.0, 1.0)
            dec_even = self._decode_output_points(point_xy=p_even, point_hidden=h_even)
            pred_even = dec_even["pred_xy"]
            assert pred_even is not None
            pred_seq.append(pred_even)
            if prev_pred_xy is None:
                delta_seq.append(torch.zeros_like(pred_even))
            else:
                delta_seq.append(pred_even - prev_pred_xy)
            gate_seq.append(torch.ones((bsz, self.output_num_points), device=feat_btsd.device, dtype=feat_btsd.dtype))
            prev_pred_xy = pred_even
            if dec_even["centroid_xy"] is not None:
                centroid_seq.append(dec_even["centroid_xy"])
            if dec_even["offsets_xy"] is not None:
                offset_seq.append(dec_even["offsets_xy"])

            h_odd = self._cross_attend(feat_btsd[:, bi], state, subframe_idx=1)
            d_odd = torch.tanh(self.delta_head(h_odd)) * self.max_delta_norm
            p_odd = (p_even + d_odd).clamp(0.0, 1.0)
            dec_odd = self._decode_output_points(point_xy=p_odd, point_hidden=h_odd)
            pred_odd = dec_odd["pred_xy"]
            assert pred_odd is not None
            pred_seq.append(pred_odd)
            if prev_pred_xy is None:
                delta_seq.append(torch.zeros_like(pred_odd))
            else:
                delta_seq.append(pred_odd - prev_pred_xy)
            gate_seq.append(torch.ones((bsz, self.output_num_points), device=feat_btsd.device, dtype=feat_btsd.dtype))
            prev_pred_xy = pred_odd
            if dec_odd["centroid_xy"] is not None:
                centroid_seq.append(dec_odd["centroid_xy"])
            if dec_odd["offsets_xy"] is not None:
                offset_seq.append(dec_odd["offsets_xy"])

            p_prev2, p_prev = p_even, p_odd
            h_state_prev = h_odd.mean(dim=1)

        pred_xy = torch.stack(pred_seq, dim=1).contiguous()  # [B,2*T_block,P_out,2]
        delta_xy = torch.stack(delta_seq, dim=1).contiguous()  # [B,2*T_block,P_out,2]
        delta_gate = torch.stack(gate_seq, dim=1).contiguous()  # [B,2*T_block,P_out]
        conf = delta_gate
        out = {
            "pred_xy": pred_xy,
            "delta_xy": delta_xy,
            "delta_gate": delta_gate,
            "conf": conf,
        }
        if self.predict_offsets_from_centroid:
            out["pred_centroid_xy"] = torch.stack(centroid_seq, dim=1).contiguous()  # [B,2*T_block,1,2]
            out["pred_offsets_xy"] = torch.stack(offset_seq, dim=1).contiguous()  # [B,2*T_block,P_out,2]
        return out


class AbsoluteCentroidOffsetTrajectoryHead(nn.Module):
    """Per-frame absolute tracker:
      - centroid coarse location is decoded from centroid-query attention map
      - centroid fine delta is predicted from centroid hidden token via MLP
      - final centroid is coarse + delta (YOLO-style local residual)
      - each point predicts offset relative to centroid on each subframe
      - no temporal delta accumulation
    """

    def __init__(
        self,
        in_dim: int,
        num_points: int,
        num_tool_classes: int = 4,
        num_action_classes: int = 4,
        num_combo_classes: int = 6,
        condition_mode: str = "tool",
        num_subframes: int = 2,
        max_offset_norm: float = 0.20,
        local_softargmax_radius: int = 0,
        subcell_r: int = 0,
        perpoint_local_readout: bool = False,
        perpoint_local_radius: int = 2,
        perpoint_gate_init: float = -3.0,
        cond_film: bool = False,
        conf_head: bool = False,
        token_detector: bool = False,
        token_det_hidden: int = 512,
        two_anchor_tool_ids: Tuple[int, ...] = (0,),
        adaptive_gate: bool = False,
        temporal_query_mix: bool = False,
        temporal_query_max_frames: int = 64,
        temporal_query_layers: int = 1,
        temporal_query_spatial: bool = False,
        temporal_query_interleave: bool = False,
        num_virtual_queries: int = 0,
        locality_prior: bool = False,
        locality_prior_median: int = 5,
        region_prior: bool = False,
        pooled_context: bool = False,
        pooled_context_levels: Tuple[int, ...] = (2, 4),
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_points = int(num_points)
        # --- E005 (2026-09-12): supervised, per-point, conditioned coarse localisation ---
        #   perpoint_local_readout: each keypoint query decodes its OWN attention map (previously
        #     computed and discarded) with a local-window expectation around its peak; blended into
        #     the shared-centroid path through a learnable gate (init sigmoid(gate_init)).
        #   cond_film: tool/action condition modulates the KEY path of the cross-attention (FiLM,
        #     zero-init) so the condition changes WHICH tokens are attended (Conditional-DETR idea).
        #   conf_head: per-point confidence logit from the point hidden + attention peak stats.
        #   The attention maps are exposed as `pred_point_attn` [B,T,P,S] for a direct CE loss.
        self.perpoint_local_readout = bool(perpoint_local_readout)
        self.perpoint_local_radius = int(perpoint_local_radius)
        self.cond_film = bool(cond_film)
        self.use_conf_head = bool(conf_head)
        self.last_point_attn = None
        self.last_pp_xy = None
        self.last_conf_logit = None
        self.last_pp_gate = 0.0
        self.last_pp_tau = 1.0
        # --- E006: per-token detector replaces the query-attention maps as the source of the per-point path ---
        #   det_head(FiLM(LN(token))) -> logits over {background, anchor-role, tip-role} per token (SuperPoint-style).
        #   Decode: single-role point = argmax of its class map; paired role = top-2 peaks (NMS radius 1) ordered by the
        #   dataset's canonical rule (axis anchor->tip, perp=(-ay,ax), slot0 = lower perpendicular score); then the same
        #   local-window expectation + gate as E005. Topology per sample from tool_id (two_anchor_tool_ids = 2A1T).
        # --- E006b: per-point adaptive gate. The scalar blend is an ensemble of two partly independent paths
        #   (E005b val: E002 path 19.8, per-point 19.2, blend 15.3, oracle per-point selection 13.5 px at stage-0);
        #   a small MLP on [peak prob, entropy, path disagreement, tool] predicts a per-point gate offset (zero-init).
        self.adaptive_gate = bool(adaptive_gate)
        # --- E007c: temporal query mixing. Pass 1 = per-frame cross-attention (as before); the per-frame query
        #   states are mixed over time by one pre-LN transformer layer whose output projections are zero-init
        #   (identity at init), and pass 2 re-attends with the mixed states added to the queries. Gives every
        #   frame's queries the identity/geometry context of the other frames (targets identity flips = tail).
        self.temporal_query_mix = bool(temporal_query_mix)
        self.temporal_query_max_frames = int(temporal_query_max_frames)
        self.temporal_query_layers = max(1, int(temporal_query_layers))  # E007g: stacked temporal layers
        # E007h (CoTracker-style alternation): after the temporal layer(s), one zero-init pre-LN transformer layer
        #   over the (1+P) queries of the SAME frame, so centroid/anchor/tips can enforce geometric consistency.
        self.temporal_query_spatial = bool(temporal_query_spatial)
        # E007i (survey H): interleave (temporal, spatial) x L blocks instead of "temporal stack + one spatial", and add
        #   V virtual/support queries (CoTracker support points) that attend to tokens and take part in the mixing but
        #   produce no outputs. All residuals zero-init -> identity at init.
        self.temporal_query_interleave = bool(temporal_query_interleave)
        self.num_virtual_queries = int(num_virtual_queries)
        # E007j (Cutie/SAM2-style distractor suppression): in pass 2, add a data-dependent locality bias
        #   -lambda_q * ||cell - ref_q||^2 to the query->token attention logits, where ref_q is the pass-1 local
        #   estimate of that point, median-filtered over frames, and lambda_q = MLP([peak prob, entropy, tool]) is
        #   zero-init (identity at init; sign free so the model can also learn anti-locality).
        self.locality_prior = bool(locality_prior)
        self.locality_prior_median = int(locality_prior_median)
        # --- C003: token region prior. A linear per-token classifier (BCE vs the GT region mask = hull of the visible
        #   points) whose logits, scaled by a learnable scalar, are added to every query's cross-attention logits.
        #   Zero-init classifier => identity at init. Motivated by the probe: a linear probe on a single token separates
        #   contact-region cells from other tissue with AUROC 0.93 while the head's centroid attention hit only 27 %.
        self.region_prior = bool(region_prior)
        self.last_region_logits = None
        self.last_centroid_attn = None
        if self.region_prior:
            self.rp_norm = nn.LayerNorm(self.in_dim)
            self.rp_head = nn.Linear(self.in_dim, 1)
            nn.init.zeros_(self.rp_head.weight)
            nn.init.zeros_(self.rp_head.bias)
            self.rp_scale = nn.Parameter(torch.tensor(1.0))
        # --- E007d: pooled multi-scale context. The 24x24 token grid is avg-pooled by the given factors (2 -> 12x12,
        #   4 -> 6x6), each level gets a learned level embedding, and a second cross-attention (zero-init out_proj)
        #   from the stage-0 queries to the pooled tokens produces a context vector that is added to the queries
        #   before the fine attention. Identity at init. Targets instrument-level selection (the wrong-instrument tail).
        self.pooled_context = bool(pooled_context)
        self.pooled_context_levels = tuple(int(x) for x in pooled_context_levels)
        self.token_detector = bool(token_detector)
        self.two_anchor_tool_ids = tuple(int(x) for x in two_anchor_tool_ids)
        self.last_det_logits = None
        self.last_det_xy = None
        # E004 F-1: restrict the coarse expectation to a disk around the attention peak.
        #   Plain expectation over a near-flat distribution is dragged to the grid centre
        #   (Bias-Compensated Integral Regression, arXiv 2301.10431). Recipe follows
        #   TAPIR/LocoTrack soft_argmax_heatmap (locotrack/utils/model_utils.py:99).
        #   0 disables => exact E002 behaviour.
        self.local_softargmax_radius = int(local_softargmax_radius)
        # E004 diagnostics, read directly by train.py (avoids the output-dict plumbing).
        self.last_subcell_kl = 0.0
        self.last_subcell_entropy = 0.0
        # E004 F-2: each token decodes its OWN 16x16 patch into an r x r sub-cell
        #   distribution from its 1024-d channel vector (SuperPoint depth-to-space,
        #   arXiv 1712.07629). Position vocabulary goes 576 -> 576*r^2.
        #   0 disables => exact E002 behaviour.
        self.subcell_r = int(subcell_r)
        self.num_tool_classes = int(num_tool_classes)
        self.num_action_classes = int(num_action_classes)
        self.num_combo_classes = int(num_combo_classes)
        self.condition_mode = str(condition_mode).lower()
        self.num_subframes = int(num_subframes)
        self.max_offset_norm = float(max_offset_norm)
        if self.in_dim <= 0:
            raise ValueError(f"`in_dim` must be > 0, got {self.in_dim}")
        if self.num_points <= 0:
            raise ValueError(f"`num_points` must be > 0, got {self.num_points}")
        if self.num_tool_classes <= 0:
            raise ValueError(f"`num_tool_classes` must be > 0, got {self.num_tool_classes}")
        if self.num_action_classes <= 0:
            raise ValueError(f"`num_action_classes` must be > 0, got {self.num_action_classes}")
        if self.num_combo_classes <= 0:
            raise ValueError(f"`num_combo_classes` must be > 0, got {self.num_combo_classes}")
        if self.condition_mode not in ("tool", "tool_action", "combo"):
            raise ValueError(
                f"Unsupported trajectory condition mode={condition_mode}, "
                "expected one of: tool,tool_action,combo"
            )
        if self.num_subframes <= 0:
            raise ValueError(f"`num_subframes` must be > 0, got {self.num_subframes}")
        if self.max_offset_norm <= 0.0:
            raise ValueError(f"`max_offset_norm` must be > 0, got {self.max_offset_norm}")

        heads = _pick_attn_heads(self.in_dim)
        self.base_state = nn.Parameter(torch.zeros(1, self.in_dim))
        self.centroid_token = nn.Parameter(torch.zeros(1, self.in_dim))
        nn.init.trunc_normal_(self.base_state, std=0.02)
        nn.init.trunc_normal_(self.centroid_token, std=0.02)
        self.tool_embed = nn.Embedding(self.num_tool_classes, self.in_dim)
        self.action_embed = nn.Embedding(self.num_action_classes, self.in_dim)
        self.combo_embed = nn.Embedding(self.num_combo_classes, self.in_dim)
        self.point_embed = nn.Embedding(self.num_points, self.in_dim)
        self.subframe_embed = nn.Embedding(self.num_subframes, self.in_dim)
        nn.init.trunc_normal_(self.tool_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.action_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.combo_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.point_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.subframe_embed.weight, std=0.02)

        self.q_norm = nn.LayerNorm(self.in_dim)
        self.kv_norm = nn.LayerNorm(self.in_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.in_dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )

        self.centroid_delta_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            nn.Linear(self.in_dim, 2),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            nn.Linear(self.in_dim, 2),
        )

        # --- E004 F-2: per-token sub-cell readout of the channel vector ---
        self.sub_head = None
        if self.subcell_r > 1:
            self.sub_norm = nn.LayerNorm(self.in_dim)
            self.sub_head = nn.Linear(self.in_dim, self.subcell_r * self.subcell_r)
            # Zero-init => uniform sub_prob => the expectation lands exactly on the patch
            # centre => bit-identical to E002 at step 0 (strict superset).
            # No structured bias is needed: the sub-cell coordinates differ per bin, so the
            # per-bin gradients differ and the symmetry breaks on its own.
            nn.init.zeros_(self.sub_head.weight)
            nn.init.zeros_(self.sub_head.bias)

        # --- E005 parameters ---
        if self.perpoint_local_readout:
            self.pp_gate_logit = nn.Parameter(torch.tensor(float(perpoint_gate_init)))
            self.pp_log_tau = nn.Parameter(torch.zeros(()))
        if self.cond_film:
            self.film = nn.Linear(self.in_dim, 2 * self.in_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if self.pooled_context:
            self.pc_level_embed = nn.Parameter(torch.zeros(len(self.pooled_context_levels), self.in_dim))
            nn.init.trunc_normal_(self.pc_level_embed, std=0.02)
            self.pc_norm = nn.LayerNorm(self.in_dim)
            self.pc_attn = nn.MultiheadAttention(embed_dim=self.in_dim, num_heads=_pick_attn_heads(self.in_dim), dropout=0.0, batch_first=True)
            nn.init.zeros_(self.pc_attn.out_proj.weight)
            nn.init.zeros_(self.pc_attn.out_proj.bias)
        if self.temporal_query_mix:
            self.tq_pos = nn.Parameter(torch.zeros(self.temporal_query_max_frames, self.in_dim))
            nn.init.trunc_normal_(self.tq_pos, std=0.02)
            self.tq_layer = nn.TransformerEncoderLayer(
                d_model=self.in_dim, nhead=_pick_attn_heads(self.in_dim), dim_feedforward=2 * self.in_dim,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            nn.init.zeros_(self.tq_layer.self_attn.out_proj.weight)
            nn.init.zeros_(self.tq_layer.self_attn.out_proj.bias)
            nn.init.zeros_(self.tq_layer.linear2.weight)
            nn.init.zeros_(self.tq_layer.linear2.bias)
            if self.locality_prior:
                self.lp_mlp = nn.Sequential(nn.Linear(2 + self.num_tool_classes, 32), nn.GELU(), nn.Linear(32, 1))
                nn.init.zeros_(self.lp_mlp[-1].weight)
                nn.init.zeros_(self.lp_mlp[-1].bias)
            if self.num_virtual_queries > 0:
                self.virtual_embed = nn.Parameter(torch.zeros(self.num_virtual_queries, self.in_dim))
                nn.init.trunc_normal_(self.virtual_embed, std=0.02)
            if self.temporal_query_spatial and self.temporal_query_interleave:
                self.tq_spatial_layers = nn.ModuleList()
                for _ in range(self.temporal_query_layers):
                    lyr = nn.TransformerEncoderLayer(
                        d_model=self.in_dim, nhead=_pick_attn_heads(self.in_dim), dim_feedforward=2 * self.in_dim,
                        dropout=0.0, batch_first=True, norm_first=True,
                    )
                    nn.init.zeros_(lyr.self_attn.out_proj.weight)
                    nn.init.zeros_(lyr.self_attn.out_proj.bias)
                    nn.init.zeros_(lyr.linear2.weight)
                    nn.init.zeros_(lyr.linear2.bias)
                    self.tq_spatial_layers.append(lyr)
            if self.temporal_query_spatial:
                self.tq_spatial_layer = nn.TransformerEncoderLayer(
                    d_model=self.in_dim, nhead=_pick_attn_heads(self.in_dim), dim_feedforward=2 * self.in_dim,
                    dropout=0.0, batch_first=True, norm_first=True,
                )
                nn.init.zeros_(self.tq_spatial_layer.self_attn.out_proj.weight)
                nn.init.zeros_(self.tq_spatial_layer.self_attn.out_proj.bias)
                nn.init.zeros_(self.tq_spatial_layer.linear2.weight)
                nn.init.zeros_(self.tq_spatial_layer.linear2.bias)
            self.tq_extra_layers = nn.ModuleList()
            for _ in range(self.temporal_query_layers - 1):
                lyr = nn.TransformerEncoderLayer(
                    d_model=self.in_dim, nhead=_pick_attn_heads(self.in_dim), dim_feedforward=2 * self.in_dim,
                    dropout=0.0, batch_first=True, norm_first=True,
                )
                nn.init.zeros_(lyr.self_attn.out_proj.weight)
                nn.init.zeros_(lyr.self_attn.out_proj.bias)
                nn.init.zeros_(lyr.linear2.weight)
                nn.init.zeros_(lyr.linear2.bias)
                self.tq_extra_layers.append(lyr)
        if self.adaptive_gate:
            if not self.perpoint_local_readout:
                raise ValueError("`adaptive_gate` requires `perpoint_local_readout`")
            self.gate_mlp = nn.Sequential(nn.Linear(3 + self.num_tool_classes, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        if self.token_detector:
            if not self.perpoint_local_readout:
                raise ValueError("`token_detector` requires `perpoint_local_readout` (it supplies the gate and local decode)")
            if self.num_points != 3:
                raise ValueError(f"`token_detector` assumes 3 tool points (1 single-role + 1 paired role), got {self.num_points}")
            self.det_norm = nn.LayerNorm(self.in_dim)
            self.det_film = nn.Linear(self.in_dim, 2 * self.in_dim)
            nn.init.zeros_(self.det_film.weight)
            nn.init.zeros_(self.det_film.bias)
            self.det_head = nn.Sequential(
                nn.Linear(self.in_dim, int(token_det_hidden)),
                nn.GELU(),
                nn.Linear(int(token_det_hidden), 3),
            )
        if self.use_conf_head:
            self.conf_mlp = nn.Sequential(
                nn.LayerNorm(self.in_dim + 2),
                nn.Linear(self.in_dim + 2, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )

    def _perpoint_local_xy(self, point_probs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Local-window expectation around each point's own attention peak. point_probs: [B,P,S] -> xy [B,P,2]."""
        bsz, npt, s = point_probs.shape
        device, dtype = point_probs.device, point_probs.dtype
        peak = point_probs.argmax(dim=-1)  # [B,P]
        py = torch.div(peak, w, rounding_mode="floor").to(dtype)
        px = (peak % w).to(dtype)
        rows = torch.arange(h, device=device, dtype=dtype)
        cols = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        gx = xx.reshape(1, 1, -1)
        gy = yy.reshape(1, 1, -1)
        cheb = torch.maximum((gx - px.unsqueeze(-1)).abs(), (gy - py.unsqueeze(-1)).abs())  # [B,P,S]
        mask = cheb <= float(self.perpoint_local_radius)
        tau = torch.exp(self.pp_log_tau).clamp(0.05, 20.0)
        logits = point_probs.clamp_min(1e-9).log() / tau
        logits = logits.masked_fill(~mask, float("-inf"))
        pl = torch.softmax(logits, dim=-1)
        grid_xy = build_grid_xy(h_patches=h, w_patches=w, device=device, dtype=dtype)  # [S,2]
        return torch.einsum("bps,sd->bpd", pl, grid_xy)

    def _detector_points(
        self,
        det_prob: torch.Tensor,
        tool_id: Optional[torch.Tensor],
        h: int,
        w: int,
        ref_xy: Optional[torch.Tensor] = None,
        margin_px: float = 4.0,
        pixel_scale: float = 383.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """det_prob: [B,S,3] (bg, anchor, tip). Returns xy [B,P=3,2] in point-index order and the per-point maps [B,P,S].
        Pair ordering: the dataset's geometric rule (slot0 = lower perpendicular score) when its margin >= margin_px,
        otherwise (ambiguous frames, whose GT order was inherited temporally) the assignment closest to `ref_xy`
        (the shared-centroid path's prediction), if given."""
        bsz, s, _ = det_prob.shape
        device, dtype = det_prob.device, det_prob.dtype
        if tool_id is None:
            two_anchor = torch.zeros((bsz,), device=device, dtype=torch.bool)
        else:
            tid = tool_id.view(-1).long()
            two_anchor = torch.zeros_like(tid, dtype=torch.bool)
            for x in self.two_anchor_tool_ids:
                two_anchor |= tid == int(x)
        anchor_map = det_prob[:, :, 1]
        tip_map = det_prob[:, :, 2]
        # single-role map / paired-role map per sample
        single_map = torch.where(two_anchor.unsqueeze(-1), tip_map, anchor_map)
        pair_map = torch.where(two_anchor.unsqueeze(-1), anchor_map, tip_map)
        # top-2 peaks of the paired map: peak2 = argmax excluding peak1's own cell (adjacent-cell pairs allowed),
        # then each peak's local window is split by a Voronoi rule so the two expectations do not pull each other.
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        gx = xx.reshape(1, -1)
        gy = yy.reshape(1, -1)
        def _cheb(peak):
            return torch.maximum((gx - (peak % w).unsqueeze(-1)).abs(), (gy - torch.div(peak, w, rounding_mode="floor").unsqueeze(-1)).abs())
        peak1 = pair_map.argmax(dim=-1)  # [B]
        c1 = _cheb(peak1)
        # NMS radius 1: a point on a cell boundary lights two adjacent cells almost equally, so the second peak must
        # be searched outside peak1's 3x3 neighbourhood (pairs closer than one cell are not separable on this grid).
        peak2 = pair_map.masked_fill(c1 <= 1, 0.0).argmax(dim=-1)
        c2 = _cheb(peak2)
        radius = float(self.perpoint_local_radius)
        win_a = (c1 <= radius) & (c1 <= c2)
        win_b = (c2 <= radius) & (c2 < c1)
        cs = _cheb(single_map.argmax(dim=-1))
        win_s = cs <= radius
        tau = torch.exp(self.pp_log_tau).clamp(0.05, 20.0)
        grid_xy = build_grid_xy(h_patches=h, w_patches=w, device=device, dtype=dtype)  # [S,2]
        def _expect(m, win):
            lg = (m.clamp_min(1e-9).log() / tau).masked_fill(~win, float("-inf"))
            return torch.softmax(lg, dim=-1) @ grid_xy
        xy_single = _expect(single_map, win_s)
        xy_a = _expect(pair_map, win_a)
        xy_b = _expect(pair_map, win_b)
        maps = torch.stack([single_map, pair_map * win_a.to(dtype), pair_map * win_b.to(dtype)], dim=1)  # [B,3,S]
        mid = 0.5 * (xy_a + xy_b)
        scale = torch.tensor([float(w), float(h)], device=device, dtype=dtype)  # perpendicular score in cell units
        axis = torch.where(two_anchor.unsqueeze(-1), (xy_single - mid), (mid - xy_single)) * scale  # anchors -> tips
        perp = torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)
        sa = ((xy_a - mid) * scale * perp).sum(-1)
        sb = ((xy_b - mid) * scale * perp).sum(-1)
        a_first = (sa <= sb)  # slot0 = lower perpendicular score
        if ref_xy is not None:
            ambiguous = (sa - sb).abs() * (float(pixel_scale) / float(max(w, 1))) < float(margin_px)
            ta_ = two_anchor
            ref0 = torch.where(ta_.unsqueeze(-1), ref_xy[:, 0], ref_xy[:, 1])  # slot0 reference point
            ref1 = torch.where(ta_.unsqueeze(-1), ref_xy[:, 1], ref_xy[:, 2])  # slot1 reference point
            cost_keep = (xy_a - ref0).norm(dim=-1) + (xy_b - ref1).norm(dim=-1)
            cost_swap = (xy_b - ref0).norm(dim=-1) + (xy_a - ref1).norm(dim=-1)
            a_first = torch.where(ambiguous, cost_keep <= cost_swap, a_first)
        a_first = a_first.unsqueeze(-1)
        slot0 = torch.where(a_first, xy_a, xy_b)
        slot1 = torch.where(a_first, xy_b, xy_a)
        map0 = torch.where(a_first, maps[:, 1], maps[:, 2])
        map1 = torch.where(a_first, maps[:, 2], maps[:, 1])
        # point-index order: 1A2T -> [anchor, tip0, tip1]; 2A1T -> [anchor0, anchor1, tip]
        ta = two_anchor.unsqueeze(-1)
        p0 = torch.where(ta, slot0, xy_single)
        p1 = torch.where(ta, slot1, slot0)
        p2 = torch.where(ta, xy_single, slot1)
        m0 = torch.where(ta, map0, maps[:, 0])
        m1 = torch.where(ta, map1, map0)
        m2 = torch.where(ta, maps[:, 0], map1)
        return torch.stack([p0, p1, p2], dim=1), torch.stack([m0, m1, m2], dim=1)

    def _build_subcell_grid(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Normalized xy of every sub-cell centre. Returns [S, r*r, 2]."""
        r = int(self.subcell_r)
        rows = torch.arange(h, device=device, dtype=dtype)
        cols = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        cell_x = xx.reshape(-1, 1)  # [S,1] patch index
        cell_y = yy.reshape(-1, 1)
        sr = torch.arange(r, device=device, dtype=dtype)
        sy, sx = torch.meshgrid(sr, sr, indexing="ij")
        off_x = ((sx.reshape(1, -1) + 0.5) / float(r))  # [1,r*r] in [0,1) within the patch
        off_y = ((sy.reshape(1, -1) + 0.5) / float(r))
        gx = (cell_x + off_x) / float(w)  # [S,r*r]
        gy = (cell_y + off_y) / float(h)
        return torch.stack((gx, gy), dim=-1)  # [S,r*r,2]

    def _local_peak_mask(self, probs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Binary [B,S] mask keeping cells within `local_softargmax_radius` of the argmax."""
        radius = float(self.local_softargmax_radius)
        bsz = int(probs.shape[0])
        device, dtype = probs.device, probs.dtype
        peak = probs.argmax(dim=-1)  # [B]
        py = torch.div(peak, w, rounding_mode="floor").to(dtype)
        px = (peak % w).to(dtype)
        rows = torch.arange(h, device=device, dtype=dtype)
        cols = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        gx = xx.reshape(1, -1).expand(bsz, -1)
        gy = yy.reshape(1, -1).expand(bsz, -1)
        d2 = (gx - px.unsqueeze(1)) ** 2 + (gy - py.unsqueeze(1)) ** 2
        return (d2 < radius * radius).to(dtype)

    @staticmethod
    def _resolve_spatial_hw(num_tokens: int, spatial_hw: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        s = int(num_tokens)
        if spatial_hw is not None:
            h = int(spatial_hw[0])
            w = int(spatial_hw[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid `spatial_hw`: {spatial_hw}")
            if h * w != s:
                raise ValueError(
                    f"`spatial_hw` mismatch with token count: H*W={h*w}, S={s}, spatial_hw={spatial_hw}"
                )
            return h, w
        side = int(round(float(s) ** 0.5))
        if side * side != s:
            raise ValueError(
                f"Cannot infer square spatial size from S={s}; pass explicit `spatial_hw=(H,W)`."
            )
        return side, side

    def _centroid_from_attn_probs(
        self,
        centroid_probs: torch.Tensor,
        spatial_hw: Optional[Tuple[int, int]],
        tokens_bsd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # centroid_probs: [B,S], already softmaxed by attention.
        if centroid_probs.ndim != 2:
            raise ValueError(f"`centroid_probs` must be [B,S], got {tuple(centroid_probs.shape)}")
        bsz, s = centroid_probs.shape
        h, w = self._resolve_spatial_hw(num_tokens=int(s), spatial_hw=spatial_hw)
        probs = centroid_probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # --- E004 F-1: local soft-argmax (restrict the expectation to a disk around the peak) ---
        if self.local_softargmax_radius > 0:
            mask = self._local_peak_mask(probs, h=h, w=w)  # [B,S]
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        heatmap = probs.reshape(bsz, 1, h, w).contiguous()

        # --- E004 F-2: factorised coarse x fine expectation on the 24r x 24r sub-grid ---
        if self.sub_head is not None and tokens_bsd is not None:
            r = int(self.subcell_r)
            sub_logits = self.sub_head(self.sub_norm(tokens_bsd))          # [B,S,r*r]
            sub_prob = torch.softmax(sub_logits.float(), dim=-1).to(probs.dtype)
            fine_grid = self._build_subcell_grid(h=h, w=w, device=probs.device, dtype=probs.dtype)  # [S,r*r,2]
            joint = probs.unsqueeze(-1) * sub_prob                          # [B,S,r*r]
            centroid_xy = torch.einsum("bsk,skd->bd", joint, fine_grid).unsqueeze(1)  # [B,1,2]
            # diagnostics: how far the per-token sub-distribution is from uniform.
            # If this stays ~0 after training, the 1024-d vector carries no sub-patch position.
            with torch.no_grad():
                sp = sub_prob.float().clamp_min(1e-8)
                kl_unif = (sp * (sp.log() + math.log(float(r * r)))).sum(-1)   # [B,S]
                w_cell = probs.float()
                sub_kl = (kl_unif * w_cell).sum(-1).mean()
                sub_ent = (-(sp * sp.log()).sum(-1) * w_cell).sum(-1).mean()
            return centroid_xy, heatmap, sub_kl.detach(), sub_ent.detach()

        grid_xy = build_grid_xy(h_patches=h, w_patches=w, device=probs.device, dtype=probs.dtype)  # [S,2]
        centroid_xy = torch.einsum("bs,sd->bd", probs, grid_xy).unsqueeze(1)  # [B,1,2]
        zero = probs.new_zeros(())
        return centroid_xy, heatmap, zero, zero

    @staticmethod
    def _centroid_delta_scale(spatial_hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h, w = int(spatial_hw[0]), int(spatial_hw[1])
        # One-cell local residual range in normalized xy coordinates.
        return torch.tensor([1.0 / float(max(w, 1)), 1.0 / float(max(h, 1))], device=device, dtype=dtype).view(1, 1, 2)

    def _pooled_context_extra(self, tokens_bsd: torch.Tensor, tool_cond_bd: torch.Tensor, subframe_idx: int, h: int, w: int) -> torch.Tensor:
        """Context vector per query from avg-pooled token levels. Returns [B,1+P,D] (zero at init)."""
        bsz = int(tokens_bsd.shape[0])
        grid = tokens_bsd.view(bsz, h, w, self.in_dim).permute(0, 3, 1, 2)  # [B,D,H,W]
        pooled = []
        for li, f in enumerate(self.pooled_context_levels):
            pl = F.avg_pool2d(grid, kernel_size=int(f), stride=int(f)).flatten(2).transpose(1, 2)  # [B,S_l,D]
            pooled.append(pl + self.pc_level_embed[li].view(1, 1, -1))
        kv = self.pc_norm(torch.cat(pooled, dim=1))
        sub = self.subframe_embed.weight[int(subframe_idx)].view(1, 1, -1)
        base = self.base_state.expand(bsz, -1) + tool_cond_bd
        q_all = torch.cat([base.unsqueeze(1) + self.centroid_token.view(1, 1, -1) + sub, base.unsqueeze(1) + self.point_embed.weight.unsqueeze(0) + sub], dim=1)
        ctx, _ = self.pc_attn(query=self.q_norm(q_all), key=kv, value=kv, need_weights=False)
        return ctx  # [B,1+P,D]

    def _cross_attend_step(
        self,
        tokens_bsd: torch.Tensor,
        tool_cond_bd: torch.Tensor,
        subframe_idx: int,
        q_extra: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        region_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = int(tokens_bsd.shape[0])
        sub = self.subframe_embed.weight[int(subframe_idx)].view(1, 1, -1)
        base = self.base_state.expand(bsz, -1) + tool_cond_bd  # [B,D]
        q_centroid = base.unsqueeze(1) + self.centroid_token.view(1, 1, -1) + sub  # [B,1,D]
        q_points = base.unsqueeze(1) + self.point_embed.weight.unsqueeze(0) + sub  # [B,P,D]
        q_all = torch.cat([q_centroid, q_points], dim=1)  # [B,1+P,D]
        if self.num_virtual_queries > 0:
            q_virtual = base.unsqueeze(1) + self.virtual_embed.unsqueeze(0) + sub  # [B,V,D]
            q_all = torch.cat([q_all, q_virtual], dim=1)  # [B,1+P+V,D]
        if q_extra is not None:
            q_all = q_all + q_extra  # E007c: temporal context from the other frames' pass-1 query states
        q_all = self.q_norm(q_all)
        kv = self.kv_norm(tokens_bsd)
        key = kv
        if self.cond_film:
            gamma, beta = self.film(tool_cond_bd).chunk(2, dim=-1)  # [B,D] each, zero-init => identity
            key = kv * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        if region_bias is not None:  # C003: [B,S] token region logits broadcast to every query row
            rb = region_bias.to(q_all.dtype).unsqueeze(1).expand(-1, int(q_all.shape[1]), -1)
            attn_bias = rb if attn_bias is None else (attn_bias.to(q_all.dtype) + rb)
        mask = None
        if attn_bias is not None:
            nh = int(self.cross_attn.num_heads)
            mask = attn_bias.to(q_all.dtype).unsqueeze(1).expand(-1, nh, -1, -1).reshape(bsz * nh, int(q_all.shape[1]), int(kv.shape[1]))
        try:
            h_all, attn_weights = self.cross_attn(
                query=q_all,
                key=key,
                value=kv,
                attn_mask=mask,
                need_weights=True,
                average_attn_weights=False,
            )
        except TypeError:
            h_all, attn_weights = self.cross_attn(
                query=q_all,
                key=key,
                value=kv,
                attn_mask=mask,
                need_weights=True,
            )
        if attn_weights is None:
            raise RuntimeError("cross_attn did not return attention weights while `need_weights=True`.")
        npt = int(self.num_points)
        if attn_weights.ndim == 4:
            # [B,heads,L,S] -> centroid query index 0 -> [B,S]; point queries 1..P -> [B,P,S] (virtual queries excluded)
            centroid_probs = attn_weights[:, :, 0, :].mean(dim=1)
            point_probs = attn_weights[:, :, 1:1 + npt, :].mean(dim=1)
        elif attn_weights.ndim == 3:
            # Fallback path from older torch: [B,L,S], already head-averaged.
            centroid_probs = attn_weights[:, 0, :]
            point_probs = attn_weights[:, 1:1 + npt, :]
        else:
            raise RuntimeError(f"Unexpected attention-weight shape: {tuple(attn_weights.shape)}")
        self._last_h_full = h_all  # [B,1+P+V,D] for the temporal/spatial mixing (pass 1)
        return h_all[:, :1, :], h_all[:, 1:1 + npt, :], centroid_probs, point_probs  # centroid_hidden, point_hidden, centroid_probs, point_probs

    def forward(
        self,
        feat_btsd: torch.Tensor,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        # feat_btsd: [B,T_block,S,D]
        if feat_btsd.ndim != 4:
            raise ValueError(f"`feat_btsd` must be [B,T,S,D], got {tuple(feat_btsd.shape)}")
        bsz, t_block, _s, d = feat_btsd.shape
        if int(d) != self.in_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.in_dim}, got {int(d)}")
        if int(t_block) <= 0:
            raise ValueError("`t_block` must be > 0")

        traj_cond = _build_trajectory_condition(
            condition_mode=self.condition_mode,
            ref_tensor=feat_btsd,
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
        pred_seq = []
        centroid_seq = []
        centroid_coarse_seq = []
        centroid_delta_seq = []
        offset_seq = []
        centroid_heatmap_seq = []
        sub_kl_seq = []
        sub_ent_seq = []
        point_attn_seq = []
        region_logit_seq = []
        centroid_attn_seq = []
        pp_xy_seq = []
        conf_logit_seq = []

        det_logit_seq = []
        det_xy_seq = []
        q_extra_seq = None
        attn_bias_seq = None
        lp_ref_seq, lp_stat_seq = [], []
        if self.temporal_query_mix:
            h_seq = []
            for bi in range(int(t_block)):
                for si in range(int(self.num_subframes)):
                    _h_c1, _h_p1, _cp1, _pp1 = self._cross_attend_step(tokens_bsd=feat_btsd[:, bi], tool_cond_bd=traj_cond, subframe_idx=si)
                    h_seq.append(self._last_h_full)  # [B,Q,D], Q = 1+P+V
                    if self.locality_prior:
                        _hh, _ww = self._resolve_spatial_hw(num_tokens=int(feat_btsd.shape[2]), spatial_hw=spatial_hw)
                        lp_ref_seq.append(self._perpoint_local_xy(_pp1, _hh, _ww).detach())  # [B,P,2]
                        _pm = _pp1.float(); lp_stat_seq.append(torch.stack([_pm.max(-1).values, -(_pm * _pm.clamp_min(1e-9).log()).sum(-1) / math.log(max(int(_pm.shape[-1]), 2))], -1).detach())  # [B,P,2]
            h_stack = torch.stack(h_seq, dim=1)  # [B,T,Q,D]
            n_t = int(h_stack.shape[1]); n_q = int(h_stack.shape[2])
            if n_t > self.temporal_query_max_frames:
                raise ValueError(f"temporal_query_mix supports up to {self.temporal_query_max_frames} frames, got {n_t}")
            x = (h_stack + self.tq_pos[:n_t].view(1, n_t, 1, -1)).permute(0, 2, 1, 3).reshape(-1, n_t, self.in_dim)  # [(B*Q),T,D]
            def _spatial(yy, lyr):
                ys = yy.reshape(bsz, n_q, n_t, self.in_dim).permute(0, 2, 1, 3).reshape(bsz * n_t, n_q, self.in_dim)
                ys = lyr(ys)
                return ys.reshape(bsz, n_t, n_q, self.in_dim).permute(0, 2, 1, 3).reshape(bsz * n_q, n_t, self.in_dim)
            if self.temporal_query_spatial and self.temporal_query_interleave:
                y = x
                temporal_layers = [self.tq_layer] + list(self.tq_extra_layers)
                for li, t_lyr in enumerate(temporal_layers):
                    y = t_lyr(y)
                    y = _spatial(y, self.tq_spatial_layers[li])
            else:
                y = self.tq_layer(x)
                for lyr in self.tq_extra_layers:
                    y = lyr(y)
                if self.temporal_query_spatial:
                    y = _spatial(y, self.tq_spatial_layer)
            delta = (y - x).reshape(bsz, n_q, n_t, self.in_dim).permute(0, 2, 1, 3)  # [B,T,Q,D], zero at init
            q_extra_seq = [delta[:, t] for t in range(n_t)]  # each [B,Q,D] (virtual rows included; pass 2 keeps them as queries)
            if self.locality_prior and lp_ref_seq:
                ref = torch.stack(lp_ref_seq, dim=1)  # [B,T,P,2]
                k = int(self.locality_prior_median)
                if k > 1:
                    pad = k // 2
                    _bp = ref.permute(0, 2, 3, 1).reshape(bsz * self.num_points * 2, 1, n_t)  # [(B*P*2),1,T]
                    refp = F.pad(_bp, (pad, pad), mode="replicate")  # [(B*P*2),1,T+2pad]
                    ref = refp.unfold(-1, k, 1).median(dim=-1).values.reshape(bsz, self.num_points, 2, n_t).permute(0, 3, 1, 2)  # [B,T,P,2]
                stat = torch.stack(lp_stat_seq, dim=1)  # [B,T,P,2]
                _hh, _ww = self._resolve_spatial_hw(num_tokens=int(feat_btsd.shape[2]), spatial_hw=spatial_hw)
                grid_xy = build_grid_xy(h_patches=_hh, w_patches=_ww, device=ref.device, dtype=ref.dtype)  # [S,2]
                if tool_id is not None:
                    tool_oh = F.one_hot(tool_id.view(-1).long().clamp(0, self.num_tool_classes - 1), self.num_tool_classes).float()
                else:
                    tool_oh = torch.zeros((bsz, self.num_tool_classes), device=ref.device)
                lam_in = torch.cat([stat, tool_oh.view(bsz, 1, 1, -1).expand(-1, n_t, self.num_points, -1)], dim=-1)  # [B,T,P,2+C]
                lam_p = self.lp_mlp(lam_in.to(self.lp_mlp[0].weight.dtype)).squeeze(-1)  # [B,T,P], zero at init
                ref_c = ref.mean(dim=2, keepdim=True)  # centroid reference [B,T,1,2]
                lam_c = lam_p.mean(dim=2, keepdim=True)
                ref_all = torch.cat([ref_c, ref], dim=2)  # [B,T,1+P,2]
                lam_all = torch.cat([lam_c, lam_p], dim=2)  # [B,T,1+P]
                scale = torch.tensor([float(_ww), float(_hh)], device=ref.device, dtype=ref.dtype)
                d2 = (((grid_xy.view(1, 1, 1, -1, 2) - ref_all.unsqueeze(3)) * scale) ** 2).sum(-1)  # [B,T,1+P,S] in cell^2
                bias = -lam_all.unsqueeze(-1) * d2  # [B,T,1+P,S]
                if n_q > 1 + self.num_points:
                    bias = torch.cat([bias, bias.new_zeros((bsz, n_t, n_q - 1 - self.num_points, int(d2.shape[-1])))], dim=2)
                attn_bias_seq = [bias[:, t] for t in range(n_t)]
        frame_i = 0
        for bi in range(int(t_block)):
            tokens = feat_btsd[:, bi]  # [B,S,D]
            det_prob_bi = None
            region_logits_bi = None
            if self.region_prior:
                region_logits_bi = self.rp_head(self.rp_norm(tokens)).squeeze(-1)  # [B,S]
            if self.token_detector:
                gamma_d, beta_d = self.det_film(traj_cond).chunk(2, dim=-1)
                det_in = self.det_norm(tokens) * (1.0 + gamma_d.unsqueeze(1)) + beta_d.unsqueeze(1)
                det_logits_bi = self.det_head(det_in)  # [B,S,3]
                det_prob_bi = torch.softmax(det_logits_bi.float(), dim=-1).to(tokens.dtype)
            for si in range(int(self.num_subframes)):
                q_extra_t = q_extra_seq[frame_i] if q_extra_seq is not None else None
                if self.pooled_context:
                    h_pc, w_pc = self._resolve_spatial_hw(num_tokens=int(tokens.shape[1]), spatial_hw=spatial_hw)
                    ctx = self._pooled_context_extra(tokens, traj_cond, si, h_pc, w_pc)
                    q_extra_t = ctx if q_extra_t is None else (q_extra_t + ctx)
                h_centroid, h_points, centroid_probs, point_probs = self._cross_attend_step(
                    tokens_bsd=tokens,
                    tool_cond_bd=traj_cond,
                    subframe_idx=si,
                    q_extra=q_extra_t,
                    attn_bias=(attn_bias_seq[frame_i] if attn_bias_seq is not None else None),
                    region_bias=((self.rp_scale * region_logits_bi) if region_logits_bi is not None else None),
                )
                frame_i += 1
                if region_logits_bi is not None:
                    region_logit_seq.append(region_logits_bi)
                centroid_attn_seq.append(centroid_probs)
                centroid_coarse_xy, centroid_heatmap, sub_kl, sub_ent = self._centroid_from_attn_probs(
                    centroid_probs=centroid_probs,
                    spatial_hw=spatial_hw,
                    tokens_bsd=tokens,
                )  # [B,1,2], [B,1,H,W], scalar, scalar
                sub_kl_seq.append(sub_kl)
                sub_ent_seq.append(sub_ent)
                h_hm, w_hm = self._resolve_spatial_hw(num_tokens=int(tokens.shape[1]), spatial_hw=spatial_hw)
                delta_scale = self._centroid_delta_scale(
                    spatial_hw=(h_hm, w_hm),
                    device=h_centroid.device,
                    dtype=h_centroid.dtype,
                )
                centroid_delta_xy = torch.tanh(self.centroid_delta_head(h_centroid)) * delta_scale  # [B,1,2]
                centroid_xy = (centroid_coarse_xy + centroid_delta_xy).clamp(0.0, 1.0)
                offsets_xy = torch.tanh(self.offset_head(h_points)) * self.max_offset_norm  # [B,P,2]
                offsets_xy = offsets_xy - offsets_xy.mean(dim=1, keepdim=True)
                pred_xy = (centroid_xy + offsets_xy).clamp(0.0, 1.0)  # [B,P,2]
                # --- E005: per-point local readout, gated into the shared-centroid path ---
                point_attn_seq.append(point_probs)
                if self.perpoint_local_readout:
                    if self.token_detector and det_prob_bi is not None:
                        xy_pp, pp_maps = self._detector_points(det_prob_bi, tool_id, h_hm, w_hm, ref_xy=pred_xy.detach())  # [B,P,2]
                        det_logit_seq.append(det_logits_bi)
                        det_xy_seq.append(xy_pp)
                    else:
                        xy_pp = self._perpoint_local_xy(point_probs, h_hm, w_hm)  # [B,P,2]
                        pp_maps = point_probs
                    g = torch.sigmoid(self.pp_gate_logit)
                    if self.adaptive_gate:
                        pm = pp_maps.float() / pp_maps.float().sum(-1, keepdim=True).clamp_min(1e-9)
                        pmax_g = pm.max(dim=-1).values  # [B,P]
                        pent_g = -(pm * pm.clamp_min(1e-9).log()).sum(-1) / math.log(max(int(pm.shape[-1]), 2))
                        dis_g = ((xy_pp - pred_xy).detach().float() * torch.tensor([float(w_hm), float(h_hm)], device=pred_xy.device)).norm(dim=-1) / 4.0  # cells / 4
                        if tool_id is not None:
                            tool_oh = F.one_hot(tool_id.view(-1).long().clamp(0, self.num_tool_classes - 1), self.num_tool_classes).float()
                        else:
                            tool_oh = torch.zeros((bsz, self.num_tool_classes), device=pred_xy.device)
                        gfeat = torch.cat([pmax_g.unsqueeze(-1), pent_g.unsqueeze(-1), dis_g.unsqueeze(-1), tool_oh.unsqueeze(1).expand(-1, pm.shape[1], -1)], dim=-1)
                        g = torch.sigmoid(self.pp_gate_logit + self.gate_mlp(gfeat.to(pred_xy.dtype)))  # [B,P,1]
                    pred_xy = ((1.0 - g) * pred_xy + g * xy_pp).clamp(0.0, 1.0)
                    # The refiner rebuilds stage-0 from (pred_centroid_xy + pred_offsets_xy), not from pred_xy,
                    # so fold the blend into the offsets (this also lifts the zero-mean constraint on them).
                    offsets_xy = pred_xy - centroid_xy
                    pp_xy_seq.append(xy_pp)
                if self.use_conf_head:
                    pmax = point_probs.max(dim=-1).values.unsqueeze(-1)  # [B,P,1]
                    pent = (-(point_probs * point_probs.clamp_min(1e-9).log()).sum(-1) / math.log(max(int(point_probs.shape[-1]), 2))).unsqueeze(-1)
                    conf_logit_seq.append(self.conf_mlp(torch.cat([h_points, pmax.to(h_points.dtype), pent.to(h_points.dtype)], dim=-1)).squeeze(-1))  # [B,P]
                pred_seq.append(pred_xy)
                centroid_seq.append(centroid_xy)
                centroid_coarse_seq.append(centroid_coarse_xy)
                centroid_delta_seq.append(centroid_delta_xy)
                offset_seq.append(offsets_xy)
                centroid_heatmap_seq.append(centroid_heatmap)

        pred_xy = torch.stack(pred_seq, dim=1).contiguous()  # [B,2*T_block,P,2]
        pred_centroid_xy = torch.stack(centroid_seq, dim=1).contiguous()  # [B,2*T_block,1,2]
        pred_centroid_coarse_xy = torch.stack(centroid_coarse_seq, dim=1).contiguous()  # [B,2*T_block,1,2]
        pred_centroid_delta_xy = torch.stack(centroid_delta_seq, dim=1).contiguous()  # [B,2*T_block,1,2]
        pred_offsets_xy = torch.stack(offset_seq, dim=1).contiguous()  # [B,2*T_block,P,2]
        pred_centroid_heatmap = torch.stack(centroid_heatmap_seq, dim=1).contiguous()  # [B,2*T_block,1,H,W]
        subcell_kl_t = torch.stack(sub_kl_seq).mean() if sub_kl_seq else pred_xy.new_zeros(())
        subcell_ent_t = torch.stack(sub_ent_seq).mean() if sub_ent_seq else pred_xy.new_zeros(())
        self.last_subcell_kl = float(subcell_kl_t.detach())
        self.last_subcell_entropy = float(subcell_ent_t.detach())
        delta_xy = torch.zeros_like(pred_xy)
        if int(pred_xy.shape[1]) > 1:
            delta_xy[:, 1:, :, :] = pred_xy[:, 1:, :, :] - pred_xy[:, :-1, :, :]
        delta_gate = torch.ones((bsz, int(pred_xy.shape[1]), self.num_points), device=feat_btsd.device, dtype=feat_btsd.dtype)
        conf = delta_gate
        # --- E005 extras (read by train._model_forward_with_cache and loss.compute_batch_losses) ---
        self.last_point_attn = torch.stack(point_attn_seq, dim=1).contiguous() if point_attn_seq else None  # [B,T,P,S]
        self.last_conf_logit = torch.stack(conf_logit_seq, dim=1).contiguous() if conf_logit_seq else None  # [B,T,P]
        self.last_pp_xy = torch.stack(pp_xy_seq, dim=1).contiguous() if pp_xy_seq else None  # [B,T,P,2] per-point path (before blend)
        self.last_det_logits = torch.stack(det_logit_seq, dim=1).contiguous() if det_logit_seq else None  # [B,T,S,3]
        self.last_det_xy = torch.stack(det_xy_seq, dim=1).contiguous() if det_xy_seq else None  # [B,T,P,2]
        self.last_region_logits = torch.stack(region_logit_seq, dim=1).contiguous() if region_logit_seq else None  # [B,T,S] (C003)
        self.last_centroid_attn = torch.stack(centroid_attn_seq, dim=1).contiguous() if centroid_attn_seq else None  # [B,T,S]
        if self.perpoint_local_readout:
            self.last_pp_gate = float(g.detach().float().mean()) if torch.is_tensor(g) else float(torch.sigmoid(self.pp_gate_logit).detach())
            self.last_pp_tau = float(torch.exp(self.pp_log_tau).detach())
        return {
            "pred_xy": pred_xy,
            "delta_xy": delta_xy,
            "delta_gate": delta_gate,
            "conf": conf,
            "pred_centroid_xy": pred_centroid_xy,
            "pred_centroid_coarse_xy": pred_centroid_coarse_xy,
            "pred_centroid_delta_xy": pred_centroid_delta_xy,
            "pred_offsets_xy": pred_offsets_xy,
            "pred_centroid_heatmap": pred_centroid_heatmap,
            "subcell_kl_uniform": subcell_kl_t,
            "subcell_entropy": subcell_ent_t,
        }


class TemporalDeconvUpsampler(nn.Module):
    """Lightweight temporal upsampler for 2x expansion on sequence axis."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                groups=groups,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,C,T]
        return self.net(x)


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


def _pick_attn_heads(embed_dim: int) -> int:
    for h in (16, 12, 8, 4, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


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


class ConditionalSpatialTokenPool(nn.Module):
    """TokenLearner/VSLNet-style conditioned local spatial evidence pooling on [B,T,S,D] tokens."""

    def __init__(
        self,
        in_dim: int,
        num_tokens: int = 8,
        condition_mode: str = "none",
        num_tool_classes: int = 0,
        num_action_classes: int = 0,
        num_combo_classes: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_tokens = int(num_tokens)
        self.condition_mode = str(condition_mode).lower()
        self.num_tool_classes = int(num_tool_classes)
        self.num_action_classes = int(num_action_classes)
        self.num_combo_classes = int(num_combo_classes)
        if self.in_dim <= 0:
            raise ValueError(f"`in_dim` must be > 0, got {self.in_dim}")
        if self.num_tokens <= 0:
            raise ValueError(f"`num_tokens` must be > 0, got {self.num_tokens}")
        if self.condition_mode not in ("none", "tool", "tool_action", "combo"):
            raise ValueError(
                f"Unsupported condition_mode={condition_mode}, expected one of: none,tool,tool_action,combo"
            )
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
            self.cond_feat_proj = nn.Linear(self.in_dim, self.in_dim)
            self.cond_query_proj = nn.Linear(self.in_dim, self.in_dim)
        else:
            self.tool_embed = None
            self.action_embed = None
            self.combo_embed = None
            self.cond_feat_proj = None
            self.cond_query_proj = None
        self.input_norm = nn.LayerNorm(self.in_dim)
        self.attn_in = nn.Conv2d(self.in_dim, self.in_dim, kernel_size=1, bias=True)
        self.attn_dw = nn.Conv2d(
            self.in_dim,
            self.in_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.in_dim,
            bias=False,
        )
        self.attn_out = nn.Conv2d(self.in_dim, self.num_tokens, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.token_pool = CrossAttentionPooler(in_dim=self.in_dim)

    def _build_condition(
        self,
        ref_tensor: torch.Tensor,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.condition_mode == "none":
            return None
        assert self.tool_embed is not None
        return _build_trajectory_condition(
            condition_mode=self.condition_mode,
            ref_tensor=ref_tensor,
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

    def forward(
        self,
        feat_flat: torch.Tensor,
        *,
        h_patches: int,
        w_patches: int,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if feat_flat.ndim != 4:
            raise ValueError(f"`feat_flat` must be [B,T,S,D], got {tuple(feat_flat.shape)}")
        bsz, t_len, s, d = feat_flat.shape
        if int(d) != self.in_dim:
            raise ValueError(f"`feat_flat` feature dim mismatch: expected {self.in_dim}, got {int(d)}")
        if int(h_patches) <= 0 or int(w_patches) <= 0 or int(h_patches) * int(w_patches) != int(s):
            raise ValueError(
                f"`h_patches` x `w_patches` must match token count S, got H={h_patches} W={w_patches} S={int(s)}"
            )
        if int(t_len) <= 0:
            return {
                "pooled": feat_flat.new_zeros((bsz, 0, self.in_dim)),
            }
        cond = self._build_condition(
            ref_tensor=feat_flat.mean(dim=(1, 2)),
            tool_id=tool_id,
            action_id=action_id,
            combo_id=combo_id,
        )
        feat_norm = self.input_norm(feat_flat)
        feat_map = feat_norm.reshape(bsz, t_len, int(h_patches), int(w_patches), self.in_dim)
        feat_map = feat_map.permute(0, 1, 4, 2, 3).reshape(bsz * t_len, self.in_dim, int(h_patches), int(w_patches))
        attn_in = feat_map
        query_bias = None
        if cond is not None:
            assert self.cond_feat_proj is not None and self.cond_query_proj is not None
            cond_feat = self.cond_feat_proj(cond).reshape(bsz, 1, self.in_dim, 1, 1)
            cond_feat = cond_feat.expand(-1, t_len, -1, -1, -1).reshape(bsz * t_len, self.in_dim, 1, 1)
            attn_in = attn_in + cond_feat.to(dtype=attn_in.dtype, device=attn_in.device)
            query_bias = self.cond_query_proj(cond).reshape(bsz, 1, self.in_dim)
            query_bias = query_bias.expand(-1, t_len, -1).reshape(bsz * t_len, self.in_dim)
        attn_h = self.act(self.attn_in(attn_in))
        attn_h = self.act(self.attn_dw(attn_h))
        attn_logits = self.attn_out(attn_h).reshape(bsz * t_len, self.num_tokens, int(s))
        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(dtype=feat_flat.dtype)
        value = feat_flat.reshape(bsz * t_len, int(s), self.in_dim)
        local_tokens = torch.einsum("bks,bsd->bkd", attn_weights, value)
        pooled = self.token_pool(local_tokens, query_bias=query_bias).reshape(bsz, t_len, self.in_dim)
        return {
            "pooled": pooled,
            "token_weights": attn_weights.reshape(bsz, t_len, self.num_tokens, int(s)),
        }


class TemporalMaxPoolContextBlock(nn.Module):
    """TemporalMaxer-style local max-pooling residual block on [B,T,D] features."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        if self.dim <= 0:
            raise ValueError(f"`dim` must be > 0, got {self.dim}")
        if self.kernel_size <= 0 or (self.kernel_size % 2) != 1:
            raise ValueError(
                f"`kernel_size` must be a positive odd integer, got {self.kernel_size}"
            )
        self.norm = nn.LayerNorm(self.dim)
        self.in_proj = nn.Linear(self.dim, self.dim)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"`x` feature dim mismatch: expected {self.dim}, got {int(x.shape[-1])}"
            )
        if int(x.shape[1]) <= 0:
            return x
        out = self.in_proj(self.norm(x))
        out = self.act(out)
        out = self.pool(out.transpose(1, 2)).transpose(1, 2)
        out = self.out_proj(out)
        return x + out


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


class TemporalLoSALiteAdapter(nn.Module):
    """LoSA-inspired long-short gated adapter for frozen [B,T,D] features."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        proj_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.proj_dim = int(proj_dim)
        self.num_heads = int(num_heads)
        if self.dim <= 0:
            raise ValueError(f"`dim` must be > 0, got {self.dim}")
        if self.kernel_size <= 0 or (self.kernel_size % 2) != 1:
            raise ValueError(
                f"`kernel_size` must be a positive odd integer, got {self.kernel_size}"
            )
        if self.proj_dim <= 0:
            raise ValueError(f"`proj_dim` must be > 0, got {self.proj_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}")
        if (self.dim % self.num_heads) != 0:
            raise ValueError(
                f"`dim` ({self.dim}) must be divisible by `num_heads` ({self.num_heads})"
            )
        self.short_norm = nn.LayerNorm(self.dim)
        self.short_in_proj = nn.Linear(self.dim, self.dim)
        self.short_conv = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        self.long_norm = nn.LayerNorm(self.dim)
        self.long_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.gate_short = nn.Parameter(torch.zeros(()))
        self.gate_long = nn.Parameter(torch.zeros(()))
        self.fuse_norm = nn.LayerNorm(self.dim * 2)
        self.fuse_proj = nn.Linear(self.dim * 2, self.dim)
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
        short = self.short_in_proj(self.short_norm(x))
        short = self.act(short)
        short = self.short_conv(short.transpose(1, 2)).transpose(1, 2)
        short = self.act(short)
        long_in = self.long_norm(x)
        long, _ = self.long_attn(long_in, long_in, long_in, need_weights=False)
        short_res = x + self.gate_short * short
        long_res = x + self.gate_long * long
        fused = torch.cat((short_res, long_res), dim=-1)
        fused = self.fuse_proj(self.fuse_norm(fused))
        fused = self.act(fused)
        adapted = x + self.out_proj(fused)
        projected = self.sim_proj(self.sim_norm(adapted))
        return {
            "adapted": adapted,
            "projected": projected,
        }


class TemporalDynamicMovementAdapter(nn.Module):
    """DyFADet/MENet-inspired local dynamic adapter for frozen [B,T,D] features."""

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

        kernel_candidates = {max(3, self.kernel_size - 2), self.kernel_size, self.kernel_size + 2}
        self.branch_kernels = tuple(sorted(k for k in kernel_candidates if k > 0 and (k % 2) == 1))
        self.num_branches = len(self.branch_kernels)
        if self.num_branches <= 0:
            raise ValueError("TemporalDynamicMovementAdapter requires at least one valid kernel branch.")

        self.input_norm = nn.LayerNorm(self.dim)
        self.mix_proj = nn.Linear(self.dim * 2, self.dim)
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(self.dim * 4),
            nn.Linear(self.dim * 4, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_branches),
        )
        self.branch_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.dim,
                    self.dim,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    groups=self.dim,
                    bias=False,
                )
                for k in self.branch_kernels
            ]
        )
        self.branch_pointwise = nn.ModuleList(
            [nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=True) for _ in self.branch_kernels]
        )
        self.branch_norm = nn.LayerNorm(self.dim)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.sim_norm = nn.LayerNorm(self.dim)
        self.sim_proj = nn.Linear(self.dim, self.proj_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _temporal_deltas(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,D], got {tuple(x.shape)}")
        if int(x.shape[1]) <= 0:
            return x, x
        forward = x.new_zeros(x.shape)
        backward = x.new_zeros(x.shape)
        if int(x.shape[1]) > 1:
            forward[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            backward[:, :-1, :] = x[:, :-1, :] - x[:, 1:, :]
        return forward, backward

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

        x_norm = self.input_norm(x)
        static = x_norm.mean(dim=1, keepdim=True).expand_as(x_norm)
        movement = x_norm - static
        forward_delta, backward_delta = self._temporal_deltas(x_norm)
        gate_in = torch.cat((x_norm, movement, forward_delta, backward_delta), dim=-1)
        branch_gate = torch.softmax(self.gate_proj(gate_in).float(), dim=-1).to(dtype=x.dtype)
        mixed = self.act(self.mix_proj(torch.cat((x_norm, movement), dim=-1)))

        branch_outs: List[torch.Tensor] = []
        mixed_bt = mixed.transpose(1, 2)
        for dw, pw in zip(self.branch_convs, self.branch_pointwise):
            y = pw(dw(mixed_bt)).transpose(1, 2)
            branch_outs.append(self.act(y))
        fused = torch.stack(branch_outs, dim=-2)
        fused = (fused * branch_gate.unsqueeze(-1)).sum(dim=-2)
        fused = self.act(self.branch_norm(fused))
        adapted = x + self.out_proj(fused)
        projected = self.sim_proj(self.sim_norm(adapted))
        return {
            "adapted": adapted,
            "projected": projected,
        }


class TemporalSoLaDyFAGatedAdapter(nn.Module):
    """Stable SoLa baseline plus zero-init DyFA motion branch."""

    def __init__(self, dim: int, kernel_size: int = 5, proj_dim: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.proj_dim = int(proj_dim)
        self.sola = TemporalSoLaAdapter(
            dim=self.dim,
            kernel_size=self.kernel_size,
            proj_dim=self.proj_dim,
        )
        self.dyfa = TemporalDynamicMovementAdapter(
            dim=self.dim,
            kernel_size=self.kernel_size,
            proj_dim=self.proj_dim,
        )
        self.dyfa_gate = nn.Parameter(torch.zeros(()))
        self.sim_norm = nn.LayerNorm(self.dim)
        self.sim_proj = nn.Linear(self.dim, self.proj_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        sola_out = self.sola(x)
        dyfa_out = self.dyfa(x)
        dyfa_delta = dyfa_out["adapted"] - x
        adapted = sola_out["adapted"] + self.dyfa_gate.to(dtype=x.dtype, device=x.device) * dyfa_delta
        projected = self.sim_proj(self.sim_norm(adapted))
        return {
            "adapted": adapted,
            "projected": projected,
        }


class TemporalSoLaDyFALiveGateAdapter(nn.Module):
    """Baseline-preserving SoLa + DyFA hybrid with a live DyFA gradient path."""

    def __init__(self, dim: int, kernel_size: int = 5, proj_dim: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.proj_dim = int(proj_dim)
        self.sola = TemporalSoLaAdapter(
            dim=self.dim,
            kernel_size=self.kernel_size,
            proj_dim=self.proj_dim,
        )
        self.dyfa = TemporalDynamicMovementAdapter(
            dim=self.dim,
            kernel_size=self.kernel_size,
            proj_dim=self.proj_dim,
        )
        # Small positive gate keeps the DyFA branch trainable even when its
        # zero-init residual projection starts at the identity function.
        self.dyfa_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        sola_out = self.sola(x)
        dyfa_out = self.dyfa(x)
        gate = self.dyfa_gate.to(dtype=x.dtype, device=x.device)
        dyfa_delta = dyfa_out["adapted"] - x
        adapted = sola_out["adapted"] + gate * dyfa_delta
        # Preserve the SoLa forward value exactly at init while still allowing
        # gradients to flow into the DyFA projection path.
        projected = sola_out["projected"] + gate * (
            dyfa_out["projected"] - dyfa_out["projected"].detach()
        )
        return {
            "adapted": adapted,
            "projected": projected,
            "reg_adapted": sola_out["adapted"],
            "score_adapted": adapted,
            "reg_projected": sola_out["projected"],
            "score_projected": projected,
        }


class TemporalSoLaDyFALocalFuseAdapter(nn.Module):
    """SoLa + DyFA hybrid with DCM-style local attentive fusion."""

    def __init__(self, dim: int, kernel_size: int = 5, proj_dim: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.proj_dim = int(proj_dim)
        self.sola = TemporalSoLaAdapter(
            dim=self.dim,
            kernel_size=self.kernel_size,
            proj_dim=self.proj_dim,
        )
        self.dyfa = TemporalDynamicMovementAdapter(
            dim=self.dim,
            kernel_size=self.kernel_size,
            proj_dim=self.proj_dim,
        )
        self.gate_norm = nn.LayerNorm(self.dim * 2)
        self.gate_proj = nn.Linear(self.dim * 2, 1)
        self.fusion_scale = 0.1
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        sola_out = self.sola(x)
        dyfa_out = self.dyfa(x)
        dyfa_delta = dyfa_out["adapted"] - x
        gate_in = torch.cat((sola_out["adapted"], dyfa_out["adapted"]), dim=-1)
        local_gate = torch.sigmoid(self.gate_proj(self.gate_norm(gate_in)))
        local_gate = local_gate.to(dtype=x.dtype, device=x.device) * float(self.fusion_scale)
        adapted = sola_out["adapted"] + local_gate * dyfa_delta
        projected = sola_out["projected"] + local_gate * (
            dyfa_out["projected"] - dyfa_out["projected"].detach()
        )
        return {
            "adapted": adapted,
            "projected": projected,
            "reg_adapted": sola_out["adapted"],
            "score_adapted": adapted,
            "reg_projected": sola_out["projected"],
            "score_projected": projected,
        }


class TemporalPyramidFusionContext(nn.Module):
    """ActionFormer/VSGN-inspired lightweight temporal pyramid fusion."""

    def __init__(self, dim: int, kernel_size: int = 3, num_levels: int = 3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.num_levels = int(num_levels)
        if self.dim <= 0:
            raise ValueError(f"`dim` must be > 0, got {self.dim}")
        if self.kernel_size <= 0 or (self.kernel_size % 2) != 1:
            raise ValueError(
                f"`kernel_size` must be a positive odd integer, got {self.kernel_size}"
            )
        if self.num_levels <= 0:
            raise ValueError(f"`num_levels` must be > 0, got {self.num_levels}")
        self.input_norm = nn.LayerNorm(self.dim)
        self.level_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.dim,
                    self.dim,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                )
                for _ in range(self.num_levels)
            ]
        )
        self.act = nn.GELU()
        self.fuse_norm = nn.LayerNorm(self.dim * self.num_levels)
        self.fuse_proj = nn.Linear(self.dim * self.num_levels, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _pool_half(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,D], got {tuple(x.shape)}")
        if int(x.shape[1]) <= 1:
            return x
        return F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        ).transpose(1, 2)

    @staticmethod
    def _resize_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        if int(x.shape[1]) == int(target_t):
            return x
        if int(x.shape[1]) <= 0:
            return x.new_zeros((int(x.shape[0]), int(target_t), int(x.shape[2])))
        return F.interpolate(
            x.transpose(1, 2),
            size=int(target_t),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B,T,D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"`x` feature dim mismatch: expected {self.dim}, got {int(x.shape[-1])}"
            )
        t_ref = int(x.shape[1])
        if t_ref <= 0:
            return x
        feat = self.input_norm(x)
        levels: List[torch.Tensor] = []
        cur = feat
        for level_idx, conv in enumerate(self.level_convs):
            y = conv(cur.transpose(1, 2)).transpose(1, 2)
            y = self.act(y)
            levels.append(self._resize_time(y, target_t=t_ref))
            if level_idx < (self.num_levels - 1):
                cur = self._pool_half(cur)
        fused = torch.cat(levels, dim=-1)
        fused = self.fuse_proj(self.fuse_norm(fused))
        fused = self.act(fused)
        return x + self.out_proj(fused)


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


class AlphaBetaPointRefiner(nn.Module):
    """Point-token spatio-temporal attention refiner.

    Inputs:
      - local_feat: [B,T,P,Df]
      - pred_xy:    [B,T,P,2]
      - pred_conf:  [B,T,P] (used only for gate)

    Pipeline:
      1) concat(local_feat_small, pred_xy) -> token
      2) same-frame point self-attention over P
      3) per-point temporal self-attention over T
      4) gated residual delta update on xy
    """

    def __init__(
        self,
        num_points: int,
        local_feat_dim_in: int = 512,
        local_feat_dim_small: int = 32,
        hidden_dim: int = 96,
        num_heads: int = 4,
        max_delta_norm: float = 0.02,
        gate_power: float = 2.0,
        gate_floor: float = 0.10,
        detach_refiner_input: bool = True,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.local_feat_dim_in = int(local_feat_dim_in)
        self.local_feat_dim_small = int(local_feat_dim_small)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.max_delta_norm = float(max_delta_norm)
        self.gate_power = float(gate_power)
        self.gate_floor = float(gate_floor)
        self.detach_refiner_input = bool(detach_refiner_input)
        self.debug_shapes = bool(debug_shapes)
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}"
            )

        # local_feat -> local_feat_small (e.g., 512 -> 32).
        self.local_feat_norm = nn.LayerNorm(self.local_feat_dim_in)
        self.local_feat_proj = nn.Linear(self.local_feat_dim_in, self.local_feat_dim_small)

        # token: [local_feat_small, x, y] -> hidden.
        self.input_proj = nn.Linear(self.local_feat_dim_small + 2, self.hidden_dim)

        # Step 1: same-frame point attention over P.
        self.point_ln1 = nn.LayerNorm(self.hidden_dim)
        self.point_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.point_ln2 = nn.LayerNorm(self.hidden_dim)
        self.point_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        # Step 2: temporal attention over T (per point).
        self.temporal_ln1 = nn.LayerNorm(self.hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.temporal_ln2 = nn.LayerNorm(self.hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.delta_head = nn.Linear(self.hidden_dim, 2)
        # Start from identity behavior and learn small corrections.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        local_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # pred_xy: [B,T,P,2], pred_conf: [B,T,P], local_feat: [B,T,P,Df]
        if pred_xy.ndim != 4 or int(pred_xy.shape[-1]) != 2:
            raise ValueError(f"`pred_xy` must be [B,T,P,2], got {tuple(pred_xy.shape)}")
        if pred_conf.ndim != 3:
            raise ValueError(f"`pred_conf` must be [B,T,P], got {tuple(pred_conf.shape)}")
        if int(pred_xy.shape[2]) != self.num_points:
            raise ValueError(f"`pred_xy` P mismatch: expected {self.num_points}, got {int(pred_xy.shape[2])}")
        bsz, t_len, n_pts, _ = pred_xy.shape

        base_xy = pred_xy.float()
        conf = pred_conf.float().clamp(0.0, 1.0)
        if local_feat is None:
            feat_in = base_xy.new_zeros((bsz, t_len, n_pts, self.local_feat_dim_in))
        else:
            if local_feat.ndim != 4 or tuple(local_feat.shape[:3]) != (bsz, t_len, n_pts):
                raise ValueError(
                    f"`local_feat` must be [B,T,P,Df] with B,T,P={(bsz, t_len, n_pts)}, got {tuple(local_feat.shape)}"
                )
            if int(local_feat.shape[-1]) != self.local_feat_dim_in:
                raise ValueError(
                    f"`local_feat` dim mismatch: expected Df={self.local_feat_dim_in}, got {int(local_feat.shape[-1])}"
                )
            feat_in = local_feat.float()

        ref_xy_in = base_xy.detach() if self.detach_refiner_input else base_xy
        ref_feat_in = feat_in.detach() if self.detach_refiner_input else feat_in

        local_feat_small = self.local_feat_proj(self.local_feat_norm(ref_feat_in))  # [B,T,P,D_small]
        token = torch.cat([local_feat_small, ref_xy_in], dim=-1)  # [B,T,P,D_small+2]
        h = self.input_proj(token)  # [B,T,P,H]

        # Step 1: same-frame point attention.
        h_point = h.reshape(bsz * t_len, n_pts, self.hidden_dim).contiguous()  # [B*T,P,H]
        qkv_p = self.point_ln1(h_point)
        h_point_attn, _ = self.point_attn(qkv_p, qkv_p, qkv_p, need_weights=False)
        h_point = h_point + h_point_attn
        h_point = h_point + self.point_ffn(self.point_ln2(h_point))
        h_spatial = h_point.reshape(bsz, t_len, n_pts, self.hidden_dim).contiguous()  # [B,T,P,H]

        # Step 2: temporal attention per point.
        h_time = h_spatial.permute(0, 2, 1, 3).reshape(bsz * n_pts, t_len, self.hidden_dim).contiguous()  # [B*P,T,H]
        qkv_t = self.temporal_ln1(h_time)
        h_time_attn, _ = self.temporal_attn(qkv_t, qkv_t, qkv_t, need_weights=False)
        h_time = h_time + h_time_attn
        h_time = h_time + self.temporal_ffn(self.temporal_ln2(h_time))
        h_out = h_time.reshape(bsz, n_pts, t_len, self.hidden_dim).permute(0, 2, 1, 3).contiguous()  # [B,T,P,H]

        delta_xy_raw = torch.tanh(self.delta_head(h_out)) * self.max_delta_norm
        gate_base = (1.0 - conf).clamp(0.0, 1.0).pow(self.gate_power)
        # Keep a non-zero correction path to avoid refine branch collapse.
        gate = (self.gate_floor + (1.0 - self.gate_floor) * gate_base).unsqueeze(-1)
        delta_xy = delta_xy_raw * gate
        refined_xy = (base_xy + delta_xy).clamp(0.0, 1.0)

        shape_debug = None
        if self.debug_shapes:
            shape_debug = {
                "local_feat_in": list(feat_in.shape),
                "local_feat_small": list(local_feat_small.shape),
                "token": list(token.shape),
                "h_point": list(h_point.shape),
                "h_spatial": list(h_spatial.shape),
                "h_time": list(h_time.shape),
                "h_out": list(h_out.shape),
                "delta_xy_raw": list(delta_xy_raw.shape),
                "delta_gate": list(gate.squeeze(-1).shape),
            }

        return {
            "pred_xy": refined_xy,
            "pred_conf": conf,
            "temporal_xy": ref_xy_in,
            "delta_xy": delta_xy,
            "delta_xy_raw": delta_xy_raw,
            "delta_gate_base": gate_base,
            "delta_gate": gate.squeeze(-1),
            "shape_debug": shape_debug,
        }


class LitePointRefiner(nn.Module):
    """Pre-upsample lightweight refiner with coordinate-rich tokenization.

    Inputs:
      - pred_xy:   [B,T,P,2]
      - pred_conf: [B,T,P]
      - local_feat: [B,T,P,Df] from argmax spatial cell feature
    """

    def __init__(
        self,
        num_points: int,
        local_feat_dim_in: int = 512,
        local_feat_dim_small: int = 32,
        hidden_dim: int = 64,
        num_freq_bands: int = 4,
        max_delta_norm: float = 0.02,
        gate_power: float = 2.0,
        gate_floor: float = 0.10,
        detach_refiner_input: bool = True,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.local_feat_dim_in = int(local_feat_dim_in)
        self.local_feat_dim_small = int(local_feat_dim_small)
        self.hidden_dim = int(hidden_dim)
        self.num_freq_bands = int(num_freq_bands)
        self.max_delta_norm = float(max_delta_norm)
        self.gate_power = float(gate_power)
        self.gate_floor = float(gate_floor)
        self.detach_refiner_input = bool(detach_refiner_input)
        self.debug_shapes = bool(debug_shapes)
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")
        if self.local_feat_dim_small <= 0:
            raise ValueError(f"`local_feat_dim_small` must be > 0, got {self.local_feat_dim_small}")
        if self.num_freq_bands <= 0:
            raise ValueError(f"`num_freq_bands` must be > 0, got {self.num_freq_bands}")

        # local feature branch (argmax-picked cell feature).
        self.local_feat_norm = nn.LayerNorm(self.local_feat_dim_in)
        self.local_feat_proj = nn.Linear(self.local_feat_dim_in, self.local_feat_dim_small)

        # Coord branch:
        #   raw = [xy, flow, vel] -> 6 dims
        #   fourier(raw) -> 6 * 2 * num_freq_bands dims
        self.coord_raw_dim = 6
        self.coord_fourier_dim = self.coord_raw_dim * 2 * self.num_freq_bands
        token_in_dim = self.local_feat_dim_small + self.coord_raw_dim + self.coord_fourier_dim + 1  # +conf

        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        # Lightweight temporal mixing over T for each point independently.
        self.temporal_dw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dim,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.delta_head = nn.Linear(self.hidden_dim, 2)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def _build_flow_and_vel(xy: torch.Tensor) -> torch.Tensor:
        # xy: [B,T,P,2]
        bsz, t_len, n_pts, _ = xy.shape
        xy0 = xy[:, :1, :, :].expand(bsz, t_len, n_pts, 2)
        flow = xy - xy0
        vel = torch.zeros_like(xy)
        if t_len > 1:
            vel[:, 1:, :, :] = xy[:, 1:, :, :] - xy[:, :-1, :, :]
        return torch.cat([xy, flow, vel], dim=-1)  # [B,T,P,6]

    def _fourier_encode(self, coord_raw: torch.Tensor) -> torch.Tensor:
        # coord_raw: [B,T,P,6]
        if coord_raw.ndim != 4 or int(coord_raw.shape[-1]) != self.coord_raw_dim:
            raise ValueError(
                f"`coord_raw` must be [B,T,P,{self.coord_raw_dim}], got {tuple(coord_raw.shape)}"
            )
        device = coord_raw.device
        dtype = coord_raw.dtype
        # Geometric frequency bands (1,2,4,8,...) * pi
        freq = (2.0 ** torch.arange(self.num_freq_bands, device=device, dtype=dtype)) * torch.pi  # [K]
        x = coord_raw.unsqueeze(-1) * freq.view(1, 1, 1, 1, -1)  # [B,T,P,6,K]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-2)  # [B,T,P,12,K]
        return emb.reshape(coord_raw.shape[0], coord_raw.shape[1], coord_raw.shape[2], -1).contiguous()

    def forward(
        self,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        local_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_xy.ndim != 4 or int(pred_xy.shape[-1]) != 2:
            raise ValueError(f"`pred_xy` must be [B,T,P,2], got {tuple(pred_xy.shape)}")
        if pred_conf.ndim != 3:
            raise ValueError(f"`pred_conf` must be [B,T,P], got {tuple(pred_conf.shape)}")
        if int(pred_xy.shape[2]) != self.num_points:
            raise ValueError(f"`pred_xy` P mismatch: expected {self.num_points}, got {int(pred_xy.shape[2])}")

        base_xy = pred_xy.float()
        conf = pred_conf.float().clamp(0.0, 1.0)
        ref_xy_in = base_xy.detach() if self.detach_refiner_input else base_xy
        if local_feat is None:
            feat_in = base_xy.new_zeros((base_xy.shape[0], base_xy.shape[1], base_xy.shape[2], self.local_feat_dim_in))
        else:
            if local_feat.ndim != 4 or tuple(local_feat.shape[:3]) != tuple(base_xy.shape[:3]):
                raise ValueError(
                    f"`local_feat` must be [B,T,P,Df] with B,T,P={tuple(base_xy.shape[:3])}, got {tuple(local_feat.shape)}"
                )
            if int(local_feat.shape[-1]) != self.local_feat_dim_in:
                raise ValueError(
                    f"`local_feat` dim mismatch: expected Df={self.local_feat_dim_in}, got {int(local_feat.shape[-1])}"
                )
            feat_in = local_feat.float()

        ref_feat_in = feat_in.detach() if self.detach_refiner_input else feat_in
        feat_small = self.local_feat_proj(self.local_feat_norm(ref_feat_in))  # [B,T,P,D_small]

        coord_raw = self._build_flow_and_vel(ref_xy_in)  # [B,T,P,6]
        coord_emb = self._fourier_encode(coord_raw)  # [B,T,P,E_coord]
        token = torch.cat([feat_small, coord_raw, coord_emb, conf.unsqueeze(-1)], dim=-1)  # [B,T,P,Dtok]
        h = self.token_proj(token)  # [B,T,P,H]

        # Temporal depthwise mixing per point.
        bsz, t_len, n_pts, _ = h.shape
        h_time = h.permute(0, 2, 3, 1).reshape(bsz * n_pts, self.hidden_dim, t_len).contiguous()  # [B*P,H,T]
        h_time = h_time + F.gelu(self.temporal_pw(self.temporal_dw(h_time)))
        h = h_time.reshape(bsz, n_pts, self.hidden_dim, t_len).permute(0, 3, 1, 2).contiguous()  # [B,T,P,H]
        h = h + self.temporal_ffn(self.temporal_ln(h))

        delta_xy_raw = torch.tanh(self.delta_head(h)) * self.max_delta_norm

        gate_base = (1.0 - conf).clamp(0.0, 1.0).pow(self.gate_power)
        gate = (self.gate_floor + (1.0 - self.gate_floor) * gate_base).unsqueeze(-1)
        delta_xy = delta_xy_raw * gate
        refined_xy = (base_xy + delta_xy).clamp(0.0, 1.0)

        shape_debug = None
        if self.debug_shapes:
            shape_debug = {
                "local_feat_in": list(feat_in.shape),
                "local_feat_small": list(feat_small.shape),
                "coord_raw": list(coord_raw.shape),
                "coord_emb": list(coord_emb.shape),
                "token": list(token.shape),
                "h_out": list(h.shape),
                "delta_xy_raw": list(delta_xy_raw.shape),
                "delta_gate": list(gate.squeeze(-1).shape),
            }

        return {
            "pred_xy": refined_xy,
            "pred_conf": conf,
            "temporal_xy": ref_xy_in,
            "delta_xy": delta_xy,
            "delta_xy_raw": delta_xy_raw,
            "delta_gate_base": gate_base,
            "delta_gate": gate.squeeze(-1),
            "shape_debug": shape_debug,
        }


class CoTrackerDeltaOffsetRefiner(nn.Module):
    """CoTracker-style residual offset refiner.

    This module copies the key contract from CoTracker:
      1) build transformer tokens from the current trajectory estimate
      2) add Fourier encoding of forward/backward motion
      3) add time embedding
      4) predict residual deltas and update the current state

    Unlike CoTracker, this first S1 version only refines offsets:
      p1 = c0 + (o0 + Delta_o1)
    """

    def __init__(
        self,
        num_points: int,
        local_feat_dim_in: int = 1024,
        local_feat_dim_small: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_freq_bands: int = 10,
        max_delta_norm: float = 0.02,
        train_clip_len: int = 16,
        detach_refiner_input: bool = True,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.local_feat_dim_in = int(local_feat_dim_in)
        self.local_feat_dim_small = int(local_feat_dim_small)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_freq_bands = int(num_freq_bands)
        self.max_delta_norm = float(max_delta_norm)
        self.detach_refiner_input = bool(detach_refiner_input)
        self.debug_shapes = bool(debug_shapes)
        if self.num_points <= 0:
            raise ValueError(f"`num_points` must be > 0, got {self.num_points}")
        if self.hidden_dim <= 0 or (self.hidden_dim % self.num_heads) != 0:
            raise ValueError(
                f"`hidden_dim` must be > 0 and divisible by `num_heads`, got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}"
            )
        if self.local_feat_dim_small <= 0:
            raise ValueError(f"`local_feat_dim_small` must be > 0, got {self.local_feat_dim_small}")
        if self.num_freq_bands <= 0:
            raise ValueError(f"`num_freq_bands` must be > 0, got {self.num_freq_bands}")
        if self.max_delta_norm <= 0.0:
            raise ValueError(f"`max_delta_norm` must be > 0, got {self.max_delta_norm}")

        # Match the CoTracker token pattern: current state + motion encoding + time embedding.
        self.motion_raw_dim = 4  # [rel_forward_xy, rel_backward_xy]
        self.motion_dim = self.motion_raw_dim * (1 + 2 * self.num_freq_bands)
        self.point_token_dim = self.local_feat_dim_small + 2 + 2 + self.motion_dim + 1
        self.centroid_token_dim = self.local_feat_dim_small + 2 + 1

        self.local_feat_norm = nn.LayerNorm(self.local_feat_dim_in)
        self.local_feat_proj = nn.Linear(self.local_feat_dim_in, self.local_feat_dim_small)
        self.centroid_feat_norm = nn.LayerNorm(self.local_feat_dim_in)
        self.centroid_feat_proj = nn.Linear(self.local_feat_dim_in, self.local_feat_dim_small)

        self.point_input_proj = nn.Linear(self.point_token_dim, self.hidden_dim)
        self.centroid_input_proj = nn.Linear(self.centroid_token_dim, self.hidden_dim)

        self.point_role_embedding = nn.Embedding(self.num_points, self.hidden_dim)
        nn.init.trunc_normal_(self.point_role_embedding.weight, std=0.02)
        self.cond_proj = nn.Linear(self.local_feat_dim_in, self.hidden_dim, bias=False)
        self.centroid_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.centroid_token, std=0.02)
        self.register_buffer("time_emb", build_time_emb_sincos(int(train_clip_len), self.hidden_dim), persistent=False)

        self.space_ln1 = nn.LayerNorm(self.hidden_dim)
        self.space_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.space_ln2 = nn.LayerNorm(self.hidden_dim)
        self.space_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.time_ln1 = nn.LayerNorm(self.hidden_dim)
        self.time_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.time_ln2 = nn.LayerNorm(self.hidden_dim)
        self.time_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.delta_offset_head = nn.Linear(self.hidden_dim, 2)
        nn.init.zeros_(self.delta_offset_head.weight)
        nn.init.zeros_(self.delta_offset_head.bias)

    @staticmethod
    def _build_rel_motion(xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_forward = xy[:, :-1] - xy[:, 1:]
        rel_backward = xy[:, 1:] - xy[:, :-1]
        rel_forward = F.pad(rel_forward, (0, 0, 0, 0, 0, 1))
        rel_backward = F.pad(rel_backward, (0, 0, 0, 0, 1, 0))
        return rel_forward, rel_backward

    def forward(
        self,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        local_feat: Optional[torch.Tensor] = None,
        *,
        pred_centroid_xy: Optional[torch.Tensor] = None,
        pred_offsets_xy: Optional[torch.Tensor] = None,
        centroid_local_feat: Optional[torch.Tensor] = None,
        traj_cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_xy.ndim != 4 or int(pred_xy.shape[-1]) != 2:
            raise ValueError(f"`pred_xy` must be [B,T,P,2], got {tuple(pred_xy.shape)}")
        if pred_conf.ndim != 3 or tuple(pred_conf.shape) != tuple(pred_xy.shape[:3]):
            raise ValueError(f"`pred_conf` must be [B,T,P], got {tuple(pred_conf.shape)}")
        bsz, t_len, n_pts, _ = pred_xy.shape
        if n_pts != self.num_points:
            raise ValueError(f"`pred_xy` P mismatch: expected {self.num_points}, got {n_pts}")

        base_xy = pred_xy.float()
        conf = pred_conf.float().clamp(0.0, 1.0)
        if pred_centroid_xy is None:
            base_centroid = base_xy.mean(dim=2, keepdim=True)
        else:
            if pred_centroid_xy.ndim == 3:
                pred_centroid_xy = pred_centroid_xy.unsqueeze(2)
            if pred_centroid_xy.ndim != 4 or tuple(pred_centroid_xy.shape[:3]) != (bsz, t_len, 1):
                raise ValueError(
                    f"`pred_centroid_xy` must be [B,T,1,2] or [B,T,2], got {tuple(pred_centroid_xy.shape)}"
                )
            base_centroid = pred_centroid_xy.float()
        if pred_offsets_xy is None:
            base_offsets = base_xy - base_centroid
        else:
            if pred_offsets_xy.ndim != 4 or tuple(pred_offsets_xy.shape[:3]) != (bsz, t_len, n_pts):
                raise ValueError(
                    f"`pred_offsets_xy` must be [B,T,P,2], got {tuple(pred_offsets_xy.shape)}"
                )
            base_offsets = pred_offsets_xy.float()

        if local_feat is None:
            feat_in = base_xy.new_zeros((bsz, t_len, n_pts, self.local_feat_dim_in))
        else:
            if local_feat.ndim != 4 or tuple(local_feat.shape[:3]) != (bsz, t_len, n_pts):
                raise ValueError(
                    f"`local_feat` must be [B,T,P,Df] with B,T,P={(bsz, t_len, n_pts)}, got {tuple(local_feat.shape)}"
                )
            if int(local_feat.shape[-1]) != self.local_feat_dim_in:
                raise ValueError(
                    f"`local_feat` dim mismatch: expected {self.local_feat_dim_in}, got {int(local_feat.shape[-1])}"
                )
            feat_in = local_feat.float()
        if centroid_local_feat is None:
            centroid_feat_in = base_xy.new_zeros((bsz, t_len, 1, self.local_feat_dim_in))
        else:
            if centroid_local_feat.ndim != 4 or tuple(centroid_local_feat.shape[:3]) != (bsz, t_len, 1):
                raise ValueError(
                    f"`centroid_local_feat` must be [B,T,1,Df], got {tuple(centroid_local_feat.shape)}"
                )
            if int(centroid_local_feat.shape[-1]) != self.local_feat_dim_in:
                raise ValueError(
                    f"`centroid_local_feat` dim mismatch: expected {self.local_feat_dim_in}, got {int(centroid_local_feat.shape[-1])}"
                )
            centroid_feat_in = centroid_local_feat.float()

        ref_xy = base_xy.detach() if self.detach_refiner_input else base_xy
        ref_offsets = base_offsets.detach() if self.detach_refiner_input else base_offsets
        ref_centroid = base_centroid.detach() if self.detach_refiner_input else base_centroid
        ref_feat = feat_in.detach() if self.detach_refiner_input else feat_in
        ref_centroid_feat = centroid_feat_in.detach() if self.detach_refiner_input else centroid_feat_in

        rel_forward, rel_backward = self._build_rel_motion(ref_xy)
        motion_state = torch.cat([rel_forward, rel_backward], dim=-1)
        motion_emb = cotracker_posenc(motion_state, min_deg=0, max_deg=self.num_freq_bands)

        point_feat_small = self.local_feat_proj(self.local_feat_norm(ref_feat))
        centroid_feat_small = self.centroid_feat_proj(self.centroid_feat_norm(ref_centroid_feat))

        point_token_in = torch.cat(
            [point_feat_small, ref_xy, ref_offsets, motion_emb, conf.unsqueeze(-1)],
            dim=-1,
        )
        centroid_conf = conf.mean(dim=2, keepdim=True).unsqueeze(-1)
        centroid_token_in = torch.cat([centroid_feat_small, ref_centroid, centroid_conf], dim=-1)

        point_tokens = self.point_input_proj(point_token_in)
        centroid_tokens = self.centroid_input_proj(centroid_token_in) + self.centroid_token

        role_ids = torch.arange(self.num_points, device=pred_xy.device, dtype=torch.long).view(1, 1, self.num_points)
        point_tokens = point_tokens + self.point_role_embedding(role_ids)
        time_bias = interpolate_time_emb(self.time_emb, int(t_len)).to(device=pred_xy.device, dtype=point_tokens.dtype)
        point_tokens = point_tokens + time_bias.unsqueeze(2)
        centroid_tokens = centroid_tokens + time_bias.unsqueeze(2)
        if traj_cond is not None:
            if traj_cond.ndim != 2 or tuple(traj_cond.shape) != (bsz, self.local_feat_dim_in):
                raise ValueError(
                    f"`traj_cond` must be [B,{self.local_feat_dim_in}], got {tuple(traj_cond.shape)}"
                )
            cond_bias = self.cond_proj(traj_cond.float()).view(bsz, 1, 1, self.hidden_dim)
            point_tokens = point_tokens + cond_bias
            centroid_tokens = centroid_tokens + cond_bias

        spatial_tokens = torch.cat([centroid_tokens, point_tokens], dim=2)
        spatial_tokens_bt = spatial_tokens.reshape(bsz * t_len, 1 + n_pts, self.hidden_dim).contiguous()
        qkv_space = self.space_ln1(spatial_tokens_bt)
        space_out, _ = self.space_attn(qkv_space, qkv_space, qkv_space, need_weights=False)
        spatial_tokens_bt = spatial_tokens_bt + space_out
        spatial_tokens_bt = spatial_tokens_bt + self.space_ffn(self.space_ln2(spatial_tokens_bt))
        spatial_tokens = spatial_tokens_bt.reshape(bsz, t_len, 1 + n_pts, self.hidden_dim).contiguous()

        point_tokens = spatial_tokens[:, :, 1:, :]
        point_tokens_tp = point_tokens.permute(0, 2, 1, 3).reshape(bsz * n_pts, t_len, self.hidden_dim).contiguous()
        qkv_time = self.time_ln1(point_tokens_tp)
        time_out, _ = self.time_attn(qkv_time, qkv_time, qkv_time, need_weights=False)
        point_tokens_tp = point_tokens_tp + time_out
        point_tokens_tp = point_tokens_tp + self.time_ffn(self.time_ln2(point_tokens_tp))
        point_tokens = point_tokens_tp.reshape(bsz, n_pts, t_len, self.hidden_dim).permute(0, 2, 1, 3).contiguous()

        delta_offsets_raw = torch.tanh(self.delta_offset_head(point_tokens)) * self.max_delta_norm
        delta_offsets = delta_offsets_raw - delta_offsets_raw.mean(dim=2, keepdim=True)
        refined_offsets = base_offsets + delta_offsets
        refined_offsets = refined_offsets - refined_offsets.mean(dim=2, keepdim=True)
        refined_xy = (base_centroid + refined_offsets).clamp(0.0, 1.0)
        delta_xy = refined_xy - base_xy

        shape_debug = None
        if self.debug_shapes:
            shape_debug = {
                "local_feat_in": list(feat_in.shape),
                "centroid_local_feat_in": list(centroid_feat_in.shape),
                "motion_state": list(motion_state.shape),
                "motion_emb": list(motion_emb.shape),
                "point_token_in": list(point_token_in.shape),
                "centroid_token_in": list(centroid_token_in.shape),
                "spatial_tokens": list(spatial_tokens.shape),
                "point_tokens_time": list(point_tokens.shape),
                "delta_offsets_raw": list(delta_offsets_raw.shape),
            }

        return {
            "pred_xy": refined_xy,
            "pred_conf": conf,
            "temporal_xy": ref_xy,
            "pred_offsets_xy": refined_offsets,
            "delta_offsets_xy": delta_offsets,
            "delta_xy": delta_xy,
            "delta_xy_raw": delta_xy,
            "delta_gate_base": torch.ones_like(conf),
            "delta_gate": torch.ones_like(conf),
            "shape_debug": shape_debug,
        }


class LocalSupportPoolProjector(nn.Module):
    """Project a small local support neighborhood back into one descriptor."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        merged_dim = self.feat_dim * 3
        self.norm = nn.LayerNorm(merged_dim)
        self.proj = nn.Sequential(
            nn.Linear(merged_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, support_feat: torch.Tensor) -> torch.Tensor:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        k_support = int(support_feat.shape[3])
        center_idx = k_support // 2
        center_feat = support_feat[:, :, :, center_idx, :]
        mean_feat = support_feat.mean(dim=3)
        max_feat = support_feat.amax(dim=3)
        merged = torch.cat([center_feat, mean_feat, max_feat], dim=-1)
        return self.proj(self.norm(merged))


class MultiLayerStagePairSupportProjector(nn.Module):
    """Project two fixed layer-specific local support neighborhoods back into one descriptor."""

    def __init__(self, feat_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_layers = int(num_layers)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}")
        merged_dim = self.feat_dim * self.num_layers * 3
        self.norm = nn.LayerNorm(merged_dim)
        self.proj = nn.Sequential(
            nn.Linear(merged_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, support_feat: torch.Tensor) -> torch.Tensor:
        if support_feat.ndim != 6:
            raise ValueError(f"`support_feat` must be [B,T,P,L,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[3]) != self.num_layers:
            raise ValueError(
                f"`support_feat` layer count mismatch: expected {self.num_layers}, got {int(support_feat.shape[3])}"
            )
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        k_support = int(support_feat.shape[4])
        center_idx = k_support // 2
        center_feat = support_feat[:, :, :, :, center_idx, :]
        mean_feat = support_feat.mean(dim=4)
        max_feat = support_feat.amax(dim=4)
        merged = torch.cat([center_feat, mean_feat, max_feat], dim=-1)
        merged = merged.reshape(*merged.shape[:3], -1)
        return self.proj(self.norm(merged))


class TemplateCenterCorrelationProjector(nn.Module):
    """Encode local evidence by matching a fixed template-center feature to current support."""

    def __init__(self, feat_dim: int, support_radius: int = 1, corr_temp: float = 10.0) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.support_radius = int(support_radius)
        self.corr_temp = float(corr_temp)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.support_radius < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {self.support_radius}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")

        self.k_support = (2 * self.support_radius + 1) ** 2
        merged_dim = self.feat_dim * 3 + (self.k_support * 2) + 6
        self.norm = nn.LayerNorm(merged_dim)
        self.proj = nn.Sequential(
            nn.Linear(merged_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def _support_rel_coords(self, support_feat: torch.Tensor) -> torch.Tensor:
        side = 2 * self.support_radius + 1
        if side * side != int(support_feat.shape[3]):
            raise ValueError(
                f"Support K mismatch for radius={self.support_radius}: expected {side * side}, got {int(support_feat.shape[3])}"
            )
        coords = torch.arange(
            -self.support_radius,
            self.support_radius + 1,
            device=support_feat.device,
            dtype=support_feat.dtype,
        )
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        rel = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, 1, side * side, 2)
        scale = float(max(self.support_radius, 1))
        return rel / scale

    def forward(self, support_feat: torch.Tensor, template_center_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        if template_center_feat.ndim not in (3, 4):
            raise ValueError(
                f"`template_center_feat` must be [B,P,D] or [B,1,P,D], got {tuple(template_center_feat.shape)}"
            )
        if template_center_feat.ndim == 4:
            if int(template_center_feat.shape[1]) != 1:
                raise ValueError(
                    f"`template_center_feat` time dim must be 1 when 4D, got {tuple(template_center_feat.shape)}"
                )
            template_center_feat = template_center_feat[:, 0, :, :]
        bsz, t_len, n_pts, k_support, feat_dim = support_feat.shape
        if tuple(template_center_feat.shape) != (bsz, n_pts, feat_dim):
            raise ValueError(
                f"`template_center_feat` must be [B,P,D]={bsz,n_pts,feat_dim}, got {tuple(template_center_feat.shape)}"
            )
        if int(k_support) != self.k_support:
            raise ValueError(
                f"`support_feat` K mismatch: expected {self.k_support}, got {int(k_support)}"
            )

        current_feat = support_feat.float()
        template_feat = template_center_feat.float().detach()
        template_norm = F.normalize(template_feat, dim=-1)
        current_norm = F.normalize(current_feat, dim=-1)
        corr = (current_norm * template_norm[:, None, :, None, :]).sum(dim=-1)
        probs = torch.softmax(corr * self.corr_temp, dim=-1)
        matched_feat = torch.einsum("btpk,btpkd->btpd", probs, current_feat)
        center_feat = current_feat[:, :, :, k_support // 2, :]
        template_tiled = template_feat[:, None, :, :].expand(-1, t_len, -1, -1)
        rel_coords = self._support_rel_coords(current_feat)
        matched_offset = torch.einsum("btpk,btpkc->btpc", probs, rel_coords.expand(bsz, t_len, n_pts, -1, -1))
        topk_vals = torch.topk(corr, k=min(2, k_support), dim=-1).values
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1] if int(topk_vals.shape[-1]) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1) / float(torch.log(torch.tensor(float(k_support), device=probs.device, dtype=probs.dtype)).item())
        merged = torch.cat(
            [
                center_feat,
                matched_feat,
                template_tiled,
                corr,
                probs,
                matched_offset,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                probs.amax(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        feat = self.proj(self.norm(merged))
        return {
            "feat": feat,
            "corr_scores": corr,
            "corr_probs": probs,
            "corr_entropy": entropy,
            "corr_margin": margin,
        }


class TemplateAllPairCorrelationProjector(nn.Module):
    """Encode local evidence with full 3x3 x 3x3 template/current correlation."""

    def __init__(self, feat_dim: int, support_radius: int = 1, corr_temp: float = 10.0) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.support_radius = int(support_radius)
        self.corr_temp = float(corr_temp)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.support_radius < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {self.support_radius}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")

        self.k_support = (2 * self.support_radius + 1) ** 2
        self.corr_embed = nn.Sequential(
            nn.LayerNorm(self.k_support * self.k_support),
            nn.Linear(self.k_support * self.k_support, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        merged_dim = self.feat_dim * 3 + 4
        self.norm = nn.LayerNorm(merged_dim)
        self.proj = nn.Sequential(
            nn.Linear(merged_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, support_feat: torch.Tensor, template_support_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        if template_support_feat.ndim not in (4, 5):
            raise ValueError(
                "`template_support_feat` must be [B,P,K,D] or [B,1,P,K,D], "
                f"got {tuple(template_support_feat.shape)}"
            )
        if template_support_feat.ndim == 5:
            if int(template_support_feat.shape[1]) != 1:
                raise ValueError(
                    f"`template_support_feat` time dim must be 1 when 5D, got {tuple(template_support_feat.shape)}"
                )
            template_support_feat = template_support_feat[:, 0, :, :, :]
        bsz, t_len, n_pts, k_support, feat_dim = support_feat.shape
        if tuple(template_support_feat.shape) != (bsz, n_pts, k_support, feat_dim):
            raise ValueError(
                f"`template_support_feat` must be [B,P,K,D]={bsz,n_pts,k_support,feat_dim}, "
                f"got {tuple(template_support_feat.shape)}"
            )
        if int(k_support) != self.k_support:
            raise ValueError(
                f"`support_feat` K mismatch: expected {self.k_support}, got {int(k_support)}"
            )

        current_feat = support_feat.float()
        template_feat = template_support_feat.float().detach()
        current_norm = F.normalize(current_feat, dim=-1)
        template_norm = F.normalize(template_feat, dim=-1)
        corr = torch.einsum("bpqd,btpkd->btpqk", template_norm, current_norm)
        corr_flat = corr.reshape(bsz, t_len, n_pts, self.k_support * self.k_support)
        corr_emb = self.corr_embed(corr_flat)
        probs = torch.softmax(corr_flat * self.corr_temp, dim=-1)
        topk_vals = torch.topk(corr_flat, k=min(2, corr_flat.shape[-1]), dim=-1).values
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1] if int(topk_vals.shape[-1]) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / float(
            torch.log(
                torch.tensor(float(self.k_support * self.k_support), device=probs.device, dtype=probs.dtype)
            ).item()
        )
        center_idx = self.k_support // 2
        current_center_feat = current_feat[:, :, :, center_idx, :]
        template_center_feat = template_feat[:, :, center_idx, :][:, None, :, :].expand(-1, t_len, -1, -1)
        merged = torch.cat(
            [
                corr_emb,
                current_center_feat,
                template_center_feat,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                probs.amax(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        feat = self.proj(self.norm(merged))
        return {
            "feat": feat,
            "corr_scores": corr_flat,
            "corr_probs": probs,
            "corr_entropy": entropy,
            "corr_margin": margin,
        }


class LayerwiseStateCarrySupportProjector(nn.Module):
    """State-conditioned local evidence encoder for layerwise trajectory refine."""

    def __init__(
        self,
        feat_dim: int,
        num_layers: int = 4,
        support_radius: int = 1,
        corr_temp: float = 10.0,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_layers = int(num_layers)
        self.support_radius = int(support_radius)
        self.corr_temp = float(corr_temp)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}")
        if self.support_radius < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {self.support_radius}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")

        self.k_support = (2 * self.support_radius + 1) ** 2
        self.init_state = nn.Parameter(torch.zeros(1, 1, 1, self.feat_dim))
        nn.init.trunc_normal_(self.init_state, std=0.02)
        self.layer_embed = nn.Embedding(self.num_layers, self.feat_dim)
        nn.init.trunc_normal_(self.layer_embed.weight, std=0.02)

        self.state_norm = nn.LayerNorm(self.feat_dim)
        self.support_norm = nn.LayerNorm(self.feat_dim)
        self.query_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.key_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.value_proj = nn.Linear(self.feat_dim, self.feat_dim)

        self.desc_norm = nn.LayerNorm((self.feat_dim * 5) + 10)
        self.desc_proj = nn.Sequential(
            nn.Linear((self.feat_dim * 5) + 10, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.state_update_norm = nn.LayerNorm((self.feat_dim * 3) + 10)
        self.state_update = nn.Sequential(
            nn.Linear((self.feat_dim * 3) + 10, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.state_gate = nn.Linear((self.feat_dim * 3) + 10, self.feat_dim)

    def _support_rel_coords(self, support_feat: torch.Tensor) -> torch.Tensor:
        side = 2 * self.support_radius + 1
        if side * side != int(support_feat.shape[3]):
            raise ValueError(
                f"Support K mismatch for radius={self.support_radius}: expected {side * side}, got {int(support_feat.shape[3])}"
            )
        coords = torch.arange(
            -self.support_radius,
            self.support_radius + 1,
            device=support_feat.device,
            dtype=support_feat.dtype,
        )
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        rel = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, 1, side * side, 2)
        scale = float(max(self.support_radius, 1))
        return rel / scale

    def forward(
        self,
        support_feat: torch.Tensor,
        *,
        prev_state: Optional[torch.Tensor],
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        layer_rank: int,
        prev_delta_xy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        bsz, t_len, n_pts, k_support, feat_dim = support_feat.shape
        if current_xy.ndim != 4 or tuple(current_xy.shape) != (bsz, t_len, n_pts, 2):
            raise ValueError(
                f"`current_xy` must be [B,T,P,2]={bsz,t_len,n_pts,2}, got {tuple(current_xy.shape)}"
            )
        if current_offsets.ndim != 4 or tuple(current_offsets.shape) != (bsz, t_len, n_pts, 2):
            raise ValueError(
                f"`current_offsets` must be [B,T,P,2]={bsz,t_len,n_pts,2}, got {tuple(current_offsets.shape)}"
            )
        if int(k_support) != self.k_support:
            raise ValueError(
                f"`support_feat` K mismatch: expected {self.k_support}, got {int(k_support)}"
            )
        if prev_state is None:
            state = self.init_state.expand(bsz, t_len, n_pts, -1)
        else:
            if prev_state.ndim != 4 or tuple(prev_state.shape) != (bsz, t_len, n_pts, feat_dim):
                raise ValueError(
                    f"`prev_state` must be [B,T,P,D]={bsz,t_len,n_pts,feat_dim}, got {tuple(prev_state.shape)}"
                )
            state = prev_state.float()

        current_feat = support_feat.float()
        center_idx = k_support // 2
        center_feat = current_feat[:, :, :, center_idx, :]
        layer_rank_clamped = int(max(0, min(self.num_layers - 1, int(layer_rank))))
        layer_bias = self.layer_embed.weight[layer_rank_clamped].view(1, 1, 1, self.feat_dim)
        state_with_layer = state + layer_bias

        query = F.normalize(self.query_proj(self.state_norm(state_with_layer)), dim=-1, eps=1e-6)
        keys = F.normalize(self.key_proj(self.support_norm(current_feat)), dim=-1, eps=1e-6)
        values = self.value_proj(current_feat)
        corr = (keys * query.unsqueeze(3)).sum(dim=-1)
        probs = torch.softmax(corr * self.corr_temp, dim=-1)
        attended_feat = torch.einsum("btpk,btpkd->btpd", probs, values)
        max_feat = current_feat.amax(dim=3)

        rel_coords = self._support_rel_coords(current_feat).expand(bsz, t_len, n_pts, -1, -1)
        expected_offset = torch.einsum("btpk,btpkc->btpc", probs, rel_coords)
        topk_vals = torch.topk(corr, k=min(2, k_support), dim=-1).values
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1] if int(topk_vals.shape[-1]) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / float(
            torch.log(torch.tensor(float(k_support), device=probs.device, dtype=probs.dtype)).item()
        )
        max_prob = probs.amax(dim=-1)

        desc_merged = torch.cat(
            [
                state,
                center_feat,
                attended_feat,
                max_feat,
                layer_bias.expand(bsz, t_len, n_pts, -1),
                current_xy.float(),
                current_offsets.float(),
                expected_offset,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        feat = self.desc_proj(self.desc_norm(desc_merged))

        state_update_in = torch.cat(
            [
                state,
                feat,
                layer_bias.expand(bsz, t_len, n_pts, -1),
                current_xy.float(),
                current_offsets.float(),
                expected_offset,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        state_update_in = self.state_update_norm(state_update_in)
        state_delta = self.state_update(state_update_in)
        state_gate = torch.sigmoid(self.state_gate(state_update_in))
        next_state = state + state_gate * state_delta
        return {
            "feat": feat,
            "state": next_state,
            "corr_scores": corr,
            "corr_probs": probs,
            "corr_entropy": entropy,
            "corr_margin": margin,
            "expected_offset": expected_offset,
            "max_prob": max_prob,
        }


class FourierBeliefStateCarrySupportProjector(nn.Module):
    """Fourier-enriched belief-state updater for single-layer iterative refine."""

    def __init__(
        self,
        feat_dim: int,
        num_layers: int = 4,
        support_radius: int = 1,
        corr_temp: float = 10.0,
        num_freq_bands: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_layers = int(num_layers)
        self.support_radius = int(support_radius)
        self.corr_temp = float(corr_temp)
        self.num_freq_bands = int(num_freq_bands)
        self.hidden_dim = int(hidden_dim)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}")
        if self.support_radius < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {self.support_radius}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")
        if self.num_freq_bands <= 0:
            raise ValueError(f"`num_freq_bands` must be > 0, got {self.num_freq_bands}")
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")

        self.k_support = (2 * self.support_radius + 1) ** 2
        self.coord_raw_dim = 8
        self.coord_fourier_dim = self.coord_raw_dim * 2 * self.num_freq_bands
        self.init_state = nn.Parameter(torch.zeros(1, 1, 1, self.feat_dim))
        nn.init.trunc_normal_(self.init_state, std=0.02)
        self.layer_embed = nn.Embedding(self.num_layers, self.feat_dim)
        nn.init.trunc_normal_(self.layer_embed.weight, std=0.02)

        self.state_norm = nn.LayerNorm(self.feat_dim)
        self.support_norm = nn.LayerNorm(self.feat_dim)
        self.query_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.key_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.value_proj = nn.Linear(self.feat_dim, self.feat_dim)

        self.obs_norm = nn.LayerNorm((self.feat_dim * 5) + 10)
        self.obs_proj = nn.Sequential(
            nn.Linear((self.feat_dim * 5) + 10, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

        token_in_dim = (self.feat_dim * 3) + self.coord_raw_dim + self.coord_fourier_dim + 6
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.temporal_dw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dim,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.temporal_to_state = nn.Linear(self.hidden_dim, self.feat_dim)
        self.state_update_norm = nn.LayerNorm((self.feat_dim * 4) + 6)
        self.state_update = nn.Sequential(
            nn.Linear((self.feat_dim * 4) + 6, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.belief_gate = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.feat_norm = nn.LayerNorm(self.feat_dim * 4)
        self.feat_proj = nn.Sequential(
            nn.Linear(self.feat_dim * 4, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        final_gate = self.belief_gate[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            nn.init.zeros_(final_gate.bias)

    def _support_rel_coords(self, support_feat: torch.Tensor) -> torch.Tensor:
        side = 2 * self.support_radius + 1
        if side * side != int(support_feat.shape[3]):
            raise ValueError(
                f"Support K mismatch for radius={self.support_radius}: expected {side * side}, got {int(support_feat.shape[3])}"
            )
        coords = torch.arange(
            -self.support_radius,
            self.support_radius + 1,
            device=support_feat.device,
            dtype=support_feat.dtype,
        )
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        rel = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, 1, side * side, 2)
        scale = float(max(self.support_radius, 1))
        return rel / scale

    def _fourier_encode(self, coord_raw: torch.Tensor) -> torch.Tensor:
        if coord_raw.ndim != 4 or int(coord_raw.shape[-1]) != self.coord_raw_dim:
            raise ValueError(
                f"`coord_raw` must be [B,T,P,{self.coord_raw_dim}], got {tuple(coord_raw.shape)}"
            )
        freq = (2.0 ** torch.arange(self.num_freq_bands, device=coord_raw.device, dtype=coord_raw.dtype)) * torch.pi
        x = coord_raw.unsqueeze(-1) * freq.view(1, 1, 1, 1, -1)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-2)
        return emb.reshape(coord_raw.shape[0], coord_raw.shape[1], coord_raw.shape[2], -1).contiguous()

    def forward(
        self,
        support_feat: torch.Tensor,
        *,
        prev_state: Optional[torch.Tensor],
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        layer_rank: int,
        prev_delta_xy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        bsz, t_len, n_pts, k_support, feat_dim = support_feat.shape
        if current_xy.ndim != 4 or tuple(current_xy.shape) != (bsz, t_len, n_pts, 2):
            raise ValueError(
                f"`current_xy` must be [B,T,P,2]={bsz,t_len,n_pts,2}, got {tuple(current_xy.shape)}"
            )
        if current_offsets.ndim != 4 or tuple(current_offsets.shape) != (bsz, t_len, n_pts, 2):
            raise ValueError(
                f"`current_offsets` must be [B,T,P,2]={bsz,t_len,n_pts,2}, got {tuple(current_offsets.shape)}"
            )
        if int(k_support) != self.k_support:
            raise ValueError(
                f"`support_feat` K mismatch: expected {self.k_support}, got {int(k_support)}"
            )
        if prev_state is None:
            state = self.init_state.expand(bsz, t_len, n_pts, -1)
        else:
            if prev_state.ndim != 4 or tuple(prev_state.shape) != (bsz, t_len, n_pts, feat_dim):
                raise ValueError(
                    f"`prev_state` must be [B,T,P,D]={bsz,t_len,n_pts,feat_dim}, got {tuple(prev_state.shape)}"
                )
            state = prev_state.float()
        if prev_delta_xy is None:
            prev_delta_xy = current_xy.new_zeros((bsz, t_len, n_pts, 2))
        elif prev_delta_xy.ndim != 4 or tuple(prev_delta_xy.shape) != (bsz, t_len, n_pts, 2):
            raise ValueError(
                f"`prev_delta_xy` must be [B,T,P,2]={bsz,t_len,n_pts,2}, got {tuple(prev_delta_xy.shape)}"
            )

        current_feat = support_feat.float()
        center_idx = k_support // 2
        center_feat = current_feat[:, :, :, center_idx, :]
        layer_rank_clamped = int(max(0, min(self.num_layers - 1, int(layer_rank))))
        layer_bias = self.layer_embed.weight[layer_rank_clamped].view(1, 1, 1, self.feat_dim)
        state_with_layer = state + layer_bias

        query = F.normalize(self.query_proj(self.state_norm(state_with_layer)), dim=-1, eps=1e-6)
        keys = F.normalize(self.key_proj(self.support_norm(current_feat)), dim=-1, eps=1e-6)
        values = self.value_proj(current_feat)
        corr = (keys * query.unsqueeze(3)).sum(dim=-1)
        probs = torch.softmax(corr * self.corr_temp, dim=-1)
        attended_feat = torch.einsum("btpk,btpkd->btpd", probs, values)
        max_feat = current_feat.amax(dim=3)

        rel_coords = self._support_rel_coords(current_feat).expand(bsz, t_len, n_pts, -1, -1)
        expected_offset = torch.einsum("btpk,btpkc->btpc", probs, rel_coords)
        topk_vals = torch.topk(corr, k=min(2, k_support), dim=-1).values
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1] if int(topk_vals.shape[-1]) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / float(
            torch.log(torch.tensor(float(k_support), device=probs.device, dtype=probs.dtype)).item()
        )
        max_prob = probs.amax(dim=-1)

        obs_merged = torch.cat(
            [
                state,
                center_feat,
                attended_feat,
                max_feat,
                layer_bias.expand(bsz, t_len, n_pts, -1),
                current_xy.float(),
                current_offsets.float(),
                expected_offset,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        obs_feat = self.obs_proj(self.obs_norm(obs_merged))

        coord_raw = torch.cat(
            [
                current_xy.float(),
                current_offsets.float(),
                expected_offset,
                prev_delta_xy.float(),
            ],
            dim=-1,
        )
        coord_fourier = self._fourier_encode(coord_raw)
        token_in = torch.cat(
            [
                state,
                obs_feat,
                layer_bias.expand(bsz, t_len, n_pts, -1),
                coord_raw,
                coord_fourier,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
                prev_delta_xy.float().norm(dim=-1, keepdim=True),
                expected_offset.float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        h = self.token_proj(token_in)
        h_time = h.permute(0, 2, 3, 1).reshape(bsz * n_pts, self.hidden_dim, t_len).contiguous()
        h_time = h_time + F.gelu(self.temporal_pw(self.temporal_dw(h_time)))
        h = h_time.reshape(bsz, n_pts, self.hidden_dim, t_len).permute(0, 3, 1, 2).contiguous()
        h = h + self.temporal_ffn(self.temporal_ln(h))
        temporal_feat = self.temporal_to_state(h)

        belief_gate_in = torch.cat(
            [
                entropy.unsqueeze(-1),
                margin.unsqueeze(-1),
                max_prob.unsqueeze(-1),
                expected_offset.float().norm(dim=-1, keepdim=True),
                prev_delta_xy.float().norm(dim=-1, keepdim=True),
                current_offsets.float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        belief_gate = torch.sigmoid(self.belief_gate(belief_gate_in)).squeeze(-1)

        state_update_in = torch.cat(
            [
                state,
                obs_feat,
                temporal_feat,
                layer_bias.expand(bsz, t_len, n_pts, -1),
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
                prev_delta_xy.float().norm(dim=-1, keepdim=True),
                expected_offset.float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        state_delta = self.state_update(self.state_update_norm(state_update_in))
        next_state = state + belief_gate.unsqueeze(-1) * state_delta
        feat = self.feat_proj(
            self.feat_norm(
                torch.cat(
                    [
                        obs_feat,
                        next_state,
                        temporal_feat,
                        layer_bias.expand(bsz, t_len, n_pts, -1),
                    ],
                    dim=-1,
                )
            )
        )
        return {
            "feat": feat,
            "state": next_state,
            "corr_scores": corr,
            "corr_probs": probs,
            "corr_entropy": entropy,
            "corr_margin": margin,
            "expected_offset": expected_offset,
            "max_prob": max_prob,
            "belief_gate": belief_gate,
        }


class ResidualFourierBeliefAdapterSupportProjector(nn.Module):
    """Keep the original state-carry projector and add a near-zero Fourier residual adapter."""

    def __init__(
        self,
        feat_dim: int,
        num_layers: int = 4,
        support_radius: int = 1,
        corr_temp: float = 10.0,
        num_freq_bands: int = 4,
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_layers = int(num_layers)
        self.support_radius = int(support_radius)
        self.corr_temp = float(corr_temp)
        self.num_freq_bands = int(num_freq_bands)
        self.hidden_dim = int(hidden_dim)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}")
        if self.support_radius < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {self.support_radius}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")
        if self.num_freq_bands <= 0:
            raise ValueError(f"`num_freq_bands` must be > 0, got {self.num_freq_bands}")
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")

        self.base = LayerwiseStateCarrySupportProjector(
            feat_dim=self.feat_dim,
            num_layers=self.num_layers,
            support_radius=self.support_radius,
            corr_temp=self.corr_temp,
        )
        self.coord_raw_dim = 8
        self.coord_fourier_dim = self.coord_raw_dim * 2 * self.num_freq_bands
        self.layer_embed = nn.Embedding(self.num_layers, self.feat_dim)
        nn.init.trunc_normal_(self.layer_embed.weight, std=0.02)

        token_in_dim = (self.feat_dim * 3) + self.coord_raw_dim + self.coord_fourier_dim + 6
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.temporal_dw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dim,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.temporal_to_state = nn.Linear(self.hidden_dim, self.feat_dim)
        self.adapter_norm = nn.LayerNorm((self.feat_dim * 4) + 6)
        self.state_resid = nn.Sequential(
            nn.Linear((self.feat_dim * 4) + 6, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.feat_resid = nn.Sequential(
            nn.Linear((self.feat_dim * 4) + 6, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.belief_gate = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        for head in (self.state_resid, self.feat_resid):
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        final_gate = self.belief_gate[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            nn.init.zeros_(final_gate.bias)

    def _fourier_encode(self, coord_raw: torch.Tensor) -> torch.Tensor:
        if coord_raw.ndim != 4 or int(coord_raw.shape[-1]) != self.coord_raw_dim:
            raise ValueError(
                f"`coord_raw` must be [B,T,P,{self.coord_raw_dim}], got {tuple(coord_raw.shape)}"
            )
        freq = (2.0 ** torch.arange(self.num_freq_bands, device=coord_raw.device, dtype=coord_raw.dtype)) * torch.pi
        x = coord_raw.unsqueeze(-1) * freq.view(1, 1, 1, 1, -1)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-2)
        return emb.reshape(coord_raw.shape[0], coord_raw.shape[1], coord_raw.shape[2], -1).contiguous()

    def forward(
        self,
        support_feat: torch.Tensor,
        *,
        prev_state: Optional[torch.Tensor],
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        layer_rank: int,
        prev_delta_xy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base_out = self.base(
            support_feat,
            prev_state=prev_state,
            current_xy=current_xy,
            current_offsets=current_offsets,
            layer_rank=layer_rank,
            prev_delta_xy=prev_delta_xy,
        )
        bsz, t_len, n_pts, _ = base_out["feat"].shape
        if prev_delta_xy is None:
            prev_delta_xy = current_xy.new_zeros((bsz, t_len, n_pts, 2))
        elif prev_delta_xy.ndim != 4 or tuple(prev_delta_xy.shape) != (bsz, t_len, n_pts, 2):
            raise ValueError(
                f"`prev_delta_xy` must be [B,T,P,2]={bsz,t_len,n_pts,2}, got {tuple(prev_delta_xy.shape)}"
            )

        layer_rank_clamped = int(max(0, min(self.num_layers - 1, int(layer_rank))))
        layer_bias = self.layer_embed.weight[layer_rank_clamped].view(1, 1, 1, self.feat_dim)
        coord_raw = torch.cat(
            [
                current_xy.float(),
                current_offsets.float(),
                base_out["expected_offset"].float(),
                prev_delta_xy.float(),
            ],
            dim=-1,
        )
        coord_fourier = self._fourier_encode(coord_raw)
        token_in = torch.cat(
            [
                base_out["state"],
                base_out["feat"],
                layer_bias.expand(bsz, t_len, n_pts, -1),
                coord_raw,
                coord_fourier,
                base_out["corr_scores"].amax(dim=-1, keepdim=True),
                base_out["corr_margin"].unsqueeze(-1),
                base_out["corr_entropy"].unsqueeze(-1),
                base_out["max_prob"].unsqueeze(-1),
                prev_delta_xy.float().norm(dim=-1, keepdim=True),
                base_out["expected_offset"].float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        h = self.token_proj(token_in)
        h_time = h.permute(0, 2, 3, 1).reshape(bsz * n_pts, self.hidden_dim, t_len).contiguous()
        h_time = h_time + F.gelu(self.temporal_pw(self.temporal_dw(h_time)))
        h = h_time.reshape(bsz, n_pts, self.hidden_dim, t_len).permute(0, 3, 1, 2).contiguous()
        h = h + self.temporal_ffn(self.temporal_ln(h))
        temporal_feat = self.temporal_to_state(h)

        gate_in = torch.cat(
            [
                base_out["corr_entropy"].unsqueeze(-1),
                base_out["corr_margin"].unsqueeze(-1),
                base_out["max_prob"].unsqueeze(-1),
                base_out["expected_offset"].float().norm(dim=-1, keepdim=True),
                prev_delta_xy.float().norm(dim=-1, keepdim=True),
                current_offsets.float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        belief_gate = torch.sigmoid(self.belief_gate(gate_in)).squeeze(-1)
        adapter_in = torch.cat(
            [
                base_out["state"],
                base_out["feat"],
                temporal_feat,
                layer_bias.expand(bsz, t_len, n_pts, -1),
                base_out["corr_scores"].amax(dim=-1, keepdim=True),
                base_out["corr_margin"].unsqueeze(-1),
                base_out["corr_entropy"].unsqueeze(-1),
                base_out["max_prob"].unsqueeze(-1),
                prev_delta_xy.float().norm(dim=-1, keepdim=True),
                base_out["expected_offset"].float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        adapter_in = self.adapter_norm(adapter_in)
        state_resid = self.state_resid(adapter_in)
        feat_resid = self.feat_resid(adapter_in)

        out = dict(base_out)
        out["state"] = base_out["state"] + belief_gate.unsqueeze(-1) * state_resid
        out["feat"] = base_out["feat"] + belief_gate.unsqueeze(-1) * feat_resid
        out["belief_gate"] = belief_gate
        return out


class ResidualCentroidAnchorAdapter(nn.Module):
    """Predict a small centroid correction from centroid-local evidence and point-layout stats."""

    def __init__(
        self,
        feat_dim: int,
        *,
        num_freq_bands: int = 4,
        hidden_dim: int = 192,
        max_delta_norm: float = 0.03,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_freq_bands = int(num_freq_bands)
        self.hidden_dim = int(hidden_dim)
        self.max_delta_norm = float(max_delta_norm)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_freq_bands <= 0:
            raise ValueError(f"`num_freq_bands` must be > 0, got {self.num_freq_bands}")
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")
        if self.max_delta_norm <= 0.0:
            raise ValueError(f"`max_delta_norm` must be > 0, got {self.max_delta_norm}")

        self.coord_raw_dim = 6
        self.coord_fourier_dim = self.coord_raw_dim * 2 * self.num_freq_bands
        token_in_dim = (self.feat_dim * 3) + self.coord_raw_dim + self.coord_fourier_dim + 4
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.temporal_dw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dim,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.feat_dim + 4),
            nn.Linear(self.hidden_dim + self.feat_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        final = self.delta_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _fourier_encode(self, coord_raw: torch.Tensor) -> torch.Tensor:
        if coord_raw.ndim != 4 or int(coord_raw.shape[-1]) != self.coord_raw_dim:
            raise ValueError(
                f"`coord_raw` must be [B,T,1,{self.coord_raw_dim}], got {tuple(coord_raw.shape)}"
            )
        freq = (2.0 ** torch.arange(self.num_freq_bands, device=coord_raw.device, dtype=coord_raw.dtype)) * torch.pi
        x = coord_raw.unsqueeze(-1) * freq.view(1, 1, 1, 1, -1)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-2)
        return emb.reshape(coord_raw.shape[0], coord_raw.shape[1], coord_raw.shape[2], -1).contiguous()

    @staticmethod
    def _pairwise_mean_distance(offsets: torch.Tensor) -> torch.Tensor:
        if offsets.ndim != 4 or int(offsets.shape[-1]) != 2:
            raise ValueError(f"`offsets` must be [B,T,P,2], got {tuple(offsets.shape)}")
        n_pts = int(offsets.shape[2])
        if n_pts <= 1:
            return offsets.new_zeros((offsets.shape[0], offsets.shape[1], 1, 1))
        pair = offsets.unsqueeze(3) - offsets.unsqueeze(2)
        pair_dist = torch.norm(pair, dim=-1)
        idx = torch.triu_indices(n_pts, n_pts, offset=1, device=offsets.device)
        pair_dist = pair_dist[:, :, idx[0], idx[1]]
        return pair_dist.mean(dim=-1, keepdim=True).unsqueeze(2)

    def forward(
        self,
        *,
        current_centroid_xy: torch.Tensor,
        centroid_local_feat: torch.Tensor,
        point_state: torch.Tensor,
        point_feat: torch.Tensor,
        current_offsets: torch.Tensor,
        expected_offset: torch.Tensor,
        corr_entropy: torch.Tensor,
        corr_margin: torch.Tensor,
        corr_max_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if current_centroid_xy.ndim != 4 or int(current_centroid_xy.shape[-1]) != 2:
            raise ValueError(f"`current_centroid_xy` must be [B,T,1,2], got {tuple(current_centroid_xy.shape)}")
        if centroid_local_feat.ndim != 4 or tuple(centroid_local_feat.shape[:3]) != tuple(current_centroid_xy.shape[:3]):
            raise ValueError(
                f"`centroid_local_feat` must be [B,T,1,D], got {tuple(centroid_local_feat.shape)}"
            )
        if point_state.ndim != 4 or int(point_state.shape[-1]) != self.feat_dim:
            raise ValueError(f"`point_state` must be [B,T,P,{self.feat_dim}], got {tuple(point_state.shape)}")
        if point_feat.ndim != 4 or int(point_feat.shape[-1]) != self.feat_dim:
            raise ValueError(f"`point_feat` must be [B,T,P,{self.feat_dim}], got {tuple(point_feat.shape)}")
        if current_offsets.ndim != 4 or tuple(current_offsets.shape[:3]) != tuple(point_state.shape[:3]):
            raise ValueError(f"`current_offsets` must be [B,T,P,2], got {tuple(current_offsets.shape)}")
        if expected_offset.ndim != 4 or tuple(expected_offset.shape[:3]) != tuple(point_state.shape[:3]):
            raise ValueError(f"`expected_offset` must be [B,T,P,2], got {tuple(expected_offset.shape)}")

        point_state_mean = point_state.float().mean(dim=2, keepdim=True)
        point_feat_mean = point_feat.float().mean(dim=2, keepdim=True)
        offset_abs = current_offsets.float().norm(dim=-1, keepdim=True)
        expected_abs = expected_offset.float().norm(dim=-1, keepdim=True)
        spread_mean = offset_abs.mean(dim=2, keepdim=True)
        spread_max = offset_abs.amax(dim=2, keepdim=True)
        expected_mean = expected_abs.mean(dim=2, keepdim=True)
        pair_mean = self._pairwise_mean_distance(current_offsets.float())
        entropy_mean = corr_entropy.float().mean(dim=2, keepdim=True).unsqueeze(-1)
        margin_mean = corr_margin.float().mean(dim=2, keepdim=True).unsqueeze(-1)
        max_prob_mean = corr_max_prob.float().mean(dim=2, keepdim=True).unsqueeze(-1)

        coord_raw = torch.cat(
            [
                current_centroid_xy.float(),
                spread_mean,
                spread_max,
                expected_mean,
                pair_mean,
            ],
            dim=-1,
        )
        coord_fourier = self._fourier_encode(coord_raw)
        token_in = torch.cat(
            [
                centroid_local_feat.float(),
                point_state_mean,
                point_feat_mean,
                coord_raw,
                coord_fourier,
                entropy_mean,
                margin_mean,
                max_prob_mean,
                spread_mean,
            ],
            dim=-1,
        )
        h = self.token_proj(token_in)
        bsz, t_len, _, _ = h.shape
        h_time = h.permute(0, 2, 3, 1).reshape(bsz, self.hidden_dim, t_len).contiguous()
        h_time = h_time + F.gelu(self.temporal_pw(self.temporal_dw(h_time)))
        h = h_time.permute(0, 2, 1).unsqueeze(2).contiguous()
        h = h + self.temporal_ffn(self.temporal_ln(h))
        delta_in = torch.cat([h, centroid_local_feat.float(), entropy_mean, margin_mean, max_prob_mean, spread_mean], dim=-1)
        delta_centroid = torch.tanh(self.delta_head(delta_in)) * self.max_delta_norm
        return {
            "delta_centroid_xy": delta_centroid,
            "spread_mean": spread_mean,
        }


class ShallowAnchorMotionVerifier(nn.Module):
    """Lightweight same-layer verifier that rescales an already-proposed centroid move."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        max_scale_residual: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_scale_residual = float(max_scale_residual)
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")
        if self.max_scale_residual <= 0.0:
            raise ValueError(f"`max_scale_residual` must be > 0, got {self.max_scale_residual}")

        self.stats_dim = 9
        self.scale_head = nn.Sequential(
            nn.LayerNorm(self.stats_dim),
            nn.Linear(self.stats_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        final = self.scale_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a, b, dim=-1, eps=1e-6).unsqueeze(-1)

    def _support_stats(self, support: torch.Tensor) -> Dict[str, torch.Tensor]:
        if support.ndim != 5:
            raise ValueError(f"`support` must be [B,T,P,K,D], got {tuple(support.shape)}")
        num_samples = int(support.shape[3])
        center_idx = num_samples // 2
        support_f = support.float()
        center_feat = support_f[:, :, :, center_idx, :]
        pooled_feat = support_f.mean(dim=3)
        if num_samples > 1:
            neighbor_feat = (support_f.sum(dim=3) - center_feat) / float(num_samples - 1)
        else:
            neighbor_feat = center_feat
        center_align = self._safe_cosine(center_feat, pooled_feat)
        neighbor_align = self._safe_cosine(center_feat, neighbor_feat)
        spread = torch.norm(support_f - pooled_feat.unsqueeze(3), dim=-1).mean(dim=3, keepdim=False).unsqueeze(-1)
        return {
            "center_feat": center_feat,
            "pooled_feat": pooled_feat,
            "center_align": center_align,
            "neighbor_align": neighbor_align,
            "spread": spread,
        }

    def forward(
        self,
        *,
        current_support: torch.Tensor,
        proposed_support: torch.Tensor,
        proposed_delta_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if current_support.shape != proposed_support.shape:
            raise ValueError(
                f"`current_support` and `proposed_support` must match, got "
                f"{tuple(current_support.shape)} vs {tuple(proposed_support.shape)}"
            )
        if proposed_delta_xy.ndim != 4 or int(proposed_delta_xy.shape[-1]) != 2:
            raise ValueError(f"`proposed_delta_xy` must be [B,T,P,2], got {tuple(proposed_delta_xy.shape)}")
        curr = self._support_stats(current_support)
        prop = self._support_stats(proposed_support)
        delta_norm = proposed_delta_xy.float().norm(dim=-1, keepdim=True)
        stats = torch.cat(
            [
                curr["center_align"],
                curr["neighbor_align"],
                curr["spread"],
                prop["center_align"],
                prop["neighbor_align"],
                prop["spread"],
                self._safe_cosine(curr["center_feat"], prop["center_feat"]),
                self._safe_cosine(curr["pooled_feat"], prop["pooled_feat"]),
                delta_norm,
            ],
            dim=-1,
        )
        scale_residual = torch.tanh(self.scale_head(stats)) * self.max_scale_residual
        scale = (1.0 + scale_residual).clamp(1.0 - self.max_scale_residual, 1.0 + self.max_scale_residual)
        return {
            "scale": scale.squeeze(-1),
            "scale_residual": scale_residual.squeeze(-1),
        }


class FactorizedToolAwareResidualAdapter(nn.Module):
    """Conservative shared-trunk residual adapter with centroid and layout heads."""

    def __init__(
        self,
        feat_dim: int,
        *,
        tool_cond_dim: Optional[int] = None,
        num_freq_bands: int = 4,
        hidden_dim: int = 192,
        max_centroid_delta_norm: float = 0.03,
        max_offset_delta_norm: float = 0.03,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.tool_cond_dim = int(tool_cond_dim if tool_cond_dim is not None else feat_dim)
        self.num_freq_bands = int(num_freq_bands)
        self.hidden_dim = int(hidden_dim)
        self.max_centroid_delta_norm = float(max_centroid_delta_norm)
        self.max_offset_delta_norm = float(max_offset_delta_norm)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.tool_cond_dim <= 0:
            raise ValueError(f"`tool_cond_dim` must be > 0, got {self.tool_cond_dim}")
        if self.num_freq_bands <= 0:
            raise ValueError(f"`num_freq_bands` must be > 0, got {self.num_freq_bands}")
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be > 0, got {self.hidden_dim}")
        if self.max_centroid_delta_norm <= 0.0:
            raise ValueError(f"`max_centroid_delta_norm` must be > 0, got {self.max_centroid_delta_norm}")
        if self.max_offset_delta_norm <= 0.0:
            raise ValueError(f"`max_offset_delta_norm` must be > 0, got {self.max_offset_delta_norm}")

        self.coord_raw_dim = 8
        self.coord_fourier_dim = self.coord_raw_dim * 2 * self.num_freq_bands
        token_in_dim = (self.feat_dim * 3) + self.coord_raw_dim + self.coord_fourier_dim + 4
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.cond_proj = nn.Linear(self.tool_cond_dim, self.hidden_dim, bias=False)
        self.temporal_dw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dim,
            bias=False,
        )
        self.temporal_pw = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        gate_stats_dim = 6
        centroid_in_dim = self.hidden_dim + (self.feat_dim * 3) + gate_stats_dim
        offset_in_dim = self.hidden_dim + self.feat_dim + 9
        mix_in_dim = (self.hidden_dim * 2) + gate_stats_dim
        self.centroid_head = nn.Sequential(
            nn.LayerNorm(centroid_in_dim),
            nn.Linear(centroid_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(offset_in_dim),
            nn.Linear(offset_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.mix_head = nn.Sequential(
            nn.LayerNorm(mix_in_dim),
            nn.Linear(mix_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        for head in (self.centroid_head, self.offset_head, self.mix_head):
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _fourier_encode(self, coord_raw: torch.Tensor) -> torch.Tensor:
        if coord_raw.ndim != 4 or int(coord_raw.shape[-1]) != self.coord_raw_dim:
            raise ValueError(
                f"`coord_raw` must be [B,T,P,{self.coord_raw_dim}], got {tuple(coord_raw.shape)}"
            )
        freq = (2.0 ** torch.arange(self.num_freq_bands, device=coord_raw.device, dtype=coord_raw.dtype)) * torch.pi
        x = coord_raw.unsqueeze(-1) * freq.view(1, 1, 1, 1, -1)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-2)
        return emb.reshape(coord_raw.shape[0], coord_raw.shape[1], coord_raw.shape[2], -1).contiguous()

    @staticmethod
    def _pairwise_mean_distance(offsets: torch.Tensor) -> torch.Tensor:
        if offsets.ndim != 4 or int(offsets.shape[-1]) != 2:
            raise ValueError(f"`offsets` must be [B,T,P,2], got {tuple(offsets.shape)}")
        n_pts = int(offsets.shape[2])
        if n_pts <= 1:
            return offsets.new_zeros((offsets.shape[0], offsets.shape[1], 1, 1))
        pair = offsets.unsqueeze(3) - offsets.unsqueeze(2)
        pair_dist = torch.norm(pair, dim=-1)
        idx = torch.triu_indices(n_pts, n_pts, offset=1, device=offsets.device)
        pair_dist = pair_dist[:, :, idx[0], idx[1]]
        return pair_dist.mean(dim=-1, keepdim=True).unsqueeze(2)

    def forward(
        self,
        *,
        current_centroid_xy: torch.Tensor,
        centroid_local_feat: torch.Tensor,
        point_state: torch.Tensor,
        point_feat: torch.Tensor,
        current_offsets: torch.Tensor,
        expected_offset: torch.Tensor,
        corr_entropy: torch.Tensor,
        corr_margin: torch.Tensor,
        corr_max_prob: torch.Tensor,
        traj_tool_cond: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if current_centroid_xy.ndim != 4 or int(current_centroid_xy.shape[-1]) != 2:
            raise ValueError(f"`current_centroid_xy` must be [B,T,1,2], got {tuple(current_centroid_xy.shape)}")
        if centroid_local_feat.ndim != 4 or tuple(centroid_local_feat.shape[:3]) != tuple(current_centroid_xy.shape[:3]):
            raise ValueError(
                f"`centroid_local_feat` must be [B,T,1,D], got {tuple(centroid_local_feat.shape)}"
            )
        if point_state.ndim != 4 or int(point_state.shape[-1]) != self.feat_dim:
            raise ValueError(f"`point_state` must be [B,T,P,{self.feat_dim}], got {tuple(point_state.shape)}")
        if point_feat.ndim != 4 or int(point_feat.shape[-1]) != self.feat_dim:
            raise ValueError(f"`point_feat` must be [B,T,P,{self.feat_dim}], got {tuple(point_feat.shape)}")
        if current_offsets.ndim != 4 or tuple(current_offsets.shape[:3]) != tuple(point_state.shape[:3]):
            raise ValueError(f"`current_offsets` must be [B,T,P,2], got {tuple(current_offsets.shape)}")
        if expected_offset.ndim != 4 or tuple(expected_offset.shape[:3]) != tuple(point_state.shape[:3]):
            raise ValueError(f"`expected_offset` must be [B,T,P,2], got {tuple(expected_offset.shape)}")

        bsz, t_len, n_pts, _ = point_state.shape
        centroid_feat_exp = centroid_local_feat.float().expand(-1, -1, n_pts, -1)
        centroid_xy_exp = current_centroid_xy.float().expand(-1, -1, n_pts, -1)
        offset_delta = current_offsets.float() - expected_offset.float()
        coord_raw = torch.cat(
            [
                current_offsets.float(),
                expected_offset.float(),
                centroid_xy_exp,
                offset_delta,
            ],
            dim=-1,
        )
        coord_fourier = self._fourier_encode(coord_raw)
        token_in = torch.cat(
            [
                point_state.float(),
                point_feat.float(),
                centroid_feat_exp,
                coord_raw,
                coord_fourier,
                corr_entropy.float().unsqueeze(-1),
                corr_margin.float().unsqueeze(-1),
                corr_max_prob.float().unsqueeze(-1),
                current_offsets.float().norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        h = self.token_proj(token_in)
        if traj_tool_cond is not None:
            if traj_tool_cond.ndim != 2 or int(traj_tool_cond.shape[0]) != bsz or int(traj_tool_cond.shape[1]) != self.tool_cond_dim:
                raise ValueError(
                    f"`traj_tool_cond` must be [B,{self.tool_cond_dim}], got {tuple(traj_tool_cond.shape)}"
                )
            cond_bias = self.cond_proj(traj_tool_cond.float()).view(bsz, 1, 1, self.hidden_dim)
            h = h + cond_bias
        else:
            cond_bias = h.new_zeros((bsz, 1, 1, self.hidden_dim))
        h_time = h.permute(0, 2, 3, 1).reshape(bsz * n_pts, self.hidden_dim, t_len).contiguous()
        h_time = h_time + F.gelu(self.temporal_pw(self.temporal_dw(h_time)))
        h = h_time.reshape(bsz, n_pts, self.hidden_dim, t_len).permute(0, 3, 1, 2).contiguous()
        h = h + self.temporal_ffn(self.temporal_ln(h))

        point_state_mean = point_state.float().mean(dim=2, keepdim=True)
        point_feat_mean = point_feat.float().mean(dim=2, keepdim=True)
        pooled_h = h.mean(dim=2, keepdim=True)
        offset_abs = current_offsets.float().norm(dim=-1, keepdim=True)
        expected_abs = expected_offset.float().norm(dim=-1, keepdim=True)
        spread_mean = offset_abs.mean(dim=2, keepdim=True)
        expected_mean = expected_abs.mean(dim=2, keepdim=True)
        pair_mean = self._pairwise_mean_distance(current_offsets.float())
        entropy_mean = corr_entropy.float().mean(dim=2, keepdim=True).unsqueeze(-1)
        margin_mean = corr_margin.float().mean(dim=2, keepdim=True).unsqueeze(-1)
        max_prob_mean = corr_max_prob.float().mean(dim=2, keepdim=True).unsqueeze(-1)
        gate_stats = torch.cat(
            [
                spread_mean,
                expected_mean,
                pair_mean,
                entropy_mean,
                margin_mean,
                max_prob_mean,
            ],
            dim=-1,
        )
        mix_in = torch.cat(
            [
                pooled_h,
                cond_bias.expand(-1, t_len, -1, -1),
                gate_stats,
            ],
            dim=-1,
        )
        mix_logits = self.mix_head(mix_in)
        anchor_gate = torch.sigmoid(mix_logits[..., :1])
        offset_gate = torch.sigmoid(mix_logits[..., 1:2])

        centroid_in = torch.cat(
            [
                pooled_h,
                centroid_local_feat.float(),
                point_state_mean,
                point_feat_mean,
                gate_stats,
            ],
            dim=-1,
        )
        delta_centroid = torch.tanh(self.centroid_head(centroid_in)) * self.max_centroid_delta_norm
        delta_centroid = delta_centroid * anchor_gate

        offset_in = torch.cat(
            [
                h,
                point_feat.float(),
                current_offsets.float(),
                expected_offset.float(),
                corr_entropy.float().unsqueeze(-1),
                corr_margin.float().unsqueeze(-1),
                corr_max_prob.float().unsqueeze(-1),
                offset_abs,
                expected_abs,
            ],
            dim=-1,
        )
        delta_offsets = torch.tanh(self.offset_head(offset_in)) * self.max_offset_delta_norm
        delta_offsets = delta_offsets - delta_offsets.mean(dim=2, keepdim=True)
        delta_offsets = delta_offsets * offset_gate.expand(-1, -1, n_pts, -1)

        return {
            "delta_centroid_xy": delta_centroid,
            "delta_offsets_xy": delta_offsets,
            "anchor_gate": anchor_gate.squeeze(-1),
            "offset_gate": offset_gate.squeeze(-1),
            "delta_offsets_abs": delta_offsets.float().norm(dim=-1),
        }


class LayerwiseContextPointEvidenceAdapter(nn.Module):
    """Support the main point with a small learned set of nearby context points."""

    def __init__(
        self,
        feat_dim: int,
        *,
        num_context_points: int = 4,
        max_offset_cells: float = 2.0,
        corr_temp: float = 10.0,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_context_points = int(num_context_points)
        self.max_offset_cells = float(max_offset_cells)
        self.corr_temp = float(corr_temp)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_context_points <= 0:
            raise ValueError(f"`num_context_points` must be > 0, got {self.num_context_points}")
        if self.max_offset_cells <= 0.0:
            raise ValueError(f"`max_offset_cells` must be > 0, got {self.max_offset_cells}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")

        self.query_norm = nn.LayerNorm((self.feat_dim * 3) + 6)
        self.query_proj = nn.Sequential(
            nn.Linear((self.feat_dim * 3) + 6, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.query_key_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.sampling_offsets = nn.Linear(self.feat_dim, self.num_context_points * 2)
        self.candidate_norm = nn.LayerNorm(self.feat_dim)
        self.key_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.value_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.out_norm = nn.LayerNorm((self.feat_dim * 5) + 10)
        self.out_proj = nn.Sequential(
            nn.Linear((self.feat_dim * 5) + 10, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.num_context_points, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / float(max(self.num_context_points, 1)))
        grid = 0.5 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(-1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid)

    def build_query_and_offsets(
        self,
        *,
        main_feat: torch.Tensor,
        main_state: torch.Tensor,
        main_support_feat: torch.Tensor,
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        expected_offset: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if main_feat.ndim != 4 or int(main_feat.shape[-1]) != self.feat_dim:
            raise ValueError(f"`main_feat` must be [B,T,P,D={self.feat_dim}], got {tuple(main_feat.shape)}")
        if main_state.ndim != 4 or tuple(main_state.shape) != tuple(main_feat.shape):
            raise ValueError(f"`main_state` must match `main_feat`, got {tuple(main_state.shape)} vs {tuple(main_feat.shape)}")
        if main_support_feat.ndim != 4 or tuple(main_support_feat.shape) != tuple(main_feat.shape):
            raise ValueError(
                f"`main_support_feat` must match `main_feat`, got {tuple(main_support_feat.shape)} vs {tuple(main_feat.shape)}"
            )
        if current_xy.ndim != 4 or tuple(current_xy.shape[:3]) != tuple(main_feat.shape[:3]) or int(current_xy.shape[-1]) != 2:
            raise ValueError(f"`current_xy` must be [B,T,P,2], got {tuple(current_xy.shape)}")
        if current_offsets.ndim != 4 or tuple(current_offsets.shape) != tuple(current_xy.shape):
            raise ValueError(
                f"`current_offsets` must match `current_xy`, got {tuple(current_offsets.shape)} vs {tuple(current_xy.shape)}"
            )
        if expected_offset.ndim != 4 or tuple(expected_offset.shape) != tuple(current_xy.shape):
            raise ValueError(
                f"`expected_offset` must match `current_xy`, got {tuple(expected_offset.shape)} vs {tuple(current_xy.shape)}"
            )
        if int(h_patches) <= 0 or int(w_patches) <= 0:
            raise ValueError(f"Invalid patch grid size: H={h_patches}, W={w_patches}")

        query_in = torch.cat(
            [
                main_feat.float(),
                main_state.float(),
                main_support_feat.float(),
                current_xy.float(),
                current_offsets.float(),
                expected_offset.float(),
            ],
            dim=-1,
        )
        query = self.query_proj(self.query_norm(query_in))
        offset_raw = self.sampling_offsets(query).view(*query.shape[:3], self.num_context_points, 2)
        offset_scale = query.new_tensor(
            [
                self.max_offset_cells / float(max(int(w_patches), 1)),
                self.max_offset_cells / float(max(int(h_patches), 1)),
            ]
        ).view(1, 1, 1, 1, 2)
        offsets = torch.tanh(offset_raw) * offset_scale
        return query, offsets

    def forward(
        self,
        *,
        main_feat: torch.Tensor,
        main_support_feat: torch.Tensor,
        context_support_feat: torch.Tensor,
        query: torch.Tensor,
        current_offsets: torch.Tensor,
        expected_offset: torch.Tensor,
        context_offsets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if main_feat.ndim != 4 or int(main_feat.shape[-1]) != self.feat_dim:
            raise ValueError(f"`main_feat` must be [B,T,P,D={self.feat_dim}], got {tuple(main_feat.shape)}")
        if main_support_feat.ndim != 4 or tuple(main_support_feat.shape) != tuple(main_feat.shape):
            raise ValueError(
                f"`main_support_feat` must match `main_feat`, got {tuple(main_support_feat.shape)} vs {tuple(main_feat.shape)}"
            )
        if context_support_feat.ndim != 5:
            raise ValueError(f"`context_support_feat` must be [B,T,P,K,D], got {tuple(context_support_feat.shape)}")
        if tuple(context_support_feat.shape[:3]) != tuple(main_feat.shape[:3]):
            raise ValueError(
                f"`context_support_feat` batch/time/point dims must match `main_feat`, got {tuple(context_support_feat.shape)} vs {tuple(main_feat.shape)}"
            )
        if int(context_support_feat.shape[3]) != self.num_context_points or int(context_support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`context_support_feat` must have shape [B,T,P,{self.num_context_points},{self.feat_dim}], got {tuple(context_support_feat.shape)}"
            )
        if query.ndim != 4 or tuple(query.shape) != tuple(main_feat.shape):
            raise ValueError(f"`query` must match `main_feat`, got {tuple(query.shape)} vs {tuple(main_feat.shape)}")
        if current_offsets.ndim != 4 or tuple(current_offsets.shape[:3]) != tuple(main_feat.shape[:3]) or int(current_offsets.shape[-1]) != 2:
            raise ValueError(f"`current_offsets` must be [B,T,P,2], got {tuple(current_offsets.shape)}")
        if expected_offset.ndim != 4 or tuple(expected_offset.shape) != tuple(current_offsets.shape):
            raise ValueError(
                f"`expected_offset` must match `current_offsets`, got {tuple(expected_offset.shape)} vs {tuple(current_offsets.shape)}"
            )
        if context_offsets.ndim != 5 or tuple(context_offsets.shape[:4]) != tuple(context_support_feat.shape[:4]) or int(context_offsets.shape[-1]) != 2:
            raise ValueError(
                f"`context_offsets` must be [B,T,P,{self.num_context_points},2], got {tuple(context_offsets.shape)}"
            )

        candidate_feat = torch.cat([main_support_feat.float().unsqueeze(3), context_support_feat.float()], dim=3)
        query_key = F.normalize(self.query_key_proj(query.float()), dim=-1, eps=1e-6)
        keys = F.normalize(self.key_proj(self.candidate_norm(candidate_feat)), dim=-1, eps=1e-6)
        values = self.value_proj(candidate_feat)
        corr = (keys * query_key.unsqueeze(3)).sum(dim=-1)
        probs = torch.softmax(corr * self.corr_temp, dim=-1)
        attended_feat = torch.einsum("btpk,btpkd->btpd", probs, values)
        mean_feat = candidate_feat.mean(dim=3)
        max_feat = candidate_feat.amax(dim=3)

        zero_offset = context_offsets.new_zeros((*context_offsets.shape[:3], 1, 2))
        candidate_offsets = torch.cat([zero_offset, context_offsets.float()], dim=3)
        context_shift = torch.einsum("btpk,btpkc->btpc", probs, candidate_offsets)

        topk_vals = torch.topk(corr, k=min(2, int(corr.shape[-1])), dim=-1).values
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1] if int(topk_vals.shape[-1]) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / float(
            torch.log(torch.tensor(float(candidate_feat.shape[3]), device=probs.device, dtype=probs.dtype)).item()
        )
        max_prob = probs.amax(dim=-1)

        merged = torch.cat(
            [
                main_feat.float(),
                attended_feat,
                mean_feat,
                max_feat,
                query.float(),
                current_offsets.float(),
                expected_offset.float(),
                context_shift,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        feat = main_feat.float() + self.out_proj(self.out_norm(merged))
        return {
            "feat": feat,
            "context_corr_scores": corr,
            "context_corr_probs": probs,
            "context_entropy": entropy,
            "context_margin": margin,
            "context_max_prob": max_prob,
            "context_shift": context_shift,
            "context_offset_abs": context_offsets.float().abs().mean(dim=(-1, -2)),
        }


class JointPointEvidenceBlock(nn.Module):
    """Fuse point evidence jointly so the point set is refined as a coupled structure."""

    def __init__(self, feat_dim: int, num_heads: int = 4, max_roles: int = 8) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_heads = int(num_heads)
        self.max_roles = int(max_roles)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}")
        if self.max_roles <= 0:
            raise ValueError(f"`max_roles` must be > 0, got {self.max_roles}")

        self.pair_proj = nn.Sequential(
            nn.Linear(3, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.token_norm = nn.LayerNorm((self.feat_dim * 3) + 6)
        self.token_proj = nn.Sequential(
            nn.Linear((self.feat_dim * 3) + 6, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.point_role_embedding = nn.Embedding(self.max_roles, self.feat_dim)
        nn.init.trunc_normal_(self.point_role_embedding.weight, std=0.02)
        self.set_token = nn.Parameter(torch.zeros(1, 1, 1, self.feat_dim))
        nn.init.trunc_normal_(self.set_token, std=0.02)

        self.space_ln1 = nn.LayerNorm(self.feat_dim)
        self.space_attn = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.space_ln2 = nn.LayerNorm(self.feat_dim)
        self.space_ffn = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2),
            nn.GELU(),
            nn.Linear(self.feat_dim * 2, self.feat_dim),
        )
        self.out_norm = nn.LayerNorm(self.feat_dim)
        self.out_proj = nn.Linear(self.feat_dim, self.feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        *,
        point_feat: torch.Tensor,
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        expected_offset: torch.Tensor,
        current_centroid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if point_feat.ndim != 4 or int(point_feat.shape[-1]) != self.feat_dim:
            raise ValueError(f"`point_feat` must be [B,T,P,D={self.feat_dim}], got {tuple(point_feat.shape)}")
        if current_xy.ndim != 4 or tuple(current_xy.shape[:3]) != tuple(point_feat.shape[:3]) or int(current_xy.shape[-1]) != 2:
            raise ValueError(f"`current_xy` must be [B,T,P,2], got {tuple(current_xy.shape)}")
        if current_offsets.ndim != 4 or tuple(current_offsets.shape) != tuple(current_xy.shape):
            raise ValueError(
                f"`current_offsets` must match `current_xy`, got {tuple(current_offsets.shape)} vs {tuple(current_xy.shape)}"
            )
        if expected_offset.ndim != 4 or tuple(expected_offset.shape) != tuple(current_xy.shape):
            raise ValueError(
                f"`expected_offset` must match `current_xy`, got {tuple(expected_offset.shape)} vs {tuple(current_xy.shape)}"
            )
        if current_centroid.ndim != 4 or tuple(current_centroid.shape[:3]) != (point_feat.shape[0], point_feat.shape[1], 1):
            raise ValueError(f"`current_centroid` must be [B,T,1,2], got {tuple(current_centroid.shape)}")

        bsz, t_len, n_pts, _ = point_feat.shape
        if n_pts > self.max_roles:
            raise ValueError(f"`point_feat` P={n_pts} exceeds `max_roles`={self.max_roles}")

        centroid_rel = current_xy.float() - current_centroid.float()
        pair_rel = current_xy.float().unsqueeze(3) - current_xy.float().unsqueeze(2)
        pair_dist = pair_rel.norm(dim=-1, keepdim=True)
        pair_feat = self.pair_proj(torch.cat([pair_rel, pair_dist], dim=-1))
        pair_mean = pair_feat.mean(dim=3)
        pair_max = pair_feat.amax(dim=3)

        token_in = torch.cat(
            [
                point_feat.float(),
                pair_mean,
                pair_max,
                centroid_rel,
                current_offsets.float(),
                expected_offset.float(),
            ],
            dim=-1,
        )
        point_tokens = self.token_proj(self.token_norm(token_in))
        role_ids = torch.arange(n_pts, device=point_tokens.device, dtype=torch.long).view(1, 1, n_pts)
        point_tokens = point_tokens + self.point_role_embedding(role_ids)
        set_token = point_tokens.mean(dim=2, keepdim=True) + self.set_token

        spatial_tokens = torch.cat([set_token, point_tokens], dim=2)
        spatial_tokens_bt = spatial_tokens.reshape(bsz * t_len, 1 + n_pts, self.feat_dim).contiguous()
        qkv_space = self.space_ln1(spatial_tokens_bt)
        space_out, _ = self.space_attn(qkv_space, qkv_space, qkv_space, need_weights=False)
        spatial_tokens_bt = spatial_tokens_bt + space_out
        spatial_tokens_bt = spatial_tokens_bt + self.space_ffn(self.space_ln2(spatial_tokens_bt))
        spatial_tokens = spatial_tokens_bt.reshape(bsz, t_len, 1 + n_pts, self.feat_dim).contiguous()
        point_tokens = spatial_tokens[:, :, 1:, :]
        delta_feat = self.out_proj(self.out_norm(point_tokens))
        return {
            "delta_feat": delta_feat,
            "pair_dist_mean": pair_dist.mean(dim=3).squeeze(-1),
        }


class LayerwiseTemplateCorrelationAdapter(nn.Module):
    """Template-conditioned local evidence adapter for late layerwise refine stages."""

    def __init__(
        self,
        feat_dim: int,
        *,
        num_layers: int = 4,
        support_radius: int = 1,
        corr_temp: float = 10.0,
        late_stage_count: int = 2,
        template_ema_alpha: float = 0.8,
        support_radii_by_stage: Optional[Tuple[int, ...]] = None,
        detach_current_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_layers = int(num_layers)
        self.support_radius = int(support_radius)
        self.corr_temp = float(corr_temp)
        self.late_stage_count = int(late_stage_count)
        self.template_ema_alpha = float(template_ema_alpha)
        self.support_radii_by_stage = tuple(int(v) for v in (support_radii_by_stage or ()))
        self.detach_current_inputs = bool(detach_current_inputs)
        if self.feat_dim <= 0:
            raise ValueError(f"`feat_dim` must be > 0, got {self.feat_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}")
        if self.support_radius < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {self.support_radius}")
        if self.corr_temp <= 0.0:
            raise ValueError(f"`corr_temp` must be > 0, got {self.corr_temp}")
        if self.late_stage_count <= 0:
            raise ValueError(f"`late_stage_count` must be > 0, got {self.late_stage_count}")
        if self.template_ema_alpha < 0.0 or self.template_ema_alpha >= 1.0:
            raise ValueError(
                "`template_ema_alpha` must be in [0,1), "
                f"got {self.template_ema_alpha}"
            )
        for radius in self.support_radii_by_stage:
            if int(radius) < 0:
                raise ValueError(
                    "`support_radii_by_stage` must contain non-negative integers, "
                    f"got {self.support_radii_by_stage}"
                )

        self.k_support = (2 * self.support_radius + 1) ** 2
        self.layer_embed = nn.Embedding(self.num_layers, self.feat_dim)
        nn.init.trunc_normal_(self.layer_embed.weight, std=0.02)

        self.template_norm = nn.LayerNorm(self.feat_dim)
        self.support_norm = nn.LayerNorm(self.feat_dim)
        self.query_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.key_proj = nn.Linear(self.feat_dim, self.feat_dim)

        merged_dim = (self.feat_dim * 6) + 12
        self.delta_norm = nn.LayerNorm(merged_dim)
        self.delta_proj = nn.Sequential(
            nn.Linear(merged_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        gate_hidden = max(self.feat_dim // 4, 32)
        self.point_gate_norm = nn.LayerNorm(6)
        self.point_gate = nn.Sequential(
            nn.Linear(6, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.stage_gate = nn.Parameter(torch.zeros(self.num_layers))

    def _support_rel_coords(self, support_feat: torch.Tensor) -> torch.Tensor:
        k_support = int(support_feat.shape[3])
        side = int(round(math.sqrt(float(k_support))))
        if side * side != k_support:
            raise ValueError(f"Support K must be a perfect square, got K={k_support}")
        support_radius = (side - 1) // 2
        coords = torch.arange(
            -support_radius,
            support_radius + 1,
            device=support_feat.device,
            dtype=support_feat.dtype,
        )
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        rel = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, 1, side * side, 2)
        scale = float(max(support_radius, 1))
        return rel / scale

    def support_radius_for_stage(self, *, stage_idx: int, num_stages: int) -> int:
        del num_stages
        idx = int(stage_idx) - 1
        if 0 <= idx < len(self.support_radii_by_stage):
            return int(self.support_radii_by_stage[idx])
        return int(self.support_radius)

    def should_apply(self, *, stage_idx: int, num_stages: int) -> bool:
        if num_stages <= 0:
            return False
        first_active = max(1, int(num_stages) - int(self.late_stage_count) + 1)
        return int(stage_idx) >= first_active

    def initialize_template(self, support_feat: torch.Tensor) -> torch.Tensor:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )
        center_idx = int(support_feat.shape[3]) // 2
        template_state = F.normalize(support_feat[:, 0, :, center_idx, :].float(), dim=-1, eps=1e-6)
        return template_state.detach()

    def update_template(
        self,
        *,
        template_state: torch.Tensor,
        support_feat: torch.Tensor,
        corr_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if template_state.ndim != 3 or int(template_state.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`template_state` must be [B,P,D={self.feat_dim}], got {tuple(template_state.shape)}"
            )
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        current_feat = support_feat.float()
        bsz, _, n_pts, k_support, feat_dim = current_feat.shape
        if tuple(template_state.shape) != (bsz, n_pts, feat_dim):
            raise ValueError(
                f"`template_state` must be [B,P,D]={bsz,n_pts,feat_dim}, got {tuple(template_state.shape)}"
            )
        center_idx = k_support // 2
        if corr_probs is None:
            update_source = current_feat[:, 0, :, center_idx, :]
        else:
            if corr_probs.ndim != 4 or tuple(corr_probs.shape[:3]) != (bsz, current_feat.shape[1], n_pts):
                raise ValueError(f"`corr_probs` must be [B,T,P,K], got {tuple(corr_probs.shape)}")
            update_source = torch.einsum("bpk,bpkd->bpd", corr_probs[:, 0, :, :].float(), current_feat[:, 0, :, :, :])
        alpha = float(self.template_ema_alpha)
        next_template = F.normalize(
            (alpha * template_state.float()) + ((1.0 - alpha) * update_source),
            dim=-1,
            eps=1e-6,
        )
        return next_template.detach()

    def forward(
        self,
        *,
        support_feat: torch.Tensor,
        main_feat: torch.Tensor,
        main_state: torch.Tensor,
        template_state: torch.Tensor,
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        expected_offset: torch.Tensor,
        layer_rank: int,
    ) -> Dict[str, torch.Tensor]:
        if support_feat.ndim != 5:
            raise ValueError(f"`support_feat` must be [B,T,P,K,D], got {tuple(support_feat.shape)}")
        if main_feat.ndim != 4 or int(main_feat.shape[-1]) != self.feat_dim:
            raise ValueError(f"`main_feat` must be [B,T,P,D={self.feat_dim}], got {tuple(main_feat.shape)}")
        if main_state.ndim != 4 or tuple(main_state.shape) != tuple(main_feat.shape):
            raise ValueError(f"`main_state` must match `main_feat`, got {tuple(main_state.shape)} vs {tuple(main_feat.shape)}")
        if template_state.ndim != 3 or int(template_state.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`template_state` must be [B,P,D={self.feat_dim}], got {tuple(template_state.shape)}"
            )
        if current_xy.ndim != 4 or tuple(current_xy.shape[:3]) != tuple(main_feat.shape[:3]) or int(current_xy.shape[-1]) != 2:
            raise ValueError(f"`current_xy` must be [B,T,P,2], got {tuple(current_xy.shape)}")
        if current_offsets.ndim != 4 or tuple(current_offsets.shape) != tuple(current_xy.shape):
            raise ValueError(
                f"`current_offsets` must match `current_xy`, got {tuple(current_offsets.shape)} vs {tuple(current_xy.shape)}"
            )
        if expected_offset.ndim != 4 or tuple(expected_offset.shape) != tuple(current_xy.shape):
            raise ValueError(
                f"`expected_offset` must match `current_xy`, got {tuple(expected_offset.shape)} vs {tuple(current_xy.shape)}"
            )
        if int(support_feat.shape[-1]) != self.feat_dim:
            raise ValueError(
                f"`support_feat` dim mismatch: expected {self.feat_dim}, got {int(support_feat.shape[-1])}"
            )

        current_feat = support_feat.float()
        base_feat = main_feat.float()
        main_feat_cond = main_feat.float()
        main_state_cond = main_state.float()
        current_xy_cond = current_xy.float()
        current_offsets_cond = current_offsets.float()
        expected_offset_cond = expected_offset.float()
        if self.detach_current_inputs:
            current_feat = current_feat.detach()
            main_feat_cond = main_feat_cond.detach()
            main_state_cond = main_state_cond.detach()
            current_xy_cond = current_xy_cond.detach()
            current_offsets_cond = current_offsets_cond.detach()
            expected_offset_cond = expected_offset_cond.detach()
        template_feat = template_state.float().detach()

        bsz, t_len, n_pts, k_support, feat_dim = current_feat.shape
        if tuple(template_feat.shape) != (bsz, n_pts, feat_dim):
            raise ValueError(
                f"`template_state` must be [B,P,D]={bsz,n_pts,feat_dim}, got {tuple(template_feat.shape)}"
            )
        center_idx = k_support // 2
        center_feat = current_feat[:, :, :, center_idx, :]

        layer_rank_clamped = int(max(0, min(self.num_layers - 1, int(layer_rank))))
        layer_bias = self.layer_embed.weight[layer_rank_clamped].view(1, 1, 1, self.feat_dim)
        template_query = F.normalize(
            self.query_proj(self.template_norm(template_feat[:, None, :, :] + layer_bias)),
            dim=-1,
            eps=1e-6,
        ).expand(-1, t_len, -1, -1)
        keys = F.normalize(self.key_proj(self.support_norm(current_feat)), dim=-1, eps=1e-6)
        corr = (keys * template_query.unsqueeze(3)).sum(dim=-1)
        probs = torch.softmax(corr * self.corr_temp, dim=-1)
        attended_feat = torch.einsum("btpk,btpkd->btpd", probs, current_feat)

        rel_coords = self._support_rel_coords(current_feat).expand(bsz, t_len, n_pts, -1, -1)
        matched_offset = torch.einsum("btpk,btpkc->btpc", probs, rel_coords)
        topk_vals = torch.topk(corr, k=min(2, k_support), dim=-1).values
        top1 = topk_vals[..., 0]
        top2 = topk_vals[..., 1] if int(topk_vals.shape[-1]) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) / float(math.log(float(max(k_support, 2))))
        max_prob = probs.amax(dim=-1)

        merged = torch.cat(
            [
                main_feat_cond,
                main_state_cond,
                center_feat,
                attended_feat,
                template_feat[:, None, :, :].expand(-1, t_len, -1, -1),
                layer_bias.expand(bsz, t_len, n_pts, -1),
                current_xy_cond,
                current_offsets_cond,
                expected_offset_cond,
                matched_offset,
                top1.unsqueeze(-1),
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
            ],
            dim=-1,
        )
        delta_feat = self.delta_proj(self.delta_norm(merged))

        gate_stats = torch.cat(
            [
                margin.unsqueeze(-1),
                entropy.unsqueeze(-1),
                max_prob.unsqueeze(-1),
                current_offsets_cond.norm(dim=-1, keepdim=True),
                expected_offset_cond.norm(dim=-1, keepdim=True),
                matched_offset.norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        point_gate = torch.sigmoid(self.point_gate(self.point_gate_norm(gate_stats)))
        stage_scale = torch.tanh(self.stage_gate[layer_rank_clamped]).to(dtype=delta_feat.dtype).view(1, 1, 1, 1)
        feat = base_feat + (stage_scale * point_gate * delta_feat)
        next_template_state = self.update_template(
            template_state=template_feat,
            support_feat=current_feat,
            corr_probs=probs,
        )
        return {
            "feat": feat,
            "next_template_state": next_template_state,
            "template_corr_scores": corr,
            "template_corr_probs": probs,
            "template_entropy": entropy,
            "template_margin": margin,
            "template_max_prob": max_prob,
            "template_gate": (stage_scale * point_gate).squeeze(-1),
            "template_offset_abs": matched_offset.abs().mean(dim=-1),
            "template_delta_abs": delta_feat.abs().mean(dim=-1),
        }


class IdentityPointRefiner(nn.Module):
    """No-op refiner used for no-refiner speed baseline."""

    def __init__(
        self,
        num_points: int,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.debug_shapes = bool(debug_shapes)

    def forward(
        self,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        local_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_xy.ndim != 4 or int(pred_xy.shape[-1]) != 2:
            raise ValueError(f"`pred_xy` must be [B,T,P,2], got {tuple(pred_xy.shape)}")
        if pred_conf.ndim != 3:
            raise ValueError(f"`pred_conf` must be [B,T,P], got {tuple(pred_conf.shape)}")
        if int(pred_xy.shape[2]) != self.num_points:
            raise ValueError(f"`pred_xy` P mismatch: expected {self.num_points}, got {int(pred_xy.shape[2])}")
        base_xy = pred_xy.float()
        conf = pred_conf.float().clamp(0.0, 1.0)
        delta_xy = torch.zeros_like(base_xy)
        shape_debug = None
        if self.debug_shapes:
            shape_debug = {
                "local_feat_in": [] if local_feat is None else list(local_feat.shape),
                "delta_xy_raw": list(delta_xy.shape),
                "delta_gate": list(conf.shape),
            }
        return {
            "pred_xy": base_xy,
            "pred_conf": conf,
            "temporal_xy": base_xy,
            "delta_xy": delta_xy,
            "delta_xy_raw": delta_xy,
            "delta_gate_base": torch.zeros_like(conf),
            "delta_gate": torch.zeros_like(conf),
            "shape_debug": shape_debug,
        }


class KeypointComboNet(nn.Module):
    """VJEPA-frozen multi-task model.

    Tasks:
      1) tool keypoints (3 points)
      2) contact keypoints (4 points)
      3) clip-level combo classification (6 classes)
      4) action range prediction (point actionness + offset regression)
    """

    def __init__(
        self,
        vjepa: VJEPAWrapper,
        proj_dim: int = 512,
        num_combo_classes: int = 6,
        num_tool_points: int = 3,
        num_contact_points: int = 4,
        output_num_frames: int = 16,
        num_range_bins: int = 10,
        temporal_upsample: str = "deconv",
        heatmap_softmax_temp: float = 1.0,
        enable_delta_branch: bool = True,
        temporal_mean_residual_alpha: float = 0.5,
        refiner_mode: str = "lite",
        trajectory_only: bool = False,
        range_only: bool = False,
        tool_only: bool = False,
        combo_probe_attn: bool = False,
        combo_probe_dual_attn: bool = False,
        combo_feature_mode: str = "tool_weighted",
        combo_head_dropout: float = 0.0,
        combo_feat_dropout: float = 0.0,
        dual_attn_tool_prior: bool = False,
        dual_attn_tool_prior_init_alpha: float = 0.0,
        dual_attn_tool_prior_detach: bool = True,
        num_tool_classes: int = 4,
        num_action_classes: int = 4,
        trajectory_target: str = "both",
        traj_num_subframes: int = 2,
        traj_condition_mode: str = "tool",
        range_condition_mode: str = "tool_action",
        range_head_mode: str = "scalar",
        range_trident_num_bins: int = 15,
        range_proposal_mode: str = "none",
        range_proposal_decode_mode: str = "argmax",
        range_quality_mode: str = "none",
        range_boundary_refine_mode: str = "none",
        range_boundary_refine_radius: int = 2,
        range_temporal_context_mode: str = "none",
        range_temporal_context_kernel_size: int = 3,
        range_temporal_context_layers: int = 1,
        range_temporal_context_levels: int = 3,
        range_spatial_pool_mode: str = "mean_max",
        range_spatial_pool_num_tokens: int = 8,
        range_adapter_mode: str = "none",
        range_adapter_kernel_size: int = 5,
        range_adapter_num_heads: int = 4,
        range_task_decouple_mode: str = "none",
        range_duration_prior_mode: str = "none",
        range_duration_prior_weight: float = 0.0,
        range_duration_prior_means: Optional[Sequence[float]] = None,
        range_duration_prior_stds: Optional[Sequence[float]] = None,
        range_duration_prior_min_std: float = 0.05,
        traj_refiner_mode: str = "none",
        traj_head_hidden_dim: Optional[int] = None,
        traj_head_up_layers: int = 2,
        traj_recurrent_max_delta_norm: float = 0.06,
        traj_dynamic_centroid_max_beta: float = 0.75,
        traj_context_num_points: int = 4,
        traj_context_offset_max_cells: float = 2.0,
        traj_context_support_radius: int = 1,
        traj_tool_centroid_offset_head: bool = False,
        traj_tool_offset_max_norm: float = 0.20,
        traj_centroid_local_softargmax_radius: int = 0,
        traj_subcell_readout_r: int = 0,
        traj_perpoint_local_readout: bool = False,
        traj_perpoint_local_radius: int = 2,
        traj_perpoint_gate_init: float = -3.0,
        traj_cond_film: bool = False,
        traj_conf_head: bool = False,
        traj_token_detector: bool = False,
        traj_token_det_hidden: int = 512,
        traj_perpoint_adaptive_gate: bool = False,
        traj_support_radius: int = 1,
        traj_temporal_query_mix: bool = False,
        traj_temporal_query_layers: int = 1,
        traj_temporal_query_spatial: bool = False,
        traj_temporal_query_interleave: bool = False,
        traj_num_virtual_queries: int = 0,
        traj_locality_prior: bool = False,
        traj_region_prior: bool = False,
        traj_pooled_context: bool = False,
        token_mask_enable: bool = False,
        token_mask_eval: bool = False,
        token_mask_strategy: str = "random",
        token_mask_value: str = "learned_pos",
        token_mask_p_max: float = 0.10,
        token_mask_p_early: Optional[float] = None,
        token_mask_delay_epochs: int = 5,
        token_mask_warmup_epochs: int = 20,
        token_mask_late_start_epoch: int = 25,
        token_mask_p_late: float = 0.15,
        token_mask_late_warmup_epochs: int = 40,
        token_mask_keep_k: int = 64,
        token_mask_mix_801010: bool = True,
        traj_stage0_centroid_noise_prob: float = 0.0,
        traj_stage0_centroid_noise_sigma: float = 0.0,
        debug_timing: bool = False,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.vjepa = vjepa
        self.vjepa_embed_dim = int(vjepa.embed_dim)
        self.output_canvas_size = int(getattr(vjepa, "img_size", 256))
        if self.output_canvas_size <= 0:
            self.output_canvas_size = 256
        self.num_combo_classes = int(num_combo_classes)
        self.num_tool_points = int(num_tool_points)
        self.num_contact_points = int(num_contact_points)
        self.output_num_frames = int(output_num_frames)
        self.num_range_bins = int(num_range_bins)
        self.temporal_upsample = str(temporal_upsample).lower()
        self.heatmap_softmax_temp = float(heatmap_softmax_temp)
        self.enable_delta_branch = bool(enable_delta_branch)
        self.temporal_mean_residual_alpha = float(temporal_mean_residual_alpha)
        self.trajectory_only = bool(trajectory_only)
        self.range_only = bool(range_only)
        self.tool_only = bool(tool_only)
        self.combo_probe_attn = bool(combo_probe_attn)
        self.combo_probe_dual_attn = bool(combo_probe_dual_attn)
        self.combo_feature_mode = str(combo_feature_mode).lower()
        self.combo_head_dropout = float(combo_head_dropout)
        self.combo_feat_dropout = float(combo_feat_dropout)
        self.dual_attn_tool_prior = bool(dual_attn_tool_prior)
        self.dual_attn_tool_prior_init_alpha = float(dual_attn_tool_prior_init_alpha)
        self.dual_attn_tool_prior_detach = bool(dual_attn_tool_prior_detach)
        self.num_tool_classes = int(num_tool_classes)
        self.num_action_classes = int(num_action_classes)
        self.trajectory_target = str(trajectory_target).lower()
        self.traj_num_subframes = int(traj_num_subframes)
        self.traj_condition_mode = str(traj_condition_mode).lower()
        self.range_condition_mode = str(range_condition_mode).lower()
        self.range_head_mode = str(range_head_mode).lower()
        self.range_trident_num_bins = int(range_trident_num_bins)
        self.range_proposal_mode = str(range_proposal_mode).lower()
        self.range_proposal_decode_mode = str(range_proposal_decode_mode).lower()
        self.range_quality_mode = str(range_quality_mode).lower()
        self.range_boundary_refine_mode = str(range_boundary_refine_mode).lower()
        self.range_boundary_refine_radius = int(range_boundary_refine_radius)
        self.range_temporal_context_mode = str(range_temporal_context_mode).lower()
        self.range_temporal_context_kernel_size = int(range_temporal_context_kernel_size)
        self.range_temporal_context_layers = int(range_temporal_context_layers)
        self.range_temporal_context_levels = int(range_temporal_context_levels)
        self.range_spatial_pool_mode = str(range_spatial_pool_mode).lower()
        self.range_spatial_pool_num_tokens = int(range_spatial_pool_num_tokens)
        self.range_adapter_mode = str(range_adapter_mode).lower()
        self.range_adapter_kernel_size = int(range_adapter_kernel_size)
        self.range_adapter_num_heads = int(range_adapter_num_heads)
        self.range_task_decouple_mode = str(range_task_decouple_mode).lower()
        self.range_duration_prior_mode = str(range_duration_prior_mode).lower()
        self.range_duration_prior_weight = float(range_duration_prior_weight)
        self.range_duration_prior_min_std = float(range_duration_prior_min_std)
        self.traj_refiner_mode = str(traj_refiner_mode).lower() if self.trajectory_only else "none"
        self.refiner_mode = "none" if self.trajectory_only else str(refiner_mode).lower()
        self.traj_head_hidden_dim = int(proj_dim if traj_head_hidden_dim is None else traj_head_hidden_dim)
        self.traj_head_up_layers = int(traj_head_up_layers)
        self.traj_recurrent_max_delta_norm = float(traj_recurrent_max_delta_norm)
        self.traj_dynamic_centroid_max_beta = float(traj_dynamic_centroid_max_beta)
        self.traj_context_num_points = int(traj_context_num_points)
        self.traj_context_offset_max_cells = float(traj_context_offset_max_cells)
        self.traj_context_support_radius = int(traj_context_support_radius)
        self.traj_context_late_stage_count = 0
        self.traj_joint_late_stage_count = 0
        self.traj_template_support_radius = 1
        self.traj_template_support_radii_by_stage: Tuple[int, ...] = ()
        self.traj_template_late_stage_count = 2
        self.traj_template_ema_alpha = 0.8
        self.traj_template_detach_current_inputs = True
        if self.traj_refiner_mode == "s3p6_singlelayer_statecarry_lategated_deformablesupport":
            self.traj_context_late_stage_count = 2
        if self.traj_refiner_mode == "s3p6a_singlelayer_statecarry_pointgated_deformablesupport":
            self.traj_context_late_stage_count = 2
            self.traj_context_offset_max_cells = 1.25
        if self.traj_refiner_mode == "s3p7_singlelayer_statecarry_jointpoint_deformablesupport":
            self.traj_context_late_stage_count = 2
            self.traj_joint_late_stage_count = 2
            self.traj_context_offset_max_cells = 1.25
        if self.traj_refiner_mode == "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr":
            self.traj_template_support_radii_by_stage = (1, 2, 1, 1)
            self.traj_template_late_stage_count = 3
            self.traj_template_detach_current_inputs = False
        self.traj_tool_centroid_offset_head = bool(traj_tool_centroid_offset_head)
        self.traj_tool_offset_max_norm = float(traj_tool_offset_max_norm)
        self.traj_centroid_local_softargmax_radius = int(traj_centroid_local_softargmax_radius)
        self.traj_subcell_readout_r = int(traj_subcell_readout_r)
        self.traj_perpoint_local_readout = bool(traj_perpoint_local_readout)
        self.traj_perpoint_local_radius = int(traj_perpoint_local_radius)
        self.traj_perpoint_gate_init = float(traj_perpoint_gate_init)
        self.traj_cond_film = bool(traj_cond_film)
        self.traj_conf_head = bool(traj_conf_head)
        self.traj_token_detector = bool(traj_token_detector)
        self.traj_token_det_hidden = int(traj_token_det_hidden)
        self.traj_perpoint_adaptive_gate = bool(traj_perpoint_adaptive_gate)
        # E008: Chebyshev radius of the state-carry refiner's local support window (1 = 3x3 tokens, E002 default)
        self.traj_support_radius = int(traj_support_radius)
        self.traj_temporal_query_mix = bool(traj_temporal_query_mix)
        self.traj_temporal_query_layers = int(traj_temporal_query_layers)
        self.traj_temporal_query_spatial = bool(traj_temporal_query_spatial)
        self.traj_temporal_query_interleave = bool(traj_temporal_query_interleave)
        self.traj_num_virtual_queries = int(traj_num_virtual_queries)
        self.traj_locality_prior = bool(traj_locality_prior)
        self.traj_region_prior = bool(traj_region_prior)
        self.traj_pooled_context = bool(traj_pooled_context)
        self.token_mask_enable = bool(token_mask_enable)
        self.token_mask_eval = bool(token_mask_eval)
        self.token_mask_strategy = str(token_mask_strategy).lower()
        self.token_mask_value = str(token_mask_value).lower()
        self.token_mask_p_max = float(token_mask_p_max)
        self.token_mask_p_early = float(self.token_mask_p_max if token_mask_p_early is None else token_mask_p_early)
        self.token_mask_delay_epochs = int(token_mask_delay_epochs)
        self.token_mask_warmup_epochs = int(token_mask_warmup_epochs)
        self.token_mask_late_start_epoch = int(token_mask_late_start_epoch)
        self.token_mask_p_late = float(token_mask_p_late)
        self.token_mask_late_warmup_epochs = int(token_mask_late_warmup_epochs)
        self.token_mask_keep_k = int(token_mask_keep_k)
        self.token_mask_mix_801010 = bool(token_mask_mix_801010)
        self.traj_stage0_centroid_noise_prob = float(traj_stage0_centroid_noise_prob)
        self.traj_stage0_centroid_noise_sigma = float(traj_stage0_centroid_noise_sigma)
        self.token_mask_epoch = 0
        self.anchor_schedule_epoch = 0
        self.anchor_schedule_delay_epochs = int(S3P9E_ANCHOR_DELAY_EPOCHS)
        self.anchor_schedule_ramp_epochs = int(S3P9E_ANCHOR_RAMP_EPOCHS)
        self.debug_timing = bool(debug_timing)
        self.debug_shapes = bool(debug_shapes)
        if self.temporal_upsample not in ("interp", "deconv"):
            raise ValueError(f"Unsupported temporal_upsample={temporal_upsample}, expected 'interp' or 'deconv'.")
        if self.heatmap_softmax_temp <= 0.0:
            raise ValueError(f"`heatmap_softmax_temp` must be > 0, got {self.heatmap_softmax_temp}")
        if self.refiner_mode not in ("none", "lite", "attn"):
            raise ValueError(f"Unsupported refiner_mode={refiner_mode}, expected one of: none,lite,attn")
        if self.trajectory_only and self.range_only:
            raise ValueError("`trajectory_only` and `range_only` are mutually exclusive.")
        if self.trajectory_only and self.tool_only:
            raise ValueError("`trajectory_only` and `tool_only` are mutually exclusive.")
        if self.trajectory_only and self.combo_probe_attn:
            raise ValueError("`trajectory_only` and `combo_probe_attn` are mutually exclusive.")
        if self.trajectory_only and self.combo_probe_dual_attn:
            raise ValueError("`trajectory_only` and `combo_probe_dual_attn` are mutually exclusive.")
        if self.range_only and self.tool_only:
            raise ValueError("`range_only` and `tool_only` are mutually exclusive.")
        if self.range_only and self.combo_probe_attn:
            raise ValueError("`range_only` and `combo_probe_attn` are mutually exclusive.")
        if self.range_only and self.combo_probe_dual_attn:
            raise ValueError("`range_only` and `combo_probe_dual_attn` are mutually exclusive.")
        if self.tool_only and self.combo_probe_attn:
            raise ValueError("`tool_only` and `combo_probe_attn` are mutually exclusive.")
        if self.tool_only and self.combo_probe_dual_attn:
            raise ValueError("`tool_only` and `combo_probe_dual_attn` are mutually exclusive.")
        if self.combo_probe_attn and self.combo_probe_dual_attn:
            raise ValueError("`combo_probe_attn` and `combo_probe_dual_attn` are mutually exclusive.")
        if self.dual_attn_tool_prior and (not self.combo_probe_dual_attn):
            raise ValueError("`dual_attn_tool_prior` requires `combo_probe_dual_attn=True`.")
        if self.combo_feature_mode not in ("tool_weighted", "direct_spatiotemporal"):
            raise ValueError(
                f"Unsupported combo_feature_mode={combo_feature_mode}, "
                "expected one of: tool_weighted,direct_spatiotemporal"
            )
        if self.combo_head_dropout < 0.0 or self.combo_head_dropout >= 1.0:
            raise ValueError(f"`combo_head_dropout` must be in [0,1), got {self.combo_head_dropout}")
        if self.combo_feat_dropout < 0.0 or self.combo_feat_dropout >= 1.0:
            raise ValueError(f"`combo_feat_dropout` must be in [0,1), got {self.combo_feat_dropout}")
        if self.traj_stage0_centroid_noise_prob < 0.0 or self.traj_stage0_centroid_noise_prob > 1.0:
            raise ValueError(
                f"`traj_stage0_centroid_noise_prob` must be in [0,1], got {self.traj_stage0_centroid_noise_prob}"
            )
        if self.traj_stage0_centroid_noise_sigma < 0.0:
            raise ValueError(
                f"`traj_stage0_centroid_noise_sigma` must be >= 0, got {self.traj_stage0_centroid_noise_sigma}"
            )
        if self.num_tool_classes <= 0:
            raise ValueError(f"`num_tool_classes` must be > 0, got {self.num_tool_classes}")
        if self.num_action_classes <= 0:
            raise ValueError(f"`num_action_classes` must be > 0, got {self.num_action_classes}")
        prior_means_arr = torch.zeros((self.num_combo_classes,), dtype=torch.float32)
        prior_stds_arr = torch.ones((self.num_combo_classes,), dtype=torch.float32)
        if range_duration_prior_means is not None:
            vals = torch.as_tensor(list(range_duration_prior_means), dtype=torch.float32).reshape(-1)
            if int(vals.numel()) != int(self.num_combo_classes):
                raise ValueError(
                    f"`range_duration_prior_means` must have {self.num_combo_classes} values, got {int(vals.numel())}"
                )
            prior_means_arr = vals.clone()
        if range_duration_prior_stds is not None:
            vals = torch.as_tensor(list(range_duration_prior_stds), dtype=torch.float32).reshape(-1)
            if int(vals.numel()) != int(self.num_combo_classes):
                raise ValueError(
                    f"`range_duration_prior_stds` must have {self.num_combo_classes} values, got {int(vals.numel())}"
                )
            prior_stds_arr = vals.clone().clamp_min(self.range_duration_prior_min_std)
        self.register_buffer("range_duration_prior_means", prior_means_arr, persistent=True)
        self.register_buffer("range_duration_prior_stds", prior_stds_arr, persistent=True)
        if self.trajectory_target not in ("both", "tool", "ct"):
            raise ValueError(
                f"Unsupported trajectory_target={trajectory_target}, expected one of: both,tool,ct"
            )
        if self.traj_num_subframes <= 0:
            raise ValueError(f"`traj_num_subframes` must be > 0, got {self.traj_num_subframes}")
        if self.traj_condition_mode not in ("tool", "tool_action", "combo"):
            raise ValueError(
                f"Unsupported traj_condition_mode={traj_condition_mode}, expected one of: tool,tool_action,combo"
            )
        if self.range_condition_mode not in ("tool", "tool_action", "combo"):
            raise ValueError(
                f"Unsupported range_condition_mode={range_condition_mode}, expected one of: tool,tool_action,combo"
            )
        if self.range_head_mode not in ("scalar", "trident"):
            raise ValueError(
                f"Unsupported range_head_mode={range_head_mode}, expected one of: scalar,trident"
            )
        if self.range_proposal_mode not in ("none", "bmn"):
            raise ValueError(
                f"Unsupported range_proposal_mode={range_proposal_mode}, expected one of: none,bmn"
            )
        if self.range_proposal_decode_mode not in ("argmax", "none"):
            raise ValueError(
                "Unsupported range_proposal_decode_mode="
                f"{range_proposal_decode_mode}, expected one of: argmax,none"
            )
        if self.range_quality_mode not in ("none", "dgqp"):
            raise ValueError(
                f"Unsupported range_quality_mode={range_quality_mode}, expected one of: none,dgqp"
            )
        if self.range_boundary_refine_mode not in ("none", "local_argmax"):
            raise ValueError(
                "Unsupported range_boundary_refine_mode="
                f"{range_boundary_refine_mode}, expected one of: none,local_argmax"
            )
        if self.range_boundary_refine_radius <= 0:
            raise ValueError(
                f"`range_boundary_refine_radius` must be > 0, got {self.range_boundary_refine_radius}"
            )
        if self.range_temporal_context_mode not in ("none", "temporalmaxer", "fpn"):
            raise ValueError(
                "Unsupported range_temporal_context_mode="
                f"{range_temporal_context_mode}, expected one of: none,temporalmaxer,fpn"
            )
        if self.range_temporal_context_kernel_size <= 0 or (self.range_temporal_context_kernel_size % 2) != 1:
            raise ValueError(
                "`range_temporal_context_kernel_size` must be a positive odd integer, "
                f"got {self.range_temporal_context_kernel_size}"
            )
        if self.range_temporal_context_layers <= 0:
            raise ValueError(
                "`range_temporal_context_layers` must be > 0, "
                f"got {self.range_temporal_context_layers}"
            )
        if self.range_temporal_context_levels <= 0:
            raise ValueError(
                "`range_temporal_context_levels` must be > 0, "
                f"got {self.range_temporal_context_levels}"
            )
        if self.range_spatial_pool_mode not in ("mean_max", "mean_max_condtoken", "mean_max_condtoken_locresid"):
            raise ValueError(
                "Unsupported range_spatial_pool_mode="
                f"{range_spatial_pool_mode}, expected one of: mean_max,mean_max_condtoken,mean_max_condtoken_locresid"
            )
        if self.range_spatial_pool_num_tokens <= 0:
            raise ValueError(
                "`range_spatial_pool_num_tokens` must be > 0, "
                f"got {self.range_spatial_pool_num_tokens}"
            )
        if self.range_adapter_mode not in (
            "none",
            "sola",
            "losa",
            "dyfa",
            "sola_dyfa",
            "sola_dyfa_livegate",
            "sola_dyfa_laf",
        ):
            raise ValueError(
                "Unsupported range_adapter_mode="
                f"{range_adapter_mode}, expected one of: none,sola,losa,dyfa,sola_dyfa,sola_dyfa_livegate,sola_dyfa_laf"
            )
        if self.range_adapter_kernel_size <= 0 or (self.range_adapter_kernel_size % 2) != 1:
            raise ValueError(
                "`range_adapter_kernel_size` must be a positive odd integer, "
                f"got {self.range_adapter_kernel_size}"
            )
        if self.range_adapter_num_heads <= 0:
            raise ValueError(
                "`range_adapter_num_heads` must be > 0, "
                f"got {self.range_adapter_num_heads}"
            )
        if self.range_task_decouple_mode not in ("none", "score_loc", "score_loc_align"):
            raise ValueError(
                "Unsupported range_task_decouple_mode="
                f"{range_task_decouple_mode}, expected one of: none,score_loc,score_loc_align"
            )
        if self.range_duration_prior_mode not in ("none", "combo_gaussian"):
            raise ValueError(
                "Unsupported range_duration_prior_mode="
                f"{range_duration_prior_mode}, expected one of: none,combo_gaussian"
            )
        if self.range_duration_prior_weight < 0.0:
            raise ValueError("`range_duration_prior_weight` must be >= 0")
        if self.range_duration_prior_min_std <= 0.0:
            raise ValueError("`range_duration_prior_min_std` must be > 0")
        if self.range_trident_num_bins <= 0:
            raise ValueError(
                f"`range_trident_num_bins` must be > 0, got {self.range_trident_num_bins}"
            )
        if self.traj_refiner_mode not in (
            "none",
            "lite",
            "attn",
            "s1_offset",
            "s2_iter_offset",
            "s3_support3x3",
            "s3p2x_finalrefine_l18",
            "s3p3_singlelayer_statecarry_refine",
            "s3p3x_singlelayer_statecarry_refine_plus1",
            "s3p8_singlelayer_statecarry_fourierbelief",
            "s3p8a_singlelayer_statecarry_residualfourierbelief",
            "s3p9_singlelayer_statecarry_anchoronly",
            "s3p9d_singlelayer_statecarry_anchoronly_detached",
            "s3p9e_singlelayer_statecarry_anchoronly_ramped",
            "s3p9f_singlelayer_statecarry_anchorverifier_l18",
            "s3p10_singlelayer_statecarry_factorizedtoolresidual",
            "s3p4_singlelayer_statecarry_dynamiccentroid",
            "s3p5_singlelayer_statecarry_deformablesupport",
            "s3p6_singlelayer_statecarry_lategated_deformablesupport",
            "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
            "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
            "s4a_template_centercorr_3x3",
            "s4b_template_allpaircorr_3x3",
            "m1_progressive_multilayer_movingrefine",
            "m1_strict_s3_multilayer_mean",
            "m2_layerwise_statecarry_refine",
            "m3_layerwise_statecarry_dynamiccentroid",
            "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
            "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
            "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
        ):
            raise ValueError(
                "Unsupported traj_refiner_mode="
                f"{traj_refiner_mode}, expected one of: "
                "none,lite,attn,s1_offset,s2_iter_offset,s3_support3x3,"
                "s3p2x_finalrefine_l18,"
                "s3p3_singlelayer_statecarry_refine,"
                "s3p3x_singlelayer_statecarry_refine_plus1,"
                "s3p8_singlelayer_statecarry_fourierbelief,"
                "s3p8a_singlelayer_statecarry_residualfourierbelief,"
                "s3p9_singlelayer_statecarry_anchoronly,"
                "s3p9d_singlelayer_statecarry_anchoronly_detached,"
                "s3p9e_singlelayer_statecarry_anchoronly_ramped,"
                "s3p9f_singlelayer_statecarry_anchorverifier_l18,"
                "s3p10_singlelayer_statecarry_factorizedtoolresidual,"
                "s3p4_singlelayer_statecarry_dynamiccentroid,"
                "s3p5_singlelayer_statecarry_deformablesupport,"
                "s3p6_singlelayer_statecarry_lategated_deformablesupport,"
                "s3p6a_singlelayer_statecarry_pointgated_deformablesupport,"
                "s3p7_singlelayer_statecarry_jointpoint_deformablesupport,"
                "s4a_template_centercorr_3x3,s4b_template_allpaircorr_3x3,"
                "m1_progressive_multilayer_movingrefine,"
                "m1_strict_s3_multilayer_mean,"
                "m2_layerwise_statecarry_refine,"
                "m3_layerwise_statecarry_dynamiccentroid,"
                "m4_layerwise_statecarry_dynamiccentroid_contextpoints,"
                "m5_layerwise_statecarry_dynamiccentroid_templatecorr,"
                "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr"
            )
        self.requires_multilayer_vjepa_tokens = bool(
            self.trajectory_only
            and self.traj_refiner_mode in MULTILAYER_TRAJ_REFINER_MODES
        )
        if self.requires_multilayer_vjepa_tokens and len(getattr(self.vjepa, "layer_indices", [])) < 2:
            raise ValueError(
                "Multi-layer trajectory refiner modes require at least 2 selected VJEPA layers via `layer_indices`."
            )
        if (
            self.trajectory_only
            and self.traj_refiner_mode in STRICT_S3_CROSSLAYER_REFINER_MODES
            and list(getattr(self.vjepa, "layer_indices", [])) != [18, 24]
        ):
            raise ValueError(
                "`s3p2x_finalrefine_l18` requires exactly `layer_indices=[18,24]` "
                f"(got {list(getattr(self.vjepa, 'layer_indices', []))})."
            )
        if (
            self.trajectory_only
            and self.traj_refiner_mode in SHALLOW_VERIFIER_REFINER_MODES
            and list(getattr(self.vjepa, "layer_indices", [])) != [18, 24]
        ):
            raise ValueError(
                "`s3p9f_singlelayer_statecarry_anchorverifier_l18` requires exactly "
                f"`layer_indices=[18,24]` (got {list(getattr(self.vjepa, 'layer_indices', []))})."
            )
        if (
            self.trajectory_only
            and self.traj_refiner_mode in LAYERWISE_STATECARRY_REFINER_MODES
            and len(getattr(self.vjepa, "layer_indices", [])) < 4
        ):
            raise ValueError("Layerwise state-carry refine modes require at least 4 selected VJEPA layers.")
        if self.traj_head_hidden_dim <= 0:
            raise ValueError(f"`traj_head_hidden_dim` must be > 0, got {self.traj_head_hidden_dim}")
        if self.traj_head_up_layers <= 0:
            raise ValueError(f"`traj_head_up_layers` must be > 0, got {self.traj_head_up_layers}")
        if self.traj_recurrent_max_delta_norm <= 0.0:
            raise ValueError(
                f"`traj_recurrent_max_delta_norm` must be > 0, got {self.traj_recurrent_max_delta_norm}"
            )
        if self.traj_dynamic_centroid_max_beta < 0.0 or self.traj_dynamic_centroid_max_beta > 1.0:
            raise ValueError(
                "`traj_dynamic_centroid_max_beta` must be in [0,1], "
                f"got {self.traj_dynamic_centroid_max_beta}"
            )
        if self.traj_context_num_points <= 0:
            raise ValueError(f"`traj_context_num_points` must be > 0, got {self.traj_context_num_points}")
        if self.traj_context_offset_max_cells <= 0.0:
            raise ValueError(
                "`traj_context_offset_max_cells` must be > 0, "
                f"got {self.traj_context_offset_max_cells}"
            )
        if self.traj_context_support_radius < 0:
            raise ValueError(
                "`traj_context_support_radius` must be >= 0, "
                f"got {self.traj_context_support_radius}"
            )
        self.traj_support_projector = (
            LocalSupportPoolProjector(feat_dim=int(self.vjepa_embed_dim))
            if (
                self.trajectory_only
                and self.traj_refiner_mode in (
                    "s3_support3x3",
                    "s3p2x_finalrefine_l18",
                    "m1_strict_s3_multilayer_mean",
                    "s3p3_singlelayer_statecarry_refine",
                    "s3p3x_singlelayer_statecarry_refine_plus1",
                    "s3p8_singlelayer_statecarry_fourierbelief",
                    "s3p8a_singlelayer_statecarry_residualfourierbelief",
                    "s3p9_singlelayer_statecarry_anchoronly",
                    "s3p9d_singlelayer_statecarry_anchoronly_detached",
                    "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                    "s3p10_singlelayer_statecarry_factorizedtoolresidual",
                    "s3p4_singlelayer_statecarry_dynamiccentroid",
                    "s3p5_singlelayer_statecarry_deformablesupport",
                    "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                    "m2_layerwise_statecarry_refine",
                    "m3_layerwise_statecarry_dynamiccentroid",
                    "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
                    "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
                    "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
                )
            )
            else None
        )
        self.traj_support_corr_projector = (
            TemplateCenterCorrelationProjector(
                feat_dim=int(self.vjepa_embed_dim),
                support_radius=1,
            )
            if (self.trajectory_only and self.traj_refiner_mode == "s4a_template_centercorr_3x3")
            else None
        )
        self.traj_support_allpair_corr_projector = (
            TemplateAllPairCorrelationProjector(
                feat_dim=int(self.vjepa_embed_dim),
                support_radius=1,
            )
            if (self.trajectory_only and self.traj_refiner_mode == "s4b_template_allpaircorr_3x3")
            else None
        )
        self.traj_multilayer_stage1_projector = (
            MultiLayerStagePairSupportProjector(
                feat_dim=int(self.vjepa_embed_dim),
                num_layers=2,
            )
            if (self.trajectory_only and self.traj_refiner_mode == "m1_progressive_multilayer_movingrefine")
            else None
        )
        self.traj_multilayer_stage2_projector = (
            MultiLayerStagePairSupportProjector(
                feat_dim=int(self.vjepa_embed_dim),
                num_layers=2,
            )
            if (self.trajectory_only and self.traj_refiner_mode == "m1_progressive_multilayer_movingrefine")
            else None
        )
        self.traj_statecarry_projector = (
            (
                FourierBeliefStateCarrySupportProjector(
                    feat_dim=int(self.vjepa_embed_dim),
                    num_layers=max(4, len(getattr(self.vjepa, "layer_indices", []))),
                    support_radius=int(self.traj_support_radius),
                )
                if self.traj_refiner_mode == "s3p8_singlelayer_statecarry_fourierbelief"
                else ResidualFourierBeliefAdapterSupportProjector(
                    feat_dim=int(self.vjepa_embed_dim),
                    num_layers=max(4, len(getattr(self.vjepa, "layer_indices", []))),
                    support_radius=int(self.traj_support_radius),
                )
                if self.traj_refiner_mode in (
                    "s3p8a_singlelayer_statecarry_residualfourierbelief",
                    "s3p9_singlelayer_statecarry_anchoronly",
                    "s3p9d_singlelayer_statecarry_anchoronly_detached",
                    "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                    "s3p10_singlelayer_statecarry_factorizedtoolresidual",
                )
                else LayerwiseStateCarrySupportProjector(
                    feat_dim=int(self.vjepa_embed_dim),
                    num_layers=max(4, len(getattr(self.vjepa, "layer_indices", []))),
                    support_radius=int(self.traj_support_radius),
                )
            )
            if (self.trajectory_only and self.traj_refiner_mode in STATECARRY_REFINER_MODES)
            else None
        )
        self.traj_centroid_anchor_adapter = (
            ResidualCentroidAnchorAdapter(
                feat_dim=int(self.vjepa_embed_dim),
            )
            if (
                self.trajectory_only
                and self.traj_refiner_mode in (
                    "s3p9_singlelayer_statecarry_anchoronly",
                    "s3p9d_singlelayer_statecarry_anchoronly_detached",
                    "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                )
            )
            else None
        )
        self.traj_shallow_anchor_verifier = (
            ShallowAnchorMotionVerifier()
            if (
                self.trajectory_only
                and self.traj_refiner_mode in SHALLOW_VERIFIER_REFINER_MODES
            )
            else None
        )
        self.traj_factorized_tool_adapter = (
            FactorizedToolAwareResidualAdapter(
                feat_dim=int(self.vjepa_embed_dim),
                tool_cond_dim=int(self.vjepa_embed_dim),
            )
            if (
                self.trajectory_only
                and self.traj_refiner_mode == "s3p10_singlelayer_statecarry_factorizedtoolresidual"
            )
            else None
        )
        self.traj_context_point_adapter = (
            LayerwiseContextPointEvidenceAdapter(
                feat_dim=int(self.vjepa_embed_dim),
                num_context_points=int(self.traj_context_num_points),
                max_offset_cells=float(self.traj_context_offset_max_cells),
            )
            if (
                self.trajectory_only
                and self.traj_refiner_mode in (
                    "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
                    "s3p5_singlelayer_statecarry_deformablesupport",
                    "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                )
            )
            else None
        )
        self.traj_context_stage_gate = (
            nn.Parameter(torch.zeros(4))
            if (
                self.trajectory_only
                and self.traj_refiner_mode in (
                    "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                )
            )
            else None
        )
        if self.traj_context_stage_gate is not None and self.traj_refiner_mode in (
            "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
            "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
        ):
            with torch.no_grad():
                self.traj_context_stage_gate.zero_()
                self.traj_context_stage_gate[-int(self.traj_context_late_stage_count):].fill_(math.atanh(0.1))
        self.traj_context_point_gate = (
            nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            if (
                self.trajectory_only
                and self.traj_refiner_mode in (
                    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                )
            )
            else None
        )
        if self.traj_context_point_gate is not None:
            final_linear = self.traj_context_point_gate[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
        self.traj_joint_point_block = (
            JointPointEvidenceBlock(
                feat_dim=int(self.vjepa_embed_dim),
                num_heads=4,
            )
            if (
                self.trajectory_only
                and self.traj_refiner_mode == "s3p7_singlelayer_statecarry_jointpoint_deformablesupport"
            )
            else None
        )
        self.traj_joint_stage_gate = (
            nn.Parameter(torch.zeros(4))
            if (
                self.trajectory_only
                and self.traj_refiner_mode == "s3p7_singlelayer_statecarry_jointpoint_deformablesupport"
            )
            else None
        )
        if self.traj_joint_stage_gate is not None:
            with torch.no_grad():
                self.traj_joint_stage_gate.zero_()
                self.traj_joint_stage_gate[-int(self.traj_joint_late_stage_count):].fill_(math.atanh(0.1))
        self.traj_template_corr_adapter = (
            LayerwiseTemplateCorrelationAdapter(
                feat_dim=int(self.vjepa_embed_dim),
                num_layers=max(4, len(getattr(self.vjepa, "layer_indices", []))),
                support_radius=int(self.traj_template_support_radius),
                late_stage_count=int(self.traj_template_late_stage_count),
                template_ema_alpha=float(self.traj_template_ema_alpha),
                support_radii_by_stage=self.traj_template_support_radii_by_stage,
                detach_current_inputs=bool(self.traj_template_detach_current_inputs),
            )
            if (
                self.trajectory_only
                and self.traj_refiner_mode in TEMPLATE_CORR_REFINER_MODES
            )
            else None
        )
        if self.traj_tool_offset_max_norm <= 0.0:
            raise ValueError(
                f"`traj_tool_offset_max_norm` must be > 0, got {self.traj_tool_offset_max_norm}"
            )
        if self.traj_tool_centroid_offset_head and (not self.trajectory_only):
            raise ValueError("`traj_tool_centroid_offset_head` requires `trajectory_only=True`.")
        if self.traj_tool_centroid_offset_head and self.trajectory_target not in ("tool", "both"):
            raise ValueError("`traj_tool_centroid_offset_head` requires trajectory_target in {tool,both}.")
        if self.traj_tool_centroid_offset_head and self.num_tool_points <= 1:
            raise ValueError(
                "`traj_tool_centroid_offset_head` requires `num_tool_points` > 1 "
                f"(got {self.num_tool_points})."
            )
        if self.token_mask_strategy not in ("random", "tube"):
            raise ValueError(f"`token_mask_strategy` must be one of {{random,tube}}, got {self.token_mask_strategy}")
        if self.token_mask_value not in ("learned_pos", "zero"):
            raise ValueError(
                f"`token_mask_value` must be one of {{learned_pos,zero}}, got {self.token_mask_value}"
            )
        if self.token_mask_p_max < 0.0 or self.token_mask_p_max >= 1.0:
            raise ValueError(f"`token_mask_p_max` must be in [0,1), got {self.token_mask_p_max}")
        if self.token_mask_p_early < 0.0 or self.token_mask_p_early >= 1.0:
            raise ValueError(f"`token_mask_p_early` must be in [0,1), got {self.token_mask_p_early}")
        if self.token_mask_delay_epochs < 0:
            raise ValueError(f"`token_mask_delay_epochs` must be >= 0, got {self.token_mask_delay_epochs}")
        if self.token_mask_warmup_epochs < 0:
            raise ValueError(f"`token_mask_warmup_epochs` must be >= 0, got {self.token_mask_warmup_epochs}")
        if self.token_mask_late_start_epoch < 0:
            raise ValueError(f"`token_mask_late_start_epoch` must be >= 0, got {self.token_mask_late_start_epoch}")
        if self.token_mask_p_late < 0.0 or self.token_mask_p_late >= 1.0:
            raise ValueError(f"`token_mask_p_late` must be in [0,1), got {self.token_mask_p_late}")
        if self.token_mask_late_warmup_epochs < 0:
            raise ValueError(
                f"`token_mask_late_warmup_epochs` must be >= 0, got {self.token_mask_late_warmup_epochs}"
            )
        if self.token_mask_keep_k <= 0:
            raise ValueError(f"`token_mask_keep_k` must be > 0, got {self.token_mask_keep_k}")
        if (self.token_mask_enable or self.token_mask_eval) and (not self.trajectory_only):
            raise ValueError("token mask options currently require `trajectory_only=True`.")

        self.shared_trunk = SharedSpatialTrunk(in_dim=self.vjepa_embed_dim, proj_dim=proj_dim)
        self.conv_refine = SharedConvRefine(channels=proj_dim)
        self.temporal_mean_residual = TemporalMeanResidual(
            channels=proj_dim,
            alpha=self.temporal_mean_residual_alpha,
        )
        self.combo_attn_pool = None
        self.combo_tool_spatial_pool = None
        self.combo_action_temporal_pool = None
        self.combo_action_tool_prior = None
        self.range_temporal_pool = None
        self.range_temporal_context = None
        self.range_spatial_pool = None
        self.range_spatial_loc_residual = None
        self.range_temporal_adapter = None
        if self.token_mask_enable or self.token_mask_eval:
            self.kv_token_masker = KVTokenMaskAugmenter(
                in_dim=self.vjepa_embed_dim,
                enable=self.token_mask_enable,
                eval_enable=self.token_mask_eval,
                strategy=self.token_mask_strategy,
                value_mode=self.token_mask_value,
                p_max=self.token_mask_p_max,
                p_early=self.token_mask_p_early,
                delay_epochs=self.token_mask_delay_epochs,
                warmup_epochs=self.token_mask_warmup_epochs,
                late_start_epoch=self.token_mask_late_start_epoch,
                p_late=self.token_mask_p_late,
                late_warmup_epochs=self.token_mask_late_warmup_epochs,
                keep_k=self.token_mask_keep_k,
                mix_801010=self.token_mask_mix_801010,
            )
            self.kv_token_masker.set_runtime_epoch(0)
        else:
            self.kv_token_masker = None

        if self.trajectory_only:
            # Residual recurrent trajectory-only mode: first absolute, then all-delta.
            if self.trajectory_target in ("both", "tool"):
                if self.traj_tool_centroid_offset_head:
                    self.tool_prob_head = AbsoluteCentroidOffsetTrajectoryHead(
                        in_dim=self.vjepa_embed_dim,
                        num_points=self.num_tool_points,
                        num_tool_classes=self.num_tool_classes,
                        num_action_classes=self.num_action_classes,
                        num_combo_classes=self.num_combo_classes,
                        condition_mode=self.traj_condition_mode,
                        num_subframes=self.traj_num_subframes,
                        max_offset_norm=self.traj_tool_offset_max_norm,
                        local_softargmax_radius=self.traj_centroid_local_softargmax_radius,
                        subcell_r=self.traj_subcell_readout_r,
                        perpoint_local_readout=self.traj_perpoint_local_readout,
                        perpoint_local_radius=self.traj_perpoint_local_radius,
                        perpoint_gate_init=self.traj_perpoint_gate_init,
                        cond_film=self.traj_cond_film,
                        conf_head=self.traj_conf_head,
                        token_detector=self.traj_token_detector,
                        token_det_hidden=self.traj_token_det_hidden,
                        adaptive_gate=self.traj_perpoint_adaptive_gate,
                        temporal_query_mix=self.traj_temporal_query_mix,
                        temporal_query_layers=self.traj_temporal_query_layers,
                        temporal_query_spatial=self.traj_temporal_query_spatial,
                        temporal_query_interleave=self.traj_temporal_query_interleave,
                        num_virtual_queries=self.traj_num_virtual_queries,
                        locality_prior=self.traj_locality_prior,
                        region_prior=self.traj_region_prior,
                        pooled_context=self.traj_pooled_context,
                    )
                else:
                    self.tool_prob_head = ResidualRecurrentTrajectoryHead(
                        in_dim=self.vjepa_embed_dim,
                        num_points=self.num_tool_points,
                        num_tool_classes=self.num_tool_classes,
                        num_action_classes=self.num_action_classes,
                        num_combo_classes=self.num_combo_classes,
                        condition_mode=self.traj_condition_mode,
                        num_subframes=self.traj_num_subframes,
                        max_delta_norm=self.traj_recurrent_max_delta_norm,
                    )
            else:
                self.tool_prob_head = None
            if self.trajectory_target in ("both", "ct"):
                self.contact_prob_head = ResidualRecurrentTrajectoryHead(
                    in_dim=self.vjepa_embed_dim,
                    num_points=self.num_contact_points,
                    num_tool_classes=self.num_tool_classes,
                    num_action_classes=self.num_action_classes,
                    num_combo_classes=self.num_combo_classes,
                    condition_mode=self.traj_condition_mode,
                    num_subframes=self.traj_num_subframes,
                    max_delta_norm=self.traj_recurrent_max_delta_norm,
                )
            else:
                self.contact_prob_head = None
            self.tool_head = None
            self.contact_head = None
            if self.traj_refiner_mode == "attn" and self.tool_prob_head is not None:
                self.tool_refiner = AlphaBetaPointRefiner(
                    num_points=self.num_tool_points,
                    local_feat_dim_in=int(self.vjepa_embed_dim),
                    debug_shapes=self.debug_shapes,
                )
            elif self.traj_refiner_mode == "lite" and self.tool_prob_head is not None:
                self.tool_refiner = LitePointRefiner(
                    num_points=self.num_tool_points,
                    local_feat_dim_in=int(self.vjepa_embed_dim),
                    debug_shapes=self.debug_shapes,
                )
            elif self.traj_refiner_mode in (
                "s1_offset",
                "s2_iter_offset",
                "s3_support3x3",
                "s3p2x_finalrefine_l18",
                "s3p3_singlelayer_statecarry_refine",
                "s3p3x_singlelayer_statecarry_refine_plus1",
                "s3p8_singlelayer_statecarry_fourierbelief",
                "s3p8a_singlelayer_statecarry_residualfourierbelief",
                "s3p9_singlelayer_statecarry_anchoronly",
                "s3p9d_singlelayer_statecarry_anchoronly_detached",
                "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                "s3p10_singlelayer_statecarry_factorizedtoolresidual",
                "s3p4_singlelayer_statecarry_dynamiccentroid",
                "s3p5_singlelayer_statecarry_deformablesupport",
                "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
                "m1_progressive_multilayer_movingrefine",
                "m1_strict_s3_multilayer_mean",
                "m2_layerwise_statecarry_refine",
                "m3_layerwise_statecarry_dynamiccentroid",
                "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
                "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
                "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
            ) and self.tool_prob_head is not None:
                self.tool_refiner = CoTrackerDeltaOffsetRefiner(
                    num_points=self.num_tool_points,
                    local_feat_dim_in=int(self.vjepa_embed_dim),
                    max_delta_norm=float(self.traj_recurrent_max_delta_norm),
                    train_clip_len=max(int(self.output_num_frames), 1),
                    detach_refiner_input=(self.traj_refiner_mode not in STATECARRY_REFINER_MODES),
                    debug_shapes=self.debug_shapes,
                )
            else:
                self.tool_refiner = IdentityPointRefiner(
                    num_points=self.num_tool_points,
                    debug_shapes=self.debug_shapes,
                )
            if self.traj_refiner_mode == "attn" and self.contact_prob_head is not None:
                self.contact_refiner = AlphaBetaPointRefiner(
                    num_points=self.num_contact_points,
                    local_feat_dim_in=int(self.vjepa_embed_dim),
                    debug_shapes=self.debug_shapes,
                )
            elif self.traj_refiner_mode == "lite" and self.contact_prob_head is not None:
                self.contact_refiner = LitePointRefiner(
                    num_points=self.num_contact_points,
                    local_feat_dim_in=int(self.vjepa_embed_dim),
                    debug_shapes=self.debug_shapes,
                )
            elif self.traj_refiner_mode in (
                "s1_offset",
                "s2_iter_offset",
                "s3_support3x3",
                "s3p2x_finalrefine_l18",
                    "s3p3_singlelayer_statecarry_refine",
                    "s3p3x_singlelayer_statecarry_refine_plus1",
                    "s3p8_singlelayer_statecarry_fourierbelief",
                    "s3p8a_singlelayer_statecarry_residualfourierbelief",
                    "s3p4_singlelayer_statecarry_dynamiccentroid",
                    "s3p5_singlelayer_statecarry_deformablesupport",
                    "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
                "m1_progressive_multilayer_movingrefine",
                "m1_strict_s3_multilayer_mean",
                "m2_layerwise_statecarry_refine",
                "m3_layerwise_statecarry_dynamiccentroid",
                "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
                "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
                "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
            ) and self.contact_prob_head is not None:
                self.contact_refiner = CoTrackerDeltaOffsetRefiner(
                    num_points=self.num_contact_points,
                    local_feat_dim_in=int(self.vjepa_embed_dim),
                    max_delta_norm=float(self.traj_recurrent_max_delta_norm),
                    train_clip_len=max(int(self.output_num_frames), 1),
                    detach_refiner_input=(self.traj_refiner_mode not in STATECARRY_REFINER_MODES),
                    debug_shapes=self.debug_shapes,
                )
            else:
                self.contact_refiner = IdentityPointRefiner(
                    num_points=self.num_contact_points,
                    debug_shapes=self.debug_shapes,
                )
            self.combo_head = None
            self.combo_tool_head = None
            self.combo_action_head = None
            self.range_head = None
            self.enable_delta_branch = True
            # Not used in trajectory_only; keep frozen for checkpoint compatibility.
            for m in (self.shared_trunk, self.conv_refine, self.temporal_mean_residual):
                for p in m.parameters():
                    p.requires_grad = False
        else:
            self.tool_prob_head = None
            self.contact_prob_head = None
            if self.tool_only:
                # Tool-only probe mode: VJEPA tokens -> single cross-attention pooler -> linear classifier.
                self.tool_head = None
                self.contact_head = None
                self.tool_refiner = None
                self.contact_refiner = None
                # Keep modules attached for checkpoint compatibility, but freeze/unused in tool-only forward.
                for m in (self.shared_trunk, self.conv_refine, self.temporal_mean_residual):
                    for p in m.parameters():
                        p.requires_grad = False
                self.combo_attn_pool = CrossAttentionPooler(in_dim=self.vjepa_embed_dim)
                self.combo_tool_head = LinearClassifierHead(
                    in_dim=self.vjepa_embed_dim,
                    num_classes=self.num_tool_classes,
                )
                self.combo_head = None
                self.combo_action_head = None
                self.range_head = None
            elif self.range_only:
                self.tool_head = None
                self.contact_head = None
                self.tool_refiner = None
                self.contact_refiner = None
                self.combo_head = None
                self.combo_tool_head = None
                self.combo_action_head = None
                for m in (self.shared_trunk, self.conv_refine, self.temporal_mean_residual):
                    for p in m.parameters():
                        p.requires_grad = False
                range_temporal_pool_in_dim = proj_dim * 2
                if self.range_spatial_pool_mode in ("mean_max_condtoken", "mean_max_condtoken_locresid"):
                    self.range_spatial_pool = ConditionalSpatialTokenPool(
                        in_dim=proj_dim,
                        num_tokens=self.range_spatial_pool_num_tokens,
                        condition_mode=self.range_condition_mode,
                        num_tool_classes=self.num_tool_classes,
                        num_action_classes=self.num_action_classes,
                        num_combo_classes=self.num_combo_classes,
                    )
                    if self.range_spatial_pool_mode == "mean_max_condtoken":
                        range_temporal_pool_in_dim = proj_dim * 3
                self.range_temporal_pool = nn.Sequential(
                    nn.LayerNorm(range_temporal_pool_in_dim),
                    nn.Linear(range_temporal_pool_in_dim, proj_dim),
                    nn.GELU(),
                )
                if self.range_spatial_pool_mode == "mean_max_condtoken_locresid":
                    self.range_spatial_loc_residual = nn.Sequential(
                        nn.LayerNorm(proj_dim * 3),
                        nn.Linear(proj_dim * 3, proj_dim),
                        nn.GELU(),
                        nn.Linear(proj_dim, proj_dim),
                    )
                    final_linear = self.range_spatial_loc_residual[-1]
                    if isinstance(final_linear, nn.Linear):
                        nn.init.zeros_(final_linear.weight)
                        nn.init.zeros_(final_linear.bias)
                if self.range_temporal_context_mode == "temporalmaxer":
                    self.range_temporal_context = nn.Sequential(
                        *[
                            TemporalMaxPoolContextBlock(
                                dim=proj_dim,
                                kernel_size=self.range_temporal_context_kernel_size,
                            )
                            for _ in range(self.range_temporal_context_layers)
                        ]
                    )
                elif self.range_temporal_context_mode == "fpn":
                    self.range_temporal_context = TemporalPyramidFusionContext(
                        dim=proj_dim,
                        kernel_size=self.range_temporal_context_kernel_size,
                        num_levels=self.range_temporal_context_levels,
                    )
                if self.range_adapter_mode == "sola":
                    self.range_temporal_adapter = TemporalSoLaAdapter(
                        dim=proj_dim,
                        kernel_size=self.range_adapter_kernel_size,
                        proj_dim=max(64, proj_dim // 2),
                    )
                elif self.range_adapter_mode == "losa":
                    self.range_temporal_adapter = TemporalLoSALiteAdapter(
                        dim=proj_dim,
                        kernel_size=self.range_adapter_kernel_size,
                        proj_dim=max(64, proj_dim // 2),
                        num_heads=self.range_adapter_num_heads,
                    )
                elif self.range_adapter_mode == "dyfa":
                    self.range_temporal_adapter = TemporalDynamicMovementAdapter(
                        dim=proj_dim,
                        kernel_size=self.range_adapter_kernel_size,
                        proj_dim=max(64, proj_dim // 2),
                    )
                elif self.range_adapter_mode == "sola_dyfa":
                    self.range_temporal_adapter = TemporalSoLaDyFAGatedAdapter(
                        dim=proj_dim,
                        kernel_size=self.range_adapter_kernel_size,
                        proj_dim=max(64, proj_dim // 2),
                    )
                elif self.range_adapter_mode == "sola_dyfa_livegate":
                    self.range_temporal_adapter = TemporalSoLaDyFALiveGateAdapter(
                        dim=proj_dim,
                        kernel_size=self.range_adapter_kernel_size,
                        proj_dim=max(64, proj_dim // 2),
                    )
                elif self.range_adapter_mode == "sola_dyfa_laf":
                    self.range_temporal_adapter = TemporalSoLaDyFALocalFuseAdapter(
                        dim=proj_dim,
                        kernel_size=self.range_adapter_kernel_size,
                        proj_dim=max(64, proj_dim // 2),
                    )
                self.range_head = TemporalRangeOffsetHead(
                    in_dim=proj_dim,
                    num_tool_classes=self.num_tool_classes,
                    num_action_classes=self.num_action_classes,
                    num_combo_classes=self.num_combo_classes,
                    condition_mode=self.range_condition_mode,
                    head_mode=self.range_head_mode,
                    trident_num_bins=self.range_trident_num_bins,
                    proposal_mode=self.range_proposal_mode,
                    quality_mode=self.range_quality_mode,
                    task_decouple_mode=self.range_task_decouple_mode,
                )
            elif self.combo_probe_dual_attn:
                # Dual-head probe: split into two independent token->xattn branches
                # (tool branch and action branch share only the VJEPA tokens).
                self.tool_head = None
                self.contact_head = None
                self.tool_refiner = None
                self.contact_refiner = None
                for m in (self.shared_trunk, self.conv_refine, self.temporal_mean_residual):
                    for p in m.parameters():
                        p.requires_grad = False
                self.combo_tool_spatial_pool = CrossAttentionPooler(in_dim=self.vjepa_embed_dim)
                self.combo_action_temporal_pool = CrossAttentionPooler(in_dim=self.vjepa_embed_dim)
                self.combo_tool_head = LinearClassifierHead(
                    in_dim=self.vjepa_embed_dim,
                    num_classes=self.num_tool_classes,
                )
                self.combo_action_head = LinearClassifierHead(
                    in_dim=self.vjepa_embed_dim,
                    num_classes=self.num_action_classes,
                )
                if self.dual_attn_tool_prior:
                    self.combo_action_tool_prior = ToolConditionedActionPrior(
                        num_tool_classes=self.num_tool_classes,
                        query_dim=self.vjepa_embed_dim,
                        init_alpha=self.dual_attn_tool_prior_init_alpha,
                        detach_tool_prob=self.dual_attn_tool_prior_detach,
                    )
                self.combo_head = None
                self.range_head = None
            elif self.combo_probe_attn:
                # Combo probe mode: VJEPA tokens -> single cross-attention pooler -> linear combo head only.
                self.tool_head = None
                self.contact_head = None
                self.tool_refiner = None
                self.contact_refiner = None
                for m in (self.shared_trunk, self.conv_refine, self.temporal_mean_residual):
                    for p in m.parameters():
                        p.requires_grad = False
                self.combo_attn_pool = CrossAttentionPooler(in_dim=self.vjepa_embed_dim)
                self.combo_head = LinearClassifierHead(
                    in_dim=self.vjepa_embed_dim,
                    num_classes=self.num_combo_classes,
                )
                self.combo_tool_head = None
                self.combo_action_head = None
                self.range_head = None
            else:
                self.tool_head = SpatialHeatmapOffsetHead(
                    in_dim=proj_dim,
                    num_points=self.num_tool_points,
                    enable_delta_branch=self.enable_delta_branch,
                )
                self.contact_head = SpatialHeatmapOffsetHead(
                    in_dim=proj_dim,
                    num_points=self.num_contact_points,
                    enable_delta_branch=self.enable_delta_branch,
                )
                if self.refiner_mode == "attn":
                    self.tool_refiner = AlphaBetaPointRefiner(
                        num_points=self.num_tool_points,
                        local_feat_dim_in=int(proj_dim),
                        debug_shapes=self.debug_shapes,
                    )
                    self.contact_refiner = AlphaBetaPointRefiner(
                        num_points=self.num_contact_points,
                        local_feat_dim_in=int(proj_dim),
                        debug_shapes=self.debug_shapes,
                    )
                elif self.refiner_mode == "lite":
                    self.tool_refiner = LitePointRefiner(
                        num_points=self.num_tool_points,
                        local_feat_dim_in=int(proj_dim),
                        debug_shapes=self.debug_shapes,
                    )
                    self.contact_refiner = LitePointRefiner(
                        num_points=self.num_contact_points,
                        local_feat_dim_in=int(proj_dim),
                        debug_shapes=self.debug_shapes,
                    )
                else:
                    self.tool_refiner = IdentityPointRefiner(
                        num_points=self.num_tool_points,
                        debug_shapes=self.debug_shapes,
                    )
                    self.contact_refiner = IdentityPointRefiner(
                        num_points=self.num_contact_points,
                        debug_shapes=self.debug_shapes,
                    )
                self.combo_tool_head = ComboHead(
                    in_dim=proj_dim,
                    num_classes=self.num_tool_classes,
                    dropout=self.combo_head_dropout,
                )
                self.combo_head = ComboHead(
                    in_dim=proj_dim,
                    num_classes=self.num_combo_classes,
                    dropout=self.combo_head_dropout,
                )
                self.combo_action_head = ComboHead(
                    in_dim=proj_dim,
                    num_classes=self.num_action_classes,
                    dropout=self.combo_head_dropout,
                )
                self.range_head = TemporalRangeOffsetHead(
                    in_dim=proj_dim,
                    head_mode=self.range_head_mode,
                    trident_num_bins=self.range_trident_num_bins,
                    proposal_mode=self.range_proposal_mode,
                    quality_mode=self.range_quality_mode,
                    task_decouple_mode=self.range_task_decouple_mode,
                )
        if (not self.range_only) and (not self.tool_only) and (not self.combo_probe_attn) and (not self.combo_probe_dual_attn) and self.temporal_upsample == "deconv":
            self.tool_time_up_logits = TemporalDeconvUpsampler(
                channels=self.num_tool_points,
                groups=self.num_tool_points,
            )
            self.contact_time_up_logits = TemporalDeconvUpsampler(
                channels=self.num_contact_points,
                groups=self.num_contact_points,
            )
            if not self.trajectory_only:
                self.tool_time_up_delta = TemporalDeconvUpsampler(
                    channels=self.num_tool_points * 2,
                    groups=self.num_tool_points * 2,
                )
                self.contact_time_up_delta = TemporalDeconvUpsampler(
                    channels=self.num_contact_points * 2,
                    groups=self.num_contact_points * 2,
                )
            else:
                self.tool_time_up_delta = None
                self.contact_time_up_delta = None
            if self.trajectory_only:
                for p in self.tool_time_up_logits.parameters():
                    p.requires_grad = False
                for p in self.contact_time_up_logits.parameters():
                    p.requires_grad = False
            if not self.enable_delta_branch:
                if self.tool_time_up_delta is not None:
                    for p in self.tool_time_up_delta.parameters():
                        p.requires_grad = False
                if self.contact_time_up_delta is not None:
                    for p in self.contact_time_up_delta.parameters():
                        p.requires_grad = False
        else:
            self.tool_time_up_logits = None
            self.tool_time_up_delta = None
            self.contact_time_up_logits = None
            self.contact_time_up_delta = None

    def set_token_mask_epoch(self, epoch: int) -> None:
        self.token_mask_epoch = int(epoch)
        self.anchor_schedule_epoch = int(epoch)
        if self.kv_token_masker is not None:
            self.kv_token_masker.set_runtime_epoch(self.token_mask_epoch)

    def _anchor_activation_alpha(self) -> float:
        if self.traj_refiner_mode != "s3p9e_singlelayer_statecarry_anchoronly_ramped":
            return 1.0
        ep = int(self.anchor_schedule_epoch)
        delay = int(self.anchor_schedule_delay_epochs)
        ramp = int(self.anchor_schedule_ramp_epochs)
        if ep <= delay:
            return 0.0
        if ramp <= 0:
            return 1.0
        frac = float(ep - delay) / float(max(ramp, 1))
        return max(0.0, min(1.0, frac))

    def _apply_stage0_centroid_carry_noise(
        self,
        centroid_xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise_abs = centroid_xy.new_zeros(centroid_xy.shape[:3])
        if (
            (not self.training)
            or self.traj_stage0_centroid_noise_prob <= 0.0
            or self.traj_stage0_centroid_noise_sigma <= 0.0
        ):
            return centroid_xy, noise_abs
        mask = (
            torch.rand(
                centroid_xy.shape[:3] + (1,),
                device=centroid_xy.device,
                dtype=torch.float32,
            )
            < float(self.traj_stage0_centroid_noise_prob)
        ).to(dtype=centroid_xy.dtype)
        noise = torch.randn_like(centroid_xy.float()) * float(self.traj_stage0_centroid_noise_sigma)
        noisy_centroid = (centroid_xy.float() + mask.float() * noise).clamp(0.0, 1.0).to(dtype=centroid_xy.dtype)
        noise_abs = (noisy_centroid.float() - centroid_xy.float()).norm(dim=-1)
        return noisy_centroid, noise_abs

    @property
    def trainable_modules(self):
        if self.trajectory_only:
            out = {}
            if self.tool_prob_head is not None:
                out["tool_prob_head"] = self.tool_prob_head
            if self.contact_prob_head is not None:
                out["contact_prob_head"] = self.contact_prob_head
            if self.traj_support_projector is not None:
                out["traj_support_projector"] = self.traj_support_projector
            if self.traj_statecarry_projector is not None:
                out["traj_statecarry_projector"] = self.traj_statecarry_projector
            if self.traj_multilayer_stage1_projector is not None:
                out["traj_multilayer_stage1_projector"] = self.traj_multilayer_stage1_projector
            if self.traj_multilayer_stage2_projector is not None:
                out["traj_multilayer_stage2_projector"] = self.traj_multilayer_stage2_projector
            if self.traj_centroid_anchor_adapter is not None:
                out["traj_centroid_anchor_adapter"] = self.traj_centroid_anchor_adapter
            if self.traj_shallow_anchor_verifier is not None:
                out["traj_shallow_anchor_verifier"] = self.traj_shallow_anchor_verifier
            if self.traj_factorized_tool_adapter is not None:
                out["traj_factorized_tool_adapter"] = self.traj_factorized_tool_adapter
            if self.traj_context_point_adapter is not None:
                out["traj_context_point_adapter"] = self.traj_context_point_adapter
            if self.traj_context_point_gate is not None:
                out["traj_context_point_gate"] = self.traj_context_point_gate
            if self.traj_joint_point_block is not None:
                out["traj_joint_point_block"] = self.traj_joint_point_block
            if self.tool_refiner is not None:
                out["tool_refiner"] = self.tool_refiner
            if self.contact_refiner is not None:
                out["contact_refiner"] = self.contact_refiner
            if self.kv_token_masker is not None:
                out["kv_token_masker"] = self.kv_token_masker
        elif self.range_only:
            out = {
                "shared_trunk": self.shared_trunk,
                "conv_refine": self.conv_refine,
                "temporal_mean_residual": self.temporal_mean_residual,
                "range_spatial_pool": self.range_spatial_pool,
                "range_spatial_loc_residual": self.range_spatial_loc_residual,
                "range_temporal_pool": self.range_temporal_pool,
                "range_head": self.range_head,
            }
            if self.range_temporal_context is not None:
                out["range_temporal_context"] = self.range_temporal_context
            if self.range_temporal_adapter is not None:
                out["range_temporal_adapter"] = self.range_temporal_adapter
        elif self.tool_only:
            out = {
                "combo_attn_pool": self.combo_attn_pool,
                "combo_tool_head": self.combo_tool_head,
            }
        elif self.combo_probe_attn:
            out = {
                "combo_attn_pool": self.combo_attn_pool,
                "combo_head": self.combo_head,
            }
        elif self.combo_probe_dual_attn:
            out = {
                "combo_tool_spatial_pool": self.combo_tool_spatial_pool,
                "combo_action_temporal_pool": self.combo_action_temporal_pool,
                "combo_tool_head": self.combo_tool_head,
                "combo_action_head": self.combo_action_head,
            }
            if self.combo_action_tool_prior is not None:
                out["combo_action_tool_prior"] = self.combo_action_tool_prior
        else:
            out = {
                "shared_trunk": self.shared_trunk,
                "conv_refine": self.conv_refine,
                "temporal_mean_residual": self.temporal_mean_residual,
                "tool_head": self.tool_head,
                "contact_head": self.contact_head,
                "tool_refiner": self.tool_refiner,
                "contact_refiner": self.contact_refiner,
                "combo_tool_head": self.combo_tool_head,
            }
            if self.combo_head is not None:
                out["combo_head"] = self.combo_head
            if self.combo_action_head is not None:
                out["combo_action_head"] = self.combo_action_head
            if self.range_head is not None:
                out["range_head"] = self.range_head
        if self.temporal_upsample == "deconv":
            if self.tool_time_up_logits is not None:
                out["tool_time_up_logits"] = self.tool_time_up_logits
            if self.contact_time_up_logits is not None:
                out["contact_time_up_logits"] = self.contact_time_up_logits
            if self.tool_time_up_delta is not None:
                out["tool_time_up_delta"] = self.tool_time_up_delta
            if self.contact_time_up_delta is not None:
                out["contact_time_up_delta"] = self.contact_time_up_delta
        return {name: module for name, module in out.items() if module is not None}

    @staticmethod
    def _temporal_interpolate_dim1(x: torch.Tensor, target_t: int) -> torch.Tensor:
        if x.shape[1] == target_t:
            return x
        bsz, t_in = x.shape[:2]
        rest = list(x.shape[2:])
        x_perm = x.permute(0, *range(2, x.ndim), 1).contiguous()  # [B,*rest,T]
        x_flat = x_perm.reshape(-1, 1, t_in)
        y_flat = F.interpolate(x_flat, size=target_t, mode="linear", align_corners=False)
        y_perm = y_flat.reshape(bsz, *rest, target_t)  # [B,*rest,T_out]
        order = [0, y_perm.ndim - 1] + list(range(1, y_perm.ndim - 1))
        return y_perm.permute(*order).contiguous()  # [B,T_out,*rest]

    def _xy_norm_to_canvas_px(self, xy_norm: torch.Tensor) -> torch.Tensor:
        if xy_norm.ndim < 2 or int(xy_norm.shape[-1]) != 2:
            raise ValueError(f"`xy_norm` must end with 2 channels (x,y), got {tuple(xy_norm.shape)}")
        scale = float(max(int(self.output_canvas_size) - 1, 1))
        return xy_norm.clamp(0.0, 1.0) * scale

    @staticmethod
    def _xy_to_spatial_stub(
        pred_xy: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Dict[str, torch.Tensor]:
        # Build compatibility logits/probs/heatmap from xy without using heatmap heads.
        if pred_xy.ndim != 4 or int(pred_xy.shape[-1]) != 2:
            raise ValueError(f"`pred_xy` must be [B,T,P,2], got {tuple(pred_xy.shape)}")
        bsz, t_len, n_pts, _ = pred_xy.shape
        h = int(h_patches)
        w = int(w_patches)
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid spatial size: H={h}, W={w}")
        s = int(h * w)

        x = (pred_xy[..., 0] * float(w)).floor().long().clamp(0, w - 1)
        y = (pred_xy[..., 1] * float(h)).floor().long().clamp(0, h - 1)
        idx = y * w + x  # [B,T,P]

        logits = pred_xy.new_full((bsz, t_len, n_pts, s), -8.0)
        logits.scatter_(dim=-1, index=idx.unsqueeze(-1), src=pred_xy.new_full((bsz, t_len, n_pts, 1), 8.0))
        probs = torch.softmax(logits, dim=-1)
        heatmap = probs.reshape(bsz, t_len, n_pts, h, w)
        return {
            "logits": logits,
            "probs": probs,
            "heatmap": heatmap,
        }

    @staticmethod
    def _decode_range_from_points(
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

    @staticmethod
    def _decode_range_from_proposals(
        range_start_logits: torch.Tensor,
        range_end_logits: torch.Tensor,
        range_proposal_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if range_start_logits.ndim != 2:
            raise ValueError(f"`range_start_logits` must be [B,T], got {tuple(range_start_logits.shape)}")
        if range_end_logits.ndim != 2:
            raise ValueError(f"`range_end_logits` must be [B,T], got {tuple(range_end_logits.shape)}")
        if range_proposal_logits.ndim != 3:
            raise ValueError(f"`range_proposal_logits` must be [B,T,T], got {tuple(range_proposal_logits.shape)}")
        bsz, t_len = range_start_logits.shape
        if range_end_logits.shape != (bsz, t_len):
            raise ValueError(
                f"`range_end_logits` must match start shape {(bsz, t_len)}, got {tuple(range_end_logits.shape)}"
            )
        if range_proposal_logits.shape != (bsz, t_len, t_len):
            raise ValueError(
                f"`range_proposal_logits` must be {(bsz, t_len, t_len)}, got {tuple(range_proposal_logits.shape)}"
            )
        if t_len <= 0:
            z = range_start_logits.new_zeros((bsz,))
            return {
                "pred_action_start_norm": z,
                "pred_action_end_norm": z,
                "pred_action_start_t": z,
                "pred_action_end_t": z,
            }
        start_prob = torch.sigmoid(range_start_logits.float()).unsqueeze(2)
        end_prob = torch.sigmoid(range_end_logits.float()).unsqueeze(1)
        proposal_prob = torch.sigmoid(range_proposal_logits.float())
        score = proposal_prob * start_prob * end_prob
        valid_mask = torch.triu(torch.ones((t_len, t_len), device=score.device, dtype=torch.bool), diagonal=0)
        score = score.masked_fill(~valid_mask.unsqueeze(0), -1.0)
        flat_idx = score.view(bsz, -1).argmax(dim=1)
        start_idx = flat_idx // t_len
        end_idx = flat_idx % t_len
        start_t = start_idx.to(range_proposal_logits.dtype)
        end_t = end_idx.to(range_proposal_logits.dtype)
        if t_len > 1:
            max_t = float(t_len - 1)
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

    @staticmethod
    def _refine_range_boundaries_local_argmax(
        decoded: Dict[str, torch.Tensor],
        range_start_logits: torch.Tensor,
        range_end_logits: torch.Tensor,
        radius: int,
    ) -> Dict[str, torch.Tensor]:
        if range_start_logits.ndim != 2:
            raise ValueError(f"`range_start_logits` must be [B,T], got {tuple(range_start_logits.shape)}")
        if range_end_logits.ndim != 2:
            raise ValueError(f"`range_end_logits` must be [B,T], got {tuple(range_end_logits.shape)}")
        if int(radius) <= 0:
            raise ValueError(f"`radius` must be > 0, got {radius}")
        start_t = decoded["pred_action_start_t"].float()
        end_t = decoded["pred_action_end_t"].float()
        bsz, t_len = range_start_logits.shape
        if range_end_logits.shape != (bsz, t_len):
            raise ValueError(
                f"`range_end_logits` must match start shape {(bsz, t_len)}, got {tuple(range_end_logits.shape)}"
            )
        if t_len <= 0:
            z = start_t.new_zeros(start_t.shape)
            return {
                "pred_action_start_norm": z,
                "pred_action_end_norm": z,
                "pred_action_start_t": z,
                "pred_action_end_t": z,
            }
        refined_start = start_t.clone()
        refined_end = end_t.clone()
        for i in range(bsz):
            start_center = int(round(float(start_t[i].item())))
            end_center = int(round(float(end_t[i].item())))
            start_lo = max(0, start_center - int(radius))
            start_hi = min(t_len - 1, start_center + int(radius))
            start_slice = range_start_logits[i, start_lo : start_hi + 1]
            start_idx = start_lo + int(torch.argmax(start_slice).item())

            end_lo = max(start_idx, end_center - int(radius))
            end_hi = min(t_len - 1, end_center + int(radius))
            if end_lo > end_hi:
                end_lo = start_idx
                end_hi = start_idx
            end_slice = range_end_logits[i, end_lo : end_hi + 1]
            end_idx = end_lo + int(torch.argmax(end_slice).item())
            if end_idx < start_idx:
                end_idx = start_idx
            refined_start[i] = float(start_idx)
            refined_end[i] = float(end_idx)
        if t_len > 1:
            max_t = float(t_len - 1)
            start_norm = refined_start / max_t
            end_norm = refined_end / max_t
        else:
            refined_start = refined_start.new_zeros(refined_start.shape)
            refined_end = refined_end.new_zeros(refined_end.shape)
            start_norm = refined_start
            end_norm = refined_end
        return {
            "pred_action_start_norm": start_norm.clamp(0.0, 1.0),
            "pred_action_end_norm": end_norm.clamp(0.0, 1.0),
            "pred_action_start_t": refined_start,
            "pred_action_end_t": refined_end,
        }

    @staticmethod
    def _norm_to_bin_logits(norm: torch.Tensor, num_bins: int, sharpness: float = 40.0) -> torch.Tensor:
        if norm.ndim != 1:
            raise ValueError(f"`norm` must be [B], got {tuple(norm.shape)}")
        centers = torch.linspace(0.0, 1.0, steps=int(num_bins), device=norm.device, dtype=norm.dtype).view(1, -1)
        dist2 = (norm.view(-1, 1) - centers).pow(2)
        return -float(sharpness) * dist2

    def _structured_combo_logits(
        self,
        combo_tool_logits: torch.Tensor,
        combo_action_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compose combo logits from tool/action logits via fixed legal mapping."""
        if combo_tool_logits.ndim != 2:
            raise ValueError(f"`combo_tool_logits` must be [B,Ct], got {tuple(combo_tool_logits.shape)}")
        if combo_action_logits.ndim != 2:
            raise ValueError(f"`combo_action_logits` must be [B,Ca], got {tuple(combo_action_logits.shape)}")
        if int(combo_tool_logits.shape[0]) != int(combo_action_logits.shape[0]):
            raise ValueError(
                f"Batch mismatch: tool={tuple(combo_tool_logits.shape)} action={tuple(combo_action_logits.shape)}"
            )

        bsz = int(combo_tool_logits.shape[0])
        combo_logits = combo_tool_logits.new_full((bsz, int(self.num_combo_classes)), -1e9)

        # combo ids:
        # 0 hook|dissect, 1 grasper|grasp, 2 clipper|clip,
        # 3 scissors|cut, 4 scissors|dissect, 5 grasper|dissect
        combo_to_tool = torch.tensor([0, 1, 2, 3, 3, 1], device=combo_logits.device, dtype=torch.long)
        combo_to_action = torch.tensor([0, 1, 2, 3, 0, 0], device=combo_logits.device, dtype=torch.long)

        if int(self.num_combo_classes) != int(combo_to_tool.numel()):
            raise ValueError(
                f"`num_combo_classes` must match mapping size {int(combo_to_tool.numel())}, got {int(self.num_combo_classes)}"
            )

        n_tool = int(combo_tool_logits.shape[1])
        n_action = int(combo_action_logits.shape[1])
        for cid in range(int(combo_to_tool.numel())):
            tid = int(combo_to_tool[cid].item())
            aid = int(combo_to_action[cid].item())
            if tid < 0 or tid >= n_tool or aid < 0 or aid >= n_action:
                continue
            combo_logits[:, cid] = combo_tool_logits[:, tid] + combo_action_logits[:, aid]

        return combo_logits

    def _decode_spatial_outputs(
        self,
        logits: torch.Tensor,
        delta_raw: Optional[torch.Tensor],
        grid_xy: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Dict[str, torch.Tensor]:
        # logits: [B,T,P,S], delta_raw: [B,T,P,S,2] or None (heatmap-only mode)
        probs = torch.softmax(logits / self.heatmap_softmax_temp, dim=-1)
        coarse_xy = torch.einsum("btps,sd->btpd", probs, grid_xy)  # [B,T,P,2]

        if delta_raw is None:
            delta_xy = torch.zeros_like(coarse_xy)
            pred_xy = coarse_xy.clamp(0.0, 1.0)
        else:
            # Sub-cell offsets are constrained within each patch cell in normalized coords.
            delta_x_scale = 0.5 / float(w_patches)
            delta_y_scale = 0.5 / float(h_patches)
            delta_scale = delta_raw.new_tensor([delta_x_scale, delta_y_scale]).view(1, 1, 1, 1, 2)
            delta = torch.tanh(delta_raw) * delta_scale
            delta_xy = torch.einsum("btps,btpsd->btpd", probs, delta)
            pred_xy = (coarse_xy + delta_xy).clamp(0.0, 1.0)

        conf = probs.max(dim=-1).values
        bsz, t_out, n_pts, _ = logits.shape
        heatmap = probs.view(bsz, t_out, n_pts, h_patches, w_patches)

        return {
            "logits": logits,
            "probs": probs,
            "pred_xy": pred_xy,
            "conf": conf,
            "heatmap": heatmap,
            "coarse_xy": coarse_xy,
            "delta_xy": delta_xy,
        }

    @staticmethod
    def _local_feat_from_argmax_cell(probs: torch.Tensor, feat_flat: torch.Tensor) -> torch.Tensor:
        """Pick local feature from argmax spatial cell for each (B,T,P)."""
        # probs: [B,T,P,S], feat_flat: [B,T,S,D]
        if probs.ndim != 4:
            raise ValueError(f"`probs` must be [B,T,P,S], got {tuple(probs.shape)}")
        if feat_flat.ndim != 4:
            raise ValueError(f"`feat_flat` must be [B,T,S,D], got {tuple(feat_flat.shape)}")
        bsz, t_len, n_pts, s = probs.shape
        if tuple(feat_flat.shape[:3]) != (bsz, t_len, s):
            raise ValueError(
                "Shape mismatch between probs and feat_flat: "
                f"probs={tuple(probs.shape)} feat_flat={tuple(feat_flat.shape)}"
            )
        d = int(feat_flat.shape[-1])
        idx = probs.argmax(dim=-1).long()  # [B,T,P]
        feat_bt = feat_flat.reshape(bsz * t_len, s, d).contiguous()  # [BT,S,D]
        idx_bt = idx.reshape(bsz * t_len, n_pts)  # [BT,P]
        idx_exp = idx_bt.unsqueeze(-1).expand(-1, -1, d)  # [BT,P,D]
        local = feat_bt.gather(dim=1, index=idx_exp)  # [BT,P,D]
        return local.reshape(bsz, t_len, n_pts, d).contiguous()  # [B,T,P,D]

    @staticmethod
    def _local_feat_from_xy_bilinear(
        xy: torch.Tensor,
        feat_flat: torch.Tensor,
        *,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Sample local features at the current point estimate with bilinear sampling.

        This follows the same idea as CoTracker: the local descriptor should depend on
        the current trajectory estimate, not only on the argmax spatial cell.
        """
        if xy.ndim != 4 or int(xy.shape[-1]) != 2:
            raise ValueError(f"`xy` must be [B,T,P,2], got {tuple(xy.shape)}")
        if feat_flat.ndim != 4:
            raise ValueError(f"`feat_flat` must be [B,T,S,D], got {tuple(feat_flat.shape)}")
        bsz, t_len, n_pts, _ = xy.shape
        if int(feat_flat.shape[0]) != bsz or int(feat_flat.shape[1]) != t_len:
            raise ValueError(
                f"Batch/time mismatch between xy={tuple(xy.shape)} and feat_flat={tuple(feat_flat.shape)}"
            )
        s = int(feat_flat.shape[2])
        if int(h_patches) * int(w_patches) != s:
            raise ValueError(
                f"spatial_hw mismatch: h_patches*w_patches={int(h_patches) * int(w_patches)} vs S={s}"
            )
        d = int(feat_flat.shape[-1])
        feat_map = feat_flat.reshape(bsz, t_len, int(h_patches), int(w_patches), d)
        feat_map = feat_map.permute(0, 1, 4, 2, 3).reshape(bsz * t_len, d, int(h_patches), int(w_patches)).contiguous()
        xy_bt = xy.reshape(bsz * t_len, n_pts, 2).float().clamp(0.0, 1.0)
        grid = xy_bt.mul(2.0).sub(1.0).unsqueeze(2)  # [BT,P,1,2]
        sampled = F.grid_sample(
            feat_map.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [BT,D,P,1]
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()
        return sampled.reshape(bsz, t_len, n_pts, d).contiguous()

    @staticmethod
    def _local_feat_support_stack_from_xy(
        xy: torch.Tensor,
        feat_flat: torch.Tensor,
        *,
        h_patches: int,
        w_patches: int,
        support_radius: int,
    ) -> torch.Tensor:
        """Sample a small local neighborhood around the current point estimate."""
        if int(support_radius) < 0:
            raise ValueError(f"`support_radius` must be >= 0, got {support_radius}")
        if int(support_radius) == 0:
            return KeypointComboNet._local_feat_from_xy_bilinear(
                xy,
                feat_flat,
                h_patches=h_patches,
                w_patches=w_patches,
            ).unsqueeze(3)
        if xy.ndim != 4 or int(xy.shape[-1]) != 2:
            raise ValueError(f"`xy` must be [B,T,P,2], got {tuple(xy.shape)}")
        if feat_flat.ndim != 4:
            raise ValueError(f"`feat_flat` must be [B,T,S,D], got {tuple(feat_flat.shape)}")
        bsz, t_len, n_pts, _ = xy.shape
        if int(feat_flat.shape[0]) != bsz or int(feat_flat.shape[1]) != t_len:
            raise ValueError(
                f"Batch/time mismatch between xy={tuple(xy.shape)} and feat_flat={tuple(feat_flat.shape)}"
            )
        s = int(feat_flat.shape[2])
        if int(h_patches) * int(w_patches) != s:
            raise ValueError(
                f"spatial_hw mismatch: h_patches*w_patches={int(h_patches) * int(w_patches)} vs S={s}"
            )
        d = int(feat_flat.shape[-1])
        feat_map = feat_flat.reshape(bsz, t_len, int(h_patches), int(w_patches), d)
        feat_map = feat_map.permute(0, 1, 4, 2, 3).reshape(bsz * t_len, d, int(h_patches), int(w_patches)).contiguous()
        xy_bt = xy.reshape(bsz * t_len, n_pts, 2).float().clamp(0.0, 1.0)
        offset_x = torch.arange(
            -int(support_radius),
            int(support_radius) + 1,
            device=xy.device,
            dtype=xy_bt.dtype,
        ) / float(max(int(w_patches), 1))
        offset_y = torch.arange(
            -int(support_radius),
            int(support_radius) + 1,
            device=xy.device,
            dtype=xy_bt.dtype,
        ) / float(max(int(h_patches), 1))
        grid_y, grid_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
        offsets = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)
        support_grid = xy_bt.unsqueeze(2) + offsets
        support_grid = support_grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        sampled = F.grid_sample(
            feat_map.float(),
            support_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [BT,D,P,K]
        sampled = sampled.permute(0, 2, 3, 1).contiguous()
        return sampled.reshape(bsz, t_len, n_pts, sampled.shape[2], d).contiguous()

    def _trajectory_local_feat_from_xy(
        self,
        xy: torch.Tensor,
        feat_flat: torch.Tensor,
        *,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        if self.traj_refiner_mode in ("s3_support3x3", "s3p2x_finalrefine_l18", "m1_strict_s3_multilayer_mean"):
            if self.traj_support_projector is None:
                raise RuntimeError(
                    "`traj_support_projector` must be initialized for support3x3 trajectory refiner modes"
                )
            support_feat = self._local_feat_support_stack_from_xy(
                xy,
                feat_flat,
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=1,
            )
            return self.traj_support_projector(support_feat)
        return self._local_feat_from_xy_bilinear(
            xy,
            feat_flat,
            h_patches=h_patches,
            w_patches=w_patches,
        )

    @staticmethod
    def _match_feat_time_steps(feat_flat: torch.Tensor, target_t: int) -> torch.Tensor:
        if feat_flat.ndim != 4:
            raise ValueError(f"`feat_flat` must be [B,T,S,D], got {tuple(feat_flat.shape)}")
        t_in = int(feat_flat.shape[1])
        t_out = int(target_t)
        if t_out <= 0:
            raise ValueError(f"`target_t` must be > 0, got {t_out}")
        if t_in == t_out:
            return feat_flat
        if t_out > t_in and (t_out % t_in) == 0:
            return feat_flat.repeat_interleave(t_out // t_in, dim=1)
        raise ValueError(f"Cannot match feature time length from T={t_in} to target_t={t_out}")

    @staticmethod
    def _match_multilayer_feat_time_steps(feat_layers: torch.Tensor, target_t: int) -> torch.Tensor:
        if feat_layers.ndim != 5:
            raise ValueError(f"`feat_layers` must be [B,K,T,S,D], got {tuple(feat_layers.shape)}")
        matched = [
            KeypointComboNet._match_feat_time_steps(feat_layers[:, layer_idx, ...], target_t=target_t)
            for layer_idx in range(int(feat_layers.shape[1]))
        ]
        return torch.stack(matched, dim=1)

    def _trajectory_local_feat_from_xy_multilayer(
        self,
        xy: torch.Tensor,
        feat_layers: torch.Tensor,
        *,
        h_patches: int,
        w_patches: int,
        projector: MultiLayerStagePairSupportProjector,
    ) -> torch.Tensor:
        if feat_layers.ndim != 5:
            raise ValueError(f"`feat_layers` must be [B,T,L,S,D], got {tuple(feat_layers.shape)}")
        if projector is None:
            raise RuntimeError("`projector` must be initialized for multi-layer trajectory refine.")
        num_layers = int(feat_layers.shape[2])
        support_list = []
        for layer_idx in range(num_layers):
            support_layer = self._local_feat_support_stack_from_xy(
                xy,
                feat_layers[:, :, layer_idx, :, :],
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=1,
            )
            support_list.append(support_layer.unsqueeze(3))
        support_feat = torch.cat(support_list, dim=3)
        return projector(support_feat)

    def _run_multilayer_cotracker_offset_refiner(
        self,
        *,
        refiner: CoTrackerDeltaOffsetRefiner,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        feat_layers: torch.Tensor,
        h_patches: int,
        w_patches: int,
        pred_centroid_xy: torch.Tensor,
        pred_offsets_xy: torch.Tensor,
        traj_cond: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if feat_layers.ndim != 5:
            raise ValueError(f"`feat_layers` must be [B,K,T,S,D], got {tuple(feat_layers.shape)}")
        if int(feat_layers.shape[1]) < 4:
            raise ValueError(
                "M1 multi-layer moving refine expects at least 4 cached layers "
                f"(got shape={tuple(feat_layers.shape)})"
            )
        if self.traj_multilayer_stage1_projector is None or self.traj_multilayer_stage2_projector is None:
            raise RuntimeError("M1 stage projectors must be initialized for multi-layer moving refine.")

        base_xy = pred_xy
        base_conf = pred_conf
        base_offsets = pred_offsets_xy
        current_xy = base_xy
        current_conf = base_conf
        current_offsets = base_offsets
        stages: List[Dict[str, Any]] = []

        stage_feat_pairs = [
            torch.stack([feat_layers[:, -1, ...], feat_layers[:, -2, ...]], dim=2),
            torch.stack([feat_layers[:, -3, ...], feat_layers[:, -4, ...]], dim=2),
        ]
        stage_projectors = [
            self.traj_multilayer_stage1_projector,
            self.traj_multilayer_stage2_projector,
        ]
        stage_layer_ids = [
            [int(feat_layers.shape[1]) - 1, int(feat_layers.shape[1]) - 2],
            [int(feat_layers.shape[1]) - 3, int(feat_layers.shape[1]) - 4],
        ]

        for stage_idx, (feat_pair, projector) in enumerate(zip(stage_feat_pairs, stage_projectors), start=1):
            current_local_feat = self._trajectory_local_feat_from_xy_multilayer(
                current_xy,
                feat_pair,
                h_patches=h_patches,
                w_patches=w_patches,
                projector=projector,
            )
            current_centroid_local_feat = self._trajectory_local_feat_from_xy_multilayer(
                pred_centroid_xy,
                feat_pair,
                h_patches=h_patches,
                w_patches=w_patches,
                projector=projector,
            )
            current_out = refiner(
                pred_xy=current_xy,
                pred_conf=current_conf,
                local_feat=current_local_feat,
                pred_centroid_xy=pred_centroid_xy,
                pred_offsets_xy=current_offsets,
                centroid_local_feat=current_centroid_local_feat,
                traj_cond=traj_cond,
            )
            current_offsets_out = current_out.get("pred_offsets_xy", current_out["pred_xy"] - pred_centroid_xy)
            current_delta_offsets = current_out.get("delta_offsets_xy", current_offsets_out - current_offsets)
            stages.append(
                {
                    "out": current_out,
                    "local_feat": current_local_feat,
                    "centroid_local_feat": current_centroid_local_feat,
                    "pred_xy": current_out["pred_xy"],
                    "pred_conf": current_out.get("pred_conf", current_conf),
                    "pred_offsets_xy": current_offsets_out,
                    "delta_offsets_xy": current_delta_offsets,
                    "stage_idx": int(stage_idx),
                    "layer_ids": list(stage_layer_ids[stage_idx - 1]),
                }
            )
            current_xy = stages[-1]["pred_xy"]
            current_conf = stages[-1]["pred_conf"]
            current_offsets = stages[-1]["pred_offsets_xy"]

        final_stage = stages[-1]
        shape_debug: Optional[Dict[str, Any]] = None
        if self.debug_shapes:
            shape_debug = {
                "num_iters": 2,
                "layer_pairs_by_stage": [stage["layer_ids"] for stage in stages],
                "local_feat_shapes": [list(stage["local_feat"].shape) for stage in stages],
                "centroid_local_feat_shapes": [list(stage["centroid_local_feat"].shape) for stage in stages],
                "iter_shape_debug": [stage["out"].get("shape_debug", None) for stage in stages],
            }

        return {
            "stage0_xy": base_xy,
            "stage0_conf": base_conf,
            "stage0_offsets_xy": base_offsets,
            "stages": stages,
            "stage1_xy": stages[0]["pred_xy"],
            "stage1_conf": stages[0]["pred_conf"],
            "stage1_offsets_xy": stages[0]["pred_offsets_xy"],
            "stage1_delta_offsets_xy": stages[0]["delta_offsets_xy"],
            "final_out": final_stage["out"],
            "final_xy": final_stage["pred_xy"],
            "final_conf": final_stage["pred_conf"],
            "final_offsets_xy": final_stage["pred_offsets_xy"],
            "final_delta_offsets_xy": final_stage["delta_offsets_xy"],
            "total_delta_xy": final_stage["pred_xy"] - base_xy,
            "total_delta_offsets_xy": final_stage["pred_offsets_xy"] - base_offsets,
            "shape_debug": shape_debug,
        }

    def _compute_layerwise_dynamic_centroid_beta(self, *, stage_idx: int, num_stages: int) -> float:
        if self.traj_refiner_mode not in DYNAMIC_CENTROID_REFINER_MODES:
            return 0.0
        if num_stages <= 1:
            return float(self.traj_dynamic_centroid_max_beta)
        progress = float(max(stage_idx - 1, 0)) / float(max(num_stages - 1, 1))
        return float(self.traj_dynamic_centroid_max_beta) * progress

    def _refresh_layerwise_centroid(
        self,
        *,
        current_centroid: torch.Tensor,
        current_xy: torch.Tensor,
        stage_idx: int,
        num_stages: int,
    ) -> Dict[str, Any]:
        current_centroid_f = current_centroid.float()
        proposed_centroid = current_xy.float().mean(dim=2, keepdim=True)
        beta = self._compute_layerwise_dynamic_centroid_beta(stage_idx=stage_idx, num_stages=num_stages)
        if beta <= 0.0:
            next_centroid = current_centroid_f
        else:
            # Keep M3 close to M2: only refresh the centroid geometry after each improved point estimate.
            next_centroid = torch.lerp(current_centroid_f, proposed_centroid, beta).clamp(0.0, 1.0)
        return {
            "proposed_centroid_xy": proposed_centroid,
            "next_centroid_xy": next_centroid,
            "beta": float(beta),
        }

    def _run_layerwise_context_point_adapter(
        self,
        *,
        adapter: LayerwiseContextPointEvidenceAdapter,
        feat_layer: torch.Tensor,
        current_support: torch.Tensor,
        current_evidence: Dict[str, torch.Tensor],
        current_state: torch.Tensor,
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Dict[str, torch.Tensor]:
        if self.traj_support_projector is None:
            raise RuntimeError("M4 context-point refine requires `traj_support_projector`.")
        if "expected_offset" not in current_evidence:
            raise RuntimeError("M4 context-point refine requires `expected_offset` from the state-carry projector.")
        if feat_layer.ndim != 4:
            raise ValueError(f"`feat_layer` must be [B,T,S,D], got {tuple(feat_layer.shape)}")
        main_support_feat = self.traj_support_projector(current_support)
        adapter_query, context_offsets = adapter.build_query_and_offsets(
            main_feat=current_evidence["feat"],
            main_state=current_state,
            main_support_feat=main_support_feat,
            current_xy=current_xy,
            current_offsets=current_offsets,
            expected_offset=current_evidence["expected_offset"],
            h_patches=h_patches,
            w_patches=w_patches,
        )
        context_xy = (current_xy.float().unsqueeze(3) + context_offsets).clamp(0.0, 1.0)
        bsz, t_len, n_pts, n_ctx, _ = context_xy.shape
        context_xy_flat = context_xy.reshape(bsz, t_len, n_pts * n_ctx, 2)
        context_support = self._local_feat_support_stack_from_xy(
            context_xy_flat,
            feat_layer,
            h_patches=h_patches,
            w_patches=w_patches,
            support_radius=int(self.traj_context_support_radius),
        )
        context_support_feat = self.traj_support_projector(context_support)
        context_support_feat = context_support_feat.reshape(bsz, t_len, n_pts, n_ctx, -1)
        context_pack = adapter(
            main_feat=current_evidence["feat"],
            main_support_feat=main_support_feat,
            context_support_feat=context_support_feat,
            query=adapter_query,
            current_offsets=current_offsets,
            expected_offset=current_evidence["expected_offset"],
            context_offsets=context_offsets,
        )
        context_pack["context_offsets"] = context_offsets
        context_pack["context_xy"] = context_xy
        return context_pack

    def _run_layerwise_template_corr_adapter(
        self,
        *,
        adapter: LayerwiseTemplateCorrelationAdapter,
        template_support: torch.Tensor,
        current_evidence: Dict[str, torch.Tensor],
        current_state: torch.Tensor,
        template_state: torch.Tensor,
        current_xy: torch.Tensor,
        current_offsets: torch.Tensor,
        layer_rank: int,
    ) -> Dict[str, torch.Tensor]:
        if "expected_offset" not in current_evidence:
            raise RuntimeError("M5 template-correlation refine requires `expected_offset` from the state-carry projector.")
        return adapter(
            support_feat=template_support,
            main_feat=current_evidence["feat"],
            main_state=current_state,
            template_state=template_state,
            current_xy=current_xy,
            current_offsets=current_offsets,
            expected_offset=current_evidence["expected_offset"],
            layer_rank=layer_rank,
        )

    def _run_layerwise_statecarry_offset_refiner(
        self,
        *,
        refiner: CoTrackerDeltaOffsetRefiner,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        feat_layers: torch.Tensor,
        h_patches: int,
        w_patches: int,
        pred_centroid_xy: torch.Tensor,
        pred_offsets_xy: torch.Tensor,
        traj_cond: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if feat_layers.ndim != 5:
            raise ValueError(f"`feat_layers` must be [B,K,T,S,D], got {tuple(feat_layers.shape)}")
        if int(feat_layers.shape[1]) < 4:
            raise ValueError(
                "Layerwise state-carry refine expects at least 4 cached layers "
                f"(got shape={tuple(feat_layers.shape)})"
            )
        if self.traj_statecarry_projector is None:
            raise RuntimeError("Layerwise state-carry projector must be initialized for layerwise refine.")
        if self.traj_support_projector is None:
            raise RuntimeError("Layerwise state-carry refine requires `traj_support_projector` for centroid support descriptors.")

        base_xy = pred_xy
        base_conf = pred_conf
        base_centroid = pred_centroid_xy
        base_offsets = pred_offsets_xy
        current_xy = base_xy
        current_conf = base_conf
        current_centroid = base_centroid
        current_offsets = base_offsets
        current_state: Optional[torch.Tensor] = None
        template_state: Optional[torch.Tensor] = None
        stages: List[Dict[str, Any]] = []
        num_layers_total = int(feat_layers.shape[1])
        layer_ids = list(range(num_layers_total - 1, max(num_layers_total - 5, -1), -1))

        corr_entropy_all: List[torch.Tensor] = []
        corr_margin_all: List[torch.Tensor] = []
        corr_maxprob_all: List[torch.Tensor] = []
        context_entropy_all: List[torch.Tensor] = []
        context_margin_all: List[torch.Tensor] = []
        context_maxprob_all: List[torch.Tensor] = []
        context_offset_abs_all: List[torch.Tensor] = []
        context_shift_all: List[torch.Tensor] = []
        template_entropy_all: List[torch.Tensor] = []
        template_margin_all: List[torch.Tensor] = []
        template_maxprob_all: List[torch.Tensor] = []
        template_gate_all: List[torch.Tensor] = []
        template_offset_abs_all: List[torch.Tensor] = []
        template_delta_abs_all: List[torch.Tensor] = []
        centroid_stage_stack: List[torch.Tensor] = [base_centroid]
        for stage_idx, layer_idx in enumerate(layer_ids, start=1):
            feat_layer = feat_layers[:, layer_idx, ...]
            current_support = self._local_feat_support_stack_from_xy(
                current_xy,
                feat_layer,
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=1,
            )
            current_evidence = self.traj_statecarry_projector(
                current_support,
                prev_state=current_state,
                current_xy=current_xy,
                current_offsets=current_offsets,
                layer_rank=stage_idx - 1,
            )
            current_state = current_evidence["state"]
            context_pack: Optional[Dict[str, torch.Tensor]] = None
            template_pack: Optional[Dict[str, torch.Tensor]] = None
            current_local_feat = current_evidence["feat"]
            if self.traj_template_corr_adapter is not None:
                template_support_radius = self.traj_template_corr_adapter.support_radius_for_stage(
                    stage_idx=stage_idx,
                    num_stages=len(layer_ids),
                )
                if int(template_support_radius) == 1:
                    template_support = current_support
                else:
                    template_support = self._local_feat_support_stack_from_xy(
                        current_xy,
                        feat_layer,
                        h_patches=h_patches,
                        w_patches=w_patches,
                        support_radius=int(template_support_radius),
                    )
                if template_state is None:
                    template_state = self.traj_template_corr_adapter.initialize_template(template_support)
                if self.traj_template_corr_adapter.should_apply(stage_idx=stage_idx, num_stages=len(layer_ids)):
                    template_pack = self._run_layerwise_template_corr_adapter(
                        adapter=self.traj_template_corr_adapter,
                        template_support=template_support,
                        current_evidence=current_evidence,
                        current_state=current_state,
                        template_state=template_state,
                        current_xy=current_xy,
                        current_offsets=current_offsets,
                        layer_rank=stage_idx - 1,
                    )
                    current_local_feat = template_pack["feat"]
                    template_state = template_pack["next_template_state"]
                else:
                    template_state = self.traj_template_corr_adapter.update_template(
                        template_state=template_state,
                        support_feat=template_support,
                    )
            if self.traj_context_point_adapter is not None:
                context_pack = self._run_layerwise_context_point_adapter(
                    adapter=self.traj_context_point_adapter,
                    feat_layer=feat_layer,
                    current_support=current_support,
                    current_evidence=current_evidence,
                    current_state=current_state,
                    current_xy=current_xy,
                    current_offsets=current_offsets,
                    h_patches=h_patches,
                    w_patches=w_patches,
                )
                current_local_feat = context_pack["feat"]
            centroid_support = self._local_feat_support_stack_from_xy(
                current_centroid,
                feat_layer,
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=1,
            )
            current_centroid_local_feat = self.traj_support_projector(centroid_support)
            current_out = refiner(
                pred_xy=current_xy,
                pred_conf=current_conf,
                local_feat=current_local_feat,
                pred_centroid_xy=current_centroid,
                pred_offsets_xy=current_offsets,
                centroid_local_feat=current_centroid_local_feat,
                traj_cond=traj_cond,
            )
            centroid_update = self._refresh_layerwise_centroid(
                current_centroid=current_centroid,
                current_xy=current_out["pred_xy"],
                stage_idx=stage_idx,
                num_stages=len(layer_ids),
            )
            next_centroid = centroid_update["next_centroid_xy"]
            current_offsets_pre_refresh = current_out.get("pred_offsets_xy", current_out["pred_xy"] - current_centroid)
            if float(centroid_update["beta"]) <= 0.0:
                current_offsets_out = current_offsets_pre_refresh
            else:
                current_offsets_out = current_out["pred_xy"] - next_centroid
            current_delta_offsets = current_offsets_out - current_offsets
            current_delta_centroid = next_centroid - current_centroid
            corr_entropy_all.append(current_evidence["corr_entropy"])
            corr_margin_all.append(current_evidence["corr_margin"])
            corr_maxprob_all.append(current_evidence["max_prob"])
            if context_pack is not None:
                context_entropy_all.append(context_pack["context_entropy"])
                context_margin_all.append(context_pack["context_margin"])
                context_maxprob_all.append(context_pack["context_max_prob"])
                context_offset_abs_all.append(context_pack["context_offset_abs"])
                context_shift_all.append(context_pack["context_shift"])
            if template_pack is not None:
                template_entropy_all.append(template_pack["template_entropy"])
                template_margin_all.append(template_pack["template_margin"])
                template_maxprob_all.append(template_pack["template_max_prob"])
                template_gate_all.append(template_pack["template_gate"])
                template_offset_abs_all.append(template_pack["template_offset_abs"])
                template_delta_abs_all.append(template_pack["template_delta_abs"])
            stages.append(
                {
                    "out": current_out,
                    "local_feat": current_local_feat,
                    "base_local_feat": current_evidence["feat"],
                    "state": current_state,
                    "pred_xy": current_out["pred_xy"],
                    "pred_conf": current_out.get("pred_conf", current_conf),
                    "pred_centroid_xy": next_centroid,
                    "pred_offsets_xy": current_offsets_out,
                    "delta_centroid_xy": current_delta_centroid,
                    "delta_offsets_xy": current_delta_offsets,
                    "centroid_local_feat": current_centroid_local_feat,
                    "corr_entropy": current_evidence["corr_entropy"],
                    "corr_margin": current_evidence["corr_margin"],
                    "corr_max_prob": current_evidence["max_prob"],
                    "expected_offset": current_evidence["expected_offset"],
                    "centroid_beta": float(centroid_update["beta"]),
                    "proposed_centroid_xy": centroid_update["proposed_centroid_xy"],
                    "context_offsets_xy": (context_pack.get("context_offsets", None) if context_pack is not None else None),
                    "context_xy": (context_pack.get("context_xy", None) if context_pack is not None else None),
                    "context_entropy": (context_pack.get("context_entropy", None) if context_pack is not None else None),
                    "context_margin": (context_pack.get("context_margin", None) if context_pack is not None else None),
                    "context_max_prob": (context_pack.get("context_max_prob", None) if context_pack is not None else None),
                    "context_shift_xy": (context_pack.get("context_shift", None) if context_pack is not None else None),
                    "context_offset_abs": (context_pack.get("context_offset_abs", None) if context_pack is not None else None),
                    "template_entropy": (template_pack.get("template_entropy", None) if template_pack is not None else None),
                    "template_margin": (template_pack.get("template_margin", None) if template_pack is not None else None),
                    "template_max_prob": (template_pack.get("template_max_prob", None) if template_pack is not None else None),
                    "template_gate": (template_pack.get("template_gate", None) if template_pack is not None else None),
                    "template_offset_abs": (template_pack.get("template_offset_abs", None) if template_pack is not None else None),
                    "template_delta_abs": (template_pack.get("template_delta_abs", None) if template_pack is not None else None),
                    "layer_id": int(layer_idx),
                    "stage_idx": int(stage_idx),
                }
            )
            current_xy = stages[-1]["pred_xy"]
            current_conf = stages[-1]["pred_conf"]
            current_centroid = stages[-1]["pred_centroid_xy"]
            current_offsets = stages[-1]["pred_offsets_xy"]
            centroid_stage_stack.append(current_centroid)

        final_stage = stages[-1]
        corr_entropy = torch.stack(corr_entropy_all, dim=0).mean(dim=0)
        corr_margin = torch.stack(corr_margin_all, dim=0).mean(dim=0)
        corr_max_prob = torch.stack(corr_maxprob_all, dim=0).mean(dim=0)
        context_entropy = (
            torch.stack(context_entropy_all, dim=0).mean(dim=0)
            if context_entropy_all
            else None
        )
        context_margin = (
            torch.stack(context_margin_all, dim=0).mean(dim=0)
            if context_margin_all
            else None
        )
        context_max_prob = (
            torch.stack(context_maxprob_all, dim=0).mean(dim=0)
            if context_maxprob_all
            else None
        )
        context_offset_abs = (
            torch.stack(context_offset_abs_all, dim=0).mean(dim=0)
            if context_offset_abs_all
            else None
        )
        context_shift_xy = (
            torch.stack(context_shift_all, dim=0).mean(dim=0)
            if context_shift_all
            else None
        )
        template_entropy = (
            torch.stack(template_entropy_all, dim=0).mean(dim=0)
            if template_entropy_all
            else None
        )
        template_margin = (
            torch.stack(template_margin_all, dim=0).mean(dim=0)
            if template_margin_all
            else None
        )
        template_max_prob = (
            torch.stack(template_maxprob_all, dim=0).mean(dim=0)
            if template_maxprob_all
            else None
        )
        template_gate = (
            torch.stack(template_gate_all, dim=0).mean(dim=0)
            if template_gate_all
            else None
        )
        template_offset_abs = (
            torch.stack(template_offset_abs_all, dim=0).mean(dim=0)
            if template_offset_abs_all
            else None
        )
        template_delta_abs = (
            torch.stack(template_delta_abs_all, dim=0).mean(dim=0)
            if template_delta_abs_all
            else None
        )
        shape_debug: Optional[Dict[str, Any]] = None
        if self.debug_shapes:
            shape_debug = {
                "num_iters": len(stages),
                "layer_ids_by_stage": [int(stage["layer_id"]) for stage in stages],
                "local_feat_shapes": [list(stage["local_feat"].shape) for stage in stages],
                "base_local_feat_shapes": [list(stage["base_local_feat"].shape) for stage in stages],
                "state_shapes": [list(stage["state"].shape) for stage in stages],
                "centroid_local_feat_shapes": [list(stage["centroid_local_feat"].shape) for stage in stages],
                "centroid_beta_by_stage": [float(stage["centroid_beta"]) for stage in stages],
                "template_active_by_stage": [bool(torch.is_tensor(stage["template_entropy"])) for stage in stages],
                "context_xy_shapes": [
                    (None if stage["context_xy"] is None else list(stage["context_xy"].shape))
                    for stage in stages
                ],
                "iter_shape_debug": [stage["out"].get("shape_debug", None) for stage in stages],
            }

        out: Dict[str, Any] = {
            "stage0_xy": base_xy,
            "stage0_conf": base_conf,
            "stage0_centroid_xy": base_centroid,
            "stage0_offsets_xy": base_offsets,
            "stages": stages,
            "stage1_xy": stages[0]["pred_xy"],
            "stage1_conf": stages[0]["pred_conf"],
            "stage1_centroid_xy": stages[0]["pred_centroid_xy"],
            "stage1_offsets_xy": stages[0]["pred_offsets_xy"],
            "stage1_delta_offsets_xy": stages[0]["delta_offsets_xy"],
            "final_out": final_stage["out"],
            "final_xy": final_stage["pred_xy"],
            "final_conf": final_stage["pred_conf"],
            "final_centroid_xy": final_stage["pred_centroid_xy"],
            "final_offsets_xy": final_stage["pred_offsets_xy"],
            "centroid_stage_stack": torch.stack(centroid_stage_stack, dim=0),
            "total_delta_centroid_xy": final_stage["pred_centroid_xy"] - base_centroid,
            "final_delta_offsets_xy": final_stage["delta_offsets_xy"],
            "total_delta_xy": final_stage["pred_xy"] - base_xy,
            "total_delta_offsets_xy": final_stage["pred_offsets_xy"] - base_offsets,
            "corr_entropy": corr_entropy,
            "corr_margin": corr_margin,
            "corr_max_prob": corr_max_prob,
            "context_entropy": context_entropy,
            "context_margin": context_margin,
            "context_max_prob": context_max_prob,
            "context_offset_abs": context_offset_abs,
            "context_shift_xy": context_shift_xy,
            "template_entropy": template_entropy,
            "template_margin": template_margin,
            "template_max_prob": template_max_prob,
            "template_gate": template_gate,
            "template_offset_abs": template_offset_abs,
            "template_delta_abs": template_delta_abs,
            "shape_debug": shape_debug,
        }
        for idx, stage in enumerate(stages, start=1):
            out[f"stage{idx}_xy"] = stage["pred_xy"]
            out[f"stage{idx}_conf"] = stage["pred_conf"]
            out[f"stage{idx}_centroid_xy"] = stage["pred_centroid_xy"]
            out[f"stage{idx}_offsets_xy"] = stage["pred_offsets_xy"]
            out[f"stage{idx}_delta_centroid_xy"] = stage["delta_centroid_xy"]
            out[f"stage{idx}_delta_offsets_xy"] = stage["delta_offsets_xy"]
            out[f"corr_entropy_stage{idx}"] = stage["corr_entropy"]
            out[f"corr_margin_stage{idx}"] = stage["corr_margin"]
            out[f"corr_max_prob_stage{idx}"] = stage["corr_max_prob"]
            if torch.is_tensor(stage.get("context_entropy", None)):
                out[f"context_entropy_stage{idx}"] = stage["context_entropy"]
            if torch.is_tensor(stage.get("context_margin", None)):
                out[f"context_margin_stage{idx}"] = stage["context_margin"]
            if torch.is_tensor(stage.get("context_max_prob", None)):
                out[f"context_max_prob_stage{idx}"] = stage["context_max_prob"]
            if torch.is_tensor(stage.get("context_offset_abs", None)):
                out[f"context_offset_abs_stage{idx}"] = stage["context_offset_abs"]
            if torch.is_tensor(stage.get("template_entropy", None)):
                out[f"template_entropy_stage{idx}"] = stage["template_entropy"]
            if torch.is_tensor(stage.get("template_margin", None)):
                out[f"template_margin_stage{idx}"] = stage["template_margin"]
            if torch.is_tensor(stage.get("template_max_prob", None)):
                out[f"template_max_prob_stage{idx}"] = stage["template_max_prob"]
            if torch.is_tensor(stage.get("template_gate", None)):
                out[f"template_gate_stage{idx}"] = stage["template_gate"]
            if torch.is_tensor(stage.get("template_offset_abs", None)):
                out[f"template_offset_abs_stage{idx}"] = stage["template_offset_abs"]
            if torch.is_tensor(stage.get("template_delta_abs", None)):
                out[f"template_delta_abs_stage{idx}"] = stage["template_delta_abs"]
        return out

    def _run_singlelayer_statecarry_offset_refiner(
        self,
        *,
        refiner: CoTrackerDeltaOffsetRefiner,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        feat_flat: torch.Tensor,
        h_patches: int,
        w_patches: int,
        pred_centroid_xy: torch.Tensor,
        pred_offsets_xy: torch.Tensor,
        traj_cond: Optional[torch.Tensor],
        traj_tool_cond: Optional[torch.Tensor] = None,
        verifier_feat_flat: Optional[torch.Tensor] = None,
        apply_stage0_centroid_noise: bool = False,
    ) -> Dict[str, Any]:
        if feat_flat.ndim != 4:
            raise ValueError(f"`feat_flat` must be [B,T,S,D], got {tuple(feat_flat.shape)}")
        if self.traj_statecarry_projector is None:
            raise RuntimeError("Single-layer state-carry refine requires `traj_statecarry_projector`.")
        if self.traj_support_projector is None:
            raise RuntimeError("Single-layer state-carry refine requires `traj_support_projector`.")

        base_xy = pred_xy
        base_conf = pred_conf
        base_centroid = pred_centroid_xy
        base_offsets = pred_offsets_xy
        current_xy = base_xy
        current_conf = base_conf
        current_centroid = base_centroid
        current_offsets = base_offsets
        current_state: Optional[torch.Tensor] = None
        stages: List[Dict[str, Any]] = []
        corr_entropy_all: List[torch.Tensor] = []
        corr_margin_all: List[torch.Tensor] = []
        corr_maxprob_all: List[torch.Tensor] = []
        context_entropy_all: List[torch.Tensor] = []
        context_margin_all: List[torch.Tensor] = []
        context_maxprob_all: List[torch.Tensor] = []
        context_offset_abs_all: List[torch.Tensor] = []
        context_shift_all: List[torch.Tensor] = []
        context_gate_all: List[torch.Tensor] = []
        belief_gate_all: List[torch.Tensor] = []
        factor_anchor_gate_all: List[torch.Tensor] = []
        factor_offset_gate_all: List[torch.Tensor] = []
        factor_delta_offset_abs_all: List[torch.Tensor] = []
        joint_gate_all: List[torch.Tensor] = []
        joint_pair_dist_all: List[torch.Tensor] = []
        anchor_delta_centroid_abs_all: List[torch.Tensor] = []
        anchor_alpha_all: List[torch.Tensor] = []
        anchor_verify_scale_all: List[torch.Tensor] = []
        anchor_verify_residual_all: List[torch.Tensor] = []
        stage0_centroid_noise_abs_all: List[torch.Tensor] = []
        centroid_stage_stack: List[torch.Tensor] = [base_centroid]
        num_stages = 5 if self.traj_refiner_mode == "s3p3x_singlelayer_statecarry_refine_plus1" else 4
        prev_delta_xy = current_xy.new_zeros(current_xy.shape)

        for stage_idx in range(1, num_stages + 1):
            current_xy_in = current_xy
            current_centroid_in = current_centroid
            current_offsets_in = current_offsets
            stage0_centroid_noise_abs = current_centroid.new_zeros(current_centroid.shape[:3])
            if stage_idx == 1 and apply_stage0_centroid_noise:
                current_centroid_in, stage0_centroid_noise_abs = self._apply_stage0_centroid_carry_noise(current_centroid)
                current_xy_in = (
                    current_centroid_in.float() + current_offsets_in.float()
                ).clamp(0.0, 1.0).to(dtype=current_xy.dtype)
            current_support = self._local_feat_support_stack_from_xy(
                current_xy_in,
                feat_flat,
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=int(self.traj_support_radius),
            )
            current_evidence = self.traj_statecarry_projector(
                current_support,
                prev_state=current_state,
                current_xy=current_xy_in,
                current_offsets=current_offsets_in,
                layer_rank=stage_idx - 1,
                prev_delta_xy=prev_delta_xy,
            )
            current_state = current_evidence["state"]
            current_local_feat = current_evidence["feat"]
            context_pack: Optional[Dict[str, torch.Tensor]] = None
            context_gate_tensor: Optional[torch.Tensor] = None
            uses_late_gated_context = self.traj_refiner_mode in (
                "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
            )
            uses_point_gated_context = (
                self.traj_refiner_mode in (
                    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                )
            )
            context_active = bool(self.traj_context_point_adapter is not None)
            if context_active and uses_late_gated_context:
                context_active = stage_idx > (num_stages - int(self.traj_context_late_stage_count))
            if context_active and self.traj_context_point_adapter is not None:
                context_pack = self._run_layerwise_context_point_adapter(
                    adapter=self.traj_context_point_adapter,
                    feat_layer=feat_flat,
                    current_support=current_support,
                    current_evidence=current_evidence,
                    current_state=current_state,
                    current_xy=current_xy_in,
                    current_offsets=current_offsets_in,
                    h_patches=h_patches,
                    w_patches=w_patches,
                )
                if uses_late_gated_context:
                    if self.traj_context_stage_gate is None:
                        raise RuntimeError("S3.6 requires `traj_context_stage_gate`.")
                    stage_gate = torch.tanh(self.traj_context_stage_gate[stage_idx - 1]).to(
                        device=current_evidence["feat"].device,
                        dtype=current_evidence["feat"].dtype,
                    )
                    if uses_point_gated_context:
                        if self.traj_context_point_gate is None:
                            raise RuntimeError("S3.6a requires `traj_context_point_gate`.")
                        context_shift_abs = context_pack["context_shift"].float().norm(dim=-1)
                        expected_offset_abs = current_evidence["expected_offset"].float().norm(dim=-1)
                        point_gate_in = torch.stack(
                            [
                                context_pack["context_entropy"].float(),
                                context_pack["context_margin"].float(),
                                context_pack["context_max_prob"].float(),
                                context_shift_abs,
                                expected_offset_abs,
                                current_offsets_in.float().norm(dim=-1),
                            ],
                            dim=-1,
                        )
                        point_gate = torch.sigmoid(self.traj_context_point_gate(point_gate_in)).squeeze(-1)
                        effective_gate = stage_gate * point_gate
                    else:
                        effective_gate = current_evidence["corr_entropy"].new_full(
                            current_evidence["corr_entropy"].shape,
                            float(stage_gate.detach().item()),
                        )
                    current_local_feat = current_evidence["feat"] + effective_gate.unsqueeze(-1) * (
                        context_pack["feat"] - current_evidence["feat"]
                    )
                    context_gate_tensor = effective_gate
                else:
                    current_local_feat = context_pack["feat"]
                    context_gate_tensor = current_evidence["corr_entropy"].new_ones(
                        current_evidence["corr_entropy"].shape
                    )
            joint_gate_tensor: Optional[torch.Tensor] = None
            if (
                self.traj_joint_point_block is not None
                and self.traj_joint_stage_gate is not None
                and stage_idx > (num_stages - int(self.traj_joint_late_stage_count))
            ):
                joint_pack = self.traj_joint_point_block(
                    point_feat=current_local_feat,
                    current_xy=current_xy_in,
                    current_offsets=current_offsets_in,
                    expected_offset=current_evidence["expected_offset"],
                    current_centroid=current_centroid_in,
                )
                joint_stage_gate = torch.tanh(self.traj_joint_stage_gate[stage_idx - 1]).to(
                    device=current_local_feat.device,
                    dtype=current_local_feat.dtype,
                )
                joint_gate_tensor = current_evidence["corr_entropy"].new_full(
                    current_evidence["corr_entropy"].shape,
                    float(joint_stage_gate.detach().item()),
                )
                current_local_feat = current_local_feat + joint_gate_tensor.unsqueeze(-1) * joint_pack["delta_feat"]
            incoming_centroid = current_centroid_in
            incoming_offsets = current_offsets_in
            centroid_support = self._local_feat_support_stack_from_xy(
                incoming_centroid,
                feat_flat,
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=int(self.traj_support_radius),
            )
            current_centroid_local_feat = self.traj_support_projector(centroid_support)
            anchor_delta_centroid_xy = incoming_centroid.new_zeros(incoming_centroid.shape)
            factor_delta_offsets_xy = incoming_offsets.new_zeros(incoming_offsets.shape)
            factor_anchor_gate = incoming_centroid.new_zeros(incoming_centroid.shape[:3])
            factor_offset_gate = incoming_centroid.new_zeros(incoming_centroid.shape[:3])
            anchor_verify_scale = incoming_centroid.new_ones(incoming_centroid.shape[:3])
            anchor_verify_residual = incoming_centroid.new_zeros(incoming_centroid.shape[:3])
            anchor_alpha_value = (
                float(self._anchor_activation_alpha()) if self.traj_centroid_anchor_adapter is not None else 0.0
            )
            anchor_alpha_tensor = incoming_centroid.new_full(incoming_centroid.shape[:3], anchor_alpha_value)
            refiner_centroid = incoming_centroid
            refiner_offsets = incoming_offsets
            if self.traj_factorized_tool_adapter is not None:
                factor_pack = self.traj_factorized_tool_adapter(
                    current_centroid_xy=incoming_centroid.detach(),
                    centroid_local_feat=current_centroid_local_feat.detach(),
                    point_state=current_state.detach(),
                    point_feat=current_local_feat.detach(),
                    current_offsets=incoming_offsets.detach(),
                    expected_offset=current_evidence["expected_offset"].detach(),
                    corr_entropy=current_evidence["corr_entropy"].detach(),
                    corr_margin=current_evidence["corr_margin"].detach(),
                    corr_max_prob=current_evidence["max_prob"].detach(),
                    traj_tool_cond=(traj_tool_cond.detach() if torch.is_tensor(traj_tool_cond) else traj_tool_cond),
                )
                anchor_delta_centroid_xy = factor_pack["delta_centroid_xy"].to(
                    device=incoming_centroid.device,
                    dtype=incoming_centroid.dtype,
                )
                factor_delta_offsets_xy = factor_pack["delta_offsets_xy"].to(
                    device=incoming_offsets.device,
                    dtype=incoming_offsets.dtype,
                )
                factor_anchor_gate = factor_pack["anchor_gate"].to(
                    device=incoming_centroid.device,
                    dtype=incoming_centroid.dtype,
                )
                factor_offset_gate = factor_pack["offset_gate"].to(
                    device=incoming_centroid.device,
                    dtype=incoming_centroid.dtype,
                )
                refiner_centroid = (
                    incoming_centroid.float() + anchor_delta_centroid_xy.float()
                ).clamp(0.0, 1.0).to(dtype=incoming_centroid.dtype)
                refiner_offsets = (current_xy_in - refiner_centroid) + factor_delta_offsets_xy
                centroid_support = self._local_feat_support_stack_from_xy(
                    refiner_centroid,
                    feat_flat,
                    h_patches=h_patches,
                    w_patches=w_patches,
                    support_radius=int(self.traj_support_radius),
                )
                current_centroid_local_feat = self.traj_support_projector(centroid_support)
            elif self.traj_centroid_anchor_adapter is not None:
                detach_anchor_inputs = self.traj_refiner_mode in (
                    "s3p9d_singlelayer_statecarry_anchoronly_detached",
                    "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                )
                if anchor_alpha_value > 0.0:
                    anchor_pack = self.traj_centroid_anchor_adapter(
                        current_centroid_xy=(incoming_centroid.detach() if detach_anchor_inputs else incoming_centroid),
                        centroid_local_feat=(
                            current_centroid_local_feat.detach()
                            if detach_anchor_inputs
                            else current_centroid_local_feat
                        ),
                        point_state=(current_state.detach() if detach_anchor_inputs else current_state),
                        point_feat=(current_local_feat.detach() if detach_anchor_inputs else current_local_feat),
                        current_offsets=(incoming_offsets.detach() if detach_anchor_inputs else incoming_offsets),
                        expected_offset=(
                            current_evidence["expected_offset"].detach()
                            if detach_anchor_inputs
                            else current_evidence["expected_offset"]
                        ),
                        corr_entropy=(
                            current_evidence["corr_entropy"].detach()
                            if detach_anchor_inputs
                            else current_evidence["corr_entropy"]
                        ),
                        corr_margin=(
                            current_evidence["corr_margin"].detach()
                            if detach_anchor_inputs
                            else current_evidence["corr_margin"]
                        ),
                        corr_max_prob=(
                            current_evidence["max_prob"].detach()
                            if detach_anchor_inputs
                            else current_evidence["max_prob"]
                        ),
                    )
                    anchor_delta_centroid_xy = anchor_pack["delta_centroid_xy"].to(
                        device=incoming_centroid.device,
                        dtype=incoming_centroid.dtype,
                    )
                    if anchor_alpha_value < 1.0:
                        anchor_delta_centroid_xy = anchor_delta_centroid_xy * float(anchor_alpha_value)
                    if self.traj_shallow_anchor_verifier is not None:
                        if verifier_feat_flat is None:
                            raise RuntimeError(
                                "S3.9f shallow anchor verification requires L18 verifier features."
                            )
                        current_shallow_support = self._local_feat_support_stack_from_xy(
                            incoming_centroid,
                            verifier_feat_flat,
                            h_patches=h_patches,
                            w_patches=w_patches,
                            support_radius=int(self.traj_support_radius),
                        )
                        proposed_centroid_for_verify = (
                            incoming_centroid.float() + anchor_delta_centroid_xy.float()
                        ).clamp(0.0, 1.0).to(dtype=incoming_centroid.dtype)
                        proposed_shallow_support = self._local_feat_support_stack_from_xy(
                            proposed_centroid_for_verify,
                            verifier_feat_flat,
                            h_patches=h_patches,
                            w_patches=w_patches,
                            support_radius=int(self.traj_support_radius),
                        )
                        verify_pack = self.traj_shallow_anchor_verifier(
                            current_support=current_shallow_support.detach(),
                            proposed_support=proposed_shallow_support.detach(),
                            proposed_delta_xy=anchor_delta_centroid_xy.detach(),
                        )
                        anchor_verify_scale = verify_pack["scale"].to(
                            device=incoming_centroid.device,
                            dtype=incoming_centroid.dtype,
                        )
                        anchor_verify_residual = verify_pack["scale_residual"].to(
                            device=incoming_centroid.device,
                            dtype=incoming_centroid.dtype,
                        )
                        anchor_delta_centroid_xy = anchor_delta_centroid_xy * anchor_verify_scale.unsqueeze(-1)
                    refiner_centroid = (
                        incoming_centroid.float() + anchor_delta_centroid_xy.float()
                    ).clamp(0.0, 1.0).to(dtype=incoming_centroid.dtype)
                    refiner_offsets = current_xy_in - refiner_centroid
                    centroid_support = self._local_feat_support_stack_from_xy(
                        refiner_centroid,
                        feat_flat,
                        h_patches=h_patches,
                        w_patches=w_patches,
                        support_radius=int(self.traj_support_radius),
                    )
                    current_centroid_local_feat = self.traj_support_projector(centroid_support)
            current_out = refiner(
                pred_xy=current_xy_in,
                pred_conf=current_conf,
                local_feat=current_local_feat,
                pred_centroid_xy=refiner_centroid,
                pred_offsets_xy=refiner_offsets,
                centroid_local_feat=current_centroid_local_feat,
                traj_cond=traj_cond,
            )
            centroid_update = self._refresh_layerwise_centroid(
                current_centroid=refiner_centroid,
                current_xy=current_out["pred_xy"],
                stage_idx=stage_idx,
                num_stages=num_stages,
            )
            next_centroid = centroid_update["next_centroid_xy"]
            current_offsets_pre_refresh = current_out.get("pred_offsets_xy", current_out["pred_xy"] - refiner_centroid)
            if float(centroid_update["beta"]) <= 0.0:
                current_offsets_out = current_offsets_pre_refresh
            else:
                current_offsets_out = current_out["pred_xy"] - next_centroid
            current_delta_offsets = current_offsets_out - incoming_offsets
            current_delta_centroid = next_centroid - incoming_centroid
            corr_entropy_all.append(current_evidence["corr_entropy"])
            corr_margin_all.append(current_evidence["corr_margin"])
            corr_maxprob_all.append(current_evidence["max_prob"])
            if torch.is_tensor(current_evidence.get("belief_gate", None)):
                belief_gate_all.append(current_evidence["belief_gate"])
            if self.traj_factorized_tool_adapter is not None:
                factor_anchor_gate_all.append(factor_anchor_gate)
                factor_offset_gate_all.append(factor_offset_gate)
                factor_delta_offset_abs_all.append(factor_delta_offsets_xy.float().norm(dim=-1))
            anchor_delta_centroid_abs_all.append(anchor_delta_centroid_xy.float().norm(dim=-1))
            anchor_alpha_all.append(anchor_alpha_tensor)
            anchor_verify_scale_all.append(anchor_verify_scale)
            anchor_verify_residual_all.append(anchor_verify_residual)
            stage0_centroid_noise_abs_all.append(stage0_centroid_noise_abs)
            if context_pack is not None:
                context_entropy_all.append(context_pack["context_entropy"])
                context_margin_all.append(context_pack["context_margin"])
                context_maxprob_all.append(context_pack["context_max_prob"])
                context_offset_abs_all.append(context_pack["context_offset_abs"])
                context_shift_all.append(context_pack["context_shift"])
                if context_gate_tensor is not None:
                    context_gate_all.append(context_gate_tensor)
            if joint_gate_tensor is not None:
                joint_gate_all.append(joint_gate_tensor)
                joint_pair_dist_all.append(joint_pack["pair_dist_mean"])
            stages.append(
                {
                    "out": current_out,
                    "local_feat": current_local_feat,
                    "base_local_feat": current_evidence["feat"],
                    "state": current_state,
                    "pred_xy": current_out["pred_xy"],
                    "pred_conf": current_out.get("pred_conf", current_conf),
                    "pred_centroid_xy": next_centroid,
                    "pred_offsets_xy": current_offsets_out,
                    "anchor_delta_centroid_xy": anchor_delta_centroid_xy,
                    "anchor_delta_centroid_abs": anchor_delta_centroid_xy.float().norm(dim=-1),
                    "anchor_alpha": anchor_alpha_tensor,
                    "anchor_verify_scale": anchor_verify_scale,
                    "anchor_verify_residual": anchor_verify_residual,
                    "stage0_centroid_noise_abs": stage0_centroid_noise_abs,
                    "factor_anchor_gate": factor_anchor_gate,
                    "factor_offset_gate": factor_offset_gate,
                    "factor_delta_offset_abs": factor_delta_offsets_xy.float().norm(dim=-1),
                    "delta_centroid_xy": current_delta_centroid,
                    "delta_offsets_xy": current_delta_offsets,
                    "centroid_local_feat": current_centroid_local_feat,
                    "corr_entropy": current_evidence["corr_entropy"],
                    "corr_margin": current_evidence["corr_margin"],
                    "corr_max_prob": current_evidence["max_prob"],
                    "expected_offset": current_evidence["expected_offset"],
                    "centroid_beta": float(centroid_update["beta"]),
                    "proposed_centroid_xy": centroid_update["proposed_centroid_xy"],
                    "context_offsets_xy": (context_pack.get("context_offsets", None) if context_pack is not None else None),
                    "context_xy": (context_pack.get("context_xy", None) if context_pack is not None else None),
                    "context_entropy": (context_pack.get("context_entropy", None) if context_pack is not None else None),
                    "context_margin": (context_pack.get("context_margin", None) if context_pack is not None else None),
                    "context_max_prob": (context_pack.get("context_max_prob", None) if context_pack is not None else None),
                    "context_shift_xy": (context_pack.get("context_shift", None) if context_pack is not None else None),
                    "context_offset_abs": (context_pack.get("context_offset_abs", None) if context_pack is not None else None),
                    "context_gate": context_gate_tensor,
                    "belief_gate": current_evidence.get("belief_gate", None),
                    "joint_gate": joint_gate_tensor,
                    "joint_pair_dist": (joint_pack.get("pair_dist_mean", None) if joint_gate_tensor is not None else None),
                    "layer_id": -1,
                    "stage_idx": int(stage_idx),
                }
            )
            prev_delta_xy = stages[-1]["pred_xy"] - current_xy_in
            current_xy = stages[-1]["pred_xy"]
            current_conf = stages[-1]["pred_conf"]
            current_centroid = stages[-1]["pred_centroid_xy"]
            current_offsets = stages[-1]["pred_offsets_xy"]
            centroid_stage_stack.append(current_centroid)

        final_stage = stages[-1]
        corr_entropy = torch.stack(corr_entropy_all, dim=0).mean(dim=0)
        corr_margin = torch.stack(corr_margin_all, dim=0).mean(dim=0)
        corr_max_prob = torch.stack(corr_maxprob_all, dim=0).mean(dim=0)
        context_entropy = (
            torch.stack(context_entropy_all, dim=0).mean(dim=0)
            if context_entropy_all
            else None
        )
        context_margin = (
            torch.stack(context_margin_all, dim=0).mean(dim=0)
            if context_margin_all
            else None
        )
        context_max_prob = (
            torch.stack(context_maxprob_all, dim=0).mean(dim=0)
            if context_maxprob_all
            else None
        )
        context_offset_abs = (
            torch.stack(context_offset_abs_all, dim=0).mean(dim=0)
            if context_offset_abs_all
            else None
        )
        context_shift_xy = (
            torch.stack(context_shift_all, dim=0).mean(dim=0)
            if context_shift_all
            else None
        )
        context_gate = (
            torch.stack(context_gate_all, dim=0).mean(dim=0)
            if context_gate_all
            else None
        )
        belief_gate = (
            torch.stack(belief_gate_all, dim=0).mean(dim=0)
            if belief_gate_all
            else None
        )
        factor_anchor_gate = (
            torch.stack(factor_anchor_gate_all, dim=0).mean(dim=0)
            if factor_anchor_gate_all
            else None
        )
        factor_offset_gate = (
            torch.stack(factor_offset_gate_all, dim=0).mean(dim=0)
            if factor_offset_gate_all
            else None
        )
        factor_delta_offset_abs = (
            torch.stack(factor_delta_offset_abs_all, dim=0).mean(dim=0)
            if factor_delta_offset_abs_all
            else None
        )
        anchor_delta_centroid_abs = (
            torch.stack(anchor_delta_centroid_abs_all, dim=0).mean(dim=0)
            if anchor_delta_centroid_abs_all
            else None
        )
        anchor_alpha = (
            torch.stack(anchor_alpha_all, dim=0).mean(dim=0)
            if anchor_alpha_all
            else None
        )
        anchor_verify_scale = (
            torch.stack(anchor_verify_scale_all, dim=0).mean(dim=0)
            if anchor_verify_scale_all
            else None
        )
        anchor_verify_residual = (
            torch.stack(anchor_verify_residual_all, dim=0).mean(dim=0)
            if anchor_verify_residual_all
            else None
        )
        stage0_centroid_noise_abs = (
            stage0_centroid_noise_abs_all[0]
            if stage0_centroid_noise_abs_all
            else None
        )
        joint_gate = (
            torch.stack(joint_gate_all, dim=0).mean(dim=0)
            if joint_gate_all
            else None
        )
        joint_pair_dist = (
            torch.stack(joint_pair_dist_all, dim=0).mean(dim=0)
            if joint_pair_dist_all
            else None
        )
        shape_debug: Optional[Dict[str, Any]] = None
        if self.debug_shapes:
            shape_debug = {
                "num_iters": len(stages),
                "single_layer_repeated": True,
                "layer_ids_by_stage": [int(stage["layer_id"]) for stage in stages],
                "local_feat_shapes": [list(stage["local_feat"].shape) for stage in stages],
                "base_local_feat_shapes": [list(stage["base_local_feat"].shape) for stage in stages],
                "state_shapes": [list(stage["state"].shape) for stage in stages],
                "centroid_local_feat_shapes": [list(stage["centroid_local_feat"].shape) for stage in stages],
                "centroid_beta_by_stage": [float(stage["centroid_beta"]) for stage in stages],
                "uses_shallow_anchor_verifier": bool(self.traj_shallow_anchor_verifier is not None),
                "uses_context_point_adapter": bool(self.traj_context_point_adapter is not None),
                "context_active_stages": [
                    int(stage["stage_idx"])
                    for stage in stages
                    if torch.is_tensor(stage.get("context_gate", None))
                ],
                "joint_active_stages": [
                    int(stage["stage_idx"])
                    for stage in stages
                    if torch.is_tensor(stage.get("joint_gate", None))
                ],
                "iter_shape_debug": [stage["out"].get("shape_debug", None) for stage in stages],
            }

        out: Dict[str, Any] = {
            "stage0_xy": base_xy,
            "stage0_conf": base_conf,
            "stage0_centroid_xy": base_centroid,
            "stage0_offsets_xy": base_offsets,
            "stages": stages,
            "stage1_xy": stages[0]["pred_xy"],
            "stage1_conf": stages[0]["pred_conf"],
            "stage1_centroid_xy": stages[0]["pred_centroid_xy"],
            "stage1_offsets_xy": stages[0]["pred_offsets_xy"],
            "stage1_delta_offsets_xy": stages[0]["delta_offsets_xy"],
            "final_out": final_stage["out"],
            "final_xy": final_stage["pred_xy"],
            "final_conf": final_stage["pred_conf"],
            "final_centroid_xy": final_stage["pred_centroid_xy"],
            "final_offsets_xy": final_stage["pred_offsets_xy"],
            "centroid_stage_stack": torch.stack(centroid_stage_stack, dim=0),
            "total_delta_centroid_xy": final_stage["pred_centroid_xy"] - base_centroid,
            "final_delta_offsets_xy": final_stage["delta_offsets_xy"],
            "total_delta_xy": final_stage["pred_xy"] - base_xy,
            "total_delta_offsets_xy": final_stage["pred_offsets_xy"] - base_offsets,
            "corr_entropy": corr_entropy,
            "corr_margin": corr_margin,
            "corr_max_prob": corr_max_prob,
            "context_entropy": context_entropy,
            "context_margin": context_margin,
            "context_max_prob": context_max_prob,
            "context_offset_abs": context_offset_abs,
            "context_shift_xy": context_shift_xy,
            "context_gate": context_gate,
            "belief_gate": belief_gate,
            "factor_anchor_gate": factor_anchor_gate,
            "factor_offset_gate": factor_offset_gate,
            "factor_delta_offset_abs": factor_delta_offset_abs,
            "anchor_delta_centroid_abs": anchor_delta_centroid_abs,
            "anchor_alpha": anchor_alpha,
            "anchor_verify_scale": anchor_verify_scale,
            "anchor_verify_residual": anchor_verify_residual,
            "stage0_centroid_noise_abs": stage0_centroid_noise_abs,
            "joint_gate": joint_gate,
            "joint_pair_dist": joint_pair_dist,
            "template_entropy": None,
            "template_margin": None,
            "template_max_prob": None,
            "template_gate": None,
            "template_offset_abs": None,
            "template_delta_abs": None,
            "shape_debug": shape_debug,
        }
        for idx, stage in enumerate(stages, start=1):
            out[f"stage{idx}_xy"] = stage["pred_xy"]
            out[f"stage{idx}_conf"] = stage["pred_conf"]
            out[f"stage{idx}_centroid_xy"] = stage["pred_centroid_xy"]
            out[f"stage{idx}_offsets_xy"] = stage["pred_offsets_xy"]
            out[f"stage{idx}_delta_centroid_xy"] = stage["delta_centroid_xy"]
            out[f"stage{idx}_delta_offsets_xy"] = stage["delta_offsets_xy"]
            out[f"corr_entropy_stage{idx}"] = stage["corr_entropy"]
            out[f"corr_margin_stage{idx}"] = stage["corr_margin"]
            out[f"corr_max_prob_stage{idx}"] = stage["corr_max_prob"]
            out[f"context_entropy_stage{idx}"] = stage["context_entropy"]
            out[f"context_margin_stage{idx}"] = stage["context_margin"]
            out[f"context_max_prob_stage{idx}"] = stage["context_max_prob"]
            out[f"context_offset_abs_stage{idx}"] = stage["context_offset_abs"]
            out[f"context_gate_stage{idx}"] = stage["context_gate"]
            out[f"belief_gate_stage{idx}"] = stage["belief_gate"]
            out[f"factor_anchor_gate_stage{idx}"] = stage["factor_anchor_gate"]
            out[f"factor_offset_gate_stage{idx}"] = stage["factor_offset_gate"]
            out[f"factor_delta_offset_abs_stage{idx}"] = stage["factor_delta_offset_abs"]
            out[f"anchor_delta_centroid_abs_stage{idx}"] = stage["anchor_delta_centroid_abs"]
            out[f"anchor_alpha_stage{idx}"] = stage["anchor_alpha"]
            out[f"anchor_verify_scale_stage{idx}"] = stage["anchor_verify_scale"]
            out[f"anchor_verify_residual_stage{idx}"] = stage["anchor_verify_residual"]
            out[f"stage0_centroid_noise_abs_stage{idx}"] = stage["stage0_centroid_noise_abs"]
            out[f"joint_gate_stage{idx}"] = stage["joint_gate"]
        return out

    @staticmethod
    def _trajectory_condition_from_head(
        head: Optional[nn.Module],
        *,
        ref_tensor: torch.Tensor,
        tool_id: Optional[torch.Tensor],
        action_id: Optional[torch.Tensor],
        combo_id: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if head is None:
            return None
        tool_embed = getattr(head, "tool_embed", None)
        action_embed = getattr(head, "action_embed", None)
        combo_embed = getattr(head, "combo_embed", None)
        condition_mode = getattr(head, "condition_mode", None)
        num_tool_classes = int(getattr(head, "num_tool_classes", 0))
        num_action_classes = int(getattr(head, "num_action_classes", 0))
        num_combo_classes = int(getattr(head, "num_combo_classes", 0))
        if tool_embed is None or condition_mode is None:
            return None
        return _build_trajectory_condition(
            condition_mode=str(condition_mode),
            ref_tensor=ref_tensor,
            tool_embed=tool_embed,
            num_tool_classes=num_tool_classes,
            tool_id=tool_id,
            action_embed=action_embed,
            num_action_classes=num_action_classes,
            action_id=action_id,
            combo_embed=combo_embed,
            num_combo_classes=num_combo_classes,
            combo_id=combo_id,
        )

    @staticmethod
    def _trajectory_tool_condition_from_head(
        head: Optional[nn.Module],
        *,
        ref_tensor: torch.Tensor,
        tool_id: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if head is None:
            return None
        tool_embed = getattr(head, "tool_embed", None)
        num_tool_classes = int(getattr(head, "num_tool_classes", 0))
        if tool_embed is None or num_tool_classes <= 0:
            return None
        return _build_trajectory_condition(
            condition_mode="tool",
            ref_tensor=ref_tensor,
            tool_embed=tool_embed,
            num_tool_classes=num_tool_classes,
            tool_id=tool_id,
        )

    def _run_cotracker_offset_refiner(
        self,
        *,
        refiner: CoTrackerDeltaOffsetRefiner,
        pred_xy: torch.Tensor,
        pred_conf: torch.Tensor,
        feat_flat: torch.Tensor,
        h_patches: int,
        w_patches: int,
        pred_centroid_xy: torch.Tensor,
        pred_offsets_xy: torch.Tensor,
        centroid_local_feat: Optional[torch.Tensor],
        traj_cond: Optional[torch.Tensor],
        num_iters: int,
        feat_flats_by_iter: Optional[List[torch.Tensor]] = None,
        centroid_local_feats_by_iter: Optional[List[Optional[torch.Tensor]]] = None,
        iter_layer_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if num_iters <= 0:
            raise ValueError(f"`num_iters` must be > 0, got {num_iters}")
        if feat_flats_by_iter is not None and len(feat_flats_by_iter) != int(num_iters):
            raise ValueError(
                "`feat_flats_by_iter` must have the same length as `num_iters`, "
                f"got len={len(feat_flats_by_iter)} num_iters={int(num_iters)}"
            )
        if centroid_local_feats_by_iter is not None and len(centroid_local_feats_by_iter) != int(num_iters):
            raise ValueError(
                "`centroid_local_feats_by_iter` must have the same length as `num_iters`, "
                f"got len={len(centroid_local_feats_by_iter)} num_iters={int(num_iters)}"
            )
        if iter_layer_ids is not None and len(iter_layer_ids) != int(num_iters):
            raise ValueError(
                "`iter_layer_ids` must have the same length as `num_iters`, "
                f"got len={len(iter_layer_ids)} num_iters={int(num_iters)}"
            )

        base_xy = pred_xy
        base_conf = pred_conf
        base_offsets = pred_offsets_xy
        current_xy = base_xy
        current_conf = base_conf
        current_offsets = base_offsets
        stages: List[Dict[str, Any]] = []
        template_center_feat: Optional[torch.Tensor] = None
        template_support_feat: Optional[torch.Tensor] = None

        use_s4a_centercorr = (
            self.traj_refiner_mode == "s4a_template_centercorr_3x3"
            and self.traj_support_corr_projector is not None
        )
        use_s4b_allpaircorr = (
            self.traj_refiner_mode == "s4b_template_allpaircorr_3x3"
            and self.traj_support_allpair_corr_projector is not None
        )
        if use_s4a_centercorr:
            template_support = self._local_feat_support_stack_from_xy(
                base_xy[:, :1, :, :].detach(),
                feat_flat[:, :1, :, :],
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=1,
            )
            template_center_feat = template_support[:, 0, :, template_support.shape[3] // 2, :].contiguous()
        if use_s4b_allpaircorr:
            template_support = self._local_feat_support_stack_from_xy(
                base_xy[:, :1, :, :].detach(),
                feat_flat[:, :1, :, :],
                h_patches=h_patches,
                w_patches=w_patches,
                support_radius=1,
            )
            template_support_feat = template_support[:, 0, :, :, :].contiguous()

        for iter_idx in range(int(num_iters)):
            feat_flat_iter = feat_flats_by_iter[iter_idx] if feat_flats_by_iter is not None else feat_flat
            centroid_local_feat_iter = (
                centroid_local_feats_by_iter[iter_idx]
                if centroid_local_feats_by_iter is not None
                else centroid_local_feat
            )
            corr_pack: Optional[Dict[str, torch.Tensor]] = None
            if use_s4a_centercorr:
                current_support = self._local_feat_support_stack_from_xy(
                    current_xy,
                    feat_flat_iter,
                    h_patches=h_patches,
                    w_patches=w_patches,
                    support_radius=1,
                )
                corr_pack = self.traj_support_corr_projector(
                    current_support,
                    template_center_feat,
                )
                current_local_feat = corr_pack["feat"]
            elif use_s4b_allpaircorr:
                current_support = self._local_feat_support_stack_from_xy(
                    current_xy,
                    feat_flat_iter,
                    h_patches=h_patches,
                    w_patches=w_patches,
                    support_radius=1,
                )
                corr_pack = self.traj_support_allpair_corr_projector(
                    current_support,
                    template_support_feat,
                )
                current_local_feat = corr_pack["feat"]
            else:
                current_local_feat = self._trajectory_local_feat_from_xy(
                    current_xy,
                    feat_flat_iter,
                    h_patches=h_patches,
                    w_patches=w_patches,
                )
            current_out = refiner(
                pred_xy=current_xy,
                pred_conf=current_conf,
                local_feat=current_local_feat,
                pred_centroid_xy=pred_centroid_xy,
                pred_offsets_xy=current_offsets,
                centroid_local_feat=centroid_local_feat_iter,
                traj_cond=traj_cond,
            )
            current_centroid_out = current_out.get("pred_centroid_xy", pred_centroid_xy)
            current_offsets_out = current_out.get("pred_offsets_xy", current_out["pred_xy"] - pred_centroid_xy)
            current_delta_offsets = current_out.get("delta_offsets_xy", current_offsets_out - current_offsets)
            stages.append(
                {
                    "out": current_out,
                    "local_feat": current_local_feat,
                    "pred_xy": current_out["pred_xy"],
                    "pred_conf": current_out.get("pred_conf", current_conf),
                    "pred_centroid_xy": current_centroid_out,
                    "pred_offsets_xy": current_offsets_out,
                    "delta_centroid_xy": current_centroid_out - pred_centroid_xy,
                    "delta_offsets_xy": current_delta_offsets,
                    "corr_entropy": (corr_pack.get("corr_entropy", None) if corr_pack is not None else None),
                    "corr_margin": (corr_pack.get("corr_margin", None) if corr_pack is not None else None),
                }
            )
            current_xy = stages[-1]["pred_xy"]
            current_conf = stages[-1]["pred_conf"]
            current_offsets = stages[-1]["pred_offsets_xy"]

        final_stage = stages[-1]
        final_centroid = final_stage.get("pred_centroid_xy", pred_centroid_xy)
        final_out = dict(final_stage["out"])
        final_out.setdefault("pred_centroid_xy", final_centroid)
        shape_debug: Optional[Dict[str, Any]] = None
        if self.debug_shapes:
            shape_debug = {
                "num_iters": int(num_iters),
                "local_feat_shapes": [list(stage["local_feat"].shape) for stage in stages],
                "iter_shape_debug": [stage["out"].get("shape_debug", None) for stage in stages],
                "iter_layer_ids": ([] if iter_layer_ids is None else [int(x) for x in iter_layer_ids]),
            }
        corr_entropy = None
        corr_margin = None
        corr_entropy_stage1 = None
        corr_margin_stage1 = None
        corr_entropy_stage2 = None
        corr_margin_stage2 = None
        if use_s4a_centercorr or use_s4b_allpaircorr:
            corr_entropy_list = [stage["corr_entropy"] for stage in stages if torch.is_tensor(stage["corr_entropy"])]
            corr_margin_list = [stage["corr_margin"] for stage in stages if torch.is_tensor(stage["corr_margin"])]
            if corr_entropy_list:
                corr_entropy = torch.stack(corr_entropy_list, dim=0).mean(dim=0)
                corr_entropy_stage1 = corr_entropy_list[0]
                if len(corr_entropy_list) > 1:
                    corr_entropy_stage2 = corr_entropy_list[-1]
            if corr_margin_list:
                corr_margin = torch.stack(corr_margin_list, dim=0).mean(dim=0)
                corr_margin_stage1 = corr_margin_list[0]
                if len(corr_margin_list) > 1:
                    corr_margin_stage2 = corr_margin_list[-1]

        return {
            "stage0_xy": base_xy,
            "stage0_conf": base_conf,
            "stage0_centroid_xy": pred_centroid_xy,
            "stage0_offsets_xy": base_offsets,
            "stages": stages,
            "stage1_xy": stages[0]["pred_xy"],
            "stage1_conf": stages[0]["pred_conf"],
            "stage1_centroid_xy": stages[0]["pred_centroid_xy"],
            "stage1_offsets_xy": stages[0]["pred_offsets_xy"],
            "stage1_delta_offsets_xy": stages[0]["delta_offsets_xy"],
            "final_out": final_out,
            "final_xy": final_stage["pred_xy"],
            "final_conf": final_stage["pred_conf"],
            "final_centroid_xy": final_centroid,
            "final_offsets_xy": final_stage["pred_offsets_xy"],
            "total_delta_centroid_xy": final_centroid - pred_centroid_xy,
            "final_delta_offsets_xy": final_stage["delta_offsets_xy"],
            "total_delta_xy": final_stage["pred_xy"] - base_xy,
            "total_delta_offsets_xy": final_stage["pred_offsets_xy"] - base_offsets,
            "corr_entropy": corr_entropy,
            "corr_margin": corr_margin,
            "corr_entropy_stage1": corr_entropy_stage1,
            "corr_margin_stage1": corr_margin_stage1,
            "corr_entropy_stage2": corr_entropy_stage2,
            "corr_margin_stage2": corr_margin_stage2,
            "iter_layer_ids": ([] if iter_layer_ids is None else [int(x) for x in iter_layer_ids]),
            "shape_debug": shape_debug,
        }

    def _upsample_spatial_outputs(
        self,
        head_out: Dict[str, torch.Tensor],
        grid_xy: torch.Tensor,
        h_patches: int,
        w_patches: int,
        target_t: int,
        head_name: str,
    ) -> Dict[str, torch.Tensor]:
        logits = head_out["logits"]
        delta_raw = head_out.get("delta_raw", None)
        t_in = int(logits.shape[1])

        logits_up: torch.Tensor
        delta_up: Optional[torch.Tensor]
        use_learned = (
            self.temporal_upsample == "deconv"
            and target_t == t_in * 2
            and head_name in ("tool", "contact")
        )
        if use_learned:
            if head_name == "tool":
                logits_up = self._temporal_deconv_logits(logits, self.tool_time_up_logits)
                if delta_raw is not None and self.tool_time_up_delta is not None:
                    delta_up = self._temporal_deconv_delta(delta_raw, self.tool_time_up_delta)
                else:
                    delta_up = None
            else:
                logits_up = self._temporal_deconv_logits(logits, self.contact_time_up_logits)
                if delta_raw is not None and self.contact_time_up_delta is not None:
                    delta_up = self._temporal_deconv_delta(delta_raw, self.contact_time_up_delta)
                else:
                    delta_up = None
            # Safety: ensure exact target length.
            if int(logits_up.shape[1]) != target_t:
                logits_up = self._temporal_interpolate_dim1(logits_up, target_t)
            if delta_up is not None and int(delta_up.shape[1]) != target_t:
                delta_up = self._temporal_interpolate_dim1(delta_up, target_t)
        else:
            logits_up = self._temporal_interpolate_dim1(logits, target_t)  # [B,T_out,P,S]
            if delta_raw is not None:
                delta_up = self._temporal_interpolate_dim1(delta_raw, target_t)  # [B,T_out,P,S,2]
            else:
                delta_up = None
        return self._decode_spatial_outputs(logits_up, delta_up, grid_xy, h_patches, w_patches)

    @staticmethod
    def _temporal_deconv_logits(logits: torch.Tensor, upsampler: nn.Module) -> torch.Tensor:
        # logits: [B,T,P,S] -> [B*S,P,T]
        bsz, t_in, n_points, s = logits.shape
        x = logits.permute(0, 3, 2, 1).reshape(bsz * s, n_points, t_in).contiguous()
        y = upsampler(x)
        t_out = int(y.shape[-1])
        return y.reshape(bsz, s, n_points, t_out).permute(0, 3, 2, 1).contiguous()  # [B,T_out,P,S]

    @staticmethod
    def _temporal_deconv_delta(delta_raw: torch.Tensor, upsampler: nn.Module) -> torch.Tensor:
        # delta_raw: [B,T,P,S,2] -> [B*S,P*2,T]
        bsz, t_in, n_points, s, xy2 = delta_raw.shape
        if xy2 != 2:
            raise ValueError(f"delta_raw last dim must be 2, got {xy2}")
        x = delta_raw.permute(0, 3, 2, 4, 1).reshape(bsz * s, n_points * 2, t_in).contiguous()
        y = upsampler(x)
        t_out = int(y.shape[-1])
        return y.reshape(bsz, s, n_points, 2, t_out).permute(0, 4, 2, 1, 3).contiguous()  # [B,T_out,P,S,2]

    def _forward_from_tokens_grid(
        self,
        tokens_grid: torch.Tensor,
        token_order: str,
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        multilayer_tokens_grid: Optional[torch.Tensor] = None
        if tokens_grid.ndim == 6:
            if self.requires_multilayer_vjepa_tokens:
                multilayer_tokens_grid = tokens_grid
                tokens_grid = multilayer_tokens_grid[:, -1, ...]
            else:
                tokens_grid = tokens_grid[:, -1, ...]
        if tokens_grid.ndim != 5:
            raise ValueError(f"Expected tokens_grid as 5D [B,T,H,W,D], got shape={tuple(tokens_grid.shape)}")
        bsz, t_enc, h_prime, w_prime, d = tokens_grid.shape
        timing_ms: Dict[str, float] = {}
        shape_debug: Dict[str, object] = {}

        def _tic() -> float:
            if self.debug_timing and tokens_grid.is_cuda:
                torch.cuda.synchronize(tokens_grid.device)
            return time.perf_counter()

        def _toc(name: str, t_start: float) -> None:
            if not self.debug_timing:
                return
            if tokens_grid.is_cuda:
                torch.cuda.synchronize(tokens_grid.device)
            timing_ms[name] = float((time.perf_counter() - t_start) * 1000.0)

        t_all = _tic()

        n_tokens_encoder = t_enc * h_prime * w_prime
        n_tokens_meta = n_tokens_encoder
        s = h_prime * w_prime
        feat_layers_flat: Optional[torch.Tensor] = None
        if multilayer_tokens_grid is not None:
            if tuple(multilayer_tokens_grid.shape[0:1] + multilayer_tokens_grid.shape[2:]) != (bsz, t_enc, h_prime, w_prime, d):
                raise ValueError(
                    "Multi-layer tokens_grid shape mismatch against selected deep layer: "
                    f"multilayer={tuple(multilayer_tokens_grid.shape)} deep={tuple(tokens_grid.shape)}"
                )
            feat_layers_flat = multilayer_tokens_grid.reshape(
                bsz,
                int(multilayer_tokens_grid.shape[1]),
                t_enc,
                s,
                d,
            ).contiguous()
            if self.debug_shapes:
                shape_debug["tokens_grid_multilayer"] = list(multilayer_tokens_grid.shape)
                shape_debug["feat_layers_bktsd"] = list(feat_layers_flat.shape)

        if self.tool_only or self.combo_probe_attn or self.combo_probe_dual_attn:
            t = _tic()
            tokens_seq = tokens_grid.reshape(bsz, n_tokens_encoder, d).contiguous()  # [B,N,D]
            combo_feat = None
            tool_feat = None
            action_feat = None
            action_query_bias = None
            action_prior_delta = None
            action_prior_tool_prob = None
            combo_action_logits_base = None
            if self.tool_only or self.combo_probe_attn:
                if self.combo_attn_pool is None:
                    raise RuntimeError("Probe mode requires `combo_attn_pool`.")
                combo_feat = self.combo_attn_pool(tokens_seq)  # [B,D]
                if self.combo_feat_dropout > 0.0:
                    combo_feat = F.dropout(combo_feat, p=self.combo_feat_dropout, training=self.training)
            if self.tool_only:
                if self.combo_tool_head is None:
                    raise RuntimeError("tool_only=True requires `combo_tool_head`.")
                combo_tool_logits = self.combo_tool_head(combo_feat)  # [B,C_tool]
                combo_action_logits = combo_tool_logits.new_zeros((bsz, self.num_action_classes))
                combo_logits = combo_tool_logits.new_zeros((bsz, self.num_combo_classes))
            elif self.combo_probe_attn:
                if self.combo_head is None:
                    raise RuntimeError("combo_probe_attn=True requires `combo_head`.")
                combo_logits = self.combo_head(combo_feat)  # [B,C_combo]
                combo_tool_logits = combo_logits.new_zeros((bsz, self.num_tool_classes))
                combo_action_logits = combo_logits.new_zeros((bsz, self.num_action_classes))
            else:
                if self.combo_tool_spatial_pool is None or self.combo_action_temporal_pool is None:
                    raise RuntimeError("combo_probe_dual_attn=True requires both tool/action xattn pools.")
                tool_feat = self.combo_tool_spatial_pool(tokens_seq)  # [B,D]
                if self.combo_feat_dropout > 0.0:
                    tool_feat_for_head = F.dropout(tool_feat, p=self.combo_feat_dropout, training=self.training)
                else:
                    tool_feat_for_head = tool_feat
                if self.combo_tool_head is None or self.combo_action_head is None:
                    raise RuntimeError("combo_probe_dual_attn=True requires tool/action heads.")
                combo_tool_logits = self.combo_tool_head(tool_feat_for_head)  # [B,C_tool]
                if self.combo_action_tool_prior is not None:
                    # Query-stage tool-conditioned bias: q_action' = q_action + alpha * Linear(softmax(tool_logits)).
                    action_prior_delta, action_prior_tool_prob = self.combo_action_tool_prior(combo_tool_logits)  # [B,D], [B,Nt]
                    action_query_bias = action_prior_delta
                action_feat = self.combo_action_temporal_pool(tokens_seq, query_bias=action_query_bias)  # [B,D]
                if self.combo_feat_dropout > 0.0:
                    action_feat = F.dropout(action_feat, p=self.combo_feat_dropout, training=self.training)
                combo_action_logits_base = self.combo_action_head(action_feat)  # [B,C_action]
                combo_action_logits = combo_action_logits_base
                combo_logits = self._structured_combo_logits(
                    combo_tool_logits=combo_tool_logits,
                    combo_action_logits=combo_action_logits,
                )  # [B,C_combo]
            _toc("combo_head_ms", t)
            _toc("model_total_ms", t_all)

            t_out = int(self.output_num_frames) if int(self.output_num_frames) > 0 else t_enc
            zeros_tool_xy = combo_tool_logits.new_zeros((bsz, t_out, self.num_tool_points, 2))
            zeros_ct_xy = combo_tool_logits.new_zeros((bsz, t_out, self.num_contact_points, 2))
            zeros_tool_xy_px = self._xy_norm_to_canvas_px(zeros_tool_xy)
            zeros_ct_xy_px = self._xy_norm_to_canvas_px(zeros_ct_xy)
            zeros_tool_conf = combo_tool_logits.new_zeros((bsz, t_out, self.num_tool_points))
            zeros_ct_conf = combo_tool_logits.new_zeros((bsz, t_out, self.num_contact_points))
            zeros_tool_probs = combo_tool_logits.new_zeros((bsz, t_out, self.num_tool_points, s))
            zeros_ct_probs = combo_tool_logits.new_zeros((bsz, t_out, self.num_contact_points, s))
            zeros_tool_hm = combo_tool_logits.new_zeros((bsz, t_out, self.num_tool_points, h_prime, w_prime))
            zeros_ct_hm = combo_tool_logits.new_zeros((bsz, t_out, self.num_contact_points, h_prime, w_prime))
            zeros_range_logits = combo_tool_logits.new_zeros((bsz, t_out))
            zeros_range_offsets = combo_tool_logits.new_zeros((bsz, t_out, 2))
            zeros_range_bin = combo_tool_logits.new_zeros((bsz, self.num_range_bins))
            zeros_range_norm = combo_tool_logits.new_zeros((bsz,))
            zeros_tool_gate = combo_tool_logits.new_zeros((bsz, t_out, self.num_tool_points))
            zeros_ct_gate = combo_tool_logits.new_zeros((bsz, t_out, self.num_contact_points))

            if self.debug_shapes:
                shape_debug["tokens_grid"] = list(tokens_grid.shape)
                shape_debug["tokens_seq"] = list(tokens_seq.shape)
                if combo_feat is not None:
                    shape_debug["combo_feat"] = list(combo_feat.shape)
                if tool_feat is not None:
                    shape_debug["tool_feat"] = list(tool_feat.shape)
                if action_feat is not None:
                    shape_debug["action_feat"] = list(action_feat.shape)
                if action_query_bias is not None:
                    shape_debug["action_query_bias"] = list(action_query_bias.shape)
                if combo_action_logits_base is not None:
                    shape_debug["combo_action_logits_base"] = list(combo_action_logits_base.shape)
                if action_prior_delta is not None:
                    shape_debug["action_prior_delta"] = list(action_prior_delta.shape)
                if action_prior_tool_prob is not None:
                    shape_debug["action_prior_tool_prob"] = list(action_prior_tool_prob.shape)

            return {
                "pred_tool_xy": zeros_tool_xy,
                "pred_ct_xy": zeros_ct_xy,
                "pred_tool_xy_px": zeros_tool_xy_px,
                "pred_ct_xy_px": zeros_ct_xy_px,
                "combo_logits": combo_logits,
                "combo_tool_logits": combo_tool_logits,
                "combo_action_logits": combo_action_logits,
                "logits_start_bin": zeros_range_bin,
                "logits_end_bin": zeros_range_bin.clone(),
                "range_actionness_logits": zeros_range_logits,
                "range_offsets": zeros_range_offsets,
                "pred_action_start_norm": zeros_range_norm,
                "pred_action_end_norm": zeros_range_norm.clone(),
                "conf_tool": zeros_tool_conf,
                "conf_ct": zeros_ct_conf,
                "probs_tool": zeros_tool_probs,
                "probs_ct": zeros_ct_probs,
                "heatmap_tool": zeros_tool_hm,
                "heatmap_ct": zeros_ct_hm,
                "logits_tool": zeros_tool_probs,
                "logits_ct": zeros_ct_probs,
                "traj_refine_temporal_tool_xy": zeros_tool_xy,
                "traj_refine_temporal_ct_xy": zeros_ct_xy,
                "traj_refine_delta_tool_xy": zeros_tool_xy,
                "traj_refine_delta_ct_xy": zeros_ct_xy,
                "traj_refine_delta_xy": combo_tool_logits.new_zeros((bsz, t_out, self.num_tool_points + self.num_contact_points, 2)),
                "traj_refine_delta_tool_gate": zeros_tool_gate,
                "traj_refine_delta_ct_gate": zeros_ct_gate,
                "meta": {
                    "t_encoder": t_enc,
                    "t_out": t_out,
                    "t_prime": t_out,
                    "h_prime": h_prime,
                    "w_prime": w_prime,
                    "h_encoder": h_prime,
                    "w_encoder": w_prime,
                    "temporal_upsample": self.temporal_upsample,
                    "heatmap_softmax_temp": self.heatmap_softmax_temp,
                    "enable_delta_branch": self.enable_delta_branch,
                    "temporal_mean_residual_alpha": self.temporal_mean_residual_alpha,
                    "refiner_mode": self.refiner_mode,
                    "trajectory_only": self.trajectory_only,
                    "range_only": self.range_only,
                    "trajectory_target": self.trajectory_target,
                    "tool_only": self.tool_only,
                    "combo_probe_attn": self.combo_probe_attn,
                    "combo_probe_dual_attn": self.combo_probe_dual_attn,
                    "combo_feature_mode": self.combo_feature_mode,
                    "combo_pool_type": ("dual_cross_attention_from_tokens" if self.combo_probe_dual_attn else "cross_attention"),
                    "combo_head_dropout": self.combo_head_dropout,
                    "combo_feat_dropout": self.combo_feat_dropout,
                    "dual_attn_tool_prior": bool(self.combo_action_tool_prior is not None),
                    "dual_attn_tool_prior_detach": bool(self.dual_attn_tool_prior_detach),
                    "dual_attn_tool_prior_alpha": (
                        0.0
                        if self.combo_action_tool_prior is None
                        else float(self.combo_action_tool_prior.alpha.detach().item())
                    ),
                    "num_tool_classes": self.num_tool_classes,
                    "num_action_classes": self.num_action_classes,
                    "traj_head_hidden_dim": self.traj_head_hidden_dim,
                    "traj_head_up_layers": self.traj_head_up_layers,
                    "traj_recurrent_max_delta_norm": self.traj_recurrent_max_delta_norm,
                    "traj_dynamic_centroid_max_beta": self.traj_dynamic_centroid_max_beta,
                    "traj_template_support_radius": self.traj_template_support_radius,
                    "traj_template_support_radii_by_stage": list(self.traj_template_support_radii_by_stage),
                    "traj_template_late_stage_count": self.traj_template_late_stage_count,
                    "traj_template_ema_alpha": self.traj_template_ema_alpha,
                    "traj_template_detach_current_inputs": bool(self.traj_template_detach_current_inputs),
                    "traj_tool_centroid_offset_head": self.traj_tool_centroid_offset_head,
                    "traj_tool_offset_max_norm": self.traj_tool_offset_max_norm,
                    "num_range_bins": self.num_range_bins,
                    "output_canvas_size": self.output_canvas_size,
                    "token_order": token_order,
                    "n_tokens": n_tokens_meta,
                    "n_tokens_encoder": n_tokens_encoder,
                    "timing_ms": timing_ms,
                    "shape_debug": shape_debug,
                },
            }

        if self.trajectory_only:
            # Directly query frozen VJEPA tokens in trajectory-only mode.
            feat_flat = tokens_grid.reshape(bsz, t_enc, s, d).contiguous()  # [B,T_enc,S,D_vjepa]
            if self.debug_shapes:
                shape_debug["tokens_grid"] = list(tokens_grid.shape)
                shape_debug["feat_flat_btsd"] = list(feat_flat.shape)
            if self.tool_prob_head is None and self.contact_prob_head is None:
                raise RuntimeError("trajectory_only=True requires at least one trajectory head (tool or ct).")
            t = _tic()
            tool_id_in = _normalize_class_ids(
                tool_id,
                name="tool_id",
                bsz=bsz,
                num_classes=self.num_tool_classes,
                ref_tensor=feat_flat,
            )
            action_id_in = _normalize_class_ids(
                action_id,
                name="action_id",
                bsz=bsz,
                num_classes=self.num_action_classes,
                ref_tensor=feat_flat,
            )
            combo_id_in = _normalize_class_ids(
                combo_id,
                name="combo_id",
                bsz=bsz,
                num_classes=self.num_combo_classes,
                ref_tensor=feat_flat,
            )
            spatial_hw = (int(h_prime), int(w_prime))
            token_mask_stats: Dict[str, object] = {
                "enabled": bool(self.kv_token_masker is not None),
                "eval_enabled": bool(self.token_mask_eval),
                "active": False,
                "applied": False,
                "epoch": int(self.token_mask_epoch),
                "p": 0.0,
                "strategy": str(self.token_mask_strategy),
                "value_mode": str(self.token_mask_value),
                "mix_801010": bool(self.token_mask_mix_801010),
                "keep_k": int(self.token_mask_keep_k),
                "keep_n": int(s),
                "mask_ratio": 0.0,
                "masked_tokens": 0,
                "masktok_ratio": 0.0,
                "random_ratio": 0.0,
                "keep_ratio": 0.0,
            }
            feat_for_traj = feat_flat
            feat_layers_for_traj = feat_layers_flat
            if self.kv_token_masker is not None:
                self.kv_token_masker.set_runtime_epoch(self.token_mask_epoch)
                apply_mask = bool(
                    (self.training and self.token_mask_enable)
                    or ((not self.training) and self.token_mask_eval)
                )
                mask_plan, token_mask_stats = self.kv_token_masker.build_mask_plan(
                    feat_flat,
                    spatial_hw=spatial_hw,
                    apply_mask=apply_mask,
                )
                feat_for_traj = self.kv_token_masker.apply_mask_plan(
                    feat_flat,
                    plan=mask_plan,
                    spatial_hw=spatial_hw,
                )
                if feat_layers_flat is not None:
                    masked_layers = [
                        self.kv_token_masker.apply_mask_plan(
                            feat_layers_flat[:, layer_idx, ...],
                            plan=mask_plan,
                            spatial_hw=spatial_hw,
                        ).unsqueeze(1)
                        for layer_idx in range(int(feat_layers_flat.shape[1]))
                    ]
                    feat_layers_for_traj = torch.cat(masked_layers, dim=1).contiguous()
                if self.debug_shapes:
                    shape_debug["feat_traj_masked_btsd"] = list(feat_for_traj.shape)
                    shape_debug["feat_layers_traj_masked_bktsd"] = (
                        [] if feat_layers_for_traj is None else list(feat_layers_for_traj.shape)
                    )
            tool_rec = (
                self.tool_prob_head(
                    feat_for_traj,
                    tool_id=tool_id_in,
                    action_id=action_id_in,
                    combo_id=combo_id_in,
                    spatial_hw=spatial_hw,
                )
                if self.tool_prob_head is not None
                else None
            )
            ct_rec = (
                self.contact_prob_head(
                    feat_for_traj,
                    tool_id=tool_id_in,
                    action_id=action_id_in,
                    combo_id=combo_id_in,
                    spatial_hw=spatial_hw,
                )
                if self.contact_prob_head is not None
                else None
            )
            _toc("traj_recurrent_ms", t)

            rec_ref = tool_rec if tool_rec is not None else ct_rec
            assert rec_ref is not None
            t_base = int(rec_ref["pred_xy"].shape[1])
            if (tool_rec is not None) and (ct_rec is not None) and (int(ct_rec["pred_xy"].shape[1]) != t_base):
                raise ValueError(
                    "tool/contact trajectory length mismatch: "
                    f"{int(tool_rec['pred_xy'].shape[1])} vs {int(ct_rec['pred_xy'].shape[1])}"
                )
            t_out = int(self.output_num_frames) if int(self.output_num_frames) > 0 else t_base
            if t_base != t_out:
                raise ValueError(
                    "trajectory_only recurrent path expects exact 2x frame expansion without interpolation: "
                    f"t_base={t_base}, t_out={t_out}."
                )
            if tool_rec is not None:
                tool_xy = tool_rec["pred_xy"]
                tool_delta = tool_rec["delta_xy"]
                tool_gate = tool_rec["delta_gate"]
                tool_conf = tool_rec["conf"]
            else:
                tool_xy = feat_for_traj.new_zeros((bsz, t_out, self.num_tool_points, 2))
                tool_delta = feat_for_traj.new_zeros((bsz, t_out, self.num_tool_points, 2))
                tool_gate = feat_for_traj.new_zeros((bsz, t_out, self.num_tool_points))
                tool_conf = feat_for_traj.new_zeros((bsz, t_out, self.num_tool_points))
            if ct_rec is not None:
                ct_xy = ct_rec["pred_xy"]
                ct_delta = ct_rec["delta_xy"]
                ct_gate = ct_rec["delta_gate"]
                ct_conf = ct_rec["conf"]
            else:
                ct_xy = feat_for_traj.new_zeros((bsz, t_out, self.num_contact_points, 2))
                ct_delta = feat_for_traj.new_zeros((bsz, t_out, self.num_contact_points, 2))
                ct_gate = feat_for_traj.new_zeros((bsz, t_out, self.num_contact_points))
                ct_conf = feat_for_traj.new_zeros((bsz, t_out, self.num_contact_points))

            h_decode = int(h_prime)
            w_decode = int(w_prime)
            t = _tic()
            tool_stub_base = self._xy_to_spatial_stub(tool_xy, h_patches=h_decode, w_patches=w_decode)
            ct_stub_base = self._xy_to_spatial_stub(ct_xy, h_patches=h_decode, w_patches=w_decode)
            n_tokens_meta = n_tokens_encoder
            _toc("spatial_stub_ms", t)

            t = _tic()
            traj_refine_active = self.traj_refiner_mode in (
                "attn",
                "lite",
                "s1_offset",
                "s2_iter_offset",
                "s3_support3x3",
                "s3p2x_finalrefine_l18",
                "s3p3_singlelayer_statecarry_refine",
                "s3p3x_singlelayer_statecarry_refine_plus1",
                "s3p8_singlelayer_statecarry_fourierbelief",
                "s3p8a_singlelayer_statecarry_residualfourierbelief",
                "s3p9_singlelayer_statecarry_anchoronly",
                "s3p9d_singlelayer_statecarry_anchoronly_detached",
                "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                "s3p10_singlelayer_statecarry_factorizedtoolresidual",
                "s3p4_singlelayer_statecarry_dynamiccentroid",
                "s3p5_singlelayer_statecarry_deformablesupport",
                "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
                "m1_progressive_multilayer_movingrefine",
                "m1_strict_s3_multilayer_mean",
                "m2_layerwise_statecarry_refine",
                "m3_layerwise_statecarry_dynamiccentroid",
                "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
                "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
                "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
            )
            feat_layers_refine: Optional[torch.Tensor] = None
            verifier_feat_refine: Optional[torch.Tensor] = None
            if traj_refine_active:
                if self.traj_refiner_mode == "m1_progressive_multilayer_movingrefine":
                    if feat_layers_for_traj is None:
                        raise RuntimeError(
                            "M1 multi-layer moving refine requires multi-layer VJEPA tokens "
                            "with shape [B,K,T,H,W,D] from cache/wrapper."
                        )
                    feat_layers_refine = self._match_multilayer_feat_time_steps(
                        feat_layers_for_traj,
                        target_t=t_out,
                    )
                    feat_refine = feat_layers_refine[:, -1, ...]
                    tool_local_feat = None
                    ct_local_feat = None
                    tool_centroid_base = (
                        tool_rec["pred_centroid_xy"]
                        if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                        else tool_xy.mean(dim=2, keepdim=True)
                    )
                    tool_offsets_base = (
                        tool_rec["pred_offsets_xy"]
                        if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                        else (tool_xy - tool_centroid_base)
                    )
                    ct_centroid_base = ct_xy.mean(dim=2, keepdim=True)
                    ct_offsets_base = ct_xy - ct_centroid_base
                    tool_centroid_local_feat = None
                    ct_centroid_local_feat = None
                    tool_traj_cond = self._trajectory_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    ct_traj_cond = self._trajectory_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    tool_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                    ct_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                elif self.traj_refiner_mode == "m1_strict_s3_multilayer_mean":
                    if feat_layers_for_traj is None:
                        raise RuntimeError(
                            "m1_strict_s3_multilayer_mean requires multi-layer VJEPA tokens "
                            "with shape [B,K,T,H,W,D] from cache/wrapper."
                        )
                    feat_layers_refine = self._match_multilayer_feat_time_steps(
                        feat_layers_for_traj,
                        target_t=t_out,
                    )
                    feat_refine = feat_layers_refine.mean(dim=1)
                    tool_local_feat = None
                    ct_local_feat = None
                    tool_centroid_base = (
                        tool_rec["pred_centroid_xy"]
                        if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                        else tool_xy.mean(dim=2, keepdim=True)
                    )
                    tool_offsets_base = (
                        tool_rec["pred_offsets_xy"]
                        if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                        else (tool_xy - tool_centroid_base)
                    )
                    ct_centroid_base = ct_xy.mean(dim=2, keepdim=True)
                    ct_offsets_base = ct_xy - ct_centroid_base
                    tool_centroid_local_feat = self._trajectory_local_feat_from_xy(
                        tool_centroid_base,
                        feat_refine,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    ct_centroid_local_feat = self._trajectory_local_feat_from_xy(
                        ct_centroid_base,
                        feat_refine,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    tool_traj_cond = self._trajectory_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    ct_traj_cond = self._trajectory_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    tool_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                    ct_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                elif self.traj_refiner_mode in STRICT_S3_CROSSLAYER_REFINER_MODES:
                    if feat_layers_for_traj is None:
                        raise RuntimeError(
                            "`s3p2x_finalrefine_l18` requires multi-layer VJEPA tokens "
                            "with shape [B,K,T,H,W,D] from cache/wrapper."
                        )
                    feat_layers_refine = self._match_multilayer_feat_time_steps(
                        feat_layers_for_traj,
                        target_t=t_out,
                    )
                    feat_refine = feat_layers_refine[:, -1, ...]
                    tool_local_feat = None
                    ct_local_feat = None
                    tool_centroid_base = (
                        tool_rec["pred_centroid_xy"]
                        if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                        else tool_xy.mean(dim=2, keepdim=True)
                    )
                    tool_offsets_base = (
                        tool_rec["pred_offsets_xy"]
                        if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                        else (tool_xy - tool_centroid_base)
                    )
                    ct_centroid_base = ct_xy.mean(dim=2, keepdim=True)
                    ct_offsets_base = ct_xy - ct_centroid_base
                    tool_centroid_local_feat = self._trajectory_local_feat_from_xy(
                        tool_centroid_base,
                        feat_refine,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    ct_centroid_local_feat = self._trajectory_local_feat_from_xy(
                        ct_centroid_base,
                        feat_refine,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    tool_traj_cond = self._trajectory_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    ct_traj_cond = self._trajectory_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    tool_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                    ct_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                elif self.traj_refiner_mode in SHALLOW_VERIFIER_REFINER_MODES:
                    if feat_layers_for_traj is None:
                        raise RuntimeError(
                            "S3.9f shallow anchor verification requires multi-layer VJEPA tokens "
                            "with shape [B,K,T,H,W,D] from cache/wrapper."
                        )
                    feat_layers_refine = self._match_multilayer_feat_time_steps(
                        feat_layers_for_traj,
                        target_t=t_out,
                    )
                    if int(feat_layers_refine.shape[1]) < 2:
                        raise RuntimeError(
                            f"S3.9f expects at least 2 refine layers, got shape={tuple(feat_layers_refine.shape)}"
                        )
                    feat_refine = feat_layers_refine[:, -1, ...]
                    verifier_feat_refine = feat_layers_refine[:, -2, ...]
                    tool_local_feat = None
                    ct_local_feat = None
                    tool_centroid_base = (
                        tool_rec["pred_centroid_xy"]
                        if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                        else tool_xy.mean(dim=2, keepdim=True)
                    )
                    tool_offsets_base = (
                        tool_rec["pred_offsets_xy"]
                        if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                        else (tool_xy - tool_centroid_base)
                    )
                    ct_centroid_base = ct_xy.mean(dim=2, keepdim=True)
                    ct_offsets_base = ct_xy - ct_centroid_base
                    tool_centroid_local_feat = self._trajectory_local_feat_from_xy(
                        tool_centroid_base,
                        feat_refine,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    ct_centroid_local_feat = self._trajectory_local_feat_from_xy(
                        ct_centroid_base,
                        feat_refine,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    tool_traj_cond = self._trajectory_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    ct_traj_cond = self._trajectory_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    tool_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                    ct_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                elif self.traj_refiner_mode in LAYERWISE_STATECARRY_REFINER_MODES:
                    if feat_layers_for_traj is None:
                        raise RuntimeError(
                            "Layerwise state-carry refine requires multi-layer VJEPA tokens "
                            "with shape [B,K,T,H,W,D] from cache/wrapper."
                        )
                    feat_layers_refine = self._match_multilayer_feat_time_steps(
                        feat_layers_for_traj,
                        target_t=t_out,
                    )
                    feat_refine = self._match_feat_time_steps(feat_for_traj, target_t=t_out)
                    tool_local_feat = None
                    ct_local_feat = None
                    tool_centroid_base = (
                        tool_rec["pred_centroid_xy"]
                        if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                        else tool_xy.mean(dim=2, keepdim=True)
                    )
                    tool_offsets_base = (
                        tool_rec["pred_offsets_xy"]
                        if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                        else (tool_xy - tool_centroid_base)
                    )
                    ct_centroid_base = ct_xy.mean(dim=2, keepdim=True)
                    ct_offsets_base = ct_xy - ct_centroid_base
                    tool_centroid_local_feat = None
                    ct_centroid_local_feat = None
                    tool_traj_cond = self._trajectory_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    ct_traj_cond = self._trajectory_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                        action_id=action_id_in,
                        combo_id=combo_id_in,
                    )
                    tool_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.tool_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                    ct_traj_tool_cond = self._trajectory_tool_condition_from_head(
                        self.contact_prob_head,
                        ref_tensor=feat_for_traj,
                        tool_id=tool_id_in,
                    )
                else:
                    feat_refine = self._match_feat_time_steps(feat_for_traj, target_t=t_out)
                    if self.traj_refiner_mode in (
                        "s1_offset",
                        "s2_iter_offset",
                        "s3_support3x3",
                        "s3p2x_finalrefine_l18",
                        "s3p3_singlelayer_statecarry_refine",
                        "s3p3x_singlelayer_statecarry_refine_plus1",
                        "s3p8_singlelayer_statecarry_fourierbelief",
                        "s3p8a_singlelayer_statecarry_residualfourierbelief",
                        "s3p9_singlelayer_statecarry_anchoronly",
                        "s3p9d_singlelayer_statecarry_anchoronly_detached",
                        "s3p9e_singlelayer_statecarry_anchoronly_ramped",
                        "s3p9f_singlelayer_statecarry_anchorverifier_l18",
                        "s3p10_singlelayer_statecarry_factorizedtoolresidual",
                        "s3p4_singlelayer_statecarry_dynamiccentroid",
                        "s3p5_singlelayer_statecarry_deformablesupport",
                        "s3p6_singlelayer_statecarry_lategated_deformablesupport",
                        "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
                        "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
                        "m1_strict_s3_multilayer_mean",
                        "m2_layerwise_statecarry_refine",
                        "m3_layerwise_statecarry_dynamiccentroid",
                        "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
                        "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
                        "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
                        "s4a_template_centercorr_3x3",
                        "s4b_template_allpaircorr_3x3",
                    ):
                        tool_local_feat = None
                        ct_local_feat = None
                        tool_centroid_base = (
                            tool_rec["pred_centroid_xy"]
                            if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                            else tool_xy.mean(dim=2, keepdim=True)
                        )
                        tool_offsets_base = (
                            tool_rec["pred_offsets_xy"]
                            if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                            else (tool_xy - tool_centroid_base)
                        )
                        ct_centroid_base = ct_xy.mean(dim=2, keepdim=True)
                        ct_offsets_base = ct_xy - ct_centroid_base
                        tool_centroid_local_feat = self._trajectory_local_feat_from_xy(
                            tool_centroid_base,
                            feat_refine,
                            h_patches=h_decode,
                            w_patches=w_decode,
                        )
                        ct_centroid_local_feat = self._trajectory_local_feat_from_xy(
                            ct_centroid_base,
                            feat_refine,
                            h_patches=h_decode,
                            w_patches=w_decode,
                        )
                        tool_traj_cond = self._trajectory_condition_from_head(
                            self.tool_prob_head,
                            ref_tensor=feat_for_traj,
                            tool_id=tool_id_in,
                            action_id=action_id_in,
                            combo_id=combo_id_in,
                        )
                        ct_traj_cond = self._trajectory_condition_from_head(
                            self.contact_prob_head,
                            ref_tensor=feat_for_traj,
                            tool_id=tool_id_in,
                            action_id=action_id_in,
                            combo_id=combo_id_in,
                        )
                        tool_traj_tool_cond = self._trajectory_tool_condition_from_head(
                            self.tool_prob_head,
                            ref_tensor=feat_for_traj,
                            tool_id=tool_id_in,
                        )
                        ct_traj_tool_cond = self._trajectory_tool_condition_from_head(
                            self.contact_prob_head,
                            ref_tensor=feat_for_traj,
                            tool_id=tool_id_in,
                        )
                    else:
                        tool_local_feat = self._local_feat_from_argmax_cell(tool_stub_base["probs"], feat_refine)
                        ct_local_feat = self._local_feat_from_argmax_cell(ct_stub_base["probs"], feat_refine)
                        tool_centroid_base = None
                        tool_offsets_base = None
                        ct_centroid_base = None
                        ct_offsets_base = None
                        tool_centroid_local_feat = None
                        ct_centroid_local_feat = None
                        tool_traj_cond = None
                        ct_traj_cond = None
                        tool_traj_tool_cond = None
                        ct_traj_tool_cond = None
            else:
                feat_refine = None
                feat_layers_refine = None
                verifier_feat_refine = None
                tool_local_feat = None
                ct_local_feat = None
                tool_centroid_base = None
                tool_offsets_base = None
                ct_centroid_base = None
                ct_offsets_base = None
                tool_centroid_local_feat = None
                ct_centroid_local_feat = None
                tool_traj_cond = None
                ct_traj_cond = None
                tool_traj_tool_cond = None
                ct_traj_tool_cond = None

            tool_ref_pack = None
            ct_ref_pack = None
            if (
                self.traj_refiner_mode == "m1_progressive_multilayer_movingrefine"
                and isinstance(self.tool_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_layers_refine is None:
                    raise RuntimeError("M1 requires matched multi-layer refine features.")
                tool_ref_pack = self._run_multilayer_cotracker_offset_refiner(
                    refiner=self.tool_refiner,
                    pred_xy=tool_xy,
                    pred_conf=tool_conf,
                    feat_layers=feat_layers_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=tool_centroid_base,
                    pred_offsets_xy=tool_offsets_base,
                    traj_cond=tool_traj_cond,
                )
                ref_tool = tool_ref_pack["final_out"]
            elif (
                self.traj_refiner_mode in LAYERWISE_STATECARRY_REFINER_MODES
                and isinstance(self.tool_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_layers_refine is None:
                    raise RuntimeError("Layerwise state-carry refine requires matched multi-layer refine features.")
                tool_ref_pack = self._run_layerwise_statecarry_offset_refiner(
                    refiner=self.tool_refiner,
                    pred_xy=tool_xy,
                    pred_conf=tool_conf,
                    feat_layers=feat_layers_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=tool_centroid_base,
                    pred_offsets_xy=tool_offsets_base,
                    traj_cond=tool_traj_cond,
                )
                ref_tool = tool_ref_pack["final_out"]
            elif (
                self.traj_refiner_mode in SINGLELAYER_STATECARRY_REFINER_MODES
                and isinstance(self.tool_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_refine is None:
                    raise RuntimeError("Single-layer state-carry refine requires matched single-layer VJEPA tokens.")
                tool_ref_pack = self._run_singlelayer_statecarry_offset_refiner(
                    refiner=self.tool_refiner,
                    pred_xy=tool_xy,
                    pred_conf=tool_conf,
                    feat_flat=feat_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=tool_centroid_base,
                    pred_offsets_xy=tool_offsets_base,
                    traj_cond=tool_traj_cond,
                    traj_tool_cond=tool_traj_tool_cond,
                    verifier_feat_flat=verifier_feat_refine,
                    apply_stage0_centroid_noise=True,
                )
                ref_tool = tool_ref_pack["final_out"]
            elif (
                self.traj_refiner_mode in STRICT_S3_CROSSLAYER_REFINER_MODES
                and isinstance(self.tool_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_layers_refine is None:
                    raise RuntimeError("`s3p2x_finalrefine_l18` requires matched multi-layer refine features.")
                iter_layer_ids = [int(self.vjepa.layer_indices[-1]), int(self.vjepa.layer_indices[-2])]
                tool_refine_feats = [
                    feat_layers_refine[:, -1, ...],
                    feat_layers_refine[:, -2, ...],
                ]
                tool_centroid_feats = [
                    self._trajectory_local_feat_from_xy(
                        tool_centroid_base,
                        layer_feat,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    for layer_feat in tool_refine_feats
                ]
                tool_ref_pack = self._run_cotracker_offset_refiner(
                    refiner=self.tool_refiner,
                    pred_xy=tool_xy,
                    pred_conf=tool_conf,
                    feat_flat=tool_refine_feats[0],
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=tool_centroid_base,
                    pred_offsets_xy=tool_offsets_base,
                    centroid_local_feat=tool_centroid_feats[0],
                    traj_cond=tool_traj_cond,
                    num_iters=2,
                    feat_flats_by_iter=tool_refine_feats,
                    centroid_local_feats_by_iter=tool_centroid_feats,
                    iter_layer_ids=iter_layer_ids,
                )
                ref_tool = tool_ref_pack["final_out"]
            elif self.traj_refiner_mode in (
                "s1_offset",
                "s2_iter_offset",
                "s3_support3x3",
                "m1_strict_s3_multilayer_mean",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
            ) and isinstance(self.tool_refiner, CoTrackerDeltaOffsetRefiner):
                tool_ref_pack = self._run_cotracker_offset_refiner(
                    refiner=self.tool_refiner,
                    pred_xy=tool_xy,
                    pred_conf=tool_conf,
                    feat_flat=feat_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=tool_centroid_base,
                    pred_offsets_xy=tool_offsets_base,
                    centroid_local_feat=tool_centroid_local_feat,
                    traj_cond=tool_traj_cond,
                    num_iters=(
                        2
                        if self.traj_refiner_mode in (
                            "s2_iter_offset",
                            "s3_support3x3",
                            "m1_strict_s3_multilayer_mean",
                            "s4a_template_centercorr_3x3",
                            "s4b_template_allpaircorr_3x3",
                        )
                        else 1
                    ),
                )
                ref_tool = tool_ref_pack["final_out"]
            else:
                ref_tool = self.tool_refiner(
                    pred_xy=tool_xy,
                    pred_conf=tool_conf,
                    local_feat=tool_local_feat,
                )
            if (
                self.traj_refiner_mode == "m1_progressive_multilayer_movingrefine"
                and isinstance(self.contact_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_layers_refine is None:
                    raise RuntimeError("M1 requires matched multi-layer refine features.")
                ct_ref_pack = self._run_multilayer_cotracker_offset_refiner(
                    refiner=self.contact_refiner,
                    pred_xy=ct_xy,
                    pred_conf=ct_conf,
                    feat_layers=feat_layers_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=ct_centroid_base,
                    pred_offsets_xy=ct_offsets_base,
                    traj_cond=ct_traj_cond,
                )
                ref_ct = ct_ref_pack["final_out"]
            elif (
                self.traj_refiner_mode in LAYERWISE_STATECARRY_REFINER_MODES
                and isinstance(self.contact_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_layers_refine is None:
                    raise RuntimeError("Layerwise state-carry refine requires matched multi-layer refine features.")
                ct_ref_pack = self._run_layerwise_statecarry_offset_refiner(
                    refiner=self.contact_refiner,
                    pred_xy=ct_xy,
                    pred_conf=ct_conf,
                    feat_layers=feat_layers_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=ct_centroid_base,
                    pred_offsets_xy=ct_offsets_base,
                    traj_cond=ct_traj_cond,
                )
                ref_ct = ct_ref_pack["final_out"]
            elif (
                self.traj_refiner_mode in SINGLELAYER_STATECARRY_REFINER_MODES
                and isinstance(self.contact_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_refine is None:
                    raise RuntimeError("Single-layer state-carry refine requires matched single-layer VJEPA tokens.")
                ct_ref_pack = self._run_singlelayer_statecarry_offset_refiner(
                    refiner=self.contact_refiner,
                    pred_xy=ct_xy,
                    pred_conf=ct_conf,
                    feat_flat=feat_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=ct_centroid_base,
                    pred_offsets_xy=ct_offsets_base,
                    traj_cond=ct_traj_cond,
                    traj_tool_cond=ct_traj_tool_cond,
                    verifier_feat_flat=verifier_feat_refine,
                    apply_stage0_centroid_noise=False,
                )
                ref_ct = ct_ref_pack["final_out"]
            elif (
                self.traj_refiner_mode in STRICT_S3_CROSSLAYER_REFINER_MODES
                and isinstance(self.contact_refiner, CoTrackerDeltaOffsetRefiner)
            ):
                if feat_layers_refine is None:
                    raise RuntimeError("`s3p2x_finalrefine_l18` requires matched multi-layer refine features.")
                iter_layer_ids = [int(self.vjepa.layer_indices[-1]), int(self.vjepa.layer_indices[-2])]
                ct_refine_feats = [
                    feat_layers_refine[:, -1, ...],
                    feat_layers_refine[:, -2, ...],
                ]
                ct_centroid_feats = [
                    self._trajectory_local_feat_from_xy(
                        ct_centroid_base,
                        layer_feat,
                        h_patches=h_decode,
                        w_patches=w_decode,
                    )
                    for layer_feat in ct_refine_feats
                ]
                ct_ref_pack = self._run_cotracker_offset_refiner(
                    refiner=self.contact_refiner,
                    pred_xy=ct_xy,
                    pred_conf=ct_conf,
                    feat_flat=ct_refine_feats[0],
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=ct_centroid_base,
                    pred_offsets_xy=ct_offsets_base,
                    centroid_local_feat=ct_centroid_feats[0],
                    traj_cond=ct_traj_cond,
                    num_iters=2,
                    feat_flats_by_iter=ct_refine_feats,
                    centroid_local_feats_by_iter=ct_centroid_feats,
                    iter_layer_ids=iter_layer_ids,
                )
                ref_ct = ct_ref_pack["final_out"]
            elif self.traj_refiner_mode in (
                "s1_offset",
                "s2_iter_offset",
                "s3_support3x3",
                "m1_strict_s3_multilayer_mean",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
            ) and isinstance(self.contact_refiner, CoTrackerDeltaOffsetRefiner):
                ct_ref_pack = self._run_cotracker_offset_refiner(
                    refiner=self.contact_refiner,
                    pred_xy=ct_xy,
                    pred_conf=ct_conf,
                    feat_flat=feat_refine,
                    h_patches=h_decode,
                    w_patches=w_decode,
                    pred_centroid_xy=ct_centroid_base,
                    pred_offsets_xy=ct_offsets_base,
                    centroid_local_feat=ct_centroid_local_feat,
                    traj_cond=ct_traj_cond,
                    num_iters=(
                        2
                        if self.traj_refiner_mode in (
                            "s2_iter_offset",
                            "s3_support3x3",
                            "m1_strict_s3_multilayer_mean",
                            "s4a_template_centercorr_3x3",
                            "s4b_template_allpaircorr_3x3",
                        )
                        else 1
                    ),
                )
                ref_ct = ct_ref_pack["final_out"]
            else:
                ref_ct = self.contact_refiner(
                    pred_xy=ct_xy,
                    pred_conf=ct_conf,
                    local_feat=ct_local_feat,
                )
            if self.debug_shapes:
                shape_debug["feat_refine_btsd"] = [] if feat_refine is None else list(feat_refine.shape)
                shape_debug["feat_layers_refine_bktsd"] = [] if feat_layers_refine is None else list(feat_layers_refine.shape)
                shape_debug["tool_local_feat"] = [] if tool_local_feat is None else list(tool_local_feat.shape)
                shape_debug["ct_local_feat"] = [] if ct_local_feat is None else list(ct_local_feat.shape)
                shape_debug["tool_refiner"] = (
                    tool_ref_pack.get("shape_debug", None) if tool_ref_pack is not None else ref_tool.get("shape_debug", None)
                )
                shape_debug["contact_refiner"] = (
                    ct_ref_pack.get("shape_debug", None) if ct_ref_pack is not None else ref_ct.get("shape_debug", None)
                )
            _toc("traj_refiner_ms", t)

            tool_xy_stage0 = tool_xy
            ct_xy_stage0 = ct_xy
            tool_conf_stage0 = tool_conf
            ct_conf_stage0 = ct_conf
            tool_offsets_stage0 = (
                tool_offsets_base
                if tool_offsets_base is not None
                else (
                    tool_rec["pred_offsets_xy"]
                    if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                    else (tool_xy_stage0 - tool_xy_stage0.mean(dim=2, keepdim=True))
                )
            )
            ct_offsets_stage0 = (
                ct_offsets_base
                if ct_offsets_base is not None
                else (ct_xy_stage0 - ct_xy_stage0.mean(dim=2, keepdim=True))
            )
            tool_stage_xy_all = [tool_xy_stage0]
            tool_stage_conf_all = [tool_conf_stage0]
            tool_centroid_stage0 = (
                tool_centroid_base
                if tool_centroid_base is not None
                else tool_xy_stage0.mean(dim=2, keepdim=True)
            )
            tool_stage_centroid_all = [tool_centroid_stage0]
            tool_stage_offsets_all = [tool_offsets_stage0]
            tool_stage_delta_offsets_all = [torch.zeros_like(tool_offsets_stage0)]
            if tool_ref_pack is not None:
                for stage in tool_ref_pack.get("stages", []):
                    tool_stage_xy_all.append(stage["pred_xy"])
                    tool_stage_conf_all.append(stage["pred_conf"])
                    tool_stage_centroid_all.append(stage["pred_centroid_xy"])
                    tool_stage_offsets_all.append(stage["pred_offsets_xy"])
                    tool_stage_delta_offsets_all.append(stage["delta_offsets_xy"])
                ref_tool_xy = tool_stage_xy_all[-1]
                ref_tool_conf = tool_stage_conf_all[-1]
                ref_tool_centroid = tool_stage_centroid_all[-1]
                ref_tool_offsets = tool_stage_offsets_all[-1]
                ref_tool_delta_offsets = tool_ref_pack.get(
                    "total_delta_offsets_xy",
                    ref_tool_offsets - tool_offsets_stage0,
                )
            else:
                ref_tool_xy = ref_tool["pred_xy"]
                ref_tool_conf = ref_tool.get("pred_conf", tool_conf)
                ref_tool_centroid = ref_tool.get("pred_centroid_xy", tool_centroid_stage0)
                ref_tool_offsets = ref_tool.get(
                    "pred_offsets_xy",
                    (
                        tool_rec["pred_offsets_xy"]
                        if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec)
                        else (ref_tool_xy - (
                            tool_rec["pred_centroid_xy"]
                            if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec)
                            else ref_tool_xy.mean(dim=2, keepdim=True)
                        ))
                    ),
                )
                ref_tool_delta_offsets = ref_tool.get("delta_offsets_xy", ref_tool_offsets - tool_offsets_stage0)
                tool_stage_xy_all.append(ref_tool_xy)
                tool_stage_conf_all.append(ref_tool_conf)
                tool_stage_centroid_all.append(ref_tool_centroid)
                tool_stage_offsets_all.append(ref_tool_offsets)
                tool_stage_delta_offsets_all.append(ref_tool_delta_offsets)
            tool_xy_stage1 = tool_stage_xy_all[1] if len(tool_stage_xy_all) > 1 else None
            tool_conf_stage1 = tool_stage_conf_all[1] if len(tool_stage_conf_all) > 1 else None
            tool_offsets_stage1 = tool_stage_offsets_all[1] if len(tool_stage_offsets_all) > 1 else None
            tool_delta_offsets_stage1 = (
                tool_stage_delta_offsets_all[1] if len(tool_stage_delta_offsets_all) > 1 else None
            )
            tool_xy_stage2 = tool_stage_xy_all[2] if len(tool_stage_xy_all) > 2 else None
            tool_conf_stage2 = tool_stage_conf_all[2] if len(tool_stage_conf_all) > 2 else None
            tool_offsets_stage2 = tool_stage_offsets_all[2] if len(tool_stage_offsets_all) > 2 else None
            tool_delta_offsets_stage2 = (
                tool_stage_delta_offsets_all[2] if len(tool_stage_delta_offsets_all) > 2 else None
            )
            tool_xy_stage3 = tool_stage_xy_all[3] if len(tool_stage_xy_all) > 3 else None
            tool_conf_stage3 = tool_stage_conf_all[3] if len(tool_stage_conf_all) > 3 else None
            tool_offsets_stage3 = tool_stage_offsets_all[3] if len(tool_stage_offsets_all) > 3 else None
            tool_delta_offsets_stage3 = (
                tool_stage_delta_offsets_all[3] if len(tool_stage_delta_offsets_all) > 3 else None
            )
            tool_xy_stage4 = tool_stage_xy_all[4] if len(tool_stage_xy_all) > 4 else None
            tool_conf_stage4 = tool_stage_conf_all[4] if len(tool_stage_conf_all) > 4 else None
            tool_offsets_stage4 = tool_stage_offsets_all[4] if len(tool_stage_offsets_all) > 4 else None
            tool_delta_offsets_stage4 = (
                tool_stage_delta_offsets_all[4] if len(tool_stage_delta_offsets_all) > 4 else None
            )
            ct_stage_xy_all = [ct_xy_stage0]
            ct_stage_conf_all = [ct_conf_stage0]
            ct_stage_offsets_all = [ct_offsets_stage0]
            ct_stage_delta_offsets_all = [torch.zeros_like(ct_offsets_stage0)]
            if ct_ref_pack is not None:
                for stage in ct_ref_pack.get("stages", []):
                    ct_stage_xy_all.append(stage["pred_xy"])
                    ct_stage_conf_all.append(stage["pred_conf"])
                    ct_stage_offsets_all.append(stage["pred_offsets_xy"])
                    ct_stage_delta_offsets_all.append(stage["delta_offsets_xy"])
                ref_ct_xy = ct_stage_xy_all[-1]
                ref_ct_conf = ct_stage_conf_all[-1]
                ref_ct_offsets = ct_stage_offsets_all[-1]
                ref_ct_delta_offsets = ct_ref_pack.get(
                    "total_delta_offsets_xy",
                    ref_ct_offsets - ct_offsets_stage0,
                )
            else:
                ref_ct_xy = ref_ct["pred_xy"]
                ref_ct_conf = ref_ct.get("pred_conf", ct_conf)
                ref_ct_offsets = ref_ct.get(
                    "pred_offsets_xy",
                    ref_ct_xy - ref_ct_xy.mean(dim=2, keepdim=True),
                )
                ref_ct_delta_offsets = ref_ct.get("delta_offsets_xy", ref_ct_offsets - ct_offsets_stage0)
                ct_stage_xy_all.append(ref_ct_xy)
                ct_stage_conf_all.append(ref_ct_conf)
                ct_stage_offsets_all.append(ref_ct_offsets)
                ct_stage_delta_offsets_all.append(ref_ct_delta_offsets)
            ct_xy_stage1 = ct_stage_xy_all[1] if len(ct_stage_xy_all) > 1 else None
            ct_conf_stage1 = ct_stage_conf_all[1] if len(ct_stage_conf_all) > 1 else None
            ct_offsets_stage1 = ct_stage_offsets_all[1] if len(ct_stage_offsets_all) > 1 else None
            ct_delta_offsets_stage1 = ct_stage_delta_offsets_all[1] if len(ct_stage_delta_offsets_all) > 1 else None
            ct_xy_stage2 = ct_stage_xy_all[2] if len(ct_stage_xy_all) > 2 else None
            ct_conf_stage2 = ct_stage_conf_all[2] if len(ct_stage_conf_all) > 2 else None
            ct_offsets_stage2 = ct_stage_offsets_all[2] if len(ct_stage_offsets_all) > 2 else None
            ct_delta_offsets_stage2 = ct_stage_delta_offsets_all[2] if len(ct_stage_delta_offsets_all) > 2 else None
            ct_xy_stage3 = ct_stage_xy_all[3] if len(ct_stage_xy_all) > 3 else None
            ct_conf_stage3 = ct_stage_conf_all[3] if len(ct_stage_conf_all) > 3 else None
            ct_offsets_stage3 = ct_stage_offsets_all[3] if len(ct_stage_offsets_all) > 3 else None
            ct_delta_offsets_stage3 = ct_stage_delta_offsets_all[3] if len(ct_stage_delta_offsets_all) > 3 else None
            ct_xy_stage4 = ct_stage_xy_all[4] if len(ct_stage_xy_all) > 4 else None
            ct_conf_stage4 = ct_stage_conf_all[4] if len(ct_stage_conf_all) > 4 else None
            ct_offsets_stage4 = ct_stage_offsets_all[4] if len(ct_stage_offsets_all) > 4 else None
            ct_delta_offsets_stage4 = ct_stage_delta_offsets_all[4] if len(ct_stage_delta_offsets_all) > 4 else None

            t = _tic()
            tool_stub = self._xy_to_spatial_stub(ref_tool_xy, h_patches=h_decode, w_patches=w_decode)
            ct_stub = self._xy_to_spatial_stub(ref_ct_xy, h_patches=h_decode, w_patches=w_decode)
            _toc("spatial_stub_refined_ms", t)

            t = _tic()
            combo_logits = tool_xy.new_zeros((bsz, self.num_combo_classes))
            combo_tool_logits = tool_xy.new_zeros((bsz, self.num_tool_classes))
            combo_action_logits = tool_xy.new_zeros((bsz, self.num_action_classes))
            _toc("combo_head_ms", t)
            t = _tic()
            range_actionness_logits = tool_xy.new_zeros((bsz, t_out))
            range_offsets = tool_xy.new_zeros((bsz, t_out, 2))
            range_decoded = {
                "pred_action_start_norm": tool_xy.new_zeros((bsz,)),
                "pred_action_end_norm": tool_xy.new_zeros((bsz,)),
            }
            logits_start_bin = tool_xy.new_zeros((bsz, self.num_range_bins))
            logits_end_bin = tool_xy.new_zeros((bsz, self.num_range_bins))
            _toc("range_head_ms", t)
            _toc("model_total_ms", t_all)

            if self.debug_shapes:
                shape_debug["tool_probs"] = list(tool_stub["probs"].shape)
                shape_debug["ct_probs"] = list(ct_stub["probs"].shape)
                shape_debug["tool_pred_xy_pre_refine"] = list(tool_xy.shape)
                shape_debug["ct_pred_xy_pre_refine"] = list(ct_xy.shape)
                shape_debug["tool_pred_xy_post_refine"] = list(ref_tool_xy.shape)
                shape_debug["ct_pred_xy_post_refine"] = list(ref_ct_xy.shape)
                shape_debug["h_decode"] = int(h_decode)
                shape_debug["w_decode"] = int(w_decode)

            tool_xy_px = self._xy_norm_to_canvas_px(ref_tool_xy)
            ct_xy_px = self._xy_norm_to_canvas_px(ref_ct_xy)
            out = {
                "pred_tool_xy": ref_tool_xy,
                "pred_ct_xy": ref_ct_xy,
                "pred_tool_xy_stage0": tool_xy_stage0,
                "pred_ct_xy_stage0": ct_xy_stage0,
                "pred_tool_xy_stage1": tool_xy_stage1,
                "pred_ct_xy_stage1": ct_xy_stage1,
                "pred_tool_xy_stage_stack": torch.stack(tool_stage_xy_all, dim=0),
                "pred_ct_xy_stage_stack": torch.stack(ct_stage_xy_all, dim=0),
                "pred_tool_xy_refine": ref_tool_xy,
                "pred_ct_xy_refine": ref_ct_xy,
                "pred_tool_xy_px": tool_xy_px,
                "pred_ct_xy_px": ct_xy_px,
                "combo_logits": combo_logits,
                "combo_tool_logits": combo_tool_logits,
                "combo_action_logits": combo_action_logits,
                "logits_start_bin": logits_start_bin,
                "logits_end_bin": logits_end_bin,
                "range_actionness_logits": range_actionness_logits,
                "range_offsets": range_offsets,
                "pred_action_start_norm": range_decoded["pred_action_start_norm"],
                "pred_action_end_norm": range_decoded["pred_action_end_norm"],
                "conf_tool": ref_tool_conf,
                "conf_ct": ref_ct_conf,
                "conf_tool_stage0": tool_conf_stage0,
                "conf_ct_stage0": ct_conf_stage0,
                "conf_tool_stage_stack": torch.stack(tool_stage_conf_all, dim=0),
                "conf_ct_stage_stack": torch.stack(ct_stage_conf_all, dim=0),
                "probs_tool": tool_stub["probs"],
                "probs_ct": ct_stub["probs"],
                "heatmap_tool": tool_stub["heatmap"],
                "heatmap_ct": ct_stub["heatmap"],
                "logits_tool": tool_stub["logits"],
                "logits_ct": ct_stub["logits"],
                "traj_refine_temporal_tool_xy": ref_tool["temporal_xy"],
                "traj_refine_temporal_ct_xy": ref_ct["temporal_xy"],
                "traj_refine_delta_tool_xy": ref_tool_xy - tool_xy_stage0,
                "traj_refine_delta_ct_xy": ref_ct_xy - ct_xy_stage0,
                "traj_refine_delta_xy": torch.cat([ref_tool_xy - tool_xy_stage0, ref_ct_xy - ct_xy_stage0], dim=2),
                "traj_refine_delta_tool_gate": ref_tool["delta_gate"],
                "traj_refine_delta_ct_gate": ref_ct["delta_gate"],
                "traj_refine_delta_gate": torch.cat([ref_tool["delta_gate"], ref_ct["delta_gate"]], dim=2),
                "traj_base_delta_tool_xy": tool_delta,
                "traj_base_delta_ct_xy": ct_delta,
                "traj_base_delta_tool_gate": tool_gate,
                "traj_base_delta_ct_gate": ct_gate,
                "meta": {
                    "t_encoder": t_enc,
                    "t_out": t_out,
                    "t_prime": t_out,
                    "h_prime": h_decode,
                    "w_prime": w_decode,
                    "h_encoder": h_prime,
                    "w_encoder": w_prime,
                    "temporal_upsample": self.temporal_upsample,
                    "heatmap_softmax_temp": self.heatmap_softmax_temp,
                    "enable_delta_branch": self.enable_delta_branch,
                    "temporal_mean_residual_alpha": self.temporal_mean_residual_alpha,
                    "refiner_mode": self.traj_refiner_mode,
                    "trajectory_only": self.trajectory_only,
                    "range_only": self.range_only,
                    "trajectory_target": self.trajectory_target,
                    "traj_num_subframes": self.traj_num_subframes,
                    "traj_condition_mode": self.traj_condition_mode,
                    "traj_refiner_mode": self.traj_refiner_mode,
                    "tool_only": self.tool_only,
                    "combo_probe_attn": self.combo_probe_attn,
                    "combo_probe_dual_attn": self.combo_probe_dual_attn,
                    "combo_feature_mode": self.combo_feature_mode,
                    "combo_head_dropout": self.combo_head_dropout,
                    "combo_feat_dropout": self.combo_feat_dropout,
                    "dual_attn_tool_prior": bool(self.combo_action_tool_prior is not None),
                    "dual_attn_tool_prior_detach": bool(self.dual_attn_tool_prior_detach),
                    "dual_attn_tool_prior_alpha": (
                        0.0
                        if self.combo_action_tool_prior is None
                        else float(self.combo_action_tool_prior.alpha.detach().item())
                    ),
                    "num_tool_classes": self.num_tool_classes,
                    "num_action_classes": self.num_action_classes,
                    "traj_head_hidden_dim": self.traj_head_hidden_dim,
                    "traj_head_up_layers": self.traj_head_up_layers,
                    "traj_recurrent_max_delta_norm": self.traj_recurrent_max_delta_norm,
                    "traj_dynamic_centroid_max_beta": self.traj_dynamic_centroid_max_beta,
                    "traj_context_num_points": self.traj_context_num_points,
                    "traj_context_offset_max_cells": self.traj_context_offset_max_cells,
                    "traj_context_support_radius": self.traj_context_support_radius,
                    "traj_context_late_stage_count": self.traj_context_late_stage_count,
                    "traj_template_support_radius": self.traj_template_support_radius,
                    "traj_template_support_radii_by_stage": list(self.traj_template_support_radii_by_stage),
                    "traj_template_late_stage_count": self.traj_template_late_stage_count,
                    "traj_template_ema_alpha": self.traj_template_ema_alpha,
                    "traj_template_detach_current_inputs": bool(self.traj_template_detach_current_inputs),
                    "traj_tool_centroid_offset_head": self.traj_tool_centroid_offset_head,
                    "traj_tool_offset_max_norm": self.traj_tool_offset_max_norm,
                    "token_mask_enable": bool(self.token_mask_enable),
                    "token_mask_eval": bool(self.token_mask_eval),
                    "token_mask_strategy": self.token_mask_strategy,
                    "token_mask_value": self.token_mask_value,
                    "token_mask_p_max": float(self.token_mask_p_max),
                    "token_mask_p_early": float(self.token_mask_p_early),
                    "token_mask_delay_epochs": int(self.token_mask_delay_epochs),
                    "token_mask_warmup_epochs": int(self.token_mask_warmup_epochs),
                    "token_mask_late_start_epoch": int(self.token_mask_late_start_epoch),
                    "token_mask_p_late": float(self.token_mask_p_late),
                    "token_mask_late_warmup_epochs": int(self.token_mask_late_warmup_epochs),
                    "token_mask_keep_k": int(self.token_mask_keep_k),
                    "token_mask_mix_801010": bool(self.token_mask_mix_801010),
                    "token_mask_epoch": int(self.token_mask_epoch),
                    "token_mask_stats": token_mask_stats,
                    "num_range_bins": self.num_range_bins,
                    "output_canvas_size": self.output_canvas_size,
                    "token_order": token_order,
                    "n_tokens": n_tokens_meta,
                    "n_tokens_encoder": n_tokens_encoder,
                    "timing_ms": timing_ms,
                    "shape_debug": shape_debug,
                },
            }
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("corr_entropy", None)):
                out["traj_refine_corr_entropy_tool"] = tool_ref_pack["corr_entropy"]
                out["traj_refine_corr_margin_tool"] = tool_ref_pack["corr_margin"]
                if torch.is_tensor(tool_ref_pack.get("corr_max_prob", None)):
                    out["traj_refine_corr_max_prob_tool"] = tool_ref_pack["corr_max_prob"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    entropy_key = f"corr_entropy_stage{stage_idx}"
                    margin_key = f"corr_margin_stage{stage_idx}"
                    maxprob_key = f"corr_max_prob_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(entropy_key, None)):
                        out[f"traj_refine_corr_entropy_tool_stage{stage_idx}"] = tool_ref_pack[entropy_key]
                    if torch.is_tensor(tool_ref_pack.get(margin_key, None)):
                        out[f"traj_refine_corr_margin_tool_stage{stage_idx}"] = tool_ref_pack[margin_key]
                    if torch.is_tensor(tool_ref_pack.get(maxprob_key, None)):
                        out[f"traj_refine_corr_max_prob_tool_stage{stage_idx}"] = tool_ref_pack[maxprob_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("context_entropy", None)):
                out["traj_refine_context_entropy_tool"] = tool_ref_pack["context_entropy"]
                out["traj_refine_context_margin_tool"] = tool_ref_pack["context_margin"]
                if torch.is_tensor(tool_ref_pack.get("context_offset_abs", None)):
                    out["traj_refine_context_offset_abs_tool"] = tool_ref_pack["context_offset_abs"]
                if torch.is_tensor(tool_ref_pack.get("context_gate", None)):
                    out["traj_refine_context_gate_tool"] = tool_ref_pack["context_gate"]
                if torch.is_tensor(tool_ref_pack.get("context_shift_xy", None)):
                    out["traj_refine_context_shift_tool"] = tool_ref_pack["context_shift_xy"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    entropy_key = f"context_entropy_stage{stage_idx}"
                    margin_key = f"context_margin_stage{stage_idx}"
                    offset_key = f"context_offset_abs_stage{stage_idx}"
                    gate_key = f"context_gate_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(entropy_key, None)):
                        out[f"traj_refine_context_entropy_tool_stage{stage_idx}"] = tool_ref_pack[entropy_key]
                    if torch.is_tensor(tool_ref_pack.get(margin_key, None)):
                        out[f"traj_refine_context_margin_tool_stage{stage_idx}"] = tool_ref_pack[margin_key]
                    if torch.is_tensor(tool_ref_pack.get(offset_key, None)):
                        out[f"traj_refine_context_offset_abs_tool_stage{stage_idx}"] = tool_ref_pack[offset_key]
                    if torch.is_tensor(tool_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_context_gate_tool_stage{stage_idx}"] = tool_ref_pack[gate_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("belief_gate", None)):
                out["traj_refine_belief_gate_tool"] = tool_ref_pack["belief_gate"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    gate_key = f"belief_gate_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_belief_gate_tool_stage{stage_idx}"] = tool_ref_pack[gate_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("anchor_delta_centroid_abs", None)):
                out["traj_refine_anchor_delta_centroid_tool"] = tool_ref_pack["anchor_delta_centroid_abs"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    delta_key = f"anchor_delta_centroid_abs_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(delta_key, None)):
                        out[f"traj_refine_anchor_delta_centroid_tool_stage{stage_idx}"] = tool_ref_pack[delta_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("anchor_alpha", None)):
                out["traj_refine_anchor_alpha_tool"] = tool_ref_pack["anchor_alpha"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    alpha_key = f"anchor_alpha_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(alpha_key, None)):
                        out[f"traj_refine_anchor_alpha_tool_stage{stage_idx}"] = tool_ref_pack[alpha_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("anchor_verify_scale", None)):
                out["traj_refine_anchor_verify_scale_tool"] = tool_ref_pack["anchor_verify_scale"]
                if torch.is_tensor(tool_ref_pack.get("anchor_verify_residual", None)):
                    out["traj_refine_anchor_verify_residual_tool"] = tool_ref_pack["anchor_verify_residual"]
                if torch.is_tensor(tool_ref_pack.get("stage0_centroid_noise_abs", None)):
                    out["traj_refine_stage0_centroid_noise_abs_tool"] = tool_ref_pack["stage0_centroid_noise_abs"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    scale_key = f"anchor_verify_scale_stage{stage_idx}"
                    residual_key = f"anchor_verify_residual_stage{stage_idx}"
                    noise_key = f"stage0_centroid_noise_abs_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(scale_key, None)):
                        out[f"traj_refine_anchor_verify_scale_tool_stage{stage_idx}"] = tool_ref_pack[scale_key]
                    if torch.is_tensor(tool_ref_pack.get(residual_key, None)):
                        out[f"traj_refine_anchor_verify_residual_tool_stage{stage_idx}"] = tool_ref_pack[residual_key]
                    if torch.is_tensor(tool_ref_pack.get(noise_key, None)):
                        out[f"traj_refine_stage0_centroid_noise_abs_tool_stage{stage_idx}"] = tool_ref_pack[noise_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("factor_anchor_gate", None)):
                out["traj_refine_factor_anchor_gate_tool"] = tool_ref_pack["factor_anchor_gate"]
                if torch.is_tensor(tool_ref_pack.get("factor_offset_gate", None)):
                    out["traj_refine_factor_offset_gate_tool"] = tool_ref_pack["factor_offset_gate"]
                if torch.is_tensor(tool_ref_pack.get("factor_delta_offset_abs", None)):
                    out["traj_refine_factor_delta_offset_abs_tool"] = tool_ref_pack["factor_delta_offset_abs"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    anchor_gate_key = f"factor_anchor_gate_stage{stage_idx}"
                    offset_gate_key = f"factor_offset_gate_stage{stage_idx}"
                    offset_abs_key = f"factor_delta_offset_abs_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(anchor_gate_key, None)):
                        out[f"traj_refine_factor_anchor_gate_tool_stage{stage_idx}"] = tool_ref_pack[anchor_gate_key]
                    if torch.is_tensor(tool_ref_pack.get(offset_gate_key, None)):
                        out[f"traj_refine_factor_offset_gate_tool_stage{stage_idx}"] = tool_ref_pack[offset_gate_key]
                    if torch.is_tensor(tool_ref_pack.get(offset_abs_key, None)):
                        out[f"traj_refine_factor_delta_offset_abs_tool_stage{stage_idx}"] = tool_ref_pack[offset_abs_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("joint_gate", None)):
                out["traj_refine_joint_gate_tool"] = tool_ref_pack["joint_gate"]
                if torch.is_tensor(tool_ref_pack.get("joint_pair_dist", None)):
                    out["traj_refine_joint_pair_dist_tool"] = tool_ref_pack["joint_pair_dist"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    gate_key = f"joint_gate_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_joint_gate_tool_stage{stage_idx}"] = tool_ref_pack[gate_key]
            if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("template_entropy", None)):
                out["traj_refine_template_entropy_tool"] = tool_ref_pack["template_entropy"]
                out["traj_refine_template_margin_tool"] = tool_ref_pack["template_margin"]
                if torch.is_tensor(tool_ref_pack.get("template_gate", None)):
                    out["traj_refine_template_gate_tool"] = tool_ref_pack["template_gate"]
                if torch.is_tensor(tool_ref_pack.get("template_offset_abs", None)):
                    out["traj_refine_template_offset_abs_tool"] = tool_ref_pack["template_offset_abs"]
                if torch.is_tensor(tool_ref_pack.get("template_delta_abs", None)):
                    out["traj_refine_template_delta_abs_tool"] = tool_ref_pack["template_delta_abs"]
                for stage_idx, _ in enumerate(tool_ref_pack.get("stages", []), start=1):
                    entropy_key = f"template_entropy_stage{stage_idx}"
                    margin_key = f"template_margin_stage{stage_idx}"
                    gate_key = f"template_gate_stage{stage_idx}"
                    offset_key = f"template_offset_abs_stage{stage_idx}"
                    delta_key = f"template_delta_abs_stage{stage_idx}"
                    if torch.is_tensor(tool_ref_pack.get(entropy_key, None)):
                        out[f"traj_refine_template_entropy_tool_stage{stage_idx}"] = tool_ref_pack[entropy_key]
                    if torch.is_tensor(tool_ref_pack.get(margin_key, None)):
                        out[f"traj_refine_template_margin_tool_stage{stage_idx}"] = tool_ref_pack[margin_key]
                    if torch.is_tensor(tool_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_template_gate_tool_stage{stage_idx}"] = tool_ref_pack[gate_key]
                    if torch.is_tensor(tool_ref_pack.get(offset_key, None)):
                        out[f"traj_refine_template_offset_abs_tool_stage{stage_idx}"] = tool_ref_pack[offset_key]
                    if torch.is_tensor(tool_ref_pack.get(delta_key, None)):
                        out[f"traj_refine_template_delta_abs_tool_stage{stage_idx}"] = tool_ref_pack[delta_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("corr_entropy", None)):
                out["traj_refine_corr_entropy_ct"] = ct_ref_pack["corr_entropy"]
                out["traj_refine_corr_margin_ct"] = ct_ref_pack["corr_margin"]
                if torch.is_tensor(ct_ref_pack.get("total_delta_centroid_xy", None)):
                    out["traj_refine_centroid_shift_ct"] = ct_ref_pack["total_delta_centroid_xy"]
                if torch.is_tensor(ct_ref_pack.get("corr_max_prob", None)):
                    out["traj_refine_corr_max_prob_ct"] = ct_ref_pack["corr_max_prob"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    entropy_key = f"corr_entropy_stage{stage_idx}"
                    margin_key = f"corr_margin_stage{stage_idx}"
                    maxprob_key = f"corr_max_prob_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(entropy_key, None)):
                        out[f"traj_refine_corr_entropy_ct_stage{stage_idx}"] = ct_ref_pack[entropy_key]
                    if torch.is_tensor(ct_ref_pack.get(margin_key, None)):
                        out[f"traj_refine_corr_margin_ct_stage{stage_idx}"] = ct_ref_pack[margin_key]
                    if torch.is_tensor(ct_ref_pack.get(maxprob_key, None)):
                        out[f"traj_refine_corr_max_prob_ct_stage{stage_idx}"] = ct_ref_pack[maxprob_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("context_entropy", None)):
                out["traj_refine_context_entropy_ct"] = ct_ref_pack["context_entropy"]
                out["traj_refine_context_margin_ct"] = ct_ref_pack["context_margin"]
                if torch.is_tensor(ct_ref_pack.get("context_offset_abs", None)):
                    out["traj_refine_context_offset_abs_ct"] = ct_ref_pack["context_offset_abs"]
                if torch.is_tensor(ct_ref_pack.get("context_gate", None)):
                    out["traj_refine_context_gate_ct"] = ct_ref_pack["context_gate"]
                if torch.is_tensor(ct_ref_pack.get("context_shift_xy", None)):
                    out["traj_refine_context_shift_ct"] = ct_ref_pack["context_shift_xy"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    entropy_key = f"context_entropy_stage{stage_idx}"
                    margin_key = f"context_margin_stage{stage_idx}"
                    offset_key = f"context_offset_abs_stage{stage_idx}"
                    gate_key = f"context_gate_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(entropy_key, None)):
                        out[f"traj_refine_context_entropy_ct_stage{stage_idx}"] = ct_ref_pack[entropy_key]
                    if torch.is_tensor(ct_ref_pack.get(margin_key, None)):
                        out[f"traj_refine_context_margin_ct_stage{stage_idx}"] = ct_ref_pack[margin_key]
                    if torch.is_tensor(ct_ref_pack.get(offset_key, None)):
                        out[f"traj_refine_context_offset_abs_ct_stage{stage_idx}"] = ct_ref_pack[offset_key]
                    if torch.is_tensor(ct_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_context_gate_ct_stage{stage_idx}"] = ct_ref_pack[gate_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("belief_gate", None)):
                out["traj_refine_belief_gate_ct"] = ct_ref_pack["belief_gate"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    gate_key = f"belief_gate_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_belief_gate_ct_stage{stage_idx}"] = ct_ref_pack[gate_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("anchor_delta_centroid_abs", None)):
                out["traj_refine_anchor_delta_centroid_ct"] = ct_ref_pack["anchor_delta_centroid_abs"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    delta_key = f"anchor_delta_centroid_abs_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(delta_key, None)):
                        out[f"traj_refine_anchor_delta_centroid_ct_stage{stage_idx}"] = ct_ref_pack[delta_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("anchor_alpha", None)):
                out["traj_refine_anchor_alpha_ct"] = ct_ref_pack["anchor_alpha"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    alpha_key = f"anchor_alpha_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(alpha_key, None)):
                        out[f"traj_refine_anchor_alpha_ct_stage{stage_idx}"] = ct_ref_pack[alpha_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("anchor_verify_scale", None)):
                out["traj_refine_anchor_verify_scale_ct"] = ct_ref_pack["anchor_verify_scale"]
                if torch.is_tensor(ct_ref_pack.get("anchor_verify_residual", None)):
                    out["traj_refine_anchor_verify_residual_ct"] = ct_ref_pack["anchor_verify_residual"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    scale_key = f"anchor_verify_scale_stage{stage_idx}"
                    residual_key = f"anchor_verify_residual_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(scale_key, None)):
                        out[f"traj_refine_anchor_verify_scale_ct_stage{stage_idx}"] = ct_ref_pack[scale_key]
                    if torch.is_tensor(ct_ref_pack.get(residual_key, None)):
                        out[f"traj_refine_anchor_verify_residual_ct_stage{stage_idx}"] = ct_ref_pack[residual_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("factor_anchor_gate", None)):
                out["traj_refine_factor_anchor_gate_ct"] = ct_ref_pack["factor_anchor_gate"]
                if torch.is_tensor(ct_ref_pack.get("factor_offset_gate", None)):
                    out["traj_refine_factor_offset_gate_ct"] = ct_ref_pack["factor_offset_gate"]
                if torch.is_tensor(ct_ref_pack.get("factor_delta_offset_abs", None)):
                    out["traj_refine_factor_delta_offset_abs_ct"] = ct_ref_pack["factor_delta_offset_abs"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    anchor_gate_key = f"factor_anchor_gate_stage{stage_idx}"
                    offset_gate_key = f"factor_offset_gate_stage{stage_idx}"
                    offset_abs_key = f"factor_delta_offset_abs_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(anchor_gate_key, None)):
                        out[f"traj_refine_factor_anchor_gate_ct_stage{stage_idx}"] = ct_ref_pack[anchor_gate_key]
                    if torch.is_tensor(ct_ref_pack.get(offset_gate_key, None)):
                        out[f"traj_refine_factor_offset_gate_ct_stage{stage_idx}"] = ct_ref_pack[offset_gate_key]
                    if torch.is_tensor(ct_ref_pack.get(offset_abs_key, None)):
                        out[f"traj_refine_factor_delta_offset_abs_ct_stage{stage_idx}"] = ct_ref_pack[offset_abs_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("joint_gate", None)):
                out["traj_refine_joint_gate_ct"] = ct_ref_pack["joint_gate"]
                if torch.is_tensor(ct_ref_pack.get("joint_pair_dist", None)):
                    out["traj_refine_joint_pair_dist_ct"] = ct_ref_pack["joint_pair_dist"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    gate_key = f"joint_gate_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_joint_gate_ct_stage{stage_idx}"] = ct_ref_pack[gate_key]
            if ct_ref_pack is not None and torch.is_tensor(ct_ref_pack.get("template_entropy", None)):
                out["traj_refine_template_entropy_ct"] = ct_ref_pack["template_entropy"]
                out["traj_refine_template_margin_ct"] = ct_ref_pack["template_margin"]
                if torch.is_tensor(ct_ref_pack.get("template_gate", None)):
                    out["traj_refine_template_gate_ct"] = ct_ref_pack["template_gate"]
                if torch.is_tensor(ct_ref_pack.get("template_offset_abs", None)):
                    out["traj_refine_template_offset_abs_ct"] = ct_ref_pack["template_offset_abs"]
                if torch.is_tensor(ct_ref_pack.get("template_delta_abs", None)):
                    out["traj_refine_template_delta_abs_ct"] = ct_ref_pack["template_delta_abs"]
                for stage_idx, _ in enumerate(ct_ref_pack.get("stages", []), start=1):
                    entropy_key = f"template_entropy_stage{stage_idx}"
                    margin_key = f"template_margin_stage{stage_idx}"
                    gate_key = f"template_gate_stage{stage_idx}"
                    offset_key = f"template_offset_abs_stage{stage_idx}"
                    delta_key = f"template_delta_abs_stage{stage_idx}"
                    if torch.is_tensor(ct_ref_pack.get(entropy_key, None)):
                        out[f"traj_refine_template_entropy_ct_stage{stage_idx}"] = ct_ref_pack[entropy_key]
                    if torch.is_tensor(ct_ref_pack.get(margin_key, None)):
                        out[f"traj_refine_template_margin_ct_stage{stage_idx}"] = ct_ref_pack[margin_key]
                    if torch.is_tensor(ct_ref_pack.get(gate_key, None)):
                        out[f"traj_refine_template_gate_ct_stage{stage_idx}"] = ct_ref_pack[gate_key]
                    if torch.is_tensor(ct_ref_pack.get(offset_key, None)):
                        out[f"traj_refine_template_offset_abs_ct_stage{stage_idx}"] = ct_ref_pack[offset_key]
                    if torch.is_tensor(ct_ref_pack.get(delta_key, None)):
                        out[f"traj_refine_template_delta_abs_ct_stage{stage_idx}"] = ct_ref_pack[delta_key]
            if (tool_rec is not None) and ("pred_centroid_xy" in tool_rec):
                out["traj_tool_centroid_xy"] = ref_tool_centroid
                out["traj_tool_centroid_xy_stage0"] = tool_rec["pred_centroid_xy"]
                out["traj_tool_centroid_xy_stage_stack"] = torch.stack(tool_stage_centroid_all, dim=0)
                if tool_ref_pack is not None and torch.is_tensor(tool_ref_pack.get("total_delta_centroid_xy", None)):
                    out["traj_refine_centroid_shift_tool"] = tool_ref_pack["total_delta_centroid_xy"]
            if (tool_rec is not None) and ("pred_offsets_xy" in tool_rec):
                out["traj_tool_offsets_xy_stage0"] = tool_rec["pred_offsets_xy"]
                out["traj_tool_offsets_xy"] = ref_tool_offsets
            elif len(tool_stage_offsets_all) > 1:
                out["traj_tool_offsets_xy"] = ref_tool_offsets
            if len(tool_stage_offsets_all) > 0:
                out["traj_tool_offsets_xy_stage_stack"] = torch.stack(tool_stage_offsets_all, dim=0)
            if len(tool_stage_offsets_all) > 1:
                out["traj_tool_delta_offset_xy"] = ref_tool_delta_offsets
            out["traj_ct_offsets_xy"] = ref_ct_offsets
            if len(ct_stage_offsets_all) > 0:
                out["traj_ct_offsets_xy_stage0"] = ct_offsets_stage0
                out["traj_ct_offsets_xy_stage_stack"] = torch.stack(ct_stage_offsets_all, dim=0)
            if len(ct_stage_offsets_all) > 1:
                out["traj_ct_delta_offset_xy"] = ref_ct_delta_offsets
            for stage_idx in range(1, len(tool_stage_xy_all)):
                out[f"pred_tool_xy_stage{stage_idx}"] = tool_stage_xy_all[stage_idx]
                out[f"conf_tool_stage{stage_idx}"] = tool_stage_conf_all[stage_idx]
                out[f"traj_tool_offsets_xy_stage{stage_idx}"] = tool_stage_offsets_all[stage_idx]
                out[f"traj_tool_delta_offset_xy_stage{stage_idx}"] = tool_stage_delta_offsets_all[stage_idx]
                out[f"traj_refine_delta_tool_xy_stage{stage_idx}"] = (
                    tool_stage_xy_all[stage_idx] - tool_stage_xy_all[stage_idx - 1]
                )
            for stage_idx in range(1, len(ct_stage_xy_all)):
                out[f"pred_ct_xy_stage{stage_idx}"] = ct_stage_xy_all[stage_idx]
                out[f"conf_ct_stage{stage_idx}"] = ct_stage_conf_all[stage_idx]
                out[f"traj_ct_offsets_xy_stage{stage_idx}"] = ct_stage_offsets_all[stage_idx]
                out[f"traj_ct_delta_offset_xy_stage{stage_idx}"] = ct_stage_delta_offsets_all[stage_idx]
                out[f"traj_refine_delta_ct_xy_stage{stage_idx}"] = (
                    ct_stage_xy_all[stage_idx] - ct_stage_xy_all[stage_idx - 1]
                )
            return out

        t = _tic()
        x = tokens_grid.reshape(bsz, t_enc, s, d)  # [B,T_enc,S,D]
        x_proj = self.shared_trunk(x)  # [B,T_enc,S,d_proj]
        if self.debug_shapes:
            shape_debug["tokens_grid"] = list(tokens_grid.shape)
            shape_debug["x_tokens"] = list(x.shape)
            shape_debug["x_proj"] = list(x_proj.shape)
        _toc("shared_trunk_ms", t)

        t = _tic()
        feat0 = x_proj.reshape(bsz, t_enc, h_prime, w_prime, -1).permute(0, 1, 4, 2, 3).reshape(
            bsz * t_enc, -1, h_prime, w_prime
        )
        feat = self.conv_refine(feat0)  # [B*T_enc,d_proj,H',W']
        feat = self.temporal_mean_residual(feat.reshape(bsz, t_enc, -1, h_prime, w_prime)).reshape(
            bsz * t_enc, -1, h_prime, w_prime
        )
        feat_flat = feat.reshape(bsz, t_enc, -1, h_prime, w_prime).permute(0, 1, 3, 4, 2).reshape(bsz, t_enc, s, -1)
        if self.debug_shapes:
            shape_debug["feat_btchw"] = list(feat.shape)
            shape_debug["feat_flat_btsd"] = list(feat_flat.shape)
        _toc("conv_refine_ms", t)

        if self.range_only:
            t = _tic()
            f_mean = feat_flat.mean(dim=2)  # [B,T_enc,d]
            f_max = feat_flat.amax(dim=2)  # [B,T_enc,d]
            if self.range_temporal_pool is None:
                raise RuntimeError("range_only=True requires range_temporal_pool.")
            f_base = torch.cat((f_mean, f_max), dim=-1)  # [B,T_enc,2d]
            spatial_pool_out = None
            if self.range_spatial_pool is not None:
                spatial_pool_out = self.range_spatial_pool(
                    feat_flat,
                    h_patches=h_prime,
                    w_patches=w_prime,
                    tool_id=tool_id,
                    action_id=action_id,
                    combo_id=combo_id,
                )
            if self.range_spatial_pool_mode == "mean_max_condtoken":
                assert spatial_pool_out is not None
                f_t_score_pre = None
                f_t = self.range_temporal_pool(torch.cat((f_mean, f_max, spatial_pool_out["pooled"]), dim=-1))
            else:
                f_t_score_pre = self.range_temporal_pool(f_base)  # [B,T_enc,d]
                if self.range_spatial_pool_mode == "mean_max_condtoken_locresid":
                    if spatial_pool_out is None or self.range_spatial_loc_residual is None:
                        raise RuntimeError(
                            "range_spatial_pool_mode=mean_max_condtoken_locresid requires "
                            "range_spatial_pool and range_spatial_loc_residual."
                        )
                    loc_delta = self.range_spatial_loc_residual(
                        torch.cat((f_mean, f_max, spatial_pool_out["pooled"]), dim=-1)
                    )
                    f_t = f_t_score_pre + loc_delta
                else:
                    f_t = f_t_score_pre
            if self.output_num_frames > 0 and self.output_num_frames != t_enc:
                f_t = self._temporal_interpolate_dim1(f_t, target_t=self.output_num_frames)
                if torch.is_tensor(f_t_score_pre):
                    f_t_score_pre = self._temporal_interpolate_dim1(f_t_score_pre, target_t=self.output_num_frames)
            if self.range_temporal_context is not None:
                f_t = self.range_temporal_context(f_t)
                if torch.is_tensor(f_t_score_pre):
                    f_t_score_pre = self.range_temporal_context(f_t_score_pre)
            range_sola_projected = None
            range_score_projected = None
            f_t_loc = f_t
            f_t_score = None
            if self.range_temporal_adapter is not None:
                adapter_out = self.range_temporal_adapter(f_t)
                f_t = adapter_out["adapted"]
                f_t_loc = adapter_out.get("reg_adapted", f_t)
                range_sola_projected = adapter_out.get("reg_projected", adapter_out.get("projected"))
                if self.range_task_decouple_mode in ("score_loc", "score_loc_align"):
                    if torch.is_tensor(f_t_score_pre):
                        score_adapter_out = self.range_temporal_adapter(f_t_score_pre)
                        f_t_score = score_adapter_out.get("score_adapted", score_adapter_out.get("adapted"))
                        range_score_projected = score_adapter_out.get(
                            "score_projected",
                            score_adapter_out.get("projected", range_sola_projected),
                        )
                    else:
                        f_t_score = adapter_out.get("score_adapted", f_t_loc)
                        range_score_projected = adapter_out.get("score_projected", range_sola_projected)
                else:
                    range_score_projected = adapter_out.get("score_projected", range_sola_projected)
                    f_t_loc = f_t
                    f_t_score = None
            else:
                if self.range_task_decouple_mode in ("score_loc", "score_loc_align") and torch.is_tensor(f_t_score_pre):
                    f_t_score = f_t_score_pre
                else:
                    f_t_loc = f_t
                    f_t_score = None
            t_out = int(f_t_loc.shape[1])
            if self.range_head is None:
                raise RuntimeError("range_only=True requires range_head.")
            range_out = self.range_head(
                f_t_loc,
                score_f_t=f_t_score,
                tool_id=tool_id,
                action_id=action_id,
                combo_id=combo_id,
            )
            range_actionness_logits = range_out["range_actionness_logits"]
            range_offsets = range_out["range_offsets"]
            range_start_logits = range_out.get(
                "range_start_logits",
                range_actionness_logits.new_zeros(range_actionness_logits.shape),
            )
            range_end_logits = range_out.get(
                "range_end_logits",
                range_actionness_logits.new_zeros(range_actionness_logits.shape),
            )
            range_proposal_logits = range_out.get("range_proposal_logits", None)
            range_quality_logits = range_out.get("range_quality_logits", None)
            if torch.is_tensor(range_proposal_logits) and self.range_proposal_decode_mode == "argmax":
                range_decoded = self._decode_range_from_proposals(
                    range_start_logits=range_start_logits,
                    range_end_logits=range_end_logits,
                    range_proposal_logits=range_proposal_logits,
                )
            else:
                range_decoded = self._decode_range_from_points(
                    range_actionness_logits=range_actionness_logits,
                    range_offsets=range_offsets,
                    range_quality_logits=range_quality_logits,
                    combo_duration_prior_mean=(
                        self.range_duration_prior_means[combo_id.long()]
                        if (self.range_duration_prior_mode == "combo_gaussian" and torch.is_tensor(combo_id))
                        else None
                    ),
                    combo_duration_prior_std=(
                        self.range_duration_prior_stds[combo_id.long()].clamp_min(self.range_duration_prior_min_std)
                        if (self.range_duration_prior_mode == "combo_gaussian" and torch.is_tensor(combo_id))
                        else None
                    ),
                    combo_duration_prior_weight=self.range_duration_prior_weight,
                )
            if self.range_boundary_refine_mode == "local_argmax":
                range_decoded = self._refine_range_boundaries_local_argmax(
                    decoded=range_decoded,
                    range_start_logits=range_start_logits,
                    range_end_logits=range_end_logits,
                    radius=self.range_boundary_refine_radius,
                )
            logits_start_bin = self._norm_to_bin_logits(
                norm=range_decoded["pred_action_start_norm"],
                num_bins=self.num_range_bins,
            )
            logits_end_bin = self._norm_to_bin_logits(
                norm=range_decoded["pred_action_end_norm"],
                num_bins=self.num_range_bins,
            )
            zeros_tool_xy = feat_flat.new_zeros((bsz, t_out, self.num_tool_points, 2))
            zeros_ct_xy = feat_flat.new_zeros((bsz, t_out, self.num_contact_points, 2))
            zeros_tool_conf = feat_flat.new_zeros((bsz, t_out, self.num_tool_points))
            zeros_ct_conf = feat_flat.new_zeros((bsz, t_out, self.num_contact_points))
            zeros_tool_probs = feat_flat.new_zeros((bsz, t_out, self.num_tool_points, s))
            zeros_ct_probs = feat_flat.new_zeros((bsz, t_out, self.num_contact_points, s))
            zeros_tool_hm = feat_flat.new_zeros((bsz, t_out, self.num_tool_points, h_prime, w_prime))
            zeros_ct_hm = feat_flat.new_zeros((bsz, t_out, self.num_contact_points, h_prime, w_prime))
            zeros_combo = feat_flat.new_zeros((bsz, self.num_combo_classes))
            zeros_tool_cls = feat_flat.new_zeros((bsz, self.num_tool_classes))
            zeros_action_cls = feat_flat.new_zeros((bsz, self.num_action_classes))
            _toc("range_head_ms", t)
            _toc("model_total_ms", t_all)
            return {
                "pred_tool_xy": zeros_tool_xy,
                "pred_ct_xy": zeros_ct_xy,
                "pred_tool_xy_px": self._xy_norm_to_canvas_px(zeros_tool_xy),
                "pred_ct_xy_px": self._xy_norm_to_canvas_px(zeros_ct_xy),
                "combo_logits": zeros_combo,
                "combo_tool_logits": zeros_tool_cls,
                "combo_action_logits": zeros_action_cls,
                "logits_start_bin": logits_start_bin,
                "logits_end_bin": logits_end_bin,
                "range_actionness_logits": range_actionness_logits,
                "range_offsets": range_offsets,
                "range_start_logits": range_start_logits,
                "range_end_logits": range_end_logits,
                "range_proposal_logits": (
                    range_proposal_logits
                    if torch.is_tensor(range_proposal_logits)
                    else range_actionness_logits.new_zeros((bsz, t_out, t_out))
                ),
                "range_sola_projected": range_sola_projected,
                "range_score_projected": range_score_projected,
                "range_quality_logits": (
                    range_quality_logits
                    if torch.is_tensor(range_quality_logits)
                    else range_actionness_logits.new_zeros((bsz, t_out))
                ),
                "pred_action_start_norm": range_decoded["pred_action_start_norm"],
                "pred_action_end_norm": range_decoded["pred_action_end_norm"],
                "conf_tool": zeros_tool_conf,
                "conf_ct": zeros_ct_conf,
                "probs_tool": zeros_tool_probs,
                "probs_ct": zeros_ct_probs,
                "heatmap_tool": zeros_tool_hm,
                "heatmap_ct": zeros_ct_hm,
                "logits_tool": zeros_tool_probs,
                "logits_ct": zeros_ct_probs,
                "traj_refine_temporal_tool_xy": zeros_tool_xy,
                "traj_refine_temporal_ct_xy": zeros_ct_xy,
                "traj_refine_delta_tool_xy": zeros_tool_xy,
                "traj_refine_delta_ct_xy": zeros_ct_xy,
                "traj_refine_delta_xy": feat_flat.new_zeros((bsz, t_out, self.num_tool_points + self.num_contact_points, 2)),
                "traj_refine_delta_tool_gate": zeros_tool_conf,
                "traj_refine_delta_ct_gate": zeros_ct_conf,
                "meta": {
                    "t_encoder": t_enc,
                    "t_out": t_out,
                    "t_prime": t_out,
                    "h_prime": h_prime,
                    "w_prime": w_prime,
                    "temporal_upsample": self.temporal_upsample,
                    "range_only": self.range_only,
                    "trajectory_only": self.trajectory_only,
                    "tool_only": self.tool_only,
                    "combo_probe_attn": self.combo_probe_attn,
                    "combo_probe_dual_attn": self.combo_probe_dual_attn,
                    "range_condition_mode": self.range_condition_mode,
                    "range_head_mode": self.range_head_mode,
                    "range_trident_num_bins": self.range_trident_num_bins,
                    "range_proposal_mode": self.range_proposal_mode,
                    "range_proposal_decode_mode": self.range_proposal_decode_mode,
                    "range_quality_mode": self.range_quality_mode,
                    "range_boundary_refine_mode": self.range_boundary_refine_mode,
                    "range_boundary_refine_radius": self.range_boundary_refine_radius,
                    "range_temporal_context_mode": self.range_temporal_context_mode,
                    "range_temporal_context_kernel_size": self.range_temporal_context_kernel_size,
                    "range_temporal_context_layers": self.range_temporal_context_layers,
                    "range_temporal_context_levels": self.range_temporal_context_levels,
                    "range_adapter_mode": self.range_adapter_mode,
                    "range_adapter_kernel_size": self.range_adapter_kernel_size,
                    "range_adapter_num_heads": self.range_adapter_num_heads,
                    "range_task_decouple_mode": self.range_task_decouple_mode,
                    "num_range_bins": self.num_range_bins,
                    "token_order": token_order,
                    "n_tokens": n_tokens_meta,
                    "n_tokens_encoder": n_tokens_encoder,
                    "timing_ms": timing_ms,
                    "shape_debug": shape_debug,
                },
            }

        t = _tic()
        h_decode = h_prime
        w_decode = w_prime
        grid_xy = build_grid_xy(h_decode, w_decode, device=feat.device, dtype=feat.dtype)  # [S,2]
        if self.tool_head is None or self.contact_head is None:
            raise RuntimeError("Standard mode requires offset heads to be initialized.")
        tool_raw = self.tool_head(feat, bsz=bsz, t_prime=t_enc, h_patches=h_decode, w_patches=w_decode)
        ct_raw = self.contact_head(feat, bsz=bsz, t_prime=t_enc, h_patches=h_decode, w_patches=w_decode)
        tool_low = self._decode_spatial_outputs(
            logits=tool_raw["logits"],
            delta_raw=tool_raw["delta_raw"],
            grid_xy=grid_xy,
            h_patches=h_decode,
            w_patches=w_decode,
        )
        ct_low = self._decode_spatial_outputs(
            logits=ct_raw["logits"],
            delta_raw=ct_raw["delta_raw"],
            grid_xy=grid_xy,
            h_patches=h_decode,
            w_patches=w_decode,
        )
        if self.debug_shapes:
            shape_debug["tool_probs"] = list(tool_low["probs"].shape)
            shape_debug["ct_probs"] = list(ct_low["probs"].shape)
            shape_debug["tool_pred_xy_pre_refine"] = list(tool_low["pred_xy"].shape)
            shape_debug["ct_pred_xy_pre_refine"] = list(ct_low["pred_xy"].shape)
            shape_debug["h_decode"] = int(h_decode)
            shape_debug["w_decode"] = int(w_decode)
        _toc("spatial_heads_decode_ms", t)

        # Refine at encoder time (before temporal upsample).
        t = _tic()
        use_local_feat = (not self.trajectory_only) and (self.refiner_mode in ("attn", "lite"))
        if use_local_feat:
            feat_refine = feat_flat
            tool_local_feat = self._local_feat_from_argmax_cell(tool_low["probs"], feat_refine)  # [B,T_enc,P,d]
            ct_local_feat = self._local_feat_from_argmax_cell(ct_low["probs"], feat_refine)  # [B,T_enc,P,d]
        else:
            feat_refine = None
            tool_local_feat = None
            ct_local_feat = None
        if self.debug_shapes:
            shape_debug["feat_refine_btsd"] = [] if feat_refine is None else list(feat_refine.shape)
            shape_debug["tool_local_feat"] = [] if tool_local_feat is None else list(tool_local_feat.shape)
            shape_debug["ct_local_feat"] = [] if ct_local_feat is None else list(ct_local_feat.shape)
        _toc("local_feat_weighted_ms", t)
        t = _tic()
        ref_tool_low = self.tool_refiner(
            pred_xy=tool_low["pred_xy"],
            pred_conf=tool_low["conf"],
            local_feat=tool_local_feat,
        )
        _toc("refiner_tool_ms", t)
        t = _tic()
        ref_ct_low = self.contact_refiner(
            pred_xy=ct_low["pred_xy"],
            pred_conf=ct_low["conf"],
            local_feat=ct_local_feat,
        )
        if self.debug_shapes:
            shape_debug["tool_refiner"] = ref_tool_low.get("shape_debug", None)
            shape_debug["contact_refiner"] = ref_ct_low.get("shape_debug", None)
        _toc("refiner_contact_ms", t)

        # Base spatial outputs are upsampled for output/loss alignment.
        t_out = t_enc
        tool_out = tool_low
        ct_out = ct_low
        t = _tic()
        if self.output_num_frames > 0 and self.output_num_frames != t_enc:
            t_out = self.output_num_frames
            tool_out = self._upsample_spatial_outputs(
                tool_raw,
                grid_xy,
                h_decode,
                w_decode,
                target_t=t_out,
                head_name="tool",
            )
            ct_out = self._upsample_spatial_outputs(
                ct_raw,
                grid_xy,
                h_decode,
                w_decode,
                target_t=t_out,
                head_name="contact",
            )
        _toc("temporal_upsample_ms", t)

        # Upsample refined trajectories from encoder-time to output-time when needed.
        ref_tool_xy = ref_tool_low["pred_xy"] if int(ref_tool_low["pred_xy"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_tool_low["pred_xy"], target_t=t_out)
        ref_ct_xy = ref_ct_low["pred_xy"] if int(ref_ct_low["pred_xy"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_ct_low["pred_xy"], target_t=t_out)
        ref_tool_temporal_xy = ref_tool_low["temporal_xy"] if int(ref_tool_low["temporal_xy"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_tool_low["temporal_xy"], target_t=t_out)
        ref_ct_temporal_xy = ref_ct_low["temporal_xy"] if int(ref_ct_low["temporal_xy"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_ct_low["temporal_xy"], target_t=t_out)
        ref_tool_delta = ref_tool_low["delta_xy"] if int(ref_tool_low["delta_xy"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_tool_low["delta_xy"], target_t=t_out)
        ref_ct_delta = ref_ct_low["delta_xy"] if int(ref_ct_low["delta_xy"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_ct_low["delta_xy"], target_t=t_out)
        ref_tool_gate = ref_tool_low["delta_gate"] if int(ref_tool_low["delta_gate"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_tool_low["delta_gate"], target_t=t_out)
        ref_ct_gate = ref_ct_low["delta_gate"] if int(ref_ct_low["delta_gate"].shape[1]) == int(t_out) else self._temporal_interpolate_dim1(ref_ct_low["delta_gate"], target_t=t_out)

        if self.trajectory_only:
            t = _tic()
            combo_logits = tool_out["pred_xy"].new_zeros((bsz, self.num_combo_classes))
            combo_tool_logits = tool_out["pred_xy"].new_zeros((bsz, self.num_tool_classes))
            combo_action_logits = tool_out["pred_xy"].new_zeros((bsz, self.num_action_classes))
            _toc("combo_head_ms", t)
            t = _tic()
            range_actionness_logits = tool_out["pred_xy"].new_zeros((bsz, t_out))
            range_offsets = tool_out["pred_xy"].new_zeros((bsz, t_out, 2))
            range_start_logits = tool_out["pred_xy"].new_zeros((bsz, t_out))
            range_end_logits = tool_out["pred_xy"].new_zeros((bsz, t_out))
            range_proposal_logits = tool_out["pred_xy"].new_zeros((bsz, t_out, t_out))
            range_quality_logits = tool_out["pred_xy"].new_zeros((bsz, t_out))
            range_decoded = {
                "pred_action_start_norm": tool_out["pred_xy"].new_zeros((bsz,)),
                "pred_action_end_norm": tool_out["pred_xy"].new_zeros((bsz,)),
            }
            logits_start_bin = tool_out["pred_xy"].new_zeros((bsz, self.num_range_bins))
            logits_end_bin = tool_out["pred_xy"].new_zeros((bsz, self.num_range_bins))
            _toc("range_head_ms", t)
        else:
            t = _tic()
            if self.combo_feature_mode == "direct_spatiotemporal":
                # Direct pooling path for pure classification.
                combo_feat = feat_flat.mean(dim=(1, 2))  # [B,d]
            else:
                # Original tool-heatmap-weighted path.
                tool_feat = torch.einsum("btps,btsd->btpd", tool_low["probs"], feat_flat)  # [B,T_enc,3,d]
                combo_feat = tool_feat.mean(dim=(1, 2))  # [B,d]
            if self.combo_feat_dropout > 0.0:
                combo_feat = F.dropout(combo_feat, p=self.combo_feat_dropout, training=self.training)
            if self.combo_tool_head is None:
                raise RuntimeError("Standard mode requires combo_tool_head.")
            combo_tool_logits = self.combo_tool_head(combo_feat)  # [B,4]
            if self.tool_only:
                combo_action_logits = combo_tool_logits.new_zeros((bsz, self.num_action_classes))
                combo_logits = combo_tool_logits.new_zeros((bsz, self.num_combo_classes))
                _toc("combo_head_ms", t)
                t = _tic()
                range_actionness_logits = combo_tool_logits.new_zeros((bsz, t_out))
                range_offsets = combo_tool_logits.new_zeros((bsz, t_out, 2))
                range_start_logits = combo_tool_logits.new_zeros((bsz, t_out))
                range_end_logits = combo_tool_logits.new_zeros((bsz, t_out))
                range_proposal_logits = combo_tool_logits.new_zeros((bsz, t_out, t_out))
                range_quality_logits = combo_tool_logits.new_zeros((bsz, t_out))
                range_decoded = {
                    "pred_action_start_norm": combo_tool_logits.new_zeros((bsz,)),
                    "pred_action_end_norm": combo_tool_logits.new_zeros((bsz,)),
                }
                logits_start_bin = combo_tool_logits.new_zeros((bsz, self.num_range_bins))
                logits_end_bin = combo_tool_logits.new_zeros((bsz, self.num_range_bins))
                _toc("range_head_ms", t)
            else:
                if self.combo_action_head is None:
                    raise RuntimeError("Standard mode requires combo_action_head.")
                combo_action_logits = self.combo_action_head(combo_feat)  # [B,4]
                combo_logits = self._structured_combo_logits(
                    combo_tool_logits=combo_tool_logits,
                    combo_action_logits=combo_action_logits,
                )  # [B,num_classes]
                _toc("combo_head_ms", t)
                # Temporal descriptor for range bins: mean over spatial tokens.
                t = _tic()
                f_t = feat_flat.mean(dim=2)  # [B,T_enc,d]
                if self.output_num_frames > 0 and self.output_num_frames != t_enc:
                    f_t = self._temporal_interpolate_dim1(f_t, target_t=self.output_num_frames)
                if self.range_head is None:
                    raise RuntimeError("Standard mode requires range head.")
                range_out = self.range_head(
                    f_t,
                    score_f_t=f_t if self.range_task_decouple_mode == "score_loc" else None,
                    tool_id=tool_id,
                    action_id=action_id,
                    combo_id=combo_id,
                )
                range_actionness_logits = range_out["range_actionness_logits"]
                range_offsets = range_out["range_offsets"]
                range_start_logits = range_out.get(
                    "range_start_logits",
                    range_actionness_logits.new_zeros(range_actionness_logits.shape),
                )
                range_end_logits = range_out.get(
                    "range_end_logits",
                    range_actionness_logits.new_zeros(range_actionness_logits.shape),
                )
                range_proposal_logits = range_out.get("range_proposal_logits", None)
                range_quality_logits = range_out.get("range_quality_logits", None)
                if torch.is_tensor(range_proposal_logits) and self.range_proposal_decode_mode == "argmax":
                    range_decoded = self._decode_range_from_proposals(
                        range_start_logits=range_start_logits,
                        range_end_logits=range_end_logits,
                        range_proposal_logits=range_proposal_logits,
                    )
                else:
                    range_decoded = self._decode_range_from_points(
                        range_actionness_logits=range_actionness_logits,
                        range_offsets=range_offsets,
                        range_quality_logits=range_quality_logits,
                        combo_duration_prior_mean=(
                            self.range_duration_prior_means[combo_id.long()]
                            if (self.range_duration_prior_mode == "combo_gaussian" and torch.is_tensor(combo_id))
                            else None
                        ),
                        combo_duration_prior_std=(
                            self.range_duration_prior_stds[combo_id.long()].clamp_min(self.range_duration_prior_min_std)
                            if (self.range_duration_prior_mode == "combo_gaussian" and torch.is_tensor(combo_id))
                            else None
                        ),
                        combo_duration_prior_weight=self.range_duration_prior_weight,
                    )
                if self.range_boundary_refine_mode == "local_argmax":
                    range_decoded = self._refine_range_boundaries_local_argmax(
                        decoded=range_decoded,
                        range_start_logits=range_start_logits,
                        range_end_logits=range_end_logits,
                        radius=self.range_boundary_refine_radius,
                    )
                logits_start_bin = self._norm_to_bin_logits(
                    norm=range_decoded["pred_action_start_norm"],
                    num_bins=self.num_range_bins,
                )
                logits_end_bin = self._norm_to_bin_logits(
                    norm=range_decoded["pred_action_end_norm"],
                    num_bins=self.num_range_bins,
                )
                _toc("range_head_ms", t)
        _toc("model_total_ms", t_all)

        ref_tool_xy_px = self._xy_norm_to_canvas_px(ref_tool_xy)
        ref_ct_xy_px = self._xy_norm_to_canvas_px(ref_ct_xy)
        return {
            "pred_tool_xy": ref_tool_xy,
            "pred_ct_xy": ref_ct_xy,
            "pred_tool_xy_px": ref_tool_xy_px,
            "pred_ct_xy_px": ref_ct_xy_px,
            "combo_logits": combo_logits,
            "combo_tool_logits": combo_tool_logits,
            "combo_action_logits": combo_action_logits,
            "logits_start_bin": logits_start_bin,
            "logits_end_bin": logits_end_bin,
            "range_actionness_logits": range_actionness_logits,
            "range_offsets": range_offsets,
            "range_start_logits": range_start_logits,
            "range_end_logits": range_end_logits,
            "range_proposal_logits": (
                range_proposal_logits
                if torch.is_tensor(range_proposal_logits)
                else range_actionness_logits.new_zeros((bsz, t_out, t_out))
            ),
            "range_quality_logits": (
                range_quality_logits
                if torch.is_tensor(range_quality_logits)
                else range_actionness_logits.new_zeros((bsz, t_out))
            ),
            "pred_action_start_norm": range_decoded["pred_action_start_norm"],
            "pred_action_end_norm": range_decoded["pred_action_end_norm"],
            # Confidence is kept from the base spatial heads (refiner only adjusts xy).
            "conf_tool": tool_out["conf"],
            "conf_ct": ct_out["conf"],
            "probs_tool": tool_out["probs"],
            "probs_ct": ct_out["probs"],
            "heatmap_tool": tool_out["heatmap"],
            "heatmap_ct": ct_out["heatmap"],
            "logits_tool": tool_out["logits"],
            "logits_ct": ct_out["logits"],
            "traj_refine_temporal_tool_xy": ref_tool_temporal_xy,
            "traj_refine_temporal_ct_xy": ref_ct_temporal_xy,
            "traj_refine_delta_tool_xy": ref_tool_delta,
            "traj_refine_delta_ct_xy": ref_ct_delta,
            "traj_refine_delta_xy": torch.cat([ref_tool_delta, ref_ct_delta], dim=2),
            "traj_refine_delta_tool_gate": ref_tool_gate,
            "traj_refine_delta_ct_gate": ref_ct_gate,
            "meta": {
                "t_encoder": t_enc,
                "t_out": t_out,
                "t_prime": t_out,
                "h_prime": h_decode,
                "w_prime": w_decode,
                "h_encoder": h_prime,
                "w_encoder": w_prime,
                "temporal_upsample": self.temporal_upsample,
                "heatmap_softmax_temp": self.heatmap_softmax_temp,
                "enable_delta_branch": self.enable_delta_branch,
                "temporal_mean_residual_alpha": self.temporal_mean_residual_alpha,
                "refiner_mode": self.refiner_mode,
                "trajectory_only": self.trajectory_only,
                "range_only": self.range_only,
                "range_proposal_mode": self.range_proposal_mode,
                "range_proposal_decode_mode": self.range_proposal_decode_mode,
                "range_quality_mode": self.range_quality_mode,
                "range_boundary_refine_mode": self.range_boundary_refine_mode,
                "range_boundary_refine_radius": self.range_boundary_refine_radius,
                "range_temporal_context_mode": self.range_temporal_context_mode,
                "range_temporal_context_kernel_size": self.range_temporal_context_kernel_size,
                "range_temporal_context_layers": self.range_temporal_context_layers,
                "range_temporal_context_levels": self.range_temporal_context_levels,
                "range_adapter_mode": self.range_adapter_mode,
                "range_adapter_kernel_size": self.range_adapter_kernel_size,
                "range_adapter_num_heads": self.range_adapter_num_heads,
                "trajectory_target": self.trajectory_target,
                "tool_only": self.tool_only,
                "combo_probe_attn": self.combo_probe_attn,
                "combo_probe_dual_attn": self.combo_probe_dual_attn,
                "combo_feature_mode": self.combo_feature_mode,
                "combo_head_dropout": self.combo_head_dropout,
                "combo_feat_dropout": self.combo_feat_dropout,
                "dual_attn_tool_prior": bool(self.combo_action_tool_prior is not None),
                "dual_attn_tool_prior_detach": bool(self.dual_attn_tool_prior_detach),
                "dual_attn_tool_prior_alpha": (
                    0.0
                    if self.combo_action_tool_prior is None
                    else float(self.combo_action_tool_prior.alpha.detach().item())
                ),
                "num_tool_classes": self.num_tool_classes,
                "num_action_classes": self.num_action_classes,
                "traj_head_hidden_dim": self.traj_head_hidden_dim,
                "traj_head_up_layers": self.traj_head_up_layers,
                "traj_recurrent_max_delta_norm": self.traj_recurrent_max_delta_norm,
                "traj_tool_centroid_offset_head": self.traj_tool_centroid_offset_head,
                "traj_tool_offset_max_norm": self.traj_tool_offset_max_norm,
                "num_range_bins": self.num_range_bins,
                "output_canvas_size": self.output_canvas_size,
                "token_order": token_order,
                "n_tokens": n_tokens_meta,
                "n_tokens_encoder": n_tokens_encoder,
                "timing_ms": timing_ms,
                "shape_debug": shape_debug,
            },
        }

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        tokens_grid: Optional[torch.Tensor] = None,
        token_order: str = "cached_t_major_then_h_w",
        tool_id: Optional[torch.Tensor] = None,
        action_id: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if tokens_grid is not None:
            return self._forward_from_tokens_grid(
                tokens_grid=tokens_grid,
                token_order=token_order,
                tool_id=tool_id,
                action_id=action_id,
                combo_id=combo_id,
            )

        if video is None:
            raise ValueError("Either `video` or `tokens_grid` must be provided.")

        tokens_grid, (t_enc, h_prime, w_prime, d), _has_cls, token_order = self.vjepa(video)
        if tokens_grid.ndim == 5:
            _bsz, t_grid, h_grid, w_grid, d_grid = tokens_grid.shape
            if (t_grid, h_grid, w_grid, d_grid) != (t_enc, h_prime, w_prime, d):
                raise ValueError(
                    "Wrapper meta does not match tokens_grid shape: "
                    f"shape={(t_grid, h_grid, w_grid, d_grid)} vs meta={(t_enc, h_prime, w_prime, d)}"
                )
        elif tokens_grid.ndim == 6:
            _bsz, _num_layers, t_grid, h_grid, w_grid, d_grid = tokens_grid.shape
            if (t_grid, h_grid, w_grid, d_grid) != (t_enc, h_prime, w_prime, d):
                raise ValueError(
                    "Wrapper multi-layer meta does not match tokens_grid shape: "
                    f"shape={(t_grid, h_grid, w_grid, d_grid)} vs meta={(t_enc, h_prime, w_prime, d)}"
                )
        else:
            raise ValueError(
                "Expected tokens_grid as 5D [B,T,H,W,D] or 6D [B,K,T,H,W,D], "
                f"got shape={tuple(tokens_grid.shape)}"
            )
        return self._forward_from_tokens_grid(
            tokens_grid=tokens_grid,
            token_order=token_order,
            tool_id=tool_id,
            action_id=action_id,
            combo_id=combo_id,
        )
