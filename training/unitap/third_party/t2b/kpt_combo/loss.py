from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


LAYERWISE_STATECARRY_STAGE_LOSS_MODES: Tuple[str, ...] = (
    "m2_layerwise_statecarry_refine",
    "m3_layerwise_statecarry_dynamiccentroid",
    "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
    "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
    "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
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
)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted masked L1 for keypoint coordinates.

    pred/target: [B,T,P,2]
    weight: [B,T,P] (can be soft vis in [0,1])
    """
    diff = (pred - target).abs().sum(dim=-1)  # [B,T,P], ||.||_1 over xy
    numer = (diff * weight).sum()
    denom = weight.sum().clamp_min(eps)
    return numer / denom


def masked_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (pred - target).abs().sum(dim=-1)  # [B,T,P]
    numer = (diff * weight).sum(dim=(1, 2))
    denom = weight.sum(dim=(1, 2)).clamp_min(eps)
    return numer / denom


def masked_l2_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weighted masked L2 (euclidean) for keypoint coordinates."""
    diff = torch.sqrt(((pred - target).pow(2).sum(dim=-1)).clamp_min(eps))  # [B,T,P]
    numer = (diff * weight).sum()
    denom = weight.sum().clamp_min(eps)
    return numer / denom


def masked_l2_per_sample(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = torch.sqrt(((pred - target).pow(2).sum(dim=-1)).clamp_min(eps))  # [B,T,P]
    numer = (diff * weight).sum(dim=(1, 2))
    denom = weight.sum(dim=(1, 2)).clamp_min(eps)
    return numer / denom


def masked_l1l2_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Hybrid keypoint loss: 0.5 * masked L1 + 0.5 * masked L2."""
    l1 = masked_l1_loss(pred=pred, target=target, weight=weight, eps=eps)
    l2 = masked_l2_loss(pred=pred, target=target, weight=weight, eps=eps)
    return 0.5 * (l1 + l2)


def masked_l1l2_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    l1 = masked_l1_per_sample(pred=pred, target=target, weight=weight, eps=eps)
    l2 = masked_l2_per_sample(pred=pred, target=target, weight=weight, eps=eps)
    return 0.5 * (l1 + l2)


def masked_pixel_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pixel_scale: float = 256.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    # CoTracker-style: convert normalized xy residual to pixel residual, then mean over xy.
    ps = float(pixel_scale)
    diff = (pred - target).abs() * ps  # [B,T,P,2]
    diff = diff.mean(dim=-1)  # [B,T,P]
    numer = (diff * weight).sum()
    denom = weight.sum().clamp_min(eps)
    return numer / denom


def masked_pixel_l1_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pixel_scale: float = 256.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    ps = float(pixel_scale)
    diff = (pred - target).abs() * ps  # [B,T,P,2]
    diff = diff.mean(dim=-1)  # [B,T,P]
    numer = (diff * weight).sum(dim=(1, 2))
    denom = weight.sum(dim=(1, 2)).clamp_min(eps)
    return numer / denom


def masked_pixel_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pixel_scale: float = 256.0,
    huber_delta: float = 6.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    # Same Huber form as CoTracker loss (per-axis), averaged on xy.
    ps = float(pixel_scale)
    delta = float(huber_delta)
    diff = (pred - target) * ps  # [B,T,P,2]
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(dtype=diff.dtype)
    huber = flag * (0.5 * diff.pow(2)) + (1.0 - flag) * (delta * (abs_diff - 0.5 * delta))
    huber = huber.mean(dim=-1)  # [B,T,P]
    numer = (huber * weight).sum()
    denom = weight.sum().clamp_min(eps)
    return numer / denom


def masked_pixel_huber_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pixel_scale: float = 256.0,
    huber_delta: float = 6.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    ps = float(pixel_scale)
    delta = float(huber_delta)
    diff = (pred - target) * ps  # [B,T,P,2]
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(dtype=diff.dtype)
    huber = flag * (0.5 * diff.pow(2)) + (1.0 - flag) * (delta * (abs_diff - 0.5 * delta))
    huber = huber.mean(dim=-1)  # [B,T,P]
    numer = (huber * weight).sum(dim=(1, 2))
    denom = weight.sum(dim=(1, 2)).clamp_min(eps)
    return numer / denom


def _apply_t0_weight(
    weight: torch.Tensor,
    t0_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Apply an extra multiplier on frame-0 weights for trajectory loss."""
    w0 = float(t0_loss_weight)
    if w0 == 1.0:
        return weight
    if weight.ndim != 3 or int(weight.shape[1]) <= 0:
        return weight
    w = weight.clone()
    w[:, 0, :] = w[:, 0, :] * w0
    return w


def _trajectory_tool_factorized_targets(
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    traj_loss_visible_only: bool,
    traj_t0_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_tool_centroid_xy = gt_tool_xy.mean(dim=2, keepdim=True)
    gt_tool_offsets_xy = gt_tool_xy - gt_tool_centroid_xy
    if bool(traj_loss_visible_only):
        centroid_weight = gt_tool_vis.amax(dim=2, keepdim=True)
        offset_weight = gt_tool_vis
    else:
        centroid_weight = torch.ones_like(gt_tool_vis[:, :, :1])
        offset_weight = torch.ones_like(gt_tool_vis)
    centroid_weight = _apply_t0_weight(centroid_weight, t0_loss_weight=traj_t0_loss_weight)
    offset_weight = _apply_t0_weight(offset_weight, t0_loss_weight=traj_t0_loss_weight)
    return gt_tool_centroid_xy, gt_tool_offsets_xy, centroid_weight, offset_weight


def _resolve_kpt_loss_type(
    kpt_loss_type: str,
    traj_loss_pixel_scale: float = 256.0,
    traj_huber_delta: float = 6.0,
):
    lt = str(kpt_loss_type).strip().lower()
    if lt == "l1":
        return masked_l1_loss, masked_l1_per_sample
    if lt == "l2":
        return masked_l2_loss, masked_l2_per_sample
    if lt in ("l1l2", "l1_l2", "hybrid"):
        return masked_l1l2_loss, masked_l1l2_per_sample
    if lt in ("cotracker_l1", "pixel_l1"):
        return (
            lambda pred, target, weight, eps=1e-6: masked_pixel_l1_loss(
                pred=pred,
                target=target,
                weight=weight,
                pixel_scale=traj_loss_pixel_scale,
                eps=1e-6,
            ),
            lambda pred, target, weight, eps=1e-6: masked_pixel_l1_per_sample(
                pred=pred,
                target=target,
                weight=weight,
                pixel_scale=traj_loss_pixel_scale,
                eps=1e-6,
            ),
        )
    if lt in ("cotracker_huber", "pixel_huber", "huber"):
        return (
            lambda pred, target, weight, eps=1e-6: masked_pixel_huber_loss(
                pred=pred,
                target=target,
                weight=weight,
                pixel_scale=traj_loss_pixel_scale,
                huber_delta=traj_huber_delta,
                eps=1e-6,
            ),
            lambda pred, target, weight, eps=1e-6: masked_pixel_huber_per_sample(
                pred=pred,
                target=target,
                weight=weight,
                pixel_scale=traj_loss_pixel_scale,
                huber_delta=traj_huber_delta,
                eps=1e-6,
            ),
        )
    raise ValueError(
        "Unsupported kpt_loss_type="
        f"{kpt_loss_type}. Expected one of: l1, l2, l1l2, cotracker_l1, cotracker_huber."
    )


def _pointwise_kpt_loss_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    kpt_loss_type: str,
    traj_loss_pixel_scale: float = 256.0,
    traj_huber_delta: float = 6.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    lt = str(kpt_loss_type).strip().lower()
    if lt == "l1":
        return (pred - target).abs().sum(dim=-1)
    if lt == "l2":
        return torch.sqrt(((pred - target).pow(2).sum(dim=-1)).clamp_min(eps))
    if lt in ("l1l2", "l1_l2", "hybrid"):
        l1 = (pred - target).abs().sum(dim=-1)
        l2 = torch.sqrt(((pred - target).pow(2).sum(dim=-1)).clamp_min(eps))
        return 0.5 * (l1 + l2)
    if lt in ("cotracker_l1", "pixel_l1"):
        diff = (pred - target).abs() * float(traj_loss_pixel_scale)
        return diff.mean(dim=-1)
    if lt in ("cotracker_huber", "pixel_huber", "huber"):
        diff = (pred - target) * float(traj_loss_pixel_scale)
        abs_diff = diff.abs()
        delta = float(traj_huber_delta)
        flag = (abs_diff <= delta).to(dtype=diff.dtype)
        huber = flag * (0.5 * diff.pow(2)) + (1.0 - flag) * (delta * (abs_diff - 0.5 * delta))
        return huber.mean(dim=-1)
    raise ValueError(
        "Unsupported kpt_loss_type="
        f"{kpt_loss_type}. Expected one of: l1, l2, l1l2, cotracker_l1, cotracker_huber."
    )


def _pointwise_pixel_error_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pixel_scale: float = 256.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    diff = (pred - target) * float(pixel_scale)
    return torch.sqrt(diff.pow(2).sum(dim=-1).clamp_min(eps))


def _build_global_late_focal_weight_maps(
    pred_stage_stack: torch.Tensor,
    gt_xy: torch.Tensor,
    traj_weight: torch.Tensor,
    *,
    stage_indices: Sequence[int],
    traj_loss_pixel_scale: float,
    gamma: float,
    clip_min: float,
    clip_max: float,
    focal_state: Optional[Dict[str, float]] = None,
    ema_decay: float = 0.95,
    warmup_steps: int = 0,
    eps: float = 1e-6,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    device = pred_stage_stack.device
    dtype = pred_stage_stack.dtype
    zero = pred_stage_stack.new_zeros(())
    out_stats = {
        "traj_late_focal_ema_error": zero,
        "traj_late_focal_batch_error_mean": zero,
        "traj_late_focal_weight_mean": zero,
        "traj_late_focal_weight_max": zero,
        "traj_late_focal_weight_p50": zero,
        "traj_late_focal_weight_p75": zero,
        "traj_late_focal_weight_p90": zero,
        "traj_late_focal_weight_p95": zero,
    }
    if pred_stage_stack.ndim != 5:
        raise ValueError(
            f"`pred_stage_stack` must be [S,B,T,P,2], got {tuple(pred_stage_stack.shape)}"
        )
    num_stages = int(pred_stage_stack.shape[0])
    stage_ids = sorted({int(idx) for idx in stage_indices if 0 <= int(idx) < num_stages})
    if not stage_ids:
        return {}, out_stats
    valid_mask = traj_weight > 0
    if not bool(valid_mask.any().item()):
        return {}, out_stats

    error_maps: Dict[int, torch.Tensor] = {}
    valid_error_slices: List[torch.Tensor] = []
    for stage_idx in stage_ids:
        err_map = _pointwise_pixel_error_map(
            pred_stage_stack[stage_idx],
            gt_xy,
            pixel_scale=traj_loss_pixel_scale,
            eps=eps,
        )
        error_maps[stage_idx] = err_map
        valid_error_slices.append(err_map[valid_mask].detach())
    if not valid_error_slices:
        return {}, out_stats

    batch_error_mean = torch.cat(valid_error_slices, dim=0).mean().detach()
    prev_ema = None
    prev_updates = 0
    if isinstance(focal_state, dict):
        try:
            prev_ema = float(focal_state.get("ema_error", 0.0))
        except Exception:
            prev_ema = None
        try:
            prev_updates = int(focal_state.get("num_updates", 0))
        except Exception:
            prev_updates = 0
    if prev_ema is None or prev_ema <= 0.0:
        updated_ema = batch_error_mean
    else:
        ema_error = batch_error_mean.new_tensor(prev_ema)
        decay = min(max(float(ema_decay), 0.0), 1.0)
        updated_ema = decay * ema_error + (1.0 - decay) * batch_error_mean
    ema_error = updated_ema.detach()
    current_update = prev_updates + 1
    if isinstance(focal_state, dict):
        focal_state["ema_error"] = float(updated_ema.item())
        focal_state["num_updates"] = int(current_update)

    denom = ema_error.clamp_min(float(eps))
    warmup_limit = max(int(warmup_steps), 0)
    warmup_active = current_update <= warmup_limit
    weight_means: List[torch.Tensor] = []
    weight_maxes: List[torch.Tensor] = []
    weight_values: List[torch.Tensor] = []
    error_values: List[torch.Tensor] = []
    weight_maps: Dict[int, torch.Tensor] = {}
    for stage_idx in stage_ids:
        if warmup_active:
            focal_w = torch.ones_like(error_maps[stage_idx], dtype=traj_weight.dtype)
        else:
            ratio = (error_maps[stage_idx].detach() / denom).clamp_min(float(eps))
            focal_w = ratio.pow(float(gamma)).clamp(float(clip_min), float(clip_max)).to(dtype=traj_weight.dtype)
        weight_maps[stage_idx] = traj_weight * focal_w
        focal_valid = focal_w[valid_mask]
        if int(focal_valid.numel()) > 0:
            stage_mean = focal_valid.mean().to(device=device, dtype=dtype)
            weight_means.append(focal_valid.mean().to(device=device, dtype=dtype))
            weight_maxes.append(focal_valid.max().to(device=device, dtype=dtype))
            weight_values.append(focal_valid.to(device=device, dtype=dtype))
            error_values.append(error_maps[stage_idx][valid_mask].detach().to(device=device, dtype=dtype))
            out_stats[f"traj_late_focal_weight_stage{stage_idx}_mean"] = stage_mean

    out_stats["traj_late_focal_ema_error"] = ema_error.to(device=device, dtype=dtype)
    out_stats["traj_late_focal_batch_error_mean"] = batch_error_mean.to(device=device, dtype=dtype)
    if weight_means:
        weight_cat = torch.cat(weight_values, dim=0)
        out_stats["traj_late_focal_weight_mean"] = weight_cat.mean()
        out_stats["traj_late_focal_weight_max"] = weight_cat.max()
        quantiles = torch.quantile(
            weight_cat.float(),
            torch.tensor([0.50, 0.75, 0.90, 0.95], device=weight_cat.device, dtype=torch.float32),
        ).to(device=device, dtype=dtype)
        out_stats["traj_late_focal_weight_p50"] = quantiles[0]
        out_stats["traj_late_focal_weight_p75"] = quantiles[1]
        out_stats["traj_late_focal_weight_p90"] = quantiles[2]
        out_stats["traj_late_focal_weight_p95"] = quantiles[3]
    if weight_maxes:
        out_stats["traj_late_focal_weight_max"] = torch.stack(weight_maxes, dim=0).max()
    if error_values and weight_values:
        error_cat = torch.cat(error_values, dim=0)
        weight_cat = torch.cat(weight_values, dim=0)
        band_specs = (
            ("le4", None, 4.0),
            ("gt4_le8", 4.0, 8.0),
            ("gt8_le12", 8.0, 12.0),
            ("gt12_le24", 12.0, 24.0),
            ("gt24_le32", 24.0, 32.0),
            ("gt32", 32.0, None),
        )
        for band_name, lower, upper in band_specs:
            band_mask = torch.ones_like(error_cat, dtype=torch.bool)
            if lower is not None:
                band_mask = band_mask & (error_cat > float(lower))
            if upper is not None:
                band_mask = band_mask & (error_cat <= float(upper))
            if bool(band_mask.any().item()):
                out_stats[f"traj_late_focal_weight_band_{band_name}_mean"] = weight_cat[band_mask].mean()
    return weight_maps, out_stats


def _resolve_symmetric_pair_indices_from_role(pt_role_row: torch.Tensor) -> Optional[Tuple[List[int], int]]:
    roles = [int(x) for x in pt_role_row.detach().cpu().reshape(-1).tolist()]
    if len(roles) != 3:
        return None
    counts: Dict[int, int] = {}
    for role in roles:
        counts[role] = counts.get(role, 0) + 1
    pair_role = None
    for role, count in counts.items():
        if int(count) == 2:
            pair_role = int(role)
            break
    if pair_role is None:
        return None
    pair_idx = [i for i, role in enumerate(roles) if int(role) == int(pair_role)]
    uniq_idx = [i for i, role in enumerate(roles) if int(role) != int(pair_role)]
    if len(pair_idx) != 2 or len(uniq_idx) != 1:
        return None
    return pair_idx, int(uniq_idx[0])


def _swap_invariant_pair_loss_impl(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    tool_pt_role: Optional[torch.Tensor],
    kpt_loss_type: str,
    traj_loss_pixel_scale: float = 256.0,
    traj_huber_delta: float = 6.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fixed_map = _pointwise_kpt_loss_map(
        pred=pred,
        target=target,
        kpt_loss_type=kpt_loss_type,
        traj_loss_pixel_scale=traj_loss_pixel_scale,
        traj_huber_delta=traj_huber_delta,
        eps=eps,
    )
    bsz = int(pred.shape[0])
    device = pred.device
    dtype = pred.dtype
    per_sample = torch.zeros((bsz,), device=device, dtype=dtype)
    total_num = pred.new_zeros(())
    total_den = pred.new_zeros(())
    fallback_weight = weight.sum(dim=(1, 2))
    fallback_num = (fixed_map * weight).sum(dim=(1, 2))
    if tool_pt_role is None or pred.ndim != 4 or int(pred.shape[2]) != 3:
        per_sample = fallback_num / fallback_weight.clamp_min(eps)
        total_num = fallback_num.sum()
        total_den = fallback_weight.sum()
        return total_num / total_den.clamp_min(eps), per_sample

    if tool_pt_role.ndim == 1:
        tool_pt_role = tool_pt_role.unsqueeze(0)
    if int(tool_pt_role.shape[0]) != bsz:
        per_sample = fallback_num / fallback_weight.clamp_min(eps)
        total_num = fallback_num.sum()
        total_den = fallback_weight.sum()
        return total_num / total_den.clamp_min(eps), per_sample

    for bi in range(bsz):
        pair_spec = _resolve_symmetric_pair_indices_from_role(tool_pt_role[bi])
        if pair_spec is None:
            num_b = fallback_num[bi]
            den_b = fallback_weight[bi]
        else:
            pair_idx, uniq_idx = pair_spec
            w_pair = weight[bi : bi + 1, :, pair_idx]
            pair_fixed_num = (fixed_map[bi : bi + 1, :, pair_idx] * w_pair).sum()
            pair_swap_map = _pointwise_kpt_loss_map(
                pred=pred[bi : bi + 1, :, pair_idx, :],
                target=target[bi : bi + 1, :, [pair_idx[1], pair_idx[0]], :],
                kpt_loss_type=kpt_loss_type,
                traj_loss_pixel_scale=traj_loss_pixel_scale,
                traj_huber_delta=traj_huber_delta,
                eps=1e-6,
            )
            pair_swap_num = (pair_swap_map * w_pair).sum()
            pair_best_num = torch.minimum(pair_fixed_num, pair_swap_num)
            unique_num = (fixed_map[bi : bi + 1, :, uniq_idx : uniq_idx + 1] * weight[bi : bi + 1, :, uniq_idx : uniq_idx + 1]).sum()
            num_b = unique_num + pair_best_num
            den_b = weight[bi : bi + 1, :, pair_idx].sum() + weight[bi : bi + 1, :, uniq_idx : uniq_idx + 1].sum()
        per_sample[bi] = num_b / den_b.clamp_min(eps)
        total_num = total_num + num_b
        total_den = total_den + den_b
    return total_num / total_den.clamp_min(eps), per_sample


def _symmetric_pair_separation_aux_impl(
    pred: torch.Tensor,
    target: torch.Tensor,
    vis: torch.Tensor,
    *,
    tool_pt_role: Optional[torch.Tensor],
    pixel_scale: float = 256.0,
    min_pred_gap_px: float = 8.0,
    gt_min_gap_px: float = 16.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = int(pred.shape[0])
    device = pred.device
    dtype = pred.dtype
    per_sample = torch.zeros((bsz,), device=device, dtype=dtype)
    total_num = pred.new_zeros(())
    total_den = pred.new_zeros(())
    if tool_pt_role is None or pred.ndim != 4 or int(pred.shape[2]) != 3:
        return total_num, per_sample

    if tool_pt_role.ndim == 1:
        tool_pt_role = tool_pt_role.unsqueeze(0)
    if int(tool_pt_role.shape[0]) != bsz:
        return total_num, per_sample

    px_scale = float(pixel_scale)
    min_gap = float(min_pred_gap_px)
    gt_gap = float(gt_min_gap_px)
    for bi in range(bsz):
        pair_spec = _resolve_symmetric_pair_indices_from_role(tool_pt_role[bi])
        if pair_spec is None:
            continue
        pair_idx, _ = pair_spec
        pair_pred = pred[bi, :, pair_idx, :].float()  # [T,2,2]
        pair_gt = target[bi, :, pair_idx, :].float()
        pair_vis = torch.minimum(vis[bi, :, pair_idx[0]], vis[bi, :, pair_idx[1]]).float()  # [T]
        pred_gap = torch.norm((pair_pred[:, 0, :] - pair_pred[:, 1, :]) * px_scale, dim=-1)
        gt_gap_px = torch.norm((pair_gt[:, 0, :] - pair_gt[:, 1, :]) * px_scale, dim=-1)
        valid = pair_vis * (gt_gap_px > gt_gap).to(dtype=pair_vis.dtype)
        sep_penalty = F.relu(pred_gap.new_tensor(min_gap) - pred_gap)
        num_b = (sep_penalty * valid).sum().to(dtype=dtype)
        den_b = valid.sum().to(dtype=dtype)
        per_sample[bi] = num_b / den_b.clamp_min(eps)
        total_num = total_num + num_b
        total_den = total_den + den_b
    return total_num / total_den.clamp_min(eps), per_sample


def _get_pred_stage_stack(
    outputs: Dict[str, torch.Tensor],
    *,
    stack_key: str,
    stage_key_prefix: str,
) -> Optional[torch.Tensor]:
    pred_stage_stack = outputs.get(stack_key, None)
    if torch.is_tensor(pred_stage_stack):
        return pred_stage_stack
    stages: List[torch.Tensor] = []
    stage_idx = 0
    while True:
        stage_tensor = outputs.get(f"{stage_key_prefix}{stage_idx}", None)
        if not torch.is_tensor(stage_tensor):
            break
        stages.append(stage_tensor)
        stage_idx += 1
    if not stages:
        return None
    return torch.stack(stages, dim=0)


def _compute_stagewise_kpt_loss(
    pred_stage_stack: torch.Tensor,
    gt_xy: torch.Tensor,
    traj_weight: torch.Tensor,
    *,
    kpt_loss_fn,
    stage_weights: Sequence[float],
    per_stage_weight_maps: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    if pred_stage_stack.ndim != 5:
        raise ValueError(
            f"`pred_stage_stack` must be [S,B,T,P,2], got {tuple(pred_stage_stack.shape)}"
        )
    num_stages = int(pred_stage_stack.shape[0])
    stage_losses = [
        kpt_loss_fn(
            pred_stage_stack[stage_idx],
            gt_xy,
            (
                per_stage_weight_maps.get(stage_idx, traj_weight)
                if isinstance(per_stage_weight_maps, dict)
                else traj_weight
            ),
        )
        for stage_idx in range(num_stages)
    ]
    stage_loss_stack = torch.stack(stage_losses, dim=0)
    weight_vec = _resample_stage_weights(
        stage_weights,
        num_stages,
        device=pred_stage_stack.device,
        dtype=stage_loss_stack.dtype,
    )
    weight_vec = weight_vec / weight_vec.sum().clamp_min(1e-8)
    view_shape = (num_stages,) + ((1,) * (stage_loss_stack.ndim - 1))
    total = (stage_loss_stack * weight_vec.view(view_shape)).sum(dim=0)
    if num_stages > 1:
        refine_weight_vec = weight_vec[1:]
        refine_weight_vec = refine_weight_vec / refine_weight_vec.sum().clamp_min(1e-8)
        refine_view_shape = (num_stages - 1,) + ((1,) * (stage_loss_stack.ndim - 1))
        refine_total = (stage_loss_stack[1:] * refine_weight_vec.view(refine_view_shape)).sum(dim=0)
    else:
        refine_total = stage_losses[-1] * 0.0
    return {
        "stage_losses": stage_losses,
        "stage_weights": weight_vec,
        "total": total,
        "final": stage_losses[-1],
        "refine_total": refine_total,
    }


def _build_gaussian_target_probs(
    gt_xy: torch.Tensor,
    h_heatmap: int,
    w_heatmap: int,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build normalized Gaussian target distribution q on heatmap grid.

    gt_xy: [B,T,P,2] normalized in [0,1]
    returns: [B,T,P,S] where S=H*W and sum_S q = 1
    """
    if gt_xy.ndim != 4 or int(gt_xy.shape[-1]) != 2:
        raise ValueError(f"`gt_xy` must be [B,T,P,2], got {tuple(gt_xy.shape)}")
    h = int(h_heatmap)
    w = int(w_heatmap)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid heatmap size: H={h} W={w}")
    sigma_v = float(sigma)
    if sigma_v <= 0.0:
        raise ValueError(f"`sigma` must be > 0, got {sigma_v}")

    device = gt_xy.device
    dtype = gt_xy.dtype
    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")  # [H,W]
    xx = xx + 0.5
    yy = yy + 0.5

    x_h = gt_xy[..., 0] * float(w) - 0.5
    y_h = gt_xy[..., 1] * float(h) - 0.5

    dx2 = (xx.view(1, 1, 1, h, w) - x_h.unsqueeze(-1).unsqueeze(-1)).pow(2)
    dy2 = (yy.view(1, 1, 1, h, w) - y_h.unsqueeze(-1).unsqueeze(-1)).pow(2)
    q = torch.exp(-(dx2 + dy2) / (2.0 * sigma_v * sigma_v))
    q = q.reshape(*q.shape[:-2], h * w)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q


def _build_gaussian_cell_targets(gt_xy: torch.Tensor, h: int, w: int, sigma: float) -> torch.Tensor:
    """E005/E006 target: Gaussian over the token grid centred at the TRUE continuous position.
    Cell (i, j) has centre ((j + 0.5) / w, (i + 0.5) / h); the peak cell is floor(gt * (w, h)).
    (T2b's `_build_gaussian_target_probs` subtracts an extra 0.5 cell and is therefore shifted by half a cell.)
    gt_xy: [B,T,P,2] in [0,1] -> [B,T,P,S] normalised to sum 1."""
    if gt_xy.ndim != 4 or int(gt_xy.shape[-1]) != 2:
        raise ValueError(f"`gt_xy` must be [B,T,P,2], got {tuple(gt_xy.shape)}")
    device, dtype = gt_xy.device, gt_xy.dtype
    rows = torch.arange(int(h), device=device, dtype=dtype) + 0.5
    cols = torch.arange(int(w), device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    cx = gt_xy[..., 0] * float(w)
    cy = gt_xy[..., 1] * float(h)
    d2 = (xx.view(1, 1, 1, h, w) - cx[..., None, None]).pow(2) + (yy.view(1, 1, 1, h, w) - cy[..., None, None]).pow(2)
    q = torch.exp(-d2 / (2.0 * float(sigma) ** 2)).reshape(*gt_xy.shape[:-1], int(h) * int(w))
    return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)



def _build_region_cell_targets(gt_xy: torch.Tensor, vis: torch.Tensor, side: int, sigma: float = 1.0) -> torch.Tensor:
    """C003 region target: [B,T,S] in [0,1]. Cells inside the convex polygon of the visible GT points (points ordered by
    angle around their centroid) get 1; every cell also gets max(., exp(-d^2/2 sigma^2)) to the nearest visible point
    (Gaussian-dilated corners), so 1- and 2-point frames still define a small region. Frames without visible points -> 0."""
    B, T, P, _ = gt_xy.shape
    dev = gt_xy.device
    ys, xs = torch.meshgrid(torch.arange(side, device=dev), torch.arange(side, device=dev), indexing="ij")
    cell = torch.stack([xs, ys], -1).float().reshape(-1, 2) + 0.5  # [S,2] cell centres in cell units
    pts = gt_xy.float() * float(side)  # [B,T,P,2]
    v = (vis > 0.5).float()  # [B,T,P]
    d2 = ((cell.view(1, 1, 1, -1, 2) - pts.unsqueeze(3)) ** 2).sum(-1)  # [B,T,P,S]
    d2 = torch.where(v.unsqueeze(-1) > 0.5, d2, torch.full_like(d2, 1e9))
    soft = torch.exp(-d2.amin(dim=2) / (2.0 * float(sigma) ** 2))  # [B,T,S]
    nvis = v.sum(-1)  # [B,T]
    cen = (pts * v.unsqueeze(-1)).sum(2) / nvis.clamp_min(1.0).unsqueeze(-1)  # [B,T,2]
    ang = torch.atan2(pts[..., 1] - cen[..., 1:2], pts[..., 0] - cen[..., 0:1])  # [B,T,P]
    ang = torch.where(v > 0.5, ang, torch.full_like(ang, 10.0))  # invisible points sort last
    order = ang.argsort(dim=-1)  # [B,T,P]
    sp = torch.gather(pts, 2, order.unsqueeze(-1).expand(-1, -1, -1, 2))  # sorted visible points first
    inside = torch.ones(B, T, cell.shape[0], device=dev, dtype=torch.bool)
    for k in range(P):
        a = sp[:, :, k]  # [B,T,2]
        idx_next = torch.where(k + 1 < nvis, torch.full_like(nvis, float(k + 1)), torch.zeros_like(nvis)).long()  # wrap to first visible
        b = torch.gather(sp, 2, idx_next.view(B, T, 1, 1).expand(-1, -1, 1, 2)).squeeze(2)  # [B,T,2]
        e = b - a
        cr = e[..., 0:1] * (cell[None, None, :, 1] - a[..., 1:2]) - e[..., 1:2] * (cell[None, None, :, 0] - a[..., 0:1])  # [B,T,S]
        active = (k < nvis).view(B, T, 1)
        inside = inside & (~active | (cr >= 0))
    inside = inside & (nvis >= 3).view(B, T, 1)
    tgt = torch.maximum(soft, inside.float())
    return tgt * (nvis > 0).view(B, T, 1).float()



def _soft_quad_mask(pred_xy: torch.Tensor, side: int, tau: float = 0.5) -> torch.Tensor:
    """C004: differentiable soft rasterization of the polygon spanned by the P predicted points (ordered by angle
    around their mean; orientation normalised by the sign of the shoelace area). Returns [B,T,S] in (0,1):
    inside(cell) = prod_edges sigmoid(cross(edge, cell - a) * sign(area) / tau), tau in cell units."""
    B, T, P, _ = pred_xy.shape
    dev = pred_xy.device
    ys, xs = torch.meshgrid(torch.arange(side, device=dev), torch.arange(side, device=dev), indexing="ij")
    cell = torch.stack([xs, ys], -1).float().reshape(-1, 2) + 0.5  # [S,2]
    pts = pred_xy.float() * float(side)  # [B,T,P,2]
    cen = pts.mean(2, keepdim=True)
    ang = torch.atan2(pts[..., 1] - cen[..., 1], pts[..., 0] - cen[..., 0])  # [B,T,P]
    order = ang.argsort(dim=-1)
    sp = torch.gather(pts, 2, order.unsqueeze(-1).expand(-1, -1, -1, 2))  # [B,T,P,2] angular order
    nxt = torch.roll(sp, shifts=-1, dims=2)
    area2 = (sp[..., 0] * nxt[..., 1] - nxt[..., 0] * sp[..., 1]).sum(-1)  # [B,T] twice the signed area
    sgn = torch.where(area2 >= 0, torch.ones_like(area2), -torch.ones_like(area2)).view(B, T, 1, 1)
    e = nxt - sp  # [B,T,P,2]
    d = cell.view(1, 1, 1, -1, 2) - sp.unsqueeze(3)  # [B,T,P,S,2]
    cross = e[..., 0:1] * d[..., 1] - e[..., 1:2] * d[..., 0]  # [B,T,P,S]
    inside = torch.sigmoid(cross * sgn / float(tau)).prod(dim=2)  # [B,T,S]
    return inside


def masked_prob_ce_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked CE between target distribution q and predicted distribution p."""
    if pred_probs.ndim != 4:
        raise ValueError(f"`pred_probs` must be [B,T,P,S], got {tuple(pred_probs.shape)}")
    if target_probs.ndim != 4:
        raise ValueError(f"`target_probs` must be [B,T,P,S], got {tuple(target_probs.shape)}")
    if weight.ndim != 3:
        raise ValueError(f"`weight` must be [B,T,P], got {tuple(weight.shape)}")
    if tuple(pred_probs.shape) != tuple(target_probs.shape):
        raise ValueError(
            f"`pred_probs` and `target_probs` shape mismatch: {tuple(pred_probs.shape)} vs {tuple(target_probs.shape)}"
        )
    if tuple(pred_probs.shape[:3]) != tuple(weight.shape):
        raise ValueError(
            f"`weight` shape mismatch with probs: probs[:3]={tuple(pred_probs.shape[:3])} weight={tuple(weight.shape)}"
        )

    ce = -(target_probs * (pred_probs.clamp_min(eps).log())).sum(dim=-1)  # [B,T,P]
    numer = (ce * weight).sum()
    denom = weight.sum().clamp_min(eps)
    return numer / denom


def masked_prob_ce_per_sample(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_probs.ndim != 4 or target_probs.ndim != 4 or weight.ndim != 3:
        raise ValueError("Expected pred_probs/target_probs=[B,T,P,S], weight=[B,T,P]")
    ce = -(target_probs * (pred_probs.clamp_min(eps).log())).sum(dim=-1)  # [B,T,P]
    numer = (ce * weight).sum(dim=(1, 2))
    denom = weight.sum(dim=(1, 2)).clamp_min(eps)
    return numer / denom


def _temporal_diff(x: torch.Tensor, order: int) -> torch.Tensor:
    out = x
    for _ in range(int(order)):
        out = out[:, 1:] - out[:, :-1]
    return out


def _temporal_min_weight(weight: torch.Tensor, order: int) -> torch.Tensor:
    out = weight
    for _ in range(int(order)):
        out = torch.minimum(out[:, 1:], out[:, :-1])
    return out


def masked_temporal_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    order: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if int(pred.shape[1]) <= int(order):
        return pred.sum() * 0.0
    pred_d = _temporal_diff(pred, order=int(order))
    target_d = _temporal_diff(target, order=int(order))
    weight_d = _temporal_min_weight(weight, order=int(order))
    return masked_l1_loss(pred_d, target_d, weight_d, eps=eps)


def masked_temporal_l1_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    order: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if int(pred.shape[1]) <= int(order):
        return pred.new_zeros((int(pred.shape[0]),))
    pred_d = _temporal_diff(pred, order=int(order))
    target_d = _temporal_diff(target, order=int(order))
    weight_d = _temporal_min_weight(weight, order=int(order))
    return masked_l1_per_sample(pred_d, target_d, weight_d, eps=eps)


def _make_stage_weights(
    n_stages: int,
    start: float,
    end: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    n = int(max(n_stages, 1))
    if n == 1:
        return torch.full((1,), float(end), device=device, dtype=dtype)
    return torch.linspace(float(start), float(end), steps=n, device=device, dtype=dtype)


def _resample_stage_weights(
    base_weights: Sequence[float],
    n_stages: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = [float(w) for w in base_weights]
    if not base:
        raise ValueError("`base_weights` must be non-empty.")
    n = int(max(n_stages, 1))
    if n == len(base):
        return torch.tensor(base, device=device, dtype=dtype)
    if n == 1:
        return torch.tensor([base[-1]], device=device, dtype=dtype)

    base_t = torch.tensor(base, device=device, dtype=dtype)
    src_pos = torch.linspace(0.0, 1.0, steps=len(base), device=device, dtype=dtype)
    dst_pos = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
    idx_hi = torch.searchsorted(src_pos, dst_pos, right=False).clamp(max=len(base) - 1)
    idx_lo = (idx_hi - 1).clamp(min=0)
    pos_lo = src_pos[idx_lo]
    pos_hi = src_pos[idx_hi]
    w_lo = base_t[idx_lo]
    w_hi = base_t[idx_hi]
    denom = (pos_hi - pos_lo).clamp_min(1e-8)
    alpha = ((dst_pos - pos_lo) / denom).clamp(0.0, 1.0)
    alpha = torch.where(idx_hi == idx_lo, torch.zeros_like(alpha), alpha)
    return w_lo + alpha * (w_hi - w_lo)


def _pad_stage_weights_to_n(weight_vec: torch.Tensor, target_n: int) -> torch.Tensor:
    if weight_vec.ndim != 1:
        raise ValueError(f"`weight_vec` must be 1D, got {tuple(weight_vec.shape)}")
    tgt = int(max(target_n, 1))
    if int(weight_vec.shape[0]) >= tgt:
        return weight_vec[:tgt]
    pad = weight_vec.new_zeros((tgt - int(weight_vec.shape[0]),))
    return torch.cat([weight_vec, pad], dim=0)


def _pad_stage_weights_to5(weight_vec: torch.Tensor) -> torch.Tensor:
    return _pad_stage_weights_to_n(weight_vec, 5)


def _pad_stage_weights_to6(weight_vec: torch.Tensor) -> torch.Tensor:
    return _pad_stage_weights_to_n(weight_vec, 6)


def _weighted_stage_masked_l1(
    stages: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    stage_weight_start: float,
    stage_weight_end: float,
) -> torch.Tensor:
    """Stage-supervised masked L1 mean over refine iterations.

    stages: [K,B,T,P,2]
    returns scalar loss = sum_k w_k * L_k / sum_k w_k
    """
    if stages.ndim != 5 or int(stages.shape[-1]) != 2:
        raise ValueError(f"`stages` must be [K,B,T,P,2], got {tuple(stages.shape)}")
    k = int(stages.shape[0])
    weights = _make_stage_weights(
        n_stages=k,
        start=float(stage_weight_start),
        end=float(stage_weight_end),
        device=stages.device,
        dtype=stages.dtype,
    )
    total = stages.new_tensor(0.0)
    for i in range(k):
        l_i = masked_l1_loss(stages[i], target, weight)
        total = total + weights[i] * l_i
    return total / weights.sum().clamp_min(1e-6)


def _weighted_stage_masked_l1_per_sample(
    stages: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    stage_weight_start: float,
    stage_weight_end: float,
) -> torch.Tensor:
    """Per-sample stage-supervised masked L1 mean over refine iterations.

    stages: [K,B,T,P,2]
    returns [B]
    """
    if stages.ndim != 5 or int(stages.shape[-1]) != 2:
        raise ValueError(f"`stages` must be [K,B,T,P,2], got {tuple(stages.shape)}")
    k = int(stages.shape[0])
    weights = _make_stage_weights(
        n_stages=k,
        start=float(stage_weight_start),
        end=float(stage_weight_end),
        device=stages.device,
        dtype=stages.dtype,
    )
    total = stages.new_zeros((int(stages.shape[1]),))
    for i in range(k):
        l_i = masked_l1_per_sample(stages[i], target, weight)
        total = total + weights[i] * l_i
    return total / weights.sum().clamp_min(1e-6)


def _make_stage4_weights(
    w0: float,
    w1: float,
    w2: float,
    w3: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    w = torch.tensor(
        [float(w0), float(w1), float(w2), float(w3)],
        device=device,
        dtype=dtype,
    ).clamp_min(0.0)
    if float(w.sum().item()) <= 1e-8:
        w = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    return w / w.sum().clamp_min(1e-6)


def _weighted_stage4_scalar(
    l0: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    l3: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    return w[0] * l0 + w[1] * l1 + w[2] * l2 + w[3] * l3


def _weighted_stage4_per_sample(
    l0: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    l3: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    return w[0] * l0 + w[1] * l1 + w[2] * l2 + w[3] * l3


def combo_ce_loss(
    combo_logits: torch.Tensor,
    combo_id: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.cross_entropy(combo_logits, combo_id, weight=class_weights)


def combo_ce_per_sample(
    combo_logits: torch.Tensor,
    combo_id: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.cross_entropy(combo_logits, combo_id, reduction="none", weight=class_weights)


def _derive_tool_action_id_from_combo(combo_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map combo_id -> (tool_id, action_id).

    Combo mapping:
      0 hook|dissect
      1 grasper|grasp
      2 clipper|clip
      3 scissors|cut
      4 scissors|dissect
      5 grasper|dissect

    tool_id mapping:
      0 hook, 1 grasper, 2 clipper, 3 scissors
    action_id mapping:
      0 dissect, 1 grasp, 2 clip, 3 cut
    """
    if combo_id.ndim == 0:
        combo = combo_id.reshape(1).long()
    else:
        combo = combo_id.long()
    combo_to_tool = torch.tensor([0, 1, 2, 3, 3, 1], device=combo.device, dtype=torch.long)
    combo_to_action = torch.tensor([0, 1, 2, 3, 0, 0], device=combo.device, dtype=torch.long)
    if bool((combo < 0).any().item()) or bool((combo >= int(combo_to_tool.shape[0])).any().item()):
        raise ValueError(
            f"`combo_id` must be in [0,{int(combo_to_tool.shape[0]) - 1}], got min={int(combo.min().item())}, "
            f"max={int(combo.max().item())}"
        )
    tool_id = combo_to_tool[combo]
    action_id = combo_to_action[combo]
    return tool_id, action_id


def _build_neighbor_soft_targets(
    gt_bin: torch.Tensor,
    num_bins: int,
    neighbor_smoothing: float,
    label_smoothing: float,
) -> torch.Tensor:
    bsz = int(gt_bin.shape[0])
    gt = gt_bin.long()
    target = F.one_hot(gt, num_classes=int(num_bins)).float()

    ns = float(max(min(neighbor_smoothing, 1.0), 0.0))
    if ns > 0.0:
        left_mass = torch.full((bsz,), ns * 0.5, device=target.device, dtype=target.dtype)
        right_mass = torch.full((bsz,), ns * 0.5, device=target.device, dtype=target.dtype)
        left_ok = (gt > 0).to(target.dtype)
        right_ok = (gt < int(num_bins) - 1).to(target.dtype)
        left_mass = left_mass * left_ok
        right_mass = right_mass * right_ok
        center_mass = 1.0 - left_mass - right_mass

        target.zero_()
        target.scatter_(1, gt.unsqueeze(1), center_mass.unsqueeze(1))
        left_idx = (gt - 1).clamp(min=0)
        right_idx = (gt + 1).clamp(max=int(num_bins) - 1)
        target.scatter_add_(1, left_idx.unsqueeze(1), left_mass.unsqueeze(1))
        target.scatter_add_(1, right_idx.unsqueeze(1), right_mass.unsqueeze(1))

    ls = float(max(min(label_smoothing, 1.0), 0.0))
    if ls > 0.0:
        target = target * (1.0 - ls) + (ls / float(num_bins))
    return target


def _weighted_soft_ce_with_focal(
    logits: torch.Tensor,
    gt_bin: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
    neighbor_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    num_bins = int(logits.shape[-1])
    gt = gt_bin.long().reshape(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    target = _build_neighbor_soft_targets(
        gt_bin=gt,
        num_bins=num_bins,
        neighbor_smoothing=float(neighbor_smoothing),
        label_smoothing=float(label_smoothing),
    )
    per = -(target * log_probs).sum(dim=-1)
    if class_weights is not None:
        per = per * class_weights[gt]
    gamma = float(max(focal_gamma, 0.0))
    if gamma > 0.0:
        p_gt = torch.softmax(logits, dim=-1).gather(1, gt.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        per = per * torch.pow(1.0 - p_gt, gamma)
    return per


def _ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsctk = lkis + rkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc
    rho = 0.5 * (rp - lp - rg + lg)
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def _sigmoid_focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    mod = torch.pow(1.0 - pt, float(max(gamma, 0.0)))
    if alpha >= 0.0:
        alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
        return ce * mod * alpha_t
    return ce * mod


def _sigmoid_varifocal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    targets_f = targets.float().clamp(0.0, 1.0)
    pred_prob = torch.sigmoid(logits.float())
    ce = F.binary_cross_entropy_with_logits(logits.float(), targets_f, reduction="none")
    neg_weight = float(alpha) * torch.pow(pred_prob, float(max(gamma, 0.0)))
    weight = torch.where(targets_f > 0.0, targets_f, neg_weight)
    return ce * weight


def _build_single_event_point_targets(
    gt_start_norm: torch.Tensor,
    gt_end_norm: torch.Tensor,
    t_len: int,
    range_mask: Optional[torch.Tensor],
    positive_mode: str = "span",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = int(gt_start_norm.shape[0])
    dev = gt_start_norm.device
    dt = gt_start_norm.dtype
    mode = str(positive_mode).lower()
    if mode not in ("span", "center"):
        raise ValueError(f"Unsupported positive_mode={positive_mode}, expected one of: span,center")
    if t_len <= 0:
        cls_target = gt_start_norm.new_zeros((bsz, 0))
        offsets = gt_start_norm.new_zeros((bsz, 0, 2))
        if range_mask is None:
            valid = torch.ones((bsz,), device=dev, dtype=torch.bool)
        else:
            valid = range_mask.reshape(-1).float() > 0.5
        return cls_target, offsets, valid

    if range_mask is None:
        valid = torch.ones((bsz,), device=dev, dtype=torch.bool)
    else:
        valid = range_mask.reshape(-1).float() > 0.5

    den = float(max(int(t_len) - 1, 1))
    s = gt_start_norm.float().clamp(0.0, 1.0) * den
    e = gt_end_norm.float().clamp(0.0, 1.0) * den
    e = torch.maximum(e, s)

    points = torch.arange(int(t_len), device=dev, dtype=dt).view(1, -1).expand(bsz, -1)
    left = points - s.view(-1, 1).to(dt)
    right = e.view(-1, 1).to(dt) - points
    if mode == "span":
        pos = torch.logical_and(left >= 0.0, right >= 0.0)
    else:
        pos = torch.zeros((bsz, int(t_len)), device=dev, dtype=torch.bool)
        center = 0.5 * (s + e)
        center_idx = torch.round(center).long().clamp(min=0, max=int(t_len) - 1)
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        for ii in valid_idx.tolist():
            pos[ii, int(center_idx[ii].item())] = True

    # Ensure at least one positive point for valid samples.
    no_pos = torch.logical_and(valid, torch.logical_not(pos.any(dim=1)))
    if bool(no_pos.any().item()) and t_len > 0:
        mid = torch.round(0.5 * (s + e)).long().clamp(min=0, max=int(t_len) - 1)
        idxs = torch.nonzero(no_pos, as_tuple=False).reshape(-1)
        for ii in idxs.tolist():
            pos[ii, int(mid[ii].item())] = True

    cls_target = pos.to(dt)
    offsets = torch.stack((left.clamp_min(0.0), right.clamp_min(0.0)), dim=-1)
    return cls_target, offsets, valid


def _build_single_event_scale_invariant_weights(
    cls_target: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    if cls_target.ndim != 2:
        raise ValueError(f"`cls_target` must be [B,T], got {tuple(cls_target.shape)}")
    if valid.ndim != 1 or int(valid.shape[0]) != int(cls_target.shape[0]):
        raise ValueError(f"`valid` must be [B], got {tuple(valid.shape)} for cls_target={tuple(cls_target.shape)}")
    alpha_f = float(alpha)
    if alpha_f < 0.0 or alpha_f > 1.0:
        raise ValueError(f"`alpha` must be in [0,1], got {alpha}")
    valid_b = valid.reshape(-1).bool()
    valid_2d = valid_b.view(-1, 1)
    pos_mask = torch.logical_and(cls_target > 0.5, valid_2d)
    neg_mask = torch.logical_and(torch.logical_not(pos_mask), valid_2d)
    pos_count = pos_mask.sum(dim=1).float().clamp_min(1.0)
    neg_count = neg_mask.sum(dim=1).float().clamp_min(1.0)
    pos_w = ((1.0 - alpha_f) / pos_count).unsqueeze(1)
    neg_w = (alpha_f / neg_count).unsqueeze(1)
    weights = pos_mask.to(dtype=cls_target.dtype) * pos_w + neg_mask.to(dtype=cls_target.dtype) * neg_w
    row_sum = weights.sum(dim=1, keepdim=True).clamp_min(float(eps))
    weights = weights / row_sum
    return weights


def _build_single_event_point_iou_targets(
    range_offsets: torch.Tensor,
    gt_offsets: torch.Tensor,
    cls_target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if range_offsets.ndim != 3 or int(range_offsets.shape[-1]) != 2:
        raise ValueError(f"`range_offsets` must be [B,T,2], got {tuple(range_offsets.shape)}")
    if gt_offsets.shape != range_offsets.shape:
        raise ValueError(
            f"`gt_offsets` must match range_offsets shape {tuple(range_offsets.shape)}, got {tuple(gt_offsets.shape)}"
        )
    if cls_target.shape != range_offsets.shape[:2]:
        raise ValueError(
            f"`cls_target` must be {tuple(range_offsets.shape[:2])}, got {tuple(cls_target.shape)}"
        )
    if valid.ndim != 1 or int(valid.shape[0]) != int(range_offsets.shape[0]):
        raise ValueError(f"`valid` must be [B], got {tuple(valid.shape)}")

    pred = range_offsets.float()
    gt = gt_offsets.float()
    lp, rp = pred[..., 0], pred[..., 1]
    lg, rg = gt[..., 0], gt[..., 1]
    inter = torch.minimum(lp, lg) + torch.minimum(rp, rg)
    union = (lp + rp) + (lg + rg) - inter
    iou = inter / union.clamp_min(float(eps))
    iou = iou.clamp(0.0, 1.0)
    return iou * cls_target.float() * valid.float().view(-1, 1)


def _normalize_positive_point_weights(
    weights: torch.Tensor,
    pos_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pos_f = pos_mask.to(dtype=weights.dtype)
    pos_sum = (weights * pos_f).sum(dim=1, keepdim=True)
    pos_count = pos_f.sum(dim=1, keepdim=True)
    scale = torch.where(
        pos_count > 0.0,
        pos_count / pos_sum.clamp_min(float(eps)),
        torch.ones_like(pos_sum),
    )
    return weights * scale


def _build_single_event_sensitivity_weights(
    gt_start_norm: torch.Tensor,
    gt_end_norm: torch.Tensor,
    t_len: int,
    range_mask: Optional[torch.Tensor],
    pos_mask: torch.Tensor,
    sensitivity_mode: str = "none",
    cls_sigma: float = 0.5,
    loc_sigma: float = 0.35,
    min_weight: float = 0.25,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(sensitivity_mode).lower()
    if mode not in ("none", "asl"):
        raise ValueError(f"Unsupported sensitivity_mode={sensitivity_mode}, expected one of: none,asl")

    bsz = int(gt_start_norm.shape[0])
    dev = gt_start_norm.device
    dt = gt_start_norm.dtype
    ones = gt_start_norm.new_ones((bsz, int(t_len)))
    zeros = gt_start_norm.new_zeros((bsz, int(t_len)))
    if mode == "none" or t_len <= 0:
        return ones, ones

    if float(cls_sigma) <= 0.0:
        raise ValueError(f"`cls_sigma` must be > 0, got {cls_sigma}")
    if float(loc_sigma) <= 0.0:
        raise ValueError(f"`loc_sigma` must be > 0, got {loc_sigma}")
    if float(min_weight) < 0.0 or float(min_weight) > 1.0:
        raise ValueError(f"`min_weight` must be in [0,1], got {min_weight}")

    if range_mask is None:
        valid = torch.ones((bsz,), device=dev, dtype=torch.bool)
    else:
        valid = range_mask.reshape(-1).float() > 0.5

    den = float(max(int(t_len) - 1, 1))
    s = gt_start_norm.float().clamp(0.0, 1.0) * den
    e = gt_end_norm.float().clamp(0.0, 1.0) * den
    e = torch.maximum(e, s)
    center = 0.5 * (s + e)
    half_extent = (0.5 * (e - s)).clamp_min(1.0)
    points = torch.arange(int(t_len), device=dev, dtype=dt).view(1, -1).expand(bsz, -1)
    rel = (points - center.view(-1, 1).to(dt)) / half_extent.view(-1, 1).to(dt)

    cls_core = torch.exp(-0.5 * torch.square(rel / float(cls_sigma)))
    loc_left = torch.exp(-0.5 * torch.square((rel + 1.0) / float(loc_sigma)))
    loc_right = torch.exp(-0.5 * torch.square((rel - 1.0) / float(loc_sigma)))
    loc_core = torch.maximum(loc_left, loc_right)
    if float(min_weight) > 0.0:
        cls_core = float(min_weight) + (1.0 - float(min_weight)) * cls_core
        loc_core = float(min_weight) + (1.0 - float(min_weight)) * loc_core

    pos_valid = torch.logical_and(pos_mask.bool(), valid.view(-1, 1))
    cls_pos = _normalize_positive_point_weights(cls_core, pos_valid, eps=eps)
    loc_pos = _normalize_positive_point_weights(loc_core, pos_valid, eps=eps)
    cls_weight = torch.where(pos_valid, cls_pos, ones)
    loc_weight = torch.where(pos_valid, loc_pos, zeros)
    return cls_weight, loc_weight


def _build_single_event_boundary_soft_targets(
    gt_start_norm: torch.Tensor,
    gt_end_norm: torch.Tensor,
    t_len: int,
    range_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = int(gt_start_norm.shape[0])
    dev = gt_start_norm.device
    dt = gt_start_norm.dtype
    if t_len <= 0:
        z = gt_start_norm.new_zeros((bsz, 0))
        if range_mask is None:
            valid = torch.ones((bsz,), device=dev, dtype=torch.bool)
        else:
            valid = range_mask.reshape(-1).float() > 0.5
        return z, z.clone(), valid

    if range_mask is None:
        valid = torch.ones((bsz,), device=dev, dtype=torch.bool)
    else:
        valid = range_mask.reshape(-1).float() > 0.5

    start_target = gt_start_norm.new_zeros((bsz, int(t_len)))
    end_target = gt_start_norm.new_zeros((bsz, int(t_len)))
    if t_len == 1:
        start_target[:, 0] = valid.to(dt)
        end_target[:, 0] = valid.to(dt)
        return start_target, end_target, valid

    max_t = float(t_len - 1)
    s = gt_start_norm.reshape(-1).float().clamp(0.0, 1.0) * max_t
    e = gt_end_norm.reshape(-1).float().clamp(0.0, 1.0) * max_t
    e = torch.maximum(e, s)

    def _scatter_linear(pos: torch.Tensor, target: torch.Tensor) -> None:
        left_idx = torch.floor(pos).long().clamp(min=0, max=int(t_len) - 1)
        right_idx = torch.ceil(pos).long().clamp(min=0, max=int(t_len) - 1)
        frac = (pos - left_idx.to(pos.dtype)).clamp(0.0, 1.0)
        target.scatter_(1, left_idx.unsqueeze(1), (1.0 - frac).unsqueeze(1).to(target.dtype))
        target.scatter_add_(1, right_idx.unsqueeze(1), frac.unsqueeze(1).to(target.dtype))

    _scatter_linear(s, start_target)
    _scatter_linear(e, end_target)
    valid_f = valid.to(dt).view(-1, 1)
    start_target = start_target * valid_f
    end_target = end_target * valid_f
    return start_target, end_target, valid


def _build_single_event_proposal_iou_targets(
    gt_start_norm: torch.Tensor,
    gt_end_norm: torch.Tensor,
    t_len: int,
    range_mask: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = int(gt_start_norm.shape[0])
    dev = gt_start_norm.device
    dt = gt_start_norm.dtype
    if range_mask is None:
        valid = torch.ones((bsz,), device=dev, dtype=torch.bool)
    else:
        valid = range_mask.reshape(-1).float() > 0.5
    if t_len <= 0:
        z = gt_start_norm.new_zeros((bsz, 0, 0))
        m = torch.zeros((0, 0), device=dev, dtype=torch.bool)
        return z, m, valid

    if t_len > 1:
        coords = torch.linspace(0.0, 1.0, steps=int(t_len), device=dev, dtype=dt)
    else:
        coords = torch.zeros((1,), device=dev, dtype=dt)
    prop_start = coords.view(1, t_len, 1)
    prop_end = coords.view(1, 1, t_len)

    gt_s = gt_start_norm.reshape(-1).float().clamp(0.0, 1.0).view(-1, 1, 1)
    gt_e = gt_end_norm.reshape(-1).float().clamp(0.0, 1.0).view(-1, 1, 1)
    gt_e = torch.maximum(gt_e, gt_s)

    inter = (torch.minimum(prop_end, gt_e) - torch.maximum(prop_start, gt_s)).clamp_min(0.0)
    union = (torch.maximum(prop_end, gt_e) - torch.minimum(prop_start, gt_s)).clamp_min(float(eps))
    iou = inter / union
    valid_pairs = torch.triu(torch.ones((t_len, t_len), device=dev, dtype=torch.bool), diagonal=0)
    iou = iou * valid_pairs.unsqueeze(0).to(iou.dtype) * valid.view(-1, 1, 1).to(iou.dtype)
    return iou, valid_pairs, valid


def _soft_target_temporal_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    targets_f = targets.float()
    per = -(targets_f * log_probs).sum(dim=-1)
    gamma = float(max(focal_gamma, 0.0))
    if gamma > 0.0:
        p_t = (torch.softmax(logits, dim=-1) * targets_f).sum(dim=-1).clamp(1e-6, 1.0)
        per = per * torch.pow(1.0 - p_t, gamma)
    return per


def _decode_single_event_from_points(
    range_actionness_logits: torch.Tensor,
    range_offsets: torch.Tensor,
    range_quality_logits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return z, z

    score = torch.sigmoid(range_actionness_logits.float())
    if torch.is_tensor(range_quality_logits):
        if range_quality_logits.shape != range_actionness_logits.shape:
            raise ValueError(
                f"`range_quality_logits` must match actionness shape {tuple(range_actionness_logits.shape)}, "
                f"got {tuple(range_quality_logits.shape)}"
            )
        score = score * torch.sigmoid(range_quality_logits.float())
    idx = score.argmax(dim=1)  # [B]
    gather_idx = idx.view(-1, 1, 1).expand(-1, 1, 2)
    picked = range_offsets.gather(dim=1, index=gather_idx).squeeze(1)  # [B,2]
    left = picked[:, 0]
    right = picked[:, 1]
    point_t = idx.to(range_offsets.dtype)
    start_t = point_t - left
    end_t = point_t + right
    if t_len > 1:
        max_t = float(t_len - 1)
        start_t = start_t.clamp(0.0, max_t)
        end_t = end_t.clamp(0.0, max_t)
        end_t = torch.maximum(end_t, start_t)
        return start_t / max_t, end_t / max_t
    z = point_t.new_zeros(point_t.shape)
    return z, z


def _decode_single_event_from_proposals(
    range_start_logits: torch.Tensor,
    range_end_logits: torch.Tensor,
    range_proposal_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return z, z

    start_prob = torch.sigmoid(range_start_logits.float()).unsqueeze(2)
    end_prob = torch.sigmoid(range_end_logits.float()).unsqueeze(1)
    proposal_prob = torch.sigmoid(range_proposal_logits.float())
    score = proposal_prob * start_prob * end_prob
    valid_pairs = torch.triu(torch.ones((t_len, t_len), device=score.device, dtype=torch.bool), diagonal=0)
    score = score.masked_fill(~valid_pairs.unsqueeze(0), -1.0)
    flat_idx = score.view(bsz, -1).argmax(dim=1)
    start_idx = flat_idx // t_len
    end_idx = flat_idx % t_len
    if t_len > 1:
        den = float(t_len - 1)
        return start_idx.to(score.dtype) / den, end_idx.to(score.dtype) / den
    z = start_idx.new_zeros(start_idx.shape, dtype=score.dtype)
    return z, z


def _refine_single_event_boundaries_local_argmax(
    pred_start_norm: torch.Tensor,
    pred_end_norm: torch.Tensor,
    range_start_logits: torch.Tensor,
    range_end_logits: torch.Tensor,
    radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_start_norm.ndim != 1:
        raise ValueError(f"`pred_start_norm` must be [B], got {tuple(pred_start_norm.shape)}")
    if pred_end_norm.ndim != 1:
        raise ValueError(f"`pred_end_norm` must be [B], got {tuple(pred_end_norm.shape)}")
    if range_start_logits.ndim != 2:
        raise ValueError(f"`range_start_logits` must be [B,T], got {tuple(range_start_logits.shape)}")
    if range_end_logits.ndim != 2:
        raise ValueError(f"`range_end_logits` must be [B,T], got {tuple(range_end_logits.shape)}")
    if int(radius) <= 0:
        raise ValueError(f"`radius` must be > 0, got {radius}")
    bsz, t_len = range_start_logits.shape
    if range_end_logits.shape != (bsz, t_len):
        raise ValueError(
            f"`range_end_logits` must match start shape {(bsz, t_len)}, got {tuple(range_end_logits.shape)}"
        )
    if int(pred_start_norm.shape[0]) != bsz or int(pred_end_norm.shape[0]) != bsz:
        raise ValueError("Prediction batch size must match boundary logits batch size.")
    if t_len <= 1:
        z = pred_start_norm.new_zeros(pred_start_norm.shape)
        return z, z
    max_t = float(t_len - 1)
    start_t = (pred_start_norm.float().clamp(0.0, 1.0) * max_t)
    end_t = (pred_end_norm.float().clamp(0.0, 1.0) * max_t)
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
    return refined_start / max_t, refined_end / max_t


def range_bin_ce_losses(
    logits_start_bin: torch.Tensor,
    logits_end_bin: torch.Tensor,
    gt_start_bin: torch.Tensor,
    gt_end_bin: torch.Tensor,
    range_mask: Optional[torch.Tensor] = None,
    start_loss_weight: float = 0.1,
    end_loss_weight: float = 0.1,
    label_smoothing: float = 0.0,
    start_bin_class_weights: Optional[torch.Tensor] = None,
    end_bin_class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
    neighbor_smoothing: float = 0.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    start_per = _weighted_soft_ce_with_focal(
        logits=logits_start_bin,
        gt_bin=gt_start_bin.long(),
        class_weights=start_bin_class_weights,
        label_smoothing=float(label_smoothing),
        neighbor_smoothing=float(neighbor_smoothing),
        focal_gamma=float(focal_gamma),
    )
    end_per = _weighted_soft_ce_with_focal(
        logits=logits_end_bin,
        gt_bin=gt_end_bin.long(),
        class_weights=end_bin_class_weights,
        label_smoothing=float(label_smoothing),
        neighbor_smoothing=float(neighbor_smoothing),
        focal_gamma=float(focal_gamma),
    )
    per = float(start_loss_weight) * start_per + float(end_loss_weight) * end_per
    if range_mask is None:
        w = torch.ones_like(per)
    else:
        w = range_mask.reshape(-1).float().clamp_min(0.0)

    denom = w.sum().clamp_min(eps)
    loss_start = (start_per * w).sum() / denom
    loss_end = (end_per * w).sum() / denom
    loss_range = (per * w).sum() / denom
    return {
        "loss_start_bin_ce": loss_start,
        "loss_end_bin_ce": loss_end,
        "loss_range": loss_range,
        "loss_start_bin_per_sample": start_per * w,
        "loss_end_bin_per_sample": end_per * w,
        "loss_range_per_sample": per * w,
    }


def _parse_range_combo_reweight_targets(spec: str) -> Tuple[str, ...]:
    allowed = ("start", "end", "cls", "reg", "boundary", "proposal", "quality", "sola")
    raw = str(spec).strip().lower()
    if (not raw) or raw == "none":
        return tuple()
    parts: List[str] = []
    seen = set()
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(
                "`range_combo_reweight_targets` must contain only: "
                + ",".join(allowed)
                + f"; got `{name}`"
            )
        if name in seen:
            continue
        seen.add(name)
        parts.append(name)
    return tuple(parts)


def _build_batch_combo_reweight(
    combo_id: torch.Tensor,
    combo_weights: Optional[torch.Tensor],
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    if combo_weights is None:
        return None
    if combo_id.ndim != 1:
        raise ValueError(f"`combo_id` must be [B], got {tuple(combo_id.shape)}")
    if valid.shape != combo_id.shape:
        raise ValueError(f"`valid` must match combo_id shape {tuple(combo_id.shape)}, got {tuple(valid.shape)}")
    weights = combo_weights.to(device=combo_id.device, dtype=torch.float32)
    sample_w = weights[combo_id.long()].float()
    valid_f = valid.float().clamp_min(0.0)
    target_mass = valid_f.sum()
    if float(target_mass.item()) <= 0.0:
        return sample_w.new_zeros(sample_w.shape)
    sample_w = sample_w * valid_f
    denom = sample_w.sum().clamp_min(eps)
    sample_w = sample_w * (target_mass / denom)
    return sample_w


def _apply_range_combo_reweight(
    range_losses: Dict[str, torch.Tensor],
    combo_id: torch.Tensor,
    combo_weights: Optional[torch.Tensor],
    range_mask: Optional[torch.Tensor],
    targets: str,
    start_loss_weight: float,
    end_loss_weight: float,
    boundary_aux_weight: float,
    proposal_weight: float,
    quality_loss_weight: float,
    sola_aux_weight: float,
    cls_loss_weight: float,
    reg_loss_weight: float,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    target_names = _parse_range_combo_reweight_targets(targets)
    if (combo_weights is None) or (len(target_names) == 0):
        return range_losses
    if range_mask is None:
        valid = torch.ones_like(combo_id, dtype=torch.float32)
    else:
        valid = range_mask.reshape(-1).float()
    sample_w = _build_batch_combo_reweight(
        combo_id=combo_id.reshape(-1).long(),
        combo_weights=combo_weights,
        valid=valid,
        eps=eps,
    )
    if sample_w is None:
        return range_losses
    mask_den = valid.sum().clamp_min(eps)
    sample_w = sample_w.reshape(-1)
    out = dict(range_losses)

    def _maybe_weight(name: str, key: str) -> torch.Tensor:
        value = out[key]
        if name in target_names:
            return value * sample_w
        return value

    start_ps = _maybe_weight("start", "loss_start_bin_per_sample")
    end_ps = _maybe_weight("end", "loss_end_bin_per_sample")
    cls_ps = _maybe_weight("cls", "loss_range_cls_per_sample")
    reg_ps = _maybe_weight("reg", "loss_range_reg_per_sample")
    boundary_ps = _maybe_weight("boundary", "loss_range_boundary_aux_per_sample")
    proposal_ps = _maybe_weight("proposal", "loss_range_proposal_per_sample")
    quality_ps = _maybe_weight("quality", "loss_range_quality_per_sample")
    sola_ps = _maybe_weight("sola", "loss_range_sola_per_sample")
    range_ps = (
        float(cls_loss_weight) * cls_ps
        + float(reg_loss_weight) * reg_ps
        + float(start_loss_weight) * start_ps
        + float(end_loss_weight) * end_ps
        + float(boundary_aux_weight) * boundary_ps
        + float(proposal_weight) * proposal_ps
        + float(quality_loss_weight) * quality_ps
        + float(sola_aux_weight) * sola_ps
    )

    out["loss_start_bin_per_sample"] = start_ps
    out["loss_end_bin_per_sample"] = end_ps
    out["loss_range_cls_per_sample"] = cls_ps
    out["loss_range_reg_per_sample"] = reg_ps
    out["loss_range_boundary_aux_per_sample"] = boundary_ps
    out["loss_range_proposal_per_sample"] = proposal_ps
    out["loss_range_quality_per_sample"] = quality_ps
    out["loss_range_sola_per_sample"] = sola_ps
    out["loss_range_per_sample"] = range_ps
    out["loss_start_bin_ce"] = start_ps.sum() / mask_den
    out["loss_end_bin_ce"] = end_ps.sum() / mask_den
    out["loss_range_cls"] = cls_ps.sum() / mask_den
    out["loss_range_reg"] = reg_ps.sum() / mask_den
    out["loss_range_boundary_aux"] = boundary_ps.sum() / mask_den
    out["loss_range_proposal"] = proposal_ps.sum() / mask_den
    out["loss_range_quality"] = quality_ps.sum() / mask_den
    out["loss_range_sola"] = sola_ps.sum() / mask_den
    out["loss_range"] = range_ps.sum() / mask_den
    return out


def _sola_similarity_matching_losses(
    projected_features: torch.Tensor,
    valid: torch.Tensor,
    similarity_k: float = 16.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if projected_features.ndim != 3:
        raise ValueError(
            f"`projected_features` must be [B,T,D], got {tuple(projected_features.shape)}"
        )
    bsz, t_len, _ = projected_features.shape
    if valid.shape != (bsz,):
        raise ValueError(f"`valid` must be {(bsz,)}, got {tuple(valid.shape)}")
    if t_len <= 0:
        return projected_features.new_zeros((bsz,))
    proj = F.normalize(projected_features.float(), dim=-1, eps=eps)
    pred_sim = torch.matmul(proj, proj.transpose(1, 2)).clamp(-1.0, 1.0)
    pred_prob = ((pred_sim + 1.0) * 0.5).clamp(eps, 1.0 - eps)
    idx = torch.arange(t_len, device=projected_features.device, dtype=projected_features.dtype)
    dist = (idx[:, None] - idx[None, :]).abs()
    target = torch.ones((t_len, t_len), device=projected_features.device, dtype=projected_features.dtype)
    off_diag = dist > 0
    if bool(off_diag.any().item()):
        dist_sq = dist[off_diag].pow(2).clamp_min(1.0)
        target_values = torch.sigmoid(float(similarity_k) / dist_sq.float()).to(dtype=target.dtype)
        target[off_diag] = target_values
    target = target.unsqueeze(0).expand(bsz, -1, -1)
    pred_logits = torch.logit(pred_prob.float(), eps=eps)
    raw = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction="none")
    ps = raw.mean(dim=(1, 2)).to(dtype=projected_features.dtype)
    return ps * valid.float()


def range_offset_losses(
    range_actionness_logits: torch.Tensor,
    range_offsets: torch.Tensor,
    gt_start_norm: torch.Tensor,
    gt_end_norm: torch.Tensor,
    range_mask: Optional[torch.Tensor] = None,
    positive_mode: str = "span",
    range_start_logits: Optional[torch.Tensor] = None,
    range_end_logits: Optional[torch.Tensor] = None,
    range_proposal_logits: Optional[torch.Tensor] = None,
    range_quality_logits: Optional[torch.Tensor] = None,
    boundary_aux_weight: float = 0.0,
    proposal_weight: float = 0.0,
    cls_target_mode: str = "binary",
    cls_varifocal_alpha: float = 0.75,
    cls_varifocal_weight: float = 1.0,
    quality_mode: str = "none",
    quality_loss_weight: float = 0.0,
    range_sola_projected: Optional[torch.Tensor] = None,
    sola_aux_weight: float = 0.0,
    sola_similarity_k: float = 16.0,
    proposal_decode_mode: str = "argmax",
    boundary_refine_mode: str = "none",
    boundary_refine_radius: int = 2,
    sensitivity_mode: str = "none",
    sensitivity_cls_sigma: float = 0.5,
    sensitivity_loc_sigma: float = 0.35,
    sensitivity_min_weight: float = 0.25,
    scale_invariant_mode: str = "none",
    scale_invariant_alpha: float = 0.8,
    start_loss_weight: float = 0.1,
    end_loss_weight: float = 0.1,
    focal_gamma: float = 2.0,
    cls_loss_weight: float = 0.1,
    reg_loss_weight: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    bsz, t_len = int(range_actionness_logits.shape[0]), int(range_actionness_logits.shape[1])
    cls_target, gt_offsets, valid = _build_single_event_point_targets(
        gt_start_norm=gt_start_norm.reshape(-1).float(),
        gt_end_norm=gt_end_norm.reshape(-1).float(),
        t_len=t_len,
        range_mask=range_mask,
        positive_mode=positive_mode,
    )
    valid_f = valid.float()
    valid_2d = valid_f.view(-1, 1)
    mask_den = valid_f.sum().clamp_min(1.0)

    pos_mask = torch.logical_and(cls_target > 0.5, valid_2d > 0.5)
    cls_target_mode = str(cls_target_mode).lower()
    if cls_target_mode not in ("binary", "varifocal_iou", "hybrid_varifocal"):
        raise ValueError(
            f"`cls_target_mode` must be one of {{binary,varifocal_iou,hybrid_varifocal}}, got {cls_target_mode}"
        )
    if float(cls_varifocal_alpha) < 0.0:
        raise ValueError(f"`cls_varifocal_alpha` must be >= 0, got {cls_varifocal_alpha}")
    if float(cls_varifocal_weight) < 0.0 or float(cls_varifocal_weight) > 1.0:
        raise ValueError(f"`cls_varifocal_weight` must be in [0,1], got {cls_varifocal_weight}")
    binary_cls_raw = _sigmoid_focal_bce(
        logits=range_actionness_logits.float(),
        targets=cls_target.float(),
        gamma=float(max(focal_gamma, 0.0)),
        alpha=0.25,
    )
    if cls_target_mode in ("varifocal_iou", "hybrid_varifocal"):
        cls_loss_target = _build_single_event_point_iou_targets(
            range_offsets=range_offsets.detach(),
            gt_offsets=gt_offsets,
            cls_target=cls_target,
            valid=valid,
            eps=eps,
        )
        varifocal_cls_raw = _sigmoid_varifocal_bce(
            logits=range_actionness_logits.float(),
            targets=cls_loss_target.float(),
            gamma=float(max(focal_gamma, 0.0)),
            alpha=float(cls_varifocal_alpha),
        )
        if cls_target_mode == "varifocal_iou":
            cls_raw = varifocal_cls_raw
        else:
            mix = float(cls_varifocal_weight)
            cls_raw = (1.0 - mix) * binary_cls_raw + mix * varifocal_cls_raw
    else:
        cls_raw = binary_cls_raw
    cls_point_weight, reg_point_weight = _build_single_event_sensitivity_weights(
        gt_start_norm=gt_start_norm.reshape(-1).float(),
        gt_end_norm=gt_end_norm.reshape(-1).float(),
        t_len=t_len,
        range_mask=range_mask,
        pos_mask=pos_mask,
        sensitivity_mode=sensitivity_mode,
        cls_sigma=float(sensitivity_cls_sigma),
        loc_sigma=float(sensitivity_loc_sigma),
        min_weight=float(sensitivity_min_weight),
        eps=eps,
    )
    scale_invariant_mode = str(scale_invariant_mode).lower()
    if scale_invariant_mode not in ("none", "cls", "cls_quality"):
        raise ValueError(
            f"`scale_invariant_mode` must be one of {{none,cls,cls_quality}}, got {scale_invariant_mode}"
        )
    if float(scale_invariant_alpha) < 0.0 or float(scale_invariant_alpha) > 1.0:
        raise ValueError(f"`scale_invariant_alpha` must be in [0,1], got {scale_invariant_alpha}")
    scale_invariant_weight = cls_target.new_ones(cls_target.shape)
    if scale_invariant_mode != "none":
        scale_invariant_weight = _build_single_event_scale_invariant_weights(
            cls_target=cls_target,
            valid=valid,
            alpha=float(scale_invariant_alpha),
            eps=eps,
        )
    num_pos = float(pos_mask.sum().item())
    normalizer = max(num_pos, 1.0)
    cls_raw_weighted = cls_raw * valid_2d * cls_point_weight * scale_invariant_weight
    cls_loss = cls_raw_weighted.sum() / valid_f.sum().clamp_min(1.0)

    if bool(pos_mask.any().item()):
        pred_pos = range_offsets[pos_mask]
        gt_pos = gt_offsets[pos_mask]
        reg_raw = _ctr_diou_loss_1d(pred_pos, gt_pos, reduction="none")
        reg_w = reg_point_weight[pos_mask]
        reg_loss = (reg_raw * reg_w).sum() / float(normalizer)
    else:
        reg_loss = range_offsets.sum() * 0.0

    boundary_ps = range_actionness_logits.new_zeros((bsz,))
    boundary_loss = range_actionness_logits.sum() * 0.0
    proposal_ps = range_actionness_logits.new_zeros((bsz,))
    proposal_loss = range_actionness_logits.sum() * 0.0
    quality_ps = range_actionness_logits.new_zeros((bsz,))
    quality_loss = range_actionness_logits.sum() * 0.0
    sola_ps = range_actionness_logits.new_zeros((bsz,))
    sola_loss = range_actionness_logits.sum() * 0.0
    if (
        float(boundary_aux_weight) > 0.0
        and (range_start_logits is not None)
        and (range_end_logits is not None)
    ):
        if range_start_logits.shape != range_actionness_logits.shape:
            raise ValueError(
                f"`range_start_logits` must match actionness shape {tuple(range_actionness_logits.shape)}, "
                f"got {tuple(range_start_logits.shape)}"
            )
        if range_end_logits.shape != range_actionness_logits.shape:
            raise ValueError(
                f"`range_end_logits` must match actionness shape {tuple(range_actionness_logits.shape)}, "
                f"got {tuple(range_end_logits.shape)}"
            )
        start_target, end_target, boundary_valid = _build_single_event_boundary_soft_targets(
            gt_start_norm=gt_start_norm.reshape(-1).float(),
            gt_end_norm=gt_end_norm.reshape(-1).float(),
            t_len=t_len,
            range_mask=range_mask,
        )
        boundary_valid_f = boundary_valid.float()
        start_bd_ps = _soft_target_temporal_ce(
            logits=range_start_logits.float(),
            targets=start_target.float(),
            focal_gamma=float(max(focal_gamma, 0.0)),
        )
        end_bd_ps = _soft_target_temporal_ce(
            logits=range_end_logits.float(),
            targets=end_target.float(),
            focal_gamma=float(max(focal_gamma, 0.0)),
        )
        boundary_ps = 0.5 * (start_bd_ps + end_bd_ps) * boundary_valid_f
        boundary_loss = boundary_ps.sum() / boundary_valid_f.sum().clamp_min(1.0)

    if range_proposal_logits is not None:
        if range_proposal_logits.shape != (bsz, t_len, t_len):
            raise ValueError(
                f"`range_proposal_logits` must be {(bsz, t_len, t_len)}, got {tuple(range_proposal_logits.shape)}"
            )
        proposal_target, proposal_pair_mask, proposal_valid = _build_single_event_proposal_iou_targets(
            gt_start_norm=gt_start_norm.reshape(-1).float(),
            gt_end_norm=gt_end_norm.reshape(-1).float(),
            t_len=t_len,
            range_mask=range_mask,
        )
        proposal_valid_f = proposal_valid.float().view(-1, 1, 1)
        proposal_pair_f = proposal_pair_mask.unsqueeze(0).to(dtype=proposal_target.dtype)
        proposal_raw = F.binary_cross_entropy_with_logits(
            range_proposal_logits.float(),
            proposal_target.float(),
            reduction="none",
        )
        proposal_raw = proposal_raw * proposal_pair_f * proposal_valid_f
        proposal_den = (proposal_pair_f * proposal_valid_f).sum(dim=(1, 2)).clamp_min(1.0)
        proposal_ps = proposal_raw.sum(dim=(1, 2)) / proposal_den
        proposal_loss = proposal_ps.sum() / valid_f.sum().clamp_min(1.0)

    quality_mode = str(quality_mode).lower()
    if quality_mode not in ("none", "dgqp"):
        raise ValueError(f"`quality_mode` must be one of {{none,dgqp}}, got {quality_mode}")
    if float(quality_loss_weight) < 0.0:
        raise ValueError(f"`quality_loss_weight` must be >= 0, got {quality_loss_weight}")
    if quality_mode != "none":
        if range_quality_logits is None:
            raise ValueError("`range_quality_logits` is required when `quality_mode != none`.")
        if range_quality_logits.shape != range_actionness_logits.shape:
            raise ValueError(
                f"`range_quality_logits` must match actionness shape {tuple(range_actionness_logits.shape)}, "
                f"got {tuple(range_quality_logits.shape)}"
            )
        quality_target = _build_single_event_point_iou_targets(
            range_offsets=range_offsets.detach(),
            gt_offsets=gt_offsets,
            cls_target=cls_target,
            valid=valid,
            eps=eps,
        )
        quality_raw = _sigmoid_varifocal_bce(
            logits=range_quality_logits.float(),
            targets=quality_target.float(),
            gamma=2.0,
            alpha=0.75,
        ) * valid_2d
        if scale_invariant_mode == "cls_quality":
            quality_raw = quality_raw * scale_invariant_weight
        quality_loss = quality_raw.sum() / valid_f.sum().clamp_min(1.0)
        quality_den = quality_raw.new_ones((bsz,))
        if scale_invariant_mode == "cls_quality":
            quality_den = (scale_invariant_weight * valid_2d).sum(dim=1).clamp_min(1.0)
        else:
            quality_den = valid_2d.sum(dim=1).clamp_min(1.0)
        quality_ps = quality_raw.sum(dim=1) / quality_den

    if float(sola_aux_weight) < 0.0:
        raise ValueError(f"`sola_aux_weight` must be >= 0, got {sola_aux_weight}")
    if float(sola_similarity_k) <= 0.0:
        raise ValueError(f"`sola_similarity_k` must be > 0, got {sola_similarity_k}")
    if float(sola_aux_weight) > 0.0:
        if range_sola_projected is None:
            raise ValueError("`range_sola_projected` is required when `sola_aux_weight > 0`.")
        sola_ps = _sola_similarity_matching_losses(
            projected_features=range_sola_projected,
            valid=valid,
            similarity_k=float(sola_similarity_k),
            eps=eps,
        )
        sola_loss = sola_ps.sum() / mask_den.clamp_min(eps)

    proposal_decode_mode = str(proposal_decode_mode).lower()
    if proposal_decode_mode not in ("argmax", "none"):
        raise ValueError(
            f"`proposal_decode_mode` must be one of {{argmax,none}}, got {proposal_decode_mode}"
        )
    boundary_refine_mode = str(boundary_refine_mode).lower()
    if boundary_refine_mode not in ("none", "local_argmax"):
        raise ValueError(
            f"`boundary_refine_mode` must be one of {{none,local_argmax}}, got {boundary_refine_mode}"
        )
    if int(boundary_refine_radius) <= 0:
        raise ValueError(f"`boundary_refine_radius` must be > 0, got {boundary_refine_radius}")

    use_proposal_decode = (
        (range_proposal_logits is not None)
        and (range_start_logits is not None)
        and (range_end_logits is not None)
        and proposal_decode_mode == "argmax"
    )
    if use_proposal_decode:
        pred_start_norm, pred_end_norm = _decode_single_event_from_proposals(
            range_start_logits=range_start_logits,
            range_end_logits=range_end_logits,
            range_proposal_logits=range_proposal_logits,
        )
    else:
        pred_start_norm, pred_end_norm = _decode_single_event_from_points(
            range_actionness_logits=range_actionness_logits,
            range_offsets=range_offsets,
            range_quality_logits=range_quality_logits if quality_mode != "none" else None,
        )
    if (
        boundary_refine_mode == "local_argmax"
        and (range_start_logits is not None)
        and (range_end_logits is not None)
    ):
        pred_start_norm, pred_end_norm = _refine_single_event_boundaries_local_argmax(
            pred_start_norm=pred_start_norm,
            pred_end_norm=pred_end_norm,
            range_start_logits=range_start_logits,
            range_end_logits=range_end_logits,
            radius=int(boundary_refine_radius),
        )
    gt_s = gt_start_norm.reshape(-1).float().clamp(0.0, 1.0)
    gt_e = gt_end_norm.reshape(-1).float().clamp(0.0, 1.0)
    start_l1_ps = (pred_start_norm - gt_s).abs() * valid_f
    end_l1_ps = (pred_end_norm - gt_e).abs() * valid_f
    start_l1 = start_l1_ps.sum() / mask_den.clamp_min(eps)
    end_l1 = end_l1_ps.sum() / mask_den.clamp_min(eps)

    cls_ps = cls_raw_weighted.sum(dim=1) / (cls_point_weight * scale_invariant_weight * valid_2d).sum(dim=1).clamp_min(eps)
    cls_ps = cls_ps * valid_f
    reg_ps = range_offsets.new_zeros((bsz,))
    for i in range(bsz):
        if not bool(valid[i].item()):
            continue
        pos_i = cls_target[i] > 0.5
        if bool(pos_i.any().item()):
            reg_raw_i = _ctr_diou_loss_1d(
                input_offsets=range_offsets[i][pos_i],
                target_offsets=gt_offsets[i][pos_i],
                reduction="none",
            )
            reg_w_i = reg_point_weight[i][pos_i]
            reg_ps[i] = (reg_raw_i * reg_w_i).sum() / reg_w_i.sum().clamp_min(eps)
    range_ps = (
        float(cls_loss_weight) * cls_ps
        + float(reg_loss_weight) * reg_ps
        + float(start_loss_weight) * start_l1_ps
        + float(end_loss_weight) * end_l1_ps
        + float(boundary_aux_weight) * boundary_ps
        + float(proposal_weight) * proposal_ps
        + float(quality_loss_weight) * quality_ps
        + float(sola_aux_weight) * sola_ps
    )
    range_total = range_ps.sum() / mask_den.clamp_min(eps)

    return {
        "loss_start_bin_ce": start_l1,
        "loss_end_bin_ce": end_l1,
        "loss_range": range_total,
        "loss_start_bin_per_sample": start_l1_ps,
        "loss_end_bin_per_sample": end_l1_ps,
        "loss_range_per_sample": range_ps,
        "loss_range_cls": cls_loss,
        "loss_range_reg": reg_loss,
        "loss_range_boundary_aux": boundary_loss,
        "loss_range_proposal": proposal_loss,
        "loss_range_quality": quality_loss,
        "loss_range_sola": sola_loss,
        "loss_range_cls_per_sample": cls_ps,
        "loss_range_reg_per_sample": reg_ps,
        "loss_range_boundary_aux_per_sample": boundary_ps,
        "loss_range_proposal_per_sample": proposal_ps,
        "loss_range_quality_per_sample": quality_ps,
        "loss_range_sola_per_sample": sola_ps,
        "pred_action_start_norm": pred_start_norm,
        "pred_action_end_norm": pred_end_norm,
    }


def _downsample_temporal_xy_vis(xy: torch.Tensor, vis: torch.Tensor, t_prime: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample GT to T' by segment-wise aggregation.

    Rules:
      - xy': mean over frames in each segment where vis>0
      - vis': max over each segment
    """
    bsz, t, p, _ = xy.shape
    if t == t_prime:
        return xy, vis

    if t % t_prime != 0:
        raise ValueError(f"Temporal alignment requires integer ratio T/T': got T={t}, T'={t_prime}")

    r = t // t_prime
    xy_seg = xy.view(bsz, t_prime, r, p, 2)
    vis_seg = vis.view(bsz, t_prime, r, p)

    support = (vis_seg > 0).float()
    denom = support.sum(dim=2, keepdim=False).clamp_min(1.0)
    xy_ds = (xy_seg * support.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)
    vis_ds = vis_seg.max(dim=2).values
    return xy_ds, vis_ds


def align_targets_to_tprime(
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    gt_ct_xy: torch.Tensor,
    gt_ct_vis: torch.Tensor,
    t_prime: int,
    strict_time_match: bool = True,
) -> Dict[str, torch.Tensor]:
    t_gt = int(gt_tool_xy.shape[1])
    if strict_time_match and t_gt != int(t_prime):
        raise ValueError(
            f"Strict time-match is enabled, but model T' != GT T: T'={int(t_prime)}, T={t_gt}. "
            "This training setup requires 16-frame outputs supervised on 16-frame GT."
        )
    tool_xy_ds, tool_vis_ds = _downsample_temporal_xy_vis(gt_tool_xy, gt_tool_vis, t_prime)
    ct_xy_ds, ct_vis_ds = _downsample_temporal_xy_vis(gt_ct_xy, gt_ct_vis, t_prime)
    return {
        "gt_tool_xy": tool_xy_ds,
        "gt_tool_vis": tool_vis_ds,
        "gt_ct_xy": ct_xy_ds,
        "gt_ct_vis": ct_vis_ds,
    }


def compute_batch_losses(
    outputs: Dict[str, torch.Tensor],
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    gt_ct_xy: torch.Tensor,
    gt_ct_vis: torch.Tensor,
    combo_id: torch.Tensor,
    gt_action_start_bin: torch.Tensor,
    gt_action_end_bin: torch.Tensor,
    action_range_mask: Optional[torch.Tensor],
    gt_action_start_norm: Optional[torch.Tensor] = None,
    gt_action_end_norm: Optional[torch.Tensor] = None,
    combo_ce_weight: float = 0.1,
    combo_tool_ce_weight: float = 0.5,
    combo_action_ce_weight: float = 0.5,
    combo_info_bottleneck_weight: float = 0.0,
    start_bin_loss_weight: float = 0.1,
    end_bin_loss_weight: float = 0.1,
    range_label_smoothing: float = 0.0,
    range_positive_mode: str = "span",
    range_sensitivity_mode: str = "none",
    range_sensitivity_cls_sigma: float = 0.5,
    range_sensitivity_loc_sigma: float = 0.35,
    range_sensitivity_min_weight: float = 0.25,
    range_scale_invariant_mode: str = "none",
    range_scale_invariant_alpha: float = 0.8,
    range_cls_target_mode: str = "binary",
    range_cls_varifocal_alpha: float = 0.75,
    range_cls_varifocal_weight: float = 1.0,
    range_quality_mode: str = "none",
    range_quality_loss_weight: float = 0.0,
    range_sola_aux_weight: float = 0.0,
    range_sola_similarity_k: float = 16.0,
    range_combo_reweight_weights: Optional[torch.Tensor] = None,
    range_combo_reweight_targets: str = "",
    combo_class_weights: Optional[torch.Tensor] = None,
    tool_class_weights: Optional[torch.Tensor] = None,
    action_class_weights: Optional[torch.Tensor] = None,
    start_bin_class_weights: Optional[torch.Tensor] = None,
    end_bin_class_weights: Optional[torch.Tensor] = None,
    range_focal_gamma: float = 0.0,
    range_neighbor_smoothing: float = 0.0,
    range_boundary_aux_weight: float = 0.0,
    range_proposal_weight: float = 0.0,
    traj_vel_weight: float = 0.2,
    traj_jerk_weight: float = 0.05,
    traj_refine_stage_ratio: float = 1.0,
    traj_iter_aux_weight: float = 0.2,
    traj_stage_weight_start: float = 0.25,
    traj_stage_weight_end: float = 1.0,
    traj_stage0_weight: float = 0.35,
    traj_stage1_weight: float = 0.35,
    traj_stage2_weight: float = 0.20,
    traj_stage3_weight: float = 0.10,
    kpt_loss_type: str = "l1l2",
    traj_loss_pixel_scale: float = 256.0,
    traj_huber_delta: float = 6.0,
    traj_t0_loss_weight: float = 1.0,
    traj_loss_visible_only: bool = True,
    traj_centroid_aux_weight: float = 0.0,
    traj_offset_aux_weight: float = 0.0,
    trajectory_target: str = "both",
    trajectory_only: bool = False,
    range_only: bool = False,
    combo_only: bool = False,
    tool_only: bool = False,
    traj_heatmap_ce_weight: float = 0.5,
    traj_heatmap_sigma: float = 2.45,
    tool_pt_role: Optional[torch.Tensor] = None,
    symmetry_swap_invariant_loss: bool = False,
    symmetry_pair_separation_loss: bool = False,
    symmetry_pair_separation_weight: float = 0.0,
    symmetry_pair_separation_min_pred_gap_px: float = 8.0,
    symmetry_pair_separation_gt_min_gap_px: float = 16.0,
    perpoint_attn_ce_weight: float = 0.0,
    perpoint_ce_sigma_cells: float = 1.0,
    conf_bce_weight: float = 0.0,
    conf_err_thresh_px: float = 8.0,
    token_det_ce_weight: float = 0.0,
    token_det_sigma_cells: float = 1.0,
    region_bce_weight: float = 0.0,
    region_sigma_cells: float = 1.0,
    centroid_region_ce_weight: float = 0.0,
    region_dice_weight: float = 0.0,
    region_dice_side: int = 48,
    region_dice_tau: float = 0.5,
    region_mask_gt=None,
    pair_gap_loss_weight: float = 0.0,
    pair_gap_huber_delta_px: float = 4.0,
    traj_m2_stage_weights: Optional[Sequence[float]] = None,
    traj_late_focal_enable: bool = False,
    traj_late_focal_stage_indices: Optional[Sequence[int]] = None,
    traj_late_focal_gamma: float = 1.5,
    traj_late_focal_clip_min: float = 0.30,
    traj_late_focal_clip_max: float = 3.0,
    traj_late_focal_ema_decay: float = 0.95,
    traj_late_focal_warmup_steps: int = 0,
    traj_late_focal_state: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    meta = outputs.get("meta", None)
    proposal_decode_mode = (
        str(meta.get("range_proposal_decode_mode", "argmax")).lower()
        if isinstance(meta, dict)
        else "argmax"
    )
    quality_mode = (
        str(meta.get("range_quality_mode", range_quality_mode)).lower()
        if isinstance(meta, dict)
        else str(range_quality_mode).lower()
    )
    boundary_refine_mode = (
        str(meta.get("range_boundary_refine_mode", "none")).lower()
        if isinstance(meta, dict)
        else "none"
    )
    boundary_refine_radius = (
        int(meta.get("range_boundary_refine_radius", 2))
        if isinstance(meta, dict)
        else 2
    )
    boundary_refine_mode = (
        str(meta.get("range_boundary_refine_mode", "none")).lower()
        if isinstance(meta, dict)
        else "none"
    )
    boundary_refine_radius = (
        int(meta.get("range_boundary_refine_radius", 2))
        if isinstance(meta, dict)
        else 2
    )
    if tool_only:
        tool_id, _ = _derive_tool_action_id_from_combo(combo_id.long())
        tool_logits = outputs["combo_tool_logits"]
        l_tool_cls = combo_ce_loss(
            tool_logits,
            tool_id,
            class_weights=tool_class_weights,
        )
        tool_cls_ps = combo_ce_per_sample(
            tool_logits,
            tool_id,
            class_weights=tool_class_weights,
        )
        bsz = int(tool_logits.shape[0])
        zero = tool_logits.new_zeros(())
        zero_ps = tool_logits.new_zeros((bsz,))
        stage_w = _pad_stage_weights_to6(tool_logits.new_tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
        out = {
            "loss_total": float(combo_tool_ce_weight) * l_tool_cls,
            "loss_tool_xy": zero,
            "loss_ct_xy": zero,
            "loss_tool_centroid_aux": zero,
            "loss_tool_offset_aux": zero,
            "loss_tool_pair_sep_aux": zero,
            "loss_tool_attn_ce": zero,
            "loss_conf_bce": zero,
            "pp_hit1": zero,
            "pp_offtool2": zero,
            "loss_tool_det_ce": zero,
            "loss_tool_region_bce": zero,
            "loss_tool_centroid_region_ce": zero,
            "region_hit1": zero,
            "loss_tool_region_dice": zero,
            "region_soft_iou": zero,
            "det_hit1": zero,
            "det_offtool2": zero,
            "loss_tool_pair_gap": zero,
            "pair_gap_ratio": zero,
            "loss_tool_heatmap_ce": zero,
            "loss_ct_heatmap_ce": zero,
            "loss_tool_xy_stage0": zero,
            "loss_ct_xy_stage0": zero,
            "loss_tool_xy_stage1": zero,
            "loss_ct_xy_stage1": zero,
            "loss_tool_xy_stage2": zero,
            "loss_ct_xy_stage2": zero,
            "loss_tool_xy_stage3": zero,
            "loss_ct_xy_stage3": zero,
            "loss_tool_xy_stage4": zero,
            "loss_ct_xy_stage4": zero,
            "loss_tool_xy_stage5": zero,
            "loss_ct_xy_stage5": zero,
            "loss_tool_xy_base": zero,
            "loss_ct_xy_base": zero,
            "loss_tool_xy_refine": zero,
            "loss_ct_xy_refine": zero,
            "loss_tool_xy_refine_total": zero,
            "loss_ct_xy_refine_total": zero,
            "loss_tool_xy_refine_stages": zero,
            "loss_ct_xy_refine_stages": zero,
            "loss_traj_vel": zero,
            "loss_traj_jerk": zero,
            "loss_combo_ce": zero,
            "loss_combo_tool_ce": l_tool_cls,
            "loss_combo_action_ce": zero,
            "loss_combo_info": zero,
            "loss_start_bin_ce": zero,
            "loss_end_bin_ce": zero,
            "loss_range": zero,
            "loss_range_cls": zero,
            "loss_range_reg": zero,
            "loss_range_boundary_aux": zero,
            "loss_range_proposal": zero,
            "loss_range_quality": zero,
            "loss_range_sola": zero,
            "combo_info_bottleneck_weight": zero,
            "traj_refine_stage_ratio": tool_logits.new_tensor(float(traj_refine_stage_ratio)),
            "traj_iter_aux_weight": tool_logits.new_tensor(float(traj_iter_aux_weight)),
            "traj_stage_weight_start": tool_logits.new_tensor(float(traj_stage_weight_start)),
            "traj_stage_weight_end": tool_logits.new_tensor(float(traj_stage_weight_end)),
            "traj_stage0_weight": stage_w[0],
            "traj_stage1_weight": stage_w[1],
            "traj_stage2_weight": stage_w[2],
            "traj_stage3_weight": stage_w[3],
            "traj_stage4_weight": stage_w[4],
            "traj_stage5_weight": stage_w[5],
            "loss_start_bin_per_sample": zero_ps,
            "loss_end_bin_per_sample": zero_ps,
            "loss_range_per_sample": zero_ps,
            "loss_combo_tool_ce_per_sample": tool_cls_ps,
            "loss_combo_action_ce_per_sample": zero_ps,
        }
        if "pred_action_start_norm" in outputs:
            out["pred_action_start_norm"] = outputs["pred_action_start_norm"]
        if "pred_action_end_norm" in outputs:
            out["pred_action_end_norm"] = outputs["pred_action_end_norm"]
        return out

    # Requested objective split:
    #   1) trajectory part -> masked keypoint loss (L1 or L2)
    #   2) combo part -> plain CE on combo_id
    #   3) action part -> keep original logic (offset regression + CE branch behavior)
    kpt_loss_fn, _ = _resolve_kpt_loss_type(
        kpt_loss_type,
        traj_loss_pixel_scale=traj_loss_pixel_scale,
        traj_huber_delta=traj_huber_delta,
    )
    meta = outputs.get("meta", {})
    traj_refiner_mode = ""
    if isinstance(meta, dict):
        traj_refiner_mode = str(meta.get("traj_refiner_mode", "")).lower()
    pred_tool_xy = outputs.get("pred_tool_xy_refine", outputs["pred_tool_xy"])
    pred_ct_xy = outputs.get("pred_ct_xy_refine", outputs["pred_ct_xy"])
    pred_tool_xy_stage0 = outputs.get("pred_tool_xy_stage0", None)
    pred_ct_xy_stage0 = outputs.get("pred_ct_xy_stage0", None)
    pred_tool_xy_stage1 = outputs.get("pred_tool_xy_stage1", None)
    pred_ct_xy_stage1 = outputs.get("pred_ct_xy_stage1", None)
    pred_tool_xy_stage2 = outputs.get("pred_tool_xy_stage2", None)
    pred_ct_xy_stage2 = outputs.get("pred_ct_xy_stage2", None)
    pred_tool_xy_stage_stack = _get_pred_stage_stack(
        outputs,
        stack_key="pred_tool_xy_stage_stack",
        stage_key_prefix="pred_tool_xy_stage",
    )
    pred_ct_xy_stage_stack = _get_pred_stage_stack(
        outputs,
        stack_key="pred_ct_xy_stage_stack",
        stage_key_prefix="pred_ct_xy_stage",
    )
    traj_target = str(trajectory_target).lower()
    if traj_target not in ("both", "tool", "ct"):
        raise ValueError(f"Unsupported trajectory_target={trajectory_target}. Expected one of: both, tool, ct.")
    traj_tool_weight = gt_tool_vis if bool(traj_loss_visible_only) else torch.ones_like(gt_tool_vis)
    traj_ct_weight = gt_ct_vis if bool(traj_loss_visible_only) else torch.ones_like(gt_ct_vis)
    traj_tool_weight = _apply_t0_weight(traj_tool_weight, t0_loss_weight=traj_t0_loss_weight)
    traj_ct_weight = _apply_t0_weight(traj_ct_weight, t0_loss_weight=traj_t0_loss_weight)
    def _tool_kpt_loss(
        pred_xy: torch.Tensor,
        gt_xy_arg: Optional[torch.Tensor] = None,
        traj_weight_arg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gt_xy_use = gt_tool_xy if gt_xy_arg is None else gt_xy_arg
        traj_weight_use = traj_tool_weight if traj_weight_arg is None else traj_weight_arg
        if bool(symmetry_swap_invariant_loss):
            return _swap_invariant_pair_loss_impl(
                pred=pred_xy,
                target=gt_xy_use,
                weight=traj_weight_use,
                tool_pt_role=tool_pt_role,
                kpt_loss_type=kpt_loss_type,
                traj_loss_pixel_scale=traj_loss_pixel_scale,
                traj_huber_delta=traj_huber_delta,
            )[0]
        return kpt_loss_fn(pred_xy, gt_xy_use, traj_weight_use)

    l_tool_final = _tool_kpt_loss(pred_tool_xy)
    l_ct_final = kpt_loss_fn(pred_ct_xy, gt_ct_xy, traj_ct_weight)
    zero_tool = l_tool_final * 0.0
    zero_ct = l_ct_final * 0.0
    use_s1_stage_loss = bool(
        trajectory_only
        and (traj_refiner_mode == "s1_offset")
        and torch.is_tensor(pred_tool_xy_stage0)
    )
    use_s2_stage_loss = bool(
        trajectory_only
        and (
            traj_refiner_mode in (
                "s2_iter_offset",
                "s3_support3x3",
                "s3p2x_finalrefine_l18",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
                "m1_strict_s3_multilayer_mean",
                "m1_progressive_multilayer_movingrefine",
            )
        )
        and torch.is_tensor(pred_tool_xy_stage0)
        and torch.is_tensor(pred_tool_xy_stage1)
        and torch.is_tensor(pred_tool_xy_stage2)
    )
    use_m2_stage_loss = bool(
        trajectory_only
        and (traj_refiner_mode in LAYERWISE_STATECARRY_STAGE_LOSS_MODES)
        and torch.is_tensor(pred_tool_xy_stage_stack)
        and (int(pred_tool_xy_stage_stack.shape[0]) >= 2)
    )
    late_focal_stats: Dict[str, torch.Tensor] = {}
    if use_m2_stage_loss:
        m2_stage_weights = (
            list(traj_m2_stage_weights)
            if traj_m2_stage_weights is not None
            else [0.35, 0.45, 0.60, 0.80, 1.00]
        )
        tool_stage_weight_maps: Optional[Dict[int, torch.Tensor]] = None
        if bool(traj_late_focal_enable) and (traj_target in ("both", "tool")):
            tool_stage_weight_maps, late_focal_stats = _build_global_late_focal_weight_maps(
                pred_tool_xy_stage_stack,
                gt_tool_xy,
                traj_tool_weight,
                stage_indices=(
                    list(traj_late_focal_stage_indices)
                    if traj_late_focal_stage_indices is not None
                    else [3, 4]
                ),
                traj_loss_pixel_scale=traj_loss_pixel_scale,
                gamma=traj_late_focal_gamma,
                clip_min=traj_late_focal_clip_min,
                clip_max=traj_late_focal_clip_max,
                focal_state=traj_late_focal_state,
                ema_decay=traj_late_focal_ema_decay,
                warmup_steps=traj_late_focal_warmup_steps,
            )
        tool_stage_loss_pack = _compute_stagewise_kpt_loss(
            pred_tool_xy_stage_stack,
            gt_tool_xy,
            traj_tool_weight,
            kpt_loss_fn=_tool_kpt_loss,
            stage_weights=m2_stage_weights,
            per_stage_weight_maps=tool_stage_weight_maps,
        )
        tool_stage_losses = list(tool_stage_loss_pack["stage_losses"])
        while len(tool_stage_losses) < 6:
            tool_stage_losses.append(zero_tool)
        l_tool_stage0 = tool_stage_losses[0]
        l_tool_stage1 = tool_stage_losses[1]
        l_tool_stage2 = tool_stage_losses[2]
        l_tool_stage3 = tool_stage_losses[3]
        l_tool_stage4 = tool_stage_losses[4]
        l_tool_stage5 = tool_stage_losses[5]
        l_tool = tool_stage_loss_pack["total"]
        l_tool_final = tool_stage_loss_pack["final"]
        l_tool_ref_total = tool_stage_loss_pack["refine_total"]
        l_tool_ref_stages = l_tool
        stage_w = _pad_stage_weights_to6(tool_stage_loss_pack["stage_weights"])
        if torch.is_tensor(pred_ct_xy_stage_stack) and int(pred_ct_xy_stage_stack.shape[0]) >= 2:
            ct_stage_loss_pack = _compute_stagewise_kpt_loss(
                pred_ct_xy_stage_stack,
                gt_ct_xy,
                traj_ct_weight,
                kpt_loss_fn=kpt_loss_fn,
                stage_weights=m2_stage_weights,
            )
            ct_stage_losses = list(ct_stage_loss_pack["stage_losses"])
            while len(ct_stage_losses) < 6:
                ct_stage_losses.append(zero_ct)
            l_ct_stage0 = ct_stage_losses[0]
            l_ct_stage1 = ct_stage_losses[1]
            l_ct_stage2 = ct_stage_losses[2]
            l_ct_stage3 = ct_stage_losses[3]
            l_ct_stage4 = ct_stage_losses[4]
            l_ct_stage5 = ct_stage_losses[5]
            l_ct = ct_stage_loss_pack["total"]
            l_ct_final = ct_stage_loss_pack["final"]
            l_ct_ref_total = ct_stage_loss_pack["refine_total"]
            l_ct_ref_stages = l_ct
        else:
            l_ct_stage0 = zero_ct
            l_ct_stage1 = zero_ct
            l_ct_stage2 = zero_ct
            l_ct_stage3 = zero_ct
            l_ct_stage4 = zero_ct
            l_ct_stage5 = zero_ct
            l_ct = zero_ct
            l_ct_final = zero_ct
            l_ct_ref_total = zero_ct
            l_ct_ref_stages = zero_ct
    elif use_s2_stage_loss:
        s2_stage1_weight = 0.7
        l_tool_stage0 = _tool_kpt_loss(pred_tool_xy_stage0)
        l_tool_stage1 = _tool_kpt_loss(pred_tool_xy_stage1)
        l_tool_stage2 = _tool_kpt_loss(pred_tool_xy_stage2)
        l_tool_stage3 = zero_tool
        l_tool_stage4 = zero_tool
        l_tool_stage5 = zero_tool
        if torch.is_tensor(pred_ct_xy_stage0):
            l_ct_stage0 = kpt_loss_fn(pred_ct_xy_stage0, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage0 = zero_ct
        if torch.is_tensor(pred_ct_xy_stage1):
            l_ct_stage1 = kpt_loss_fn(pred_ct_xy_stage1, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage1 = zero_ct
        if torch.is_tensor(pred_ct_xy_stage2):
            l_ct_stage2 = kpt_loss_fn(pred_ct_xy_stage2, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage2 = zero_ct
        l_ct_stage3 = zero_ct
        l_ct_stage4 = zero_ct
        l_ct_stage5 = zero_ct
        l_tool = l_tool_stage0 + s2_stage1_weight * l_tool_stage1 + l_tool_stage2
        l_ct = l_ct_stage0 + s2_stage1_weight * l_ct_stage1 + l_ct_stage2
        l_tool_final = l_tool_stage2
        l_ct_final = l_ct_stage2
        l_tool_ref_total = 0.7 * l_tool_stage1 + l_tool_stage2
        l_ct_ref_total = 0.7 * l_ct_stage1 + l_ct_stage2
        l_tool_ref_stages = l_tool
        l_ct_ref_stages = l_ct
        stage_w = _pad_stage_weights_to6(l_tool.new_tensor([1.0, 0.7, 1.0, 0.0, 0.0]))
    elif use_s1_stage_loss:
        l_tool_stage0 = _tool_kpt_loss(pred_tool_xy_stage0)
        if torch.is_tensor(pred_ct_xy_stage0):
            l_ct_stage0 = kpt_loss_fn(pred_ct_xy_stage0, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage0 = zero_ct
        l_tool_stage1 = l_tool_final
        l_ct_stage1 = l_ct_final
        l_tool_stage2 = zero_tool
        l_ct_stage2 = zero_ct
        l_tool_stage3 = zero_tool
        l_ct_stage3 = zero_ct
        l_tool_stage4 = zero_tool
        l_ct_stage4 = zero_ct
        l_tool_stage5 = zero_tool
        l_ct_stage5 = zero_ct
        l_tool = l_tool_stage0 + l_tool_final
        l_ct = l_ct_stage0 + l_ct_final
        l_tool_ref_total = l_tool_final
        l_ct_ref_total = l_ct_final
        l_tool_ref_stages = l_tool
        l_ct_ref_stages = l_ct
        stage_w = _pad_stage_weights_to6(l_tool.new_tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
    else:
        l_tool_stage0 = zero_tool
        l_ct_stage0 = zero_ct
        l_tool_stage1 = zero_tool
        l_ct_stage1 = zero_ct
        l_tool_stage2 = zero_tool
        l_ct_stage2 = zero_ct
        l_tool_stage3 = zero_tool
        l_ct_stage3 = zero_ct
        l_tool_stage4 = zero_tool
        l_ct_stage4 = zero_ct
        l_tool_stage5 = zero_tool
        l_ct_stage5 = zero_ct
        l_tool = l_tool_final
        l_ct = l_ct_final
        l_tool_ref_total = l_tool
        l_ct_ref_total = l_ct
        l_tool_ref_stages = l_tool
        l_ct_ref_stages = l_ct
        stage_w = _pad_stage_weights_to6(l_tool.new_tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    if traj_target == "tool":
        l_ct_stage0 = l_ct_stage0 * 0.0
        l_ct_stage1 = l_ct_stage1 * 0.0
        l_ct_stage2 = l_ct_stage2 * 0.0
        l_ct_stage3 = l_ct_stage3 * 0.0
        l_ct_stage4 = l_ct_stage4 * 0.0
        l_ct_final = l_ct_final * 0.0
        l_ct_ref_total = l_ct_ref_total * 0.0
        l_ct_ref_stages = l_ct_ref_stages * 0.0
        l_ct = l_ct * 0.0
    elif traj_target == "ct":
        l_tool_stage0 = l_tool_stage0 * 0.0
        l_tool_stage1 = l_tool_stage1 * 0.0
        l_tool_stage2 = l_tool_stage2 * 0.0
        l_tool_stage3 = l_tool_stage3 * 0.0
        l_tool_stage4 = l_tool_stage4 * 0.0
        l_tool_stage5 = l_tool_stage5 * 0.0
        l_tool_final = l_tool_final * 0.0
        l_tool_ref_total = l_tool_ref_total * 0.0
        l_tool_ref_stages = l_tool_ref_stages * 0.0
        l_tool = l_tool * 0.0
    l_tool_heatmap_ce = l_tool * 0.0
    l_ct_heatmap_ce = l_ct * 0.0
    l_tool_centroid_aux = l_tool * 0.0
    l_tool_offset_aux = l_tool * 0.0
    l_tool_pair_sep_aux = l_tool * 0.0
    bsz = int(pred_tool_xy.shape[0])
    zero_ps = l_tool.new_zeros((bsz,))
    l_tool_cls = l_tool * 0.0
    l_action_cls = l_tool * 0.0
    tool_cls_ps = zero_ps
    action_cls_ps = zero_ps
    l_range_boundary_aux = l_tool * 0.0
    l_range_proposal = l_tool * 0.0
    l_range_quality = l_tool * 0.0
    l_range_sola = l_tool * 0.0

    if trajectory_only:
        probs_tool = outputs.get("probs_tool", None)
        probs_ct = outputs.get("probs_ct", None)
        heatmap_tool = outputs.get("heatmap_tool", None)
        heatmap_ct = outputs.get("heatmap_ct", None)
        if (
            float(traj_heatmap_ce_weight) > 0.0
            and (probs_tool is not None)
            and (probs_ct is not None)
            and (heatmap_tool is not None)
            and (heatmap_ct is not None)
            and bool(getattr(probs_tool, "requires_grad", False))
            and bool(getattr(probs_ct, "requires_grad", False))
        ):
            h_tool = int(heatmap_tool.shape[-2])
            w_tool = int(heatmap_tool.shape[-1])
            h_ct = int(heatmap_ct.shape[-2])
            w_ct = int(heatmap_ct.shape[-1])
            q_tool = _build_gaussian_target_probs(
                gt_xy=gt_tool_xy,
                h_heatmap=h_tool,
                w_heatmap=w_tool,
                sigma=float(traj_heatmap_sigma),
            )
            q_ct = _build_gaussian_target_probs(
                gt_xy=gt_ct_xy,
                h_heatmap=h_ct,
                w_heatmap=w_ct,
                sigma=float(traj_heatmap_sigma),
            )
            l_tool_heatmap_ce = masked_prob_ce_loss(
                pred_probs=probs_tool,
                target_probs=q_tool,
                weight=gt_tool_vis,
            )
            l_ct_heatmap_ce = masked_prob_ce_loss(
                pred_probs=probs_ct,
                target_probs=q_ct,
                weight=gt_ct_vis,
            )
        if traj_target == "tool":
            l_ct_heatmap_ce = l_ct_heatmap_ce * 0.0
        elif traj_target == "ct":
            l_tool_heatmap_ce = l_tool_heatmap_ce * 0.0
        pred_tool_centroid_xy = outputs.get("traj_tool_centroid_xy", None)
        pred_tool_offsets_xy = outputs.get("traj_tool_offsets_xy", None)
        if (
            (pred_tool_centroid_xy is not None)
            and (pred_tool_offsets_xy is not None)
            and (traj_target in ("both", "tool"))
        ):
            gt_tool_centroid_xy, gt_tool_offsets_xy, traj_tool_centroid_weight, traj_tool_offset_weight = (
                _trajectory_tool_factorized_targets(
                    gt_tool_xy=gt_tool_xy,
                    gt_tool_vis=gt_tool_vis,
                    traj_loss_visible_only=bool(traj_loss_visible_only),
                    traj_t0_loss_weight=float(traj_t0_loss_weight),
                )
            )
            if float(traj_centroid_aux_weight) > 0.0:
                l_tool_centroid_aux = kpt_loss_fn(
                    pred_tool_centroid_xy,
                    gt_tool_centroid_xy,
                    traj_tool_centroid_weight,
                )
            if float(traj_offset_aux_weight) > 0.0:
                l_tool_offset_aux = kpt_loss_fn(
                    pred_tool_offsets_xy,
                    gt_tool_offsets_xy,
                    traj_tool_offset_weight,
                )
        # --- E005: direct supervision of the per-point coarse attention maps + confidence BCE ---
        zero_pp = l_tool * 0.0
        l_tool_attn_ce = zero_pp
        l_conf_bce = zero_pp
        pp_hit1 = zero_pp
        pp_offtool2 = zero_pp
        _pa = outputs.get("pred_point_attn", None)
        if (
            _pa is not None
            and (traj_target in ("both", "tool"))
            and int(_pa.shape[1]) == int(gt_tool_xy.shape[1])
            and int(_pa.shape[2]) == int(gt_tool_xy.shape[2])
        ):
            _pa = _pa.float()
            _s = int(_pa.shape[-1])
            _side = int(round(float(_s) ** 0.5))
            if _side * _side == _s:
                _w_pp = (gt_tool_vis > 0.5).to(_pa.dtype)  # hard visibility mask [B,T,P]
                _q = _build_gaussian_cell_targets(gt_tool_xy.float(), _side, _side, sigma=float(perpoint_ce_sigma_cells))
                if float(perpoint_attn_ce_weight) > 0.0:
                    l_tool_attn_ce = masked_prob_ce_loss(_pa, _q, _w_pp)
                with torch.no_grad():
                    _peak = _pa.argmax(dim=-1)
                    _px = (_peak % _side).float()
                    _py = torch.div(_peak, _side, rounding_mode="floor").float()
                    _gx = (gt_tool_xy[..., 0].float() * _side).floor().clamp(0, _side - 1)
                    _gy = (gt_tool_xy[..., 1].float() * _side).floor().clamp(0, _side - 1)
                    _cheb = torch.maximum((_px - _gx).abs(), (_py - _gy).abs())
                    _den = _w_pp.sum().clamp_min(1.0)
                    pp_hit1 = ((_cheb <= 1.0).float() * _w_pp).sum() / _den
                    pp_offtool2 = ((_cheb > 2.0).float() * _w_pp).sum() / _den
        # --- E006: per-token detector CE (soft Gaussian targets over {bg, anchor, tip}) + decode diagnostics ---
        l_tool_det_ce = zero_pp
        det_hit1 = zero_pp
        det_offtool2 = zero_pp
        _dl = outputs.get("pred_det_logits", None)
        if (
            _dl is not None
            and (traj_target in ("both", "tool"))
            and int(_dl.shape[1]) == int(gt_tool_xy.shape[1])
            and tool_pt_role is not None
        ):
            _dl = _dl.float()  # [B,T,S,3]
            _s = int(_dl.shape[2]); _side = int(round(float(_s) ** 0.5))
            if _side * _side == _s:
                _vis = (gt_tool_vis > 0.5).to(_dl.dtype)  # [B,T,P]
                _g = _build_gaussian_cell_targets(gt_tool_xy.float(), _side, _side, sigma=float(token_det_sigma_cells))  # [B,T,P,S], sums to 1
                _g = _g / _g.amax(dim=-1, keepdim=True).clamp_min(1e-9)  # peak = 1 at the GT cell
                _role = tool_pt_role.view(int(_dl.shape[0]), 1, -1, 1).to(_dl.dtype)  # [B,1,P,1]; 0 = anchor, 1 = tip
                _q_anchor = ((_g * _vis.unsqueeze(-1)) * (1.0 - _role)).amax(dim=2)  # [B,T,S]
                _q_tip = ((_g * _vis.unsqueeze(-1)) * _role).amax(dim=2)
                _q_bg = (1.0 - _q_anchor - _q_tip).clamp(0.0, 1.0)
                _q = torch.stack([_q_bg, _q_anchor, _q_tip], dim=-1)
                _q = _q / _q.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                _logp = torch.log_softmax(_dl, dim=-1)
                _ce = -(_q * _logp).sum(-1)  # [B,T,S]
                _wtok = 1.0 + 2.0 * (_q_anchor + _q_tip)  # class weights (1,3,3) in soft form
                _fr = (_vis.sum(-1) > 0).to(_dl.dtype)  # frames with any visible point
                if float(token_det_ce_weight) > 0.0:
                    l_tool_det_ce = ((_ce * _wtok).mean(-1) * _fr).sum() / _fr.sum().clamp_min(1.0)
                _dxy = outputs.get("pred_xy_det", None)
                if _dxy is not None and tuple(_dxy.shape) == tuple(gt_tool_xy.shape):
                    with torch.no_grad():
                        _dx = (_dxy[..., 0].float() * _side).floor().clamp(0, _side - 1)
                        _dy = (_dxy[..., 1].float() * _side).floor().clamp(0, _side - 1)
                        _gx = (gt_tool_xy[..., 0].float() * _side).floor().clamp(0, _side - 1)
                        _gy = (gt_tool_xy[..., 1].float() * _side).floor().clamp(0, _side - 1)
                        _cheb = torch.maximum((_dx - _gx).abs(), (_dy - _gy).abs())
                        _den = _vis.sum().clamp_min(1.0)
                        det_hit1 = ((_cheb <= 1.0).float() * _vis).sum() / _den
                        det_offtool2 = ((_cheb > 2.0).float() * _vis).sum() / _den
        # --- C003: token region prior BCE + centroid attention CE against the GT region interior (hull of visible points) ---
        l_tool_region_bce = zero_pp
        l_tool_centroid_region_ce = zero_pp
        region_hit1 = zero_pp
        _rl = outputs.get("pred_region_logits", None)
        _ca = outputs.get("pred_centroid_attn", None)
        if (traj_target in ("both", "tool")) and ((_rl is not None) or (_ca is not None)):
            _ref = _rl if _rl is not None else _ca
            if int(_ref.shape[1]) == int(gt_tool_xy.shape[1]):
                _s = int(_ref.shape[2]); _side = int(round(float(_s) ** 0.5))
                if _side * _side == _s:
                    _vis = (gt_tool_vis > 0.5)
                    _tgt = _build_region_cell_targets(gt_tool_xy.float(), _vis, _side, sigma=float(region_sigma_cells))  # [B,T,S]
                    _fr = (_vis.sum(-1) > 0).float()  # [B,T]
                    if _rl is not None:
                        _rlf = _rl.float()
                        _pos = _tgt.sum(-1, keepdim=True).clamp_min(1.0)
                        _pw = ((float(_s) - _pos) / _pos).clamp(1.0, 50.0)  # per-frame positive weight (balance)
                        _bce = F.binary_cross_entropy_with_logits(_rlf, _tgt, reduction="none")
                        _wt = torch.where(_tgt > 0.5, _pw.expand_as(_tgt), torch.ones_like(_tgt))
                        _bce = (_bce * _wt).sum(-1) / _wt.sum(-1).clamp_min(1.0)  # [B,T]
                        if float(region_bce_weight) > 0.0:
                            l_tool_region_bce = (_bce * _fr).sum() / _fr.sum().clamp_min(1.0)
                        with torch.no_grad():
                            _am = _rlf.argmax(-1)  # [B,T]
                            _hit = (torch.gather(_tgt, -1, _am.unsqueeze(-1)).squeeze(-1) > 0.5).float()
                            region_hit1 = (_hit * _fr).sum() / _fr.sum().clamp_min(1.0)
                    if _ca is not None and float(centroid_region_ce_weight) > 0.0:
                        _q = _tgt / _tgt.sum(-1, keepdim=True).clamp_min(1e-9)
                        _ce = -(_q * torch.log(_ca.float().clamp_min(1e-9))).sum(-1)  # [B,T]
                        l_tool_centroid_region_ce = (_ce * _fr).sum() / _fr.sum().clamp_min(1.0)
        # --- C004: soft region Dice between the quad spanned by the predicted points and the GT region (order-free, the
        #   user's metric is the convex-hull IoU of the 4 contact points) ---
        l_tool_region_dice = zero_pp
        region_soft_iou = zero_pp
        if float(region_dice_weight) > 0.0 and (traj_target in ("both", "tool")) and int(pred_tool_xy.shape[2]) >= 3:
            _sd = int(region_dice_side)
            _visq = (gt_tool_vis > 0.5)
            _frq = (_visq.sum(-1) >= 3).float()  # frames whose GT region is a proper polygon
            if region_mask_gt is not None and int(region_mask_gt.shape[1]) == int(pred_tool_xy.shape[1]):
                _rm = region_mask_gt.float(); _sd = int(_rm.shape[-1]); _frq = ((_rm.flatten(2).amax(-1) > 0.5).float())  # C006: precomputed target (e.g. hindsight hotspot)
            if float(_frq.sum()) > 0:
                _mp = _soft_quad_mask(pred_tool_xy.float(), _sd, tau=float(region_dice_tau))  # [B,T,S]
                _mg = (_rm.flatten(2) if region_mask_gt is not None and int(region_mask_gt.shape[1]) == int(pred_tool_xy.shape[1]) else _build_region_cell_targets(gt_tool_xy.float(), _visq, _sd, sigma=0.5))  # [B,T,S]
                _inter = (_mp * _mg).sum(-1); _dice = 1.0 - (2.0 * _inter + 1.0) / (_mp.sum(-1) + _mg.sum(-1) + 1.0)
                l_tool_region_dice = (_dice * _frq).sum() / _frq.sum().clamp_min(1.0)
                with torch.no_grad():
                    _iou = (_inter / (_mp.sum(-1) + _mg.sum(-1) - _inter).clamp_min(1e-6))
                    region_soft_iou = (_iou * _frq).sum() / _frq.sum().clamp_min(1.0)
        # --- E007f: pair-gap consistency. Every run so far predicts the symmetric pair ~0.8x as far apart as GT
        #   (regression-to-the-mean shrinkage). Huber on the predicted vs GT pair distance (px), pairs with both
        #   points visible; also reports the mean predicted/GT gap ratio as a diagnostic.
        l_tool_pair_gap = zero_pp
        pair_gap_ratio = zero_pp
        if (traj_target in ("both", "tool")) and tool_pt_role is not None and int(pred_tool_xy.shape[2]) == 3:
            _roles = tool_pt_role.view(int(pred_tool_xy.shape[0]), -1).long()  # [B,P]
            _cnt = (_roles.unsqueeze(1) == _roles.unsqueeze(2)).sum(-1)  # [B,P] how many points share my role
            _pair_mask = (_cnt == 2)
            _valid_b = _pair_mask.sum(-1) == 2
            if bool(_valid_b.any()):
                _idx = torch.stack([torch.nonzero(_pair_mask[b]).flatten()[:2] if _valid_b[b] else torch.tensor([0, 1], device=pred_tool_xy.device) for b in range(int(pred_tool_xy.shape[0]))])  # [B,2]
                _ia = _idx[:, 0].view(-1, 1, 1, 1).expand(-1, int(pred_tool_xy.shape[1]), 1, 2)
                _ib = _idx[:, 1].view(-1, 1, 1, 1).expand(-1, int(pred_tool_xy.shape[1]), 1, 2)
                _pa = pred_tool_xy.float().gather(2, _ia).squeeze(2); _pb = pred_tool_xy.float().gather(2, _ib).squeeze(2)
                _ga = gt_tool_xy.float().gather(2, _ia).squeeze(2); _gb = gt_tool_xy.float().gather(2, _ib).squeeze(2)
                _va = gt_tool_vis.gather(2, _idx[:, 0].view(-1, 1, 1).expand(-1, int(gt_tool_vis.shape[1]), 1)).squeeze(2)
                _vb = gt_tool_vis.gather(2, _idx[:, 1].view(-1, 1, 1).expand(-1, int(gt_tool_vis.shape[1]), 1)).squeeze(2)
                _wv = ((_va > 0.5) & (_vb > 0.5)).float() * _valid_b.view(-1, 1).float()  # [B,T]
                _gp = ((_pa - _pb) * float(traj_loss_pixel_scale)).norm(dim=-1)
                _gg = ((_ga - _gb) * float(traj_loss_pixel_scale)).norm(dim=-1)
                _den = _wv.sum().clamp_min(1.0)
                if float(pair_gap_loss_weight) > 0.0:
                    l_tool_pair_gap = (F.smooth_l1_loss(_gp, _gg, reduction="none", beta=float(pair_gap_huber_delta_px)) * _wv).sum() / _den
                with torch.no_grad():
                    pair_gap_ratio = ((_gp / _gg.clamp_min(1.0)) * _wv).sum() / _den
        _cl = outputs.get("pred_conf_logit", None)
        if (
            _cl is not None
            and float(conf_bce_weight) > 0.0
            and (traj_target in ("both", "tool"))
            and tuple(_cl.shape) == tuple(gt_tool_vis.shape)
        ):
            with torch.no_grad():
                _err_px = (pred_tool_xy.detach().float() - gt_tool_xy.float()).norm(dim=-1) * float(traj_loss_pixel_scale)
                _tgt = ((_err_px < float(conf_err_thresh_px)) & (gt_tool_vis > 0.5)).float()
            l_conf_bce = F.binary_cross_entropy_with_logits(_cl.float(), _tgt)
        if (
            bool(symmetry_pair_separation_loss)
            and float(symmetry_pair_separation_weight) > 0.0
            and (traj_target in ("both", "tool"))
        ):
            l_tool_pair_sep_aux, _ = _symmetric_pair_separation_aux_impl(
                pred=pred_tool_xy,
                target=gt_tool_xy,
                vis=gt_tool_vis,
                tool_pt_role=tool_pt_role,
                pixel_scale=traj_loss_pixel_scale,
                min_pred_gap_px=symmetry_pair_separation_min_pred_gap_px,
                gt_min_gap_px=symmetry_pair_separation_gt_min_gap_px,
            )
        l_combo = l_tool * 0.0
        l_start_ce = l_tool * 0.0
        l_end_ce = l_tool * 0.0
        l_range = l_tool * 0.0
        start_ce_ps = zero_ps
        end_ce_ps = zero_ps
        range_ps = zero_ps
        l_range_cls = l_tool * 0.0
        l_range_reg = l_tool * 0.0
        l_range_sola = l_tool * 0.0
    elif tool_only:
        tool_id, _ = _derive_tool_action_id_from_combo(combo_id.long())
        l_tool_cls = combo_ce_loss(
            outputs["combo_tool_logits"],
            tool_id,
            class_weights=tool_class_weights,
        )
        tool_cls_ps = combo_ce_per_sample(
            outputs["combo_tool_logits"],
            tool_id,
            class_weights=tool_class_weights,
        )
        l_combo = l_tool * 0.0
        l_action_cls = l_tool * 0.0
        action_cls_ps = zero_ps
        l_start_ce = l_tool * 0.0
        l_end_ce = l_tool * 0.0
        l_range = l_tool * 0.0
        start_ce_ps = zero_ps
        end_ce_ps = zero_ps
        range_ps = zero_ps
        l_range_cls = l_tool * 0.0
        l_range_reg = l_tool * 0.0
        l_tool = l_tool * 0.0
        l_ct = l_ct * 0.0
    elif range_only:
        l_combo = l_tool * 0.0
        l_tool_cls = l_tool * 0.0
        l_action_cls = l_tool * 0.0
        tool_cls_ps = zero_ps
        action_cls_ps = zero_ps
        use_range_offset = (
            ("range_actionness_logits" in outputs)
            and ("range_offsets" in outputs)
            and (gt_action_start_norm is not None)
            and (gt_action_end_norm is not None)
        )
        if use_range_offset:
            range_losses = range_offset_losses(
                range_actionness_logits=outputs["range_actionness_logits"],
                range_offsets=outputs["range_offsets"],
                gt_start_norm=gt_action_start_norm,
                gt_end_norm=gt_action_end_norm,
                range_mask=action_range_mask,
                positive_mode=str(range_positive_mode),
                range_start_logits=outputs.get("range_start_logits", None),
                range_end_logits=outputs.get("range_end_logits", None),
                range_proposal_logits=outputs.get("range_proposal_logits", None),
                range_quality_logits=outputs.get("range_quality_logits", None),
                range_sola_projected=outputs.get("range_sola_projected", None),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                cls_target_mode=str(range_cls_target_mode),
                cls_varifocal_alpha=float(range_cls_varifocal_alpha),
                cls_varifocal_weight=float(range_cls_varifocal_weight),
                quality_mode=quality_mode,
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                sola_similarity_k=float(range_sola_similarity_k),
                proposal_decode_mode=proposal_decode_mode,
                boundary_refine_mode=boundary_refine_mode,
                boundary_refine_radius=boundary_refine_radius,
                sensitivity_mode=str(range_sensitivity_mode),
                sensitivity_cls_sigma=float(range_sensitivity_cls_sigma),
                sensitivity_loc_sigma=float(range_sensitivity_loc_sigma),
                sensitivity_min_weight=float(range_sensitivity_min_weight),
                scale_invariant_mode=str(range_scale_invariant_mode),
                scale_invariant_alpha=float(range_scale_invariant_alpha),
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                focal_gamma=float(range_focal_gamma),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
            )
            range_losses = _apply_range_combo_reweight(
                range_losses=range_losses,
                combo_id=combo_id,
                combo_weights=range_combo_reweight_weights,
                range_mask=action_range_mask,
                targets=range_combo_reweight_targets,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
                eps=1e-6,
            )
        else:
            range_losses = range_bin_ce_losses(
                logits_start_bin=outputs["logits_start_bin"],
                logits_end_bin=outputs["logits_end_bin"],
                gt_start_bin=gt_action_start_bin,
                gt_end_bin=gt_action_end_bin,
                range_mask=action_range_mask,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                label_smoothing=float(range_label_smoothing),
                start_bin_class_weights=start_bin_class_weights,
                end_bin_class_weights=end_bin_class_weights,
                focal_gamma=float(range_focal_gamma),
                neighbor_smoothing=float(range_neighbor_smoothing),
            )
        l_start_ce = range_losses["loss_start_bin_ce"]
        l_end_ce = range_losses["loss_end_bin_ce"]
        l_range = range_losses["loss_range"]
        start_ce_ps = range_losses["loss_start_bin_per_sample"]
        end_ce_ps = range_losses["loss_end_bin_per_sample"]
        range_ps = range_losses["loss_range_per_sample"]
        l_range_cls = range_losses.get("loss_range_cls", l_tool * 0.0)
        l_range_reg = range_losses.get("loss_range_reg", l_tool * 0.0)
        l_range_boundary_aux = range_losses.get("loss_range_boundary_aux", l_tool * 0.0)
        l_range_proposal = range_losses.get("loss_range_proposal", l_tool * 0.0)
        l_range_quality = range_losses.get("loss_range_quality", l_tool * 0.0)
        l_range_sola = range_losses.get("loss_range_sola", l_tool * 0.0)
        l_tool = l_tool * 0.0
        l_ct = l_ct * 0.0
    elif combo_only:
        l_combo = combo_ce_loss(
            outputs["combo_logits"],
            combo_id.long(),
            class_weights=combo_class_weights,
        )
        tool_id, action_id = _derive_tool_action_id_from_combo(combo_id.long())
        l_tool_cls = combo_ce_loss(
            outputs["combo_tool_logits"],
            tool_id,
            class_weights=tool_class_weights,
        )
        l_action_cls = combo_ce_loss(
            outputs["combo_action_logits"],
            action_id,
            class_weights=action_class_weights,
        )
        tool_cls_ps = combo_ce_per_sample(
            outputs["combo_tool_logits"],
            tool_id,
            class_weights=tool_class_weights,
        )
        action_cls_ps = combo_ce_per_sample(
            outputs["combo_action_logits"],
            action_id,
            class_weights=action_class_weights,
        )
        l_start_ce = l_tool * 0.0
        l_end_ce = l_tool * 0.0
        l_range = l_tool * 0.0
        start_ce_ps = zero_ps
        end_ce_ps = zero_ps
        range_ps = zero_ps
        l_range_cls = l_tool * 0.0
        l_range_reg = l_tool * 0.0
        l_range_quality = l_tool * 0.0
        l_range_sola = l_tool * 0.0
        l_tool = l_tool * 0.0
        l_ct = l_ct * 0.0
    else:
        l_combo = combo_ce_loss(
            outputs["combo_logits"],
            combo_id.long(),
            class_weights=combo_class_weights,
        )

        use_range_offset = (
            ("range_actionness_logits" in outputs)
            and ("range_offsets" in outputs)
            and (gt_action_start_norm is not None)
            and (gt_action_end_norm is not None)
        )
        if use_range_offset:
            range_losses = range_offset_losses(
                range_actionness_logits=outputs["range_actionness_logits"],
                range_offsets=outputs["range_offsets"],
                gt_start_norm=gt_action_start_norm,
                gt_end_norm=gt_action_end_norm,
                range_mask=action_range_mask,
                positive_mode=str(range_positive_mode),
                range_start_logits=outputs.get("range_start_logits", None),
                range_end_logits=outputs.get("range_end_logits", None),
                range_proposal_logits=outputs.get("range_proposal_logits", None),
                range_quality_logits=outputs.get("range_quality_logits", None),
                range_sola_projected=outputs.get("range_sola_projected", None),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                cls_target_mode=str(range_cls_target_mode),
                cls_varifocal_alpha=float(range_cls_varifocal_alpha),
                cls_varifocal_weight=float(range_cls_varifocal_weight),
                quality_mode=quality_mode,
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                sola_similarity_k=float(range_sola_similarity_k),
                proposal_decode_mode=proposal_decode_mode,
                boundary_refine_mode=boundary_refine_mode,
                boundary_refine_radius=boundary_refine_radius,
                sensitivity_mode=str(range_sensitivity_mode),
                sensitivity_cls_sigma=float(range_sensitivity_cls_sigma),
                sensitivity_loc_sigma=float(range_sensitivity_loc_sigma),
                sensitivity_min_weight=float(range_sensitivity_min_weight),
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                focal_gamma=float(range_focal_gamma),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
            )
            range_losses = _apply_range_combo_reweight(
                range_losses=range_losses,
                combo_id=combo_id,
                combo_weights=range_combo_reweight_weights,
                range_mask=action_range_mask,
                targets=range_combo_reweight_targets,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
                eps=1e-6,
            )
        else:
            range_losses = range_bin_ce_losses(
                logits_start_bin=outputs["logits_start_bin"],
                logits_end_bin=outputs["logits_end_bin"],
                gt_start_bin=gt_action_start_bin,
                gt_end_bin=gt_action_end_bin,
                range_mask=action_range_mask,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                label_smoothing=float(range_label_smoothing),
                start_bin_class_weights=start_bin_class_weights,
                end_bin_class_weights=end_bin_class_weights,
                focal_gamma=float(range_focal_gamma),
                neighbor_smoothing=float(range_neighbor_smoothing),
            )
        l_start_ce = range_losses["loss_start_bin_ce"]
        l_end_ce = range_losses["loss_end_bin_ce"]
        l_range = range_losses["loss_range"]
        start_ce_ps = range_losses["loss_start_bin_per_sample"]
        end_ce_ps = range_losses["loss_end_bin_per_sample"]
        range_ps = range_losses["loss_range_per_sample"]
        l_range_cls = range_losses.get("loss_range_cls", l_tool * 0.0)
        l_range_reg = range_losses.get("loss_range_reg", l_tool * 0.0)
        l_range_boundary_aux = range_losses.get("loss_range_boundary_aux", l_tool * 0.0)
        l_range_proposal = range_losses.get("loss_range_proposal", l_tool * 0.0)
        l_range_quality = range_losses.get("loss_range_quality", l_tool * 0.0)
        l_range_sola = range_losses.get("loss_range_sola", l_tool * 0.0)

    # Disable auxiliary regularizers in the simplified loss.
    combo_info_w = 0.0
    l_combo_info = l_combo * 0.0
    l_traj_vel = l_tool * 0.0
    l_traj_jerk = l_tool * 0.0

    # Keep compatibility keys expected by logging/debug code.
    l_tool_s0 = l_tool_stage0 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_tool
    l_tool_s1 = l_tool_stage1 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_tool
    l_tool_s2 = l_tool_stage2 if (use_s2_stage_loss or use_m2_stage_loss) else l_tool
    l_tool_s3 = l_tool_stage3 if use_m2_stage_loss else (l_tool if not (use_s1_stage_loss or use_s2_stage_loss) else l_tool_stage3)
    l_tool_s4 = l_tool_stage4 if use_m2_stage_loss else (l_tool if not (use_s1_stage_loss or use_s2_stage_loss) else l_tool_stage4)
    l_tool_s5 = l_tool_stage5 if use_m2_stage_loss else zero_tool
    l_ct_s0 = l_ct_stage0 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_ct
    l_ct_s1 = l_ct_stage1 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_ct
    l_ct_s2 = l_ct_stage2 if (use_s2_stage_loss or use_m2_stage_loss) else l_ct
    l_ct_s3 = l_ct_stage3 if use_m2_stage_loss else (l_ct if not (use_s1_stage_loss or use_s2_stage_loss) else l_ct_stage3)
    l_ct_s4 = l_ct_stage4 if use_m2_stage_loss else (l_ct if not (use_s1_stage_loss or use_s2_stage_loss) else l_ct_stage4)
    l_ct_s5 = l_ct_stage5 if use_m2_stage_loss else zero_ct
    if trajectory_only:
        l_total = (
            l_tool
            + l_ct
            + float(traj_heatmap_ce_weight) * (l_tool_heatmap_ce + l_ct_heatmap_ce)
            + float(traj_centroid_aux_weight) * l_tool_centroid_aux
            + float(traj_offset_aux_weight) * l_tool_offset_aux
            + float(symmetry_pair_separation_weight) * l_tool_pair_sep_aux
            + float(perpoint_attn_ce_weight) * l_tool_attn_ce
            + float(conf_bce_weight) * l_conf_bce
            + float(token_det_ce_weight) * l_tool_det_ce
            + float(region_bce_weight) * l_tool_region_bce
            + float(centroid_region_ce_weight) * l_tool_centroid_region_ce
            + float(region_dice_weight) * l_tool_region_dice
            + float(pair_gap_loss_weight) * l_tool_pair_gap
        )
    elif tool_only:
        l_total = float(combo_tool_ce_weight) * l_tool_cls
    elif range_only:
        l_total = l_range
    elif combo_only:
        l_total = (
            float(combo_ce_weight) * l_combo
            + float(combo_tool_ce_weight) * l_tool_cls
            + float(combo_action_ce_weight) * l_action_cls
        )
    else:
        l_total = (
            l_tool
            + l_ct
            + float(combo_ce_weight) * l_combo
            + l_range
        )
    out = {
        "loss_total": l_total,
        "loss_tool_xy": l_tool,
        "loss_ct_xy": l_ct,
        "loss_tool_centroid_aux": l_tool_centroid_aux,
        "loss_tool_offset_aux": l_tool_offset_aux,
        "loss_tool_pair_sep_aux": l_tool_pair_sep_aux,
        "loss_tool_attn_ce": l_tool_attn_ce,
        "loss_conf_bce": l_conf_bce,
        "pp_hit1": pp_hit1,
        "pp_offtool2": pp_offtool2,
        "loss_tool_det_ce": l_tool_det_ce,
        "loss_tool_region_bce": l_tool_region_bce,
        "loss_tool_centroid_region_ce": l_tool_centroid_region_ce,
        "region_hit1": region_hit1,
        "loss_tool_region_dice": l_tool_region_dice,
        "region_soft_iou": region_soft_iou,
        "det_hit1": det_hit1,
        "det_offtool2": det_offtool2,
        "loss_tool_pair_gap": l_tool_pair_gap,
        "pair_gap_ratio": pair_gap_ratio,
        "loss_tool_heatmap_ce": l_tool_heatmap_ce,
        "loss_ct_heatmap_ce": l_ct_heatmap_ce,
        "loss_tool_xy_stage0": l_tool_s0,
        "loss_ct_xy_stage0": l_ct_s0,
        "loss_tool_xy_stage1": l_tool_s1,
        "loss_ct_xy_stage1": l_ct_s1,
        "loss_tool_xy_stage2": l_tool_s2,
        "loss_ct_xy_stage2": l_ct_s2,
        "loss_tool_xy_stage3": l_tool_s3,
        "loss_ct_xy_stage3": l_ct_s3,
        "loss_tool_xy_stage4": l_tool_s4,
        "loss_ct_xy_stage4": l_ct_s4,
        "loss_tool_xy_stage5": l_tool_s5,
        "loss_ct_xy_stage5": l_ct_s5,
        # Backward-compatible aliases.
        "loss_tool_xy_base": l_tool_s0,
        "loss_ct_xy_base": l_ct_s0,
        "loss_tool_xy_refine": l_tool_final,
        "loss_ct_xy_refine": l_ct_final,
        "loss_tool_xy_refine_total": l_tool_ref_total,
        "loss_ct_xy_refine_total": l_ct_ref_total,
        "loss_tool_xy_refine_stages": l_tool_ref_stages,
        "loss_ct_xy_refine_stages": l_ct_ref_stages,
        "loss_traj_vel": l_traj_vel,
        "loss_traj_jerk": l_traj_jerk,
        "loss_combo_ce": l_combo,
        "loss_combo_tool_ce": l_tool_cls,
        "loss_combo_action_ce": l_action_cls,
        "loss_combo_info": l_combo_info,
        "loss_start_bin_ce": l_start_ce,
        "loss_end_bin_ce": l_end_ce,
        "loss_range": l_range,
        "loss_range_cls": l_range_cls,
        "loss_range_reg": l_range_reg,
        "loss_range_boundary_aux": l_range_boundary_aux,
        "loss_range_proposal": l_range_proposal,
        "loss_range_quality": l_range_quality,
        "loss_range_sola": l_range_sola,
        "combo_info_bottleneck_weight": l_tool.new_tensor(combo_info_w),
        "traj_refine_stage_ratio": l_tool.new_tensor(float(traj_refine_stage_ratio)),
        "traj_iter_aux_weight": l_tool.new_tensor(float(traj_iter_aux_weight)),
        "traj_stage_weight_start": l_tool.new_tensor(float(traj_stage_weight_start)),
        "traj_stage_weight_end": l_tool.new_tensor(float(traj_stage_weight_end)),
        "traj_stage0_weight": stage_w[0],
        "traj_stage1_weight": stage_w[1],
        "traj_stage2_weight": stage_w[2],
        "traj_stage3_weight": stage_w[3],
        "traj_stage4_weight": stage_w[4],
        "traj_stage5_weight": stage_w[5],
        "loss_start_bin_per_sample": start_ce_ps,
        "loss_end_bin_per_sample": end_ce_ps,
        "loss_range_per_sample": range_ps,
        "loss_combo_tool_ce_per_sample": tool_cls_ps,
        "loss_combo_action_ce_per_sample": action_cls_ps,
    }
    if "pred_action_start_norm" in outputs:
        out["pred_action_start_norm"] = outputs["pred_action_start_norm"]
    if "pred_action_end_norm" in outputs:
        out["pred_action_end_norm"] = outputs["pred_action_end_norm"]
    out.update(late_focal_stats)
    return out


def compute_batch_losses_per_sample(
    outputs: Dict[str, torch.Tensor],
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    gt_ct_xy: torch.Tensor,
    gt_ct_vis: torch.Tensor,
    combo_id: torch.Tensor,
    gt_action_start_bin: torch.Tensor,
    gt_action_end_bin: torch.Tensor,
    action_range_mask: Optional[torch.Tensor],
    gt_action_start_norm: Optional[torch.Tensor] = None,
    gt_action_end_norm: Optional[torch.Tensor] = None,
    combo_ce_weight: float = 0.1,
    combo_tool_ce_weight: float = 0.5,
    combo_action_ce_weight: float = 0.5,
    combo_info_bottleneck_weight: float = 0.0,
    start_bin_loss_weight: float = 0.1,
    end_bin_loss_weight: float = 0.1,
    range_label_smoothing: float = 0.0,
    range_positive_mode: str = "span",
    range_sensitivity_mode: str = "none",
    range_sensitivity_cls_sigma: float = 0.5,
    range_sensitivity_loc_sigma: float = 0.35,
    range_sensitivity_min_weight: float = 0.25,
    range_scale_invariant_mode: str = "none",
    range_scale_invariant_alpha: float = 0.8,
    range_cls_target_mode: str = "binary",
    range_cls_varifocal_alpha: float = 0.75,
    range_cls_varifocal_weight: float = 1.0,
    range_quality_mode: str = "none",
    range_quality_loss_weight: float = 0.0,
    range_sola_aux_weight: float = 0.0,
    range_sola_similarity_k: float = 16.0,
    range_combo_reweight_weights: Optional[torch.Tensor] = None,
    range_combo_reweight_targets: str = "",
    combo_class_weights: Optional[torch.Tensor] = None,
    tool_class_weights: Optional[torch.Tensor] = None,
    action_class_weights: Optional[torch.Tensor] = None,
    start_bin_class_weights: Optional[torch.Tensor] = None,
    end_bin_class_weights: Optional[torch.Tensor] = None,
    range_focal_gamma: float = 0.0,
    range_neighbor_smoothing: float = 0.0,
    range_boundary_aux_weight: float = 0.0,
    range_proposal_weight: float = 0.0,
    traj_vel_weight: float = 0.2,
    traj_jerk_weight: float = 0.05,
    traj_refine_stage_ratio: float = 1.0,
    traj_iter_aux_weight: float = 0.2,
    traj_stage_weight_start: float = 0.25,
    traj_stage_weight_end: float = 1.0,
    traj_stage0_weight: float = 0.35,
    traj_stage1_weight: float = 0.35,
    traj_stage2_weight: float = 0.20,
    traj_stage3_weight: float = 0.10,
    kpt_loss_type: str = "l1l2",
    traj_loss_pixel_scale: float = 256.0,
    traj_huber_delta: float = 6.0,
    traj_t0_loss_weight: float = 1.0,
    traj_loss_visible_only: bool = True,
    traj_centroid_aux_weight: float = 0.0,
    traj_offset_aux_weight: float = 0.0,
    trajectory_target: str = "both",
    trajectory_only: bool = False,
    range_only: bool = False,
    combo_only: bool = False,
    tool_only: bool = False,
    traj_heatmap_ce_weight: float = 0.5,
    traj_heatmap_sigma: float = 2.45,
    tool_pt_role: Optional[torch.Tensor] = None,
    symmetry_swap_invariant_loss: bool = False,
    symmetry_pair_separation_loss: bool = False,
    symmetry_pair_separation_weight: float = 0.0,
    symmetry_pair_separation_min_pred_gap_px: float = 8.0,
    symmetry_pair_separation_gt_min_gap_px: float = 16.0,
    perpoint_attn_ce_weight: float = 0.0,
    perpoint_ce_sigma_cells: float = 1.0,
    conf_bce_weight: float = 0.0,
    conf_err_thresh_px: float = 8.0,
    token_det_ce_weight: float = 0.0,
    token_det_sigma_cells: float = 1.0,
    region_bce_weight: float = 0.0,
    region_sigma_cells: float = 1.0,
    centroid_region_ce_weight: float = 0.0,
    region_dice_weight: float = 0.0,
    region_dice_side: int = 48,
    region_dice_tau: float = 0.5,
    region_mask_gt=None,
    pair_gap_loss_weight: float = 0.0,
    pair_gap_huber_delta_px: float = 4.0,
    traj_m2_stage_weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    meta = outputs.get("meta", None)
    proposal_decode_mode = (
        str(meta.get("range_proposal_decode_mode", "argmax")).lower()
        if isinstance(meta, dict)
        else "argmax"
    )
    quality_mode = (
        str(meta.get("range_quality_mode", range_quality_mode)).lower()
        if isinstance(meta, dict)
        else str(range_quality_mode).lower()
    )
    boundary_refine_mode = (
        str(meta.get("range_boundary_refine_mode", "none")).lower()
        if isinstance(meta, dict)
        else "none"
    )
    boundary_refine_radius = (
        int(meta.get("range_boundary_refine_radius", 2))
        if isinstance(meta, dict)
        else 2
    )
    if tool_only:
        tool_id, _ = _derive_tool_action_id_from_combo(combo_id.long())
        tool_logits = outputs["combo_tool_logits"]
        tool_cls_ps = combo_ce_per_sample(
            tool_logits,
            tool_id,
            class_weights=tool_class_weights,
        )
        bsz = int(tool_logits.shape[0])
        zero_ps = tool_logits.new_zeros((bsz,))
        zero_scalar = tool_logits.new_zeros(())
        stage_w = _pad_stage_weights_to6(tool_logits.new_tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
        out = {
            "loss_total": float(combo_tool_ce_weight) * tool_cls_ps,
            "loss_tool_xy": zero_ps,
            "loss_ct_xy": zero_ps,
            "loss_tool_centroid_aux": zero_ps,
            "loss_tool_offset_aux": zero_ps,
            "loss_tool_pair_sep_aux": zero_ps,
            "loss_tool_heatmap_ce": zero_ps,
            "loss_ct_heatmap_ce": zero_ps,
            "loss_tool_xy_stage0": zero_ps,
            "loss_ct_xy_stage0": zero_ps,
            "loss_tool_xy_stage1": zero_ps,
            "loss_ct_xy_stage1": zero_ps,
            "loss_tool_xy_stage2": zero_ps,
            "loss_ct_xy_stage2": zero_ps,
            "loss_tool_xy_stage3": zero_ps,
            "loss_ct_xy_stage3": zero_ps,
            "loss_tool_xy_stage4": zero_ps,
            "loss_ct_xy_stage4": zero_ps,
            "loss_tool_xy_stage5": zero_ps,
            "loss_ct_xy_stage5": zero_ps,
            "loss_tool_xy_base": zero_ps,
            "loss_ct_xy_base": zero_ps,
            "loss_tool_xy_refine": zero_ps,
            "loss_ct_xy_refine": zero_ps,
            "loss_tool_xy_refine_total": zero_ps,
            "loss_ct_xy_refine_total": zero_ps,
            "loss_tool_xy_refine_stages": zero_ps,
            "loss_ct_xy_refine_stages": zero_ps,
            "loss_traj_vel": zero_ps,
            "loss_traj_jerk": zero_ps,
            "loss_combo_ce": zero_ps,
            "loss_combo_tool_ce": tool_cls_ps,
            "loss_combo_action_ce": zero_ps,
            "loss_combo_info": zero_ps,
            "loss_start_bin_ce": zero_ps,
            "loss_end_bin_ce": zero_ps,
            "loss_range": zero_ps,
            "loss_range_cls": zero_ps,
            "loss_range_reg": zero_ps,
            "combo_info_bottleneck_weight": zero_ps,
            "traj_refine_stage_ratio": tool_logits.new_full((bsz,), float(traj_refine_stage_ratio)),
            "traj_iter_aux_weight": tool_logits.new_full((bsz,), float(traj_iter_aux_weight)),
            "traj_stage_weight_start": tool_logits.new_full((bsz,), float(traj_stage_weight_start)),
            "traj_stage_weight_end": tool_logits.new_full((bsz,), float(traj_stage_weight_end)),
            "traj_stage0_weight": tool_logits.new_full((bsz,), float(stage_w[0].item())),
            "traj_stage1_weight": tool_logits.new_full((bsz,), float(stage_w[1].item())),
            "traj_stage2_weight": tool_logits.new_full((bsz,), float(stage_w[2].item())),
            "traj_stage3_weight": tool_logits.new_full((bsz,), float(stage_w[3].item())),
            "traj_stage4_weight": tool_logits.new_full((bsz,), float(stage_w[4].item())),
            "traj_stage5_weight": tool_logits.new_full((bsz,), float(stage_w[5].item())),
            "loss_start_bin_ce_scalar": zero_scalar,
            "loss_end_bin_ce_scalar": zero_scalar,
            "loss_range_scalar": zero_scalar,
            "loss_combo_tool_ce_scalar": tool_cls_ps.mean(),
            "loss_combo_action_ce_scalar": zero_scalar,
        }
        if "pred_action_start_norm" in outputs:
            out["pred_action_start_norm"] = outputs["pred_action_start_norm"]
        if "pred_action_end_norm" in outputs:
            out["pred_action_end_norm"] = outputs["pred_action_end_norm"]
        return out

    _, kpt_loss_fn_ps = _resolve_kpt_loss_type(
        kpt_loss_type,
        traj_loss_pixel_scale=traj_loss_pixel_scale,
        traj_huber_delta=traj_huber_delta,
    )
    meta = outputs.get("meta", {})
    traj_refiner_mode = ""
    if isinstance(meta, dict):
        traj_refiner_mode = str(meta.get("traj_refiner_mode", "")).lower()
    pred_tool_xy = outputs.get("pred_tool_xy_refine", outputs["pred_tool_xy"])
    pred_ct_xy = outputs.get("pred_ct_xy_refine", outputs["pred_ct_xy"])
    pred_tool_xy_stage0 = outputs.get("pred_tool_xy_stage0", None)
    pred_ct_xy_stage0 = outputs.get("pred_ct_xy_stage0", None)
    pred_tool_xy_stage1 = outputs.get("pred_tool_xy_stage1", None)
    pred_ct_xy_stage1 = outputs.get("pred_ct_xy_stage1", None)
    pred_tool_xy_stage2 = outputs.get("pred_tool_xy_stage2", None)
    pred_ct_xy_stage2 = outputs.get("pred_ct_xy_stage2", None)
    pred_tool_xy_stage_stack = _get_pred_stage_stack(
        outputs,
        stack_key="pred_tool_xy_stage_stack",
        stage_key_prefix="pred_tool_xy_stage",
    )
    pred_ct_xy_stage_stack = _get_pred_stage_stack(
        outputs,
        stack_key="pred_ct_xy_stage_stack",
        stage_key_prefix="pred_ct_xy_stage",
    )
    traj_tool_weight = gt_tool_vis if bool(traj_loss_visible_only) else torch.ones_like(gt_tool_vis)
    traj_ct_weight = gt_ct_vis if bool(traj_loss_visible_only) else torch.ones_like(gt_ct_vis)
    traj_tool_weight = _apply_t0_weight(traj_tool_weight, t0_loss_weight=traj_t0_loss_weight)
    traj_ct_weight = _apply_t0_weight(traj_ct_weight, t0_loss_weight=traj_t0_loss_weight)
    def _tool_kpt_loss_ps(
        pred_xy: torch.Tensor,
        gt_xy_arg: Optional[torch.Tensor] = None,
        traj_weight_arg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gt_xy_use = gt_tool_xy if gt_xy_arg is None else gt_xy_arg
        traj_weight_use = traj_tool_weight if traj_weight_arg is None else traj_weight_arg
        if bool(symmetry_swap_invariant_loss):
            return _swap_invariant_pair_loss_impl(
                pred=pred_xy,
                target=gt_xy_use,
                weight=traj_weight_use,
                tool_pt_role=tool_pt_role,
                kpt_loss_type=kpt_loss_type,
                traj_loss_pixel_scale=traj_loss_pixel_scale,
                traj_huber_delta=traj_huber_delta,
            )[1]
        return kpt_loss_fn_ps(pred_xy, gt_xy_use, traj_weight_use)

    l_tool_final = _tool_kpt_loss_ps(pred_tool_xy)
    l_ct_final = kpt_loss_fn_ps(pred_ct_xy, gt_ct_xy, traj_ct_weight)
    zero_tool = l_tool_final * 0.0
    zero_ct = l_ct_final * 0.0
    use_s1_stage_loss = bool(
        trajectory_only
        and (traj_refiner_mode == "s1_offset")
        and torch.is_tensor(pred_tool_xy_stage0)
    )
    use_s2_stage_loss = bool(
        trajectory_only
        and (
            traj_refiner_mode in (
                "s2_iter_offset",
                "s3_support3x3",
                "s3p2x_finalrefine_l18",
                "s4a_template_centercorr_3x3",
                "s4b_template_allpaircorr_3x3",
                "m1_strict_s3_multilayer_mean",
                "m1_progressive_multilayer_movingrefine",
            )
        )
        and torch.is_tensor(pred_tool_xy_stage0)
        and torch.is_tensor(pred_tool_xy_stage1)
        and torch.is_tensor(pred_tool_xy_stage2)
    )
    use_m2_stage_loss = bool(
        trajectory_only
        and (traj_refiner_mode in LAYERWISE_STATECARRY_STAGE_LOSS_MODES)
        and torch.is_tensor(pred_tool_xy_stage_stack)
        and (int(pred_tool_xy_stage_stack.shape[0]) >= 2)
    )
    if use_m2_stage_loss:
        m2_stage_weights = (
            list(traj_m2_stage_weights)
            if traj_m2_stage_weights is not None
            else [0.35, 0.45, 0.60, 0.80, 1.00]
        )
        tool_stage_loss_pack = _compute_stagewise_kpt_loss(
            pred_tool_xy_stage_stack,
            gt_tool_xy,
            traj_tool_weight,
            kpt_loss_fn=_tool_kpt_loss_ps,
            stage_weights=m2_stage_weights,
        )
        tool_stage_losses = list(tool_stage_loss_pack["stage_losses"])
        while len(tool_stage_losses) < 6:
            tool_stage_losses.append(zero_tool)
        l_tool_stage0 = tool_stage_losses[0]
        l_tool_stage1 = tool_stage_losses[1]
        l_tool_stage2 = tool_stage_losses[2]
        l_tool_stage3 = tool_stage_losses[3]
        l_tool_stage4 = tool_stage_losses[4]
        l_tool_stage5 = tool_stage_losses[5]
        l_tool = tool_stage_loss_pack["total"]
        l_tool_final = tool_stage_loss_pack["final"]
        l_tool_ref_total = tool_stage_loss_pack["refine_total"]
        l_tool_ref_stages = l_tool
        stage_w = _pad_stage_weights_to6(tool_stage_loss_pack["stage_weights"])
        if torch.is_tensor(pred_ct_xy_stage_stack) and int(pred_ct_xy_stage_stack.shape[0]) >= 2:
            ct_stage_loss_pack = _compute_stagewise_kpt_loss(
                pred_ct_xy_stage_stack,
                gt_ct_xy,
                traj_ct_weight,
                kpt_loss_fn=kpt_loss_fn_ps,
                stage_weights=m2_stage_weights,
            )
            ct_stage_losses = list(ct_stage_loss_pack["stage_losses"])
            while len(ct_stage_losses) < 6:
                ct_stage_losses.append(zero_ct)
            l_ct_stage0 = ct_stage_losses[0]
            l_ct_stage1 = ct_stage_losses[1]
            l_ct_stage2 = ct_stage_losses[2]
            l_ct_stage3 = ct_stage_losses[3]
            l_ct_stage4 = ct_stage_losses[4]
            l_ct_stage5 = ct_stage_losses[5]
            l_ct = ct_stage_loss_pack["total"]
            l_ct_final = ct_stage_loss_pack["final"]
            l_ct_ref_total = ct_stage_loss_pack["refine_total"]
            l_ct_ref_stages = l_ct
        else:
            l_ct_stage0 = zero_ct
            l_ct_stage1 = zero_ct
            l_ct_stage2 = zero_ct
            l_ct_stage3 = zero_ct
            l_ct_stage4 = zero_ct
            l_ct_stage5 = zero_ct
            l_ct = zero_ct
            l_ct_final = zero_ct
            l_ct_ref_total = zero_ct
            l_ct_ref_stages = zero_ct
    elif use_s2_stage_loss:
        s2_stage1_weight = 0.7
        l_tool_stage0 = _tool_kpt_loss_ps(pred_tool_xy_stage0)
        l_tool_stage1 = _tool_kpt_loss_ps(pred_tool_xy_stage1)
        l_tool_stage2 = _tool_kpt_loss_ps(pred_tool_xy_stage2)
        l_tool_stage3 = zero_tool
        l_tool_stage4 = zero_tool
        l_tool_stage5 = zero_tool
        if torch.is_tensor(pred_ct_xy_stage0):
            l_ct_stage0 = kpt_loss_fn_ps(pred_ct_xy_stage0, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage0 = zero_ct
        if torch.is_tensor(pred_ct_xy_stage1):
            l_ct_stage1 = kpt_loss_fn_ps(pred_ct_xy_stage1, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage1 = zero_ct
        if torch.is_tensor(pred_ct_xy_stage2):
            l_ct_stage2 = kpt_loss_fn_ps(pred_ct_xy_stage2, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage2 = zero_ct
        l_ct_stage3 = zero_ct
        l_ct_stage4 = zero_ct
        l_ct_stage5 = zero_ct
        l_tool = l_tool_stage0 + s2_stage1_weight * l_tool_stage1 + l_tool_stage2
        l_ct = l_ct_stage0 + s2_stage1_weight * l_ct_stage1 + l_ct_stage2
        l_tool_final = l_tool_stage2
        l_ct_final = l_ct_stage2
        l_tool_ref_total = 0.7 * l_tool_stage1 + l_tool_stage2
        l_ct_ref_total = 0.7 * l_ct_stage1 + l_ct_stage2
        l_tool_ref_stages = l_tool
        l_ct_ref_stages = l_ct
        stage_w = _pad_stage_weights_to6(l_tool.new_tensor([1.0, 0.7, 1.0, 0.0, 0.0]))
    elif use_s1_stage_loss:
        l_tool_stage0 = _tool_kpt_loss_ps(pred_tool_xy_stage0)
        if torch.is_tensor(pred_ct_xy_stage0):
            l_ct_stage0 = kpt_loss_fn_ps(pred_ct_xy_stage0, gt_ct_xy, traj_ct_weight)
        else:
            l_ct_stage0 = zero_ct
        l_tool_stage1 = l_tool_final
        l_ct_stage1 = l_ct_final
        l_tool_stage2 = zero_tool
        l_ct_stage2 = zero_ct
        l_tool_stage3 = zero_tool
        l_ct_stage3 = zero_ct
        l_tool_stage4 = zero_tool
        l_ct_stage4 = zero_ct
        l_tool_stage5 = zero_tool
        l_ct_stage5 = zero_ct
        l_tool = l_tool_stage0 + l_tool_final
        l_ct = l_ct_stage0 + l_ct_final
        l_tool_ref_total = l_tool_final
        l_ct_ref_total = l_ct_final
        l_tool_ref_stages = l_tool
        l_ct_ref_stages = l_ct
        stage_w = _pad_stage_weights_to6(l_tool.new_tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
    else:
        l_tool_stage0 = zero_tool
        l_ct_stage0 = zero_ct
        l_tool_stage1 = zero_tool
        l_ct_stage1 = zero_ct
        l_tool_stage2 = zero_tool
        l_ct_stage2 = zero_ct
        l_tool_stage3 = zero_tool
        l_ct_stage3 = zero_ct
        l_tool_stage4 = zero_tool
        l_ct_stage4 = zero_ct
        l_tool_stage5 = zero_tool
        l_ct_stage5 = zero_ct
        l_tool = l_tool_final
        l_ct = l_ct_final
        l_tool_ref_total = l_tool
        l_ct_ref_total = l_ct
        l_tool_ref_stages = l_tool
        l_ct_ref_stages = l_ct
        stage_w = _pad_stage_weights_to6(l_tool.new_tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    traj_target = str(trajectory_target).lower()
    if traj_target not in ("both", "tool", "ct"):
        raise ValueError(f"Unsupported trajectory_target={trajectory_target}. Expected one of: both, tool, ct.")
    if traj_target == "tool":
        l_ct_stage0 = l_ct_stage0 * 0.0
        l_ct_stage1 = l_ct_stage1 * 0.0
        l_ct_stage2 = l_ct_stage2 * 0.0
        l_ct_stage3 = l_ct_stage3 * 0.0
        l_ct_stage4 = l_ct_stage4 * 0.0
        l_ct_stage5 = l_ct_stage5 * 0.0
        l_ct_final = l_ct_final * 0.0
        l_ct_ref_total = l_ct_ref_total * 0.0
        l_ct_ref_stages = l_ct_ref_stages * 0.0
        l_ct = l_ct * 0.0
    elif traj_target == "ct":
        l_tool_stage0 = l_tool_stage0 * 0.0
        l_tool_stage1 = l_tool_stage1 * 0.0
        l_tool_stage2 = l_tool_stage2 * 0.0
        l_tool_stage3 = l_tool_stage3 * 0.0
        l_tool_stage4 = l_tool_stage4 * 0.0
        l_tool_stage5 = l_tool_stage5 * 0.0
        l_tool_final = l_tool_final * 0.0
        l_tool_ref_total = l_tool_ref_total * 0.0
        l_tool_ref_stages = l_tool_ref_stages * 0.0
        l_tool = l_tool * 0.0
    l_tool_heatmap_ce = l_tool * 0.0
    l_ct_heatmap_ce = l_ct * 0.0
    l_tool_centroid_aux = l_tool * 0.0
    l_tool_offset_aux = l_tool * 0.0
    l_tool_pair_sep_aux = l_tool * 0.0
    l_tool_cls = l_tool * 0.0
    l_action_cls = l_tool * 0.0
    tool_cls_ps = l_tool.new_zeros(l_tool.shape)
    action_cls_ps = l_tool.new_zeros(l_tool.shape)
    l_range_boundary_aux = l_tool.new_zeros(l_tool.shape)
    l_range_proposal = l_tool.new_zeros(l_tool.shape)
    l_range_quality = l_tool.new_zeros(l_tool.shape)
    l_range_sola = l_tool.new_zeros(l_tool.shape)

    if trajectory_only:
        probs_tool = outputs.get("probs_tool", None)
        probs_ct = outputs.get("probs_ct", None)
        heatmap_tool = outputs.get("heatmap_tool", None)
        heatmap_ct = outputs.get("heatmap_ct", None)
        if (
            float(traj_heatmap_ce_weight) > 0.0
            and (probs_tool is not None)
            and (probs_ct is not None)
            and (heatmap_tool is not None)
            and (heatmap_ct is not None)
            and bool(getattr(probs_tool, "requires_grad", False))
            and bool(getattr(probs_ct, "requires_grad", False))
        ):
            h_tool = int(heatmap_tool.shape[-2])
            w_tool = int(heatmap_tool.shape[-1])
            h_ct = int(heatmap_ct.shape[-2])
            w_ct = int(heatmap_ct.shape[-1])
            q_tool = _build_gaussian_target_probs(
                gt_xy=gt_tool_xy,
                h_heatmap=h_tool,
                w_heatmap=w_tool,
                sigma=float(traj_heatmap_sigma),
            )
            q_ct = _build_gaussian_target_probs(
                gt_xy=gt_ct_xy,
                h_heatmap=h_ct,
                w_heatmap=w_ct,
                sigma=float(traj_heatmap_sigma),
            )
            l_tool_heatmap_ce = masked_prob_ce_per_sample(
                pred_probs=probs_tool,
                target_probs=q_tool,
                weight=gt_tool_vis,
            )
            l_ct_heatmap_ce = masked_prob_ce_per_sample(
                pred_probs=probs_ct,
                target_probs=q_ct,
                weight=gt_ct_vis,
            )
        if traj_target == "tool":
            l_ct_heatmap_ce = l_ct_heatmap_ce * 0.0
        elif traj_target == "ct":
            l_tool_heatmap_ce = l_tool_heatmap_ce * 0.0
        pred_tool_centroid_xy = outputs.get("traj_tool_centroid_xy", None)
        pred_tool_offsets_xy = outputs.get("traj_tool_offsets_xy", None)
        if (
            (pred_tool_centroid_xy is not None)
            and (pred_tool_offsets_xy is not None)
            and (traj_target in ("both", "tool"))
        ):
            gt_tool_centroid_xy, gt_tool_offsets_xy, traj_tool_centroid_weight, traj_tool_offset_weight = (
                _trajectory_tool_factorized_targets(
                    gt_tool_xy=gt_tool_xy,
                    gt_tool_vis=gt_tool_vis,
                    traj_loss_visible_only=bool(traj_loss_visible_only),
                    traj_t0_loss_weight=float(traj_t0_loss_weight),
                )
            )
            if float(traj_centroid_aux_weight) > 0.0:
                l_tool_centroid_aux = kpt_loss_fn_ps(
                    pred_tool_centroid_xy,
                    gt_tool_centroid_xy,
                    traj_tool_centroid_weight,
                )
            if float(traj_offset_aux_weight) > 0.0:
                l_tool_offset_aux = kpt_loss_fn_ps(
                    pred_tool_offsets_xy,
                    gt_tool_offsets_xy,
                    traj_tool_offset_weight,
                )
        if (
            bool(symmetry_pair_separation_loss)
            and float(symmetry_pair_separation_weight) > 0.0
            and (traj_target in ("both", "tool"))
        ):
            _, l_tool_pair_sep_aux = _symmetric_pair_separation_aux_impl(
                pred=pred_tool_xy,
                target=gt_tool_xy,
                vis=gt_tool_vis,
                tool_pt_role=tool_pt_role,
                pixel_scale=traj_loss_pixel_scale,
                min_pred_gap_px=symmetry_pair_separation_min_pred_gap_px,
                gt_min_gap_px=symmetry_pair_separation_gt_min_gap_px,
            )
        l_combo = l_tool * 0.0
        l_start_ce = l_tool.new_zeros(())
        l_end_ce = l_tool.new_zeros(())
        l_range = l_tool.new_zeros(())
        start_ce_ps = l_tool.new_zeros(l_tool.shape)
        end_ce_ps = l_tool.new_zeros(l_tool.shape)
        range_ps = l_tool.new_zeros(l_tool.shape)
        l_range_cls = l_tool * 0.0
        l_range_reg = l_tool * 0.0
        l_range_sola = l_tool * 0.0
    elif tool_only:
        tool_id, _ = _derive_tool_action_id_from_combo(combo_id.long())
        tool_cls_ps = combo_ce_per_sample(
            outputs["combo_tool_logits"],
            tool_id,
            class_weights=tool_class_weights,
        )
        l_tool_cls = tool_cls_ps.mean()
        l_combo = l_tool * 0.0
        l_action_cls = l_tool * 0.0
        action_cls_ps = l_tool.new_zeros(l_tool.shape)
        l_start_ce = l_tool.new_zeros(())
        l_end_ce = l_tool.new_zeros(())
        l_range = l_tool.new_zeros(())
        start_ce_ps = l_tool.new_zeros(l_tool.shape)
        end_ce_ps = l_tool.new_zeros(l_tool.shape)
        range_ps = l_tool.new_zeros(l_tool.shape)
        l_range_cls = l_tool * 0.0
        l_range_reg = l_tool * 0.0
        l_tool = l_tool * 0.0
        l_ct = l_ct * 0.0
    elif range_only:
        l_combo = l_tool.new_zeros(l_tool.shape)
        tool_cls_ps = l_tool.new_zeros(l_tool.shape)
        action_cls_ps = l_tool.new_zeros(l_tool.shape)
        l_tool_cls = l_tool.new_zeros(())
        l_action_cls = l_tool.new_zeros(())
        use_range_offset = (
            ("range_actionness_logits" in outputs)
            and ("range_offsets" in outputs)
            and (gt_action_start_norm is not None)
            and (gt_action_end_norm is not None)
        )
        if use_range_offset:
            range_losses = range_offset_losses(
                range_actionness_logits=outputs["range_actionness_logits"],
                range_offsets=outputs["range_offsets"],
                gt_start_norm=gt_action_start_norm,
                gt_end_norm=gt_action_end_norm,
                range_mask=action_range_mask,
                positive_mode=str(range_positive_mode),
                range_start_logits=outputs.get("range_start_logits", None),
                range_end_logits=outputs.get("range_end_logits", None),
                range_proposal_logits=outputs.get("range_proposal_logits", None),
                range_quality_logits=outputs.get("range_quality_logits", None),
                range_sola_projected=outputs.get("range_sola_projected", None),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                cls_target_mode=str(range_cls_target_mode),
                cls_varifocal_alpha=float(range_cls_varifocal_alpha),
                cls_varifocal_weight=float(range_cls_varifocal_weight),
                quality_mode=quality_mode,
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                sola_similarity_k=float(range_sola_similarity_k),
                proposal_decode_mode=proposal_decode_mode,
                boundary_refine_mode=boundary_refine_mode,
                boundary_refine_radius=boundary_refine_radius,
                sensitivity_mode=str(range_sensitivity_mode),
                sensitivity_cls_sigma=float(range_sensitivity_cls_sigma),
                sensitivity_loc_sigma=float(range_sensitivity_loc_sigma),
                sensitivity_min_weight=float(range_sensitivity_min_weight),
                scale_invariant_mode=str(range_scale_invariant_mode),
                scale_invariant_alpha=float(range_scale_invariant_alpha),
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                focal_gamma=float(range_focal_gamma),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
            )
            range_losses = _apply_range_combo_reweight(
                range_losses=range_losses,
                combo_id=combo_id,
                combo_weights=range_combo_reweight_weights,
                range_mask=action_range_mask,
                targets=range_combo_reweight_targets,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
                eps=1e-6,
            )
        else:
            range_losses = range_bin_ce_losses(
                logits_start_bin=outputs["logits_start_bin"],
                logits_end_bin=outputs["logits_end_bin"],
                gt_start_bin=gt_action_start_bin,
                gt_end_bin=gt_action_end_bin,
                range_mask=action_range_mask,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                label_smoothing=float(range_label_smoothing),
                start_bin_class_weights=start_bin_class_weights,
                end_bin_class_weights=end_bin_class_weights,
                focal_gamma=float(range_focal_gamma),
                neighbor_smoothing=float(range_neighbor_smoothing),
            )
        l_start_ce = range_losses["loss_start_bin_ce"]
        l_end_ce = range_losses["loss_end_bin_ce"]
        l_range = range_losses["loss_range"]
        start_ce_ps = range_losses["loss_start_bin_per_sample"]
        end_ce_ps = range_losses["loss_end_bin_per_sample"]
        range_ps = range_losses["loss_range_per_sample"]
        l_range_cls = range_losses.get("loss_range_cls_per_sample", l_tool * 0.0)
        l_range_reg = range_losses.get("loss_range_reg_per_sample", l_tool * 0.0)
        l_range_boundary_aux = range_losses.get("loss_range_boundary_aux_per_sample", l_tool * 0.0)
        l_range_proposal = range_losses.get("loss_range_proposal_per_sample", l_tool * 0.0)
        l_range_quality = range_losses.get("loss_range_quality_per_sample", l_tool * 0.0)
        l_range_sola = range_losses.get("loss_range_sola_per_sample", l_tool * 0.0)
        l_tool = l_tool * 0.0
        l_ct = l_ct * 0.0
    elif combo_only:
        l_combo = combo_ce_per_sample(
            outputs["combo_logits"],
            combo_id.long(),
            class_weights=combo_class_weights,
        )
        tool_id, action_id = _derive_tool_action_id_from_combo(combo_id.long())
        tool_cls_ps = combo_ce_per_sample(
            outputs["combo_tool_logits"],
            tool_id,
            class_weights=tool_class_weights,
        )
        action_cls_ps = combo_ce_per_sample(
            outputs["combo_action_logits"],
            action_id,
            class_weights=action_class_weights,
        )
        l_tool_cls = tool_cls_ps.mean()
        l_action_cls = action_cls_ps.mean()
        l_start_ce = l_tool.new_zeros(())
        l_end_ce = l_tool.new_zeros(())
        l_range = l_tool.new_zeros(())
        start_ce_ps = l_tool.new_zeros(l_tool.shape)
        end_ce_ps = l_tool.new_zeros(l_tool.shape)
        range_ps = l_tool.new_zeros(l_tool.shape)
        l_range_cls = l_tool * 0.0
        l_range_reg = l_tool * 0.0
        l_range_quality = l_tool * 0.0
        l_range_sola = l_tool * 0.0
        l_tool = l_tool * 0.0
        l_ct = l_ct * 0.0
    else:
        l_combo = combo_ce_per_sample(
            outputs["combo_logits"],
            combo_id.long(),
            class_weights=combo_class_weights,
        )

        use_range_offset = (
            ("range_actionness_logits" in outputs)
            and ("range_offsets" in outputs)
            and (gt_action_start_norm is not None)
            and (gt_action_end_norm is not None)
        )
        if use_range_offset:
            range_losses = range_offset_losses(
                range_actionness_logits=outputs["range_actionness_logits"],
                range_offsets=outputs["range_offsets"],
                gt_start_norm=gt_action_start_norm,
                gt_end_norm=gt_action_end_norm,
                range_mask=action_range_mask,
                positive_mode=str(range_positive_mode),
                range_start_logits=outputs.get("range_start_logits", None),
                range_end_logits=outputs.get("range_end_logits", None),
                range_proposal_logits=outputs.get("range_proposal_logits", None),
                range_quality_logits=outputs.get("range_quality_logits", None),
                range_sola_projected=outputs.get("range_sola_projected", None),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                cls_target_mode=str(range_cls_target_mode),
                cls_varifocal_alpha=float(range_cls_varifocal_alpha),
                cls_varifocal_weight=float(range_cls_varifocal_weight),
                quality_mode=quality_mode,
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                sola_similarity_k=float(range_sola_similarity_k),
                proposal_decode_mode=proposal_decode_mode,
                boundary_refine_mode=boundary_refine_mode,
                boundary_refine_radius=boundary_refine_radius,
                sensitivity_mode=str(range_sensitivity_mode),
                sensitivity_cls_sigma=float(range_sensitivity_cls_sigma),
                sensitivity_loc_sigma=float(range_sensitivity_loc_sigma),
                sensitivity_min_weight=float(range_sensitivity_min_weight),
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                focal_gamma=float(range_focal_gamma),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
            )
            range_losses = _apply_range_combo_reweight(
                range_losses=range_losses,
                combo_id=combo_id,
                combo_weights=range_combo_reweight_weights,
                range_mask=action_range_mask,
                targets=range_combo_reweight_targets,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                boundary_aux_weight=float(range_boundary_aux_weight),
                proposal_weight=float(range_proposal_weight),
                quality_loss_weight=float(range_quality_loss_weight),
                sola_aux_weight=float(range_sola_aux_weight),
                cls_loss_weight=0.1,
                reg_loss_weight=1.0,
                eps=1e-6,
            )
        else:
            range_losses = range_bin_ce_losses(
                logits_start_bin=outputs["logits_start_bin"],
                logits_end_bin=outputs["logits_end_bin"],
                gt_start_bin=gt_action_start_bin,
                gt_end_bin=gt_action_end_bin,
                range_mask=action_range_mask,
                start_loss_weight=float(start_bin_loss_weight),
                end_loss_weight=float(end_bin_loss_weight),
                label_smoothing=float(range_label_smoothing),
                start_bin_class_weights=start_bin_class_weights,
                end_bin_class_weights=end_bin_class_weights,
                focal_gamma=float(range_focal_gamma),
                neighbor_smoothing=float(range_neighbor_smoothing),
            )
        l_start_ce = range_losses["loss_start_bin_ce"]
        l_end_ce = range_losses["loss_end_bin_ce"]
        l_range = range_losses["loss_range"]
        start_ce_ps = range_losses["loss_start_bin_per_sample"]
        end_ce_ps = range_losses["loss_end_bin_per_sample"]
        range_ps = range_losses["loss_range_per_sample"]
        l_range_cls = range_losses.get("loss_range_cls_per_sample", l_tool * 0.0)
        l_range_reg = range_losses.get("loss_range_reg_per_sample", l_tool * 0.0)
        l_range_boundary_aux = range_losses.get("loss_range_boundary_aux_per_sample", l_tool * 0.0)
        l_range_proposal = range_losses.get("loss_range_proposal_per_sample", l_tool * 0.0)
        l_range_quality = range_losses.get("loss_range_quality_per_sample", l_tool * 0.0)
        l_range_sola = range_losses.get("loss_range_sola_per_sample", l_tool * 0.0)

    combo_info_w = 0.0
    l_combo_info = l_combo * 0.0
    l_traj_vel = l_tool * 0.0
    l_traj_jerk = l_tool * 0.0

    l_tool_s0 = l_tool_stage0 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_tool
    l_tool_s1 = l_tool_stage1 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_tool
    l_tool_s2 = l_tool_stage2 if (use_s2_stage_loss or use_m2_stage_loss) else l_tool
    l_tool_s3 = l_tool_stage3 if use_m2_stage_loss else (l_tool if not (use_s1_stage_loss or use_s2_stage_loss) else l_tool_stage3)
    l_tool_s4 = l_tool_stage4 if use_m2_stage_loss else (l_tool if not (use_s1_stage_loss or use_s2_stage_loss) else l_tool_stage4)
    l_tool_s5 = l_tool_stage5 if use_m2_stage_loss else zero_tool
    l_ct_s0 = l_ct_stage0 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_ct
    l_ct_s1 = l_ct_stage1 if (use_s1_stage_loss or use_s2_stage_loss or use_m2_stage_loss) else l_ct
    l_ct_s2 = l_ct_stage2 if (use_s2_stage_loss or use_m2_stage_loss) else l_ct
    l_ct_s3 = l_ct_stage3 if use_m2_stage_loss else (l_ct if not (use_s1_stage_loss or use_s2_stage_loss) else l_ct_stage3)
    l_ct_s4 = l_ct_stage4 if use_m2_stage_loss else (l_ct if not (use_s1_stage_loss or use_s2_stage_loss) else l_ct_stage4)
    l_ct_s5 = l_ct_stage5 if use_m2_stage_loss else zero_ct

    if trajectory_only:
        l_total = (
            l_tool
            + l_ct
            + float(traj_heatmap_ce_weight) * (l_tool_heatmap_ce + l_ct_heatmap_ce)
            + float(traj_centroid_aux_weight) * l_tool_centroid_aux
            + float(traj_offset_aux_weight) * l_tool_offset_aux
            + float(symmetry_pair_separation_weight) * l_tool_pair_sep_aux
        )
    elif tool_only:
        l_total = float(combo_tool_ce_weight) * tool_cls_ps
    elif range_only:
        l_total = range_ps
    elif combo_only:
        l_total = (
            float(combo_ce_weight) * l_combo
            + float(combo_tool_ce_weight) * tool_cls_ps
            + float(combo_action_ce_weight) * action_cls_ps
        )
    else:
        l_total = (
            l_tool
            + l_ct
            + float(combo_ce_weight) * l_combo
            + range_ps
        )
    out = {
        "loss_total": l_total,
        "loss_tool_xy": l_tool,
        "loss_ct_xy": l_ct,
        "loss_tool_centroid_aux": l_tool_centroid_aux,
        "loss_tool_offset_aux": l_tool_offset_aux,
        "loss_tool_pair_sep_aux": l_tool_pair_sep_aux,
        "loss_tool_heatmap_ce": l_tool_heatmap_ce,
        "loss_ct_heatmap_ce": l_ct_heatmap_ce,
        "loss_tool_xy_stage0": l_tool_s0,
        "loss_ct_xy_stage0": l_ct_s0,
        "loss_tool_xy_stage1": l_tool_s1,
        "loss_ct_xy_stage1": l_ct_s1,
        "loss_tool_xy_stage2": l_tool_s2,
        "loss_ct_xy_stage2": l_ct_s2,
        "loss_tool_xy_stage3": l_tool_s3,
        "loss_ct_xy_stage3": l_ct_s3,
        "loss_tool_xy_stage4": l_tool_s4,
        "loss_ct_xy_stage4": l_ct_s4,
        "loss_tool_xy_stage5": l_tool_s5,
        "loss_ct_xy_stage5": l_ct_s5,
        # Backward-compatible aliases.
        "loss_tool_xy_base": l_tool_s0,
        "loss_ct_xy_base": l_ct_s0,
        "loss_tool_xy_refine": l_tool_final,
        "loss_ct_xy_refine": l_ct_final,
        "loss_tool_xy_refine_total": l_tool_ref_total,
        "loss_ct_xy_refine_total": l_ct_ref_total,
        "loss_tool_xy_refine_stages": l_tool_ref_stages,
        "loss_ct_xy_refine_stages": l_ct_ref_stages,
        "loss_traj_vel": l_traj_vel,
        "loss_traj_jerk": l_traj_jerk,
        "loss_combo_ce": l_combo,
        "loss_combo_tool_ce": tool_cls_ps,
        "loss_combo_action_ce": action_cls_ps,
        "loss_combo_info": l_combo_info,
        "loss_start_bin_ce": start_ce_ps,
        "loss_end_bin_ce": end_ce_ps,
        "loss_range": range_ps,
        "loss_range_cls": l_range_cls,
        "loss_range_reg": l_range_reg,
        "loss_range_boundary_aux": l_range_boundary_aux,
        "loss_range_proposal": l_range_proposal,
        "loss_range_quality": l_range_quality,
        "loss_range_sola": l_range_sola,
        "combo_info_bottleneck_weight": l_tool.new_full(l_tool.shape, combo_info_w),
        "traj_refine_stage_ratio": l_tool.new_full(l_tool.shape, float(traj_refine_stage_ratio)),
        "traj_iter_aux_weight": l_tool.new_full(l_tool.shape, float(traj_iter_aux_weight)),
        "traj_stage_weight_start": l_tool.new_full(l_tool.shape, float(traj_stage_weight_start)),
        "traj_stage_weight_end": l_tool.new_full(l_tool.shape, float(traj_stage_weight_end)),
        "traj_stage0_weight": l_tool.new_full(l_tool.shape, float(stage_w[0].item())),
        "traj_stage1_weight": l_tool.new_full(l_tool.shape, float(stage_w[1].item())),
        "traj_stage2_weight": l_tool.new_full(l_tool.shape, float(stage_w[2].item())),
        "traj_stage3_weight": l_tool.new_full(l_tool.shape, float(stage_w[3].item())),
        "traj_stage4_weight": l_tool.new_full(l_tool.shape, float(stage_w[4].item())),
        "traj_stage5_weight": l_tool.new_full(l_tool.shape, float(stage_w[5].item())),
        "loss_start_bin_ce_scalar": l_start_ce,
        "loss_end_bin_ce_scalar": l_end_ce,
        "loss_range_scalar": l_range,
        "loss_range_boundary_aux_scalar": l_range_boundary_aux.mean() if l_range_boundary_aux.numel() > 0 else l_tool.new_zeros(()),
        "loss_range_proposal_scalar": l_range_proposal.mean() if l_range_proposal.numel() > 0 else l_tool.new_zeros(()),
        "loss_range_quality_scalar": l_range_quality.mean() if l_range_quality.numel() > 0 else l_tool.new_zeros(()),
        "loss_range_sola_scalar": l_range_sola.mean() if l_range_sola.numel() > 0 else l_tool.new_zeros(()),
        "loss_combo_tool_ce_scalar": l_tool_cls,
        "loss_combo_action_ce_scalar": l_action_cls,
    }
    if "pred_action_start_norm" in outputs:
        out["pred_action_start_norm"] = outputs["pred_action_start_norm"]
    if "pred_action_end_norm" in outputs:
        out["pred_action_end_norm"] = outputs["pred_action_end_norm"]
    return out
