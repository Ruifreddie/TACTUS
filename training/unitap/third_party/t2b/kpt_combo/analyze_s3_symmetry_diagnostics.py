import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import (
    SurgKptComboDataset,
    TrainableEMA,
    VJEPATokenCache,
    _derive_tool_action_id_from_combo_id,
    _maybe_use_tool_centroid_gt,
    _model_forward_with_cache,
    _safe_div,
    _to_device,
    align_targets_to_tprime,
    build_model,
    load_trainable_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("S3 symmetry diagnostics: oracle-swap, pair-collapse, GT order audit.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true")
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    return parser.parse_args()


def _combo_to_tool_name(combo_name: str) -> str:
    if "|" not in combo_name:
        return str(combo_name)
    return str(combo_name).split("|", 1)[0]


def _build_runtime_args(ckpt_args: Dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace(**dict(ckpt_args))
    args.device = str(cli_args.device)
    args.batch_size = int(cli_args.batch_size)
    args.num_workers = int(cli_args.num_workers)
    args.max_val_steps = 0
    args.no_vis = True
    args.no_init_eval_vis = True
    args.no_checkpoint_vis = True
    args.run_init_eval_vis = False
    args.run_checkpoint_vis = False
    return args


def _make_loader_and_cache(
    args: argparse.Namespace,
    split: str,
    device: torch.device,
) -> Tuple[DataLoader, VJEPATokenCache, Dict[int, str], bool]:
    ds_cache = SurgKptComboDataset(
        root=args.data_root,
        split=split,
        split_file=(args.train_split_file if split == "train" else args.val_split_file) or None,
        input_size=args.input_size,
        normalize=True,
        load_video=False,
        combo_only_pure=bool(getattr(args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)),
    )
    ds_video = SurgKptComboDataset(
        root=args.data_root,
        split=split,
        split_file=(args.train_split_file if split == "train" else args.val_split_file) or None,
        input_size=args.input_size,
        normalize=True,
        load_video=True,
        combo_only_pure=bool(getattr(args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)),
    )
    cache_root = (
        Path(args.vjepa_cache_dir)
        if str(getattr(args, "vjepa_cache_dir", "")).strip()
        else (Path(args.output_dir) / "vjepa_token_cache")
    )
    token_cache = VJEPATokenCache(
        root=cache_root,
        enabled=bool(getattr(args, "vjepa_cache", True)),
        dtype_name=str(getattr(args, "vjepa_cache_dtype", "fp16")),
    )
    use_cache_only = bool(token_cache.enabled) and token_cache.has_all(split, [str(x) for x in ds_cache.sids])
    ds = ds_cache if use_cache_only else ds_video
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    combo_id_to_name = ds_video.combo_id_to_name or ds_cache.combo_id_to_name
    return loader, token_cache, combo_id_to_name, bool(use_cache_only)


def _load_sid_meta(data_root: Path) -> Dict[str, Dict[str, Any]]:
    meta_path = data_root / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        rows = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sid", "")).strip()
        if not sid:
            continue
        out[sid] = row
    return out


def _resolve_pair_indices(pt_role: Sequence[int]) -> Optional[Tuple[Tuple[int, int], int]]:
    roles = [int(x) for x in pt_role]
    if len(roles) != 3:
        return None
    counts: Dict[int, int] = {}
    for r in roles:
        counts[r] = counts.get(r, 0) + 1
    pair_role = None
    for r, c in counts.items():
        if int(c) == 2:
            pair_role = int(r)
            break
    if pair_role is None:
        return None
    pair_idx = [i for i, r in enumerate(roles) if int(r) == int(pair_role)]
    uniq_idx = [i for i, r in enumerate(roles) if int(r) != int(pair_role)]
    if len(pair_idx) != 2 or len(uniq_idx) != 1:
        return None
    return (int(pair_idx[0]), int(pair_idx[1])), int(uniq_idx[0])


def _pair_pixel_huber_frame_loss(
    pred_pair_xy: torch.Tensor,
    gt_pair_xy: torch.Tensor,
    *,
    pixel_scale: float,
    huber_delta: float,
) -> torch.Tensor:
    diff = (pred_pair_xy - gt_pair_xy) * float(pixel_scale)
    abs_diff = diff.abs()
    delta = float(huber_delta)
    flag = (abs_diff <= delta).to(dtype=diff.dtype)
    huber = flag * (0.5 * diff.pow(2)) + (1.0 - flag) * (delta * (abs_diff - 0.5 * delta))
    return huber.mean(dim=(-1, -2))


def _pair_distance_px(pair_xy: torch.Tensor, scale_px: float) -> torch.Tensor:
    delta = (pair_xy[:, 0, :] - pair_xy[:, 1, :]) * float(scale_px)
    return torch.norm(delta, dim=-1)


def _canonical_pair_sign(
    gt_xy_frame: torch.Tensor,
    *,
    pair_idx: Tuple[int, int],
    uniq_idx: int,
    eps: float = 1e-6,
) -> int:
    p0 = gt_xy_frame[int(pair_idx[0])].float()
    p1 = gt_xy_frame[int(pair_idx[1])].float()
    u = gt_xy_frame[int(uniq_idx)].float()
    m = 0.5 * (p0 + p1)
    main = m - u
    main_norm = float(torch.norm(main, p=2).item())
    if main_norm <= float(eps):
        return 0
    perp = torch.stack((-main[1], main[0]))
    order_score = float(torch.dot(p1 - p0, perp).item())
    if abs(order_score) <= float(eps):
        return 0
    return 1 if order_score > 0.0 else -1


def _empty_swap_meter() -> Dict[str, float]:
    return {
        "valid_frames": 0.0,
        "fixed_sum": 0.0,
        "swap_sum": 0.0,
        "best_sum": 0.0,
        "delta_sum": 0.0,
        "swap_win_frames": 0.0,
    }


def _empty_collapse_meter() -> Dict[str, float]:
    return {
        "valid_frames": 0.0,
        "pred_dist_sum": 0.0,
        "gt_dist_sum": 0.0,
        "pred_lt_4": 0.0,
        "pred_lt_8": 0.0,
        "collapse_4_12": 0.0,
        "collapse_8_16": 0.0,
    }


def _empty_gt_audit_meter() -> Dict[str, float]:
    return {
        "valid_frames": 0.0,
        "canonical_match_frames": 0.0,
        "flip_count": 0.0,
        "transition_count": 0.0,
    }


def _finalize_swap_meter(m: Dict[str, float]) -> Dict[str, float]:
    return {
        "valid_frames": int(m["valid_frames"]),
        "mean_L_fixed": float(_safe_div(m["fixed_sum"], m["valid_frames"])),
        "mean_L_swap": float(_safe_div(m["swap_sum"], m["valid_frames"])),
        "mean_L_best": float(_safe_div(m["best_sum"], m["valid_frames"])),
        "mean_Delta_swap": float(_safe_div(m["delta_sum"], m["valid_frames"])),
        "swap_win_rate": float(_safe_div(m["swap_win_frames"], m["valid_frames"])),
    }


def _finalize_collapse_meter(m: Dict[str, float]) -> Dict[str, float]:
    return {
        "valid_frames": int(m["valid_frames"]),
        "mean_pred_pair_dist_px": float(_safe_div(m["pred_dist_sum"], m["valid_frames"])),
        "mean_gt_pair_dist_px": float(_safe_div(m["gt_dist_sum"], m["valid_frames"])),
        "pred_pair_dist_lt_4_rate": float(_safe_div(m["pred_lt_4"], m["valid_frames"])),
        "pred_pair_dist_lt_8_rate": float(_safe_div(m["pred_lt_8"], m["valid_frames"])),
        "collapse_rate_pred_lt4_gt_gt12": float(_safe_div(m["collapse_4_12"], m["valid_frames"])),
        "collapse_rate_pred_lt8_gt_gt16": float(_safe_div(m["collapse_8_16"], m["valid_frames"])),
    }


def _finalize_gt_audit_meter(m: Dict[str, float]) -> Dict[str, float]:
    return {
        "valid_frames": int(m["valid_frames"]),
        "canonical_match_ratio": float(_safe_div(m["canonical_match_frames"], m["valid_frames"])),
        "clip_flip_rate": float(_safe_div(m["flip_count"], m["transition_count"])),
        "valid_transitions": int(m["transition_count"]),
    }


def _collect_quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def run_analysis(cli_args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(cli_args.checkpoint).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    runtime_args = _build_runtime_args(ckpt["args"], cli_args)
    device = torch.device(runtime_args.device if (runtime_args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    data_root = Path(runtime_args.data_root).expanduser().resolve()
    sid_meta = _load_sid_meta(data_root=data_root)

    tmp_ds = SurgKptComboDataset(
        root=runtime_args.data_root,
        split=cli_args.split,
        split_file=(runtime_args.train_split_file if cli_args.split == "train" else runtime_args.val_split_file) or None,
        input_size=runtime_args.input_size,
        normalize=True,
        load_video=False,
        combo_only_pure=bool(getattr(runtime_args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(runtime_args, "tool_kpt_only_data", False)),
    )
    num_combo_classes = int(tmp_ds.num_combos)
    model = build_model(runtime_args, num_combo_classes=num_combo_classes).to(device)
    load_trainable_state_dict(model, ckpt["modules"])
    model.eval()

    ema = None
    if bool(cli_args.use_ema) and ("ema" in ckpt):
        ema = TrainableEMA(model=model, decay=float(getattr(runtime_args, "ema_decay", 0.999)))
        ema.load_state_dict(ckpt["ema"], model=model)

    loader, token_cache, combo_id_to_name, use_cache_only = _make_loader_and_cache(runtime_args, cli_args.split, device)

    loss_scale_px = float(getattr(runtime_args, "traj_loss_pixel_scale", 256.0))
    eval_scale_px = float(max(int(getattr(runtime_args, "input_size", 256)) - 1, 1))
    huber_delta = float(getattr(runtime_args, "traj_huber_delta", 6.0))

    swap_global = _empty_swap_meter()
    swap_by_tool = defaultdict(_empty_swap_meter)
    swap_by_combo = defaultdict(_empty_swap_meter)

    collapse_global = _empty_collapse_meter()
    collapse_by_tool = defaultdict(_empty_collapse_meter)
    collapse_by_combo = defaultdict(_empty_collapse_meter)
    pred_dist_values: List[float] = []
    gt_dist_values: List[float] = []

    gt_audit_global = _empty_gt_audit_meter()
    gt_audit_by_tool = defaultdict(_empty_gt_audit_meter)
    gt_audit_by_combo = defaultdict(_empty_gt_audit_meter)
    gt_audit_signs_by_sid: Dict[str, List[int]] = defaultdict(list)

    swap_examples: List[Dict[str, Any]] = []
    collapse_examples: List[Dict[str, Any]] = []

    ctx = torch.no_grad()
    with ctx:
        if ema is not None:
            backup = ema.apply_shadow(model)
        else:
            backup = None
        try:
            for batch in loader:
                batch = _to_device(batch, device)
                video = batch.get("video", None)
                gt_tool_xy = batch["gt_tool_xy"]
                gt_tool_vis = batch["gt_tool_vis"]
                gt_tool_xy, gt_tool_vis = _maybe_use_tool_centroid_gt(
                    gt_tool_xy=gt_tool_xy,
                    gt_tool_vis=gt_tool_vis,
                    use_tool_centroid_only=bool(getattr(runtime_args, "tool_centroid_only", False)),
                )
                gt_ct_xy = batch["gt_ct_xy"]
                gt_ct_vis = batch["gt_ct_vis"]
                combo_id = batch["combo_id"].long()
                tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)
                sid_raw = batch.get("sid", [])
                if isinstance(sid_raw, (list, tuple)):
                    sid_list = [str(x) for x in sid_raw]
                elif isinstance(sid_raw, str):
                    sid_list = [sid_raw]
                else:
                    sid_list = [str(sid_raw)]

                outputs = _model_forward_with_cache(
                    model=model,
                    video=video,
                    sids=sid_list,
                    split=cli_args.split,
                    token_cache=token_cache,
                    tool_id=tool_id,
                    action_id=action_id,
                    combo_id=combo_id,
                )
                t_prime = int(outputs["pred_tool_xy"].shape[1])
                aligned = align_targets_to_tprime(
                    gt_tool_xy,
                    gt_tool_vis,
                    gt_ct_xy,
                    gt_ct_vis,
                    t_prime=t_prime,
                )

                pred_tool_xy = outputs["pred_tool_xy"].float()
                gt_tool_xy_aligned = aligned["gt_tool_xy"].float()
                gt_tool_vis_aligned = aligned["gt_tool_vis"].float()

                for i, sid in enumerate(sid_list):
                    meta_row = sid_meta.get(str(sid), {})
                    pt_role = meta_row.get("pt_role", None)
                    pair_spec = _resolve_pair_indices(pt_role if isinstance(pt_role, list) else [])
                    if pair_spec is None:
                        continue
                    pair_idx, uniq_idx = pair_spec
                    combo_name = combo_id_to_name.get(int(combo_id[i].item()), f"combo_{int(combo_id[i].item())}")
                    tool_name = _combo_to_tool_name(combo_name)

                    pred_pair = pred_tool_xy[i, :, list(pair_idx), :]
                    gt_pair = gt_tool_xy_aligned[i, :, list(pair_idx), :]
                    pair_vis = gt_tool_vis_aligned[i, :, list(pair_idx)]
                    valid_pair = (pair_vis > 0.5).all(dim=-1)

                    if bool(valid_pair.any().item()):
                        fixed_loss = _pair_pixel_huber_frame_loss(
                            pred_pair_xy=pred_pair,
                            gt_pair_xy=gt_pair,
                            pixel_scale=loss_scale_px,
                            huber_delta=huber_delta,
                        )
                        swap_loss = _pair_pixel_huber_frame_loss(
                            pred_pair_xy=pred_pair,
                            gt_pair_xy=gt_pair[:, [1, 0], :],
                            pixel_scale=loss_scale_px,
                            huber_delta=huber_delta,
                        )
                        pred_pair_dist = _pair_distance_px(pred_pair, scale_px=eval_scale_px)
                        gt_pair_dist = _pair_distance_px(gt_pair, scale_px=eval_scale_px)

                        valid_idx = torch.nonzero(valid_pair, as_tuple=False).flatten().tolist()
                        for t in valid_idx:
                            fixed_val = float(fixed_loss[t].item())
                            swap_val = float(swap_loss[t].item())
                            best_val = min(fixed_val, swap_val)
                            delta_val = float(fixed_val - best_val)
                            pred_dist = float(pred_pair_dist[t].item())
                            gt_dist = float(gt_pair_dist[t].item())
                            swap_win = float(swap_val < fixed_val)

                            for meter in (swap_global, swap_by_tool[tool_name], swap_by_combo[combo_name]):
                                meter["valid_frames"] += 1.0
                                meter["fixed_sum"] += fixed_val
                                meter["swap_sum"] += swap_val
                                meter["best_sum"] += best_val
                                meter["delta_sum"] += delta_val
                                meter["swap_win_frames"] += swap_win

                            for meter in (collapse_global, collapse_by_tool[tool_name], collapse_by_combo[combo_name]):
                                meter["valid_frames"] += 1.0
                                meter["pred_dist_sum"] += pred_dist
                                meter["gt_dist_sum"] += gt_dist
                                meter["pred_lt_4"] += float(pred_dist < 4.0)
                                meter["pred_lt_8"] += float(pred_dist < 8.0)
                                meter["collapse_4_12"] += float((pred_dist < 4.0) and (gt_dist > 12.0))
                                meter["collapse_8_16"] += float((pred_dist < 8.0) and (gt_dist > 16.0))

                            pred_dist_values.append(pred_dist)
                            gt_dist_values.append(gt_dist)

                            if delta_val > 0.0:
                                swap_examples.append(
                                    {
                                        "sid": str(sid),
                                        "frame_idx": int(t),
                                        "combo_name": str(combo_name),
                                        "tool_name": str(tool_name),
                                        "fixed_loss": fixed_val,
                                        "swap_loss": swap_val,
                                        "delta_swap": delta_val,
                                    }
                                )
                            collapse_gap = float(gt_dist - pred_dist)
                            if (pred_dist < 8.0) and (gt_dist > 16.0):
                                collapse_examples.append(
                                    {
                                        "sid": str(sid),
                                        "frame_idx": int(t),
                                        "combo_name": str(combo_name),
                                        "tool_name": str(tool_name),
                                        "pred_pair_dist_px": pred_dist,
                                        "gt_pair_dist_px": gt_dist,
                                        "collapse_gap_px": collapse_gap,
                                    }
                                )

                    gt_frame_vis = gt_tool_vis_aligned[i]
                    gt_frame_xy = gt_tool_xy_aligned[i]
                    pair_vis_for_audit = (gt_frame_vis[:, list(pair_idx)] > 0.5).all(dim=-1)
                    uniq_vis_for_audit = gt_frame_vis[:, int(uniq_idx)] > 0.5
                    valid_gt_audit = pair_vis_for_audit & uniq_vis_for_audit
                    signs: List[int] = []
                    for t in torch.nonzero(valid_gt_audit, as_tuple=False).flatten().tolist():
                        sign = _canonical_pair_sign(
                            gt_xy_frame=gt_frame_xy[t],
                            pair_idx=pair_idx,
                            uniq_idx=uniq_idx,
                        )
                        if int(sign) == 0:
                            continue
                        match = float(sign > 0)
                        signs.append(int(sign))
                        for meter in (gt_audit_global, gt_audit_by_tool[tool_name], gt_audit_by_combo[combo_name]):
                            meter["valid_frames"] += 1.0
                            meter["canonical_match_frames"] += match
                    if signs:
                        gt_audit_signs_by_sid[str(sid)].extend(signs)
                        for prev, cur in zip(signs[:-1], signs[1:]):
                            flip = float(prev != cur)
                            for meter in (gt_audit_global, gt_audit_by_tool[tool_name], gt_audit_by_combo[combo_name]):
                                meter["flip_count"] += flip
                                meter["transition_count"] += 1.0
        finally:
            if (ema is not None) and (backup is not None):
                ema.restore(model, backup)

    swap_examples = sorted(swap_examples, key=lambda x: (-float(x["delta_swap"]), x["sid"], x["frame_idx"]))[: int(cli_args.top_k)]
    collapse_examples = sorted(
        collapse_examples,
        key=lambda x: (-float(x["collapse_gap_px"]), x["sid"], x["frame_idx"]),
    )[: int(cli_args.top_k)]

    result = {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)),
        "split": str(cli_args.split),
        "device": str(device),
        "used_ema": bool(cli_args.use_ema and ("ema" in ckpt)),
        "use_cache_only": bool(use_cache_only),
        "diagnostic_contract": {
            "oracle_swap_pair_loss": {
                "type": "pair_only_pixel_huber",
                "pixel_scale": float(loss_scale_px),
                "huber_delta_px": float(huber_delta),
                "valid_frame_rule": "both symmetric points visible",
            },
            "pair_collapse": {
                "distance_scale_px": float(eval_scale_px),
                "valid_frame_rule": "both symmetric points visible",
                "primary_thresholds": {
                    "pred_lt4_gt_gt12": {"pred_lt_px": 4.0, "gt_gt_px": 12.0},
                    "pred_lt8_gt_gt16": {"pred_lt_px": 8.0, "gt_gt_px": 16.0},
                },
            },
            "gt_order_audit": {
                "rule": "canonical perpendicular-axis ordering on the symmetric pair",
                "valid_frame_rule": "all 3 points visible and non-degenerate canonical axis",
                "flip_rate_scope": "within each 16-frame sid clip",
            },
        },
        "oracle_swap": {
            "summary": _finalize_swap_meter(swap_global),
            "by_tool": {k: _finalize_swap_meter(v) for k, v in sorted(swap_by_tool.items())},
            "by_combo": {k: _finalize_swap_meter(v) for k, v in sorted(swap_by_combo.items())},
            "top_delta_examples": swap_examples,
        },
        "pair_collapse": {
            "summary": {
                **_finalize_collapse_meter(collapse_global),
                "pred_pair_dist_quantiles_px": _collect_quantiles(pred_dist_values),
                "gt_pair_dist_quantiles_px": _collect_quantiles(gt_dist_values),
            },
            "by_tool": {k: _finalize_collapse_meter(v) for k, v in sorted(collapse_by_tool.items())},
            "by_combo": {k: _finalize_collapse_meter(v) for k, v in sorted(collapse_by_combo.items())},
            "top_collapse_examples": collapse_examples,
        },
        "gt_order_audit": {
            "summary": _finalize_gt_audit_meter(gt_audit_global),
            "by_tool": {k: _finalize_gt_audit_meter(v) for k, v in sorted(gt_audit_by_tool.items())},
            "by_combo": {k: _finalize_gt_audit_meter(v) for k, v in sorted(gt_audit_by_combo.items())},
            "clips_with_valid_frames": int(sum(1 for _, signs in gt_audit_signs_by_sid.items() if len(signs) > 0)),
        },
    }
    return result


def _default_output_json(cli_args: argparse.Namespace) -> Path:
    ckpt_path = Path(cli_args.checkpoint).expanduser().resolve()
    return ckpt_path.parent / f"s3_symmetry_diagnostics_{cli_args.split}.json"


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    out_json = Path(args.output_json).expanduser().resolve() if str(args.output_json).strip() else _default_output_json(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    swap_summary = result["oracle_swap"]["summary"]
    collapse_summary = result["pair_collapse"]["summary"]
    gt_summary = result["gt_order_audit"]["summary"]

    print(f"[s3_symdiag] saved: {out_json}")
    print(
        "[s3_symdiag] oracle-swap "
        f"valid_frames={swap_summary['valid_frames']} "
        f"mean_delta={swap_summary['mean_Delta_swap']:.4f} "
        f"swap_win_rate={swap_summary['swap_win_rate']:.4f}"
    )
    print(
        "[s3_symdiag] pair-collapse "
        f"valid_frames={collapse_summary['valid_frames']} "
        f"pred_lt4_gt_gt12={collapse_summary['collapse_rate_pred_lt4_gt_gt12']:.4f} "
        f"pred_lt8_gt_gt16={collapse_summary['collapse_rate_pred_lt8_gt_gt16']:.4f}"
    )
    print(
        "[s3_symdiag] gt-order "
        f"valid_frames={gt_summary['valid_frames']} "
        f"canonical_match_ratio={gt_summary['canonical_match_ratio']:.4f} "
        f"clip_flip_rate={gt_summary['clip_flip_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
