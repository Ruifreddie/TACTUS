import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from dataset import SurgKptComboDataset
    from model import KeypointComboNet
    from vjepa_wrapper import VJEPAWrapper
else:
    from .dataset import SurgKptComboDataset
    from .model import KeypointComboNet
    from .vjepa_wrapper import VJEPAWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inference for combo + keypoint + action-range-bin model with export")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="work_dirs/kpt_combo/infer")

    parser.add_argument("--vjepa-repo", type=str, default="vjepa2")
    parser.add_argument("--vjepa-arch", type=str, default="")
    parser.add_argument("--vjepa-hf-repo", type=str, default="")
    parser.add_argument("--vjepa-hf-local-files-only", action="store_true")
    parser.add_argument("--vjepa-checkpoint", type=str, default="")
    parser.add_argument("--vjepa-pretrained-hub", action="store_true")

    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=0, help="<=0 means use checkpoint setting")
    parser.add_argument(
        "--temporal-upsample",
        type=str,
        default="",
        choices=["", "interp", "deconv"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-condition-mode",
        type=str,
        default="",
        choices=["", "tool", "tool_action", "combo"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-head-mode",
        type=str,
        default="",
        choices=["", "scalar", "trident"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-proposal-mode",
        type=str,
        default="",
        choices=["", "none", "bmn"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-proposal-decode-mode",
        type=str,
        default="",
        choices=["", "argmax", "none"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-quality-mode",
        type=str,
        default="",
        choices=["", "none", "dgqp"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-boundary-refine-mode",
        type=str,
        default="",
        choices=["", "none", "local_argmax"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-boundary-refine-radius",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-temporal-context-mode",
        type=str,
        default="",
        choices=["", "none", "temporalmaxer", "fpn"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-temporal-context-kernel-size",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-temporal-context-layers",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-temporal-context-levels",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-spatial-pool-mode",
        type=str,
        default="",
        choices=["", "mean_max", "mean_max_condtoken", "mean_max_condtoken_locresid"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-spatial-pool-num-tokens",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-adapter-mode",
        type=str,
        default="",
        choices=["", "none", "sola", "losa", "dyfa", "sola_dyfa", "sola_dyfa_livegate", "sola_dyfa_laf"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-adapter-kernel-size",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-adapter-num-heads",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-task-decouple-mode",
        type=str,
        default="",
        choices=["", "none", "score_loc", "score_loc_align"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-duration-prior-mode",
        type=str,
        default="",
        choices=["", "none", "combo_gaussian"],
        help="empty=use checkpoint setting",
    )
    parser.add_argument(
        "--range-duration-prior-weight",
        type=float,
        default=-1.0,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-duration-prior-min-std",
        type=float,
        default=-1.0,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument(
        "--range-trident-num-bins",
        type=int,
        default=-1,
        help="<0 means use checkpoint setting",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--repeat-to-16f", action="store_true")

    return parser.parse_args()


def _get_amp_dtype(amp_dtype: str) -> torch.dtype:
    return torch.float16 if amp_dtype == "fp16" else torch.bfloat16


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _repeat_time_to_target(x: torch.Tensor, target_t: int = 16) -> torch.Tensor:
    # x first dim is temporal
    if x.shape[0] == target_t:
        return x
    if x.shape[0] <= 0:
        return x
    if target_t % x.shape[0] != 0:
        # Fallback: nearest interpolation on temporal axis
        idx = torch.linspace(0, x.shape[0] - 1, steps=target_t, device=x.device).round().long()
        return x.index_select(0, idx)
    rep = target_t // x.shape[0]
    return x.repeat_interleave(rep, dim=0)


def _bins_to_abs(
    start_bin: int,
    end_bin: int,
    num_bins: int,
    sampled_abs_frames: np.ndarray,
) -> tuple[float, float, int, int]:
    n = max(int(num_bins), 1)
    s_bin = int(np.clip(int(start_bin), 0, n - 1))
    e_bin = int(np.clip(int(end_bin), 0, n - 1))
    if e_bin < s_bin:
        e_bin = s_bin
    s_norm = (float(s_bin) + 0.5) / float(n)
    e_norm = (float(e_bin) + 0.5) / float(n)

    fr = np.asarray(sampled_abs_frames, dtype=np.float64).reshape(-1)
    if fr.size < 2:
        return s_norm, e_norm, 0, 0
    clip_s = int(round(float(fr[0])))
    clip_e = int(round(float(fr[-1])))
    span = float(clip_e - clip_s)
    if span <= 0.0:
        return s_norm, e_norm, clip_s, clip_s

    s_abs = int(round(float(clip_s) + s_norm * span))
    e_abs = int(round(float(clip_s) + e_norm * span))
    if e_abs < s_abs:
        e_abs = s_abs
    return s_norm, e_norm, s_abs, e_abs


def _derive_tool_action_id_from_combo_id(combo_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    combo = combo_id.long()
    combo_to_tool = torch.tensor([0, 1, 2, 3, 3, 1], device=combo.device, dtype=torch.long)
    combo_to_action = torch.tensor([0, 1, 2, 3, 0, 0], device=combo.device, dtype=torch.long)
    if bool((combo < 0).any().item()) or bool((combo >= int(combo_to_tool.shape[0])).any().item()):
        raise ValueError(
            f"`combo_id` out of mapping range [0,{int(combo_to_tool.shape[0]) - 1}] "
            f"(min={int(combo.min().item())}, max={int(combo.max().item())})"
        )
    return combo_to_tool[combo], combo_to_action[combo]


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    ckpt_args = ckpt.get("args", {})
    ckpt_vjepa = ckpt.get("vjepa", {})

    vjepa_arch = args.vjepa_arch or ckpt_vjepa.get("arch") or ckpt_args.get("vjepa_arch", "vit_large")
    vjepa_hf_repo = args.vjepa_hf_repo or ckpt_vjepa.get("hf_repo", "") or ckpt_args.get("vjepa_hf_repo", "")
    vjepa_ckpt = args.vjepa_checkpoint or ckpt_vjepa.get("checkpoint") or ckpt_args.get("vjepa_checkpoint", "")
    vjepa_pretrained_hub = bool(
        args.vjepa_pretrained_hub
        or ckpt_vjepa.get("pretrained_hub", False)
        or ckpt_args.get("vjepa_pretrained_hub", False)
    )
    input_size = int(args.input_size if args.input_size > 0 else ckpt_args.get("input_size", 256))
    proj_dim = int(args.proj_dim if args.proj_dim > 0 else ckpt_args.get("proj_dim", 256))
    temporal_upsample = str(args.temporal_upsample or ckpt_args.get("temporal_upsample", "interp"))
    num_range_bins = int(ckpt_args.get("range_num_bins", 10))
    range_condition_mode = str(args.range_condition_mode or ckpt_args.get("range_condition_mode", "tool_action"))
    range_head_mode = str(args.range_head_mode or ckpt_args.get("range_head_mode", "scalar"))
    range_proposal_mode = str(args.range_proposal_mode or ckpt_args.get("range_proposal_mode", "none"))
    range_proposal_decode_mode = str(
        args.range_proposal_decode_mode or ckpt_args.get("range_proposal_decode_mode", "argmax")
    )
    range_quality_mode = str(args.range_quality_mode or ckpt_args.get("range_quality_mode", "none"))
    range_boundary_refine_mode = str(
        args.range_boundary_refine_mode or ckpt_args.get("range_boundary_refine_mode", "none")
    )
    range_boundary_refine_radius = int(
        args.range_boundary_refine_radius
        if int(args.range_boundary_refine_radius) > 0
        else ckpt_args.get("range_boundary_refine_radius", 2)
    )
    range_temporal_context_mode = str(
        args.range_temporal_context_mode or ckpt_args.get("range_temporal_context_mode", "none")
    )
    range_temporal_context_kernel_size = int(
        args.range_temporal_context_kernel_size
        if int(args.range_temporal_context_kernel_size) > 0
        else ckpt_args.get("range_temporal_context_kernel_size", 3)
    )
    range_temporal_context_layers = int(
        args.range_temporal_context_layers
        if int(args.range_temporal_context_layers) > 0
        else ckpt_args.get("range_temporal_context_layers", 1)
    )
    range_temporal_context_levels = int(
        args.range_temporal_context_levels
        if int(args.range_temporal_context_levels) > 0
        else ckpt_args.get("range_temporal_context_levels", 3)
    )
    range_spatial_pool_mode = str(
        args.range_spatial_pool_mode or ckpt_args.get("range_spatial_pool_mode", "mean_max")
    )
    range_spatial_pool_num_tokens = int(
        args.range_spatial_pool_num_tokens
        if int(args.range_spatial_pool_num_tokens) > 0
        else ckpt_args.get("range_spatial_pool_num_tokens", 8)
    )
    range_adapter_mode = str(args.range_adapter_mode or ckpt_args.get("range_adapter_mode", "none"))
    range_adapter_kernel_size = int(
        args.range_adapter_kernel_size
        if int(args.range_adapter_kernel_size) > 0
        else ckpt_args.get("range_adapter_kernel_size", 5)
    )
    range_adapter_num_heads = int(
        args.range_adapter_num_heads
        if int(args.range_adapter_num_heads) > 0
        else ckpt_args.get("range_adapter_num_heads", 4)
    )
    range_task_decouple_mode = str(
        args.range_task_decouple_mode or ckpt_args.get("range_task_decouple_mode", "none")
    ).lower()
    range_duration_prior_mode = str(
        args.range_duration_prior_mode or ckpt_args.get("range_duration_prior_mode", "none")
    ).lower()
    range_duration_prior_weight = float(
        args.range_duration_prior_weight
        if float(args.range_duration_prior_weight) >= 0.0
        else ckpt_args.get("range_duration_prior_weight", 0.0)
    )
    range_duration_prior_min_std = float(
        args.range_duration_prior_min_std
        if float(args.range_duration_prior_min_std) > 0.0
        else ckpt_args.get("range_duration_prior_min_std", 0.05)
    )
    range_duration_prior_means = ckpt_args.get("range_duration_prior_means", None)
    range_duration_prior_stds = ckpt_args.get("range_duration_prior_stds", None)
    range_trident_num_bins = int(
        args.range_trident_num_bins if int(args.range_trident_num_bins) > 0 else ckpt_args.get("range_trident_num_bins", 15)
    )
    range_only = bool(ckpt_args.get("range_only", False))
    tool_only = bool(ckpt_args.get("tool_only", False))
    combo_probe_attn = bool(ckpt_args.get("combo_probe_attn", False))
    combo_probe_dual_attn = bool(ckpt_args.get("combo_probe_dual_attn", False))
    dual_attn_tool_prior = bool(ckpt_args.get("dual_attn_tool_prior", False))
    dual_attn_tool_prior_init_alpha = float(ckpt_args.get("dual_attn_tool_prior_init_alpha", 0.0))
    dual_attn_tool_prior_detach = bool(ckpt_args.get("dual_attn_tool_prior_detach", True))

    if vjepa_hf_repo and (vjepa_ckpt or vjepa_pretrained_hub):
        raise ValueError("`--vjepa-hf-repo` cannot be combined with checkpoint/hub pretrained flags.")

    ds = SurgKptComboDataset(root=args.data_root, split=args.split, input_size=input_size, normalize=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_combo_classes = ds.num_combos
    vjepa = VJEPAWrapper(
        vjepa_repo=args.vjepa_repo,
        arch=vjepa_arch,
        img_size=input_size,
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        hf_repo=vjepa_hf_repo,
        hf_local_files_only=args.vjepa_hf_local_files_only,
        pretrained_from_hub=vjepa_pretrained_hub,
        checkpoint_path=vjepa_ckpt,
        freeze=True,
    )
    model = KeypointComboNet(
        vjepa=vjepa,
        proj_dim=proj_dim,
        num_combo_classes=num_combo_classes,
        num_range_bins=num_range_bins,
        temporal_upsample=temporal_upsample,
        range_only=range_only,
        tool_only=tool_only,
        combo_probe_attn=combo_probe_attn,
        combo_probe_dual_attn=combo_probe_dual_attn,
        dual_attn_tool_prior=dual_attn_tool_prior,
        dual_attn_tool_prior_init_alpha=dual_attn_tool_prior_init_alpha,
        dual_attn_tool_prior_detach=dual_attn_tool_prior_detach,
        range_condition_mode=range_condition_mode,
        range_head_mode=range_head_mode,
        range_trident_num_bins=range_trident_num_bins,
        range_proposal_mode=range_proposal_mode,
        range_proposal_decode_mode=range_proposal_decode_mode,
        range_quality_mode=range_quality_mode,
        range_boundary_refine_mode=range_boundary_refine_mode,
        range_boundary_refine_radius=range_boundary_refine_radius,
        range_temporal_context_mode=range_temporal_context_mode,
        range_temporal_context_kernel_size=range_temporal_context_kernel_size,
        range_temporal_context_layers=range_temporal_context_layers,
        range_temporal_context_levels=range_temporal_context_levels,
        range_spatial_pool_mode=range_spatial_pool_mode,
        range_spatial_pool_num_tokens=range_spatial_pool_num_tokens,
        range_adapter_mode=range_adapter_mode,
        range_adapter_kernel_size=range_adapter_kernel_size,
        range_adapter_num_heads=range_adapter_num_heads,
        range_task_decouple_mode=range_task_decouple_mode,
        range_duration_prior_mode=range_duration_prior_mode,
        range_duration_prior_weight=range_duration_prior_weight,
        range_duration_prior_means=range_duration_prior_means,
        range_duration_prior_stds=range_duration_prior_stds,
        range_duration_prior_min_std=range_duration_prior_min_std,
    ).to(device)
    model.eval()

    modules = ckpt.get("modules", {})
    for name, module in model.trainable_modules.items():
        if name in modules:
            module.load_state_dict(modules[name], strict=True)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = _get_amp_dtype(args.amp_dtype)

    exported = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            sids = batch["sid"]
            video = batch["video"]
            combo_id = batch["combo_id"].long()
            tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(video, tool_id=tool_id, action_id=action_id, combo_id=combo_id)

            combo_prob = torch.softmax(outputs["combo_logits"], dim=-1)
            pred_combo = outputs["combo_logits"].argmax(dim=-1)

            for i, sid in enumerate(sids):
                sample_dir = out_root / sid
                sample_dir.mkdir(parents=True, exist_ok=True)

                tool_xy = outputs["pred_tool_xy"][i].detach().cpu()
                ct_xy = outputs["pred_ct_xy"][i].detach().cpu()
                tool_hm = outputs["heatmap_tool"][i].detach().cpu()
                ct_hm = outputs["heatmap_ct"][i].detach().cpu()
                conf_tool = outputs["conf_tool"][i].detach().cpu()
                conf_ct = outputs["conf_ct"][i].detach().cpu()
                start_logits = outputs["logits_start_bin"][i].detach().cpu()
                end_logits = outputs["logits_end_bin"][i].detach().cpu()
                sampled_abs = batch["sampled_abs_frames"][i].detach().cpu().numpy().astype(np.int64)
                num_bins = int(start_logits.shape[-1])
                pred_start_bin = int(start_logits.argmax(dim=-1).item())
                pred_end_bin = int(end_logits.argmax(dim=-1).item())
                pred_end_bin = max(pred_end_bin, pred_start_bin)
                pred_start_norm, pred_end_norm, pred_start_abs, pred_end_abs = _bins_to_abs(
                    start_bin=pred_start_bin,
                    end_bin=pred_end_bin,
                    num_bins=num_bins,
                    sampled_abs_frames=sampled_abs,
                )

                if args.repeat_to_16f:
                    tool_xy = _repeat_time_to_target(tool_xy, 16)
                    ct_xy = _repeat_time_to_target(ct_xy, 16)
                    tool_hm = _repeat_time_to_target(tool_hm, 16)
                    ct_hm = _repeat_time_to_target(ct_hm, 16)
                    conf_tool = _repeat_time_to_target(conf_tool, 16)
                    conf_ct = _repeat_time_to_target(conf_ct, 16)

                np.save(sample_dir / "pred_tool_xy.npy", tool_xy.numpy().astype(np.float32))
                np.save(sample_dir / "pred_ct_xy.npy", ct_xy.numpy().astype(np.float32))
                np.save(sample_dir / "tool_heatmap.npy", tool_hm.numpy().astype(np.float32))
                np.save(sample_dir / "ct_heatmap.npy", ct_hm.numpy().astype(np.float32))
                np.save(sample_dir / "conf_tool.npy", conf_tool.numpy().astype(np.float32))
                np.save(sample_dir / "conf_ct.npy", conf_ct.numpy().astype(np.float32))
                np.save(sample_dir / "combo_logits.npy", outputs["combo_logits"][i].detach().cpu().numpy().astype(np.float32))
                np.save(sample_dir / "combo_probs.npy", combo_prob[i].detach().cpu().numpy().astype(np.float32))
                np.save(sample_dir / "pred_combo_id.npy", np.array(int(pred_combo[i].item()), dtype=np.int64))
                np.save(sample_dir / "start_logits_bin.npy", start_logits.numpy().astype(np.float32))
                np.save(sample_dir / "end_logits_bin.npy", end_logits.numpy().astype(np.float32))
                np.save(sample_dir / "pred_start_bin.npy", np.array(pred_start_bin, dtype=np.int64))
                np.save(sample_dir / "pred_end_bin.npy", np.array(pred_end_bin, dtype=np.int64))

                meta = {
                    "sid": sid,
                    "split": args.split,
                    "pred_combo_id": int(pred_combo[i].item()),
                    "num_range_bins": num_bins,
                    "pred_start_bin": pred_start_bin,
                    "pred_end_bin": pred_end_bin,
                    "pred_start_norm": pred_start_norm,
                    "pred_end_norm": pred_end_norm,
                    "pred_action_start_abs": pred_start_abs,
                    "pred_action_end_abs": pred_end_abs,
                    "clip_start_abs": int(sampled_abs[0]) if sampled_abs.size > 0 else 0,
                    "clip_end_abs": int(sampled_abs[-1]) if sampled_abs.size > 0 else 0,
                    "t_prime": int(outputs["pred_tool_xy"].shape[1]),
                    "tool_heatmap_shape": list(tool_hm.shape),
                    "ct_heatmap_shape": list(ct_hm.shape),
                }
                if "gt_action_start_bin" in batch and "gt_action_end_bin" in batch and "action_range_mask" in batch:
                    meta["gt_start_bin"] = int(batch["gt_action_start_bin"][i].detach().cpu().item())
                    meta["gt_end_bin"] = int(batch["gt_action_end_bin"][i].detach().cpu().item())
                    meta["gt_action_range_mask"] = float(batch["action_range_mask"][i].detach().cpu().item())
                (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

                exported += 1
                if args.max_samples > 0 and exported >= args.max_samples:
                    print(f"[done] exported {exported} samples to {out_root}")
                    return

    print(f"[done] exported {exported} samples to {out_root}")


if __name__ == "__main__":
    main()
