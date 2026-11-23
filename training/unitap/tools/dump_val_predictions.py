#!/usr/bin/env python
"""Dump per-clip predictions of a T2b checkpoint on a split (cache-only forward, EMA weights) for offline failure analysis.
Writes <out>/<sid>.npz with: pred_xy [T',P,2] (final), stage_xy [K,T',P,2], conf [T',P], gt_xy [T',P,2], gt_vis [T',P],
and any spatial prob map the model exposes (probs_tool) if it is a tensor."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--split", default="val")
ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--num-workers", type=int, default=2); ap.add_argument("--use-ema", type=int, default=1)
cli = ap.parse_args()
KC = Path(__file__).resolve().parent.parent / "third_party/t2b/kpt_combo"; sys.path.insert(0, str(KC)); sys.path.insert(0, str(KC.parent))
from train import SurgKptComboDataset, TrainableEMA, _derive_tool_action_id_from_combo_id, _maybe_use_tool_centroid_gt, _model_forward_with_cache, _to_device, align_targets_to_tprime, build_model, load_trainable_state_dict
from analyze_s3_symmetry_diagnostics import _build_runtime_args, _make_loader_and_cache
ckpt = torch.load(cli.checkpoint, map_location="cpu"); args = _build_runtime_args(ckpt["args"], cli)
device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
tmp = SurgKptComboDataset(root=args.data_root, split=cli.split, split_file=(args.train_split_file if cli.split == "train" else args.val_split_file) or None, input_size=args.input_size, normalize=True, load_video=False, combo_only_pure=bool(getattr(args, "combo_only_pure", False)), tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)))
model = build_model(args, num_combo_classes=int(tmp.num_combos)).to(device); load_trainable_state_dict(model, ckpt["modules"]); model.eval()
ema = None
if cli.use_ema and "ema" in ckpt:
    ema = TrainableEMA(model=model, decay=float(getattr(args, "ema_decay", 0.999))); ema.load_state_dict(ckpt["ema"], model=model)
def _loader_for_split(args, split, device):
    """Like analyze_s3_symmetry_diagnostics._make_loader_and_cache but with a proper split file for 'test'."""
    from torch.utils.data import DataLoader
    from train import VJEPATokenCache
    root = Path(args.data_root)
    split_file = {"train": args.train_split_file, "val": args.val_split_file}.get(split) or str(root / f"{split}.txt")
    ds = SurgKptComboDataset(root=args.data_root, split=split, split_file=split_file, input_size=args.input_size, normalize=True, load_video=False,
                             combo_only_pure=bool(getattr(args, "combo_only_pure", False)), tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)))
    cache_root = Path(args.vjepa_cache_dir) if str(getattr(args, "vjepa_cache_dir", "")).strip() else (Path(args.output_dir) / "vjepa_token_cache")
    tc = VJEPATokenCache(root=cache_root, enabled=True, dtype_name=str(getattr(args, "vjepa_cache_dtype", "fp16")))
    if not tc.has_all(split, [str(x) for x in ds.sids]):
        raise RuntimeError(f"token cache incomplete for split={split} under {cache_root}")
    return DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=False, drop_last=False), tc, ds.combo_id_to_name, True
loader, token_cache, combo_id_to_name, use_cache_only = _loader_for_split(args, cli.split, device)
print(f"[dump] ckpt epoch={ckpt.get('epoch')} cache_only={use_cache_only} n={len(loader.dataset)} device={device}", flush=True)
out = Path(cli.out); out.mkdir(parents=True, exist_ok=True); n = 0; keys_seen = None
with torch.no_grad():
    backup = ema.apply_shadow(model) if ema is not None else None
    try:
        for batch in loader:
            batch = _to_device(batch, device); video = batch.get("video", None)
            gt_xy, gt_vis = _maybe_use_tool_centroid_gt(gt_tool_xy=batch["gt_tool_xy"], gt_tool_vis=batch["gt_tool_vis"], use_tool_centroid_only=bool(getattr(args, "tool_centroid_only", False)))
            combo_id = batch["combo_id"].long(); tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)
            sids = [str(x) for x in batch.get("sid", [])]
            o = _model_forward_with_cache(model=model, video=video, sids=sids, split=cli.split, token_cache=token_cache, tool_id=tool_id, action_id=action_id, combo_id=combo_id)
            t_prime = int(o["pred_tool_xy"].shape[1])
            al = align_targets_to_tprime(gt_xy, gt_vis, batch["gt_ct_xy"], batch["gt_ct_vis"], t_prime)
            a_xy, a_vis = (al[0], al[1]) if isinstance(al, (tuple, list)) else (al["gt_tool_xy"], al["gt_tool_vis"])
            if keys_seen is None:
                keys_seen = {k: tuple(v.shape) for k, v in o.items() if torch.is_tensor(v)}; print("[dump] output tensors:", {k: v for k, v in keys_seen.items() if any(s in k for s in ("tool", "probs", "heatmap", "conf"))}, flush=True)
            for i, sid in enumerate(sids):
                conf_src = torch.sigmoid(o["pred_conf_logit"][i].float()) if torch.is_tensor(o.get("pred_conf_logit")) else (o["conf_tool"][i].float() if torch.is_tensor(o.get("conf_tool")) else None)
                d = {"pred_xy": o["pred_tool_xy"][i].float().cpu().numpy(), "conf": conf_src.cpu().numpy() if conf_src is not None else np.zeros(0),
                     "gt_xy": a_xy[i].float().cpu().numpy(), "gt_vis": a_vis[i].float().cpu().numpy(), "t_prime": t_prime}
                if torch.is_tensor(o.get("pred_tool_xy_stage_stack")): d["stage_xy"] = o["pred_tool_xy_stage_stack"][:, i].float().cpu().numpy()
                if torch.is_tensor(o.get("pred_tool_xy_stage0")): d["stage0_xy"] = o["pred_tool_xy_stage0"][i].float().cpu().numpy()
                for k in ("pred_xy_perpoint", "pred_xy_det"):
                    if torch.is_tensor(o.get(k)): d[k] = o[k][i].float().cpu().numpy()
                for k in ("probs_tool", "heatmap_tool"):
                    if torch.is_tensor(o.get(k)) and o[k].numel() < 5_000_000: d[k] = o[k][i].float().cpu().numpy().astype(np.float16)
                np.savez_compressed(out / f"{sid}.npz", **d); n += 1
    finally:
        if backup is not None: ema.restore(model, backup)
print(f"[dump] wrote {n} clips to {out}", flush=True)
