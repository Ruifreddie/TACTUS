import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Derive an A2-canon dataset by canonicalizing the symmetric tool-point pair."
    )
    parser.add_argument("--src-root", type=str, required=True, help="Processed dataset root or its data/ directory.")
    parser.add_argument("--dst-root", type=str, required=True, help="Output dataset root.")
    parser.add_argument("--margin-threshold-px", type=float, default=4.0)
    parser.add_argument("--axis-eps-px", type=float, default=1.0)
    parser.add_argument("--vis-threshold", type=float, default=0.5)
    parser.add_argument("--link-unchanged", action="store_true", default=True)
    parser.add_argument("--copy-unchanged", action="store_true")
    return parser.parse_args()


def _resolve_data_root(root_like: Path) -> Path:
    root_like = root_like.expanduser().resolve()
    if (root_like / "labels.json").exists():
        return root_like
    data_dir = root_like / "data"
    if data_dir.exists() and (data_dir / "labels.json").exists():
        return data_dir
    raise FileNotFoundError(f"Could not find processed dataset data root under: {root_like}")


def _pair_spec_from_topology(topology: str) -> Tuple[Tuple[int, int], int]:
    if topology == "1A2T":
        return (1, 2), 0
    if topology == "2A1T":
        return (0, 1), 2
    raise ValueError(f"Unsupported topology: {topology}")


def _canonical_metrics(
    xy_frame: np.ndarray,
    *,
    pair_idx: Tuple[int, int],
    uniq_idx: int,
    topology: str,
    width: float,
    height: float,
    axis_eps_px: float,
) -> Optional[Dict[str, float]]:
    p0 = np.asarray(xy_frame[pair_idx[0]], dtype=np.float32)
    p1 = np.asarray(xy_frame[pair_idx[1]], dtype=np.float32)
    pu = np.asarray(xy_frame[uniq_idx], dtype=np.float32)
    scale = np.asarray([float(width), float(height)], dtype=np.float32)
    p0_px = p0 * scale
    p1_px = p1 * scale
    pu_px = pu * scale
    midpoint = 0.5 * (p0_px + p1_px)
    if topology == "1A2T":
        main = midpoint - pu_px
    else:
        main = pu_px - midpoint
    main_norm = float(np.linalg.norm(main))
    if main_norm <= float(axis_eps_px):
        return None
    side_unit = np.asarray([-main[1], main[0]], dtype=np.float32) / max(main_norm, 1e-8)
    s0 = float(np.dot(p0_px - midpoint, side_unit))
    s1 = float(np.dot(p1_px - midpoint, side_unit))
    return {
        "score0": s0,
        "score1": s1,
        "pair_gap_px": float(abs(s1 - s0)),
        "main_axis_px": main_norm,
        "swap_needed": bool(s0 > s1),
    }


def _swap_pair_frame(arr: np.ndarray, pair_idx: Tuple[int, int]) -> np.ndarray:
    out = np.asarray(arr).copy()
    i0, i1 = int(pair_idx[0]), int(pair_idx[1])
    out[..., [i0, i1], :] = out[..., [i1, i0], :]
    return out


def _swap_pair_vis(arr: np.ndarray, pair_idx: Tuple[int, int]) -> np.ndarray:
    out = np.asarray(arr).copy()
    i0, i1 = int(pair_idx[0]), int(pair_idx[1])
    out[..., [i0, i1]] = out[..., [i1, i0]]
    return out


def _link_or_copy(src: Path, dst: Path, *, link_unchanged: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if link_unchanged:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy_json_with_updates(src: Path, dst: Path, updates: Dict[str, Any]) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    data.update(updates)
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _empty_stats() -> Dict[str, float]:
    return {
        "frames_total": 0.0,
        "frames_visible_all3": 0.0,
        "frames_direct_canonical": 0.0,
        "frames_inherit_prev": 0.0,
        "frames_uid_fallback": 0.0,
        "frames_swapped": 0.0,
        "frames_direct_swapped": 0.0,
        "frames_inherit_swapped": 0.0,
        "frames_low_margin": 0.0,
        "frames_low_vis_or_axis": 0.0,
    }


def _finalize_stats(stats: Dict[str, float]) -> Dict[str, Any]:
    frames_total = max(float(stats["frames_total"]), 1.0)
    out: Dict[str, Any] = {k: int(v) if k.startswith("frames_") else v for k, v in stats.items()}
    out["swap_fraction"] = float(stats["frames_swapped"]) / frames_total
    out["direct_fraction"] = float(stats["frames_direct_canonical"]) / frames_total
    out["inherit_fraction"] = float(stats["frames_inherit_prev"]) / frames_total
    out["uid_fallback_fraction"] = float(stats["frames_uid_fallback"]) / frames_total
    out["low_margin_fraction"] = float(stats["frames_low_margin"]) / frames_total
    return out


def derive_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    src_data = _resolve_data_root(Path(args.src_root))
    src_root = src_data.parent
    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_data = dst_root / "data"
    dst_data.mkdir(parents=True, exist_ok=True)

    link_unchanged = bool(args.link_unchanged and not args.copy_unchanged)
    labels = json.loads((src_data / "labels.json").read_text(encoding="utf-8"))
    meta = json.loads((src_data / "meta.json").read_text(encoding="utf-8"))
    sid_to_meta = {str(row["sid"]): row for row in meta}

    unchanged_entries = [
        "combo_ids",
        "video_16_npy",
        "vjepa_token_cache",
        "debug_aug_10",
        "train.txt",
        "val.txt",
        "test.txt",
        "index.txt",
        "meta.json",
        "comboid_split_report.json",
        "comboid_split_report.tsv",
        "fold1_aug_detailed_report.json",
        "fold1_aug_detailed_report.tsv",
        "fold1_aug_v3_detailed_report.json",
        "fold1_aug_v3_detailed_report.tsv",
        "kpt_combo_contract_audit.json",
        "selected_case_folders.json",
        "selected_case_folders.tsv",
        "split_case_combo_report.json",
        "split_case_combo_report.tsv",
        "split_sid_details.tsv",
        "convert_vjepa.log",
        "convert_vjepa_cmd.sh",
    ]
    for name in unchanged_entries:
        src = src_data / name
        if src.exists():
            _link_or_copy(src, dst_data / name, link_unchanged=link_unchanged)

    for name in ["build.log", "build_cmd.sh"]:
        src = src_root / name
        if src.exists():
            _link_or_copy(src, dst_root / name, link_unchanged=link_unchanged)

    gt_tool_xy_dst = dst_data / "gt_tool_xy"
    gt_tool_vis_dst = dst_data / "gt_tool_vis"
    gt_tool_xy_dst.mkdir(exist_ok=True)
    gt_tool_vis_dst.mkdir(exist_ok=True)

    global_stats = _empty_stats()
    by_tool: DefaultDict[str, Dict[str, float]] = defaultdict(_empty_stats)
    by_combo: DefaultDict[str, Dict[str, float]] = defaultdict(_empty_stats)

    margin_threshold_px = float(args.margin_threshold_px)
    axis_eps_px = float(args.axis_eps_px)
    vis_threshold = float(args.vis_threshold)

    for sid_path in sorted((src_data / "gt_tool_xy").glob("*.npy")):
        sid = sid_path.stem
        meta_row = sid_to_meta.get(sid)
        if meta_row is None:
            raise KeyError(f"Missing meta row for sid={sid}")
        topology = str(meta_row["topology"])
        tool = str(meta_row["tool"])
        combo = str(meta_row["combo"])
        width = float(meta_row["W"])
        height = float(meta_row["H"])
        pair_idx, uniq_idx = _pair_spec_from_topology(topology)

        xy = np.load(sid_path).astype(np.float32)
        vis = np.load(src_data / "gt_tool_vis" / f"{sid}.npy").astype(np.float32)
        xy_out = xy.copy()
        vis_out = vis.copy()

        prev_swap: Optional[bool] = None
        T = int(xy.shape[0])
        for t in range(T):
            for stats in (global_stats, by_tool[tool], by_combo[combo]):
                stats["frames_total"] += 1.0

            frame_vis = np.asarray(vis[t], dtype=np.float32)
            all_visible = bool(
                frame_vis[pair_idx[0]] > vis_threshold
                and frame_vis[pair_idx[1]] > vis_threshold
                and frame_vis[uniq_idx] > vis_threshold
            )
            if all_visible:
                for stats in (global_stats, by_tool[tool], by_combo[combo]):
                    stats["frames_visible_all3"] += 1.0

            metrics = None
            if all_visible:
                metrics = _canonical_metrics(
                    xy[t],
                    pair_idx=pair_idx,
                    uniq_idx=uniq_idx,
                    topology=topology,
                    width=width,
                    height=height,
                    axis_eps_px=axis_eps_px,
                )

            use_prev = False
            swap_now = False
            if metrics is not None and float(metrics["pair_gap_px"]) >= margin_threshold_px:
                swap_now = bool(metrics["swap_needed"])
                prev_swap = swap_now
                for stats in (global_stats, by_tool[tool], by_combo[combo]):
                    stats["frames_direct_canonical"] += 1.0
                    stats["frames_direct_swapped"] += float(swap_now)
            else:
                if metrics is not None:
                    for stats in (global_stats, by_tool[tool], by_combo[combo]):
                        stats["frames_low_margin"] += 1.0
                else:
                    for stats in (global_stats, by_tool[tool], by_combo[combo]):
                        stats["frames_low_vis_or_axis"] += 1.0

                if prev_swap is not None:
                    use_prev = True
                    swap_now = bool(prev_swap)
                    for stats in (global_stats, by_tool[tool], by_combo[combo]):
                        stats["frames_inherit_prev"] += 1.0
                        stats["frames_inherit_swapped"] += float(swap_now)
                else:
                    swap_now = False
                    for stats in (global_stats, by_tool[tool], by_combo[combo]):
                        stats["frames_uid_fallback"] += 1.0

            if swap_now:
                xy_out[t, [pair_idx[0], pair_idx[1]], :] = xy_out[t, [pair_idx[1], pair_idx[0]], :]
                vis_out[t, [pair_idx[0], pair_idx[1]]] = vis_out[t, [pair_idx[1], pair_idx[0]]]
                for stats in (global_stats, by_tool[tool], by_combo[combo]):
                    stats["frames_swapped"] += 1.0

        np.save(gt_tool_xy_dst / f"{sid}.npy", xy_out.astype(np.float32))
        np.save(gt_tool_vis_dst / f"{sid}.npy", vis_out.astype(np.float32))

    labels_out = dict(labels)
    tool_point_order = dict(labels_out.get("tool_point_order", {}))
    tool_point_order["2A1T"] = "[anchor0, anchor1, tip0] with anchor pair canonicalized by tool-centric perpendicular-axis order"
    tool_point_order["1A2T"] = "[anchor0, tip0, tip1] with tip pair canonicalized by tool-centric perpendicular-axis order"
    tool_point_order["note"] = (
        "symmetric pair order is canonicalized from GT geometry; if pair gap < margin threshold, "
        "inherit previous frame order in the clip when available, else fall back to uid order"
    )
    labels_out["tool_point_order"] = tool_point_order
    labels_out["canonicalization"] = {
        "enabled": True,
        "margin_threshold_px": margin_threshold_px,
        "axis_eps_px": axis_eps_px,
        "vis_threshold": vis_threshold,
        "rule": {
            "1A2T": "anchor -> tip-midpoint main axis; sort tips by perpendicular score",
            "2A1T": "anchor-midpoint -> tip main axis; sort anchors by perpendicular score",
            "slot_convention": "slot0 = lower perpendicular score; slot1 = higher perpendicular score",
        },
        "ambiguous_frame_policy": "if margin < threshold or visibility/axis is weak, inherit previous frame order in clip; otherwise uid order",
        "source_dataset_root": str(src_root),
    }
    (dst_data / "labels.json").write_text(json.dumps(labels_out, indent=2), encoding="utf-8")

    report = {
        "source_dataset_root": str(src_root),
        "source_data_root": str(src_data),
        "output_dataset_root": str(dst_root),
        "output_data_root": str(dst_data),
        "canonicalization": labels_out["canonicalization"],
        "global": _finalize_stats(global_stats),
        "by_tool": {k: _finalize_stats(v) for k, v in sorted(by_tool.items())},
        "by_combo": {k: _finalize_stats(v) for k, v in sorted(by_combo.items())},
    }
    (dst_data / "canonicalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = derive_dataset(args)
    print(f"[a2-canon] saved dataset: {report['output_dataset_root']}")
    g = report["global"]
    print(
        "[a2-canon] summary "
        f"frames={g['frames_total']} "
        f"swapped={g['frames_swapped']} "
        f"swap_fraction={g['swap_fraction']:.4f} "
        f"low_margin_fraction={g['low_margin_fraction']:.4f} "
        f"inherit_fraction={g['inherit_fraction']:.4f} "
        f"uid_fallback_fraction={g['uid_fallback_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
