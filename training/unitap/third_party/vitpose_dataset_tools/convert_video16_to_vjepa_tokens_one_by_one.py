#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-by-one converter: `video_16_npy/{sid}.npy` -> VJEPA token cache.

Per-sample processing order (strictly one sid at a time):
  1) load one `video_16_npy/{sid}.npy`
  2) (default) delete the video file first
  3) run frozen VJEPA forward
  4) save token grid to `{cache_root}/{split}/{sid}.npy`

Token file format:
  - single-layer mode: per sid [T', H', W', D] (4D)
  - multi-layer mode: per sid [K, T', H', W', D] (5D)
  - layout: root/{split}/{sid}.npy

Important note for `multi4` mode:
  - `multi4` keeps four encoder layers in the same `.npy`, using layer ids 6,12,18,24
  - it does NOT create four sibling cache folders
  - the extra layer axis is the first axis `K`
  - current training compatibility paths read these 5D caches by taking the last cached layer,
    unless a later experiment explicitly adds multi-layer consumption
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from kpt_combo.dataset import IMAGENET_MEAN, IMAGENET_STD
    from kpt_combo.vjepa_wrapper import VJEPAWrapper
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from kpt_combo.dataset import IMAGENET_MEAN, IMAGENET_STD
    from kpt_combo.vjepa_wrapper import VJEPAWrapper


def _cache_np_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "fp16":
        return np.float16
    if dtype_name == "fp32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _parse_layer_indices(spec: str) -> List[int]:
    spec = str(spec).strip()
    if not spec:
        return []
    out: List[int] = []
    seen = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        val = int(item)
        if val <= 0:
            raise ValueError(f"`layer_indices` must contain positive 1-based ids, got {val}")
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _read_split_ids(split_file: Path) -> List[str]:
    if not split_file.is_file():
        return []
    lines = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def _prepare_video_tensor(video_thwc_u8: np.ndarray, input_size: int) -> torch.Tensor:
    if video_thwc_u8.ndim != 4 or int(video_thwc_u8.shape[-1]) != 3:
        raise ValueError(f"video_16_npy must be [T,H,W,3], got {tuple(video_thwc_u8.shape)}")

    # Exported by OpenCV path (BGR); convert to RGB to match train preprocessing.
    video_rgb = np.ascontiguousarray(video_thwc_u8[..., ::-1])
    video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).contiguous().float() / 255.0  # [T,3,H,W]
    if int(video.shape[0]) != 16:
        raise ValueError(f"Expected 16 frames, got {int(video.shape[0])}")

    if int(input_size) > 0:
        video = F.interpolate(video, size=(int(input_size), int(input_size)), mode="bilinear", align_corners=False)

    video = (video - IMAGENET_MEAN) / IMAGENET_STD
    return video.unsqueeze(0).contiguous()  # [1,T,3,H,W]


def _validate_token_file(path: Path) -> Tuple[bool, Optional[Tuple[int, ...]]]:
    if not path.is_file():
        return False, None
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim not in (4, 5):
            return False, None
        return True, tuple(arr.shape)
    except Exception:
        return False, None


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    # Avoid numpy auto-appending ".npy" by writing via file handle.
    with tmp_path.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp_path, path)


def _resolve_split_file(data_root: Path, split: str, override: str) -> Path:
    if override and str(override).strip():
        p = Path(str(override))
        if not p.is_absolute():
            p = data_root / p
        return p
    return data_root / f"{split}.txt"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="Dataset root containing video_16_npy + split txt files.")
    ap.add_argument(
        "--video-dir",
        type=str,
        default="",
        help="Optional video_16_npy directory override (e.g., external disk path).",
    )
    ap.add_argument("--cache-root", type=str, default="", help="Output token cache root. Default: <data-root>/vjepa_token_cache")
    ap.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to process.")
    ap.add_argument("--train-split-file", type=str, default="")
    ap.add_argument("--val-split-file", type=str, default="")
    ap.add_argument("--test-split-file", type=str, default="")
    ap.add_argument("--start-index", type=int, default=0, help="Start sid index within each split (after reading split file).")
    ap.add_argument("--num-samples", type=int, default=-1, help="Number of sids to process from --start-index per split (-1 means all).")
    ap.add_argument("--max-samples", type=int, default=-1, help="Max processed sids per split (-1 means all).")
    ap.add_argument(
        "--sid-chunk-size",
        type=int,
        default=-1,
        help="Process split in sid chunks. <=0 means whole split (default).",
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Token forward batch size (>=1).")
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--token-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument(
        "--cache-layer-mode",
        type=str,
        default="single_final",
        choices=["single_final", "multi4"],
        help=(
            "How many VJEPA encoder layers to cache. "
            "`single_final` keeps the current behavior. "
            "`multi4` caches four evenly spaced layers: 6,12,18,24."
        ),
    )
    ap.add_argument(
        "--layer-indices",
        type=str,
        default="",
        help=(
            "Optional custom comma-separated 1-based encoder layer ids, "
            "for example `6,12,18,24`. Overrides --cache-layer-mode."
        ),
    )
    ap.add_argument("--vjepa-repo", type=str, default="vjepa2", help="Local V-JEPA repo path for repo backend.")
    ap.add_argument("--vjepa-arch", type=str, default="vit_large", help="Repo-backend arch name.")
    ap.add_argument("--vjepa-hf-repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--vjepa-hf-local-files-only", action="store_true")
    ap.add_argument("--vjepa-pretrained-hub", action="store_true", help="Use repo backend pretrained hub weights.")
    ap.add_argument("--vjepa-checkpoint", type=str, default="", help="Optional repo-backend checkpoint path.")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument(
        "--sleep-after-new-tokens",
        type=int,
        default=0,
        help="If >0, sleep after writing this many new cache files.",
    )
    ap.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep duration used with --sleep-after-new-tokens.",
    )
    ap.add_argument("--keep-video", action="store_true", help="Do not delete video_16_npy files.")
    ap.add_argument(
        "--delete-stage",
        type=str,
        default="after_save",
        choices=["before_forward", "after_forward", "after_save"],
        help=(
            "When to delete input video file if keep-video is false: "
            "before_forward | after_forward | after_save. "
            "Default is after_save (compute -> save tokens -> delete inputs)."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    video_dir = (
        Path(str(args.video_dir)).expanduser().resolve()
        if str(args.video_dir).strip()
        else (data_root / "video_16_npy")
    )
    if not video_dir.is_dir():
        raise FileNotFoundError(f"missing video dir: {video_dir}")

    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if str(args.cache_root).strip()
        else (data_root / "vjepa_token_cache")
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    np_dtype = _cache_np_dtype(str(args.token_dtype))
    batch_size = max(1, int(args.batch_size))
    sid_chunk_size = int(args.sid_chunk_size)
    delete_video = not bool(args.keep_video)
    delete_stage = str(args.delete_stage)
    layer_indices = _parse_layer_indices(str(args.layer_indices))
    cache_layer_mode = str(args.cache_layer_mode)
    sleep_after_new_tokens = max(0, int(args.sleep_after_new_tokens))
    sleep_seconds = max(0.0, float(args.sleep_seconds))
    if not layer_indices and cache_layer_mode == "multi4":
        layer_indices = [6, 12, 18, 24]

    print("[setup]")
    print(f"  data_root={data_root}")
    print(f"  video_dir={video_dir}")
    print(f"  cache_root={cache_root}")
    print(f"  device={device}")
    print(f"  token_dtype={args.token_dtype}")
    print(f"  cache_layer_mode={cache_layer_mode}")
    print(f"  layer_indices={layer_indices if layer_indices else '[final-only]'}")
    print(f"  sid_chunk_size={sid_chunk_size}")
    print(f"  batch_size={batch_size}")
    print(f"  delete_video={delete_video}")
    print(f"  delete_stage={delete_stage}")
    print(f"  sleep_after_new_tokens={sleep_after_new_tokens}")
    print(f"  sleep_seconds={sleep_seconds}")
    print(f"  vjepa_repo={args.vjepa_repo}")
    print(f"  vjepa_arch={args.vjepa_arch}")
    print(f"  hf_repo={args.vjepa_hf_repo}")
    print(f"  hf_local_files_only={bool(args.vjepa_hf_local_files_only)}")
    print(f"  pretrained_hub={bool(args.vjepa_pretrained_hub)}")
    print(f"  checkpoint={args.vjepa_checkpoint or '(none)'}")

    if args.dry_run:
        print("[dry-run] no file will be written/deleted.")

    model = VJEPAWrapper(
        vjepa_repo=str(args.vjepa_repo),
        arch=str(args.vjepa_arch),
        img_size=int(args.input_size),
        num_frames=16,
        patch_size=16,
        tubelet_size=2,
        hf_repo=str(args.vjepa_hf_repo),
        hf_local_files_only=bool(args.vjepa_hf_local_files_only),
        pretrained_from_hub=bool(args.vjepa_pretrained_hub),
        checkpoint_path=str(args.vjepa_checkpoint),
        freeze=True,
        layer_indices=layer_indices,
    ).to(device)
    model.eval()

    split_names = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    split_override = {
        "train": str(args.train_split_file),
        "val": str(args.val_split_file),
        "test": str(args.test_split_file),
    }
    split_plan: Dict[str, Dict[str, object]] = {}
    sid_owner: Dict[str, str] = {}
    overlap_pairs: List[Tuple[str, str, str]] = []
    for split in split_names:
        split_file = _resolve_split_file(data_root=data_root, split=split, override=split_override.get(split, ""))
        sids = _read_split_ids(split_file)
        start_index = max(0, int(args.start_index))
        if start_index >= len(sids):
            sids = []
        else:
            sids = sids[start_index:]
        if int(args.num_samples) > 0:
            sids = sids[: int(args.num_samples)]
        elif int(args.max_samples) > 0:
            sids = sids[: int(args.max_samples)]
        split_plan[split] = {"split_file": split_file, "sids": sids}
        for sid in sids:
            prev = sid_owner.get(sid, "")
            if prev and prev != split:
                overlap_pairs.append((sid, prev, split))
            else:
                sid_owner[sid] = split
    if len(overlap_pairs) > 0:
        examples = ", ".join([f"{sid}({a}|{b})" for sid, a, b in overlap_pairs[:20]])
        raise ValueError(
            "Split overlap detected in conversion plan. "
            f"Overlapped sid count={len(overlap_pairs)} examples={examples}"
        )

    summary: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, str]] = []

    for split in split_names:
        split_file = split_plan[split]["split_file"]  # type: ignore[index]
        sids = list(split_plan[split]["sids"])  # type: ignore[index]
        n_total = len(sids)
        if n_total == 0:
            print(f"[split={split}] no samples from: {split_file}")
            summary[split] = {
                "total": 0,
                "existing_tokens": 0,
                "new_tokens": 0,
                "deleted_videos": 0,
                "failed": 0,
            }
            continue

        print(f"[split={split}] start total={n_total} split_file={split_file}")
        out_split_dir = cache_root / split
        if not args.dry_run:
            out_split_dir.mkdir(parents=True, exist_ok=True)

        count_existing = 0
        count_new = 0
        count_deleted = 0
        count_failed = 0
        new_since_sleep = 0

        if sid_chunk_size <= 0:
            sid_chunk_size = n_total
        chunk_start = 0
        while chunk_start < n_total:
            chunk_end = min(n_total, chunk_start + int(sid_chunk_size))
            print(
                f"[split={split}] chunk {chunk_start}:{chunk_end} "
                f"(size={chunk_end - chunk_start})"
            )
            idx = chunk_start
            while idx < chunk_end:
                batch_sids = sids[idx : min(idx + batch_size, chunk_end)]
                idx_end = idx + len(batch_sids)

                to_compute: List[Dict[str, object]] = []

                # Stage A: inspect existing tokens, or load videos for this batch.
                for sid in batch_sids:
                    token_path = out_split_dir / f"{sid}.npy"
                    video_path = video_dir / f"{sid}.npy"
                    try:
                        token_ok, _ = _validate_token_file(token_path)
                        if token_ok:
                            count_existing += 1
                            if delete_video and video_path.is_file() and (not args.dry_run):
                                video_path.unlink()
                                count_deleted += 1
                            continue

                        if not video_path.is_file():
                            raise FileNotFoundError(f"missing both valid token and video: sid={sid}")

                        video_thwc_u8 = np.load(video_path, allow_pickle=False)
                        x = _prepare_video_tensor(video_thwc_u8=video_thwc_u8, input_size=int(args.input_size))
                        to_compute.append(
                            {
                                "sid": sid,
                                "token_path": token_path,
                                "video_path": video_path,
                                "video_np": video_thwc_u8,
                                "x": x,
                            }
                        )
                    except Exception as e:
                        count_failed += 1
                        failures.append({"split": split, "sid": sid, "error": str(e)})
                        print(f"[ERROR][split={split} sid={sid}] {e}")

                # Stage B: optional delete-before-forward for this batch.
                if delete_video and (delete_stage == "before_forward") and (not args.dry_run):
                    for rec in to_compute:
                        vp = rec["video_path"]  # type: ignore[index]
                        if isinstance(vp, Path) and vp.is_file():
                            vp.unlink()
                            count_deleted += 1
                            rec["deleted_input"] = True
                        else:
                            rec["deleted_input"] = False
                else:
                    for rec in to_compute:
                        rec["deleted_input"] = False

                # Stage C: one forward for this batch, then save token one-by-one.
                if len(to_compute) > 0:
                    try:
                        x_batch = torch.cat([rec["x"] for rec in to_compute], dim=0).to(device)  # type: ignore[index]
                        with torch.no_grad():
                            tokens_grid, _, _, _ = model(x_batch)  # [B,T',H',W',D] or [B,K,T',H',W',D]
                        tok_batch = tokens_grid.detach().float().cpu().numpy()

                        # Requested default mode: delete inputs after forward, before token save.
                        if delete_video and (delete_stage == "after_forward") and (not args.dry_run):
                            for rec in to_compute:
                                vp = rec["video_path"]  # type: ignore[index]
                                if isinstance(vp, Path) and vp.is_file():
                                    vp.unlink()
                                    count_deleted += 1
                                    rec["deleted_input"] = True

                        for i_rec, rec in enumerate(to_compute):
                            sid = str(rec["sid"])
                            token_path = rec["token_path"]  # type: ignore[index]
                            video_path = rec["video_path"]  # type: ignore[index]
                            video_np = rec["video_np"]  # type: ignore[index]
                            try:
                                # Per-sid cache layout stays flat under {cache_root}/{split}/{sid}.npy.
                                # In multi-layer mode we store [K,T',H',W',D] in that single file,
                                # not one subfolder per encoder layer.
                                tok = tok_batch[i_rec].astype(np_dtype, copy=False)
                                if tok.ndim == 5 and int(tok.shape[0]) == 1:
                                    tok = tok[0]
                                if tok.ndim not in (4, 5):
                                    raise ValueError(f"token ndim must be 4 or 5, got {tok.ndim} for sid={sid}")

                                if not args.dry_run:
                                    _atomic_save_npy(token_path, tok)
                                    ok_saved, _ = _validate_token_file(token_path)
                                    if not ok_saved:
                                        raise RuntimeError(f"saved token validation failed: {token_path}")

                                if (
                                    delete_video
                                    and (delete_stage == "after_save")
                                    and (not args.dry_run)
                                    and video_path.is_file()
                                ):
                                    video_path.unlink()
                                    count_deleted += 1

                                count_new += 1
                                new_since_sleep += 1
                                if (
                                    sleep_after_new_tokens > 0
                                    and sleep_seconds > 0.0
                                    and new_since_sleep >= sleep_after_new_tokens
                                ):
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                    gc.collect()
                                    print(
                                        f"[sleep][split={split}] wrote={new_since_sleep} new tokens "
                                        f"-> sleep {sleep_seconds:.1f}s"
                                    )
                                    time.sleep(sleep_seconds)
                                    new_since_sleep = 0
                            except Exception as e:
                                count_failed += 1
                                failures.append({"split": split, "sid": sid, "error": str(e)})
                                print(f"[ERROR][split={split} sid={sid}] {e}")
                                # Restore deleted input on per-sid save failure.
                                try:
                                    if (
                                        bool(rec.get("deleted_input", False))
                                        and delete_video
                                        and (not args.dry_run)
                                        and (not video_path.is_file())
                                    ):
                                        np.save(video_path, video_np)
                                        count_deleted = max(0, count_deleted - 1)
                                except Exception:
                                    pass

                    except Exception as e:
                        # Whole-batch forward failed: restore any deleted inputs from memory.
                        for rec in to_compute:
                            sid = str(rec["sid"])
                            video_path = rec["video_path"]  # type: ignore[index]
                            video_np = rec["video_np"]  # type: ignore[index]
                            try:
                                if (
                                    bool(rec.get("deleted_input", False))
                                    and delete_video
                                    and (not args.dry_run)
                                    and (not video_path.is_file())
                                ):
                                    np.save(video_path, video_np)
                                    count_deleted = max(0, count_deleted - 1)
                            except Exception:
                                pass
                            count_failed += 1
                            failures.append({"split": split, "sid": sid, "error": f"batch_forward_failed: {e}"})
                        print(f"[ERROR][split={split}] batch forward failed on {len(to_compute)} samples: {e}")

                    finally:
                        if "x_batch" in locals():
                            del x_batch
                        if "tokens_grid" in locals():
                            del tokens_grid
                        if "tok_batch" in locals():
                            del tok_batch
                        for rec in to_compute:
                            if "x" in rec:
                                del rec["x"]
                            if "video_np" in rec:
                                del rec["video_np"]
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if (idx_end % int(args.log_interval) == 0) or idx_end == n_total:
                    print(
                        f"[split={split}] {idx_end}/{n_total} existing={count_existing} "
                        f"new={count_new} deleted={count_deleted} failed={count_failed}"
                    )

                idx = idx_end
            # Hard safety check per chunk: every sid in this chunk must now have a valid token.
            if not args.dry_run:
                missing_tokens: List[str] = []
                for sid in sids[chunk_start:chunk_end]:
                    token_ok, _ = _validate_token_file(out_split_dir / f"{sid}.npy")
                    if not token_ok:
                        missing_tokens.append(str(sid))
                if len(missing_tokens) > 0:
                    ex = ", ".join(missing_tokens[:20])
                    raise RuntimeError(
                        f"[split={split}] chunk token completeness check failed: "
                        f"chunk={chunk_start}:{chunk_end} missing={len(missing_tokens)} "
                        f"examples={ex}"
                    )
            chunk_start = chunk_end
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary[split] = {
            "total": int(n_total),
            "existing_tokens": int(count_existing),
            "new_tokens": int(count_new),
            "deleted_videos": int(count_deleted),
            "failed": int(count_failed),
        }
        print(f"[split={split}] done -> {summary[split]}")

    report = {
        "data_root": str(data_root),
        "video_dir": str(video_dir),
        "cache_root": str(cache_root),
        "device": str(device),
        "token_dtype": str(args.token_dtype),
        "cache_layer_mode": str(cache_layer_mode),
        "layer_indices": [int(x) for x in layer_indices],
        "sid_chunk_size": int(args.sid_chunk_size),
        "delete_video": bool(delete_video),
        "delete_stage": str(delete_stage),
        "summary": summary,
        "failures": failures,
    }
    report_path = cache_root / "convert_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[done]")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"  report={report_path}")
    if failures:
        print(f"  failures={len(failures)} (check report json)")
        raise RuntimeError(
            f"Conversion finished with failures={len(failures)}. "
            "Stop here to avoid partial/inconsistent raw deletion state."
        )


if __name__ == "__main__":
    main()
