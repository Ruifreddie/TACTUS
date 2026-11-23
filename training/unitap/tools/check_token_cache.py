#!/usr/bin/env python3
"""Validate V-JEPA token cache CONTENT, not just loadability.

Level 1 (default, CPU only):
  * loadable, ndim 4/5, fp16, expected shape, all finite
  * per-frame: no frame all-zero, no frame with ~zero std (dead / half-written output)
  * per-token: no all-zero token vectors
  * per-file mean-norm compared against the population (robust z-score); outliers reported
Level 2 (--recompute N, needs GPU + video_16_npy + the same encoder):
  * re-encode N random cached clips from the raw video and compare value-by-value
--fix deletes bad files (and stray .tmp) so the converter recomputes them.
"""
import argparse, time, collections, random, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache-root", required=True)
ap.add_argument("--fix", action="store_true")
ap.add_argument("--min-age-s", type=float, default=10.0, help="skip files newer than this (writer may be active)")
ap.add_argument("--expect-shape", default="8,24,24,1024")
ap.add_argument("--z-thresh", type=float, default=6.0, help="robust z-score threshold for per-file norm outliers")
ap.add_argument("--recompute", type=int, default=0, help="re-encode this many random cached clips and compare; -1 = ALL")
ap.add_argument("--recompute-sleep", type=float, default=0.3, help="pause between clips (gentle GPU)")
ap.add_argument("--report", default="", help="write per-file recompute results here")
ap.add_argument("--data-root", default="", help="dataset root (video_16_npy) for --recompute")
ap.add_argument("--vjepa-repo", default=""); ap.add_argument("--vjepa-checkpoint", default="")
ap.add_argument("--t2b-root", default="", help="path containing kpt_combo/ for --recompute")
ap.add_argument("--layer-indices", default="", help="comma-separated 1-based encoder layers for a multi-layer cache, e.g. 6,12,18,24 (expect-shape then has 5 dims)")
a = ap.parse_args()

root = Path(a.cache_root); now = time.time()
expect = tuple(int(x) for x in a.expect_shape.split(","))
bad, skipped, ok_files, stats = [], 0, [], {}

for tmp in root.rglob("*.tmp"):
    if now - tmp.stat().st_mtime >= a.min_age_s: bad.append((tmp, "stray .tmp"))

for p in sorted(root.rglob("*.npy")):
    if now - p.stat().st_mtime < a.min_age_s: skipped += 1; continue
    try:
        arr = np.load(p, allow_pickle=False)
        if arr.dtype != np.float16: raise ValueError(f"dtype={arr.dtype}")
        if arr.shape != expect: raise ValueError(f"shape={arr.shape} expected {expect}")
        x = arr.astype(np.float32)
        if not np.isfinite(x).all(): raise ValueError(f"non-finite: {int((~np.isfinite(x)).sum())} values")
        # 4D [T,H,W,D] or 5D [K,T,H,W,D] (multi-layer): run the frame checks per layer slice
        layers = x if x.ndim == 5 else x[None]
        for k, xl in enumerate(layers):
            T = xl.shape[0]
            fr = xl.reshape(T, -1)
            fabs = np.abs(fr).max(1); fstd = fr.std(1)
            tag = f"layer{k} " if x.ndim == 5 else ""
            if (fabs == 0).any(): raise ValueError(f"{tag}all-zero frames: {np.where(fabs == 0)[0].tolist()}")
            if (fstd < 1e-4).any(): raise ValueError(f"{tag}dead frames (std<1e-4): {np.where(fstd < 1e-4)[0].tolist()}")
            # frames identical to each other is suspicious (repeated write of one frame)
            if T > 1 and np.allclose(fr[0], fr[1:], atol=1e-3): raise ValueError(f"{tag}all frames identical")
        if x.ndim == 5 and x.shape[0] > 1:
            # layers identical to each other = wrong out_layers wiring
            if np.allclose(x[0], x[-1], atol=1e-3): raise ValueError("first and last layer identical")
        tok = x.reshape(-1, x.shape[-1]); tnorm = np.linalg.norm(tok, axis=1)
        nz = int((tnorm == 0).sum())
        if nz: raise ValueError(f"{nz} all-zero token vectors")
        stats[p] = (float(tnorm.mean()), float(tnorm.std()), float(fstd.min()))
        ok_files.append(p)
    except Exception as e:
        bad.append((p, str(e)))

# population outliers on mean token norm
if len(stats) >= 20:
    m = np.array([v[0] for v in stats.values()])
    med = np.median(m); mad = np.median(np.abs(m - med)) * 1.4826 + 1e-6
    for p, (mn, _, _) in stats.items():
        z = (mn - med) / mad
        if abs(z) > a.z_thresh: bad.append((p, f"mean token norm {mn:.2f} is outlier (z={z:.1f}, median {med:.2f})"))
    print(f"population mean-token-norm: median={med:.2f} MAD={mad:.2f} min={m.min():.2f} max={m.max():.2f}")

print(f"ok={len(ok_files) - sum(1 for p,_ in bad if p in stats)} bad={len(bad)} skipped_recent={skipped}")
by_split = collections.Counter(p.parent.name for p in ok_files)
for k, v in sorted(by_split.items()): print(f"  {k}: {v}")
for p, why in bad:
    print(f"  BAD {p}: {why}")
    if a.fix and p.exists(): p.unlink(); print("    deleted")

if a.recompute != 0:
    import torch, torch.nn.functional as F
    sys.path.insert(0, a.t2b_root)
    from kpt_combo.dataset import IMAGENET_MEAN, IMAGENET_STD
    from kpt_combo.vjepa_wrapper import VJEPAWrapper
    conv_dir = Path(__file__).resolve().parent.parent / "third_party" / "vitpose_dataset_tools"
    sys.path.insert(0, str(conv_dir))
    from convert_video16_to_vjepa_tokens_one_by_one import _prepare_video_tensor
    model = VJEPAWrapper(vjepa_repo=a.vjepa_repo, arch="vjepa2_1_vit_large_384", img_size=384, num_frames=16,
                         patch_size=16, tubelet_size=2, hf_repo="", hf_local_files_only=False,
                         pretrained_from_hub=False, checkpoint_path=a.vjepa_checkpoint, freeze=True,
                         layer_indices=[int(v) for v in a.layer_indices.split(",") if v.strip()]).cuda().eval()
    good = [p for p in ok_files if p not in dict(bad)]
    random.seed(0); sample = good if a.recompute < 0 else random.sample(good, min(a.recompute, len(good)))
    rep = open(a.report, "w") if a.report else None
    worst = 0.0; nbad = 0
    for p in sample:
        vid = Path(a.data_root) / "video_16_npy" / p.name
        v = np.load(vid, allow_pickle=False)
        x = _prepare_video_tensor(video_thwc_u8=v, input_size=384).cuda()
        with torch.no_grad(): tg, _, _, _ = model(x)
        ref = tg[0].float().cpu().numpy(); cached = np.load(p).astype(np.float32)
        d = np.abs(ref - cached).max(); rel = d / (np.abs(ref).max() + 1e-6)
        cos = float((ref.ravel() @ cached.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(cached) + 1e-6))
        flag = "" if (cos > 0.999 and rel < 0.05) else "  <-- MISMATCH"
        if flag: nbad += 1
        worst = max(worst, 1 - cos)
        line = f"{p.parent.name}/{p.name} maxdiff={d:.4f} rel={rel:.4f} cos={cos:.6f}{flag}"
        if rep: rep.write(line + "\n"); rep.flush()
        if flag or a.recompute > 0: print("  recompute " + line)
        if flag and a.fix: p.unlink(); print("    deleted")
        time.sleep(a.recompute_sleep)
    print(f"recompute: {len(sample)} checked, {nbad} mismatches, worst 1-cos={worst:.2e}")
    if rep: rep.close()
