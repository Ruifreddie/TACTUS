#!/usr/bin/env python3
"""Frozen V-JEPA 2.1 features of the When baselines: TactusNet's own encoder, so only the localization method differs.

  pool       CPU  8 steps: mean over the 24x24 grid of TactusNet's cached tokens ex.tokens(sid) [8,24,24,1024]
                  -> <root>/s8/<sid>.npy fp16 [8,1024]. Step k = protocol frames 2k, 2k+1, centre (2k+0.5)/15.
  extract32  GPU  32 steps: 64 frames uniform over the window (round(linspace(abs_start, abs_end, 64)), end frames
                  included, read with ex.raw_frames), TactusNet's preprocessing (RGB, /255, bilinear squash to 384x384,
                  ImageNet mean/std), the same V-JEPA 2.1 ViT-L/384 in fp32, 4 chunks of 16 frames -> 32 tokens, rounded
                  to fp16 like the token cache, mean over space -> <root>/s32/<sid>.npy fp16 [32,1024]. Centre (2k+0.5)/63.
  check      GPU  re-encodes the 16 protocol frames of a few clips with the extract32 code path and compares the tokens
                  with ex.tokens(sid) (must agree up to fp16 rounding) -> <root>/s32/_token_check.json
  text       CPU  CLIP ViT-B/32 text-token features (last_hidden_state) of QD-DETR's fixed pair templates
                  -> <root>/text/clip_vitb32.npz (CLIP weights downloaded once into $BL_WEIGHTS)

  python -m baselines.when.features pool      [--splits train,val,test] [--feat-root DIR]
  python -m baselines.when.features check     [--sids 000000,000006 | --n 2]
  python -m baselines.when.features extract32 [--splits train,val,test] [--limit N] [--batch 4] [--shard i/n]
  python -m baselines.when.features text
Every command skips outputs that already exist, so an interrupted run is resumed by re-running it.
--splits takes train, val, test, silver or all (all = every clip of the export, all.txt).
"""
import argparse, collections, importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from baselines.common import config
from baselines.common.data import get_export

from .data import DIM, feat_dir, feat_root

REPO = Path(__file__).resolve().parents[2]
ARCH = dict(arch="vjepa2_1_vit_large_384", img_size=384, num_frames=16, patch_size=16, tubelet_size=2)
GRID = 24          # 384 / patch 16
CHUNK = 16         # frames per encoder call (TactusNet's clip length)
N32 = 64           # frames behind the 32-step setting


# ---------------------------------------------------------------------------------------------------------------- io
def _sids(ex, splits, limit=None, shard=None):
    out = []
    for sp in splits.split(","):
        if sp == "all":
            out += [l.strip() for l in (ex.root / "all.txt").read_text().splitlines() if l.strip()]
        else:
            out += ex.sids(sp)
    out = list(dict.fromkeys(out))                              # unique, order kept
    if shard:
        i, n = (int(v) for v in shard.split("/")); out = out[i::n]
    return out[:limit] if limit else out


def _save(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f: np.save(f, arr)
    os.replace(tmp, path)


def _done(path: Path, steps: int) -> bool:
    if not path.is_file(): return False
    try: return np.load(path, mmap_mode="r").shape == (steps, DIM)
    except Exception: return False


# ---------------------------------------------------------------------------------------------------------------- pool
def cmd_pool(a):
    ex = get_export(); root = feat_root(a.feat_root); out = feat_dir(root, 8); sids = _sids(ex, a.splits, a.limit, a.shard)
    todo = [s for s in sids if not _done(out / f"{s}.npy", 8)]
    print(f"[pool] {len(sids)} clips, {len(sids) - len(todo)} done, {len(todo)} to pool from {a.tok_root or config.TOKENS} -> {out}")

    def one(s):
        tok = np.asarray(ex.tokens(s, a.tok_root), dtype=np.float32)            # [8,24,24,1024], fp16 on disk
        if tok.shape != (8, GRID, GRID, DIM): raise ValueError(f"{s}: token shape {tok.shape}")
        _save(out / f"{s}.npy", tok.mean(axis=(1, 2)).astype(np.float16))

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(one, todo)):
            if (i + 1) % 1000 == 0: print(f"[pool] {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"[pool] done: {len(todo)} new files in {time.time() - t0:.1f} s")


# ---------------------------------------------------------------------------------------------------------------- encoder
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(frames_rgb_u8, size=384):
    """[N,H,W,3] uint8 RGB -> [N,3,size,size] float32 on CPU, exactly like the token converter / UniTAP preprocess():
    /255, bilinear squash to size x size (align_corners=False, no antialias), ImageNet mean/std."""
    import torch, torch.nn.functional as F
    v = torch.from_numpy(np.ascontiguousarray(frames_rgb_u8)).permute(0, 3, 1, 2).contiguous().float() / 255.0
    v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (v - mean) / std


def _default_vjepa_repo():
    for c in (os.environ.get("VJEPA_REPO"), os.environ.get("U") and Path(os.environ["U"]) / "third_party/t2b/vjepa2",
              Path(os.environ.get("BL_ROOT", "/nonexistent")) / "smoke/code/unitap/third_party/t2b/vjepa2"):
        if c and Path(c).is_dir(): return str(c)
    return None


def _load_wrapper_module(vjepa_repo):
    """The V-JEPA wrapper TactusNet's token converter used: this repo's tactusnet/_vendored_vjepa_wrapper.py (identical
    final-layer path to UniTAP's kpt_combo/vjepa_wrapper.py), loaded by file path so the tactusnet package (and its
    imports) is not needed; falls back to kpt_combo/vjepa_wrapper.py next to the vjepa2 checkout."""
    cands = [REPO / "tactusnet" / "_vendored_vjepa_wrapper.py", Path(vjepa_repo).parent / "kpt_combo" / "vjepa_wrapper.py"]
    for p in cands:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("_when_vjepa_wrapper", p); mod = importlib.util.module_from_spec(spec)
            old, sys.dont_write_bytecode = sys.dont_write_bytecode, True        # never write a __pycache__ next to that file
            try: spec.loader.exec_module(mod)
            finally: sys.dont_write_bytecode = old
            return mod, str(p)
    raise SystemExit(f"no V-JEPA wrapper found in {cands}")


class VJEPAEncoder:
    """V-JEPA 2.1 ViT-L/384, final layer, built and loaded exactly like TactusNet's token cache (fp32, default TF32 flags)."""

    def __init__(self, vjepa_repo, checkpoint, device="cuda"):
        import torch
        if not vjepa_repo or not Path(vjepa_repo).is_dir(): raise SystemExit(f"vjepa2 checkout not found: {vjepa_repo} (set --vjepa-repo or VJEPA_REPO)")
        if not checkpoint or not Path(checkpoint).is_file(): raise SystemExit(f"V-JEPA checkpoint not found: {checkpoint} (set --ckpt or VJEPA_CKPT)")
        mod, self.wrapper_path = _load_wrapper_module(vjepa_repo)
        self.device = torch.device(device)
        self.net = mod.VJEPAWrapper(vjepa_repo=str(vjepa_repo), hf_repo="", pretrained_from_hub=False, checkpoint_path=str(checkpoint),
                                    freeze=True, **ARCH).to(self.device).eval()
        self.info = dict(wrapper=self.wrapper_path, vjepa_repo=str(vjepa_repo), checkpoint=str(checkpoint), load_report=self.net.load_report,
                         torch=torch.__version__, device=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                         precision="fp32", matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                         cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), **ARCH)

    def encode(self, x, batch=1, oom_retries=20, oom_wait=30.0):
        """x: preprocessed float32 [16m, 3, 384, 384] (CPU) -> tokens float32 [8m, 24, 24, 1024] (CPU), chunk by chunk."""
        import torch
        m = x.shape[0] // CHUNK
        if m * CHUNK != x.shape[0]: raise ValueError(f"frame count {x.shape[0]} is not a multiple of {CHUNK}")
        x = x.view(m, CHUNK, *x.shape[1:]); out = []
        for i in range(0, m, batch):
            for attempt in range(oom_retries + 1):
                try:
                    with torch.inference_mode():
                        tg = self.net(x[i:i + batch].to(self.device, non_blocking=True))[0]        # [b,8,24,24,1024]
                    out.append(tg.float().cpu()); break
                except torch.cuda.OutOfMemoryError:
                    if attempt == oom_retries: raise
                    torch.cuda.empty_cache(); print(f"[encode] CUDA OOM (shared GPU), retry {attempt + 1}/{oom_retries} in {oom_wait:.0f} s", flush=True)
                    time.sleep(oom_wait)
        return torch.cat(out).reshape(-1, GRID, GRID, DIM)


def _encoder(a):
    return VJEPAEncoder(a.vjepa_repo or _default_vjepa_repo(), a.ckpt or os.environ.get("VJEPA_CKPT"), a.device)


def frames64(ex, sid):
    """Absolute indices of the 64 frames behind the 32-step setting (np.round = the rule of the 16 protocol frames)."""
    m = ex.meta[sid]
    return np.round(np.linspace(int(m["abs_start"]), int(m["abs_end"]), N32)).astype(np.int64)


def pool_tokens(tok):
    """tokens float32 [T,24,24,1024] -> [T,1024] fp16: rounded to fp16 first (as the token cache), then the spatial mean."""
    import torch
    return tok.half().float().mean(dim=(1, 2)).half().numpy()


def _prefetch(fn, items, workers, depth):
    """Yield (item, fn(item) or exception) in order, with up to `depth` items decoded ahead by a thread pool."""
    with ThreadPoolExecutor(workers) as pool:
        q = collections.deque(); it = iter(items)
        for item in it:
            q.append((item, pool.submit(fn, item)))
            if len(q) >= depth: break
        while q:
            item, f = q.popleft()
            try: yield item, f.result()
            except Exception as e: yield item, e
            nxt = next(it, None)
            if nxt is not None: q.append((nxt, pool.submit(fn, nxt)))


# ---------------------------------------------------------------------------------------------------------------- extract32
def cmd_extract32(a):
    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    ex = get_export(); root = feat_root(a.feat_root); out = feat_dir(root, 32); sids = _sids(ex, a.splits, a.limit, a.shard)
    todo = [s for s in sids if not _done(out / f"{s}.npy", 32)]
    print(f"[extract32] {len(sids)} clips, {len(sids) - len(todo)} done, {len(todo)} to encode -> {out} (batch {a.batch} chunks of {CHUNK} frames)")
    if not todo: return
    enc = _encoder(a); info = dict(enc.info, frames=N32, chunks=N32 // CHUNK, steps=32, sampling="round(linspace(abs_start, abs_end, 64))",
                                   pooling="fp16 tokens, mean over 24x24", batch_chunks=a.batch)
    (out / "_extract32_info.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "_extract32_info.json").write_text(json.dumps(info, indent=1))

    def load(s):
        x = preprocess(ex.raw_frames(s, frames64(ex, s)))                     # [64,3,384,384] float32
        return x.pin_memory() if enc.device.type == "cuda" else x

    t0 = time.time(); n_ok, fails = 0, []
    for i, (s, x) in enumerate(_prefetch(load, todo, a.workers, a.prefetch)):
        if isinstance(x, Exception):
            fails.append({"sid": s, "error": repr(x)}); print(f"[extract32] FAIL {s}: {x!r}", flush=True); continue
        tok = enc.encode(x, a.batch, a.oom_retries, a.oom_wait)                 # [32,24,24,1024]
        _save(out / f"{s}.npy", pool_tokens(tok)); n_ok += 1
        if n_ok in (1, 5) or n_ok % 50 == 0:
            el = time.time() - t0; print(f"[extract32] {i + 1}/{len(todo)}  {el / (i + 1):.2f} s/clip  ETA {el / (i + 1) * (len(todo) - i - 1) / 3600:.2f} h", flush=True)
    el = time.time() - t0
    summ = dict(n_new=n_ok, n_failed=len(fails), failures=fails[:50], seconds=el, s_per_clip=el / max(1, n_ok), **info)
    (out / f"_extract32_run_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(summ, indent=1))
    print(f"[extract32] done: {n_ok} new, {len(fails)} failed, {el / max(1, n_ok):.2f} s/clip")
    if fails: sys.exit(1)


# ---------------------------------------------------------------------------------------------------------------- check
def cmd_check(a):
    """Encode the 16 protocol frames with the extract32 code path; compare with TactusNet's cached tokens."""
    import torch
    ex = get_export(); root = feat_root(a.feat_root)
    sids = a.sids.split(",") if a.sids else ex.sids("test")[:a.n]
    enc = _encoder(a); rows = []
    for s in sids:
        fr = ex.raw_frames(s, ex.sampled_abs(s))                                # the 16 protocol frames, RGB
        same_frames = bool((fr == ex.frames(s)).all())
        mine = enc.encode(preprocess(fr), batch=1).numpy()                       # [8,24,24,1024] float32
        ref = np.asarray(ex.tokens(s, a.tok_root), dtype=np.float32)            # fp16 cache
        d = np.abs(mine - ref); mine16 = mine.astype(np.float16).astype(np.float32)
        ulp = np.spacing(np.abs(ref).astype(np.float16)).astype(np.float32)     # fp16 spacing at the cached value
        ok = d <= ulp + 1e-5                                                     # within one fp16 ulp (+ slack near 0)
        p_mine = pool_tokens(torch.from_numpy(mine)).astype(np.float32); p_ref = ref.mean(axis=(1, 2))
        s8 = feat_dir(root, 8) / f"{s}.npy"; p_s8 = np.load(s8).astype(np.float32) if s8.is_file() else None
        r = dict(sid=s, frames_equal_export=same_frames, max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                 max_abs_diff_after_fp16=float(np.abs(mine16 - ref).max()), share_equal_after_fp16=float((mine16 == ref).mean()),
                 max_diff_in_fp16_ulps=float((d / np.maximum(ulp, 2.0 ** -14)).max()), share_within_1ulp=float(ok.mean()),
                 max_abs_token=float(np.abs(ref).max()), cosine=float((mine.ravel() @ ref.ravel()) / (np.linalg.norm(mine) * np.linalg.norm(ref))),
                 pooled_max_abs_diff=float(np.abs(p_mine - p_ref).max()), pooled_mean_abs_diff=float(np.abs(p_mine - p_ref).mean()),
                 pooled_vs_s8_max_abs_diff=None if p_s8 is None else float(np.abs(p_mine - p_s8).max()),
                 passed=bool(ok.all() and same_frames))
        rows.append(r); print(json.dumps(r))
    res = dict(passed=all(r["passed"] for r in rows), criterion="|mine_fp32 - cache_fp16| <= fp16 ulp(cache) + 1e-5 for every element",
               clips=rows, encoder=enc.info)
    p = feat_dir(root, 32) / "_token_check.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, indent=1))
    print(f"[check] {'PASSED' if res['passed'] else 'FAILED'} on {len(rows)} clips -> {p}")
    if not res["passed"]: sys.exit(1)


# ---------------------------------------------------------------------------------------------------------------- text
CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"       # HF main as of 2026-11-05 (OpenAI's CLIP ViT-B/32 weights)
CLIP_FILES = ["config.json", "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin"]


def clip_dir():
    d = config.WEIGHTS / "clip-vit-base-patch32"
    if not (d / "pytorch_model.bin").is_file():
        from huggingface_hub import snapshot_download
        print(f"[text] downloading {CLIP_REPO}@{CLIP_REVISION[:8]} -> {d}", flush=True)
        snapshot_download(CLIP_REPO, revision=CLIP_REVISION, local_dir=str(d), allow_patterns=CLIP_FILES)
    return d


def text_path(root): return Path(root) / "text" / "clip_vitb32.npz"


def cmd_text(a):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    from .methods.qddetr import PAIR_TEMPLATES
    ex = get_export(); root = feat_root(a.feat_root); out = text_path(root)
    if out.is_file() and not a.force: print(f"[text] exists: {out}"); return
    d = clip_dir(); tok = CLIPTokenizer.from_pretrained(str(d)); model = CLIPTextModel.from_pretrained(str(d)).eval()
    arrays = {}
    with torch.no_grad():
        for p, name in enumerate(ex.pairs):
            ids = tok(PAIR_TEMPLATES[name], return_tensors="pt")              # <|startoftext|> ... <|endoftext|>
            arrays[f"pair{p}"] = model(**ids).last_hidden_state[0].float().numpy()   # [L_p, 512], after final LayerNorm
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pairs=np.array(ex.pairs), templates=np.array([PAIR_TEMPLATES[n] for n in ex.pairs]),
             model=np.array(f"{CLIP_REPO}@{CLIP_REVISION}"), **arrays)
    print(f"[text] wrote {out}: " + ", ".join(f"{n} -> {PAIR_TEMPLATES[n]!r} {arrays[f'pair{p}'].shape}" for p, n in enumerate(ex.pairs)))


# ---------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pool", "extract32", "check", "text"])
    ap.add_argument("--feat-root", help="default $BL_WHEN_FEATS or $BL_OUT/when/feats")
    ap.add_argument("--splits", default="train,val,test"); ap.add_argument("--limit", type=int); ap.add_argument("--shard", help="i/n")
    ap.add_argument("--tok-root", help="token cache (default $BL_TOKENS)"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vjepa-repo", help="vjepa2 checkout (default $VJEPA_REPO, $U/third_party/t2b/vjepa2, or the smoke copy)")
    ap.add_argument("--ckpt", help="default $VJEPA_CKPT"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4, help="16-frame chunks per encoder call (4 = one clip; use 1 on a shared GPU)")
    ap.add_argument("--prefetch", type=int, default=4); ap.add_argument("--oom-retries", type=int, default=20); ap.add_argument("--oom-wait", type=float, default=30.0)
    ap.add_argument("--sids"); ap.add_argument("--n", type=int, default=2); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    {"pool": cmd_pool, "extract32": cmd_extract32, "check": cmd_check, "text": cmd_text}[a.cmd](a)


if __name__ == "__main__":
    main()
