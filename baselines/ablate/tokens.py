#!/usr/bin/env python3
"""Token caches of alternative frozen encoders, in TactusNet's layout, for the backbone-swap ablation (README.md).

  extract  GPU  <out>/all/<sid>.npy  fp16 [8,h,w,D] for every clip of --splits (backbones.py gives h, w, D per encoder)
                + <out>/manifest.json (encoder, weights, input size, layout, clip count, seconds per clip).
                Existing valid files are skipped, so a killed run resumes; --shard i/n splits the clips over GPUs.
  check    CPU  every clip of --splits has a finite token file of the expected shape (run before any training: the
                trainers never encode frames in the ablation, launch.py refuses cache misses).
  verify   GPU  re-encodes --n clips with V-JEPA 2.1 ViT-L/384 through this code path and compares with TactusNet's own
                cache ($BL_TOKENS): proves that the preprocessing here is the token converter's.

  python -m baselines.ablate.tokens extract --backbone dinov3_l16_384 --out $WORK/ablate/tok/dinov3_l16_384 [--splits train,val,test]
         [--shard 0/1] [--batch 2] [--workers 6] [--limit N] [--amp none|bf16]
  python -m baselines.ablate.tokens check   --backbone dinov3_l16_384 --out $WORK/ablate/tok/dinov3_l16_384
  python -m baselines.ablate.tokens verify  --out $BL_OUT/ablate/verify --n 3

The dir <out> plays the role of $WORK/tok_full in pipeline/make_exp.py (TOK/all/<sid>.npy), so an ablation
experiment dir is laid out by make_exp.py itself (run.sh: ablate_expdir).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

from baselines.common import config
from baselines.common.data import get_export

from . import backbones as BB


def split_sids(ex, splits):
    out = []
    for s in splits.split(","):
        s = s.strip()
        if s: out += ex.sids(s)
    seen = set(); return [s for s in out if not (s in seen or seen.add(s))]


def _shard(sids, shard):
    i, n = (int(v) for v in shard.split("/"))
    if not 0 <= i < n: raise SystemExit(f"bad --shard {shard}")
    return sids[i::n]


def _valid(path, shape):
    try:
        a = np.load(path, mmap_mode="r"); return tuple(a.shape) == tuple(shape) and a.dtype == np.float16
    except Exception:
        return False


def _save(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f: np.save(f, arr)
    os.replace(tmp, path)


class _Frames:
    """Dataset of native protocol frames (uint8 RGB), read by DataLoader workers; resizing happens on the GPU."""
    def __init__(self, ex, sids): self.ex, self.sids = ex, sids
    def __len__(self): return len(self.sids)
    def __getitem__(self, i): s = self.sids[i]; return s, self.ex.frames(s)


def _loader(ex, sids, batch, workers):
    import torch
    return torch.utils.data.DataLoader(_Frames(ex, sids), batch_size=batch, shuffle=False, num_workers=workers,
                                       collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b]),
                                       prefetch_factor=(2 if workers > 0 else None), persistent_workers=False)


def expected_shape(name):
    s = BB.SPECS[name]; return (8, s["grid"], s["grid"], s["dim"])


def cmd_extract(a):
    ex = get_export(a.export, a.hf_raw); out = Path(a.out); (out / "all").mkdir(parents=True, exist_ok=True)
    sids = split_sids(ex, a.splits)
    if a.limit: sids = sids[: a.limit]
    sids = _shard(sids, a.shard); shape = expected_shape(a.backbone)
    todo = [s for s in sids if not _valid(out / "all" / f"{s}.npy", shape)]
    print(f"[tokens] {a.backbone}: {len(sids)} clips in shard {a.shard}, {len(sids) - len(todo)} already done, {len(todo)} to encode", flush=True)
    if todo:
        enc = BB.Encoder(a.backbone, device=a.device, amp=a.amp, tf32=a.tf32)
        t0 = time.time(); n = 0
        for sids_b, frames_b in _loader(ex, todo, a.batch, a.workers):
            for s, tok in zip(sids_b, enc.encode(frames_b)):
                if tuple(tok.shape) != shape: raise SystemExit(f"{s}: tokens {tok.shape}, expected {shape}")
                _save(out / "all" / f"{s}.npy", tok); n += 1
            if n % 200 < len(sids_b): print(f"[tokens] {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/clip", flush=True)
        sec = (time.time() - t0) / max(n, 1)
        desc = enc.describe()
    else:
        sec, desc = None, dict(BB.SPECS[a.backbone], name=a.backbone)
    man = out / f"manifest.shard{a.shard.replace('/', 'of')}.json"
    json.dump(dict(desc, splits=a.splits, shard=a.shard, n_clips=len(sids), seconds_per_clip=sec, shape=list(shape),
                   export=str(ex.root), time=time.strftime("%Y-%m-%d %H:%M:%S")), open(man, "w"), indent=1)
    print(f"[tokens] done; {man}", flush=True)


def cmd_check(a):
    ex = get_export(a.export, a.hf_raw); out = Path(a.out); shape = expected_shape(a.backbone)
    sids = split_sids(ex, a.splits); bad, nonfinite = [], []
    for i, s in enumerate(sids):
        p = out / "all" / f"{s}.npy"
        if not _valid(p, shape): bad.append(s); continue
        if a.full or i % 50 == 0:
            if not np.isfinite(np.load(p)).all(): nonfinite.append(s)
    meta = {m["sid"]: m for m in json.load(open(ex.root / "meta.json"))}
    for sp in a.splits.split(","):               # make_exp.py builds the split lists from meta.json: they must agree
        lm = sorted(m["sid"] for m in meta.values() if m["split"] == sp); le = sorted(ex.sids(sp))
        if lm != le: raise SystemExit(f"split {sp}: all_{sp}.txt ({len(le)}) differs from meta.json ({len(lm)})")
    res = dict(backbone=a.backbone, n=len(sids), missing_or_bad=len(bad), nonfinite=len(nonfinite), examples=bad[:10] + nonfinite[:10])
    print(json.dumps(res)); json.dump(res, open(out / "check.json", "w"), indent=1)
    if bad or nonfinite: raise SystemExit(1)


def cmd_verify(a):
    """V-JEPA 2.1/384 through backbones.Encoder vs TactusNet's cache: differences must be fp16 rounding only."""
    ex = get_export(a.export, a.hf_raw); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sids = ex.sids("test")[: a.n] if not a.sids else a.sids.split(",")
    enc = BB.Encoder("vjepa21_l384", device=a.device, amp="none"); rows = []
    for s in sids:
        mine = enc.encode([ex.frames(s)])[0].astype(np.float64); ref = np.asarray(ex.tokens(s, a.tok_root), np.float64)
        d = np.abs(mine - ref); ulp = np.abs(np.spacing(ref.astype(np.float16)).astype(np.float64))   # fp16 step at each value
        rows.append(dict(sid=s, shape=list(mine.shape), max_abs=float(d.max()), mean_abs=float(d.mean()), ref_abs_mean=float(np.abs(ref).mean()),
                         rel=float(d.mean() / max(np.abs(ref).mean(), 1e-9)), frac_within_1ulp=float((d <= ulp + 1e-12).mean()),
                         cos=float((mine * ref).sum() / (np.linalg.norm(mine) * np.linalg.norm(ref)))))
        print(json.dumps(rows[-1]), flush=True)
    res = dict(rows=rows, ok=all(r["rel"] < 1e-3 and r["cos"] > 0.99999 for r in rows))
    json.dump(res, open(out / "verify_vjepa21_l384.json", "w"), indent=1); print("verify ok" if res["ok"] else "VERIFY FAILED")
    if not res["ok"]: raise SystemExit(1)


def cmd_prefetch(a):
    """Download the pinned weight files of the listed encoders into $HF_HOME (CPU, network); V-JEPA 2.1 needs none."""
    from huggingface_hub import hf_hub_download
    files = {"timm_vit": ["model.safetensors"], "lemonfm": ["lemonfm.pth"], "vjepa2_hf": ["config.json", "model.safetensors"],
             "videomae_hf": ["config.json", "model.safetensors"], "vjepa_repo": [], "endofm": []}
    for n in a.backbones.split():
        s = BB.SPECS[n]
        if s["kind"] == "endofm":                     # Google Drive checkpoint of the cls family, teacher backbone extracted once
            from baselines.cls import weights as W
            print(n, W.endofm_backbone(), flush=True); continue
        for f in files[s["kind"]]: print(n, hf_hub_download(s["repo"], f, revision=s["revision"]), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("prefetch"); pf.add_argument("--backbones", required=True, help="space-separated names of backbones.SPECS")
    for name in ("extract", "check", "verify"):
        p = sub.add_parser(name); p.add_argument("--out", required=True); p.add_argument("--export"); p.add_argument("--hf-raw")
        p.add_argument("--device", default="cuda")
        if name != "verify":
            p.add_argument("--backbone", required=True, choices=sorted(BB.SPECS)); p.add_argument("--splits", default="train,val,test")
        if name == "extract":
            p.add_argument("--shard", default="0/1"); p.add_argument("--batch", type=int, default=2); p.add_argument("--workers", type=int, default=6)
            p.add_argument("--limit", type=int, default=0); p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
            p.add_argument("--tf32", action="store_true", help="TF32 matmuls (default: plain fp32, as the token converter)")
        if name == "check": p.add_argument("--full", action="store_true", help="load every file to test finiteness (default: every 50th)")
        if name == "verify": p.add_argument("--n", type=int, default=3); p.add_argument("--sids", default=""); p.add_argument("--tok-root", default=str(config.TOKENS))
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "check": cmd_check, "verify": cmd_verify, "prefetch": cmd_prefetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
