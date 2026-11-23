#!/usr/bin/env python3
"""Pool TactusNet's token shards ($GV/tokens/full/*.tar: 500 flat <sid>.npy each, [8,24,24,1024] fp16) into the 8-step
When features ($BL_WHEN_FEATS/s8/<sid>.npy, [8,1024] fp16), streaming each shard once instead of unpacking ~200 GB.
Same arithmetic as baselines/when/features.py pool: float32 mean over the 24x24 grid, saved as fp16.

  python -m baselines.pod.pool_shards --shards $GV/tokens/full --out $BL_WHEN_FEATS/s8 [--splits train,val,test] [--jobs 6]
"""
import argparse, glob, io, os, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def pool_one(args):
    shard, out, keep = args; out = Path(out); n = 0
    for attempt in range(6):                                   # the volume's client can fail under load
        try:
            with tarfile.open(shard, "r|") as t:
                for m in t:
                    if not m.isfile() or not m.name.endswith(".npy"): continue
                    sid = os.path.basename(m.name)[:-4]
                    if keep is not None and sid not in keep: continue
                    dst = out / f"{sid}.npy"
                    if dst.exists(): continue
                    a = np.load(io.BytesIO(t.extractfile(m).read()))
                    if a.shape != (8, 24, 24, 1024): raise ValueError(f"{sid}: {a.shape}")
                    tmp = out / f".{sid}.tmp"
                    with open(tmp, "wb") as f: np.save(f, a.astype(np.float32).mean(axis=(1, 2)).astype(np.float16))
                    os.replace(tmp, dst); n += 1
            return shard, n
        except Exception as ex:
            print("retry", shard, repr(ex)[:160], flush=True); time.sleep(20 * (attempt + 1))
    raise RuntimeError("cannot read " + shard)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val,test"); ap.add_argument("--jobs", type=int, default=int(os.environ.get("VOL_JOBS", 6)))
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = None
    if a.splits != "all":
        from baselines.common.data import get_export
        ex = get_export(); keep = set(s for sp in a.splits.split(",") for s in ex.sids(sp))
    shards = sorted(glob.glob(os.path.join(a.shards, "*.tar"))); t0 = time.time(); total = 0
    print(f"pool: {len(shards)} shards -> {out} ({'all clips' if keep is None else f'{len(keep)} clips'})", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex_:
        for i, (s, n) in enumerate(ex_.map(pool_one, [(s, str(out), keep) for s in shards])):
            total += n; print(f"[{i + 1}/{len(shards)}] {os.path.basename(s)}: {n} new ({time.time() - t0:.0f} s)", flush=True)
    have = len(glob.glob(str(out / "*.npy")))
    missing = [] if keep is None else sorted(k for k in keep if not (out / f"{k}.npy").exists())
    print(f"done: {total} pooled now, {have} files, {len(missing)} missing", missing[:5], flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
