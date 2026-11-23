#!/usr/bin/env python3
"""Pre-resize the 16 protocol frames of every clip to uint8 arrays, so pod training reads small square arrays instead
of the native-resolution export arrays and never resizes on the fly (CPU, run once per export).

  python -m baselines.cls.pack                      # default plan: train clips at 256 and 320, val/test clips at 224
  python -m baselines.cls.pack --sizes 224 --splits val,test [--workers N]

Sizes: 256 = official pre-crop size of TSN/TSM/VideoMAE/X-CLIP, 320 = that of Endo-FM/MViTv2 (short side jitter
256-320), 224 = the test view of every model. Writes $BL_OUT/cls_cache/s<size>/<sid>.npy (uint8 [16,size,size,3], RGB,
data.squash of ex.frames) and MANIFEST.json (export key = sha1 of meta.json). Idempotent and resumable: complete files
are skipped. The dataset uses an entry only when the manifest matches the export it reads.
Disk: 3.1 MB (256) + 4.9 MB (320) per train clip, 2.4 MB per val/test clip (Gold: about 100 GB + 8 GB).
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

from baselines.common.data import get_export
from . import data as D

PLAN = (("256,320", "train"), ("224", "val,test"))


def _work(job):
    sid, sizes, root = job; ex = get_export(); todo = [s for s in sizes if not D.cache_ok(D.cache_dir(s, root) / f"{sid}.npy", s)]
    if not todo: return sid, 0
    f = ex.frames(sid)
    for s in todo:
        p = D.cache_dir(s, root) / f"{sid}.npy"; tmp = p.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh: np.save(fh, D.squash(f, s))
        os.replace(tmp, p)
    return sid, len(todo)


def pack(ex, sizes, splits, workers, root):
    key = D.export_key(ex); sids = list(dict.fromkeys(s for sp in splits for s in ex.sids(sp)))
    for s in sizes:
        d = D.cache_dir(s, root); d.mkdir(parents=True, exist_ok=True); m = d / "MANIFEST.json"
        old = json.loads(m.read_text()) if m.is_file() else {}
        if old and old.get("export_key") != key:
            raise SystemExit(f"{d} was packed from another export ({old.get('export')}); remove it or use --root")
        m.write_text(json.dumps(dict(old, export=str(ex.root), export_key=key, size=s, resize=D.RESIZE,
                                     layout="uint8 [16,S,S,3] RGB", splits=sorted(set(old.get("splits", [])) | set(splits)),
                                     complete=False), indent=1))
    t0 = time.time(); done = wrote = 0; jobs = [(sid, sizes, root) for sid in sids]
    with Pool(max(1, workers)) as pool:
        for sid, k in pool.imap_unordered(_work, jobs, chunksize=4):
            done += 1; wrote += k
            if done % 500 == 0 or done == len(jobs):
                print(f"[pack] {','.join(splits)} at {sizes}: {done}/{len(jobs)} clips, {wrote} arrays written, {time.time() - t0:.0f}s", flush=True)
    for s in sizes:
        d = D.cache_dir(s, root); m = d / "MANIFEST.json"; man = json.loads(m.read_text())
        n_ok = sum(D.cache_ok(d / f"{sid}.npy", s) for sid in sids)
        man.update(complete=n_ok == len(sids), n_clips=len(list(d.glob("*.npy"))), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        m.write_text(json.dumps(man, indent=1)); print(f"[pack] s{s}: {n_ok}/{len(sids)} clips of {','.join(splits)} complete -> {d}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes"); ap.add_argument("--splits")
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--root", default=str(D.CACHE))
    a = ap.parse_args(argv); ex = get_export()
    plan = PLAN if a.sizes is None and a.splits is None else ((a.sizes or "224", a.splits or "val,test"),)
    for sizes, splits in plan:
        pack(ex, [int(s) for s in sizes.split(",")], splits.split(","), a.workers, a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
