#!/usr/bin/env python3
"""Write the export's video_16_npy for the Gold clips by decoding the raw clips (pod preparation).

The light export on the volume (results/baselines/export_gt.tgz) has no video_16_npy. The fine-tuning loader and UniTAP's
online pipeline read the 16 protocol frames from it, so they are decoded once from the raw clips with
Export.raw_frames, which reproduces the exported frames pixel for pixel (baselines/tests/test_common.py), and stored
uncompressed in BGR, the exporter's byte order. No JPEG cache is involved.

  python -m baselines.ft.frames --splits train,val,test --workers 32 [--check DIR_WITH_REFERENCE_NPY]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_EX = None


def _init(root, hf_raw):
    global _EX
    os.environ.pop("BL_FRAME_CACHE", None)
    from baselines.common.data import get_export
    _EX = get_export(root, hf_raw)


def _one(sid):
    dst = _EX.root / "video_16_npy" / f"{sid}.npy"
    if dst.is_file():
        return sid, 0
    f = _EX.raw_frames(sid, _EX.sampled_abs(sid), rgb=False)          # BGR uint8 [16,H,W,3]
    if f.shape[0] != 16 or f.dtype != np.uint8:
        raise RuntimeError(f"{sid}: frames {f.shape} {f.dtype}")
    tmp = dst.with_name(f".{sid}.{os.getpid()}.npy")
    np.save(tmp, f)
    os.replace(tmp, dst)
    return sid, 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--export")
    ap.add_argument("--hf-raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", help="dir of reference video_16_npy files: compare every clip found there")
    a = ap.parse_args(argv)
    _init(a.export, a.hf_raw)
    ex = _EX
    (ex.root / "video_16_npy").mkdir(exist_ok=True)
    sids = [s for sp in a.splits.split(",") for s in ex.sids(sp.strip())]
    sids = list(dict.fromkeys(sids))[: a.limit or None]
    t0 = time.time()
    n_new = 0
    with Pool(a.workers, initializer=_init, initargs=(a.export, a.hf_raw)) as pool:
        for i, (sid, new) in enumerate(pool.imap_unordered(_one, sids, chunksize=8)):
            n_new += new
            if (i + 1) % 500 == 0:
                print(f"[frames] {i + 1}/{len(sids)} {time.time() - t0:.0f}s", flush=True)
    res = dict(clips=len(sids), written=n_new, seconds=time.time() - t0, root=str(ex.root))
    if a.check:
        bad, n = [], 0
        for s in sids:
            ref = Path(a.check) / f"{s}.npy"
            if ref.is_file():
                n += 1
                d = int(np.abs(np.load(ref).astype(np.int16) - np.load(ex.root / "video_16_npy" / f"{s}.npy").astype(np.int16)).max())
                if d:
                    bad.append((s, d))
        res.update(checked=n, mismatched=len(bad), examples=bad[:5])
    print(json.dumps(res), flush=True)
    if a.check and res.get("mismatched"):
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
