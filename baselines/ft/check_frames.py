#!/usr/bin/env python3
"""Do the decoded frames give experiment A's encoder input? The frozen V-JEPA 2.1 encodes a few Gold clips from
<exp>/data/case32/v1/video_16_npy in fp32 as TactusNet's converter (data.fp32_mode), and the tokens are compared with A's own token cache, the cache A
was trained on (tar shards of <sid>.npy on the volume: $GV/tokens/full).

  python -m baselines.ft.check_frames --exp $EXP_A --shards $GV/tokens/full --n 12 --device cpu --out check_frames.json

Pass: relative error <= --tol for every clip (the fp16 rounding of the cache alone gives about 2e-4). Exit code 3 if not.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import encoder as E


def pick(shards, have, n):
    """n clips that have frames here, spread over the first shard that holds enough of them (members are <sid>.npy)."""
    tars = sorted(Path(shards).glob("full_*.tar"))
    if not tars:
        raise SystemExit(f"no token shards full_*.tar in {shards}")
    for t in tars:
        names = subprocess.run(["tar", "-tf", str(t)], capture_output=True, text=True, check=True).stdout.split()
        sids = sorted(m[:-4] for m in names if m.endswith(".npy") and m[:-4] in have)
        if len(sids) >= n:
            step = max(1, len(sids) // n)
            return t, sids[::step][:n]
    raise SystemExit(f"no shard in {shards} holds {n} clips that have frames")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="restored experiment A")
    ap.add_argument("--shards", required=True, help="A's token cache as tar shards ($GV/tokens/full)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vjepa-ckpt", default=os.environ.get("VJEPA_CKPT"))
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="a root with video_16_npy/ (default <exp>/data/case32/v1)")
    ap.add_argument("--fp32", default=None, choices=("converter", "strict", "both"),
                    help="precision of the check (default FT_FP32, i.e. data.fp32_mode); both: report the two, pass on FT_FP32's")
    a = ap.parse_args(argv)
    t0 = time.time()
    v1 = Path(a.frames) if a.frames else Path(a.exp) / "data/case32/v1"
    have = {p.stem for p in (v1 / "video_16_npy").glob("*.npy")}
    shard, sids = pick(a.shards, have, a.n)
    primary = D.fp32_mode() if a.fp32 in (None, "both") else a.fp32
    modes = [primary] + ([m for m in ("converter", "strict") if m != primary] if a.fp32 == "both" else [])
    tmp = Path(tempfile.mkdtemp(prefix="check_frames_", dir=os.environ.get("TMPDIR")))
    by_mode = {}
    try:
        (tmp / "ref").mkdir()
        subprocess.run(["tar", "-xf", str(shard), "-C", str(tmp / "ref")] + [f"{s}.npy" for s in sids], check=True)
        if a.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 1)
        enc = E.Encoder(a.exp, a.vjepa_ckpt, trainable=False).to(a.device)
        for mode in modes:
            os.environ["FT_FP32"] = mode                 # write_tokens sets the flags of this mode and restores them
            D.write_tokens(enc, sids, v1, tmp / mode, batch=1, workers=2, amp=False, device=a.device)
            rows = []
            for s in sids:
                ref = np.load(tmp / "ref" / f"{s}.npy").astype(np.float64)
                our = np.load(tmp / mode / f"{s}.npy").astype(np.float64)
                if ref.size != our.size:
                    rows.append(dict(sid=s, rel=float("inf"), cos=0.0, max_abs=float("inf"), shapes=[list(ref.shape), list(our.shape)]))
                    continue
                ref, our = ref.reshape(-1, ref.shape[-1]), our.reshape(-1, our.shape[-1])
                nr, no = float(np.linalg.norm(ref)), float(np.linalg.norm(our))
                rows.append(dict(sid=s, rel=float(np.linalg.norm(our - ref)) / nr, cos=float((our * ref).sum()) / (nr * no),
                                 max_abs=float(np.abs(our - ref).max())))
            by_mode[mode] = dict(worst_rel=max(r["rel"] for r in rows), cos_min=min(r["cos"] for r in rows), clips=rows)
    finally:
        os.environ["FT_FP32"] = primary
        shutil.rmtree(tmp, ignore_errors=True)
    worst = by_mode[primary]["worst_rel"]
    ok = bool(worst <= a.tol)
    res = dict(ok=ok, worst_rel=worst, tol=a.tol, fp32=primary, shard=str(shard), device=a.device,
               torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu",
               modes=by_mode, seconds=time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for m, r in by_mode.items():
        print(f"[check_frames] fp32 {m}: worst relative error {r['worst_rel']:.2e}, cos min {r['cos_min']:.7f} over {len(sids)} clips",
              flush=True)
    print(f"[check_frames] {'OK' if ok else 'MISMATCH'} ({primary}, tol {a.tol:.0e}), {res['seconds']:.0f} s", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
