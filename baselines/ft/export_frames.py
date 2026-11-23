#!/usr/bin/env python3
"""Experiment A's exact frames: video_16_npy of the Gold clips, taken from A's full export on the volume ($GV/export, tar
shards of about 4 GB written by pipeline/volume_shards.py with the members sorted by path, so the frames form the
tail of the archive and run in clip order). Frames decoded again on another CPU differ slightly (check_frames: relative
token error 1e-3 to 4e-3), so the fine-tuning runs read these.

Only the shards that hold a wanted clip are read (from the first member of every shard); tar reads each one in order (a
seek costs a round trip on the volume) and stops after its last wanted member. Several shards at a time, with retries; a
shard is marked done when its wanted members are written, so a re-run only does what is missing.

  python -m baselines.ft.export_frames --shards $GV/export --root $EXPORT [--out DIR] [--jobs 6] [--only-sids a,b]
"""
import argparse
import bisect
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRE = "video_16_npy/"


def gold_sids(root):
    from baselines.common.data import get_export
    ex = get_export(str(root), os.environ.get("BL_HF_RAW") or os.environ.get("HF_RAW"))
    return sorted({s for sp in ("train", "val", "test") for s in ex.sids(sp)})


def retry(fn, what, n=6):
    for i in range(n):
        try:
            return fn()
        except Exception as ex:                 # the volume client can fail under load
            print(f"[export_frames] retry {what}: {repr(ex)[:160]}", flush=True)
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"gave up on {what}")


def first_member(shard):
    def f():
        with tarfile.open(shard, "r:") as t:
            return t.next().name
    return retry(f, f"first member of {shard.name}")


def vid(name):
    return name[len(PRE):-4] if name.startswith(PRE) and name.endswith(".npy") else None


def extract(shard, names, out, mark):
    """names: the wanted members in this shard's range of clips. tar reads the shard in order (a seek costs a round trip on
    the volume, a sequential read does not) and stops after the last of them (--occurrence)."""
    m = mark / f"{shard.name}.done"
    if m.exists():
        return json.loads(m.read_text())["n"]
    lst = mark / f"{shard.name}.list"
    lst.write_text("\n".join(names) + "\n")

    def f():           # dd reads 16 MB blocks: tar's own 10 KB reads cost one round trip each on the volume
        dd = subprocess.Popen(["dd", f"if={shard}", "bs=16M", "status=none"], stdout=subprocess.PIPE)
        try:
            subprocess.run(["tar", "-xf", "-", "-C", str(out), "--occurrence=1", "-T", str(lst)], stdin=dd.stdout, check=True)
        finally:
            dd.stdout.close()
            dd.kill()
            dd.wait()
        miss = [n for n in names if not (out / n).is_file()]
        if miss:
            raise IOError(f"{len(miss)} members missing after tar, e.g. {miss[:3]}")
    retry(f, shard.name)
    m.write_text(json.dumps({"n": len(names)}))
    return len(names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="A's export on the volume ($GV/export)")
    ap.add_argument("--root", required=True, help="the light export here (split lists of the Gold clips)")
    ap.add_argument("--out", default=None, help="where video_16_npy/ goes (default --root)")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("VOL_JOBS", 6)))
    ap.add_argument("--only-sids", default="", help="test: these clips only")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out or a.root)
    (out / PRE).mkdir(parents=True, exist_ok=True)
    mark = out / ".export_frames"
    mark.mkdir(exist_ok=True)
    want = sorted(x for x in a.only_sids.split(",") if x) if a.only_sids else gold_sids(a.root)
    man = json.load(open(Path(a.shards) / "MANIFEST.json"))
    shards = [Path(a.shards) / f"export_{i:05d}.tar" for i in range(man["shards"])]
    with ThreadPoolExecutor(a.jobs) as ex:
        firsts = list(ex.map(first_member, shards))
    need = []
    for i, s in enumerate(shards):
        hi = vid(firsts[i + 1]) if i + 1 < len(shards) else "~"
        if hi is None:                              # the next shard still starts before the frames: none in this one
            continue
        lo = vid(firsts[i]) or ""
        names = [f"{PRE}{x}.npy" for x in want[bisect.bisect_left(want, lo):bisect.bisect_left(want, hi)]]
        if names:
            need.append((s, names))
    print(f"[export_frames] {len(want)} clips wanted, {len(need)} of {len(shards)} shards hold them", flush=True)
    done = [0]

    def job(sn):
        n = extract(sn[0], sn[1], out, mark)
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(need):
            print(f"[export_frames] {done[0]}/{len(need)} shards, {time.time() - t0:.0f} s", flush=True)
        return n
    with ThreadPoolExecutor(a.jobs) as ex:
        got = sum(ex.map(job, need))
    present = sum(1 for s in want if (out / PRE / f"{s}.npy").is_file())
    rep = dict(wanted=len(want), extracted=got, present=present, shards_read=len(need), seconds=time.time() - t0,
               source=str(a.shards))
    json.dump(rep, open(out / "export_frames.json", "w"), indent=1)
    print("[export_frames]", json.dumps(rep), flush=True)
    if present != len(want):
        missing = [s for s in want if not (out / PRE / f"{s}.npy").is_file()]
        raise SystemExit(f"{len(missing)} wanted clips are not in the export, e.g. {missing[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
