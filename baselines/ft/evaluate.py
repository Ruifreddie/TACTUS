#!/usr/bin/env python3
"""Score one epoch snapshot of a fine-tuning run exactly like TactusNet's rows of Tables 3 and 4.

  python -m baselines.ft.evaluate all --exp $EXP_A --run-dir $FT/runs/joint --epoch select:combined --out $FT/eval/joint
      --epoch N | select:<combined|semantic|temporal|spatial>   (selection.json of the run)

Parts (each a separate process, because the two trainer snapshots both define the module `train`):
  tokens   fp32 tokens (TactusNet's converter: FT_FP32, data.fp32_mode) of the snapshot's encoder for the full frames of --split
  heads    Semantic / Temporal Head: baselines.ablate.heads on those tokens -> <out>/cls|when/pred_<split>.npz
  spatial  Spatial Head: UniTAP's online pipeline (tools/infer_pipeline.py: stage 1 -> zoom box from its own points ->
           re-encoding of the crop -> zoom and contact decoders -> pixel refiner of A) with the snapshot's encoder and
           decoders -> <out>/pipeline_<split>/<sid>.npz and <out>/kpt[_stage1|_stage2]/pred_<split>.npz
  score    baselines.common.score with bootstrap CIs -> <out>/metrics_<part>_<split>.json, and <out>/summary_<split>.json
The true tool and action condition When, How and Where, as in the benchmark protocol.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PARTS = ("tokens", "heads", "spatial", "score")


def epoch_dir(run_dir, epoch):
    run_dir = Path(run_dir)
    if str(epoch).startswith("select:"):
        key = str(epoch).split(":", 1)[1]
        sel = json.load(open(run_dir / "selection.json"))["selection"]
        if key not in sel:
            raise SystemExit(f"{run_dir}/selection.json has no '{key}' (has {sorted(sel)})")
        epoch = sel[key]
    d = run_dir / "epochs" / f"e{int(epoch):02d}"
    if not d.is_dir():
        raise SystemExit(f"no snapshot {d}")
    return d, int(epoch)


def _env_py():
    return os.environ.get("BL_PY") or sys.executable


def part_tokens(a, ed):
    import torch
    from . import data as D
    from . import encoder as E
    D.set_fp32_flags()
    v1 = Path(a.exp) / "data/case32/v1"
    sids = [l.strip() for l in open(v1 / f"{a.split}.txt") if l.strip()]
    enc = E.Encoder(a.exp, a.vjepa_ckpt, trainable=False).to(a.device)
    src = "frozen V-JEPA 2.1"
    if (ed / "encoder.pt").exists():
        sd = torch.load(ed / "encoder.pt", map_location="cpu", weights_only=False)["weights"]
        enc.net.load_state_dict(sd, strict=True)
        src = str(ed / "encoder.pt")
    t0 = time.time()
    n = D.write_tokens(enc, sids, v1, Path(a.out) / "tok" / a.split, batch=a.batch, workers=a.workers, amp=False,
                       device=a.device, skip_existing=True)
    json.dump(dict(encoder=src, split=a.split, clips=len(sids), written=n, seconds=time.time() - t0, precision=f"fp32 ({D.fp32_mode()})"),
              open(Path(a.out) / f"tokens_{a.split}.json", "w"), indent=1)


def part_heads(a, ed):
    kc = a.kc_rng or os.environ.get("KC_RNG", "")
    kc = str(Path(kc).parent) if Path(kc).name == "kpt_combo" else kc
    v1 = Path(a.exp) / "data/case32/v1"
    want = set(a.only.split(",")) if a.only else {"sem", "tmp"}
    for name, kind in (("sem", "cls"), ("tmp", "when")):
        if not (ed / f"{name}.pt").exists() or name not in want:
            continue
        out = Path(a.out) / kind / f"pred_{a.split}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [_env_py(), "-m", "baselines.ablate.heads", "--kind", kind, "--code", kc, "--ckpt", str(ed / f"{name}.pt"),
               "--root", str(v1), "--split", a.split, "--out", str(out), "--cache", str(Path(a.out) / "tok"), "--device", a.device]
        subprocess.run(cmd, check=True)


def part_spatial(a, ed):
    """In its own process: UniTAP's online pipeline with the snapshot's encoder and decoders (A's refiner)."""
    import torch
    from baselines.common import preds as P
    from baselines.common.data import get_export
    from . import data as D
    D.set_fp32_flags()                     # FT_FP32: PyTorch's defaults, as A's own pipeline ran (or strict)
    if not all((ed / f"{n}.pt").exists() for n in ("s1", "zm", "ct")):
        return
    E = Path(a.exp)
    ex = get_export(a.export, a.hf_raw)
    sids = ex.sids(a.split)
    out = Path(a.out)
    dd = out / f"pipeline_{a.split}"
    dd.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(E)
    sys.path.insert(0, str(E / "tools"))
    import infer_pipeline as IP          # noqa: E402  (UniTAP; its defaults = the experiment's own checkpoints)
    import pixel_refiner as PR           # noqa: E402
    pipe = IP.Pipeline(ckpts={"stage1": str(ed / "s1.pt"), "stage2": str(ed / "zm.pt"), "contact": str(ed / "ct.pt")},
                       device=a.device, refine_decode="full")
    src = "frozen V-JEPA 2.1"
    if (ed / "encoder.pt").exists():
        sd = torch.load(ed / "encoder.pt", map_location="cpu", weights_only=False)["weights"]
        pipe.vjepa.encoder.load_state_dict(sd, strict=True)
        src = str(ed / "encoder.pt")
    t0 = time.time()
    for i, s in enumerate(sids):
        f = dd / f"{s}.npz"
        if f.exists():
            continue
        tool, act = ex.pairs[ex.pair_id(s)].split("|")
        r = pipe.run_clip(PR.read_clip_frames(s), tool, act)
        np.savez_compressed(f, pred_xy=r["tool_xy"].astype(np.float32), pred_xy_stage2=r["tool_xy_stage2"].astype(np.float32),
                            pred_xy_stage1=r["tool_xy_stage1"].astype(np.float32), contact_xy=r["contact_xy"].astype(np.float32),
                            box=np.array(r["box"]))
        if (i + 1) % 100 == 0:
            print(f"[ft-eval] {i + 1}/{len(sids)} {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)
    os.chdir(cwd)

    def _load(f):                          # read and close: an open NpzFile per clip exhausts the open-file limit
        with np.load(f) as d:
            return {k: d[k] for k in d.files}
    z = {s: _load(dd / f"{s}.npz") for s in sids}
    info = dict(method="tactusnet-ft", snapshot=str(ed), encoder=src, split=a.split, source="online pipeline (tools/infer_pipeline.py)",
                conditioning="true tool and action", tokens=f"fp32 ({D.fp32_mode()})")
    P.save(out / "kpt" / f"pred_{a.split}.npz", "kpt", sids, info, tool_xy=np.stack([z[s]["pred_xy"] for s in sids]),
           ct_xy=np.stack([z[s]["contact_xy"] for s in sids]))
    for st, key in (("stage1", "pred_xy_stage1"), ("stage2", "pred_xy_stage2")):
        P.save(out / f"kpt_{st}" / f"pred_{a.split}.npz", "kpt", sids, dict(info, stage=st), tool_xy=np.stack([z[s][key] for s in sids]))


def part_score(a, ed):
    from baselines.common import preds as P
    from baselines.common import score as SC
    from baselines.common.data import get_export
    ex = get_export(a.export, a.hf_raw)
    out = Path(a.out)
    summ = dict(snapshot=str(ed), split=a.split)
    for part in ("cls", "when", "kpt", "kpt_stage1", "kpt_stage2"):
        f = out / part / f"pred_{a.split}.npz"
        if not f.exists():
            continue
        res = SC.score_file(P.load(f), ex, split=a.split, boot=True)
        json.dump(res, open(out / f"metrics_{part}_{a.split}.json", "w"), indent=1)
        summ[part] = {k: v for k, v in res["subsets"]["all"].items() if k not in ("per_pair_acc",)}
    json.dump(summ, open(out / f"summary_{a.split}.json", "w"), indent=1)
    print(json.dumps(summ)[:2000])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=PARTS + ("all",))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epoch", default="select:combined")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--kc-rng", default=None)
    ap.add_argument("--vjepa-ckpt", default=os.environ.get("VJEPA_CKPT"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--export")
    ap.add_argument("--hf-raw")
    ap.add_argument("--only", default="", help="comma list of sem,tmp,spatial (default: every head of the snapshot)")
    a = ap.parse_args(argv)
    a.exp = str(Path(a.exp).resolve())
    ed, ep = epoch_dir(a.run_dir, a.epoch)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(run_dir=str(a.run_dir), epoch=ep, snapshot=str(ed), selected_by=str(a.epoch)), open(Path(a.out) / "snapshot.json", "w"), indent=1)
    if a.part != "all":
        return {"tokens": part_tokens, "heads": part_heads, "spatial": part_spatial, "score": part_score}[a.part](a, ed)
    base = [_env_py(), "-m", "baselines.ft.evaluate"]
    rest = ["--exp", a.exp, "--run-dir", str(a.run_dir), "--epoch", str(ep), "--out", a.out, "--split", a.split, "--device", a.device,
            "--batch", str(a.batch), "--workers", str(a.workers)]
    for k in ("kc_rng", "vjepa_ckpt", "export", "hf_raw"):
        v = getattr(a, k)
        if v:
            rest += [f"--{k.replace('_', '-')}", str(v)]
    want = set(a.only.split(",")) if a.only else {"sem", "tmp", "spatial"}
    if a.only:
        rest += ["--only", a.only]
    # a re-run after a failure skips the token pass when the head predictions it feeds are already there
    has_heads = any((ed / f"{n}.pt").exists() and n in want and not (Path(a.out) / kind / f"pred_{a.split}.npz").exists()
                    for n, kind in (("sem", "cls"), ("tmp", "when")))
    parts = (["tokens", "heads"] if has_heads else []) + (["spatial"] if "spatial" in want else []) + ["score"]
    for part in parts:
        subprocess.run(base + [part] + rest, check=True)
        if part == "heads":           # the test tokens (about 14 GB per snapshot) are only read by the two heads
            import shutil
            shutil.rmtree(Path(a.out) / "tok", ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
