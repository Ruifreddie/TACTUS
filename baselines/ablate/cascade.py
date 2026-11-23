#!/usr/bin/env python3
"""TactusNet's Spatial Head with the tool and action PREDICTED by its Semantic Head (the cascade of Sec. 4), on test.

The benchmark gives the heads the true tool and action. Here the full Spatial Head (stage 1 -> zoom box -> re-encoding ->
stage 2 + contact decoder -> pixel refiner) runs online through UniTAP's own tools/infer_pipeline.py with the predicted
pair instead. Clips whose predicted pair is the true one keep the true-condition outputs of the same pipeline
(<exp>/results/pipeline_test/<sid>.npz, written by pipeline/run_exp.sh evaluate): their inputs are identical.
--verify N re-runs N of those clips to show that the pipeline reproduces its dumps.

Both rows are written from the pipeline outputs (tool points AND contact points of the online pipeline), so they differ
only in the condition. Note: Tab. 3's Where uses the cached-token contact dump (pred_C006_best_test), which matches the
online contact points up to fp16 token rounding; the true-condition row here is the online one.

  python -m baselines.ablate.cascade --exp $WORK/exp/A --cls-pred OUT/ours/cls/pred_test.npz --out OUT/cascade [--verify 8]
writes OUT/{gt,pred}/pred_test.npz (kpt: tool_xy, ct_xy), OUT/dumps/<sid>.npz (re-run clips) and OUT/cascade.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

from baselines.common import preds as P
from baselines.common.data import get_export


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--cls-pred", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--split", default="test"); ap.add_argument("--verify", type=int, default=8)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--limit", type=int, default=0, help="smoke: at most this many re-run clips")
    ap.add_argument("--export"); ap.add_argument("--hf-raw"); a = ap.parse_args(argv)
    E = Path(a.exp).resolve(); out = Path(a.out); (out / "dumps").mkdir(parents=True, exist_ok=True)
    ex = get_export(a.export, a.hf_raw); sids = ex.sids(a.split)
    cp = P.load(a.cls_pred); pp = {s: int(l.argmax()) for s, l in zip(cp["sids"], cp["pair_logits"])}
    miss = [s for s in sids if s not in pp]
    if miss: raise SystemExit(f"{len(miss)} clips without a predicted pair, e.g. {miss[:3]}")
    dump_dir = E / f"results/pipeline_{a.split}"
    changed = [s for s in sids if pp[s] != ex.pair_id(s)]; same = [s for s in sids if pp[s] == ex.pair_id(s)]
    print(f"[cascade] {len(changed)} of {len(sids)} clips get a wrong pair from the Semantic Head", flush=True)
    os.chdir(E); sys.path.insert(0, str(E / "tools"))
    import infer_pipeline as IP          # noqa: E402  (UniTAP's online pipeline; default checkpoints = the experiment's E007k/E009/C006/E008b)
    pipe = IP.Pipeline(device=a.device, refine_decode="full")
    import pixel_refiner as PR           # noqa: E402

    def run(sid, pair):
        tool, act = ex.pairs[pair].split("|"); frames = PR.read_clip_frames(sid)
        r = pipe.run_clip(frames, tool, act)
        return dict(pred_xy=r["tool_xy"].astype(np.float32), pred_xy_stage2=r["tool_xy_stage2"].astype(np.float32),
                    pred_xy_stage1=r["tool_xy_stage1"].astype(np.float32), contact_xy=r["contact_xy"].astype(np.float32), box=np.array(r["box"]))

    rerun = changed[: a.limit] if a.limit else changed; t0 = time.time()
    for i, s in enumerate(rerun):
        f = out / "dumps" / f"{s}.npz"
        if not f.exists(): np.savez_compressed(f, pair=pp[s], **run(s, pp[s]))
        if (i + 1) % 25 == 0: print(f"[cascade] {i + 1}/{len(rerun)} {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)
    ver = []
    for s in same[: a.verify]:                                  # same input -> must reproduce the stored true-condition dump
        r = run(s, ex.pair_id(s)); z = np.load(dump_dir / f"{s}.npz")
        ver.append(dict(sid=s, tool_px=float(np.abs(r["pred_xy"] - z["pred_xy"]).max() * 384), contact_px=float(np.abs(r["contact_xy"] - z["contact_xy"]).max() * 384)))
    gt_t, gt_c, pr_t, pr_c, used = [], [], [], [], []
    for s in sids:
        z = np.load(dump_dir / f"{s}.npz"); gt_t.append(z["pred_xy"]); gt_c.append(z["contact_xy"])
        if s in changed:
            f = out / "dumps" / f"{s}.npz"
            if f.exists(): z = np.load(f); used.append(s)
            elif not a.limit: raise SystemExit(f"{s} has no re-run")
        pr_t.append(z["pred_xy"]); pr_c.append(z["contact_xy"])
    base = {"method": "tactusnet", "split": a.split, "exp": str(E), "source": "online pipeline (tools/infer_pipeline.py)"}
    if a.limit and len(used) < len(changed): base["SMOKE_INCOMPLETE"] = f"{len(changed) - len(used)} wrong-pair clips kept their true-condition output"
    P.save(out / "gt/pred_test.npz", "kpt", sids, dict(base, conditioning="true tool and action"), tool_xy=np.stack(gt_t), ct_xy=np.stack(gt_c))
    P.save(out / "pred/pred_test.npz", "kpt", sids, dict(base, conditioning=f"pair predicted by the Semantic Head ({a.cls_pred})"),
           tool_xy=np.stack(pr_t), ct_xy=np.stack(pr_c))
    rep = dict(n=len(sids), changed=len(changed), rerun=len(used), seconds_per_clip=(time.time() - t0) / max(len(rerun), 1), verify=ver,
               verify_max_tool_px=max([v["tool_px"] for v in ver], default=None), verify_max_contact_px=max([v["contact_px"] for v in ver], default=None))
    json.dump(rep, open(out / "cascade.json", "w"), indent=1); print(json.dumps(rep)[:800])


if __name__ == "__main__":
    sys.exit(main())
