#!/usr/bin/env python3
"""The val sweep of one When method: every config x every seed, selection by mean val tIoU, test reported once.

  python -m baselines.when.sweep --method actionformer [--steps 8] --out $BL_OUT/when/actionformer_s8 [--configs main,b64]
         [--seeds 0,1,2] [--jobs 3] [--device cuda] [--feat-root DIR] [--smoke]

--configs defaults to the method's SWEEP (main = the official batch and LR, b64 = batch 64 with the LR scaled linearly;
QD-DETR: main only); every entry of the method's CONFIGS can be named. --jobs N runs up to N trainings at a time on the
one GPU, as subprocesses that log to <run>/train.log (the heads are launch-bound: 3 concurrent runs give about twice the
throughput of one).

Layout: <out>/<config>/seed<k>/ (one trainer run each, see train.py), <out>/<config>/summary.json (val per seed),
<out>/selection.json (the configs' mean val tIoU, the selected one, and its test numbers per seed with mean and std),
and <out>/pred_{val,test}.npz from the selected config's best validation seed, for scoring with baselines.common.score.
Only the selected config is scored on test. Use --configs main --seeds 0 for the run reported in Table 3(b).
Re-running resumes: finished runs are skipped (their predictions are rewritten from best.pt), interrupted ones resume.
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np

from .methods import METHODS
from .train import aggregate, run_parallel, run_result, run_seeds, test_scores, train_cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(METHODS)); ap.add_argument("--steps", type=int, choices=(8, 32))
    ap.add_argument("--out", required=True); ap.add_argument("--configs", help="default: the method's SWEEP")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="concurrent trainings on the GPU")
    ap.add_argument("--device"); ap.add_argument("--feat-root"); ap.add_argument("--keep-optimizer", action="store_true")
    a = ap.parse_args(argv)
    cls = METHODS[a.method]; steps = a.steps or cls.main_steps; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    configs = a.configs.split(",") if a.configs else list(cls.SWEEP or cls.CONFIGS); seeds = [int(s) for s in a.seeds.split(",")]
    for c in configs:
        if c not in cls.CONFIGS: raise SystemExit(f"{a.method}: unknown config {c!r}, choose from {list(cls.CONFIGS)}")
    summ = {}
    if a.jobs > 1:                                   # all config x seed runs in one pool, then aggregate per config
        jobs = [(train_cmd(a.method, steps, out / c / f"seed{sd}", sd, c, a.set, a.smoke, a.device, a.feat_root, a.keep_optimizer),
                 out / c / f"seed{sd}") for c in configs for sd in seeds]
        failed = run_parallel(jobs, a.jobs)
        if failed: raise SystemExit(f"runs failed: {failed} (see their train.log)")
        for c in configs:
            summ[c] = aggregate(a.method, out / c, seeds, [run_result(out / c / f"seed{sd}") for sd in seeds], score_test=False)
    else:
        for c in configs:
            s = run_seeds(a.method, steps, out / c, seeds, score_test=False, config=c, set_items=a.set, smoke=a.smoke,
                          device=a.device, root=a.feat_root, keep_optimizer=a.keep_optimizer)
            if s is None: return 3
            summ[c] = s
    best = max(configs, key=lambda c: summ[c]["val_tiou_mean"])
    per_seed = []
    for ps in summ[best]["per_seed"]:
        per_seed.append(dict(ps, test=test_scores(out / best / f"seed{ps['seed']}" / "pred_test.npz")))
    t = {k: np.array([p["test"][k] for p in per_seed]) for k in ("tiou", "r7")}
    sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0
    sel = dict(method=a.method, steps=steps, seeds=seeds, configs={c: {"val_tiou_mean": summ[c]["val_tiou_mean"], "val_tiou_std": summ[c]["val_tiou_std"]}
                                                                  for c in configs},
               selected=best, rule="highest mean val tIoU over seeds; test scored only for the selected config",
               selected_per_seed=per_seed, test_tiou_mean=float(t["tiou"].mean()), test_tiou_std=sd(t["tiou"]),
               test_r7_mean=float(t["r7"].mean()), test_r7_std=sd(t["r7"]), representative_seed=summ[best]["representative_seed"])
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    for f in ("pred_val.npz", "pred_test.npz", "info.json", "summary.json"):
        shutil.copyfile(out / best / f, out / f)
    print(f"[sweep] {a.method} s{steps}: selected {best} (val tIoU {summ[best]['val_tiou_mean']:.4f}); test tIoU "
          f"{sel['test_tiou_mean']:.4f} +- {sel['test_tiou_std']:.4f}, R@0.7 {100 * sel['test_r7_mean']:.1f} +- {100 * sel['test_r7_std']:.1f} "
          f"-> {out}/selection.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
