#!/usr/bin/env python3
"""Train one When baseline on Gold train, keep the best epoch by val tIoU, predict val and test (trainer contract).

  python -m baselines.when.train --method actionformer --steps 8 --out $BL_OUT/when/actionformer_s8/main/seed0
  python -m baselines.when.train --method mstcn --out OUT --seeds 0,1,2            # 3 seeds -> OUT/seed{k}, OUT/summary.json
  python -m baselines.when.train --method qddetr --out OUT --smoke                 # a few steps on tiny batches

  --method   actionformer | tridet | qddetr | mstcn          --steps 8 | 32 (default: the method's main-table setting)
  --config   one of the method's val-sweep configs (default main; see README)   --set key=value  override any recipe entry
  --seed S / --seeds 0,1,2   --device cuda|cpu   --feat-root DIR   --fresh (discard last.pt)   --stop-after N (testing aid)

Contract (baselines/README.md): resumes from <out>/last.pt when it exists (same method, steps, recipe and seed, else
an error unless --fresh), keeps <out>/best.pt by val tIoU (common.train_utils.Checkpointer), writes <out>/log.jsonl
(one line per epoch), <out>/pred_val.npz and <out>/pred_test.npz from best.pt (common.preds, kind "when", with the
fallback flags), and <out>/info.json with the recipe actually used. Train = Gold train clips with an annotated interval;
val and test predictions cover every clip of the split; test is never used for any choice.
With --seeds the representative <out>/pred_{val,test}.npz are those of the seed with the best val tIoU, and
<out>/summary.json has every seed's val and test numbers with their mean and std (the Tab. 3 entry is mean +- std).
"""
import argparse, json, math, random, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

from baselines.common import metrics as M
from baselines.common import preds as P
from baselines.common.data import get_export
from baselines.common.train_utils import Checkpointer, JsonlLog, set_seed

from .data import Split, feat_root, n_frames
from .methods import METHODS

SMOKE_TRAIN_CLIPS = 64
EVAL_BATCH = 512
_DATA = {}


def parse_set(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not _: raise SystemExit(f"--set expects key=value, got {it!r}")
        try: out[k] = json.loads(v)
        except json.JSONDecodeError: out[k] = v
    return out


def load_data(ex, root, steps, device, smoke=False):
    key = (str(root), steps, str(device), smoke)
    if key not in _DATA:
        splits = {s: Split(ex, root, steps, s, standin=smoke) for s in ("train", "val", "test")}
        n = sum(sp.n_standin for sp in splits.values())
        if n: print(f"[smoke] {n} clips without real {steps}-step features use 8-step features interpolated to {steps} steps "
                    "(smoke runs only; a real run requires extract32)", flush=True)
        for sp in splits.values():
            sp.xt = torch.from_numpy(sp.x).to(device); sp.pt = torch.from_numpy(sp.pair).to(device)
            sp.st = torch.from_numpy(sp.seg).to(device)
        _DATA[key] = splits
    return _DATA[key]


def text_feats(root):
    from .features import text_path
    from .methods.qddetr import PAIR_TEMPLATES
    p = text_path(root)
    if not p.is_file(): raise SystemExit(f"missing {p}: run python -m baselines.when.features text --feat-root {root}")
    z = np.load(p); pairs = [str(s) for s in z["pairs"]]; tmpl = [str(s) for s in z["templates"]]
    if tmpl != [PAIR_TEMPLATES[n] for n in pairs]: raise SystemExit(f"{p} was built from other templates than methods/qddetr.py; rebuild with --force")
    return [z[f"pair{i}"] for i in range(len(pairs))], dict(zip(pairs, tmpl))


@torch.no_grad()
def predict_split(method, sp):
    st, en, fb = [], [], []
    for i in range(0, len(sp), EVAL_BATCH):
        s, e, f = method.predict(sp.xt[i:i + EVAL_BATCH].float(), sp.pt[i:i + EVAL_BATCH])
        st.append(s); en.append(e); fb.append(f)
    cat = lambda v, d: np.concatenate(v) if v else np.zeros(0, d)
    return cat(st, np.float32), cat(en, np.float32), cat(fb, bool)


def scores(ex, sp, start, end, fallback):
    acc = M.score_when(ex, sp.sids, start, end)
    return {"tiou": acc["tiou"].value(), "r3": acc["r3"].value(), "r5": acc["r5"].value(), "r7": acc["r7"].value(),
            "n": acc["tiou"].n(), "fallback_rate": float(np.mean(fallback)) if len(fallback) else 0.0}


def _slim(path):
    """After a finished run the optimizer and scheduler are dead weight; drop them (atomic rewrite)."""
    if not path.is_file(): return
    s = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("optimizer", "sched", "rng_torch", "rng_cuda"): s.pop(k, None)
    s["slim"] = True; tmp = str(path) + ".tmp"; torch.save(s, tmp); Path(tmp).replace(path)


def run_one(method_name, steps, out, config="main", overrides=None, seed=0, smoke=False, device=None, root=None,
            fresh=False, stop_after=None, keep_optimizer=False):
    """One training run; returns a dict with the best val metrics and the prediction paths (None if stopped early)."""
    t_start = time.time(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cls = METHODS[method_name]; steps = steps or cls.main_steps; n_frames(steps)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); root = feat_root(root)
    recipe = cls.make_recipe(steps, config, overrides, smoke)
    meta = dict(method=method_name, steps=steps, config=config, recipe=recipe, seed=seed, smoke=smoke)
    if fresh:
        for f in ("last.pt", "best.pt", "log.jsonl", "info.json", "pred_val.npz", "pred_test.npz"): (out / f).unlink(missing_ok=True)

    ex = get_export(); data = load_data(ex, root, steps, device, smoke); tr, va, te = data["train"], data["val"], data["test"]
    train_idx = np.flatnonzero(tr.valid)
    if smoke: train_idx = train_idx[:SMOKE_TRAIN_CLIPS]
    extra = {}
    if method_name == "qddetr":
        tf, templates = text_feats(root); extra["text_feats"] = tf; meta["templates"] = templates
    meta = json.loads(json.dumps(meta))                                       # plain JSON types, comparable after a reload
    set_seed(seed); torch.backends.cudnn.benchmark = False
    method = cls(recipe, steps, len(ex.pairs), device, **extra); method.build()
    B = method.batch_size(); drop_last = method_name in ("actionformer", "tridet") and len(train_idx) >= B   # as their loaders
    ipe = len(train_idx) // B if drop_last else math.ceil(len(train_idx) / B)
    opt, sched = method.make_optimizer(ipe); E = method.total_epochs()

    ck = Checkpointer(out, "tiou", "max"); log = JsonlLog(out / "log.jsonl"); start_epoch = 0
    if (out / "last.pt").exists():
        s = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if s.get("meta") != meta:
            old = s.get("meta") or {}; diff = [k for k in meta if k != "recipe" and old.get(k) != meta[k]]
            diff += [f"recipe.{k}" for k in sorted(set(meta["recipe"]) | set(old.get("recipe", {})))
                     if old.get("recipe", {}).get(k) != meta["recipe"].get(k)]
            raise SystemExit(f"{out}/last.pt belongs to another run (differs in {', '.join(diff) or 'unknown'}); "
                             "use --fresh or another --out")
        if s.get("slim"):
            start_epoch = int(s["epoch"]) + 1; ck.best, ck.best_epoch = s["best"], s["best_epoch"]
        else:
            start_epoch = ck.resume(method.model, opt); method.load_state(s)
            if sched is not None and s.get("sched") is not None: sched.load_state_dict(s["sched"])
            if s.get("rng_torch") is not None: torch.set_rng_state(s["rng_torch"])
            if s.get("rng_cuda") is not None and device.type == "cuda": torch.cuda.set_rng_state(s["rng_cuda"], device)
        print(f"[{method_name}] resumed from {out}/last.pt at epoch {start_epoch}/{E} (best val tIoU {ck.best:.4f} @ {ck.best_epoch})", flush=True)
    print(f"[{method_name}] steps={steps} config={config} seed={seed} train={len(train_idx)} val={len(va)} test={len(te)} "
          f"batch={B} iters/epoch={ipe} epochs={E} params={method.n_params() / 1e6:.2f}M device={device}", flush=True)

    ran = 0
    for epoch in range(start_epoch, E):
        t0 = time.time(); method.train_mode(); rng = np.random.default_rng([seed, epoch]); perm = rng.permutation(train_idx)
        random.seed(seed * 1000003 + epoch)                                      # the official crop (truncate_feats) uses `random`
        agg = {}
        for it in range(ipe):
            idx = torch.as_tensor(perm[it * B:(it + 1) * B], device=device)
            loss, logs = method.loss(tr.xt[idx].float(), tr.pt[idx], tr.st[idx], rng)
            opt.zero_grad(set_to_none=True); loss.backward()
            if method.grad_clip(): torch.nn.utils.clip_grad_norm_(method.model.parameters(), method.grad_clip())
            opt.step()
            if sched is not None and cls.sched_per == "iter": sched.step()
            method.after_step()
            for k, v in logs.items(): agg[k] = agg.get(k, 0.0) + v / ipe
        if sched is not None and cls.sched_per == "epoch": sched.step()
        val = scores(ex, va, *predict_split(method, va))
        state = dict(method.state(), optimizer=opt.state_dict(), sched=sched.state_dict() if sched is not None else None,
                     rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None, meta=meta)
        better = ck.step(epoch, val, state)
        row = dict(epoch=epoch, lr=float(opt.param_groups[0]["lr"]), train=agg, val=val, best=float(ck.best), best_epoch=int(ck.best_epoch),
                   sec=round(time.time() - t0, 2))
        log.write(**row)
        print(f"[{method_name}] epoch {epoch + 1}/{E} loss {agg.get('final_loss', agg.get('loss', float('nan'))):.4f} val tIoU {val['tiou']:.4f} "
              f"R@0.7 {val['r7']:.3f} fb {val['fallback_rate']:.3f}{' *' if better else ''} ({row['sec']:.1f} s)", flush=True)
        ran += 1
        if stop_after and ran >= stop_after and epoch + 1 < E:
            print(f"[{method_name}] --stop-after {stop_after}: stopping after epoch {epoch + 1} (re-run the same command to resume)")
            return None

    # predictions from best.pt
    s = torch.load(out / "best.pt", map_location="cpu", weights_only=False); method.load_state(s)
    res = {"method": method_name, "steps": steps, "config": config, "seed": seed, "best_epoch": int(s["best_epoch"]), "out": str(out)}
    pinfo = dict(method=method_name, title=cls.title, steps=steps, config=config, seed=seed, checkpoint=str(out / "best.pt"),
                 best_epoch=int(s["best_epoch"]), features=f"frozen V-JEPA 2.1 ViT-L/384, {steps} steps", conditioning=cls.conditioning,
                 decoding=cls.decoding, smoke=smoke)
    for sp in (va, te):
        st, en, fb = predict_split(method, sp)
        P.save(out / f"pred_{sp.name}.npz", "when", sp.sids, dict(pinfo, split=sp.name), start=st, end=en, fallback=fb)
        if sp is va: res["val"] = scores(ex, va, st, en, fb)
    res["pred_val"], res["pred_test"] = str(out / "pred_val.npz"), str(out / "pred_test.npz")
    hist = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines() if l.strip()]
    info = dict(
        method=method_name, title=cls.title, paper=cls.paper, code=cls.code_url, steps=steps, config=config, seed=seed, smoke=smoke,
        features=dict(encoder="V-JEPA 2.1 ViT-L/384 (TactusNet's, frozen), final layer, mean over the 24x24 grid", steps=steps,
                      frames=n_frames(steps), sampling=("the 16 protocol frames (TactusNet's cached tokens)" if steps == 8 else
                                                        "64 frames round(linspace(abs_start, abs_end, 64)), 4 chunks of 16"),
                      step_centre=f"(2k + 0.5) / {n_frames(steps) - 1} on the window axis", input=f"{steps} x 1024", root=str(root)),
        initialization="head trained from scratch (random init); features frozen", frozen=True, conditioning=cls.conditioning,
        decoding=cls.decoding, official={"source": f"{cls.code_url} {cls.official_source}", **cls.OFFICIAL},
        adaptations=cls.adaptations(steps, recipe), recipe=recipe, epochs=E, batch=B, iters_per_epoch=ipe, lr=recipe.get("lr"),
        smoke_standin_clips=sum(sp.n_standin for sp in (tr, va, te)),
        optimizer={"actionformer": "AdamW, linear warmup + cosine per iteration, EMA 0.999 for eval",
                   "tridet": "AdamW, linear warmup + cosine per iteration, EMA 0.999 for eval",
                   "qddetr": "AdamW, StepLR(lr_drop) per epoch", "mstcn": "Adam, constant LR"}[method_name],
        n_params=method.n_params(), n_train=int(len(train_idx)), n_val=len(va), n_test=len(te), best_epoch=int(s["best_epoch"]),
        best_val=res["val"], device=str(device) + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
        torch=torch.__version__, wall_clock_s=round(time.time() - t_start, 1), train_s=round(sum(h.get("sec", 0) for h in hist), 1),
        complete=True, templates=meta.get("templates"))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    if not keep_optimizer and not smoke:
        _slim(out / "last.pt"); _slim(out / "best.pt")
    print(f"[{method_name}] best epoch {res['best_epoch'] + 1}: val tIoU {res['val']['tiou']:.4f} R@0.7 {res['val']['r7']:.3f} "
          f"fallback {res['val']['fallback_rate']:.3f} -> {out}", flush=True)
    return res


def test_scores(pred_path):
    from baselines.common import score
    ex = get_export(); r = score.score_file(P.load(pred_path), ex, boot=False)["subsets"]["all"]
    return {k: r[k] for k in ("tiou", "r3", "r5", "r7", "fallback_rate") if k in r}


def train_cmd(method_name, steps, out, seed, config="main", set_items=(), smoke=False, device=None, root=None, keep_optimizer=False):
    """The command line of one run (for concurrent runs in subprocesses)."""
    cmd = [sys.executable, "-m", "baselines.when.train", "--method", method_name, "--out", str(out), "--seed", str(seed), "--config", config]
    if steps: cmd += ["--steps", str(steps)]
    if set_items: cmd += ["--set", *set_items]
    if smoke: cmd += ["--smoke"]
    if device: cmd += ["--device", device]
    if root: cmd += ["--feat-root", str(root)]
    if keep_optimizer: cmd += ["--keep-optimizer"]
    return cmd


def run_parallel(jobs, n):
    """Run [(cmd, out_dir)] with at most n at a time on the one GPU (the heads are launch-bound, so a few concurrent
    runs roughly double the throughput). Each run logs to <out_dir>/train.log. Returns the failed out dirs."""
    import subprocess
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < n:
            cmd, d = pending.pop(0); Path(d).mkdir(parents=True, exist_ok=True)
            f = open(Path(d) / "train.log", "a"); running.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f, d))
            print(f"[jobs] started {d}", flush=True)
        time.sleep(2)
        for p, f, d in list(running):
            if p.poll() is not None:
                f.close(); running.remove((p, f, d))
                print(f"[jobs] {'done' if p.returncode == 0 else f'FAILED ({p.returncode})'} {d}", flush=True)
                if p.returncode != 0: failed.append(d)
    return failed


def run_result(d):
    """Result of a finished run, read back from its info.json (as run_one returns it)."""
    i = json.loads((Path(d) / "info.json").read_text())
    if not i.get("complete"): raise SystemExit(f"{d} has not finished")
    return {"method": i["method"], "steps": i["steps"], "config": i["config"], "seed": i["seed"], "best_epoch": i["best_epoch"],
            "out": str(d), "val": i["best_val"], "pred_val": str(Path(d) / "pred_val.npz"), "pred_test": str(Path(d) / "pred_test.npz")}


def run_seeds(method_name, steps, out, seeds, score_test=True, jobs=1, set_items=(), **kw):
    """Train every seed into <out>/seed{k} (jobs > 1: concurrently, in subprocesses), then aggregate."""
    out = Path(out)
    if jobs > 1:
        cmds = [(train_cmd(method_name, steps, out / f"seed{sd}", sd, kw.get("config", "main"), set_items, kw.get("smoke", False),
                           kw.get("device"), kw.get("root"), kw.get("keep_optimizer", False)), out / f"seed{sd}") for sd in seeds]
        failed = run_parallel(cmds, jobs)
        if failed: raise SystemExit(f"runs failed: {failed} (see their train.log)")
        rows = [run_result(out / f"seed{sd}") for sd in seeds]
    else:
        kw = dict(kw, overrides=parse_set(set_items)) if set_items and "overrides" not in kw else kw
        rows = []
        for sd in seeds:
            r = run_one(method_name, steps, out / f"seed{sd}", seed=sd, **kw)
            if r is None: return None
            rows.append(r)
    return aggregate(method_name, out, seeds, rows, score_test)


def aggregate(method_name, out, seeds, rows, score_test=True):
    """<out>/summary.json and the representative <out>/pred_{val,test}.npz (the best-val seed)."""
    out = Path(out)
    rep = max(rows, key=lambda r: r["val"]["tiou"])
    for sp in ("val", "test"):
        z = P.load(rep[f"pred_{sp}"]); info = dict(z["info"], representative_seed=rep["seed"], seeds=list(seeds))
        P.save(out / f"pred_{sp}.npz", "when", z["sids"], info, start=z["start"], end=z["end"], fallback=z["fallback"])
    summ = dict(method=method_name, steps=rows[0]["steps"], config=rows[0]["config"], seeds=list(seeds), representative_seed=rep["seed"],
                per_seed=[{"seed": r["seed"], "best_epoch": r["best_epoch"], "val": r["val"]} for r in rows])
    v = np.array([r["val"]["tiou"] for r in rows]); summ["val_tiou_mean"], summ["val_tiou_std"] = float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)
    if score_test:
        for r, ps in zip(rows, summ["per_seed"]): ps["test"] = test_scores(r["pred_test"])
        for k in ("tiou", "r7"):
            t = np.array([ps["test"][k] for ps in summ["per_seed"]])
            summ[f"test_{k}_mean"], summ[f"test_{k}_std"] = float(t.mean()), float(t.std(ddof=1) if len(t) > 1 else 0.0)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    shutil.copyfile(Path(rep["out"]) / "info.json", out / "info.json")
    print(f"[{method_name}] {len(seeds)} seeds: val tIoU {summ['val_tiou_mean']:.4f} +- {summ['val_tiou_std']:.4f}"
          + (f", test tIoU {summ['test_tiou_mean']:.4f} +- {summ['test_tiou_std']:.4f}, R@0.7 {summ['test_r7_mean']:.3f}" if score_test else "")
          + f" -> {out}/summary.json", flush=True)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(METHODS)); ap.add_argument("--steps", type=int, choices=(8, 32))
    ap.add_argument("--out", required=True); ap.add_argument("--config", default="main"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device"); ap.add_argument("--feat-root"); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--stop-after", type=int, help="testing aid: stop after N epochs of this invocation, as if preempted")
    ap.add_argument("--keep-optimizer", action="store_true", help="keep optimizer state in last.pt/best.pt after the run")
    ap.add_argument("--jobs", type=int, default=1, help="with --seeds: run up to N seeds concurrently on the GPU (subprocesses)")
    a = ap.parse_args(argv)
    kw = dict(config=a.config, overrides=parse_set(a.set), smoke=a.smoke, device=a.device, root=a.feat_root, fresh=a.fresh,
              stop_after=a.stop_after, keep_optimizer=a.keep_optimizer)
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",")]
        if a.jobs > 1:
            kw = {k: v for k, v in kw.items() if k in ("config", "smoke", "device", "root", "keep_optimizer")}
            r = run_seeds(a.method, a.steps, a.out, seeds, jobs=a.jobs, set_items=a.set, **kw)
        else:
            r = run_seeds(a.method, a.steps, a.out, seeds, **kw)
    else:
        r = run_one(a.method, a.steps, a.out, seed=a.seed, **kw)
    return 3 if r is None else 0


if __name__ == "__main__":
    sys.exit(main())
