#!/usr/bin/env python3
"""Shared harness of the classification baselines (Which, What, Pair). Each model runs its OFFICIAL recipe
(models.RECIPES: optimizer, LR schedule, augmentation, loss), with only the TACTUS adaptations listed there.

  python -m baselines.cls.train --model {tsn,tsm,videomae,xclip,endofm,mvitv2} --out $BL_OUT/cls/<model> [--smoke]
      [--batch B] [--eff-batch 64] [--workers 8] [--grad-ckpt] [--no-cache] [--seed 0] [--stop-after N] [--bench N]
      [--train-splits train]

Protocol (baselines/README.md): Gold train only, model selection on Gold val, one prediction on Gold test; input = the
16 protocol frames of the clip; 6 pair outputs; no ground truth as input. Validation after every epoch; best.pt by val
Pair accuracy (common Checkpointer); last.pt resumes a killed run (rerun the same command); at the end pred_val.npz /
pred_test.npz from best.pt (common.preds, kind "cls") and info.json with the recipe actually used and every adaptation.
bf16 autocast; gradient accumulation to an effective batch of 64 clips; the official LR scaled linearly to it.

--batch          clips (videos) per micro-step; default: the recipe's choice for one 24 GB GPU
--stop-after N   stop this invocation after N epochs (simulates preemption; rerun to continue)
--bench N        time N training micro-steps and the val pass, write <out>/bench.json, train nothing
--train-splits   kept for completeness; every reported row uses the default (Gold train)
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, json, math, platform, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import preds as P
from baselines.common.data import get_export
from baselines.common.train_utils import Checkpointer, JsonlLog, set_seed
from . import augment as A
from . import data as D
from . import models as M
from . import weights as W

KEY_SETTINGS = ("model", "train_splits", "epochs", "eff_batch", "steps_per_epoch", "peak_lr", "seed", "smoke")


class Ckpt(Checkpointer):
    """common.train_utils.Checkpointer; best.pt keeps the model only (it is never resumed from), last.pt everything."""
    def _save(self, path, state):
        if Path(path).name == "best.pt": state = {k: v for k, v in state.items() if k != "optimizer"}
        super()._save(path, state)


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(M.RECIPES)); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 3 steps on tiny batches (local code-path check)")
    ap.add_argument("--train-splits", default="train"); ap.add_argument("--epochs", type=int, help="override (smoke only)")
    ap.add_argument("--batch", type=int); ap.add_argument("--eff-batch", type=int, help="default 64 (smoke: 2 x batch)")
    ap.add_argument("--workers", type=int); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (ViTs) if the micro-batch does not fit")
    ap.add_argument("--no-cache", action="store_true", help="ignore the pack.py cache and resize on the fly")
    ap.add_argument("--stop-after", type=int); ap.add_argument("--bench", type=int)
    return ap.parse_args(argv)


def make_optimizer(r, groups, peak):
    for g in groups: g["lr"] = peak * g["lr_scale"]
    if r["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=peak, momentum=r["momentum"], nesterov=r["nesterov"])
    try: return torch.optim.AdamW(groups, lr=peak, betas=r["betas"], eps=r["eps"], fused=True)
    except (RuntimeError, TypeError): return torch.optim.AdamW(groups, lr=peak, betas=r["betas"], eps=r["eps"])


def group_lr(r, peak, scale, t, spe, epochs, peak_base):
    """LR of a param group (lr_scale `scale`) at optimizer step t, with the official schedule of the recipe.
    peak: the recipe LR scaled to our batch; spe: optimizer steps per epoch; ratios scale with the batch like the LR."""
    s = r["schedule"]; k = s["kind"]; ep = t // spe; e = t / spe
    if k == "step":            # TSM main.py adjust_learning_rate (lr_type step, applied at each epoch start)
        return peak * scale * s["gamma"] ** sum(ep >= m for m in s["steps"])
    if k == "cosine_epoch":    # Endo-FM: CosineAnnealingLR(T_max=epochs, eta_min=0), scheduler.step() after each epoch
        return peak * scale * 0.5 * (1 + math.cos(math.pi * ep / epochs))
    if k == "videomae":        # VideoMAE utils.cosine_scheduler, per optimizer step; lr_scale multiplies the schedule
        ratio = peak / r["lr"]; start, final = s["warmup_lr"] * ratio, s["min_lr"] * ratio
        wi = int(s["warmup_epochs"] * spe); n = epochs * spe - wi
        v = (start + (peak - start) * t / max(1, wi - 1)) if t < wi else final + 0.5 * (peak - final) * (1 + math.cos(math.pi * (t - wi) / n))
        return v * scale
    if k == "timm_cosine":     # X-CLIP: timm CosineLRScheduler(lr_min=LR/100, warmup_lr_init=0, warmup_t=5 epochs), per step
        g = peak * scale; lr_min = peak_base * s["min_ratio"]; W_ = s["warmup_epochs"]
        return g * e / W_ if e < W_ else lr_min + 0.5 * (g - lr_min) * (1 + math.cos(math.pi * e / epochs))
    if k == "slowfast_cosine":  # PySlowFast lr_policy.get_lr_at_epoch, cosine with COSINE_AFTER_WARMUP, per iteration
        ratio = peak / r["lr"]; start, end = s["warmup_start"] * ratio, s["end"] * ratio; W_ = s["warmup_epochs"]
        cos = lambda x: end + (peak - end) * (math.cos(math.pi * (x - W_) / (epochs - W_)) + 1.0) * 0.5
        return (start + e * (cos(W_) - start) / W_ if e < W_ else cos(e)) * scale
    raise KeyError(k)


def loader(ds, batch, workers, sampler=None, drop_last=False):
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    if workers > 0: kw["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=False, drop_last=drop_last, **kw)


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); out = []
    for x, _, _ in dl:
        x = x.to(dev, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def cls_metrics(ex, sids, logits):
    """Which / What / Pair accuracy and Pair mAcc of argmax pairs, as score.py computes them (all clips pooled)."""
    pp = logits.argmax(1); g = np.array([ex.pair_id(s) for s in sids])
    per = [float((pp[g == p] == p).mean()) for p in range(len(ex.pairs)) if (g == p).any()]
    return dict(pair=float((pp == g).mean()), which=float((ex.pair_tool[pp] == ex.pair_tool[g]).mean()),
                what=float((ex.pair_act[pp] == ex.pair_act[g]).mean()), pair_macc=float(np.mean(per)), n=len(sids))


def wall_from_log(path):
    if not Path(path).is_file(): return 0.0
    rows = {}
    for l in Path(path).read_text().splitlines():
        try: d = json.loads(l)
        except ValueError: continue
        if "epoch" in d and "epoch_time_s" in d: rows[d["epoch"]] = d["epoch_time_s"] + d.get("val_time_s", 0.0)
    return float(sum(rows.values()))


def main(argv=None):
    a = parse(argv); r = M.RECIPES[a.model]
    if not torch.cuda.is_available(): raise SystemExit("needs a CUDA GPU")
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    set_seed(a.seed); ex = get_export(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    free, tot = torch.cuda.mem_get_info(); print(f"[cls] {a.model}: GPU {torch.cuda.get_device_name(0)}, free {free / 2**30:.1f}/{tot / 2**30:.1f} GB")

    # ---------------- data
    splits = [s for s in a.train_splits.split(",") if s]
    if "test" in splits or "val" in splits: raise SystemExit("train splits may only be train (and silver)")
    tr_sids = list(dict.fromkeys(s for sp in splits for s in ex.sids(sp))); va_sids, te_sids = ex.sids("val"), ex.sids("test")
    batch = a.batch or (2 if a.smoke else r["batch"]); eff = a.eff_batch or (2 * batch if a.smoke else 64)
    accum = max(1, eff // batch); eff = batch * accum; k = r["num_sample"]
    epochs = a.epochs or (2 if a.smoke else r["epochs"]); steps = len(tr_sids) // eff
    if a.smoke: steps = min(steps, 3)
    if a.bench: steps = max(steps, 1)
    if steps < 1: raise SystemExit(f"{len(tr_sids)} training clips < effective batch {eff}")
    workers = a.workers if a.workers is not None else (2 if a.smoke else min(16, os.cpu_count() or 1))
    if workers < 1: raise SystemExit("--workers must be >= 1 (per-sample seeding of the augmentation)")
    trset = D.ClipSet(ex, tr_sids, r["pre"], train_tf=A.TRAIN[a.model], norm=r["norm"], num_sample=k, use_cache=not a.no_cache)
    vaset = D.ClipSet(ex, va_sids, D.EVAL_SIZE, norm=r["norm"], use_cache=not a.no_cache)
    teset = D.ClipSet(ex, te_sids, D.EVAL_SIZE, norm=r["norm"], use_cache=not a.no_cache)
    n_items = steps * eff if not a.bench else max(steps * eff, (a.bench + 2 * accum) * batch)
    sampler = D.EpochSampler(len(trset), n_items, a.seed)
    tr_dl = loader(trset, batch, workers, sampler, drop_last=True)
    va_dl = loader(vaset, 2 * batch, workers); te_dl = loader(teset, 2 * batch, workers)
    print(f"[cls] train {len(tr_sids)} clips ({'+'.join(splits)}) at {r['pre']}px (cached {trset.n_cached}), val "
          f"{len(va_sids)} / test {len(te_sids)} at 224px (cached {vaset.n_cached} / {teset.n_cached}); "
          f"{k} view(s) per clip; input 16x224x224")

    # ---------------- model, optimizer, loss
    model = M.build(a.model, len(ex.pairs), ex.pairs, grad_ckpt=a.grad_ckpt).to(dev)
    peak = r["lr"] * eff / r["lr_batch"]
    groups = model.param_groups(r); opt = make_optimizer(r, groups, peak)
    n_par = sum(p.numel() for p in model.parameters()); n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    if n_opt != n_tr: raise RuntimeError(f"optimizer groups cover {n_opt} of {n_tr} trainable parameters")
    n_cls = len(ex.pairs)
    mix = {"timm": A.TimmMixup, "xclip": A.XClipBlend}[r["mix"]](n_cls) if r["mix"] else None
    ce = nn.CrossEntropyLoss(label_smoothing=r["label_smoothing"])
    settings = dict(model=a.model, train_splits="+".join(splits), epochs=epochs, eff_batch=eff, steps_per_epoch=steps,
                    peak_lr=peak, seed=a.seed, smoke=a.smoke)

    def train_step(x, y, rng):
        x = x.to(dev, non_blocking=True).float(); y = y.to(dev, non_blocking=True)
        if k > 1: x = x.flatten(0, 1); y = y.repeat_interleave(k)     # repeated augmentation: views of a clip adjacent
        if mix is not None: x, tgt = mix(x, y, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(x)
        logits = logits.float()
        loss = ce(logits, y) if mix is None else torch.sum(-tgt * F.log_softmax(logits, dim=-1), dim=-1).mean()
        (loss / accum).backward()
        return loss.detach() * len(y), (logits.detach().argmax(1) == y).sum(), len(y)

    def opt_step(t):
        for g in opt.param_groups: g["lr"] = group_lr(r, peak, g["lr_scale"], t, steps, epochs, peak)
        if r.get("clip_grad"): torch.nn.utils.clip_grad_norm_(model.parameters(), r["clip_grad"])
        opt.step(); opt.zero_grad(set_to_none=True)
        return opt.param_groups[0]["lr"]

    if a.bench:   # ---------------- throughput only
        rng = np.random.default_rng(a.seed); model.train(); it = iter(tr_dl); torch.cuda.reset_peak_memory_stats()
        warm = 2 * accum; timed = int(math.ceil(a.bench / accum)) * accum; done = 0; t0 = time.time()
        for j in range(warm + timed):
            try: x, y, _ = next(it)
            except StopIteration: it = iter(tr_dl); x, y, _ = next(it)
            if j == warm: torch.cuda.synchronize(); t0 = time.time(); done = 0
            done += train_step(x, y, rng)[2] // k
            if (j + 1) % accum == 0: opt_step(0)
        torch.cuda.synchronize(); tr_cps = done / (time.time() - t0); peak_mem = torch.cuda.max_memory_allocated() / 2**30
        predict(model, va_dl, dev); torch.cuda.synchronize(); t1 = time.time(); predict(model, va_dl, dev)
        torch.cuda.synchronize(); ev_cps = len(va_sids) / (time.time() - t1)
        res = dict(model=a.model, gpu=torch.cuda.get_device_name(0), micro_batch=batch, views=k, accum=accum,
                   grad_ckpt=a.grad_ckpt, workers=workers, train_clips_per_s=tr_cps, eval_clips_per_s=ev_cps,
                   peak_mem_gb=peak_mem, timed_micro_steps=timed,
                   note="train clips/s counts clips of the dataset (each with num_sample views), incl. data loading")
        (out / "bench.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res)); return 0

    # ---------------- resume
    run_json = out / "run.json"
    if run_json.is_file():
        old = json.loads(run_json.read_text())
        diff = {kk: (old.get(kk), settings[kk]) for kk in KEY_SETTINGS if old.get(kk) != settings[kk]}
        if diff: raise SystemExit(f"{out} holds a run with other settings {diff}; use another --out")
    else:
        run_json.write_text(json.dumps(settings, indent=1))
    ck = Ckpt(out, "pair", "max"); start = ck.resume(model, opt)
    log = JsonlLog(out / "log.jsonl")
    print(f"[cls] {epochs} epochs x {steps} steps, micro-batch {batch} x accum {accum} = {eff} clips"
          f"{f' x {k} views' if k > 1 else ''}, peak LR {peak:.4g} ({r['schedule']['kind']}); "
          + (f"resumed at epoch {start} (best {ck.best:.4f} @ {ck.best_epoch})" if start else "fresh run"))

    # ---------------- train
    for epoch in range(start, epochs):
        if a.stop_after is not None and epoch - start >= a.stop_after:
            print(f"[cls] --stop-after {a.stop_after}: stopping before epoch {epoch}; rerun to continue"); return 0
        sampler.set_epoch(epoch); rng = np.random.default_rng(a.seed * 1000003 + epoch); model.train()
        opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); lsum = torch.zeros((), device=dev); csum = torch.zeros((), device=dev, dtype=torch.long); n = 0; lr = 0.0
        for it, (x, y, _) in enumerate(tr_dl):
            l, c, m = train_step(x, y, rng); lsum += l; csum += c; n += m
            if (it + 1) % accum == 0: lr = opt_step(epoch * steps + it // accum)
        torch.cuda.synchronize(); t_ep = time.time() - t0
        t1 = time.time(); val = cls_metrics(ex, va_sids, predict(model, va_dl, dev)); t_val = time.time() - t1
        better = ck.step(epoch, val, {"model": model.state_dict(), "optimizer": opt.state_dict(), "settings": settings})
        row = dict(epoch=epoch, lr=lr, train_loss=float(lsum) / n, train_acc_vs_true=int(csum) / n, train_views=n,
                   **{f"val_{kk}": v for kk, v in val.items() if kk != "n"}, best=better, epoch_time_s=t_ep, val_time_s=t_val,
                   train_clips_per_s=n / k / t_ep, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
        log.write(**row)
        print(f"[cls] epoch {epoch}: loss {row['train_loss']:.4f} | val pair {val['pair']:.4f} which {val['which']:.4f} "
              f"what {val['what']:.4f}{' *' if better else ''} | lr {lr:.3g} | {t_ep:.0f}s, {row['train_clips_per_s']:.1f} "
              f"clips/s, {row['peak_mem_gb']:.1f} GB")

    # ---------------- predict from best.pt
    best = ck.load_best(model); model.to(dev)
    common = dict(method=a.model, input="16x224x224", init=r["init"], conditioning="none", checkpoint=str(out / "best.pt"),
                  best_epoch=int(best["epoch"]), train_splits="+".join(splits))
    res = {}
    for split, sids, dl in (("val", va_sids, va_dl), ("test", te_sids, te_dl)):
        lg = predict(model, dl, dev); res[split] = cls_metrics(ex, sids, lg)
        P.save(out / f"pred_{split}.npz", "cls", sids, dict(common, split=split), pair_logits=lg)
    man = json.loads((W.ROOT / "MANIFEST.json").read_text()) if (W.ROOT / "MANIFEST.json").is_file() else {}
    wman = man.get(a.model) if a.model != "endofm" else {"checkpoint": man.get("endofm"),
                                                         "teacher_backbone": man.get("endofm_teacher_backbone")}
    info = dict(
        method=r["name"], model=a.model, paper=r["paper"], backbone=r["backbone"], init=r["init"],
        pretraining_data=r["pretrain_data"], weights=W.SOURCES[a.model]["desc"], weights_manifest=wman,
        official_recipe=r["official"], adaptations=r["adaptations"],
        finetuning="full fine-tune" + (" of the video side (text side frozen, official FIX_TEXT)" if r.get("fix_text") else "")
                   + (" (partial BN: every BatchNorm but the first frozen, official)" if r.get("partial_bn") else ""),
        conditioning="none: pixels only, no tool or action ground truth",
        frames=16, frames_official=r["frames_official"],
        sampling="the 16 protocol frames (uniform over the affordance window, end frames included), ex.frames",
        input="16x224x224", pre_crop_size=r["pre"], resize=D.RESIZE, normalization=dict(zip(("mean", "std"), r["norm"])),
        augmentation=r["augmentation"], views_per_clip=k, test="single view: the whole frame squashed to 224 x 224",
        head="6-way pair output (labels.json combo2id_6class); Which/What from the pair",
        loss=("soft-target cross-entropy after " + {"timm": "timm-style Mixup/CutMix", "xclip": "X-CLIP CutmixMixupBlending"}[r["mix"]]
              + f", label smoothing {r['label_smoothing']}") if r["mix"] else f"cross-entropy (label smoothing {r['label_smoothing']})",
        train_splits="+".join(splits), n_train=len(tr_sids), n_train_by_source=dict(Counter(ex.source(s) for s in tr_sids)),
        n_val=len(va_sids), n_test=len(te_sids), cached={"train": trset.n_cached, "val": vaset.n_cached, "test": teset.n_cached},
        epochs=epochs, steps_per_epoch=steps, micro_batch=batch, accum=accum, eff_batch=eff, optimizer=r["optimizer"],
        optimizer_args={kk: r[kk] for kk in ("momentum", "nesterov", "betas", "eps") if kk in r}, weight_decay=r["weight_decay"],
        lr_official=r["lr"], lr_official_batch=r["lr_batch"], peak_lr=peak, schedule=r["schedule"],
        lr_groups=[dict(name=g.get("name"), lr_scale=g["lr_scale"], weight_decay=g["weight_decay"],
                        n_params=sum(p.numel() for p in g["params"])) for g in opt.param_groups],
        clip_grad=r.get("clip_grad"), drop_path=r.get("drop_path"), dropout=r.get("dropout"), layer_decay=r.get("layer_decay"),
        x_clip_prompts=getattr(model, "prompts", None), attention=getattr(model, "attn", None),
        endofm_time_embeddings=getattr(model, "time_embed_len", None),
        precision="bf16 autocast, fp32 master weights", grad_ckpt=a.grad_ckpt, seed=a.seed, smoke=a.smoke,
        params=n_par, trainable_params=n_tr, best_epoch=int(best["epoch"]), best_val_pair=float(best["best"]),
        val=res["val"], test_quick=res["test"], wall_clock_train_h=wall_from_log(out / "log.jsonl") / 3600,
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version(),
        time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import transformers, timm; info["transformers"] = transformers.__version__; info["timm"] = timm.__version__
    except ImportError: pass
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(f"[cls] best epoch {info['best_epoch']} (val pair {info['best_val_pair']:.4f}); wrote {out}/pred_val.npz, "
          f"pred_test.npz, info.json; test pair {res['test']['pair']:.4f} (score with baselines.common.score)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
