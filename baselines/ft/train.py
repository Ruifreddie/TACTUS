#!/usr/bin/env python3
"""Fine-tune V-JEPA 2.1 inside TactusNet: the runs of README.md, all from experiment A's trained heads (LP-FT).

  python -m baselines.ft.train --measure-l0 --exp $EXP_A --out $FT/l0.json          # once, before the runs (shared)
  python -m baselines.ft.train --run joint --exp $EXP_A --l0 $FT/l0.json --out $FT/runs/joint [--micro 8]

Runs (heads.SPECS names):  control  all heads, encoder frozen          joint  all heads, one trainable encoder
                           sem / tmp / spa   one head (spa: its three decoders) with its own trainable encoder
Every run: the same clip stream (data.epoch_plan, seed), 10 epochs of 32 clips per step, VideoMAE's schedule shape and
encoder recipe (encoder.py), one schedule for all weights, EMA 0.999 of all weights, bf16 autocast in training, and
fp32 tokens (as TactusNet's cache: data.fp32_mode) for validation. A step: the encoder encodes the full frames (and the zoom
crops when a zoom decoder is in the run); each head takes its recipe step on its clips with these tokens as a leaf;
d loss_h / d tokens, weighted by 1 / L0_h divided over the run's heads (every head group counts equally, the decoders
of the Spatial Head share their group's share), flows back through the encoder; each head clips its own gradient with
its recipe's grad_clip; the encoder is not clipped (VideoMAE). Validation after every epoch with the EMA weights, by
each head's own criterion; selection.json picks epochs from history.jsonl (see `select`).
Outputs in --out: run.json, history.jsonl, selection.json, epochs/eNN/<head>.pt (trainer format, EMA weights in
`modules`) + encoder.pt (EMA ViT weights; not for control), state_last.pt (resume), log.txt.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import encoder as E
from . import heads as H
from . import snapshots

RUNS = {
    "control": dict(heads=H.ORDER, trainable=False, desc="all heads, frozen encoder"),
    "joint": dict(heads=H.ORDER, trainable=True, desc="all heads, one shared trainable encoder"),
    "sem": dict(heads=("sem",), trainable=True, desc="Semantic Head alone, its own trainable encoder"),
    "tmp": dict(heads=("tmp",), trainable=True, desc="Temporal Head alone, its own trainable encoder"),
    "spa": dict(heads=("s1", "zm", "ct"), trainable=True, desc="Spatial Head alone (stage 1, zoom, contact), its own trainable encoder"),
}
L0_SEED = 12345           # the clip stream of the L0 measurement (shared by every run)


def log(out, *msg):
    s = f"[ft {time.strftime('%m-%d %H:%M:%S')}] " + " ".join(str(m) for m in msg)
    print(s, flush=True)
    if out is not None:
        with open(Path(out) / "log.txt", "a") as f:
            f.write(s + "\n")


def _save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _kc_dir(a):
    kc = a.kc_rng or os.environ.get("KC_RNG", "")
    kc = Path(kc)
    return kc if kc.name == "kpt_combo" else kc / "kpt_combo"


def build_heads(a, names, out=None):
    sk = snapshots.load("kc_rng", _kc_dir(a))
    su = snapshots.load("unitap", Path(a.exp) / "third_party/t2b/kpt_combo")
    hs = []
    for n in H.ORDER:
        if n in names:
            hs.append(H.Head(n, a.exp, sk if H.SPECS[n]["snap"] == "kc_rng" else su, a.device, a.vjepa_ckpt, batch=a.batch, out_dir=out))
    return hs, su


def stream_weights(snap_u, exp):
    """Pair weights of every Gold training clip: the trainers' _build_train_reweighting (power 0.5, cap 10, the recipes'
    values), the same for every run because it depends only on the v1 training split."""
    root = Path(exp) / "data/case32/v1"
    sids = [l.strip() for l in open(root / "train.txt") if l.strip()]
    with snap_u.active() as T:
        rw = T._build_train_reweighting(data_root=root, train_sids=sids, num_combo_classes=6, num_range_bins=int(T.RANGE_NUM_BINS),
                                        power=0.5, max_factor=10.0, include_range_bins=False)
    w = [float(rw["sample_weight_by_sid"][s]) for s in sids]
    return sids, w


def _encode(enc, video, trainable, device):
    if video is None:
        return None
    v = video.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(trainable):
        return enc(v).float()


def run_micro(mb, hs, enc, trainable, weights, epoch, lrs, accum, device):
    """Forward the encoder once per video kind, one recipe step per head, then the weighted backward into the encoder."""
    tf = _encode(enc, mb["video_full"] if any(h.tokens == "full" for h in hs) else None, trainable, device)
    tz = _encode(enc, mb["video_zoom"] if any(h.tokens == "zoom" for h in hs) else None, trainable, device)
    src = {"full": tf, "zoom": tz}
    grads = {k: (torch.zeros_like(t) if (trainable and t is not None) else None) for k, t in src.items()}
    losses = {}
    for h in hs:
        if h.name not in mb["heads"]:
            continue
        sub, rows = mb["heads"][h.name]
        rows = rows.to(device)
        leaf = src[h.tokens].detach().index_select(0, rows).requires_grad_(trainable)
        m = h.train_step(sub, leaf, epoch, lrs[h.name])
        losses[h.name] = float(m["loss_total"])
        if trainable:
            if leaf.grad is None:
                raise RuntimeError(f"{h.name}: no gradient reached the tokens")
            grads[h.tokens].index_add_(0, rows, leaf.grad * (weights[h.name] / accum))
    if trainable:
        ts = [src[k] for k in ("full", "zoom") if src[k] is not None]
        gs = [grads[k] for k in ("full", "zoom") if src[k] is not None]
        torch.autograd.backward(ts, gs)
    return losses


def measure_l0(a):
    """L0_h: mean training loss of each head at the start (A's weights, frozen encoder, train mode), over a fixed stream."""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hs, su = build_heads(a, H.ORDER, out.parent)
    enc = E.Encoder(a.exp, a.vjepa_ckpt, trainable=False).to(a.device)
    sids, w = stream_weights(su, a.exp)
    ds = D.JointTrain(sids, Path(a.exp) / "data/case32/v1", hs, zoom=D.Zoom(a.exp))
    micros, _ = D.epoch_plan(w, a.batch, a.micro, L0_SEED, 0)
    micros = micros[: a.l0_micro]
    acc = {h.name: [] for h in hs}
    for mb in D.train_loader(ds, micros, hs, a.workers):
        lrs = {h.name: 0.0 for h in hs}
        ls = run_micro(mb, hs, enc, False, {h.name: 0.0 for h in hs}, 1, lrs, 1, a.device)
        for h in hs:
            h.zero_grad()
        for k, v in ls.items():
            acc[k].append(v)
    l0 = {k: float(np.mean(v)) for k, v in acc.items()}
    json.dump(dict(l0=l0, n_micro=len(micros), micro=a.micro, seed=L0_SEED, exp=str(a.exp),
                   note="mean training loss per micro-batch with A's weights and the frozen encoder (bf16), train mode"),
              open(out, "w"), indent=1)
    log(None, "L0", json.dumps(l0))


def loss_weights(hs, l0):
    groups = sorted({h.group for h in hs})
    per_group = {g: [h for h in hs if h.group == g] for g in groups}
    return {h.name: 1.0 / (float(l0[h.name]) * len(groups) * len(per_group[h.group])) for h in hs}


def norm_score(v, v0, mode):
    if not (np.isfinite(v) and np.isfinite(v0)) or v0 == 0 or v == 0:
        return float("nan")
    return float(v / v0) if mode == "min" else float(v0 / v)


def select(history, heads_present, first_epoch=1):
    """Epoch choices from the validation history (lower score = better; score of a head = its criterion relative to
    epoch 0, inverted for max criteria). group: mean over the group's heads; combined: mean over the groups."""
    h0 = next(r for r in history if r["epoch"] == 0)["val"]
    groups = {g: [n for n in H.GROUPS[g] if n in heads_present] for g in H.GROUPS}
    groups = {g: v for g, v in groups.items() if v}
    rows = []
    for r in history:
        if r["epoch"] < first_epoch:
            continue
        s = {n: norm_score(r["val"][n]["value"], h0[n]["value"], h0[n]["mode"]) for n in heads_present}
        gs = {g: float(np.mean([s[n] for n in v])) for g, v in groups.items()}
        rows.append(dict(epoch=r["epoch"], head=s, group=gs, combined=float(np.mean(list(gs.values())))))
    if not rows:
        return {}
    pick = lambda key: min(rows, key=lambda x: (x[key] if np.isfinite(x[key]) else np.inf))["epoch"]  # noqa: E731
    sel = {"combined": pick("combined")}
    for g in groups:
        sel[g] = min(rows, key=lambda x: (x["group"][g] if np.isfinite(x["group"][g]) else np.inf))["epoch"]
    return dict(selection=sel, scores=rows, groups=groups)


class Validator:
    """Val tokens of the current (EMA) encoder in fp32 into <out>/valtok/{full,zoom}/val, then every head's evaluate()."""

    def __init__(self, a, hs, out, zoom):
        self.a, self.hs, self.out, self.zoom = a, hs, Path(out), zoom
        self.v1 = Path(a.exp) / "data/case32/v1"
        self.sids = {k: sorted({s for h in hs if h.tokens == k for s in h.val_sids}) for k in ("full", "zoom")}
        self.done_frozen = False

    def __call__(self, enc_eval, epoch, frozen):
        t0 = time.time()
        if not (frozen and self.done_frozen):
            for k, sids in self.sids.items():
                if not sids:
                    continue
                d = self.out / "valtok" / k / "val"
                if d.exists():
                    shutil.rmtree(d)
                D.write_tokens(enc_eval, sids, self.v1, d, zoom=(self.zoom if k == "zoom" else None), batch=self.a.val_batch,
                               workers=self.a.workers, amp=False, device=self.a.device)
            self.done_frozen = frozen
        t1 = time.time()
        res = {h.name: h.validate(self.out / "valtok" / h.tokens, epoch, batch=int(h.args.batch_size), workers=4) for h in self.hs}
        return res, dict(tokens_s=t1 - t0, heads_s=time.time() - t1)


def epoch0_check(hs, val, tol_rel=0.01, tol_acc=0.005):
    """Before any step: on the restored data, A's heads must reproduce A's own recorded validation (the trainers saved
    val_metrics with their best checkpoint). Live fp32 tokens differ from the fp16 cache by rounding only, so the
    criterion agrees within tol_rel. The Semantic Head's loss gains the tool CE here, so its two accuracies are checked."""
    bad = []
    for h in hs:
        m, ref = val[h.name]["metrics"], h.ckpt_val
        if h.name == "sem":
            for k in ("tool_acc", "action_acc"):
                if k in ref and abs(float(m[k]) - float(ref[k])) > tol_acc:
                    bad.append(f"{h.name}.{k}: {m[k]:.5f} vs A's {ref[k]:.5f}")
            continue
        k = val[h.name]["criterion"]
        if k in ref:
            v, r = float(val[h.name]["value"]), float(ref[k])
            if not np.isfinite(v) or abs(v - r) > tol_rel * max(abs(r), 1e-9):
                bad.append(f"{h.name}.{k}: {v:.5f} vs A's {r:.5f}")
        else:
            bad.append(f"{h.name}: A's checkpoint has no recorded {k}")
    return bad


def snapshot(hs, enc_ema, trainable, epoch, out, val):
    """epochs/eNN: every head in its trainer's format with the EMA weights in `modules` (no `ema` key: loaders use the
    weights as they are), and the EMA encoder (ViT state dict) unless the encoder is frozen."""
    d = Path(out) / "epochs" / f"e{epoch:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for h in hs:
        with h.snap.active() as T, T._use_ema_weights(h.model, h.ema):
            ck = h.checkpoint(epoch, val=val.get(h.name), extra=dict(weights="EMA", encoder=("encoder.pt" if trainable else "frozen V-JEPA 2.1")))
        ck.pop("ema", None)
        _save(ck, d / f"{h.name}.pt")
    if trainable:
        _save(dict(weights=enc_ema.model.state(), epoch=epoch, note="EMA of the fine-tuned V-JEPA 2.1 ViT-L (VJEPAWrapper.encoder keys)"),
              d / "encoder.pt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=sorted(RUNS))
    ap.add_argument("--measure-l0", action="store_true")
    ap.add_argument("--exp", required=True, help="restored experiment A (pipeline/make_exp.py layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--l0", help="l0.json of --measure-l0 (required for runs)")
    ap.add_argument("--kc-rng", default=None, help="repro/action_range/code (default $KC_RNG)")
    ap.add_argument("--vjepa-ckpt", default=os.environ.get("VJEPA_CKPT"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32, help="clips per optimizer step (every run)")
    ap.add_argument("--micro", type=int, default=8, help="clips per forward (memory only; gradients are accumulated)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-batch", type=int, default=4, help="clips per fp32 encoder forward when writing val tokens")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blr", type=float, default=2e-3, help="encoder LR per 256 clips (VideoMAE ViT-L K400)")
    ap.add_argument("--warmup-blr", type=float, default=1e-6)
    ap.add_argument("--min-blr", type=float, default=1e-6)
    ap.add_argument("--warmup-frac", type=float, default=5 / 35, help="VideoMAE: 5 warmup epochs of 35")
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--no-act-ckpt", action="store_true")
    ap.add_argument("--l0-micro", type=int, default=32, help="micro-batches of the L0 measurement")
    ap.add_argument("--limit-clips", type=int, default=0, help="smoke: clips per epoch")
    ap.add_argument("--keep-epochs", default="all", help="all | selected (delete other epoch snapshots at the end)")
    ap.add_argument("--no-epoch0-check", action="store_true", help="train even if A's validation is not reproduced")
    ap.add_argument("--pilot-steps", type=int, default=0, help="pilot: stop after N steps of epoch 1, write pilot.json")
    a = ap.parse_args(argv)
    a.exp = str(Path(a.exp).resolve())
    if not a.vjepa_ckpt:
        raise SystemExit("set --vjepa-ckpt or $VJEPA_CKPT")
    D.set_fp32_flags()                     # FT_FP32: PyTorch's defaults, as experiment A ran (or strict)
    if a.measure_l0:
        return measure_l0(a)
    if not a.run or not a.l0:
        raise SystemExit("--run and --l0 are required")
    spec = RUNS[a.run]
    trainable = bool(spec["trainable"])
    out = Path(a.out)
    (out / "epochs").mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)                    # written again when this invocation finishes
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    hs, su = build_heads(a, spec["heads"], out)
    enc = E.Encoder(a.exp, a.vjepa_ckpt, trainable=trainable, drop_path=a.drop_path, act_ckpt=not a.no_act_ckpt).to(a.device)
    ema = E.EncoderEMA(enc, a.ema) if trainable else None
    l0 = json.load(open(a.l0))["l0"]
    weights = loss_weights(hs, l0)
    sids, w = stream_weights(su, a.exp)
    zoom = D.Zoom(a.exp) if any(h.tokens == "zoom" for h in hs) else None
    ds = D.JointTrain(sids, Path(a.exp) / "data/case32/v1", hs, zoom=zoom)
    n_clips = a.limit_clips or len(sids)
    steps_per_epoch = math.ceil(n_clips / a.batch)
    total = steps_per_epoch * a.epochs
    warmup = int(round(a.warmup_frac * total))
    scale = a.batch / 256.0
    base = a.blr * scale
    sched = E.videomae_schedule(base, total, warmup, a.warmup_blr * scale, a.min_blr * scale) / base   # shape, peak 1
    groups = E.param_groups(enc.net, base, a.layer_decay, a.weight_decay) if trainable else []
    enc_opt = torch.optim.AdamW([dict(params=g["params"], lr=base * g["lr_scale"], weight_decay=g["weight_decay"])
                                 for g in groups], betas=(0.9, 0.999), eps=1e-8) if trainable else None
    info = dict(run=a.run, desc=spec["desc"], trainable_encoder=trainable, heads=[h.describe() for h in hs],
                loss_weights=weights, l0=l0, clips_per_epoch=n_clips, steps_per_epoch=steps_per_epoch, epochs=a.epochs,
                batch=a.batch, micro=a.micro, warmup_steps=warmup, total_steps=total, seed=a.seed,
                encoder=dict(peak_lr=base if trainable else 0.0, layer_decay=a.layer_decay, drop_path=enc.drop_path,
                             weight_decay=a.weight_decay, act_ckpt=bool(trainable and not a.no_act_ckpt), ema=a.ema,
                             groups=[dict(layer=g["layer"], lr_scale=g["lr_scale"], weight_decay=g["weight_decay"], n=len(g["params"]))
                                     for g in groups],
                             trainable_params=sum(p.numel() for p in enc.parameters() if p.requires_grad)),
                schedule="VideoMAE cosine_scheduler shape for every weight (warmup from 5e-4 x peak, cosine to 5e-4 x peak)",
                precision=f"bf16 autocast in training; fp32 val tokens ({D.fp32_mode()})", exp=a.exp, vjepa_ckpt=a.vjepa_ckpt,
                argv=sys.argv[1:] if argv is None else list(argv))
    json.dump(info, open(out / "run.json", "w"), indent=1)
    log(out, f"run {a.run}: {spec['desc']}; heads {[h.name for h in hs]}; {n_clips} clips/epoch, {steps_per_epoch} steps x {a.epochs} epochs; "
             f"loss weights {json.dumps({k: round(v, 6) for k, v in weights.items()})}")
    validator = Validator(a, hs, out, zoom)
    hist_path = out / "history.jsonl"
    history, start_epoch = [], 1
    st_path = out / "state_last.pt"
    if st_path.exists():
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        for h in hs:
            h.load_state(st["heads"][h.name])
        if trainable:
            enc.net.load_state_dict(st["enc"])
            ema.load_state_dict(st["enc_ema"])
            enc_opt.load_state_dict(st["enc_opt"])
        history, start_epoch = st["history"], int(st["epoch"]) + 1
        log(out, f"resumed after epoch {st['epoch']}")
    if not history:
        val, tm = validator(ema.model if trainable else enc, 0, frozen=not trainable)
        bad = epoch0_check(hs, val)
        if bad and not a.no_epoch0_check:
            log(out, "EPOCH-0 CHECK FAILED (the restored data or heads differ from experiment A):", "; ".join(bad))
            raise SystemExit(2)
        history.append(dict(epoch=0, val=val, val_time=tm, epoch0_check=bad or "ok"))
        with open(hist_path, "a") as f:
            f.write(json.dumps(history[-1]) + "\n")
        snapshot(hs, ema, trainable, 0, out, val)
        log(out, "epoch 0 val", json.dumps({n: round(v["value"], 5) for n, v in val.items()}),
            "| A's recorded", json.dumps({h.name: h.ckpt_val.get(v["criterion"]) for h, v in zip(hs, val.values())}))
    step_global = (start_epoch - 1) * steps_per_epoch
    for epoch in range(start_epoch, a.epochs + 1):
        t0 = time.time()
        micros, step_of = D.epoch_plan(w, a.batch, a.micro, a.seed, epoch, limit=a.limit_clips)
        loader = D.train_loader(ds, micros, hs, a.workers)
        enc.train(True)
        sums = {h.name: [] for h in hs}
        gnorms = {h.name: [] for h in hs}
        it = iter(loader)
        i = 0
        for k in range(steps_per_epoch):
            f = float(sched[step_global])
            lrs = {h.name: h.peak_lr * f for h in hs}
            for h in hs:
                h.zero_grad()
            if trainable:
                enc_opt.zero_grad(set_to_none=True)
            n_micro = sum(1 for s in step_of if s == k)
            for _ in range(n_micro):
                mb = next(it)
                i += 1
                ls = run_micro(mb, hs, enc, trainable, weights, epoch, lrs, n_micro, a.device)
                for n, v in ls.items():
                    sums[n].append(v)
            for h in hs:
                gnorms[h.name].append(h.clip_and_step(lrs[h.name], n_micro))
            if trainable:
                for g, gg in zip(enc_opt.param_groups, groups):
                    g["lr"] = base * f * gg["lr_scale"]
                enc_opt.step()
                ema.update()
            step_global += 1
            if k == 0:
                first_step_s = time.time() - t0
            if a.pilot_steps and k + 1 >= a.pilot_steps:
                el = time.time() - t0
                rep = dict(run=a.run, steps=k + 1, clips_per_step=a.batch, micro=a.micro, seconds_per_step=el / (k + 1),
                           seconds_per_step_after_first=(el - first_step_s) / max(k, 1),
                           max_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, gpu=torch.cuda.get_device_name(0),
                           epoch_hours=steps_per_epoch * (el - first_step_s) / max(k, 1) / 3600, steps_per_epoch=steps_per_epoch)
                json.dump(rep, open(out / "pilot.json", "w"), indent=1)
                log(out, "PILOT", json.dumps(rep))
                return
            if k % 25 == 0 or k == steps_per_epoch - 1:
                el = time.time() - t0
                log(out, f"e{epoch} step {k + 1}/{steps_per_epoch} lr x{f:.3f} "
                         + " ".join(f"{n}={np.mean(v[-25 * a.batch // a.micro:]):.4f}" for n, v in sums.items() if v)
                         + f" | {el / (k + 1):.1f} s/step, GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        if i != len(micros):
            raise RuntimeError(f"consumed {i} of {len(micros)} micro-batches")
        train_time = time.time() - t0
        for h in hs:                                            # gradients are not needed for validation: free them
            h.zero_grad()
        if trainable:
            enc_opt.zero_grad(set_to_none=True)
        del loader, it
        torch.cuda.empty_cache()
        val, tm = validator(ema.model if trainable else enc, epoch, frozen=not trainable)
        rec = dict(epoch=epoch, train=dict(loss={n: float(np.mean(v)) for n, v in sums.items() if v},
                                           grad_norm={n: float(np.mean(v)) for n, v in gnorms.items()}, seconds=train_time),
                   val=val, val_time=tm)
        history.append(rec)
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        snapshot(hs, ema, trainable, epoch, out, val)
        st = dict(epoch=epoch, history=history, heads={h.name: h.state() for h in hs})
        if trainable:
            st.update(enc=enc.state(), enc_ema=ema.state_dict(), enc_opt=enc_opt.state_dict())
        _save(st, st_path)
        sel = select(history, [h.name for h in hs])
        json.dump(sel, open(out / "selection.json", "w"), indent=1)
        log(out, f"epoch {epoch} done in {train_time / 60:.1f} min (+{sum(tm.values()) / 60:.1f} min val); val",
            json.dumps({n: round(v["value"], 5) for n, v in val.items()}), "| selection", json.dumps(sel.get("selection")))
    sel = select(history, [h.name for h in hs])
    json.dump(sel, open(out / "selection.json", "w"), indent=1)
    if a.keep_epochs == "selected":
        keep = {0} | set(sel["selection"].values())
        for d in (out / "epochs").glob("e*"):
            if int(d.name[1:]) not in keep:
                shutil.rmtree(d)
    (out / "DONE").write_text(json.dumps(sel.get("selection")))
    log(out, "finished", json.dumps(sel.get("selection")))


if __name__ == "__main__":
    sys.exit(main())
