"""Training / evaluation loop shared by the three keypoint baselines (HRNet-W32 + DARK, ViTPose-B, DSTA).

A model file defines a Spec (official config, model, optimizer groups, loss, predict) and calls `main(spec)`. Contract
(baselines/README.md): Gold train only; val ADE selects best.pt (lower is better); last.pt resumes; log.jsonl;
pred_val.npz / pred_test.npz from best.pt; info.json with the recipe actually used.

TACTUS adaptations, identical for the three models (everything else is the official configuration):
 (a) 36 outputs = 12 tool (4 tools x 3) + 24 contact (6 pairs x 4); the official loss with target weight = visibility x
     true-group mask (only the true tool's 3 and the true pair's 4 outputs); the test reads the same 7 outputs;
 (b) inputs: the full content frame resized to the official input pixel budget turned landscape (no detector, no box);
     image models train on the about-1-fps protocol frames (ex.frames_1fps) and are tested on all 16; DSTA takes the
     +-2 consecutive raw neighbours of each protocol frame;
 (c) schedule length set in epochs over TACTUS, with the official schedule shape (warmup in iterations, step decays at
     the same fractions of the epochs);
 (d) the official total batch reached on one GPU by gradient accumulation (micro-batch = official per-GPU batch), LR
     linearly scaled if the total batch differs from the official one.
Not changed: architecture, optimizer, LR schedule shape, loss, augmentation (minus flips and the human half-body crop),
decoding. Flip test is off. Mixed precision (bf16 autocast) only changes the arithmetic.
"""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.common import config, metrics as Mx, preds as P
from baselines.common.data import get_export
from baselines.common.train_utils import Checkpointer, JsonlLog, set_seed

from . import core
from .datasets import FullFrameClips, FullFrames, collate_clips

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255     # ImageNet RGB, all three official pipelines
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
FRAMES_1FPS_CACHE = Path(config.OUT) / "kpt_cache" / "frames_1fps_train.json"
# The export's contact-slot order is not consistent across clips (baselines/where/check_order.py), so the targets use one
# canonical permutation per clip: slots sorted by the angle of their mean position, clockwise from the image's left.
CONTACT_ORDER = "baselines.where.canonical_order:canonical"


class Spec:
    """Model-specific part of a keypoint baseline."""
    name = paper_row = ""
    input_wh = (256, 192)                    # landscape input (w, h)
    clips = False                            # True: DSTA (5 raw frames per key frame)
    channels_last = False
    jitter = {}                              # official random scale / rotation
    official = {}                            # official config values: batch, per_gpu, gpus, lr, epochs, source
    schedule = {}                            # warmup_iters, warmup_ratio, milestones (fractions of the epochs), gamma, grad_clip
    defaults = {}                            # epochs, workers, eval_batch, val_every

    def build_model(self, args): raise NotImplementedError
    def param_groups(self, model, lr): raise NotImplementedError
    def optimizer(self, groups, lr): raise NotImplementedError
    def loss(self, model, img, kp, vis, ch): raise NotImplementedError    # img normalized float on the GPU
    def predict(self, model, img, ch): raise NotImplementedError          # -> numpy [B,7,2] in input pixels
    def recipe(self, args): return {}


def build_args(spec, argv=None):
    d, o = spec.defaults, spec.official
    ap = argparse.ArgumentParser(description=f"TACTUS keypoint baseline: {spec.paper_row}")
    ap.add_argument("--out", required=True, help="output dir, e.g. $BL_OUT/kpt_" + spec.name)
    ap.add_argument("--smoke", action="store_true", help="a few steps on tiny batches (local smoke data)")
    ap.add_argument("--contact-order", default=CONTACT_ORDER,
                    help="training-target order of the 4 contact slots: identity (export uid order) or module:function "
                         "(see core.load_contact_order); default: the where family's canonical clip-level order")
    ap.add_argument("--epochs", type=int, default=None, help=f"default {d.get('epochs', o['epochs'])} (official); --smoke: 2")
    ap.add_argument("--micro-batch", type=int, default=o["per_gpu"], help="samples per forward pass (official per-GPU batch)")
    ap.add_argument("--accum", type=int, default=None, help="micro-batches per optimizer step (default: official batch / micro-batch)")
    ap.add_argument("--lr", type=float, default=None, help="override; default = official LR x total batch / official batch")
    ap.add_argument("--workers", type=int, default=d.get("workers", 8)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=d.get("val_every", 1), help="epochs between val passes (last epoch always)")
    ap.add_argument("--max-steps", type=int, default=0, help="cap on micro-batches per epoch (0 = full epoch)")
    ap.add_argument("--eval-batch", type=int, default=d.get("eval_batch", 64))
    ap.add_argument("--export"); ap.add_argument("--hf-raw")
    for a in getattr(spec, "extra_args", []): ap.add_argument(*a[0], **a[1])
    a = ap.parse_args(argv)
    if a.accum is None: a.accum = max(1, round(o["batch"] / a.micro_batch))
    if a.epochs is None: a.epochs = 2 if a.smoke else d.get("epochs", o["epochs"])
    if a.smoke:   # tiny, fast, and kind to the shared local GPU; still exercises accumulation and resume
        a.micro_batch = min(a.micro_batch, 2); a.accum = min(a.accum, 2)
        a.max_steps = a.max_steps or 4; a.workers = min(a.workers, 2); a.eval_batch = min(a.eval_batch, 8); a.val_every = 1
        if hasattr(a, "kf_per_item"): a.kf_per_item = min(a.kf_per_item, a.micro_batch)
    a.total_batch = a.micro_batch * a.accum
    a.lr_used = a.lr if a.lr is not None else o["lr"] * a.total_batch / o["batch"]
    return a


def lr_mult(it, epoch, epochs, s):
    """mmcv/mmengine linear warmup over `warmup_iters` optimizer steps (from warmup_ratio), times the official
    epoch-based MultiStep decay at round(fraction x epochs)."""
    m = 1.0
    for f in s.get("milestones", ()):
        if epoch >= int(round(f * epochs)): m *= s.get("gamma", 0.1)
    w = s.get("warmup_iters", 0)
    if w and it < w:
        r = s.get("warmup_ratio", 0.001); m *= r + (1 - r) * it / w
    return m


def to_input(img, channels_last):
    """uint8 [B,3,H,W] (or [B,F,3,H,W]) on the GPU -> ImageNet-normalized float (channels_last only for 4-D input)."""
    shp = img.shape; x = (img.reshape(-1, *shp[-3:]).float() - MEAN.to(img.device)) / STD.to(img.device)
    if len(shp) == 5: return x.reshape(shp)
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


def make_loader(ds, batch, workers, train, clips, seed=0, kf_per_item=1):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=max(1, batch // kf_per_item) if clips else batch, shuffle=False,   # datasets order themselves
                      num_workers=workers, drop_last=train, pin_memory=True, collate_fn=collate_clips if clips else None,
                      generator=g, prefetch_factor=4 if workers else None)


def make_dataset(spec, ex, sids, args, train, contact_order, train_frames=None):
    kw = dict(input_wh=spec.input_wh, train=train, jitter=spec.jitter if train else None, contact_order=contact_order, seed=args.seed)
    if spec.clips: return FullFrameClips(ex, sids, kf_per_clip=args.kf_per_clip, kf_per_item=args.kf_per_item, **kw)
    return FullFrames(ex, sids, train_frames=train_frames, **kw)


@torch.no_grad()
def predict_split(spec, model, ds, args, dev):
    """Normalized predictions of all 16 frames of every clip of ds: tool_xy [N,16,3,2], ct_xy [N,16,4,2]."""
    model.eval(); out = np.full((len(ds.sids), 16, core.N_SEL, 2), np.nan, np.float32)
    kfi = 16 if spec.clips else 1
    for b in make_loader(ds, max(args.eval_batch, kfi), args.workers, False, spec.clips, kf_per_item=kfi):
        img = to_input(b["img"].to(dev, non_blocking=True), spec.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            kp = spec.predict(model, img, b["ch"].to(dev))
        M = b["M"].numpy().astype(np.float64); rows = np.asarray(b["row"]).reshape(-1); ts = np.asarray(b["t"]).reshape(-1)
        for j in range(len(kp)):
            W, H = ds.size[ds.sids[rows[j]]]
            out[rows[j], ts[j]] = np.clip(core.px_to_norm(core.affine_points(core.invert_affine(M[j]), kp[j]), W, H), 0, 1)
    if np.isnan(out).any(): raise RuntimeError("some frames were not predicted")
    return out[:, :, :core.N_TOOL_PTS], out[:, :, core.N_TOOL_PTS:]


def val_metrics(ex, sids, tool_xy, ct_xy):
    how, med = Mx.score_how(ex, sids, tool_xy); where = Mx.score_where(ex, sids, ct_xy)
    return {"ade": how["ade"].value(), "gt32": how["gt32"].value(), "median_px": med, "dice": where["dice"].value()}


def main(spec, argv=None):
    args = build_args(spec, argv); t_start = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); set_seed(args.seed)
    torch.backends.cudnn.benchmark = True; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex = get_export(args.export, args.hf_raw); contact_order = core.load_contact_order(args.contact_order)

    tr = ex.sids("train")
    train_frames = None if spec.clips else core.frames_1fps(ex, tr, FRAMES_1FPS_CACHE)
    train_ds = make_dataset(spec, ex, tr, args, True, contact_order, train_frames)
    eval_ds = {sp: make_dataset(spec, ex, ex.sids(sp), args, False, contact_order) for sp in ("val", "test")}

    model = spec.build_model(args).to(dev)
    if spec.channels_last: model = model.to(memory_format=torch.channels_last)
    groups = spec.param_groups(model, args.lr_used)
    for g in groups: g["base_lr"] = g["lr"]
    opt = spec.optimizer(groups, args.lr_used)
    ck = Checkpointer(out, "ade", mode="min"); start = ck.resume(model, opt)
    log = JsonlLog(out / "log.jsonl"); kfi = getattr(args, "kf_per_item", 1)
    n_batches = len(make_loader(train_ds, args.micro_batch, 0, True, spec.clips, kf_per_item=kfi))
    spe = min(n_batches, args.max_steps) if args.max_steps else n_batches       # micro-batches per epoch
    opt_spe = spe // args.accum
    print(f"[{spec.name}] train {len(train_ds)} items, {spe} micro-batches of {args.micro_batch} x accum {args.accum} "
          f"(total batch {args.total_batch}, lr {args.lr_used:.3g}); val {len(eval_ds['val'].sids)} / test {len(eval_ds['test'].sids)} clips; "
          f"epochs {args.epochs}, start {start}, device {dev}", flush=True)

    for epoch in range(start, args.epochs):
        model.train(); train_ds.set_epoch(epoch); t0 = time.time(); losses = []
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(make_loader(train_ds, args.micro_batch, args.workers, True, spec.clips, seed=args.seed * 1000 + epoch, kf_per_item=kfi)):
            if i >= opt_spe * args.accum: break
            it = epoch * opt_spe + i // args.accum; m = lr_mult(it, epoch, args.epochs, spec.schedule)
            for g in opt.param_groups: g["lr"] = g["base_lr"] * m
            img = to_input(b["img"].to(dev, non_blocking=True), spec.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = spec.loss(model, img, b["kp"].to(dev), b["vis"].to(dev), b["ch"].to(dev))
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch {epoch} step {i}")
            (loss / args.accum).backward(); losses.append(loss.detach().item())
            if (i + 1) % args.accum == 0:
                if spec.schedule.get("grad_clip"): torch.nn.utils.clip_grad_norm_(model.parameters(), spec.schedule["grad_clip"])
                opt.step(); opt.zero_grad(set_to_none=True)
            if i % (50 * args.accum) == 0:
                print(f"  epoch {epoch} step {i // args.accum}/{opt_spe} loss {losses[-1]:.6f} lr {opt.param_groups[-1]['lr']:.3g}", flush=True)
        t_train = time.time() - t0; row = dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else None,
                                              micro_batches=len(losses), lr=opt.param_groups[-1]["lr"], train_time=t_train)
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            val = val_metrics(ex, eval_ds["val"].sids, *predict_split(spec, model, eval_ds["val"], args, dev))
            better = ck.step(epoch, val, {"model": model.state_dict(), "optimizer": opt.state_dict()})
            row.update({f"val_{k}": v for k, v in val.items()}, best=bool(better))
            print(f"[{spec.name}] epoch {epoch}: loss {row['train_loss'] or float('nan'):.6f}  val ADE {val['ade']:.2f}  "
                  f">32px {100 * val['gt32']:.1f}%  Dice {val['dice']:.3f} {'*' if better else ''} ({time.time() - t0:.0f}s)", flush=True)
        log.write(**row, epoch_time=time.time() - t0)

    best = ck.load_best(model); preds = {}
    for sp in ("val", "test"):
        preds[sp] = predict_split(spec, model, eval_ds[sp], args, dev)
        P.save(out / f"pred_{sp}.npz", "kpt", eval_ds[sp].sids, {"method": spec.name, "row": spec.paper_row, "split": sp,
               "checkpoint": str(out / "best.pt"), "best_epoch": int(best["epoch"]), "input": f"full content frame {spec.input_wh[0]}x{spec.input_wh[1]}",
               "conditioning": "true tool (3 points) and true pair (4 points) of a 36-output union"}, tool_xy=preds[sp][0], ct_xy=preds[sp][1])
    vm = val_metrics(ex, eval_ds["val"].sids, *preds["val"])
    rows = [json.loads(l) for l in open(out / "log.jsonl")] if (out / "log.jsonl").exists() else []
    n_train_frames = len(train_ds) * (args.kf_per_item if spec.clips else 1)
    info = {"method": spec.name, "row": spec.paper_row, **spec.recipe(args), "official": spec.official,
            "input": {"what": "full content frame (ex.frames / ex.raw_frames), plain rescale to the input, no box", "input_wh": list(spec.input_wh),
                      "official_input_wh": spec.official.get("input_wh"), "frames": spec.frames_desc},
            "train_items_per_epoch": len(train_ds), "train_frames_per_epoch": n_train_frames,
            "conditioning": "36 outputs = 12 tool (4 tools x 3) + 24 contact (6 pairs x 4); loss on the true tool's 3 and the true pair's 4 "
                            "outputs (target weight = visibility x group mask); the test reads the same 7", "contact_order": args.contact_order,
            "fine_tuned": "end to end (pretrained backbone, every head and decoder from scratch)",
            "epochs": args.epochs, "micro_batch": args.micro_batch, "accum": args.accum, "total_batch": args.total_batch, "lr": args.lr_used,
            "lr_rule": "official LR x total batch / official batch" if args.lr is None else "--lr override",
            "optimizer_steps_per_epoch": opt_spe, "optimizer_steps": opt_spe * args.epochs,
            "schedule": {**spec.schedule, "milestone_epochs": [int(round(f * args.epochs)) for f in spec.schedule.get("milestones", ())]},
            "val_every": args.val_every, "selection": "val ADE (lower is better), checked every val_every epochs", "precision": "bf16 autocast",
            "seed": args.seed, "smoke": bool(args.smoke), "max_steps": args.max_steps, "best_epoch": int(best["epoch"]),
            "val_best": {k: float(v) for k, v in vm.items()}, "wall_clock_train_s": float(sum(r.get("epoch_time", 0) for r in rows)),
            "wall_clock_this_run_s": time.time() - t_start, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "python": platform.python_version(), "argv": sys.argv[1:] if argv is None else argv}
    core.write_json(out / "info.json", info)
    print(f"[{spec.name}] best epoch {best['epoch']}  val ADE {vm['ade']:.2f}  Dice {vm['dice']:.3f}  -> {out}/pred_val.npz, pred_test.npz", flush=True)
    return info
