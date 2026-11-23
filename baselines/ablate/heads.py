#!/usr/bin/env python3
"""Per-clip predictions of TactusNet's Semantic Head (cls) and Temporal Head (interval) as standard prediction files.

The classification and interval checkpoints come from the repro trainer (pipeline/run_exp.sh: cls1/cls2, interval;
baselines/ablate/run.sh for the ablation runs). eval_head.py only reports aggregates; this script runs the same forward
(EMA weights, cached tokens, the trainer's autocast) and keeps every clip, so the rows are scored by
baselines.common.score like every baseline (bootstrap CIs, subsets, Pair mAcc).

  python -m baselines.ablate.heads --kind cls  --ckpt <exp>/experiments/runs/cls2/checkpoints/best.pt --root <exp>/data/case32/v1 \
         --split test --out OUT/pred_test.npz [--check <exp>/results/cls_test.json]
  python -m baselines.ablate.heads --kind when --ckpt <exp>/experiments/runs/interval/checkpoints/best.pt --root ... --split test \
         --out OUT/pred_test.npz [--cond gt | null | pred:<cls pred file>] [--check <exp>/results/interval_test.json]

--cond (interval only): gt = the true tool and action (benchmark protocol, default); pred:<file> = the pair predicted by
the Semantic Head in that cls prediction file (the cascade); null = constant condition, for a head trained with
ABLATE_NULL_COND=1 (launch.py). A run dir with ablate.json records the training switch; a mismatch is an error.
Pair logits are the trainer's structured logits (tool logit + action logit of each valid pair): the predicted pair is
their argmax, as in the paper. <out>.heads.npz also keeps the tool and action logits, because the trainer's Which and
What accuracies (paper: 96.2 / 96.5) read the two heads directly, whereas score.py reads both from the predicted pair.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

from baselines.common import preds as P
from baselines.common.data import get_export

from .backbones import unitap_dir
from .launch import patch_model_null_cond


def _import_trainer(code):
    for p in (str(Path(code) / "kpt_combo"), str(code)):
        if p not in sys.path: sys.path.insert(0, p)
    import train as T
    return T


def build(T, ckpt, root, cache, device):
    import argparse as ap, torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); a = ap.Namespace(**dict(ck["args"]))
    a.data_root = str(root); a.vjepa_cache_dir = str(cache); a.device = str(device)
    first = next(iter(sorted(Path(cache).glob("*/*.npy"))), None)
    if first is None: raise SystemExit(f"no token files under {cache}/<split>/")
    width = int(np.load(first, mmap_mode="r").shape[-1])
    # The frozen encoder is never run (tokens are cached): build an unweighted one of the right width instead of loading 5 GB.
    a.vjepa_arch = {1024: "vjepa2_1_vit_large_384", 768: "vjepa2_1_vit_base_384"}[width]
    a.vjepa_checkpoint = ""; a.vjepa_hf_repo = ""; a.vjepa_pretrained_hub = False
    if not Path(str(getattr(a, "vjepa_repo", ""))).is_dir(): a.vjepa_repo = str(unitap_dir() / "third_party/t2b/vjepa2")
    m = T.build_model(a, num_combo_classes=6).to(device); T.load_trainable_state_dict(m, ck["modules"]); m.eval()
    if "ema" in ck:
        e = T.TrainableEMA(model=m, decay=float(getattr(a, "ema_decay", 0.999))); e.load_state_dict(ck["ema"], model=m); e.apply_shadow(m)
    return m, a, ck


def run(kind, code, ckpt, root, split, out, cond="gt", cache=None, check=None, device="cuda", batch=32, workers=4, export=None, hf_raw=None):
    import torch
    T = _import_trainer(code); dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    root = Path(root); cache = Path(cache or root / "vjepa_token_cache")
    rec = Path(ckpt).parent.parent / "ablate.json"
    trained_null = json.load(open(rec)).get("null_condition", False) if rec.exists() else False
    if kind == "when" and trained_null != (cond == "null"):
        raise SystemExit(f"{ckpt}: trained with null_condition={trained_null}, but --cond {cond}")
    m, a, ck = build(T, ckpt, root, cache, dev)
    if cond == "null": patch_model_null_cond(sys.modules["model"])
    pred_pair = None
    if cond.startswith("pred:"):
        cp = P.load(cond[5:]); pred_pair = {s: int(l.argmax()) for s, l in zip(cp["sids"], cp["pair_logits"])}
    ds = T.SurgKptComboDataset(root=str(root), split="val", split_file=str(root / f"{split}.txt"), input_size=a.input_size, normalize=True,
                               load_video=False, combo_only_pure=bool(getattr(a, "combo_only_pure", False)), tool_kpt_only=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    tc = T.VJEPATokenCache(root=cache, enabled=True, dtype_name=str(getattr(a, "vjepa_cache_dtype", "fp16")))
    amp = bool(getattr(a, "amp", False)) and dev.type == "cuda"; amp_dtype = T._get_amp_dtype(getattr(a, "amp_dtype", "bf16"))
    m.set_token_mask_epoch(0)
    S, PL, TL, AL, ST, EN, G = [], [], [], [], [], [], {"combo": [], "s": [], "e": [], "mask": []}
    with torch.no_grad():
        for b in loader:
            sids = [str(x) for x in b["sid"]]; combo = b["combo_id"].long().to(dev)
            if pred_pair is not None:
                miss = [s for s in sids if s not in pred_pair]
                if miss: raise SystemExit(f"{len(miss)} clips have no predicted pair, e.g. {miss[:3]}")
                combo = torch.tensor([pred_pair[s] for s in sids], dtype=torch.long, device=dev)
            tool, act = T._derive_tool_action_id_from_combo_id(combo)
            tok = tc.get_or_compute(model=m, video=None, sids=sids, split=split, target_device=dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp):
                o = m(video=None, tokens_grid=tok, token_order="cached_t_major_then_h_w", tool_id=tool, action_id=act, combo_id=combo)
            S += sids; G["combo"].append(b["combo_id"].numpy())
            G["s"].append(b["gt_action_start_norm"].numpy()); G["e"].append(b["gt_action_end_norm"].numpy()); G["mask"].append(b["action_range_mask"].numpy())
            if kind == "cls":
                tl = o.get("combo_tool_logits", o["combo_logits"]); al = o.get("combo_action_logits", o["combo_logits"])
                PL.append(T._structured_combo_logits_from_tool_action(o["combo_logits"], tl, al).float().cpu().numpy())
                TL.append(tl.float().cpu().numpy()); AL.append(al.float().cpu().numpy())
            else:
                ST.append(o["pred_action_start_norm"].reshape(-1).float().cpu().numpy()); EN.append(o["pred_action_end_norm"].reshape(-1).float().cpu().numpy())
    G = {k: np.concatenate(v) for k, v in G.items()}
    gt_tool, gt_act = (x.numpy() for x in T._derive_tool_action_id_from_combo_id(torch.from_numpy(G["combo"]).long()))
    info = {"method": "tactusnet", "head": kind, "ckpt": str(ckpt), "epoch": ck.get("epoch"), "split": split, "cache": str(cache),
            "condition": cond if kind == "when" else "none (classification)", "trained_null_condition": trained_null}
    trainer = {}
    if kind == "cls":
        pl = np.concatenate(PL); tl = np.concatenate(TL); al = np.concatenate(AL)
        trainer = {"combo_acc": float((pl.argmax(1) == G["combo"]).mean()), "tool_acc": float((tl.argmax(1) == gt_tool).mean()),
                   "action_acc": float((al.argmax(1) == gt_act).mean())}
        P.save(out, "cls", S, info, pair_logits=pl)
        np.savez_compressed(str(out).replace(".npz", ".heads.npz"), sids=np.array(S), tool_logits=tl, action_logits=al,
                            note="tool/action ids in the trainer's order: tool hook, grasper, clipper, scissors; action dissect, grasp, clip, cut")
    else:
        st = np.concatenate(ST); en = np.concatenate(EN)
        ps = np.clip(st, 0, 1); pe = np.maximum(np.clip(en, 0, 1), ps); gs = np.clip(G["s"], 0, 1); ge = np.maximum(np.clip(G["e"], 0, 1), gs)
        inter = np.clip(np.minimum(pe, ge) - np.maximum(ps, gs), 0, None); uni = np.maximum(np.maximum(pe, ge) - np.minimum(ps, gs), 1e-6)
        v = G["mask"] > 0.5; tiou = inter / uni
        trainer = {"range_tiou": float(tiou[v].mean()), "range_r07": float((tiou[v] >= 0.7).mean()), "range_count": int(v.sum())}
        P.save(out, "when", S, info, start=st, end=en)
    res = {"out": str(out), "n": len(S), "trainer_metrics": trainer}
    if check and Path(check).exists():
        ref = json.load(open(check))["metrics"]; res["check"] = {k: {"here": v, "eval_head": ref.get(k)} for k, v in trainer.items() if k in ref}
        if cond == "gt":
            bad = {k: d for k, d in res["check"].items() if d["eval_head"] is not None and abs(d["here"] - d["eval_head"]) > 2e-3}
            res["check_ok"] = not bad
            if bad: print(f"[heads] WARNING: differs from eval_head.py: {bad}", flush=True)
    ex = get_export(export, hf_raw)
    from baselines.common import score as SC
    sc = SC.score_file(P.load(out), ex, split=split, boot=False)["subsets"]["all"]
    res["score_py"] = {k: v for k, v in sc.items() if k not in ("n", "per_pair_acc")}
    json.dump(res, open(str(out).replace(".npz", ".check.json"), "w"), indent=1); print(json.dumps(res)[:1500])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["cls", "when"])
    ap.add_argument("--code", default=None, help="repro interval/classification code dir ($KC_RNG)"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="<exp>/data/case32/v1"); ap.add_argument("--cache", default=None, help="token cache root (default <root>/vjepa_token_cache)")
    ap.add_argument("--split", default="test"); ap.add_argument("--out", required=True); ap.add_argument("--cond", default="gt")
    ap.add_argument("--check", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--export"); ap.add_argument("--hf-raw")
    a = ap.parse_args(argv)
    import os
    code = a.code or os.environ.get("KC_RNG") or str(Path(os.environ.get("BL_ROOT", "")) / "smoke/code/repro/action_range/code")
    if a.kind == "cls" and a.cond != "gt": raise SystemExit("--cond is for the interval head")
    run(a.kind, code, a.ckpt, a.root, a.split, a.out, a.cond, a.cache, a.check, a.device, a.batch, a.workers, a.export, a.hf_raw)


if __name__ == "__main__":
    sys.exit(main())
