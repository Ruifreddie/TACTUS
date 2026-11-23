#!/usr/bin/env python3
"""Rows of the encoder fine-tuning experiments (README.md) and the answer to "does joint learning pull a head down?".

  python -m baselines.ft.collect --ft $FT [--out $FT/results]

Rows (each part = the evaluation of one run at one selected epoch; selection.json of that run):
  tactusnet   experiment A itself (epoch-0 snapshot of the control run) through the same evaluation path
  control     frozen encoder, heads trained 10 more epochs; each head group at its own best epoch (a frozen encoder is
              the same at every epoch, so the heads still form one model)
  joint       one fine-tuned encoder for all heads, ONE epoch for the whole model (combined criterion)
  separate    one fine-tuned encoder per head group: Semantic from run sem, Temporal from tmp, Spatial from spa
  joint_each  supplement: the joint run with each head group at its own best epoch (tells the cost of sharing the
              gradient apart from the cost of choosing one epoch for all heads)
Paired bootstraps over test videos (score.py): joint - separate, joint - control, separate - control, joint_each - separate.
Verdict per metric: worse / better when the 95% CI of the difference excludes 0, else no clear difference.
"""
import argparse
import json
from pathlib import Path

from baselines.common import metrics as M
from baselines.common import preds as P
from baselines.common import score as SC
from baselines.common.data import get_export

ROWS = {
    "tactusnet": dict(cls=("control", "epoch:0"), when=("control", "epoch:0"), kpt=("control", "epoch:0")),
    "control": dict(cls=("control", "semantic"), when=("control", "temporal"), kpt=("control", "spatial")),
    "joint": dict(cls=("joint", "combined"), when=("joint", "combined"), kpt=("joint", "combined")),
    "separate": dict(cls=("sem", "semantic"), when=("tmp", "temporal"), kpt=("spa", "spatial")),
    "joint_each": dict(cls=("joint", "semantic"), when=("joint", "temporal"), kpt=("joint", "spatial")),
}
COMPARE = (("joint", "separate"), ("joint", "control"), ("separate", "control"), ("joint_each", "separate"))
# metric -> (part, key in accs_for / summary, +1 if higher is better)
METRICS = {
    "which": ("cls", "which", 1), "what": ("cls", "what", 1), "pair": ("cls", "pair", 1),
    "tiou": ("when", "tiou", 1), "r7": ("when", "r7", 1),
    "ade": ("kpt", "how_ade", -1), "gt32": ("kpt", "how_gt32", -1), "dice": ("kpt", "where_dice", 1),
}


def epoch_of(ft, run, sel):
    if sel.startswith("epoch:"):
        return int(sel.split(":")[1])
    f = Path(ft) / "runs" / run / "selection.json"
    if not f.exists():
        return None
    return json.load(open(f))["selection"].get(sel)


def pred_file(ft, run, epoch, part, split):
    return Path(ft) / "eval" / run / f"e{epoch:02d}" / part / f"pred_{split}.npz"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--export")
    ap.add_argument("--hf-raw")
    a = ap.parse_args(argv)
    ft = Path(a.ft)
    out = Path(a.out or ft / "results")
    out.mkdir(parents=True, exist_ok=True)
    ex = get_export(a.export, a.hf_raw)
    rows, files, accs = {}, {}, {}
    for row, parts in ROWS.items():
        rows[row], files[row], accs[row] = {}, {}, {}
        for part, (run, sel) in parts.items():
            ep = epoch_of(ft, run, sel)
            f = pred_file(ft, run, ep, part, a.split) if ep is not None else None
            if f is None or not f.exists():
                rows[row][part] = dict(missing=str(f) if f else f"{run}: no selection {sel}")
                continue
            pred = P.load(f)
            sc = SC.score_file(pred, ex, split=a.split, boot=True)["subsets"]["all"]
            rows[row][part] = dict(run=run, selected_by=sel, epoch=ep, file=str(f),
                                   **({k: v for k, v in sc.items() if k != "per_pair_acc"}))
            files[row][part] = str(f)
            accs[row][part] = SC.accs_for(pred, ex, a.split)
        # Table 4(a)/(b): How after stage 1 and after the zoom stage, when present
        run, sel = parts["kpt"]
        ep = epoch_of(ft, run, sel)
        for st in ("kpt_stage1", "kpt_stage2"):
            f1 = pred_file(ft, run, ep, st, a.split) if ep is not None else None
            if f1 is not None and f1.exists():
                rows[row][st] = SC.score_file(P.load(f1), ex, split=a.split, boot=True)["subsets"]["all"]
        # Table 4(c) prints Which from the tool head (Table 3 from the predicted pair): baselines.ablate.heads' own metrics
        run, sel = parts["cls"]
        ep = epoch_of(ft, run, sel)
        cj = pred_file(ft, run, ep, "cls", a.split).with_suffix(".check.json") if ep is not None else None
        if cj is not None and cj.exists():
            rows[row]["cls_heads"] = json.load(open(cj)).get("trainer_metrics", {})
    comps = {}
    for x, y in COMPARE:
        c = {}
        for m, (part, key, sign) in METRICS.items():
            if part in accs[x] and part in accs[y] and key in accs[x][part] and key in accs[y][part]:
                r = M.paired_bootstrap(accs[x][part][key], accs[y][part][key])
                lo, hi = r["ci95"]
                better = (lo > 0) if sign > 0 else (hi < 0)
                worse = (hi < 0) if sign > 0 else (lo > 0)
                r["verdict"] = "worse" if worse else ("better" if better else "no clear difference")
                r["higher_is_better"] = sign > 0
                c[m] = r
        comps[f"{x}-{y}"] = c
    runs = {}
    for r in ("control", "joint", "sem", "tmp", "spa"):
        rj = ft / "runs" / r / "run.json"
        if rj.exists():
            info = json.load(open(rj))
            hist = [json.loads(l) for l in open(ft / "runs" / r / "history.jsonl")] if (ft / "runs" / r / "history.jsonl").exists() else []
            runs[r] = dict(desc=info["desc"], trainable_encoder_params=info["encoder"]["trainable_params"],
                           head_params={h["head"]: h["trainable_params"] for h in info["heads"]},
                           epochs_done=max([h["epoch"] for h in hist], default=0),
                           train_hours=sum(h.get("train", {}).get("seconds", 0) for h in hist) / 3600,
                           selection=json.load(open(ft / "runs" / r / "selection.json"))["selection"] if (ft / "runs" / r / "selection.json").exists() else None)
    res = dict(split=a.split, rows=rows, comparisons=comps, runs=runs,
               note="paired bootstraps resample the test videos (score.py); verdicts compare the first row with the second")
    json.dump(res, open(out / f"ft_results_{a.split}.json", "w"), indent=1)
    # Table 3: Which (pair), What, Pair, mAcc | tIoU, R@0.7 | ADE, >32px | Dice.  Table 4: Which (tool head), stage-1/2 ADE, Median.
    cols = [("Which", "pct"), ("Which (tool head)", "pct"), ("What", "pct"), ("Pair", "pct"), ("mAcc", "pct"), ("tIoU", "f3"),
            ("R@0.7", "pct"), ("ADE", "f2"), ("Median", "f2"), (">32px", "pct"), ("Dice", "f3"), ("ADE stage 1", "f2"), ("ADE stage 2", "f2")]
    lines = ["| row | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for row, d in rows.items():
        c, w = d.get("cls", {}), d.get("when", {})
        how, where = d.get("kpt", {}).get("how", {}), d.get("kpt", {}).get("where", {})
        cells = [c.get("which"), d.get("cls_heads", {}).get("tool_acc"), c.get("what"), c.get("pair"), c.get("pair_macc"),
                 w.get("tiou"), w.get("r7"), how.get("ade"), how.get("median_px"), how.get("gt32"), where.get("dice"),
                 d.get("kpt_stage1", {}).get("how", {}).get("ade"), d.get("kpt_stage2", {}).get("how", {}).get("ade")]
        fmt = {"pct": lambda v: f"{100 * v:.1f}", "f3": lambda v: f"{v:.3f}", "f2": lambda v: f"{v:.2f}"}
        lines.append(f"| {row} | " + " | ".join("-" if v is None else fmt[f](v) for v, (_, f) in zip(cells, cols)) + " |")
    lines += ["", "| comparison | metric | diff | 95% CI | verdict |", "|---|---|---|---|---|"]
    for name, c in comps.items():
        for m, r in c.items():
            lines.append(f"| {name} | {m} | {r['diff']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['verdict']} |")
    (out / f"ft_results_{a.split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
