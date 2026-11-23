#!/usr/bin/env python3
"""Score every ablation row with baselines.common.score and fill table_ablation.tex.

Reads the prediction files that run.sh writes under $AO (= $BL_OUT/ablate):
  enc/<encoder>/{cls,when,stage1}/pred_test.npz     (b) frozen-encoder swap (ours = TactusNet's own heads and tokens)
  components/{stage1,stage2,refined,stage1_refiner}/pred_test.npz                            (c) Spatial Head
  cond/pred/when, cond/cascade/{gt,pred}, temporal/ours/nocond                                (d) conditions
  temporal/ours/{noquality,scalar}, stage1/ours/nocond                                       supplement (f), (g)
  silver/{A,B}/pred_test.npz                                                                 (a) Gold vs Gold + Silver
and, for (e), the human-target retrain of baselines/where (--where: pred_test.npz, unitap/contact_hindsight_*_test.json).
Writes $AO/results.json (all metrics with 95% bootstrap CIs over test videos, paired differences of every variant against
its reference row) and the table with every available \\abl{key}{...} replaced by its value.

  python -m baselines.ablate.collect --ao $AO --template baselines/ablate/table_ablation.tex --out $AO/table_ablation.tex [--where DIR]
"""
import argparse, glob, json, re, sys
from pathlib import Path

from baselines.common import metrics as M
from baselines.common import preds as P
from baselines.common import score as SC
from baselines.common.data import get_export

PCT = ("which", "what", "pair", "pmacc", "r5", "r7", "gt32")


def fmt(key, v):
    k = key.split(".")[-1].replace("_ns", "").replace("_noyt", "")
    if k in PCT: return f"{100 * v:.1f}"
    if k.startswith("ade"): return f"{v:.2f}"
    if k == "med": return f"{v:.1f}"
    return f"{v:.3f}"                                     # tiou, dice


def metrics_of(r, kind):
    if kind == "cls": return {"which": r["which"], "what": r["what"], "pair": r["pair"], "pmacc": r["pair_macc"]}
    if kind == "when": return {"tiou": r["tiou"], "r5": r["r5"], "r7": r["r7"]}
    out = {}
    if "how" in r: out.update(ade=r["how"]["ade"], med=r["how"]["median_px"], gt32=r["how"]["gt32"], pck4=r["how"]["pck4"])
    if "where" in r: out.update(dice=r["where"]["dice"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--ao", required=True); ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--where", default=None, help="baselines/where human-target output dir")
    ap.add_argument("--exp-a"); ap.add_argument("--exp-b"); ap.add_argument("--export"); ap.add_argument("--hf-raw")
    ap.add_argument("--no-boot", action="store_true"); a = ap.parse_args(argv)
    ex = get_export(a.export, a.hf_raw); AO = Path(a.ao); vals, rows, preds = {}, {}, {}
    yt = tuple(sorted({ex.task(s) for s in ex.sids("test") if ex.source(s) == "YouTube"}))

    def add(row, f, vkey):
        """Score file f as row `row`; its metrics become values vkey.<metric> (and vkey.<metric>_ns on the 8 non-SurgVU videos)."""
        f = Path(f)
        if not f.exists(): return None
        p = P.load(f); preds[row] = p; sc = SC.score_file(p, ex, boot=not a.no_boot)["subsets"]
        rows[row] = {"file": str(f), "info": p["info"], "all": sc.get("all"), "non_surgvu": sc.get("non_surgvu")}
        for k, v in metrics_of(sc["all"], p["kind"]).items(): vals[f"{vkey}.{k}"] = v
        if "non_surgvu" in sc:
            for k, v in metrics_of(sc["non_surgvu"], p["kind"]).items(): vals[f"{vkey}.{k}_ns"] = v
        if p["kind"] == "kpt" and yt and "tool_xy" in p:  # LEMON was collected from YouTube: ADE without the YouTube test videos
            r = SC.score_file(p, ex, exclude=yt, boot=False)["subsets"]["all"]; vals[f"{vkey}.ade_noyt"] = r["how"]["ade"]
        return p

    encs = sorted(d.name for d in (AO / "enc").glob("*")) if (AO / "enc").exists() else []
    for e in encs:
        for part in ("cls", "when", "stage1"): add(f"enc.{e}.{part}", AO / f"enc/{e}/{part}/pred_test.npz", f"enc.{e}")
    for st in ("stage1", "stage2", "refined", "stage1_refiner"): add(f"comp.{st}", AO / f"components/{st}/pred_test.npz", f"comp.{st}")
    add("cond.pred.when", AO / "cond/pred/when/pred_test.npz", "cond.pred")
    add("cond.gt.kpt", AO / "cond/cascade/gt/pred_test.npz", "cond.gt")
    add("cond.pred.kpt", AO / "cond/cascade/pred/pred_test.npz", "cond.pred")
    add("cond.none.when", AO / "temporal/ours/nocond/pred_test.npz", "cond.none")
    for v in ("noquality", "scalar"): add(f"tmp.{v}.when", AO / f"temporal/ours/{v}/pred_test.npz", f"tmp.{v}")
    add("s1.nocond.stage1", AO / "stage1/ours/nocond/pred_test.npz", "s1.nocond")
    for E in ("A", "B"):
        if add(f"silver.{E}.kpt", AO / f"silver/{E}/pred_test.npz", f"silver.{E}") is not None:
            for k in ("ade", "med", "gt32", "dice"):
                if f"silver.{E}.{k}_ns" in vals: vals[f"silver.{E}.ns.{k}"] = vals[f"silver.{E}.{k}_ns"]
    if a.where:
        W = Path(a.where); add("ct.human.kpt", W / "pred_test.npz", "ct.human")
        hs = sorted(glob.glob(str(W / "unitap/contact_hindsight_*_test.json")))
        if hs:                                            # UniTAP eval_hindsight.py: {"dice_mean": ...} (older files: {"summary": {...}})
            d = json.load(open(hs[0])); vals["ct.human.dice_hs"] = d.get("summary", d)["dice_mean"]

    # paired bootstrap over the test videos: every variant against its reference row (difference = variant - reference)
    pairs = [("silver.B.kpt", "silver.A.kpt"), ("cond.pred.kpt", "cond.gt.kpt"), ("cond.pred.when", "enc.ours.when"),
             ("cond.none.when", "enc.ours.when"), ("tmp.noquality.when", "enc.ours.when"), ("tmp.scalar.when", "enc.ours.when"),
             ("comp.stage2", "comp.stage1"), ("comp.refined", "comp.stage2"), ("comp.stage1_refiner", "comp.stage1"),
             ("s1.nocond.stage1", "enc.ours.stage1")]
    pairs += [(f"enc.{e}.{p}", f"enc.ours.{p}") for e in encs if e != "ours" for p in ("cls", "when", "stage1")]
    paired = {}
    if not a.no_boot:
        for x, y in pairs:
            if x in preds and y in preds and preds[x]["kind"] == preds[y]["kind"]:
                A, B = SC.accs_for(preds[x], ex), SC.accs_for(preds[y], ex)
                paired[f"{x} - {y}"] = {k: M.paired_bootstrap(A[k], B[k]) for k in A if k in B}
    json.dump({"values": vals, "rows": rows, "paired": paired, "youtube_test_tasks": len(yt)}, open(AO / "results.json", "w"), indent=1)

    tex = open(a.template).read(); filled = []

    def sub(m):
        k = m.group(1)
        if k in vals and vals[k] == vals[k]: filled.append(k); return fmt(k, vals[k])
        return m.group(0)
    tex = re.sub(r"\\abl\{([^}]*)\}\{[^}]*\}", sub, tex)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); open(a.out, "w").write(tex)
    left = sorted(set(re.findall(r"\\abl\{([^}]*)\}", tex)))
    print(json.dumps({"results": str(AO / "results.json"), "table": a.out, "filled": len(set(filled)), "placeholders_left": left}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
