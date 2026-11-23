#!/usr/bin/env python3
"""Helpers of tactus_human.sh (TactusNet, human-region targets).

  copy-exp SRC DST        working copy of a UniTAP experiment dir for a smoke run (the original stays untouched):
                          code, data/case32/{v1,v1_zoom} (zoom tokens included), the E007k/C006 dumps and results;
                          symlinks that point into SRC are re-pointed into DST. v1_zoom_ct and v1_zoom_hs are NOT
                          copied: tactus_human.sh rebuilds them with the pipeline's own `labels` step.
  convert --exp --run     contact dumps experiments/analysis/pred_<RUN>_best_<split> (zoom-box coordinates) -> standard
                          kpt files <out>/pred_<split>.npz with ct_xy in content-crop coordinates, mapped with
                          baselines.common.tactus.load_contact (the mapping of UniTAP eval_contact_stage.py); checks that
                          score.py reproduces UniTAP's own region_hull_dice of the same dump.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np

from ..common import preds as P
from ..common import score as SC
from ..common import tactus
from ..common.data import get_export


def _repoint(dst_root: Path, src_roots):
    """Rewrite absolute symlinks under dst_root that point into any of src_roots so that they point into dst_root."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(dst_root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if not p.is_symlink(): continue
            t = os.readlink(p)
            for s in src_roots:
                if t == s or t.startswith(s + os.sep):
                    p.unlink(); p.symlink_to(str(dst_root) + t[len(s):]); n += 1; break
    return n


def copy_exp(src, dst):
    src, dst = Path(src), Path(dst)
    if (dst / ".copy_ok").exists(): print("exists", dst); return
    if dst.exists(): shutil.rmtree(dst)
    (dst / "data/case32").mkdir(parents=True); (dst / "experiments/analysis").mkdir(parents=True)
    for d in ("tools", "third_party", "results"):
        if (src / d).exists(): shutil.copytree(src / d, dst / d, symlinks=True)
    raw = src / "data/raw_case32"
    if raw.is_symlink(): os.symlink(os.readlink(raw), dst / "data/raw_case32")
    elif raw.exists(): shutil.copytree(raw, dst / "data/raw_case32", symlinks=True)
    for d in ("v1", "v1_zoom"):
        shutil.copytree(src / "data/case32" / d, dst / "data/case32" / d, symlinks=True)
    for p in sorted((src / "experiments/analysis").glob("pred_*_best_*")):
        shutil.copytree(p, dst / "experiments/analysis" / p.name, symlinks=True)
    n = _repoint(dst, {str(src), os.path.realpath(src)})
    (dst / ".copy_ok").write_text(f"copy of {src}\n")
    print(f"copied {src} -> {dst} ({n} symlinks re-pointed)")


def convert(exp, run, splits, out, export=None, hf_raw=None):
    ex = get_export(export, hf_raw); E = Path(exp); res = {}
    boxes = json.load(open(E / "data/case32/v1_zoom/boxes.json"))["boxes"]     # the zoom boxes the dumps were made in
    for split in splits:
        dump = E / f"experiments/analysis/pred_{run}_best_{split}"; cont = tactus.load_contact(dump, boxes, ex)
        sids = ex.sids(split); miss = [s for s in sids if s not in cont]
        if miss: raise SystemExit(f"{len(miss)} {split} clips missing from {dump}, e.g. {miss[:5]}")
        info = {"method": "tactusnet_human_targets", "split": split, "exp": str(E), "run": run, "dump": str(dump),
                "conditioning": "true tool and action", "targets": "human contact points (data/case32/v1_zoom_ct)"}
        p = P.save(Path(out) / f"pred_{split}.npz", "kpt", sids, info, ct_xy=np.stack([cont[s] for s in sids]))
        r = SC.score_file(P.load(p), ex, split=split, boot=False)["subsets"]["all"]["where"]
        ref_p = E / f"results/contact_human_{run}_{split}.json"
        ref = json.load(open(ref_p))["summary"]["region_hull_dice"]["mean"] if ref_p.exists() else None
        res[split] = {"dice": r["dice"], "unitap_region_hull_dice": ref, "frames": r["n"]["dice"]}
        print(f"[convert] {split}: {p}  Where Dice {r['dice']:.6f}" + (f"  (UniTAP eval_contact_stage {ref:.6f})" if ref is not None else ""))
        if ref is not None and abs(r["dice"] - ref) > 1e-9: raise SystemExit(f"score.py Dice {r['dice']} != UniTAP {ref}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy-exp"); c.add_argument("src"); c.add_argument("dst")
    v = sub.add_parser("convert"); v.add_argument("--exp", required=True); v.add_argument("--run", default="C006H")
    v.add_argument("--splits", default="val,test"); v.add_argument("--out", required=True); v.add_argument("--export"); v.add_argument("--hf-raw")
    a = ap.parse_args(argv)
    if a.cmd == "copy-exp": copy_exp(a.src, a.dst)
    else:
        res = convert(a.exp, a.run, a.splits.split(","), a.out, a.export, a.hf_raw)
        json.dump(res, open(Path(a.out) / "convert_check.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
