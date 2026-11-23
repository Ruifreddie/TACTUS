"""Regenerate tactusnet/_vendored_heads.py and tactusnet/_vendored_vjepa_wrapper.py verbatim from their sources.

    python -I tactusnet/tools/extract_vendored.py --zip UniTAP_project.zip

The head blocks are cut out of third_party/t2b/kpt_combo/model.py (inside the UniTAP zip) with `ast`, so nothing is retyped.
The V-JEPA wrapper is copied byte for byte from repro_exports/affordance_vjepa21_code_20260911/affordance/vjepa_wrapper.py.
"""
import argparse, ast, hashlib, textwrap, zipfile
from pathlib import Path

MODEL_IN_ZIP = "third_party/t2b/kpt_combo/model.py"
TOP_LEVEL = ["_pick_attn_heads", "_normalize_class_ids", "_build_trajectory_condition", "CrossAttentionPooler", "LinearClassifierHead",
             "ToolConditionedActionPrior", "ComboHead", "TemporalSoLaAdapter", "TemporalRangeOffsetHead"]
STATIC_IN_NET = [("KeypointComboNet", "_decode_range_from_points", "decode_range_from_points")]
HEADER = '''"""VERBATIM EXTRACT -- do not edit by hand; regenerate with tactusnet/tools/extract_vendored.py.

Source : {src}  (sha256 {sha})
Blocks : {blocks}
Only the module-level `_decode_range_from_points` staticmethod was un-indented, de-decorated and renamed `decode_range_from_points`.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


'''


def seg(src: str, node: ast.AST) -> str:
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    lines = src.splitlines()[first - 1: node.end_lineno]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--zip", required=True); ap.add_argument("--wrapper", default=None); ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args(); out = Path(a.out)
    src = zipfile.ZipFile(a.zip).read(MODEL_IN_ZIP).decode("utf-8"); sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    tree = ast.parse(src); top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    blocks, names = [], []
    for name in TOP_LEVEL:
        blocks.append(seg(src, top[name])); names.append(name)
    for cls, fn, new in STATIC_IN_NET:
        cnode = top[cls]; fnode = next(n for n in cnode.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        text = textwrap.dedent(seg(src, fnode)); text = "\n".join(l for l in text.splitlines() if l.strip() != "@staticmethod").replace(f"def {fn}(", f"def {new}(", 1)
        blocks.append(text); names.append(f"{cls}.{fn} -> {new}")
    (out / "_vendored_heads.py").write_text(HEADER.format(src=f"{Path(a.zip).name}::{MODEL_IN_ZIP}", sha=sha, blocks=", ".join(names)) + "\n\n\n".join(blocks) + "\n", encoding="utf-8")
    print("wrote", out / "_vendored_heads.py", "| source sha256", sha[:12], "| blocks:", len(blocks))
    if a.wrapper:
        data = Path(a.wrapper).read_bytes(); (out / "_vendored_vjepa_wrapper.py").write_bytes(data)
        print("wrote", out / "_vendored_vjepa_wrapper.py", "| sha256", hashlib.sha256(data).hexdigest()[:12])


if __name__ == "__main__":
    main()
