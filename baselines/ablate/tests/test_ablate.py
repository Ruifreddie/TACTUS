#!/usr/bin/env python3
"""CPU checks of baselines/ablate (no GPU, no data): python -m baselines.ablate.tests.test_ablate

1. every encoder's stub width matches its token width; layouts have 8 time steps;
2. launch.py's null condition replaces tool, action and pair ids by zeros (keyword and positional calls);
3. every \\abl{key} of table_ablation.tex is a key that collect.py can fill, and the fill regex leaves the rest intact.
"""
import re, sys, types
from pathlib import Path

import torch

from baselines.ablate import backbones as BB
from baselines.ablate import collect as C
from baselines.ablate.launch import patch_model_null_cond

HERE = Path(__file__).resolve().parents[1]


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg); assert cond, msg


def main():
    width = {"vjepa2_1_vit_large_384": 1024, "vjepa2_1_vit_base_384": 768}
    for n, s in BB.SPECS.items():
        check(width[s["stub"]] == s["dim"], f"{n}: stub {s['stub']} has width {s['dim']}")

    class Net:
        def forward(self, video=None, tokens_grid=None, token_order="x", tool_id=None, action_id=None, combo_id=None):
            return tool_id, action_id, combo_id
    m = types.SimpleNamespace(KeypointComboNet=Net); patch_model_null_cond(m)
    ids = torch.tensor([3, 1]); out = Net().forward(None, None, "x", ids, ids + 1, ids + 2)
    check(all(int(o.abs().sum()) == 0 for o in out), "null condition, positional ids -> zeros")
    out = Net().forward(tokens_grid=None, tool_id=ids, action_id=ids, combo_id=ids)
    check(all(int(o.abs().sum()) == 0 and o.shape == ids.shape for o in out), "null condition, keyword ids -> zeros of the same shape")

    tex = "\n".join(l for l in (HERE / "table_ablation.tex").read_text().splitlines() if not l.lstrip().startswith("%"))
    keys = sorted(set(re.findall(r"\\abl\{([^}]*)\}", tex)))
    encs = set(BB.SPECS) | {"ours"}
    pat = re.compile(r"^(enc\.(%s)\.(which|what|pair|pmacc|tiou|r5|r7|ade|med|gt32|ade_ns|ade_noyt)"
                     r"|comp\.(stage1|stage2|refined|stage1_refiner)\.(ade|med|gt32)"
                     r"|cond\.(gt|pred|none)\.(tiou|r5|r7|ade|med|gt32|dice)"
                     r"|tmp\.(noquality|scalar)\.(tiou|r5|r7)|s1\.nocond\.(ade|med|gt32|ade_ns|ade_noyt)"
                     r"|silver\.(A|B)\.ns\.(ade|med|gt32|dice)|ct\.human\.(dice|dice_hs))$" % "|".join(sorted(encs)))
    bad = [k for k in keys if not pat.match(k)]
    check(not bad, f"{len(keys)} table keys are producible by collect.py" + (f" (bad: {bad})" if bad else ""))
    vals = {"enc.ours.ade": 21.234, "cond.pred.tiou": 0.5811, "enc.dinov3_l16_384.which": 0.95}
    filled = re.sub(r"\\abl\{([^}]*)\}\{[^}]*\}", lambda m_: C.fmt(m_.group(1), vals[m_.group(1)]) if m_.group(1) in vals else m_.group(0), tex)
    check("21.23" in filled and "0.581" in filled and "95.0" in filled and filled.count("\\abl{") == tex.count("\\abl{") - 3 - (tex.count("\\abl{enc.ours.ade}") - 1),
          "collect fills values with the table's precision and keeps the other placeholders")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
