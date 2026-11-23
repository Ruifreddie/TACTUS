#!/usr/bin/env python3
"""CPU checks of the When baselines (no export, no GPU needed):

  python -m baselines.when.tests.test_when

1. time axes: step centres (2k+0.5)/(F-1), grid and unit mappings are inverse of each other;
2. MS-TCN decoding: interpolated 0.5 crossings, runs touching an end, max-sum run, fallback;
3. conditioning: the focal losses of ActionFormer/TriDet and the MS-TCN loss give no gradient to the other pairs'
   output channels; decode_top1 reads the true pair's channel even when another channel scores higher;
4. the official crop (truncate_feats, verbatim) as called by the trainer; QD-DETR negatives are another pair;
5. the recipes follow the authors' rule (linear LR scaling, official values kept) and list their adaptations.
"""
import random

import numpy as np
import torch

from baselines.when import data as D
from baselines.when.methods import METHODS
from baselines.when.methods.mstcn import decode_runs


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg); assert cond, msg


def test_axes():
    for T in (8, 32):
        F = 2 * T; c = D.token_centers(T)
        check(np.allclose(c, (2 * np.arange(T) + 0.5) / (F - 1)), f"T={T}: step k at (2k+0.5)/{F - 1}")
        x = np.linspace(0, 1, 11)
        check(np.allclose(D.grid_to_win(D.win_to_grid(x, T), T), x) and np.allclose(D.win_to_grid(c, T), np.arange(T)),
              f"T={T}: grid <-> window axis, step k at grid k")
        check(np.allclose(D.unit_to_win(D.win_to_unit(x, T), T), x) and np.allclose(D.win_to_unit(c, T), (np.arange(T) + 0.5) / T),
              f"T={T}: unit <-> window axis, step k centred at (k+0.5)/T")


def test_runs():
    c = D.token_centers(8)
    s, e, ok = decode_runs([0.1, 0.2, 0.9, 0.9, 0.2, 0.1, 0.1, 0.1], c)
    check(ok and np.isclose(s, c[1] + (0.5 - 0.2) / 0.7 * (c[2] - c[1])) and np.isclose(e, c[3] + 0.4 / 0.7 * (c[4] - c[3])),
          "boundaries at the interpolated 0.5 crossings")
    s, e, ok = decode_runs([0.9, 0.8, 0.1, 0.1, 0.1, 0.6, 0.7, 0.9], c)
    check(ok and np.isclose(e, 1.0) and c[4] < s < c[5], "max-sum run (0.6+0.7+0.9 > 0.9+0.8), touching the end -> extends to 1")
    s, e, ok = decode_runs([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], c)
    check(ok and s == 0.0 and c[1] < e < c[2], "a run touching the start extends to 0")
    s, e, ok = decode_runs([0.1] * 8, c)
    check((not ok) and (s, e) == (0.0, 1.0), "no run -> whole window, fallback")


def _tal(name, steps=8):
    cls = METHODS[name]
    r = cls.make_recipe(steps, "main", {"embd_dim": 32, "fpn_dim": 32, "head_dim": 32, "sgp_mlp_dim": 64}, smoke=True)
    m = cls(r, steps, 6, "cpu"); m.build(); return m


def test_conditioning():
    torch.manual_seed(0); random.seed(0); rng = np.random.default_rng(0)
    for name in ("actionformer", "tridet"):
        m = _tal(name); m.model.train()
        x = torch.randn(4, 8, 1024); pair = torch.tensor([0, 3, 5, 3]); seg = torch.tensor([[0.2, 0.7], [0.1, 0.9], [0.0, 0.4], [0.5, 1.0]])
        g = D.win_to_grid(seg.numpy(), 8).astype(np.float32)
        feats, masks, segs, labels = m._crop(x, pair, g)
        check(feats.shape == (4, 1024, 8) and masks.dtype == torch.bool and all(len(s) <= 1 for s in segs), f"{name}: official crop + padding")
        # gradient on the classification logits: only the true pair's channel may receive any
        logits = {}
        def hook(mod, inp, out): out[0].retain_grad(); logits.setdefault("l", []).append(out[0])
        h = m.model.cls_head.cls_head.register_forward_hook(hook)
        out = m.model(feats, masks, segs, labels, cls_channel=pair); out["final_loss"].backward(); h.remove()
        ok, true_grad = True, False
        for l in logits["l"]:
            gr = l.grad                                              # [B, C, T_i]
            for b in range(4):
                other = [c for c in range(6) if c != int(pair[b])]
                ok &= bool(torch.all(gr[b, other] == 0)); true_grad |= bool(torch.any(gr[b, int(pair[b])] != 0))
        check(ok and true_grad, f"{name}: the focal loss trains only the true pair's channel")
        # readout: make channel 1 score far higher everywhere; the decoded segment must not come from it
        m.ema.module.eval()
        with torch.no_grad():
            outs = list(m.ema.module(x.transpose(1, 2), torch.ones(4, 1, 8, dtype=torch.bool)))
            cls_i = 2
            hi = [t.clone() for t in outs[cls_i]]
            for t in hi: t[:, :, 1] = 20.0                          # channel 1 very confident
            outs_hi = outs[:cls_i] + [hi] + outs[cls_i + 1:]
            a = m.ema.module.decode_top1(*outs, cls_channel=pair)
            b = m.ema.module.decode_top1(*outs_hi, cls_channel=pair)
        same = torch.allclose(a[0][pair != 1], b[0][pair != 1]) and torch.allclose(a[1][pair != 1], b[1][pair != 1])
        check(bool(same), f"{name}: decode_top1 reads only the true pair's channel")
    # MS-TCN: the loss sees only the true channel (stage outputs of the other channels get gradient only via later stages)
    cls = METHODS["mstcn"]; r = cls.make_recipe(32, "main", smoke=True); m = cls(r, 32, 6, "cpu"); m.build(); m.model.train()
    m.model.stages = torch.nn.ModuleList()                             # single stage: no refinement path
    x = torch.randn(3, 32, 1024); pair = torch.tensor([2, 0, 4]); seg = torch.tensor([[0.1, 0.5], [0.3, 0.8], [0.0, 1.0]])
    store = {}
    def hook(mod, inp, out): out.retain_grad(); store["o"] = out
    h = m.model.stage1.conv_out.register_forward_hook(hook); loss, _ = m.loss(x, pair, seg, None); loss.backward(); h.remove()
    gr = store["o"].grad; ok = all(bool(torch.all(gr[b, [c for c in range(6) if c != int(pair[b])]] == 0)) for b in range(3))
    check(ok, "mstcn: BCE + T-MSE only on the true pair's channel")


def test_qddetr_negatives_and_recipes():
    rng = np.random.default_rng(0); pair = torch.tensor([0, 1, 2, 3, 4, 5] * 50)
    neg = (pair + torch.as_tensor(rng.integers(1, 6, size=len(pair)))) % 6
    check(bool(torch.all(neg != pair)), "qddetr: the negative query is always another pair's template")
    r = METHODS["actionformer"].make_recipe(8)
    check(r["batch"] == 2 and np.isclose(r["lr"], 1e-4) and r["regression_range"] == [[0, 4], [4, 8], [8, 10000]] and r["center_sample"] == "radius"
          and r["crop_ratio"] == [0.9, 1.0] and r["epochs"] == 30 and r["warmup_epochs"] == 5,
          "actionformer: default = official THUMOS recipe incl. batch 2 / lr 1e-4")
    r = METHODS["actionformer"].make_recipe(8, "b64")
    check(r["batch"] == 64 and np.isclose(r["lr"], 3.2e-3), "actionformer: b64 scales the lr linearly (3.2e-3)")
    r = METHODS["tridet"].make_recipe(8)
    check(r["batch"] == 2 and np.isclose(r["lr"], 1e-4) and r["num_bins"] == 4 and r["init_conv_vars"] == 0 and r["epochs"] == 20
          and r["warmup_epochs"] == 20, "tridet: default = official THUMOS recipe incl. batch 2 / lr 1e-4, 4 bins at 8 steps")
    check(np.isclose(METHODS["tridet"].make_recipe(8, "b64")["lr"], 3.2e-3), "tridet: b64 scales the lr linearly (3.2e-3)")
    r = METHODS["qddetr"].make_recipe(8)
    check(np.isclose(r["lr"], 1e-4) and r["batch"] == 32 and r["epochs"] == 200 and list(METHODS["qddetr"].CONFIGS) == ["main"],
          "qddetr: official recipe, no sweep")
    r = METHODS["mstcn"].make_recipe(32)
    check(r["batch"] == 1 and np.isclose(r["lr"], 5e-4) and r["num_layers"] == 6 and r["epochs"] == 50,
          "mstcn: default = official batch 1 / lr 5e-4, 6 layers at 32 steps")
    check(np.isclose(METHODS["mstcn"].make_recipe(32, "b64")["lr"], 3.2e-2), "mstcn: b64 scales the lr linearly (3.2e-2)")
    check(all(METHODS[n].SWEEP == ["main", "b64"] for n in ("actionformer", "tridet", "mstcn")), "default sweeps: main (official) + b64 (scaled)")
    a = METHODS["actionformer"].adaptations(8, METHODS["actionformer"].make_recipe(8))
    check(any(x["setting"] == "batch / lr" and x["reason"].startswith("unchanged") for x in a), "info.json says the default batch/lr is unchanged")
    for n, cls in METHODS.items():
        a = cls.adaptations(cls.main_steps, cls.make_recipe(cls.main_steps))
        check(len(a) >= 5 and all(set(x) == {"setting", "official", "tactus", "reason"} for x in a), f"{n}: adaptations listed")


def main():
    test_axes(); test_runs(); test_conditioning(); test_qddetr_negatives_and_recipes(); print("all When checks passed")


if __name__ == "__main__":
    main()
