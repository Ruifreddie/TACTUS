"""MS-TCN: multi-stage dilated temporal convolutions, framewise (stepwise) in/out labels, then the main segment.

Official configuration kept (main.py / model.py): 4 stages of dilated residual layers (dilation 2^i), 64 feature maps,
dropout 0.5, the loss (per-stage classification loss + 0.15 x truncated MSE with tau = 4, summed over the stages), Adam
5e-4, batch 1, 50 epochs. What TACTUS changes (adaptations() lists them): (a) six per-pair sigmoid channels with the loss
on the true pair's channel; (b) layers per stage matched to the sequence (6 at T = 32: dilations 1..32; the official 10
is the extra config `l10`). The default keeps the official batch 1 and LR 5e-4 (one window per step); the sweep entry
`b64` (batch 64, LR scaled linearly to 3.2e-2) is an extra config that val may choose.

Six per-pair sigmoid channels instead of a 7-way softmax, so a wrong pair cannot suppress the true channel. Each step k
gets target 1 on the true pair's channel if its centre x_k = (2k + 0.5) / (F - 1) lies inside the annotated interval,
else 0; the loss sees only the true pair's channel (the other five are masked). The refinement stages take the sigmoid
of the previous stage's six channels (the original takes the softmax over its classes).

Loss per stage, summed over the stages as in the paper: binary cross-entropy on the true channel, plus lambda times the
truncated MSE smoothing on its log-probabilities, here the binary pair (log p, log(1 - p)):
mean over steps of min((log y_t - log y_{t-1}.detach())^2, tau^2), lambda = 0.15, tau = 4.

Decoding (fixed a priori, frozen on val): threshold the last stage's true-channel probability at 0.5, take the
contiguous run of steps above it with the highest summed probability, and put each boundary at the linearly
interpolated 0.5 crossing between neighbouring step centres; a run touching the first or last step extends to 0 or 1.
No run: the whole window, flagged as fallback.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..data import token_centers
from .base import Method, adapt, batch_adapt, finish


def decode_runs(p, centers, thr=0.5):
    """p [T] probabilities of the true channel, centers [T] on the window axis -> (start, end, found)."""
    p = np.asarray(p, np.float64); above = p > thr; T = len(p)
    if not above.any(): return 0.0, 1.0, False
    runs, k = [], 0
    while k < T:
        if above[k]:
            a = k
            while k + 1 < T and above[k + 1]: k += 1
            runs.append((a, k))
        k += 1
    a, b = max(runs, key=lambda r: p[r[0]:r[1] + 1].sum())                       # first run on ties
    s = 0.0 if a == 0 else centers[a - 1] + (thr - p[a - 1]) / (p[a] - p[a - 1]) * (centers[a] - centers[a - 1])
    e = 1.0 if b == T - 1 else centers[b] + (p[b] - thr) / (p[b] - p[b + 1]) * (centers[b + 1] - centers[b])
    return float(s), float(e), True


class MSTCN(Method):
    name, title = "mstcn", "MS-TCN"
    paper = "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation, CVPR 2019 (Farha, Gall)"
    code_url = "https://github.com/yabufarha/ms-tcn"
    official_source = "main.py + model.py @ 33ed91c0"
    main_steps = 32
    sched_per = "iter"
    OFFICIAL = dict(features="I3D (2048-d, 15 fps), one whole video per sample", num_layers=10, batch=1, lr=5e-4, epochs=50,
                    classes="softmax over the dataset's action classes")
    DEFAULTS = dict(num_stages=4, num_layers=None, num_f_maps=64, dropout=0.5, base_lr=5e-4, base_batch=1, batch=1, lr=None,
                    epochs=50, tmse_lambda=0.15, tmse_tau=4.0, threshold=0.5)
    CONFIGS = {
        "main": {},                                                    # 6 layers per stage at T=32, the official batch 1, lr 5e-4
        "b64": {"batch": 64},                                          # 6 layers, batch 64, lr scaled linearly to 3.2e-2
        "l10": {"num_layers": 10},                                     # the official 10 layers, batch 1 (optional)
        "l10_b64": {"num_layers": 10, "batch": 64},                    # the official 10 layers, batch 64 (optional)
    }
    SWEEP = ["main", "b64"]
    SMOKE = dict(epochs=2)
    conditioning = "6 per-pair sigmoid channels; BCE + T-MSE only on the true pair's channel, the others masked; test reads it"
    decoding = ("max-sum contiguous run of p > 0.5 on the true channel, boundaries at the interpolated 0.5 crossings between "
                "step centres (runs touching an end extend to 0 or 1); else whole window")

    @classmethod
    def derived(cls, steps, r):
        d = super().derived(steps, r)
        d["num_layers"] = int(np.log2(steps)) + 1 if r.get("num_layers") is None else int(r["num_layers"])   # largest dilation = T
        return d

    @classmethod
    def adaptations(cls, steps, r):
        return [
            adapt("classes", cls.OFFICIAL["classes"], "6 per-pair sigmoid channels, BCE + binary T-MSE on the true pair's channel only",
                  "(a) a softmax would let a wrong pair suppress the true channel; recognition errors are kept out of When"),
            adapt("refinement-stage input", "softmax of the previous stage", "sigmoid of the previous stage's 6 channels",
                  "(a) follows from the sigmoid channels"),
            adapt("layers per stage", cls.OFFICIAL["num_layers"], r["num_layers"],
                  f"(b) dilations 1..{2 ** (r['num_layers'] - 1)} for {steps} steps (the official 10 reach 512; config l10 keeps them)"),
            batch_adapt(r, unit="video"),
            adapt("schedule", f"{cls.OFFICIAL['epochs']} epochs", f"{r['epochs']} epochs; iterations per epoch = training windows / batch",
                  "(c) official epochs over our windows"),
            adapt("decoding", "framewise argmax", "max-sum run of p > 0.5 on the true channel, interpolated 0.5 crossings, ends extend to 0/1; "
                  "whole window if no run", "(a) one interval per window from the true channel"),
            adapt("input features", cls.OFFICIAL["features"], f"V-JEPA 2.1 ViT-L/384 (frozen, TactusNet), 1024-d, {steps} steps",
                  "the shared TACTUS features"),
        ]

    def build(self):
        from ..third_party.ms_tcn import MultiStageModel
        r = self.recipe
        if float(r["dropout"]) != 0.5: raise SystemExit("MS-TCN's dropout is fixed at 0.5 in the vendored model (nn.Dropout())")
        self.model = MultiStageModel(int(r["num_stages"]), int(r["num_layers"]), int(r["num_f_maps"]), 1024, self.n_pairs,
                                     stage_input="sigmoid").to(self.device)
        self.centers = token_centers(self.steps)
        self.centers_t = torch.as_tensor(self.centers, dtype=torch.float32, device=self.device)

    def make_optimizer(self, iters_per_epoch):
        return torch.optim.Adam(self.model.parameters(), lr=float(self.recipe["lr"])), None

    def _forward(self, x):
        feats = x.transpose(1, 2).contiguous()                                        # [B, D, T]
        mask = torch.ones(feats.shape[0], self.n_pairs, feats.shape[2], device=feats.device)
        return self.model(feats, mask)                                                # [S, B, C, T]

    def loss(self, x, pair, seg, rng):
        r = self.recipe; preds = self._forward(x); S, B, C, T = preds.shape
        z = preds.gather(2, pair.view(1, B, 1, 1).expand(S, B, 1, T)).squeeze(2)      # [S, B, T] true channel
        y = ((self.centers_t[None] >= seg[:, :1]) & (self.centers_t[None] <= seg[:, 1:])).float()   # [B, T]
        tau2 = float(r["tmse_tau"]) ** 2; loss = 0; logs = {}
        for s in range(S):
            ce = F.binary_cross_entropy_with_logits(z[s], y)
            logp = torch.stack([F.logsigmoid(z[s]), F.logsigmoid(-z[s])], dim=1)       # [B, 2, T]
            tmse = torch.clamp((logp[:, :, 1:] - logp.detach()[:, :, :-1]) ** 2, min=0, max=tau2).mean()
            loss = loss + ce + float(r["tmse_lambda"]) * tmse
            if s == S - 1: logs.update(bce_last=float(ce.detach()), tmse_last=float(tmse.detach()))
        logs["loss"] = float(loss.detach())
        return loss, logs

    @torch.no_grad()
    def predict(self, x, pair):
        self.model.eval()
        preds = self._forward(x); S, B, C, T = preds.shape
        p = torch.sigmoid(preds[-1].gather(1, pair.view(B, 1, 1).expand(B, 1, T)).squeeze(1)).cpu().numpy()
        out = [decode_runs(pi, self.centers, float(self.recipe["threshold"])) for pi in p]
        s, e, found = (np.array(v) for v in zip(*out)) if out else (np.zeros(0), np.zeros(0), np.zeros(0, bool))
        return finish(s, e, found)
