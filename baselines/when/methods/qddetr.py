"""QD-DETR: query-dependent DETR for moment retrieval, with one fixed text template per pair as the query.

Official configuration kept (qd_detr/config.py defaults + scripts/train.sh of QVHighlights): architecture (T2V encoder,
2 + 2 layers, hidden 256, 10 queries, DAB-style decoder), losses (L1 + gIoU + CE with Hungarian matching, saliency
margin + rank-contrastive + negative-pair losses), AdamW 1e-4, weight decay 1e-4, batch 32, 200 epochs, StepLR(400)
per epoch, gradient clipping 0.1, top-1 moment by foreground score. What TACTUS changes (adaptations() lists them):
video features, the query, the time axis, the saliency labels, the negative query and the rounding of the output.

Video: the frozen V-JEPA 2.1 sequence only (no CLIP image features, so every row shares the same video features),
L2-normalized per step plus QD-DETR's temporal endpoint features (ctx_mode video_tef). Query: CLIP ViT-B/32 text-token
features (last_hidden_state, L2-normalized per token, as officially) of the fixed template of the true pair. The
templates below were written once before any run and are never tuned. Output: the top-1 query slot by foreground
probability, its (centre, width) mapped to the window axis and clipped to it (the official clip_ts). QD-DETR's official
round_multiple step, which snaps boundaries to multiples of QVHighlights' 2-s clips (its labels are quantized to them),
is not applied: TACTUS intervals are continuous. The model always outputs a moment, so there is no fallback.

Time mapping: QD-DETR normalizes spans by the clip length, step k spanning [k/T, (k+1)/T]; with step k centred at
x = (2k + 0.5) / (F - 1) on the window axis this is x = (F u - 0.5) / (F - 1) (data.unit_to_win).

Saliency targets (training only): 1 for the steps whose centre lies inside the annotated interval (the step nearest to
the interval centre if none does), 0 outside; QD-DETR's two positive and two negative steps are sampled from them, as
its loader does for queries without saliency scores (get_saliency_labels_sub_as_query). If no step lies outside the
interval, the negatives repeat the positives, which zeroes that clip's margin term.
Negative query: the template of a different, randomly drawn pair (the original rolls the batch, which with six queries
would often give the clip's own query).
"""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ..data import token_centers, unit_to_win, win_to_unit
from .base import Method, adapt, finish

PAIR_TEMPLATES = {                       # fixed English template per valid pair (written once; never tuned on val or test)
    "hook|dissect": "a hook dissects tissue",
    "grasper|grasp": "a grasper grasps tissue",
    "clipper|clip": "a clipper applies a clip",
    "scissors|cut": "scissors cut tissue",
    "scissors|dissect": "scissors dissect tissue",
    "grasper|dissect": "a grasper dissects tissue",
}


class QDDETR(Method):
    name, title = "qddetr", "QD-DETR"
    paper = "Query-Dependent Video Representation for Moment Retrieval and Highlight Detection, CVPR 2023 (Moon et al.)"
    code_url = "https://github.com/wjun0830/QD-DETR"
    official_source = "qd_detr/config.py + qd_detr/scripts/train.sh (QVHighlights) @ f8628f79"
    sched_per = "epoch"                  # QD-DETR steps its StepLR once per epoch
    OFFICIAL = dict(features="SlowFast (2304-d) + CLIP ViT-B/32 image (512-d) per 2-s clip, + TEF", query="the natural-language query",
                    batch=32, lr=1e-4, epochs=200)
    DEFAULTS = dict(enc_layers=2, dec_layers=2, dim_feedforward=1024, hidden_dim=256, input_dropout=0.5, dropout=0.1, nheads=8,
                    num_queries=10, pre_norm=False, n_input_proj=2, position_embedding="sine", use_txt_pos=False, max_q_l=32, max_v_l=75,
                    contrastive_align_loss=False, contrastive_hdim=64, temperature=0.07, lw_saliency=1.0, saliency_margin=0.2,
                    aux_loss=True, span_loss_type="l1", set_cost_span=10, set_cost_giou=1, set_cost_class=4, span_loss_coef=10,
                    giou_loss_coef=1, label_loss_coef=4, eos_coef=0.1, contrastive_align_loss_coef=0.0,
                    normalize_v=True, normalize_t=True, use_tef=True, base_lr=1e-4, base_batch=32, batch=32, lr=None, wd=1e-4,
                    lr_drop=400, epochs=200, grad_clip=0.1)
    CONFIGS = {"main": {}}               # the official configuration has no free parameter left to adapt (batch 32 = official)
    SMOKE = dict(epochs=2, batch=8)
    conditioning = "the true pair's fixed CLIP ViT-B/32 text template is the query (class-conditioned single-moment regression)"
    decoding = "top-1 query slot by foreground probability, (centre, width) -> [start, end], clipped to the window (official clip_ts)"

    @classmethod
    def adaptations(cls, steps, r):
        return [
            adapt("video features", cls.OFFICIAL["features"], f"V-JEPA 2.1 ViT-L/384 (frozen, TactusNet), 1024-d, {steps} steps, + TEF",
                  "the shared TACTUS features; no CLIP image features so that every row uses the same video features"),
            adapt("query", cls.OFFICIAL["query"], "a fixed template per pair: " + "; ".join(f"{k} = '{v}'" for k, v in PAIR_TEMPLATES.items()),
                  "(a) the true pair is given; written once, never tuned"),
            adapt("time axis", "2-s clips, spans normalized by the video length", f"step k spans [k/T, (k+1)/T], centred at (2k+0.5)/{2 * steps - 1} "
                  "on the window axis", "(b) one window = T steps"),
            adapt("saliency labels", "annotator scores (QVHighlights); inside/outside the window for queries without scores",
                  "1 for steps whose centre is inside the interval (nearest step if none), 0 outside; 2 positives / 2 negatives sampled "
                  "from them (official get_saliency_labels_sub_as_query); negatives repeat the positives if no step is outside",
                  "TACTUS has intervals, no saliency scores"),
            adapt("negative query", "the batch rolled by one", "the template of a different, random pair",
                  "(a) with six queries the rolled query is often the clip's own (about half the time at the Gold pair frequencies)"),
            adapt("output rounding", "round_multiple to the 2-s clip length (QVHighlights labels are clip-quantized)", "none (clip_ts only)",
                  "TACTUS intervals are continuous on the window axis"),
            adapt("batch / lr / epochs", f"{r['base_batch']} / {r['base_lr']} / 200", f"{r['batch']} / {r['lr']:.1e} / {r['epochs']}",
                  "(c, d) unchanged: official batch and epochs; iterations follow from our data size"),
        ]

    def __init__(self, recipe, steps, n_pairs, device, text_feats=None):
        super().__init__(recipe, steps, n_pairs, device)
        self.text_feats = text_feats

    def build(self):
        from ..third_party.qd_detr import build_model
        r = self.recipe
        if self.text_feats is None: raise SystemExit("QD-DETR needs the CLIP text features: python -m baselines.when.features text")
        tf = [np.asarray(self.text_feats[p], np.float32) for p in range(self.n_pairs)]
        if r["normalize_t"]: tf = [t / np.linalg.norm(t, axis=-1, keepdims=True) for t in tf]      # l2_normalize_np_array
        tf = [t[:r["max_q_l"]] for t in tf]; L = max(len(t) for t in tf); D = tf[0].shape[1]
        txt = np.zeros((self.n_pairs, L, D), np.float32); txt_mask = np.zeros((self.n_pairs, L), np.float32)
        for p, t in enumerate(tf): txt[p, :len(t)] = t; txt_mask[p, :len(t)] = 1
        self.txt = torch.from_numpy(txt).to(self.device); self.txt_mask = torch.from_numpy(txt_mask).to(self.device)
        v_dim = 1024 + (2 if r["use_tef"] else 0)
        args = SimpleNamespace(**r, device=str(self.device), a_feat_dir=None, a_feat_dim=None, dset_name="hl", v_feat_dim=v_dim, t_feat_dim=D)
        self.model, self.criterion = build_model(args)
        self.model.to(self.device); self.criterion.to(self.device)
        self.weight_dict = self.criterion.weight_dict
        cen = token_centers(self.steps); self.centers = cen

    def make_optimizer(self, iters_per_epoch):
        r = self.recipe
        param_dicts = [{"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}]
        opt = torch.optim.AdamW(param_dicts, lr=r["lr"], weight_decay=r["wd"])
        return opt, torch.optim.lr_scheduler.StepLR(opt, int(r["lr_drop"]))

    def grad_clip(self):
        v = float(self.recipe["grad_clip"]); return v if v > 0 else None

    def train_mode(self):
        self.model.train(); self.criterion.train()

    def _video(self, x):
        r = self.recipe
        if r["normalize_v"]: x = x / x.norm(dim=-1, keepdim=True)                 # l2_normalize_np_array
        B, T, _ = x.shape
        if r["use_tef"]:
            st = torch.arange(0, T, 1.0, device=x.device) / T; tef = torch.stack([st, st + 1.0 / T], dim=1)   # (Lv, 2)
            x = torch.cat([x, tef[None].expand(B, -1, -1)], dim=2)
        return x, torch.ones(B, T, device=x.device)

    def _targets(self, pair, seg, rng):
        """Span labels (cxw on QD-DETR's normalized axis) and saliency labels, as described in the module docstring."""
        B = len(pair); T = self.steps; seg = seg.detach().cpu().numpy().astype(np.float64)
        u = np.clip(win_to_unit(seg, self.steps), 0.0, 1.0)                        # [B, 2]
        spans = torch.as_tensor(np.stack([u.mean(1), u[:, 1] - u[:, 0]], 1), dtype=torch.float32, device=self.device)
        pos_idx = np.zeros((B, 2), np.int64); neg_idx = np.zeros((B, 2), np.int64); sal = np.zeros((B, T), np.float32)
        for i in range(B):
            inside = (self.centers >= seg[i, 0]) & (self.centers <= seg[i, 1])
            if not inside.any(): inside[np.argmin(np.abs(self.centers - seg[i].mean()))] = True
            sal[i] = inside; pos = np.flatnonzero(inside); neg = np.flatnonzero(~inside)
            pos_idx[i] = rng.choice(pos, 2, replace=False) if len(pos) >= 2 else [pos[0], pos[0]]
            neg_idx[i] = rng.choice(neg, 2, replace=len(neg) < 2) if len(neg) else pos_idx[i]
        return {"span_labels": [dict(spans=spans[i:i + 1]) for i in range(B)],
                "saliency_pos_labels": torch.as_tensor(pos_idx, device=self.device),
                "saliency_neg_labels": torch.as_tensor(neg_idx, device=self.device),
                "saliency_all_labels": torch.as_tensor(sal, device=self.device)}

    def loss(self, x, pair, seg, rng):
        src_vid, vid_mask = self._video(x)
        neg = (pair + torch.as_tensor(rng.integers(1, self.n_pairs, size=len(pair)), device=pair.device)) % self.n_pairs
        out = self.model(self.txt[pair], self.txt_mask[pair], src_vid, vid_mask,
                         src_txt_neg=self.txt[neg], src_txt_mask_neg=self.txt_mask[neg])
        loss_dict = self.criterion(out, self._targets(pair, seg, rng))
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict if k in self.weight_dict)
        val = lambda v: float(v.detach()) if torch.is_tensor(v) else float(v)
        return loss, {"loss": val(loss), **{k: val(loss_dict[k]) for k in ("loss_span", "loss_giou", "loss_label", "loss_saliency") if k in loss_dict}}

    @torch.no_grad()
    def predict(self, x, pair):
        self.model.eval()
        src_vid, vid_mask = self._video(x)
        out = self.model(self.txt[pair], self.txt_mask[pair], src_vid, vid_mask)
        prob = F.softmax(out["pred_logits"], -1)[..., 0]                            # foreground is label 0
        best = prob.argmax(dim=1)
        span = out["pred_spans"][torch.arange(len(best), device=best.device), best].cpu().numpy().astype(np.float64)   # (cx, w)
        u0, u1 = span[:, 0] - 0.5 * span[:, 1], span[:, 0] + 0.5 * span[:, 1]
        s = np.clip(unit_to_win(u0, self.steps), 0, 1); e = np.clip(unit_to_win(u1, self.steps), 0, 1)
        return finish(s, e, np.ones(len(s), bool))                                  # always a moment: no fallback

    def state(self):
        return {"model": self.model.state_dict()}
