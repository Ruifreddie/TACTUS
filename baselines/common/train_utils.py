"""Small training helpers shared by the baseline trainers: seeding, schedules, checkpoints with resume, JSONL logs.

Every trainer follows the same contract:
- train on Gold train only, evaluate on val after every epoch, keep the checkpoint with the best val metric (best.pt),
  and keep last.pt so a killed run resumes from its last epoch (--resume is the default behaviour);
- write <out>/log.jsonl (one line per epoch) and, at the end, <out>/pred_val.npz and <out>/pred_test.npz from best.pt;
- the test split is never used for any choice.
"""
import json, math, os, random, time
from pathlib import Path

import numpy as np


def set_seed(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(step, total, warmup, min_ratio=0.0):
    """LR multiplier: linear warmup, then cosine to min_ratio."""
    if step < warmup: return (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def layer_decay_groups(named_params, base_lr, weight_decay, layer_of, n_layers, decay, no_decay=lambda n, p: p.ndim <= 1 or n.endswith(".bias")):
    """Param groups with layer-wise LR decay (BEiT/VideoMAE style). layer_of(name) -> 0..n_layers (n_layers = head)."""
    groups = {}
    for n, p in named_params:
        if not p.requires_grad: continue
        lid = layer_of(n); wd = 0.0 if no_decay(n, p) else weight_decay; key = (lid, wd)
        g = groups.setdefault(key, {"params": [], "lr": base_lr * decay ** (n_layers - lid), "weight_decay": wd})
        g["params"].append(p)
    return list(groups.values())


class Checkpointer:
    """best.pt by a val metric, last.pt for resume. Saves atomically."""
    def __init__(self, out, metric, mode="max"):
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True); self.metric = metric; self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf; self.best_epoch = -1

    def _save(self, path, state):
        import torch
        tmp = str(path) + ".tmp"; torch.save(state, tmp); os.replace(tmp, path)

    def step(self, epoch, val, state):
        v = float(val[self.metric]); better = v > self.best if self.mode == "max" else v < self.best
        state = dict(state, epoch=epoch, best=self.best if not better else v, best_epoch=epoch if better else self.best_epoch)
        if better:
            self.best, self.best_epoch = v, epoch; self._save(self.out / "best.pt", state)
        self._save(self.out / "last.pt", state)
        return better

    def resume(self, model, optimizer=None, scaler=None):
        """Load last.pt if present; returns the next epoch to run (0 for a fresh run)."""
        import torch
        p = self.out / "last.pt"
        if not p.exists(): return 0
        s = torch.load(p, map_location="cpu", weights_only=False); model.load_state_dict(s["model"])
        if optimizer is not None and "optimizer" in s: optimizer.load_state_dict(s["optimizer"])
        if scaler is not None and "scaler" in s: scaler.load_state_dict(s["scaler"])
        self.best, self.best_epoch = s.get("best", self.best), s.get("best_epoch", -1)
        return int(s["epoch"]) + 1

    def load_best(self, model):
        import torch
        s = torch.load(self.out / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(s["model"]); return s


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row):
        row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a") as f: f.write(json.dumps(row) + "\n")
