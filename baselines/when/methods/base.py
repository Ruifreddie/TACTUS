"""Interface every When method implements; the trainer (train.py) only talks to this.

A method owns its model(s), optimizer and schedule, the loss on a batch of windows, and the decoding to one interval
[start, end] on the window axis per clip, always reading the true pair's output only.

Recipe rule: each method keeps its OFFICIAL configuration (architecture, optimizer, LR schedule
shape, losses, decoding) and changes only what TACTUS requires:
  (a) the output channels (6 pairs), with the loss and the readout on the true pair's channel;
  (b) the sequence length of the windows (8 or 32 steps): pyramid levels, regression ranges and similar settings;
  (c) the schedule length, scaled to our data size (the official number of epochs; iterations follow from our data);
  (d) the batch size, with linear LR scaling (lr = official lr * batch / official batch).
Where the official batch is tiny (ActionFormer/TriDet 2, MS-TCN 1) the DEFAULT config uses the
official batch and LR unchanged; the linearly scaled batch-64 variant is only an extra sweep entry that val may choose.
OFFICIAL holds the official values, DEFAULTS the recipe after these adaptations, adaptations() lists every change with
its reason (it goes into info.json and the supplement table). CONFIGS are all the configs a run can use; SWEEP is the
subset sweep.py runs by default (the val sweep only varies free parameters of the adaptations).
"""
import copy

import numpy as np
import torch


class Method:
    name = ""                 # CLI name
    title = ""                # table name
    paper = ""
    code_url = ""
    official_source = ""      # file and commit of the official configuration
    main_steps = 8            # setting of the main table (Tab. 3)
    sched_per = "iter"        # the LR scheduler steps per "iter" or per "epoch"
    OFFICIAL = {}             # official configuration values (for info.json / README)
    DEFAULTS = {}             # recipe = official values + the required adaptations
    CONFIGS = {"main": {}}    # name -> overrides (free parameters of the adaptations only)
    SWEEP = None              # configs sweep.py runs by default (None = all CONFIGS)
    SMOKE = {}                # overrides for --smoke
    conditioning = ""         # how the true pair enters (info.json)
    decoding = ""             # how the interval is read out (info.json)

    def __init__(self, recipe, steps, n_pairs, device):
        self.recipe, self.steps, self.n_pairs, self.device = recipe, steps, n_pairs, torch.device(device)
        self.model = None

    # ------------------------------------------------------------------ recipe
    @classmethod
    def steps_defaults(cls, steps):
        """Recipe entries that depend on the number of steps (adaptation b)."""
        return {}

    @classmethod
    def make_recipe(cls, steps, config="main", overrides=None, smoke=False):
        if config not in cls.CONFIGS: raise SystemExit(f"{cls.name}: unknown config {config!r}, choose from {list(cls.CONFIGS)}")
        r = copy.deepcopy(cls.DEFAULTS); r.update(cls.steps_defaults(steps)); r.update(copy.deepcopy(cls.CONFIGS[config]))
        r.update(overrides or {})
        if smoke: r.update(cls.SMOKE)
        r.update(cls.derived(steps, r))
        return r

    @classmethod
    def derived(cls, steps, r):
        """Entries computed from the others. Linear LR scaling (d) unless lr is set explicitly."""
        d = {}
        if r.get("lr") is None: d["lr"] = float(r["base_lr"]) * int(r["batch"]) / int(r["base_batch"])
        return d

    @classmethod
    def adaptations(cls, steps, r):
        """Every difference from the official configuration: [{setting, official, tactus, reason}]."""
        return []

    # ------------------------------------------------------------------ training
    def build(self): raise NotImplementedError

    def total_epochs(self): return int(self.recipe["epochs"])

    def batch_size(self): return int(self.recipe["batch"])

    def make_optimizer(self, iters_per_epoch): raise NotImplementedError

    def loss(self, x, pair, seg, rng):
        """x float32 [B,T,D], pair long [B], seg float32 [B,2] (window axis) -> (loss tensor, {name: float})."""
        raise NotImplementedError

    def grad_clip(self): return None

    def after_step(self): pass

    def train_mode(self): self.model.train()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, x, pair):
        """-> start [N], end [N] on the window axis (clipped to [0,1]), fallback [N] bool (whole window was output)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ checkpoints
    def state(self):
        """Everything besides the optimizer and the scheduler needed to resume and to predict."""
        return {"model": self.model.state_dict()}

    def load_state(self, s):
        self.model.load_state_dict(s["model"])

    def n_params(self):
        return int(sum(p.numel() for p in self.model.parameters()))


def adapt(setting, official, tactus, reason):
    return {"setting": setting, "official": official, "tactus": tactus, "reason": reason}


def batch_adapt(r, unit="samples"):
    """The batch / LR entry: unchanged at the official batch, else linear LR scaling (d)."""
    w = "window" if int(r["batch"]) == 1 else "windows"
    off, ours = f"{r['base_batch']} {unit} / lr {r['base_lr']:.1e}", f"{r['batch']} {w} / lr {r['lr']:.1e}"
    if int(r["batch"]) == int(r["base_batch"]) and abs(float(r["lr"]) - float(r["base_lr"])) < 1e-12:
        return adapt("batch / lr", off, ours, "unchanged: the official batch and LR")
    return adapt("batch / lr", off, ours, "(d) batch changed, LR scaled linearly (lr x batch / official batch)")


def finish(start, end, found):
    """Clip to the window, fall back to the whole window where nothing was found."""
    start = np.clip(np.asarray(start, np.float64), 0.0, 1.0); end = np.clip(np.asarray(end, np.float64), 0.0, 1.0)
    end = np.maximum(end, start); found = np.asarray(found, bool)
    start = np.where(found, start, 0.0); end = np.where(found, end, 1.0)
    return start.astype(np.float32), end.astype(np.float32), ~found


def range_ladder(levels, steps):
    """The official regression ranges [0,4], [4,8], [8,16], ... (feature-grid steps) cut to the available FPN levels, the
    last one open-ended (10000, as officially), and scaled by 4 at 32 steps."""
    scale = steps // 8; b = [4 * scale * 2 ** i for i in range(levels - 1)]
    return [[int(l), int(h)] for l, h in zip([0] + b, b + [10000])]
