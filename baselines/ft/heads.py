"""TactusNet's heads in the fine-tuning runs, each trained and validated by its own trainer's code.

A head is one KeypointComboNet of experiment A, rebuilt from its best checkpoint exactly as baselines/ablate/heads.py
rebuilds it for evaluation (stub encoder of the right width, `load_trainable_state_dict`, the checkpoint's EMA). Its
training step is its trainer's own `train_one_epoch`, run on a single micro-batch, with two stand-ins:
  * the token cache returns the tokens of the shared encoder for this micro-batch (a leaf tensor, so the step's
    backward leaves d loss / d tokens in its .grad for ft.train to pass on to the encoder);
  * the optimizer does nothing: ft.train steps every optimizer once all heads and the encoder have their gradients.
So the forward, the targets, the loss and its weights (class weights, stage curriculum, token masking, centroid noise)
are the recipe's own code. ft.train clips each head's gradient with its recipe's grad_clip after accumulation, which is
why the step itself runs with grad_clip = inf. Validation is the trainer's own `evaluate` on cached tokens with the EMA
weights applied, and its checkpoint criterion (`_best_monitor_name`: loss_total, or range_tiou for the Temporal Head).

Deviations from the recipes, the same in every run:
  * Semantic Head: one loss for both branches, tool CE (weight 1.0, recipe cls1) + action CE (1.0, recipe cls2), both
    branches trainable; the recipes train the tool branch first and freeze it while the action branch trains.
  * every head keeps an EMA of its weights (decay 0.999); the interval recipe has none. Validation and test use it.
  * batch 32 clips for every head, LR scaled linearly from the recipe's batch (cls, interval 64 -> LR / 2; point
    decoders 32 -> unchanged), and one schedule for all weights (encoder.videomae_schedule).
"""
import argparse
import copy
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SPECS = {
    "sem": dict(group="semantic", snap="kc_rng", ckpt="experiments/runs/cls2/checkpoints/best.pt", root="data/case32/v1",
                tokens="full", recipe_batch=64, recipe="repro/classification/train_stage{1,2}.sh"),
    "tmp": dict(group="temporal", snap="kc_rng", ckpt="experiments/runs/interval/checkpoints/best.pt", root="data/case32/v1",
                tokens="full", recipe_batch=64, recipe="repro/action_range/train_best.sh"),
    "s1": dict(group="spatial", snap="unitap", ckpt="experiments/runs/E007k/out/checkpoints/best.pt", root="data/case32/v1",
               tokens="full", recipe_batch=32, recipe="experiments/runs/E007k/run.sh"),
    "zm": dict(group="spatial", snap="unitap", ckpt="experiments/runs/E009/out/checkpoints/best.pt", root="data/case32/v1_zoom",
               tokens="zoom", recipe_batch=32, recipe="experiments/runs/E009/run.sh"),
    "ct": dict(group="spatial", snap="unitap", ckpt="experiments/runs/C006/out/checkpoints/best.pt", root="data/case32/v1_zoom_hs",
               tokens="zoom", recipe_batch=32, recipe="experiments/runs/C006/run.sh"),
}
GROUPS = {"semantic": ("sem",), "temporal": ("tmp",), "spatial": ("s1", "zm", "ct")}
ORDER = ("sem", "tmp", "s1", "zm", "ct")
SEMANTIC_FT_WEIGHTS = dict(combo_tool_ce_weight=1.0, combo_action_ce_weight=1.0, combo_ce_weight=0.0)


class _Inject:
    """Stands in for the trainer's VJEPATokenCache inside one training step: the tokens of this micro-batch."""
    enabled = True

    def __init__(self, tokens):
        self.tokens = tokens

    def get_or_compute(self, *, model, video, sids, split, target_device):
        if len(sids) != int(self.tokens.shape[0]):
            raise RuntimeError(f"{len(sids)} clips but tokens for {int(self.tokens.shape[0])}")
        return self.tokens


class _NoStep:
    """The step backpropagates; ft.train steps the real optimizers after every head of the micro-batches."""

    def __init__(self):
        self.param_groups = [{"lr": 0.0}]

    def zero_grad(self, set_to_none=True):
        return None

    def step(self, closure=None):
        return None


def _num(v):
    try:
        return float(v)
    except Exception:
        return v


def _cpu(tree):
    if torch.is_tensor(tree):
        return tree.detach().cpu()
    if isinstance(tree, dict):
        return {k: _cpu(v) for k, v in tree.items()}
    return tree


class Head:
    def __init__(self, name, exp, snap, device, vjepa_ckpt, batch=32, out_dir=None):
        if name not in SPECS:
            raise SystemExit(f"unknown head {name}; choose from {list(SPECS)}")
        sp = SPECS[name]
        self.name, self.group, self.tokens = name, sp["group"], sp["tokens"]
        self.exp, self.snap, self.device = Path(exp), snap, torch.device(device)
        self.root = self.exp / sp["root"]
        self.ckpt_path = self.exp / sp["ckpt"]
        T = snap.T
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        a = argparse.Namespace(**dict(ck["args"]))
        # paths of this experiment (the checkpoint records the pod that trained it)
        a.data_root = str(self.root)
        a.train_split_file = str(self.root / "train.txt")
        a.val_split_file = str(self.root / "val.txt")
        a.device = str(self.device)
        a.vjepa_repo = str(self.exp / "third_party/t2b/vjepa2")
        a.vjepa_checkpoint = str(vjepa_ckpt)
        for k, v in (("max_train_steps", 0), ("range_duration_prior_means", None), ("range_duration_prior_stds", None)):
            if not hasattr(a, k):
                setattr(a, k, v)
        if str(getattr(a, "range_combo_reweight_mode", "none")).lower() != "none":
            raise SystemExit(f"{name}: range_combo_reweight_mode={a.range_combo_reweight_mode} is not supported here")
        if str(getattr(a, "range_duration_prior_mode", "none")).lower() != "none":
            raise SystemExit(f"{name}: range_duration_prior_mode={a.range_duration_prior_mode} is not supported here")
        if name == "sem":
            if not (bool(getattr(a, "combo_probe_dual_attn", False)) and float(a.combo_action_ce_weight) == 1.0
                    and float(a.combo_ce_weight) == 0.0 and not bool(getattr(a, "tool_only", False))):
                raise SystemExit(f"{self.ckpt_path} is not the cls2 (action stage) checkpoint of the Semantic Head")
            for k, v in SEMANTIC_FT_WEIGHTS.items():
                setattr(a, k, v)
        self.args = a                                           # recipe args (+ the Semantic loss), as saved
        self.args_step = copy.copy(a)                           # what the trainer code sees in a step
        self.args_step.grad_clip = math.inf                     # clipped in ft.train after accumulation
        self.args_step.use_progress = False
        self.args_step.log_interval = 0
        self.args_step.debug_train_check = False
        self.args_step.max_train_steps = 0
        self.grad_clip = float(a.grad_clip)
        self.recipe_lr = float(a.lr)
        self.recipe_batch = int(sp["recipe_batch"])
        self.peak_lr = self.recipe_lr * float(batch) / self.recipe_batch
        self.weight_decay = float(a.weight_decay)
        self.start_epoch = int(ck.get("epoch", 0))
        self.ckpt_val = {k: _num(v) for k, v in dict(ck.get("val_metrics", {}) or {}).items() if not isinstance(v, (dict, list))}
        self.out_dir = Path(out_dir or ".")
        # model: stub encoder of the right width (never run: tokens come from the shared encoder), then the weights
        with snap.active():
            b = copy.copy(a)
            b.vjepa_arch = "vjepa2_1_vit_large_384"
            b.vjepa_checkpoint, b.vjepa_hf_repo, b.vjepa_pretrained_hub, b.allow_random_vjepa_init = "", "", False, True
            m = T.build_model(b, num_combo_classes=6)
            m.vjepa.encoder = torch.nn.Module()                 # drops 300M random weights; the head reads injected tokens
            T.load_trainable_state_dict(m, ck["modules"])
            m.to(self.device)
            self.ema = T.TrainableEMA(model=m, decay=float(getattr(a, "ema_decay", 0.999)))
            self.recipe_ema = "ema" in ck
            if self.recipe_ema:
                self.ema.load_state_dict(ck["ema"], model=m)
            # datasets of the head's own root and splits, targets only (videos come from the shared loader)
            kw = dict(input_size=int(a.input_size), normalize=True, load_video=False,
                      combo_only_pure=bool(getattr(a, "combo_only_pure", False)),
                      tool_kpt_only=bool(getattr(a, "tool_kpt_only_data", False)),
                      range_only_contract=bool(getattr(a, "range_only", False)))
            self.ds_train = T.SurgKptComboDataset(root=str(self.root), split="train", split_file=a.train_split_file, **kw)
            self.ds_val = T.SurgKptComboDataset(root=str(self.root), split="val", split_file=a.val_split_file, **kw)
            # the class weights main() hands to train_one_epoch (use_weighted_ce is on in every recipe)
            include_rb = (not bool(getattr(a, "combo_only", False))) and (not bool(getattr(a, "trajectory_only", False))) \
                and (not bool(getattr(a, "tool_kpt_only_data", False)))
            rw = T._build_train_reweighting(data_root=self.root, train_sids=[str(s) for s in self.ds_train.sids],
                                            num_combo_classes=6, num_range_bins=int(T.RANGE_NUM_BINS),
                                            power=float(a.reweight_power), max_factor=float(a.reweight_max_factor),
                                            include_range_bins=include_rb)
        tw = lambda key: torch.tensor(rw[key], dtype=torch.float32, device=self.device) if a.use_weighted_ce else None  # noqa: E731
        self.cw, self.sw, self.ew = tw("combo_weights"), tw("start_bin_weights"), tw("end_bin_weights")
        self.sample_weight_by_sid = dict(rw["sample_weight_by_sid"])
        self.late_focal = dict(ck.get("traj_late_focal_state", {}) or {})
        self.model = m
        self.params = [p for p in m.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=self.peak_lr, weight_decay=self.weight_decay)
        self._stub_opt = _NoStep()
        try:
            self._scaler = torch.amp.GradScaler("cuda", enabled=False)
        except Exception:
            self._scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.train_index = {str(s): i for i, s in enumerate(self.ds_train.sids)}
        self.val_sids = [str(s) for s in self.ds_val.sids]
        self.n_params = sum(p.numel() for p in self.params)

    # ------------------------------------------------------------------------------------------------ training
    def train_step(self, sub_batch, tokens, epoch, lr):
        """One recipe step on one micro-batch (backward included, no optimizer step). epoch: 1-based fine-tuning epoch;
        the recipe's epoch-based schedules (token masking, stage curriculum) continue from the checkpoint's epoch."""
        self._stub_opt.param_groups[0]["lr"] = float(lr)
        with self.snap.active() as T:
            metrics, _ = T.train_one_epoch(
                model=self.model, loader=[sub_batch], optimizer=self._stub_opt, scaler=self._scaler, device=self.device,
                epoch=self.start_epoch + int(epoch), args=self.args_step, out_dir=self.out_dir, writer=None,
                global_step_start=0, token_cache=_Inject(tokens), combo_class_weights=self.cw,
                start_bin_class_weights=self.sw, end_bin_class_weights=self.ew, range_combo_reweight_weights=None,
                ema=None, traj_late_focal_state=self.late_focal)
        return metrics

    def clip_and_step(self, lr, accum):
        """Gradient of the mean over the micro-batches, the recipe's clipping, one AdamW step, the EMA update."""
        if accum > 1:
            for p in self.params:
                if p.grad is not None:
                    p.grad.div_(accum)
        gn = torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.grad_clip)
        for g in self.opt.param_groups:
            g["lr"] = float(lr)
        self.opt.step()
        with self.snap.active():
            self.ema.update(self.model)
        return float(gn)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------------------------------------- validation
    def validate(self, cache_root, epoch, batch=16, workers=4):
        """The trainer's evaluate() on the val split with the EMA weights, tokens from cache_root/val/<sid>.npy."""
        loader = DataLoader(self.ds_val, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
        with self.snap.active() as T:
            tc = T.VJEPATokenCache(root=Path(cache_root), enabled=True, dtype_name="fp16")
            with T._use_ema_weights(self.model, self.ema):
                r = T.evaluate(self.model, loader, self.device, self.args_step, self.ds_val.combo_id_to_name,
                               epoch=self.start_epoch + int(epoch), token_cache=tc)
            metrics = r[0] if isinstance(r, tuple) else r
            name, mode = str(T._best_monitor_name(self.args)), str(T._best_monitor_mode(self.args))
            value = float(T._extract_best_monitor_value(self.args, metrics, float("nan")))
        flat = {k: _num(v) for k, v in dict(metrics).items() if not isinstance(v, (dict, list))}
        return {"criterion": name, "mode": mode, "value": value, "metrics": flat}

    # ------------------------------------------------------------------------------------------------- outputs
    def checkpoint(self, epoch, val=None, extra=None):
        """A checkpoint in the trainers' own format, so every evaluation tool loads it unchanged (args, modules, ema).
        The weights are the live ones and the EMA shadow, exactly as the recipes save them."""
        with self.snap.active() as T:
            mods = T.trainable_state_dict(self.model)
        return {"epoch": self.start_epoch + int(epoch), "global_step": 0, "modules": _cpu(mods), "ema": _cpu(self.ema.state_dict()),
                "args": dict(vars(self.args)), "val_metrics": dict((val or {}).get("metrics", {})),
                "ft": dict(head=self.name, ft_epoch=int(epoch), source=str(self.ckpt_path), **(extra or {}))}

    def state(self):
        with self.snap.active() as T:
            mods = T.trainable_state_dict(self.model)
        return {"modules": _cpu(mods), "ema": _cpu(self.ema.state_dict()), "opt": self.opt.state_dict(),
                "late_focal": self.late_focal}

    def load_state(self, st):
        with self.snap.active() as T:
            T.load_trainable_state_dict(self.model, st["modules"])
            self.ema.load_state_dict(st["ema"], model=self.model)
        self.opt.load_state_dict(st["opt"])
        self.late_focal.clear()
        self.late_focal.update(st.get("late_focal", {}) or {})

    def describe(self):
        return dict(head=self.name, group=self.group, tokens=self.tokens, checkpoint=str(self.ckpt_path),
                    start_epoch=self.start_epoch, recipe_lr=self.recipe_lr, recipe_batch=self.recipe_batch,
                    peak_lr=self.peak_lr, weight_decay=self.weight_decay, grad_clip=self.grad_clip,
                    recipe_ema=self.recipe_ema, ema_decay=self.ema.decay, trainable_params=self.n_params,
                    train_clips=len(self.ds_train), val_clips=len(self.ds_val),
                    loss_weights=({k: getattr(self.args, k) for k in SEMANTIC_FT_WEIGHTS} if self.name == "sem" else None))
