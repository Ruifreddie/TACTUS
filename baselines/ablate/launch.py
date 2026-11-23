#!/usr/bin/env python3
"""Run a TactusNet trainer or UniTAP tool script with the ablation switches, without editing any UniTAP or repro file.

pyshim.sh stands in for the recipes' python ($PY in UniTAP run.sh files via pipeline/recipe.sh, $PYTHON in the
classification and interval recipes), so every recipe runs unchanged:

  launch.py <dir>/kpt_combo/train.py <trainer args>        a trainer: imported as module `train`, then train.main()
  launch.py <exp>/tools/dump_val_predictions.py <args>      any other script: run as __main__ (set ABLATE_KC, see below)

Switches (environment, so that the recipe command lines stay byte-identical):
  ABLATE_STUB_ENCODER=<arch>  every token comes from the cache, so the frozen encoder is never run. The trainer builds a
                              randomly initialized <arch> without loading any weight; <arch> only sets the token width of
                              the heads (vjepa2_1_vit_large_384: 1024, vjepa2_1_vit_base_384: 768). Implies strict cache.
  ABLATE_STRICT_CACHE=1       a token-cache miss raises instead of encoding frames (with a stub, the trainer would otherwise
                              encode them with the random encoder and write the result into the cache).
  ABLATE_NULL_COND=1          ablation "no condition": every forward of KeypointComboNet gets tool, action and pair id 0
                              (a constant), in training and in evaluation. The heads keep their FiLM layers, which then
                              learn a constant modulation, so the architecture and parameter count do not change.
  ABLATE_KC=<dir>             kpt_combo dir whose `model` module is patched for ABLATE_NULL_COND before a tool script runs
                              (trainers patch their own dir).
Each switch is reported on stderr and written to $ABLATE_RECORD (json) when that variable is set.
"""
import importlib.util, json, os, runpy, sys


def _env(name):
    v = os.environ.get(name, "").strip()
    return "" if v in ("", "0") else v


def patch_model_null_cond(model_mod):
    """KeypointComboNet.forward with tool_id, action_id and combo_id replaced by zeros (same shape, dtype and device)."""
    import torch
    cls = model_mod.KeypointComboNet
    if getattr(cls.forward, "_ablate_null", False): return
    orig = cls.forward
    names = ("video", "tokens_grid", "token_order", "tool_id", "action_id", "combo_id")

    def forward(self, *args, **kw):
        args = list(args)
        for i, n in enumerate(names):
            if n in ("tool_id", "action_id", "combo_id"):
                if i < len(args) and torch.is_tensor(args[i]): args[i] = torch.zeros_like(args[i])
                elif torch.is_tensor(kw.get(n)): kw[n] = torch.zeros_like(kw[n])
        return orig(self, *args, **kw)
    forward._ablate_null = True
    cls.forward = forward


def patch_trainer(T, stub, strict, null):
    if stub:
        orig_parse = T.parse_args

        def parse_args(*a, **k):
            args = orig_parse(*a, **k)
            args.vjepa_arch = stub; args.vjepa_checkpoint = ""; args.vjepa_hf_repo = ""
            args.vjepa_pretrained_hub = False; args.allow_random_vjepa_init = True
            return args
        T.parse_args = parse_args
    if strict or stub:
        C = T.VJEPATokenCache; orig_get = C.get_or_compute

        def get_or_compute(self, *, model, video, sids, split, target_device):
            if self.enabled:
                miss = [str(s) for s in sids if not self._path(split=split, sid=str(s)).is_file()]
                if miss: raise RuntimeError(f"[ablate] token cache miss (split={split}, e.g. {miss[:3]}) under {self.root}: "
                                            "extract the tokens first; the ablation never encodes frames inside a trainer")
            return orig_get(self, model=model, video=video, sids=sids, split=split, target_device=target_device)
        C.get_or_compute = get_or_compute
    if null:
        patch_model_null_cond(sys.modules["model"])


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]      # nothing from baselines/ablate shadows
    script = os.path.abspath(sys.argv[1]); sys.argv = [script] + sys.argv[2:]
    stub, strict, null = _env("ABLATE_STUB_ENCODER"), bool(_env("ABLATE_STRICT_CACHE")), bool(_env("ABLATE_NULL_COND"))
    rec = dict(script=script, stub_encoder=stub or None, strict_cache=bool(strict or stub), null_condition=null, argv=sys.argv[1:])
    print(f"[ablate] {os.path.basename(script)}: stub_encoder={stub or '-'} strict_cache={bool(strict or stub)} null_condition={null}",
          file=sys.stderr, flush=True)
    if _env("ABLATE_RECORD"):
        with open(os.environ["ABLATE_RECORD"], "w") as f: json.dump(rec, f, indent=1)
    sdir = os.path.dirname(script)
    if os.path.basename(script) == "train.py":
        sys.path.insert(0, sdir)                       # what `python train.py` does; the trainer imports dataset/model/loss from here
        spec = importlib.util.spec_from_file_location("train", script)
        T = importlib.util.module_from_spec(spec); sys.modules["train"] = T; spec.loader.exec_module(T)
        patch_trainer(T, stub, strict, null)
        T.main()
    else:
        sys.path.insert(0, sdir)
        if null:
            kc = os.environ.get("ABLATE_KC") or os.path.join(os.path.dirname(sdir), "third_party/t2b/kpt_combo")
            sys.path.insert(0, kc); sys.path.insert(1, os.path.dirname(kc))
            import model as _m                         # the module train.py will import: patched before the script runs
            patch_model_null_cond(_m)
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
