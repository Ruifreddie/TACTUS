"""Load TactusNet's two trainer snapshots side by side in one process.

The Semantic and Temporal Heads train with repro/action_range/code/kpt_combo ($KC_RNG), the three decoders of the
Spatial Head (stage 1, zoom, contact) with UniTAP's third_party/t2b/kpt_combo. Both are script directories whose
modules share names (train, model, loss, dataset, vjepa_wrapper), and train.py imports its siblings by bare name at
module level. So each snapshot is imported once with its own directory first on sys.path, then its modules leave
sys.modules again. `Snapshot.active()` puts them back under the bare names while trainer code runs, so that any import
made during a call resolves to the same snapshot.
"""
import contextlib
import importlib
import sys
from pathlib import Path

_BARE = ("train", "model", "loss", "dataset", "vjepa_wrapper")
_LOADED = {}


class Snapshot:
    def __init__(self, tag, kc_dir, modules):
        self.tag, self.kc_dir, self.modules = tag, Path(kc_dir), dict(modules)
        self.T = self.modules["train"]

    @contextlib.contextmanager
    def active(self):
        saved = {n: sys.modules.get(n) for n in self.modules}
        sys.modules.update(self.modules)
        try:
            yield self.T
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m

    def __repr__(self):
        return f"Snapshot({self.tag}, {self.kc_dir})"


def _modules_from(kc_dir):
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if f and "." not in name and Path(f).resolve().parent == kc_dir:
            out[name] = m
    return out


def load(tag, kc_dir):
    """kc_dir: the directory that holds train.py (.../kpt_combo). Returns the cached Snapshot on a second call."""
    kc_dir = Path(kc_dir).resolve()
    if not (kc_dir / "train.py").is_file():
        raise SystemExit(f"no trainer at {kc_dir}/train.py")
    if str(kc_dir) in _LOADED:
        return _LOADED[str(kc_dir)]
    held = {n: sys.modules.pop(n) for n in _BARE if n in sys.modules}     # another snapshot's modules, if any
    sys.path.insert(0, str(kc_dir))
    try:
        importlib.invalidate_caches()
        importlib.import_module("train")
        mods = _modules_from(kc_dir)
    finally:
        sys.path.remove(str(kc_dir))
        for n in list(sys.modules):
            m = sys.modules[n]
            f = getattr(m, "__file__", None)
            if f and "." not in n and Path(f).resolve().parent == kc_dir:
                del sys.modules[n]
        sys.modules.update(held)
    missing = [n for n in ("train", "model", "loss", "dataset") if n not in mods]
    if missing:
        raise RuntimeError(f"{kc_dir}: modules {missing} were not imported from this directory")
    for n, m in mods.items():
        sys.modules[f"_ft_{tag}_{n}"] = m          # keeps them alive and visible for debugging
    snap = Snapshot(tag, kc_dir, mods)
    _LOADED[str(kc_dir)] = snap
    return snap
