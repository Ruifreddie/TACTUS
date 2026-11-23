"""Paths shared by every baseline. Each one can be overridden by an environment variable.

The defaults follow training/pipeline/env.sh (the export and raw clips on the fast local disk $WORK, everything that
must survive on the persistent storage $GV). baselines/env.example.sh lists the variables with placeholder paths.
"""
import os
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


EXPORT = _path("BL_EXPORT", "/work/export")                    # shared per-clip arrays (pipeline/export_clips.py)
HF_RAW = _path("BL_HF_RAW", "/work/hf")                        # raw clips, HF layout: <task>/clips/<stem>.mp4
TOKENS = _path("BL_TOKENS", "/work/tok_full/all")              # TactusNet's V-JEPA 2.1 tokens: <sid>.npy [8,24,24,1024] fp16
OUT = _path("BL_OUT", "/work/baselines")                       # predictions, checkpoints and logs of the baselines
WEIGHTS = _path("BL_WEIGHTS", "/workspace/weights/baselines")  # downloaded pretrained weights (kept on the volume)

METRIC_SIZE = 384   # every spatial metric uses coordinates normalized to the content crop, times 384 (TactusNet convention)
N_FRAMES = 16       # protocol frames per clip, uniform over the affordance window, end frames included
