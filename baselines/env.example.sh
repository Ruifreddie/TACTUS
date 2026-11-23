# Environment of the TACTUS baselines. Copy this file, edit the paths, then `source` it from the repo root:
#   source baselines/env.example.sh
# Every path below is a placeholder under $BL_ROOT. Nothing is downloaded or created by this file except empty
# directories. baselines/run_baselines.sh does not use this file: it sets the same variables itself from $GV and $WORK.

# ---- root
export BL_ROOT="${BL_ROOT:-/path/to/tactus_baselines}"   # one directory of your choice that holds data, weights, caches and outputs

# ---- data (read only)
export BL_EXPORT="$BL_ROOT/export"       # per-clip arrays: labels.json, meta.json, all_{train,val,test}.txt, gt_*/, ... (training/pipeline/export_clips.py)
export BL_HF_RAW="$BL_ROOT/hf"           # raw clips of the dataset, Hugging Face layout: <task>/clips/<stem>.mp4
export BL_TOKENS="$BL_ROOT/tokens/all"   # TactusNet's frozen V-JEPA 2.1 tokens, <sid>.npy [8,24,24,1024] fp16 (When baselines, ex.tokens)
export VJEPA_CKPT="$BL_ROOT/weights/vjepa2_1_vitl_dist_vitG_384.pt"   # V-JEPA 2.1 ViT-L/384 checkpoint (when/features.py extract32 and check, where/tactus_human.sh)
export BL_SMOKE_EXP="$BL_ROOT/exp/A"     # optional: a TactusNet experiment directory (tests --exp, smoke_where, CHECK_EXP)

# ---- written by the baselines
export BL_WEIGHTS="$BL_ROOT/weights/baselines"   # pretrained weights of the baselines, downloaded here when missing
export BL_OUT="$BL_ROOT/out"                     # predictions, checkpoints, logs and metrics, one directory per run
# Optional overrides (defaults in parentheses):
# export BL_WHEN_FEATS="$BL_OUT/when/feats"      # pooled When features ($BL_OUT/when/feats)
# export BL_CLS_CACHE="$BL_OUT/cls_cache"        # pre-resized frames of the classification baselines ($BL_OUT/cls_cache)
# export BL_FRAME_CACHE="$BL_ROOT/frame_cache"   # JPEG cache of decoded protocol frames, used when the export has no video_16_npy (off)

# ---- code
_BL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"   # the repo root (the parent of baselines/)
export BL_PY="${BL_PY:-python}"                              # the Python interpreter that runs every baseline
export PYTHONPATH="$_BL_REPO${PYTHONPATH:+:$PYTHONPATH}"     # makes `python -m baselines....` importable
# Needed only by the steps that call the TactusNet training code (when/features.py extract32, where/tactus_human.sh):
# export U="$_BL_REPO/training/unitap"           # Spatial Head code; holds third_party/t2b/vjepa2
# export X="$_BL_REPO/training/pipeline"         # orchestration scripts
# export VJEPA_REPO="$U/third_party/t2b/vjepa2"  # vjepa2 source checkout, if it is somewhere else

# ---- caches (keep them next to the data, not in $HOME)
export HF_HOME="$BL_ROOT/cache/hf"           # Hugging Face hub cache
export TORCH_HOME="$BL_ROOT/cache/torch"     # torch hub and torchvision weights cache
export PIP_CACHE_DIR="$BL_ROOT/cache/pip"    # pip download cache
export TMPDIR="$BL_ROOT/tmp"                 # temporary files

export PYTHONUNBUFFERED=1
mkdir -p "$BL_OUT" "$BL_WEIGHTS" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR" 2>/dev/null || true
