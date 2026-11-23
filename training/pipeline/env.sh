# Paths shared by every step. Override any of them before sourcing.
export GV="${GV:-/workspace}"                 # persistent storage: raw data tars, weights, code, token shards, results
export WORK="${WORK:-/work}"                  # fast local disk: everything being built or trained on
export BUNDLE="${BUNDLE:-$WORK/code}"         # unpacked code bundle
export U="$BUNDLE/unitap"                     # UniTAP repo (tools/, third_party/)
export KC_RNG="$BUNDLE/repro/action_range/code"
export X="$BUNDLE/pipeline"
export HF_RAW="$WORK/hf"                      # HF layout: <task>/{clips,annotation}, index_*.json
export EXPORT="$WORK/export"                  # shared per-clip arrays for all clips of all experiments
export VJEPA_CKPT="${VJEPA_CKPT:-$GV/weights/vjepa2_1_vitl_dist_vitG_384.pt}"
export PY="${PY:-python}"
export N_GPU="${N_GPU:-2}"
export CACHE_BATCH="${CACHE_BATCH:-4}"
export WORKERS="${WORKERS:-16}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK"
