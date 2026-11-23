#!/usr/bin/env bash
# Step functions of the keypoint baselines (How and Where), sourced by run_baselines.sh.
# Needs BL_PY, BL_OUT, BL_EXPORT, BL_HF_RAW, BL_WEIGHTS and PYTHONPATH = repo root (see baselines/env.example.sh).
#   prep_kpt       CPU: backbone weights into $BL_WEIGHTS/kpt, the 1-fps training frames of Gold train, CPU checks
#   train_hrnet    GPU: HRNet-W32 + DARK   -> $BL_OUT/kpt_hrnet   (pred_val.npz, pred_test.npz, metrics_test.json)
#   train_vitpose  GPU: ViTPose-B          -> $BL_OUT/kpt_vitpose
#   train_dsta     GPU: DSTA ResNet-50     -> $BL_OUT/kpt_dsta
# Extra arguments go to the trainer (e.g. train_hrnet --epochs 120 --workers 12). Re-running a step resumes from last.pt.

prep_kpt() {
  "$BL_PY" -m baselines.kpt.weights || return 1
  "$BL_PY" - <<'EOF' || return 1
from baselines.common.data import get_export
from baselines.kpt import core, engine
ex = get_export(); f = core.frames_1fps(ex, ex.sids("train"), engine.FRAMES_1FPS_CACHE)
print(f"frames_1fps: {len(f)} Gold-train clips, {sum(map(len, f.values()))} training frames -> {engine.FRAMES_1FPS_CACHE}")
EOF
  CUDA_VISIBLE_DEVICES="" "$BL_PY" -m baselines.kpt.tests.test_kpt --weights
}

_kpt_score() { "$BL_PY" -m baselines.common.score --pred "$BL_OUT/$1/pred_test.npz" --out "$BL_OUT/$1/metrics_test.json"; }

train_hrnet()   { "$BL_PY" -m baselines.kpt.hrnet   --out "$BL_OUT/kpt_hrnet"   "$@" && _kpt_score kpt_hrnet; }
train_vitpose() { "$BL_PY" -m baselines.kpt.vitpose --out "$BL_OUT/kpt_vitpose" "$@" && _kpt_score kpt_vitpose; }
train_dsta()    { "$BL_PY" -m baselines.kpt.dsta    --out "$BL_OUT/kpt_dsta"    "$@" && _kpt_score kpt_dsta; }
