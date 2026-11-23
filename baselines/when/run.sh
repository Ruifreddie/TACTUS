#!/usr/bin/env bash
# Step functions of the When baselines (Tab. 3 block b). Source this file, then call the steps, e.g.
#   source baselines/env.example.sh   # or the variables set by run_baselines.sh
#   source baselines/when/run.sh
#   prep_when            # CPU: 8-step features from TactusNet's token cache + CLIP text features of the pair templates
#   extract32            # GPU: token-equivalence check, then the 32-step features (needed by MS-TCN)
#   train_actionformer   # GPU: val sweep x 3 seeds, selection on val, test once  (likewise train_tridet, train_qddetr, train_mstcn)
#   score_when           # CPU: Tab. 3 metrics with bootstrap CIs for every finished row
# The default sweep of ActionFormer, TriDet and MS-TCN is `main` (the official batch 2 / 2 / 1 and LR) plus `b64` (batch 64,
# LR scaled linearly); QD-DETR runs `main` only. Extra configs: train_actionformer --configs main,b64,l2.
# Every step is idempotent: existing features are skipped, finished runs are not retrained, interrupted runs resume.
# Variables: BL_PY (python), BL_OUT (results root), BL_WHEN_FEATS (feature root; keep it on the volume on the pod),
#            WHEN_SPLITS (splits for pool/extract32, default the Gold splits train,val,test),
#            WHEN_JOBS (trainings run concurrently on the one GPU, default 3: the heads are launch-bound).

WHEN_PY=${BL_PY:-python}
WHEN_OUT=${WHEN_OUT:-${BL_OUT:?set BL_OUT}/when}
export BL_WHEN_FEATS=${BL_WHEN_FEATS:-$WHEN_OUT/feats}
WHEN_SPLITS=${WHEN_SPLITS:-train,val,test}
WHEN_JOBS=${WHEN_JOBS:-3}

prep_when() {    # CPU
  $WHEN_PY -m baselines.when.features pool --splits "$WHEN_SPLITS" "$@" &&
  $WHEN_PY -m baselines.when.features text
}

extract32() {    # GPU, ~0.6-0.8 s per clip on one RTX 4090 (fp32, as TactusNet's token cache)
  $WHEN_PY -m baselines.when.features check --n 3 &&
  $WHEN_PY -m baselines.when.features extract32 --splits "$WHEN_SPLITS" --batch 4 --workers 6 "$@"
}

_when_sweep() {  # method default_steps [steps] [extra sweep.py args, e.g. --configs main,b64,l2 --seeds 0]
  local m=$1 s=$2; shift 2
  if [[ ${1:-} =~ ^[0-9]+$ ]]; then s=$1; shift; fi
  $WHEN_PY -m baselines.when.sweep --method "$m" --steps "$s" --out "$WHEN_OUT/${m}_s${s}" --seeds 0,1,2 --jobs "$WHEN_JOBS" "$@"
}
train_actionformer() { _when_sweep actionformer 8 "$@"; }   # main table: 8 steps (train_actionformer 32 for the 32-step path)
train_tridet()       { _when_sweep tridet 8 "$@"; }         # main table: 8 steps
train_qddetr()       { _when_sweep qddetr 8 "$@"; }         # main table: 8 steps
train_mstcn()        { _when_sweep mstcn 32 "$@"; }         # main table: 32 steps (run extract32 first)

score_when() {   # CPU
  local d
  for d in "$WHEN_OUT"/actionformer_s8 "$WHEN_OUT"/tridet_s8 "$WHEN_OUT"/qddetr_s8 "$WHEN_OUT"/mstcn_s32; do
    [ -f "$d/pred_test.npz" ] || { echo "skip $d (no pred_test.npz)"; continue; }
    $WHEN_PY -m baselines.common.score --pred "$d/pred_test.npz" --out "$d/metrics_test.json" > /dev/null && echo "wrote $d/metrics_test.json"
  done
}

smoke_when() {   # a few steps of every method on the current export (writes under $WHEN_OUT/smoke)
  local m
  for m in actionformer:8 tridet:8 qddetr:8 mstcn:32; do
    $WHEN_PY -m baselines.when.train --method "${m%%:*}" --steps "${m##*:}" --smoke --fresh --out "$WHEN_OUT/smoke/${m%%:*}_s${m##*:}" "$@" || return 1
  done
}
