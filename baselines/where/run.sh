#!/usr/bin/env bash
# Step functions of the Where family. Source this file (run_baselines.sh does), then call a function:
#   CPU: prep_where, floor_hindsight, score_where, restore_exp_A (pod, needs a GPU only if zoom tokens are missing)
#   GPU: train_segformer, train_tactus_human
#   smoke run of everything: smoke_where (after `source baselines/env.example.sh`, with BL_SMOKE_EXP set)
# Variables: BL_PY, BL_OUT, BL_WEIGHTS, BL_EXPORT, BL_HF_RAW (baselines/README.md); on the pod also GV (volume,
# default /workspace) and WORK (pod disk, default /work), laid out as in training/pipeline/env.sh.
WHERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; BL_CODE="$(cd "$WHERE_DIR/../.." && pwd)"
_wbl(){ PYTHONPATH="$BL_CODE${PYTHONPATH:+:$PYTHONPATH}" "${BL_PY:-python}" "$@"; }

prep_where(){          # CPU: pinned extras, SegFormer weights into $BL_WEIGHTS, contact-slot order check on the export
  "${BL_PY:-python}" -m pip install -q -r "$WHERE_DIR/requirements.txt" || return 1
  _wbl -c "from baselines.where.segformer import ensure_weights; print('weights', ensure_weights())" || return 1
  _wbl -m baselines.where.check_order --out "$BL_OUT/where/check_order.json"
}

floor_hindsight(){     # CPU: hindsight target from the true tips -> $BL_OUT/hindsight_floor{,_corners}/pred_{val,test}.npz
  # CHECK_EXP=<UniTAP exp dir with data/case32/v1_zoom_hs> adds the agreement check with UniTAP's own labels
  _wbl -m baselines.where.hindsight_floor --out "$BL_OUT" ${CHECK_EXP:+--check-exp "$CHECK_EXP"} "$@"
}

train_segformer(){     # GPU: SegFormer-B2 -> $BL_OUT/segformer/pred_{val,test}.npz (resumes from last.pt); SMOKE=1 for a smoke run
  local extra=(); [ "${SMOKE:-0}" = 1 ] && extra=(--smoke)
  _wbl -m baselines.where.segformer --out "$BL_OUT/segformer" "${extra[@]}" "$@"
}

restore_exp_A(){       # pod: rebuild experiment A's UniTAP dir $WORK/exp/A from the volume, without training or writing back
  # Same commands as pipeline/run_exp.sh A on a new pod: make_exp.py, persist.py load (state/A/state.tar: step
  # markers, dumps incl. the stage-1 dumps, results, every best.pt), zoom boxes rebuilt from the stage-1 dumps
  # (build_zoom_dataset.py, deterministic; --no-video because the tokens come back from $GV/tokens/zoom_A), and the
  # labels step (make_ct_root.py -> v1_zoom_ct, build_hindsight_labels.py -> v1_zoom_hs). Needs: $WORK/export and
  # $WORK/hf (raw annotations) on the pod disk, the code bundle $GV/code.tar, $GV/state/A, $GV/tokens/zoom_A.
  ( set -euo pipefail
    export GV=${GV:-/workspace} WORK=${WORK:-/work}
    [ -d "$WORK/code/unitap" ] || { mkdir -p "$WORK/code"; tar -xf "$GV/code.tar" -C "$WORK/code"; }
    source "$WORK/code/pipeline/env.sh"; export N_GPU=1; EXP="$WORK/exp/A"; mkdir -p "$EXP/.done" "$EXP/logs"
    "$PY" "$X/make_exp.py" A "$U" "$EXPORT" "$WORK/tok_full" "$HF_RAW" "$EXP"
    "$PY" "$X/persist.py" load A "$EXP" "$GV/state/A"
    cd "$EXP"; export PYTHONPATH="$EXP/third_party/t2b"
    [ -f data/case32/v1_zoom/boxes.json ] || "$PY" tools/build_zoom_dataset.py --workers "$WORKERS" --no-video \
        --dump val=experiments/analysis/pred_E007k_best_val --dump test=experiments/analysis/pred_E007k_best_test
    for s in train val test; do "$PY" "$X/shards.py" unpack "$GV/tokens/zoom_A" "data/case32/v1_zoom/vjepa_token_cache/$s" "$s"; done
    [ -f data/case32/v1_zoom_ct/meta.json ] || "$PY" "$X/make_ct_root.py"
    [ -d data/case32/v1_zoom_hs/region_mask ] || "$PY" tools/build_hindsight_labels.py
    # the rebuilt boxes must be A's: the restored C006 test dump scored with them reproduces A's saved Where Dice
    "$PY" tools/eval_contact_stage.py --dump experiments/analysis/pred_C006_best_test --save results/.restore_check_test.json > /dev/null
    "$PY" -c "import json,sys; r=lambda p: json.load(open(p))['summary']['region_hull_dice']['mean']; a,b=r('results/contact_human_test.json'),r('results/.restore_check_test.json'); print('restore check, C006 test Dice: saved',a,'rebuilt boxes',b); sys.exit(abs(a-b)>1e-9)"
    echo "experiment A restored in $EXP" )
}

train_tactus_human(){  # GPU: retrain only TactusNet's contact decoder on human regions -> $BL_OUT/tactus_human/pred_{val,test}.npz
  # $1 = experiment dir (default $WORK/exp/A after restore_exp_A); SMOKE=1 for 1 epoch at batch 2
  local exp=${1:-${WORK:-/work}/exp/A}
  U=${U:-${WORK:-/work}/code/unitap} X=${X:-${WORK:-/work}/code/pipeline} PY=${PY:-${BL_PY:-python}} BL_PY=${BL_PY:-python} \
    bash "$WHERE_DIR/tactus_human.sh" "$exp" "$BL_OUT/tactus_human"
}

score_where(){         # CPU: the paper's Where metrics of every Where prediction file (test, then val)
  local m s
  for m in segformer tactus_human hindsight_floor hindsight_floor_corners; do for s in test val; do
    [ -f "$BL_OUT/$m/pred_$s.npz" ] || continue
    echo "== $m $s"; _wbl -m baselines.common.score --pred "$BL_OUT/$m/pred_$s.npz" --split "$s" --out "$BL_OUT/$m/metrics_$s.json" | head -8
  done; done
}

smoke_where(){         # everything on the smoke data (source baselines/env.example.sh first)
  ( set -euo pipefail
    : "${BL_SMOKE_EXP:?source baselines/env.example.sh}"
    prep_where
    CHECK_EXP="$BL_SMOKE_EXP" floor_hindsight
    SMOKE=1 train_segformer --wait-gpu-mb 3000
    # a working copy of the read-only smoke experiment, then the pipeline's own labels step inside it
    local_exp="$BL_OUT/where/expA_copy"; _wbl -m baselines.where.tactus_human copy-exp "$BL_SMOKE_EXP" "$local_exp"
    ( cd "$local_exp"
      [ -f data/case32/v1_zoom_ct/meta.json ] || "$BL_PY" "$BL_CODE/training/pipeline/make_ct_root.py"
      [ -d data/case32/v1_zoom_hs/region_mask ] || "$BL_PY" tools/build_hindsight_labels.py )
    U="$BL_ROOT/smoke/code/unitap" X="$BL_CODE/training/pipeline" PY="$BL_PY" SMOKE=1 train_tactus_human "$local_exp"
    score_where )
}
