#!/usr/bin/env bash
# Step functions of the classification baselines (Which, What, Pair), sourced by run_baselines.sh or by hand:
#
#   source baselines/cls/run.sh
#   prep_cls                 # CPU: weights into $BL_WEIGHTS/cls; frames packed into $BL_OUT/cls_cache
#                            #      (train clips at 256 and 320 = official pre-crop sizes, val/test clips at 224)
#   train_tsm                # GPU: train on Gold train, select on Gold val, predict Gold test, score
#   train_tsn; train_videomae; train_xclip; train_endofm; train_mvitv2 (reserve)
#   bench_cls videomae       # GPU: 2-minute throughput / memory check before a long run
#   smoke_cls                # every model with --smoke on the current export (train, resume, predict, score)
#
# Needs BL_PY, BL_OUT, BL_WEIGHTS (plus BL_EXPORT, BL_HF_RAW for the data) and the repo on PYTHONPATH (baselines/env.example.sh
# lists all of them). Extra arguments go to baselines.cls.train, e.g. `train_videomae --batch 2`. Every step is idempotent:
# a rerun resumes from last.pt, or only re-predicts and re-scores when training is complete.

: "${BL_PY:?set BL_PY}" "${BL_OUT:?set BL_OUT}" "${BL_WEIGHTS:?set BL_WEIGHTS}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
CLS_MODELS="tsn tsm videomae xclip endofm mvitv2"
# Test videos that Endo-FM's self-supervised pretraining saw (CholecTriplet); its row is also scored without them.
CLS_ENDOFM_EXCLUDE="${CLS_ENDOFM_EXCLUDE:-cholec80_13,cholec80_22}"

prep_cls() {        # CPU only; about 8 MB per train clip (256 + 320) and 2.4 MB per val/test clip (Gold: ~108 GB)
    "$BL_PY" -m baselines.cls.weights || return 1
    [ "${CLS_NO_PACK:-0}" = 1 ] && { echo "CLS_NO_PACK=1: no pre-resized cache, frames are resized on the fly"; return 0; }
    "$BL_PY" -m baselines.cls.pack --workers "${CLS_PACK_WORKERS:-$(nproc)}"      # default plan, see pack.py
}

_score_cls() {      # $1 = run dir
    "$BL_PY" -m baselines.common.score --pred "$1/pred_test.npz" --out "$1/metrics_test.json" > "$1/score_test.log" &&
    "$BL_PY" -m baselines.common.score --pred "$1/pred_val.npz" --split val --no-boot --out "$1/metrics_val.json" > /dev/null
}

_train_cls() {      # $1 = model, rest = train args
    local m=$1; shift; local d="$BL_OUT/cls/$m"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null
    "$BL_PY" -m baselines.cls.train --model "$m" --out "$d" "$@" || return 1
    _score_cls "$d" || return 1
    if [ "$m" = endofm ]; then
        "$BL_PY" -m baselines.common.score --pred "$d/pred_test.npz" --exclude-tasks "$CLS_ENDOFM_EXCLUDE" \
            --out "$d/metrics_test_excl_pretrain_videos.json" > /dev/null || return 1
    fi
    echo "[cls] $m done: $d/metrics_test.json"
}

train_tsn()      { _train_cls tsn "$@"; }
train_tsm()      { _train_cls tsm "$@"; }
train_videomae() { _train_cls videomae "$@"; }
train_xclip()    { _train_cls xclip "$@"; }
train_endofm()   { _train_cls endofm "$@"; }
train_mvitv2()   { _train_cls mvitv2 "$@"; }   # reserve row
train_cls_all()  { local m; for m in tsn tsm videomae xclip endofm; do "train_$m" "$@" || return 1; done; }

bench_cls() {       # $1 = model, rest = train args (e.g. --batch 8)
    local m=${1:?model}; shift; "$BL_PY" -m baselines.cls.train --model "$m" --out "$BL_OUT/cls_bench/$m" --bench 20 "$@"
}

smoke_cls() {       # local check of every code path (numbers are meaningless)
    local m d
    for m in ${1:-$CLS_MODELS}; do
        d="$BL_OUT/cls_smoke/$m"
        "$BL_PY" -m baselines.cls.train --model "$m" --out "$d" --smoke --stop-after 1 || return 1   # epoch 0, then "preempted"
        "$BL_PY" -m baselines.cls.train --model "$m" --out "$d" --smoke || return 1                  # resumes at epoch 1
        "$BL_PY" -m baselines.common.score --pred "$d/pred_test.npz" --no-boot > "$d/score_test.log" || return 1
        echo "[smoke] $m ok"
    done
}
