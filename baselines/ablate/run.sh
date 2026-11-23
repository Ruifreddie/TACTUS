#!/usr/bin/env bash
# TactusNet ablations (baselines/ablate/README.md). Step functions in the style of the other families' run.sh files:
# CPU steps prep_*, GPU steps ablate_*, and a job queue that runs the GPU steps in parallel, one lane per GPU.
#
#   source baselines/ablate/run.sh   # defines the steps, runs nothing
#   prep_ablate                      # CPU pod or GPU pod: restore A and B, V-JEPA 2.1 tokens, weights, ablation dirs, CPU rows
#   ablate_queue                     # GPU pod (4x RTX 4090): every GPU job of ablate_jobs, lanes = $ABL_GPUS (default: all GPUs)
#   ablate_collect                   # CPU: score every row with baselines.common.score, fill table_ablation.tex
# or: bash baselines/ablate/run.sh <step> [args]       e.g. bash baselines/ablate/run.sh ablate_queue
#
# Variables: the pod's pipeline/env.sh names (GV, WORK, BUNDLE, U, X, KC_RNG, EXPORT, HF_RAW, VJEPA_CKPT, PY) and the
# baselines' names (BL_EXPORT, BL_HF_RAW, BL_TOKENS, BL_OUT, BL_WEIGHTS, BL_PY). Ablation dirs: AB (pod disk: tokens,
# experiment dirs), AO (= $BL_OUT/ablate: prediction files, metrics, logs, filled table; copied to $GV/baselines/ablate).
# SMOKE=1: 1 epoch, batch $SBATCH (default 2), for a smoke test of the code path.
# Every step leaves a marker and is skipped when re-run; a job queue survives a restart (its .done markers persist).

BL_CODE=${BL_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GV=${GV:-/workspace}; WORK=${WORK:-/work}; BUNDLE=${BUNDLE:-$WORK/code}
U=${U:-$BUNDLE/unitap}; X=${X:-$BUNDLE/pipeline}; KC_RNG=${KC_RNG:-$BUNDLE/repro/action_range/code}
EXPORT=${EXPORT:-$WORK/export}; HF_RAW=${HF_RAW:-$WORK/hf}
BL_EXPORT=${BL_EXPORT:-$EXPORT}; BL_HF_RAW=${BL_HF_RAW:-$HF_RAW}; BL_TOKENS=${BL_TOKENS:-$WORK/tok_full/all}
BL_OUT=${BL_OUT:-$WORK/baselines}; BL_WEIGHTS=${BL_WEIGHTS:-$GV/weights/baselines}
PY=${PY:-python}; BL_PY=${BL_PY:-$PY}
VJEPA_CKPT=${VJEPA_CKPT:-$WORK/weights/vjepa2_1_vitl_dist_vitG_384.pt}
EXP_A=${EXP_A:-$WORK/exp/A}; EXP_B=${EXP_B:-$WORK/exp/B}     # restored experiments of the paper (A = TactusNet, B = + Silver)
TOK_FULL=${TOK_FULL:-$(dirname "$BL_TOKENS")}                # V-JEPA 2.1 tokens of every clip: $TOK_FULL/all/<sid>.npy
AB=${AB:-$WORK/ablate}; AO=${AO:-$BL_OUT/ablate}
HF_HOME=${HF_HOME:-$BL_WEIGHTS/hf}
SHIM="$BL_CODE/baselines/ablate/pyshim.sh"
ABL_BACKBONES=${ABL_BACKBONES:-"vjepa2_l256 videomae_l224 endofm_b224"}   # Table 4(a) rows besides ours (video encoders), longest first
ABL_S1_SCHEDULE=${ABL_S1_SCHEDULE:-full}   # full: E007k's own 80-epoch schedule (Table 4a; the ours row is experiment A); short: 20 epochs
ABL_OPTIONAL=${ABL_OPTIONAL:-0}                                            # 1: also the supplement-only GPU jobs (see ablate_jobs)
export GV WORK BUNDLE U X KC_RNG EXPORT HF_RAW BL_EXPORT BL_HF_RAW BL_TOKENS BL_OUT BL_WEIGHTS PY BL_PY VJEPA_CKPT HF_HOME AB AO ABL_BACKBONES ABL_OPTIONAL ABL_S1_SCHEDULE
export PYTHONPATH="$BL_CODE${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABLATE_REAL_PY="$PY"
mkdir -p "$AB/.done" "$AO/logs" 2>/dev/null

_log(){ echo "[ablate $(date +%F_%T)] $*"; }
_mark(){ [ -f "$AB/.done/$1" ]; }
_once(){ local n=$1; shift; if _mark "$n"; then _log "skip $n"; return 0; fi; _log "start $n"; "$@" && touch "$AB/.done/$n" && _log "done $n"; }
_kb(){ local v; v=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); echo "${KB:-$(( ${v:-24000} > 30000 ? 32 : 24 ))}"; }
_stub(){ case "$1" in lemonfm_384|videomae_b224|endofm_b224) echo vjepa2_1_vit_base_384;; *) echo vjepa2_1_vit_large_384;; esac; }
_exp(){ echo "$AB/exp/$1"; }                                  # ablation experiment dir: ours (V-JEPA 2.1 tokens) or a backbone
_tok(){ [ "$1" = ours ] && echo "$TOK_FULL" || echo "$AB/tok/$1"; }
_persist(){ [ -d "$GV" ] && [ -w "$GV" ] && mkdir -p "$GV/baselines/ablate" && cp -r "$AO/." "$GV/baselines/ablate/" 2>/dev/null; return 0; }

# ---------------------------------------------------------------------------------------------------------------- CPU
# Experiments A and B of the paper, exactly as pipeline/run_exp.sh lays them out and restores them from the volume.
ablate_restore(){ local E=$1 D; D=$([ "$1" = A ] && echo "$EXP_A" || echo "$EXP_B")
  [ -f "$D/.done/evaluate" ] && { _log "$E already restored in $D"; return 0; }
  "$PY" "$X/make_exp.py" "$E" "$U" "$EXPORT" "$TOK_FULL" "$HF_RAW" "$D" && "$PY" "$X/persist.py" load "$E" "$D" "$GV/state/$E" && [ -f "$D/.done/evaluate" ]; }
ablate_tok_full(){ "$PY" "$X/shards.py" unpack "$GV/tokens/full" "$TOK_FULL/all" full; }
ablate_weights(){   # the V-JEPA 2.1 weight at the path A's checkpoints recorded ($WORK/weights, as run_all.sh), the other encoders into $HF_HOME
  local L="$WORK/weights/$(basename "$VJEPA_CKPT")" S="$GV/weights/$(basename "$VJEPA_CKPT")"
  if [ ! -f "$L" ] && [ -f "$S" ]; then mkdir -p "$WORK/weights" && cp "$S" "$L.part" && mv "$L.part" "$L" || return 1; fi
  "$BL_PY" -m baselines.ablate.tokens prefetch --backbones "$ABL_BACKBONES $([ "$ABL_OPTIONAL" = 1 ] && echo vjepa21_l256)"; }
# An ablation experiment dir = a complete UniTAP repo laid out by pipeline/make_exp.py (train = Gold train, as A),
# whose token cache links point to the encoder's tokens: <tok>/all/<sid>.npy.
ablate_expdir(){ local n=$1; "$PY" "$X/make_exp.py" "abl_$n" "$U" "$EXPORT" "$(_tok "$n")" "$HF_RAW" "$(_exp "$n")"; }
# The classification recipes, made exactly like run_exp.sh's cls_sh() but under $AB/recipes (the bundle stays untouched).
ablate_cls_recipes(){ mkdir -p "$AB/recipes/classification"; ln -sfn "$BUNDLE/repro/action_range" "$AB/recipes/action_range"
  for s in 1 2; do sed 's#classification/original_snapshot/code#action_range/code#' "$BUNDLE/repro/classification/train_stage$s.sh" > "$AB/recipes/classification/train_stage$s.v21.sh"; done; }
# Spatial Head components (A2) and the Silver rows (A5, A6) need no GPU: re-score the pipeline dumps with score.py.
ablate_components(){ for st in stage1 stage2 refined; do
    "$BL_PY" -m baselines.common.tactus --exp "$EXP_A" --stage "$st" --no-contact --out "$AO/components/$st/pred_test.npz" || return 1; done; }
ablate_silver(){ for E in A B; do local D; D=$([ $E = A ] && echo "$EXP_A" || echo "$EXP_B")
    "$BL_PY" -m baselines.common.tactus --exp "$D" --out "$AO/silver/$E/pred_test.npz" || return 1; done; }
prep_ablate(){
  [ -n "${SKIP_RESTORE:-}" ] || { _once restore_A ablate_restore A && _once restore_B ablate_restore B && _once tok_full ablate_tok_full; } || return 1
  _once weights ablate_weights || return 1
  for n in ours $ABL_BACKBONES $([ "$ABL_OPTIONAL" = 1 ] && echo vjepa21_l256); do _once "expdir_$n" ablate_expdir "$n" || return 1; done
  _once cls_recipes ablate_cls_recipes && { [ -d "$EXP_A" ] && _once components ablate_components || [ -n "${SKIP_RESTORE:-}" ]; } && { [ -d "$EXP_B" ] && _once silver ablate_silver || true; } && _persist
}

# ---------------------------------------------------------------------------------------------------------------- GPU
_vj(){ VJ=(--vjepa-repo "$1/third_party/t2b/vjepa2" --vjepa-arch vjepa2_1_vit_large_384 --vjepa-hf-repo "" --vjepa-checkpoint "$VJEPA_CKPT" --input-size 384); }
CLSX="--val-interval 1 --save-interval 100000 --save-best-only --early-stop-patience 20 --early-stop-min-delta 0 --early-stop-warmup-epochs 5"
_ep(){ [ "${SMOKE:-0}" = 1 ] && echo "--epochs ${2:-1} --batch-size ${SBATCH:-2}" || echo "--num-workers 8"; }
_kp(){ [ "${SMOKE:-0}" = 1 ] && _ep x 1 || echo "--num-workers 8 --batch-size $(_kb)"; }
# Stage-1 schedule. Default (ABL_S1_SCHEDULE=full): E007k's own 80-epoch recipe, no overrides, so the ours row of Table 4(a) is
# experiment A's stage 1. ABL_S1_SCHEDULE=short: 20 of its 80 epochs, every epoch-based schedule of the recipe scaled by 1/4
# (token masking, stage curriculum); constant LR and EMA as in the recipe.
SHORT="--epochs 20 --token-mask-delay-epochs 1 --token-mask-warmup-epochs 5 --token-mask-late-start-epoch 5 --token-mask-late-warmup-epochs 3 \
--traj-stage-curriculum-phase1-end-epoch 4 --traj-stage-curriculum-ramp12-end-epoch 5 --traj-stage-curriculum-phase2-end-epoch 11 --traj-stage-curriculum-ramp23-end-epoch 13"

# tokens of an alternative encoder for Gold train/val/test, then the completeness check every trainer relies on
ablate_tokens(){ local bb=$1 g=${2:-0} shard=${3:-0/1}
  CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.tokens extract --backbone "$bb" --out "$AB/tok/$bb" --splits train,val,test --shard "$shard" \
    --batch "${TOK_BATCH:-2}" --workers "${TOK_WORKERS:-6}" $([ "${SMOKE:-0}" = 1 ] && echo "--limit ${SMOKE_CLIPS:-0}") || return 1
  [ "$shard" != 0/1 ] || "$BL_PY" -m baselines.ablate.tokens check --backbone "$bb" --out "$AB/tok/$bb" --splits train,val,test; }

# Semantic Head: cls1 then cls2 (run_exp.sh's recipes and flags), then per-clip predictions on test (and val)
ablate_cls(){ local n=$1 g=${2:-0} E; E=$(_exp "$n"); _vj "$E"; local R="$E/experiments/runs" b m
  export ABLATE_STUB_ENCODER=$(_stub "$n"); unset ABLATE_NULL_COND
  if [ ! -f "$R/cls1/.done" ]; then mkdir -p "$R/cls1"
    ABLATE_RECORD="$R/cls1/ablate.json" DATA_ROOT="$E/data/case32/v1" OUTPUT_DIR="$R/cls1" CUDA_VISIBLE_DEVICES=$g PYTHONPATH= PYTHON="$SHIM" \
      bash "$AB/recipes/classification/train_stage1.v21.sh" "${VJ[@]}" $CLSX $(_ep x 1) > "$R/cls1.log" 2>&1 || { tail -5 "$R/cls1.log"; return 1; }
    touch "$R/cls1/.done"; fi
  if [ ! -f "$R/cls2/.done" ]; then mkdir -p "$R/cls2"
    b=$("$PY" -c "import torch;print(int(torch.load('$R/cls1/checkpoints/best.pt',map_location='cpu',weights_only=False)['epoch']))") || return 1
    m=$(( b + 50 > 73 ? b + 50 : 73 )); [ "${SMOKE:-0}" = 1 ] && m=$((b + 1))
    "$PY" -c "import torch;c=torch.load('$R/cls1/checkpoints/best.pt',map_location='cpu',weights_only=False);c.pop('best_monitor',None);c.pop('early_stop',None);c.pop('early_stop_state',None);c['val_metrics']={};torch.save(c,'$R/cls1/init_stage2.pt')" || return 1
    ABLATE_RECORD="$R/cls2/ablate.json" DATA_ROOT="$E/data/case32/v1" OUTPUT_DIR="$R/cls2" INIT_CKPT="$R/cls1/init_stage2.pt" CUDA_VISIBLE_DEVICES=$g PYTHONPATH= PYTHON="$SHIM" \
      bash "$AB/recipes/classification/train_stage2.v21.sh" "${VJ[@]}" $CLSX --epochs "$m" $([ "${SMOKE:-0}" = 1 ] && echo "--batch-size ${SBATCH:-2}") > "$R/cls2.log" 2>&1 || { tail -5 "$R/cls2.log"; return 1; }
    touch "$R/cls2/.done"; fi
  for s in test val; do CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.heads --kind cls --code "$KC_RNG" --ckpt "$R/cls2/checkpoints/best.pt" \
      --root "$E/data/case32/v1" --split $s --out "$AO/enc/$n/cls/pred_$s.npz" || return 1; done; }

# Temporal Head: the interval recipe (+ variant flags), then per-clip predictions on test (and val).
# variant: base | nocond (ABLATE_NULL_COND) | scalar (direct regression, no bins, no quality) | noquality (bins, no quality)
ablate_interval(){ local n=$1 g=${2:-0} v=${3:-base} E; E=$(_exp "$n"); _vj "$E"; local run="interval_$v" R extra=() cond=gt
  R="$E/experiments/runs"; export ABLATE_STUB_ENCODER=$(_stub "$n"); unset ABLATE_NULL_COND
  case $v in base) ;; nocond) export ABLATE_NULL_COND=1; cond=null;;
    scalar) extra=(--range-head-mode scalar --range-quality-mode none --range-quality-loss-weight 0.0);;
    noquality) extra=(--range-quality-mode none --range-quality-loss-weight 0.0);; *) echo "unknown variant $v"; return 1;; esac
  if [ ! -f "$R/$run/.done" ]; then mkdir -p "$R/$run"
    ABLATE_RECORD="$R/$run/ablate.json" DATA_ROOT="$E/data/case32/v1" OUTPUT_DIR="$R/$run" CUDA_VISIBLE_DEVICES=$g PYTHONPATH= PYTHON="$SHIM" \
      bash "$BUNDLE/repro/action_range/train_best.sh" "${VJ[@]}" --traj-loss-pixel-scale 384 --target-range-tiou 0 $(_ep x 1) "${extra[@]}" > "$R/$run.log" 2>&1 \
      || { tail -5 "$R/$run.log"; return 1; }
    touch "$R/$run/.done"; fi
  unset ABLATE_NULL_COND
  local grp; grp=$([ "$v" = base ] && echo "enc/$n/when" || echo "temporal/$n/$v")
  for s in test val; do CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.heads --kind when --code "$KC_RNG" --ckpt "$R/$run/checkpoints/best.pt" \
      --root "$E/data/case32/v1" --split $s --cond $cond --out "$AO/$grp/pred_$s.npz" || return 1; done; }

_s1args(){ [ "$ABL_S1_SCHEDULE" = short ] && echo "$SHORT"; true; }
# stage-1 tool decoder (E007k) on the encoder's full-frame tokens, schedule $ABL_S1_SCHEDULE; variant base | nocond
ablate_stage1(){ local n=$1 g=${2:-0} v=${3:-base} E; E=$(_exp "$n"); local out="${ABL_S1_SCHEDULE}_$v" R
  R="$E/experiments/runs/E007k"; mkdir -p "$R/$out"; export ABLATE_STUB_ENCODER=$(_stub "$n"); unset ABLATE_NULL_COND
  [ "$v" = nocond ] && export ABLATE_NULL_COND=1
  if [ ! -f "$R/$out/.done" ]; then
    # pipeline/recipe.sh's edits of the recipe, written to a per-variant file so parallel variants never share a script
    sed -e "s#^WS=.*#WS=\"$E\"#" -e "s#^PY=.*#PY=\"$SHIM\"#" -e '/sudo/d' -e '/^( while true/d' -e '/^MON=/d' -e 's#nice -n 5 taskset -c [0-9-]* ##' \
        -e 's#--vjepa-checkpoint "$WS/weights/vjepa2_1_vitl_dist_vitG_384.pt"#--vjepa-checkpoint "$VJEPA_CKPT"#' "$U/experiments/runs/E007k/run.sh" > "$R/run.$out.sh"
    ( cd "$E" && ABLATE_RECORD="$R/$out/ablate.json" OUT_DIR="$R/$out" LOG_FILE="$R/$out/train.log" EXTRA_ARGS="$(_s1args) $(_kp)" \
        CUDA_VISIBLE_DEVICES=$g bash "$R/run.$out.sh" > "$R/$out/launcher.log" 2>&1 ) || { tail -5 "$R/$out/train.log"; return 1; }
    [ -f "$R/$out/checkpoints/best.pt" ] && touch "$R/$out/.done" || return 1; fi
  local D="$E/experiments/analysis/pred_E007k_${out}_best_test"
  ( cd "$E" && PYTHONPATH="$E/third_party/t2b" CUDA_VISIBLE_DEVICES=$g "$SHIM" tools/dump_val_predictions.py --checkpoint "$R/$out/checkpoints/best.pt" \
      --out "$D" --split test --device cuda --batch-size 16 ) > "$R/$out/dump_test.log" 2>&1 || { tail -5 "$R/$out/dump_test.log"; return 1; }
  unset ABLATE_NULL_COND
  local grp; grp=$([ "$v" = base ] && echo "enc/$n/stage1" || echo "stage1/$n/$v")
  "$BL_PY" -m baselines.ablate.convert dump --dump "$D" --out "$AO/$grp/pred_test.npz" --info "encoder=$n" "schedule=$ABL_S1_SCHEDULE" "variant=$v"; }

# ours: Semantic and Temporal Heads of experiment A itself (no retraining): per-clip files with the true conditions, and
# the Temporal Head with the pair predicted by the Semantic Head (A3, cascade part 1)
ablate_ours_heads(){ local g=${1:-0} R="$EXP_A/experiments/runs" C="$EXP_A/data/case32/v1"
  for s in test val; do
    CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.heads --kind cls --code "$KC_RNG" --ckpt "$R/cls2/checkpoints/best.pt" --root "$C" --split $s \
      --out "$AO/enc/ours/cls/pred_$s.npz" --check "$EXP_A/results/cls_$s.json" || return 1
    CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.heads --kind when --code "$KC_RNG" --ckpt "$R/interval/checkpoints/best.pt" --root "$C" --split $s \
      --out "$AO/enc/ours/when/pred_$s.npz" --check "$EXP_A/results/interval_$s.json" || return 1; done
  CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.heads --kind when --code "$KC_RNG" --ckpt "$R/interval/checkpoints/best.pt" --root "$C" --split test \
    --cond "pred:$AO/enc/ours/cls/pred_test.npz" --out "$AO/cond/pred/when/pred_test.npz"; }
# cascade part 2: the whole Spatial Head with the predicted pair (A3)
ablate_cascade(){ local g=${1:-0}; [ -f "$AO/enc/ours/cls/pred_test.npz" ] || ablate_ours_heads "$g" || return 1
  CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.cascade --exp "$EXP_A" --cls-pred "$AO/enc/ours/cls/pred_test.npz" --out "$AO/cond/cascade" \
    $([ "${SMOKE:-0}" = 1 ] && echo "--limit 2 --verify 1"); }
# stage 1 + pixel refiner without the zoom stage (A2, 4th row)
ablate_refine_s1(){ local g=${1:-0}
  # the online pipeline's own stage-1 points (pred_xy_stage1), the source of the other rows of block c
  CUDA_VISIBLE_DEVICES=$g "$BL_PY" -m baselines.ablate.convert refine --exp "$EXP_A" --dump "$EXP_A/results/pipeline_test" --key pred_xy_stage1 \
    --out "$AO/components/stage1_refiner/pred_test.npz"; }

# ---------------------------------------------------------------------------------------------------------------- queue
# Jobs with their inputs. Listed longest first, so that greedy lanes finish together.
# Table 4 needs: tokens + Semantic, Temporal and stage-1 heads per backbone (a), and the predicted-pair cascade (d). The ours row
# of (a) and blocks (b), (c) are experiments A and B. ABL_OPTIONAL=1 adds the supplement-only jobs (last line of ablate_jobs).
ablate_jobs(){ local bb
  for bb in $ABL_BACKBONES; do echo "tok_$bb"; done; echo "ours_heads cascade"
  for bb in $ABL_BACKBONES; do echo "s1_$bb cls_$bb"; done
  for bb in $ABL_BACKBONES; do echo "int_$bb"; done
  [ "$ABL_OPTIONAL" = 1 ] && echo "s1_ours refine_s1 intv_ours_nocond intv_ours_noquality intv_ours_scalar"; true; }
_deps(){ case $1 in cascade) echo ours_heads;; s1_ours|ours_heads|refine_s1|intv_*|s1v_*|tok_*) ;; *_*) echo "tok_${1#*_}";; esac; }
_job(){ local j=$1 g=$2
  case $j in tok_*) ablate_tokens "${j#tok_}" "$g";; s1_*) ablate_stage1 "${j#s1_}" "$g";; cls_*) ablate_cls "${j#cls_}" "$g";;
    int_*) ablate_interval "${j#int_}" "$g";; intv_ours_*) ablate_interval ours "$g" "${j#intv_ours_}";; s1v_ours_*) ablate_stage1 ours "$g" "${j#s1v_ours_}";;
    ours_heads) ablate_ours_heads "$g";; cascade) ablate_cascade "$g";; refine_s1) ablate_refine_s1 "$g";; *) echo "unknown job $j"; return 1;; esac; }
ablate_queue(){ local Q="$AO/queue" gpus; mkdir -p "$Q"; gpus=${ABL_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
  _log "lanes on GPUs: $gpus; jobs: $(ablate_jobs | tr '\n' ' ')"
  for g in $gpus; do _lane "$g" "$Q" & done; wait; _persist
  _log "queue finished: $(ls "$Q" | grep -c '\.done$') done, $(ls "$Q" | grep -c '\.failed$') failed"; ls "$Q" | grep '\.failed$' || true; }
_lane(){ local g=$1 Q=$2 j next pending d bad
  while :; do next=""; pending=0
    for j in $(ablate_jobs); do
      [ -f "$Q/$j.done" ] || [ -f "$Q/$j.failed" ] && continue; pending=1; [ -d "$Q/$j.lock" ] && continue
      bad=0; for d in $(_deps "$j"); do [ -f "$Q/$d.failed" ] && bad=1; [ -f "$Q/$d.done" ] || bad=${bad/0/2}; done
      [ "$bad" = 1 ] && { mkdir "$Q/$j.lock" 2>/dev/null && echo "input failed" > "$Q/$j.failed"; continue; }; [ "$bad" = 2 ] && continue
      mkdir "$Q/$j.lock" 2>/dev/null || continue; next=$j; break; done
    [ -z "$next" ] && { [ "$pending" = 0 ] && return 0; sleep "${ABL_POLL:-60}"; continue; }
    _log "GPU $g: start $next"; local t0; t0=$(date +%s)
    if ( _job "$next" "$g" ) > "$AO/logs/$next.log" 2>&1; then echo $(( $(date +%s) - t0 )) > "$Q/$next.done"; _log "GPU $g: done $next ($(( ($(date +%s) - t0) / 60 )) min)"
    else echo $(( $(date +%s) - t0 )) > "$Q/$next.failed"; _log "GPU $g: FAILED $next, see $AO/logs/$next.log"; fi
    _persist; done; }

ablate_status(){ local Q="$AO/queue" j; for j in $(ablate_jobs); do
    if [ -f "$Q/$j.done" ]; then echo "done    $j  $(( $(cat "$Q/$j.done") / 60 )) min"; elif [ -f "$Q/$j.failed" ]; then echo "FAILED  $j"
    elif [ -d "$Q/$j.lock" ]; then echo "running $j"; else echo "pending $j"; fi; done; }
ablate_collect(){ "$BL_PY" -m baselines.ablate.collect --ao "$AO" --exp-a "$EXP_A" --exp-b "$EXP_B" --where "${WHERE_OUT:-$BL_OUT/tactus_human}" \
    --template "$BL_CODE/baselines/ablate/table_ablation.tex" --out "$AO/table_ablation.tex" && _persist; }

[ "${BASH_SOURCE[0]}" = "$0" ] && [ $# -gt 0 ] && { "$@"; exit $?; }
true
