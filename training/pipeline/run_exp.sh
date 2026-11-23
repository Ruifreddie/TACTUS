#!/usr/bin/env bash
# One experiment end to end. Every step leaves a marker in .done/, so re-running continues after a crash or on a new machine.
# usage: run_exp.sh A|B|C   (C: Silver only. With 4 GPUs: spatial on 0 and 3, classification on 1, interval then refiner on 2)
set -uo pipefail
E=$1; source "$(dirname "$0")/env.sh"; EXP="$WORK/exp/$E"; RAW="$HF_RAW"
mkdir -p "$EXP/.done" "$EXP/logs"
"$PY" "$X/make_exp.py" "$E" "$U" "$EXPORT" "$WORK/tok_full" "$RAW" "$EXP" || exit 1
"$PY" "$X/persist.py" load "$E" "$EXP" "$GV/state/$E"
cd "$EXP"; export PYTHONPATH="$EXP/third_party/t2b" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
G0=0; G1=$(( N_GPU > 1 ? 1 : 0 )); F4=$(( N_GPU > 3 ? 1 : 0 ))
GR=$(( F4 ? 2 : G1 )); GC=$(( F4 ? 3 : G1 )); GF=$(( F4 ? 2 : G0 ))
ep(){ [ "${SMOKE:-0}" = 1 ] && echo "--epochs ${2:-1} --batch-size ${SBATCH:-8}" || echo "--num-workers 8"; }
# Keypoint recipes use batch 32 and about 28 GB of GPU memory (measured: batch 8 9 GB, 16 15.5 GB). A 24 GB card holds batch 24; lr stays as in the recipe.
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); KB=${KB:-$(( VRAM > 30000 ? 32 : 24 ))}; echo "[$E] GPU memory $VRAM MiB, keypoint batch $KB"
kp(){ [ "${SMOKE:-0}" = 1 ] && ep x 1 || echo "--num-workers 8 --batch-size $KB"; }
CLSX="--val-interval 1 --save-interval 100000 --save-best-only --early-stop-patience 20 --early-stop-min-delta 0 --early-stop-warmup-epochs 5"
step(){ local n=$1; shift; [ -f ".done/$n" ] && { echo "[$E] skip $n"; return 0; }
  echo "[$E] $(date +%F_%T) start $n"; if "$@" > "logs/$n.log" 2>&1; then touch ".done/$n"; echo "[$E] $(date +%F_%T) done $n"; "$PY" "$X/persist.py" save "$E" "$EXP" "$GV/state/$E" >/dev/null 2>&1; else echo "[$E] FAILED $n, see $EXP/logs/$n.log"; tail -5 "logs/$n.log"; return 1; fi; }
join(){ local rc=0; for p in "$@"; do wait "$p" || rc=1; done; [ $rc = 0 ] || { echo "[$E] stopped after a failed step"; exit 1; }; }
dump(){ "$PY" tools/dump_val_predictions.py --checkpoint "experiments/runs/$1/out/checkpoints/best.pt" --out "experiments/analysis/pred_$1_best_$2" --split "$2" --device cuda --batch-size 16; }

VJ=(--vjepa-repo "$EXP/third_party/t2b/vjepa2" --vjepa-arch vjepa2_1_vit_large_384 --vjepa-hf-repo "" --vjepa-checkpoint "$VJEPA_CKPT" --input-size 384)
# The classification recipe's flags run on the interval code: the old snapshot can only build V-JEPA 2, the newer trainer also builds 2.1.
cls_sh(){ sed 's#classification/original_snapshot/code#action_range/code#' "$BUNDLE/repro/classification/train_stage$1.sh" > "$BUNDLE/repro/classification/train_stage$1.v21.sh"; echo "$BUNDLE/repro/classification/train_stage$1.v21.sh"; }
cls1(){ DATA_ROOT="$EXP/data/case32/v1" OUTPUT_DIR="$EXP/experiments/runs/cls1" CUDA_VISIBLE_DEVICES=$G1 PYTHONPATH= PYTHON="$PY" bash "$(cls_sh 1)" "${VJ[@]}" $CLSX $(ep x 1); }
cls2(){ local b; b=$("$PY" -c "import torch;print(int(torch.load('experiments/runs/cls1/checkpoints/best.pt',map_location='cpu',weights_only=False)['epoch']))"); local n=$(( b + 50 > 73 ? b + 50 : 73 )); [ "${SMOKE:-0}" = 1 ] && n=$((b+1))
  # stage 2 optimises another loss, so stage 1's best value must not carry over
  "$PY" -c "import torch;c=torch.load('experiments/runs/cls1/checkpoints/best.pt',map_location='cpu',weights_only=False);c.pop('best_monitor',None);c.pop('early_stop',None);c.pop('early_stop_state',None);c['val_metrics']={};torch.save(c,'experiments/runs/cls1/init_stage2.pt')" || return 1
  DATA_ROOT="$EXP/data/case32/v1" OUTPUT_DIR="$EXP/experiments/runs/cls2" INIT_CKPT="$EXP/experiments/runs/cls1/init_stage2.pt" CUDA_VISIBLE_DEVICES=$G1 PYTHONPATH= PYTHON="$PY" bash "$(cls_sh 2)" "${VJ[@]}" $CLSX --epochs "$n" $([ "${SMOKE:-0}" = 1 ] && echo --batch-size 8); }
rng(){ DATA_ROOT="$EXP/data/case32/v1" OUTPUT_DIR="$EXP/experiments/runs/interval" CUDA_VISIBLE_DEVICES=$GR PYTHONPATH= PYTHON="$PY" bash "$BUNDLE/repro/action_range/train_best.sh" "${VJ[@]}" --traj-loss-pixel-scale 384 --target-range-tiou 0 $(ep x 1); }
stage1(){ CUDA_VISIBLE_DEVICES=$G0 EXTRA_ARGS="$(kp)" bash "$X/recipe.sh" E007k $G0; }
s1_after(){ CUDA_VISIBLE_DEVICES=$G0 dump E007k val && CUDA_VISIBLE_DEVICES=$G0 dump E007k test && "$PY" "$X/make_bank.py" experiments/analysis/pred_E007k_best_val experiments/analysis/error_bank_E007j_val_384px.npy; }
zoom_build(){ "$PY" tools/build_zoom_dataset.py --workers "$WORKERS" --dump val=experiments/analysis/pred_E007k_best_val --dump test=experiments/analysis/pred_E007k_best_test
  if [ "$E" = B ]; then "$PY" "$X/reuse_zoom_tokens.py" "$WORK/exp/A/data/case32/v1_zoom" data/case32/v1_zoom; fi; }
zoom_tok(){ for s in train val test; do local d="data/case32/v1_zoom/vjepa_token_cache/$s"; "$PY" "$X/shards.py" unpack "$GV/tokens/zoom_$E" "$d" "$s"
    CACHE_GPUS="$([ $F4 = 1 ] && echo "$G0 $GC" || echo "")" bash "$X/cache_tokens.sh" "$EXP/data/case32/v1_zoom" "$EXP/data/case32/v1_zoom/vjepa_token_cache" $s || return 1; "$PY" "$X/shards.py" pack "$d" "$GV/tokens/zoom_$E" "$s" || return 1; done; rm -rf data/case32/v1_zoom/video_16_npy; }
labels(){ "$PY" "$X/make_ct_root.py" && "$PY" tools/build_hindsight_labels.py; }
stage2(){ CUDA_VISIBLE_DEVICES=$G0 EXTRA_ARGS="$(kp)" bash "$X/recipe.sh" E009 $G0 && for s in val test; do CUDA_VISIBLE_DEVICES=$G0 dump E009 $s || return 1; done; }
contact(){ CUDA_VISIBLE_DEVICES=$GC EXTRA_ARGS="$(kp)" bash "$X/recipe.sh" C006 $GC && for s in val test; do CUDA_VISIBLE_DEVICES=$GC dump C006 $s || return 1; done; }
refiner(){ mkdir -p experiments/runs/E008b/pixref; CUDA_VISIBLE_DEVICES=$GF "$PY" tools/pixel_refiner.py train --out experiments/runs/E008b/pixref --epochs $([ "${SMOKE:-0}" = 1 ] && echo 1 || echo 20) --device cuda --workers 8 --clips-per-batch 4 --max-crops 256 --val-dump experiments/analysis/pred_E007k_best_val; }
evaluate(){ mkdir -p results
  for s in val test; do
    CUDA_VISIBLE_DEVICES=$G0 "$PY" "$X/eval_head.py" --code "$KC_RNG" --ckpt experiments/runs/cls2/checkpoints/best.pt --root "$EXP/data/case32/v1" --split $s --out results/cls_$s.json || return 1
    CUDA_VISIBLE_DEVICES=$G0 "$PY" "$X/eval_head.py" --code "$KC_RNG" --ckpt experiments/runs/interval/checkpoints/best.pt --root "$EXP/data/case32/v1" --split $s --out results/interval_$s.json || return 1
    "$PY" tools/eval_zoom_stage2.py --zoom-dump experiments/analysis/pred_E009_best_$s --stage1-dump experiments/analysis/pred_E007k_best_$s --save results/stage2_$s.json || return 1
    "$PY" tools/eval_contact_stage.py --dump experiments/analysis/pred_C006_best_$s --save results/contact_human_$s.json || return 1
    "$PY" tools/eval_hindsight.py --dump experiments/analysis/pred_C006_best_$s --save results/contact_hindsight_$s.json || return 1
  done
  CUDA_VISIBLE_DEVICES=$G0 "$PY" tools/infer_pipeline.py --split test --dump-dir results/pipeline_test --device cuda > results/pipeline_test.log 2>&1 || return 1; tail -20 results/pipeline_test.log; }

g_a(){ step stage1 stage1 && step s1_after s1_after; }
g_b(){ step cls1 cls1 && step cls2 cls2 && step interval rng; }
g_c(){ step stage2 stage2; }
g_d(){ step contact contact; }
both(){ if [ "$N_GPU" -gt 1 ]; then "$1" & local x=$!; "$2" & local y=$!; join $x $y; else "$1" && "$2" || { echo "[$E] stopped after a failed step"; exit 1; }; fi; }
if [ $F4 = 1 ]; then
  rm -f .spa_failed
  t_spa(){ { g_a && step zoom_build zoom_build && step zoom_tok zoom_tok && step labels labels && both g_c g_d; } || { touch .spa_failed; return 1; }; }
  t_cls(){ step cls1 cls1 && step cls2 cls2; }
  # the refiner needs only the stage-1 val dump, so it follows the short interval run on GPU 2
  t_rng(){ step interval rng || return 1; while [ ! -f .done/s1_after ]; do [ -f .spa_failed ] && return 1; sleep 60; done; step refiner refiner; }
  t_spa & P1=$!; t_cls & P2=$!; t_rng & P3=$!; join $P1 $P2 $P3
else
  both g_a g_b
  step zoom_build zoom_build || exit 1
  step zoom_tok zoom_tok || exit 1
  step labels labels || exit 1
  both g_c g_d
  step refiner refiner || exit 1
fi
step evaluate evaluate || exit 1
[ -d "$GV" ] && mkdir -p "$GV/results/$E" && cp -r results logs "$GV/results/$E/"
"$PY" "$X/persist.py" save "$E" "$EXP" "$GV/state/$E"
echo "[$E] finished"
