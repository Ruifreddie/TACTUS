#!/usr/bin/env bash
# V-JEPA 2.1 tokens with the recipe's converter, one shard per GPU. Finished tokens are skipped, so it can be re-run.
# usage: cache_tokens.sh DATA_ROOT CACHE_ROOT SPLIT [--keep-video]
set -euo pipefail
ROOT=$1; CACHE=$2; SPLIT=$3; KEEP=--keep-video
source "$(dirname "$0")/env.sh"
GPUS=(${CACHE_GPUS:-$(seq 0 $((N_GPU-1)))}); NG=${#GPUS[@]}
n=$(grep -c . "$ROOT/$SPLIT.txt"); per=$(( (n + NG - 1) / NG )); echo "[cache] $SPLIT: $n clips on $NG GPU(s) at batch $CACHE_BATCH"
cd "$U/third_party/t2b"; export PYTHONPATH="$U/third_party/t2b"
pids=()
for k in $(seq 0 $((NG-1))); do g=${GPUS[$k]}
  CUDA_VISIBLE_DEVICES=$g "$PY" "$U/third_party/vitpose_dataset_tools/convert_video16_to_vjepa_tokens_one_by_one.py" \
    --data-root "$ROOT" --cache-root "$CACHE" --splits "$SPLIT" --start-index $((k*per)) --num-samples $per \
    --input-size 384 --batch-size "$CACHE_BATCH" --token-dtype fp16 --cache-layer-mode single_final \
    --vjepa-repo "$U/third_party/t2b/vjepa2" --vjepa-arch vjepa2_1_vit_large_384 --vjepa-hf-repo "" --vjepa-checkpoint "$VJEPA_CKPT" \
    $KEEP --log-interval 200 > "$CACHE.$SPLIT.gpu$g.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
have=$(find "$CACHE/$SPLIT" -name '*.npy' | wc -l); echo "[cache] $SPLIT: $have tokens"; [ "$have" -ge "$n" ]
