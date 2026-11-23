#!/usr/bin/env bash
# Real data, real batch: both keypoint trainers side by side for a few minutes. Fails if a trainer dies (OOM included), and logs speed and GPU memory for the time estimate.
set -uo pipefail
source "$(dirname "$0")/env.sh"; "$PY" "$X/make_exp.py" pilot "$U" "$EXPORT" "$WORK/tok_full" "$HF_RAW" "$WORK/exp/pilot" || exit 1; cd "$WORK/exp/pilot"; export PYTHONPATH="$PWD/third_party/t2b"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); KB=${KB:-$(( VRAM > 30000 ? 32 : 24 ))}; rm -f gpu.log
(while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> gpu.log; sleep 5; done) & M=$!
P=()
for g in $(seq 0 $((N_GPU-1))); do OUT_DIR="$PWD/experiments/runs/E007k/p$g" LOG_FILE="$PWD/p$g.log" EXTRA_ARGS="--epochs 1 --batch-size $KB --num-workers ${PILOT_WORKERS:-8} --log-interval 20 --save-interval 100000" timeout 600 bash "$X/recipe.sh" E007k $g > "p$g.out" 2>&1 & P+=($!); done
bad=0; for p in "${P[@]}"; do wait "$p"; c=$?; [ "$c" = 0 ] || [ "$c" = 124 ] || bad=1; done      # 124: still training when the time ran out, which is what a pilot wants
kill "$M" 2>/dev/null; echo "peak GPU memory: $(sort -n gpu.log | tail -1)"; grep -h -E "step|it/s|sec" p0.log 2>/dev/null | tail -3 | cut -c1-200
if [ "$bad" = 1 ]; then grep -h -E "OutOfMemory|Error" p*.out | tail -3 | cut -c1-200; echo "pilot failed at batch $KB, see $PWD/p*.out"; exit 1; fi
echo "pilot ok, batch $KB"
