#!/usr/bin/env bash
# E007k = E007i (interleaved x2 + 8 virtual queries) + locality prior (E007j). Combination of the two single-factor gains.
# CE to a Gaussian target at the GT cell). No FiLM conditioning, no conf head (those are E005b / E005c).
set -euo pipefail
WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # experiment root (pipeline/recipe.sh sets it)
PY="${PY:-python}"
T2B="$WS/third_party/t2b"; DATA="$WS/data/case32/v1"; RUN="$WS/experiments/runs/E007k"; OUT="${OUT_DIR:-$RUN/out}"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONPATH="$T2B" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$T2B"
"$PY" "$T2B/kpt_combo/train.py" \
  --data-root "$DATA" --train-split-file "$DATA/train.txt" --val-split-file "$DATA/val.txt" \
  --output-dir "$OUT" \
  --vjepa-repo "$T2B/vjepa2" --vjepa-arch vjepa2_1_vit_large_384 --vjepa-hf-repo "" \
  --vjepa-checkpoint "${VJEPA_CKPT:-$WS/weights/vjepa2_1_vitl_dist_vitG_384.pt}" \
  --input-size 384 --vjepa-cache-dir "$DATA/vjepa_token_cache" --vjepa-cache-dtype fp16 \
  --trajectory-only --trajectory-target tool --traj-num-subframes 2 --traj-condition-mode tool_action \
  --traj-refiner-mode s3p9d_singlelayer_statecarry_anchoronly_detached \
  --tool-kpt-only-data --traj-tool-centroid-offset-head --traj-tool-offset-max-norm 0.2 \
  --token-mask-enable --token-mask-strategy random --token-mask-value learned_pos \
  --token-mask-p-max 0.10 --token-mask-p-early 0.10 --token-mask-delay-epochs 5 --token-mask-warmup-epochs 20 \
  --token-mask-late-start-epoch 20 --token-mask-p-late 0.15 --token-mask-late-warmup-epochs 10 \
  --token-mask-keep-k 64 --token-mask-mix-801010 \
  --traj-stage0-centroid-noise-prob 0.50 --traj-stage0-centroid-noise-sigma 0.015 \
  --traj-stage-curriculum-enable --traj-stage-curriculum-phase1-end-epoch 16 --traj-stage-curriculum-ramp12-end-epoch 20 \
  --traj-stage-curriculum-phase2-end-epoch 44 --traj-stage-curriculum-ramp23-end-epoch 50 \
  --traj-stage-curriculum-phase1-weights 0.60,0.50,0.45,0.40,0.50 \
  --traj-stage-curriculum-phase2-weights 0.35,0.45,0.60,0.80,1.00 \
  --traj-stage-curriculum-phase3-weights 0.15,0.25,0.50,0.90,1.00 \
  --temporal-upsample deconv --heatmap-softmax-temp 1.0 --temporal-mean-residual-alpha 0.5 --refiner-mode lite \
  --combo-ce-weight 0.1 --combo-tool-ce-weight 0.5 --combo-action-ce-weight 0.5 \
  --traj-heatmap-ce-weight 0.0 --traj-heatmap-sigma 2.45 --traj-head-hidden-dim 256 --traj-head-up-layers 2 \
  --traj-recurrent-max-delta-norm 0.06 \
  --kpt-loss-type cotracker_huber --traj-loss-pixel-scale 384 --traj-huber-delta 6 --traj-t0-loss-weight 1.0 --traj-loss-visible-only \
  --epochs 80 --batch-size 32 --num-workers 2 --lr 1e-4 --weight-decay 0.05 --grad-clip 1.0 \
  --traj-perpoint-local-readout --traj-perpoint-local-radius 2 --traj-perpoint-gate-init 0.0 \
  --traj-perpoint-attn-ce-weight 2.0 --traj-perpoint-ce-sigma-cells 1.0 \
  --traj-temporal-query-mix --traj-locality-prior --traj-temporal-query-spatial --traj-temporal-query-interleave --traj-temporal-query-layers 2 --traj-num-virtual-queries 8 --traj-cond-film --traj-conf-head --traj-conf-bce-weight 0.5 --traj-conf-err-thresh-px 8.0 ${EXTRA_ARGS:-} \
  --save-interval 1 --save-best-only --log-interval 50 --amp --amp-dtype bf16 --device cuda --seed 42 --train-platform none \
  --reweight-power 0.5 --reweight-max-factor 10.0 --no-init-eval-vis --no-checkpoint-vis --no-vis --ema \
  2>&1 | tee "${LOG_FILE:-$RUN/train.log}"
