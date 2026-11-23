#!/usr/bin/env bash
# Run a UniTAP recipe script unchanged except for machine lines (paths, sudo power caps, CPU pinning).
# usage: recipe.sh RUN_ID GPU  (cwd = experiment repo). Extra trainer flags come from EXTRA_ARGS and override the recipe's.
set -euo pipefail
RUN=$1; G=$2; EXP=$(pwd); mkdir -p "experiments/runs/$RUN"
sed -e "s#^WS=.*#WS=\"$EXP\"#" -e "s#^PY=.*#PY=\"$PY\"#" -e '/sudo/d' -e '/^( while true/d' -e '/^MON=/d' \
    -e 's#nice -n 5 taskset -c [0-9-]* ##' -e 's#--vjepa-checkpoint "$WS/weights/vjepa2_1_vitl_dist_vitG_384.pt"#--vjepa-checkpoint "$VJEPA_CKPT"#' \
    "$U/experiments/runs/$RUN/run.sh" > "experiments/runs/$RUN/run.sh"
CUDA_VISIBLE_DEVICES=$G bash "experiments/runs/$RUN/run.sh"
