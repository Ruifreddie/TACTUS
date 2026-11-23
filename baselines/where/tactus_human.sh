#!/usr/bin/env bash
# TactusNet, human-region targets (Tab. 3 c, last row): retrain ONLY TactusNet's contact decoder (UniTAP run C006 of
# experiment A) on the human contact regions, with everything else as in experiment A; then dump val/test, run UniTAP's
# own evaluations, and convert the dumps into standard prediction files.
#
# The change is one line of UniTAP experiments/runs/C006/run.sh (after pipeline/recipe.sh's machine-line edits):
#   DATA="$WS/data/case32/v1_zoom_hs"  ->  DATA="$WS/data/case32/v1_zoom_ct"      (and RUN .../C006 -> .../$RUN)
# v1_zoom_hs holds the hindsight targets: region_mask/ [16,48,48] and 4 pseudo-corners in the tool slots
# (tools/build_hindsight_labels.py). v1_zoom_ct holds the 4 human contact points in the tool slots, pt_role 1,1,1,1
# (pipeline/make_ct_root.py, the same zoom frames and tokens). Without region_mask/ the trainer's region Dice
# (--traj-region-dice-weight 60) falls back to the region of the visible human points (kpt_combo/loss.py, the C004
# branch: _build_region_cell_targets, frames with >= 3 visible points), and the permutation-invariant point loss uses
# the human corners. The result is exactly UniTAP's C004 recipe (C004/run.sh differs from C006/run.sh only in DATA);
# C005 = C004 on data/union_ct_v1 (5,082 extra clips), which experiment A does not use.
#
# usage: tactus_human.sh EXP [OUT]
#   EXP  UniTAP experiment dir of experiment A after its `labels` step (data/case32/v1_zoom_ct, v1_zoom_hs, zoom tokens).
#        Pod: $WORK/exp/A restored by run.sh restore_exp_A. Local smoke: a copy made by run.sh (never the original).
#   OUT  prediction files and copies of logs, results and the checkpoint (default $BL_OUT/tactus_human)
# env: U (UniTAP code, has experiments/runs/C006/run.sh), X (pipeline), PY (python for UniTAP), BL_PY,
#      VJEPA_CKPT, GPU (default 0), SMOKE=1 (1 epoch, batch $SBATCH=2), KB (batch, default 24 below 30 GB of GPU memory,
#      like run_exp.sh), RUN (default C006H), KEEP_CKPT (default 1: copy best.pt to OUT)
set -euo pipefail
EXP=$(cd "$1" && pwd); BL_CODE=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${2:-${BL_OUT:?}/tactus_human}; RUN=${RUN:-C006H}; GPU=${GPU:-0}; PY=${PY:-python}; BL_PY=${BL_PY:-$PY}
: "${U:?set U to the UniTAP code dir}" "${VJEPA_CKPT:?set VJEPA_CKPT}"
SRC_RUN="$U/experiments/runs/C006/run.sh"; R="$EXP/experiments/runs/$RUN"; mkdir -p "$R" "$OUT" "$EXP/results"
log(){ echo "[tactus_human $(date +%F_%T)] $*"; }
for f in data/case32/v1_zoom_ct/meta.json data/case32/v1_zoom_hs/meta.json data/case32/v1_zoom/boxes.json \
         third_party/t2b/kpt_combo/train.py tools/dump_val_predictions.py; do
  [ -e "$EXP/$f" ] || { echo "missing $EXP/$f: restore experiment A first (run.sh restore_exp_A)"; exit 1; }; done
for s in train val test; do
  n=$(grep -c . "$EXP/data/case32/v1_zoom_ct/$s.txt"); h=$( (find -L "$EXP/data/case32/v1_zoom_ct/vjepa_token_cache/$s" -name '*.npy' 2>/dev/null || true) | wc -l)
  [ "$h" -ge "$n" ] || { echo "zoom tokens incomplete for $s: $h of $n"; exit 1; }; done

# 1. the recipe: recipe.sh's edits (verbatim), plus the target switch; the diff against C006 must be that one line
SED=(-e "s#^WS=.*#WS=\"$EXP\"#" -e "s#^PY=.*#PY=\"$PY\"#" -e '/sudo/d' -e '/^( while true/d' -e '/^MON=/d'
     -e 's#nice -n 5 taskset -c [0-9-]* ##' -e 's#--vjepa-checkpoint "$WS/weights/vjepa2_1_vitl_dist_vitG_384.pt"#--vjepa-checkpoint "$VJEPA_CKPT"#')
sed "${SED[@]}" "$SRC_RUN" > "$R/run.C006.sh"
sed "${SED[@]}" -e 's#DATA="$WS/data/case32/v1_zoom_hs"#DATA="$WS/data/case32/v1_zoom_ct"#' \
    -e "s#RUN=\"\$WS/experiments/runs/C006\"#RUN=\"\$WS/experiments/runs/$RUN\"#" "$SRC_RUN" > "$R/run.sh"
diff "$R/run.C006.sh" "$R/run.sh" > "$R/recipe_change.diff" || true
[ "$(grep -c '^<' "$R/recipe_change.diff")" = 1 ] && [ "$(grep -c '^>' "$R/recipe_change.diff")" = 1 ] && grep -q 'v1_zoom_ct' "$R/recipe_change.diff" \
  || { echo "unexpected recipe change:"; cat "$R/recipe_change.diff"; exit 1; }
log "recipe = C006 with one changed line:"; cat "$R/recipe_change.diff"

# 2. train (the recipe saves best.pt only, by the lowest val loss; a killed run restarts from scratch)
if [ "${SMOKE:-0}" = 1 ]; then EXTRA="--epochs 1 --batch-size ${SBATCH:-2} --num-workers 2"; DB=4
else VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); KB=${KB:-$(( VRAM > 30000 ? 32 : 24 ))}; EXTRA="--num-workers 8 --batch-size $KB"; DB=16; fi
export PYTHONPATH="$EXP/third_party/t2b" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$EXP"
if [ ! -f "$R/.done_train" ]; then
  log "train $RUN ($EXTRA) -> $R/train.log"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU EXTRA_ARGS="$EXTRA" bash "$R/run.sh" > "$R/launcher.log" 2>&1 || { log "training failed"; tail -20 "$R/train.log"; exit 1; }
  [ -f "$R/out/checkpoints/best.pt" ] || { log "no best.pt"; exit 1; }
  echo $(( $(date +%s) - t0 )) > "$R/train_seconds"; touch "$R/.done_train"
fi
# 3. dump val/test with the EMA weights of best.pt, exactly like run_exp.sh's dump()
for s in val test; do
  D="experiments/analysis/pred_${RUN}_best_$s"
  [ -f "$R/.done_dump_$s" ] || { CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/dump_val_predictions.py --checkpoint "$R/out/checkpoints/best.pt" \
      --out "$D" --split "$s" --device cuda --batch-size $DB > "$R/dump_$s.log" 2>&1 && touch "$R/.done_dump_$s"; } || { tail -5 "$R/dump_$s.log"; exit 1; }
  # 4. UniTAP's own evaluations (human regions, hindsight hotspots), as run_exp.sh's evaluate() does for C006
  "$PY" tools/eval_contact_stage.py --dump "$D" --save "results/contact_human_${RUN}_$s.json" > "$R/eval_human_$s.log" 2>&1 || { tail -5 "$R/eval_human_$s.log"; exit 1; }
  "$PY" tools/eval_hindsight.py --dump "$D" --save "results/contact_hindsight_${RUN}_$s.json" > "$R/eval_hindsight_$s.log" 2>&1 || { tail -5 "$R/eval_hindsight_$s.log"; exit 1; }
done
# 5. standard prediction files (+ check that score.py reproduces UniTAP's region_hull_dice), copies for the record
PYTHONPATH="$BL_CODE" "$BL_PY" -m baselines.where.tactus_human convert --exp "$EXP" --run "$RUN" --out "$OUT"
mkdir -p "$OUT/unitap"; cp "$R/run.sh" "$R/recipe_change.diff" "$EXP"/results/contact_*_"$RUN"_*.json "$OUT/unitap/"
sed -E 's/youtube_[A-Za-z0-9_]+/<youtube video>/g' "$R/train.log" > "$OUT/unitap/train.log"   # YouTube task names carry a person's name
cp "$R/out/epoch_metrics.jsonl" "$R/out/train_args.json" "$OUT/unitap/" 2>/dev/null || true
[ "${KEEP_CKPT:-1}" = 1 ] && cp "$R/out/checkpoints/best.pt" "$OUT/unitap/best.pt"
"$BL_PY" - "$OUT" "$R" "$EXTRA" "${SMOKE:-0}" <<'EOF'
import json, os, sys
out, r, extra, smoke = sys.argv[1:5]; a = json.load(open(f"{r}/out/train_args.json")) if os.path.exists(f"{r}/out/train_args.json") else {}
sec = open(f"{r}/train_seconds").read().strip() if os.path.exists(f"{r}/train_seconds") else None
info = {"method": "TactusNet, human-region targets", "recipe": "UniTAP experiments/runs/C006/run.sh with DATA=data/case32/v1_zoom_ct (= UniTAP C004)",
        "trainer_extra_args": extra, "smoke": smoke == "1", "input": "frozen V-JEPA 2.1 ViT-L tokens of the 16 zoomed frames (384^2)",
        "init": "random contact decoder, frozen encoder", "targets": "4 human contact points per frame (box-normalized), region Dice on their hull",
        "epochs": a.get("epochs"), "batch": a.get("batch_size"), "lr": a.get("lr"), "weight_decay": a.get("weight_decay"),
        "optimizer": "AdamW (UniTAP kpt_combo/train.py)", "ema": a.get("ema"), "selection": "lowest val loss_total (recipe)", "train_seconds": sec}
json.dump(info, open(f"{out}/info.json", "w"), indent=1); print(json.dumps(info))
EOF
log "done: $OUT/pred_val.npz $OUT/pred_test.npz"
