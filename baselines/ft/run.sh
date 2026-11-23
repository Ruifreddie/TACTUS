#!/usr/bin/env bash
# Encoder fine-tuning of TactusNet (baselines/ft/README.md) on one multi-GPU pod with the global volume at $GV.
#
#   bash run.sh prep_cpu        CPU pod, once: decode the exact protocol frames of the Gold clips -> $GV/ft/frames shards
#   bash run.sh prep            GPU pod: code, light export, raw clips, V-JEPA weight, frames (shards), experiment A, zoom labels
#   bash run.sh pilot [GPU]     GPU: L0, then 20 steps of the joint run after the epoch-0 check (speed, memory)
#   nohup bash run.sh all > $WORK/ft_run.log 2>&1 &     prep + L0 + every run (one lane per GPU) + evaluation + collect
#   bash run.sh status
# Re-run after a crash or on a new pod: finished steps are skipped (markers in $FT/queue), training resumes from
# state_last.pt, evaluation skips finished clips. Results are copied to $GV/results/ft every 15 minutes and at the end;
# the selected epoch snapshots (heads + encoder) to $GV/state/ft.
# Inputs on the volume (as for run_baselines.sh): baselines_code.tgz (make_bundle.py), code.tar (UniTAP + repro),
# results/baselines/export_gt.tgz, raw/*.tar, weights/vjepa2_1_vitl_dist_vitG_384.pt, state/A/state.tar.
set -uo pipefail
# An SSH session does not see the container's ENV (PIP_BREAK_SYSTEM_PACKAGES, the CUDA paths): take those from PID 1.
if [ -r /proc/1/environ ]; then while IFS= read -r -d '' kv; do
  case "$kv" in PIP_BREAK_SYSTEM_PACKAGES=*|PIP_ROOT_USER_ACTION=*|PATH=*|LD_LIBRARY_PATH=*|LIBRARY_PATH=*) export "$kv" ;; esac
done < /proc/1/environ; fi
export GV=${GV:-/workspace-global} WORK=${WORK:-/work}
export BUNDLE=$WORK/code U=$WORK/code/unitap X=$WORK/code/pipeline KC_RNG=$WORK/code/repro/action_range/code
export EXPORT=$WORK/export HF_RAW=$WORK/hf PY=${PY:-python}
export BL_CODE=$WORK/blcode BL_EXPORT=$WORK/export BL_HF_RAW=$WORK/hf BL_PY=$PY
export VJEPA_CKPT=$WORK/weights/vjepa2_1_vitl_dist_vitG_384.pt EXP_A=$WORK/exp/A FT=${FT:-$WORK/ft}
export HF_HOME=$WORK/hf_cache TORCH_HOME=$WORK/torch_cache PIP_CACHE_DIR=$WORK/pip_cache TMPDIR=$WORK/tmp
export PYTHONPATH=$BL_CODE PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BL_FRAME_CACHE                               # lossy JPEG cache: frames.py writes the exact video_16_npy instead
export VOL_JOBS=${VOL_JOBS:-6}
FT_RUNS=${FT_RUNS:-"joint spa sem tmp control"}    # longest first: joint and spa encode the full frame and the zoom crop
FT_EPOCHS=${FT_EPOCHS:-10} FT_MICRO=${FT_MICRO:-32}   # micro 32 = one forward per step (H100: 6.9 s/step joint, 51 GiB)
 FT_WORKERS=${FT_WORKERS:-8} FT_VAL_BATCH=${FT_VAL_BATCH:-4}
FT_EXTRA=${FT_EXTRA:-}                             # extra train.py flags for every run (e.g. --no-act-ckpt), memory/speed only
DEVICES=($(nvidia-smi -L 2>/dev/null | grep -o '^GPU [0-9]*' | awk '{print $2}'))
N_GPU=${N_GPU:-${#DEVICES[@]}}
Q=$FT/queue; mkdir -p "$Q" "$FT/logs" "$TMPDIR" 2>/dev/null
log(){ echo "[ft $(date +%m-%d_%H:%M:%S)] $*"; }
once(){ local n=$1; shift; [ -f "$Q/$n.done" ] && { log "skip $n"; return 0; }; log "start $n"
  if ( "$@" ) > "$FT/logs/$n.log" 2>&1; then touch "$Q/$n.done"; log "done $n"; else log "FAILED $n ($FT/logs/$n.log)"; tail -5 "$FT/logs/$n.log"; return 1; fi; }

# --------------------------------------------------------------------------------------------- CPU preparation
setup(){ mkdir -p "$BL_CODE" && tar -xzf "$GV/baselines_code.tgz" -C "$BL_CODE" || return 1
  [ -f "$BUNDLE/.ok" ] || { mkdir -p "$BUNDLE" && tar -xf "$GV/code.tar" -C "$BUNDLE" && touch "$BUNDLE/.ok"; } || return 1
  $PY -m pip install -q -r "$X/requirements.txt" || return 1      # experiment A's pinned environment (pipeline/run_all.sh)
  $PY -c "import torch, timm, cv2, einops, decord; print(torch.__version__, timm.__version__, cv2.__version__, torch.cuda.device_count())"; }
unpack_gt(){ tar -xzf "$GV/results/baselines/export_gt.tgz" -C "$WORK" && test -f "$EXPORT/meta.json"; }
unpack_raw(){ mkdir -p "$HF_RAW" "$WORK/.raw"; cp "$GV"/raw/*.json "$HF_RAW/" 2>/dev/null
  ls "$GV"/raw/*.tar | xargs -P "$VOL_JOBS" -I{} bash -c 'm="'"$WORK"'/.raw/$(basename "$1")"; [ -f "$m" ] && exit 0;
    for i in 1 2 3 4 5 6; do tar -xf "$1" -C "'"$HF_RAW"'" && { touch "$m"; exit 0; }; sleep $((i*20)); done; echo "cannot read $1"; exit 1' _ {}; }
copy_vjepa(){ mkdir -p "$WORK/weights"; [ -s "$VJEPA_CKPT" ] || { cp "$GV/weights/vjepa2_1_vitl_dist_vitG_384.pt" "$VJEPA_CKPT.part" && mv "$VJEPA_CKPT.part" "$VJEPA_CKPT"; }; }
frames(){ $PY -m baselines.ft.frames --splits train,val,test --workers "${FRAME_WORKERS:-$(nproc)}"; }   # exact, from the raw clips
# Decoding the 15.5k clips is CPU work (about 5 s per clip and core): a CPU pod does it once and packs the frames onto the
# volume (prep_cpu), so the GPU pod only unpacks them (prep). Without shards, prep decodes on the GPU pod instead.
# Shards of 500 clips (shards.py's layout), written and read in parallel; <shard>.ok marks a complete shard.
pack_frames(){ local src="$EXPORT/video_16_npy" dst="$GV/ft/frames" L="$WORK/.pack"; mkdir -p "$dst" "$L"; rm -f "$L"/list_*
  ( cd "$src" && find . -maxdepth 1 -name '*.npy' -printf '%f\n' | sort ) | split -l 500 -d -a 3 - "$L/list_"
  ls "$L"/list_* | xargs -P "${PACK_JOBS:-3}" -I{} bash -c 'l="$1"; f=$(head -1 "$l"); t=$(tail -1 "$l"); n=$(wc -l < "$l")
      name="'"$dst"'/frames_${f%.npy}_${t%.npy}_${n}.tar"; [ -f "$name.ok" ] && exit 0
      for i in 1 2 3 4; do tar -cf "$name" -C "'"$src"'" -T "$l" && { touch "$name.ok"; exit 0; }; sleep $((i*20)); done
      echo "cannot write $name"; exit 1' _ {} || return 1
  local want got; want=$(ls "$L"/list_* | wc -l); got=$(ls "$dst"/frames_*.tar.ok | wc -l); log "frames shards: $got of $want"
  [ "$got" -eq "$want" ] && touch "$dst/READY"; }
unpack_frames(){ mkdir -p "$EXPORT/video_16_npy" "$WORK/.frames"
  ls "$GV"/ft/frames/frames_*.tar | xargs -P "$VOL_JOBS" -I{} bash -c 'm="'"$WORK"'/.frames/$(basename "$1")"; [ -f "$m" ] && exit 0
    [ -f "$1.ok" ] || { echo "incomplete shard $1"; exit 1; }
    for i in 1 2 3 4 5 6; do tar -xf "$1" -C "'"$EXPORT"'/video_16_npy" && { touch "$m"; exit 0; }; sleep $((i*20)); done
    echo "cannot read $1"; exit 1' _ {}; }
# FRAMES_FROM picks the source of video_16_npy: decode (default, from the raw clips), export (experiment A's full export
# on the volume) or shards (the frame shards written by prep_cpu). check_frames compares the result with A's token cache.
export_frames(){ $PY -m baselines.ft.export_frames --shards "$GV/export" --root "$EXPORT" --jobs "$VOL_JOBS"; }
get_frames(){ case "${FRAMES_FROM:-decode}" in
    export) export_frames || return 1 ;;
    decode) frames || return 1 ;;
    shards) unpack_frames || return 1 ;;
    *) echo "FRAMES_FROM=export|decode|shards"; return 1 ;; esac
  local n; n=$(ls "$EXPORT/video_16_npy" | wc -l); log "video_16_npy: $n clips"; [ "$n" -ge "${MIN_CLIPS:-15400}" ]; }
# fp32 tokens of the frozen encoder from these frames = A's own token cache (the cache A was trained on)?
check_frames(){ local dev=cpu; [ "${#DEVICES[@]}" -gt 0 ] && dev=cuda
  CUDA_VISIBLE_DEVICES=${DEVICES[0]:-} $PY -m baselines.ft.check_frames --exp "$EXP_A" --shards "$GV/tokens/full" --n "${CHECK_N:-12}" \
    --device "${CHECK_DEVICE:-$dev}" --fp32 both --out "$FT/check_frames.json"; }
# CPU pod: frames -> volume, and a dry run of the CPU steps of prep (restore_A, zoom_labels) with the frame check.
prep_cpu(){ once setup setup || return 1
  local p1 p2 p3                                   # wait for these only: all() runs a persist loop in the background
  once unpack_gt unpack_gt & p1=$!; once unpack_raw unpack_raw & p2=$!; once copy_vjepa copy_vjepa & p3=$!; wait $p1 $p2 $p3
  for s in unpack_gt unpack_raw copy_vjepa; do [ -f "$Q/$s.done" ] || return 1; done
  once frames frames || return 1
  once pack_frames pack_frames & local P=$! r p
  once restore_A restore_A && once zoom_labels zoom_labels && once save_zoom save_zoom && once check_frames check_frames; r=$?
  wait $P; p=$?; log "prep_cpu: checks $r, pack $p"; [ $r -eq 0 ] && [ $p -eq 0 ]; }
restore_A(){ mkdir -p "$WORK/tok_none/all"     # no token cache: every token comes from the encoder in these runs
  [ -f "$EXP_A/.done/evaluate" ] || { $PY "$X/make_exp.py" A "$U" "$EXPORT" "$WORK/tok_none" "$HF_RAW" "$EXP_A" \
    && $PY "$X/persist.py" load A "$EXP_A" "$GV/state/A"; }
  [ -f "$EXP_A/.done/evaluate" ] && ls "$EXP_A"/experiments/runs/{cls2,interval}/checkpoints/best.pt "$EXP_A"/experiments/runs/{E007k,E009,C006}/out/checkpoints/best.pt; }
# Zoom and contact roots of A, as run_exp.sh's zoom_build + labels steps, but without the crop videos (the loader cuts
# the crops from the native frames with the same boxes): same seed, same stage-1 dumps -> the same boxes.json.
zoom_labels(){ cd "$EXP_A" && $PY tools/build_zoom_dataset.py --workers "${ZOOM_WORKERS:-16}" --no-video \
    --dump val=experiments/analysis/pred_E007k_best_val --dump test=experiments/analysis/pred_E007k_best_test \
  && $PY "$X/make_ct_root.py" && $PY tools/build_hindsight_labels.py && test -f data/case32/v1_zoom_hs/train.txt; }
# The zoom, contact and hindsight roots are CPU work (build_hindsight_labels.py is single-threaded): the CPU pod writes them
# once to the volume (save_zoom), the GPU pod unpacks them (get_zoom) and builds them itself only when they are missing.
ZOOM_DIRS="v1_zoom v1_zoom_ct v1_zoom_hs"
save_zoom(){ tar -cf "$GV/ft/expA_zoom.tar" -C "$EXP_A/data/case32" $ZOOM_DIRS && touch "$GV/ft/expA_zoom.tar.ok"; }
get_zoom(){ if [ -f "$GV/ft/expA_zoom.tar.ok" ]; then tar -xf "$GV/ft/expA_zoom.tar" -C "$EXP_A/data/case32" && \
    test -f "$EXP_A/data/case32/v1_zoom_hs/train.txt" && test -f "$EXP_A/data/case32/v1_zoom/boxes.json"; else zoom_labels; fi; }
prep(){ once setup setup || return 1
  local p1 p2 p3                                   # wait for these only: all() runs a persist loop in the background
  once unpack_gt unpack_gt & p1=$!; once unpack_raw unpack_raw & p2=$!; once copy_vjepa copy_vjepa & p3=$!; wait $p1 $p2 $p3
  for s in unpack_gt unpack_raw copy_vjepa; do [ -f "$Q/$s.done" ] || return 1; done
  once get_frames get_frames && once restore_A restore_A && once get_zoom get_zoom && once check_frames check_frames; }

# --------------------------------------------------------------------------------------------- GPU
_dev(){ echo "${DEVICES[$1]}"; }
l0(){ CUDA_VISIBLE_DEVICES=$(_dev "${1:-0}") $PY -m baselines.ft.train --measure-l0 --exp "$EXP_A" --out "$FT/l0.json" \
        --micro "$FT_MICRO" --workers "$FT_WORKERS"; }
pilot(){ local g=${1:-0} out=${PILOT_OUT:-$FT/pilot}; once l0 l0 "$g" || return 1     # PILOT_OUT: a second pilot beside the first
  CUDA_VISIBLE_DEVICES=$(_dev "$g") $PY -m baselines.ft.train --run joint --exp "$EXP_A" --l0 "$FT/l0.json" --out "$out" \
    --epochs "$FT_EPOCHS" --micro "$FT_MICRO" --workers "$FT_WORKERS" --val-batch "$FT_VAL_BATCH" --pilot-steps "${PILOT_STEPS:-20}" \
    $FT_EXTRA ${PILOT_EXTRA:-} 2>&1 | tee "$FT/logs/$(basename "$out").log" | grep "^\[ft" ; cat "$out/pilot.json"; }
train_run(){ local r=$1 g=$2
  CUDA_VISIBLE_DEVICES=$g $PY -m baselines.ft.train --run "$r" --exp "$EXP_A" --l0 "$FT/l0.json" --out "$FT/runs/$r" \
    --epochs "$FT_EPOCHS" --micro "$FT_MICRO" --workers "$FT_WORKERS" --val-batch "$FT_VAL_BATCH" $FT_EXTRA; }
# the epochs to evaluate per run (unique), and which heads each needs: "epoch only" lines
_evals(){ $PY - "$FT/runs/$1/selection.json" "$1" <<'EOF'
import json, sys
sel = json.load(open(sys.argv[1]))["selection"]; run = sys.argv[2]
need = {"control": {"semantic": "sem", "temporal": "tmp", "spatial": "spatial"},
        "joint": {"combined": "sem,tmp,spatial", "semantic": "sem", "temporal": "tmp", "spatial": "spatial"},
        "sem": {"semantic": "sem"}, "tmp": {"temporal": "tmp"}, "spa": {"spatial": "spatial"}}[run]
by = {}
for key, only in need.items():
    e = sel[key]; by.setdefault(e, set()).update(only.split(","))
if run == "control": by.setdefault(0, set()).update({"sem", "tmp", "spatial"})     # experiment A through the same path
for e, parts in sorted(by.items()): print(e, ",".join(sorted(parts)))
EOF
}
eval_run(){ local r=$1 g=$2 e only
  while read -r e only; do
    CUDA_VISIBLE_DEVICES=$g $PY -m baselines.ft.evaluate all --exp "$EXP_A" --run-dir "$FT/runs/$r" --epoch "$e" \
      --out "$FT/eval/$r/e$(printf %02d "$e")" --only "$only" --workers "$FT_WORKERS" || return 1
  done < <(_evals "$r"); }
job(){ local r=$1 slot=$2 g; g=$(_dev "$slot")   # a run, then its evaluation, on one GPU
  once "train_$r" train_run "$r" "$g" && once "eval_$r" eval_run "$r" "$g"; }
lane(){ local slot=$1 r
  for r in $FT_RUNS; do mkdir "$Q/$r.lock" 2>/dev/null || continue; job "$r" "$slot"; done; }
persist(){ mkdir -p "$GV/results/ft" "$GV/state/ft"
  ( cd "$FT" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.md' -o -name '*.log' -o -name 'pred_*.npz' \) \
      -not -path './*/valtok/*' -not -path './eval/*/tok/*' -print0 | tar --null -cf - -T - ) | tar -xf - -C "$GV/results/ft" 2>/dev/null
  for r in $FT_RUNS; do [ -f "$FT/runs/$r/selection.json" ] || continue
    for e in $(_evals "$r" | awk '{print $1}'); do local d; d=$(printf "e%02d" "$e")
      [ -d "$FT/runs/$r/epochs/$d" ] && [ ! -f "$GV/state/ft/$r/$d/.ok" ] && [ -f "$FT/runs/$r/DONE" ] && \
        { mkdir -p "$GV/state/ft/$r/$d" && cp "$FT/runs/$r/epochs/$d"/*.pt "$GV/state/ft/$r/$d/" && touch "$GV/state/ft/$r/$d/.ok"; }
    done; done; true; }
collect(){ $PY -m baselines.ft.collect --ft "$FT" --out "$FT/results"; }
status(){ for r in $FT_RUNS; do local s="pending"
    [ -d "$Q/$r.lock" ] && s="running"; [ -f "$Q/train_$r.done" ] && s="trained"; [ -f "$Q/eval_$r.done" ] && s="evaluated"
    echo "$r: $s  $(tail -1 "$FT/runs/$r/log.txt" 2>/dev/null | cut -c1-160)"; done; ls "$Q"; }
all(){ rm -rf "$Q"/*.lock                      # a re-run: the processes that held them are gone
  ( while :; do sleep 900; persist; done ) & local P=$!
  prep || { log "prep failed"; kill $P; exit 1; }
  once l0 l0 0 || { kill $P; exit 1; }
  for s in $(seq 0 $((N_GPU - 1))); do lane "$s" & done; wait $(jobs -p | grep -v "^$P$")
  once collect collect; persist; kill $P 2>/dev/null; log "finished"; status; }

case "${1:-all}" in
  prep_cpu|prep|pilot|status|persist|collect|all|l0) "$@" ;;
  *) if declare -F "$1" >/dev/null; then "$@"     # any step, e.g. run.sh once restore_A restore_A
     else echo "usage: run.sh prep_cpu (CPU pod) | prep | pilot [GPU] | all | status | persist | collect | <step>"; exit 1; fi ;;
esac
