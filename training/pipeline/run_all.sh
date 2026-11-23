#!/usr/bin/env bash
# Two phases, one persistent directory (GV). Every part can be re-run after a crash, and neither phase starts unless GV is complete.
# PHASE=cpu on a CPU machine: unpack, crop boxes, export clips, crop sheets, pack the export into GV.
# Default (PHASE=gpu) on the GPU machine: restore the export, V-JEPA 2.1 tokens, pilot, A, B, compare.
set -uo pipefail
GV=${GV:-/workspace}; WORK=${WORK:-/work}; PHASE=${PHASE:-gpu}; mkdir -p "$WORK"
[ -f "$GV/raw/READY" ] || { echo "volume not ready: run fill_volume.py first"; exit 1; }
[ -d "$WORK/code" ] || { mkdir -p "$WORK/code"; tar -xf "$GV/code.tar" -C "$WORK/code"; }
source "$WORK/code/pipeline/env.sh"; [ "${SKIP_PIP:-0}" = 1 ] || "$PY" -m pip -q install -r "$X/requirements.txt" || exit 1
# A network volume's client can return "Cannot allocate memory" when too many files are read at once, so reads are few and retried.
export VOL_JOBS=${VOL_JOBS:-6}
unpack_raw(){ [ -f "$WORK/.raw_ok" ] && return 0; mkdir -p "$HF_RAW" "$WORK/.raw"; cp "$GV"/raw/*.json "$HF_RAW/" || return 1
  ls "$GV"/raw/*.tar | xargs -P "$VOL_JOBS" -I{} bash -c 'm="$WORK/.raw/$(basename "$1")"; [ -f "$m" ] && exit 0; for i in 1 2 3 4 5 6; do tar -xf "$1" -C "$HF_RAW" && { touch "$m"; exit 0; }; sleep $((i*20)); done; echo "cannot read $1"; exit 1' _ {} && touch "$WORK/.raw_ok"; }
step(){ local n=$1; shift; if [ -f "$WORK/.done/$n" ]; then echo "skip $n"; return 0; fi; "$@" && touch "$WORK/.done/$n"; }
mkdir -p "$WORK/.done"

if [ "$PHASE" = cpu ]; then
  exp1(){ { [ -s "$WORK/video_boxes.json" ] || "$PY" "$X/video_boxes.py" --raw "$HF_RAW" --out "$WORK/video_boxes.json" --workers "$WORKERS"; } && "$PY" "$X/export_clips.py" --boxes "$WORK/video_boxes.json" --unitap "$U" --raw "$HF_RAW" --out "$EXPORT" --split-dir "$X/splits/gold_v2" --silver-index "$HF_RAW/index_silver.json" --groups gold,silver --workers "$WORKERS" && "$PY" "$X/crop_sheet.py" --raw "$HF_RAW" --export "$EXPORT" --boxes "$WORK/video_boxes.json" --out "$GV/results/crop"; }
  pack1(){ "$PY" "$X/volume_shards.py" pack "$EXPORT" "$GV/export" && cp "$WORK/video_boxes.json" "$GV/export/video_boxes.json" && touch "$GV/export/READY"; }
  unpack_raw || { echo "unpacking the raw data failed"; exit 1; }
  step export exp1 || exit 1
  [ "${STOP_BEFORE_PACK:-0}" = 1 ] && { echo "stopped before pack: read the crop sheets"; exit 0; }
  step pack pack1 || exit 1
  echo "cpu phase done"; exit 0
fi

[ -f "$GV/export/READY" ] || { echo "export not on the volume: run PHASE=cpu first"; exit 1; }
"$PY" -c "import torch;n=torch.cuda.device_count();print('GPUs',n,[torch.cuda.get_device_name(i) for i in range(n)]);assert n>=1"
unpack_raw || { echo "unpacking the raw data failed"; exit 1; }
step export "$PY" "$X/volume_shards.py" unpack "$GV/export" "$EXPORT" || exit 1
cp "$GV/export/video_boxes.json" "$WORK/video_boxes.json" || exit 1
# Token caching and every trainer load the weight, so it is read from the local disk.
L="$WORK/weights/$(basename "$VJEPA_CKPT")"; if [ ! -f "$L" ]; then mkdir -p "$WORK/weights"; cp "$VJEPA_CKPT" "$L.part" && [ "$(stat -c %s "$L.part")" = "$(stat -c %s "$VJEPA_CKPT")" ] && mv "$L.part" "$L" || { echo "cannot copy the weight"; exit 1; }; fi; export VJEPA_CKPT="$L"
"$PY" "$X/shards.py" unpack "$GV/tokens/full" "$WORK/tok_full/all" full
tok1(){ bash "$X/cache_tokens.sh" "$EXPORT" "$WORK/tok_full" all && "$PY" "$X/shards.py" pack "$WORK/tok_full/all" "$GV/tokens/full" full; }
step tokens tok1 || exit 1
step pilot bash "$X/pilot.sh" || exit 1
[ "${STOP_AFTER_PILOT:-0}" = 1 ] && { echo "stopped after pilot"; exit 0; }
bash "$X/run_exp.sh" A || exit 1
bash "$X/run_exp.sh" B || exit 1
"$PY" "$X/compare.py" "$WORK/exp/A" "$WORK/exp/B" "$GV/results/compare.md"; cat "$GV/results/compare.md"; touch "$GV/results/ALL_DONE"; echo ALL DONE
