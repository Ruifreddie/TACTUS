#!/usr/bin/env bash
# TACTUS baselines (Tab. 3) and TactusNet ablations (Tab. 4) on one 4x RTX 4090 machine with persistent storage at $GV
# and a fast local disk at $WORK. It sets its own BL_* variables (it does not read baselines/env.example.sh) and expects
# the packed inputs listed below on $GV; they are not part of the public release (see baselines/README.md).
#   nohup bash run_baselines.sh > /work/run.log 2>&1 &     # re-run after any crash: finished jobs are skipped, trainers resume
#   bash run_baselines.sh status                           # what is done, running, pending, failed
# One global job queue: every GPU picks the next job whose inputs are ready (longest first). CPU preparation runs in the
# background and marks its steps done in the same queue. Results are copied to the volume every 15 minutes:
#   $GV/results/baselines/runs/  (pred_*.npz, metrics_*.json, info.json, logs)   $GV/state/baselines/ (best.pt, last.pt)
# Inputs on the volume: baselines_code.tgz (make_bundle.py), code.tar (UniTAP, for TactusNet), raw/*.tar (raw clips),
# results/baselines/export_gt.tgz (export without video_16_npy), tokens/full/*.tar, weights/, state/{A,B}/state.tar.
set -uo pipefail
export GV=${GV:-/workspace-global} WORK=${WORK:-/work}
export BUNDLE=$WORK/code U=$WORK/code/unitap X=$WORK/code/pipeline KC_RNG=$WORK/code/repro/action_range/code
export EXPORT=$WORK/export HF_RAW=$WORK/hf PY=${PY:-python}
# GPU lanes: MIG slices must be addressed by UUID (CUDA_VISIBLE_DEVICES=MIG-...), whole GPUs by index
DEVICES=($(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-fA-F-]*'))
[ ${#DEVICES[@]} -eq 0 ] && DEVICES=($(nvidia-smi -L 2>/dev/null | grep -o '^GPU [0-9]*' | awk '{print $2}'))
export N_GPU=${N_GPU:-${#DEVICES[@]}}
export BL_CODE=$WORK/blcode BL_EXPORT=$WORK/export BL_HF_RAW=$WORK/hf BL_OUT=$WORK/bl BL_WEIGHTS=$WORK/weights_bl   # pod disk: the volume refuses utime/symlinks
export BL_PY=$PY BL_TOKENS=$WORK/tok_full/all BL_WHEN_FEATS=$WORK/bl/when/feats BL_FRAME_CACHE=$WORK/frame_cache
export VJEPA_CKPT=$WORK/weights/vjepa2_1_vitl_dist_vitG_384.pt
export HF_HOME=$WORK/hf_cache TORCH_HOME=$WORK/torch_cache PIP_CACHE_DIR=$WORK/pip_cache TMPDIR=$WORK/tmp
export PYTHONPATH=$WORK/blcode PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VOL_JOBS=${VOL_JOBS:-6} WHEN_JOBS=${WHEN_JOBS:-3} CLS_PACK_WORKERS=${CLS_PACK_WORKERS:-24}
# TactusNet backbone ablation, Table 4(a): the three video encoders besides ours (baselines/ablate/backbones.py), longest first
export ABL_BACKBONES=${ABL_BACKBONES:-"vjepa2_l256 videomae_l224 endofm_b224"} ABL_S1_SCHEDULE=${ABL_S1_SCHEDULE:-full}
# POD_ROLE: which share of the jobs this machine runs (the work can be split over several machines)
#   kpt_where  keypoints + SegFormer | cls_when  classification + When | abl_ours  TactusNet ablations + 3 backbones
#   abl_more   3 more backbones (no restore of experiments A/B or of the 216 GB token cache) | all  everything (N_GPU lanes)
export POD_ROLE=${POD_ROLE:-all}
case $POD_ROLE in
  kpt_where) JOB_RE='^(kpt_|where_)'; NEED="kpt where";;
  cls_when)  JOB_RE='^(cls_|when_)'; NEED="cls when"; export CLS_NO_PACK=${CLS_NO_PACK:-1};;   # 200 GB disk: JPEG frame cache only
  abl_all)   JOB_RE='^abl_'; NEED="ablate";;   # every ablation job (all ABL_BACKBONES) on one multi-GPU pod
  *)         JOB_RE='.'; NEED="cls when kpt where ablate";;   # all: Tables 3 and 4 on one pod (N_GPU lanes, one queue)
esac
needs(){ [[ " $NEED " == *" $1 "* ]]; }
Q=$WORK/queue; mkdir -p "$Q" "$WORK/logs" "$TMPDIR" "$BL_OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ------------------------------------------------------------------------------------------------ CPU preparation
setup(){
  mkdir -p "$BL_CODE" && tar -xzf "$GV/baselines_code.tgz" -C "$BL_CODE" || return 1
  [ -f "$BUNDLE/.ok" ] || { mkdir -p "$BUNDLE" && tar -xf "$GV/code.tar" -C "$BUNDLE" && touch "$BUNDLE/.ok"; } || return 1
  cat "$BL_CODE"/baselines/*/requirements.txt 2>/dev/null | grep -v '^\s*#' | grep -v '^\s*$' | sort -u > "$WORK/requirements_bl.txt"
  $PY -m pip install -q "transformers==5.1.0" "timm==1.0.24" opencv-python-headless scipy pillow && \
  $PY -m pip install -q -r "$WORK/requirements_bl.txt" || return 1
  $PY -c "import torch, transformers, timm, cv2; print(torch.__version__, transformers.__version__, timm.__version__, torch.cuda.device_count())"
}
unpack_gt(){ tar -xzf "$GV/results/baselines/export_gt.tgz" -C "$WORK" && test -f "$EXPORT/meta.json"; }
unpack_raw(){
  mkdir -p "$HF_RAW" "$WORK/.raw"; cp "$GV"/raw/*.json "$HF_RAW/" 2>/dev/null
  ls "$GV"/raw/*.tar | xargs -P "$VOL_JOBS" -I{} bash -c 'm="'"$WORK"'/.raw/$(basename "$1")"; [ -f "$m" ] && exit 0;
    for i in 1 2 3 4 5 6; do tar -xf "$1" -C "'"$HF_RAW"'" && { touch "$m"; exit 0; }; sleep $((i*20)); done; echo "cannot read $1"; exit 1' _ {}
}
copy_vjepa(){ mkdir -p "$WORK/weights"; [ -s "$VJEPA_CKPT" ] || cp "$GV/weights/vjepa2_1_vitl_dist_vitG_384.pt" "$VJEPA_CKPT"; }
pool_tokens(){ $PY -m baselines.pod.pool_shards --shards "$GV/tokens/full" --out "$BL_WHEN_FEATS/s8" --splits train,val,test; }
fam(){ source "$BL_CODE/baselines/$1/run.sh"; }        # load one family's step functions in a subshell
prep(){   # prep NAME cmd...: a CPU step, marked in the queue dir
  local n=$1; shift; [ -f "$Q/$n.done" ] && return 0
  log "prep start $n"; if ( "$@" ) > "$WORK/logs/$n.log" 2>&1; then touch "$Q/$n.done"; log "prep done $n"; else touch "$Q/$n.failed"; log "prep FAILED $n (logs/$n.log)"; fi
}
prep_all(){
  prep setup setup; [ -f "$Q/setup.done" ] || return 1
  prep unpack_gt unpack_gt & prep unpack_raw unpack_raw & prep copy_vjepa copy_vjepa & wait
  [ -f "$Q/unpack_gt.done" ] && [ -f "$Q/unpack_raw.done" ] && [ -f "$Q/copy_vjepa.done" ] && touch "$Q/prep_data.done"
  needs when   && ( prep pool_tokens pool_tokens; [ -f "$Q/pool_tokens.done" ] && prep prep_when bash -c 'source "$BL_CODE/baselines/when/run.sh" && prep_when' ) &
  needs kpt    && prep prep_kpt bash -c 'source "$BL_CODE/baselines/kpt/run.sh" && prep_kpt' &
  needs where  && prep prep_where bash -c 'source "$BL_CODE/baselines/where/run.sh" && prep_where' &
  needs ablate && prep prep_ablate bash -c 'source "$BL_CODE/baselines/ablate/run.sh" && prep_ablate' &
  needs cls    && prep prep_cls bash -c 'source "$BL_CODE/baselines/cls/run.sh" && prep_cls' &
  wait; log "preparation finished"
}

# ------------------------------------------------------------------------------------------------ GPU job table
# name|deps|family|command  (longest first; estimates on one 4090 in the families' READMEs). Every baseline runs its
# official schedule: HRNet and ViTPose 210 epochs, SegFormer 160k iterations, the rest as published.
base_jobs(){ cat <<'EOF'
kpt_hrnet|prep_kpt|kpt|train_hrnet
where_segformer|prep_where|where|train_segformer
when_extract32|prep_data|when|$BL_PY -m baselines.when.features extract32 --splits train,val,test --batch 4 --workers 6   # token check done locally (exact); the pod has no unpacked token cache
kpt_dsta|prep_kpt|kpt|train_dsta
kpt_vitpose|prep_kpt|kpt|train_vitpose
cls_videomae|prep_cls|cls|train_videomae
cls_xclip|prep_cls|cls|train_xclip
cls_endofm|prep_cls|cls|train_endofm
when_tridet|prep_when|when|train_tridet --seeds 0
when_actionformer|prep_when|when|train_actionformer --seeds 0
when_mstcn|when_extract32,prep_when|when|train_mstcn --seeds 0
when_qddetr|prep_when|when|train_qddetr --seeds 0
cls_tsn|prep_cls|cls|train_tsn
cls_tsm|prep_cls|cls|train_tsm
EOF
}
abl_jobs(){ ( source "$BL_CODE/baselines/ablate/run.sh"
  for j in $(ablate_jobs); do d=$(_deps "$j"); echo "abl_$j|prep_ablate${d:+,abl_$d}|ablate|_job $j \$GPU"; done ) 2>/dev/null; }
ordered_jobs(){ local b a; b=$(base_jobs); a=$(abl_jobs)
  { grep -E '^kpt_(hrnet|vitpose)\|' <<< "$b"; grep -E '^abl_tok_' <<< "$a"; grep -E '^abl_s1_' <<< "$a"
    grep -vE '^kpt_(hrnet|vitpose)\|' <<< "$b" | grep -E '^(where_|when_extract32|kpt_dsta|cls_(videomae|xclip|endofm))'
    grep -E '^abl_(cls|ours_heads|cascade)' <<< "$a"
    grep -vE '^kpt_(hrnet|vitpose)\|' <<< "$b" | grep -vE '^(where_|when_extract32|kpt_dsta|cls_(videomae|xclip|endofm))'
    grep -vE '^abl_(tok|s1|cls|ours_heads|cascade)' <<< "$a"; } | grep -v '^$'; }
all_jobs(){ ordered_jobs | awk -F'|' -v re="$JOB_RE" '
  { L[NR] = $0; if ($1 ~ re) S[$1] = 1 }
  END { for (i = 1; i <= NR; i++) { split(L[i], f, "|"); if (!(f[1] in S)) continue
          n = split(f[2], d, ","); keep = ""
          for (k = 1; k <= n; k++) if (d[k] !~ /^abl_/ || (d[k] in S)) keep = keep (keep == "" ? "" : ",") d[k]
          print f[1] "|" keep "|" f[3] "|" f[4] } }'; }

run_job(){   # run_job DEVICE family command...   (DEVICE: a GPU index or a MIG UUID)
  local g=$1 f=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g GPU=$g bash -c 'source "$BL_CODE/baselines/'"$f"'/run.sh" && '"$*"
}
lane(){
  local g=$1 line name deps f cmd next d bad pending
  while :; do next=""; pending=0
    while IFS='|' read -r name deps f cmd; do
      [ -z "$name" ] && continue
      { [ -f "$Q/$name.done" ] || [ -f "$Q/$name.failed" ]; } && continue
      pending=1; [ -d "$Q/$name.lock" ] && continue
      bad=0; for d in ${deps//,/ }; do [ -f "$Q/$d.failed" ] && bad=1; [ -f "$Q/$d.done" ] || [ "$bad" = 1 ] || bad=2; done
      [ "$bad" = 1 ] && { mkdir "$Q/$name.lock" 2>/dev/null && echo "input failed" > "$Q/$name.failed"; continue; }
      [ "$bad" = 2 ] && continue
      mkdir "$Q/$name.lock" 2>/dev/null || continue; next="$name|$f|$cmd"; break
    done < <(all_jobs)
    [ -z "$next" ] && { [ "$pending" = 0 ] && return 0; sleep 60; continue; }
    IFS='|' read -r name f cmd <<< "$next"; local dev=${DEVICES[$g]:-$g}; cmd=${cmd//\$GPU/$dev}
    log "GPU $g: start $name"; local t0; t0=$(date +%s)
    if run_job "$dev" "$f" "$cmd" > "$WORK/logs/$name.log" 2>&1; then echo $(( $(date +%s) - t0 )) > "$Q/$name.done"; log "GPU $g: done $name ($(( ($(date +%s) - t0) / 60 )) min)"
    else echo $(( $(date +%s) - t0 )) > "$Q/$name.failed"; log "GPU $g: FAILED $name (logs/$name.log)"; tail -3 "$WORK/logs/$name.log"; fi
  done
}

# ------------------------------------------------------------------------------------------------ results to the volume
persist(){ $PY - "$BL_OUT" "$GV" <<'EOF'
import os, shutil, sys
src, gv = sys.argv[1:3]; skip_dirs = {"cache", "cls_cache", "feats", "frame_cache", "tok", "__pycache__"}
for d, ds, fs in os.walk(src):
    ds[:] = [x for x in ds if x not in skip_dirs]
    for f in fs:
        p = os.path.join(d, f); rel = os.path.relpath(p, src)
        if f.endswith(".pt"):
            if f not in ("best.pt", "last.pt"): continue
            dst = os.path.join(gv, "state/baselines", rel)
        elif f.endswith((".npz", ".json", ".jsonl", ".log", ".txt", ".tex", ".md", ".csv")):
            dst = os.path.join(gv, "results/baselines/runs", rel)
        else:
            continue
        try:
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(p) or os.path.getsize(dst) != os.path.getsize(p):
                os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(p, dst + ".part"); os.replace(dst + ".part", dst)
        except OSError as ex:
            print("persist failed", rel, ex)
EOF
mkdir -p "$GV/results/baselines/runs/pods/$POD_ROLE"; cp -r "$WORK/logs" "$Q" "$GV/results/baselines/runs/pods/$POD_ROLE/" 2>/dev/null; true; }

status(){ while IFS='|' read -r name deps f cmd; do
    if [ -f "$Q/$name.done" ]; then echo "done    $name $(( $(cat "$Q/$name.done") / 60 )) min"; elif [ -f "$Q/$name.failed" ]; then echo "FAILED  $name"
    elif [ -d "$Q/$name.lock" ]; then echo "running $name"; else echo "pending $name (needs $deps)"; fi; done < <(all_jobs)
  ls "$Q" | grep -E '^(prep_|setup|unpack|copy|pool)' ; }

finish(){
  needs when   && ( source "$BL_CODE/baselines/when/run.sh" && score_when ) > "$WORK/logs/score_when.log" 2>&1
  needs where  && ( source "$BL_CODE/baselines/where/run.sh" && score_where ) > "$WORK/logs/score_where.log" 2>&1
  needs ablate && ( source "$BL_CODE/baselines/ablate/run.sh" && ablate_collect ) > "$WORK/logs/ablate_collect.log" 2>&1
  persist; log "finished: $(ls "$Q" | grep -c '\.done$') done, $(ls "$Q" | grep -c '\.failed$') failed"; ls "$Q" | grep '\.failed$' || true
}

main(){
  ( while :; do sleep 900; persist; done ) & local PERSIST_PID=$!
  prep_all &
  until [ -f "$Q/setup.done" ] || [ -f "$Q/setup.failed" ]; do sleep 20; done
  [ -f "$Q/setup.failed" ] && { log "setup failed"; cat "$WORK/logs/setup.log"; exit 1; }
  for g in $(seq 0 $((N_GPU - 1))); do lane "$g" & done
  wait %2 2>/dev/null; wait $(jobs -p | grep -v "^$PERSIST_PID$") 2>/dev/null
  kill $PERSIST_PID 2>/dev/null; finish
}

case "${1:-main}" in status) status;; persist) persist;; finish) finish;; *) main;; esac
