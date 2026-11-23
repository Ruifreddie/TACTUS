# Training pipeline

These scripts train all TactusNet heads on the data as published at [hding2455/surgical-action-affordance](https://huggingface.co/datasets/hding2455/surgical-action-affordance). Nothing is re-annotated. Start with [`../README.md`](../README.md) for prerequisites, environment variables and the step-to-checkpoint table.

## Experiments

| Id | Training set | Val and test |
| --- | --- | --- |
| A | Gold train (79 videos) | Gold val (11 videos) and Gold test (11 videos) |
| B | Gold train + Silver (40 more videos) | same as A |
| C | Silver only | same as A |

The split is in `splits/gold_v2/`. All three experiments use the same recipes. Silver annotations are used as published.

## Order

```bash
export GV=/path/to/persistent WORK=/path/to/fast_disk
mkdir -p "$GV" && tar -cf "$GV/code.tar" -C training pipeline unitap repro   # from the repository root
python training/pipeline/fill_volume.py           # data and encoder weights -> $GV, ends with $GV/raw/READY
PHASE=cpu bash training/pipeline/run_all.sh       # crop boxes, clip export, crop sheets -> $GV/export
bash training/pipeline/run_all.sh                 # tokens, pilot, experiment A, experiment B, compare.md
bash "$WORK/code/pipeline/run_exp.sh" C           # optional: Silver only
```

- `fill_volume.py` needs `huggingface_hub` and `requests`. It handles one video at a time: download, check file sizes against Hugging Face, tar, copy to `$GV/raw/`, delete the local copy. It also downloads `vjepa2_1_vitl_dist_vitG_384.pt` and checks its SHA-256.
- `run_all.sh` refuses to start without `$GV/raw/READY`. The GPU phase refuses to start without `$GV/export/READY`.
- The CPU phase needs no GPU. `STOP_BEFORE_PACK=1` stops it after the crop sheets, so the crop can be checked before the export is packed.
- The GPU phase runs a pilot before the experiments. `STOP_AFTER_PILOT=1` stops there. If the pilot runs out of memory, set `KB` to a smaller batch. Pilot and experiments both honour it.
- Every step is resumable. Finished steps leave marker files in `$WORK/.done/` and `$WORK/exp/<E>/.done/`.

## Scripts

| Script | Purpose |
| --- | --- |
| `env.sh` | Paths and defaults shared by every step. |
| `fill_volume.py` | Downloads the dataset and the encoder weights into `$GV`. |
| `run_all.sh` | CPU phase and GPU phase, end to end. |
| `video_boxes.py` | One crop box per video. `--raw RAW --out boxes.json` |
| `export_clips.py` | Dataset clips to per-clip arrays, using the exporter in `unitap/tools/export_case32.py`. Clip ids: Gold `000000`…, Silver `100000`…, sorted by video, start, end. |
| `crop_sheet.py` | Crop check images and `crop_report.json` for an export. `--raw --export --boxes --out` |
| `audit_crop.py` | The same check on a small sample fetched from Hugging Face, before the full download. `--split-dir --hf-meta --out --code`. It reads `index_silver.json` and `review_status.json` from `--hf-meta`. |
| `cache_tokens.sh` | V-JEPA 2.1 tokens, one process per GPU. `DATA_ROOT CACHE_ROOT SPLIT` |
| `pilot.sh` | Short run of the stage-1 recipe on every GPU, to check memory and speed. |
| `run_exp.sh` | One experiment end to end. `A`, `B` or `C` |
| `make_exp.py` | Lays out `$WORK/exp/<E>` as a complete Spatial Head working directory with split lists and token links. |
| `recipe.sh` | Runs one recipe of `unitap/experiments/runs/` inside an experiment directory. `RUN_ID GPU` |
| `make_bank.py` | Error bank for the pixel refiner from the stage-1 val dump. |
| `make_ct_root.py` | Dataset root with human contact points as targets (`data/case32/v1_zoom_ct`). |
| `reuse_zoom_tokens.py` | Experiment B reuses A's zoom tokens for clips whose zoom box is identical. |
| `eval_head.py` | Evaluates a Semantic or Temporal Head checkpoint on one split with the trainer's own evaluation. `--code --ckpt --root --split --out` |
| `compare.py` | Table of A against B on the test set. `A_DIR B_DIR OUT_MD` |
| `persist.py` | Saves and restores an experiment's state between `$WORK` and `$GV`. |
| `shards.py`, `volume_shards.py` | Pack token files and the export into tar shards on `$GV`, and unpack them. |
| `smoke_fetch.py` | Downloads a small subset of the data for a code test. |
| `splits/make_gold_split.py` | The script that produced `splits/gold_v2/`. |

## Storage

| Where | What |
| --- | --- |
| `$GV/raw/` | One tar per video, the index files, `READY`. |
| `$GV/weights/` | Encoder weights. |
| `$GV/code.tar` | The code bundle. |
| `$GV/export/` | The clip export as tar shards of about 4 GB, `MANIFEST.json`, `video_boxes.json`, `READY`. |
| `$GV/tokens/full/`, `$GV/tokens/zoom_<E>/` | Token shards of 500 files each. |
| `$GV/state/<E>/state.tar` | Step markers, best checkpoints, prediction dumps, results and logs of experiment `<E>`. |
| `$GV/results/` | `crop/`, one directory per experiment, `compare.md`, `ALL_DONE`. |
| `$WORK/hf/`, `$WORK/export/`, `$WORK/tok_full/all/` | Unpacked dataset, clip arrays, full-frame tokens. |
| `$WORK/exp/<E>/` | One experiment: code copy, data links, runs, logs, results. |

The scripts assume that `$GV` may be slow network or object storage without atomic rename. They write few large files to it, use marker files instead of renames, and retry reads and writes. Training reads and writes on `$WORK` only.

In the original run the clip arrays took about 400 GB and the tokens about 500 GB on `$WORK`.

A new machine can continue an experiment: `run_exp.sh` restores `$GV/state/<E>/state.tar` when the experiment directory has no markers. The zoom dataset and the contact-target roots are then rebuilt from the restored prediction dumps, and the zoom tokens come back from the shards.

## Crop

`video_boxes.py` gives each video one fixed box. It builds a per-pixel map of how much the image changes over time and trims each side to where the motion starts. Black bars and burned-in panels do not move; the surgical view does. Labels are never read, so train, val, test and Silver follow the same rule. SurgVU videos come from one recording system and share the median side and bottom limits. A box that keeps less than half of the frame width or height, or that was built from fewer than 8 frames, is flagged `CHECK` in the output.

`export_clips.py` applies the box to all clips of the video. Points outside the box clip to its edge. Text burned into the surgical view stays, because removing it would cut tissue.

`crop_sheet.py` runs right after the export and writes one frame per video to `$GV/results/crop/`: the source frame with its box on the left, the frame the encoder receives on the right.

## What differs from the recipes

The recipe files are kept as they were. `run_exp.sh` changes the following at launch.

- **Batch size.** The Spatial Head recipes say batch 32. On cards with 30 GB or less they run at batch 24. The learning rate is unchanged.
- **Encoder.** The launchers in `repro/` default to a V-JEPA 2 model at 256 px. `run_exp.sh` passes `--vjepa-arch vjepa2_1_vit_large_384 --input-size 384 --vjepa-checkpoint "$VJEPA_CKPT" --vjepa-hf-repo ""` to every trainer, so all heads train on V-JEPA 2.1 tokens at 384 px.
- **Semantic Head trainer.** `repro/classification/train_stage{1,2}.sh` point at a code snapshot (`classification/original_snapshot/code`) that is not included, because it can only build V-JEPA 2. `run_exp.sh` writes copies named `train_stage{1,2}.v21.sh` that use the trainer in `repro/action_range/code` with the same flags. It validates every epoch and stops after 20 epochs without improvement.
- **Semantic Head, stage 2.** Stage 2 starts from the best stage-1 checkpoint with stage 1's best-value bookkeeping removed, because the two stages optimise different losses. It runs to epoch `max(best stage-1 epoch + 50, 73)`.
- **Temporal Head.** `train_best.sh` runs with `--traj-loss-pixel-scale 384 --target-range-tiou 0`. The second flag disables stopping at a target tIoU.
- **Contact decoder.** It trains on the experiment's own training set only.

## Experiment C: Silver only

```bash
bash "$WORK/code/pipeline/run_exp.sh" C
```

C trains every head on the Silver clips only. Val and test are the same Gold clips as in A and B, and every recipe is the same. It needs the Silver clips in the export and in the full-frame token cache; the default `run_all.sh` phases produce both. C computes its own zoom tokens.

Results are written to `$WORK/exp/C/results/` and copied to `$GV/results/C/`, in the same files as for A and B:

| File | Content |
| --- | --- |
| `cls_{val,test}.json` | Tool, action and pair accuracy. |
| `interval_{val,test}.json` | Interval tIoU. |
| `stage2_{val,test}.json` | Tool-point error of stage 2, with stage 1 for comparison. |
| `contact_human_{val,test}.json` | Contact region against the human regions. |
| `contact_hindsight_{val,test}.json` | Contact region against the hindsight targets. |
| `pipeline_test/`, `pipeline_test.log` | Full Spatial Head on test, per-clip predictions and a summary. |


## Code test on a small subset

`smoke_fetch.py SPLIT_DIR HF_META OUT` downloads a few short clips per tool–action pair and writes them in the dataset layout: `OUT/hf/`, `OUT/split/index_gold_{train,val,test}.json`, `OUT/index_silver.json`. `SPLIT_DIR` is `splits/gold_v2` and `HF_META` is a directory that holds the dataset's `index_silver.json`.

With `SMOKE=1`, `run_exp.sh` runs every training step for one epoch at a small batch. The repository has no script that chains the two. To use the subset, run the export and token steps of `run_all.sh` (`video_boxes.py`, `export_clips.py`, `cache_tokens.sh`) on it with `$WORK/hf` holding the subset. Results from such a run only show that the code runs.
