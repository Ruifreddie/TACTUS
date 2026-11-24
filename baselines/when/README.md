# When baselines (Tab. 3, block b)

Interval localization on TactusNet's frozen V-JEPA 2.1 features, so only the localization method differs from TactusNet.
The whole-window floor is in `common/floors.py`. Each clip is one action window; given the clip
and its true tool-action pair, a method outputs one interval [start, end] on the window axis (0 = first protocol frame,
1 = last; `ex.interval(sid)` is the ground truth). Metrics: `baselines.common.score` (continuous tIoU, R@0.7).

| Method (CLI name) | Paper | Paradigm | Main-table input |
|---|---|---|---|
| ActionFormer (`actionformer`) | ECCV 2022 | anchor-free temporal action detection | 8 steps |
| TriDet (`tridet`) | CVPR 2023 | anchor-free TAD with the Trident boundary head | 8 steps |
| QD-DETR (`qddetr`) | CVPR 2023 | DETR moment retrieval with a text query | 8 steps |
| MS-TCN (`mstcn`) | CVPR 2019 | framewise segmentation, then the main run | 32 steps |

## Commands

Table 3(b) reports each method with its `main` configuration and seed 0 (Supplement D.5, Table S15). Use the following commands for those runs.

```bash
source baselines/env.example.sh               # or your edited copy; run_baselines.sh sets the same variables itself
$BL_PY -m pip install -r baselines/when/requirements.txt
source baselines/when/run.sh
prep_when            # CPU: pool the 8-step features (train/val/test) + CLIP text features of the pair templates
extract32            # GPU: token-equivalence check on 3 test clips, then the 32-step features (MS-TCN)
train_actionformer --configs main --seeds 0
train_tridet --configs main --seeds 0
train_qddetr --configs main --seeds 0
train_mstcn --configs main --seeds 0
score_when           # CPU: metrics_test.json (bootstrap CIs) of every finished row
```

The pieces, one by one:

```bash
$BL_PY -m baselines.when.features pool  [--splits train,val,test]       # -> $BL_WHEN_FEATS/s8/<sid>.npy  fp16 [8,1024]
$BL_PY -m baselines.when.features text                                  # -> $BL_WHEN_FEATS/text/clip_vitb32.npz
$BL_PY -m baselines.when.features check --n 3                           # -> $BL_WHEN_FEATS/s32/_token_check.json
$BL_PY -m baselines.when.features extract32 [--splits ...] [--batch 4]  # -> $BL_WHEN_FEATS/s32/<sid>.npy fp16 [32,1024]
$BL_PY -m baselines.when.train --method tridet --out $BL_OUT/when/x [--steps 8] [--config main] [--seed 0 | --seeds 0,1,2] [--set k=v ...]
$BL_PY -m baselines.when.sweep --method tridet --out $BL_OUT/when/tridet_s8 [--configs main,b64,l2] [--seeds 0,1,2] [--jobs 3]
$BL_PY -m baselines.common.score --pred $BL_OUT/when/tridet_s8/pred_test.npz
$BL_PY -m baselines.when.tests.test_when                                # CPU unit checks (axes, decoding, conditioning, recipes)
```

`$BL_WHEN_FEATS` defaults to `$BL_OUT/when/feats`; keep it on persistent storage (extract32 costs GPU hours).
Trainer contract (baselines/README.md): `--smoke`, resume from `last.pt` (refused if the method, recipe or seed differ;
`--fresh` discards it), `best.pt` by val tIoU (`common.train_utils.Checkpointer`), `log.jsonl` per epoch,
`pred_val.npz`/`pred_test.npz` (`common.preds`, kind `when`, with the fallback flags), `info.json` with the recipe actually
used, the official configuration it starts from and every adaptation (`adaptations`: setting, official, TACTUS, reason).
After a finished run the optimizer state is dropped from the checkpoints (`--keep-optimizer` keeps it).
For Table 3(b), the [results collector](../pod/collect_v4.py) reads `<out>/main/seed0/pred_test.npz` for each method.
For an optional sweep, `<out>/<config>/summary.json` records validation results by seed.
`<out>/selection.json` identifies the configuration with the highest mean validation tIoU and reports that configuration's test tIoU and R@0.7 per seed, with their mean and standard deviation.
The top-level prediction files, `<out>/pred_{val,test}.npz`, come from the selected configuration's best validation seed.

## Features

All rows use TactusNet's frozen V-JEPA 2.1 ViT-L/384 (final layer), mean-pooled over the 24x24 grid.

- **8 steps** (`pool`, CPU): the mean of `ex.tokens(sid)` [8,24,24,1024]. Step k = protocol frames 2k, 2k+1, centred at
  (2k+0.5)/15 on the window axis.
- **32 steps** (`extract32`, GPU): 64 frames `round(linspace(abs_start, abs_end, 64))` (end frames included; the rule of the
  16 protocol frames), read with `ex.raw_frames`; TactusNet's preprocessing (RGB, /255, bilinear squash to 384x384, ImageNet
  mean/std; the token converter's CPU code path); the same encoder loaded the same way (this repo's
  `tactusnet/_vendored_vjepa_wrapper.py`, the vjepa2 source in `$U/third_party/t2b/vjepa2` or `$VJEPA_REPO`, the
  checkpoint `$VJEPA_CKPT`), fp32 with default
  TF32 flags as the cache; 4 chunks of 16 frames -> 32 tokens, rounded to fp16 like the cache, then the spatial mean.
  Step k is centred at (2k+0.5)/63. Windows shorter than 64 frames repeat frames.
- **Token-equivalence check** (`check`): encodes the 16 protocol frames of a few clips with the extract32 code and
  compares the result with `ex.tokens(sid)`. The two must agree after fp16 rounding. The report goes to
  `$BL_WHEN_FEATS/s32/_token_check.json`. `extract32` in `run.sh` runs it first.

## Recipes

Rule: each method keeps its **official configuration** (architecture, optimizer, LR schedule shape,
losses, decoding) and changes only what TACTUS requires: (a) 6 output channels, one per valid pair, with the loss and the
readout on the true pair's channel; (b) the sequence length (8 or 32 steps): pyramid levels, regression ranges and similar
settings; (c) the schedule length, scaled to our data (the official number of epochs, so the iterations follow from the
number of Gold train windows); (d) the batch size, with linear LR scaling. Where the official batch is tiny
(ActionFormer and TriDet 2, MS-TCN 1), the **default config `main` keeps the official batch and LR** (one window per
sample); the linearly scaled `b64` (batch 64) is only an extra entry of the default sweep that val may choose. Every
change is in `info.json["adaptations"]` (the batch/LR entry reads "unchanged" for `main`).

| | ActionFormer | TriDet | QD-DETR | MS-TCN |
|---|---|---|---|---|
| official config | THUMOS14 `configs/thumos_i3d.yaml` @61ea7eb9 | THUMOS14 `configs/thumos_i3d.yaml` @764b8165 | QVHighlights defaults (`config.py`, `train.sh`) @f8628f79 | `main.py` @33ed91c0 |
| features | V-JEPA 2.1, 1024-d | V-JEPA 2.1, 1024-d | V-JEPA 2.1 (L2-norm) + TEF; query: CLIP ViT-B/32 text tokens | V-JEPA 2.1, 1024-d |
| steps | 8 | 8 | 8 | 32 |
| conditioning | 6 sigmoid classes; focal loss and readout on the true pair's channel | same + the true channel's Trident bins | the true pair's fixed text template is the query | 6 sigmoid channels; BCE + T-MSE and readout on the true channel |
| architecture | convTransformer (2, 2, L-1), 512-d, global attention, FPN levels L = 3 | SGP (2, 2, L-1), 512-d, k 5, MLP 768, L = 3, 4 Trident bins | 2 T2V + 2 enc + 2 dec layers, 256-d, 8 heads, 10 queries | 4 stages x 6 dilated layers, 64 maps, dropout 0.5 |
| regression ranges | [0,4], [4,8], [8,inf) | [0,4], [4,8], [8,inf) | - | - |
| loss | focal + DIoU, centre sampling r 1.5 | IoU-weighted focal + DIoU, r 1.5 | L1 + gIoU + CE (Hungarian), saliency (margin + rank + negative pair) | BCE + 0.15 x T-MSE (tau 4), all stages |
| epochs | 5 warmup + 30 cosine | 20 warmup + 20 cosine | 200, StepLR(400) | 50 |
| batch (`main`) | 2 (official) | 2 (official) | 32 (official) | 1 window (official: 1 video) |
| LR (`main`) | 1e-4 (official) | 1e-4 (official) | 1e-4 (official) | 5e-4 (official) |
| optimizer | AdamW wd 0.05, clip 1.0, EMA 0.999 for eval | AdamW wd 0.025, clip 1.0, EMA 0.999 | AdamW wd 1e-4, clip 0.1 | Adam |
| augmentation | official crop (ratio [0.9, 1], trunc 0.5) | official crop + input noise 5e-4 | - | - |
| default sweep (x 3 seeds) | `main`; `b64` (batch 64, LR 3.2e-3) | `main`; `b64` (LR 3.2e-3) | `main` only | `main`; `b64` (LR 3.2e-2) |
| extra configs (`--configs`) | `l2` (2 levels) | `l2` | - | `l10` (official 10 layers), `l10_b64` |
| iterations / epoch (`main`) | Gold train clips / 2 (about 6.2k) | same | Gold train clips / 32 (about 390) | Gold train clips (about 12.4k) |

Required adaptations, in detail (also in each `info.json`):
- *all*: the input features (the shared TACTUS features, not the official I3D/SlowFast/CLIP ones), 6 pair channels, and the
  official number of epochs over our windows.
- *ActionFormer/TriDet*: `max_seq_len` 2304 -> T; FPN levels 6 -> 3 (strides 1, 2, 4; a window spans 7.5 steps, so levels
  whose range starts at 8 steps never get positives) and the official range ladder cut to them; ActionFormer's local window
  19 -> global attention (longer than the sequence); TriDet's 16 Trident bins -> 4 (bins beyond the 8-step window would
  only see padding; 16 at 32 steps). Kept official: batch 2 and LR 1e-4 (`main`; `b64` = batch 64 with LR x32), centre
  sampling, the THUMOS crop (`truncate_feats`, called verbatim; at T = 8 it drops 0 or 1 step), EMA evaluation, TriDet's
  `init_conv_vars = 0`.
- *QD-DETR*: the query is a fixed template per pair (`hook|dissect` "a hook dissects tissue", `grasper|grasp` "a grasper grasps
  tissue", `clipper|clip` "a clipper applies a clip", `scissors|cut` "scissors cut tissue", `scissors|dissect` "scissors
  dissect tissue", `grasper|dissect` "a grasper dissects tissue"; written once, never tuned); saliency labels from the
  interval (1 for steps whose centre is inside; 2 positives / 2 negatives sampled, as the official loader for queries
  without scores); the negative query is another pair's template (the official batch roll would repeat the clip's own
  query about half the time with six templates); no rounding to QVHighlights' 2-s clips (our intervals are continuous).
- *MS-TCN*: per-pair sigmoid channels instead of a softmax over classes (a wrong pair could suppress the true channel), so
  the refinement stages take sigmoids; 6 layers per stage at 32 steps (dilations 1..32; the official 10 is the extra
  config `l10`). Kept official: batch 1 (one window instead of one video) and LR 5e-4 (`main`; `b64` = batch 64, LR x64).

Decoding (fixed before any run, identical on val and test):
- ActionFormer / TriDet: the top-1 point of the true pair's channel over all FPN levels, after the official filters
  (score > 0.001, duration > 0.05 steps); with the official per-class soft-NMS (`multiclass_nms`, which disables voting)
  the top segment is never changed, so no NMS is run. No point left: the whole window (fallback flag).
- QD-DETR: the top-1 query slot by foreground probability, clipped to the window; it always outputs a moment.
- MS-TCN: the contiguous run of the true channel above 0.5 with the highest summed probability; boundaries at the
  linearly interpolated 0.5 crossings between step centres; a run touching the first or last step extends to 0 or 1.
  No run: the whole window (fallback flag).

Time mapping: step k is at (2k+0.5)/(F-1) on the window axis (F = 16 or 64 frames); ActionFormer/TriDet points sit at
grid g = k, QD-DETR spans use u with step k = [k/T, (k+1)/T] (`data.py`). Predictions are clipped to [0, 1].

## GPU hours on one RTX 4090 (estimates)

Estimates, scaled from timings of the full-size models on a smaller GPU (+-30%); not measured on a 4090. The heads are
small and launch-bound: at the official batches an iteration costs about as much as at batch 64, so `main` runs are
long, and 3 concurrent runs on one GPU give about twice the throughput of one. `run.sh` therefore runs the sweeps
with `--jobs 3`.

| Step | Cost of one run | Default sweep, 3 seeds, `--jobs 3` (sequential) |
|---|---|---|
| `pool` (CPU) | reads 9.4 MB of tokens per clip | 15.5k Gold clips = about 146 GB read: minutes to ~30 min |
| `extract32` (fp32) | ~0.7 s per clip | Gold (15.5k): ~3 h; all ~23k incl. Silver: ~4.5 h |
| ActionFormer | `main` 1.2-1.6 h (about 6.2k it x 35 epochs); `b64` 9 min | ~2.6 h (4.7 h) |
| TriDet | `main` 1.5-1.8 h (about 6.2k it x 40 epochs); `b64` 9 min | ~3.0 h (5.4 h) |
| QD-DETR | 40 min (about 390 it x 200 epochs) | ~1.0-1.3 h (2.0 h) |
| MS-TCN | `main` 1.3 h (about 12.4k it x 50 epochs); `b64` 1.4 min | ~2.0 h (3.9 h) |
| optional extras | `l2` = `main` cost; MS-TCN `l10` ~1.6x `main` | |

Total with `--jobs 3`: about 9 GPU-hours for the four sweeps plus ~3 h for extract32 (Gold only). Only MS-TCN needs the
32-step features, and only on the Gold splits (Silver is not used). bf16 autocast would make extract32 about 4x faster,
but its tokens would no longer match TactusNet's cache, so it is not offered.

## Smoke runs and tests

- `smoke_when` (`run.sh`) trains every method with `--smoke --fresh` on the current export and writes under
  `$BL_OUT/when/smoke`. `--smoke` runs 2 epochs on the first training clips. If 32-step features are missing, a
  smoke run replaces them with the 8-step features interpolated to 32 steps and counts them in
  `info.json["smoke_standin_clips"]`; a real run refuses missing features. Smoke numbers only prove the code path.
- Resume: a run stopped with `--stop-after N` and started again with the same command continues from `last.pt`.
  Resuming with another method, recipe or seed is refused; `--fresh` discards `last.pt`.
- `tests/test_when.py` (CPU) checks the axes, the run decoding, that the losses train only the true pair's channel,
  that the decoders read only that channel, the crop, the negative queries and the recipes.
