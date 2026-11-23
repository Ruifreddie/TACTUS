# TACTUS baselines

Published methods adapted to the TACTUS benchmark, trained on TACTUS Gold and scored with the same code as TactusNet
(the baseline table of the paper, Tab. 3). This directory also holds the shared data reader, the prediction-file format
and the scorer, so it is the place to start if you want to evaluate your own method.

| Directory | Family | Methods |
|---|---|---|
| `common/` | shared data reader, metrics, prediction files, scorer, floors, TactusNet converter | Majority class, Whole window, Pair-mean interval, Fixed quadrilateral |
| `cls/` | video classification (Which, What, Pair) | TSN, TSM, VideoMAE, X-CLIP, Endo-FM (reserve: MViTv2-S) |
| `when/` | interval localization on frozen V-JEPA 2.1 features (When) | ActionFormer, TriDet, QD-DETR, MS-TCN |
| `kpt/` | tool and contact keypoints (How, Where) | HRNet-W32 + DARK, ViTPose-B, DSTA |
| `where/` | contact region and contact targets (Where) | SegFormer-B2, TactusNet with human-region targets, hindsight target from true tips |
| `ablate/`, `ft/` | TactusNet ablations and encoder fine-tuning | see their own READMEs |
| `pod/` | helpers of `run_baselines.sh` (token pooling, result tables, qualitative data) | - |
| `tests/` | checks of `common/` on a real export | - |

Names used in the scripts: `training/` of this repo is the TactusNet training bundle (`$BUNDLE`). `training/unitap`
(`$U`) is the Spatial Head code, whose working name was UniTAP. `training/pipeline` (`$X`) holds the orchestration
scripts. `$GV` is a persistent storage directory and `$WORK` a fast local disk.

## The five tasks

| Task | Output per clip | Metric (`common/metrics.py`) |
|---|---|---|
| Which / What | one of 6 valid tool-action pairs; tool and action are read from the predicted pair | tool, action and pair accuracy; mean accuracy over the pairs |
| When | one interval `[start, end]` on the window axis | tIoU, recall at tIoU >= 0.3 / 0.5 / 0.7 |
| How | 3 tool points on each of the 16 protocol frames | ADE in pixels at 384 x 384, share of visible points with error > 32 px, median, PCK at 4 / 8 / 16 px |
| Where | a contact region on each of the 16 protocol frames (4 points, or a mask) | Dice with the human region; share of frames with Dice >= 0.5 |

The Gold benchmark has 15,499 clips: 12,372 train, 1,604 val and 1,523 test, split by video
(see `evaluation/README.md`).

## Environment

```bash
cp baselines/env.example.sh baselines/my_env.sh   # edit BL_ROOT and the paths below it; keep the copy in baselines/
source baselines/my_env.sh
$BL_PY -m baselines.tests.test_common             # checks the export, the frame decoder, the floors and the scorer
```

`baselines/env.example.sh` lists every variable with a one-line comment. The ones the code reads:

| Variable | Holds | Default in `common/config.py` |
|---|---|---|
| `BL_EXPORT` | per-clip arrays (the export, see below) | `/work/export` |
| `BL_HF_RAW` | raw clips in the dataset's layout, `<task>/clips/<stem>.mp4` | `/work/hf` |
| `BL_TOKENS` | TactusNet's frozen V-JEPA 2.1 tokens, `<sid>.npy` `[8,24,24,1024]` fp16 | `/work/tok_full/all` |
| `BL_OUT` | predictions, checkpoints and logs | `/work/baselines` |
| `BL_WEIGHTS` | pretrained weights of the baselines | `/workspace/weights/baselines` |
| `VJEPA_CKPT` | V-JEPA 2.1 ViT-L/384 checkpoint (`when/features.py`, `where/tactus_human.sh`) | none |
| `BL_PY` | the Python interpreter used by every `run.sh` | none |
| `PYTHONPATH` | must contain the repo root, so that `python -m baselines....` resolves | none |

Optional: `BL_WHEN_FEATS` (When features, default `$BL_OUT/when/feats`), `BL_CLS_CACHE` (classification frame cache,
default `$BL_OUT/cls_cache`), `BL_FRAME_CACHE` (JPEG cache of decoded protocol frames), `BL_SMOKE_EXP` (a TactusNet
experiment directory for the `--exp` checks).

Base packages: torch, torchvision, numpy, opencv, pillow, scipy, `transformers==5.1.0`, `timm==1.0.24`. Each family
adds a `requirements.txt`.

**The export.** The dataset is at https://huggingface.co/datasets/hding2455/surgical-action-affordance. The baselines do
not read it directly. They read an export: one array file per clip and annotation, written by
`training/pipeline/export_clips.py` (see `training/pipeline/README.md`). The reader expects, under `$BL_EXPORT`:
`labels.json`, `meta.json`, `all_{train,val,test,silver}.txt`, `gt_tool_xy/`, `gt_tool_vis/`, `gt_ct_xy/`,
`gt_ct_vis/`, `sampled_abs_frames/`, `action_range_abs/`, `action_range_mask/` and optionally `video_16_npy/`
(`<sid>.npy` in each directory). Without `video_16_npy/` the frames are decoded from the raw clips in `$BL_HF_RAW`.

## Data reader (`common/data.py`)

```python
from baselines.common.data import get_export, resize_frames
ex = get_export()                      # reads $BL_EXPORT and $BL_HF_RAW
ex.sids("train" | "val" | "test")      # Gold lists of clip ids (6-digit strings); "test" is the evaluation set of every row
ex.frames(sid)                         # uint8 RGB [16,H,W,3]: protocol frames, content crop, native resolution
ex.raw_frames(sid, abs_idx)            # any other frame of the clip (same crop), e.g. ex.sampled_abs(sid)[t] + [-2..2]
ex.frames_1fps(sid)                    # indices (0..15) of the protocol frames nearest to each whole second
ex.tool_points(sid)                    # xy [16,3,2] in [0,1] of the content crop, vis [16,3]
ex.contact_points(sid)                 # xy [16,4,2], vis [16,4]  (4 tracks; the region is their convex hull)
ex.interval(sid)                       # (start, end, valid) on the window axis, 0 = first and 1 = last protocol frame
ex.tokens(sid)                         # TactusNet's frozen V-JEPA 2.1 tokens [8,24,24,1024] fp16 (token k = frames 2k, 2k+1)
ex.pair_id(sid), ex.tool_id(sid), ex.act_id(sid), ex.tip_index(sid)
ex.pairs, ex.pair_tool, ex.pair_act    # pair names "tool|action" in id order; pair id -> tool id, pair id -> action id
ex.task(sid), ex.source(sid)           # source video (the bootstrap unit) and source dataset (subsets)
resize_frames(frames, size)            # [T,H,W,3] -> [T,size,size,3], squashed (bilinear), like TactusNet's preprocessing
```

## Protocol every baseline follows

- Train on Gold train only, select every choice on Gold val, predict Gold test once.
- Inputs: the 16 protocol frames (uniform over the action window, end frames included). Image methods run per frame.
  Video methods take the 16-frame clip, or denser frames of the same window through `ex.raw_frames`.
- When, How and Where receive the true tool and action: one output group per pair (or per tool for tool points), the
  loss only on the true group, and the true group read at test. Classification receives no label.
- Horizontal flips are off for keypoint and contact methods (tip order matters). They are allowed for classification.
- Each method keeps its official configuration. The adaptations to TACTUS are listed in the family READMEs and
  written to each run's `info.json`.

## Trainer contract

`$BL_PY -m baselines.<family>.<module> --out $BL_OUT/<name> [--smoke] [...]`

- `--smoke`: a few steps on tiny batches, to prove the code path end to end.
- Resumes from `<out>/last.pt` when it exists. Keeps `<out>/best.pt` by the val metric
  (`common/train_utils.Checkpointer`).
- Writes `<out>/log.jsonl`, `<out>/pred_val.npz`, `<out>/pred_test.npz` (`common/preds.save`), and `<out>/info.json`
  with the recipe actually used (frames, sampling, input size, initialization, frozen or fine-tuned, conditioning,
  epochs, batch, LR, optimizer).
- Pretrained weights are downloaded into `$BL_WEIGHTS` when missing, never into the repo.

Each family has a `run.sh` with step functions (`prep_<family>` on CPU, `train_<method>` on GPU). Source it and call
the steps, as its README shows.

## Scoring

```bash
$BL_PY -m baselines.common.score --pred <out>/pred_test.npz --out <out>/metrics_test.json
$BL_PY -m baselines.common.score --pred <out>/pred_val.npz --split val --no-boot
$BL_PY -m baselines.common.score --pred A.npz --compare B.npz          # paired bootstrap of A - B over test videos
```

- The evaluation set is always the full list of the split. A clip without a prediction is an error.
- The JSON holds the subsets `all`, `non_surgvu`, `source/<name>` and `pair/<tool|action>`. `all` and `non_surgvu`
  carry a 95% bootstrap CI over test videos (2,000 resamples). `--no-boot` skips it.
- `--exclude-tasks t1,t2` drops source videos. `cls/run.sh` uses it for Endo-FM, with
  `cholec80_13,cholec80_22`, the two test videos that were in its pretraining data.
- `--export` and `--hf-raw` override `$BL_EXPORT` and `$BL_HF_RAW`.

Floors (no learning beyond training-set statistics):

```bash
$BL_PY -m baselines.common.floors --which all --out $BL_OUT [--tip-pred <kpt file with tool_xy>]
```

`common/tactus.py` converts TactusNet's own dumps (an experiment directory of `training/unitap`) into a `kpt`
prediction file, so TactusNet is scored by the same `score.py`.

## Add your own baseline

Any method can be scored if it writes the prediction files below. You do not need the trainers of this directory.

### 1. Read the clips

Use `get_export()` and the reader calls listed above. Train on `ex.sids("train")`, select on `ex.sids("val")`, and
predict `ex.sids("test")` once. A clip is identified by its `sid`.

### 2. Predict

| Task | What to predict | Label input allowed |
|---|---|---|
| Which / What | scores over the 6 pairs, in the id order of `ex.pairs` | none |
| When | `start` and `end` on the window axis: 0 = first protocol frame, 1 = last | true tool and action |
| How | `(x, y)` of the 3 tool points on each of the 16 protocol frames | true tool and action |
| Where | 4 contact points on each of the 16 protocol frames, or a region mask per frame | true tool and action |

You may predict any subset of the tasks.

### 3. Write the prediction files (`common/preds.py`)

One file per method, split and kind: `preds.save(path, kind, sids, info, **arrays)`. `N` is the number of clips in
the file, in the order of `sids`.

| Kind | Array | Shape | Required | Meaning |
|---|---|---|---|---|
| `cls` | `pair_logits` | `[N,6]` | yes | scores over the 6 pairs; the scorer takes the argmax |
| `when` | `start` | `[N]` | yes | interval start on the window axis |
| `when` | `end` | `[N]` | yes | interval end on the window axis |
| `when` | `fallback` | `[N]` bool | no | marks clips where the method fell back to the whole window; reported as `fallback_rate` |
| `kpt` | `tool_xy` | `[N,16,3,2]` | one of the three | tool points, `(x, y)` in `[0,1]` of the content crop (How) |
| `kpt` | `ct_xy` | `[N,16,4,2]` | one of the three | contact points, same coordinates; the region is their convex hull (Where) |
| `kpt` | `ct_frame_dice` | `[N,16]` | one of the three | for mask methods: per-frame Dice, NaN on frames that are not scored (Where) |

Rules checked by `preds.save`:

- A `cls` file needs `pair_logits`. A `when` file needs `start` and `end`. A `kpt` file needs at least one of
  `tool_xy`, `ct_xy`, `ct_frame_dice`. Unknown array names are refused.
- Every array has one row per entry of `sids`.
- Every file also stores `kind`, `sids` and a JSON `info` dict (free form: method, split, input, conditioning, ...).
- All arrays are stored as float32, `fallback` as bool.

What `score.py` does with them:

- Rows are matched by `sid`, so the order is free and extra clips are ignored. Every clip of the split must be there.
- `cls`: Which and What are read from the predicted pair through `ex.pair_tool` and `ex.pair_act`. There is no
  separate tool or action output.
- `when`: `start` and `end` are clipped to `[0, 1]` and `end` is raised to `start` if it is smaller. Only clips with
  a valid ground-truth interval are scored.
- `kpt`, How: a file with `tool_xy` gets a `how` block. Errors are measured on the visible ground-truth points only,
  but every frame needs a prediction for all 3 points. There is no predicted visibility.
- `kpt`, Where: a file with `ct_xy` or `ct_frame_dice` gets a `where` block. If both are present, `ct_xy` is used.
  All 4 predicted points enter the hull. The ground-truth region is the hull of the visible human points, and only
  frames with at least 2 of them are scored.

Mask methods compute `ct_frame_dice` themselves with the scorer's own function, on boolean 384 x 384 masks (the
content crop squashed to the metric grid), as `where/segformer.py` does:

```python
from baselines.common.metrics import contact_frame_dice
gt_xy, gt_vis = ex.contact_points(sid)
dice = contact_frame_dice(gt_xy, gt_vis, pred_masks=masks)   # masks: bool [16,384,384] -> dice [16], NaN where not scored
```

A minimal sketch. The three `my_*` calls are yours; everything else exists in this directory:

```python
import numpy as np
from baselines.common import config, preds
from baselines.common.data import get_export, resize_frames

ex = get_export()                                   # reads $BL_EXPORT and $BL_HF_RAW
sids = ex.sids("test"); n = len(sids)
pair_logits = np.zeros((n, 6), np.float32)
start, end = np.zeros(n, np.float32), np.ones(n, np.float32)
tool_xy = np.zeros((n, 16, 3, 2), np.float32)
ct_xy = np.zeros((n, 16, 4, 2), np.float32)
for i, sid in enumerate(sids):
    frames = resize_frames(ex.frames(sid), 224)     # uint8 RGB [16,224,224,3]
    pair, tool = ex.pair_id(sid), ex.tool_id(sid)   # true labels: for When / How / Where only
    # pair_logits[i] = my_classifier(frames)                       # no label input
    # start[i], end[i] = my_localizer(frames, pair)                # window axis, 0..1
    # tool_xy[i], ct_xy[i] = my_point_model(frames, tool, pair)    # x, y in [0,1] of the frame

out = config.OUT / "my_method"                      # $BL_OUT/my_method
info = {"method": "my_method", "split": "test"}
preds.save(out / "cls/pred_test.npz", "cls", sids, info, pair_logits=pair_logits)
preds.save(out / "when/pred_test.npz", "when", sids, info, start=start, end=end)
preds.save(out / "kpt/pred_test.npz", "kpt", sids, info, tool_xy=tool_xy, ct_xy=ct_xy)
```

### 4. Score and compare

```bash
$BL_PY -m baselines.common.score --pred $BL_OUT/my_method/kpt/pred_test.npz --out $BL_OUT/my_method/kpt/metrics_test.json
$BL_PY -m baselines.common.score --pred $BL_OUT/my_method/kpt/pred_test.npz --compare $BL_OUT/kpt_hrnet/pred_test.npz
```

Keys of the `all` subset:

| Kind | Keys |
|---|---|
| `cls` | `which`, `what`, `pair`, `pair_macc`, `per_pair_acc` |
| `when` | `tiou`, `r3`, `r5`, `r7`, and `fallback_rate` if the file has `fallback` |
| `kpt` | `how`: `ade`, `gt32`, `pck4`, `pck8`, `pck16`, `median_px`; `where`: `dice`, `dice_ge50` |

`--compare` needs two files of the same kind. For every metric both files have, it prints `diff` (first file minus
second), `ci95` and `n_units`. The bootstrap resamples the test videos the two files share, 2,000 times, and
recomputes both rows on the same resampled videos. A CI that excludes 0 marks a difference. For `how_ade` and
`how_gt32` lower is better, so a negative `diff` favours the first file. `--compare` always uses the full split, and
`--exclude-tasks` does not apply to it.

### Conventions that commonly go wrong

- **RGB or BGR.** `ex.frames` and `ex.raw_frames` return RGB. The export stores BGR; pass `rgb=False` to get it.
  Apply your backbone's own mean and std after that.
- **Coordinates.** Points are `(x, y)`, normalized to the content crop: `x` by its width, `y` by its height. The
  content crop is what `ex.frames` returns, not the full video frame. `resize_frames` squashes each axis, so a point
  normalized in the resized frame is already in the right coordinates. If you keep the aspect ratio, pad or crop,
  undo that before saving.
- **The 384 scaling.** Every spatial metric multiplies normalized coordinates by 384 in both axes
  (`config.METRIC_SIZE`). Pixel errors are therefore errors in the squashed 384 x 384 frame, not in native pixels.
  Masks for `contact_frame_dice` must be 384 x 384.
- **Tool-point order.** Hook: anchor, anchor, tip. Grasper, scissors and clipper: anchor, tip, tip.
  `ex.tip_index(sid)` gives the tip indices (`[2]` or `[1, 2]`). How is scored point by point, so the order of your
  outputs must match. A horizontal flip changes which tip is which; the baselines train without it.
- **Contact-point order.** Where is the Dice of the hull, so the order of the 4 predicted points does not matter at
  test. It matters for training targets when the 4 slots are fixed output channels: the export's slot order has no
  fixed spatial meaning across clips. `where/canonical_order.py` gives one permutation per clip that sorts the slots
  by angle (`perm = canonical_export(ex, sid)`, then `apply(perm, *ex.contact_points(sid))`). The keypoint baselines
  use it by default.
- **True tool and action for When, How and Where.** The main comparison gives these tasks the true tool and action.
  Use them as an input or to select an output group, and say so in `info`. A method that does not use them is still
  scored the same way.
- **Which and What come from the pair.** Output 6 pair scores in the id order of `ex.pairs` (`labels.json`,
  `combo2id_6class`). Do not write separate tool and action predictions.
- **The window axis.** `start` and `end` are fractions of the span between the first and the last protocol frame,
  not seconds and not frame indices. Frame `t` of the 16 sits at `t / 15`.
- **Test once.** Select checkpoints and thresholds on val. `score.py` reads the split named by `--split`
  (default `test`).

## Checks

| Command | What it checks |
|---|---|
| `$BL_PY -m baselines.tests.test_common [--exp $BL_SMOKE_EXP]` | the export reads; `raw_frames` decodes the exported pixels; floors, prediction files, subsets and the paired bootstrap run. With `--exp` (a TactusNet experiment directory): `score.py` reproduces the How and Where numbers of the training code's own evaluation |
| `$BL_PY -m baselines.cls.tests.test_cls` | CPU checks of the classification harness |
| `$BL_PY -m baselines.when.tests.test_when` | CPU checks of the When methods |
| `$BL_PY -m baselines.kpt.tests.test_kpt [--weights]` | CPU checks of the keypoint transforms, codecs and weight loading |
| `$BL_PY -m baselines.where.tests.test_where [--exp DIR] [--preds $BL_OUT]` | mask path of the metric, canonical order, hindsight floor, Where files |

## Running everything (`run_baselines.sh`)

`run_baselines.sh` is the orchestration that produced the paper's tables on a multi-GPU machine: one job queue, one
lane per GPU, CPU preparation in the background, results copied to `$GV` every 15 minutes. It does not use
`env.example.sh`; it sets the `BL_*` variables from `$GV` and `$WORK`. It expects packed inputs on `$GV` (a code
archive, raw-clip tars, the export archive, token shards, the V-JEPA weight and TactusNet experiment state) that are
not part of this release. To reproduce single rows, source a family's `run.sh` and call its steps instead.

## Per-family docs

Each family directory has a `README.md` (commands, recipe table, deviations from the official configuration, GPU-hour
estimate for one RTX 4090), a `THIRD_PARTY.md` (vendored code: source URL, commit or version, license) and a `run.sh`
(step functions).
