# Contact supervision

The contact decoder of the Spatial Head predicts four points per sampled frame on the zoomed crop. Their convex hull is the predicted contact region. The released recipes train the same decoder on either tip-derived targets or human tissue regions. Within a given experiment, both use the same training flags, zoomed frames, and cached tokens. The dataset root selects the target.

**TactusNet uses tip-derived targets by default** (called *hindsight* targets in recipe `C006`). `run_exp.sh` trains this option, and `unitap/tools/infer_pipeline.py` loads its checkpoint by default.

The benchmark's Where metric compares the direct predictions on sampled frames with human-annotated tissue regions. [Evaluation protocol](../evaluation/README.md)

## Reported contact Dice

| Evaluation | Training target | Dice against human tissue regions |
| --- | --- | ---: |
| Gold benchmark, 1,523 test clips from 11 held-out videos (Table 3) | Tip-derived, default Gold-trained TactusNet | 0.242 |
| Earlier SurgVU development split, 417 held-out clips (Table S8) | Human regions, with 5,082 additional Gold clips from other sources | 0.421 |
| Same development split | Human regions, SurgVU training clips only | 0.419 |
| Same development split | Tip-derived, SurgVU training clips only | 0.232 |

The earlier split used distinct time segments of the same 32 SurgVU videos, yielding 2,692 training, 419 validation, and 417 test clips. During development, some predictions from the human-target decoder enclosed only part of the instrument's activity. The tip-derived target incorporates later visible tip positions to encourage coverage from a single prompt frame before CoTracker3 propagation. The 0.232 result comes from the variant trained on the SurgVU training segments alone. The contact decoder used to generate Silver annotations also trained on 5,082 Gold clips outside SurgVU (Supplement B.6, B.7, and C.3).

## Side by side

| | Hindsight targets (default) | Human-region targets |
| --- | --- | --- |
| Recipe | `unitap/experiments/runs/C006/run.sh` | `unitap/experiments/runs/C004/run.sh` |
| Dataset root | `data/case32/v1_zoom_hs` | `data/case32/v1_zoom_ct` |
| Target | The region the instrument tips visit during the interaction window, from the current frame onward. | The four tissue points placed by annotators and tracked through the clip. |
| Built by | `unitap/tools/build_hindsight_labels.py` | `pipeline/make_ct_root.py` |
| Region target of the Dice loss | Precomputed soft mask `region_mask/<sid>.npy`, shape `[16, 48, 48]`. | Built in the loss from the visible human points, on frames with at least 3 of them. |
| Point target | Four pseudo-corners derived from the mean and spread of the visible tip positions. | The four human points. |
| Trained by `run_exp.sh` | Yes, step `contact`. | No. Use `baselines/where/tactus_human.sh` or the manual commands below. |
| Checkpoint | `experiments/runs/C006/out/checkpoints/best.pt` | `experiments/runs/C006H/out/checkpoints/best.pt` (script) or `experiments/runs/C004/out/checkpoints/best.pt` (manual) |

Paths are relative to an experiment directory, `$WORK/exp/<E>`. Both roots are built by the `labels` step of `run_exp.sh`, so both exist after any experiment has passed that step.

Shared training settings, from the recipes: four output points (`--num-tool-points 4`), a point loss that is invariant to the order of the four points within a clip (`--traj-clip-permutation-invariant`), and a soft region Dice loss on a 48 × 48 grid (`--traj-region-dice-weight 60 --traj-region-dice-side 48 --traj-region-dice-tau 0.5`). In both roots the contact targets are stored in the trainer's tool-point slots (`gt_tool_xy`, `gt_tool_vis`), normalised to the zoom box.

## Hindsight targets (C006)

### What the target is

For each of the 16 sampled frames, the target is where the instrument tips go from that frame until the end of the interaction window.

`build_hindsight_labels.py` does the following for each clip:

1. Read the annotated tool points of type `tool_tip` on every annotated frame. Average the visible tips of a frame to one position.
2. Keep the frames inside the interaction window (`action_start_abs` to `action_end_abs`).
3. For sampled frame `t`, take the tip positions at window frames `t' >= t`. If there are none, take all window frames.
4. Map the positions into the clip's zoom box.
5. Draw a Gaussian blob at each position (sigma 14 px at the 384 px frame scale) and take the maximum. This is the soft mask on a 48 × 48 grid.
6. Derive four pseudo-corners: the centroid of the positions plus and minus 1.5 standard deviations along the two principal axes of their spread (the blob width is added to the spread).

Tissue is treated as static within a clip: tip positions from later frames are used in the coordinates of the current frame without motion compensation. Clips with no visible tip inside the interaction window get empty masks and no visible points, so they carry no contact target.

### Build the labels

Inside an experiment directory that has the zoom dataset (`data/case32/v1_zoom`):

```bash
source "$WORK/code/pipeline/env.sh" && cd "$WORK/exp/A"
"$PY" "$X/make_ct_root.py" && "$PY" tools/build_hindsight_labels.py
```

This is the `labels` step of `run_exp.sh`. `make_ct_root.py` must run first, because the hindsight root links its frames and tokens through `v1_zoom_ct`. The input and output roots can be changed with the environment variables `HS_SRC`, `HS_RAW`, `HS_ZOOM`, `HS_CT` and `HS_DST`.

### Train

As part of a full experiment:

```bash
bash "$WORK/code/pipeline/run_exp.sh" A
```

The decoder alone, after the steps up to `labels` have run:

```bash
source "$WORK/code/pipeline/env.sh" && cd "$WORK/exp/A"
export PYTHONPATH="$PWD/third_party/t2b"
EXTRA_ARGS="--num-workers 8 --batch-size 24" bash "$X/recipe.sh" C006 0
```

The last argument is the GPU index. Use `--batch-size 32` on a card with more than 30 GB.

### Dump and evaluate

```bash
mkdir -p results
for s in val test; do
  "$PY" tools/dump_val_predictions.py --checkpoint experiments/runs/C006/out/checkpoints/best.pt \
      --out experiments/analysis/pred_C006_best_$s --split $s --device cuda --batch-size 16
  "$PY" tools/eval_contact_stage.py --dump experiments/analysis/pred_C006_best_$s --save results/contact_human_$s.json
  "$PY" tools/eval_hindsight.py     --dump experiments/analysis/pred_C006_best_$s --save results/contact_hindsight_$s.json
done
```

These are the calls made by the `contact` and `evaluate` steps of `run_exp.sh`.

## Human-region targets (C004)

### What the target is

The target is the human-annotated contact region: four tissue points per clip, tracked over the action window. The export keeps the four tracks that are visible on the most sampled frames. `build_zoom_dataset.py` maps them into the zoom box and marks points outside the box as not visible.

`make_ct_root.py` builds `data/case32/v1_zoom_ct` from `data/case32/v1_zoom` with links only. It puts the contact tracks (`gt_ct_xy`, `gt_ct_vis`) in the tool-point slots and marks all four points with the same role. There is no `region_mask/` in this root, so the trainer builds the Dice target from the visible human points of each frame.

### Train with the script

`baselines/where/tactus_human.sh` retrains only the contact decoder on human targets, starting from a finished experiment directory. The example below uses experiment A's video-level Gold benchmark split. It takes the `C006` recipe, changes the dataset root to `v1_zoom_ct`, checks that this is the only changed line, trains under the run id `C006H`, dumps val and test, and runs both evaluations.

From the repository root:

```bash
U="$WORK/code/unitap" X="$WORK/code/pipeline" PY=python \
VJEPA_CKPT="$GV/weights/vjepa2_1_vitl_dist_vitG_384.pt" \
bash baselines/where/tactus_human.sh "$WORK/exp/A" /path/to/out
```

Requirements and outputs:

- The experiment must have passed the `labels` step and must have its zoom tokens.
- `GPU` selects the GPU (default 0). `KB` sets the batch (default 24 on a card with 30 GB or less). `SMOKE=1` runs one epoch at a small batch.
- Results: `results/contact_human_C006H_{val,test}.json` and `results/contact_hindsight_C006H_{val,test}.json` in the experiment directory.
- Its last step converts the dumps to benchmark prediction files (`pred_{val,test}.npz` in the output directory). That step uses the `baselines` package and its environment; see [`../baselines/where/README.md`](../baselines/where/README.md).

### Train manually with the C004 recipe

`C004/run.sh` is the same recipe with the dataset root already set to `v1_zoom_ct`. `run_exp.sh` has no step for it. The calls below are the ones `run_exp.sh` makes for `C006`, with the run id changed.

```bash
source "$WORK/code/pipeline/env.sh" && cd "$WORK/exp/A"
export PYTHONPATH="$PWD/third_party/t2b"
"$PY" "$X/make_ct_root.py"                                             # only if the labels step has not run
EXTRA_ARGS="--num-workers 8 --batch-size 24" bash "$X/recipe.sh" C004 0

mkdir -p results
for s in val test; do
  "$PY" tools/dump_val_predictions.py --checkpoint experiments/runs/C004/out/checkpoints/best.pt \
      --out experiments/analysis/pred_C004_best_$s --split $s --device cuda --batch-size 16
  "$PY" tools/eval_contact_stage.py --dump experiments/analysis/pred_C004_best_$s --save results/contact_human_C004_$s.json
  "$PY" tools/eval_hindsight.py     --dump experiments/analysis/pred_C004_best_$s --save results/contact_hindsight_C004_$s.json
done
```

## Evaluation

Both evaluation tools read a prediction dump and run from the experiment directory. Either tool can score a decoder trained on either target.

| Tool | Compared with | What it reports |
| --- | --- | --- |
| `tools/eval_contact_stage.py` | The human contact points of the original export (`data/case32/v1/gt_ct_xy`, `gt_ct_vis`). Predictions are mapped from the zoom box back to the frame. | Region overlap between the hull of the four predicted points and the hull of the visible human points, on frames with at least 2 visible human points (`region_hull_dice`, `region_hull_iou`). Point errors under the best per-clip permutation and under per-frame matching. |
| `tools/eval_hindsight.py` | The hindsight masks (`data/case32/v1_zoom_hs/region_mask`). | Per-frame Dice between the hull of the four predicted points and the mask thresholded at 0.5, on frames with a non-empty mask. Share of frames whose predicted centroid lies inside the mask. |

`region_hull_dice` of `eval_contact_stage.py` is the quantity the benchmark's Where metric measures. For benchmark numbers, convert the dumps to a prediction file and score it with `baselines.common.score`, as described in [`../evaluation/README.md`](../evaluation/README.md) and [`../baselines/README.md`](../baselines/README.md).

## Using a contact checkpoint at inference

The Spatial Head tools load `experiments/runs/C006/out/checkpoints/best.pt` by default. To use another contact decoder, from the experiment directory:

```bash
python tools/infer_pipeline.py --split test --contact-ckpt experiments/runs/C004/out/checkpoints/best.pt --dump-dir results/pipeline_test_C004 --device cuda
```

In the architecture package, pass `ckpts={"contact": "<path>"}` to `UniTAPSpatial`.
