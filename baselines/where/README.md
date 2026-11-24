# Where: contact-region baselines and checks

This directory implements SegFormer for Table 3(c), along with two contact-supervision checks and a contact-slot ordering check. The Where metric compares each predicted region with the region formed from visible human-annotated contact points. Dice is averaged over frames with at least two visible points (Supplement D.2). For the earlier SurgVU development comparison in Supplement C.3, see the [contact-supervision guide](../../training/CONTACT_SUPERVISION.md).

| Method or diagnostic | Code | Prediction file (`$BL_OUT/...`) |
|---|---|---|
| SegFormer-B2 | `segformer.py` | `segformer/pred_{val,test}.npz`, `ct_frame_dice` (mask method) |
| TactusNet, human-region targets | `tactus_human.sh` + `tactus_human.py` | `tactus_human/pred_{val,test}.npz`, `ct_xy` |
| Hindsight target from the true tips | `hindsight_floor.py` | `hindsight_floor/pred_{val,test}.npz`, `ct_frame_dice` (+ `hindsight_floor_corners/`, `ct_xy`, for reference) |
| (for `kpt/`) contact-slot order | `check_order.py`, `canonical_order.py` | `where/check_order.json` |

## Commands

```bash
source baselines/env.example.sh      # or your edited copy; run_baselines.sh sets BL_* (and GV, WORK) itself
source baselines/where/run.sh

prep_where                           # CPU: pinned extras, MiT-B2 weights into $BL_WEIGHTS, slot-order check
floor_hindsight                      # CPU: hindsight floor (CHECK_EXP=<UniTAP exp> adds the agreement check)
restore_exp_A                        # CPU: rebuild $WORK/exp/A from the packed experiment state on $GV (see below)
train_segformer                      # GPU: SegFormer-B2, resumes from last.pt
train_tactus_human                   # GPU: retrain TactusNet's contact decoder on human regions in $WORK/exp/A
score_where                          # CPU: metrics_{test,val}.json of every Where file
$BL_PY -m baselines.where.tests.test_where [--exp $BL_SMOKE_EXP] [--preds $BL_OUT]    # checks
```

Single steps: `$BL_PY -m baselines.where.segformer --out $BL_OUT/segformer [--smoke] [--batch 16] [--max-iters N]`,
`$BL_PY -m baselines.where.hindsight_floor --out $BL_OUT [--check-exp EXP]`,
`U=<unitap> X=<pipeline> PY=python bash baselines/where/tactus_human.sh EXP [OUT]`,
`$BL_PY -m baselines.where.check_order`.

## Recipes

| Row | Input | Init | Schedule | Batch | LR | Optimizer | Targets |
|---|---|---|---|---|---|---|---|
| SegFormer-B2 | train: 1-fps protocol frames (`ex.frames_1fps`), official 512x512 crops; test: all 16 protocol frames at short side 512, true-pair logits upsampled to the 384x384 metric grid | encoder `nvidia/mit-b2@3bb39e8739149c3777d0325349b2a6c32c6413db` (ImageNet-1k); all-MLP decoder (768 ch) and 6-channel classifier random | official 160k iterations, poly (power 1), linear warmup 1500 (ratio 1e-6), 10 evaluations; `--max-iters` cuts it | 16 (official total) | 6e-5 x batch / 16, decoder x10 | AdamW (0.9, 0.999), wd 0.01, no decay on norms | metric region of the frame, BCE + soft Dice on the true pair's channel; frames with >= 2 visible points |
| TactusNet, human-region targets | frozen V-JEPA 2.1 ViT-L tokens of the 16 zoomed frames (384x384), zoom boxes of experiment A | contact decoder random, encoder frozen (as C006) | 80 epochs, best by val loss (C006 recipe) | 24 on a 24 GB card (A's batch; recipe 32) | 1e-4 constant | AdamW, wd 0.05, EMA 0.999, bf16 | the 4 human contact points per frame in the zoom box (outside -> invisible): permutation-invariant point loss + region Dice (weight 60) on the region of >= 3 visible points |
| Hindsight floor | true tool tips of every raw annotated frame of the interaction window | none | none | - | - | - | it is the target: UniTAP's hotspot, rasterized in the 384x384 content-crop grid |

### SegFormer: official configuration and the TACTUS adaptations

Official: NVlabs/SegFormer `local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py` (+ `ade20k_repeat.py`):
mit_b2 backbone (drop path 0.1) from `mit_b2.pth`, SegFormerHead (embed 768, dropout 0.1), train pipeline
Resize((2048, 512), ratio 0.5-2.0) -> RandomCrop(512, cat_max_ratio 0.75) -> RandomFlip -> PhotoMetricDistortion ->
Normalize -> Pad(512, seg 255), test at (2048, 512) keep ratio, AdamW 6e-5 / 0.01 with custom keys (norm and pos_block
no decay, head lr x10), poly power 1 with 1500 linear warmup iterations, 160k iterations at 8 x 2 images. Kept as is,
except (also in `info.json` "adaptations"):

- **init** (rule of the baselines: pretrained backbone only): encoder `nvidia/mit-b2` (ImageNet-1k, the official init too),
  decoder and classifier random. `info.json` "load_report" records 364 encoder tensors loaded, 16 decoder tensors new.
- **(a)** 6 sigmoid channels, one per pair, instead of 150 softmax classes; loss and readout on the true pair's channel;
  loss = binary CE + soft Dice (official softmax CE; `--loss bce` drops the Dice term); padding ignored as officially.
- **(b)** training images = Gold-train protocol frames at about 1 fps (`ex.frames_1fps`) with >= 2 visible contact
  points; the target is the metric's own region, resized with the image (nearest); testing uses all 16 protocol frames.
- **(c)** schedule: the official 160k iterations are kept by default; `--max-iters` shortens it (poly shape, warmup and
  10 evaluations kept), and `info.json` states which ("max_iters", "official_iters_kept", "passes_over_train_frames").
- **(d)** one GPU: `--batch` (default 16 = official total) with linear LR scaling; iterations scale by 16 / batch.
- no RandomFlip (TACTUS rule for contact methods); test logits upsampled to 384x384 before the 0.5 threshold and Dice;
  BatchNorm instead of SyncBN; bf16 autocast (`--no-amp` = fp32); best.pt = best val Dice over the 10 evaluations.

Frames are read only through `ex.frames(sid)` (training takes frame t of it), so without `video_16_npy` in the export
they are decoded from the raw clip. That reads 16 frames per training sample: keep `--workers` at 8 or more.

## GPU hours on one RTX 4090, 24 GB (estimates)

Estimates, scaled from timings on other GPUs; not measured on a 4090.

| Step | Basis | Estimate on one 4090 |
|---|---|---|
| `train_segformer`, official 160k iterations x 16 images | assumed 37-55 img/s on 512x512 crops; batch 16 needs about 18 GiB | **13-19 h** (2.56M images) + about 0.5 h for 10 val passes; `--max-iters 80000`: 7-10 h; `--max-iters 40000`: 3.5-5 h. The trainer prints s/it every 50 iterations, so the first minutes give the exact figure |
| `train_tactus_human` (C006 recipe, 80 epochs over about 12.4k Gold-train clips, batch 24) | assumed about 0.030 s per clip-step | **about 8-9 h** (990k clip-steps + 80 val passes) |
| `floor_hindsight`, `prep_where`, `score_where` | CPU, about 0.1 s per clip | minutes, no GPU |
| `restore_exp_A` | state.tar (about 4 GB: seven best.pt), zoom-token shards (9.4 MB per clip, about 145 GB on `$WORK`), labels (CPU) | no GPU (unless zoom tokens are missing), about 20-40 min of reads from `$GV` |

## TactusNet, human-region targets

Experiment A is TactusNet trained on Gold train (`training/pipeline/README.md`). UniTAP is the working name of the
Spatial Head code, `training/unitap` (`$U`); `pipeline/` below is `training/pipeline` (`$X`). Experiment A trains the
contact decoder with `pipeline/run_exp.sh` step `contact`:
`EXTRA_ARGS="--num-workers 8 --batch-size $KB" bash recipe.sh C006`, where `recipe.sh` copies UniTAP
`experiments/runs/C006/run.sh` with machine lines edited (WS, PY, no sudo, no CPU pinning, V-JEPA checkpoint path), then
`tools/dump_val_predictions.py` writes `experiments/analysis/pred_C006_best_{val,test}`. C006 trains on
`data/case32/v1_zoom_hs` (hindsight targets: `region_mask/` + pseudo-corners, `tools/build_hindsight_labels.py`). The
human-target switch is the data root `data/case32/v1_zoom_ct` (`pipeline/make_ct_root.py`: the human contact
points in the tool slots, same zoom frames and tokens), which UniTAP used in C004 (the same flags as C006). Without `region_mask/` the trainer's region Dice falls back to the human
region (`kpt_combo/loss.py`, `_build_region_cell_targets`). So the whole change is one line of the recipe:

```diff
-T2B="$WS/third_party/t2b"; DATA="$WS/data/case32/v1_zoom_hs"; RUN="$WS/experiments/runs/C006"; OUT="${OUT_DIR:-$RUN/out}"
+T2B="$WS/third_party/t2b"; DATA="$WS/data/case32/v1_zoom_ct"; RUN="$WS/experiments/runs/C006H"; OUT="${OUT_DIR:-$RUN/out}"
```

`tactus_human.sh` generates it with `recipe.sh`'s own sed lines plus this substitution, refuses to run if the diff
against C006 is anything else, trains, dumps `pred_C006H_best_{val,test}`, runs UniTAP's `eval_contact_stage.py` and
`eval_hindsight.py`, and converts with `common.tactus.load_contact` into `$BL_OUT/tactus_human/pred_{val,test}.npz`
(checking that score.py reproduces UniTAP's `region_hull_dice`). Copies of the recipe, diff, log (YouTube task names
masked), UniTAP results and `best.pt` go to `$BL_OUT/tactus_human/unitap/`. The recipe keeps best.pt only, so a killed
training restarts from scratch.

**What `restore_exp_A` needs.** `train_tactus_human` runs inside a finished experiment A. `restore_exp_A` rebuilds
one from packed state on `$GV`; that state (`code.tar`, `state/A/state.tar`, `tokens/zoom_A`) is not part of this
release, so experiment A has to be trained first (`training/pipeline`) and its directory passed to
`train_tactus_human`. The restore runs the same commands as `run_exp.sh A` on a new machine, minus training and
minus writing back to `$GV`: the code bundle `$GV/code.tar` (unitap/, pipeline/), `$WORK/export` and
`$WORK/hf` (raw annotations) on the local disk; `make_exp.py A`; `persist.py load A` from `$GV/state/A/state.tar` (step
markers, `experiments/analysis` dumps including the stage-1 dumps `pred_E007k_best_{val,test}`, results, every best.pt);
the zoom boxes rebuilt from those stage-1 dumps (`tools/build_zoom_dataset.py --no-video`, deterministic, seed 0); the
zoom tokens unpacked from `$GV/tokens/zoom_A`; the labels step (`make_ct_root.py` -> `v1_zoom_ct`,
`tools/build_hindsight_labels.py` -> `v1_zoom_hs`). It ends with a check: the restored C006 test dump, scored with the
rebuilt boxes, must give A's saved Dice (`results/contact_human_test.json`). Set `GV` to the storage directory
(default `/workspace`). Then `train_tactus_human` uses the V-JEPA weight `$VJEPA_CKPT` from `pipeline/env.sh`.

## Hindsight floor

`hindsight_floor.py` re-implements UniTAP `tools/build_hindsight_labels.py` (eq. hindsight): for sampled frame t, the
mean visible tip of every raw annotated frame of the interaction window at or after t (all visible-tip frames in the window if none remain),
Gaussian blobs of sigma 14 px at 384 along x (isotropic in native pixels, so 14 W / H px along y in the squashed grid),
hotspot = max > 0.5, rasterized in the 384x384 content-crop grid instead of the 48x48 zoom-box grid, then scored with
`contact_frame_dice(pred_masks=...)`. Clips without a visible tip in the interaction window get an empty hotspot
(UniTAP: empty mask, no loss), so their frames score 0. `--iso384` (isotropic 14 px at 384) and `--tip-source export`
(16 protocol frames only) are sensitivity variants.

`--check-exp EXP` (or `CHECK_EXP=EXP floor_hindsight`) compares the result with UniTAP's own labels in an experiment
directory that has `data/case32/v1_zoom_hs`: the same formula on UniTAP's box grid against its `region_mask`, the
384-grid hotspot against the box mask mapped through the zoom box, the hotspot area that falls outside the zoom box,
and the pseudo-corners against `gt_tool_xy`.

## Contact-slot order (for `baselines/kpt/`)

The export orders the 4 tracks by (first visible protocol frame, angle of the track mean around the mean of the 4
means), UniTAP `tools/export_case32.py`. When the tracks do not all start in the same frame, the first-visible key
overrides the angle, so a slot has no fixed spatial meaning across clips. `check_order.py` reproduces the export key
and measures the consistency of each rule on your export (`prep_where` runs it and writes
`$BL_OUT/where/check_order.json`).

`canonical_order.canonical(ct_xy, ct_vis, tool_xy, tool_vis) -> perm` (new slot j = export slot perm[j]) fixes this
with a ground-truth-only rule: drop the first-visible key and sort the slots by the angle of their mean position (over
their visible frames) around the centre of the slot means, clockwise on screen starting from the image's left direction
(ascending atan2 from -pi, y down); never-visible slots go last. Slot 0 is then usually the up-left corner, 1 up-right,
2 down-right, 3 down-left. It equals the export order whenever the tracks start together. Two other rules are kept for
`check_order.py`: `image_up` (cut at the up direction) and `tool` (cut at the backward tool direction). One
permutation per clip keeps the tracks whole (video methods), it is idempotent, and Dice is order-free, so evaluation
does not change. Usage:
`perm = canonical_order.canonical_export(ex, sid); ct_xy, ct_vis = canonical_order.apply(perm, *ex.contact_points(sid))`.

## Files

`segformer.py`, `tactus_human.sh`, `tactus_human.py` (copy-exp, convert), `hindsight_floor.py`, `check_order.py`,
`canonical_order.py`, `run.sh`, `tests/test_where.py`, `requirements.txt` (pinned extras), `THIRD_PARTY.md`.
