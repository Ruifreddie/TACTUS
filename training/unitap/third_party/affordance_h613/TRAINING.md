# H613 Training and Evaluation Protocol

## Exact Configuration and Code

`configs/h613_train.json` contains all 504 saved H613 arguments. `PARAMETERS.md`
lists every value, including disabled experimental options. Only four runtime
path fields were replaced with portable paths/null. `affordance/training_core.py`
is the complete original affordance trainer, including the dataset reader,
model constructors, geometry utilities, losses, corruption, AdamW loop, EMA,
checkpoint selection and evaluation. Its hash is in PROVENANCE.json.

The standalone launcher bypasses the historical chain of shell-script defaults
by supplying the recorded argument namespace directly. It adds preflight data
checks and refuses to overwrite a populated run directory. Internal alternative
region-model branches remain in the original trainer; the H613 config selects
`proposal_point_temporal`. These are region models, not classification or tool
keypoint task decoders. Changes to the JSON are new experiments, not exact replay.

## Backbone and Targets

Use pretrained frozen V-JEPA2.1 ViT-L/384, 16 input frames, patch size 16,
tubelet size 2, final-layer tokens [8,24,24,1024]. See README.md for RGB/BGR,
resize, normalization, version and backbone patch. No encoder gradients or
encoder optimizer state occur in region-head training: it reads token caches.

The target is 16 frames of four all-visible contact/operation-region points,
normalized xy coordinates. The dataset reader sorts each quad by angle around
its centroid and rotates the first vertex to minimum x+y. AABB targets are
derived from those four points with bbox_pad_ratio=0 and min_box_size=0.
The principal output is a free quadrilateral, not a tool keypoint skeleton.

H613 uses no RGB sidecar, tool mask, tool trajectory, teacher, checkpoint
warmstart or materialized data augmentation. It DOES use token corruption,
condition dropout and noisy-query supervision during training.

Action-range features are zeroed before model input (`use_role_features=false`).
The original inside_action_range labels ARE used in the loss because
`use_role_loss_weighting=true`: active frames weight 1.5, inactive frames 0.75.
Do not omit the action-range files when reproducing H613. The portable launcher
requires them; the original reader's silent missing-file fallback is not replay.

## Data Contract

Supply your own authorized dataset in this layout; no real IDs or samples are
included in this code package:

```text
data/
  meta.json
  train.txt
  val.txt
  gt_ct_xy/<sample>.npy
  action_range_16/<sample>.npz
  vjepa_token_cache/train/<sample>.npy
  vjepa_token_cache/val/<sample>.npy
```

meta.json is a JSON list of records with `sid`, `case`, `tool_id`, `act_id`,
and `combo_id`. See README.md for the six label combinations. Split files
contain one sample ID per line. Points are float32 [16,4,2]; token files are
float16/float32 [8,24,24,1024], cast to float32 by the reader. Action NPZ files
contain `features` float32 [16,4] and `inside_action_range` [16] (0/1).
The point/token/action arrays must refer to the same 16 sampled frames and
the same image coordinate system. Use all-visible point targets; the trainer
does not implement a missing-point visibility mask for this contract.

The historical H612 split has 5622 training clips/46 cases and 1471 validation
clips/28 cases, case-disjoint. Validation clips were easy-selected using H342
predictions from a held-out case pool. Counts alone do not recreate this split.
Exact replay requires the original split and sampling/preparation artifacts,
which are intentionally not bundled. Results on other splits are not comparable
to the historical H613 number. `--allow-custom-data` permits other counts while
still checking sample/case disjointness and required files.

## Prepare Features

Provide uint8 frame arrays [16,H,W,3], already sampled in temporal order.
The new cache utility never deletes source frames and refuses to overwrite a
token file. It uses the same preprocessing/final-layer adapter as inference.

```bash
python cache_features.py --vjepa-repo vjepa2 --frames-root frames \
  --split-file data/train.txt --output-dir data/vjepa_token_cache/train \
  --color-order rgb --device cuda
python cache_features.py --vjepa-repo vjepa2 --frames-root frames \
  --split-file data/val.txt --output-dir data/vjepa_token_cache/val \
  --color-order rgb --device cuda
```

For OpenCV BGR arrays specify `--color-order bgr`. This utility does not choose
frames, derive annotations, or recreate the original easy-validation selection.

## Optimization

| Setting | H613 value |
|---|---:|
| Epochs | 9 |
| Batch size / workers | 4 / 2 |
| Seed | 161 |
| Optimizer | AdamW, default betas (0.9,0.999), eps 1e-8 |
| Learning rate | 0.000075, constant (no scheduler) |
| Weight decay | 0.0028 |
| Gradient norm clipping | 1.0 |
| AMP | enabled on CUDA, GradScaler |
| EMA decay | 0.994 after each optimizer step |
| Hidden dimension / decoder layers | 176 / 2 |
| Proposal count | 8 base proposals, plus enabled corner-heatmap source |
| Selection | soft, temperature 0.75, topk=0 |

Training shuffles samples, does not drop the final batch, and has no gradient
accumulation, LR warmup, LR decay or distributed training in this loop. Seed
initialization covers Python, NumPy and torch; this is not a guarantee of
bitwise reproducibility across hardware, versions or CUDA kernels.

Regularization ramps by min(epoch/2,1): half strength in epoch 1, full strength
from epoch 2. Gaussian token noise std=0.075; token dropout=0.19; temporal token
dropout=0.14; block dropout=0.48 with 5x5 blocks. Dropped tokens use the learned
1024-D mask token. Condition dropout=0.26; condition ID dropout=0.55 uses null
IDs tool=8/action=8/combo=16. Clean-supervised pass weight=0.12 and detached
clean/corrupt consistency weight=0.36. The consistency term is SmoothL1 on
cxcywh plus 0.5 times SmoothL1 on quad coordinates.

## Losses

The exact reductions, matching targets and conditional gates are in
`compute_loss`, `forward_roi_model_for_batch`, and `main` of training_core.py.
Do not implement the loss by summing reported scalar metrics: some are
diagnostics, some are nested auxiliary terms and some options are disabled.

| Loss group | H613 coefficients/settings |
|---|---|
| Final box | SmoothL1 1.30, GIoU 1.65, CIoU 0.38 |
| Center / temporal box | center CE 0.06, smoothness 0.022 |
| Box geometry | area 0.10, size 0.30, aspect 0.10 |
| Quad points | point 1.80, pairwise shape 0.70, temporal 0.30 |
| Quad axis / topology | axis 0.70, axis-length subweight 0.28, topology 2.00, margin 0.0008 |
| Coarse / intermediate decoder | coarse 0.30, decoder auxiliary 0.22 |
| Corner heatmap | modified focal 0.12, sigma 1.25 |
| Proposal geometry | box best-of-K 0.36, quad best-of-K 0.90 |
| Proposal ranking / quality | rank CE 0.12, quality 0.12, varifocal 0.48 |
| Pairwise / soft rank | 0.16 (margin 0.2, IoU gap 0.03), 0.20 (tau 0.06) |
| Task-aligned assignment | 0.15, topk 3, alpha 1, beta 6 |
| PCL cluster | 0.06, topk 2, foreground IoU 0.30, background 0.08 |
| Noisy GT refiner queries | 0.16, groups 4, noise scale 0.75, minimum scale 0.25 |
| Small-box frame boost | 0.30, reference area 0.015, maximum scale 2.1 |

PCL `proposal_pcl_cluster_weight=0.06` and the separate
`proposal_pcl_cluster_ce_weight=0` are different settings; preserve both.
Query noise scale is 0.4; stage2_bridge_noise_mode is local_only. Denoising GT
queries provide training supervision; the public inference API does not require
GT. Proposal-oracle metrics are diagnostics, not deployable predictions.
Mask losses, teacher distillation, late candidate reranking and source4 eval
bias are disabled in H613. See PARAMETERS.md for every gate and coefficient.

## Run and Evaluate

```bash
python train.py --data-root data --out-dir runs/preflight --dry-run
python train.py --data-root data --out-dir runs/h613 --device cuda
python train.py --data-root data --out-dir runs/h613_eval \
  --checkpoint runs/h613/best.pt --device cuda
```

For a custom dataset add `--allow-custom-data`. CUDA is required by the saved
setting; pass `--device cpu` explicitly for a small CPU experiment. The wrapper
refuses to silently turn a full CUDA run into a CPU run. Optional command-line
overrides include epochs, batch size, workers and seed. Other modifications
belong in a copied JSON configuration. The effective configuration is saved
in the output directory's args.json.

Each epoch evaluates raw and EMA weights. best.pt is selected by EMA
`quad_iou_mean`; its `model` key already contains the chosen EMA state.
best_proposal_oracle.pt is selected using candidate oracle IoU and is a separate
diagnostic checkpoint. It should not replace best.pt for normal inference.
The loop writes history.json, best_metrics.json and best_proposal_oracle_metrics.json.
Eval-only writes eval_condition_ablation.json. Checkpoints contain model,
args, epoch and metrics, but not optimizer/scaler state: there is no exact
mid-training resume. `init_checkpoint` would warm-start weights only and is
disabled in this replay. Eval uses trusted local checkpoint deserialization
as in the original trainer; do not supply untrusted pickle checkpoints.

Quad IoU uses the original 64x64 grid approximation. `quad_iou_mean` averages
all frame IoUs; the loss-report `quad_iou` has different aggregation and must
not be substituted. The final head selected epoch 5 in the historical run.
The previously reported Dice value came from a separate analysis; this trainer
does not emit that Dice report. Training details and aggregate historical
context here do not imply the original validation samples are included.

## Verification and Limits

```bash
python smoke_train.py
python verify.py --checkpoint weights/h613_best.pt
```

smoke_train.py creates temporary synthetic features/targets, runs one training
epoch with backward, AdamW, EMA and checkpoint saving, then evaluates that
checkpoint. It cleans the temporary data on exit. This verifies the complete
code path without bundling or using real samples. Full nine-epoch retraining
and fresh video-backbone encoding are not performed during packaging.
