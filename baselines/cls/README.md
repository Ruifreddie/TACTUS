# Classification baselines: Which, What, Pair (Tab. 3 a)

Five video classifiers (TSN, TSM, VideoMAE, X-CLIP, Endo-FM) and the reserve MViTv2-S, trained by one harness
(`train.py`). Rule: **every method keeps its official configuration** (architecture, input size,
optimizer, LR schedule shape, augmentation, loss); only the four TACTUS adaptations below differ, and each run writes
them to `info.json` (`official_recipe`, `adaptations`) for the supplement table.

| | TACTUS adaptation (the only changes) |
|---|---|
| (a) output | 6 pair classes (labels.json `combo2id_6class`); a new head trained from scratch (X-CLIP: 6 fixed prompts). Which and What are read from the predicted pair. |
| (b) input | the 16 protocol frames of the clip; each frame is the whole content crop squashed to a square. The official train augmentation runs on that square at the official pre-crop size (256, or 320 for the 256-320 jitter); the test view is the whole frame at the official input size 224 x 224 instead of the official short-side resize + center crop, which would cut tool tips at the borders. Single view. Endo-FM and the TSM fine-tuning example officially use 8 frames. |
| (c) schedule length | the official number of epochs and schedule shape on the Gold train clips (iterations scale with the data size). |
| (d) batch | effective batch 64 clips per optimizer step by gradient accumulation on one GPU; the official LR scaled linearly from the official batch. |

Also: bf16 autocast (the originals use fp32 or fp16 AMP); horizontal flip stays on for VideoMAE and MViTv2, whose
SSv2 recipes switch it off only because SSv2 classes are left/right sensitive (their code keeps it for other datasets).

## Commands

```bash
source baselines/env.example.sh                  # or your edited copy; run_baselines.sh sets the same variables itself
$BL_PY -m pip install -r baselines/cls/requirements.txt
source baselines/cls/run.sh
prep_cls                                         # CPU: weights -> $BL_WEIGHTS/cls, frames -> $BL_OUT/cls_cache
bench_cls videomae                               # optional, ~2 min: throughput and memory on this GPU
train_tsn; train_tsm; train_videomae; train_xclip; train_endofm   # GPU, one after the other; each resumable
train_mvitv2                                     # reserve row only
```

Each `train_<m>` runs `python -m baselines.cls.train --model <m> --out $BL_OUT/cls/<m>` and then
`python -m baselines.common.score --pred $BL_OUT/cls/<m>/pred_test.npz --out $BL_OUT/cls/<m>/metrics_test.json`
(Endo-FM also `metrics_test_excl_pretrain_videos.json`, without the two test videos in its pretraining data).
A run directory holds `best.pt` (model, best val Pair accuracy), `last.pt` (resume), `run.json`, `log.jsonl`,
`pred_val.npz`, `pred_test.npz`, `info.json`, `metrics_*.json`. Rerunning a command resumes from `last.pt`; after the last
epoch it only re-predicts and re-scores. Gold train only, selection on Gold val, one prediction on Gold test (no Silver
run: the Gold + Silver comparison is a TactusNet ablation). Useful flags: `--batch` (micro-batch), `--grad-ckpt`,
`--workers`, `--stop-after N` (stop after N epochs, to test resume or split a run).

Checks: `$BL_PY -m baselines.cls.tests.test_cls` (CPU: cache, augmentations, mixing, LR landmarks) and
`smoke_cls` (GPU: every model trains 2 epochs x 3 steps with `--smoke` on the current export, stops after epoch 0
with `--stop-after 1`, resumes from `last.pt`, predicts and scores). Smoke numbers are meaningless; the run only
proves the code path.

## Recipes (official settings; LR shown at the official batch and at our effective batch 64)

| | TSN | TSM | VideoMAE | X-CLIP | Endo-FM | MViTv2-S (reserve) |
|---|---|---|---|---|---|---|
| official source | TSM README: train from ImageNet on SSv2, no `--shift` | TSM README: fine-tuning example (`--tune_from`) | `scripts/ssv2/…_epoch_2400/finetune.sh` + `run_class_finetuning.py` | `configs/k400/16_16.yaml` + `utils/config.py`, `optimizer.py`, `datasets/build.py` | `scripts/eval_finetune_polypdiag.sh` + `eval_finetune.py` | PySlowFast `configs/SSv2/MVITv2_S_16x4.yaml` (fine-tunes K400) |
| backbone | ResNet-50, avg consensus | ResNet-50 + shift 1/8 blockres | ViT-B/16, tubelet 2 | CLIP ViT-B/16 + CCT, MIT, prompts | ViT-B/16 TimeSformer (div. ST) | MViTv2-S |
| init, pretraining data | ImageNet-1k (torchvision V1; no SSv2 TSN checkpoint exists) | TSM SSv2 16-segment checkpoint (ImageNet, then SSv2) | `videomae-base-finetuned-ssv2` (SSv2 self-sup. 2400 ep, then SSv2 sup.) | `xclip-base-patch16-16-frames` (CLIP WIT-400M, then K400) | Endo-FM teacher (33K endoscopy clips, self-sup.; from SVT K400 self-sup.) | torchvision `KINETICS400_V1` (K400 sup.) |
| frames (official) | 16 (8) | 16 (16 in the checkpoint, 8 in the example) | 16 (16) | 16 (16) | 16 (8) | 16 (16) |
| sampling | the 16 protocol frames: uniform over the action window, end frames included | same | same | same | same | same |
| input: train pre-crop -> crop; test | 256 -> 224; 224 | 256 -> 224; 224 | 256 -> 224; 224 | 256 -> 224; 224 | 320 -> 224; 224 | 320 -> 224; 224 |
| fine-tuned / frozen | full | full; partial BN (all BN but the first frozen) | full | video side; text side frozen (FIX_TEXT) | full | full |
| epochs | 50 | 25 | 30 | 30 | 20 | 100 |
| batch: official -> ours (micro x accum) | 64 -> 64 (8 x 8) | 64 -> 64 (8 x 8) | 384 videos x 2 views -> 64 x 2 (4 x 16) | 256 -> 64 (8 x 8) | 4 -> 64 (4 x 16) | 16 per shard -> 64 (8 x 8) |
| LR: official -> ours | 0.01 -> 0.01 | 0.001 -> 0.001 | 5e-4 per 256 -> 1.25e-4 | 8e-6 per 256 -> 2e-6 (x10 prompts/MIT/message) | 1e-3 per 256 -> 2.5e-4 | 2.5e-3 per 16 -> 0.01 |
| optimizer | SGD 0.9, WD 5e-4 (no WD on BN/bias), fc LR x5 / bias x10, clip 20 | same | AdamW 0.9/0.999, WD 0.05, layer decay 0.75 | AdamW 0.9/0.98, eps 1e-8, WD 0.001 | SGD 0.9, WD 0 | SGD 0.9 Nesterov, WD 1e-4 (not 1-D/bias) |
| schedule | x0.1 at epochs 20, 40 | x0.1 at epochs 10, 20 | 5 warmup epochs from 1e-6/256, cosine to 1e-6/256 (per step) | 5 warmup epochs from 0, cosine to LR/100 (per step) | cosine to 0, stepped per epoch | 3 warmup epochs from 1e-4/16, cosine to 1e-6/16 (per step) |
| regularization | dropout 0.5 | dropout 0.8 | drop path 0.1, head init x0.001 | drop path 0 | drop path 0.1 | drop path 0.4, dropout 0.5 |
| augmentation | multi-scale crop (1, .875, .75, .66; 13 fixed offsets) + flip | same | RandAugment m7-n4-mstd0.5-inc1, RRC 0.08-1, flip, random erasing 0.25, 2 views per clip | multi-scale crop (5 fixed offsets), flip, color jitter p 0.8 (per frame), gray p 0.2 | short side 256-320, random crop, flip | RandAugment m7-n4-mstd0.5-inc1, short side 256-320 (inverse uniform), crop, flip, random erasing 0.25 |
| loss | CE | CE | soft CE, Mixup 0.8 / CutMix 1.0, smoothing 0.1 | soft CE, Mixup 0.8 / CutMix 1.0 (X-CLIP blending), smoothing 0.1 | CE | soft CE, Mixup 0.8 / CutMix 1.0, smoothing 0.1 |
| normalization | ImageNet | ImageNet | ImageNet | ImageNet | mean 0.45, std 0.225 | mean 0.45, std 0.225 |

Paper column "Backbone, pretraining" and "Input" (Tab. 3 a): TSN = ResNet-50, IN-1k (not SSv2); TSM = ResNet-50, SSv2;
VideoMAE = ViT-B, SSv2; X-CLIP = CLIP ViT-B/16, K400; Endo-FM = ViT-B, endoscopy videos; MViTv2-S = MViTv2-S, K400.
Every row is 16 x 224^2.

## GPU hours on one RTX 4090 (estimates)

Estimated from FLOPs, not measured: training = 3 x forward FLOPs per
view, 4090 bf16 dense peak 165 TFLOPS at about 35 % utilization (20 % for MViTv2's pooling attention), validation every
epoch, +15 % for loading and checkpoints. `bench_cls <model>` measures the real throughput on your GPU (about
2 minutes each).

| model | GFLOPs / view (fwd) | views x clips x epochs | est. GPU-h | micro-batch on 24 GB | CPU workers needed |
|---|---|---|---|---|---|
| TSN | 66 | 1 x 12.4k x 50 | 0.7 | 8 | ~4 (8 ms / clip) |
| TSM | 66 | 1 x 12.4k x 25 | 0.35 | 8 | ~4 |
| VideoMAE | 180 | 2 x 12.4k x 30 | 2.3 | 4 videos = 8 views | ~8 (110 ms / clip) |
| X-CLIP | 287 | 1 x 12.4k x 30 | 1.8 | 8 | ~4 (45 ms / clip) |
| Endo-FM | ~400 | 1 x 12.4k x 20 | 1.7 | 4 | ~2 (22 ms / clip) |
| **five rows** | | | **~7** | | |
| MViTv2-S (reserve) | 64 | 1 x 12.4k x 100 | 2.4 | 8 | ~12 (69 ms / clip) |

The official augmentations run on the CPU (`--workers`, default min(16, CPUs)); MViTv2's RandAugment at 320 px needs
about 12 workers to keep a 4090 busy. Disk: the cache needs about 108 GB for Gold (`prep_cls`); put it on a fast
local disk (set `BL_CLS_CACHE`) if `$BL_OUT` is on network storage. `CLS_NO_PACK=1` skips the cache; frames are then
resized on the fly.

## Files

| file | role |
|---|---|
| `train.py` | the harness: data, official optimizer / schedule / loss per recipe, accumulation, val per epoch, checkpoints, predictions, info.json |
| `models.py` | `RECIPES` (official settings, sources and adaptations), model wrappers and parameter groups |
| `augment.py` | each method's official train augmentation and batch mixing, re-implemented on the protocol frames |
| `data.py` | `ClipSet` (cache or on-the-fly squash, per-sample seeding), `EpochSampler`, cache helpers |
| `pack.py` | frame cache: train clips at 256 and 320, val/test clips at 224 (uint8 `.npy` under `$BL_OUT/cls_cache`) |
| `weights.py` | pinned downloads into `$BL_WEIGHTS/cls` with sha256 manifest; Endo-FM teacher extraction |
| `vendor/` | TSM `temporal_shift.py`, Endo-FM TimeSformer (see `THIRD_PARTY.md`) |
| `run.sh` | `prep_cls`, `train_<model>`, `bench_cls`, `smoke_cls` |
| `tests/test_cls.py` | CPU checks |
