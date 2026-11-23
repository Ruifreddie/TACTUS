# Keypoint baselines: How (tool points) and Where (contact region)

Rows of Tab. 3, block (c): **HRNet-W32 + DARK**, **ViTPose-B** and **DSTA**. For each of the 16 protocol frames of a test
clip, given the true tool and action, each model predicts:
- the 3 tool keypoints (How);
- the 4 tissue contact points, whose convex hull is the predicted contact region (Where).

Scoring uses the same `baselines/common/score.py` as TactusNet.

## Commands

```bash
source baselines/env.example.sh          # or your edited copy; run_baselines.sh sets the same variables itself
source baselines/kpt/run.sh
prep_kpt                                 # CPU: backbone weights, the 1-fps training frames of Gold train, CPU checks
train_hrnet                              # GPU, then scores pred_test.npz into metrics_test.json
train_vitpose
train_dsta
```

Each step calls a trainer, for example `$BL_PY -m baselines.kpt.hrnet --out $BL_OUT/kpt_hrnet`. The trainers accept these options:
- `--smoke`: a few steps on tiny batches;
- `--epochs`, `--micro-batch`, `--accum`, `--lr`, `--workers` and `--val-every`;
- `--contact-order module:function`, described under Protocol below.

A step that is run again resumes from `last.pt` (saved at every validation epoch). Each run writes these files to its output directory:
- `log.jsonl`;
- `best.pt` (lowest val ADE) and `last.pt`;
- `pred_val.npz` and `pred_test.npz` (`baselines.common.preds`, kind `kpt`, `tool_xy` [N,16,3,2], `ct_xy` [N,16,4,2]);
- `info.json`, which holds the recipe actually used, the official values next to it, and the list of adaptations.

To score a run: `$BL_PY -m baselines.common.score --pred $BL_OUT/kpt_hrnet/pred_test.npz`.

## Protocol (the same for the three models)

**Initialization.** Only public pretrained backbone weights are loaded: ImageNet-1k, or MAE on ImageNet-1k. Every head and
decoder is trained from scratch on TACTUS. No COCO pose or COCO detection checkpoint is used.

**Input.** The full content frame, read only through `ex.frames` (image models) or `ex.raw_frames` (DSTA's
neighbours). `ex.frames` decodes the raw clip when the export has no `video_16_npy`. For the image models, each epoch
shuffles the clips and keeps a clip's frames together, so each clip is read about once per epoch. The frame is
resized to the official input pixel budget turned landscape, which is a plain rescale with no detector, box, padding or
aspect fix (TactusNet also squashes the content crop). Predictions map back to content-crop-normalized coordinates by the
inverse rescale.

**Frames.**
- HRNet and ViTPose train on the about-1-fps protocol frames, `ex.frames_1fps(sid)`, about 4-5 per Gold-train clip, and are
  tested on all 16 protocol frames. Every frame is predicted on its own.
- DSTA uses a key frame `t` plus its 5 consecutive raw frames `sampled_abs[t] + [-2..2]`, clamped at the clip ends and
  warped with the key frame's transform. Each epoch draws 8 of the 16 key frames per clip, and the test uses all 16.

**Conditioning.**
- Every model has 36 outputs: 12 tool outputs (4 tools x 3 points, because a hook's points mean something different from
  a grasper's) and 24 contact outputs (6 tool-action pairs x 4 points).
- The official loss runs with target weight = visibility x true-group mask. Only the true tool's 3 outputs and the true
  pair's 4 outputs get loss, and the test reads the same 7. Points outside the input get weight 0, as in the official
  code.

**Augmentation.** Each recipe's own random scale and rotation, applied to the full frame. There is no flip anywhere
(tip order matters), no half-body crop (there are no body halves) and no flip test.

**Selection.** `best.pt` has the lowest val ADE, measured on all 16 frames of the val clips every `--val-every` epochs
and at the last epoch. The test is predicted once, from `best.pt`.

**Contact-slot order.** The export's slot order is not consistent across clips (`baselines/where/check_order.py`), so the
training targets use `baselines.where.canonical_order:canonical` by default. It gives one permutation per clip: slots
sorted by the angle of their mean position, clockwise from the image's left, with never-visible slots last. The order is
a hook: `--contact-order identity` keeps the export order, and any `module:function` is called as
`fn(ct_xy [16,4,2], ct_vis [16,4], **kw)`. The function receives whichever of `ex, sid, tool_xy, tool_vis, tip_index`
it accepts and returns `(ct_xy, ct_vis)` or a slot permutation, [4] or [16,4]. Evaluation is order-free (Dice of the hull).

## Recipes

Official configuration kept, plus the TACTUS adaptations (a)-(d) listed in the next section. Batches are reached on one GPU by
gradient accumulation, with micro-batch = the official per-GPU batch.

| | HRNet-W32 + DARK | ViTPose-B | DSTA (ResNet-50) |
|---|---|---|---|
| Official config | DarkPose `w32_384x288_adam_lr1e-3.yaml` | ViTPose `ViTPose_base_coco_256x192.py` | DSTA `configs-resnet50.yaml` |
| Backbone, pretraining | HRNet-W32, ImageNet-1k (`hrnet_w32-36af842e.pth`) | ViT-B/16, MAE ImageNet-1k (`mae_pretrain_vit_base.pth`) | ResNet-50, ImageNet-1k (torchvision `IMAGENET1K_V1`) |
| New from scratch | 1x1 head, 36 ch | classic decoder (2 deconv + 1x1), 36 ch | joint decoder, spatial + temporal blocks, heads, flow |
| Train frames | 1 fps (`frames_1fps`) | 1 fps (`frames_1fps`) | 8 of 16 key frames per clip per epoch |
| Test frames | 16 protocol frames | 16 protocol frames | 16 key frames |
| Frames per sample | 1 | 1 | 5 (key +-2 raw) |
| Input crop (W x H) | full frame, 384 x 288 (official 288 x 384) | full frame, 256 x 192 (official 192 x 256) | full frame, 5 x 256 x 192 (official 192 x 256) |
| Output | heatmaps 96 x 72, sigma 3 | heatmaps 64 x 48, sigma 2 (UDP) | (x, y, sigma) regression |
| Loss | JointsMSE x 0.5, target weight | JointsMSE, target weight | RLE (RealNVP), target weight |
| Decoding | DARK, blur kernel 11 | UDP + DARK-UDP, kernel 11 | coordinates |
| Epochs | 210 (official) | 210 (official) | 20 (official) |
| Total batch | 112 = 28 x 4 accum | 512 = 64 x 8 accum | 32 key frames (160 images) |
| LR | 1e-3 | 5e-4 (layer decay 0.75) | 5e-5 |
| Optimizer | Adam, no weight decay | AdamW, wd 0.1, betas 0.9/0.999, grad clip 1.0, drop path 0.3 | AdamW, wd 0.01 |
| Schedule | x0.1 at epochs 170, 200 | 500-iter linear warmup, x0.1 at 170, 200 | x0.1 at epochs 10, 15 |
| Augmentation | scale 0.35, rotation 45 (p 0.6) | scale 0.5, rotation 40 (p 0.6) | scale 0.35, rotation 45 (p 0.6) |
| Val every | 10 epochs | 10 epochs (official) | 1 epoch |

Milestones are fractions of the epochs, so `--epochs` rescales them.

## TACTUS adaptations (supplement table)

Everything not listed here is the official configuration.

| | HRNet-W32 + DARK | ViTPose-B | DSTA |
|---|---|---|---|
| (a) keypoints | 17 -> 36 outputs (12 tool + 24 contact). The loss and readout are on the true tool (3) and the true pair (4). | same | same; the spatial local-awareness groups become the 4 tool groups (3 points) and the 6 pair groups (4 points), each group size with its own positional table, instead of PoseTrack's limbs |
| (b) inputs | full content frame, plain rescale to 384 x 288 (official 288 x 384 turned landscape); 1-fps training frames; all 16 frames at test | same, 256 x 192 (official 192 x 256), patch grid 16 x 12, MAE position embeddings interpolated 14x14 -> 12x16 | same, 256 x 192; key frame +-2 consecutive raw frames (released config: +-1) |
| (c) schedule length | 210 epochs over the 1-fps frames (official count) | 210 epochs (official count) | 20 epochs (official count) of 8 key frames per clip |
| (d) batch | 112 = 28 x 4 accumulation steps (official 4 GPUs x 28) | 512 = 64 x 8 (official 8 GPUs x 64) | 32 (official, 1 GPU) |
| initialization | ImageNet HRNet-W32, as the official config | MAE ViT-B, as the official config | torchvision ImageNet ResNet-50 (released config: COCO RLE ResNet-50) |
| removed | flip, half-body crop, flip test | flip, half-body crop, flip test | flip, half-body crop |
| precision | bf16 autocast (official fp32) | bf16 autocast, SDPA attention | bf16 autocast, RLE loss in fp32 |
| validation | every 10 epochs (DarkPose: every epoch) | every 10 epochs (official) | every epoch |

## GPU hours on one RTX 4090 (estimates)

These are not measured on a 4090. Forward FLOPs per sample come from `torch.utils.flop_counter`, and training is taken
as 3x the forward. The data: about 12.4k Gold-train clips, about 56k 1-fps frames (4-5 per clip) and about 99k DSTA key
frames per epoch.

| | fwd GFLOPs/sample | samples seen | assumed bf16 throughput | GPU-h |
|---|---|---|---|---|
| HRNet-W32 + DARK, 384x288 | 34.4 | 210 x 56k = 11.8M | ~35 TFLOPS (HRNet's many small convs) | **~10** (7-12) |
| ViTPose-B, 256x192 | 35.7 | 210 x 56k = 11.8M | ~70 TFLOPS | **~5** (4-7) |
| DSTA, 5 x 256x192 | 40.1 | 20 x 99k = 2.0M | ~50 TFLOPS; raw-frame decoding likely sets the pace | **~2-4** (CPU-bound, use 12-16 workers) |

The total at the official epoch counts is about 17-19 GPU-h. HRNet is the largest share. `train_hrnet --epochs 105` halves
it, and the step decays follow the same fractions. Validation every 10 epochs (DSTA: every epoch) adds about 5%. Each
`last.pt` is about 0.35 GB (HRNet, DSTA) or 1.1 GB (ViTPose).

## Smoke runs

`--smoke` runs 2 epochs of a few tiny micro-batches, with accumulation, validation, checkpoints, resume, prediction
files and `info.json`. Score the result with `baselines.common.score --no-boot`. Smoke numbers only prove the code
path.

## Files

- `core.py`: output channels, contact-order hook, full-frame transforms, the `frames_1fps` cache.
- `codecs.py`: DARK (DarkPose) and UDP (ViTPose) heatmap codecs, and JointsMSELoss.
- `datasets.py`: full-frame image samples and the 5-frame clip samples of DSTA.
- `engine.py`: the shared training and evaluation loop (accumulation, schedules, checkpoints, predictions, info.json).
- `hrnet.py`, `vitpose.py` and `dsta.py`: the three trainers.
- `weights.py`: backbone downloads with pinned hashes.
- `third_party/`: the vendored HRNet code and the ported DSTA code (see `THIRD_PARTY.md`).
- `tests/test_kpt.py`: CPU checks of the transforms, codecs, hook and weight loading.
- `run.sh`: `prep_kpt`, `train_hrnet`, `train_vitpose` and `train_dsta`.

No tool detector and no detect-then-crop stage is used: all three models run on the full frame.
