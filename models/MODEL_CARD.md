---
license: other
license_name: mixed
tags:
  - surgical-video
  - video-understanding
  - keypoint-detection
  - temporal-localization
datasets:
  - hding2455/surgical-action-affordance
---

# TACTUS checkpoints

Trained weights for **TACTUS** (Tool, Action, Contact and Trajectory Understanding in Surgery): the reference network TactusNet, its heads retrained on other frozen encoders, and the baselines of the benchmark.

**[Code](https://github.com/Ruifreddie/TACTUS)** · **[Dataset](https://huggingface.co/datasets/hding2455/surgical-action-affordance)**

Every file is a PyTorch checkpoint written by the training code in the repository, unchanged apart from its file name. `SHA256SUMS` lists the checksum of each file.

## TactusNet

Heads trained on a frozen V-JEPA 2.1 ViT-L encoder (16 frames at 384 × 384). The encoder is not included: download `vjepa2_1_vitl_dist_vitG_384.pt` from the official [V-JEPA 2 release](https://github.com/facebookresearch/vjepa2).

| Directory | Training data |
| --- | --- |
| `tactusnet_gold/` | Gold training split. This is the model reported as TactusNet. |
| `tactusnet_gold_silver/` | Gold training split plus Silver. |

| File | Head | Predicts | Recipe in the repository |
| --- | --- | --- | --- |
| `semantic_head.pt` | Semantic | Tool and action, as one of the six valid pairs | `training/repro/classification/train_stage2.sh` |
| `temporal_head.pt` | Temporal | Interaction window inside the action window | `training/repro/action_range/train_best.sh` |
| `spatial_stage1.pt` | Spatial, stage 1 | Three instrument landmarks on the full frame | `training/unitap/experiments/runs/E007k/run.sh` |
| `spatial_stage2_zoom.pt` | Spatial, stage 2 | The same landmarks on the zoomed crop | `training/unitap/experiments/runs/E009/run.sh` |
| `contact_decoder_hindsight.pt` | Spatial, contact | Four contact-region corners, trained on tip-derived (hindsight) targets | `training/unitap/experiments/runs/C006/run.sh` |
| `pixel_refiner.pt` | Spatial, refiner | Refined landmarks at native resolution | `training/unitap/tools/pixel_refiner.py` |

The checkpoints of the Semantic, Temporal and Spatial heads hold the model weights together with the training state saved by the trainer, which is why the Spatial files are about 1.3 GB each.

To use them, place the files where the pipeline expects its checkpoints (see `training/README.md` in the repository for the path each step writes) and run the evaluation step, or load a file with `torch.load(path, map_location="cpu", weights_only=False)` and read its keys.

## Other frozen encoders

`backbones/<encoder>/` holds TactusNet's Semantic Head, Temporal Head and stage 1 of the Spatial Head retrained on the tokens of another frozen encoder, with the same recipes. See `baselines/ablate/README.md` in the repository.

| Directory | Encoder | Input |
| --- | --- | --- |
| `backbones/vjepa2_l256/` | V-JEPA 2 ViT-L | 16 × 256² |
| `backbones/videomae_l224/` | VideoMAE ViT-L, self-supervised on Kinetics-400 | 16 × 224² |
| `backbones/endofm_b224/` | Endo-FM ViT-B | 16 × 224² |

## Baselines

Published methods trained on the Gold training split with their official configurations, adapted as described in `baselines/` of the repository.

| File | Method | Task |
| --- | --- | --- |
| `baselines/classification/tsn.pt` | TSN | Which, What |
| `baselines/classification/tsm.pt` | TSM | Which, What |
| `baselines/classification/videomae.pt` | VideoMAE | Which, What |
| `baselines/classification/xclip.pt` | X-CLIP | Which, What |
| `baselines/classification/endofm.pt` | Endo-FM | Which, What |
| `baselines/temporal/actionformer.pt` | ActionFormer | When |
| `baselines/temporal/tridet.pt` | TriDet | When |
| `baselines/temporal/qddetr.pt` | QD-DETR | When |
| `baselines/temporal/mstcn.pt` | MS-TCN | When |
| `baselines/keypoints/hrnet_dark.pt` | HRNet-W32 + DARK | How, Where |
| `baselines/keypoints/vitpose.pt` | ViTPose-B | How, Where |
| `baselines/keypoints/dsta.pt` | DSTA | How, Where |
| `baselines/contact/segformer.pt` | SegFormer-B2 | Where |

## License

The TactusNet heads (`tactusnet_gold/`, `tactusnet_gold_silver/`, `backbones/`) are released under the MIT License, like the code.

The baseline checkpoints were initialized from third-party pretrained weights and inherit the terms of those weights, some of which do not allow commercial use. `THIRD_PARTY.md` in each baseline directory of the repository lists the source and license of every initialization. Check them before any use beyond research.

All weights were trained on videos from Cholec80, HeiChole, CRCD, SurgVU and a YouTube collection. They are intended for research and are not a medical device.
