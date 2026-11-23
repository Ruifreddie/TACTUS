# Third-party code and weights in `baselines/kpt/`

## Vendored or ported code

| File here | Source | Version | License | What was changed |
|---|---|---|---|---|
| `third_party/hrnet/pose_hrnet.py` | [leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) `lib/models/pose_hrnet.py` | commit `6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1` (2022-12-13) | MIT (`third_party/hrnet/LICENSE`) | One line: `is '*'` became `== '*'` (a SyntaxWarning in Python 3.8 and later). The weights are loaded by `hrnet.py`, not by `init_weights(pretrained)`. |
| `third_party/dsta/dsta.py` | [zgspose/DSTA](https://github.com/zgspose/DSTA): `posetimation/zoo/DSTA/dsta_std_resnet50.py`, `posetimation/zoo/backbones/Resnet.py`, `posetimation/loss/criterion.py` | commit `cb683c1e21d6bd6977bedb7965d2bfb3911399c4` (2025-04-29) | Apache-2.0 (`third_party/dsta/LICENSE`) | Ported to plain PyTorch 2.x. 36 joints and 5 frames are arguments. The spatial groups are the 4 tool groups and the 6 pair groups. The RLE flow term moved from `forward` to `rle_loss`. The input is `[B,F,3,H,W]`. `init_weights` takes a state dict. The notice of changes is at the top of the file (Apache-2.0 §4(b)). |
| `third_party/dsta/dsta.py`: `RealNVP` | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) `mmpose/models/utils/realnvp.py` | v1.3.2, commit `5408bc76f5b848cf925a0d1857899011d8c5b497` | Apache-2.0 | Prior N(0.5, I), as in DSTA. DSTA imports this class from its own copy of mmpose, which is not in the DSTA repo. |
| `third_party/dsta/dsta.py`: `drop_path` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) `timm/layers/drop.py` | timm 1.0.24 | Apache-2.0 | Inlined. |
| `codecs.py`: `DarkCodec` | [ilovepose/DarkPose](https://github.com/ilovepose/DarkPose): `lib/dataset/JointsDataset.py` (`generate_target`, `adjust_target_weight`), `lib/core/inference.py` (`get_max_preds`, `gaussian_blur`, `taylor`), `lib/core/loss.py` (`JointsMSELoss`) | commit `0185b427f34a08d93ea7e948ccc3e7fdc78d88c5` | Apache-2.0 | Encoding is batched in torch. Decoding is vectorized in numpy, with the same arithmetic. |
| `codecs.py`: `UDPCodec` | [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose), mmpose 0.x: `TopDownGenerateTarget` (UDP, GaussianHeatmap) in `mmpose/datasets/pipelines/top_down_transform.py`; `_get_max_preds` and `post_dark_udp` in `mmpose/core/evaluation/top_down_eval.py` | commit `c050ed29112da7704797cc1a65af0234b525010d` | Apache-2.0 | Encoding is batched in torch. Decoding has the same arithmetic. |
| `vitpose.py`: `mae_to_vitpose`, `layer_id`, the optimizer groups | ViTPose `mmcv_custom/checkpoint.py` (bicubic pos-embed interpolation) and `mmcv_custom/layer_decay_optimizer_constructor.py` | same commit | Apache-2.0 | Rewritten for the Hugging Face parameter names. |

Used as a library, not vendored:
- Hugging Face `transformers` 5.1.0 (Apache-2.0), `VitPoseForPoseEstimation`. It is built from a config without pose weights. Drop path is added by `vitpose.py`.
- torchvision (BSD-3-Clause), the ResNet-50 weights API.

## Pretrained weights (backbones only, all downloaded into `$BL_WEIGHTS/kpt/`)

| Model | File | Source | Hash | Terms |
|---|---|---|---|---|
| HRNet-W32 | `hrnet_w32-36af842e.pth`: HRNet-W32, ImageNet-1k classification. This is the file that `MODEL.PRETRAINED` of the official HRNet and DarkPose COCO configs names. | [OpenMMLab mirror](https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth) of the HRNet model zoo | md5 `36af842eb9294652ab0330de3f5d2503`, sha256 `d6c378d1ce08a1d53da1abff673258b6df16d69883cdc3828df04377f76100d3` | HRNet (MIT) |
| ViTPose-B | `mae_pretrain_vit_base.pth`: MAE ViT-B/16, self-supervised on ImageNet-1k, encoder only. The ViTPose README points to this file. | [official MAE release](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | sha256 `aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d` (343,249,461 bytes) | **CC-BY-NC 4.0** (MAE repository license): non-commercial |
| DSTA | `resnet50-0676ba61.pth`: torchvision `ResNet50_Weights.IMAGENET1K_V1`, ImageNet-1k supervised | download.pytorch.org | torchvision hash check (sha256 prefix `0676ba61`) | torchvision (BSD-3-Clause) |

No COCO pose checkpoint and no COCO detection checkpoint is used. Every head and decoder is trained from scratch on TACTUS.
The keypoint ground truth was not seen by any of these weights. None of them was pretrained on surgical video.
