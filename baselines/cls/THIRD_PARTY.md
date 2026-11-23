# Third-party code and weights in `baselines/cls`

## Vendored code (copied into `vendor/`, license file next to it)

| file | source | commit | license | changes |
|---|---|---|---|---|
| `vendor/tsm/temporal_shift.py` | [mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module) `ops/temporal_shift.py` | `1f6a834b02120441d99fa706b7a13ccc5293c6fa` | MIT (`vendor/tsm/LICENSE`) | none (sha256 `90dfd7f9…c718ea`) |
| `vendor/endofm/timesformer.py` | [med-air/Endo-FM](https://github.com/med-air/Endo-FM) `models/timesformer.py` (itself from TimeSformer / timm) | `206427ebfb77a937ef0cd60370331bcedd74e5a2` | Apache-2.0 (`vendor/endofm/LICENSE`) | the 3 import lines only: `vit_utils` made package-relative, `models.helpers.load_pretrained` replaced by a stub (we load the checkpoint ourselves); original sha256 `3600f338…b627` |
| `vendor/endofm/vit_utils.py` | same repo, `models/vit_utils.py` | same | Apache-2.0 | none (sha256 `9c96edc0…9855`) |

Behaviour added outside the vendored files (in `models.py`, no edit of the copies):
- Endo-FM `Attention.forward` is rebound to `F.scaled_dot_product_attention` (same math: softmax(qk^T * scale) v, dropout 0);
  optional activation checkpointing wraps each block's `forward`.
- TSN/TSM wrapper re-implements the lines of the TSM repo's `ops/models.py` `TSN` class that matter here (ResNet-50 from
  torchvision, fc -> dropout + new fc with N(0, 0.001) init, average consensus, partial BN in `train()`,
  `get_optim_policies` LR/WD multipliers) and its `main.py --tune_from` key handling (strip `module.`, drop `fc` keys).
- VideoMAE: `VideoMAELayer.forward` is rebound to add stochastic depth (VideoMAE's `Block`; the HF port has none).

## Re-implemented, not copied (`augment.py`, LR schedules in `train.py`)

Each method's official training augmentation, batch mixing and LR schedule are written anew from these files (read at
the commits below), so no code of a non-commercial repository is in this directory:

| method | files followed | license of the source |
|---|---|---|
| TSN, TSM | TSM `ops/transforms.py` (GroupMultiScaleCrop, GroupRandomHorizontalFlip), `ops/models.py` (`get_augmentation`, `get_optim_policies`, partial BN), `main.py` (`adjust_learning_rate`, `--tune_from`, clip-grad), `opts.py`, README training commands @ `1f6a834b` | MIT |
| VideoMAE | [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) @ `14ef8d85`: `kinetics.py` (`_aug_frame`, `spatial_sampling`), `video_transforms.py` (`_get_param_spatial_crop`, `random_resized_crop`), `random_erasing.py` (cube), `mixup.py`, `utils.cosine_scheduler`, `optim_factory.py` (layer decay), `run_class_finetuning.py`, `scripts/ssv2/…_epoch_2400/finetune.sh` | CC BY-NC 4.0 (logic only) |
| RandAugment | `timm.data.auto_augment.rand_augment_transform('rand-m7-n4-mstd0.5-inc1')` ops (timm 1.0.24, Apache-2.0); the per-clip application (one draw of ops, coin, magnitude and sign for all frames) follows the video versions in VideoMAE `rand_augment.py` and PySlowFast `slowfast/datasets/rand_augment.py`. The op list is identical in all three. | Apache-2.0 |
| X-CLIP | [microsoft/VideoX](https://github.com/microsoft/VideoX) @ `1744b215`: `X-CLIP/datasets/build.py` (train/val pipelines), `datasets/pipeline.py` (MultiScaleCrop, ColorJitter, GrayScale), `datasets/blending.py` (CutmixMixupBlending), `utils/config.py`, `utils/optimizer.py` (`fix_text`, 10x groups, timm CosineLRScheduler), `main.py`, `configs/k400/16_16.yaml` | MIT |
| Endo-FM | Endo-FM @ `206427eb`: `eval_finetune.py`, `scripts/eval_finetune_polypdiag.sh`, `datasets/ucf101.py`, `datasets/data_utils.py` (`spatial_sampling`), `datasets/transform.py`, `utils/defaults.py`, `models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml` | Apache-2.0 |
| MViTv2-S | [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) @ `287ec007`: `configs/SSv2/MVITv2_S_16x4.yaml` (official fine-tuning of the K400 model), `slowfast/datasets/utils.py` (`aug_frame`), `transform.py`, `random_erasing.py`, `mixup.py`, `slowfast/utils/lr_policy.py`, `slowfast/models/optimizer.py`, `slowfast/config/defaults.py` | Apache-2.0 |

timm 1.0's own `Mixup` cuts `x[:, :, y, x]`, which is wrong for 5-D video; `augment.TimmMixup` therefore re-implements
the timm logic with `x[..., y, x]` as VideoMAE's and PySlowFast's copies do.

## Libraries used as published (not vendored)

| model | code | weights (pinned) | license |
|---|---|---|---|
| VideoMAE | `transformers==5.1.0` `VideoMAEForVideoClassification` | [MCG-NJU/videomae-base-finetuned-ssv2](https://huggingface.co/MCG-NJU/videomae-base-finetuned-ssv2) @ `e3f23b7a9bcc6f6b0bbf44da097c0f94548c6ec5` | weights CC BY-NC 4.0 (academic benchmarking only); code Apache-2.0 |
| X-CLIP | `transformers==5.1.0` `XCLIPModel` + its CLIP tokenizer | [microsoft/xclip-base-patch16-16-frames](https://huggingface.co/microsoft/xclip-base-patch16-16-frames) @ `c4bb0025a08ee79a5e74ef77e511c6f83f328d77` | MIT |
| TSN | torchvision `resnet50` | `resnet50-0676ba61.pth` (IMAGENET1K_V1) from download.pytorch.org | BSD-3-Clause |
| TSM | torchvision `resnet50` + vendored shift | `TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth` from hanlab18.mit.edu (link in the TSM README) | MIT |
| Endo-FM | vendored model | `endo_fm.pth`, Google Drive id `1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl` (the "pre-trained Endo-FM" link of the github.com/med-air/Endo-FM README; the EndoFM-LV README links only its own SharePoint weights) | Apache-2.0 |
| MViTv2-S | torchvision `mvit_v2_s` | `mvit_v2_s-ae3be167.pth` (MViT_V2_S_Weights.KINETICS400_V1) | BSD-3-Clause |
| RandAugment ops | `timm==1.0.24` | – | Apache-2.0 |

`weights.py` downloads all of them into `$BL_WEIGHTS/cls/<model>/` and records the sha256 of each file in
`$BL_WEIGHTS/cls/MANIFEST.json` (expected: TSN `0676ba61…fb8a`, TSM `4ecf097c…dc1e`, VideoMAE `b960dafd…82fb`,
X-CLIP `d6b66745…d91c`, Endo-FM `6fc7a64a…ea70`, MViTv2-S `ae3be167…b975`). The Endo-FM file also holds the student,
motion branches, optimizer and the run's `argparse.Namespace`; `weights.endofm_backbone()` extracts the teacher backbone
once with `torch.load(weights_only=True)` (allow-listing `argparse.Namespace`, `codecs.encode`, numpy scalar/dtypes), and
training loads only that plain state dict. The file is the checkpoint of pretraining epoch 31 (`ck["epoch"]`).
