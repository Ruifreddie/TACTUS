# Third-party code and weights of `baselines/when/`

All vendored code lives in `third_party/<name>/`, with the upstream `LICENSE` next to it. Files listed as verbatim are
byte-identical to the upstream file at the given commit; modified files start with a header that lists every change, and
each change is marked `[TACTUS]` in the code. Nothing is compiled: the CPU NMS C++ extension of ActionFormer/TriDet is
not vendored (top-1 decoding needs no NMS, see README).

| Code | Source | Commit (date) | License |
|---|---|---|---|
| ActionFormer | https://github.com/happyharrycn/actionformer_release | `61ea7eb9308a568b0cf45e3804830836e30061de` (2024-04-11) | MIT, (c) 2021 University of Wisconsin-Madison |
| TriDet | https://github.com/dingfengshi/TriDet | `764b8165b0d7eb3dfce29f126493564e15003d6e` (2023-12-27) | MIT, (c) 2023 Dingfeng Shi |
| QD-DETR | https://github.com/wjun0830/QD-DETR | `f8628f79f7c651b586300b142dbe9b85e43857cc` (2025-08-12) | MIT, (c) 2022 the QD-DETR authors, including Moment-DETR (MIT, (c) 2021 Jie Lei); `attention.py` carries DAB-DETR / Conditional DETR / DETR headers (Apache-2.0) |
| MS-TCN | https://github.com/yabufarha/ms-tcn | `33ed91c0c7576650a2367efc602553af4c5295b1` (2021-08-16) | MIT + Commons Clause (no selling; academic use is fine) |

## Files

**ActionFormer** (`third_party/actionformer/`)
- verbatim: `libs/modeling/{models,blocks,weight_init,backbones,necks,loc_generators}.py`, `libs/utils/lr_schedulers.py`
- `meta_archs.py` (from `libs/modeling/meta_archs.py`): heads, `__init__`, `label_points*` and the loss unchanged;
  `forward()` takes batched tensors instead of a list of dicts; `losses(cls_channel=...)` restricts the focal loss to the
  true pair's channel; `decode_top1()` replaces `inference()`/`postprocessing()` (top-1 of the true channel with the same
  score/duration filters, no NMS).
- `losses.py`: the `@torch.jit.script` decorators removed (same math, eager PyTorch).
- `train_utils.py`: `make_optimizer()` and `ModelEma` from `libs/utils/train_utils.py`, unchanged except the import path.
- `data_utils.py`: `truncate_feats()` from `libs/datasets/data_utils.py`, unchanged (the official training crop).
- not vendored: `libs/utils/csrc/nms_cpu.cpp`, `libs/utils/nms.py`, `postprocessing.py`, `metrics.py`, datasets, `train.py`/`eval.py`.

**TriDet** (`third_party/tridet/`)
- verbatim: `libs/modeling/{models,blocks,weight_init,backbones,necks,loc_generators}.py`
- `meta_archs.py`, `losses.py`, `train_utils.py`, `data_utils.py`: the same kind of changes as for ActionFormer (TriDet's
  `truncate_feats` has no `offset` argument). The Trident boundary distributions are read from the true pair's channel.
- the LR schedule: TriDet's `libs/utils/lr_schedulers.py` is identical to ActionFormer's, so it is imported from there.

**QD-DETR** (`third_party/qd_detr/`)
- verbatim: `qd_detr/{attention,position_encoding,span_utils,misc}.py`
- `model.py`: relative imports; `QDDETR.forward()` takes optional explicit negative queries (the trainer passes another
  pair's template; the original rolls the batch).
- `transformer.py`: the decoder's zero target is created on the input's device instead of `.cuda()`.
- `matcher.py`: relative import only.
- not vendored: datasets/loaders, `train.py`, `inference.py`, post-processing, the audio variant, checkpoints.

**MS-TCN** (`third_party/ms_tcn/model.py`): `MultiStageModel`, `SingleStageModel`, `DilatedResidualLayer` from `model.py`;
`MultiStageModel(stage_input="sigmoid")` feeds the refinement stages with sigmoid instead of softmax probabilities
(per-pair sigmoid channels). The Python 2 `Trainer` class is not vendored; its loss is re-implemented in `methods/mstcn.py`.

## Weights and code used at run time (not vendored, never written into the repo)

| What | Where it comes from | License |
|---|---|---|
| V-JEPA 2.1 ViT-L/384 (TactusNet's frozen encoder), `vjepa2_1_vitl_dist_vitG_384.pt` | the file `$VJEPA_CKPT` points to; this code does not download it | Meta V-JEPA 2 release terms |
| vjepa2 source (`src/hub/backbones.py`, `app/vjepa_2_1/...`) | `$U/third_party/t2b/vjepa2` (`$U` = `training/unitap` of this repo), or `$VJEPA_REPO` | MIT (facebookresearch/vjepa2) |
| V-JEPA wrapper | `tactusnet/_vendored_vjepa_wrapper.py` of this repo (= UniTAP `kpt_combo/vjepa_wrapper.py` for the final layer), loaded by file path | project code |
| CLIP ViT-B/32 text encoder | Hugging Face `openai/clip-vit-base-patch32` @ `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, downloaded once into `$BL_WEIGHTS/clip-vit-base-patch32` | MIT (OpenAI CLIP) |
