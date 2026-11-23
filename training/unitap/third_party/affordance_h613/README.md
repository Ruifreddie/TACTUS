# Affordance Region Model + V-JEPA2.1

Real H613 region-model code, using the H342 proposal/point-temporal architecture.
This package contains the complete original affordance training code, all 504
recorded H613 training arguments, loss/optimization documentation, portable
training and evaluation launchers, the V-JEPA adapter, and inference examples.
Start with [TRAINING.md](TRAINING.md) for the complete protocol and commands,
and [PARAMETERS.md](PARAMETERS.md) for every recorded parameter. No weights, videos,
annotations, sample IDs, dataset splits, caches, or raw experiment results are bundled.

## Training Package Entry Points

- `train.py`: full H613 training and evaluation from recorded configuration.
- `configs/h613_train.json`: all saved arguments with portable paths.
- `affordance/training_core.py`: complete original region trainer and losses.
- `cache_features.py`: V-JEPA2.1 feature preparation without deleting frames.
- `TRAINING.md`: data contract, supervision, optimizer, regularization, losses,
  EMA, checkpoint selection, evaluation and reproducibility limits.
- `PARAMETERS.md`: complete parameter reference, including inactive options.
- `smoke_train.py`: synthetic backward/save/reload/evaluation check.

## Architecture

16 RGB frames -> frozen V-JEPA2.1 ViT-L/384 -> dense final-layer tokens
-> conditioned region proposals -> soft selection -> four-point temporal
refinement -> 16 predicted contact/affordance quadrilaterals.

The backbone is pretrained and frozen. "From scratch" in H613 refers to the
region head, not the V-JEPA backbone. The approximately 20 MB H613 checkpoint
contains the region head; it is not an end-to-end video encoder checkpoint.

The head consumes [B,8,24,24,1024] tokens and integer tool/action/combo IDs.
Its output is [B,16,4,2], normalized (x,y) coordinates in [0,1]. Multiply x by
image width and y by image height for image coordinates. A raster mask may be
derived from these four corners; this is not a separate dense mask decoder.
The 8 temporal tokens come from tubelets of 2 frames. The head produces 16
output frames internally; do not duplicate tokens to 16 before calling it.

## Setup

Use Python 3.10+ and install requirements.txt in your chosen environment.
Cached-feature inference requires only this directory and an explicit H613
checkpoint. Run commands from this directory.

```bash
python -m pip install -r requirements.txt
python infer_tokens.py --help
```

## Cached Tokens

```python
import numpy as np
import torch
from affordance import build_model, predict_regions

model = build_model('weights/h613_best.pt', device='cpu')
tokens = torch.from_numpy(np.load('tokens.npy', allow_pickle=False))[None]
# Use the condition IDs from the checkpoint's training label mapping.
quad = predict_regions(model, tokens, [tool_id], [action_id], [combo_id])
```

The six condition combinations used by the H613 data interface are:

| Tool | Action | tool_id | action_id | combo_id |
|---|---|---:|---:|---:|
| hook | dissect | 0 | 0 | 0 |
| grasper | grasp | 1 | 1 | 1 |
| clipper | clip | 3 | 3 | 2 |
| scissors | cut | 2 | 2 | 3 |
| scissors | dissect | 2 | 0 | 4 |
| grasper | dissect | 1 | 0 | 5 |

IDs must preserve the original mapping; do not renumber classes or infer IDs
from alphabetical order. Embedding capacities are 9 tools, 9 actions, and 17
combinations, not a claim that every index was trained on the final split.
The model does not classify these labels. They are supplied by the caller.

## From Video Frames

Supply a compatible local V-JEPA2.1 source checkout and pretrained encoder.
The adapter uses its `src.hub.backbones.vjepa2_1_vit_large_384` factory. Upstream
source and pretrained weights are not bundled. `pretrained_from_hub=True`
may download the encoder through the upstream factory.

The local adapter source was used with V-JEPA commit
`204698b45b3712590f06245fbfba32d3be539812` plus the included
`vjepa_official_weights.patch`. That patch replaces a localhost test download
endpoint with Meta's official weight URL. For that exact checkout, run
`git -C vjepa2 apply ../vjepa_official_weights.patch` from this package directory.
If your checkout already uses the official URL, no patch is needed. Install
the dependencies required by your V-JEPA checkout separately.

```python
from affordance.vjepa_wrapper import VJEPAWrapper
from affordance import encode_video, build_model, predict_regions

encoder = VJEPAWrapper(
    vjepa_repo='vjepa2', arch='vjepa2_1_vit_large_384',
    img_size=384, num_frames=16, patch_size=16, tubelet_size=2,
    hf_repo='', pretrained_from_hub=True, freeze=True,
).to('cuda').eval()
model = build_model('weights/h613_best.pt', device='cuda')
# rgb_frames: torch.uint8 [B,16,H,W,3], sampled in temporal order.
tokens = encode_video(encoder, rgb_frames)
quad = predict_regions(model, tokens, tool_ids, action_ids, combo_ids)
```

Preprocessing matches the local cache converter: RGB, divide by 255, bilinear
resize to 384x384 with align_corners=False, ImageNet mean/std normalization.
For OpenCV BGR frames, reverse channels first. `encode_video` takes uint8 RGB,
so do not pre-normalize its input. Preserve the original clip sampling policy
when comparing to historical runs. Use final-layer dense tokens, not pooled
embeddings, predictor outputs, or a V-JEPA2 256px variant.

The strict H613 inference input has no keypoint trajectory, tool mask, RGB side branch,
GT quad, or action-range features. Training does use GT quad supervision and
action-range loss weighting; TRAINING.md distinguishes these from model inputs.
The API fills disabled auxiliary arguments
with zeros. It uses soft selection, temperature 0.75 and no evaluation source
bias. Loading is strict; the stored `model` state is selected as in the original
evaluation code, with no separate EMA substitution.

## Source Scope

affordance/model.py is an unchanged AST-selected dependency closure from the
original experimental model source, not the entire training script. Existing
optional region-model branches remain to preserve checkpoint compatibility;
h613.json locks the H613 inference settings. The complete original trainer is
also included as training_core.py so losses and optimization are reproducible.
It retains historical optional branches, disabled by configs/h613_train.json.
There are no other task decoders or actual evaluation datasets.
PROVENANCE.json records relative source names and SHA-256 hashes.

## Verification

```bash
python verify.py
python verify.py --checkpoint weights/h613_best.pt
```

At packaging time the head strictly loaded the local H613 checkpoint and gave
bit-identical outputs to the original model on synthetic tokens (CPU, PyTorch
2.10.0). There are 5,177,949 parameters, including the training mask-token
parameter retained for checkpoint compatibility. No real samples were used.
The pretrained video encoder was not run end-to-end during this verification;
it requires a separately supplied compatible V-JEPA checkout and its weights.

This package supports training, evaluation and inference with user-supplied data.
Exact historical replay additionally requires the original prepared data/splits.
Building without a checkpoint produces random head weights, suitable only
for shape tests. No redistribution license is newly granted here; upstream
V-JEPA code and weights retain their own terms.
