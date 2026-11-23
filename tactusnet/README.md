# TactusNet architecture package

`tactusnet/` is the architecture and inference wiring of TactusNet: one frozen V-JEPA 2.1 ViT-L encoder (16 frames, 384 × 384) and the heads that read its tokens. Each layer's output conditions the next one.

```
frames -> content crop -> encode once -> Semantic Head   Which / What   one of 6 tool-action pairs
                                      -> Temporal Head   When           interaction interval (conditioned on the pair)
                                      -> Spatial Head    How / Where    3 tool points + 4 contact points per sampled frame
            seed frames -> 7 prompt points -> tracking (optional, off by default)
```

Nothing in this package trains a model. Training lives in [`training/`](../training/README.md). For the model description see [`models/README.md`](../models/README.md).

Naming: the code calls the paper's **action window** the clip (`ClipSpec`, "affordance range") and the paper's **interaction window** the `action_range`. The Spatial Head code keeps its working name, UniTAP.

## Files

| File | Contents |
| --- | --- |
| `labels.py` | The six tool–action pairs. Tool and action ids under the two numberings used in the code base (`kpt_combo`, `h613`), always derived from the pair id. |
| `schema.py` | Data containers passed between layers, the 16-frame sampling rule, `Annotation.to_json()`. |
| `encoder.py` | `TokenEncoder` interface. `VJEPAEncoder` (standalone), `UniTAPTokenEncoder` (reuses the encoder the Spatial Head already loaded), `FakeEncoder` (random tokens for tests). |
| `semantic.py` | `SemanticHead` (`joint` or `dual` mode), pair logits composed from tool and action logits, `semantic_loss`. |
| `temporal.py` | `TemporalLayer`: per-step features, temporal adapter, point-based interval head, decoding to frame indices. |
| `spatial.py` | `UniTAPSpatial` (wraps the Spatial Head code in `training/unitap`), `MockSpatial` (deterministic stand-in for tests). |
| `seeds.py` | Seed-frame selection, assembly of the 7 prompt points, mapping from the 16-step axis to frame indices. |
| `tracking.py` | `JobRef`, `DryRunTracker`, `ServerTracker` (see [Tracking](#tracking)). |
| `pyramid.py` | `TactusNet.annotate()`: runs the layers in order. |
| `_vendored_heads.py` | Head building blocks extracted verbatim from `training/unitap/third_party/t2b/kpt_combo/model.py`. The header records the SHA-256 of the source file. |
| `_vendored_vjepa_wrapper.py` | V-JEPA wrapper, byte-identical to `training/unitap/third_party/affordance_h613/affordance/vjepa_wrapper.py`. |
| `tools/extract_vendored.py` | Regenerates the two `_vendored_*` files. It reads `model.py` from a zip archive of the Spatial Head tree (`--zip`). |
| `patches/unitap_deploy.patch` | Optional patch for `training/unitap/tools` (see [Deploy patch](#deploy-patch)). |
| `tests/test_architecture.py` | Architecture tests. |

Do not edit the `_vendored_*` files by hand. Change the source and regenerate them.

## Usage

`annotate()` takes the 16 sampled frames of one clip as uncropped BGR `uint8` with shape `[16, H, W, 3]`. The frame indices are `ClipSpec(...).sampled_local()`.

Plumbing check on CPU, with random tokens and no weights:

```python
import numpy as np
from tactusnet import TactusNet, SemanticHead, TemporalLayer, MockSpatial, FakeEncoder, ClipSpec

model = TactusNet(FakeEncoder(), SemanticHead(), TemporalLayer(), MockSpatial())
frames = np.zeros((16, 480, 854, 3), np.uint8)
ann = model.annotate(frames, ClipSpec(a0=0, n=344))
print(ann.to_json().keys())
```

With the real encoder and Spatial Head:

```python
from tactusnet import TactusNet, SemanticHead, TemporalLayer, UniTAPSpatial, ClipSpec

spatial = UniTAPSpatial("<unitap_root>", device="cuda", vjepa_ckpt="<vjepa2_1_vitl_dist_vitG_384.pt>")
model = TactusNet(spatial.token_encoder(), SemanticHead(), TemporalLayer(), spatial)   # one V-JEPA in memory
# load trained Semantic and Temporal Head weights here (see Limitations)

ann = model.annotate(frames_bgr16, ClipSpec(a0=14778, n=344))                                   # all layers predict
ann = model.annotate(frames_bgr16, ClipSpec(a0=14778, n=344), tool="hook", action="dissect")    # pair given
ann = model.annotate(frames_bgr16, ClipSpec(a0=14778, n=344), tool="hook", action="dissect",
                     action_range=(42, 273))                                                    # pair and interval given
print(ann.to_json())
```

`a0` is the absolute index of the clip's first frame and `n` its number of frames. `action_range` is given as frame indices inside the clip.

`<unitap_root>` must contain:

- `tools/` and `third_party/` from `training/unitap`;
- `data/case32/v1/labels.json`;
- the Spatial Head checkpoints at `experiments/runs/{E007k,E009,C006}/out/checkpoints/best.pt` and `experiments/runs/E008b/pixref/best.pt`, or pass other paths with `ckpts=dict(stage1=..., stage2=..., contact=..., refiner=...)`.

An experiment directory written by the training pipeline (`$WORK/exp/<E>`) has this layout.

## Tracking

The tracking stage is optional and off by default. `annotate()` only builds the 7 prompt points (3 tool points and 4 contact points on their seed frames) and returns them in `Annotation.prompt`.

Tracks are requested only when all three hold: the model was built with a tracker, `annotate(..., track=True)` is called, and a `JobRef` is passed.

| Tracker | Behaviour |
| --- | --- |
| none (default) | No tracking. `track=True` raises an error. |
| `DryRunTracker` | Returns the request body. Sends nothing. |
| `ServerTracker(session, base_url)` | Submits the prompt to an annotation server that runs a point tracker, then polls for the result. The caller supplies the logged-in session and the URL. No address or credential is stored in this package. |

`ServerTracker` is a client for the annotation tool used to build the dataset. That server is not part of this release. Most users will not need it. To get tracks, pass the prompt points to a point tracker of your choice.

## Tests

```bash
python -I tactusnet/tests/test_architecture.py
```

Run it from the repository root. It needs `numpy` and `torch`, no GPU, no network and no weights. It uses `FakeEncoder` and `MockSpatial`. 33 tests are defined. 31 run in this repository and 2 are skipped (see below).

The tests cover:

- Labels: pair round trip, rejection of illegal pairs, the two id numberings.
- Seeds and prompts: the 16-frame sampling rule, seed selection inside the interval, the fallback for very short intervals, prompt format and clamping.
- Semantic Head: output shapes in both modes, pair logits composed from tool and action logits, parameter names matching the `kpt_combo` trainer, gradients reaching only the head.
- Temporal Head: the interval decoding rule, output shapes of both head modes, dependence on the condition, a valid predicted interval.
- Tracking: request body format, `DryRunTracker` sends nothing, `ServerTracker` submits once and never overwrites a job that already has a newer version (tested against a fake session).
- `TactusNet.annotate()`: the three input modes, tracking is opt-in, invalid inputs raise.

Two tests compare the `_vendored_*` files with their original sources. They look for the original archives in the repository root and are skipped when these are absent, as they are here.

## Limitations

- **Architecture and inference wiring only.** This package defines the modules and how they connect. It contains no training code and no weights.
- **Head weights are loaded separately.** `SemanticHead()` and `TemporalLayer()` start with random weights. The package has no checkpoint loader. `SemanticHead` uses the parameter names of the `kpt_combo` trainer so that a trained checkpoint can be mapped onto it, but loading a checkpoint from `training/` into these modules has not been tested.
- **The temporal feature builder is simplified.** `TemporalFeatureBuilder` omits two modules of the training-time model (a depth-wise convolution refinement and a temporal mean residual). Its equivalence to the trained Temporal Head has not been checked. To reproduce the trained heads, use the trainer's own model code; `training/pipeline/eval_head.py` shows how.
- **Real-model paths are not covered by the tests.** `VJEPAEncoder` and `UniTAPSpatial` are not exercised by the test suite. `UniTAPSpatial` calls the same functions as `training/unitap/tools/annotate_clips.py`.
- **Token grid.** All modules expect V-JEPA 2.1 ViT-L/384 tokens of shape `[B, 8, 24, 24, 1024]`. The launchers in `training/repro` default to a V-JEPA 2 model at 256 px (16 × 16 grid); the training pipeline overrides this to 2.1/384. Heads trained on the 256 px tokens do not fit this package.
- **Confidences are not calibrated.** Tool and contact confidences come from different heads and are not comparable. Seeds are chosen separately for each.

## Deploy patch

Spatial Head checkpoints store the V-JEPA source and weight paths of the machine that trained them. `patches/unitap_deploy.patch` changes two files of `training/unitap/tools`:

- `infer_pipeline.py`: re-points these paths to the checkout (`third_party/t2b/vjepa2`, and `weights/vjepa2_1_vitl_dist_vitG_384.pt` or the file named by the environment variable `VJEPA_CKPT`). It also prefers slim checkpoints in `UniTAP_deploy_ckpt/` when that directory exists.
- `annotate_clips.py`: the suggested frame of the command-line tool is chosen inside the interaction interval.

Apply it when the checkpoints were trained on another machine. The `vjepa_ckpt` argument of `UniTAPSpatial` only takes effect with the patch. Without the patch, the paths stored in the checkpoints are used.

```bash
cd training/unitap && patch -p1 < ../../tactusnet/patches/unitap_deploy.patch
```

`make_exp.py` copies `tools/` into each experiment directory, so apply the patch before creating the experiment, or apply it inside the experiment directory.

## License note

The vendored code comes from this repository. V-JEPA code and weights follow their upstream terms. The weights are not included.
