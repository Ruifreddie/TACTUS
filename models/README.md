# TactusNet

TactusNet is the reference network for the five TACTUS tasks. In the default configuration it samples 16 frames from an action window, encodes them at 384 × 384 resolution with a frozen V-JEPA 2.1 ViT-L encoder, and shares the video features across three prediction heads.

| Head | Prediction |
| --- | --- |
| Semantic | The instrument and action, decoded as one of the six valid tool–action pairs. |
| Temporal | The start and end of the interaction window within the action window. |
| Spatial | Three instrument landmarks and four contact-region corners. The head first localizes the instrument in the full frame, then makes predictions on a crop. A refiner adjusts the tool landmarks at native resolution. |

For contact-region training, TactusNet derives each sampled frame's targets from instrument-tip positions in the interaction window, using visible tips at or after that frame when available. The benchmark scores the predicted region against the human-defined tissue region on Gold test clips using direct predictions on the sampled frames, before track propagation.

## Code

| Part | Where |
| --- | --- |
| Architecture and inference wiring | [`tactusnet/`](../tactusnet/README.md) |
| Training of every head, from the dataset release to test scores | [`training/`](../training/README.md) |
| Contact decoder with tip-derived (default) or human-annotated targets | [`training/CONTACT_SUPERVISION.md`](../training/CONTACT_SUPERVISION.md) |
| Other frozen encoders under the same heads | [`baselines/ablate/`](../baselines/ablate/README.md) |
| Fine-tuning the encoder | [`baselines/ft/`](../baselines/ft/README.md) |

The [evaluation protocol](../evaluation/README.md) describes the inputs and metrics used in the paper.

## Checkpoints

Trained weights are on Hugging Face: **[abccbd/TACTUS-TactusNet](https://huggingface.co/abccbd/TACTUS-TactusNet)**.

| Directory | Contents |
| --- | --- |
| `tactusnet_gold/` | TactusNet trained on the Gold training split: Semantic Head, Temporal Head, both stages of the Spatial Head, the contact decoder (hindsight targets), and the pixel refiner. |
| `tactusnet_gold_silver/` | The same heads trained on Gold plus Silver. |
| `backbones/` | Semantic Head, Temporal Head and stage 1 of the Spatial Head retrained on V-JEPA 2, VideoMAE and Endo-FM tokens. |
| `baselines/` | The thirteen baselines of the benchmark. |

```bash
pip install huggingface_hub
hf download abccbd/TACTUS-TactusNet --include "tactusnet_gold/*" --local-dir checkpoints
```

The frozen V-JEPA 2.1 encoder is not included; get it from the official V-JEPA 2 release. [`MODEL_CARD.md`](MODEL_CARD.md) lists every file, the recipe that produced it, and the license terms of the baseline weights.
