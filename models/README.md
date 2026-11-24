# TactusNet

TactusNet is the reference network for the five TACTUS tasks. The default model uniformly samples 16 frames across the action window, crops them to the video's fixed content box, and resizes them to 384 × 384. Three prediction heads share the tokens from a frozen V-JEPA 2.1 ViT-L/16 encoder.

| Head | Prediction |
| --- | --- |
| Semantic (Which, What) | The tool scores condition the action query. The head selects the valid pair with the largest sum of tool and action logits across the six pairs. |
| Temporal (When) | The start and end of the interaction window within the action window, conditioned on the tool and action classes. |
| Spatial (How, Where) | Three tool landmarks and four contact corners on each sampled frame, conditioned on the tool and action classes. Full-frame tool predictions define one zoom box per clip. The encoder processes the cropped frames again for the tool and contact decoders. A refiner adjusts the tool landmarks on native-resolution image crops. |

The main benchmark (Table 3) supplies the ground-truth tool and action to both localization heads. It scores How and Where directly on the 16 sampled frames before tracking. For each sampled frame, the contact decoder learns a region from visible tool-tip positions in the interaction window, using positions at or after that frame when available. The four predicted corners form a convex hull, scored against the human-defined tissue region. On 1,523 Gold test clips, the default Gold-trained model achieves 19.15 px average displacement error for tool landmarks and 0.242 contact Dice. [Evaluation protocol](../evaluation/README.md)

## Spatial prompts for Silver

The early Silver generator used the same four Spatial Head components with weights trained for prompt placement. The two tool decoders and the refiner trained on SurgVU alone. The contact decoder also used 5,082 Gold clips from the other sources. Human annotators supplied the classes and temporal windows. Within the interaction window, each group of points (three tool landmarks or four contact corners) used the sampled frame with its highest mean confidence. CoTracker3 propagated the seven prompts across the entire action window (Supplement B.6 and C.1).

## Code

| Part | Where |
| --- | --- |
| Architecture and inference wiring | [`tactusnet/`](../tactusnet/README.md) |
| Training of every head, from the dataset release to test scores | [`training/`](../training/README.md) |
| Contact decoder with tip-derived (default) or human-annotated targets | [`training/CONTACT_SUPERVISION.md`](../training/CONTACT_SUPERVISION.md) |
| Other frozen encoders under the same heads | [`baselines/ablate/`](../baselines/ablate/README.md) |
| Fine-tuning the encoder | [`baselines/ft/`](../baselines/ft/README.md) |

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
