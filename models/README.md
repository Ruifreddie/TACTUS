# TactusNet

TactusNet is the reference network for the five TACTUS tasks. In the default configuration it samples 16 frames from an action window, encodes them at 384 × 384 resolution with a frozen V-JEPA 2.1 ViT-L encoder, and shares the video features across three prediction heads.

| Head | Prediction |
| --- | --- |
| Semantic | The instrument and action, decoded as one of the six valid tool–action pairs. |
| Temporal | The start and end of the interaction window within the action window. |
| Spatial | Three instrument landmarks and four contact-region corners. The head first localizes the instrument in the full frame, then makes predictions on a crop. A refiner adjusts the tool landmarks at native resolution. |

For contact-region training, TactusNet derives each sampled frame's targets from instrument-tip positions in the interaction window, using visible tips at or after that frame when available. The benchmark scores the predicted region against the human-defined tissue region on Gold test clips using direct predictions on the sampled frames, before track propagation.

The model implementation will be added here. The [evaluation protocol](../evaluation/README.md) describes the inputs and metrics used in the paper.
