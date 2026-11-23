# Third-party code and weights of `baselines/where/`

Nothing is vendored: every external piece is either a pip package, a downloaded checkpoint, or project code used in place.

| What | Source | Version | License | How it is used |
|---|---|---|---|---|
| MiT-B2 encoder weights, ImageNet-1k pretrained (backbone only, no decoder) | https://huggingface.co/nvidia/mit-b2 (paper: Xie et al., NeurIPS 2021, https://github.com/NVlabs/SegFormer) | `nvidia/mit-b2@3bb39e8739149c3777d0325349b2a6c32c6413db` (2022-08-06), `pytorch_model.bin` 98,978,917 bytes | "other" on the model card: NVIDIA Source Code License for SegFormer (https://github.com/NVlabs/SegFormer/blob/master/LICENSE), non-commercial research and evaluation use | downloaded once into `$BL_WEIGHTS/mit-b2` by `segformer.ensure_weights()`; only `segformer.encoder.*` is loaded (the ImageNet classifier is unused); the all-MLP decoder (768 ch) and the 6-channel classifier start from random init (baselines rule: pretrained backbone only) |
| SegFormer model code | Hugging Face `transformers` (`SegformerForSemanticSegmentation`) | 5.1.0 (pip) | Apache-2.0 | imported, not copied |
| `huggingface_hub` | pip | 1.8.0 | Apache-2.0 | `snapshot_download` of the pinned revision |
| UniTAP (working name of TactusNet's Spatial Head code, this project) | `$U` = `training/unitap` of this repo | this repo | project code | used in place: `experiments/runs/C006/run.sh` (recipe, one line changed by `tactus_human.sh`), `third_party/t2b/kpt_combo/train.py` (trainer), `tools/dump_val_predictions.py`, `tools/eval_contact_stage.py`, `tools/eval_hindsight.py`, `tools/build_hindsight_labels.py`, `tools/build_zoom_dataset.py` |
| Hindsight hotspot definition | UniTAP `tools/build_hindsight_labels.py`; crop transform `transform_points_to_crop` of `third_party/vitpose_dataset_tools/build_phase4p5_delta_aug.py` | same | project code | re-implemented in `hindsight_floor.py` (`raw_tips`, `hotspot_points`, `unitap_box_soft`, `corners_frame`); `--check-exp` shows it reproduces UniTAP's labels exactly |
| Experiment plumbing | `training/pipeline/` (`$X`: `recipe.sh` edits, `make_exp.py`, `persist.py`, `shards.py`, `make_ct_root.py`, `env.sh`) | this repo | project code | `tactus_human.sh` repeats `recipe.sh`'s sed edits verbatim; `run.sh` calls the others unchanged |

Contact-slot order: `canonical_order.py` and `check_order.py` are new code. The export rule they compare against is UniTAP
`tools/export_case32.py` `uid_contact_tracks` (project code, read only).
